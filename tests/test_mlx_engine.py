"""Tests for mlx_engine.py: scheduler, prefix cache, HTTP surface, backend spec.

Everything runs on tiny randomly initialised models built in-process (a 2-layer
llama and a 4-layer Qwen3.5-style hybrid with DeltaNet ArraysCache layers), so
nothing is downloaded and each test takes well under a second. The point of
most tests is an equivalence: whatever the engine does (batch membership
changing mid-answer, a restored or trimmed prefix, a hybrid snapshot, a
speculative request folded into the batch), the greedy tokens must equal
mlx-lm's plain generate_step from an empty cache.

LLM_ENGINE_LIVE=1 adds a test on the cached Qwen2.5-Coder-3B-Instruct-4bit
(offline; never downloads).
"""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import local_llm.config as config_mod  # noqa: E402

config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-mlx-engine-"))

from local_llm.backends import get_backend, request_model_for  # noqa: E402
from local_llm.mlx_engine import (  # noqa: E402
    EngineBusy, EngineOptions, MLXEngine, PrefixCache, _eng_cache_copy, _eng_parser,
    engine_plan, start_engine_server,
)
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.sysutil import ADAPTER_DIR  # noqa: E402

NO_EOS = 999   # outside the tiny vocabulary: answers end on max_tokens or a stop string


# --------------------------------------------------------------------------- #
# Fixtures: tiny models and a stub tokenizer
# --------------------------------------------------------------------------- #
class _StubTok:
    """ids 0-3 are specials (3 opens a turn, 2 closes it); ids 4..259 are one char each.

    decode(encode(x)) round-trips any generated id >= 4, so a re-rendered
    assistant turn tokenizes back to exactly what the model generated.
    """

    vocab_size = 260
    has_chat_template = True

    def __init__(self, eos: int = NO_EOS) -> None:
        self.eos_token_ids = {eos}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        out = []
        for ch in text:
            o = ord(ch)
            out.append(o - 0x100 if 0x104 <= o < 0x204 else 4 + (o % 256))
        return out

    def decode(self, ids, **_kw) -> str:
        return "".join(chr(0x100 + int(i)) for i in ids if 4 <= int(i) < 260)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True,
                            tools=None, **_kw):
        ids: list[int] = []
        for m in messages:
            ids += [3] + self.encode(f"{m['role']}:{m.get('content') or ''}") + [2]
        if add_generation_prompt:
            ids += [3] + self.encode("assistant:")
        return ids if tokenize else self.decode(ids)

    @property
    def detokenizer(self):
        from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer
        return NaiveStreamingDetokenizer(self)


_MODELS: dict = {}


def _llama(seed: int = 0, layers: int = 2, hidden: int = 64):
    key = ("llama", seed, layers, hidden)
    if key not in _MODELS:
        import mlx.core as mx
        from mlx.utils import tree_map
        from mlx_lm.models import llama

        mx.random.seed(seed)
        args = llama.ModelArgs(model_type="llama", hidden_size=hidden, num_hidden_layers=layers,
                               intermediate_size=2 * hidden, num_attention_heads=4, rms_norm_eps=1e-5,
                               vocab_size=260, num_key_value_heads=2,
                               max_position_embeddings=4096, tie_word_embeddings=False)
        m = llama.Model(args)
        # Scaled weights: an untouched random init repeats one token forever,
        # which would make every equivalence below trivially true.
        m.update(tree_map(lambda p: p * 4 if p.ndim > 1 else p, m.parameters()))
        mx.eval(m.parameters())
        _MODELS[key] = m
    return _MODELS[key]


def _hybrid():
    if "hybrid" not in _MODELS:
        import mlx.core as mx
        from mlx_lm.models import qwen3_5

        mx.random.seed(0)
        args = qwen3_5.TextModelArgs(
            model_type="qwen3_5_text", hidden_size=64, intermediate_size=128, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, vocab_size=260,
            linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16,
            linear_value_head_dim=16, linear_conv_kernel_dim=4, head_dim=16,
            full_attention_interval=2, tie_word_embeddings=False, max_position_embeddings=4096)
        m = qwen3_5.TextModel(args)
        mx.eval(m.parameters())
        _MODELS["hybrid"] = m
    return _MODELS["hybrid"]


def _ref(model, prompt: list[int], n: int, eos: set | None = None, **kw) -> list[int]:
    """Greedy tokens from an empty cache: the ground truth every path must match."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    out = []
    for tok, _ in generate_step(mx.array(prompt), model, max_tokens=n, **kw):
        out.append(int(tok))
        if eos and int(tok) in eos:
            break
    return out


def _engine(model=None, tok=None, draft=None, **kw) -> MLXEngine:
    kw.setdefault("prefix_cache_bytes", 64 << 20)
    kw.setdefault("kv_budget_bytes", 0)
    kw.setdefault("prefill_step_size", 16)
    kw.setdefault("max_batch", 4)
    opts = EngineOptions(model="tiny/test-model", **kw)
    eng = MLXEngine(opts, model=model if model is not None else _llama(),
                    tokenizer=tok or _StubTok(), draft_model=draft)
    eng.start()
    assert eng.wait_ready(60), eng.dead
    return eng


def _msgs(text: str, system: str = "") -> list[dict]:
    out = [{"role": "system", "content": system}] if system else []
    return out + [{"role": "user", "content": text}]


def _submit(eng: MLXEngine, body: dict):
    req = eng.prepare(body)
    events: list = []
    req.emit = events.append
    eng.submit(req)
    return req, events


def _run(eng: MLXEngine, body: dict, timeout: float = 60):
    req, events = _submit(eng, body)
    assert req.done.wait(timeout), "request did not finish"
    return req, events


def _text(events) -> str:
    return "".join(ev[1] for ev in events if ev[0] == "delta")


def _wait(cond, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.005)
    return False


# --------------------------------------------------------------------------- #
# Prefix cache (radix tree) with fake caches: no model needed
# --------------------------------------------------------------------------- #
class _FakeLayer:
    def __init__(self, n: int, trimmable: bool = True, per_token: int = 10) -> None:
        self.offset = n
        self._trim = trimmable
        self.per_token = per_token

    def is_trimmable(self) -> bool:
        return self._trim

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        return n

    @property
    def nbytes(self) -> int:
        return self.offset * self.per_token


def test_radix_hit_miss_trim_and_dedupe():
    pc = PrefixCache(10_000)
    base = list(range(1, 11))
    assert pc.insert(base, [_FakeLayer(10)], "final")
    cache, n = pc.lookup(base + [11, 12])
    assert n == 10 and cache[0].offset == 10
    # Diverging after 5 tokens: the longer trimmable entry is trimmed on a COPY.
    cache, n = pc.lookup(base[:5] + [99, 98])
    assert n == 5 and cache[0].offset == 5
    assert pc._mem and next(iter(pc._mem.values())).cache[0].offset == 10
    # The exact prompt: one token must stay unprocessed.
    cache, n = pc.lookup(base)
    assert n == 9 and cache[0].offset == 9
    assert pc.lookup([7, 8, 9])[1] == 0
    # A prefix of an existing trimmable entry is redundant; a longer one replaces it.
    assert not pc.insert(base[:6], [_FakeLayer(6)])
    assert pc.insert(base + [11, 12, 13], [_FakeLayer(13)])
    st = pc.stats()
    assert st["entries"] == 1 and st["deduped"] == 1 and st["redundant"] == 1
    assert st["hits"] == 3 and st["misses"] == 1 and st["trimmed_hits"] == 2
    assert pc.peek(base + [11, 50]) == 11


def test_radix_lru_eviction_by_bytes():
    pc = PrefixCache(250)
    a, b, c = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [20] * 10, [30] * 10
    for toks in (a, b):
        assert pc.insert(toks, [_FakeLayer(10)])
    pc.lookup(a + [0])                      # a is now the most recent
    assert pc.insert(c, [_FakeLayer(10)])   # 300 bytes > 250: evict the LRU (b)
    assert pc.lookup(b + [0])[1] == 0 and pc.lookup(a + [0])[1] == 10
    assert pc.stats()["evictions"] == 1 and pc.bytes == 200
    assert not pc.insert(list(range(100)), [_FakeLayer(100)])   # larger than the cache
    assert pc.stats()["too_large"] == 1


def test_radix_non_trimmable_snapshots():
    pc = PrefixCache(10_000)
    toks = list(range(1, 21))
    assert pc.insert(toks[:4], [_FakeLayer(4, trimmable=False)], "system")
    assert pc.insert(toks[:8], [_FakeLayer(8, trimmable=False)], "turn")
    assert pc.stats()["entries"] == 2 and pc.stats()["snapshots"] == 2
    # Nearest snapshot at or below the prompt; never a longer one, never trimmed.
    assert pc.lookup(toks[:6] + [99])[1] == 4
    assert pc.lookup(toks[:12])[1] == 8
    assert pc.lookup(toks[:8])[1] == 4          # exactly 8 would leave nothing to run
    assert pc.lookup([5, 6])[1] == 0


# --------------------------------------------------------------------------- #
# Engine scheduling
# --------------------------------------------------------------------------- #
def test_batch_matches_reference_and_runs_concurrently():
    eng = _engine()
    try:
        bodies = [{"messages": _msgs(f"prompt number {i} " * (i + 1)), "max_tokens": 24}
                  for i in range(3)]
        started = [_submit(eng, b) for b in bodies]
        for req, _ in started:
            assert req.done.wait(60)
        for req, events in started:
            assert req.tokens == _ref(_llama(), req.prompt, 24)
            assert req.finish_reason == "length" and req.usage["completion_tokens"] == 24
            assert _text(events) == eng.tokenizer.decode(req.tokens)
            assert events[-1][0] == "done"
        st = eng.stats()
        assert st["batch"]["peak"] >= 2 and st["batch"]["avg_decode_batch"] > 1
        assert st["requests"]["completed"] == 3 and st["mode"]["batching"] is True
    finally:
        eng.stop()


def test_join_and_leave_mid_answer():
    eng = _engine()
    try:
        order: list[str] = []
        a = eng.prepare({"messages": _msgs("the long one"), "max_tokens": 150})
        b = eng.prepare({"messages": _msgs("short"), "max_tokens": 4})
        a.emit = lambda ev: order.append("a") if ev[0] == "done" else None
        b.emit = lambda ev: order.append("b") if ev[0] == "done" else None
        eng.submit(a)
        assert _wait(lambda: len(a.tokens) >= 3)
        eng.submit(b)               # joins between two decode steps of a
        assert a.done.wait(60) and b.done.wait(60)
        assert order == ["b", "a"], order       # b left while a kept decoding
        assert a.tokens == _ref(_llama(), a.prompt, 150)
        assert b.tokens == _ref(_llama(), b.prompt, 4)
        assert eng.stats()["batch"]["peak"] == 2
    finally:
        eng.stop()


def test_prefix_reuse_greedy_equivalence():
    eng = _engine()
    try:
        sys_prompt = "You are a terse assistant. " * 3
        r1, ev1 = _run(eng, {"messages": _msgs("first question", sys_prompt), "max_tokens": 12})
        assert r1.n_cached == 0
        turn2 = _msgs("first question", sys_prompt) + [
            {"role": "assistant", "content": _text(ev1)}, {"role": "user", "content": "and then?"}]
        r2, _ = _run(eng, {"messages": turn2, "max_tokens": 12})
        # Turn 1's prompt plus its whole reply came from the cache.
        assert r2.n_cached >= len(r1.prompt) + len(r1.tokens) - 1, (r2.n_cached, len(r1.prompt))
        assert r2.usage["prompt_tokens_details"]["cached_tokens"] == r2.n_cached
        assert r2.tokens == _ref(_llama(), r2.prompt, 12)
        # Another conversation with the same system prompt: a trimmed hit.
        r3, _ = _run(eng, {"messages": _msgs("unrelated", sys_prompt), "max_tokens": 12})
        assert r3.n_cached > len(sys_prompt) and r3.tokens == _ref(_llama(), r3.prompt, 12)
        st = eng.stats()["prefix_cache"]
        assert st["hits"] == 2 and st["trimmed_hits"] >= 1 and st["misses"] == 1
    finally:
        eng.stop()


def test_hybrid_snapshots_restore_equivalence():
    model = _hybrid()
    eng = _engine(model=model)
    try:
        assert eng.trimmable is False and eng.batchable is True
        sys_prompt = "System rules for the agent. " * 2
        r1, ev1 = _run(eng, {"messages": _msgs("hello there", sys_prompt), "max_tokens": 10})
        kinds = sorted(e["kind"] for e in eng.prefix.entries())
        assert "system" in kinds and "prompt" in kinds and "final" in kinds, kinds
        assert r1.tokens == _ref(model, r1.prompt, 10)
        # Second conversation, same system prompt: resumes from the system snapshot.
        r2, _ = _run(eng, {"messages": _msgs("something else", sys_prompt), "max_tokens": 10})
        sys_end = min(e["tokens"] for e in eng.prefix.entries() if e["kind"] == "system")
        assert r2.n_cached == sys_end and r2.tokens == _ref(model, r2.prompt, 10)
        # Follow-up turn: resumes from turn 1's final state (or its prompt snapshot).
        turn2 = _msgs("hello there", sys_prompt) + [
            {"role": "assistant", "content": _text(ev1)}, {"role": "user", "content": "more"}]
        r3, ev3 = _run(eng, {"messages": turn2, "max_tokens": 10})
        assert r3.n_cached >= len(r1.prompt) - 1
        assert r3.tokens == _ref(model, r3.prompt, 10)
        # Turn 3: the end of assistant turn 2 was snapshotted while turn 3 prefilled.
        turn3 = turn2 + [{"role": "assistant", "content": _text(ev3)},
                         {"role": "user", "content": "last"}]
        r4, _ = _run(eng, {"messages": turn3, "max_tokens": 10})
        assert r4.n_cached >= len(r3.prompt) - 1 and r4.tokens == _ref(model, r4.prompt, 10)
        assert any(e["kind"] == "turn" for e in eng.prefix.entries())
        assert eng.stats()["prefix_cache"]["snapshots_taken"] >= 3
    finally:
        eng.stop()


def test_hybrid_sequential_path_snapshots():
    # kv_bits forces the one-at-a-time path; a start beyond any prompt keeps the
    # KV in fp16 so the outputs can be compared exactly with generate_step.
    model = _hybrid()
    eng = _engine(model=model, kv_bits=8, quantized_kv_start=10**6)
    try:
        assert eng.batchable is False and eng.trimmable is False
        sys_prompt = "Sequential system prompt. " * 2
        r1, ev1 = _run(eng, {"messages": _msgs("one", sys_prompt), "max_tokens": 10})
        assert r1.mode == "sequential" and r1.tokens == _ref(model, r1.prompt, 10)
        kinds = {e["kind"] for e in eng.prefix.entries()}
        assert {"system", "prompt", "final"} <= kinds, kinds
        turn2 = _msgs("one", sys_prompt) + [
            {"role": "assistant", "content": _text(ev1)}, {"role": "user", "content": "two"}]
        r2, _ = _run(eng, {"messages": turn2, "max_tokens": 10})
        assert r2.n_cached >= len(r1.prompt) - 1 and r2.tokens == _ref(model, r2.prompt, 10)
    finally:
        eng.stop()


def test_speculative_with_quantized_kv_runs():
    main, draft = _llama(0, hidden=128), _llama(3, hidden=128)
    eng = _engine(model=main, draft=draft, draft_model="tiny/draft", kv_bits=8,
                  kv_group_size=32, quantized_kv_start=0)
    try:
        assert eng.spec_ok and not eng.batchable
        r1, ev1 = _run(eng, {"messages": _msgs("quantized drafts"), "max_tokens": 24})
        assert r1.mode == "speculative" and len(r1.tokens) == 24
        turn2 = _msgs("quantized drafts") + [{"role": "assistant", "content": _text(ev1)},
                                             {"role": "user", "content": "go on"}]
        r2, _ = _run(eng, {"messages": turn2, "max_tokens": 8})
        assert r2.n_cached >= len(r1.prompt) and len(r2.tokens) == 8
        st = eng.stats()
        assert st["speculative"]["rounds"] > 0 and st["requests"]["errors"] == 0
    finally:
        eng.stop()


def test_cancellation_frees_the_slot():
    eng = _engine(max_batch=1)
    try:
        req, events = _submit(eng, {"messages": _msgs("cancel me"), "max_tokens": 2000})
        waiting, _ = _submit(eng, {"messages": _msgs("waiting behind"), "max_tokens": 5})
        assert _wait(lambda: len(req.tokens) >= 5)
        assert not waiting.done.is_set() and eng.stats()["queue"]["waiting"] == 1
        req.cancel()
        assert req.done.wait(10) and req.finish_reason == "cancelled"
        assert events[-1][0] == "done" and events[-1][1] == "cancelled"
        assert waiting.done.wait(30) and waiting.tokens == _ref(_llama(), waiting.prompt, 5)
        # The cancelled request's computed state is kept for a retry.
        assert any(e["kind"] == "partial" for e in eng.prefix.entries())
        st = eng.stats()
        assert st["requests"]["cancelled"] == 1 and st["batch"]["active"] == 0
        # A cancelled request still waiting in the queue never starts.
        a, _ = _submit(eng, {"messages": _msgs("long"), "max_tokens": 300})
        b, _ = _submit(eng, {"messages": _msgs("never"), "max_tokens": 5})
        b.cancel()
        assert b.done.wait(10) and b.tokens == [] and b.finish_reason == "cancelled"
        a.cancel()
        assert a.done.wait(10)
    finally:
        eng.stop()


def test_stop_strings_and_eos():
    eng = _engine()
    try:
        body = {"messages": _msgs("stop test"), "max_tokens": 40}
        ref = _ref(_llama(), eng.prepare(body).prompt, 40)
        full = eng.tokenizer.decode(ref)
        stop = full[10:13]
        r, ev = _run(eng, {**body, "stop": [stop, "ǿǾǽ-never"]})
        assert r.finish_reason == "stop" and _text(ev) == full[:full.index(stop)]
        assert "".join(e[1] for e in ev if e[0] == "delta") == full[:full.index(stop)]
    finally:
        eng.stop()
    # EOS: a tokenizer whose eos is the 6th reference token ends the answer there.
    eos = ref[5]
    eng = _engine(tok=_StubTok(eos=eos))
    try:
        r, ev = _run(eng, body)
        cut = ref.index(eos) + 1
        assert r.finish_reason == "stop" and r.tokens == ref[:cut]
        assert _text(ev) == eng.tokenizer.decode(ref[:cut - 1])
    finally:
        eng.stop()


def test_queue_limit_and_sampling_options():
    eng = _engine(max_batch=1, max_queue=1)
    try:
        a, _ = _submit(eng, {"messages": _msgs("a"), "max_tokens": 3000})
        assert _wait(lambda: len(a.tokens) >= 1)
        b, _ = _submit(eng, {"messages": _msgs("b"), "max_tokens": 5})
        try:
            _submit(eng, {"messages": _msgs("c"), "max_tokens": 5})
            raise AssertionError("expected EngineBusy")
        except EngineBusy as exc:
            assert "queue is full" in str(exc)
        assert eng.stats()["requests"]["rejected"] == 1
        a.cancel()
        assert b.done.wait(30)
    finally:
        eng.stop()
    eng = _engine()
    try:
        # Seeded sampling is reproducible whatever else is in the batch.
        body = {"messages": _msgs("sample"), "max_tokens": 16, "temperature": 0.9, "seed": 7,
                "top_p": 0.95}
        s1, _ = _run(eng, body)
        noise = [_submit(eng, {"messages": _msgs(f"noise {i}"), "max_tokens": 30,
                               "temperature": 1.0}) for i in range(2)]
        s2, _ = _run(eng, body)
        for req, _ in noise:
            assert req.done.wait(30)
        assert s1.tokens == s2.tokens and len(s1.tokens) == 16
        rp, _ = _run(eng, {"messages": _msgs("penalty"), "max_tokens": 12,
                           "repetition_penalty": 1.3, "logit_bias": {"5": 2.0}})
        assert rp.finish_reason == "length" and len(rp.tokens) == 12
        for bad in ({"messages": []}, {"messages": [{"content": "x"}]},
                    {"messages": _msgs("x"), "max_tokens": 0},
                    {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}):
            try:
                eng.prepare(bad)
                raise AssertionError(f"accepted {bad}")
            except ValueError:
                pass
    finally:
        eng.stop()


def test_speculative_alone_then_batched():
    main, draft = _llama(0), _llama(seed=3)
    eng = _engine(model=main, draft=draft, draft_model="tiny/draft", num_draft_tokens=3)
    try:
        assert eng.spec_ok
        r1, _ = _run(eng, {"messages": _msgs("speculate"), "max_tokens": 30})
        assert r1.mode == "speculative" and r1.tokens == _ref(main, r1.prompt, 30)
        assert eng.stats()["speculative"]["rounds"] > 0
        # A second request arrives while one drafts: the first joins the batch.
        a = eng.prepare({"messages": _msgs("long speculative answer"), "max_tokens": 200})
        a.emit = lambda ev: None
        eng.submit(a)
        assert _wait(lambda: len(a.tokens) >= 4)
        b, _ = _run(eng, {"messages": _msgs("concurrent"), "max_tokens": 8})
        assert a.done.wait(60)
        assert a.mode == "batch" and eng.stats()["speculative"]["migrations"] == 1
        assert a.tokens == _ref(main, a.prompt, 200)
        assert b.tokens == _ref(main, b.prompt, 8)
        # Same draft prefix reused on a follow-up of the first conversation.
        assert eng.stats()["draft_prefix_cache"]["inserts"] >= 1
    finally:
        eng.stop()


def test_speculative_request_migrates_during_prefill():
    main, draft = _llama(0), _llama(seed=3)
    eng = _engine(model=main, draft=draft, draft_model="tiny/draft")
    try:
        a = eng.prepare({"messages": _msgs("long prompt " * 160), "max_tokens": 12})
        a.emit = lambda ev: None
        eng.submit(a)
        assert _wait(lambda: eng.stats()["mode"]["running"] == "speculative", 10)
        b, _ = _run(eng, {"messages": _msgs("arrives during prefill"), "max_tokens": 6})
        assert a.done.wait(60) and a.mode == "batch"
        assert eng.stats()["speculative"]["migrations_in_prefill"] == 1
        assert a.tokens == _ref(main, a.prompt, 12) and b.tokens == _ref(main, b.prompt, 6)
    finally:
        eng.stop()


def test_sequential_prefill_is_cancellable():
    model = _llama(hidden=128)
    eng = _engine(model=model, kv_bits=8, kv_group_size=32, quantized_kv_start=0)
    try:
        req, _ = _submit(eng, {"messages": _msgs("very long prompt " * 300), "max_tokens": 5})
        assert _wait(lambda: eng._job is not None and eng._job.pos > 0, 10)
        req.cancel()
        assert req.done.wait(10) and req.finish_reason == "cancelled" and req.tokens == []
        part = [e for e in eng.prefix.entries() if e["kind"] == "partial"]
        assert part and 0 < part[0]["tokens"] < len(req.prompt)
    finally:
        eng.stop()


def test_kv_bits_serves_sequentially_with_prefix_cache():
    # mx.quantize needs head_dim divisible by the group size (32 at least).
    model = _llama(hidden=128)
    eng = _engine(model=model, kv_bits=8, kv_group_size=32, quantized_kv_start=0)
    try:
        assert eng.batchable is False
        # Same prefill chunking as the engine: with quantized_kv_start=0 the first
        # chunk attends in fp16 and later ones to quantized keys, so the chunk size
        # is part of the numerics.
        kw = dict(kv_bits=8, kv_group_size=32, quantized_kv_start=0, prefill_step_size=16)
        a, ev_a = _submit(eng, {"messages": _msgs("quantized one"), "max_tokens": 20})
        b, _ = _submit(eng, {"messages": _msgs("quantized two"), "max_tokens": 20})
        assert a.done.wait(60) and b.done.wait(60)
        assert eng.stats()["batch"]["peak"] == 1     # one at a time
        assert a.tokens == _ref(model, a.prompt, 20, **kw)
        assert b.tokens == _ref(model, b.prompt, 20, **kw)
        turn2 = _msgs("quantized one") + [{"role": "assistant", "content": _text(ev_a)},
                                          {"role": "user", "content": "again"}]
        c, _ = _run(eng, {"messages": turn2, "max_tokens": 20})
        assert c.n_cached >= len(a.prompt)
        entry = next(e for e in eng.prefix.entries() if e["kind"] == "final")
        assert entry["trimmable"]
        assert c.tokens == _ref(model, c.prompt, 20, **kw)
    finally:
        eng.stop()


def test_memory_admission_waits_for_room():
    eng = _engine(kv_budget_bytes=1)       # any second sequence exceeds the budget
    try:
        a, _ = _submit(eng, {"messages": _msgs("first"), "max_tokens": 40})
        b, _ = _submit(eng, {"messages": _msgs("second"), "max_tokens": 40})
        assert a.done.wait(60) and b.done.wait(60)
        st = eng.stats()
        assert st["batch"]["peak"] == 1 and st["requests"]["admission_waits"] == 1
        assert st["admission"]["kv_bytes_per_token"] > 0
        assert b.tokens == _ref(_llama(), b.prompt, 40)
    finally:
        eng.stop()


def test_disk_persistence_roundtrip():
    tmp = Path(tempfile.mkdtemp(prefix="engine-disk-"))
    body = {"messages": _msgs("persist me " * 6), "max_tokens": 10}
    eng = _engine(prefix_cache_dir=str(tmp))
    try:
        r1, ev1 = _run(eng, body)
    finally:
        eng.stop()          # flushes memory entries to disk
    files = list(tmp.rglob("*.safetensors"))
    assert files and (files[0].parent / "index.json").is_file()
    eng = _engine(prefix_cache_dir=str(tmp))
    try:
        assert eng.prefix.stats()["disk"]["entries"] == 1
        turn2 = body["messages"] + [{"role": "assistant", "content": _text(ev1)},
                                    {"role": "user", "content": "next"}]
        r2, _ = _run(eng, {"messages": turn2, "max_tokens": 10})
        assert r2.n_cached >= len(r1.prompt) and eng.prefix.stats()["disk"]["loads"] == 1
        assert r2.tokens == _ref(_llama(), r2.prompt, 10)
    finally:
        eng.stop()
    # A different adapter is a different namespace: nothing is served across it.
    eng = _engine(prefix_cache_dir=str(tmp), adapter_path=str(tmp / "other-adapter"))
    try:
        assert eng.prefix.stats()["disk"]["entries"] == 0
    finally:
        eng.stop()


def test_snapshot_copy_is_independent():
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    model = _llama()
    prompt = list(range(10, 40))
    cache = make_prompt_cache(model)
    model(mx.array(prompt[:20])[None], cache=cache)
    snap = _eng_cache_copy(cache)
    model(mx.array(prompt[20:])[None], cache=cache)          # the original keeps going
    assert cache[0].offset == 30 and snap[0].offset == 20
    from mlx_lm.generate import generate_step
    out = [int(t) for t, _ in generate_step(mx.array(prompt[20:]), model, max_tokens=8,
                                            prompt_cache=snap)]
    assert out == _ref(model, prompt, 8)


# --------------------------------------------------------------------------- #
# HTTP surface
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_http_surface():
    import httpx

    eng = _engine(max_batch=1, max_queue=1)
    port = _free_port()
    server = start_engine_server(eng, port=port)
    base = f"http://127.0.0.1:{port}"
    try:
        assert httpx.get(base + "/health").json()["status"] == "ok"
        assert httpx.get(base + "/v1/models").json()["data"][0]["id"] == eng.served_name
        body = {"model": "default_model", "messages": _msgs("http"), "max_tokens": 8}
        r = httpx.post(base + "/v1/chat/completions", json=body, timeout=30)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["object"] == "chat.completion" and data["choices"][0]["finish_reason"] == "length"
        assert data["usage"]["completion_tokens"] == 8
        assert data["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        # Streaming with usage; the repeat is served from the prefix cache.
        frames = []
        with httpx.stream("POST", base + "/v1/chat/completions", timeout=30,
                          json={**body, "stream": True, "stream_options": {"include_usage": True}}) as s:
            for line in s.iter_lines():
                if line.startswith("data: "):
                    frames.append(line[6:])
        assert frames[-1] == "[DONE]"
        chunks = [json.loads(f) for f in frames[:-1]]
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
        assert text == data["choices"][0]["message"]["content"]
        assert [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]][-1] == "length"
        assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
        # Errors: unknown model, bad body.
        assert httpx.post(base + "/v1/chat/completions", json={**body, "model": "other/x"}).status_code == 404
        assert httpx.post(base + "/v1/chat/completions", json={"messages": []}).status_code == 400
        assert httpx.post(base + "/v1/chat/completions", content=b"{nope").status_code == 400
        # Disconnecting a stream cancels it; a full queue answers 503.
        with httpx.stream("POST", base + "/v1/chat/completions", timeout=30,
                          json={**body, "stream": True, "max_tokens": 4000}) as s:
            lines = s.iter_lines()
            for _ in range(4):
                next(lines)
            held = threading.Thread(target=lambda: httpx.post(
                base + "/v1/chat/completions", json={**body, "max_tokens": 3}, timeout=60), daemon=True)
            held.start()
            assert _wait(lambda: eng.stats()["queue"]["waiting"] == 1)
            busy = httpx.post(base + "/v1/chat/completions", json=body, timeout=30)
            assert busy.status_code == 503 and busy.headers.get("retry-after")
        assert _wait(lambda: eng.stats()["requests"]["cancelled"] >= 1)
        held.join(30)
        stats = httpx.get(base + "/v1/engine/stats").json()
        for key in ("batch", "queue", "prefix_cache", "decode", "ttft_ms", "speculative", "admission"):
            assert key in stats
        assert stats["ttft_ms"]["n"] >= 3 and stats["decode"]["tok_s"] > 0
        assert stats["prefix_cache"]["hit_tokens"] > 0
    finally:
        server.should_exit = True
        server.thread.join(10)
        eng.stop()


# --------------------------------------------------------------------------- #
# Backend registration
# --------------------------------------------------------------------------- #
def test_backend_spec_and_launcher():
    spec = get_backend("mlx-engine")
    assert spec.name == "mlx-engine" and "prompt_cache" in spec.capabilities
    mm = ModelServerManager("org/test-model", 8091, ADAPTER_DIR, backend="mlx-engine")
    mm.kv_bits, mm.prompt_cache_bytes, mm.prompt_cache_dir = 0, 123456789, "/tmp/pc"
    mm.prefill_step_size, mm.decode_concurrency, mm.web_port = 256, 0, 8000
    mm.max_concurrent_generations = 2
    cmd = spec.launcher(mm, False)
    assert cmd[0] == sys.executable and "--adapter-path" not in cmd
    flags = cmd[cmd.index("--model"):]
    args = _eng_parser().parse_args(flags)
    assert args.model == "org/test-model" and args.port == 8091
    assert args.prefix_cache_bytes == 123456789 and args.prefix_cache_dir == "/tmp/pc"
    assert args.prefill_step_size == 256 and args.max_batch == 3
    assert args.allowed_origins.startswith("http://127.0.0.1:8000")
    assert "--max-batch" in mm._accepted_levers
    assert request_model_for(mm) == "default_model"
    assert "batching" in spec.caps_for(mm) and "kv_quant" not in spec.caps_for(mm)
    mm.kv_bits = 8
    plan = engine_plan(mm)
    assert plan["batching"] is False and any("one at a time" in d for d in plan["decisions"])
    assert "kv_quant" in spec.caps_for(mm) and "batching" not in spec.caps_for(mm)
    assert set(spec.launch_inputs) >= {"kv_bits", "prompt_cache_bytes", "draft_model",
                                       "max_concurrent_generations"}
    assert spec.probe_timeout(mm) >= 90


# --------------------------------------------------------------------------- #
# Live (opt-in): the cached 3B model
# --------------------------------------------------------------------------- #
LIVE_MODEL = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"


def test_live_engine_on_cached_model():
    if os.environ.get("LLM_ENGINE_LIVE") != "1":
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    from mlx_lm import load

    model, tok = load(LIVE_MODEL)
    eng = MLXEngine(EngineOptions(model=LIVE_MODEL, kv_budget_bytes=0), model=model, tokenizer=tok)
    eng.start()
    try:
        assert eng.wait_ready(120)
        msgs = _msgs("Write a Python one-liner that reverses a string.", "You are concise.")
        r1, ev1 = _run(eng, {"messages": msgs, "max_tokens": 40}, timeout=300)
        assert r1.tokens == _ref(model, r1.prompt, 40, eos=eng._eos)
        turn2 = msgs + [{"role": "assistant", "content": _text(ev1)},
                        {"role": "user", "content": "Now in JavaScript."}]
        r2, _ = _run(eng, {"messages": turn2, "max_tokens": 40}, timeout=300)
        assert r2.n_cached > len(r1.prompt) // 2
        assert r2.tokens == _ref(model, r2.prompt, 40, eos=eng._eos)
    finally:
        eng.stop()


def _run_standalone():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"FAIL  {name}: {exc}")
            failed += 1
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
