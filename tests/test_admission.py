"""Tests for memory-aware admission control (item 6, local_llm/admission.py).

Controller tests drive AdmissionController directly on a fresh event loop with
a fake memory probe, so they never depend on this machine's free memory. Route
tests reuse test_app's construction (copied, not imported) and hold slots on
the app's own loop through the TestClient portal, so the queue they observe is
the real one, not a mock.
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_llm.config as config_mod  # noqa: E402

config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-admission-"))

from fastapi.testclient import TestClient  # noqa: E402

from local_llm.admission import (  # noqa: E402
    AdmissionController, _adm_parse_vm_stat, admission_cost_tokens,
    estimate_kv_bytes_per_token)
from local_llm.api import create_app  # noqa: E402
from local_llm.auth import hash_password  # noqa: E402
from local_llm.config import Config  # noqa: E402
from local_llm.core import ADAPTER_DIR, GenerationBusy, GenerationGate  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.sysutil import hf_cache_dir  # noqa: E402
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.training import RetrainManager  # noqa: E402

MB = 1024 * 1024


def build_app(**overrides):
    """Same construction as tests/test_app.py:build_app."""
    base = dict(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                allow_test_user=False, node_token="secret-node-token",
                skills_dir=tempfile.mkdtemp(prefix="local-llm-test-skills-"))
    base.update(overrides)
    cfg = Config(**base)
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)  # never started
    rm = RetrainManager(db, mm, cfg)
    app = create_app(cfg, db, mm, rm, ToolRegistry(cfg, db))
    app.state.model_manager = mm
    return app, cfg, db


def login(client, username, password):
    return client.post("/api/auth/login", json={"username": username, "password": password})


def _final(answer: str = "ok") -> dict:
    return {"type": "final", "answer": answer, "steps": 1, "trace": [],
            "tools_used": [], "elapsed_ms": 1, "prompt_tokens": 1,
            "completion_tokens": 1, "truncated": False, "changed_files": [], "diff": ""}


def _sse_events(text: str) -> list:
    out = []
    for line in text.splitlines():
        if line.startswith("data: "):
            body = line[len("data: "):]
            out.append(body if body == "[DONE]" else json.loads(body))
    return out


# --------------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------------- #
def test_is_a_drop_in_generation_gate():
    gate = AdmissionController(2, 3)
    assert isinstance(gate, GenerationGate)
    assert (gate.limit, gate.max_waiting, gate.waiting) == (2, 6, 0)

    async def main():
        # The old ticket-less form still works in one coroutine.
        await gate.acquire()
        assert gate.running == 1
        gate.release()
        assert gate.running == 0
        async with gate.slot(cost_tokens=10):
            assert gate.running == 1
        assert gate.running == 0
        gate.release()          # a stray release never goes negative or raises
        assert gate.running == 0

    asyncio.run(main())


def test_waiters_are_admitted_fifo():
    gate = AdmissionController(1, 8, memory_aware=False)
    order: list[str] = []

    async def worker(name: str, cost: int):
        ticket = await gate.acquire(cost_tokens=cost)
        order.append(name)
        await asyncio.sleep(0)
        gate.release(ticket)

    async def main():
        first = await gate.acquire()
        tasks = []
        for name, cost in (("a", 5000), ("b", 1), ("c", 300)):
            tasks.append(asyncio.ensure_future(worker(name, cost)))
            await asyncio.sleep(0)   # enqueue in this order
        assert gate.waiting == 3
        gate.release(first)
        await asyncio.gather(*tasks)

    asyncio.run(main())
    assert order == ["a", "b", "c"]


def test_bounded_queue_raises_generation_busy():
    gate = AdmissionController(1, 1)

    async def main():
        held = await gate.acquire()
        queued = asyncio.ensure_future(gate.acquire())
        await asyncio.sleep(0)
        assert gate.waiting == 1
        try:
            await gate.acquire()
        except GenerationBusy:
            pass
        else:
            raise AssertionError("third request should have been refused")
        assert gate.snapshot(probe=False)["rejected_total"] == 1
        gate.release(held)
        gate.release(await queued)
        assert gate.running == 0

    asyncio.run(main())


def test_queue_positions_are_reported_and_cancel_leaves_the_queue():
    gate = AdmissionController(1, 8, memory_aware=False)
    seen: dict[str, list[int]] = {"a": [], "b": [], "c": []}

    async def main():
        held = await gate.acquire()
        a = asyncio.ensure_future(gate.acquire(on_wait=lambda p, s: seen["a"].append(p)))
        await asyncio.sleep(0)
        b = asyncio.ensure_future(gate.acquire(on_wait=lambda p, s: seen["b"].append(p)))
        await asyncio.sleep(0)
        c = asyncio.ensure_future(gate.acquire(on_wait=lambda p, s: seen["c"].append(p)))
        await asyncio.sleep(0)
        assert (seen["a"], seen["b"], seen["c"]) == ([1], [2], [3])
        b.cancel()                     # client gone while queued
        await asyncio.sleep(0)
        assert gate.waiting == 2 and seen["c"] == [3, 2]
        gate.release(held)
        gate.release(await a)
        gate.release(await c)
        assert seen["c"] == [3, 2, 1]
        assert gate.running == 0 and gate.waiting == 0

    asyncio.run(main())


def test_memory_gate_defers_until_memory_returns():
    free = {"bytes": 10_000 * MB}
    gate = AdmissionController(4, 4, kv_bytes_per_token=MB, min_free_mb=1000,
                               probe=lambda: free["bytes"], recheck_seconds=0.01)

    async def main():
        first = await gate.acquire(cost_tokens=2000)          # 2000MB reserved
        assert gate.running == 1
        # 10000 - 2000 reserved - 8000 cost = 0 < 1000 floor: must wait even
        # though three slots are free.
        second = asyncio.ensure_future(gate.acquire(cost_tokens=8000))
        await asyncio.sleep(0.03)
        assert not second.done() and gate.waiting == 1
        snap = gate.snapshot()
        assert snap["deferred_total"] == 1 and snap["memory_aware"] is True
        assert snap["memory_probe"] == "custom"
        # Memory comes back without any release here (another process quit):
        # the recheck timer admits it.
        free["bytes"] = 20_000 * MB
        ticket = await asyncio.wait_for(second, 1)
        assert gate.running == 2
        gate.release(ticket)
        gate.release(first)
        assert gate.running == 0 and gate.reserved_bytes == 0

    asyncio.run(main())


def test_release_admits_a_memory_blocked_head():
    gate = AdmissionController(4, 4, kv_bytes_per_token=MB, min_free_mb=500,
                               probe=lambda: 4_000 * MB, recheck_seconds=10)

    async def main():
        first = await gate.acquire(cost_tokens=3000)
        second = asyncio.ensure_future(gate.acquire(cost_tokens=1000))
        await asyncio.sleep(0)
        assert not second.done()
        gate.release(first)           # running 0: the head is lone, so admitted
        gate.release(await asyncio.wait_for(second, 1))

    asyncio.run(main())


def test_lone_request_is_always_admitted():
    gate = AdmissionController(2, 2, kv_bytes_per_token=MB, min_free_mb=4096,
                               probe=lambda: 0)

    async def main():
        ticket = await asyncio.wait_for(gate.acquire(cost_tokens=100_000), 1)
        assert gate.running == 1
        gate.release(ticket)

    asyncio.run(main())


def test_memory_awareness_off_or_unmeasurable_is_a_plain_slot_gate():
    for kwargs in ({"memory_aware": False, "probe": lambda: 0},
                   {"memory_aware": True, "probe": lambda: None}):
        gate = AdmissionController(2, 2, kv_bytes_per_token=MB, min_free_mb=4096, **kwargs)

        async def main():
            a = await gate.acquire(cost_tokens=50_000)
            b = await asyncio.wait_for(gate.acquire(cost_tokens=50_000), 1)
            assert gate.running == 2
            snap = gate.snapshot()
            assert snap["memory_aware"] is False
            gate.release(a)
            gate.release(b)

        asyncio.run(main())


def test_resize_grows_and_shrinks_live():
    gate = AdmissionController(1, 2, memory_aware=False)

    async def main():
        a = await gate.acquire()
        b = asyncio.ensure_future(gate.acquire())
        await asyncio.sleep(0)
        assert not b.done()
        gate.resize(2)                                 # grow: admitted at once
        tb = await asyncio.wait_for(b, 1)
        assert gate.running == 2 and gate.max_waiting == 4
        gate.resize(1, queue_multiplier=3)             # shrink: nobody interrupted
        assert gate.running == 2 and gate.max_waiting == 3
        c = asyncio.ensure_future(gate.acquire())
        await asyncio.sleep(0)
        gate.release(a)                                # running 1, limit 1: c waits
        await asyncio.sleep(0)
        assert not c.done()
        gate.release(tb)
        gate.release(await asyncio.wait_for(c, 1))

    asyncio.run(main())


def test_resize_from_a_worker_thread_wakes_the_loop():
    gate = AdmissionController(1, 2, memory_aware=False)

    async def main():
        a = await gate.acquire()
        b = asyncio.ensure_future(gate.acquire())
        await asyncio.sleep(0)
        # The sync /api/config handler runs in a thread pool.
        await asyncio.to_thread(gate.resize, 2)
        gate.release(await asyncio.wait_for(b, 1))
        gate.release(a)

    asyncio.run(main())


def test_snapshot_shape():
    gate = AdmissionController(3, 2, kv_bytes_per_token=1000, min_free_mb=256,
                               probe=lambda: 8_000 * MB)

    async def main():
        t = await gate.acquire(cost_tokens=2048)
        snap = gate.snapshot()
        assert snap["running"] == 1 and snap["waiting"] == 0
        assert snap["limit"] == 3 and snap["max_waiting"] == 6
        assert snap["free_mb"] == 8000 and snap["min_free_mb"] == 256
        assert snap["kv_bytes_per_token"] == 1000
        assert abs(snap["reserved_mb"] - round(2048 * 1000 / MB, 1)) < 0.11
        assert snap["admitted_total"] == 1
        for key in ("rejected_total", "deferred_total", "wait_ms_p50", "wait_ms_p95",
                    "queue_multiplier", "memory_probe"):
            assert key in snap
        json.dumps(snap)            # served as JSON by /api/health
        gate.release(t)

    asyncio.run(main())


def test_configure_switches_memory_awareness_live():
    gate = AdmissionController(4, 2, kv_bytes_per_token=MB, min_free_mb=0,
                               memory_aware=False, probe=lambda: 1000 * MB)

    async def main():
        a = await gate.acquire(cost_tokens=10)
        b = await gate.acquire(cost_tokens=2000)        # unaware: admitted
        gate.configure(memory_aware=True, min_free_mb=100)
        c = asyncio.ensure_future(gate.acquire(cost_tokens=10))
        await asyncio.sleep(0)
        assert not c.done()                             # 1000 - 2010 reserved < 100
        gate.release(b)
        gate.release(await asyncio.wait_for(c, 1))
        gate.release(a)

    asyncio.run(main())


def test_cost_tokens_counts_prompt_and_reply_and_caps_at_context():
    messages = [{"role": "system", "content": "x" * 400},
                {"role": "user", "content": "hello there"}]
    cost = admission_cost_tokens(messages, 512)
    assert cost > 512
    assert admission_cost_tokens(messages, 512, context_size=300) == 300
    assert admission_cost_tokens([], 0) == 0
    # Non-string content (image parts) is measured, not dropped.
    assert admission_cost_tokens([{"role": "user", "content": [{"type": "text",
                                                                "text": "y" * 200}]}], 0) > 0


def test_kv_bytes_per_token_from_the_cached_qwen_config():
    repo = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
    snaps = hf_cache_dir() / ("models--" + repo.replace("/", "--")) / "snapshots"
    configs = sorted(snaps.glob("*/config.json")) if snaps.is_dir() else []
    if not configs:
        print("SKIP  Qwen2.5-Coder-3B is not cached on this machine")
        return
    cfg = json.loads(configs[0].read_text(encoding="utf-8"))
    heads = cfg["num_attention_heads"]
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // heads
    expected = 2 * cfg["num_hidden_layers"] * cfg.get("num_key_value_heads", heads) * head_dim * 2
    assert estimate_kv_bytes_per_token(repo) == expected
    # 36 layers x 2 KV heads x 128 dims x K and V x fp16.
    assert expected == 36864
    assert estimate_kv_bytes_per_token(repo, kv_bits=8) == expected // 2
    assert estimate_kv_bytes_per_token(repo, kv_bits=4) == expected // 4


def test_kv_bytes_per_token_falls_back_by_ram_tier():
    unknown = "nobody/not-a-cached-model-xyz"
    small = estimate_kv_bytes_per_token(unknown, ram_gb=8)
    big = estimate_kv_bytes_per_token(unknown, ram_gb=64)
    assert 0 < small < big
    assert estimate_kv_bytes_per_token(unknown, kv_bits=4, ram_gb=8) == small // 4


def test_vm_stat_parser():
    sample = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
              "Pages free:                               10000.\n"
              "Pages active:                             50000.\n"
              "Pages inactive:                           20000.\n"
              "Pages speculative:                         1000.\n")
    assert _adm_parse_vm_stat(sample) == (10000 + 20000 + 1000) * 16384
    assert _adm_parse_vm_stat("garbage") is None


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
def test_create_app_wires_one_controller_everywhere():
    app, cfg, _ = build_app(max_concurrent_generations=2, generation_queue_depth=3)
    gate = app.state.ctx.generation_gate
    assert isinstance(gate, AdmissionController)
    assert app.state.tasks.gate is gate
    assert (gate.limit, gate.max_waiting) == (2, 6)
    # With no ADMISSION_* env set, behaviour matches the old slot gate.
    assert gate.memory_aware is Config().admission_memory_aware


def test_busy_chat_does_not_leak_chat_in_flight():
    """Regression: /api/chat raised chat_in_flight before acquiring the gate and
    returned 503 without lowering it, so every refusal leaked +1 and the
    scheduler (which waits for zero) never ran a task again."""
    app, cfg, _ = build_app(max_concurrent_generations=1, generation_queue_depth=1,
                            agent_enabled=False, knowledge_triage=False)
    app.state.model_manager.status = "ready"

    async def fake_run(message, history, conversation_id=None, max_tokens=None,
                       temperature=None, cancel=None):
        yield _final()

    app.state.agent.run_iterating = fake_run
    gate = app.state.ctx.generation_gate
    tasks = app.state.tasks
    with TestClient(app) as c:
        assert login(c, "admin", "adminpw123").status_code == 200
        held = c.portal.call(gate.acquire)          # the one slot, on the app's loop
        gate.max_waiting = 0                        # and a full waiting room
        try:
            for _ in range(3):
                r = c.post("/api/chat", json={"message": "hi", "agent": True})
                assert r.status_code == 503, r.text
                assert r.headers.get("retry-after") == "5"
            assert tasks.chat_in_flight == 0
            r = c.post("/api/chat/stream", json={"message": "hi", "agent": True})
            events = _sse_events(r.text)
            assert events[0]["type"] == "error" and events[-1] == "[DONE]"
            assert tasks.chat_in_flight == 0
        finally:
            gate.max_waiting = 1
            c.portal.call(gate.release, held)
        # And with the slot free again the same route works and gives it back.
        r = c.post("/api/chat", json={"message": "hi", "agent": True})
        assert r.status_code == 200, r.text
        assert tasks.chat_in_flight == 0
        assert c.portal.call(gate.snapshot)["running"] == 0


def test_stream_emits_queued_then_normal_events():
    app, cfg, _ = build_app(max_concurrent_generations=1, generation_queue_depth=4)
    app.state.model_manager.status = "ready"

    async def fake_run(message, history, conversation_id=None, max_tokens=None,
                       temperature=None, cancel=None):
        yield {"type": "token", "text": "o"}
        yield _final()

    app.state.agent.run_iterating = fake_run
    gate = app.state.ctx.generation_gate
    with TestClient(app) as c:
        assert login(c, "admin", "adminpw123").status_code == 200
        held = c.portal.call(gate.acquire)
        # Free the slot from another thread once the stream is queued behind it.
        released = threading.Event()

        def free_later():
            for _ in range(200):
                if c.portal.call(lambda: gate.waiting) >= 1:
                    break
                threading.Event().wait(0.01)
            c.portal.call(gate.release, held)
            released.set()

        threading.Thread(target=free_later, daemon=True).start()
        r = c.post("/api/chat/stream", json={"message": "hello", "agent": True})
        assert released.wait(5)
        events = _sse_events(r.text)
        kinds = [e["type"] if isinstance(e, dict) else e for e in events]
        assert kinds[0] == "queued", kinds
        assert events[0]["position"] == 1 and events[0]["running"] == 1
        assert events[0]["limit"] == 1
        assert kinds.index("start") > 0 and "final" in kinds and kinds[-1] == "[DONE]"
        snap = c.portal.call(gate.snapshot)
        assert snap["running"] == 0 and snap["waiting"] == 0
        assert app.state.tasks.chat_in_flight == 0


def test_health_queue_snapshot_admin_and_user():
    app, cfg, db = build_app()
    with TestClient(app) as c:
        assert login(c, "admin", "adminpw123").status_code == 200
        queue = c.get("/api/health").json()["queue"]
        assert queue["limit"] == cfg.max_concurrent_generations
        assert "free_mb" in queue and "admitted_total" in queue
        c.post("/api/auth/logout")
        c.cookies.clear()
        db.create_user("carol", password_hash=hash_password("carolpw123"), role="user")
        assert login(c, "carol", "carolpw123").status_code == 200
        queue = c.get("/api/health").json()["queue"]
        assert set(queue) == {"running", "waiting", "limit", "max_waiting"}


def test_config_resizes_and_reconfigures_the_controller():
    app, cfg, _ = build_app(max_concurrent_generations=1, generation_queue_depth=2)
    gate = app.state.ctx.generation_gate
    with TestClient(app) as c:
        assert login(c, "admin", "adminpw123").status_code == 200
        r = c.post("/api/config", json={"max_concurrent_generations": 3,
                                        "generation_queue_depth": 5})
        assert r.status_code == 200, r.text
        assert (gate.limit, gate.max_waiting) == (3, 15)
        r = c.post("/api/config", json={"admission_memory_aware": True,
                                        "admission_min_free_mb": 640})
        assert r.status_code == 200, r.text
        assert gate.memory_aware is True and gate.min_free_mb == 640
        r = c.post("/api/config", json={"max_concurrent_generations": 999})
        assert gate.limit == cfg.max_concurrent_generations == 64


def test_scheduled_run_queues_and_cancels_while_waiting():
    from local_llm.tasks import TaskRun

    app, cfg, _ = build_app()
    tasks = app.state.tasks
    emitted: list[dict] = []
    tasks._emit = lambda run, event: emitted.append(event)
    task = {"goal": "summarise the logs"}

    async def main():
        gate = AdmissionController(1, 2, memory_aware=False)
        tasks.gate = gate
        held = await gate.acquire()
        run = TaskRun("r1", "t1", "nightly")
        waiting = asyncio.ensure_future(tasks._acquire_generation_slot(run, task, cfg, []))
        await asyncio.sleep(0.01)
        assert gate.waiting == 1
        assert emitted and emitted[0]["type"] == "queued" and emitted[0]["position"] == 1
        run.cancel.set()                       # Cancel works while queued
        try:
            await asyncio.wait_for(waiting, 1)
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("a cancelled run must not get a slot")
        await asyncio.sleep(0)
        assert gate.waiting == 0
        # A run that gets in holds a real, cost-carrying ticket.
        gate.release(held)
        ticket = await tasks._acquire_generation_slot(TaskRun("r2", "t1", "nightly"),
                                                      task, cfg, [])
        assert ticket is not None and ticket.cost_tokens > cfg.max_tokens - 1
        assert gate.running == 1
        gate.release(ticket)

    asyncio.run(main())


def _run_standalone():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            import traceback; traceback.print_exc(); print(f"FAIL  {name}: {exc}"); failed += 1
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
