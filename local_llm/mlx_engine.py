"""mlx-engine: the app's own OpenAI-compatible MLX inference server.

Items 1 (a high-performance MLX backend), 5 (a cross-request prefix cache that
also works for hybrid DeltaNet / Mamba models) and the in-engine half of 6
(memory-aware admission). Run it as

    python -m local_llm.mlx_engine --model REPO --port 8090 [...]
    python deploy.py mlx-engine --model REPO --port 8090 [...]   (bundled build)

or select it with MODEL_BACKEND=mlx-engine, which registers the BackendSpec
below so ModelServerManager launches and supervises it like mlx_lm.server.

Why another server when mlx_lm.server exists: the stock server makes three
process-wide trade-offs this app keeps losing. A loaded draft model turns
continuous batching off for every request, so speculative decoding (the best
single-user speed-up) and batching (the only multi-user one) exclude each other.
Its prompt cache is an entry-count LRU whose byte cap is enforced only when a
request joins a batch. And for hybrid models (Qwen3.5 / Qwen3-Next DeltaNet,
Mamba) whose recurrent state cannot be trimmed, a multi-turn agent re-prefills
the whole conversation whenever the re-rendered history differs by one token
from what was generated. This engine is built from the same mlx-lm primitives
and makes those decisions per request instead.

Facts about the installed mlx-lm 0.32.2 this module builds on (read from
.venv/.../mlx_lm; line numbers are that version's):

  * BatchGenerator (generate.py:1541) is continuous batching: insert_segments()
    queues a sequence with an optional restored cache plus the tokens it already
    holds (all_tokens); every _next() (generate.py:1812) runs one decode step for
    the running batch and then ONE prefill chunk (prefill_step_size) for the
    prompt batch, so a long prompt delays running answers by one chunk at a time.
    Segment boundaries come back as PromptProcessingBatch.Response with
    end_of_segment=True, and extract_cache() (generate.py:1727) returns the cache
    holding exactly the tokens processed so far.
  * A finished sequence returns its cache and the tokens in it
    (GenerationBatch.next, generate.py:1468): the prompt plus every generated
    token, because each token is fed back before it is returned.
  * Batching needs a `merge` on every cache layer (_merge_caches,
    generate.py:826). KVCache, RotatingKVCache, ArraysCache and CacheList have
    one; QuantizedKVCache, ChunkedKVCache and ConcatenateKVCache do not
    (models/cache.py). So with --kv-bits this engine serves one sequence at a
    time, exactly like mlx_lm.server (server.py:768-773).
  * speculative_generate_step (generate.py:485) needs a trimmable cache, and
    rewinds both caches in its `finally` (generate.py:664) to "every emitted
    token except the last one" after a round's bonus token. ArraysCache (the
    recurrent state of DeltaNet / Mamba layers) is not trimmable
    (_BaseCache.is_trimmable, models/cache.py:164), so hybrid models never draft.
  * mlx_lm.server marks a model with a draft as non-batchable (server.py:460).
    Here the draft runs while exactly one request is active; when a second one
    arrives, the running request is folded into the batch at its next draft-round
    boundary (its cache is precisely the BatchGenerator insert contract there).
  * MTP (multi-token prediction heads): not supported, because mlx-lm 0.32.2 has
    no MTP decoding path (nothing in generate.py or server.py) and the model
    files drop the MTP weights at load (models/qwen3_5.py:354,
    models/qwen3_next.py:489, models/mimo.py:189).

Prefix cache (item 5): a token-level radix tree over prompt token ids. An entry
whose cache layers are all trimmable (attention-only models) serves ANY prompt
sharing a prefix with it: the copy is trimmed back to the common length. A
non-trimmable entry (hybrid model: ArraysCache next to KVCache) can only resume
from exactly its own length, so for those the engine records snapshots at chat
segment boundaries while it prefills: the end of the system prompt, the end of
each completed assistant turn, and the end of the prompt. A follow-up turn then
resumes from the nearest snapshot instead of from token zero. Entries are LRU by
bytes (not count) and optionally spill to safetensors files under
--prefix-cache-dir, keyed by model + adapter fingerprint + KV settings, so a
retrained adapter can never be served a stale cache.

Copies are cheap and safe: mx.array values are immutable to other holders. A
KVCache update is `keys[..., a:b, :] = new`, which rebinds that cache object's
array; a copy holding `keys[...]` keeps the old buffer, and MLX only updates a
buffer in place when nothing else references it. So a restored entry is a set of
lazy views, and the stored entry is never mutated by the request that reuses it
(tests/test_mlx_engine.py checks restored-prefix greedy output equals
from-scratch greedy output).
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import copy
import hashlib
import importlib.util
import json
import logging
import os
import queue
import re
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .sysutil import *  # noqa: F401,F403
from .backends import *  # noqa: F401,F403


ENGINE_BACKEND = "mlx-engine"
# The supervisor's watchdog matches this line (BackendSpec.fatal_patterns), so
# a dead generation thread is restarted instead of serving 503s forever.
ENGINE_FATAL_LINE = "mlx-engine: generation thread died"
# Entries recorded at chat-segment boundaries (only for non-trimmable caches).
_ENG_SNAPSHOT_KINDS = frozenset({"system", "turn", "prompt"})
_ENG_TTFT_WINDOW = 512
_ENG_RECENT_WINDOW_S = 10.0
# Reply tokens a waiting request is assumed to need when its KV cost is
# estimated for admission [inference: agent replies are rarely longer, and a
# full max_tokens budget would make every admission check fail on 8GB].
_ENG_REPLY_ESTIMATE = 1024


class EngineBusy(Exception):
    """The engine cannot take the request now (queue full, loading, stopped): 503."""


class EngineBadRequest(ValueError):
    """The request body cannot be served: 400."""


@dataclass
class EngineOptions:
    """Everything the CLI (or a test) configures. Negative sizes mean "auto"."""

    model: str = ""
    adapter_path: str = ""
    draft_model: str = ""
    num_draft_tokens: int = 3
    kv_bits: int = 0
    kv_group_size: int = 64
    quantized_kv_start: int = 5000
    max_kv_size: int = 0
    max_batch: int = 4
    prefill_batch: int = 0
    prefill_step_size: int = 512
    prefix_cache_bytes: int = -1
    prefix_cache_dir: str = ""
    prefix_cache_disk_bytes: int = 4 << 30
    max_queue: int = 0
    kv_budget_bytes: int = -1
    default_max_tokens: int = 512
    default_temperature: float = 0.0
    served_name: str = ""
    max_turn_snapshots: int = 4


def _eng_auto_prefix_bytes() -> int:
    # About a tenth of RAM, between 256 MiB and 2 GiB: on 8GB that is ~800 MB,
    # roughly 22k tokens of a 3B model's KV, which holds the six agent lanes'
    # system prefixes plus a few live conversations.
    ram_gb = float(TOTAL_RAM_GB or 8)
    return int(min(2 << 30, max(256 << 20, ram_gb * 0.10 * (1 << 30))))


def _eng_decode_cap(ram_gb: float) -> int:
    # Same tiers as mlx-lm-turbo: every decode slot is a live KV sequence in the
    # unified memory that also holds the weights.
    if ram_gb < 12:
        return 3
    if ram_gb < 24:
        return 4
    if ram_gb < 48:
        return 8
    return 16


def _eng_prefill_parallel(ram_gb: float) -> int:
    # Prompts prefilled together multiply the activation peak that fails the
    # Metal command buffer on small machines (core._default_prefill_step), so
    # parallel prefill stays off below 16 GB, as on mlx-lm-turbo.
    if ram_gb < 16:
        return 1
    if ram_gb < 32:
        return 2
    if ram_gb < 64:
        return 4
    return 8


# --------------------------------------------------------------------------- #
# Cache helpers (duck-typed over mlx-lm's cache classes)
# --------------------------------------------------------------------------- #
def _eng_is_cache_obj(obj: Any) -> bool:
    return hasattr(obj, "is_trimmable") and hasattr(obj, "__dict__")


def _eng_dup(obj: Any) -> Any:
    """A copy of a cache structure whose arrays are fresh lazy views."""
    import mlx.core as mx

    if isinstance(obj, mx.array):
        return obj[...]
    if isinstance(obj, list):
        return [_eng_dup(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_eng_dup(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _eng_dup(v) for k, v in obj.items()}
    if _eng_is_cache_obj(obj):
        new = copy.copy(obj)
        new.__dict__ = {k: _eng_dup(v) for k, v in obj.__dict__.items()}
        return new
    return obj


def _eng_cache_copy(cache: list) -> list:
    return [_eng_dup(c) for c in cache]


def _eng_cache_nbytes(cache: list) -> int:
    total = 0
    for c in cache or []:
        try:
            total += int(c.nbytes)
        except Exception:  # noqa: BLE001
            pass
    return total


def _eng_cache_trimmable(cache: list) -> bool:
    try:
        return bool(cache) and all(c.is_trimmable() for c in cache)
    except Exception:  # noqa: BLE001
        return False


def _eng_cache_trim(cache: list, n: int) -> None:
    if n > 0:
        for c in cache:
            c.trim(n)


def _eng_cache_len(cache: list) -> int | None:
    """Tokens a cache holds, from the first layer that counts them."""
    for c in cache or []:
        layers = getattr(c, "caches", None)
        if layers:
            got = _eng_cache_len(list(layers))
            if got is not None:
                return got
        offset = getattr(c, "offset", None)
        if isinstance(offset, int):
            return offset
    return None


def _eng_flat_layers(cache: list) -> list:
    out = []
    for c in cache:
        inner = getattr(c, "caches", None)
        out.extend(_eng_flat_layers(list(inner)) if inner else [c])
    return out


def _eng_lcp(a: list, b: list) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _eng_percentiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "p50": None, "p95": None}
    xs = sorted(values)

    def pick(q: float) -> float:
        return round(xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))] * 1000.0, 1)

    return {"n": len(xs), "p50": pick(0.50), "p95": pick(0.95)}


# --------------------------------------------------------------------------- #
# Prefix cache: token radix tree, LRU by bytes, optional disk spill
# --------------------------------------------------------------------------- #
class _EngNode:
    __slots__ = ("edge", "kids", "entry", "parent", "depth")

    def __init__(self, edge: tuple, parent: "_EngNode | None", depth: int) -> None:
        self.edge = edge
        self.kids: dict[int, _EngNode] = {}
        self.entry: _EngEntry | None = None
        self.parent = parent
        self.depth = depth


class _EngEntry:
    __slots__ = ("node", "cache", "nbytes", "trimmable", "kind", "disk", "pending")

    def __init__(self, node: _EngNode, cache: list | None, nbytes: int, trimmable: bool,
                 kind: str) -> None:
        self.node = node
        self.cache = cache          # in memory (None when only on disk)
        self.nbytes = nbytes
        self.trimmable = trimmable
        self.kind = kind
        self.disk = ""              # file path once written (or being written)
        self.pending = None         # cache handed to the writer, until it is on disk


class PrefixCache:
    """Snapshots of model state keyed by the exact token prefix they hold.

    lookup() returns a private copy (trimmed when the best entry is longer than
    the shared prefix) plus the number of prompt tokens it covers; it always
    leaves at least one token to process, because the next token's logits come
    from running the model on it. insert() stores a cache by its tokens. All
    methods are thread-safe; the generation thread is the only caller that
    touches mlx arrays, apart from the disk writer saving finished entries.
    """

    def __init__(self, max_bytes: int, *, name: str = "main", min_disk_tokens: int = 32) -> None:
        self.name = name
        self.max_bytes = max(0, int(max_bytes))
        self.min_disk_tokens = max(1, int(min_disk_tokens))
        self._root = _EngNode((), None, 0)
        self._lock = threading.RLock()
        self._mem: collections.OrderedDict[int, _EngEntry] = collections.OrderedDict()
        self._disk: collections.OrderedDict[int, _EngEntry] = collections.OrderedDict()
        self.bytes = 0
        self.disk_bytes = 0
        self.disk_dir: Path | None = None
        self.disk_max_bytes = 0
        self._index: dict[str, dict] = {}
        self._wq: queue.Queue | None = None
        self._writer: threading.Thread | None = None
        self.c: collections.Counter = collections.Counter()

    # ---- trie ----------------------------------------------------------- #
    def _path(self, tokens: list) -> _EngNode:
        node, i, n = self._root, 0, len(tokens)
        while i < n:
            t = tokens[i]
            child = node.kids.get(t)
            if child is None:
                new = _EngNode(tuple(tokens[i:]), node, n)
                node.kids[t] = new
                return new
            edge = child.edge
            lim = min(len(edge), n - i)
            k = 1
            while k < lim and edge[k] == tokens[i + k]:
                k += 1
            if k == len(edge):
                node, i = child, i + k
                continue
            mid = _EngNode(edge[:k], node, node.depth + k)
            node.kids[t] = mid
            child.edge = edge[k:]
            child.parent = mid
            mid.kids[child.edge[0]] = child
            i += k
            if i == n:
                return mid
            new = _EngNode(tuple(tokens[i:]), mid, n)
            mid.kids[tokens[i]] = new
            return new
        return node

    def _walk(self, tokens: list) -> tuple[list, _EngNode, _EngNode | None, int]:
        """(entry nodes fully on the path, deepest full node, partial child, its match)."""
        path: list[_EngNode] = []
        node, i, n = self._root, 0, len(tokens)
        while i < n:
            child = node.kids.get(tokens[i])
            if child is None:
                return path, node, None, 0
            edge = child.edge
            lim = min(len(edge), n - i)
            k = 1
            while k < lim and edge[k] == tokens[i + k]:
                k += 1
            if k < len(edge):
                return path, node, child, k
            node, i = child, i + k
            if node.entry is not None:
                path.append(node)
        return path, node, None, 0

    @staticmethod
    def _first_trimmable(start: _EngNode, include_self: bool, limit: int = 256) -> _EngEntry | None:
        stack, seen = [start], 0
        while stack and seen < limit:
            node = stack.pop()
            seen += 1
            e = node.entry
            if e is not None and e.trimmable and (include_self or node is not start):
                return e
            stack.extend(node.kids.values())
        return None

    def _best(self, tokens: list, max_len: int) -> tuple[_EngEntry | None, int]:
        path, node, child, k = self._walk(tokens)
        best, best_n = None, 0
        for nd in path:
            e = nd.entry
            if nd.depth <= max_len:
                cand = nd.depth
            elif e.trimmable:
                cand = max_len
            else:
                continue
            if cand > best_n:
                best, best_n = e, cand
        if child is not None:
            e = self._first_trimmable(child, True)
            cand = min(node.depth + k, max_len)
            if e is not None and cand > best_n:
                best, best_n = e, cand
        e = self._first_trimmable(node, False)
        cand = min(node.depth, max_len)
        if e is not None and cand > best_n:
            best, best_n = e, cand
        return best, best_n

    def _tokens_of(self, node: _EngNode) -> list[int]:
        parts = []
        while node is not None and node.parent is not None:
            parts.append(node.edge)
            node = node.parent
        return [t for edge in reversed(parts) for t in edge]

    def _prune(self, node: _EngNode) -> None:
        while node is not self._root and node.entry is None and not node.kids:
            parent = node.parent
            if parent is None:
                break
            if parent.kids.get(node.edge[0]) is node:
                del parent.kids[node.edge[0]]
            node = parent

    # ---- bookkeeping ---------------------------------------------------- #
    def _touch(self, e: _EngEntry) -> None:
        key = id(e)
        if key in self._mem:
            self._mem.move_to_end(key)
        elif key in self._disk:
            self._disk.move_to_end(key)

    def _drop(self, e: _EngEntry, prune: bool = True) -> None:
        key = id(e)
        if self._mem.pop(key, None) is not None:
            self.bytes -= e.nbytes
        self._disk.pop(key, None)
        if e.disk:
            self.disk_bytes -= e.nbytes
            self._submit(("delete", e.disk))
            e.disk = ""
        e.cache = None
        e.pending = None
        if e.node.entry is e:
            e.node.entry = None
            if prune:
                self._prune(e.node)

    def _evict(self) -> None:
        while self.bytes > self.max_bytes and self._mem:
            oldest = next(iter(self._mem.values()))
            self.c["evictions"] += 1
            self._spill(oldest)

    def _spill(self, e: _EngEntry) -> None:
        """Move an entry out of memory: to disk when enabled and worth it, else drop it."""
        worth = (self.disk_dir is not None and e.node.depth >= self.min_disk_tokens
                 and e.nbytes <= self.disk_max_bytes)
        if not worth:
            self._drop(e)
            return
        self._mem.pop(id(e), None)
        self.bytes -= e.nbytes
        if not e.disk:
            self._schedule_write(e, e.cache)
        e.cache = None
        self._disk[id(e)] = e
        self.c["spills"] += 1
        while self.disk_bytes > self.disk_max_bytes and self._disk:
            self._drop(next(iter(self._disk.values())))

    def release(self, n_bytes: int) -> int:
        """Give back at least n_bytes of memory (LRU first). Returns bytes freed."""
        freed = 0
        with self._lock:
            while freed < n_bytes and self._mem:
                oldest = next(iter(self._mem.values()))
                freed += oldest.nbytes
                self.c["released"] += 1
                self._spill(oldest)
        return freed

    # ---- public API ----------------------------------------------------- #
    def peek(self, tokens: list) -> int:
        """Prompt tokens a lookup would reuse, without copying anything."""
        with self._lock:
            if len(tokens) < 2:
                return 0
            return self._best(tokens, len(tokens) - 1)[1]

    def lookup(self, tokens: list, max_len: int | None = None) -> tuple[list | None, int]:
        with self._lock:
            n_tok = len(tokens)
            limit = n_tok - 1 if max_len is None else min(int(max_len), n_tok - 1)
            self.c["lookups"] += 1
            self.c["lookup_tokens"] += n_tok
            e, n = self._best(tokens, limit) if limit > 0 else (None, 0)
            if e is None:
                self.c["misses"] += 1
                return None, 0
            src = e.cache if e.cache is not None else e.pending
            if src is None:
                src = self._disk_load(e)
                if src is None:
                    self.c["misses"] += 1
                    return None, 0
            cache = _eng_cache_copy(src)
            depth = e.node.depth
            if depth > n:
                _eng_cache_trim(cache, depth - n)
                self.c["trimmed_hits"] += 1
            self._touch(e)
            self.c["hits"] += 1
            self.c["hit_tokens"] += n
            return cache, n

    def insert(self, tokens: list, cache: list | None, kind: str = "final") -> bool:
        if not tokens or not cache:
            return False
        nbytes = _eng_cache_nbytes(cache)
        trimmable = _eng_cache_trimmable(cache)
        with self._lock:
            if self.max_bytes <= 0 or nbytes > self.max_bytes:
                self.c["too_large"] += 1
                return False
            if trimmable:
                # A longer trimmable entry already serves this prefix by trimming.
                path, node, child, k = self._walk(tokens)
                longer = None
                if node.depth == len(tokens):
                    longer = self._first_trimmable(node, False)
                elif child is not None and node.depth + k == len(tokens):
                    longer = self._first_trimmable(child, True)
                if longer is not None:
                    self._touch(longer)
                    self.c["redundant"] += 1
                    return False
            node = self._path(list(tokens))
            if node.entry is not None:
                self._drop(node.entry, prune=False)
            e = _EngEntry(node, cache, nbytes, trimmable, kind)
            node.entry = e
            self._mem[id(e)] = e
            self.bytes += nbytes
            self.c["inserts"] += 1
            if kind in _ENG_SNAPSHOT_KINDS:
                self.c["snapshots"] += 1
            if trimmable:
                # Its own prefixes are now redundant: this entry serves them.
                stale = []
                up = node.parent
                while up is not None:
                    if up.entry is not None and up.entry.trimmable:
                        stale.append(up.entry)
                    up = up.parent
                for old in stale:
                    self.c["deduped"] += 1
                    self._drop(old, prune=False)
            self._evict()
            return node.entry is e

    def clear(self) -> None:
        with self._lock:
            for e in list(self._mem.values()) + list(self._disk.values()):
                self._drop(e, prune=False)
            self._root = _EngNode((), None, 0)

    def entries(self) -> list[dict]:
        """(depth, kind, where) per entry, for tests and the stats endpoint."""
        with self._lock:
            out = []
            for where, table in (("memory", self._mem), ("disk", self._disk)):
                for e in table.values():
                    out.append({"tokens": e.node.depth, "kind": e.kind, "where": where,
                                "bytes": e.nbytes, "trimmable": e.trimmable})
            return out

    def stats(self) -> dict:
        with self._lock:
            c = self.c
            lookups = c["lookups"]
            snaps = sum(1 for e in self._mem.values() if e.kind in _ENG_SNAPSHOT_KINDS)
            return {
                "entries": len(self._mem),
                "bytes": self.bytes,
                "max_bytes": self.max_bytes,
                "snapshots": snaps,
                "snapshots_taken": c["snapshots"],
                "lookups": lookups,
                "hits": c["hits"],
                "misses": c["misses"],
                "hit_rate": round(c["hits"] / lookups, 3) if lookups else None,
                "hit_tokens": c["hit_tokens"],
                "lookup_tokens": c["lookup_tokens"],
                "trimmed_hits": c["trimmed_hits"],
                "inserts": c["inserts"],
                "evictions": c["evictions"],
                "deduped": c["deduped"],
                "redundant": c["redundant"],
                "too_large": c["too_large"],
                "disk": {
                    "enabled": self.disk_dir is not None,
                    "dir": str(self.disk_dir) if self.disk_dir else "",
                    "entries": len(self._disk),
                    "bytes": self.disk_bytes,
                    "max_bytes": self.disk_max_bytes,
                    "writes": c["disk_writes"],
                    "loads": c["disk_loads"],
                    "errors": c["disk_errors"],
                    "spills": c["spills"],
                },
            }

    # ---- disk ----------------------------------------------------------- #
    def enable_disk(self, directory: Path | str, max_bytes: int) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self.disk_dir = d
            self.disk_max_bytes = max(0, int(max_bytes))
            if self._writer is None:
                self._wq = queue.Queue()
                self._writer = threading.Thread(target=self._writer_loop, daemon=True,
                                                name=f"prefix-cache-writer-{self.name}")
                self._writer.start()
            self._load_index()

    def _load_index(self) -> None:
        assert self.disk_dir is not None
        for stray in self.disk_dir.glob("*.tmp.safetensors"):
            try:
                stray.unlink()
            except OSError:
                pass
        try:
            data = json.loads((self.disk_dir / "index.json").read_text())
        except (OSError, ValueError):
            data = {}
        for fname, info in (data.get("entries") or {}).items():
            path = self.disk_dir / fname
            tokens = info.get("tokens")
            if (not path.is_file() or not isinstance(tokens, list) or not tokens
                    or not all(isinstance(t, int) for t in tokens)):
                continue
            node = self._path(tokens)
            if node.entry is not None:
                continue
            e = _EngEntry(node, None, int(info.get("nbytes") or path.stat().st_size),
                          bool(info.get("trimmable")), str(info.get("kind") or "final"))
            e.disk = str(path)
            node.entry = e
            self._disk[id(e)] = e
            self.disk_bytes += e.nbytes
            self._index[fname] = info
        while self.disk_bytes > self.disk_max_bytes and self._disk:
            self._drop(next(iter(self._disk.values())))

    def _schedule_write(self, e: _EngEntry, cache: list) -> None:
        import mlx.core as mx

        assert self.disk_dir is not None
        # Evaluate here, on the generation thread: a lazy array still belongs to
        # that thread's stream, and the writer thread cannot evaluate it
        # ("There is no Stream(gpu, N) in current thread").
        mx.eval([c.state for c in cache])
        tokens = self._tokens_of(e.node)
        digest = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:32]
        e.disk = str(self.disk_dir / f"{digest}.safetensors")
        e.pending = cache
        self.disk_bytes += e.nbytes
        self.c["disk_writes"] += 1
        self._submit(("save", e, cache, tokens))

    def _submit(self, job: tuple) -> None:
        if self._wq is not None:
            self._wq.put(job)

    def _writer_loop(self) -> None:
        from mlx_lm.models.cache import save_prompt_cache

        assert self._wq is not None
        while True:
            job = self._wq.get()
            if job is None:
                self._write_index()
                self._wq.task_done()
                return
            try:
                if job[0] == "save":
                    _, e, cache, tokens = job
                    path = Path(e.disk) if e.disk else None
                    if path is not None:
                        tmp = path.with_name(path.stem + ".tmp.safetensors")
                        save_prompt_cache(str(tmp), cache, {"kind": e.kind})
                        os.replace(tmp, path)
                        with self._lock:
                            if e.pending is cache:
                                e.pending = None
                            if e.disk == str(path):
                                self._index[path.name] = {
                                    "tokens": tokens, "kind": e.kind, "nbytes": e.nbytes,
                                    "trimmable": e.trimmable}
                            else:   # dropped while it was being written
                                path.unlink(missing_ok=True)
                elif job[0] == "delete":
                    path = Path(job[1])
                    path.unlink(missing_ok=True)
                    with self._lock:
                        self._index.pop(path.name, None)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self.c["disk_errors"] += 1
                    if job[0] == "save":
                        e = job[1]
                        e.pending = None
                        if e.cache is None:
                            self._drop(e)
                log(f"mlx-engine prefix cache: disk {job[0]} failed ({exc}).", logging.WARNING)
            if self._wq.empty():
                self._write_index()
            self._wq.task_done()

    def _write_index(self) -> None:
        if self.disk_dir is None:
            return
        with self._lock:
            text = json.dumps({"version": 1, "entries": self._index})
        tmp = self.disk_dir / "index.json.tmp"
        try:
            tmp.write_text(text)
            os.replace(tmp, self.disk_dir / "index.json")
        except OSError as exc:
            log(f"mlx-engine prefix cache: index write failed ({exc}).", logging.WARNING)

    def _disk_load(self, e: _EngEntry) -> list | None:
        from mlx_lm.models.cache import load_prompt_cache

        try:
            cache = load_prompt_cache(e.disk)
        except Exception as exc:  # noqa: BLE001
            self.c["disk_errors"] += 1
            log(f"mlx-engine prefix cache: cannot load {e.disk} ({exc}); dropped.", logging.WARNING)
            self._drop(e)
            return None
        self.c["disk_loads"] += 1
        self._disk.pop(id(e), None)
        e.cache = cache
        self._mem[id(e)] = e
        self.bytes += e.nbytes
        self._evict()
        return cache

    def flush(self, timeout: float = 10.0) -> None:
        """Write every in-memory entry to disk (shutdown), then stop the writer."""
        if self.disk_dir is None or self._wq is None:
            return
        with self._lock:
            for e in list(self._mem.values()):
                if (not e.disk and e.node.depth >= self.min_disk_tokens
                        and self.disk_bytes + e.nbytes <= self.disk_max_bytes):
                    self._schedule_write(e, e.cache)
        self._wq.put(None)
        if self._writer is not None:
            self._writer.join(timeout)
        self._writer = None
        self._wq = None


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class EngineRequest:
    """One chat completion travelling from the HTTP layer to the generation thread.

    The generation thread calls `emit` with ("delta", text), ("done", reason,
    usage) or ("error", message); the HTTP layer bridges that into its event loop.
    """

    def __init__(self, *, prompt: list[int], boundaries: list, max_tokens: int,
                 sampler: Any, processors: list, stops: list[str], speculative_ok: bool) -> None:
        self.id = "chatcmpl-" + uuid.uuid4().hex[:24]
        self.created = int(time.time())
        self.prompt = prompt
        self.boundaries = boundaries
        self.max_tokens = max_tokens
        self.sampler = sampler
        self.processors = processors
        self.stops = stops
        self.speculative_ok = speculative_ok
        self.emit: Callable[[tuple], None] = lambda ev: None
        self.cancelled = threading.Event()
        self.done = threading.Event()
        self.tokens: list[int] = []
        self.n_cached = 0
        self.finish_reason: str | None = None
        self.usage: dict | None = None
        self.mode = ""
        self.t_submit = 0.0
        self.t_first: float | None = None
        self.detok: Any = None
        self.hold = ""
        self.stopped_by_string = False
        self.kinds: collections.deque = collections.deque()
        self.admission_waited = False
        self._wake: threading.Event | None = None

    def cancel(self) -> None:
        self.cancelled.set()
        if self._wake is not None:
            self._wake.set()


class _EngJob:
    """The single sequence the speculative / sequential path is decoding."""

    def __init__(self, req: EngineRequest, gen: Any, cache: list, dcache: list | None) -> None:
        self.req = req
        self.gen = gen
        self.cache = cache
        self.dcache = dcache
        self.draft = dcache is not None
        self.emitted: list[int] = []
        self.first = True
        self.migrate = False
        self.pos = 0                # prompt tokens in `cache` (prefill progress)
        self.dpos = 0               # prompt tokens in `dcache`
        self.marks: list = []       # snapshot points still ahead (hybrid models)


_ENG_SAMPLERS: dict = {}


def _eng_make_sampler(temp: float, top_p: float, min_p: float, top_k: int, seed: int | None) -> Any:
    """None for greedy (the generator's shared sampler keeps the batch on one call)."""
    if temp <= 0:
        return None
    if seed is None:
        from mlx_lm.sample_utils import make_sampler

        key = (temp, top_p, min_p, top_k)
        sampler = _ENG_SAMPLERS.get(key)
        if sampler is None:
            if len(_ENG_SAMPLERS) > 64:
                _ENG_SAMPLERS.clear()
            sampler = _ENG_SAMPLERS[key] = make_sampler(temp, top_p=top_p, min_p=min_p, top_k=top_k)
        return sampler
    # A seed must not depend on what else is in the batch, so a seeded request
    # draws from its own key chain instead of the process-wide generator
    # (mlx_lm.server serves seeded requests one at a time for that reason).
    import mlx.core as mx
    from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p

    state = {"key": mx.random.key(int(seed) % (1 << 63))}

    def sampler(logprobs: Any) -> Any:
        if 0 < top_p < 1.0:
            logprobs = apply_top_p(logprobs, top_p)
        if min_p > 0:
            logprobs = apply_min_p(logprobs, min_p, 1)
        if top_k > 0:
            logprobs = apply_top_k(logprobs, top_k)
        key, sub = mx.random.split(state["key"])
        state["key"] = key
        return mx.random.categorical(logprobs * (1.0 / temp), key=sub)

    return sampler


def _eng_render(tok: Any, messages: list, *, tools: Any = None, add_generation_prompt: bool = True,
                template_kwargs: dict | None = None) -> list[int]:
    kw = dict(template_kwargs or {})
    if tools:
        kw["tools"] = tools
    if getattr(tok, "has_chat_template", True) is False:
        text = "".join(f"{m.get('role')}: {m.get('content') or ''}\n" for m in messages)
        if add_generation_prompt:
            text += "assistant: "
        return [int(t) for t in tok.encode(text)]
    out = tok.apply_chat_template(messages, add_generation_prompt=add_generation_prompt,
                                  tokenize=True, **kw)
    if hasattr(out, "keys"):
        out = out["input_ids"]
    return [int(t) for t in out]


def _eng_boundaries(tok: Any, messages: list, prompt: list[int], *, tools: Any, template_kwargs: dict,
                    floor: int, max_turns: int) -> list[tuple[int, str]]:
    """Snapshot points (position, kind) past `floor`: system end, assistant-turn ends.

    Each position is the common prefix of the prompt and the conversation
    rendered up to that point, so a template that re-renders history
    differently (Qwen3 dropping old reasoning) only moves a boundary earlier;
    the snapshot is taken while processing THIS prompt, so it always holds
    exactly prompt[:position].
    """
    n = len(prompt)
    found: dict[int, str] = {}
    turns = [k for k, m in enumerate(messages[:-1]) if m.get("role") == "assistant"]
    for k in reversed(turns[-max_turns:] if max_turns > 0 else []):
        try:
            toks = _eng_render(tok, messages[:k + 1], tools=tools, add_generation_prompt=False,
                               template_kwargs=template_kwargs)
        except Exception:  # noqa: BLE001
            continue
        b = _eng_lcp(toks, prompt)
        if b <= floor:
            break
        if b < n - 1:
            found.setdefault(b, "turn")
    n_sys = 0
    for m in messages:
        if m.get("role") not in ("system", "developer"):
            break
        n_sys += 1
    if 0 < n_sys < len(messages):
        try:
            toks = _eng_render(tok, messages[:n_sys] + [{"role": "user", "content": ""}],
                               tools=tools, add_generation_prompt=False,
                               template_kwargs=template_kwargs)
            b = _eng_lcp(toks, prompt)
            if floor < b < n - 1:
                found.setdefault(b, "system")
        except Exception:  # noqa: BLE001
            pass
    return sorted(found.items())


def _eng_segments(prompt: list[int], start: int, boundaries: list) -> tuple[list, list]:
    segments, kinds, pos = [], [], start
    for b, kind in boundaries:
        if pos < b < len(prompt) - 1:
            segments.append(prompt[pos:b])
            kinds.append(kind)
            pos = b
    segments.append(prompt[pos:])
    return segments, kinds


def _eng_prefill(model: Any, cache: list, tokens: list[int], step: int,
                 kv: dict | None = None) -> None:
    """Run tokens through the model in chunks, quantizing the KV like generate_step."""
    import mlx.core as mx
    from mlx_lm.generate import generation_stream, maybe_quantize_kv_cache

    step = max(1, int(step))
    bits = (kv or {}).get("kv_bits")
    with mx.stream(generation_stream):
        for i in range(0, len(tokens), step):
            model(mx.array(tokens[i:i + step])[None], cache=cache)
            if bits:
                maybe_quantize_kv_cache(cache, kv["quantized_kv_start"], kv["kv_group_size"], bits)
            mx.eval([c.state for c in cache])
    mx.clear_cache()


def _eng_adapter_fingerprint(adapter_path: str) -> str:
    if not adapter_path:
        return ""
    p = Path(adapter_path)
    files = sorted(p.glob("*.safetensors")) + [p / "adapter_config.json"] if p.is_dir() else [p]
    parts = []
    for f in files:
        try:
            st = f.stat()
        except OSError:
            continue
        parts.append(f"{f.name}:{st.st_size}:{st.st_mtime_ns}")
    return ";".join(parts)


def _eng_model_key(opts: EngineOptions) -> str:
    """Disk namespace: anything that changes what a cached state means is in it."""
    try:
        import mlx_lm
        version = getattr(mlx_lm, "__version__", "")
    except Exception:  # noqa: BLE001
        version = ""
    raw = json.dumps({
        "model": opts.model, "adapter": opts.adapter_path,
        "adapter_fp": _eng_adapter_fingerprint(opts.adapter_path),
        "kv_bits": opts.kv_bits, "kv_group_size": opts.kv_group_size,
        "quantized_kv_start": opts.quantized_kv_start, "max_kv_size": opts.max_kv_size,
        "mlx_lm": version,
    }, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# The engine: one generation thread owns the model
# --------------------------------------------------------------------------- #
class MLXEngine:
    """Scheduler + model owner. Thread-safe entry points: prepare, submit, stats, stop."""

    def __init__(self, opts: EngineOptions, *, model: Any = None, tokenizer: Any = None,
                 draft_model: Any = None) -> None:
        self.opts = opts
        self.model = model
        self.tokenizer = tokenizer
        self.draft = draft_model
        self.ready = threading.Event()
        self.dead = ""
        self.on_fatal: Callable[[], None] | None = None
        self._lock = threading.RLock()
        self._tok_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending: collections.deque[EngineRequest] = collections.deque()
        self._uid_req: dict[int, EngineRequest] = {}
        self._job: _EngJob | None = None
        self._bg: Any = None
        self.batchable = False
        self.trimmable = False
        self.spec_ok = False
        self.max_batch = max(1, int(opts.max_batch or 1))
        self.prefill_batch = max(1, min(self.max_batch, int(opts.prefill_batch or 0)
                                        or _eng_prefill_parallel(float(TOTAL_RAM_GB or 8))))
        self.max_queue = int(opts.max_queue) if opts.max_queue > 0 else max(8, 4 * self.max_batch)
        pbytes = opts.prefix_cache_bytes if opts.prefix_cache_bytes >= 0 else _eng_auto_prefix_bytes()
        self.prefix = PrefixCache(pbytes, name="main")
        self.draft_prefix = PrefixCache(pbytes // 4 if draft_model is not None or opts.draft_model
                                        else 0, name="draft")
        self.served_name = opts.served_name or opts.model or "default_model"
        self.model_key = ""
        self._eos: set[int] = set()
        self._kv_per_token = 0
        self._kv_fixed = 0
        self._kv_budget = 0
        self._mem_limit = 0
        self._errors: collections.deque[float] = collections.deque()
        self._st: collections.Counter = collections.Counter()
        self._ttft: collections.deque = collections.deque(maxlen=_ENG_TTFT_WINDOW)
        self._recent: collections.deque = collections.deque()
        self._seq_decode_time = 0.0
        self._seq_decode_tokens = 0
        self._seq_prefill_time = 0.0
        self._seq_prefill_tokens = 0
        self._peak_batch = 0
        self._occ_sum = 0
        self._occ_steps = 0
        self.started_at = time.time()

    # ---- lifecycle ------------------------------------------------------- #
    def start(self) -> "MLXEngine":
        self._thread = threading.Thread(target=self._run, name="mlx-engine-generation", daemon=True)
        self._thread.start()
        return self

    def wait_ready(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.time() + timeout
        while not self.ready.is_set():
            if self.dead:
                return False
            if deadline is not None and time.time() > deadline:
                return False
            self.ready.wait(0.05)
        return True

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)

    def _run(self) -> None:
        try:
            self._setup()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._fatal(f"model load failed: {exc!r}")
            return
        self.ready.set()
        while not self._stop.is_set():
            try:
                worked = self._tick()
            except Exception as exc:  # noqa: BLE001
                self._tick_failed(exc)
                worked = True
            if self.dead:
                return
            if not worked:
                self._wake.wait(0.05)
                self._wake.clear()
        self._shutdown()

    def _setup(self) -> None:
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm.models.cache import make_prompt_cache

        o = self.opts
        if self.model is None:
            from mlx_lm import load

            self.model, self.tokenizer = load(o.model, adapter_path=o.adapter_path or None)
            if o.draft_model:
                self.draft, dtok = load(o.draft_model)
                if getattr(dtok, "vocab_size", None) != getattr(self.tokenizer, "vocab_size", None):
                    log("mlx-engine: the draft model's vocabulary differs from the model's; "
                        "drafts will rarely be accepted.", logging.WARNING)
        tok = self.tokenizer
        eos = getattr(tok, "eos_token_ids", None) or []
        self._eos = {int(t) for t in eos}
        if not self._eos and getattr(tok, "eos_token_id", None) is not None:
            self._eos = {int(tok.eos_token_id)}
        probe = make_prompt_cache(self.model, o.max_kv_size or None)
        self.trimmable = _eng_cache_trimmable(probe)
        self.batchable = o.kv_bits <= 0 and all(hasattr(c, "merge") for c in probe)
        if self.draft is not None:
            self.spec_ok = self.trimmable and _eng_cache_trimmable(make_prompt_cache(self.draft))
            if not self.spec_ok:
                log("mlx-engine: this model's cache cannot be trimmed (hybrid recurrent layers), "
                    "and speculative decoding rewinds it; the draft model is not used.",
                    logging.WARNING)
                self.draft = None
                self.draft_prefix.max_bytes = 0
        self._measure_kv(probe)
        weights = sum(v.nbytes for _, v in tree_flatten(self.model.parameters()))
        if self.draft is not None:
            weights += sum(v.nbytes for _, v in tree_flatten(self.draft.parameters()))
        self._set_budget(weights)
        if self.batchable:
            self._bg = self._new_batch_generator()
        else:
            try:
                from mlx_lm.generate import maybe_set_recommended_wired_limit
                maybe_set_recommended_wired_limit()
            except Exception:  # noqa: BLE001
                pass
        self.model_key = _eng_model_key(o)
        if o.prefix_cache_dir and self.prefix.max_bytes > 0:
            self.prefix.enable_disk(Path(o.prefix_cache_dir).expanduser() / self.model_key,
                                    o.prefix_cache_disk_bytes)
        mx.clear_cache()
        log(f"mlx-engine ready: {self.served_name} "
            f"(batching {'on, up to ' + str(self.max_batch) if self.batchable else 'off'}"
            f", speculative {'on' if self.spec_ok else 'off'}"
            f", kv_bits {o.kv_bits or 'off'}, cache {'trimmable' if self.trimmable else 'hybrid/snapshots'}"
            f", prefix cache {self.prefix.max_bytes >> 20} MiB)")
        log_event(get_logger("engine"), logging.INFO, "engine.ready", model=self.served_name,
                  batching=self.batchable, speculative=self.spec_ok, kv_bits=o.kv_bits,
                  trimmable=self.trimmable, prefix_cache_bytes=self.prefix.max_bytes)

    def _new_batch_generator(self) -> Any:
        from mlx_lm.generate import BatchGenerator

        return BatchGenerator(
            self.model,
            max_tokens=self.opts.default_max_tokens,
            stop_tokens=[[t] for t in sorted(self._eos)] or None,
            completion_batch_size=self.max_batch,
            prefill_batch_size=self.prefill_batch,
            prefill_step_size=max(1, self.opts.prefill_step_size),
            max_kv_size=self.opts.max_kv_size or None,
        )

    def _measure_kv(self, cache: list) -> None:
        """KV bytes per token and fixed (recurrent) bytes per sequence, measured once."""
        import mlx.core as mx

        try:
            self.model(mx.array([[0]]), cache=cache)
            mx.eval([c.state for c in cache])
            per_tok, fixed = 0.0, 0
            for c in _eng_flat_layers(cache):
                keys = getattr(c, "keys", None)
                if isinstance(keys, (tuple, list)):
                    keys = keys[0] if keys else None
                shape = getattr(keys, "shape", None)
                if shape is not None and len(shape) >= 3 and shape[2]:
                    per_tok += c.nbytes / shape[2]
                else:
                    fixed += int(getattr(c, "nbytes", 0) or 0)
            self._kv_per_token, self._kv_fixed = int(per_tok), fixed
        except Exception as exc:  # noqa: BLE001
            log(f"mlx-engine: could not measure KV bytes per token ({exc}); "
                "memory-aware admission estimates are off.", logging.WARNING)

    def _set_budget(self, weight_bytes: int) -> None:
        import mlx.core as mx

        b = int(self.opts.kv_budget_bytes)
        if b > 0:
            self._kv_budget, self._mem_limit = b, 0
            return
        if b == 0:
            self._kv_budget, self._mem_limit = 0, 0
            return
        try:
            rec = int(mx.device_info().get("max_recommended_working_set_size") or 0)
        except Exception:  # noqa: BLE001
            rec = 0
        if rec <= 0:
            return
        # What is left for live sequences once the weights, the prefix cache and
        # some headroom for activations and the Metal buffer cache are paid for.
        self._kv_budget = max(self._kv_per_token * 4096,
                              rec - weight_bytes - self.prefix.max_bytes - (512 << 20))
        self._mem_limit = int(rec * 0.92)

    def _shutdown(self) -> None:
        for req in self._all_requests():
            self._fail(req, "engine stopping")
        self._uid_req.clear()
        self._job = None
        try:
            self.prefix.flush()
        except Exception as exc:  # noqa: BLE001
            log(f"mlx-engine: prefix cache flush failed ({exc}).", logging.WARNING)
        if self._bg is not None:
            try:
                self._bg.close()
            except Exception:  # noqa: BLE001
                pass

    def _all_requests(self) -> list[EngineRequest]:
        with self._lock:
            reqs = list(self._pending)
            self._pending.clear()
        reqs.extend(self._uid_req.values())
        if self._job is not None:
            reqs.append(self._job.req)
        return reqs

    def _fatal(self, message: str) -> None:
        log(f"{ENGINE_FATAL_LINE}: {message}", logging.ERROR)
        log_event(get_logger("engine"), logging.ERROR, "engine.fatal", error=message)
        self.dead = message
        for req in self._all_requests():
            self._fail(req, message)
        self._uid_req.clear()
        self._job = None
        self._stop.set()
        if self.on_fatal is not None:
            try:
                self.on_fatal()
            except Exception:  # noqa: BLE001
                pass

    def _tick_failed(self, exc: Exception) -> None:
        """A failed step fails the requests in flight and rebuilds the batch."""
        traceback.print_exc()
        message = redact_text(f"{type(exc).__name__}: {exc}")
        log(f"mlx-engine: generation step failed ({message}); failing the requests in "
            "flight and resetting the batch.", logging.ERROR)
        for req in list(self._uid_req.values()):
            self._fail(req, message)
        self._uid_req.clear()
        if self._job is not None:
            self._fail(self._job.req, message)
            self._job = None
        now = time.time()
        self._errors.append(now)
        while self._errors and now - self._errors[0] > 60:
            self._errors.popleft()
        if len(self._errors) >= 3:
            self._fatal(f"3 failed generation steps within a minute (last: {message})")
            return
        if self.batchable:
            try:
                if self._bg is not None:
                    self._bg.close()
            except Exception:  # noqa: BLE001
                pass
            self._bg = self._new_batch_generator()

    # ---- request intake -------------------------------------------------- #
    def accepted_model_names(self) -> set[str]:
        names = {"default_model", self.served_name}
        if self.opts.model:
            names.add(self.opts.model)
            names.add(self.opts.model.rstrip("/").rsplit("/", 1)[-1])
        return names

    def prepare(self, body: dict) -> EngineRequest:
        """Validate and tokenize an OpenAI chat body. Raises EngineBadRequest/EngineBusy."""
        if self.dead:
            raise EngineBusy(f"engine stopped: {self.dead}")
        if not self.ready.is_set():
            raise EngineBusy("model is loading")
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise EngineBadRequest("messages must be a non-empty list")
        messages = []
        for m in msgs:
            if not isinstance(m, dict) or not isinstance(m.get("role"), str):
                raise EngineBadRequest("every message needs a string role")
            content = m.get("content")
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, str):
                        parts.append(p)
                    elif isinstance(p, dict) and p.get("type") in ("text", "input_text"):
                        parts.append(str(p.get("text") or ""))
                    else:
                        raise EngineBadRequest("mlx-engine serves text only (no image or audio parts)")
                content = "".join(parts)
            elif content is None:
                content = ""
            elif not isinstance(content, str):
                content = str(content)
            nm = dict(m)
            nm["content"] = content
            messages.append(nm)
        if body.get("n", 1) not in (1, None):
            raise EngineBadRequest("n > 1 is not supported")

        def num(name: str, default: Any, kind: type, lo: float | None = None,
                hi: float | None = None) -> Any:
            value = body.get(name, default)
            if value is None:
                return default
            try:
                value = kind(value)
            except (TypeError, ValueError):
                raise EngineBadRequest(f"{name} must be a number") from None
            if (lo is not None and value < lo) or (hi is not None and value > hi):
                raise EngineBadRequest(f"{name} out of range")
            return value

        max_tokens = num("max_completion_tokens", None, int, 1)
        if max_tokens is None:
            max_tokens = num("max_tokens", self.opts.default_max_tokens, int, 1)
        temp = num("temperature", self.opts.default_temperature, float, 0.0, 10.0)
        top_p = num("top_p", 1.0, float, 0.0, 1.0)
        top_k = num("top_k", 0, int, 0)
        min_p = num("min_p", 0.0, float, 0.0, 1.0)
        seed = body.get("seed")
        if seed is not None:
            seed = num("seed", None, int)
        stop = body.get("stop")
        if isinstance(stop, str):
            stops = [stop]
        elif isinstance(stop, list) and all(isinstance(s, str) for s in stop):
            stops = list(stop)
        elif stop is None:
            stops = []
        else:
            raise EngineBadRequest("stop must be a string or a list of strings")
        stops = [s for s in stops if s][:16]
        processors = self._processors(body, num)
        tools = body.get("tools") or None
        tkw = body.get("chat_template_kwargs")
        tkw = dict(tkw) if isinstance(tkw, dict) else {}
        if isinstance(body.get("enable_thinking"), bool) and "enable_thinking" not in tkw:
            tkw["enable_thinking"] = body["enable_thinking"]
        with self._tok_lock:
            try:
                prompt = _eng_render(self.tokenizer, messages, tools=tools, template_kwargs=tkw)
            except Exception as exc:  # noqa: BLE001
                raise EngineBadRequest(f"chat template failed: {exc}") from None
            if not prompt:
                raise EngineBadRequest("the prompt is empty after the chat template")
            boundaries: list = []
            if not self.trimmable and self.prefix.max_bytes > 0:
                boundaries = _eng_boundaries(
                    self.tokenizer, messages, prompt, tools=tools, template_kwargs=tkw,
                    floor=self.prefix.peek(prompt), max_turns=self.opts.max_turn_snapshots)
        return EngineRequest(
            prompt=prompt, boundaries=boundaries, max_tokens=max_tokens,
            sampler=_eng_make_sampler(temp, top_p, min_p, top_k, seed),
            processors=processors, stops=stops, speculative_ok=True)

    @staticmethod
    def _processors(body: dict, num: Callable) -> list:
        rp = num("repetition_penalty", None, float, 0.0)
        pp = num("presence_penalty", None, float, -2.0, 2.0)
        fp = num("frequency_penalty", None, float, -2.0, 2.0)
        lb = body.get("logit_bias")
        bias = None
        if isinstance(lb, dict) and lb:
            try:
                bias = {int(k): float(v) for k, v in lb.items()}
            except (TypeError, ValueError):
                raise EngineBadRequest("logit_bias maps token ids to numbers") from None
        rp = rp if rp and rp != 1.0 else None
        pp = pp or None
        fp = fp or None
        if not (rp or pp or fp or bias):
            return []
        from mlx_lm.sample_utils import make_logits_processors

        ctx = num("repetition_context_size", 20, int, 1)
        return list(make_logits_processors(logit_bias=bias, repetition_penalty=rp,
                                           repetition_context_size=ctx, presence_penalty=pp,
                                           frequency_penalty=fp))

    def _new_detok(self) -> Any:
        tok = self.tokenizer
        try:
            det = tok.detokenizer
        except AttributeError:
            det = None
        if det is None:
            from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer

            det = NaiveStreamingDetokenizer(tok)
        det.reset()
        return det

    def submit(self, req: EngineRequest) -> None:
        if self.dead:
            raise EngineBusy(f"engine stopped: {self.dead}")
        with self._lock:
            if len(self._pending) >= self.max_queue:
                self._st["rejected"] += 1
                raise EngineBusy(f"engine queue is full ({len(self._pending)} waiting, "
                                 f"{len(self._uid_req) + (1 if self._job else 0)} running)")
            req.t_submit = time.perf_counter()
            req._wake = self._wake
            req.detok = self._new_detok()
            self._pending.append(req)
            self._st["requests"] += 1
        self._wake.set()

    # ---- scheduler -------------------------------------------------------- #
    def _tick(self) -> bool:
        self._reap()
        self._admit()
        if self._job is not None:
            self._job_step()
            return True
        if self._uid_req:
            self._batch_step()
            return True
        return False

    def _reap(self) -> None:
        for uid, req in list(self._uid_req.items()):
            if not req.cancelled.is_set():
                continue
            got = self._bg.remove([uid], return_prompt_caches=True).get(uid)
            if got:
                cache, toks = got
                if len(toks) > req.n_cached:
                    self.prefix.insert(list(toks), cache, "partial")
            del self._uid_req[uid]
            self._finish(req, "cancelled")
        with self._lock:
            if not any(r.cancelled.is_set() for r in self._pending):
                return
            gone = [r for r in self._pending if r.cancelled.is_set()]
            self._pending = collections.deque(r for r in self._pending if not r.cancelled.is_set())
        for req in gone:
            self._finish(req, "cancelled")

    def _pop(self, req: EngineRequest) -> None:
        with self._lock:
            if self._pending and self._pending[0] is req:
                self._pending.popleft()

    def _admit(self) -> None:
        while True:
            with self._lock:
                if not self._pending:
                    return
                req = self._pending[0]
                waiting = len(self._pending)
            if not self.batchable:
                if self._job is not None:
                    return
                self._pop(req)
                self._start(req, use_draft=self.spec_ok)
                return
            if self._job is not None:
                # A second request while one decodes speculatively: fold the
                # running one into the batch at its next round boundary.
                self._job.migrate = True
                return
            if (not self._uid_req and self.spec_ok and req.speculative_ok and waiting == 1):
                self._pop(req)
                self._start(req, use_draft=True)
                return
            if len(self._uid_req) >= self.max_batch:
                return
            if not self._memory_admits(req):
                if not req.admission_waited:
                    req.admission_waited = True
                    self._st["admission_waits"] += 1
                return
            self._pop(req)
            try:
                self._batch_insert(req)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self._fail(req, redact_text(f"{type(exc).__name__}: {exc}"))

    def _estimate(self, req: EngineRequest) -> int:
        return self._kv_fixed + self._kv_per_token * (
            len(req.prompt) + min(req.max_tokens, _ENG_REPLY_ESTIMATE))

    def _memory_admits(self, req: EngineRequest) -> bool:
        """Item 6, in the engine: KV the batch already holds + this request's must fit.

        Always true when nothing runs, so one huge request is slow, never stuck.
        """
        active = list(self._uid_req.values())
        if not active:
            return True
        need = self._estimate(req)
        if self._kv_budget > 0 and sum(self._estimate(r) for r in active) + need > self._kv_budget:
            return False
        if self._mem_limit > 0:
            import mlx.core as mx

            over = mx.get_active_memory() + need - self._mem_limit
            if over > 0:
                # Cached prefixes are the cheapest memory to give back.
                freed = self.prefix.release(over)
                if freed < over:
                    freed += self.draft_prefix.release(over - freed)
                if freed < over:
                    return False
        return True

    def _lookup(self, req: EngineRequest) -> tuple[list | None, int]:
        self._st["prompt_tokens"] += len(req.prompt)
        if self.prefix.max_bytes <= 0:
            return None, 0
        cache, n = self.prefix.lookup(req.prompt)
        self._st["cached_tokens"] += n
        return cache, n

    def _batch_insert(self, req: EngineRequest) -> None:
        cache, n = self._lookup(req)
        req.n_cached = n
        req.mode = "batch"
        if self.trimmable:
            segments, kinds = [req.prompt[n:]], []
        else:
            segments, kinds = _eng_segments(req.prompt, n, req.boundaries)
        req.kinds = collections.deque(kinds)
        (uid,) = self._bg.insert_segments(
            [segments], max_tokens=[req.max_tokens], caches=[cache],
            all_tokens=[list(req.prompt[:n])], samplers=[req.sampler],
            logits_processors=[list(req.processors)])
        self._uid_req[uid] = req
        self._peak_batch = max(self._peak_batch, len(self._uid_req))

    def _batch_step(self) -> None:
        prompt_responses, gen_responses = self._bg.next()
        if not self.trimmable and self.prefix.max_bytes > 0:
            for r in prompt_responses:
                if not r.end_of_segment or r.end_of_prompt:
                    continue
                req = self._uid_req.get(r.uid)
                if req is None:
                    continue
                kind = req.kinds.popleft() if req.kinds else "prompt"
                got = self._bg.extract_cache([r.uid]).get(r.uid)
                if got:
                    self.prefix.insert(list(got[1]), got[0], kind)
        if gen_responses:
            self._occ_sum += len(gen_responses)
            self._occ_steps += 1
        for r in gen_responses:
            req = self._uid_req.get(r.uid)
            if req is None:
                continue
            reason = self._on_token(req, int(r.token))
            if r.finish_reason is not None and reason is None:
                reason = "stop" if r.finish_reason == "stop" else "length"
            if reason is None:
                continue
            if r.finish_reason is not None and r.prompt_cache is not None:
                self.prefix.insert(list(r.all_tokens), r.prompt_cache, "final")
            else:
                got = self._bg.remove([r.uid], return_prompt_caches=True).get(r.uid)
                if got:
                    self.prefix.insert(list(got[1]), got[0], "final")
            del self._uid_req[r.uid]
            self._finish(req, reason)

    def _start(self, req: EngineRequest, use_draft: bool) -> None:
        try:
            self._start_job(req, use_draft)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._job = None
            self._fail(req, redact_text(f"{type(exc).__name__}: {exc}"))

    def _start_job(self, req: EngineRequest, use_draft: bool) -> None:
        from mlx_lm.generate import maybe_quantize_kv_cache
        from mlx_lm.models.cache import make_prompt_cache

        o = self.opts
        cache, n = self._lookup(req)
        if cache is None:
            cache, n = make_prompt_cache(self.model, o.max_kv_size or None), 0
        req.n_cached = n
        req.mode = "speculative" if use_draft else "sequential"
        dcache = None
        dn = 0
        if use_draft:
            dcache, dn = self.draft_prefix.lookup(req.prompt, max_len=n)
            if dcache is None:
                dcache, dn = make_prompt_cache(self.draft), 0
            if o.kv_bits > 0:
                # The speculative generator quantizes a slice of the list, not
                # this list, so quantize up front: every layer is then already a
                # QuantizedKVCache and the objects stored afterwards are current.
                maybe_quantize_kv_cache(cache, 0, o.kv_group_size, o.kv_bits)
                maybe_quantize_kv_cache(dcache, 0, o.kv_group_size, o.kv_bits)
        job = _EngJob(req, None, cache, dcache)
        job.pos, job.dpos = n, dn
        last = len(req.prompt) - 1
        if not self.trimmable and self.prefix.max_bytes > 0:
            # Same snapshot points as the batched path, taken between chunks.
            job.marks = [(b, k) for b, k in req.boundaries if n < b < last]
            if n < last:
                job.marks.append((last, "prompt"))
        self._job = job
        self._peak_batch = max(self._peak_batch, 1)

    def _prefill_tick(self, job: _EngJob) -> None:
        """One prefill chunk of the lone sequence, so arrivals and cancels are seen
        between chunks (an 8k-token prompt is ~25 s of prefill on an 8GB Mac)."""
        import mlx.core as mx
        from mlx_lm.generate import generate_step, speculative_generate_step

        o = self.opts
        req, prompt = job.req, job.req.prompt
        step = max(1, o.prefill_step_size)
        kv = dict(kv_bits=o.kv_bits or None, kv_group_size=o.kv_group_size,
                  quantized_kv_start=o.quantized_kv_start)
        last = len(prompt) - 1
        stop_at = job.marks[0][0] if job.marks else last
        end = min(job.pos + step, stop_at)
        if end > job.pos:
            t0 = time.perf_counter()
            _eng_prefill(self.model, job.cache, prompt[job.pos:end], step, kv)
            if job.draft and end > job.dpos:
                _eng_prefill(self.draft, job.dcache, prompt[job.dpos:end], step, kv)
                job.dpos = end
            self._seq_prefill_time += time.perf_counter() - t0
            self._seq_prefill_tokens += end - job.pos
            job.pos = end
        if job.marks and job.pos >= job.marks[0][0]:
            b, kind = job.marks.pop(0)
            self.prefix.insert(prompt[:b], _eng_cache_copy(job.cache), kind)
        if job.pos < last:
            return
        if job.draft and job.dpos < job.pos:
            _eng_prefill(self.draft, job.dcache, prompt[job.dpos:job.pos], step, kv)
            job.dpos = job.pos
        rest = mx.array(prompt[job.pos:])
        if job.draft:
            job.gen = speculative_generate_step(
                rest, self.model, self.draft, num_draft_tokens=max(1, o.num_draft_tokens),
                max_tokens=req.max_tokens, sampler=req.sampler,
                logits_processors=req.processors or None, prompt_cache=job.cache + job.dcache,
                prefill_step_size=step, **kv)
        else:
            job.gen = generate_step(
                rest, self.model, max_tokens=req.max_tokens, sampler=req.sampler,
                logits_processors=req.processors or None, prompt_cache=job.cache,
                prefill_step_size=step, **kv)

    def _job_step(self) -> None:
        job = self._job
        req = job.req
        if job.gen is None:
            if req.cancelled.is_set():
                self._end_job("cancelled")
            elif job.migrate and job.draft and self.batchable:
                self._migrate_job()
            else:
                self._prefill_tick(job)
            return
        try:
            while True:
                t0 = time.perf_counter()
                item = next(job.gen)
                dt = time.perf_counter() - t0
                if job.first:
                    job.first = False   # the last prompt token plus the first step
                    self._seq_prefill_time += dt
                    self._seq_prefill_tokens += 1
                else:
                    self._seq_decode_time += dt
                    self._seq_decode_tokens += 1
                tok = int(item[0])
                from_draft = bool(item[2]) if len(item) > 2 else False
                job.emitted.append(tok)
                if job.draft:
                    self._st["spec_accepted" if from_draft else "spec_rounds"] += 1
                reason = self._on_token(req, tok)
                if reason is not None:
                    self._end_job(reason)
                    return
                if req.cancelled.is_set():
                    self._end_job("cancelled")
                    return
                if not from_draft:
                    # A round boundary (or one plain token): the only point
                    # where the cache is exactly "everything but the last token".
                    if job.migrate and job.draft and self.batchable:
                        self._migrate_job()
                    return
        except StopIteration:
            self._end_job("length")

    def _store_job_caches(self, job: _EngJob, tokens: list[int], kind: str = "final") -> None:
        covered = _eng_cache_len(job.cache)
        if covered and covered <= len(tokens) and covered > job.req.n_cached:
            self.prefix.insert(tokens[:covered], job.cache, kind)
        if job.dcache is not None:
            dcov = _eng_cache_len(job.dcache)
            if dcov and dcov <= len(tokens):
                self.draft_prefix.insert(tokens[:dcov], job.dcache, "final")

    def _end_job(self, reason: str) -> None:
        job, self._job = self._job, None
        if job.gen is not None:
            try:
                job.gen.close()
            except Exception:  # noqa: BLE001
                pass
        self._store_job_caches(job, job.req.prompt + job.emitted,
                               "partial" if reason == "cancelled" else "final")
        self._finish(job.req, reason)

    def _migrate_job(self) -> None:
        """Hand the speculative sequence to the BatchGenerator, mid-answer."""
        job, self._job = self._job, None
        req = job.req
        if job.gen is not None:
            job.gen.close()     # its finally rewinds both caches to the accepted tokens
        else:
            self._st["spec_migrations_in_prefill"] += 1
        tokens = req.prompt + job.emitted
        covered = _eng_cache_len(job.cache) or 0
        if covered >= len(tokens):
            _eng_cache_trim(job.cache, covered - (len(tokens) - 1))
            covered = len(tokens) - 1
        if job.dcache is not None:
            dcov = _eng_cache_len(job.dcache)
            if dcov and dcov <= len(tokens):
                self.draft_prefix.insert(tokens[:dcov], job.dcache, "final")
        remaining = max(1, req.max_tokens - len(req.tokens))
        (uid,) = self._bg.insert(
            [tokens[covered:]], max_tokens=[remaining], caches=[job.cache],
            all_tokens=[list(tokens[:covered])], samplers=[req.sampler],
            logits_processors=[list(req.processors)])
        req.mode = "batch"
        self._uid_req[uid] = req
        self._st["spec_migrations"] += 1
        self._peak_batch = max(self._peak_batch, len(self._uid_req))

    # ---- output ---------------------------------------------------------- #
    def _on_token(self, req: EngineRequest, tok: int) -> str | None:
        now = time.perf_counter()
        req.tokens.append(tok)
        if req.t_first is None:
            req.t_first = now
            # "prefix_hit" means most of the prompt came from the cache; a few
            # shared template tokens do not make a cold prompt warm.
            self._ttft.append((now - req.t_submit, 2 * req.n_cached >= len(req.prompt)))
        sec = int(now)
        if self._recent and self._recent[-1][0] == sec:
            self._recent[-1][1] += 1
        else:
            self._recent.append([sec, 1])
            while self._recent and now - self._recent[0][0] > _ENG_RECENT_WINDOW_S + 1:
                self._recent.popleft()
        if tok in self._eos:
            return "stop"
        req.detok.add_token(tok)
        segment = req.detok.last_segment
        if segment and self._emit_text(req, segment):
            return "stop"
        if len(req.tokens) >= req.max_tokens:
            return "length"
        return None

    @staticmethod
    def _emit_text(req: EngineRequest, text: str) -> bool:
        """Emit text, holding back a possible stop-string prefix. True on a stop."""
        if not req.stops:
            req.emit(("delta", text))
            return False
        buf = req.hold + text
        cut = -1
        for s in req.stops:
            i = buf.find(s)
            if i != -1 and (cut == -1 or i < cut):
                cut = i
        if cut != -1:
            if cut:
                req.emit(("delta", buf[:cut]))
            req.hold = ""
            req.stopped_by_string = True
            return True
        keep = 0
        for s in req.stops:
            for k in range(min(len(s) - 1, len(buf)), 0, -1):
                if buf.endswith(s[:k]):
                    keep = max(keep, k)
                    break
        out, req.hold = buf[:len(buf) - keep], buf[len(buf) - keep:]
        if out:
            req.emit(("delta", out))
        return False

    def _finish(self, req: EngineRequest, reason: str) -> None:
        if req.done.is_set():
            return
        if reason in ("stop", "length") and not req.stopped_by_string and req.detok is not None:
            try:
                req.detok.finalize()
                tail = req.detok.last_segment
            except Exception:  # noqa: BLE001
                tail = ""
            if tail and self._emit_text(req, tail):
                reason = "stop"
            if req.hold and not req.stopped_by_string:
                req.emit(("delta", req.hold))
            req.hold = ""
        req.finish_reason = reason
        prompt_tokens = len(req.prompt)
        req.usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(req.tokens),
            "total_tokens": prompt_tokens + len(req.tokens),
            "prompt_tokens_details": {"cached_tokens": req.n_cached},
        }
        self._st["cancelled" if reason == "cancelled" else "completed"] += 1
        self._st["generated_tokens"] += len(req.tokens)
        req.emit(("done", reason, req.usage))
        req.done.set()

    def _fail(self, req: EngineRequest, message: str) -> None:
        if req.done.is_set():
            return
        self._st["errors"] += 1
        req.finish_reason = "error"
        req.emit(("error", message))
        req.done.set()

    # ---- stats ----------------------------------------------------------- #
    def stats(self) -> dict:
        with self._lock:
            waiting = len(self._pending)
        counters = getattr(self._bg, "_counters", None)
        b_dec_tok = int(getattr(counters, "generation_tokens", 0) or 0)
        b_dec_time = float(getattr(counters, "decode_time", 0.0) or 0.0)
        b_pre_tok = int(getattr(counters, "prompt_tokens", 0) or 0)
        b_pre_time = float(getattr(counters, "prompt_time", 0.0) or 0.0)
        dec_tok = b_dec_tok + self._seq_decode_tokens
        dec_time = b_dec_time + self._seq_decode_time
        pre_tok = b_pre_tok + self._seq_prefill_tokens
        pre_time = b_pre_time + self._seq_prefill_time
        now = time.perf_counter()
        recent = [b for b in list(self._recent) if now - b[0] <= _ENG_RECENT_WINDOW_S]
        recent_tok_s = None
        if recent:
            span = max(1.0, now - recent[0][0])
            recent_tok_s = round(sum(b[1] for b in recent) / span, 1)
        ttft = list(self._ttft)
        st = self._st
        rounds = st["spec_rounds"]
        nd = max(1, self.opts.num_draft_tokens)
        job = self._job
        active = len(self._uid_req) + (1 if job else 0)
        try:
            import mlx.core as mx
            memory = {"active_bytes": int(mx.get_active_memory()),
                      "peak_bytes": int(mx.get_peak_memory()),
                      "cache_bytes": int(mx.get_cache_memory())}
        except Exception:  # noqa: BLE001
            memory = {}
        return {
            "engine": ENGINE_BACKEND,
            "model": self.served_name,
            "adapter": self.opts.adapter_path,
            "ready": self.ready.is_set(),
            "dead": self.dead,
            "uptime_s": round(time.time() - self.started_at, 1),
            "mode": {
                "batching": self.batchable,
                "speculative": self.spec_ok,
                "draft_model": self.opts.draft_model if self.spec_ok else "",
                "kv_bits": self.opts.kv_bits,
                "trimmable_cache": self.trimmable,
                "running": ("speculative" if job and job.draft else "sequential" if job
                            else "batch" if self._uid_req else "idle"),
            },
            "batch": {
                "active": active,
                "max": self.max_batch,
                "occupancy": round(active / self.max_batch, 3),
                "avg_decode_batch": round(self._occ_sum / self._occ_steps, 2) if self._occ_steps else None,
                "peak": self._peak_batch,
                "prefill_batch": self.prefill_batch,
            },
            "queue": {"waiting": waiting, "max": self.max_queue},
            "requests": {k: st[k] for k in ("requests", "completed", "cancelled", "rejected",
                                            "errors", "admission_waits")},
            "tokens": {"prompt": st["prompt_tokens"], "cached": st["cached_tokens"],
                       "generated": st["generated_tokens"]},
            "prefix_cache": self.prefix.stats(),
            "draft_prefix_cache": self.draft_prefix.stats() if self.spec_ok else None,
            "decode": {
                "tokens": dec_tok,
                "tok_s": round(dec_tok / dec_time, 1) if dec_time > 0 else None,
                "recent_tok_s": recent_tok_s,
                "prefill_tokens": pre_tok,
                "prefill_tok_s": round(pre_tok / pre_time, 1) if pre_time > 0 else None,
            },
            "ttft_ms": {
                **_eng_percentiles([t for t, _ in ttft]),
                "cold": _eng_percentiles([t for t, hit in ttft if not hit]),
                "prefix_hit": _eng_percentiles([t for t, hit in ttft if hit]),
            },
            "speculative": {
                "rounds": rounds,
                "accepted": st["spec_accepted"],
                "num_draft_tokens": nd if self.spec_ok else 0,
                "acceptance_rate": round(st["spec_accepted"] / (rounds * nd), 3) if rounds else None,
                "migrations": st["spec_migrations"],
                "migrations_in_prefill": st["spec_migrations_in_prefill"],
            },
            "admission": {
                "kv_budget_bytes": self._kv_budget,
                "memory_limit_bytes": self._mem_limit,
                "kv_bytes_per_token": self._kv_per_token,
                "kv_fixed_bytes": self._kv_fixed,
                "estimated_active_bytes": sum(self._estimate(r) for r in list(self._uid_req.values())),
                "waits": st["admission_waits"],
            },
            "memory": memory,
        }

    def completion_body(self, req: EngineRequest, text: str) -> dict:
        return {
            "id": req.id,
            "object": "chat.completion",
            "created": req.created,
            "model": self.served_name,
            "system_fingerprint": _eng_fingerprint(),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": req.finish_reason if req.finish_reason in ("stop", "length") else "stop",
                "logprobs": None,
            }],
            "usage": req.usage,
        }


def _eng_fingerprint() -> str:
    try:
        import mlx_lm
        return f"mlx-engine-{getattr(mlx_lm, '__version__', '')}"
    except Exception:  # noqa: BLE001
        return "mlx-engine"


# --------------------------------------------------------------------------- #
# HTTP surface (Starlette: plain handlers, no annotation resolution)
# --------------------------------------------------------------------------- #
def build_engine_app(engine: MLXEngine, *, allowed_origins: str = "") -> Any:
    """The OpenAI-compatible ASGI app over an engine."""
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Route

    def err(status: int, message: str, etype: str = "invalid_request_error",
            code: str | None = None, retry: str | None = None) -> Any:
        headers = {"Retry-After": retry} if retry else None
        return JSONResponse({"error": {"message": message, "type": etype, "code": code}},
                            status_code=status, headers=headers)

    def sse(payload: dict) -> str:
        return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

    async def chat(request: Any) -> Any:
        if engine.dead:
            return err(503, f"engine stopped: {engine.dead}", "server_error", "engine_dead")
        if not engine.ready.is_set():
            return err(503, "model is loading", "server_error", "loading", "2")
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return err(400, "the body must be JSON")
        if not isinstance(body, dict):
            return err(400, "the body must be a JSON object")
        model = body.get("model")
        if model not in (None, "") and str(model) not in engine.accepted_model_names():
            return err(404, f"model {model!r} is not served here; this engine serves "
                            f"{engine.served_name!r} (\"default_model\" always works)",
                       code="model_not_found")
        try:
            req = await asyncio.to_thread(engine.prepare, body)
        except EngineBadRequest as exc:
            return err(400, str(exc))
        except EngineBusy as exc:
            return err(503, str(exc), "server_error", "busy", "1")
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def emit(ev: tuple) -> None:
            try:
                loop.call_soon_threadsafe(q.put_nowait, ev)
            except RuntimeError:
                pass

        req.emit = emit
        try:
            engine.submit(req)
        except EngineBusy as exc:
            return err(503, str(exc), "server_error", "queue_full", "1")
        opts = body.get("stream_options")
        include_usage = isinstance(opts, dict) and bool(opts.get("include_usage"))
        if body.get("stream"):
            return StreamingResponse(_stream(req, q, include_usage), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        parts: list[str] = []
        try:
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        req.cancel()
                        return err(499, "client disconnected")
                    continue
                if ev[0] == "delta":
                    parts.append(ev[1])
                elif ev[0] == "done":
                    break
                else:
                    return err(500, str(ev[1]), "server_error")
        except asyncio.CancelledError:
            req.cancel()
            raise
        return JSONResponse(engine.completion_body(req, "".join(parts)))

    async def _stream(req: EngineRequest, q: asyncio.Queue, include_usage: bool):
        base = {"id": req.id, "object": "chat.completion.chunk", "created": req.created,
                "model": engine.served_name, "system_fingerprint": _eng_fingerprint()}

        def chunk(delta: dict, finish: str | None = None) -> str:
            return sse({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish,
                                             "logprobs": None}]})

        finished = False
        try:
            yield chunk({"role": "assistant", "content": ""})
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    # A comment line: keeps proxies open, and is the write that
                    # notices a client that went away while the request waited.
                    yield ": keepalive\n\n"
                    continue
                if ev[0] == "delta":
                    text = [ev[1]]
                    tail = None
                    while not q.empty():
                        nxt = q.get_nowait()
                        if nxt[0] == "delta":
                            text.append(nxt[1])
                        else:
                            tail = nxt
                            break
                    yield chunk({"content": "".join(text)})
                    if tail is None:
                        continue
                    ev = tail
                if ev[0] == "done":
                    finished = True
                    reason = ev[1] if ev[1] in ("stop", "length") else "stop"
                    yield chunk({}, reason)
                    if include_usage:
                        yield sse({**base, "choices": [], "usage": ev[2]})
                    yield "data: [DONE]\n\n"
                    return
                finished = True
                yield sse({"error": {"message": str(ev[1]), "type": "server_error"}})
                yield "data: [DONE]\n\n"
                return
        finally:
            if not finished:
                req.cancel()

    async def models(_request: Any) -> Any:
        return JSONResponse({"object": "list", "data": [{
            "id": engine.served_name, "object": "model", "created": int(engine.started_at),
            "owned_by": ENGINE_BACKEND}]})

    async def health(_request: Any) -> Any:
        if engine.dead:
            return JSONResponse({"status": "error", "error": engine.dead}, status_code=503)
        if not engine.ready.is_set():
            return JSONResponse({"status": "loading"}, status_code=503)
        return JSONResponse({"status": "ok", "model": engine.served_name})

    async def stats(_request: Any) -> Any:
        return JSONResponse(engine.stats())

    routes = [
        Route("/v1/chat/completions", chat, methods=["POST"]),
        Route("/chat/completions", chat, methods=["POST"]),
        Route("/v1/models", models, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        Route("/v1/engine/stats", stats, methods=["GET"]),
    ]
    middleware = []
    origins = [o.strip() for o in (allowed_origins or "").split(",") if o.strip()]
    if origins:
        middleware.append(Middleware(CORSMiddleware, allow_origins=origins,
                                     allow_methods=["GET", "POST", "OPTIONS"],
                                     allow_headers=["*"]))
    return Starlette(routes=routes, middleware=middleware)


def start_engine_server(engine: MLXEngine, *, host: str = "127.0.0.1", port: int = 8090,
                        allowed_origins: str = "") -> Any:
    """Serve an engine from a background thread (tests, embedding). Returns the uvicorn.Server."""
    import uvicorn

    config = uvicorn.Config(build_engine_app(engine, allowed_origins=allowed_origins), host=host,
                            port=port, log_level="warning", access_log=False, lifespan="off")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="mlx-engine-http", daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started and thread.is_alive() and time.time() < deadline:
        time.sleep(0.02)
    server.thread = thread  # type: ignore[attr-defined]
    return server


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _eng_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mlx-engine",
        description="OpenAI-compatible MLX server: continuous batching, speculative decoding "
                    "while one request runs, quantized KV, and a token-level prefix cache with "
                    "state snapshots for hybrid models.")
    p.add_argument("--model", required=True, help="HF repo id or local path")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--adapter-path", default="", help="LoRA adapter directory")
    p.add_argument("--draft-model", default="", help="draft model for speculative decoding")
    p.add_argument("--num-draft-tokens", type=int, default=3)
    p.add_argument("--kv-bits", type=int, default=0,
                   help="quantize the KV cache to this many bits (0 = off; turns batching off)")
    p.add_argument("--kv-group-size", type=int, default=64)
    p.add_argument("--quantized-kv-start", type=int, default=5000)
    p.add_argument("--max-kv-size", type=int, default=0, help="rotating KV cache size (0 = unbounded)")
    p.add_argument("--max-batch", type=int, default=4, help="sequences decoded together")
    p.add_argument("--prefill-batch", type=int, default=0, help="prompts prefilled together (0 = auto)")
    p.add_argument("--prefill-step-size", type=int, default=512)
    p.add_argument("--prefix-cache-bytes", type=int, default=-1, help="-1 = auto, 0 = off")
    p.add_argument("--prefix-cache-dir", default="", help="spill prefix-cache entries here")
    p.add_argument("--prefix-cache-disk-bytes", type=int, default=4 << 30)
    p.add_argument("--max-queue", type=int, default=0, help="waiting requests before 503 (0 = auto)")
    p.add_argument("--kv-budget-bytes", type=int, default=-1,
                   help="KV bytes live sequences may hold (-1 = auto from the GPU working set, 0 = off)")
    p.add_argument("--max-tokens", type=int, default=512, help="default max_tokens")
    p.add_argument("--temp", type=float, default=0.0, help="default temperature")
    p.add_argument("--served-model-name", default="")
    p.add_argument("--allowed-origins", default="", help="comma-separated CORS origins (empty = none)")
    return p


def engine_options_from_args(args: argparse.Namespace) -> EngineOptions:
    return EngineOptions(
        model=args.model, adapter_path=args.adapter_path or "", draft_model=args.draft_model or "",
        num_draft_tokens=args.num_draft_tokens, kv_bits=max(0, args.kv_bits),
        kv_group_size=args.kv_group_size, quantized_kv_start=max(0, args.quantized_kv_start),
        max_kv_size=max(0, args.max_kv_size), max_batch=max(1, args.max_batch),
        prefill_batch=max(0, args.prefill_batch), prefill_step_size=max(16, args.prefill_step_size),
        prefix_cache_bytes=args.prefix_cache_bytes, prefix_cache_dir=args.prefix_cache_dir or "",
        prefix_cache_disk_bytes=max(0, args.prefix_cache_disk_bytes), max_queue=max(0, args.max_queue),
        kv_budget_bytes=args.kv_budget_bytes, default_max_tokens=max(1, args.max_tokens),
        default_temperature=max(0.0, args.temp), served_name=args.served_model_name or "")


def mlx_engine_cli(argv: list[str] | None = None) -> int:
    """`deploy.py mlx-engine ...` / `python -m local_llm.mlx_engine ...`."""
    args = _eng_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    for name in ("mlx_lm", "starlette", "uvicorn"):
        if importlib.util.find_spec(name) is None:
            print(f"mlx-engine needs {name}, which is not installed.", file=sys.stderr)
            return 2
    import uvicorn

    engine = MLXEngine(engine_options_from_args(args))
    app = build_engine_app(engine, allowed_origins=args.allowed_origins)
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level="warning",
                                           access_log=False, lifespan="off",
                                           timeout_graceful_shutdown=5))

    def _exit() -> None:
        server.should_exit = True

    engine.on_fatal = _exit
    # The port opens before the weights finish loading (mlx_lm.server does the
    # same): the supervisor waits for the port, then probes until a completion
    # succeeds, and /health answers 503 "loading" meanwhile.
    engine.start()
    try:
        server.run()
    finally:
        engine.stop()
    return 1 if engine.dead else 0


# --------------------------------------------------------------------------- #
# Backend registration (backends.py contract)
# --------------------------------------------------------------------------- #
_ENG_LAUNCH_INPUTS = (
    "kv_bits", "kv_group_size", "quantized_kv_start", "max_kv_size", "prefill_step_size",
    "prompt_cache_bytes", "prompt_cache_dir", "decode_concurrency", "prompt_concurrency",
    "max_concurrent_generations", "draft_model", "num_draft_tokens", "web_port",
)


def _eng_entry_argv() -> list[str]:
    """How to start this module as a process, in the package and in the bundle."""
    if __package__:
        root = str(Path(__file__).resolve().parent.parent)
        boot = (f"import sys; sys.path.insert(0, {root!r}); "
                f"from {__package__}.mlx_engine import mlx_engine_cli; "
                "raise SystemExit(mlx_engine_cli(sys.argv[1:]))")
        return [sys.executable, "-c", boot]
    return [sys.executable, str(Path(__file__).resolve()), "mlx-engine"]


def engine_plan(m: Any) -> dict:
    """Every flag the launcher passes, plus the reasoning behind each."""
    ram = float(TOTAL_RAM_GB or 8)
    conc = int(getattr(m, "max_concurrent_generations", 0) or 0) or _default_concurrent_generations(ram)
    decisions: list[str] = []
    flags: list[tuple[str, str]] = []
    dc = int(getattr(m, "decode_concurrency", 0) or 0)
    kv_bits = int(getattr(m, "kv_bits", 0) or 0)
    if dc > 0:
        max_batch = dc
        why = "DECODE_CONCURRENCY"
    else:
        cap = _eng_decode_cap(ram)
        max_batch = min(cap, conc + 1)
        why = (f"MAX_CONCURRENT_GENERATIONS {conc} plus one slot so the watchdog probe is "
               f"batched beside a long answer, capped at {cap} for {ram:.0f} GB")
    flags.append(("--max-batch", str(max_batch)))
    pc = int(getattr(m, "prompt_concurrency", 0) or 0)
    if pc > 0:
        flags.append(("--prefill-batch", str(min(pc, max_batch))))
    step = int(getattr(m, "prefill_step_size", 0) or 0)
    if step > 0:
        flags.append(("--prefill-step-size", str(step)))
    if kv_bits > 0:
        flags += [("--kv-bits", str(kv_bits)),
                  ("--kv-group-size", str(int(getattr(m, "kv_group_size", 64) or 64))),
                  ("--quantized-kv-start", str(int(getattr(m, "quantized_kv_start", 0) or 0)))]
        decisions.append(
            f"KV_BITS={kv_bits}: QuantizedKVCache has no merge() on mlx-lm 0.32, so requests are "
            "served one at a time (as on mlx_lm.server); the prefix cache still applies.")
    else:
        decisions.append(f"Continuous batching: up to {max_batch} sequences decode together ({why}).")
    max_kv = int(getattr(m, "max_kv_size", 0) or 0)
    if max_kv > 0:
        flags.append(("--max-kv-size", str(max_kv)))
    cfg = _backends_model_config(getattr(m, "model_id", "") or "")
    hybrid = _backends_is_hybrid(cfg) if cfg else None
    draft, note = _backends_resolve_draft(m)
    if note:
        decisions.append(note)
    if draft and hybrid:
        decisions.append(f"Draft model {draft} not loaded: this hybrid model's recurrent cache "
                         "cannot be rewound, which speculative decoding requires.")
        draft = ""
    if draft:
        flags.append(("--draft-model", draft))
        ndt = int(getattr(m, "num_draft_tokens", 0) or 0)
        if ndt > 0:
            flags.append(("--num-draft-tokens", str(ndt)))
        decisions.append(
            f"Speculative decoding with {draft} while a single request runs; a second request "
            "folds the running one into the batch at its next draft round, so drafting never "
            "turns batching off.")
    pbytes = int(getattr(m, "prompt_cache_bytes", 0) or 0)
    if pbytes > 0:
        flags.append(("--prefix-cache-bytes", str(pbytes)))
    decisions.append(
        "Prefix cache: token-level radix tree, LRU by bytes "
        + (f"(PROMPT_CACHE_BYTES={pbytes})" if pbytes > 0
           else f"(auto {_eng_auto_prefix_bytes() >> 20} MiB)")
        + "; PROMPT_CACHE_SIZE is not used.")
    if hybrid:
        decisions.append("Hybrid model: snapshots at the end of the system prompt, of each completed "
                         "assistant turn and of the prompt, so a follow-up turn resumes from the "
                         "nearest one.")
    cache_dir = str(getattr(m, "prompt_cache_dir", "") or "")
    if cache_dir:
        flags.append(("--prefix-cache-dir", cache_dir))
        decisions.append(f"PROMPT_CACHE_DIR: evicted prefixes spill to {cache_dir} as safetensors, "
                         "keyed by model, adapter fingerprint and KV settings.")
    origins = _backends_allowed_origins(m)
    if origins:
        flags.append(("--allowed-origins", origins))
    return {"flags": flags, "max_batch": max_batch, "batching": kv_bits <= 0,
            "draft_model": draft, "hybrid_model": hybrid, "decisions": decisions}


def _eng_launcher(m: Any, use_adapter: bool) -> list[str]:
    plan = engine_plan(m)
    cmd = _eng_entry_argv() + ["--model", str(m.model_id), "--port", str(m.model_port)]
    adapter = m.adapter_path() if use_adapter else None
    if adapter is not None:
        cmd += ["--adapter-path", str(adapter)]
    accepted = []
    for flag, value in plan["flags"]:
        cmd += [flag, value]
        accepted.append(flag)
    m._accepted_levers = accepted
    return cmd


def _eng_available() -> tuple[bool, str]:
    missing = [n for n in ("mlx_lm", "starlette", "uvicorn") if importlib.util.find_spec(n) is None]
    if missing:
        return False, "needs " + ", ".join(missing)
    return True, ""


def _eng_effective_caps(m: Any) -> frozenset:
    plan = engine_plan(m)
    caps = {"prompt_cache", "cached_tokens"}
    if plan["batching"]:
        caps.add("batching")
    if plan["draft_model"]:
        caps.add("speculative")
    if int(getattr(m, "kv_bits", 0) or 0) > 0:
        caps.add("kv_quant")
    return frozenset(caps)


def _eng_probe_timeout(m: Any) -> float:
    base = float(getattr(m, "probe_timeout", 25.0) or 25.0)
    # Sequential serving (--kv-bits) queues the probe behind the answer in front.
    return max(base, 90.0) if int(getattr(m, "kv_bits", 0) or 0) > 0 else base


def _eng_cache_plan(m: Any) -> dict:
    plan = engine_plan(m)
    base = prompt_cache_plan(m, batching=plan["batching"],
                             concurrency=int(getattr(m, "max_concurrent_generations", 0) or 1),
                             adopt=False)
    pbytes = int(getattr(m, "prompt_cache_bytes", 0) or 0) or _eng_auto_prefix_bytes()
    base.update({"size": 0, "bytes": pbytes, "bytes_enforced": True,
                 "notes": [d for d in plan["decisions"] if "refix" in d or "ybrid" in d]})
    return base


register_backend(BackendSpec(
    name=ENGINE_BACKEND,
    label="MLX engine (batching + prefix cache)",
    help_module="",
    launcher=_eng_launcher,
    fatal_patterns=tuple(METAL_FATAL_PATTERNS) + (re.compile(re.escape(ENGINE_FATAL_LINE).encode()),),
    capabilities=frozenset({"batching", "speculative", "kv_quant", "prompt_cache", "cached_tokens"}),
    available=_eng_available,
    request_model=lambda _config: "default_model",
    description=("This app's own OpenAI-compatible MLX server: continuous batching that keeps "
                 "speculative decoding for a lone request, a byte-bounded token-level prefix cache "
                 "with state snapshots for hybrid models (optionally persisted to "
                 "PROMPT_CACHE_DIR), and memory-aware admission."),
    decide=lambda m: engine_plan(m)["decisions"],
    effective_capabilities=_eng_effective_caps,
    supports_adapter=True,
    probe_timeout=_eng_probe_timeout,
    cache_plan=_eng_cache_plan,
    launch_inputs=_ENG_LAUNCH_INPUTS,
))



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'ENGINE_BACKEND',
    'ENGINE_FATAL_LINE',
    'EngineBadRequest',
    'EngineBusy',
    'EngineOptions',
    'EngineRequest',
    'MLXEngine',
    'PrefixCache',
    '_eng_boundaries',
    '_eng_cache_copy',
    '_eng_entry_argv',
    '_eng_make_sampler',
    '_eng_model_key',
    '_eng_parser',
    '_eng_render',
    'build_engine_app',
    'engine_options_from_args',
    'engine_plan',
    'mlx_engine_cli',
    'start_engine_server',
]


if __name__ == "__main__":
    raise SystemExit(mlx_engine_cli(sys.argv[1:]))
