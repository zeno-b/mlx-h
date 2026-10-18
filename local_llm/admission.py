"""Memory-aware admission control for model generations (item 6).

GenerationGate (core.py) admits a fixed number of turns and sheds load past a
bounded queue. That number is a guess made once from total RAM, and it is the
wrong unit: what actually runs a Mac out of memory is the sum of the KV caches
of everything in flight, and a 4,000-token agent turn costs a hundred times
what a "thanks!" does. On an 8GB machine two long turns next to a 3B model is
the difference between "slower" and macOS compressing memory until tok/s
collapses for everyone, and nothing in the old gate could see it coming.

The batching itself happens in the backend. mlx-lm 0.32 runs a BatchGenerator
(continuous batching: new sequences join the running decode batch between
steps) sized by --decode-concurrency and --prompt-concurrency; requests past
that wait inside the server where the app cannot see them, and --kv-bits or a
draft model turn batching off entirely (requests are then served one by one).
This controller sits in front of whatever backend is configured and decides
WHEN a request is handed over, using three facts per request:

  * a slot: at most `limit` run at once (the old gate's rule, unchanged);
  * its KV cost: cost_tokens (prompt estimate + reply budget, capped at the
    context window) times kv_bytes_per_token, read from the model's own
    config.json (see estimate_kv_bytes_per_token);
  * the machine's free memory now: when admitting it would push projected free
    memory below admission_min_free_mb, the request waits for a running one to
    finish instead of starting.

A request is ALWAYS admitted when nothing is running, so a single huge request
on a busy-elsewhere machine is slow, never deadlocked. Waiters are strictly
FIFO (a small request does not overtake a large one that is waiting for
memory, which would starve it), the waiting room is bounded exactly like
GenerationGate (limit * queue multiplier, then GenerationBusy -> 503), and an
on_wait callback reports the queue position so a streaming client can show
"queued (2nd)" instead of a spinner that looks like a hang.

Free memory comes from psutil when it is installed, else from `vm_stat` (free +
inactive + speculative pages, cached for about a second). When neither works
(not macOS, sandboxed) the probe returns None and memory awareness switches
itself off: the controller then behaves exactly like GenerationGate. With
ADMISSION_MEMORY_AWARE=0 it does so unconditionally.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import logging
import re
import subprocess
import threading
import time
import weakref
from pathlib import Path
from typing import Any, Callable

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .sysutil import *  # noqa: F401,F403


_ADM_MB = 1024 * 1024
_adm_logger = get_logger("admission")

# --------------------------------------------------------------------------- #
# Free-memory probe
# --------------------------------------------------------------------------- #
# psutil is optional (not in REQUIRED_PACKAGES). A failed import is not cached
# by Python, so remember the outcome instead of re-searching sys.path on every
# admission decision.
_adm_psutil_state: dict[str, Any] = {"checked": False, "module": None}
_adm_probe_cache: dict[str, Any] = {"at": 0.0, "value": None, "source": None}
_adm_probe_lock = threading.Lock()
_ADM_VM_PAGE = re.compile(r"page size of (\d+) bytes")
_ADM_VM_LINE = re.compile(r"^Pages (free|inactive|speculative):\s+(\d+)\.?\s*$", re.M)


def _adm_psutil_available_bytes() -> int | None:
    if not _adm_psutil_state["checked"]:
        _adm_psutil_state["checked"] = True
        try:
            import psutil  # optional
            _adm_psutil_state["module"] = psutil
        except Exception:  # noqa: BLE001 - absent or broken: fall through to vm_stat
            _adm_psutil_state["module"] = None
    psutil = _adm_psutil_state["module"]
    if psutil is None:
        return None
    try:
        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001
        return None


def _adm_parse_vm_stat(text: str) -> int | None:
    """Bytes reclaimable without paging, from `vm_stat` output, or None.

    free + inactive + speculative is what macOS can hand to a new allocation
    without compressing or swapping anything; "free" alone reads near zero on
    a healthy Mac because the kernel keeps file pages cached in inactive.
    """
    page = _ADM_VM_PAGE.search(text or "")
    counts = {name: int(value) for name, value in _ADM_VM_LINE.findall(text or "")}
    if page is None or not {"free", "inactive", "speculative"} <= set(counts):
        return None
    return (counts["free"] + counts["inactive"] + counts["speculative"]) * int(page.group(1))


def _adm_vm_stat_available_bytes() -> int | None:
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return _adm_parse_vm_stat(out.stdout)


def free_memory_probe(max_age: float = 1.0) -> tuple[int | None, str | None]:
    """(bytes available for new allocations, source) or (None, None).

    Cached for `max_age` seconds: the admission path consults it on every
    contended acquire and /api/health on every poll, and a fresh `vm_stat`
    (about 2ms measured on an M1) per call is waste for a number that does not
    move that fast. The cache also holds a None result, so a machine without
    vm_stat does not spawn a failing process per request.
    """
    now = time.monotonic()
    with _adm_probe_lock:
        if _adm_probe_cache["at"] and now - _adm_probe_cache["at"] < max_age:
            return _adm_probe_cache["value"], _adm_probe_cache["source"]
    value, source = _adm_psutil_available_bytes(), "psutil"
    if value is None:
        value, source = _adm_vm_stat_available_bytes(), "vm_stat"
    if value is None:
        source = None
    with _adm_probe_lock:
        _adm_probe_cache.update(at=time.monotonic(), value=value, source=source)
    return value, source


def free_memory_bytes(max_age: float = 1.0) -> int | None:
    """Bytes available for new allocations now, or None when unmeasurable."""
    return free_memory_probe(max_age)[0]


# --------------------------------------------------------------------------- #
# KV cost estimate
# --------------------------------------------------------------------------- #
# fp16 KV bytes per token of the RAM-tier default model (core._default_model_for_ram),
# used when the configured model's config.json is not on disk yet (first run,
# before the download). Values computed from the published Qwen2.5-Coder
# configs, 2 * layers * kv_heads * head_dim * 2 bytes: 32B (64, 8, 128),
# 14B (48, 8, 128), 7B (28, 4, 128), 3B (36, 2, 128). Inference from those
# model cards, not read from this machine's cache (only the 3B is cached here,
# and the test checks that one against its real file).
_ADM_FALLBACK_KV = ((48, 262144), (24, 196608), (14, 57344), (0, 36864))
_adm_kv_cache: dict[tuple[str, int], int] = {}


def _adm_model_config_path(model_id: str) -> Path | None:
    """config.json of a local model directory or a cached HF repo, or None."""
    try:
        local = Path(model_id).expanduser()
        if local.is_dir():
            candidate = local / "config.json"
            return candidate if candidate.is_file() else None
        repo = hf_cache_dir() / ("models--" + model_id.replace("/", "--"))
        snapshots = repo / "snapshots"
        ref = repo / "refs" / "main"
        if ref.is_file():
            pinned = snapshots / ref.read_text(encoding="utf-8").strip() / "config.json"
            if pinned.is_file():
                return pinned
        if snapshots.is_dir():
            for snap in sorted(snapshots.iterdir()):
                if (snap / "config.json").is_file():
                    return snap / "config.json"
    except (OSError, ValueError):
        return None
    return None


def _adm_kv_from_config(cfg: dict, element_bytes: float) -> int | None:
    # Multimodal repos nest the language model's shape under text_config.
    src = cfg
    if "num_hidden_layers" not in cfg and isinstance(cfg.get("text_config"), dict):
        src = cfg["text_config"]
    try:
        layers = int(src.get("num_hidden_layers") or 0)
        heads = int(src.get("num_attention_heads") or 0)
        kv_heads = int(src.get("num_key_value_heads") or heads)
        head_dim = int(src.get("head_dim") or 0)
        if not head_dim and heads:
            head_dim = int(src.get("hidden_size") or 0) // heads
    except (TypeError, ValueError):
        return None
    # Hybrid models (linear-attention or sliding-window layers mixed with full
    # attention) only grow a per-token cache on the full-attention layers; the
    # rest hold a fixed-size state. Counting every layer would overstate their
    # cost several times over and serialize requests that fit easily.
    layer_types = src.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        full = sum(1 for kind in layer_types if "full" in str(kind))
        if full:
            layers = full
    if layers <= 0 or kv_heads <= 0 or head_dim <= 0:
        return None
    return int(2 * layers * kv_heads * head_dim * element_bytes)


def estimate_kv_bytes_per_token(model_id: str, kv_bits: int = 0,
                                ram_gb: float | None = None) -> int:
    """KV-cache bytes one token of context costs for this model.

    2 (keys and values) * num_hidden_layers * num_key_value_heads * head_dim *
    bytes per element, where head_dim falls back to hidden_size /
    num_attention_heads and bytes is kv_bits/8 with KV quantization, else 2
    (fp16/bf16). Read from the model's own config.json (a local directory or
    the Hugging Face cache, never downloaded); the model is never loaded. When
    no config is on disk the RAM-tier default model's figure is used, and that
    fallback is not cached, so a model downloaded later is picked up.
    """
    bits = int(kv_bits or 0)
    element_bytes = bits / 8.0 if bits > 0 else 2.0
    key = (str(model_id or ""), bits)
    cached = _adm_kv_cache.get(key)
    if cached:
        return cached
    path = _adm_model_config_path(str(model_id)) if model_id else None
    if path is not None:
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cfg = None
        if isinstance(cfg, dict):
            value = _adm_kv_from_config(cfg, element_bytes)
            if value:
                _adm_kv_cache[key] = value
                return value
    ram = TOTAL_RAM_GB if ram_gb is None else float(ram_gb)
    for floor, fp16_bytes in _ADM_FALLBACK_KV:
        if ram >= floor:
            return int(fp16_bytes * element_bytes / 2.0)
    return int(_ADM_FALLBACK_KV[-1][1] * element_bytes / 2.0)


def admission_cost_tokens(messages: list[dict], reply_tokens: int,
                          context_size: int = 0) -> int:
    """Tokens of KV one generation holds at its peak: prompt plus reply.

    Capped at the context window, because the request is trimmed to fit it
    (trim_to_context, reply_budget) before the backend sees it. Non-string
    content (image parts, content blocks) is measured by its text form, which
    over-counts and is therefore the safe direction for an admission check.
    """
    prompt = 0
    for message in messages or []:
        content = message.get("content", "") if isinstance(message, dict) else message
        if not isinstance(content, str):
            content = str(content)
        prompt += estimate_tokens(content) + 4
    total = prompt + max(0, int(reply_tokens or 0))
    if context_size and int(context_size) > 0:
        total = min(total, int(context_size))
    return max(0, total)


# --------------------------------------------------------------------------- #
# The controller
# --------------------------------------------------------------------------- #
class _AdmissionTicket:
    """One admitted generation. Released exactly once; stale after a loop rebind."""

    __slots__ = ("cost_tokens", "cost_bytes", "epoch", "released", "__weakref__")

    def __init__(self, cost_tokens: int, cost_bytes: int, epoch: int):
        self.cost_tokens = cost_tokens
        self.cost_bytes = cost_bytes
        self.epoch = epoch
        self.released = False


class _AdmissionWaiter:
    __slots__ = ("future", "cost_tokens", "cost_bytes", "on_wait", "position",
                 "enqueued_at", "memory_noted")

    def __init__(self, future, cost_tokens: int, cost_bytes: int, on_wait):
        self.future = future
        self.cost_tokens = cost_tokens
        self.cost_bytes = cost_bytes
        self.on_wait = on_wait
        self.position = 0
        self.enqueued_at = time.monotonic()
        self.memory_noted = False


class AdmissionController(GenerationGate):
    """GenerationGate's interface with KV-cost and free-memory awareness.

    Drop-in: acquire(), release(), slot(), `limit`, `max_waiting`, `waiting`
    and GenerationBusy behave as they do on GenerationGate, so every existing
    call site keeps working. It subclasses GenerationGate only so isinstance
    checks and type hints keep holding; none of the semaphore machinery is
    used (a semaphore cannot express "a slot is free but memory is not").

    acquire() returns a ticket and release(ticket) gives back exactly that
    reservation. release() with no argument (the GenerationGate form) releases
    the ticket the current asyncio task acquired, or failing that the oldest
    outstanding one: correct for the acquire-then-release-in-one-coroutine
    pattern, but a ticket-less release from a different task can return
    someone else's KV reservation early. Prefer slot() or explicit tickets.

    All state is mutated on the event loop thread. resize() and configure()
    may be called from a worker thread (the sync /api/config handler runs in
    one); they hand the re-dispatch to the loop with call_soon_threadsafe.
    """

    def __init__(self, limit: int, queue_multiplier: int = 4, *,
                 kv_bytes_per_token: int | Callable[[], int] = 0,
                 min_free_mb: int = 0, memory_aware: bool = True,
                 probe: Callable[[], int | None] | None = None,
                 recheck_seconds: float = 0.5, wait_samples: int = 256):
        # GenerationGate.__init__ is deliberately not called: `waiting` is a
        # property here (derived from the FIFO), not a counter to keep in sync.
        self.limit = max(1, int(limit))
        self.queue_multiplier = max(1, int(queue_multiplier))
        self.max_waiting = self.limit * self.queue_multiplier
        self.memory_aware = bool(memory_aware)
        self.min_free_mb = max(0, int(min_free_mb))
        self.recheck_seconds = max(0.01, float(recheck_seconds))
        self._kv = kv_bytes_per_token
        # None = the built-in psutil / vm_stat probe; tests pass a fake.
        self._probe = probe
        self._free_cached: int | None = None
        self._probe_source: str | None = None
        self.running = 0
        self.reserved_bytes = 0
        self.admitted_total = 0
        self.rejected_total = 0
        self.deferred_total = 0
        self._waiters: collections.deque[_AdmissionWaiter] = collections.deque()
        self._outstanding: dict[int, _AdmissionTicket] = {}
        self._owned: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._waits_ms: collections.deque[float] = collections.deque(
            maxlen=max(8, int(wait_samples)))
        self._recheck_handle: Any = None
        self._epoch = 0
        self._sem = None          # GenerationGate attribute, unused here
        self._loop: Any = None

    # ------------------------------------------------------------ state --

    @property
    def waiting(self) -> int:
        return len(self._waiters)

    @property
    def kv_bytes_per_token(self) -> int:
        value = self._kv
        try:
            if callable(value):
                value = value()
            return max(0, int(value or 0))
        except Exception:  # noqa: BLE001 - an estimator bug must not block admission
            return 0

    def _bind(self) -> asyncio.AbstractEventLoop:
        # Same rule as GenerationGate: futures belong to the loop that made
        # them, and tests start a fresh loop per app. State from a dead loop is
        # unreachable (its waiters can never be woken, its holders never
        # release on this loop), so a rebind starts clean and bumps the epoch
        # so a late release of an old ticket is a no-op, not a double free.
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._loop = loop
            self._epoch += 1
            self.running = 0
            self.reserved_bytes = 0
            self._waiters.clear()
            self._outstanding.clear()
            self._owned = weakref.WeakKeyDictionary()
            self._recheck_handle = None
        return loop

    def _free_bytes(self) -> int | None:
        try:
            if self._probe is not None:
                value, source = self._probe(), "custom"
            else:
                value, source = free_memory_probe()
        except Exception:  # noqa: BLE001
            value, source = None, None
        self._free_cached = None if value is None else int(value)
        self._probe_source = source if value is not None else None
        return self._free_cached

    def _memory_allows(self, cost_bytes: int) -> bool:
        if not self.memory_aware:
            return True
        free = self._free_bytes()
        if free is None:
            return True       # unmeasurable: memory awareness is off, not "full"
        # Running requests' reservations are subtracted even though part of
        # their cache is already resident (and so already missing from
        # `free`). Double-counting errs toward waiting, which on unified
        # memory is the cheap mistake; the opposite one is swap.
        projected = free - self.reserved_bytes - cost_bytes
        return projected >= self.min_free_mb * _ADM_MB

    def _admissible(self, cost_bytes: int) -> tuple[bool, str]:
        if self.running >= self.limit:
            return False, "slots"
        if self.running == 0:
            return True, ""   # a lone request is never refused for memory
        if not self._memory_allows(cost_bytes):
            return False, "memory"
        return True, ""

    def _admit(self, cost_tokens: int, cost_bytes: int,
               since: float | None) -> _AdmissionTicket:
        self.running += 1
        self.reserved_bytes += cost_bytes
        self.admitted_total += 1
        self._waits_ms.append(0.0 if since is None else (time.monotonic() - since) * 1000)
        ticket = _AdmissionTicket(cost_tokens, cost_bytes, self._epoch)
        self._outstanding[id(ticket)] = ticket
        return ticket

    def _own(self, ticket: _AdmissionTicket) -> None:
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if task is None:
            return
        held = [t for t in self._owned.get(task, ()) if not t.released]
        held.append(ticket)
        self._owned[task] = held

    def adopt(self, ticket: _AdmissionTicket | None) -> None:
        """Bind a ticket to the calling task for ticket-less release().

        A ticket acquired inside a helper task (the SSE route acquires in one
        so it can keep yielding queue updates) is owned by that helper; code
        later running in the main task that calls release() with no argument
        would otherwise fall back to the oldest outstanding ticket.
        """
        if ticket is not None and not ticket.released:
            self._own(ticket)

    def _owned_ticket(self) -> _AdmissionTicket | None:
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if task is not None:
            held = [t for t in self._owned.get(task, ()) if not t.released]
            if held:
                return held[-1]
        return next(iter(self._outstanding.values()), None)

    # --------------------------------------------------------- dispatch --

    def _snapshot_light(self) -> dict:
        """snapshot() without a fresh memory probe, for callbacks on the loop."""
        return self.snapshot(probe=False)

    def _notify_positions(self) -> None:
        snap = None
        for position, waiter in enumerate(self._waiters, 1):
            if waiter.position == position or waiter.future.done():
                continue
            waiter.position = position
            if waiter.on_wait is None:
                continue
            if snap is None:
                snap = self._snapshot_light()
            try:
                waiter.on_wait(position, snap)
            except Exception:  # noqa: BLE001 - a UI callback must not wedge the queue
                pass

    def _schedule_recheck(self) -> None:
        # Memory can come back without any release here (another process
        # exits, the backend trims its prompt cache), so a head blocked only on
        # memory is re-evaluated on a timer, not just on the next release.
        if self._recheck_handle is not None or self._loop is None:
            return
        try:
            self._recheck_handle = self._loop.call_later(self.recheck_seconds,
                                                         self._recheck)
        except RuntimeError:
            self._recheck_handle = None

    def _recheck(self) -> None:
        self._recheck_handle = None
        self._dispatch()

    def _dispatch(self) -> None:
        changed = False
        while self._waiters:
            head = self._waiters[0]
            if head.future.done():          # cancelled while queued
                self._waiters.popleft()
                changed = True
                continue
            ok, reason = self._admissible(head.cost_bytes)
            if not ok:
                if reason == "memory":
                    if not head.memory_noted:
                        head.memory_noted = True
                        self.deferred_total += 1
                        log_event(_adm_logger, logging.INFO, "admission.memory_deferred",
                                  running=self.running, limit=self.limit,
                                  cost_tokens=head.cost_tokens,
                                  free_mb=None if self._free_cached is None
                                  else round(self._free_cached / _ADM_MB),
                                  reserved_mb=round(self.reserved_bytes / _ADM_MB, 1),
                                  min_free_mb=self.min_free_mb)
                    self._schedule_recheck()
                break
            self._waiters.popleft()
            head.future.set_result(self._admit(head.cost_tokens, head.cost_bytes,
                                               head.enqueued_at))
            changed = True
        if changed:
            self._notify_positions()

    def _kick(self) -> None:
        """Re-run dispatch on the loop, from the loop or from any thread."""
        loop = self._loop
        if loop is None:
            return
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        try:
            if current is loop:
                self._dispatch()
            elif not loop.is_closed():
                loop.call_soon_threadsafe(self._dispatch)
        except RuntimeError:
            pass          # loop torn down under us; the next acquire rebinds

    # ------------------------------------------------------- public API --

    async def acquire(self, cost_tokens: int = 0,
                      on_wait: Callable[[int, dict], Any] | None = None) -> _AdmissionTicket:
        """Take one slot for a generation costing `cost_tokens` of KV.

        Waits (FIFO) while every slot is taken, or while starting now would
        leave less than min_free_mb free. Raises GenerationBusy instead of
        queueing when the waiting room is full. `on_wait(position, snapshot)`
        is called when the request is queued and whenever its 1-based position
        changes; it runs on the event loop and must not block.
        """
        loop = self._bind()
        cost_tokens = max(0, int(cost_tokens or 0))
        cost_bytes = cost_tokens * self.kv_bytes_per_token
        if not self._waiters:
            ok, _reason = self._admissible(cost_bytes)
            if ok:
                ticket = self._admit(cost_tokens, cost_bytes, None)
                self._own(ticket)
                return ticket
        if len(self._waiters) >= self.max_waiting:
            self.rejected_total += 1
            log_event(_adm_logger, logging.INFO, "admission.rejected",
                      running=self.running, waiting=len(self._waiters),
                      limit=self.limit, max_waiting=self.max_waiting,
                      cost_tokens=cost_tokens)
            raise GenerationBusy(
                f"{len(self._waiters)} generations already queued; try again shortly")
        waiter = _AdmissionWaiter(loop.create_future(), cost_tokens, cost_bytes, on_wait)
        self._waiters.append(waiter)
        self._dispatch()
        if not waiter.future.done():
            self._notify_positions()
        try:
            ticket = await waiter.future
        except asyncio.CancelledError:
            # Cancelled while queued (client gone), or in the instant between
            # being admitted and resuming: give back what was granted, else
            # leave the queue and let whoever is behind move up.
            if waiter.future.done() and not waiter.future.cancelled():
                self.release(waiter.future.result())
            else:
                with contextlib.suppress(ValueError):
                    self._waiters.remove(waiter)
                self._dispatch()
                self._notify_positions()
            raise
        self._own(ticket)
        return ticket

    def release(self, ticket: _AdmissionTicket | None = None) -> None:
        """Give a reservation back. Never raises, so it is safe in a finally.

        Idempotent per ticket; a ticket from before a loop rebind is ignored.
        """
        try:
            if ticket is None:
                ticket = self._owned_ticket()
            if ticket is None or ticket.released or ticket.epoch != self._epoch:
                return
            ticket.released = True
            self._outstanding.pop(id(ticket), None)
            self.running = max(0, self.running - 1)
            self.reserved_bytes = max(0, self.reserved_bytes - ticket.cost_bytes)
            self._kick()
        except Exception:  # noqa: BLE001
            pass

    @contextlib.asynccontextmanager
    async def slot(self, cost_tokens: int = 0,
                   on_wait: Callable[[int, dict], Any] | None = None):
        """Hold one generation slot for the duration of a block."""
        ticket = await self.acquire(cost_tokens=cost_tokens, on_wait=on_wait)
        try:
            yield ticket
        finally:
            self.release(ticket)

    @contextlib.asynccontextmanager
    async def suspend(self, ticket: _AdmissionTicket | None = None,
                      on_wait: Callable[[int, dict], Any] | None = None):
        """Give a held slot back for the duration of a block, then take it again.

        For a turn that waits on a human (tool approval): holding a slot while
        nobody computes starves everyone else. The SAME ticket object is
        re-armed on the way out, so the holder's own `finally: release(ticket)`
        stays correct. If the block raises, or re-acquiring does (queue full,
        cancelled), the ticket stays released and the holder's release is a
        harmless no-op.
        """
        if ticket is None:
            ticket = self._owned_ticket()
        if ticket is None or ticket.released:
            yield
            return
        cost_tokens = ticket.cost_tokens
        self.release(ticket)
        # Only a normal exit re-acquires. A block that raised (the turn was
        # cancelled or failed while waiting) is ending, and queueing for a slot
        # inside that unwind could replace the real error with GenerationBusy.
        yield
        fresh = await self.acquire(cost_tokens=cost_tokens, on_wait=on_wait)
        # Transfer the fresh reservation into the caller's handle.
        self._outstanding.pop(id(fresh), None)
        fresh.released = True
        ticket.cost_bytes, ticket.epoch, ticket.released = (
            fresh.cost_bytes, fresh.epoch, False)
        self._outstanding[id(ticket)] = ticket
        self._own(ticket)

    def resize(self, limit: int, queue_multiplier: int | None = None) -> None:
        """Change the slot count (and optionally the queue multiplier) live.

        Growing admits waiters at once. Shrinking never interrupts running
        generations: they finish, and new ones start only once running drops
        below the new limit. Waiters already queued past a smaller waiting room
        keep their place; only new arrivals are refused.
        """
        self.limit = max(1, int(limit))
        if queue_multiplier is not None:
            self.queue_multiplier = max(1, int(queue_multiplier))
        self.max_waiting = self.limit * self.queue_multiplier
        self._kick()

    def configure(self, *, memory_aware: bool | None = None,
                  min_free_mb: int | None = None,
                  kv_bytes_per_token: int | Callable[[], int] | None = None) -> None:
        """Adopt changed memory settings live (the /api/config path)."""
        if memory_aware is not None:
            self.memory_aware = bool(memory_aware)
        if min_free_mb is not None:
            self.min_free_mb = max(0, int(min_free_mb))
        if kv_bytes_per_token is not None:
            self._kv = kv_bytes_per_token
        self._kick()

    def snapshot(self, probe: bool = True) -> dict:
        """Queue state for /api/health and the cluster heartbeat.

        Safe to call from a worker thread (it only reads). `probe=False` uses
        the last memory reading instead of taking a new one.
        """
        free = self._free_bytes() if (probe and self.memory_aware) else self._free_cached
        try:
            waits = sorted(self._waits_ms)
        except RuntimeError:        # mutated mid-copy by the loop thread
            waits = []

        def pct(q: float) -> float:
            if not waits:
                return 0.0
            index = min(len(waits) - 1, max(0, int(round(q * (len(waits) - 1)))))
            return round(waits[index], 1)

        return {
            "running": self.running,
            "waiting": len(self._waiters),
            "limit": self.limit,
            "max_waiting": self.max_waiting,
            "queue_multiplier": self.queue_multiplier,
            "free_mb": None if free is None else round(free / _ADM_MB),
            "reserved_mb": round(self.reserved_bytes / _ADM_MB, 1),
            "min_free_mb": self.min_free_mb,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "admitted_total": self.admitted_total,
            "rejected_total": self.rejected_total,
            "deferred_total": self.deferred_total,
            "wait_ms_p50": pct(0.50),
            "wait_ms_p95": pct(0.95),
            # Effective, not configured: with no working probe the controller
            # is a plain slot gate, and the dashboard should say so.
            "memory_aware": bool(self.memory_aware and free is not None),
            "memory_probe": self._probe_source if self.memory_aware else None,
        }


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'AdmissionController',
    'admission_cost_tokens',
    'estimate_kv_bytes_per_token',
    'free_memory_bytes',
    'free_memory_probe',
    '_adm_parse_vm_stat',
]
