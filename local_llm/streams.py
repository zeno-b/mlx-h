"""Chat turns that outlive their HTTP response: resume, cancel, reconnect.

A streaming chat turn used to BE its SSE response generator. That made three
things impossible. A dropped connection (a laptop lid, a proxy timeout, a page
reload) killed the turn mid-answer and threw the work away, because the only
code that could store the answer ran inside the generator Starlette had just
closed. Nothing could stop a turn except dropping the connection, so a "Stop"
that should keep the partial answer lost it instead. And a reloaded page had no
way to find the answer still being written for the conversation it showed.

So a turn now runs as its own asyncio task that pushes every event into a
bounded, sequence-numbered buffer (ChatTurn), and the HTTP response is only a
subscriber reading that buffer. Any number of subscribers can follow the same
turn from any point (GET /api/chat/turns/{id}/events?after=N), POST .../cancel
interrupts it cooperatively (and then for real, if the loop does not notice in
time), and GET /api/chat/active finds the running turn for a conversation.

Defaults keep the old wire behaviour exactly. Sequence ids are only written
when the client sends `X-Stream-Ids: 1` (the existing UI parser skips any line
that does not start with "data: ", but a client written before this module
should never see a frame shape it was not built for). A client that did not ask
for ids cannot resume, so its disconnect still cancels the turn at once, as it
always did; a client that did ask gets TURN_DETACH_GRACE_S seconds to come
back before the turn is cancelled and its partial answer stored.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import itertools
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Awaitable, Callable

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .llm import *  # noqa: F401,F403
from .turnctx import *  # noqa: F401,F403


# Per-turn buffer bound. A long answer is one event per token, so this holds a
# few thousand tokens of answer plus its tool and step events; past it the
# oldest events are dropped and a late subscriber is told about the gap
# instead of the buffer growing with the reply.
_STREAMS_MAX_EVENTS = 6000
# A finished turn stays readable this long, so a client that reconnects just
# after the final event can still fetch the tail it missed.
_STREAMS_RETAIN_S = 300.0
# Registry bound (finished turns are pruned first; running ones never are).
_STREAMS_MAX_TURNS = 256
# How long a cooperative cancel (the agent loop checking the turn's cancel
# event) gets before the turn's task is cancelled outright. A model call in
# prefill can take tens of seconds on an 8GB machine without yielding once.
_STREAMS_HARD_CANCEL_S = 5.0
# Idle time before a subscriber writes an SSE comment so a proxy does not cut
# a stream that is quiet because the turn waits on a human (tool approval).
_STREAMS_KEEPALIVE_S = 15.0

SSE_DONE = "data: [DONE]\n\n"


def sse_frame(event: dict, seq: int | None = None) -> str:
    """One SSE frame; `id:` line only when the client opted into ids."""
    data = f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    return data if seq is None else f"id: {seq}\n{data}"


def wants_stream_ids(headers: Any) -> bool:
    """True when the request carries `X-Stream-Ids: 1` (opt-in resume ids)."""
    try:
        value = headers.get("x-stream-ids") if headers is not None else None
    except Exception:  # noqa: BLE001
        value = None
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def optional_feature(module: str, name: str) -> Any:
    """`name` from a feature module another agent may or may not have shipped.

    The single-file build has one namespace, so the name is simply there when
    the module was bundled; in the package it is imported on demand. None when
    the feature is absent, which callers turn into a clear "not available".
    """
    found = globals().get(name)
    if found is not None:
        return found
    package = __package__ or ""
    if not package:
        return None
    try:
        mod = importlib.import_module(f"{package}.{module}")
    except ImportError:
        return None
    return getattr(mod, name, None)


def parse_message_meta(raw: Any) -> dict | None:
    """A stored message's meta column as a dict (None when absent or bad)."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def turn_message_meta(trace: list | None = None, reasoning: str = "",
                      tools: list | None = None, cancelled: bool = False) -> dict | None:
    """The meta stored with an assistant message, or None when it adds nothing.

    reasoning and tools are what the UI needs to replay a turn after a reload:
    the reasoning pane and the tool chips used to exist only in the live
    stream, so a reopened conversation showed bare answers.
    """
    meta: dict = {}
    if trace:
        meta["trace"] = trace
    if reasoning:
        meta["reasoning"] = reasoning
    if tools:
        meta["tools"] = list(tools)
    if cancelled:
        meta["cancelled"] = True
    return meta or None


def reasoning_of(text: str) -> str:
    """The <think> part of a complete reply (empty when there is none)."""
    if not text:
        return ""
    splitter = ThinkSplitter()
    pieces = list(splitter.feed(text)) + list(splitter.flush())
    return "".join(p for kind, p in pieces if kind == "think").strip()


def chat_role_capabilities(role: str, config: Any) -> set[str]:
    """Capability keys a role may use; every key when no role policy exists.

    The role policy itself is another feature's (a `capabilities_for_role(role,
    config)` hook, or a ROLE_CAPABILITIES JSON object {"user": [...]} on the
    config). Without either, a role is not narrowed, which is today's rule:
    the profile alone decides what its agent can do.
    """
    every = set(CAPABILITY_ORDER)
    hook = optional_feature("config", "capabilities_for_role")
    if callable(hook):
        try:
            got = hook(role, config)
        except Exception as exc:  # noqa: BLE001 - a broken policy must not 500 chat
            log(f"role capability hook failed: {exc}", logging.WARNING)
            got = None
        if got is not None:
            return {c for c in got if isinstance(c, str)} & every
    raw = getattr(config, "role_capabilities", "") or ""
    if isinstance(raw, str) and raw.strip():
        try:
            table = json.loads(raw)
        except ValueError:
            table = None
        if isinstance(table, dict) and role in table and isinstance(table[role], list):
            return {c for c in table[role] if isinstance(c, str)} & every
    return every


@dataclass
class ChatTurnOptions:
    """What a chat request asked for beyond the message, resolved and checked."""

    cfg: Any = None                 # the config this turn runs with
    agent: Any = None               # the agent that runs it
    client: Any = None              # the model client the plain lane uses
    profile: dict | None = None     # the agent profile, with effective capabilities
    project: dict | None = None
    attachments: list = field(default_factory=list)   # resolved content parts
    model: str = ""                 # the honoured model override ("" = served model)
    # Whether the answering model can see this turn's image parts (None: no
    # image parts, no decision); becomes turn.extras["vision"].
    vision: bool | None = None

    @property
    def forces_agent(self) -> bool:
        # A chosen agent profile IS an agent, and only the agent lane builds
        # multimodal content: the plain lane would ignore both silently.
        return bool(self.profile or self.attachments)


class ChatTurn:
    """One turn's event buffer, cancel switch and subscriber bookkeeping.

    Every method runs on the event loop that created the turn except emit(),
    which tool handlers call from worker threads.
    """

    def __init__(self, turn_id: str, user_id: str | None, conversation_id: str | None,
                 *, resumable: bool = False, detachable: bool = True,
                 interactive: bool = True, max_events: int = _STREAMS_MAX_EVENTS):
        self.turn_id = turn_id
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.resumable = resumable
        self.detachable = detachable
        self.interactive = interactive
        self.loop = asyncio.get_running_loop()
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.events: deque = deque(maxlen=max(16, int(max_events)))
        self.seq = 0
        self.done = False
        self.status = "running"          # running | done | cancelled | error
        self.cancel = asyncio.Event()
        self.cancel_reason = ""
        self.task: asyncio.Task | None = None
        self.subscribers = 0
        self.persisted = False           # the answer (or partial) is stored
        self.ended = False               # a final or cancelled event was pushed
        self._changed = asyncio.Event()
        self._detach_handle: asyncio.TimerHandle | None = None
        self._hard_handle: asyncio.TimerHandle | None = None
        self._partial: list[str] = []
        self._reasoning: list[str] = []

    # ------------------------------------------------------------ events --
    def push(self, event: dict) -> int:
        """Append one event; returns its seq (0 once the turn has finished)."""
        if self.done or not isinstance(event, dict):
            return 0
        kind = event.get("type")
        if kind == "token":
            self._partial.append(str(event.get("token") or event.get("text") or ""))
        elif kind == "think_token":
            self._reasoning.append(str(event.get("token") or event.get("text") or ""))
        elif kind in ("cancelled", "error") and self.status == "running":
            self.status = "cancelled" if kind == "cancelled" else "error"
        if kind in ("final", "cancelled"):
            self.ended = True
        self.seq += 1
        self.events.append((self.seq, event))
        self._wake()
        return self.seq

    def emit(self, event: dict) -> None:
        """push() from any thread (TurnContext.emit points here)."""
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self.loop:
            self.push(event)
            return
        try:
            self.loop.call_soon_threadsafe(self.push, event)
        except RuntimeError:
            pass  # loop closed: the turn is gone, nobody is listening

    def _wake(self) -> None:
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()

    def first_seq(self) -> int:
        return self.events[0][0] if self.events else self.seq + 1

    def events_after(self, cursor: int) -> list:
        if not self.events or cursor >= self.seq:
            return []
        start = max(0, cursor - self.events[0][0] + 1)
        return list(itertools.islice(self.events, start, None))

    def partial_text(self) -> str:
        return "".join(self._partial)

    def reasoning_text(self) -> str:
        return "".join(self._reasoning).strip()

    def finish(self, status: str | None = None) -> None:
        if self.done:
            return
        if status:
            self.status = status
        elif self.status == "running":
            self.status = "done"
        self.done = True
        self.finished_at = time.time()
        for handle in (self._detach_handle, self._hard_handle):
            if handle is not None:
                handle.cancel()
        self._detach_handle = self._hard_handle = None
        self._wake()

    # -------------------------------------------------------- cancelling --
    def request_cancel(self, reason: str = "cancelled",
                       hard_after: float | None = None) -> bool:
        """Ask the turn to stop; cancel its task outright after `hard_after`
        seconds (default _STREAMS_HARD_CANCEL_S, read at call time)."""
        if self.done:
            return False
        if hard_after is None:
            hard_after = _STREAMS_HARD_CANCEL_S
        if not self.cancel_reason:
            self.cancel_reason = reason
        self.cancel.set()
        if hard_after <= 0:
            self._hard_cancel()
        elif self._hard_handle is None and self.task is not None:
            self._hard_handle = self.loop.call_later(hard_after, self._hard_cancel)
        log_event(get_logger("chat"), logging.INFO, "chat.cancel_requested",
                  turn_id=self.turn_id, conversation_id=self.conversation_id,
                  user_id=self.user_id, reason=reason)
        return True

    def _hard_cancel(self) -> None:
        self._hard_handle = None
        if self.task is not None and not self.task.done():
            self.task.cancel()

    # ------------------------------------------------------- subscribers --
    def attach(self, ids: bool = False) -> None:
        self.subscribers += 1
        # A subscriber that reads ids can come back after a drop, so from now
        # on a disconnect gets the grace period instead of an instant cancel.
        self.resumable = self.resumable or bool(ids)
        if self._detach_handle is not None:
            self._detach_handle.cancel()
            self._detach_handle = None

    def detach(self, grace: float = 0.0) -> None:
        self.subscribers = max(0, self.subscribers - 1)
        if self.done or self.subscribers or not self.detachable:
            return
        if not self.resumable or grace <= 0:
            # Nobody can resume this turn: the old rule, stop it now.
            self.request_cancel("client disconnected", hard_after=0)
            return
        if self._detach_handle is None:
            self._detach_handle = self.loop.call_later(grace, self._detach_expired)

    def _detach_expired(self) -> None:
        self._detach_handle = None
        if not self.done and not self.subscribers:
            self.request_cancel("client did not reconnect")

    def snapshot(self) -> dict:
        return {"turn_id": self.turn_id, "conversation_id": self.conversation_id,
                "status": self.status, "seq": self.seq, "done": self.done,
                "started_at": self.started_at, "finished_at": self.finished_at,
                "resumable": self.resumable, "interactive": self.interactive,
                "subscribers": self.subscribers,
                "cancel_requested": self.cancel.is_set()}


class TurnRegistry:
    """Every live (and recently finished) chat turn, by id."""

    def __init__(self, config: Any, *, max_turns: int = _STREAMS_MAX_TURNS,
                 retain_s: float = _STREAMS_RETAIN_S, max_events: int = _STREAMS_MAX_EVENTS):
        self.config = config
        self.max_turns = max_turns
        self.retain_s = retain_s
        self.max_events = max_events
        self._turns: dict[str, ChatTurn] = {}

    def grace(self) -> float:
        """TURN_DETACH_GRACE_S, read live (it is a runtime setting)."""
        try:
            return max(0.0, float(getattr(self.config, "turn_detach_grace_s", 0.0) or 0.0))
        except (TypeError, ValueError):
            return 0.0

    def create(self, user_id: str | None, conversation_id: str | None, **kw: Any) -> ChatTurn:
        self.prune()
        turn = ChatTurn(uuid.uuid4().hex[:16], user_id, conversation_id,
                        max_events=self.max_events, **kw)
        self._turns[turn.turn_id] = turn
        return turn

    def get(self, turn_id: str, user_id: str | None) -> ChatTurn | None:
        """The turn, only for its owner. Admins included: a turn's stream is
        the user's conversation, and isolation is not an admin privilege."""
        turn = self._turns.get(str(turn_id or ""))
        if turn is None or turn.user_id != user_id:
            return None
        return turn

    def active_for(self, user_id: str | None, conversation_id: str | None = None) -> list[ChatTurn]:
        found = [t for t in self._turns.values()
                 if t.user_id == user_id and not t.done
                 and (not conversation_id or t.conversation_id == conversation_id)]
        return sorted(found, key=lambda t: t.started_at, reverse=True)

    def prune(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        for tid, turn in list(self._turns.items()):
            if turn.done and (turn.finished_at or 0) + self.retain_s < now:
                del self._turns[tid]
        if len(self._turns) >= self.max_turns:
            finished = sorted((t for t in self._turns.values() if t.done),
                              key=lambda t: t.finished_at or 0)
            for turn in finished[:len(self._turns) - self.max_turns + 1]:
                self._turns.pop(turn.turn_id, None)

    def __len__(self) -> int:
        return len(self._turns)


async def drive_turn(turn: ChatTurn, events: AsyncGenerator[dict, None],
                     on_hard_cancel: Callable[[], Awaitable[dict | None]] | None = None) -> None:
    """Pump a turn's event generator into its buffer until it ends.

    Owns the end of the turn: a task cancellation (the hard half of a cancel,
    or the app shutting down) still stores the partial answer through
    on_hard_cancel, and the turn is always finished so subscribers end.
    """
    try:
        async for event in events:
            turn.push(event)
    except asyncio.CancelledError:
        task = asyncio.current_task()
        if task is not None:
            task.uncancel()
        if on_hard_cancel is not None:
            try:
                extra = await on_hard_cancel()
            except Exception as exc:  # noqa: BLE001
                log(f"storing a cancelled turn failed: {exc}", logging.WARNING)
                extra = None
            if extra:
                turn.push(extra)
        turn.finish("cancelled")
    except Exception as exc:  # noqa: BLE001 - the stream must end with an event
        log(f"chat turn {turn.turn_id} failed: {type(exc).__name__}: {exc}", logging.ERROR)
        turn.push({"type": "error", "error": redact_text(str(exc))})
        turn.finish("error")
    finally:
        turn.finish()
        # A hard-cancelled turn dies inside its approval wait without
        # answering it; left pending, the prompt would sit in
        # /api/approvals/pending until its timeout for a turn that is gone.
        broker = optional_feature("policy", "BROKER")
        cancel_pending = getattr(broker, "cancel_turn", None)
        if callable(cancel_pending):
            try:
                cancel_pending(turn.turn_id)
            except Exception as exc:  # noqa: BLE001 - the turn has ended anyway
                log(f"clearing approvals of turn {turn.turn_id} failed: {exc}",
                    logging.DEBUG)


async def turn_frames(turn: ChatTurn, after: int = 0, *, ids: bool = False,
                      grace: Callable[[], float] | None = None,
                      keepalive: float = _STREAMS_KEEPALIVE_S) -> AsyncGenerator[str, None]:
    """SSE frames for one subscriber, from seq `after` on, ending with [DONE]."""
    turn.attach(ids)
    try:
        cursor = max(0, int(after or 0))
        while True:
            first = turn.first_seq()
            if cursor < first - 1:
                yield sse_frame({"type": "notice", "info": True,
                                 "message": f"{first - 1 - cursor} earlier stream events "
                                            "are no longer buffered"})
                cursor = first - 1
            batch = turn.events_after(cursor)
            for seq, event in batch:
                cursor = seq
                yield sse_frame(event, seq if ids else None)
            if turn.done and cursor >= turn.seq:
                break
            if batch:
                continue
            changed = turn._changed
            try:
                await asyncio.wait_for(changed.wait(), timeout=keepalive)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
        yield SSE_DONE
    finally:
        turn.detach(grace() if grace is not None else 0.0)


_STREAMS_DECISIONS = ("allow", "deny")
_STREAMS_APPROVAL_SCOPES = ("once", "conversation", "tool")
_STREAMS_CONSENT_SCOPES = ("once", "conversation", "always")


def register_streams(app: Any, ctx: Any) -> None:
    """Turn routes: resume, cancel, find the running turn, answer prompts."""
    from fastapi import Header
    from fastapi.responses import JSONResponse, StreamingResponse

    turns = getattr(app.state, "turns", None)
    if turns is None:
        turns = TurnRegistry(ctx.config)
        app.state.turns = turns
    ctx.extras["turns"] = turns
    USER = ctx.USER

    def not_found() -> Any:
        return JSONResponse({"error": "not found"}, status_code=404)

    def decision_of(body: Any, scopes: tuple) -> tuple[str, str] | Any:
        if not isinstance(body, dict):
            return JSONResponse({"error": "a JSON object is required"}, status_code=400)
        decision = str(body.get("decision") or "").strip().lower()
        scope = str(body.get("scope") or "once").strip().lower()
        if decision not in _STREAMS_DECISIONS:
            return JSONResponse({"error": "decision must be allow or deny"}, status_code=400)
        if scope not in scopes:
            return JSONResponse({"error": f"scope must be one of {', '.join(scopes)}"},
                                status_code=400)
        return decision, scope

    @app.get("/api/chat/active")
    async def chat_active(conversation_id: str = "", user: dict = USER):
        running = turns.active_for(user["id"], conversation_id or None)
        return {"turn": running[0].snapshot() if running else None,
                "turns": [t.snapshot() for t in running]}

    @app.get("/api/chat/turns/{turn_id}/events")
    async def chat_turn_events(turn_id: str, after: int = 0,
                               x_stream_ids: str | None = Header(None),
                               last_event_id: str | None = Header(None),
                               user: dict = USER):
        turn = turns.get(turn_id, user["id"])
        if turn is None:
            return not_found()
        # The browser's EventSource reconnects with Last-Event-ID; honour it
        # like ?after= so either kind of client resumes where it stopped.
        try:
            after = max(int(after or 0), int(last_event_id or 0))
        except ValueError:
            after = int(after or 0)
        ids = wants_stream_ids({"x-stream-ids": x_stream_ids})
        return StreamingResponse(
            turn_frames(turn, after, ids=ids, grace=turns.grace),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"})

    @app.post("/api/chat/turns/{turn_id}/cancel")
    async def chat_turn_cancel(turn_id: str, user: dict = USER):
        turn = turns.get(turn_id, user["id"])
        if turn is None:
            return not_found()
        accepted = turn.request_cancel("cancelled by the user")
        return {"cancelled": accepted, "turn": turn.snapshot()}

    @app.post("/api/chat/turns/{turn_id}/approvals/{approval_id}")
    async def chat_turn_approval(turn_id: str, approval_id: str, body: dict,
                                 user: dict = USER):
        turn = turns.get(turn_id, user["id"])
        if turn is None:
            return not_found()
        parsed = decision_of(body, _STREAMS_APPROVAL_SCOPES)
        if not isinstance(parsed, tuple):
            return parsed
        decision, scope = parsed
        args = body.get("args")
        if args is not None and not isinstance(args, dict):
            return JSONResponse({"error": "args must be an object"}, status_code=400)
        broker = optional_feature("policy", "BROKER")
        if broker is None:
            return JSONResponse({"error": "approvals not available"}, status_code=404)
        extra: dict = {"args": args}
        try:
            # A broker that can also check the approval belongs to THIS turn
            # gets the turn id, so an id from another turn of the same user
            # cannot be answered through this one's URL.
            if "turn_id" in inspect.signature(broker.resolve).parameters:
                extra["turn_id"] = turn.turn_id
        except (TypeError, ValueError):
            pass
        try:
            ok = broker.resolve(approval_id, user["id"], decision, scope, **extra)
        except Exception as exc:  # noqa: BLE001 - ValueError: a malformed answer
            return JSONResponse({"error": redact_text(str(exc))}, status_code=400)
        if not ok:
            return JSONResponse({"error": "approval not found or already resolved"},
                                status_code=404)
        return {"resolved": True, "decision": decision, "scope": scope}

    @app.post("/api/chat/turns/{turn_id}/consent/{consent_id}")
    async def chat_turn_consent(turn_id: str, consent_id: str, body: dict,
                                user: dict = USER):
        turn = turns.get(turn_id, user["id"])
        if turn is None:
            return not_found()
        parsed = decision_of(body, _STREAMS_CONSENT_SCOPES)
        if not isinstance(parsed, tuple):
            return parsed
        decision, scope = parsed
        # The cloud module owns consent; either a resolve_consent function or
        # a CONSENT broker with the approval broker's resolve signature.
        resolver = optional_feature("cloud", "resolve_consent")
        if resolver is None:
            broker = optional_feature("cloud", "CONSENT")
            resolver = getattr(broker, "resolve", None) if broker is not None else None
        if not callable(resolver):
            return JSONResponse({"error": "consent not available"}, status_code=404)
        try:
            ok = resolver(consent_id, user["id"], decision, scope)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": redact_text(str(exc))}, status_code=400)
        if not ok:
            return JSONResponse({"error": "consent request not found or already resolved"},
                                status_code=404)
        return {"resolved": True, "decision": decision, "scope": scope}


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'SSE_DONE',
    'sse_frame',
    'wants_stream_ids',
    'optional_feature',
    'parse_message_meta',
    'turn_message_meta',
    'reasoning_of',
    'chat_role_capabilities',
    'ChatTurnOptions',
    'ChatTurn',
    'TurnRegistry',
    'drive_turn',
    'turn_frames',
    'register_streams',
    '_STREAMS_MAX_EVENTS',
    '_STREAMS_RETAIN_S',
    '_STREAMS_MAX_TURNS',
    '_STREAMS_HARD_CANCEL_S',
    '_STREAMS_KEEPALIVE_S',
]
