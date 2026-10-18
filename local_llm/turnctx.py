"""The per-turn context: what one chat turn is allowed and asked to do.

A turn used to be described only by the arguments threaded through
Agent.run_iterating, and every new concern (interrupting a turn, waiting for a
human to approve a tool call, asking consent before sending a prompt to a cloud
model, confining file tools to the caller's project) would have meant another
parameter on a dozen call sites, several of which tests replace with lambdas.

Instead the route that starts a turn sets one TurnContext in a contextvar, the
same mechanism set_acting_user already uses for data isolation. asyncio tasks
and asyncio.to_thread copy contextvars, so the agent loop, the tool handlers it
runs in worker threads and the model client all see the same object without a
signature change. Code that runs a turn without a route (scheduled tasks,
selftest, the CLI) simply has no context and gets today's behaviour.
"""

from __future__ import annotations

import contextvars
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TurnContext:
    """One chat turn's identity, controls and out-of-band channel."""

    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    user_id: str | None = None
    conversation_id: str | None = None
    role: str = "user"
    # True when a person is watching this turn's stream and can answer an
    # approval or consent prompt. Scheduled tasks and API clients that cannot
    # render a prompt run with interactive=False, so "ask" resolves to the
    # policy's unattended decision instead of waiting for nobody.
    interactive: bool = True
    # Set by the route; the agent loop and long tool waits check it.
    cancel: Any = None                      # asyncio.Event | None
    # Push an event into this turn's stream from anywhere (the agent loop,
    # a policy wait, the cloud-consent check). None when nothing listens.
    emit: Callable[[dict], None] | None = None
    # Async context manager factory that gives the generation slot back while
    # the turn waits on a human, then re-acquires it. None = keep the slot.
    suspend_admission: Callable[[], Any] | None = None
    agent_profile: dict | None = None        # the profile this turn runs as
    project: dict | None = None              # the workspace/project row, if any
    extras: dict = field(default_factory=dict)

    def cancelled(self) -> bool:
        return bool(self.cancel is not None and self.cancel.is_set())

    def send(self, event: dict) -> None:
        """Emit an out-of-band event; never raises into the caller."""
        if self.emit is None:
            return
        try:
            self.emit(event)
        except Exception:  # noqa: BLE001 - telemetry must not break a turn
            pass

    @asynccontextmanager
    async def waiting_on_human(self):
        """Release the generation slot (if the route allows) while waiting."""
        if self.suspend_admission is None:
            yield
            return
        async with self.suspend_admission():
            yield


_current_turn: contextvars.ContextVar = contextvars.ContextVar("current_turn", default=None)


def current_turn() -> TurnContext | None:
    """The TurnContext of the turn this code runs inside, or None."""
    return _current_turn.get()


def set_current_turn(turn: TurnContext | None):
    """Bind a turn for the current context; returns a token for reset."""
    return _current_turn.set(turn)


def reset_current_turn(token) -> None:
    try:
        _current_turn.reset(token)
    except (ValueError, RuntimeError):
        _current_turn.set(None)


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'TurnContext',
    'current_turn',
    'set_current_turn',
    'reset_current_turn',
]
