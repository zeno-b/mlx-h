"""The shared context handed to feature modules that register their own routes.

create_app() used to be the only place a route could live, because every route
is a closure over the objects it builds (config, db, the agent, the generation
gate, the auth dependencies). That made every new feature another few hundred
lines inside one function. A feature module now exports a
`register_<feature>(app, ctx)` function instead, and create_app calls it once
with this context at the end of app construction. The context carries exactly
the objects those closures used, so a feature route behaves like a built-in one:
same auth dependencies, same admission gate, same readiness check.

Rules for feature modules (they are bundled into one namespace by bundle.py):

  * prefix private top-level names with the feature (`_oai_...`), because the
    single-file build has no module boundaries to keep them apart;
  * declare Pydantic request models at module scope, never inside register_*:
    under `from __future__ import annotations` FastAPI resolves annotations
    against module globals, and a model it cannot find turns the body into a
    query parameter (every POST then answers 422). `body: dict` with explicit
    validation is the simplest safe choice;
  * use `ctx.USER` / `ctx.ADMIN` as dependencies, exactly like the built-ins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ApiContext:
    """Everything a feature route needs from create_app, by reference."""

    config: Any
    db: Any
    model_manager: Any
    retrain_manager: Any
    registry: Any
    model_client: Any
    agent: Any
    tasks: Any
    auth: Any
    router: Any
    node_registry: Any
    health_monitor: Any
    importer: Any
    generation_gate: Any
    # FastAPI dependency objects: Depends(auth.require_user / require_admin).
    USER: Any
    ADMIN: Any
    # Closure helpers from create_app, shared so feature routes make the same
    # decisions as /api/chat: readiness, ownership-checked history, per-request
    # generation overrides, and the capability-gated agent for a profile.
    not_ready: Callable[[], Any]
    load_history: Callable[..., Any]
    overrides: Callable[..., Any]
    agent_for_profile: Callable[[dict], Any]
    # A cluster-less client for work that must stay on this node.
    node_worker_client: Any = None
    # Free-form slot for features to publish objects other features read
    # (e.g. the turn registry for interrupt/resume, the approval broker).
    extras: dict = field(default_factory=dict)


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'ApiContext',
]
