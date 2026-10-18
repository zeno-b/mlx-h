"""Workspaces and projects: whose files a turn's tools may touch.

Until now there was ONE file root per installation: PROJECT_DIR, or the
./workspace sandbox when none was set. That was the right model for one person
on one Mac, and the wrong one the moment a second account existed: every user's
read_file, write_file, create_document and the download endpoint all pointed at
the same directory, so one user's generated report was another user's to
overwrite, and "the workspace" leaked between people who never chose to share.

This module adds two things on top of that single root, both off by default:

  * a per-user default root, WORKSPACE_DIR/users/<user_id>, created on first use
    when WORKSPACES_ENABLED=1. Administrators keep the configured PROJECT_DIR
    (POST /api/config {"project_dir"} stays the admin default project);
  * projects: a named root a turn can be confined to. A user creates projects
    only under their own root; an administrator creates shared ones under one
    of WORKSPACE_ROOTS. Access is the owner, explicit members, or everyone for
    a shared project, and resolve_project() is the one place that decides.

The root a tool sees is resolved per CONTEXT (contextvars), never stored on the
shared ToolRegistry, because the registry is one object serving every
concurrent request. Resolution order, first match wins:

  1. a root pinned with bind_workspace_root() (the execute_code bridge, a
     download for a project, the context preview);
  2. the project on the current TurnContext (the chat route binds it);
  3. WORKSPACES_ENABLED=1 and a non-admin acting user: their own root;
  4. nothing: ToolRegistry._root() falls back to today's single root.

The same module holds the per-turn half of the registry's state
(skills_used, documents_written, changed_files, checkpoint_id,
conversation_id). Those were plain attributes on the shared registry, so two
users' concurrent turns wrote into the same list: one user's final event could
offer the other user's document as a download, and one turn's checkpoint id
could swallow the other turn's writes. They now live on the turn (or an
explicit scope), keyed by registry so a sub-agent's child registry resetting
its own lists cannot wipe the parent's.
"""

from __future__ import annotations

import contextvars
import logging
import re
import threading
import time
import uuid
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .database import *  # noqa: F401,F403
from .turnctx import *  # noqa: F401,F403

_ws_log = get_logger("workspace")

# Roles a project member can hold. "manager" may rename the project and manage
# its members; "member" may work in it. The owner is the projects row's
# owner_user_id and is never stored as a member.
PROJECT_ROLES = ("member", "manager")


# --------------------------------------------------------------------------- #
# Schema                                                                       #
# --------------------------------------------------------------------------- #
def _ws_schema(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            root_path TEXT NOT NULL,
            owner_user_id TEXT NOT NULL,
            shared INTEGER NOT NULL DEFAULT 0,
            rag_scope TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_projects_owner "
                 "ON projects(owner_user_id)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS project_members (
            project_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'member',
            PRIMARY KEY (project_id, user_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_project_members_user "
                 "ON project_members(user_id)")


register_schema("workspaces", _ws_schema)


# --------------------------------------------------------------------------- #
# Per-turn registry state                                                      #
# --------------------------------------------------------------------------- #
class WorkspaceTurnState:
    """The registry attributes that belong to ONE turn, not to the process."""

    def __init__(self) -> None:
        self.skills_used: list[str] = []
        self.documents_written: list[str] = []
        self.changed_files: set[str] = set()
        self.checkpoint_id: str | None = None
        self.conversation_id: str | None = None


# An explicit scope (bind_turn_scope) wins over the TurnContext. Code that runs
# a turn without a route (a scheduled task, a test driving two turns by hand)
# can isolate it without constructing a TurnContext.
_ws_scope_var: contextvars.ContextVar = contextvars.ContextVar(
    "workspace_turn_scope", default=None)
_ws_state_lock = threading.Lock()


def bind_turn_scope():
    """Start a fresh per-turn registry state for this context; returns a token."""
    return _ws_scope_var.set(weakref.WeakKeyDictionary())


def reset_turn_scope(token) -> None:
    try:
        _ws_scope_var.reset(token)
    except (ValueError, RuntimeError):
        _ws_scope_var.set(None)


@contextmanager
def workspace_turn_scope():
    """`with workspace_turn_scope():` isolates one turn's registry state."""
    token = bind_turn_scope()
    try:
        yield
    finally:
        reset_turn_scope(token)


def registry_turn_state(registry: Any) -> WorkspaceTurnState | None:
    """The per-turn state of `registry` in this context, or None outside a turn.

    Keyed by registry (weakly) inside the turn's holder: a delegate_task child
    agent builds its own ToolRegistry and resets ITS lists at the start of its
    run, and a holder keyed only by turn would have wiped the parent's.
    """
    holder = _ws_scope_var.get()
    if holder is None:
        turn = current_turn()
        if turn is None:
            return None
        extras = getattr(turn, "extras", None)
        if not isinstance(extras, dict):
            return None
        holder = extras.get("_ws_states")
        if holder is None:
            with _ws_state_lock:
                holder = extras.setdefault("_ws_states", weakref.WeakKeyDictionary())
    try:
        state = holder.get(registry)
    except TypeError:
        return None                    # an unhashable stand-in: no turn scoping
    if state is None:
        with _ws_state_lock:
            state = holder.get(registry)
            if state is None:
                state = WorkspaceTurnState()
                holder[registry] = state
    return state


# --------------------------------------------------------------------------- #
# Roots                                                                        #
# --------------------------------------------------------------------------- #
_ws_root_var: contextvars.ContextVar = contextvars.ContextVar(
    "workspace_acting_root", default=None)


def bind_workspace_root(root: Path | str | None):
    """Pin the file root for this context (None unpins); returns a token."""
    return _ws_root_var.set(Path(root).resolve() if root else None)


def reset_workspace_root(token) -> None:
    try:
        _ws_root_var.reset(token)
    except (ValueError, RuntimeError):
        _ws_root_var.set(None)


def acting_workspace_root() -> Path | None:
    """The root pinned with bind_workspace_root() in this context, if any."""
    return _ws_root_var.get()


@contextmanager
def workspace_root_bound(root: Path | str | None):
    token = bind_workspace_root(root)
    try:
        yield
    finally:
        reset_workspace_root(token)


def path_inside(path: Path, parent: Path) -> bool:
    """True when `path` is `parent` or below it (both already resolved)."""
    return path == parent or parent in path.parents


def _ws_safe_segment(user_id: str) -> str:
    """A directory name for a user id that cannot traverse and cannot collide.

    Ids are hex today, but OIDC subjects and imported ids are not ours to
    trust: anything that needed rewriting gets a hash suffix, so "a.b" and
    "a_b" never share a directory.
    """
    raw = str(user_id or "")
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", raw)[:64] or "user"
    if clean != raw:
        import hashlib
        clean = f"{clean}-{hashlib.sha256(raw.encode()).hexdigest()[:8]}"
    return clean


def user_workspace_root(user_id: str, create: bool = True) -> Path:
    """WORKSPACE_DIR/users/<user_id>, created on demand."""
    root = (WORKSPACE_DIR / "users" / _ws_safe_segment(user_id))
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def workspace_allowed_parents(config: Config) -> list[Path]:
    """Directories an administrator may put a shared project's root under.

    WORKSPACE_ROOTS (comma list) when set; otherwise the workspace directory
    and PROJECT_DIR, which are the two places a root could already be today.
    """
    raw = [p.strip() for p in str(getattr(config, "workspace_roots", "") or "").split(",")
           if p.strip()]
    if not raw:
        raw = [str(WORKSPACE_DIR)]
        if (config.project_dir or "").strip():
            raw.append(config.project_dir.strip())
    out: list[Path] = []
    for item in raw:
        try:
            resolved = Path(item).expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        if resolved not in out:
            out.append(resolved)
    return out


_ws_role_cache: dict[str, tuple[str, float]] = {}


def _ws_role_of(db: Any, user_id: str) -> str:
    """The acting user's role, cheaply: _root() runs many times per turn."""
    if user_id == SENTINEL_LOCAL_USER:
        return "admin"                 # auth disabled: the implicit administrator
    turn = current_turn()
    if turn is not None and getattr(turn, "user_id", None) == user_id and turn.role:
        return str(turn.role)
    cached = _ws_role_cache.get(user_id)
    now = time.monotonic()
    if cached and now - cached[1] < 30.0:
        return cached[0]
    role = "user"
    if db is not None:
        try:
            row = db.get_user(user_id)
            role = str((row or {}).get("role") or "user")
        except Exception:  # noqa: BLE001 - fail towards the confined root
            role = "user"
    if len(_ws_role_cache) > 4096:
        _ws_role_cache.clear()
    _ws_role_cache[user_id] = (role, now)
    return role


def _ws_project_root(project: dict | None) -> Path | None:
    if not isinstance(project, dict):
        return None
    raw = project.get("root") or project.get("root_path")
    if not raw:
        return None
    root = Path(str(raw)).expanduser()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass          # reported by the tool that tries to use it
    return root.resolve()


def workspace_root_for(config: Config, db: Any = None) -> Path | None:
    """The root this context is confined to, or None for today's single root."""
    pinned = _ws_root_var.get()
    if pinned is not None:
        return pinned
    turn = current_turn()
    if turn is not None:
        root = _ws_project_root(getattr(turn, "project", None))
        if root is not None:
            return root
    if not getattr(config, "workspaces_enabled", False):
        return None
    user_id = get_acting_user()
    if not user_id or _ws_role_of(db, user_id) == "admin":
        return None
    return user_workspace_root(user_id)


def default_root_for_user(config: Config, user: dict) -> Path | None:
    """A user's root with no project selected; None = the installation's root."""
    if not getattr(config, "workspaces_enabled", False):
        return None
    if user.get("role") == "admin" or user.get("id") == SENTINEL_LOCAL_USER:
        return None
    return user_workspace_root(user["id"])


def exec_isolation_warning(config: Config) -> str | None:
    """The honest caveat when per-user roots meet the local exec backend.

    Per-user roots confine the FILE tools. run_shell, run_python and
    execute_code on EXEC_BACKEND=local are subprocesses with the server's own
    privileges and can read any user's directory; only the docker backend
    (which mounts the acting root alone) is a boundary.
    """
    if not getattr(config, "workspaces_enabled", False):
        return None
    if not (config.allow_shell or config.allow_python):
        return None
    if (config.exec_backend or "local").lower() != "local":
        return None
    return ("WORKSPACES_ENABLED=1 with code execution on EXEC_BACKEND=local: local "
            "exec is NOT a security boundary. Shell and Python run with the "
            "server's privileges and can read every user's workspace. Use "
            "EXEC_BACKEND=docker, which mounts only the acting project root, or "
            "turn shell/Python off for multi-user installs.")


# --------------------------------------------------------------------------- #
# Projects                                                                     #
# --------------------------------------------------------------------------- #
def _ws_row(row: Any) -> dict:
    item = dict(row)
    item["shared"] = bool(item.get("shared"))
    item["root"] = item.get("root_path")
    return item


def _ws_user_id(user: dict | str | None) -> str | None:
    if isinstance(user, dict):
        return user.get("id")
    return user


def _ws_is_admin(db: Any, user: dict | str | None) -> bool:
    if isinstance(user, dict):
        return user.get("role") == "admin" or user.get("id") == SENTINEL_LOCAL_USER
    if not user:
        return False
    return _ws_role_of(db, str(user)) == "admin"


def get_project(db: Any, project_id: str) -> dict | None:
    db.ensure_schema("workspaces")
    row = db.execute("SELECT * FROM projects WHERE id = ?", (str(project_id),)).fetchone()
    return _ws_row(row) if row else None


def project_members(db: Any, project_id: str) -> list[dict]:
    db.ensure_schema("workspaces")
    rows = db.execute(
        "SELECT pm.user_id AS user_id, pm.role AS role, u.username AS username "
        "FROM project_members pm LEFT JOIN users u ON u.id = pm.user_id "
        "WHERE pm.project_id = ? ORDER BY u.username", (str(project_id),)).fetchall()
    return [dict(r) for r in rows]


def project_access(db: Any, user: dict | str | None, project: dict) -> str | None:
    """How `user` may use `project`: admin, owner, manager, member, or None."""
    uid = _ws_user_id(user)
    if _ws_is_admin(db, user):
        return "admin"
    if not uid:
        return None
    if project.get("owner_user_id") == uid:
        return "owner"
    row = db.execute("SELECT role FROM project_members WHERE project_id = ? AND user_id = ?",
                     (project["id"], uid)).fetchone()
    if row:
        return str(row["role"] or "member")
    if project.get("shared"):
        return "member"
    return None


def resolve_project(db: Any, user: dict | str | None, project_id: str | None) -> dict | None:
    """The project row a request may work in, with "root" and "access" set.

    None when no project was asked for (today's behaviour). LookupError when it
    does not exist, PermissionError when the caller is not allowed in: routes
    answer both with 404, so a project id does not confirm its own existence
    to someone outside it.
    """
    if project_id is None or str(project_id).strip() == "":
        return None
    project = get_project(db, str(project_id).strip())
    if project is None:
        raise LookupError(f"no project {project_id!r}")
    access = project_access(db, user, project)
    if access is None:
        raise PermissionError(f"no access to project {project_id!r}")
    project["access"] = access
    return project


def list_projects(db: Any, user: dict | str | None) -> list[dict]:
    """Projects the user can open: all for an admin; else owned, joined, shared."""
    db.ensure_schema("workspaces")
    uid = _ws_user_id(user)
    if _ws_is_admin(db, user):
        rows = db.execute("SELECT * FROM projects ORDER BY created_at, name").fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM projects WHERE owner_user_id = ? OR shared = 1 OR id IN "
            "(SELECT project_id FROM project_members WHERE user_id = ?) "
            "ORDER BY created_at, name", (uid, uid)).fetchall()
    out = []
    for row in rows:
        item = _ws_row(row)
        item["access"] = project_access(db, user, item)
        out.append(item)
    return out


def _ws_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(name or "")).strip("-")[:48].lower()


def _ws_clean_name(name: Any) -> str:
    text = re.sub(r"\s+", " ", str(name or "")).strip()
    if not text:
        raise ValueError("name must not be empty")
    if len(text) > 80:
        raise ValueError("name must be at most 80 characters")
    return text


def _ws_clean_scope(scope: Any) -> str:
    return ",".join(p.strip() for p in str(scope or "").split(",") if p.strip())[:4000]


def create_project(db: Any, config: Config, user: dict, name: str,
                   root_path: str | None = None, shared: bool = False,
                   rag_scope: str = "") -> dict:
    """Create a project. ValueError for a bad name or a root outside the rules.

    An administrator's root must sit under WORKSPACE_ROOTS (default: the
    workspace directory and PROJECT_DIR); everyone else's sits under their own
    WORKSPACE_DIR/users/<id>. A relative root is taken relative to that base,
    so a user writes "thesis" and never has to know where their root lives.
    """
    db.ensure_schema("workspaces")
    clean_name = _ws_clean_name(name)
    admin = _ws_is_admin(db, user)
    if shared and not admin:
        raise PermissionError("only administrators create shared projects")
    project_id = uuid.uuid4().hex[:12]
    if admin:
        parents = workspace_allowed_parents(config)
        base = parents[0] if parents else WORKSPACE_DIR.resolve()
        default = (WORKSPACE_DIR / "projects" / (_ws_slug(clean_name) or project_id))
    else:
        base = user_workspace_root(user["id"])
        parents = [base]
        default = base / (_ws_slug(clean_name) or project_id)
    raw = str(root_path or "").strip()
    candidate = Path(raw).expanduser() if raw else default
    if not candidate.is_absolute():
        candidate = base / candidate
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"unusable root: {exc}") from exc
    if not any(path_inside(resolved, parent) for parent in parents):
        where = ", ".join(str(p) for p in parents)
        raise ValueError(f"project root must be inside {where}")
    if ".git" in resolved.parts:
        raise ValueError("a project root cannot be inside a .git directory")
    # WORKSPACE_DIR is an allowed parent by default and CONTAINS everyone's
    # users/<id>: a shared project rooted there would publish one user's files
    # to every account. Their own projects are created by them, not for them.
    if admin and path_inside(resolved, (WORKSPACE_DIR / "users").resolve()):
        raise ValueError("users/ holds per-user workspaces; a shared project "
                         "cannot be rooted inside one")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"{resolved} exists and is not a directory")
    # Two of one user's projects in one directory would be one project with two
    # names; the second gets its id appended instead of silently aliasing.
    if not raw and db.execute("SELECT 1 FROM projects WHERE root_path = ?",
                              (str(resolved),)).fetchone():
        resolved = resolved.with_name(f"{resolved.name}-{project_id[:6]}")
    resolved.mkdir(parents=True, exist_ok=True)
    db.execute(
        "INSERT INTO projects (id, name, root_path, owner_user_id, shared, rag_scope) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (project_id, clean_name, str(resolved), user["id"], 1 if shared else 0,
         _ws_clean_scope(rag_scope)))
    db.commit()
    log_event(_ws_log, logging.INFO, "workspace.project_created", project_id=project_id,
              user_id=user["id"], shared=bool(shared))
    project = get_project(db, project_id) or {}
    project["access"] = project_access(db, user, project)
    return project


def update_project(db: Any, user: dict, project_id: str, name: Any = None,
                   shared: Any = None, rag_scope: Any = None) -> dict:
    """Rename, re-scope or (admins) share a project. Owner/manager/admin only."""
    project = resolve_project(db, user, project_id)
    if project is None:
        raise LookupError("no project id")
    if project["access"] not in ("admin", "owner", "manager"):
        raise PermissionError("only the owner or a manager can change this project")
    sets, params = [], []
    if name is not None:
        sets.append("name = ?")
        params.append(_ws_clean_name(name))
    if rag_scope is not None:
        sets.append("rag_scope = ?")
        params.append(_ws_clean_scope(rag_scope))
    if shared is not None:
        if project["access"] != "admin":
            raise PermissionError("only administrators share a project with everyone")
        sets.append("shared = ?")
        params.append(1 if bool(shared) else 0)
    if sets:
        db.execute(f"UPDATE projects SET {', '.join(sets)} WHERE id = ?",
                   (*params, project["id"]))
        db.commit()
    updated = get_project(db, project["id"]) or {}
    updated["access"] = project["access"]
    return updated


def delete_project(db: Any, user: dict, project_id: str) -> dict:
    """Forget a project (owner or admin). Its files are left on disk."""
    project = resolve_project(db, user, project_id)
    if project is None:
        raise LookupError("no project id")
    if project["access"] not in ("admin", "owner"):
        raise PermissionError("only the owner can delete this project")
    db.execute("DELETE FROM project_members WHERE project_id = ?", (project["id"],))
    db.execute("DELETE FROM projects WHERE id = ?", (project["id"],))
    db.commit()
    log_event(_ws_log, logging.INFO, "workspace.project_deleted",
              project_id=project["id"], user_id=_ws_user_id(user))
    return project


def set_project_member(db: Any, user: dict, project_id: str, member_id: str,
                       role: str | None = "member") -> list[dict]:
    """Add, change or (role None) remove a member. Owner/manager/admin only."""
    project = resolve_project(db, user, project_id)
    if project is None:
        raise LookupError("no project id")
    if project["access"] not in ("admin", "owner", "manager"):
        raise PermissionError("only the owner or a manager can manage members")
    member = db.get_user(str(member_id)) if member_id else None
    if member is None:
        raise LookupError(f"no user {member_id!r}")
    if member["id"] == project["owner_user_id"]:
        raise ValueError("the owner is always a member")
    if role is None:
        db.execute("DELETE FROM project_members WHERE project_id = ? AND user_id = ?",
                   (project["id"], member["id"]))
    else:
        if role not in PROJECT_ROLES:
            raise ValueError(f"role must be one of {', '.join(PROJECT_ROLES)}")
        db.execute(
            "INSERT INTO project_members (project_id, user_id, role) VALUES (?, ?, ?) "
            "ON CONFLICT(project_id, user_id) DO UPDATE SET role = excluded.role",
            (project["id"], member["id"], role))
    db.commit()
    log_event(_ws_log, logging.INFO, "workspace.member_changed", project_id=project["id"],
              member_id=member["id"], role=role or "removed", by=_ws_user_id(user))
    return project_members(db, project["id"])


def user_allowed_roots(db: Any, config: Config, user: dict,
                       installation_root: Path | None = None) -> list[Path]:
    """Every root this user may work in right now (checkpoint rollback uses it).

    With workspaces off a user's default root IS the installation's root, which
    the caller passes in (this module cannot ask the registry without a cycle).
    """
    roots: list[Path] = []
    default = default_root_for_user(config, user)
    if default is not None:
        roots.append(default)
    elif installation_root is not None:
        roots.append(Path(installation_root).resolve())
    for project in list_projects(db, user):
        root = project.get("root_path")
        if root:
            roots.append(Path(root).resolve())
    return roots


# --------------------------------------------------------------------------- #
# Context injection and RAG scope                                              #
# --------------------------------------------------------------------------- #
def _ws_context_names(registry: Any) -> tuple[str, ...]:
    """CONTEXT_FILES lives in tools.py, which imports this module; read it from
    the registry's own module so the package and the single-file bundle agree."""
    import sys
    module = sys.modules.get(type(registry).__module__)
    names = getattr(module, "CONTEXT_FILES", None) or globals().get("CONTEXT_FILES")
    return tuple(names or ())


def project_context_preview(registry: Any, project: dict | None = None) -> dict:
    """What a turn in `project` (or the current context) gets as project context."""
    override = _ws_project_root(project) if project else None
    with (workspace_root_bound(override) if override is not None else _ws_null_cm()):
        root = registry._root()
        text = registry.project_context() or ""
        files = [name for name in _ws_context_names(registry) if (root / name).is_file()]
    return {
        "project": (project or {}).get("id"),
        "name": (project or {}).get("name"),
        "root": str(root),
        "files": files if text else [],
        "chars": len(text),
        "text": text,
    }


@contextmanager
def _ws_null_cm():
    yield


def context_injected_event(registry: Any, project: dict | None = None) -> dict | None:
    """A `context_injected` stream event for this turn, or None when nothing was.

    Emitted by the agent or chat route after the system prompt is built, so the
    UI can say "using AGENTS.md from <project>" instead of the model quietly
    following instructions the user cannot see.
    """
    if project is None:
        turn = current_turn()
        project = getattr(turn, "project", None) if turn is not None else None
    try:
        preview = project_context_preview(registry, project)
    except Exception as exc:  # noqa: BLE001 - telemetry must not break a turn
        log(f"context_injected preview failed: {exc}", logging.DEBUG)
        return None
    if not preview["chars"]:
        return None
    return {"type": "context_injected",
            "project": (project or {}).get("name") or (project or {}).get("id"),
            "project_id": (project or {}).get("id"),
            "files": preview["files"], "chars": preview["chars"]}


def effective_rag_scope(config: Config) -> str:
    """RAG_SCOPE for this context: the active project's rag_scope when it has
    one, else the installation's. Same comma-list format as RAG_SCOPE."""
    turn = current_turn()
    project = getattr(turn, "project", None) if turn is not None else None
    if isinstance(project, dict) and str(project.get("rag_scope") or "").strip():
        return _ws_clean_scope(project["rag_scope"])
    return config.rag_scope or ""


# --------------------------------------------------------------------------- #
# Routes                                                                       #
# --------------------------------------------------------------------------- #
def _ws_error(exc: Exception):
    from fastapi.responses import JSONResponse
    message = str(exc)
    if message.startswith(("no project", "no access to project")):
        # Not found and not yours look the same from outside.
        return JSONResponse({"error": "no such project"}, status_code=404)
    if isinstance(exc, PermissionError):
        return JSONResponse({"error": message}, status_code=403)
    if isinstance(exc, (ValueError, LookupError)):
        return JSONResponse({"error": redact_text(message)}, status_code=400)
    return JSONResponse({"error": redact_text(f"{type(exc).__name__}: {exc}")},
                        status_code=500)


def register_workspaces(app, ctx) -> None:
    """GET/POST /api/projects, POST/DELETE /api/projects/{id},
    POST /api/projects/{id}/members, GET /api/projects/{id}/context."""
    config, db, registry = ctx.config, ctx.db, ctx.registry
    USER = ctx.USER
    db.ensure_schema("workspaces")
    warning = exec_isolation_warning(config)
    if warning:
        log(warning, logging.WARNING)
        log_event(_ws_log, logging.WARNING, "workspace.exec_not_isolated",
                  exec_backend=config.exec_backend)

    def _envelope(user: dict) -> dict:
        default = default_root_for_user(config, user)
        out = {"enabled": bool(config.workspaces_enabled),
               "default_root": str(default) if default is not None else None}
        caveat = exec_isolation_warning(config)
        if caveat:
            out["warning"] = caveat
        return out

    @app.get("/api/projects")
    def projects_list(user: dict = USER):
        out = _envelope(user)
        out["projects"] = list_projects(db, user)
        return out

    @app.post("/api/projects")
    def projects_create(body: dict, user: dict = USER):
        try:
            project = create_project(
                db, config, user, body.get("name"), body.get("root") or body.get("root_path"),
                shared=bool(body.get("shared")), rag_scope=body.get("rag_scope") or "")
        except Exception as exc:  # noqa: BLE001
            return _ws_error(exc)
        out = _envelope(user)
        out["project"] = project
        return out

    @app.post("/api/projects/{project_id}/members")
    def projects_members(project_id: str, body: dict, user: dict = USER):
        member = body.get("user_id")
        if not member and body.get("username"):
            found = db.get_user_by_username(str(body["username"]))
            member = found["id"] if found else None
        if not member:
            from fastapi.responses import JSONResponse
            return JSONResponse({"error": "user_id or username is required"},
                                status_code=400)
        role = None if body.get("remove") else str(body.get("role") or "member")
        try:
            members = set_project_member(db, user, project_id, member, role)
        except Exception as exc:  # noqa: BLE001
            return _ws_error(exc)
        return {"project_id": project_id, "members": members}

    @app.get("/api/projects/{project_id}/context")
    def projects_context(project_id: str, user: dict = USER):
        try:
            project = resolve_project(db, user, project_id)
            preview = project_context_preview(registry, project)
        except Exception as exc:  # noqa: BLE001
            return _ws_error(exc)
        preview["enabled"] = bool(config.context_files_enabled)
        preview["members"] = project_members(db, project_id)
        return preview

    @app.post("/api/projects/{project_id}")
    def projects_update(project_id: str, body: dict, user: dict = USER):
        try:
            project = update_project(db, user, project_id, name=body.get("name"),
                                     shared=body.get("shared"),
                                     rag_scope=body.get("rag_scope"))
        except Exception as exc:  # noqa: BLE001
            return _ws_error(exc)
        return {"project": project}

    @app.delete("/api/projects/{project_id}")
    def projects_delete(project_id: str, user: dict = USER):
        try:
            project = delete_project(db, user, project_id)
        except Exception as exc:  # noqa: BLE001
            return _ws_error(exc)
        return {"deleted": project["id"], "root": project.get("root_path"),
                "files_kept": True}


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'PROJECT_ROLES',
    'WorkspaceTurnState',
    'bind_turn_scope',
    'reset_turn_scope',
    'workspace_turn_scope',
    'registry_turn_state',
    'bind_workspace_root',
    'reset_workspace_root',
    'acting_workspace_root',
    'workspace_root_bound',
    'path_inside',
    'user_workspace_root',
    'workspace_allowed_parents',
    'workspace_root_for',
    'default_root_for_user',
    'exec_isolation_warning',
    'get_project',
    'project_members',
    'project_access',
    'resolve_project',
    'list_projects',
    'create_project',
    'update_project',
    'delete_project',
    'set_project_member',
    'user_allowed_roots',
    'project_context_preview',
    'context_injected_event',
    'effective_rag_scope',
    'register_workspaces',
    '_ws_project_root',
    '_ws_role_of',
    '_ws_role_cache',
    '_ws_schema',
]
