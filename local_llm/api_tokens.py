"""Personal API tokens for coding tools, the CLI and the menu-bar app.

Why this exists: the only credential the server used to hand out was the
browser session cookie, and /api/auth/login returns it only in Set-Cookie. A
terminal client (Claude Code, Aider, Codex, Continue, the app's own CLI) had no
supported way to obtain a long-lived credential, and the ones that need one
for /v1 were pointed at a copied cookie, which dies with the browser session
and carries every right the user has.

An API token is:

  * shaped ``llm_<8 char prefix>_<40 char secret>``: the fixed prefix lets
    obslog redact it anywhere it leaks into a log line, and the short public
    prefix identifies a token in listings without revealing it;
  * stored only as its SHA-256 (the secret has ~238 bits of entropy, so a fast
    hash is enough; scrypt is for low-entropy passwords) and shown in plaintext
    exactly once, in the response that created it;
  * scoped: ``chat`` (chat routes and /v1, the default), ``status`` (health and
    metrics reads, for the menu-bar app), ``control`` (model restart/select,
    task run/cancel; administrators only), ``admin`` (everything the user's role
    allows). A token never exceeds its user's role: role checks still read the
    live user row, so ``admin`` on a non-admin is plain user rights;
  * bound to the role its user had when it was issued: any role change (an
    admin edit, or Entra changing the role on the next sign-in) revokes it, as
    do disable, delete and the users route's password reset.

Routes: GET/POST/DELETE /api/auth/tokens, POST /api/auth/cli/login (local
accounts, throttled with the web login's own buckets), and the OIDC device-code
pair /api/auth/device/start and /api/auth/device/poll (the Entra round trip
itself lives in auth.py next to the web flow it mirrors).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import secrets
import string
import sys
import threading
import time
import uuid
from datetime import timedelta
from typing import Any

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .database import *  # noqa: F401,F403
from .remote_access import *  # noqa: F401,F403

# Route annotations are strings under `from __future__ import annotations` and
# FastAPI resolves them against MODULE globals, so Request must live here (an
# import inside register_api_tokens would turn `request` into a query param).
try:
    from fastapi import Request  # noqa: F401
except Exception:  # pragma: no cover  (selftest on a machine without FastAPI)
    Request = Any  # type: ignore

API_TOKEN_PREFIX = "llm_"
API_TOKEN_SCOPES = ("chat", "status", "control", "admin")
DEFAULT_API_TOKEN_SCOPES = ("chat",)
# More than this many live tokens per user is a leak waiting to happen (and an
# unbounded table under a script that mints one per run).
API_TOKEN_MAX_PER_USER = 50
_tok_log = get_logger("auth")
_TOK_ALPHABET = string.ascii_letters + string.digits
_TOK_SHAPE = re.compile(r"llm_[a-z0-9]{8}_[A-Za-z0-9]{40}")
# last_used_at is an audit nicety; writing it on every /v1 call would turn each
# streamed completion into a SQLite write. Once a minute per token is enough.
_TOK_TOUCH_INTERVAL_S = 60.0
_tok_touch_lock = threading.Lock()
_tok_touched: dict[str, float] = {}

# What each scope reaches, as (scope, methods or None for any, path regex).
# Matched against the router's own path (request.scope["path"]), so a path the
# router would not route cannot pass here either. Anything not listed needs the
# admin scope: a new route is closed to narrow tokens until someone decides it
# belongs to one, instead of silently open to every token.
_TOK_ROUTES: tuple = tuple(
    (scope, methods, re.compile(pattern)) for scope, methods, pattern in (
        # chat: the conversation surface a coding tool or the CLI needs.
        ("chat", None, r"/v1/.*"),
        ("chat", None, r"/api/chat(/.*)?"),
        ("chat", None, r"/api/conversations?(/.*)?"),
        ("chat", None, r"/api/uploads(/.*)?"),
        # Owner-only files a chat turn or a /v1 audio call uses.
        ("chat", None, r"/api/attachments(/.*)?"),
        # The MCP server hands a client the same role-gated tools /v1 agent
        # mode already runs unattended, under the same policy and audit; without
        # this only an admin-scoped token could connect Claude Code or Codex.
        ("chat", None, r"/mcp"),
        ("chat", ("GET",), r"/api/models(/aliases)?"),
        ("chat", ("GET",), r"/api/agents(/.*)?"),
        ("chat", ("GET",), r"/api/projects(/.*)?"),
        ("chat", ("POST",), r"/api/feedback"),
        ("chat", ("GET",), r"/api/approvals/pending"),
        ("chat", ("GET",), r"/api/auth/me"),
        ("chat", ("GET",), r"/api/me/preferences"),
        ("chat", ("GET",), r"/api/files/download"),
        ("chat", ("GET",), r"/api/integrations/snippets"),
        # status: read-only health for a menu-bar or monitoring client.
        ("status", ("GET",), r"/api/health"),
        ("status", ("GET",), r"/api/metrics(/.*)?"),
        ("status", ("GET",), r"/api/models(/aliases)?"),
        ("status", ("GET",), r"/api/backends"),
        ("status", ("GET",), r"/api/cluster/nodes"),
        ("status", ("GET",), r"/api/routing/.*"),
        ("status", ("GET",), r"/api/bench/status"),
        ("status", ("GET",), r"/api/tasks(/.*)?"),
        ("status", ("GET",), r"/api/runs/.*"),
        ("status", ("GET",), r"/api/approvals/pending"),
        ("status", ("GET",), r"/api/auth/me"),
        # control: the few levers a menu-bar app pulls (the routes themselves
        # are ADMIN routes, so the role check still applies on top).
        ("control", ("POST",), r"/api/model/restart"),
        ("control", ("POST",), r"/api/models/select"),
        ("control", ("POST",), r"/api/tasks/[^/]+/(run|cancel)"),
        ("control", ("POST",), r"/api/runs/[^/]+/cancel"),
    ))
# A scope also grants what the narrower scopes it contains grant.
_TOK_IMPLIES = {
    "chat": ("chat",),
    "status": ("status",),
    "control": ("control", "status"),
    "admin": API_TOKEN_SCOPES,
}


# --------------------------------------------------------------------------- #
# Schema                                                                      #
# --------------------------------------------------------------------------- #
def _tok_schema(conn: Any) -> None:
    """api_tokens table (idempotent; runs on every start and on first use)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_tokens (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL DEFAULT '',
            token_hash TEXT NOT NULL UNIQUE,
            prefix TEXT NOT NULL,
            scopes TEXT NOT NULL DEFAULT '["chat"]',
            created_at TEXT NOT NULL,
            last_used_at TEXT,
            expires_at TEXT,
            revoked_at TEXT,
            created_ip TEXT,
            issued_role TEXT NOT NULL DEFAULT '',
            revoked_reason TEXT
        )
    """)
    have = {row[1] for row in conn.execute("PRAGMA table_info(api_tokens)").fetchall()}
    for column, ddl in (("issued_role", "TEXT NOT NULL DEFAULT ''"),
                        ("revoked_reason", "TEXT")):
        if column not in have:
            conn.execute(f"ALTER TABLE api_tokens ADD COLUMN {column} {ddl}")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_api_tokens_user ON api_tokens(user_id)")


register_schema("api_tokens", _tok_schema)


# --------------------------------------------------------------------------- #
# Token values and scopes                                                     #
# --------------------------------------------------------------------------- #
def looks_like_api_token(value: Any) -> bool:
    """True for a string shaped like one of this app's API tokens."""
    return isinstance(value, str) and bool(_TOK_SHAPE.fullmatch(value.strip()))


def _tok_hash(token: str) -> str:
    return hashlib.sha256(token.strip().encode("utf-8")).hexdigest()


def generate_api_token() -> tuple[str, str]:
    """A fresh (plaintext, prefix). Letters and digits only, so a double-click
    in a terminal selects the whole token."""
    prefix = "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(8))
    secret = "".join(secrets.choice(_TOK_ALPHABET) for _ in range(40))
    return f"{API_TOKEN_PREFIX}{prefix}_{secret}", prefix


def normalize_scopes(scopes: Any, role: str = "user") -> list[str]:
    """Validate requested scopes for a user of `role`. Raises ValueError."""
    if scopes is None or scopes == "" or scopes == []:
        return list(DEFAULT_API_TOKEN_SCOPES)
    if isinstance(scopes, str):
        items = [s.strip().lower() for s in scopes.split(",")]
    elif isinstance(scopes, (list, tuple)):
        items = [str(s).strip().lower() for s in scopes]
    else:
        raise ValueError("scopes must be a list of strings")
    out: list[str] = []
    for item in items:
        if not item:
            continue
        if item not in API_TOKEN_SCOPES:
            raise ValueError(f"unknown scope {item[:40]!r} (expected one of: "
                             + ", ".join(API_TOKEN_SCOPES) + ")")
        if item not in out:
            out.append(item)
    if not out:
        return list(DEFAULT_API_TOKEN_SCOPES)
    if "control" in out and role != "admin":
        raise ValueError("the control scope is for administrators only")
    return out


def effective_scopes(scopes: Any) -> set[str]:
    """Every scope a token holding `scopes` satisfies (admin implies all)."""
    held: set[str] = set()
    for scope in scopes or ():
        held.update(_TOK_IMPLIES.get(str(scope), ()))
    return held


def token_allows_route(scopes: Any, method: str, path: str) -> bool:
    """Whether a token with `scopes` may call `method path` at all.

    This is the coarse gate applied to every USER/ADMIN route for API-token
    requests; the route's own role check still runs afterwards.
    """
    held = effective_scopes(scopes)
    if "admin" in held:
        return True
    method = str(method or "GET").upper()
    if method == "HEAD":
        method = "GET"
    for scope, methods, pattern in _TOK_ROUTES:
        if scope in held and (methods is None or method in methods) \
                and pattern.fullmatch(path or ""):
            return True
    return False


# --------------------------------------------------------------------------- #
# Store                                                                       #
# --------------------------------------------------------------------------- #
def _tok_row(row: Any) -> dict:
    d = dict(row)
    try:
        scopes = json.loads(d.get("scopes") or "[]")
    except (TypeError, ValueError):
        scopes = []
    d["scopes"] = [s for s in scopes if s in API_TOKEN_SCOPES] or list(DEFAULT_API_TOKEN_SCOPES)
    return d


def _tok_active(row: dict, now_iso: str | None = None) -> bool:
    if row.get("revoked_at"):
        return False
    expires = row.get("expires_at")
    return not (expires and str(expires) < (now_iso or iso(utc_now())))


def public_token_row(row: dict) -> dict:
    """A token row safe to return to its owner or an admin (never the hash)."""
    return {
        "id": row.get("id"),
        "user_id": row.get("user_id"),
        "username": row.get("username"),
        "name": row.get("name") or "",
        "prefix": row.get("prefix"),
        "hint": f"{API_TOKEN_PREFIX}{row.get('prefix')}_...",
        "scopes": list(row.get("scopes") or []),
        "created_at": row.get("created_at"),
        "last_used_at": row.get("last_used_at"),
        "expires_at": row.get("expires_at"),
        "revoked_at": row.get("revoked_at"),
        "revoked_reason": row.get("revoked_reason"),
        "created_ip": row.get("created_ip"),
        "active": _tok_active(row),
    }


def create_api_token(db: Any, user: dict, name: str = "", scopes: Any = None,
                     ttl_days: Any = None, config: Any = None,
                     ip: str | None = None) -> tuple[str, dict]:
    """Issue a token for `user`. Returns (plaintext shown once, public row).

    Raises ValueError for a bad scope, lifetime or a user over the cap.
    """
    db.ensure_schema("api_tokens")
    if not user or not user.get("id") or user.get("synthetic"):
        raise ValueError("API tokens need a real user account (authentication is off)")
    role = str(user.get("role") or "user")
    scope_list = normalize_scopes(scopes, role)
    default_ttl = int(getattr(config, "api_token_default_ttl_days", 90) or 0)
    if ttl_days is None or ttl_days == "":
        days = default_ttl
    else:
        try:
            days = int(ttl_days)
        except (TypeError, ValueError):
            raise ValueError("ttl_days must be a whole number of days")
        if days < 0 or days > 3650:
            raise ValueError("ttl_days must be between 0 and 3650")
        # The configured lifetime is also the ceiling: a request may shorten
        # it, never extend it (0 = no expiry only when the install allows it).
        if default_ttl and (days == 0 or days > default_ttl):
            raise ValueError(f"tokens on this server expire within {default_ttl} days")
    clean_name = " ".join(str(name or "").split())[:80] or "token"
    now = utc_now()
    now_iso = iso(now)
    live = [r for r in list_api_tokens(db, user["id"]) if r["active"]]
    if len(live) >= API_TOKEN_MAX_PER_USER:
        raise ValueError(f"too many active tokens ({API_TOKEN_MAX_PER_USER}); revoke some first")
    plaintext, prefix = generate_api_token()
    token_id = uuid.uuid4().hex
    expires = iso(now + timedelta(days=days)) if days else None
    db.execute(
        "INSERT INTO api_tokens (id, user_id, name, token_hash, prefix, scopes, "
        "created_at, expires_at, created_ip, issued_role) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (token_id, user["id"], clean_name, _tok_hash(plaintext), prefix,
         json.dumps(scope_list), now_iso, expires, (ip or None), role))
    db.commit()
    log_event(_tok_log, logging.INFO, "auth.token_created", user_id=user["id"],
              token_id=token_id, prefix=prefix, scopes=",".join(scope_list),
              expires_at=expires, name_=clean_name)
    row = get_api_token(db, token_id) or {}
    return plaintext, public_token_row(row)


def get_api_token(db: Any, token_id: str) -> dict | None:
    db.ensure_schema("api_tokens")
    row = db.execute(
        "SELECT t.*, u.username AS username FROM api_tokens t "
        "LEFT JOIN users u ON u.id = t.user_id WHERE t.id = ?", (str(token_id),)).fetchone()
    return _tok_row(row) if row else None


def list_api_tokens(db: Any, user_id: str | None = None) -> list[dict]:
    """Public rows, newest first; user_id None = every user's (admin)."""
    db.ensure_schema("api_tokens")
    sql = ("SELECT t.*, u.username AS username FROM api_tokens t "
           "LEFT JOIN users u ON u.id = t.user_id")
    params: tuple = ()
    if user_id is not None:
        sql += " WHERE t.user_id = ?"
        params = (user_id,)
    sql += " ORDER BY t.created_at DESC"
    return [public_token_row(_tok_row(r)) for r in db.execute(sql, params).fetchall()]


def revoke_api_token(db: Any, token_id: str, user_id: str | None = None,
                     reason: str = "revoked") -> bool:
    """Revoke one token; owner-scoped when user_id is given. True if it was live."""
    db.ensure_schema("api_tokens")
    sql = ("UPDATE api_tokens SET revoked_at = ?, revoked_reason = ? "
           "WHERE id = ? AND revoked_at IS NULL")
    params: tuple = (iso(utc_now()), str(reason)[:80], str(token_id))
    if user_id is not None:
        sql += " AND user_id = ?"
        params = params + (user_id,)
    cur = db.execute(sql, params)
    db.commit()
    if cur.rowcount:
        log_event(_tok_log, logging.INFO, "auth.token_revoked", token_id=str(token_id),
                  reason=str(reason)[:80])
    return cur.rowcount > 0


def revoke_api_token_value(db: Any, token: str, reason: str = "logout") -> bool:
    """Revoke the token whose plaintext is `token` (CLI logout)."""
    if not looks_like_api_token(token):
        return False
    db.ensure_schema("api_tokens")
    row = db.execute("SELECT id FROM api_tokens WHERE token_hash = ?",
                     (_tok_hash(token),)).fetchone()
    return bool(row) and revoke_api_token(db, row["id"], reason=reason)


def revoke_user_tokens(db: Any, user_id: str, reason: str = "user changed") -> int:
    """Revoke every live token of a user (disable, role change, password reset)."""
    db.ensure_schema("api_tokens")
    cur = db.execute(
        "UPDATE api_tokens SET revoked_at = ?, revoked_reason = ? "
        "WHERE user_id = ? AND revoked_at IS NULL",
        (iso(utc_now()), str(reason)[:80], str(user_id)))
    db.commit()
    if cur.rowcount:
        log_event(_tok_log, logging.INFO, "auth.tokens_revoked", user_id=str(user_id),
                  count=cur.rowcount, reason=str(reason)[:80])
    return cur.rowcount


def resolve_api_token(db: Any, token: str | None) -> dict | None:
    """The user an API token acts as (with token_scopes), or None.

    None for anything that is not a live token of an enabled user whose role is
    unchanged since issue. A role mismatch revokes the token on the spot, which
    covers every path that changes a role (admin edit, Entra role refresh on
    the next web sign-in) without each of them having to remember to.
    """
    if not looks_like_api_token(token):
        return None
    db.ensure_schema("api_tokens")
    row = db.execute("SELECT * FROM api_tokens WHERE token_hash = ?",
                     (_tok_hash(str(token)),)).fetchone()
    if not row:
        return None
    rec = _tok_row(row)
    now_iso = iso(utc_now())
    if not _tok_active(rec, now_iso):
        return None
    user = db.get_user(rec["user_id"])
    if not user:
        revoke_api_token(db, rec["id"], reason="user deleted")
        return None
    if user.get("disabled"):
        return None
    if rec.get("issued_role") and rec["issued_role"] != user.get("role"):
        revoke_api_token(db, rec["id"], reason="role changed")
        return None
    now = time.monotonic()
    with _tok_touch_lock:
        due = now - _tok_touched.get(rec["id"], -1e9) >= _TOK_TOUCH_INTERVAL_S
        if due:
            _tok_touched[rec["id"]] = now
            if len(_tok_touched) > 4096:
                _tok_touched.clear()
                _tok_touched[rec["id"]] = now
    if due:
        try:
            db.execute("UPDATE api_tokens SET last_used_at = ? WHERE id = ?", (now_iso, rec["id"]))
            db.commit()
        except Exception as exc:  # noqa: BLE001  (a busy DB must not fail the request)
            log(f"api token touch failed: {redact_text(str(exc))}", logging.DEBUG)
    user = dict(user)
    user["token_scopes"] = list(rec["scopes"])
    user["token_id"] = rec["id"]
    user["auth_kind"] = "api_token"
    return user


# --------------------------------------------------------------------------- #
# Routes                                                                      #
# --------------------------------------------------------------------------- #
def _tok_safe_user(user: dict) -> dict:
    keep = ("id", "username", "role", "source", "email", "display_name")
    return {k: user.get(k) for k in keep}


def _tok_login_limit() -> int:
    """LOGIN_MAX_FAILURES as /api/auth/login sees it.

    The web login reads the name from api.py's globals (tests lower it there),
    so the CLI login reads the same binding instead of its own import-time copy.
    In the single-file build there is only one namespace and this is the global.
    """
    api_mod = sys.modules.get(f"{__package__}.api") if __package__ else None
    value = getattr(api_mod, "LOGIN_MAX_FAILURES", None) if api_mod is not None else None
    return int(value if value is not None else LOGIN_MAX_FAILURES)


class _TokThrottle:
    """Fallback failed-login buckets with /api/auth/login's semantics.

    Used only when create_app did not publish its own (ctx.extras
    "login_throttle"), so the CLI login is never unthrottled.
    """

    def __init__(self) -> None:
        self.failures: dict[str, list[float]] = {}

    def sweep(self, now: float) -> None:
        for key in [k for k, v in self.failures.items()
                    if not v or now - v[-1] >= LOGIN_WINDOW_S]:
            self.failures.pop(key, None)

    def blocked(self, now: float, key: str, limit: int) -> int:
        recent = [t for t in self.failures.get(key, []) if now - t < LOGIN_WINDOW_S]
        if recent:
            self.failures[key] = recent
        else:
            self.failures.pop(key, None)
        if len(recent) < limit:
            return 0
        return int(LOGIN_WINDOW_S - (now - recent[0])) + 1

    def record(self, now: float, key: str, limit: int) -> None:
        self.failures[key] = (self.failures.get(key, []) + [now])[-limit:]


def register_api_tokens(app, ctx) -> None:
    """Mount token management, CLI login and the OIDC device-code routes."""
    from fastapi.responses import JSONResponse

    config, db, auth, USER = ctx.config, ctx.db, ctx.auth, ctx.USER
    db.ensure_schema("api_tokens")
    fallback = _TokThrottle()

    def _throttle() -> tuple:
        shared = ctx.extras.get("login_throttle")
        if shared:
            return (shared["failures"], shared["blocked"], shared["record"], shared["sweep"])
        return (fallback.failures, fallback.blocked, fallback.record, fallback.sweep)

    def _auth_off() -> Any:
        return JSONResponse(
            {"error": "authentication is disabled on this server (AUTH_ENABLED=0): "
                      "clients need no token", "auth_enabled": False}, status_code=400)

    def _client_ip(request: Any) -> str:
        try:
            return client_ip_for(request, config)
        except Exception:  # noqa: BLE001
            return "?"

    @app.get("/api/auth/tokens")
    def tokens_list(all: int = 0, user: dict = USER):  # noqa: A002
        if not auth.enabled:
            return {"tokens": [], "auth_enabled": False}
        if all and user.get("role") == "admin":
            return {"tokens": list_api_tokens(db, None), "all": True}
        return {"tokens": list_api_tokens(db, user["id"]), "all": False}

    @app.post("/api/auth/tokens")
    def tokens_create(body: dict, request: Request, user: dict = USER):
        if not auth.enabled:
            return _auth_off()
        try:
            plaintext, row = create_api_token(
                db, user, name=str(body.get("name") or ""), scopes=body.get("scopes"),
                ttl_days=body.get("ttl_days"), config=config, ip=_client_ip(request))
        except ValueError as exc:
            return JSONResponse({"error": redact_text(str(exc))}, status_code=400)
        return {"token": plaintext, "api_token": row,
                "note": "Copy the token now: it is shown only once."}

    @app.delete("/api/auth/tokens/{token_id}")
    def tokens_revoke(token_id: str, user: dict = USER):
        if not auth.enabled:
            return _auth_off()
        row = get_api_token(db, token_id)
        is_admin = user.get("role") == "admin"
        # Someone else's token is "not found", not "forbidden": the id itself
        # must not confirm that a token exists.
        if not row or (row["user_id"] != user["id"] and not is_admin):
            return JSONResponse({"error": "no such token"}, status_code=404)
        reason = "revoked by owner" if row["user_id"] == user["id"] else "revoked by admin"
        changed = revoke_api_token(db, token_id, reason=reason)
        return {"ok": True, "revoked": changed}

    @app.post("/api/auth/cli/login")
    async def tokens_cli_login(body: dict, request: Request):
        """Username/password -> a fresh API token, for terminals.

        Throttled with the SAME buckets as /api/auth/login (published by
        create_app), so the CLI is not a second, independent guessing oracle.
        """
        if not auth.enabled:
            return _auth_off()
        username = str(body.get("username", "")).strip()
        password = str(body.get("password", ""))
        failures, blocked, record, sweep = _throttle()
        limit = _tok_login_limit()
        client_ip = _client_ip(request)
        user_key, ip_key = f"{client_ip}|{username.lower()}", f"{client_ip}|"
        ip_limit = limit * 3
        now = time.time()
        if len(failures) > 512:
            sweep(now)
        retry_in = blocked(now, user_key, limit) or blocked(now, ip_key, ip_limit)
        if retry_in:
            log_event(_tok_log, logging.WARNING, "auth.login_throttled",
                      username=username, client=client_ip, retry_in_s=retry_in, via="cli")
            return JSONResponse(
                {"error": f"too many failed attempts; try again in {retry_in}s"},
                status_code=429, headers={"Retry-After": str(retry_in)})
        user = await asyncio.to_thread(auth.authenticate_local, username, password)
        if not user:
            record(now, user_key, limit)
            record(now, ip_key, ip_limit)
            log_event(_tok_log, logging.WARNING, "auth.login_failed",
                      username=username, client=client_ip, via="cli")
            return JSONResponse({"error": "invalid username or password"}, status_code=401)
        failures.pop(user_key, None)
        failures.pop(ip_key, None)
        try:
            plaintext, row = await asyncio.to_thread(
                create_api_token, db, user, str(body.get("name") or "cli"),
                body.get("scopes"), body.get("ttl_days"), config, client_ip)
        except ValueError as exc:
            return JSONResponse({"error": redact_text(str(exc))}, status_code=400)
        await asyncio.to_thread(db.touch_login, user["id"])
        log_event(_tok_log, logging.INFO, "auth.login", user_id=user["id"],
                  username=user["username"], role=user["role"], via="cli")
        return {"token": plaintext, "api_token": row, "user": _tok_safe_user(user)}

    @app.post("/api/auth/device/start")
    def tokens_device_start(request: Request):
        """Begin an Entra device-code sign-in for a terminal."""
        if not auth.device_flow_enabled:
            return JSONResponse(
                {"error": "terminal sign-in with Microsoft is not configured on this "
                          "server: an administrator must set OIDC_CLI_CLIENT_ID (an Entra "
                          "app with public client flows enabled) next to OIDC_ENABLED=1 "
                          "and OIDC_TENANT_ID. Use a username and password, or create a "
                          "token in Settings."}, status_code=404)
        try:
            return auth.oidc_device_start(client_ip=_client_ip(request))
        except OverflowError as exc:
            return JSONResponse({"error": str(exc)}, status_code=429)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": redact_text(str(exc))[:300]}, status_code=502)

    @app.post("/api/auth/device/poll")
    def tokens_device_poll(body: dict, request: Request):
        if not auth.device_flow_enabled:
            return JSONResponse({"error": "terminal sign-in with Microsoft is not "
                                          "configured on this server"}, status_code=404)
        device_code = str(body.get("device_code") or "").strip()
        if not device_code:
            return JSONResponse({"error": "device_code is required"}, status_code=400)
        try:
            outcome = auth.oidc_device_poll(device_code)
        except ValueError as exc:
            return JSONResponse({"status": "error", "error": redact_text(str(exc))[:300]},
                                status_code=400)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"status": "error", "error": redact_text(str(exc))[:300]},
                                status_code=502)
        status = outcome.get("status")
        if status in ("pending", "slow_down"):
            return {"status": status, "interval": outcome.get("interval")}
        if status == "denied":
            return JSONResponse({"status": "denied", "error": "sign-in was declined"},
                                status_code=403)
        if status != "ok":
            return JSONResponse({"status": "expired", "error": "the code expired or is "
                                 "unknown; start again"}, status_code=410)
        user = outcome["user"]
        if not user or user.get("disabled"):
            return JSONResponse({"status": "denied", "error": "account disabled"},
                                status_code=403)
        try:
            plaintext, row = create_api_token(
                db, user, name=str(body.get("name") or "cli (Microsoft)"),
                scopes=body.get("scopes"), ttl_days=body.get("ttl_days"),
                config=config, ip=_client_ip(request))
        except ValueError as exc:
            return JSONResponse({"status": "error", "error": redact_text(str(exc))},
                                status_code=400)
        db.touch_login(user["id"])
        log_event(_tok_log, logging.INFO, "auth.oidc_login", user_id=user["id"],
                  role=user.get("role"), via="device")
        return {"status": "ok", "token": plaintext, "api_token": row,
                "user": _tok_safe_user(user)}


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    "API_TOKEN_MAX_PER_USER",
    "API_TOKEN_PREFIX",
    "API_TOKEN_SCOPES",
    "DEFAULT_API_TOKEN_SCOPES",
    "_TokThrottle",
    "_tok_hash",
    "_tok_login_limit",
    "_tok_safe_user",
    "_tok_schema",
    "create_api_token",
    "effective_scopes",
    "generate_api_token",
    "get_api_token",
    "list_api_tokens",
    "looks_like_api_token",
    "normalize_scopes",
    "public_token_row",
    "register_api_tokens",
    "resolve_api_token",
    "revoke_api_token",
    "revoke_api_token_value",
    "revoke_user_tokens",
    "token_allows_route",
]
