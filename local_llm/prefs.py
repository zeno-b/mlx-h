"""Per-user preferences: the server-side store behind the Settings panel.

Before this, everything a user chose (theme, density, their own temperature and
reply length) lived in the browser's localStorage. That had three failures: a
second browser or the phone started from scratch, the CLI and API clients never
saw the choice at all (so the same person got a different temperature from
`local-llm chat` than from the web page), and a user's "my defaults" could only
be applied by the page remembering to send them on every request.

The store keeps one row per (user, key) in `user_preferences`, owned by this
module through database.register_schema, so database.py does not grow a table it
does not use. Values are JSON so a bool stays a bool and None stays "use the
installation default". Every read re-validates against settings_schema.USER_PREFS:
a bound that tightened since a value was saved is applied on the way out, and a
key that was removed simply disappears, so a stale row can never become a value a
later request trips over.

Routes (registered by register_prefs from create_app's extension anchor):

  GET  /api/settings/schema   the schema for the caller's role, the caller's
                              preferences, and (admins only) the live global
                              values, masked exactly like GET /api/config
  GET  /api/me/preferences    {"preferences": {...}, "defaults": {...}}
  POST /api/me/preferences    {key: value}; null resets a key to its default

Isolation: every read and write is keyed by the authenticated caller's id from
the USER dependency, never by anything in the request body, so one user cannot
read or change another's preferences. With auth disabled the caller is the
synthetic local user and the panel works as a single-user install.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .database import *  # noqa: F401,F403
from .settings_schema import *  # noqa: F401,F403


PREFS_SCHEMA_NAME = "user_preferences"


def _prefs_schema(conn) -> None:
    """Idempotent: one row per (user, key), value stored as JSON text."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            updated_at REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, key)
        )
        """
    )


register_schema(PREFS_SCHEMA_NAME, _prefs_schema)


def _prefs_delete_user(db, user_id: str) -> None:
    """Database.delete_user hook: a deleted account's preferences go with it."""
    db.ensure_schema(PREFS_SCHEMA_NAME)
    db.execute("DELETE FROM user_preferences WHERE user_id = ?", (str(user_id),))
    db.commit()


register_user_cleanup(PREFS_SCHEMA_NAME, _prefs_delete_user)


class PreferenceStore:
    """Read and write one user's preferences. Stateless apart from the db handle,
    so building one per call is cheap (the schema check is cached on the db)."""

    def __init__(self, db) -> None:
        self.db = db

    def _ready(self) -> None:
        self.db.ensure_schema(PREFS_SCHEMA_NAME)

    def stored(self, user_id: str) -> dict[str, Any]:
        """Only the keys this user has explicitly set, re-validated."""
        self._ready()
        rows = self.db.execute(
            "SELECT key, value FROM user_preferences WHERE user_id = ?",
            (str(user_id),)).fetchall()
        raw: dict[str, Any] = {}
        for row in rows:
            try:
                raw[row["key"]] = json.loads(row["value"])
            except (TypeError, ValueError):
                continue
        accepted, _ignored = validate_user_prefs(raw)
        return accepted

    def get(self, user_id: str) -> dict[str, Any]:
        """Every USER_PREFS key for this user, defaults filled in."""
        merged = default_user_prefs()
        merged.update(self.stored(user_id))
        return merged

    def update(self, user_id: str, updates: dict) -> dict[str, Any]:
        """Validate and persist a patch. None resets a key to its default.

        Returns {"preferences", "changed", "ignored"}: changed lists the keys
        whose effective value moved, so a no-op save reports nothing and a
        clamped value shows up in preferences at the value actually kept.
        """
        accepted, ignored = validate_user_prefs(updates if isinstance(updates, dict) else {})
        before = self.get(user_id)
        defaults = default_user_prefs()
        if accepted:
            now = time.time()
            for key, value in accepted.items():
                if value is None or value == defaults.get(key):
                    # Storing the default would freeze it: a later change to the
                    # installation default would never reach this user.
                    self.db.execute(
                        "DELETE FROM user_preferences WHERE user_id = ? AND key = ?",
                        (str(user_id), key))
                else:
                    self.db.execute(
                        "INSERT INTO user_preferences (user_id, key, value, updated_at) "
                        "VALUES (?, ?, ?, ?) ON CONFLICT(user_id, key) DO UPDATE SET "
                        "value = excluded.value, updated_at = excluded.updated_at",
                        (str(user_id), key, json.dumps(value), now))
            self.db.commit()
        after = self.get(user_id)
        changed = [key for key in after if after[key] != before.get(key)]
        return {"preferences": after, "changed": changed, "ignored": ignored}

    def generation_defaults(self, user_id: str | None) -> dict[str, Any]:
        """The caller's max_tokens/temperature preferences that are set (not None)."""
        if not user_id:
            return {}
        mine = self.stored(user_id)
        return {key: mine[key] for key in ("max_tokens", "temperature")
                if mine.get(key) is not None}


def user_generation_prefs(db, user_id: str | None) -> dict[str, Any]:
    """Preference-supplied generation defaults for create_app's overrides().

    Never raises: a damaged preferences table must cost a user their saved
    temperature, not their chat.
    """
    try:
        return PreferenceStore(db).generation_defaults(user_id)
    except Exception as exc:
        log(f"preferences unavailable for this request: {redact_text(str(exc))}",
            logging.DEBUG)
        return {}


def register_prefs(app, ctx) -> None:
    """Mount the settings-schema and per-user preference routes."""
    from fastapi.responses import JSONResponse

    store = PreferenceStore(ctx.db)
    ctx.extras["prefs"] = store
    USER = ctx.USER

    @app.get("/api/settings/schema")
    async def settings_schema_route(user: dict = USER):
        admin = user.get("role") == "admin"
        payload = schema_for_role(admin)
        payload["role"] = "admin" if admin else "user"
        # config.public() is the GET /api/config view: secrets reported only as
        # "***set***" or "". Non-admins get no installation values at all.
        payload["values"] = ctx.config.public() if admin else {}
        payload["preferences"] = await asyncio.to_thread(store.get, user["id"])
        return payload

    @app.get("/api/me/preferences")
    async def my_preferences(user: dict = USER):
        prefs = await asyncio.to_thread(store.get, user["id"])
        return {"preferences": prefs, "defaults": default_user_prefs()}

    @app.post("/api/me/preferences")
    async def set_my_preferences(body: dict, user: dict = USER):
        if not isinstance(body, dict):
            return JSONResponse({"error": "expected a JSON object of preferences"},
                                status_code=400)
        result = await asyncio.to_thread(store.update, user["id"], body)
        if result["changed"]:
            log_event(get_logger("prefs"), logging.INFO, "prefs.updated",
                      changed=",".join(result["changed"]))
        return result


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'PREFS_SCHEMA_NAME',
    'PreferenceStore',
    'user_generation_prefs',
    'register_prefs',
    '_prefs_schema',
    '_prefs_delete_user',
]
