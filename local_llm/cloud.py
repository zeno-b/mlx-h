"""Optional cloud fallback for prompts this Mac cannot prefill well.

On an 8 GB machine the local model has a small context window and reads a
prompt at a few hundred tokens a second. A 30k-token paste either does not fit
at all (the agent then reads it in parts, or trim_to_context drops history it
cannot drop) or takes minutes before the first word. Some installations have a
cloud model they are allowed to use for exactly those cases and asked for an
escape hatch.

The hatch is narrow on purpose, because this project's identity is that data
stays here:
  * off by default: with CLOUD_FALLBACK_ENABLED=0 cloud_offload returns None
    before looking at anything, so the request path is today's;
  * only for roles and users an administrator listed (CLOUD_ALLOWED_ROLES);
  * only after the user said yes for this turn (or, in per_user mode, for this
    conversation or always). An interactive turn gets a consent_required event
    and waits inside turn.waiting_on_human(); an unattended turn (scheduled
    task, API client) never asks and reaches the cloud only under an "always"
    consent;
  * minimised: raw audio parts are dropped and every text goes through
    redact_text before it leaves;
  * capped per user and calendar month (CLOUD_MAX_TOKENS_PER_USER_MONTH);
  * logged as metadata only: cloud.fallback carries user, conversation, token
    counts, provider, model, latency and the consent scope, never content, plus
    a routing_events row with node "cloud" and a cloud_usage row for the cap.

A refusal, a timeout, a missing key, the cap, or a cloud error before the first
token all leave the prompt on the local path with today's chunk/trim behaviour.

The model client calls cloud_offload() at the top of its request path. The
CloudOffload it may return streams the same typed events the local path yields
(content, reasoning, tool_call_delta, finish, usage), so ModelClient.stream
wraps cloud reasoning in <think> exactly as it does for a local model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .database import *  # noqa: F401,F403
from .turnctx import *  # noqa: F401,F403


CLOUD_SCHEMA_NAME = "cloud"
CLOUD_PROVIDERS = ("openai", "anthropic")
CLOUD_CONSENT_MODES = ("per_request", "per_user")
CLOUD_CONSENT_SCOPES = ("once", "conversation", "always")
CLOUD_ANTHROPIC_VERSION = "2023-06-01"

_CLOUD_DEFAULT_URLS = {"openai": "https://api.openai.com/v1",
                       "anthropic": "https://api.anthropic.com"}
# register_cloud binds the live config and database, so the model client (which
# has neither a db handle nor a route) can check consents, the cap and usage.
_CLOUD_BOUND: dict[str, Any] = {"config": None, "db": None}
# Test seam: an httpx transport (httpx.MockTransport) used instead of the network.
_CLOUD_HOOKS: dict[str, Any] = {"transport": None}
# Measured prompt speed per database, cached briefly: it is read on every model
# call while the prefill trigger is on, and the metrics table only grows.
_CLOUD_TPS_CACHE: dict[str, tuple[float, Any]] = {}
_CLOUD_TPS_TTL_S = 60.0
_CLOUD_TPS_MIN_PROMPT = 256
_CLOUD_AUDIO_PARTS = frozenset({"input_audio", "audio", "audio_url"})
# Request keys a caller's extra may carry to an OpenAI-compatible provider. The
# local server's own knobs (chat_template_kwargs, enable_thinking, xtc, ...)
# mean nothing there and some providers reject unknown keys.
_CLOUD_PASSTHROUGH = frozenset({
    "tools", "tool_choice", "parallel_tool_calls", "response_format", "stop", "seed",
    "top_p", "presence_penalty", "frequency_penalty",
})
_CLOUD_WARNED: set[str] = set()
_cloud_log = get_logger("cloud")


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def _cloud_schema(conn) -> None:
    """Idempotent: consent grants and per-call usage (the monthly cap reads it)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cloud_consents (
            id INTEGER PRIMARY KEY,
            user_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            conversation_id TEXT,
            granted_at TEXT NOT NULL,
            revoked_at TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cloud_consents_user "
                 "ON cloud_consents(user_id, revoked_at)")
    # Metadata only, by design: no prompt, no reply, no file names.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cloud_usage (
            id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL,
            month TEXT NOT NULL,
            user_id TEXT NOT NULL,
            conversation_id TEXT,
            turn_id TEXT,
            kind TEXT,
            provider TEXT,
            model TEXT,
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            latency_ms INTEGER,
            status TEXT,
            consent_scope TEXT,
            trigger TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cloud_usage_user_month "
                 "ON cloud_usage(user_id, month)")


register_schema(CLOUD_SCHEMA_NAME, _cloud_schema)


def bind_cloud(config: Any = None, db: Any = None) -> None:
    """Bind the live config/db for model-client calls that carry neither."""
    if config is not None:
        _CLOUD_BOUND["config"] = config
    if db is not None:
        _CLOUD_BOUND["db"] = db


def _cloud_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cloud_month(now: datetime | None = None) -> str:
    """The cap's accounting period: the UTC calendar month, YYYY-MM."""
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m")


def _cloud_warn_once(key: str, message: str) -> None:
    if key in _CLOUD_WARNED:
        return
    _CLOUD_WARNED.add(key)
    log(message, logging.WARNING)


# --------------------------------------------------------------------------- #
# Settings and eligibility
# --------------------------------------------------------------------------- #
def cloud_settings(config: Any) -> dict | None:
    """Provider, endpoint, model and key; None when the fallback cannot run."""
    if config is None:
        return None
    provider = str(getattr(config, "cloud_provider", "openai") or "openai").strip().lower()
    if provider not in CLOUD_PROVIDERS:
        return None
    model = str(getattr(config, "cloud_model", "") or "").strip()
    key = str(getattr(config, "cloud_api_key", "") or "").strip()
    explicit = str(getattr(config, "cloud_base_url", "") or "").strip().rstrip("/")
    base = explicit or _CLOUD_DEFAULT_URLS[provider]
    # A key is required for the public APIs. An explicit OpenAI-compatible
    # endpoint (a LAN vLLM, LM Studio) may legitimately take none.
    if not model or (not key and not (provider == "openai" and explicit)):
        return None
    if provider == "anthropic":
        url = base + ("/messages" if base.endswith("/v1") else "/v1/messages")
    else:
        url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    return {"provider": provider, "model": model, "api_key": key, "base_url": base,
            "url": url}


def _cloud_allow_entries(config: Any) -> set[str]:
    raw = str(getattr(config, "cloud_allowed_roles", "") or "")
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def cloud_user_allowed(config: Any, user_id: str | None, role: str | None,
                       username: str | None = None) -> bool:
    """Is this user on the administrator's allowlist (roles, user:<id|name>, *)?"""
    entries = _cloud_allow_entries(config)
    if not entries:
        return False
    if "*" in entries:
        return True
    if role and str(role).strip().lower() in entries:
        return True
    for ident in (user_id, username):
        if ident and f"user:{str(ident).strip().lower()}" in entries:
            return True
    return False


def _cloud_identity(turn: TurnContext | None, db: Any, user_id: str,
                    config: Any) -> tuple[str | None, str | None]:
    """(role, username) for the allowlist; the user row only when it matters."""
    role = turn.role if turn is not None else None
    username = None
    needs_row = role is None or any(e.startswith("user:") for e in _cloud_allow_entries(config))
    if needs_row and db is not None:
        try:
            row = db.get_user(user_id)
        except Exception:  # noqa: BLE001 - an unknown user is simply not allowed
            row = None
        if row:
            role = role or row.get("role")
            username = row.get("username")
    if role is None and user_id == SENTINEL_LOCAL_USER:
        role = "admin"      # auth disabled: the implicit local user is the admin
    return role, username


# --------------------------------------------------------------------------- #
# Consent store
# --------------------------------------------------------------------------- #
def _cloud_row(row: Any) -> dict:
    out = dict(row)
    out["active"] = out.get("revoked_at") is None and out.get("scope") != "once"
    return out


def grant_cloud_consent(db: Any, user_id: str, scope: str,
                        conversation_id: str | None = None) -> int | None:
    """Record a grant. once rows are the audit trail; they never match later turns."""
    if db is None:
        return None
    if scope not in CLOUD_CONSENT_SCOPES:
        raise ValueError(f"scope must be one of {', '.join(CLOUD_CONSENT_SCOPES)}")
    if scope == "conversation" and not conversation_id:
        raise ValueError("a conversation consent needs a conversation")
    db.ensure_schema(CLOUD_SCHEMA_NAME)
    now = _cloud_now()
    # A new standing grant replaces the same one still active, so the list a
    # user reviews (and revokes from) never shows duplicates.
    if scope == "always":
        db.execute("UPDATE cloud_consents SET revoked_at = ? WHERE user_id = ? AND "
                   "scope = 'always' AND revoked_at IS NULL", (now, user_id))
    elif scope == "conversation":
        db.execute("UPDATE cloud_consents SET revoked_at = ? WHERE user_id = ? AND "
                   "scope = 'conversation' AND conversation_id = ? AND revoked_at IS NULL",
                   (now, user_id, conversation_id))
    cur = db.execute(
        "INSERT INTO cloud_consents (user_id, scope, conversation_id, granted_at, revoked_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (user_id, scope, conversation_id if scope != "always" else None, now,
         now if scope == "once" else None))
    db.commit()
    return cur.lastrowid


def list_cloud_consents(db: Any, user_id: str, include_history: bool = False) -> list[dict]:
    """The user's own grants, newest first; active standing grants by default."""
    db.ensure_schema(CLOUD_SCHEMA_NAME)
    sql = "SELECT * FROM cloud_consents WHERE user_id = ?"
    if not include_history:
        sql += " AND revoked_at IS NULL AND scope != 'once'"
    rows = db.execute(sql + " ORDER BY id DESC LIMIT 500", (user_id,)).fetchall()
    return [_cloud_row(r) for r in rows]


def revoke_cloud_consents(db: Any, user_id: str, consent_id: int | None = None) -> int:
    """Revoke one (owner-checked) or every active grant of the user."""
    db.ensure_schema(CLOUD_SCHEMA_NAME)
    now = _cloud_now()
    if consent_id is None:
        cur = db.execute("UPDATE cloud_consents SET revoked_at = ? WHERE user_id = ? "
                         "AND revoked_at IS NULL", (now, user_id))
    else:
        cur = db.execute("UPDATE cloud_consents SET revoked_at = ? WHERE id = ? AND "
                         "user_id = ? AND revoked_at IS NULL", (now, int(consent_id), user_id))
    db.commit()
    return int(cur.rowcount or 0)


def standing_cloud_consent(db: Any, user_id: str, conversation_id: str | None,
                           scopes: tuple = ("conversation", "always")) -> str | None:
    """"conversation" or "always" when a live grant covers this turn, else None.

    The narrower grant wins when both exist, so the logged scope says what the
    user granted for this conversation. `scopes` limits what may answer (an
    unattended run passes ("always",)).
    """
    if db is None:
        return None
    db.ensure_schema(CLOUD_SCHEMA_NAME)
    rows = db.execute("SELECT scope, conversation_id FROM cloud_consents WHERE user_id = ? "
                      "AND revoked_at IS NULL AND scope IN ('conversation', 'always')",
                      (user_id,)).fetchall()
    found = None
    for row in rows:
        if row["scope"] not in scopes:
            continue
        if row["scope"] == "conversation" and conversation_id \
                and row["conversation_id"] == conversation_id:
            return "conversation"
        if row["scope"] == "always":
            found = "always"
    return found


# --------------------------------------------------------------------------- #
# Usage and the monthly cap
# --------------------------------------------------------------------------- #
def cloud_month_usage(db: Any, user_id: str, month: str | None = None) -> int:
    """Cloud tokens (prompt + reply) the user spent in the month."""
    if db is None:
        return 0
    db.ensure_schema(CLOUD_SCHEMA_NAME)
    row = db.execute("SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0) AS used "
                     "FROM cloud_usage WHERE user_id = ? AND month = ?",
                     (user_id, month or cloud_month())).fetchone()
    return int((row["used"] if row is not None else 0) or 0)


def record_cloud_usage(db: Any, **fields: Any) -> None:
    """Append one usage row. Never raises into a generation."""
    if db is None:
        return
    cols = ("user_id", "conversation_id", "turn_id", "kind", "provider", "model",
            "prompt_tokens", "completion_tokens", "latency_ms", "status",
            "consent_scope", "trigger")
    try:
        db.ensure_schema(CLOUD_SCHEMA_NAME)
        db.execute(
            f"INSERT INTO cloud_usage (created_at, month, {', '.join(cols)}) "
            f"VALUES (?, ?, {', '.join('?' for _ in cols)})",
            (_cloud_now(), cloud_month(), *[fields.get(c) for c in cols]))
        db.commit()
    except Exception as exc:  # noqa: BLE001 - accounting failure is logged, not fatal
        log(f"cloud fallback: usage write failed: {redact_text(str(exc))}", logging.ERROR)


def cloud_usage_report(db: Any, month: str | None = None, config: Any = None) -> dict:
    """Per-user totals for a month (admin view)."""
    month = month or cloud_month()
    db.ensure_schema(CLOUD_SCHEMA_NAME)
    select = ("SELECT c.user_id AS user_id, {name} COUNT(*) AS requests, "
              "SUM(CASE WHEN c.status = 'ok' THEN 1 ELSE 0 END) AS ok, "
              "COALESCE(SUM(c.prompt_tokens), 0) AS prompt_tokens, "
              "COALESCE(SUM(c.completion_tokens), 0) AS completion_tokens "
              "FROM cloud_usage c {join} WHERE c.month = ? GROUP BY c.user_id "
              "ORDER BY prompt_tokens + completion_tokens DESC")
    try:
        rows = db.execute(select.format(name="u.username AS username,",
                                        join="LEFT JOIN users u ON u.id = c.user_id"),
                          (month,)).fetchall()
    except Exception:  # noqa: BLE001 - a database without a users table
        rows = db.execute(select.format(name="", join=""), (month,)).fetchall()
    users = []
    for row in rows:
        item = dict(row)
        item.setdefault("username", None)
        item["total_tokens"] = int(item["prompt_tokens"] or 0) + int(item["completion_tokens"] or 0)
        users.append(item)
    cfg = config or _CLOUD_BOUND["config"]
    return {
        "month": month,
        "cap_per_user": int(getattr(cfg, "cloud_max_tokens_per_user_month", 0) or 0),
        "users": users,
        "totals": {
            "requests": sum(int(u["requests"] or 0) for u in users),
            "prompt_tokens": sum(int(u["prompt_tokens"] or 0) for u in users),
            "completion_tokens": sum(int(u["completion_tokens"] or 0) for u in users),
        },
    }


# --------------------------------------------------------------------------- #
# The decision rule
# --------------------------------------------------------------------------- #
def cloud_measured_prefill_tps(db: Any) -> float | None:
    """Median local prompt speed (tokens/s) over recent sizeable prompts.

    From the metrics table the chat routes already write (prompt_tokens and
    ttft_ms per generation). Small prompts are left out: their time to first
    token is mostly fixed overhead, which would make prefill look slow. None
    when nothing has been measured yet: the prefill trigger then does not guess.
    """
    if db is None:
        return None
    key = str(getattr(db, "db_path", "") or id(db))
    now = time.monotonic()
    hit = _CLOUD_TPS_CACHE.get(key)
    if hit is not None and now - hit[0] < _CLOUD_TPS_TTL_S:
        return hit[1]
    value = None
    base = ("SELECT prompt_tokens, ttft_ms FROM metrics m WHERE prompt_tokens >= ? "
            "AND ttft_ms > 0 {skip} ORDER BY id DESC LIMIT 40")
    # A turn the cloud answered writes its (fast) time to first token into the
    # same metrics table; leave those out, or every offload would make local
    # prefill look quicker and the trigger fire less. Rows of the conversation
    # within two minutes of a cloud call are skipped (losing a local sample or
    # two is harmless).
    skip = ("AND NOT EXISTS (SELECT 1 FROM cloud_usage c WHERE c.conversation_id = "
            "m.conversation_id AND ABS(julianday(c.created_at) - julianday(m.timestamp)) "
            "< 2.0 / 1440)")
    try:
        try:
            rows = db.execute(base.format(skip=skip), (_CLOUD_TPS_MIN_PROMPT,)).fetchall()
        except Exception:  # noqa: BLE001 - no cloud_usage table yet
            rows = db.execute(base.format(skip=""), (_CLOUD_TPS_MIN_PROMPT,)).fetchall()
        rates = sorted(float(r["prompt_tokens"]) / (float(r["ttft_ms"]) / 1000.0)
                       for r in rows)
        if rates:
            value = rates[len(rates) // 2]
    except Exception:  # noqa: BLE001 - no metrics table yet
        value = None
    _CLOUD_TPS_CACHE[key] = (now, value)
    return value


@dataclass
class CloudTrigger:
    """Why a prompt qualifies for the cloud: context, size or prefill."""

    reason: str
    prompt_tokens: int
    prefill_s: float | None = None
    detail: str = ""


def cloud_trigger(config: Any, messages: list[dict], max_tokens: int | None = None,
                  prefill_tps: float | None = None) -> CloudTrigger | None:
    """The size rule alone (no consent, no allowlist): does this prompt qualify?

    context: the prompt does not fit the local window even with the smallest
      reply budget. Trimming has already run by the time a prompt reaches the
      model client, and trim_to_context never drops the system prompt, pinned
      messages or the current turn, so an overflow here is required content.
    size: at least CLOUD_MIN_PROMPT_TOKENS (when set).
    prefill: at least CLOUD_MIN_PREFILL_S of local prefill at the measured
      prompt speed (when set and measured).
    """
    tokens = messages_tokens(messages or [])
    window = int(getattr(config, "context_size", 0) or 0)
    floor = int(getattr(config, "min_max_tokens", 128) or 128)
    est_s = (tokens / prefill_tps) if prefill_tps and prefill_tps > 0 else None
    if window and tokens + floor + CONTEXT_SAFETY_MARGIN > window:
        return CloudTrigger("context", tokens, est_s,
                            f"The prompt is about {tokens:,} tokens, more than the local "
                            f"{window:,}-token context can hold.")
    min_tokens = int(getattr(config, "cloud_min_prompt_tokens", 0) or 0)
    if min_tokens and tokens >= min_tokens:
        return CloudTrigger("size", tokens, est_s,
                            f"The prompt is about {tokens:,} tokens (the cloud threshold "
                            f"is {min_tokens:,}).")
    min_s = float(getattr(config, "cloud_min_prefill_s", 0) or 0)
    if min_s and est_s is not None and est_s >= min_s:
        return CloudTrigger("prefill", tokens, est_s,
                            f"Reading this {tokens:,}-token prompt locally would take "
                            f"about {int(est_s)} s before the first word.")
    return None


# --------------------------------------------------------------------------- #
# Data minimisation and provider payloads
# --------------------------------------------------------------------------- #
def _cloud_is_audio(part: dict) -> bool:
    if str(part.get("type") or "") in _CLOUD_AUDIO_PARTS:
        return True
    url = part.get("image_url")
    url = url.get("url") if isinstance(url, dict) else url
    return isinstance(url, str) and url[:11].lower() == "data:audio/"


def _cloud_redact_calls(calls: Any) -> list:
    out = []
    for call in calls if isinstance(calls, list) else []:
        if not isinstance(call, dict):
            continue
        call = dict(call)
        fn = call.get("function")
        if isinstance(fn, dict):
            fn = dict(fn)
            args = fn.get("arguments")
            if isinstance(args, str):
                fn["arguments"] = redact_text(args)
            elif isinstance(args, (dict, list)):
                fn["arguments"] = json.dumps(redact_obj(args), ensure_ascii=False)
            call["function"] = fn
        out.append(call)
    return out


def cloud_minimise_messages(messages: list[dict]) -> tuple[list[dict], int]:
    """A copy safe to send: audio parts dropped, every text redacted.

    Returns (messages, number of parts dropped). A voice note's transcript is
    text the user chose to send and stays; the recording itself never leaves.
    Unknown part types are dropped too, so a new attachment kind is opt-in here
    rather than leaking by default.
    """
    out: list[dict] = []
    dropped = 0
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                kind = str(part.get("type") or "text")
                if _cloud_is_audio(part):
                    dropped += 1
                elif kind == "text":
                    parts.append({"type": "text", "text": redact_text(str(part.get("text") or ""))})
                elif kind == "image_url":
                    parts.append({"type": "image_url", "image_url": part.get("image_url")})
                else:
                    dropped += 1
            content = parts
        elif content is None:
            content = ""
        else:
            content = redact_text(str(content))
        item: dict[str, Any] = {"role": role, "content": content}
        if message.get("tool_calls"):
            item["tool_calls"] = _cloud_redact_calls(message["tool_calls"])
        for key in ("tool_call_id", "name"):
            if message.get(key):
                item[key] = message[key]
        out.append(item)
    return out, dropped


def cloud_openai_payload(settings: dict, messages: list[dict], max_tokens: int,
                         temperature: float | None, stream: bool = True,
                         extra: dict | None = None) -> dict:
    """An OpenAI-compatible /chat/completions body (messages already minimised)."""
    body: dict[str, Any] = {"model": settings["model"], "messages": messages,
                            "stream": stream, "max_tokens": int(max_tokens)}
    if temperature is not None:
        body["temperature"] = float(temperature)
    if stream:
        body["stream_options"] = {"include_usage": True}
    for key, value in (extra or {}).items():
        if key in _CLOUD_PASSTHROUGH and value is not None:
            body[key] = value
    return body


def _cloud_text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text") or "") for p in content
                         if isinstance(p, dict) and p.get("type", "text") == "text")
    return ""


def _cloud_anthropic_blocks(content: Any) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content.strip() else []
    blocks: list[dict] = []
    for part in content if isinstance(content, list) else []:
        if not isinstance(part, dict):
            continue
        if part.get("type", "text") == "text":
            text = str(part.get("text") or "")
            if text.strip():
                blocks.append({"type": "text", "text": text})
        elif part.get("type") == "image_url":
            url = part.get("image_url")
            url = url.get("url") if isinstance(url, dict) else url
            if not isinstance(url, str):
                continue
            match = re.match(r"^data:(image/[A-Za-z0-9.+-]+);base64,(.*)$", url, re.S)
            if match:
                blocks.append({"type": "image", "source": {
                    "type": "base64", "media_type": match.group(1), "data": match.group(2)}})
            elif url.startswith(("https://", "http://")):
                blocks.append({"type": "image", "source": {"type": "url", "url": url}})
    return blocks


def _cloud_tool_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            value = json.loads(raw)
        except ValueError:
            return {"input": raw}
        return value if isinstance(value, dict) else {"input": value}
    return {}


def _cloud_anthropic_tools(tools: Any) -> list[dict]:
    out = []
    for tool in tools if isinstance(tools, list) else []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        schema = fn.get("parameters") or fn.get("input_schema")
        out.append({"name": name, "description": str(fn.get("description") or ""),
                    "input_schema": schema if isinstance(schema, dict)
                    else {"type": "object", "properties": {}}})
    return out


def _cloud_anthropic_choice(choice: Any) -> dict | None:
    if isinstance(choice, str):
        return {"auto": {"type": "auto"}, "required": {"type": "any"},
                "any": {"type": "any"}, "none": {"type": "none"}}.get(choice.lower())
    if isinstance(choice, dict):
        fn = choice.get("function") if isinstance(choice.get("function"), dict) else choice
        if fn.get("name"):
            return {"type": "tool", "name": str(fn["name"])}
    return None


def cloud_anthropic_payload(settings: dict, messages: list[dict], max_tokens: int,
                            temperature: float | None, stream: bool = True,
                            extra: dict | None = None) -> dict:
    """A /v1/messages body from OpenAI-shaped (already minimised) messages.

    System messages become the top-level system string; tool results become
    tool_result blocks when they answer a known tool_use id (plain text
    otherwise, which the API always accepts); consecutive turns of one role are
    merged, since the Messages API wants user and assistant to alternate.
    """
    system: list[str] = []
    turns: list[dict] = []
    known_ids: set[str] = set()

    def push(role: str, blocks: list[dict]) -> None:
        if not blocks:
            return
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(blocks)
        else:
            turns.append({"role": role, "content": list(blocks)})

    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            text = _cloud_text_of(content)
            if text.strip():
                system.append(text)
            continue
        if role == "tool":
            text = _cloud_text_of(content) or "(no output)"
            call_id = str(message.get("tool_call_id") or "")
            if call_id and call_id in known_ids:
                push("user", [{"type": "tool_result", "tool_use_id": call_id, "content": text}])
            else:
                push("user", [{"type": "text", "text": f"[tool result]\n{text}"}])
            continue
        blocks = _cloud_anthropic_blocks(content)
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                call_id = str(call.get("id") or "") or ("toolu_" + uuid.uuid4().hex[:20])
                known_ids.add(call_id)
                blocks.append({"type": "tool_use", "id": call_id,
                               "name": str(fn.get("name") or "tool"),
                               "input": _cloud_tool_args(fn.get("arguments"))})
            push("assistant", blocks)
        else:
            push("user", blocks)
    if not turns or turns[0]["role"] != "user":
        turns.insert(0, {"role": "user",
                         "content": [{"type": "text", "text": "(continuing the conversation)"}]})
    body: dict[str, Any] = {"model": settings["model"], "max_tokens": int(max_tokens),
                            "messages": turns, "stream": stream}
    if system:
        body["system"] = "\n\n".join(system)
    if temperature is not None:
        body["temperature"] = min(1.0, max(0.0, float(temperature)))
    opts = extra or {}
    stop = opts.get("stop")
    if isinstance(stop, str) and stop:
        body["stop_sequences"] = [stop]
    elif isinstance(stop, list) and stop:
        body["stop_sequences"] = [str(s) for s in stop if s][:8]
    if isinstance(opts.get("top_p"), (int, float)):
        body["top_p"] = float(opts["top_p"])
    tools = _cloud_anthropic_tools(opts.get("tools"))
    if tools:
        body["tools"] = tools
        choice = _cloud_anthropic_choice(opts.get("tool_choice"))
        if choice is not None:
            body["tool_choice"] = choice
    return body


def _cloud_request(settings: dict, messages: list[dict], max_tokens: int,
                   temperature: float | None, extra: dict | None) -> tuple[str, dict, dict]:
    """(url, headers, body) for the configured provider."""
    if settings["provider"] == "anthropic":
        headers = {"x-api-key": settings["api_key"],
                   "anthropic-version": CLOUD_ANTHROPIC_VERSION,
                   "content-type": "application/json", "accept": "text/event-stream"}
        body = cloud_anthropic_payload(settings, messages, max_tokens, temperature, True, extra)
    else:
        headers = {"content-type": "application/json", "accept": "text/event-stream"}
        if settings["api_key"]:
            headers["authorization"] = f"Bearer {settings['api_key']}"
        body = cloud_openai_payload(settings, messages, max_tokens, temperature, True, extra)
    return settings["url"], headers, body


# --------------------------------------------------------------------------- #
# Stream translation into the model client's typed events
# --------------------------------------------------------------------------- #
def _cloud_tool_delta(call: dict, position: int) -> dict:
    """An OpenAI delta.tool_calls entry in the shape model_client emits."""
    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
    index = call.get("index")
    if not isinstance(index, int) or isinstance(index, bool):
        index = position
    args = fn.get("arguments")
    if isinstance(args, (dict, list)):
        args = json.dumps(args, ensure_ascii=False)
    return {"type": "tool_call_delta", "index": index, "id": str(call.get("id") or ""),
            "name": str(fn.get("name") or ""), "arguments": str(args or ""),
            "source": "native", "done": False}


def _cloud_provider_error(data: Any) -> str:
    """The error type only: a provider's message may quote the prompt back."""
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        return str(err.get("type") or err.get("code") or "error")[:80]
    return "error"


async def _cloud_openai_events(resp: Any) -> AsyncGenerator[dict, None]:
    usage = None
    async for line in resp.aiter_lines():
        if not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if chunk == "[DONE]":
            break
        try:
            data = json.loads(chunk)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        if data.get("error"):
            raise RuntimeError(f"cloud provider error ({_cloud_provider_error(data)})")
        if data.get("usage"):
            usage = data["usage"]
        choices = data.get("choices") or []
        if not choices:
            continue
        choice = choices[0] or {}
        delta = choice.get("delta") or {}
        reasoning = delta.get("reasoning") or delta.get("reasoning_content") or ""
        content = delta.get("content") or ""
        if isinstance(reasoning, str) and reasoning:
            yield {"type": "reasoning", "text": reasoning}
        if isinstance(content, str) and content:
            yield {"type": "content", "text": content}
        calls = delta.get("tool_calls") or []
        for position, call in enumerate(calls if isinstance(calls, list) else []):
            if isinstance(call, dict):
                yield _cloud_tool_delta(call, position)
        if choice.get("finish_reason"):
            yield {"type": "finish", "reason": str(choice["finish_reason"])}
    if usage:
        yield {"type": "usage", "usage": usage}


_CLOUD_STOP_REASONS = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length",
                       "tool_use": "tool_calls", "pause_turn": "stop", "refusal": "stop"}


async def _cloud_anthropic_events(resp: Any) -> AsyncGenerator[dict, None]:
    prompt_tokens = 0
    output_tokens = 0
    tools: dict[Any, int] = {}
    async for line in resp.aiter_lines():
        if not line.startswith("data:"):
            continue
        try:
            data = json.loads(line[5:].strip())
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        etype = data.get("type")
        if etype == "message_start":
            usage = (data.get("message") or {}).get("usage") or {}
            prompt_tokens = sum(int(usage.get(k) or 0) for k in (
                "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
            output_tokens = int(usage.get("output_tokens") or 0)
        elif etype == "content_block_start":
            block = data.get("content_block") or {}
            btype = block.get("type")
            if btype == "tool_use":
                index = tools[data.get("index")] = len(tools)
                initial = block.get("input")
                yield {"type": "tool_call_delta", "index": index,
                       "id": str(block.get("id") or ""), "name": str(block.get("name") or ""),
                       "arguments": json.dumps(initial) if isinstance(initial, dict) and initial
                       else "", "source": "native", "done": False}
            elif btype == "text" and block.get("text"):
                yield {"type": "content", "text": str(block["text"])}
            elif btype == "thinking" and block.get("thinking"):
                yield {"type": "reasoning", "text": str(block["thinking"])}
        elif etype == "content_block_delta":
            delta = data.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta" and delta.get("text"):
                yield {"type": "content", "text": str(delta["text"])}
            elif dtype == "thinking_delta" and delta.get("thinking"):
                yield {"type": "reasoning", "text": str(delta["thinking"])}
            elif dtype == "input_json_delta" and data.get("index") in tools:
                yield {"type": "tool_call_delta", "index": tools[data["index"]], "id": "",
                       "name": "", "arguments": str(delta.get("partial_json") or ""),
                       "source": "native", "done": False}
        elif etype == "message_delta":
            usage = data.get("usage") or {}
            if usage.get("output_tokens") is not None:
                output_tokens = int(usage["output_tokens"] or 0)
            stop = (data.get("delta") or {}).get("stop_reason")
            if stop:
                yield {"type": "finish", "reason": _CLOUD_STOP_REASONS.get(stop, "stop")}
        elif etype == "message_stop":
            break
        elif etype == "error":
            raise RuntimeError(f"cloud provider error ({_cloud_provider_error(data)})")
    yield {"type": "usage", "usage": {"prompt_tokens": prompt_tokens,
                                      "completion_tokens": output_tokens,
                                      "total_tokens": prompt_tokens + output_tokens}}


# --------------------------------------------------------------------------- #
# Consent broker
# --------------------------------------------------------------------------- #
class CloudConsentBroker:
    """Pending consent prompts and their answers.

    Same shape as policy.ApprovalBroker, kept separate because a consent is
    about where a prompt goes, not about a tool call, and is stored in its own
    table. Thread-safe: the turn waits on the event loop, the consent route
    may answer from the same loop or a worker thread, so waiting polls a
    threading.Event rather than an asyncio.Event bound to one loop.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, dict] = {}

    def request(self, turn: TurnContext, *, reason: str, trigger: str,
                est_prompt_tokens: int, provider: str, model: str, scopes: list[str],
                timeout: float, db: Any = None) -> str:
        consent_id = "cc_" + uuid.uuid4().hex[:16]
        entry = {
            "id": consent_id, "turn_id": turn.turn_id, "turn": turn,
            "user_id": turn.user_id or SENTINEL_LOCAL_USER,
            "conversation_id": turn.conversation_id, "reason": reason, "trigger": trigger,
            "est_prompt_tokens": int(est_prompt_tokens), "provider": provider,
            "model": model, "scopes": list(scopes), "created": time.monotonic(),
            "timeout": max(1.0, float(timeout)), "db": db, "event": threading.Event(),
            "claimed": False, "decision": None, "scope": None,
        }
        self._prune()
        with self._lock:
            self._pending[consent_id] = entry
        log_event(_cloud_log, logging.INFO, "cloud.consent_requested", consent_id=consent_id,
                  user_id=entry["user_id"], conversation_id=entry["conversation_id"],
                  turn_id=entry["turn_id"], trigger=trigger,
                  est_prompt_tokens=entry["est_prompt_tokens"], provider=provider, model=model)
        return consent_id

    def _prune(self) -> None:
        now = time.monotonic()
        with self._lock:
            for key in [k for k, e in self._pending.items()
                        if now - e["created"] > e["timeout"] * 2 + 60]:
                self._pending.pop(key, None)

    def _view(self, entry: dict) -> dict:
        remaining = max(0, int(entry["timeout"] - (time.monotonic() - entry["created"])))
        return {"id": entry["id"], "turn_id": entry["turn_id"],
                "conversation_id": entry["conversation_id"], "reason": entry["reason"],
                "trigger": entry["trigger"], "est_prompt_tokens": entry["est_prompt_tokens"],
                "provider": entry["provider"], "model": entry["model"],
                "scopes": list(entry["scopes"]), "expires_in": remaining}

    def event(self, consent_id: str) -> dict:
        """The consent_required stream event for a pending request."""
        with self._lock:
            entry = self._pending.get(consent_id)
        if entry is None:
            return {"type": "consent_required", "id": consent_id, "scopes": []}
        return {"type": "consent_required", **{k: v for k, v in self._view(entry).items()
                                               if k not in ("turn_id", "conversation_id")}}

    def pending(self, user_id: str | None = None) -> list[dict]:
        with self._lock:
            entries = [e for e in self._pending.values() if not e["claimed"]]
        if user_id is not None:
            entries = [e for e in entries if e["user_id"] == (user_id or SENTINEL_LOCAL_USER)]
        return [self._view(e) for e in sorted(entries, key=lambda e: e["created"])]

    def _finish(self, entry: dict, decision: str, scope: str, by: str) -> None:
        entry["decision"], entry["scope"] = decision, scope
        entry["event"].set()
        log_event(_cloud_log, logging.INFO, "cloud.consent_decided", consent_id=entry["id"],
                  user_id=entry["user_id"], conversation_id=entry["conversation_id"],
                  turn_id=entry["turn_id"], decision=decision, scope=scope, by=by,
                  latency_ms=int((time.monotonic() - entry["created"]) * 1000))

    def _expire(self, entry: dict, decision: str, by: str) -> None:
        with self._lock:
            if entry["claimed"]:
                return
            entry["claimed"] = True
        self._finish(entry, decision, "once", by)

    def resolve(self, consent_id: str, user_id: str | None, decision: str,
                scope: str = "once", *, turn_id: str | None = None) -> bool:
        """Answer a pending consent. Owner-checked; False when not found or not yours.

        ValueError for a malformed answer (unknown decision, or a scope this
        request did not offer) so the route answers 400 rather than 404.
        """
        action = str(decision or "").strip().lower()
        if action not in ("allow", "deny"):
            raise ValueError("decision must be allow or deny")
        scope = str(scope or "once").strip().lower()
        with self._lock:
            entry = self._pending.get(consent_id)
            if entry is None or entry["claimed"]:
                return False
            if entry["user_id"] != (user_id or SENTINEL_LOCAL_USER):
                return False
            if turn_id is not None and entry["turn_id"] != turn_id:
                return False
            if action == "allow" and scope not in entry["scopes"]:
                raise ValueError(f"scope must be one of {', '.join(entry['scopes'])}")
            entry["claimed"] = True
        if action == "deny":
            scope = "once"
        # The grant is written BEFORE the waiter is released, so a second model
        # call of the same conversation already finds it.
        db = entry["db"] if entry["db"] is not None else _CLOUD_BOUND["db"]
        if action == "allow" and db is not None:
            try:
                grant_cloud_consent(db, entry["user_id"], scope, entry["conversation_id"])
            except Exception as exc:  # noqa: BLE001 - the answer itself still counts
                log(f"cloud fallback: could not save the {scope} consent: "
                    f"{redact_text(str(exc))}", logging.WARNING)
        self._finish(entry, action, scope, "user")
        return True

    async def wait(self, consent_id: str, timeout: float | None = None) -> tuple[str, str]:
        """(decision, scope): the owner's answer, "timeout", or deny on cancel."""
        with self._lock:
            entry = self._pending.get(consent_id)
        if entry is None:
            return "deny", "once"
        deadline = entry["created"] + (entry["timeout"] if timeout is None else float(timeout))
        while not entry["event"].is_set():
            turn = entry.get("turn")
            if turn is not None and turn.cancelled():
                self._expire(entry, "deny", "cancel")
            elif time.monotonic() >= deadline:
                self._expire(entry, "timeout", "timeout")
            else:
                await asyncio.sleep(0.05)
        with self._lock:
            self._pending.pop(consent_id, None)
        return entry["decision"] or "deny", entry["scope"] or "once"

    def cancel_turn(self, turn_id: str) -> int:
        with self._lock:
            entries = [e for e in self._pending.values()
                       if e["turn_id"] == turn_id and not e["claimed"]]
        for entry in entries:
            self._expire(entry, "deny", "cancel")
        return len(entries)

    def clear(self) -> None:
        """Forget everything (tests)."""
        with self._lock:
            self._pending.clear()


CONSENT = CloudConsentBroker()


def resolve_consent(consent_id: str, user_id: str | None, decision: str,
                    scope: str = "once", turn_id: str | None = None) -> bool:
    """The consent route's resolver (streams.chat_turn_consent looks it up).

    `turn_id` is the turn named in the URL: a consent id of another turn of the
    same user is not answered through this one's URL (the approval route makes
    the same check).
    """
    return CONSENT.resolve(consent_id, user_id, decision, scope, turn_id=turn_id)


# --------------------------------------------------------------------------- #
# One cloud generation
# --------------------------------------------------------------------------- #
class CloudOffload:
    """A prompt cleared for the cloud: streams it and accounts for it.

    `served` is False after events() only when the cloud failed before any
    output; the model client then runs the local path as if nothing happened.
    """

    def __init__(self, config: Any, settings: dict, trigger: CloudTrigger, *,
                 user_id: str, conversation_id: str | None, turn: TurnContext | None,
                 scope: str, kind: str, db: Any) -> None:
        self.config = config
        self.settings = settings
        self.trigger = trigger
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.turn = turn
        self.scope = scope
        self.kind = kind
        self.db = db
        self.served = False
        self.error = ""

    @property
    def provider(self) -> str:
        return self.settings["provider"]

    @property
    def model(self) -> str:
        return self.settings["model"]

    def _budget(self, max_tokens: int | None) -> int:
        # Not clamped to the local window: the point is the cloud's larger one.
        return max(16, int(max_tokens or getattr(self.config, "max_tokens", 512) or 512))

    async def events(self, messages: list[dict], max_tokens: int | None = None,
                     temperature: float | None = None, stats: Any = None,
                     extra: dict | None = None) -> AsyncGenerator[dict, None]:
        import httpx
        safe, dropped = cloud_minimise_messages(messages)
        temp = getattr(self.config, "temperature", None) if temperature is None else temperature
        url, headers, body = _cloud_request(self.settings, safe, self._budget(max_tokens),
                                            temp, extra)
        started = time.time()
        first_at: float | None = None
        usage: dict | None = None
        finish = ""
        out_chars = 0
        outcome = "cancelled"
        produced = False
        try:
            read_s = max(120.0, float(getattr(self.config, "stall_timeout", 60) or 60))
            timeout = httpx.Timeout(read_s, connect=15.0, pool=15.0)
            async with httpx.AsyncClient(timeout=timeout,
                                         transport=_CLOUD_HOOKS.get("transport")) as client:
                async with client.stream("POST", url, headers=headers, json=body) as resp:
                    if resp.status_code != 200:
                        try:
                            detail = _cloud_provider_error(json.loads(await resp.aread()))
                        except Exception:  # noqa: BLE001
                            detail = "error"
                        raise RuntimeError(
                            f"cloud provider returned {resp.status_code} ({detail})")
                    parser = (_cloud_anthropic_events if self.provider == "anthropic"
                              else _cloud_openai_events)
                    async for event in parser(resp):
                        etype = event["type"]
                        if etype == "usage":
                            usage = event["usage"]
                        elif etype == "finish":
                            finish = event["reason"]
                        else:
                            if first_at is None:
                                first_at = time.time()
                            produced = True
                            out_chars += len(event.get("text") or event.get("arguments") or "")
                        yield event
            outcome = "ok"
            self.served = True
        except Exception as exc:
            outcome = "failed"
            self.error = redact_text(f"{type(exc).__name__}: {exc}")[:300]
            if produced:
                self.served = True
                raise
            # Nothing reached the caller yet: say so and let the local path run.
            if self.turn is not None:
                self.turn.send({"type": "notice", "info": True,
                                "message": f"The cloud model is unavailable ({self.error}); "
                                           "continuing on the local model."})
        finally:
            # A consumer that leaves after output (the agent stops at a
            # completed tool call) was still served by the cloud.
            if produced:
                self.served = True
            self._account(safe, dropped, started, first_at, usage, finish, out_chars,
                          outcome, stats)

    async def complete(self, messages: list[dict], max_tokens: int | None = None,
                       temperature: float | None = None, stats: Any = None,
                       extra: dict | None = None) -> str | None:
        """The whole reply as text (native tool calls as protocol lines);
        None when the cloud failed before answering (run locally instead)."""
        text: list[str] = []
        calls: dict[int, dict] = {}
        async for event in self.events(messages, max_tokens, temperature, stats, extra):
            if event["type"] == "content":
                text.append(event["text"])
            elif event["type"] == "tool_call_delta":
                slot = calls.setdefault(event["index"], {"name": "", "arguments": ""})
                slot["name"] = slot["name"] or event.get("name") or ""
                slot["arguments"] += event.get("arguments") or ""
        if not self.served:
            return None
        answer = "".join(text)
        lines = [json.dumps({"tool": slot["name"], "args": _cloud_tool_args(slot["arguments"])},
                            ensure_ascii=False)
                 for _i, slot in sorted(calls.items()) if slot["name"].strip()]
        if lines:
            answer = (answer.rstrip() + "\n" if answer.strip() else "") + "\n".join(lines)
        return answer

    def _account(self, safe: list[dict], dropped: int, started: float,
                 first_at: float | None, usage: dict | None, finish: str, out_chars: int,
                 outcome: str, stats: Any) -> None:
        """Stats in place, the usage row, the routing row and one log event."""
        usage = usage if isinstance(usage, dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        if not usage and outcome != "failed":
            # No usage reported (the consumer left early, or a server that does
            # not send it): count the estimate, so the cap is never bypassed by
            # cancelling.
            prompt_tokens = messages_tokens(safe)
            completion_tokens = out_chars // CHARS_PER_TOKEN
        latency_ms = int((time.time() - started) * 1000)
        ttft_ms = int((first_at - started) * 1000) if first_at else None
        if stats is not None:
            try:
                stats.prompt_tokens = prompt_tokens or messages_tokens(safe)
                stats.completion_tokens = completion_tokens
                stats.ttft_ms = float(ttft_ms or 0)
                stats.total_ms = float(latency_ms)
                stats.from_server = bool(usage)
                stats.finish_reason = finish
            except Exception:  # noqa: BLE001 - a foreign stats object
                pass
        turn_id = self.turn.turn_id if self.turn is not None else None
        if self.served and self.turn is not None:
            self.turn.extras["cloud_used"] = {"provider": self.provider, "model": self.model}
        record_cloud_usage(self.db, user_id=self.user_id, conversation_id=self.conversation_id,
                           turn_id=turn_id, kind=self.kind, provider=self.provider,
                           model=self.model, prompt_tokens=prompt_tokens,
                           completion_tokens=completion_tokens, latency_ms=latency_ms,
                           status=outcome, consent_scope=self.scope,
                           trigger=self.trigger.reason)
        if self.db is not None:
            try:
                self.db.log_routing_event(
                    correlation_id=get_correlation_id(), task_id=turn_id,
                    user_id=self.user_id, conversation_id=self.conversation_id,
                    kind=self.kind, requested_model=getattr(self.config, "model", None),
                    selected_model=self.model, selected_node="cloud",
                    reason=f"cloud fallback ({self.trigger.reason}, consent {self.scope})",
                    status=outcome, attempt=0, duration_ms=latency_ms,
                    error=self.error or None)
            except Exception as exc:  # noqa: BLE001 - observability is not fatal
                log(f"cloud fallback: routing row failed: {redact_text(str(exc))}",
                    logging.WARNING)
        # Metadata only: never the prompt, the reply or a file name.
        log_event(_cloud_log, logging.WARNING if outcome == "failed" else logging.INFO,
                  "cloud.fallback", user_id=self.user_id, conversation_id=self.conversation_id,
                  turn_id=turn_id, kind=self.kind, provider=self.provider, model=self.model,
                  trigger=self.trigger.reason, consent_scope=self.scope, status=outcome,
                  prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                  latency_ms=latency_ms, ttft_ms=ttft_ms, dropped_parts=dropped,
                  error=self.error or None)


# --------------------------------------------------------------------------- #
# The entry point the model client calls
# --------------------------------------------------------------------------- #
def _cloud_skip(reason: str, user_id: str, conversation_id: str | None,
                trigger: CloudTrigger, level: int = logging.INFO) -> None:
    log_event(_cloud_log, level, "cloud.skipped", reason=reason, user_id=user_id,
              conversation_id=conversation_id, trigger=trigger.reason,
              est_prompt_tokens=trigger.prompt_tokens)


def _cloud_remember(turn: TurnContext | None, decision: str, scope: str,
                    settings: dict) -> None:
    """One answer per turn: an agent run makes many model calls, and asking
    again on each would be a prompt storm (or a silent re-ask after a no)."""
    if turn is not None:
        turn.extras["cloud"] = {"decision": decision, "scope": scope,
                                "provider": settings["provider"], "model": settings["model"]}


async def cloud_offload(config: Any, messages: list[dict], max_tokens: int | None = None,
                        kind: str = "chat", conversation_id: str | None = None,
                        extra: dict | None = None) -> CloudOffload | None:
    """A CloudOffload when this prompt should and may go to the cloud, else None.

    None is the overwhelmingly common answer and costs one attribute read when
    the feature is off. Never raises: any failure here keeps the prompt local.
    """
    if config is None or not getattr(config, "cloud_fallback_enabled", False):
        return None
    try:
        return await _cloud_route(config, messages, max_tokens, kind, conversation_id)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - the local path must still run
        log(f"cloud fallback: routing check failed, staying local: "
            f"{redact_text(str(exc))}", logging.WARNING)
        return None


async def cloud_preflight(config: Any, messages: list[dict], max_tokens: int | None = None,
                          kind: str = "agent", conversation_id: str | None = None) -> bool:
    """Settle the cloud question BEFORE a caller reshapes an oversized prompt.

    The agent splits an input too large for one local prefill into parts
    (CHUNK_LARGE_PROMPTS) before its first model call, so no single request
    would look oversized to cloud_offload and the user would never be offered
    the cloud for the very prompts it exists for. A caller about to split asks
    here first. True: this turn goes to the cloud (the user said yes, or a
    standing grant covers it), so keep the prompt whole; the model call that
    follows finds the answer remembered on the turn and does not ask again.
    False (off, not allowed, capped, unattended, declined): split as today.
    Never raises, and costs one attribute read when the feature is off.
    """
    return await cloud_offload(config, messages, max_tokens, kind, conversation_id) is not None


async def _cloud_route(config: Any, messages: list[dict], max_tokens: int | None,
                       kind: str, conversation_id: str | None) -> CloudOffload | None:
    db = _CLOUD_BOUND["db"]
    tps = (cloud_measured_prefill_tps(db)
           if float(getattr(config, "cloud_min_prefill_s", 0) or 0) > 0 else None)
    trigger = cloud_trigger(config, messages, max_tokens, tps)
    if trigger is None:
        return None
    turn = current_turn()
    user_id = ((turn.user_id if turn is not None and turn.user_id else None)
               or get_acting_user() or SENTINEL_LOCAL_USER)
    conv = ((turn.conversation_id if turn is not None and turn.conversation_id else None)
            or conversation_id or get_acting_conversation())
    settings = cloud_settings(config)
    if settings is None:
        _cloud_warn_once("unconfigured", "Cloud fallback is enabled but CLOUD_MODEL and "
                         "CLOUD_API_KEY (or an explicit CLOUD_BASE_URL) are not set; large "
                         "prompts stay local.")
        return None
    role, username = _cloud_identity(turn, db, user_id, config)
    if not cloud_user_allowed(config, user_id, role, username):
        _cloud_skip("not_allowed", user_id, conv, trigger, logging.DEBUG)
        return None
    cap = int(getattr(config, "cloud_max_tokens_per_user_month", 0) or 0)
    if cap:
        budget = int(max_tokens or getattr(config, "max_tokens", 512) or 512)
        # Without a database the cap cannot be checked, so it fails closed.
        used = cloud_month_usage(db, user_id) if db is not None else cap
        if used + trigger.prompt_tokens + budget > cap:
            _cloud_skip("cap", user_id, conv, trigger)
            if turn is not None and not (turn.extras.get("cloud") or {}).get("cap_noted"):
                turn.extras.setdefault("cloud", {})["cap_noted"] = True
                turn.send({"type": "notice", "info": True,
                           "message": "This prompt is large, but your monthly cloud allowance "
                                      "is used up; it runs on the local model."})
            return None
    if turn is None:
        return await _cloud_consent_for(config, settings, trigger, None, user_id, conv,
                                        kind, db)
    lock = turn.extras.get("_cloud_lock")
    if lock is None:
        lock = turn.extras["_cloud_lock"] = asyncio.Lock()
    async with lock:
        return await _cloud_consent_for(config, settings, trigger, turn, user_id, conv,
                                        kind, db)


async def _cloud_consent_for(config: Any, settings: dict, trigger: CloudTrigger,
                             turn: TurnContext | None, user_id: str, conv: str | None,
                             kind: str, db: Any) -> CloudOffload | None:
    def offload(scope: str) -> CloudOffload:
        return CloudOffload(config, settings, trigger, user_id=user_id, conversation_id=conv,
                            turn=turn, scope=scope, kind=kind, db=db)

    memo = (turn.extras.get("cloud") or {}) if turn is not None else {}
    if memo.get("decision") == "allow":
        return offload(memo.get("scope") or "once")
    if memo.get("decision") == "deny":
        return None
    per_user = str(getattr(config, "cloud_consent", "per_request")) == "per_user"
    interactive = bool(turn is not None and turn.interactive and turn.emit is not None)
    if per_user:
        # Unattended runs honour only "always": nobody is there to notice a
        # conversation grant being used from a scheduled task.
        standing = standing_cloud_consent(
            db, user_id, conv, ("conversation", "always") if interactive else ("always",))
        if standing is not None:
            _cloud_remember(turn, "allow", standing, settings)
            if turn is not None:
                turn.send({"type": "notice", "info": True,
                           "message": f"{trigger.detail} Sending it to {settings['provider']} "
                                      f"{settings['model']} under your saved cloud consent "
                                      f"({standing})."})
            return offload(standing)
    if not interactive:
        _cloud_skip("unattended", user_id, conv, trigger)
        _cloud_remember(turn, "deny", "once", settings)
        return None
    scopes = ["once"]
    if per_user:
        scopes += (["conversation"] if conv else []) + ["always"]
    timeout = float(getattr(config, "tool_approval_timeout_s", 120) or 120)
    consent_id = CONSENT.request(turn, reason=trigger.detail, trigger=trigger.reason,
                                 est_prompt_tokens=trigger.prompt_tokens,
                                 provider=settings["provider"], model=settings["model"],
                                 scopes=scopes, timeout=timeout, db=db)
    turn.send(CONSENT.event(consent_id))
    async with turn.waiting_on_human():
        decision, scope = await CONSENT.wait(consent_id)
    turn.send({"type": "consent_resolved", "id": consent_id, "decision": decision,
               "scope": scope})
    allowed = decision == "allow"
    _cloud_remember(turn, "allow" if allowed else "deny", scope, settings)
    if not allowed:
        _cloud_skip("denied" if decision == "deny" else decision, user_id, conv, trigger)
        return None
    return offload(scope)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
_CLOUD_MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def register_cloud(app, ctx) -> None:
    """GET/DELETE /api/cloud/consents (own), GET /api/cloud/usage (admin).

    The consent answer itself, POST /api/chat/turns/{turn_id}/consent/{id},
    lives with the other turn routes in streams.py and calls resolve_consent.
    """
    from fastapi.responses import JSONResponse

    bind_cloud(ctx.config, ctx.db)
    ctx.extras["cloud_consent"] = CONSENT
    db = ctx.db
    USER = ctx.USER
    ADMIN = ctx.ADMIN

    @app.get("/api/cloud/consents")
    async def cloud_consents(history: int = 0, user: dict = USER):
        cfg = ctx.config
        uid = user["id"]
        rows = await asyncio.to_thread(list_cloud_consents, db, uid, bool(history))
        used = await asyncio.to_thread(cloud_month_usage, db, uid)
        return {
            "consents": rows,
            "pending": CONSENT.pending(uid),
            "enabled": bool(getattr(cfg, "cloud_fallback_enabled", False)),
            "available": cloud_settings(cfg) is not None,
            "allowed": cloud_user_allowed(cfg, uid, user.get("role"), user.get("username")),
            "mode": getattr(cfg, "cloud_consent", "per_request"),
            "provider": getattr(cfg, "cloud_provider", ""),
            "model": getattr(cfg, "cloud_model", ""),
            "month": cloud_month(),
            "used_tokens": used,
            "cap_tokens": int(getattr(cfg, "cloud_max_tokens_per_user_month", 0) or 0),
        }

    @app.delete("/api/cloud/consents")
    async def cloud_consents_revoke_all(user: dict = USER):
        count = await asyncio.to_thread(revoke_cloud_consents, db, user["id"], None)
        log_event(_cloud_log, logging.INFO, "cloud.consent_revoked", user_id=user["id"],
                  count=count)
        return {"revoked": count}

    @app.delete("/api/cloud/consents/{consent_id}")
    async def cloud_consent_revoke(consent_id: int, user: dict = USER):
        count = await asyncio.to_thread(revoke_cloud_consents, db, user["id"], consent_id)
        if not count:
            return JSONResponse({"error": "consent not found"}, status_code=404)
        log_event(_cloud_log, logging.INFO, "cloud.consent_revoked", user_id=user["id"],
                  consent_row=consent_id, count=count)
        return {"revoked": count}

    @app.get("/api/cloud/usage")
    async def cloud_usage(month: str | None = None, _u: dict = ADMIN):
        if month is not None and not _CLOUD_MONTH_RE.match(month):
            return JSONResponse({"error": "month must be YYYY-MM"}, status_code=400)
        return await asyncio.to_thread(cloud_usage_report, db, month, ctx.config)


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'CLOUD_SCHEMA_NAME',
    'CLOUD_PROVIDERS',
    'CLOUD_CONSENT_MODES',
    'CLOUD_CONSENT_SCOPES',
    'CLOUD_ANTHROPIC_VERSION',
    'bind_cloud',
    'cloud_month',
    'cloud_settings',
    'cloud_user_allowed',
    'grant_cloud_consent',
    'list_cloud_consents',
    'revoke_cloud_consents',
    'standing_cloud_consent',
    'cloud_month_usage',
    'record_cloud_usage',
    'cloud_usage_report',
    'cloud_measured_prefill_tps',
    'CloudTrigger',
    'cloud_trigger',
    'cloud_minimise_messages',
    'cloud_openai_payload',
    'cloud_anthropic_payload',
    'CloudConsentBroker',
    'CONSENT',
    'resolve_consent',
    'CloudOffload',
    'cloud_offload',
    'cloud_preflight',
    'register_cloud',
    '_CLOUD_BOUND',
    '_CLOUD_HOOKS',
    '_CLOUD_TPS_CACHE',
    '_cloud_schema',
    '_cloud_openai_events',
    '_cloud_anthropic_events',
    '_cloud_tool_delta',
    '_cloud_request',
]
