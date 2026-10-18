"""Admin dashboards: time-bucketed aggregates over the app's own structured data.

Everything a dashboard needs was already being written, but nothing read it
back as a picture. SQLite holds one row per request and per generation
(`metrics`), per routing decision (`routing_events`), per tool call
(`tool_calls`), per approval decision (`tool_decisions`) and per rating
(`feedback`); obslog's EVENT_SINK holds the recent structured log records. This
module turns those into KPI tiles, time series and breakdowns (per user, per
model / adapter / backend, per node, per tool) behind three ADMIN routes:

  GET /api/dashboards/summary?window=1h|24h|7d
  GET /api/dashboards/timeseries?metric=a,b&window=&bucket=
  GET /api/dashboards/events?level=&limit=&event=

Design notes
------------
* **Mixed timestamps.** Columns filled by CURRENT_TIMESTAMP read
  "YYYY-MM-DD HH:MM:SS" (UTC), rows written from Python may be ISO 8601 with a
  "T" and an offset, and a few writers store epoch seconds. Comparing those
  strings directly misorders them, so every window filter and bucket goes
  through one normalising SQL expression (`_dash_epoch_sql`) built on
  datetime(): text through datetime(), numbers through 'unixepoch'.
* **No double counting.** The non-streaming chat route logs one "chat" row with
  the turn's token totals AND, on the agent lane, one "agent_step" row per
  generation. Token sums therefore count generation rows, compat rows, and a
  "chat" row only when no agent_step rows of the same turn exist.
* **Adapter attribution.** Feedback and metrics rows gain nullable
  adapter_id / backend / cached_prompt_tokens / queue_ms columns (own schema,
  additive). Adapter ids are FINGERPRINTS of the adapter weights, not the
  "latest" label: "latest" names a different adapter after every retrain, and
  an ELO rating (elo.py) or a thumbs-up rate keyed by it would silently move to
  the next model. The fingerprint survives the copy into backups/ (copytree
  keeps size and mtime), so a rating follows the weights, not the folder.
* **Tagging without touching the DAO.** Database.log_metric has ten call sites
  in three modules; rather than widen its signature everywhere, this module
  wraps it once: the original insert runs unchanged, then the same thread's
  connection tags the row it just wrote (last_insert_rowid() is per
  connection, and the Database keeps one connection per thread).
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import hashlib
import logging
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .database import *  # noqa: F401,F403
from .sysutil import *  # noqa: F401,F403


DASH_SCHEMA_NAME = "dashboards"

# window name -> (seconds, default bucket seconds). The default buckets keep
# every chart between 28 and 60 points: readable on a phone, cheap to draw.
DASH_WINDOWS: dict[str, tuple[int, int]] = {
    "1h": (3600, 60),
    "24h": (86400, 1800),
    "7d": (7 * 86400, 6 * 3600),
}
DASH_BUCKETS: dict[str, int] = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "6h": 21600, "1d": 86400,
}
_DASH_MAX_POINTS = 500

# Endpoints whose rows are one generation, not one request. Everything else in
# `metrics` is a request (chat, chat_stream, v1_chat, v1_responses, ...).
_DASH_GEN_ENDPOINTS = ("agent_step", "chat_stream_gen")


# --------------------------------------------------------------------------- #
# Schema                                                                       #
# --------------------------------------------------------------------------- #
def _dash_schema(conn) -> None:
    """Additive, idempotent: attribution columns plus the indexes windows use."""
    def columns(table: str) -> set[str]:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}

    have = columns("metrics")
    for column, ddl in (("adapter_id", "TEXT"), ("backend", "TEXT"),
                        ("cached_prompt_tokens", "INTEGER"), ("queue_ms", "REAL")):
        if column not in have:
            conn.execute(f"ALTER TABLE metrics ADD COLUMN {column} {ddl}")
    if "adapter_id" not in columns("feedback"):
        conn.execute("ALTER TABLE feedback ADD COLUMN adapter_id TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_dash_ts ON metrics(timestamp)")
    # The token de-duplication looks up a turn's agent_step rows by conversation.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_dash_conv "
                 "ON metrics(conversation_id, id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_dash_created "
                 "ON tool_calls(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_dash_adapter "
                 "ON feedback(adapter_id)")


register_schema(DASH_SCHEMA_NAME, _dash_schema)


# --------------------------------------------------------------------------- #
# Adapter identity                                                             #
# --------------------------------------------------------------------------- #
_DASH_RUNTIME: dict[str, Any] = {"mm": None, "config": None}
_DASH_ACTIVE_CACHE: dict[str, Any] = {"sig": None, "until": 0.0, "value": None}
_DASH_ACTIVE_TTL = 10.0
_DASH_LOCK = threading.Lock()


def base_adapter_key(model_id: str | None) -> str:
    """The id of "no adapter" for one base model (ratings never mix bases)."""
    name = str(model_id or "model").rstrip("/").split("/")[-1] or "model"
    return f"base@{name}"[:80]


def adapter_fingerprint(path: Path | None) -> str | None:
    """A stable id for an adapter's weights: size + mtime of its safetensors.

    Content hashing a multi-hundred-MB file on every lookup is too slow; size
    and whole-second mtime identify a promoted adapter well enough, and both
    survive shutil.copytree (copy2), so the backup of an adapter keeps its id.
    """
    if path is None:
        return None
    try:
        target = path / "adapters.safetensors"
        if not target.exists():
            files = [f for f in path.iterdir() if f.is_file()
                     and f.suffix in (".safetensors", ".npz")]
            if not files:
                return None
            target = max(files, key=lambda f: f.stat().st_mtime)
        st = target.stat()
    except OSError:
        return None
    digest = hashlib.sha256(f"{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()
    return "ad-" + digest[:10]


def adapter_key_for_choice(choice: str | None, model_id: str | None) -> str:
    """Map a UI adapter choice ("none", "latest", a backup name) onto its key."""
    path = resolve_adapter_quietly(choice)
    if path is None:
        return base_adapter_key(model_id)
    return adapter_fingerprint(path) or base_adapter_key(model_id)


def adapter_catalog(model_id: str | None = None) -> list[dict]:
    """Every loadable adapter with its key and current label, base model first."""
    model_id = model_id or getattr(_DASH_RUNTIME.get("config"), "model", None)
    out = [{"choice": "none", "key": base_adapter_key(model_id),
            "label": "base model", "base_model": model_id, "modified": None}]
    seen = {out[0]["key"]}
    try:
        entries = list_adapters()
    except Exception as exc:  # noqa: BLE001 - a listing failure is not fatal here
        log(f"dashboards: adapter listing failed ({exc})", logging.DEBUG)
        entries = []
    for entry in entries:
        key = adapter_fingerprint(Path(entry["path"]))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({"choice": entry["id"], "key": key,
                    "label": entry["id"], "base_model": entry.get("base_model"),
                    "modified": entry.get("modified")})
    return out


def adapter_labels(model_id: str | None = None) -> dict[str, str]:
    """key -> human label ("latest", a backup timestamp, "base model")."""
    labels = {row["key"]: row["label"] for row in adapter_catalog(model_id)}
    return labels


def _dash_label(key: str | None, labels: dict[str, str]) -> str:
    if not key:
        return "(untagged)"
    if key in labels:
        return labels[key]
    if key.startswith("base@"):
        return "base " + key[5:]
    return key + " (removed)"


def active_adapter(mm: Any = None) -> dict:
    """{key, choice, backend, caps} for what the local server answers with now.

    Cached for a few seconds: this runs after every metric insert, and
    resolving the adapter touches the filesystem.
    """
    mm = mm if mm is not None else _DASH_RUNTIME.get("mm")
    if mm is None:
        return {"key": None, "choice": None, "backend": None, "caps": []}
    sig = (id(mm), getattr(mm, "adapter_choice", None), getattr(mm, "model_id", None),
           getattr(getattr(mm, "backend", None), "name", None))
    now = time.monotonic()
    with _DASH_LOCK:
        cached = _DASH_ACTIVE_CACHE
        if cached["sig"] == sig and cached["until"] > now and cached["value"] is not None:
            return dict(cached["value"])
    spec = getattr(mm, "backend", None)
    try:
        caps = sorted(spec.caps_for(mm)) if spec is not None else []
    except Exception:  # noqa: BLE001
        caps = []
    try:
        served = bool(spec.serves_adapter(mm)) if spec is not None else True
    except Exception:  # noqa: BLE001
        served = True
    path = None
    if served:
        try:
            path = mm.adapter_path()
        except Exception:  # noqa: BLE001
            path = None
    key = adapter_fingerprint(path) if path is not None else None
    value = {
        "key": key or base_adapter_key(getattr(mm, "model_id", None)),
        "choice": getattr(mm, "adapter_choice", None) if path is not None else "none",
        "backend": getattr(spec, "name", None),
        "caps": caps,
    }
    with _DASH_LOCK:
        _DASH_ACTIVE_CACHE.update(sig=sig, until=now + _DASH_ACTIVE_TTL, value=dict(value))
    return value


def dash_reset_active_cache() -> None:
    """Forget the cached active adapter (after a swap, or in tests)."""
    with _DASH_LOCK:
        _DASH_ACTIVE_CACHE.update(sig=None, until=0.0, value=None)


def dash_bind_runtime(config: Any = None, model_manager: Any = None) -> None:
    if config is not None:
        _DASH_RUNTIME["config"] = config
    if model_manager is not None:
        _DASH_RUNTIME["mm"] = model_manager
    dash_reset_active_cache()


# --------------------------------------------------------------------------- #
# Per-request context: queue wait, caller                                      #
# --------------------------------------------------------------------------- #
# Set by the chat routes right after the admission slot is granted; read by
# the metric tag below. A ContextVar follows asyncio.to_thread (it copies the
# context), so the stream's off-loop log_metric calls see it too.
_DASH_QUEUE_MS: contextvars.ContextVar = contextvars.ContextVar("dash_queue_ms", default=None)


def dash_note_queue_wait(ms: float | None) -> None:
    try:
        _DASH_QUEUE_MS.set(None if ms is None else max(0.0, float(ms)))
    except (TypeError, ValueError):
        pass


async def dash_timed_acquire(gate: Any, **kwargs: Any) -> Any:
    """gate.acquire(**kwargs), noting how long the caller waited for its slot."""
    started = time.monotonic()
    ticket = await gate.acquire(**kwargs)
    dash_note_queue_wait((time.monotonic() - started) * 1000)
    return ticket


def dash_request_user_id(request: Any) -> str | None:
    """The authenticated caller of a finished request, for http.request logs."""
    try:
        user = getattr(request.state, "user", None)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(user, dict):
        return user.get("id")
    return None


# --------------------------------------------------------------------------- #
# Row tagging (wrapped DAO methods)                                            #
# --------------------------------------------------------------------------- #
def _dash_tag_last_metric(db: Any, stats: Any) -> None:
    active = active_adapter()
    cached = None
    if stats is not None and "cached_tokens" in (active.get("caps") or []):
        # Only a backend that REPORTS cached tokens gets a number: elsewhere 0
        # means "not reported", and counting it would drag the hit ratio to 0.
        cached = int(getattr(stats, "cached_tokens", 0) or 0)
    queue_ms = None
    if stats is not None and getattr(stats, "queue_ms", 0):
        queue_ms = float(stats.queue_ms)
    elif _DASH_QUEUE_MS.get() is not None:
        queue_ms = _DASH_QUEUE_MS.get()
    if active.get("key") is None and cached is None and queue_ms is None:
        return
    db.ensure_schema(DASH_SCHEMA_NAME)
    conn = db._connection()
    conn.execute(
        "UPDATE metrics SET adapter_id = ?, backend = ?, cached_prompt_tokens = ?, "
        "queue_ms = ? WHERE id = last_insert_rowid()",
        (active.get("key"), active.get("backend"), cached,
         None if queue_ms is None else round(queue_ms, 1)))
    conn.commit()


def _dash_adapter_for_conversation(db: Any, conversation_id: str | None) -> str | None:
    """The adapter that produced a conversation's latest answer, if tagged."""
    if not conversation_id:
        return None
    try:
        row = db.execute(
            "SELECT adapter_id FROM metrics WHERE conversation_id = ? "
            "AND adapter_id IS NOT NULL ORDER BY id DESC LIMIT 1",
            (conversation_id,)).fetchone()
    except Exception:  # noqa: BLE001 - column missing on an unmigrated db
        return None
    return row[0] if row else None


def dash_tag_feedback(db: Any, feedback_id: int | None,
                      conversation_id: str | None = None) -> str | None:
    """Stamp a feedback row with the adapter that wrote the rated answer.

    The answer was generated before the rating arrived, possibly before an
    adapter swap, so the conversation's own metric rows are asked first; the
    adapter serving now is the fallback. feedback_id None means "the row this
    thread just inserted". Never raises: attribution is best effort.
    """
    try:
        db.ensure_schema(DASH_SCHEMA_NAME)
        conversation_id = conversation_id or get_acting_conversation()
        key = (_dash_adapter_for_conversation(db, conversation_id)
               or active_adapter().get("key"))
        if not key:
            return None
        conn = db._connection()
        if feedback_id is None:
            conn.execute("UPDATE feedback SET adapter_id = ? WHERE id = last_insert_rowid() "
                         "AND adapter_id IS NULL", (key,))
        else:
            conn.execute("UPDATE feedback SET adapter_id = ? WHERE id = ? "
                         "AND adapter_id IS NULL", (key, int(feedback_id)))
        conn.commit()
        return key
    except Exception as exc:  # noqa: BLE001
        log(f"dashboards: feedback attribution skipped ({exc})", logging.DEBUG)
        return None


def _dash_install_hooks() -> None:
    """Wrap Database.log_metric / record_feedback once (idempotent)."""
    original_metric = Database.log_metric
    if not getattr(original_metric, "_dash_wrapped", False):
        @functools.wraps(original_metric)
        def log_metric(self, *args, **kwargs):
            result = original_metric(self, *args, **kwargs)
            stats = kwargs.get("stats") if "stats" in kwargs else (
                args[4] if len(args) > 4 else None)
            try:
                _dash_tag_last_metric(self, stats)
            except Exception as exc:  # noqa: BLE001 - metrics never fail a request
                log(f"dashboards: metric attribution skipped ({exc})", logging.DEBUG)
            return result
        log_metric._dash_wrapped = True
        Database.log_metric = log_metric

    original_feedback = Database.record_feedback
    if not getattr(original_feedback, "_dash_wrapped", False):
        @functools.wraps(original_feedback)
        def record_feedback(self, *args, **kwargs):
            result = original_feedback(self, *args, **kwargs)
            dash_tag_feedback(self, None)
            return result
        record_feedback._dash_wrapped = True
        Database.record_feedback = record_feedback


_dash_install_hooks()


# --------------------------------------------------------------------------- #
# Time helpers                                                                 #
# --------------------------------------------------------------------------- #
def _dash_epoch_sql(col: str) -> str:
    """SQL for a timestamp column as integer epoch seconds, whatever its format."""
    num = f"(CASE WHEN {col} > 100000000000 THEN {col} / 1000.0 ELSE {col} END)"
    txt = (f"(CASE WHEN CAST({col} AS REAL) > 100000000000 "
           f"THEN CAST({col} AS REAL) / 1000.0 ELSE CAST({col} AS REAL) END)")
    return (
        "CAST(strftime('%s', CASE "
        f"WHEN typeof({col}) IN ('integer', 'real') THEN datetime({num}, 'unixepoch') "
        f"WHEN {col} GLOB '[0-9]*' AND {col} NOT GLOB '*[-:]*' "
        f"THEN datetime({txt}, 'unixepoch') "
        f"ELSE datetime({col}) END) AS INTEGER)"
    )


def dash_window(window: str | None) -> tuple[str, int, int]:
    """(name, seconds, default bucket) for a window name; ValueError if unknown."""
    name = (window or "24h").strip().lower()
    if name not in DASH_WINDOWS:
        raise ValueError(f"window must be one of {', '.join(DASH_WINDOWS)}")
    seconds, bucket = DASH_WINDOWS[name]
    return name, seconds, bucket


def dash_bucket(bucket: str | int | None, window_s: int, default_s: int) -> int:
    """Bucket seconds from "5m"/"1h"/an int, clamped to at most 500 points."""
    if bucket in (None, ""):
        size = default_s
    elif isinstance(bucket, int) or str(bucket).isdigit():
        size = int(bucket)
    else:
        key = str(bucket).strip().lower()
        if key not in DASH_BUCKETS:
            raise ValueError(f"bucket must be one of {', '.join(DASH_BUCKETS)} or seconds")
        size = DASH_BUCKETS[key]
    size = max(60, size)
    while window_s / size > _DASH_MAX_POINTS:
        size *= 2
    return size


def _dash_pct(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile; None for no samples."""
    data = sorted(v for v in values if v is not None)
    if not data:
        return None
    index = min(len(data) - 1, max(0, int(round(q * (len(data) - 1)))))
    return round(float(data[index]), 1)


def _dash_avg(values: list[float]) -> float | None:
    data = [float(v) for v in values if v is not None]
    return round(sum(data) / len(data), 2) if data else None


def _dash_rate(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def _dash_table_exists(db: Any, name: str) -> bool:
    row = db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                     (name,)).fetchone()
    return row is not None


def _dash_error_type(status: Any, error: Any) -> str:
    """Group "4 generations queued" with "7 generations queued": digits -> #."""
    text = str(error or "").strip()
    if not text:
        return f"HTTP {status}"
    text = re.sub(r"\d+", "#", redact_text(text))
    return text[:80]


# The SQL predicate for rows whose tokens count toward totals (see the module
# docstring): generation rows, request rows of other endpoints that carry
# their own stats, and a non-streaming "chat" row only when its turn has no
# agent_step rows (the steps already carry the same tokens).
_DASH_TOKEN_ROWS = (
    "(m.endpoint IN ('agent_step', 'chat_stream_gen') "
    " OR (m.endpoint NOT IN ('chat', 'chat_stream', 'agent_step', 'chat_stream_gen') "
    "     AND m.prompt_tokens IS NOT NULL) "
    " OR (m.endpoint = 'chat' AND m.prompt_tokens IS NOT NULL AND NOT EXISTS ("
    "     SELECT 1 FROM metrics s WHERE s.conversation_id = m.conversation_id "
    "       AND s.endpoint = 'agent_step' AND s.id < m.id "
    "       AND s.id > COALESCE((SELECT MAX(p.id) FROM metrics p "
    "           WHERE p.conversation_id = m.conversation_id "
    "             AND p.endpoint IN ('chat', 'chat_stream') AND p.id < m.id), 0))))"
)


# --------------------------------------------------------------------------- #
# Aggregations                                                                 #
# --------------------------------------------------------------------------- #
def _dash_metric_rows(db: Any, since: int, until: int) -> list[dict]:
    ts = _dash_epoch_sql("m.timestamp")
    rows = db.execute(
        f"SELECT m.id, {ts} AS t, m.endpoint, m.duration_ms, m.status_code, m.error, "
        "m.prompt_tokens, m.completion_tokens, m.ttft_ms, m.decode_tps, m.model, "
        "m.user_id, m.adapter_id, m.backend, m.cached_prompt_tokens, m.queue_ms, "
        f"CASE WHEN {_DASH_TOKEN_ROWS} THEN 1 ELSE 0 END AS counts_tokens "
        f"FROM metrics m WHERE {ts} >= ? AND {ts} <= ? ORDER BY m.id",
        (since, until)).fetchall()
    return [dict(r) for r in rows]


def _dash_rows(db: Any, table: str, cols: str, ts_col: str, since: int, until: int) -> list[dict]:
    if not _dash_table_exists(db, table):
        return []
    ts = _dash_epoch_sql(ts_col)
    rows = db.execute(
        f"SELECT {cols}, {ts} AS t FROM {table} WHERE {ts} >= ? AND {ts} <= ? ORDER BY id",
        (since, until)).fetchall()
    return [dict(r) for r in rows]


def _dash_usernames(db: Any) -> dict[str, str]:
    try:
        return {r["id"]: r["username"] for r in db.execute("SELECT id, username FROM users")}
    except Exception:  # noqa: BLE001
        return {}


def _dash_is_request(row: dict) -> bool:
    return row["endpoint"] not in _DASH_GEN_ENDPOINTS


def dashboard_summary(db: Any, window: str = "24h", *, now: float | None = None,
                      gate: Any = None, sink: Any = None, model_id: str | None = None) -> dict:
    """KPIs and breakdowns for one window (see the module docstring)."""
    db.ensure_schema(DASH_SCHEMA_NAME)
    name, seconds, bucket = dash_window(window)
    until = int(now if now is not None else time.time())
    since = until - seconds
    rows = _dash_metric_rows(db, since, until)
    requests = [r for r in rows if _dash_is_request(r)]
    ok = [r for r in requests if (r["status_code"] or 0) < 400]
    failed = [r for r in requests if (r["status_code"] or 0) >= 400]
    token_rows = [r for r in rows if r["counts_tokens"]]
    gen_rows = [r for r in rows if r["ttft_ms"] is not None or r["decode_tps"] is not None]
    cache_rows = [r for r in token_rows if r["cached_prompt_tokens"] is not None]
    cached_sum = sum(int(r["cached_prompt_tokens"] or 0) for r in cache_rows)
    cache_prompt_sum = sum(int(r["prompt_tokens"] or 0) for r in cache_rows)
    queue_values = [r["queue_ms"] for r in requests if r["queue_ms"] is not None]
    labels = adapter_labels(model_id)
    names = _dash_usernames(db)

    errors: dict[tuple, int] = {}
    for r in failed:
        key = (int(r["status_code"] or 0), _dash_error_type(r["status_code"], r["error"]))
        errors[key] = errors.get(key, 0) + 1

    per_user: dict[str, dict] = {}
    for r in requests:
        uid = r["user_id"] or "(none)"
        u = per_user.setdefault(uid, {"user_id": uid, "username": names.get(uid, uid),
                                      "requests": 0, "errors": 0, "prompt_tokens": 0,
                                      "completion_tokens": 0, "_lat": []})
        u["requests"] += 1
        if (r["status_code"] or 0) >= 400:
            u["errors"] += 1
        elif r["duration_ms"] is not None:
            u["_lat"].append(r["duration_ms"])
    for r in token_rows:
        uid = r["user_id"] or "(none)"
        u = per_user.setdefault(uid, {"user_id": uid, "username": names.get(uid, uid),
                                      "requests": 0, "errors": 0, "prompt_tokens": 0,
                                      "completion_tokens": 0, "_lat": []})
        u["prompt_tokens"] += int(r["prompt_tokens"] or 0)
        u["completion_tokens"] += int(r["completion_tokens"] or 0)
    users_out = []
    for u in per_user.values():
        lat = u.pop("_lat")
        u["latency_p50"] = _dash_pct(lat, 0.5)
        u["error_rate"] = _dash_rate(u["errors"], u["requests"])
        users_out.append(u)
    users_out.sort(key=lambda x: (-x["requests"], -x["prompt_tokens"]))

    per_model: dict[tuple, dict] = {}
    for r in token_rows + [g for g in gen_rows if not g["counts_tokens"]]:
        key = (r["model"] or "(unknown)", r["adapter_id"], r["backend"])
        m = per_model.setdefault(key, {
            "model": key[0], "adapter_id": key[1], "adapter_label": _dash_label(key[1], labels),
            "backend": key[2] or "(untagged)", "generations": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "_ttft": [], "_tps": []})
        if r["counts_tokens"]:
            m["generations"] += 1
            m["prompt_tokens"] += int(r["prompt_tokens"] or 0)
            m["completion_tokens"] += int(r["completion_tokens"] or 0)
        if r["ttft_ms"] is not None:
            m["_ttft"].append(r["ttft_ms"])
        if r["decode_tps"] is not None:
            m["_tps"].append(r["decode_tps"])
    models_out = []
    for m in per_model.values():
        m["ttft_p50"] = _dash_pct(m.pop("_ttft"), 0.5)
        m["decode_tps_avg"] = _dash_avg(m.pop("_tps"))
        models_out.append(m)
    models_out.sort(key=lambda x: -(x["prompt_tokens"] + x["completion_tokens"]))

    routing = _dash_rows(db, "routing_events",
                         "id, selected_node, status, duration_ms, error, kind",
                         "created_at", since, until)
    nodes: dict[str, dict] = {}
    for r in routing:
        node = r["selected_node"] or "(none)"
        n = nodes.setdefault(node, {"node": node, "requests": 0, "failed": 0, "_d": []})
        n["requests"] += 1
        status = str(r["status"] or "").lower()
        if r["error"] or status in ("failed", "error", "timeout", "rejected"):
            n["failed"] += 1
        if r["duration_ms"] is not None:
            n["_d"].append(r["duration_ms"])
    nodes_out = []
    for n in nodes.values():
        d = n.pop("_d")
        n["error_rate"] = _dash_rate(n["failed"], n["requests"])
        n["duration_p50"] = _dash_pct(d, 0.5)
        n["duration_p95"] = _dash_pct(d, 0.95)
        nodes_out.append(n)
    nodes_out.sort(key=lambda x: -x["requests"])

    tool_rows = _dash_rows(db, "tool_calls", "id, name, duration_ms, error",
                           "created_at", since, until)
    tools: dict[str, dict] = {}
    for r in tool_rows:
        t = tools.setdefault(r["name"], {"name": r["name"], "calls": 0, "errors": 0, "_d": []})
        t["calls"] += 1
        if r["error"]:
            t["errors"] += 1
        if r["duration_ms"] is not None:
            t["_d"].append(r["duration_ms"])
    tools_out = []
    for t in tools.values():
        d = t.pop("_d")
        t["error_rate"] = _dash_rate(t["errors"], t["calls"])
        t["duration_avg"] = _dash_avg(d)
        t["duration_p95"] = _dash_pct(d, 0.95)
        tools_out.append(t)
    tools_out.sort(key=lambda x: -x["calls"])
    tool_errors = sum(t["errors"] for t in tools_out)

    decisions = _dash_rows(db, "tool_decisions", "id, tool, decision, latency_ms, approver_id",
                           "created_at", since, until)
    by_decision: dict[str, int] = {}
    by_tool: dict[str, dict] = {}
    human_latency = []
    for r in decisions:
        decision = str(r["decision"] or "?")
        by_decision[decision] = by_decision.get(decision, 0) + 1
        t = by_tool.setdefault(r["tool"], {"tool": r["tool"], "total": 0})
        t["total"] += 1
        t[decision] = t.get(decision, 0) + 1
        if r["approver_id"] and r["latency_ms"] is not None:
            human_latency.append(r["latency_ms"])
    approvals = {
        "total": len(decisions),
        "by_decision": [{"decision": k, "count": v}
                        for k, v in sorted(by_decision.items(), key=lambda kv: -kv[1])],
        "by_tool": sorted(by_tool.values(), key=lambda x: -x["total"])[:15],
        "human_latency_p50_ms": _dash_pct(human_latency, 0.5),
    }

    fb_rows = _dash_rows(db, "feedback",
                         "id, rating, corrected_response, pending_approval, adapter_id, source",
                         "created_at", since, until)
    fb_up = sum(1 for r in fb_rows if (r["rating"] or 0) > 0)
    fb_down = sum(1 for r in fb_rows if (r["rating"] or 0) < 0)
    fb_corr = sum(1 for r in fb_rows if (r["corrected_response"] or "").strip())
    by_adapter: dict[str, dict] = {}
    for r in fb_rows:
        key = r["adapter_id"] or ""
        a = by_adapter.setdefault(key, {"adapter_id": r["adapter_id"],
                                        "label": _dash_label(r["adapter_id"], labels),
                                        "total": 0, "up": 0, "down": 0})
        a["total"] += 1
        if (r["rating"] or 0) > 0:
            a["up"] += 1
        elif (r["rating"] or 0) < 0:
            a["down"] += 1
    fb_by_adapter = []
    for a in by_adapter.values():
        a["up_rate"] = _dash_rate(a["up"], a["up"] + a["down"])
        fb_by_adapter.append(a)
    fb_by_adapter.sort(key=lambda x: -x["total"])

    sink = sink if sink is not None else EVENT_SINK
    counts = sink.counts()
    top_events = sorted(counts["events"].items(), key=lambda kv: -kv[1])[:30]
    problems = sink.records(level="WARNING", limit=10)

    queue_live = None
    if gate is not None:
        try:
            queue_live = gate.snapshot(probe=False)
        except Exception:  # noqa: BLE001
            queue_live = None

    minutes = seconds / 60.0
    lat = [r["duration_ms"] for r in ok if r["duration_ms"] is not None]
    return {
        "window": name,
        "since": datetime.fromtimestamp(since, timezone.utc).isoformat(),
        "until": datetime.fromtimestamp(until, timezone.utc).isoformat(),
        "bucket_s": bucket,
        "kpis": {
            "requests": len(requests),
            "requests_per_min": round(len(requests) / minutes, 3) if minutes else 0,
            "errors": len(failed),
            "error_rate": _dash_rate(len(failed), len(requests)),
            "latency_ms": {"p50": _dash_pct(lat, 0.5), "p95": _dash_pct(lat, 0.95),
                           "avg": _dash_avg(lat)},
            "ttft_ms": {"p50": _dash_pct([r["ttft_ms"] for r in gen_rows], 0.5),
                        "p95": _dash_pct([r["ttft_ms"] for r in gen_rows], 0.95),
                        "avg": _dash_avg([r["ttft_ms"] for r in gen_rows])},
            "decode_tps": {"p50": _dash_pct([r["decode_tps"] for r in gen_rows], 0.5),
                           "avg": _dash_avg([r["decode_tps"] for r in gen_rows])},
            "generations": len(token_rows),
            "prompt_tokens": sum(int(r["prompt_tokens"] or 0) for r in token_rows),
            "completion_tokens": sum(int(r["completion_tokens"] or 0) for r in token_rows),
            "cached_prompt_tokens": cached_sum,
            "cache_hit_ratio": _dash_rate(cached_sum, cache_prompt_sum),
            "queue_ms": {"p50": _dash_pct(queue_values, 0.5),
                         "p95": _dash_pct(queue_values, 0.95), "samples": len(queue_values)},
            "tool_calls": len(tool_rows),
            "tool_errors": tool_errors,
            "tool_error_rate": _dash_rate(tool_errors, len(tool_rows)),
            "approvals": len(decisions),
            "feedback": {"total": len(fb_rows), "up": fb_up, "down": fb_down,
                         "corrections": fb_corr, "up_rate": _dash_rate(fb_up, fb_up + fb_down)},
            "active_users": len({r["user_id"] for r in requests if r["user_id"]}),
        },
        "errors_by_type": [{"status": k[0], "type": k[1], "count": v}
                           for k, v in sorted(errors.items(), key=lambda kv: -kv[1])][:20],
        "per_user": users_out[:50],
        "per_model": models_out[:30],
        "per_node": nodes_out,
        "tools": tools_out[:40],
        "approvals": approvals,
        "feedback_by_adapter": fb_by_adapter,
        "queue": queue_live,
        "events": {
            "top": [{"event": k, "count": v,
                     "levels": counts["event_levels"].get(k, {})} for k, v in top_events],
            "levels": counts["levels"],
            "since": counts["since"],
            "recent_problems": problems,
        },
        "active_adapter": {k: v for k, v in active_adapter().items() if k != "caps"},
    }


# metric name -> (label, unit). The timeseries route accepts any subset.
DASH_SERIES: dict[str, tuple[str, str]] = {
    "requests": ("Requests", ""),
    "errors": ("Errors", ""),
    "error_rate": ("Error rate", "%"),
    "latency_p50": ("Latency p50", "ms"),
    "latency_p95": ("Latency p95", "ms"),
    "ttft_p50": ("TTFT p50", "ms"),
    "ttft_p95": ("TTFT p95", "ms"),
    "decode_tps": ("Decode", "tok/s"),
    "prompt_tokens": ("Prompt tokens", "tok"),
    "completion_tokens": ("Completion tokens", "tok"),
    "cache_hit_ratio": ("Cache hit", "%"),
    "queue_p95": ("Queue wait p95", "ms"),
    "tool_calls": ("Tool calls", ""),
    "tool_errors": ("Tool errors", ""),
    "feedback_up": ("Thumbs up", ""),
    "feedback_down": ("Thumbs down", ""),
}


def dashboard_timeseries(db: Any, metrics: list[str] | str, window: str = "24h",
                         bucket: str | int | None = None, *,
                         now: float | None = None) -> dict:
    """Bucketed series for the named metrics; empty buckets are 0 or null."""
    db.ensure_schema(DASH_SCHEMA_NAME)
    if isinstance(metrics, str):
        metrics = [m.strip() for m in metrics.split(",") if m.strip()]
    metrics = list(dict.fromkeys(metrics or ["requests"]))
    unknown = [m for m in metrics if m not in DASH_SERIES]
    if unknown:
        raise ValueError(f"unknown metric(s): {', '.join(unknown)}; "
                         f"known: {', '.join(DASH_SERIES)}")
    if len(metrics) > 8:
        raise ValueError("at most 8 metrics per request")
    name, seconds, default_bucket = dash_window(window)
    size = dash_bucket(bucket, seconds, default_bucket)
    until = int(now if now is not None else time.time())
    first = (until - seconds) // size * size
    since = until - seconds
    starts = list(range(first, until + 1, size))
    index = {b: i for i, b in enumerate(starts)}

    def slot(t: Any) -> int | None:
        if t is None:
            return None
        return index.get(int(t) // size * size)

    want_metrics = any(m not in ("tool_calls", "tool_errors", "feedback_up", "feedback_down")
                       for m in metrics)
    rows = _dash_metric_rows(db, since, until) if want_metrics else []
    groups: dict[str, list[list]] = {}

    def bag(key: str) -> list[list]:
        if key not in groups:
            groups[key] = [[] for _ in starts]
        return groups[key]

    for r in rows:
        i = slot(r["t"])
        if i is None:
            continue
        if _dash_is_request(r):
            bag("req")[i].append(r)
            if (r["status_code"] or 0) < 400 and r["duration_ms"] is not None:
                bag("lat")[i].append(r["duration_ms"])
            if r["queue_ms"] is not None:
                bag("queue")[i].append(r["queue_ms"])
        if r["ttft_ms"] is not None:
            bag("ttft")[i].append(r["ttft_ms"])
        if r["decode_tps"] is not None:
            bag("tps")[i].append(r["decode_tps"])
        if r["counts_tokens"]:
            bag("tok")[i].append(r)

    if "tool_calls" in metrics or "tool_errors" in metrics:
        for r in _dash_rows(db, "tool_calls", "id, error", "created_at", since, until):
            i = slot(r["t"])
            if i is not None:
                bag("tool")[i].append(r)
    if "feedback_up" in metrics or "feedback_down" in metrics:
        for r in _dash_rows(db, "feedback", "id, rating", "created_at", since, until):
            i = slot(r["t"])
            if i is not None:
                bag("fb")[i].append(r)

    def series(metric: str) -> list:
        out = []
        for i in range(len(starts)):
            req = bag("req")[i]
            if metric == "requests":
                out.append(len(req))
            elif metric == "errors":
                out.append(sum(1 for r in req if (r["status_code"] or 0) >= 400))
            elif metric == "error_rate":
                bad = sum(1 for r in req if (r["status_code"] or 0) >= 400)
                out.append(round(100.0 * bad / len(req), 1) if req else None)
            elif metric == "latency_p50":
                out.append(_dash_pct(bag("lat")[i], 0.5))
            elif metric == "latency_p95":
                out.append(_dash_pct(bag("lat")[i], 0.95))
            elif metric == "ttft_p50":
                out.append(_dash_pct(bag("ttft")[i], 0.5))
            elif metric == "ttft_p95":
                out.append(_dash_pct(bag("ttft")[i], 0.95))
            elif metric == "decode_tps":
                out.append(_dash_avg(bag("tps")[i]))
            elif metric == "prompt_tokens":
                out.append(sum(int(r["prompt_tokens"] or 0) for r in bag("tok")[i]))
            elif metric == "completion_tokens":
                out.append(sum(int(r["completion_tokens"] or 0) for r in bag("tok")[i]))
            elif metric == "cache_hit_ratio":
                rows_c = [r for r in bag("tok")[i] if r["cached_prompt_tokens"] is not None]
                total = sum(int(r["prompt_tokens"] or 0) for r in rows_c)
                hit = sum(int(r["cached_prompt_tokens"] or 0) for r in rows_c)
                out.append(round(100.0 * hit / total, 1) if total else None)
            elif metric == "queue_p95":
                out.append(_dash_pct(bag("queue")[i], 0.95))
            elif metric == "tool_calls":
                out.append(len(bag("tool")[i]))
            elif metric == "tool_errors":
                out.append(sum(1 for r in bag("tool")[i] if r["error"]))
            elif metric == "feedback_up":
                out.append(sum(1 for r in bag("fb")[i] if (r["rating"] or 0) > 0))
            elif metric == "feedback_down":
                out.append(sum(1 for r in bag("fb")[i] if (r["rating"] or 0) < 0))
        return out

    return {
        "window": name,
        "bucket_s": size,
        "buckets": starts,
        "labels": [datetime.fromtimestamp(b, timezone.utc).isoformat() for b in starts],
        "series": [{"metric": m, "label": DASH_SERIES[m][0], "unit": DASH_SERIES[m][1],
                    "values": series(m)} for m in metrics],
    }


def dashboard_events(level: str | None = None, limit: int = 100,
                     event: str | None = None, *, sink: Any = None) -> dict:
    sink = sink if sink is not None else EVENT_SINK
    limit = max(1, min(500, int(limit or 100)))
    return {"events": sink.records(level=level or None, limit=limit, event=event or None),
            "counts": sink.counts()}


# --------------------------------------------------------------------------- #
# Routes                                                                       #
# --------------------------------------------------------------------------- #
def _dash_attach_sink() -> None:
    """Attach EVENT_SINK when configure_logging never ran (tests, embedding)."""
    root = logging.getLogger("local_llm")
    if EVENT_SINK not in root.handlers:
        root.addHandler(EVENT_SINK)


def register_dashboards(app, ctx) -> None:
    """Mount /api/dashboards/* (ADMIN)."""
    from fastapi.responses import JSONResponse

    dash_bind_runtime(ctx.config, ctx.model_manager)
    _dash_attach_sink()
    db = ctx.db
    ADMIN = ctx.ADMIN
    try:
        db.ensure_schema(DASH_SCHEMA_NAME)
    except Exception as exc:  # noqa: BLE001
        log(f"dashboards: schema not applied yet ({exc})", logging.WARNING)

    @app.get("/api/dashboards/summary")
    async def dashboards_summary(window: str = "24h", _a: dict = ADMIN):
        try:
            dash_window(window)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return await asyncio.to_thread(
            dashboard_summary, db, window, gate=ctx.generation_gate,
            model_id=getattr(ctx.config, "model", None))

    @app.get("/api/dashboards/timeseries")
    async def dashboards_timeseries(metric: str = "requests", window: str = "24h",
                                    bucket: str | None = None, _a: dict = ADMIN):
        try:
            dash_window(window)
            return await asyncio.to_thread(dashboard_timeseries, db, metric, window, bucket)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.get("/api/dashboards/events")
    async def dashboards_events(level: str | None = None, limit: int = 100,
                                event: str | None = None, _a: dict = ADMIN):
        if level and level.strip().upper() not in ("TRACE", "DEBUG", "INFO", "WARN",
                                                   "WARNING", "ERROR", "CRITICAL"):
            return JSONResponse({"error": "level must be DEBUG, INFO, WARNING or ERROR"},
                                status_code=400)
        return dashboard_events(level, limit, event)


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    "DASH_SCHEMA_NAME",
    "DASH_WINDOWS",
    "DASH_BUCKETS",
    "DASH_SERIES",
    "_DASH_GEN_ENDPOINTS",
    "_DASH_MAX_POINTS",
    "_DASH_RUNTIME",
    "_DASH_ACTIVE_CACHE",
    "_DASH_ACTIVE_TTL",
    "_DASH_LOCK",
    "_DASH_QUEUE_MS",
    "_DASH_TOKEN_ROWS",
    "_dash_schema",
    "base_adapter_key",
    "adapter_fingerprint",
    "adapter_key_for_choice",
    "adapter_catalog",
    "adapter_labels",
    "_dash_label",
    "active_adapter",
    "dash_reset_active_cache",
    "dash_bind_runtime",
    "dash_note_queue_wait",
    "dash_timed_acquire",
    "dash_request_user_id",
    "_dash_tag_last_metric",
    "_dash_adapter_for_conversation",
    "dash_tag_feedback",
    "_dash_install_hooks",
    "_dash_epoch_sql",
    "dash_window",
    "dash_bucket",
    "_dash_pct",
    "_dash_avg",
    "_dash_rate",
    "_dash_table_exists",
    "_dash_error_type",
    "_dash_metric_rows",
    "_dash_rows",
    "_dash_usernames",
    "_dash_is_request",
    "dashboard_summary",
    "dashboard_timeseries",
    "dashboard_events",
    "_dash_attach_sink",
    "register_dashboards",
]
