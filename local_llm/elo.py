"""Preference evaluation of LoRA adapters: pairwise matches and ELO ratings.

The feedback loop retrains the shared adapter from thumbs and corrections, and
every retrain moves the previous adapter into backups/. Nothing answered the
question that loop raises: is the new adapter actually better than the one it
replaced, or than the base model? A thumbs-up rate per adapter (dashboards.py)
is a weak answer, because each adapter answered different prompts from
different people. A pairwise comparison on the SAME prompt is the strong one,
and ELO turns a pile of pairwise outcomes into one comparable number.

Matches come from three sources, all stored in `adapter_matches`:

  * vote        a person picked A, B or tie between two answers to one prompt
                (POST /api/eval/compare/vote). Blind pairs prepared by a
                tournament are open to every user; a free-form vote naming its
                own adapters and answers is admin-only, since nothing proves the
                answers came from those adapters.
  * implicit    feedback that already exists: a thumbs-up for adapter X and a
                thumbs-down for adapter Y on the same normalised prompt is a
                match X won (at most one per prompt and ordered adapter pair, so
                a burst of thumbs on one prompt cannot outweigh the rest).
  * tournament  POST /api/eval/tournament (ADMIN) runs a fixed prompt set
                against each adapter in turn through the model manager's swap,
                then either asks the local model to judge each pair (blind,
                sides shuffled against position bias) or leaves the pairs for
                human voting.

Ratings are standard ELO (K=24, ties score 0.5, everyone starts at 1500),
updated as each match is stored and kept in `adapter_ratings`. They are a pure
function of the matches in insertion order, so POST /api/eval/ratings/recompute
can always rebuild them. Adapters are identified by the weight FINGERPRINT from
dashboards.py, never by the "latest" label, so a rating follows the weights
into backups/ instead of jumping to the next retrain.

Prompts are user content. A match stores the prompt's hash always and its text
only when LOG_CHAT_CONTENT=full (redacted, through content_for_log), the same
rule app.log follows. Tournament pairs keep their prompts and answers in full:
the prompt set is the admin's, and voters must be able to read the answers.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import logging
import random
import re
import threading
import time
import uuid
from typing import Any

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .database import *  # noqa: F401,F403
from .sysutil import *  # noqa: F401,F403
from .toolparse import *  # noqa: F401,F403
from .dashboards import *  # noqa: F401,F403

ELO_SCHEMA_NAME = "elo"
ELO_K = 24.0
ELO_INITIAL = 1500.0
ELO_WINNERS = ("a", "b", "tie")
ELO_JUDGES = ("human", "model")
ELO_SOURCES = ("vote", "implicit", "tournament")
# A prepared pair closes after this many votes: three opinions on one prompt
# are evidence, thirty are one prompt outweighing the whole set.
ELO_PAIR_MAX_VOTES = 3
ELO_TOURNAMENT_MAX_ADAPTERS = 6
ELO_TOURNAMENT_MAX_PROMPTS = 20
# Implicit matches are recomputed from feedback lazily, at most this often.
_ELO_SYNC_EVERY_S = 30.0

# The fixed set a tournament uses unless the admin supplies one. Short, varied
# and cheap to answer on an 8 GB machine: facts, reasoning, code, writing,
# instructions, refusal-free safety advice.
ELO_DEFAULT_PROMPTS: list[str] = [
    "Explain the difference between a process and a thread in two short paragraphs.",
    "A train leaves at 14:10 and arrives at 17:45. How long is the journey? Show your working.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Summarise the main causes of the French Revolution in five bullet points.",
    "Draft a polite email declining a meeting invitation and proposing two alternative times.",
    "What are three practical ways to reduce memory usage when running a language model on a laptop?",
    "Convert 72 degrees Fahrenheit to Celsius and explain the formula.",
    "Give step-by-step instructions to undo the last git commit while keeping the changes.",
]

_ELO_LOCK = threading.Lock()
_ELO_SYNC_STATE: dict[str, float] = {"at": 0.0}
_ELO_ID_RE = re.compile(r"^[A-Za-z0-9@._:/+-]{1,120}$")
_ELO_VERDICT_RE = re.compile(r"\b(TIE|A|B)\b")


# --------------------------------------------------------------------------- #
# Schema                                                                       #
# --------------------------------------------------------------------------- #
def _elo_schema(conn) -> None:
    """Additive, idempotent: matches, ratings and prepared pairs."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS adapter_matches (
            id INTEGER PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            prompt_hash TEXT NOT NULL,
            prompt TEXT,
            adapter_a TEXT NOT NULL,
            adapter_b TEXT NOT NULL,
            winner TEXT NOT NULL CHECK (winner IN ('a', 'b', 'tie')),
            judge TEXT NOT NULL DEFAULT 'human',
            user_id TEXT,
            source TEXT NOT NULL DEFAULT 'vote',
            ref TEXT,
            pair_id TEXT
        )
    """)
    # ref de-duplicates: one vote per user per pair, one implicit match per
    # prompt and adapter pair, one judged match per tournament slot.
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_adapter_matches_ref "
                 "ON adapter_matches(ref)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_adapter_matches_a ON adapter_matches(adapter_a)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_adapter_matches_b ON adapter_matches(adapter_b)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS adapter_ratings (
            adapter_id TEXT PRIMARY KEY,
            elo REAL NOT NULL DEFAULT 1500,
            games INTEGER NOT NULL DEFAULT 0,
            wins INTEGER NOT NULL DEFAULT 0,
            losses INTEGER NOT NULL DEFAULT 0,
            ties INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eval_pairs (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            run_id TEXT,
            prompt TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            answer_a TEXT NOT NULL,
            answer_b TEXT NOT NULL,
            adapter_a TEXT NOT NULL,
            adapter_b TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            votes INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_pairs_status ON eval_pairs(status, created_at)")


register_schema(ELO_SCHEMA_NAME, _elo_schema)


# --------------------------------------------------------------------------- #
# ELO math                                                                     #
# --------------------------------------------------------------------------- #
def elo_expected(rating_a: float, rating_b: float) -> float:
    """A's expected score against B (0..1)."""
    return 1.0 / (1.0 + 10.0 ** ((float(rating_b) - float(rating_a)) / 400.0))


def elo_update(rating_a: float, rating_b: float, winner: str,
               k: float = ELO_K) -> tuple[float, float]:
    """New (rating_a, rating_b) after one game. winner: "a", "b" or "tie"."""
    if winner not in ELO_WINNERS:
        raise ValueError("winner must be a, b or tie")
    score_a = {"a": 1.0, "b": 0.0, "tie": 0.5}[winner]
    expected_a = elo_expected(rating_a, rating_b)
    delta = k * (score_a - expected_a)
    # Zero-sum: what A gains B loses, so the pool's mean stays at the start.
    return float(rating_a) + delta, float(rating_b) - delta


def elo_normalize_prompt(text: str | None) -> str:
    """Case, whitespace and trailing punctuation do not make a different prompt."""
    norm = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    return norm.rstrip(" ?!.;:")


def elo_prompt_hash(text: str | None) -> str:
    return hashlib.sha256(elo_normalize_prompt(text).encode("utf-8")).hexdigest()[:16]


def _elo_prompt_for_storage(text: str | None) -> str | None:
    """The prompt as app.log would keep it: text only at LOG_CHAT_CONTENT=full."""
    rendered = content_for_log(text or "")
    return rendered if isinstance(rendered, str) else None


def _elo_valid_id(value: Any) -> bool:
    return isinstance(value, str) and bool(_ELO_ID_RE.match(value))


# --------------------------------------------------------------------------- #
# Storage                                                                      #
# --------------------------------------------------------------------------- #
def _elo_rating_row(conn, adapter_id: str) -> dict:
    row = conn.execute("SELECT elo, games, wins, losses, ties FROM adapter_ratings "
                       "WHERE adapter_id = ?", (adapter_id,)).fetchone()
    if row is None:
        return {"elo": ELO_INITIAL, "games": 0, "wins": 0, "losses": 0, "ties": 0}
    return {"elo": float(row[0]), "games": int(row[1]), "wins": int(row[2]),
            "losses": int(row[3]), "ties": int(row[4])}


def _elo_apply(conn, adapter_a: str, adapter_b: str, winner: str) -> dict:
    """Fold one game into adapter_ratings (caller holds _ELO_LOCK and commits)."""
    ra = _elo_rating_row(conn, adapter_a)
    rb = _elo_rating_row(conn, adapter_b)
    new_a, new_b = elo_update(ra["elo"], rb["elo"], winner)
    for key, rec, new, won, lost in (
            (adapter_a, ra, new_a, winner == "a", winner == "b"),
            (adapter_b, rb, new_b, winner == "b", winner == "a")):
        rec["elo"] = new
        rec["games"] += 1
        rec["wins"] += 1 if won else 0
        rec["losses"] += 1 if lost else 0
        rec["ties"] += 1 if winner == "tie" else 0
        conn.execute(
            "INSERT INTO adapter_ratings (adapter_id, elo, games, wins, losses, ties, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(adapter_id) DO UPDATE SET elo = excluded.elo, games = excluded.games, "
            "wins = excluded.wins, losses = excluded.losses, ties = excluded.ties, "
            "updated_at = excluded.updated_at",
            (key, round(new, 4), rec["games"], rec["wins"], rec["losses"], rec["ties"]))
    return {"a": {"adapter_id": adapter_a, "elo": round(ra["elo"], 1), "games": ra["games"]},
            "b": {"adapter_id": adapter_b, "elo": round(rb["elo"], 1), "games": rb["games"]}}


def elo_record_match(db: Any, adapter_a: str, adapter_b: str, winner: str, *,
                     prompt: str | None = None, prompt_hash: str | None = None,
                     judge: str = "human", source: str = "vote",
                     user_id: str | None = None, ref: str | None = None,
                     pair_id: str | None = None, store_prompt: bool = True) -> dict | None:
    """Store one match and update both ratings. None when `ref` was seen before."""
    if winner not in ELO_WINNERS:
        raise ValueError("winner must be a, b or tie")
    if judge not in ELO_JUDGES:
        raise ValueError("judge must be human or model")
    if source not in ELO_SOURCES:
        raise ValueError("unknown match source")
    if not adapter_a or not adapter_b or adapter_a == adapter_b:
        raise ValueError("a match needs two different adapters")
    db.ensure_schema(ELO_SCHEMA_NAME)
    phash = prompt_hash or elo_prompt_hash(prompt)
    stored = _elo_prompt_for_storage(prompt) if (store_prompt and prompt) else None
    with _ELO_LOCK:
        conn = db._connection()
        cur = conn.execute(
            "INSERT OR IGNORE INTO adapter_matches (prompt_hash, prompt, adapter_a, adapter_b, "
            "winner, judge, user_id, source, ref, pair_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (phash, stored, adapter_a, adapter_b, winner, judge, user_id, source, ref, pair_id))
        if cur.rowcount == 0:
            conn.commit()
            return None
        match_id = cur.lastrowid
        ratings = _elo_apply(conn, adapter_a, adapter_b, winner)
        conn.commit()
    log_event(get_logger("eval"), logging.INFO, "eval.match", match_id=match_id,
              source=source, judge=judge, winner=winner, adapter_a=adapter_a,
              adapter_b=adapter_b, user_id=user_id)
    return {"id": match_id, "adapter_a": adapter_a, "adapter_b": adapter_b,
            "winner": winner, "judge": judge, "source": source, "ratings": ratings}


def elo_recompute(db: Any) -> int:
    """Rebuild adapter_ratings by replaying every match in insertion order."""
    db.ensure_schema(ELO_SCHEMA_NAME)
    with _ELO_LOCK:
        conn = db._connection()
        conn.execute("DELETE FROM adapter_ratings")
        rows = conn.execute("SELECT adapter_a, adapter_b, winner FROM adapter_matches "
                            "ORDER BY id").fetchall()
        for row in rows:
            _elo_apply(conn, row[0], row[1], row[2])
        conn.commit()
    return len(rows)


def elo_ratings(db: Any, model_id: str | None = None, *, include_unrated: bool = True) -> list[dict]:
    """Every rated adapter (plus unrated loadable ones), best first, with labels."""
    db.ensure_schema(ELO_SCHEMA_NAME)
    try:
        catalog = adapter_catalog(model_id)
    except Exception:  # noqa: BLE001 - labels are cosmetic
        catalog = []
    labels = {c["key"]: c["label"] for c in catalog}
    choices = {c["key"]: c["choice"] for c in catalog}
    try:
        active_key = active_adapter().get("key")
    except Exception:  # noqa: BLE001
        active_key = None
    rows = db.execute("SELECT adapter_id, elo, games, wins, losses, ties, updated_at "
                      "FROM adapter_ratings").fetchall()
    out = []
    seen = set()
    for r in rows:
        seen.add(r["adapter_id"])
        out.append({"adapter_id": r["adapter_id"], "label": _dash_label(r["adapter_id"], labels),
                    "choice": choices.get(r["adapter_id"]), "elo": round(float(r["elo"]), 1),
                    "games": int(r["games"]), "wins": int(r["wins"]), "losses": int(r["losses"]),
                    "ties": int(r["ties"]), "updated_at": r["updated_at"], "provisional": False})
    if include_unrated:
        for c in catalog:
            if c["key"] not in seen:
                out.append({"adapter_id": c["key"], "label": c["label"], "choice": c["choice"],
                            "elo": ELO_INITIAL, "games": 0, "wins": 0, "losses": 0, "ties": 0,
                            "updated_at": None, "provisional": True})
    # Rated adapters first, best first; the unrated ones trail at the start value.
    out.sort(key=lambda x: (x["games"] == 0, -x["elo"]))
    for rank, item in enumerate(out, 1):
        decided = item["wins"] + item["losses"]
        item["rank"] = rank
        item["win_rate"] = round(item["wins"] / decided, 4) if decided else None
        item["active"] = item["adapter_id"] == active_key
        # Fewer than 10 games: the number mostly reflects the starting value.
        item["provisional"] = item["provisional"] or item["games"] < 10
    return out


def elo_matches(db: Any, limit: int = 100, adapter: str | None = None,
                model_id: str | None = None) -> list[dict]:
    db.ensure_schema(ELO_SCHEMA_NAME)
    limit = max(1, min(500, int(limit or 100)))
    if adapter:
        rows = db.execute(
            "SELECT * FROM adapter_matches WHERE adapter_a = ? OR adapter_b = ? "
            "ORDER BY id DESC LIMIT ?", (adapter, adapter, limit)).fetchall()
    else:
        rows = db.execute("SELECT * FROM adapter_matches ORDER BY id DESC LIMIT ?",
                          (limit,)).fetchall()
    try:
        labels = adapter_labels(model_id)
    except Exception:  # noqa: BLE001
        labels = {}
    out = []
    for r in rows:
        item = {k: r[k] for k in ("id", "created_at", "prompt_hash", "prompt", "adapter_a",
                                  "adapter_b", "winner", "judge", "user_id", "source", "pair_id")}
        item["label_a"] = _dash_label(r["adapter_a"], labels)
        item["label_b"] = _dash_label(r["adapter_b"], labels)
        out.append(item)
    return out


# --------------------------------------------------------------------------- #
# Implicit matches from feedback                                               #
# --------------------------------------------------------------------------- #
def elo_sync_implicit(db: Any, *, limit: int = 20000, force: bool = True) -> int:
    """Turn opposite ratings of one prompt across adapters into matches.

    Idempotent (ref-keyed), so running it again only adds what is new. Returns
    the number of matches added. Needs feedback.adapter_id (dashboards schema):
    rows rated before attribution existed carry no adapter and are skipped.
    """
    now = time.monotonic()
    if not force and now - _ELO_SYNC_STATE["at"] < _ELO_SYNC_EVERY_S:
        return 0
    _ELO_SYNC_STATE["at"] = now
    db.ensure_schema(DASH_SCHEMA_NAME)
    db.ensure_schema(ELO_SCHEMA_NAME)
    rows = db.execute(
        "SELECT id, user_prompt, rating, adapter_id FROM feedback "
        "WHERE adapter_id IS NOT NULL AND rating IS NOT NULL AND rating != 0 "
        "ORDER BY id DESC LIMIT ?", (max(1, int(limit)),)).fetchall()
    groups: dict[str, dict] = {}
    for r in rows:
        norm = elo_normalize_prompt(r["user_prompt"])
        if not norm:
            continue
        g = groups.setdefault(norm, {"up": set(), "down": set(), "prompt": r["user_prompt"]})
        (g["up"] if int(r["rating"]) > 0 else g["down"]).add(r["adapter_id"])
    added = 0
    for norm, g in groups.items():
        phash = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]
        for winner_key in sorted(g["up"]):
            for loser_key in sorted(g["down"]):
                if winner_key == loser_key:
                    continue
                result = elo_record_match(
                    db, winner_key, loser_key, "a", prompt=g["prompt"], prompt_hash=phash,
                    judge="human", source="implicit",
                    ref=f"imp:{phash}:{winner_key}>{loser_key}")
                if result is not None:
                    added += 1
    if added:
        log_event(get_logger("eval"), logging.INFO, "eval.implicit.sync", added=added,
                  prompts=len(groups))
    return added


# --------------------------------------------------------------------------- #
# Prepared pairs and votes                                                     #
# --------------------------------------------------------------------------- #
def elo_create_pair(db: Any, prompt: str, answer_a: str, answer_b: str,
                    adapter_a: str, adapter_b: str, run_id: str | None = None) -> str:
    db.ensure_schema(ELO_SCHEMA_NAME)
    pair_id = uuid.uuid4().hex[:16]
    db.execute(
        "INSERT INTO eval_pairs (id, run_id, prompt, prompt_hash, answer_a, answer_b, "
        "adapter_a, adapter_b) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (pair_id, run_id, prompt, elo_prompt_hash(prompt), answer_a, answer_b,
         adapter_a, adapter_b))
    db.commit()
    return pair_id


def elo_next_pair(db: Any, user_id: str) -> dict:
    """The oldest open pair this user has not voted on, blind (no adapter ids)."""
    db.ensure_schema(ELO_SCHEMA_NAME)
    unvoted = ("p.status = 'open' AND NOT EXISTS (SELECT 1 FROM adapter_matches m "
               "WHERE m.pair_id = p.id AND m.user_id = ?)")
    remaining = db.execute(f"SELECT COUNT(*) FROM eval_pairs p WHERE {unvoted}",
                           (user_id,)).fetchone()[0]
    row = db.execute(
        "SELECT id, created_at, run_id, prompt, answer_a, answer_b FROM eval_pairs p "
        f"WHERE {unvoted} ORDER BY p.created_at, p.id LIMIT 1", (user_id,)).fetchone()
    return {"pair": dict(row) if row else None, "remaining": int(remaining or 0)}


def _elo_resolve_adapter(value: Any, catalog: list[dict], db: Any) -> str | None:
    """A catalog choice ("latest", a backup name) or a known key -> its key."""
    if not _elo_valid_id(value):
        return None
    for entry in catalog:
        if value == entry["key"] or value == entry["choice"]:
            return entry["key"]
    if value.startswith("base@"):
        return value
    row = db.execute("SELECT 1 FROM adapter_ratings WHERE adapter_id = ?", (value,)).fetchone()
    return value if row else None


def elo_vote(db: Any, body: dict, user: dict, *, model_id: str | None = None) -> tuple[int, dict]:
    """Validate and store one A/B vote. Returns (http status, payload)."""
    db.ensure_schema(ELO_SCHEMA_NAME)
    winner = str(body.get("winner") or "").strip().lower()
    if winner not in ELO_WINNERS:
        return 400, {"error": "winner must be a, b or tie"}
    user_id = str(user.get("id") or "")
    pair_id = body.get("pair_id")
    if pair_id is not None:
        if not _elo_valid_id(pair_id):
            return 400, {"error": "invalid pair_id"}
        row = db.execute("SELECT * FROM eval_pairs WHERE id = ?", (pair_id,)).fetchone()
        if row is None or row["status"] != "open":
            return 404, {"error": "no open pair with that id"}
        result = elo_record_match(
            db, row["adapter_a"], row["adapter_b"], winner, prompt=row["prompt"],
            prompt_hash=row["prompt_hash"], judge="human", source="tournament",
            user_id=user_id, ref=f"pair:{pair_id}:{user_id}", pair_id=pair_id,
            store_prompt=False)
        if result is None:
            return 409, {"error": "you already voted on this pair"}
        with _ELO_LOCK:
            db.execute("UPDATE eval_pairs SET votes = votes + 1, status = CASE WHEN votes + 1 >= ? "
                       "THEN 'closed' ELSE status END WHERE id = ?",
                       (ELO_PAIR_MAX_VOTES, pair_id))
            db.commit()
        labels = adapter_labels(model_id)
        result["revealed"] = {"a": _dash_label(row["adapter_a"], labels),
                              "b": _dash_label(row["adapter_b"], labels)}
        return 200, {"match": result}

    # Free-form: the caller names the adapters and supplies the answers.
    if user.get("role") != "admin":
        return 403, {"error": "only admins can submit a free-form comparison; "
                              "vote on a prepared pair instead"}
    prompt = body.get("prompt")
    answer_a = body.get("answer_a")
    answer_b = body.get("answer_b")
    for name, value, cap in (("prompt", prompt, 20000), ("answer_a", answer_a, 50000),
                             ("answer_b", answer_b, 50000)):
        if not isinstance(value, str) or not value.strip():
            return 400, {"error": f"{name} is required"}
        if len(value) > cap:
            return 400, {"error": f"{name} is longer than {cap} characters"}
    catalog = adapter_catalog(model_id)
    key_a = _elo_resolve_adapter(body.get("adapter_a"), catalog, db)
    key_b = _elo_resolve_adapter(body.get("adapter_b"), catalog, db)
    if key_a is None or key_b is None:
        return 400, {"error": "adapter_a and adapter_b must name known adapters "
                              "(a choice such as latest or none, or an adapter id)"}
    if key_a == key_b:
        return 400, {"error": "adapter_a and adapter_b are the same weights"}
    digest = hashlib.sha256(json.dumps([elo_normalize_prompt(prompt), answer_a, answer_b,
                                        key_a, key_b]).encode("utf-8")).hexdigest()[:16]
    result = elo_record_match(db, key_a, key_b, winner, prompt=prompt, judge="human",
                              source="vote", user_id=user_id, ref=f"vote:{user_id}:{digest}")
    if result is None:
        return 409, {"error": "this comparison was already recorded"}
    return 200, {"match": result}


# --------------------------------------------------------------------------- #
# Offline tournament                                                           #
# --------------------------------------------------------------------------- #
class _EloTournament:
    """State of the one tournament a server may run at a time."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.cancel = False
        self.state: dict = {"running": False, "run_id": None, "started_at": None,
                            "finished_at": None, "phase": "", "done": 0, "total": 0,
                            "judge": None, "adapters": [], "prompts": 0, "matches": 0,
                            "pairs": 0, "skipped": 0, "error": "", "log": []}

    def update(self, **fields: Any) -> None:
        with self.lock:
            self.state.update(fields)

    def bump(self, field: str, by: int = 1) -> None:
        with self.lock:
            self.state[field] = int(self.state.get(field) or 0) + by

    def line(self, text: str) -> None:
        with self.lock:
            self.state["log"] = (self.state["log"] + [str(text)])[-40:]

    def snapshot(self) -> dict:
        with self.lock:
            return {**self.state, "running": self.running, "log": list(self.state["log"])}


def _elo_busy(ctx: Any) -> str:
    """Why a tournament must not start now, or "" when it may."""
    if (getattr(ctx.retrain_manager, "status", None) or {}).get("running"):
        return "A training run is in progress and owns the model server."
    bench = ctx.extras.get("bench_job")
    if bench is not None and getattr(bench, "running", False):
        return "A benchmark sweep owns the model server."
    try:
        queue = ctx.generation_gate.snapshot() or {}
    except Exception:  # noqa: BLE001 - a plain GenerationGate has no snapshot
        queue = {}
    running = int(queue.get("running") or 0)
    waiting = int(queue.get("waiting") or 0)
    in_flight = int(getattr(ctx.tasks, "chat_in_flight", 0) or 0)
    if running or waiting or in_flight:
        return (f"{max(running, in_flight)} generation(s) running and {waiting} waiting. "
                "Let them finish first.")
    runs = list(getattr(ctx.tasks, "by_task", {}) or {})
    if runs:
        return f"{len(runs)} task run(s) in flight. Cancel them first."
    return ""


def _elo_default_load(ctx: Any, choice: str) -> None:
    """Point the shared model server at one adapter and wait until it serves."""
    manager = ctx.model_manager
    changed = manager.swap(None, choice)
    if changed or getattr(manager, "status", "") != "ready":
        manager.restart()
    if getattr(manager, "status", "") != "ready":
        raise RuntimeError(f"model server did not come up with adapter {choice!r} "
                           f"(status: {getattr(manager, 'status', '?')})")
    dash_reset_active_cache()


def _elo_default_generate(ctx: Any, messages: list[dict], max_tokens: int,
                          temperature: float) -> str:
    """One non-streaming completion from the local model server (blocking)."""
    import urllib.request
    client = ctx.model_client
    body = client.payload(messages, False, max_tokens=max_tokens, temperature=temperature)
    timeout = client.request_timeout(messages, max_tokens)
    req = urllib.request.Request(
        client._local_chat_url(), data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - loopback only
        doc = json.loads(resp.read().decode("utf-8") or "{}")
    choices = doc.get("choices") or [{}]
    return str(((choices[0] or {}).get("message") or {}).get("content") or "")


_ELO_JUDGE_SYSTEM = (
    "You are an impartial judge comparing two answers to the same question. "
    "Prefer the answer that is correct, helpful and clear; length is not a merit. "
    "Reply with exactly one word: A, B or TIE.")


def _elo_judge_messages(prompt: str, first: str, second: str) -> list[dict]:
    return [
        {"role": "system", "content": _ELO_JUDGE_SYSTEM},
        {"role": "user", "content": (
            f"Question:\n{prompt}\n\nAnswer A:\n{first[:4000]}\n\nAnswer B:\n{second[:4000]}"
            "\n\nWhich answer is better? Reply A, B or TIE.")},
    ]


def elo_parse_verdict(text: str | None) -> str | None:
    """The judge's reply -> "a" / "b" / "tie", or None when it is not a verdict."""
    body = strip_reasoning(str(text or "")).strip().upper()
    match = _ELO_VERDICT_RE.search(body)
    if not match:
        return None
    return {"A": "a", "B": "b", "TIE": "tie"}[match.group(1)]


def _elo_clean_answer(text: str) -> str:
    return strip_reasoning(str(text or "")).strip()[:20000]


def _elo_tournament_body(job: _EloTournament, ctx: Any, opts: dict) -> None:
    manager = ctx.model_manager
    load = ctx.extras.get("eval_load_adapter") or _elo_default_load
    generate = ctx.extras.get("eval_generate") or _elo_default_generate
    db = ctx.db
    run_id = opts["run_id"]
    prompts: list[str] = opts["prompts"]
    entrants: list[dict] = opts["entrants"]
    judge = opts["judge"]
    rng = random.Random(opts.get("seed"))
    original_choice = getattr(manager, "adapter_choice", None)
    original_status = getattr(manager, "status", None)
    loaded: str | None = None
    elog = get_logger("eval")
    # The judge is the base model when it is loaded anyway; generating the base
    # model's answers last lets the judging phase reuse it without a restart.
    judge_choice = opts.get("judge_adapter") or "none"
    order = sorted(entrants, key=lambda e: e["choice"] == judge_choice)
    answers: dict[str, list[str]] = {}
    try:
        total = len(order) * len(prompts)
        if judge == "model":
            total += len(prompts) * (len(entrants) * (len(entrants) - 1) // 2)
        job.update(total=total, phase="answering")
        for entrant in order:
            job.line(f"Loading {entrant['label']}.")
            load(ctx, entrant["choice"])
            loaded = entrant["choice"]
            # Chats see "Model not ready" while the tournament owns the server,
            # instead of being answered by whichever adapter is under test.
            manager.status = "evaluating"
            texts = []
            for prompt in prompts:
                if job.cancel:
                    raise RuntimeError("cancelled")
                job.update(current=f"{entrant['label']}: {prompt[:60]}")
                texts.append(_elo_clean_answer(generate(
                    ctx, [{"role": "user", "content": prompt}], opts["max_tokens"], 0.3)))
                job.bump("done")
            answers[entrant["key"]] = texts
        pairs = list(itertools.combinations([e["key"] for e in entrants], 2))
        if judge == "model":
            job.update(phase="judging")
            if loaded != judge_choice:
                job.line("Loading the judge (base model).")
                load(ctx, judge_choice)
                loaded = judge_choice
                manager.status = "evaluating"
            for key_x, key_y in pairs:
                for index, prompt in enumerate(prompts):
                    if job.cancel:
                        raise RuntimeError("cancelled")
                    # Blind and shuffled: the judge sees no adapter names, and
                    # which side each adapter sits on is a coin flip per prompt.
                    key_a, key_b = (key_x, key_y) if rng.random() < 0.5 else (key_y, key_x)
                    ans_a = answers[key_a][index]
                    ans_b = answers[key_b][index]
                    verdict = elo_parse_verdict(generate(
                        ctx, _elo_judge_messages(prompt, ans_a, ans_b), 8, 0.0))
                    job.bump("done")
                    if verdict is None:
                        job.bump("skipped")
                        continue
                    stored = elo_record_match(
                        db, key_a, key_b, verdict, prompt=prompt, judge="model",
                        source="tournament", ref=f"t:{run_id}:{index}:{key_x}:{key_y}",
                        store_prompt=False)
                    if stored is not None:
                        job.bump("matches")
        else:
            job.update(phase="pairing")
            for key_x, key_y in pairs:
                for index, prompt in enumerate(prompts):
                    key_a, key_b = (key_x, key_y) if rng.random() < 0.5 else (key_y, key_x)
                    elo_create_pair(db, prompt, answers[key_a][index], answers[key_b][index],
                                    key_a, key_b, run_id=run_id)
                    job.bump("pairs")
        job.update(phase="done")
        snap = job.snapshot()
        log_event(elog, logging.INFO, "eval.tournament.done", run_id=run_id, judge=judge,
                  adapters=len(entrants), prompts=len(prompts), matches=snap["matches"],
                  pairs=snap["pairs"], skipped=snap["skipped"])
    except Exception as exc:  # noqa: BLE001
        message = redact_text(f"{type(exc).__name__}: {exc}")
        job.update(error=message, phase="failed")
        log_event(elog, logging.WARNING, "eval.tournament.failed", run_id=run_id,
                  error=message)
    finally:
        try:
            if loaded is not None and loaded != original_choice:
                job.line("Restoring the serving adapter.")
                load(ctx, original_choice if original_choice is not None else "none")
            elif loaded is not None:
                manager.status = original_status or "ready"
        except Exception as exc:  # noqa: BLE001
            log(f"Could not restore the serving adapter after the tournament: {exc}",
                logging.ERROR)
            previous = job.snapshot()["error"]
            job.update(error=(previous + " " if previous else "")
                       + redact_text(f"restore failed: {exc}"))
            try:
                manager.status = original_status or "stopped"
            except Exception:  # noqa: BLE001
                pass
        dash_reset_active_cache()
        with job.lock:
            job.running = False
            job.cancel = False
            job.state.update(running=False, finished_at=iso(utc_now()), current="")


def _elo_tournament_opts(body: dict, catalog: list[dict]) -> dict:
    """Validate a tournament request (ValueError with a user-facing message)."""
    judge = str(body.get("judge") or "model").strip().lower()
    if judge not in ELO_JUDGES:
        raise ValueError("judge must be model or human")
    by_choice = {c["choice"]: c for c in catalog}
    wanted = body.get("adapters")
    if wanted in (None, "", []):
        entrants = list(catalog)
    else:
        if not isinstance(wanted, list) or not all(isinstance(w, str) for w in wanted):
            raise ValueError("adapters must be a list of adapter choices")
        entrants = []
        for choice in wanted:
            if choice not in by_choice:
                raise ValueError(f"unknown adapter {choice[:80]!r}")
            entrants.append(by_choice[choice])
    seen: set[str] = set()
    unique = []
    for entrant in entrants:
        if entrant["key"] not in seen:
            seen.add(entrant["key"])
            unique.append(entrant)
    if len(unique) < 2:
        raise ValueError("a tournament needs at least two different adapters "
                         "(train one, or keep a backup)")
    if len(unique) > ELO_TOURNAMENT_MAX_ADAPTERS:
        raise ValueError(f"at most {ELO_TOURNAMENT_MAX_ADAPTERS} adapters per tournament")
    prompts = body.get("prompts")
    if prompts in (None, "", []):
        prompts = list(ELO_DEFAULT_PROMPTS)
    if (not isinstance(prompts, list) or not prompts
            or not all(isinstance(p, str) and p.strip() for p in prompts)):
        raise ValueError("prompts must be a list of non-empty strings")
    if len(prompts) > ELO_TOURNAMENT_MAX_PROMPTS:
        raise ValueError(f"at most {ELO_TOURNAMENT_MAX_PROMPTS} prompts")
    if any(len(p) > 2000 for p in prompts):
        raise ValueError("each prompt must be at most 2000 characters")
    try:
        max_tokens = int(body.get("max_tokens") or 256)
    except (TypeError, ValueError):
        raise ValueError("max_tokens must be a number") from None
    return {"judge": judge, "entrants": [dict(e) for e in unique],
            "prompts": [p.strip() for p in prompts],
            "max_tokens": max(32, min(1024, max_tokens)),
            "seed": body.get("seed") if isinstance(body.get("seed"), int) else None,
            "run_id": uuid.uuid4().hex[:12]}


# --------------------------------------------------------------------------- #
# Routes                                                                       #
# --------------------------------------------------------------------------- #
def register_elo(app, ctx) -> None:
    """Mount /api/eval/* (ratings, matches, votes, tournament)."""
    from fastapi.responses import JSONResponse

    db = ctx.db
    config = ctx.config
    job = _EloTournament()
    app.state.eval_job = job
    ctx.extras["eval_job"] = job
    try:
        db.ensure_schema(ELO_SCHEMA_NAME)
    except Exception as exc:  # noqa: BLE001
        log(f"eval: schema not applied yet ({exc})", logging.WARNING)

    def model_id() -> str | None:
        return getattr(config, "model", None)

    @app.get("/api/eval/ratings")
    async def eval_ratings(_a: dict = ctx.ADMIN):
        def load() -> dict:
            elo_sync_implicit(db, force=False)
            open_pairs = db.execute("SELECT COUNT(*) FROM eval_pairs WHERE status = 'open'"
                                    ).fetchone()[0]
            total = db.execute("SELECT COUNT(*) FROM adapter_matches").fetchone()[0]
            by_source = {r[0]: r[1] for r in db.execute(
                "SELECT source, COUNT(*) FROM adapter_matches GROUP BY source")}
            return {"ratings": elo_ratings(db, model_id()), "k": ELO_K, "initial": ELO_INITIAL,
                    "matches": int(total or 0), "by_source": by_source,
                    "open_pairs": int(open_pairs or 0),
                    "active": {k: v for k, v in active_adapter().items() if k != "caps"}}
        return await asyncio.to_thread(load)

    @app.post("/api/eval/ratings/recompute")
    async def eval_recompute(_a: dict = ctx.ADMIN):
        def run() -> dict:
            added = elo_sync_implicit(db, force=True)
            return {"replayed": elo_recompute(db), "implicit_added": added}
        return await asyncio.to_thread(run)

    @app.get("/api/eval/matches")
    async def eval_matches(limit: int = 100, adapter: str | None = None,
                           _a: dict = ctx.ADMIN):
        if adapter is not None and not _elo_valid_id(adapter):
            return JSONResponse({"error": "invalid adapter id"}, status_code=400)
        rows = await asyncio.to_thread(elo_matches, db, limit, adapter, model_id())
        return {"matches": rows}

    @app.get("/api/eval/compare/next")
    async def eval_compare_next(user: dict = ctx.USER):
        return await asyncio.to_thread(elo_next_pair, db, str(user.get("id") or ""))

    @app.post("/api/eval/compare/vote")
    async def eval_compare_vote(body: dict, user: dict = ctx.USER):
        if not isinstance(body, dict):
            return JSONResponse({"error": "expected a JSON object"}, status_code=400)
        try:
            status, payload = await asyncio.to_thread(elo_vote, db, body, user,
                                                      model_id=model_id())
        except ValueError as exc:
            return JSONResponse({"error": redact_text(str(exc))}, status_code=400)
        if status != 200:
            return JSONResponse(payload, status_code=status)
        return payload

    @app.get("/api/eval/tournament")
    def eval_tournament_status(_a: dict = ctx.ADMIN):
        return job.snapshot()

    @app.post("/api/eval/tournament/cancel")
    def eval_tournament_cancel(_a: dict = ctx.ADMIN):
        with job.lock:
            if not job.running:
                return JSONResponse({"error": "no tournament is running"}, status_code=409)
            job.cancel = True
        return {"cancelling": True}

    @app.post("/api/eval/tournament")
    async def eval_tournament_start(body: dict, _a: dict = ctx.ADMIN):
        if not isinstance(body, dict) or body.get("confirm") is not True:
            return JSONResponse({"error": "A tournament takes over the model server (chats get "
                                          "'Model not ready' until it ends). Send "
                                          "{\"confirm\": true} to proceed."}, status_code=400)
        catalog = await asyncio.to_thread(adapter_catalog, model_id())
        try:
            opts = _elo_tournament_opts(body, catalog)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        busy = _elo_busy(ctx)
        if busy:
            return JSONResponse({"error": busy}, status_code=409)
        with job.lock:
            if job.running:
                return JSONResponse({"error": "A tournament is already running."},
                                    status_code=409)
            job.running = True
            job.cancel = False
            job.state.update(running=True, run_id=opts["run_id"], started_at=iso(utc_now()),
                             finished_at=None, phase="starting", done=0, total=0,
                             judge=opts["judge"], prompts=len(opts["prompts"]),
                             adapters=[{"key": e["key"], "label": e["label"],
                                        "choice": e["choice"]} for e in opts["entrants"]],
                             matches=0, pairs=0, skipped=0, error="", log=[], current="")
        log_event(get_logger("eval"), logging.INFO, "eval.tournament.start",
                  run_id=opts["run_id"], judge=opts["judge"], adapters=len(opts["entrants"]),
                  prompts=len(opts["prompts"]))
        threading.Thread(target=_elo_tournament_body, args=(job, ctx, opts),
                         daemon=True, name="eval-tournament").start()
        return JSONResponse({"started": True, "status": job.snapshot()}, status_code=202)


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    "ELO_SCHEMA_NAME",
    "ELO_K",
    "ELO_INITIAL",
    "ELO_WINNERS",
    "ELO_JUDGES",
    "ELO_SOURCES",
    "ELO_PAIR_MAX_VOTES",
    "ELO_TOURNAMENT_MAX_ADAPTERS",
    "ELO_TOURNAMENT_MAX_PROMPTS",
    "ELO_DEFAULT_PROMPTS",
    "_ELO_SYNC_EVERY_S",
    "_ELO_LOCK",
    "_ELO_SYNC_STATE",
    "_ELO_ID_RE",
    "_ELO_VERDICT_RE",
    "_ELO_JUDGE_SYSTEM",
    "_elo_schema",
    "elo_expected",
    "elo_update",
    "elo_normalize_prompt",
    "elo_prompt_hash",
    "_elo_prompt_for_storage",
    "_elo_valid_id",
    "_elo_rating_row",
    "_elo_apply",
    "elo_record_match",
    "elo_recompute",
    "elo_ratings",
    "elo_matches",
    "elo_sync_implicit",
    "elo_create_pair",
    "elo_next_pair",
    "_elo_resolve_adapter",
    "elo_vote",
    "_EloTournament",
    "_elo_busy",
    "_elo_default_load",
    "_elo_default_generate",
    "_elo_judge_messages",
    "elo_parse_verdict",
    "_elo_clean_answer",
    "_elo_tournament_body",
    "_elo_tournament_opts",
    "register_elo",
]
