"""Built-in evaluation and regression suite (item 19).

Six suites score the agent the way users meet it, and run on every model or
adapter change:

  tool_calling  the real Agent picks tools from a sandboxed ToolRegistry: tool
                choice, argument validity, no invented tool names, completion
  rag           fixture documents indexed into a throwaway per-user knowledge
                base: the answer contains the fact and cites its source path
  multi_turn    task continuity: create an artifact, follow-up edits, continue
                after truncation, language drift (taskstate / codecheck)
  routing       the intent, lane, tool set and retrieval rule a message gets
                (intents.classify, upgrade/ROUTING_AND_TOOLING.md P9); no model
                in scripted mode, the whole agent against the model in live
  parsers       toolparse fixtures across every model family format (no model)
  api           /v1 compat request/response shapes against an in-process app
                with a scripted model (no model)

Modes
-----
scripted  a deterministic fake model replays each case's canned outputs. It
          proves the harness end to end and runs in CI; its runs are keyed to
          model "scripted" so they never mix with real baselines.
live      the real local model through a cluster-less ModelClient with its own
          connection pool (the server's shared pool is bound to the server's
          event loop and must not be rebound from this thread). Refused while
          the model is not ready, while anything is generating, training, or a
          benchmark or ELO tournament owns the model server. parsers and api
          need no model and run the same way in both modes.

The sandbox: a temp workspace and SQLite file, a real ToolRegistry built from a
copy of the live Config (MCP servers, cloud fallback, skills and workspaces
off; approval policy bypassed), and a `call` that records every call. Only a
fixed list of local read-only tools really runs (against the sandbox); every
other tool is stubbed with a canned or generic result. The api suite builds a
whole app, which rebinds process-wide state (policy, cloud, dashboards), so
inside the server it runs in a child process; the CLI and tests run it here.

Results: per case pass / score / latency / tokens / checks, per suite
aggregates, stored in eval_runs + eval_results and as JSON under
DATA_DIR/evals/. Baselines are keyed by model + adapter fingerprint + mode; a
regression is any suite whose score drops more than EVAL_MAX_REGRESSION below
the baseline's.

Triggers: `deploy.py eval [--suite a,b] [--live] [--save-baseline] [--compare]
[--json]`; ADMIN routes POST /api/evals/run, GET /api/evals/runs,
GET /api/evals/runs/{id}, GET|POST /api/evals/baseline, POST /api/evals/cancel;
and EVAL_ON_PROMOTE=1, where training.py calls eval_after_promote() after a
retrain promoted an adapter and restarted the server: the live model suites run
against the previous adapter's baseline and a regression rolls the adapter back
through RetrainManager._rollback_adapter, recording why in training_meta.json.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import is_dataclass, replace as _eval_dc_replace
from pathlib import Path
from typing import Any

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .database import *  # noqa: F401,F403
from .sysutil import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .toolparse import *  # noqa: F401,F403
from .textutil import *  # noqa: F401,F403
from .taskstate import *  # noqa: F401,F403
from .codecheck import *  # noqa: F401,F403
from .llm import *  # noqa: F401,F403
from .tools import *  # noqa: F401,F403
from .policy import *  # noqa: F401,F403
from .model_client import *  # noqa: F401,F403
from .agent import *  # noqa: F401,F403
from .dashboards import *  # noqa: F401,F403
from .api_tokens import *  # noqa: F401,F403
from .intents import *  # noqa: F401,F403
from .eval_suites import *  # noqa: F401,F403


EVAL_SCHEMA_NAME = "evals"
EVAL_SUITES = ("tool_calling", "rag", "multi_turn", "routing", "parsers", "api")
# Suites that need a model in live mode; the other two are deterministic in
# both modes (routing is too in scripted mode, where classify needs no model).
EVAL_MODEL_SUITES = ("tool_calling", "rag", "multi_turn", "routing")
# What the EVAL_ON_PROMOTE gate runs after a retrain. Not routing: its intent,
# lane, tool and retrieval scores come from the lexicon, which an adapter
# cannot move, and a live routing run is some seventy agent turns.
EVAL_PROMOTE_SUITES = ("tool_calling", "rag", "multi_turn")
EVAL_MODES = ("scripted", "live")
EVAL_SCRIPTED_MODEL = "scripted"
EVAL_SCRIPTED_ADAPTER = "scripted"
EVAL_API_CASES = ("models_list", "auth_required", "invalid_request", "chat_completion",
                  "chat_stream", "tool_call", "anthropic_message", "anthropic_stream",
                  "anthropic_error")

_EVAL_USER = "eval-runner"
_EVAL_UNSET: Any = object()
# Seconds one case may take: a scripted case is instant, a live one generates.
_EVAL_CASE_TIMEOUT = {"scripted": 60.0, "live": 420.0}
# How long the promotion gate waits for the server to go idle before skipping.
_EVAL_PROMOTE_WAIT_S = 90.0
_EVAL_KEEP_JSON = 200

# Tools that really run in the sandbox: local, read-only, and confined to the
# sandbox workspace or the sandbox database. Everything else is stubbed.
_EVAL_REAL_TOOLS = frozenset({
    "calculator", "current_time", "list_files", "read_file", "file_info", "search_files",
    "search_docs", "recall_memory", "recall_feedback", "search_past_conversations",
})

# The agent configuration every eval turn runs with, in both modes, so runs
# stay comparable whatever the admin tuned: the router, fast path and
# auto-iterate/auto-continue lanes off (tool choice is the agent loop's), one
# step minimum, no skills, no approvals (the sandbox stubs side effects), no
# MCP servers (configuring them is process-wide) and never the cloud.
_EVAL_CONFIG_OVERRIDES: dict = {
    "agent_enabled": True, "knowledge_triage": False, "fast_path": False,
    "auto_iterate_rounds": 0, "auto_continue": False, "incremental_reasoning": False,
    "chunk_large_prompts": False, "skills_enabled": False, "agent_min_steps": 1,
    "agent_max_steps": 4, "tool_approval_mode": "off", "workspaces_enabled": False,
    "agent_tools": "", "mcp_servers": "", "cloud_fallback_enabled": False,
    "rag_enabled": True, "rag_scope": "", "temperature": 0.0, "allow_python": False,
    "allow_shell": False, "task_state_enabled": True,
}
_EVAL_SCRIPTED_OVERRIDES: dict = {"ready_wait_timeout": 0.05, "resilient_retries": 1}

_EVAL_RUNTIME: dict = {
    "ctx": None,             # ApiContext once register_evals ran (busy checks)
    "dir": None,             # override for DATA_DIR/evals (tests)
    "api_runner": None,      # replaces the api-suite child process (tests)
    "client_factory": None,  # replaces EvalLiveClient(config) (tests)
}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _eval_lookup(module: str, name: str) -> Any:
    """A name from another module, resolved at call time (package or bundle)."""
    if __package__:
        try:
            return getattr(importlib.import_module(f"{__package__}.{module}"), name, None)
        except Exception:  # noqa: BLE001 - optional at run time
            return None
    return globals().get(name)


def _eval_data_root() -> Path:
    """DATA_DIR as the config module holds it now (tests redirect it there)."""
    mod = sys.modules.get(getattr(Config, "__module__", "") or "")
    value = getattr(mod, "DATA_DIR", None) if mod is not None else None
    return Path(value) if value is not None else Path(DATA_DIR)


def eval_dir() -> Path:
    """Where run reports and baselines are written as JSON."""
    override = _EVAL_RUNTIME.get("dir")
    return Path(override) if override else _eval_data_root() / "evals"


def _eval_now() -> str:
    return iso(utc_now())


def _eval_new_run_id() -> str:
    return "ev-" + utc_now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)


def _eval_slug(text: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "-"))[:80] or "-"


def _eval_json(value: Any, default: Any = None) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def eval_parse_suites(value: Any) -> list[str]:
    """A suite list from "a,b", a list, None or "all" (every suite)."""
    if value in (None, "", "all", []):
        return list(EVAL_SUITES)
    items = value.split(",") if isinstance(value, str) else list(value)
    out: list[str] = []
    for item in items:
        name = str(item).strip().lower().replace("-", "_")
        if not name:
            continue
        if name not in EVAL_SUITES:
            raise ValueError(f"unknown suite {name[:40]!r} (expected: {', '.join(EVAL_SUITES)})")
        if name not in out:
            out.append(name)
    return out or list(EVAL_SUITES)


def eval_config(config: Any = None, **extra: Any) -> Any:
    """A copy of `config` with the eval overrides (the shared one is untouched)."""
    base = config if config is not None else Config()
    changes = {k: v for k, v in {**_EVAL_CONFIG_OVERRIDES, **extra}.items() if hasattr(base, k)}
    if is_dataclass(base):
        return _eval_dc_replace(base, **changes)
    clone = type("_EvalConfig", (), {})()
    clone.__dict__.update(getattr(base, "__dict__", {}))
    for key, value in changes.items():
        setattr(clone, key, value)
    return clone


# --------------------------------------------------------------------------- #
# Schema and store                                                             #
# --------------------------------------------------------------------------- #
def _eval_schema(conn) -> None:
    """Additive, idempotent: runs, per-case results and baselines."""
    conn.execute("""CREATE TABLE IF NOT EXISTS eval_runs (
        id TEXT PRIMARY KEY, created_at TEXT NOT NULL, finished_at TEXT,
        mode TEXT NOT NULL, trigger TEXT, status TEXT NOT NULL,
        model TEXT, adapter_key TEXT, adapter_label TEXT, suites TEXT,
        scores TEXT, score REAL, passed INTEGER, total INTEGER,
        baseline_id TEXT, comparison TEXT, regressed INTEGER NOT NULL DEFAULT 0,
        regressions TEXT, error TEXT, user_id TEXT, elapsed_ms REAL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS eval_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
        suite TEXT NOT NULL, case_id TEXT NOT NULL, passed INTEGER NOT NULL,
        score REAL NOT NULL, latency_ms REAL, prompt_tokens INTEGER,
        completion_tokens INTEGER, checks TEXT, detail TEXT, error TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS eval_baselines (
        model TEXT NOT NULL, adapter_key TEXT NOT NULL, mode TEXT NOT NULL,
        run_id TEXT NOT NULL, saved_at TEXT NOT NULL, scores TEXT,
        PRIMARY KEY (model, adapter_key, mode))""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_results_run ON eval_results(run_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_runs_created ON eval_runs(created_at)")


register_schema(EVAL_SCHEMA_NAME, _eval_schema)


def _eval_ensure(db: Any) -> None:
    db.ensure_schema(EVAL_SCHEMA_NAME)


def _eval_write_json(path: Path, data: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n",
                       encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        log(f"evals: could not write {path} ({exc})", logging.WARNING)


def _eval_prune_json(folder: Path, keep: int = _EVAL_KEEP_JSON) -> None:
    try:
        files = sorted(folder.glob("ev-*.json"), key=lambda f: f.name)
    except OSError:
        return
    for stale in files[:-keep] if len(files) > keep else []:
        try:
            stale.unlink()
        except OSError:
            pass


def eval_save_run(db: Any, report: dict) -> None:
    """Persist a run: its row, its per-case results and runs/<id>.json."""
    _eval_write_json(eval_dir() / "runs" / f"{report['id']}.json", report)
    _eval_prune_json(eval_dir() / "runs")
    if db is None:
        return
    _eval_ensure(db)
    passed = sum(1 for r in report.get("results") or [] if r.get("passed"))
    db.execute(
        "INSERT OR REPLACE INTO eval_runs (id, created_at, finished_at, mode, trigger, status, "
        "model, adapter_key, adapter_label, suites, scores, score, passed, total, baseline_id, "
        "comparison, regressed, regressions, error, user_id, elapsed_ms) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (report["id"], report["started_at"], report.get("finished_at"), report["mode"],
         report.get("trigger"), report["status"], report.get("model"), report.get("adapter_key"),
         report.get("adapter_label"), json.dumps(report.get("suites") or []),
         json.dumps(report.get("scores") or {}), report.get("score"), passed,
         len(report.get("results") or []), (report.get("baseline") or {}).get("run_id"),
         json.dumps(report.get("comparison") or {}), 1 if report.get("regressed") else 0,
         json.dumps(report.get("regressions") or []), report.get("error") or "",
         report.get("user_id"), report.get("elapsed_ms")))
    db.execute("DELETE FROM eval_results WHERE run_id = ?", (report["id"],))
    for row in report.get("results") or []:
        db.execute(
            "INSERT INTO eval_results (run_id, suite, case_id, passed, score, latency_ms, "
            "prompt_tokens, completion_tokens, checks, detail, error) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (report["id"], row["suite"], row["case"], 1 if row.get("passed") else 0,
             float(row.get("score") or 0.0), row.get("latency_ms"), row.get("prompt_tokens"),
             row.get("completion_tokens"), json.dumps(row.get("checks") or {}),
             json.dumps(row.get("detail") or {}, default=str), row.get("error") or ""))
    db.commit()


def _eval_run_row(row: Any) -> dict:
    data = dict(row)
    for key, default in (("suites", []), ("scores", {}), ("comparison", {}),
                         ("regressions", [])):
        data[key] = _eval_json(data.get(key), default)
    data["regressed"] = bool(data.get("regressed"))
    return data


def eval_list_runs(db: Any, limit: int = 30) -> list[dict]:
    _eval_ensure(db)
    limit = min(500, max(1, int(limit or 30)))
    rows = db.execute("SELECT * FROM eval_runs ORDER BY created_at DESC, id DESC LIMIT ?",
                      (limit,)).fetchall()
    return [_eval_run_row(r) for r in rows]


def eval_get_run(db: Any, run_id: str) -> dict | None:
    """One run with its per-case results, from the database or its JSON file."""
    if db is not None:
        _eval_ensure(db)
        row = db.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
        if row is not None:
            data = _eval_run_row(row)
            results = []
            for r in db.execute("SELECT * FROM eval_results WHERE run_id = ? ORDER BY id",
                                (run_id,)).fetchall():
                item = dict(r)
                results.append({"suite": item["suite"], "case": item["case_id"],
                                "passed": bool(item["passed"]), "score": item["score"],
                                "latency_ms": item["latency_ms"],
                                "prompt_tokens": item["prompt_tokens"],
                                "completion_tokens": item["completion_tokens"],
                                "checks": _eval_json(item["checks"], {}),
                                "detail": _eval_json(item["detail"], {}),
                                "error": item.get("error") or ""})
            data["results"] = results
            return data
    if not re.fullmatch(r"ev-[\w-]{1,60}", str(run_id or "")):
        return None
    path = eval_dir() / "runs" / f"{run_id}.json"
    data = _eval_json(path.read_text(encoding="utf-8"), None) if path.exists() else None
    return data if isinstance(data, dict) else None


def _eval_baseline_path(model: str, adapter_key: str, mode: str) -> Path:
    return eval_dir() / "baselines" / f"{_eval_slug(model)}__{_eval_slug(adapter_key)}__{mode}.json"


def eval_save_baseline(db: Any, report: dict) -> dict:
    """Make a finished run the baseline for its model + adapter + mode."""
    if report.get("status") != "done":
        raise ValueError("only a finished run can become a baseline")
    record = {"run_id": report["id"], "model": report.get("model") or "",
              "adapter_key": report.get("adapter_key") or "", "mode": report["mode"],
              "saved_at": _eval_now(), "scores": report.get("scores") or {},
              "score": report.get("score")}
    _eval_write_json(_eval_baseline_path(record["model"], record["adapter_key"], record["mode"]),
                     record)
    if db is not None:
        _eval_ensure(db)
        db.execute("INSERT OR REPLACE INTO eval_baselines (model, adapter_key, mode, run_id, "
                   "saved_at, scores) VALUES (?,?,?,?,?,?)",
                   (record["model"], record["adapter_key"], record["mode"], record["run_id"],
                    record["saved_at"], json.dumps(record["scores"])))
        db.commit()
    return record


def eval_get_baseline(db: Any, model: str, adapter_key: str, mode: str) -> dict | None:
    if db is not None:
        _eval_ensure(db)
        row = db.execute("SELECT * FROM eval_baselines WHERE model = ? AND adapter_key = ? "
                         "AND mode = ?", (model or "", adapter_key or "", mode)).fetchone()
        if row is not None:
            data = dict(row)
            data["scores"] = _eval_json(data.get("scores"), {})
            return data
    path = _eval_baseline_path(model or "", adapter_key or "", mode)
    data = _eval_json(path.read_text(encoding="utf-8"), None) if path.exists() else None
    return data if isinstance(data, dict) else None


def eval_list_baselines(db: Any) -> list[dict]:
    _eval_ensure(db)
    out = []
    for row in db.execute("SELECT * FROM eval_baselines ORDER BY saved_at DESC").fetchall():
        data = dict(row)
        data["scores"] = _eval_json(data.get("scores"), {})
        out.append(data)
    return out


def eval_compare(scores: dict, baseline_scores: dict | None,
                 max_regression: float) -> tuple[dict, list[str]]:
    """Per-suite score vs baseline; a suite regresses past `max_regression`."""
    comparison: dict = {}
    regressions: list[str] = []
    for suite, current in (scores or {}).items():
        base = (baseline_scores or {}).get(suite)
        if not isinstance(base, dict) or base.get("score") is None:
            continue
        before = float(base["score"])
        now = current.get("score") if isinstance(current, dict) else None
        delta = None if now is None else round(float(now) - before, 4)
        regressed = now is None or (before - float(now)) > float(max_regression) + 1e-9
        comparison[suite] = {"baseline": before, "current": now, "delta": delta,
                             "regressed": regressed}
        if regressed:
            regressions.append(suite)
    return comparison, regressions


# --------------------------------------------------------------------------- #
# Model clients                                                                #
# --------------------------------------------------------------------------- #
class EvalScriptedClient:
    """A deterministic stand-in for ModelClient: each case's canned outputs.

    begin_case() loads the case's `script` (or an override keyed "suite/case"
    or "case", which is how tests degrade a case to prove regressions are
    caught). Every generation, streamed or not, takes the next output; past
    the end it answers `fallback`. An output is text, or {"text", "finish"}
    where finish "length" is a reply cut off at its token budget. Streams are
    split into small chunks so the agent's streaming tool-call detector runs
    exactly as it does against a real server.
    """

    def __init__(self, overrides: dict | None = None, fallback: str = "Done.") -> None:
        self.overrides = dict(overrides or {})
        self.fallback = fallback
        self.outputs: list = []
        self.prompts: list[list[dict]] = []
        self.replies: list = []
        self.generations = 0

    def begin_case(self, suite: str, case: dict) -> None:
        script = self.overrides.get(f"{suite}/{case['id']}",
                                    self.overrides.get(case["id"], case.get("script") or []))
        self.outputs = list(script)
        self.prompts = []
        self.replies = []

    def _next(self) -> tuple[str, str]:
        self.generations += 1
        item = self.outputs.pop(0) if self.outputs else self.fallback
        if isinstance(item, dict):
            text, finish = str(item.get("text") or ""), str(item.get("finish") or "stop")
        else:
            text, finish = str(item), "stop"
        self.replies.append(text)
        return text, finish

    @staticmethod
    def _fill(stats: Any, messages: list[dict], text: str, finish: str) -> Any:
        if stats is None:
            stats = GenerationStats()
        stats.prompt_tokens = sum(estimate_tokens(m.get("content")) for m in messages or [])
        stats.completion_tokens = estimate_tokens(text)
        stats.finish_reason = finish
        stats.total_ms = 1.0
        return stats

    def stream(self, messages, max_tokens=None, temperature=None, stats=None, kind="chat",
               conversation_id=None, extra=None):
        self.prompts.append(list(messages or []))
        text, finish = self._next()
        # Filled before the first chunk: the agent stops reading at a complete
        # tool call, so code after the last yield would never run.
        self._fill(stats, messages, text, finish)
        pieces = [text[i:i + 24] for i in range(0, len(text), 24)] or [""]

        async def gen():
            for piece in pieces:
                await asyncio.sleep(0)
                yield piece
        return gen()

    async def complete_with_stats(self, messages, max_tokens=None, temperature=None,
                                  kind="chat", conversation_id=None, extra=None):
        self.prompts.append(list(messages or []))
        text, finish = self._next()
        return text, self._fill(None, messages, text, finish)

    async def complete(self, messages, max_tokens=None, temperature=None, extra=None):
        text, _ = await self.complete_with_stats(messages, max_tokens, temperature)
        return text

    def reply_budget(self, messages, max_tokens=None, quiet=False):
        return int(max_tokens or 512)

    def request_timeout(self, messages, max_tokens=None):
        return 30.0

    async def wait_until_ready(self, timeout: float = 0.0) -> bool:
        return True

    @staticmethod
    def classify_error(exc: Exception) -> str:
        return "the scripted model failed"


class EvalLiveClient(ModelClient):
    """The real local model, cluster-less, with a connection pool of its own.

    ModelClient's pool is a class attribute bound to whichever event loop used
    it last; rebinding it from the eval thread would hand the server's loop a
    client from another loop mid-request. This subclass keeps its own pool for
    its own loop, skips bind_tool_call_config (the parser keeps reading the
    server's live Config, not this copy) and records every prompt it sends so
    the RAG suite can tell retrieval from luck.
    """

    def __init__(self, config: Any) -> None:  # noqa: D107 - see class docstring
        self.config = config
        self.cluster = None
        self.prompts: list[list[dict]] = []
        # Raw replies (lists of streamed pieces), so an invented tool name the
        # agent silently corrected still counts against the model.
        self.replies: list = []
        self._eval_http: Any = None
        self._eval_loop: Any = None

    def begin_case(self, suite: str, case: dict) -> None:
        self.prompts = []
        self.replies = []

    def _client_for(self, node: Any):
        import httpx
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        client = self._eval_http
        if client is None or client.is_closed or self._eval_loop is not loop:
            self._eval_http = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=15.0,
                                                                      pool=15.0))
            self._eval_loop = loop
        return self._eval_http

    def stream(self, messages, *args, **kwargs):
        self.prompts.append(list(messages or []))
        inner = super().stream(messages, *args, **kwargs)
        parts: list[str] = []
        self.replies.append(parts)

        async def gen():
            try:
                async for piece in inner:
                    parts.append(piece)
                    yield piece
            finally:
                await inner.aclose()
        return gen()

    async def complete_with_stats(self, messages, *args, **kwargs):
        self.prompts.append(list(messages or []))
        text, stats = await super().complete_with_stats(messages, *args, **kwargs)
        self.replies.append([text])
        return text, stats

    async def aclose_eval(self) -> None:
        client, self._eval_http = self._eval_http, None
        if client is not None and not client.is_closed:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001
                pass


def _eval_live_client(config: Any) -> Any:
    factory = _EVAL_RUNTIME.get("client_factory")
    if callable(factory):
        return factory(config)
    return EvalLiveClient(eval_config(config))


# --------------------------------------------------------------------------- #
# Sandbox                                                                      #
# --------------------------------------------------------------------------- #
class _EvalAgent(Agent):
    """The real agent loop; the sandbox, not the approval policy, contains it."""

    async def _ag_decide(self, name: str, args: dict, turn: Any) -> tuple[str, str, str, Any]:
        return "allow", "", "evaluation sandbox", None


class _EvalSandbox:
    """Temp workspace + database + a ToolRegistry whose side effects are stubbed."""

    def __init__(self, config: Any, mode: str, *, files: dict | None = None,
                 docs: dict | None = None, user: str = _EVAL_USER,
                 workspace_name: str = "workspace", overrides: dict | None = None) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="llm-eval-"))
        # The folder name is visible to the agent (the project basename is a
        # codebase term to the intent rules), so a suite may choose it.
        self.workspace = self.root / workspace_name
        self.workspace.mkdir(parents=True)
        for rel, text in (files or {}).items():
            path = self.workspace / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        self.db = Database(self.root / "eval.db")
        extra = dict(_EVAL_SCRIPTED_OVERRIDES) if mode == "scripted" else {}
        extra.update(overrides or {})
        self.config = eval_config(config, project_dir=str(self.workspace),
                                  skills_dir=str(self.root / "skills"), **extra)
        self.indexed = 0
        if docs and getattr(self.db, "fts_enabled", False):
            for rel, text in docs.items():
                self.indexed += 1 if self.db.index_document(rel, text, Path(rel).name,
                                                            user_id=user) else 0
        self.registry = ToolRegistry(self.config, self.db)
        # The undo store too. The registry roots it at DATA_DIR/checkpoints and
        # every agent turn prunes it to its newest entries, so an eval run was
        # trimming the live install's checkpoints (and adding its own).
        store = self.registry.checkpoints
        self.registry.checkpoints = type(store)(self.root / "checkpoints", keep=store.keep,
                                                max_file_bytes=store.max_file_bytes)
        # A subagent would multiply generations and escape the per-case script.
        self.registry._tools.pop("delegate_task", None)
        self.calls: list[dict] = []
        self.canned: dict = {}
        self.registry.call = self.call

    def begin_case(self, case: dict) -> None:
        self.calls = []
        self.canned = dict(case.get("canned") or {})

    def call(self, name: str, args: dict, conversation_id: str | None = None):
        args = dict(args) if isinstance(args, dict) else {}
        tool = self.registry.get(name)
        entry = {"name": name, "args": args, "known": tool is not None, "stubbed": False}
        self.calls.append(entry)
        if tool is None:
            error = f"Unknown tool: {name}. Available: {', '.join(self.registry.names())}"
            entry["error"] = error
            return error, error
        clean = self.registry.normalise_args(tool, args)
        missing = [k for k in tool.required if k not in clean or clean[k] in (None, "")]
        if missing:
            expected = ", ".join(f"{k} ({v})" for k, v in tool.parameters.items())
            error = (f"Missing required argument(s) for {name}: {', '.join(missing)}. "
                     f"Expected arguments: {expected}")
            entry["error"] = error
            return error, error
        entry["risk"] = classify_tool_risk(name, clean, tool)
        if name in self.canned:
            entry["stubbed"] = True
            return str(self.canned[name]), None
        if name not in _EVAL_REAL_TOOLS:
            entry["stubbed"] = True
            return f"OK. (Evaluation sandbox: {name} was recorded, not run.)", None
        try:
            result, error = str(tool.handler(**clean)), None
        except Exception as exc:  # noqa: BLE001 - a tool error is the model's to handle
            result = error = f"{type(exc).__name__}: {exc}"
            entry["error"] = error
        return result[:int(getattr(self.config, "tool_raw_chars", 20000) or 20000)], error

    def agent(self, client: Any) -> Agent:
        agent = _EvalAgent(self.config, self.registry, client)
        agent._partials_dir = self.root / "partials"
        return agent

    def close(self) -> None:
        try:
            self.db.close()
        except Exception:  # noqa: BLE001
            pass
        shutil.rmtree(self.root, ignore_errors=True)


async def _eval_turn(agent: Agent, message: str, history: list[dict], conversation_id: str,
                     max_tokens: int, timeout: float,
                     attachments: list[dict] | None = None) -> tuple[list[dict], str]:
    events: list[dict] = []

    async def drain() -> None:
        async for event in agent.run_iterating(message, history, conversation_id,
                                               max_tokens, 0.0, attachments=attachments):
            events.append(event)
    try:
        await asyncio.wait_for(drain(), timeout)
        return events, ""
    except asyncio.TimeoutError:
        return events, f"timed out after {timeout:.0f} s"
    except Exception as exc:  # noqa: BLE001 - a crashed turn is a failed case
        return events, f"{type(exc).__name__}: {exc}"


def _eval_final(events: list[dict]) -> dict:
    finals = [e for e in events if e.get("type") == "final"]
    return finals[-1] if finals else {}


def _eval_tokens(final: dict) -> tuple[int, int]:
    return int(final.get("prompt_tokens") or 0), int(final.get("completion_tokens") or 0)


def _eval_case_row(suite: str, case_id: str, checks: dict, *, latency_ms: float,
                   prompt_tokens: int = 0, completion_tokens: int = 0,
                   detail: dict | None = None, error: str = "") -> dict:
    values = [bool(v) for v in checks.values()]
    score = round(sum(values) / len(values), 4) if values else 0.0
    return {"suite": suite, "case": case_id, "passed": bool(values) and all(values) and not error,
            "score": 0.0 if error and not values else score, "latency_ms": round(latency_ms, 1),
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "checks": {k: bool(v) for k, v in checks.items()}, "detail": detail or {},
            "error": error}


# --------------------------------------------------------------------------- #
# Suite: tool calling                                                          #
# --------------------------------------------------------------------------- #
def _eval_predicate(value: Any, spec: Any) -> bool:
    if not isinstance(spec, dict):
        return value == spec
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    low = str(text or "").lower()
    for key, want in spec.items():
        if key == "present" and bool(want) != (value not in (None, "", [], {})):
            return False
        if key == "contains" and str(want).lower() not in low:
            return False
        if key == "contains_all" and not all(str(w).lower() in low for w in want):
            return False
        if key == "regex" and not re.search(str(want), str(text or ""), re.I):
            return False
        if key == "equals" and value != want:
            return False
        if key == "one_of" and value not in want:
            return False
        if key == "number":
            try:
                float(value)
            except (TypeError, ValueError):
                return False
    return True


def _eval_args_ok(registry: Any, name: str, args: dict, specs: dict) -> tuple[bool, list[str]]:
    tool = registry.get(name)
    if tool is None:
        return False, ["unknown tool"]
    clean = registry.normalise_args(tool, args or {})
    problems = [f"missing {k}" for k in tool.required if clean.get(k) in (None, "")]
    for key, spec in (specs or {}).items():
        if not _eval_predicate(clean.get(key, (args or {}).get(key)), spec):
            problems.append(f"{key} does not match {json.dumps(spec)}")
    return not problems, problems


def _eval_contains_all(answer: str, wanted: list) -> bool:
    low = (answer or "").lower()
    return all(str(w).lower() in low for w in wanted or [])


def _eval_invented_names(client: Any, known: set) -> set[str]:
    """Tool names the model wrote that the registry does not have.

    The agent corrects an unregistered name without dispatching it (no
    tool_call event, no registry call), so the raw replies are the evidence.
    """
    names: set[str] = set()
    for reply in list(getattr(client, "replies", None) or []):
        text = "".join(reply) if isinstance(reply, list) else str(reply or "")
        try:
            calls = parse_tool_calls(text, None, formats="auto", repair=True, final=True)
        except Exception:  # noqa: BLE001 - unparseable text names nothing
            calls = []
        names.update(c.name for c in calls if c.name and c.name not in known)
    return names


async def _eval_tool_suite(ctx: dict) -> list[dict]:
    sandbox = _EvalSandbox(ctx["config"], ctx["mode"], files=EVAL_WORKSPACE_FILES)
    try:
        agent = sandbox.agent(ctx["client"])
        known = set(sandbox.registry.names())
        rows = []
        for case in EVAL_TOOL_CASES:
            ctx["tick"]("tool_calling", case["id"])
            sandbox.begin_case(case)
            ctx["client"].begin_case("tool_calling", case)
            started = time.perf_counter()
            events, error = await _eval_turn(agent, case["prompt"], [],
                                             f"eval-{ctx['run_id']}-tool-{case['id']}",
                                             sandbox.config.max_tokens, ctx["timeout"])
            elapsed = (time.perf_counter() - started) * 1000
            final = _eval_final(events)
            answer = str(final.get("answer") or "")
            calls = [{"name": str(e.get("name") or ""), "args": e.get("args") or {}}
                     for e in events if e.get("type") == "tool_call"]
            names = [c["name"] for c in calls] + [c["name"] for c in sandbox.calls]
            hallucinated = sorted({n for n in names if n and n not in known}
                                  | _eval_invented_names(ctx["client"], known))
            expect = case.get("expect_tool")
            problems: list[str] = []
            checks: dict = {}
            if expect:
                checks["tool_choice"] = bool(calls) and calls[0]["name"] == expect
                chosen = next((c for c in calls if c["name"] == expect), None)
                ok, problems = (_eval_args_ok(sandbox.registry, expect, chosen["args"],
                                              case.get("args") or {})
                                if chosen else (False, ["the expected tool was never called"]))
                checks["args_valid"] = ok
            else:
                checks["tool_choice"] = not calls
            checks["no_hallucination"] = not hallucinated
            checks["completion"] = (bool(final) and bool(answer.strip()) and not error
                                    and _eval_contains_all(answer, case.get("answer_contains")))
            prompt_tokens, completion_tokens = _eval_tokens(final)
            rows.append(_eval_case_row(
                "tool_calling", case["id"], checks, latency_ms=elapsed,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, error=error,
                detail={"expected": expect, "calls": calls[:6], "hallucinated": hallucinated,
                        "problems": problems, "answer": answer[:600]}))
        return rows
    finally:
        sandbox.close()


# --------------------------------------------------------------------------- #
# Suite: RAG                                                                   #
# --------------------------------------------------------------------------- #
def _eval_prompt_text(prompts: list[list[dict]]) -> str:
    parts = []
    for messages in prompts or []:
        for message in messages or []:
            content = message.get("content")
            parts.append(content if isinstance(content, str)
                         else json.dumps(content, ensure_ascii=False, default=str))
    return "\n".join(parts)


async def _eval_rag_suite(ctx: dict) -> list[dict]:
    sandbox = _EvalSandbox(ctx["config"], ctx["mode"], docs=EVAL_RAG_DOCS)
    try:
        agent = sandbox.agent(ctx["client"])
        rows = []
        for case in EVAL_RAG_CASES:
            ctx["tick"]("rag", case["id"])
            sandbox.begin_case(case)
            ctx["client"].begin_case("rag", case)
            started = time.perf_counter()
            events, error = await _eval_turn(agent, case["prompt"], [],
                                             f"eval-{ctx['run_id']}-rag-{case['id']}",
                                             sandbox.config.max_tokens, ctx["timeout"])
            elapsed = (time.perf_counter() - started) * 1000
            final = _eval_final(events)
            answer = str(final.get("answer") or "")
            low = answer.lower()
            source = case["source"]
            seen = _eval_prompt_text(getattr(ctx["client"], "prompts", []))
            seen += "\n".join(str(e.get("result") or "") for e in events
                              if e.get("type") == "tool_result")
            checks = {
                "fact": any(str(f).lower() in low for f in case["facts"]),
                "cited": source.lower() in low or Path(source).name.lower() in low,
                "retrieved": source in seen,
            }
            if error:
                checks["completion"] = False
            prompt_tokens, completion_tokens = _eval_tokens(final)
            row = _eval_case_row("rag", case["id"], checks, latency_ms=elapsed,
                                 prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                                 error=error, detail={"source": source, "answer": answer[:600],
                                                      "indexed": sandbox.indexed})
            # Passing needs the fact and the citation; retrieval is scored but a
            # model that answers from a seeded tool result still passes.
            row["passed"] = bool(checks["fact"] and checks["cited"] and not error)
            rows.append(row)
        return rows
    finally:
        sandbox.close()


# --------------------------------------------------------------------------- #
# Suite: multi-turn task continuity                                            #
# --------------------------------------------------------------------------- #
_EVAL_DEF_PATTERNS = (
    r"\bdef\s+{name}\s*\(", r"\bfunction\s+{name}\s*\(", r"\bfunc\s+(?:\([^)]*\)\s*)?{name}\s*\(",
    r"\bfn\s+{name}\b", r"\b{name}\s*[:=]\s*(?:async\s+)?(?:function\b|\()",
    r"\b(?:sub|proc)\s+{name}\b",
)


def _eval_defined(code: str, language: str, name: str) -> bool:
    try:
        if any(found == name for found, _s, _e in definitions(code, language)):
            return True
    except Exception:  # noqa: BLE001 - fall back to the patterns below
        pass
    return any(re.search(p.format(name=re.escape(name)), code) for p in _EVAL_DEF_PATTERNS)


def _eval_blocks(answer: str) -> list[tuple[str, str]]:
    """(canonical language, code) for each fenced block of an answer."""
    out = []
    for info, code in extract_code_blocks(strip_truncation_note(answer or "")):
        lang = canonical_language(info) or code_language(code) or ""
        out.append((lang, code))
    return out


def _eval_turn_checks(spec: dict, answer: str, final: dict) -> tuple[dict, dict]:
    blocks = _eval_blocks(answer)
    want = canonical_language(spec.get("language")) or str(spec.get("language") or "")
    mine = [code for lang, code in blocks if not lang or same_language(want, lang)]
    code = "\n".join(mine) if mine else ""
    checks: dict = {}
    detail: dict = {"languages": [lang for lang, _ in blocks]}
    if "language" in spec:
        checks["language"] = bool(blocks) and any(lang and same_language(want, lang)
                                                  for lang, _ in blocks)
    if spec.get("no_drift"):
        checks["no_drift"] = all(not lang or same_language(want, lang) for lang, _ in blocks)
    if spec.get("defines"):
        missing = [n for n in spec["defines"] if not _eval_defined(code, want, n)]
        checks["defines"] = not missing
        detail["undefined"] = missing
    if spec.get("contains"):
        checks["contains"] = all(s in code for s in spec["contains"])
    if spec.get("clean"):
        findings = [f for f in check_code(code, want) if f.severity == "error"] if code else []
        checks["clean"] = bool(code) and not findings
        detail["findings"] = [f.render() for f in findings][:4]
    if "truncated" in spec:
        # The agent lane marks a cut-off answer with the truncation note (its
        # final event's flag stays False); the continuation lane sets the flag.
        cut = bool(final.get("truncated")) or was_truncated(answer)
        checks["truncated"] = cut == bool(spec["truncated"])
    if spec.get("continued"):
        checks["continued"] = bool(final.get("continued"))
    return checks, detail


async def _eval_multi_turn_suite(ctx: dict) -> list[dict]:
    sandbox = _EvalSandbox(ctx["config"], ctx["mode"])
    try:
        agent = sandbox.agent(ctx["client"])
        rows = []
        for case in EVAL_MULTI_TURN_CASES:
            ctx["tick"]("multi_turn", case["id"])
            sandbox.begin_case(case)
            ctx["client"].begin_case("multi_turn", case)
            conversation = f"eval-{ctx['run_id']}-mt-{case['id']}"
            history: list[dict] = []
            checks: dict = {}
            turns: list[dict] = []
            error = ""
            prompt_tokens = completion_tokens = 0
            started = time.perf_counter()
            for index, turn in enumerate(case["turns"], 1):
                events, error = await _eval_turn(agent, turn["user"], list(history),
                                                 conversation, sandbox.config.max_tokens,
                                                 ctx["timeout"])
                final = _eval_final(events)
                answer = str(final.get("answer") or "")
                got, detail = _eval_turn_checks(turn.get("checks") or {}, answer, final)
                for key, value in got.items():
                    checks[f"t{index}.{key}"] = value
                p, c = _eval_tokens(final)
                prompt_tokens += p
                completion_tokens += c
                turns.append({"turn": index, "checks": got, "answer": answer[:400], **detail})
                if error:
                    checks[f"t{index}.completed"] = False
                    break
                # What the chat route stores: a cut-off answer keeps its marker,
                # which is how the next "continue" knows to resume.
                if final.get("truncated") and not was_truncated(answer):
                    answer += truncation_note(int(sandbox.config.max_tokens),
                                              int(sandbox.config.max_tokens),
                                              int(sandbox.config.context_size))
                history += [{"role": "user", "content": turn["user"]},
                            {"role": "assistant", "content": answer}]
            elapsed = (time.perf_counter() - started) * 1000
            rows.append(_eval_case_row("multi_turn", case["id"], checks, latency_ms=elapsed,
                                       prompt_tokens=prompt_tokens,
                                       completion_tokens=completion_tokens, error=error,
                                       detail={"turns": turns}))
        return rows
    finally:
        sandbox.close()


# --------------------------------------------------------------------------- #
# Suite: routing (upgrade/ROUTING_AND_TOOLING.md P1 and P9)                    #
# --------------------------------------------------------------------------- #
# The routing sandbox registers what the lexicon routes to: skills on (the
# shared overrides switch them off so tool_calling stays comparable) and no
# extra or Office 365 tools, so "check my email" is a switched-off capability
# in every install and an expected lane never depends on what an admin enabled.
_EVAL_ROUTING_OVERRIDES: dict = {
    "skills_enabled": True, "extra_tools": "", "o365_tenant_id": "", "o365_client_id": "",
    "o365_client_secret": "",
}
_EVAL_ROUTING_DIMENSIONS = ("intent", "lane", "tools", "retrieval")
# The router's own call is not the turn's lane: its prompt is skipped when the
# lane is read off the prompts the agent sent (Agent.route words it this way).
_EVAL_ROUTER_HEAD = "You are a router"
# Everything a document turn's narrowed catalogue can hold. A tools-lane intent
# never offers create_document, and the full catalogue holds far more than this.
_EVAL_DOC_LANE_TOOLS = frozenset(
    {"create_document", "final_answer"} | set(Agent.DOCUMENT_LOOKUP_TOOLS)
    | {name for names in DOCUMENT_RESEARCH_TOOLS.values() for name in names})


def _eval_routing_attachments(case: dict) -> list[dict]:
    """The case's files as the chat route hands them over: headed text parts."""
    return [{"type": "text",
             "text": f"[attached file: {name}]\n{EVAL_ROUTING_ATTACHMENTS.get(name, '')}"}
            for name in case.get("attachments") or ()]


def _eval_routing_offered(intent: Any, registered: frozenset) -> list[str]:
    """The tools a classify() result hands the turn, narrowed the way the agent does."""
    if intent is None:
        return []
    if intent.lane == "tools":
        return [t for t in intent.tool_subset if t in registered or t == "final_answer"]
    if intent.lane == "document":
        names = ["create_document"]
        for source in intent.research_sources or ():
            names += [t for t in DOCUMENT_RESEARCH_TOOLS.get(source, ())
                      if t in registered and t not in names]
        return names
    return []


def _eval_routing_tools(offered: list[str], case: dict,
                        registered: frozenset) -> tuple[bool, dict]:
    """Every required tool the registry has is offered, and no forbidden one is.

    A required tool the sandbox does not register (a P3 tool not built yet)
    cannot be offered by any routing, so it is reported, not failed.
    """
    have = set(offered)
    wanted = list(case.get("offer") or ())
    missing = [t for t in wanted if t in registered and t not in have]
    forbidden = [t for t in case.get("not_offer") or () if t in have]
    return not missing and not forbidden, {
        "missing": missing, "forbidden": forbidden,
        "unregistered": [t for t in wanted if t not in registered]}


def _eval_system_text(messages: list[dict]) -> str:
    for message in messages or []:
        if message.get("role") == "system":
            content = message.get("content")
            return content if isinstance(content, str) else json.dumps(content, default=str)
    return ""


def _eval_routing_observe(prompts: list[list[dict]], events: list[dict],
                          registered: frozenset, intent: Any) -> dict:
    """What the agent actually did with the turn, read off its events and prompts.

    The intent is the "Understood as" notice. The lane and the tools come from
    the first prompt after the router's: tool specs in its system prompt make
    it a tools lane (a document lane when the catalogue is create_document and
    its lookups), none make it prose, and no prompt at all means the turn was
    answered deterministically (reply, or clarify when the answer is the
    intent's question). Retrieval is whether a knowledge-base fixture's path
    is in that first prompt: auto-RAG injects before any tool runs.
    """
    notice = next((e for e in events if e.get("type") == "notice" and e.get("intent")), None)
    lane_prompts = [p for p in prompts or []
                    if not _eval_system_text(p).startswith(_EVAL_ROUTER_HEAD)]
    first = lane_prompts[0] if lane_prompts else []
    system = _eval_system_text(first)
    offered = [name for name in sorted(registered) if f"\n- {name}: " in system]
    if not lane_prompts:
        answer = str(_eval_final(events).get("answer") or "").strip()
        question = str(getattr(intent, "clarify_question", "") or "").strip()
        observed = "clarify" if question and answer == question else "reply"
    elif offered:
        observed = ("document" if "create_document" in offered
                    and set(offered) <= _EVAL_DOC_LANE_TOOLS else "tools")
    else:
        observed = "prose"
    text = _eval_prompt_text([first]) if first else ""
    return {"intent": str(notice["intent"]) if notice else None,
            "lane": observed if notice else "pipeline", "observed_lane": observed,
            "offered": offered,
            "retrieved": sorted(path for path in EVAL_ROUTING_KB_DOCS if path in text),
            "calls": [str(e.get("name") or "") for e in events
                      if e.get("type") == "tool_call"][:8]}


async def _eval_routing_case(ctx: dict, sandbox: _EvalSandbox, agent: Agent, case: dict,
                             registered: frozenset) -> dict:
    """One case: classify's decision, then the agent's where the agent runs it."""
    names = tuple(case.get("attachments") or ())
    history = [dict(m) for m in EVAL_ROUTING_HISTORY] if case.get("history") else []
    started = time.perf_counter()
    error = ""
    intent = None
    try:
        intent = classify(case["message"], has_attachment=bool(names), attachment_names=names,
                          has_history=bool(history), project_basename=sandbox.workspace.name,
                          registered_tools=registered)
    except Exception as exc:  # noqa: BLE001 - a crashing rule is a failed case
        error = f"classify failed: {type(exc).__name__}: {exc}"[:300]
    got = {"intent": intent.name if intent else None,
           "lane": intent.lane if intent else "pipeline",
           "offered": _eval_routing_offered(intent, registered),
           "retrieval": intent.retrieval if intent else "auto",
           "source": intent.source if intent else "",
           "matched": intent.matched if intent else ""}
    tools_ok, tools_detail = _eval_routing_tools(got["offered"], case, registered)
    checks = {"intent": got["intent"] == case["intent"], "lane": got["lane"] == case["lane"],
              "tools": tools_ok, "retrieval": got["retrieval"] == case["retrieval"]}
    detail: dict = {"message": case["message"][:200],
                    "expected": {k: case.get(k) for k in
                                 ("intent", "lane", "offer", "not_offer", "retrieval")},
                    "classified": got, "tools": tools_detail}
    prompt_tokens = completion_tokens = 0
    if ctx["mode"] == "live" or case.get("agent"):
        sandbox.begin_case(case)
        ctx["client"].begin_case("routing", case)
        events, turn_error = await _eval_turn(
            agent, case["message"], history, f"eval-{ctx['run_id']}-rt-{case['id']}",
            sandbox.config.max_tokens, ctx["timeout"],
            attachments=_eval_routing_attachments(case) or None)
        final = _eval_final(events)
        seen = _eval_routing_observe(getattr(ctx["client"], "prompts", None) or [], events,
                                     registered, intent)
        checks["agent.intent"] = seen["intent"] == case["intent"]
        checks["agent.lane"] = seen["lane"] == case["lane"]
        if case["intent"] is not None:
            # A turn the lexicon leaves alone gets whatever the router picks;
            # which tools that is belongs to the tool_calling suite.
            checks["agent.tools"], seen["tools"] = _eval_routing_tools(seen["offered"], case,
                                                                       registered)
        if case["retrieval"] == "off":
            checks["agent.retrieval"] = not seen["retrieved"]
        elif case["retrieval"] == "kb":
            checks["agent.retrieval"] = bool(seen["retrieved"]) or bool(
                {"search_docs", "kb_overview"} & set(seen["offered"]))
        checks["agent.completed"] = bool(final) and not turn_error
        seen["answer"] = str(final.get("answer") or "")[:400]
        detail["agent"] = seen
        prompt_tokens, completion_tokens = _eval_tokens(final)
        error = error or turn_error
    elapsed = (time.perf_counter() - started) * 1000
    return _eval_case_row("routing", case["id"], checks, latency_ms=elapsed,
                          prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                          detail=detail, error=error)


async def _eval_routing_suite(ctx: dict) -> list[dict]:
    """Intent, lane, tool set and retrieval per message (classify, then the agent).

    Scripted mode needs no model for classify; the cases marked `agent` (the
    three reviewed failures) also run through the real Agent with their
    canned outputs, so the lane taken and the tools the system prompt lists
    are checked end to end. Live mode runs every case through the Agent.
    """
    sandbox = _EvalSandbox(ctx["config"], ctx["mode"], files=EVAL_ROUTING_WORKSPACE_FILES,
                           docs=EVAL_ROUTING_KB_DOCS, workspace_name=EVAL_ROUTING_PROJECT,
                           overrides=_EVAL_ROUTING_OVERRIDES)
    try:
        agent = sandbox.agent(ctx["client"])
        registered = frozenset(sandbox.registry.names())
        rows = []
        for case in EVAL_ROUTING_CASES:
            ctx["tick"]("routing", case["id"])
            rows.append(await _eval_routing_case(ctx, sandbox, agent, case, registered))
        return rows
    finally:
        sandbox.close()


def eval_routing_accuracy(rows: list[dict]) -> dict:
    """Share of passing checks per dimension: intent, lane, tools, retrieval.

    A dimension pools the classify check with the agent's check of the same
    name where the agent ran; one that no row measured is None.
    """
    out: dict = {}
    for dimension in _EVAL_ROUTING_DIMENSIONS:
        values = [bool(v) for row in rows for key, v in (row.get("checks") or {}).items()
                  if key.rsplit(".", 1)[-1] == dimension]
        out[dimension] = round(sum(values) / len(values), 4) if values else None
    return out


# --------------------------------------------------------------------------- #
# Suite: parsers (no model)                                                    #
# --------------------------------------------------------------------------- #
def _eval_detect(text: str, step: int, known: set) -> tuple[list, list]:
    det = ToolCallStreamDetector(known, formats="auto", repair=True)
    events: list[dict] = []
    size = step or len(text) or 1
    for i in range(0, len(text), size):
        events += det.feed(text[i:i + size])
    events += det.flush()
    calls = [[e["name"], e["args"]] for e in events if e["type"] == "call_end"]
    aborted = [e.get("raw") for e in events if e["type"] == "call_abort"]
    return calls, aborted


def _eval_parsers_suite(tick: Any) -> list[dict]:
    known = set(EVAL_PARSER_KNOWN)
    rows = []
    for case_id, text, expected, fmt in EVAL_PARSER_FORMATS:
        tick("parsers", case_id)
        started = time.perf_counter()
        calls = parse_tool_calls(text, known, formats="auto", repair=True, final=True)
        got = [[c.name, c.args] for c in calls]
        checks = {"parsed": got == expected,
                  "format": bool(calls) and calls[0].format == fmt,
                  "clean": bool(calls) and not calls[0].recovered}
        detail: dict = {"got": got[:4], "format": calls[0].format if calls else None}
        if case_id in EVAL_PARSER_STREAMED:
            bad = []
            for step in EVAL_PARSER_CHUNKS:
                streamed, aborted = _eval_detect(text, step, known)
                if streamed != expected or aborted:
                    bad.append(step or len(text))
            checks["streamed"] = not bad
            detail["stream_failures"] = bad
        rows.append(_eval_case_row("parsers", case_id, checks,
                                   latency_ms=(time.perf_counter() - started) * 1000,
                                   detail=detail))
    for case_id, text, tag, args in EVAL_PARSER_REPAIRS:
        tick("parsers", f"repair_{case_id}")
        started = time.perf_counter()
        calls = parse_tool_calls(text, known, formats="auto", repair=True, final=True)
        checks = {"parsed": bool(calls) and calls[0].args == args,
                  "recorded": bool(calls) and calls[0].recovered and tag in calls[0].repairs,
                  "refused_without_repair": not parse_tool_calls(text, known, formats="auto",
                                                                 repair=False, final=True)
                  if tag not in ("unclosed_brackets", "name_noise") else True}
        rows.append(_eval_case_row("parsers", f"repair_{case_id}", checks,
                                   latency_ms=(time.perf_counter() - started) * 1000,
                                   detail={"repairs": calls[0].repairs if calls else []}))
    for case_id, text in EVAL_PARSER_PROSE:
        tick("parsers", f"prose_{case_id}")
        started = time.perf_counter()
        calls = parse_tool_calls(text, known, formats="auto", repair=True, final=True)
        streamed, _aborted = _eval_detect(text, 5, known)
        checks = {"no_call": not calls, "no_streamed_call": not streamed}
        rows.append(_eval_case_row("parsers", f"prose_{case_id}", checks,
                                   latency_ms=(time.perf_counter() - started) * 1000,
                                   detail={"got": [[c.name, c.args] for c in calls][:3]}))
    return rows


# --------------------------------------------------------------------------- #
# Suite: /v1 API shapes (no model)                                             #
# --------------------------------------------------------------------------- #
_EVAL_API_TEXT = ("Hello from the ", "evaluation model.")
_EVAL_API_WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}


class _EvalApiModel:
    """Scripted raw server events in place of ModelClient._stream_core."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def core(self, messages, max_tokens=None, temperature=None, stats=None,
                   kind="chat", conversation_id=None, extra=None):
        extra = dict(extra or {})
        self.calls.append({"messages": messages, "extra": extra})
        if extra.get("known_tools") or extra.get("tools"):
            pieces = ['{"tool": "get_weather", ', '"args": {"city": "Paris"}}']
        else:
            pieces = list(_EVAL_API_TEXT)
        for piece in pieces:
            yield {"type": "content", "text": piece}
        yield {"type": "finish", "reason": "stop"}
        yield {"type": "usage", "usage": {"prompt_tokens": 40, "completion_tokens": 7}}
        if stats is not None:
            stats.prompt_tokens, stats.completion_tokens = 40, 7


def _eval_sse_data(text: str) -> list:
    out = []
    for line in (text or "").splitlines():
        if line.startswith("data: "):
            data = line[6:]
            out.append(data if data == "[DONE]" else _eval_json(data, {}))
    return out


def _eval_sse_events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in (text or "").strip().split("\n\n"):
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = _eval_json(line[6:], None)
        if data is not None:
            out.append((name, data))
    return out


def _eval_api_checks(http: Any, auth: dict, model: str) -> dict:
    """case id -> (checks, detail) for every /v1 shape check."""
    raw = f"raw/{model}"
    text = "".join(_EVAL_API_TEXT)
    hi = [{"role": "user", "content": "hi"}]
    out: dict = {}

    def run(case_id: str, fn) -> None:
        try:
            out[case_id] = fn()
        except Exception as exc:  # noqa: BLE001 - a broken shape is a failed check
            out[case_id] = ({"shape": False}, {"error": f"{type(exc).__name__}: {exc}"})

    def models_list():
        r = http.get("/v1/models", headers=auth)
        body = r.json()
        ids = [m.get("id") for m in body.get("data") or []]
        return ({"status": r.status_code == 200, "object": body.get("object") == "list",
                 "items": bool(ids) and all(m.get("object") == "model" for m in body["data"]),
                 "raw_model": raw in ids}, {"ids": ids[:8]})

    def auth_required():
        r = http.post("/v1/chat/completions", json={"model": raw, "messages": hi})
        err = (r.json() or {}).get("error") or {}
        return ({"status": r.status_code == 401,
                 "type": err.get("type") == "authentication_error",
                 "message": bool(err.get("message"))}, {"status": r.status_code})

    def invalid_request():
        r = http.post("/v1/chat/completions", json={"model": raw, "messages": []}, headers=auth)
        err = (r.json() or {}).get("error") or {}
        return ({"status": r.status_code == 400,
                 "type": err.get("type") == "invalid_request_error",
                 "param": err.get("param") == "messages"}, {"error": err})

    def chat_completion():
        r = http.post("/v1/chat/completions", json={"model": raw, "messages": hi}, headers=auth)
        body = r.json()
        choice = (body.get("choices") or [{}])[0]
        usage = body.get("usage") or {}
        return ({"status": r.status_code == 200,
                 "object": body.get("object") == "chat.completion" and bool(body.get("id")),
                 "message": (choice.get("message") or {}).get("role") == "assistant"
                 and (choice.get("message") or {}).get("content") == text,
                 "finish": choice.get("finish_reason") == "stop",
                 "usage": all(isinstance(usage.get(k), int) for k in
                              ("prompt_tokens", "completion_tokens", "total_tokens"))
                 and usage.get("total_tokens") == usage.get("prompt_tokens", 0)
                 + usage.get("completion_tokens", 0)}, {"status": r.status_code})

    def chat_stream():
        r = http.post("/v1/chat/completions", headers=auth,
                      json={"model": raw, "messages": hi, "stream": True})
        frames = _eval_sse_data(r.text)
        chunks = [f for f in frames if isinstance(f, dict) and f.get("choices")]
        content = "".join((f["choices"][0].get("delta") or {}).get("content") or ""
                          for f in chunks)
        return ({"status": r.status_code == 200, "done": bool(frames) and frames[-1] == "[DONE]",
                 "object": bool(chunks) and all(f.get("object") == "chat.completion.chunk"
                                                for f in chunks),
                 "content": content == text,
                 "finish": bool(chunks) and chunks[-1]["choices"][0].get("finish_reason") == "stop"},
                {"frames": len(frames)})

    def tool_call():
        r = http.post("/v1/chat/completions", headers=auth, json={
            "model": raw, "tools": [_EVAL_API_WEATHER],
            "messages": [{"role": "user", "content": "weather in Paris?"}]})
        choice = (r.json().get("choices") or [{}])[0]
        calls = (choice.get("message") or {}).get("tool_calls") or [{}]
        fn = calls[0].get("function") or {}
        return ({"status": r.status_code == 200,
                 "finish": choice.get("finish_reason") == "tool_calls",
                 "call": calls[0].get("type") == "function" and fn.get("name") == "get_weather"
                 and str(calls[0].get("id") or "").startswith("call_"),
                 "arguments": _eval_json(fn.get("arguments"), None) == {"city": "Paris"}},
                {"tool_calls": calls[:2]})

    def anthropic_message():
        r = http.post("/v1/messages", headers=auth,
                      json={"model": raw, "max_tokens": 64, "messages": hi})
        body = r.json()
        blocks = body.get("content") or [{}]
        usage = body.get("usage") or {}
        return ({"status": r.status_code == 200,
                 "type": body.get("type") == "message" and body.get("role") == "assistant",
                 "content": blocks[0].get("type") == "text" and blocks[0].get("text") == text,
                 "stop_reason": body.get("stop_reason") == "end_turn",
                 "usage": isinstance(usage.get("input_tokens"), int)
                 and isinstance(usage.get("output_tokens"), int)}, {"status": r.status_code})

    def anthropic_stream():
        r = http.post("/v1/messages", headers=auth,
                      json={"model": raw, "max_tokens": 64, "messages": hi, "stream": True})
        events = _eval_sse_events(r.text)
        names = [n for n, _ in events]
        deltas = "".join((d.get("delta") or {}).get("text") or "" for n, d in events
                         if n == "content_block_delta")
        return ({"status": r.status_code == 200,
                 "framing": bool(names) and names[0] == "message_start"
                 and names[-1] == "message_stop",
                 "blocks": names.count("content_block_start") == names.count("content_block_stop")
                 and names.count("content_block_start") >= 1,
                 "content": deltas == text}, {"events": names[:12]})

    def anthropic_error():
        r = http.post("/v1/messages", json={"model": raw, "max_tokens": 5, "messages": hi})
        body = r.json() or {}
        err = body.get("error") or {}
        return ({"status": r.status_code == 401, "type": body.get("type") == "error",
                 "error": err.get("type") == "authentication_error" and bool(err.get("message"))},
                {"status": r.status_code})

    for case_id, fn in (("models_list", models_list), ("auth_required", auth_required),
                        ("invalid_request", invalid_request), ("chat_completion", chat_completion),
                        ("chat_stream", chat_stream), ("tool_call", tool_call),
                        ("anthropic_message", anthropic_message),
                        ("anthropic_stream", anthropic_stream),
                        ("anthropic_error", anthropic_error)):
        run(case_id, fn)
    return out


def _eval_api_inprocess(config: Any, tick: Any) -> list[dict]:
    """Build a throwaway app with a scripted model and check every /v1 shape.

    Never inside the server process: create_app rebinds process-wide state
    (the approval policy, cloud fallback, dashboards runtime).
    """
    from fastapi.testclient import TestClient
    create_app = _eval_lookup("api", "create_app")
    manager_cls = _eval_lookup("model_server", "ModelServerManager")
    retrain_cls = _eval_lookup("training", "RetrainManager")
    if not (create_app and manager_cls and retrain_cls):
        raise RuntimeError("the app modules are not available in this build")
    root = Path(tempfile.mkdtemp(prefix="llm-eval-api-"))
    model = str(getattr(config, "model", "") or "mlx-community/eval-model")
    cfg = Config(auth_enabled=True, admin_username="eval-admin",
                 admin_password=secrets.token_urlsafe(18), allow_test_user=False,
                 node_token=secrets.token_urlsafe(18), model=model,
                 skills_dir=str(root / "skills"), agent_enabled=True, knowledge_triage=False,
                 mcp_servers="", cloud_fallback_enabled=False, v1_raw_enabled=True,
                 workspaces_enabled=False)
    db = Database(root / "api.db")
    try:
        (root / "adapters").mkdir()
        # Never started: the scripted _stream_core below answers every request.
        manager = manager_cls(cfg.model, 8090, root / "adapters")
        app = create_app(cfg, db, manager, retrain_cls(db, manager, cfg), ToolRegistry(cfg, db))
        manager.status = "ready"
        app.state.ctx.model_client._stream_core = _EvalApiModel().core
        user = db.create_user("eval-api", role="user")
        token, _row = create_api_token(db, user, "evaluation", ["chat"], config=cfg)
        # No `with`: the lifespan would start the scheduler and, on exit, close
        # ModelClient's process-wide connection pool.
        http = TestClient(app)
        for case_id in EVAL_API_CASES:
            tick("api", case_id)
        started = time.perf_counter()
        checks = _eval_api_checks(http, {"Authorization": f"Bearer {token}"}, cfg.model)
        each = (time.perf_counter() - started) * 1000 / max(1, len(checks))
        rows = []
        for case_id in EVAL_API_CASES:
            got, detail = checks.get(case_id) or ({"shape": False}, {"error": "not run"})
            rows.append(_eval_case_row("api", case_id, got, latency_ms=each, detail=detail))
        return rows
    finally:
        try:
            db.close()
        except Exception:  # noqa: BLE001
            pass
        shutil.rmtree(root, ignore_errors=True)


def _eval_entry_argv() -> list[str]:
    """How to run `eval` as a child process, in the package and in the bundle."""
    if __package__:
        root = str(Path(__file__).resolve().parent.parent)
        boot = (f"import sys; sys.path.insert(0, {root!r}); "
                f"from {__package__}.evals import eval_cli; "
                "raise SystemExit(eval_cli(sys.argv[1:]))")
        return [sys.executable, "-c", boot]
    return [sys.executable, str(Path(__file__).resolve()), "eval"]


def _eval_api_subprocess(config: Any, tick: Any) -> list[dict]:
    """The api suite in a child process, so the server's globals stay its own."""
    for case_id in EVAL_API_CASES:
        tick("api", case_id)
    argv = _eval_entry_argv() + ["--suite", "api", "--json", "--no-persist"]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=300)  # noqa: S603
    for line in reversed((proc.stdout or "").splitlines()):
        data = _eval_json(line.strip(), None) if line.strip().startswith("{") else None
        if isinstance(data, dict) and isinstance(data.get("results"), list):
            return [r for r in data["results"] if r.get("suite") == "api"]
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
    raise RuntimeError(f"api suite child exited {proc.returncode}: {' | '.join(tail)[:300]}")


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #
class _EvalCancelled(Exception):
    pass


_EVAL_MODEL_RUNNERS = {
    "tool_calling": _eval_tool_suite,
    "rag": _eval_rag_suite,
    "multi_turn": _eval_multi_turn_suite,
    "routing": _eval_routing_suite,
}
_EVAL_CASE_COUNTS = {
    "tool_calling": len(EVAL_TOOL_CASES), "rag": len(EVAL_RAG_CASES),
    "multi_turn": len(EVAL_MULTI_TURN_CASES), "routing": len(EVAL_ROUTING_CASES),
    "parsers": len(EVAL_PARSER_FORMATS) + len(EVAL_PARSER_REPAIRS) + len(EVAL_PARSER_PROSE),
    "api": len(EVAL_API_CASES),
}


def eval_aggregate(rows: list[dict], error: str = "") -> dict:
    """Per-suite score (mean case score), pass counts, latency and tokens."""
    total = len(rows)
    passed = sum(1 for r in rows if r.get("passed"))
    latencies = [float(r["latency_ms"]) for r in rows if r.get("latency_ms") is not None]
    return {"score": round(sum(float(r.get("score") or 0.0) for r in rows) / total, 4)
            if total else (0.0 if error else None),
            "passed": passed, "failed": total - passed, "total": total,
            "latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in rows),
            "completion_tokens": sum(int(r.get("completion_tokens") or 0) for r in rows),
            "error": error}


def run_evals(suites: Any = None, mode: str = "scripted", *, config: Any = None, db: Any = None,
              client: Any = None, model_id: str | None = None, adapter_key: str | None = None,
              adapter_label: str | None = None, trigger: str = "cli",
              user_id: str | None = None, persist: bool = True, save_baseline: bool = False,
              isolate_app: bool = False, overrides: dict | None = None,
              baseline: Any = _EVAL_UNSET, max_regression: float | None = None,
              run_id: str | None = None, progress: Any = None, cancel: Any = None) -> dict:
    """Run suites and return the report (persisted unless persist=False).

    `client` is the model for the model suites: omitted, scripted mode builds
    an EvalScriptedClient (with `overrides`) and live mode an EvalLiveClient.
    `baseline` defaults to the stored one for (model, adapter_key, mode); pass
    a record (or None) to compare against something else. `isolate_app` runs
    the api suite in a child process (required inside the server).
    """
    suites = eval_parse_suites(suites)
    mode = str(mode or "scripted").strip().lower()
    if mode not in EVAL_MODES:
        raise ValueError(f"mode must be one of: {', '.join(EVAL_MODES)}")
    cfg = config if config is not None else Config()
    needs_model = any(s in EVAL_MODEL_SUITES for s in suites)
    if mode == "scripted":
        model_id = model_id or EVAL_SCRIPTED_MODEL
        adapter_key = adapter_key or EVAL_SCRIPTED_ADAPTER
        adapter_label = adapter_label or "scripted model"
        client = client or EvalScriptedClient(overrides)
    else:
        model_id = model_id or str(getattr(cfg, "model", "") or "")
        if not adapter_key:
            adapter_key = adapter_key_for_choice(getattr(cfg, "adapter", None), model_id)
        adapter_label = adapter_label or adapter_key
        if needs_model and client is None:
            client = _eval_live_client(cfg)
    if max_regression is None:
        max_regression = float(getattr(cfg, "eval_max_regression", 0.05) or 0.0)
    report: dict = {
        "id": run_id or _eval_new_run_id(), "mode": mode, "trigger": trigger,
        "model": model_id, "adapter_key": adapter_key, "adapter_label": adapter_label,
        "suites": suites, "started_at": _eval_now(), "finished_at": None, "status": "running",
        "scores": {}, "score": None, "results": [], "baseline": None, "comparison": {},
        "regressions": [], "regressed": False, "max_regression": max_regression,
        "error": "", "user_id": user_id, "elapsed_ms": None,
    }
    total = sum(_EVAL_CASE_COUNTS[s] for s in suites)
    state = {"done": 0}

    def tick(suite: str, case_id: str) -> None:
        if cancel is not None and cancel():
            raise _EvalCancelled()
        if progress is not None:
            try:
                progress(state["done"], total, f"{suite}/{case_id}")
            except Exception:  # noqa: BLE001 - progress is cosmetic
                pass
        state["done"] += 1

    started = time.perf_counter()
    log_event(get_logger("eval"), logging.INFO, "eval.run.start", run_id=report["id"],
              mode=mode, suites=",".join(suites), trigger=trigger)
    run_ctx = {"config": cfg, "mode": mode, "client": client, "run_id": report["id"],
               "timeout": _EVAL_CASE_TIMEOUT[mode], "tick": tick}

    async def run_all() -> None:
        set_acting_user(_EVAL_USER)
        try:
            for suite in suites:
                rows: list[dict] = []
                error = ""
                try:
                    if suite in _EVAL_MODEL_RUNNERS:
                        rows = await _EVAL_MODEL_RUNNERS[suite](run_ctx)
                    elif suite == "parsers":
                        rows = _eval_parsers_suite(tick)
                    else:
                        runner = (_EVAL_RUNTIME.get("api_runner") or _eval_api_subprocess
                                  if isolate_app else _eval_api_inprocess)
                        rows = await asyncio.to_thread(runner, cfg, tick)
                except _EvalCancelled:
                    raise
                except Exception as exc:  # noqa: BLE001 - one broken suite is a result
                    error = f"{type(exc).__name__}: {exc}"[:500]
                    log(f"evals: suite {suite} failed ({error})", logging.WARNING)
                report["results"].extend(rows)
                report["scores"][suite] = eval_aggregate(rows, error)
                if suite == "routing" and rows:
                    report["scores"][suite]["accuracy"] = eval_routing_accuracy(rows)
        finally:
            closer = getattr(client, "aclose_eval", None)
            if callable(closer):
                await closer()

    try:
        asyncio.run(run_all())
        report["status"] = "done"
    except _EvalCancelled:
        report["status"] = "cancelled"
        report["error"] = "cancelled"
    except Exception as exc:  # noqa: BLE001
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"[:500]
        log(f"evals: run {report['id']} failed ({report['error']})", logging.WARNING)
    scored = [s["score"] for s in report["scores"].values() if s.get("score") is not None]
    report["score"] = round(sum(scored) / len(scored), 4) if scored else None
    report["finished_at"] = _eval_now()
    report["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    if baseline is _EVAL_UNSET:
        try:
            baseline = eval_get_baseline(db, model_id, adapter_key, mode)
        except Exception as exc:  # noqa: BLE001
            log(f"evals: baseline lookup failed ({exc})", logging.WARNING)
            baseline = None
    if baseline and report["status"] == "done":
        comparison, regressions = eval_compare(report["scores"], baseline.get("scores"),
                                               max_regression)
        report.update(baseline={k: baseline.get(k) for k in
                                ("run_id", "model", "adapter_key", "mode", "saved_at", "scores")},
                      comparison=comparison, regressions=regressions,
                      regressed=bool(regressions))
    if persist:
        try:
            eval_save_run(db, report)
        except Exception as exc:  # noqa: BLE001 - the report is still returned
            log(f"evals: could not store run {report['id']} ({exc})", logging.WARNING)
    if save_baseline and report["status"] == "done":
        report["saved_baseline"] = eval_save_baseline(db if persist else None, report)
    log_event(get_logger("eval"), logging.INFO, "eval.run.done", run_id=report["id"],
              status=report["status"], score=report["score"],
              regressed=report["regressed"], elapsed_ms=report["elapsed_ms"])
    return report


# --------------------------------------------------------------------------- #
# Busy checks and the background job                                          #
# --------------------------------------------------------------------------- #
def _eval_busy(ctx: Any, *, training_ok: bool = False) -> str:
    """Why a live run must not start now, or "" when it may."""
    if ctx is None:
        return ""
    if not training_ok and (getattr(ctx.retrain_manager, "status", None) or {}).get("running"):
        return "A training run is in progress and owns the model server."
    extras = getattr(ctx, "extras", None) or {}
    for key, what in (("bench_job", "A benchmark sweep"), ("eval_job", "An adapter tournament")):
        job = extras.get(key)
        if job is not None and getattr(job, "running", False):
            return f"{what} owns the model server."
    try:
        queue = ctx.generation_gate.snapshot() or {}
    except Exception:  # noqa: BLE001 - a plain gate has no snapshot
        queue = {}
    running = int(queue.get("running") or 0)
    waiting = int(queue.get("waiting") or 0)
    in_flight = int(getattr(ctx.tasks, "chat_in_flight", 0) or 0)
    if running or waiting or in_flight:
        return (f"{max(running, in_flight)} generation(s) running and {waiting} waiting. "
                "Let them finish first: a live evaluation measures the model alone.")
    if list(getattr(ctx.tasks, "by_task", {}) or {}):
        return "Task runs are in flight. Let them finish first."
    return ""


class _EvalJob:
    """The one evaluation a process runs at a time (API or promotion gate)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.cancel = False
        self.state: dict = {"running": False, "run_id": None, "mode": None, "suites": [],
                            "trigger": None, "started_at": None, "finished_at": None,
                            "done": 0, "total": 0, "current": "", "status": "idle",
                            "score": None, "regressed": False, "error": ""}

    def claim(self, **fields: Any) -> bool:
        with self.lock:
            if self.running:
                return False
            self.running = True
            self.cancel = False
            self.state.update(running=True, started_at=_eval_now(), finished_at=None, done=0,
                              total=0, current="", status="running", score=None,
                              regressed=False, error="", **fields)
            return True

    def progress(self, done: int, total: int, current: str) -> None:
        with self.lock:
            self.state.update(done=done, total=total, current=current)

    def release(self, **fields: Any) -> None:
        with self.lock:
            self.running = False
            self.state.update(running=False, finished_at=_eval_now(), current="", **fields)

    def snapshot(self) -> dict:
        with self.lock:
            return {**self.state, "running": self.running}


_EVAL_JOB = _EvalJob()


def _eval_active(ctx: Any) -> dict:
    """The model and adapter the local server answers with now."""
    mm = ctx.model_manager
    model = getattr(mm, "model_id", None) or getattr(ctx.config, "model", "") or ""
    info = active_adapter(mm)
    key = info.get("key") or base_adapter_key(model)
    try:
        label = _dash_label(key, adapter_labels(model))
    except Exception:  # noqa: BLE001
        label = key
    return {"model": model, "adapter_key": key, "adapter_label": label,
            "adapter_choice": info.get("choice"), "status": getattr(mm, "status", None)}


def _eval_job_body(job: _EvalJob, ctx: Any, opts: dict) -> None:
    try:
        report = run_evals(opts["suites"], opts["mode"], config=ctx.config, db=ctx.db,
                           model_id=opts.get("model"), adapter_key=opts.get("adapter_key"),
                           adapter_label=opts.get("adapter_label"), trigger="api",
                           user_id=opts.get("user_id"), save_baseline=opts["save_baseline"],
                           isolate_app=True, run_id=opts["run_id"], progress=job.progress,
                           cancel=lambda: job.cancel)
        job.release(status=report["status"], score=report.get("score"),
                    regressed=bool(report.get("regressed")), error=report.get("error") or "")
    except Exception as exc:  # noqa: BLE001
        log(f"evals: background run failed ({exc})", logging.ERROR)
        job.release(status="error", error=f"{type(exc).__name__}: {exc}"[:500])


# --------------------------------------------------------------------------- #
# Post-promotion gate (EVAL_ON_PROMOTE)                                        #
# --------------------------------------------------------------------------- #
def _eval_adapter_dir() -> Path:
    found = _eval_lookup("training", "ADAPTER_DIR")
    return Path(found) if found is not None else Path(ADAPTER_DIR)


def _eval_wait_idle(job: _EvalJob, deadline: float) -> str:
    """Claim the eval job once the server is idle; "" on success, else why not."""
    ctx = _EVAL_RUNTIME.get("ctx")
    while True:
        busy = _eval_busy(ctx, training_ok=True)
        if not busy and job.claim(mode="live", suites=list(EVAL_PROMOTE_SUITES),
                                  trigger="promote", run_id=None):
            return ""
        if time.monotonic() >= deadline:
            return busy or "another evaluation is running"
        time.sleep(1.0)


def _eval_reason(report: dict) -> str:
    parts = []
    for suite in report.get("regressions") or []:
        row = (report.get("comparison") or {}).get(suite) or {}
        now, before = row.get("current"), row.get("baseline")
        if now is None:
            parts.append(f"{suite} failed to run (baseline {before:.2f})")
        else:
            parts.append(f"{suite} {before:.2f} -> {now:.2f} ({now - before:+.2f})")
    return (", ".join(parts) + f"; more than the {report.get('max_regression', 0):.2f} "
            "EVAL_MAX_REGRESSION allows") if parts else ""


def _eval_promote_gate(rm: Any, backup_path: Any, feedback_ids: Any) -> dict:
    cfg = rm.config
    mm = rm.model_manager
    adapter_dir = _eval_adapter_dir()
    model = str(getattr(cfg, "model", "") or "")
    if getattr(mm, "status", "") != "ready":
        return {"verdict": "skipped", "message": (
            f"Evaluation skipped: the model server is {getattr(mm, 'status', 'not ready')}.")}
    new_key = adapter_fingerprint(adapter_dir) or base_adapter_key(model)
    backup = Path(backup_path) if backup_path else None
    prev_key = (adapter_fingerprint(backup) if backup is not None and backup.exists() else None) \
        or base_adapter_key(model)
    baseline = (eval_get_baseline(rm.db, model, prev_key, "live")
                or eval_get_baseline(rm.db, model, base_adapter_key(model), "live"))
    report = run_evals(list(EVAL_PROMOTE_SUITES), "live", config=cfg, db=rm.db, model_id=model,
                       adapter_key=new_key, adapter_label="latest", trigger="promote",
                       baseline=baseline)
    summary = {"run_id": report["id"], "at": report["finished_at"],
               "scores": {k: v.get("score") for k, v in report["scores"].items()},
               "baseline": ({"run_id": baseline.get("run_id"),
                             "adapter_key": baseline.get("adapter_key"),
                             "scores": {k: (v or {}).get("score") for k, v in
                                        (baseline.get("scores") or {}).items()}}
                            if baseline else None),
               "regressions": report.get("regressions") or [],
               "max_regression": report.get("max_regression")}
    if report["status"] != "done":
        summary.update(verdict="error", reason=report.get("error") or report["status"])
        message = f"Evaluation could not finish ({summary['reason']}); the adapter was kept."
    elif not report.get("regressed"):
        verdict = "passed" if baseline else "unverified"
        summary.update(verdict=verdict, reason="" if baseline else
                       "no baseline for the previous adapter; this run is now the baseline")
        try:
            eval_save_baseline(rm.db, report)
        except Exception as exc:  # noqa: BLE001
            log(f"evals: could not save the promoted adapter's baseline ({exc})", logging.WARNING)
        message = (f"Evaluation passed (score {report.get('score') or 0:.2f})." if baseline else
                   f"Evaluation recorded as the new baseline (score {report.get('score') or 0:.2f}).")
    else:
        summary.update(verdict="rolled_back", reason=_eval_reason(report))
        meta = adapter_meta(adapter_dir)
        meta["eval"] = summary
        _eval_write_json(eval_dir() / "rejected" / f"{report['id']}.json", meta)
        rm._rollback_adapter(backup_path)
        if feedback_ids:
            # The rows were stamped trained when the adapter was promoted; a
            # rolled-back run must leave them for the next, larger run.
            ids = [int(i) for i in feedback_ids]
            marks = ",".join("?" for _ in ids)
            rm.db.execute(f"UPDATE feedback SET trained_at = NULL WHERE id IN ({marks})",
                          tuple(ids))
            rm.db.commit()
        if adapter_dir.exists() and any(adapter_dir.iterdir()):
            restored = adapter_meta(adapter_dir)
            restored["eval_rollback"] = summary
            write_adapter_meta(adapter_dir, restored)
        try:
            mm.restart()
        except Exception as exc:  # noqa: BLE001
            log(f"evals: restart after rollback failed ({exc})", logging.ERROR)
        log_event(get_logger("eval"), logging.WARNING, "eval.promote.rollback",
                  run_id=report["id"], regressions=",".join(summary["regressions"]))
        return {"verdict": "rolled_back", "message": (
            f"Evaluation found a regression ({summary['reason']}); the previous adapter was "
            "restored."), "summary": summary}
    # Only into a directory that holds an adapter: a file in an empty one would
    # make adapter_ready() report an adapter that is not there.
    if adapter_dir.exists() and any(adapter_dir.iterdir()):
        meta = adapter_meta(adapter_dir)
        meta["eval"] = summary
        write_adapter_meta(adapter_dir, meta)
    return {"verdict": summary["verdict"], "message": message, "summary": summary}


def eval_after_promote(rm: Any, backup_path: Any = None, feedback_ids: Any = None) -> dict:
    """EVAL_ON_PROMOTE: evaluate the adapter a retrain just promoted.

    Called by RetrainManager (still holding its lock) after the model server
    restarted with the new adapter. Never raises: an evaluation that cannot
    run keeps the adapter, as promotion did before this gate existed.
    """
    prior = str((getattr(rm, "status", None) or {}).get("message") or "")
    job = _EVAL_JOB
    claimed = False
    outcome: dict = {"verdict": "error", "message": "Evaluation failed; the adapter was kept."}
    try:
        rm.status = {"running": True, "message": "Evaluating the new adapter before keeping it."}
        why = _eval_wait_idle(job, time.monotonic() + _EVAL_PROMOTE_WAIT_S)
        if why:
            outcome = {"verdict": "skipped", "message": f"Evaluation skipped: {why}"}
        else:
            claimed = True
            outcome = _eval_promote_gate(rm, backup_path, feedback_ids)
    except Exception as exc:  # noqa: BLE001 - the gate must never fail a promotion
        log(f"evals: promotion gate failed ({exc})", logging.ERROR)
        outcome = {"verdict": "error", "message": f"Evaluation failed ({exc}); the adapter was kept."}
    finally:
        if claimed:
            job.release(status=outcome.get("verdict", "error"))
    rm.status = {"running": False, "message": (prior + " " + outcome["message"]).strip()}
    return outcome


# --------------------------------------------------------------------------- #
# Routes                                                                       #
# --------------------------------------------------------------------------- #
def register_evals(app, ctx) -> None:
    """Mount /api/evals/* (ADMIN)."""
    from fastapi.responses import JSONResponse

    _EVAL_RUNTIME["ctx"] = ctx
    ctx.extras["evals_job"] = _EVAL_JOB
    db = ctx.db
    ADMIN = ctx.ADMIN
    try:
        db.ensure_schema(EVAL_SCHEMA_NAME)
    except Exception as exc:  # noqa: BLE001
        log(f"evals: schema not applied yet ({exc})", logging.WARNING)

    @app.post("/api/evals/run")
    async def evals_run(body: dict, user: dict = ADMIN):
        body = body if isinstance(body, dict) else {}
        try:
            suites = eval_parse_suites(body.get("suites"))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        mode = str(body.get("mode") or "scripted").strip().lower()
        if mode not in EVAL_MODES:
            return JSONResponse({"error": "mode must be scripted or live"}, status_code=400)
        if _EVAL_JOB.running:
            return JSONResponse({"error": "An evaluation is already running."}, status_code=409)
        active = await asyncio.to_thread(_eval_active, ctx)
        opts = {"suites": suites, "mode": mode, "run_id": _eval_new_run_id(),
                "save_baseline": bool(body.get("save_baseline")),
                "user_id": (user or {}).get("id")}
        if mode == "live":
            if any(s in EVAL_MODEL_SUITES for s in suites) and active["status"] != "ready":
                return JSONResponse({"error": f"The model is not ready (status: "
                                              f"{active['status']}). Live evaluations need the "
                                              "local model loaded."}, status_code=503)
            busy = _eval_busy(ctx)
            if busy:
                return JSONResponse({"error": busy}, status_code=409)
            opts.update(model=active["model"], adapter_key=active["adapter_key"],
                        adapter_label=active["adapter_label"])
        if not _EVAL_JOB.claim(run_id=opts["run_id"], mode=mode, suites=suites, trigger="api"):
            return JSONResponse({"error": "An evaluation is already running."}, status_code=409)
        threading.Thread(target=_eval_job_body, args=(_EVAL_JOB, ctx, opts), daemon=True,
                         name="eval-run").start()
        return JSONResponse({"started": True, "run_id": opts["run_id"],
                             "status": _EVAL_JOB.snapshot()}, status_code=202)

    @app.post("/api/evals/cancel")
    async def evals_cancel(_a: dict = ADMIN):
        with _EVAL_JOB.lock:
            if not _EVAL_JOB.running:
                return JSONResponse({"error": "No evaluation is running."}, status_code=409)
            _EVAL_JOB.cancel = True
        return {"cancelling": True}

    @app.get("/api/evals/runs")
    async def evals_runs(limit: int = 30, _a: dict = ADMIN):
        runs = await asyncio.to_thread(eval_list_runs, db, limit)
        active = await asyncio.to_thread(_eval_active, ctx)
        return {"runs": runs, "job": _EVAL_JOB.snapshot(), "active": active,
                "suites": list(EVAL_SUITES), "model_suites": list(EVAL_MODEL_SUITES),
                "max_regression": float(getattr(ctx.config, "eval_max_regression", 0.05)),
                "on_promote": bool(getattr(ctx.config, "eval_on_promote", False))}

    @app.get("/api/evals/runs/{run_id}")
    async def evals_run_detail(run_id: str, _a: dict = ADMIN):
        data = await asyncio.to_thread(eval_get_run, db, run_id)
        if data is None:
            job = _EVAL_JOB.snapshot()
            if job.get("run_id") == run_id and job.get("running"):
                return {"id": run_id, "status": "running", "job": job, "results": []}
            return JSONResponse({"error": "no such evaluation run"}, status_code=404)
        return data

    @app.get("/api/evals/baseline")
    async def evals_baseline(mode: str = "live", model: str | None = None,
                             adapter: str | None = None, _a: dict = ADMIN):
        mode = (mode or "live").strip().lower()
        if mode not in EVAL_MODES:
            return JSONResponse({"error": "mode must be scripted or live"}, status_code=400)
        active = await asyncio.to_thread(_eval_active, ctx)
        if mode == "scripted":
            model, adapter = model or EVAL_SCRIPTED_MODEL, adapter or EVAL_SCRIPTED_ADAPTER
        baseline = await asyncio.to_thread(eval_get_baseline, db, model or active["model"],
                                           adapter or active["adapter_key"], mode)
        baselines = await asyncio.to_thread(eval_list_baselines, db)
        return {"baseline": baseline, "baselines": baselines, "active": active, "mode": mode}

    @app.post("/api/evals/baseline")
    async def evals_baseline_save(body: dict, _a: dict = ADMIN):
        run_id = str((body or {}).get("run_id") or "") if isinstance(body, dict) else ""
        report = await asyncio.to_thread(eval_get_run, db, run_id) if run_id else None
        if report is None:
            return JSONResponse({"error": "no such evaluation run"}, status_code=404)
        try:
            record = await asyncio.to_thread(eval_save_baseline, db, report)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return {"baseline": record}


# --------------------------------------------------------------------------- #
# CLI: deploy.py eval                                                          #
# --------------------------------------------------------------------------- #
def _eval_pct(value: Any) -> str:
    return "-" if value is None else f"{float(value) * 100:5.1f}%"


def eval_format_report(report: dict) -> str:
    lines = [f"Evaluation {report['id']} ({report['mode']}): model {report.get('model')}, "
             f"adapter {report.get('adapter_label') or report.get('adapter_key')}"]
    lines.append(f"  {'suite':<14}{'score':>8}{'passed':>9}{'latency':>10}{'baseline':>10}"
                 f"{'change':>9}")
    for suite in report.get("suites") or []:
        row = (report.get("scores") or {}).get(suite) or {}
        cmp_row = (report.get("comparison") or {}).get(suite) or {}
        latency = row.get("latency_ms")
        delta = cmp_row.get("delta")
        flag = "  REGRESSED" if cmp_row.get("regressed") else ""
        lines.append(f"  {suite:<14}{_eval_pct(row.get('score')):>8}"
                     f"{str(row.get('passed', 0)) + '/' + str(row.get('total', 0)):>9}"
                     f"{('-' if latency is None else f'{latency:.0f} ms'):>10}"
                     f"{_eval_pct(cmp_row.get('baseline')):>10}"
                     f"{('-' if delta is None else f'{delta * 100:+.1f}'):>9}{flag}")
        if row.get("error"):
            lines.append(f"    error: {row['error']}")
        accuracy = row.get("accuracy") or {}
        if accuracy:
            lines.append("    accuracy: " + ", ".join(
                f"{dim} {_eval_pct(value).strip()}" for dim, value in accuracy.items()))
    lines.append(f"Overall {_eval_pct(report.get('score')).strip()} ({report['status']}, "
                 f"{(report.get('elapsed_ms') or 0) / 1000:.1f} s).")
    failed = [r for r in report.get("results") or [] if not r.get("passed")]
    if failed:
        lines.append("Failed cases:")
        for r in failed[:25]:
            bad = ", ".join(k for k, v in (r.get("checks") or {}).items() if not v)
            lines.append(f"  {r['suite']}/{r['case']}: {bad or r.get('error') or 'failed'}")
    if report.get("baseline"):
        lines.append(f"Baseline: run {report['baseline'].get('run_id')} "
                     f"({report['baseline'].get('saved_at')}).")
        if report.get("regressed"):
            lines.append("REGRESSION: " + _eval_reason(report))
    else:
        lines.append("No baseline for this model and adapter yet (use --save-baseline).")
    if report.get("saved_baseline"):
        lines.append("Saved as the baseline.")
    return "\n".join(lines)


def eval_cli(argv: list[str] | None = None, *, db: Any = None, config: Any = None,
             client: Any = None, out: Any = None) -> int:
    """deploy.py eval: run the suites from the command line.

    Exit codes: 0 done, 1 regression (with --compare), 2 refused or no
    baseline to compare against, 3 the run itself failed.
    """
    parser = argparse.ArgumentParser(
        prog="deploy.py eval",
        description="Run the built-in evaluation and regression suites.")
    parser.add_argument("--suite", default="all",
                        help=f"Comma list of suites: {', '.join(EVAL_SUITES)} (default all).")
    parser.add_argument("--live", action="store_true",
                        help="Use the real local model instead of the scripted one.")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Store this run as the baseline for its model and adapter.")
    parser.add_argument("--compare", action="store_true",
                        help="Exit 1 on a regression against the baseline (2 when none exists).")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    parser.add_argument("--no-persist", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(list(argv or []))
    write = out or print
    try:
        suites = eval_parse_suites(args.suite)
    except ValueError as exc:
        write(f"eval: {exc}")
        return 2
    cfg = config if config is not None else Config()
    mode = "live" if args.live else "scripted"
    model_id = adapter_key = None
    if mode == "live" and any(s in EVAL_MODEL_SUITES for s in suites):
        client = client or _eval_live_client(cfg)
        ready = asyncio.run(client.wait_until_ready(timeout=5.0))
        if not ready:
            write(f"eval: refused, the model server at 127.0.0.1:{getattr(cfg, 'model_port', '?')} "
                  "is not ready. Start the server first, or run without --live.")
            return 2
        write("eval: note, the CLI cannot see the web server's queue; run it while nothing "
              "else is generating (the Admin > Evals tab checks that for you).")
        model_id = str(getattr(cfg, "model", "") or "")
        adapter_key = adapter_key_for_choice(getattr(cfg, "adapter", None), model_id)
    persist = not args.no_persist
    if persist and db is None:
        # A fresh checkout has no data/ until the server first starts.
        data_root = _eval_data_root()
        data_root.mkdir(parents=True, exist_ok=True)
        db = Database(data_root / "feedback.db")
    report = run_evals(suites, mode, config=cfg, db=db if persist else None, client=client,
                       model_id=model_id, adapter_key=adapter_key, trigger="cli",
                       persist=persist, save_baseline=args.save_baseline)
    write(json.dumps(report, ensure_ascii=False, default=str) if args.json
          else eval_format_report(report))
    if report["status"] != "done":
        return 3
    if args.compare:
        if not report.get("baseline"):
            return 2
        return 1 if report.get("regressed") else 0
    return 0


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    "EVAL_SCHEMA_NAME",
    "EVAL_SUITES",
    "EVAL_MODEL_SUITES",
    "EVAL_PROMOTE_SUITES",
    "EVAL_MODES",
    "EVAL_SCRIPTED_MODEL",
    "EVAL_SCRIPTED_ADAPTER",
    "EVAL_API_CASES",
    "_EVAL_USER",
    "_EVAL_UNSET",
    "_EVAL_CASE_TIMEOUT",
    "_EVAL_PROMOTE_WAIT_S",
    "_EVAL_KEEP_JSON",
    "_EVAL_REAL_TOOLS",
    "_EVAL_CONFIG_OVERRIDES",
    "_EVAL_SCRIPTED_OVERRIDES",
    "_EVAL_RUNTIME",
    "_eval_lookup",
    "_eval_data_root",
    "eval_dir",
    "_eval_now",
    "_eval_new_run_id",
    "_eval_slug",
    "_eval_json",
    "eval_parse_suites",
    "eval_config",
    "_eval_schema",
    "_eval_ensure",
    "_eval_write_json",
    "_eval_prune_json",
    "eval_save_run",
    "_eval_run_row",
    "eval_list_runs",
    "eval_get_run",
    "_eval_baseline_path",
    "eval_save_baseline",
    "eval_get_baseline",
    "eval_list_baselines",
    "eval_compare",
    "EvalScriptedClient",
    "EvalLiveClient",
    "_eval_live_client",
    "_EvalAgent",
    "_EvalSandbox",
    "_eval_turn",
    "_eval_final",
    "_eval_tokens",
    "_eval_case_row",
    "_eval_predicate",
    "_eval_args_ok",
    "_eval_contains_all",
    "_eval_invented_names",
    "_eval_tool_suite",
    "_eval_prompt_text",
    "_eval_rag_suite",
    "_EVAL_DEF_PATTERNS",
    "_eval_defined",
    "_eval_blocks",
    "_eval_turn_checks",
    "_eval_multi_turn_suite",
    "_EVAL_ROUTING_OVERRIDES",
    "_EVAL_ROUTING_DIMENSIONS",
    "_EVAL_ROUTER_HEAD",
    "_EVAL_DOC_LANE_TOOLS",
    "_eval_routing_attachments",
    "_eval_routing_offered",
    "_eval_routing_tools",
    "_eval_system_text",
    "_eval_routing_observe",
    "_eval_routing_case",
    "_eval_routing_suite",
    "eval_routing_accuracy",
    "_eval_detect",
    "_eval_parsers_suite",
    "_EVAL_API_TEXT",
    "_EVAL_API_WEATHER",
    "_EvalApiModel",
    "_eval_sse_data",
    "_eval_sse_events",
    "_eval_api_checks",
    "_eval_api_inprocess",
    "_eval_entry_argv",
    "_eval_api_subprocess",
    "_EvalCancelled",
    "_EVAL_MODEL_RUNNERS",
    "_EVAL_CASE_COUNTS",
    "eval_aggregate",
    "run_evals",
    "_eval_busy",
    "_EvalJob",
    "_EVAL_JOB",
    "_eval_active",
    "_eval_job_body",
    "_eval_adapter_dir",
    "_eval_wait_idle",
    "_eval_reason",
    "_eval_promote_gate",
    "eval_after_promote",
    "register_evals",
    "_eval_pct",
    "eval_format_report",
    "eval_cli",
]
