"""Human-in-the-loop tool approval: risk classes, layered policy, broker, audit.

Until this module the only question asked about a tool call was "is this tool
in the agent's allowlist?". Once a capability was on, every call it unlocked ran
without a person seeing it: a model that had read a web page saying "now run
`curl ... | sh`" could do exactly that inside the same turn, and nothing
recorded who allowed it. This module adds the missing questions:

  * how risky is this call?  Every tool has a risk class: read, write, exec or
    external (sends something out of the machine). MCP tools take theirs from
    the server's readOnlyHint/destructiveHint annotations, otherwise external.
    A few tools depend on their arguments (o365_mail "send", an HTTP verb).
  * what does policy say?  decide() folds the layers, each able to override the
    one before it:
        install default (TOOL_APPROVAL_MODE off|risky|all)
          -> global override rows -> agent profile -> role   (administrators)
          -> user override -> conversation override           (the user)
          -> taint escalation -> one-shot approvals -> unattended mapping
    A deny set by an administrator layer is final: a user's "always allow"
    cannot loosen it. The user layers may tighten anything and may turn an
    administrator's "ask" into "allow" (that is what "always allow this tool
    here" means).
  * who approved it?  The ApprovalBroker (BROKER) holds a pending approval
    until the turn's owner answers, the timeout passes or the turn is
    cancelled. Every decision that is not today's silent allow lands in the
    append-only tool_decisions table, which is never deleted with a
    conversation (deleting the evidence with the transcript would make the
    audit trail worthless exactly when it matters).

Taint (prompt-injection defence): once a turn has read untrusted outside
content (fetch_url, web_search, index_url, Office 365 mail/files/calendar or any
MCP tool), a later write, exec or external call in the SAME turn needs a human
again even where an override said "allow", because the text that asked for it
may have come from the page rather than the user. Only a one-shot approval of
that exact call satisfies it. TOOL_APPROVAL_TAINT=0 switches this off.

Unattended work (a scheduled task, a direct /api/tools/call, any code with no
turn or turn.interactive False) cannot show a prompt, so "ask" becomes
TOOL_APPROVAL_UNATTENDED (deny by default).

Enforcement is two-sided on purpose. The agent loop calls decide() and asks the
human through the broker; ToolRegistry.call then calls enforce_tool_policy()
synchronously and fails closed on deny and on "ask" without a one-shot grant.
That second check is what covers callers that cannot show UI and were written
before approvals existed: execute_code's RPC bridge, /api/tools/call and any
old dispatch site. It reads the turn from the contextvar (turnctx), so the
signature of ToolRegistry.call is unchanged and tests that replace it with a
lambda keep working.

Defaults keep today's behaviour exactly: TOOL_APPROVAL_MODE=off with no
override rows allows every call without touching the database and writes no
audit rows. Explicit override rows (and agent-profile or role rules) still apply
with the mode off, so an administrator can deny one tool without turning the
whole approval flow on.

Routes (register_policy, from create_app's extension anchor):
  GET    /api/approvals/pending           the caller's pending approvals
  GET    /api/tool-policies               overrides visible to the caller + mode
  POST   /api/tool-policies               {scope, scope_id?, tool, decision, expires_in_s?}
  DELETE /api/tool-policies/{policy_id}
  GET    /api/audit/tool-decisions        admins: all rows; users: their own
The chat route's POST /api/chat/turns/{turn_id}/approvals/{approval_id} (in the
chat-api module) answers a prompt by calling BROKER.resolve().
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .database import *  # noqa: F401,F403
from .turnctx import *  # noqa: F401,F403


POLICY_SCHEMA_NAME = "policy"

RISK_CLASSES = ("read", "write", "exec", "external")
POLICY_DECISIONS = ("allow", "ask", "deny")
POLICY_SCOPES = ("conversation", "user", "profile", "global")
APPROVAL_SCOPES = ("once", "conversation", "tool")
APPROVAL_MODES = ("off", "risky", "all")

# Risks that taint escalates: the ones that change or send something.
_POLICY_ESCALATED = ("write", "exec", "external")

# The built-in tools by risk. A tool missing from here and without its own
# Tool.risk is classed "external": an unknown capability is assumed to reach
# outside the machine until someone says otherwise.
_POLICY_RISK_BY_TOOL: dict[str, str] = {
    **{name: "read" for name in (
        "read_file", "list_files", "search_files", "file_info", "web_search",
        "fetch_url", "calculator", "current_time", "weather", "recall_feedback",
        "recall_memory", "search_docs", "search_skills", "load_skill",
        "search_past_conversations", "final_answer", "delegate_task",
        "o365_files", "o365_calendar", "git_diff")},
    **{name: "write" for name in (
        "write_file", "edit_file", "create_document", "remember", "forget",
        "save_skill", "index_docs", "index_url")},
    **{name: "exec" for name in (
        "run_shell", "run_python", "run_tests", "execute_code")},
    # o365_mail is read unless it sends; see classify_tool_risk.
    "o365_mail": "read",
}

# Tools whose RESULT is untrusted outside content (see "Taint" above).
_POLICY_TAINT_TOOLS = {"fetch_url", "web_search", "index_url"}
_POLICY_TAINT_PREFIXES = ("mcp_", "o365_")

_POLICY_RULE_KEY = re.compile(
    r"^(\*|risk:(read|write|exec|external|send)|cap:[a-z0-9_]{1,64}|[A-Za-z0-9_.\-]{1,128})$")

# Module binding: register_policy binds the live config and database so decide()
# works for callers that pass neither (the agent loop's pre-check). Without a
# binding decide() allows and ToolRegistry.call, which always passes its own
# config, remains the enforcement point.
_POLICY_BOUND: dict[str, Any] = {"config": None, "db": None}
# Per database file: whether any override row may exist. Lets mode "off" skip
# the query entirely (today's cost) until somebody writes a rule.
_POLICY_ROWS: dict[str, bool] = {}
_POLICY_ROWS_LOCK = threading.Lock()

_policy_log = get_logger("tool")
_policy_change_log = get_logger("policy")


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def _policy_schema(conn) -> None:
    """Idempotent: override rows, the append-only audit table, agents.tool_policy."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tool_policy_overrides (
            id INTEGER PRIMARY KEY,
            scope TEXT NOT NULL,
            scope_id TEXT NOT NULL DEFAULT '',
            tool TEXT NOT NULL,
            decision TEXT NOT NULL,
            created_by TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TEXT,
            UNIQUE(scope, scope_id, tool)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_policy_scope "
                 "ON tool_policy_overrides(scope, scope_id)")
    # No foreign key to messages/conversations and no user_id cascade: rows
    # must outlive the conversation they describe.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tool_decisions (
            id INTEGER PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_id TEXT,
            conversation_id TEXT,
            turn_id TEXT,
            tool TEXT NOT NULL,
            args_redacted TEXT,
            risk TEXT,
            decision TEXT NOT NULL,
            rule TEXT,
            approver_id TEXT,
            latency_ms INTEGER
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_decisions_user "
                 "ON tool_decisions(user_id, id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_decisions_conversation "
                 "ON tool_decisions(conversation_id, id)")
    # Append-only at the database level, not just by convention: a bug (or a
    # well-meaning cleanup) that DELETEs or UPDATEs audit rows fails loudly.
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS tool_decisions_no_update
        BEFORE UPDATE ON tool_decisions
        BEGIN SELECT RAISE(ABORT, 'tool_decisions is append-only'); END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS tool_decisions_no_delete
        BEFORE DELETE ON tool_decisions
        BEGIN SELECT RAISE(ABORT, 'tool_decisions is append-only'); END
    """)
    # Optional per-profile policy, a JSON object {rule key: decision}, edited
    # with set_profile_tool_policy. Constant default so old rows read as "none".
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(agents)")}
    if cols and "tool_policy" not in cols:
        conn.execute("ALTER TABLE agents ADD COLUMN tool_policy TEXT NOT NULL DEFAULT ''")


register_schema(POLICY_SCHEMA_NAME, _policy_schema)


# --------------------------------------------------------------------------- #
# Values
# --------------------------------------------------------------------------- #
@dataclass
class Decision:
    """What policy says about one call. `rule` names the layer that decided."""

    action: str                      # allow | ask | deny
    risk: str
    reason: str
    rule: str
    approval_id: str | None = None   # set when a one-shot approval decided
    approver_id: str | None = None
    latency_ms: int | None = None

    def to_dict(self) -> dict:
        return {"action": self.action, "risk": self.risk, "reason": self.reason,
                "rule": self.rule}


@dataclass
class Resolution:
    """How a pending approval ended. `args` is what to run (possibly edited)."""

    decision: str                    # allow | deny | timeout
    scope: str = "once"
    args: dict | None = None
    by: str | None = None            # approver id, or "timeout" / "cancel"
    latency_ms: int = 0
    reason: str = ""
    approval_id: str | None = None
    call_id: str | None = None


def _policy_norm_decision(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {"approve": "allow", "approved": "allow", "yes": "allow", "always": "allow",
               "allowed": "allow", "reject": "deny", "rejected": "deny", "no": "deny",
               "block": "deny", "never": "deny", "denied": "deny", "prompt": "ask",
               "confirm": "ask"}
    return aliases.get(text, text)


def _policy_mode(config: Any) -> str:
    mode = str(getattr(config, "tool_approval_mode", "off") or "off").strip().lower()
    return mode if mode in APPROVAL_MODES else "risky"


def _policy_rules_from(obj: Any) -> dict[str, str]:
    """A {rule key: decision} map from a dict or JSON text; bad entries dropped."""
    if isinstance(obj, str):
        text = obj.strip()
        if not text:
            return {}
        try:
            obj = json.loads(text)
        except (TypeError, ValueError):
            return {}
    if not isinstance(obj, dict):
        return {}
    rules: dict[str, str] = {}
    for key, value in obj.items():
        key = str(key).strip()
        if key.startswith("risk:send"):
            key = "risk:external"
        decision = _policy_norm_decision(value)
        if _POLICY_RULE_KEY.match(key) and decision in POLICY_DECISIONS:
            rules[key] = decision
    return rules


_POLICY_ROLE_CACHE: dict[str, dict[str, dict[str, str]]] = {}


def _policy_role_rules(config: Any, role: str | None) -> dict[str, str]:
    raw = str(getattr(config, "tool_approval_roles", "") or "").strip()
    if not raw or not role:
        return {}
    parsed = _POLICY_ROLE_CACHE.get(raw)
    if parsed is None:
        try:
            obj = json.loads(raw)
        except (TypeError, ValueError):
            obj = {}
        parsed = ({str(r): _policy_rules_from(v) for r, v in obj.items()}
                  if isinstance(obj, dict) else {})
        if len(_POLICY_ROLE_CACHE) > 32:
            _POLICY_ROLE_CACHE.clear()
        _POLICY_ROLE_CACHE[raw] = parsed
    return parsed.get(str(role), {})


def profile_tool_policy(profile: dict | None) -> dict[str, str]:
    """The rules stored on an agent profile row (agents.tool_policy)."""
    if not isinstance(profile, dict):
        return {}
    return _policy_rules_from(profile.get("tool_policy"))


def _policy_fingerprint(args: Any) -> str:
    try:
        text = json.dumps(args or {}, sort_keys=True, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        text = repr(args)
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:32]


def _policy_redact_args(args: Any, limit: int = 2000) -> str:
    """Args for the audit row: secrets dropped, long values (file bodies) cut."""
    def cut(value: Any, depth: int = 0) -> Any:
        if isinstance(value, str):
            return value if len(value) <= 300 else value[:300] + f"... [{len(value)} chars]"
        if isinstance(value, dict) and depth < 4:
            return {str(k): cut(v, depth + 1) for k, v in list(value.items())[:40]}
        if isinstance(value, (list, tuple)) and depth < 4:
            return [cut(v, depth + 1) for v in list(value)[:40]]
        return value
    try:
        text = json.dumps(cut(redact_obj(args if isinstance(args, dict) else {})),
                          ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = "{}"
    return text[:limit]


def _policy_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _policy_expired(expires_at: Any) -> bool:
    if not expires_at:
        return False
    try:
        moment = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
    except ValueError:
        return False
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment <= datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Risk classes
# --------------------------------------------------------------------------- #
def risk_from_mcp_annotations(annotations: dict | None) -> str:
    """MCP tool annotations -> risk class. Missing hints mean "external".

    readOnlyHint wins (the server says it changes nothing); an explicit
    destructiveHint false on a writing tool is "write" (it changes local state
    additively); anything else may destroy or send, so it is external.
    """
    if not isinstance(annotations, dict):
        return "external"
    if annotations.get("readOnlyHint") is True:
        return "read"
    if annotations.get("destructiveHint") is False:
        return "write"
    return "external"


def classify_tool_risk(name: str, args: dict | None = None, tool: Any = None) -> str:
    """The risk class of one call: read, write, exec or external."""
    name = str(name or "")
    args = args if isinstance(args, dict) else {}
    declared = str(getattr(tool, "risk", None) or "").strip().lower()
    if declared == "send":
        declared = "external"
    if name == "o365_mail":
        # Read unless it sends. An unrecognised action is treated as a send:
        # guessing "read" for a verb we do not know is the wrong way to fail.
        action = str(args.get("action") or "read").strip().lower()
        return "read" if action in ("read", "search", "list", "get") else "external"
    if "http" in name.lower() and "method" in args:
        method = str(args.get("method") or "GET").strip().upper()
        if method not in ("GET", "HEAD", "OPTIONS"):
            return "external"
        return declared if declared in RISK_CLASSES else "read"
    if declared in RISK_CLASSES:
        return declared
    if name in _POLICY_RISK_BY_TOOL:
        return _POLICY_RISK_BY_TOOL[name]
    annotations = getattr(tool, "annotations", None)
    if annotations is not None or name.startswith("mcp_"):
        return risk_from_mcp_annotations(annotations)
    return "external"


def tool_taints(name: str, tool: Any = None) -> bool:
    """Whether this tool's result is untrusted outside content."""
    flag = getattr(tool, "taints", None)
    if flag is not None:
        return bool(flag)
    return name in _POLICY_TAINT_TOOLS or name.startswith(_POLICY_TAINT_PREFIXES)


def _policy_capability(name: str, tool: Any) -> str | None:
    declared = getattr(tool, "capability", None)
    if declared:
        return str(declared)
    for key, group in CAPABILITY_GROUPS.items():
        if name in (group.get("tools") or []):
            return key
    return None


def _policy_match(rules: dict[str, str], name: str, risk: str,
                  capability: str | None) -> tuple[str, str] | None:
    """Most specific rule first: tool name, capability, risk class, "*"."""
    if not rules:
        return None
    for key in (name, f"cap:{capability}" if capability else None, f"risk:{risk}", "*"):
        if key and key in rules:
            return rules[key], key
    return None


# --------------------------------------------------------------------------- #
# Override rows and the audit table
# --------------------------------------------------------------------------- #
def _policy_db_key(db: Any) -> str:
    return str(getattr(db, "db_path", id(db)))


def _policy_has_rows(db: Any) -> bool:
    if db is None:
        return False
    key = _policy_db_key(db)
    with _POLICY_ROWS_LOCK:
        known = _POLICY_ROWS.get(key)
    if known is not None:
        return known
    try:
        db.ensure_schema(POLICY_SCHEMA_NAME)
        found = db.execute("SELECT 1 FROM tool_policy_overrides LIMIT 1").fetchone() is not None
    except Exception as exc:  # noqa: BLE001 - a broken table must not stop tools
        log(f"tool policy: could not read overrides: {redact_text(str(exc))}", logging.WARNING)
        return False
    with _POLICY_ROWS_LOCK:
        _POLICY_ROWS[key] = found
    return found


def _policy_mark_rows(db: Any, value: bool | None) -> None:
    with _POLICY_ROWS_LOCK:
        if value is None:
            _POLICY_ROWS.pop(_policy_db_key(db), None)
        else:
            _POLICY_ROWS[_policy_db_key(db)] = value


def _policy_row(row: Any) -> dict:
    out = dict(row)
    out["expired"] = _policy_expired(out.get("expires_at"))
    return out


def list_tool_overrides(db: Any, scope: str | None = None,
                        scope_id: str | None = None) -> list[dict]:
    db.ensure_schema(POLICY_SCHEMA_NAME)
    where, params = [], []
    if scope:
        where.append("scope = ?"); params.append(scope)
    if scope_id is not None:
        where.append("scope_id = ?"); params.append(scope_id)
    sql = "SELECT * FROM tool_policy_overrides"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY scope, scope_id, tool"
    return [_policy_row(r) for r in db.execute(sql, tuple(params)).fetchall()]


def get_tool_override(db: Any, override_id: int) -> dict | None:
    db.ensure_schema(POLICY_SCHEMA_NAME)
    row = db.execute("SELECT * FROM tool_policy_overrides WHERE id = ?",
                     (int(override_id),)).fetchone()
    return _policy_row(row) if row else None


def set_tool_override(db: Any, scope: str, scope_id: str | None, tool: str, decision: str,
                      created_by: str | None = None,
                      expires_in_s: float | None = None) -> dict:
    """Insert or replace one override (UNIQUE per scope, scope_id, tool)."""
    scope = str(scope or "").strip().lower()
    if scope not in POLICY_SCOPES:
        raise ValueError(f"scope must be one of {', '.join(POLICY_SCOPES)}")
    tool = str(tool or "").strip()
    if tool.startswith("risk:send"):
        tool = "risk:external"
    if not _POLICY_RULE_KEY.match(tool):
        raise ValueError("tool must be a tool name, cap:<capability>, risk:<class> or *")
    decision = _policy_norm_decision(decision)
    if decision not in POLICY_DECISIONS:
        raise ValueError("decision must be allow, ask or deny")
    scope_id = "" if scope == "global" else str(scope_id or "").strip()
    if scope != "global" and not scope_id:
        raise ValueError(f"scope_id is required for scope {scope}")
    expires_at = None
    if expires_in_s:
        expires_at = (datetime.now(timezone.utc)
                      + timedelta(seconds=max(1.0, float(expires_in_s)))).isoformat()
    db.ensure_schema(POLICY_SCHEMA_NAME)
    db.execute(
        "INSERT INTO tool_policy_overrides (scope, scope_id, tool, decision, created_by, "
        "created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(scope, scope_id, tool) DO UPDATE SET decision = excluded.decision, "
        "created_by = excluded.created_by, created_at = excluded.created_at, "
        "expires_at = excluded.expires_at",
        (scope, scope_id, tool, decision, created_by, _policy_now_iso(), expires_at))
    db.commit()
    _policy_mark_rows(db, True)
    row = db.execute("SELECT * FROM tool_policy_overrides WHERE scope = ? AND scope_id = ? "
                     "AND tool = ?", (scope, scope_id, tool)).fetchone()
    result = _policy_row(row) if row else {}
    log_event(_policy_change_log, logging.INFO, "policy.changed", action="set",
              scope=scope, scope_id=scope_id, tool=tool, decision=decision,
              by=created_by, expires_at=expires_at)
    return result


def delete_tool_override(db: Any, override_id: int, by: str | None = None) -> dict | None:
    row = get_tool_override(db, override_id)
    if not row:
        return None
    db.execute("DELETE FROM tool_policy_overrides WHERE id = ?", (int(override_id),))
    db.commit()
    _policy_mark_rows(db, None)
    log_event(_policy_change_log, logging.INFO, "policy.changed", action="delete",
              scope=row["scope"], scope_id=row["scope_id"], tool=row["tool"],
              decision=row["decision"], by=by)
    return row


def set_profile_tool_policy(db: Any, agent_id: str, policy: dict | str | None,
                            by: str | None = None) -> dict[str, str]:
    """Store an agent profile's rules on agents.tool_policy (validated)."""
    rules = _policy_rules_from(policy or {})
    db.ensure_schema(POLICY_SCHEMA_NAME)
    db.execute("UPDATE agents SET tool_policy = ? WHERE id = ?",
               (json.dumps(rules) if rules else "", agent_id))
    db.commit()
    log_event(_policy_change_log, logging.INFO, "policy.changed", action="profile",
              scope="profile", scope_id=agent_id, rules=len(rules), by=by)
    return rules


def _policy_scoped_rules(db: Any, scopes: list[tuple[str, str]]) -> dict[tuple[str, str], dict]:
    """{(scope, scope_id): {key: (decision, row id)}} for live (unexpired) rows."""
    out: dict[tuple[str, str], dict] = {}
    if db is None or not scopes or not _policy_has_rows(db):
        return out
    clauses = " OR ".join("(scope = ? AND scope_id = ?)" for _ in scopes)
    params: list[Any] = []
    for scope, scope_id in scopes:
        params += [scope, scope_id]
    try:
        rows = db.execute(f"SELECT id, scope, scope_id, tool, decision, expires_at "
                          f"FROM tool_policy_overrides WHERE {clauses}",
                          tuple(params)).fetchall()
    except Exception as exc:  # noqa: BLE001
        log(f"tool policy: could not read overrides: {redact_text(str(exc))}", logging.WARNING)
        return out
    for row in rows:
        if _policy_expired(row["expires_at"]):
            continue
        out.setdefault((row["scope"], row["scope_id"]), {})[row["tool"]] = (
            row["decision"], row["id"])
    return out


def record_tool_decision(db: Any, *, tool: str, decision: str, risk: str | None = None,
                         rule: str | None = None, args: Any = None,
                         user_id: str | None = None, conversation_id: str | None = None,
                         turn_id: str | None = None, approver_id: str | None = None,
                         latency_ms: int | None = None) -> None:
    """Append one audit row. Never raises into a tool call."""
    if db is None:
        return
    try:
        db.ensure_schema(POLICY_SCHEMA_NAME)
        db.execute(
            "INSERT INTO tool_decisions (created_at, user_id, conversation_id, turn_id, tool, "
            "args_redacted, risk, decision, rule, approver_id, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (_policy_now_iso(), user_id, conversation_id, turn_id, str(tool)[:128],
             _policy_redact_args(args), risk, decision, (rule or "")[:200], approver_id,
             int(latency_ms) if latency_ms is not None else None))
        db.commit()
    except Exception as exc:  # noqa: BLE001 - audit failure is logged, not fatal
        log(f"tool policy: audit write failed: {redact_text(str(exc))}", logging.ERROR)


def list_tool_decisions(db: Any, *, user_id: str | None = None,
                        conversation_id: str | None = None, tool: str | None = None,
                        limit: int = 100, before_id: int | None = None) -> list[dict]:
    """Newest first. user_id None = every user (admin view)."""
    db.ensure_schema(POLICY_SCHEMA_NAME)
    where, params = [], []
    if user_id is not None:
        where.append("user_id = ?"); params.append(user_id)
    if conversation_id:
        where.append("conversation_id = ?"); params.append(conversation_id)
    if tool:
        where.append("tool = ?"); params.append(tool)
    if before_id:
        where.append("id < ?"); params.append(int(before_id))
    sql = "SELECT * FROM tool_decisions"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(max(1, min(int(limit or 100), 1000)))
    out = []
    for row in db.execute(sql, tuple(params)).fetchall():
        item = dict(row)
        try:
            item["args"] = json.loads(item.pop("args_redacted") or "{}")
        except (TypeError, ValueError):
            item["args"] = {}
        out.append(item)
    return out


# --------------------------------------------------------------------------- #
# Binding
# --------------------------------------------------------------------------- #
def bind_policy(config: Any = None, db: Any = None) -> None:
    """Bind the live config/db for decide() callers that pass neither."""
    if config is not None:
        _POLICY_BOUND["config"] = config
    if db is not None:
        _POLICY_BOUND["db"] = db


def _policy_identity(turn: TurnContext | None, conversation_id: str | None = None):
    user_id = (turn.user_id if turn is not None and turn.user_id else None) or get_acting_user()
    conv = ((turn.conversation_id if turn is not None and turn.conversation_id else None)
            or conversation_id or get_acting_conversation())
    return user_id, conv


def _policy_role(turn: TurnContext | None, db: Any, user_id: str | None) -> str | None:
    if turn is not None:
        return turn.role
    if db is not None and user_id:
        try:
            user = db.get_user(user_id)
        except Exception:  # noqa: BLE001
            user = None
        if user:
            return user.get("role")
    return None


def _policy_turn_taint(turn: TurnContext | None) -> list[str]:
    if turn is None:
        return []
    extras = getattr(turn, "extras", None)
    if not isinstance(extras, dict):
        return []
    return list(extras.get("policy_taint") or [])


def note_tool_result(name: str, tool: Any = None, error: str | None = None,
                     turn: TurnContext | None = None) -> None:
    """After a call: remember that this turn has now read untrusted content."""
    turn = turn if turn is not None else current_turn()
    if turn is None or error or not tool_taints(name, tool):
        return
    extras = getattr(turn, "extras", None)
    if not isinstance(extras, dict):
        return
    sources = extras.setdefault("policy_taint", [])
    if name not in sources:
        sources.append(name)


# --------------------------------------------------------------------------- #
# decide()
# --------------------------------------------------------------------------- #
def decide(tool_name: str, args: dict | None = None, turn: TurnContext | None = None, *,
           registry: Any = None, config: Any = None, db: Any = None, tool: Any = None,
           conversation_id: str | None = None, consume: bool = False,
           audit: bool = True) -> Decision:
    """Fold every policy layer into one Decision for this call.

    `turn` defaults to current_turn(). Pass `registry` (or config/db/tool) when
    you have it; otherwise the values bound by register_policy are used. With
    consume=True a matching one-shot approval is used up (ToolRegistry.call
    does this; the agent's pre-check must not). A deny is written to the audit
    table unless audit=False: the agent loop does not call the registry after
    a deny, so this is the only place that refusal would be recorded. Allows
    are audited by ToolRegistry.call and approvals by the broker.
    """
    if turn is None:
        turn = current_turn()
    if registry is not None:
        config = config if config is not None else getattr(registry, "config", None)
        db = db if db is not None else getattr(registry, "db", None)
        if tool is None:
            try:
                tool = registry.get(tool_name)
            except Exception:  # noqa: BLE001
                tool = None
    config = config if config is not None else _POLICY_BOUND["config"]
    db = db if db is not None else _POLICY_BOUND["db"]
    args = args if isinstance(args, dict) else {}
    risk = classify_tool_risk(tool_name, args, tool)
    if config is None:
        return Decision("allow", risk, "no tool policy is configured", "unconfigured")
    if _policy_fast_allow(config, db, turn):
        return Decision("allow", risk, "tool approvals are off", "default:off")

    mode = _policy_mode(config)
    if mode == "off":
        action, rule, reason = "allow", "default:off", "tool approvals are off"
    elif mode == "all":
        action, rule, reason = "ask", "default:all", "every tool call needs approval"
    elif risk == "read":
        action, rule, reason = "allow", "default:risky", "read-only tools run without approval"
    else:
        action, rule, reason = ("ask", "default:risky",
                                f"{tool_name} is a {risk} tool, which needs approval")

    user_id, conv = _policy_identity(turn, conversation_id)
    profile = turn.agent_profile if turn is not None else None
    profile_id = str(profile.get("id") or "") if isinstance(profile, dict) else ""
    capability = _policy_capability(tool_name, tool)

    scopes = [("global", "")]
    if profile_id:
        scopes.append(("profile", profile_id))
    if user_id:
        scopes.append(("user", str(user_id)))
    if conv:
        scopes.append(("conversation", str(conv)))
    rows = _policy_scoped_rules(db, scopes)

    def flat(key: tuple[str, str]) -> dict[str, str]:
        return {k: v[0] for k, v in rows.get(key, {}).items()}

    profile_rules = profile_tool_policy(profile)
    if profile_id:
        profile_rules.update(flat(("profile", profile_id)))
    # The role is only looked up (a users-table read without a turn) when
    # role rules exist at all.
    role = (_policy_role(turn, db, user_id)
            if str(getattr(config, "tool_approval_roles", "") or "").strip() else None)
    layers: list[tuple[str, dict[str, str], bool]] = [
        ("global", flat(("global", "")), True),
        (f"profile:{profile_id}" if profile_id else "profile", profile_rules, True),
        (f"role:{role}", _policy_role_rules(config, role), True),
        ("user", flat(("user", str(user_id))) if user_id else {}, False),
        ("conversation", flat(("conversation", str(conv))) if conv else {}, False),
    ]
    sticky = False
    for label, rules, admin in layers:
        hit = _policy_match(rules, tool_name, risk, capability)
        if hit is None or sticky:
            continue
        action, key = hit
        rule = f"{label}:{key}"
        reason = f"{label} rule {key} says {action}"
        if admin and action == "deny":
            # An administrator's deny is final; later layers cannot loosen it.
            sticky = True

    # Taint: untrusted content read earlier in this turn may be what is asking.
    taint = _policy_turn_taint(turn)
    if (action == "allow" and taint and mode != "off" and risk in _POLICY_ESCALATED
            and bool(getattr(config, "tool_approval_taint", True))):
        action, rule = "ask", f"taint:{taint[0]}"
        reason = (f"this turn read untrusted content ({', '.join(taint)}), so a {risk} "
                  f"call needs a fresh approval")

    approval_id = approver = latency = None
    if action in ("ask", "allow") and turn is not None:
        grant = BROKER.take_grant(turn, tool_name, args, consume=consume)
        if grant is not None and action == "ask":
            action, rule = "allow", f"oneshot:{grant['approval_id']}"
            reason = "approved once by the user"
            approval_id, approver, latency = (grant["approval_id"], grant["by"],
                                              grant.get("latency_ms"))

    if action == "ask" and (turn is None or not getattr(turn, "interactive", False)):
        unattended = str(getattr(config, "tool_approval_unattended", "deny") or "deny").lower()
        if unattended == "allow":
            action, rule = "allow", "unattended:allow"
            reason = f"{reason}; nobody can approve here and unattended calls are allowed"
        else:
            action, rule = "deny", "unattended:deny"
            reason = f"{reason}, and nobody can approve it here (unattended run)"
    if audit and action == "deny":
        record_tool_decision(db, tool=tool_name, decision="deny", risk=risk, rule=rule,
                             args=args, user_id=user_id, conversation_id=conv,
                             turn_id=turn.turn_id if turn is not None else None)
    return Decision(action, risk, reason, rule, approval_id, approver, latency)


def _policy_fast_allow(config: Any, db: Any, turn: TurnContext | None) -> bool:
    """True when the answer is today's allow without consulting anything."""
    if config is None:
        return True
    if _policy_mode(config) != "off":
        return False
    if str(getattr(config, "tool_approval_roles", "") or "").strip():
        return False
    if turn is not None and profile_tool_policy(turn.agent_profile):
        return False
    return not _policy_has_rows(db)


def enforce_tool_policy(registry: Any, tool: Any, name: str, args: dict | None,
                        conversation_id: str | None = None, *,
                        audit_args: Any = None) -> str | None:
    """Synchronous fail-closed check run inside ToolRegistry.call.

    Returns None to run the call, or the refusal text. "ask" without a
    one-shot approval is refused here: this code runs in a worker thread with
    no way to show a prompt, and the agent loop is the place that asks.
    """
    # The registry's own config and db, never the module binding: a registry
    # built outside create_app (CLI, tests, a sub-agent) enforces its own
    # settings, and binding it here would leak them into every later decide().
    config = getattr(registry, "config", None)
    db = getattr(registry, "db", None)
    turn = current_turn()
    if _policy_fast_allow(config, db, turn):
        return None
    decision = decide(name, args, turn, config=config, db=db, tool=tool,
                      conversation_id=conversation_id, consume=True, audit=False)
    if decision.action == "ask":
        decision = Decision(
            "deny", decision.risk,
            f"{decision.reason}; it was not approved and this call path cannot ask",
            f"enforce:{decision.rule}")
    user_id, conv = _policy_identity(turn, conversation_id)
    mode = _policy_mode(config)
    if decision.action == "deny" or mode != "off" or not decision.rule.startswith("default:"):
        record_tool_decision(
            db, tool=name, decision=decision.action, risk=decision.risk, rule=decision.rule,
            args=audit_args if audit_args is not None else args, user_id=user_id,
            conversation_id=conv, turn_id=turn.turn_id if turn is not None else None,
            approver_id=decision.approver_id, latency_ms=decision.latency_ms)
    if decision.action == "deny":
        log_event(_policy_log, logging.WARNING, "tool.denied", tool=name,
                  risk=decision.risk, rule=decision.rule,
                  turn_id=turn.turn_id if turn is not None else None)
        return f"Tool call refused by policy ({decision.rule}): {decision.reason}."
    return None


def reference_refusal(registry: Any, token: str) -> str | None:
    """Policy for one @reference in a message (expand_references).

    A pasted @url is a fetch and @file a read, so they get the same decision
    and the same audit row as the tool would, and a fetched page taints the
    turn. "ask" cannot be answered during reference expansion, so it skips the
    reference with a note; the model can still call the tool, which asks.
    """
    token = str(token or "")
    if token == "diff":
        name, args = "git_diff", {"path": "diff"}
    elif token.lower().startswith(("http://", "https://")):
        name, args = "fetch_url", {"url": token}
    else:
        name, args = "read_file", {"path": token}
    tool = None
    try:
        tool = registry.get(name)
    except Exception:  # noqa: BLE001
        tool = None
    refusal = enforce_tool_policy(registry, tool, name, args,
                                  audit_args={**args, "via": "@reference"})
    if refusal:
        return f"skipped @{token}: {refusal}"
    note_tool_result(name, tool, None)
    return None


# --------------------------------------------------------------------------- #
# The approval broker
# --------------------------------------------------------------------------- #
class ApprovalBroker:
    """Pending approvals, one-shot grants, and their resolution.

    Thread-safe: the agent waits on the event loop, the approval route may
    resolve from the same loop or a worker thread, so waiting polls a
    threading.Event instead of an asyncio.Event bound to one loop.
    """

    GRANT_TTL_S = 600.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, dict] = {}
        self._grants: dict[tuple[str, str, str], dict] = {}

    # ---- config ---------------------------------------------------------- #
    @staticmethod
    def _config() -> Any:
        return _POLICY_BOUND["config"]

    @staticmethod
    def _db() -> Any:
        return _POLICY_BOUND["db"]

    def default_timeout(self) -> float:
        cfg = self._config()
        try:
            return float(getattr(cfg, "tool_approval_timeout_s", 120) or 120)
        except (TypeError, ValueError):
            return 120.0

    # ---- requests -------------------------------------------------------- #
    def request(self, turn: TurnContext | None, call_id: str | None, name: str,
                args: dict | None, risk: str, reason: str, *,
                timeout: float | None = None) -> str:
        """Register a pending approval for this turn; returns its id."""
        turn = turn if turn is not None else current_turn()
        if turn is None:
            raise ValueError("an approval needs a turn (nobody could answer it)")
        self._prune()
        approval_id = "ap_" + uuid.uuid4().hex[:16]
        wait_s = float(timeout if timeout is not None else self.default_timeout())
        entry = {
            "id": approval_id, "turn_id": turn.turn_id, "turn": turn,
            "user_id": turn.user_id, "conversation_id": turn.conversation_id,
            "call_id": call_id, "name": str(name), "args": dict(args or {}),
            "risk": risk, "reason": reason, "created": time.monotonic(),
            "created_at": _policy_now_iso(), "timeout": wait_s,
            "event": threading.Event(), "claimed": False, "resolution": None,
        }
        with self._lock:
            self._pending[approval_id] = entry
        log_event(_policy_log, logging.INFO, "tool.approval_requested",
                  approval_id=approval_id, tool=name, risk=risk, call_id=call_id,
                  turn_id=turn.turn_id, conversation_id=turn.conversation_id,
                  timeout_s=wait_s)
        return approval_id

    def _prune(self) -> None:
        now = time.monotonic()
        with self._lock:
            stale = [k for k, e in self._pending.items()
                     if now - e["created"] > e["timeout"] * 2 + 60]
            for key in stale:
                self._pending.pop(key, None)
            old = [k for k, g in self._grants.items() if g["expires"] < now]
            for key in old:
                self._grants.pop(key, None)

    def get(self, approval_id: str) -> dict | None:
        with self._lock:
            entry = self._pending.get(approval_id)
        return self._view(entry) if entry else None

    def _view(self, entry: dict) -> dict:
        remaining = max(0, int(entry["timeout"] - (time.monotonic() - entry["created"])))
        return {"id": entry["id"], "turn_id": entry["turn_id"],
                "conversation_id": entry["conversation_id"], "call_id": entry["call_id"],
                "name": entry["name"], "args": redact_obj(entry["args"]),
                "risk": entry["risk"], "reason": entry["reason"],
                "created_at": entry["created_at"], "expires_in": remaining,
                "scopes": self._scopes(entry)}

    @staticmethod
    def _scopes(entry: dict) -> list[str]:
        return [s for s in APPROVAL_SCOPES
                if s != "conversation" or entry.get("conversation_id")]

    def pending(self, user_id: str | None = None) -> list[dict]:
        """Unanswered approvals, the caller's own (None = everyone's)."""
        with self._lock:
            entries = [e for e in self._pending.values() if not e["claimed"]]
        if user_id is not None:
            entries = [e for e in entries
                       if (e["user_id"] or SENTINEL_LOCAL_USER) == user_id]
        return [self._view(e) for e in sorted(entries, key=lambda e: e["created"])]

    def event(self, approval_id: str, step: Any = None) -> dict:
        """The approval_required stream event for a pending approval."""
        with self._lock:
            entry = self._pending.get(approval_id)
        if entry is None:
            return {"type": "approval_required", "id": approval_id, "scopes": []}
        view = self._view(entry)
        return {"type": "approval_required", "id": approval_id, "call_id": view["call_id"],
                "name": view["name"], "args": view["args"], "risk": view["risk"],
                "reason": view["reason"], "step": step, "expires_in": view["expires_in"],
                "scopes": view["scopes"]}

    @staticmethod
    def resolved_event(resolution: Resolution) -> dict:
        """The approval_resolved stream event for a finished wait."""
        return {"type": "approval_resolved", "id": resolution.approval_id,
                "call_id": resolution.call_id, "decision": resolution.decision,
                "scope": resolution.scope, "by": resolution.by}

    # ---- waiting --------------------------------------------------------- #
    def _claim(self, approval_id: str) -> dict | None:
        with self._lock:
            entry = self._pending.get(approval_id)
            if entry is None or entry["claimed"]:
                return None
            entry["claimed"] = True
            return entry

    def _finish(self, entry: dict, resolution: Resolution) -> None:
        resolution.approval_id = entry["id"]
        resolution.call_id = entry["call_id"]
        entry["resolution"] = resolution
        entry["event"].set()
        log_event(_policy_log, logging.INFO, "tool.approval_decided",
                  approval_id=entry["id"], tool=entry["name"], decision=resolution.decision,
                  scope=resolution.scope, by=resolution.by,
                  latency_ms=resolution.latency_ms, turn_id=entry["turn_id"])

    def _expire(self, entry: dict, decision: str, by: str, reason: str) -> None:
        if self._claim(entry["id"]) is None:
            return
        latency = int((time.monotonic() - entry["created"]) * 1000)
        record_tool_decision(
            self._db(), tool=entry["name"], decision=decision, risk=entry["risk"],
            rule=f"approval:{by}", args=entry["args"], user_id=entry["user_id"],
            conversation_id=entry["conversation_id"], turn_id=entry["turn_id"],
            approver_id=None, latency_ms=latency)
        self._finish(entry, Resolution(decision if decision != "cancelled" else "deny",
                                       "once", None, by=by, latency_ms=latency,
                                       reason=reason))

    def _step(self, approval_id: str, deadline: float) -> Resolution | None:
        """One poll: the resolution if finished (expiring it when due)."""
        with self._lock:
            entry = self._pending.get(approval_id)
        if entry is None:
            return Resolution("deny", "once", None, by="unknown",
                              reason="unknown or expired approval", approval_id=approval_id)
        if not entry["event"].is_set() and not entry["claimed"]:
            turn = entry.get("turn")
            if turn is not None and turn.cancelled():
                self._expire(entry, "deny", "cancel", "the turn was cancelled")
            elif time.monotonic() >= deadline:
                self._expire(entry, "timeout", "timeout", "nobody answered in time")
        if entry["event"].is_set():
            with self._lock:
                self._pending.pop(approval_id, None)
            return entry["resolution"]
        return None

    def _deadline(self, approval_id: str, timeout: float | None) -> float:
        with self._lock:
            entry = self._pending.get(approval_id)
        if timeout is not None:
            return time.monotonic() + max(0.0, float(timeout))
        if entry is None:
            return time.monotonic()
        return entry["created"] + entry["timeout"]

    async def wait(self, approval_id: str, timeout: float | None = None) -> Resolution:
        """Wait for the owner's answer, the timeout, or the turn's cancellation."""
        deadline = self._deadline(approval_id, timeout)
        while True:
            result = self._step(approval_id, deadline)
            if result is not None:
                return result
            await asyncio.sleep(0.05)

    def wait_sync(self, approval_id: str, timeout: float | None = None) -> Resolution:
        """wait() for code running in a worker thread."""
        deadline = self._deadline(approval_id, timeout)
        while True:
            result = self._step(approval_id, deadline)
            if result is not None:
                return result
            with self._lock:
                entry = self._pending.get(approval_id)
            if entry is not None:
                entry["event"].wait(0.05)

    # ---- answering ------------------------------------------------------- #
    def resolve(self, approval_id: str, user_id: str | None, decision: str,
                scope: str = "once", args: dict | None = None, *,
                turn_id: str | None = None) -> bool:
        """Answer a pending approval. Owner-checked; False if not found/not yours.

        Raises ValueError for a malformed answer (unknown decision or scope,
        args that are not an object, a conversation scope with no
        conversation) so the route can say 400 rather than 404.
        """
        action = _policy_norm_decision(decision)
        if action not in ("allow", "deny"):
            raise ValueError("decision must be allow or deny")
        scope = str(scope or "once").strip().lower()
        if scope not in APPROVAL_SCOPES:
            raise ValueError(f"scope must be one of {', '.join(APPROVAL_SCOPES)}")
        if args is not None and not isinstance(args, dict):
            raise ValueError("args must be a JSON object")
        with self._lock:
            entry = self._pending.get(approval_id)
            if entry is None or entry["claimed"]:
                return False
            owner = entry["user_id"] or SENTINEL_LOCAL_USER
            if owner != (user_id or SENTINEL_LOCAL_USER):
                return False
            if turn_id is not None and entry["turn_id"] != turn_id:
                return False
            if scope == "conversation" and not entry["conversation_id"]:
                raise ValueError("this turn has no conversation; use scope once or tool")
            entry["claimed"] = True
        final_args = (args if args is not None else entry["args"]) if action == "allow" else None
        latency = int((time.monotonic() - entry["created"]) * 1000)
        db = self._db()
        # Grant and override rows are written BEFORE the waiter is released:
        # the agent calls ToolRegistry.call the moment wait() returns, and a
        # grant written after that would make the call fail closed.
        if action == "allow":
            key = (entry["turn_id"], entry["name"], _policy_fingerprint(final_args))
            with self._lock:
                self._grants[key] = {"approval_id": approval_id, "by": user_id,
                                     "latency_ms": latency,
                                     "expires": time.monotonic() + self.GRANT_TTL_S}
        if scope != "once" and db is not None:
            try:
                if scope == "conversation":
                    set_tool_override(db, "conversation", entry["conversation_id"],
                                      entry["name"], action, created_by=user_id)
                else:
                    set_tool_override(db, "user", owner, entry["name"], action,
                                      created_by=user_id)
            except Exception as exc:  # noqa: BLE001 - the answer itself still counts
                log(f"tool policy: could not save the {scope} override: "
                    f"{redact_text(str(exc))}", logging.WARNING)
        record_tool_decision(
            db, tool=entry["name"], decision=action, risk=entry["risk"],
            rule=f"approval:{scope}", args=final_args if final_args is not None else entry["args"],
            user_id=entry["user_id"], conversation_id=entry["conversation_id"],
            turn_id=entry["turn_id"], approver_id=user_id, latency_ms=latency)
        self._finish(entry, Resolution(action, scope, final_args, by=user_id,
                                       latency_ms=latency))
        return True

    def cancel_turn(self, turn_id: str) -> int:
        """Deny every pending approval of a turn (it was cancelled or ended)."""
        with self._lock:
            entries = [e for e in self._pending.values()
                       if e["turn_id"] == turn_id and not e["claimed"]]
        for entry in entries:
            self._expire(entry, "deny", "cancel", "the turn was cancelled")
        return len(entries)

    # ---- one-shot grants ------------------------------------------------- #
    def take_grant(self, turn: TurnContext | None, name: str, args: dict | None,
                   consume: bool = True) -> dict | None:
        """The approval of exactly this call in this turn, if any."""
        if turn is None:
            return None
        key = (turn.turn_id, str(name), _policy_fingerprint(args or {}))
        with self._lock:
            grant = self._grants.get(key)
            if grant is None:
                return None
            if grant["expires"] < time.monotonic():
                self._grants.pop(key, None)
                return None
            if consume:
                self._grants.pop(key, None)
            return dict(grant)

    def clear(self) -> None:
        """Forget everything (tests)."""
        with self._lock:
            self._pending.clear()
            self._grants.clear()


BROKER = ApprovalBroker()


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
def _policy_owns_conversation(db: Any, conversation_id: str, user_id: str) -> bool:
    row = db.execute("SELECT 1 FROM messages WHERE conversation_id = ? AND user_id = ? LIMIT 1",
                     (conversation_id, user_id)).fetchone()
    return row is not None


def _policy_visible(db: Any, row: dict, user: dict) -> bool:
    """Whether a non-admin may see (and delete) an override row."""
    uid = user["id"]
    if row["scope"] == "user":
        return row["scope_id"] == uid
    if row["scope"] == "conversation":
        return (row.get("created_by") == uid
                or _policy_owns_conversation(db, row["scope_id"], uid))
    return False


def register_policy(app, ctx) -> None:
    """Mount the approval, tool-policy and audit routes."""
    from fastapi.responses import JSONResponse

    bind_policy(ctx.config, ctx.db)
    ctx.extras["approvals"] = BROKER
    db = ctx.db
    USER = ctx.USER

    def is_admin(user: dict) -> bool:
        return user.get("role") == "admin"

    def risk_table() -> dict[str, str]:
        registry = ctx.registry
        names = []
        try:
            names = list(registry.names()) if registry is not None else []
        except Exception:  # noqa: BLE001
            names = []
        out = {}
        for name in sorted(set(names) | set(_POLICY_RISK_BY_TOOL)):
            tool = registry.get(name) if registry is not None else None
            out[name] = classify_tool_risk(name, {}, tool)
        return out

    @app.get("/api/approvals/pending")
    async def approvals_pending(user: dict = USER):
        return {"approvals": BROKER.pending(user["id"])}

    @app.get("/api/tool-policies")
    async def tool_policies(scope: str | None = None, scope_id: str | None = None,
                            user: dict = USER):
        rows = await asyncio.to_thread(list_tool_overrides, db, scope or None,
                                       scope_id if scope_id else None)
        if not is_admin(user):
            visible = []
            for row in rows:
                if row["scope"] == "global" or await asyncio.to_thread(
                        _policy_visible, db, row, user):
                    visible.append(row)
            rows = visible
        cfg = ctx.config
        return {
            "policies": rows,
            "mode": _policy_mode(cfg),
            "unattended": str(getattr(cfg, "tool_approval_unattended", "deny")),
            "timeout_s": getattr(cfg, "tool_approval_timeout_s", 120),
            "taint": bool(getattr(cfg, "tool_approval_taint", True)),
            "risks": risk_table(),
            "risk_classes": list(RISK_CLASSES),
            "scopes": list(POLICY_SCOPES),
        }

    @app.post("/api/tool-policies")
    async def set_tool_policy(body: dict, user: dict = USER):
        if not isinstance(body, dict):
            return JSONResponse({"error": "expected a JSON object"}, status_code=400)
        scope = str(body.get("scope") or "").strip().lower()
        if scope not in POLICY_SCOPES:
            return JSONResponse({"error": f"scope must be one of {', '.join(POLICY_SCOPES)}"},
                                status_code=400)
        scope_id = str(body.get("scope_id") or "").strip()
        admin = is_admin(user)
        if scope in ("profile", "global") and not admin:
            return JSONResponse({"error": "administrator access required"}, status_code=403)
        if scope == "user":
            scope_id = scope_id or user["id"]
            if scope_id != user["id"] and not admin:
                return JSONResponse({"error": "you can only set your own user policy"},
                                    status_code=403)
        elif scope == "conversation":
            if not scope_id or len(scope_id) > 128:
                return JSONResponse({"error": "scope_id (the conversation id) is required"},
                                    status_code=400)
            # Ownership, not can_access_conversation: that one also passes for a
            # conversation nobody has written to yet, and an "allow" planted on
            # someone else's future conversation id would loosen THEIR policy.
            # (A turn's own approvals write these rows through BROKER.resolve,
            # which is owner-checked against the turn instead.)
            if not admin and not await asyncio.to_thread(
                    _policy_owns_conversation, db, scope_id, user["id"]):
                return JSONResponse({"error": "conversation not found"}, status_code=404)
        elif scope == "profile":
            if not scope_id or not await asyncio.to_thread(db.get_agent, scope_id):
                return JSONResponse({"error": "agent profile not found"}, status_code=404)
        expires = body.get("expires_in_s")
        try:
            expires_s = float(expires) if expires not in (None, "", 0) else None
        except (TypeError, ValueError):
            return JSONResponse({"error": "expires_in_s must be a number of seconds"},
                                status_code=400)
        try:
            row = await asyncio.to_thread(
                set_tool_override, db, scope, scope_id, body.get("tool"),
                body.get("decision"), user["id"], expires_s)
        except ValueError as exc:
            return JSONResponse({"error": redact_text(str(exc))}, status_code=400)
        return {"policy": row}

    @app.delete("/api/tool-policies/{policy_id}")
    async def delete_tool_policy(policy_id: int, user: dict = USER):
        row = await asyncio.to_thread(get_tool_override, db, policy_id)
        if not row:
            return JSONResponse({"error": "policy not found"}, status_code=404)
        if not is_admin(user):
            if not await asyncio.to_thread(_policy_visible, db, row, user):
                return JSONResponse({"error": "policy not found"}, status_code=404)
            # Users remove their own rules; one an administrator set on their
            # user or conversation stays until an administrator removes it.
            if row.get("created_by") != user["id"]:
                return JSONResponse({"error": "set by an administrator"}, status_code=403)
        await asyncio.to_thread(delete_tool_override, db, policy_id, user["id"])
        return {"deleted": policy_id}

    @app.get("/api/audit/tool-decisions")
    async def tool_decisions(limit: int = 100, conversation_id: str | None = None,
                             tool: str | None = None, user_id: str | None = None,
                             before_id: int | None = None, user: dict = USER):
        scoped = user_id if is_admin(user) else user["id"]
        rows = await asyncio.to_thread(
            list_tool_decisions, db, user_id=scoped or None,
            conversation_id=(conversation_id or None), tool=(tool or None),
            limit=limit, before_id=before_id)
        return {"decisions": rows}


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'POLICY_SCHEMA_NAME',
    'RISK_CLASSES',
    'POLICY_DECISIONS',
    'POLICY_SCOPES',
    'APPROVAL_SCOPES',
    'APPROVAL_MODES',
    'Decision',
    'Resolution',
    'ApprovalBroker',
    'BROKER',
    'decide',
    'enforce_tool_policy',
    'reference_refusal',
    'note_tool_result',
    'tool_taints',
    'classify_tool_risk',
    'risk_from_mcp_annotations',
    'profile_tool_policy',
    'set_profile_tool_policy',
    'list_tool_overrides',
    'get_tool_override',
    'set_tool_override',
    'delete_tool_override',
    'record_tool_decision',
    'list_tool_decisions',
    'bind_policy',
    'register_policy',
    '_policy_schema',
    '_policy_mode',
    '_policy_fingerprint',
    '_policy_rules_from',
    '_POLICY_RISK_BY_TOOL',
    '_POLICY_BOUND',
]
