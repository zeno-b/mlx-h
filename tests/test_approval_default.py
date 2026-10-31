"""The "ask before every action" rule: per-user approval modes (policy.py).

The rule, in the user's words: always ask the user for every action unless
this is overwritten with a setting that is always visible in the UI. So:

  * a user who never chose gets APPROVAL_DEFAULT_MODE, which is "all": even
    list_files waits for a yes in an interactive turn;
  * "risky" (Ask for changes) asks for writes, code and sends but not reads;
    "off" (Don't ask) asks for nothing;
  * TOOL_APPROVAL_MODE is the administrator's floor: stricter wins, and an
    administrator deny rule is still final;
  * an unattended run (POST /api/chat, /v1 agent mode, a scheduled task, the
    /mcp server) cannot ask, so it refuses with a message naming the header
    control, and logs the same;
  * the preference is per user, persisted, and isolated between users;
  * the header control is in the page for every role, outside Settings, with
    literal ids that exist.

tests/conftest.py runs the rest of the suite with APPROVAL_DEFAULT_MODE=off
(those tests predate the rule); every test here sets the mode it tests
explicitly, so this file does not depend on that and runs standalone too.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import local_llm.config as config_mod  # noqa: E402

config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-approval-default-"))

from local_llm import policy  # noqa: E402
from local_llm.config import Config  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.policy import (  # noqa: E402
    APPROVAL_UNATTENDED_HINT, APPROVAL_UNATTENDED_HINT_ADMIN,
    APPROVAL_UNATTENDED_HINT_CHANGES, BROKER, approval_state, bind_policy, decide,
    reference_refusal, set_tool_override, user_approval_choice)
from local_llm.prefs import PreferenceStore  # noqa: E402
from local_llm.settings_schema import (  # noqa: E402
    SETTINGS_FIELDS, USER_PREFS, settings_field_sub, validate_user_prefs)
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.turnctx import TurnContext, reset_current_turn, set_current_turn  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _registry(**overrides):
    """A registry on a temp DB, its tools stubbed to record. Mode "all" by
    default: the rule this file tests, whatever the session default is."""
    base = dict(skills_dir=tempfile.mkdtemp(prefix="llm-test-appr-skills-"),
                project_dir=tempfile.mkdtemp(prefix="llm-test-appr-proj-"),
                approval_default_mode="all")
    base.update(overrides)
    cfg = Config(**base)
    db = Database(Path(tempfile.mkdtemp()) / "approvals.db")
    reg = ToolRegistry(cfg, db)
    calls: list[tuple[str, dict]] = []

    def stub(name, text):
        def handler(**kw):
            calls.append((name, kw))
            return text
        reg._tools[name].handler = handler

    for name, text in (("list_files", "README.md\nsrc/"), ("read_file", "file text"),
                       ("write_file", "wrote it")):
        stub(name, text)
    bind_policy(cfg, db)
    BROKER.clear()
    return cfg, db, reg, calls


_SAVED_BINDING = dict(policy._POLICY_BOUND)


def teardown_function(_fn=None):
    policy._POLICY_BOUND.update(_SAVED_BINDING)
    BROKER.clear()


class _Bound:
    def __init__(self, turn):
        self.turn = turn

    def __enter__(self):
        self.token = set_current_turn(self.turn)
        return self.turn

    def __exit__(self, *exc):
        reset_current_turn(self.token)


def _turn(user="u1", conv="c1", interactive=True, **kw):
    return TurnContext(user_id=user, conversation_id=conv, interactive=interactive, **kw)


def _choose(db, user, mode):
    return PreferenceStore(db).update(user, {"approval_mode": mode})


class _Scripted:
    """A ModelClient stand-in that replays canned replies."""

    def __init__(self, replies):
        self.replies = list(replies)

    def _next(self):
        return self.replies.pop(0) if self.replies else "done."

    def stream(self, messages, *a, **kw):
        text = self._next()

        async def gen():
            yield text
        return gen()

    async def complete_with_stats(self, messages, *a, **kw):
        from local_llm.llm import GenerationStats
        return self._next(), GenerationStats()

    async def complete(self, messages, *a, **kw):
        return (await self.complete_with_stats(messages))[0]

    def reply_budget(self, messages, max_tokens=None, quiet=False):
        return max_tokens or 512

    async def wait_until_ready(self, timeout=0.0):
        return True

    @staticmethod
    def classify_error(exc):
        return "stalled"


_AGENT_CFG = dict(agent_enabled=True, knowledge_triage=True, fast_path=False,
                  incremental_reasoning=False, chunk_large_prompts=False, agent_max_steps=3,
                  agent_min_steps=1, resilient_retries=2, ready_wait_timeout=0.01)
_LIST_CALL = '{"tool": "list_files", "args": {"path": "."}}'


def _run_agent(agent, turn, message="/agent list the files in my project"):
    async def drain():
        return [ev async for ev in agent.run(message, [], conversation_id=turn.conversation_id,
                                             max_tokens=256)]
    with _Bound(turn):
        return asyncio.run(drain())


class _Capture(logging.Handler):
    """Records from one package logger (configure_logging stops propagation,
    so pytest's caplog would not see them)."""

    def __init__(self, name):
        super().__init__(logging.DEBUG)
        self.logger = logging.getLogger(name)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)

    def __enter__(self):
        self.logger.addHandler(self)
        self._level = self.logger.level
        self.logger.setLevel(logging.DEBUG)
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self)
        self.logger.setLevel(self._level)


# --------------------------------------------------------------------------- #
# Config and preference schema
# --------------------------------------------------------------------------- #
def test_installation_default_is_ask_every_action():
    saved = os.environ.pop("APPROVAL_DEFAULT_MODE", None)
    try:
        assert Config().approval_default_mode == "all"
        os.environ["APPROVAL_DEFAULT_MODE"] = "Don't know"
        assert Config().approval_default_mode == "all", "a typo must fail towards asking"
        os.environ["APPROVAL_DEFAULT_MODE"] = "risky"
        assert Config().approval_default_mode == "risky"
        os.environ["APPROVAL_DEFAULT_MODE"] = "off"
        assert Config().approval_default_mode == "off"
    finally:
        os.environ.pop("APPROVAL_DEFAULT_MODE", None)
        if saved is not None:
            os.environ["APPROVAL_DEFAULT_MODE"] = saved
    cfg = Config(approval_default_mode="all")
    assert "approval_default_mode" in Config.MUTABLE
    cfg.apply({"approval_default_mode": "bogus"})
    assert cfg.approval_default_mode == "all"
    cfg.apply({"approval_default_mode": "changes"})
    assert cfg.approval_default_mode == "risky"
    # The administrator floor keeps its own default (off: users decide).
    assert cfg.tool_approval_mode == "off"


def test_the_preference_is_a_validated_user_pref():
    spec = USER_PREFS["approval_mode"]
    assert spec["options"] == ["all", "risky", "off"]
    assert spec["default"] is None, "None = follow APPROVAL_DEFAULT_MODE"
    assert validate_user_prefs({"approval_mode": "risky"}) == ({"approval_mode": "risky"}, [])
    assert validate_user_prefs({"approval_mode": None}) == ({"approval_mode": None}, [])
    assert validate_user_prefs({"approval_mode": "sometimes"}) == ({}, ["approval_mode"])
    assert validate_user_prefs({"approval_mode": True}) == ({}, ["approval_mode"])
    # Settings > Tools > Approvals shows the same preference to every role.
    field = next(f for f in SETTINGS_FIELDS if f["key"] == "approval_mode")
    assert field["scope"] == "user" and field["section"] == "tools"
    assert settings_field_sub(field) == "approvals"
    assert [o["value"] for o in field["options"]] == ["all", "risky", "off"]
    assert field.get("autosave") is True


# --------------------------------------------------------------------------- #
# The modes
# --------------------------------------------------------------------------- #
def test_a_default_user_is_asked_before_list_files():
    cfg, db, reg, calls = _registry()
    d = decide("list_files", {"path": "."}, _turn(), registry=reg)
    assert (d.action, d.risk, d.rule) == ("ask", "read", "user-mode:all"), d
    assert "Ask every action" in d.reason
    # The registry's own check fails closed: nothing ran without a yes.
    with _Bound(_turn()):
        out, error = reg.call("list_files", {"path": "."}, "c1")
    assert error and calls == []
    db.close()


def test_the_agent_shows_an_approval_card_for_list_files_with_all_scopes():
    """The interactive loop end to end: the card for a read tool, the three
    scopes (so a long turn needs one click, not one per read), and a "tool"
    answer that stops the asking for that tool."""
    from local_llm.agent import Agent
    cfg, db, reg, calls = _registry(**_AGENT_CFG)
    agent = Agent(cfg, reg, _Scripted([_LIST_CALL, "There are two entries."]))
    stop = threading.Event()
    answered = []

    def approver():
        while not stop.is_set():
            for item in BROKER.pending("u1"):
                answered.append(item)
                BROKER.resolve(item["id"], "u1", "allow", "tool")
            time.sleep(0.02)

    worker = threading.Thread(target=approver, daemon=True)
    worker.start()
    try:
        events = _run_agent(agent, _turn(conv="t", cancel=asyncio.Event()))
    finally:
        stop.set()
        worker.join(2)
    asked = [e for e in events if e.get("type") == "approval_required"]
    assert asked and asked[0]["name"] == "list_files", [e.get("type") for e in events]
    assert asked[0]["scopes"] == ["once", "conversation", "tool"]
    assert asked[0]["risk"] == "read"
    assert calls == [("list_files", {"path": "."})], calls
    # "Always allow this tool" wrote the user's own rule: no card next time.
    d = decide("list_files", {"path": "."}, _turn(conv="other"), registry=reg)
    assert d.action == "allow" and d.rule.startswith("user:"), d
    db.close()


def test_ask_for_changes_asks_for_write_file_but_not_read_file():
    cfg, db, reg, calls = _registry()
    _choose(db, "u1", "risky")
    turn = _turn()
    assert decide("read_file", {"path": "a"}, turn, registry=reg).action == "allow"
    assert decide("list_files", {}, turn, registry=reg).action == "allow"
    d = decide("write_file", {"path": "a", "content": "x"}, turn, registry=reg)
    assert (d.action, d.rule) == ("ask", "user-mode:risky"), d
    assert "Ask for changes" in d.reason
    assert decide("run_shell", {"command": "ls"}, turn, registry=reg).action == "ask"
    assert decide("o365_mail", {"action": "send"}, turn, registry=reg).action == "ask"
    with _Bound(turn):
        assert reg.call("read_file", {"path": "a"}, "c1") == ("file text", None)
    db.close()


def test_dont_ask_asks_for_nothing():
    cfg, db, reg, calls = _registry()
    _choose(db, "u1", "off")
    for interactive in (True, False):
        turn = _turn(interactive=interactive)
        for name in ("list_files", "read_file", "write_file", "run_shell"):
            assert decide(name, {"path": "a"}, turn, registry=reg).action == "allow", name
        with _Bound(turn):
            out, error = reg.call("write_file", {"path": "a.txt", "content": "x"}, "c1")
        assert error is None and out == "wrote it"
    assert [c[0] for c in calls] == ["write_file", "write_file"]
    db.close()


def test_a_turn_without_a_user_keeps_the_administrator_mode():
    """Internal helpers and eval sandboxes (no user on the turn, or no turn at
    all) have nobody whose choice could apply: they keep TOOL_APPROVAL_MODE."""
    cfg, db, reg, calls = _registry()
    for turn in (None, TurnContext(user_id=None, interactive=True)):
        d = decide("write_file", {"path": "a"}, turn, registry=reg)
        assert d.action == "allow" and d.rule == "default:off", d
    assert reg.call("list_files", {}, None) == ("README.md\nsrc/", None)
    db.close()


def test_a_reference_the_user_typed_is_not_asked_about_again():
    """@README.md is the user's own action; their mode does not ask them to
    confirm it. The administrator's floor still does."""
    cfg, db, reg, calls = _registry()
    with _Bound(_turn()):
        assert reference_refusal(reg, "README.md") is None
    cfg.apply({"tool_approval_mode": "all"})
    with _Bound(_turn()):
        assert "skipped @README.md" in (reference_refusal(reg, "README.md") or "")
    db.close()


# --------------------------------------------------------------------------- #
# The administrator floor and rules
# --------------------------------------------------------------------------- #
def test_a_stricter_administrator_floor_wins():
    cfg, db, reg, calls = _registry(tool_approval_mode="all")
    _choose(db, "u1", "off")
    d = decide("list_files", {}, _turn(), registry=reg)
    assert (d.action, d.rule) == ("ask", "default:all"), d
    state = approval_state(cfg, db, "u1")
    assert state["preference"] == "off" and state["chosen"] == "off"
    assert state["effective"] == "all" and state["locked"] is True
    assert state["source"] == "admin"

    cfg.apply({"tool_approval_mode": "risky"})
    assert decide("read_file", {}, _turn(), registry=reg).action == "allow"
    assert decide("write_file", {}, _turn(), registry=reg).rule == "default:risky"
    # The user asks for more than the floor: the user's mode wins.
    _choose(db, "u1", "all")
    d = decide("read_file", {}, _turn(), registry=reg)
    assert (d.action, d.rule) == ("ask", "user-mode:all"), d
    state = approval_state(cfg, db, "u1")
    assert state["effective"] == "all" and state["locked"] is False
    db.close()


def test_administrator_deny_rules_stay_final():
    cfg, db, reg, calls = _registry()
    _choose(db, "u1", "off")
    set_tool_override(db, "global", "", "write_file", "deny")
    d = decide("write_file", {"path": "a"}, _turn(), registry=reg)
    assert d.action == "deny" and d.rule.startswith("global:"), d
    # A user's own "allow" cannot loosen it either.
    set_tool_override(db, "user", "u1", "write_file", "allow")
    assert decide("write_file", {"path": "a"}, _turn(), registry=reg).action == "deny"
    db.close()


# --------------------------------------------------------------------------- #
# Unattended runs
# --------------------------------------------------------------------------- #
def test_unattended_runs_refuse_with_the_header_hint_and_log_it():
    cfg, db, reg, calls = _registry()
    turn = _turn(interactive=False, extras={"source": "mcp"})
    with _Capture("local_llm.tool") as cap, _Bound(turn):
        out, error = reg.call("list_files", {"path": "."}, None)
    assert error and APPROVAL_UNATTENDED_HINT in out, out
    assert "approval needed: switch Approvals to Ask for changes or Don't ask in the header" in out
    assert calls == []
    logged = [r for r in cap.records if getattr(r, "event", "") == "tool.approval_needed"]
    assert logged and logged[0].hint == APPROVAL_UNATTENDED_HINT, cap.records
    assert logged[0].source == "mcp" and logged[0].tool == "list_files"

    # Ask for changes: a read runs, a write names the one switch that helps.
    _choose(db, "u1", "risky")
    with _Bound(_turn(interactive=False)):
        assert reg.call("list_files", {"path": "."}, None) == ("README.md\nsrc/", None)
        out, error = reg.call("write_file", {"path": "a", "content": "x"}, None)
    assert error and APPROVAL_UNATTENDED_HINT_CHANGES in out, out

    # The floor asks: the header cannot help, so the hint says who can.
    cfg.apply({"tool_approval_mode": "all"})
    _choose(db, "u1", "off")
    with _Bound(_turn(interactive=False)):
        out, error = reg.call("list_files", {"path": "."}, None)
    assert error and APPROVAL_UNATTENDED_HINT_ADMIN in out and "header" not in out, out
    db.close()


def test_a_scheduled_task_run_reports_the_hint_in_its_tool_result():
    """A task run is an agent turn with interactive=False (tasks.py): the
    refusal is the tool result, so the run's trace says what to change."""
    from local_llm.agent import Agent
    cfg, db, reg, calls = _registry(**_AGENT_CFG)
    agent = Agent(cfg, reg, _Scripted([_LIST_CALL, "Could not list the files."]))
    turn = TurnContext(turn_id="task-run1", user_id="u1", conversation_id="task-t",
                       interactive=False, cancel=asyncio.Event())
    events = _run_agent(agent, turn)
    assert not [e for e in events if e.get("type") == "approval_required"]
    results = [e for e in events if e.get("type") == "tool_result"]
    assert results, [e.get("type") for e in events]
    text = str(results[0].get("error") or results[0].get("result") or "")
    assert APPROVAL_UNATTENDED_HINT in text, results[0]
    assert calls == []
    db.close()


# --------------------------------------------------------------------------- #
# Persistence, isolation and the routes
# --------------------------------------------------------------------------- #
def test_the_choice_persists_per_user_and_stays_isolated():
    path = Path(tempfile.mkdtemp()) / "persist.db"
    db = Database(path)
    _choose(db, "alice", "off")
    _choose(db, "bob", "risky")
    db.close()
    db = Database(path)   # a restart: the choice is in the database, not a session
    assert user_approval_choice(db, "alice") == "off"
    assert user_approval_choice(db, "bob") == "risky"
    assert user_approval_choice(db, "carol") is None
    cfg = Config(approval_default_mode="all")
    assert approval_state(cfg, db, "carol")["effective"] == "all"
    for user, action in (("alice", "allow"), ("bob", "allow"), ("carol", "ask")):
        d = decide("list_files", {}, _turn(user=user), config=cfg, db=db)
        assert d.action == action, (user, d)
    # Resetting to null follows the installation default again.
    PreferenceStore(db).update("alice", {"approval_mode": None})
    assert user_approval_choice(db, "alice") is None
    assert user_approval_choice(db, "bob") == "risky"
    db.close()


def _build_app(**overrides):
    from local_llm.api import create_app
    from local_llm.core import ADAPTER_DIR
    from local_llm.model_server import ModelServerManager
    from local_llm.training import RetrainManager
    base = dict(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                allow_test_user=False, node_token="secret-node-token",
                skills_dir=tempfile.mkdtemp(prefix="llm-test-appr-skills-"),
                project_dir=tempfile.mkdtemp(prefix="llm-test-appr-proj-"),
                approval_default_mode="all")
    base.update(overrides)
    cfg = Config(**base)
    db = Database(Path(tempfile.mkdtemp()) / "app.db")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)  # never started
    app = create_app(cfg, db, mm, RetrainManager(db, mm, cfg), ToolRegistry(cfg, db))
    app.state.model_manager = mm
    return app, cfg, db


def _login(client, username, password):
    r = client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text


def test_the_routes_change_one_users_mode_only():
    from fastapi.testclient import TestClient
    app, cfg, db = _build_app()
    with TestClient(app) as admin:
        _login(admin, "admin", "adminpw123")
        assert admin.post("/api/users", json={"username": "bob", "password": "bobpass123",
                                              "role": "user"}).status_code == 200
        with TestClient(app) as bob:
            _login(bob, "bob", "bobpass123")
            for client in (admin, bob):
                state = client.get("/api/me/approval-mode").json()
                assert state["preference"] is None and state["effective"] == "all"
                assert state["label"] == "Ask every action"
                assert [m["value"] for m in state["modes"]] == ["all", "risky", "off"]
            r = bob.post("/api/me/preferences", json={"approval_mode": "off"})
            assert r.status_code == 200 and r.json()["preferences"]["approval_mode"] == "off"
            assert bob.get("/api/me/approval-mode").json()["effective"] == "off"
            assert admin.get("/api/me/approval-mode").json()["effective"] == "all"
            bad = bob.post("/api/me/preferences", json={"approval_mode": "sometimes"}).json()
            assert bad["ignored"] == ["approval_mode"]
            assert bob.get("/api/me/approval-mode").json()["preference"] == "off"
            # The administrator's floor locks bob's looser choice.
            assert admin.post("/api/config", json={"tool_approval_mode": "risky"}).status_code == 200
            state = bob.get("/api/me/approval-mode").json()
            assert state["locked"] is True and state["effective"] == "risky"
            assert state["admin_floor"] == "risky" and state["chosen"] == "off"
            policies = bob.get("/api/tool-policies").json()
            assert policies["mode"] == "risky" and policies["effective_mode"] == "risky"
            assert policies["user_mode"] == "off"
            # Back to the installation default, which the administrator moved.
            assert admin.post("/api/config", json={"tool_approval_mode": "off",
                                                   "approval_default_mode": "risky"}
                              ).status_code == 200
            bob.post("/api/me/preferences", json={"approval_mode": None})
            state = bob.get("/api/me/approval-mode").json()
            assert state["preference"] is None and state["effective"] == "risky"
            assert state["source"] == "default"
    with TestClient(app) as anon:
        assert anon.get("/api/me/approval-mode").status_code == 401
    db.close()


def test_the_mcp_server_refuses_with_the_hint_until_the_user_switches():
    from fastapi.testclient import TestClient
    app, cfg, db = _build_app(mcp_server_enabled=True)

    def rpc(method, mid, **params):
        return {"jsonrpc": "2.0", "id": mid, "method": method, "params": params}

    with TestClient(app) as c:
        _login(c, "admin", "adminpw123")
        r = c.post("/mcp", json=rpc("initialize", 1, protocolVersion="2025-06-18",
                                    capabilities={}, clientInfo={"name": "t"}))
        assert r.status_code == 200, r.text
        h = {"Mcp-Session-Id": r.headers["mcp-session-id"],
             "MCP-Protocol-Version": "2025-06-18"}
        call = rpc("tools/call", 2, name="calculator", arguments={"expression": "2+2"})
        out = c.post("/mcp", json=call, headers=h).json()["result"]
        assert out["isError"] is True and APPROVAL_UNATTENDED_HINT in out["content"][0]["text"]
        assert c.post("/api/me/preferences", json={"approval_mode": "off"}).status_code == 200
        out = c.post("/mcp", json=call, headers=h).json()["result"]
        assert out["isError"] is False and out["content"][0]["text"] == "4"
    db.close()


# --------------------------------------------------------------------------- #
# The header control
# --------------------------------------------------------------------------- #
def _header(page: str) -> str:
    match = re.search(r"<header>(.*?)</header>", page, re.S)
    assert match, "no <header>"
    return match.group(1)


def test_the_header_control_renders_for_every_role_outside_settings():
    from fastapi.testclient import TestClient
    from local_llm.ui import render_ui
    page = render_ui()
    header = _header(page)
    for element in ("apprMenu", "apprBtn", "apprPop", "apprLabel", "apprShort",
                    "apprOptAll", "apprOptRisky", "apprOptOff", "apprLockNote"):
        assert page.count(f'id="{element}"') == 1, element
        assert f'id="{element}"' in header, f"{element} is not in the header"
    # Not role-gated and not part of the Settings view.
    block = header[header.index('id="apprMenu"'):header.index("<!-- approvals end -->")]
    assert "admin-only" not in block and "hidden" not in block.split("apprPop")[0]
    settings = page[page.index('id="settingsView"'):]
    assert 'id="apprBtn"' not in settings
    for label in ("Ask every action", "Ask for changes", "Don't ask"):
        assert label in block, label
    # The same page goes to an administrator and to a user.
    app, cfg, db = _build_app()
    with TestClient(app) as admin:
        _login(admin, "admin", "adminpw123")
        admin.post("/api/users", json={"username": "bob", "password": "bobpass123",
                                       "role": "user"})
        with TestClient(app) as bob:
            _login(bob, "bob", "bobpass123")
            for client in (admin, bob):
                html = client.get("/").text
                assert 'id="apprBtn"' in _header(html)
    db.close()


def test_the_header_control_script_and_styles_hold_up():
    from local_llm.ui import render_ui
    from local_llm.ui_script_settings import UI_JS_SETTINGS
    from local_llm.ui_styles import UI_CSS
    page = render_ui()
    script = re.search(r"<script>(.*?)</script>", page, re.S).group(1)
    block = UI_JS_SETTINGS[UI_JS_SETTINGS.index("approvals (header)"):]
    defined = set(re.findall(r'\bid="([A-Za-z0-9_]+)"', page))
    refs = set(re.findall(r"getElementById\(['\"]([A-Za-z0-9_]+)['\"]\)", block))
    assert {"apprBtn", "apprPop", "apprLabel", "prefApprovalMode"} <= refs
    assert not (refs - defined), sorted(refs - defined)
    assert not re.search(r"getElementById\([^'\"]", block), "computed ids cannot be checked"
    for fn in ("apprToggle", "apprChoose", "apprOpenSettings", "apprLoad", "apprRender",
               "apprAdoptPrefs", "apprWatchSession"):
        assert re.search(r"function\s+%s\s*\(" % fn, script), fn
    assert "/api/me/approval-mode" in block and '"/api/me/preferences"' in block
    assert "apprWatchSession();" in script
    # The Settings copy of the preference is bound and visible to every role.
    field = re.search(r'<div class="([^"]*)" data-field="approval_mode" data-scope="user">', page)
    assert field and "admin-only" not in field.group(1)
    assert 'id="prefApprovalMode"' in page and 'data-autosave="1"' in page
    # No rule hides the control itself (or the header actions) at any width;
    # narrow widths only drop its words.
    never_hidden = {".appr", ".appr-btn", "#apprMenu", "#apprBtn", ".appr-icon",
                    "header .actions", "body > header", ".actions"}
    for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", UI_CSS):
        if not re.search(r"display\s*:\s*none", body):
            continue
        for selector in selectors.split(","):
            selector = selector.strip()
            if selector.startswith(("body.locked", "body.booting")):
                continue
            assert selector not in never_hidden, f"{selector} is hidden somewhere"
    assert chr(0x2014) not in block and chr(0x2014) not in _header(page)


def _run_standalone():
    os.environ.setdefault("APPROVAL_DEFAULT_MODE", "off")
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"FAIL  {name}: {exc}")
            failed += 1
        finally:
            teardown_function()
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
