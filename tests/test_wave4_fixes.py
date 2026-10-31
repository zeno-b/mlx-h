"""Regression tests for the Wave 4b fixes: the open items earlier integrators
reported (cloud preflight before chunking, training paths under a patched
DATA_DIR, folder-free project views, server-side audit filters, the question
stored while queued, the dead admin tab, the acting-user leak in one test, the
restart flag on /api/config, per-user and per-conversation cleanup, and the
admission wait reaching usage events)."""
from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import local_llm.config as config_mod  # noqa: E402

config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-wave4-fixes-"))

from fastapi.testclient import TestClient  # noqa: E402

import local_llm.agent as agent_mod  # noqa: E402
import local_llm.tools as tools_mod  # noqa: E402
import local_llm.workspaces as ws_mod  # noqa: E402
from local_llm.agent import Agent  # noqa: E402
from local_llm.api import create_app  # noqa: E402
from local_llm.api_tokens import create_api_token, list_api_tokens  # noqa: E402
from local_llm.attachments import AttachmentStore  # noqa: E402
from local_llm.auth import hash_password  # noqa: E402
from local_llm.config import Config  # noqa: E402
from local_llm.core import ADAPTER_DIR, SFT_DIR, REPLAY_FILE, estimate_tokens  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.llm import GenerationStats  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.policy import list_tool_decisions, record_tool_decision  # noqa: E402
from local_llm.prefs import PreferenceStore  # noqa: E402
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.train_methods import train_replay_file, train_sft_dir  # noqa: E402
from local_llm.training import RetrainManager  # noqa: E402
from local_llm.turnctx import (  # noqa: E402
    TurnContext, current_turn, reset_current_turn, set_current_turn,
)

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Construction (copied from tests/test_app.py, never imported from it)
# --------------------------------------------------------------------------- #
def build_app(manager_kw: dict | None = None, **overrides):
    base = dict(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                allow_test_user=False, node_token="secret-node-token",
                skills_dir=tempfile.mkdtemp(prefix="local-llm-test-skills-"))
    base.update(overrides)
    cfg = Config(**base)
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR, **(manager_kw or {}))  # never started
    rm = RetrainManager(db, mm, cfg)
    app = create_app(cfg, db, mm, rm, ToolRegistry(cfg, db))
    app.state.model_manager = mm
    return app, cfg, db


def login(client, username, password):
    return client.post("/api/auth/login", json={"username": username, "password": password})


def _user_client(app, db, username):
    user = db.create_user(username, password_hash=hash_password(f"{username}pw123"), role="user")
    client = TestClient(app)
    client.__enter__()
    assert login(client, username, f"{username}pw123").status_code == 200
    return client, user


def _sse_events(text: str) -> list:
    out = []
    for line in text.splitlines():
        if line.startswith("data: "):
            body = line[len("data: "):]
            out.append(body if body == "[DONE]" else json.loads(body))
    return out


def _final(answer: str = "ok") -> dict:
    return {"type": "final", "answer": answer, "steps": 1, "trace": [],
            "tools_used": [], "elapsed_ms": 1, "prompt_tokens": 1,
            "completion_tokens": 1, "truncated": False, "changed_files": [], "diff": ""}


class _ScriptedClient:
    """A ModelClient stand-in that replays canned replies."""

    def __init__(self, replies=()):
        self.replies = list(replies)
        self.prompts: list = []

    def _next(self):
        return self.replies.pop(0) if self.replies else "done."

    def stream(self, messages, *a, **kw):
        self.prompts.append(messages)
        text = self._next()

        async def gen():
            await asyncio.sleep(0)
            yield text

        return gen()

    async def complete_with_stats(self, messages, *a, **kw):
        self.prompts.append(messages)
        text = self._next()
        stats = GenerationStats()
        stats.prompt_tokens = sum(estimate_tokens(str(m.get("content") or "")) for m in messages)
        stats.completion_tokens = estimate_tokens(text)
        return text, stats

    async def complete(self, messages, *a, **kw):
        text, _ = await self.complete_with_stats(messages, *a, **kw)
        return text

    def reply_budget(self, messages, max_tokens=None, quiet=False):
        return max_tokens or 512

    async def wait_until_ready(self, timeout=0.0):
        return True

    @staticmethod
    def classify_error(exc):
        return "the model stalled"


def _agent(**overrides):
    base = dict(agent_enabled=True, knowledge_triage=False, fast_path=False,
                incremental_reasoning=False, chunk_large_prompts=False,
                agent_max_steps=3, agent_min_steps=1, resilient_retries=1,
                ready_wait_timeout=0.01, auto_iterate_rounds=0,
                skills_dir=tempfile.mkdtemp(prefix="llm-test-wave4-skills-"))
    base.update(overrides)
    cfg = Config(**base)
    db = Database(Path(tempfile.mkdtemp()) / "agent.db")
    reg = ToolRegistry(cfg, db)
    client = _ScriptedClient()
    return Agent(cfg, reg, client), cfg, db, client


def _run_agent(agent, message, turn=None):
    async def main():
        token = set_current_turn(turn) if turn is not None else None
        try:
            return [ev async for ev in agent.run(message, [], conversation_id="t",
                                                 max_tokens=256)]
        finally:
            if token is not None:
                reset_current_turn(token)
    return asyncio.run(main())


# --------------------------------------------------------------------------- #
# 1. Cloud fallback is asked before an oversized prompt is split
# --------------------------------------------------------------------------- #
def _chunk_phase(events) -> bool:
    return any(e.get("type") == "phase" and "reading it in" in str(e.get("label"))
               for e in events)


def test_cloud_preflight_keeps_an_oversized_prompt_whole_when_the_cloud_takes_it():
    big = "Summarise the following log. " + ("disk sector 42 read error; " * 900)
    calls = []

    async def yes(config, messages, max_tokens=None, kind="agent", conversation_id=None):
        calls.append((kind, conversation_id, messages[-1]["content"][:20]))
        return True

    real = agent_mod.cloud_preflight
    agent_mod.cloud_preflight = yes
    try:
        agent, *_ = _agent(chunk_large_prompts=True, context_size=4096)
        events = _run_agent(agent, big)
    finally:
        agent_mod.cloud_preflight = real
    assert calls and calls[0][0] == "agent" and calls[0][1] == "t", calls
    assert not _chunk_phase(events), [e for e in events if e.get("type") == "phase"]


def test_cloud_off_still_splits_an_oversized_prompt():
    big = "Summarise the following log. " + ("disk sector 42 read error; " * 900)
    agent, cfg, *_ = _agent(chunk_large_prompts=True, context_size=4096)
    assert not getattr(cfg, "cloud_enabled", False)
    events = _run_agent(agent, big)       # the real preflight: feature off, False
    assert _chunk_phase(events)


def test_a_small_prompt_never_asks_the_cloud_preflight():
    calls = []

    async def spy(*a, **kw):
        calls.append(a)
        return False

    real = agent_mod.cloud_preflight
    agent_mod.cloud_preflight = spy
    try:
        agent, *_ = _agent(chunk_large_prompts=True, context_size=4096)
        _run_agent(agent, "explain how tcp handshakes work")
    finally:
        agent_mod.cloud_preflight = real
    assert calls == []


# --------------------------------------------------------------------------- #
# 2. Training export follows a DATA_DIR patched on the config module
# --------------------------------------------------------------------------- #
def test_train_paths_follow_a_redirected_data_dir():
    saved = config_mod.DATA_DIR
    tmp = Path(tempfile.mkdtemp(prefix="llm-test-wave4-data-"))
    config_mod.DATA_DIR = tmp
    repo_train = REPO / "data" / "sft" / "train.jsonl"
    before = repo_train.stat().st_mtime_ns if repo_train.exists() else None
    try:
        # The star-imported constants still name the repo; the helpers do not.
        assert train_sft_dir(SFT_DIR) == tmp / "sft"
        assert train_replay_file(REPLAY_FILE) == tmp / "sft" / "replay.jsonl"
        # A sandbox that redirects SFT_DIR itself still wins.
        own = Path(tempfile.mkdtemp()) / "sft"
        assert train_sft_dir(own) == own
        cfg = Config(train_replay_ratio=0, train_min_examples=1,
                     skills_dir=tempfile.mkdtemp(prefix="local-llm-test-skills-"))
        db = Database(Path(tempfile.mkdtemp()) / "train.db")
        for i in range(4):
            db.execute("INSERT INTO feedback (user_prompt, assistant_response, rating, "
                       "approved_for_training) VALUES (?, ?, 1, 1)",
                       (f"question {i}?", f"a short but complete answer ({i})."))
        db.commit()
        rm = RetrainManager(db, ModelServerManager(cfg.model, 8090, ADAPTER_DIR), cfg)
        count, _ids = rm.export_feedback()
        assert count >= 1
        assert (tmp / "sft" / "train.jsonl").exists()
        after = repo_train.stat().st_mtime_ns if repo_train.exists() else None
        assert after == before, "the export wrote into the repo's data/sft"
        db.close()
    finally:
        config_mod.DATA_DIR = saved


# --------------------------------------------------------------------------- #
# 3. Viewing projects creates no folder; creating one does
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _workspace_dir():
    tmp = Path(tempfile.mkdtemp(prefix="llm-ws-root-")).resolve()
    saved = (tools_mod.WORKSPACE_DIR, ws_mod.WORKSPACE_DIR)
    tools_mod.WORKSPACE_DIR = tmp
    ws_mod.WORKSPACE_DIR = tmp
    try:
        yield tmp
    finally:
        tools_mod.WORKSPACE_DIR, ws_mod.WORKSPACE_DIR = saved


def test_viewing_projects_creates_no_user_folder():
    with _workspace_dir() as root:
        app, cfg, db = build_app(workspaces_enabled=True)
        with TestClient(app):
            client, user = _user_client(app, db, "wanda")
            try:
                r = client.get("/api/projects")
                assert r.status_code == 200, r.text
                home = root / "users" / user["id"]
                assert r.json()["default_root"], r.json()
                assert not home.exists(), "a GET created the user's workspace folder"
                r = client.post("/api/projects", json={"name": "notes"})
                assert r.status_code in (200, 201), r.text
                assert home.is_dir()
            finally:
                client.__exit__(None, None, None)


# --------------------------------------------------------------------------- #
# 4. Audit filters run on the server, with paging
# --------------------------------------------------------------------------- #
def test_audit_route_filters_server_side_and_pages():
    app, cfg, db = build_app()
    with TestClient(app) as admin:
        assert login(admin, "admin", "adminpw123").status_code == 200
        alice = db.create_user("auditalice", password_hash=hash_password("alicepw123"))
        # One old row, to filter by date (the table is append-only: insert it
        # with its timestamp rather than backdating a row).
        db.ensure_schema("policy")
        db.execute("INSERT INTO tool_decisions (created_at, user_id, tool, args_redacted, "
                   "risk, decision, rule) VALUES ('2020-01-02T10:00:00+00:00', ?, "
                   "'run_shell', '{}', 'high', 'deny', '')", (alice["id"],))
        db.commit()
        for i in range(4):
            record_tool_decision(db, tool="run_shell", decision="deny", user_id=alice["id"])
        for i in range(3):
            record_tool_decision(db, tool="read_file", decision="allow", user_id=alice["id"])
        record_tool_decision(db, tool="read_file", decision="allow", user_id="someone-else")

        get = lambda q: admin.get("/api/audit/tool-decisions?" + q)  # noqa: E731
        r = get(f"decision=deny&user_id={alice['id']}")
        assert r.status_code == 200, r.text
        rows = r.json()["decisions"]
        assert len(rows) == 5 and {x["decision"] for x in rows} == {"deny"}
        assert len(get("tool=read_file").json()["decisions"]) == 4
        assert len(get("since=2021-01-01").json()["decisions"]) == 8
        assert len(get("until=2020-01-02").json()["decisions"]) == 1
        assert len(get("since=2020-01-02T09:00:00Z&until=2020-01-02T11:00:00Z")
                   .json()["decisions"]) == 1
        assert get("since=yesterday").status_code == 400
        # Paging: has_more and the keyset cursor walk the whole set once.
        page = get("limit=4").json()
        assert len(page["decisions"]) == 4 and page["has_more"] is True
        seen = [x["id"] for x in page["decisions"]]
        while page["has_more"]:
            page = get(f"limit=4&before_id={page['next_before_id']}").json()
            seen += [x["id"] for x in page["decisions"]]
        assert len(seen) == len(set(seen)) == 9
        assert len(get("limit=4&offset=8").json()["decisions"]) == 1
        # A user only ever sees their own rows, whatever user_id they send.
        client = TestClient(app)
        with client:
            assert login(client, "auditalice", "alicepw123").status_code == 200
            mine = client.get("/api/audit/tool-decisions?user_id=someone-else").json()
            assert len(mine["decisions"]) == 8
            assert {x["user_id"] for x in mine["decisions"]} == {alice["id"]}
    assert len(list_tool_decisions(db, decision="allow")) == 4


def test_audit_ui_sends_its_filters_to_the_server():
    from local_llm.ui_script_integrations import UI_JS_INTEGRATIONS as js
    for param in ("&decision=", "&since=", "&until=", "&user_id=", "&tool=", "&before_id="):
        assert param in js, param


# --------------------------------------------------------------------------- #
# 5 + 10. A queued turn stores its question first, and carries its wait
# --------------------------------------------------------------------------- #
def test_queued_question_is_stored_before_admission_and_queue_ms_reaches_the_turn():
    app, cfg, db = build_app(max_concurrent_generations=1, generation_queue_depth=4)
    app.state.model_manager.status = "ready"
    seen: dict = {}

    async def fake_run(message, history, conversation_id=None, max_tokens=None,
                       temperature=None, cancel=None, **kw):
        turn = current_turn()
        seen["queue_ms"] = (turn.extras.get("queue_ms") if turn is not None else None)
        yield {"type": "token", "text": "o"}
        yield _final()

    app.state.agent.run_iterating = fake_run
    gate = app.state.ctx.generation_gate
    question = "what does the queued question look like after a reload"
    stored_while_queued = threading.Event()

    def count_questions():
        return db.execute("SELECT COUNT(*) AS n FROM messages WHERE role = 'user' "
                          "AND content = ?", (question,)).fetchone()["n"]

    with TestClient(app) as c:
        assert login(c, "admin", "adminpw123").status_code == 200
        held = c.portal.call(gate.acquire)

        def free_later():
            for _ in range(300):
                if c.portal.call(lambda: gate.waiting) >= 1:
                    break
                time.sleep(0.01)
            # The stream is waiting for a slot now: the question must already
            # be in the transcript, where a reload would find it.
            for _ in range(300):
                if count_questions() == 1:
                    stored_while_queued.set()
                    break
                time.sleep(0.01)
            time.sleep(0.05)
            c.portal.call(gate.release, held)

        threading.Thread(target=free_later, daemon=True).start()
        r = c.post("/api/chat/stream", json={"message": question, "agent": True,
                                             "conversation_id": "queued-conv"})
        kinds = [e["type"] if isinstance(e, dict) else e for e in _sse_events(r.text)]
        assert kinds[0] == "queued" and "start" in kinds and "final" in kinds, kinds
    assert stored_while_queued.is_set(), "the question was not stored while queued"
    assert count_questions() == 1, "the question was stored twice"
    assert seen.get("queue_ms") is not None and seen["queue_ms"] > 0, seen


def test_an_immediately_admitted_turn_stores_its_question_once():
    app, cfg, db = build_app()
    app.state.model_manager.status = "ready"

    async def fake_run(message, history, conversation_id=None, max_tokens=None,
                       temperature=None, cancel=None, **kw):
        yield _final()

    app.state.agent.run_iterating = fake_run
    with TestClient(app) as c:
        assert login(c, "admin", "adminpw123").status_code == 200
        r = c.post("/api/chat/stream", json={"message": "hello once", "agent": True})
        assert "queued" not in r.text
    n = db.execute("SELECT COUNT(*) AS n FROM messages WHERE role = 'user' "
                   "AND content = 'hello once'").fetchone()["n"]
    assert n == 1


def test_usage_events_carry_the_turns_queue_wait():
    agent, *_ = _agent()
    turn = TurnContext(user_id="u1", conversation_id="t", interactive=False)
    turn.extras["queue_ms"] = 123.4
    usage = [e for e in _run_agent(agent, "explain how tcp handshakes work", turn)
             if e.get("type") == "usage"]
    assert usage and all(e.get("queue_ms") == 123.4 for e in usage), usage
    # Without a turn (CLI, selftest) the value stays 0, as before.
    usage = [e for e in _run_agent(agent, "explain how tcp handshakes work")
             if e.get("type") == "usage"]
    assert usage and all(not e.get("queue_ms") for e in usage), usage


# --------------------------------------------------------------------------- #
# 6. No dead "tools" admin tab
# --------------------------------------------------------------------------- #
def test_admin_tabs_have_no_dead_tools_entry():
    from local_llm.ui_script_admin import UI_JS_ADMIN as js
    assert 'key: "tools"' not in js
    assert "loadToolsTab" not in js


# --------------------------------------------------------------------------- #
# 7. The recall test unbinds the acting user it set
# --------------------------------------------------------------------------- #
def test_recall_test_resets_the_acting_user_in_a_finally():
    tree = ast.parse((REPO / "tests" / "test_app.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == "test_cross_session_recall_is_a_tool_and_is_per_user")
    resets = [call for node in ast.walk(fn) if isinstance(node, ast.Try)
              for stmt in node.finalbody for call in ast.walk(stmt)
              if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "set_acting_user"
              and call.args and isinstance(call.args[0], ast.Constant)
              and call.args[0].value is None]
    assert resets, "set_acting_user(None) must run in a finally"


# --------------------------------------------------------------------------- #
# 8. /api/config says when the model server must restart
# --------------------------------------------------------------------------- #
def test_set_config_reports_restart_required():
    app, cfg, db = build_app(
        model_backend="mlx-lm-turbo", max_concurrent_generations=2,
        manager_kw={"backend": "mlx-lm-turbo", "max_concurrent_generations": 2})
    mm = app.state.model_manager
    mm.context_size = cfg.context_size
    with TestClient(app) as c:
        assert login(c, "admin", "adminpw123").status_code == 200
        r = c.post("/api/config", json={"max_tokens": 700})
        assert r.status_code == 200 and r.json()["restart_required"] is False, r.text
        # turbo sizes its command by the concurrency: a restart is due.
        r = c.post("/api/config", json={"max_concurrent_generations": 3})
        assert r.json()["restart_required"] is True, r.text
        assert mm.max_concurrent_generations == 3
        # A start holding the manager's lock must not hold the request hostage.
        locked, release = threading.Event(), threading.Event()

        def hold():
            with mm.lock:
                locked.set()
                release.wait(10)

        threading.Thread(target=hold, daemon=True).start()
        assert locked.wait(5)
        wanted = cfg.context_size + 1024
        t0 = time.monotonic()
        r = c.post("/api/config", json={"context_size": wanted})
        assert time.monotonic() - t0 < 3, "set_config blocked on the start lock"
        assert r.status_code == 200 and r.json()["restart_required"] is True, r.text
        assert mm.context_size != wanted          # not adopted under a running start
        release.set()
        for _ in range(300):
            if mm.context_size == wanted:
                break
            time.sleep(0.01)
        assert mm.context_size == wanted          # adopted once the start let go


def test_set_config_default_backend_needs_no_restart_for_context_size():
    app, cfg, db = build_app()
    with TestClient(app) as c:
        assert login(c, "admin", "adminpw123").status_code == 200
        r = c.post("/api/config", json={"context_size": cfg.context_size + 512})
        assert r.status_code == 200 and r.json()["restart_required"] is False, r.text


# --------------------------------------------------------------------------- #
# 9. Deleting a user or a conversation leaves nothing of theirs behind
# --------------------------------------------------------------------------- #
def test_deleting_a_user_removes_prefs_attachments_and_tokens():
    app, cfg, db = build_app()
    with TestClient(app) as admin:
        assert login(admin, "admin", "adminpw123").status_code == 200
        user = db.create_user("leaver", password_hash=hash_password("leaverpw123"))
        uid = user["id"]
        PreferenceStore(db).update(uid, {"theme": "dark"})
        create_api_token(db, user, name="laptop")
        store = AttachmentStore(db, cfg)
        row = store.save(uid, b"some notes\n", "notes.txt", conversation_id="c-leaver")
        path = store.path_of(row)
        assert path.exists() and list_api_tokens(db, uid)
        record_tool_decision(db, tool="read_file", decision="allow", user_id=uid)

        r = admin.delete(f"/api/users/{uid}")
        assert r.status_code == 200 and r.json()["deleted"] is True, r.text
    assert PreferenceStore(db).stored(uid) == {}
    assert db.execute("SELECT COUNT(*) AS n FROM user_preferences WHERE user_id = ?",
                      (uid,)).fetchone()["n"] == 0
    assert db.execute("SELECT COUNT(*) AS n FROM api_tokens WHERE user_id = ?",
                      (uid,)).fetchone()["n"] == 0
    assert store.list(uid) == [] and not path.exists()
    # The audit trail is kept on purpose.
    assert len(list_tool_decisions(db, user_id=uid)) == 1


def test_a_failing_cleanup_hook_does_not_stop_the_others():
    from local_llm.database import USER_CLEANUP_HOOKS, register_user_cleanup
    db = Database(Path(tempfile.mkdtemp()) / "hooks.db")
    user = db.create_user("hooked")
    ran = []

    def boom(_db, _uid):
        raise RuntimeError("disk on fire")

    register_user_cleanup("zz-test-boom", boom)
    register_user_cleanup("zz-test-ok", lambda _db, uid: ran.append(uid))
    try:
        assert db.delete_user(user["id"]) is True
        assert ran == [user["id"]]
        # An unknown user runs nothing.
        assert db.delete_user("no-such-user") is False and ran == [user["id"]]
    finally:
        USER_CLEANUP_HOOKS.pop("zz-test-boom", None)
        USER_CLEANUP_HOOKS.pop("zz-test-ok", None)


def test_deleting_a_conversation_removes_its_attachments():
    app, cfg, db = build_app()
    with TestClient(app):
        client, user = _user_client(app, db, "carla")
        try:
            uid = user["id"]
            db.add_message("conv-a", "user", "hello", user_id=uid)
            store = AttachmentStore(db, cfg)
            gone = store.save(uid, b"first\n", "a.txt", conversation_id="conv-a")
            kept = store.save(uid, b"second\n", "b.txt", conversation_id="conv-b")
            r = client.delete("/api/conversation/conv-a")
            assert r.status_code == 200, r.text
            assert store.get(gone["id"], uid) is None
            assert not store.path_of(gone).exists()
            assert store.get(kept["id"], uid) is not None and store.path_of(kept).exists()
        finally:
            client.__exit__(None, None, None)


def _run_standalone():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            import traceback; traceback.print_exc(); print(f"FAIL  {name}: {exc}"); failed += 1
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
