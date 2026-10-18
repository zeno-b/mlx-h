"""Wave 3a end-to-end: the real agent loop behind the real chat routes.

The feature tests each fake the other side of their contract (a fake policy
for the agent, a fake agent for the routes). These drive the whole path with
only the model faked: POST /api/chat/stream runs the real Agent, whose
_dispatch consults the real policy and BROKER; a second client finds the
pending approval and answers it through POST /api/chat/turns/{id}/approvals/
{aid}; the waiting turn resumes (or not) and the stream ends with a final.
The TestClient reads a streaming response to the end, so the approving
"second client" runs on another thread while the first request is in flight.
"""
from __future__ import annotations

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
config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-wave3-integ-"))

from fastapi.testclient import TestClient  # noqa: E402

import local_llm.tools as tools_mod  # noqa: E402
import local_llm.workspaces as ws_mod  # noqa: E402
from local_llm.api import create_app  # noqa: E402
from local_llm.config import Config  # noqa: E402
from local_llm.core import ADAPTER_DIR, estimate_tokens  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.llm import GenerationStats  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.policy import BROKER, list_tool_decisions  # noqa: E402
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.training import RetrainManager  # noqa: E402
from local_llm.turnctx import current_turn  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _ScriptedModel:
    """The agent's ModelClient stand-in: streamed replies come from a script.

    Every prompt is kept, so a test can check what the model was told (a
    denial, a project's AGENTS.md). Unscripted calls answer "done." so a
    background helper that asks the model something never steals a reply.
    """

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[list[dict]] = []

    def stream(self, messages, *a, **kw):
        self.prompts.append(messages)
        reply = self.replies.pop(0) if self.replies else "done."

        async def gen():
            for piece in ([reply] if isinstance(reply, str) else reply):
                await asyncio.sleep(0)
                yield piece

        return gen()

    async def complete_with_stats(self, messages, *a, **kw):
        stats = GenerationStats()
        stats.prompt_tokens = sum(estimate_tokens(str(m.get("content"))) for m in messages)
        stats.completion_tokens = 1
        return "done.", stats

    async def complete(self, messages, *a, **kw):
        return "done."

    def reply_budget(self, messages, max_tokens=None, quiet=False):
        return max_tokens or 512

    async def wait_until_ready(self, timeout=0.0):
        return True

    @staticmethod
    def classify_error(exc):
        return "the model stalled"

    def prompt_text(self) -> str:
        out = []
        for prompt in self.prompts:
            for m in prompt:
                content = m.get("content")
                out.append(content if isinstance(content, str) else json.dumps(content))
        return "\n".join(out)


def build_app(replies, **overrides):
    """tests/test_app.py:build_app, plus a scripted model behind the real agent."""
    base = dict(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                allow_test_user=False, node_token="secret-node-token",
                skills_dir=tempfile.mkdtemp(prefix="local-llm-test-skills-"),
                agent_enabled=True, knowledge_triage=False, fast_path=False,
                incremental_reasoning=False, chunk_large_prompts=False,
                agent_max_steps=4, agent_min_steps=1, resilient_retries=1,
                ready_wait_timeout=0.01, auto_iterate_rounds=0)
    base.update(overrides)
    cfg = Config(**base)
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)  # never started
    app = create_app(cfg, db, mm, RetrainManager(db, mm, cfg), ToolRegistry(cfg, db))
    app.state.model_manager = mm
    mm.status = "ready"
    model = _ScriptedModel(replies)
    app.state.agent.client = model
    BROKER.clear()
    return app, cfg, db, model


def _cookie(c: TestClient, username: str, password: str) -> str:
    c.cookies.clear()
    r = c.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    value = "; ".join(f"{k}={v}" for k, v in c.cookies.items())
    c.cookies.clear()
    return value


def _add_user(c: TestClient, admin: str, username: str) -> str:
    r = c.post("/api/users", json={"username": username, "password": f"{username}pass12",
                                   "role": "user"}, headers={"cookie": admin})
    assert r.status_code in (200, 201), r.text
    return _cookie(c, username, f"{username}pass12")


def _frames(text: str) -> list[tuple[int | None, object]]:
    out = []
    for block in text.split("\n\n"):
        seq, data = None, None
        for line in block.splitlines():
            if line.startswith("id: "):
                seq = int(line[4:])
            elif line.startswith("data: "):
                data = line[6:]
        if data is not None:
            out.append((seq, data if data == "[DONE]" else json.loads(data)))
    return out


def _events(text: str) -> list[dict]:
    return [p for _, p in _frames(text) if isinstance(p, dict)]


def _of(events, kind):
    return [e for e in events if e.get("type") == kind]


def _kinds(events) -> list[str]:
    return [e["type"] for e in events]


def _answer_first_approval(c, cookie, cid, body, seen, *, action="answer", before=None):
    """A second client: wait for the pending approval, then answer (or cancel)."""
    def run():
        H = {"cookie": cookie}
        end = time.time() + 15
        while time.time() < end:
            pending = c.get("/api/approvals/pending", headers=H).json()["approvals"]
            if pending:
                ap = pending[0]
                if before is not None:
                    before(ap)
                active = c.get("/api/chat/active", params={"conversation_id": cid},
                               headers=H).json()["turn"]
                if action == "cancel":
                    r = c.post(f"/api/chat/turns/{active['turn_id']}/cancel", headers=H)
                else:
                    r = c.post(f"/api/chat/turns/{active['turn_id']}/approvals/{ap['id']}",
                               json=body, headers=H)
                seen.append({"approval": ap, "active": active, "status": r.status_code,
                             "body": r.json()})
                return
            time.sleep(0.02)
        seen.append({"timeout": True})

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


async def _drive(app, path: str, body: dict, headers: dict, stop) -> tuple[int, str]:
    """POST to the ASGI app; report http.disconnect once stop(text_so_far)."""
    chunks: list[str] = []
    gone = asyncio.Event()
    sent = False
    status: dict = {}

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": json.dumps(body).encode(),
                    "more_body": False}
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            status["code"] = message["status"]
        elif message["type"] == "http.response.body":
            piece = (message.get("body") or b"").decode()
            if piece:
                chunks.append(piece)
            if stop("".join(chunks)):
                gone.set()

    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    raw.append((b"content-type", b"application/json"))
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
             "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": path, "raw_path": path.encode(), "query_string": b"",
             "root_path": "", "headers": raw, "client": ("127.0.0.1", 50000),
             "server": ("testserver", 80)}
    await asyncio.wait_for(app(scope, receive, send), timeout=15)
    return status.get("code", 0), "".join(chunks)


@contextlib.contextmanager
def _workspace_dir():
    """Per-user roots under a temp dir, never the checkout's ./workspace."""
    tmp = Path(tempfile.mkdtemp(prefix="llm-integ-ws-")).resolve()
    saved = (tools_mod.WORKSPACE_DIR, ws_mod.WORKSPACE_DIR)
    tools_mod.WORKSPACE_DIR = tmp
    ws_mod.WORKSPACE_DIR = tmp
    try:
        yield tmp
    finally:
        tools_mod.WORKSPACE_DIR, ws_mod.WORKSPACE_DIR = saved


CALC = json.dumps({"tool": "calculator", "args": {"expression": "2+2"}})


# --------------------------------------------------------------------------- #
# Approvals over HTTP
# --------------------------------------------------------------------------- #
def test_approval_allow_from_a_second_client_resumes_the_turn():
    app, cfg, db, model = build_app(
        [CALC, ["<think>the tool", " said 4</think>", "It is 4."]],
        tool_approval_mode="all")
    with TestClient(app) as c:
        admin = _cookie(c, "admin", "adminpw123")
        bob = _add_user(c, admin, "bob")
        seen: list = []
        intruder: list = []

        def bob_tries(ap):
            # bob cannot answer the admin's prompt, nor see it in his list.
            r = c.post(f"/api/chat/turns/{ap['turn_id']}/approvals/{ap['id']}",
                       json={"decision": "allow"}, headers={"cookie": bob})
            mine = c.get("/api/approvals/pending", headers={"cookie": bob}).json()
            intruder.append((r.status_code, mine["approvals"]))

        approver = _answer_first_approval(c, admin, "conv-allow",
                                          {"decision": "allow", "scope": "once"}, seen,
                                          before=bob_tries)
        r = c.post("/api/chat/stream",
                   json={"message": "what is 2+2? use the calculator", "agent": True,
                         "conversation_id": "conv-allow"},
                   headers={"cookie": admin})
        approver.join(timeout=20)
        assert r.status_code == 200, r.text
        events = _events(r.text)
        kinds = _kinds(events)
        assert seen and seen[0].get("status") == 200, seen
        assert intruder and intruder[0][0] == 404 and intruder[0][1] == [], intruder

        start = events[0]
        assert start["type"] == "start" and start["turn_id"]
        assert seen[0]["active"]["turn_id"] == start["turn_id"]
        asked = _of(events, "approval_required")
        resolved = _of(events, "approval_resolved")
        call = _of(events, "tool_call")[0]
        result = _of(events, "tool_result")[0]
        assert len(asked) == 1 and len(resolved) == 1, kinds
        assert asked[0]["call_id"] == call["id"] == result["id"]
        assert asked[0]["id"] == seen[0]["approval"]["id"]
        assert asked[0]["name"] == "calculator" and asked[0]["expires_in"] > 0
        assert resolved[0]["decision"] == "allow" and resolved[0]["scope"] == "once"
        # The order the UI renders: ask, answer, then the call actually runs.
        assert (kinds.index("approval_required") < kinds.index("approval_resolved")
                < kinds.index("tool_result") < kinds.index("final"))
        assert result["error"] is None and "4" in str(result["result"]), result
        final = _of(events, "final")[-1]
        assert "4" in final["answer"] and final["turn_id"] == start["turn_id"]
        # Reasoning: split from the answer, carried on final, stored, replayed.
        assert "said 4" in final["reasoning"], final
        assert all("<think>" not in str(e.get("token", "")) for e in _of(events, "token"))
        assert r.text.rstrip().endswith("data: [DONE]")
        stored = c.get("/api/conversation/conv-allow",
                       headers={"cookie": admin}).json()["messages"][-1]
        assert stored["role"] == "assistant" and "said 4" in stored["meta"]["reasoning"]
        assert "calculator" in json.dumps(stored["meta"].get("tools") or stored["meta"])

        # Nothing left pending; the audit has the answer and the granted call.
        assert c.get("/api/approvals/pending",
                     headers={"cookie": admin}).json()["approvals"] == []
        rows = c.get("/api/audit/tool-decisions",
                     headers={"cookie": admin}).json()["decisions"]
        rules = {(row["decision"], row["rule"].split(":")[0]) for row in rows
                 if row["tool"] == "calculator"}
        assert ("allow", "approval") in rules and ("allow", "oneshot") in rules, rows
        assert app.state.tasks.chat_in_flight == 0
    db.close()


def test_approval_deny_never_runs_the_tool_and_the_model_replans():
    app, cfg, db, model = build_app([CALC, "I was not allowed to calculate that."],
                                    tool_approval_mode="all")
    ran: list = []
    real_call = app.state.agent.registry.call

    def counting_call(name, args, *a, **kw):
        ran.append(name)
        return real_call(name, args, *a, **kw)

    app.state.agent.registry.call = counting_call
    with TestClient(app) as c:
        admin = _cookie(c, "admin", "adminpw123")
        seen: list = []
        approver = _answer_first_approval(c, admin, "conv-deny",
                                          {"decision": "deny", "scope": "once"}, seen)
        r = c.post("/api/chat/stream",
                   json={"message": "what is 2+2? use the calculator", "agent": True,
                         "conversation_id": "conv-deny"},
                   headers={"cookie": admin})
        approver.join(timeout=20)
        events = _events(r.text)
        assert seen and seen[0]["status"] == 200, seen
        resolved = _of(events, "approval_resolved")
        assert resolved and resolved[0]["decision"] == "deny", _kinds(events)
        result = _of(events, "tool_result")[0]
        assert "denied by the user" in str(result["error"]), result
        assert "calculator" not in ran, ran
        # The model saw the refusal and answered without the tool.
        assert "denied by the user" in model.prompt_text()
        final = _of(events, "final")[-1]
        assert "not allowed" in final["answer"]
        rows = c.get("/api/audit/tool-decisions",
                     headers={"cookie": admin}).json()["decisions"]
        assert any(row["decision"] == "deny" and row["rule"] == "approval:once"
                   for row in rows), rows
        # A second answer to the same prompt is refused: it is single use.
        again = c.post(f"/api/chat/turns/{seen[0]['active']['turn_id']}/approvals/"
                       f"{seen[0]['approval']['id']}", json={"decision": "allow"},
                       headers={"cookie": admin})
        assert again.status_code == 404
    db.close()


def test_non_streaming_chat_is_unattended_and_denies_an_ask():
    """/api/chat cannot show a prompt: an "ask" takes TOOL_APPROVAL_UNATTENDED."""
    app, cfg, db, model = build_app([CALC, "Could not use the tool."],
                                    tool_approval_mode="all")
    with TestClient(app) as c:
        admin = _cookie(c, "admin", "adminpw123")
        r = c.post("/api/chat", json={"message": "what is 2+2? use the calculator",
                                      "agent": True, "conversation_id": "conv-unatt"},
                   headers={"cookie": admin})
        assert r.status_code == 200, r.text
        assert BROKER.pending(None) == []
        assert "nobody" in model.prompt_text() or "denied" in model.prompt_text()
    db.close()


def test_stop_during_an_approval_wait_cancels_the_turn_and_keeps_it():
    app, cfg, db, model = build_app([["Let me check. ", CALC]], tool_approval_mode="all")
    with TestClient(app) as c:
        admin = _cookie(c, "admin", "adminpw123")
        seen: list = []
        stopper = _answer_first_approval(c, admin, "conv-stop", None, seen,
                                         action="cancel")
        r = c.post("/api/chat/stream",
                   json={"message": "what is 2+2? use the calculator", "agent": True,
                         "conversation_id": "conv-stop"},
                   headers={"cookie": admin})
        stopper.join(timeout=20)
        events = _events(r.text)
        kinds = _kinds(events)
        assert seen and seen[0]["status"] == 200 and seen[0]["body"]["cancelled"], seen
        assert "approval_required" in kinds and kinds[-1] == "cancelled", kinds
        assert "final" not in kinds, kinds
        # The call never ran: either no result at all, or the refusal.
        assert all(x["error"] for x in _of(events, "tool_result")), events
        assert c.get("/api/approvals/pending",
                     headers={"cookie": admin}).json()["approvals"] == []
        stored = c.get("/api/conversation/conv-stop",
                       headers={"cookie": admin}).json()["messages"][-1]
        assert stored["role"] == "assistant" and stored["meta"].get("cancelled"), stored
        assert c.get("/api/chat/active", params={"conversation_id": "conv-stop"},
                     headers={"cookie": admin}).json()["turn"] is None
        assert app.state.tasks.chat_in_flight == 0
    db.close()


def test_a_dropped_stream_resumes_across_an_approval():
    """X-Stream-Ids client drops while the turn waits for approval; it is
    approved from elsewhere and the client resumes with ?after=N."""
    app, cfg, db, model = build_app([CALC, "It is 4."], tool_approval_mode="all",
                                    turn_detach_grace_s=20)
    with TestClient(app) as c:
        admin = _cookie(c, "admin", "adminpw123")
        status, text = c.portal.call(
            _drive, app, "/api/chat/stream",
            {"message": "what is 2+2? use the calculator", "agent": True,
             "conversation_id": "conv-resume"},
            {"cookie": admin, "x-stream-ids": "1"}, lambda t: "approval_required" in t)
        assert status == 200
        head = _frames(text)
        assert all(seq is not None for seq, _ in head), head
        turn_id, last = head[0][1]["turn_id"], head[-1][0]
        asked = [p for _, p in head if isinstance(p, dict)
                 and p["type"] == "approval_required"][0]
        time.sleep(0.2)
        active = c.get("/api/chat/active", params={"conversation_id": "conv-resume"},
                       headers={"cookie": admin}).json()["turn"]
        assert active and active["turn_id"] == turn_id and active["resumable"]
        r = c.post(f"/api/chat/turns/{turn_id}/approvals/{asked['id']}",
                   json={"decision": "allow"}, headers={"cookie": admin})
        assert r.status_code == 200, r.text
        r = c.get(f"/api/chat/turns/{turn_id}/events", params={"after": last},
                  headers={"cookie": admin, "X-Stream-Ids": "1"})
        tail = _frames(r.text)
        assert tail[0][0] == last + 1, tail
        kinds = [p["type"] if isinstance(p, dict) else p for _, p in tail]
        assert "approval_required" not in kinds        # nothing replayed twice
        assert kinds[0] == "approval_resolved" and kinds[-2:] == ["final", "[DONE]"], kinds
        result = [p for _, p in tail if isinstance(p, dict) and p["type"] == "tool_result"]
        assert result and "4" in str(result[0]["result"])
        stored = c.get("/api/conversation/conv-resume",
                       headers={"cookie": admin}).json()["messages"][-1]
        assert stored["role"] == "assistant" and "4" in stored["content"]
    db.close()


def test_approvals_off_is_todays_stream():
    """Default mode: no approval events, no audit rows, the call just runs."""
    app, cfg, db, model = build_app([CALC, "It is 4."])
    with TestClient(app) as c:
        admin = _cookie(c, "admin", "adminpw123")
        r = c.post("/api/chat/stream",
                   json={"message": "what is 2+2? use the calculator", "agent": True,
                         "conversation_id": "conv-off"},
                   headers={"cookie": admin})
        events = _events(r.text)
        kinds = _kinds(events)
        assert "approval_required" not in kinds and "final" in kinds, kinds
        assert "id: " not in r.text                     # default framing
        assert "4" in str(_of(events, "tool_result")[0]["result"])
        assert c.get("/api/audit/tool-decisions",
                     headers={"cookie": admin}).json()["decisions"] == []
    db.close()


# --------------------------------------------------------------------------- #
# Workspaces inside a chat turn
# --------------------------------------------------------------------------- #
def test_a_project_turn_confines_file_tools_and_execute_code_to_the_project():
    write = json.dumps({"tool": "write_file",
                        "args": {"path": "note.txt", "content": "hello project"}})
    code = ("tools.write_file(path='from_code.txt', content='C')\n"
            "print('read:', tools.read_file(path='note.txt').strip())\n")
    run = json.dumps({"tool": "execute_code", "args": {"code": code}})
    with _workspace_dir() as base:
        app, cfg, db, model = build_app([write, run, "Saved both."],
                                        workspaces_enabled=True, allow_python=True,
                                        tool_timeout=60)
        with TestClient(app) as c:
            admin = _cookie(c, "admin", "adminpw123")
            alice = _add_user(c, admin, "alice")
            r = c.post("/api/projects", json={"name": "Thesis"}, headers={"cookie": alice})
            assert r.status_code == 200, r.text
            project = r.json()["project"]
            root = Path(project["root"]).resolve()
            assert str(root).startswith(str(base))
            (root / "AGENTS.md").write_text("Always cite chapter numbers.", encoding="utf-8")
            r = c.post("/api/chat/stream",
                       json={"message": "save a note, then check it with code",
                             "agent": True, "conversation_id": "conv-proj",
                             "project_id": project["id"]},
                       headers={"cookie": alice})
            assert r.status_code == 200, r.text
            events = _events(r.text)
            injected = _of(events, "context_injected")
            assert injected and injected[0]["files"] == ["AGENTS.md"], _kinds(events)
            assert "Always cite chapter numbers." in model.prompt_text()
            results = _of(events, "tool_result")
            assert [x["name"] for x in results] == ["write_file", "execute_code"], results
            assert all(x["error"] is None for x in results), results
            assert "read: hello project" in str(results[1]["result"]), results[1]
            assert (root / "note.txt").read_text() == "hello project"
            assert (root / "from_code.txt").read_text() == "C"
            # Nothing landed in the user's own root or the shared default root.
            assert not (root.parent / "note.txt").exists()
            assert not (Path(cfg.project_dir or base) / "from_code.txt").exists()

            # Bob is not a member: the same project id is a 404 before streaming.
            bob = _add_user(c, admin, "bob")
            r = c.post("/api/chat/stream",
                       json={"message": "hi", "agent": True, "conversation_id": "conv-b",
                             "project_id": project["id"]},
                       headers={"cookie": bob})
            assert r.status_code == 404, r.text
        db.close()
    assert current_turn() is None


# --------------------------------------------------------------------------- #
# Agent profiles: capability gating and the profile's own tool policy
# --------------------------------------------------------------------------- #
def test_a_profile_turn_only_has_its_capabilities_and_its_tool_policy():
    from local_llm.policy import set_profile_tool_policy

    write = json.dumps({"tool": "write_file", "args": {"path": "x.txt", "content": "x"}})
    recall = json.dumps({"tool": "recall_memory", "args": {"query": "anything"}})
    app, cfg, db, model = build_app([write, recall, "I could do neither."],
                                    project_dir=tempfile.mkdtemp(prefix="llm-integ-pd-"))
    built: list = []
    real_factory = app.state.agent_factory

    def factory(turn_cfg, profile=None):
        agent = real_factory(turn_cfg, profile)
        agent.client = model
        built.append(agent)
        return agent

    app.state.agent_factory = factory
    profile = db.create_agent("Recaller", "memory only", ["memory"])
    set_profile_tool_policy(db, profile["id"], {"recall_memory": "deny"})
    with TestClient(app) as c:
        admin = _cookie(c, "admin", "adminpw123")
        r = c.post("/api/chat/stream",
                   json={"message": "save x.txt then recall", "conversation_id": "conv-prof",
                         "agent_id": profile["id"]},
                   headers={"cookie": admin})
        assert r.status_code == 200, r.text
        events = _events(r.text)
        assert built, "the profile turn must run on a per-turn agent"
        names = set(built[0].registry.names())
        assert "recall_memory" in names and "write_file" not in names, sorted(names)
        assert "approval_required" not in _kinds(events)
        assert not (Path(cfg.project_dir) / "x.txt").exists()
        results = {x["name"]: x for x in _of(events, "tool_result")}
        assert "denied by policy" in str(results["recall_memory"]["error"]), results
        assert _of(events, "final"), _kinds(events)
        # The default agent is untouched: it still has the file tools.
        assert "write_file" in set(app.state.agent.registry.names())
        # A disabled profile is invisible to a non-admin.
        db.update_agent(profile["id"], enabled=False)
        bob = _add_user(c, admin, "bob")
        r = c.post("/api/chat/stream", json={"message": "hi", "conversation_id": "conv-p2",
                                             "agent_id": profile["id"]},
                   headers={"cookie": bob})
        assert r.status_code == 404, r.text
    db.close()


# --------------------------------------------------------------------------- #
# Scheduled tasks run as unattended turns
# --------------------------------------------------------------------------- #
def test_a_scheduled_run_is_an_unattended_turn_and_an_ask_is_denied():
    import local_llm.tasks as tasks_mod
    from local_llm.tasks import TaskManager, TaskRun

    cfg = Config(agent_enabled=True, knowledge_triage=False, fast_path=False,
                 incremental_reasoning=False, chunk_large_prompts=False,
                 agent_max_steps=3, agent_min_steps=1, resilient_retries=1,
                 ready_wait_timeout=0.01, auto_iterate_rounds=0,
                 tool_approval_mode="all",
                 skills_dir=tempfile.mkdtemp(prefix="local-llm-test-skills-"))
    db = Database(Path(tempfile.mkdtemp()) / "tasks.db")
    owner = db.create_user("tasker", role="user")
    task = db.create_task(user_id=owner["id"], name="nightly", goal="what is 2+2? calculator")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)  # never started
    model = _ScriptedModel([CALC, "No tool allowed."])
    manager = TaskManager(cfg, db, mm, RetrainManager(db, mm, cfg), model)
    seen: list = []
    real_agent = tasks_mod.Agent

    class _Agent(real_agent):
        def __init__(self, config, registry, client):
            super().__init__(config, registry, model)

        async def run(self, *a, **kw):
            turn = current_turn()
            seen.append(turn)
            async for event in super().run(*a, **kw):
                yield event

    tasks_mod.Agent = _Agent
    try:
        run = TaskRun(db.create_run(task["id"], "manual", cfg.model), task["id"], task["name"])
        asyncio.run(asyncio.wait_for(manager._execute(task, run, "manual"), timeout=30))
    finally:
        tasks_mod.Agent = real_agent
    turn = seen[0]
    assert turn is not None and turn.interactive is False, turn
    assert turn.user_id == owner["id"] and turn.role == "user"
    assert turn.turn_id.startswith("task-")
    assert BROKER.pending(None) == []
    assert "nobody" in model.prompt_text(), model.prompt_text()[-400:]
    rows = [r for r in list_tool_decisions(db, user_id=owner["id"])
            if r["tool"] == "calculator"]
    assert rows and rows[0]["decision"] == "deny" and rows[0]["turn_id"] == turn.turn_id, rows
    assert current_turn() is None
    db.close()


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
