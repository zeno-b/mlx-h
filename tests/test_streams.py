"""Tests for resumable, cancellable chat turns (local_llm/streams.py + chat routes).

Route tests drive the real app. A dropped connection is simulated by calling the
ASGI app directly on the TestClient's own event loop with a receive() that
reports http.disconnect once the frames seen so far satisfy a predicate: the
TestClient itself always reads a streaming response to the end, so it cannot
drop one. Fake agents poll a threading.Event, so a test thread controls when a
turn may finish, and read current_turn() the way the real agent loop does.
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_llm.config as config_mod  # noqa: E402
config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-streams-"))

from fastapi.testclient import TestClient  # noqa: E402

import local_llm.streams as streams_mod  # noqa: E402
from local_llm.api import create_app  # noqa: E402
from local_llm.auth import hash_password  # noqa: E402
from local_llm.cluster import Node  # noqa: E402
from local_llm.config import CAPABILITY_GROUPS, Config  # noqa: E402
from local_llm.core import ADAPTER_DIR  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.llm import GenerationStats  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.streams import (  # noqa: E402
    TurnRegistry, optional_feature, parse_message_meta, reasoning_of, turn_frames,
    turn_message_meta, wants_stream_ids)
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.training import RetrainManager  # noqa: E402
from local_llm.turnctx import current_turn  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def build_app(**overrides):
    """Same construction as tests/test_app.py:build_app."""
    base = dict(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                allow_test_user=False, node_token="secret-node-token",
                skills_dir=tempfile.mkdtemp(prefix="local-llm-test-skills-"),
                agent_enabled=True, knowledge_triage=False)
    base.update(overrides)
    cfg = Config(**base)
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)  # never started
    rm = RetrainManager(db, mm, cfg)
    app = create_app(cfg, db, mm, rm, ToolRegistry(cfg, db))
    app.state.model_manager = mm
    mm.status = "ready"
    return app, cfg, db


def _cookie(c: TestClient, username: str, password: str) -> str:
    """Log in and return the session as an explicit Cookie header value.

    One TestClient (one event loop) serves every user in a test: requests pass
    their user's cookie explicitly, which httpx prefers over its jar.
    """
    c.cookies.clear()
    r = c.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    value = "; ".join(f"{k}={v}" for k, v in c.cookies.items())
    c.cookies.clear()
    return value


def _final(answer: str = "ok", **extra) -> dict:
    event = {"type": "final", "answer": answer, "steps": 1, "trace": [],
             "tools_used": [], "elapsed_ms": 1, "prompt_tokens": 1,
             "completion_tokens": 1, "truncated": False, "changed_files": [], "diff": ""}
    event.update(extra)
    return event


def _blocking_agent(app, release: threading.Event, *, cooperative: bool = True,
                    tokens=("par", "tial")):
    """Streams `tokens`, then waits for `release` (or a cancel, if cooperative)."""
    async def fake_run(message, history, conversation_id=None, max_tokens=None,
                       temperature=None, cancel=None, **kw):
        for t in tokens:
            yield {"type": "token", "text": t, "step": 1}
        turn = current_turn()
        while not release.is_set():
            if cooperative and turn is not None and turn.cancelled():
                yield {"type": "cancelled", "step": 1, "partial": "".join(tokens),
                       "trace": []}
                return
            await asyncio.sleep(0.01)
        yield _final("".join(tokens) + " done")

    app.state.agent.run_iterating = fake_run


def _frames(text: str) -> list[tuple[int | None, object]]:
    """[(id or None, payload)] for every data frame; payload "[DONE]" or a dict."""
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


def _kinds(frames) -> list:
    return [p["type"] if isinstance(p, dict) else p for _, p in frames]


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
    await asyncio.wait_for(app(scope, receive, send), timeout=10)
    return status.get("code", 0), "".join(chunks)


def _wait(predicate, timeout: float = 5.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _active(c, cookie, cid):
    return c.get("/api/chat/active", params={"conversation_id": cid},
                 headers={"cookie": cookie}).json()["turn"]


def _messages(c, cookie, cid):
    return c.get(f"/api/conversation/{cid}", headers={"cookie": cookie}).json()["messages"]


# --------------------------------------------------------------------------- #
# Unit: the turn buffer and registry
# --------------------------------------------------------------------------- #
def test_buffer_is_bounded_and_a_late_subscriber_is_told_about_the_gap():
    async def scenario():
        reg = TurnRegistry(Config(), max_events=16)
        turn = reg.create("u1", "c1")
        for i in range(40):
            turn.push({"type": "token", "token": str(i)})
        turn.finish()
        frames = [f async for f in turn_frames(turn, 0, ids=True)]
        return _frames("".join(frames))

    got = asyncio.run(scenario())
    assert got[0][0] is None and "no longer buffered" in got[0][1]["message"]
    assert [s for s, _ in got[1:-1]] == list(range(25, 41))
    assert got[-1][1] == "[DONE]"


def test_emit_from_a_worker_thread_reaches_the_buffer():
    async def scenario():
        turn = TurnRegistry(Config()).create("u1", "c1")
        await asyncio.to_thread(turn.emit, {"type": "notice", "message": "from a tool"})
        await asyncio.sleep(0.05)
        return turn.events_after(0)

    got = asyncio.run(scenario())
    assert got == [(1, {"type": "notice", "message": "from a tool"})]


def test_detach_cancels_at_once_unless_resumable_then_after_the_grace():
    async def scenario():
        reg = TurnRegistry(Config())
        plain = reg.create("u1", "c1")
        plain.attach(ids=False)
        plain.detach(grace=30)
        resumable = reg.create("u1", "c2")
        resumable.attach(ids=True)
        resumable.detach(grace=0.1)
        early = resumable.cancel.is_set()
        await asyncio.sleep(0.25)
        back = reg.create("u1", "c3")
        back.attach(ids=True)
        back.detach(grace=0.1)
        back.attach(ids=True)          # reconnected inside the grace
        await asyncio.sleep(0.25)
        return plain.cancel.is_set(), early, resumable.cancel.is_set(), back.cancel.is_set()

    assert asyncio.run(scenario()) == (True, False, True, False)


def test_registry_is_owner_scoped_and_prunes_finished_turns():
    async def scenario():
        reg = TurnRegistry(Config(), retain_s=0.0)
        turn = reg.create("alice", "c1")
        assert reg.get(turn.turn_id, "alice") is turn
        assert reg.get(turn.turn_id, "bob") is None
        assert [t.turn_id for t in reg.active_for("alice", "c1")] == [turn.turn_id]
        assert reg.active_for("bob") == []
        turn.finish()
        await asyncio.sleep(0.01)
        reg.prune()
        return len(reg)

    assert asyncio.run(scenario()) == 0


def test_helpers():
    assert wants_stream_ids({"x-stream-ids": "1"}) and not wants_stream_ids({})
    assert parse_message_meta('{"reasoning": "r"}') == {"reasoning": "r"}
    assert parse_message_meta("not json") is None and parse_message_meta(None) is None
    assert turn_message_meta() is None
    assert turn_message_meta([], "why", ["web_search"], True) == {
        "reasoning": "why", "tools": ["web_search"], "cancelled": True}
    assert reasoning_of("<think>because</think>answer") == "because"
    assert reasoning_of("plain answer") == ""
    cfg = Config()
    assert cfg.turn_detach_grace_s == 30.0
    assert "turn_detach_grace_s" in Config.MUTABLE
    cfg.apply({"turn_detach_grace_s": -5})
    assert cfg.turn_detach_grace_s == 0.0


# --------------------------------------------------------------------------- #
# Framing
# --------------------------------------------------------------------------- #
def test_default_framing_is_unchanged_and_start_and_final_carry_the_turn():
    app, cfg, db = build_app()

    async def fake_run(message, history, conversation_id=None, max_tokens=None,
                       temperature=None, cancel=None):
        yield {"type": "token", "text": "o", "step": 1}
        yield _final()

    app.state.agent.run_iterating = fake_run
    with TestClient(app) as c:
        cookie = _cookie(c, "admin", "adminpw123")
        r = c.post("/api/chat/stream", json={"message": "hi", "agent": True},
                   headers={"cookie": cookie})
        assert r.status_code == 200
        lines = [ln for ln in r.text.splitlines() if ln.strip()]
        assert all(ln.startswith("data: ") for ln in lines), lines
        frames = _frames(r.text)
        assert all(seq is None for seq, _ in frames)
        assert _kinds(frames) == ["start", "token", "final", "[DONE]"]
        start, final = frames[0][1], frames[2][1]
        assert start["turn_id"] and final["turn_id"] == start["turn_id"]
        assert final["reasoning"] == ""


def test_stream_ids_header_numbers_every_frame():
    app, cfg, db = build_app()

    async def fake_run(message, history, conversation_id=None, max_tokens=None,
                       temperature=None, cancel=None):
        for t in "abc":
            yield {"type": "token", "text": t, "step": 1}
        yield _final("abc")

    app.state.agent.run_iterating = fake_run
    with TestClient(app) as c:
        cookie = _cookie(c, "admin", "adminpw123")
        r = c.post("/api/chat/stream", json={"message": "hi", "agent": True},
                   headers={"cookie": cookie, "X-Stream-Ids": "1"})
        frames = _frames(r.text)
        assert [s for s, _ in frames[:-1]] == list(range(1, len(frames)))
        assert frames[-1] == (None, "[DONE]")
        # A finished turn stays readable: a replay from any point is the same tail.
        turn_id = frames[0][1]["turn_id"]
        again = _frames(c.get(f"/api/chat/turns/{turn_id}/events", params={"after": 2},
                              headers={"cookie": cookie, "X-Stream-Ids": "1"}).text)
        assert again == frames[2:]


# --------------------------------------------------------------------------- #
# Resume, detach grace, cancel
# --------------------------------------------------------------------------- #
def test_resume_after_a_dropped_stream_with_ids():
    app, cfg, db = build_app(turn_detach_grace_s=10)
    release = threading.Event()
    _blocking_agent(app, release)
    with TestClient(app) as c:
        cookie = _cookie(c, "admin", "adminpw123")
        status, text = c.portal.call(
            _drive, app, "/api/chat/stream",
            {"message": "hi", "agent": True, "conversation_id": "conv-resume"},
            {"cookie": cookie, "x-stream-ids": "1"}, lambda t: '"tial"' in t)
        assert status == 200
        seen = _frames(text)
        assert _kinds(seen)[:3] == ["start", "token", "token"], seen
        turn_id, last = seen[0][1]["turn_id"], seen[-1][0]
        # The client is gone, the turn is not: a reload finds it.
        time.sleep(0.2)
        active = _active(c, cookie, "conv-resume")
        assert active and active["turn_id"] == turn_id and active["resumable"]
        release.set()
        r = c.get(f"/api/chat/turns/{turn_id}/events", params={"after": last},
                  headers={"cookie": cookie, "X-Stream-Ids": "1"})
        tail = _frames(r.text)
        assert tail[0][0] == last + 1, tail
        assert _kinds(tail)[-2:] == ["final", "[DONE]"]
        assert "token" not in _kinds(tail)          # nothing replayed twice
        stored = _messages(c, cookie, "conv-resume")
        assert stored[-1]["role"] == "assistant" and stored[-1]["content"] == "partial done"
        assert _active(c, cookie, "conv-resume") is None
        assert app.state.tasks.chat_in_flight == 0


def test_disconnect_without_ids_cancels_at_once_and_keeps_the_partial():
    app, cfg, db = build_app(turn_detach_grace_s=30)
    release = threading.Event()
    _blocking_agent(app, release, cooperative=False)
    with TestClient(app) as c:
        cookie = _cookie(c, "admin", "adminpw123")
        status, text = c.portal.call(
            _drive, app, "/api/chat/stream",
            {"message": "hi", "agent": True, "conversation_id": "conv-drop"},
            {"cookie": cookie}, lambda t: '"tial"' in t)
        assert status == 200 and "id: " not in text
        assert _wait(lambda: _active(c, cookie, "conv-drop") is None, 3)
        stored = _messages(c, cookie, "conv-drop")
        assert stored[-1]["content"] == "partial"
        assert stored[-1]["meta"]["cancelled"] is True
        assert c.portal.call(app.state.ctx.generation_gate.snapshot)["running"] == 0
        assert app.state.tasks.chat_in_flight == 0
    release.set()


def test_detach_grace_expires_then_the_turn_is_cancelled_and_stored():
    app, cfg, db = build_app(turn_detach_grace_s=0.3)
    release = threading.Event()
    _blocking_agent(app, release)
    with TestClient(app) as c:
        cookie = _cookie(c, "admin", "adminpw123")
        _, text = c.portal.call(
            _drive, app, "/api/chat/stream",
            {"message": "hi", "agent": True, "conversation_id": "conv-grace"},
            {"cookie": cookie, "x-stream-ids": "1"}, lambda t: '"tial"' in t)
        turn_id = _frames(text)[0][1]["turn_id"]
        time.sleep(0.1)
        assert _active(c, cookie, "conv-grace") is not None    # inside the grace
        assert _wait(lambda: _active(c, cookie, "conv-grace") is None, 4)
        replay = _frames(c.get(f"/api/chat/turns/{turn_id}/events",
                               headers={"cookie": cookie}).text)
        cancelled = [p for _, p in replay if isinstance(p, dict) and p["type"] == "cancelled"]
        assert cancelled and cancelled[0]["partial"] == "partial"
        assert cancelled[0]["turn_id"] == turn_id
        stored = _messages(c, cookie, "conv-grace")
        assert stored[-1]["content"] == "partial" and stored[-1]["meta"]["cancelled"]
    release.set()


def test_reconnecting_inside_the_grace_keeps_the_turn_alive():
    app, cfg, db = build_app(turn_detach_grace_s=0.5)
    release = threading.Event()
    _blocking_agent(app, release)
    with TestClient(app) as c:
        cookie = _cookie(c, "admin", "adminpw123")
        _, text = c.portal.call(
            _drive, app, "/api/chat/stream",
            {"message": "hi", "agent": True, "conversation_id": "conv-back"},
            {"cookie": cookie, "x-stream-ids": "1"}, lambda t: '"tial"' in t)
        first = _frames(text)
        # Finish well after the grace would have expired.
        threading.Timer(1.2, release.set).start()
        r = c.get(f"/api/chat/turns/{first[0][1]['turn_id']}/events",
                  params={"after": first[-1][0]},
                  headers={"cookie": cookie, "X-Stream-Ids": "1"})
        kinds = _kinds(_frames(r.text))
        assert "final" in kinds and "cancelled" not in kinds, kinds


def _stream_in_thread(c, body, headers):
    out: list = []
    worker = threading.Thread(
        target=lambda: out.append(c.post("/api/chat/stream", json=body, headers=headers)),
        daemon=True)
    worker.start()
    return worker, out


def test_cancel_endpoint_stops_the_turn_and_stores_the_partial():
    app, cfg, db = build_app()
    release = threading.Event()
    _blocking_agent(app, release)
    with TestClient(app) as c:
        cookie = _cookie(c, "admin", "adminpw123")
        worker, out = _stream_in_thread(
            c, {"message": "hi", "agent": True, "conversation_id": "conv-stop"},
            {"cookie": cookie})
        assert _wait(lambda: _active(c, cookie, "conv-stop") is not None)
        turn_id = _active(c, cookie, "conv-stop")["turn_id"]
        r = c.post(f"/api/chat/turns/{turn_id}/cancel", headers={"cookie": cookie})
        assert r.status_code == 200 and r.json()["cancelled"] is True
        worker.join(5)
        frames = _frames(out[0].text)
        assert _kinds(frames)[-2:] == ["cancelled", "[DONE]"]
        assert frames[-2][1]["partial"] == "partial"
        assert _messages(c, cookie, "conv-stop")[-1]["meta"]["cancelled"] is True
        again = c.post(f"/api/chat/turns/{turn_id}/cancel", headers={"cookie": cookie})
        assert again.json()["cancelled"] is False
    release.set()


def test_cancel_falls_back_to_cancelling_the_task_when_the_loop_ignores_it():
    app, cfg, db = build_app()
    release = threading.Event()
    _blocking_agent(app, release, cooperative=False)
    saved = streams_mod._STREAMS_HARD_CANCEL_S
    streams_mod._STREAMS_HARD_CANCEL_S = 0.2
    try:
        with TestClient(app) as c:
            cookie = _cookie(c, "admin", "adminpw123")
            worker, out = _stream_in_thread(
                c, {"message": "hi", "agent": True, "conversation_id": "conv-hard"},
                {"cookie": cookie})
            assert _wait(lambda: _active(c, cookie, "conv-hard") is not None)
            turn_id = _active(c, cookie, "conv-hard")["turn_id"]
            c.post(f"/api/chat/turns/{turn_id}/cancel", headers={"cookie": cookie})
            worker.join(5)
            frames = _frames(out[0].text)
            assert _kinds(frames)[-2:] == ["cancelled", "[DONE]"], _kinds(frames)
            assert frames[-2][1]["reason"] == "cancelled by the user"
            assert _messages(c, cookie, "conv-hard")[-1]["content"] == "partial"
            assert app.state.tasks.chat_in_flight == 0
    finally:
        streams_mod._STREAMS_HARD_CANCEL_S = saved
        release.set()


def test_turn_routes_are_owner_checked():
    app, cfg, db = build_app()
    db.create_user("bob", password_hash=hash_password("bobpw12345"), role="user")
    release = threading.Event()
    _blocking_agent(app, release)
    with TestClient(app) as c:
        admin = _cookie(c, "admin", "adminpw123")
        bob = _cookie(c, "bob", "bobpw12345")
        worker, out = _stream_in_thread(
            c, {"message": "hi", "agent": True, "conversation_id": "conv-own"},
            {"cookie": admin})
        assert _wait(lambda: _active(c, admin, "conv-own") is not None)
        turn_id = _active(c, admin, "conv-own")["turn_id"]
        H = {"cookie": bob}
        assert c.get(f"/api/chat/turns/{turn_id}/events", headers=H).status_code == 404
        assert c.post(f"/api/chat/turns/{turn_id}/cancel", headers=H).status_code == 404
        assert c.post(f"/api/chat/turns/{turn_id}/approvals/x", headers=H,
                      json={"decision": "allow"}).status_code == 404
        assert c.post(f"/api/chat/turns/{turn_id}/consent/x", headers=H,
                      json={"decision": "allow"}).status_code == 404
        assert c.get("/api/chat/active", params={"conversation_id": "conv-own"},
                     headers=H).json() == {"turn": None, "turns": []}
        # The owner reaches the turn; prompts that do not exist are 404, bad
        # answers 400.
        A = {"cookie": admin}
        assert c.get("/api/chat/turns/nope/events", headers=A).status_code == 404
        assert c.post(f"/api/chat/turns/{turn_id}/approvals/nope", headers=A,
                      json={"decision": "maybe"}).status_code == 400
        assert c.post(f"/api/chat/turns/{turn_id}/approvals/nope", headers=A,
                      json={"decision": "allow", "scope": "once"}).status_code == 404
        assert c.post(f"/api/chat/turns/{turn_id}/consent/nope", headers=A,
                      json={"decision": "deny"}).status_code == 404
        assert c.post(f"/api/chat/turns/{turn_id}/cancel", headers=H).status_code == 404
        release.set()
        worker.join(5)
        assert "final" in _kinds(_frames(out[0].text))


# --------------------------------------------------------------------------- #
# Reasoning persisted and replayed
# --------------------------------------------------------------------------- #
def test_reasoning_is_stored_and_replayed_on_both_lanes():
    app, cfg, db = build_app(agent_enabled=False, reasoning_visible=True)
    client = app.state.ctx.model_client

    async def fake_stream(messages, max_tokens, temperature, stats=None, **kw):
        for piece in ("<think>weighing", " it</think>", "the answer"):
            yield piece

    async def fake_complete(messages, max_tokens, temperature, **kw):
        return "<think>because</think>plain answer", GenerationStats()

    client.stream = fake_stream
    client.complete_with_stats = fake_complete
    with TestClient(app) as c:
        cookie = _cookie(c, "admin", "adminpw123")
        H = {"cookie": cookie}
        r = c.post("/api/chat/stream", json={"message": "hello", "conversation_id": "conv-th"},
                   headers=H)
        final = [p for _, p in _frames(r.text) if isinstance(p, dict) and p["type"] == "final"][0]
        assert final["answer"] == "the answer" and final["reasoning"] == "weighing it"
        stored = _messages(c, cookie, "conv-th")[-1]
        assert stored["meta"] == {"reasoning": "weighing it"}

        r = c.post("/api/chat", json={"message": "hello", "conversation_id": "conv-th2"},
                   headers=H)
        body = r.json()
        assert body["answer"] == "plain answer" and body["reasoning"] == "because"
        assert body["turn_id"]
        assert _messages(c, cookie, "conv-th2")[-1]["meta"]["reasoning"] == "because"

        # Agent lane: an agent that streams think tokens but reports no
        # reasoning on its final still gets it stored, with its tools.
        async def fake_run(message, history, conversation_id=None, max_tokens=None,
                           temperature=None, cancel=None):
            yield {"type": "think_token", "token": "plan", "step": 1}
            yield _final("done", tools_used=["web_search"])

        app.state.agent.run_iterating = fake_run
        r = c.post("/api/chat/stream", json={"message": "hi", "agent": True,
                                             "conversation_id": "conv-th3"}, headers=H)
        final = [p for _, p in _frames(r.text) if isinstance(p, dict) and p["type"] == "final"][0]
        assert final["reasoning"] == "plan"
        meta = _messages(c, cookie, "conv-th3")[-1]["meta"]
        assert meta["reasoning"] == "plan" and meta["tools"] == ["web_search"]


# --------------------------------------------------------------------------- #
# Agent profiles, models, projects, attachments
# --------------------------------------------------------------------------- #
class _ProfileAgent:
    def __init__(self, cfg):
        self.cfg = cfg

    async def run_iterating(self, message, history, conversation_id=None, max_tokens=None,
                            temperature=None, cancel=None, **kw):
        turn = current_turn()
        yield _final("from profile",
                     tools_used=[(turn.agent_profile or {}).get("name", "")] if turn else [])


def test_agent_id_is_gated_by_enabled_and_role_capabilities():
    app, cfg, db = build_app()
    db.create_user("bob", password_hash=hash_password("bobpw12345"), role="user")
    cfg.role_capabilities = json.dumps({"user": ["web_api", "knowledge"]})
    built: list = []

    def factory(turn_cfg, profile):
        built.append((turn_cfg, profile))
        return _ProfileAgent(turn_cfg)

    app.state.agent_factory = factory
    off = db.create_agent("off", capabilities=["web_api"], enabled=False)
    on = db.create_agent("on", capabilities=["web_api", "code_exec"], enabled=True)
    web = set(CAPABILITY_GROUPS["web_api"]["tools"])
    code = set(CAPABILITY_GROUPS["code_exec"]["tools"])
    with TestClient(app) as c:
        admin = {"cookie": _cookie(c, "admin", "adminpw123")}
        bob = {"cookie": _cookie(c, "bob", "bobpw12345")}
        msg = {"message": "hi"}
        assert c.post("/api/chat", json=dict(msg, agent_id=off["id"]), headers=bob).status_code == 404
        assert c.post("/api/chat/stream", json=dict(msg, agent_id=off["id"]),
                      headers=bob).status_code == 404
        assert c.post("/api/chat", json=dict(msg, agent_id="missing"), headers=admin).status_code == 404
        r = c.post("/api/chat", json=dict(msg, agent_id=off["id"]), headers=admin)
        assert r.status_code == 200 and r.json()["answer"] == "from profile"

        r = c.post("/api/chat", json=dict(msg, agent_id=on["id"]), headers=bob)
        assert r.status_code == 200, r.text
        turn_cfg, profile = built[-1]
        assert profile["capabilities"] == ["web_api"]        # code_exec cut by the role
        tools = set(turn_cfg.agent_tools.split(","))
        assert web <= tools and not (code & tools)

        r = c.post("/api/chat", json=dict(msg, agent_id=on["id"]), headers=admin)
        assert built[-1][1]["capabilities"] == ["code_exec", "web_api"]

        # Cached per profile: a repeat builds nothing, a profile edit rebuilds.
        count = len(built)
        c.post("/api/chat", json=dict(msg, agent_id=on["id"]), headers=bob)
        assert len(built) == count
        db.update_agent(on["id"], capabilities=["web_api", "knowledge"])
        c.post("/api/chat", json=dict(msg, agent_id=on["id"]), headers=bob)
        assert len(built) == count + 1
        assert built[-1][1]["capabilities"] == ["web_api", "knowledge"]

        # No agent_id: the global agent, the factory is never consulted.
        async def fake_run(message, history, conversation_id=None, max_tokens=None,
                           temperature=None, cancel=None):
            yield _final("global")

        app.state.agent.run_iterating = fake_run
        count = len(built)
        assert c.post("/api/chat", json=dict(msg, agent=True), headers=bob).json()["answer"] == "global"
        assert len(built) == count


def test_model_must_be_served_or_a_route_target():
    app, cfg, db = build_app()
    built: list = []
    app.state.agent_factory = lambda turn_cfg, profile: built.append(turn_cfg) or _ProfileAgent(turn_cfg)

    async def fake_run(message, history, conversation_id=None, max_tokens=None,
                       temperature=None, cancel=None):
        yield _final("served")

    app.state.agent.run_iterating = fake_run
    with TestClient(app) as c:
        H = {"cookie": _cookie(c, "admin", "adminpw123")}
        msg = {"message": "hi", "agent": True}
        r = c.post("/api/chat", json=dict(msg, model="nope/not-served"), headers=H)
        assert r.status_code == 400 and cfg.model in r.json()["served"]
        assert c.post("/api/chat/stream", json=dict(msg, model="nope/not-served"),
                      headers=H).status_code == 400
        r = c.post("/api/chat", json=dict(msg, model=cfg.model), headers=H)
        assert r.status_code == 200 and r.json()["answer"] == "served" and not built
        studio = Node(name="studio", role="secondary", is_local=False,
                      remote_url="http://10.0.0.2:8000", model="mlx-community/Other-4bit")
        app.state.node_registry.nodes.append(studio)
        try:
            r = c.post("/api/chat", json=dict(msg, model="mlx-community/Other-4bit"), headers=H)
            assert r.status_code == 200, r.text
            assert built and built[-1].model == "mlx-community/Other-4bit"
            assert cfg.model != "mlx-community/Other-4bit"       # live config untouched
        finally:
            app.state.node_registry.nodes.remove(studio)


def test_project_and_attachments_are_validated_before_streaming():
    app, cfg, db = build_app()
    with TestClient(app) as c:
        H = {"cookie": _cookie(c, "admin", "adminpw123")}
        r = c.post("/api/chat/stream", json={"message": "hi", "project_id": "no-such"}, headers=H)
        if optional_feature("workspaces", "resolve_project") is None:
            assert r.status_code == 400 and "projects not available" in r.text
        else:
            assert r.status_code == 404, r.text
        r = c.post("/api/chat", json={"message": "hi", "attachments": ["no-such"]}, headers=H)
        if optional_feature("attachments", "content_parts_for") is None:
            assert r.status_code == 400 and "attachments not available" in r.text
        else:
            assert 400 <= r.status_code < 500, r.text


def test_turn_context_is_bound_for_the_agent():
    app, cfg, db = build_app()
    seen: list = []

    async def fake_run(message, history, conversation_id=None, max_tokens=None,
                       temperature=None, cancel=None):
        turn = current_turn()
        seen.append((turn.turn_id, turn.interactive, turn.conversation_id,
                     turn.suspend_admission is not None, turn.role))
        yield _final()

    app.state.agent.run_iterating = fake_run
    with TestClient(app) as c:
        H = {"cookie": _cookie(c, "admin", "adminpw123")}
        r = c.post("/api/chat/stream", json={"message": "hi", "agent": True,
                                             "conversation_id": "conv-ctx"}, headers=H)
        start = _frames(r.text)[0][1]
        assert seen[-1] == (start["turn_id"], True, "conv-ctx", True, "admin")
        r = c.post("/api/chat", json={"message": "hi", "agent": True,
                                      "conversation_id": "conv-ctx"}, headers=H)
        assert seen[-1] == (r.json()["turn_id"], False, "conv-ctx", True, "admin")
    assert current_turn() is None


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
