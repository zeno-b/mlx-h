"""Tests for the OpenAI / Anthropic compatible /v1 API (openai_compat.py, anthropic_compat.py).

The model is faked one layer below stream_events: ModelClient._stream_core is
replaced on the app's shared client with a scripted generator of raw server
events (content, reasoning, native tool_call_delta fragments, finish, usage).
That keeps the REAL stream_events in the path, so the native joining, the
prompt-protocol ToolCallStreamDetector and the JSON repair are exercised as in
production. The backend's capabilities are pinned per test (native tool calls
or not) by patching compat_backend_caps. Agent mode uses a fake run_iterating.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_llm.config as config_mod  # noqa: E402
config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-compat-"))

from fastapi.testclient import TestClient  # noqa: E402

import local_llm.openai_compat as oai  # noqa: E402
from local_llm.anthropic_compat import compat_parse_messages  # noqa: E402
from local_llm.api import create_app  # noqa: E402
from local_llm.auth import hash_password  # noqa: E402
from local_llm.config import CAPABILITY_GROUPS, Config  # noqa: E402
from local_llm.core import ADAPTER_DIR, GenerationBusy  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.training import RetrainManager  # noqa: E402
from local_llm.turnctx import current_turn  # noqa: E402

MODEL = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def build_app(**overrides):
    """Same construction as tests/test_app.py:build_app."""
    base = dict(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                allow_test_user=False, node_token="secret-node-token", model=MODEL,
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


def _bearer(c: TestClient, username: str = "admin", password: str = "adminpw123") -> dict:
    """A Bearer session token, the way an SDK client authenticates today."""
    c.cookies.clear()
    r = c.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    token = c.cookies.get("llm_session")
    c.cookies.clear()
    assert token
    return {"Authorization": f"Bearer {token}"}


class FakeModel:
    """Scripted raw server events in place of ModelClient._stream_core."""

    def __init__(self, *scripts, usage=None):
        self.scripts = list(scripts)
        self.calls: list[dict] = []
        self.usage = usage or {"prompt_tokens": 40, "completion_tokens": 7,
                               "prompt_tokens_details": {"cached_tokens": 12}}

    def install(self, app):
        app.state.ctx.model_client._stream_core = self.core
        return self

    async def core(self, messages, max_tokens=None, temperature=None, stats=None,
                   kind="chat", conversation_id=None, extra=None):
        self.calls.append({"messages": messages, "max_tokens": max_tokens,
                           "temperature": temperature, "extra": dict(extra or {})})
        script = self.scripts.pop(0) if len(self.scripts) > 1 else self.scripts[0]
        for event in script:
            yield dict(event)
        yield {"type": "usage", "usage": self.usage}
        if stats is not None:
            stats.prompt_tokens = self.usage["prompt_tokens"]
            stats.completion_tokens = self.usage["completion_tokens"]


def _content(*pieces, finish="stop"):
    return [*({"type": "content", "text": p} for p in pieces),
            {"type": "finish", "reason": finish}]


def _caps(monkeypatch, *caps):
    monkeypatch.setattr(oai, "compat_backend_caps", lambda ctx: frozenset(caps))


def _sse_data(text: str) -> list:
    out = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                data = line[6:]
                out.append(data if data == "[DONE]" else json.loads(data))
    return out


def _sse_events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if data is not None:
            out.append((name, data))
    return out


def _final(answer="agent answer", **extra):
    event = {"type": "final", "answer": answer, "steps": 1, "trace": [],
             "tools_used": [], "elapsed_ms": 3, "prompt_tokens": 11,
             "completion_tokens": 5, "truncated": False, "changed_files": [], "diff": "",
             "reasoning": ""}
    event.update(extra)
    return event


def _fake_agent(app, seen: list, events=None):
    async def run(message, history, conversation_id=None, max_tokens=None,
                  temperature=None, cancel=None, **kw):
        turn = current_turn()
        seen.append({"message": message, "history": history, "kw": kw,
                     "interactive": turn.interactive if turn else None,
                     "profile": turn.agent_profile if turn else None})
        for ev in events or [{"type": "token", "token": "agent "},
                             {"type": "token", "token": "answer"},
                             _final("agent answer")]:
            yield ev
    app.state.agent.run_iterating = run


# --------------------------------------------------------------------------- #
# Model naming, listing, auth and error shapes
# --------------------------------------------------------------------------- #
def test_models_list_both_shapes_and_profile_visibility():
    app, cfg, db = build_app(model_aliases=json.dumps({"coder": MODEL}))
    db.create_user("bob", password_hash=hash_password("bobpw12345"), role="user")
    db.create_agent("Research Bot", capabilities=["web_api"], enabled=True)
    db.create_agent("Hidden", capabilities=["web_api"], enabled=False)
    with TestClient(app) as c:
        admin, bob = _bearer(c), _bearer(c, "bob", "bobpw12345")
        r = c.get("/v1/models", headers=bob)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "list"
        ids = [m["id"] for m in body["data"]]
        assert MODEL in ids and "coder" in ids and f"raw/{MODEL}" in ids and "raw/coder" in ids
        assert "agent" in ids and "agent/research-bot" in ids and "agent/hidden" not in ids
        assert all(m["object"] == "model" for m in body["data"])
        assert "agent/hidden" in [m["id"] for m in c.get("/v1/models", headers=admin).json()["data"]]
        ant = c.get("/v1/models", headers={**bob, "anthropic-version": "2023-06-01"}).json()
        assert ant["has_more"] is False and ant["data"][0]["type"] == "model"
        assert c.get("/v1/models/raw/coder", headers=bob).json()["id"] == "raw/coder"
        missing = c.get("/v1/models/nope", headers=bob)
        assert missing.status_code == 404 and missing.json()["error"]["code"] == "model_not_found"


def test_errors_are_provider_shaped():
    app, cfg, db = build_app()
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
        assert r.status_code == 401
        assert r.json()["error"]["type"] == "authentication_error"
        assert r.json()["error"]["message"]
        r = c.post("/v1/messages", json={"model": "m", "max_tokens": 5,
                                         "messages": [{"role": "user", "content": "x"}]})
        assert r.status_code == 401
        assert r.json() == {"type": "error", "error": {"type": "authentication_error",
                                                       "message": r.json()["error"]["message"]}}
        auth = _bearer(c)
        bad = c.post("/v1/chat/completions", content=b"{nope", headers=auth)
        assert bad.status_code == 400 and bad.json()["error"]["type"] == "invalid_request_error"
        assert c.post("/v1/chat/completions", json={"messages": []},
                      headers=auth).json()["error"]["param"] == "messages"
        r = c.post("/v1/messages", json={"messages": [{"role": "user", "content": "x"}]},
                   headers=auth)
        assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
        assert "max_tokens" in r.json()["error"]["message"]
        r = c.post("/v1/responses", json={"input": "hi", "previous_response_id": "resp_1"},
                   headers=auth)
        assert r.status_code == 400 and "stateless" in r.json()["error"]["message"]


def test_raw_mode_model_must_be_served_and_can_be_disabled(monkeypatch):
    app, cfg, db = build_app()
    _caps(monkeypatch)
    FakeModel(_content("hi")).install(app)
    msgs = [{"role": "user", "content": "hello"}]
    with TestClient(app) as c:
        auth = _bearer(c)
        r = c.post("/v1/chat/completions", json={"model": "raw/other/model", "messages": msgs},
                   headers=auth)
        assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"
        r = c.post("/v1/messages", json={"model": "raw/gpt-x", "max_tokens": 9,
                                         "messages": msgs}, headers=auth)
        assert r.status_code == 404 and r.json()["error"]["type"] == "not_found_error"
        # V1_DEFAULT_MODE=raw makes a plain name raw (and strict).
        cfg.v1_default_mode = "raw"
        assert c.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": msgs},
                      headers=auth).status_code == 404
        r = c.post("/v1/chat/completions", json={"model": MODEL, "messages": msgs}, headers=auth)
        assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "hi"
        cfg.v1_raw_enabled = False
        r = c.post("/v1/chat/completions", json={"model": "raw/" + MODEL, "messages": msgs},
                   headers=auth)
        assert r.status_code == 403 and r.json()["error"]["type"] == "permission_error"


def test_not_ready_and_admission_are_503_provider_shaped():
    app, cfg, db = build_app()
    msgs = [{"role": "user", "content": "hello"}]
    with TestClient(app) as c:
        auth = _bearer(c)
        app.state.model_manager.status = "loading"
        r = c.post("/v1/chat/completions", json={"messages": msgs}, headers=auth)
        assert r.status_code == 503 and r.json()["error"]["type"] == "server_error"
        app.state.model_manager.status = "ready"

        async def busy(*a, **kw):
            raise GenerationBusy("8 generations already queued; try again shortly")

        app.state.ctx.generation_gate.acquire = busy
        r = c.post("/v1/chat/completions", json={"messages": msgs, "stream": True}, headers=auth)
        assert r.status_code == 503 and r.headers["retry-after"] == "5"
        assert r.json()["error"]["code"] == "overloaded"
        r = c.post("/v1/messages", json={"max_tokens": 5, "messages": msgs}, headers=auth)
        assert r.status_code == 503 and r.json()["error"]["type"] == "overloaded_error"
        r = c.post("/v1/responses", json={"input": "hi"}, headers=auth)
        assert r.status_code == 503 and "error" in r.json()


# --------------------------------------------------------------------------- #
# Agent mode (today's /v1/chat/completions behaviour)
# --------------------------------------------------------------------------- #
def test_old_client_without_model_gets_the_agent():
    app, cfg, db = build_app()
    seen: list = []
    _fake_agent(app, seen)
    msgs = [{"role": "system", "content": "ignored"}, {"role": "user", "content": "first"},
            {"role": "assistant", "content": "earlier"}, {"role": "user", "content": "now"}]
    with TestClient(app) as c:
        auth = _bearer(c)
        r = c.post("/v1/chat/completions", json={"messages": msgs}, headers=auth)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "chat.completion" and body["model"] == MODEL
        assert body["choices"][0]["message"] == {"role": "assistant", "content": "agent answer"}
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["x_tools_used"] == [] and body["usage"]["prompt_tokens"] == 11
        assert seen[-1]["message"] == "now" and seen[-1]["interactive"] is False
        assert seen[-1]["history"] == [{"role": "user", "content": "first"},
                                       {"role": "assistant", "content": "earlier"}]
        r = c.post("/v1/chat/completions", json={"messages": msgs, "stream": True}, headers=auth)
        frames = _sse_data(r.text)
        assert frames[-1] == "[DONE]"
        assert frames[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
        text = "".join(f["choices"][0]["delta"].get("content") or "" for f in frames[:-1])
        assert text == "agent answer"
        assert frames[-2]["choices"][0]["finish_reason"] == "stop"
        # Any placeholder model name still reaches the agent by default.
        r = c.post("/v1/chat/completions", json={"model": "gpt-3.5-turbo", "messages": msgs},
                   headers=auth)
        assert r.status_code == 200 and r.json()["model"] == "gpt-3.5-turbo"
        assert c.post("/v1/chat/completions", json={"messages": [
            {"role": "assistant", "content": "x"}]}, headers=auth).status_code == 400
    rows = [dict(r) for r in db.execute("SELECT * FROM metrics").fetchall()]
    assert any(r["endpoint"] == "v1_chat" for r in rows)


def test_agent_profile_naming_is_gated():
    app, cfg, db = build_app()
    db.create_user("bob", password_hash=hash_password("bobpw12345"), role="user")
    cfg.role_capabilities = json.dumps({"user": ["web_api"]})
    off = db.create_agent("off", capabilities=["web_api"], enabled=False)
    db.create_agent("Coder", capabilities=["web_api", "code_exec"], enabled=True)
    built: list = []
    seen: list = []

    class ProfileAgent:
        def __init__(self, turn_cfg):
            self.config = turn_cfg

        async def run_iterating(self, message, history, conversation_id=None, max_tokens=None,
                                temperature=None, cancel=None, **kw):
            seen.append(current_turn().agent_profile)
            yield {"type": "tool_call", "id": "c1", "name": "web_search", "args": {"q": "x"},
                   "step": 1}
            yield {"type": "tool_result", "id": "c1", "name": "web_search", "result": "r",
                   "error": None, "step": 1}
            yield _final("from profile", tools_used=["web_search"])

    def factory(turn_cfg, profile):
        built.append((turn_cfg, profile))
        return ProfileAgent(turn_cfg)

    app.state.agent_factory = factory
    msgs = [{"role": "user", "content": "go"}]
    with TestClient(app) as c:
        admin, bob = _bearer(c), _bearer(c, "bob", "bobpw12345")
        r = c.post("/v1/chat/completions", json={"model": f"agent/{off['id']}", "messages": msgs},
                   headers=bob)
        assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"
        assert c.post("/v1/chat/completions", json={"model": "agent/off", "messages": msgs},
                      headers=admin).status_code == 200
        r = c.post("/v1/chat/completions", json={"model": "agent/coder", "messages": msgs,
                                                 "x_include_trace": True}, headers=bob)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["choices"][0]["message"]["content"] == "from profile"
        assert body["x_tools_used"] == ["web_search"]
        assert [e["type"] for e in body["x_trace"]] == ["tool_call", "tool_result"]
        turn_cfg, profile = built[-1]
        assert profile["capabilities"] == ["web_api"]          # code_exec cut by the role
        code = set(CAPABILITY_GROUPS["code_exec"]["tools"])
        assert not (code & set(turn_cfg.agent_tools.split(",")))
        assert seen[-1]["capabilities"] == ["web_api"]
        # Without x_include_trace there is no trace field.
        r = c.post("/v1/messages", json={"model": "agent/coder", "max_tokens": 50,
                                         "messages": msgs}, headers=bob)
        assert r.status_code == 200 and "x_trace" not in r.json()
        assert r.json()["content"] == [{"type": "text", "text": "from profile"}]


# --------------------------------------------------------------------------- #
# Raw mode: OpenAI chat completions
# --------------------------------------------------------------------------- #
def test_raw_chat_text_reasoning_usage_and_params(monkeypatch):
    app, cfg, db = build_app()
    _caps(monkeypatch)
    fake = FakeModel([{"type": "reasoning", "text": "think "},
                      *_content("Hel", "lo")]).install(app)
    with TestClient(app) as c:
        auth = _bearer(c)
        r = c.post("/v1/chat/completions", headers=auth, json={
            "model": f"raw/{MODEL}", "temperature": 0.2, "top_p": 0.9, "seed": 7,
            "max_completion_tokens": 64,
            "messages": [{"role": "developer", "content": "Be brief."},
                         {"role": "user", "content": [{"type": "text", "text": "hi"}]}]})
        assert r.status_code == 200, r.text
        body = r.json()
        msg = body["choices"][0]["message"]
        assert msg["content"] == "Hello" and msg["reasoning_content"] == "think "
        assert body["usage"]["prompt_tokens"] == 40 and body["usage"]["total_tokens"] == 47
        assert body["usage"]["prompt_tokens_details"]["cached_tokens"] == 12
        call = fake.calls[-1]
        assert call["messages"] == [{"role": "system", "content": "Be brief."},
                                    {"role": "user", "content": "hi"}]
        assert call["extra"]["seed"] == 7 and call["extra"]["top_p"] == 0.9
        assert call["temperature"] == 0.2 and call["max_tokens"] == 64
        assert "tools" not in call["extra"] and "parse_tool_calls" not in call["extra"]
        r = c.post("/v1/chat/completions", headers=auth, json={
            "model": f"raw/{MODEL}", "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "hi"}]})
        frames = _sse_data(r.text)
        assert frames[-1] == "[DONE]"
        deltas = [f["choices"][0]["delta"] for f in frames[:-2]]
        assert "".join(d.get("reasoning_content") or "" for d in deltas) == "think "
        assert "".join(d.get("content") or "" for d in deltas) == "Hello"
        assert frames[-2]["choices"] == [] and frames[-2]["usage"]["completion_tokens"] == 7
        assert frames[-3]["choices"][0]["finish_reason"] == "stop"


def test_raw_native_tool_calls_stream_and_round_trip(monkeypatch):
    app, cfg, db = build_app()
    _caps(monkeypatch, "native_tool_calls")
    fake = FakeModel([
        {"type": "tool_call_delta", "index": 0, "id": "call_abc", "name": "get_weather",
         "arguments": '{"city": ', "source": "native", "done": False},
        {"type": "tool_call_delta", "index": 0, "id": "", "name": "",
         "arguments": '"Paris"}', "source": "native", "done": False},
        {"type": "finish", "reason": "tool_calls"}]).install(app)
    history = [{"role": "user", "content": "weather?"},
               {"role": "assistant", "content": None, "tool_calls": [
                   {"id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'}}]},
               {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
               {"role": "user", "content": "and Paris?"}]
    with TestClient(app) as c:
        auth = _bearer(c)
        r = c.post("/v1/chat/completions", headers=auth, json={
            "model": f"raw/{MODEL}", "stream": True, "tools": [WEATHER], "tool_choice": "auto",
            "messages": history})
        frames = _sse_data(r.text)[:-1]
        calls = [d for f in frames for d in f["choices"][0]["delta"].get("tool_calls") or []]
        assert calls[0] == {"index": 0, "id": "call_abc", "type": "function",
                            "function": {"name": "get_weather", "arguments": ""}}
        args = "".join(d["function"].get("arguments") or "" for d in calls)
        assert json.loads(args) == {"city": "Paris"}
        assert frames[-1]["choices"][0]["finish_reason"] == "tool_calls"
        sent = fake.calls[-1]
        assert sent["extra"]["tools"] == [WEATHER] and sent["extra"]["tool_choice"] == "auto"
        assert sent["extra"]["parse_tool_calls"] is True
        # Native path: history keeps OpenAI tool_calls and role "tool" messages.
        assert sent["messages"][1]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert sent["messages"][2] == {"role": "tool", "tool_call_id": "call_1",
                                       "name": "get_weather", "content": "sunny"}
        r = c.post("/v1/chat/completions", headers=auth, json={
            "model": f"raw/{MODEL}", "tools": [WEATHER], "messages": history})
        choice = r.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls" and choice["message"]["content"] is None
        tc = choice["message"]["tool_calls"][0]
        assert tc["id"] == "call_abc" and json.loads(tc["function"]["arguments"]) == {
            "city": "Paris"}


def test_raw_prompt_protocol_tool_round_trip(monkeypatch):
    app, cfg, db = build_app()
    _caps(monkeypatch)                         # no native tool calls: prompt protocol
    fake = FakeModel(_content('{"tool": "get_weather", ', '"args": {"city": "Paris"}}')
                     ).install(app)
    history = [{"role": "system", "content": "You are terse."},
               {"role": "user", "content": "weather in Rome?"},
               {"role": "assistant", "content": "", "tool_calls": [
                   {"id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'}}]},
               {"role": "tool", "tool_call_id": "call_1", "content": "sunny, 24C"},
               {"role": "user", "content": "and Paris?"}]
    with TestClient(app) as c:
        auth = _bearer(c)
        r = c.post("/v1/chat/completions", headers=auth, json={
            "model": f"raw/{MODEL}", "tools": [WEATHER], "tool_choice": "required",
            "messages": history})
        assert r.status_code == 200, r.text
        choice = r.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        call = choice["message"]["tool_calls"][0]
        assert call["function"]["name"] == "get_weather" and call["id"].startswith("call_")
        assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}
        assert not choice["message"]["content"]
        sent = fake.calls[-1]
        system = sent["messages"][0]["content"]
        assert system.startswith("You are terse.")
        assert '{"tool": "tool_name", "args"' in system and "- get_weather: Current weather" in system
        assert "city (string)" in system and "You must call one of the tools" in system
        roles = [m["role"] for m in sent["messages"]]
        assert roles == ["system", "user", "assistant", "user"]   # merged, alternating
        assert json.loads(sent["messages"][2]["content"]) == {"tool": "get_weather",
                                                              "args": {"city": "Rome"}}
        assert sent["messages"][3]["content"].startswith("TOOL RESULT [get_weather]:\nsunny")
        assert sent["messages"][3]["content"].endswith("and Paris?")
        assert "tools" not in sent["extra"] and sent["extra"]["known_tools"] == {"get_weather"}
        # Streaming: the call arrives as one tool_calls entry with valid JSON args.
        r = c.post("/v1/chat/completions", headers=auth, json={
            "model": f"raw/{MODEL}", "tools": [WEATHER], "messages": history, "stream": True})
        frames = _sse_data(r.text)[:-1]
        assert not "".join(f["choices"][0]["delta"].get("content") or "" for f in frames)
        calls = [d for f in frames for d in f["choices"][0]["delta"].get("tool_calls") or []]
        assert calls[0]["function"]["name"] == "get_weather"
        assert json.loads("".join(d["function"]["arguments"] for d in calls)) == {"city": "Paris"}


def test_raw_stop_sequences_json_mode_and_vision(monkeypatch):
    app, cfg, db = build_app()
    _caps(monkeypatch)
    FakeModel(_content("one two ST", "OP three"), _content("```json\n{'a': 1,}\n```")
              ).install(app)
    with TestClient(app) as c:
        auth = _bearer(c)
        r = c.post("/v1/messages", headers=auth, json={
            "model": f"raw/{MODEL}", "max_tokens": 20, "stop_sequences": ["STOP"],
            "messages": [{"role": "user", "content": "count"}]})
        body = r.json()
        assert body["content"] == [{"type": "text", "text": "one two "}]
        assert body["stop_reason"] == "stop_sequence" and body["stop_sequence"] == "STOP"
        r = c.post("/v1/chat/completions", headers=auth, json={
            "model": f"raw/{MODEL}", "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": "json please"}]})
        assert json.loads(r.json()["choices"][0]["message"]["content"]) == {"a": 1}
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        r = c.post("/v1/chat/completions", headers=auth, json={
            "model": f"raw/{MODEL}", "messages": [{"role": "user", "content": [
                {"type": "text", "text": "what is this"}, image]}]})
        assert r.status_code == 400 and "images" in r.json()["error"]["message"]
        r = c.post("/v1/messages", headers=auth, json={
            "model": f"raw/{MODEL}", "max_tokens": 5, "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": "AAAA"}}]}]})
        assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"


# --------------------------------------------------------------------------- #
# Responses API
# --------------------------------------------------------------------------- #
def test_responses_text_tools_and_stream(monkeypatch):
    app, cfg, db = build_app()
    _caps(monkeypatch)
    fake = FakeModel(_content("Hi there"),
                     _content('{"tool": "get_weather", "args": {"city": "Oslo"}}')).install(app)
    tool = {"type": "function", "name": "get_weather", "description": "Weather.",
            "parameters": WEATHER["function"]["parameters"]}
    with TestClient(app) as c:
        auth = _bearer(c)
        r = c.post("/v1/responses", headers=auth, json={
            "model": f"raw/{MODEL}", "instructions": "Be nice.", "input": "hello",
            "max_output_tokens": 30})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "response" and body["status"] == "completed"
        assert body["output"][0]["type"] == "message"
        assert body["output"][0]["content"][0] == {"type": "output_text", "text": "Hi there",
                                                   "annotations": []}
        assert body["output_text"] == "Hi there"
        assert body["usage"]["input_tokens"] == 40 and body["usage"]["output_tokens"] == 7
        assert body["usage"]["input_tokens_details"]["cached_tokens"] == 12
        assert fake.calls[-1]["messages"][0] == {"role": "system", "content": "Be nice."}
        items = [{"role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
                 {"type": "function_call", "call_id": "call_9", "name": "get_weather",
                  "arguments": '{"city": "Rome"}'},
                 {"type": "function_call_output", "call_id": "call_9", "output": "rainy"},
                 {"role": "user", "content": "and Oslo?"}]
        r = c.post("/v1/responses", headers=auth, json={
            "model": f"raw/{MODEL}", "input": items, "tools": [tool, {"type": "web_search"}],
            "stream": True})
        events = _sse_events(r.text)
        names = [n for n, _ in events]
        assert names[0] == "response.created" and names[-1] == "response.completed"
        assert [d["sequence_number"] for _, d in events] == list(range(len(events)))
        added = [d for n, d in events if n == "response.output_item.added"]
        assert added[0]["item"]["type"] == "function_call"
        assert added[0]["item"]["name"] == "get_weather"
        args = "".join(d["delta"] for n, d in events
                       if n == "response.function_call_arguments.delta")
        assert json.loads(args) == {"city": "Oslo"}
        done = events[-1][1]["response"]
        assert done["output"][0]["type"] == "function_call"
        assert done["output"][0]["status"] == "completed"
        assert "TOOL RESULT [get_weather]:\nrainy" in fake.calls[-1]["messages"][-1]["content"]


def test_responses_stream_text_and_reasoning(monkeypatch):
    app, cfg, db = build_app()
    _caps(monkeypatch)
    FakeModel([{"type": "content", "text": "<think>hmm</think>Answer"},
               {"type": "finish", "reason": "length"}]).install(app)
    with TestClient(app) as c:
        auth = _bearer(c)
        r = c.post("/v1/responses", headers=auth, json={
            "model": f"raw/{MODEL}", "input": "q", "stream": True})
        events = _sse_events(r.text)
        names = [n for n, _ in events]
        assert "response.reasoning_summary_text.delta" in names
        text = "".join(d["delta"] for n, d in events if n == "response.output_text.delta")
        assert text == "Answer"
        assert names[-1] == "response.incomplete"
        final = events[-1][1]["response"]
        assert final["incomplete_details"] == {"reason": "max_output_tokens"}
        assert [i["type"] for i in final["output"]] == ["reasoning", "message"]


# --------------------------------------------------------------------------- #
# Anthropic Messages
# --------------------------------------------------------------------------- #
ANT_TOOL = {"name": "get_weather", "description": "Weather for a city.",
            "input_schema": WEATHER["function"]["parameters"]}


def test_anthropic_tool_use_round_trip(monkeypatch):
    app, cfg, db = build_app()
    _caps(monkeypatch)
    fake = FakeModel(_content('Checking.\n{"tool": "get_weather", "args": {"city": "Paris"}}')
                     ).install(app)
    messages = [
        {"role": "user", "content": "weather in Rome?"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "old", "signature": "sig"},
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
             "input": {"city": "Rome"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1",
             "content": [{"type": "text", "text": "sunny"}]},
            {"type": "text", "text": "Now Paris."}]}]
    with TestClient(app) as c:
        auth = _bearer(c)
        r = c.post("/v1/messages", headers=auth, json={
            "model": f"raw/{MODEL}", "max_tokens": 100, "system": [
                {"type": "text", "text": "You help.", "cache_control": {"type": "ephemeral"}}],
            "tools": [ANT_TOOL, {"type": "bash_20250124", "name": "bash"}],
            "tool_choice": {"type": "any"}, "messages": messages})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["type"] == "message" and body["stop_reason"] == "tool_use"
        kinds = [b["type"] for b in body["content"]]
        assert "tool_use" in kinds
        use = next(b for b in body["content"] if b["type"] == "tool_use")
        assert use["name"] == "get_weather" and use["input"] == {"city": "Paris"}
        assert use["id"].startswith("toolu_")
        assert body["usage"] == {"input_tokens": 28, "output_tokens": 7,
                                 "cache_creation_input_tokens": 0, "cache_read_input_tokens": 12}
        sent = fake.calls[-1]["messages"]
        assert sent[0]["content"].startswith("You help.") and "bash" not in sent[0]["content"]
        assert "old" not in json.dumps(sent)                       # thinking dropped
        assert "Let me check.\n{\"tool\": \"get_weather\"" in sent[2]["content"]
        assert sent[3]["content"] == "TOOL RESULT [get_weather]:\nsunny\n\nNow Paris."
        r = c.post("/v1/messages", headers=auth, json={
            "model": f"raw/{MODEL}", "max_tokens": 100, "tools": [ANT_TOOL],
            "messages": messages, "stream": True})
        events = _sse_events(r.text)
        names = [n for n, _ in events]
        assert names[:2] == ["message_start", "ping"] and names[-1] == "message_stop"
        starts = [d["content_block"] for n, d in events if n == "content_block_start"]
        tool_block = next(b for b in starts if b["type"] == "tool_use")
        assert tool_block["name"] == "get_weather" and tool_block["input"] == {}
        partial = "".join(d["delta"]["partial_json"] for n, d in events
                          if n == "content_block_delta"
                          and d["delta"]["type"] == "input_json_delta")
        assert json.loads(partial) == {"city": "Paris"}
        delta = next(d for n, d in events if n == "message_delta")
        assert delta["delta"]["stop_reason"] == "tool_use"
        assert delta["usage"]["output_tokens"] == 7
        assert names.count("content_block_start") == names.count("content_block_stop")


def test_anthropic_thinking_blocks_only_when_enabled(monkeypatch):
    app, cfg, db = build_app()
    _caps(monkeypatch)
    fake = FakeModel([{"type": "reasoning", "text": "pondering"}, *_content("Done.")]
                     ).install(app)
    base = {"model": f"raw/{MODEL}", "max_tokens": 64,
            "messages": [{"role": "user", "content": "think"}]}
    with TestClient(app) as c:
        auth = _bearer(c)
        body = c.post("/v1/messages", headers=auth, json=base).json()
        assert body["content"] == [{"type": "text", "text": "Done."}]
        assert body["stop_reason"] == "end_turn"
        on = dict(base, thinking={"type": "enabled", "budget_tokens": 1024})
        body = c.post("/v1/messages", headers=auth, json=on).json()
        assert body["content"][0] == {"type": "thinking", "thinking": "pondering", "signature": ""}
        assert body["content"][1] == {"type": "text", "text": "Done."}
        assert fake.calls[-1]["extra"]["enable_thinking"] is True
        events = _sse_events(c.post("/v1/messages", headers=auth,
                                    json=dict(on, stream=True)).text)
        deltas = [d["delta"] for n, d in events if n == "content_block_delta"]
        assert deltas[0] == {"type": "thinking_delta", "thinking": "pondering"}
        assert deltas[1] == {"type": "text_delta", "text": "Done."}
        indexes = [d["index"] for n, d in events if n == "content_block_start"]
        assert indexes == [0, 1]
        events = _sse_events(c.post("/v1/messages", headers=auth,
                                    json=dict(base, stream=True)).text)
        assert all(d["delta"]["type"] != "thinking_delta" for n, d in events
                   if n == "content_block_delta")


def test_anthropic_count_tokens_and_api_keys(monkeypatch):
    app, cfg, db = build_app()
    _caps(monkeypatch)
    FakeModel(_content("ok")).install(app)
    with TestClient(app) as c:
        auth = {**_bearer(c), "anthropic-version": "2023-06-01"}
        small = c.post("/v1/messages/count_tokens", headers=auth, json={
            "model": "anything", "messages": [{"role": "user", "content": "hi"}]})
        assert small.status_code == 200, small.text
        big = c.post("/v1/messages/count_tokens", headers=auth, json={
            "model": "anything", "tools": [ANT_TOOL],
            "messages": [{"role": "user", "content": "hi"}]})
        assert big.json()["input_tokens"] > small.json()["input_tokens"] > 0
        bad = c.post("/v1/messages", headers={"x-api-key": "wrong"}, json={
            "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]})
        assert bad.status_code == 401 and bad.json()["type"] == "error"
        try:  # API tokens are the tokens-wire feature; x-api-key only takes those.
            from local_llm.api_tokens import create_api_token
        except ImportError:
            return
        admin = db.get_user_by_username("admin")
        chat_key, _row = create_api_token(db, admin, "claude-code", ["chat"], config=cfg)
        r = c.post("/v1/messages", headers={"x-api-key": chat_key}, json={
            "model": f"raw/{MODEL}", "max_tokens": 5,
            "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200 and r.json()["content"][0]["text"] == "ok"
        status_key, _row = create_api_token(db, admin, "monitor", ["status"], config=cfg)
        r = c.post("/v1/messages", headers={"x-api-key": status_key}, json={
            "model": f"raw/{MODEL}", "max_tokens": 5,
            "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 403 and r.json()["error"]["type"] == "permission_error"
        r = c.post("/v1/chat/completions", headers={"Authorization": f"Bearer {status_key}"},
                   json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 403 and r.json()["error"]["type"] == "permission_error"


def test_parse_messages_translation_unit():
    creq = compat_parse_messages({
        "model": "m", "max_tokens": 10, "stop_sequences": ["X"],
        "tool_choice": {"type": "tool", "name": "get_weather",
                        "disable_parallel_tool_use": True},
        "tools": [ANT_TOOL],
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}},
            {"type": "text", "text": "look"}]}]})
    assert creq.tool_choice == {"type": "function", "function": {"name": "get_weather"}}
    assert creq.parallel_tool_calls is False and creq.stop == ["X"]
    assert creq.messages[0]["content"][0] == {"type": "image_url",
                                              "image_url": {"url": "https://x/y.png"}}
    assert creq.call_prefix == "toolu_" and creq.reasoning is False


def test_agent_mode_on_responses_and_messages(monkeypatch):
    app, cfg, db = build_app()
    seen: list = []
    _fake_agent(app, seen, events=[
        {"type": "think_token", "token": "plan", "step": 1},
        {"type": "tool_call", "id": "c1", "name": "calculator", "args": {"e": "1+1"},
         "step": 1},
        {"type": "tool_result", "id": "c1", "name": "calculator", "result": "2",
         "error": None, "step": 1},
        {"type": "token", "token": "It is 2."},
        _final("It is 2.", tools_used=["calculator"], reasoning="plan")])
    with TestClient(app) as c:
        auth = _bearer(c)
        body = c.post("/v1/responses", headers=auth, json={"input": "1+1?"}).json()
        assert body["output_text"] == "It is 2." and body["x_tools_used"] == ["calculator"]
        assert body["output"][0]["type"] == "reasoning"
        events = _sse_events(c.post("/v1/messages", headers=auth, json={
            "model": "agent", "max_tokens": 50, "stream": True, "x_include_trace": True,
            "thinking": {"type": "enabled", "budget_tokens": 100},
            "messages": [{"role": "user", "content": "1+1?"}]}).text)
        names = [n for n, _ in events]
        assert names.count("x_trace") == 2 and names[-1] == "message_stop"
        deltas = [d["delta"] for n, d in events if n == "content_block_delta"]
        assert deltas == [{"type": "thinking_delta", "thinking": "plan"},
                          {"type": "text_delta", "text": "It is 2."}]
        assert seen[-1]["message"] == "1+1?"


def test_model_failure_is_provider_shaped(monkeypatch):
    app, cfg, db = build_app()
    _caps(monkeypatch)

    async def broken(*a, **kw):
        raise RuntimeError("model server returned 500: boom")
        yield  # pragma: no cover

    app.state.ctx.model_client._stream_core = broken
    msgs = [{"role": "user", "content": "x"}]
    with TestClient(app) as c:
        auth = _bearer(c)
        r = c.post("/v1/chat/completions", headers=auth,
                   json={"model": f"raw/{MODEL}", "messages": msgs})
        assert r.status_code == 503 and "boom" in r.json()["error"]["message"]
        frames = _sse_data(c.post("/v1/chat/completions", headers=auth, json={
            "model": f"raw/{MODEL}", "messages": msgs, "stream": True}).text)
        assert any(isinstance(f, dict) and "error" in f for f in frames)
        assert frames[-1] == "[DONE]"
        events = _sse_events(c.post("/v1/messages", headers=auth, json={
            "model": f"raw/{MODEL}", "max_tokens": 5, "messages": msgs, "stream": True}).text)
        assert events[-1][0] == "error" and events[-1][1]["type"] == "error"
        assert app.state.ctx.generation_gate.running == 0


def test_stream_releases_admission_slot(monkeypatch):
    app, cfg, db = build_app()
    _caps(monkeypatch)
    FakeModel(_content("a", "b")).install(app)
    gate = app.state.ctx.generation_gate
    with TestClient(app) as c:
        auth = _bearer(c)
        for path, body in (("/v1/chat/completions", {"model": f"raw/{MODEL}", "stream": True,
                                                     "messages": [{"role": "user",
                                                                   "content": "x"}]}),
                           ("/v1/messages", {"model": f"raw/{MODEL}", "stream": True,
                                             "max_tokens": 5,
                                             "messages": [{"role": "user", "content": "x"}]})):
            r = c.post(path, headers=auth, json=body)
            assert r.status_code == 200
            assert gate.running == 0, path
    rows = {dict(r)["endpoint"] for r in db.execute("SELECT * FROM metrics").fetchall()}
    assert {"v1_chat", "v1_messages"} <= rows


def _run_standalone():
    import inspect

    class _MP:
        def __init__(self):
            self.undo = []

        def setattr(self, obj, name, value):
            self.undo.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)

        def restore(self):
            for obj, name, value in reversed(self.undo):
                setattr(obj, name, value)

    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        mp = _MP()
        try:
            if "monkeypatch" in inspect.signature(fn).parameters:
                fn(mp)
            else:
                fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"FAIL  {name}: {exc}")
            failed += 1
        finally:
            mp.restore()
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
