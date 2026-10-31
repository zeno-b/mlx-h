"""Wave 3b end-to-end: tokens, /v1, wiring, the MCP server and attachments.

Each feature agent tested its own side with the other side faked. These tests
drive the seams with only the model faked: a token minted through
/api/auth/tokens authenticates /v1/messages over x-api-key and
/v1/chat/completions over Bearer; the model name the wire helper and the
snippets route hand a client is one /v1 accepts; the MCP server lists exactly
the role- or profile-gated tools and every call goes through the policy and
its audit; the capability catalogue carries the named keys while the legacy
keys expand as before; and an image reaches the model's prompt from the chat
composer, from /v1 content parts and from image_analyze under one vision rule
(vision.served_vision, VISION_MODE included).
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_llm.config as config_mod  # noqa: E402
config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-wave3b-integ-"))

from fastapi.testclient import TestClient  # noqa: E402

import local_llm.openai_compat as oai  # noqa: E402
import local_llm.tools_extra as tx  # noqa: E402
import local_llm.wire as wire_mod  # noqa: E402
from local_llm.api import create_app  # noqa: E402
from local_llm.api_tokens import token_allows_route  # noqa: E402
from local_llm.config import (CAPABILITY_ORDER, Config,  # noqa: E402
                              tools_for_capabilities)
from local_llm.core import ADAPTER_DIR, estimate_tokens  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.llm import GenerationStats  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.policy import BROKER, list_tool_decisions, set_profile_tool_policy  # noqa: E402
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.training import RetrainManager  # noqa: E402

MODEL = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00"
       b"\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05"
       b"\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82")
IMAGE_URL = "data:image/png;base64,iVBORw0KGgo="

# The six legacy keys expand to exactly these tools (HEAD's CAPABILITY_GROUPS);
# saved profiles and roles keep meaning what they meant before Wave 3b.
LEGACY_EXPANSION = {
    "file_ops": ["read_file", "write_file", "edit_file", "list_files", "search_files",
                 "file_info", "create_document"],
    "code_exec": ["run_shell", "run_python", "run_tests"],
    "web_api": ["web_search", "fetch_url"],
    # Phase B (upgrade/ROUTING_AND_TOOLING.md P3): RAG plus the knowledge tools.
    "knowledge": ["search_docs", "kb_overview"],
    "memory": ["remember", "recall_memory", "forget", "recall_feedback"],
    "office365": ["o365_mail", "o365_files", "o365_calendar"],
}
# The twenty capability keys of the tool catalogue (item 10).
NAMED_KEYS = ["web_search", "browse_page", "code_interpreter", "shell", "file_ops", "git",
              "create_document", "spreadsheet", "pdf_extract", "image_analyze",
              "image_generate", "email", "calendar", "contacts", "http_request", "database",
              "knowledge_search", "memory", "task_scheduler", "notify"]


# --------------------------------------------------------------------------- #
# Fakes and helpers
# --------------------------------------------------------------------------- #
class _ScriptedModel:
    """The agent's ModelClient stand-in; every prompt is kept for inspection."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[list[dict]] = []

    def stream(self, messages, *a, **kw):
        self.prompts.append(messages)
        reply = self.replies.pop(0) if self.replies else "done."

        async def gen():
            await asyncio.sleep(0)
            yield reply

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

    def image_parts(self) -> list[dict]:
        out = []
        for prompt in self.prompts:
            for m in prompt:
                content = m.get("content")
                if isinstance(content, list):
                    out.extend(p for p in content
                               if isinstance(p, dict) and p.get("type") == "image_url")
        return out

    def text(self) -> str:
        return "\n".join(m.get("content") if isinstance(m.get("content"), str)
                         else json.dumps(m.get("content"))
                         for prompt in self.prompts for m in prompt)


class _RawCore:
    """Scripted server events in place of ModelClient._stream_core (raw mode)."""

    def __init__(self, text="raw says hi"):
        self.text = text
        self.calls: list[dict] = []

    def install(self, app):
        app.state.ctx.model_client._stream_core = self.core
        return self

    async def core(self, messages, max_tokens=None, temperature=None, stats=None,
                   kind="chat", conversation_id=None, extra=None):
        self.calls.append({"messages": messages, "extra": dict(extra or {})})
        yield {"type": "content", "text": self.text}
        yield {"type": "finish", "reason": "stop"}
        yield {"type": "usage", "usage": {"prompt_tokens": 10, "completion_tokens": 3}}


def build_app(replies=(), **overrides):
    """tests/test_app.py:build_app plus a scripted model behind the real agent."""
    base = dict(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                allow_test_user=False, node_token="secret-node-token", model=MODEL,
                skills_dir=tempfile.mkdtemp(prefix="local-llm-test-skills-"),
                project_dir=tempfile.mkdtemp(prefix="local-llm-test-root-"),
                agent_enabled=True, knowledge_triage=False, fast_path=False,
                incremental_reasoning=False, chunk_large_prompts=False,
                agent_max_steps=3, agent_min_steps=1, resilient_retries=1,
                ready_wait_timeout=0.01, auto_iterate_rounds=0,
                notify_macos=False, checkpoints_enabled=False)
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


class _patched:
    """setattr for the duration of a with-block (works without pytest)."""

    def __init__(self, obj, name, value):
        self.obj, self.name, self.value = obj, name, value

    def __enter__(self):
        self.old = getattr(self.obj, self.name)
        setattr(self.obj, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.obj, self.name, self.old)


def _no_caps():
    # The never-started manager's backend must not decide raw-mode rendering.
    return _patched(oai, "compat_backend_caps", lambda ctx: frozenset())


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


def _token(c: TestClient, cookie: str, scopes=("chat",), name="cli") -> dict:
    r = c.post("/api/auth/tokens", json={"name": name, "scopes": list(scopes)},
               headers={"cookie": cookie})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["token"].startswith("llm_")
    return body


def _rpc(method, mid=1, **params):
    return {"jsonrpc": "2.0", "id": mid, "method": method, "params": params}


def _mcp_session(c: TestClient, headers: dict) -> dict:
    r = c.post("/mcp", json=_rpc("initialize", protocolVersion="2025-06-18",
                                 capabilities={}, clientInfo={"name": "integ"}),
               headers=headers)
    assert r.status_code == 200, r.text
    return {**headers, "Mcp-Session-Id": r.headers["mcp-session-id"],
            "MCP-Protocol-Version": "2025-06-18"}


def _mcp_tools(c: TestClient, headers: dict) -> list[str]:
    names, cursor = [], None
    while True:
        params = {"cursor": cursor} if cursor else {}
        r = c.post("/mcp", json=_rpc("tools/list", 2, **params), headers=headers)
        assert r.status_code == 200, r.text
        result = r.json()["result"]
        names += [t["name"] for t in result["tools"]]
        cursor = result.get("nextCursor")
        if not cursor:
            return names


def _mcp_call(c: TestClient, headers: dict, name: str, args: dict) -> dict:
    r = c.post("/mcp", json=_rpc("tools/call", 3, name=name, arguments=args), headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


# --------------------------------------------------------------------------- #
# Tokens end to end on /v1 (items 2 and 21)
# --------------------------------------------------------------------------- #
def test_minted_token_authenticates_v1_over_x_api_key_and_bearer():
    app, cfg, db, model = build_app(["Agent mode answer."])
    raw = _RawCore("raw answer").install(app)
    with _no_caps(), TestClient(app) as c:
        admin = _cookie(c, "admin", "adminpw123")
        minted = _token(c, admin)
        token = minted["token"]
        assert minted["api_token"]["scopes"] == ["chat"]

        # Anthropic SDK / Claude Code with ANTHROPIC_API_KEY: x-api-key.
        r = c.post("/v1/messages", headers={"x-api-key": token,
                                            "anthropic-version": "2023-06-01"},
                   json={"model": f"raw/{MODEL}", "max_tokens": 16,
                         "messages": [{"role": "user", "content": "hello"}]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["type"] == "message" and body["content"] == [
            {"type": "text", "text": "raw answer"}]
        # OpenAI SDK / Codex / aider: Bearer.
        r = c.post("/v1/chat/completions", headers={"Authorization": f"Bearer {token}"},
                   json={"model": f"raw/{MODEL}",
                         "messages": [{"role": "user", "content": "hello"}]})
        assert r.status_code == 200, r.text
        assert r.json()["choices"][0]["message"]["content"] == "raw answer"
        assert len(raw.calls) == 2
        # Agent mode under the same token runs the real agent, unattended.
        r = c.post("/v1/messages", headers={"x-api-key": token},
                   json={"model": "agent", "max_tokens": 64,
                         "messages": [{"role": "user", "content": "say something"}]})
        assert r.status_code == 200, r.text
        assert "Agent mode answer." in r.json()["content"][-1]["text"]
        r = c.get("/v1/models", headers={"x-api-key": token})
        assert r.status_code == 200 and f"raw/{MODEL}" in [m["id"] for m in r.json()["data"]]

        # Refusals stay provider-shaped. A session token is never read from
        # x-api-key (only llm_ tokens are), so it is simply missing there.
        session = admin.split("=", 1)[1]
        r = c.post("/v1/messages", headers={"x-api-key": session},
                   json={"model": "agent", "max_tokens": 5,
                         "messages": [{"role": "user", "content": "x"}]})
        assert r.status_code == 401 and r.json()["type"] == "error", r.text
        status_tok = _token(c, admin, scopes=["status"], name="menubar")["token"]
        r = c.post("/v1/chat/completions", headers={"Authorization": f"Bearer {status_tok}"},
                   json={"model": "agent", "messages": [{"role": "user", "content": "x"}]})
        assert r.status_code == 403 and "scopes" in r.json()["error"]["message"], r.text
        r = c.post("/v1/messages", headers={"x-api-key": status_tok},
                   json={"model": "agent", "max_tokens": 5,
                         "messages": [{"role": "user", "content": "x"}]})
        assert r.status_code == 403 and r.json()["error"]["type"] == "permission_error", r.text

        # Revoking the token closes /v1 at once.
        r = c.delete(f"/api/auth/tokens/{minted['api_token']['id']}",
                     headers={"cookie": admin})
        assert r.status_code == 200 and r.json()["revoked"]
        r = c.post("/v1/chat/completions", headers={"Authorization": f"Bearer {token}"},
                   json={"model": "agent", "messages": [{"role": "user", "content": "x"}]})
        assert r.status_code == 401 and r.json()["error"]["type"], r.text
    db.close()


def test_chat_scope_reaches_attachments_and_mcp_but_not_admin_routes():
    for path, method in (("/v1/messages", "POST"), ("/api/attachments", "POST"),
                         ("/api/attachments/abc", "GET"), ("/api/attachments/abc", "DELETE"),
                         ("/mcp", "POST"), ("/api/agents/capabilities", "GET")):
        assert token_allows_route(["chat"], method, path), (method, path)
    for path, method in (("/api/config", "POST"), ("/api/users", "GET"),
                         ("/api/audit/tool-decisions", "GET"), ("/mcp", "POST")):
        if path == "/mcp":
            assert not token_allows_route(["status"], method, path)
        else:
            assert not token_allows_route(["chat"], method, path), (method, path)


# --------------------------------------------------------------------------- #
# The wire helper's model names are names /v1 accepts (item 21)
# --------------------------------------------------------------------------- #
def _wire_through(c: TestClient):
    def fake_http(method, url, body=None, headers=None, timeout=10.0):
        path = url.split("://", 1)[-1]
        path = path[path.index("/"):] if "/" in path else "/"
        r = c.request(method, path, json=body, headers=headers or {})
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {}
    return _patched(wire_mod, "_wire_http", fake_http)


def test_wire_model_names_resolve_on_v1():
    app, cfg, db, model = build_app(["Agent here."])
    _RawCore("raw here").install(app)
    with _no_caps(), TestClient(app) as c:
        admin = _cookie(c, "admin", "adminpw123")
        token = _token(c, admin)["token"]
        auth = {"Authorization": f"Bearer {token}"}
        with _wire_through(c):
            picked = wire_mod._wire_server_model("http://testserver", token)
        assert picked == f"raw/{MODEL}", picked
        snippet = c.get("/api/integrations/snippets", params={"client": "claude-code"},
                        headers=auth).json()
        env = {e["name"]: e["value"] for e in snippet["env"]}
        assert env["ANTHROPIC_MODEL"] == snippet["model"]
        assert snippet["model"].startswith("raw/")
        for name in {picked, snippet["model"],
                     wire_mod.build_wiring("codex", "http://testserver", token,
                                           picked)["model"]}:
            r = c.post("/v1/chat/completions", headers=auth, json={
                "model": name, "messages": [{"role": "user", "content": "hi"}]})
            assert r.status_code == 200, (name, r.text)
            assert r.json()["model"] == name
        # With raw access switched off the helper wires clients to the agent,
        # never to a raw/ name the server would refuse.
        cfg.v1_raw_enabled = False
        with _wire_through(c):
            picked = wire_mod._wire_server_model("http://testserver", token)
        assert picked == "agent"
        assert wire_mod.wire_model_name(picked) == "agent"
        snippet = c.get("/api/integrations/snippets", params={"client": "aider"},
                        headers=auth).json()
        assert snippet["model"] == "agent"
        r = c.post("/v1/chat/completions", headers=auth, json={
            "model": "agent", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200 and "Agent here." in \
            r.json()["choices"][0]["message"]["content"]
        r = c.post("/v1/chat/completions", headers=auth, json={
            "model": f"raw/{MODEL}", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 403
    db.close()


# --------------------------------------------------------------------------- #
# MCP server: gated tool set, policy and audit (item 10)
# --------------------------------------------------------------------------- #
def test_mcp_server_lists_role_gated_tools_and_calls_go_through_policy():
    app, cfg, db, model = build_app(
        mcp_server_enabled=True, role_capabilities=json.dumps({"user": ["web_search"]}))
    with TestClient(app) as c:
        admin = _cookie(c, "admin", "adminpw123")
        bob = _add_user(c, admin, "bob")
        bob_token = _token(c, bob)["token"]
        H_bob = _mcp_session(c, {"Authorization": f"Bearer {bob_token}"})
        H_admin = _mcp_session(c, {"cookie": admin})

        registry = app.state.ctx.registry
        hidden = {"final_answer", "delegate_task"}
        every = [n for n in registry.names() if n not in hidden]
        allowed = set(tools_for_capabilities(["web_search"]))
        assert _mcp_tools(c, H_admin) == every     # admin: no ROLE_CAPABILITIES entry
        bob_tools = _mcp_tools(c, H_bob)
        assert bob_tools == [n for n in every if n in allowed], bob_tools
        assert "read_file" in every and "read_file" not in bob_tools

        # A tool outside the role's grant is unknown to bob, even by name.
        out = _mcp_call(c, H_bob, "read_file", {"path": "x.txt"})
        assert out["error"]["code"] == -32602
        # An allowed call runs (policy off: today's allow) ...
        out = _mcp_call(c, H_bob, "calculator", {"expression": "2+2"})
        assert out["result"]["isError"] is False and "4" in out["result"]["content"][0]["text"]
        # ... and with every tool needing approval, the unattended MCP turn is
        # refused by the policy and the refusal is audited for bob.
        cfg.tool_approval_mode = "all"
        try:
            out = _mcp_call(c, H_bob, "calculator", {"expression": "3+3"})
        finally:
            cfg.tool_approval_mode = "off"
        assert out["result"]["isError"] is True
        assert "refused by policy" in out["result"]["content"][0]["text"]
        rows = list_tool_decisions(db, tool="calculator")
        assert rows and rows[0]["decision"] == "deny"
        bob_id = next(u["id"] for u in db.list_users() if u["username"] == "bob")
        assert rows[0]["user_id"] == bob_id

        # Bob's session is his: another user's session id is unknown.
        r = c.post("/mcp", json=_rpc("tools/list", 9),
                   headers={**H_admin, "Mcp-Session-Id": H_bob["Mcp-Session-Id"]})
        assert r.status_code == 404
    db.close()


def test_mcp_server_profile_session_uses_the_profiles_tools_and_policy():
    app, cfg, db, model = build_app(mcp_server_enabled=True)
    profile = db.create_agent("Notes", "memory only", capabilities=["memory"])
    set_profile_tool_policy(db, profile["id"], {"calculator": "deny"})
    with TestClient(app) as c:
        admin = _cookie(c, "admin", "adminpw123")
        token = _token(c, admin)["token"]
        H = _mcp_session(c, {"Authorization": f"Bearer {token}",
                             "X-Agent-Id": profile["id"]})
        names = _mcp_tools(c, H)
        expected = [n for n in tools_for_capabilities(["memory"])
                    if n not in ("final_answer", "delegate_task")]
        assert set(names) <= set(expected) and "remember" in names, names
        assert "read_file" not in names
        # The profile's own rule applies to its MCP session (turn.agent_profile).
        out = _mcp_call(c, H, "calculator", {"expression": "1+1"})
        assert out["result"]["isError"] is True, out
        rows = list_tool_decisions(db, tool="calculator")
        assert rows and rows[0]["decision"] == "deny" and "profile" in rows[0]["rule"]
        # Without the profile the same caller's call is allowed.
        H2 = _mcp_session(c, {"Authorization": f"Bearer {token}"})
        out = _mcp_call(c, H2, "calculator", {"expression": "1+1"})
        assert out["result"]["isError"] is False, out
        # An unknown profile is refused at initialize.
        r = c.post("/mcp", json=_rpc("initialize", protocolVersion="2025-06-18",
                                     capabilities={}, clientInfo={"name": "x"}),
                   headers={"Authorization": f"Bearer {token}", "X-Agent-Id": "nope"})
        assert r.status_code == 404
    db.close()


# --------------------------------------------------------------------------- #
# Capability catalogue (item 10)
# --------------------------------------------------------------------------- #
def test_capability_catalogue_has_the_named_keys_and_legacy_keys_expand_as_before():
    app, cfg, db, model = build_app()
    with TestClient(app) as c:
        admin = _cookie(c, "admin", "adminpw123")
        token = _token(c, admin)["token"]
        r = c.get("/api/agents/capabilities", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200, r.text
        keys = [cap["key"] for cap in r.json()["capabilities"]]
    assert keys[:6] == list(LEGACY_EXPANSION)          # legacy keys first, same order
    assert set(NAMED_KEYS) <= set(keys), set(NAMED_KEYS) - set(keys)
    assert len(set(NAMED_KEYS)) == 20 and keys == [k for k in CAPABILITY_ORDER]
    always = tools_for_capabilities([])
    for key, tools in LEGACY_EXPANSION.items():
        assert tools_for_capabilities([key]) == always + [t for t in tools if t not in always]
    db.close()


# --------------------------------------------------------------------------- #
# Attachments and images: chat, /v1 and image_analyze (item 13)
# --------------------------------------------------------------------------- #
def test_chat_attachment_image_reaches_the_model_prompt():
    app, cfg, db, model = build_app(["A single red pixel."], vision_mode="on")
    with TestClient(app) as c:
        admin = _cookie(c, "admin", "adminpw123")
        token = _token(c, admin)["token"]
        auth = {"Authorization": f"Bearer {token}"}
        # Uploaded with a chat-scoped token, the way a CLI would.
        up = c.post("/api/attachments", content=PNG, headers={**auth, "X-Filename": "p.png"})
        assert up.status_code == 200, up.text
        doc = c.post("/api/attachments", content=b"the doc says forty-two",
                     headers={**auth, "X-Filename": "notes.txt"}).json()
        r = c.post("/api/chat", headers=auth, json={
            "message": "what is in the picture?", "attachments": [up.json()["id"], doc["id"]]})
        assert r.status_code == 200, r.text
        assert "A single red pixel." in json.dumps(r.json())
    parts = model.image_parts()
    assert parts and parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "forty-two" in model.text()
    db.close()


def test_chat_attachment_image_on_a_text_model_is_not_sent():
    app, cfg, db, model = build_app(["I cannot see it."])     # auto + text model
    with TestClient(app) as c:
        admin = _cookie(c, "admin", "adminpw123")
        img = c.post("/api/attachments", content=PNG,
                     headers={"cookie": admin, "X-Filename": "p.png"}).json()["id"]
        r = c.post("/api/chat", headers={"cookie": admin},
                   json={"message": "describe", "attachments": [img]})
        assert r.status_code == 200, r.text
    assert model.image_parts() == []
    assert "cannot see images" in model.text()
    db.close()


def test_v1_content_parts_reach_the_agent_and_the_raw_backend():
    app, cfg, db, model = build_app(["Looks like a dot.", "Also a dot."], vision_mode="on")
    raw = _RawCore("raw sees it").install(app)
    image_part = {"type": "image_url", "image_url": {"url": IMAGE_URL}}
    with _no_caps(), TestClient(app) as c:
        admin = _cookie(c, "admin", "adminpw123")
        token = _token(c, admin)["token"]
        # OpenAI content parts, agent mode.
        r = c.post("/v1/chat/completions", headers={"Authorization": f"Bearer {token}"},
                   json={"model": "agent", "messages": [{"role": "user", "content": [
                       {"type": "text", "text": "what is this?"}, image_part]}]})
        assert r.status_code == 200, r.text
        assert model.image_parts()[-1]["image_url"]["url"] == IMAGE_URL
        # Anthropic image block, agent mode: becomes the same OpenAI part.
        r = c.post("/v1/messages", headers={"x-api-key": token}, json={
            "model": "agent", "max_tokens": 32, "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": "iVBORw0KGgo="}},
                {"type": "text", "text": "and this?"}]}]})
        assert r.status_code == 200, r.text
        assert len(model.image_parts()) == 2
        # Raw mode forwards the part to the backend when the model can see.
        r = c.post("/v1/chat/completions", headers={"Authorization": f"Bearer {token}"},
                   json={"model": f"raw/{MODEL}", "messages": [{"role": "user", "content": [
                       {"type": "text", "text": "raw?"}, image_part]}]})
        assert r.status_code == 200, r.text
        sent = raw.calls[-1]["messages"][-1]["content"]
        assert {"type": "image_url", "image_url": {"url": IMAGE_URL}} in sent
    db.close()

    # VISION_MODE=off: the same request is refused in raw mode, and the agent
    # is told it cannot see instead of being handed the picture.
    app, cfg, db, model = build_app(["No eyes."], vision_mode="off")
    _RawCore().install(app)
    with _no_caps(), TestClient(app) as c:
        admin = _cookie(c, "admin", "adminpw123")
        token = _token(c, admin)["token"]
        r = c.post("/v1/chat/completions", headers={"Authorization": f"Bearer {token}"},
                   json={"model": f"raw/{MODEL}", "messages": [{"role": "user", "content": [
                       {"type": "text", "text": "raw?"}, image_part]}]})
        assert r.status_code == 400 and "images" in r.json()["error"]["message"]
        r = c.post("/v1/chat/completions", headers={"Authorization": f"Bearer {token}"},
                   json={"model": "agent", "messages": [{"role": "user", "content": [
                       {"type": "text", "text": "what is this?"}, image_part]}]})
        assert r.status_code == 200, r.text
    assert model.image_parts() == [] and "cannot see images" in model.text()
    db.close()


def test_image_analyze_follows_the_same_vision_rule():
    root = Path(tempfile.mkdtemp(prefix="llm-test-wave3b-img-"))
    (root / "dot.png").write_bytes(PNG)
    seen: dict = {}

    def fake_complete(registry, messages, max_tokens):
        seen["messages"] = messages
        return "one pixel"

    def registry(mode):
        return ToolRegistry(Config(project_dir=str(root), model=MODEL, vision_mode=mode,
                                   agent_tools=",".join(tools_for_capabilities(
                                       ["image_analyze"])),
                                   notify_macos=False, checkpoints_enabled=False,
                                   skills_dir=tempfile.mkdtemp(prefix="llm-test-sk-")), None)

    with _patched(tx, "_tx_model_complete", fake_complete):
        # auto + a text model behind mlx-lm: the served model cannot see.
        _out, err = registry("auto").call("image_analyze", {"path": "dot.png"}, None)
        assert err and "needs a vision model" in err
        # VISION_MODE=on (an engine auto-detection cannot see) lets it through,
        # with the image sent as an OpenAI image part.
        out, err = registry("on").call("image_analyze", {"path": "dot.png",
                                                         "question": "what?"}, None)
        assert err is None and out == "one pixel"
        parts = seen["messages"][0]["content"]
        assert parts[0] == {"type": "text", "text": "what?"}
        assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
        _out, err = registry("off").call("image_analyze", {"path": "dot.png"}, None)
        assert err and "needs a vision model" in err


def test_agent_vision_rule_matches_served_vision():
    app, cfg, db, model = build_app(vision_mode="on")
    assert app.state.agent.vision_capable() is True
    cfg.vision_mode = "auto"
    assert app.state.agent.vision_capable() is False     # text model on mlx-lm
    db.close()


# --------------------------------------------------------------------------- #
# Entra sign-in no longer re-enables a locally disabled account (item 21)
# --------------------------------------------------------------------------- #
def test_entra_sign_in_keeps_a_disabled_account_disabled():
    app, cfg, db, model = build_app()
    auth = app.state.ctx.auth
    claims = {"oid": "entra-oid-1", "email": "eve@example.com", "name": "Eve"}
    eve = auth.oidc_upsert_user(claims)
    assert not eve.get("disabled")
    db.update_user(eve["id"], disabled=1)
    again = auth.oidc_upsert_user(claims)              # web callback and device poll
    assert again["id"] == eve["id"] and again.get("disabled")
    with _patched(auth, "oidc_exchange", lambda code, state, uri: db.get_user(eve["id"])):
        with TestClient(app) as c:
            r = c.get("/api/auth/oidc/callback", params={"code": "x", "state": "y"},
                      follow_redirects=False)
            assert r.status_code == 403 and "disabled" in r.text
            assert not c.cookies.get(cfg.auth_cookie_name)
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
