"""Tests for the MCP client's Streamable HTTP transport, stdio env sanitising,
MCP capability gating, and this app's own MCP server endpoint (/mcp)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import local_llm.config as config_mod  # noqa: E402
config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-mcp-http-"))

from fastapi.testclient import TestClient  # noqa: E402

from local_llm.config import (  # noqa: E402
    CAPABILITY_ORDER, Config, capabilities_for_role, tools_for_capabilities)
from local_llm.database import Database  # noqa: E402
from local_llm.mcp import (  # noqa: E402
    MCP, MCPHttpServer, MCPManager, parse_mcp_spec)
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.training import RetrainManager  # noqa: E402
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.policy import classify_tool_risk  # noqa: E402
from local_llm.api import create_app  # noqa: E402
from local_llm.core import ADAPTER_DIR  # noqa: E402

# Registries get their own skill library: never the real data/skills.
_SKILLS = tempfile.mkdtemp(prefix="llm-test-mcp-skills-")


# --------------------------------------------------------------------------- #
# An in-process Streamable HTTP MCP server
# --------------------------------------------------------------------------- #
class FakeMCP:
    """State and behaviour switches for one fake server."""

    def __init__(self, *, sse: bool = False, min_version: str | None = None,
                 fail_lists: int = 0) -> None:
        self.sse = sse
        self.min_version = min_version          # reject newer offers with an error
        self.fail_lists = fail_lists            # first N tools/list calls answer 500
        self.sessions: set[str] = set()
        self.requests: list[dict] = []          # (method, headers) for assertions
        self.lists = 0
        self.changed = False                    # after "mutate", a third tool appears
        self.forget_next = False                # drop the session before the next request
        self.deleted: list[str] = []
        self.lock = threading.Lock()

    def tools(self) -> list[dict]:
        tools = [
            {"name": "echo", "description": "Echo the text back",
             "inputSchema": {"type": "object",
                             "properties": {"text": {"type": "string",
                                                     "description": "what to echo"}},
                             "required": ["text"]},
             "annotations": {"readOnlyHint": True}},
            {"name": "mutate", "description": "Changes the tool list",
             "inputSchema": {"type": "object", "properties": {}},
             "annotations": {"destructiveHint": True}},
        ]
        if self.changed:
            tools.append({"name": "added_later", "description": "New",
                          "inputSchema": {"type": "object", "properties": {}}})
        return tools


def _start_fake(fake: FakeMCP) -> tuple[ThreadingHTTPServer, str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # noqa: D401 - quiet
            pass

        def _send(self, code, payload=None, headers=None, sse_events=None):
            self.send_response(code)
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            if sse_events is not None:
                body = "".join(f"event: message\ndata: {json.dumps(e)}\n\n" for e in sse_events)
                raw = body.encode()
                self.send_header("Content-Type", "text/event-stream")
            elif payload is not None:
                raw = json.dumps(payload).encode()
                self.send_header("Content-Type", "application/json")
            else:
                raw = b""
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_DELETE(self):  # noqa: N802
            sid = self.headers.get("Mcp-Session-Id")
            fake.deleted.append(sid)
            fake.sessions.discard(sid)
            self._send(200, {})

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            msg = json.loads(self.rfile.read(length) or b"{}")
            method, mid = msg.get("method"), msg.get("id")
            with fake.lock:
                fake.requests.append({"method": method, "headers": dict(self.headers)})
            assert "text/event-stream" in self.headers.get("Accept", "")
            if method == "initialize":
                asked = msg["params"]["protocolVersion"]
                if fake.min_version and asked > fake.min_version:
                    return self._send(200, {"jsonrpc": "2.0", "id": mid, "error": {
                        "code": -32602, "message": f"unsupported version {asked}"}})
                sid = f"sess-{len(fake.sessions) + 1}-{len(fake.requests)}"
                fake.sessions.add(sid)
                return self._send(200, {"jsonrpc": "2.0", "id": mid, "result": {
                    "protocolVersion": asked, "capabilities": {"tools": {"listChanged": True}},
                    "serverInfo": {"name": "fake-http", "version": "1"}}},
                    headers={"Mcp-Session-Id": sid})
            sid = self.headers.get("Mcp-Session-Id")
            if fake.forget_next:
                fake.forget_next = False
                fake.sessions.discard(sid)
            if sid not in fake.sessions:
                return self._send(404, {"error": "unknown session"})
            if mid is None:
                return self._send(202)
            if method == "tools/list":
                fake.lists += 1
                if fake.fail_lists > 0:
                    fake.fail_lists -= 1
                    return self._send(500, {"error": "boom"})
                tools = fake.tools()
                cursor = (msg.get("params") or {}).get("cursor")
                # Two pages: the first tool, then the rest.
                if cursor is None:
                    result = {"tools": tools[:1], "nextCursor": "page2"}
                else:
                    result = {"tools": tools[1:]}
                reply = {"jsonrpc": "2.0", "id": mid, "result": result}
            elif method == "tools/call":
                name = msg["params"]["name"]
                args = msg["params"].get("arguments") or {}
                if name == "echo":
                    reply = {"jsonrpc": "2.0", "id": mid, "result": {
                        "content": [{"type": "text", "text": f"echo: {args.get('text')}"}]}}
                elif name == "mutate":
                    fake.changed = True
                    reply = {"jsonrpc": "2.0", "id": mid, "result": {
                        "content": [{"type": "text", "text": "mutated"}]}}
                else:
                    reply = {"jsonrpc": "2.0", "id": mid,
                             "error": {"code": -32601, "message": "no such tool"}}
            else:
                reply = {"jsonrpc": "2.0", "id": mid,
                         "error": {"code": -32601, "message": "method not found"}}
            if fake.sse:
                events = [{"jsonrpc": "2.0", "method": "notifications/progress",
                           "params": {"progress": 1}}]
                if method == "tools/call" and fake.changed:
                    events.append({"jsonrpc": "2.0",
                                   "method": "notifications/tools/list_changed"})
                events.append(reply)
                return self._send(200, sse_events=events)
            return self._send(200, reply)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/mcp"


def _http_spec(url: str, name: str = "fakehttp", **extra) -> str:
    return json.dumps([{"name": name, "url": url, "type": "http", **extra}])


# --------------------------------------------------------------------------- #
# Client: Streamable HTTP
# --------------------------------------------------------------------------- #
def test_http_json_replies_session_header_and_pagination():
    fake = FakeMCP()
    server, url = _start_fake(fake)
    manager = MCPManager()
    try:
        manager.configure(_http_spec(url, headers={"X-Api-Key": "${MCPT_KEY}"}), timeout=10)
        os.environ["MCPT_KEY"] = "k-123"
        specs = manager.tools()
        names = [s.local_name for s in specs]
        # Both pages were followed.
        assert names == ["mcp_fakehttp_echo", "mcp_fakehttp_mutate"], names
        echo = specs[0]
        assert echo.annotations == {"readOnlyHint": True}
        assert echo.required == ["text"] and "what to echo" in echo.parameters["text"]
        srv = manager.servers["fakehttp"]
        assert isinstance(srv, MCPHttpServer) and srv.protocol_version == "2025-06-18"
        listed = [r for r in fake.requests if r["method"] == "tools/list"]
        assert len(listed) == 2
        for req in listed:
            # The session id from initialize and the negotiated version ride
            # on every later request, plus the spec's own (expanded) header.
            assert req["headers"].get("Mcp-Session-Id", "").startswith("sess-")
            assert req["headers"].get("MCP-Protocol-Version") == "2025-06-18"
            assert req["headers"].get("X-Api-Key") == "k-123"
        assert manager.call("fakehttp", "echo", {"text": "hi"}) == "echo: hi"
        status = manager.status()[0]
        assert status["url"] == url and status["alive"] is True and status["command"] == ""
    finally:
        manager.stop_all()
        os.environ.pop("MCPT_KEY", None)
        server.shutdown()
    # stop() ends the session politely.
    assert fake.deleted and fake.deleted[0].startswith("sess-")


def test_http_sse_replies_and_list_changed_relists():
    fake = FakeMCP(sse=True)
    server, url = _start_fake(fake)
    manager = MCPManager()
    try:
        manager.configure(_http_spec(url), timeout=10)
        assert len(manager.tools()) == 2
        assert fake.lists == 2              # two pages
        manager.tools()
        assert fake.lists == 2              # cached: not listed again
        # The call's SSE stream carries a list_changed before the reply.
        assert manager.call("fakehttp", "mutate", {}) == "mutated"
        assert manager.call("fakehttp", "echo", {"text": "x"}) == "echo: x"
        assert manager.servers["fakehttp"].tools_stale is True
        names = [s.local_name for s in manager.tools()]
        assert "mcp_fakehttp_added_later" in names, names
        assert manager.servers["fakehttp"].tools_stale is False
    finally:
        manager.stop_all()
        server.shutdown()


def test_http_protocol_version_falls_back_for_an_old_server():
    fake = FakeMCP(min_version="2024-11-05")
    server, url = _start_fake(fake)
    manager = MCPManager()
    try:
        manager.configure(_http_spec(url), timeout=10)
        assert manager.tools(), manager.status()
        assert manager.servers["fakehttp"].protocol_version == "2024-11-05"
        inits = [r for r in fake.requests if r["method"] == "initialize"]
        assert len(inits) == 3              # 2025-06-18, 2025-03-26, 2024-11-05
    finally:
        manager.stop_all()
        server.shutdown()


def test_http_expired_session_reinitializes_once():
    fake = FakeMCP()
    server, url = _start_fake(fake)
    manager = MCPManager()
    try:
        manager.configure(_http_spec(url), timeout=10)
        manager.tools()
        first = manager.servers["fakehttp"].session_id
        fake.forget_next = True
        assert manager.call("fakehttp", "echo", {"text": "again"}) == "echo: again"
        assert manager.servers["fakehttp"].session_id != first
    finally:
        manager.stop_all()
        server.shutdown()


def test_a_failed_listing_is_retried_after_a_backoff_not_forever_skipped():
    fake = FakeMCP(fail_lists=1)
    server, url = _start_fake(fake)
    manager = MCPManager()
    try:
        manager.configure(_http_spec(url), timeout=10)
        assert manager.tools() == []
        srv = manager.servers["fakehttp"]
        assert "tools/list failed" in srv.error
        manager.tools()
        assert fake.lists == 1              # inside the backoff: not hammered
        srv._retry_at = 0                   # the backoff has passed
        assert [s.remote_name for s in manager.tools()] == ["echo", "mutate"]
        assert srv.error == ""
    finally:
        manager.stop_all()
        server.shutdown()


def test_http_spec_shapes():
    out = parse_mcp_spec(json.dumps({"mcpServers": {
        "gh": {"url": "https://example.com/mcp", "bearer": "tok"},
        "old": {"url": "https://example.com/sse", "type": "sse"},
        "bad": {"url": "ftp://example.com"},
        "local": {"command": "x", "args": ["1"]}}}))
    assert out == [
        {"name": "gh", "type": "http", "url": "https://example.com/mcp",
         "headers": {"Authorization": "Bearer tok"}},
        {"name": "local", "command": "x", "args": ["1"], "env": {}, "cwd": None},
    ], out


def test_mcp_servers_are_masked_in_public_config():
    cfg = Config(mcp_servers=json.dumps([
        {"name": "gh", "url": "https://user:pw@example.com/mcp",
         "headers": {"Authorization": "Bearer abc"}},
        {"name": "fs", "command": "npx", "env": {"GITHUB_TOKEN": "ghp_x"}}]))
    shown = cfg.public()["mcp_servers"]
    assert "abc" not in shown and "ghp_x" not in shown and "pw@" not in shown, shown
    assert "npx" in shown and "example.com" in shown
    # The real value is untouched (the servers still start with it).
    assert "ghp_x" in cfg.mcp_servers


# --------------------------------------------------------------------------- #
# Registry: capability gating, annotations, sanitised stdio env
# --------------------------------------------------------------------------- #
def test_mcp_tools_carry_capability_and_annotations_and_need_a_grant():
    fake = FakeMCP()
    server, url = _start_fake(fake)
    try:
        base = dict(mcp_servers=_http_spec(url), mcp_timeout=10,
                    project_dir=tempfile.mkdtemp(), skills_dir=_SKILLS)
        reg = ToolRegistry(Config(**base), None)
        tool = reg.get("mcp_fakehttp_echo")
        assert tool is not None and tool.capability == "mcp:fakehttp"
        assert classify_tool_risk(tool.name, {}, tool) == "read"
        mutate = reg.get("mcp_fakehttp_mutate")
        assert classify_tool_risk(mutate.name, {}, mutate) == "external"
        # The per-server capability key is published for profiles and roles.
        assert "mcp:fakehttp" in CAPABILITY_ORDER
        assert CAPABILITY_ORDER.index("mcp:fakehttp") == CAPABILITY_ORDER.index("mcp") + 1

        def gated(caps):
            cfg = Config(**base, agent_tools=",".join(tools_for_capabilities(caps)))
            return {n for n in ToolRegistry(cfg, None).names() if n.startswith("mcp_")}

        assert gated(["web_search"]) == set()
        assert gated(["mcp"]) == {"mcp_fakehttp_echo", "mcp_fakehttp_mutate"}
        assert gated(["mcp:fakehttp"]) == {"mcp_fakehttp_echo", "mcp_fakehttp_mutate"}
        assert gated(["mcp:someone_else"]) == set()
        # A role granted "mcp" also holds every per-server key.
        role_cfg = Config(role_capabilities=json.dumps({"user": ["mcp"]}))
        assert "mcp:fakehttp" in capabilities_for_role("user", role_cfg)
    finally:
        MCP.stop_all()
        server.shutdown()


_ENV_SERVER = textwrap.dedent('''
    import json, os, sys
    for line in sys.stdin:
        msg = json.loads(line)
        mid = msg.get("id")
        if mid is None:
            continue
        if msg["method"] == "initialize":
            out = {"protocolVersion": msg["params"]["protocolVersion"], "capabilities": {}}
        elif msg["method"] == "tools/list":
            out = {"tools": [{"name": "env", "inputSchema": {"type": "object"}}]}
        else:
            out = {"content": [{"type": "text", "text": json.dumps(dict(os.environ))}]}
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "result": out}) + "\\n")
        sys.stdout.flush()
''')


def test_stdio_children_get_a_sanitised_env_plus_their_own():
    script = Path(tempfile.mkdtemp()) / "env_server.py"
    script.write_text(_ENV_SERVER)
    os.environ["MCPT_SECRET_TOKEN"] = "leak-me"
    os.environ["O365_CLIENT_SECRET"] = "also-secret"
    os.environ["MCPT_REF"] = "referenced"
    spec = json.dumps([{"name": "envsrv", "command": sys.executable, "args": [str(script)],
                        "env": {"EXPLICIT": "yes", "FROM_REF": "${MCPT_REF}"}}])
    try:
        reg = ToolRegistry(Config(mcp_servers=spec, mcp_timeout=10,
                                  project_dir=tempfile.mkdtemp(), skills_dir=_SKILLS), None)
        out, err = reg.call("mcp_envsrv_env", {}, None)
        assert err is None, err
        env = json.loads(out)
        assert env.get("EXPLICIT") == "yes" and env.get("FROM_REF") == "referenced"
        assert "MCPT_SECRET_TOKEN" not in env and "O365_CLIENT_SECRET" not in env
        assert "PATH" in env
    finally:
        MCP.stop_all()
        for key in ("MCPT_SECRET_TOKEN", "O365_CLIENT_SECRET", "MCPT_REF"):
            os.environ.pop(key, None)


# --------------------------------------------------------------------------- #
# This app as an MCP server
# --------------------------------------------------------------------------- #
def build_app(**overrides):
    base = dict(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                allow_test_user=False, node_token="secret-node-token",
                skills_dir=tempfile.mkdtemp(prefix="llm-test-mcp-skills-"),
                project_dir=tempfile.mkdtemp(prefix="llm-test-mcp-proj-"))
    base.update(overrides)
    cfg = Config(**base)
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)  # never started
    app = create_app(cfg, db, mm, RetrainManager(db, mm, cfg), ToolRegistry(cfg, db))
    app.state.model_manager = mm
    return app, cfg, db


def _login(client, username, password):
    r = client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return r


def _rpc(method, mid=1, **params):
    return {"jsonrpc": "2.0", "id": mid, "method": method, "params": params}


def _initialize(client, **headers):
    r = client.post("/mcp", json=_rpc("initialize", protocolVersion="2025-06-18",
                                      capabilities={}, clientInfo={"name": "t"}),
                    headers=headers)
    assert r.status_code == 200, r.text
    sid = r.headers.get("mcp-session-id")
    assert sid
    return sid, r.json()


def test_the_mcp_server_is_off_by_default_and_needs_sign_in():
    app, cfg, _db = build_app()
    with TestClient(app) as c:
        assert c.post("/mcp", json=_rpc("initialize")).status_code == 401
        _login(c, "admin", "adminpw123")
        r = c.post("/mcp", json=_rpc("initialize", protocolVersion="2025-06-18"))
        assert r.status_code == 404 and "disabled" in r.text


def test_initialize_list_call_and_session_binding():
    app, cfg, _db = build_app(mcp_server_enabled=True)
    with TestClient(app) as admin:
        _login(admin, "admin", "adminpw123")
        sid, body = _initialize(admin)
        assert body["result"]["protocolVersion"] == "2025-06-18"
        assert body["result"]["capabilities"]["tools"] == {"listChanged": False}
        h = {"Mcp-Session-Id": sid, "MCP-Protocol-Version": "2025-06-18"}
        assert admin.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                          headers=h).status_code == 202
        listed = admin.post("/mcp", json=_rpc("tools/list", 2), headers=h).json()["result"]
        names = {t["name"] for t in listed["tools"]}
        assert "calculator" in names and "read_file" in names
        assert "final_answer" not in names and "delegate_task" not in names
        calc = next(t for t in listed["tools"] if t["name"] == "calculator")
        assert calc["annotations"]["readOnlyHint"] is True
        assert calc["inputSchema"]["required"] == ["expression"]
        out = admin.post("/mcp", json=_rpc("tools/call", 3, name="calculator",
                                           arguments={"expression": "2+2"}), headers=h).json()
        assert out["result"]["isError"] is False and out["result"]["content"][0]["text"] == "4"
        # ping, unknown method, unknown tool
        assert admin.post("/mcp", json=_rpc("ping", 4), headers=h).json()["result"] == {}
        assert admin.post("/mcp", json=_rpc("resources/list", 5),
                          headers=h).json()["error"]["code"] == -32601
        bad = admin.post("/mcp", json=_rpc("tools/call", 6, name="nope", arguments={}),
                         headers=h).json()
        assert bad["error"]["code"] == -32602
        # Session rules: missing -> 400, unknown -> 404, unsupported version -> 400.
        assert admin.post("/mcp", json=_rpc("ping", 7)).status_code == 400
        assert admin.post("/mcp", json=_rpc("ping", 8),
                          headers={"Mcp-Session-Id": "nope"}).status_code == 404
        assert admin.post("/mcp", json=_rpc("ping", 9),
                          headers={**h, "MCP-Protocol-Version": "1999-01-01"}).status_code == 400
        # A browser page on another origin cannot drive it.
        assert admin.post("/mcp", json=_rpc("ping", 10),
                          headers={**h, "Origin": "https://evil.example"}).status_code == 403
        # An SSE-only client gets the reply as one event.
        sse = admin.post("/mcp", json=_rpc("ping", 11),
                         headers={**h, "Accept": "text/event-stream"})
        assert sse.headers["content-type"].startswith("text/event-stream")
        assert '"id": 11' in sse.text
        admin.post("/api/users", json={"username": "bob", "password": "bobpass123",
                                       "role": "user"})
        with TestClient(app) as bob:
            _login(bob, "bob", "bobpass123")
            # Another user's session id does not exist for bob.
            assert bob.post("/mcp", json=_rpc("ping", 1),
                            headers={"Mcp-Session-Id": sid}).status_code == 404
        assert admin.delete("/mcp", headers={"Mcp-Session-Id": sid}).status_code == 200
        assert admin.post("/mcp", json=_rpc("ping", 12), headers=h).status_code == 404


def test_the_role_narrows_what_mcp_offers_and_allows():
    app, cfg, _db = build_app(mcp_server_enabled=True,
                              role_capabilities=json.dumps({"user": ["web_search"]}))
    with TestClient(app) as admin:
        _login(admin, "admin", "adminpw123")
        admin.post("/api/users", json={"username": "carol", "password": "carolpw123",
                                       "role": "user"})
    with TestClient(app) as carol:
        _login(carol, "carol", "carolpw123")
        sid, _ = _initialize(carol)
        h = {"Mcp-Session-Id": sid}
        names = {t["name"] for t in carol.post("/mcp", json=_rpc("tools/list"),
                                               headers=h).json()["result"]["tools"]}
        assert "web_search" in names and "calculator" in names
        assert "read_file" not in names and "write_file" not in names
        refused = carol.post("/mcp", json=_rpc("tools/call", 2, name="read_file",
                                               arguments={"path": "x"}), headers=h).json()
        assert refused["error"]["code"] == -32602


def test_mcp_calls_go_through_the_unattended_approval_policy():
    app, cfg, db = build_app(mcp_server_enabled=True, tool_approval_mode="all",
                             tool_approval_unattended="deny")
    with TestClient(app) as admin:
        _login(admin, "admin", "adminpw123")
        sid, _ = _initialize(admin)
        out = admin.post("/mcp", json=_rpc("tools/call", 2, name="calculator",
                                           arguments={"expression": "1+1"}),
                         headers={"Mcp-Session-Id": sid}).json()
        # Nobody can answer an approval prompt over MCP: "ask" became deny.
        assert out["result"]["isError"] is True, out
        rows = db.execute("SELECT name, error FROM tool_calls WHERE name='calculator'").fetchall()
        assert rows and rows[-1]["error"], [dict(r) for r in rows]


def test_an_agent_profile_scopes_the_mcp_session():
    app, cfg, db = build_app(mcp_server_enabled=True)
    profile = db.create_agent("Researcher", "web only", ["web_search"])
    with TestClient(app) as admin:
        _login(admin, "admin", "adminpw123")
        sid, _ = _initialize(admin, **{"X-Agent-Id": profile["id"]})
        names = {t["name"] for t in admin.post("/mcp", json=_rpc("tools/list"),
                                               headers={"Mcp-Session-Id": sid}
                                               ).json()["result"]["tools"]}
        assert "web_search" in names and "read_file" not in names, names
        r = admin.post("/mcp", json=_rpc("initialize", protocolVersion="2025-06-18"),
                       headers={"X-Agent-Id": "no-such-agent"})
        assert r.status_code == 404


def _run_standalone():
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
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
