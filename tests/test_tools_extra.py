"""Tests for the named capability tools (tools_extra.py) and capability/role
expansion (config.py)."""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import local_llm.config as config_mod  # noqa: E402
config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-tools-extra-"))

from local_llm.config import (  # noqa: E402
    CAPABILITY_GROUPS, CAPABILITY_ORDER, Config, capabilities_for_role,
    tools_for_capabilities)
from local_llm.database import Database  # noqa: E402
from local_llm.documents import render  # noqa: E402
from local_llm.obslog import set_acting_user  # noqa: E402
from local_llm.policy import classify_tool_risk  # noqa: E402
from local_llm.streams import chat_role_capabilities  # noqa: E402
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.turnctx import TurnContext, reset_current_turn, set_current_turn  # noqa: E402
import local_llm.tools_extra as tx  # noqa: E402


def _reg(caps=None, db=None, **overrides):
    """A registry whose allowlist is what these capabilities unlock."""
    base = dict(project_dir=tempfile.mkdtemp(prefix="llm-test-tx-root-"),
                notify_macos=False, checkpoints_enabled=False,
                # Its own skill library: never the real data/skills.
                skills_dir=tempfile.mkdtemp(prefix="llm-test-tx-skills-"))
    if caps is not None:
        base["agent_tools"] = ",".join(tools_for_capabilities(caps))
    base.update(overrides)
    return ToolRegistry(Config(**base), db)


def _root(reg) -> Path:
    return Path(reg.config.project_dir)


class _Site:
    """A tiny local website + JSON echo endpoint + webhook sink."""

    def __init__(self):
        self.posts: list[dict] = []
        site = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, kind="text/html"):
                raw = body.encode()
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):  # noqa: N802
                if self.path == "/":
                    return self._send(200, (
                        "<html><head><title>Home</title></head><body><h1>Welcome</h1>"
                        "<p>Main page text.</p><a href='/about'>About us</a> "
                        "<a href='https://elsewhere.example/x'>Elsewhere</a></body></html>"))
                if self.path == "/about":
                    return self._send(200, "<html><title>About</title><p>We make tools.</p></html>")
                if self.path == "/moved":
                    self.send_response(302)
                    self.send_header("Location", "/about")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return None
                return self._send(404, "nope", "text/plain")

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length).decode()
                site.posts.append({"path": self.path, "body": body,
                                   "headers": dict(self.headers)})
                return self._send(201, json.dumps({"got": json.loads(body or "null")}),
                                  "application/json")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()


# --------------------------------------------------------------------------- #
# Capabilities and roles
# --------------------------------------------------------------------------- #
def test_legacy_capability_keys_expand_exactly_as_before():
    always = ["final_answer", "calculator", "current_time", "weather"]
    assert tools_for_capabilities(["file_ops"]) == always + [
        "read_file", "write_file", "edit_file", "list_files", "search_files", "file_info",
        "create_document"]
    assert tools_for_capabilities(["code_exec"]) == always + ["run_shell", "run_python", "run_tests"]
    assert tools_for_capabilities(["web_api"]) == always + ["web_search", "fetch_url"]
    # Phase B (upgrade/ROUTING_AND_TOOLING.md P3): "knowledge" used to grant RAG
    # only; it now also grants the knowledge-base tools. Deliberate change.
    assert tools_for_capabilities(["knowledge"]) == always + ["search_docs", "kb_overview"]
    assert tools_for_capabilities(["office365"]) == always + [
        "o365_mail", "o365_files", "o365_calendar"]
    assert CAPABILITY_ORDER[:6] == ["file_ops", "code_exec", "web_api", "knowledge",
                                    "memory", "office365"]


def test_every_named_capability_exists_and_maps_to_tools():
    named = ["web_search", "browse_page", "code_interpreter", "shell", "file_ops", "git",
             "create_document", "spreadsheet", "pdf_extract", "image_analyze",
             "image_generate", "email", "calendar", "contacts", "http_request", "database",
             "memory", "knowledge_search", "task_scheduler", "notify", "mcp"]
    for key in named:
        assert key in CAPABILITY_GROUPS and key in CAPABILITY_ORDER, key
    assert "run_shell" in tools_for_capabilities(["shell"])
    assert "search_docs" in tools_for_capabilities(["knowledge_search"])
    assert {"git", "git_commit"} <= set(tools_for_capabilities(["git"]))
    # Every tool a named key unlocks is either a built-in or a tools_extra tool.
    reg = _reg(None, Database(Path(tempfile.mkdtemp()) / "x.db"), extra_tools="all",
               allow_python=True, allow_shell=True)
    known = set(reg.names()) | set(tx.EXTRA_TOOL_NAMES) | {
        "o365_mail", "o365_files", "o365_calendar", "run_python"}
    for key in named:
        for name in CAPABILITY_GROUPS[key]["tools"]:
            assert name in known, (key, name)


def test_roles_default_to_everything_and_narrow_when_configured():
    assert capabilities_for_role("user", Config()) == list(CAPABILITY_ORDER)
    cfg = Config(role_capabilities=json.dumps({"user": ["web_api", "memory"],
                                               "admin": ["*"]}))
    user = capabilities_for_role("user", cfg)
    # web_api's tools contain web_search's, so that key is held too; shell is not.
    assert "web_search" in user and "memory" in user and "shell" not in user
    assert capabilities_for_role("admin", cfg) == list(CAPABILITY_ORDER)
    assert capabilities_for_role("auditor", cfg) == list(CAPABILITY_ORDER)
    # The chat route's hook uses it.
    assert chat_role_capabilities("user", cfg) == set(user)
    # A malformed value is refused rather than silently narrowing or widening.
    bad = Config(role_capabilities='{"user": "web_api"}')
    assert bad.role_capabilities == ""


def test_new_tools_register_only_when_asked_for():
    default = _reg(None)
    assert not set(default.names()) & set(tx.EXTRA_TOOL_NAMES)
    profile = _reg(["browse_page", "http_request", "git", "code_interpreter"])
    names = set(profile.names())
    assert {"browse_page", "fetch_url", "http_request", "git", "git_commit"} <= names
    assert "code_interpreter" not in names          # needs ALLOW_PYTHON
    assert "read_file" not in names                  # not granted
    assert "code_interpreter" in _reg(["code_interpreter"], allow_python=True).names()
    # Graph tools need an Azure app, exactly like the old office365 stubs.
    assert "email" not in _reg(["email"]).names()
    graph = _reg(["email", "calendar", "contacts"], o365_tenant_id="t", o365_client_id="c",
                 o365_client_secret="s")
    assert {"email", "email_send", "calendar", "calendar_create", "contacts"} <= set(graph.names())
    some = _reg(None, extra_tools="git, notify")
    assert {"git", "notify"} <= set(some.names())
    assert "browse_page" not in some.names()


def test_risk_and_capability_metadata_drive_policy():
    reg = _reg(["http_request", "git", "notify", "spreadsheet"])
    http = reg.get("http_request")
    assert classify_tool_risk("http_request", {"method": "GET"}, http) == "read"
    assert classify_tool_risk("http_request", {"method": "POST"}, http) == "external"
    assert classify_tool_risk("git", {}, reg.get("git")) == "read"
    assert classify_tool_risk("git_commit", {}, reg.get("git_commit")) == "write"
    assert classify_tool_risk("notify", {}, reg.get("notify")) == "external"
    assert reg.get("spreadsheet_write").capability == "spreadsheet"


# --------------------------------------------------------------------------- #
# browse_page / http_request
# --------------------------------------------------------------------------- #
def test_browse_page_reads_text_links_and_follows_one_same_site_link():
    site = _Site()
    try:
        reg = _reg(["browse_page"], allow_local_fetch=True)
        out, err = reg.call("browse_page", {"url": site.base + "/"}, None)
        assert err is None, err
        assert "Home" in out and "Main page text." in out
        assert f"About us -> {site.base}/about" in out
        out2, err2 = reg.call("browse_page", {"url": site.base + "/", "follow": "About"}, None)
        assert err2 is None and "We make tools." in out2, out2
        _o, err3 = reg.call("browse_page", {"url": site.base + "/", "follow": "Elsewhere"}, None)
        assert err3 and "same site" in err3
        # Redirects are followed (and re-checked) by hand.
        out4, err4 = reg.call("browse_page", {"url": site.base + "/moved"}, None)
        assert err4 is None and "We make tools." in out4
    finally:
        site.close()


def test_browse_and_http_refuse_private_addresses():
    reg = _reg(["browse_page", "http_request"])      # allow_local_fetch off
    for name in ("browse_page", "http_request"):
        _out, err = reg.call(name, {"url": "http://127.0.0.1:9/"}, None)
        assert err and "private or non-public" in err, (name, err)
    _out, err = reg.call("http_request", {"url": "http://169.254.169.254/latest"}, None)
    assert err and "non-public" in err


def test_http_request_methods_headers_and_body():
    site = _Site()
    try:
        reg = _reg(["http_request"], allow_local_fetch=True)
        out, err = reg.call("http_request", {
            "url": site.base + "/api", "method": "POST",
            "headers": {"X-Trace": "abc", "Host": "evil"}, "json": {"a": 1}}, None)
        assert err is None, err
        assert out.startswith("HTTP 201") and '"got": {"a": 1}' in out, out
        sent = site.posts[-1]
        assert sent["headers"].get("X-Trace") == "abc"
        assert not sent["headers"].get("Host", "").startswith("evil")
        _o, err2 = reg.call("http_request", {"url": site.base, "method": "TRACE"}, None)
        assert err2 and "method must be" in err2
        _o, err3 = reg.call("http_request", {"url": site.base, "body": "x"}, None)
        assert err3 and "cannot carry a body" in err3
    finally:
        site.close()


# --------------------------------------------------------------------------- #
# code_interpreter
# --------------------------------------------------------------------------- #
def test_code_interpreter_keeps_state_between_calls_and_resets():
    reg = _reg(["code_interpreter"], allow_python=True)
    cid = "conv-ci-1"
    out, err = reg.call("code_interpreter", {"code": "import math\nx = 41\n"
                                             "def double(v):\n    return v * 2\n"}, cid)
    assert err is None and "exit code 0" in out, out
    out, err = reg.call("code_interpreter", {"code": "x + 1"}, cid)
    assert "42" in out, out
    out, err = reg.call("code_interpreter", {"code": "print(double(x), round(math.pi, 2))"}, cid)
    assert "82 3.14" in out, out
    # Another conversation has its own session.
    out, err = reg.call("code_interpreter", {"code": "print('x' in dir())"}, "conv-ci-2")
    assert "False" in out, out
    out, err = reg.call("code_interpreter", {"code": "print('x' in dir())", "reset": "true"}, cid)
    assert "False" in out, out
    # The session lives in an ignored cache folder inside the root.
    assert (_root(reg) / ".cache" / "code_interpreter" / cid / "state.pkl").exists()


# --------------------------------------------------------------------------- #
# git
# --------------------------------------------------------------------------- #
def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_git_reads_commits_and_never_pushes():
    reg = _reg(["git"])
    root = _root(reg)
    _git(root, "init", "-q")
    (root / "a.txt").write_text("one\n")
    out, err = reg.call("git", {"action": "status"}, None)
    assert err is None and "a.txt" in out, (out, err)
    out, err = reg.call("git_commit", {"action": "commit", "message": "first",
                                       "paths": "a.txt"}, None)
    assert err is None and "nothing was pushed" in out, (out, err)
    out, err = reg.call("git", {"action": "log"}, None)
    assert err is None and "first" in out, out
    (root / "a.txt").write_text("two\n")
    out, err = reg.call("git", {"action": "diff"}, None)
    assert "+two" in out, out
    for action in ("push", "reset", "checkout"):
        _o, err = reg.call("git_commit", {"action": action}, None)
        assert err, action
    _o, err = reg.call("git", {"action": "push"}, None)
    assert err and "push is refused" in err
    _o, err = reg.call("git", {"action": "show", "rev": "--output=/tmp/x"}, None)
    assert err and "not a revision" in err
    _o, err = reg.call("git_commit", {"action": "add", "paths": "../outside.txt"}, None)
    assert err and "escapes" in err


def test_git_refuses_a_root_inside_a_bigger_repository():
    outer = Path(tempfile.mkdtemp(prefix="llm-test-tx-outer-"))
    _git(outer, "init", "-q")
    inner = outer / "sub"
    inner.mkdir()
    reg = _reg(["git"], project_dir=str(inner))
    _o, err = reg.call("git", {"action": "status"}, None)
    assert err and "repository root" in err, err


# --------------------------------------------------------------------------- #
# spreadsheet / pdf_extract
# --------------------------------------------------------------------------- #
def test_spreadsheet_write_then_read_cells_and_ranges():
    reg = _reg(["spreadsheet"])
    rows = [["name", "qty", "note"], ["apple", "3", "red | green"], ["pear", "10", ""]]
    out, err = reg.call("spreadsheet_write", {"path": "out/fruit.xlsx", "rows": json.dumps(rows)},
                        None)
    assert err is None and "3 row(s)" in out, (out, err)
    out, err = reg.call("spreadsheet", {"path": "out/fruit.xlsx"}, None)
    assert err is None and "Data" in out and "apple" in out, out
    out, err = reg.call("spreadsheet", {"path": "out/fruit.xlsx", "range": "A2:B3"}, None)
    assert err is None, err
    assert "2 | apple | 3" in out and "3 | pear | 10" in out and "note" not in out, out
    out, err = reg.call("spreadsheet_write", {"path": "t.csv", "rows": "a,b\n1,2\n"}, None)
    assert err is None
    out, err = reg.call("spreadsheet", {"path": "t.csv", "range": "B2"}, None)
    assert "2 | 2" in out, out
    _o, err = reg.call("spreadsheet", {"path": "missing.xlsx"}, None)
    assert err and "no such file" in err


def test_pdf_extract_counts_pages_and_picks_them():
    reg = _reg(["pdf_extract"])
    body = "# Report\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(80))
    (_root(reg) / "r.pdf").write_bytes(render(body, ".pdf"))
    out, err = reg.call("pdf_extract", {"path": "r.pdf"}, None)
    assert err is None and out.startswith("[pdf: "), out[:200]
    out2, err2 = reg.call("pdf_extract", {"path": "r.pdf", "pages": "2"}, None)
    assert err2 is None and "--- page 2 ---" in out2 and "--- page 1 ---" not in out2, out2[:300]
    _o, err3 = reg.call("pdf_extract", {"path": "r.pdf", "pages": "999"}, None)
    assert err3 and "no such pages" in err3
    _o, err4 = reg.call("pdf_extract", {"path": "nope.txt"}, None)
    assert err4


# --------------------------------------------------------------------------- #
# image_analyze / image_generate
# --------------------------------------------------------------------------- #
_PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00"
        b"\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05"
        b"\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82")


def test_image_analyze_needs_a_vision_model_and_sends_the_image():
    reg = _reg(["image_analyze"])
    (_root(reg) / "dot.png").write_bytes(_PNG)
    _o, err = reg.call("image_analyze", {"path": "dot.png"}, None)
    assert err and "needs a vision model" in err
    seen = {}

    def fake_complete(registry, messages, max_tokens):
        seen["messages"] = messages
        return "a single pixel"

    original = tx._tx_model_complete
    tx._tx_model_complete = fake_complete
    token = set_current_turn(TurnContext(extras={"vision": True}))
    try:
        out, err = reg.call("image_analyze", {"path": "dot.png", "question": "what?"}, None)
    finally:
        reset_current_turn(token)
        tx._tx_model_complete = original
    assert err is None and out == "a single pixel"
    parts = seen["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "what?"}
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_image_generate_is_unavailable_without_a_generator_and_saves_output():
    reg = _reg(["image_generate"])
    if not tx.shutil.which("mflux-generate"):
        _o, err = reg.call("image_generate", {"prompt": "a cat"}, None)
        assert err and "unavailable" in err
    script = Path(tempfile.mkdtemp()) / "gen.py"
    script.write_text("import sys\nopen(sys.argv[2], 'wb').write(%r)\n" % _PNG)
    reg2 = _reg(["image_generate"],
                image_gen_command=f"{sys.executable} {script} {{prompt}} {{output}}")
    out, err = reg2.call("image_generate", {"prompt": "a cat; rm -rf /", "path": "art/cat.png"},
                         None)
    assert err is None and "art/cat.png" in out, (out, err)
    assert (_root(reg2) / "art" / "cat.png").read_bytes() == _PNG


# --------------------------------------------------------------------------- #
# database
# --------------------------------------------------------------------------- #
def test_database_is_read_only():
    # *.db is in the default FILE_DENY_GLOBS (Phase B, P8): a .db file in the
    # project is reachable by name only once an admin narrows the list.
    # DATABASE_URLS (below) is the supported way to name a database.
    guarded = _reg(["database"])
    reg = _reg(["database"], project_dir=str(_root(guarded)), file_deny_globs=".env,data/**")
    path = _root(reg) / "shop.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE items (id INTEGER, name TEXT)")
    conn.executemany("INSERT INTO items VALUES (?, ?)", [(1, "pen"), (2, "ink")])
    conn.commit()
    conn.close()
    _o, err = guarded.call("database", {"database": "shop.db", "sql": "SELECT 1"}, None)
    assert err and "protected (FILE_DENY_GLOBS)" in err, err
    out, err = reg.call("database", {"database": "shop.db",
                                     "sql": "SELECT name, replace(name, 'p', 'P') FROM items "
                                            "ORDER BY id"}, None)
    assert err is None and "pen\tPen" in out and "(2 rows)" in out, (out, err)
    out, err = reg.call("database", {"database": "shop.db", "sql": "PRAGMA table_info(items)"},
                        None)
    assert err is None and "name" in out
    for sql in ("DELETE FROM items", "INSERT INTO items VALUES (3, 'x')",
                "SELECT 1; DROP TABLE items", "ATTACH DATABASE '/etc/x.db' AS x",
                "WITH t AS (SELECT 1) DELETE FROM items", "PRAGMA journal_mode = DELETE",
                "REPLACE INTO items VALUES (1, 'y')"):
        _o, err = reg.call("database", {"database": "shop.db", "sql": sql}, None)
        assert err and ("read-only" in err or "one statement" in err or "SQL error" in err), (sql, err)
    check = sqlite3.connect(path)
    assert check.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 2
    check.close()
    # A configured database by name; other engines are refused clearly.
    cfg_reg = _reg(["database"], database_urls=json.dumps(
        {"shop": f"sqlite:///{path}", "pg": "postgresql://u:p@h/db"}))
    out, err = cfg_reg.call("database", {"database": "shop", "sql": "SELECT COUNT(*) FROM items"},
                            None)
    assert err is None and out.splitlines()[1] == "2", out
    _o, err = cfg_reg.call("database", {"database": "pg", "sql": "SELECT 1"}, None)
    assert err and "only SQLite" in err
    _o, err = reg.call("database", {"database": "../x.db", "sql": "SELECT 1"}, None)
    assert err and "escapes" in err


# --------------------------------------------------------------------------- #
# task_scheduler
# --------------------------------------------------------------------------- #
def test_task_scheduler_clamps_tools_to_the_callers_own():
    db = Database(Path(tempfile.mkdtemp()) / "tasks.db")
    alice = db.create_user("alice", role="user")["id"]
    bob = db.create_user("bob", role="user")["id"]
    reg = _reg(["web_search", "task_scheduler"], db)
    set_acting_user(alice)
    try:
        out, err = reg.call("task_scheduler", {
            "action": "create", "goal": "check the news", "every_minutes": "60",
            "tools": "web_search,run_shell,write_file"}, None)
        assert err is None, err
        assert "Not granted" in out and "run_shell" in out
        task = [t for t in db.list_tasks() if t["user_id"] == alice][0]
        granted = set(task["tools"].split(","))
        assert granted <= set(reg.names()) and "run_shell" not in granted
        assert "task_scheduler" not in granted and "final_answer" in granted
        assert task["interval_seconds"] == 3600 and task["next_run_at"]
        # Unspecified tools = the caller's own set, never "" (which means every tool).
        reg.call("task_scheduler", {"action": "create", "goal": "once", "start_in_minutes": "5"},
                 None)
        once = [t for t in db.list_tasks() if t["goal"] == "once"][0]
        assert once["tools"] and once["interval_seconds"] == 0 and once["next_run_at"]
        listed, _ = reg.call("task_scheduler", {"action": "list"}, None)
        assert "check the news" in listed
    finally:
        set_acting_user(None)
    # Bob neither sees nor cancels Alice's tasks.
    set_acting_user(bob)
    try:
        listed, _ = reg.call("task_scheduler", {"action": "list"}, None)
        assert "check the news" not in listed
        _o, err = reg.call("task_scheduler", {"action": "cancel", "task_id": task["id"]}, None)
        assert err and "no such task" in err
    finally:
        set_acting_user(None)
    set_acting_user(alice)
    try:
        out, err = reg.call("task_scheduler", {"action": "cancel", "task_id": task["id"]}, None)
        assert err is None and not db.get_task(task["id"])["enabled"]
    finally:
        set_acting_user(None)
    db.close()


def test_task_scheduler_respects_a_restricted_role():
    db = Database(Path(tempfile.mkdtemp()) / "tasks2.db")
    carol = db.create_user("carol", role="user")["id"]
    # The default agent (no allowlist) has file tools, but carol's role does not.
    reg = _reg(None, db, extra_tools="task_scheduler",
               role_capabilities=json.dumps({"user": ["web_search", "task_scheduler"]}))
    set_acting_user(carol)
    try:
        out, err = reg.call("task_scheduler", {"action": "create", "goal": "g",
                                               "tools": "write_file,web_search"}, None)
        assert err is None and "write_file" in out.split("Not granted")[-1], out
        task = db.list_tasks()[0]
        assert "write_file" not in task["tools"].split(",")
    finally:
        set_acting_user(None)
    db.close()


# --------------------------------------------------------------------------- #
# notify
# --------------------------------------------------------------------------- #
def test_notify_stores_in_app_and_posts_the_webhook():
    site = _Site()
    db = Database(Path(tempfile.mkdtemp()) / "notify.db")
    uid = db.create_user("dave", role="user")["id"]
    try:
        reg = _reg(["notify"], db, notify_webhook_url=site.base + "/hook")
        set_acting_user(uid)
        out, err = reg.call("notify", {"message": "build finished", "title": "CI"}, None)
        assert err is None and "in-app" in out and "webhook" in out, (out, err)
        rows = tx.list_notifications(db, uid)
        assert rows[0]["body"] == "build finished" and rows[0]["title"] == "CI"
        assert tx.list_notifications(db, "someone-else") == []
        hook = json.loads(site.posts[-1]["body"])
        assert hook["message"] == "build finished" and "CI" in hook["text"]
        # The webhook URL is a secret: never shown by public().
        assert reg.config.public()["notify_webhook_url"] == "***set***"
    finally:
        set_acting_user(None)
        site.close()
        db.close()
    lonely = _reg(["notify"], None)
    _o, err = lonely.call("notify", {"message": "x"}, None)
    assert err and "could not be delivered" in err


# --------------------------------------------------------------------------- #
# email / calendar / contacts (Graph faked)
# --------------------------------------------------------------------------- #
def test_graph_tools_use_only_the_callers_own_mailbox():
    db = Database(Path(tempfile.mkdtemp()) / "graph.db")
    erin = db.create_user("erin", role="user")["id"]
    db.execute("UPDATE users SET email = ? WHERE id = ?", ("erin@contoso.com", erin))
    db.commit()
    frank = db.create_user("frank", role="user")["id"]
    reg = _reg(["email", "calendar", "contacts"], db, o365_tenant_id="t", o365_client_id="c",
               o365_client_secret="s", o365_user="boss@contoso.com")
    calls = []

    def fake_graph(config, method, path, params=None, body=None):
        calls.append((method, path, params, body))
        if path.endswith("/contacts"):
            return {"value": [{"displayName": "Zed", "emailAddresses": [{"address": "z@x.io"}]},
                              {"displayName": "Amy", "emailAddresses": []}]}
        return {"value": [{"id": "m1", "subject": "Hello", "from": {"emailAddress": {
            "name": "A", "address": "a@x.io"}}, "receivedDateTime": "2026-10-01",
            "bodyPreview": "hi"}]}

    original = tx._tx_graph
    tx._tx_graph = fake_graph
    try:
        set_acting_user(erin)
        out, err = reg.call("email", {"action": "list"}, None)
        assert err is None and "Hello" in out
        assert calls[-1][1] == "/users/erin@contoso.com/mailFolders/inbox/messages"
        out, err = reg.call("email_send", {"to": "x@y.com", "subject": "s", "body": "b"}, None)
        assert err is None and calls[-1][1] == "/users/erin@contoso.com/sendMail"
        assert classify_tool_risk("email_send", {}, reg.get("email_send")) == "external"
        out, err = reg.call("contacts", {"query": "zed"}, None)
        assert "Zed" in out and "Amy" not in out
        _o, err = reg.call("email_send", {"to": "not-an-address", "subject": "s", "body": "b"},
                           None)
        assert err and "not an email address" in err
        # A user without an address never falls back to someone else's mailbox.
        set_acting_user(frank)
        before = len(calls)
        _o, err = reg.call("calendar", {}, None)
        assert err and "no Microsoft 365 mailbox" in err and len(calls) == before
    finally:
        tx._tx_graph = original
        set_acting_user(None)
        db.close()


def test_sql_checker_unit():
    assert tx._tx_sql_check("select 1;") == "select 1"
    assert tx._tx_sql_check("-- c\nSELECT 'delete me'") .endswith("'delete me'")
    for bad in ("", "update t set a=1", "pragma writable_schema = 1", "select 1; select 2"):
        try:
            tx._tx_sql_check(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


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
