"""Tests for the menu-bar companion: the status summary route, the SwiftBar
plugin, and the `deploy.py companion` build/install helpers.

Nothing here starts a model server or launches the GUI app. The swiftc test
compiles the Swift source into a temp folder and only checks the bundle.
"""
from __future__ import annotations

import json
import os
import plistlib
import stat
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_llm.config as config_mod  # noqa: E402

config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-companion-"))

from fastapi.testclient import TestClient  # noqa: E402

from local_llm.api import create_app  # noqa: E402
from local_llm.companion import (  # noqa: E402
    COMPANION_APP_NAME, COMPANION_BUNDLE_ID, COMPANION_EXECUTABLE, COMPANION_SUMMARY_PATH,
    CompanionError, companion_build, companion_build_command, companion_can, companion_cli,
    companion_find_swiftc, companion_hour_metrics, companion_indicator, companion_install,
    companion_install_plugin, companion_launch_agent_plist, companion_plugin_source,
    companion_render_info_plist, companion_reset_cache, companion_source_dir,
    companion_uninstall)
from local_llm.config import Config  # noqa: E402
from local_llm.core import ADAPTER_DIR  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.training import RetrainManager  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = companion_plugin_source(ROOT)
NODE_SECRET = "secret-node-token-companion"
ADMIN_PW = "adminpw123"


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def build_app(**overrides):
    base = dict(auth_enabled=True, admin_username="admin", admin_password=ADMIN_PW,
                allow_test_user=False, node_token=NODE_SECRET,
                skills_dir=tempfile.mkdtemp(prefix="local-llm-test-skills-"))
    base.update(overrides)
    cfg = Config(**base)
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)  # never started
    rm = RetrainManager(db, mm, cfg)
    app = create_app(cfg, db, mm, rm, ToolRegistry(cfg, db))
    app.state.model_manager = mm
    return app, cfg, db


def _login(app, username="admin", password=ADMIN_PW):
    client = TestClient(app)
    r = client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return client


def _token(client, scopes):
    r = client.post("/api/auth/tokens", json={"scopes": scopes})
    assert r.status_code == 200, r.text
    return r.json()["token"]


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


def _detail(r):
    body = r.json()
    return str(body.get("error") or body.get("detail") or "")


# --------------------------------------------------------------------------- #
# Pure pieces                                                                 #
# --------------------------------------------------------------------------- #
def test_indicator_states():
    assert companion_indicator("ready", True, {"waiting": 0}) == \
        {"state": "ready", "label": "ready", "color": "green"}
    assert companion_indicator("ready", True, {"waiting": 3})["label"] == "queue 3"
    assert companion_indicator("ready", False, {"waiting": 0})["state"] == "degraded"
    assert companion_indicator("loading", None, None)["color"] == "yellow"
    assert companion_indicator("starting", None, {})["state"] == "loading"
    assert companion_indicator("error: boom", True, {})["state"] == "error"
    assert companion_indicator("stopped", None, {})["color"] == "gray"
    assert companion_indicator(None, None, {"waiting": "x"})["state"] == "unknown"


def test_companion_can_needs_admin_role_and_token_scope():
    admin_session = {"role": "admin", "token_scopes": None}
    assert companion_can(admin_session, "POST", "/api/model/restart")
    assert not companion_can({"role": "user", "token_scopes": None}, "POST", "/api/model/restart")
    control = {"role": "admin", "token_scopes": ["control"]}
    assert companion_can(control, "POST", "/api/model/restart")
    assert companion_can(control, "POST", "/api/tasks/x/cancel")
    # The api_tokens route table keeps /api/retrain for the admin scope.
    assert not companion_can(control, "POST", "/api/retrain")
    assert companion_can({"role": "admin", "token_scopes": ["admin"]}, "POST", "/api/retrain")
    assert not companion_can({"role": "admin", "token_scopes": ["status"]},
                             "POST", "/api/model/restart")


def test_hour_metrics_scope_and_percentiles():
    db = Database(Path(tempfile.mkdtemp()) / "m.db")
    rows = [("chat", 120.0, 200, 100.0, 20.0, "u1"),
            ("chat", 300.0, 500, 300.0, 40.0, "u1"),
            ("chat", 200.0, 200, 200.0, 30.0, "u2"),
            ("chat_stream_gen", 50.0, 200, 150.0, 25.0, "u2")]
    for endpoint, dur, code, ttft, tps, uid in rows:
        db.execute("INSERT INTO metrics (endpoint, duration_ms, status_code, ttft_ms, "
                   "decode_tps, user_id) VALUES (?, ?, ?, ?, ?, ?)",
                   (endpoint, dur, code, ttft, tps, uid))
    db.commit()
    companion_reset_cache()
    everyone = companion_hour_metrics(db, None, now=time.time() + 5)
    # chat_stream_gen is a generation row, not a request.
    assert everyone["requests"] == 3 and everyone["errors"] == 1
    assert everyone["scope"] == "all" and everyone["window"] == "1h"
    assert everyone["ttft_p50_ms"] in (150.0, 200.0)
    assert everyone["decode_tps_p50"] in (25.0, 30.0)
    own = companion_hour_metrics(db, "u1", now=time.time() + 5)
    assert own["scope"] == "own" and own["requests"] == 2 and own["errors"] == 1
    assert own["ttft_p50_ms"] in (100.0, 300.0)
    # Rows older than an hour drop out.
    later = companion_hour_metrics(db, None, now=time.time() + 7200)
    assert later["requests"] == 0 and later["ttft_p50_ms"] is None
    # The live (now=None) path caches and still answers.
    assert companion_hour_metrics(db, None)["requests"] == 3
    companion_reset_cache()


# --------------------------------------------------------------------------- #
# Route                                                                       #
# --------------------------------------------------------------------------- #
_TOP_KEYS = {"ui_build", "server_time", "indicator", "model", "queue", "approvals", "metrics",
             "user", "controls", "api_base_path", "nodes", "multi_node", "tasks", "retrain"}


def test_summary_route_shape_for_admin_session():
    companion_reset_cache()
    app, cfg, db = build_app()
    admin = _login(app)
    r = admin.get(COMPANION_SUMMARY_PATH)
    assert r.status_code == 200, r.text
    body = r.json()
    assert _TOP_KEYS <= set(body)
    assert set(body["indicator"]) == {"state", "label", "color"}
    assert set(body["model"]) >= {"status", "healthy", "id", "alias", "adapter", "backend",
                                  "capabilities"}
    assert set(body["queue"]) == {"running", "waiting", "limit"}
    assert isinstance(body["approvals"]["pending"], int)
    assert body["metrics"]["scope"] == "all"
    assert body["user"]["role"] == "admin" and body["user"]["scopes"] is None
    assert body["controls"] == {"restart_model": True, "cancel_tasks": True, "retrain": True}
    assert isinstance(body["nodes"], list)
    for node in body["nodes"]:
        assert set(node) == {"name", "state", "role", "active", "local"}
    assert set(body["tasks"]) == {"running", "running_ids"}
    assert set(body["retrain"]) == {"running", "message"}
    assert body["api_base_path"] == "/v1"
    # Never a secret.
    text = r.text
    assert NODE_SECRET not in text and ADMIN_PW not in text
    assert "token_hash" not in text and "llm_" not in text


def test_summary_route_scopes():
    companion_reset_cache()
    app, cfg, db = build_app()
    admin = _login(app)
    bare = TestClient(app)
    assert bare.get(COMPANION_SUMMARY_PATH).status_code == 401

    status = _token(admin, ["status"])
    r = bare.get(COMPANION_SUMMARY_PATH, headers=_bearer(status))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["user"]["scopes"] == ["status"]
    assert body["controls"] == {"restart_model": False, "cancel_tasks": False, "retrain": False}
    assert status not in r.text and NODE_SECRET not in r.text
    # Admin role, so machine-wide telemetry is present.
    assert isinstance(body["nodes"], list) and body["tasks"] is not None

    chat = _token(admin, ["chat"])
    r = bare.get(COMPANION_SUMMARY_PATH, headers=_bearer(chat))
    assert r.status_code == 403 and "status scope" in _detail(r)

    control = _token(admin, ["control"])
    r = bare.get(COMPANION_SUMMARY_PATH, headers=_bearer(control))
    assert r.status_code == 200
    assert r.json()["controls"] == {"restart_model": True, "cancel_tasks": True,
                                    "retrain": False}
    # The levers the summary advertises are reachable with that token (the
    # route itself may fail on an unstarted model or unknown task, never 403).
    assert bare.post("/api/tasks/nope/cancel", headers=_bearer(control)).status_code != 403
    assert bare.post("/api/retrain", headers=_bearer(control)).status_code == 403

    full = _token(admin, ["admin"])
    r = bare.get(COMPANION_SUMMARY_PATH, headers=_bearer(full))
    assert r.status_code == 200 and r.json()["controls"]["retrain"] is True

    # A bad token is a 401, not a crash.
    r = bare.get(COMPANION_SUMMARY_PATH, headers=_bearer("llm_abcdefgh_" + "x" * 40))
    assert r.status_code == 401


def test_summary_route_non_admin_sees_only_own_basics():
    companion_reset_cache()
    app, cfg, db = build_app()
    admin = _login(app)
    r = admin.post("/api/users", json={"username": "bob", "password": "bobpw12345",
                                       "role": "user"})
    assert r.status_code == 200, r.text
    bob = _login(app, "bob", "bobpw12345")
    r = bob.get(COMPANION_SUMMARY_PATH)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["user"]["role"] == "user"
    assert body["nodes"] is None and body["tasks"] is None and body["retrain"] is None
    assert body["multi_node"] is None
    assert body["metrics"]["scope"] == "own"
    assert not any(body["controls"].values())
    status = _token(bob, ["status"])
    r = TestClient(app).get(COMPANION_SUMMARY_PATH, headers=_bearer(status))
    assert r.status_code == 200 and r.json()["nodes"] is None


def test_summary_route_with_auth_off():
    companion_reset_cache()
    app, cfg, db = build_app(auth_enabled=False)
    r = TestClient(app).get(COMPANION_SUMMARY_PATH)
    assert r.status_code == 200, r.text
    assert r.json()["indicator"]["state"] in {"stopped", "unknown", "loading", "error",
                                               "ready", "busy", "degraded"}


# --------------------------------------------------------------------------- #
# Info.plist, build command, build/install with fakes                         #
# --------------------------------------------------------------------------- #
def _template() -> str:
    return (companion_source_dir(ROOT) / "macos" / "Info.plist").read_text(encoding="utf-8")


def test_info_plist_rendering():
    out = companion_render_info_plist(_template())
    data = plistlib.loads(out.encode("utf-8"))
    assert data["LSUIElement"] is True
    assert data["CFBundleIdentifier"] == COMPANION_BUNDLE_ID == "local.llm.bar"
    assert data["CFBundleExecutable"] == COMPANION_EXECUTABLE
    assert data["CFBundleName"] == COMPANION_APP_NAME
    assert data["LSMinimumSystemVersion"] == "13.0"
    assert data["CFBundlePackageType"] == "APPL"
    assert "{{" not in out
    # Overrides are XML-escaped.
    named = plistlib.loads(companion_render_info_plist(_template(), name="A & B").encode())
    assert named["CFBundleName"] == "A & B"
    with pytest.raises(CompanionError):
        companion_render_info_plist(_template() + "<!-- {{MISSING}} -->")
    with pytest.raises(CompanionError):
        companion_render_info_plist("<plist><dict><key>x</key></plist>")


def test_build_command_construction():
    src, out = Path("/s/LocalLLMBar.swift"), Path("/o/LocalLLMBar")
    cmd = companion_build_command(src, out, swiftc="/usr/bin/swiftc", arch="arm64")
    assert cmd[0] == "/usr/bin/swiftc" and "-O" in cmd
    assert cmd[cmd.index("-target") + 1] == "arm64-apple-macos13.0"
    assert "-parse-as-library" in cmd
    assert cmd[cmd.index("-swift-version") + 1] == "5"
    for framework in ("AppKit", "Security"):
        i = [n for n, v in enumerate(cmd) if v == framework]
        assert i and cmd[i[0] - 1] == "-framework"
    assert cmd[cmd.index("-o") + 1] == str(out) and cmd[-1] == str(src)
    assert "-O" not in companion_build_command(src, out, optimize=False)

    plan = companion_build(Path("/nonexistent/dist"), swiftc="/usr/bin/swiftc",
                           arch="universal", dry_run=True, root=ROOT)
    assert plan["dry_run"] and plan["app"] == Path("/nonexistent/dist") / "LocalLLM Bar.app"
    commands = plan["commands"]
    targets = [c[c.index("-target") + 1] for c in commands if "-target" in c]
    assert targets == ["arm64-apple-macos13.0", "x86_64-apple-macos13.0"]
    assert any(c[0] == "/usr/bin/lipo" and "-create" in c for c in commands)
    assert not Path("/nonexistent/dist").exists()
    unsigned = companion_build(Path("/nonexistent/dist"), arch="arm64", sign=False,
                               dry_run=True, root=ROOT)["commands"]
    assert len(unsigned) == 1 and not any("codesign" in c[0] for c in unsigned)
    with pytest.raises(CompanionError):
        companion_build(Path("/nonexistent/dist"), arch="sparc", dry_run=True, root=ROOT)
    with pytest.raises(CompanionError):
        companion_build(Path("/nonexistent/dist"), dry_run=True, root=Path("/nonexistent"))


def test_build_with_fake_compiler_swaps_bundle_in(tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "-o" in cmd:
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\xcf\xfa\xed\xfe fake")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    result = companion_build(tmp_path, swiftc="/usr/bin/true", arch="arm64", sign=False,
                             root=ROOT, run=fake_run)
    app = result["app"]
    assert app == tmp_path / "LocalLLM Bar.app" and len(calls) == 1
    assert (app / "Contents" / "MacOS" / COMPANION_EXECUTABLE).exists()
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    assert info["LSUIElement"] is True and info["CFBundleIdentifier"] == "local.llm.bar"
    assert (app / "Contents" / "PkgInfo").read_text() == "APPL????"
    assert not any(p.name.startswith(".") for p in tmp_path.iterdir())

    # A failed compile leaves the previous app in place and no staging folder.
    def failing_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "error: nope")

    with pytest.raises(CompanionError, match="nope"):
        companion_build(tmp_path, swiftc="/usr/bin/true", arch="arm64", sign=False,
                        root=ROOT, run=failing_run)
    assert (app / "Contents" / "MacOS" / COMPANION_EXECUTABLE).exists()
    assert not any(p.name.startswith(".") for p in tmp_path.iterdir())


def test_install_and_uninstall_with_login_item(tmp_path):
    def fake_run(cmd, **kwargs):
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"bin")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    built = companion_build(tmp_path / "dist", swiftc="/usr/bin/true", arch="arm64",
                            sign=False, root=ROOT, run=fake_run)["app"]
    home = tmp_path / "home"
    with pytest.raises(CompanionError):
        companion_install(tmp_path / "missing.app", home=home)
    result = companion_install(built, home=home, login=True)
    app = home / "Applications" / "LocalLLM Bar.app"
    agent = home / "Library" / "LaunchAgents" / "local.llm.bar.plist"
    assert result["app"] == app and result["launch_agent"] == agent
    assert (app / "Contents" / "MacOS" / COMPANION_EXECUTABLE).exists()
    plist = plistlib.loads(agent.read_bytes())
    assert plist["Label"] == "local.llm.bar" and plist["RunAtLoad"] is True
    assert plist["KeepAlive"] is False
    assert plist["ProgramArguments"] == [str(app / "Contents" / "MacOS" / COMPANION_EXECUTABLE)]
    assert plistlib.loads(companion_launch_agent_plist(app)) == plist
    # Reinstall over the top without login keeps working.
    assert companion_install(built, home=home)["launch_agent"] is None
    removed = companion_uninstall(home=home)["removed"]
    assert set(removed) == {app, agent} and not app.exists() and not agent.exists()
    assert companion_uninstall(home=home)["removed"] == []


def test_cli_dry_run_and_plugin_install(tmp_path, capsys):
    assert companion_cli(["build", "--dry-run", "--arch", "arm64", "--no-sign"]) == 0
    out = capsys.readouterr().out
    assert "swiftc" in out and "-parse-as-library" in out and "'" in out  # quoted spaces
    assert companion_cli(["plugin", "--print"]) == 0
    assert capsys.readouterr().out.strip().endswith("local-llm.30s.py")
    assert companion_cli(["plugin", "--dir", str(tmp_path / "plugins")]) == 0
    dest = tmp_path / "plugins" / "local-llm.30s.py"
    assert dest.exists() and dest.stat().st_mode & stat.S_IXUSR
    assert companion_install_plugin(tmp_path / "again", root=ROOT).read_bytes() == \
        PLUGIN.read_bytes()
    with pytest.raises(SystemExit):
        companion_cli(["bogus"])


# --------------------------------------------------------------------------- #
# SwiftBar plugin against a fake server                                       #
# --------------------------------------------------------------------------- #
PLUGIN_TOKEN = "llm_plugtest_" + "Z" * 40


class _FakeServer:
    """A tiny stdlib server that answers /api/status/summary and records posts."""

    def __init__(self, summary, *, want_token=PLUGIN_TOKEN):
        self.summary = summary
        self.want_token = want_token
        self.posts = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep pytest output clean
                pass

            def _send(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _authed(self):
                if outer.want_token is None:
                    return True
                return self.headers.get("Authorization") == f"Bearer {outer.want_token}"

            def do_GET(self):
                if self.path != COMPANION_SUMMARY_PATH:
                    return self._send(404, {"error": "not found"})
                if not self._authed():
                    return self._send(401, {"error": "authentication required"})
                return self._send(200, outer.summary)

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                outer.posts.append((self.path, self._authed()))
                return self._send(200 if self._authed() else 401, {"ok": True})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def _summary(**over):
    body = {
        "ui_build": "test", "server_time": 0,
        "indicator": {"state": "busy", "label": "queue 2", "color": "orange"},
        "model": {"status": "ready", "healthy": True, "id": "mlx-community/Qwen",
                  "alias": "qwen", "adapter": "v3", "backend": "mlx-lm",
                  "capabilities": ["tools", "vision"]},
        "queue": {"running": 1, "waiting": 2, "limit": 4},
        "approvals": {"pending": 1},
        "metrics": {"window": "1h", "scope": "all", "requests": 12, "errors": 0,
                    "ttft_p50_ms": 240.4, "decode_tps_p50": 41.26},
        "user": {"username": "admin", "role": "admin", "scopes": ["control"]},
        "controls": {"restart_model": True, "cancel_tasks": True, "retrain": False},
        "api_base_path": "/v1",
        "nodes": [{"name": "mini", "state": "healthy", "role": "worker", "active": True,
                   "local": True},
                  {"name": "studio", "state": "unreachable", "role": "primary",
                   "active": True, "local": False}],
        "multi_node": True,
        "tasks": {"running": 2, "running_ids": ["t1", "t/2"]},
        "retrain": {"running": False, "message": ""},
    }
    body.update(over)
    return body


def _plugin_env(tmp_path, url=None, token=None, **extra):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path / "home")}
    if url:
        env["LLM_URL"] = url
    if token:
        env["LLM_API_TOKEN"] = token
    env.update(extra)
    return env


def _run_plugin(env, *args):
    proc = subprocess.run([sys.executable, "-I", str(PLUGIN), *args], env=env,
                          capture_output=True, text=True, timeout=30)
    return proc


def test_plugin_renders_summary_without_leaking_token(tmp_path):
    with _FakeServer(_summary()) as server:
        proc = _run_plugin(_plugin_env(tmp_path, server.url, PLUGIN_TOKEN))
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert lines[0].startswith("● queue 2 |") and "color=#ff9500" in lines[0]
    assert lines[1] == "---"
    out = proc.stdout
    assert "Model: qwen (ready)" in out and "Adapter: v3" in out
    assert "Backend: mlx-lm (tools, vision)" in out
    assert "Queue: 1 running, 2 waiting, limit 4" in out
    assert "Last hour: 12 requests, TTFT 240 ms, 41.3 tok/s" in out
    assert "Nodes: 1 of 2 healthy" in out and "--studio: unreachable, primary" in out
    assert "Tasks running: 2" in out
    assert f"Open Web UI | href={server.url}" in out
    assert f"Approvals pending: 1 | href={server.url}" in out
    restart = next(line for line in lines if line.startswith("Restart Model"))
    assert "bash=" in restart and "param1=--action" in restart and "param2=restart" in restart
    assert any(line.startswith("Cancel Running Tasks (2)") for line in lines)
    assert any(line.startswith("Copy API URL") for line in lines)
    # The token is never part of what SwiftBar shows, caches or logs.
    assert PLUGIN_TOKEN not in out and PLUGIN_TOKEN not in proc.stderr


def test_plugin_hides_levers_the_token_cannot_pull(tmp_path):
    summary = _summary(controls={"restart_model": False, "cancel_tasks": False,
                                 "retrain": False}, nodes=None, tasks=None,
                       approvals={"pending": 0})
    with _FakeServer(summary) as server:
        proc = _run_plugin(_plugin_env(tmp_path, server.url, PLUGIN_TOKEN))
    assert proc.returncode == 0, proc.stderr
    assert "Restart Model" not in proc.stdout and "Cancel Running" not in proc.stdout
    assert "Approvals pending" not in proc.stdout and "Nodes:" not in proc.stdout


def test_plugin_problem_states(tmp_path):
    with _FakeServer(_summary()) as server:
        denied = _run_plugin(_plugin_env(tmp_path, server.url, "llm_wrong_" + "q" * 40))
        url = server.url
    assert denied.returncode == 0
    assert denied.stdout.splitlines()[0].startswith("● sign in")
    assert "llm_wrong_" not in denied.stdout
    # The server is gone now: offline, in red.
    offline = _run_plugin(_plugin_env(tmp_path, url, PLUGIN_TOKEN))
    assert offline.returncode == 0
    first = offline.stdout.splitlines()[0]
    assert first.startswith("● offline") and "#ff3b30" in first
    assert PLUGIN_TOKEN not in offline.stdout


def test_plugin_reads_cli_credentials_file(tmp_path):
    with _FakeServer(_summary()) as server:
        creds = tmp_path / "home" / ".config" / "local-llm" / "credentials.json"
        creds.parent.mkdir(parents=True)
        creds.write_text(json.dumps({"url": server.url, "token": PLUGIN_TOKEN}))
        proc = _run_plugin(_plugin_env(tmp_path))
        assert proc.stdout.splitlines()[0].startswith("● queue 2"), proc.stdout
        # The servers-map shape works too, and the environment wins over the file.
        creds.write_text(json.dumps({"default": server.url,
                                     "servers": {server.url: {"token": PLUGIN_TOKEN}}}))
        assert _run_plugin(_plugin_env(tmp_path)).stdout.startswith("● queue 2")
        env_wins = _run_plugin(_plugin_env(tmp_path, token="llm_env_" + "e" * 40))
        assert env_wins.stdout.startswith("● sign in")
    assert PLUGIN_TOKEN not in proc.stdout


def _fake_curl(tmp_path) -> tuple[Path, Path]:
    log = tmp_path / "curl.log"
    script = tmp_path / "fake-curl"
    script.write_text("#!/bin/sh\n"
                      f'printf "ARGS %s\\n" "$*" >> "{log}"\n'
                      f'printf "STDIN " >> "{log}"\n'
                      f'cat >> "{log}"\n'
                      'exit 0\n')
    script.chmod(0o755)
    return script, log


def test_plugin_actions_pass_token_through_stdin_not_argv(tmp_path):
    curl, log = _fake_curl(tmp_path)
    with _FakeServer(_summary()) as server:
        env = _plugin_env(tmp_path, server.url, PLUGIN_TOKEN, LLM_CURL=str(curl))
        restart = _run_plugin(env, "--action", "restart", "--yes")
        assert restart.returncode == 0, restart.stderr
        cancel = _run_plugin(env, "--action", "cancel-tasks", "--yes")
        assert cancel.returncode == 0, cancel.stderr
        assert _run_plugin(env, "--action", "nonsense").returncode == 2
    text = log.read_text()
    args = [line for line in text.splitlines() if line.startswith("ARGS ")]
    assert len(args) == 3
    assert f"{server.url}/api/model/restart" in args[0] and "-X POST" in args[0]
    assert f"{server.url}/api/tasks/t1/cancel" in args[1]
    assert f"{server.url}/api/tasks/t%2F2/cancel" in args[2]
    assert all("-K -" in a and PLUGIN_TOKEN not in a for a in args)
    assert text.count(f'header = "Authorization: Bearer {PLUGIN_TOKEN}"') == 3


# --------------------------------------------------------------------------- #
# Real compile (swiftc ships with the Xcode command line tools)               #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(sys.platform != "darwin" or companion_find_swiftc() is None,
                    reason="swiftc is not available")
def test_swiftc_compiles_the_companion(tmp_path):
    result = companion_build(tmp_path, arch="arm64", optimize=False, root=ROOT)
    app = result["app"]
    binary = app / "Contents" / "MacOS" / COMPANION_EXECUTABLE
    assert binary.exists() and os.access(binary, os.X_OK)
    assert binary.read_bytes()[:4] in (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe")
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    assert info["LSUIElement"] is True and info["CFBundleExecutable"] == COMPANION_EXECUTABLE
    if Path("/usr/bin/codesign").exists():
        check = subprocess.run(["/usr/bin/codesign", "--verify", str(app)],
                               capture_output=True, text=True)
        assert check.returncode == 0, check.stderr
