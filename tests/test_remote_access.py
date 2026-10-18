"""Tests for remote access: bind policy, trusted proxies, public URL, `deploy.py remote`."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_llm.config as config_mod  # noqa: E402

config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-remote-"))

from fastapi.testclient import TestClient  # noqa: E402

from local_llm.config import Config  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.training import RetrainManager  # noqa: E402
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.core import ADAPTER_DIR  # noqa: E402
from local_llm.api import create_app  # noqa: E402
from local_llm import remote_access  # noqa: E402
from local_llm.remote_access import (  # noqa: E402
    bind_host_or_exit, bind_host_problem, client_ip_for, public_url_for, remote_cli)
from local_llm.cli import build_config, build_parser, subcommands  # noqa: E402


def _req(peer: str, xff: str = "", scheme: str = "http", netloc: str = "127.0.0.1:8000"):
    headers = {"x-forwarded-for": xff} if xff else {}
    return SimpleNamespace(client=SimpleNamespace(host=peer), headers=headers,
                           url=SimpleNamespace(scheme=scheme, netloc=netloc))


def _run(argv, runner=None, config=None):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        code = remote_cli(argv, runner=runner, config=config or Config())
    return code, out.getvalue()


class _Runner:
    def __init__(self, status_json: dict | None = None):
        self.calls: list[list[str]] = []
        self.status_json = status_json or {"Self": {"DNSName": "mini.tail1234.ts.net.",
                                                    "TailscaleIPs": ["100.64.0.7"]},
                                           "BackendState": "Running"}

    def __call__(self, cmd):
        self.calls.append(list(cmd))
        if cmd[1:3] == ["status", "--json"]:
            return 0, json.dumps(self.status_json)
        return 0, "ok"


# --------------------------------------------------------------------------- #
# Client IP behind proxies
# --------------------------------------------------------------------------- #
def test_client_ip_defaults_to_socket_peer():
    cfg = Config()
    assert client_ip_for(_req("203.0.113.9", "1.2.3.4"), cfg) == "203.0.113.9"


def test_client_ip_behind_trusted_proxy_walks_from_the_right():
    cfg = Config(trusted_proxies="127.0.0.1, 10.0.0.0/8")
    assert client_ip_for(_req("127.0.0.1", "198.51.100.4"), cfg) == "198.51.100.4"
    # A forged left-most entry is ignored; the hop the proxy appended wins.
    assert client_ip_for(_req("127.0.0.1", "6.6.6.6, 198.51.100.4"), cfg) == "198.51.100.4"
    # Further trusted hops are skipped.
    assert client_ip_for(_req("127.0.0.1", "198.51.100.4, 10.1.2.3"), cfg) == "198.51.100.4"
    # An untrusted peer's header is never believed.
    assert client_ip_for(_req("203.0.113.9", "1.1.1.1"), cfg) == "203.0.113.9"
    # Garbage stops the walk: fall back to the peer.
    assert client_ip_for(_req("127.0.0.1", "not-an-ip"), cfg) == "127.0.0.1"


def test_public_url_for_prefers_public_base_url():
    req = _req("127.0.0.1", netloc="127.0.0.1:8000")
    assert public_url_for(req, Config(), "/cb") == "http://127.0.0.1:8000/cb"
    cfg = Config(public_base_url="https://llm.example.com/")
    assert public_url_for(req, cfg, "/api/auth/oidc/callback") == \
        "https://llm.example.com/api/auth/oidc/callback"


def test_login_throttle_keys_on_real_client_behind_trusted_proxy():
    cfg = Config(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                 allow_test_user=False, trusted_proxies="10.0.0.0/8",
                 skills_dir=tempfile.mkdtemp(prefix="llm-test-remote-skills-"))
    db = Database(Path(tempfile.mkdtemp()) / "t.db")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)
    app = create_app(cfg, db, mm, RetrainManager(db, mm, cfg), ToolRegistry(cfg, db))
    with TestClient(app, client=("10.9.9.9", 4321)) as c:
        statuses = []
        for _ in range(40):
            r = c.post("/api/auth/login", json={"username": "admin", "password": "nope"},
                       headers={"X-Forwarded-For": "192.0.2.77"})
            statuses.append(r.status_code)
            if r.status_code == 429:
                break
        assert statuses[-1] == 429 and statuses[0] == 401
        # A different real client behind the same proxy is not locked out.
        r = c.post("/api/auth/login", json={"username": "admin", "password": "nope"},
                   headers={"X-Forwarded-For": "192.0.2.78"})
        assert r.status_code == 401


# --------------------------------------------------------------------------- #
# Bind policy
# --------------------------------------------------------------------------- #
def test_bind_policy():
    assert bind_host_problem(Config()) == ("", "")
    assert bind_host_problem(Config(web_host="::1")) == ("", "")
    refusal, _ = bind_host_problem(Config(web_host="0.0.0.0", auth_enabled=False))
    assert "AUTH_ENABLED" in refusal
    refusal, warning = bind_host_problem(Config(web_host="0.0.0.0", auth_enabled=True))
    assert not refusal and "AUTH_COOKIE_SECURE" in warning
    assert bind_host_problem(Config(web_host="0.0.0.0", auth_enabled=True,
                                    auth_cookie_secure=True)) == ("", "")
    try:
        bind_host_or_exit(Config(web_host="192.168.1.10", auth_enabled=False))
    except SystemExit:
        pass
    else:
        raise AssertionError("a public bind without auth must exit")
    assert bind_host_or_exit(Config()) == "127.0.0.1"
    assert Config(web_host="").web_host == "127.0.0.1"


def test_cli_host_flag_and_subcommands():
    args = build_parser().parse_args(["--host", "0.0.0.0"])
    assert build_config(args).web_host == "0.0.0.0"
    assert build_config(build_parser().parse_args([])).web_host == "127.0.0.1"
    commands = subcommands()
    assert "cluster" in commands and "remote" in commands


# --------------------------------------------------------------------------- #
# deploy.py remote ...
# --------------------------------------------------------------------------- #
def test_remote_tailscale_dry_run_runs_nothing_but_status():
    saved = remote_access.tailscale_binary
    remote_access.tailscale_binary = lambda: "/usr/local/bin/tailscale"
    try:
        runner = _Runner()
        code, out = _run(["tailscale"], runner)
        assert code == 0
        assert f"/usr/local/bin/tailscale serve --bg {Config().web_port}" in out
        assert "PUBLIC_BASE_URL=https://mini.tail1234.ts.net" in out
        assert "TRUSTED_PROXIES=127.0.0.1,::1" in out
        assert all(call[1] == "status" for call in runner.calls)
        code, out = _run(["tailscale", "--apply", "--port", "8123"], runner)
        assert code == 0
        assert ["/usr/local/bin/tailscale", "serve", "--bg", "8123"] in runner.calls
        # Funnel publishes to the internet: refused with auth off even on --apply.
        runner = _Runner()
        code, out = _run(["funnel", "--apply"], runner, Config(auth_enabled=False))
        assert code == 2
        assert not any(call[1] == "funnel" for call in runner.calls)
        assert "NO Tailscale identity" in out
    finally:
        remote_access.tailscale_binary = saved


def test_remote_cloudflare_dry_run_and_notes():
    runner = _Runner()
    code, out = _run(["cloudflare", "--hostname", "llm.example.com"], runner)
    assert code == 0 and runner.calls == []
    assert f"tunnel run --url http://127.0.0.1:{Config().web_port} local-llm" in out
    assert "Cloudflare Access" in out
    assert "PUBLIC_BASE_URL=https://llm.example.com" in out


def test_remote_gen_key_and_status():
    code, out = _run(["gen-key"])
    assert code == 0
    lines = dict(line.split("=", 1) for line in out.splitlines()
                 if line and not line.startswith("#"))
    assert len(lines["NODE_TOKEN"]) >= 40 and len(lines["NODE_E2E_KEY"]) == 43
    code, out = _run(["status"], _Runner(), Config(web_host="0.0.0.0", auth_enabled=False))
    assert code == 1 and "refusing to listen" in out
    code, out = _run(["status"], _Runner(), Config())
    assert code == 0 and "loopback only" in out


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
