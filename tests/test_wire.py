"""Tests for `deploy.py wire` (client wiring) and /api/integrations/snippets."""
from __future__ import annotations

import io
import json
import os
import stat
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_llm.config as config_mod  # noqa: E402

config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-wire-"))

import local_llm.wire as wire_mod  # noqa: E402
from local_llm.wire import WIRE_CLIENTS, WIRE_TOKEN_PLACEHOLDER, build_wiring, wire_cli  # noqa: E402

TOKEN = "llm_ab12cd34_" + "A1b2C3d4E5" * 4
URL = "http://127.0.0.1:9"     # never contacted: --no-probe or a patched _wire_http


def _no_network(*a, **kw):
    raise AssertionError("wire must not touch the network in this test")


def _run(argv, home, http=_no_network):
    out, err = io.StringIO(), io.StringIO()
    saved = wire_mod._wire_http
    wire_mod._wire_http = http
    try:
        code = wire_cli(argv, out=out, err=err, home=home)
    finally:
        wire_mod._wire_http = saved
    return code, out.getvalue(), err.getvalue()


def _home():
    return Path(tempfile.mkdtemp(prefix="wire-home-"))


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def _backups(path):
    return sorted(path.parent.glob(path.name + ".bak-*"))


# --------------------------------------------------------------------------- #
def test_print_is_default_and_writes_nothing_for_every_client():
    expect = {
        "claude-code": ["ANTHROPIC_BASE_URL=http://127.0.0.1:9", "ANTHROPIC_AUTH_TOKEN=" + TOKEN,
                        "ANTHROPIC_MODEL=raw/coder", "ANTHROPIC_SMALL_FAST_MODEL=raw/coder",
                        "~/.claude/settings.json"],
        "aider": ["OPENAI_API_BASE=http://127.0.0.1:9/v1", "OPENAI_API_KEY=" + TOKEN,
                  "aider --model openai/raw/coder", "openai-api-base:"],
        "codex": ["[model_providers.local]", 'base_url = "http://127.0.0.1:9/v1"',
                  'env_key = "LLM_API_TOKEN"', 'wire_api = "responses"', 'model = "raw/coder"',
                  "export LLM_API_TOKEN=" + TOKEN],
        "continue": ["provider: \"openai\"", "apiBase: \"http://127.0.0.1:9/v1\"",
                     "model: \"raw/coder\"", "~/.continue/config.yaml"],
        "opencode": ['"@ai-sdk/openai-compatible"', '"baseURL": "http://127.0.0.1:9/v1"',
                     '"model": "local-llm/raw/coder"'],
        "hermes": ["OPENAI_BASE_URL=http://127.0.0.1:9/v1", "OPENAI_API_KEY=" + TOKEN,
                   "could not be confirmed offline"],
        "generic": ["OPENAI_BASE_URL=http://127.0.0.1:9/v1", "ANTHROPIC_BASE_URL=http://127.0.0.1:9",
                    "ANTHROPIC_API_KEY=" + TOKEN],
    }
    assert set(expect) == set(WIRE_CLIENTS)
    for client, needles in expect.items():
        home = _home()
        code, out, err = _run([client, "--url", URL + "/", "--token", TOKEN, "--model", "coder",
                               "--no-probe"], home)
        assert code == 0, (client, err)
        for needle in needles:
            assert needle in out, (client, needle, out)
        assert list(home.iterdir()) == [], client     # --print never writes


def test_print_without_token_uses_placeholder_and_codex_chat_wire_api():
    code, out, err = _run(["codex", "--url", URL, "--no-probe", "--wire-api", "chat",
                           "--model", "raw/coder"], _home())
    assert code == 0 and WIRE_TOKEN_PLACEHOLDER in out and 'wire_api = "chat"' in out
    assert "--login" in err


def test_write_claude_code_merges_env_and_backs_up():
    home = _home()
    settings = home / ".claude" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"theme": "dark", "env": {"FOO": "1",
                                    "ANTHROPIC_MODEL": "old"}, "permissions": {"allow": ["x"]}}))
    code, out, err = _run(["claude-code", "--url", URL, "--token", TOKEN, "--model", "coder",
                           "--write", "--no-probe"], home)
    assert code == 0, err
    data = json.loads(settings.read_text())
    assert data["theme"] == "dark" and data["permissions"] == {"allow": ["x"]}
    assert data["env"]["FOO"] == "1"
    assert data["env"]["ANTHROPIC_MODEL"] == "raw/coder"
    assert data["env"]["ANTHROPIC_AUTH_TOKEN"] == TOKEN
    assert data["env"]["ANTHROPIC_BASE_URL"] == URL
    assert _mode(settings) == 0o600
    backups = _backups(settings)
    assert len(backups) == 1 and json.loads(backups[0].read_text())["env"]["ANTHROPIC_MODEL"] == "old"
    assert _mode(backups[0]) == 0o600
    assert "backup:" in out


def test_write_creates_private_dirs_and_files():
    home = _home()
    code, out, err = _run(["continue", "--url", URL, "--token", TOKEN, "--model", "coder",
                           "--write", "--no-probe"], home)
    assert code == 0, err
    cfg = home / ".continue" / "config.yaml"
    assert _mode(cfg.parent) == 0o700 and _mode(cfg) == 0o600
    text = cfg.read_text()
    assert "models:" in text and f'apiKey: "{TOKEN}"' in text and "schema: v1" in text
    assert _backups(cfg) == []
    # opencode's nested ~/.config/opencode: every created level is 0700.
    code, _, err = _run(["opencode", "--url", URL, "--token", TOKEN, "--write", "--no-probe",
                         "--model", "coder"], home)
    assert code == 0, err
    oc = home / ".config" / "opencode" / "opencode.json"
    assert _mode(home / ".config") == 0o700 and _mode(oc.parent) == 0o700 and _mode(oc) == 0o600


def test_write_codex_toml_keeps_unrelated_tables():
    import tomllib
    home = _home()
    cfg = home / ".codex" / "config.toml"
    cfg.parent.mkdir()
    cfg.write_text('# my codex\nmodel = "o3"\napproval_policy = "on-request"\n\n'
                   '[mcp_servers.docs]\ncommand = "npx"\nargs = ["-y", "docs"]\n\n'
                   '[model_providers.local]\nname = "old"\nbase_url = "http://old/v1"\n'
                   'extra = 1\n\n[profiles.fast]\nmodel = "gpt-x"\n')
    os.chmod(cfg, 0o644)
    code, out, err = _run(["codex", "--url", URL, "--token", TOKEN, "--model", "coder",
                           "--write", "--no-probe"], home)
    assert code == 0, err
    data = tomllib.loads(cfg.read_text())
    assert data["model"] == "raw/coder" and data["model_provider"] == "local"
    assert data["approval_policy"] == "on-request"
    assert data["mcp_servers"]["docs"]["args"] == ["-y", "docs"]
    assert data["profiles"]["fast"]["model"] == "gpt-x"
    assert data["model_providers"]["local"] == {
        "name": "Local LLM", "base_url": URL + "/v1", "env_key": "LLM_API_TOKEN",
        "wire_api": "responses"}
    assert TOKEN not in cfg.read_text()          # Codex reads it from the env
    assert "# my codex" in cfg.read_text()
    assert _mode(cfg) == 0o644                   # no token: mode kept
    assert f"export LLM_API_TOKEN={TOKEN}" in out
    # Running it again is idempotent (one provider table, same content).
    before = cfg.read_text()
    assert _run(["codex", "--url", URL, "--token", TOKEN, "--model", "coder", "--write",
                 "--no-probe"], home)[0] == 0
    assert cfg.read_text() == before and len(_backups(cfg)) == 2


def test_write_aider_flat_yaml_merge():
    home = _home()
    cfg = home / ".aider.conf.yml"
    cfg.write_text("# aider settings\nmodel: gpt-4o\nauto-commits: false\nread:\n  - CONVENTIONS.md\n")
    code, _, err = _run(["aider", "--url", URL, "--token", TOKEN, "--model", "coder",
                         "--write", "--no-probe"], home)
    assert code == 0, err
    text = cfg.read_text()
    assert 'model: "openai/raw/coder"' in text and "gpt-4o" not in text
    assert "auto-commits: false" in text and "  - CONVENTIONS.md" in text
    assert f'openai-api-key: "{TOKEN}"' in text and "show-model-warnings: false" in text
    assert _mode(cfg) == 0o600


def test_write_continue_replaces_only_our_entry():
    home = _home()
    cfg = home / ".continue" / "config.yaml"
    cfg.parent.mkdir()
    cfg.write_text("name: Mine\nversion: 0.0.1\nmodels:\n  - name: Other\n    provider: ollama\n"
                   "    model: llama3\n  - name: Local LLM\n    provider: openai\n"
                   "    model: raw/old\n    roles:\n      - chat\ncontext:\n  - provider: code\n")
    for _ in range(2):
        code, _, err = _run(["continue", "--url", URL, "--token", TOKEN, "--model", "coder",
                             "--write", "--no-probe"], home)
        assert code == 0, err
    text = cfg.read_text()
    assert text.count("name: \"Local LLM\"") == 1 and "raw/old" not in text
    assert "  - name: Other\n    provider: ollama" in text
    assert "context:\n  - provider: code" in text and "name: Mine" in text
    try:
        import yaml  # optional: validate the structure when PyYAML is around
    except ImportError:
        return
    data = yaml.safe_load(text)
    assert [m["name"] for m in data["models"]] == ["Other", "Local LLM"]
    assert data["models"][1]["apiBase"] == URL + "/v1"


def test_write_opencode_keeps_other_providers():
    home = _home()
    cfg = home / ".config" / "opencode" / "opencode.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"theme": "x", "provider": {"ollama": {"name": "Ollama"}}}))
    code, _, err = _run(["opencode", "--url", URL, "--token", TOKEN, "--model", "coder",
                         "--write", "--no-probe"], home)
    assert code == 0, err
    data = json.loads(cfg.read_text())
    assert data["theme"] == "x" and data["provider"]["ollama"] == {"name": "Ollama"}
    assert data["provider"]["local-llm"]["options"]["apiKey"] == TOKEN
    assert data["model"] == "local-llm/raw/coder"


def test_unparseable_file_is_refused_untouched():
    home = _home()
    settings = home / ".claude" / "settings.json"
    settings.parent.mkdir()
    settings.write_text("{not json")
    code, out, err = _run(["claude-code", "--url", URL, "--token", TOKEN, "--write",
                           "--no-probe", "--model", "coder"], home)
    assert code == 1 and "Nothing written" in err
    assert settings.read_text() == "{not json" and _backups(settings) == []


def test_write_refusals():
    home = _home()
    code, _, err = _run(["hermes", "--url", URL, "--token", TOKEN, "--write", "--no-probe",
                         "--model", "coder"], home)
    assert code == 2 and "not supported" in err
    code, _, err = _run(["claude-code", "--url", URL, "--write", "--no-probe", "--model", "c"],
                        home)
    assert code == 2 and "needs a token" in err
    assert list(home.iterdir()) == []
    code, out, _ = _run([], home)
    assert code == 2 and "claude-code" in out


def test_login_flow_and_server_probe():
    calls = []

    def fake_http(method, url, body=None, headers=None, timeout=10.0):
        calls.append((method, url, body, headers))
        if url.endswith("/api/auth/config"):
            return 200, {"auth_enabled": True}
        if url.endswith("/api/auth/cli/login"):
            ok = body == {"username": "dev", "password": "pw",
                          "name": body.get("name")} and body["name"].startswith("wire aider")
            return (200, {"token": TOKEN}) if ok else (401, {"error": "bad"})
        if url.endswith("/v1/models"):
            assert headers == {"Authorization": f"Bearer {TOKEN}"}
            return 200, {"data": [{"id": "agent/default"}, {"id": "raw/fast"}]}
        raise AssertionError(url)

    import builtins
    import getpass
    orig = (builtins.input, getpass.getpass)
    builtins.input = lambda prompt="": "dev"
    getpass.getpass = lambda prompt="": "pw"
    try:
        code, out, err = _run(["aider", "--url", URL, "--login"], _home(), http=fake_http)
    finally:
        builtins.input, getpass.getpass = orig
    assert code == 0, err
    assert f"OPENAI_API_KEY={TOKEN}" in out and "aider --model openai/raw/fast" in out
    assert [c[1].rsplit("/", 2)[-1] for c in calls] == ["config", "login", "models"]


def test_auth_disabled_server_needs_no_token():
    def fake_http(method, url, body=None, headers=None, timeout=10.0):
        if url.endswith("/api/auth/config"):
            return 200, {"auth_enabled": False}
        return 0, {}
    code, out, err = _run(["generic", "--url", URL, "--model", "coder"], _home(), http=fake_http)
    assert code == 0 and "OPENAI_API_KEY=no-auth-needed" in out
    # Remote + auth off is called out loudly.
    code, out, err = _run(["generic", "--url", "http://10.0.0.5:8000", "--model", "coder"],
                          _home(), http=fake_http)
    assert code == 0 and "anyone who can reach it is an administrator" in err


def test_plan_rejects_unknown_client_and_normalises_models():
    try:
        build_wiring("vim", URL, TOKEN)
        raise AssertionError("unknown client must raise")
    except ValueError:
        pass
    assert build_wiring("aider", URL + "/v1", TOKEN, "agent/research")["model"] == "agent/research"
    assert build_wiring("aider", URL, TOKEN, "Qwen/Qwen3-8B")["model"] == "raw/Qwen/Qwen3-8B"
    assert build_wiring("aider", URL, TOKEN, None)["model"].startswith("raw/")


def test_wire_subcommand_is_registered():
    from local_llm.cli import subcommands
    assert subcommands()["wire"] is wire_cli


def test_snippets_route_uses_placeholder_only():
    from fastapi.testclient import TestClient
    from local_llm.api import create_app
    from local_llm.config import Config
    from local_llm.core import ADAPTER_DIR
    from local_llm.database import Database
    from local_llm.model_server import ModelServerManager
    from local_llm.tools import ToolRegistry
    from local_llm.training import RetrainManager
    cfg = Config(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                 allow_test_user=False, node_token="secret-node-token",
                 skills_dir=tempfile.mkdtemp(prefix="local-llm-test-skills-"))
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)
    app = create_app(cfg, db, mm, RetrainManager(db, mm, cfg), ToolRegistry(cfg, db))
    client = TestClient(app)
    assert client.get("/api/integrations/snippets?client=aider").status_code == 401
    assert client.post("/api/auth/login", json={"username": "admin",
                                                "password": "adminpw123"}).status_code == 200
    listing = client.get("/api/integrations/snippets").json()
    assert {c["id"] for c in listing["clients"]} == set(WIRE_CLIENTS)
    r = client.get("/api/integrations/snippets?client=claude-code")
    assert r.status_code == 200, r.text
    body = r.json()
    assert WIRE_TOKEN_PLACEHOLDER in body["text"] and "llm_" not in body["text"]
    assert body["url"] == "http://testserver" and body["model"].startswith("raw/")
    assert body["files"][0]["path"] == "~/.claude/settings.json"
    assert client.get("/api/integrations/snippets?client=vim").status_code == 400
    # A chat-scoped API token may read it too (the CLI's own scope).
    token = client.post("/api/auth/tokens", json={}).json()["token"]
    bare = TestClient(app)
    assert bare.get("/api/integrations/snippets?client=codex",
                    headers={"Authorization": f"Bearer {token}"}).status_code == 200


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
