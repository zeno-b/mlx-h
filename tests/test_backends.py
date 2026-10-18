"""Tests for the pluggable inference backends (backends.py, ModelServerManager).

The load-bearing promise is backward compatibility: with no new setting, the
mlx_lm.server command line must be exactly what the pre-backend manager built.
test_default_command_matches_head runs HEAD's own _build_cmd (read with git)
against the same manager state and compares argv lists.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import local_llm.config as config_mod  # noqa: E402
config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-backends-"))

from local_llm.backends import (  # noqa: E402
    BACKEND_CAPABILITIES, BackendSpec, PROMPT_CACHE_LANES, _turbo_decode_cap,
    expand_backend_command, get_backend, lane_aware_cache_size, public_backends,
    register_backend, request_model_for, turbo_plan, _ext_available,
)
from local_llm.config import ADAPTER_DIR, Config  # noqa: E402
from local_llm.core import TOTAL_RAM_GB  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.sysutil import add_if_supported, help_has_flag  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# Every flag the manager knows, spelled the way argparse prints them, so the
# substring and whole-flag matchers agree and only the manager logic differs.
FULL_HELP = textwrap.dedent("""\
    usage: python -m mlx_lm.server [-h] [--model MODEL] [--adapter-path ADAPTER_PATH]
      --model MODEL
      --port PORT
      --max-kv-size MAX_KV_SIZE
      --kv-bits KV_BITS
      --kv-group-size KV_GROUP_SIZE
      --quantized-kv-start QUANTIZED_KV_START
      --prefill-step-size PREFILL_STEP_SIZE
      --prompt-cache-size PROMPT_CACHE_SIZE
      --prompt-cache-bytes PROMPT_CACHE_BYTES
      --decode-concurrency DECODE_CONCURRENCY
      --prompt-concurrency PROMPT_CONCURRENCY
      --prompt-cache-dir PROMPT_CACHE_DIR
      --adapter-path ADAPTER_PATH
      --draft-model DRAFT_MODEL
      --num-draft-tokens NUM_DRAFT_TOKENS
      --allowed-origins ALLOWED_ORIGINS
""")
# mlx-lm 0.32's shape: a prompt cache SIZE flag and no persistent cache flag.
HELP_032 = ("[--model MODEL] [--adapter-path ADAPTER_PATH] [--port PORT] "
            "[--allowed-origins ALLOWED_ORIGINS] [--prompt-cache-size PROMPT_CACHE_SIZE] "
            "[--prompt-cache-bytes PROMPT_CACHE_BYTES] [--kv-bits KV_BITS]")


def _manager(model: str = "org/test-model", help_text: str = FULL_HELP, **kw) -> ModelServerManager:
    mm = ModelServerManager(model, 8090, ADAPTER_DIR, **kw)
    mm._server_help = help_text
    return mm


def _head_source(path: str) -> str | None:
    try:
        out = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=REPO, capture_output=True,
                             text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _extract(source: str, name: str, cls: str | None = None) -> str:
    tree = ast.parse(source)
    nodes = tree.body
    if cls is not None:
        nodes = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    fn = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    return textwrap.dedent(ast.get_source_segment(source, fn))


def _head_build_cmd():
    """HEAD's ModelServerManager._build_cmd and add_if_supported, or None without git."""
    ms_src = _head_source("local_llm/model_server.py")
    su_src = _head_source("local_llm/sysutil.py")
    if not ms_src or not su_src:
        return None
    import logging
    ns: dict = {"sys": sys, "logging": logging, "log": lambda *a, **k: None,
                "Path": Path, "Any": object}
    exec(compile("from __future__ import annotations\n" + _extract(su_src, "add_if_supported"),
                 "<head-sysutil>", "exec"), ns)
    exec(compile("from __future__ import annotations\n"
                 + _extract(ms_src, "_build_cmd", "ModelServerManager"),
                 "<head-model_server>", "exec"), ns)
    return ns["_build_cmd"]


# --------------------------------------------------------------------------- #
# Backward compatibility of the default command line
# --------------------------------------------------------------------------- #
def test_default_command_matches_head():
    head = _head_build_cmd()
    cases = [
        ("defaults", FULL_HELP, {}, False),
        ("all levers", FULL_HELP, dict(max_kv_size=4096, kv_bits=8, kv_group_size=32,
                                       quantized_kv_start=512, prefill_step_size=256,
                                       prompt_cache_size=4, prompt_cache_bytes=1 << 30,
                                       decode_concurrency=2, prompt_concurrency=1), False),
        ("prompt cache dir", FULL_HELP, dict(prompt_cache_dir="/tmp/pc"), False),
        ("adapter", FULL_HELP, dict(prefill_step_size=512), True),
        ("no help", "", dict(kv_bits=8, prefill_step_size=512), True),
        ("old build", "--model --port", dict(prefill_step_size=512, decode_concurrency=2), False),
        ("0.32 shape", HELP_032, dict(prompt_cache_size=2, prompt_cache_bytes=1 << 29), True),
    ]
    adapter = Path(tempfile.mkdtemp(prefix="llm-test-adapter-"))
    for label, help_text, kw, use_adapter in cases:
        mm = _manager(help_text=help_text, **kw)
        mm.adapter_path = lambda: adapter
        now = mm._build_cmd(use_adapter)
        if head is not None:
            before = head(mm, use_adapter)
            assert now == before, f"{label}: {now} != HEAD {before}"
        assert now[:3] == [sys.executable, "-m", "mlx_lm.server"], now
    if head is None:   # no git (bundled copy): pin the plainest case literally
        mm = _manager()
        assert mm._build_cmd(False) == [sys.executable, "-m", "mlx_lm.server", "--model",
                                        "org/test-model", "--port", "8090"]


def test_cli_construction_without_new_settings_keeps_the_command():
    """What cli.py builds from a default Config, minus the web port, is HEAD's command."""
    head = _head_build_cmd()
    cfg = Config()
    mm = ModelServerManager(
        "org/test-model", cfg.model_port, ADAPTER_DIR, cfg.max_kv_size, cfg.adapter,
        kv_bits=cfg.kv_bits, prefill_step_size=cfg.prefill_step_size,
        prompt_cache_size=cfg.prompt_cache_size, prompt_cache_bytes=cfg.prompt_cache_bytes,
        decode_concurrency=cfg.decode_concurrency, prompt_concurrency=cfg.prompt_concurrency,
        backend=cfg.model_backend, draft_model=cfg.draft_model,
        num_draft_tokens=cfg.num_draft_tokens, backend_command=cfg.backend_command,
        backend_extra_args=cfg.backend_extra_args,
        max_concurrent_generations=cfg.max_concurrent_generations,
        context_size=cfg.context_size)
    mm._server_help = FULL_HELP
    assert mm.backend.name == "mlx-lm"
    if head is not None:
        assert mm._build_cmd(False) == head(mm, False)


# --------------------------------------------------------------------------- #
# Whole-flag matching (the "ambiguous option" startup failure)
# --------------------------------------------------------------------------- #
def test_add_if_supported_matches_whole_flags_only():
    cmd: list[str] = []
    assert not add_if_supported(cmd, HELP_032, ["--prompt-cache-dir", "--prompt-cache"], "/x")
    assert cmd == []
    assert add_if_supported(cmd, HELP_032, ["--adapter", "--adapter-path"], "/a")
    assert cmd == ["--adapter-path", "/a"], cmd
    assert help_has_flag("[--kv-bits KV_BITS]", "--kv-bits")
    assert help_has_flag("--kv-bits=KV_BITS", "--kv-bits")
    assert not help_has_flag("--kv-bits-extra", "--kv-bits")
    assert not help_has_flag("", "--kv-bits")
    assert help_has_flag("-lr LEARNING_RATE", "-lr")
    assert not help_has_flag("--slr X", "-lr")


def test_prompt_cache_dir_no_longer_breaks_mlx_lm_032():
    mm = _manager(help_text=HELP_032, prompt_cache_dir="/tmp/pc", prompt_cache_size=2)
    cmd = mm._build_cmd(False)
    assert "--prompt-cache" not in cmd, cmd
    assert "/tmp/pc" not in cmd, cmd
    assert cmd[cmd.index("--prompt-cache-size") + 1] == "2"


# --------------------------------------------------------------------------- #
# The request "model" value (adapter silently dropped on mlx-lm)
# --------------------------------------------------------------------------- #
def test_installed_mlx_lm_maps_only_default_model_to_the_cli_adapter():
    """Pins the mlx-lm behaviour the fix relies on (server.py ModelProvider.load)."""
    try:
        from mlx_lm.server import ModelProvider
    except Exception:  # noqa: BLE001 - not installed: nothing to pin
        return
    provider = object.__new__(ModelProvider)
    provider._model_map = {"default_model": "org/repo"}
    provider._adapter_map = {"default_model": "/adapters/latest"}
    provider._draft_model_map = {"default_model": None}
    provider.model = provider.tokenizer = None
    loads: list[tuple] = []

    def fake_load(model_path, adapter_path=None, draft_model_path=None):
        loads.append((model_path, adapter_path, draft_model_path))
        provider.model_key = (model_path, adapter_path, draft_model_path)

    provider._load = fake_load
    provider.model_key = ("org/repo", "/adapters/latest", None)   # the startup load
    provider.load("default_model", None, "default_model")
    assert loads == [], "default_model must reuse the CLI model and adapter"
    provider.load("org/repo", None, "default_model")
    assert loads == [("org/repo", None, None)], loads   # reloaded WITHOUT the adapter


def test_every_local_request_path_sends_the_backend_request_model():
    cfg = Config()
    assert request_model_for(cfg) == "default_model"
    from local_llm.model_client import ModelClient
    client = ModelClient(cfg)
    assert client.payload([{"role": "user", "content": "hi"}], False)["model"] == "default_model"

    mm = _manager()
    assert mm.request_model() == "default_model"
    import httpx
    sent: list[dict] = []

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    original = httpx.post
    httpx.post = lambda url, json=None, timeout=None: (sent.append(json), _Resp())[1]
    try:
        assert mm._probe_generation()
    finally:
        httpx.post = original
    assert sent[0]["model"] == "default_model"

    import urllib.request
    from local_llm.sysutil import wait_for_model_ready
    bodies: list[dict] = []

    class _Ok:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    real_urlopen = urllib.request.urlopen
    urllib.request.urlopen = lambda req, timeout=None: (bodies.append(json.loads(req.data)), _Ok())[1]
    try:
        wait_for_model_ready(mm.request_model(), 8090, timeout=5)
    finally:
        urllib.request.urlopen = real_urlopen
    assert bodies[0]["model"] == "default_model"

    ext = Config(model_backend="openai-external", backend_command="eng {model} {port}")
    assert request_model_for(ext) == ext.model
    assert ModelClient(ext).payload([{"role": "user", "content": "x"}], False)["model"] == ext.model


# --------------------------------------------------------------------------- #
# Allowed origins, extra args, unknown backend
# --------------------------------------------------------------------------- #
def test_allowed_origins_restricted_to_the_web_ui_when_supported():
    mm = _manager(web_port=8765)
    cmd = mm._build_cmd(False)
    value = cmd[cmd.index("--allowed-origins") + 1]
    assert value == "http://127.0.0.1:8765,http://localhost:8765,http://[::1]:8765", value
    assert "--allowed-origins" in mm.describe()["accepted_levers"]
    old = _manager(help_text="--model --port", web_port=8765)
    assert "--allowed-origins" not in old._build_cmd(False)


def test_extra_args_are_appended_and_bad_quoting_is_ignored():
    mm = _manager(backend_extra_args="--log-level DEBUG --temp '0.5'")
    assert mm._build_cmd(False)[-4:] == ["--log-level", "DEBUG", "--temp", "0.5"]
    bad = _manager(backend_extra_args="--x 'unbalanced")
    assert bad._build_cmd(False)[-2:] == ["--port", "8090"]


def test_unknown_backend_falls_back_to_mlx_lm():
    assert get_backend("no-such-engine").name == "mlx-lm"
    mm = _manager(backend="No-Such-Engine")
    assert mm.backend.name == "mlx-lm"
    info = mm.describe()
    assert info["backend"] == "mlx-lm" and info["backend_requested"] == "no-such-engine"
    assert any("no-such-engine" in d for d in info["decisions"]), info["decisions"]
    assert mm._build_cmd(False)[:3] == [sys.executable, "-m", "mlx_lm.server"]


def test_register_backend_rejects_unknown_capabilities():
    try:
        register_backend(BackendSpec(name="bad-caps", label="x", capabilities={"telepathy"}))
    except ValueError:
        pass
    else:
        raise AssertionError("unknown capability accepted")
    assert "bad-caps" not in [row["name"] for row in public_backends(Config())]


# --------------------------------------------------------------------------- #
# describe() and apply_config()
# --------------------------------------------------------------------------- #
def test_describe_reports_backend_capabilities_levers_and_decisions():
    mm = _manager(prefill_step_size=512, prompt_cache_size=2, max_concurrent_generations=2)
    mm._build_cmd(False)
    info = mm.describe()
    for key in ("model", "adapter", "status", "cached", "backend", "capabilities",
                "accepted_levers", "decisions", "prompt_cache", "request_model"):
        assert key in info, key
    assert info["status"] == "stopped"
    assert info["backend"] == "mlx-lm"
    assert set(info["capabilities"]) <= BACKEND_CAPABILITIES
    assert "batching" in info["capabilities"]
    assert "--prefill-step-size" in info["accepted_levers"]
    plan = info["prompt_cache"]
    assert plan["configured_size"] == 2 and plan["size"] == 2       # mlx-lm never resizes
    assert plan["recommended_size"] == lane_aware_cache_size(2)
    assert any("PROMPT_CACHE_SIZE=2" in d for d in info["decisions"]), info["decisions"]
    json.dumps(info)   # the route serialises it


def test_apply_config_reports_when_a_restart_is_due():
    mm = _manager()
    assert mm.apply_config(Config(model_backend="mlx-lm-turbo")) is True
    assert mm.backend.name == "mlx-lm-turbo"
    assert mm.apply_config(Config(model_backend="mlx-lm-turbo")) is False
    cfg = Config(model_backend="mlx-lm-turbo", watchdog_probe_timeout=40.0)
    assert mm.apply_config(cfg) is False and mm.probe_timeout == 40.0
    # Concurrency sizes turbo's command, but not plain mlx-lm's.
    conc = Config(model_backend="mlx-lm-turbo", max_concurrent_generations=5)
    assert mm.apply_config(conc) is True
    plain = _manager()
    plain.apply_config(Config())
    assert plain.apply_config(Config(max_concurrent_generations=5)) is False
    assert plain.apply_config(Config(draft_model="org/draft")) is True


# --------------------------------------------------------------------------- #
# mlx-lm-turbo decisions
# --------------------------------------------------------------------------- #
def test_turbo_batches_and_sizes_the_prompt_cache():
    mm = _manager(backend="mlx-lm-turbo", max_concurrent_generations=2,
                  decode_concurrency=2, prompt_concurrency=1, prompt_cache_size=2)
    cmd = mm._build_cmd(False)
    cap = _turbo_decode_cap(TOTAL_RAM_GB)
    assert cmd[cmd.index("--decode-concurrency") + 1] == str(min(cap, 3))
    want = min(64, lane_aware_cache_size(2))
    assert cmd[cmd.index("--prompt-cache-size") + 1] == str(want), cmd
    assert want >= len(PROMPT_CACHE_LANES) + 1 + 2
    plan = turbo_plan(mm)
    assert plan["batching"] and plan["draft_model"] == ""
    assert any("Continuous batching on" in d for d in plan["decisions"])
    assert "batching" in mm.describe()["capabilities"]
    assert mm._probe_timeout_now() == mm.probe_timeout


def test_turbo_without_batching_raises_the_probe_timeout():
    mm = _manager(backend="mlx-lm-turbo", kv_bits=8, max_concurrent_generations=2)
    plan = turbo_plan(mm)
    assert not plan["batching"]
    assert any("Continuous batching off" in d and "KV_BITS=8" in d for d in plan["decisions"])
    assert mm._probe_timeout_now() >= 90.0
    assert "batching" not in mm.describe()["capabilities"]


def test_turbo_draft_model_auto_yields_to_batching_and_explicit_draft_wins():
    auto = _manager(backend="mlx-lm-turbo", draft_model="auto", max_concurrent_generations=2)
    plan = turbo_plan(auto)
    assert plan["draft_model"] == "" and plan["batching"]
    assert any("DRAFT_MODEL=auto" in d for d in plan["decisions"])
    forced = _manager(backend="mlx-lm-turbo", draft_model="org/tiny-draft",
                      num_draft_tokens=4, max_concurrent_generations=2)
    cmd = forced._build_cmd(False)
    assert cmd[cmd.index("--draft-model") + 1] == "org/tiny-draft"
    assert cmd[cmd.index("--num-draft-tokens") + 1] == "4"
    assert not turbo_plan(forced)["batching"]
    assert "speculative" in forced.describe()["capabilities"]
    same = _manager(draft_model="org/test-model")
    assert "--draft-model" not in same._build_cmd(False)


# --------------------------------------------------------------------------- #
# openai-external
# --------------------------------------------------------------------------- #
def test_external_template_expansion():
    values = {"python": "/py", "model": "org/m", "port": "9000", "host": "127.0.0.1",
              "adapter": "", "draft_model": ""}
    tpl = ("{python} -m eng --model {model} --port {port} --adapter-path {adapter} "
           "--chat-template-args '{\"enable_thinking\": false}'")
    assert expand_backend_command(tpl, values) == [
        "/py", "-m", "eng", "--model", "org/m", "--port", "9000",
        "--chat-template-args", '{"enable_thinking": false}']
    with_adapter = dict(values, adapter="/ad")
    assert "--adapter-path" in expand_backend_command(tpl, with_adapter)
    assert expand_backend_command("eng --port={port} {model}", values) == ["eng", "--port=9000", "org/m"]
    for bad in ("", "eng {model}", "eng {model} {port} {secret}", "eng 'open {model} {port}"):
        try:
            expand_backend_command(bad, values)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_external_backend_launch_and_describe():
    mm = _manager(backend="openai-external", help_text="",
                  backend_command=f"{sys.executable} -m eng --model {{model}} --port {{port}}",
                  backend_capabilities="batching,vision,telepathy")
    assert mm._build_cmd(False) == [sys.executable, "-m", "eng", "--model", "org/test-model",
                                    "--port", "8090"]
    info = mm.describe()
    assert info["capabilities"] == ["batching", "vision"]
    assert info["adapter_served"] is False
    assert info["request_model"] == "org/test-model"
    assert any("{adapter}" in d for d in info["decisions"]), info["decisions"]
    ok, _ = _ext_available(Config(backend_command="definitely-not-a-binary-xyz {model} {port}"))
    assert not ok
    ok, reason = _ext_available(Config(backend_command=""))
    assert not ok and "BACKEND_COMMAND" in reason
    broken = _manager(backend="openai-external", help_text="", backend_command="eng {model}")
    try:
        broken._start_internal()
    except ValueError:
        pass
    else:
        raise AssertionError("a template without {port} started")
    assert broken.status.startswith("error:"), broken.status


# --------------------------------------------------------------------------- #
# Config, CLI, route
# --------------------------------------------------------------------------- #
def test_config_fields_are_restart_only_and_clamped():
    cfg = Config()
    assert cfg.model_backend == "mlx-lm" and cfg.draft_model == "" and cfg.num_draft_tokens == 0
    for name in ("model_backend", "draft_model", "num_draft_tokens", "backend_command",
                 "backend_capabilities", "backend_extra_args"):
        assert name not in Config.MUTABLE, name
    odd = Config(model_backend="  MLX-LM-Turbo ", num_draft_tokens=999,
                 backend_capabilities=" Batching , ,VISION ")
    assert odd.model_backend == "mlx-lm-turbo"
    assert odd.num_draft_tokens == 64
    assert odd.backend_capabilities == "batching,vision"
    assert Config(num_draft_tokens=-3).num_draft_tokens == 0


def test_cli_flags():
    from local_llm.cli import build_config, build_parser
    cfg = build_config(build_parser().parse_args([]))
    assert cfg.model_backend == "mlx-lm" and cfg.draft_model == ""
    cfg = build_config(build_parser().parse_args(
        ["--backend", "mlx-lm-turbo", "--draft-model", "auto", "--num-draft-tokens", "5"]))
    assert (cfg.model_backend, cfg.draft_model, cfg.num_draft_tokens) == ("mlx-lm-turbo", "auto", 5)


def test_backends_route():
    from fastapi.testclient import TestClient
    from local_llm.api import create_app
    from local_llm.database import Database
    from local_llm.tools import ToolRegistry
    from local_llm.training import RetrainManager
    cfg = Config(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                 allow_test_user=False, node_token="secret-node-token",
                 skills_dir=tempfile.mkdtemp(prefix="local-llm-test-skills-"))
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)  # never started
    app = create_app(cfg, db, mm, RetrainManager(db, mm, cfg), ToolRegistry(cfg, db))
    app.state.model_manager = mm
    with TestClient(app) as client:
        assert client.get("/api/backends").status_code in (401, 403)
        client.post("/api/auth/login", json={"username": "admin", "password": "adminpw123"})
        r = client.get("/api/backends")
        assert r.status_code == 200, r.text
        data = r.json()
        names = [row["name"] for row in data["backends"]]
        assert names[:3] == ["mlx-lm", "mlx-lm-turbo", "openai-external"], names
        assert [row["name"] for row in data["backends"] if row["current"]] == ["mlx-lm"]
        ext = next(row for row in data["backends"] if row["name"] == "openai-external")
        assert ext["available"] is False and "BACKEND_COMMAND" in ext["reason"]
        assert data["configured"] == "mlx-lm"
        assert data["current"]["backend"] == "mlx-lm"
        assert data["current"]["request_model"] == "default_model"
        assert set(data["capability_vocabulary"]) == BACKEND_CAPABILITIES


def test_adapter_served_needs_an_active_adapter():
    """mlx-lm CAN serve the LoRA, but with no adapter active nothing is served."""
    mm = _manager()
    mm.adapter_path = lambda: None
    assert mm.describe()["adapter_served"] is False
    mm.adapter_path = lambda: Path(tempfile.mkdtemp(prefix="w2a-adapter-"))
    info = mm.describe()
    assert info["adapter_active"] is True and info["adapter_served"] is True


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
