"""Tests for the model catalogue, aliases and the hardware benchmark sweep.

Nothing is built in: what a model is comes from its own files in the Hugging
Face cache (tests/fake_models.py lays fake ones out in a temporary cache) and a
guess from the repo name before that, and the only aliases are MODEL_ALIASES.
Nothing here loads a model: the sweep runs against fake managers and a fake
OpenAI-compatible streaming server on a loopback port.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import local_llm.config as config_mod  # noqa: E402
config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-catalog-"))
# Applying a recommendation writes .env; never the real one, also when this
# file runs standalone without conftest.
os.environ["LOCAL_LLM_ENV_FILE"] = str(Path(tempfile.mkdtemp(prefix="llm-test-catalog-env-"))
                                       / ".env")

import local_llm.catalog as catalog_mod  # noqa: E402
from local_llm.catalog import (  # noqa: E402
    downloaded_models, draft_for, fits_in_ram, model_aliases, model_metadata,
    model_snapshot_dir, parse_model_aliases, read_recommendation, resolve_model,
    write_recommendation,
)
from local_llm.config import Config  # noqa: E402
from local_llm.core import ADAPTER_DIR, _default_context_for_ram  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.training import RetrainManager  # noqa: E402
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.sysutil import get_free_port, model_catalog, port_open  # noqa: E402
from local_llm.cluster import ClusterRouter, NodeRegistry  # noqa: E402
from local_llm.diagnostics import (  # noqa: E402
    _bench_apply, _bench_sizes, apply_recommendation, bench_matrix, benchmark,
    locate_model_server, recommend,
)
from local_llm.envfile import dotenv_path, read_dotenv  # noqa: E402
from fake_models import (  # noqa: E402
    LLAMA_TEMPLATE, QWEN25_TEMPLATE, QWEN3_TEMPLATE, fake_hf_cache, fake_model,
)

# catalog binds core's DATA_DIR at import; point it (bench files, the
# recommendation, the saved-settings lookup) at a temp dir for every test here.
catalog_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-catalog-data-"))

TINY = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
FAST = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
SMALL = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
SEVEN = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
FOURTEEN = "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"
LARGE = "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit"


def _fresh_data_dir() -> Path:
    path = Path(tempfile.mkdtemp(prefix="llm-test-catalog-data-"))
    catalog_mod.DATA_DIR = path
    return path


@contextmanager
def _env(**values):
    """Set (or, with None, unset) environment variables for the block."""
    saved = {name: os.environ.get(name) for name in values}
    for name, value in values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# --------------------------------------------------------------------------- #
# What a model is: its own files, else its name
# --------------------------------------------------------------------------- #

def test_metadata_is_read_from_the_downloaded_files():
    with fake_hf_cache() as hub:
        fake_model(hub, SEVEN, size_gb=4.3, context=32768, template=QWEN25_TEMPLATE)
        meta = model_metadata(SEVEN)
        required = {"id", "family", "params_b", "quant", "approx_size_gb", "min_ram_gb",
                    "context_max", "thinking", "tool_format", "vision", "draft", "notes"}
        assert required <= set(meta)
        assert meta["local"] is True and meta["guessed"] is False
        assert meta["family"] == "qwen2" and meta["quant"] == "4bit"
        assert meta["approx_size_gb"] == 4.3 and meta["params_b"] == 7
        # 4.3GB + runtime does not fit an 8GB Mac's GPU share: the floor is 16.
        assert meta["min_ram_gb"] == 16 and meta["context_max"] == 32768
        assert meta["thinking"] is False and meta["tool_format"] == "hermes"
        assert meta["vision"] is False and meta["vocab_size"] == 151936
        assert model_metadata(SEVEN) is not model_metadata(SEVEN)   # a fresh dict each time


def test_metadata_reads_thinking_tools_and_vision_from_the_model_files():
    with fake_hf_cache() as hub:
        fake_model(hub, "mlx-community/Qwen3-8B-4bit", size_gb=4.6, model_type="qwen3",
                   template=QWEN3_TEMPLATE)
        fake_model(hub, "mlx-community/Llama-3.2-3B-Instruct-4bit", size_gb=1.8,
                   model_type="llama", vocab_size=128256, template=LLAMA_TEMPLATE)
        fake_model(hub, "mlx-community/Qwen2.5-VL-3B-Instruct-4bit", size_gb=2.0,
                   model_type="qwen2_5_vl", vision=True)
        snap = fake_model(hub, "someone/coder-next", size_gb=2.25, template=None)
        (snap / "chat_template.jinja").write_text("<function=run>{{ args }}</function>")
        assert model_metadata("mlx-community/Qwen3-8B-4bit")["thinking"] is True
        llama = model_metadata("mlx-community/Llama-3.2-3B-Instruct-4bit")
        assert llama["thinking"] is False and llama["tool_format"] == "llama3"
        assert model_metadata("mlx-community/Qwen2.5-VL-3B-Instruct-4bit")["vision"] is True
        # The template may live in chat_template.jinja. A name with no size in
        # it gets its parameter count from the weights, marked as an estimate.
        coder = model_metadata("someone/coder-next")
        assert coder["tool_format"] == "qwen3_coder"
        assert coder["approximate"] is True and 4.0 < coder["params_b"] < 4.5


def test_downloaded_models_lists_complete_repos_and_follows_refs_main():
    with fake_hf_cache() as hub:
        first = fake_model(hub, SMALL, size_gb=1.7)
        fake_model(hub, "someone/half-downloaded-7B-4bit", weights=False)
        fake_model(hub, "someone/no-tokenizer-1B-4bit", tokenizer=False)
        (hub / "models--someone--empty").mkdir()
        assert downloaded_models() == [SMALL]
        assert model_snapshot_dir(SMALL) == first
        assert model_snapshot_dir("someone/never-downloaded") is None
        assert model_snapshot_dir("") is None and model_snapshot_dir("not-a-repo") is None
        # A newer snapshot is served only once refs/main names it, and its
        # files are read afresh rather than from the first snapshot's entry.
        second = fake_model(hub, SMALL, size_gb=1.9,
                            revision="0000000000000000000000000000000000000002")
        assert model_snapshot_dir(SMALL) == second
        assert model_metadata(SMALL)["approx_size_gb"] == 1.9
        refs = hub / ("models--" + SMALL.replace("/", "--")) / "refs" / "main"
        refs.write_text(first.name)
        assert model_snapshot_dir(SMALL) == first
        assert model_metadata(SMALL)["approx_size_gb"] == 1.7
        # Without refs/main, the newest complete snapshot.
        refs.unlink()
        assert model_snapshot_dir(SMALL) == second
        # A local folder is a model when it holds a config.json.
        assert model_snapshot_dir(str(first)) == first
        assert model_snapshot_dir(str(hub)) is None


def test_a_model_not_downloaded_is_estimated_from_its_name():
    with fake_hf_cache():
        guessed = model_metadata(LARGE)
        assert guessed["guessed"] is True and guessed["local"] is False
        assert guessed["params_b"] == 32 and guessed["approx_size_gb"] == 17.6
        # Unknown until the files are on disk, never assumed.
        assert guessed["thinking"] is None and guessed["vision"] is None
        assert model_metadata(LARGE, guess=False) is None
        assert fits_in_ram(LARGE, 8) is False and fits_in_ram(LARGE, 64) is True
        assert fits_in_ram(SMALL, 8) is True
        assert fits_in_ram(SMALL, 8, headroom_gb=4) is False
        assert model_metadata("someone/Foo-70B-4bit")["params_b"] == 70
        assert fits_in_ram("someone/Foo-70B-4bit", 16) is False
        assert model_metadata("someone/SmolLM-135M-8bit")["params_b"] == 0.135
        assert model_metadata("someone/no-size-in-name") is None
        assert fits_in_ram("someone/no-size-in-name", 8) is None


def test_draft_for_is_a_downloaded_quarter_size_sibling_with_the_same_tokenizer():
    with fake_hf_cache() as hub:
        fake_model(hub, SEVEN, size_gb=4.3)
        fake_model(hub, TINY, size_gb=0.28)
        fake_model(hub, FAST, size_gb=0.87)
        fake_model(hub, "mlx-community/Llama-3.2-1B-Instruct-4bit", size_gb=0.7,
                   model_type="llama", vocab_size=128256, template=LLAMA_TEMPLATE)
        assert draft_for(SEVEN) == TINY          # the smallest one that qualifies
        assert draft_for(FAST) is None           # nothing a quarter its size or less
        assert draft_for(TINY) is None
        assert draft_for(LARGE) is None          # not downloaded: nothing to compare
        assert draft_for("nobody/unknown-model") is None


def test_backends_auto_draft_finds_catalog_draft_for():
    from local_llm.backends import _backends_catalog_draft_for
    finder = _backends_catalog_draft_for()
    assert finder is not None
    with fake_hf_cache() as hub:
        fake_model(hub, FOURTEEN, size_gb=8.3)
        fake_model(hub, TINY, size_gb=0.28)
        assert finder(FOURTEEN) == draft_for(FOURTEEN) == TINY


# --------------------------------------------------------------------------- #
# The served model and aliases: MODEL_ID and MODEL_ALIASES, nothing built in
# --------------------------------------------------------------------------- #

def test_the_served_model_is_model_id_and_nothing_else():
    with _env(MODEL_ID=None):
        assert Config().model == ""
    with _env(MODEL_ID=" org/some-model "):
        assert Config().model == "org/some-model"
    assert all(entry["id"] for entry in model_catalog(Config(model="")))


def test_resolve_model_uses_only_configured_aliases():
    none = Config(model_aliases="")
    assert resolve_model("large", none) == "large" and resolve_model("coder", none) == "coder"
    assert model_aliases(none) == {}
    assert resolve_model(SMALL, none) == SMALL          # repo ids are never aliases
    assert resolve_model("", none) == ""
    cfg = Config(model_aliases=json.dumps({"coder": "me/my-coder", "chat": "Coder",
                                           "loop": "loop2", "loop2": "loop"}))
    assert resolve_model("  CODER ", cfg) == "me/my-coder"
    assert resolve_model("chat", cfg) == "me/my-coder"   # chains to another alias
    assert resolve_model("loop", cfg) in ("loop", "loop2")  # a cycle terminates
    table = model_aliases(cfg)
    assert table["coder"] == {"id": "me/my-coder", "source": "env"}
    assert table["chat"]["id"] == "me/my-coder"


def test_parse_model_aliases_rejects_bad_shapes():
    assert parse_model_aliases("") == ({}, "")
    assert parse_model_aliases('{"A": " x/y "}') == ({"a": "x/y"}, "")
    for bad in ("{bad", "[1, 2]", '{"a": 3}', '{"a": ""}', '{"x/y": "z"}', '"str"'):
        aliases, error = parse_model_aliases(bad)
        assert aliases == {} and error, bad


def test_invalid_model_aliases_rejected_previous_value_kept():
    assert Config(model_aliases="{not json").model_aliases == ""
    cfg = Config(model_aliases='{"chat": "a/b"}')
    assert cfg.apply({"model_aliases": "{broken"}) == []
    assert cfg.model_aliases == '{"chat": "a/b"}'
    assert cfg.apply({"model_aliases": '["list"]'}) == []
    assert cfg.model_aliases == '{"chat": "a/b"}'
    assert cfg.apply({"model_aliases": '{"chat": "c/d"}'}) == ["model_aliases"]
    assert resolve_model("chat", cfg) == "c/d"
    assert cfg.apply({"model_aliases": ""}) == ["model_aliases"]   # clearing is allowed


def test_aliases_resolve_before_large_model_markers():
    from local_llm.cli import build_config, build_parser
    with _env(MODEL_ALIASES=json.dumps({"big": LARGE, "my14b": SMALL})):
        cfg = build_config(build_parser().parse_args(["--model", "big"]))
        assert cfg.model == LARGE and cfg.model_alias == "big"
        router = ClusterRouter(cfg, NodeRegistry(cfg))
        assert router.classify().needs_large_model is True
        # An alias whose NAME matches a marker routes by what it points at.
        cfg = build_config(build_parser().parse_args(["--model", "my14b"]))
        assert cfg.model == SMALL and cfg.model_alias == "my14b"
        assert ClusterRouter(cfg, NodeRegistry(cfg)).classify().needs_large_model is False
        # A repo id is untouched and records no alias.
        cfg = build_config(build_parser().parse_args(["--model", SMALL]))
        assert cfg.model == SMALL and cfg.model_alias == ""
        # Without --model, MODEL_ID decides, and it may be an alias too.
        with _env(MODEL_ID=SEVEN):
            cfg = build_config(build_parser().parse_args([]))
            assert cfg.model == SEVEN and cfg.model_alias == ""
        with _env(MODEL_ID="big"):
            cfg = build_config(build_parser().parse_args([]))
            assert cfg.model == LARGE and cfg.model_alias == "big"


def test_model_catalog_entries_are_additive():
    entries = model_catalog(Config(model="coder", model_aliases=json.dumps({"coder": SEVEN})))
    first = entries[0]
    assert {"id", "cached", "current"} <= set(first)
    assert {"aliases", "meta", "fits_in_ram", "recommended"} <= set(first)
    current = [e for e in entries if e["current"]]
    assert [e["id"] for e in current] == [SEVEN]
    assert "coder" in current[0]["aliases"]
    assert len({e["id"] for e in entries}) == len(entries)


# --------------------------------------------------------------------------- #
# Recommendation logic (synthetic results)
# --------------------------------------------------------------------------- #

def _row(model_id, tps, peak_mb, context=4096, backend="mlx-lm", ok=True):
    return {"model_id": model_id, "context_size": context, "backend": backend,
            "kv_bits": 0, "success": ok, "decode_tps": tps, "prompt_tps": 300.0,
            "ttft_ms": 200.0, "peak_rss_mb": peak_mb}


def test_recommend_highest_quality_that_fits_and_is_fast_enough():
    rows = [_row(SMALL, 40, 2500), _row(SEVEN, 15, 5000), _row(FOURTEEN, 6, 9000),
            _row("x/broken-70B", 0, 0, ok=False)]
    with _env(MODEL_ALIASES=json.dumps({"coder": SEVEN})):
        rec = recommend(rows, 8.0, min_decode_tps=10)
    assert rec["model_id"] == SEVEN and rec["alias"] == "coder"
    assert set(rec) >= {"model_id", "alias", "context_size", "kv_bits", "backend", "reason"}
    assert recommend(rows, 8.0, min_decode_tps=20)["model_id"] == SMALL
    # Plenty of RAM but the 14B is too slow: still the 7B.
    assert recommend(rows, 64.0, min_decode_tps=10)["model_id"] == SEVEN
    assert recommend(rows, 64.0, min_decode_tps=5)["model_id"] == FOURTEEN
    # Explicit headroom shrinks the budget.
    assert recommend(rows, 8.0, min_decode_tps=10, headroom_gb=4.0)["model_id"] == SMALL


def test_recommend_prefers_larger_context_then_faster_backend():
    rows = [_row(SMALL, 30, 2400, context=4096), _row(SMALL, 25, 3000, context=8192),
            _row(SMALL, 35, 3000, context=8192, backend="mlx-lm-turbo")]
    rec = recommend(rows, 8.0, min_decode_tps=10)
    assert (rec["context_size"], rec["backend"]) == (8192, "mlx-lm-turbo")


def test_recommend_fallbacks():
    rows = [_row(SMALL, 4, 2500), _row(SEVEN, 2, 5000)]
    rec = recommend(rows, 8.0, min_decode_tps=10)
    assert rec["model_id"] == SMALL and "no run reached" in rec["reason"]
    rec = recommend(rows, 4.0, min_decode_tps=1)
    assert rec["model_id"] is None and "nothing fit" in rec["reason"]
    assert recommend([], 8.0)["model_id"] is None


# --------------------------------------------------------------------------- #
# The recommendation file: a record, not a setting
# --------------------------------------------------------------------------- #

def test_recommendation_file_round_trip_and_machine_stamp():
    data = _fresh_data_dir()
    path = data / "recommendation.json"
    assert read_recommendation(data) is None
    written = write_recommendation({"model_id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
                                    "context_size": 8192}, data_dir=data, ram_gb=8.0)
    assert written == path
    on_disk = read_recommendation(data, ram_gb=8.0)
    assert on_disk["context_size"] == 8192 and on_disk["ram_gb"] == 8.0 and on_disk["applied_at"]
    # A file stamped for another machine size is ignored.
    assert read_recommendation(data, ram_gb=64.0) is None
    assert read_recommendation(data)["model_id"] == "mlx-community/Llama-3.2-3B-Instruct-4bit"
    # It is not a default anything falls back on: the context window is the
    # RAM tier until CONTEXT_SIZE says otherwise, and the model is MODEL_ID.
    assert _default_context_for_ram(8.0) == 4096 and _default_context_for_ram(64.0) == 32768
    try:
        write_recommendation({"model_id": ""}, data_dir=data)
    except ValueError:
        pass
    else:
        raise AssertionError("a recommendation without a model was written")
    # Garbage on disk never breaks startup.
    path.write_text("{not json")
    assert read_recommendation(data) is None


def test_apply_writes_the_pick_into_env_and_keeps_everything_else():
    data = _fresh_data_dir()
    env_file = data / "app.env"
    env_file.write_text("# my settings\nMODEL_ID=old/model   # the old pick\nAUTH_ENABLED=1\n"
                        "# CONTEXT_SIZE=8192\nLOG_LEVEL=INFO\n", encoding="utf-8")
    with _env(MODEL_ID=None, CONTEXT_SIZE=None):
        done = apply_recommendation({"model_id": SMALL, "context_size": 8192},
                                    data_dir=data, env_file=env_file)
        # The running process is left alone; the next start reads the file.
        assert os.environ.get("MODEL_ID") is None
    assert done["updates"] == {"MODEL_ID": SMALL, "CONTEXT_SIZE": 8192}
    assert done["shadowed"] == [] and done["env_file"] == str(env_file)
    values = read_dotenv(env_file)
    assert values["MODEL_ID"] == SMALL and values["CONTEXT_SIZE"] == "8192"
    lines = env_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# my settings" and "AUTH_ENABLED=1" in lines and "LOG_LEVEL=INFO" in lines
    # A new key lands right after its commented example, so it stays in its section.
    assert lines[lines.index("# CONTEXT_SIZE=8192") + 1] == "CONTEXT_SIZE=8192"
    assert read_recommendation(data)["model_id"] == SMALL
    # A value exported in the shell wins over the file: written, and reported.
    with _env(MODEL_ID="shell/model"):
        done = apply_recommendation({"model_id": SEVEN}, data_dir=data, env_file=env_file)
    assert done["shadowed"] == ["MODEL_ID"] and read_dotenv(env_file)["MODEL_ID"] == SEVEN
    try:
        apply_recommendation({"model_id": None}, data_dir=data, env_file=env_file)
    except ValueError:
        pass
    else:
        raise AssertionError("a recommendation without a model was applied")


def test_recommend_apply_cli_says_what_it_wrote():
    import contextlib
    import io
    _fresh_data_dir()
    env_file = Path(tempfile.mkdtemp(prefix="llm-test-catalog-env-")) / ".env"
    out = io.StringIO()
    with _env(LOCAL_LLM_ENV_FILE=str(env_file), MODEL_ID=None, CONTEXT_SIZE=None), \
            contextlib.redirect_stdout(out):
        assert _bench_apply({"model_id": SMALL, "context_size": 4096, "backend": "mlx-lm",
                             "kv_bits": 0}) == 0
        assert _bench_apply({"model_id": None}) == 1
    assert f"Wrote MODEL_ID={SMALL} CONTEXT_SIZE=4096 to {env_file}" in out.getvalue()
    assert "Nothing applied." in out.getvalue()
    assert read_dotenv(env_file)["MODEL_ID"] == SMALL


# --------------------------------------------------------------------------- #
# benchmark() and the sweep, against a fake streaming server
# --------------------------------------------------------------------------- #

class _FakeOpenAI(BaseHTTPRequestHandler):
    def log_message(self, *_a):  # quiet
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append(body)
        prompt = len(body["messages"][0]["content"]) // 4
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for word in ("The", " fox", " ran", " far", "."):
            chunk = {"choices": [{"delta": {"content": word}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.002)
        usage = {"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": 5}}
        self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())


def _fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    server.requests = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class _FakeProc:
    def __init__(self, pid):
        self.pid = pid


class _FakeManagers:
    """Factory for fake managers; records starts/stops and concurrency."""

    def __init__(self, port, fail=()):
        self.port = port
        self.fail = set(fail)
        self.events = []
        self.alive = 0
        self.max_alive = 0
        self.pids = {}

    def __call__(self, model_id, port, context, backend, config):
        factory = self

        class Manager:
            model_port = factory.port

            def __init__(self):
                self.model_id = model_id
                self.proc = None
                self.status = "stopped"

            def start(self):
                factory.events.append(("start", model_id, context, backend, port))
                if model_id in factory.fail:
                    raise RuntimeError("Model server exited immediately with code 1.")
                factory.alive += 1
                factory.max_alive = max(factory.max_alive, factory.alive)
                pid = 100000 + len(factory.events)
                factory.pids[pid] = model_id
                self.proc = _FakeProc(pid)
                self.status = "ready"

            def stop(self):
                factory.events.append(("stop", model_id))
                if self.proc is not None:
                    factory.alive -= 1
                self.proc = None
                self.status = "stopped"

        return Manager()


def test_benchmark_sizes_follow_prompts_parameter():
    assert _bench_sizes(3, 512) == [64, 512, 2048]      # the historical default
    assert _bench_sizes(1, 512) == [64]
    assert _bench_sizes(4, 100) == [64, 100, 400, 1600]


def test_benchmark_against_fake_server_collects_samples():
    server = _fake_server()
    try:
        port = server.server_address[1]
        cfg = Config(model="cli/model", model_port=port)
        samples: list = []
        lines: list = []
        code = benchmark(cfg, prompts=2, prompt_tokens=128, collect=samples,
                         out=lambda *a, **_k: lines.append(" ".join(map(str, a))))
        assert code == 0
        assert [s["requested"] for s in samples] == [64, 128]
        assert all(s["tps"] > 0 and s["prompt_tps"] > 0 for s in samples)
        assert len(server.requests) == 2
    finally:
        server.shutdown()


def test_locate_model_server_reads_the_running_command_line():
    import shutil
    if not shutil.which("lsof"):
        return
    port = get_free_port(23000)
    code = ("import socket,sys,time\n"
            "s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
            "s.bind(('127.0.0.1', int(sys.argv[2]))); s.listen(); time.sleep(30)")
    proc = subprocess.Popen([sys.executable, "-c", code, "--port", str(port),
                             "--model", "org/running-model"])
    try:
        for _ in range(100):
            if port_open(port):
                break
            time.sleep(0.05)
        preferred = port - 3
        if any(port_open(p) for p in range(preferred, port)):
            preferred = port
        found = locate_model_server(preferred, "cli/env-model")
        assert found["port"] == port, found
        assert found["model"] == "org/running-model" and found["how"] == "process"
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_bench_matrix_one_server_at_a_time_and_saves():
    data = _fresh_data_dir()
    server = _fake_server()
    try:
        managers = _FakeManagers(server.server_address[1], fail={FAST})
        cfg = Config(model=SMALL, model_aliases=json.dumps({"tiny": TINY, "fast": FAST}))
        rss = {}

        def sampler(pid):
            model = managers.pids.get(pid, "")
            return rss.setdefault(pid, 900.0 if "0.5B" in model else 2100.0)

        doc = bench_matrix(cfg, ["tiny", SMALL, "fast", LARGE, "tiny"], [1024], ["mlx-lm"],
                           manager_factory=managers, port_finder=lambda preferred: 45678,
                           rss_sampler=sampler, allow_download=True, out=lambda *_a: None)
        assert managers.max_alive == 1 and managers.alive == 0
        starts = [e for e in managers.events if e[0] == "start"]
        stops = [e for e in managers.events if e[0] == "stop"]
        assert len(starts) == len(stops)
        assert all(e[4] == 45678 for e in starts)
        by_model = {r["model_id"]: r for r in doc["results"]}
        assert len(doc["results"]) == 4                      # "tiny" twice = one cell
        tiny = by_model[TINY]
        assert tiny["success"] and tiny["alias"] == "tiny" and tiny["peak_rss_mb"] == 900.0
        assert tiny["decode_tps"] > 0 and tiny["prompt_tps"] > 0 and tiny["ttft_ms"] >= 0
        failed = by_model[FAST]
        assert failed["success"] is False and "exited" in failed["error"]
        large = by_model[LARGE]
        if fits_in_ram(LARGE) is False:
            assert large.get("skipped") and "RAM" in large["error"]
        saved = list((data / "bench").glob("*.json"))
        assert [p.name for p in saved] == [doc["file"]]
        on_disk = json.loads(saved[0].read_text())
        assert on_disk["results"] == json.loads(json.dumps(doc["results"]))
        assert on_disk["recommendation"]["model_id"] in (SMALL, TINY)
    finally:
        server.shutdown()


def test_bench_matrix_skips_uncached_without_download():
    _fresh_data_dir()
    managers = _FakeManagers(1)
    with fake_hf_cache():
        doc = bench_matrix(Config(model=SMALL), ["nobody/never-downloaded-1B-4bit"], [2048],
                           ["mlx-lm"], manager_factory=managers, out=lambda *_a: None)
    assert managers.events == []
    assert doc["results"][0]["skipped"] and "not downloaded" in doc["results"][0]["error"]


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

def _build_app(**overrides):
    from local_llm.api import create_app
    base = dict(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                allow_test_user=False, node_token="secret-node-token",
                skills_dir=tempfile.mkdtemp(prefix="llm-test-catalog-skills-"))
    base.update(overrides)
    cfg = Config(**base)
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)  # never started
    mm.restart = lambda: None       # a switch must never launch a real server here
    rm = RetrainManager(db, mm, cfg)
    app = create_app(cfg, db, mm, rm, ToolRegistry(cfg, db))
    app.state.model_manager = mm
    return app, cfg, db


def _login(client, username, password):
    r = client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text


def test_routes_rbac():
    from fastapi.testclient import TestClient
    app, _cfg, _db = _build_app(model_aliases=json.dumps({"large": LARGE}))
    admin_only = ["/api/bench/results", "/api/bench/recommendation", "/api/bench/status"]
    with TestClient(app) as anon:
        assert anon.get("/api/models/aliases").status_code == 401
        for path in admin_only:
            assert anon.get(path).status_code == 401, path
        assert anon.post("/api/bench/run", json={"confirm": True}).status_code == 401
    with TestClient(app) as admin:
        _login(admin, "admin", "adminpw123")
        assert admin.post("/api/users", json={"username": "bob", "password": "bobpass123",
                                              "role": "user"}).status_code in (200, 201)
    with TestClient(app) as user:
        _login(user, "bob", "bobpass123")
        r = user.get("/api/models/aliases")
        assert r.status_code == 200 and r.json()["aliases"]["large"]["id"] == LARGE
        for path in admin_only:
            assert user.get(path).status_code == 403, path
        assert user.post("/api/bench/run", json={"confirm": True}).status_code == 403
        assert user.post("/api/models/select", json={"model": "large"}).status_code == 403


def test_models_routes_accept_aliases_and_report_metadata():
    from fastapi.testclient import TestClient
    _fresh_data_dir()
    aliases = {"chat": "fast", "fast": FAST}
    app, cfg, _db = _build_app(model_aliases=json.dumps(aliases))
    with TestClient(app) as c:
        _login(c, "admin", "adminpw123")
        body = c.get("/api/models").json()
        assert {"current", "catalog", "adapters", "cache_dir"} <= set(body)
        assert body["aliases"]["chat"] == {"id": FAST, "source": "env"}   # through the chain
        assert body["recommended"] is None and body["ram_gb"] > 0
        assert all("fits_in_ram" in e and "meta" in e for e in body["catalog"])
        r = c.post("/api/models/select", json={"model": "chat", "restart": False})
        assert r.status_code == 200, r.text
        assert r.json()["resolved"] == FAST
        assert cfg.model == FAST and cfg.model_alias == "chat"
        assert app.state.model_manager.model_id == cfg.model
        r = c.post("/api/models/select", json={"model": SMALL, "restart": False})
        assert r.json()["resolved"] == SMALL and cfg.model_alias == ""
        # Invalid aliases over /api/config are refused and reported.
        r = c.post("/api/config", json={"model_aliases": "{oops"})
        assert r.status_code == 200 and "model_aliases" in r.json()["ignored"]
        assert json.loads(cfg.model_aliases) == aliases
        # A sweep in progress owns the model server.
        app.state.bench_job.running = True
        try:
            assert c.post("/api/models/select", json={"model": "fast"}).status_code == 409
        finally:
            app.state.bench_job.running = False


def test_bench_run_requires_confirm_and_refuses_while_busy():
    from fastapi.testclient import TestClient
    app, _cfg, _db = _build_app()
    ctx = app.state.ctx

    class BusyGate:
        def snapshot(self):
            return {"running": 1, "waiting": 0}

    with TestClient(app) as c:
        _login(c, "admin", "adminpw123")
        r = c.post("/api/bench/run", json={})
        assert r.status_code == 400 and "confirm" in r.json()["error"]
        assert c.post("/api/bench/run", json={"confirm": True, "contexts": [10]}).status_code == 400
        real_gate = ctx.generation_gate
        ctx.generation_gate = BusyGate()
        try:
            r = c.post("/api/bench/run", json={"confirm": True})
            assert r.status_code == 409 and "generation" in r.json()["error"]
        finally:
            ctx.generation_gate = real_gate
        app.state.ctx.retrain_manager.status["running"] = True
        try:
            assert c.post("/api/bench/run", json={"confirm": True}).status_code == 409
        finally:
            app.state.ctx.retrain_manager.status["running"] = False


def test_bench_run_end_to_end_with_fakes():
    from fastapi.testclient import TestClient
    data = _fresh_data_dir()
    server = _fake_server()
    try:
        app, _cfg, _db = _build_app(model_aliases=json.dumps({"tiny": TINY}))
        ctx = app.state.ctx
        managers = _FakeManagers(server.server_address[1])
        ctx.extras["bench_manager_factory"] = managers
        ctx.extras["bench_port_finder"] = lambda preferred: 45679
        ctx.extras["bench_rss_sampler"] = lambda pid: 1500.0
        with TestClient(app) as c:
            _login(c, "admin", "adminpw123")
            assert c.get("/api/bench/recommendation").json()["computed"] is None
            r = c.post("/api/bench/run", json={"confirm": True, "models": ["tiny", SMALL],
                                               "contexts": [2048], "backends": ["mlx-lm"],
                                               "apply": True, "download": True})
            assert r.status_code == 202, r.text
            for _ in range(200):
                status = c.get("/api/bench/status").json()
                if not status["running"]:
                    break
                time.sleep(0.05)
            assert status["running"] is False and status["error"] == "", status
            assert status["file"] and status["applied"] is True
            assert managers.max_alive == 1 and managers.alive == 0
            # The never-started serving manager is not started afterwards.
            assert app.state.model_manager.proc is None
            runs = c.get("/api/bench/results").json()
            assert runs["runs"][0]["file"] == status["file"] and runs["runs"][0]["succeeded"] == 2
            one = c.get("/api/bench/results", params={"file": status["file"]}).json()
            assert len(one["run"]["results"]) == 2
            assert c.get("/api/bench/results", params={"file": "../x.json"}).status_code == 400
            rec = c.get("/api/bench/recommendation").json()
            assert rec["computed"]["model_id"] == SMALL         # higher quality, both fast
            assert rec["applied"]["model_id"] == SMALL and rec["applied"]["context_size"] == 2048
            assert (data / "recommendation.json").is_file()
            # Applied means written into .env, the one place the served model comes from.
            assert status["env_file"] == str(dotenv_path())
            assert read_dotenv(dotenv_path())["MODEL_ID"] == SMALL
            assert read_dotenv(dotenv_path())["CONTEXT_SIZE"] == "2048"
    finally:
        server.shutdown()


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
