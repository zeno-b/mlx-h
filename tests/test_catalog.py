"""Tests for the model catalogue, aliases and the hardware benchmark sweep.

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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import local_llm.config as config_mod  # noqa: E402
config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-catalog-"))

import local_llm.catalog as catalog_mod  # noqa: E402
import local_llm.core as core_mod  # noqa: E402
from local_llm.catalog import (  # noqa: E402
    BUILTIN_MODEL_ALIASES, MODEL_METADATA, draft_for, fits_in_ram, model_aliases,
    model_metadata, parse_model_aliases, read_recommendation, resolve_model,
    write_recommendation,
)
from local_llm.config import Config  # noqa: E402
from local_llm.core import ADAPTER_DIR, TOTAL_RAM_GB  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.training import RetrainManager  # noqa: E402
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.sysutil import DEFAULT_MODEL_CATALOG, get_free_port, model_catalog, port_open  # noqa: E402
from local_llm.cluster import ClusterRouter, NodeRegistry  # noqa: E402
from local_llm.diagnostics import (  # noqa: E402
    _bench_sizes, bench_matrix, benchmark, locate_model_server, recommend,
)

# catalog binds core's DATA_DIR at import; point it (bench files, the
# recommendation, the saved-settings lookup) at a temp dir for every test here.
catalog_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-catalog-data-"))

LARGE = BUILTIN_MODEL_ALIASES["large"]
SMALL = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"


def _fresh_data_dir() -> Path:
    path = Path(tempfile.mkdtemp(prefix="llm-test-catalog-data-"))
    catalog_mod.DATA_DIR = path
    return path


# --------------------------------------------------------------------------- #
# Catalogue metadata and aliases
# --------------------------------------------------------------------------- #

def test_metadata_covers_default_catalog_and_ram_tier_defaults():
    ids = {entry["id"] for entry in MODEL_METADATA}
    for model_id in DEFAULT_MODEL_CATALOG:
        assert model_id in ids, model_id
    for ram in (8, 16, 32, 64):
        assert core_mod._default_model_for_ram(ram, Path("/nonexistent/rec.json")) in ids
    required = {"id", "family", "params_b", "quant", "approx_size_gb", "min_ram_gb",
                "context_max", "thinking", "tool_format", "vision", "draft", "notes"}
    for entry in MODEL_METADATA:
        assert required <= set(entry), entry["id"]
        if entry["draft"]:
            # A draft is smaller and the same family's tokenizer.
            draft = model_metadata(entry["draft"])
            assert draft and draft["params_b"] < entry["params_b"], entry["id"]
    for name, target in BUILTIN_MODEL_ALIASES.items():
        assert target in ids, (name, target)
    assert set(BUILTIN_MODEL_ALIASES) == {"fast", "balanced", "coder", "reasoning",
                                          "vision", "tiny", "large"}


def test_draft_for_and_fits_in_ram():
    assert draft_for("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    assert draft_for(SMALL) is None
    assert draft_for("nobody/unknown-model") is None
    assert fits_in_ram(model_metadata(LARGE), 8) is False
    assert fits_in_ram(model_metadata(LARGE), 64) is True
    assert fits_in_ram(SMALL, 8) is True
    assert fits_in_ram(SMALL, 8, headroom_gb=4) is False
    # Unknown repo: estimated from the name; nothing to go on -> unknown.
    guessed = model_metadata("someone/Foo-70B-4bit")
    assert guessed["guessed"] and guessed["params_b"] == 70 and fits_in_ram(guessed, 16) is False
    assert fits_in_ram("someone/no-size-in-name", 8) is None


def test_backends_auto_draft_finds_catalog_draft_for():
    from local_llm.backends import _backends_catalog_draft_for
    finder = _backends_catalog_draft_for()
    assert finder is not None
    assert finder("mlx-community/Qwen2.5-Coder-14B-Instruct-4bit") == draft_for(
        "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit")


def test_resolve_model_builtin_user_and_passthrough():
    assert resolve_model("large") == LARGE
    assert resolve_model("  Coder ") == BUILTIN_MODEL_ALIASES["coder"]
    assert resolve_model(SMALL) == SMALL                    # repo ids are never aliases
    assert resolve_model("not-an-alias") == "not-an-alias"
    assert resolve_model("") == ""
    cfg = Config(model_aliases=json.dumps({"coder": "me/my-coder", "chat": "fast",
                                           "loop": "loop2", "loop2": "loop"}))
    assert resolve_model("coder", cfg) == "me/my-coder"     # user overrides built-in
    assert resolve_model("chat", cfg) == BUILTIN_MODEL_ALIASES["fast"]   # chains to built-in
    assert resolve_model("loop", cfg) in ("loop", "loop2")  # a cycle terminates
    table = model_aliases(cfg)
    assert table["coder"] == {"id": "me/my-coder", "source": "user"}
    assert table["chat"]["id"] == BUILTIN_MODEL_ALIASES["fast"]
    assert table["large"]["source"] == "builtin"


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
    old = os.environ.get("MODEL_ALIASES")
    try:
        os.environ["MODEL_ALIASES"] = json.dumps({"my14b": SMALL})
        cfg = build_config(build_parser().parse_args(["--model", "large"]))
        assert cfg.model == LARGE and cfg.model_alias == "large"
        router = ClusterRouter(cfg, NodeRegistry(cfg))
        assert router.classify().needs_large_model is True
        # A user alias whose NAME matches a marker routes by what it points at.
        cfg = build_config(build_parser().parse_args(["--model", "my14b"]))
        assert cfg.model == SMALL and cfg.model_alias == "my14b"
        assert ClusterRouter(cfg, NodeRegistry(cfg)).classify().needs_large_model is False
        # A repo id is untouched and records no alias.
        cfg = build_config(build_parser().parse_args(["--model", SMALL]))
        assert cfg.model == SMALL and cfg.model_alias == ""
    finally:
        if old is None:
            os.environ.pop("MODEL_ALIASES", None)
        else:
            os.environ["MODEL_ALIASES"] = old


def test_model_catalog_entries_are_additive():
    entries = model_catalog(Config(model="coder"))
    first = entries[0]
    assert {"id", "cached", "current"} <= set(first)
    assert {"aliases", "meta", "fits_in_ram", "recommended"} <= set(first)
    current = [e for e in entries if e["current"]]
    assert [e["id"] for e in current] == [BUILTIN_MODEL_ALIASES["coder"]]
    assert "coder" in current[0]["aliases"]
    assert len({e["id"] for e in entries}) == len(entries)


# --------------------------------------------------------------------------- #
# Recommendation logic (synthetic results)
# --------------------------------------------------------------------------- #

def _row(model_id, tps, peak_mb, context=4096, backend="mlx-lm", ok=True):
    return {"model_id": model_id, "context_size": context, "backend": backend,
            "kv_bits": 0, "success": ok, "decode_tps": tps, "prompt_tps": 300.0,
            "ttft_ms": 200.0, "peak_rss_mb": peak_mb}


SEVEN = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
FOURTEEN = "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"


def test_recommend_highest_quality_that_fits_and_is_fast_enough():
    rows = [_row(SMALL, 40, 2500), _row(SEVEN, 15, 5000), _row(FOURTEEN, 6, 9000),
            _row("x/broken-70B", 0, 0, ok=False)]
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
# The recommendation file and the RAM-tier defaults
# --------------------------------------------------------------------------- #

def _without_env(*names):
    saved = {n: os.environ.pop(n) for n in names if n in os.environ}

    def restore():
        for n in names:
            os.environ.pop(n, None)
        os.environ.update(saved)
    return restore


def test_recommendation_file_read_by_ram_tier_defaults():
    restore = _without_env("MODEL_ID", "CONTEXT_SIZE")
    try:
        data = _fresh_data_dir()
        path = data / "recommendation.json"
        assert core_mod._default_model_for_ram(8.0, path) == SMALL          # no file: tier
        assert core_mod._default_context_for_ram(8.0, path) == 4096
        written = write_recommendation({"model_id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
                                        "context_size": 8192}, data_dir=data, ram_gb=8.0)
        assert written == path
        assert core_mod._default_model_for_ram(8.0, path) == "mlx-community/Llama-3.2-3B-Instruct-4bit"
        assert core_mod._default_context_for_ram(8.0, path) == 8192
        # A file stamped for another machine size is ignored.
        assert core_mod._default_model_for_ram(64.0, path) == LARGE
        assert core_mod._default_context_for_ram(64.0, path) == 32768
        assert read_recommendation(data, ram_gb=64.0) is None
        assert read_recommendation(data, ram_gb=8.0)["context_size"] == 8192
        # MODEL_ID set: the model tier ignores it, and a context sized for the
        # recommended model is not applied to a different one.
        os.environ["MODEL_ID"] = "someone/other-model"
        assert core_mod._default_model_for_ram(8.0, path) == SMALL
        assert core_mod._default_context_for_ram(8.0, path) == 4096
        os.environ["MODEL_ID"] = "mlx-community/Llama-3.2-3B-Instruct-4bit"
        assert core_mod._default_context_for_ram(8.0, path) == 8192
        os.environ.pop("MODEL_ID")
        os.environ["CONTEXT_SIZE"] = "2048"
        assert core_mod._default_context_for_ram(8.0, path) == 4096
        # Garbage on disk never breaks startup.
        path.write_text("{not json")
        assert core_mod._default_model_for_ram(8.0, path) == SMALL
    finally:
        restore()


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
        managers = _FakeManagers(server.server_address[1], fail={"mlx-community/Qwen2.5-1.5B-Instruct-4bit"})
        cfg = Config(model=SMALL)
        rss = {}

        def sampler(pid):
            model = managers.pids.get(pid, "")
            return rss.setdefault(pid, 900.0 if "0.5B" in model else 2100.0)

        doc = bench_matrix(cfg, ["tiny", SMALL, "fast", "large", "tiny"], [1024], ["mlx-lm"],
                           manager_factory=managers, port_finder=lambda preferred: 45678,
                           rss_sampler=sampler, allow_download=True, out=lambda *_a: None)
        assert managers.max_alive == 1 and managers.alive == 0
        starts = [e for e in managers.events if e[0] == "start"]
        stops = [e for e in managers.events if e[0] == "stop"]
        assert len(starts) == len(stops)
        assert all(e[4] == 45678 for e in starts)
        by_model = {r["model_id"]: r for r in doc["results"]}
        assert len(doc["results"]) == 4                      # "tiny" twice = one cell
        tiny = by_model[BUILTIN_MODEL_ALIASES["tiny"]]
        assert tiny["success"] and tiny["alias"] == "tiny" and tiny["peak_rss_mb"] == 900.0
        assert tiny["decode_tps"] > 0 and tiny["prompt_tps"] > 0 and tiny["ttft_ms"] >= 0
        failed = by_model[BUILTIN_MODEL_ALIASES["fast"]]
        assert failed["success"] is False and "exited" in failed["error"]
        large = by_model[LARGE]
        if fits_in_ram(LARGE) is False:
            assert large.get("skipped") and "RAM" in large["error"]
        saved = list((data / "bench").glob("*.json"))
        assert [p.name for p in saved] == [doc["file"]]
        on_disk = json.loads(saved[0].read_text())
        assert on_disk["results"] == json.loads(json.dumps(doc["results"]))
        assert on_disk["recommendation"]["model_id"] in (SMALL, BUILTIN_MODEL_ALIASES["tiny"])
    finally:
        server.shutdown()


def test_bench_matrix_skips_uncached_without_download():
    _fresh_data_dir()
    managers = _FakeManagers(1)
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
    app, _cfg, _db = _build_app()
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
        assert user.post("/api/models/select", json={"model": "tiny"}).status_code == 403


def test_models_routes_accept_aliases_and_report_metadata():
    from fastapi.testclient import TestClient
    _fresh_data_dir()
    app, cfg, _db = _build_app(model_aliases=json.dumps({"chat": "fast"}))
    with TestClient(app) as c:
        _login(c, "admin", "adminpw123")
        body = c.get("/api/models").json()
        assert {"current", "catalog", "adapters", "cache_dir"} <= set(body)
        assert body["aliases"]["chat"]["source"] == "user"
        assert body["recommended"] is None and body["ram_gb"] > 0
        assert all("fits_in_ram" in e and "meta" in e for e in body["catalog"])
        r = c.post("/api/models/select", json={"model": "chat", "restart": False})
        assert r.status_code == 200, r.text
        assert r.json()["resolved"] == BUILTIN_MODEL_ALIASES["fast"]
        assert cfg.model == BUILTIN_MODEL_ALIASES["fast"] and cfg.model_alias == "chat"
        assert app.state.model_manager.model_id == cfg.model
        r = c.post("/api/models/select", json={"model": SMALL, "restart": False})
        assert r.json()["resolved"] == SMALL and cfg.model_alias == ""
        # Invalid aliases over /api/config are refused and reported.
        r = c.post("/api/config", json={"model_aliases": "{oops"})
        assert r.status_code == 200 and "model_aliases" in r.json()["ignored"]
        assert json.loads(cfg.model_aliases) == {"chat": "fast"}
        # A sweep in progress owns the model server.
        app.state.bench_job.running = True
        try:
            assert c.post("/api/models/select", json={"model": "tiny"}).status_code == 409
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
        app, _cfg, _db = _build_app()
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
