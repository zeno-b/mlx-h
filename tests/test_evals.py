"""Tests for the evaluation and regression suite (evals.py, eval_suites.py).

Scripted mode runs every suite end to end through the real Agent, the real
ToolRegistry (side effects stubbed), the real toolparse and an in-process app.
Degraded scripts prove each scorer catches what it is for; baselines and the
regression rule are checked on stored runs; the routes are checked for RBAC
and the background job; the EVAL_ON_PROMOTE hook is driven through
RetrainManager.run with the fake-trainer pattern from tests/test_app.py
(copied here, not imported). No test talks to a model server.
"""
from __future__ import annotations

import contextlib
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_llm.config as config_mod  # noqa: E402
config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-evals-"))

from fastapi.testclient import TestClient  # noqa: E402

import local_llm.evals as evals  # noqa: E402
import local_llm.sysutil as sysutil_mod  # noqa: E402
import local_llm.toolparse as toolparse_mod  # noqa: E402
import local_llm.training as training_mod  # noqa: E402
from local_llm.api import create_app  # noqa: E402
from local_llm.auth import hash_password  # noqa: E402
from local_llm.config import Config  # noqa: E402
from local_llm.core import ADAPTER_DIR  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.eval_suites import (EVAL_MULTI_TURN_CASES, EVAL_RAG_CASES,  # noqa: E402
                                   EVAL_TOOL_CASES)
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.training import RetrainManager  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MODEL = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"


def _cfg(**overrides) -> Config:
    base = dict(model=MODEL, skills_dir=tempfile.mkdtemp(prefix="llm-test-evals-skills-"))
    base.update(overrides)
    return Config(**base)


def _db() -> Database:
    return Database(Path(tempfile.mkdtemp()) / "evals.db")


def _run(suites, db=None, **kw):
    return evals.run_evals(suites, "scripted", config=kw.pop("config", None) or _cfg(),
                           db=db, persist=db is not None, **kw)


def _row(report, suite, case):
    return next(r for r in report["results"] if r["suite"] == suite and r["case"] == case)


# --------------------------------------------------------------------------- #
# Scripted mode, every suite                                                   #
# --------------------------------------------------------------------------- #
def test_scripted_mode_runs_every_suite_end_to_end_and_persists():
    db = _db()
    report = _run(None, db)
    assert report["status"] == "done", report["error"]
    assert report["suites"] == list(evals.EVAL_SUITES)
    for suite in evals.EVAL_SUITES:
        score = report["scores"][suite]
        assert score["score"] == 1.0 and score["failed"] == 0, (suite, score, [
            (r["case"], r["checks"], r["error"]) for r in report["results"]
            if r["suite"] == suite and not r["passed"]])
        assert score["total"] == evals._EVAL_CASE_COUNTS[suite]
    assert report["score"] == 1.0 and not report["regressed"]
    # The tool cases went through the real agent loop: tool_call events with
    # the expected names, and tokens accounted per case.
    weather = _row(report, "tool_calling", "weather_city")
    assert weather["detail"]["calls"][0] == {"name": "weather", "args": {"location": "Ghent"}}
    assert weather["prompt_tokens"] > 0 and weather["latency_ms"] >= 0
    # Retrieval put the source into the prompt; the continuation really resumed.
    assert _row(report, "rag", "vpn_port")["checks"]["retrieved"]
    assert _row(report, "multi_turn", "continue_after_truncation")["checks"]["t2.continued"]
    # Stored: the run row, its results, and the JSON report.
    runs = evals.eval_list_runs(db)
    assert runs[0]["id"] == report["id"] and runs[0]["status"] == "done"
    stored = evals.eval_get_run(db, report["id"])
    assert len(stored["results"]) == len(report["results"])
    assert (evals.eval_dir() / "runs" / f"{report['id']}.json").exists()
    db.close()


def test_tool_calling_scores_choice_arguments_invented_names_and_completion():
    report = _run("tool_calling", overrides={
        # The wrong tool, then an answer.
        "weather_city": ['{"tool": "web_search", "args": {"query": "Ghent weather"}}',
                         "It is 14 C."],
        # The right tool with an argument that does not match.
        "read_project_file": ['{"tool": "read_file", "args": {"path": "README.md"}}',
                              "The port is 9137."],
        # A tool that does not exist.
        "no_invented_tool": ['{"tool": "send_sms", "args": {"to": "colleague", "text": "late"}}',
                             "Sent."],
        # A tool where none was needed.
        "no_tool_needed": ['{"tool": "web_search", "args": {"query": "hash map"}}',
                           "A hash map maps a key to a value."],
        # No answer facts.
        "calculator_expression": ['{"tool": "calculator", "args": {"expression": "(17 * 23) + sqrt(144)"}}',
                                  "Done."],
    })
    assert report["status"] == "done"
    weather = _row(report, "tool_calling", "weather_city")
    assert not weather["passed"] and not weather["checks"]["tool_choice"]
    assert not weather["checks"]["args_valid"]
    read = _row(report, "tool_calling", "read_project_file")
    assert read["checks"]["tool_choice"] and not read["checks"]["args_valid"]
    assert any("path" in p for p in read["detail"]["problems"])
    sms = _row(report, "tool_calling", "no_invented_tool")
    assert not sms["checks"]["no_hallucination"] and "send_sms" in sms["detail"]["hallucinated"]
    assert not _row(report, "tool_calling", "no_tool_needed")["checks"]["tool_choice"]
    calc = _row(report, "tool_calling", "calculator_expression")
    # The calculator ran for real in the sandbox (read-only), the answer lacks 403.
    # (A terminal tool may answer with its own result, so only check what we scripted.)
    assert calc["checks"]["tool_choice"] and calc["checks"]["args_valid"]
    untouched = _row(report, "tool_calling", "time_in_zone")
    assert untouched["passed"]
    assert report["scores"]["tool_calling"]["score"] < 1.0


def test_side_effecting_tools_are_stubbed_and_reads_stay_in_the_sandbox():
    sandbox = evals._EvalSandbox(_cfg(), "scripted", files={"a.txt": "hello sandbox"})
    try:
        sandbox.begin_case({"id": "x"})
        out, err = sandbox.call("write_file", {"path": "b.txt", "content": "x"})
        assert err is None and "recorded, not run" in out
        assert not (sandbox.workspace / "b.txt").exists()
        out, err = sandbox.call("read_file", {"path": "a.txt"})
        assert err is None and "hello sandbox" in out
        out, err = sandbox.call("nope_tool", {})
        assert err and "Unknown tool" in out
        assert [c["name"] for c in sandbox.calls] == ["write_file", "read_file", "nope_tool"]
        assert sandbox.calls[0]["stubbed"] and not sandbox.calls[1]["stubbed"]
        assert "delegate_task" not in sandbox.registry.names()
        assert not sandbox.config.mcp_servers and not sandbox.config.cloud_fallback_enabled
    finally:
        sandbox.close()
    assert not sandbox.root.exists()


def test_rag_scores_fact_citation_and_retrieval():
    report = _run("rag", overrides={
        "vpn_port": ["The VPN gateway listens on UDP port 51820."],          # no citation
        "leave_days": ["Employees get 20 days (source: kb/handbook/leave-policy.md)."],
    })
    vpn = _row(report, "rag", "vpn_port")
    assert vpn["checks"]["fact"] and not vpn["checks"]["cited"] and not vpn["passed"]
    assert vpn["checks"]["retrieved"]
    leave = _row(report, "rag", "leave_days")
    assert not leave["checks"]["fact"] and leave["checks"]["cited"] and not leave["passed"]
    assert _row(report, "rag", "meal_limit")["passed"]
    assert report["scores"]["rag"]["passed"] == len(EVAL_RAG_CASES) - 2


def test_multi_turn_catches_drift_broken_code_and_a_second_cut_off():
    drift = list(next(c for c in EVAL_MULTI_TURN_CASES if c["id"] == "language_drift")["script"])
    drift[1] = ("```python\nimport unicodedata\n\n\ndef reverse(s):\n"
                "    return s[::-1]\n```")
    cut = list(next(c for c in EVAL_MULTI_TURN_CASES
                    if c["id"] == "continue_after_truncation")["script"])
    cut[1] = {"text": "  return function (...args) {\n    const now = Date.now();\n",
              "finish": "length"}
    report = _run("multi_turn", overrides={"language_drift": drift,
                                           "continue_after_truncation": cut})
    row = _row(report, "multi_turn", "language_drift")
    assert row["checks"]["t1.language"] and row["checks"]["t1.defines"]
    # The agent notices the drift and regenerates; the scripted model has
    # nothing better, so the turn ends without Go code and fails.
    assert not row["checks"]["t2.language"] and not row["checks"]["t2.defines"]
    assert not row["passed"] and 0 < row["score"] < 1
    cont = _row(report, "multi_turn", "continue_after_truncation")
    assert cont["checks"]["t1.truncated"] and cont["checks"]["t2.continued"]
    assert not cont["checks"]["t2.clean"] and not cont["passed"]
    assert _row(report, "multi_turn", "artifact_then_edits")["passed"]


def test_parser_suite_needs_no_model_and_catches_a_parser_regression(monkeypatch):
    report = _run("parsers")
    assert report["scores"]["parsers"]["score"] == 1.0
    names = {r["case"] for r in report["results"]}
    assert {"hermes", "qwen3_coder", "gpt_oss_harmony", "repair_smart_quotes",
            "prose_nameless"} <= names
    assert _row(report, "parsers", "canonical")["checks"]["streamed"]
    # The suite pins formats="auto" and repair on, whatever the live config says.
    previous = toolparse_mod._TP_POLICY["config"]
    toolparse_mod.bind_tool_call_config(Config(tool_call_format="json", tool_call_repair=False))
    try:
        assert _run("parsers")["scores"]["parsers"]["score"] == 1.0
    finally:
        toolparse_mod.bind_tool_call_config(previous)
    real = evals.parse_tool_calls

    def broken(text, known=None, **kw):
        calls = real(text, known, **kw)
        for call in calls:
            call.format = "json"
        return calls
    monkeypatch.setattr(evals, "parse_tool_calls", broken)
    bad = _run("parsers")
    assert bad["scores"]["parsers"]["score"] < 1.0
    assert not _row(bad, "parsers", "hermes")["checks"]["format"]


def test_api_suite_checks_v1_shapes_in_process_and_isolates_in_the_server():
    report = _run("api")
    assert report["scores"]["api"]["score"] == 1.0, [
        (r["case"], r["checks"], r["detail"]) for r in report["results"] if not r["passed"]]
    assert [r["case"] for r in report["results"]] == list(evals.EVAL_API_CASES)
    # Inside the server the api suite never builds an app in-process.
    seen = []

    def child(config, tick):
        seen.append(config)
        return [evals._eval_case_row("api", "models_list", {"status": True}, latency_ms=1.0)]
    evals._EVAL_RUNTIME["api_runner"] = child
    try:
        isolated = evals.run_evals("api", "scripted", config=_cfg(), persist=False,
                                   isolate_app=True)
    finally:
        evals._EVAL_RUNTIME["api_runner"] = None
    assert seen and isolated["scores"]["api"]["total"] == 1
    argv = evals._eval_entry_argv()
    assert argv[0] == sys.executable and "eval_cli" in argv[-1]


# --------------------------------------------------------------------------- #
# Baselines and regressions                                                    #
# --------------------------------------------------------------------------- #
def test_baseline_save_compare_and_regression_detection():
    db = _db()
    first = _run("tool_calling,parsers", db, save_baseline=True)
    assert first["saved_baseline"]["run_id"] == first["id"]
    base = evals.eval_get_baseline(db, evals.EVAL_SCRIPTED_MODEL, evals.EVAL_SCRIPTED_ADAPTER,
                                   "scripted")
    assert base["run_id"] == first["id"] and base["scores"]["tool_calling"]["score"] == 1.0
    again = _run("tool_calling,parsers", db)
    assert again["baseline"]["run_id"] == first["id"]
    assert again["comparison"]["tool_calling"]["delta"] == 0.0 and not again["regressed"]
    worse = {c["id"]: ["I would rather not use tools today."] for c in EVAL_TOOL_CASES[:3]}
    regressed = _run("tool_calling,parsers", db, overrides=worse)
    assert regressed["regressed"] and regressed["regressions"] == ["tool_calling"]
    row = regressed["comparison"]["tool_calling"]
    assert row["regressed"] and row["delta"] < -0.05
    assert not regressed["comparison"]["parsers"]["regressed"]
    # The allowed drop is EVAL_MAX_REGRESSION, from the config or the caller.
    tolerant = _run("tool_calling", db, overrides=worse, config=_cfg(eval_max_regression=0.5))
    assert not tolerant["regressed"] and tolerant["max_regression"] == 0.5
    stored = evals.eval_list_runs(db)[0]
    assert stored["id"] == tolerant["id"] and stored["baseline_id"] == first["id"]
    # A suite that could not run at all regresses against a baseline that had it.
    comparison, regressions = evals.eval_compare(
        {"rag": {"score": None}}, {"rag": {"score": 0.8}}, 0.05)
    assert regressions == ["rag"] and comparison["rag"]["current"] is None
    # Baselines are also JSON, so a run without a database still finds one.
    assert evals.eval_get_baseline(None, evals.EVAL_SCRIPTED_MODEL,
                                   evals.EVAL_SCRIPTED_ADAPTER, "scripted")["run_id"] == first["id"]
    with contextlib.suppress(ValueError):
        evals.eval_save_baseline(db, {**first, "status": "error"})
        raise AssertionError("an unfinished run must not become a baseline")
    db.close()


def test_live_mode_uses_its_own_client_and_never_rebinds_the_parser():
    live_cfg = _cfg()
    previous = toolparse_mod._TP_POLICY["config"]
    toolparse_mod.bind_tool_call_config(live_cfg)
    try:
        client = evals.EvalLiveClient(evals.eval_config(live_cfg))
        assert toolparse_mod._TP_POLICY["config"] is live_cfg
        assert client.cluster is None and not client.config.cloud_fallback_enabled
    finally:
        toolparse_mod.bind_tool_call_config(previous)
    # Live mode keys runs by the configured model and adapter, not "scripted".
    report = evals.run_evals("tool_calling", "live", config=_cfg(adapter="none"),
                             client=evals.EvalScriptedClient(), persist=False)
    assert report["mode"] == "live" and report["model"] == MODEL
    assert report["adapter_key"] == evals.base_adapter_key(MODEL)
    assert report["scores"]["tool_calling"]["score"] == 1.0


def test_cli_eval_json_baseline_and_compare_exit_codes():
    from local_llm.cli import subcommands
    assert subcommands()["eval"] is evals.eval_cli
    db = _db()
    lines: list[str] = []
    assert evals.eval_cli(["--suite", "parsers", "--json"], db=db, out=lines.append) == 0
    data = json.loads(lines[-1])
    assert data["mode"] == "scripted" and data["scores"]["parsers"]["score"] == 1.0
    assert evals.eval_cli(["--suite", "nope"], db=db, out=lines.append) == 2
    # The newest baseline is per model + adapter + mode; none saved yet here.
    evals._EVAL_RUNTIME["dir"] = str(Path(tempfile.mkdtemp()) / "evals")
    try:
        assert evals.eval_cli(["--suite", "parsers", "--compare"], db=_db(),
                              out=lines.append) == 2
        fresh = _db()
        assert evals.eval_cli(["--suite", "parsers", "--save-baseline"], db=fresh,
                              out=lines.append) == 0
        assert "Saved as the baseline." in lines[-1]
        assert evals.eval_cli(["--suite", "parsers", "--compare"], db=fresh,
                              out=lines.append) == 0
        assert "parsers" in lines[-1] and "Baseline: run" in lines[-1]
    finally:
        evals._EVAL_RUNTIME["dir"] = None
    db.close()


def test_config_fields_are_clamped_mutable_and_in_settings():
    assert Config().eval_on_promote is False and Config().eval_max_regression == 0.05
    assert Config(eval_max_regression=5).eval_max_regression == 1.0
    assert Config(eval_max_regression=-1).eval_max_regression == 0.0
    assert {"eval_on_promote", "eval_max_regression"} <= set(Config.MUTABLE)
    from local_llm.settings_schema import SETTINGS_FIELDS, SETTINGS_SUBTABS
    fields = {f["key"]: f for f in SETTINGS_FIELDS}
    assert fields["eval_on_promote"]["sub"] == "evals"
    assert fields["eval_max_regression"]["max"] == 1.0
    keys = [t["key"] for t in SETTINGS_SUBTABS["admin"]]
    assert "evals" in keys and keys.index("evals") == keys.index("dashboards") + 1


# --------------------------------------------------------------------------- #
# Routes                                                                       #
# --------------------------------------------------------------------------- #
def _build_app(**overrides):
    base = dict(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                allow_test_user=False, node_token="secret-node-token", model=MODEL,
                skills_dir=tempfile.mkdtemp(prefix="local-llm-test-skills-"),
                agent_enabled=True, knowledge_triage=False)
    base.update(overrides)
    cfg = Config(**base)
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)  # never started
    app = create_app(cfg, db, mm, RetrainManager(db, mm, cfg), ToolRegistry(cfg, db))
    app.state.model_manager = mm
    return app, cfg, db, mm


def _bearer(c: TestClient, username: str = "admin", password: str = "adminpw123") -> dict:
    c.cookies.clear()
    r = c.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    token = c.cookies.get("llm_session")
    c.cookies.clear()
    return {"Authorization": f"Bearer {token}"}


def _wait_idle(timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = evals._EVAL_JOB.snapshot()
        if not snap["running"]:
            return snap
        time.sleep(0.05)
    raise AssertionError("the evaluation never finished")


def test_routes_are_admin_only_and_run_in_the_background(monkeypatch):
    app, cfg, db, mm = _build_app()
    db.create_user("bob", password_hash=hash_password("bobpw12345"), role="user")
    c = TestClient(app)
    admin, bob = _bearer(c), _bearer(c, "bob", "bobpw12345")
    for method, path in (("get", "/api/evals/runs"), ("get", "/api/evals/runs/ev-x"),
                         ("get", "/api/evals/baseline"), ("post", "/api/evals/run"),
                         ("post", "/api/evals/baseline"), ("post", "/api/evals/cancel")):
        kw = {"json": {}} if method == "post" else {}
        assert getattr(c, method)(path, headers=bob, **kw).status_code == 403, path
        assert getattr(c, method)(path, **kw).status_code == 401, path
    assert c.post("/api/evals/run", json={"suites": "nope"}, headers=admin).status_code == 400
    assert c.post("/api/evals/run", json={"mode": "x"}, headers=admin).status_code == 400
    assert c.post("/api/evals/cancel", json={}, headers=admin).status_code == 409
    # Live: refused while the model is not ready, and while anything generates.
    mm.status = "stopped"
    r = c.post("/api/evals/run", json={"mode": "live", "suites": ["rag"]}, headers=admin)
    assert r.status_code == 503 and "not ready" in r.json()["error"]
    mm.status = "ready"
    monkeypatch.setattr(evals, "_eval_busy", lambda ctx, **kw: "1 generation(s) running")
    r = c.post("/api/evals/run", json={"mode": "live", "suites": ["rag"]}, headers=admin)
    assert r.status_code == 409 and "generation" in r.json()["error"]
    monkeypatch.undo()
    # One run at a time.
    assert evals._EVAL_JOB.claim(run_id="ev-held", mode="scripted", suites=[], trigger="test")
    try:
        assert c.post("/api/evals/run", json={}, headers=admin).status_code == 409
    finally:
        evals._EVAL_JOB.release(status="done")
    # A scripted run of the parser suite, in the background.
    r = c.post("/api/evals/run", json={"suites": ["parsers"], "mode": "scripted",
                                       "save_baseline": True}, headers=admin)
    assert r.status_code == 202, r.text
    run_id = r.json()["run_id"]
    snap = _wait_idle()
    assert snap["status"] == "done" and snap["run_id"] == run_id and snap["score"] == 1.0
    body = c.get("/api/evals/runs", headers=admin).json()
    assert body["runs"][0]["id"] == run_id and body["job"]["running"] is False
    assert body["active"]["model"] == MODEL and "parsers" in body["suites"]
    assert body["max_regression"] == 0.05 and body["on_promote"] is False
    detail = c.get(f"/api/evals/runs/{run_id}", headers=admin).json()
    assert detail["scores"]["parsers"]["score"] == 1.0 and detail["results"]
    assert c.get("/api/evals/runs/ev-missing", headers=admin).status_code == 404
    base = c.get("/api/evals/baseline?mode=scripted", headers=admin).json()
    assert base["baseline"]["run_id"] == run_id and base["mode"] == "scripted"
    assert c.post("/api/evals/baseline", json={"run_id": "ev-missing"},
                  headers=admin).status_code == 404
    assert c.post("/api/evals/baseline", json={"run_id": run_id},
                  headers=admin).json()["baseline"]["run_id"] == run_id
    # The api suite runs in a child process inside the server.
    called = []

    def child(config, tick):
        called.append(True)
        return [evals._eval_case_row("api", "models_list", {"status": True}, latency_ms=1.0)]
    evals._EVAL_RUNTIME["api_runner"] = child
    try:
        r = c.post("/api/evals/run", json={"suites": "api"}, headers=admin)
        assert r.status_code == 202
        assert _wait_idle()["status"] == "done" and called
    finally:
        evals._EVAL_RUNTIME["api_runner"] = None
    db.close()


# --------------------------------------------------------------------------- #
# EVAL_ON_PROMOTE (fake trainer pattern from tests/test_app.py, copied)        #
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def training_sandbox(**overrides):
    """A RetrainManager whose data/adapter/log directories are all throwaway."""
    root = Path(tempfile.mkdtemp())
    sft, adapters, backups, logs = (root / "sft", root / "latest",
                                    root / "backups", root / "logs")
    for d in (sft, adapters, backups, logs):
        d.mkdir(parents=True, exist_ok=True)
    patches = {"SFT_DIR": sft, "ADAPTER_DIR": adapters,
               "ADAPTER_BACKUP_DIR": backups, "LOG_DIR": logs,
               "REPLAY_FILE": sft / "replay.jsonl"}
    saved = []
    for mod in (training_mod, sysutil_mod):
        for name, value in patches.items():
            if hasattr(mod, name):
                saved.append((mod, name, getattr(mod, name)))
                setattr(mod, name, value)
    cfg = Config(**overrides)
    db = Database(root / "test.db")
    mm = ModelServerManager(cfg.model, 8090, adapters)   # never started
    rm = RetrainManager(db, mm, cfg)
    try:
        yield rm, cfg, db, patches
    finally:
        for mod, name, value in saved:
            setattr(mod, name, value)
        db.close()
        shutil.rmtree(root, ignore_errors=True)


def approve(db, n, prefix="q", answer="a short but complete answer."):
    for i in range(n):
        db.execute(
            """INSERT INTO feedback (user_prompt, assistant_response, rating,
                                     approved_for_training) VALUES (?, ?, 1, 1)""",
            (f"{prefix}{i}?", f"{answer} ({i})"))
    db.commit()


def _fake_trainer(rm, patches, losses, *, exit_code=0, checkpoints=()):
    """Stand in for mlx_lm.lora: emit val losses and write adapter files."""
    adapters = patches["ADAPTER_DIR"]

    def fake_run(cmd, stdout=None, stderr=None, timeout=None):
        for it, loss in losses:
            stdout.write(f"Iter {it}: Val loss {loss:.4f}, Val took 0.1s\n")
        stdout.flush()
        (adapters / "adapters.safetensors").write_text("final, a longer adapter")
        for it in checkpoints:
            (adapters / f"{it:07d}_adapters.safetensors").write_text(f"ckpt{it}")

        class Result:
            returncode = exit_code
        return Result()

    training_mod.subprocess.run = fake_run
    rm._lora_help = "--model --train --data --adapter-path --iters --steps-per-eval --save-every"
    rm.model_manager.stop = lambda *a, **k: None
    rm.model_manager.start = lambda *a, **k: None
    restarts = []
    rm.model_manager.restart = lambda *a, **k: restarts.append(True)
    return restarts


def _promote(monkeypatch, overrides=None, *, on=True, baseline=1.0):
    """Retrain with EVAL_ON_PROMOTE; the "live" model is a scripted client."""
    real_run = training_mod.subprocess.run
    monkeypatch.setitem(evals._EVAL_RUNTIME, "ctx", None)
    monkeypatch.setitem(evals._EVAL_RUNTIME, "dir", str(Path(tempfile.mkdtemp()) / "evals"))
    monkeypatch.setitem(evals._EVAL_RUNTIME, "client_factory",
                        lambda config: evals.EvalScriptedClient(overrides))
    out = {}
    with training_sandbox(train_replay_ratio=0, train_tool_ratio=0, model=MODEL,
                          eval_on_promote=on,
                          skills_dir=tempfile.mkdtemp()) as (rm, cfg, db, patches):
        approve(db, 20)
        adapters = patches["ADAPTER_DIR"]
        (adapters / "adapters.safetensors").write_text("original")
        prev_key = evals.adapter_fingerprint(adapters)
        if baseline is not None:
            evals.eval_save_baseline(db, {
                "id": "ev-previous", "status": "done", "mode": "live", "model": MODEL,
                "adapter_key": prev_key, "score": baseline,
                "scores": {s: {"score": baseline} for s in evals.EVAL_MODEL_SUITES}})
        restarts = _fake_trainer(rm, patches, [(1, 2.5), (30, 1.8), (60, 1.7)])
        rm.model_manager.status = "ready"
        try:
            rm.run("test")
        finally:
            training_mod.subprocess.run = real_run
        out.update(status=dict(rm.status), weights=(adapters / "adapters.safetensors").read_text(),
                   meta=sysutil_mod.adapter_meta(adapters), untrained=db.get_untrained_count(),
                   restarts=len(restarts), prev_key=prev_key,
                   runs=evals.eval_list_runs(db), dir=evals.eval_dir(),
                   new_baseline=evals.eval_get_baseline(
                       db, MODEL, evals.adapter_fingerprint(adapters), "live"))
    return out


def test_promote_gate_rolls_back_a_regressing_adapter(monkeypatch):
    worse = {c["id"]: ["I would rather not."] for c in EVAL_TOOL_CASES}
    out = _promote(monkeypatch, worse)
    assert out["weights"] == "original", out["status"]
    rollback = out["meta"]["eval_rollback"]
    assert rollback["verdict"] == "rolled_back" and "tool_calling" in rollback["regressions"]
    assert rollback["baseline"]["run_id"] == "ev-previous"
    assert "tool_calling" in rollback["reason"]
    assert "regression" in out["status"]["message"] and out["status"]["running"] is False
    # Rows stay untrained for the next, larger run; the server restarted twice
    # (promotion, then rollback).
    assert out["untrained"] == 20 and out["restarts"] == 2
    assert out["runs"][0]["trigger"] == "promote" and out["runs"][0]["regressed"]
    assert list((out["dir"] / "rejected").glob("ev-*.json"))


def test_promote_gate_keeps_a_passing_adapter_and_records_it(monkeypatch):
    out = _promote(monkeypatch)
    assert out["weights"].startswith("final"), out["status"]
    assert out["meta"]["eval"]["verdict"] == "passed"
    assert out["meta"]["eval"]["scores"]["tool_calling"] == 1.0
    assert "Evaluation passed" in out["status"]["message"]
    assert "Retraining complete" in out["status"]["message"]
    assert out["untrained"] == 0 and out["restarts"] == 1
    # The promoted adapter's run is the next promotion's baseline.
    assert out["new_baseline"]["run_id"] == out["runs"][0]["id"]


def test_promote_gate_without_a_baseline_records_one(monkeypatch):
    out = _promote(monkeypatch, baseline=None)
    assert out["weights"].startswith("final")
    assert out["meta"]["eval"]["verdict"] == "unverified"
    assert out["new_baseline"] is not None


def test_promote_gate_is_untouched_when_off_and_never_fails_a_promotion(monkeypatch):
    def explode(*a, **k):
        raise AssertionError("the gate must not run with EVAL_ON_PROMOTE=0")
    monkeypatch.setattr(evals, "eval_after_promote", explode)
    out = _promote(monkeypatch, on=False)
    assert out["weights"].startswith("final") and "eval" not in out["meta"]
    assert out["status"]["message"].startswith("Retraining complete")
    assert not out["runs"] and out["restarts"] == 1
    monkeypatch.undo()
    # On, a gate that crashes keeps the adapter and says so.
    monkeypatch.setattr(evals, "_eval_promote_gate",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out = _promote(monkeypatch)
    assert out["weights"].startswith("final")
    assert "Evaluation failed (boom)" in out["status"]["message"]
    assert not evals._EVAL_JOB.snapshot()["running"]


# --------------------------------------------------------------------------- #
# UI                                                                           #
# --------------------------------------------------------------------------- #
def _script(page: str) -> str:
    match = re.search(r"<script>(.*?)</script>", page, re.S)
    assert match
    return match.group(1)


def test_evals_tab_panel_script_and_wiring():
    from local_llm.ui import HTML_PAGE
    from local_llm.ui_script_evals import UI_JS_EVALS
    from local_llm.ui_settings import UI_SETTINGS_HTML
    # The panels render once, in the admin-only Admin > Evals tab.
    for panel in ("evals", "evalscores", "evalruns", "evalcases"):
        assert UI_SETTINGS_HTML.count(f'data-panel="{panel}"') == 1, panel
        at = UI_SETTINGS_HTML.index(f'data-panel="{panel}"')
        sec_at = UI_SETTINGS_HTML.rindex("<section ", 0, at)
        sec_tag = UI_SETTINGS_HTML[sec_at:UI_SETTINGS_HTML.index(">", sec_at)]
        assert 'data-section="admin"' in sec_tag and "admin-only" in sec_tag, panel
        sub_at = UI_SETTINGS_HTML.rindex('class="sub-panel', sec_at, at)
        assert 'data-sub="evals"' in UI_SETTINGS_HTML[sub_at:UI_SETTINGS_HTML.index(">", sub_at)]
    # The regression-gate settings render in the same tab, below the cards.
    gate = UI_SETTINGS_HTML.index('id="cfgEvalOnPromote"')
    assert UI_SETTINGS_HTML.index('data-panel="evalcases"') < gate
    start = UI_SETTINGS_HTML.index('data-panel="evals"')
    block = UI_SETTINGS_HTML[UI_SETTINGS_HTML.rindex("<div", 0, start):
                             UI_SETTINGS_HTML.index('data-panel="evalcases"')]
    assert 'style="' not in block
    # Every id the script looks up exists exactly once; every onclick exists.
    code = re.sub(r"//[^\n]*", "", UI_JS_EVALS)
    for el_id in set(re.findall(r'getElementById\("([^"]+)"\)', code)):
        assert UI_SETTINGS_HTML.count(f'id="{el_id}"') == 1, el_id
    for fn in set(re.findall(r'onclick="(\w+)\(', block + UI_SETTINGS_HTML[start:])):
        if fn.startswith("evals"):
            assert re.search(rf"function {fn}\(", UI_JS_EVALS), fn
    for route in ("/api/evals/runs", "/api/evals/run", "/api/evals/cancel", "/api/evals/baseline"):
        assert route in UI_JS_EVALS, route
    # ES5, toasts, own prefix, no markup injection.
    assert "=>" not in code and "`" not in code
    assert not re.search(r"^\s*(let|const|class)\s", code, re.M)
    assert "alert(" not in code and not re.search(r"(?<![\w.])confirm\(", code)
    assert not re.search(r"\.innerHTML\s*=", code) and "—" not in UI_JS_EVALS
    mine = re.findall(r"^\s*(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(", UI_JS_EVALS, re.M)
    assert mine and all(n.startswith("evals") for n in mine), mine
    script = _script(HTML_PAGE)
    assert script.index("var ADMIN_TABS = [") < script.index("function evalsTabHook")
    assert re.search(r"^\s*evalsTabHook\(\);", script, re.M)
    assert "uiChart(host" in UI_JS_EVALS
    names = re.findall(r"^\s*(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(", script, re.M)
    assert not sorted({n for n in names if names.count(n) > 1})
    # Bundled and self-tested like every other module.
    bundle = (REPO / "bundle.py").read_text()
    order = bundle[bundle.index("ORDER = ["):]
    assert '"ui_script_evals"' in order[:order.index('"ui", ')]
    assert order.index('"evals"') < order.index('"api",')
    selftest = (REPO / "local_llm" / "selftest.py").read_text()
    for mod in ("eval_suites", "evals", "ui_script_evals"):
        assert f'"{mod}"' in selftest, mod
