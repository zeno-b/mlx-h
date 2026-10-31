"""The routing evaluation suite (evals.py "routing", eval_suites.py EVAL_ROUTING_*).

upgrade/ROUTING_AND_TOOLING.md P9: each message must reach the right intent,
lane, tool set and retrieval rule before the model writes a word. Scripted
mode calls intents.classify() directly and runs the three reviewed failures
through the real Agent with canned outputs; these tests check that the suite
is registered everywhere a suite is listed, that its cases cover every
intent and ordinary questions, that it scores full marks today, and that it
reproduces the reviewed failures when the lexicon is switched off. The
phrasings classify still gets wrong (EVAL_ROUTING_GAPS) run as expected
failures. The last section checks the session isolation tests/conftest.py
installs (temp LOG_DIR, the model-server guard). No test talks to a model.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_llm.config as config_mod  # noqa: E402
config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-routing-"))

import local_llm.core as core_mod  # noqa: E402
import local_llm.evals as evals  # noqa: E402
import local_llm.model_server as model_server_mod  # noqa: E402
import local_llm.obslog as obslog_mod  # noqa: E402
import local_llm.ui_script_evals as ui_evals_mod  # noqa: E402
import local_llm.ui_settings as ui_settings_mod  # noqa: E402
from local_llm.config import Config  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.eval_suites import (EVAL_ROUTING_ATTACHMENTS, EVAL_ROUTING_CASES,  # noqa: E402
                                   EVAL_ROUTING_GAPS, EVAL_ROUTING_PROJECT)
from local_llm.intents import (DOCUMENT_RESEARCH_TOOLS, INTENT_LANES,  # noqa: E402
                               INTENT_NAMES, INTENT_RETRIEVAL, INTENT_TOOL_SUBSETS)

REPO = Path(__file__).resolve().parent.parent
REVIEWED = ("reviewed_add_to_skills", "reviewed_kb_pdf", "reviewed_codebase_review")


def _cfg(**overrides) -> Config:
    base = dict(skills_dir=tempfile.mkdtemp(prefix="llm-test-routing-skills-"))
    base.update(overrides)
    return Config(**base)


def _run(mode="scripted", **kw):
    evals._EVAL_RUNTIME["dir"] = str(Path(tempfile.mkdtemp()) / "evals")
    try:
        return evals.run_evals("routing", mode, config=kw.pop("config", None) or _cfg(),
                               db=kw.pop("db", None), persist=kw.pop("persist", False), **kw)
    finally:
        evals._EVAL_RUNTIME["dir"] = None


def _rows(report) -> dict:
    return {r["case"]: r for r in report["results"] if r["suite"] == "routing"}


def _failures(report) -> list:
    return [(r["case"], {k: v for k, v in r["checks"].items() if not v}, r["error"])
            for r in report["results"] if not r["passed"]]


# --------------------------------------------------------------------------- #
# Registration                                                                 #
# --------------------------------------------------------------------------- #
def test_routing_is_listed_wherever_suites_are_listed():
    assert "routing" in evals.EVAL_SUITES and "routing" in evals.EVAL_MODEL_SUITES
    assert evals.eval_parse_suites("routing") == ["routing"]
    assert evals.eval_parse_suites("all") == list(evals.EVAL_SUITES)
    assert evals._EVAL_MODEL_RUNNERS["routing"] is evals._eval_routing_suite
    assert evals._EVAL_CASE_COUNTS["routing"] == len(EVAL_ROUTING_CASES)
    # The promotion gate keeps its three suites: an adapter cannot move the
    # lexicon's scores, and a live routing run is one agent turn per case.
    assert evals.EVAL_PROMOTE_SUITES == ("tool_calling", "rag", "multi_turn")
    # Admin > Evals: a checkbox to run it and a label for its scores.
    markup = Path(ui_settings_mod.__file__).read_text(encoding="utf-8")
    assert 'value="routing" checked' in markup
    assert 'routing: "Routing"' in ui_evals_mod.UI_JS_EVALS


def test_cases_cover_every_intent_and_ordinary_questions():
    ids = [c["id"] for c in EVAL_ROUTING_CASES + EVAL_ROUTING_GAPS]
    assert len(ids) == len(set(ids))
    expected = {c["intent"] for c in EVAL_ROUTING_CASES}
    assert set(INTENT_NAMES) <= expected, set(INTENT_NAMES) - expected
    others = [c for c in EVAL_ROUTING_CASES if c["id"] not in REVIEWED]
    assert len(others) >= 40
    assert sum(1 for c in EVAL_ROUTING_CASES if c["intent"] is None) >= 15
    for case in EVAL_ROUTING_CASES + EVAL_ROUTING_GAPS:
        assert case["lane"] in INTENT_LANES + ("pipeline",), case["id"]
        assert (case["intent"] is None) == (case["lane"] == "pipeline"), case["id"]
        assert case["retrieval"] in INTENT_RETRIEVAL, case["id"]
        assert set(case["offer"]).isdisjoint(case["not_offer"]), case["id"]
    for case in EVAL_ROUTING_GAPS:
        assert case["finding"], case["id"]


def test_every_tool_a_case_names_exists_somewhere():
    """A misspelt name in not_offer would pass forever, so names are checked."""
    sandbox = evals._EvalSandbox(_cfg(), "scripted", overrides=evals._EVAL_ROUTING_OVERRIDES)
    try:
        known = set(sandbox.registry.names()) | {"delegate_task"}
    finally:
        sandbox.close()
    for names in list(INTENT_TOOL_SUBSETS.values()) + list(DOCUMENT_RESEARCH_TOOLS.values()):
        known.update(names)
    for case in EVAL_ROUTING_CASES + EVAL_ROUTING_GAPS:
        unknown = set(case["offer"]) | set(case["not_offer"])
        assert unknown <= known, (case["id"], unknown - known)


def test_the_reviewed_messages_are_the_ones_that_failed():
    cases = {c["id"]: c for c in EVAL_ROUTING_CASES}
    assert cases["reviewed_add_to_skills"]["message"] == "add this to skills"
    assert cases["reviewed_add_to_skills"]["attachments"] == ["devils-advocate.md"]
    skill = EVAL_ROUTING_ATTACHMENTS["devils-advocate.md"]
    assert skill.startswith("---\nname: devils-advocate\ndescription: ")
    assert sum(1 for line in skill.splitlines() if line.startswith("- ")) == 2
    assert cases["reviewed_kb_pdf"]["message"] == (
        "use the knowledge base to craft a pdf showing an overview of usefull facts in the "
        "knowledge base")
    assert cases["reviewed_codebase_review"]["message"] == (
        "check the codebase for design mistakes and summarize")
    assert EVAL_ROUTING_PROJECT == "mlx-h"
    assert all(cases[c].get("agent") and cases[c].get("script") for c in REVIEWED)


# --------------------------------------------------------------------------- #
# Scripted mode                                                                #
# --------------------------------------------------------------------------- #
def test_scripted_routing_scores_intent_lane_tools_and_retrieval():
    db = Database(Path(tempfile.mkdtemp()) / "evals.db")
    report = _run(db=db, persist=True)
    assert report["status"] == "done", report["error"]
    score = report["scores"]["routing"]
    assert score["score"] == 1.0 and score["failed"] == 0, _failures(report)
    assert score["total"] == len(EVAL_ROUTING_CASES)
    assert score["accuracy"] == {"intent": 1.0, "lane": 1.0, "tools": 1.0, "retrieval": 1.0}
    rows = _rows(report)
    # Only the reviewed cases ran through the agent; the rest is classify alone.
    with_agent = {case for case, row in rows.items() if "agent.intent" in row["checks"]}
    assert with_agent == set(REVIEWED)
    plain = rows["plain_resume_skill"]
    assert plain["detail"]["classified"]["intent"] is None
    assert plain["detail"]["classified"]["lane"] == "pipeline"
    # Stored like any other suite, accuracy included.
    stored = evals.eval_get_run(db, report["id"])
    assert stored["scores"]["routing"]["accuracy"]["intent"] == 1.0
    assert len(stored["results"]) == len(EVAL_ROUTING_CASES)
    text = evals.eval_format_report(report)
    assert "routing" in text and "accuracy: intent 100.0%, lane 100.0%" in text
    db.close()


def test_the_reviewed_messages_take_the_right_lane_in_the_agent():
    rows = _rows(_run())
    skill = rows["reviewed_add_to_skills"]["detail"]["agent"]
    assert skill["intent"] == "skill_save" and skill["lane"] == "tools"
    assert "save_skill" in skill["offered"]
    assert not {"web_search", "search_docs", "create_document"} & set(skill["offered"])
    assert skill["calls"] == ["save_skill"] and skill["retrieved"] == []

    pdf = rows["reviewed_kb_pdf"]["detail"]["agent"]
    assert pdf["intent"] == "document_create" and pdf["lane"] == "document"
    assert {"create_document", "search_docs"} <= set(pdf["offered"])
    assert not {"web_search", "fetch_url"} & set(pdf["offered"])
    assert pdf["calls"][:2] == ["search_docs", "create_document"]

    code = rows["reviewed_codebase_review"]["detail"]["agent"]
    assert code["intent"] == "codebase_review" and code["lane"] == "tools"
    assert {"list_files", "read_file", "search_files"} <= set(code["offered"])
    assert not {"web_search", "search_docs", "create_document"} & set(code["offered"])
    assert code["retrieved"] == [] and code["calls"] == ["list_files"]


def test_with_the_lexicon_off_the_suite_reproduces_the_reviewed_failures():
    """INTENT_LEXICON_ENABLED=0 is the pipeline the three conversations went through."""
    report = _run(config=_cfg(intent_lexicon_enabled=False))
    rows = _rows(report)
    # The rules themselves are unchanged, so classify's own checks still pass.
    for case in REVIEWED:
        checks = rows[case]["checks"]
        assert checks["intent"] and checks["lane"] and checks["tools"], case
        assert not checks["agent.intent"] and not checks["agent.lane"], case
        assert not checks["agent.tools"], case
    skill = rows["reviewed_add_to_skills"]["detail"]["agent"]
    code = rows["reviewed_codebase_review"]["detail"]["agent"]
    pdf = rows["reviewed_kb_pdf"]["detail"]["agent"]
    # Turns 1 and 3: an unrelated knowledge-base passage was injected.
    assert skill["retrieved"] and not rows["reviewed_add_to_skills"]["checks"]["agent.retrieval"]
    assert code["retrieved"] and not rows["reviewed_codebase_review"]["checks"]["agent.retrieval"]
    # Turn 2: the document lane had create_document and nothing to research with.
    assert "search_docs" not in pdf["offered"] and "create_document" in pdf["offered"]
    assert report["scores"]["routing"]["score"] < 1.0
    accuracy = report["scores"]["routing"]["accuracy"]
    assert accuracy["intent"] < 1.0 and accuracy["lane"] < 1.0 and accuracy["tools"] < 1.0


def test_each_dimension_is_scored_on_its_own(monkeypatch):
    """A classifier that only gets retrieval wrong costs retrieval accuracy alone."""
    real = evals.classify

    def kb_as_auto(*args, **kwargs):
        found = real(*args, **kwargs)
        if found is not None and found.retrieval == "kb":
            return replace(found, retrieval="auto")
        return found

    monkeypatch.setattr(evals, "classify", kb_as_auto)
    report = _run()
    accuracy = report["scores"]["routing"]["accuracy"]
    assert accuracy["intent"] == accuracy["lane"] == accuracy["tools"] == 1.0
    assert accuracy["retrieval"] < 1.0
    bad = {case for case, row in _rows(report).items() if not row["passed"]}
    kb_cases = {c["id"] for c in EVAL_ROUTING_CASES if c["retrieval"] == "kb"}
    assert bad == kb_cases

    monkeypatch.setattr(evals, "classify", lambda *a, **k: None)
    report = _run()
    rows = _rows(report)
    assert rows["plain_hash_map"]["passed"] and not rows["skill_list_mine"]["passed"]
    # The agent runs its own rules, so the reviewed turns still route correctly.
    assert all(rows[c]["checks"]["agent.intent"] for c in REVIEWED)
    assert report["scores"]["routing"]["accuracy"]["intent"] < 1.0


def test_live_mode_runs_every_case_through_the_agent():
    """Live mode with a scripted stand-in for the model: one agent turn per case."""
    client = evals.EvalScriptedClient()
    report = _run("live", client=client, model_id="test-model", adapter_key="none")
    assert report["status"] == "done", report["error"]
    rows = _rows(report)
    assert len(rows) == len(EVAL_ROUTING_CASES)
    for case, row in rows.items():
        assert "agent.intent" in row["checks"] and "agent.lane" in row["checks"], case
    lanes = {case: row["detail"]["agent"]["lane"] for case, row in rows.items()}
    assert lanes["plain_capital"] == "pipeline" and lanes["conv_delete"] == "reply"
    assert lanes["skill_save_no_referent"] == "clarify"
    assert lanes["kb_query_search"] == "tools" and lanes["document_kb_deck"] == "document"
    assert all(row["checks"]["agent.intent"] and row["checks"]["agent.lane"]
               for row in rows.values()), _failures(report)


def test_the_sandbox_keeps_agent_checkpoints_out_of_the_live_data_dir():
    """Every agent turn prunes the registry's undo store to its newest entries."""
    sandbox = evals._EvalSandbox(_cfg(), "scripted", overrides=evals._EVAL_ROUTING_OVERRIDES)
    try:
        root = sandbox.registry.checkpoints.root
        assert sandbox.root in root.parents
        assert config_mod.DATA_DIR not in root.parents and REPO not in root.parents
    finally:
        sandbox.close()


def test_cli_runs_the_routing_suite():
    lines: list[str] = []
    evals._EVAL_RUNTIME["dir"] = str(Path(tempfile.mkdtemp()) / "evals")
    try:
        code = evals.eval_cli(["--suite", "routing", "--no-persist"], config=_cfg(),
                              out=lines.append)
    finally:
        evals._EVAL_RUNTIME["dir"] = None
    text = "\n".join(lines)
    assert code == 0, text
    assert "routing" in text and "accuracy: intent 100.0%" in text


# --------------------------------------------------------------------------- #
# Known gaps in classify(): findings, run as expected failures                 #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def routing_sandbox():
    sandbox = evals._EvalSandbox(_cfg(), "scripted", workspace_name=EVAL_ROUTING_PROJECT,
                                 overrides=evals._EVAL_ROUTING_OVERRIDES)
    yield sandbox, frozenset(sandbox.registry.names())
    sandbox.close()


@pytest.mark.parametrize("case", [
    pytest.param(case, id=case["id"], marks=pytest.mark.xfail(reason=case["finding"],
                                                               strict=False))
    for case in EVAL_ROUTING_GAPS])
def test_known_routing_gap(routing_sandbox, case):
    sandbox, registered = routing_sandbox
    ctx = {"mode": "scripted", "client": None, "run_id": "gaps", "timeout": 5.0}
    row = asyncio.run(evals._eval_routing_case(ctx, sandbox, None, case, registered))
    assert row["passed"], (row["checks"], row["detail"]["classified"])


# --------------------------------------------------------------------------- #
# tests/conftest.py: logs in a temp folder, no real model server              #
# --------------------------------------------------------------------------- #
def _conftest():
    module = sys.modules.get("conftest")
    if module is None or not hasattr(module, "ct_launches_model_server"):
        pytest.skip("tests/conftest.py is not loaded")
    return module


def test_logs_go_to_a_session_temp_folder():
    _conftest()
    log_dir = Path(os.environ["LOG_DIR"])
    repo_logs = REPO / "logs"
    assert log_dir != repo_logs and repo_logs not in log_dir.parents
    for module in (core_mod, obslog_mod, model_server_mod):
        assert Path(module.LOG_DIR) == log_dir, module.__name__
    for name, module in list(sys.modules.items()):
        if name.startswith("local_llm") and hasattr(module, "LOG_DIR"):
            assert Path(module.LOG_DIR) != repo_logs, name
    assert obslog_mod._log_dir() == log_dir
    assert Path(Config().log_dir) == log_dir


def test_star_imported_data_paths_leave_the_repository():
    """Checkpoints, partial replies and imports used to land in the repo's data/."""
    conftest = _conftest()
    repo_data = REPO / "data"
    for name, module in list(sys.modules.items()):
        if not name.startswith("local_llm"):
            continue
        for attr, value in list(vars(module).items()):
            if attr.isupper() and isinstance(value, Path):
                assert value != repo_data and repo_data not in value.parents, (name, attr)
    import local_llm.tools as tools_mod
    import local_llm.agent as agent_mod
    assert Path(tools_mod.DATA_DIR) == conftest._CT_DATA_DIR
    assert Path(agent_mod.DATA_DIR) == conftest._CT_DATA_DIR


def test_model_server_guard_refuses_a_launch_and_names_the_test(monkeypatch):
    conftest = _conftest()
    trips = conftest._CT_TRIPS
    before = len(trips)
    # Harmless if the guard ever failed: `python -c pass` with a server-like
    # argument, never a real `-m mlx_lm.server`.
    try:
        with pytest.raises(RuntimeError, match="tests must not launch a real model server"):
            subprocess.Popen([sys.executable, "-c", "pass", "mlx_lm.server"])
        with pytest.raises(RuntimeError, match="tests must not launch a real model server"):
            subprocess.run([sys.executable, "-c", "pass", "-m", "mlx_lm.server"], check=False)
        with pytest.raises(RuntimeError):
            model_server_mod.subprocess.Popen([sys.executable, "-c",
                                               "pass  # local_llm.mlx_engine"])
        assert len(trips) == before + 3
        assert trips[-1]["test"].endswith(
            "test_routing_suite.py::test_model_server_guard_refuses_a_launch_and_names_the_test")
    finally:
        del trips[before:]   # these refusals are this test's, not findings
    # A --help probe and an ordinary child process still run.
    probe = subprocess.run([sys.executable, "-c", "pass", "mlx_lm.server", "--help"],
                           capture_output=True)
    assert probe.returncode == 0
    assert subprocess.run([sys.executable, "-c", "print('ok')"], capture_output=True,
                          text=True).stdout.strip() == "ok"
    assert not conftest.ct_launches_model_server(
        [sys.executable, "-m", "mlx_lm.lora", "--train"])
    assert conftest.ct_launches_model_server(
        [sys.executable, "/tmp/deploy.py", "mlx-engine", "--port", "1"])

    # A test that mocks Popen gets its mock, as local_llm.model_server sees it too.
    class FakePopen:
        def __init__(self, args, *a, **k):
            self.args = args

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    fake = model_server_mod.subprocess.Popen([sys.executable, "-m", "mlx_lm.server"])
    assert isinstance(fake, FakePopen) and len(trips) == before
