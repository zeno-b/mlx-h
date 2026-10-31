"""File safety: the overwrite guard, partial-read markers, answer hygiene, Undo.

The turn these pin down: a 3B model read about 1,500 characters of a 103 KB
README.md with nothing saying it was a part, replaced the whole file with 841
characters through write_file (approvals off, no guard), and answered that the
README "is not available", echoing "TOOL RESULT" labels and never mentioning
the overwrite. Each section below is one link of that chain:

1. write_file on an existing file: an approval card in an interactive turn,
   a refusal (naming edit_file and outputs/<name>) where nobody is asked.
2. read_file says, at the top and the bottom, which part of a file it shows,
   pages with offset/limit, and the context cut keeps that honest.
3. The final answer loses scratch labels and gains a code-built Changes
   footer that tells the truth even when the reply does not.
4. Undo: the chat's changed-files node rolls the turn's checkpoint back, which
   its owner may do and another user may not.
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import local_llm.config as config_mod  # noqa: E402

config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-file-safety-"))

from fastapi.testclient import TestClient  # noqa: E402

import local_llm.agent as agent_mod  # noqa: E402
from local_llm import policy  # noqa: E402
from local_llm.agent import Agent  # noqa: E402
from local_llm.checkpoints import CheckpointStore  # noqa: E402
from local_llm.config import Config  # noqa: E402
from local_llm.core import ADAPTER_DIR  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.llm import GenerationStats  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.obslog import set_acting_user  # noqa: E402
from local_llm.policy import BROKER, bind_policy, set_tool_override  # noqa: E402
from local_llm.tools import (ToolRegistry, answer_denies_changes,  # noqa: E402
                             changes_footer, cut_for_context, overwrite_stat,
                             partial_read_marker, strip_changes_footer,
                             strip_protocol_labels)
from local_llm.training import RetrainManager  # noqa: E402
from local_llm.turnctx import TurnContext, reset_current_turn, set_current_turn  # noqa: E402
from local_llm.ui_script_chat import UI_JS_CHAT  # noqa: E402

# The README of the reported turn, to the character.
README_CHARS = 103_363
DRAFT_CHARS = 800


def _readme_text(n: int = README_CHARS) -> str:
    line = "This project runs a local model with tools, approvals and checkpoints.\n"
    text = "# mlx-h\n\n" + line * (n // len(line) + 1)
    return text[:n]


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
class _Scripted:
    """A ModelClient stand-in that replays canned replies."""

    def __init__(self, replies):
        self.replies = list(replies)

    def _next(self):
        return self.replies.pop(0) if self.replies else "done."

    def stream(self, messages, *a, **kw):
        text = self._next()

        async def gen():
            yield text
        return gen()

    async def complete_with_stats(self, messages, *a, **kw):
        return self._next(), GenerationStats()

    async def complete(self, messages, *a, **kw):
        return (await self.complete_with_stats(messages))[0]

    def reply_budget(self, messages, max_tokens=None, quiet=False):
        return max_tokens or 512

    async def wait_until_ready(self, timeout=0.0):
        return True

    @staticmethod
    def classify_error(exc):
        return "stalled"


class _NoSummaries(_Scripted):
    """Fails the test if anything asks the model to summarise."""

    async def complete_with_stats(self, messages, *a, **kw):
        raise AssertionError("a read_file result must never be summarised")


_SAVED_BINDING = dict(policy._POLICY_BOUND)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """No lexicon or network surprises; the policy binding restored after."""
    monkeypatch.setattr(agent_mod, "classify", lambda message, **kw: None)

    async def online(*a, **k):
        return True

    monkeypatch.setattr(agent_mod, "has_internet", online)
    BROKER.clear()
    yield
    policy._POLICY_BOUND.update(_SAVED_BINDING)
    BROKER.clear()


def _setup(replies=(), readme: str | None = None, **overrides):
    """A real registry on a temp project holding a 103 KB README.md."""
    project = Path(tempfile.mkdtemp(prefix="llm-test-fs-proj-"))
    if readme is not False:
        (project / "README.md").write_text(readme or _readme_text(), encoding="utf-8")
    base = dict(project_dir=str(project), agent_enabled=True, knowledge_triage=False,
                fast_path=False, incremental_reasoning=False, chunk_large_prompts=False,
                agent_max_steps=3, agent_min_steps=1, resilient_retries=1,
                ready_wait_timeout=0.01, auto_iterate_rounds=0, rag_enabled=False,
                summarise_tool_results=False, reference_expansion=False,
                approval_default_mode="off", tool_approval_mode="off",
                skills_dir=tempfile.mkdtemp(prefix="llm-test-fs-skills-"))
    base.update(overrides)
    cfg = Config(**base)
    db = Database(Path(tempfile.mkdtemp()) / "fs.db")
    reg = ToolRegistry(cfg, db)
    reg.checkpoints = CheckpointStore(Path(tempfile.mkdtemp()) / "cp")
    bind_policy(cfg, db)
    agent = Agent(cfg, reg, _Scripted(list(replies)))
    return agent, cfg, db, reg, project


def _write_call(path="README.md", content=None) -> str:
    return json.dumps({"tool": "write_file",
                       "args": {"path": path, "content": content or ("# New\n" + "x" * (DRAFT_CHARS - 6))}})


def _run(agent, message, turn=None, approve=None):
    """Drive one turn; `approve` ("allow"/"deny") answers every card it sees."""
    stop = threading.Event()
    seen: list[dict] = []

    def approver():
        while not stop.is_set():
            for item in BROKER.pending():
                seen.append(item)
                BROKER.resolve(item["id"], turn.user_id if turn else None, approve, "once")
            time.sleep(0.01)

    worker = None
    if approve:
        worker = threading.Thread(target=approver, daemon=True)
        worker.start()

    async def drain():
        token = set_current_turn(turn) if turn is not None else None
        try:
            return [ev async for ev in agent.run(message, [], conversation_id="c1",
                                                 max_tokens=256)]
        finally:
            if token is not None:
                reset_current_turn(token)
    try:
        return asyncio.run(drain())
    finally:
        stop.set()
        if worker is not None:
            worker.join(2)


def _of(events, kind):
    return [e for e in events if e.get("type") == kind]


def _turn(interactive=True):
    return TurnContext(user_id="u1", conversation_id="c1", interactive=interactive)


# --------------------------------------------------------------------------- #
# 1. The overwrite guard
# --------------------------------------------------------------------------- #
def test_overwrite_of_a_100kb_file_is_an_approval_card_in_an_interactive_turn():
    agent, cfg, db, reg, project = _setup(
        [_write_call(), "Wrote the README."], approval_default_mode="all")
    events = _run(agent, "/agent write a readme", turn=_turn(), approve="allow")
    cards = [e for e in _of(events, "approval_required") if e["name"] == "write_file"]
    assert len(cards) == 1, [e.get("name") for e in _of(events, "approval_required")]
    card = cards[0]
    # Path, size before -> after, the diff stat and the plain warning.
    assert card["reason"].startswith("Overwrite README.md: 103,363 -> 800 characters"), card
    assert "lines +" in card["reason"] and "removes most of the file" in card["reason"]
    assert "outputs/README.md" in card["reason"]
    assert card["scopes"] == ["once"]
    assert card["overwrite"]["before"] == README_CHARS and card["overwrite"]["after"] == DRAFT_CHARS
    assert card["overwrite"]["destructive"] is True
    # The drafted content is on the card, editable like any other argument.
    assert len(card["args"]["content"]) == DRAFT_CHARS
    # Approved: exactly that write went through.
    assert len((project / "README.md").read_text()) == DRAFT_CHARS
    result = [e for e in _of(events, "tool_result") if e["name"] == "write_file"][0]
    assert result["error"] is None and "103,363 -> 800 characters" in result["result"]
    db.close()


def test_a_denied_overwrite_card_leaves_the_file_alone():
    agent, cfg, db, reg, project = _setup(
        [_write_call(), "Okay, I left it."], approval_default_mode="all")
    events = _run(agent, "/agent write a readme", turn=_turn(), approve="deny")
    assert any(e["name"] == "write_file" for e in _of(events, "approval_required"))
    assert len((project / "README.md").read_text()) == README_CHARS
    db.close()


def test_the_card_shows_even_when_a_policy_rule_allows_writes():
    """"Even if approvals are otherwise off for writes": an override that
    always allows write_file does not make an overwrite silent."""
    agent, cfg, db, reg, project = _setup(
        [_write_call(), "Done."], approval_default_mode="risky")
    set_tool_override(db, "user", "u1", "write_file", "allow", created_by="u1")
    events = _run(agent, "/agent write a readme", turn=_turn(), approve="allow")
    cards = [e for e in _of(events, "approval_required") if e["name"] == "write_file"]
    assert len(cards) == 1 and "overwrite" in cards[0]
    db.close()


def test_a_new_file_is_not_put_on_an_overwrite_card():
    agent, cfg, db, reg, project = _setup(
        [_write_call(path="NOTES.md"), "Done."], approval_default_mode="risky")
    set_tool_override(db, "user", "u1", "write_file", "allow", created_by="u1")
    events = _run(agent, "/agent write notes", turn=_turn(), approve="allow")
    assert not [e for e in _of(events, "approval_required") if e["name"] == "write_file"]
    assert (project / "NOTES.md").is_file()
    db.close()


def test_dont_ask_mode_refuses_the_destructive_overwrite_without_asking():
    agent, cfg, db, reg, project = _setup([_write_call(), "Done."])
    events = _run(agent, "/agent write a readme", turn=_turn())
    assert not _of(events, "approval_required")
    result = [e for e in _of(events, "tool_result") if e["name"] == "write_file"][0]
    assert result["error"] and result["error"].startswith("ValueError: write_file refused")
    # The way out, in one step: edit_file, or the outputs path spelled out.
    assert "edit_file" in result["error"] and 'path="outputs/README.md"' in result["error"]
    assert "removing 99% of it" in result["error"]
    assert len((project / "README.md").read_text()) == README_CHARS
    db.close()


def test_a_non_interactive_turn_gets_the_same_refusal():
    # The user's mode asks, nobody can answer, unattended calls are allowed:
    # the policy lets the call run and the guard still refuses it.
    agent, cfg, db, reg, project = _setup(
        [_write_call(), "Done."], approval_default_mode="all",
        tool_approval_unattended="allow")
    events = _run(agent, "/agent write a readme", turn=_turn(interactive=False))
    assert not _of(events, "approval_required")
    result = [e for e in _of(events, "tool_result") if e["name"] == "write_file"][0]
    assert "write_file refused" in (result["error"] or ""), result
    assert len((project / "README.md").read_text()) == README_CHARS
    # No turn at all (a task run, /api/tools/call, the CLI): refused too.
    out, err = reg.call("write_file", {"path": "README.md", "content": "short"}, None)
    assert err and "outputs/README.md" in err
    db.close()


def test_small_edits_and_growth_are_allowed():
    agent, cfg, db, reg, project = _setup()
    original = (project / "README.md").read_text()
    # A whole-file write that changes one line: not destructive.
    changed = original.replace("# mlx-h", "# mlx-h (local models)", 1)
    out, err = reg.call("write_file", {"path": "README.md", "content": changed}, None)
    assert err is None and out.startswith("Updated README.md: 103,363 -> 103,378"), out
    # Shrinking by 1,000 characters (1%) is a change, not a loss.
    out, err = reg.call("write_file", {"path": "README.md", "content": changed[:-1000]}, None)
    assert err is None, err
    # By more than 2,000 characters is refused even though it is under half.
    current = (project / "README.md").read_text()
    out, err = reg.call("write_file", {"path": "README.md", "content": current[:-2500]}, None)
    assert err and "removing 2,500 characters of it" in err, err
    # edit_file is the normal way, and reports the size change.
    out, err = reg.call("edit_file", {"path": "README.md", "find": "(local models)",
                                      "replace": "(local)"}, None)
    assert err is None and "-> " in out and "characters)" in out, out
    # A tiny file can be rewritten freely: halving 8 characters is not a loss.
    (project / "tiny.txt").write_text("ORIGINAL")
    assert reg.call("write_file", {"path": "tiny.txt", "content": "v0"}, None)[1] is None
    db.close()


def test_the_suggestion_never_points_at_the_protected_file():
    agent, cfg, db, reg, project = _setup(readme=False)
    (project / "outputs").mkdir()
    (project / "outputs" / "report.md").write_text("r" * 5000)
    out, err = reg.call("write_file", {"path": "outputs/report.md", "content": "r"}, None)
    assert err and 'path="outputs/report-new.md"' in err, err
    db.close()


def test_overwrite_stat_thresholds():
    assert overwrite_stat("a" * 10_000, "a" * 4_000)["destructive"]       # 60% gone
    assert overwrite_stat("a" * 100_000, "a" * 97_000)["destructive"]     # 3,000 gone
    assert not overwrite_stat("a" * 100_000, "a" * 99_000)["destructive"]
    assert overwrite_stat("a" * 300, "b")["destructive"]                  # 299 of 300 gone
    assert not overwrite_stat("a" * 200, "")["destructive"]               # under the floor
    stat = overwrite_stat("one\ntwo\nthree\n", "one\n2\nthree\nfour\n")
    assert (stat["lines_added"], stat["lines_removed"]) == (2, 1)


# --------------------------------------------------------------------------- #
# 2. Partial reads
# --------------------------------------------------------------------------- #
def test_read_file_marks_a_partial_read_at_the_top_and_bottom_and_pages():
    agent, cfg, db, reg, project = _setup(tool_raw_chars=2000)
    out, err = reg.call("read_file", {"path": "README.md"}, None)
    assert err is None
    lines = out.split("\n")
    marker = "[README.md: showing characters 1-1600 of 103,363. Call read_file with offset=1600 to continue.]"
    assert lines[0] == marker and lines[-1] == marker, (lines[0], lines[-1])
    assert len(out) <= cfg.tool_raw_chars
    text = (project / "README.md").read_text()
    assert "\n".join(lines[1:-1]) == text[:1600]
    # Paging continues exactly where the marker said, and accepts the aliases.
    nxt, err = reg.call("read_file", {"path": "README.md", "start": 1600, "length": 500}, None)
    assert err is None and nxt.startswith("[README.md: showing characters 1601-2100 of 103,363. "
                                          "Call read_file with offset=2100 to continue.]")
    assert nxt.split("\n", 1)[1].rsplit("\n", 1)[0] == text[1600:2100]
    last, err = reg.call("read_file", {"path": "README.md", "offset": 103_000}, None)
    assert err is None and "the end of the file" in last.split("\n")[0]
    assert last.split("\n", 1)[1].rsplit("\n", 1)[0] == text[103_000:]
    past, err = reg.call("read_file", {"path": "README.md", "offset": 200_000}, None)
    assert err and "past the end" in err
    db.close()


def test_a_file_that_fits_is_returned_unmarked_and_the_indexer_gets_no_marker():
    agent, cfg, db, reg, project = _setup(tool_raw_chars=2000)
    (project / "small.txt").write_text("hello\n")
    assert reg.call("read_file", {"path": "small.txt"}, None) == ("hello\n", None)
    plain = reg._read_file("README.md", marked=False)
    assert not plain.startswith("[README.md") and len(plain) == 2000
    db.close()


def test_the_context_cut_keeps_an_honest_marker():
    agent, cfg, db, reg, project = _setup(tool_raw_chars=20000)
    agent.client = _NoSummaries([])
    agent.config.summarise_tool_results = True
    result, _ = reg.call("read_file", {"path": "README.md"}, None)
    cut, summarised = asyncio.run(agent.compress_tool_result("read_file", result, 1500))
    assert summarised is False
    first, last = cut.split("\n")[0], cut.split("\n")[-1]
    assert first == last and first.startswith("[README.md: showing characters 1-")
    shown = int(first.split("1-")[1].split(" ")[0])
    assert f"offset={shown} to continue" in first and "of 103,363" in first
    body = cut.split("\n", 1)[1].rsplit("\n", 1)[0]
    assert len(body) == shown and len(cut) <= 1500
    # Any other tool's cut says so at both ends too (no bare "[truncated]").
    other = cut_for_context("web_search", "r" * 5000, 1000)
    assert other.startswith("[web_search result cut to fit the context: showing characters 1-")
    assert other.endswith("more characters of this web_search result were not shown.]")
    # A whole small file cut by the budget still gets a paging marker.
    whole = cut_for_context("read_file", "w" * 3000, 1000)
    assert whole.startswith("[read_file: showing characters 1-") and "offset=" in whole
    assert partial_read_marker("a.md", 0, 10, 20) == (
        "[a.md: showing characters 1-10 of 20. Call read_file with offset=10 to continue.]")
    db.close()


# --------------------------------------------------------------------------- #
# 3. Final-answer hygiene and the Changes footer
# --------------------------------------------------------------------------- #
def test_the_footer_tells_the_truth_and_labels_are_stripped():
    note = json.dumps({"tool": "write_file", "args": {"path": "notes.txt", "content": "hello"}})
    agent, cfg, db, reg, project = _setup(
        [note, "TOOL RESULT [write_file]: I did not write anything. The README is not available."])
    events = _run(agent, "/agent save hello to notes.txt", turn=_turn())
    final = _of(events, "final")[-1]
    answer = final["answer"]
    assert "TOOL RESULT" not in answer
    assert answer.startswith("I did not write anything.")
    assert "**Changes made this turn**" in answer
    assert "- `notes.txt`: created, 5 bytes" in answer
    assert final["checkpoint"] and f"checkpoint `{final['checkpoint']}`" in answer
    assert "Note: the reply above says" in answer
    assert final["changes"] == [{"path": "notes.txt", "kind": "created", "before": None,
                                 "after": 5}]
    db.close()


def test_a_modified_file_reports_before_and_after_and_a_forged_footer_is_replaced():
    edit = json.dumps({"tool": "edit_file", "args": {"path": "README.md", "find": "# mlx-h",
                                                     "replace": "# mlx-h!"}})
    forged = ("Updated the title.\n\n---\n**Changes made this turn** (checkpoint `fake`):\n"
              "- `other.txt`: removed, was 1 bytes")
    agent, cfg, db, reg, project = _setup([edit, forged])
    events = _run(agent, "/agent add a bang to the title", turn=_turn())
    answer = _of(events, "final")[-1]["answer"]
    assert "`fake`" not in answer and "other.txt" not in answer
    assert answer.count("**Changes made this turn**") == 1
    assert "- `README.md`: modified, 103,363 -> 103,364 bytes" in answer
    assert "Note:" not in answer
    db.close()


def test_a_turn_that_changes_nothing_has_no_footer():
    agent, cfg, db, reg, project = _setup(["Nothing to do."])
    events = _run(agent, "/agent hello", turn=_turn())
    final = _of(events, "final")[-1]
    assert "Changes made this turn" not in final["answer"] and final["changes"] == []
    db.close()


def test_label_stripping_and_contradiction_helpers():
    text = ("TOOL RESULT [read_file]: the README is short.\n**HELPER RESULT:** ok\n"
            "```\nTOOL RESULT [x]: quoted on purpose\n```")
    cleaned = strip_protocol_labels(text)
    assert cleaned.startswith("the README is short.\nok\n"), cleaned
    assert "TOOL RESULT [x]: quoted on purpose" in cleaned
    assert strip_protocol_labels("> - PAGE TEXT [x]: quoted") == "> - quoted"
    assert strip_protocol_labels("TOOL RESULT:") == "TOOL RESULT:"     # never empties a reply
    # Only a footer-shaped tail is taken for a forged footer.
    mid = "See **Changes made this turn** below.\n\nMore text."
    assert strip_changes_footer(mid) == mid
    rows = [{"path": "README.md", "kind": "modified", "before": 10, "after": 5}]
    assert answer_denies_changes("The README is not available.", rows) == "README.md"
    assert answer_denies_changes("No files were changed.", rows) == ""
    assert answer_denies_changes("I updated the README.", rows) is None
    footer = changes_footer(rows, None)
    assert "no checkpoint was recorded" in footer and "10 -> 5 bytes" in footer
    assert strip_changes_footer("Answer." + footer) == "Answer."
    assert "—" not in footer


# --------------------------------------------------------------------------- #
# 4. Undo
# --------------------------------------------------------------------------- #
def _app(project):
    cfg = Config(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                 allow_test_user=False, node_token="secret-node-token",
                 project_dir=str(project),
                 skills_dir=tempfile.mkdtemp(prefix="llm-test-fs-skills-"))
    db = Database(Path(tempfile.mkdtemp()) / "app.db")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)  # never started
    reg = ToolRegistry(cfg, db)
    from local_llm.api import create_app
    app = create_app(cfg, db, mm, RetrainManager(db, mm, cfg), reg)
    return app, db, reg


def _client(app, admin, username):
    r = admin.post("/api/users", json={"username": username, "password": f"{username}pass12",
                                       "role": "user"})
    assert r.status_code in (200, 201), r.text
    client = TestClient(app)
    assert client.post("/api/auth/login", json={"username": username,
                                                "password": f"{username}pass12"}).status_code == 200
    return client


def test_the_owner_can_undo_their_turn_and_another_user_cannot():
    project = Path(tempfile.mkdtemp(prefix="llm-test-fs-undo-")).resolve()
    (project / "keep.txt").write_text("ORIGINAL", encoding="utf-8")
    app, db, reg = _app(project)
    reg.checkpoints = CheckpointStore(Path(tempfile.mkdtemp()) / "cp")
    with TestClient(app) as admin:
        assert admin.post("/api/auth/login", json={"username": "admin",
                                                   "password": "adminpw123"}).status_code == 200
        alice = _client(app, admin, "alice")
        bob = _client(app, admin, "bob")
        alice_id = db.get_user_by_username("alice")["id"]

        def turn(content):
            # A turn acting for alice, as the chat route would run it.
            set_acting_user(alice_id)
            reg.checkpoint_id = reg.checkpoints.new_id("c1")
            reg.call("write_file", {"path": "keep.txt", "content": content}, "c1")
            reg.call("write_file", {"path": "made.txt", "content": "NEW"}, "c1")
            return reg.checkpoint_id

        import contextvars
        cid = contextvars.copy_context().run(turn, "CHANGED")
        assert bob.post(f"/api/checkpoints/{cid}/rollback").status_code == 404
        assert (project / "keep.txt").read_text() == "CHANGED"
        done = alice.post(f"/api/checkpoints/{cid}/rollback")
        assert done.status_code == 200, done.text
        assert sorted(done.json()["restored"]) == ["keep.txt", "made.txt"]
        assert (project / "keep.txt").read_text() == "ORIGINAL"
        assert not (project / "made.txt").exists()
        # An administrator may undo anyone's turn.
        time.sleep(0.002)
        cid2 = contextvars.copy_context().run(turn, "AGAIN")
        assert admin.post(f"/api/checkpoints/{cid2}/rollback").status_code == 200
        assert (project / "keep.txt").read_text() == "ORIGINAL"
    db.close()


def test_the_changed_files_node_offers_undo_through_the_rollback_route():
    start = UI_JS_CHAT.index("var changed = (event.changed_files || [])")
    node = UI_JS_CHAT[start:UI_JS_CHAT.index("setRegenerateTarget", start)]
    assert "event.checkpoint" in node and '"Undo these changes"' in node
    assert "chatUndoTurn(" in node
    fn = UI_JS_CHAT[UI_JS_CHAT.index("async function chatUndoTurn"):]
    fn = fn[:fn.index("\n   }\n") + 6]
    assert fn.index("uiConfirm(") < fn.index("postJSON(")
    assert '"/api/checkpoints/" + encodeURIComponent(checkpointId) + "/rollback"' in fn
    assert "restored" in fn and "problems" in fn
    assert "—" not in fn and "style=" not in fn
