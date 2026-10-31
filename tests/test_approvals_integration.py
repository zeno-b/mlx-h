"""How the three README fixes work together in one turn.

Three parts were built separately and meet in Agent._dispatch:

  * the user's Approvals mode, set in the header control on every view
    (policy.py: all / risky / off, APPROVAL_DEFAULT_MODE = all);
  * the overwrite guard, which puts write_file on an existing file on an
    approval card unless that mode is off, and otherwise refuses a write that
    removes most of the file (tools.py, Agent._ag_overwrite_card);
  * the codebase_write intent for "check the codebase and write a readme.md",
    which offers file tools only and lists write_file/edit_file as
    confirm_tools (intents.py).

The reported turn under each part's own tests could still go wrong where they
meet: a second card for the same call (the intent's confirmation on top of
the guard's card or of the policy's), a confirmation after the user had said
"Don't ask" or "always allow this tool", a header change that waits for the
next turn, or a fetch_url the intent never offered running anyway. Each test
below runs the REAL classify, policy, registry and Agent on a temp project
with a scripted model; nothing here starts a server or loads a model.
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import local_llm.config as config_mod  # noqa: E402

config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-appr-integration-"))

import local_llm.agent as agent_mod  # noqa: E402
from local_llm import policy  # noqa: E402
from local_llm.agent import Agent  # noqa: E402
from local_llm.checkpoints import CheckpointStore  # noqa: E402
from local_llm.config import Config  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.llm import GenerationStats  # noqa: E402
from local_llm.obslog import set_acting_user  # noqa: E402
from local_llm.policy import BROKER, bind_policy, set_tool_override  # noqa: E402
from local_llm.prefs import PreferenceStore  # noqa: E402
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.turnctx import TurnContext, reset_current_turn, set_current_turn  # noqa: E402

__all__: list[str] = []

MESSAGE = "check the codebase and write a readme.md"
README_CHARS = 103_363
DRAFT = "# Local LLM\n\n" + "A self-hosted assistant with tools and approvals.\n" * 16


def _readme() -> str:
    line = "This project runs a local model with tools, approvals and checkpoints.\n"
    return ("# mlx-h\n\n" + line * (README_CHARS // len(line) + 1))[:README_CHARS]


def _call(tool: str, **args) -> str:
    return json.dumps({"tool": tool, "args": args})


class _Scripted:
    """A ModelClient stand-in that replays canned replies and keeps the prompts."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[list[dict]] = []

    def _next(self, messages):
        self.prompts.append(messages)
        return self.replies.pop(0) if self.replies else "done."

    def stream(self, messages, *a, **kw):
        text = self._next(messages)

        async def gen():
            yield text
        return gen()

    async def complete_with_stats(self, messages, *a, **kw):
        return self._next(messages), GenerationStats()

    async def complete(self, messages, *a, **kw):
        return (await self.complete_with_stats(messages))[0]

    def reply_budget(self, messages, max_tokens=None, quiet=False):
        return max_tokens or 512

    async def wait_until_ready(self, timeout=0.0):
        return True

    @staticmethod
    def classify_error(exc):
        return "stalled"


_SAVED_BINDING = dict(policy._POLICY_BOUND)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    async def online(*a, **k):
        return True

    monkeypatch.setattr(agent_mod, "has_internet", online)
    BROKER.clear()
    yield
    policy._POLICY_BOUND.update(_SAVED_BINDING)
    BROKER.clear()


def _setup(replies, mode=None, **overrides):
    """Real registry and policy on a temp project with a 103 KB README.md.

    `mode` is the user's header choice (None: never chose, so
    APPROVAL_DEFAULT_MODE=all applies). The administrator floor stays off.
    """
    project = Path(tempfile.mkdtemp(prefix="llm-test-appr-int-proj-"))
    (project / "README.md").write_text(_readme(), encoding="utf-8")
    (project / "app.py").write_text("print('hello')\n", encoding="utf-8")
    base = dict(project_dir=str(project), agent_enabled=True, knowledge_triage=True,
                fast_path=False, incremental_reasoning=False, chunk_large_prompts=False,
                agent_max_steps=8, agent_min_steps=1, resilient_retries=1,
                ready_wait_timeout=0.01, auto_iterate_rounds=0, rag_enabled=False,
                summarise_tool_results=False, reference_expansion=False,
                approval_default_mode="all", tool_approval_mode="off",
                skills_dir=tempfile.mkdtemp(prefix="llm-test-appr-int-skills-"))
    base.update(overrides)
    cfg = Config(**base)
    db = Database(Path(tempfile.mkdtemp()) / "appr-int.db")
    if mode is not None:
        PreferenceStore(db).update("u1", {"approval_mode": mode})
    reg = ToolRegistry(cfg, db)
    reg.checkpoints = CheckpointStore(Path(tempfile.mkdtemp()) / "cp")
    fetched: list[dict] = []

    def no_network(**kw):
        fetched.append(kw)
        return "fetched"

    for name in ("fetch_url", "web_search"):
        if reg.get(name) is not None:
            reg._tools[name].handler = no_network
    bind_policy(cfg, db)
    client = _Scripted(replies)
    agent = Agent(cfg, reg, client)
    agent._partials_dir = Path(tempfile.mkdtemp(prefix="llm-test-appr-int-partials-"))
    return agent, client, db, reg, project, fetched


def _drive(agent, on_card=None, message=MESSAGE):
    """Run one interactive turn for user u1; on_card(event) -> (decision, scope).

    The card is answered inside the event loop, the way the chat route's
    POST .../approvals/{id} reaches BROKER.resolve while the turn waits.
    """
    turn = TurnContext(user_id="u1", conversation_id="c1", interactive=True,
                       cancel=asyncio.Event())

    async def drain():
        token = set_current_turn(turn)
        set_acting_user("u1")
        events = []
        try:
            async for ev in agent.run(message, [], conversation_id="c1", max_tokens=256):
                events.append(ev)
                if ev.get("type") == "approval_required":
                    decision, scope = (on_card(ev) if on_card else ("allow", "once"))
                    BROKER.resolve(ev["id"], "u1", decision, scope)
        finally:
            reset_current_turn(token)
        return events
    return asyncio.run(drain())


def _of(events, kind):
    return [e for e in events if e.get("type") == kind]


def _final(events) -> dict:
    return next(e for e in events if e.get("type") == "final")


def test_ask_every_action_puts_each_call_on_exactly_one_card():
    """The default user: every call asks once. The README overwrite is ONE
    card (the guard's, with the size before -> after), not the guard's plus
    the intent's confirmation; denied, the file stays and the model's write
    to outputs/README.md is a card of its own."""
    agent, client, db, reg, project, fetched = _setup([
        _call("list_files", path="."),
        _call("read_file", path="README.md"),
        _call("write_file", path="README.md", content=DRAFT),
        _call("write_file", path="outputs/README.md", content=DRAFT),
        "TOOL RESULT [write_file]: I wrote outputs/README.md.",
    ])

    def answer(card):
        if card["name"] == "write_file" and card["args"]["path"] == "README.md":
            return "deny", "once"
        return "allow", "once"

    events = _drive(agent, answer)
    notice = next(e for e in events if e.get("type") == "notice" and e.get("intent"))
    assert notice["intent"] == "codebase_write"
    calls = [(e["name"], e["args"].get("path")) for e in _of(events, "tool_call")]
    cards = [(e["name"], e["args"].get("path")) for e in _of(events, "approval_required")]
    assert cards == calls, (cards, calls)
    by_path = {e["args"].get("path"): e for e in _of(events, "approval_required")
               if e["name"] == "write_file"}
    guard = by_path["README.md"]
    assert guard["scopes"] == ["once"] and "confirm" not in guard
    assert guard["overwrite"]["before"] == README_CHARS
    assert guard["reason"].startswith(f"Overwrite README.md: {README_CHARS:,} -> {len(DRAFT)}")
    assert by_path["outputs/README.md"]["scopes"] == ["once", "conversation", "tool"]
    assert len((project / "README.md").read_text()) == README_CHARS
    assert (project / "outputs" / "README.md").read_text() == DRAFT
    # No web tool in any prompt of the turn.
    for messages in client.prompts:
        system = str(messages[0].get("content") or "")
        assert "\n- fetch_url: " not in system and "\n- web_search: " not in system
    answer_text = _final(events)["answer"]
    assert "TOOL RESULT" not in answer_text
    assert "**Changes made this turn**" in answer_text and "`outputs/README.md`" in answer_text
    db.close()


def test_dont_ask_runs_without_cards_and_the_guard_refuses_the_overwrite():
    """Header at Don't ask: no card at all, the intent's confirmation
    included. The 103 KB README is protected by the guard, whose refusal
    names outputs/README.md, and the model's retry there goes through."""
    agent, client, db, reg, project, fetched = _setup([
        _call("list_files", path="."),
        _call("read_file", path="README.md"),
        _call("write_file", path="README.md", content=DRAFT),
        _call("write_file", path="outputs/README.md", content=DRAFT),
        "Saved the new README as outputs/README.md.",
    ], mode="off")
    events = _drive(agent, lambda card: pytest.fail(f"asked about {card['name']}"))
    assert not _of(events, "approval_required")
    results = [e for e in _of(events, "tool_result") if e["name"] == "write_file"]
    assert "write_file refused" in (results[0]["error"] or "")
    assert 'path="outputs/README.md"' in results[0]["error"]
    assert results[1]["error"] is None and "Created outputs/README.md" in results[1]["result"]
    assert len((project / "README.md").read_text()) == README_CHARS
    assert "**Changes made this turn**" in _final(events)["answer"]
    db.close()


def test_an_always_allow_answer_is_not_asked_again_by_the_intent():
    """"Always allow this tool" on a card writes the user's own rule. On a
    codebase_write turn the intent's confirmation used to ask about edit_file
    anyway, so the scope saved no clicks; now the user's rule stands."""
    agent, client, db, reg, project, fetched = _setup([
        _call("edit_file", path="app.py", old="hello", new="hello, world"),
        "Updated app.py.",
    ])
    set_tool_override(db, "user", "u1", "edit_file", "allow", created_by="u1")
    events = _drive(agent, lambda card: pytest.fail(f"asked about {card['name']}"))
    assert not _of(events, "approval_required")
    assert "hello, world" in (project / "app.py").read_text()
    db.close()


def test_an_administrator_allow_still_gets_the_intents_one_confirmation():
    """A global "allow write_file" is the administrator's, not the user's
    answer: the codebase_write turn still confirms the write, once, with a
    reason that names the file."""
    agent, client, db, reg, project, fetched = _setup([
        _call("write_file", path="NOTES.md", content="# Notes\n"),
        "Wrote NOTES.md.",
    ])
    set_tool_override(db, "global", "", "write_file", "allow", created_by="admin")
    events = _drive(agent)
    cards = _of(events, "approval_required")
    assert len(cards) == 1 and cards[0]["confirm"] is True and cards[0]["scopes"] == ["once"]
    assert cards[0]["reason"] == "check this change to NOTES.md before it runs"
    assert (project / "NOTES.md").read_text() == "# Notes\n"
    db.close()


def test_a_conversation_scope_answer_covers_the_next_call_of_that_tool():
    agent, client, db, reg, project, fetched = _setup([
        _call("read_file", path="app.py"),
        _call("read_file", path="README.md"),
        "Read both.",
    ])
    events = _drive(agent, lambda card: ("allow", "conversation"))
    cards = _of(events, "approval_required")
    assert [c["args"]["path"] for c in cards] == ["app.py"]
    assert len([e for e in _of(events, "tool_result") if e["name"] == "read_file"]) == 2
    db.close()


def test_a_header_change_applies_to_the_next_tool_call():
    """The user switches to Don't ask while the first card is open: the
    second call of the same turn no longer asks."""
    agent, client, db, reg, project, fetched = _setup([
        _call("list_files", path="."),
        _call("read_file", path="app.py"),
        "Done.",
    ])

    def answer(card):
        PreferenceStore(db).update("u1", {"approval_mode": "off"})
        return "allow", "once"

    events = _drive(agent, answer)
    assert [c["name"] for c in _of(events, "approval_required")] == ["list_files"]
    assert [e["name"] for e in _of(events, "tool_result")] == ["list_files", "read_file"]
    db.close()


def test_a_tool_the_intent_did_not_offer_is_neither_run_nor_asked_about():
    """The reported turn's hallucinated fetch_url: answered with the tools
    that are available, without a card and without a request."""
    agent, client, db, reg, project, fetched = _setup([
        _call("fetch_url", url="https://raw.githubusercontent.com/LocalLLM/README.md"),
        "I will work from the files instead.",
    ])
    events = _drive(agent, lambda card: pytest.fail(f"asked about {card['name']}"))
    assert fetched == []
    result = next(e for e in _of(events, "tool_result") if e["name"] == "fetch_url")
    assert "not available for this request" in result["result"]
    assert "list_files" in result["result"] and "write_file" in result["result"]
    db.close()


def test_an_approved_overwrite_is_undone_by_its_owner_checkpoint():
    """Ask for changes: reads run, the overwrite card is the only one, and
    the turn's checkpoint belongs to the turn's user (the rollback route
    lets the owner undo) and restores the 103 KB file."""
    agent, client, db, reg, project, fetched = _setup([
        _call("read_file", path="README.md"),
        _call("write_file", path="README.md", content=DRAFT),
        "Replaced the README.",
    ], mode="risky")
    events = _drive(agent, lambda card: ("allow", "once"))
    cards = _of(events, "approval_required")
    assert [(c["name"], "overwrite" in c) for c in cards] == [("write_file", True)]
    assert (project / "README.md").read_text() == DRAFT
    final = _final(events)
    checkpoint = reg.checkpoints.get(final["checkpoint"])
    assert checkpoint is not None and checkpoint.user_id == "u1"
    restored, problems = reg.checkpoints.rollback(final["checkpoint"])
    assert restored == ["README.md"] and problems == []
    assert len((project / "README.md").read_text()) == README_CHARS
    db.close()
