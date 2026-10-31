"""Phase B of upgrade/ROUTING_AND_TOOLING.md, agent side.

Lane selection from an Intent (P1), confirm_tools through the approval card,
intent-gated and relevance-fixed retrieval (P2), the attachment summary and the
drafted save_skill call (P4), the one-write document lane (P5), and the
reasoning instruction gate plus the per-step think cap (P6).

intents.classify is replaced with small fakes here, so these tests pin the
agent's half of the contract whatever rules the lexicon grows.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import local_llm.config as config_mod  # noqa: E402

config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-phase-b-agent-"))

import local_llm.agent as agent_mod  # noqa: E402
from local_llm.agent import Agent  # noqa: E402
from local_llm.config import Config  # noqa: E402
from local_llm.core import estimate_tokens, rag_query_tokens  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.intents import Intent  # noqa: E402
from local_llm.llm import (GenerationStats, PROSE_REASONING_INSTRUCTION,  # noqa: E402
                           REASONING_INSTRUCTION, reasoning_prompt_wanted)
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.turnctx import TurnContext, reset_current_turn, set_current_turn  # noqa: E402


# --------------------------------------------------------------------------- #
# Harness: a scripted model, a recorded registry, a fake lexicon.
# --------------------------------------------------------------------------- #
class _Client:
    """A ModelClient stand-in: each reply is a string or a list of chunks."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[list[dict]] = []

    def _next(self):
        return self.replies.pop(0) if self.replies else "done."

    def stream(self, messages, *a, **kw):
        self.prompts.append(messages)
        reply = self._next()
        chunks = [reply] if isinstance(reply, str) else list(reply)

        async def gen():
            for chunk in chunks:
                await asyncio.sleep(0)
                yield chunk

        return gen()

    async def complete_with_stats(self, messages, *a, **kw):
        self.prompts.append(messages)
        reply = self._next()
        text = reply if isinstance(reply, str) else "".join(reply)
        stats = GenerationStats()
        stats.prompt_tokens = sum(estimate_tokens(m.get("content")) for m in messages)
        stats.completion_tokens = estimate_tokens(text)
        return text, stats

    async def complete(self, messages, *a, **kw):
        text, _ = await self.complete_with_stats(messages, *a, **kw)
        return text

    def reply_budget(self, messages, max_tokens=None, quiet=False):
        return max_tokens or 512

    async def wait_until_ready(self, timeout=0.0):
        return True

    @staticmethod
    def classify_error(exc):
        return "the model stalled"


def _agent(replies, **overrides):
    base = dict(agent_enabled=True, knowledge_triage=False, fast_path=False,
                incremental_reasoning=False, chunk_large_prompts=False,
                agent_max_steps=4, agent_min_steps=1, resilient_retries=1,
                ready_wait_timeout=0.01, auto_iterate_rounds=0, show_internals=False,
                rag_enabled=False,
                skills_dir=tempfile.mkdtemp(prefix="llm-test-phase-b-skills-"))
    base.update(overrides)
    cfg = Config(**base)
    db = Database(Path(tempfile.mkdtemp()) / "agent.db")
    reg = ToolRegistry(cfg, db)
    client = _Client(replies)
    return Agent(cfg, reg, client), cfg, db, reg, client


def _record_calls(reg, result="tool output here"):
    """Replace the registry's executor; nothing touches the filesystem."""
    calls = []

    def call(name, args, cid=None):
        calls.append((name, dict(args)))
        if name == "create_document":
            reg.documents_written = list(reg.documents_written) + [str(args.get("path"))]
            return f"Created {args.get('path')}.", None
        return result, None

    reg.call = call
    return calls


def _drain(agent, message, turn=None, **kw):
    async def main():
        token = set_current_turn(turn) if turn is not None else None
        try:
            return [ev async for ev in agent.run(message, [], "t", 256, None, **kw)]
        finally:
            if token is not None:
                reset_current_turn(token)
    return asyncio.run(main())


def _of(events, kind):
    return [e for e in events if e.get("type") == kind]


def _final(events) -> dict:
    return _of(events, "final")[-1]


def _call(tool, **args) -> str:
    return json.dumps({"tool": tool, "args": args})


def _is_router(prompt) -> bool:
    return bool(prompt) and str(prompt[0].get("content", "")).startswith("You are a router.")


@pytest.fixture
def lexicon(monkeypatch):
    """Set what intents.classify returns for every message; records each call."""
    seen: list[dict] = []
    state = {"intent": None}

    def fake(message, **kw):
        seen.append({"message": message, **kw})
        return state["intent"]

    monkeypatch.setattr(agent_mod, "classify", fake)

    async def online(*a, **k):
        return True

    monkeypatch.setattr(agent_mod, "has_internet", online)

    def use(intent):
        state["intent"] = intent
        return seen

    return use


SKILL_FILE = """---
name: devils-advocate
description: "Adversarial reviewer for plans. Returns exactly three concerns."
---

# Devil's Advocate Agent

**Role:** Adversarial thinker. Finds what's wrong before others do.

""" + ("Rule: give exactly three specific concerns, each rated by severity. " * 60)


def _attachment(name="devils-advocate.md", text=SKILL_FILE):
    return [{"type": "text", "text": f"[attached file: {name}]\n{text}"}]


# --------------------------------------------------------------------------- #
# P1: lanes.
# --------------------------------------------------------------------------- #
def test_reply_lane_answers_without_a_model_call(lexicon):
    agent, cfg, db, reg, client = _agent(["should never be used"])
    lexicon(Intent(name="disabled_capability", lane="reply",
                   reply="The shell is switched off. Settings > Tools > Catalogue."))
    events = _drain(agent, "run the tests")
    assert client.prompts == []
    notes = [e["message"] for e in _of(events, "notice")]
    assert "Understood as: disabled_capability (lexicon)" in notes
    assert _final(events)["answer"].startswith("The shell is switched off.")
    assert _final(events)["steps"] == 0
    db.close()


def test_clarify_lane_asks_its_one_question(lexicon):
    agent, cfg, db, reg, client = _agent(["should never be used"])
    lexicon(Intent(name="memory_forget", lane="clarify",
                   clarify_question="Which note should I forget: timezone or editor?"))
    events = _drain(agent, "forget that")
    assert client.prompts == []
    assert _final(events)["answer"] == "Which note should I forget: timezone or editor?"
    db.close()


def test_tools_lane_skips_the_router_and_narrows_the_catalogue(lexicon):
    agent, cfg, db, reg, client = _agent([_call("list_files"), "The repo has two modules."],
                                         knowledge_triage=True)
    calls = _record_calls(reg)
    seen = lexicon(Intent(name="codebase_review", lane="tools", retrieval="off",
                          tool_subset=("repo_map", "list_files", "search_files",
                                       "read_file", "final_answer")))
    events = _drain(agent, "check the codebase for design mistakes and summarize")
    # The lexicon saw the registered tools and the typed message.
    assert seen[0]["message"].startswith("check the codebase")
    assert "read_file" in seen[0]["registered_tools"]
    assert not any(_is_router(p) for p in client.prompts), "the router was asked anyway"
    system = client.prompts[0][0]["content"]
    for name in ("list_files", "search_files", "read_file", "final_answer"):
        assert f"- {name}:" in system, name
    for name in ("web_search", "create_document", "remember", "save_skill"):
        assert f"- {name}:" not in system, name
    # Nor is the model advised to prefer a web_search it was not given.
    assert "web_search" not in system
    assert calls == [("list_files", {})]
    assert _final(events)["answer"] == "The repo has two modules."
    db.close()


def test_prose_lane_has_no_tool_protocol_and_no_router(lexicon):
    agent, cfg, db, reg, client = _agent(["Here is how it works."], knowledge_triage=True)
    lexicon(Intent(name="self_describe", lane="prose"))
    events = _drain(agent, "what can you do for me today")
    assert len(client.prompts) == 1 and not _is_router(client.prompts[0])
    assert "Available tools" not in client.prompts[0][0]["content"]
    assert _final(events)["answer"] == "Here is how it works."
    db.close()


def test_no_intent_leaves_routing_as_it_was(lexicon):
    agent, cfg, db, reg, client = _agent(['{"action":"answer"}', "Paris."],
                                         knowledge_triage=True)
    lexicon(None)
    events = _drain(agent, "what is the capital of france")
    assert _is_router(client.prompts[0])
    assert not [e for e in _of(events, "notice") if "Understood as" in e["message"]]
    db.close()


def test_a_tools_lane_with_nothing_registered_falls_back_to_the_router(lexicon):
    agent, cfg, db, reg, client = _agent(['{"action":"answer"}', "Nothing to list."],
                                         knowledge_triage=True)
    lexicon(Intent(name="skill_list", lane="tools", tool_subset=("no_such_tool",)))
    events = _drain(agent, "list every widget you know about")
    assert _is_router(client.prompts[0])
    assert not [e for e in _of(events, "notice") if "Understood as" in e["message"]]
    db.close()


def test_explicit_search_command_is_not_reinterpreted(lexicon):
    agent, cfg, db, reg, client = _agent(["ok."])
    seen = lexicon(Intent(name="kb_query", lane="tools",
                          tool_subset=("search_docs", "final_answer")))
    _drain(agent, "/no-search what is in my knowledge base")
    assert seen == []
    db.close()


def test_router_decision_is_logged_at_info_with_the_intent_metadata(lexicon):
    agent, cfg, db, reg, client = _agent(["Saved."])
    _record_calls(reg)
    lexicon(Intent(name="memory_write", lane="tools",
                   tool_subset=("remember", "final_answer"), retrieval="off"))
    records: list[logging.LogRecord] = []

    class _Keep(logging.Handler):
        def emit(self, record):
            records.append(record)

    target = logging.getLogger("local_llm.agent")
    handler, before = _Keep(level=logging.DEBUG), target.level
    target.addHandler(handler)
    target.setLevel(logging.INFO)
    try:
        _drain(agent, "remember that my timezone is Europe/Brussels")
    finally:
        target.removeHandler(handler)
        target.setLevel(before)
    decisions = [r for r in records if "router.decision" in r.getMessage()]
    assert decisions and decisions[0].levelno == logging.INFO
    fields = decisions[0].__dict__
    assert fields.get("intent") == "memory_write" and fields.get("lane") == "tools"
    assert fields.get("tool_subset") == ["remember", "final_answer"]
    # Metadata only: the message never reaches the log line.
    assert "Brussels" not in json.dumps({k: str(v) for k, v in fields.items()})
    db.close()


# --------------------------------------------------------------------------- #
# P1: confirm_tools.
# --------------------------------------------------------------------------- #
class _Decision:
    def __init__(self, action, reason=""):
        self.action, self.risk, self.reason, self.rule = action, "", reason, "test"


class _Broker:
    def __init__(self):
        self.requests: list[tuple] = []

    def request(self, turn, call_id, name, args, risk, reason):
        self.requests.append((name, dict(args)))
        return f"ap{len(self.requests)}"

    async def wait(self, approval_id, timeout):
        await asyncio.sleep(0)
        return {"decision": "allow", "scope": "once", "args": None}

    def resolve(self, *a, **k):
        return True


@pytest.fixture
def allow_policy(monkeypatch):
    broker = _Broker()

    class _Policy:
        BROKER = broker

        @staticmethod
        def decide(name, args, turn):
            # The real policy's default allow carries this reason.
            return _Decision("allow", "tool approvals are off")

    monkeypatch.setattr(agent_mod, "_ag_policy", lambda: _Policy)
    return broker


SAVE = Intent(name="skill_save", lane="tools", retrieval="off",
              tool_subset=("save_skill", "search_skills", "final_answer"),
              confirm_tools=("save_skill",), targets=("skills", "attachment"),
              referent="attachment")


def test_confirm_tools_ask_in_an_interactive_turn(lexicon, allow_policy):
    agent, cfg, db, reg, client = _agent(
        [_call("save_skill", name="note-taking", description="d", body="step one"),
         "Saved it."])
    calls = _record_calls(reg)
    lexicon(SAVE)
    turn = TurnContext(user_id="u1", conversation_id="t", interactive=True,
                       cancel=asyncio.Event())
    events = _drain(agent, "save this as a skill", turn=turn)
    asked = _of(events, "approval_required")
    assert len(asked) == 1 and asked[0]["name"] == "save_skill"
    assert asked[0]["confirm"] is True and asked[0]["scopes"] == ["once"]
    assert asked[0]["args"]["body"] == "step one"
    # The card says why it asks, not why the policy would have allowed it.
    assert asked[0]["reason"] == "check what will be saved before it runs"
    assert [c[0] for c in calls] == ["save_skill"]
    db.close()


def test_confirm_tools_follow_policy_when_nobody_is_watching(lexicon, allow_policy):
    agent, cfg, db, reg, client = _agent(
        [_call("save_skill", name="note-taking", description="d", body="step one"),
         "Saved it."])
    calls = _record_calls(reg)
    lexicon(SAVE)
    turn = TurnContext(user_id="u1", conversation_id="t", interactive=False,
                       cancel=asyncio.Event())
    events = _drain(agent, "save this as a skill", turn=turn)
    assert not _of(events, "approval_required")
    assert allow_policy.requests == []
    assert [c[0] for c in calls] == ["save_skill"]
    db.close()


def test_a_tool_outside_confirm_tools_is_not_asked_about(lexicon, allow_policy):
    agent, cfg, db, reg, client = _agent([_call("search_skills", query="notes"), "None."])
    _record_calls(reg)
    lexicon(SAVE)
    turn = TurnContext(user_id="u1", conversation_id="t", interactive=True,
                       cancel=asyncio.Event())
    events = _drain(agent, "save this as a skill", turn=turn)
    assert not _of(events, "approval_required")
    db.close()


# --------------------------------------------------------------------------- #
# P4: the attachment summary and the drafted save_skill call.
# --------------------------------------------------------------------------- #
def test_the_router_reads_a_one_line_attachment_summary(lexicon):
    agent, cfg, db, reg, client = _agent(['{"action":"answer"}', "Noted."],
                                         knowledge_triage=True)
    lexicon(None)
    _drain(agent, "add this to skills", attachments=_attachment())
    router = [p for p in client.prompts if _is_router(p)]
    assert router, "the router was not asked"
    asked = router[0][-1]["content"]
    size = agent_mod._ag_size(len(SKILL_FILE.encode()))
    assert asked == ("add this to skills\n[attached: devils-advocate.md, markdown, "
                     f"{size}, frontmatter name=devils-advocate]"), asked
    # The router never receives the file's text itself.
    assert "Adversarial thinker" not in asked
    db.close()


def test_the_lexicon_sees_the_attachment_names(lexicon):
    agent, cfg, db, reg, client = _agent(["ok."])
    seen = lexicon(None)
    _drain(agent, "add this to skills", attachments=_attachment())
    assert seen[0]["has_attachment"] is True
    assert seen[0]["attachment_names"] == ("devils-advocate.md",)
    db.close()


def test_skill_save_drafts_the_call_and_fills_in_the_attached_body(lexicon, allow_policy):
    marker = agent_mod._AG_ATTACHMENT_BODY
    agent, cfg, db, reg, client = _agent(
        [_call("save_skill", name="devils-advocate", description="Adversarial reviewer",
               body=marker), "Saved devils-advocate."])
    calls = _record_calls(reg)
    lexicon(SAVE)
    turn = TurnContext(user_id="u1", conversation_id="t", interactive=True,
                       cancel=asyncio.Event())
    events = _drain(agent, "add this to skills", turn=turn, attachments=_attachment())
    first_user = client.prompts[0][-1]["content"]
    assert '"name": "devils-advocate"' in first_user
    assert '"description": "Adversarial reviewer for plans. Returns exactly three concerns."' \
        in first_user
    assert marker in first_user
    body = SKILL_FILE.split("---\n", 2)[2].strip()
    assert calls == [("save_skill", {"name": "devils-advocate",
                                     "description": "Adversarial reviewer",
                                     "body": body})]
    # The person confirming sees the real body, not the marker.
    assert _of(events, "approval_required")[0]["args"]["body"] == body
    db.close()


def test_attachment_records_read_name_kind_size_and_frontmatter():
    files = agent_mod._ag_attachments(_attachment() + [{"type": "image_url",
                                                        "image_url": {"url": "x"}}])
    assert files[0]["name"] == "devils-advocate.md" and files[0]["kind"] == "markdown"
    assert files[0]["meta"]["name"] == "devils-advocate"
    assert files[1]["kind"] == "image"
    assert agent_mod._ag_size(6012) == "6 KB"
    assert agent_mod._ag_skill_draft(files)["name"] == "devils-advocate"
    assert agent_mod._ag_skill_draft(agent_mod._ag_attachments(
        _attachment("plain.md", "# No frontmatter here"))) is None


# --------------------------------------------------------------------------- #
# P2: retrieval gating, the overlap fix, the frequency ceiling, two per document.
# --------------------------------------------------------------------------- #
class _StubKB:
    fts_enabled = True

    def __init__(self, hits):
        self.hits = hits

    def search_documents(self, query, limit=3, only=None, user_id=None):
        return list(self.hits)


def _rag_agent(hits, **overrides):
    agent, cfg, db, reg, client = _agent([], rag_enabled=True, **overrides)
    reg.db = _StubKB(hits)
    return agent, db


def test_retrieval_off_never_injects_and_kb_overrides_the_heuristics():
    agent, db = _rag_agent([{"path": "notes/crud.md",
                             "chunk": "A crud program reads and writes rows."}])
    code = "write a basic c++ crud program"
    assert agent.with_retrieved_context(code, 4000) == code               # auto: held back
    assert "notes/crud.md" in agent.with_retrieved_context(code, 4000, retrieval="kb")
    question = "explain crud rows"
    assert "notes/crud.md" in agent.with_retrieved_context(question, 4000)
    assert agent.with_retrieved_context(question, 4000, retrieval="off") == question
    db.close()


def test_an_intent_with_retrieval_off_keeps_passages_out_of_the_prompt(lexicon):
    agent, cfg, db, reg, client = _agent(["Reviewed."], rag_enabled=True)
    reg.db = _StubKB([{"path": "skills/x.md", "chunk": "codebase design mistakes review"}])
    lexicon(Intent(name="codebase_review", lane="tools", retrieval="off",
                   tool_subset=("list_files", "read_file", "final_answer")))
    _drain(agent, "check the codebase for design mistakes")
    assert "Reference material" not in client.prompts[0][-1]["content"]
    db.close()


def test_a_word_in_every_path_no_longer_admits_a_passage():
    hit = {"path": "claude-skills-main/skills/cooking/SKILL.md",
           "chunk": "Braise the vegetables slowly over a low heat."}
    agent, db = _rag_agent([hit])
    assert rag_query_tokens("skills?") == {"skills"}
    assert agent.with_retrieved_context("skills?", 4000) == "skills?"
    db.close()


def test_a_path_counts_when_the_user_named_that_file():
    hit = {"path": "kb/book-to-skill.md", "chunk": "Nothing in here matches."}
    agent, db = _rag_agent([hit])
    named = "summarise book-to-skill.md"
    assert "kb/book-to-skill.md" in agent.with_retrieved_context(named, 4000)
    unnamed = "summarise the book skill"
    assert agent.with_retrieved_context(unnamed, 4000) == unnamed
    db.close()


def test_at_most_two_passages_per_document():
    hits = [{"path": "a.md", "chunk": f"zebra fact {i}"} for i in range(4)]
    hits.append({"path": "b.md", "chunk": "zebra elsewhere"})
    agent, db = _rag_agent(hits)
    out = agent.with_retrieved_context("zebra", 4000)
    assert out.count("[a.md]") == 2 and out.count("[b.md]") == 1, out
    db.close()


def _indexed(n_docs: int, **overrides):
    agent, cfg, db, reg, client = _agent([], rag_enabled=True, **overrides)
    if not getattr(db, "fts_enabled", False) or getattr(db, "search_mode", "") != "fts5":
        db.close()
        pytest.skip("SQLite FTS5 is not available in this Python")
    for i in range(n_docs):
        extra = " Zebra migration patterns across the plains." if i == 7 else ""
        db.index_document(f"repo/skills/doc{i}.md",
                          f"Skills catalogue entry {i}: notes about skills and "
                          f"teamwork number {i}.{extra}", f"doc{i}")
    return agent, db


def test_a_word_in_most_documents_is_ignored_by_the_overlap_test():
    agent, db = _indexed(25)
    only_common = "tell me about skills"
    assert agent.with_retrieved_context(only_common, 4000) == only_common
    mixed = agent.with_retrieved_context("skills zebra", 4000)
    assert "[repo/skills/doc7.md]" in mixed
    assert mixed.count("[repo/") == 1, mixed
    db.close()


def test_the_ceiling_is_off_at_one_and_below_twenty_documents():
    agent, db = _indexed(25, rag_df_ceiling=1.0)
    assert "Reference material" in agent.with_retrieved_context("tell me about skills", 4000)
    db.close()
    agent, db = _indexed(5)
    assert "Reference material" in agent.with_retrieved_context("tell me about skills", 4000)
    db.close()


def test_the_frequency_is_cached_per_word():
    agent, db = _indexed(25)
    agent.with_retrieved_context("skills zebra", 4000)
    cached = {key for (_db, key) in agent._ag_df_cache}
    assert {"", "skills", "zebra"} <= cached
    db.close()


# --------------------------------------------------------------------------- #
# P5: one write per document turn.
# --------------------------------------------------------------------------- #
REPORT = ("# The Solar System\n\nAn overview.\n\n## Planets\n\nEight of them.\n\n"
          "## Moons\n\nMany.\n\n## Missions\n\nVoyager and others.\n")


def test_a_document_turn_ends_after_the_first_successful_write(lexicon):
    agent, cfg, db, reg, client = _agent(
        [_call("create_document", path="solar.pdf", content=REPORT),
         _call("create_document", path="solar.pdf", content=REPORT + "more"),
         "should never be reached"], agent_min_steps=2)
    calls = _record_calls(reg)
    lexicon(None)
    events = _drain(agent, "write me a pdf report about the solar system")
    assert [c[0] for c in calls] == ["create_document"]
    assert len(client.prompts) == 1, "a step was spent after the file existed"
    answer = _final(events)["answer"]
    assert "solar.pdf" in answer
    for heading in ("- Planets", "- Moons", "- Missions"):
        assert heading in answer, answer
    assert _final(events)["documents"] == ["solar.pdf"]
    db.close()


def test_a_revision_request_allows_a_second_write(lexicon):
    agent, cfg, db, reg, client = _agent(
        [_call("create_document", path="mars.pdf", content="# Mars\n\n## Draft\n"),
         _call("create_document", path="mars.pdf", content="# Mars\n\n## Final\n"),
         "Both passes are done."])
    calls = _record_calls(reg)
    lexicon(None)
    events = _drain(agent, "write a pdf report about mars, then revise it once")
    assert [c[0] for c in calls] == ["create_document", "create_document"]
    assert _final(events)["answer"] == "Both passes are done."
    db.close()


def test_a_second_write_to_the_same_path_is_refused(lexicon):
    agent, cfg, db, reg, client = _agent(
        [_call("create_document", path="notes.md", content="# A"),
         _call("create_document", path="./notes.md", content="# B"),
         "Done."])
    calls = _record_calls(reg)
    lexicon(Intent(name="kb_write", lane="tools",
                   tool_subset=("create_document", "final_answer")))
    events = _drain(agent, "put my notes somewhere safe")
    assert calls == [("create_document", {"path": "notes.md", "content": "# A"})]
    refused = _of(events, "tool_result")[-1]
    assert refused["error"] and "already" in refused["result"]
    assert "already created in this turn" in client.prompts[-1][-1]["content"]
    assert _final(events)["answer"] == "Done."
    db.close()


def test_a_document_intent_researches_its_named_sources(lexicon):
    agent, cfg, db, reg, client = _agent(
        [_call("search_docs", query="useful facts"),
         _call("create_document", path="kb-overview.pdf", content=REPORT)])
    calls = _record_calls(reg)
    lexicon(Intent(name="document_create", lane="document", retrieval="kb",
                   research_sources=("knowledge_base",),
                   targets=("knowledge_base", "document")))
    events = _drain(agent, "use the knowledge base to craft a pdf showing an overview "
                           "of usefull facts in the knowledge base")
    system = client.prompts[0][0]["content"]
    assert "- search_docs:" in system and "- create_document:" in system
    assert "- web_search:" not in system and "- fetch_url:" not in system
    directive = client.prompts[0][-1]["content"]
    assert "STEP 1, RESEARCH" in directive and "knowledge base" in directive
    assert [c[0] for c in calls] == ["search_docs", "create_document"]
    assert "kb-overview.pdf" in _final(events)["answer"]
    db.close()


def test_the_document_reply_falls_back_to_first_lines_without_headings():
    reply = agent_mod._ag_document_done(
        "out/a.md", "# Title\n\n- First point here.\n- Second point.\n", "Markdown")
    assert reply.startswith("I created **a.md** (Markdown).")
    assert "- First point here." in reply and "- Second point." in reply


# --------------------------------------------------------------------------- #
# P6: the reasoning instruction gate and the think cap.
# --------------------------------------------------------------------------- #
THINKING_MODEL = "mlx-community/Qwen3-8B-4bit"
PLAIN_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"


def _system(model, tools=True, **overrides):
    agent, cfg, db, reg, client = _agent([], model=model, **overrides)
    base, _ = agent.build_base([], "what is two plus two", 256, tools=tools)
    db.close()
    return base[0]["content"]


def test_reasoning_instruction_only_for_catalogue_thinking_models():
    assert REASONING_INSTRUCTION in _system(THINKING_MODEL)
    assert PROSE_REASONING_INSTRUCTION in _system(THINKING_MODEL, tools=False)
    assert REASONING_INSTRUCTION not in _system(PLAIN_MODEL)
    assert REASONING_INSTRUCTION not in _system("someone/unknown-model-3b")
    assert PROSE_REASONING_INSTRUCTION not in _system("someone/unknown-model-3b", tools=False)


def test_reasoning_prompt_always_and_never_override_the_catalogue():
    assert REASONING_INSTRUCTION in _system(PLAIN_MODEL, reasoning_prompt="always")
    assert REASONING_INSTRUCTION not in _system(THINKING_MODEL, reasoning_prompt="never")
    assert REASONING_INSTRUCTION not in _system(THINKING_MODEL, reasoning_visible=False)
    assert Config(reasoning_prompt="sometimes").reasoning_prompt == "auto"
    assert reasoning_prompt_wanted(Config(model=THINKING_MODEL)) is True


def test_a_step_that_thinks_past_the_cap_is_asked_for_the_answer():
    agent, cfg, db, reg, client = _agent(
        [["<think>", "a" * 30, "b" * 30, "c" * 30, "</think>never shown"],
         "The answer is 4."], think_max_tokens=10)
    events = asyncio.run(_collect(agent.run("what is two plus two", [], "t", 256, None)))
    assert len(client.prompts) == 2
    retry = client.prompts[1]
    assert retry[-1]["content"] == agent_mod._AG_THINK_CAP_TOOLS
    assert retry[-2]["role"] == "assistant" and "aaaa" in retry[-2]["content"]
    assert "cccc" not in retry[-2]["content"], "the stream was not stopped at the cap"
    assert any("still thinking" in e["message"] for e in _of(events, "notice"))
    final = _final(events)
    assert final["answer"] == "The answer is 4."
    assert "aaaa" in final["reasoning"]
    db.close()


def test_a_retry_that_thinks_past_the_cap_too_is_salvaged():
    agent, cfg, db, reg, client = _agent(
        [["<think>", "a" * 60], ["<think>", "d" * 60]], think_max_tokens=10)
    events = asyncio.run(_collect(agent.run("what is two plus two", [], "t", 256, None)))
    assert len(client.prompts) == 2
    answer = _final(events)["answer"]
    assert answer.startswith("I ran out of thinking room") and "aaaa" in answer
    db.close()


def test_no_cap_at_zero():
    agent, cfg, db, reg, client = _agent(
        [["<think>", "a" * 400, "</think>", "Four."]], think_max_tokens=0)
    events = asyncio.run(_collect(agent.run("what is two plus two", [], "t", 256, None)))
    assert len(client.prompts) == 1
    assert _final(events)["answer"] == "Four."
    db.close()


async def _collect(gen):
    return [ev async for ev in gen]


def test_new_settings_are_clamped():
    cfg = Config(rag_df_ceiling=5.0, think_max_tokens=-3)
    assert cfg.rag_df_ceiling == 1.0 and cfg.think_max_tokens == 0
    assert {"rag_df_ceiling", "reasoning_prompt", "think_max_tokens"} <= set(Config.MUTABLE)
