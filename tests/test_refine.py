"""Tests for Auto mode's answer check (local_llm/refine.py) and parallel search.

Three layers. The pure helpers that decide what may be searched (and what
never leaves the machine), the thread primitives in websearch.py, and the
check itself driven through a fake model client and fake tools, then the
POST /api/chat/refine route on the real app with the agent, the model and
the network replaced. The agent's page prefetch (_AgPrefetch) is tested on
its own with a stand-in agent.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_llm.config as config_mod  # noqa: E402
config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-refine-"))

from fastapi.testclient import TestClient  # noqa: E402

import local_llm.refine as refine_mod  # noqa: E402
from local_llm.agent import _AgPrefetch  # noqa: E402
from local_llm.api import create_app  # noqa: E402
from local_llm.config import Config  # noqa: E402
from local_llm.core import ADAPTER_DIR  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.llm import GenerationStats  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.refine import (  # noqa: E402
    AnswerRefiner, excerpt, is_confirmation, parse_query_lines, pick_source_urls,
    query_messages, read_urls, refine_messages, refine_queries, refine_skip_reason,
    split_page)
from local_llm.streams import TurnRegistry  # noqa: E402
from local_llm.textutil import own_material, search_subject  # noqa: E402
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.training import RetrainManager  # noqa: E402
from local_llm.websearch import parallel_map, search_workers, thread_task  # noqa: E402


QUESTION = "What is the population of Tokyo?"
ANSWER = "Tokyo has about 13 million people."
PRIVATE = "Zanzibar-Quokka-Ledger"      # a word only the answer contains

SEARCH_RESULT = (
    "1. Tokyo population 2026\n   https://stats.example.org/tokyo\n   Tokyo population figures\n"
    "2. Tokyo - Wikipedia\n   https://en.example.org/wiki/Tokyo\n   Tokyo is the capital\n"
    "3. Japan cities\n   https://cities.example.net/japan/tokyo\n   Tokyo population and area")
# What a later round's different query finds: pages no earlier round read.
SEARCH_CENSUS = (
    "1. Tokyo census 2025\n   https://census.example.org/tokyo\n   Tokyo census population\n"
    "2. Tokyo by ward\n   https://wards.example.org/tokyo\n   Tokyo population by ward")
PAGE = ("Tokyo population\n{url}\n\n"
        "Tokyo is the capital of Japan and its most populous city with a population of "
        "about 14 million people in the city proper, according to the 2026 estimate.\n\n"
        "The Greater Tokyo Area has a population of about 37 million people, the largest "
        "metropolitan area in the world by most measures.\n\n"
        "Home | About | Contact\n\n"
        "The city's population density is among the highest of any major city, with "
        "about 6,400 people per square kilometre across the twenty-three wards.")


# --------------------------------------------------------------------------- #
# Thread primitives
# --------------------------------------------------------------------------- #
def test_parallel_map_runs_at_once_keeps_order_context_and_per_item_errors():
    marker = contextvars.ContextVar("marker", default="unset")
    marker.set("caller")
    threads: set[str] = set()

    def job(n: int) -> str:
        threads.add(threading.current_thread().name)
        time.sleep(0.2)
        if n == 2:
            raise ValueError("boom")
        return f"{n}:{marker.get()}"

    started = time.time()
    out = parallel_map(job, [0, 1, 2, 3], 4)
    took = time.time() - started
    assert took < 0.6, f"four 0.2s jobs on four threads took {took:.2f}s"
    assert [r for r, _ in out] == ["0:caller", "1:caller", None, "3:caller"]
    assert isinstance(out[2][1], ValueError) and out[0][1] is None
    assert len(threads) > 1
    # One worker (or one item) runs inline, in order.
    assert [r for r, _ in parallel_map(lambda n: n * 2, [1, 2, 3], 1)] == [2, 4, 6]
    assert parallel_map(lambda n: n, [], 4) == []


def test_thread_task_returns_errors_as_values_and_honours_the_gate():
    async def scenario():
        gate = asyncio.Semaphore(1)
        running = []
        peak = []

        def job(n):
            running.append(n)
            peak.append(len(running))
            time.sleep(0.05)
            running.remove(n)
            if n == 1:
                raise RuntimeError("bad")
            return n

        results = await asyncio.gather(*(thread_task(job, n, gate) for n in range(3)))
        return results, max(peak)

    results, peak = asyncio.run(scenario())
    assert results[0] == (0, None) and results[2] == (2, None)
    assert results[1][0] is None and isinstance(results[1][1], RuntimeError)
    assert peak == 1, "the gate admits one job at a time"


def test_search_workers_is_capped_by_the_setting_the_batch_and_eight():
    assert search_workers(Config(search_workers=4), 10) == 4
    assert search_workers(Config(search_workers=4), 2) == 2
    assert search_workers(Config(search_workers=1), 5) == 1
    assert Config(search_workers=50).search_workers == 8          # clamped
    assert Config(auto_refine_delay_s=1).auto_refine_delay_s == 10
    assert Config(auto_refine_sources=0).auto_refine_sources == 1


# --------------------------------------------------------------------------- #
# What may be checked, and what may be sent
# --------------------------------------------------------------------------- #
def test_own_material_matches_what_search_subject_refuses():
    for text in ("fix the bug in config.py", "summarise my notes",
                 "```\nprint(1)\n```", "key=value\nother=thing"):
        assert own_material(text), text
        assert search_subject(text) is None, text
    assert not own_material(QUESTION)
    assert search_subject(QUESTION) == QUESTION


def test_skip_reason_refuses_private_or_unsuitable_exchanges():
    final = {"answer": ANSWER, "tools_used": ["web_search"]}
    assert refine_skip_reason(QUESTION, ANSWER, final, ["web_search", "fetch_url"]) is None
    assert refine_skip_reason("what does config.py set the timeout to?", ANSWER, final)
    assert refine_skip_reason(QUESTION, ANSWER, {"tools_used": ["search_docs"]})
    assert "read_file" in refine_skip_reason(QUESTION, ANSWER, final, ["read_file"])
    assert refine_skip_reason(QUESTION, ANSWER, final, context_injected=True)
    assert refine_skip_reason(QUESTION, ANSWER, dict(final, documents=["a.docx"]))
    assert refine_skip_reason(QUESTION, ANSWER, final, question_meta={"attachments": ["x"]})
    assert refine_skip_reason("hi", ANSWER, final)                       # not substantive
    assert refine_skip_reason(QUESTION, "x" * 7000, final)               # too long to rewrite
    assert refine_skip_reason(QUESTION, "", final)
    # Writing has no facts of its own to check, and a user's own business is
    # nothing a web page knows (the 2027 outreach strategy turn was both).
    assert "writing" in refine_skip_reason(
        "prepare a 2027 cold outreach strategy for my it consulting company", ANSWER, final)
    assert "writing" in refine_skip_reason("draft an email to my landlord", ANSWER, final)
    assert "business" in refine_skip_reason("how do I grow my consulting business?", ANSWER,
                                            final)
    assert refine_skip_reason("plan a 3 day trip to Rome", ANSWER, final) is None


def test_queries_are_vetted_anchored_deduplicated_and_capped():
    proposed = ["Tokyo population 2026 estimate", "tokyo population",
                "best pizza in Naples",                        # not about the question
                "Tokyo population in my notes",                # the user's own material
                "Tokyo metropolitan area official statistics",
                "Tokyo census results"]
    got = refine_queries(QUESTION, proposed)
    assert got[0] == QUESTION
    assert "best pizza in Naples" not in got and "Tokyo population in my notes" not in got
    assert len(got) == 3
    assert got[1] == "Tokyo population 2026 estimate"
    # A question too long to search still lets vetted proposals through.
    long_q = "Tokyo " + "population history " * 20
    assert refine_queries(long_q, ["Tokyo population history"]) == ["Tokyo population history"]


def test_query_lines_lose_numbering_bullets_and_quotes():
    text = '<think>hmm</think>1. "tokyo population"\n- tokyo census\n\n* `greater tokyo`\nQuery 4: x'
    assert parse_query_lines(text) == ["tokyo population", "tokyo census", "greater tokyo", "x"]


def test_later_rounds_repeat_no_query_and_fall_back_to_the_question():
    used = [QUESTION, "Tokyo population 2026"]
    got = refine_queries(QUESTION, ["Tokyo population 2026", "Tokyo census results"], used)
    assert got == ["Tokyo census results"]
    # Nothing new proposed: the question again, since results move on and
    # only pages no round has read get fetched.
    assert refine_queries(QUESTION, ["tokyo population 2026"], used) == [QUESTION]
    prompt = json.dumps(query_messages(QUESTION, used))
    assert "Already searched" in prompt and "Tokyo population 2026" in prompt
    assert "Already searched" not in json.dumps(query_messages(QUESTION))


def test_later_rounds_number_sources_after_the_ones_already_cited():
    sources = [{"url": "https://a.example.org/x", "title": "A", "text": "alpha"},
               {"url": "https://b.example.org/y", "title": "B", "text": "beta"}]
    first = json.dumps(refine_messages(QUESTION, ANSWER, sources))
    assert "[1] A" in first and "[2] B" in first and "already checked" not in first
    later = json.dumps(refine_messages(QUESTION, ANSWER, sources, first_index=4, round_no=3))
    assert "[4] A" in later and "[5] B" in later and "[1] A" not in later
    assert "already checked against other sources" in later


def test_the_query_prompt_never_sees_the_answer():
    text = json.dumps(query_messages(QUESTION))
    assert QUESTION in text and ANSWER not in text


def test_source_urls_round_robin_across_queries_and_skip_read_and_listing_pages():
    block_a = ("1. A1\n   https://a.example.org/one\n   tokyo population\n"
               "2. A2\n   https://a.example.org/two\n   tokyo population")
    block_b = ("1. B1\n   https://b.example.org/one\n   tokyo population\n"
               "2. Tag\n   https://b.example.org/tag/tokyo\n   tokyo population")
    got = pick_source_urls([block_a, block_b], QUESTION, exclude={"https://a.example.org/one"},
                           limit=3)
    assert got == ["https://b.example.org/one", "https://a.example.org/two"]
    assert read_urls({"trace": [{"name": "fetch_url", "args": {"url": "u1"}},
                                {"name": "web_search", "args": {"query": "q"}}]}) == {"u1"}


def test_excerpt_keeps_relevant_prose_in_page_order_within_budget():
    url = "https://stats.example.org/tokyo"
    title, body = split_page(PAGE.format(url=url), url)
    assert title == "Tokyo population"
    # The best-scoring paragraph (question words twice, one more answer word)
    # is the one a tight budget keeps.
    cut = excerpt(body, QUESTION, ANSWER, 260)
    assert "37 million" in cut and "14 million" not in cut and len(cut) <= 260
    assert "Home | About" not in excerpt(body, QUESTION, ANSWER, 5000)
    full = excerpt(body, QUESTION, ANSWER, 5000)
    assert full.index("14 million") < full.index("37 million") < full.index("6,400")


def test_confirmation_is_the_marker_or_a_short_prose_verdict():
    assert is_confirmation("NO CHANGES")
    assert is_confirmation("**No changes.**")
    assert is_confirmation("The earlier answer is accurate.")
    assert not is_confirmation("The earlier answer is accurate, but " + "x" * 300)
    assert not is_confirmation("Tokyo has about 14 million people [1].")


# --------------------------------------------------------------------------- #
# The check, with a fake model and fake tools
# --------------------------------------------------------------------------- #
class FakeClient:
    """complete_with_stats for the query call, stream for the verdict."""

    def __init__(self, reply: str, queries: str = "Tokyo population 2026\nTokyo census",
                 finish: str = "stop"):
        self.reply = reply
        self.queries = queries
        self.finish = finish
        self.prompts: list = []
        self.closed = False
        self.streamed = 0

    async def complete_with_stats(self, messages, max_tokens=None, temperature=None, **kw):
        self.prompts.append(("complete", messages, kw))
        last = messages[-1]["content"]
        if "Relevant findings from this source" in last:
            # A source's note: the first lines of the one page it was shown.
            return "Note: " + last.split("):\n", 1)[-1][:160], GenerationStats()
        return self.queries, GenerationStats()

    def verdict_prompt(self):
        return [p for p in self.prompts if p[0] == "stream"][-1][1]

    def note_prompts(self):
        return [p[1] for p in self.prompts
                if p[0] == "complete" and "Relevant findings" in p[1][-1]["content"]]

    def stream(self, messages, max_tokens=None, temperature=None, stats=None, **kw):
        self.prompts.append(("stream", messages, kw))
        client = self

        async def gen():
            try:
                for i in range(0, len(client.reply), 4):
                    client.streamed += 1
                    yield client.reply[i:i + 4]
                if stats is not None:
                    stats.finish_reason = client.finish
            finally:
                client.closed = True
        return gen()


class FakeTools:
    """ToolRegistry.call stand-in: records calls and the threads they ran on."""

    def __init__(self, dead: set[str] = frozenset(), delay: float = 0.15):
        self.calls: list = []
        self.threads: set[str] = set()
        self.dead = set(dead)
        self.delay = delay
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def __call__(self, name, args):
        with self.lock:
            self.calls.append((name, dict(args)))
            self.threads.add(threading.current_thread().name)
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.delay)
            if name == "web_search":
                census = "census" in str(args.get("query", "")).lower()
                return (SEARCH_CENSUS if census else SEARCH_RESULT), None
            url = args["url"]
            if url in self.dead:
                return "ConnectError: dead", "ConnectError: dead"
            return PAGE.format(url=url), None
        finally:
            with self.lock:
                self.active -= 1


def _run(refiner, **kw):
    async def scenario():
        events, result = [], None
        async for event in refiner.run(QUESTION, ANSWER + " " + PRIVATE, **kw):
            if "__result__" in event:
                result = event["__result__"]
            else:
                events.append(event)
        return events, result
    return asyncio.run(scenario())


def _cfg(**kw):
    base = dict(context_size=8192, max_tokens=1024, auto_refine_sources=2, search_workers=4)
    base.update(kw)
    return Config(**base)


def test_check_searches_and_reads_in_parallel_and_returns_an_improved_answer():
    client = FakeClient("Tokyo has about 14 million people in the city proper [1].")
    tools = FakeTools()
    events, result = _run(AnswerRefiner(_cfg(), client, tools))
    assert result["changed"] is True
    assert result["answer"].startswith("Tokyo has about 14 million")
    assert result["queries"][0] == QUESTION and len(result["queries"]) == 3
    assert len(result["sources"]) == 2
    searches = [a for n, a in tools.calls if n == "web_search"]
    fetches = [a for n, a in tools.calls if n == "fetch_url"]
    assert len(searches) == 3 and len(fetches) == 2
    assert tools.peak >= 2, "the searches and the fetches run at the same time"
    # Nothing of the answer went to the search engine; the model wrote the
    # queries from the question alone.
    assert all(PRIVATE not in a["query"] and "13 million" not in a["query"] for a in searches)
    kind, messages, kw = client.prompts[0]
    assert kind == "complete" and PRIVATE not in json.dumps(messages)
    assert kw.get("extra") == {"enable_thinking": False}
    # The verdict prompt carries the answer and the excerpts.
    verdict = json.dumps(client.verdict_prompt())
    assert PRIVATE in verdict and "14 million" in verdict
    types = [e["type"] for e in events]
    assert types.count("tool_call") == 5 and types.count("tool_result") == 5
    assert "token" in types


def test_every_page_gets_its_own_bounded_pass_and_never_sees_the_answer():
    # One page per model call: the prompt that held every page at once ran an
    # 8GB Mac out of GPU memory and took the model server down mid-reply.
    client = FakeClient("Tokyo has about 14 million people in the city proper [1].")
    _events, result = _run(AnswerRefiner(_cfg(), client, FakeTools(delay=0)))
    notes = client.note_prompts()
    assert len(notes) == len(result["sources"]) == 2
    for prompt in notes:
        text = json.dumps(prompt)
        assert text.count("Source [") == 1 and PRIVATE not in text and "13 million" not in text
    verdict = json.dumps(client.verdict_prompt())
    assert "NOTES FROM WEB SOURCES" in verdict and "Note: " in verdict
    assert "6,400 people" not in verdict, "the verdict reads notes, not whole pages"


def test_a_stream_the_server_dropped_is_never_stored():
    # No finish reason: the server stopped mid-reply (out of GPU memory).
    client = FakeClient("Tokyo has about 14 million people and the Greater", finish="")
    _events, result = _run(AnswerRefiner(_cfg(), client, FakeTools(delay=0)))
    assert result["changed"] is False and "stopped before finishing" in result["reason"]


def test_a_reply_that_repeats_the_prompt_is_never_stored():
    client = FakeClient("QUESTION:\nWhat is the population of Tokyo?\n\nYOUR EARLIER ANSWER:\n"
                        "Tokyo has about 13 million people.\n\nTokyo has 14 million [1].")
    _events, result = _run(AnswerRefiner(_cfg(), client, FakeTools(delay=0)))
    assert result["changed"] is False and "own instructions" in result["reason"]


def test_a_leaked_tool_name_is_dropped_from_what_the_check_reads_and_writes():
    client = FakeClient("web_search\n\nTokyo has about 14 million people [1].")
    refiner = AnswerRefiner(_cfg(), client, FakeTools(delay=0))

    async def scenario():
        result = None
        async for event in refiner.run(QUESTION, "web_search\n\nTokyo has about 13 million."):
            if "__result__" in event:
                result = event["__result__"]
        return result

    result = asyncio.run(scenario())
    assert result["changed"] is True and result["answer"].startswith("Tokyo has about 14")
    asked = json.dumps(client.verdict_prompt())
    assert "YOUR EARLIER ANSWER:\\nTokyo has about 13 million." in asked


def test_check_that_finds_nothing_wrong_stops_the_model_early():
    client = FakeClient("NO CHANGES\n" + "padding " * 200)
    events, result = _run(AnswerRefiner(_cfg(), client, FakeTools(delay=0)))
    assert result["changed"] is False and result["reason"] == "confirmed"
    assert client.closed and client.streamed < 10, "the stream is closed after the verdict"
    assert not [e for e in events if e["type"] == "token"]


def test_check_never_adopts_a_truncated_rewrite():
    client = FakeClient("Tokyo has about 14 million people and", finish="length")
    _events, result = _run(AnswerRefiner(_cfg(), client, FakeTools(delay=0)))
    assert result["changed"] is False and "budget" in result["reason"]


def test_dead_pages_are_replaced_from_the_spares_in_a_second_batch():
    dead = {"https://stats.example.org/tokyo"}
    client = FakeClient("Tokyo has about 14 million people [1].")
    tools = FakeTools(dead=dead, delay=0)
    _events, result = _run(AnswerRefiner(_cfg(), client, tools))
    urls = [s["url"] for s in result["sources"]]
    assert len(urls) == 2 and not set(urls) & dead
    assert len([1 for n, _ in tools.calls if n == "fetch_url"]) == 3


def test_check_with_no_readable_page_changes_nothing():
    client = FakeClient("anything", queries="Tokyo population 2026")
    tools = FakeTools(dead={"https://stats.example.org/tokyo", "https://en.example.org/wiki/Tokyo",
                            "https://cities.example.net/japan/tokyo"}, delay=0)
    _events, result = _run(AnswerRefiner(_cfg(), client, tools))
    assert result["changed"] is False and "could be read" in result["reason"]
    assert [p for p in client.prompts if p[0] == "stream"] == []


def test_a_later_round_reads_only_new_pages_and_says_when_none_are_left():
    client = FakeClient("Tokyo has about 14.1 million people [3].",
                        queries="Tokyo census results 2025")
    tools = FakeTools(delay=0)
    read_before = ["https://stats.example.org/tokyo", "https://en.example.org/wiki/Tokyo"]
    _events, result = _run(AnswerRefiner(_cfg(), client, tools), exclude_urls=read_before,
                           used_queries=[QUESTION], first_index=3, round_no=2)
    assert result["changed"] is True and result["queries"] == ["Tokyo census results 2025"]
    assert [s["index"] for s in result["sources"]] == [3, 4]
    assert {s["url"] for s in result["sources"]} == {"https://census.example.org/tokyo",
                                                     "https://wards.example.org/tokyo"}
    assert set(result["tried"]) == {s["url"] for s in result["sources"]}
    fetched = {a["url"] for n, a in tools.calls if n == "fetch_url"}
    assert not fetched & set(read_before)
    # Every page the searches can reach already read: nothing to do, and say so.
    every = read_before + ["https://cities.example.net/japan/tokyo",
                           "https://census.example.org/tokyo", "https://wards.example.org/tokyo"]
    _events, result = _run(AnswerRefiner(_cfg(), FakeClient("anything", queries=""),
                                         FakeTools(delay=0)),
                           exclude_urls=every, used_queries=[QUESTION], round_no=5)
    assert result["exhausted"] is True and result["changed"] is False
    assert "no pages left" in result["reason"]


def test_cancel_stops_the_check_before_the_model_is_asked():
    class Cancelled:
        def is_set(self):
            return True
    client = FakeClient("anything")
    _events, result = _run(AnswerRefiner(_cfg(), client, FakeTools(delay=0)), cancel=Cancelled())
    assert result["reason"] == "cancelled" and client.prompts == []


# --------------------------------------------------------------------------- #
# The route, on the real app
# --------------------------------------------------------------------------- #
def build_app(**overrides):
    base = dict(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                allow_test_user=False, node_token="secret-node-token",
                skills_dir=tempfile.mkdtemp(prefix="local-llm-test-skills-"),
                agent_enabled=True, knowledge_triage=False, context_size=8192,
                max_tokens=1024, auto_refine_sources=2)
    base.update(overrides)
    cfg = Config(**base)
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)  # never started
    rm = RetrainManager(db, mm, cfg)
    app = create_app(cfg, db, mm, rm, ToolRegistry(cfg, db))
    app.state.model_manager = mm
    mm.status = "ready"
    return app, cfg, db


def _wire(app, answer=ANSWER, reply="Tokyo has about 14 million people [1].", tools_used=()):
    """A chat turn that answers `answer`, then a model and tools for the check."""
    async def fake_run(message, history, conversation_id=None, max_tokens=None,
                       temperature=None, cancel=None, **kw):
        yield {"type": "token", "text": answer, "step": 1}
        yield {"type": "final", "answer": answer, "steps": 1, "trace": [],
               "tools_used": list(tools_used), "elapsed_ms": 1, "prompt_tokens": 1,
               "completion_tokens": 1, "truncated": False, "changed_files": [], "diff": ""}

    agent = app.state.agent
    agent.run_iterating = fake_run
    agent.client = FakeClient(reply)
    tools = FakeTools(delay=0)
    tools.kwargs = []

    def call(name, args, cid=None, **kw):
        tools.kwargs.append(kw)
        return tools(name, args)
    agent.registry.call = call

    async def online(*_a, **_k):
        return True
    refine_mod.has_internet = online
    return tools


def _frames(text: str) -> list:
    out = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                data = line[6:]
                out.append(data if data == "[DONE]" else json.loads(data))
    return out


def _chat(c, message, cid):
    r = c.post("/api/chat/stream", json={"message": message, "conversation_id": cid,
                                         "agent": True})
    assert r.status_code == 200, r.text
    frames = _frames(r.text)
    start = next(f for f in frames if isinstance(f, dict) and f["type"] == "start")
    return start["turn_id"]


def test_route_checks_the_answer_and_stores_the_improvement():
    # Approvals off: the check runs without asking anyone.
    app, _, db = build_app(approval_default_mode="off")
    tools = _wire(app)
    with TestClient(app) as c:
        assert c.post("/api/auth/login", json={"username": "admin",
                                               "password": "adminpw123"}).status_code == 200
        assert c.get("/api/health").json()["auto_refine_delay_s"] == 60
        turn_id = _chat(c, QUESTION, "conv1")
        r = c.post("/api/chat/refine", json={"conversation_id": "conv1", "turn_id": turn_id})
        assert r.status_code == 200 and "text/event-stream" in r.headers["content-type"], r.text
        frames = [f for f in _frames(r.text) if isinstance(f, dict)]
        assert frames[0]["type"] == "start" and frames[0]["refine"] is True
        final = frames[-1]
        assert final["type"] == "final" and final["changed"] is True and final["stored"] is True
        assert final["answer"].startswith("Tokyo has about 14 million")
        messages = c.get("/api/conversation/conv1").json()["messages"]
        assert messages[-1]["content"] == final["answer"]
        check = messages[-1]["meta"]["web_check"]
        assert check["original"] == ANSWER and check["changed"] is True
        assert len(check["sources"]) == 2 and check["queries"][0] == QUESTION
        assert any(n == "web_search" for n, _ in tools.calls)
        assert all(kw == {} for kw in tools.kwargs), "nobody was asked, so no own_request"
        # Once checked, the same answer is not checked again (a second tab).
        again = c.post("/api/chat/refine", json={"conversation_id": "conv1", "turn_id": turn_id})
        assert "skipped" in again.json()
        # The check is not the conversation's running turn.
        assert c.get("/api/chat/active", params={"conversation_id": "conv1"}).json()["turn"] is None


def test_route_asks_first_when_approvals_ask_before_every_action():
    # The installation default (conftest switches it off for the suite): ask
    # before every action. The check asks once, and a yes runs its calls as
    # the user's own request.
    app, _, _ = build_app(approval_default_mode="all")
    tools = _wire(app)
    with TestClient(app) as c:
        c.post("/api/auth/login", json={"username": "admin", "password": "adminpw123"})
        turn_id = _chat(c, QUESTION, "conv7")
        r = c.post("/api/chat/refine", json={"conversation_id": "conv7", "turn_id": turn_id})
        assert r.json()["ask"] is True and r.json()["pages"] == 2
        assert tools.calls == [], "nothing runs before the user says yes"
        r = c.post("/api/chat/refine", json={"conversation_id": "conv7", "turn_id": turn_id,
                                             "consent": True})
        final = [f for f in _frames(r.text) if isinstance(f, dict)][-1]
        assert final["type"] == "final" and final["changed"] is True
        assert tools.kwargs and all(kw == {"own_request": True} for kw in tools.kwargs)


def test_route_never_overrides_the_administrators_floor():
    # TOOL_APPROVAL_MODE=all is the administrator's: the user's yes cannot
    # stand in for it, so the check does not run at all.
    app, _, _ = build_app(tool_approval_mode="all")
    tools = _wire(app)
    with TestClient(app) as c:
        c.post("/api/auth/login", json={"username": "admin", "password": "adminpw123"})
        turn_id = _chat(c, QUESTION, "conv8")
        for consent in (False, True):
            r = c.post("/api/chat/refine", json={"conversation_id": "conv8", "turn_id": turn_id,
                                                 "consent": consent})
            assert "approval policy" in r.json()["skipped"]
        assert tools.calls == []


def test_permission_follows_the_users_mode_the_floor_and_unattended_rules():
    from local_llm.refine import refine_permission
    from local_llm.turnctx import TurnContext

    def permission(**cfg):
        app, _, _ = build_app(**cfg)
        # A user with no Approvals choice of their own gets APPROVAL_DEFAULT_MODE.
        turn = TurnContext(user_id="u1", conversation_id="c", interactive=False)
        return refine_permission(app.state.agent.registry, turn, QUESTION, "c")

    assert permission(approval_default_mode="off") == "allow"
    assert permission(approval_default_mode="risky") == "allow"      # read-only tools
    assert permission(approval_default_mode="all") == "consent"
    assert permission(approval_default_mode="all", tool_approval_unattended="allow") == "allow"
    assert permission(approval_default_mode="off", tool_approval_mode="all") == "deny"


def test_route_confirmation_marks_the_answer_checked_without_rewriting_it():
    app, _, _ = build_app(approval_default_mode="off")
    _wire(app, reply="NO CHANGES")
    with TestClient(app) as c:
        c.post("/api/auth/login", json={"username": "admin", "password": "adminpw123"})
        turn_id = _chat(c, QUESTION, "conv2")
        r = c.post("/api/chat/refine", json={"conversation_id": "conv2", "turn_id": turn_id})
        final = [f for f in _frames(r.text) if isinstance(f, dict)][-1]
        assert final["changed"] is False and final["reason"] == "confirmed"
        last = c.get("/api/conversation/conv2").json()["messages"][-1]
        assert last["content"] == ANSWER
        assert last["meta"]["web_check"]["changed"] is False
        assert "original" not in last["meta"]["web_check"]


def test_route_skips_private_stale_unknown_and_switched_off():
    app, cfg, _ = build_app()
    tools = _wire(app)
    with TestClient(app) as c:
        c.post("/api/auth/login", json={"username": "admin", "password": "adminpw123"})
        # The user's own material is never searched for.
        private = _chat(c, "what timeout does config.py set for the server?", "conv3")
        r = c.post("/api/chat/refine", json={"conversation_id": "conv3", "turn_id": private})
        assert "own material" in r.json()["skipped"]
        # A turn that read the knowledge base is not checked either.
        _wire(app, tools_used=["search_docs"])
        kb = _chat(c, QUESTION, "conv4")
        r = c.post("/api/chat/refine", json={"conversation_id": "conv4", "turn_id": kb})
        assert "search_docs" in r.json()["skipped"]
        # The conversation moved on: the checked answer is no longer the last.
        tools = _wire(app)
        first = _chat(c, QUESTION, "conv5")
        _chat(c, "And what is the population of Osaka?", "conv5")
        r = c.post("/api/chat/refine", json={"conversation_id": "conv5", "turn_id": first})
        assert "skipped" in r.json()
        r = c.post("/api/chat/refine", json={"conversation_id": "conv5", "turn_id": "nope"})
        assert "skipped" in r.json()
        assert c.post("/api/chat/refine", json={}).status_code == 400
        cfg.auto_refine = False
        latest = _chat(c, QUESTION, "conv6")
        r = c.post("/api/chat/refine", json={"conversation_id": "conv6", "turn_id": latest})
        assert r.json()["skipped"] == "AUTO_REFINE is off"
        assert not [1 for n, _ in tools.calls if n == "web_search"], "nothing was searched"


def _age_last_round(db, cid):
    """Make the answer's last round look long past, as if an interval had gone by."""
    row = db.get_messages(cid, 1)[-1]
    meta = json.loads(row["meta"])
    meta["web_check"]["checked_at"] = "2000-01-01T00:00:00+00:00"
    db.update_message(row["id"], row["content"], meta)


def _round(c, cid, **body):
    r = c.post("/api/chat/refine", json={"conversation_id": cid, **body})
    if "text/event-stream" not in r.headers.get("content-type", ""):
        return r.json()
    return [f for f in _frames(r.text) if isinstance(f, dict)][-1]


def test_route_rounds_keep_reading_new_pages_until_none_are_left():
    app, cfg, db = build_app(approval_default_mode="off")
    tools = _wire(app)
    client = app.state.agent.client
    with TestClient(app) as c:
        c.post("/api/auth/login", json={"username": "admin", "password": "adminpw123"})
        turn_id = _chat(c, QUESTION, "conv9")
        client.queries = "Tokyo population 2026"         # round 1 reaches the first pages only
        first = _round(c, "conv9", turn_id=turn_id)
        assert first["round"] == 1 and first["changed"] is True and first["message_id"]
        mid = first["message_id"]
        # Straight away is too soon: the server keeps its own gap between rounds.
        assert "too soon" in _round(c, "conv9", message_id=mid)["skipped"]
        # Round 2 sends queries no round sent and reads pages no round read,
        # numbered after the two the answer already cites.
        _age_last_round(db, "conv9")
        client.queries = "Tokyo census results 2025"
        client.reply = "Tokyo has about 14.1 million people [1][3]."
        second = _round(c, "conv9", message_id=mid)
        assert second["round"] == 2 and second["changed"] is True
        assert [s["index"] for s in second["sources"]] == [3, 4]
        assert second["queries"] == ["Tokyo census results 2025"]
        verdict = json.dumps(client.prompts[-1][1])
        assert "[3]" in verdict and "already checked against other sources" in verdict
        fetched = [a["url"] for n, a in tools.calls if n == "fetch_url"]
        assert len(fetched) == len(set(fetched)), "a page was fetched twice"
        last = c.get("/api/conversation/conv9").json()["messages"][-1]
        check = last["meta"]["web_check"]
        assert check["rounds"] == 2 and len(check["sources"]) == 4
        assert check["original"] == ANSWER and last["content"] == second["answer"]
        # Round 3: no new query left, so the question again; one unread page remains.
        _age_last_round(db, "conv9")
        client.reply = "NO CHANGES"
        third = _round(c, "conv9", message_id=mid)
        assert third["round"] == 3 and third["changed"] is False and not third["exhausted"]
        assert [s["url"] for s in third["sources"]] == ["https://cities.example.net/japan/tokyo"]
        # Round 4: every page reachable has been read. Nothing changes, and it says so.
        _age_last_round(db, "conv9")
        fourth = _round(c, "conv9", message_id=mid)
        assert fourth["round"] == 4 and fourth["exhausted"] is True and fourth["stored"] is True
        # The round cap ends the series; another message id is another answer.
        cfg.auto_refine_max_rounds = 4
        _age_last_round(db, "conv9")
        assert "AUTO_REFINE_MAX_ROUNDS" in _round(c, "conv9", message_id=mid)["skipped"]
        assert "moved on" in _round(c, "conv9", message_id=mid + 999)["skipped"]
        assert c.get("/api/conversation/conv9").json()["messages"][-1]["content"] \
            == second["answer"]


def test_turn_kind_keeps_a_check_out_of_the_running_turn_lookup():
    reg = TurnRegistry(Config())

    async def scenario():
        chat = reg.create("u1", "c1")
        check = reg.create("u1", "c1", kind="refine")
        return chat, check

    chat, check = asyncio.run(scenario())
    assert reg.active_for("u1", "c1") == [chat]
    assert reg.active_for("u1", "c1", kind="refine") == [check]
    assert check.snapshot()["kind"] == "refine"
    # The latest chat turn is the only one a check may be about.
    assert reg.latest_for("u1", "c1") is chat
    assert reg.latest_for("u1", "c1", kind="refine") is check
    assert reg.latest_for("u2", "c1") is None


# --------------------------------------------------------------------------- #
# The agent's page prefetch
# --------------------------------------------------------------------------- #
class _PrefetchAgent:
    def __init__(self, deny: set[str] = frozenset()):
        self.config = Config(search_workers=4)
        self.deny = set(deny)
        self.fetched: list[str] = []
        agent = self

        class Registry:
            def call(self, name, args, cid=None):
                time.sleep(0.1)
                agent.fetched.append(args["url"])
                return f"page {args['url']}", None
        self.registry = Registry()

    async def _ag_decide(self, name, args, turn):
        return ("ask" if args["url"] in self.deny else "allow"), "", "", None


def test_prefetch_starts_only_what_the_loop_still_needs_and_only_allowed_calls():
    async def scenario():
        agent = _PrefetchAgent(deny={"u2"})
        pre = _AgPrefetch(agent, "c1", ["u1", "u2", "u3", "u4"])
        started = time.time()
        await pre.top_up(2)
        assert set(pre._started) == {"u1", "u3"}, "u2 needs a person; it is left to dispatch"
        assert pre.runner("u2") is None
        first, second = pre.runner("u1"), pre.runner("u3")
        got = await asyncio.gather(first({"url": "u1"}), second({"url": "u3"}))
        took = time.time() - started
        # An approval that edited the address fetches what was approved.
        await pre.top_up(1)
        edited = pre.runner("u4")
        other = await edited({"url": "u9"})
        pre.close()
        return got, took, other, agent.fetched

    got, took, other, fetched = asyncio.run(scenario())
    assert got == [("page u1", None), ("page u3", None)]
    assert took < 0.19, f"two 0.1s fetches ran one after the other ({took:.2f}s)"
    assert other == ("page u9", None)
    assert "u2" not in fetched


def test_prefetch_close_drops_downloads_nobody_took():
    async def scenario():
        agent = _PrefetchAgent()
        agent.config = Config(search_workers=1)
        pre = _AgPrefetch(agent, "c1", ["u1", "u2", "u3"])
        await pre.top_up(3)
        tasks = list(pre._started.values())
        pre.close()
        await asyncio.sleep(0.25)
        return tasks, agent.fetched

    tasks, fetched = asyncio.run(scenario())
    assert all(t.cancelled() or t.done() for t in tasks)
    assert len(fetched) <= 1, "with one worker, only the fetch already running finishes"


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
