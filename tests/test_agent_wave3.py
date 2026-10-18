"""Tests for the wave 3 agent loop: dispatch policy, interrupt, reasoning split,
streamed tool-call detection, multimodal parts and subagent capabilities."""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import local_llm.config as config_mod  # noqa: E402

config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-agent-wave3-"))

import local_llm.agent as agent_mod  # noqa: E402
from local_llm.agent import Agent  # noqa: E402
from local_llm.config import Config  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.llm import GenerationStats  # noqa: E402
from local_llm.core import estimate_tokens  # noqa: E402
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.turnctx import TurnContext, reset_current_turn, set_current_turn  # noqa: E402


class _ChunkClient:
    """A ModelClient stand-in: each reply is a string or a list of chunks.

    `on_token(i)` runs before chunk i of every streamed reply, so a test can
    set a cancel event mid-stream exactly where it wants.
    """

    def __init__(self, replies, on_token=None):
        self.replies = list(replies)
        self.prompts: list[list[dict]] = []
        self.on_token = on_token

    def _next(self):
        return self.replies.pop(0) if self.replies else "done."

    def stream(self, messages, *a, **kw):
        self.prompts.append(messages)
        reply = self._next()
        chunks = [reply] if isinstance(reply, str) else list(reply)
        hook = self.on_token

        async def gen():
            for i, chunk in enumerate(chunks):
                if hook is not None:
                    hook(i)
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


def _agent(replies, on_token=None, **overrides):
    base = dict(agent_enabled=True, knowledge_triage=False, fast_path=False,
                incremental_reasoning=False, chunk_large_prompts=False,
                agent_max_steps=3, agent_min_steps=1, resilient_retries=1,
                ready_wait_timeout=0.01, auto_iterate_rounds=0,
                skills_dir=tempfile.mkdtemp(prefix="llm-test-wave3-skills-"))
    base.update(overrides)
    cfg = Config(**base)
    db = Database(Path(tempfile.mkdtemp()) / "agent.db")
    reg = ToolRegistry(cfg, db)
    client = _ChunkClient(replies, on_token=on_token)
    return Agent(cfg, reg, client), cfg, db, reg, client


def _record_calls(reg, result="file contents here"):
    calls = []

    def call(name, args, cid=None):
        calls.append((name, dict(args)))
        return result, None

    reg.call = call
    return calls


async def _collect(gen):
    return [ev async for ev in gen]


def _drain(agent, message="explain how tcp handshakes work", turn=None, **kw):
    async def main():
        token = set_current_turn(turn) if turn is not None else None
        try:
            return await _collect(agent.run_iterating(message, [], "t", 256, None, **kw))
        finally:
            if token is not None:
                reset_current_turn(token)
    return asyncio.run(main())


def _of(events, kind):
    return [e for e in events if e.get("type") == kind]


def _prompt_text(client) -> str:
    out = []
    for prompt in client.prompts:
        for m in prompt:
            content = m.get("content")
            out.append(content if isinstance(content, str) else json.dumps(content))
    return "\n".join(out)


class _Decision:
    def __init__(self, action, risk="exec", reason="it runs code"):
        self.action, self.risk, self.reason, self.rule = action, risk, reason, "test"


class _Broker:
    def __init__(self, decision="allow", args=None):
        self.decision, self.args = decision, args
        self.requests: list[tuple] = []
        self.resolved: list[tuple] = []

    def request(self, turn, call_id, name, args, risk, reason):
        self.requests.append((turn.turn_id if turn else None, call_id, name, dict(args)))
        return f"ap{len(self.requests)}"

    async def wait(self, approval_id, timeout):
        await asyncio.sleep(0)
        return {"decision": self.decision, "scope": "once", "args": self.args}

    def resolve(self, approval_id, user_id, decision, scope, args=None):
        self.resolved.append((approval_id, decision))
        return True


def _with_policy(action, broker=None, raises=False):
    """Patch the agent's policy lookup; returns a restore function."""
    seen: list[tuple] = []

    class _Policy:
        BROKER = broker

        @staticmethod
        def decide(name, args, turn):
            seen.append((name, dict(args), turn))
            if raises:
                raise RuntimeError("policy exploded")
            return _Decision(action)

    previous = agent_mod._ag_policy
    agent_mod._ag_policy = lambda: _Policy
    return seen, (lambda: setattr(agent_mod, "_ag_policy", previous))


CALL = json.dumps({"tool": "read_file", "args": {"path": "a.txt"}})


# --------------------------------------------------------------------------- #
# Dispatch through the policy.
# --------------------------------------------------------------------------- #
def test_an_allowed_call_runs_and_every_event_carries_one_id():
    agent, cfg, db, reg, client = _agent([CALL, "The file says hello."])
    calls = _record_calls(reg)
    seen, restore = _with_policy("allow")
    try:
        events = _drain(agent)
    finally:
        restore()
    assert calls == [("read_file", {"path": "a.txt"})], calls
    assert seen and seen[0][0] == "read_file"
    start, call, result = _of(events, "tool_call_start"), _of(events, "tool_call"), \
        _of(events, "tool_result")
    assert len(start) == len(call) == len(result) == 1, [e["type"] for e in events]
    assert start[0]["id"] and start[0]["id"] == call[0]["id"] == result[0]["id"]
    assert start[0]["name"] == "read_file"
    assert not _of(events, "approval_required")
    assert _of(events, "final")[-1]["answer"] == "The file says hello."
    db.close()


def test_ask_then_allow_waits_for_the_person_and_runs_the_edited_call():
    agent, cfg, db, reg, client = _agent([CALL, CALL, "Done reading b.txt."])
    calls = _record_calls(reg)
    broker = _Broker("allow", args={"path": "b.txt"})
    _, restore = _with_policy("ask", broker)
    turn = TurnContext(user_id="u1", conversation_id="t", interactive=True,
                       cancel=asyncio.Event())
    try:
        events = _drain(agent, turn=turn)
    finally:
        restore()
    asked, resolved = _of(events, "approval_required"), _of(events, "approval_resolved")
    # Two: the model repeats its ORIGINAL call after the edited one ran.
    assert len(asked) == 2 and len(resolved) == 2, [e["type"] for e in events]
    call_id = _of(events, "tool_call")[0]["id"]
    assert asked[0]["call_id"] == call_id and asked[0]["id"] == "ap1"
    assert asked[0]["scopes"] == ["once", "conversation", "tool"]
    assert asked[0]["risk"] == "exec" and asked[0]["expires_in"] > 0
    assert resolved[0]["decision"] == "allow" and resolved[0]["by"] == "u1"
    assert broker.requests[0][0] == turn.turn_id
    # The edited arguments are what ran, and what the trace reports.
    assert calls[0] == ("read_file", {"path": "b.txt"}), calls
    edited = [e for e in _of(events, "tool_call") if e.get("edited")]
    assert edited and edited[0]["args"] == {"path": "b.txt"}
    # The model's repeat of its ORIGINAL call is not the duplicate (the edited
    # one is), so it is asked about again rather than silently swallowed.
    assert len(broker.requests) == 2, broker.requests
    final = _of(events, "final")[-1]
    assert final["trace"][0]["args"] == {"path": "b.txt"}
    assert final["turn_id"] == turn.turn_id
    db.close()


def test_ask_then_deny_does_not_run_and_lets_the_model_replan():
    agent, cfg, db, reg, client = _agent([CALL, CALL, "I cannot read it, sorry."])
    calls = _record_calls(reg)
    broker = _Broker("deny")
    _, restore = _with_policy("ask", broker)
    turn = TurnContext(user_id="u1", interactive=True, cancel=asyncio.Event())
    try:
        events = _drain(agent, turn=turn)
    finally:
        restore()
    assert calls == [], "a denied call reached the registry"
    assert _of(events, "approval_resolved")[0]["decision"] == "deny"
    results = _of(events, "tool_result")
    assert results and all("denied by the user" in r["error"] for r in results)
    assert "TOOL RESULT [read_file]: denied by the user" in _prompt_text(client)
    # The identical retry is answered from the refusal, not asked twice.
    assert len(broker.requests) == 1, broker.requests
    assert _of(events, "final")[-1]["answer"] == "I cannot read it, sorry."
    db.close()


def test_a_policy_deny_never_asks_and_never_runs():
    agent, cfg, db, reg, client = _agent([CALL, "Answered without the file."])
    calls = _record_calls(reg)
    broker = _Broker("allow")
    _, restore = _with_policy("deny", broker)
    try:
        events = _drain(agent)
    finally:
        restore()
    assert calls == [] and broker.requests == []
    assert not _of(events, "approval_required")
    assert "denied by policy" in _of(events, "tool_result")[0]["error"]
    assert "denied by policy" in _prompt_text(client)
    db.close()


def test_ask_with_nobody_watching_takes_the_unattended_decision():
    # No turn bound at all (CLI, selftest): nobody can answer, so deny.
    agent, cfg, db, reg, client = _agent([CALL, "Fine."])
    calls = _record_calls(reg)
    broker = _Broker("allow")
    _, restore = _with_policy("ask", broker)
    try:
        events = _drain(agent)
        assert calls == [] and broker.requests == []
        assert not _of(events, "approval_required")
        # A scheduled task (interactive=False) with TOOL_APPROVAL_UNATTENDED=allow.
        agent2, cfg2, db2, reg2, _ = _agent([CALL, "Fine."])
        # A per-test Config instance; the field may not exist yet (the policy
        # feature adds it), and the agent reads it with getattr.
        cfg2.tool_approval_unattended = "allow"
        calls2 = _record_calls(reg2)
        _drain(agent2, turn=TurnContext(interactive=False))
        assert calls2 == [("read_file", {"path": "a.txt"})], calls2
        db2.close()
    finally:
        restore()
    db.close()


def test_a_missing_or_broken_policy_allows_as_before():
    agent, cfg, db, reg, client = _agent([CALL, "ok."])
    calls = _record_calls(reg)
    _, restore = _with_policy("deny", raises=True)
    try:
        _drain(agent)
    finally:
        restore()
    assert calls == [("read_file", {"path": "a.txt"})]
    previous = agent_mod._ag_policy
    agent_mod._ag_policy = lambda: None
    try:
        agent2, _, db2, reg2, _ = _agent([CALL, "ok."])
        calls2 = _record_calls(reg2)
        _drain(agent2)
        assert calls2 == [("read_file", {"path": "a.txt"})]
        db2.close()
    finally:
        agent_mod._ag_policy = previous
    db.close()


def test_stop_during_an_approval_wait_denies_and_ends_the_turn():
    agent, cfg, db, reg, client = _agent([CALL, "never reached."])
    calls = _record_calls(reg)

    class _SlowBroker(_Broker):
        async def wait(self, approval_id, timeout):
            await asyncio.sleep(30)

    broker = _SlowBroker()
    _, restore = _with_policy("ask", broker)
    turn = TurnContext(user_id="u1", interactive=True, cancel=asyncio.Event())

    async def main():
        token = set_current_turn(turn)
        out = []
        try:
            async for ev in agent.run_iterating("explain tcp", [], "t", 256, None):
                out.append(ev)
                if ev.get("type") == "approval_required":
                    turn.cancel.set()
        finally:
            reset_current_turn(token)
        return out

    try:
        events = asyncio.run(asyncio.wait_for(main(), timeout=10))
    finally:
        restore()
    assert calls == []
    assert _of(events, "approval_resolved")[0]["decision"] == "deny"
    assert broker.resolved == [("ap1", "deny")]
    assert _of(events, "cancelled") and not _of(events, "final")
    db.close()


def test_the_real_policy_round_trip_when_it_is_installed():
    """Against policy.py itself (skipped on a build without it): approval mode
    "all" asks, the owner allows, and the registry's own synchronous check
    accepts the one-shot grant the approval left behind."""
    try:
        import local_llm.policy as policy
    except ImportError:
        return
    calc = json.dumps({"tool": "calculator", "args": {"expression": "2+2"}})
    agent, cfg, db, reg, client = _agent([calc, "It is 4."], tool_approval_mode="all")
    turn = TurnContext(user_id="u1", conversation_id="t", interactive=True,
                       cancel=asyncio.Event())

    async def main():
        token = set_current_turn(turn)
        out = []
        try:
            async for ev in agent.run_iterating("what is 2+2, use the calculator", [],
                                                "t", 256, None):
                out.append(ev)
                if ev.get("type") == "approval_required":
                    assert policy.BROKER.resolve(ev["id"], "u1", "allow", "once")
        finally:
            reset_current_turn(token)
        return out

    events = asyncio.run(asyncio.wait_for(main(), timeout=20))
    assert _of(events, "approval_required"), [e["type"] for e in events]
    assert _of(events, "approval_resolved")[0]["decision"] == "allow"
    result = _of(events, "tool_result")[0]
    assert result["error"] is None and "4" in result["result"], result
    db.close()


# --------------------------------------------------------------------------- #
# Interrupt.
# --------------------------------------------------------------------------- #
def test_the_turn_cancel_event_stops_the_stream_with_a_partial():
    words = ["TCP ", "opens ", "with ", "a ", "SYN ", "then ", "more ", "text."]
    turn = TurnContext(interactive=True, cancel=asyncio.Event())

    def hook(i):
        if i == 3:
            turn.cancel.set()

    agent, cfg, db, reg, client = _agent([words], on_token=hook)
    events = _drain(agent, turn=turn)          # no cancel argument: the turn's is used
    cancelled = _of(events, "cancelled")
    assert cancelled, [e["type"] for e in events]
    assert cancelled[0]["partial"] == "TCP opens with", cancelled[0]
    assert not _of(events, "final"), "a cancelled turn still produced a final"
    db.close()


def test_an_explicit_cancel_reaches_the_continuation_lane():
    cancel = asyncio.Event()

    def hook(i):
        if i == 2:
            cancel.set()

    agent, cfg, db, reg, client = _agent([["more ", "of ", "the ", "answer"]], on_token=hook)

    async def main():
        return await _collect(agent.continue_answer(("write it", "Part one, "), [], 256,
                                                    None, cancel, "t"))
    events = asyncio.run(main())
    stopped = _of(events, "cancelled")
    assert stopped and stopped[0]["partial"] == "more of ", stopped
    db.close()


# --------------------------------------------------------------------------- #
# Reasoning.
# --------------------------------------------------------------------------- #
def test_reasoning_never_reaches_token_events_and_lands_in_final():
    reply = ["<thi", "nk>secret ", "plan</th", "ink>The hand", "shake is SYN."]
    agent, cfg, db, reg, client = _agent([reply])
    events = _drain(agent)
    tokens = "".join(e["token"] for e in _of(events, "token"))
    thinking = "".join(e["token"] for e in _of(events, "think_token"))
    assert "secret" not in tokens and "think" not in tokens, tokens
    assert tokens == "The handshake is SYN.", tokens
    assert thinking == "secret plan", thinking
    final = _of(events, "final")[-1]
    assert final["reasoning"] == "secret plan" and "secret" not in final["answer"]
    db.close()


def test_a_turn_without_reasoning_reports_an_empty_string():
    agent, cfg, db, reg, client = _agent(["Plain answer."])
    final = _of(_drain(agent), "final")[-1]
    assert final["reasoning"] == "" and "turn_id" not in final
    db.close()


def test_the_continuation_lane_splits_reasoning_too():
    agent, cfg, db, reg, client = _agent([["<think>where was ", "I</think>", " and the end."]])

    async def main():
        return await _collect(agent.continue_answer(("write it", "Part one"), [], 256,
                                                    None, None, "t"))
    events = asyncio.run(main())
    tokens = "".join(e["token"] for e in _of(events, "token"))
    assert "where was" not in tokens and tokens == " and the end.", tokens
    final = _of(events, "final")[-1]
    assert final["reasoning"] == "where was I"
    assert "where was" not in final["answer"]
    db.close()


# --------------------------------------------------------------------------- #
# Streamed tool-call detection.
# --------------------------------------------------------------------------- #
def test_a_streamed_call_is_announced_progressively_and_never_shown_as_text():
    chunks = ['{"to', 'ol": "read', '_file", "args": {"pa', 'th": "a.txt"}}']
    agent, cfg, db, reg, client = _agent([chunks, "It says hello."])
    _record_calls(reg)
    events = _drain(agent)
    step1_tokens = [e["token"] for e in _of(events, "token") if e["step"] == 1]
    assert step1_tokens == [], f"call JSON leaked into the answer: {step1_tokens}"
    start = _of(events, "tool_call_start")
    deltas = _of(events, "tool_call_delta")
    assert len(start) == 1 and start[0]["name"] == "read_file"
    assert deltas and {d["id"] for d in deltas} == {start[0]["id"]}
    raw = "".join(d["args_delta"] for d in deltas)
    assert "a.txt" in raw, raw
    # tool_call_start comes before its deltas, and the dispatch reuses the id.
    order = [e["type"] for e in events]
    assert order.index("tool_call_start") < order.index("tool_call_delta")
    assert _of(events, "tool_call")[0]["id"] == start[0]["id"]
    assert _of(events, "tool_result")[0]["id"] == start[0]["id"]
    assert len(_of(events, "tool_call_start")) == 1, "announced twice"
    db.close()


def test_a_final_answer_call_streams_its_answer_as_text_not_as_a_call():
    chunks = ['{"tool": "final_answer", "args": {"answer": "Hel',
              'lo \\"world\\"\\nbye"}}']
    agent, cfg, db, reg, client = _agent([chunks])
    events = _drain(agent)
    assert not _of(events, "tool_call_start") and not _of(events, "tool_call")
    tokens = "".join(e["token"] for e in _of(events, "token"))
    assert tokens == 'Hello "world"\nbye', repr(tokens)
    assert _of(events, "final")[-1]["answer"] == 'Hello "world"\nbye'
    db.close()


def test_prose_that_is_not_a_call_still_streams_as_tokens():
    reply = ['Use ', '{"key": 1} ', 'in the config.']
    agent, cfg, db, reg, client = _agent([reply])
    events = _drain(agent)
    tokens = "".join(e["token"] for e in _of(events, "token"))
    assert tokens == 'Use {"key": 1} in the config.', tokens
    assert not _of(events, "tool_call_start")
    db.close()


def test_an_unregistered_streamed_call_is_never_announced():
    bogus = json.dumps({"tool": "teleport", "args": {"to": "mars"}})
    agent, cfg, db, reg, client = _agent([bogus, "Plain answer."])
    calls = _record_calls(reg)
    events = _drain(agent)
    assert calls == []
    assert not _of(events, "tool_call_start"), _of(events, "tool_call_start")
    db.close()


# --------------------------------------------------------------------------- #
# Multimodal content parts.
# --------------------------------------------------------------------------- #
IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}


def test_images_reach_the_user_message_when_the_model_can_see():
    agent, cfg, db, reg, client = _agent(["A red square."])
    events = _drain(agent, "what is in this picture?",
                    attachments=[IMAGE, {"type": "text", "text": "caption: test"}],
                    vision=True)
    prompt = client.prompts[-1]
    parts = [m for m in prompt if isinstance(m.get("content"), list)]
    assert len(parts) == 1 and parts[0]["role"] == "user", prompt
    content = parts[0]["content"]
    assert content[0]["type"] == "text" and "what is in this picture?" in content[0]["text"]
    assert IMAGE in content
    assert "caption: test" in content[0]["text"]
    assert _of(events, "final")[-1]["answer"] == "A red square."
    db.close()


def test_images_on_a_text_only_model_are_declined_out_loud():
    agent, cfg, db, reg, client = _agent(["I cannot see images."])
    events = _drain(agent, "what is in this picture?", attachments=[IMAGE], vision=False)
    notices = " ".join(e["message"] for e in _of(events, "notice"))
    assert "cannot see images" in notices, notices
    assert not any(isinstance(m.get("content"), list) for p in client.prompts for m in p)
    assert "cannot see images" in _prompt_text(client)
    db.close()


def test_the_vision_check_needs_both_the_backend_and_the_model():
    agent, cfg, db, reg, client = _agent([])
    # The configured default is a text model: not vision, whatever the backend.
    assert agent.vision_capable() is False
    turn = TurnContext(extras={"vision": True})
    token = set_current_turn(turn)
    try:
        assert agent.vision_capable() is True      # the route knows best
    finally:
        reset_current_turn(token)
    db.close()


# --------------------------------------------------------------------------- #
# Subagent capabilities.
# --------------------------------------------------------------------------- #
def test_a_subagent_cannot_get_capabilities_its_parent_lacks():
    agent, cfg, db, reg, client = _agent(["done."], agent_tools="web_search,fetch_url")
    assert "read_file" not in reg.names() and "web_search" in reg.names()
    captured = {}
    real = Agent.run

    async def spy(self, message, history, **kw):
        if self is not agent:
            captured.setdefault("tools", sorted(self.registry.names()))
        async for ev in real(self, message, history, **kw):
            yield ev

    Agent.run = spy
    try:
        asyncio.run(agent.run_subagent("read the secrets", capabilities="file_ops,code_exec"))
        escalated = captured.pop("tools")
        asyncio.run(agent.run_subagent("look it up", capabilities="web_api"))
        granted = captured.pop("tools")
    finally:
        Agent.run = real
    for name in ("read_file", "write_file", "run_shell", "run_python"):
        assert name not in escalated, escalated
    assert "web_search" in granted and "read_file" not in granted, granted
    assert "delegate_task" not in granted
    db.close()


def test_delegation_goes_through_the_policy_gate():
    call = json.dumps({"tool": "delegate_task",
                       "args": {"task": "summarise the repo", "capabilities": "file_ops"}})
    agent, cfg, db, reg, client = _agent([call, "Parent answer."], delegation_enabled=True)
    spawned = []

    async def fake_subagent(task, caps="", parent=None):
        spawned.append(task)
        return "helper result"

    agent.run_subagent = fake_subagent
    _, restore = _with_policy("deny")
    try:
        events = _drain(agent)
    finally:
        restore()
    assert spawned == [], "a denied delegation still spawned a helper"
    assert "denied by policy" in _of(events, "tool_result")[0]["error"]
    db.close()


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
