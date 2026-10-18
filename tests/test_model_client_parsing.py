"""Tests for ModelClient response intake: separated reasoning, native tool
calls, null content, cached-token usage, typed stream events and the request
passthrough, against a fake OpenAI-compatible server (httpx.MockTransport)."""
from __future__ import annotations
import asyncio, json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import local_llm.config as config_mod
config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-model-client-"))
from local_llm.config import Config
from local_llm.llm import GenerationStats, ThinkSplitter, parse_tool_call, parse_tool_calls
from local_llm.model_client import ModelClient

MESSAGES = [{"role": "user", "content": "hi"}]


def _chunk(delta: dict | None = None, finish: str | None = None, usage: dict | None = None) -> str:
    data: dict = {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if usage is not None:
        data = {"choices": [], "usage": usage}
    return "data: " + json.dumps(data) + "\n\n"


class FakeServer:
    """Answers /v1/chat/completions with a scripted SSE stream or JSON body,
    shaped like mlx_lm/server.py generate_response."""

    def __init__(self, stream_chunks: list[str] | None = None, message: dict | None = None,
                 usage: dict | None = None, finish: str = "stop"):
        self.stream_chunks = stream_chunks or []
        self.message = message or {"role": "assistant", "content": "ok"}
        self.usage = usage
        self.finish = finish
        self.bodies: list[dict] = []

    def handler(self, request):
        import httpx
        body = json.loads(request.content)
        self.bodies.append(body)
        if body.get("stream"):
            payload = "".join(self.stream_chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, content=payload.encode(),
                                  headers={"content-type": "text/event-stream"})
        data = {"choices": [{"index": 0, "message": self.message, "finish_reason": self.finish}]}
        if self.usage is not None:
            data["usage"] = self.usage
        return httpx.Response(200, json=data)


def _client(server: FakeServer, **cfg) -> ModelClient:
    import httpx
    client = ModelClient(Config(**cfg))
    mock = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
    client._client_for = lambda node: mock
    return client


async def _collect(agen) -> list:
    return [item async for item in agen]


def test_separated_reasoning_streams_as_think_tags_and_is_counted():
    for key in ("reasoning", "reasoning_content"):
        server = FakeServer([
            _chunk({"role": "assistant"}),
            _chunk({key: "Weighing "}), _chunk({key: "it up."}),
            _chunk({"content": "The answer "}), _chunk({"content": "is 42."}),
            _chunk(finish="stop"),
            _chunk(usage={"prompt_tokens": 50, "completion_tokens": 9,
                          "prompt_tokens_details": {"cached_tokens": 32}}),
        ])
        stats = GenerationStats()
        pieces = asyncio.run(_collect(_client(server).stream(MESSAGES, stats=stats)))
        text = "".join(pieces)
        assert text == "<think>Weighing it up.</think>The answer is 42.", key
        sp = ThinkSplitter()
        out = []
        for piece in pieces:
            out += sp.feed(piece)
        out += sp.flush()
        assert "".join(t for k, t in out if k == "think") == "Weighing it up."
        assert "".join(t for k, t in out if k == "answer") == "The answer is 42."
        assert stats.cached_tokens == 32 and stats.prompt_tokens == 50
        assert stats.reasoning_tokens > 0 and stats.finish_reason == "stop"
        assert stats.ttft_measured


def test_native_streamed_tool_calls_become_the_canonical_protocol():
    server = FakeServer([
        _chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                "function": {"name": "web_search", "arguments": ""}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"query": '}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"mlx"}'}}]}),
        _chunk({"content": None}),
        _chunk(finish="tool_calls"),
    ])
    text = "".join(asyncio.run(_collect(_client(server).stream(MESSAGES))))
    assert json.loads(text) == {"tool": "web_search", "args": {"query": "mlx"}}
    assert parse_tool_call(text, {"web_search"}) == ("web_search", {"query": "mlx"})


def test_complete_handles_null_content_tool_calls_and_cached_usage():
    # mlx_lm/server.py: content is None when only tool calls were produced,
    # reasoning and tool_calls sit beside it, arguments are a JSON string.
    server = FakeServer(message={
        "role": "assistant", "content": None, "reasoning": "Need to search.",
        "tool_calls": [
            {"id": "a", "type": "function",
             "function": {"name": "web_search", "arguments": '{"query": "mlx"}'}},
            {"id": "b", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city": "Ghent"}'}}]},
        usage={"prompt_tokens": 100, "completion_tokens": 20,
               "prompt_tokens_details": {"cached_tokens": 64},
               "completion_tokens_details": {"reasoning_tokens": 5}},
        finish="tool_calls")
    text, stats = asyncio.run(_client(server).complete_with_stats(MESSAGES))
    calls = parse_tool_calls(text, {"web_search", "get_weather"})
    assert [(c.name, c.args) for c in calls] == [("web_search", {"query": "mlx"}),
                                                  ("get_weather", {"city": "Ghent"})]
    # The non-streaming path never carried reasoning text, and still does not.
    assert "Need to search" not in text
    assert stats.cached_tokens == 64 and stats.reasoning_tokens == 5
    assert stats.finish_reason == "tool_calls"
    # No first-token time on a non-streaming call: the rate is flagged.
    assert stats.ttft_measured is False and stats.as_event()["decode_tps_estimated"] is True

    empty = FakeServer(message={"role": "assistant", "content": None})
    text, stats = asyncio.run(_client(empty).complete_with_stats(MESSAGES))
    assert text == "" and stats.cached_tokens == 0


def test_stream_events_are_typed_and_pass_native_calls_through():
    server = FakeServer([
        _chunk({"reasoning": "hmm"}),
        _chunk({"content": "Searching."}),
        _chunk({"tool_calls": [{"index": 0, "id": "c", "type": "function",
                                "function": {"name": "web_search", "arguments": '{"query": "x"}'}}]}),
        _chunk(finish="tool_calls"),
        _chunk(usage={"prompt_tokens": 5, "completion_tokens": 3}),
    ])
    events = asyncio.run(_collect(_client(server).stream_events(MESSAGES)))
    kinds = [e["type"] for e in events]
    assert kinds[:2] == ["reasoning", "content"]
    native = [e for e in events if e["type"] == "tool_call_delta"]
    assert native[0]["source"] == "native" and native[0]["done"] is False
    assert native[-1]["done"] is True and native[-1]["args"] == {"query": "x"}
    assert kinds.index("tool_call_delta") < kinds.index("finish") < kinds.index("usage")
    assert [e for e in events if e["type"] == "finish"][0]["reason"] == "tool_calls"


def test_stream_events_can_detect_prompt_protocol_calls():
    server = FakeServer([
        _chunk({"content": "Let me check.\n<tool_"}),
        _chunk({"content": 'call>{"name": "web_search", '}),
        _chunk({"content": '"arguments": {"query": "mlx"}}</tool_call>'}),
        _chunk(finish="stop"),
    ])
    events = asyncio.run(_collect(_client(server).stream_events(
        MESSAGES, extra={"parse_tool_calls": True, "known_tools": {"web_search"}})))
    shown = "".join(e["text"] for e in events if e["type"] == "content")
    assert shown == "Let me check.\n"
    done = [e for e in events if e["type"] == "tool_call_delta" and e["done"]]
    assert len(done) == 1 and done[0]["source"] == "prompt"
    assert (done[0]["name"], done[0]["args"], done[0]["format"]) == ("web_search", {"query": "mlx"}, "hermes")
    # The request itself never carried the client-side option.
    assert "parse_tool_calls" not in server.bodies[0] and "known_tools" not in server.bodies[0]


def test_payload_passthrough_is_a_whitelist_and_thinking_is_consistent():
    client = ModelClient(Config())
    base = client.payload(MESSAGES, True)
    assert "chat_template_kwargs" not in base and "enable_thinking" not in base
    body = client.payload(MESSAGES, False, extra={
        "tools": [{"type": "function", "function": {"name": "web_search"}}],
        "tool_choice": "auto", "model": "evil/other-model", "messages": [],
        "api_key": "x", "parse_tool_calls": True, "top_p": None})
    assert body["tools"][0]["function"]["name"] == "web_search" and body["tool_choice"] == "auto"
    assert body["model"] == base["model"] and body["messages"] == MESSAGES
    for key in ("api_key", "parse_tool_calls", "top_p"):
        assert key not in body
    off = ModelClient(Config(disable_thinking=True, reasoning_visible=False))
    default = off.payload(MESSAGES, False)
    assert default["enable_thinking"] is False
    assert default["chat_template_kwargs"] == {"enable_thinking": False}
    on = off.payload(MESSAGES, False, extra={"enable_thinking": True})
    assert on["enable_thinking"] is True and on["chat_template_kwargs"]["enable_thinking"] is True
    kw = ModelClient(Config()).payload(MESSAGES, False, extra={
        "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"}})
    assert kw["enable_thinking"] is False
    assert kw["chat_template_kwargs"] == {"enable_thinking": False, "reasoning_effort": "low"}


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
