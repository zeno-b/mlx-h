"""Tests for multi-format tool-call parsing, JSON repair, streaming detection,
reasoning splitting and the token math they feed (toolparse.py, llm.py, core.py)."""
from __future__ import annotations
import sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import local_llm.config as config_mod
config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-toolparse-"))
from local_llm.config import Config
from local_llm.core import IMAGE_TOKEN_ESTIMATE, estimate_tokens, messages_tokens, trim_to_context
from local_llm.llm import (GenerationStats, ThinkSplitter, is_degenerate_tool_call,
                           is_truncated_tool_call, parse_tool_call, strip_reasoning)
from local_llm.toolparse import (ToolCallStreamDetector, bind_tool_call_config,
                                 looks_like_call_prefix, parse_tool_calls, repair_json)
from local_llm.settings_schema import SETTINGS_FIELDS

KNOWN = {"web_search", "get_weather", "calculator", "read_file"}

# (label, text, expected [(name, args)], expected format of the first call)
FORMATS = [
    ("canonical", '{"tool": "web_search", "args": {"query": "mlx"}}',
     [("web_search", {"query": "mlx"})], "json"),
    ("name/parameters", '{"name": "web_search", "parameters": {"query": "mlx"}}',
     [("web_search", {"query": "mlx"})], "json"),
    ("string args", '{"tool": "web_search", "args": "{\\"query\\": \\"mlx\\"}"}',
     [("web_search", {"query": "mlx"})], "json"),
    ("openai function",
     '{"type": "function", "function": {"name": "web_search", "arguments": "{\\"query\\": \\"mlx\\"}"}}',
     [("web_search", {"query": "mlx"})], "openai"),
    ("openai tool_calls",
     '{"tool_calls": [{"id": "c1", "function": {"name": "web_search", "arguments": "{\\"query\\": \\"a\\"}"}},'
     ' {"function": {"name": "calculator", "arguments": {"expression": "1+1"}}}]}',
     [("web_search", {"query": "a"}), ("calculator", {"expression": "1+1"})], "openai"),
    ("function_call",
     '{"function_call": {"name": "web_search", "arguments": "{\\"query\\": \\"mlx\\"}"}}',
     [("web_search", {"query": "mlx"})], "openai"),
    ("hermes", 'Let me look.\n<tool_call>\n{"name": "web_search", "arguments": {"query": "mlx"}}\n</tool_call>',
     [("web_search", {"query": "mlx"})], "hermes"),
    ("hermes x2",
     '<tool_call>{"name": "web_search", "arguments": {"query": "a"}}</tool_call>\n'
     '<tool_call>{"name": "get_weather", "arguments": {"city": "Ghent"}}</tool_call>',
     [("web_search", {"query": "a"}), ("get_weather", {"city": "Ghent"})], "hermes"),
    ("qwen3-coder",
     "<tool_call>\n<function=get_weather>\n<parameter=city>\nGhent\n</parameter>\n</function>\n</tool_call>",
     [("get_weather", {"city": "Ghent"})], "qwen3_coder"),
    ("llama python_tag", '<|python_tag|>{"name": "web_search", "parameters": {"query": "mlx"}}<|eom_id|>',
     [("web_search", {"query": "mlx"})], "llama3"),
    ("mistral array", '[TOOL_CALLS] [{"name": "web_search", "arguments": {"query": "mlx"}}]',
     [("web_search", {"query": "mlx"})], "mistral"),
    ("mistral [ARGS]", '[TOOL_CALLS]web_search[ARGS]{"query": "mlx"}',
     [("web_search", {"query": "mlx"})], "mistral"),
    ("pythonic", '[web_search(query="mlx"), get_weather(city="Ghent", days=3)]',
     [("web_search", {"query": "mlx"}), ("get_weather", {"city": "Ghent", "days": 3})], "pythonic"),
    ("pythonic tokens", '<|tool_call_start|>[web_search(query="mlx")]<|tool_call_end|>',
     [("web_search", {"query": "mlx"})], "pythonic"),
    ("gemma tool_code", '```tool_code\nweb_search(query="mlx")\n```',
     [("web_search", {"query": "mlx"})], "gemma"),
    ("function gemma",
     "<start_function_call>call:get_weather{city:<escape>Ghent<escape>}<end_function_call>",
     [("get_weather", {"city": "Ghent"})], "gemma"),
    ("deepseek v3",
     "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>web_search\n"
     '```json\n{"query": "mlx"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
     [("web_search", {"query": "mlx"})], "deepseek"),
    ("kimi k2",
     "<|tool_calls_section_begin|><|tool_call_begin|>functions.web_search:0"
     '<|tool_call_argument_begin|>{"query": "mlx"}<|tool_call_end|><|tool_calls_section_end|>',
     [("web_search", {"query": "mlx"})], "kimi"),
    ("glm-4.5", "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Ghent</arg_value>\n</tool_call>",
     [("get_weather", {"city": "Ghent"})], "glm"),
    ("gpt-oss harmony",
     "<|channel|>analysis<|message|>Need search.<|end|><|start|>assistant<|channel|>commentary "
     'to=functions.web_search <|constrain|>json<|message|>{"query": "mlx"}<|call|>',
     [("web_search", {"query": "mlx"})], "harmony"),
]


def test_every_model_family_format_parses():
    for label, text, expected, fmt in FORMATS:
        calls = parse_tool_calls(text, KNOWN, final=True)
        assert [(c.name, c.args) for c in calls] == expected, label
        assert calls[0].format == fmt, (label, calls[0].format)
        assert not calls[0].recovered, label


def test_parse_tool_call_returns_the_first_call_as_before():
    for label, text, expected, _fmt in FORMATS:
        assert parse_tool_call(text, KNOWN) == expected[0], label


# (text, repair tag that must be recorded, expected args)
REPAIRS = [
    ("{'tool': 'web_search', 'args': {'query': 'mlx',}}", "single_quotes", {"query": "mlx"}),
    ("{'tool': 'web_search', 'args': {'query': 'mlx',}}", "trailing_commas", {"query": "mlx"}),
    ('{“tool”: “web_search”, “args”: {“query”: “mlx”}}', "smart_quotes", {"query": "mlx"}),
    ('{"tool": "calculator", "args": {"expression": "1+1", "exact": True, "x": None}}',
     "python_literals", {"expression": "1+1", "exact": True, "x": None}),
    ('{tool: "web_search", args: {query: "mlx"}}', "unquoted_keys", {"query": "mlx"}),
    ('{"tool": "read_file", "args": {"text": "a\nb"}}', "raw_control_chars", {"text": "a\nb"}),
    ('{"tool": "web_search", "args": {"query": "mlx"', "unclosed_brackets", {"query": "mlx"}),
    ('{"tool": "functions.web_search()", "args": {"query": "mlx"}}', "name_noise", {"query": "mlx"}),
    ('{"tool": "web​_search", "args": {"query": "mlx"}}', "zero_width", {"query": "mlx"}),
]


def test_broken_4bit_json_is_recovered_and_every_repair_is_recorded():
    for text, tag, args in REPAIRS:
        calls = parse_tool_calls(text, KNOWN, final=True)
        assert calls, text
        call = calls[0]
        assert call.name == "web_search" or call.name in KNOWN, text
        assert call.args == args, text
        assert call.recovered and tag in call.repairs, (text, call.repairs)


def test_repair_can_be_switched_off_and_a_truncated_call_waits_for_the_end():
    broken = "{'tool': 'web_search', 'args': {'query': 'mlx',}}"
    assert parse_tool_calls(broken, KNOWN, repair=False) == []
    cut = '{"tool": "web_search", "args": {"query": "mlx"'
    # Mid-stream the arguments are still arriving: never close them early.
    assert parse_tool_calls(cut, KNOWN) == []
    assert parse_tool_calls(cut, KNOWN, final=True)[0].args == {"query": "mlx"}
    # A value cut off inside a string is never completed: "ml" is not "mlx",
    # and half a file body must not be written as though it were the file.
    assert parse_tool_calls('{"tool": "web_search", "args": {"query": "ml', KNOWN, final=True) == []
    value, repairs = repair_json("{'a': 1,}")
    assert value == {"a": 1} and "single_quotes" in repairs and "trailing_commas" in repairs


def test_the_live_config_decides_repair_and_formats():
    cfg = Config(tool_call_repair=False)
    try:
        bind_tool_call_config(cfg)
        assert parse_tool_calls("{'tool': 'web_search', 'args': {}}", KNOWN) == []
        cfg.apply({"tool_call_repair": True})
        assert parse_tool_calls("{'tool': 'web_search', 'args': {}}", KNOWN)
        q3 = "<function=get_weather><parameter=city>Ghent</parameter></function>"
        assert parse_tool_calls(q3, KNOWN)
        cfg.apply({"tool_call_format": "json"})
        assert parse_tool_calls(q3, KNOWN) == []
        # The canonical protocol can never be switched off.
        assert parse_tool_calls('{"tool": "web_search", "args": {}}', KNOWN)
    finally:
        bind_tool_call_config(None)


def test_prose_stays_prose_and_names_are_never_invented():
    for text in ('Here is a config example: {"name": "my-app", "version": 2}',
                 "Use [x(1)] in your code.",
                 "[1, 2, 3] is a list",
                 "The answer is 42.",
                 '{"name": "launch_rockets", "arguments": {}}',          # unknown, implicit
                 '<tool_call>{"arguments": {"query": "mlx"}}</tool_call>',  # nameless
                 '{"tool": "", "args": {}}'):
        assert parse_tool_calls(text, KNOWN, final=True) == [], text
        assert parse_tool_call(text, KNOWN) is None, text
    # The explicit "tool" key still names a tool the registry will reject:
    # the historical rule, so the model is told the tool does not exist.
    assert parse_tool_call('{"tool": "launch_rockets", "args": {}}', KNOWN) == ("launch_rockets", {})
    # A near miss is not "corrected" into a real tool.
    assert parse_tool_call('{"tool": "websearch", "args": {}}', KNOWN) == ("websearch", {})


def test_reasoning_is_never_parsed_as_the_call():
    text = '<think>maybe {"tool": "calculator", "args": {}}</think>{"tool": "web_search", "args": {"query": "a"}}'
    assert parse_tool_call(text, KNOWN) == ("web_search", {"query": "a"})
    assert parse_tool_call('<think>{"tool": "calculator", "args": {}}', KNOWN) is None


def test_truncated_and_degenerate_checks_know_the_new_formats():
    assert is_truncated_tool_call('{"tool": "web_search", "args": {"query": "x', KNOWN)
    assert is_truncated_tool_call('<tool_call>\n<function=get_weather>\n<parameter=city>\nGh', KNOWN)
    assert not is_truncated_tool_call("Plain answer.", KNOWN)
    assert not is_truncated_tool_call('{"tool": "web_search", "args": {}}', KNOWN)
    assert is_degenerate_tool_call("<tool_call></tool_call>")
    assert is_degenerate_tool_call("[TOOL_CALLS][]")
    assert not is_degenerate_tool_call("The answer is 42.")


def _detect(text: str, step: int, known=KNOWN):
    det = ToolCallStreamDetector(known)
    events = []
    for i in range(0, len(text), step):
        events += det.feed(text[i:i + step])
    events += det.flush()
    shown = "".join(e["text"] for e in events if e["type"] == "text")
    calls = [(e["name"], e["args"]) for e in events if e["type"] == "call_end"]
    aborted = [e["raw"] for e in events if e["type"] == "call_abort"]
    return shown, calls, aborted, events


def test_stream_detector_finds_calls_at_any_chunking():
    cases = [
        ('{"tool": "web_search", "args": {"query": "a"}}', "", [("web_search", {"query": "a"})]),
        ("Let me check.\n<tool_call>\n{\"name\": \"web_search\", \"arguments\": {\"query\": \"m\"}}\n</tool_call>",
         "Let me check.\n", [("web_search", {"query": "m"})]),
        ("<tool_call>\n<function=get_weather>\n<parameter=city>\nGhent\n</parameter>\n</function>\n</tool_call>",
         "", [("get_weather", {"city": "Ghent"})]),
        ('[TOOL_CALLS]web_search[ARGS]{"query": "mlx"}', "", [("web_search", {"query": "mlx"})]),
        ('[web_search(query="mlx")]', "", [("web_search", {"query": "mlx"})]),
        ('<|python_tag|>{"name": "web_search", "parameters": {"query": "q"}}<|eom_id|>', "",
         [("web_search", {"query": "q"})]),
        ("Hello world, no call here.", "Hello world, no call here.", []),
        ("[1, 2, 3] is a list", "[1, 2, 3] is a list", []),
    ]
    for text, shown_expected, calls_expected in cases:
        for step in (1, 2, 3, 7, len(text)):
            shown, calls, aborted, events = _detect(text, step)
            assert shown == shown_expected, (text, step, shown)
            assert calls == calls_expected, (text, step)
            assert not aborted, (text, step)
            if calls:
                kinds = [e["type"] for e in events]
                assert kinds.index("call_start") < kinds.index("call_end")
                assert any(e["type"] == "args_delta" for e in events)


def test_stream_detector_releases_json_that_was_not_a_call():
    shown, calls, aborted, _ = _detect('{"name": "my-app", "version": 2} is package.json', 3)
    assert calls == [] and aborted == ['{"name": "my-app", "version": 2}']
    assert shown == " is package.json"
    # A call after prose is still found, but only at flush and marked late,
    # because its text has already been shown.
    _s, calls, _a, events = _detect('Sure!\n```json\n{"tool": "web_search", "args": {"query": "z"}}\n```', 4)
    assert calls == [("web_search", {"query": "z"})]
    assert [e for e in events if e["type"] == "call_end"][0]["late"] is True


def test_stream_detector_is_linear_on_a_long_call():
    body = "x" * 120_000
    text = '<tool_call>{"name": "read_file", "arguments": {"text": "' + body + '"}}</tool_call>'
    started = time.perf_counter()
    _s, calls, _a, _e = _detect(text, 4)
    elapsed = time.perf_counter() - started
    assert calls and calls[0][1]["text"] == body
    assert elapsed < 10.0, elapsed


def test_looks_like_call_prefix():
    for text in ("<tool", "<tool_call>", '{"to', "[TOOL_C", "<|python_t"):
        assert looks_like_call_prefix(text), text
    for text in ("Hello", "The answer", "42"):
        assert not looks_like_call_prefix(text), text


def _split(text: str, step: int, **kw):
    sp = ThinkSplitter(**kw)
    out = []
    for i in range(0, len(text), step):
        out += sp.feed(text[i:i + step])
    out += sp.flush()
    return ("".join(t for k, t in out if k == "think"),
            "".join(t for k, t in out if k == "answer"))


def test_splitter_and_strip_reasoning_know_seed_and_harmony():
    harmony = ("<|channel|>analysis<|message|>Think hard.<|end|><|start|>assistant"
               "<|channel|>final<|message|>Final answer<|return|>")
    for text, think, answer in (("<seed:think>plan</seed:think>Answer", "plan", "Answer"),
                                (harmony, "Think hard.", "Final answer")):
        for step in (1, 3, 5, len(text)):
            assert _split(text, step) == (think, answer), (text, step)
        assert strip_reasoning(text) == answer


def test_a_bare_leading_close_tag_is_reasoning():
    text = "weighing it up</think>The answer is 42."
    assert strip_reasoning(text) == "The answer is 42."
    # A splitter told the template opened <think> in the prompt agrees exactly.
    for step in (1, 4, len(text)):
        assert _split(text, step, opened=True) == ("weighing it up", "The answer is 42.")
    # Without that, the bare close tag itself is never shown.
    assert "</think>" not in _split(text, 3)[1]
    # Only the FIRST tag can be a bare close: a later stray one is text.
    assert _split("<think>a</think>b </think> c", 2) == ("a", "b </think> c")
    assert strip_reasoning("<think>a</think>b </think> c") == "b </think> c"


def test_generation_stats_do_not_invent_a_decode_rate():
    stats = GenerationStats(prompt_tokens=10, completion_tokens=100, ttft_ms=2000.0,
                            total_ms=2000.0, from_server=True)
    stats.ttft_measured = False
    # Old formula: 100 tokens / max(1 ms, 0 ms) = 100000 tok/s.
    assert round(stats.decode_tps) == 50
    event = stats.as_event()
    assert event["decode_tps_estimated"] is True
    for key in ("cached_tokens", "reasoning_tokens", "queue_ms"):
        assert key in event
    streamed = GenerationStats(completion_tokens=100, ttft_ms=1000.0, total_ms=2000.0)
    assert round(streamed.decode_tps) == 100 and streamed.as_event()["decode_tps_estimated"] is False


def test_token_math_counts_images_as_images():
    data_url = "data:image/png;base64," + "A" * 400_000
    parts = [{"type": "text", "text": "what is in this picture?"},
             {"type": "image_url", "image_url": {"url": data_url}}]
    cost = estimate_tokens(parts)
    assert cost == IMAGE_TOKEN_ESTIMATE + estimate_tokens("what is in this picture?")
    assert messages_tokens([{"role": "user", "content": parts}]) == cost + 4
    assert messages_tokens([{"role": "assistant", "content": None}]) == 4
    assert estimate_tokens("abcd" * 10) == 10
    system = {"role": "system", "content": "sys"}
    history = [{"role": "user", "content": parts}, {"role": "assistant", "content": "a cat"}]
    sent, dropped = trim_to_context(system, history, {"role": "user", "content": "and now?"},
                                    context_size=8192, reserve=512)
    # One screenshot no longer looks like 100k tokens and evicts the history.
    assert dropped == 0 and sent[1]["content"] == parts


def test_tool_call_settings_are_mutable_clamped_and_described():
    for key in ("tool_call_format", "tool_call_repair"):
        assert key in Config.MUTABLE
        assert any(f["key"] == key and f["section"] == "tools" for f in SETTINGS_FIELDS), key
    cfg = Config()
    assert cfg.tool_call_format == "auto" and cfg.tool_call_repair is True
    cfg.apply({"tool_call_format": "Hermes, nonsense ,qwen3_coder"})
    assert cfg.tool_call_format == "hermes,qwen3_coder"
    cfg.apply({"tool_call_format": "bogus"})
    assert cfg.tool_call_format == "auto"


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
