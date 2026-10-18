"""Tool-call parsing for every format a local model actually emits.

The agent's prompt asks for one protocol, {"tool": NAME, "args": {...}}, and a
3B model mostly writes it. Mostly. Swap in a Qwen3-Coder, a Llama 3.2, a
Mistral or a gpt-oss and the model falls back on the format its chat template
trained it on: <tool_call> blocks, <function=...> XML, [TOOL_CALLS] arrays,
<|python_tag|>, pythonic lists, harmony "to=functions.NAME" headers. Each of
those used to be shown to the user as a raw answer, because the only parser
understood the canonical JSON. And 4-bit quantisation makes even the canonical
JSON fragile: smart quotes, trailing commas, Python True/None, an unquoted key,
a raw newline inside a string or one missing closing brace is all it takes for
json.loads to refuse a call the model plainly meant.

This module turns any of those into ParsedToolCall objects, records every
repair it had to make (so a recovered call is visible in logs, never silent),
and never invents a tool name: the only name cleaning it does is removing
deterministic noise ("functions.", a trailing "()") and only when the cleaned
name is a registered tool.

It also owns the reasoning-block primitives (THINK_BLOCK, strip_reasoning),
because the parser must see exactly the visible text the streaming splitter in
llm.py shows the user; one definition keeps the two from disagreeing.

Token spellings come from the parsers that ship with mlx-lm 0.32
(.venv/lib/python3.14/site-packages/mlx_lm/tool_parsers/): json_tools.py
(<tool_call>), qwen3_coder.py:14-16,119-121 (<function=..>/<parameter=..>),
glm47.py:15-22 (<arg_key>/<arg_value>), mistral.py:9-12 ([TOOL_CALLS],
[ARGS]), kimi_k2.py:16-25 (<|tool_calls_section_begin|>, <|tool_call_begin|>,
<|tool_call_argument_begin|>), pythonic.py:12,115-116 (<|tool_call_start|>),
function_gemma.py:11-47 (<start_function_call>call:NAME{...}). mlx-lm has no
parser for Llama 3.x <|python_tag|>, Gemma ```tool_code fences, DeepSeek V3
(<｜tool▁call▁begin｜> with full-width bars) or gpt-oss harmony; those follow
the models' published chat templates.
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .core import *  # noqa: F401,F403


# --------------------------------------------------------------------------- #
# Reasoning blocks
# --------------------------------------------------------------------------- #
# <seed:think> is ByteDance Seed-OSS. gpt-oss (harmony) has no tags at all: its
# reasoning is the "analysis" channel, closed by <|end|>, and the answer is the
# "final" channel whose header has to be removed from the visible text too.
REASONING_TAGS = ("think", "thinking", "reasoning", "seed:think")
_TP_TAG_ALT = "think|thinking|reasoning|seed:think"
HARMONY_ANALYSIS_OPEN = "<|channel|>analysis<|message|>"
HARMONY_END = "<|end|>"
HARMONY_NOISE = ("<|start|>assistant<|channel|>final<|message|>",
                 "<|channel|>final<|message|>", "<|return|>")

THINK_BLOCK = re.compile(
    rf"(?is)<({_TP_TAG_ALT})>.*?</\1>|<\|channel\|>analysis<\|message\|>.*?<\|end\|>")
OPEN_THINK = re.compile(
    rf"(?is)(?:<({_TP_TAG_ALT})>|<\|channel\|>analysis<\|message\|>).*\Z")
_TP_NOISE_RE = re.compile("|".join(re.escape(n) for n in HARMONY_NOISE))
_TP_FIRST_TAG = re.compile(rf"(?i)<(/?)({_TP_TAG_ALT})>|<\|channel\|>analysis<\|message\|>")


def strip_reasoning(text: str) -> str:
    """Remove a reasoning model's chain-of-thought block.

    Qwen3.5 and later emit <think>...</think> before the answer. extract_json_object
    scans for the first balanced JSON object anywhere in the reply, so reasoning
    that talks through candidate arguments gets parsed as the tool call itself.
    An unterminated block is stripped too, because mid-stream the closing tag
    has not arrived yet and the agent tests the buffer on every token.

    A bare leading close tag ("...reasoning</think>answer") is what a template
    that opens <think> inside the PROMPT produces (QwQ, R1-style distills): the
    reply never contains the opening tag. When the first reasoning tag in the
    text is a close tag, everything up to it is reasoning.
    """
    if not text or "<" not in text:
        return text
    first = _TP_FIRST_TAG.search(text)
    if first is not None and first.group(1) == "/":
        text = text[first.end():]
    text = THINK_BLOCK.sub("", text)
    text = OPEN_THINK.sub("", text)
    if "<|" in text:
        text = _TP_NOISE_RE.sub("", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Policy: which formats, whether to repair
# --------------------------------------------------------------------------- #
# "json" is the canonical protocol and its historical variants; it is always on
# because it is what the system prompt asks for.
TOOL_CALL_FORMATS = ("json", "openai", "hermes", "qwen3_coder", "glm", "llama3",
                     "mistral", "pythonic", "gemma", "deepseek", "kimi", "harmony")

# The live Config whose tool_call_format / tool_call_repair are read at parse
# time. A reference, not a copy, so POST /api/config takes effect immediately.
# ModelClient binds it; with nothing bound the environment decides.
_TP_POLICY: dict = {"config": None}


def bind_tool_call_config(config: Any) -> None:
    """Point the parser at the running Config (read on every parse)."""
    _TP_POLICY["config"] = config


def normalise_tool_call_format(value: Any) -> str:
    """"auto", or a comma list of known format names. Unknown names fall back
    to "auto" rather than silently disabling every non-canonical format."""
    raw = str(value or "auto").strip().lower()
    if raw in ("", "auto", "all"):
        return "auto"
    names = [part.strip() for part in raw.split(",") if part.strip()]
    good = [n for n in names if n in TOOL_CALL_FORMATS]
    if not good:
        return "auto"
    return ",".join(dict.fromkeys(good))


def tool_call_policy() -> tuple[str, bool]:
    """(format setting, repair on) for the next parse."""
    cfg = _TP_POLICY["config"]
    fmt = getattr(cfg, "tool_call_format", None) if cfg is not None else None
    rep = getattr(cfg, "tool_call_repair", None) if cfg is not None else None
    if fmt is None:
        fmt = os.environ.get("TOOL_CALL_FORMAT", "auto")
    if rep is None:
        rep = os.environ.get("TOOL_CALL_REPAIR", "1") == "1"
    return normalise_tool_call_format(fmt), bool(rep)


def _tp_enabled(formats: Any) -> frozenset:
    if formats is None:
        formats = tool_call_policy()[0]
    if isinstance(formats, str):
        spec = normalise_tool_call_format(formats)
        names = TOOL_CALL_FORMATS if spec == "auto" else spec.split(",")
    else:
        names = [str(f).strip().lower() for f in formats]
    return frozenset(set(names) | {"json"})


# --------------------------------------------------------------------------- #
# The result type
# --------------------------------------------------------------------------- #
@dataclass
class ParsedToolCall:
    """One tool call recovered from model text.

    `recovered` is True when any repair was needed, and `repairs` names each
    one ("single_quotes", "trailing_commas", "unclosed_brackets", ...), so a
    call that only parsed because it was fixed up is never indistinguishable
    from a clean one.
    """

    name: str
    args: dict
    raw: str = ""
    format: str = "json"
    recovered: bool = False
    repairs: list = field(default_factory=list)
    id: str = ""

    def as_tuple(self) -> tuple[str, dict]:
        return self.name, self.args

    def as_protocol(self) -> str:
        """The canonical protocol text for this call."""
        return json.dumps({"tool": self.name, "args": self.args}, ensure_ascii=False)


def _tp_call(name: str, args: dict, raw: str, fmt: str, repairs: list,
             call_id: str = "") -> ParsedToolCall:
    reps = list(dict.fromkeys(repairs))
    return ParsedToolCall(name=name, args=args, raw=raw, format=fmt,
                          recovered=bool(reps), repairs=reps, id=call_id or "")


# --------------------------------------------------------------------------- #
# Lenient JSON (repair mode only: strict json.loads is always tried first)
# --------------------------------------------------------------------------- #
_TP_ZW_RE = re.compile("[​‌‍⁠﻿]")
_TP_IDENT = re.compile(r"[A-Za-z_$][\w$\-.]*")
_TP_NUMBER = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_TP_INT = re.compile(r"-?\d+")
# Opening quote -> the characters that may close it. Smart quotes are what a
# model trained on prose reaches for; "„" opens German quotes.
_TP_CLOSERS = {'"': '"', "'": "'", "“": "”“\"", "”": "”\"",
               "„": "“”", "‘": "’‘'", "’": "’'"}
_TP_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n",
               "r": "\r", "t": "\t", "'": "'"}
_TP_LITERALS = {"true": True, "false": False, "null": None}
_TP_PY_LITERALS = {"True": True, "False": False, "None": None}
_TP_DECODER = json.JSONDecoder()


class _TpBroken(ValueError):
    """Not recoverable: the text is not JSON-shaped enough to repair."""


class _TpLenient:
    """A forgiving recursive-descent JSON reader that records what it forgave.

    `close_open` allows the text to END inside an object or array (the
    brackets are closed for it). It never allows ending inside a string: a
    string cut off mid-value is a reply that hit its token budget, and
    completing it would run a tool on half an argument.
    """

    def __init__(self, text: str, close_open: bool) -> None:
        self.s = text
        self.n = len(text)
        self.i = 0
        self.close_open = close_open
        self.repairs: list[str] = []

    def note(self, tag: str) -> None:
        if tag not in self.repairs:
            self.repairs.append(tag)

    def ws(self) -> None:
        s, n, i = self.s, self.n, self.i
        while i < n and s[i] in " \t\r\n":
            i += 1
        self.i = i

    def eof(self) -> None:
        if not self.close_open:
            raise _TpBroken("text ends inside a container")
        self.note("unclosed_brackets")

    def value(self) -> Any:
        self.ws()
        if self.i >= self.n:
            raise _TpBroken("missing value")
        c = self.s[self.i]
        if c == "{":
            return self.obj()
        if c == "[":
            return self.arr()
        if c in _TP_CLOSERS:
            return self.string()
        if c in "-.0123456789":
            m = _TP_NUMBER.match(self.s, self.i)
            if not m:
                raise _TpBroken("bad number")
            self.i = m.end()
            txt = m.group(0)
            if _TP_INT.fullmatch(txt):
                return int(txt)
            if txt.endswith(".") or txt.lstrip("-").startswith("."):
                self.note("number_format")
            return float(txt)
        m = _TP_IDENT.match(self.s, self.i)
        if m:
            word = m.group(0)
            self.i = m.end()
            if word in _TP_LITERALS:
                return _TP_LITERALS[word]
            if word in _TP_PY_LITERALS:
                self.note("python_literals")
                return _TP_PY_LITERALS[word]
            self.note("unquoted_values")
            return word
        raise _TpBroken(f"unexpected {c!r}")

    def obj(self) -> dict:
        self.i += 1
        out: dict = {}
        while True:
            self.ws()
            if self.i >= self.n:
                self.eof()
                return out
            c = self.s[self.i]
            if c == "}":
                self.i += 1
                return out
            if c == "]":
                self.note("mismatched_brackets")
                return out
            if c == ",":
                self.i += 1
                self.note("stray_commas")
                continue
            if c in _TP_CLOSERS:
                key = self.string()
            else:
                m = _TP_IDENT.match(self.s, self.i)
                if not m:
                    raise _TpBroken("bad key")
                key = m.group(0)
                self.i = m.end()
                self.note("unquoted_keys")
            self.ws()
            if self.i >= self.n:
                self.eof()
                return out
            if self.s[self.i] != ":":
                raise _TpBroken("missing colon")
            self.i += 1
            self.ws()
            if self.i >= self.n:
                self.eof()
                return out
            out[str(key)] = self.value()
            self.ws()
            if self.i >= self.n:
                self.eof()
                return out
            c = self.s[self.i]
            if c == ",":
                self.i += 1
                self.ws()
                if self.i < self.n and self.s[self.i] == "}":
                    self.note("trailing_commas")
                continue
            if c == "}":
                self.i += 1
                return out
            if c == "]":
                self.note("mismatched_brackets")
                return out
            if c in _TP_CLOSERS or _TP_IDENT.match(self.s, self.i):
                self.note("missing_commas")
                continue
            raise _TpBroken(f"unexpected {c!r} in object")

    def arr(self) -> list:
        self.i += 1
        out: list = []
        while True:
            self.ws()
            if self.i >= self.n:
                self.eof()
                return out
            c = self.s[self.i]
            if c == "]":
                self.i += 1
                return out
            if c == "}":
                self.note("mismatched_brackets")
                return out
            if c == ",":
                self.i += 1
                self.note("stray_commas")
                continue
            out.append(self.value())
            self.ws()
            if self.i >= self.n:
                self.eof()
                return out
            c = self.s[self.i]
            if c == ",":
                self.i += 1
                self.ws()
                if self.i < self.n and self.s[self.i] == "]":
                    self.note("trailing_commas")
                continue
            if c == "]":
                self.i += 1
                return out
            if c == "}":
                self.note("mismatched_brackets")
                return out
            self.note("missing_commas")

    def _closes(self, i: int) -> bool:
        """Is the quote at i the end of the string, or a quote inside it?

        A 4-bit model writes `"he said "hi" twice"` without escaping. A quote
        followed by a separator (or the end) closes; anything else is content.
        """
        s, n = self.s, self.n
        j = i + 1
        newline = False
        while j < n and s[j] in " \t\r\n":
            newline = newline or s[j] in "\r\n"
            j += 1
        if j >= n:
            return True
        ch = s[j]
        if ch in ",:}]":
            return True
        return newline and ch in _TP_CLOSERS

    def string(self) -> str:
        s, n = self.s, self.n
        q = s[self.i]
        closers = _TP_CLOSERS[q]
        if q == "'":
            self.note("single_quotes")
        elif q != '"':
            self.note("smart_quotes")
        i = self.i + 1
        buf: list[str] = []
        while i < n:
            c = s[i]
            if c == "\\" and i + 1 < n:
                e = s[i + 1]
                if e in _TP_ESCAPES:
                    buf.append(_TP_ESCAPES[e])
                    i += 2
                    continue
                if e == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", s[i + 2:i + 6]):
                    buf.append(chr(int(s[i + 2:i + 6], 16)))
                    i += 6
                    continue
                # "\d" in a regex argument: keep it literally rather than lose it.
                self.note("bad_escapes")
                buf.append("\\" + e)
                i += 2
                continue
            if c in closers:
                if self._closes(i):
                    self.i = i + 1
                    out = "".join(buf)
                    if any("\ud800" <= ch <= "\udfff" for ch in out):
                        out = out.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
                    return out
                self.note("inner_quotes")
            elif c in "\n\r\t":
                self.note("raw_control_chars")
            buf.append(c)
            i += 1
        raise _TpBroken("unterminated string")


def _tp_decode_one(text: str, repair: bool, close_open: bool) -> tuple[Any, list, int]:
    """Decode the JSON value at the start of `text`.

    Returns (value, repairs, end offset); value is None when nothing decodes.
    Strict first, so valid JSON never meets a heuristic.
    """
    s = text.lstrip()
    lead = len(text) - len(s)
    if not s:
        return None, [], 0
    try:
        value, end = _TP_DECODER.raw_decode(s)
        return value, [], lead + end
    except (ValueError, RecursionError):
        pass
    if not repair:
        return None, [], 0
    try:
        p = _TpLenient(s, close_open)
        value = p.value()
        return value, p.repairs, lead + p.i
    except (_TpBroken, RecursionError):
        return None, [], 0


def repair_json(text: str, close_open: bool = False) -> tuple[Any, list[str]]:
    """Public lenient loader: (value or None, repairs applied)."""
    value, repairs, _ = _tp_decode_one(_TP_ZW_RE.sub("", text or ""), True, close_open)
    return value, repairs


def _tp_unfence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"\A```[\w-]*\s*", "", s)
        s = re.sub(r"\s*```\s*\Z", "", s)
    return s


# --------------------------------------------------------------------------- #
# JSON objects -> calls
# --------------------------------------------------------------------------- #
_TP_ARG_KEYS = ("args", "arguments", "parameters")
_TP_NAME_PREFIXES = ("functions.", "function.", "tools.", "tool.", "default_api.", "api.")


def _tp_clean_name(name: str, known: Any, repair: bool, repairs: list) -> str:
    """Strip deterministic noise from a name, only if that yields a known tool.

    Never fuzzy: "websearch" is NOT turned into "web_search". A model that
    named a tool that does not exist gets told so by the registry.
    """
    if repair and _TP_ZW_RE.search(name):
        # A zero-width space is never part of an identifier; it is tokenizer
        # debris, and "web<ZWSP>_search" must not reach the registry as a
        # different (unknown) tool.
        name = _TP_ZW_RE.sub("", name)
        repairs.append("zero_width")
    if not repair or known is None or name in known:
        return name
    cleaned = name.strip().strip("`'\"")
    for prefix in _TP_NAME_PREFIXES:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    cleaned = re.sub(r"\s*\(\s*\)?\s*\Z", "", cleaned).strip()
    if cleaned != name and cleaned in known:
        repairs.append("name_noise")
        return cleaned
    return name


def _tp_args(value: Any, repair: bool, repairs: list) -> dict:
    """The historical args rules: a JSON string is decoded, any other string
    becomes {"query": s}, anything not a dict becomes {}."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            decoded = None
            if repair and value.strip().startswith("{"):
                got, reps, _ = _tp_decode_one(value.strip(), True, True)
                if isinstance(got, dict):
                    decoded = got
                    repairs.append("string_args")
                    repairs.extend(reps)
            value = decoded if decoded is not None else {"query": value}
    if not isinstance(value, dict):
        value = {}
    return value


def _tp_obj_calls(obj: Any, known: Any, *, repaired: bool, marker: bool,
                  allow_openai: bool, repair: bool, fmt: str = "json",
                  raw: str = "") -> list[ParsedToolCall] | None:
    """One decoded JSON value -> its calls, or None when it is not a call.

    `marker` is True when a format token already said "this is a tool call"
    (so an unregistered name still counts, like the explicit "tool" key).
    `repaired` objects need a stronger shape: an explicit "tool" key or an
    args key, so `{name: 'value'}` prose that a repair made parseable is not
    promoted to a call.
    """
    if isinstance(obj, list):
        if not obj:
            return None
        out: list[ParsedToolCall] = []
        for item in obj:
            got = _tp_obj_calls(item, known, repaired=repaired, marker=marker,
                                allow_openai=allow_openai, repair=repair, fmt=fmt, raw=raw)
            if not got:
                return None
            out.extend(got)
        return out
    if not isinstance(obj, dict):
        return None
    if allow_openai:
        listed = obj.get("tool_calls")
        if isinstance(listed, list) and listed:
            return _tp_obj_calls(listed, known, repaired=repaired, marker=marker,
                                 allow_openai=True, repair=repair, fmt="openai", raw=raw)
        for key in ("function_call", "function"):
            inner = obj.get(key)
            if isinstance(inner, dict) and isinstance(inner.get("name"), str) and inner["name"]:
                reps: list[str] = []
                name = _tp_clean_name(inner["name"].strip(), known, repair, reps)
                if known is not None and name not in known and not marker:
                    return None
                args = _tp_args(inner.get("arguments", inner.get("parameters")) or {},
                                repair, reps)
                call_id = obj.get("id") if isinstance(obj.get("id"), str) else ""
                return [_tp_call(name, args, raw, "openai", reps, call_id)]
    explicit = "tool" in obj or "tool_name" in obj
    name = obj.get("tool") or obj.get("name") or obj.get("tool_name")
    if not isinstance(name, str) or not name:
        return None
    if repaired and not explicit and not any(k in obj for k in _TP_ARG_KEYS):
        return None
    reps = []
    name = _tp_clean_name(name.strip(), known, repair, reps)
    if known is not None and name not in known and not (explicit or marker):
        return None
    raw_args = obj.get("args") or obj.get("arguments") or obj.get("parameters") or {}
    args = _tp_args(raw_args, repair, reps)
    call_id = obj.get("id") if isinstance(obj.get("id"), str) else ""
    return [_tp_call(name, args, raw, fmt, reps, call_id)]


_TP_FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def tool_json_candidates(text: str) -> tuple[list[tuple[int, str, bool]], int]:
    """Every candidate JSON object in `text`, in the order the parser tries them.

    Returns ([(start, text, fenced)], start of an object still open at the end
    or -1). The scan is the historical extract_json_object one, kept
    byte-for-byte (a fenced object first, then each balanced top-level object),
    because which object counts as "the reply's JSON" decides whether a prose
    answer with an example in it stays prose.
    """
    out: list[tuple[int, str, bool]] = []
    if not text:
        return out, -1
    fenced = _TP_FENCED.search(text)
    if fenced:
        out.append((fenced.start(1), fenced.group(1), True))
    depth = 0
    start = -1
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                out.append((start, text[start:index + 1], False))
                start = -1
    return out, (start if depth > 0 and start >= 0 else -1)


def _tp_more_json_calls(cands, after: int, known, repair: bool, allow_openai: bool) -> list:
    """Further calls written after the first one, stopping at the first
    object that is not a call (a hallucinated tool RESULT ends the run)."""
    out: list[ParsedToolCall] = []
    for start, cand, fenced in cands:
        if fenced or start <= after:
            continue
        repaired = False
        reps: list[str] = []
        try:
            obj = json.loads(cand)
        except Exception:
            if not repair:
                break
            obj, reps, _ = _tp_decode_one(cand, True, False)
            repaired = True
        calls = _tp_obj_calls(obj, known, repaired=repaired, marker=False,
                              allow_openai=allow_openai, repair=repair, raw=cand)
        if not calls:
            break
        for c in calls:
            if reps:
                c.repairs = list(dict.fromkeys(reps + c.repairs))
                c.recovered = True
        out.extend(calls)
    return out


def _tp_json_chain(work: str, known, enabled, repair: bool, final: bool) -> list:
    cands, open_start = tool_json_candidates(work)
    allow_openai = "openai" in enabled
    # 1. Exactly the historical rule: the first object that parses decides.
    for start, cand, _fenced in cands:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        calls = _tp_obj_calls(obj, known, repaired=False, marker=False,
                              allow_openai=allow_openai, repair=repair, raw=cand)
        if not calls:
            return []          # prose that contains JSON stays prose
        return calls + _tp_more_json_calls(cands, start, known, repair, allow_openai)
    if not repair:
        return []
    # 2. Nothing parsed strictly: the first object that a repair makes
    # parseable decides, under the same rule.
    for start, cand, _fenced in cands:
        obj, reps, _ = _tp_decode_one(cand, True, False)
        if not isinstance(obj, dict):
            continue
        calls = _tp_obj_calls(obj, known, repaired=True, marker=False,
                              allow_openai=allow_openai, repair=True, raw=cand)
        if not calls:
            return []
        for c in calls:
            c.repairs = list(dict.fromkeys(reps + c.repairs))
            c.recovered = bool(c.repairs)
        return calls + _tp_more_json_calls(cands, start, known, repair, allow_openai)
    # 3. An object still open when the reply ENDED (final=True only): close it.
    if final and open_start >= 0:
        tail = work[open_start:]
        obj, reps, _ = _tp_decode_one(tail, True, True)
        if isinstance(obj, dict):
            calls = _tp_obj_calls(obj, known, repaired=True, marker=False,
                                  allow_openai=allow_openai, repair=True, raw=tail)
            if calls:
                for c in calls:
                    c.repairs = list(dict.fromkeys(reps + c.repairs))
                    c.recovered = True
                return calls
    return []


# --------------------------------------------------------------------------- #
# Special-token formats
# --------------------------------------------------------------------------- #
_TP_NAME_RE = re.compile(r"[A-Za-z_][\w.\-:]*\Z")
_TP_HERMES = re.compile(r"<tool_call>(.*?)(</tool_call>|\Z)", re.S)
_TP_Q3_FUNC = re.compile(r"<function=\s*([^\s<>]+?)\s*>(.*?)(</function>|\Z)", re.S)
_TP_Q3_PARAM = re.compile(
    r"<parameter=\s*([^\s<>]+?)\s*>(.*?)(?:</parameter>|(?=<parameter=)|\Z)", re.S)
_TP_GLM_PAIR = re.compile(
    r"<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>", re.S)
_TP_PYTAG = re.compile(
    r"<\|python_tag\|>(.*?)(<\|eom_id\|>|<\|eot_id\|>|(?=<\|python_tag\|>)|\Z)", re.S)
_TP_MISTRAL_HEAD = re.compile(r"\s*([\w.\-]+)\s*(?:\[CALL_ID\]\s*([\w\-]+)\s*)?\[ARGS\]\s*")
_TP_PY_BLOCK = re.compile(r"<\|tool_call_start\|>(.*?)(<\|tool_call_end\|>|\Z)", re.S)
_TP_TOOL_CODE = re.compile(r"```tool_code[ \t]*\n?(.*?)(```|\Z)", re.S)
_TP_FGEMMA = re.compile(
    r"<start_function_call>\s*call:([\w.\-]+)\{(.*?)\}\s*(<end_function_call>|\Z)", re.S)
_TP_DS_CALL = re.compile(
    "<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)"
    "(<｜tool▁call▁end｜>|(?=<｜tool▁call▁begin｜>)"
    "|<｜tool▁calls▁end｜>|\\Z)", re.S)
_TP_KIMI = re.compile(
    r"<\|tool_call_begin\|>\s*((?:functions\.)?([^\s:<]+?)(?::\d+)?)\s*"
    r"<\|tool_call_argument_begin\|>(.*?)(<\|tool_call_end\|>|\Z)", re.S)
_TP_HARMONY = re.compile(
    r"to=functions\.([\w.\-]+)(.*?)<\|message\|>(.*?)(<\|call\|>|<\|end\|>|\Z)", re.S)

DS_TOOL_CALLS_BEGIN = "<｜tool▁calls▁begin｜>"
DS_TOOL_CALLS_END = "<｜tool▁calls▁end｜>"
DS_TOOL_CALL_BEGIN = "<｜tool▁call▁begin｜>"
DS_TOOL_CALL_END = "<｜tool▁call▁end｜>"

# Marker -> formats that can use it. Only formats whose marker occurs in the
# text are tried, so an ordinary prose reply costs a dozen substring checks.
_TP_MARKERS = (
    ("<tool_call>", ("hermes", "qwen3_coder", "glm")),
    ("<function=", ("qwen3_coder",)),
    ("<|python_tag|>", ("llama3",)),
    ("[TOOL_CALLS]", ("mistral",)),
    ("<|tool_call_start|>", ("pythonic",)),
    ("```tool_code", ("gemma",)),
    ("<start_function_call>", ("gemma",)),
    (DS_TOOL_CALL_BEGIN, ("deepseek",)),
    ("<|tool_call_begin|>", ("kimi",)),
    ("to=functions.", ("harmony",)),
)
_TP_LEADING_MARKER = re.compile(
    r"\A\s*(?:<tool_call>|<function=|<\|python_tag\|>|\[TOOL_CALLS\]|<\|tool_call_start\|>"
    r"|```tool_code|<start_function_call>|" + re.escape(DS_TOOL_CALLS_BEGIN) + "|"
    + re.escape(DS_TOOL_CALL_BEGIN) + r"|<\|tool_calls_section_begin\|>|<\|tool_call_begin\|>"
    r"|<\|start\|>|<\|channel\|>|to=functions\.)")


def _tp_args_text(text: str, repair: bool, close_open: bool, repairs: list) -> dict | None:
    """An arguments body (JSON object text) -> dict, or None."""
    body = _tp_unfence(text)
    if not body:
        return {}
    value, reps, _ = _tp_decode_one(body, repair, close_open)
    if isinstance(value, str):
        value = _tp_args(value, repair, reps)
    if not isinstance(value, dict):
        return None
    repairs.extend(reps)
    return value


def _tp_deserialize(value: str) -> Any:
    """GLM's value rule (glm47.py _deserialize): JSON, then a Python literal,
    else the text itself."""
    try:
        return json.loads(value)
    except Exception:
        pass
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _tp_q3_value(value: str) -> Any:
    """Qwen3-Coder parameter text. One leading and trailing newline belong to
    the template (qwen3_coder.py strips them). Without the tool's JSON schema
    mlx-lm keeps the value a string; containers are decoded so a list argument
    is still a list."""
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    s = value.strip()
    if s[:1] in ("{", "["):
        try:
            return json.loads(s)
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                return value
    return value


def _tp_qwen3(work: str, enabled, repair: bool, final: bool, out: list) -> None:
    if "qwen3_coder" not in enabled:
        return
    for m in _TP_Q3_FUNC.finditer(work):
        name = m.group(1).strip()
        closed = bool(m.group(3))
        reps: list[str] = []
        if not closed:
            if not (final and repair):
                continue
            reps.append("unclosed_tags")
        if not _TP_NAME_RE.match(name):
            continue
        args = {p.group(1).strip(): _tp_q3_value(p.group(2))
                for p in _TP_Q3_PARAM.finditer(m.group(2))}
        out.append((m.start(), _tp_call(name, args, m.group(0), "qwen3_coder", reps)))


def _tp_glm(body: str, raw: str, pos: int, out: list) -> None:
    name = body.split("<arg_key>", 1)[0].strip()
    if not _TP_NAME_RE.match(name):
        return
    args = {p.group(1).strip(): _tp_deserialize(p.group(2))
            for p in _TP_GLM_PAIR.finditer(body)}
    out.append((pos, _tp_call(name, args, raw, "glm", [])))


def _tp_hermes(work: str, known, enabled, repair: bool, final: bool, out: list) -> None:
    for m in _TP_HERMES.finditer(work):
        body = m.group(1).strip()
        closed = bool(m.group(2))
        if not body or "<function=" in body:
            continue                   # empty, or Qwen3-Coder XML (parsed on its own)
        if "<arg_key>" in body or (closed and _TP_NAME_RE.match(body)):
            if "glm" in enabled and (closed or final):
                _tp_glm(body, m.group(0), m.start(), out)
            continue
        if "hermes" not in enabled:
            continue
        rest = _tp_unfence(body)
        while rest:
            value, reps, end = _tp_decode_one(rest, repair, closed or final)
            if value is None:
                break
            calls = _tp_obj_calls(value, known, repaired=bool(reps), marker=True,
                                  allow_openai=True, repair=repair, fmt="hermes",
                                  raw=m.group(0))
            if not calls:
                break
            for c in calls:
                c.repairs = list(dict.fromkeys(reps + c.repairs))
                c.recovered = bool(c.repairs)
                out.append((m.start(), c))
            rest = rest[end:].lstrip().lstrip(",;").lstrip()


def _tp_literal(node: ast.AST) -> Any:
    """ast.literal_eval with JSON's true/false/null accepted (chat templates
    render nested values as JSON; pythonic.py _JSONLiterals does the same)."""
    class _Lit(ast.NodeTransformer):
        def visit_Name(self, n: ast.Name) -> ast.AST:
            if n.id in _TP_LITERALS:
                return ast.copy_location(ast.Constant(_TP_LITERALS[n.id]), n)
            return n
    return ast.literal_eval(_Lit().visit(node))


def _tp_dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _tp_dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _tp_py_call(node: ast.AST) -> tuple[str, dict] | None:
    """One ast.Call -> (name, kwargs). Literals only: nothing is evaluated."""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "print" and len(node.args) == 1 and not node.keywords):
        node = node.args[0]           # Gemma writes print(f(...))
    if not isinstance(node, ast.Call) or node.args:
        return None                   # positional args cannot be mapped to names
    name = _tp_dotted(node.func)
    if not name:
        return None
    if name.endswith(".call"):
        name = name[:-5]              # Llama built-ins: brave_search.call(query=...)
    if name.startswith("default_api."):
        name = name[len("default_api."):]
    args: dict = {}
    for kw in node.keywords:
        if kw.arg is None:
            return None
        try:
            args[kw.arg] = _tp_literal(kw.value)
        except Exception:
            return None
    return name, args


def _tp_pythonic(src: str) -> list[tuple[str, dict]] | None:
    """`[f(a=1), g(b="x")]` or `f(a=1)` -> calls, via ast literals only."""
    src = src.strip()
    if not src:
        return None
    try:
        tree = ast.parse(src, mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        return None
    body = tree.body
    items = body.elts if isinstance(body, (ast.List, ast.Tuple)) else [body]
    calls = []
    for item in items:
        got = _tp_py_call(item)
        if got is None:
            return None
        calls.append(got)
    return calls or None


def _tp_llama(work: str, known, repair: bool, final: bool, out: list) -> None:
    for m in _TP_PYTAG.finditer(work):
        body = m.group(1).strip()
        if not body:
            continue
        if body.startswith("{"):
            rest = body
            while rest.startswith("{"):
                value, reps, end = _tp_decode_one(rest, repair, final or bool(m.group(2)))
                calls = _tp_obj_calls(value, known, repaired=bool(reps), marker=True,
                                      allow_openai=False, repair=repair, fmt="llama3",
                                      raw=m.group(0)) if value is not None else None
                if not calls:
                    break
                for c in calls:
                    c.repairs = list(dict.fromkeys(reps + c.repairs))
                    c.recovered = bool(c.repairs)
                    out.append((m.start(), c))
                rest = rest[end:].lstrip().lstrip(";").lstrip()
            continue
        for line in re.split(r"[;\n]+", body):
            got = _tp_pythonic(line)
            for name, args in got or []:
                out.append((m.start(), _tp_call(name, args, m.group(0), "llama3", [])))


def _tp_mistral(work: str, known, repair: bool, final: bool, out: list) -> None:
    marker = "[TOOL_CALLS]"
    pos = work.find(marker)
    while pos >= 0:
        nxt = work.find(marker, pos + len(marker))
        seg = work[pos + len(marker): nxt if nxt >= 0 else len(work)]
        closed = nxt >= 0 or final
        s = seg.lstrip()
        if s.startswith("["):
            value, reps, _ = _tp_decode_one(s, repair, closed)
            calls = _tp_obj_calls(value, known, repaired=bool(reps), marker=True,
                                  allow_openai=True, repair=repair, fmt="mistral",
                                  raw=marker + seg) if isinstance(value, list) else None
            for c in calls or []:
                c.repairs = list(dict.fromkeys(reps + c.repairs))
                c.recovered = bool(c.repairs)
                out.append((pos, c))
        else:
            at = 0
            while True:
                head = _TP_MISTRAL_HEAD.match(seg, at)
                if head is None:
                    break
                reps: list[str] = []
                value, dreps, end = _tp_decode_one(seg[head.end():], repair, closed)
                if not isinstance(value, dict):
                    break
                reps.extend(dreps)
                out.append((pos, _tp_call(head.group(1), value, marker + seg, "mistral",
                                          reps, head.group(2) or "")))
                at = head.end() + end
        pos = nxt


def _tp_fgemma_args(text: str) -> dict:
    """function_gemma.py's argument grammar: key:value pairs, strings wrapped
    in <escape>...<escape>, everything else JSON or left as text."""
    args: dict = {}
    rest = text
    escape = "<escape>"
    while rest:
        split = rest.find(":")
        if split < 0:
            break
        key = rest[:split].strip().lstrip(",").strip()
        rest = rest[split + 1:]
        if rest.startswith(escape):
            rest = rest[len(escape):]
            end = rest.find(escape)
            if end < 0:
                args[key] = rest
                break
            args[key] = rest[:end]
            rest = rest[end + len(escape):].lstrip(",")
            continue
        split = rest.find(",")
        value, rest = (rest, "") if split < 0 else (rest[:split], rest[split + 1:])
        try:
            args[key] = json.loads(value)
        except Exception:
            args[key] = value
    return args


def _tp_gemma(work: str, final: bool, out: list) -> None:
    for m in _TP_TOOL_CODE.finditer(work):
        if not m.group(2) and not final:
            continue
        for line in m.group(1).splitlines():
            got = _tp_pythonic(line)
            for name, args in got or []:
                out.append((m.start(), _tp_call(name, args, m.group(0), "gemma", [])))
    for m in _TP_FGEMMA.finditer(work):
        if not m.group(3) and not final:
            continue
        out.append((m.start(), _tp_call(m.group(1), _tp_fgemma_args(m.group(2)),
                                        m.group(0), "gemma", [])))


def _tp_named_args_format(regex, work: str, fmt: str, repair: bool, final: bool,
                          out: list, name_group: int, args_group: int, end_group: int,
                          id_group: int | None = None) -> None:
    for m in regex.finditer(work):
        name = m.group(name_group).strip()
        closed = bool(m.group(end_group)) or m.end(end_group) < len(work)
        argtxt = m.group(args_group)
        if fmt == "deepseek" and name == "function":
            # V3: <｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME\n```json\n{..}\n```
            name, _, argtxt = argtxt.lstrip().partition("\n")
            name = name.strip()
        if not _TP_NAME_RE.match(name or ""):
            continue
        reps: list[str] = []
        args = _tp_args_text(argtxt, repair, closed or final, reps)
        if args is None:
            continue
        call_id = m.group(id_group).strip() if id_group else ""
        out.append((m.start(), _tp_call(name, args, m.group(0), fmt, reps, call_id)))


def _tp_marker_calls(work: str, known, enabled, repair: bool, final: bool) -> list:
    present = {fmt for marker, fmts in _TP_MARKERS if marker in work for fmt in fmts}
    present &= enabled
    if not present:
        return []
    found: list[tuple[int, ParsedToolCall]] = []
    if present & {"hermes", "glm"}:
        _tp_hermes(work, known, enabled, repair, final, found)
    if "qwen3_coder" in present:
        _tp_qwen3(work, enabled, repair, final, found)
    if "llama3" in present:
        _tp_llama(work, known, repair, final, found)
    if "mistral" in present:
        _tp_mistral(work, known, repair, final, found)
    if "pythonic" in present:
        for m in _TP_PY_BLOCK.finditer(work):
            if not m.group(2) and not final:
                continue
            for name, args in _tp_pythonic(m.group(1)) or []:
                found.append((m.start(), _tp_call(name, args, m.group(0), "pythonic", [])))
    if "gemma" in present:
        _tp_gemma(work, final, found)
    if "deepseek" in present:
        _tp_named_args_format(_TP_DS_CALL, work, "deepseek", repair, final, found, 1, 2, 3)
    if "kimi" in present:
        _tp_named_args_format(_TP_KIMI, work, "kimi", repair, final, found, 2, 3, 4, 1)
    if "harmony" in present:
        _tp_named_args_format(_TP_HARMONY, work, "harmony", repair, final, found, 1, 3, 4)
    if not found:
        return []
    # A tool-call token at the very start means the reply IS a call, so an
    # unregistered name still goes through (the registry then says so, as it
    # does for an explicit {"tool": ...}). Mid-prose, only registered names
    # count: an answer EXPLAINING the Hermes format is not a call.
    leading = bool(_TP_LEADING_MARKER.match(work))
    calls: list[ParsedToolCall] = []
    for _pos, call in sorted(found, key=lambda item: item[0]):
        reps: list[str] = []
        call.name = _tp_clean_name(call.name, known, repair, reps)
        if reps:
            call.repairs = list(dict.fromkeys(call.repairs + reps))
            call.recovered = True
        if known is None or call.name in known or leading:
            calls.append(call)
    return calls


_TP_BARE_PY = re.compile(r"\A\[\s*[A-Za-z_][\w.]*\s*\(")


def _tp_bare_pythonic(work: str, known) -> list:
    s = _tp_unfence(work)
    if not _TP_BARE_PY.match(s):
        return []
    got = _tp_pythonic(s)
    if not got:
        return []
    if known is not None and not all(name in known for name, _ in got):
        return []
    return [_tp_call(name, args, s, "pythonic", []) for name, args in got]


# --------------------------------------------------------------------------- #
# The entry point
# --------------------------------------------------------------------------- #
def parse_tool_calls(text: str, known: Any = None, *, formats: Any = None,
                     repair: bool | None = None, final: bool = False) -> list[ParsedToolCall]:
    """Every tool call in a model reply, in order; [] when it is not a call.

    `known` is the set of registered tool names (None accepts any name, the
    historical behaviour). `formats` overrides TOOL_CALL_FORMAT and `repair`
    overrides TOOL_CALL_REPAIR. `final=True` says the reply has ENDED (and did
    not stop on its token budget): only then may a call whose closing brackets
    never came be closed for it. Mid-stream that would execute a call whose
    arguments are still arriving, so it is off by default.
    """
    if not text:
        return []
    visible = strip_reasoning(text)
    if not visible:
        return []
    fmt_setting, repair_default = tool_call_policy()
    repair = repair_default if repair is None else bool(repair)
    enabled = _tp_enabled(fmt_setting if formats is None else formats)
    calls = _tp_parse_visible(visible, known, enabled, repair, final)
    if not calls and repair and _TP_ZW_RE.search(visible):
        calls = _tp_parse_visible(_TP_ZW_RE.sub("", visible), known, enabled, repair, final)
        for c in calls:
            c.repairs = list(dict.fromkeys(["zero_width"] + c.repairs))
            c.recovered = True
    return calls


def _tp_parse_visible(work: str, known, enabled, repair: bool, final: bool) -> list:
    calls = _tp_marker_calls(work, known, enabled, repair, final)
    if calls:
        return calls
    calls = _tp_json_chain(work, known, enabled, repair, final)
    if calls:
        return calls
    if "pythonic" in enabled and "[" in work:
        return _tp_bare_pythonic(work, known)
    return []


def tool_call_unterminated(text: str, known: Any = None) -> bool:
    """True when the reply ends inside a special-token tool call that never
    finished (its token budget ran out), so it neither parses nor is prose."""
    body = strip_reasoning(text or "").strip()
    if not body:
        return False
    pairs = (("<tool_call>", "</tool_call>"), ("<function=", "</function>"),
             ("<|tool_call_start|>", "<|tool_call_end|>"), ("```tool_code", "```"),
             ("<start_function_call>", "<end_function_call>"),
             (DS_TOOL_CALL_BEGIN, DS_TOOL_CALL_END),
             ("<|tool_call_begin|>", "<|tool_call_end|>"),
             ("to=functions.", "<|call|>"))
    open_ended = False
    for opener, closer in pairs:
        at = body.rfind(opener)
        if at >= 0 and body.find(closer, at + len(opener)) < 0:
            open_ended = True
            break
    if not open_ended:
        for opener in ("<|python_tag|>", "[TOOL_CALLS]"):
            at = body.rfind(opener)
            if at >= 0:
                tail = body[at + len(opener):].strip()
                _c, open_at = tool_json_candidates(tail)
                if open_at >= 0 or tail.count("[") > tail.count("]"):
                    open_ended = True
                    break
    if not open_ended or parse_tool_calls(body, known, repair=False):
        return False
    if known:
        return any(re.search(rf"(?<![\w.]){re.escape(name)}(?![\w])", body) for name in known)
    return True


_TP_EMPTY_WRAPPERS = re.compile(
    r"\A(?:\s*(?:<tool_call>\s*(?:\{\s*\})?\s*</tool_call>|\[TOOL_CALLS\]\s*(?:\[\s*\])?"
    r"|<\|python_tag\|>\s*(?:\{\s*\})?\s*(?:<\|eom_id\|>)?"
    r"|<\|tool_call_start\|>\s*(?:\[\s*\])?\s*<\|tool_call_end\|>))+\s*\Z")


def is_empty_tool_wrapper(text: str) -> bool:
    """True for a special-token call with nothing in it (<tool_call></tool_call>)."""
    return bool(text) and bool(_TP_EMPTY_WRAPPERS.match(text))


# --------------------------------------------------------------------------- #
# Streaming detection
# --------------------------------------------------------------------------- #
_TP_STREAM_MARKERS = (
    ("<tool_call>", "</tool_call>", ("hermes", "qwen3_coder", "glm")),
    ("<function=", "</function>", ("qwen3_coder",)),
    ("<|python_tag|>", None, ("llama3",)),
    ("[TOOL_CALLS]", None, ("mistral",)),
    ("<|tool_call_start|>", "<|tool_call_end|>", ("pythonic",)),
    ("```tool_code", "```", ("gemma",)),
    ("<start_function_call>", "<end_function_call>", ("gemma",)),
    (DS_TOOL_CALLS_BEGIN, DS_TOOL_CALLS_END, ("deepseek",)),
    (DS_TOOL_CALL_BEGIN, DS_TOOL_CALL_END, ("deepseek",)),
    ("<|tool_calls_section_begin|>", "<|tool_calls_section_end|>", ("kimi",)),
    ("<|tool_call_begin|>", "<|tool_call_end|>", ("kimi",)),
    ("<|start|>assistant<|channel|>commentary to=functions.", "<|call|>", ("harmony",)),
    ("<|start|>assistant to=functions.", "<|call|>", ("harmony",)),
    ("<|channel|>commentary to=functions.", "<|call|>", ("harmony",)),
)
_TP_PYTAG_ENDS = ("<|eom_id|>", "<|eot_id|>")
_TP_CALL_KEYS = ("tool", "tool_name", "name", "function", "tool_calls", "function_call")
_TP_JSON_CALL_HEAD = re.compile(
    r"\A\s*(?:```(?:json)?\s*)?\{\s*[\"']?(?:tool|tool_name|name|function|tool_calls"
    r"|function_call)[\"']?\s*:")
_TP_PY_HEAD = re.compile(r"\A\s*\[\s*[A-Za-z_][\w.]*\s*\(")
_TP_PY_PREFIX = re.compile(r"\A\s*\[\s*(?:[A-Za-z_][\w.]*\s*)?\Z")


def _tp_stream_markers(enabled) -> list[tuple[str, str | None, str]]:
    return [(o, c, next(f for f in fmts if f in enabled) if any(f in enabled for f in fmts) else "")
            for o, c, fmts in _TP_STREAM_MARKERS if any(f in enabled for f in fmts)]


def _tp_prefix_tail(data: str, markers: Iterable[str]) -> int:
    """Length of the longest tail of `data` that is a proper prefix of a marker."""
    markers = list(markers)
    longest = max((len(m) for m in markers), default=0)
    for size in range(min(longest - 1, len(data)), 0, -1):
        tail = data[-size:]
        if any(m.startswith(tail) for m in markers):
            return size
    return 0


def looks_like_call_prefix(text: str, formats: Any = None) -> bool:
    """Could `text` (the reply so far) still turn out to be a tool call?

    For holding tokens back from display: True while the reply is a prefix of
    a tool-call token or of a canonical {"tool": ...} head.
    """
    s = (text or "").lstrip()
    if not s:
        return False
    enabled = _tp_enabled(formats)
    for opener, _close, _fmt in _tp_stream_markers(enabled):
        if s.startswith(opener) or opener.startswith(s):
            return True
    if "```json".startswith(s) or (s.startswith("```") and "```".startswith(s[:3])
                                   and re.fullmatch(r"```(?:j(?:s(?:o(?:n)?)?)?)?\s*", s)):
        return True
    if _TP_JSON_CALL_HEAD.match(s):
        return True
    t = re.sub(r"\A```(?:json)?\s*", "", s) if s.startswith("```") else s
    if t.startswith("{"):
        rest = t[1:].lstrip()
        if not rest:
            return True
        if rest[0] in "\"'":
            key = rest[1:]
            if re.fullmatch(r"\w*", key):
                return any(k.startswith(key) for k in _TP_CALL_KEYS)
            m = re.match(r"(\w+)[\"']\s*\Z", key)
            return bool(m) and m.group(1) in _TP_CALL_KEYS
        if re.fullmatch(r"\w+", rest):
            return any(k.startswith(rest) for k in _TP_CALL_KEYS)
        return False
    if "pythonic" in enabled and t.startswith("["):
        return bool(_TP_PY_PREFIX.match(t) or _TP_PY_HEAD.match(t))
    return False


class ToolCallStreamDetector:
    """Find tool calls in a token stream as it arrives.

    feed(chunk) returns a list of events (dicts with a "type"):
      text        {"text"}: ordinary answer text, safe to show
      call_start  {"index", "format", "name": None}: a call has begun
      args_delta  {"index", "delta"}: raw call text as it arrives
      call_end    {"index", "call", "name", "args"}: a complete, parsed call
      call_abort  {"index", "raw", "reason"}: it looked like a call and was not
                  (show `raw` as text; nothing is lost)
    flush() releases what is held back when the stream ends; a call still
    open then is parsed as final unless flush(truncated=True) says the reply
    hit its budget. A canonical JSON call that only appears after prose
    ("Sure!\n```json {...}```") is reported at flush with "late": True,
    because by then its text has already been shown.

    Amortised O(n): each chunk is scanned once for markers (plus a tail no
    longer than the longest marker), a close token is searched for only in
    new text, and JSON bracket depth is tracked incrementally.
    """

    def __init__(self, known: Any = None, *, formats: Any = None,
                 repair: bool | None = None) -> None:
        fmt, rep = tool_call_policy()
        self.known = known
        self.formats = sorted(_tp_enabled(fmt if formats is None else formats))
        self.repair = rep if repair is None else bool(repair)
        self._markers = _tp_stream_markers(set(self.formats))
        self._openers = [m[0] for m in self._markers]
        self._mode = "probe"
        self._pending = ""
        self._fence_pending = False
        self._index = -1
        self._calls = 0
        self._text_parts: list[str] = []
        self._reset_call()

    def _reset_call(self) -> None:
        # The call's text as a list of chunks plus its length: `str +=` on an
        # attribute copies the whole buffer per token (CPython only resizes
        # in place for locals), which is quadratic on a long create_document.
        self._parts: list[str] = []
        self._raw_len = 0
        self._tail = ""
        self._open_len = 0
        self._close: str | None = None
        self._kind = ""
        self._bracket = False
        self._depth = 0
        self._in_str = ""
        self._esc = False

    # -- events --------------------------------------------------------- #
    def _text(self, events: list, s: str) -> None:
        if s:
            events.append({"type": "text", "text": s})
            self._text_parts.append(s)

    def _open(self, events: list, kind: str, close: str | None, open_len: int,
              bracket: bool) -> None:
        self._reset_call()
        self._mode = "call"
        self._kind = kind
        self._close = close
        self._open_len = open_len
        self._bracket = bracket
        self._index += 1
        events.append({"type": "call_start", "index": self._index, "format": kind,
                       "name": None})

    def _finish(self, events: list, block: str, final: bool, reason: str) -> None:
        calls = parse_tool_calls(block, self.known, formats=self.formats,
                                 repair=self.repair, final=final)
        if calls:
            for k, call in enumerate(calls):
                if k:
                    self._index += 1
                    events.append({"type": "call_start", "index": self._index,
                                   "format": call.format, "name": call.name})
                events.append({"type": "call_end", "index": self._index, "call": call,
                               "name": call.name, "args": call.args})
            self._calls += len(calls)
        else:
            events.append({"type": "call_abort", "index": self._index, "raw": block,
                           "reason": reason})

    # -- scanning ------------------------------------------------------- #
    def _raw_text(self) -> str:
        if len(self._parts) > 1:
            self._parts = ["".join(self._parts)]
        return self._parts[0] if self._parts else ""

    def _scan_brackets(self, s: str, base: int) -> int:
        """Scan only the new chunk `s` (at absolute offset `base`); returns the
        absolute end of the balanced call, or -1."""
        i = 0
        pythonic = self._kind == "pythonic"
        while i < len(s):
            c = s[i]
            if self._in_str:
                if self._esc:
                    self._esc = False
                elif c == "\\":
                    self._esc = True
                elif c == self._in_str:
                    self._in_str = ""
            elif c == '"' or (pythonic and c == "'"):
                self._in_str = c
            elif c in "{[":
                self._depth += 1
            elif c in "}]":
                self._depth -= 1
                if self._depth <= 0:
                    return base + i + 1
            i += 1
        return -1

    def _call_feed(self, data: str, events: list) -> str:
        start = self._raw_len
        self._parts.append(data)
        self._raw_len += len(data)
        close_at = -1
        if self._bracket:
            close_at = self._scan_brackets(data, start)
        else:
            ends = (self._close,) if self._close else (
                _TP_PYTAG_ENDS if self._kind == "llama3" else ())
            # Search the new chunk plus a short tail of the old text (a close
            # token split across two chunks), never the whole buffer again.
            window_at = start - len(self._tail)
            window = self._tail + data
            for end in ends:
                lo = max(self._open_len, start - len(end) + 1) - window_at
                j = window.find(end, max(0, lo))
                if j >= 0:
                    close_at = window_at + j + len(end)
                    break
            self._tail = window[-64:]
        upto = self._raw_len if close_at < 0 else close_at
        delta = data[max(0, self._open_len - start):upto - start]
        if delta:
            events.append({"type": "args_delta", "index": self._index, "delta": delta})
        if close_at < 0:
            return ""
        raw = self._raw_text()
        block, leftover = raw[:close_at], raw[close_at:]
        fenced = self._kind == "json" and block.lstrip().startswith("```")
        self._finish(events, block, True, "unparseable")
        self._reset_call()
        self._mode = "probe"
        self._fence_pending = fenced
        return leftover

    def feed(self, chunk: str) -> list[dict]:
        events: list[dict] = []
        data = self._pending + (chunk or "")
        self._pending = ""
        while data:
            if self._mode == "call":
                data = self._call_feed(data, events)
                continue
            if self._fence_pending:
                stripped = data.lstrip()
                if stripped.startswith("```"):
                    data = stripped[3:]
                    self._fence_pending = False
                    continue
                if not stripped or "```".startswith(stripped):
                    self._pending = data
                    return events
                self._fence_pending = False
            if self._mode == "probe":
                stripped = data.lstrip()
                if not stripped:
                    self._pending = data
                    return events
                if _TP_JSON_CALL_HEAD.match(stripped):
                    self._text(events, data[:len(data) - len(stripped)])
                    self._open(events, "json", None, 0, True)
                    data = stripped
                    continue
                if "pythonic" in self.formats and _TP_PY_HEAD.match(stripped):
                    self._text(events, data[:len(data) - len(stripped)])
                    self._open(events, "pythonic", None, 0, True)
                    data = stripped
                    continue
                if looks_like_call_prefix(stripped, self.formats) and not any(
                        stripped.startswith(o) for o in self._openers):
                    self._pending = data
                    return events
                self._mode = "text"
            # text mode: earliest marker wins
            best, best_marker = -1, None
            for marker in self._markers:
                j = data.find(marker[0])
                if j >= 0 and (best < 0 or j < best):
                    best, best_marker = j, marker
            if best_marker is None:
                keep = _tp_prefix_tail(data, self._openers)
                self._text(events, data[:len(data) - keep])
                self._pending = data[len(data) - keep:] if keep else ""
                return events
            self._text(events, data[:best])
            opener, close, kind = best_marker
            self._open(events, kind, close, len(opener), False)
            data = data[best:]
        return events

    def flush(self, truncated: bool = False) -> list[dict]:
        events: list[dict] = []
        if self._mode == "call":
            block = self._raw_text()
            self._reset_call()
            self._mode = "text"
            self._finish(events, block, not truncated, "unterminated")
        held, self._pending = self._pending, ""
        if held and not (self._fence_pending and "```".startswith(held.strip())):
            self._text(events, held)
        self._fence_pending = False
        if not self._calls and self._text_parts:
            late = parse_tool_calls("".join(self._text_parts), self.known,
                                    formats=self.formats, repair=self.repair,
                                    final=not truncated)
            for call in late:
                self._index += 1
                events.append({"type": "call_start", "index": self._index,
                               "format": call.format, "name": call.name, "late": True})
                events.append({"type": "call_end", "index": self._index, "call": call,
                               "name": call.name, "args": call.args, "late": True})
            self._calls += len(late)
        return events


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'DS_TOOL_CALLS_BEGIN',
    'DS_TOOL_CALLS_END',
    'DS_TOOL_CALL_BEGIN',
    'DS_TOOL_CALL_END',
    'HARMONY_ANALYSIS_OPEN',
    'HARMONY_END',
    'HARMONY_NOISE',
    'OPEN_THINK',
    'ParsedToolCall',
    'REASONING_TAGS',
    'THINK_BLOCK',
    'TOOL_CALL_FORMATS',
    'ToolCallStreamDetector',
    '_TP_NOISE_RE',
    '_TP_POLICY',
    '_TP_TAG_ALT',
    '_TP_ZW_RE',
    '_tp_args',
    '_tp_decode_one',
    '_tp_enabled',
    '_tp_prefix_tail',
    'bind_tool_call_config',
    'is_empty_tool_wrapper',
    'looks_like_call_prefix',
    'normalise_tool_call_format',
    'parse_tool_calls',
    'repair_json',
    'strip_reasoning',
    'tool_call_policy',
    'tool_call_unterminated',
    'tool_json_candidates',
]
