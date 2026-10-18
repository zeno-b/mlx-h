"""Prompt assembly, reasoning splitting and tool-call parsing.

Split out of the original single-file deploy.py. The tool-call grammar itself
(every model family's format, JSON repair, streaming detection) and the
reasoning-block primitives live in toolparse.py; this module keeps the
historical entry points (parse_tool_call, strip_reasoning, ThinkSplitter) so
every caller and test keeps its import.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Sequence

from .core import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .tools import *  # noqa: F401,F403
from .toolparse import *  # noqa: F401,F403


def note_prefix(prompt: str, lane: str = "tools") -> None:
    """Track the cached prompt prefix, per lane.

    There are two stable prefixes, not one: the tool-calling prompt and the
    prose-only prompt used on an answer-routed turn. Keying the check per lane
    means ordinary alternation between them is not reported as the prefix
    changing (and does not bump a generation counter every turn) -- a real
    change is an edit to the system prompt or the tool set within one lane.
    """
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    lanes = PREFIX_STATE.setdefault("lanes", {})
    state = lanes.setdefault(lane, {"hash": None, "changed_at": None, "generation": 0})
    if state["hash"] == digest:
        return
    first = state["hash"] is None
    state["hash"] = digest
    state["changed_at"] = iso(utc_now())
    state["generation"] += 1
    if lane == "tools":
        # Mirror the tool lane at the top level: /api telemetry and the UI have
        # read those keys since before there was a second lane.
        PREFIX_STATE["hash"] = state["hash"]
        PREFIX_STATE["changed_at"] = state["changed_at"]
        PREFIX_STATE["generation"] = state["generation"]
    if not first:
        log(
            f"Agent prompt prefix changed ({lane} lane: system prompt or tool set). "
            "Any cached prefix for that lane is now invalid and the next few steps "
            "will re-prefill in full.",
            logging.WARNING,
        )


# NOTE on wording: this used to end "give your final answer", immediately after
# describing a pair of XML tags. A 3B model completed the symmetry and wrapped
# its reply in <final_answer> tags, or labelled it "Final answer:" -- measured at
# roughly one reply in eight. Saying "the answer itself" and forbidding a label
# costs a few tokens and removes the invitation. unwrap_answer still cleans the
# output, because prompt wording is a suggestion, not a guarantee.
REASONING_INSTRUCTION = (
    "\n\nThink before you answer. Begin your reply with your reasoning enclosed in "
    "<think> and </think> tags: work through the problem step by step, consider "
    "edge cases, and question your assumptions there. After the closing </think> "
    "tag, give the answer itself (or, if you are calling a tool, the single JSON "
    "object) with no heading, no label and no tags around it. Put ONLY reasoning "
    "inside the tags and never the tool JSON."
)


# The same instruction for the prose lane, with the tool-call clause removed:
# mentioning a "single JSON object" at the very end of the prompt is the most
# available thing in the model's context when it starts writing.
PROSE_REASONING_INSTRUCTION = (
    "\n\nThink before you answer. Begin your reply with your reasoning enclosed in "
    "<think> and </think> tags: work through the problem step by step, consider "
    "edge cases, and question your assumptions there. After the closing </think> "
    "tag, give the answer itself in plain text, with no heading, no label and no "
    "tags around it. Put ONLY reasoning inside the tags."
)


SKILL_PREAMBLE = (
    "\n\nYOUR SKILLS (name and summary only). Call load_skill with a name to read "
    "the procedure BEFORE doing that kind of task; do not guess one you have a "
    "skill for.\n"
)


def project_block(registry: ToolRegistry) -> str:
    """Project instruction files, or "". Safe to call with anything."""
    try:
        return registry.project_context()
    except Exception as exc:      # a hostile or unreadable file cannot break a turn
        log(f"Project context unavailable: {exc}", logging.WARNING)
        return ""


def skills_block(registry: ToolRegistry) -> str:
    """The catalogue lines for the prompt, or "" when there is nothing to add.

    This is the visible half of progressive disclosure and the only part the
    prompt pays for. Returns "" on a fresh install, so a machine with no skills
    spends not one token on the feature.

    Takes no query on purpose: the block is part of the system prompt, and a
    block that varied with the message would change the prefix every turn and
    cost a full re-prefill. Relevance selection lives in search_skills, where it
    is paid for only when used.
    """
    config = getattr(registry, "config", None)
    library = getattr(registry, "skills", None)
    if library is None or config is None or not getattr(config, "skills_enabled", False):
        return ""
    if "load_skill" not in registry.names():
        return ""          # tools disabled: advertising skills would be a lie
    try:
        catalogue = library.catalogue(limit=config.skills_in_prompt)
        hidden = library.overflowing(config.skills_in_prompt)
    except Exception as exc:      # a broken skill file must never break a turn
        log(f"Skill catalogue unavailable: {exc}", logging.WARNING)
        return ""
    if not catalogue:
        return ""
    block = SKILL_PREAMBLE + catalogue
    if hidden:
        block += (f"\n({hidden} more not listed: use search_skills to find them.)")
    return block


# A tool call the model ran out of reply budget partway through. The JSON never
# closes, so parse_tool_call refuses it -- and the loop then hands two thousand
# characters of half-written document to the user as though it were the answer.
def is_truncated_tool_call(text: str, known: "set[str] | None" = None) -> bool:
    """True when `text` is the start of a tool call that never finished.

    The canonical JSON rule is unchanged; the special-token formats (an
    unclosed <tool_call>, <function=...>, [TOOL_CALLS] [...) are recognised
    too, because a Qwen3-Coder cut off mid-<parameter> is the same failure.
    """
    body = strip_reasoning(text or "").strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z]*\s*", "", body)
    if not body.startswith("{"):
        return tool_call_unterminated(body, known)
    if body.rstrip().endswith("}") and body.count("{") == body.count("}"):
        return False                # balanced: it failed to parse for some other reason
    if not re.search(r'"(?:tool|action|name)"\s*:', body):
        return False
    if known:
        return any(f'"{name}"' in body for name in known)
    return True


def build_agent_system_prompt(base_prompt: str, registry: ToolRegistry,
                              reasoning: bool = False,
                              only: Sequence[str] | None = None) -> str:
    """The tool-lane prompt: the protocol plus one spec per tool.

    `only` narrows the catalogue to the tools a turn can actually use. Every
    spec is ~65 tokens and the full list is ~2200 of a 4096-token window, so a
    document turn that can only ever call create_document was spending 1600
    tokens describing twenty-one tools it would not touch -- and then had barely
    a thousand tokens left to write the document itself into a tool call. It got
    cut off mid-JSON, the call never parsed, and the turn produced no file.
    """
    lines = []
    specs = registry.specs()
    if only:
        wanted = set(only)
        narrowed = [t for t in specs if t["name"] in wanted]
        # Never narrow to nothing: an empty catalogue with the protocol still in
        # front of it is the worst of both, a model told to reply with a tool
        # call and given no tool to name.
        if narrowed:
            specs = narrowed
    for tool in specs:
        params = ", ".join(
            f"{name} ({desc})" for name, desc in tool["parameters"].items()
        ) or "no arguments"
        required = ", ".join(tool["required"]) or "none"
        lines.append(f"- {tool['name']}: {tool['description']}\n  args: {params}\n  required: {required}")
    prompt = f"{base_prompt}\n\n{TOOL_PROTOCOL}" + "\n".join(lines)
    prompt += project_block(registry)
    prompt += skills_block(registry)
    if reasoning:
        prompt += REASONING_INSTRUCTION
    # A narrowed catalogue is its own cached prefix. Reporting it on the "tools"
    # lane would log "the tool set changed" every time a document turn followed
    # an ordinary one, which is routine alternation and not the edit that
    # warning exists to catch.
    note_prefix(prompt, lane="tools-subset" if only else "tools")
    return prompt


def build_plain_system_prompt(base_prompt: str, reasoning: bool = False,
                              registry: ToolRegistry | None = None) -> str:
    """The prompt for a turn that will not call a tool.

    The tool protocol plus every tool spec is ~1500 tokens of a 4096-token
    window, and its first instruction is "reply with a single JSON object and
    nothing else". Once routing has decided the turn needs no tool, that text
    only costs context and invites the failure it describes: a request to write
    a C++ program answered with "{}". This lane says nothing about tools at all.
    """
    prompt = base_prompt
    # Skills belong on THIS lane most of all. The prose lane is where the model
    # answers from its own knowledge, which is exactly where a written-down
    # procedure beats a 3B model's recall. The registry is optional so callers
    # that have no tools at all still work.
    if registry is not None:
        prompt += project_block(registry)
        block = skills_block(registry)
        if block:
            # One extra sentence, because this lane is told nothing about tools:
            # without it the model has a list it has no way to act on.
            prompt += block + (
                "\nTo read one, reply with exactly this and nothing else: "
                '{"tool": "load_skill", "args": {"name": "<skill-name>"}}'
            )
    if reasoning:
        prompt += PROSE_REASONING_INSTRUCTION
    note_prefix(prompt, lane="prose")
    return prompt


# What a small model writes when it means "no". A field read with `is not
# False` only ever sees a real JSON boolean, and a 3B asked for one writes
# "false", "no" or 0 about as often as it writes false -- so a router that
# correctly reported it did not know the answer was read as saying it did, and
# the whole signal quietly did nothing while looking like it worked.
_MODEL_NO = {"false", "no", "n", "0", "nope", "unsure", "not sure", "unknown",
             "none", "null", "negative"}


def said_no(value) -> bool:
    """True when a model-emitted field clearly means no.

    Only a CLEAR no counts. Anything absent, empty, unparseable or merely odd
    reads as "not a no", because every caller here uses this to decide whether
    to spend something -- a search, a step, a generation -- and a field the
    model simply did not fill in must never be the reason it gets spent.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value is False
    if isinstance(value, (int, float)):
        return value == 0
    if isinstance(value, str):
        return value.strip().strip('"\'').lower() in _MODEL_NO
    return False


def extract_json_object(text: str) -> dict | None:
    """Pull the first balanced JSON object out of a model reply.

    Small models wrap the object in prose or a code fence at random, so scanning
    for balanced braces beats expecting a clean response. The scan is shared
    with the tool-call parser (tool_json_candidates) so the two can never
    disagree about which object is "the reply's JSON".
    """
    if not text:
        return None
    candidates, _open = tool_json_candidates(text)
    for _start, candidate, _fenced in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


# THINK_BLOCK, OPEN_THINK and strip_reasoning are defined in toolparse.py (the
# parser must strip exactly what this splitter hides) and re-exported here.
_THINK_NOISE_ALT = "|".join(re.escape(n) for n in HARMONY_NOISE)


class ThinkSplitter:
    """Streaming splitter that separates a model's <think> reasoning from its
    answer as tokens arrive.

    A reasoning model emits <think>...</think> before its answer. We want the
    reasoning shown live in its own visible "thinking" area, not stripped and
    hidden and not dumped raw into the answer. Tokens can split a tag across
    chunk boundaries, so a partial tag at the end of a feed is held over to the
    next one. feed() returns a list of ("think"|"answer", text) pieces.

    The tag set is THINK_BLOCK's, case-insensitively, and that is not a detail:
    a literal "<think>" match recognised only three of the four spellings this
    module already strips. On <thinking>, <reasoning> or <THINK> the whole
    reasoning block streamed into the ANSWER bubble, tags and all, and the
    final event then replaced the bubble with the cleaned text -- which is
    precisely the whole-message rewrite this class exists to eliminate,
    reintroduced for every spelling but one.

    The same holds for the newer spellings: <seed:think> (Seed-OSS) and the
    gpt-oss harmony analysis channel (<|channel|>analysis<|message|> ... <|end|>),
    whose final-channel header and <|return|> are protocol and are dropped
    from the answer exactly as strip_reasoning drops them.

    `opened=True` starts inside a reasoning block, for a caller that knows the
    chat template opened <think> in the prompt. Without it, a bare leading
    </think> can only be dropped: the text before it has already streamed as
    answer, and strip_reasoning (which sees the whole reply) removes it when
    the final event lands.
    """

    TAGS = REASONING_TAGS
    OPEN = "<think>"
    CLOSE = "</think>"
    _OPEN_RE = re.compile(rf"(?i)<({_TP_TAG_ALT})>")
    # No complete tag is a prefix of another ("<think>" and "<thinking>" differ
    # at the 7th character), so a complete match is never ambiguous and only
    # INCOMPLETE tails have to be held back.
    _OPEN_TAGS = tuple(f"<{t}>" for t in REASONING_TAGS)
    _CLOSE_TAGS = tuple(f"</{t}>" for t in REASONING_TAGS)
    _ANSWER_RE = re.compile(
        rf"(?i)<(?P<tag>{_TP_TAG_ALT})>|(?P<harmony><\|channel\|>analysis<\|message\|>)"
        rf"|(?P<noise>{_THINK_NOISE_ALT})|(?P<close></(?:{_TP_TAG_ALT})>)")

    def __init__(self, opened: bool = False) -> None:
        self.in_think = bool(opened)
        self.carry = ""
        # Which spelling opened the current block, so the close has to match it
        # exactly as THINK_BLOCK's backreference does. Closing on any of the
        # three would split "<think>x</reasoning>y" differently from
        # strip_reasoning, and the two must not disagree about the same text.
        self.tag = "think" if opened else ""
        # Whether any reasoning tag has been seen. Only the FIRST tag can be a
        # bare close (strip_reasoning's rule); a stray close tag after that is
        # ordinary answer text in both places.
        self.seen = bool(opened)

    @staticmethod
    def _prefix_tail(data: str, tags) -> int:
        """Length of a trailing slice of data that is a proper prefix of a tag.

        So "...</thi" holds back 4 chars until the rest of </think> arrives.
        Case-insensitive, and takes the whole candidate set: a tail that could
        still grow into ANY recognised tag has to wait.
        """
        if isinstance(tags, str):
            tags = (tags,)
        low = [t.lower() for t in tags]
        longest = max((len(t) for t in low), default=0)
        for size in range(min(longest - 1, len(data)), 0, -1):
            tail = data[-size:].lower()
            if any(t.startswith(tail) for t in low):
                return size
        return 0

    def _close_tag(self) -> str:
        if self.tag == "harmony":
            return HARMONY_END
        return f"</{self.tag or 'think'}>"

    def _answer_holds(self) -> tuple:
        holds = self._OPEN_TAGS + (HARMONY_ANALYSIS_OPEN,) + HARMONY_NOISE
        return holds if self.seen else holds + self._CLOSE_TAGS

    def _next_marker(self, data: str):
        for m in self._ANSWER_RE.finditer(data):
            if m.group("close") and self.seen:
                continue            # a stray close tag is answer text
            return m
        return None

    @staticmethod
    def _find_ci(data: str, needle: str) -> int:
        """Index of needle in data, ignoring case, in DATA's own offsets.

        Not data.lower().find(): lower() is not length-preserving in Unicode --
        "İ".lower() is two characters -- so every index after one of those
        would be off by one and the split would cut the answer in the wrong
        place. re.search reports offsets into the original string.
        """
        m = re.search(re.escape(needle), data, re.I)
        return m.start() if m else -1

    def feed(self, text: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        data = self.carry + (text or "")
        self.carry = ""
        while data:
            if self.in_think:
                close = self._close_tag()
                end = self._find_ci(data, close)
                if end == -1:
                    keep = self._prefix_tail(data, close)
                    if keep:
                        self.carry = data[len(data) - keep:]
                        data = data[:len(data) - keep]
                    if data:
                        out.append(("think", data))
                    data = ""
                else:
                    if end:
                        out.append(("think", data[:end]))
                    data = data[end + len(close):]
                    self.in_think = False
                    self.tag = ""
            else:
                m = self._next_marker(data)
                if m is None:
                    keep = self._prefix_tail(data, self._answer_holds())
                    if keep:
                        self.carry = data[len(data) - keep:]
                        data = data[:len(data) - keep]
                    if data:
                        out.append(("answer", data))
                    data = ""
                else:
                    if m.start():
                        out.append(("answer", data[:m.start()]))
                    data = data[m.end():]
                    if m.group("tag"):
                        self.in_think = True
                        self.tag = m.group("tag").lower()
                        self.seen = True
                    elif m.group("harmony"):
                        self.in_think = True
                        self.tag = "harmony"
                        self.seen = True
                    elif m.group("close"):
                        self.seen = True      # bare leading close: dropped
                    # noise (harmony final header, <|return|>): dropped
        return out

    def flush(self) -> list[tuple[str, str]]:
        """Release whatever is still held back. Call once, when the stream ends.

        feed() holds back a trailing slice that is a proper prefix of a tag, so
        "...</thi" waits for the rest of </think> rather than being emitted as
        answer text. Nothing ever arrives after the last token, so without this
        that tail is simply dropped: an answer ending in "<", "<t" or "a < b"
        loses its last few characters from the streamed view. It is corrected
        when the final event lands, which is exactly the whole-message rewrite
        the streaming split exists to avoid -- so it has to be flushed here
        instead.
        """
        held, self.carry = self.carry, ""
        if not held:
            return []
        return [("think" if self.in_think else "answer", held)]


def parse_tool_call(text: str, known: set[str] | None = None) -> tuple[str, dict] | None:
    """Return (tool name, args) if the reply is a tool call, else None.

    `known` is the set of registered tool names. When it is supplied, a JSON
    object whose name is not a real tool is only treated as a call if it used
    the explicit "tool" key. That stops a final answer that happens to contain
    JSON (a config snippet, a parsed record) from being executed as a tool call.

    Delegates to toolparse.parse_tool_calls (every model family's format plus
    JSON repair) and returns the first call; parse_tool_calls returns them all.
    """
    calls = parse_tool_calls(text, known)
    if not calls:
        return None
    return calls[0].name, calls[0].args


# Keys that only ever belong to the tool-call protocol. A reply made up of
# nothing but these (or of nothing at all) is an unfilled tool call.
_CALL_ONLY_KEYS = frozenset(
    {"tool", "tool_name", "name", "args", "arguments", "parameters", "action"})


def is_degenerate_tool_call(text: str) -> bool:
    """True if the reply is a malformed tool call rather than an answer.

    "{}" -- or {"args": {}}, or {"action": "answer"} echoed back from the router
    -- is a tool call the model never filled in. parse_tool_call rightly refuses
    to execute it, and the loop then treated the raw text as the reply: a request
    for a C++ program that arrived as "{}". A JSON object carrying real content
    keys is somebody asking for JSON, so it is left alone.
    """
    visible = strip_reasoning(text or "").strip()
    if not visible:
        return False
    # The special-token spelling of the same failure: <tool_call></tool_call>,
    # [TOOL_CALLS][] or a bare <|python_tag|>{} is a call the model never filled.
    if is_empty_tool_wrapper(visible):
        return True
    wrapped = re.fullmatch(r"<tool_call>\s*(\{.*\})\s*</tool_call>", visible, re.S)
    if wrapped:
        visible = wrapped.group(1).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", visible, re.S)
    if fenced:
        visible = fenced.group(1).strip()
    if not (visible.startswith("{") and visible.endswith("}")):
        return False
    try:
        parsed = json.loads(visible)     # the WHOLE reply must be one object
    except Exception:
        return False
    if not isinstance(parsed, dict):
        return False
    if parse_tool_call(visible) is not None:
        return False                     # a real call: the caller decides
    return all(str(k).lower() in _CALL_ONLY_KEYS for k in parsed)


@dataclass
class GenerationStats:
    """What one model call actually cost.

    prompt_tokens is the number that matters most on this hardware: an agent
    re-sends its whole prompt every step, so prefill, not decode, is where a
    multi-step run spends its time. Watching this climb step over step within a
    single run is how you see the quadratic.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    from_server: bool = False
    # Why generation stopped: "stop" (the model finished) or "length" (it hit the
    # reply budget and was CUT OFF). Nothing read this before, so a truncated
    # answer -- a program ending mid-function -- was shown as if it were complete.
    finish_reason: str = ""
    # Prompt tokens the server answered from its prompt cache
    # (usage.prompt_tokens_details.cached_tokens). prompt_tokens minus this is
    # what the step actually prefilled, which is the number that shows whether
    # the stable-prefix work is paying off.
    cached_tokens: int = 0
    # Tokens the model spent reasoning (usage.completion_tokens_details, or an
    # estimate from the reasoning text the server returned separately).
    reasoning_tokens: int = 0
    # Time the request waited for a generation slot before it was sent.
    queue_ms: float = 0.0
    # False when no first-token time exists (a non-streaming call): ttft_ms is
    # then the whole call and total - ttft is zero, which used to report a
    # decode rate of completion_tokens * 1000 tok/s.
    ttft_measured: bool = True

    @property
    def decode_tps(self) -> float:
        if self.ttft_measured:
            decode_ms = max(1.0, self.total_ms - self.ttft_ms)
        else:
            # No first token to separate prefill from decode: the whole call is
            # the honest denominator, so this is a lower bound (estimated).
            decode_ms = max(1.0, self.total_ms)
        return self.completion_tokens / (decode_ms / 1000.0)

    def as_event(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ttft_ms": round(self.ttft_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "decode_tps": round(self.decode_tps, 2),
            "estimated": not self.from_server,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "queue_ms": round(self.queue_ms, 1),
            "decode_tps_estimated": not self.ttft_measured,
        }


_CONNECTIVITY: dict = {"online": True, "checked_at": 0.0}


async def has_internet(recheck_after: float = 30.0) -> bool:
    """Best-effort connectivity check, cached briefly.

    Used to decide whether lookups are even possible. When offline the agent
    answers from its own knowledge instead of attempting a search that would
    fail slowly. The result is cached for recheck_after seconds so it costs at
    most one tiny request per window. Any failure is read as offline.
    """
    now = time.time()
    if now - _CONNECTIVITY["checked_at"] < recheck_after:
        return _CONNECTIVITY["online"]
    online = False
    try:
        import httpx
        # A couple of reliable, lightweight endpoints; success on either is enough.
        async with httpx.AsyncClient(timeout=3.0) as client:
            for url in ("https://1.1.1.1", "https://dns.google"):
                try:
                    resp = await client.head(url)
                    if resp.status_code < 500:
                        online = True
                        break
                except Exception:
                    continue
    except Exception:
        online = False
    _CONNECTIVITY.update(online=online, checked_at=now)
    return online



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'is_truncated_tool_call',
    'SKILL_PREAMBLE',
    'project_block',
    'skills_block',
    'GenerationStats',
    'OPEN_THINK',
    'REASONING_INSTRUCTION',
    'THINK_BLOCK',
    'ThinkSplitter',
    '_CONNECTIVITY',
    'build_agent_system_prompt',
    'extract_json_object',
    'said_no',
    'is_degenerate_tool_call',
    '_CALL_ONLY_KEYS',
    'build_plain_system_prompt',
    'PROSE_REASONING_INSTRUCTION',
    'has_internet',
    'note_prefix',
    'parse_tool_call',
    'strip_reasoning',
    # From toolparse, re-exported so `from .llm import *` keeps reaching them.
    'ParsedToolCall',
    'ToolCallStreamDetector',
    'bind_tool_call_config',
    'looks_like_call_prefix',
    'parse_tool_calls',
    'repair_json',
    'tool_call_policy',
    '_THINK_NOISE_ALT',
]
