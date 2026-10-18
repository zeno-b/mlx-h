"""Anthropic-compatible API: POST /v1/messages and /v1/messages/count_tokens.

Claude Code, and anything else built on the Anthropic SDK, can point
ANTHROPIC_BASE_URL at this server. The request is translated into the same
canonical OpenAI-shaped request openai_compat.py runs (one engine, one set of
modes and model-naming rules, one admission path), and the result is rendered
back as Anthropic content blocks: text, tool_use (the model's calls for the
CLIENT to run; raw mode never executes them here) and thinking, the last only
when the request turned thinking on, because a client that did not ask for
thinking blocks has nowhere to put them.

Translation rules worth knowing:
  * tool_result blocks (which Anthropic requires first in a user message)
    become tool messages ahead of that message's own text, so the
    prompt-protocol path renders them as "TOOL RESULT [name]:" and the name is
    recovered from the matching tool_use earlier in the conversation;
  * thinking and redacted_thinking blocks the client echoes back are dropped:
    a local model never produced a signature worth verifying, and re-feeding
    old reasoning only costs context;
  * Anthropic-defined tools (a "type" such as bash_20250124 with no
    input_schema) are skipped: they are executed by Anthropic's own platform
    contract, not described to a model;
  * usage follows Anthropic's split: input_tokens excludes the prompt-cache hit,
    which is reported as cache_read_input_tokens.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .openai_compat import *  # noqa: F401,F403


_ANT_STOP_REASONS = {"tool_calls": "tool_use", "length": "max_tokens",
                     "stop_sequence": "stop_sequence"}


# --------------------------------------------------------------------------- #
# Request translation
# --------------------------------------------------------------------------- #
def _ant_blocks(content: Any, where: str) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        raise CompatError(400, f"{where}.content must be a string or a list of blocks")
    for block in content:
        if not isinstance(block, dict) or not isinstance(block.get("type"), str):
            raise CompatError(400, f"{where}.content: every block needs a type")
    return content


def _ant_image_part(block: dict, where: str) -> dict:
    source = block.get("source") if isinstance(block.get("source"), dict) else {}
    if source.get("type") == "base64" and source.get("data"):
        url = f"data:{source.get('media_type') or 'image/png'};base64,{source['data']}"
    elif source.get("type") == "url" and source.get("url"):
        url = str(source["url"])
    else:
        raise CompatError(400, f"{where}: image source must be base64 or url")
    return {"type": "image_url", "image_url": {"url": url}}


def _ant_result_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, dict) and block.get("type") == "image":
                parts.append("[image]")
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False)


def _ant_tool(tool: Any, i: int) -> dict | None:
    if not isinstance(tool, dict):
        raise CompatError(400, f"tools[{i}] must be an object")
    if tool.get("type") not in (None, "custom") and "input_schema" not in tool:
        return None
    schema = tool.get("input_schema")
    if schema is not None and not isinstance(schema, dict):
        raise CompatError(400, f"tools[{i}].input_schema must be an object")
    return _oai_tool_from_function({"name": tool.get("name"),
                                    "description": tool.get("description") or "",
                                    "parameters": schema}, f"tools[{i}]")


def compat_parse_messages(body: dict, *, require_max_tokens: bool = True) -> CompatRequest:
    """An Anthropic Messages body as a CompatRequest, or a 400."""
    max_tokens = body.get("max_tokens")
    if require_max_tokens and max_tokens is None:
        raise CompatError(400, "max_tokens: Field required")
    messages: list[dict] = []
    system = body.get("system")
    if system:
        text = (system if isinstance(system, str) else
                "\n\n".join(str(b.get("text") or "") for b in _ant_blocks(system, "system")
                            if b.get("type") == "text"))
        if text.strip():
            messages.append({"role": "system", "content": text})
    raw = body.get("messages")
    if not isinstance(raw, list) or not raw:
        raise CompatError(400, "messages: must be a non-empty list")
    for i, msg in enumerate(raw):
        where = f"messages[{i}]"
        if not isinstance(msg, dict) or msg.get("role") not in ("user", "assistant"):
            raise CompatError(400, f"{where}.role must be user or assistant")
        blocks = _ant_blocks(msg.get("content"), where)
        if msg["role"] == "user":
            parts: list[dict] = []
            for block in blocks:
                kind = block["type"]
                if kind == "tool_result":
                    text = _ant_result_text(block.get("content"))
                    if block.get("is_error"):
                        text = f"Error: {text}"
                    messages.append({"role": "tool",
                                     "tool_call_id": str(block.get("tool_use_id") or ""),
                                     "content": text})
                elif kind == "text":
                    parts.append({"type": "text", "text": str(block.get("text") or "")})
                elif kind == "image":
                    parts.append(_ant_image_part(block, where))
                elif kind == "document":
                    source = block.get("source") if isinstance(block.get("source"), dict) else {}
                    if source.get("type") == "text":
                        parts.append({"type": "text", "text": str(source.get("data") or "")})
                    else:
                        raise CompatError(400, f"{where}: only text documents are supported")
            if parts:
                text_only = all(p["type"] == "text" for p in parts)
                messages.append({"role": "user", "content": "".join(
                    p["text"] for p in parts) if text_only else parts})
        else:
            text_parts, calls = [], []
            for block in blocks:
                kind = block["type"]
                if kind == "text":
                    text_parts.append(str(block.get("text") or ""))
                elif kind == "tool_use":
                    name = block.get("name")
                    if not isinstance(name, str) or not name:
                        raise CompatError(400, f"{where}: tool_use needs a name")
                    calls.append({"id": str(block.get("id") or ""), "type": "function",
                                  "function": {"name": name, "arguments": json.dumps(
                                      block.get("input") or {}, ensure_ascii=False)}})
                # thinking / redacted_thinking / server tool blocks: dropped.
            entry: dict = {"role": "assistant", "content": "".join(text_parts) or None}
            if calls:
                entry["tool_calls"] = calls
            messages.append(entry)
    tools = [t for t in (_ant_tool(tool, i) for i, tool in enumerate(body.get("tools") or []))
             if t is not None]
    choice = body.get("tool_choice")
    parallel = None
    if isinstance(choice, dict):
        if choice.get("disable_parallel_tool_use"):
            parallel = False
        kind = choice.get("type")
        if kind == "auto":
            choice = "auto"
        elif kind == "any":
            choice = "required"
        elif kind == "none":
            choice = "none"
        elif kind == "tool":
            choice = {"type": "function", "function": {"name": choice.get("name")}}
        else:
            raise CompatError(400, "tool_choice.type must be auto, any, tool or none")
    elif choice is not None:
        raise CompatError(400, "tool_choice must be an object")
    stops = body.get("stop_sequences")
    if stops is not None and (not isinstance(stops, list)
                              or not all(isinstance(s, str) for s in stops)):
        raise CompatError(400, "stop_sequences must be a list of strings")
    thinking = body.get("thinking") if isinstance(body.get("thinking"), dict) else {}
    thinking_on = thinking.get("type") in ("enabled", "adaptive")
    creq = CompatRequest(
        model=str(body.get("model") or ""), messages=messages, tools=tools,
        tool_choice=_oai_check_tool_choice(choice, tools),
        max_tokens=_oai_num({"max_tokens": max_tokens}, "max_tokens", int, low=1),
        temperature=_oai_num(body, "temperature", float, low=0.0, high=1.0),
        top_p=_oai_num(body, "top_p", float, low=0.0, high=1.0),
        top_k=_oai_num(body, "top_k", int, low=0),
        stop=_oai_stop_list(stops),
        reasoning=thinking_on,
        thinking=True if thinking_on else (False if thinking.get("type") == "disabled"
                                           else None),
        parallel_tool_calls=parallel,
        include_trace=bool(body.get("x_include_trace")),
        stream=bool(body.get("stream")),
        call_prefix="toolu_",
    )
    return creq


# --------------------------------------------------------------------------- #
# Response rendering
# --------------------------------------------------------------------------- #
def _ant_stop_reason(finish: str) -> str:
    return _ANT_STOP_REASONS.get(finish, "end_turn")


def anthropic_usage(usage: dict) -> dict:
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = min(prompt, int(usage.get("cached_tokens") or 0))
    return {"input_tokens": prompt - cached, "output_tokens": int(usage.get("completion_tokens") or 0),
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": cached}


def _ant_prompt_estimate(plan: CompatPlan) -> int:
    if plan.mode == "raw":
        messages = plan.messages
    else:
        messages = [*plan.history, {"role": "user", "content": plan.prompt}]
    return sum(estimate_tokens(m.get("content") or "") + 4 for m in messages)


def anthropic_messages_json(result: CompatResult, creq: CompatRequest, target: CompatTarget,
                            plan: CompatPlan) -> dict:
    content: list[dict] = []
    if result.reasoning and creq.reasoning:
        content.append({"type": "thinking", "thinking": result.reasoning, "signature": ""})
    if result.text or not result.tool_calls:
        content.append({"type": "text", "text": result.text})
    for call in result.tool_calls:
        content.append({"type": "tool_use", "id": call["id"], "name": call["name"],
                        "input": call["args"]})
    body = {"id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant",
            "model": target.name or target.model, "content": content,
            "stop_reason": _ant_stop_reason(result.finish),
            "stop_sequence": result.stop_sequence if result.finish == "stop_sequence" else None,
            "usage": anthropic_usage(result.usage)}
    if plan.mode == "agent":
        body["x_tools_used"] = result.tools_used
        if creq.include_trace:
            body["x_trace"] = result.trace
    return body


def _ant_sse(etype: str, payload: dict) -> str:
    return f"event: {etype}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def anthropic_messages_stream(events: AsyncIterator[dict], creq: CompatRequest,
                                    target: CompatTarget, plan: CompatPlan) -> AsyncIterator[str]:
    """message_start, ping, content blocks with their deltas, message_delta, message_stop."""
    estimate = _ant_prompt_estimate(plan)
    yield _ant_sse("message_start", {"type": "message_start", "message": {
        "id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant",
        "model": target.name or target.model, "content": [], "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": estimate, "output_tokens": 0,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}})
    yield _ant_sse("ping", {"type": "ping"})
    block = {"index": -1, "kind": ""}

    def start(kind: str, content_block: dict) -> list[str]:
        out = stop()
        block["index"] += 1
        block["kind"] = kind
        out.append(_ant_sse("content_block_start", {
            "type": "content_block_start", "index": block["index"],
            "content_block": content_block}))
        return out

    def stop() -> list[str]:
        if not block["kind"]:
            return []
        block["kind"] = ""
        return [_ant_sse("content_block_stop", {"type": "content_block_stop",
                                                "index": block["index"]})]

    def delta(payload: dict) -> str:
        return _ant_sse("content_block_delta", {"type": "content_block_delta",
                                                "index": block["index"], "delta": payload})

    finish, usage, stop_sequence = "stop", None, None
    async for ev in events:
        etype = ev["type"]
        if etype == "text":
            if block["kind"] != "text":
                for f in start("text", {"type": "text", "text": ""}):
                    yield f
            yield delta({"type": "text_delta", "text": ev["text"]})
        elif etype == "reasoning":
            if not creq.reasoning:
                continue
            if block["kind"] != "thinking":
                for f in start("thinking", {"type": "thinking", "thinking": "",
                                            "signature": ""}):
                    yield f
            yield delta({"type": "thinking_delta", "thinking": ev["text"]})
        elif etype == "tool_start":
            for f in start("tool_use", {"type": "tool_use", "id": ev["id"], "name": ev["name"],
                                        "input": {}}):
                yield f
        elif etype == "tool_args":
            if block["kind"] == "tool_use":
                yield delta({"type": "input_json_delta", "partial_json": ev["delta"]})
        elif etype == "tool_end":
            for f in stop():
                yield f
        elif etype == "trace":
            if creq.include_trace:
                yield _ant_sse("x_trace", {"type": "x_trace", "event": ev["event"]})
        elif etype == "error":
            for f in stop():
                yield f
            yield _ant_sse("error", anthropic_error_body(int(ev.get("status") or 502),
                                                         ev["message"]))
            return
        elif etype == "done":
            finish, usage = ev.get("finish") or "stop", ev.get("usage")
            stop_sequence = ev.get("stop_sequence")
    for f in stop():
        yield f
    final_usage = anthropic_usage(usage or {})
    if not usage or not usage.get("prompt_tokens"):
        final_usage["input_tokens"] = estimate
    yield _ant_sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": _ant_stop_reason(finish),
                  "stop_sequence": stop_sequence if finish == "stop_sequence" else None},
        "usage": final_usage})
    yield _ant_sse("message_stop", {"type": "message_stop"})


def anthropic_count_tokens(ctx: Any, creq: CompatRequest) -> int:
    """The prompt the raw engine would send, estimated (no tokenizer loaded).

    Rendered exactly like a raw request (tool section included on the
    prompt-protocol path), because that is what fills the context window.
    Images count at the flat per-image estimate, without the vision check: a
    count is not a generation.
    """
    tools_active = bool(creq.tools) and creq.tool_choice != "none"
    native = tools_active and "native_tool_calls" in compat_backend_caps(ctx)
    messages = compat_render_messages(creq, native=native, vision=True,
                                      tools_active=tools_active)
    total = sum(estimate_tokens(m.get("content") or "") + 4 for m in messages)
    if native:
        total += estimate_tokens(json.dumps(creq.tools, ensure_ascii=False))
    return total


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
def register_anthropic_compat(app: Any, ctx: Any) -> None:
    """POST /v1/messages and POST /v1/messages/count_tokens."""
    from fastapi.responses import JSONResponse

    @app.post("/v1/messages/count_tokens")
    async def v1_messages_count_tokens(request: _OaiRequest):
        """Anthropic token counting: an estimate of the rendered prompt."""
        try:
            await compat_user(ctx, request)
            body = await compat_json_body(request)
            creq = compat_parse_messages(body, require_max_tokens=False)
        except CompatError as exc:
            return compat_error_response(exc, "anthropic")
        return JSONResponse({"input_tokens": anthropic_count_tokens(ctx, creq)})

    @app.post("/v1/messages")
    async def v1_messages(request: _OaiRequest):
        """Anthropic Messages: raw model or this app's agent (see openai_compat)."""
        return await compat_handle(app, ctx, request, provider="anthropic",
                                   endpoint="v1_messages", parse=compat_parse_messages,
                                   to_json=anthropic_messages_json,
                                   to_stream=anthropic_messages_stream)


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'anthropic_count_tokens',
    'anthropic_messages_json',
    'anthropic_messages_stream',
    'anthropic_usage',
    'compat_parse_messages',
    'register_anthropic_compat',
    '_ANT_STOP_REASONS',
    '_ant_blocks',
    '_ant_image_part',
    '_ant_prompt_estimate',
    '_ant_result_text',
    '_ant_sse',
    '_ant_stop_reason',
    '_ant_tool',
]
