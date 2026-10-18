"""OpenAI-compatible API: /v1/models, /v1/chat/completions, /v1/responses.

Also the engine the Anthropic surface (anthropic_compat.py) runs on, because
the two wire formats differ only at the edges: both come down to a list of
chat messages, a set of client tools and a stream of text, reasoning and tool
calls going back.

Two modes, because two very different kinds of client call these routes:

  * RAW mode is for coding clients (Claude Code, Codex, Aider, Continue,
    OpenCode, Hermes). They run their OWN tool loop: the request carries their
    tools, the model's tool calls must come back to them as tool_calls /
    tool_use blocks, and nothing may execute here. Running the app's agent for
    them executed our tools on our machine in place of theirs and handed back
    prose where they waited for a call, so their loop simply stopped.
  * AGENT mode is what /v1/chat/completions always did: this app's agent with
    its server-side tools, capability-gated by an agent profile, checked by the
    approval policy and never waiting for a click (nobody watches an API call,
    so the turn is not interactive and "ask" gets the unattended answer).

The model name picks the mode: "raw/<alias>" and "agent/<profile>" force one,
"agent" is the default agent, and a plain alias or repo id uses
V1_DEFAULT_MODE ("agent" by default, so an existing client that never sent a
model name gets exactly what it got before). Raw mode only accepts a model that
resolves to one actually served here: an answer from a different model than the
one asked for is worse than a refusal.

Raw-mode tool calls take one of two paths. A backend that parses tool calls
itself (the "native_tool_calls" capability) gets the client's tools and
tool_choice in the request body; any other gets them rendered into the system
prompt in the same {"tool": ..., "args": ...} protocol the agent uses, and the
calls are parsed back out of the text by toolparse. Both run the text through
the stream detector, because a server that advertises native parsing still
leaves the call in the content for a tokenizer without a tool parser.

Errors are always provider-shaped (OpenAI {"error": {...}}, Anthropic
{"type": "error", "error": {...}}), including authentication and admission,
because SDKs decide whether to retry by reading that body.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .llm import *  # noqa: F401,F403
from .model_client import *  # noqa: F401,F403
from .catalog import *  # noqa: F401,F403
from .backends import *  # noqa: F401,F403
from .admission import *  # noqa: F401,F403
from .turnctx import *  # noqa: F401,F403
from .streams import *  # noqa: F401,F403

# Resolvable at MODULE scope for the same reason api.py imports it there: with
# postponed annotations FastAPI resolves `request: _OaiRequest` against this
# module's globals. Guarded so --selftest imports without FastAPI.
try:
    from fastapi import Request as _OaiRequest  # noqa: F401
except Exception:  # pragma: no cover
    _OaiRequest = Any  # type: ignore


_oai_log = get_logger("compat")

# A tool description is the client's, and a coding client's are long (Claude
# Code's Bash tool alone is several thousand characters). Rendered into the
# prompt of a 4096-token model in full, the catalogue would not leave room for
# the conversation, so the prompt-protocol path keeps the first sentences only.
_OAI_TOOL_DESC_CHARS = 300
_OAI_PARAM_DESC_CHARS = 80
_OAI_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_.:\-]{1,128}$")


# --------------------------------------------------------------------------- #
# Provider-shaped errors
# --------------------------------------------------------------------------- #
class CompatError(Exception):
    """A refusal with an HTTP status, rendered in the caller's provider shape."""

    def __init__(self, status: int, message: str, *, kind: str = "", code: str | None = None,
                 param: str | None = None, headers: dict | None = None) -> None:
        super().__init__(message)
        self.status = int(status)
        self.message = str(message)
        self.kind = kind
        self.code = code
        self.param = param
        self.headers = dict(headers or {})


_OAI_ERROR_TYPES = {
    400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
    404: "invalid_request_error", 413: "invalid_request_error",
    422: "invalid_request_error", 429: "rate_limit_error", 500: "server_error",
    502: "server_error", 503: "server_error", 504: "server_error",
}
_OAI_ERROR_CODES = {401: "invalid_api_key", 403: "insufficient_permissions",
                    404: "model_not_found", 429: "rate_limit_exceeded", 503: "overloaded"}
_ANT_ERROR_TYPES = {
    400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
    404: "not_found_error", 413: "request_too_large", 422: "invalid_request_error",
    429: "rate_limit_error", 500: "api_error", 502: "api_error", 503: "overloaded_error",
    504: "api_error",
}


def openai_error_body(status: int, message: str, *, kind: str = "", code: str | None = None,
                      param: str | None = None) -> dict:
    return {"error": {"message": message,
                      "type": kind or _OAI_ERROR_TYPES.get(status, "server_error"),
                      "param": param,
                      "code": code if code is not None else _OAI_ERROR_CODES.get(status)}}


def anthropic_error_body(status: int, message: str, *, kind: str = "") -> dict:
    return {"type": "error",
            "error": {"type": kind or _ANT_ERROR_TYPES.get(status, "api_error"),
                      "message": message}}


def compat_error_response(exc: CompatError, provider: str = "openai") -> Any:
    from fastapi.responses import JSONResponse
    if provider == "anthropic":
        body = anthropic_error_body(exc.status, exc.message,
                                    kind=exc.kind if exc.kind in _ANT_ERROR_TYPES.values() else "")
    else:
        body = openai_error_body(exc.status, exc.message, kind=exc.kind if exc.kind in
                                 _OAI_ERROR_TYPES.values() else "", code=exc.code,
                                 param=exc.param)
    return JSONResponse(body, status_code=exc.status, headers=exc.headers or None)


# --------------------------------------------------------------------------- #
# Request plumbing shared by every /v1 route
# --------------------------------------------------------------------------- #
async def compat_user(ctx: Any, request: Any) -> dict:
    """The caller, through the app's own auth dependency, or a CompatError.

    Calling the dependency (instead of declaring it) is what makes a 401/403 a
    provider-shaped body: FastAPI renders a dependency's HTTPException as
    {"detail": ...}, which no SDK understands. The dependency is also where
    Bearer and x-api-key credentials and API-token scopes are checked, so no
    credential is read here: a second reader would be a way around the scopes.
    """
    try:
        return await ctx.auth.require_user(request)
    except Exception as exc:  # noqa: BLE001 - HTTPException without importing FastAPI
        status = getattr(exc, "status_code", None)
        if not isinstance(status, int):
            raise
        detail = getattr(exc, "detail", "") or ("authentication required" if status == 401
                                                else "forbidden")
        raise CompatError(status, str(detail), headers=getattr(exc, "headers", None))


async def compat_json_body(request: Any) -> dict:
    """The request body as a JSON object; malformed is a provider-shaped 400."""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        raise CompatError(400, "request body must be valid JSON")
    if not isinstance(body, dict):
        raise CompatError(400, "request body must be a JSON object")
    return body


def compat_check_ready(ctx: Any) -> None:
    """ctx.not_ready() as a CompatError (503)."""
    blocked = ctx.not_ready()
    if blocked is None:
        return
    message = "model not ready"
    try:
        message = json.loads(bytes(blocked.body).decode("utf-8")).get("error") or message
    except Exception:  # noqa: BLE001
        pass
    raise CompatError(int(getattr(blocked, "status_code", 503) or 503), str(message),
                      headers={"Retry-After": "10"})


async def compat_metric(ctx: Any, endpoint: str, started: float, status: int,
                        error: str | None = None, *, stats: Any = None, model: str | None = None,
                        user_id: str | None = None) -> None:
    try:
        await asyncio.to_thread(ctx.db.log_metric, endpoint, (time.time() - started) * 1000,
                                status, error, stats=stats, model=model, user_id=user_id)
    except Exception as exc:  # noqa: BLE001 - metrics never fail a request
        log(f"compat metric failed: {exc}", logging.DEBUG)


def _oai_num(body: dict, key: str, kind: type, *, low: float | None = None,
             high: float | None = None) -> Any:
    value = body.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise CompatError(400, f"{key} must be a number", param=key)
    try:
        number = kind(value)
    except (TypeError, ValueError):
        raise CompatError(400, f"{key} must be a number", param=key)
    if kind is int and isinstance(value, float) and value != int(value):
        raise CompatError(400, f"{key} must be an integer", param=key)
    if (low is not None and number < low) or (high is not None and number > high):
        raise CompatError(400, f"{key} must be between {low} and {high}", param=key)
    return number


# --------------------------------------------------------------------------- #
# Model naming: "<alias|repo>", "raw/<alias>", "agent", "agent/<profile>"
# --------------------------------------------------------------------------- #
@dataclass
class CompatTarget:
    """What a request's model name resolved to."""

    mode: str                      # "raw" | "agent"
    model: str                     # the served repo id the request runs on
    name: str                      # the model name echoed back to the client
    profile: dict | None = None    # agent mode: the capability-gated profile


def compat_served_models(ctx: Any) -> set:
    """Every model id a request may run on: this node's plus route targets."""
    config = ctx.config
    served = {config.model, resolve_model(config.model, config)}
    for node in list(getattr(ctx.node_registry, "nodes", None) or []):
        if getattr(node, "model", None):
            served.add(node.model)
    return {m for m in served if m}


def _oai_slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")


async def compat_visible_profiles(ctx: Any, user: dict) -> list[dict]:
    """Agent profiles this caller may run: non-admins see enabled ones only."""
    try:
        rows = await asyncio.to_thread(ctx.db.list_agents)
    except Exception:  # noqa: BLE001
        return []
    admin = (user.get("role") or "user") == "admin"
    return [r for r in rows if admin or r.get("enabled")]


async def _oai_find_profile(ctx: Any, user: dict, wanted: str) -> dict | None:
    row = await asyncio.to_thread(ctx.db.get_agent, wanted)
    if row is None:
        key = wanted.strip().lower()
        for candidate in await asyncio.to_thread(ctx.db.list_agents):
            if (str(candidate.get("name") or "").strip().lower() == key
                    or _oai_slug(candidate.get("name")) == _oai_slug(wanted)):
                row = candidate
                break
    # A disabled profile does not exist for a non-admin: /api/agents does not
    # list it either, so "not found" rather than "forbidden".
    if row is None or (not row.get("enabled") and (user.get("role") or "user") != "admin"):
        return None
    allowed = chat_role_capabilities(user.get("role") or "user", ctx.config)
    wanted_caps = set(row.get("capabilities") or [])
    return dict(row, capabilities=[c for c in CAPABILITY_ORDER
                                   if c in wanted_caps and c in allowed])


async def compat_resolve_target(ctx: Any, name: Any, user: dict) -> CompatTarget:
    """The mode, model and profile a request's model name stands for.

    Agent mode is lenient about a plain name (an existing client may send any
    placeholder, and the agent always ran on the served model); raw mode is not,
    because there the name IS the model the client gets.
    """
    config = ctx.config
    raw_name = str(name or "").strip()
    lowered = raw_name.lower()
    if lowered == "agent" or lowered.startswith("agent/"):
        wanted = raw_name[6:].strip() if lowered.startswith("agent/") else ""
        profile = None
        if wanted:
            profile = await _oai_find_profile(ctx, user, wanted)
            if profile is None:
                raise CompatError(404, f"agent profile {wanted!r} not found",
                                  code="model_not_found", param="model")
        return CompatTarget("agent", config.model, raw_name, profile)
    forced_raw = lowered == "raw" or lowered.startswith("raw/")
    if forced_raw:
        raw_name_model = raw_name[4:].strip() if lowered.startswith("raw/") else ""
    else:
        raw_name_model = raw_name
    mode = "raw" if forced_raw else (getattr(config, "v1_default_mode", "agent") or "agent")
    if mode == "raw" and not getattr(config, "v1_raw_enabled", True):
        if forced_raw:
            raise CompatError(403, "raw model access is disabled on this server "
                              "(V1_RAW_ENABLED=0); use model 'agent'", param="model")
        mode = "agent"
    if mode == "agent":
        return CompatTarget("agent", config.model, raw_name or config.model, None)
    wanted = resolve_model(raw_name_model, config) if raw_name_model else config.model
    served = compat_served_models(ctx)
    if wanted not in served:
        raise CompatError(
            404, f"model {raw_name_model!r} is not served here (served: "
                 f"{', '.join(sorted(served))}). Add an alias in MODEL_ALIASES to map a "
                 "client's model name onto the served model.",
            code="model_not_found", param="model")
    return CompatTarget("raw", wanted, raw_name or config.model, None)


async def compat_model_entries(ctx: Any, user: dict) -> list[dict]:
    """Every name /v1 accepts, in the order /v1/models lists them."""
    config = ctx.config
    served = compat_served_models(ctx)
    # The configured model first: clients that take the first entry (the wire
    # helper takes the first raw/ one) should land on this node's own model.
    plain: list[str] = sorted(served, key=lambda m: (m != config.model, m))
    for alias, info in sorted(model_aliases(config).items()):
        if info.get("id") in served and alias not in plain:
            plain.append(alias)
    entries: list[dict] = []
    default_mode = getattr(config, "v1_default_mode", "agent")
    raw_ok = bool(getattr(config, "v1_raw_enabled", True))
    for name in plain:
        entries.append({"id": name, "mode": default_mode if raw_ok else "agent"})
    if raw_ok:
        entries.extend({"id": f"raw/{name}", "mode": "raw"} for name in plain)
    entries.append({"id": "agent", "mode": "agent"})
    for profile in await compat_visible_profiles(ctx, user):
        slug = _oai_slug(profile.get("name")) or str(profile.get("id"))
        entries.append({"id": f"agent/{slug}", "mode": "agent",
                        "profile_id": profile.get("id"), "name": profile.get("name")})
    return entries


# --------------------------------------------------------------------------- #
# The canonical request every wire format is translated into
# --------------------------------------------------------------------------- #
@dataclass
class CompatRequest:
    """One /v1 request in OpenAI chat terms, whichever wire format it came in.

    `messages` are OpenAI chat messages (system/user/assistant/tool, content a
    string or a list of text and image_url parts, assistant tool_calls), and
    `tools` are OpenAI function tools. The Anthropic and Responses parsers
    produce exactly this, so one engine serves all three.
    """

    model: str = ""
    messages: list = field(default_factory=list)
    tools: list = field(default_factory=list)
    tool_choice: Any = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop: list = field(default_factory=list)
    seed: int | None = None
    response_format: dict | None = None
    # Whether the client gets reasoning back (Anthropic: only with thinking on).
    reasoning: bool = True
    # Force the template's thinking switch; None keeps the server's default.
    thinking: bool | None = None
    reasoning_effort: str | None = None
    parallel_tool_calls: bool | None = None
    include_trace: bool = False
    include_usage: bool = False
    stream: bool = False
    call_prefix: str = "call_"


_OAI_ROLES = {"system", "developer", "user", "assistant", "tool", "function"}


def _oai_tool_from_function(fn: Any, where: str) -> dict:
    if not isinstance(fn, dict):
        raise CompatError(400, f"{where} must be an object", param="tools")
    name = fn.get("name")
    if not isinstance(name, str) or not _OAI_TOOL_NAME_RE.match(name):
        raise CompatError(400, f"{where}: name must match {_OAI_TOOL_NAME_RE.pattern}",
                          param="tools")
    params = fn.get("parameters")
    if params is not None and not isinstance(params, dict):
        raise CompatError(400, f"{where}: parameters must be a JSON schema object",
                          param="tools")
    return {"type": "function", "function": {
        "name": name, "description": str(fn.get("description") or ""),
        "parameters": params or {"type": "object", "properties": {}}}}


def compat_tool_names(tools: list) -> list[str]:
    return [t["function"]["name"] for t in tools or []]


def _oai_check_tool_choice(choice: Any, tools: list) -> Any:
    """Normalise tool_choice to "auto" | "none" | "required" | {function name}."""
    if choice is None:
        return None
    if isinstance(choice, str):
        if choice not in ("auto", "none", "required"):
            raise CompatError(400, "tool_choice must be auto, none, required or a function",
                              param="tool_choice")
        return choice
    if isinstance(choice, dict):
        fn = choice.get("function") if isinstance(choice.get("function"), dict) else choice
        name = fn.get("name")
        if not isinstance(name, str) or name not in compat_tool_names(tools):
            raise CompatError(400, f"tool_choice names an unknown tool {name!r}",
                              param="tool_choice")
        return {"type": "function", "function": {"name": name}}
    raise CompatError(400, "tool_choice must be a string or an object", param="tool_choice")


def _oai_check_parts(content: Any, where: str) -> Any:
    """Validate an OpenAI content value: a string, null, or a list of parts."""
    if content is None or isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise CompatError(400, f"{where}: content must be a string or a list of parts",
                          param="messages")
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            raise CompatError(400, f"{where}: every content part must be an object",
                              param="messages")
        kind = part.get("type", "text")
        if kind in ("text", "input_text", "output_text"):
            parts.append({"type": "text", "text": str(part.get("text") or "")})
        elif kind in ("image_url", "input_image"):
            url = part.get("image_url")
            if isinstance(url, dict):
                url = url.get("url")
            if not isinstance(url, str) or not url:
                raise CompatError(400, f"{where}: image part without a url", param="messages")
            parts.append({"type": "image_url", "image_url": {"url": url}})
        elif kind == "refusal":
            parts.append({"type": "text", "text": str(part.get("refusal") or "")})
        else:
            raise CompatError(400, f"{where}: unsupported content part type {kind!r}",
                              param="messages")
    return parts


def compat_parse_chat(body: dict) -> CompatRequest:
    """A /v1/chat/completions body as a CompatRequest, or a 400."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise CompatError(400, "messages must be a non-empty list of objects", param="messages")
    out: list[dict] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise CompatError(400, "messages must be a list of objects", param="messages")
        role = msg.get("role")
        if role not in _OAI_ROLES:
            raise CompatError(400, f"messages[{i}].role {role!r} is not supported",
                              param="messages")
        content = _oai_check_parts(msg.get("content"), f"messages[{i}]")
        if role == "developer":
            role = "system"
        entry: dict = {"role": "tool" if role == "function" else role, "content": content}
        if role in ("tool", "function"):
            entry["tool_call_id"] = str(msg.get("tool_call_id") or "")
            if msg.get("name"):
                entry["name"] = str(msg["name"])
        if role == "assistant":
            calls = msg.get("tool_calls")
            if calls is None and isinstance(msg.get("function_call"), dict):
                calls = [{"id": "", "type": "function", "function": msg["function_call"]}]
            if calls:
                if not isinstance(calls, list):
                    raise CompatError(400, f"messages[{i}].tool_calls must be a list",
                                      param="messages")
                entry["tool_calls"] = [_oai_norm_call(c, f"messages[{i}].tool_calls")
                                       for c in calls]
            if msg.get("reasoning_content"):
                entry["reasoning_content"] = str(msg["reasoning_content"])
        out.append(entry)
    tools: list[dict] = []
    raw_tools = body.get("tools")
    if raw_tools is not None:
        if not isinstance(raw_tools, list):
            raise CompatError(400, "tools must be a list", param="tools")
        for i, tool in enumerate(raw_tools):
            if not isinstance(tool, dict) or tool.get("type", "function") != "function":
                raise CompatError(400, f"tools[{i}]: only function tools are supported",
                                  param="tools")
            tools.append(_oai_tool_from_function(tool.get("function"), f"tools[{i}].function"))
    elif isinstance(body.get("functions"), list):
        tools = [_oai_tool_from_function(fn, f"functions[{i}]")
                 for i, fn in enumerate(body["functions"])]
    choice = body.get("tool_choice")
    if choice is None and body.get("function_call") is not None:
        fc = body["function_call"]
        choice = fc if isinstance(fc, str) else {"type": "function", "function": fc}
    # One choice per request: n > 1 would need n generations per slot.
    _oai_num(body, "n", int, low=1, high=1)
    stream_options = body.get("stream_options")
    max_tokens = body.get("max_completion_tokens", body.get("max_tokens"))
    creq = CompatRequest(
        model=str(body.get("model") or ""),
        messages=out, tools=tools,
        tool_choice=_oai_check_tool_choice(choice, tools),
        max_tokens=_oai_num({"max_tokens": max_tokens}, "max_tokens", int, low=1),
        temperature=_oai_num(body, "temperature", float, low=0.0, high=2.0),
        top_p=_oai_num(body, "top_p", float, low=0.0, high=1.0),
        seed=_oai_num(body, "seed", int),
        stop=_oai_stop_list(body.get("stop")),
        response_format=_oai_response_format(body.get("response_format")),
        reasoning_effort=(str(body["reasoning_effort"]) if body.get("reasoning_effort")
                          else None),
        parallel_tool_calls=(bool(body["parallel_tool_calls"])
                             if body.get("parallel_tool_calls") is not None else None),
        include_trace=bool(body.get("x_include_trace")),
        include_usage=(isinstance(stream_options, dict)
                       and bool(stream_options.get("include_usage"))),
        stream=bool(body.get("stream")),
    )
    if creq.reasoning_effort == "none":
        creq.thinking = False
    return creq


def _oai_norm_call(call: Any, where: str) -> dict:
    if not isinstance(call, dict):
        raise CompatError(400, f"{where}: every tool call must be an object", param="messages")
    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = fn.get("name") or call.get("name")
    if not isinstance(name, str) or not name:
        raise CompatError(400, f"{where}: a tool call needs function.name", param="messages")
    args = fn.get("arguments", call.get("arguments"))
    if isinstance(args, (dict, list)):
        args = json.dumps(args, ensure_ascii=False)
    return {"id": str(call.get("id") or ""), "type": "function",
            "function": {"name": name, "arguments": str(args or "{}")}}


def _oai_stop_list(stop: Any) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop] if stop else []
    if isinstance(stop, list) and all(isinstance(s, str) for s in stop):
        if len(stop) > 16:
            raise CompatError(400, "at most 16 stop sequences", param="stop")
        return [s for s in stop if s]
    raise CompatError(400, "stop must be a string or a list of strings", param="stop")


def _oai_response_format(fmt: Any) -> dict | None:
    if fmt is None:
        return None
    if not isinstance(fmt, dict) or fmt.get("type") not in ("text", "json_object", "json_schema"):
        raise CompatError(400, "response_format.type must be text, json_object or json_schema",
                          param="response_format")
    if fmt["type"] == "text":
        return None
    if fmt["type"] == "json_schema":
        schema = fmt.get("json_schema")
        if not isinstance(schema, dict):
            raise CompatError(400, "response_format.json_schema must be an object",
                              param="response_format")
    return fmt


# --------------------------------------------------------------------------- #
# Content helpers
# --------------------------------------------------------------------------- #
def compat_text_of(content: Any) -> str:
    """The text of a content value (string or parts); images are left out."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(p.get("text") or "") for p in content
                       if isinstance(p, dict) and p.get("type") == "text")
    return str(content)


def compat_image_parts(content: Any) -> list[dict]:
    if not isinstance(content, list):
        return []
    return [p for p in content if isinstance(p, dict) and p.get("type") == "image_url"]


def compat_has_images(messages: list) -> bool:
    return any(compat_image_parts(m.get("content")) for m in messages or [])


def compat_backend_caps(ctx: Any) -> frozenset:
    """Capabilities of the backend serving this node's model."""
    manager = ctx.model_manager
    backend = getattr(manager, "backend", None)
    try:
        if backend is None:
            backend = get_backend(getattr(ctx.config, "model_backend", None))
        return frozenset(backend.caps_for(manager))
    except Exception:  # noqa: BLE001
        return frozenset()


def compat_vision(ctx: Any, model_id: str) -> bool:
    """Whether the served model can read images: backend AND catalogue agree.

    The backend alone is not enough (mlx-lm declares vision, and a 3B text
    model behind it would be handed image parts it cannot see). The chat
    route's rule (vision.served_vision) decides when it is there, so the
    composer, /v1 and image_analyze agree and VISION_MODE=on/off applies here
    too; a model served by another node is judged from the config alone.
    """
    served = optional_feature("vision", "served_vision")
    if callable(served):
        config = ctx.config
        own = not model_id or model_id in (config.model, resolve_model(config.model, config))
        try:
            judged = config if own else dataclasses.replace(config, model=str(model_id))
            return bool(served(ctx.model_manager if own else None, judged)[0])
        except Exception:  # noqa: BLE001
            return False
    if "vision" not in compat_backend_caps(ctx):
        return False
    try:
        meta = model_metadata(str(model_id or ""))
    except Exception:  # noqa: BLE001
        meta = None
    return bool(isinstance(meta, dict) and meta.get("vision"))


# --------------------------------------------------------------------------- #
# Raw mode: rendering the client's conversation for the model
# --------------------------------------------------------------------------- #
def _oai_clip(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    stop = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[:stop + 1] if stop > limit // 3 else cut.rstrip()) + " ..."


def _oai_param_line(schema: dict) -> tuple[str, str]:
    props = schema.get("properties") if isinstance(schema, dict) else None
    required = schema.get("required") if isinstance(schema, dict) else None
    required = [r for r in (required or []) if isinstance(r, str)]
    if not isinstance(props, dict) or not props:
        return "no arguments", "none"
    parts = []
    for name, spec in props.items():
        spec = spec if isinstance(spec, dict) else {}
        kind = spec.get("type")
        if isinstance(kind, list):
            kind = "|".join(str(k) for k in kind)
        if not kind and isinstance(spec.get("enum"), list):
            kind = "one of " + "/".join(str(v) for v in spec["enum"][:8])
        desc = _oai_clip(spec.get("description") or "", _OAI_PARAM_DESC_CHARS)
        label = f"{name} ({kind or 'any'}"
        label += f": {desc})" if desc else ")"
        parts.append(label)
    return ", ".join(parts), ", ".join(required) or "none"


def compat_tool_section(tools: list, tool_choice: Any) -> str:
    """The client's tools in the agent's prompt protocol, compactly.

    Same shape as build_agent_system_prompt's catalogue (TOOL_PROTOCOL then one
    "- name: description / args / required" entry per tool), so a model this
    app was tuned on sees the protocol it knows. TOOL RESULT [name] is how the
    client's tool results come back (compat_render_messages).
    """
    lines = [
        "You can call tools. To call one, reply with a single JSON object and nothing else:",
        '{"tool": "tool_name", "args": {"arg": "value"}}',
        "",
        "Rules:",
        "- Use the exact tool and argument names listed below.",
        "- The result comes back in a message beginning with TOOL RESULT [tool_name]. "
        "Read it before deciding what to do next.",
        "- When you have enough information, reply in plain text with no JSON.",
        "- Never invent tool output.",
    ]
    if tool_choice == "required":
        lines.append("- You must call one of the tools in this reply.")
    elif isinstance(tool_choice, dict):
        lines.append(f"- You must call the tool {tool_choice['function']['name']} in this reply.")
    lines += ["", "Available tools:"]
    for tool in tools:
        fn = tool["function"]
        params, required = _oai_param_line(fn.get("parameters") or {})
        desc = _oai_clip(fn.get("description") or "", _OAI_TOOL_DESC_CHARS) or "(no description)"
        lines.append(f"- {fn['name']}: {desc}\n  args: {params}\n  required: {required}")
    return "\n".join(lines)


def compat_format_instruction(fmt: dict | None) -> str:
    if not fmt:
        return ""
    if fmt.get("type") == "json_schema":
        spec = fmt.get("json_schema") or {}
        schema = spec.get("schema", spec)
        return ("Reply with one JSON value that matches this JSON schema, and nothing else "
                "(no prose, no code fence):\n" + json.dumps(schema, ensure_ascii=False,
                                                          separators=(",", ":")))
    return "Reply with one valid JSON object and nothing else (no prose, no code fence)."


def _oai_call_args(arguments: str) -> dict:
    try:
        value = json.loads(arguments or "{}")
    except ValueError:
        value, _ = repair_json(arguments or "", close_open=True)
    return value if isinstance(value, dict) else {}


def _oai_user_content(content: Any, vision: bool) -> Any:
    images = compat_image_parts(content)
    if not images:
        return compat_text_of(content)
    if not vision:
        raise CompatError(400, "this model cannot read images; send text only or serve a "
                               "vision model", param="messages")
    return [p for p in content if isinstance(p, dict) and p.get("type") in ("text", "image_url")]


def compat_render_messages(creq: CompatRequest, *, native: bool, vision: bool,
                           tools_active: bool) -> list[dict]:
    """The messages the model sees in raw mode.

    Native path: OpenAI messages as the client sent them (assistant tool_calls
    and role "tool" messages), which mlx-lm's chat template renders itself.
    Prompt path: the agent's protocol. A tool call becomes the assistant's
    {"tool", "args"} line and a tool result becomes a user message starting
    "TOOL RESULT [name]:", and consecutive same-role messages are merged
    because many chat templates insist on strict user/assistant alternation.
    """
    names_by_id: dict[str, str] = {}
    for msg in creq.messages:
        for call in msg.get("tool_calls") or []:
            if call.get("id"):
                names_by_id[call["id"]] = call["function"]["name"]
    system_parts = [compat_text_of(m.get("content")) for m in creq.messages
                    if m["role"] == "system" and compat_text_of(m.get("content")).strip()]
    if tools_active and not native:
        system_parts.append(compat_tool_section(creq.tools, creq.tool_choice))
    elif tools_active and native and (creq.tool_choice == "required"
                                      or isinstance(creq.tool_choice, dict)):
        system_parts.append(
            "You must call a tool in this reply." if creq.tool_choice == "required"
            else f"You must call the tool {creq.tool_choice['function']['name']} in this reply.")
    fmt = compat_format_instruction(creq.response_format)
    if fmt:
        system_parts.append(fmt)
    out: list[dict] = []
    if system_parts:
        out.append({"role": "system", "content": "\n\n".join(system_parts)})
    for msg in creq.messages:
        role = msg["role"]
        if role == "system":
            continue
        if role == "user":
            out.append({"role": "user", "content": _oai_user_content(msg.get("content"), vision)})
        elif role == "assistant":
            text = compat_text_of(msg.get("content"))
            calls = msg.get("tool_calls") or []
            if native:
                entry: dict = {"role": "assistant", "content": text}
                if calls:
                    entry["tool_calls"] = [
                        {"id": c.get("id") or f"call_{i}", "type": "function",
                         "function": {"name": c["function"]["name"],
                                      "arguments": c["function"]["arguments"]}}
                        for i, c in enumerate(calls)]
                out.append(entry)
            else:
                lines = [text.strip()] if text.strip() else []
                lines += [json.dumps({"tool": c["function"]["name"],
                                      "args": _oai_call_args(c["function"]["arguments"])},
                                     ensure_ascii=False) for c in calls]
                out.append({"role": "assistant", "content": "\n".join(lines)})
        elif role == "tool":
            name = msg.get("name") or names_by_id.get(msg.get("tool_call_id") or "") or "tool"
            result = compat_text_of(msg.get("content"))
            if native:
                out.append({"role": "tool", "tool_call_id": msg.get("tool_call_id") or "",
                            "name": name, "content": result})
            else:
                out.append({"role": "user", "content": f"TOOL RESULT [{name}]:\n{result}"})
    if native:
        return out
    merged: list[dict] = []
    for msg in out:
        if (merged and merged[-1]["role"] == msg["role"] and msg["role"] in ("user", "assistant")
                and isinstance(merged[-1]["content"], str) and isinstance(msg["content"], str)):
            merged[-1] = {"role": msg["role"],
                          "content": merged[-1]["content"] + "\n\n" + msg["content"]}
        else:
            merged.append(msg)
    return merged


# --------------------------------------------------------------------------- #
# Plans: everything decided before admission
# --------------------------------------------------------------------------- #
@dataclass
class CompatPlan:
    mode: str
    messages: list = field(default_factory=list)    # raw: what the model sees
    native: bool = False
    tool_names: list = field(default_factory=list)
    max_tokens: int = 0
    temperature: float = 0.7
    cost: int = 0
    # agent mode
    prompt: str = ""
    history: list = field(default_factory=list)
    attachments: list = field(default_factory=list)
    agent: Any = None
    profile: dict | None = None


def _oai_profile_agent(app: Any, ctx: Any, profile: dict) -> Any:
    """The capability-gated agent for a profile, cached like the chat route's."""
    config = ctx.config
    caps = tuple(profile.get("capabilities") or [])
    factory = getattr(app.state, "agent_factory", None)
    stamp = tuple(repr(getattr(config, name, None)) for name in Config.MUTABLE)
    key = (profile.get("id"), profile.get("updated_at"), caps, config.model, stamp, id(factory))
    cache = ctx.extras.setdefault("compat_agents", {})
    hit = cache.get(key)
    if hit is not None:
        return hit
    if factory is None:
        _cfg, agent = ctx.agent_for_profile(dict(profile, capabilities=list(caps)))
    else:
        cfg = dataclasses.replace(
            config, agent_tools=",".join(tools_for_capabilities(caps)),
            rag_enabled=rag_for_capabilities(caps) and config.rag_enabled)
        agent = factory(cfg, profile)
    while len(cache) >= 16:
        cache.pop(next(iter(cache)))
    cache[key] = agent
    return agent


async def compat_prepare(app: Any, ctx: Any, creq: CompatRequest, target: CompatTarget
                         ) -> CompatPlan:
    """Render the request for its mode; every 400 is raised here, pre-stream."""
    config = ctx.config
    max_tokens = int(creq.max_tokens or config.max_tokens)
    temperature = float(config.temperature if creq.temperature is None else creq.temperature)
    if target.mode == "raw":
        tools_active = bool(creq.tools) and creq.tool_choice != "none"
        native = tools_active and "native_tool_calls" in compat_backend_caps(ctx)
        vision = compat_has_images(creq.messages) and compat_vision(ctx, target.model)
        messages = compat_render_messages(creq, native=native, vision=vision,
                                          tools_active=tools_active)
        if not any(m["role"] != "system" for m in messages):
            raise CompatError(400, "the conversation has no user message", param="messages")
        plan = CompatPlan("raw", messages=messages, native=native,
                          tool_names=compat_tool_names(creq.tools) if tools_active else [],
                          max_tokens=max_tokens, temperature=temperature)
        plan.cost = admission_cost_tokens(messages, max_tokens, config.context_size)
        return plan
    # Agent mode: the last user message is the prompt and earlier user and
    # assistant turns are the history, as the original handler did. The
    # client's system prompt and tools are not used: the agent has its own.
    last = None
    for i in range(len(creq.messages) - 1, -1, -1):
        if creq.messages[i]["role"] == "user":
            last = i
            break
    if last is None:
        raise CompatError(400, "no user message provided", param="messages")
    content = creq.messages[last].get("content")
    prompt = compat_text_of(content)
    images = compat_image_parts(content)
    if not prompt.strip() and not images:
        raise CompatError(400, "no user message provided", param="messages")
    history = [{"role": m["role"], "content": compat_text_of(m.get("content"))}
               for m in creq.messages[:last]
               if m["role"] in ("user", "assistant") and compat_text_of(m.get("content"))]
    agent = getattr(app.state, "agent", None) or ctx.agent
    if target.profile is not None:
        try:
            agent = _oai_profile_agent(app, ctx, target.profile)
        except Exception as exc:  # noqa: BLE001
            log(f"compat: profile agent failed to build: {exc}", logging.WARNING)
            raise CompatError(500, redact_text(str(exc)))
    plan = CompatPlan("agent", max_tokens=max_tokens, temperature=temperature,
                      prompt=prompt, history=history, attachments=images, agent=agent,
                      profile=target.profile)
    plan.cost = admission_cost_tokens(
        [{"role": "system", "content": config.system_prompt_with_identity}, *history,
         {"role": "user", "content": prompt}], max_tokens, config.context_size)
    return plan


async def compat_admit(ctx: Any, plan: CompatPlan) -> Any:
    """One generation slot for the plan's KV cost, or a provider-shaped 503."""
    gate = ctx.generation_gate
    try:
        try:
            return await gate.acquire(cost_tokens=plan.cost)
        except TypeError:
            await gate.acquire()
            return None
    except GenerationBusy as exc:
        raise CompatError(503, str(exc), headers={"Retry-After": "5"})


def compat_release(ctx: Any, ticket: Any) -> None:
    try:
        if ticket is None:
            ctx.generation_gate.release()
        else:
            ctx.generation_gate.release(ticket)
    except TypeError:
        ctx.generation_gate.release()
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# Engines: normalised events
# --------------------------------------------------------------------------- #
# Both engines yield the same small vocabulary, which each wire format renders:
#   text {text}            reasoning {text}
#   tool_start {index, id, name}   tool_args {index, delta}
#   tool_end {index, id, name, arguments, args}
#   trace {event}          (agent mode: tool_call / tool_result / approvals)
#   error {message, status}
#   done {finish: stop|length|tool_calls|stop_sequence|error, stop_sequence,
#         usage {prompt_tokens, completion_tokens, cached_tokens, reasoning_tokens},
#         answer (agent mode: the authoritative final text), tools_used, stats}
class _OaiStopFilter:
    """Client stop sequences, enforced here so the matched one is known.

    The server would stop on them too, but then the text never contains the
    sequence and Anthropic's stop_reason "stop_sequence" (with the sequence
    itself) cannot be reported. Holding back len(longest)-1 characters keeps a
    sequence split across two deltas from leaking its first half.
    """

    def __init__(self, stops: list) -> None:
        self.stops = [s for s in stops or [] if s]
        self.hold = max((len(s) for s in self.stops), default=1) - 1
        self.buf = ""
        self.matched: str | None = None

    def feed(self, text: str) -> str:
        if not self.stops:
            return text
        if self.matched is not None:
            return ""
        self.buf += text
        best = None
        for s in self.stops:
            i = self.buf.find(s)
            if i >= 0 and (best is None or i < best[0]):
                best = (i, s)
        if best is not None:
            out, self.buf, self.matched = self.buf[:best[0]], "", best[1]
            return out
        if len(self.buf) > self.hold:
            cut = len(self.buf) - self.hold
            out, self.buf = self.buf[:cut], self.buf[cut:]
            return out
        return ""

    def flush(self) -> str:
        out, self.buf = self.buf, ""
        return out


def compat_json_answer(text: str) -> str:
    """Best-effort JSON for response_format: unfenced, located, repaired."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"\A```[\w-]*\s*", "", s)
        s = re.sub(r"\s*```\s*\Z", "", s)
    try:
        return json.dumps(json.loads(s), ensure_ascii=False)
    except ValueError:
        pass
    starts = [i for i in (s.find("{"), s.find("[")) if i >= 0]
    if not starts:
        return text
    value, _repairs = repair_json(s[min(starts):], close_open=True)
    return json.dumps(value, ensure_ascii=False) if value is not None else text


def _oai_usage_dict(prompt: int = 0, completion: int = 0, cached: int = 0,
                    reasoning: int = 0) -> dict:
    return {"prompt_tokens": int(prompt or 0), "completion_tokens": int(completion or 0),
            "cached_tokens": int(cached or 0), "reasoning_tokens": int(reasoning or 0)}


def _oai_client_for(ctx: Any, model_id: str) -> Any:
    client = ctx.model_client
    if model_id and model_id != ctx.config.model and model_id != resolve_model(
            ctx.config.model, ctx.config):
        # A route target's model: same routing, the request names that model.
        client = ModelClient(dataclasses.replace(ctx.config, model=model_id),
                             cluster=ctx.router)
    return client


def _oai_extra(creq: CompatRequest, plan: CompatPlan) -> dict:
    extra: dict = {}
    if creq.top_p is not None:
        extra["top_p"] = creq.top_p
    if creq.top_k is not None:
        extra["top_k"] = creq.top_k
    if creq.seed is not None:
        extra["seed"] = creq.seed
    if creq.thinking is not None:
        extra["enable_thinking"] = bool(creq.thinking)
    if creq.reasoning_effort and creq.reasoning_effort != "none":
        extra["reasoning_effort"] = creq.reasoning_effort
    if plan.tool_names:
        if plan.native:
            extra["tools"] = creq.tools
            if creq.tool_choice is not None:
                extra["tool_choice"] = creq.tool_choice
            if creq.parallel_tool_calls is not None:
                extra["parallel_tool_calls"] = creq.parallel_tool_calls
        # Both paths: a native backend still leaves calls in the text for a
        # tokenizer without a tool parser. Only the client's own names count.
        extra["parse_tool_calls"] = True
        extra["known_tools"] = set(plan.tool_names)
    return extra


async def compat_raw_events(ctx: Any, creq: CompatRequest, plan: CompatPlan,
                            target: CompatTarget) -> AsyncIterator[dict]:
    """Raw mode: the model with the client's tools; nothing executes here."""
    client = _oai_client_for(ctx, target.model)
    stats = GenerationStats()
    splitter = ThinkSplitter()
    stopper = _OaiStopFilter(creq.stop)
    json_mode = bool(creq.response_format)
    held: list[str] = []
    finish = ""
    usage_raw: dict | None = None
    seen: set = set()
    n_calls = 0

    def text_events(kind: str, piece: str) -> list[dict]:
        if not piece:
            return []
        if kind == "think":
            return [{"type": "reasoning", "text": piece}]
        safe = stopper.feed(piece)
        if json_mode:
            held.append(safe)
            return []
        return [{"type": "text", "text": safe}] if safe else []

    gen = client.stream_events(plan.messages, max_tokens=plan.max_tokens,
                               temperature=plan.temperature, stats=stats, kind="chat",
                               extra=_oai_extra(creq, plan))
    try:
        async for ev in gen:
            etype = ev.get("type")
            if etype == "content":
                for kind, piece in splitter.feed(ev.get("text") or ""):
                    for out in text_events(kind, piece):
                        yield out
                if stopper.matched is not None:
                    break
            elif etype == "reasoning":
                if ev.get("text"):
                    yield {"type": "reasoning", "text": ev["text"]}
            elif etype == "tool_call_delta":
                # Whole calls only: a native backend's fragments are joined and
                # repaired by stream_events (done=True carries parsed args), and
                # a prompt-protocol call's raw text is not its arguments. A
                # client always receives valid JSON arguments this way.
                if not ev.get("done"):
                    continue
                key = (ev.get("source"), ev.get("index"))
                name = str(ev.get("name") or "")
                if key in seen or not name:
                    continue
                seen.add(key)
                args = ev.get("args")
                if not isinstance(args, dict):
                    args = _oai_call_args(str(ev.get("arguments") or ""))
                # Text held back by the stop filter belongs before the call.
                pending = stopper.flush() if stopper.matched is None else ""
                for out in text_events("answer", pending) if pending else []:
                    yield out
                index = n_calls
                n_calls += 1
                call_id = str(ev.get("id") or "") or f"{creq.call_prefix}{uuid.uuid4().hex[:24]}"
                arguments = json.dumps(args, ensure_ascii=False)
                yield {"type": "tool_start", "index": index, "id": call_id, "name": name}
                yield {"type": "tool_args", "index": index, "delta": arguments}
                yield {"type": "tool_end", "index": index, "id": call_id, "name": name,
                       "arguments": arguments, "args": args}
            elif etype == "finish":
                finish = str(ev.get("reason") or "")
            elif etype == "usage":
                usage_raw = ev.get("usage") if isinstance(ev.get("usage"), dict) else None
    finally:
        await gen.aclose()
    for kind, piece in splitter.flush():
        for out in text_events(kind, piece):
            yield out
    tail = stopper.flush() if stopper.matched is None else ""
    if tail:
        if json_mode:
            held.append(tail)
        else:
            yield {"type": "text", "text": tail}
    if json_mode:
        answer = "".join(held)
        if answer.strip():
            yield {"type": "text", "text": compat_json_answer(answer)}
    if stopper.matched is not None:
        reason = "stop_sequence"
    elif n_calls:
        reason = "tool_calls"
    elif finish == "length":
        reason = "length"
    else:
        reason = "stop"
    raw = usage_raw or {}
    details = raw.get("completion_tokens_details") if isinstance(raw, dict) else None
    usage = _oai_usage_dict(
        raw.get("prompt_tokens") or stats.prompt_tokens,
        raw.get("completion_tokens") or stats.completion_tokens,
        _mc_cached_tokens(raw) or stats.cached_tokens,
        (details or {}).get("reasoning_tokens") if isinstance(details, dict)
        else stats.reasoning_tokens)
    yield {"type": "done", "finish": reason, "stop_sequence": stopper.matched,
           "usage": usage, "stats": stats}


_OAI_TRACE_EVENTS = frozenset({
    "tool_call_start", "tool_call", "tool_result", "approval_required",
    "approval_resolved", "consent_required", "consent_resolved", "context_injected",
})


async def compat_agent_events(ctx: Any, creq: CompatRequest, plan: CompatPlan,
                              user: dict, ticket: Any) -> AsyncIterator[dict]:
    """Agent mode: this app's agent, unattended, mapped onto the vocabulary."""
    config = ctx.config
    conversation_id = f"openai-{uuid.uuid4().hex[:8]}"
    turn = TurnContext(user_id=user["id"], conversation_id=conversation_id,
                       role=user.get("role") or "user", interactive=False,
                       cancel=asyncio.Event(), agent_profile=plan.profile)
    if plan.attachments:
        # The same decision the chat route hands the agent, so an image sent
        # over /v1 reaches the model exactly when it would from the composer.
        turn.extras["vision"] = await asyncio.to_thread(compat_vision, ctx, config.model)
    gate = ctx.generation_gate
    if ticket is not None and hasattr(gate, "suspend"):
        turn.suspend_admission = lambda: gate.suspend(ticket)
    bound = set_current_turn(turn)
    kwargs = {"attachments": plan.attachments} if plan.attachments else {}
    streamed_text = False
    streamed_reasoning = False
    final: dict | None = None
    partial: str | None = None
    gen = plan.agent.run_iterating(plan.prompt, plan.history, conversation_id,
                                   plan.max_tokens, plan.temperature, **kwargs)
    try:
        async for ev in gen:
            etype = ev.get("type")
            if etype == "token":
                piece = ev.get("token") or ev.get("text") or ""
                if piece:
                    streamed_text = True
                    yield {"type": "text", "text": piece}
            elif etype == "think_token":
                piece = ev.get("token") or ev.get("text") or ""
                if piece:
                    streamed_reasoning = True
                    yield {"type": "reasoning", "text": piece}
            elif etype in _OAI_TRACE_EVENTS:
                yield {"type": "trace", "event": ev}
            elif etype == "error":
                yield {"type": "error", "message": str(ev.get("error") or "agent error"),
                       "status": 502}
            elif etype == "cancelled":
                partial = str(ev.get("partial") or "")
                break
            elif etype == "final":
                final = ev
    finally:
        turn.cancel.set()
        try:
            await gen.aclose()
        finally:
            reset_current_turn(bound)
    if final is None:
        answer = partial or ""
        if answer and not streamed_text:
            yield {"type": "text", "text": answer}
        yield {"type": "done", "finish": "stop", "usage": _oai_usage_dict(), "answer": answer,
               "tools_used": [], "stats": None}
        return
    answer = str(final.get("answer") or "")
    reasoning = str(final.get("reasoning") or "")
    if reasoning and not streamed_reasoning:
        yield {"type": "reasoning", "text": reasoning}
    # If the answer never streamed as tokens, send it now.
    if answer and not streamed_text:
        yield {"type": "text", "text": answer}
    stats = GenerationStats(prompt_tokens=int(final.get("prompt_tokens") or 0),
                            completion_tokens=int(final.get("completion_tokens") or 0),
                            total_ms=float(final.get("elapsed_ms") or 0.0))
    yield {"type": "done", "finish": "length" if final.get("truncated") else "stop",
           "usage": _oai_usage_dict(stats.prompt_tokens, stats.completion_tokens),
           "answer": answer, "tools_used": list(final.get("tools_used") or []),
           "stats": stats, "model": config.model}


async def compat_events(ctx: Any, creq: CompatRequest, plan: CompatPlan,
                        target: CompatTarget, user: dict, ticket: Any) -> AsyncIterator[dict]:
    """The plan's engine, with failures turned into an error + done pair."""
    # Re-bound inside the generator: a StreamingResponse body runs outside the
    # request handler's context, and RAG, memory and the cluster records read
    # the acting user from this contextvar.
    set_acting_user(user["id"])
    if plan.mode == "raw":
        src = compat_raw_events(ctx, creq, plan, target)
    else:
        src = compat_agent_events(ctx, creq, plan, user, ticket)
    try:
        async for ev in src:
            yield ev
    except Exception as exc:  # noqa: BLE001
        message = redact_text(f"{type(exc).__name__}: {exc}")
        log_event(_oai_log, logging.ERROR, "compat.failed", mode=plan.mode, error=message)
        yield {"type": "error", "message": message, "status": 503}
        yield {"type": "done", "finish": "error", "usage": _oai_usage_dict(), "stats": None}
    finally:
        await src.aclose()


@dataclass
class CompatResult:
    """A whole (non-streaming) generation, collected from the events."""

    text: str = ""
    reasoning: str = ""
    tool_calls: list = field(default_factory=list)
    finish: str = "stop"
    stop_sequence: str | None = None
    usage: dict = field(default_factory=_oai_usage_dict)
    trace: list = field(default_factory=list)
    tools_used: list = field(default_factory=list)
    error: str = ""
    error_status: int = 0
    stats: Any = None


async def compat_collect(events: AsyncIterator[dict]) -> CompatResult:
    result = CompatResult()
    text: list[str] = []
    reasoning: list[str] = []
    answer = None
    try:
        async for ev in events:
            etype = ev["type"]
            if etype == "text":
                text.append(ev["text"])
            elif etype == "reasoning":
                reasoning.append(ev["text"])
            elif etype == "tool_end":
                result.tool_calls.append({"id": ev["id"], "name": ev["name"],
                                          "arguments": ev["arguments"], "args": ev["args"]})
            elif etype == "trace":
                result.trace.append(ev["event"])
            elif etype == "error":
                result.error = result.error or ev["message"]
                result.error_status = result.error_status or int(ev.get("status") or 502)
            elif etype == "done":
                result.finish = ev.get("finish") or "stop"
                result.stop_sequence = ev.get("stop_sequence")
                result.usage = ev.get("usage") or _oai_usage_dict()
                result.tools_used = list(ev.get("tools_used") or [])
                result.stats = ev.get("stats")
                answer = ev.get("answer")
    finally:
        await events.aclose()
    # Agent mode: the final answer is authoritative (revision passes may have
    # rewritten what streamed).
    result.text = answer if isinstance(answer, str) else "".join(text)
    result.reasoning = "".join(reasoning)
    return result


# --------------------------------------------------------------------------- #
# Driving a request: auth -> parse -> target -> readiness -> plan -> admission
# --------------------------------------------------------------------------- #
_oai_stream_class: list = []


def compat_stream_response(body: Any, on_close: Callable[[], None],
                           media_type: str = "text/event-stream") -> Any:
    """A StreamingResponse that always runs `on_close` when the response ends.

    The generation slot is taken before the response starts (so a full queue is
    an honest 503, not an error frame after a 200). A body generator's finally
    cannot be the only release: a client that disconnects before the first
    chunk means the generator never starts and its finally never runs.
    """
    if not _oai_stream_class:
        from fastapi.responses import StreamingResponse

        class _OaiClosingStream(StreamingResponse):
            def __init__(self, content: Any, closer: Callable[[], None], **kw: Any) -> None:
                super().__init__(content, **kw)
                self._closer = closer

            async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
                try:
                    await super().__call__(scope, receive, send)
                finally:
                    try:
                        self._closer()
                    except Exception:  # noqa: BLE001
                        pass

        _oai_stream_class.append(_OaiClosingStream)
    return _oai_stream_class[0](body, on_close, media_type=media_type,
                                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def compat_handle(app: Any, ctx: Any, request: Any, *, provider: str, endpoint: str,
                        parse: Callable[[dict], CompatRequest],
                        to_json: Callable[..., dict],
                        to_stream: Callable[..., AsyncIterator[str]],
                        check: Callable[[dict], None] | None = None) -> Any:
    """One /v1 generation request, end to end, in the caller's wire format."""
    from fastapi.responses import JSONResponse
    started = time.time()
    uid = None
    try:
        user = await compat_user(ctx, request)
        uid = user["id"]
        body = await compat_json_body(request)
        if check is not None:
            check(body)
        creq = parse(body)
        target = await compat_resolve_target(ctx, creq.model, user)
        compat_check_ready(ctx)
        plan = await compat_prepare(app, ctx, creq, target)
        ticket = await compat_admit(ctx, plan)
    except CompatError as exc:
        await compat_metric(ctx, endpoint, started, exc.status, exc.message, user_id=uid)
        return compat_error_response(exc, provider)
    log_event(_oai_log, logging.INFO, "compat.start", endpoint=endpoint, mode=target.mode,
              model=target.model, stream=creq.stream, tools=len(creq.tools), user_id=uid,
              native=plan.native)
    events = compat_events(ctx, creq, plan, target, user, ticket)
    if creq.stream:
        seen: dict = {}

        async def tap() -> AsyncIterator[dict]:
            async for ev in events:
                if ev["type"] in ("done", "error"):
                    seen.setdefault(ev["type"], ev)
                yield ev

        async def body_iter() -> AsyncIterator[str]:
            try:
                async for frame in to_stream(tap(), creq, target, plan):
                    yield frame
            finally:
                # Release first and synchronously: under a cancelled scope (the
                # client went away) every await below raises again at once.
                compat_release(ctx, ticket)
                await events.aclose()
                done = seen.get("done") or {}
                err = seen.get("error")
                await compat_metric(ctx, endpoint, started, int(err.get("status") or 502)
                                    if err else 200, err.get("message") if err else None,
                                    stats=done.get("stats"), model=target.model, user_id=uid)

        return compat_stream_response(body_iter(), lambda: compat_release(ctx, ticket))
    try:
        result = await compat_collect(events)
    finally:
        compat_release(ctx, ticket)
    if result.error and not (result.text or result.tool_calls):
        await compat_metric(ctx, endpoint, started, result.error_status or 502, result.error,
                            model=target.model, user_id=uid)
        return compat_error_response(CompatError(result.error_status or 502, result.error),
                                     provider)
    await compat_metric(ctx, endpoint, started, 200, stats=result.stats, model=target.model,
                        user_id=uid)
    return JSONResponse(to_json(result, creq, target, plan))


# --------------------------------------------------------------------------- #
# OpenAI wire format: chat completions
# --------------------------------------------------------------------------- #
def openai_usage(usage: dict) -> dict:
    p, c = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c,
            "prompt_tokens_details": {"cached_tokens": int(usage.get("cached_tokens") or 0)},
            "completion_tokens_details": {
                "reasoning_tokens": int(usage.get("reasoning_tokens") or 0)}}


def _oai_finish(finish: str) -> str:
    if finish == "stop_sequence" or finish == "error":
        return "stop"
    return finish if finish in ("stop", "length", "tool_calls") else "stop"


def openai_chat_json(result: CompatResult, creq: CompatRequest, target: CompatTarget,
                     plan: CompatPlan) -> dict:
    message: dict = {"role": "assistant",
                     "content": result.text if (result.text or not result.tool_calls) else None}
    if result.tool_calls:
        message["tool_calls"] = [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": c["arguments"]}}
            for c in result.tool_calls]
    if result.reasoning:
        message["reasoning_content"] = result.reasoning
    body = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": target.name or target.model,
        "system_fingerprint": None,
        "choices": [{"index": 0, "message": message, "logprobs": None,
                     "finish_reason": _oai_finish(result.finish)}],
        "usage": openai_usage(result.usage),
    }
    if plan.mode == "agent":
        # Kept from the original handler: clients built against it read it.
        body["x_tools_used"] = result.tools_used
        if creq.include_trace:
            body["x_trace"] = result.trace
    return body


def _oai_sse(payload: Any) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def openai_chat_stream(events: AsyncIterator[dict], creq: CompatRequest,
                             target: CompatTarget, plan: CompatPlan) -> AsyncIterator[str]:
    """chat.completion.chunk frames, then [DONE]."""
    made = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model = target.name or target.model

    def chunk(delta: dict, finish: str | None = None, **extra: Any) -> str:
        payload: dict = {"id": made, "object": "chat.completion.chunk", "created": created,
                         "model": model, "system_fingerprint": None,
                         "choices": [{"index": 0, "delta": delta, "logprobs": None,
                                      "finish_reason": finish}]}
        if creq.include_usage:
            payload["usage"] = None
        payload.update(extra)
        return _oai_sse(payload)

    yield chunk({"role": "assistant", "content": ""})
    finish, usage = "stop", _oai_usage_dict()
    out_index: dict = {}
    async for ev in events:
        etype = ev["type"]
        if etype == "text":
            yield chunk({"content": ev["text"]})
        elif etype == "reasoning":
            yield chunk({"reasoning_content": ev["text"]})
        elif etype == "tool_start":
            out_index[ev["index"]] = len(out_index)
            yield chunk({"tool_calls": [{"index": out_index[ev["index"]], "id": ev["id"],
                                         "type": "function",
                                         "function": {"name": ev["name"], "arguments": ""}}]})
        elif etype == "tool_args":
            yield chunk({"tool_calls": [{"index": out_index.get(ev["index"], 0),
                                         "function": {"arguments": ev["delta"]}}]})
        elif etype == "trace":
            if creq.include_trace:
                yield chunk({}, x_event=ev["event"])
        elif etype == "error":
            yield _oai_sse(openai_error_body(int(ev.get("status") or 502), ev["message"]))
        elif etype == "done":
            finish, usage = ev.get("finish") or "stop", ev.get("usage") or usage
            if plan.mode == "agent" and creq.include_trace:
                yield chunk({}, x_tools_used=list(ev.get("tools_used") or []))
    yield chunk({}, _oai_finish(finish))
    if creq.include_usage:
        yield _oai_sse({"id": made, "object": "chat.completion.chunk", "created": created,
                        "model": model, "choices": [], "usage": openai_usage(usage)})
    yield "data: [DONE]\n\n"


# --------------------------------------------------------------------------- #
# OpenAI wire format: Responses (subset)
# --------------------------------------------------------------------------- #
def _oai_responses_text(content: Any, where: str) -> Any:
    if content is None or isinstance(content, str):
        return content or ""
    return _oai_check_parts(content, where)


def compat_parse_responses(body: dict) -> CompatRequest:
    """A /v1/responses body as a CompatRequest.

    Stateless: previous_response_id and item references would need this server
    to store every response, so they are refused and the client sends the whole
    conversation in `input` (every Responses client can).
    """
    if body.get("previous_response_id"):
        raise CompatError(400, "previous_response_id is not supported: this server is "
                               "stateless, send the full conversation in input",
                          param="previous_response_id")
    messages: list[dict] = []
    if body.get("instructions"):
        messages.append({"role": "system", "content": str(body["instructions"])})
    items = body.get("input")
    if isinstance(items, str):
        items = [{"role": "user", "content": items}]
    if not isinstance(items, list) or not items:
        raise CompatError(400, "input must be a string or a non-empty list of items",
                          param="input")
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise CompatError(400, f"input[{i}] must be an object", param="input")
        kind = item.get("type") or ("message" if item.get("role") else "")
        if kind == "message":
            role = item.get("role")
            if role not in ("system", "developer", "user", "assistant"):
                raise CompatError(400, f"input[{i}].role {role!r} is not supported",
                                  param="input")
            content = _oai_responses_text(item.get("content"), f"input[{i}]")
            messages.append({"role": "system" if role == "developer" else role,
                             "content": content})
        elif kind == "function_call":
            name = item.get("name")
            if not isinstance(name, str) or not name:
                raise CompatError(400, f"input[{i}]: function_call needs a name", param="input")
            call = {"id": str(item.get("call_id") or item.get("id") or ""), "type": "function",
                    "function": {"name": name, "arguments": str(item.get("arguments") or "{}")}}
            if messages and messages[-1]["role"] == "assistant":
                messages[-1].setdefault("tool_calls", []).append(call)
            else:
                messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
        elif kind == "function_call_output":
            output = item.get("output")
            if not isinstance(output, str):
                output = compat_text_of(_oai_check_parts(output, f"input[{i}].output")) \
                    if isinstance(output, list) else json.dumps(output, ensure_ascii=False)
            messages.append({"role": "tool", "tool_call_id": str(item.get("call_id") or ""),
                             "content": output})
        elif kind == "reasoning":
            continue
        elif kind == "item_reference":
            raise CompatError(400, "item_reference is not supported: this server is "
                                   "stateless, send the items themselves", param="input")
        else:
            raise CompatError(400, f"input[{i}]: unsupported item type {kind!r}", param="input")
    tools: list[dict] = []
    for i, tool in enumerate(body.get("tools") or []):
        if isinstance(tool, dict) and tool.get("type", "function") == "function":
            tools.append(_oai_tool_from_function(tool, f"tools[{i}]"))
        else:
            # Hosted tools (web_search, file_search, ...) are OpenAI's servers'
            # tools; there is nothing here to run them. Skipped, not fatal, so
            # a client that always sends one still works with its own tools.
            log_event(_oai_log, logging.INFO, "compat.tool_skipped",
                      tool_type=str((tool or {}).get("type") if isinstance(tool, dict) else ""))
    choice = body.get("tool_choice")
    if isinstance(choice, dict) and choice.get("type") == "function":
        choice = {"type": "function", "function": {"name": choice.get("name")}}
    elif isinstance(choice, dict):
        choice = None
    fmt = (body.get("text") or {}).get("format") if isinstance(body.get("text"), dict) else None
    response_format = None
    if isinstance(fmt, dict) and fmt.get("type") == "json_object":
        response_format = {"type": "json_object"}
    elif isinstance(fmt, dict) and fmt.get("type") == "json_schema":
        response_format = {"type": "json_schema", "json_schema": {
            "name": fmt.get("name") or "response", "schema": fmt.get("schema") or {}}}
    reasoning = body.get("reasoning") if isinstance(body.get("reasoning"), dict) else {}
    creq = CompatRequest(
        model=str(body.get("model") or ""), messages=messages, tools=tools,
        tool_choice=_oai_check_tool_choice(choice, tools),
        max_tokens=_oai_num(body, "max_output_tokens", int, low=1),
        temperature=_oai_num(body, "temperature", float, low=0.0, high=2.0),
        top_p=_oai_num(body, "top_p", float, low=0.0, high=1.0),
        response_format=response_format,
        reasoning_effort=str(reasoning["effort"]) if reasoning.get("effort") else None,
        parallel_tool_calls=(bool(body["parallel_tool_calls"])
                             if body.get("parallel_tool_calls") is not None else None),
        include_trace=bool(body.get("x_include_trace")),
        stream=bool(body.get("stream")),
    )
    if creq.reasoning_effort == "none":
        creq.thinking = False
    return creq


def _oai_responses_usage(usage: dict) -> dict:
    p, c = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
    return {"input_tokens": p,
            "input_tokens_details": {"cached_tokens": int(usage.get("cached_tokens") or 0)},
            "output_tokens": c,
            "output_tokens_details": {"reasoning_tokens": int(usage.get("reasoning_tokens") or 0)},
            "total_tokens": p + c}


def _oai_response_shell(rid: str, creq: CompatRequest, target: CompatTarget, created: int,
                        status: str) -> dict:
    return {
        "id": rid, "object": "response", "created_at": created, "status": status,
        "error": None, "incomplete_details": None,
        "instructions": next((compat_text_of(m.get("content")) for m in creq.messages
                              if m["role"] == "system"), None),
        "max_output_tokens": creq.max_tokens, "model": target.name or target.model,
        "output": [], "parallel_tool_calls": creq.parallel_tool_calls is not False,
        "previous_response_id": None, "reasoning": {"effort": creq.reasoning_effort,
                                                    "summary": None},
        "store": False, "temperature": creq.temperature, "top_p": creq.top_p,
        "text": {"format": {"type": "json_object"} if creq.response_format else {"type": "text"}},
        "tool_choice": creq.tool_choice or "auto",
        "tools": [dict(t["function"], type="function") for t in creq.tools],
        "truncation": "disabled", "usage": None, "metadata": {},
    }


def _oai_msg_item(item_id: str, text: str, status: str = "completed") -> dict:
    return {"id": item_id, "type": "message", "status": status, "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def _oai_finish_response(resp: dict, finish: str, usage: dict, output: list) -> dict:
    resp["output"] = output
    resp["usage"] = _oai_responses_usage(usage)
    if finish == "length":
        resp["status"] = "incomplete"
        resp["incomplete_details"] = {"reason": "max_output_tokens"}
    elif finish == "error":
        resp["status"] = "failed"
    else:
        resp["status"] = "completed"
    resp["output_text"] = "".join(
        part.get("text", "") for item in output if item.get("type") == "message"
        for part in item.get("content") or [])
    return resp


def openai_responses_json(result: CompatResult, creq: CompatRequest, target: CompatTarget,
                          plan: CompatPlan) -> dict:
    output: list[dict] = []
    if result.reasoning:
        output.append({"id": f"rs_{uuid.uuid4().hex[:24]}", "type": "reasoning",
                       "summary": [{"type": "summary_text", "text": result.reasoning}]})
    if result.text or not result.tool_calls:
        output.append(_oai_msg_item(f"msg_{uuid.uuid4().hex[:24]}", result.text))
    for call in result.tool_calls:
        output.append({"id": f"fc_{uuid.uuid4().hex[:24]}", "type": "function_call",
                       "status": "completed", "call_id": call["id"], "name": call["name"],
                       "arguments": call["arguments"]})
    resp = _oai_response_shell(f"resp_{uuid.uuid4().hex[:24]}", creq, target,
                               int(time.time()), "completed")
    resp = _oai_finish_response(resp, result.finish, result.usage, output)
    if plan.mode == "agent":
        resp["x_tools_used"] = result.tools_used
        if creq.include_trace:
            resp["x_trace"] = result.trace
    return resp


async def openai_responses_stream(events: AsyncIterator[dict], creq: CompatRequest,
                                  target: CompatTarget, plan: CompatPlan) -> AsyncIterator[str]:
    """Responses API events: created, output items and their deltas, completed."""
    seq = [0]

    def frame(etype: str, **data: Any) -> str:
        data = {"type": etype, "sequence_number": seq[0], **data}
        seq[0] += 1
        return f"event: {etype}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    created = int(time.time())
    resp = _oai_response_shell(f"resp_{uuid.uuid4().hex[:24]}", creq, target, created,
                               "in_progress")
    yield frame("response.created", response=dict(resp))
    yield frame("response.in_progress", response=dict(resp))
    output: list[dict] = []
    cur: dict = {}            # the open item: {"kind", "item", "index", "text"}
    calls: dict = {}          # engine tool index -> open function_call state

    def close_current() -> list[str]:
        out: list[str] = []
        if not cur:
            return out
        item, index, text = cur["item"], cur["index"], "".join(cur["text"])
        if cur["kind"] == "message":
            item.update(_oai_msg_item(item["id"], text))
            part = item["content"][0]
            out.append(frame("response.output_text.done", item_id=item["id"],
                             output_index=index, content_index=0, text=text))
            out.append(frame("response.content_part.done", item_id=item["id"],
                             output_index=index, content_index=0, part=part))
        else:
            item["summary"] = [{"type": "summary_text", "text": text}]
            out.append(frame("response.reasoning_summary_text.done", item_id=item["id"],
                             output_index=index, summary_index=0, text=text))
            out.append(frame("response.reasoning_summary_part.done", item_id=item["id"],
                             output_index=index, summary_index=0,
                             part={"type": "summary_text", "text": text}))
        out.append(frame("response.output_item.done", output_index=index, item=item))
        cur.clear()
        return out

    def open_item(kind: str) -> list[str]:
        out = close_current()
        index = len(output)
        if kind == "message":
            item = {"id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message",
                    "status": "in_progress", "role": "assistant", "content": []}
            output.append(item)
            out.append(frame("response.output_item.added", output_index=index, item=dict(item)))
            out.append(frame("response.content_part.added", item_id=item["id"],
                             output_index=index, content_index=0,
                             part={"type": "output_text", "text": "", "annotations": []}))
        else:
            item = {"id": f"rs_{uuid.uuid4().hex[:24]}", "type": "reasoning", "summary": []}
            output.append(item)
            out.append(frame("response.output_item.added", output_index=index, item=dict(item)))
            out.append(frame("response.reasoning_summary_part.added", item_id=item["id"],
                             output_index=index, summary_index=0,
                             part={"type": "summary_text", "text": ""}))
        cur.update(kind=kind, item=item, index=index, text=[])
        return out

    finish, usage = "stop", _oai_usage_dict()
    failed: dict | None = None
    tools_used: list = []
    async for ev in events:
        etype = ev["type"]
        if etype == "text":
            if cur.get("kind") != "message":
                for f in open_item("message"):
                    yield f
            cur["text"].append(ev["text"])
            yield frame("response.output_text.delta", item_id=cur["item"]["id"],
                        output_index=cur["index"], content_index=0, delta=ev["text"])
        elif etype == "reasoning":
            if cur.get("kind") != "reasoning":
                for f in open_item("reasoning"):
                    yield f
            cur["text"].append(ev["text"])
            yield frame("response.reasoning_summary_text.delta", item_id=cur["item"]["id"],
                        output_index=cur["index"], summary_index=0, delta=ev["text"])
        elif etype == "tool_start":
            for f in close_current():
                yield f
            item = {"id": f"fc_{uuid.uuid4().hex[:24]}", "type": "function_call",
                    "status": "in_progress", "call_id": ev["id"], "name": ev["name"],
                    "arguments": ""}
            calls[ev["index"]] = {"item": item, "index": len(output)}
            output.append(item)
            yield frame("response.output_item.added", output_index=len(output) - 1,
                        item=dict(item))
        elif etype == "tool_args":
            state = calls.get(ev["index"])
            if state is not None:
                state["item"]["arguments"] += ev["delta"]
                yield frame("response.function_call_arguments.delta",
                            item_id=state["item"]["id"], output_index=state["index"],
                            delta=ev["delta"])
        elif etype == "tool_end":
            state = calls.get(ev["index"])
            if state is not None:
                item = state["item"]
                item["arguments"], item["status"] = ev["arguments"], "completed"
                yield frame("response.function_call_arguments.done", item_id=item["id"],
                            output_index=state["index"], arguments=ev["arguments"])
                yield frame("response.output_item.done", output_index=state["index"],
                            item=dict(item))
        elif etype == "trace":
            if creq.include_trace:
                yield frame("response.x_trace", event=ev["event"])
        elif etype == "error":
            failed = {"code": "server_error", "message": ev["message"]}
            yield frame("error", code="server_error", message=ev["message"], param=None)
        elif etype == "done":
            finish, usage = ev.get("finish") or "stop", ev.get("usage") or usage
            tools_used = list(ev.get("tools_used") or [])
    for f in close_current():
        yield f
    resp = _oai_finish_response(resp, "error" if failed else finish, usage, output)
    if failed:
        resp["error"] = failed
    if plan.mode == "agent":
        resp["x_tools_used"] = tools_used
    final_type = ("response.failed" if failed else
                  "response.incomplete" if resp["status"] == "incomplete" else
                  "response.completed")
    yield frame(final_type, response=resp)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
def register_openai_compat(app: Any, ctx: Any) -> None:
    """GET /v1/models, POST /v1/chat/completions, POST /v1/responses."""

    @app.get("/v1/models")
    async def v1_models(request: _OaiRequest):
        anthropic = bool(request.headers.get("anthropic-version"))
        provider = "anthropic" if anthropic else "openai"
        try:
            user = await compat_user(ctx, request)
        except CompatError as exc:
            return compat_error_response(exc, provider)
        entries = await compat_model_entries(ctx, user)
        if anthropic:
            data = [{"type": "model", "id": e["id"], "display_name": e.get("name") or e["id"],
                     "created_at": "1970-01-01T00:00:00Z"} for e in entries]
            return {"data": data, "has_more": False,
                    "first_id": data[0]["id"] if data else None,
                    "last_id": data[-1]["id"] if data else None}
        return {"object": "list", "data": [
            {"id": e["id"], "object": "model", "created": 0, "owned_by": "local",
             "x_mode": e["mode"]} for e in entries]}

    @app.get("/v1/models/{model_id:path}")
    async def v1_model(model_id: str, request: _OaiRequest):
        anthropic = bool(request.headers.get("anthropic-version"))
        provider = "anthropic" if anthropic else "openai"
        try:
            user = await compat_user(ctx, request)
        except CompatError as exc:
            return compat_error_response(exc, provider)
        for e in await compat_model_entries(ctx, user):
            if e["id"] == model_id:
                if anthropic:
                    return {"type": "model", "id": e["id"],
                            "display_name": e.get("name") or e["id"],
                            "created_at": "1970-01-01T00:00:00Z"}
                return {"id": e["id"], "object": "model", "created": 0, "owned_by": "local",
                        "x_mode": e["mode"]}
        return compat_error_response(
            CompatError(404, f"model {model_id!r} not found", code="model_not_found"), provider)

    @app.post("/v1/chat/completions")
    async def v1_chat_completions(request: _OaiRequest):
        """OpenAI chat completions: raw model or this app's agent (see module doc)."""
        return await compat_handle(app, ctx, request, provider="openai", endpoint="v1_chat",
                                   parse=compat_parse_chat, to_json=openai_chat_json,
                                   to_stream=openai_chat_stream)

    @app.post("/v1/responses")
    async def v1_responses(request: _OaiRequest):
        """OpenAI Responses API (stateless subset: no previous_response_id)."""
        return await compat_handle(app, ctx, request, provider="openai",
                                   endpoint="v1_responses", parse=compat_parse_responses,
                                   to_json=openai_responses_json,
                                   to_stream=openai_responses_stream)


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'CompatError',
    'CompatPlan',
    'CompatRequest',
    'CompatResult',
    'CompatTarget',
    'anthropic_error_body',
    'compat_admit',
    'compat_agent_events',
    'compat_backend_caps',
    'compat_check_ready',
    'compat_collect',
    'compat_error_response',
    'compat_events',
    'compat_format_instruction',
    'compat_handle',
    'compat_has_images',
    'compat_image_parts',
    'compat_json_answer',
    'compat_json_body',
    'compat_metric',
    'compat_model_entries',
    'compat_parse_chat',
    'compat_parse_responses',
    'compat_prepare',
    'compat_raw_events',
    'compat_release',
    'compat_render_messages',
    'compat_resolve_target',
    'compat_served_models',
    'compat_stream_response',
    'compat_text_of',
    'compat_tool_names',
    'compat_tool_section',
    'compat_user',
    'compat_vision',
    'compat_visible_profiles',
    'openai_chat_json',
    'openai_chat_stream',
    'openai_error_body',
    'openai_responses_json',
    'openai_responses_stream',
    'openai_usage',
    'register_openai_compat',
    '_OaiRequest',
    '_OaiStopFilter',
    '_oai_call_args',
    '_oai_check_parts',
    '_oai_check_tool_choice',
    '_oai_clip',
    '_oai_num',
    '_oai_sse',
    '_oai_stop_list',
    '_oai_tool_from_function',
    '_oai_usage_dict',
]
