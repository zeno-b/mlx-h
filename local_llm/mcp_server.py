"""This app as an MCP server: its tools, offered to MCP clients over Streamable HTTP.

Claude Code, IDE agents and other MCP clients can use the same tools this app's
own agent uses (files in the user's workspace, web search, the knowledge base,
memory, the gated capability tools) without a second copy of any of them:

    POST   /mcp    JSON-RPC: initialize, ping, tools/list, tools/call, notifications
    GET    /mcp    405 (no server-initiated stream; replies come with each POST)
    DELETE /mcp    end the session

Who may do what is decided exactly as for chat, never more:

  * auth is the same USER dependency as every route (Bearer session or API
    token, or the session cookie); MCP_SERVER_ENABLED=0 (the default) answers
    404 to everyone, because an endpoint that runs tools is a new way in;
  * the tool set is the default agent's registry, narrowed by the caller's role
    (ROLE_CAPABILITIES via config.capabilities_for_role), or an agent profile's
    when the client names one (X-Agent-Id header or ?agent_id= at initialize),
    intersected with the role the same way the chat route does;
  * every call goes through ToolRegistry.call with a TurnContext bound for the
    caller and interactive=False, so the approval policy applies (an "ask"
    becomes TOOL_APPROVAL_UNATTENDED, deny by default: nobody can answer a
    prompt here) and the call lands in tool_calls / tool_decisions like any
    other.

Sessions: initialize issues an Mcp-Session-Id bound to the user who created it;
any other user presenting it gets 404 (indistinguishable from an unknown id).
They live in memory with an idle timeout, so a restart simply makes clients
initialize again, which the protocol already requires them to handle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import threading
import time
from dataclasses import replace as dataclass_replace
from typing import Any

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .turnctx import *  # noqa: F401,F403
from .policy import *  # noqa: F401,F403
from .mcp import *  # noqa: F401,F403
from .tools import *  # noqa: F401,F403

# Module level, because route annotations are strings here (postponed
# evaluation) and FastAPI resolves them in this module's globals.
try:
    from fastapi import Request as _McpsRequest  # noqa: F401
except ImportError:  # pragma: no cover - fastapi is optional for --selftest
    _McpsRequest = Any  # type: ignore


MCP_SERVER_PATH = "/mcp"
_MCPS_SESSIONS: dict[str, dict] = {}
_MCPS_LOCK = threading.Lock()
_MCPS_IDLE_S = 12 * 3600
_MCPS_PER_USER = 16
_MCPS_PAGE = 50
# Loop-only tools: final_answer ends an agent turn and delegate_task spawns a
# child agent; neither means anything to an outside client.
_MCPS_HIDDEN = {"final_answer", "delegate_task"}
_MCPS_PROFILE_CACHE: dict[tuple, Any] = {}


def _mcps_prune(now: float) -> None:
    for sid in [s for s, row in _MCPS_SESSIONS.items() if now - row["last"] > _MCPS_IDLE_S]:
        _MCPS_SESSIONS.pop(sid, None)


def mcp_server_new_session(user_id: str, protocol: str, agent_id: str | None) -> str:
    """Issue a session id bound to this user (oldest dropped past the cap)."""
    now = time.time()
    sid = secrets.token_urlsafe(24)
    with _MCPS_LOCK:
        _mcps_prune(now)
        mine = sorted((row["last"], s) for s, row in _MCPS_SESSIONS.items()
                      if row["user_id"] == user_id)
        for _last, old in mine[:max(0, len(mine) - _MCPS_PER_USER + 1)]:
            _MCPS_SESSIONS.pop(old, None)
        _MCPS_SESSIONS[sid] = {"user_id": user_id, "protocol": protocol,
                               "agent_id": agent_id, "created": now, "last": now}
    return sid


def mcp_server_session(sid: str | None, user_id: str) -> dict | None:
    """The live session for this id and user, or None (unknown, expired or not theirs)."""
    if not sid:
        return None
    now = time.time()
    with _MCPS_LOCK:
        row = _MCPS_SESSIONS.get(sid)
        if row is None or now - row["last"] > _MCPS_IDLE_S or row["user_id"] != user_id:
            return None
        row["last"] = now
        return dict(row)


def _mcps_profile_registry(ctx: Any, row: dict, caps: list[str]) -> Any:
    key = (row.get("id"), row.get("updated_at"), tuple(caps), id(ctx.config))
    hit = _MCPS_PROFILE_CACHE.get(key)
    if hit is not None:
        return hit
    tools = ",".join(tools_for_capabilities(caps)) or "final_answer"
    cfg = dataclass_replace(ctx.config, agent_tools=tools,
                            rag_enabled=rag_for_capabilities(caps) and ctx.config.rag_enabled)
    registry = ToolRegistry(cfg, ctx.db)
    if len(_MCPS_PROFILE_CACHE) > 32:
        _MCPS_PROFILE_CACHE.clear()
    _MCPS_PROFILE_CACHE[key] = registry
    return registry


def mcp_server_toolset(ctx: Any, user: dict, agent_id: str | None = None) -> tuple[Any, list]:
    """(registry, [Tool]) this user may use over MCP. LookupError for a bad profile."""
    role = str(user.get("role") or "user")
    role_caps = capabilities_for_role(role, ctx.config)
    restricted = set(role_caps) != set(CAPABILITY_ORDER)
    if agent_id:
        row = ctx.db.get_agent(str(agent_id))
        if row is None or (not row.get("enabled") and role != "admin"):
            raise LookupError("agent not found")
        wanted = set(row.get("capabilities") or [])
        caps = [c for c in CAPABILITY_ORDER if c in wanted and c in set(role_caps)]
        registry = _mcps_profile_registry(ctx, row, caps)
        names = registry.names()
    else:
        registry = ctx.registry
        names = registry.names()
        if restricted:
            allowed = set(tools_for_capabilities(role_caps))
            names = [n for n in names if n in allowed]
    tools = [registry.get(n) for n in names if n not in _MCPS_HIDDEN]
    return registry, [t for t in tools if t is not None]


def mcp_tool_descriptor(tool: Any) -> dict:
    """A registry Tool in MCP's tools/list shape, with behaviour hints.

    Parameters are prose in this registry, so properties carry a description
    and no type: the handlers coerce what they get, and a guessed "string"
    would make strict clients refuse a perfectly good number.
    """
    props = {str(k): {"description": str(v)} for k, v in (tool.parameters or {}).items()}
    risk = classify_tool_risk(tool.name, {}, tool)
    outside = bool(tool_taints(tool.name, tool)) or risk == "external"
    return {
        "name": tool.name,
        "title": tool.name.replace("_", " "),
        "description": tool.description,
        "inputSchema": {"type": "object", "properties": props,
                        "required": [r for r in tool.required if r in props]},
        "annotations": {
            "readOnlyHint": risk == "read",
            "destructiveHint": risk in ("write", "exec", "external"),
            "openWorldHint": outside,
        },
    }


def _mcps_error(mid: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def _mcps_result(mid: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


async def _mcps_dispatch(ctx: Any, user: dict, session: dict, message: Any) -> dict | None:
    """One JSON-RPC message from a client. None for notifications and responses."""
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return _mcps_error(None, -32600, "invalid request")
    method = message.get("method")
    if method is None:
        return None                       # a response to us; we never ask anything
    if "id" not in message:
        return None                       # notifications/initialized, cancelled, ...
    mid = message.get("id")
    params = message.get("params") if isinstance(message.get("params"), dict) else {}
    if method == "ping":
        return _mcps_result(mid, {})
    if method == "logging/setLevel":
        return _mcps_result(mid, {})
    if method == "initialize":
        return _mcps_error(mid, -32600, "already initialized")
    if method not in ("tools/list", "tools/call"):
        return _mcps_error(mid, -32601, f"method not found: {method}")
    try:
        registry, tools = await asyncio.to_thread(
            mcp_server_toolset, ctx, user, session.get("agent_id"))
    except LookupError as exc:
        return _mcps_error(mid, -32602, str(exc))
    if method == "tools/list":
        cursor = params.get("cursor")
        try:
            start = int(cursor) if cursor not in (None, "") else 0
        except (TypeError, ValueError):
            return _mcps_error(mid, -32602, "invalid cursor")
        page = tools[start:start + _MCPS_PAGE]
        result: dict = {"tools": [mcp_tool_descriptor(t) for t in page]}
        if start + _MCPS_PAGE < len(tools):
            result["nextCursor"] = str(start + _MCPS_PAGE)
        return _mcps_result(mid, result)
    name = str(params.get("name") or "")
    arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
    if name not in {t.name for t in tools}:
        return _mcps_error(mid, -32602, f"Unknown tool: {name}")
    # A profile's own tool policy (profile_tool_policy) applies to its MCP
    # session exactly as it does in chat: the turn names the profile.
    profile = None
    if session.get("agent_id"):
        profile = await asyncio.to_thread(ctx.db.get_agent, str(session["agent_id"]))
    turn = TurnContext(user_id=user["id"], conversation_id=None,
                       role=str(user.get("role") or "user"), interactive=False,
                       agent_profile=profile, extras={"source": "mcp"})
    token = set_current_turn(turn)
    try:
        # to_thread copies this context: the worker sees the turn and the
        # acting user the USER dependency bound.
        out, error = await asyncio.to_thread(registry.call, name, arguments, None)
    finally:
        reset_current_turn(token)
    log_event(get_logger("mcp"), logging.INFO, "mcp_server.call", tool=name,
              user_id=user["id"], ok=error is None)
    return _mcps_result(mid, {"content": [{"type": "text", "text": str(out)}],
                              "isError": error is not None})


def register_mcp_server(app: Any, ctx: Any) -> None:
    """Mount POST/GET/DELETE /mcp (see the module docstring)."""
    from fastapi.responses import JSONResponse, Response

    config = ctx.config

    def disabled() -> Any:
        if getattr(config, "mcp_server_enabled", False):
            return None
        return JSONResponse({"error": "the MCP server is disabled (MCP_SERVER_ENABLED=0)"},
                            status_code=404)

    def bad_origin(request: _McpsRequest) -> Any:
        # DNS-rebinding defence the transport spec requires: a browser page on
        # another origin must not drive this endpoint with the user's cookie.
        origin = request.headers.get("origin")
        if not origin:
            return None
        host = (request.headers.get("host") or "").lower()
        if origin.lower().split("://", 1)[-1].rstrip("/") == host:
            return None
        return JSONResponse({"error": "origin not allowed"}, status_code=403)

    def reply(request: _McpsRequest, payload: Any, headers: dict | None = None) -> Any:
        accept = (request.headers.get("accept") or "").lower()
        if "text/event-stream" in accept and "application/json" not in accept:
            body = f"event: message\ndata: {json.dumps(payload)}\n\n"
            return Response(body, media_type="text/event-stream", headers=headers or {})
        return JSONResponse(payload, headers=headers or {})

    @app.post(MCP_SERVER_PATH)
    async def mcp_post(request: _McpsRequest, user: dict = ctx.USER):
        refusal = disabled() or bad_origin(request)
        if refusal is not None:
            return refusal
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse(_mcps_error(None, -32700, "parse error"), status_code=400)
        messages = body if isinstance(body, list) else [body]
        if not messages:
            return JSONResponse(_mcps_error(None, -32600, "empty batch"), status_code=400)
        first = messages[0] if isinstance(messages[0], dict) else {}
        if first.get("method") == "initialize":
            if len(messages) != 1 or "id" not in first:
                return JSONResponse(_mcps_error(first.get("id"), -32600,
                                                "initialize must be sent alone"),
                                    status_code=400)
            params = first.get("params") if isinstance(first.get("params"), dict) else {}
            asked = str(params.get("protocolVersion") or "")
            agreed = asked if asked in SUPPORTED_PROTOCOL_VERSIONS else PROTOCOL_VERSION
            agent_id = (request.headers.get("x-agent-id")
                        or request.query_params.get("agent_id") or None)
            if agent_id:
                try:
                    await asyncio.to_thread(mcp_server_toolset, ctx, user, agent_id)
                except LookupError:
                    return JSONResponse(_mcps_error(first.get("id"), -32602,
                                                    "agent not found"), status_code=404)
            sid = mcp_server_new_session(user["id"], agreed, agent_id)
            log_event(get_logger("mcp"), logging.INFO, "mcp_server.session",
                      user_id=user["id"], protocol=agreed, agent_id=agent_id)
            result = {
                "protocolVersion": agreed,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": APP_NAME, "title": APP_NAME, "version": str(UI_BUILD)},
                "instructions": (f"Tools of {APP_NAME}, run as you, in your own workspace and "
                                 "under your account's tool policy. Calls that would need "
                                 "approval are refused here unless the policy allows "
                                 "unattended use."),
            }
            return reply(request, _mcps_result(first.get("id"), result),
                         {"Mcp-Session-Id": sid})
        version = request.headers.get("mcp-protocol-version")
        if version and version not in SUPPORTED_PROTOCOL_VERSIONS:
            return JSONResponse(_mcps_error(None, -32600,
                                            f"unsupported MCP-Protocol-Version {version}"),
                                status_code=400)
        sid = request.headers.get("mcp-session-id")
        if not sid:
            return JSONResponse(_mcps_error(None, -32600, "missing Mcp-Session-Id; "
                                            "send initialize first"), status_code=400)
        session = mcp_server_session(sid, user["id"])
        if session is None:
            return JSONResponse(_mcps_error(None, -32001, "session not found"),
                                status_code=404)
        replies = []
        for message in messages:
            answer = await _mcps_dispatch(ctx, user, session, message)
            if answer is not None:
                replies.append(answer)
        if not replies:
            return Response(status_code=202)
        return reply(request, replies if isinstance(body, list) else replies[0])

    @app.get(MCP_SERVER_PATH)
    async def mcp_get(user: dict = ctx.USER):
        refusal = disabled()
        if refusal is not None:
            return refusal
        return JSONResponse({"error": "no server-initiated stream; POST requests"},
                            status_code=405, headers={"Allow": "POST, DELETE"})

    @app.delete(MCP_SERVER_PATH)
    async def mcp_delete(request: _McpsRequest, user: dict = ctx.USER):
        refusal = disabled()
        if refusal is not None:
            return refusal
        sid = request.headers.get("mcp-session-id")
        if mcp_server_session(sid, user["id"]) is None:
            return JSONResponse({"error": "session not found"}, status_code=404)
        with _MCPS_LOCK:
            _MCPS_SESSIONS.pop(sid, None)
        return {"ok": True}


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'MCP_SERVER_PATH',
    'mcp_server_new_session',
    'mcp_server_session',
    'mcp_server_toolset',
    'mcp_tool_descriptor',
    'register_mcp_server',
]
