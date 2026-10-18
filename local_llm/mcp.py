"""A Model Context Protocol client, so external MCP servers show up as tools.

Scope is deliberate: the stdio and Streamable HTTP transports, `initialize`,
`tools/list` (with pagination and `notifications/tools/list_changed`) and
`tools/call`. That is the part of MCP that turns somebody else's server into a
tool this agent can use. Resources, prompts, sampling and elicitation are not
implemented; a server that offers them still works, its tools are just all we
take.

Synchronous by design. ToolRegistry handlers are synchronous and the agent
already runs them off the event loop with asyncio.to_thread, so an async client
here would only add a second concurrency model to reason about.

Processes are shared PROCESS-WIDE, not per registry. A ToolRegistry is built for
every subagent and every task run, and starting a fresh `npx` server each time
would cost seconds per turn and leak processes.

Two server shapes are accepted in MCP_SERVERS:
  stdio  {"name", "command", "args"?, "env"?, "cwd"?}            (unchanged)
  http   {"name", "url", "headers"?, "bearer"?, "type": "http"}  (Streamable HTTP)
Header and env values may say ${NAME} to read NAME from the server's own
environment at connect time, so a token does not have to live in MCP_SERVERS.

Why stdio children no longer inherit os.environ: a third-party `npx` server
started with the whole server environment received AUTH_*, O365_CLIENT_SECRET,
NODE_TOKEN and every HF token, exactly the leak the code-execution sandbox was
built to stop. A child now gets the sandbox allowlist (ToolRegistry._sandbox_env,
handed over by the registry) plus the env its spec names explicitly.

MCP tools carry the capability "mcp:<server>": an agent profile or role reaches
them only by granting "mcp" (every server) or "mcp:<server>". configure()
publishes one capability key per server in config.CAPABILITY_GROUPS so the
Agents screen offers it and the profile/role validation accepts it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403


# The newest revision is offered first. A server answers with the revision it
# speaks; any of these is accepted, and a server that rejects the offer outright
# (old servers answer an unknown version with a JSON-RPC error) is asked again
# with the next one down.
PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

# tools/list pages followed before giving up: a server whose cursor never ends
# must not hold registry construction forever.
_MCP_MAX_PAGES = 50
# Backoff for a server whose start or listing failed: 5s, 10s, 20s ... 300s.
_MCP_BACKOFF_BASE_S = 5.0
_MCP_BACKOFF_MAX_S = 300.0
_MCP_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
# Used only when no registry handed over its sandbox environment (a direct
# MCPManager in a test or script): what a toolchain needs to FIND things.
_MCP_FALLBACK_ENV = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "TZ", "LANG",
                     "LC_ALL", "LC_CTYPE", "TMPDIR", "TEMP", "TMP", "NVM_DIR",
                     "NODE_PATH", "npm_config_cache", "VIRTUAL_ENV",
                     "SSL_CERT_FILE", "SSL_CERT_DIR")


@dataclass
class MCPToolSpec:
    """One remote tool, in the shape ToolRegistry needs."""

    server: str
    remote_name: str
    local_name: str
    description: str
    parameters: dict[str, str] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    # readOnlyHint / destructiveHint / openWorldHint / idempotentHint as the
    # server sent them; policy.py derives the risk class from these.
    annotations: dict | None = None
    title: str = ""


def _local_name(server: str, tool: str) -> str:
    """A registry-safe name that says where the tool came from.

    Prefixed rather than flattened so an MCP server cannot shadow a built-in
    tool: a server called "core" offering "read_file" must not become the
    read_file the agent already relies on.
    """
    def clean(text: str) -> str:
        out = "".join(c if (c.isalnum() or c == "_") else "_" for c in str(text).lower())
        return out.strip("_")[:40]
    return f"mcp_{clean(server)}_{clean(tool)}"


def _describe_schema(schema: Any) -> tuple[dict[str, str], list[str]]:
    """Turn a JSON Schema into (parameter descriptions, required names).

    The registry advertises parameters to the model as one line of prose each,
    so the useful part of a schema is the description and the type, not the
    constraints. Anything unparseable degrades to "no arguments" rather than
    breaking the tool.
    """
    if not isinstance(schema, dict):
        return {}, []
    props = schema.get("properties")
    if not isinstance(props, dict):
        return {}, []
    parameters: dict[str, str] = {}
    for name, spec in props.items():
        if not isinstance(spec, dict):
            parameters[str(name)] = "value"
            continue
        kind = spec.get("type") or "value"
        text = str(spec.get("description") or "").strip()
        parameters[str(name)] = f"{text} ({kind})" if text else str(kind)
    required = [str(r) for r in (schema.get("required") or []) if isinstance(r, str)]
    return parameters, required


def _mcp_expand(value: Any) -> str:
    """${NAME} references resolved from this process's environment."""
    return _MCP_ENV_REF.sub(lambda m: os.environ.get(m.group(1), ""), str(value))


def _mcp_fallback_env() -> dict[str, str]:
    env = {k: os.environ[k] for k in _MCP_FALLBACK_ENV if k in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin")
    return env


def mcp_child_env(base: dict | None, explicit: dict | None) -> dict[str, str]:
    """The environment a stdio MCP server starts with: sandbox base + its own env."""
    env = dict(base) if base is not None else _mcp_fallback_env()
    for key, value in (explicit or {}).items():
        env[str(key)] = _mcp_expand(value)
    return env


def _mcp_error_text(detail: Any) -> str:
    text = detail.get("message") if isinstance(detail, dict) else str(detail)
    return str(text)[:300]


class MCPProtocolError(RuntimeError):
    """The server answered with a JSON-RPC error (as opposed to not answering)."""


class _MCPClientBase:
    """What both transports share: handshake, listing, calling, backoff."""

    transport = "stdio"

    def __init__(self, name: str, timeout: float = 30.0):
        self.name = name
        self.timeout = max(1.0, float(timeout))
        self.tools: list[MCPToolSpec] = []
        self.error: str = ""
        self.protocol_version: str = ""
        # Set by notifications/tools/list_changed: the cached list is stale and
        # the next MCPManager.tools() lists this server again.
        self.tools_stale = False
        self._failures = 0
        self._retry_at = 0.0

    # Subclasses: start, stop, alive, _request, _notify.
    def start(self) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    def stop(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def alive(self) -> bool:  # pragma: no cover - abstract
        return False

    def _request(self, method: str, params: dict) -> dict:  # pragma: no cover
        raise NotImplementedError

    def _notify(self, method: str, params: dict) -> None:  # pragma: no cover
        raise NotImplementedError

    # ------------------------------------------------------------ backoff --
    def _note_failure(self, message: str) -> None:
        """A failed start or listing: remembered, and retried after a backoff.

        The error used to be sticky: one tools/list timeout at boot left the
        server toolless until the process restarted. Now it is retried, but not
        on every registry build (each subagent and task builds one), which
        would stall every turn on a server that is down.
        """
        self.error = message
        self._failures += 1
        delay = min(_MCP_BACKOFF_MAX_S, _MCP_BACKOFF_BASE_S * (2 ** (self._failures - 1)))
        self._retry_at = time.time() + delay
        log(f"MCP server {self.name!r}: {message} (retry in {delay:.0f}s)", logging.WARNING)

    def _note_success(self) -> None:
        self.error = ""
        self._failures = 0
        self._retry_at = 0.0

    def due_for_retry(self) -> bool:
        return time.time() >= self._retry_at

    def needs_listing(self) -> bool:
        if self.tools_stale:
            return True
        if self.tools:
            return False
        return not self.error or self.due_for_retry()

    # ------------------------------------------------------------ protocol --
    def _handle_message(self, message: Any) -> None:
        """A server-initiated message that is not our reply."""
        if isinstance(message, dict) and \
                message.get("method") == "notifications/tools/list_changed":
            self.tools_stale = True

    def _initialize(self) -> None:
        """Negotiate a protocol revision, newest first. Raises on failure."""
        last: Exception | None = None
        for version in SUPPORTED_PROTOCOL_VERSIONS:
            try:
                result = self._request("initialize", {
                    "protocolVersion": version,
                    "capabilities": {},
                    "clientInfo": {"name": APP_NAME, "version": str(UI_BUILD)},
                })
            except MCPProtocolError as exc:
                last = exc          # an old server refusing the offer: go lower
                continue
            agreed = str(result.get("protocolVersion") or version)
            if agreed not in SUPPORTED_PROTOCOL_VERSIONS:
                raise RuntimeError(f"unsupported protocol version {agreed!r}")
            self.protocol_version = agreed
            self._notify("notifications/initialized", {})
            return
        raise RuntimeError(str(last) if last else "initialize failed")

    # ----------------------------------------------------------------- tools --
    def list_tools(self) -> list[MCPToolSpec]:
        if not self.start():
            self._note_failure(self.error or "would not start")
            return []
        entries: list[dict] = []
        cursor: str | None = None
        seen: set[str] = set()
        try:
            for _page in range(_MCP_MAX_PAGES):
                result = self._request("tools/list", {"cursor": cursor} if cursor else {})
                entries.extend(e for e in (result.get("tools") or []) if isinstance(e, dict))
                cursor = result.get("nextCursor")
                if not cursor or not isinstance(cursor, str) or cursor in seen:
                    break
                seen.add(cursor)
        except Exception as exc:
            self._note_failure(f"tools/list failed: {exc}")
            return []
        specs: list[MCPToolSpec] = []
        for entry in entries:
            remote = str(entry.get("name") or "").strip()
            if not remote:
                continue
            parameters, required = _describe_schema(entry.get("inputSchema"))
            annotations = entry.get("annotations")
            specs.append(MCPToolSpec(
                server=self.name,
                remote_name=remote,
                local_name=_local_name(self.name, remote),
                description=(str(entry.get("description") or remote).strip()
                             + f" (via the {self.name} MCP server)"),
                parameters=parameters,
                required=required,
                annotations=dict(annotations) if isinstance(annotations, dict) else None,
                title=str(entry.get("title") or (annotations.get("title") if
                                                 isinstance(annotations, dict) else "") or ""),
            ))
        self.tools = specs
        self.tools_stale = False
        self._note_success()
        return specs

    def call(self, remote_name: str, arguments: dict) -> str:
        """Call a remote tool and flatten its content blocks to text."""
        if not self.start():
            return f"(the {self.name} MCP server is unavailable: {self.error})"
        try:
            result = self._request("tools/call",
                                   {"name": remote_name, "arguments": arguments or {}})
        except Exception as exc:
            # A dead server should be retried on the next call, not permanently
            # written off: `npx` servers get killed by all sorts of things.
            self.stop()
            raise ValueError(f"{self.name} MCP server: {exc}") from None
        parts: list[str] = []
        for block in (result.get("content") or []):
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                parts.append(str(block.get("text") or ""))
            elif kind == "resource":
                resource = block.get("resource") or {}
                parts.append(str(resource.get("text") or resource.get("uri") or ""))
            elif kind == "resource_link":
                parts.append(f"[resource: {block.get('name') or ''} {block.get('uri') or ''}]".strip())
            else:
                parts.append(f"[{kind} content omitted]")
        text = "\n".join(p for p in parts if p).strip()
        if not text and isinstance(result.get("structuredContent"), (dict, list)):
            text = json.dumps(result["structuredContent"], ensure_ascii=False)[:20000]
        if result.get("isError"):
            raise ValueError(text or "the MCP tool reported an error")
        return text or "(the tool returned no content)"

    def describe(self) -> dict:
        return {"name": self.name, "transport": self.transport, "alive": self.alive,
                "tools": [t.local_name for t in self.tools], "error": self.error,
                "protocol_version": self.protocol_version}


class MCPServer(_MCPClientBase):
    """One MCP server subprocess, spoken to over JSON-RPC on stdio."""

    transport = "stdio"

    def __init__(self, name: str, command: str, args: list[str] | None = None,
                 env: dict | None = None, cwd: str | None = None,
                 timeout: float = 30.0, base_env: dict | None = None):
        super().__init__(name, timeout)
        self.command = command
        self.args = list(args or [])
        self.env = dict(env or {})
        self.cwd = cwd
        # The sanitised sandbox environment (see the module docstring); None
        # falls back to a minimal allowlist, never to os.environ.
        self.base_env = base_env
        self.proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._next_id = 0

    # ------------------------------------------------------------ lifecycle --
    def start(self) -> bool:
        """Start the process and complete the handshake. False on any failure.

        Never raises. An MCP server that will not start is a missing capability,
        not a broken agent, so it degrades to a logged warning and no tools.
        """
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                return True
            if shutil.which(self.command) is None and not Path(self.command).exists():
                self.error = f"{self.command!r} is not on PATH"
                return False
            try:
                self.proc = subprocess.Popen(
                    [self.command, *self.args],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True, bufsize=1,
                    cwd=self.cwd or None,
                    env=mcp_child_env(self.base_env, self.env),
                )
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                self.proc = None
                return False
        try:
            self._initialize()
        except Exception as exc:
            self.error = f"handshake failed: {exc}"
            self.stop()
            return False
        self.error = ""
        return True

    def stop(self) -> None:
        with self._lock:
            proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass      # already gone

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # -------------------------------------------------------------- protocol --
    def _notify(self, method: str, params: dict) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _send(self, message: dict) -> None:
        proc = self.proc
        if proc is None or proc.stdin is None:
            raise RuntimeError("server is not running")
        proc.stdin.write(json.dumps(message) + "\n")
        proc.stdin.flush()

    def _request(self, method: str, params: dict) -> dict:
        """One request/response round trip.

        Reads until the reply with OUR id arrives, discarding log lines and
        handling notifications in between: a server is free to log progress and
        send unrelated notifications while we wait, and treating the first line
        back as the answer is how that turns into random failures.
        """
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            self._send({"jsonrpc": "2.0", "id": request_id,
                        "method": method, "params": params})
            proc = self.proc
            if proc is None or proc.stdout is None:
                raise RuntimeError("server is not running")
            deadline = time.time() + self.timeout
            while time.time() < deadline:
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError("server closed its output")
                try:
                    message = json.loads(line)
                except Exception:
                    continue          # a log line, not protocol
                if not isinstance(message, dict):
                    continue
                if message.get("id") != request_id:
                    self._handle_message(message)   # a notification or someone else's
                    continue
                if message.get("error"):
                    raise MCPProtocolError(_mcp_error_text(message["error"]))
                result = message.get("result")
                return result if isinstance(result, dict) else {}
            raise TimeoutError(f"no reply within {self.timeout:.0f}s")

    def describe(self) -> dict:
        return {**super().describe(), "command": self.command, "args": self.args}


class _MCPSessionExpired(RuntimeError):
    """The server no longer knows our Mcp-Session-Id (HTTP 404)."""


class MCPHttpServer(_MCPClientBase):
    """One MCP server over the Streamable HTTP transport.

    Every message is a POST of one JSON-RPC object to the endpoint, accepting
    either a JSON reply or a text/event-stream whose events carry the reply
    (and possibly notifications first). The server may issue an Mcp-Session-Id
    on initialize, which is echoed on every later request; a 404 for that id
    means the session expired, so the client initializes again once and
    retries. The optional GET stream for server-initiated messages is not
    opened: list_changed is picked up whenever it arrives inside a reply.
    """

    transport = "http"

    def __init__(self, name: str, url: str, headers: dict | None = None,
                 timeout: float = 30.0):
        super().__init__(name, timeout)
        self.url = url
        self.headers = {str(k): str(v) for k, v in (headers or {}).items()}
        self.session_id: str | None = None
        self._ready = False
        self._init_lock = threading.Lock()
        self._id_lock = threading.Lock()
        self._next_id = 0

    @property
    def command(self) -> str:      # status() compatibility
        return ""

    @property
    def args(self) -> list:
        return []

    def start(self) -> bool:
        with self._init_lock:
            if self._ready:
                return True
            self.session_id = None
            self.protocol_version = ""
            try:
                self._initialize()
            except Exception as exc:
                self.error = f"handshake failed: {exc}"
                self.session_id = None
                return False
            self._ready = True
            self.error = ""
            return True

    def stop(self) -> None:
        with self._init_lock:
            session, self.session_id, self._ready = self.session_id, None, False
        if not session:
            return
        # Polite session teardown (the spec's DELETE); a server that does not
        # support it answers 405, and either way we are done with it.
        try:
            import httpx
            with httpx.Client(timeout=min(5.0, self.timeout)) as client:
                client.delete(self.url, headers={**self._headers(), "Mcp-Session-Id": session})
        except Exception:
            pass

    @property
    def alive(self) -> bool:
        return self._ready

    def _headers(self) -> dict[str, str]:
        out = {k: _mcp_expand(v) for k, v in self.headers.items()}
        out["Content-Type"] = "application/json"
        out["Accept"] = "application/json, text/event-stream"
        if self.protocol_version:
            out["MCP-Protocol-Version"] = self.protocol_version
        if self.session_id:
            out["Mcp-Session-Id"] = self.session_id
        return out

    def _post(self, message: dict, request_id: Any = None) -> dict | None:
        import httpx
        with httpx.Client(timeout=self.timeout, follow_redirects=False) as client:
            with client.stream("POST", self.url, json=message,
                               headers=self._headers()) as resp:
                if resp.status_code == 404 and self.session_id \
                        and message.get("method") != "initialize":
                    raise _MCPSessionExpired("session expired")
                if resp.status_code >= 400:
                    body = resp.read().decode("utf-8", "replace")[:200]
                    if message.get("method") == "initialize" and resp.status_code == 400:
                        raise MCPProtocolError(f"HTTP 400: {body}")
                    raise RuntimeError(f"HTTP {resp.status_code}: {body}")
                session = resp.headers.get("mcp-session-id")
                if session and message.get("method") == "initialize":
                    self.session_id = session
                if request_id is None:
                    return None          # a notification: 202 Accepted, no body
                kind = (resp.headers.get("content-type") or "").lower()
                if "text/event-stream" in kind:
                    return self._read_sse(resp, request_id)
                raw = resp.read()
        try:
            payload = json.loads(raw.decode("utf-8", "replace") or "null")
        except Exception as exc:
            raise RuntimeError(f"unparseable reply: {exc}") from None
        messages = payload if isinstance(payload, list) else [payload]
        for item in messages:
            reply = self._match(item, request_id)
            if reply is not None:
                return reply
        raise RuntimeError("the reply did not answer the request")

    def _match(self, item: Any, request_id: Any) -> dict | None:
        if not isinstance(item, dict):
            return None
        if item.get("id") != request_id or "method" in item:
            self._handle_message(item)
            return None
        if item.get("error"):
            raise MCPProtocolError(_mcp_error_text(item["error"]))
        result = item.get("result")
        return result if isinstance(result, dict) else {}

    def _read_sse(self, resp: Any, request_id: Any) -> dict:
        """Events until the one that answers our id; notifications on the way."""
        data: list[str] = []
        deadline = time.time() + self.timeout
        for line in resp.iter_lines():
            if time.time() > deadline:
                break
            if line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
                continue
            if line.strip() or not data:
                continue                       # id:, event:, retry:, comments
            text, data = "\n".join(data), []
            try:
                item = json.loads(text)
            except Exception:
                continue
            for entry in (item if isinstance(item, list) else [item]):
                reply = self._match(entry, request_id)
                if reply is not None:
                    return reply
        if data:                               # a final event without a blank line
            try:
                item = json.loads("\n".join(data))
            except Exception:
                item = None
            for entry in (item if isinstance(item, list) else [item]):
                reply = self._match(entry, request_id)
                if reply is not None:
                    return reply
        raise RuntimeError("the event stream ended without a reply")

    def _notify(self, method: str, params: dict) -> None:
        self._post({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(self, method: str, params: dict) -> dict:
        with self._id_lock:
            self._next_id += 1
            request_id = self._next_id
        message = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        try:
            return self._post(message, request_id) or {}
        except _MCPSessionExpired:
            # Start a new session once and retry; a second 404 is an error.
            with self._init_lock:
                self._ready = False
                self.session_id = None
            if not self.start():
                raise RuntimeError(self.error or "could not re-initialize") from None
            return self._post(message, request_id) or {}

    def describe(self) -> dict:
        return {**super().describe(), "command": "", "args": [], "url": self.url}


class MCPManager:
    """Process-wide registry of MCP servers.

    A single instance, because ToolRegistry is constructed per subagent and per
    task run: starting a fresh `npx` server for each would cost seconds a turn
    and leak processes. Tool lists are cached until the server says they
    changed (notifications/tools/list_changed); a listing that failed is retried
    with a backoff. Registries already built keep the tools they were built
    with; new ones (profile agents, subagents, task runs) see the new list.
    """

    def __init__(self) -> None:
        self.servers: dict[str, _MCPClientBase] = {}
        self._lock = threading.Lock()
        self._configured: str = ""
        self.base_env: dict | None = None

    def configure(self, spec: str, timeout: float = 30.0,
                  base_env: dict | None = None) -> None:
        """Adopt a server list, given as JSON. Idempotent for the same JSON."""
        with self._lock:
            if base_env is not None:
                self.base_env = dict(base_env)
                for server in self.servers.values():
                    if isinstance(server, MCPServer):
                        server.base_env = self.base_env
            if spec == self._configured:
                return
            self._configured = spec
            for server in self.servers.values():
                server.stop()
            self.servers = {}
            for entry in parse_mcp_spec(spec):
                name = entry["name"]
                if entry.get("type") == "http":
                    self.servers[name] = MCPHttpServer(
                        name=name, url=entry["url"], headers=entry.get("headers") or {},
                        timeout=timeout)
                else:
                    self.servers[name] = MCPServer(
                        name=name, command=entry["command"], args=entry.get("args") or [],
                        env=entry.get("env") or {}, cwd=entry.get("cwd"), timeout=timeout,
                        base_env=self.base_env)
            # Only the process-wide manager speaks for the capability catalogue;
            # a private manager (a test, a script) must not rewrite it.
            if self is globals().get("MCP"):
                mcp_sync_capabilities(list(self.servers))

    def tools(self) -> list[MCPToolSpec]:
        """Every remote tool; each server is listed once until it changes."""
        out: list[MCPToolSpec] = []
        for server in list(self.servers.values()):
            if server.needs_listing():
                server.list_tools()
            out.extend(server.tools)
        return out

    def call(self, server: str, remote_name: str, arguments: dict) -> str:
        target = self.servers.get(server)
        if target is None:
            raise ValueError(f"no MCP server named {server!r}")
        return target.call(remote_name, arguments)

    def status(self) -> list[dict]:
        return [s.describe() for s in self.servers.values()]

    def stop_all(self) -> None:
        for server in list(self.servers.values()):
            server.stop()


def mcp_sync_capabilities(server_names: list[str]) -> None:
    """Publish one "mcp:<server>" capability per configured server.

    Profile and role validation accept only keys in CAPABILITY_ORDER, so a
    per-server grant needs its key there; the keys of servers no longer
    configured are withdrawn. The generic "mcp" key (every server) is static.
    """
    for key in [k for k in CAPABILITY_ORDER if k.startswith("mcp:")]:
        CAPABILITY_ORDER.remove(key)
        CAPABILITY_GROUPS.pop(key, None)
    anchor = CAPABILITY_ORDER.index("mcp") + 1 if "mcp" in CAPABILITY_ORDER \
        else len(CAPABILITY_ORDER)
    for offset, name in enumerate(server_names):
        key = f"mcp:{name}"
        CAPABILITY_GROUPS[key] = {
            "label": f"MCP: {name}",
            "description": f"The tools of the {name} MCP server.",
            "tools": [],
        }
        CAPABILITY_ORDER.insert(anchor + offset, key)


def mcp_tools_for_capabilities(capabilities) -> list[str]:
    """Concrete MCP tool names unlocked by "mcp" / "mcp:<server>" grants.

    Registered with config.register_capability_resolver, so
    tools_for_capabilities() (profiles, subagents, the MCP server endpoint)
    turns those grants into the dynamic mcp_<server>_<tool> names.
    """
    caps = set(capabilities or [])
    wants_all = "mcp" in caps
    servers = {c[4:] for c in caps if isinstance(c, str) and c.startswith("mcp:")}
    if not (wants_all or servers) or not MCP.servers:
        return []
    try:
        specs = MCP.tools()
    except Exception as exc:  # noqa: BLE001 - a missing capability, not an error
        log(f"MCP listing failed while expanding capabilities: {exc}", logging.WARNING)
        return []
    return [s.local_name for s in specs if wants_all or s.server in servers]


def parse_mcp_spec(spec: str) -> list[dict]:
    """Validate a server list. Bad entries are dropped with a warning.

    Accepts either the array form or the `{"mcpServers": {...}}` object form that
    other MCP clients use, so a config file can be copied across unchanged.
    stdio entries normalise to exactly {"name","command","args","env","cwd"};
    HTTP entries to {"name","type":"http","url","headers"}.
    """
    text = (spec or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except Exception as exc:
        log(f"MCP_SERVERS is not valid JSON ({exc}); no MCP servers configured.",
            logging.WARNING)
        return []
    entries: list[dict] = []
    if isinstance(data, dict) and isinstance(data.get("mcpServers"), dict):
        for name, body in data["mcpServers"].items():
            if isinstance(body, dict):
                entries.append({**body, "name": name})
    elif isinstance(data, list):
        entries = [e for e in data if isinstance(e, dict)]
    elif isinstance(data, dict):
        entries = [{**body, "name": name} for name, body in data.items()
                   if isinstance(body, dict)]
    out: list[dict] = []
    seen: set[str] = set()
    for entry in entries:
        if entry.get("enabled") is False:
            continue
        name = str(entry.get("name") or "").strip()
        command = str(entry.get("command") or "").strip()
        url = str(entry.get("url") or "").strip()
        kind = str(entry.get("type") or entry.get("transport") or "").strip().lower()
        if not name or not (command or url):
            log(f"Ignoring an MCP server entry with no name or command: "
                f"{str(entry)[:120]}", logging.WARNING)
            continue
        if name in seen:
            log(f"Ignoring a duplicate MCP server named {name!r}.", logging.WARNING)
            continue
        if url and not command:
            if kind == "sse":
                log(f"Ignoring MCP server {name!r}: the old HTTP+SSE transport is not "
                    "supported; use its Streamable HTTP endpoint.", logging.WARNING)
                continue
            if not url.lower().startswith(("http://", "https://")):
                log(f"Ignoring MCP server {name!r}: url must be http(s).", logging.WARNING)
                continue
            raw_headers = entry.get("headers")
            headers = ({str(k): str(v) for k, v in raw_headers.items()}
                       if isinstance(raw_headers, dict) else {})
            bearer = entry.get("bearer") or entry.get("token")
            if bearer and not any(k.lower() == "authorization" for k in headers):
                headers["Authorization"] = f"Bearer {bearer}"
            seen.add(name)
            out.append({"name": name, "type": "http", "url": url, "headers": headers})
            continue
        seen.add(name)
        args = entry.get("args") or []
        out.append({
            "name": name,
            "command": command,
            "args": [str(a) for a in args] if isinstance(args, list) else [],
            "env": entry.get("env") if isinstance(entry.get("env"), dict) else {},
            "cwd": str(entry["cwd"]) if entry.get("cwd") else None,
        })
    return out


# The one manager for this process.
MCP = MCPManager()
register_capability_resolver(mcp_tools_for_capabilities)


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'MCP',
    'MCPManager',
    'MCPServer',
    'MCPHttpServer',
    'MCPProtocolError',
    'MCPToolSpec',
    'PROTOCOL_VERSION',
    'SUPPORTED_PROTOCOL_VERSIONS',
    'mcp_child_env',
    'mcp_sync_capabilities',
    'mcp_tools_for_capabilities',
    'parse_mcp_spec',
]
