"""Terminal client for the served model: `deploy.py chat|login|logout|whoami`.

The client talks to the same multi-user backend the browser does, over the
existing API, so a terminal user gets the same conversations, tools, approvals,
model aliases and agent profiles. It is stdlib only (http.client, json, and
readline when present) and imports nothing from this package, so the file also
runs on its own from a plain python3 on another Mac:

    python3 client.py login --url http://studio.local:8000
    python3 client.py chat

Pieces:

* Credentials: --token, else $LLM_API_TOKEN, else
  ~/.config/local-llm/credentials.json (directory 0700, file 0600, one entry per
  server URL, plus "default": the last server signed in to). The menu-bar
  companion reads the same file.
* Sign-in: username/password through POST /api/auth/cli/login (the password is
  read with getpass and never echoed, logged or stored), or the Entra device
  code flow through /api/auth/device/start + /poll. Both return an API token.
* Chat: POST /api/chat/stream with `X-Stream-Ids: 1`; frames are parsed as SSE
  (id/data/comments, LF/CRLF/CR). Tokens render live, reasoning dimmed, tools as
  one-line summaries, approvals and cloud consent as a prompt whose answer is
  POSTed back to the turn. Ctrl-C cancels the turn (POST .../cancel); a second
  Ctrl-C exits. A dropped connection resumes from the last seen id through
  GET /api/chat/turns/{id}/events?after=N.

Exit codes: 0 ok, 1 error, 2 usage, 3 not signed in / refused, 4 server
unreachable, 130 interrupted or cancelled.
"""

from __future__ import annotations

import argparse
import getpass
import http.client
import json
import os
import socket
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

CLIENT_DEFAULT_URL = "http://127.0.0.1:8000"
CLIENT_EXIT_OK = 0
CLIENT_EXIT_ERROR = 1
CLIENT_EXIT_USAGE = 2
CLIENT_EXIT_AUTH = 3
CLIENT_EXIT_UNREACHABLE = 4
CLIENT_EXIT_INTERRUPTED = 130

_CL_USER_AGENT = "local-llm-cli/1"
# Server keepalives arrive every 15s; a read that waits much longer than that
# means the connection is gone even if no RST ever arrived.
_CL_STREAM_TIMEOUT_S = 75.0
_CL_REQUEST_TIMEOUT_S = 30.0
_CL_MAX_RESUMES = 8


class ClientError(Exception):
    """A failure to report to the user, with the exit code it maps to."""

    def __init__(self, message: str, status: int = 0, code: int = CLIENT_EXIT_ERROR):
        super().__init__(message)
        self.status = status
        self.code = code


# --------------------------------------------------------------------------- #
# URLs and credentials                                                        #
# --------------------------------------------------------------------------- #
def _cl_norm_url(url: Any) -> str:
    raw = str(url or "").strip().rstrip("/") or CLIENT_DEFAULT_URL
    if "://" not in raw:
        raw = "http://" + raw
    parts = urllib.parse.urlsplit(raw)
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise ClientError(f"not an http(s) URL: {raw}", code=CLIENT_EXIT_USAGE)
    return urllib.parse.urlunsplit((parts.scheme.lower(), parts.netloc.lower(),
                                    parts.path.rstrip("/"), "", ""))


def client_credentials_path(env: Any = None) -> Path:
    """~/.config/local-llm/credentials.json, or $LLM_CREDENTIALS_FILE."""
    env = os.environ if env is None else env
    explicit = str(env.get("LLM_CREDENTIALS_FILE") or "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / ".config" / "local-llm" / "credentials.json"


def load_credentials(path: Path | None = None) -> dict:
    """{"version", "default", "servers": {url: entry}}; empty when absent or unreadable."""
    target = Path(path or client_credentials_path())
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = None
    if not isinstance(data, dict):
        data = {}
    servers = data.get("servers")
    data["servers"] = servers if isinstance(servers, dict) else {}
    data.setdefault("version", 1)
    return data


def _cl_write_private(path: Path, data: dict) -> None:
    """Write JSON readable by the owner only (dir 0700, file 0600), atomically."""
    parent = path.parent
    created = not parent.exists()
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if created or parent.name == "local-llm":
        os.chmod(parent, 0o700)
    tmp = parent / f".{path.name}.{os.getpid()}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.chmod(path, 0o600)


def save_credential(url: str, entry: dict, path: Path | None = None) -> Path:
    """Store the sign-in for `url` and make it the default server."""
    target = Path(path or client_credentials_path())
    base = _cl_norm_url(url)
    data = load_credentials(target)
    data["servers"][base] = dict(entry, url=base)
    data["default"] = base
    _cl_write_private(target, data)
    return target


def remove_credential(url: str, path: Path | None = None) -> dict | None:
    """Forget the sign-in for `url`; returns the removed entry, if any."""
    target = Path(path or client_credentials_path())
    base = _cl_norm_url(url)
    data = load_credentials(target)
    entry = data["servers"].pop(base, None)
    if entry is None:
        return None
    if data.get("default") == base:
        data["default"] = sorted(data["servers"])[0] if data["servers"] else None
        if data["default"] is None:
            data.pop("default")
    _cl_write_private(target, data)
    return entry


def client_resolve(url: Any = None, token: Any = None, env: Any = None,
                   path: Path | None = None) -> tuple[str, str | None, dict | None]:
    """(base URL, token or None, stored entry or None).

    URL: --url, $LLM_URL, the last server signed in to, http://127.0.0.1:8000.
    Token: --token, $LLM_API_TOKEN, the credentials entry for that URL.
    """
    env = os.environ if env is None else env
    data = load_credentials(path)
    chosen = url or env.get("LLM_URL") or data.get("default") or CLIENT_DEFAULT_URL
    base = _cl_norm_url(chosen)
    entry = data["servers"].get(base)
    entry = entry if isinstance(entry, dict) else None
    if token:
        return base, str(token).strip(), entry
    if str(env.get("LLM_API_TOKEN") or "").strip():
        return base, str(env["LLM_API_TOKEN"]).strip(), entry
    return base, (entry or {}).get("token") or None, entry


# --------------------------------------------------------------------------- #
# HTTP and SSE                                                                #
# --------------------------------------------------------------------------- #
def _cl_json(data: bytes) -> dict:
    text = data.decode("utf-8", "replace") if isinstance(data, (bytes, bytearray)) else str(data)
    try:
        value = json.loads(text) if text.strip() else {}
    except ValueError:
        return {"error": text.strip()[:300]}
    return value if isinstance(value, dict) else {"data": value}


def _cl_error_text(status: int, body: dict) -> str:
    if body.get("error"):
        return str(body["error"])
    detail = body.get("detail")
    if isinstance(detail, list) and detail:
        first = detail[0] if isinstance(detail[0], dict) else {}
        where = ".".join(str(p) for p in (first.get("loc") or [])[1:])
        return f"{where}: {first.get('msg')}" if where else str(first.get("msg") or detail)
    if detail:
        return str(detail)
    return f"HTTP {status}"


def _cl_http_error(status: int, body: dict, url: str = "") -> ClientError:
    text = _cl_error_text(status, body)
    if status == 0:
        return ClientError(text, status, CLIENT_EXIT_UNREACHABLE)
    if status == 401:
        hint = f" (run: deploy.py login --url {url})" if url else ""
        return ClientError(f"not signed in: {text}{hint}", status, CLIENT_EXIT_AUTH)
    if status == 403:
        return ClientError(f"refused: {text}", status, CLIENT_EXIT_AUTH)
    return ClientError(text, status, CLIENT_EXIT_ERROR)


class SSEParser:
    """Incremental text/event-stream parser.

    feed() takes raw text in any chunking with LF, CRLF or lone CR line ends and
    returns the events it completed as (id or None, data). Comment lines (": x")
    are skipped, multi-line data is joined with "\\n", and fields other than
    id/data (event, retry) are ignored.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._data: list[str] = []
        self._id: str | None = None
        self.last_id: str | None = None

    def feed(self, text: str) -> list[tuple[str | None, str]]:
        self._buf += text
        out: list[tuple[str | None, str]] = []
        while True:
            lf, cr = self._buf.find("\n"), self._buf.find("\r")
            if lf < 0 and cr < 0:
                return out
            if cr >= 0 and (lf < 0 or cr < lf):
                # A CR at the very end may be the first half of a CRLF.
                if cr + 1 == len(self._buf):
                    return out
                cut, skip = cr, 2 if self._buf[cr + 1] == "\n" else 1
            else:
                cut, skip = lf, 1
            line, self._buf = self._buf[:cut], self._buf[cut + skip:]
            event = self.line(line)
            if event is not None:
                out.append(event)

    def line(self, line: str) -> tuple[str | None, str] | None:
        """One line without its terminator; returns an event on a blank line."""
        if line == "":
            if not self._data:
                self._id = None
                return None
            event = (self._id, "\n".join(self._data))
            self._data, self._id = [], None
            return event
        if line.startswith(":"):
            return None
        name, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if name == "data":
            self._data.append(value)
        elif name == "id" and "\0" not in value:
            self._id = value
            self.last_id = value
        return None


class LLMClient:
    """Minimal HTTP client for the app's API (stdlib http.client)."""

    def __init__(self, url: str, token: str | None = None, *,
                 timeout: float = _CL_REQUEST_TIMEOUT_S,
                 stream_timeout: float = _CL_STREAM_TIMEOUT_S):
        self.url = _cl_norm_url(url)
        parts = urllib.parse.urlsplit(self.url)
        self._https = parts.scheme == "https"
        self._host = parts.hostname or "127.0.0.1"
        self._port = parts.port
        self._prefix = parts.path
        self.token = token or None
        self.timeout = timeout
        self.stream_timeout = stream_timeout

    def _conn(self, timeout: float) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self._https else http.client.HTTPConnection
        return cls(self._host, self._port, timeout=timeout)

    def _headers(self, extra: dict | None = None) -> dict:
        headers = {"Accept": "application/json", "User-Agent": _CL_USER_AGENT}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        headers.update(extra or {})
        return headers

    def request(self, method: str, path: str, body: Any = None, *, raw: bytes | None = None,
                headers: dict | None = None, timeout: float | None = None,
                auth: bool = True) -> tuple[int, dict]:
        """(status, JSON body). Status 0 means the server could not be reached."""
        hdrs = self._headers(headers)
        if not auth:
            hdrs.pop("Authorization", None)
        payload = raw
        if raw is None and body is not None:
            payload = json.dumps(body).encode("utf-8")
            hdrs["Content-Type"] = "application/json"
        conn = self._conn(timeout or self.timeout)
        try:
            conn.request(method, self._prefix + path, body=payload, headers=hdrs)
            resp = conn.getresponse()
            data = resp.read()
            status = resp.status
        except (OSError, http.client.HTTPException) as exc:
            return 0, {"error": f"server unreachable at {self.url}: {_cl_exc(exc)}"}
        finally:
            conn.close()
        return status, _cl_json(data)

    def open_stream(self, method: str, path: str, body: Any = None) -> tuple[Any, Any]:
        """(connection, response) for an SSE request with resumable ids."""
        hdrs = self._headers({"Accept": "text/event-stream", "X-Stream-Ids": "1"})
        payload = None
        if body is not None:
            payload = json.dumps(body).encode("utf-8")
            hdrs["Content-Type"] = "application/json"
        conn = self._conn(self.stream_timeout)
        try:
            conn.request(method, self._prefix + path, body=payload, headers=hdrs)
            return conn, conn.getresponse()
        except BaseException:
            conn.close()
            raise

    @staticmethod
    def iter_events(resp: Any) -> Iterator[tuple[str | None, str]]:
        """(id, data) per SSE event until EOF."""
        parser = SSEParser()
        while True:
            chunk = resp.readline()
            if not chunk:
                return
            yield from parser.feed(chunk.decode("utf-8", "replace"))


def _cl_exc(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _cl_close(conn: Any) -> None:
    try:
        if conn is not None:
            conn.close()
    except Exception:  # noqa: BLE001 - closing a dead socket must not raise
        pass


# --------------------------------------------------------------------------- #
# Terminal output                                                             #
# --------------------------------------------------------------------------- #
def _cl_isatty(stream: Any) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:  # noqa: BLE001
        return False


def _cl_color_wanted(out: Any, env: Any = None, disabled: bool = False) -> bool:
    env = os.environ if env is None else env
    return not disabled and "NO_COLOR" not in env and _cl_isatty(out)


def _cl_short(value: Any, limit: int = 60) -> str:
    if isinstance(value, dict):
        parts = []
        for key, val in value.items():
            text = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
            parts.append(f"{key}={text}")
        text = ", ".join(parts)
    elif isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _cl_terminal_ask(stdin: Any, err: Any,
                     use_stdin: bool = False) -> Callable[[str], str | None]:
    """Ask on the terminal: stdin when it is one (or use_stdin: a scripted REPL),
    else /dev/tty (stdin is the piped message), else nobody."""
    def ask(prompt: str) -> str | None:
        err.write(prompt)
        err.flush()
        if use_stdin or _cl_isatty(stdin):
            line = stdin.readline()
        else:
            try:
                with open("/dev/tty", encoding="utf-8", errors="replace") as tty:
                    line = tty.readline()
            except OSError:
                err.write("(no terminal to answer on)\n")
                return None
        return line.rstrip("\r\n") if line else None
    return ask


class _ClOutputClosed(Exception):
    """stdout or stderr went away (a closed pipe such as `| head`).

    Not an OSError on purpose: the stream loop reads OSError as a dropped
    server connection and would try to resume.
    """


def _cl_emit(stream: Any, text: str) -> None:
    try:
        stream.write(text)
        stream.flush()
    except BrokenPipeError:
        raise _ClOutputClosed() from None


class _ClConsole:
    """Rendering for one session: answer on stdout, everything else on stderr."""

    _CODES = {"dim": "2", "bold": "1", "red": "31", "green": "32", "yellow": "33",
              "cyan": "36"}

    def __init__(self, out: Any, err: Any, *, color: bool = False, live: bool = True,
                 think: bool = False, json_mode: bool = False, quiet: bool = False,
                 ask: Callable[[str], str | None] | None = None):
        self.out, self.err = out, err
        self.color, self.live, self.think = color, live, think
        self.json_mode, self.quiet = json_mode, quiet
        self.ask = ask or (lambda prompt: None)
        self._open: Any = None  # the stream holding an unterminated line

    def style(self, text: str, *names: str) -> str:
        if not self.color or not names:
            return text
        codes = ";".join(self._CODES[n] for n in names)
        return f"\033[{codes}m{text}\033[0m"

    def _write(self, stream: Any, text: str) -> None:
        if not text:
            return
        if self._open is not None and self._open is not stream:
            self.close_line()
        _cl_emit(stream, text)
        self._open = None if text.endswith("\n") else stream

    def close_line(self) -> None:
        if self._open is not None:
            stream, self._open = self._open, None
            _cl_emit(stream, "\n")

    def answer(self, text: str) -> None:
        self._write(self.out, text)

    def raw_event(self, event: dict) -> None:
        _cl_emit(self.out, json.dumps(event, ensure_ascii=False) + "\n")

    def reasoning(self, text: str) -> None:
        if self.think and not self.json_mode:
            self._write(self.err, self.style(text, "dim"))

    def info(self, text: str) -> None:
        if not self.quiet and not self.json_mode:
            self._write(self.err, self.style(text, "dim") + "\n")

    def status(self, text: str, *names: str) -> None:
        if not self.quiet and not self.json_mode:
            self._write(self.err, self.style(text, *names) + "\n")

    def alert(self, text: str) -> None:
        """Something the user must see even in --json and --quiet mode."""
        self._write(self.err, self.style(text, "yellow", "bold") + "\n")

    def warn(self, text: str) -> None:
        if not self.json_mode:
            self._write(self.err, self.style(text, "yellow") + "\n")

    def error(self, text: str) -> None:
        self._write(self.err, self.style(f"error: {text}", "red") + "\n")

    def print(self, text: str = "") -> None:
        """Command output (lists, help) on stdout."""
        self._write(self.out, text + "\n")


# --------------------------------------------------------------------------- #
# Chat session                                                                #
# --------------------------------------------------------------------------- #
_CL_APPROVAL_KEYS = {"a": ("allow", "once"), "c": ("allow", "conversation"),
                     "t": ("allow", "tool"), "d": ("deny", "once")}
_CL_CONSENT_KEYS = {"a": ("allow", "once"), "c": ("allow", "conversation"),
                    "t": ("allow", "always"), "d": ("deny", "once")}


class _ClTurn:
    """What one turn produced, as the stream told it."""

    def __init__(self) -> None:
        self.turn_id: str | None = None
        self.last_seq: int | None = None
        self.final: dict | None = None
        self.cancelled: dict | None = None
        self.error: str | None = None
        self.usage: list[dict] = []
        self.shown = ""  # answer text of the current step that reached the terminal
        self.queued_at: int | None = None
        self.interrupts = 0
        self.resumes = 0

    @property
    def ended(self) -> bool:
        return self.final is not None or self.cancelled is not None or self.error is not None

    def result(self, conversation_id: str | None) -> dict:
        answer = ""
        if self.final is not None:
            answer = str(self.final.get("answer") or "")
        elif self.cancelled is not None:
            answer = str(self.cancelled.get("partial") or "")
        return {"answer": answer, "final": self.final, "cancelled": self.cancelled,
                "error": self.error, "turn_id": self.turn_id,
                "conversation_id": conversation_id, "usage": list(self.usage),
                "resumes": self.resumes}


class ChatSession:
    """One terminal conversation: per-turn options plus the streaming loop."""

    def __init__(self, client: LLMClient, console: _ClConsole, *,
                 conversation_id: str | None = None, model: str | None = None,
                 agent_id: str | None = None, project_id: str | None = None,
                 mode: str = "auto", max_tokens: int | None = None,
                 temperature: float | None = None, sleep: Callable[[float], Any] = time.sleep,
                 max_resumes: int = _CL_MAX_RESUMES):
        self.client, self.console = client, console
        self.conversation_id = conversation_id
        self.model, self.agent_id, self.project_id = model, agent_id, project_id
        self.mode = mode
        self.max_tokens, self.temperature = max_tokens, temperature
        self.pending_attachments: list[dict] = []
        self.sleep = sleep
        self.max_resumes = max_resumes

    # ---- requests ---------------------------------------------------------- #
    def payload(self, message: str) -> dict:
        body: dict = {"message": message}
        if self.conversation_id:
            body["conversation_id"] = self.conversation_id
        if self.mode == "chat":
            body["agent"] = False
        elif self.mode == "agent":
            body["agent"] = True
        for key, value in (("model", self.model), ("agent_id", self.agent_id),
                           ("project_id", self.project_id),
                           ("max_tokens", self.max_tokens),
                           ("temperature", self.temperature)):
            if value is not None:
                body[key] = value
        if self.pending_attachments:
            body["attachments"] = [a["id"] for a in self.pending_attachments]
        return body

    def call(self, method: str, path: str, body: Any = None, **kw: Any) -> dict:
        status, data = self.client.request(method, path, body, **kw)
        if status != 200:
            raise _cl_http_error(status, data, self.client.url)
        return data

    def attach(self, path: str) -> dict:
        file = Path(path).expanduser()
        try:
            data = file.read_bytes()
        except OSError as exc:
            raise ClientError(f"cannot read {file}: {exc.strerror or exc}") from None
        headers = {"Content-Type": "application/octet-stream",
                   "X-Filename": urllib.parse.quote(file.name)}
        if self.conversation_id:
            headers["X-Conversation-Id"] = self.conversation_id
        row = self.call("POST", "/api/attachments", raw=data, headers=headers, timeout=120)
        if not row.get("id"):
            raise ClientError("the server did not return an attachment id")
        self.pending_attachments.append(row)
        return row

    def cancel_active(self) -> int:
        """Cancel running turns of this conversation (e.g. one left by a lost link)."""
        if not self.conversation_id:
            return 0
        data = self.call("GET", "/api/chat/active?conversation_id="
                         + urllib.parse.quote(self.conversation_id))
        count = 0
        for turn in data.get("turns") or []:
            tid = turn.get("turn_id") or turn.get("id")
            if tid and self.client.request("POST", f"/api/chat/turns/{tid}/cancel")[0] == 200:
                count += 1
        return count

    # ---- one turn ---------------------------------------------------------- #
    def send(self, message: str) -> dict:
        """Run one turn to its end. KeyboardInterrupt escapes only on a second Ctrl-C."""
        turn = _ClTurn()
        try:
            conn, resp = self.client.open_stream("POST", "/api/chat/stream", self.payload(message))
        except (OSError, http.client.HTTPException) as exc:
            raise ClientError(f"server unreachable at {self.client.url}: {_cl_exc(exc)}",
                              code=CLIENT_EXIT_UNREACHABLE) from None
        if resp.status != 200:
            try:
                body = _cl_json(resp.read())
            except (OSError, http.client.HTTPException):
                body = {}
            _cl_close(conn)
            raise _cl_http_error(resp.status, body, self.client.url)
        # Accepted: the attachments now belong to this message.
        self.pending_attachments = []
        self._pump(turn, conn, resp)
        self.console.close_line()
        return turn.result(self.conversation_id)

    def _pump(self, turn: _ClTurn, conn: Any, resp: Any) -> None:
        """Read the turn to its end, resuming after a drop and cancelling on Ctrl-C."""
        failures = 0
        while True:
            before = turn.last_seq
            try:
                if resp is None:
                    conn, resp = self._resume(turn)
                    if resp is None:
                        return
                done = self._consume(turn, resp)
                _cl_close(conn)
                conn = resp = None
                if done or turn.ended:
                    return
                raise ConnectionError("the stream ended early")
            except KeyboardInterrupt:
                _cl_close(conn)
                conn = resp = None
                self._interrupted(turn)
            except (OSError, http.client.HTTPException) as exc:
                _cl_close(conn)
                conn = resp = None
                if turn.ended:
                    return
                failures = 1 if turn.last_seq != before else failures + 1
                self._check_resumable(turn, failures, exc)
                self.console.warn(f"connection lost ({_cl_exc(exc)}); resuming...")
                try:
                    self.sleep(min(0.25 * (2 ** (failures - 1)), 5.0))
                except KeyboardInterrupt:
                    self._interrupted(turn)

    def _check_resumable(self, turn: _ClTurn, failures: int, exc: BaseException) -> None:
        if not turn.turn_id:
            raise ClientError(f"lost the connection before the turn started ({_cl_exc(exc)})")
        if failures > self.max_resumes:
            raise ClientError(f"lost the connection to turn {turn.turn_id} "
                              f"({_cl_exc(exc)}); it may still be running: /stop cancels it")

    def _interrupted(self, turn: _ClTurn) -> None:
        """First Ctrl-C: cancel the turn and keep reading its end. Second: exit."""
        turn.interrupts += 1
        if turn.interrupts > 1:
            self.console.close_line()
            raise KeyboardInterrupt
        self.console.alert("cancelling... (Ctrl-C again to quit)")
        if turn.turn_id:
            status, body = self.client.request("POST", f"/api/chat/turns/{turn.turn_id}/cancel")
            if status != 200:
                self.console.warn(f"cancel failed: {_cl_error_text(status, body)}")
            return
        # Still queued, so the id is not known yet: cancel what runs in this
        # conversation, else the server's detach grace cancels the turn.
        try:
            self.cancel_active()
        except ClientError:
            pass
        raise ClientError("cancelled before the turn started", code=CLIENT_EXIT_INTERRUPTED)

    def _resume(self, turn: _ClTurn) -> tuple[Any, Any]:
        if not turn.turn_id:
            raise ClientError("lost the connection before the turn started")
        turn.resumes += 1
        after = turn.last_seq or 0
        conn, resp = self.client.open_stream(
            "GET", f"/api/chat/turns/{turn.turn_id}/events?after={after}")
        if resp.status == 404:
            _cl_close(conn)
            if turn.interrupts:
                turn.cancelled = turn.cancelled or {"type": "cancelled", "partial": ""}
                return None, None
            raise ClientError(f"turn {turn.turn_id} is no longer available on the server")
        if resp.status != 200:
            body = _cl_json(resp.read())
            _cl_close(conn)
            raise _cl_http_error(resp.status, body, self.client.url)
        return conn, resp

    def _consume(self, turn: _ClTurn, resp: Any) -> bool:
        """Handle events until [DONE] (True) or the stream ends (False)."""
        for seq, data in self.client.iter_events(resp):
            if data.strip() == "[DONE]":
                return True
            try:
                event = json.loads(data)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if seq is not None:
                try:
                    number = int(seq)
                except ValueError:
                    number = None
                if number is not None:
                    if turn.last_seq is not None and number <= turn.last_seq:
                        continue  # already handled before a reconnect
                    turn.last_seq = number
            self._handle(turn, event)
        return False

    # ---- events ------------------------------------------------------------ #
    def _handle(self, turn: _ClTurn, event: dict) -> None:
        con = self.console
        kind = str(event.get("type") or "")
        if con.json_mode:
            con.raw_event(event)
        if event.get("turn_id") and not turn.turn_id:
            turn.turn_id = str(event["turn_id"])
        if kind == "start":
            self.conversation_id = event.get("conversation_id") or self.conversation_id
        elif kind == "queued":
            position = event.get("position")
            if position != turn.queued_at:
                turn.queued_at = position
                con.status(f"[queued] position {position} "
                           f"({event.get('running', 0)}/{event.get('limit', 0)} running)",
                           "yellow")
        elif kind == "token":
            text = str(event.get("token") or event.get("text") or "")
            if con.live and not con.json_mode and text:
                con.answer(text)
                turn.shown += text
        elif kind in ("think_token", "reason_token"):
            con.reasoning(str(event.get("token") or event.get("text") or ""))
        elif kind == "reason_step" and con.think:
            con.info(f"[reasoning {event.get('step')}/{event.get('total')}] "
                     f"{event.get('label') or ''}".rstrip())
        elif kind == "step":
            self._new_step(turn)
            if int(event.get("step") or 1) > 1:
                limit = event.get("max_steps")
                con.info(f"[step {event.get('step')}{'/' + str(limit) if limit else ''}]")
        elif kind == "tool_call":
            self._new_step(turn)
            con.status(f"[tool] {event.get('name') or 'tool'}({_cl_short(event.get('args') or {})})",
                       "cyan")
        elif kind == "tool_result":
            self._new_step(turn)
            name = event.get("name") or "tool"
            if event.get("error"):
                con.status(f"[tool] {name} failed: {_cl_short(event.get('error'), 100)}", "red")
            else:
                size = len(str(event.get("result") or ""))
                con.status(f"[tool] {name} ok ({size} chars)", "green")
        elif kind in ("approval_required", "consent_required"):
            self._decide(turn, event, consent=(kind == "consent_required"))
        elif kind in ("approval_resolved", "consent_resolved"):
            what = "approval" if kind == "approval_resolved" else "consent"
            con.info(f"[{what}] {event.get('decision')} ({event.get('scope') or 'once'})")
        elif kind == "usage":
            turn.usage.append(event)
        elif kind == "notice":
            text = str(event.get("message") or "")
            if text:
                (con.info if event.get("info") else con.warn)(f"[notice] {text}")
        elif kind == "context":
            kept, total = event.get("kept"), event.get("total")
            if isinstance(kept, int) and isinstance(total, int):
                con.info(f"[context] window full: kept {kept} of {total} earlier messages")
        elif kind == "context_injected":
            files = event.get("files") or event.get("paths") or []
            names = ", ".join(str(f.get("path") if isinstance(f, dict) else f) for f in files)
            con.info(f"[project] instructions read: {names or 'yes'}")
        elif kind == "final":
            turn.final = event
            self.conversation_id = event.get("conversation_id") or self.conversation_id
            self._finish_answer(turn, str(event.get("answer") or ""))
        elif kind == "cancelled":
            turn.cancelled = event
            self._finish_answer(turn, str(event.get("partial") or ""))
            con.warn("[cancelled]")
        elif kind == "error":
            turn.error = str(event.get("error") or "the turn failed")

    def _new_step(self, turn: _ClTurn) -> None:
        # Text streamed before a tool call or a new step was a draft; the
        # final answer is compared against the current step's text only.
        if turn.shown:
            self.console.close_line()
        turn.shown = ""

    def _finish_answer(self, turn: _ClTurn, answer: str) -> None:
        con = self.console
        if con.json_mode:
            return
        streamed = turn.shown if con.live else ""
        if streamed and " ".join(streamed.split()) == " ".join(answer.split()):
            pass
        elif streamed and answer.startswith(streamed):
            con.answer(answer[len(streamed):])
        elif answer:
            if streamed:
                con.close_line()
            con.answer(answer)
        con.close_line()

    def _decide(self, turn: _ClTurn, event: dict, *, consent: bool) -> None:
        con = self.console
        item_id = str(event.get("id") or "")
        turn_id = turn.turn_id or event.get("turn_id")
        keys = _CL_CONSENT_KEYS if consent else _CL_APPROVAL_KEYS
        offered = [str(s) for s in (event.get("scopes") or [])]
        if consent:
            target = "/".join(str(x) for x in (event.get("provider"), event.get("model")) if x)
            con.alert(f"[consent] send this turn to the cloud model {target or '(cloud)'}"
                      f" ({event.get('trigger') or 'fallback'}, ~{event.get('est_prompt_tokens') or '?'}"
                      f" tokens): {event.get('reason') or ''}".rstrip(": "))
            prompt = "  [a]llow once / allow for [c]onversation / allow [t]his always / [d]eny: "
        else:
            con.alert(f"[approval] {event.get('name') or 'tool'}({_cl_short(event.get('args') or {}, 120)})"
                      f" risk {event.get('risk') or '?'}: {event.get('reason') or ''}".rstrip(": "))
            prompt = "  [a]llow once / allow for [c]onversation / allow [t]ool / [d]eny: "
        if not item_id or not turn_id:
            con.warn("cannot answer: the prompt has no id")
            return
        decision = None
        for _ in range(5):
            con.close_line()
            answer = con.ask(prompt)
            if answer is None:
                con.alert("no answer possible here: denied")
                decision = ("deny", "once")
                break
            choice = keys.get(answer.strip().lower()[:1])
            if choice is None:
                continue
            if offered and choice[1] not in offered and choice[0] == "allow":
                con.warn(f"'{choice[1]}' is not offered for this one; pick another")
                continue
            decision = choice
            break
        decision = decision or ("deny", "once")
        route = "consent" if consent else "approvals"
        status, body = self.client.request(
            "POST", f"/api/chat/turns/{turn_id}/{route}/{urllib.parse.quote(item_id)}",
            {"decision": decision[0], "scope": decision[1]})
        if status != 200:
            con.warn(f"answer not accepted: {_cl_error_text(status, body)}")


def _cl_usage_footer(result: dict) -> str:
    usage = result.get("usage") or []
    parts = []
    if usage:
        completion = sum(int(u.get("completion_tokens") or 0) for u in usage)
        ttft = next((u.get("ttft_ms") for u in usage if u.get("ttft_ms")), None)
        tps = next((u.get("decode_tps") for u in reversed(usage) if u.get("decode_tps")), None)
        queue = next((u.get("queue_ms") for u in usage if u.get("queue_ms")), None)
        parts.append(f"{completion} tokens")
        if ttft is not None:
            parts.append(f"TTFT {float(ttft):.0f} ms")
        if tps is not None:
            parts.append(f"{float(tps):.1f} tok/s")
        if queue:
            parts.append(f"queued {float(queue) / 1000:.1f} s")
    if result.get("resumes"):
        parts.append(f"resumed {result['resumes']}x")
    return "[" + " | ".join(parts) + "]" if parts else ""


def _cl_result_code(result: dict) -> int:
    if result.get("error"):
        return CLIENT_EXIT_ERROR
    if result.get("cancelled") is not None:
        return CLIENT_EXIT_INTERRUPTED
    return CLIENT_EXIT_OK if result.get("final") is not None else CLIENT_EXIT_ERROR


# --------------------------------------------------------------------------- #
# REPL                                                                        #
# --------------------------------------------------------------------------- #
_CL_HELP = """commands:
  /new                 start a new conversation
  /history             list your recent conversations
  /open <id>           continue a conversation (shows its last messages)
  /model [alias|off]   answer with another served model (no argument: list)
  /agent [id|off]      run turns as a saved agent profile (no argument: list)
  /project [id|off]    confine the agent to one of your projects
  /mode auto|chat|agent  let the server route, plain chat, or always the agent
  /attach <path>       upload a file; it goes with your next message
  /think               show or hide the model's reasoning
  /stop                cancel a turn still running in this conversation
  /help                this list
  /quit                leave (Ctrl-D too)
A line ending in \\ continues on the next line; //text sends "/text" literally.
Ctrl-C during an answer cancels it; a second Ctrl-C quits."""


def _cl_list_conversations(session: ChatSession, limit: int = 20) -> list[dict]:
    return session.call("GET", f"/api/conversations?limit={int(limit)}").get("conversations") or []


def _cl_print_conversations(session: ChatSession, rows: list[dict]) -> None:
    con = session.console
    if not rows:
        con.print("(no conversations yet)")
        return
    for row in rows:
        cid = str(row.get("conversation_id") or "")
        mark = "*" if cid == session.conversation_id else " "
        pin = "^" if row.get("pinned") else " "
        title = " ".join(str(row.get("title") or "").split())[:70]
        con.print(f"{mark}{pin}{cid}  {int(row.get('messages') or 0):>4} msgs  "
                  f"{str(row.get('last_at') or '')[:16]}  {title}")


def _cl_command(session: ChatSession, line: str) -> bool:
    """Run one /command; False means quit."""
    con = session.console
    name, _, arg = line[1:].partition(" ")
    name, arg = name.strip().lower(), arg.strip()
    off = arg.lower() in ("off", "none", "default", "-")
    if name in ("quit", "exit", "q"):
        return False
    if name in ("help", "h", "?"):
        con.print(_CL_HELP)
    elif name == "new":
        session.conversation_id = None
        session.pending_attachments = []
        con.info("[new conversation]")
    elif name == "history":
        _cl_print_conversations(session, _cl_list_conversations(session))
    elif name == "open":
        if not arg:
            raise ClientError("usage: /open <conversation id>", code=CLIENT_EXIT_USAGE)
        data = session.call("GET", f"/api/conversation/{urllib.parse.quote(arg)}?limit=200")
        session.conversation_id = arg
        session.pending_attachments = []
        messages = data.get("messages") or []
        con.info(f"[conversation {arg}: {len(messages)} messages]")
        for msg in messages[-10:]:
            who = "you" if msg.get("role") == "user" else str(msg.get("role") or "?")
            text = str(msg.get("content") or "").strip()
            con.print(con.style(f"{who}: ", "bold") + (text if len(text) <= 600
                                                      else text[:600] + " ..."))
    elif name == "model":
        data = session.call("GET", "/api/models/aliases")
        aliases = data.get("aliases") or {}
        if not arg:
            con.print(f"model: {session.model or '(server default: ' + str(data.get('current')) + ')'}")
            for alias in sorted(aliases):
                con.print(f"  {alias:<14} {aliases[alias].get('id')}")
        elif off:
            session.model = None
            con.info("[model: server default]")
        else:
            known = {a.lower() for a in aliases}
            known |= {str(v.get("id") or "").lower() for v in aliases.values()}
            known.add(str(data.get("current") or "").lower())
            if arg.lower() not in known:
                raise ClientError(f"unknown model {arg!r}; aliases: "
                                  + (", ".join(sorted(aliases)) or "none"))
            session.model = arg
            con.info(f"[model: {arg}]")
    elif name == "agent":
        if off:
            session.agent_id = None
            con.info("[agent profile: none]")
            return True
        agents = session.call("GET", "/api/agents").get("agents") or []
        if not arg:
            con.print(f"agent profile: {session.agent_id or '(none)'}")
            for row in agents:
                con.print(f"  {row.get('id')}  {row.get('name') or ''}")
            return True
        match = next((a for a in agents if str(a.get("id")) == arg
                      or str(a.get("name") or "").lower() == arg.lower()), None)
        if match is None:
            raise ClientError(f"no agent profile {arg!r}; /agent lists them")
        session.agent_id = str(match["id"])
        con.info(f"[agent profile: {match.get('name') or match['id']}]")
    elif name == "project":
        if off:
            session.project_id = None
            con.info("[project: none]")
            return True
        status, data = session.client.request("GET", "/api/projects")
        projects = (data.get("projects") or []) if status == 200 else []
        if not arg:
            con.print(f"project: {session.project_id or '(none)'}")
            for row in projects:
                con.print(f"  {row.get('id')}  {row.get('name') or ''}")
            return True
        match = next((p for p in projects if str(p.get("id")) == arg
                      or str(p.get("name") or "").lower() == arg.lower()), None)
        if status == 200 and match is None:
            raise ClientError(f"no project {arg!r}; /project lists them")
        session.project_id = str(match["id"]) if match else arg
        con.info(f"[project: {session.project_id}]")
    elif name == "mode":
        if arg not in ("auto", "chat", "agent"):
            raise ClientError("usage: /mode auto|chat|agent", code=CLIENT_EXIT_USAGE)
        session.mode = arg
        con.info(f"[mode: {arg}]")
    elif name == "attach":
        if not arg:
            raise ClientError("usage: /attach <path>", code=CLIENT_EXIT_USAGE)
        row = session.attach(arg)
        con.info(f"[attached {row.get('filename') or arg} ({row.get('kind')}, "
                 f"{row.get('bytes')} bytes): goes with your next message]")
    elif name == "think":
        con.think = not con.think
        con.info(f"[reasoning {'shown' if con.think else 'hidden'}]")
    elif name == "stop":
        count = session.cancel_active()
        con.info(f"[cancelled {count} running turn{'s' if count != 1 else ''}]")
    else:
        raise ClientError(f"unknown command /{name}; /help lists them", code=CLIENT_EXIT_USAGE)
    return True


def _cl_reader(stdin: Any, out: Any) -> Callable[[str], str | None]:
    """Line input: readline-backed input() on a terminal, plain lines otherwise."""
    if _cl_isatty(stdin) and stdin is sys.stdin:
        try:
            import readline  # noqa: F401 - line editing and history for input()
            hist = client_credentials_path().parent / "history"
            try:
                readline.read_history_file(str(hist))
            except OSError:
                pass

            import atexit

            def save_history() -> None:
                try:
                    hist.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    readline.set_history_length(1000)
                    readline.write_history_file(str(hist))
                    os.chmod(hist, 0o600)
                except OSError:
                    pass
            atexit.register(save_history)
        except ImportError:
            pass

        def read(prompt: str) -> str | None:
            try:
                return input(prompt)
            except EOFError:
                out.write("\n")
                return None
        return read

    def read_plain(prompt: str) -> str | None:
        line = stdin.readline()
        return line.rstrip("\r\n") if line else None
    return read_plain


def _cl_repl(session: ChatSession, read: Callable[[str], str | None], who: str) -> int:
    con = session.console
    con.info(f"connected to {session.client.url}{' as ' + who if who else ''}. "
             "/help lists commands, /quit leaves.")
    prompt = con.style("you> ", "bold")
    while True:
        try:
            line = read(prompt)
            while line is not None and line.endswith("\\"):
                more = read("...  ")
                line = line[:-1] + "\n" + (more or "")
                if more is None:
                    break
        except KeyboardInterrupt:
            con.close_line()
            con.info("(Ctrl-C) /quit or Ctrl-D leaves")
            continue
        if line is None:
            return CLIENT_EXIT_OK
        if not line.strip():
            continue
        if line.startswith("/") and not line.startswith("//"):
            try:
                if not _cl_command(session, line.strip()):
                    return CLIENT_EXIT_OK
            except ClientError as exc:
                con.error(str(exc))
            continue
        message = line[1:] if line.startswith("//") else line
        try:
            result = session.send(message)
        except KeyboardInterrupt:
            con.error("interrupted")
            return CLIENT_EXIT_INTERRUPTED
        except ClientError as exc:
            con.error(str(exc))
            continue
        if result.get("error"):
            con.error(result["error"])
        footer = _cl_usage_footer(result)
        if footer:
            con.info(footer)


# --------------------------------------------------------------------------- #
# Subcommands                                                                 #
# --------------------------------------------------------------------------- #
def _cl_streams(stdin: Any, stdout: Any, stderr: Any) -> tuple[Any, Any, Any]:
    return (stdin if stdin is not None else sys.stdin,
            stdout if stdout is not None else sys.stdout,
            stderr if stderr is not None else sys.stderr)


def _cl_parser(prog: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("--url", help="server URL (env LLM_URL; default: the last server "
                                      f"signed in to, else {CLIENT_DEFAULT_URL})")
    return parser


def _cl_parse(parser: argparse.ArgumentParser, argv: list[str], err: Any) -> Any:
    try:
        return parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)


def chat_cli(argv: list[str] | None = None, *, stdin: Any = None, stdout: Any = None,
             stderr: Any = None, env: Any = None, credentials: Path | None = None,
             ask: Callable[[str], str | None] | None = None,
             sleep: Callable[[float], Any] = time.sleep,
             client_factory: Callable[..., LLMClient] = LLMClient) -> int:
    """`deploy.py chat`: an interactive REPL, or one-shot with -m / a stdin pipe."""
    stdin, out, err = _cl_streams(stdin, stdout, stderr)
    env = os.environ if env is None else env
    parser = _cl_parser("deploy.py chat", "Chat with the served model from the terminal.")
    parser.add_argument("-m", "--message", help="one-shot: ask this, print the answer, exit")
    parser.add_argument("--stdin", action="store_true",
                        help="with -m: append what is piped on stdin to the message")
    parser.add_argument("--token", help="API token (env LLM_API_TOKEN, else the saved sign-in)")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--conversation", "-c", help="continue this conversation id")
    which.add_argument("--new", action="store_true",
                       help="start a new conversation (ignores LLM_CONVERSATION)")
    parser.add_argument("--list", action="store_true", help="list your conversations and exit")
    parser.add_argument("--json", action="store_true",
                        help="print the raw stream events as JSON lines")
    parser.add_argument("--model", help="a served model or alias (see /model)")
    parser.add_argument("--agent-id", "--profile", dest="agent_id",
                        help="run as this saved agent profile")
    parser.add_argument("--project", help="confine the agent to this project id")
    parser.add_argument("--mode", choices=("auto", "chat", "agent"), default="auto")
    parser.add_argument("--attach", action="append", default=[], metavar="PATH",
                        help="upload a file with the (first) message; repeatable")
    parser.add_argument("--think", action="store_true", help="show the model's reasoning")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="one-shot: no status lines on stderr, only errors")
    parser.add_argument("--repl", action="store_true",
                        help="interactive even when stdin is not a terminal")
    args = _cl_parse(parser, list(argv or []), err)
    if isinstance(args, int):
        return args
    try:
        base, token, entry = client_resolve(args.url, args.token, env, credentials)
    except ClientError as exc:
        err.write(f"error: {exc}\n")
        return exc.code
    client = client_factory(base, token)
    one_shot = args.message is not None or (not args.repl and not _cl_isatty(stdin))
    color = _cl_color_wanted(out, env, args.no_color)
    live = _cl_isatty(out) and not args.json
    console = _ClConsole(out, err, color=color, live=live, think=args.think,
                         json_mode=args.json, quiet=args.quiet and one_shot,
                         ask=ask or _cl_terminal_ask(stdin, err, use_stdin=not one_shot))
    conversation = None if args.new else (args.conversation or env.get("LLM_CONVERSATION") or None)
    session = ChatSession(client, console, conversation_id=conversation, model=args.model,
                          agent_id=args.agent_id, project_id=args.project, mode=args.mode,
                          max_tokens=args.max_tokens, temperature=args.temperature,
                          sleep=sleep)
    try:
        if args.list:
            rows = _cl_list_conversations(session, 50)
            if args.json:
                out.write(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
            else:
                _cl_print_conversations(session, rows)
            return CLIENT_EXIT_OK
        for path in args.attach:
            row = session.attach(path)
            console.info(f"[attached {row.get('filename') or path}]")
        if one_shot:
            return _cl_one_shot(session, args, stdin)
        if not token and entry is None:
            status, me = client.request("GET", "/api/auth/me")
            if status == 0:
                raise _cl_http_error(0, me)
            if me.get("auth_enabled") and not me.get("authenticated"):
                raise ClientError(f"not signed in to {base} (run: deploy.py login --url {base})",
                                  code=CLIENT_EXIT_AUTH)
        who = str(((entry or {}).get("user") or {}).get("username") or "")
        return _cl_repl(session, _cl_reader(stdin, out), who)
    except ClientError as exc:
        console.error(str(exc))
        return exc.code
    except KeyboardInterrupt:
        try:
            console.close_line()
        except _ClOutputClosed:
            pass
        return CLIENT_EXIT_INTERRUPTED
    except _ClOutputClosed:
        # The reader left (`| head`): stop quietly. Point the real stdout at
        # /dev/null so the interpreter's exit flush does not complain.
        if out is sys.stdout:
            try:
                os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
            except (OSError, ValueError):
                pass
        return CLIENT_EXIT_OK


def _cl_one_shot(session: ChatSession, args: Any, stdin: Any) -> int:
    console = session.console
    if args.message is not None:
        message = args.message
        if args.stdin:
            piped = stdin.read()
            message = f"{message}\n\n{piped}" if piped.strip() else message
    else:
        message = stdin.read()
    if not message.strip():
        console.error("nothing to send: pass -m \"question\" or pipe text on stdin")
        return CLIENT_EXIT_USAGE
    result = session.send(message)
    if result.get("error"):
        console.error(result["error"])
    elif not args.json:
        footer = _cl_usage_footer(result)
        if result.get("conversation_id"):
            footer = (footer + " " if footer else "") + f"[conversation {result['conversation_id']}]"
        if footer:
            console.info(footer)
    return _cl_result_code(result)


def _cl_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _cl_store_login(base: str, body: dict, method: str, path: Path | None) -> Path:
    row = body.get("api_token") if isinstance(body.get("api_token"), dict) else {}
    user = body.get("user") if isinstance(body.get("user"), dict) else {}
    entry = {"token": str(body["token"]), "token_id": row.get("id"),
             "token_hint": row.get("hint"), "scopes": row.get("scopes"),
             "expires_at": row.get("expires_at"), "method": method, "saved_at": _cl_now(),
             "user": {k: user.get(k) for k in ("id", "username", "role", "source")
                      if user.get(k) is not None}}
    return save_credential(base, entry, path)


def _cl_device_login(client: LLMClient, name: str, out: Any, err: Any,
                     sleep: Callable[[float], Any]) -> tuple[int, dict]:
    status, start = client.request("POST", "/api/auth/device/start", {}, auth=False)
    if status != 200 or not start.get("device_code"):
        if status == 0:
            return CLIENT_EXIT_UNREACHABLE, start
        err.write(f"Microsoft sign-in is not available: {_cl_error_text(status, start)}\n")
        return CLIENT_EXIT_AUTH, {}
    uri = start.get("verification_uri") or start.get("verification_url") or ""
    err.write(f"To sign in, open {uri} and enter the code {start.get('user_code')}\n")
    err.flush()
    interval = max(1, int(start.get("interval") or 5))
    deadline = time.monotonic() + max(30, int(start.get("expires_in") or 900))
    hiccups = 0
    while time.monotonic() < deadline:
        sleep(interval)
        status, body = client.request("POST", "/api/auth/device/poll",
                                      {"device_code": start["device_code"], "name": name},
                                      auth=False)
        state = str(body.get("status") or "")
        if status == 200 and state == "ok" and body.get("token"):
            return CLIENT_EXIT_OK, body
        if state == "pending":
            hiccups = 0
            continue
        if state == "slow_down":
            # RFC 8628: add 5 seconds to the interval on every slow_down.
            interval = max(interval + 5, int(body.get("interval") or 0))
            continue
        if state == "denied" or status == 403:
            err.write(f"Sign-in was declined: {_cl_error_text(status, body)}\n")
            return CLIENT_EXIT_AUTH, {}
        if state == "expired" or status == 410:
            err.write("The code expired before sign-in finished; run login --oidc again.\n")
            return CLIENT_EXIT_AUTH, {}
        if status == 0 or status >= 500:
            hiccups += 1
            if hiccups <= 5:
                continue
        err.write(f"Sign-in failed: {_cl_error_text(status, body)}\n")
        return CLIENT_EXIT_ERROR, {}
    err.write("Sign-in timed out; run login --oidc again.\n")
    return CLIENT_EXIT_AUTH, {}


def login_cli(argv: list[str] | None = None, *, stdin: Any = None, stdout: Any = None,
              stderr: Any = None, env: Any = None, credentials: Path | None = None,
              ask: Callable[[str], str | None] | None = None,
              ask_secret: Callable[[str], str] | None = None,
              sleep: Callable[[float], Any] = time.sleep) -> int:
    """`deploy.py login [--url U] [--oidc] [--name N] [--username U]`."""
    stdin, out, err = _cl_streams(stdin, stdout, stderr)
    env = os.environ if env is None else env
    parser = _cl_parser("deploy.py login", "Sign in and save an API token for this server.")
    parser.add_argument("--oidc", "--device", dest="oidc", action="store_true",
                        help="sign in with Microsoft (device code) instead of a password")
    parser.add_argument("--name", help="label for the token (default: cli on <host>)")
    parser.add_argument("--username", "-u", help="local username (asked when omitted)")
    args = _cl_parse(parser, list(argv or []), err)
    if isinstance(args, int):
        return args
    try:
        base = _cl_norm_url(args.url or env.get("LLM_URL") or CLIENT_DEFAULT_URL)
    except ClientError as exc:
        err.write(f"error: {exc}\n")
        return exc.code
    client = LLMClient(base)
    name = (args.name or f"cli on {socket.gethostname()[:40]}")[:100]
    status, me = client.request("GET", "/api/auth/me", auth=False)
    if status == 0:
        err.write(f"error: {me.get('error')}\n")
        return CLIENT_EXIT_UNREACHABLE
    if status == 200 and me.get("auth_enabled") is False:
        out.write(f"Authentication is off on {base}: no sign-in needed.\n")
        return CLIENT_EXIT_OK
    try:
        if args.oidc:
            code, body = _cl_device_login(client, name, out, err, sleep)
            if code != CLIENT_EXIT_OK:
                if code == CLIENT_EXIT_UNREACHABLE:
                    err.write(f"error: {body.get('error')}\n")
                return code
            method = "oidc"
        else:
            reader = ask or _cl_terminal_ask(stdin, err)
            username = (args.username or reader("Username: ") or "").strip()
            if not username:
                err.write("error: a username is required\n")
                return CLIENT_EXIT_USAGE
            secret = (ask_secret or (lambda p: getpass.getpass(p, stream=err)))("Password: ")
            status, body = client.request("POST", "/api/auth/cli/login",
                                          {"username": username, "password": secret,
                                           "name": name})
            del secret
            if status != 200 or not body.get("token"):
                if body.get("auth_enabled") is False:
                    out.write(f"Authentication is off on {base}: no sign-in needed.\n")
                    return CLIENT_EXIT_OK
                err.write(f"Sign-in failed: {_cl_error_text(status, body)}\n")
                if status == 0:
                    return CLIENT_EXIT_UNREACHABLE
                return CLIENT_EXIT_AUTH if status in (401, 403, 429) else CLIENT_EXIT_ERROR
            method = "password"
    except KeyboardInterrupt:
        err.write("\nSign-in cancelled.\n")
        return CLIENT_EXIT_INTERRUPTED
    path = _cl_store_login(base, body, method, credentials)
    user = body.get("user") or {}
    out.write(f"Signed in to {base} as {user.get('username') or '?'} "
              f"({user.get('role') or 'user'}). Token saved to {path}\n")
    return CLIENT_EXIT_OK


def logout_cli(argv: list[str] | None = None, *, stdout: Any = None, stderr: Any = None,
               env: Any = None, credentials: Path | None = None) -> int:
    """`deploy.py logout [--url U]`: revoke the saved token and forget it."""
    _, out, err = _cl_streams(None, stdout, stderr)
    env = os.environ if env is None else env
    parser = _cl_parser("deploy.py logout", "Revoke this server's saved token and remove it.")
    parser.add_argument("--token", help="revoke this token instead of the saved one")
    args = _cl_parse(parser, list(argv or []), err)
    if isinstance(args, int):
        return args
    try:
        data = load_credentials(credentials)
        base = _cl_norm_url(args.url or env.get("LLM_URL") or data.get("default")
                            or CLIENT_DEFAULT_URL)
    except ClientError as exc:
        err.write(f"error: {exc}\n")
        return exc.code
    entry = data["servers"].get(base) if isinstance(data["servers"].get(base), dict) else None
    token = args.token or (entry or {}).get("token")
    if not token:
        out.write(f"Not signed in to {base}.\n")
        return CLIENT_EXIT_OK
    client = LLMClient(base, token)
    revoked, why = False, ""
    token_id = None if args.token else (entry or {}).get("token_id")
    if token_id:
        status, body = client.request("DELETE", f"/api/auth/tokens/{urllib.parse.quote(str(token_id))}")
        revoked = status == 200
        why = "" if revoked else _cl_error_text(status, body)
    if not revoked:
        # A chat-scoped token cannot use the token admin route; logging out
        # with the token itself retires it the same way.
        status, body = client.request("POST", "/api/auth/logout")
        if status == 200:
            status, me = client.request("GET", "/api/auth/me")
            revoked = status == 200 and not me.get("authenticated")
        why = why or _cl_error_text(status, body)
    if entry is not None and not args.token:
        remove_credential(base, credentials)
    if revoked:
        out.write(f"Signed out of {base}: token revoked and removed.\n")
    else:
        err.write(f"warning: could not revoke the token on {base} ({why}); "
                  "removed it locally. Revoke it in Settings > API tokens.\n")
    return CLIENT_EXIT_OK


def whoami_cli(argv: list[str] | None = None, *, stdout: Any = None, stderr: Any = None,
               env: Any = None, credentials: Path | None = None) -> int:
    """`deploy.py whoami [--url U] [--token T] [--json]`."""
    _, out, err = _cl_streams(None, stdout, stderr)
    env = os.environ if env is None else env
    parser = _cl_parser("deploy.py whoami", "Show who the saved sign-in is on this server.")
    parser.add_argument("--token")
    parser.add_argument("--json", action="store_true")
    args = _cl_parse(parser, list(argv or []), err)
    if isinstance(args, int):
        return args
    try:
        base, token, entry = client_resolve(args.url, args.token, env, credentials)
    except ClientError as exc:
        err.write(f"error: {exc}\n")
        return exc.code
    status, me = LLMClient(base, token).request("GET", "/api/auth/me")
    if status != 200:
        exc = _cl_http_error(status, me, base)
        err.write(f"error: {exc}\n")
        return exc.code
    if args.json:
        out.write(json.dumps(dict(me, url=base), indent=2) + "\n")
    if me.get("auth_enabled") is False:
        if not args.json:
            out.write(f"Authentication is off on {base}: everyone is the local user.\n")
        return CLIENT_EXIT_OK
    if not me.get("authenticated"):
        reason = "the token was rejected (revoked or expired)" if token else "no saved sign-in"
        err.write(f"Not signed in to {base}: {reason}. Run: deploy.py login --url {base}\n")
        return CLIENT_EXIT_AUTH
    if not args.json:
        user = me.get("user") or {}
        via = (entry or {}).get("token_hint") if entry and token == entry.get("token") else None
        out.write(f"{user.get('username')} ({user.get('role')}, {user.get('source') or 'local'}) "
                  f"on {base}" + (f" via {via}" if via else "") + "\n")
    return CLIENT_EXIT_OK


def client_main(argv: list[str] | None = None) -> int:
    """Standalone entry: `python3 client.py chat|login|logout|whoami ...`."""
    argv = sys.argv[1:] if argv is None else list(argv)
    commands = {"chat": chat_cli, "login": login_cli, "logout": logout_cli,
                "whoami": whoami_cli}
    if not argv or argv[0] not in commands:
        sys.stderr.write("usage: client.py {chat,login,logout,whoami} [options]\n")
        return CLIENT_EXIT_USAGE
    return commands[argv[0]](argv[1:])


__all__ = [
    "CLIENT_DEFAULT_URL",
    "CLIENT_EXIT_AUTH",
    "CLIENT_EXIT_ERROR",
    "CLIENT_EXIT_INTERRUPTED",
    "CLIENT_EXIT_OK",
    "CLIENT_EXIT_UNREACHABLE",
    "CLIENT_EXIT_USAGE",
    "ChatSession",
    "ClientError",
    "LLMClient",
    "SSEParser",
    "chat_cli",
    "client_credentials_path",
    "client_main",
    "client_resolve",
    "load_credentials",
    "login_cli",
    "logout_cli",
    "remove_credential",
    "save_credential",
    "whoami_cli",
]


if __name__ == "__main__":
    sys.exit(client_main())
