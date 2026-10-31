"""Tests for the terminal client (local_llm/client.py, `deploy.py chat|login|logout|whoami`).

The client is exercised against the REAL app served by uvicorn on a free
loopback port in a background thread (never 8000/8080), with a scripted agent
in place of the model: the client speaks raw HTTP through http.client, so
nothing short of a real socket would test its SSE parsing, resume and cancel.
Every call passes its own env and credentials path, so the user's
LLM_API_TOKEN and ~/.config/local-llm are never read or written.
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import socket
import stat
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_llm.config as config_mod  # noqa: E402
config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-client-"))

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

import local_llm.client as client_mod  # noqa: E402
from local_llm.api import create_app  # noqa: E402
from local_llm.auth import hash_password  # noqa: E402
from local_llm.client import (  # noqa: E402
    CLIENT_EXIT_AUTH, CLIENT_EXIT_INTERRUPTED, CLIENT_EXIT_OK, CLIENT_EXIT_UNREACHABLE,
    LLMClient, SSEParser, chat_cli, client_resolve, load_credentials, login_cli,
    logout_cli, remove_credential, save_credential, whoami_cli)
from local_llm.config import Config  # noqa: E402
from local_llm.core import ADAPTER_DIR  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.training import RetrainManager  # noqa: E402
from local_llm.turnctx import current_turn  # noqa: E402

PASSWORD = "adminpw123"
SECRET_PW = "s3cret-Pa55-do-not-log"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def build_app(**overrides):
    """Same construction as tests/test_streams.py:build_app."""
    base = dict(auth_enabled=True, admin_username="admin", admin_password=PASSWORD,
                allow_test_user=False, node_token="secret-node-token",
                skills_dir=tempfile.mkdtemp(prefix="local-llm-test-skills-"),
                agent_enabled=True, knowledge_triage=False)
    base.update(overrides)
    cfg = Config(**base)
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)  # never started
    rm = RetrainManager(db, mm, cfg)
    app = create_app(cfg, db, mm, rm, ToolRegistry(cfg, db))
    app.state.model_manager = mm
    mm.status = "ready"
    return app, cfg, db


class Recorder:
    """ASGI wrapper noting every request path and each /api/chat/stream body."""

    def __init__(self, app):
        self.app = app
        self.state = app.state
        self.paths: list[str] = []
        self.bodies: list[dict] = []

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        query = scope.get("query_string", b"").decode()
        self.paths.append(f"{scope['method']} {scope['path']}" + (f"?{query}" if query else ""))
        if scope["path"] != "/api/chat/stream":
            return await self.app(scope, receive, send)
        chunks: list[bytes] = []

        async def recording_receive():
            message = await receive()
            if message["type"] == "http.request":
                chunks.append(message.get("body", b""))
                if not message.get("more_body"):
                    self.bodies.append(json.loads(b"".join(chunks) or b"{}"))
            return message
        return await self.app(scope, recording_receive, send)


def _free_port() -> int:
    for _ in range(50):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port not in (8000, 8080):
            return port
    raise RuntimeError("no free port")


@contextlib.contextmanager
def serve(app):
    """Run `app` on a free loopback port in a thread; yields its base URL."""
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="warning", lifespan="on",
                                           timeout_graceful_shutdown=2))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started:
        if time.time() > deadline or not thread.is_alive():
            raise RuntimeError("test server did not start")
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(15)


class TTYBuffer(io.StringIO):
    """A StringIO that says it is a terminal (live token rendering)."""

    def isatty(self) -> bool:
        return True


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.texts: list[str] = []

    def emit(self, record):
        try:
            self.texts.append(record.getMessage() + " " + repr(record.__dict__))
        except Exception:  # noqa: BLE001
            self.texts.append(repr(record.__dict__))


@contextlib.contextmanager
def capture_logs():
    cap = LogCapture()
    names = ("", "local_llm", "uvicorn", "uvicorn.access", "uvicorn.error")
    loggers = [logging.getLogger(n) for n in names]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.addHandler(cap)
        lg.setLevel(logging.DEBUG)
    try:
        yield cap
    finally:
        for lg, level in zip(loggers, levels):
            lg.removeHandler(cap)
            lg.setLevel(level)


def _final(answer: str, **extra) -> dict:
    event = {"type": "final", "answer": answer, "steps": 1, "trace": [],
             "tools_used": [], "elapsed_ms": 1, "prompt_tokens": 1,
             "completion_tokens": 1, "truncated": False, "changed_files": [], "diff": ""}
    event.update(extra)
    return event


def _usage() -> dict:
    return {"type": "usage", "step": 1, "prompt_tokens": 12, "completion_tokens": 4,
            "ttft_ms": 210.0, "total_ms": 400.0, "decode_tps": 31.5, "estimated": False,
            "cached_tokens": 0, "reasoning_tokens": 0, "queue_ms": 0.0,
            "decode_tps_estimated": False}


def _scripted_agent(app, seen: list | None = None):
    """Reasoning, a tool call and its result, streamed tokens, usage, final."""
    async def fake_run(message, history, conversation_id=None, max_tokens=None,
                       temperature=None, cancel=None, **kw):
        if seen is not None:
            seen.append({"message": message, "conversation_id": conversation_id, **kw})
        yield {"type": "step", "step": 1, "max_steps": 6}
        yield {"type": "think_token", "token": "pondering", "step": 1}
        yield {"type": "tool_call", "id": "c1", "name": "calc", "args": {"expr": "2+2"},
               "step": 1}
        yield {"type": "tool_result", "id": "c1", "name": "calc", "result": "4",
               "error": None, "step": 1}
        yield {"type": "step", "step": 2, "max_steps": 6}
        for piece in ("Hello ", "there"):
            yield {"type": "token", "token": piece, "step": 2}
        yield _usage()
        yield _final("Hello there")
    app.state.agent.run_iterating = fake_run


def _login(url: str, creds: Path, password: str = PASSWORD) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = login_cli(["--url", url, "--username", "admin"], stdout=out, stderr=err, env={},
                     credentials=creds, ask_secret=lambda prompt: password)
    return code, out.getvalue(), err.getvalue()


def _chat(argv, creds: Path, *, stdin=None, stdout=None, env=None, **kw):
    out = stdout if stdout is not None else io.StringIO()
    err = io.StringIO()
    code = chat_cli(argv, stdin=stdin if stdin is not None else io.StringIO(""),
                    stdout=out, stderr=err, env=env if env is not None else {"NO_COLOR": "1"},
                    credentials=creds, **kw)
    return code, out.getvalue(), err.getvalue()


def _creds() -> Path:
    return Path(tempfile.mkdtemp(prefix="llm-cli-home-")) / "local-llm" / "credentials.json"


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #
def test_sse_parser_handles_ids_comments_crlf_cr_and_multiline_data():
    parser = SSEParser()
    events = parser.feed(": keepalive\r\n\r\nid: 3\r\ndata: {\"a\": 1}\r\n\r\n")
    assert events == [("3", '{"a": 1}')]
    # Split across chunks, including inside a CRLF and a field name.
    assert parser.feed("id: 4\ndat") == []
    assert parser.feed("a: x\ndata:y\r") == []
    assert parser.feed("\n\r\n") == [("4", "x\ny")]
    # Lone CR line ends; an event without an id; retry/event fields ignored.
    assert parser.feed("event: msg\rretry: 10\rdata: [DONE]\r\r\n") == [(None, "[DONE]")]
    assert parser.last_id == "4"
    # A blank line with no data dispatches nothing.
    assert parser.feed("\n\n") == []


def test_credentials_file_is_private_per_url_and_resolved_in_order():
    creds = _creds()
    save_credential("http://Studio.local:8000/", {"token": "tok-a", "token_id": "1"}, creds)
    save_credential("127.0.0.1:9000", {"token": "tok-b"}, creds)
    assert stat.S_IMODE(os.stat(creds).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(creds.parent).st_mode) == 0o700
    data = load_credentials(creds)
    assert set(data["servers"]) == {"http://studio.local:8000", "http://127.0.0.1:9000"}
    assert data["default"] == "http://127.0.0.1:9000"
    assert data["servers"]["http://studio.local:8000"]["url"] == "http://studio.local:8000"
    # URL: --url, LLM_URL, the last sign-in; token: --token, LLM_API_TOKEN, file.
    assert client_resolve(None, None, {}, creds)[:2] == ("http://127.0.0.1:9000", "tok-b")
    assert client_resolve(None, None, {"LLM_URL": "http://studio.local:8000"},
                          creds)[:2] == ("http://studio.local:8000", "tok-a")
    assert client_resolve("http://studio.local:8000", None,
                          {"LLM_API_TOKEN": "env-tok"}, creds)[1] == "env-tok"
    assert client_resolve("http://studio.local:8000", "flag-tok",
                          {"LLM_API_TOKEN": "env-tok"}, creds)[1] == "flag-tok"
    assert client_resolve("http://other:1", None, {}, creds)[1] is None
    assert client_resolve(None, None, {}, _creds())[0] == "http://127.0.0.1:8000"
    # Removing the default moves it to a remaining server; the mode stays 0600.
    assert remove_credential("http://127.0.0.1:9000", creds)["token"] == "tok-b"
    assert load_credentials(creds)["default"] == "http://studio.local:8000"
    assert stat.S_IMODE(os.stat(creds).st_mode) == 0o600
    assert remove_credential("http://nowhere:1", creds) is None
    # A corrupt file reads as empty instead of crashing the client.
    creds.write_text("{not json")
    assert load_credentials(creds)["servers"] == {}


def test_cli_dispatches_the_client_subcommands():
    from local_llm.cli import subcommands
    commands = subcommands()
    for name in ("chat", "login", "logout", "whoami"):
        assert commands[name].__module__ == client_mod.__name__


def test_unreachable_server_exits_4():
    port = _free_port()  # nothing listens there
    code, out, err = _chat(["-m", "hi", "--url", f"http://127.0.0.1:{port}"], _creds())
    assert code == CLIENT_EXIT_UNREACHABLE and out == ""
    assert "unreachable" in err


# --------------------------------------------------------------------------- #
# Against the real app
# --------------------------------------------------------------------------- #
def test_login_one_shot_whoami_and_the_password_never_leaks():
    app, cfg, db = build_app()
    seen: list = []
    _scripted_agent(app, seen)
    db.create_user("carol", hash_password(SECRET_PW), "user")
    creds = _creds()
    with serve(app) as url, capture_logs() as logs:
        out, err = io.StringIO(), io.StringIO()
        code = login_cli(["--url", url], stdout=out, stderr=err, env={}, credentials=creds,
                         ask=lambda prompt: "carol", ask_secret=lambda prompt: SECRET_PW)
        assert code == CLIENT_EXIT_OK, err.getvalue()
        assert "carol" in out.getvalue()
        bad_out, bad_err = io.StringIO(), io.StringIO()
        assert login_cli(["--url", url, "-u", "carol"], stdout=bad_out, stderr=bad_err,
                         env={}, credentials=_creds(),
                         ask_secret=lambda p: SECRET_PW + "x") == CLIENT_EXIT_AUTH
        assert "invalid" in bad_err.getvalue()

        entry = load_credentials(creds)["servers"][url]
        assert entry["token"].startswith("llm_") and entry["user"]["username"] == "carol"
        assert entry["method"] == "password"

        w_out, w_err = io.StringIO(), io.StringIO()
        assert whoami_cli(["--url", url], stdout=w_out, stderr=w_err, env={},
                          credentials=creds) == CLIENT_EXIT_OK
        assert w_out.getvalue().startswith("carol (user")

        # One-shot: only the answer on stdout; tools, usage, notices on stderr.
        code, answer, status = _chat(["-m", "hello", "--url", url, "--think"], creds)
        assert code == CLIENT_EXIT_OK, status
        assert answer == "Hello there\n"
        assert "[tool] calc(expr=2+2)" in status and "[tool] calc ok" in status
        assert "pondering" in status and "TTFT 210 ms" in status and "31.5 tok/s" in status
        assert "[conversation " in status
        assert seen[-1]["message"] == "hello"

        # A stdin pipe is the message; --quiet leaves stderr empty.
        code, answer, status = _chat(["--url", url, "-q"], creds,
                                     stdin=io.StringIO("question from a pipe\n"))
        assert code == CLIENT_EXIT_OK and answer == "Hello there\n" and status == ""
        assert seen[-1]["message"].strip() == "question from a pipe"
        code, _, _ = _chat(["-m", "summarise", "--stdin", "--url", url, "-q"], creds,
                           stdin=io.StringIO("piped body"))
        assert seen[-1]["message"] == "summarise\n\npiped body"

        # --json: one raw event per line, nothing rendered.
        code, raw, _ = _chat(["-m", "hi", "--json", "--url", url], creds)
        kinds = [json.loads(line)["type"] for line in raw.splitlines()]
        assert code == CLIENT_EXIT_OK and kinds[0] in ("start", "queued")
        assert "tool_call" in kinds and kinds[-1] == "final"

        # --list prints the conversations this user has.
        code, listing, _ = _chat(["--list", "--url", url], creds)
        assert code == CLIENT_EXIT_OK and "hello" in listing

        # Live rendering on a terminal prints each token once, then a newline.
        tty = TTYBuffer()
        code, _, _ = _chat(["-m", "hi", "--url", url], creds, stdout=tty)
        assert code == CLIENT_EXIT_OK and tty.getvalue() == "Hello there\n"
        # NO_COLOR unset on a terminal: colour; NO_COLOR set: none (above).
        tty = TTYBuffer()
        _, _, styled = _chat(["-m", "hi", "--url", url], creds, stdout=tty, env={})
        assert tty.getvalue() == "Hello there\n"  # the answer itself is never styled
        assert "\x1b[" in styled and "\x1b[" not in status

    everything = "\n".join([out.getvalue(), err.getvalue(), bad_out.getvalue(),
                            bad_err.getvalue(), w_out.getvalue(), creds.read_text(),
                            *logs.texts])
    assert SECRET_PW not in everything
    assert any("auth.login" in t for t in logs.texts)  # the capture did see the server


class ClosedPipe(io.StringIO):
    def write(self, text):
        raise BrokenPipeError(32, "Broken pipe")


def test_a_closed_stdout_pipe_stops_quietly_without_resuming():
    app, _, _ = build_app()
    _scripted_agent(app)
    rec = Recorder(app)
    creds = _creds()
    with serve(rec) as url:
        assert _login(url, creds)[0] == CLIENT_EXIT_OK
        code, _, err = _chat(["-m", "hi", "--url", url], creds, stdout=ClosedPipe())
    assert code == CLIENT_EXIT_OK, err
    assert "connection lost" not in err
    assert not any("/events?after=" in p for p in rec.paths)


def test_not_signed_in_exits_3():
    app, _, _ = build_app()
    _scripted_agent(app)
    with serve(app) as url:
        code, out, err = _chat(["-m", "hi", "--url", url], _creds())
        assert code == CLIENT_EXIT_AUTH and out == ""
        assert "not signed in" in err and "login" in err
        code, _, err = _chat(["-m", "hi", "--url", url, "--token", "llm_bogus"], _creds())
        assert code == CLIENT_EXIT_AUTH
        w_err = io.StringIO()
        assert whoami_cli(["--url", url], stdout=io.StringIO(), stderr=w_err, env={},
                          credentials=_creds()) == CLIENT_EXIT_AUTH


def test_repl_commands_from_a_scripted_stdin():
    app, cfg, db = build_app()
    cfg.model_aliases = json.dumps({"mine": cfg.model})
    seen: list = []
    _scripted_agent(app, seen)
    rec = Recorder(app)
    creds = _creds()
    note = Path(tempfile.mkdtemp()) / "notes.txt"
    note.write_text("remember the milk\n")
    script = "\n".join([
        "/help", "/mode agent", "/model nope", "/model mine", "/think", "/think",
        f"/attach {note}", "hello there", "/history", "/new", "second \\", "question",
        "/open conv-cli-1", "third question", "/bogus", "//literal slash", "/quit",
        "never sent"]) + "\n"
    with serve(rec) as url:
        assert _login(url, creds)[0] == CLIENT_EXIT_OK
        code, out, err = _chat(["--repl", "--url", url, "--conversation", "conv-cli-1"],
                               creds, stdin=io.StringIO(script))
    assert code == CLIENT_EXIT_OK, err
    bodies = rec.bodies
    assert [b["message"] for b in bodies] == ["hello there", "second \nquestion",
                                              "third question", "/literal slash"]
    first, second, third = bodies[0], bodies[1], bodies[2]
    assert first["conversation_id"] == "conv-cli-1" and first["agent"] is True
    assert first["model"] == "mine" and len(first["attachments"]) == 1
    assert "attachments" not in second and "conversation_id" not in second
    assert third["conversation_id"] == "conv-cli-1"
    assert seen[0].get("attachments") and not seen[1].get("attachments")
    assert "/attach <path>" in out                       # /help
    assert "conv-cli-1" in out                           # /history
    assert "you: hello there" in out                     # /open replays the transcript
    assert out.count("Hello there") >= 4                 # four answers
    assert "unknown model 'nope'" in err and "unknown command /bogus" in err
    assert "[reasoning shown]" in err and "[reasoning hidden]" in err
    assert "[attached notes.txt" in err


def test_approval_and_consent_prompts_are_answered_and_posted():
    from local_llm.cloud import CONSENT
    from local_llm.policy import BROKER
    app, cfg, db = build_app()
    rec = Recorder(app)

    async def fake_run(message, history, conversation_id=None, max_tokens=None,
                       temperature=None, cancel=None, **kw):
        turn = current_turn()
        aid = BROKER.request(turn, "call-1", "shell", {"command": "rm -rf build"}, "high",
                             "runs a shell command")
        yield BROKER.event(aid, step=1)
        res = await BROKER.wait(aid, 10)
        yield BROKER.resolved_event(res)
        cid = CONSENT.request(turn, reason="the local model is busy", trigger="busy",
                              est_prompt_tokens=42, provider="openai", model="gpt-x",
                              scopes=["once", "conversation", "always"], timeout=10)
        yield CONSENT.event(cid)
        decision, scope = await CONSENT.wait(cid, 10)
        yield {"type": "consent_resolved", "id": cid, "decision": decision, "scope": scope}
        yield _final(f"approval={res.decision}/{res.scope} consent={decision}/{scope}")
    app.state.agent.run_iterating = fake_run

    creds = _creds()
    with serve(rec) as url:
        assert _login(url, creds)[0] == CLIENT_EXIT_OK
        answers = iter(["?", "a", "c"])  # an unknown key is asked again
        prompts: list[str] = []

        def ask(prompt):
            prompts.append(prompt)
            return next(answers)
        code, out, err = _chat(["-m", "clean up", "--url", url, "--conversation", "conv-ap"],
                               creds, ask=ask)
        assert code == CLIENT_EXIT_OK, err
        assert out == "approval=allow/once consent=allow/conversation\n"
        assert "[a]llow once / allow for [c]onversation / allow [t]ool / [d]eny" in prompts[0]
        assert len(prompts) == 3 and "allow [t]his always" in prompts[2]
        assert "[approval] shell(command=rm -rf build) risk high" in err
        assert "[consent] send this turn to the cloud model openai/gpt-x" in err
        assert any("/approvals/ap_" in p for p in rec.paths)
        assert any("/consent/cc_" in p for p in rec.paths)

        # No way to answer (no terminal): denied, and the turn still finishes.
        code, out, _ = _chat(["-m", "again", "--url", url], creds, ask=lambda p: None)
        assert code == CLIENT_EXIT_OK and out.startswith("approval=deny/once")


def _blocking_agent(app, release: threading.Event, flags: dict, tokens=("par", "tial")):
    async def fake_run(message, history, conversation_id=None, max_tokens=None,
                       temperature=None, cancel=None, **kw):
        for piece in tokens:
            yield {"type": "token", "token": piece, "step": 1}
        turn = current_turn()
        while not release.is_set():
            if turn is not None and turn.cancelled():
                flags["cancelled"] = True
                yield {"type": "cancelled", "step": 1, "partial": "".join(tokens),
                       "trace": []}
                return
            await asyncio.sleep(0.01)
        yield _final("".join(tokens) + " done")
    app.state.agent.run_iterating = fake_run


class InterruptingOut(TTYBuffer):
    """A terminal whose first answer token arrives with a Ctrl-C."""

    def __init__(self, times: int = 1):
        super().__init__()
        self.times = times

    def write(self, text):
        if self.times and text and text.strip():
            self.times -= 1
            raise KeyboardInterrupt
        return super().write(text)


class InterruptingErr(io.StringIO):
    """stderr that delivers a Ctrl-C when the client says it is cancelling."""

    def write(self, text):
        if "cancelling" in text:
            raise KeyboardInterrupt
        return super().write(text)


def test_ctrl_c_cancels_the_turn_and_a_second_ctrl_c_exits():
    app, cfg, db = build_app()
    release, flags = threading.Event(), {}
    _blocking_agent(app, release, flags)
    rec = Recorder(app)
    creds = _creds()
    try:
        with serve(rec) as url:
            assert _login(url, creds)[0] == CLIENT_EXIT_OK
            out = InterruptingOut()
            code, _, err = _chat(["-m", "long", "--url", url], creds, stdout=out)
            assert code == CLIENT_EXIT_INTERRUPTED, err
            assert flags.get("cancelled") is True
            assert any(p.startswith("POST /api/chat/turns/") and p.endswith("/cancel")
                       for p in rec.paths)
            assert "cancelling" in err and "[cancelled]" in err
            assert "partial" in out.getvalue()  # the partial answer is still shown

            # A second Ctrl-C (while cancelling) leaves at once, without the cancel.
            rec.paths.clear()
            flags.clear()
            out, err_stream = InterruptingOut(), InterruptingErr()
            code = chat_cli(["-m", "long", "--url", url], stdin=io.StringIO(""), stdout=out,
                            stderr=err_stream, env={"NO_COLOR": "1"}, credentials=creds)
            assert code == CLIENT_EXIT_INTERRUPTED
            assert not any(p.endswith("/cancel") for p in rec.paths)
            release.set()
    finally:
        release.set()


def test_a_dropped_connection_resumes_from_the_last_id():
    app, cfg, db = build_app()
    tokens = ("one ", "two ", "three ", "four")

    async def fake_run(message, history, conversation_id=None, max_tokens=None,
                       temperature=None, cancel=None, **kw):
        for piece in tokens:
            yield {"type": "token", "token": piece, "step": 1}
            await asyncio.sleep(0.05)
        yield _final("".join(tokens))
    app.state.agent.run_iterating = fake_run
    rec = Recorder(app)

    class DroppingClient(LLMClient):
        drops = 1

        def iter_events(self, resp):
            count = 0
            for item in super().iter_events(resp):
                yield item
                count += 1
                if self.drops and count == 3:
                    self.drops -= 1
                    raise ConnectionResetError("simulated drop")

    creds = _creds()
    sleeps: list = []
    with serve(rec) as url:
        assert _login(url, creds)[0] == CLIENT_EXIT_OK
        out = TTYBuffer()
        code, _, err = _chat(["-m", "count", "--url", url], creds, stdout=out,
                             client_factory=DroppingClient, sleep=sleeps.append)
    assert code == CLIENT_EXIT_OK, err
    assert "connection lost" in err and "resumed 1x" in err
    assert sleeps and sleeps[0] > 0
    text = out.getvalue()
    assert text.replace("\n", "") == "one two three four"
    for piece in tokens:
        assert text.count(piece.strip()) == 1
    resumed = [p for p in rec.paths if "/events?after=" in p]
    assert len(resumed) == 1 and resumed[0].startswith("GET /api/chat/turns/")
    assert resumed[0].endswith("after=3")


def test_logout_revokes_the_token_and_forgets_it():
    app, cfg, db = build_app()
    rec = Recorder(app)
    creds = _creds()
    with serve(rec) as url:
        assert _login(url, creds)[0] == CLIENT_EXIT_OK
        token = load_credentials(creds)["servers"][url]["token"]
        status, me = LLMClient(url, token).request("GET", "/api/auth/me")
        assert status == 200 and me["authenticated"] is True
        out, err = io.StringIO(), io.StringIO()
        assert logout_cli(["--url", url], stdout=out, stderr=err, env={},
                          credentials=creds) == CLIENT_EXIT_OK
        assert "token revoked" in out.getvalue(), err.getvalue()
        assert url not in load_credentials(creds)["servers"]
        status, me = LLMClient(url, token).request("GET", "/api/auth/me")
        assert status == 200 and me["authenticated"] is False
        assert any(p.startswith("DELETE /api/auth/tokens/") or p == "POST /api/auth/logout"
                   for p in rec.paths)
        out = io.StringIO()
        assert logout_cli(["--url", url], stdout=out, stderr=io.StringIO(), env={},
                          credentials=creds) == CLIENT_EXIT_OK
        assert "Not signed in" in out.getvalue()
        assert whoami_cli(["--url", url], stdout=io.StringIO(), stderr=io.StringIO(),
                          env={}, credentials=creds) == CLIENT_EXIT_AUTH


def test_login_when_auth_is_off_needs_nothing():
    app, cfg, db = build_app(auth_enabled=False)
    creds = _creds()
    with serve(app) as url:
        out = io.StringIO()
        assert login_cli(["--url", url], stdout=out, stderr=io.StringIO(), env={},
                         credentials=creds, ask=lambda p: "x",
                         ask_secret=lambda p: "y") == CLIENT_EXIT_OK
        assert "no sign-in needed" in out.getvalue()
        assert not creds.exists()
        out = io.StringIO()
        assert whoami_cli(["--url", url], stdout=out, stderr=io.StringIO(), env={},
                          credentials=creds) == CLIENT_EXIT_OK
        assert "Authentication is off" in out.getvalue()


def _device_app(script: list):
    """A stand-in server for the device-code routes: polls follow `script`."""
    app = FastAPI()
    calls: list = []

    @app.get("/api/auth/me")
    def me():
        return {"authenticated": False, "auth_enabled": True, "oidc_enabled": True}

    @app.post("/api/auth/device/start")
    def start():
        return {"device_code": "dev-123", "user_code": "ABCD-EFGH",
                "verification_uri": "https://microsoft.com/devicelogin",
                "interval": 2, "expires_in": 600, "message": "ignored"}

    @app.post("/api/auth/device/poll")
    def poll(body: dict):
        calls.append(body)
        status, payload = script.pop(0)
        return JSONResponse(payload, status_code=status)
    return app, calls


def test_device_flow_polls_at_the_interval_and_handles_every_outcome():
    ok = (200, {"status": "ok", "token": "llm_device_token", "api_token": {"id": "t9"},
                "user": {"username": "dana", "role": "user", "source": "oidc"}})
    script = [(200, {"status": "pending", "interval": 2}),
              (200, {"status": "slow_down", "interval": 2}),
              (200, {"status": "pending"}), ok]
    app, calls = _device_app(script)
    creds = _creds()
    with serve(app) as url:
        sleeps: list = []
        out, err = io.StringIO(), io.StringIO()
        code = login_cli(["--url", url, "--oidc", "--name", "my laptop"], stdout=out,
                         stderr=err, env={}, credentials=creds, sleep=sleeps.append)
        assert code == CLIENT_EXIT_OK, err.getvalue()
        assert "https://microsoft.com/devicelogin" in err.getvalue()
        assert "ABCD-EFGH" in err.getvalue()
        assert sleeps == [2, 2, 7, 7]  # slow_down adds 5 seconds
        assert calls[0] == {"device_code": "dev-123", "name": "my laptop"}
        entry = load_credentials(creds)["servers"][url]
        assert entry["token"] == "llm_device_token" and entry["method"] == "oidc"
        assert entry["token_id"] == "t9" and entry["user"]["username"] == "dana"

        for outcome, words in (((403, {"status": "denied", "error": "sign-in was declined"}),
                                "declined"),
                               ((410, {"status": "expired", "error": "expired"}), "expired")):
            script[:] = [(200, {"status": "pending"}), outcome]
            err = io.StringIO()
            code = login_cli(["--url", url, "--oidc"], stdout=io.StringIO(), stderr=err,
                             env={}, credentials=_creds(), sleep=lambda s: None)
            assert code == CLIENT_EXIT_AUTH and words in err.getvalue()


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
