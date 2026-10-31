"""Tests for the chat surface's stream contract (ui_script_chat.py).

The chat surface is one script part that reads the turn's event stream and
renders it. Most of what can break is a contract the server and the page
share without either importing the other: an event type the stream emits that
the handler has no branch for (or two branches, the second unreachable), a
route the page must call (cancel, re-attach, approvals, consent) whose path
drifted, the X-Stream-Ids header that makes a turn resumable, or an id the
script looks up that the markup lost. These tests scan the rendered page and
the script text the way tests/test_app.py does, and check that every route the
page calls is actually served by the app.
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_llm.config as config_mod  # noqa: E402

config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-ui-chat-events-"))

from local_llm.api import create_app  # noqa: E402
from local_llm.config import Config  # noqa: E402
from local_llm.core import ADAPTER_DIR  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.training import RetrainManager  # noqa: E402
from local_llm.ui import render_ui  # noqa: E402
from local_llm.ui_markup import UI_BODY  # noqa: E402
from local_llm.ui_script_chat import UI_JS_CHAT  # noqa: E402
from local_llm.ui_script_panels import UI_JS_PANELS  # noqa: E402

# Every event type the stream can carry (wave 3 vocabulary plus queued and
# the cloud consent pair). Each needs exactly one branch in handleEvent.
EVENT_TYPES = (
    "start", "queued", "phase", "notice", "detail", "step", "context", "token", "think_token",
    "reason_step", "reason_token", "reason_done", "usage", "tool_call_start", "tool_call_delta",
    "tool_call", "tool_result", "approval_required", "approval_resolved", "consent_required",
    "consent_resolved", "context_injected", "cancelled", "error", "final",
)
NEW_EVENT_TYPES = (
    "queued", "tool_call_start", "tool_call_delta", "approval_required", "approval_resolved",
    "consent_required", "consent_resolved", "context_injected", "usage",
)


def build_app(**overrides):
    """A fresh app + temp DB, as tests/test_app.py:build_app builds it."""
    base = dict(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                allow_test_user=False, node_token="secret-node-token",
                skills_dir=tempfile.mkdtemp(prefix="local-llm-test-skills-"))
    base.update(overrides)
    cfg = Config(**base)
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)  # never started
    rm = RetrainManager(db, mm, cfg)
    app = create_app(cfg, db, mm, rm, ToolRegistry(cfg, db))
    app.state.model_manager = mm
    return app, cfg, db


def _function_body(src: str, name: str) -> str:
    """The text of `function name(...) {...}` in src, by brace matching."""
    m = re.search(r"(?:async )?function " + re.escape(name) + r"\(", src)
    assert m, f"{name} is not defined"
    start = src.index("{", m.end())
    depth = 0
    for i in range(start, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
    raise AssertionError(f"unbalanced body for {name}")


def _branches() -> dict[str, int]:
    body = _function_body(UI_JS_CHAT, "handleEvent")
    seen: dict[str, int] = {}
    for m in re.finditer(r'\bif \(event\.type === "(\w+)"\)', body):
        seen[m.group(1)] = seen.get(m.group(1), 0) + 1
    return seen


def test_handle_event_has_exactly_one_branch_per_event_type():
    seen = _branches()
    dupes = {k: v for k, v in seen.items() if v > 1}
    assert not dupes, f"unreachable duplicate branches: {dupes}"
    missing = [t for t in EVENT_TYPES if t not in seen]
    assert not missing, f"handleEvent has no branch for {missing}"
    for t in NEW_EVENT_TYPES:
        assert seen.get(t) == 1, t


def test_cancelled_settles_the_answer_and_says_stopped():
    start = UI_JS_CHAT.index('if (event.type === "cancelled")')
    assert "settleAnswer" in UI_JS_CHAT[start:start + 400]
    branch = UI_JS_CHAT[start:UI_JS_CHAT.index("} else if", start)]
    # The partial the server stored renders when nothing streamed (a resume).
    assert "chatShowPartial(trace, event.partial)" in branch
    assert "chatStoppedLabel" in branch
    assert '"stopped"' in _function_body(UI_JS_CHAT, "chatStoppedLabel")


def test_the_stream_sends_x_stream_ids_and_parses_sse_properly():
    send = _function_body(UI_JS_CHAT, "send")
    call = send[send.index('fetch("/api/chat/stream"'):]
    call = call[:call.index("});") + 3]
    assert '"X-Stream-Ids": "1"' in call, "the chat stream must ask for numbered frames"
    # The re-attach request asks for them too, so a resume can skip overlap.
    reopen = _function_body(UI_JS_CHAT, "chatOpenEvents")
    assert '"X-Stream-Ids": "1"' in reopen and "/events?after=" in reopen
    # One SSE reader for both, following the grammar (CR/LF/CRLF, id:,
    # comments, multi-line data), not the old split on a blank line.
    sse = _function_body(UI_JS_CHAT, "readSSE")
    assert "/[\\r\\n]/" in sse and 'field === "id"' in sse and 'charAt(0) === ":"' in sse
    assert 'data.join("\\n")' in sse
    assert 'split("\\n\\n")' not in UI_JS_CHAT
    pump = _function_body(UI_JS_CHAT, "chatPump")
    assert "readSSE(" in pump and "seq <= run.lastSeq" in pump, "overlap of a resume must be skipped"


def test_stop_cancels_on_the_server_before_aborting():
    assert 'id="stopBtn" onclick="chatStop()"' in UI_BODY
    stop = _function_body(UI_JS_CHAT, "chatStop")
    # The abort (stopStream) is the fallback after the cancel, not the first move.
    assert stop.index("chatCancelTurn(run)") < stop.rindex("stopStream()")
    assert "run.stopping = true" in stop[:stop.index("chatCancelTurn(run)")]
    cancel = _function_body(UI_JS_CHAT, "chatCancelTurn")
    assert '"/api/chat/turns/"' in cancel and '"/cancel"' in cancel
    # A turn that has not said its id yet is found by conversation.
    assert "/api/chat/active?conversation_id=" in _function_body(UI_JS_CHAT, "chatFindTurn")


def test_resume_reattaches_on_boot_and_after_a_dropped_connection():
    resume = _function_body(UI_JS_CHAT, "chatResumeActive")
    assert "/api/chat/active?conversation_id=" in resume and "chatOpenEvents(run)" in resume
    follow = _function_body(UI_JS_CHAT, "chatFollow")
    assert "chatOpenEvents(run)" in follow and "chatFindTurn(run)" in follow
    # openConversation (boot replay included) re-attaches and restores prompts.
    opener = _function_body(UI_JS_PANELS, "openConversation")
    assert "chatResumeActive(id, lastUser)" in opener
    assert "chatLoadPendingDecisions()" in opener
    assert "stopStream(" in opener.split("fetchJSON")[0], "openConversation must stop the stream first"


def test_approval_and_consent_cards_post_to_the_turn_routes():
    decide = _function_body(UI_JS_CHAT, "chatDecide")
    assert '"/api/chat/turns/"' in decide
    assert '"/approvals/"' in decide and '"/consent/"' in decide
    assert "body.args = parsed" in decide, "edited arguments travel with an allow"
    pending = _function_body(UI_JS_CHAT, "chatLoadPendingDecisions")
    assert '"/api/approvals/pending"' in pending
    # Scope labels for both kinds match the server's accepted scopes.
    from local_llm.streams import _STREAMS_APPROVAL_SCOPES, _STREAMS_CONSENT_SCOPES
    for scope in _STREAMS_APPROVAL_SCOPES:
        assert re.search(r"approval: \{[^}]*\b" + scope + r":", UI_JS_CHAT), scope
    for scope in _STREAMS_CONSENT_SCOPES:
        assert re.search(r"consent: \{[^}]*\b" + scope + r":", UI_JS_CHAT), scope


def test_history_replay_renders_meta():
    opener = _function_body(UI_JS_PANELS, "openConversation")
    assert "chatReplayMeta(div, m.meta)" in opener
    replay = _function_body(UI_JS_CHAT, "chatReplayMeta")
    for key in ("meta.reasoning", "meta.tools", "meta.cancelled"):
        assert key in replay, key


def test_composer_and_header_controls_exist_and_are_wired():
    page = render_ui()
    for element in ("projectSel", "agentSel", "attBtn", "attInput", "attChips", "micBtn",
                    "composerMeta", "notifMenu", "notifBtn", "notifBadge", "notifPop", "notifList"):
        assert f'id="{element}"' in page, element
    script = re.search(r"<script>(.*?)</script>", page, re.S).group(1)
    defined = set(re.findall(r'\bid="([A-Za-z0-9_]+)"', page))
    looked_up = set(re.findall(r"getElementById\(\"([A-Za-z0-9_]+)\"\)", script))
    assert looked_up <= defined, sorted(looked_up - defined)
    send = _function_body(UI_JS_CHAT, "send")
    for field in ("body.project_id", "body.agent_id", "body.attachments"):
        assert field in send, field
    upload = _function_body(UI_JS_CHAT, "chatUploadBlob")
    assert '"/api/attachments"' in upload and '"X-Filename"' in upload
    assert "/transcribe" in _function_body(UI_JS_CHAT, "chatMicTranscribe")
    assert "/api/notifications" in _function_body(UI_JS_CHAT, "chatPollNotifications")
    # The bell rides on the health poll.
    assert "chatPollNotifications(" in _function_body(UI_JS_CHAT, "refreshHealth")
    # A project turn's document links name the project.
    assert '"&project_id="' in UI_JS_CHAT


def test_new_dom_is_built_without_inline_styles():
    # The only style writes are geometry (textarea auto-grow, meter width).
    writes = re.findall(r"\.style\.(\w+)\s*=", UI_JS_CHAT)
    assert set(writes) <= {"height", "width"}, writes
    assert 'style="' not in UI_JS_CHAT
    assert "innerHTML = text" not in UI_JS_CHAT


def test_every_route_the_chat_surface_calls_is_served():
    app, _, _ = build_app()
    served = {getattr(r, "path", "") for r in app.routes}
    for path in ("/api/chat/stream", "/api/chat/active", "/api/chat/turns/{turn_id}/events",
                 "/api/chat/turns/{turn_id}/cancel",
                 "/api/chat/turns/{turn_id}/approvals/{approval_id}",
                 "/api/chat/turns/{turn_id}/consent/{consent_id}",
                 "/api/approvals/pending", "/api/cloud/consents", "/api/projects", "/api/agents",
                 "/api/attachments", "/api/attachments/{attachment_id}",
                 "/api/attachments/{attachment_id}/transcribe", "/api/notifications",
                 "/api/notifications/read-all", "/api/notifications/{notification_id}/read",
                 "/api/conversation/{conversation_id}"):
        assert path in served, f"the chat surface calls {path}, which the app does not serve"


def test_served_page_carries_the_stream_contract():
    from fastapi.testclient import TestClient
    app, _, _ = build_app()
    with TestClient(app) as c:
        r = c.get("/")
        assert r.status_code == 200
        html = r.text
    assert html.count("<script>") == 1
    assert '"X-Stream-Ids": "1"' in html
    assert "/approvals/" in html and "/consent/" in html and "/cancel" in html


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
