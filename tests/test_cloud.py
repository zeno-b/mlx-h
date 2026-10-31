"""Tests for the optional cloud fallback (cloud.py): the decision rule, consent
(interactive, unattended, standing), the deny path, data minimisation, the
monthly cap, provider payloads and streams, metadata-only logging, the routes,
and that the default (off) is exactly today's local path."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_llm.config as config_mod  # noqa: E402

config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-cloud-"))

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import local_llm.cloud as cloud_mod  # noqa: E402
from local_llm.api import create_app  # noqa: E402
from local_llm.auth import hash_password  # noqa: E402
from local_llm.cloud import (  # noqa: E402
    CONSENT, CloudTrigger, bind_cloud, cloud_anthropic_payload, cloud_measured_prefill_tps,
    cloud_minimise_messages, cloud_month_usage, cloud_offload, cloud_openai_payload,
    cloud_settings, cloud_trigger, cloud_user_allowed, grant_cloud_consent,
    list_cloud_consents, record_cloud_usage, standing_cloud_consent,
)
from local_llm.config import Config  # noqa: E402
from local_llm.core import ADAPTER_DIR  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.llm import GenerationStats  # noqa: E402
from local_llm.model_client import ModelClient  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.training import RetrainManager  # noqa: E402
from local_llm.turnctx import TurnContext, reset_current_turn, set_current_turn  # noqa: E402

API_KEY = "sk-cloudtestkey0123456789abcdefghij"
PROMPT_MARK = "ZEBRA-PROMPT-MARKER"
REPLY_MARK = "OKAPI-REPLY-MARKER"
LOCAL = cloud_mod.SENTINEL_LOCAL_USER    # the implicit user with no turn
BIG = (PROMPT_MARK + " lorem ipsum dolor sit amet ") * 40      # ~1.3k chars


def _cfg(**over) -> Config:
    base = dict(cloud_fallback_enabled=True, cloud_provider="openai",
                cloud_base_url="https://cloud.test/v1", cloud_model="big-model",
                cloud_api_key=API_KEY, cloud_min_prompt_tokens=50, cloud_min_prefill_s=0,
                cloud_allowed_roles="admin", cloud_consent="per_request")
    base.update(over)
    return Config(**base)


def _db() -> Database:
    return Database(Path(tempfile.mkdtemp(prefix="llm-test-cloud-db-")) / "t.db")


def _messages(text: str = BIG) -> list[dict]:
    return [{"role": "system", "content": "You are helpful."},
            {"role": "user", "content": text}]


def _sse(events: list[dict], done: bool = True) -> bytes:
    out = "".join("data: " + json.dumps(e) + "\n\n" for e in events)
    return (out + ("data: [DONE]\n\n" if done else "")).encode()


class FakeCloud:
    """A scripted provider endpoint (OpenAI-compatible or Anthropic)."""

    def __init__(self, provider: str = "openai", status: int = 200,
                 reply: str = REPLY_MARK + " from the cloud"):
        self.provider = provider
        self.status = status
        self.reply = reply
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.bodies.append(json.loads(request.content))
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {
                "type": "overloaded_error", "message": "echo " + PROMPT_MARK}})
        if self.provider == "anthropic":
            return httpx.Response(200, content=_sse([
                {"type": "message_start", "message": {"usage": {"input_tokens": 321,
                                                                "output_tokens": 1}}},
                {"type": "content_block_start", "index": 0,
                 "content_block": {"type": "thinking", "thinking": ""}},
                {"type": "content_block_delta", "index": 0,
                 "delta": {"type": "thinking_delta", "thinking": "pondering"}},
                {"type": "content_block_start", "index": 1,
                 "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 1,
                 "delta": {"type": "text_delta", "text": self.reply}},
                {"type": "content_block_start", "index": 2, "content_block": {
                    "type": "tool_use", "id": "toolu_1", "name": "calc", "input": {}}},
                {"type": "content_block_delta", "index": 2,
                 "delta": {"type": "input_json_delta", "partial_json": "{\"x\": "}},
                {"type": "content_block_delta", "index": 2,
                 "delta": {"type": "input_json_delta", "partial_json": "2}"}},
                {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
                 "usage": {"output_tokens": 42}},
                {"type": "message_stop"},
            ], done=False), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, content=_sse([
            {"choices": [{"index": 0, "delta": {"reasoning_content": "thinking hard"},
                          "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"content": self.reply},
                          "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 400, "completion_tokens": 12,
                                      "total_tokens": 412}},
        ]), headers={"content-type": "text/event-stream"})


class FakeLocal:
    """The local mlx server (non-streaming and streaming)."""

    def __init__(self):
        self.bodies: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if body.get("stream"):
            return httpx.Response(200, content=_sse([
                {"choices": [{"index": 0, "delta": {"content": "local answer"},
                              "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"index": 0, "finish_reason": "stop",
                                                      "message": {"role": "assistant",
                                                                  "content": "local answer"}}]})


def _setup(cfg: Config, db: Database | None, cloud: FakeCloud | None = None):
    # The acting user/conversation are context variables, and an earlier test
    # file in the same process may leave them set; a turn-less call here must
    # resolve to the implicit local user, as it does in a fresh process.
    cloud_mod.set_acting_user(None)
    cloud_mod.set_acting_conversation(None)
    CONSENT.clear()
    cloud_mod._CLOUD_TPS_CACHE.clear()
    cloud_mod._CLOUD_BOUND["config"] = None
    cloud_mod._CLOUD_BOUND["db"] = None
    bind_cloud(cfg, db)
    cloud = cloud or FakeCloud(getattr(cfg, "cloud_provider", "openai"))
    cloud_mod._CLOUD_HOOKS["transport"] = httpx.MockTransport(cloud.handler)
    local = FakeLocal()
    client = ModelClient(cfg)
    mock = httpx.AsyncClient(transport=httpx.MockTransport(local.handler))
    client._client_for = lambda node: mock
    return client, cloud, local


def _turn(**kw) -> tuple[TurnContext, list[dict]]:
    events: list[dict] = []
    base = dict(user_id=LOCAL, conversation_id="conv-1", role="admin",
                interactive=True, emit=events.append)
    base.update(kw)
    if base.get("emit") is None:
        base.pop("emit")
    return TurnContext(**base), events


async def _answer(user_id: str, decision: str, scope: str = "once") -> str:
    """Answer the first consent prompt of the user, as the route would."""
    for _ in range(400):
        pending = CONSENT.pending(user_id)
        if pending:
            assert CONSENT.resolve(pending[0]["id"], user_id, decision, scope)
            return pending[0]["id"]
        await asyncio.sleep(0.01)
    raise AssertionError("no consent prompt appeared")


async def _stream_text(client: ModelClient, turn: TurnContext | None, messages=None,
                       stats=None, answer: tuple | None = None) -> str:
    async def consume():
        token = set_current_turn(turn) if turn is not None else None
        try:
            return "".join([p async for p in client.stream(messages or _messages(),
                                                           stats=stats)])
        finally:
            if token is not None:
                reset_current_turn(token)

    if answer is None:
        return await consume()
    text, _ = await asyncio.gather(consume(), _answer(*answer))
    return text


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=0)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)

    def dump(self) -> str:
        parts = []
        for r in self.records:
            parts.append(r.getMessage())
            parts.append(json.dumps({k: v for k, v in r.__dict__.items()
                                     if k not in ("args", "msg", "exc_info")}, default=str))
        return "\n".join(parts)


def _capture():
    handler = _Capture()
    pkg = logging.getLogger("local_llm")
    saved = pkg.level
    pkg.addHandler(handler)
    pkg.setLevel(logging.DEBUG)
    return handler, lambda: (pkg.removeHandler(handler), pkg.setLevel(saved))


# --------------------------------------------------------------------------- #
# Default off
# --------------------------------------------------------------------------- #
def test_default_off_is_todays_local_path():
    cfg = Config()
    assert cfg.cloud_fallback_enabled is False
    assert cfg.cloud_consent == "per_request"
    client, cloud, local = _setup(cfg, _db())
    huge = _messages(BIG * 40)
    assert asyncio.run(cloud_offload(cfg, huge)) is None
    turn, events = _turn()
    text = asyncio.run(_stream_text(client, turn, huge))
    assert text == "local answer"
    out, _stats = asyncio.run(client.complete_with_stats(huge))
    assert out == "local answer"
    assert cloud.requests == [] and len(local.bodies) == 2
    assert events == []
    assert "cloud_api_key" in Config.SECRET_FIELDS
    assert "cloud_api_key" not in Config.MUTABLE


# --------------------------------------------------------------------------- #
# Decision rule and eligibility
# --------------------------------------------------------------------------- #
def test_decision_rule_context_size_prefill():
    small = _messages("hello there")
    cfg = SimpleNamespace(context_size=4096, min_max_tokens=128, cloud_min_prompt_tokens=0,
                          cloud_min_prefill_s=0)
    assert cloud_trigger(cfg, small) is None
    over = cloud_trigger(cfg, _messages("word " * 6000))
    assert over is not None and over.reason == "context" and over.prompt_tokens > 4096
    cfg.cloud_min_prompt_tokens = 50
    assert cloud_trigger(cfg, _messages()).reason == "size"
    assert cloud_trigger(cfg, small) is None
    cfg.cloud_min_prompt_tokens = 0
    cfg.cloud_min_prefill_s = 30
    # Not measured yet: the prefill trigger does not guess.
    assert cloud_trigger(cfg, _messages(), prefill_tps=None) is None
    slow = cloud_trigger(cfg, _messages(), prefill_tps=5.0)
    assert slow.reason == "prefill" and slow.prefill_s >= 30
    assert cloud_trigger(cfg, _messages(), prefill_tps=5000.0) is None


def test_measured_prefill_speed_skips_cloud_answered_turns():
    db = _db()
    cloud_mod._CLOUD_TPS_CACHE.clear()
    assert cloud_measured_prefill_tps(db) is None
    for _ in range(3):
        db.log_metric("chat", 1000, 200, stats=GenerationStats(prompt_tokens=1000,
                                                               ttft_ms=4000.0),
                      conversation_id="local-conv")
    cloud_mod._CLOUD_TPS_CACHE.clear()
    assert abs(cloud_measured_prefill_tps(db) - 250.0) < 1e-6
    # A cloud-answered turn writes a very fast row; it must not count.
    record_cloud_usage(db, user_id="u", conversation_id="cloud-conv", provider="openai",
                       model="m", prompt_tokens=10, completion_tokens=1, status="ok")
    for _ in range(5):
        db.log_metric("chat", 100, 200, stats=GenerationStats(prompt_tokens=30000,
                                                              ttft_ms=1000.0),
                      conversation_id="cloud-conv")
    cloud_mod._CLOUD_TPS_CACHE.clear()
    assert abs(cloud_measured_prefill_tps(db) - 250.0) < 1e-6


def test_settings_and_allowlist():
    assert cloud_settings(_cfg(cloud_model="")) is None
    assert cloud_settings(_cfg(cloud_api_key="", cloud_base_url="")) is None
    # An explicit OpenAI-compatible endpoint (a LAN server) may take no key.
    lan = cloud_settings(_cfg(cloud_api_key="", cloud_base_url="http://10.0.0.5:8000/v1"))
    assert lan["url"] == "http://10.0.0.5:8000/v1/chat/completions"
    ant = cloud_settings(_cfg(cloud_provider="anthropic", cloud_base_url=""))
    assert ant["url"] == "https://api.anthropic.com/v1/messages"
    assert Config(cloud_provider="bogus").cloud_provider == "openai"
    assert Config(cloud_consent="sometimes").cloud_consent == "per_request"
    assert Config(cloud_base_url="file:///etc/passwd").cloud_base_url == ""
    cfg = _cfg(cloud_allowed_roles="admin, user:alice")
    assert cloud_user_allowed(cfg, "u1", "admin")
    assert not cloud_user_allowed(cfg, "u2", "user")
    assert cloud_user_allowed(cfg, "u3", "user", "Alice")
    assert cloud_user_allowed(_cfg(cloud_allowed_roles="*"), "u4", "viewer")
    assert not cloud_user_allowed(_cfg(cloud_allowed_roles=""), "u5", "admin")


def test_not_allowed_user_stays_local_without_asking():
    cfg = _cfg(cloud_allowed_roles="admin")
    client, cloud, local = _setup(cfg, _db())
    turn, events = _turn(role="user", user_id="someone")
    assert asyncio.run(_stream_text(client, turn)) == "local answer"
    assert events == [] and cloud.requests == [] and CONSENT.pending() == []


# --------------------------------------------------------------------------- #
# Consent
# --------------------------------------------------------------------------- #
def test_interactive_consent_once_goes_to_cloud_and_accounts():
    db = _db()
    cfg = _cfg()
    client, cloud, local = _setup(cfg, db)
    turn, events = _turn()
    stats = GenerationStats()
    text = asyncio.run(_stream_text(client, turn, stats=stats,
                                    answer=(LOCAL, "allow", "once")))
    assert text == "<think>thinking hard</think>" + REPLY_MARK + " from the cloud"
    assert local.bodies == [] and len(cloud.requests) == 1
    kinds = [e["type"] for e in events]
    assert kinds[:2] == ["consent_required", "consent_resolved"], kinds
    req = events[0]
    assert {"id", "reason", "est_prompt_tokens", "provider", "model", "scopes"} <= set(req)
    assert req["provider"] == "openai" and req["model"] == "big-model"
    assert req["scopes"] == ["once"] and req["est_prompt_tokens"] > 50
    assert events[1] == {"type": "consent_resolved", "id": req["id"], "decision": "allow",
                         "scope": "once"}
    assert stats.prompt_tokens == 400 and stats.completion_tokens == 12
    assert turn.extras["cloud_used"] == {"provider": "openai", "model": "big-model"}
    # The once grant is history only: it never covers a later turn.
    assert list_cloud_consents(db, LOCAL) == []
    assert [r["scope"] for r in list_cloud_consents(db, LOCAL, True)] == ["once"]
    assert cloud_month_usage(db, LOCAL) == 412
    row = db.execute("SELECT selected_node, selected_model, status, user_id, conversation_id "
                     "FROM routing_events").fetchone()
    assert dict(row) == {"selected_node": "cloud", "selected_model": "big-model",
                         "status": "ok", "user_id": LOCAL, "conversation_id": "conv-1"}
    # A second model call of the same turn (an agent step) does not ask again.
    events.clear()
    text = asyncio.run(_stream_text(client, turn))
    assert REPLY_MARK in text and not [e for e in events if e["type"] == "consent_required"]


def test_deny_stays_local_with_original_prompt_and_is_not_asked_again():
    db = _db()
    cfg = _cfg()
    client, cloud, local = _setup(cfg, db)
    turn, events = _turn()
    secret_prompt = BIG + " password=hunter2secret"
    text = asyncio.run(_stream_text(client, turn, _messages(secret_prompt),
                                    answer=(LOCAL, "deny")))
    assert text == "local answer"
    assert cloud.requests == []
    # The local path is today's: the prompt is not minimised or redacted.
    assert local.bodies[0]["messages"][-1]["content"] == secret_prompt
    assert events[1]["decision"] == "deny"
    events.clear()
    assert asyncio.run(_stream_text(client, turn)) == "local answer"
    assert events == [] and CONSENT.pending() == []
    assert list_cloud_consents(db, LOCAL, True) == []
    assert cloud_month_usage(db, LOCAL) == 0


def test_preflight_asks_once_before_the_prompt_is_split():
    # Off: no question, no cloud, the caller splits as today.
    off = Config()
    client, cloud, local = _setup(off, _db())
    turn, events = _turn()
    token = set_current_turn(turn)
    try:
        assert asyncio.run(cloud_mod.cloud_preflight(off, _messages(BIG * 40))) is False
    finally:
        reset_current_turn(token)
    assert events == [] and cloud.requests == []

    # Allowed: the answer is remembered, so the model call that follows (a
    # bigger assembled prompt) goes to the cloud without a second prompt.
    cfg = _cfg()
    client, cloud, local = _setup(cfg, _db())
    turn, events = _turn()

    async def allow_then_call():
        token = set_current_turn(turn)
        try:
            ok, _ = await asyncio.gather(
                cloud_mod.cloud_preflight(cfg, [{"role": "user", "content": BIG}]),
                _answer(LOCAL, "allow"))
            text = "".join([p async for p in client.stream(_messages())])
            return ok, text
        finally:
            reset_current_turn(token)

    ok, text = asyncio.run(allow_then_call())
    assert ok is True and REPLY_MARK in text
    assert [e["type"] for e in events].count("consent_required") == 1
    assert local.bodies == [] and len(cloud.requests) == 1

    # Declined: False (split locally), and the turn is not asked again.
    client, cloud, local = _setup(cfg, _db())
    turn, events = _turn()

    async def deny_then_call():
        token = set_current_turn(turn)
        try:
            ok, _ = await asyncio.gather(
                cloud_mod.cloud_preflight(cfg, [{"role": "user", "content": BIG}]),
                _answer(LOCAL, "deny"))
            text = "".join([p async for p in client.stream(_messages())])
            return ok, text
        finally:
            reset_current_turn(token)

    ok, text = asyncio.run(deny_then_call())
    assert ok is False and text == "local answer"
    assert [e["type"] for e in events].count("consent_required") == 1
    assert cloud.requests == []

    # Unattended: never asks, stays local.
    client, cloud, local = _setup(cfg, _db())
    turn, events = _turn(interactive=False, emit=None)
    token = set_current_turn(turn)
    try:
        assert asyncio.run(cloud_mod.cloud_preflight(cfg, _messages())) is False
    finally:
        reset_current_turn(token)
    assert events == [] and CONSENT.pending() == [] and cloud.requests == []


def test_cancelled_turn_denies_the_pending_consent():
    cfg = _cfg()
    client, cloud, local = _setup(cfg, _db())

    async def run():
        turn, events = _turn(cancel=asyncio.Event())

        async def cancel_soon():
            while not CONSENT.pending():
                await asyncio.sleep(0.01)
            turn.cancel.set()

        token = set_current_turn(turn)
        try:
            offload, _ = await asyncio.gather(cloud_offload(cfg, _messages()), cancel_soon())
        finally:
            reset_current_turn(token)
        return offload, events

    offload, events = asyncio.run(run())
    assert offload is None and events[-1]["decision"] == "deny"
    assert cloud.requests == []


def test_unattended_turn_never_asks_and_stays_local():
    for mode in ("per_request", "per_user"):
        cfg = _cfg(cloud_consent=mode)
        db = _db()
        client, cloud, local = _setup(cfg, db)
        if mode == "per_user":
            # A conversation grant is not enough for an unattended run.
            grant_cloud_consent(db, LOCAL, "conversation", "conv-1")
        turn, events = _turn(interactive=False)
        assert asyncio.run(_stream_text(client, turn)) == "local answer"
        # No turn at all (an API client, a background job) is unattended too.
        assert asyncio.run(_stream_text(client, None)) == "local answer"
        assert events == [] and cloud.requests == [] and CONSENT.pending() == []


def test_per_user_conversation_and_always_consents():
    db = _db()
    cfg = _cfg(cloud_consent="per_user")
    client, cloud, local = _setup(cfg, db)
    turn, events = _turn()
    asyncio.run(_stream_text(client, turn, answer=(LOCAL, "allow", "conversation")))
    assert events[0]["scopes"] == ["once", "conversation", "always"]
    assert standing_cloud_consent(db, LOCAL, "conv-1") == "conversation"
    assert standing_cloud_consent(db, LOCAL, "conv-2") is None
    # A new turn in the same conversation goes without asking (a notice says so).
    turn2, events2 = _turn()
    assert REPLY_MARK in asyncio.run(_stream_text(client, turn2))
    assert [e["type"] for e in events2] == ["notice"]
    assert "saved cloud consent" in events2[0]["message"]
    # Another conversation asks again; answer always.
    turn3, events3 = _turn(conversation_id="conv-2")
    asyncio.run(_stream_text(client, turn3, answer=(LOCAL, "allow", "always")))
    assert events3[0]["type"] == "consent_required"
    assert standing_cloud_consent(db, LOCAL, "conv-9") == "always"
    # With an always consent, unattended runs may use the cloud.
    before = len(cloud.requests)
    turn4, events4 = _turn(interactive=False, emit=None)
    assert REPLY_MARK in asyncio.run(_stream_text(client, turn4))
    assert REPLY_MARK in asyncio.run(_stream_text(client, None))
    assert len(cloud.requests) == before + 2
    scopes = db.execute("SELECT consent_scope FROM cloud_usage ORDER BY id").fetchall()
    assert [r["consent_scope"] for r in scopes] == ["conversation", "conversation",
                                                    "always", "always", "always"]
    # Revoking stops it again.
    cloud_mod.revoke_cloud_consents(db, LOCAL)
    assert asyncio.run(_stream_text(client, None)) == "local answer"


def test_answer_with_a_scope_not_offered_is_rejected():
    cfg = _cfg()     # per_request: only "once"
    client, cloud, local = _setup(cfg, _db())

    async def run():
        turn, _events = _turn()

        async def bad_then_good():
            while not CONSENT.pending(LOCAL):
                await asyncio.sleep(0.01)
            cid = CONSENT.pending(LOCAL)[0]["id"]
            try:
                CONSENT.resolve(cid, LOCAL, "allow", "always")
            except ValueError:
                pass
            else:
                raise AssertionError("always accepted in per_request mode")
            # Someone else cannot answer it.
            assert CONSENT.resolve(cid, "intruder", "allow", "once") is False
            assert CONSENT.resolve(cid, LOCAL, "deny") is True

        token = set_current_turn(turn)
        try:
            out, _ = await asyncio.gather(cloud_offload(cfg, _messages()), bad_then_good())
        finally:
            reset_current_turn(token)
        return out

    assert asyncio.run(run()) is None


# --------------------------------------------------------------------------- #
# Data minimisation
# --------------------------------------------------------------------------- #
def test_minimisation_redacts_and_drops_audio():
    messages = [
        {"role": "system", "content": "token sk-abcdefghijklmnopqrstuvwxyz123456 here"},
        {"role": "user", "content": [
            {"type": "text", "text": "my password=hunter2secret ok"},
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
            {"type": "image_url", "image_url": {"url": "data:audio/wav;base64,AAAA"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR"}},
            {"type": "file", "file": {"name": "x.pdf"}},
        ]},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "f", "arguments": "{\"api_key\": \"sk-zzzzzzzzzzzzzzzzzzzzzzzzzz\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "done"},
    ]
    safe, dropped = cloud_minimise_messages(messages)
    blob = json.dumps(safe)
    assert dropped == 3
    assert "hunter2secret" not in blob and "sk-abcdefghijklmnop" not in blob
    assert "sk-zzzzzzzzzzzz" not in blob
    assert "input_audio" not in blob and "data:audio" not in blob
    assert "data:image/png;base64,iVBOR" in blob
    assert safe[3] == {"role": "tool", "content": "done", "tool_call_id": "c1"}
    # The original is untouched (the local path still sees it whole).
    assert messages[1]["content"][1]["type"] == "input_audio"


def test_what_reaches_the_provider_is_minimised():
    db = _db()
    cfg = _cfg(cloud_consent="per_user")
    client, cloud, local = _setup(cfg, db)
    grant_cloud_consent(db, LOCAL, "always")
    messages = [{"role": "user", "content": [
        {"type": "text", "text": BIG + " password=hunter2secret"},
        {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}}]}]
    assert REPLY_MARK in asyncio.run(_stream_text(client, None, messages))
    sent = cloud.requests[0].content.decode()
    assert "hunter2secret" not in sent and "UklGRg==" not in sent
    assert PROMPT_MARK in sent       # the prompt itself does go, by consent
    assert cloud.requests[0].headers["authorization"] == f"Bearer {API_KEY}"


# --------------------------------------------------------------------------- #
# Monthly cap
# --------------------------------------------------------------------------- #
def test_monthly_cap_keeps_prompts_local():
    db = _db()
    cfg = _cfg(cloud_consent="per_user", cloud_max_tokens_per_user_month=5000,
               max_tokens=256)
    client, cloud, local = _setup(cfg, db)
    grant_cloud_consent(db, LOCAL, "always")
    record_cloud_usage(db, user_id=LOCAL, provider="openai", model="m",
                       prompt_tokens=4500, completion_tokens=100, status="ok")
    turn, events = _turn()
    assert asyncio.run(_stream_text(client, turn)) == "local answer"
    assert cloud.requests == []
    assert [e["type"] for e in events] == ["notice"]
    assert "allowance" in events[0]["message"]
    # Another user's usage does not count against this one.
    assert cloud_month_usage(db, "someone-else") == 0
    # Under the cap it goes.
    db.execute("DELETE FROM cloud_usage")
    db.commit()
    turn2, _ = _turn()
    assert REPLY_MARK in asyncio.run(_stream_text(client, turn2))
    # Without a database the cap cannot be checked, so it fails closed.
    _setup(cfg, None, cloud)
    before = len(cloud.requests)
    assert asyncio.run(cloud_offload(cfg, _messages())) is None
    assert len(cloud.requests) == before


# --------------------------------------------------------------------------- #
# Provider payloads and streams
# --------------------------------------------------------------------------- #
def test_openai_payload_and_tool_call_stream():
    settings = {"provider": "openai", "model": "gpt-x"}
    body = cloud_openai_payload(settings, _messages("hi"), 300, 0.2, True, {
        "tools": [{"type": "function", "function": {"name": "calc"}}],
        "chat_template_kwargs": {"enable_thinking": False}, "xtc_probability": 0.5})
    assert body["model"] == "gpt-x" and body["max_tokens"] == 300 and body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["tools"][0]["function"]["name"] == "calc"
    assert "chat_template_kwargs" not in body and "xtc_probability" not in body

    class ToolCloud(FakeCloud):
        def handler(self, request):
            self.requests.append(request)
            return httpx.Response(200, content=_sse([
                {"choices": [{"index": 0, "delta": {"tool_calls": [{
                    "index": 0, "id": "call_1", "type": "function",
                    "function": {"name": "calc", "arguments": "{\"x\": "}}]},
                    "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {"tool_calls": [{
                    "index": 0, "function": {"arguments": "1}"}}]}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            ]), headers={"content-type": "text/event-stream"})

    db = _db()
    cfg = _cfg(cloud_consent="per_user")
    client, cloud, local = _setup(cfg, db, ToolCloud())
    grant_cloud_consent(db, LOCAL, "always")
    text = asyncio.run(_stream_text(client, None))
    assert json.loads(text.strip()) == {"tool": "calc", "args": {"x": 1}}
    # No usage reported: the estimate is still counted, so the cap holds.
    assert cloud_month_usage(db, LOCAL) > 50
    out, stats = asyncio.run(client.complete_with_stats(_messages()))
    assert json.loads(out.strip()) == {"tool": "calc", "args": {"x": 1}}
    assert local.bodies == []


def test_anthropic_payload_translation():
    settings = {"provider": "anthropic", "model": "claude-x"}
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "assistant", "content": "earlier note"},
        {"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR"}}]},
        {"role": "user", "content": "and this"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_9", "type": "function",
             "function": {"name": "calc", "arguments": "{\"x\": 3}"}}]},
        {"role": "tool", "tool_call_id": "call_9", "content": "3"},
        {"role": "tool", "tool_call_id": "unknown", "content": "orphan"},
    ]
    body = cloud_anthropic_payload(settings, messages, 500, 1.7, True, {
        "stop": "END", "tools": [{"type": "function", "function": {
            "name": "calc", "description": "math", "parameters": {"type": "object"}}}],
        "tool_choice": "required"})
    assert body["system"] == "Be brief." and body["model"] == "claude-x"
    assert body["temperature"] == 1.0 and body["stop_sequences"] == ["END"]
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "user"]
    assert body["messages"][0]["content"][0]["text"] == "(continuing the conversation)"
    merged = body["messages"][2]["content"]
    assert merged[0] == {"type": "text", "text": "look"}
    assert merged[1] == {"type": "image", "source": {"type": "base64",
                                                     "media_type": "image/png",
                                                     "data": "iVBOR"}}
    assert merged[2] == {"type": "text", "text": "and this"}
    assert body["messages"][3]["content"] == [{"type": "tool_use", "id": "call_9",
                                               "name": "calc", "input": {"x": 3}}]
    last = body["messages"][4]["content"]
    assert last[0] == {"type": "tool_result", "tool_use_id": "call_9", "content": "3"}
    assert last[1]["type"] == "text" and "orphan" in last[1]["text"]
    assert body["tools"] == [{"name": "calc", "description": "math",
                              "input_schema": {"type": "object"}}]
    assert body["tool_choice"] == {"type": "any"}


def test_anthropic_stream_through_the_model_client():
    db = _db()
    cfg = _cfg(cloud_provider="anthropic", cloud_base_url="https://anthropic.test",
               cloud_consent="per_user")
    client, cloud, local = _setup(cfg, db, FakeCloud("anthropic"))
    grant_cloud_consent(db, LOCAL, "always")
    stats = GenerationStats()
    text = asyncio.run(_stream_text(client, None, stats=stats))
    req = cloud.requests[0]
    assert str(req.url) == "https://anthropic.test/v1/messages"
    assert req.headers["x-api-key"] == API_KEY
    assert req.headers["anthropic-version"] == cloud_mod.CLOUD_ANTHROPIC_VERSION
    assert "authorization" not in req.headers
    assert cloud.bodies[0]["system"] == "You are helpful."
    assert text.startswith("<think>pondering</think>" + REPLY_MARK)
    assert json.loads(text.splitlines()[-1]) == {"tool": "calc", "args": {"x": 2}}
    assert stats.prompt_tokens == 321 and stats.completion_tokens == 42
    assert stats.finish_reason == "tool_calls"
    assert cloud_month_usage(db, LOCAL) == 363


def test_cloud_error_before_output_falls_back_to_local():
    db = _db()
    cfg = _cfg()
    client, cloud, local = _setup(cfg, db, FakeCloud(status=529))
    turn, events = _turn()
    handler, restore = _capture()
    try:
        text = asyncio.run(_stream_text(client, turn, answer=(LOCAL, "allow")))
    finally:
        restore()
    assert text == "local answer" and len(cloud.requests) == 1 and len(local.bodies) == 1
    notices = [e for e in events if e["type"] == "notice"]
    assert notices and "529" in notices[0]["message"]
    # The provider's message (which may quote the prompt) is never surfaced.
    assert PROMPT_MARK not in json.dumps(events) and PROMPT_MARK not in handler.dump()
    row = db.execute("SELECT status FROM cloud_usage").fetchone()
    assert row["status"] == "failed"
    assert "cloud_used" not in turn.extras


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def test_logging_is_metadata_only():
    db = _db()
    cfg = _cfg()
    client, cloud, local = _setup(cfg, db)
    turn, _events = _turn()
    handler, restore = _capture()
    try:
        text = asyncio.run(_stream_text(client, turn, answer=(LOCAL, "allow")))
    finally:
        restore()
    assert REPLY_MARK in text
    fallback = [r for r in handler.records if getattr(r, "event", None) == "cloud.fallback"]
    assert len(fallback) == 1
    rec = fallback[0]
    for key, value in {"user_id": LOCAL, "conversation_id": "conv-1",
                       "provider": "openai", "model": "big-model", "consent_scope": "once",
                       "prompt_tokens": 400, "completion_tokens": 12, "status": "ok"}.items():
        assert getattr(rec, key) == value, key
    assert isinstance(rec.latency_ms, int)
    dump = handler.dump()
    assert "cloud.consent_requested" in dump and "cloud.consent_decided" in dump
    assert PROMPT_MARK not in dump and REPLY_MARK not in dump and "lorem" not in dump
    assert API_KEY not in dump and "thinking hard" not in dump
    usage_cols = {r["name"] for r in db.execute("PRAGMA table_info(cloud_usage)")}
    assert not usage_cols & {"prompt", "content", "reply", "messages", "text"}


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
def _build_app(**over):
    base = dict(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                allow_test_user=False, node_token="secret-node-token",
                skills_dir=tempfile.mkdtemp(prefix="llm-test-cloud-skills-"))
    base.update(over)
    cfg = Config(**base)
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)  # never started
    rm = RetrainManager(db, mm, cfg)
    app = create_app(cfg, db, mm, rm, ToolRegistry(cfg, db))
    app.state.model_manager = mm
    return app, cfg, db


def _login(c, username, password):
    r = c.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text


def test_routes_consents_usage_and_owner_checks():
    app, cfg, db = _build_app(cloud_fallback_enabled=True, cloud_model="big-model",
                              cloud_api_key=API_KEY, cloud_consent="per_user",
                              cloud_max_tokens_per_user_month=100000)
    db.create_user("bob", password_hash=hash_password("bobpw12345"), role="user")
    with TestClient(app) as c:
        _login(c, "admin", "adminpw123")
        admin_id = db.execute("SELECT id FROM users WHERE username = 'admin'").fetchone()["id"]
        bob_id = db.execute("SELECT id FROM users WHERE username = 'bob'").fetchone()["id"]
        grant_cloud_consent(db, admin_id, "always")
        grant_cloud_consent(db, admin_id, "conversation", "c-1")
        record_cloud_usage(db, user_id=admin_id, provider="openai", model="big-model",
                           prompt_tokens=700, completion_tokens=30, status="ok")
        mine = c.get("/api/cloud/consents").json()
        assert mine["enabled"] and mine["available"] and mine["allowed"]
        assert mine["mode"] == "per_user" and mine["used_tokens"] == 730
        assert mine["cap_tokens"] == 100000 and "api_key" not in json.dumps(mine)
        assert API_KEY not in json.dumps(mine)
        assert sorted(r["scope"] for r in mine["consents"]) == ["always", "conversation"]
        conv_row = [r for r in mine["consents"] if r["scope"] == "conversation"][0]
        usage = c.get("/api/cloud/usage").json()
        assert usage["totals"]["prompt_tokens"] == 700
        assert usage["users"][0]["username"] == "admin"
        assert c.get("/api/cloud/usage", params={"month": "2026-13"}).status_code == 400

        _login(c, "bob", "bobpw12345")
        assert c.get("/api/cloud/usage").status_code == 403
        theirs = c.get("/api/cloud/consents").json()
        assert theirs["consents"] == [] and theirs["used_tokens"] == 0
        assert theirs["allowed"] is False
        # Bob cannot revoke the admin's grant (404, not a silent success).
        assert c.delete(f"/api/cloud/consents/{conv_row['id']}").status_code == 404
        assert c.delete("/api/cloud/consents").json() == {"revoked": 0}
        assert standing_cloud_consent(db, admin_id, "c-1") == "conversation"
        assert bob_id

        _login(c, "admin", "adminpw123")
        assert c.delete(f"/api/cloud/consents/{conv_row['id']}").json() == {"revoked": 1}
        assert c.delete("/api/cloud/consents").json() == {"revoked": 1}
        assert c.get("/api/cloud/consents").json()["consents"] == []


def test_consent_route_resolves_a_waiting_turn():
    app, cfg, db = _build_app(cloud_fallback_enabled=True, cloud_model="big-model",
                              cloud_base_url="https://cloud.test/v1", cloud_api_key=API_KEY,
                              cloud_min_prompt_tokens=50, cloud_min_prefill_s=0)
    db.create_user("bob", password_hash=hash_password("bobpw12345"), role="user")
    cloud = FakeCloud()
    with TestClient(app) as c:
        _login(c, "admin", "adminpw123")
        admin_id = db.execute("SELECT id FROM users WHERE username = 'admin'").fetchone()["id"]
        CONSENT.clear()
        cloud_mod._CLOUD_HOOKS["transport"] = httpx.MockTransport(cloud.handler)
        events: list[dict] = []
        result: dict = {}

        def run():
            async def go():
                # A ChatTurn binds the loop it is created on.
                chat_turn = app.state.turns.create(admin_id, "conv-r")
                result["turn_id"] = chat_turn.turn_id
                result["other_turn_id"] = app.state.turns.create(admin_id, "conv-o").turn_id
                turn = TurnContext(turn_id=chat_turn.turn_id, user_id=admin_id,
                                   conversation_id="conv-r", role="admin",
                                   interactive=True, emit=events.append)
                token = set_current_turn(turn)
                try:
                    offload = await cloud_offload(cfg, _messages())
                    result["text"] = await offload.complete(_messages()) if offload else None
                finally:
                    reset_current_turn(token)
            asyncio.run(go())

        worker = threading.Thread(target=run)
        worker.start()
        deadline = time.time() + 10
        while not CONSENT.pending(admin_id) and time.time() < deadline:
            time.sleep(0.02)
        cid = CONSENT.pending(admin_id)[0]["id"]
        url = f"/api/chat/turns/{result['turn_id']}/consent/{cid}"
        # A scope this request did not offer is a 400; another user gets 404.
        assert c.post(url, json={"decision": "allow", "scope": "always"}).status_code == 400
        _login(c, "bob", "bobpw12345")
        assert c.post(url, json={"decision": "allow"}).status_code == 404
        _login(c, "admin", "adminpw123")
        # The owner's OTHER turn's URL does not answer this turn's consent.
        wrong = f"/api/chat/turns/{result['other_turn_id']}/consent/{cid}"
        assert c.post(wrong, json={"decision": "allow"}).status_code == 404
        r = c.post(url, json={"decision": "allow", "scope": "once"})
        assert r.status_code == 200, r.text
        assert r.json() == {"resolved": True, "decision": "allow", "scope": "once"}
        worker.join(10)
        assert not worker.is_alive()
        assert REPLY_MARK in (result.get("text") or "")
        assert [e["type"] for e in events][:2] == ["consent_required", "consent_resolved"]
        # Answered once: a second answer is 404.
        assert c.post(url, json={"decision": "deny"}).status_code == 404
        cloud_mod._CLOUD_HOOKS["transport"] = None


def _run_standalone():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"FAIL  {name}: {exc}")
            failed += 1
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
