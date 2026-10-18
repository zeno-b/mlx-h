"""Tests for the multi-node cluster: N peers, protocol 2, pairing, node security.

The two-node tests run two real apps in one process: the primary's ModelClient
reaches the peer's /api/node/generate through httpx.ASGITransport (installed
via nodesec.NODE_TEST_HOOKS, so the production peer transport, policy checks
and E2E sealing all run), and both apps' "local mlx server" is an
httpx.MockTransport told apart by model_port.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import stat
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_llm.config as config_mod  # noqa: E402

config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-cluster-"))

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from local_llm.config import Config  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.training import RetrainManager  # noqa: E402
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.core import ADAPTER_DIR  # noqa: E402
from local_llm.api import create_app  # noqa: E402
from local_llm.model_client import ModelClient  # noqa: E402
from local_llm.llm import GenerationStats  # noqa: E402
from local_llm.obslog import redact_text  # noqa: E402
from local_llm.auth import Auth  # noqa: E402
from local_llm import nodesec  # noqa: E402
from local_llm.cluster import (  # noqa: E402
    ClusterRouter, HealthMonitor, NodeRegistry, DEGRADED, HEALTHY, OVERLOADED,
    STARTING, UNAVAILABLE, cluster_node_specs, read_paired_nodes,
    remote_health_state, cluster_paired_path)

TOKEN = "secret-node-token"
PRIMARY_PORT, PEER_PORT = 8191, 8192
STUDIO_URL = "http://studio.local:8000"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def build_app(**overrides):
    base = dict(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                allow_test_user=False, node_token=TOKEN,
                skills_dir=tempfile.mkdtemp(prefix="llm-test-cluster-skills-"))
    base.update(overrides)
    cfg = Config(**base)
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)  # never started
    rm = RetrainManager(db, mm, cfg)
    app = create_app(cfg, db, mm, rm, ToolRegistry(cfg, db))
    app.state.model_manager = mm
    return app, cfg, db


def login(client, username="admin", password="adminpw123"):
    return client.post("/api/auth/login", json={"username": username, "password": password})


class _FakeAEAD:
    """Stand-in for ChaCha20Poly1305 (cryptography is not installed here):
    a SHA-256 keystream plus an HMAC tag over nonce, AAD and ciphertext, so a
    tampered AAD, body or key fails exactly like InvalidTag would."""

    def __init__(self, key: bytes):
        self.key = key

    def _ks(self, nonce: bytes, n: int) -> bytes:
        out, i = b"", 0
        while len(out) < n:
            out += hashlib.sha256(self.key + nonce + i.to_bytes(4, "big")).digest()
            i += 1
        return out[:n]

    def _tag(self, nonce, aad, ct):
        return hmac.new(self.key, nonce + (aad or b"") + ct, "sha256").digest()[:16]

    def encrypt(self, nonce, data, aad):
        ct = bytes(a ^ b for a, b in zip(data, self._ks(nonce, len(data))))
        return ct + self._tag(nonce, aad, ct)

    def decrypt(self, nonce, data, aad):
        ct, tag = data[:-16], data[-16:]
        if not hmac.compare_digest(tag, self._tag(nonce, aad, ct)):
            raise ValueError("InvalidTag")
        return bytes(a ^ b for a, b in zip(ct, self._ks(nonce, len(ct))))


class _Mlx:
    """Fake mlx_lm.server for every app in the process, keyed by model_port."""

    def __init__(self):
        self.calls: list[int] = []
        self.fail_ports: set[int] = set()

    def handler(self, request: httpx.Request) -> httpx.Response:
        port = request.url.port
        self.calls.append(port)
        if port in self.fail_ports:
            return httpx.Response(500, text="metal command buffer failed")
        word = "studio" if port == PEER_PORT else "mini"
        body = json.loads(request.content or b"{}")
        if body.get("stream"):
            frames = [{"choices": [{"delta": {"reasoning": "think "}}]},
                      {"choices": [{"delta": {"content": "hello "}}]},
                      {"choices": [{"delta": {"content": word}, "finish_reason": "length"}]},
                      {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}]
            text = "".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  text=text)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": f"hello {word}", "reasoning": "think"},
                         "finish_reason": "length"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3}})


class _Cluster:
    """A primary ModelClient + a real peer app, wired in-process."""

    def __init__(self, e2e_key: str = "", peer_e2e_key: str | None = None):
        self.mlx = _Mlx()
        self.peer_app, self.peer_cfg, _ = build_app(
            model_port=PEER_PORT, node_name="studio", node_role="secondary",
            node_e2e_key=e2e_key if peer_e2e_key is None else peer_e2e_key)
        self.peer_app.state.model_manager.status = "ready"
        self._saved = (ModelClient.__dict__["_client"], dict(nodesec.NODE_TEST_HOOKS))
        if e2e_key or peer_e2e_key:
            # Before the registry is built: it applies the E2E policy per peer.
            nodesec.NODE_TEST_HOOKS["aead"] = _FakeAEAD
        self.cfg = Config(node_token=TOKEN, model_port=PRIMARY_PORT, node_e2e_key=e2e_key,
                          cluster_nodes=json.dumps([{"name": "studio", "url": STUDIO_URL,
                                                     "ram_gb": 64}]))
        self.db = Database(Path(tempfile.mkdtemp()) / "primary.db")
        self.registry = NodeRegistry(self.cfg)
        self.router = ClusterRouter(self.cfg, self.registry, self.db)
        self.client = ModelClient(self.cfg, cluster=self.router)
        self.local = self.registry.local_node()
        self.studio = self.registry.by_name("studio")

    def __enter__(self):
        local = httpx.AsyncClient(transport=httpx.MockTransport(self.mlx.handler))
        ModelClient._client = classmethod(lambda cls: local)
        ModelClient._peer_http = {}
        nodesec.NODE_TEST_HOOKS["async_transport"] = (
            lambda _cfg: httpx.ASGITransport(app=self.peer_app))
        # Studio up, primary still loading: the router prefers the studio and
        # keeps the primary as the failover target.
        self.registry.update("studio", state=HEALTHY)
        self.registry.update(self.local.name, state=STARTING)
        return self

    def __exit__(self, *exc):
        ModelClient._client = self._saved[0]
        ModelClient._peer_http = {}
        nodesec.NODE_TEST_HOOKS.clear()
        nodesec.NODE_TEST_HOOKS.update(self._saved[1])
        return False

    def events(self):
        return list(reversed(self.db.list_routing_events(limit=50)))


async def _collect(agen):
    return [event async for event in agen]


MSGS = [{"role": "user", "content": "hi"}]


# --------------------------------------------------------------------------- #
# The crash: ModelClient brackets every dispatch with router.begin/end
# --------------------------------------------------------------------------- #
def test_router_delegates_begin_end_to_registry():
    cfg = Config(node_token=TOKEN, studio_node_url=STUDIO_URL)
    registry = NodeRegistry(cfg)
    router = ClusterRouter(cfg, registry)
    studio = registry.by_name("mac-studio")
    router.begin(studio)
    assert studio.active == 1
    router.end(studio, True, 12.5)
    assert studio.active == 0 and studio.last_latency_ms == 12.5


def test_two_node_non_stream_routes_to_peer():
    with _Cluster() as cl:
        text, stats = asyncio.run(cl.client.complete_with_stats(MSGS, max_tokens=32))
        assert text == "hello studio"
        # The peer forwards the real finish reason instead of a constant "stop".
        assert stats.finish_reason == "length"
        assert PEER_PORT in cl.mlx.calls and PRIMARY_PORT not in cl.mlx.calls
        events = cl.events()
        assert [(e["selected_node"], e["status"]) for e in events] == [("studio", "ok")]
        assert cl.studio.active == 0 and cl.local.active == 0


def test_two_node_stream_forwards_reasoning_and_finish_reason():
    with _Cluster() as cl:
        stats = GenerationStats()
        events = asyncio.run(_collect(cl.client.stream_events(MSGS, max_tokens=32,
                                                              stats=stats)))
        kinds = [(e.get("type"), e.get("text") or e.get("reason")) for e in events]
        assert ("reasoning", "think ") in kinds
        content = "".join(e.get("text", "") for e in events if e.get("type") == "content")
        assert content == "hello studio"
        assert ("finish", "length") in kinds
        assert [(e["selected_node"], e["status"]) for e in cl.events()] == [("studio", "ok")]
        assert cl.studio.active == 0


def test_stream_fails_over_before_first_token_on_peer_error_frame():
    with _Cluster() as cl:
        cl.mlx.fail_ports.add(PEER_PORT)
        events = asyncio.run(_collect(cl.client.stream_events(MSGS, max_tokens=32)))
        content = "".join(e.get("text", "") for e in events if e.get("type") == "content")
        # The peer's failure is an error FRAME, never answer text.
        assert content == "hello mini"
        assert "node error" not in content
        assert [(e["selected_node"], e["status"]) for e in cl.events()] == [
            ("studio", "failed"), (cl.local.name, "failover_ok")]
        assert cl.studio.consecutive_failures == 1
        assert cl.studio.active == 0 and cl.local.active == 0


def test_non_stream_fails_over_on_peer_502():
    with _Cluster() as cl:
        cl.mlx.fail_ports.add(PEER_PORT)
        text, _ = asyncio.run(cl.client.complete_with_stats(MSGS, max_tokens=32))
        assert text == "hello mini"
        # One call to the peer's model: no max_tokens-less retry against a peer.
        assert cl.mlx.calls.count(PEER_PORT) == 1
        assert [e["status"] for e in cl.events()] == ["failed", "failover_ok"]


def test_peer_generate_error_is_502_not_answer_text():
    app, _, _ = build_app(model_port=PEER_PORT)
    app.state.model_manager.status = "ready"
    mlx = _Mlx()
    mlx.fail_ports.add(PEER_PORT)
    saved = ModelClient.__dict__["_client"]
    local = httpx.AsyncClient(transport=httpx.MockTransport(mlx.handler))
    ModelClient._client = classmethod(lambda cls: local)
    try:
        with TestClient(app) as c:
            r = c.post("/api/node/generate", json={"messages": MSGS},
                       headers={"Authorization": f"Bearer {TOKEN}"})
            assert r.status_code == 502
            assert r.headers.get("x-node-protocol") == "2"
            assert "500" in r.json()["error"]
            r = c.post("/api/node/generate", json={"messages": MSGS, "stream": True},
                       headers={"Authorization": f"Bearer {TOKEN}"})
            frames = [json.loads(line[6:]) for line in r.text.splitlines()
                      if line.startswith("data: ") and line[6:] != "[DONE]"]
            assert frames[-1]["error"]["type"] == "node_error"
            assert not any("[node error" in json.dumps(f) for f in frames)
            # The admission slot was given back on both paths.
            assert app.state.ctx.generation_gate.running == 0
    finally:
        ModelClient._client = saved


def test_peer_refused_by_policy_is_never_dispatched():
    # E2E key without the cryptography package: no plaintext fallback.
    cfg = Config(node_token=TOKEN, cluster_nodes=STUDIO_URL,
                 node_e2e_key=nodesec.node_generate_e2e_key())
    registry = NodeRegistry(cfg)
    peer = [n for n in registry.nodes if not n.is_local][0]
    if not nodesec.node_crypto_available():
        assert "cryptography" in registry.refusal(peer)
        assert peer.state == UNAVAILABLE
        targets = ModelClient(cfg, cluster=ClusterRouter(cfg, registry))._targets()
        assert all(node is None or node.is_local for node, *_ in targets)
    # Plain http:// to a public address is refused unless explicitly allowed.
    public = Config(node_token=TOKEN, cluster_nodes="http://8.8.8.8:8000")
    reg = NodeRegistry(public)
    assert "plain http" in reg.refusal(reg.nodes[1])
    allowed = Config(node_token=TOKEN, cluster_nodes="http://8.8.8.8:8000",
                     node_allow_plaintext=True)
    reg = NodeRegistry(allowed)
    assert reg.refusal(reg.nodes[1]) == ""


# --------------------------------------------------------------------------- #
# E2E sealing over the real peer route
# --------------------------------------------------------------------------- #
def test_e2e_sealed_two_node_stream_and_non_stream():
    key = nodesec.node_generate_e2e_key()
    with _Cluster(e2e_key=key) as cl:
        text, _ = asyncio.run(cl.client.complete_with_stats(MSGS, max_tokens=32))
        assert text == "hello studio"
        events = asyncio.run(_collect(cl.client.stream_events(MSGS, max_tokens=32)))
        content = "".join(e.get("text", "") for e in events if e.get("type") == "content")
        assert content == "hello studio"
        assert [e["status"] for e in cl.events()] == ["ok", "ok"]


def test_e2e_key_mismatch_fails_over_and_never_downgrades():
    with _Cluster(e2e_key=nodesec.node_generate_e2e_key(),
                  peer_e2e_key=nodesec.node_generate_e2e_key()) as cl:
        text, _ = asyncio.run(cl.client.complete_with_stats(MSGS, max_tokens=32))
        assert text == "hello mini"
        assert PEER_PORT not in cl.mlx.calls      # the peer never ran it
    # A sealed primary against a peer without a key: refused, not read in clear.
    with _Cluster(e2e_key=nodesec.node_generate_e2e_key(), peer_e2e_key="") as cl:
        text, _ = asyncio.run(cl.client.complete_with_stats(MSGS, max_tokens=32))
        assert text == "hello mini" and PEER_PORT not in cl.mlx.calls


def test_e2e_seal_open_rejects_tamper_replay_and_skew():
    nodesec.NODE_TEST_HOOKS["aead"] = _FakeAEAD
    try:
        e2e = nodesec.NodeE2E(os.urandom(32))
        header, nonce, body = e2e.seal_request("POST", "/api/node/generate", b'{"a": 1}')
        envelope = json.loads(body)
        assert b'"a"' not in body                       # body is not plaintext
        got_nonce, plain = e2e.open_request("POST", "/api/node/generate", header, envelope)
        assert got_nonce == nonce and json.loads(plain) == {"a": 1}
        for call in (
                lambda: e2e.open_request("POST", "/api/node/generate", header, envelope),
                lambda: e2e.open_request("POST", "/api/node/health", *e2e.seal_request(
                    "POST", "/api/node/generate", b"{}")[::2]),
                lambda: e2e.open_request("POST", "/api/node/generate", *e2e.seal_request(
                    "POST", "/api/node/generate", b"{}", now=time.time() - 120)[::2])):
            try:
                call()
            except nodesec.NodeSecurityError:
                pass
            else:
                raise AssertionError("sealed request should have been refused")
        frame = e2e.seal_frame("/api/node/generate", nonce, 0, b"data")
        assert e2e.open_frame("/api/node/generate", nonce, 0, frame) == b"data"
        try:
            e2e.open_frame("/api/node/generate", nonce, 1, frame)   # reordered frame
        except nodesec.NodeSecurityError:
            pass
        else:
            raise AssertionError("a frame opened at the wrong sequence number")
    finally:
        nodesec.NODE_TEST_HOOKS["aead"] = None


# --------------------------------------------------------------------------- #
# N nodes, protocol 2 health, routing inputs
# --------------------------------------------------------------------------- #
def test_cluster_nodes_parsing_and_studio_alias():
    cfg = Config(cluster_nodes=json.dumps([
        {"name": "Studio A", "url": "https://a.example:8000", "ram_gb": 96},
        "https://b.example:8000", {"name": "no-url"}]),
        studio_node_url="https://studio.example:8000/")
    specs = cluster_node_specs(cfg)
    assert [s["url"] for s in specs] == ["https://a.example:8000", "https://b.example:8000",
                                         "https://studio.example:8000/"]
    registry = NodeRegistry(cfg)
    names = [n.name for n in registry.nodes if not n.is_local]
    assert names == ["studio-a", "b", "mac-studio"]
    a, b, studio = (registry.by_name(n) for n in names)
    assert "large_model" in a.capabilities          # from the configured RAM
    assert "large_model" not in b.capabilities      # unknown RAM: base set
    assert "large_model" in studio.capabilities     # STUDIO_NODE_URL promise kept
    # Comma form, and a duplicate STUDIO_NODE_URL is not added twice.
    cfg = Config(cluster_nodes="https://x.example:1, https://y.example:2",
                 studio_node_url="https://x.example:1")
    assert len(NodeRegistry(cfg).nodes) == 3
    assert NodeRegistry(Config(cluster_nodes="[not json")).multi_node is False


def test_remote_status_mapping_unknown_is_degraded():
    assert remote_health_state({"state": "ready"})[0] == HEALTHY
    assert remote_health_state({"state": "loading"})[0] == STARTING
    assert remote_health_state({"state": "overloaded"})[0] == OVERLOADED
    assert remote_health_state({"state": "error: metal crashed"})[0] == DEGRADED
    assert remote_health_state({})[0] == DEGRADED


def test_probe_remote_reads_protocol_2_fields_and_ram_capabilities():
    answers = {"body": {"state": "ready", "model": "m", "cpu_pct": 10, "mem_pct": 20,
                        "protocol": 2, "ram_gb": 128, "free_ram_gb": 90.5,
                        "machine": "Mac Studio", "cores": 24, "load_ratio": 0.2,
                        "backend": "mlx-lm", "queue": 3, "shard_capable": True}}
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        seen["proto"] = request.headers.get("x-node-protocol")
        return httpx.Response(200, json=answers["body"], headers={"X-Node-Protocol": "2"})

    cfg = Config(node_token="new-token,old-token", cluster_nodes="http://10.0.0.5:8000")
    registry = NodeRegistry(cfg)
    node = registry.nodes[1]
    assert "large_model" not in node.capabilities
    nodesec.NODE_TEST_HOOKS["sync_transport"] = lambda _cfg: httpx.MockTransport(handler)
    try:
        HealthMonitor(cfg, registry)._probe_remote(node)
        assert seen == {"auth": "Bearer new-token", "proto": "2"}
        assert node.state == HEALTHY and node.protocol == 2
        assert (node.ram_gb, node.free_ram_gb, node.queue, node.cores) == (128, 90.5, 3, 24)
        assert node.machine == "Mac Studio" and node.shard_capable is True
        assert "large_model" in node.capabilities
        answers["body"] = {"state": "error: boom"}       # a protocol-1 style answer
        HealthMonitor(cfg, registry)._probe_remote(node)
        assert node.state == DEGRADED and "boom" in node.detail
    finally:
        nodesec.NODE_TEST_HOOKS["sync_transport"] = None


def test_route_queue_depth_and_heartbeat_timeout_take_effect():
    cfg = Config(node_token=TOKEN, cluster_nodes=STUDIO_URL, route_queue_depth=2,
                 heartbeat_timeout=30)
    registry = NodeRegistry(cfg)
    router = ClusterRouter(cfg, registry)
    peer = registry.nodes[1]
    registry.update(peer.name, state=HEALTHY, queue=2)
    assert router._is_overloaded(peer)[0] is True
    registry.update(peer.name, queue=1)
    assert router._is_overloaded(peer)[0] is False
    registry.update(peer.name, last_ok_heartbeat=time.time() - 60)
    req = router.classify(model=cfg.model)
    assert router._eligible(peer, req) is False
    registry.update(peer.name, last_ok_heartbeat=time.time())
    assert router._eligible(peer, req) is True


def test_node_health_reports_protocol_2_fields():
    app, _, _ = build_app()
    with TestClient(app) as c:
        r = c.get("/api/node/health", headers={"Authorization": f"Bearer {TOKEN}"})
        assert r.status_code == 200
        assert r.headers["x-node-protocol"] == "2"
        body = r.json()
        for key in ("ram_gb", "free_ram_gb", "machine", "cores", "load_ratio", "backend",
                    "model", "queue", "shard_capable", "state"):
            assert key in body, key


def test_node_token_comma_list_rotation():
    auth = Auth(Config(node_token="new-one, old-one"), None)

    class _Req:
        def __init__(self, token):
            self.headers = {"authorization": f"Bearer {token}"}

    assert auth.check_node_token(_Req("new-one")) and auth.check_node_token(_Req("old-one"))
    assert not auth.check_node_token(_Req("new-one, old-one"))
    assert not auth.check_node_token(_Req(""))


def test_anthropic_key_shape_is_redacted():
    key = "sk-ant-api03-" + "A1b2C3d4E5f6G7h8I9j0" * 3
    assert key not in redact_text(f"upstream said {key}")


# --------------------------------------------------------------------------- #
# Pairing bootstrap
# --------------------------------------------------------------------------- #
def test_pairing_issue_join_single_use_and_persist():
    config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-cluster-pair-"))
    app, cfg, _ = build_app()
    with TestClient(app) as c:
        assert c.post("/api/cluster/pairing", json={}).status_code == 401
        login(c)
        issued = c.post("/api/cluster/pairing", json={"ttl_s": 300}).json()
        assert issued["expires_at"] > time.time() and len(issued["code"]) == 11
        c.post("/api/auth/logout")
        c.cookies.clear()
        join = {"code": issued["code"], "name": "Studio 2", "url": STUDIO_URL, "ram_gb": 64}
        r = c.post("/api/cluster/join", json=join)
        assert r.status_code == 200, r.text
        token = r.json()["node_token"]
        assert r.json()["name"] == "studio-2" and len(token) >= 32
        assert c.post("/api/cluster/join", json=join).status_code == 403   # single use
        bad = dict(join, url="http://8.8.8.8:8000")
        assert c.post("/api/cluster/join", json=bad).status_code == 400
    path = cluster_paired_path(cfg)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    stored = read_paired_nodes(cfg)
    assert [(n["name"], n["token"]) for n in stored] == [("studio-2", token)]
    # The plaintext code is never stored: the book holds hashes only.
    assert issued["code"] not in json.dumps(app.state.cluster_pairing._codes)
    # Read at startup (heartbeat start) when CLUSTER_NODES is unset ...
    registry = NodeRegistry(Config(node_token=TOKEN))
    assert registry.load_paired() == 1
    node = registry.by_name("studio-2")
    assert node.token == token and "large_model" in node.capabilities
    assert nodesec.node_auth_headers(cfg, node)["Authorization"] == f"Bearer {token}"
    # ... and ignored when CLUSTER_NODES names the nodes explicitly.
    registry = NodeRegistry(Config(node_token=TOKEN, cluster_nodes="https://z.example:1"))
    assert registry.load_paired() == 0


def test_join_is_rate_limited():
    config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-cluster-rl-"))
    app, _, _ = build_app()
    with TestClient(app) as c:
        statuses = [c.post("/api/cluster/join", json={"code": "WRONG-CODE1", "name": "x",
                                                       "url": STUDIO_URL}).status_code
                    for _ in range(7)]
    assert statuses[:5] == [403] * 5 and statuses[5:] == [429, 429]


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
