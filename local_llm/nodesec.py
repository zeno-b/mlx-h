"""Security for traffic between nodes: TLS, the plaintext policy, optional E2E sealing.

The primary reaches a peer node over plain HTTP with one shared bearer token
(`POST /api/node/generate`, `GET /api/node/health`). That was fine on a desk
LAN, and wrong the moment a node sits on another network: the token and every
prompt crossed the wire in the clear, and nothing stopped STUDIO_NODE_URL from
pointing at a public address. This module is the one place that decides how a
peer is spoken to, so the router's probe, the generation client and the CLI
all apply the same rules:

* TLS options (NODE_CA_FILE to pin a private CA, NODE_CLIENT_CERT and
  NODE_CLIENT_KEY for mutual TLS) on every peer httpx client.
* Plain http:// only to loopback, private (RFC 1918 / ULA / link-local),
  tailnet (100.64.0.0/10, fd7a:115c:a1e0::/48, *.ts.net) or LAN-suffixed
  hosts; anything else is refused unless NODE_ALLOW_PLAINTEXT=1.
* Optional end-to-end payload sealing (NODE_E2E_KEY, 32 bytes) with
  ChaCha20-Poly1305 from the optional `cryptography` package, for links that
  pass through something you do not trust to terminate TLS (a tunnel, a
  proxy). Random 96-bit nonces; the request AAD binds method, path, timestamp
  and nonce; a 60 s clock-skew window plus a replay cache stops a captured
  request from being re-sent; every SSE frame of a streamed answer is sealed
  separately with its sequence number in the AAD, so frames cannot be
  dropped, reordered or spliced between responses. When NODE_E2E_KEY is set
  and `cryptography` is missing, node traffic is REFUSED with a clear message:
  there is no plaintext fallback, because a silent downgrade is exactly the
  failure the key exists to prevent.
* NODE_TOKEN may be a comma list so a token can be rotated without downtime
  (accept new,old; switch the sender; drop old).

The sealing lives in an httpx transport wrapper, so ModelClient's streaming
and non-streaming loops parse plaintext exactly as before and never see an
envelope. The same wrapper turns a peer's in-band error frame into an
exception (NodePeerError) instead of letting it pass as answer text.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import importlib.util
import ipaddress
import json
import os
import secrets
import ssl
import threading
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403


# Version 2 adds the hardware/queue fields to /api/node/health, error frames,
# real finish_reason and forwarded reasoning on /api/node/generate.
NODE_PROTOCOL = 2
NODE_PROTOCOL_HEADER = "X-Node-Protocol"
NODE_E2E_HEADER = "X-Node-E2E"               # request: v1;ts=<int>;n=<nonce>[;t=<tag>]
NODE_E2E_SEALED_HEADER = "X-Node-E2E-Sealed"  # response: v1
NODE_E2E_SKEW_S = 60.0
_NS_NONCE_BYTES = 12
_NS_KEY_BYTES = 32
_NS_VERSION = "v1"
_NS_REPLAY_MAX = 50_000

_nodesec_log = get_logger("nodesec")

# Test seams, shared by reference across the star imports: "aead" is a
# callable(key) -> object with encrypt/decrypt(nonce, data, aad) used instead
# of cryptography's ChaCha20Poly1305; "async_transport" / "sync_transport" are
# callables(config) -> the INNER httpx transport (an httpx.MockTransport or
# ASGITransport in tests). Production code never sets them.
NODE_TEST_HOOKS: dict[str, Any] = {"aead": None, "async_transport": None,
                                   "sync_transport": None}


class NodeSecurityError(RuntimeError):
    """Node traffic refused for a security reason. Never retried silently."""


class NodePeerError(RuntimeError):
    """The peer node reported an error in-band (an SSE error frame)."""


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #
def node_tokens(config: Any) -> list[str]:
    """Every accepted inter-node token. NODE_TOKEN is a comma list for rotation."""
    raw = str(getattr(config, "node_token", "") or "")
    return [t.strip() for t in raw.split(",") if t.strip()]


def node_send_token(config: Any, node: Any = None) -> str:
    """The token to SEND: a paired node's own token, else the first NODE_TOKEN."""
    own = str(getattr(node, "token", "") or "") if node is not None else ""
    if own:
        return own
    tokens = node_tokens(config)
    return tokens[0] if tokens else ""


def node_auth_headers(config: Any, node: Any = None) -> dict[str, str]:
    headers = {NODE_PROTOCOL_HEADER: str(NODE_PROTOCOL)}
    token = node_send_token(config, node)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


# --------------------------------------------------------------------------- #
# Plaintext policy
# --------------------------------------------------------------------------- #
_NS_TAILNET_NETS = (ipaddress.ip_network("100.64.0.0/10"),
                    ipaddress.ip_network("fd7a:115c:a1e0::/48"))
# Name suffixes that only resolve on a local network or a tailnet. A *.ts.net
# name answers publicly only through Funnel, which serves HTTPS, so a plain
# http:// URL to one is tailnet traffic (WireGuard-encrypted underneath).
_NS_PRIVATE_SUFFIXES = (".local", ".lan", ".home.arpa", ".internal", ".localhost",
                        ".ts.net")


def node_host_is_private(host: str) -> bool:
    """True for loopback, private, link-local, tailnet and LAN-only names."""
    h = (host or "").strip().strip("[]").lower().rstrip(".")
    if not h:
        return False
    if h == "localhost" or h.endswith(_NS_PRIVATE_SUFFIXES):
        return True
    try:
        ip = ipaddress.ip_address(h.split("%", 1)[0])
    except ValueError:
        # A single-label name ("studio") is resolved by the LAN's own DNS or
        # mDNS search domain; a dotted public name is not trusted for plaintext.
        return "." not in h
    if ip.is_loopback or ip.is_private or ip.is_link_local:
        return True
    return any(ip in net for net in _NS_TAILNET_NETS if net.version == ip.version)


def node_safe_url(url: str) -> str:
    """The URL without any user:password@ part, for logs and the admin panel."""
    try:
        parts = urlsplit(url or "")
        if parts.username or parts.password:
            netloc = parts.hostname or ""
            if parts.port:
                netloc += f":{parts.port}"
            return urlunsplit((parts.scheme, netloc, parts.path, parts.query, ""))
    except ValueError:
        return ""
    return url or ""


def node_url_problem(config: Any, url: str) -> str:
    """Why a peer URL must not be used ("" when it is fine)."""
    try:
        parts = urlsplit(str(url or "").strip())
        host = parts.hostname
    except ValueError:
        return "invalid node URL"
    if parts.scheme not in ("http", "https") or not host:
        return f"node URL must look like https://host:port, got {node_safe_url(str(url))!r}"
    if (parts.scheme == "http" and not node_host_is_private(host)
            and not getattr(config, "node_allow_plaintext", False)):
        return (f"refusing plain http:// to {host}: it is not a loopback, private or "
                "tailnet address, so the node token and every prompt would cross the "
                "network in the clear. Use https:// (NODE_CA_FILE for a private CA), "
                "a tailnet address, or set NODE_ALLOW_PLAINTEXT=1 to accept the risk")
    return ""


# --------------------------------------------------------------------------- #
# TLS
# --------------------------------------------------------------------------- #
_ns_ssl_cache: dict[tuple, Any] = {}


def node_tls_settings(config: Any) -> tuple[str, str, str]:
    return (str(getattr(config, "node_ca_file", "") or "").strip(),
            str(getattr(config, "node_client_cert", "") or "").strip(),
            str(getattr(config, "node_client_key", "") or "").strip())


def node_ssl_context(config: Any) -> Any:
    """An SSLContext for peer HTTPS, or True (httpx's default verification).

    NODE_CA_FILE replaces the public roots with your own CA (a private CA or a
    self-signed node certificate), so a public certificate for some other name
    cannot impersonate a node. NODE_CLIENT_CERT/KEY present a client
    certificate for mutual TLS. Missing files raise NodeSecurityError rather
    than quietly falling back to default verification.
    """
    ca, cert, key = node_tls_settings(config)
    if not ca and not cert:
        return True
    cache_key = (ca, cert, key)
    if cache_key in _ns_ssl_cache:
        return _ns_ssl_cache[cache_key]
    for label, path in (("NODE_CA_FILE", ca), ("NODE_CLIENT_CERT", cert),
                        ("NODE_CLIENT_KEY", key)):
        if path and not os.path.isfile(os.path.expanduser(path)):
            raise NodeSecurityError(f"{label} {path!r} does not exist")
    if ca:
        ctx = ssl.create_default_context(cafile=os.path.expanduser(ca))
    else:
        try:
            import certifi  # httpx depends on it, so it is normally present
            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:  # noqa: BLE001
            ctx = ssl.create_default_context()
    if cert:
        try:
            ctx.load_cert_chain(os.path.expanduser(cert),
                                os.path.expanduser(key) if key else None)
        except (ssl.SSLError, OSError) as exc:
            raise NodeSecurityError(f"cannot load NODE_CLIENT_CERT/KEY: {exc}") from exc
    _ns_ssl_cache[cache_key] = ctx
    return ctx


# --------------------------------------------------------------------------- #
# End-to-end sealing
# --------------------------------------------------------------------------- #
def _ns_b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _ns_b64d(text: str) -> bytes:
    text = str(text or "")
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def node_e2e_key_bytes(config: Any) -> bytes | None:
    """The 32-byte key from NODE_E2E_KEY (hex or base64), None when unset."""
    raw = str(getattr(config, "node_e2e_key", "") or "").strip()
    if not raw:
        return None
    key = None
    if len(raw) == 64:
        try:
            key = bytes.fromhex(raw)
        except ValueError:
            key = None
    if key is None:
        try:
            key = _ns_b64d(raw.replace("+", "-").replace("/", "_"))
        except (binascii.Error, ValueError):
            key = None
    if key is None or len(key) != _NS_KEY_BYTES:
        raise NodeSecurityError("NODE_E2E_KEY must be 32 bytes, as 64 hex characters or "
                                "base64; generate one with `deploy.py remote gen-key`")
    return key


def node_crypto_available() -> bool:
    return importlib.util.find_spec("cryptography") is not None


def node_e2e_problem(config: Any) -> str:
    """Why node traffic must be refused under the E2E setting ("" when fine)."""
    try:
        key = node_e2e_key_bytes(config)
    except NodeSecurityError as exc:
        return str(exc)
    if key is None:
        return ""
    if NODE_TEST_HOOKS.get("aead") is None and not node_crypto_available():
        return ("NODE_E2E_KEY is set but the optional `cryptography` package is not "
                "installed (pip install cryptography); refusing node traffic rather "
                "than sending it unsealed")
    return ""


def node_e2e_enabled(config: Any) -> bool:
    return bool(str(getattr(config, "node_e2e_key", "") or "").strip())


def _ns_make_aead(key: bytes) -> Any:
    hook = NODE_TEST_HOOKS.get("aead")
    if hook is not None:
        return hook(key)
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    return ChaCha20Poly1305(key)


class NodeE2E:
    """Seal and open node payloads with one shared key.

    One instance per key per process (see for_config), so the replay cache is
    shared by every request this process receives.
    """

    def __init__(self, key: bytes, skew_s: float = NODE_E2E_SKEW_S):
        self._aead = _ns_make_aead(key)
        self.skew_s = float(skew_s)
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    _instances: dict[str, "NodeE2E"] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def for_config(cls, config: Any) -> "NodeE2E | None":
        """The sealer for this config, None when E2E is off. Raises when refused."""
        problem = node_e2e_problem(config)
        if problem:
            raise NodeSecurityError(problem)
        key = node_e2e_key_bytes(config)
        if key is None:
            return None
        ident = hashlib.sha256(key + repr(NODE_TEST_HOOKS.get("aead")).encode()).hexdigest()
        with cls._instances_lock:
            inst = cls._instances.get(ident)
            if inst is None:
                inst = cls._instances[ident] = cls(key)
            return inst

    # ---- AAD ------------------------------------------------------------ #
    @staticmethod
    def _req_aad(method: str, path: str, ts: int, nonce_b64: str) -> bytes:
        return f"node-e2e/{_NS_VERSION}\nreq\n{method.upper()}\n{path}\n{ts}\n{nonce_b64}".encode()

    @staticmethod
    def _resp_aad(path: str, req_nonce_b64: str, seq: int) -> bytes:
        return f"node-e2e/{_NS_VERSION}\nresp\n{path}\n{req_nonce_b64}\n{seq}".encode()

    # ---- requests ------------------------------------------------------- #
    def seal_request(self, method: str, path: str, body: bytes | None,
                     now: float | None = None) -> tuple[str, str, bytes | None]:
        """(header value, request nonce, sealed JSON body or None)."""
        ts = int(now if now is not None else time.time())
        nonce = secrets.token_bytes(_NS_NONCE_BYTES)
        n64 = _ns_b64e(nonce)
        aad = self._req_aad(method, path, ts, n64)
        if body:
            ct = self._aead.encrypt(nonce, bytes(body), aad)
            envelope = json.dumps({"e2e": 1, "ct": _ns_b64e(ct)}).encode()
            return f"{_NS_VERSION};ts={ts};n={n64}", n64, envelope
        # No body (GET): seal an empty payload so the header itself is
        # authenticated and only genuine requests reach the replay cache.
        tag = self._aead.encrypt(nonce, b"", aad)
        return f"{_NS_VERSION};ts={ts};n={n64};t={_ns_b64e(tag)}", n64, None

    def open_request(self, method: str, path: str, header: str, envelope: Any,
                     now: float | None = None) -> tuple[str, bytes | None]:
        """Verify a request; (request nonce, plaintext body or None). Raises."""
        fields = {}
        parts = str(header or "").split(";")
        if not parts or parts[0] != _NS_VERSION:
            raise NodeSecurityError("unsupported or missing X-Node-E2E header")
        for part in parts[1:]:
            name, _, value = part.partition("=")
            fields[name.strip()] = value.strip()
        try:
            ts = int(fields["ts"])
            n64 = fields["n"]
            nonce = _ns_b64d(n64)
        except (KeyError, ValueError, binascii.Error) as exc:
            raise NodeSecurityError("malformed X-Node-E2E header") from exc
        if len(nonce) != _NS_NONCE_BYTES:
            raise NodeSecurityError("malformed X-Node-E2E nonce")
        now = float(now if now is not None else time.time())
        if abs(now - ts) > self.skew_s:
            raise NodeSecurityError(
                f"sealed request is {abs(now - ts):.0f}s from this node's clock "
                f"(limit {self.skew_s:.0f}s); check NTP on both machines")
        aad = self._req_aad(method, path, ts, n64)
        plain: bytes | None = None
        try:
            if isinstance(envelope, dict) and envelope.get("e2e"):
                plain = self._aead.decrypt(nonce, _ns_b64d(envelope.get("ct", "")), aad)
            else:
                if self._aead.decrypt(nonce, _ns_b64d(fields.get("t", "")), aad) != b"":
                    raise ValueError("non-empty tag payload")
        except NodeSecurityError:
            raise
        except Exception as exc:  # noqa: BLE001 - InvalidTag and friends
            raise NodeSecurityError("sealed request failed authentication") from exc
        with self._lock:
            if n64 in self._seen:
                raise NodeSecurityError("replayed sealed request refused")
            self._seen[n64] = now
            if len(self._seen) > 1024:
                horizon = now - 2 * self.skew_s
                for k in [k for k, t in self._seen.items() if t < horizon]:
                    del self._seen[k]
                while len(self._seen) > _NS_REPLAY_MAX:
                    self._seen.pop(next(iter(self._seen)))
        return n64, plain

    # ---- responses ------------------------------------------------------ #
    def seal_frame(self, path: str, req_nonce: str, seq: int, plain: bytes) -> dict:
        nonce = secrets.token_bytes(_NS_NONCE_BYTES)
        ct = self._aead.encrypt(nonce, bytes(plain), self._resp_aad(path, req_nonce, seq))
        return {"e2e": 1, "n": _ns_b64e(nonce), "ct": _ns_b64e(ct)}

    def open_frame(self, path: str, req_nonce: str, seq: int, envelope: Any) -> bytes:
        try:
            nonce = _ns_b64d(envelope["n"])
            return self._aead.decrypt(nonce, _ns_b64d(envelope["ct"]),
                                      self._resp_aad(path, req_nonce, seq))
        except Exception as exc:  # noqa: BLE001
            raise NodeSecurityError("sealed response failed authentication") from exc


def node_generate_e2e_key() -> str:
    """A fresh NODE_E2E_KEY value (base64url, 32 random bytes)."""
    return _ns_b64e(secrets.token_bytes(_NS_KEY_BYTES))


def node_canonical_path(path: str) -> str:
    """The path bound into the AAD: from "/api/node/" on.

    A node published under a prefix (https://host/llm) is reached at
    /llm/api/node/... while the app itself sees /api/node/... after the proxy
    strips the prefix; both sides must authenticate the same string.
    """
    path = str(path or "")
    i = path.find("/api/node/")
    return path[i:] if i >= 0 else path


# --------------------------------------------------------------------------- #
# Server side: used by the /api/node/* route bodies
# --------------------------------------------------------------------------- #
class NodeServerSeal:
    """Seals one response (JSON or SSE frames) to the request that asked."""

    def __init__(self, e2e: NodeE2E, path: str, req_nonce: str):
        self.e2e, self.path, self.req_nonce, self.seq = e2e, path, req_nonce, 0

    def headers(self) -> dict[str, str]:
        return {NODE_E2E_SEALED_HEADER: _NS_VERSION,
                NODE_PROTOCOL_HEADER: str(NODE_PROTOCOL)}

    def json(self, obj: Any) -> dict:
        env = self.e2e.seal_frame(self.path, self.req_nonce, self.seq,
                                  json.dumps(obj).encode())
        self.seq += 1
        return env

    def frame(self, payload: str) -> str:
        """One SSE `data:` frame. `payload` is the plaintext after "data: "."""
        env = self.e2e.seal_frame(self.path, self.req_nonce, self.seq, payload.encode())
        self.seq += 1
        return f"data: {json.dumps(env)}\n\n"


def node_server_open(config: Any, method: str, path: str, headers: Any,
                     body: Any = None) -> tuple[Any, "NodeServerSeal | None", tuple[int, str] | None]:
    """Admit one inbound node request under the E2E policy.

    Returns (plaintext body, response sealer or None, (status, error) or None).
    With E2E off the body passes through untouched and a sealed request is
    rejected (this node could not read it). With E2E on an unsealed request
    is rejected: accepting it would let a misconfigured peer downgrade.
    """
    header = ""
    try:
        header = headers.get(NODE_E2E_HEADER) or headers.get(NODE_E2E_HEADER.lower()) or ""
    except Exception:  # noqa: BLE001
        header = ""
    problem = node_e2e_problem(config)
    if problem:
        return None, None, (503, problem)
    if not node_e2e_enabled(config):
        if header or (isinstance(body, dict) and body.get("e2e")):
            return None, None, (400, "this node has no NODE_E2E_KEY, so it cannot read "
                                     "sealed node traffic; set the same key on both nodes")
        return body, None, None
    if not header:
        return None, None, (400, "this node requires sealed node traffic (NODE_E2E_KEY); "
                                 "set the same key on the calling node")
    try:
        e2e = NodeE2E.for_config(config)
        nonce, plain = e2e.open_request(method, path, header, body)
        decoded = json.loads(plain.decode("utf-8")) if plain is not None else None
    except NodeSecurityError as exc:
        log_event(_nodesec_log, 30, "nodesec.request_refused", path=path, error=str(exc))
        return None, None, (401, str(exc))
    except ValueError:
        return None, None, (400, "sealed body is not JSON")
    return decoded, NodeServerSeal(e2e, path, nonce), None


# --------------------------------------------------------------------------- #
# Client side: an httpx transport wrapper for every peer client
# --------------------------------------------------------------------------- #
def _ns_peer_error(plain: bytes) -> str | None:
    """The message of an in-band error frame, None for an ordinary frame."""
    if not plain.startswith(b"{") or b'"error"' not in plain:
        return None
    try:
        data = json.loads(plain)
    except ValueError:
        return None
    if not isinstance(data, dict) or "error" not in data or data.get("choices"):
        return None
    err = data["error"]
    if isinstance(err, dict):
        return str(err.get("message") or err.get("type") or "node error")
    return str(err)


class _NodeSSEDecoder:
    """Rewrites a peer's SSE byte stream: opens sealed frames, raises on errors."""

    def __init__(self, ctx: tuple | None):
        self.ctx, self.buf, self.seq = ctx, b"", 0

    def feed(self, chunk: bytes) -> bytes:
        self.buf += chunk
        out = []
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            out.append(self._line(line.rstrip(b"\r")))
        return b"".join(out)

    def flush(self) -> bytes:
        line, self.buf = self.buf, b""
        return self._line(line.rstrip(b"\r")) if line.strip() else b""

    def _line(self, line: bytes) -> bytes:
        if not line.startswith(b"data:"):
            return b""          # comments / blank separators: re-emitted per frame
        payload = line[5:].lstrip(b" ")
        if self.ctx is not None:
            e2e, path, nonce = self.ctx
            try:
                env = json.loads(payload)
            except ValueError as exc:
                raise NodeSecurityError("peer sent an unsealed frame on a sealed stream") from exc
            payload = e2e.open_frame(path, nonce, self.seq, env)
            self.seq += 1
        message = _ns_peer_error(payload)
        if message is not None:
            raise NodePeerError(f"peer node error: {message}")
        return b"data: " + payload + b"\n\n"


_ns_classes: dict[str, Any] = {}


def _ns_transport_classes() -> dict[str, Any]:
    """Build the transport classes on first use (httpx is imported lazily)."""
    if _ns_classes:
        return _ns_classes
    import httpx

    def prepare(config: Any, request: Any, body: bytes) -> tuple[Any, tuple | None]:
        problem = node_url_problem(config, str(request.url))
        if problem:
            raise NodeSecurityError(problem)
        e2e = NodeE2E.for_config(config)       # raises when E2E is refused
        request.headers[NODE_PROTOCOL_HEADER] = str(NODE_PROTOCOL)
        # The frame decoder works on the raw transport stream, which is still
        # content-encoded; asking for identity keeps it plain bytes.
        request.headers["Accept-Encoding"] = "identity"
        if e2e is None:
            return request, None
        headers = httpx.Headers(request.headers)
        path = node_canonical_path(request.url.path)
        header, nonce, sealed = e2e.seal_request(request.method, path, body or None)
        headers[NODE_E2E_HEADER] = header
        headers.pop("Content-Length", None)
        if sealed is not None:
            headers["Content-Type"] = "application/json"
        new = httpx.Request(request.method, request.url, headers=headers,
                            content=sealed if sealed is not None else b"",
                            extensions=request.extensions)
        return new, (e2e, path, nonce)

    def check_sealed(response: Any, ctx: tuple | None) -> bool:
        sealed = response.headers.get(NODE_E2E_SEALED_HEADER) == _NS_VERSION
        if ctx is not None and response.status_code == 200 and not sealed:
            raise NodeSecurityError("peer answered in plaintext while NODE_E2E_KEY is set")
        return sealed and ctx is not None

    def is_sse(response: Any) -> bool:
        return "text/event-stream" in response.headers.get("content-type", "")

    def open_json(raw: bytes, ctx: tuple) -> bytes:
        e2e, path, nonce = ctx
        try:
            env = json.loads(raw)
        except ValueError as exc:
            raise NodeSecurityError("peer sent an unsealed body on a sealed response") from exc
        return e2e.open_frame(path, nonce, 0, env)

    def clean_headers(response: Any) -> list:
        drop = {"content-length", "content-encoding", "transfer-encoding"}
        return [(k, v) for k, v in response.headers.multi_items() if k.lower() not in drop]

    class _AsyncStream(httpx.AsyncByteStream):
        def __init__(self, inner: Any, decoder: _NodeSSEDecoder):
            self.inner, self.decoder = inner, decoder

        async def __aiter__(self):
            async for chunk in self.inner:
                out = self.decoder.feed(chunk)
                if out:
                    yield out
            tail = self.decoder.flush()
            if tail:
                yield tail

        async def aclose(self) -> None:
            await self.inner.aclose()

    class _SyncStream(httpx.SyncByteStream):
        def __init__(self, inner: Any, decoder: _NodeSSEDecoder):
            self.inner, self.decoder = inner, decoder

        def __iter__(self):
            for chunk in self.inner:
                out = self.decoder.feed(chunk)
                if out:
                    yield out
            tail = self.decoder.flush()
            if tail:
                yield tail

        def close(self) -> None:
            self.inner.close()

    class NodeAsyncTransport(httpx.AsyncBaseTransport):
        def __init__(self, inner: Any, config: Any):
            self.inner, self.config = inner, config

        async def handle_async_request(self, request: Any) -> Any:
            body = await request.aread()
            new, ctx = prepare(self.config, request, body)
            response = await self.inner.handle_async_request(new)
            sealed = check_sealed(response, ctx)
            if is_sse(response) and response.status_code == 200:
                return httpx.Response(response.status_code, headers=clean_headers(response),
                                      stream=_AsyncStream(response.stream,
                                                          _NodeSSEDecoder(ctx if sealed else None)),
                                      extensions=response.extensions)
            if sealed:
                raw = await response.aread()
                await response.aclose()
                return httpx.Response(response.status_code, headers=clean_headers(response),
                                      content=open_json(raw, ctx), extensions=response.extensions)
            return response

        async def aclose(self) -> None:
            await self.inner.aclose()

    class NodeSyncTransport(httpx.BaseTransport):
        def __init__(self, inner: Any, config: Any):
            self.inner, self.config = inner, config

        def handle_request(self, request: Any) -> Any:
            body = request.read()
            new, ctx = prepare(self.config, request, body)
            response = self.inner.handle_request(new)
            sealed = check_sealed(response, ctx)
            if is_sse(response) and response.status_code == 200:
                return httpx.Response(response.status_code, headers=clean_headers(response),
                                      stream=_SyncStream(response.stream,
                                                         _NodeSSEDecoder(ctx if sealed else None)),
                                      extensions=response.extensions)
            if sealed:
                raw = response.read()
                response.close()
                return httpx.Response(response.status_code, headers=clean_headers(response),
                                      content=open_json(raw, ctx), extensions=response.extensions)
            return response

        def close(self) -> None:
            self.inner.close()

    _ns_classes.update(async_transport=NodeAsyncTransport, sync_transport=NodeSyncTransport)
    return _ns_classes


def node_client_key(config: Any) -> tuple:
    """What a cached peer client depends on (a changed setting needs a new one)."""
    return (node_tls_settings(config), bool(getattr(config, "node_allow_plaintext", False)),
            hashlib.sha256(str(getattr(config, "node_e2e_key", "") or "").encode()).hexdigest(),
            id(NODE_TEST_HOOKS.get("async_transport")), id(NODE_TEST_HOOKS.get("aead")))


def node_async_client(config: Any, **client_kwargs: Any) -> Any:
    """An httpx.AsyncClient for peer nodes: TLS options + policy + sealing."""
    import httpx
    hook = NODE_TEST_HOOKS.get("async_transport")
    inner = hook(config) if hook is not None else httpx.AsyncHTTPTransport(
        verify=node_ssl_context(config),
        limits=httpx.Limits(max_connections=16, max_keepalive_connections=8,
                            keepalive_expiry=120.0))
    cls = _ns_transport_classes()["async_transport"]
    client_kwargs.setdefault("timeout", httpx.Timeout(120.0, connect=15.0, pool=15.0))
    return httpx.AsyncClient(transport=cls(inner, config), **client_kwargs)


def node_sync_client(config: Any, **client_kwargs: Any) -> Any:
    """An httpx.Client for peer nodes (health probes, the cluster CLI)."""
    import httpx
    hook = NODE_TEST_HOOKS.get("sync_transport")
    inner = hook(config) if hook is not None else httpx.HTTPTransport(
        verify=node_ssl_context(config))
    cls = _ns_transport_classes()["sync_transport"]
    return httpx.Client(transport=cls(inner, config), **client_kwargs)


def node_security_summary(config: Any) -> dict:
    """What `deploy.py remote status` and the cluster CLI report."""
    ca, cert, key = node_tls_settings(config)
    return {
        "tokens": len(node_tokens(config)),
        "ca_file": ca or None, "client_cert": cert or None,
        "client_key": bool(key),
        "allow_plaintext": bool(getattr(config, "node_allow_plaintext", False)),
        "e2e": node_e2e_enabled(config),
        "e2e_problem": node_e2e_problem(config) or None,
        "cryptography": node_crypto_available(),
        "protocol": NODE_PROTOCOL,
    }


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'NODE_PROTOCOL', 'NODE_PROTOCOL_HEADER', 'NODE_E2E_HEADER', 'NODE_E2E_SEALED_HEADER',
    'NODE_E2E_SKEW_S', 'NODE_TEST_HOOKS',
    'NodeSecurityError', 'NodePeerError', 'NodeE2E', 'NodeServerSeal',
    'node_tokens', 'node_send_token', 'node_auth_headers',
    'node_host_is_private', 'node_safe_url', 'node_url_problem',
    'node_tls_settings', 'node_ssl_context',
    'node_e2e_key_bytes', 'node_crypto_available', 'node_e2e_problem', 'node_e2e_enabled',
    'node_generate_e2e_key', 'node_server_open', 'node_canonical_path',
    'node_client_key', 'node_async_client', 'node_sync_client', 'node_security_summary',
    '_ns_peer_error', '_NodeSSEDecoder', '_ns_b64e', '_ns_b64d',
]
