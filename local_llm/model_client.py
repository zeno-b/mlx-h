"""HTTP client for the local model server, with retries.

Split out of the original single-file deploy.py; behaviour is unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, AsyncGenerator

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .llm import *  # noqa: F401,F403
from .toolparse import *  # noqa: F401,F403
from .nodesec import *  # noqa: F401,F403
from .backends import *  # noqa: F401,F403
# Optional cloud fallback for oversized prefill: cloud_offload is None (one
# attribute read) unless CLOUD_FALLBACK_ENABLED and the user consented.
from .cloud import *  # noqa: F401,F403


class ModelClient:
    """Async client for the OpenAI-compatible model backend.

    When a cluster (ClusterRouter) is attached it becomes node-aware: each
    generation is routed to the best node (Mac Mini primary / Mac Studio
    secondary) and fails over to the next candidate on a connection failure,
    recording every decision. With no cluster, or a single-node cluster, it
    targets the local mlx server exactly as before.
    """

    def __init__(self, config: Config, cluster: Any = None):
        self.config = config
        self.cluster = cluster
        # The tool-call parser reads TOOL_CALL_FORMAT / TOOL_CALL_REPAIR from
        # the live config at parse time (toolparse.tool_call_policy), so a
        # Settings change applies to the next reply without a restart.
        bind_tool_call_config(config)

    # One pooled HTTP client for the whole process, not one per request and not
    # one per ModelClient. A fresh AsyncClient per request means a fresh TCP
    # connection per request: on a multi-step agent run that is a connect
    # handshake before every token stream, and the non-streaming path throws the
    # pool away between its two calls. It is a class attribute because
    # /api/agents/run builds a ModelClient per agent, so per-instance pools would
    # give the fan-out no reuse at all. Every call site passes its own timeout,
    # so nothing config-specific is baked into the shared client.
    _shared_http: Any = None
    _shared_loop: Any = None
    # Peer nodes get their own pool: TLS options (NODE_CA_FILE, client cert),
    # the plaintext policy and E2E sealing live in its transport (nodesec.py),
    # so the local mlx server is never routed through them.
    _peer_http: dict = {}

    @classmethod
    def _client(cls):
        """The shared AsyncClient, created on first use inside the running loop.

        A connection pool belongs to the event loop that created it, and this
        process starts a fresh loop per test app, so the client is rebound
        whenever the running loop changes; otherwise it would hand out sockets
        the current loop cannot await.
        """
        import httpx
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        client = cls._shared_http
        if client is None or cls._shared_loop is not loop or client.is_closed:
            cls._shared_http = httpx.AsyncClient(
                # Generous per-host pool: concurrent chats and agent fan-out all
                # target the same one or two nodes.
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=16,
                                    keepalive_expiry=120.0),
                timeout=httpx.Timeout(120.0, connect=15.0, pool=15.0),
            )
            cls._shared_loop = loop
        return cls._shared_http

    @classmethod
    def _peer_client(cls, config: Any):
        """The AsyncClient for remote nodes, per event loop and security settings."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        key = (id(loop), node_client_key(config))
        client = cls._peer_http.get(key)
        if client is None or client.is_closed:
            # Clients bound to another (finished) loop cannot be awaited here.
            for stale in [k for k in cls._peer_http if k[0] != id(loop)]:
                cls._peer_http.pop(stale, None)
            client = cls._peer_http[key] = node_async_client(config)
        return client

    def _client_for(self, node: Any):
        if node is None or getattr(node, "is_local", True):
            return self._client()
        return self._peer_client(self.config)

    @classmethod
    async def aclose(cls) -> None:
        """Release the pooled connections on shutdown. Safe to call twice."""
        peers, cls._peer_http = list(cls._peer_http.values()), {}
        for peer in peers:
            try:
                await peer.aclose()
            except Exception:  # noqa: BLE001 - shutdown must never raise
                pass
        client, cls._shared_http, cls._shared_loop = cls._shared_http, None, None
        if client is not None and not client.is_closed:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001 - shutdown must never raise
                pass

    def _local_chat_url(self) -> str:
        return f"http://127.0.0.1:{self.config.model_port}/v1/chat/completions"

    @property
    def url(self) -> str:
        return self._local_chat_url()

    def _targets(self, kind: str = "chat"):
        """Ordered list of (node, chat_url, headers, decision) to try.

        A single-node / no-cluster client yields exactly one local target, so the
        generation path is byte-for-byte the old behaviour.
        """
        cluster = self.cluster
        if cluster is None or not getattr(cluster.registry, "multi_node", False):
            return [(None, self._local_chat_url(), {}, None)]
        req = cluster.classify(model=self.config.model, kind=kind)
        decision = cluster.select(req)
        targets = []
        refusal = getattr(cluster.registry, "refusal", None)
        for node in decision.candidates:
            if node.is_local:
                targets.append((node, self._local_chat_url(), {}, decision))
                continue
            # A peer the security policy refuses (plain http to a public host,
            # E2E key without `cryptography`) is never dispatched to; the
            # heartbeat already shows why in the cluster panel.
            problem = refusal(node) if refusal is not None else ""
            if problem:
                continue
            # A paired node has its own token; others use the first NODE_TOKEN
            # (the list form exists for rotation on the receiving side).
            headers = node_auth_headers(self.config, node)
            url = node.remote_url.rstrip("/") + "/api/node/generate"
            targets.append((node, url, headers, decision))
        if not targets:
            local = cluster.registry.local_node()
            targets.append((local, self._local_chat_url(), {}, decision))
        return targets

    @staticmethod
    def _is_connection_error(exc: Exception) -> bool:
        """True for failures where trying another node is the right move.

        Covers transport errors (dropped/refused/timed-out connection) AND a 5xx
        from the model server itself: a 500/502/503 usually means the primary mlx
        server crashed or is reloading, so a healthy secondary should be tried
        rather than hammering the dead primary.
        """
        import httpx
        if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout,
                            httpx.RemoteProtocolError, httpx.ReadTimeout,
                            httpx.PoolTimeout)):
            return True
        text = str(exc).lower()
        if re.search(r"returned 5\d\d", text):  # "model server returned 5xx: ..."
            return True
        return any(s in text for s in ("connection", "refused", "reset",
                                       "timed out", "unreachable"))

    @classmethod
    def _should_fail_over(cls, node: Any, exc: Exception) -> bool:
        """_is_connection_error, plus the ways only a PEER node can fail.

        A peer that refuses the call (401/403 token or E2E key mismatch, 400
        "this node cannot read sealed traffic", a policy refusal raised by the
        peer transport) or reports an in-band error frame cannot serve this
        request, but another node may. The local server never answers those,
        so local behaviour is unchanged.
        """
        if cls._is_connection_error(exc):
            return True
        if node is None or getattr(node, "is_local", True):
            return False
        if isinstance(exc, (NodeSecurityError, NodePeerError)):
            return True
        return bool(re.search(r"returned 4\d\d", str(exc).lower()))

    async def wait_until_ready(self, timeout: float = 40.0) -> bool:
        """Wait until the server can actually GENERATE, or until timeout.

        A /v1/models ping only proves the process is up. It does not prove the
        generation thread is alive: a backend crash (e.g. mlx-lm dying inside
        _step) leaves the HTTP server answering 200 on /v1/models while every
        completion stalls forever. So probe with a real one-token completion.
        If that returns, the server can generate; if it errors or times out, it
        cannot, and the caller degrades honestly instead of retrying into a
        corpse. Returns True only on a genuine completion.
        """
        import httpx
        deadline = time.time() + timeout
        delay = 0.5
        probe_body = {
            # The backend's request name, not the repo id: on mlx-lm a repo id
            # reloads the weights WITHOUT the LoRA adapter (backends.py).
            "model": request_model_for(self.config),
            "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
        while time.time() < deadline:
            try:
                async with httpx.AsyncClient(timeout=min(15.0, timeout)) as client:
                    resp = await client.post(self.url, json=probe_body)
                if resp.status_code < 500:
                    data = resp.json()
                    # A live generation thread returns a choices array; a crashed
                    # backend that still serves HTTP will not.
                    if data.get("choices"):
                        return True
            except Exception:
                pass      # still starting: retry until the deadline below
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 3.0)
        return False

    @staticmethod
    def classify_error(exc: Exception) -> str:
        """A short, honest label for a generation failure.

        The retry notice used to say "memory limit" for every exception, which
        hid stalls, resets and Cloudflare-style pages behind a wrong cause. This
        names what actually happened so the UI and logs are truthful.
        """
        import httpx
        name = type(exc).__name__
        text = str(exc).lower()
        if isinstance(exc, httpx.ReadTimeout) or "readtimeout" in name.lower() or "timed out" in text:
            return "the model stalled (no output in time)"
        if isinstance(exc, (httpx.ConnectError, httpx.RemoteProtocolError)) or \
                "connection" in text or "reset" in text or "refused" in text:
            return "the model server dropped, likely out of memory and restarting"
        if "memory" in text or "oom" in text or "alloc" in text:
            return "the model server ran out of memory"
        return f"a generation error ({name})"

    def reply_budget(self, messages: list[dict], max_tokens: int | None = None,
                     quiet: bool = False) -> int:
        """The reply budget this prompt can actually afford.

        prompt + max_tokens must fit n_ctx. Nothing enforced that: a 2759-token
        prompt asking for 1536 reply tokens against a 4096-token window is a
        4295-token request, and what the backend does with the overflow (shift,
        truncate, refuse) is its business, not something the answer should
        depend on. Clamp here, once, where the real message list is known.
        """
        wanted = max_tokens or self.config.max_tokens
        room = self.config.context_size - messages_tokens(messages) - CONTEXT_SAFETY_MARGIN
        if room >= wanted:
            return wanted
        clamped = max(self.config.min_max_tokens, room)
        if not quiet:
            log(f"reply budget clamped {wanted} -> {clamped}: prompt is "
                f"{messages_tokens(messages)} tokens of a {self.config.context_size}-token "
                "context. The prompt is the thing to shrink.", logging.WARNING)
        return clamped

    def request_timeout(self, messages: list[dict], max_tokens: int | None = None) -> float:
        """Read timeout for a NON-streaming completion.

        On a stream the read timeout is the gap between chunks, which is what
        stall_timeout is written to mean. On a single POST there are no chunks,
        so the very same number silently becomes a cap on total generation time:
        a healthy server producing a 1536-token reply at ~16 tok/s needs ~96
        seconds and was cut off at 60, then reported through classify_error as
        "the model stalled (no output in time)" -- a false diagnosis that also
        triggered the shrink-and-retry loop. Scale with the work requested
        instead, against a pessimistic floor rate, keeping stall_timeout as the
        allowance for prefill and queueing before the first token.
        """
        budget = self.reply_budget(messages, max_tokens)
        derived = self.config.stall_timeout + (budget / self.config.decode_floor_tps)
        return min(self.config.max_generation_timeout, derived)

    def payload(
        self,
        messages: list[dict],
        stream: bool,
        max_tokens: int | None = None,
        temperature: float | None = None,
        extra: dict | None = None,
    ) -> dict:
        body = {
            # mlx-lm maps only "default_model" onto its CLI --adapter-path; the
            # repo id made every request reload the base weights without the
            # feedback adapter (backends.request_model_for). A peer node's
            # /api/node/generate ignores this field and builds its own.
            "model": request_model_for(self.config),
            "messages": messages,
            "stream": stream,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.reply_budget(messages, max_tokens),
        }
        # mlx_lm.server reads these from the request body; they are not OpenAI
        # parameters. They are OMITTED by default: on recent mlx-lm the
        # repetition-penalty path builds a null logits-processor list and the
        # generation thread crashes on every request. Only send them when
        # explicitly re-enabled.
        if (self.config.repetition_penalty_enabled
                and self.config.repetition_penalty
                and self.config.repetition_penalty != 1.0):
            body["repetition_penalty"] = self.config.repetition_penalty
            body["repetition_context_size"] = self.config.repetition_context_size
        if stream:
            # Ask for usage on the final chunk. Servers that do not know the
            # option ignore it, and the estimate below covers them.
            body["stream_options"] = {"include_usage": True}
        if self.config.disable_thinking and not self.config.reasoning_visible:
            # Qwen3.5 and friends default to thinking-on. Chain of thought is a
            # bad trade in a tool loop: it burns the KV cache on tokens the
            # protocol discards, and the reasoning text confuses JSON parsing.
            # Both spellings are in circulation; unknown keys are ignored.
            body["chat_template_kwargs"] = {"enable_thinking": False}
            body["enable_thinking"] = False
        if extra:
            for key, value in extra.items():
                if key in _MC_PASSTHROUGH and value is not None:
                    body[key] = value
        # One answer to "is thinking on", in both spellings. A caller's extra
        # may set either one (or the template kwargs dict); whichever it set
        # wins over the config default, and the other spelling is made to
        # agree, because a server that reads one and a template that reads the
        # other disagreeing is how a "thinking off" request still thinks.
        kwargs = body.get("chat_template_kwargs")
        kwargs = dict(kwargs) if isinstance(kwargs, dict) else {}
        thinking = None
        if extra and isinstance(extra.get("enable_thinking"), bool):
            thinking = extra["enable_thinking"]
        elif extra and isinstance((extra.get("chat_template_kwargs") or {}).get("enable_thinking"), bool):
            thinking = extra["chat_template_kwargs"]["enable_thinking"]
        elif "enable_thinking" in body or "enable_thinking" in kwargs:
            thinking = bool(kwargs.get("enable_thinking", body.get("enable_thinking")))
        if thinking is not None:
            kwargs["enable_thinking"] = thinking
            body["enable_thinking"] = thinking
        if kwargs:
            body["chat_template_kwargs"] = kwargs
        return body

    def _stats_from_usage(self, usage: dict | None, messages: list[dict], text: str,
                          started: float, first_token_at: float | None,
                          reasoning_text: str = "") -> GenerationStats:
        total_ms = (time.time() - started) * 1000
        ttft_ms = ((first_token_at - started) * 1000) if first_token_at else total_ms
        details = (usage or {}).get("completion_tokens_details") or {}
        reasoning_tokens = int(details.get("reasoning_tokens") or 0) if isinstance(details, dict) else 0
        if not reasoning_tokens and reasoning_text:
            reasoning_tokens = estimate_tokens(reasoning_text)
        if usage:
            stats = GenerationStats(
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                ttft_ms=ttft_ms, total_ms=total_ms, from_server=True,
            )
        else:
            stats = GenerationStats(
                prompt_tokens=messages_tokens(messages),
                completion_tokens=estimate_tokens(text) + estimate_tokens(reasoning_text),
                ttft_ms=ttft_ms, total_ms=total_ms, from_server=False,
            )
        stats.cached_tokens = _mc_cached_tokens(usage)
        stats.reasoning_tokens = reasoning_tokens
        stats.ttft_measured = first_token_at is not None
        return stats

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        extra: dict | None = None,
    ) -> str:
        text, _ = await self.complete_with_stats(messages, max_tokens, temperature,
                                                 extra=extra)
        return text

    async def complete_with_stats(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        kind: str = "chat",
        conversation_id: str | None = None,
        extra: dict | None = None,
    ) -> tuple[str, GenerationStats]:
        import httpx
        import uuid as _uuid
        # Cloud fallback (cloud.py): a consented oversized prompt is answered
        # there; None, or a cloud failure before any output, runs the local path.
        offload = await cloud_offload(self.config, messages, max_tokens, kind,
                                      conversation_id, extra)
        if offload is not None:
            cloud_stats = GenerationStats()
            cloud_text = await offload.complete(messages, max_tokens, temperature,
                                                cloud_stats, extra)
            if cloud_text is not None:
                return cloud_text, cloud_stats
        targets = self._targets(kind)
        task_id = _uuid.uuid4().hex[:16]
        cid = get_correlation_id()
        last_error: Exception | None = None
        active_node = None
        node_ended = True
        if self.cluster is not None:
            self.cluster.claim(task_id)
        try:
            for attempt, (node, url, headers, decision) in enumerate(targets):
                started = time.time()
                active_node, node_ended = node, False
                if self.cluster is not None and node is not None:
                    self.cluster.begin(node)
                try:
                    timeout = httpx.Timeout(
                        self.request_timeout(messages, max_tokens), connect=15.0, pool=15.0)
                    client = self._client_for(node)
                    resp = await client.post(url, headers=headers, timeout=timeout,
                                             json=self.payload(messages, False, max_tokens, temperature, extra))
                    # The max_tokens-less retry is for a local mlx server that
                    # rejects the field; a peer applies it itself, and its 502
                    # or 503 must reach the failover below, not run twice.
                    if resp.status_code != 200 and (node is None or node.is_local):
                        fallback = self.payload(messages, False, max_tokens, temperature, extra)
                        fallback.pop("max_tokens", None)
                        resp = await client.post(url, headers=headers, timeout=timeout,
                                                 json=fallback)
                    if resp.status_code != 200:
                        raise RuntimeError(f"model server returned {resp.status_code}: {resp.text[:300]}")
                    data = resp.json()
                    # content can be null (a pure tool-call reply, or mlx-lm's
                    # "schema requires content" None); reasoning and native
                    # tool_calls arrive beside it (mlx_lm/server.py
                    # generate_response: message.reasoning, message.tool_calls).
                    # The reasoning is NOT prepended here: callers of the
                    # non-streaming path parse the text directly (routing JSON)
                    # and never saw reasoning before; only its cost is counted.
                    text, reasoning_text = _mc_message_text(data["choices"][0].get("message"))
                    stats = self._stats_from_usage(data.get("usage"), messages, text, started,
                                                   None, reasoning_text)
                    stats.finish_reason = str(
                        (data["choices"][0] or {}).get("finish_reason") or "")
                    dur = (time.time() - started) * 1000
                    if self.cluster is not None and node is not None:
                        self.cluster.end(node, True, dur)
                        node_ended = True
                        self.cluster.record(decision, node,
                                            status=("ok" if attempt == 0 else "failover_ok"),
                                            attempt=attempt, duration_ms=dur,
                                            correlation_id=cid, task_id=task_id,
                                            conversation_id=conversation_id, user_id=get_acting_user())
                    return text, stats
                except Exception as exc:
                    last_error = exc
                    if self.cluster is not None and node is not None:
                        self.cluster.end(node, False)
                        node_ended = True
                        self.cluster.record(decision, node, status="failed", attempt=attempt,
                                            error=self.classify_error(exc), correlation_id=cid,
                                            task_id=task_id, conversation_id=conversation_id, user_id=get_acting_user())
                    if (node is not None and attempt + 1 < len(targets)
                            and self._should_fail_over(node, exc)):
                        log_event(get_logger("model"), 30, "model.failover",
                                  from_node=node.name, error=self.classify_error(exc),
                                  correlation_id=cid, task_id=task_id)
                        continue
                    raise
            raise last_error or RuntimeError("no model node available")
        finally:
            # Safety net for a cancellation (CancelledError is BaseException, not
            # caught above): decrement the active count so the node is not left
            # "busy". A consumer leaving is NOT a node failure, so this end is
            # recorded as success -- exactly as stream() does. Real failures are
            # handled in the except above and set node_ended, so they are never
            # double-counted here.
            if self.cluster is not None:
                if active_node is not None and not node_ended:
                    self.cluster.end(active_node, True)
                self.cluster.release(task_id)

    async def stream(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stats: GenerationStats | None = None,
        kind: str = "chat",
        conversation_id: str | None = None,
        extra: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yield content deltas. When `stats` is passed it is filled in place.

        In place rather than returned because this is an async generator and the
        caller needs the numbers even when it breaks out of the loop early, which
        the agent does on every completed tool call. Fails over to another node
        only before the first token arrives (never mid-stream).

        A server that separates reasoning (mlx_lm/server.py puts it in
        delta.reasoning; others use reasoning_content) has it re-wrapped here
        as <think>...</think>, so ThinkSplitter shows it live exactly as it does
        for a model that writes the tags itself; before, it was dropped and a
        thinking model's whole reasoning phase streamed as silence. Native
        delta.tool_calls are collected and emitted at the end as the canonical
        {"tool": ..., "args": ...} text the agent loop already parses.
        """
        events = self._stream_core(messages, max_tokens, temperature, stats, kind,
                                   conversation_id, extra)
        in_think = False
        said = False
        calls = _McToolCalls()
        try:
            async for event in events:
                etype = event["type"]
                if etype == "reasoning":
                    if not in_think:
                        in_think = True
                        yield "<think>"
                    yield event["text"]
                elif etype == "content":
                    if in_think:
                        in_think = False
                        yield "</think>"
                    said = said or bool(event["text"].strip())
                    yield event["text"]
                elif etype == "tool_call_delta":
                    calls.add(event)
            if in_think:
                in_think = False
                yield "</think>"
            synthesized = calls.protocol_text()
            if synthesized:
                yield ("\n" + synthesized) if said else synthesized
        finally:
            await events.aclose()

    async def stream_events(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stats: GenerationStats | None = None,
        kind: str = "chat",
        conversation_id: str | None = None,
        extra: dict | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Typed generation events, for callers that want structure, not text.

        Yields dicts with a "type":
          content          {"text"}
          reasoning        {"text"}
          tool_call_delta  {"index", "id", "name", "arguments", "source", "done"}
                           source "native" passes the server's delta.tool_calls
                           through as they arrive (and a final done=True event
                           per call carrying parsed "args"); source "prompt" is
                           a call found in the TEXT by ToolCallStreamDetector,
                           only when extra={"parse_tool_calls": True} (with an
                           optional "known_tools" set). Detected call text is
                           then not repeated as content.
          finish           {"reason"}  ("stop", "length", "tool_calls")
          usage            {"usage"}   the server's raw usage object
        Same routing, failover and cluster bracketing as stream(); `stats` is
        filled in place the same way.
        """
        opts = extra or {}
        detector = (ToolCallStreamDetector(opts.get("known_tools"))
                    if opts.get("parse_tool_calls") else None)
        events = self._stream_core(messages, max_tokens, temperature, stats, kind,
                                   conversation_id, extra)
        calls = _McToolCalls()
        flushed = False
        try:
            async for event in events:
                etype = event["type"]
                if etype == "content" and detector is not None:
                    for out in _mc_detector_events(detector.feed(event["text"])):
                        yield out
                    continue
                if etype == "tool_call_delta":
                    calls.add(event)
                    yield event
                    continue
                if etype == "finish":
                    for done in calls.finished():
                        yield done
                    if detector is not None and not flushed:
                        flushed = True
                        for out in _mc_detector_events(
                                detector.flush(truncated=event["reason"] == "length")):
                            yield out
                yield event
            for done in calls.finished():
                yield done
            if detector is not None and not flushed:
                flushed = True
                for out in _mc_detector_events(detector.flush()):
                    yield out
        finally:
            await events.aclose()

    async def _stream_core(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stats: GenerationStats | None = None,
        kind: str = "chat",
        conversation_id: str | None = None,
        extra: dict | None = None,
    ) -> AsyncGenerator[dict, None]:
        """The one streaming implementation: routing, failover, cluster
        bracketing and stats, yielding raw typed events (see stream_events)."""
        import httpx
        import uuid as _uuid
        # Cloud fallback (cloud.py): the same typed events from a consented
        # cloud model; a failure before the first output falls through to local.
        offload = await cloud_offload(self.config, messages, max_tokens, kind,
                                      conversation_id, extra)
        if offload is not None:
            cloud_events = offload.events(messages, max_tokens, temperature, stats, extra)
            try:
                async for cloud_event in cloud_events:
                    yield cloud_event
            finally:
                await cloud_events.aclose()
            if offload.served:
                return
        targets = self._targets(kind)
        task_id = _uuid.uuid4().hex[:16]
        cid = get_correlation_id()
        started = time.time()
        first_token_at: float | None = None
        usage: dict | None = None
        finish_reason = ""
        text_len = 0
        reasoning_len = 0
        last_error: Exception | None = None
        active_node = None
        node_ended = True
        streaming_started = False
        if self.cluster is not None:
            self.cluster.claim(task_id)
        try:
            for attempt, (node, url, headers, decision) in enumerate(targets):
                started = time.time()
                first_token_at = None
                usage = None
                text_len = 0
                reasoning_len = 0
                streaming_started = False
                active_node, node_ended = node, False
                if self.cluster is not None and node is not None:
                    self.cluster.begin(node)
                try:
                    # stall_timeout is correct HERE and only here: on a stream
                    # the read timeout is the gap between chunks, so it detects
                    # silence rather than capping total generation time. See
                    # request_timeout for the non-streaming path.
                    timeout = httpx.Timeout(self.config.stall_timeout, connect=15.0, pool=15.0)
                    client = self._client_for(node)
                    async with client.stream(
                        "POST", url, headers=headers, timeout=timeout,
                        json=self.payload(messages, True, max_tokens, temperature, extra)
                    ) as resp:
                        if resp.status_code != 200:
                            body = (await resp.aread()).decode("utf-8", "replace")
                            raise RuntimeError(f"model server returned {resp.status_code}: {body[:300]}")
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            chunk = line[6:]
                            if chunk.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(chunk)
                            except Exception:
                                continue
                            if data.get("usage"):
                                usage = data["usage"]
                            choices = data.get("choices") or []
                            if not choices:
                                continue
                            choice = choices[0] or {}
                            delta = choice.get("delta") or {}
                            content = delta.get("content") or ""
                            reasoning = (delta.get("reasoning")
                                         or delta.get("reasoning_content") or "")
                            native = delta.get("tool_calls") or []
                            if not native and isinstance(delta.get("function_call"), dict):
                                native = [{"index": 0, "function": delta["function_call"]}]
                            if not isinstance(native, list):
                                native = []
                            if (content or reasoning or native) and first_token_at is None:
                                first_token_at = time.time()
                                streaming_started = True
                            if isinstance(reasoning, str) and reasoning:
                                reasoning_len += len(reasoning)
                                yield {"type": "reasoning", "text": reasoning}
                            if isinstance(content, str) and content:
                                text_len += len(content)
                                yield {"type": "content", "text": content}
                            for position, call in enumerate(native):
                                if isinstance(call, dict):
                                    event = _mc_tool_delta(call, position)
                                    text_len += len(event["arguments"]) + len(event["name"])
                                    yield event
                            if choice.get("finish_reason"):
                                finish_reason = str(choice["finish_reason"])
                                yield {"type": "finish", "reason": finish_reason}
                        if usage:
                            yield {"type": "usage", "usage": usage}
                    # Completed this node successfully.
                    if self.cluster is not None and node is not None:
                        self.cluster.end(node, True, (time.time() - started) * 1000)
                        node_ended = True
                        self.cluster.record(decision, node,
                                            status=("ok" if attempt == 0 else "failover_ok"),
                                            attempt=attempt, duration_ms=(time.time() - started) * 1000,
                                            correlation_id=cid, task_id=task_id,
                                            conversation_id=conversation_id, user_id=get_acting_user())
                    return
                except Exception as exc:
                    last_error = exc
                    if self.cluster is not None and node is not None:
                        self.cluster.end(node, False)
                        node_ended = True
                        self.cluster.record(decision, node, status="failed", attempt=attempt,
                                            error=self.classify_error(exc), correlation_id=cid,
                                            task_id=task_id, conversation_id=conversation_id, user_id=get_acting_user())
                    if (node is not None and not streaming_started
                            and attempt + 1 < len(targets) and self._should_fail_over(node, exc)):
                        log_event(get_logger("model"), 30, "model.failover",
                                  from_node=node.name, error=self.classify_error(exc),
                                  correlation_id=cid, task_id=task_id)
                        continue
                    raise
            if last_error:
                raise last_error
        finally:
            # Safety net for early consumer close (GeneratorExit / cancellation):
            # decrement the active count so a node is never left "busy" after the
            # agent stops reading. A consumer leaving is NOT a node failure, so
            # this end is recorded as success (real failures are handled above and
            # set node_ended). Also releases the idempotency claim.
            if (self.cluster is not None and active_node is not None and not node_ended):
                self.cluster.end(active_node, True, (time.time() - started) * 1000)
            if self.cluster is not None:
                self.cluster.release(task_id)
            if stats is not None:
                measured = self._stats_from_usage(
                    usage, messages, "x" * text_len, started, first_token_at,
                    "x" * reasoning_len,
                )
                stats.prompt_tokens = measured.prompt_tokens
                stats.completion_tokens = measured.completion_tokens
                stats.finish_reason = finish_reason
                stats.ttft_ms = measured.ttft_ms
                stats.total_ms = measured.total_ms
                stats.from_server = measured.from_server
                stats.cached_tokens = measured.cached_tokens
                stats.reasoning_tokens = measured.reasoning_tokens
                stats.ttft_measured = measured.ttft_measured


# ---------------------------------------------------------------------------
# Response intake helpers
# ---------------------------------------------------------------------------
# Request keys a caller may pass through payload(extra=...). A whitelist, not a
# blind merge: "model", "messages", "stream" and "max_tokens" are owned by
# payload() (routing, the adapter alias and the context budget depend on
# them), and an arbitrary key from a route body must never reach the server.
_MC_PASSTHROUGH = frozenset({
    "tools", "tool_choice", "parallel_tool_calls", "response_format", "stop",
    "seed", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty",
    "logit_bias", "logprobs", "top_logprobs", "chat_template_kwargs",
    "enable_thinking", "reasoning_effort", "num_draft_tokens",
    "xtc_probability", "xtc_threshold",
})


def _mc_cached_tokens(usage: Any) -> int:
    """usage.prompt_tokens_details.cached_tokens (mlx_lm/server.py sets it
    from its prompt cache), or a flat usage.cached_tokens; 0 when absent."""
    if not isinstance(usage, dict):
        return 0
    details = usage.get("prompt_tokens_details")
    value = details.get("cached_tokens") if isinstance(details, dict) else None
    if value is None:
        value = usage.get("cached_tokens")
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _mc_tool_delta(call: dict, position: int) -> dict:
    """One native tool_calls entry as a typed event.

    Shape per mlx_lm/server.py ToolCallFormatter._format: {"id", "type":
    "function", "function": {"name", "arguments": <JSON string>}, "index"}.
    OpenAI-style servers stream the name once and the arguments in fragments,
    so every field is optional here and _McToolCalls joins them by index.
    """
    fn = call.get("function")
    fn = fn if isinstance(fn, dict) else {}
    index = call.get("index")
    if not isinstance(index, int) or isinstance(index, bool):
        index = position
    args = fn.get("arguments", call.get("arguments"))
    if isinstance(args, (dict, list)):
        args = json.dumps(args, ensure_ascii=False)
    elif args is None:
        args = ""
    name = fn.get("name") or call.get("name") or ""
    return {"type": "tool_call_delta", "index": index, "id": str(call.get("id") or ""),
            "name": str(name), "arguments": str(args), "source": "native", "done": False}


class _McToolCalls:
    """Native tool-call deltas joined by index into whole calls."""

    def __init__(self) -> None:
        self._calls: dict[int, dict] = {}
        self._emitted: set[int] = set()

    def add(self, event: dict) -> None:
        if event.get("source", "native") != "native" or event.get("done"):
            return
        slot = self._calls.setdefault(event["index"], {"id": "", "name": "", "arguments": ""})
        if event.get("id") and not slot["id"]:
            slot["id"] = event["id"]
        if event.get("name") and not slot["name"]:
            slot["name"] = event["name"]
        slot["arguments"] += event.get("arguments") or ""

    def parsed(self) -> list[tuple[int, ParsedToolCall]]:
        _fmt, repair = tool_call_policy()
        out = []
        for index in sorted(self._calls):
            slot = self._calls[index]
            name = slot["name"].strip()
            if not name:
                continue            # never invent a name for a nameless call
            repairs: list = []
            raw = slot["arguments"]
            args = _tp_args(raw, repair, repairs) if raw.strip() else {}
            out.append((index, ParsedToolCall(
                name=name, args=args, raw=raw, format="openai",
                recovered=bool(repairs), repairs=list(dict.fromkeys(repairs)),
                id=slot["id"])))
        return out

    def protocol_text(self) -> str:
        """The calls as the canonical {"tool", "args"} text the agent parses."""
        return "\n".join(call.as_protocol() for _i, call in self.parsed())

    def finished(self) -> list[dict]:
        """A done=True event per whole call not reported yet."""
        events = []
        for index, call in self.parsed():
            if index in self._emitted:
                continue
            self._emitted.add(index)
            events.append({"type": "tool_call_delta", "index": index, "id": call.id,
                           "name": call.name, "arguments": call.raw, "args": call.args,
                           "source": "native", "done": True, "recovered": call.recovered,
                           "repairs": call.repairs})
        return events


def _mc_content_text(content: Any) -> str:
    """message.content as text: null is "" (a pure tool-call reply), and a
    list of content parts is joined from its text parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text") or "") for part in content
                       if isinstance(part, dict) and part.get("type", "text") == "text")
    return ""


def _mc_message_text(message: Any) -> tuple[str, str]:
    """(answer text, reasoning text) of a non-streaming choices[0].message.

    Native message.tool_calls are appended as canonical protocol text, so a
    server that parsed the call out of the model's text (mlx-lm does for any
    tokenizer with a tool parser) no longer hands the agent an empty reply.
    """
    if not isinstance(message, dict):
        return "", ""
    text = _mc_content_text(message.get("content"))
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    native = message.get("tool_calls") or []
    if not native and isinstance(message.get("function_call"), dict):
        native = [{"function": message["function_call"]}]
    if isinstance(native, list) and native:
        calls = _McToolCalls()
        for position, call in enumerate(native):
            if isinstance(call, dict):
                calls.add(_mc_tool_delta(call, position))
        synthesized = calls.protocol_text()
        if synthesized:
            text = (text.rstrip() + "\n" + synthesized) if text.strip() else synthesized
    return text, reasoning if isinstance(reasoning, str) else ""


def _mc_detector_events(events: list[dict]) -> list[dict]:
    """ToolCallStreamDetector events as stream_events events (source "prompt").

    An aborted call's text is released as content: it was held back while it
    looked like a call, so nothing the model wrote is lost.
    """
    out = []
    for ev in events:
        kind = ev["type"]
        if kind == "text":
            out.append({"type": "content", "text": ev["text"]})
        elif kind == "args_delta":
            out.append({"type": "tool_call_delta", "index": ev["index"], "id": "",
                        "name": "", "arguments": ev["delta"], "source": "prompt",
                        "done": False})
        elif kind == "call_end":
            call = ev["call"]
            out.append({"type": "tool_call_delta", "index": ev["index"], "id": call.id,
                        "name": call.name,
                        "arguments": json.dumps(call.args, ensure_ascii=False),
                        "args": call.args, "source": "prompt", "done": True,
                        "format": call.format, "recovered": call.recovered,
                        "repairs": call.repairs, "late": bool(ev.get("late"))})
        elif kind == "call_abort":
            out.append({"type": "content", "text": ev["raw"], "aborted_call": ev["index"]})
    return out


# ---------------------------------------------------------------------------
# Request routing
# ---------------------------------------------------------------------------
# The old approach tried to recognise every phrasing of "search this" or "what
# is the weather in X" with regular expressions. That is a losing game: users
# phrase things in unbounded ways, and each new phrasing needed another pattern.
# The general fix is to make the *model* the router. It emits one structured
# decision (a small JSON object) saying how to handle the message and, for a
# lookup, what to search or which place and day it is about. Deterministic code
# then executes that decision. The model does what models are good at (reading
# intent from free text); the code does what code is good at (reliably running
# the chosen tool). This module holds the two deterministic shortcuts kept for
# cost reasons, plus the JSON extraction the router relies on.

# A bare arithmetic expression: only digits, spaces and operators. Routed to the
# calculator without a model call, since "17*23" needs no interpretation.
ARITHMETIC_ONLY = re.compile(r"^[\d\s+\-*/().,^%]+$")
# A message that is nothing but a URL. Routed straight to fetch_url.
BARE_URL = re.compile(r"^\s*(https?://\S+)\s*$", re.I)

# Greetings and acknowledgements that are not worth a routing round trip. These
# get a plain reply; everything longer is eligible for model routing.
TRIVIAL_MESSAGE = re.compile(
    r"^\s*(?:hi|hey|hello|yo|sup)(?:\s+(?:there|all|everyone|folks|claude))?"
    r"|^\s*(?:thanks|thank you|thx|ok|okay|cool|nice|got it|great|perfect|"
    r"lol|haha|bye|goodbye|good (?:morning|evening|night)|"
    r"how are you|what's up|whats up)\b",
    re.I,
)
_TRIVIAL_TAIL = re.compile(r"[\s!.?]*$")



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'ARITHMETIC_ONLY',
    'BARE_URL',
    'ModelClient',
    'TRIVIAL_MESSAGE',
    '_MC_PASSTHROUGH',
    '_McToolCalls',
    '_TRIVIAL_TAIL',
    '_mc_cached_tokens',
    '_mc_content_text',
    '_mc_detector_events',
    '_mc_message_text',
    '_mc_tool_delta',
]
