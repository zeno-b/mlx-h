"""Web search backend, URL guards, and HTML stripping.

No API keys are required. Search uses DuckDuckGo Lite EXCLUSIVELY
(https://lite.duckduckgo.com/lite/). This is a hard project constraint, not a
default: there is no configuration, environment variable or code path that can
point web search at Google, Bing, Brave, Tavily, SearXNG or any other engine.
The single allowed endpoint lives in SearchBackend.ENDPOINT and the invariant is
covered by the self-test suite.

Split out of the original single-file deploy.py; public names are kept
compatible with the original module.
"""

from __future__ import annotations

import asyncio
import contextvars
import html
import ipaddress
import re
import socket
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from .core import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403


TOOL_TRAINING_PREAMBLE = (
    "You call tools by replying with a single JSON object and nothing else, in "
    'the form {"tool": "tool_name", "args": {"arg": "value"}}.'
)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0 Safari/537.36"
)

# Hard ceiling on a single fetch_url download, independent of tool_result_chars.
MAX_FETCH_BYTES = 2_000_000

_ALLOWED_FETCH_SCHEMES = {"http", "https"}

# Most worker threads one batch of searches or page fetches may use, whatever
# SEARCH_WORKERS says. The work is network wait, so threads overlap it well;
# the cap keeps one batch from opening more connections to a single engine
# than looks like a person using it.
_PARALLEL_MAX_WORKERS = 8


def search_workers(config: Any, wanted: int) -> int:
    """Worker threads for a batch of `wanted` network jobs (SEARCH_WORKERS caps it)."""
    try:
        cap = int(getattr(config, "search_workers", 4) or 1)
    except (TypeError, ValueError):
        cap = 4
    return max(1, min(_PARALLEL_MAX_WORKERS, cap, int(wanted or 1)))


def parallel_map(fn: Callable[[Any], Any], items: Any,
                 workers: int) -> list[tuple[Any, BaseException | None]]:
    """fn(item) for every item on up to `workers` threads; (result, error) in input order.

    Every job runs in a copy of the caller's context, so the acting user, the
    conversation and the turn (the contextvars the approval policy and the
    tool log read inside ToolRegistry.call) are the same on each worker thread
    as on the calling one. A plain ThreadPoolExecutor.submit would run the job
    with an empty context, and the policy would judge the call as nobody's.
    A job that raises fills its own slot with (None, exc); the others are not
    affected.
    """
    batch = list(items)

    def run(item: Any) -> tuple[Any, BaseException | None]:
        try:
            return fn(item), None
        except Exception as exc:  # noqa: BLE001 - reported in the item's slot
            return None, exc

    if workers <= 1 or len(batch) <= 1:
        return [run(item) for item in batch]
    with ThreadPoolExecutor(max_workers=min(int(workers), len(batch)),
                            thread_name_prefix="parallel") as pool:
        futures = [pool.submit(contextvars.copy_context().run, run, item) for item in batch]
        return [future.result() for future in futures]


async def thread_task(fn: Callable[[Any], Any], item: Any,
                      gate: "asyncio.Semaphore | None" = None) -> tuple[Any, BaseException | None]:
    """fn(item) on a worker thread, as (result, error); waits for `gate` first.

    For a caller that starts several of these and consumes them one at a
    time, which parallel_map (it waits for the whole batch) cannot do. An
    exception comes back as a value, so a task nobody ends up awaiting never
    logs "Task exception was never retrieved". asyncio.to_thread copies the
    caller's context into the thread, as parallel_map does.
    """
    async def run() -> tuple[Any, BaseException | None]:
        try:
            return await asyncio.to_thread(fn, item), None
        except Exception as exc:  # noqa: BLE001 - reported to whoever awaits
            return None, exc

    if gate is None:
        return await run()
    async with gate:
        return await run()


def guard_public_url(url: str) -> None:
    """Reject URLs that resolve to localhost or private/reserved networks.

    DNS is resolved before the request so hostnames pointing at private
    addresses are rejected too.
    """
    parsed = urllib.parse.urlparse(url)

    if parsed.scheme.lower() not in _ALLOWED_FETCH_SCHEMES:
        raise ValueError(
            f"unsupported URL scheme: {parsed.scheme!r}; "
            "only http and https are allowed"
        )

    host = parsed.hostname
    if not host:
        raise ValueError("url has no host")

    # Reject obvious IP literals before DNS resolution.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None

    if literal is not None:
        _reject_non_public_address(host, literal)
        return

    try:
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise ValueError(f"invalid URL port: {exc}") from exc

    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"could not resolve {host}: {exc}") from exc

    if not infos:
        raise ValueError(f"could not resolve {host}")

    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        _reject_non_public_address(host, address)


def _reject_non_public_address(host: str, address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    """Reject addresses that must never be fetched by an external URL tool."""
    if (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    ):
        raise ValueError(
            f"refusing to fetch {host} ({address}): "
            "private or non-public address"
        )


def strip_html(raw: str) -> str:
    """Turn an HTML fragment into readable plain text."""
    raw = re.sub(
        r"(?is)<(script|style|noscript|svg|head)\b.*?</\1>",
        " ",
        raw,
    )
    raw = re.sub(r"(?i)<br\s*/?>", "\n", raw)
    raw = re.sub(
        r"(?i)</(p|div|li|tr|h[1-6])>",
        "\n",
        raw,
    )
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
    raw = re.sub(r"\n\s*\n\s*", "\n\n", raw)
    return raw.strip()


def _ddg_unwrap(href: str) -> str:
    """Unwrap a DuckDuckGo redirect URL."""
    if href.startswith("//"):
        href = "https:" + href

    parsed = urllib.parse.urlparse(href)

    if (
        parsed.netloc.endswith("duckduckgo.com")
        and parsed.path.startswith("/l/")
    ):
        target = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return urllib.parse.unquote(target)

    return href


@dataclass(frozen=True, slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class SearchBackend:
    """API-key-free web search, DuckDuckGo Lite ONLY.

    DuckDuckGo Lite is the sole search provider for this project, by design and
    policy. There is deliberately no switch for Google, Bing, Brave, Tavily,
    SearXNG or any other engine: `ENDPOINT` below is the only URL this class will
    ever contact, and it is the single place the constraint is expressed. The
    self-test asserts that the provider is Lite and that no other host is
    reachable from here.
    """

    PROVIDER = "duckduckgo_lite"
    ENDPOINT = "https://lite.duckduckgo.com/lite/"
    # One-element tuple, kept so any caller that iterates endpoints still works
    # and so "only DuckDuckGo Lite" is verifiable in exactly one location.
    ENDPOINTS = (ENDPOINT,)

    def __init__(self, config: Config):
        self.config = config
        # A Config must never carry a non-Lite backend; even if a stale value
        # somehow arrives, this class still only ever contacts ENDPOINT.
        self.provider = self.PROVIDER

    def search(
        self,
        query: str,
        num_results: int | None = None,
    ) -> list[SearchResult]:
        if not query or not query.strip():
            return []

        count = num_results or self.config.search_results
        count = max(1, min(count, 50))

        return self._search_lite(query.strip(), count)

    def _client(self):
        import httpx

        return httpx.Client(
            timeout=self.config.tool_timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )

    def _search_lite(self, query: str, count: int) -> list[SearchResult]:
        """POST the query to DuckDuckGo Lite and parse its results table."""
        with self._client() as client:
            try:
                response = client.post(
                    self.ENDPOINT,
                    data={"q": query, "kl": "wt-wt"},
                )
                response.raise_for_status()
            except Exception as exc:
                raise RuntimeError(f"DuckDuckGo Lite search failed: {exc}") from exc

            return self._parse_lite(response.text, count)

    @staticmethod
    def _parse_lite(page: str, count: int) -> list[SearchResult]:
        """Parse the DuckDuckGo Lite results page.

        Lite links are DDG redirect anchors (``//duckduckgo.com/l/?uddg=...``)
        or, occasionally, direct URLs. The earlier ``href="http...`` regex missed
        the protocol-relative redirect form entirely, which would have returned
        no results once Lite is the only endpoint. This unwraps every anchor and
        keeps the ones that resolve to a real external page, pairing each with the
        snippet cell that follows it.
        """
        results: list[SearchResult] = []
        seen: set[str] = set()

        # Snippet cells, in document order, to pair with the result links.
        snippets = [
            strip_html(m)
            for m in re.findall(
                r'''class=["'][^"']*\bresult-snippet\b[^"']*["'][^>]*>(.*?)</td''',
                page,
                re.IGNORECASE | re.DOTALL,
            )
        ]
        snippet_idx = 0

        for match in re.finditer(
            r'<a\b[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
            page,
            re.IGNORECASE | re.DOTALL,
        ):
            url = _ddg_unwrap(html.unescape(match.group(1)))
            title = strip_html(match.group(2))

            if not url or not title:
                continue
            if not url.lower().startswith(("http://", "https://")):
                continue

            parsed = urllib.parse.urlparse(url)
            # Skip DuckDuckGo's own navigation/help/settings links.
            if parsed.netloc.endswith("duckduckgo.com"):
                continue
            if url in seen:
                continue
            seen.add(url)

            snippet = snippets[snippet_idx] if snippet_idx < len(snippets) else ""
            snippet_idx += 1

            results.append(SearchResult(title=title, url=url, snippet=snippet))
            if len(results) >= count:
                break

        return results


# Explicitly preserve the original flat-module exports.
__all__ = [
    "MAX_FETCH_BYTES",
    "SearchBackend",
    "SearchResult",
    "TOOL_TRAINING_PREAMBLE",
    "USER_AGENT",
    "_ddg_unwrap",
    "guard_public_url",
    "parallel_map",
    "search_workers",
    "strip_html",
    "thread_task",
]
