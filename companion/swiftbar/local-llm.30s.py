#!/usr/bin/env python3
# <xbar.title>Local LLM status</xbar.title>
# <xbar.version>v1.0</xbar.version>
# <xbar.desc>Model health, queue and one-click control for a self-hosted Local LLM server.</xbar.desc>
# <xbar.dependencies>python3,curl</xbar.dependencies>
# <xbar.var>string(LLM_URL="http://127.0.0.1:8000"): Server URL</xbar.var>
# <swiftbar.hideRunInTerminal>true</swiftbar.hideRunInTerminal>
# <swiftbar.hideDisappearPlugin>true</swiftbar.hideDisappearPlugin>
"""SwiftBar / xbar plugin for the Local LLM server (the no-compile companion).

Every 30 s (the ``.30s.`` in the file name) it reads GET /api/status/summary
and prints the menu-bar line plus a dropdown. Python 3 standard library only.

Credentials, in this order: $LLM_URL and $LLM_API_TOKEN, else the CLI's saved
sign-in at ~/.config/local-llm/credentials.json (read, never written). With
neither, it talks to http://127.0.0.1:8000 without a token, which is right for
a server with sign-in turned off.

The token never appears in what this prints: SwiftBar shows, caches and logs
plugin output. The action lines call this same file back
(``bash=<this file> param1=--action ...``) and it runs curl with the
Authorization header fed through curl's stdin config (``-K -``), so the token
is not in the menu, not on a command line, and not in ``ps``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8000"
CREDENTIALS = Path("~/.config/local-llm/credentials.json")
SUMMARY_PATH = "/api/status/summary"
CURL = os.environ.get("LLM_CURL") or "/usr/bin/curl"
TIMEOUT_S = 6
COLORS = {"green": "#34c759", "yellow": "#ffcc00", "orange": "#ff9500",
          "red": "#ff3b30", "gray": "#8e8e93"}
_URL_KEYS = ("url", "server", "base_url", "server_url")
_TOKEN_KEYS = ("token", "api_token", "access_token")


# --------------------------------------------------------------------------- #
# Credentials                                                                 #
# --------------------------------------------------------------------------- #
def _pick(data: dict, keys: tuple) -> str | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def read_credentials_file(path: Path | None = None) -> tuple[str | None, str | None]:
    """(url, token) from the CLI's credentials file; (None, None) if absent.

    Tolerant of a flat {"url", "token"} object and of
    {"default": url, "servers": {url: {"token": ...}}}.
    """
    target = Path(path or CREDENTIALS).expanduser()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    if not isinstance(data, dict):
        return None, None
    url, token = _pick(data, _URL_KEYS), _pick(data, _TOKEN_KEYS)
    servers = data.get("servers")
    if isinstance(servers, dict) and servers:
        chosen = _pick(data, ("default", "current")) or url or sorted(servers)[0]
        entry = servers.get(chosen)
        if isinstance(entry, dict):
            url = _pick(entry, _URL_KEYS) or chosen
            token = token or _pick(entry, _TOKEN_KEYS)
    return url, token


def normalise_url(raw: str | None) -> str:
    value = (raw or "").strip() or DEFAULT_URL
    if not value.lower().startswith(("http://", "https://")):
        value = "http://" + value
    return value.rstrip("/")


def credentials() -> tuple[str, str | None]:
    """(server URL, token or None): environment first, then the CLI's file."""
    url = os.environ.get("LLM_URL") or None
    token = os.environ.get("LLM_API_TOKEN") or None
    if url is None or token is None:
        file_url, file_token = read_credentials_file()
        url = url or file_url
        token = token or file_token
    return normalise_url(url), token


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #
def clean(text: object, limit: int = 120) -> str:
    """One menu line: no pipe (xbar's separator), no newline, bounded length."""
    value = " ".join(str(text if text is not None else "").split())
    return value.replace("|", "/")[:limit]


def plugin_path() -> str:
    return os.environ.get("SWIFTBAR_PLUGIN_PATH") or os.path.abspath(sys.argv[0] or __file__)


def action_params(action: str, *args: str) -> str:
    """bash=/param lines that call this file back; quoted for paths with spaces."""
    parts = [f'bash="{plugin_path()}"', "param1=--action", f"param2={action}"]
    for index, value in enumerate(args, start=3):
        parts.append(f"param{index}={value}")
    parts += ["terminal=false", "refresh=true"]
    return " ".join(parts)


def _fmt_num(value: object, pattern: str) -> str | None:
    try:
        return pattern % float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def render(summary: dict, url: str) -> list[str]:
    """The plugin output (first line is the menu bar) for one summary."""
    indicator = summary.get("indicator") or {}
    color = COLORS.get(str(indicator.get("color") or "gray"), COLORS["gray"])
    lines = [f"● {clean(indicator.get('label') or 'unknown', 24)} | color={color}", "---"]
    model = summary.get("model") or {}
    name = model.get("alias") or model.get("id") or "unknown model"
    lines.append(clean(f"Model: {name} ({model.get('status') or 'unknown'})"))
    if model.get("adapter"):
        lines.append(clean(f"Adapter: {model['adapter']}"))
    backend = f"Backend: {model.get('backend') or 'unknown'}"
    caps = [str(c) for c in (model.get("capabilities") or [])]
    if caps:
        backend += f" ({', '.join(caps)})"
    lines.append(clean(backend))
    queue = summary.get("queue") or {}
    queue_line = f"Queue: {queue.get('running') or 0} running, {queue.get('waiting') or 0} waiting"
    if queue.get("limit") is not None:
        queue_line += f", limit {queue['limit']}"
    lines.append(clean(queue_line))
    metrics = summary.get("metrics")
    if isinstance(metrics, dict):
        line = f"Last hour: {metrics.get('requests') or 0} requests"
        ttft = _fmt_num(metrics.get("ttft_p50_ms"), "%.0f")
        tps = _fmt_num(metrics.get("decode_tps_p50"), "%.1f")
        if ttft is not None:
            line += f", TTFT {ttft} ms"
        if tps is not None:
            line += f", {tps} tok/s"
        lines.append(clean(line))
    nodes = summary.get("nodes")
    if isinstance(nodes, list) and len(nodes) > 1:
        healthy = sum(1 for n in nodes if n.get("state") == "healthy")
        lines.append(f"Nodes: {healthy} of {len(nodes)} healthy")
        for node in nodes[:8]:
            role = f", {node['role']}" if node.get("role") else ""
            lines.append("--" + clean(f"{node.get('name') or '?'}: "
                                      f"{node.get('state') or 'unknown'}{role}"))
    tasks = summary.get("tasks")
    running_ids: list[str] = []
    if isinstance(tasks, dict):
        running_ids = [str(t) for t in (tasks.get("running_ids") or [])]
        lines.append(f"Tasks running: {tasks.get('running') or len(running_ids)}")
    retrain = summary.get("retrain")
    if isinstance(retrain, dict) and retrain.get("running"):
        lines.append(clean(f"Retraining: {retrain.get('message') or 'running'}"))
    controls = summary.get("controls") or {}
    pending = int((summary.get("approvals") or {}).get("pending") or 0)

    lines.append("---")
    lines.append(f"Open Web UI | href={url}")
    if pending:
        lines.append(f"Approvals pending: {pending} | href={url}")
    if controls.get("restart_model"):
        lines.append(f"Restart Model... | {action_params('restart')}")
    if controls.get("cancel_tasks") and running_ids:
        lines.append(f"Cancel Running Tasks ({len(running_ids)})... | "
                     f"{action_params('cancel-tasks')}")
    lines.append(f"Copy API URL | {action_params('copy-api-url')}")
    lines.append("Refresh | refresh=true")
    return lines


def render_problem(label: str, detail: str, url: str, color: str = "gray") -> list[str]:
    return [f"● {clean(label, 24)} | color={COLORS.get(color, COLORS['gray'])}", "---",
            clean(f"Server: {url}"), clean(detail, 200), "---",
            f"Open Web UI | href={url}", "Refresh | refresh=true"]


# --------------------------------------------------------------------------- #
# HTTP                                                                        #
# --------------------------------------------------------------------------- #
def fetch_summary(url: str, token: str | None) -> tuple[int, dict | None, str]:
    """(HTTP status or 0 when unreachable, parsed body, error text)."""
    request = urllib.request.Request(url + SUMMARY_PATH, headers={
        "Accept": "application/json", "User-Agent": "local-llm-swiftbar/1.0",
        **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return response.status, json.loads(response.read().decode("utf-8")), ""
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            body = None
        detail = (body or {}).get("detail") or (body or {}).get("error") or exc.reason
        return exc.code, None, str(detail)
    except Exception as exc:  # noqa: BLE001  (refused, timeout, DNS, bad JSON)
        return 0, None, str(getattr(exc, "reason", None) or exc)


def curl_post(url: str, token: str | None, path: str) -> int:
    """POST with curl; the Authorization header goes in through stdin, not argv."""
    config = ""
    if token:
        safe = token.replace("\\", "\\\\").replace('"', '\\"')
        config = f'header = "Authorization: Bearer {safe}"\n'
    proc = subprocess.run(
        [CURL, "-fsS", "-m", "15", "-o", "/dev/null", "-K", "-", "-X", "POST",
         "-H", "Content-Type: application/json", "--data", "{}", url + path],
        input=config, text=True, capture_output=True)
    if proc.returncode != 0:
        sys.stderr.write((proc.stderr or "").strip()[:300] + "\n")
    return proc.returncode


def confirm(question: str, button: str) -> bool:
    """A native yes/no dialog (osascript); False when dismissed or unavailable."""
    script = (f'display dialog "{question}" buttons {{"Cancel", "{button}"}} '
              f'default button "{button}" with icon caution with title "Local LLM"')
    try:
        return subprocess.run(["/usr/bin/osascript", "-e", script],
                              capture_output=True, timeout=120).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def run_action(action: str, *, yes: bool = False) -> int:
    url, token = credentials()
    if action == "copy-api-url":
        subprocess.run(["/usr/bin/pbcopy"], input=url + "/v1", text=True)
        return 0
    if action == "restart":
        if not yes and not confirm("Restart the model? Requests in flight fail.", "Restart"):
            return 1
        return curl_post(url, token, "/api/model/restart")
    if action == "cancel-tasks":
        status, summary, _ = fetch_summary(url, token)
        ids = [str(t) for t in (((summary or {}).get("tasks") or {}).get("running_ids") or [])]
        if status != 200 or not ids:
            return 1
        if not yes and not confirm(f"Cancel {len(ids)} running task(s)?", "Cancel Tasks"):
            return 1
        failed = 0
        for task_id in ids:
            path = "/api/tasks/" + urllib.parse.quote(task_id, safe="") + "/cancel"
            failed += 1 if curl_post(url, token, path) else 0
        return 1 if failed else 0
    sys.stderr.write(f"unknown action {action!r}\n")
    return 2


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "--action":
        return run_action(argv[1], yes="--yes" in argv[2:])
    url, token = credentials()
    status, summary, error = fetch_summary(url, token)
    if status == 200 and isinstance(summary, dict):
        lines = render(summary, url)
    elif status == 401:
        lines = render_problem("sign in", "The server wants a token: set LLM_API_TOKEN or "
                               "sign in with the CLI.", url)
    elif status == 403:
        lines = render_problem("no access", error or "This token lacks the status scope.", url)
    elif status == 0:
        lines = render_problem("offline", f"Cannot reach the server: {error}", url, "red")
    else:
        lines = render_problem("error", f"HTTP {status}: {error}", url, "red")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
