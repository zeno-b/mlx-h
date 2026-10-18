"""The named capability tools: browse_page, code_interpreter, git, spreadsheet,
pdf_extract, image_analyze, image_generate, email, calendar, contacts,
http_request, database, task_scheduler and notify.

Twenty capabilities are offered to agent profiles and roles (config.py
CAPABILITY_GROUPS). Several already had a tool with a different name
(web_search, run_shell, the file tools, create_document, the memory tools,
search_docs), and agent.py hard-codes those names in its lanes, so they keep
them and the capability maps onto them. This module adds the rest.

Why registration is opt-in: every registered tool's description is re-sent in
the agent's system prompt on every step, which is prefill cost on an 8 GB Mac
and changes the cached prompt prefix. So a tool here is registered only when
the registry's allowlist names it (a profile, role, task or subagent granted
its capability) or, for the unrestricted default agent, when EXTRA_TOOLS lists
it. With neither, the default agent is exactly what it was.

Each tool declares its risk class (policy.py asks before "write", "exec" and
"send" calls when approvals are on) and its capability key, and the ones that
can both read and act are split in two (email / email_send, git / git_commit,
calendar / calendar_create, spreadsheet / spreadsheet_write), because a risk is
per tool: one "email" tool would make every mailbox search ask, or every send
not ask. User-visible failures raise ValueError so ToolRegistry.call records a
FAILED row, like the built-ins.

This module never imports tools.py (tools.py imports it): register_extra_tools
receives the registry and the Tool class from the one hook line in
ToolRegistry._register_defaults.
"""

from __future__ import annotations

import asyncio
import base64
import csv
import io
import json
import logging
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .database import *  # noqa: F401,F403
from .turnctx import *  # noqa: F401,F403
from .websearch import *  # noqa: F401,F403
from .documents import *  # noqa: F401,F403


# Registration order is fixed (prompt prefix stability), and this is it.
EXTRA_TOOL_NAMES = (
    "browse_page", "http_request", "code_interpreter", "git", "git_commit",
    "spreadsheet", "spreadsheet_write", "pdf_extract", "image_analyze",
    "image_generate", "database", "email", "email_send", "calendar",
    "calendar_create", "contacts", "task_scheduler", "notify",
)

_TX_HTTP_METHODS = ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS")
_TX_DROP_HEADERS = {"host", "content-length", "transfer-encoding", "connection"}
_TX_LINK = re.compile(r"(?is)<a\s[^>]*?href\s*=\s*([\"'])(.*?)\1[^>]*>(.*?)</a>")
_TX_IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                   ".gif": "image/gif", ".webp": "image/webp"}
_TX_GRAPH = "https://graph.microsoft.com/v1.0"
_TX_TOKEN_CACHE: dict[str, tuple[str, float]] = {}
_TX_TOKEN_LOCK = threading.Lock()
_TX_SQL_START = ("select", "with", "explain", "values", "pragma")
_TX_SQL_PRAGMAS = {"table_info", "table_xinfo", "index_list", "index_info",
                   "foreign_key_list", "table_list", "database_list", "user_version"}
_TX_SQL_WRITE = re.compile(
    r"(?i)\b(insert|update|delete|create|drop|alter|attach|detach|vacuum|reindex)\b")
_TX_GIT_REV = re.compile(r"^[A-Za-z0-9_./~^@{}:+-]{1,120}$")
_TX_GIT_REFUSED = {"push", "pull", "fetch", "remote", "reset", "checkout", "switch",
                   "rebase", "merge", "clean", "gc", "config", "submodule", "stash",
                   "restore", "rm", "mv", "tag", "cherry-pick", "revert", "am", "apply"}
_TX_MAX_TASKS_PER_USER = 20
_TX_MIN_INTERVAL_MIN = 5
_TX_NOTIFY_SCHEMA = "notifications"


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
def _tx_wanted(registry: Any) -> set[str]:
    """The extra tools this registry should have, before prerequisites."""
    allow = {n.strip() for n in (registry.config.agent_tools or "").split(",") if n.strip()}
    if allow:
        return set(EXTRA_TOOL_NAMES) & allow
    raw = str(getattr(registry.config, "extra_tools", "") or "").strip()
    if raw.lower() in ("all", "*"):
        return set(EXTRA_TOOL_NAMES)
    return {n.strip() for n in raw.split(",") if n.strip()} & set(EXTRA_TOOL_NAMES)


def register_extra_tools(registry: Any, tool_cls: Any) -> list[str]:
    """Add the wanted extra tools whose prerequisites hold. Returns their names.

    Never raises: a tool that cannot be built is a missing capability, logged,
    exactly like an MCP server that will not start.
    """
    try:
        wanted = _tx_wanted(registry)
    except Exception as exc:  # noqa: BLE001
        log(f"extra tools skipped: {exc}", logging.WARNING)
        return []
    added: list[str] = []
    for name in EXTRA_TOOL_NAMES:
        if name not in wanted or name in getattr(registry, "_tools", {}):
            continue
        try:
            tool = _TX_BUILDERS[name](registry, tool_cls)
        except Exception as exc:  # noqa: BLE001
            log(f"extra tool {name!r} could not be built: {exc}", logging.WARNING)
            continue
        if tool is not None:
            registry._add(tool)
            added.append(name)
    return added


def _tx_feature(module: str, name: str) -> Any:
    """`name` from another package module, imported on demand (or bundled)."""
    found = globals().get(name)
    if found is not None:
        return found
    package = __package__ or ""
    if not package:
        return None
    try:
        import importlib
        mod = importlib.import_module(f"{package}.{module}")
    except ImportError:
        return None
    return getattr(mod, name, None)


def _tx_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        number = int(float(str(value).strip())) if value not in (None, "") else default
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _tx_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("1", "true", "yes", "on", "y")


def _tx_json_arg(value: Any, what: str) -> Any:
    """An argument that may arrive as JSON text or as the structure itself."""
    if value in (None, ""):
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except ValueError as exc:
        raise ValueError(f"{what} must be JSON: {exc}") from None


def _tx_role(registry: Any) -> str:
    """The acting role: the turn's, else the user row's; local runs are admin."""
    turn = current_turn()
    if turn is not None and getattr(turn, "role", None):
        return str(turn.role)
    uid = get_acting_user()
    if uid in (None, SENTINEL_LOCAL_USER):
        return "admin"
    db = getattr(registry, "db", None)
    try:
        row = db.get_user(uid) if db is not None else None
    except Exception:  # noqa: BLE001
        row = None
    return str((row or {}).get("role") or "user")


def _tx_cap(registry: Any, text: str) -> str:
    limit = int(getattr(registry.config, "tool_raw_chars", 20000) or 20000)
    return text if len(text) <= limit else text[:limit] + "\n[truncated]"


# --------------------------------------------------------------------------- #
# HTTP: browse_page and http_request
# --------------------------------------------------------------------------- #
def _tx_fetch(registry: Any, method: str, url: str, *, headers: dict | None = None,
              content: bytes | None = None, json_body: Any = None,
              byte_cap: int = 400_000, follow: bool = True) -> dict:
    """One bounded HTTP exchange, SSRF-checked on every hop.

    Redirects are followed by hand (GET/HEAD only) so each Location is checked
    before it is connected to, the same rule fetch_url follows: httpx's own
    redirect handling would connect to a 30x target like 169.254.169.254
    before anything could look at it.
    """
    import httpx
    if not str(url).lower().startswith(("http://", "https://")):
        raise ValueError("url must start with http:// or https://")
    guard = not getattr(registry.config, "allow_local_fetch", False)
    current = str(url)
    with httpx.Client(timeout=registry.config.tool_timeout, follow_redirects=False,
                      headers={"User-Agent": USER_AGENT}) as client:
        for _hop in range(6):
            if guard:
                guard_public_url(current)
            with client.stream(method, current, headers=headers, content=content,
                               json=json_body) as resp:
                if (follow and method in ("GET", "HEAD") and resp.is_redirect
                        and resp.headers.get("location")):
                    current = str(resp.url.join(resp.headers["location"]))
                    continue
                chunks: list[bytes] = []
                total = 0
                if method != "HEAD":
                    for chunk in resp.iter_bytes():
                        chunks.append(chunk)
                        total += len(chunk)
                        if total >= byte_cap:
                            break
                return {"status": resp.status_code, "reason": resp.reason_phrase,
                        "url": str(resp.url), "headers": dict(resp.headers),
                        "data": b"".join(chunks)[:byte_cap], "truncated": total >= byte_cap,
                        "encoding": resp.encoding or "utf-8"}
        raise ValueError("too many redirects")


def _tx_links(html: str, base: str, limit: int = 60) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for _q, href, label in _TX_LINK.findall(html):
        href = href.strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:", "data:")):
            continue
        target = urllib.parse.urljoin(base, href).split("#", 1)[0]
        if not target.lower().startswith(("http://", "https://")) or target in seen:
            continue
        seen.add(target)
        out.append((strip_html(label).strip()[:80] or target, target))
        if len(out) >= limit:
            break
    return out


def _tx_same_site(a: str, b: str) -> bool:
    def host(u: str) -> str:
        h = (urllib.parse.urlparse(u).hostname or "").lower()
        return h[4:] if h.startswith("www.") else h
    return bool(host(a)) and host(a) == host(b)


def _tx_read_page(registry: Any, url: str) -> tuple[str, str, str, list]:
    """(title, final url, readable text, links) of one page."""
    got = _tx_fetch(registry, "GET", url, byte_cap=min(MAX_FETCH_BYTES, 1_500_000))
    if got["status"] >= 400:
        raise ValueError(f"{url} answered HTTP {got['status']} {got['reason']}")
    kind = (got["headers"].get("content-type") or "").lower()
    data: bytes = got["data"]
    final = got["url"]
    if "pdf" in kind or final.lower().split("?", 1)[0].endswith(".pdf"):
        extract = getattr(registry, "_extract_pdf", None)
        text = extract(io.BytesIO(data), registry.config.tool_raw_chars) if extract else "[pdf]"
        return "", final, text, []
    if kind and not any(k in kind for k in ("text/", "html", "xml", "json", "javascript", "csv")):
        return "", final, f"[skipped: unsupported content type {kind}]", []
    if b"\x00" in data[:4096]:
        return "", final, "[skipped: binary content]", []
    body = data.decode(got["encoding"], "replace")
    is_html = "html" in kind or "<" in body[:300]
    title = ""
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", body)
    if match:
        title = strip_html(match.group(1)).strip()
    text = strip_html(body) if is_html else body
    links = _tx_links(body, final) if is_html else []
    if got["truncated"]:
        text += "\n[truncated: the page exceeded the size cap]"
    return title, final, text, links


def _tx_browse_page(registry: Any, url: str, follow: Any = None, max_chars: Any = None) -> str:
    cap = _tx_int(max_chars, registry.config.tool_raw_chars, 500, registry.config.tool_raw_chars)
    title, final, text, links = _tx_read_page(registry, url)
    parts = [f"{title}\n{final}".strip(), "", text.strip()[:cap]]
    if follow not in (None, ""):
        wanted = str(follow).strip()
        target = None
        absolute = urllib.parse.urljoin(final, wanted)
        for label, href in links:
            if href == absolute or wanted.lower() in label.lower():
                target = href
                break
        if target is None and absolute.lower().startswith(("http://", "https://")):
            target = absolute
        if target is None:
            raise ValueError(f"no link on {final} matches {wanted!r}")
        if not _tx_same_site(final, target):
            raise ValueError(f"refusing to follow {target}: only links on the same site "
                             f"({urllib.parse.urlparse(final).hostname}) are followed")
        t2, f2, text2, links = _tx_read_page(registry, target)
        parts += ["", f"--- followed: {t2 or f2}\n{f2}", "", text2.strip()[:cap]]
    if links:
        parts += ["", "Links:"] + [f"[{i}] {label} -> {href}"
                                   for i, (label, href) in enumerate(links[:40], 1)]
    return _tx_cap(registry, "\n".join(parts).strip())


def _tx_headers_arg(value: Any) -> dict[str, str]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        raw = value
    else:
        text = str(value).strip()
        if text.startswith("{"):
            raw = _tx_json_arg(text, "headers")
        else:
            raw = {}
            for line in text.splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    raw[key.strip()] = val.strip()
    if not isinstance(raw, dict):
        raise ValueError("headers must be a JSON object or 'Name: value' lines")
    return {str(k): str(v) for k, v in raw.items()
            if str(k).strip() and str(k).strip().lower() not in _TX_DROP_HEADERS}


def _tx_http_request(registry: Any, url: str, method: Any = "GET", headers: Any = None,
                     json: Any = None, body: Any = None, max_bytes: Any = None) -> str:  # noqa: A002
    verb = str(method or "GET").strip().upper()
    if verb not in _TX_HTTP_METHODS:
        raise ValueError(f"method must be one of {', '.join(_TX_HTTP_METHODS)}")
    send_headers = _tx_headers_arg(headers)
    payload = _tx_json_arg(json, "json") if json not in (None, "") else None
    content = None
    if payload is None and body not in (None, ""):
        content = str(body).encode("utf-8")
    if (payload is not None or content is not None) and verb in ("GET", "HEAD"):
        raise ValueError(f"a {verb} request cannot carry a body; use POST/PUT/PATCH")
    cap = _tx_int(max_bytes, 200_000, 1_000, MAX_FETCH_BYTES)
    got = _tx_fetch(registry, verb, url, headers=send_headers or None, content=content,
                    json_body=payload, byte_cap=cap, follow=verb in ("GET", "HEAD"))
    lines = [f"HTTP {got['status']} {got['reason']}", f"URL: {got['url']}"]
    for name in ("content-type", "content-length", "location", "retry-after"):
        if got["headers"].get(name):
            lines.append(f"{name}: {got['headers'][name]}")
    data: bytes = got["data"]
    kind = (got["headers"].get("content-type") or "").lower()
    if verb == "HEAD" or not data:
        lines.append("\n(no body)")
    elif b"\x00" in data[:4096] or (kind and not any(
            k in kind for k in ("text", "json", "xml", "html", "javascript", "csv", "form"))):
        lines.append(f"\n[binary body: {len(data)} bytes]")
    else:
        lines.append("\n" + data.decode(got["encoding"], "replace"))
    if got["truncated"]:
        lines.append(f"\n[truncated at {cap} bytes]")
    return _tx_cap(registry, "\n".join(lines))


# --------------------------------------------------------------------------- #
# code_interpreter: a Python session that survives between calls
# --------------------------------------------------------------------------- #
# The runner executes one cell in a fresh, sandboxed interpreter (the same
# _exec run_python uses), but first restores the session: top-level imports
# and def/class statements are re-run from their recorded source (functions
# and modules do not pickle), then every picklable variable is loaded. After
# the cell, the variables are pickled again. A cell whose last statement is an
# expression prints its repr, like a REPL.
_TX_RUNNER = r'''
import ast, json, os, pickle, sys, traceback, types
state_path, defs_path, cell_path = sys.argv[1:4]
ns = {"__name__": "__main__", "__builtins__": __builtins__}
notes = []
try:
    with open(defs_path) as fh:
        defs = json.load(fh)
except FileNotFoundError:
    defs = {"imports": [], "defs": {}}
except Exception as exc:
    defs = {"imports": [], "defs": {}}
    notes.append("[session definitions could not be restored: %s]" % exc)
for src in defs.get("imports", []):
    try:
        exec(src, ns)
    except Exception as exc:
        notes.append("[could not re-run %r: %s]" % (src, exc))
for name, src in defs.get("defs", {}).items():
    try:
        exec(src, ns)
    except Exception as exc:
        notes.append("[could not restore %s: %s]" % (name, exc))
try:
    with open(state_path, "rb") as fh:
        ns.update(pickle.load(fh))
except FileNotFoundError:
    pass
except Exception as exc:
    notes.append("[session variables could not be restored: %s]" % exc)
with open(cell_path) as fh:
    code = fh.read()
try:
    tree = ast.parse(code)
except SyntaxError:
    traceback.print_exc()
    sys.exit(1)
lines = code.splitlines()
def source_of(node):
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return "\n".join(lines[start - 1:node.end_lineno])
new_imports, new_defs = [], {}
for node in tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        new_imports.append(source_of(node))
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        new_defs[node.name] = source_of(node)
last = None
if tree.body and isinstance(tree.body[-1], ast.Expr):
    last = ast.Expression(tree.body.pop().value)
ok = True
try:
    exec(compile(tree, "<cell>", "exec"), ns)
    if last is not None:
        value = eval(compile(last, "<cell>", "eval"), ns)
        if value is not None:
            print(repr(value))
except BaseException:
    traceback.print_exc()
    ok = False
if ok:
    imports = [i for i in defs.get("imports", []) if i not in new_imports] + new_imports
    defs["imports"] = imports[-200:]
    merged = dict(defs.get("defs", {}))
    merged.update(new_defs)
    defs["defs"] = merged
defined = set(defs.get("defs", {}))
keep, skipped = {}, []
for key, value in list(ns.items()):
    if key.startswith("__") or isinstance(value, types.ModuleType):
        continue
    if key in defined and (callable(value) or isinstance(value, type)):
        continue
    try:
        pickle.dumps(value)
    except Exception:
        skipped.append(key)
        continue
    keep[key] = value
with open(state_path + ".tmp", "wb") as fh:
    pickle.dump(keep, fh)
os.replace(state_path + ".tmp", state_path)
with open(defs_path + ".tmp", "w") as fh:
    json.dump(defs, fh)
os.replace(defs_path + ".tmp", defs_path)
if skipped:
    print("[not kept between calls (not picklable): %s]" % ", ".join(sorted(skipped)),
          file=sys.stderr)
for note in notes:
    print(note, file=sys.stderr)
sys.exit(0 if ok else 1)
'''


def _tx_session_key() -> str:
    raw = get_acting_conversation() or ""
    if not raw:
        turn = current_turn()
        raw = getattr(turn, "conversation_id", None) or ""
    key = re.sub(r"[^A-Za-z0-9_-]", "_", str(raw))[:64]
    return key or "default"


def _tx_code_interpreter(registry: Any, code: str, reset: Any = None) -> str:
    root = registry._root()
    rel = Path(".cache") / "code_interpreter" / _tx_session_key()
    folder = root / rel
    folder.mkdir(parents=True, exist_ok=True)
    if _tx_bool(reset):
        for name in ("state.pkl", "defs.json"):
            try:
                (folder / name).unlink()
            except FileNotFoundError:
                pass
    (folder / "runner.py").write_text(_TX_RUNNER)
    (folder / "cell.py").write_text(str(code))
    # Relative paths: the command runs in the root, which is also /work in
    # the Docker backend, so the same command line works for both.
    return registry._exec([sys.executable, str(rel / "runner.py"), str(rel / "state.pkl"),
                           str(rel / "defs.json"), str(rel / "cell.py")])


# --------------------------------------------------------------------------- #
# git: read-mostly, never push
# --------------------------------------------------------------------------- #
def _tx_git_run(registry: Any, args: list[str], *, identity: bool = False) -> tuple[int, str]:
    root = registry._root()
    cmd = ["git", "--no-pager", "-c", "color.ui=never", "-c", "core.pager=cat",
           # The repository's own hooks are code the agent did not choose to
           # run; a commit made through this tool does not execute them.
           "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false"]
    if identity:
        cmd += ["-c", "user.name=Local LLM agent", "-c", "user.email=agent@localhost"]
    env = registry._sandbox_env({"GIT_OPTIONAL_LOCKS": "0"})
    try:
        proc = subprocess.run(cmd + args, cwd=str(root), env=env, capture_output=True,
                              text=True, errors="replace", stdin=subprocess.DEVNULL,
                              timeout=registry.config.tool_timeout)
    except FileNotFoundError:
        raise ValueError("git is not installed on the server") from None
    except subprocess.TimeoutExpired:
        raise ValueError(f"git timed out after {registry.config.tool_timeout}s") from None
    out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr.strip() else "")
    return proc.returncode, out.strip()


def _tx_git_repo(registry: Any) -> None:
    """Refuse unless the acting root is itself the top of a git repository.

    A root inside a bigger repository would let status/diff/log show (and a
    commit include) files outside the folder this user's tools are confined to.
    """
    code, out = _tx_git_run(registry, ["rev-parse", "--show-toplevel"])
    if code != 0:
        raise ValueError("the working folder is not a git repository")
    try:
        top = Path(out.splitlines()[0].strip()).resolve()
    except Exception:  # noqa: BLE001
        raise ValueError("could not locate the repository") from None
    if top != registry._root().resolve():
        raise ValueError("the repository starts above the working folder; git is only "
                         "available when the working folder is the repository root")


def _tx_git_paths(registry: Any, raw: Any) -> list[str]:
    if raw in (None, ""):
        return []
    items = raw if isinstance(raw, list) else re.split(r"[,\n]+|\s+(?=\S)", str(raw))
    out = []
    root = registry._root()
    for item in items:
        item = str(item).strip()
        if not item:
            continue
        target = registry._resolve(item)
        out.append(str(target.relative_to(root)) if target != root else ".")
    return out


def _tx_git_rev(value: Any) -> str:
    rev = str(value or "").strip()
    if not rev:
        return ""
    if rev.startswith("-") or not _TX_GIT_REV.match(rev):
        raise ValueError(f"not a revision: {rev!r}")
    return rev


def _tx_git(registry: Any, action: str, path: Any = None, rev: Any = None,
            staged: Any = None, limit: Any = None) -> str:
    verb = str(action or "").strip().lower()
    if verb in _TX_GIT_REFUSED or verb in ("add", "commit"):
        raise ValueError(_tx_git_refusal(verb))
    _tx_git_repo(registry)
    paths = _tx_git_paths(registry, path)
    revision = _tx_git_rev(rev)
    if verb == "status":
        args = ["status", "--short", "--branch"]
    elif verb == "diff":
        args = ["diff", "--stat", "--patch"] + (["--cached"] if _tx_bool(staged) else [])
        args += [revision] if revision else []
    elif verb == "log":
        args = ["log", f"-n{_tx_int(limit, 20, 1, 200)}", "--date=short",
                "--format=%h %ad %an  %s%d"] + ([revision] if revision else [])
    elif verb == "show":
        args = ["show", "--stat", "--patch", revision or "HEAD"]
    elif verb in ("branch", "branches"):
        args = ["branch", "--list", "-vv"]
        paths = []
    else:
        raise ValueError("action must be status, diff, log, show or branch "
                         "(stage and commit with git_commit)")
    code, out = _tx_git_run(registry, args + (["--", *paths] if paths else []))
    if code != 0:
        raise ValueError(out or f"git {verb} failed")
    return registry._cap_output(out or "(nothing)")


def _tx_git_refusal(verb: str) -> str:
    if verb == "push":
        return ("git push is refused: review the commit and push it yourself. The agent "
                "never publishes work on its own.")
    if verb in ("add", "commit"):
        return f"use git_commit to {verb}"
    return f"git {verb} is not available to the agent (it can change or discard work)"


def _tx_git_commit(registry: Any, action: str, paths: Any = None, message: Any = None) -> str:
    verb = str(action or "").strip().lower()
    if verb != "add" and verb != "commit":
        raise ValueError(_tx_git_refusal(verb) if verb in _TX_GIT_REFUSED
                         else "action must be add or commit")
    _tx_git_repo(registry)
    if verb == "add":
        targets = _tx_git_paths(registry, paths)
        if not targets:
            raise ValueError("name the paths to stage (\".\" for everything)")
        code, out = _tx_git_run(registry, ["add", "--", *targets])
        if code != 0:
            raise ValueError(out or "git add failed")
        _code, status = _tx_git_run(registry, ["status", "--short"])
        return "Staged.\n" + (status or "(nothing to commit)")
    text = str(message or "").strip()
    if not text:
        raise ValueError("a commit message is required")
    _c, name = _tx_git_run(registry, ["config", "user.email"])
    targets = _tx_git_paths(registry, paths)
    if targets:
        code, out = _tx_git_run(registry, ["add", "--", *targets])
        if code != 0:
            raise ValueError(out or "git add failed")
    code, out = _tx_git_run(registry, ["commit", "-m", text[:2000]], identity=not name.strip())
    if code != 0:
        raise ValueError(out or "git commit failed")
    return out + "\n(committed locally; nothing was pushed)"


# --------------------------------------------------------------------------- #
# spreadsheet: cells and ranges of .xlsx / .csv
# --------------------------------------------------------------------------- #
_TX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
          "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
          "pr": "http://schemas.openxmlformats.org/package/2006/relationships"}


def _tx_col_index(letters: str) -> int:
    index = 0
    for ch in letters.upper():
        index = index * 26 + (ord(ch) - 64)
    return index - 1


def _tx_col_name(index: int) -> str:
    name, index = "", index + 1
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def _tx_cell_ref(ref: str) -> tuple[int, int]:
    match = re.match(r"^\$?([A-Za-z]{1,3})\$?(\d{1,7})$", ref.strip())
    if not match:
        raise ValueError(f"not a cell reference: {ref!r}")
    return int(match.group(2)) - 1, _tx_col_index(match.group(1))


def _tx_xlsx_sheets(path: Path) -> list[tuple[str, list[list[Any]]]]:
    """Every sheet of an .xlsx as (name, rows), with the standard library only.

    openpyxl is optional here and usually absent; the reader in tools.py needs
    it. The format is a zip of XML, and values, shared strings and inline
    strings are all a reader needs (formulas come back as their cached value).
    """
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        raise ValueError(f"{path.name} is not a valid .xlsx file") from None
    with archive:
        names = set(archive.namelist())
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", _TX_NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{_TX_NS['m']}}}t")))
        rels = {}
        if "xl/_rels/workbook.xml.rels" in names:
            for rel in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels")):
                target = rel.get("Target") or ""
                target = target.lstrip("/")
                rels[rel.get("Id")] = target if target.startswith("xl/") else f"xl/{target}"
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        out = []
        for sheet in book.iter(f"{{{_TX_NS['m']}}}sheet"):
            rid = sheet.get(f"{{{_TX_NS['r']}}}id")
            part = rels.get(rid) or ""
            if part not in names:
                continue
            grid: dict[tuple[int, int], Any] = {}
            data = ET.fromstring(archive.read(part))
            for cell in data.iter(f"{{{_TX_NS['m']}}}c"):
                ref = cell.get("r")
                if not ref:
                    continue
                kind = cell.get("t") or "n"
                value_el = cell.find("m:v", _TX_NS)
                raw = value_el.text if value_el is not None else None
                if kind == "s" and raw is not None:
                    value: Any = shared[int(raw)] if int(raw) < len(shared) else ""
                elif kind == "inlineStr":
                    value = "".join(t.text or "" for t in cell.iter(f"{{{_TX_NS['m']}}}t"))
                elif kind == "b":
                    value = raw == "1"
                elif kind in ("str", "e"):
                    value = raw or ""
                elif raw is None:
                    continue
                else:
                    try:
                        number = float(raw)
                        value = int(number) if number.is_integer() else number
                    except ValueError:
                        value = raw
                grid[_tx_cell_ref(ref)] = value
            rows_n = max((r for r, _c in grid), default=-1) + 1
            cols_n = max((c for _r, c in grid), default=-1) + 1
            rows = [[grid.get((r, c), "") for c in range(cols_n)] for r in range(rows_n)]
            out.append((sheet.get("name") or f"Sheet{len(out) + 1}", rows))
        return out


def _tx_read_sheets(target: Path) -> list[tuple[str, list[list[Any]]]]:
    suffix = target.suffix.lower()
    if suffix in (".xlsx", ".xlsm"):
        return _tx_xlsx_sheets(target)
    if suffix in (".csv", ".tsv"):
        text = target.read_text(encoding="utf-8", errors="replace")
        delim = "\t" if suffix == ".tsv" else ","
        return [(target.stem, [list(r) for r in csv.reader(io.StringIO(text), delimiter=delim)])]
    raise ValueError("spreadsheet reads .xlsx, .xlsm, .csv and .tsv files")


def _tx_spreadsheet(registry: Any, path: str, sheet: Any = None, range: Any = None,  # noqa: A002
                    max_rows: Any = None) -> str:
    target = registry._resolve(path)
    if not target.is_file():
        raise ValueError(f"no such file: {path}")
    sheets = _tx_read_sheets(target)
    if not sheets:
        raise ValueError(f"{path} has no sheets")
    if sheet in (None, "") and range in (None, ""):
        lines = [f"{registry._rel(target)}: {len(sheets)} sheet(s)"]
        for name, rows in sheets:
            width = max((len(r) for r in rows), default=0)
            lines.append(f"- {name}: {len(rows)} rows x {width} cols"
                         + (f" (A1:{_tx_col_name(width - 1)}{len(rows)})" if rows and width else ""))
        name, rows = sheets[0]
        lines.append("")
        lines.append(_tx_render_grid(name, rows, 0, 0, min(len(rows), 20) - 1,
                                     max((len(r) for r in rows), default=1) - 1))
        return _tx_cap(registry, "\n".join(lines))
    chosen = None
    if sheet not in (None, ""):
        wanted = str(sheet).strip()
        for index, (name, rows) in enumerate(sheets, 1):
            if name.lower() == wanted.lower() or wanted == str(index):
                chosen = (name, rows)
                break
        if chosen is None:
            raise ValueError(f"no sheet {wanted!r}; sheets: {', '.join(n for n, _ in sheets)}")
    else:
        chosen = sheets[0]
    name, rows = chosen
    width = max((len(r) for r in rows), default=1)
    r0, c0, r1, c1 = 0, 0, len(rows) - 1, width - 1
    if range not in (None, ""):
        spec = str(range).strip()
        if "!" in spec:
            sheet_part, spec = spec.split("!", 1)
            if sheet in (None, ""):
                return _tx_spreadsheet(registry, path, sheet_part.strip("'\""), spec, max_rows)
        start, _, end = spec.partition(":")
        r0, c0 = _tx_cell_ref(start)
        r1, c1 = _tx_cell_ref(end) if end else (r0, c0)
        r0, r1 = sorted((r0, r1))
        c0, c1 = sorted((c0, c1))
    r1 = min(r1, r0 + _tx_int(max_rows, 200, 1, 5000) - 1)
    return _tx_cap(registry, _tx_render_grid(name, rows, r0, c0, r1, c1))


def _tx_render_grid(name: str, rows: list[list[Any]], r0: int, c0: int, r1: int, c1: int) -> str:
    if r1 < r0 or c1 < c0 or not rows:
        return f"# sheet: {name} (empty)"
    head = f"# sheet: {name}, {_tx_col_name(c0)}{r0 + 1}:{_tx_col_name(c1)}{r1 + 1}"
    lines = [head, "row | " + " | ".join(_tx_col_name(c) for c in range(c0, c1 + 1))]
    for r in range(r0, r1 + 1):
        row = rows[r] if r < len(rows) else []
        cells = ["" if c >= len(row) or row[c] is None else str(row[c])
                 for c in range(c0, c1 + 1)]
        lines.append(f"{r + 1} | " + " | ".join(cells))
    return "\n".join(lines)


def _tx_rows_arg(value: Any) -> list[list[str]]:
    data: Any = value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            data = _tx_json_arg(text, "rows")
        else:
            data = [list(r) for r in csv.reader(io.StringIO(text))]
    if not isinstance(data, list) or not data:
        raise ValueError("rows must be a JSON array of rows (arrays) or CSV text")
    rows = []
    for row in data:
        if isinstance(row, dict):
            raise ValueError("rows must be arrays, not objects")
        cells = row if isinstance(row, list) else [row]
        rows.append(["" if c is None else str(c) for c in cells])
    return rows


def _tx_spreadsheet_write(registry: Any, path: str, rows: Any, title: Any = None) -> str:
    grid = _tx_rows_arg(rows)
    target = registry._resolve(path)
    suffix = target.suffix.lower()
    target.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        buffer = io.StringIO()
        csv.writer(buffer).writerows(grid)
        registry._save_bytes(target, buffer.getvalue().encode("utf-8"))
    elif suffix == ".xlsx":
        width = max(len(r) for r in grid)

        def cell(text: str) -> str:
            # Markdown table cells: a pipe would split the cell, a newline the row.
            return text.replace("|", "∣").replace("\r", " ").replace("\n", " ").strip()

        padded = [r + [""] * (width - len(r)) for r in grid]
        table = ["| " + " | ".join(cell(c) or " " for c in padded[0]) + " |",
                 "|" + "---|" * width]
        table += ["| " + " | ".join(cell(c) for c in r) + " |" for r in padded[1:]]
        registry._save_bytes(target, render("\n".join(table), ".xlsx", str(title or "")))
    else:
        raise ValueError("spreadsheet_write writes .xlsx or .csv files")
    return (f"Wrote {len(grid)} row(s) x {max(len(r) for r in grid)} column(s) to "
            f"{registry._rel(target)} (first row is the header).")


# --------------------------------------------------------------------------- #
# pdf_extract
# --------------------------------------------------------------------------- #
def _tx_pages(spec: str, count: int) -> list[int]:
    chosen: list[int] = []
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            start = int(a) if a else 1
            end = int(b) if b else count
        else:
            start = end = int(part)
        for page in range(max(1, start), min(count, end) + 1):
            if page not in chosen:
                chosen.append(page)
        if len(chosen) >= 500:
            break
    return chosen


def _tx_pdf_extract(registry: Any, path: str, pages: Any = None, max_chars: Any = None) -> str:
    target = registry._resolve(path)
    if not target.is_file():
        raise ValueError(f"no such file: {path}")
    if target.suffix.lower() != ".pdf":
        raise ValueError("pdf_extract reads .pdf files")
    cap = _tx_int(max_chars, registry.config.tool_raw_chars, 500, registry.config.tool_raw_chars)
    reader_cls = None
    for module, attr in (("pypdf", "PdfReader"), ("PyPDF2", "PdfReader")):
        try:
            reader_cls = getattr(__import__(module), attr)
            break
        except ImportError:
            continue
    if reader_cls is None:
        raise ValueError("no PDF library is installed on the server: `pip install pypdf` "
                         "to read PDFs")
    if pages in (None, ""):
        return registry._extract_pdf(str(target), cap)
    try:
        reader = reader_cls(str(target))
        count = len(reader.pages)
        chosen = _tx_pages(str(pages), count)
    except ValueError as exc:
        raise ValueError(f"pages must look like 1-3,5: {exc}") from None
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"could not read the PDF: {exc}") from None
    if not chosen:
        raise ValueError(f"no such pages; the PDF has {count} page(s)")
    parts = [f"[pdf: {count} pages; showing {', '.join(map(str, chosen[:20]))}"
             + ("..." if len(chosen) > 20 else "") + "]"]
    for page in chosen:
        try:
            text = (reader.pages[page - 1].extract_text() or "").strip()
        except Exception as exc:  # noqa: BLE001
            text = f"[could not extract: {exc}]"
        parts.append(f"--- page {page} ---\n{text or '[no extractable text]'}")
        if sum(len(p) for p in parts) > cap:
            break
    return "\n\n".join(parts)[:cap]


# --------------------------------------------------------------------------- #
# image_analyze / image_generate
# --------------------------------------------------------------------------- #
def _tx_vision_capable(registry: Any) -> bool:
    """Same rule as Agent.vision_capable: the route's word, else backend + catalogue."""
    turn = current_turn()
    extras = getattr(turn, "extras", None)
    if isinstance(extras, dict) and isinstance(extras.get("vision"), bool):
        return extras["vision"]
    manager = getattr(registry, "model_manager", None)
    # The chat route's rule (vision.served_vision), so the tool, the chat
    # composer and /v1 agree: VISION_MODE=on/off is honoured, and mlx-lm, which
    # declares "vision" for every model, only counts when it ships a processor
    # for the model's vision tower.
    served = _tx_feature("vision", "served_vision")
    if callable(served):
        try:
            return bool(served(manager, registry.config)[0])
        except Exception:  # noqa: BLE001
            return False
    backend = getattr(manager, "backend", None)
    if backend is None:
        get_backend = _tx_feature("backends", "get_backend")
        try:
            backend = (get_backend(getattr(registry.config, "model_backend", None))
                       if callable(get_backend) else None)
        except Exception:  # noqa: BLE001
            backend = None
    try:
        caps = backend.caps_for(manager) if backend is not None else frozenset()
    except Exception:  # noqa: BLE001
        caps = frozenset()
    if "vision" not in caps:
        return False
    metadata = _tx_feature("catalog", "model_metadata")
    model = getattr(manager, "model", None) or getattr(registry.config, "model", "")
    try:
        meta = metadata(str(model or "")) if callable(metadata) else None
    except Exception:  # noqa: BLE001
        meta = None
    return bool(isinstance(meta, dict) and meta.get("vision"))


def _tx_run_async(coro: Any) -> Any:
    """Run a coroutine from a tool handler (a worker thread, normally)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: dict = {}

    def runner() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    worker = threading.Thread(target=runner, daemon=True)
    worker.start()
    worker.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _tx_model_complete(registry: Any, messages: list[dict], max_tokens: int) -> str:
    client_cls = _tx_feature("model_client", "ModelClient")
    if client_cls is None:
        raise ValueError("the model client is not available")
    client = client_cls(registry.config)
    return str(_tx_run_async(client.complete(messages, max_tokens=max_tokens,
                                             temperature=0.2)) or "")


def _tx_image_analyze(registry: Any, path: str, question: Any = None) -> str:
    if not _tx_vision_capable(registry):
        raise ValueError(
            f"image_analyze needs a vision model, and the served model "
            f"({registry.config.model}) cannot read images. Serve a vision model "
            "(for example an MLX build of Gemma 3 or Qwen2.5-VL) to use it.")
    target = registry._resolve(path)
    if not target.is_file():
        raise ValueError(f"no such file: {path}")
    mime = _TX_IMAGE_TYPES.get(target.suffix.lower())
    if mime is None:
        raise ValueError("image_analyze reads .png, .jpg, .jpeg, .gif and .webp images")
    data = target.read_bytes()
    if len(data) > 8_000_000:
        raise ValueError("the image is larger than 8 MB")
    prompt = str(question or "").strip() or "Describe this image in detail."
    url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": url}}]}]
    answer = _tx_model_complete(registry, messages,
                                min(int(registry.config.max_tokens or 512), 1024)).strip()
    if not answer:
        raise ValueError("the model returned no description")
    return answer


def _tx_image_command(config: Any) -> str:
    template = str(getattr(config, "image_gen_command", "") or "").strip()
    if template:
        return template
    if shutil.which("mflux-generate"):
        return ("mflux-generate --model schnell --quantize 4 --prompt {prompt} "
                "--steps {steps} --width {width} --height {height} --output {output}")
    return ""


def _tx_image_generate(registry: Any, prompt: str, path: Any = None, width: Any = None,
                       height: Any = None, steps: Any = None) -> str:
    template = _tx_image_command(registry.config)
    if not template:
        raise ValueError("image generation is unavailable on this server: install mflux "
                         "(`pip install mflux`) or set IMAGE_GEN_COMMAND to a command "
                         "with {prompt} and {output}.")
    text = str(prompt or "").strip()
    if not text:
        raise ValueError("a prompt is required")
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "image"
    rel = str(path or f"images/{slug}-{int(time.time())}.png")
    target = registry._resolve(rel)
    if target.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        target = target.with_suffix(".png")
    values = {"prompt": text[:2000], "width": str(_tx_int(width, 1024, 256, 2048)),
              "height": str(_tx_int(height, 1024, 256, 2048)),
              "steps": str(_tx_int(steps, 4, 1, 50))}
    with tempfile.TemporaryDirectory(prefix="llm-imagegen-") as scratch:
        output = str(Path(scratch) / f"out{target.suffix.lower()}")
        values["output"] = output
        # Filled per argument, never through a shell: the prompt is model text.
        argv = []
        for token in shlex.split(template):
            for key, value in values.items():
                token = token.replace("{" + key + "}", value)
            argv.append(token)
        try:
            proc = subprocess.run(argv, cwd=scratch, env=registry._sandbox_env(),
                                  capture_output=True, text=True, errors="replace",
                                  stdin=subprocess.DEVNULL,
                                  timeout=registry.config.image_gen_timeout_s)
        except FileNotFoundError:
            raise ValueError(f"the image generator {argv[0]!r} is not installed") from None
        except subprocess.TimeoutExpired:
            raise ValueError(f"image generation timed out after "
                             f"{registry.config.image_gen_timeout_s}s") from None
        produced = Path(output)
        if proc.returncode != 0 or not produced.is_file():
            tail = (proc.stderr or proc.stdout or "").strip()[-600:]
            raise ValueError(f"the image generator failed (exit {proc.returncode}): {tail}")
        data = produced.read_bytes()
    if not (data.startswith(b"\x89PNG") or data.startswith(b"\xff\xd8")):
        raise ValueError("the image generator did not produce a PNG or JPEG")
    target.parent.mkdir(parents=True, exist_ok=True)
    registry._save_bytes(target, data)
    return f"Generated {registry._rel(target)} ({len(data) // 1024} KB) for: {text[:200]}"


# --------------------------------------------------------------------------- #
# database: read-only SQL
# --------------------------------------------------------------------------- #
def _tx_db_targets(config: Any) -> dict[str, str]:
    raw = str(getattr(config, "database_urls", "") or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        log("DATABASE_URLS is not valid JSON; no databases configured.", logging.WARNING)
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _tx_db_path(registry: Any, database: Any) -> Path:
    targets = _tx_db_targets(registry.config)
    name = str(database or "").strip()
    if not name:
        if len(targets) == 1:
            name = next(iter(targets))
        else:
            known = ", ".join(sorted(targets)) or "none configured"
            raise ValueError(f"name the database (configured: {known}; or a .db file "
                             "in the project)")
    if name in targets:
        url = targets[name]
        if "://" in url and not url.lower().startswith("sqlite:"):
            raise ValueError(f"{name}: only SQLite databases are supported "
                             f"({url.split('://', 1)[0]} needs a driver this server lacks)")
        path = re.sub(r"(?i)^sqlite:/*", "/", url) if url.lower().startswith("sqlite:") else url
        target = Path(path).expanduser()
    else:
        target = registry._resolve(name)
        if target.suffix.lower() not in (".db", ".sqlite", ".sqlite3", ".db3"):
            raise ValueError(f"{name!r} is neither a configured database nor a "
                             ".db/.sqlite file in the project")
    if not target.is_file():
        raise ValueError(f"database file not found: {name}")
    return target


def _tx_sql_check(sql: str) -> str:
    """One read-only statement, or a clear refusal (belt to the authorizer's braces)."""
    text = re.sub(r"--[^\n]*|/\*.*?\*/", " ", str(sql or ""), flags=re.S).strip()
    text = text.rstrip(";").strip()
    if not text:
        raise ValueError("an SQL statement is required")
    if ";" in text:
        raise ValueError("one statement at a time")
    first = text.split(None, 1)[0].lower()
    if first not in _TX_SQL_START:
        raise ValueError("read-only: only SELECT, WITH, EXPLAIN, VALUES and "
                         "informational PRAGMA statements run here")
    if first == "pragma":
        match = re.match(r"(?i)pragma\s+(?:\w+\.)?(\w+)\s*(\(|$)", text)
        if not match or match.group(1).lower() not in _TX_SQL_PRAGMAS or "=" in text:
            raise ValueError(f"read-only: allowed PRAGMAs are {', '.join(sorted(_TX_SQL_PRAGMAS))}")
    elif _TX_SQL_WRITE.search(re.sub(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"", "''", text)):
        raise ValueError("read-only: the statement contains a writing keyword")
    return text


def _tx_database(registry: Any, sql: str, database: Any = None, limit: Any = None) -> str:
    statement = _tx_sql_check(sql)
    target = _tx_db_path(registry, database)
    rows_cap = _tx_int(limit, 100, 1, 1000)
    uri = f"file:{urllib.parse.quote(str(target))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5, check_same_thread=False)
    deadline = time.time() + float(registry.config.tool_timeout)
    try:
        conn.execute("PRAGMA query_only = ON")
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                   sqlite3.SQLITE_PRAGMA, getattr(sqlite3, "SQLITE_RECURSIVE", 33)}

        def authorizer(action, arg1, arg2, dbname, source):  # noqa: ANN001
            # The real enforcement: the connection is read-only (mode=ro) and
            # this denies anything that is not reading, ATTACH included (which
            # would otherwise open any file the server can read).
            if action == sqlite3.SQLITE_PRAGMA:
                return sqlite3.SQLITE_OK if str(arg1).lower() in _TX_SQL_PRAGMAS | {
                    "query_only"} else sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

        conn.set_authorizer(authorizer)
        conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10_000)
        try:
            cursor = conn.execute(statement)
            columns = [d[0] for d in (cursor.description or [])]
            rows = cursor.fetchmany(rows_cap + 1)
        except sqlite3.DatabaseError as exc:
            message = str(exc)
            if "interrupted" in message:
                message = f"query cancelled after {registry.config.tool_timeout}s"
            raise ValueError(f"SQL error: {message}") from None
    finally:
        conn.close()
    more = len(rows) > rows_cap
    rows = rows[:rows_cap]
    lines = ["\t".join(columns)] if columns else []
    for row in rows:
        lines.append("\t".join("NULL" if v is None else
                               (f"<{len(v)} bytes>" if isinstance(v, bytes) else str(v))
                               for v in row))
    lines.append(f"({len(rows)} row{'s' if len(rows) != 1 else ''}"
                 + (f", stopped at the limit of {rows_cap}" if more else "") + ")")
    return _tx_cap(registry, "\n".join(lines))


# --------------------------------------------------------------------------- #
# email / calendar / contacts: Microsoft Graph, app credentials
# --------------------------------------------------------------------------- #
def _tx_graph_token(config: Any) -> str:
    key = f"{config.o365_tenant_id}:{config.o365_client_id}"
    with _TX_TOKEN_LOCK:
        cached = _TX_TOKEN_CACHE.get(key)
        if cached and cached[1] > time.time() + 60:
            return cached[0]
    import httpx
    resp = httpx.post(
        f"https://login.microsoftonline.com/{config.o365_tenant_id}/oauth2/v2.0/token",
        data={"client_id": config.o365_client_id, "client_secret": config.o365_client_secret,
              "scope": config.o365_scopes or "https://graph.microsoft.com/.default",
              "grant_type": "client_credentials"},
        timeout=config.tool_timeout)
    if resp.status_code >= 400:
        raise ValueError(f"Microsoft sign-in failed (HTTP {resp.status_code}): "
                         + redact_text(resp.text[:200]))
    body = resp.json()
    token = str(body.get("access_token") or "")
    if not token:
        raise ValueError("Microsoft sign-in returned no token")
    with _TX_TOKEN_LOCK:
        _TX_TOKEN_CACHE[key] = (token, time.time() + float(body.get("expires_in") or 3000))
    return token


def _tx_graph(config: Any, method: str, path: str, params: dict | None = None,
              body: Any = None) -> dict:
    import httpx
    resp = httpx.request(method, f"{_TX_GRAPH}{path}", params=params, json=body,
                         headers={"Authorization": f"Bearer {_tx_graph_token(config)}"},
                         timeout=config.tool_timeout)
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("error", {}).get("message") or resp.text
        except Exception:  # noqa: BLE001
            detail = resp.text
        raise ValueError(f"Microsoft Graph error (HTTP {resp.status_code}): "
                         + redact_text(str(detail)[:300]))
    if resp.status_code == 202 or not resp.content:
        return {}
    return resp.json()


def _tx_mailbox(registry: Any) -> str:
    """The acting user's own mailbox, never someone else's.

    The app credentials can read any mailbox in the tenant, so the address is
    taken from the signed-in user's own record; O365_USER is only the fallback
    for a local (auth off) run or an administrator without an address.
    """
    uid = get_acting_user()
    db = getattr(registry, "db", None)
    row = None
    if uid not in (None, SENTINEL_LOCAL_USER) and db is not None:
        try:
            row = db.get_user(uid)
        except Exception:  # noqa: BLE001
            row = None
    for value in ((row or {}).get("email"), (row or {}).get("username")):
        if value and "@" in str(value):
            return urllib.parse.quote(str(value).strip(), safe="@.")
    fallback = str(getattr(registry.config, "o365_user", "") or "").strip()
    if fallback and (uid in (None, SENTINEL_LOCAL_USER) or (row or {}).get("role") == "admin"):
        return urllib.parse.quote(fallback, safe="@.")
    raise ValueError("no Microsoft 365 mailbox is linked to your account (sign in with "
                     "Microsoft, or ask an admin)")


def _tx_mail_line(msg: dict) -> str:
    sender = ((msg.get("from") or {}).get("emailAddress") or {})
    unread = "" if msg.get("isRead", True) else " [unread]"
    return (f"- id={msg.get('id')}\n  {msg.get('receivedDateTime', '')} from "
            f"{sender.get('name') or ''} <{sender.get('address') or ''}>{unread}\n"
            f"  {msg.get('subject') or '(no subject)'}\n  {(msg.get('bodyPreview') or '')[:200]}")


def _tx_email(registry: Any, action: Any = "list", query: Any = None, id: Any = None,  # noqa: A002
              folder: Any = None, limit: Any = None) -> str:
    verb = str(action or "list").strip().lower()
    box = _tx_mailbox(registry)
    top = _tx_int(limit, 10, 1, 50)
    select = "id,subject,from,receivedDateTime,isRead,bodyPreview"
    if verb == "read":
        if not id:
            raise ValueError("read needs the message id (from list or search)")
        msg = _tx_graph(registry.config, "GET",
                        f"/users/{box}/messages/{urllib.parse.quote(str(id), safe='')}",
                        {"$select": "subject,from,toRecipients,ccRecipients,receivedDateTime,body"})
        body = (msg.get("body") or {})
        text = body.get("content") or ""
        if str(body.get("contentType") or "").lower() == "html":
            text = strip_html(text)
        to = ", ".join(((r.get("emailAddress") or {}).get("address") or "")
                       for r in msg.get("toRecipients") or [])
        sender = ((msg.get("from") or {}).get("emailAddress") or {})
        return _tx_cap(registry, f"Subject: {msg.get('subject')}\nFrom: {sender.get('address')}\n"
                                 f"To: {to}\nDate: {msg.get('receivedDateTime')}\n\n{text.strip()}")
    if verb == "search":
        if not query:
            raise ValueError("search needs a query")
        data = _tx_graph(registry.config, "GET", f"/users/{box}/messages",
                         {"$search": f"\"{str(query).replace(chr(34), '')}\"", "$top": top,
                          "$select": select})
    elif verb == "list":
        mail_folder = urllib.parse.quote(str(folder or "inbox"), safe="")
        data = _tx_graph(registry.config, "GET", f"/users/{box}/mailFolders/{mail_folder}/messages",
                         {"$top": top, "$select": select, "$orderby": "receivedDateTime desc"})
    else:
        raise ValueError("action must be list, search or read (send with email_send)")
    items = data.get("value") or []
    if not items:
        return "No messages."
    return _tx_cap(registry, "\n".join(_tx_mail_line(m) for m in items))


def _tx_recipients(raw: Any) -> list[dict]:
    out = []
    for item in re.split(r"[,;\s]+", str(raw or "")):
        item = item.strip()
        if item:
            if "@" not in item:
                raise ValueError(f"not an email address: {item!r}")
            out.append({"emailAddress": {"address": item}})
    return out


def _tx_email_send(registry: Any, to: str, subject: str, body: str, cc: Any = None) -> str:
    recipients = _tx_recipients(to)
    if not recipients:
        raise ValueError("at least one recipient is required")
    box = _tx_mailbox(registry)
    message = {"subject": str(subject)[:400],
               "body": {"contentType": "Text", "content": str(body)},
               "toRecipients": recipients}
    if cc:
        message["ccRecipients"] = _tx_recipients(cc)
    _tx_graph(registry.config, "POST", f"/users/{box}/sendMail",
              body={"message": message, "saveToSentItems": True})
    return f"Sent to {', '.join(r['emailAddress']['address'] for r in recipients)}."


def _tx_when(value: Any, default: datetime) -> datetime:
    if value in (None, ""):
        return default
    text = str(value).strip()
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"not an ISO date/time: {text!r}") from None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _tx_calendar(registry: Any, start: Any = None, end: Any = None, limit: Any = None) -> str:
    box = _tx_mailbox(registry)
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    begin = _tx_when(start, now)
    finish = _tx_when(end, begin + timedelta(days=7))
    data = _tx_graph(registry.config, "GET", f"/users/{box}/calendarView", {
        "startDateTime": begin.isoformat(), "endDateTime": finish.isoformat(),
        "$top": _tx_int(limit, 25, 1, 100), "$orderby": "start/dateTime",
        "$select": "subject,start,end,location,organizer,isAllDay"})
    events = data.get("value") or []
    if not events:
        return f"No events between {begin.date()} and {finish.date()}."
    lines = []
    for ev in events:
        where = (ev.get("location") or {}).get("displayName") or ""
        lines.append(f"- {(ev.get('start') or {}).get('dateTime', '')[:16]} to "
                     f"{(ev.get('end') or {}).get('dateTime', '')[:16]}  "
                     f"{ev.get('subject') or '(no subject)'}" + (f" @ {where}" if where else ""))
    return _tx_cap(registry, "\n".join(lines))


def _tx_calendar_create(registry: Any, subject: str, start: str, end: str,
                        attendees: Any = None, location: Any = None, body: Any = None,
                        timezone_name: Any = None) -> str:
    box = _tx_mailbox(registry)
    tz = str(timezone_name or "UTC")
    begin = _tx_when(start, datetime.now(timezone.utc))
    finish = _tx_when(end, begin + timedelta(hours=1))
    if finish <= begin:
        raise ValueError("end must be after start")
    event: dict[str, Any] = {
        "subject": str(subject)[:400],
        "start": {"dateTime": begin.replace(tzinfo=None).isoformat(), "timeZone": tz},
        "end": {"dateTime": finish.replace(tzinfo=None).isoformat(), "timeZone": tz},
    }
    if location:
        event["location"] = {"displayName": str(location)[:200]}
    if body:
        event["body"] = {"contentType": "Text", "content": str(body)}
    people = _tx_recipients(attendees) if attendees else []
    if people:
        event["attendees"] = [{**p, "type": "required"} for p in people]
    made = _tx_graph(registry.config, "POST", f"/users/{box}/events", body=event)
    return f"Created \"{made.get('subject') or subject}\" (id={made.get('id', '?')})."


def _tx_contacts(registry: Any, query: Any = None, limit: Any = None) -> str:
    box = _tx_mailbox(registry)
    data = _tx_graph(registry.config, "GET", f"/users/{box}/contacts", {
        "$top": 200, "$select": "displayName,emailAddresses,mobilePhone,businessPhones,companyName"})
    wanted = str(query or "").strip().lower()
    lines = []
    for person in data.get("value") or []:
        emails = [e.get("address") or "" for e in person.get("emailAddresses") or []]
        haystack = " ".join([person.get("displayName") or "", person.get("companyName") or "",
                             *emails]).lower()
        if wanted and wanted not in haystack:
            continue
        phones = [p for p in [person.get("mobilePhone"), *(person.get("businessPhones") or [])] if p]
        lines.append(f"- {person.get('displayName') or '(no name)'}"
                     + (f" ({person.get('companyName')})" if person.get("companyName") else "")
                     + (f"  {', '.join(emails)}" if emails else "")
                     + (f"  {', '.join(phones)}" if phones else ""))
        if len(lines) >= _tx_int(limit, 20, 1, 100):
            break
    return _tx_cap(registry, "\n".join(lines) or "No matching contacts.")


# --------------------------------------------------------------------------- #
# task_scheduler: the caller's own tasks, never more tools than the caller has
# --------------------------------------------------------------------------- #
def _tx_caller_tools(registry: Any) -> set[str]:
    """What a task created now may use: the caller's own gated set.

    The registry the caller is running in is already narrowed by their profile
    (and role, in chat); a restricted role narrows further here because the
    default agent is not role-gated today. task_scheduler itself is left out so
    a task cannot multiply itself.
    """
    names = set(registry.names()) - {"task_scheduler"}
    role = _tx_role(registry)
    caps = capabilities_for_role(role, registry.config)
    if set(caps) != set(CAPABILITY_ORDER):
        names &= set(tools_for_capabilities(caps))
    return names | {"final_answer"}


def _tx_task_scheduler(registry: Any, action: Any = "list", goal: Any = None, name: Any = None,
                       every_minutes: Any = None, start_in_minutes: Any = None,
                       tools: Any = None, task_id: Any = None) -> str:
    db = registry.db
    owner = get_acting_user() or SENTINEL_LOCAL_USER
    verb = str(action or "list").strip().lower()
    own = [t for t in db.list_tasks() if str(t.get("user_id") or "") == str(owner)]
    if verb == "list":
        if not own:
            return "You have no scheduled tasks."
        lines = []
        for task in own:
            every = int(task.get("interval_seconds") or 0)
            lines.append(f"- {task['id']} {task.get('name')!r} "
                         f"{'enabled' if task.get('enabled') else 'disabled'}"
                         + (f", every {every // 60} min" if every else ", once")
                         + f", next: {task.get('next_run_at') or '-'}"
                         + f"\n  goal: {str(task.get('goal') or '')[:160]}"
                         + f"\n  tools: {task.get('tools') or '-'}")
        return "\n".join(lines)
    if verb in ("cancel", "delete", "disable"):
        task = next((t for t in own if t["id"] == str(task_id or "").strip()), None)
        if task is None:
            raise ValueError("no such task of yours (see action=list)")
        db.update_task(task["id"], {"enabled": False})
        return f"Cancelled task {task['id']} ({task.get('name')}); its history is kept."
    if verb != "create":
        raise ValueError("action must be create, list or cancel")
    text = str(goal or "").strip()
    if not text:
        raise ValueError("a goal is required: what the task should do each run")
    if sum(1 for t in own if t.get("enabled")) >= _TX_MAX_TASKS_PER_USER:
        raise ValueError(f"you already have {_TX_MAX_TASKS_PER_USER} enabled tasks; "
                         "cancel one first")
    allowed = _tx_caller_tools(registry)
    requested = [t.strip() for t in re.split(r"[,\s]+", str(tools or "")) if t.strip()]
    if requested:
        chosen = [t for t in requested if t in allowed]
        refused = [t for t in requested if t not in allowed]
    else:
        chosen, refused = sorted(allowed), []
    if "final_answer" not in chosen:
        chosen.append("final_answer")
    interval = 0
    if every_minutes not in (None, "", 0, "0"):
        interval = _tx_int(every_minutes, 60, _TX_MIN_INTERVAL_MIN, 60 * 24 * 31) * 60
    created = db.create_task(
        user_id=owner, name=str(name or text[:60]).strip()[:120], goal=text,
        enabled=True, interval_seconds=interval, tools=",".join(chosen),
        max_steps=min(6, int(registry.config.agent_max_steps or 6)))
    delay = _tx_int(start_in_minutes, 0 if interval else 1, 0, 60 * 24 * 365)
    first = datetime.now(timezone.utc) + timedelta(minutes=delay)
    db.execute("UPDATE tasks SET next_run_at = ? WHERE id = ? AND user_id = ?",
               (iso(first), created["id"], owner))
    db.commit()
    note = (f"\nNot granted (beyond your own tools): {', '.join(refused)}" if refused else "")
    return (f"Scheduled task {created['id']} ({created.get('name')}): "
            + (f"every {interval // 60} min" if interval else "once")
            + f", first run {iso(first)}.\nTools: {', '.join(chosen)}{note}")


# --------------------------------------------------------------------------- #
# notify
# --------------------------------------------------------------------------- #
def _tx_notify_schema(conn: Any) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT,
            body TEXT NOT NULL,
            level TEXT NOT NULL DEFAULT 'info',
            source TEXT NOT NULL DEFAULT 'agent',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            read_at TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_notifications_user "
                 "ON notifications(user_id, read_at)")


register_schema(_TX_NOTIFY_SCHEMA, _tx_notify_schema)


def store_notification(db: Any, user_id: str, body: str, title: str = "",
                       level: str = "info", source: str = "agent") -> int:
    db.ensure_schema(_TX_NOTIFY_SCHEMA)
    cursor = db.execute(
        "INSERT INTO notifications (user_id, title, body, level, source) VALUES (?, ?, ?, ?, ?)",
        (user_id, title[:120], body[:1000], level, source))
    db.commit()
    return int(cursor.lastrowid)


def list_notifications(db: Any, user_id: str, unread_only: bool = False,
                       limit: int = 50) -> list[dict]:
    db.ensure_schema(_TX_NOTIFY_SCHEMA)
    where = "user_id = ?" + (" AND read_at IS NULL" if unread_only else "")
    rows = db.execute(f"SELECT * FROM notifications WHERE {where} ORDER BY id DESC LIMIT ?",
                      (user_id, max(1, min(200, int(limit))))).fetchall()
    return [dict(r) for r in rows]


def _tx_on_console() -> bool:
    """Whether a macOS notification would reach someone at this Mac."""
    if sys.platform != "darwin" or not shutil.which("osascript"):
        return False
    return not (os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"))


def _tx_applescript(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _tx_notify(registry: Any, message: str, title: Any = None, level: Any = None) -> str:
    body = re.sub(r"\s+", " ", str(message or "")).strip()[:1000]
    if not body:
        raise ValueError("a message is required")
    heading = str(title or APP_NAME).strip()[:120]
    grade = str(level or "info").strip().lower()
    grade = grade if grade in ("info", "warning", "urgent") else "info"
    owner = get_acting_user() or SENTINEL_LOCAL_USER
    delivered, failed = [], []
    if registry.db is not None:
        try:
            store_notification(registry.db, owner, body, heading, grade)
            delivered.append("in-app")
        except Exception as exc:  # noqa: BLE001
            failed.append(f"in-app ({exc})")
    url = str(getattr(registry.config, "notify_webhook_url", "") or "").strip()
    if url:
        try:
            import httpx
            if "ntfy" in (urllib.parse.urlparse(url).hostname or ""):
                resp = httpx.post(url, content=body.encode("utf-8"), timeout=10,
                                  headers={"Title": heading.encode("ascii", "replace").decode(),
                                           "Priority": "high" if grade == "urgent" else "default"})
            else:
                resp = httpx.post(url, json={"text": f"*{heading}*: {body}", "title": heading,
                                             "message": body, "level": grade}, timeout=10)
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}")
            delivered.append("webhook")
        except Exception as exc:  # noqa: BLE001
            failed.append(f"webhook ({redact_text(str(exc))[:120]})")
    # The server's own screen belongs to whoever sits at it: only a local run
    # or an administrator's notification goes there, never another user's.
    if getattr(registry.config, "notify_macos", False) and _tx_on_console() \
            and _tx_role(registry) == "admin":
        try:
            subprocess.run(["osascript", "-e", f"display notification {_tx_applescript(body)} "
                                               f"with title {_tx_applescript(heading)}"],
                           capture_output=True, timeout=10, stdin=subprocess.DEVNULL)
            delivered.append("macOS")
        except Exception as exc:  # noqa: BLE001
            failed.append(f"macOS ({exc})")
    if not delivered:
        raise ValueError("the notification could not be delivered: "
                         + ("; ".join(failed) or "no channel is configured"))
    return f"Notified via {', '.join(delivered)}." + (
        f" Failed: {'; '.join(failed)}." if failed else "")


def register_tools_extra(app: Any, ctx: Any) -> None:
    """In-app notifications written by the notify tool (owner-scoped)."""
    from fastapi.responses import JSONResponse

    db = ctx.db

    @app.get("/api/notifications")
    async def notifications_list(unread: int = 0, limit: int = 50, user: dict = ctx.USER):
        rows = await asyncio.to_thread(list_notifications, db, user["id"], bool(unread), limit)
        return {"notifications": rows}

    @app.post("/api/notifications/read-all")
    async def notifications_read_all(user: dict = ctx.USER):
        def mark() -> int:
            db.ensure_schema(_TX_NOTIFY_SCHEMA)
            cur = db.execute("UPDATE notifications SET read_at = ? WHERE user_id = ? "
                             "AND read_at IS NULL", (iso(utc_now()), user["id"]))
            db.commit()
            return cur.rowcount
        return {"marked": await asyncio.to_thread(mark)}

    @app.post("/api/notifications/{notification_id}/read")
    async def notifications_read(notification_id: int, user: dict = ctx.USER):
        def mark() -> int:
            db.ensure_schema(_TX_NOTIFY_SCHEMA)
            cur = db.execute("UPDATE notifications SET read_at = ? WHERE id = ? AND user_id = ?",
                             (iso(utc_now()), notification_id, user["id"]))
            db.commit()
            return cur.rowcount
        if not await asyncio.to_thread(mark):
            return JSONResponse({"error": "not found"}, status_code=404)
        return {"ok": True}


# --------------------------------------------------------------------------- #
# Builders: (registry, Tool) -> Tool or None when a prerequisite is missing
# --------------------------------------------------------------------------- #
def _tx_bind(registry: Any, fn: Any, name: str) -> Any:
    def handler(**kwargs: Any) -> str:
        return fn(registry, **kwargs)
    handler.__name__ = name
    return handler


def _tx_build(name: str, description: str, parameters: dict, required: list, fn: Any, *,
              risk: str, capability: str, taints: bool | None = None,
              prerequisite: Any = None) -> Any:
    def build(registry: Any, tool_cls: Any) -> Any:
        if prerequisite is not None and not prerequisite(registry):
            return None
        return tool_cls(name=name, description=description, parameters=parameters,
                        required=required, handler=_tx_bind(registry, fn, name),
                        risk=risk, capability=capability, taints=taints)
    return build


def _tx_has_o365(registry: Any) -> bool:
    check = getattr(registry, "_o365_configured", None)
    return bool(check()) if callable(check) else False


_TX_BUILDERS: dict[str, Any] = {
    "browse_page": _tx_build(
        "browse_page",
        "Open a web page and read it as text, with its numbered links. Optionally "
        "follow ONE link on the same site (by its text or URL) and read that too.",
        {"url": "page URL (http/https)", "follow": "optional: link text or URL to follow",
         "max_chars": "optional cap on text per page"},
        ["url"], _tx_browse_page, risk="read", capability="browse_page", taints=True),
    "http_request": _tx_build(
        "http_request",
        "Call a public HTTP API: any method, headers and a JSON or text body. Returns "
        "status, key headers and the body. Anything but GET/HEAD may need approval.",
        {"url": "endpoint URL", "method": "GET | POST | PUT | PATCH | DELETE | HEAD",
         "headers": "optional JSON object of headers", "json": "optional JSON body",
         "body": "optional raw text body", "max_bytes": "optional response cap"},
        ["url"], _tx_http_request, risk="read", capability="http_request", taints=True),
    "code_interpreter": _tx_build(
        "code_interpreter",
        "Run Python in a session that keeps its variables, imports and functions "
        "between calls in this conversation (a REPL: the last expression is printed). "
        "Set reset=true to start over.",
        {"code": "Python code", "reset": "optional: true to clear the session first"},
        ["code"], _tx_code_interpreter, risk="exec", capability="code_interpreter",
        prerequisite=lambda r: bool(r.config.allow_python)),
    "git": _tx_build(
        "git",
        "Inspect the project's git repository: status, diff (staged=true for the "
        "index), log, show <rev>, branch.",
        {"action": "status | diff | log | show | branch", "path": "optional path filter",
         "rev": "optional revision (show/diff/log)", "staged": "diff the index",
         "limit": "log entries (default 20)"},
        ["action"], _tx_git, risk="read", capability="git",
        prerequisite=lambda r: bool(shutil.which("git"))),
    "git_commit": _tx_build(
        "git_commit",
        "Stage files (action=add, paths) or commit (action=commit, message, optional "
        "paths to stage first) in the project repository. Never pushes.",
        {"action": "add | commit", "paths": "paths to stage (comma separated)",
         "message": "commit message"},
        ["action"], _tx_git_commit, risk="write", capability="git",
        prerequisite=lambda r: bool(shutil.which("git"))),
    "spreadsheet": _tx_build(
        "spreadsheet",
        "Read a spreadsheet (.xlsx/.csv): with only a path, lists sheets and previews "
        "the first; with sheet and/or range (A1:D20) returns those cells.",
        {"path": "spreadsheet file", "sheet": "optional sheet name or number",
         "range": "optional cell range like A1:D20", "max_rows": "optional row cap"},
        ["path"], _tx_spreadsheet, risk="read", capability="spreadsheet"),
    "spreadsheet_write": _tx_build(
        "spreadsheet_write",
        "Write a sheet to an .xlsx or .csv file from rows (JSON array of arrays, first "
        "row the header, or CSV text). Overwrites the file.",
        {"path": "output .xlsx or .csv", "rows": "JSON array of rows, or CSV text",
         "title": "optional document title"},
        ["path", "rows"], _tx_spreadsheet_write, risk="write", capability="spreadsheet"),
    "pdf_extract": _tx_build(
        "pdf_extract",
        "Page count and text of a PDF in the project; pages like 1-3,7 picks pages.",
        {"path": "PDF file", "pages": "optional page list like 1-3,7",
         "max_chars": "optional text cap"},
        ["path"], _tx_pdf_extract, risk="read", capability="pdf_extract"),
    "image_analyze": _tx_build(
        "image_analyze",
        "Look at an image file in the project and describe it or answer a question "
        "about it (needs a vision model).",
        {"path": "image file (.png/.jpg/.gif/.webp)", "question": "optional question"},
        ["path"], _tx_image_analyze, risk="read", capability="image_analyze"),
    "image_generate": _tx_build(
        "image_generate",
        "Generate an image from a text prompt with the server's local generator and "
        "save it in the project.",
        {"prompt": "what to draw", "path": "optional output path (.png)",
         "width": "optional pixels", "height": "optional pixels",
         "steps": "optional diffusion steps"},
        ["prompt"], _tx_image_generate, risk="exec", capability="image_generate"),
    "database": _tx_build(
        "database",
        "Run ONE read-only SQL query (SELECT/WITH/EXPLAIN/PRAGMA table_info) against a "
        "configured SQLite database or a .db file in the project.",
        {"sql": "the query", "database": "configured name or .db path",
         "limit": "optional row cap (default 100)"},
        ["sql"], _tx_database, risk="read", capability="database"),
    "email": _tx_build(
        "email",
        "Your Outlook mail: list a folder, search, or read one message by id.",
        {"action": "list | search | read", "query": "search text", "id": "message id (read)",
         "folder": "folder for list (default inbox)", "limit": "optional count"},
        [], _tx_email, risk="read", capability="email", taints=True,
        prerequisite=_tx_has_o365),
    "email_send": _tx_build(
        "email_send",
        "Send an email from your Outlook mailbox (asks you first when approvals are on).",
        {"to": "recipient address(es), comma separated", "subject": "subject",
         "body": "plain-text body", "cc": "optional cc address(es)"},
        ["to", "subject", "body"], _tx_email_send, risk="send", capability="email",
        prerequisite=_tx_has_o365),
    "calendar": _tx_build(
        "calendar",
        "Your Outlook calendar events between start and end (ISO dates; default the "
        "next 7 days).",
        {"start": "optional ISO start", "end": "optional ISO end", "limit": "optional count"},
        [], _tx_calendar, risk="read", capability="calendar", taints=True,
        prerequisite=_tx_has_o365),
    "calendar_create": _tx_build(
        "calendar_create",
        "Create an event in your Outlook calendar, optionally inviting attendees.",
        {"subject": "title", "start": "ISO start", "end": "ISO end",
         "attendees": "optional addresses", "location": "optional place",
         "body": "optional description", "timezone_name": "optional time zone (UTC)"},
        ["subject", "start", "end"], _tx_calendar_create, risk="send", capability="calendar",
        prerequisite=_tx_has_o365),
    "contacts": _tx_build(
        "contacts",
        "Search your Outlook contacts by name, company or address.",
        {"query": "optional text to match", "limit": "optional count"},
        [], _tx_contacts, risk="read", capability="contacts",
        prerequisite=_tx_has_o365),
    "task_scheduler": _tx_build(
        "task_scheduler",
        "Your scheduled agent tasks: create (goal, every_minutes or start_in_minutes, "
        "optional tools), list, or cancel (task_id). A task never gets more tools than you.",
        {"action": "create | list | cancel", "goal": "what the task does each run",
         "name": "optional short name", "every_minutes": "repeat interval (>= 5)",
         "start_in_minutes": "delay before the first run",
         "tools": "optional comma list of tool names", "task_id": "task to cancel"},
        [], _tx_task_scheduler, risk="write", capability="task_scheduler",
        prerequisite=lambda r: r.db is not None),
    "notify": _tx_build(
        "notify",
        "Send the user a short notification (in-app, plus the configured webhook or "
        "macOS banner).",
        {"message": "the notification text", "title": "optional title",
         "level": "info | warning | urgent"},
        ["message"], _tx_notify, risk="send", capability="notify"),
}


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'EXTRA_TOOL_NAMES',
    'register_extra_tools',
    'register_tools_extra',
    'store_notification',
    'list_notifications',
    '_tx_sql_check',
    '_tx_xlsx_sheets',
    '_tx_caller_tools',
]
