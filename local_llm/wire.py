"""One-command wiring of coding tools to this server: `deploy.py wire <client>`.

Why this exists: pointing Claude Code, Aider, Codex, Continue or opencode at a
self-hosted endpoint means finding each tool's own variable names, file
locations and model-name conventions, and the usual result is a half-edited
config with a token pasted into a world-readable file. This module knows each
client's shape and produces either copy-paste instructions (the default,
--print: nothing on disk changes) or, with --write, merges the settings into
the client's own config file:

  * existing files are backed up first (<file>.bak-<YYYYmmdd-HHMMSS>, 0600);
  * only the keys this tool owns are set, every unrelated key is kept, and a
    file that does not parse is left alone (we refuse instead of clobbering);
  * new parent directories are created 0700, and a file that holds the token is
    written 0600 through a temp file and an atomic rename.

Model names follow the compat endpoints' convention: ``raw/<alias>`` is the
model with the client's own tools (what a coding agent wants; it runs its own
tool loop), ``agent/<profile>`` is this app's agent with server-side tools.

The token comes from --token, --login (username/password -> /api/auth/cli/login),
--device (Microsoft sign-in by device code), or $LLM_API_TOKEN.

Hermes Agent (Nous Research) speaks to any OpenAI-compatible endpoint, but its
config file format could not be confirmed offline when this was written, so it
gets environment-variable instructions only and --write refuses for it.
"""

from __future__ import annotations

import argparse
import getpass
import ipaddress
import json
import os
import shlex
import shutil
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .catalog import *  # noqa: F401,F403
from .remote_access import *  # noqa: F401,F403

# Same shape as the server's route annotations: FastAPI resolves them against
# module globals (see appctx.py).
try:
    from fastapi import Request  # noqa: F401
except Exception:  # pragma: no cover
    Request = Any  # type: ignore


WIRE_CLIENTS = ("claude-code", "aider", "codex", "continue", "opencode", "hermes", "generic")
WIRE_TOKEN_PLACEHOLDER = "<YOUR_API_TOKEN>"
# Any non-empty value works when the server runs with AUTH_ENABLED=0, but most
# clients refuse to start without one.
WIRE_NO_AUTH_TOKEN = "no-auth-needed"
WIRE_TOKEN_ENV = "LLM_API_TOKEN"
_WIRE_PROVIDER_ID = "local-llm"
_WIRE_ENTRY_NAME = "Local LLM"
_WIRE_TITLES = {
    "claude-code": "Claude Code", "aider": "Aider", "codex": "Codex CLI",
    "continue": "Continue", "opencode": "opencode", "hermes": "Hermes Agent",
    "generic": "any OpenAI / Anthropic SDK client",
}


# --------------------------------------------------------------------------- #
# Plans: what a client needs, independent of printing or writing             #
# --------------------------------------------------------------------------- #
def wire_model_name(model: Any) -> str:
    """A model name in the compat convention (bare names become raw/<name>)."""
    name = str(model or "").strip()
    if not name:
        name = alias_of(DEFAULT_MODEL) or DEFAULT_MODEL
    # "agent" (the default agent) is a compat name of its own; prefixing it
    # would ask for a served model called "agent" and get a 404.
    if name.startswith(("raw/", "agent/")) or name == "agent":
        return name
    return f"raw/{name}"


def _wire_base(url: str) -> str:
    base = str(url or "").strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


def build_wiring(client: str, url: str, token: str | None, model: Any = None,
                 wire_api: str = "responses") -> dict:
    """Everything one client needs, as data (env, files with merge functions,
    commands, notes). Raises ValueError for an unknown client."""
    client = str(client or "").strip().lower()
    if client not in WIRE_CLIENTS:
        raise ValueError(f"unknown client {client!r}; choose one of: " + ", ".join(WIRE_CLIENTS))
    base = _wire_base(url)
    v1 = base + "/v1"
    tok = token or WIRE_TOKEN_PLACEHOLDER
    mdl = wire_model_name(model)
    plan: dict = {"client": client, "title": _WIRE_TITLES[client], "url": base,
                  "model": mdl, "env": [], "files": [], "commands": [], "notes": [],
                  "writable": True}

    if client == "claude-code":
        env = [("ANTHROPIC_BASE_URL", base), ("ANTHROPIC_AUTH_TOKEN", tok),
               ("ANTHROPIC_MODEL", mdl), ("ANTHROPIC_SMALL_FAST_MODEL", mdl),
               # Newer Claude Code names for the model behind each tier, so a
               # subagent that asks for "haiku" or "sonnet" stays local too.
               ("ANTHROPIC_DEFAULT_HAIKU_MODEL", mdl),
               ("ANTHROPIC_DEFAULT_SONNET_MODEL", mdl),
               ("ANTHROPIC_DEFAULT_OPUS_MODEL", mdl)]
        plan["env"] = env

        def _claude(data: dict) -> dict:
            block = data.get("env")
            if not isinstance(block, dict):
                block = {}
            block.update(dict(env))
            data["env"] = block
            return data

        plan["files"].append(_wire_json_file("~/.claude/settings.json", _claude, True))
        plan["commands"] = ["claude"]
        plan["notes"].append("ANTHROPIC_AUTH_TOKEN is sent as 'Authorization: Bearer'. "
                             "ANTHROPIC_API_KEY (sent as x-api-key) works too, but Claude "
                             "Code then asks once to approve the custom key.")
    elif client == "aider":
        plan["env"] = [("OPENAI_API_BASE", v1), ("OPENAI_API_KEY", tok)]
        pairs = [("openai-api-base", v1), ("openai-api-key", tok),
                 ("model", f"openai/{mdl}"),
                 # The local model is not in litellm's catalogue; without this
                 # aider warns about unknown context limits on every start.
                 ("show-model-warnings", False)]
        plan["files"].append({
            "path": "~/.aider.conf.yml", "format": "yaml", "holds_token": True,
            "merge": lambda text: _wire_merge_flat_yaml(text, pairs)})
        plan["commands"] = [f"aider --model openai/{mdl}"]
    elif client == "codex":
        api = "chat" if str(wire_api).lower() == "chat" else "responses"
        plan["env"] = [(WIRE_TOKEN_ENV, tok)]
        top = [("model", mdl), ("model_provider", "local")]
        table = [("name", _WIRE_ENTRY_NAME), ("base_url", v1),
                 ("env_key", WIRE_TOKEN_ENV), ("wire_api", api)]
        plan["files"].append({
            "path": "~/.codex/config.toml", "format": "toml", "holds_token": False,
            "merge": lambda text: _wire_merge_codex_toml(text, top, "model_providers.local",
                                                         table)})
        plan["commands"] = ["codex"]
        plan["notes"].append(f"Codex reads the key from ${WIRE_TOKEN_ENV} (env_key), so the "
                             "token never lands in config.toml: export it in your shell "
                             "profile.")
        plan["notes"].append(f"wire_api = {api!r}. Current Codex releases speak the "
                             "Responses API; pass --wire-api chat for an older release "
                             "that still uses Chat Completions.")
    elif client == "continue":
        entry = [("name", _WIRE_ENTRY_NAME), ("provider", "openai"), ("model", mdl),
                 ("apiBase", v1), ("apiKey", tok)]
        plan["files"].append({
            "path": "~/.continue/config.yaml", "format": "yaml", "holds_token": True,
            "merge": lambda text: _wire_merge_continue_yaml(text, entry,
                                                            ["chat", "edit", "apply"])})
        plan["commands"] = ["Reload the Continue extension, then pick "
                            f"'{_WIRE_ENTRY_NAME}' in its model menu."]
    elif client == "opencode":
        provider = {"npm": "@ai-sdk/openai-compatible", "name": _WIRE_ENTRY_NAME,
                    "options": {"baseURL": v1, "apiKey": tok},
                    "models": {mdl: {"name": mdl}}}

        def _opencode(data: dict) -> dict:
            data.setdefault("$schema", "https://opencode.ai/config.json")
            providers = data.get("provider")
            if not isinstance(providers, dict):
                providers = {}
            providers[_WIRE_PROVIDER_ID] = provider
            data["provider"] = providers
            data["model"] = f"{_WIRE_PROVIDER_ID}/{mdl}"
            return data

        plan["files"].append(_wire_json_file("~/.config/opencode/opencode.json", _opencode,
                                             True))
        plan["commands"] = [f"opencode --model {_WIRE_PROVIDER_ID}/{mdl}"]
    elif client == "hermes":
        plan["env"] = [("OPENAI_BASE_URL", v1), ("OPENAI_API_KEY", tok)]
        plan["writable"] = False
        plan["commands"] = ["hermes model    # choose the custom OpenAI-compatible endpoint"]
        plan["notes"].append(
            "Hermes Agent's own config file format could not be confirmed offline, so "
            "nothing is written for it. Point its custom OpenAI-compatible endpoint at "
            f"{v1} with the key above and the model name {mdl} (for example with "
            "`hermes model`, or in ~/.hermes/.env if your version reads it).")
    else:  # generic
        plan["env"] = [("OPENAI_BASE_URL", v1), ("OPENAI_API_BASE", v1),
                       ("OPENAI_API_KEY", tok), ("ANTHROPIC_BASE_URL", base),
                       ("ANTHROPIC_API_KEY", tok), ("LLM_MODEL", mdl)]
        plan["writable"] = False
        plan["notes"].append("OpenAI SDKs read OPENAI_BASE_URL (older ones OPENAI_API_BASE); "
                             "Anthropic SDKs read ANTHROPIC_BASE_URL and send the key as "
                             "x-api-key. Use the model name above in requests.")
    return plan


def _wire_json_file(path: str, update: Callable[[dict], dict], holds_token: bool) -> dict:
    return {"path": path, "format": "json", "holds_token": holds_token,
            "merge": lambda text: _wire_merge_json(text, update)}


# --------------------------------------------------------------------------- #
# Merges: set only our keys, keep everything else, refuse what we cannot parse #
# --------------------------------------------------------------------------- #
def _wire_merge_json(text: str | None, update: Callable[[dict], dict]) -> str:
    data: Any = {}
    if text and text.strip():
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise ValueError(f"existing file is not valid JSON ({exc}); not touching it")
        if not isinstance(data, dict):
            raise ValueError("existing file is not a JSON object; not touching it")
    return json.dumps(update(data), indent=2, ensure_ascii=False) + "\n"


def _wire_yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    # A JSON string is a valid YAML double-quoted scalar.
    return json.dumps(str(value), ensure_ascii=False)


def _wire_merge_flat_yaml(text: str | None, pairs: list) -> str:
    """Set top-level `key: value` pairs in a flat YAML file (aider's)."""
    wanted = dict(pairs)
    lines = (text or "").splitlines()
    out: list[str] = []
    done: set[str] = set()
    skipping = False
    for line in lines:
        if skipping:
            # Drop the old value's continuation (indented or list lines).
            if line.strip() and line[:1] in (" ", "\t", "-"):
                continue
            skipping = False
        key = line.split(":", 1)[0].strip() if ":" in line and line[:1] not in (" ", "\t", "#", "-") else ""
        if key in wanted and key not in done:
            out.append(f"{key}: {_wire_yaml_scalar(wanted[key])}")
            done.add(key)
            skipping = True
            continue
        out.append(line)
    missing = [k for k, _ in pairs if k not in done]
    if missing:
        if out and out[-1].strip():
            out.append("")
        out.append("# Added by deploy.py wire aider")
        out.extend(f"{k}: {_wire_yaml_scalar(wanted[k])}" for k in missing)
    return "\n".join(out) + "\n"


def _wire_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(str(value), ensure_ascii=False)


def _wire_merge_codex_toml(text: str | None, top: list, table_name: str, table: list) -> str:
    """Set top-level keys and replace one [table] in a TOML file (Codex's)."""
    lines = (text or "").splitlines()
    first_table = next((i for i, ln in enumerate(lines) if ln.strip().startswith("[")),
                       len(lines))
    head, rest = lines[:first_table], lines[first_table:]
    done: set[str] = set()
    for i, line in enumerate(head):
        key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else ""
        for name, value in top:
            if key == name and name not in done:
                head[i] = f"{name} = {_wire_toml_value(value)}"
                done.add(name)
    # New top-level keys go at the end of the top-level region, before the first
    # table, or TOML would read them as members of that table.
    while head and not head[-1].strip():
        head.pop()
    head += [f"{name} = {_wire_toml_value(value)}" for name, value in top if name not in done]
    block = [f"[{table_name}]"] + [f"{k} = {_wire_toml_value(v)}" for k, v in table]
    header = f"[{table_name}]"
    start = next((i for i, ln in enumerate(rest) if ln.strip() == header), None)
    if start is None:
        while rest and not rest[-1].strip():
            rest.pop()
        rest = rest + ([""] if rest else []) + block
    else:
        end = next((j for j in range(start + 1, len(rest))
                    if rest[j].strip().startswith("[")), len(rest))
        tail = rest[end:]
        rest = rest[:start] + block + ([""] if tail else []) + tail
    merged = "\n".join(head + ([""] if head and rest else []) + rest).strip("\n") + "\n"
    try:
        import tomllib
    except Exception:  # pragma: no cover  (Python < 3.11)
        return merged
    try:
        tomllib.loads(merged)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"could not merge into the existing config.toml ({exc}); "
                         "not touching it, add the settings by hand")
    return merged


def _wire_continue_entry(entry: list, roles: list, indent: int) -> list[str]:
    pad = " " * indent
    out = []
    for i, (key, value) in enumerate(entry):
        lead = f"{pad}- " if i == 0 else f"{pad}  "
        out.append(f"{lead}{key}: {_wire_yaml_scalar(value)}")
    out.append(f"{pad}  roles:")
    out.extend(f"{pad}    - {role}" for role in roles)
    return out


def _wire_merge_continue_yaml(text: str | None, entry: list, roles: list) -> str:
    """Add or replace our entry in a Continue config.yaml `models:` list."""
    name = dict(entry)["name"]
    lines = (text or "").splitlines()
    if not any(ln.strip() for ln in lines):
        return "\n".join(["name: Local Assistant", "version: 1.0.0", "schema: v1",
                          "models:"] + _wire_continue_entry(entry, roles, 2)) + "\n"
    idx = next((i for i, ln in enumerate(lines)
                if ln.startswith("models:") and ln.split(":", 1)[1].strip() in ("", "[]")),
               None)
    if idx is None:
        if any(ln.startswith("models:") for ln in lines):
            raise ValueError("config.yaml has an inline 'models:' value; not touching it")
        while lines and not lines[-1].strip():
            lines.pop()
        return "\n".join(lines + ["models:"] + _wire_continue_entry(entry, roles, 2)) + "\n"
    lines[idx] = "models:"
    # The block runs until the next top-level key (a column-0 "- " item is still
    # part of it: YAML allows a list flush with its key).
    end = len(lines)
    for j in range(idx + 1, len(lines)):
        ln = lines[j]
        if ln.strip() and not ln.lstrip().startswith("#") and ln[:1] not in (" ", "\t", "-"):
            end = j
            break
    items = [j for j in range(idx + 1, end) if lines[j].lstrip().startswith("- ")]
    indent = (len(lines[items[0]]) - len(lines[items[0]].lstrip())) if items else 2
    new = _wire_continue_entry(entry, roles, indent)
    for j in items:
        ln = lines[j]
        if len(ln) - len(ln.lstrip()) != indent:
            continue
        value = ln.strip()[2:].strip()
        if value.startswith("name:") and value.split(":", 1)[1].strip().strip("\"'") == name:
            stop = next((k for k in range(j + 1, end)
                         if lines[k].strip() and (len(lines[k]) - len(lines[k].lstrip())) <= indent),
                        end)
            return "\n".join(lines[:j] + new + lines[stop:]) + "\n"
    return "\n".join(lines[:idx + 1] + new + lines[idx + 1:]) + "\n"


# --------------------------------------------------------------------------- #
# Output                                                                      #
# --------------------------------------------------------------------------- #
def _wire_expand(path: str, home: Path | None = None) -> Path:
    if path.startswith("~/"):
        return (Path(home) if home else Path.home()) / path[2:]
    return Path(path)


def render_wiring(plan: dict) -> str:
    """The copy-paste instructions for a plan (what --print shows)."""
    out = [f"# Connect {plan['title']} to {plan['url']}  (model {plan['model']})"]
    if plan["env"]:
        out += ["", "# Environment (put these in your shell profile to keep them):"]
        out += [f"export {k}={shlex.quote(str(v))}" for k, v in plan["env"]]
    for spec in plan["files"]:
        out += ["", f"# {spec['path']}  (deploy.py wire {plan['client']} --write merges "
                    "this into the existing file):"]
        out += spec["merge"]("").rstrip("\n").splitlines()
    if plan["commands"]:
        out += ["", "# Then:"] + list(plan["commands"])
    for note in plan["notes"]:
        out += ["", f"# Note: {note}"]
    return "\n".join(out) + "\n"


def _wire_mkdirs(path: Path) -> None:
    """Create missing parents 0700 each (os.makedirs' mode skips intermediates)."""
    missing = []
    cur = path
    while not cur.exists():
        missing.append(cur)
        if cur.parent == cur:
            break
        cur = cur.parent
    for d in reversed(missing):
        d.mkdir(mode=0o700)
        os.chmod(d, 0o700)


def _wire_write_file(path: Path, text: str, holds_token: bool) -> Path | None:
    """Back up, then atomically replace `path`. Returns the backup path."""
    if path.is_symlink():
        # Dotfile managers symlink these; write through to the real file
        # instead of replacing the link with a plain file.
        path = path.resolve()
    _wire_mkdirs(path.parent)
    backup = None
    mode = 0o600 if holds_token else 0o644
    if path.exists():
        if not holds_token:
            mode = path.stat().st_mode & 0o777
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = path.with_name(f"{path.name}.bak-{stamp}")
        n = 1
        while backup.exists():
            backup = path.with_name(f"{path.name}.bak-{stamp}-{n}")
            n += 1
        shutil.copy2(path, backup)
        os.chmod(backup, 0o600)   # an old token may be in it
    tmp = path.with_name(f".{path.name}.wire-tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return backup


def apply_wiring(plan: dict, home: Path | None = None) -> list[dict]:
    """Merge every file of a plan into place. Returns [{path, backup}].

    All merges are computed before anything is written, so a file we refuse to
    touch never leaves the others half-applied.
    """
    if not plan.get("writable"):
        raise ValueError(f"--write is not supported for {plan['client']}; "
                         "use the printed settings instead")
    staged = []
    for spec in plan["files"]:
        path = _wire_expand(spec["path"], home)
        real = path.resolve() if path.is_symlink() else path
        existing = real.read_text(encoding="utf-8") if real.exists() else None
        try:
            merged = spec["merge"](existing)
        except ValueError as exc:
            raise ValueError(f"{spec['path']}: {exc}")
        staged.append((path, merged, spec["holds_token"]))
    results = []
    for path, merged, holds in staged:
        backup = _wire_write_file(path, merged, holds)
        results.append({"path": str(path), "backup": str(backup) if backup else None})
    return results


# --------------------------------------------------------------------------- #
# Server calls (the CLI side)                                                 #
# --------------------------------------------------------------------------- #
def _wire_http(method: str, url: str, body: dict | None = None,
               headers: dict | None = None, timeout: float = 10.0) -> tuple[int, dict]:
    """One JSON request with the standard library. (status, body); 0 = unreachable."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8", "replace")
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace") if exc.fp else ""
        status = exc.code
    except (urllib.error.URLError, OSError, ValueError):
        return 0, {}
    try:
        parsed = json.loads(raw) if raw else {}
    except ValueError:
        parsed = {"error": raw[:200]}
    return status, parsed if isinstance(parsed, dict) else {"data": parsed}


def _wire_is_loopback(url: str) -> bool:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _wire_default_url() -> str:
    explicit = os.environ.get("LLM_BASE_URL") or os.environ.get("PUBLIC_BASE_URL")
    if explicit:
        return explicit.strip().rstrip("/")
    return f"http://127.0.0.1:{os.environ.get('WEB_PORT', '8000')}"


def _wire_server_model(url: str, token: str | None) -> str | None:
    """The first raw/ model the server lists, if it answers."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    status, body = _wire_http("GET", url + "/v1/models", headers=headers, timeout=5)
    if status != 200:
        return None
    ids = [str(m.get("id")) for m in (body.get("data") or []) if isinstance(m, dict) and m.get("id")]
    raw = [i for i in ids if i.startswith("raw/")]
    if raw:
        return raw[0]
    # No raw/ entries means V1_RAW_ENABLED=0: a plain id would be rewritten
    # to raw/<id> and refused, so wire the client to the agent instead.
    return "agent" if "agent" in ids else (ids[0] if ids else None)


def _wire_login(url: str, client: str, ask: Callable | None = None,
                ask_secret: Callable | None = None, out=sys.stderr) -> str | None:
    # Looked up at call time (not bound as defaults) so a test can patch them.
    username = (ask or input)("Username: ").strip()
    password = (ask_secret or getpass.getpass)("Password: ")
    status, body = _wire_http("POST", url + "/api/auth/cli/login", {
        "username": username, "password": password,
        "name": f"wire {client} on {socket.gethostname()[:40]}"})
    if status == 200 and body.get("token"):
        return str(body["token"])
    if body.get("auth_enabled") is False:
        return WIRE_NO_AUTH_TOKEN
    msg = body.get("error") or ("server unreachable" if status == 0 else f"HTTP {status}")
    print(f"Login failed: {msg}", file=out)
    return None


def _wire_device_login(url: str, client: str, out=sys.stderr,
                       sleep: Callable = time.sleep) -> str | None:
    status, start = _wire_http("POST", url + "/api/auth/device/start", {})
    if status != 200 or not start.get("device_code"):
        print(f"Microsoft sign-in unavailable: {start.get('error') or f'HTTP {status}'}",
              file=out)
        return None
    print(start.get("message") or (f"Open {start.get('verification_uri')} and enter the "
                                   f"code {start.get('user_code')}"), file=out)
    interval = max(1, int(start.get("interval") or 5))
    deadline = time.monotonic() + int(start.get("expires_in") or 900)
    while time.monotonic() < deadline:
        sleep(interval)
        status, body = _wire_http("POST", url + "/api/auth/device/poll", {
            "device_code": start["device_code"],
            "name": f"wire {client} on {socket.gethostname()[:40]}"})
        state = body.get("status")
        if status == 200 and state == "ok" and body.get("token"):
            return str(body["token"])
        if state in ("pending", "slow_down"):
            interval = max(interval, int(body.get("interval") or interval))
            continue
        print(f"Sign-in failed: {body.get('error') or state or f'HTTP {status}'}", file=out)
        return None
    print("Sign-in timed out.", file=out)
    return None


def wire_cli(argv: list[str], out=None, err=None, home: Path | None = None) -> int:
    """`deploy.py wire <client> [--url U] [--token T|--login|--device] [--model M]
    [--write|--print]`."""
    out = out or sys.stdout
    err = err or sys.stderr
    parser = argparse.ArgumentParser(
        prog="deploy.py wire",
        description="Connect a coding tool to this server. Prints the settings by "
                    "default; --write merges them into the tool's config (with a backup).")
    parser.add_argument("client", nargs="?", choices=WIRE_CLIENTS)
    parser.add_argument("--url", default=None,
                        help="server base URL (default $LLM_BASE_URL, else "
                             "http://127.0.0.1:$WEB_PORT)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--token", default=None, help=f"API token (default ${WIRE_TOKEN_ENV})")
    source.add_argument("--login", action="store_true",
                        help="sign in with username and password to get a token")
    source.add_argument("--device", action="store_true",
                        help="sign in with Microsoft (device code) to get a token")
    parser.add_argument("--model", default=None, help="model name (default: the server's "
                        "first raw/ model, else raw/<this machine's default alias>)")
    parser.add_argument("--wire-api", choices=("responses", "chat"), default="responses",
                        help="Codex only: which OpenAI API Codex speaks")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true",
                      help="merge into the client's config file (backup first)")
    mode.add_argument("--print", dest="print_only", action="store_true",
                      help="print the settings only (the default)")
    parser.add_argument("--no-probe", action="store_true",
                        help="do not contact the server (no auth or model discovery)")
    args = parser.parse_args(argv)
    if not args.client:
        parser.print_help(file=out)
        return 2
    url = _wire_base(args.url or _wire_default_url())
    if not url.startswith(("http://", "https://")):
        print(f"--url must start with http:// or https:// (got {url!r})", file=err)
        return 2

    auth_enabled = None
    if not args.no_probe:
        status, cfg = _wire_http("GET", url + "/api/auth/config", timeout=5)
        if status == 200:
            auth_enabled = bool(cfg.get("auth_enabled"))
        elif status == 0:
            print(f"(server at {url} did not answer; continuing offline)", file=err)
    token = args.token or os.environ.get(WIRE_TOKEN_ENV) or None
    if args.login:
        token = _wire_login(url, args.client, out=err)
        if not token:
            return 1
    elif args.device:
        token = _wire_device_login(url, args.client, out=err)
        if not token:
            return 1
    if not token and auth_enabled is False:
        token = WIRE_NO_AUTH_TOKEN
        if not _wire_is_loopback(url):
            print("WARNING: this server runs without authentication (AUTH_ENABLED=0) and "
                  "is not on loopback: anyone who can reach it is an administrator.",
                  file=err)
    if url.startswith("http://") and not _wire_is_loopback(url) and token:
        print("WARNING: plain http to a remote host sends the token in clear text; "
              "use https (deploy.py remote tailscale|cloudflare).", file=err)
    model = args.model
    if not model and not args.no_probe:
        model = _wire_server_model(url, token if token not in (None, WIRE_NO_AUTH_TOKEN) else None)
    plan = build_wiring(args.client, url, token, model, wire_api=args.wire_api)

    if not args.write:
        out.write(render_wiring(plan))
        if not token:
            print(f"\n(no token: replace {WIRE_TOKEN_PLACEHOLDER}; create one in Settings, "
                  "or rerun with --login / --device / --token)", file=err)
        return 0
    if not plan["writable"]:
        out.write(render_wiring(plan))
        print(f"\n--write is not supported for {args.client}; use the settings above.",
              file=err)
        return 2
    if not token and any(f["holds_token"] for f in plan["files"]):
        print("--write needs a token: pass --token, --login or --device, or set "
              f"${WIRE_TOKEN_ENV}.", file=err)
        return 2
    try:
        results = apply_wiring(plan, home=home)
    except (ValueError, OSError) as exc:
        print(f"Nothing written: {exc}", file=err)
        return 1
    for item in results:
        line = f"wrote {item['path']}"
        if item["backup"]:
            line += f"  (backup: {item['backup']})"
        print(line, file=out)
    exports = [(k, v) for k, v in plan["env"]] if args.client == "codex" else []
    if exports:
        print("\n# Also export (Codex reads the token from the environment):", file=out)
        for k, v in exports:
            print(f"export {k}={shlex.quote(str(v))}", file=out)
    if plan["commands"]:
        print("\n# Then: " + plan["commands"][-1], file=out)
    return 0


# --------------------------------------------------------------------------- #
# Route for the UI's "Connect a client" panel                                 #
# --------------------------------------------------------------------------- #
def register_wire(app, ctx) -> None:
    """GET /api/integrations/snippets?client= : the --print text, token as a
    placeholder (a token is only ever shown by the route that creates it)."""
    from fastapi.responses import JSONResponse

    config, USER = ctx.config, ctx.USER

    @app.get("/api/integrations/snippets")
    def wire_snippets(request: Request, client: str = "", _u: dict = USER):
        clients = [{"id": c, "title": _WIRE_TITLES[c]} for c in WIRE_CLIENTS]
        if not client:
            return {"clients": clients}
        try:
            base = public_url_for(request, config, "")
        except Exception:  # noqa: BLE001
            base = f"http://127.0.0.1:{config.web_port}"
        model = alias_of(config.model, config) or config.model
        if not getattr(config, "v1_raw_enabled", True):
            model = "agent"            # raw/<alias> would answer 403
        try:
            plan = build_wiring(client, base, None, model)
        except ValueError as exc:
            return JSONResponse({"error": str(exc), "clients": clients}, status_code=400)
        return {
            "client": plan["client"], "title": plan["title"], "url": plan["url"],
            "model": plan["model"], "token_placeholder": WIRE_TOKEN_PLACEHOLDER,
            "auth_enabled": bool(config.auth_enabled),
            "env": [{"name": k, "value": v} for k, v in plan["env"]],
            "files": [{"path": f["path"], "format": f["format"], "content": f["merge"]("")}
                      for f in plan["files"]],
            "commands": plan["commands"], "notes": plan["notes"],
            "command": f"python deploy.py wire {plan['client']} --url {plan['url']} --login",
            "text": render_wiring(plan),
        }


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    "WIRE_CLIENTS",
    "WIRE_NO_AUTH_TOKEN",
    "WIRE_TOKEN_ENV",
    "WIRE_TOKEN_PLACEHOLDER",
    "_wire_base",
    "_wire_default_url",
    "_wire_device_login",
    "_wire_expand",
    "_wire_http",
    "_wire_is_loopback",
    "_wire_json_file",
    "_wire_login",
    "_wire_merge_codex_toml",
    "_wire_merge_continue_yaml",
    "_wire_merge_flat_yaml",
    "_wire_merge_json",
    "_wire_mkdirs",
    "_wire_server_model",
    "_wire_toml_value",
    "_wire_continue_entry",
    "_wire_write_file",
    "_wire_yaml_scalar",
    "apply_wiring",
    "build_wiring",
    "register_wire",
    "render_wiring",
    "wire_cli",
    "wire_model_name",
]
