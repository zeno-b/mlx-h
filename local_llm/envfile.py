"""A tiny standard-library .env loader and writer, run before anything reads the environment.

Why it exists: core.py computes the RAM tiers, APP_NAME and a handful of other
constants at IMPORT time from os.environ, and Config reads the rest (MODEL_ID
included) when it is built. Until now `.env` only took effect when the shell or a
process manager exported it, which `deploy.py init` and the launchd service
cannot rely on. deploy.py and `python -m local_llm` therefore load the file
before the package is imported (deploy.py loads this module by path so the
package __init__, which imports everything, does not run first).

The model settings (MODEL_ID, MODEL_ALIASES, WHISPER_MODEL, TTS_MODEL) live
ONLY here: nothing is built in, and when the web UI is told which model to run
it writes the choice back with update_dotenv(), so the next start reads the
same value from the same place.

The rules are deliberately boring so a value means the same thing here, in a
shell (`set -a; . ./.env`) and in `env $(cat .env | xargs)`:

  * KEY=VALUE per line; blank lines and `#` comment lines are skipped;
  * an optional `export ` prefix is accepted;
  * 'single quotes' are literal; "double quotes" understand \\n \\t \\" \\\\ \\$;
  * an unquoted value ends at ` #` (an inline comment) and is stripped;
  * NO interpolation: `$HOME` or `${X}` stays exactly that text, so a password
    containing `$` is never silently rewritten;
  * a variable already set in the environment always wins (override=False), so
    `MODEL_ID=x python3 deploy.py` and launchd's own variables keep the last word.

Lines that do not parse are skipped and reported by parse_dotenv (and by
`deploy.py --doctor`), never fatal: a typo in .env must not stop the app.

LOCAL_LLM_ENV_FILE points at another file; LOCAL_LLM_DOTENV=0 turns loading off.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

# Same rule as core.ROOT: the project root holds deploy.py, one level above this
# file inside the package, or this file's own directory in the single-file build.
_ENVFILE_MODULE_DIR = Path(__file__).resolve().parent
ENVFILE_ROOT = (_ENVFILE_MODULE_DIR.parent if (_ENVFILE_MODULE_DIR / "__init__.py").exists()
                else _ENVFILE_MODULE_DIR)

# Environment switches. The marker records which file was applied; it doubles as
# the re-exec guard in __main__.py and lets --doctor say what was loaded.
ENV_FILE_VAR = "LOCAL_LLM_ENV_FILE"
DOTENV_SWITCH_VAR = "LOCAL_LLM_DOTENV"
DOTENV_LOADED_VAR = "LOCAL_LLM_ENV_LOADED"

_ENVFILE_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENVFILE_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "$": "$"}
# A value that needs no quoting in .env, a shell or launchd.
_ENVFILE_BARE_VALUE = re.compile(r"[A-Za-z0-9_./:@,+=-]*")
# One writer at a time: two Settings saves racing would otherwise each read
# the old file and the second rename would drop the first one's change.
_ENVFILE_WRITE_LOCK = threading.Lock()


def dotenv_path() -> Path:
    """The .env this process reads: LOCAL_LLM_ENV_FILE, else <project root>/.env."""
    override = os.environ.get(ENV_FILE_VAR, "").strip()
    return Path(override).expanduser() if override else ENVFILE_ROOT / ".env"


def _envfile_double_quoted(body: str) -> tuple[str, int] | None:
    """(value, index after the closing quote) for text after an opening ", or None."""
    out = []
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            out.append(_ENVFILE_ESCAPES.get(nxt, "\\" + nxt))
            i += 2
            continue
        if ch == '"':
            return "".join(out), i + 1
        out.append(ch)
        i += 1
    return None


def _envfile_parse_value(raw: str) -> tuple[str | None, str]:
    """(value, problem). value is None when the line cannot be used at all."""
    raw = raw.lstrip()
    if not raw:
        return "", ""
    quote = raw[0]
    if quote in ("'", '"'):
        if quote == "'":
            end = raw.find("'", 1)
            parsed = (raw[1:end], end + 1) if end != -1 else None
        else:
            parsed = _envfile_double_quoted(raw[1:])
            if parsed is not None:
                parsed = (parsed[0], parsed[1] + 1)
        if parsed is None:
            # An unclosed quote is a typo, not a multi-line value: swallowing the
            # rest of the file into one variable would be the surprise to avoid.
            return None, f"unclosed {quote} quote"
        value, after = parsed
        rest = raw[after:].strip()
        if rest and not rest.startswith("#"):
            return value, f"text after the closing quote ignored: {rest[:20]!r}"
        return value, ""
    # Unquoted: an inline comment starts at whitespace followed by '#', so a
    # value such as `abc#123` (no space) keeps its '#'.
    cut = re.search(r"\s#", raw)
    value = (raw[:cut.start()] if cut else raw).strip()
    return value, ""


def parse_dotenv(text: str) -> tuple[dict[str, str], list[str]]:
    """({KEY: value}, problems). Later duplicates win, as when a shell sources it."""
    values: dict[str, str] = {}
    problems: list[str] = []
    seen: set[str] = set()
    for number, line in enumerate(str(text or "").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export ") or stripped.startswith("export\t"):
            stripped = stripped[len("export"):].lstrip()
        if "=" not in stripped:
            problems.append(f"line {number}: no '=' (ignored)")
            continue
        key, raw = stripped.split("=", 1)
        key = key.strip()
        if not _ENVFILE_KEY.match(key):
            problems.append(f"line {number}: {key[:40]!r} is not a variable name (ignored)")
            continue
        value, problem = _envfile_parse_value(raw)
        if problem:
            problems.append(f"line {number}: {key}: {problem}")
        if value is None:
            continue
        if key in seen:
            problems.append(f"line {number}: {key} is set again (the later value wins)")
        seen.add(key)
        values[key] = value
    return values, problems


def read_dotenv(path: Path | str | None = None) -> dict[str, str]:
    """The variables a .env file defines ({} when it does not exist)."""
    target = Path(path) if path is not None else dotenv_path()
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    return parse_dotenv(text)[0]


def load_dotenv(path: Path | str | None = None, override: bool = False,
                environ: dict | None = None) -> dict[str, str]:
    """Apply a .env file to the environment; returns the variables it actually set.

    Never overrides a variable that is already set unless override=True. A
    missing or unreadable file is not an error (every setting has a default).
    """
    env = os.environ if environ is None else environ
    if str(env.get(DOTENV_SWITCH_VAR, "1")).strip() == "0":
        return {}
    target = Path(path) if path is not None else dotenv_path()
    try:
        text = target.read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        return {}
    except OSError as exc:
        import sys
        print(f"[local-llm] not reading {target}: {exc}", file=sys.stderr)
        return {}
    values, _problems = parse_dotenv(text)
    applied: dict[str, str] = {}
    for key, value in values.items():
        if not override and key in env:
            continue
        env[key] = value
        applied[key] = value
    env[DOTENV_LOADED_VAR] = str(target)
    return applied


def env_line(key: str, value: object) -> str:
    """KEY=value that this loader, `set -a; . .env` and launchd all read the same."""
    text = str(value)
    if _ENVFILE_BARE_VALUE.fullmatch(text):
        return f"{key}={text}"
    if "'" not in text and "\n" not in text:
        return f"{key}='{text}'"
    escaped = (text.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")
               .replace("\n", "\\n"))
    return f'{key}="{escaped}"'


def _envfile_key_of(line: str) -> str:
    """The variable an active KEY=VALUE line sets, or "" for anything else."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return ""
    if stripped.startswith("export ") or stripped.startswith("export\t"):
        stripped = stripped[len("export"):].lstrip()
    key = stripped.split("=", 1)[0].strip() if "=" in stripped else ""
    return key if _ENVFILE_KEY.match(key) else ""


def _envfile_placeholder_of(line: str) -> str:
    """The variable a commented-out `# KEY=...` example line names, or ""."""
    stripped = line.strip()
    if not stripped.startswith("#"):
        return ""
    return _envfile_key_of(stripped.lstrip("#").strip())


def update_dotenv(updates: dict[str, object | None], path: Path | str | None = None,
                  environ: dict | None = None) -> Path:
    """Set (or, with None, remove) variables in the .env file, keeping everything else.

    An existing KEY= line is rewritten in place and any later duplicate of it
    dropped; a new key goes right after its commented `# KEY=` example when the
    file has one (so it stays in its section), else at the end. Comments, order
    and every other line survive byte for byte. The file is replaced atomically
    and stays private (0600): it holds secrets. The same values are applied to
    the environment, so this process and the next start agree.

    Raises OSError when the file cannot be written and ValueError for a key
    that is not a variable name. Loading being switched off (LOCAL_LLM_DOTENV=0)
    is the caller's to report: the file is still the one place to keep it.
    """
    for key in updates:
        if not _ENVFILE_KEY.match(str(key)):
            raise ValueError(f"{key!r} is not an environment variable name")
    env = os.environ if environ is None else environ
    target = Path(path) if path is not None else dotenv_path()
    with _ENVFILE_WRITE_LOCK:
        try:
            lines = target.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            lines = ["# Local LLM settings. Variables already set in the environment win.",
                     "# This file holds secrets: keep it private (mode 600), never commit it."]
        pending = dict(updates)
        out: list[str] = []
        for line in lines:
            key = _envfile_key_of(line)
            if key and key in updates:
                if key in pending:
                    value = pending.pop(key)
                    if value is not None:
                        out.append(env_line(key, value))
                continue        # a removed key, or a duplicate of one just written
            out.append(line)
        # Keys not in the file yet: after their commented example, else at the end.
        for key in [k for k, v in pending.items() if v is not None]:
            spot = next((i for i, line in enumerate(out)
                         if _envfile_placeholder_of(line) == key), None)
            if spot is None:
                out.append(env_line(key, pending[key]))
            else:
                out.insert(spot + 1, env_line(key, pending[key]))
        text = "\n".join(out).rstrip("\n") + "\n"
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(tmp, 0o600)
        os.replace(tmp, target)
    for key, value in updates.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)
    return target


# The single-file build (deploy_bundled.py) has no deploy.py to run the loader
# first. bundle.py lists this module first, so applying .env here still lands
# before core computes its import-time defaults. Inside the package this never
# runs: deploy.py and __main__.py load .env themselves.
if not (_ENVFILE_MODULE_DIR / "__init__.py").exists():
    load_dotenv()


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'DOTENV_LOADED_VAR',
    'DOTENV_SWITCH_VAR',
    'ENVFILE_ROOT',
    'ENV_FILE_VAR',
    '_envfile_double_quoted',
    '_envfile_key_of',
    '_envfile_parse_value',
    '_envfile_placeholder_of',
    'dotenv_path',
    'env_line',
    'load_dotenv',
    'parse_dotenv',
    'read_dotenv',
    'update_dotenv',
]
