"""First-run setup: `deploy.py init`, `deploy.py service`, and the install
checks `deploy.py --doctor` prints after its port report.

init writes .env from a RAM-aware recommendation, so a fresh Mac starts on a
model that fits it with headroom, with a context window and admission limits
sized to match. It is the step install.sh (and the Homebrew wrapper) run after
the checkout exists:

  * the model: an applied benchmark recommendation for this machine
    (`deploy.py recommend --apply`) when there is one, otherwise core's RAM
    tier, stepped down while it would not fit with headroom;
  * --multi-user turns login on, binds all interfaces and generates a strong
    AUTH_ADMIN_PASSWORD (printed once);
  * --cluster primary also generates NODE_TOKEN; --cluster secondary pairs
    with the primary through the existing `deploy.py cluster join` flow, which
    adds the node token the primary issues to the file just written;
  * an existing .env is never overwritten without --force: init writes
    .env.new beside it (a complete drop-in file, with every setting init does
    not manage carried over) and prints what differs. --force keeps a .env.bak.

init also creates ./.venv and installs the dependencies first, through the
same core.bootstrap() the first server launch uses (--skip-deps to skip it).

service manages a launchd LaunchAgent that runs the venv's python on
deploy.py with the checkout as its working directory, restarts it when it
crashes, and logs to logs/. `service status` only reads (launchctl print).
"""

from __future__ import annotations

import argparse
import os
import platform
import plistlib
import re
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .core import (DB_PATH, DEFAULT_MODEL, REQUIRED_MODULES, ROOT, TOTAL_RAM_GB,
                   _default_concurrent_generations, _default_context_for_ram,
                   _default_model_for_ram, bootstrap, in_venv)
from .catalog import fits_in_ram, model_metadata, read_recommendation, resolve_model
from .envfile import DOTENV_LOADED_VAR, ENV_FILE_VAR, parse_dotenv
from .sharding import cluster_cli
from .sysutil import hf_cache_dir, model_is_cached, port_open

SERVICE_LABEL = "local.llm.server"
# The RAM tiers core._default_model_for_ram picks from, largest first. init
# walks down this ladder while a model would not fit with headroom.
_SETUP_LADDER = (
    "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
    "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
    "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
)
_SETUP_SECRET_KEY = re.compile(r"PASSWORD|TOKEN|SECRET|(^|_)KEY$")
_SETUP_BARE_VALUE = re.compile(r"[A-Za-z0-9_./:@,+=-]*")
_SETUP_TRUE_WORDS = ("true", "yes", "on")
_SETUP_FALSE_WORDS = ("false", "no", "off")
# launchd starts jobs with PATH=/usr/bin:/bin:/usr/sbin:/sbin; the app shells
# out to git, tailscale and Homebrew tools, so the service gets the usual PATH.
_SETUP_SERVICE_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


# --------------------------------------------------------------------------- #
# The recommendation
# --------------------------------------------------------------------------- #

def setup_headroom_gb(ram_gb: float) -> float:
    """RAM kept free beside the weights: the benchmark's default rule."""
    return max(2.5, float(ram_gb) * 0.25)


def setup_model_fits(model_id: str, ram_gb: float, guess: bool = False) -> bool | None:
    """Whether a model fits this Mac with headroom; None for an unknown model.

    Two conditions: the weights plus runtime fit the GPU's default share of
    RAM (catalog.fits_in_ram), and weights + runtime + headroom fit the RAM, so
    the KV cache, the web process and a LoRA run still have room.
    """
    meta = model_metadata(model_id, guess=guess)
    if not meta or not meta.get("approx_size_gb"):
        return None
    if fits_in_ram(meta, ram_gb) is False:
        return False
    return float(meta["approx_size_gb"]) + 1.0 + setup_headroom_gb(ram_gb) <= float(ram_gb)


def recommend_starter(ram_gb: float | None = None, data_dir: Path | None = None) -> dict:
    """The model, context and admission limits init writes for this Mac.

    Pure: reads only the applied recommendation under data_dir, never the
    environment (which already holds the current .env when init runs).
    """
    ram = float(TOTAL_RAM_GB if ram_gb is None else ram_gb)
    data_dir = Path(data_dir) if data_dir is not None else ROOT / "data"
    applied = read_recommendation(data_dir, ram_gb=ram)
    source, reason = "tier", f"RAM tier for {ram:.0f}GB"
    model, context = "", 0
    if applied and setup_model_fits(str(applied["model_id"]), ram) is not False:
        model = str(applied["model_id"])
        try:
            context = int(applied.get("context_size") or 0)
        except (TypeError, ValueError):
            context = 0
        source, reason = "benchmark", "measured on this Mac (deploy.py recommend --apply)"
    if not model:
        # A path that cannot exist keeps core's helpers on the tiers alone.
        nowhere = data_dir / "__no_recommendation__.json"
        model = _default_model_for_ram(ram, nowhere)
        ladder = list(_SETUP_LADDER)
        start = ladder.index(model) if model in ladder else len(ladder)
        for candidate in ladder[start:]:
            model = candidate
            if setup_model_fits(candidate, ram) is not False:
                break
        if model != _default_model_for_ram(ram, nowhere):
            reason += ", stepped down to keep headroom"
    if not 512 <= context <= 131072:
        context = _default_context_for_ram(ram, data_dir / "__no_recommendation__.json")
    meta = model_metadata(model, guess=False) or {}
    return {
        "model_id": model,
        "context_size": int(context),
        "approx_size_gb": meta.get("approx_size_gb"),
        "max_concurrent_generations": _default_concurrent_generations(ram),
        # The same RAM-scaled floor config.admission_min_free_mb defaults to.
        "admission_min_free_mb": 384 if ram < 14 else 768 if ram <= 24 else 1536,
        "ram_gb": round(ram, 1),
        "headroom_gb": round(setup_headroom_gb(ram), 1),
        "source": source,
        "reason": reason,
    }


# --------------------------------------------------------------------------- #
# .env rendering
# --------------------------------------------------------------------------- #

def setup_is_secret(key: str) -> bool:
    return bool(_SETUP_SECRET_KEY.search(str(key)))


def setup_env_line(key: str, value: Any) -> str:
    """KEY=value that envfile, `set -a; . .env` and launchd all read the same."""
    text = str(value)
    if _SETUP_BARE_VALUE.fullmatch(text):
        return f"{key}={text}"
    if "'" not in text and "\n" not in text:
        return f"{key}='{text}'"
    escaped = (text.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")
               .replace("\n", "\\n"))
    return f'{key}="{escaped}"'


def setup_generate_secret(nbytes: int = 18) -> str:
    """URL-safe random text: no character needs quoting in .env or a shell."""
    return secrets.token_urlsafe(nbytes)


def build_env_values(rec: dict, *, multi_user: bool = False, cluster: str = "",
                     host: str = "", admin_user: str = "admin",
                     old: dict | None = None) -> tuple[dict[str, str], dict[str, str]]:
    """(values init manages, secrets it generated just now).

    Secrets already in the old .env are reused, never rotated: an admin that
    exists keeps its password, and paired nodes keep their NODE_TOKEN. Login is
    never switched off by a re-run (old AUTH_ENABLED=1 keeps multi-user).
    """
    old = dict(old or {})
    generated: dict[str, str] = {}
    values: dict[str, str] = {
        "MODEL_ID": str(rec["model_id"]),
        "CONTEXT_SIZE": str(int(rec["context_size"])),
        "MAX_CONCURRENT_GENERATIONS": str(int(rec["max_concurrent_generations"])),
        "ADMISSION_MEMORY_AWARE": "1",
        "ADMISSION_MIN_FREE_MB": str(int(rec["admission_min_free_mb"])),
    }
    multi = bool(multi_user or cluster or old.get("AUTH_ENABLED", "").strip() == "1")
    # A bind beyond loopback is only kept while login stays on: the app refuses
    # WEB_HOST=0.0.0.0 with AUTH_ENABLED=0.
    values["WEB_HOST"] = host or ("0.0.0.0" if (multi_user or cluster) else
                                  (old.get("WEB_HOST") if multi else "") or "127.0.0.1")
    values["AUTH_ENABLED"] = "1" if multi else "0"
    if multi:
        values["AUTH_ADMIN_USERNAME"] = old.get("AUTH_ADMIN_USERNAME") or admin_user or "admin"
        password = old.get("AUTH_ADMIN_PASSWORD", "")
        if not password:
            password = generated["AUTH_ADMIN_PASSWORD"] = setup_generate_secret()
        values["AUTH_ADMIN_PASSWORD"] = password
    if cluster:
        values["NODE_ROLE"] = cluster
        token = old.get("NODE_TOKEN", "")
        if cluster == "primary" and not token:
            token = generated["NODE_TOKEN"] = setup_generate_secret(32)
        if token:
            values["NODE_TOKEN"] = token
    return values, generated


def render_env(values: dict[str, str], rec: dict, *, carried: list[str] | None = None,
               now: float | None = None) -> str:
    """The .env text: a short header, the managed block, then carried-over lines."""
    stamp = time.strftime("%Y-%m-%d", time.localtime(now if now is not None else time.time()))
    size = rec.get("approx_size_gb")
    lines = [
        f"# Written by `deploy.py init` on {stamp} for this Mac "
        f"({rec.get('ram_gb')}GB RAM, {platform.machine() or 'unknown'}).",
        f"# Model: {rec.get('reason')}" + (f", ~{size}GB of weights." if size else "."),
        "# Variables already set in the environment win over this file.",
        "# Every other setting keeps its default; .env.example lists them all.",
        "# This file holds secrets: keep it private (mode 600) and never commit it.",
        "",
    ]
    lines += [setup_env_line(key, value) for key, value in values.items()]
    if carried:
        lines += ["", "# Kept from the previous .env (settings init does not manage):"]
        lines += carried
    return "\n".join(lines) + "\n"


def setup_carry_over(old_text: str, managed: set[str]) -> list[str]:
    """The previous .env's KEY=VALUE lines for keys init does not write."""
    kept = []
    for line in str(old_text or "").splitlines():
        values, _problems = parse_dotenv(line)
        if values and not (set(values) & managed):
            kept.append(line.rstrip())
    return kept


def setup_env_diff(old: dict[str, str], new: dict[str, str]) -> list[str]:
    """Human summary of what a new .env changes; secrets are never printed."""
    def shown(key: str, value: str) -> str:
        return "(secret)" if setup_is_secret(key) and value else repr(value)
    out = []
    for key, value in new.items():
        if key not in old:
            out.append(f"  + {key}={shown(key, value)}")
        elif old[key] != value:
            out.append(f"  ~ {key}: {shown(key, old[key])} -> {shown(key, value)}")
    same = sum(1 for k, v in new.items() if old.get(k) == v)
    out.append(f"  ({same} unchanged)")
    return out


def setup_write_private(path: Path, text: str) -> None:
    """Write text readable by its owner only (0600), even when the file existed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(path, 0o600)


# --------------------------------------------------------------------------- #
# deploy.py init
# --------------------------------------------------------------------------- #

def _setup_init_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deploy.py init",
        description="Write .env for this Mac: a model that fits its RAM, plus login and "
                    "cluster settings. Also creates ./.venv and installs dependencies.")
    p.add_argument("--non-interactive", action="store_true",
                   help="Ask nothing; take the recommendation and the flags as given.")
    p.add_argument("--multi-user", action="store_true",
                   help="Require login (AUTH_ENABLED=1), bind all interfaces and generate "
                        "an admin password.")
    p.add_argument("--cluster", choices=["primary", "secondary"], default="",
                   help="primary: also generate NODE_TOKEN. secondary: pair with "
                        "--primary-url using --code (implies --multi-user).")
    p.add_argument("--primary-url", default="", help="The primary's URL (secondary only).")
    p.add_argument("--code", default="", help="Pairing code from the primary's Settings > "
                                              "Cluster (secondary only).")
    p.add_argument("--model", default="", help="Use this model (id or alias) instead of "
                                               "the recommendation.")
    p.add_argument("--host", default="", help="WEB_HOST to write (default 127.0.0.1, or "
                                              "0.0.0.0 with --multi-user/--cluster).")
    p.add_argument("--admin-user", default="admin", help="AUTH_ADMIN_USERNAME for a new admin.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing .env (the old one is kept as .env.bak).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the file init would write (secrets masked); change nothing.")
    p.add_argument("--skip-deps", action="store_true",
                   help="Do not create ./.venv or install dependencies.")
    p.add_argument("--deps-only", action="store_true",
                   help="Only create ./.venv and install dependencies; leave .env alone.")
    return p


def _setup_ask(question: str, default: str, ask: Callable[[str], str]) -> str:
    try:
        answer = ask(f"{question} [{default}]: ").strip()
    except EOFError:
        answer = ""
    return answer or default


def _setup_yes(question: str, ask: Callable[[str], str]) -> bool:
    return _setup_ask(question + " (y/N)", "n", ask).lower() in ("y", "yes")


def _setup_join(primary_url: str, code: str, env_file: Path) -> int:
    """Pair with a primary through `deploy.py cluster join --env-file`."""
    return int(cluster_cli(["join", primary_url, "--code", code,
                            "--env-file", str(env_file)]) or 0)


def run_init(args: argparse.Namespace, *, root: Path | None = None,
             ram_gb: float | None = None, ask: Callable[[str], str] | None = None,
             join: Callable[[str, str, Path], int] | None = None,
             out: Any = None, now: float | None = None) -> int:
    """Everything init does after the dependencies exist. Returns an exit code."""
    root = Path(root) if root is not None else ROOT
    out = out or sys.stdout
    join = join or _setup_join
    say = lambda text="": print(text, file=out)  # noqa: E731
    override = os.environ.get(ENV_FILE_VAR, "").strip() if root == ROOT else ""
    env_path = Path(override).expanduser() if override else root / ".env"
    interactive = (not args.non_interactive and ask is not None)

    old_text = ""
    if env_path.exists():
        old_text = env_path.read_text(encoding="utf-8")
    old_values, _problems = parse_dotenv(old_text)

    cluster = str(args.cluster or "")
    primary_url, code = str(args.primary_url or "").strip(), str(args.code or "").strip()
    multi_user = bool(args.multi_user)
    rec = recommend_starter(ram_gb, root / "data")
    model = str(args.model or "").strip()

    say(f"This Mac: {rec['ram_gb']:g}GB RAM, keeping {rec['headroom_gb']:g}GB of headroom.")
    say(f"Recommended model: {rec['model_id']} (context {rec['context_size']}; {rec['reason']}).")
    if interactive:
        model = _setup_ask("Model to use (Enter keeps the recommendation)", model or
                           rec["model_id"], ask)
        if not cluster and not multi_user and old_values.get("AUTH_ENABLED") != "1":
            multi_user = _setup_yes("Will other people sign in to this server?", ask)
        if cluster == "secondary" and not primary_url:
            primary_url = _setup_ask("Primary URL (e.g. http://mac-mini.local:8000)", "", ask)
        if cluster == "secondary" and not code:
            code = _setup_ask("Pairing code (primary: Settings > Cluster > Pair a node)", "",
                              ask)
    if cluster == "secondary" and not (primary_url and code):
        say("--cluster secondary needs --primary-url and --code (a pairing code from the "
            "primary's Settings > Cluster, or POST /api/cluster/pairing).")
        return 2
    if model and model != rec["model_id"]:
        resolved = resolve_model(model)
        fits = setup_model_fits(resolved, rec["ram_gb"], guess=True)
        if fits is False:
            say(f"warning: {resolved} does not fit {rec['ram_gb']:g}GB with headroom; "
                "it may swap or fail to load.")
        rec = {**rec, "model_id": resolved, "source": "chosen",
               "reason": "chosen with --model" if not interactive else "chosen at the prompt",
               "approx_size_gb": (model_metadata(resolved, guess=False) or {})
               .get("approx_size_gb")}

    values, generated = build_env_values(rec, multi_user=multi_user, cluster=cluster,
                                         host=str(args.host or "").strip(),
                                         admin_user=args.admin_user, old=old_values)
    carried = setup_carry_over(old_text, set(values))
    text = render_env(values, rec, carried=carried, now=now)

    if args.dry_run:
        say(f"\n--- {env_path} (dry run: nothing written) ---")
        for line in text.splitlines():
            key = line.split("=", 1)[0]
            say(f"{key}=(secret)" if "=" in line and not line.startswith("#")
                and setup_is_secret(key) else line)
        return 0

    target = env_path
    if env_path.exists() and not args.force:
        target = env_path.with_name(env_path.name + ".new")
        say(f"\n{env_path} exists, so it was left alone. The recommended file is {target}:")
        for row in setup_env_diff(old_values, parse_dotenv(text)[0]):
            say(row)
        say(f"Review it, then: mv {target} {env_path}   (or re-run init with --force)")
    elif env_path.exists():
        backup = env_path.with_name(env_path.name + ".bak")
        setup_write_private(backup, old_text)
        say(f"\nPrevious {env_path.name} saved as {backup}.")
    setup_write_private(target, text)
    say(f"Wrote {target} (mode 600).")

    if cluster == "secondary":
        say(f"Pairing with {primary_url} ...")
        code_ = join(primary_url, code, target)
        if code_:
            say(f"Pairing failed; {target} has no node token yet. Fix the URL or get a fresh "
                f"code, then: deploy.py cluster join {primary_url} --code CODE "
                f"--env-file {target}")
            return int(code_)

    if generated.get("AUTH_ADMIN_PASSWORD"):
        say("\nAdmin sign-in (shown once; also stored in .env as AUTH_ADMIN_PASSWORD):")
        say(f"  user     : {values.get('AUTH_ADMIN_USERNAME')}")
        say(f"  password : {generated['AUTH_ADMIN_PASSWORD']}")
        db_path = root / "data" / DB_PATH.name
        if db_path.exists():
            say(f"  note: {db_path} already exists. If it has an admin, that admin keeps "
                "its password (a re-run never resets it); change it under Settings > Users.")
    if generated.get("NODE_TOKEN"):
        say("\nNODE_TOKEN generated for this primary. Secondaries pair with a one-time code "
            "instead: Settings > Cluster > Pair a node, then on each secondary:")
        say("  ./install.sh --join http://THIS-MAC.local:8000 --code CODE")
    if values.get("WEB_HOST") not in ("127.0.0.1", "localhost", "::1"):
        say(f"\nWEB_HOST={values['WEB_HOST']}: reachable from the network with login on. "
            "Cookies travel over plain HTTP until TLS is in front (deploy.py remote "
            "tailscale, or a reverse proxy) and AUTH_COOKIE_SECURE=1.")
    say("\nNext: python3 deploy.py   (or `deploy.py service install` to start it at login)")
    return 0


def init_cli(argv: list[str], **hooks: Any) -> int:
    """`deploy.py init ...`: see the module docstring."""
    args = _setup_init_parser().parse_args(argv)
    if not args.skip_deps and not args.dry_run:
        # Creates ./.venv and installs dependencies when needed, then re-execs
        # this same command inside the venv; a no-op once that is done.
        bootstrap()
    if args.deps_only:
        print(f"Dependencies ready in {sys.prefix}.")
        return 0
    if "ask" not in hooks and not args.non_interactive and sys.stdin.isatty():
        hooks["ask"] = input
    return run_init(args, **hooks)


# --------------------------------------------------------------------------- #
# deploy.py service install|uninstall|status (launchd)
# --------------------------------------------------------------------------- #

def service_plist_path(label: str = SERVICE_LABEL, agents_dir: Path | None = None) -> Path:
    base = Path(agents_dir) if agents_dir is not None else Path.home() / "Library" / "LaunchAgents"
    return base / f"{label}.plist"


def render_service_plist(root: Path, python: Path, *, label: str = SERVICE_LABEL,
                         args: list[str] | tuple = (), env_file: str = "") -> bytes:
    """The LaunchAgent: the venv python on deploy.py, restarted after a crash.

    KeepAlive {SuccessfulExit: false} restarts the server when it exits with an
    error and leaves it stopped after a clean exit (Ctrl-C, `service
    uninstall`). Secrets stay in .env, which deploy.py reads itself; nothing
    secret is copied into the plist.
    """
    root = Path(root)
    env = {"PATH": _SETUP_SERVICE_PATH, "PYTHONUNBUFFERED": "1"}
    if env_file:
        env[ENV_FILE_VAR] = str(env_file)
    doc = {
        "Label": label,
        "ProgramArguments": [str(python), str(root / "deploy.py"), *[str(a) for a in args]],
        "WorkingDirectory": str(root),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 10,
        "ProcessType": "Standard",
        "StandardOutPath": str(root / "logs" / "service.out.log"),
        "StandardErrorPath": str(root / "logs" / "service.err.log"),
        "EnvironmentVariables": env,
    }
    return plistlib.dumps(doc)


def _setup_launchctl(runner: Callable, *args: str) -> subprocess.CompletedProcess:
    return runner(["launchctl", *args], capture_output=True, text=True, timeout=30)


def service_status(*, label: str = SERVICE_LABEL, agents_dir: Path | None = None,
                   runner: Callable = subprocess.run) -> dict:
    """Read-only: is the plist there, is the job loaded, its pid and last exit."""
    plist = service_plist_path(label, agents_dir)
    info = {"label": label, "plist": str(plist), "installed": plist.exists(),
            "loaded": False, "state": "", "pid": None, "last_exit": None}
    try:
        done = _setup_launchctl(runner, "print", f"gui/{os.getuid()}/{label}")
    except (OSError, subprocess.SubprocessError):
        return info
    if done.returncode != 0:
        return info
    info["loaded"] = True
    for line in (done.stdout or "").splitlines():
        key, _, value = line.strip().partition(" = ")
        if key == "state":
            info["state"] = value.strip()
        elif key == "pid" and value.strip().isdigit():
            info["pid"] = int(value.strip())
        elif key == "last exit code":
            info["last_exit"] = value.strip()
    return info


def service_cli(argv: list[str], *, root: Path | None = None, agents_dir: Path | None = None,
                runner: Callable = subprocess.run, out: Any = None) -> int:
    """`deploy.py service install|uninstall|status [--dry-run] [-- deploy.py flags]`."""
    parser = argparse.ArgumentParser(
        prog="deploy.py service",
        description="Run the app as a launchd LaunchAgent (starts at login, restarts "
                    "after a crash, logs to logs/).")
    parser.add_argument("action", choices=["install", "uninstall", "status"])
    parser.add_argument("--label", default=SERVICE_LABEL)
    parser.add_argument("--python", default="", help="Interpreter (default ./.venv/bin/python).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plist and the launchctl commands; change nothing.")
    parser.epilog = "Flags after -- are passed to deploy.py, e.g. service install -- --agent"
    argv = list(argv)
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    args = parser.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    out = out or sys.stdout
    say = lambda text="": print(text, file=out)  # noqa: E731
    root = Path(root) if root is not None else ROOT
    plist = service_plist_path(args.label, agents_dir)
    domain = f"gui/{os.getuid()}"

    if args.action == "status":
        info = service_status(label=args.label, agents_dir=agents_dir, runner=runner)
        say(f"plist    : {info['plist']} ({'present' if info['installed'] else 'absent'})")
        if not info["loaded"]:
            say("launchd  : not loaded")
            return 0 if not info["installed"] else 1
        say(f"launchd  : loaded, state {info['state'] or '?'}"
            + (f", pid {info['pid']}" if info["pid"] else "")
            + (f", last exit {info['last_exit']}" if info["last_exit"] else ""))
        say(f"logs     : {root / 'logs' / 'service.err.log'}")
        return 0

    if args.action == "uninstall":
        if args.dry_run:
            say(f"+ launchctl bootout {domain}/{args.label}\n+ rm {plist}")
            return 0
        _setup_launchctl(runner, "bootout", f"{domain}/{args.label}")
        if plist.exists():
            plist.unlink()
            say(f"Removed {plist}; the service is stopped.")
        else:
            say(f"{plist} was not installed.")
        return 0

    python = Path(args.python).expanduser() if args.python else root / ".venv" / "bin" / "python"
    if not python.exists() and not args.dry_run:
        say(f"{python} does not exist. Run `python3 deploy.py init` first (it creates the "
            "venv), or pass --python.")
        return 1
    env_file = os.environ.get(ENV_FILE_VAR, "").strip() if root == ROOT else ""
    data = render_service_plist(root, python, label=args.label, args=extra, env_file=env_file)
    if args.dry_run:
        say(f"--- {plist} ---\n{data.decode()}")
        say(f"+ launchctl bootout {domain}/{args.label}\n+ launchctl bootstrap {domain} {plist}")
        return 0
    (root / "logs").mkdir(parents=True, exist_ok=True)
    plist.parent.mkdir(parents=True, exist_ok=True)
    # Reinstalling replaces a loaded job: bootout first (a miss is fine).
    _setup_launchctl(runner, "bootout", f"{domain}/{args.label}")
    plist.write_bytes(data)
    done = _setup_launchctl(runner, "bootstrap", domain, str(plist))
    if done.returncode != 0:
        say(f"launchctl bootstrap failed ({done.returncode}): "
            f"{(done.stderr or done.stdout or '').strip()[:300]}")
        return 1
    say(f"Installed {plist}: the app starts now and at every login, and restarts after a "
        f"crash. Logs: {root / 'logs'}/service.*.log. Status: deploy.py service status")
    return 0


# --------------------------------------------------------------------------- #
# --doctor: install checks
# --------------------------------------------------------------------------- #

def _setup_translated() -> bool:
    """True when this process runs under Rosetta (x86_64 on Apple Silicon)."""
    try:
        done = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"],
                              capture_output=True, text=True, timeout=2)
        return done.stdout.strip() == "1"
    except (OSError, subprocess.SubprocessError):
        return False


def _setup_venv_probe(python: Path) -> tuple[str, list[str] | None]:
    """(machine, missing modules) for the venv's interpreter; None when it will not run."""
    code = ("import importlib.util as u, platform; print(platform.machine()); "
            f"print(','.join(n for n in {list(REQUIRED_MODULES)!r} if u.find_spec(n) is None))")
    try:
        done = subprocess.run([str(python), "-c", code], capture_output=True, text=True,
                              timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "", None
    if done.returncode != 0:
        return "", None
    lines = (done.stdout or "").splitlines() + ["", ""]
    return lines[0].strip(), [m for m in lines[1].strip().split(",") if m]


def _setup_free_gb(path: Path) -> float | None:
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free / (1024 ** 3)
    except OSError:
        return None


def gather_install_facts(web_port: int = 8000, model_port: int = 8080, *,
                         root: Path | None = None) -> dict:
    """Everything the install checks judge, read from this machine."""
    root = Path(root) if root is not None else ROOT
    venv_python = root / ".venv" / "bin" / "python"
    venv_machine, venv_missing = (_setup_venv_probe(venv_python) if venv_python.exists()
                                  else ("", None))
    try:
        model = resolve_model(os.environ.get("MODEL_ID") or DEFAULT_MODEL)
    except Exception:  # noqa: BLE001
        model = os.environ.get("MODEL_ID") or DEFAULT_MODEL
    meta = model_metadata(model, guess=True) or {}
    env_path = Path(os.environ.get(ENV_FILE_VAR, "").strip() or root / ".env").expanduser()
    try:
        env_text = env_path.read_text(encoding="utf-8")
        env_mode = env_path.stat().st_mode & 0o777
    except OSError:
        env_text, env_mode = None, None
    cached = model_is_cached(model)
    return {
        "machine": platform.machine(), "translated": _setup_translated(),
        "python": tuple(sys.version_info[:3]), "in_venv": in_venv(),
        "venv_python": str(venv_python), "venv_exists": venv_python.exists(),
        "venv_machine": venv_machine, "venv_missing": venv_missing,
        "model": model, "model_cached": cached,
        "model_size_gb": meta.get("approx_size_gb"), "model_min_ram_gb": meta.get("min_ram_gb"),
        "ram_gb": float(TOTAL_RAM_GB),
        "disk_free_gb": _setup_free_gb(hf_cache_dir()), "disk_path": str(hf_cache_dir()),
        "root_free_gb": _setup_free_gb(root),
        "env_path": str(env_path), "env_text": env_text, "env_mode": env_mode,
        "env_loaded": os.environ.get(DOTENV_LOADED_VAR, ""),
        "environ": {k: v for k, v in os.environ.items()},
        "ports": {"web": (int(web_port), port_open(int(web_port))),
                  "model": (int(model_port), port_open(int(model_port)))},
        "service_installed": service_plist_path().exists(),
    }


def evaluate_install_facts(facts: dict) -> list[tuple[str, str, str]]:
    """[(level, topic, message)] with level ok | warn | fail | info. Pure."""
    rows: list[tuple[str, str, str]] = []
    add = lambda level, topic, msg: rows.append((level, topic, msg))  # noqa: E731

    machine = str(facts.get("machine") or "")
    version = tuple(facts.get("python") or (0, 0, 0))
    vtext = ".".join(str(x) for x in version)
    if machine != "arm64":
        why = "Rosetta (x86_64)" if facts.get("translated") or machine == "x86_64" else machine
        add("fail", "python", f"{vtext} runs as {why}: MLX needs native arm64 Python "
                              "(brew install python@3.12 from a native terminal)")
    elif version < (3, 10):
        add("fail", "python", f"{vtext} is too old: 3.10 or newer (brew install python@3.12)")
    else:
        add("ok", "python", f"{vtext} arm64 (native)")

    if not facts.get("venv_exists"):
        add("warn", "venv", f"{facts.get('venv_python')} missing: run `python3 deploy.py init` "
                            "(or the first launch) to create it")
    elif facts.get("venv_missing") is None:
        add("fail", "venv", f"{facts.get('venv_python')} does not run: delete .venv and run "
                            "`python3 deploy.py init`")
    elif facts.get("venv_machine") and facts.get("venv_machine") != "arm64":
        add("fail", "venv", f"built for {facts['venv_machine']}: delete .venv and re-create it "
                            "with a native arm64 python3")
    elif facts.get("venv_missing"):
        add("warn", "venv", "missing " + ", ".join(facts["venv_missing"])
            + ": the next launch installs them (or `python3 deploy.py init --deps-only`)")
    else:
        add("ok", "venv", "all required packages present")

    model = str(facts.get("model") or "")
    size = facts.get("model_size_gb")
    if facts.get("model_cached"):
        add("ok", "model", f"{model} is downloaded")
    else:
        add("warn", "model", f"{model} is not downloaded yet: the first start fetches "
            + (f"~{float(size):.1f}GB" if size else "it") + " from Hugging Face")

    need = (0.0 if facts.get("model_cached") else float(size or 0)) + 5.0
    free = facts.get("disk_free_gb")
    if free is None:
        add("info", "disk", "free space unknown")
    elif free < 2.0:
        add("fail", "disk", f"{free:.1f}GB free at {facts.get('disk_path')}: not enough for "
                            "the venv and logs")
    elif free < need:
        add("warn", "disk", f"{free:.1f}GB free at {facts.get('disk_path')}, want {need:.0f}GB "
                            "(model download + venv + adapters)")
    else:
        add("ok", "disk", f"{free:.0f}GB free")

    ram = float(facts.get("ram_gb") or 0)
    floor = facts.get("model_min_ram_gb")
    if not size or not floor:
        add("info", "ram", f"{ram:.0f}GB; {model} is not in the catalogue, so its size is unknown")
    elif ram < float(floor):
        add("fail", "ram", f"{model} (~{float(size):.1f}GB) needs a {floor}GB Mac; this one has "
                           f"{ram:.0f}GB. Run `deploy.py init` for one that fits")
    elif float(size) + 1.0 + setup_headroom_gb(ram) > ram:
        add("warn", "ram", f"{model} fits {ram:.0f}GB with little headroom for the KV cache "
                           "and training")
    else:
        add("ok", "ram", f"{ram:.0f}GB holds {model} (~{float(size):.1f}GB) with headroom")

    rows += _setup_env_rows(facts)

    for name, (port, listening) in (facts.get("ports") or {}).items():
        if listening:
            add("info", "ports", f"{name} port {port} is in use (this app, or another program: "
                                 "then the app moves to the next free port)")
        else:
            add("ok", "ports", f"{name} port {port} is free")
    if facts.get("service_installed"):
        add("info", "service", "launchd service installed (deploy.py service status)")
    return rows


def _setup_env_rows(facts: dict) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    text = facts.get("env_text")
    path = facts.get("env_path")
    if text is None:
        return [("info", ".env", f"no {path}: every setting is at its default "
                                 "(`deploy.py init` writes one)")]
    values, problems = parse_dotenv(text)
    rows += [("warn", ".env", p) for p in problems]
    mode = facts.get("env_mode")
    has_secret = any(setup_is_secret(k) and v for k, v in values.items())
    if mode is not None and mode & 0o077 and has_secret:
        rows.append(("warn", ".env", f"holds secrets but is readable by others "
                                     f"(mode {mode:o}): chmod 600 {path}"))
    for key, value in values.items():
        word = value.strip().lower()
        if word in _SETUP_TRUE_WORDS:
            rows.append(("warn", ".env", f"{key}={value} is read as OFF: switches here are 1/0, "
                                         f"so write {key}=1"))
        elif word in _SETUP_FALSE_WORDS:
            rows.append(("info", ".env", f"{key}={value} works as off, but switches here are "
                                         f"1/0: write {key}=0"))
    environ = facts.get("environ") or {}
    loaded = facts.get("env_loaded") or ""
    shadowed = sorted(k for k, v in values.items() if k in environ and environ[k] != v)
    if shadowed:
        rows.append(("info", ".env", "set in the environment, which wins over .env: "
                                     + ", ".join(shadowed[:8])
                                     + (" ..." if len(shadowed) > 8 else "")))
    if not any(level == "warn" for level, _t, _m in rows):
        rows.append(("ok", ".env", f"{len(values)} settings parse cleanly"
                                   + (" (loaded)" if loaded else "")))
    return rows


def print_install_checks(web_port: int = 8000, model_port: int = 8080, *,
                         facts: dict | None = None, out: Any = None) -> int:
    """Print the install checks; returns how many failed."""
    out = out or sys.stdout
    rows = evaluate_install_facts(facts if facts is not None
                                  else gather_install_facts(web_port, model_port))
    print("\ninstall checks:", file=out)
    tags = {"ok": "[ ok ]", "warn": "[warn]", "fail": "[FAIL]", "info": "[info]"}
    for level, topic, message in rows:
        print(f"  {tags.get(level, level)} {topic:8}: {message}", file=out)
    return sum(1 for level, _t, _m in rows if level == "fail")


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'SERVICE_LABEL',
    '_SETUP_LADDER',
    '_setup_ask',
    '_setup_env_rows',
    '_setup_free_gb',
    '_setup_init_parser',
    '_setup_join',
    '_setup_launchctl',
    '_setup_translated',
    '_setup_venv_probe',
    '_setup_yes',
    'build_env_values',
    'evaluate_install_facts',
    'gather_install_facts',
    'init_cli',
    'print_install_checks',
    'recommend_starter',
    'render_env',
    'render_service_plist',
    'run_init',
    'service_cli',
    'service_plist_path',
    'service_status',
    'setup_carry_over',
    'setup_env_diff',
    'setup_env_line',
    'setup_generate_secret',
    'setup_headroom_gb',
    'setup_is_secret',
    'setup_model_fits',
    'setup_write_private',
]
