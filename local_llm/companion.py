"""The native macOS companion: one status route for it, and the helpers that build it.

A menu-bar app (companion/macos/LocalLLMBar.swift) or a SwiftBar/xbar plugin
(companion/swiftbar/local-llm.30s.py) wants one cheap answer every few seconds:
is the model up, how deep is the queue, is a node down, is something waiting
for my approval. Stitching that together from /api/health, /api/backends,
/api/cluster/nodes, /api/dashboards/summary and /api/approvals/pending would be
five requests per poll, three of them ADMIN routes a narrow token cannot reach,
and one (the dashboard summary) far too heavy to run every five seconds. So the
server answers it in one place:

  GET /api/status/summary   session users, auth-off callers, and API tokens
                            holding the ``status`` scope (control and admin
                            imply it). A chat-only token is refused.

What the caller sees follows /api/health's rule: everyone gets the service
basics (model state, backend, queue counts, their own pending approvals and
their own last-hour request figures); only administrators get machine-wide
telemetry (cluster nodes, running tasks, retrain progress, every user's
last-hour figures). Nothing in it is a secret: no token, no node URL, no path.

The companion's levers are the existing ADMIN routes (POST /api/model/restart,
POST /api/tasks/{id}/cancel, POST /api/retrain); the summary's ``controls``
block says which of them this credential may pull, so the app greys out a menu
item instead of letting the user discover a 403.

The second half of the module is `deploy.py companion build|install|uninstall|
plugin`: compile the Swift app with swiftc into dist/LocalLLM Bar.app, copy it
to ~/Applications (optionally with a login LaunchAgent), remove both again, and
install the SwiftBar plugin. Building loads no model and starts nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import platform
import plistlib
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as _companion_xml_escape

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .api_tokens import *  # noqa: F401,F403
from .policy import *  # noqa: F401,F403
from .dashboards import *  # noqa: F401,F403

# Route annotations are strings under `from __future__ import annotations` and
# FastAPI resolves them against MODULE globals, so Request must live here (an
# import inside register_companion would turn `request` into a query param).
try:
    from fastapi import Request  # noqa: F401
except Exception:  # pragma: no cover  (selftest on a machine without FastAPI)
    Request = Any  # type: ignore

COMPANION_SUMMARY_PATH = "/api/status/summary"
COMPANION_APP_NAME = "LocalLLM Bar"
COMPANION_EXECUTABLE = "LocalLLMBar"
COMPANION_BUNDLE_ID = "local.llm.bar"
COMPANION_VERSION = "1.0.0"
COMPANION_MIN_MACOS = "13.0"
COMPANION_PLUGIN_NAME = "local-llm.30s.py"
COMPANION_LAUNCH_AGENT = f"{COMPANION_BUNDLE_ID}.plist"
# The companion polls every 5 s; the last-hour figures come from a scan of the
# metrics table, so they are computed at most this often per audience.
_COMPANION_METRICS_TTL_S = 10.0
_COMPANION_METRICS_CACHE: dict = {}
_COMPANION_LOCK = threading.Lock()
_COMPANION_MAX_TASK_IDS = 20
_COMPANION_MAX_NODES = 16
_companion_log = get_logger("companion")


class CompanionError(RuntimeError):
    """A build/install step that cannot proceed, with a message for the user."""


# --------------------------------------------------------------------------- #
# Summary pieces (pure, so tests and the SwiftBar plugin agree on them)       #
# --------------------------------------------------------------------------- #
def companion_indicator(model_status: Any, healthy: Any, queue: dict | None) -> dict:
    """The menu-bar dot: {state, label, color} from the model state and queue.

    Computed on the server so the Swift app and the SwiftBar plugin cannot
    disagree about what "ready" means. Clients only add the states the server
    cannot report about itself (offline, sign in).
    """
    status = str(model_status or "unknown")
    try:
        waiting = int((queue or {}).get("waiting") or 0)
    except (TypeError, ValueError):
        waiting = 0
    if status.startswith("error"):
        return {"state": "error", "label": "error", "color": "red"}
    if status in ("starting", "loading", "restarting"):
        return {"state": "loading", "label": "loading", "color": "yellow"}
    if status == "stopped":
        return {"state": "stopped", "label": "stopped", "color": "gray"}
    if status == "ready":
        if waiting > 0:
            return {"state": "busy", "label": f"queue {waiting}", "color": "orange"}
        if healthy is False:
            # Ready by the manager's account but the last probe failed: worth a
            # glance, not an alarm (one slow probe flips it until the next).
            return {"state": "degraded", "label": "unhealthy", "color": "yellow"}
        return {"state": "ready", "label": "ready", "color": "green"}
    return {"state": "unknown", "label": status[:16], "color": "gray"}


def companion_can(user: dict, method: str, path: str) -> bool:
    """Whether this caller may pull an ADMIN lever (role AND token scope)."""
    if (user or {}).get("role") != "admin":
        return False
    scopes = user.get("token_scopes")
    return scopes is None or token_allows_route(scopes, method, path)


def companion_hour_metrics(db: Any, user_id: str | None = None, *,
                           now: float | None = None) -> dict | None:
    """Last-hour request count, TTFT p50 and decode tok/s (p50), cached briefly.

    `user_id` narrows the figures to one caller (what a non-admin sees); None
    means every user (admins). Reuses the dashboards row reader and percentile
    so the numbers match the Dashboards panel. None when metrics are unreadable.
    """
    key = (id(db), user_id or "*")
    mono = time.monotonic()
    if now is None:
        with _COMPANION_LOCK:
            hit = _COMPANION_METRICS_CACHE.get(key)
            if hit is not None and hit[0] > mono:
                return dict(hit[1])
    try:
        db.ensure_schema(DASH_SCHEMA_NAME)
        until = int(now if now is not None else time.time())
        rows = _dash_metric_rows(db, until - 3600, until)
    except Exception as exc:  # noqa: BLE001
        log_event(_companion_log, 10, "companion.metrics_unavailable",
                  error=redact_text(f"{type(exc).__name__}: {exc}"))
        return None
    if user_id:
        rows = [r for r in rows if r.get("user_id") == user_id]
    requests = [r for r in rows if _dash_is_request(r)]
    out = {
        "window": "1h",
        "scope": "own" if user_id else "all",
        "requests": len(requests),
        "errors": sum(1 for r in requests if (r.get("status_code") or 0) >= 400),
        "ttft_p50_ms": _dash_pct([r.get("ttft_ms") for r in rows], 0.5),
        "decode_tps_p50": _dash_pct([r.get("decode_tps") for r in rows], 0.5),
    }
    if now is None:
        with _COMPANION_LOCK:
            _COMPANION_METRICS_CACHE[key] = (mono + _COMPANION_METRICS_TTL_S, dict(out))
    return out


def companion_reset_cache() -> None:
    """Forget cached last-hour figures (tests, or after a metrics purge)."""
    with _COMPANION_LOCK:
        _COMPANION_METRICS_CACHE.clear()


def _companion_nodes(node_registry: Any) -> list[dict] | None:
    """Cluster nodes as name/state/role only: a node's URL and token stay put."""
    if node_registry is None:
        return None
    try:
        snapshot = node_registry.snapshot()
    except Exception:  # noqa: BLE001
        return None
    out = []
    for node in list(snapshot or [])[:_COMPANION_MAX_NODES]:
        out.append({
            "name": str(node.get("name") or "")[:64],
            "state": str(node.get("state") or "unknown")[:32],
            "role": str(node.get("role") or "")[:32],
            "active": bool(node.get("active", True)),
            "local": bool(node.get("is_local", False)),
        })
    return out


def _companion_queue(gate: Any) -> dict:
    if gate is None:
        return {"running": None, "waiting": None, "limit": None}
    try:
        # probe=False: reuse the last memory reading; /api/health (polled by
        # every open tab) keeps it fresh, and vm_stat every 5 s is not free.
        snap = gate.snapshot(probe=False)
    except TypeError:
        snap = gate.snapshot()
    except Exception:  # noqa: BLE001
        snap = {}
    return {k: (snap or {}).get(k) for k in ("running", "waiting", "limit")}


def _companion_model(config: Any, manager: Any, healthy: Any) -> dict:
    status = str(getattr(manager, "status", "unknown") or "unknown")
    if status.startswith("error"):
        # The manager stores the exception text; it can carry a path or a URL.
        status = redact_text(status)[:200]
    try:
        active = active_adapter(manager) if manager is not None else {}
    except Exception:  # noqa: BLE001
        active = {}
    choice = active.get("choice")
    return {
        "status": status,
        "healthy": healthy,
        "id": getattr(manager, "model_id", None) or getattr(config, "model", None),
        "alias": (getattr(config, "model_alias", "") or None),
        "adapter": None if choice in (None, "none") else choice,
        "backend": active.get("backend") or getattr(config, "model_backend", None),
        "capabilities": list(active.get("caps") or []),
    }


def _companion_tasks(tasks: Any) -> dict:
    try:
        ids = [str(t) for t in list(getattr(tasks, "by_task", {}) or {})]
    except Exception:  # noqa: BLE001
        ids = []
    return {"running": len(ids), "running_ids": ids[:_COMPANION_MAX_TASK_IDS]}


async def companion_summary(ctx: Any, user: dict, manager: Any = None) -> dict:
    """Everything GET /api/status/summary returns for `user` (see module doc)."""
    config = ctx.config
    manager = manager if manager is not None else ctx.model_manager
    is_admin = user.get("role") == "admin"
    probe = getattr(manager, "health_probe_cached", None)
    healthy: Any = None
    if probe is not None:
        try:
            healthy = bool(await probe())
        except Exception:  # noqa: BLE001
            healthy = False
    queue = await asyncio.to_thread(_companion_queue, ctx.generation_gate)
    model = _companion_model(config, manager, healthy)
    try:
        pending = len(BROKER.pending(user["id"]))
    except Exception:  # noqa: BLE001
        pending = 0
    metrics = await asyncio.to_thread(
        companion_hour_metrics, ctx.db, None if is_admin else user["id"])
    scopes = user.get("token_scopes")
    out: dict = {
        "ui_build": UI_BUILD,
        "server_time": int(time.time()),
        "indicator": companion_indicator(model["status"], healthy, queue),
        "model": model,
        "queue": queue,
        "approvals": {"pending": pending},
        "metrics": metrics,
        "user": {"username": user.get("username"), "role": user.get("role"),
                 "scopes": list(scopes) if scopes is not None else None},
        "controls": {
            "restart_model": companion_can(user, "POST", "/api/model/restart"),
            "cancel_tasks": companion_can(user, "POST", "/api/tasks/x/cancel"),
            "retrain": companion_can(user, "POST", "/api/retrain"),
        },
        "api_base_path": "/v1",
        # Machine-wide telemetry is admin-only, exactly as on /api/health.
        "nodes": None,
        "multi_node": None,
        "tasks": None,
        "retrain": None,
    }
    if is_admin:
        registry = ctx.node_registry
        out["nodes"] = _companion_nodes(registry)
        out["multi_node"] = bool(getattr(registry, "multi_node", False))
        out["tasks"] = _companion_tasks(ctx.tasks)
        status = dict(getattr(ctx.retrain_manager, "status", None) or {})
        out["retrain"] = {"running": bool(status.get("running")),
                          "message": redact_text(str(status.get("message") or ""))[:200]}
    return out


# --------------------------------------------------------------------------- #
# Route                                                                       #
# --------------------------------------------------------------------------- #
def register_companion(app: Any, ctx: Any) -> None:
    """GET /api/status/summary (session users, or API tokens with the status scope)."""
    from fastapi import Depends, HTTPException

    auth = ctx.auth

    async def companion_caller(request: Request) -> dict:
        # api_tokens._TOK_ROUTES lists this path under the status scope, and
        # the check is repeated here so the route stays exactly as narrow
        # whatever that table says: status (or a scope implying it) for
        # tokens; sessions and auth-off callers pass as they do everywhere.
        # The rest mirrors require_user.
        user = auth.user_for_request(request)
        if not user:
            raise HTTPException(status_code=401, detail="authentication required")
        held = user.get("token_scopes")
        if held is not None and "status" not in effective_scopes(held):
            log_event(get_logger("auth"), 30, "auth.token_scope_denied",
                      route=COMPANION_SUMMARY_PATH, method="GET", user_id=user["id"],
                      token_id=user.get("token_id"), scopes=",".join(held))
            raise HTTPException(
                status_code=403,
                detail=f"this API token's scopes ({', '.join(held)}) do not allow "
                       f"GET {COMPANION_SUMMARY_PATH} (it needs the status scope)")
        bind_context(user_id=user["id"], username=user.get("username"),
                     role=user.get("role"))
        set_acting_user(user["id"])
        request.state.user = user
        return user

    @app.get(COMPANION_SUMMARY_PATH)
    async def companion_status_summary(user: dict = Depends(companion_caller)):
        manager = getattr(app.state, "model_manager", None) or ctx.model_manager
        return await companion_summary(ctx, user, manager)


# --------------------------------------------------------------------------- #
# Build / install helpers (deploy.py companion ...)                           #
# --------------------------------------------------------------------------- #
def companion_source_dir(root: Path | None = None) -> Path:
    """companion/ next to deploy.py (a checkout; the single-file build has none)."""
    return Path(root if root is not None else ROOT) / "companion"


def companion_render_info_plist(template: str, **overrides: Any) -> str:
    """Fill the {{NAME}} placeholders of companion/macos/Info.plist and validate it."""
    values = {
        "BUNDLE_ID": COMPANION_BUNDLE_ID,
        "EXECUTABLE": COMPANION_EXECUTABLE,
        "NAME": COMPANION_APP_NAME,
        "VERSION": COMPANION_VERSION,
        "BUILD": time.strftime("%Y%m%d%H%M", time.gmtime()),
        "MIN_MACOS": COMPANION_MIN_MACOS,
    }
    values.update({str(k).upper(): v for k, v in overrides.items()})
    out = template
    for key, value in values.items():
        out = out.replace("{{" + key + "}}", _companion_xml_escape(str(value)))
    if "{{" in out:
        start = out.index("{{")
        raise CompanionError(f"Info.plist template has an unfilled placeholder near "
                             f"{out[start:start + 24]!r}")
    try:
        plistlib.loads(out.encode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise CompanionError(f"rendered Info.plist is not a valid plist: {exc}") from exc
    return out


def companion_find_swiftc(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit if Path(explicit).exists() else None
    found = shutil.which("swiftc")
    if found:
        return found
    return "/usr/bin/swiftc" if Path("/usr/bin/swiftc").exists() else None


def _companion_arches(arch: str | None) -> list[str]:
    arch = (arch or platform.machine() or "arm64").strip().lower()
    if arch in ("universal", "both"):
        return ["arm64", "x86_64"]
    if arch not in ("arm64", "x86_64"):
        raise CompanionError(f"unknown arch {arch!r} (arm64, x86_64 or universal)")
    return [arch]


def companion_build_command(source: Path, binary: Path, *, swiftc: str = "/usr/bin/swiftc",
                            arch: str = "arm64", optimize: bool = True) -> list[str]:
    """The swiftc invocation for one architecture.

    Swift 5 language mode: the app is plain AppKit on the main actor, and Swift
    6's strict concurrency would turn the toolchain's version into a build
    failure. -parse-as-library because the file declares @main.
    """
    cmd = [str(swiftc)]
    if optimize:
        cmd.append("-O")
    cmd += ["-parse-as-library", "-swift-version", "5",
            "-target", f"{arch}-apple-macos{COMPANION_MIN_MACOS}",
            "-framework", "AppKit", "-framework", "Security",
            "-o", str(binary), str(source)]
    return cmd


def companion_build(out_dir: Path | None = None, *, swiftc: str | None = None,
                    arch: str | None = None, sign: bool = True, dry_run: bool = False,
                    optimize: bool = True, root: Path | None = None,
                    run: Any = subprocess.run) -> dict:
    """Compile LocalLLMBar.swift into <out_dir>/LocalLLM Bar.app.

    Builds into a staging bundle and swaps it in at the end, so a failed
    compile never leaves a half-written app where a working one was. Returns
    {"app", "commands"}; with dry_run nothing is written or run.
    """
    src_dir = companion_source_dir(root) / "macos"
    source = src_dir / "LocalLLMBar.swift"
    template = src_dir / "Info.plist"
    for needed in (source, template):
        if not needed.exists():
            raise CompanionError(f"{needed} is missing: build the companion from a "
                                 f"checkout of the project (the single-file build has no "
                                 f"companion/ sources)")
    compiler = companion_find_swiftc(swiftc)
    if compiler is None and not dry_run:
        raise CompanionError("swiftc not found: install the Xcode command line tools "
                             "(xcode-select --install) or pass --swiftc PATH")
    compiler = compiler or (swiftc or "/usr/bin/swiftc")
    out_dir = Path(out_dir if out_dir is not None else Path(root if root is not None else ROOT) / "dist")
    app = out_dir / f"{COMPANION_APP_NAME}.app"
    staging = out_dir / f".{COMPANION_APP_NAME}.app.build"
    macos_dir = staging / "Contents" / "MacOS"
    binary = macos_dir / COMPANION_EXECUTABLE
    arches = _companion_arches(arch)
    commands: list[list[str]] = []
    if len(arches) == 1:
        commands.append(companion_build_command(source, binary, swiftc=compiler,
                                                arch=arches[0], optimize=optimize))
    else:
        parts = [macos_dir / f"{COMPANION_EXECUTABLE}-{a}" for a in arches]
        commands += [companion_build_command(source, p, swiftc=compiler, arch=a,
                                             optimize=optimize)
                     for a, p in zip(arches, parts)]
        commands.append(["/usr/bin/lipo", "-create", "-output", str(binary), *map(str, parts)])
    codesign = "/usr/bin/codesign"
    if sign and Path(codesign).exists():
        # Ad-hoc signature over the whole bundle: the Keychain ties the stored
        # token to the app's code identity, and an unsealed bundle would ask
        # for the login password again after every rebuild.
        commands.append([codesign, "--force", "--sign", "-", "--timestamp=none", str(staging)])
    if dry_run:
        return {"app": app, "commands": commands, "dry_run": True}

    if staging.exists():
        shutil.rmtree(staging)
    macos_dir.mkdir(parents=True)
    (staging / "Contents" / "Resources").mkdir(parents=True)
    (staging / "Contents" / "Info.plist").write_text(
        companion_render_info_plist(template.read_text(encoding="utf-8")), encoding="utf-8")
    (staging / "Contents" / "PkgInfo").write_text("APPL????", encoding="ascii")
    try:
        for cmd in commands:
            proc = run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                tail = ((proc.stderr or "") + (proc.stdout or "")).strip()[-2000:]
                raise CompanionError(f"{Path(cmd[0]).name} failed ({proc.returncode}):\n{tail}")
        for leftover in macos_dir.glob(f"{COMPANION_EXECUTABLE}-*"):
            leftover.unlink()
        if app.exists():
            shutil.rmtree(app)
        staging.rename(app)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    log_event(_companion_log, 20, "companion.built", app=str(app), arch=",".join(arches))
    return {"app": app, "commands": commands, "dry_run": False}


def companion_install_paths(home: Path | None = None) -> dict:
    home = Path(home if home is not None else Path.home())
    return {"app": home / "Applications" / f"{COMPANION_APP_NAME}.app",
            "launch_agent": home / "Library" / "LaunchAgents" / COMPANION_LAUNCH_AGENT}


def companion_launch_agent_plist(app_path: Path) -> bytes:
    """A per-user LaunchAgent that starts the app at login (and never restarts it).

    KeepAlive stays off so Quit in the menu means quit; LimitLoadToSessionType
    Aqua keeps launchd from trying to start a GUI app in an SSH session.
    """
    return plistlib.dumps({
        "Label": COMPANION_BUNDLE_ID,
        "ProgramArguments": [str(Path(app_path) / "Contents" / "MacOS" / COMPANION_EXECUTABLE)],
        "RunAtLoad": True,
        "KeepAlive": False,
        "ProcessType": "Interactive",
        "LimitLoadToSessionType": "Aqua",
    })


def companion_install(app_src: Path | None = None, *, home: Path | None = None,
                      login: bool = False, root: Path | None = None) -> dict:
    """Copy the built app to ~/Applications; with login, add the LaunchAgent."""
    src = Path(app_src) if app_src is not None else \
        Path(root if root is not None else ROOT) / "dist" / f"{COMPANION_APP_NAME}.app"
    if not (src / "Contents" / "MacOS" / COMPANION_EXECUTABLE).exists():
        raise CompanionError(f"{src} is not a built companion app: run "
                             f"`deploy.py companion build` first")
    paths = companion_install_paths(home)
    dest = paths["app"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.installing")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp, symlinks=True)
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    agent = None
    if login:
        agent = paths["launch_agent"]
        agent.parent.mkdir(parents=True, exist_ok=True)
        agent.write_bytes(companion_launch_agent_plist(dest))
    log_event(_companion_log, 20, "companion.installed", app=str(dest), login=bool(login))
    return {"app": dest, "launch_agent": agent}


def companion_uninstall(*, home: Path | None = None) -> dict:
    """Remove the installed app and its LaunchAgent (whichever exist)."""
    paths = companion_install_paths(home)
    removed = []
    if paths["launch_agent"].exists():
        paths["launch_agent"].unlink()
        removed.append(paths["launch_agent"])
    if paths["app"].exists():
        shutil.rmtree(paths["app"])
        removed.append(paths["app"])
    log_event(_companion_log, 20, "companion.uninstalled", removed=len(removed))
    return {"removed": removed, **paths}


def companion_plugin_source(root: Path | None = None) -> Path:
    return companion_source_dir(root) / "swiftbar" / COMPANION_PLUGIN_NAME


def companion_swiftbar_dir() -> Path | None:
    """SwiftBar's configured plugin folder, else xbar's default one, else None."""
    try:
        proc = subprocess.run(["/usr/bin/defaults", "read", "com.ameba.SwiftBar",
                               "PluginDirectory"], capture_output=True, text=True, timeout=5)
        if proc.returncode == 0 and proc.stdout.strip():
            return Path(proc.stdout.strip()).expanduser()
    except Exception:  # noqa: BLE001
        pass
    xbar = Path.home() / "Library" / "Application Support" / "xbar" / "plugins"
    return xbar if xbar.is_dir() else None


def companion_install_plugin(dest_dir: Path, *, root: Path | None = None) -> Path:
    """Copy the SwiftBar/xbar plugin into `dest_dir` (executable, no secrets in it)."""
    src = companion_plugin_source(root)
    if not src.exists():
        raise CompanionError(f"{src} is missing: install the plugin from a checkout")
    dest_dir = Path(dest_dir).expanduser()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / COMPANION_PLUGIN_NAME
    shutil.copyfile(src, dest)
    dest.chmod(0o755)
    return dest


def _companion_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deploy.py companion",
        description="Build and install the macOS menu-bar companion (LocalLLM Bar) "
                    "or the SwiftBar/xbar plugin.")
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build", help="compile dist/LocalLLM Bar.app with swiftc")
    build.add_argument("--out", default=None, help="output folder (default: dist/)")
    build.add_argument("--arch", default=None, help="arm64, x86_64 or universal (default: this Mac)")
    build.add_argument("--swiftc", default=None, help="path to swiftc")
    build.add_argument("--no-sign", action="store_true", help="skip the ad-hoc codesign")
    build.add_argument("--no-optimize", action="store_true",
                       help="compile without -O (seconds instead of most of a minute)")
    build.add_argument("--dry-run", action="store_true", help="print the commands only")
    install = sub.add_parser("install", help="copy the app to ~/Applications")
    install.add_argument("--app", default=None, help="built app to install (default: dist/)")
    install.add_argument("--login", action="store_true",
                         help="also start it at login (a per-user LaunchAgent)")
    sub.add_parser("uninstall", help="remove the app and its login item")
    plugin = sub.add_parser("plugin", help="install the SwiftBar/xbar plugin")
    plugin.add_argument("--dir", default=None,
                        help="plugin folder (default: SwiftBar's, else xbar's)")
    plugin.add_argument("--print", dest="print_path", action="store_true",
                        help="print the plugin's path in this checkout and exit")
    return parser


def companion_cli(argv: list[str]) -> int:
    """`deploy.py companion build|install|uninstall|plugin`."""
    args = _companion_parser().parse_args(argv)
    try:
        if args.action == "build":
            result = companion_build(Path(args.out) if args.out else None, swiftc=args.swiftc,
                                     arch=args.arch, sign=not args.no_sign,
                                     dry_run=args.dry_run, optimize=not args.no_optimize)
            if result["dry_run"]:
                for cmd in result["commands"]:
                    print(" ".join(f"'{c}'" if " " in c else c for c in cmd))
                return 0
            print(f"Built {result['app']}")
            print("Install it with: python3 deploy.py companion install [--login]")
            return 0
        if args.action == "install":
            result = companion_install(Path(args.app) if args.app else None, login=args.login)
            print(f"Installed {result['app']}")
            if result["launch_agent"]:
                print(f"Login item: {result['launch_agent']} (takes effect at next login)")
            print(f"Start it now with: open '{result['app']}'")
            print("Then set the server URL and an API token (status scope; add control "
                  "for restart/cancel) under Settings in its menu.")
            return 0
        if args.action == "uninstall":
            result = companion_uninstall()
            if not result["removed"]:
                print("Nothing to remove: the companion is not installed.")
            for path in result["removed"]:
                print(f"Removed {path}")
            print("If it is still running, choose Quit from its menu.")
            return 0
        if args.action == "plugin":
            if args.print_path:
                print(companion_plugin_source())
                return 0
            target = Path(args.dir).expanduser() if args.dir else companion_swiftbar_dir()
            if target is None:
                print("No SwiftBar or xbar plugin folder found: pass --dir PATH.")
                return 2
            dest = companion_install_plugin(target)
            print(f"Installed {dest}")
            print("It reads LLM_URL and LLM_API_TOKEN, else ~/.config/local-llm/credentials.json.")
            return 0
    except CompanionError as exc:
        print(f"error: {exc}")
        return 1
    return 2


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'COMPANION_APP_NAME',
    'COMPANION_BUNDLE_ID',
    'COMPANION_EXECUTABLE',
    'COMPANION_LAUNCH_AGENT',
    'COMPANION_MIN_MACOS',
    'COMPANION_PLUGIN_NAME',
    'COMPANION_SUMMARY_PATH',
    'COMPANION_VERSION',
    'CompanionError',
    '_COMPANION_LOCK',
    '_COMPANION_MAX_NODES',
    '_COMPANION_MAX_TASK_IDS',
    '_COMPANION_METRICS_CACHE',
    '_COMPANION_METRICS_TTL_S',
    '_companion_arches',
    '_companion_log',
    '_companion_model',
    '_companion_nodes',
    '_companion_parser',
    '_companion_queue',
    '_companion_tasks',
    '_companion_xml_escape',
    'companion_build',
    'companion_build_command',
    'companion_can',
    'companion_cli',
    'companion_find_swiftc',
    'companion_hour_metrics',
    'companion_indicator',
    'companion_install',
    'companion_install_paths',
    'companion_install_plugin',
    'companion_launch_agent_plist',
    'companion_plugin_source',
    'companion_render_info_plist',
    'companion_reset_cache',
    'companion_source_dir',
    'companion_summary',
    'companion_swiftbar_dir',
    'companion_uninstall',
    'register_companion',
]
