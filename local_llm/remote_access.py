"""Secure remote access: bind policy, proxy awareness, and `deploy.py remote`.

The server always listened on 127.0.0.1, which was safe and useless from a
phone or a second office. Reaching it remotely means one of three things, each
with its own trap:

* Binding a LAN/public interface (WEB_HOST / --host). With AUTH_ENABLED=0 every
  visitor is the implicit admin, so a non-loopback bind is REFUSED unless
  authentication is on, and warned about unless the cookie is HTTPS-only.
* A reverse proxy or tunnel (Tailscale Serve/Funnel, cloudflared). Every
  request then arrives from 127.0.0.1, so the login throttle would lock out
  all users together, and the OIDC redirect URI would be built from the
  internal origin. TRUSTED_PROXIES makes X-Forwarded-For believable ONLY when
  the socket peer is a listed proxy (anyone can send the header otherwise),
  and PUBLIC_BASE_URL is the origin the browser actually sees.
* Identity-aware proxies (Cloudflare Access, Tailscale identity headers) are
  an extra gate in front of the app, not a replacement for its own login: the
  app does not trust their identity headers, so AUTH_ENABLED=1 stays on.

`deploy.py remote status|tailscale|funnel|cloudflare|gen-key` prints the exact
commands (dry run) and runs them only with --apply.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import secrets
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .nodesec import *  # noqa: F401,F403


_ra_log = get_logger("remote")
_RA_LOOPBACK_NAMES = ("127.0.0.1", "::1", "localhost")
# The macOS app ships the CLI inside the bundle; a shell alias is not visible
# to subprocess, so the bundle path is probed directly.
_RA_TAILSCALE_APP = "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
_ra_net_cache: dict[str, list] = {}


# --------------------------------------------------------------------------- #
# Proxy-aware request helpers (used by api.py)
# --------------------------------------------------------------------------- #
def _ra_networks(spec: str) -> list:
    spec = str(spec or "").strip()
    if spec in _ra_net_cache:
        return _ra_net_cache[spec]
    nets = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            nets.append(ipaddress.ip_network(item, strict=False))
        except ValueError:
            log(f"TRUSTED_PROXIES: ignoring {item!r} (not an IP or CIDR)", logging.WARNING)
    _ra_net_cache[spec] = nets
    return nets


def _ra_ip(text: str) -> Any:
    try:
        return ipaddress.ip_address(str(text or "").strip().strip("[]").split("%", 1)[0])
    except ValueError:
        return None


def ip_is_trusted_proxy(config: Any, host: str) -> bool:
    ip = _ra_ip(host)
    if ip is None:
        return False
    return any(ip.version == net.version and ip in net
               for net in _ra_networks(getattr(config, "trusted_proxies", "")))


def client_ip_for(request: Any, config: Any) -> str:
    """The real client address, for the login throttle and the join limiter.

    With TRUSTED_PROXIES empty (the default) this is the socket peer, exactly
    as before. Behind a listed proxy, X-Forwarded-For is walked from the right
    (the entry the nearest proxy appended) past further trusted hops, and the
    first untrusted address is the client. A forged left-most entry is
    therefore never chosen while a trusted proxy appended the true one.
    """
    peer = request.client.host if getattr(request, "client", None) else "?"
    if not getattr(config, "trusted_proxies", "") or not ip_is_trusted_proxy(config, peer):
        return peer
    try:
        raw = request.headers.get("x-forwarded-for") or ""
    except Exception:  # noqa: BLE001
        raw = ""
    hops = [h.strip() for h in raw.split(",") if h.strip()]
    for hop in reversed(hops):
        ip = _ra_ip(hop)
        if ip is None:
            break           # garbage: stop trusting the chain here
        if not ip_is_trusted_proxy(config, str(ip)):
            return str(ip)
    return peer


def public_url_for(request: Any, config: Any, path: str) -> str:
    """An absolute URL the BROWSER can reach (PUBLIC_BASE_URL behind a proxy)."""
    base = str(getattr(config, "public_base_url", "") or "").strip().rstrip("/")
    if not base:
        base = f"{request.url.scheme}://{request.url.netloc}"
    return base + path


# --------------------------------------------------------------------------- #
# Bind policy (used by cli.py for uvicorn.run)
# --------------------------------------------------------------------------- #
def host_is_loopback(host: str) -> bool:
    h = str(host or "").strip().strip("[]").lower()
    if h in _RA_LOOPBACK_NAMES:
        return True
    ip = _ra_ip(h)
    return bool(ip is not None and ip.is_loopback)


def bind_host_problem(config: Any) -> tuple[str, str]:
    """(refusal, warning) for the configured WEB_HOST; empty strings when fine."""
    host = str(getattr(config, "web_host", "") or "127.0.0.1")
    if host_is_loopback(host):
        return "", ""
    if not getattr(config, "auth_enabled", False):
        return (f"refusing to listen on {host}: AUTH_ENABLED=0 makes every visitor an "
                "unauthenticated admin. Set AUTH_ENABLED=1 (and an admin password or "
                "Entra ID), or keep WEB_HOST=127.0.0.1 and publish through "
                "`deploy.py remote tailscale`."), ""
    if not getattr(config, "auth_cookie_secure", False):
        return "", (f"listening on {host} with AUTH_COOKIE_SECURE=0: the session cookie "
                    "will travel over plain HTTP. Put TLS in front (Tailscale Serve, "
                    "cloudflared, a reverse proxy) and set AUTH_COOKIE_SECURE=1.")
    return "", ""


def bind_host_or_exit(config: Any) -> str:
    """The host uvicorn should bind, or SystemExit when the policy refuses it."""
    refusal, warning = bind_host_problem(config)
    if refusal:
        log(refusal, logging.ERROR)
        raise SystemExit(refusal)
    if warning:
        log(warning, logging.WARNING)
    host = str(getattr(config, "web_host", "") or "127.0.0.1")
    if not host_is_loopback(host):
        log_event(_ra_log, logging.WARNING, "remote.bind_non_loopback", host=host,
                  trusted_proxies=bool(getattr(config, "trusted_proxies", "")))
    return host


# --------------------------------------------------------------------------- #
# deploy.py remote ...
# --------------------------------------------------------------------------- #
def tailscale_binary() -> str:
    found = shutil.which("tailscale")
    if found:
        return found
    return _RA_TAILSCALE_APP if Path(_RA_TAILSCALE_APP).exists() else ""


def _ra_default_runner(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _ra_tailscale_info(runner: Callable[[list[str]], tuple[int, str]]) -> dict:
    ts = tailscale_binary()
    if not ts:
        return {"installed": False}
    code, out = runner([ts, "status", "--json"])
    info: dict = {"installed": True, "binary": ts, "running": code == 0}
    if code == 0:
        try:
            data = json.loads(out)
            me = data.get("Self") or {}
            info["dns_name"] = str(me.get("DNSName") or "").rstrip(".")
            info["ips"] = me.get("TailscaleIPs") or []
            info["backend_state"] = data.get("BackendState")
        except ValueError:
            pass
    else:
        info["error"] = out.strip()[:200]
    return info


def _ra_print_cmds(cmds: list[list[str]]) -> None:
    for cmd in cmds:
        print("  $ " + shlex.join(cmd))


def _ra_apply(cmds: list[list[str]], runner: Callable) -> int:
    for cmd in cmds:
        print("$ " + shlex.join(cmd))
        code, out = runner(cmd)
        if out.strip():
            print(out.rstrip())
        if code != 0:
            print(f"command failed with exit code {code}; stopping.", file=sys.stderr)
            return code or 1
    return 0


def _ra_env_lines(config: Any, public_url: str, proxies: str) -> list[str]:
    lines = ["AUTH_ENABLED=1", "AUTH_COOKIE_SECURE=1", "WEB_HOST=127.0.0.1",
             f"TRUSTED_PROXIES={proxies}"]
    if public_url:
        lines.append(f"PUBLIC_BASE_URL={public_url}")
        if getattr(config, "oidc_client_id", ""):
            lines.append(f"# register {public_url}/api/auth/oidc/callback as the Entra "
                         "redirect URI (or set OIDC_REDIRECT_URI to it)")
    return lines


def remote_status(config: Any, runner: Callable) -> int:
    refusal, warning = bind_host_problem(config)
    print("Remote access")
    print(f"  bind host        {config.web_host}"
          + ("  (loopback only)" if host_is_loopback(config.web_host) else ""))
    print(f"  auth             {'on' if config.auth_enabled else 'OFF'}"
          f"   secure cookie {'on' if config.auth_cookie_secure else 'off'}")
    print(f"  trusted proxies  {config.trusted_proxies or '(none: socket peer is the client)'}")
    print(f"  public base url  {config.public_base_url or '(derived from each request)'}")
    if refusal:
        print(f"  ! {refusal}")
    if warning:
        print(f"  ! {warning}")
    sec = node_security_summary(config)
    print("Node traffic")
    print(f"  tokens           {sec['tokens']} accepted (NODE_TOKEN, comma list rotates)")
    print(f"  tls              ca={sec['ca_file'] or 'system roots'}  "
          f"client cert={'yes' if sec['client_cert'] else 'no'}")
    print(f"  plaintext        {'allowed anywhere' if sec['allow_plaintext'] else 'private/tailnet peers only'}")
    print(f"  e2e sealing      {'on' if sec['e2e'] else 'off'}"
          f"   cryptography {'installed' if sec['cryptography'] else 'not installed'}")
    if sec["e2e_problem"]:
        print(f"  ! {sec['e2e_problem']}")
    ts = _ra_tailscale_info(runner)
    print("Tools")
    if ts.get("installed"):
        state = ts.get("backend_state") or ("running" if ts.get("running") else "not running")
        print(f"  tailscale        {ts['binary']}  ({state})"
              + (f"  {ts['dns_name']}" if ts.get("dns_name") else ""))
    else:
        print("  tailscale        not installed (https://tailscale.com/download)")
    cf = shutil.which("cloudflared")
    print(f"  cloudflared      {cf or 'not installed (brew install cloudflared)'}")
    return 1 if refusal else 0


def remote_tailscale(config: Any, args: Any, runner: Callable, funnel: bool) -> int:
    ts = tailscale_binary() or "tailscale"
    info = _ra_tailscale_info(runner) if tailscale_binary() else {"installed": False}
    port = int(args.port or config.web_port)
    verb = "funnel" if funnel else "serve"
    cmds = [[ts, verb, "--bg", str(port)]]
    dns = info.get("dns_name") or "<this-machine>.<tailnet>.ts.net"
    public_url = f"https://{dns}"
    title = ("Tailscale Funnel: PUBLIC internet -> this app" if funnel
             else "Tailscale Serve: your tailnet only -> this app")
    print(title)
    print(f"Commands ({'will run' if args.apply else 'dry run, add --apply to run'}):")
    _ra_print_cmds(cmds)
    print(f"  (undo: {shlex.join([ts, verb, 'reset'])})")
    print("Then set in .env and restart:")
    for line in _ra_env_lines(config, public_url, "127.0.0.1,::1"):
        print("  " + line)
    print("Notes:")
    print("  * Tailscale terminates HTTPS with a real certificate and proxies to "
          f"127.0.0.1:{port}, so WEB_HOST stays loopback and every request arrives from "
          "127.0.0.1: that is why TRUSTED_PROXIES must name it.")
    if funnel:
        print("  * Funnel publishes to the whole internet on 443/8443/10000 only. "
              "Requests carry NO Tailscale identity, so the app's own login (ideally "
              "Entra ID) is the only gate: AUTH_ENABLED=1 is required.")
    else:
        print("  * Serve adds Tailscale-User-Login / Tailscale-User-Name headers for "
              "tailnet users. This app does not treat them as a login (anyone on the "
              "tailnet machine itself could forge them), so keep AUTH_ENABLED=1; use "
              "tailnet ACLs to decide who can reach the port at all.")
    if not info.get("installed"):
        print("  * Tailscale is not installed here: https://tailscale.com/download")
    if not args.apply:
        return 0
    if funnel and not config.auth_enabled:
        print("Refusing to open a Funnel with AUTH_ENABLED=0.", file=sys.stderr)
        return 2
    if not tailscale_binary():
        print("tailscale not found; install it first.", file=sys.stderr)
        return 2
    return _ra_apply(cmds, runner)


def remote_cloudflare(config: Any, args: Any, runner: Callable) -> int:
    cf = shutil.which("cloudflared") or "cloudflared"
    port = int(args.port or config.web_port)
    name = args.tunnel_name
    hostname = args.hostname or "<llm.example.com>"
    setup = [[cf, "tunnel", "create", name],
             [cf, "tunnel", "route", "dns", name, hostname]]
    run = [cf, "tunnel", "run", "--url", f"http://127.0.0.1:{port}", name]
    print("Cloudflare Tunnel: public hostname -> this app, no inbound port opened")
    print(f"Commands ({'will run' if args.apply else 'dry run, add --apply to run'}):")
    _ra_print_cmds([[cf, "tunnel", "login"]] + setup + [run])
    print("Then set in .env and restart:")
    for line in _ra_env_lines(config, f"https://{hostname}", "127.0.0.1,::1"):
        print("  " + line)
    print("Notes:")
    print("  * Put a Cloudflare Access application (Zero Trust > Access > Applications) "
          f"in front of {hostname} with an allow policy for your Entra ID group: an "
          "identity-aware proxy that stops strangers before they reach the login page. "
          "The app does not verify the Cf-Access-Jwt-Assertion header, so keep "
          "AUTH_ENABLED=1 as the second gate.")
    print("  * cloudflared connects from this machine, so requests arrive from "
          "127.0.0.1; TRUSTED_PROXIES makes the login throttle key on the real client "
          "(cloudflared sets X-Forwarded-For).")
    print("  * `tunnel login` opens a browser and `tunnel run` stays in the foreground; "
          "--apply runs only the create and route steps. Run the last command yourself "
          "or install it as a service with `cloudflared service install`.")
    if not shutil.which("cloudflared"):
        print("  * cloudflared is not installed here: brew install cloudflared")
    if not args.apply:
        return 0
    if not shutil.which("cloudflared"):
        print("cloudflared not found; install it first.", file=sys.stderr)
        return 2
    if not args.hostname:
        print("--hostname is required with --apply.", file=sys.stderr)
        return 2
    return _ra_apply(setup, runner)


def remote_gen_key() -> int:
    print("# Put the SAME two lines in the .env of every node, then restart them.")
    print(f"NODE_TOKEN={secrets.token_urlsafe(32)}")
    print(f"NODE_E2E_KEY={node_generate_e2e_key()}")
    if not node_crypto_available():
        print("# NODE_E2E_KEY needs the optional package on every node: "
              "pip install cryptography (node traffic is refused without it).")
    return 0


def remote_cli(argv: list[str], runner: Callable | None = None,
               config: Any = None) -> int:
    """`deploy.py remote <status|tailscale|funnel|cloudflare|gen-key> [--apply]`."""
    parser = argparse.ArgumentParser(prog="deploy.py remote",
                                     description="Secure remote access helpers.")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("status", help="Show bind, proxy, node-security and tool state.")
    for verb, text in (("tailscale", "Publish to your tailnet with Tailscale Serve."),
                       ("funnel", "Publish to the internet with Tailscale Funnel.")):
        p = sub.add_parser(verb, help=text)
        p.add_argument("--port", type=int, default=None)
        p.add_argument("--apply", action="store_true")
    p = sub.add_parser("cloudflare", help="Publish through a Cloudflare Tunnel.")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--hostname", default="")
    p.add_argument("--tunnel-name", default="local-llm")
    p.add_argument("--apply", action="store_true")
    sub.add_parser("gen-key", help="Print a fresh NODE_TOKEN and NODE_E2E_KEY.")
    args = parser.parse_args(argv)
    runner = runner or _ra_default_runner
    config = config or Config()
    if args.cmd == "status":
        return remote_status(config, runner)
    if args.cmd in ("tailscale", "funnel"):
        return remote_tailscale(config, args, runner, funnel=args.cmd == "funnel")
    if args.cmd == "cloudflare":
        return remote_cloudflare(config, args, runner)
    if args.cmd == "gen-key":
        return remote_gen_key()
    parser.print_help()
    return 2


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'client_ip_for', 'public_url_for', 'ip_is_trusted_proxy',
    'host_is_loopback', 'bind_host_problem', 'bind_host_or_exit',
    'tailscale_binary', 'remote_cli', 'remote_status',
    '_ra_networks', '_ra_ip',
]
