"""The tool catalogue route: what each tool is, how risky, and who may use it.

GET /api/tools lists the default agent's tools by name and description only.
Settings > Tools > Catalogue also needs, per tool, the capability that grants
it, its risk class (the approval policy's view), where it comes from (built
in, the named extra tools, or an MCP server) and which agent profiles and roles
reach it. Those answers live in four modules: config's capability table and
role grants, policy's risk classifier, tools_extra's tool names and mcp's live
servers. Working them out in the browser would mean a second copy of
capabilities_for_role and of the MCP naming rule, which would drift from the
copy the server enforces with. This one read-only route asks the same
functions the server uses to decide.

Visibility follows the data-isolation rule of the rest of the app: a user sees
the default agent's tools plus the tools of the profiles they may pick (enabled
ones, narrowed by their role), and the profiles column names only enabled
profiles. Administrators see every known tool, including ones no agent
registers today, the per-role grants and the MCP settings facts.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from .core import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .policy import *  # noqa: F401,F403
from .mcp import *  # noqa: F401,F403
from .tools_extra import *  # noqa: F401,F403
from .mcp_server import *  # noqa: F401,F403


# Display order of the tool sources (the panel groups by it).
TOOL_CATALOG_SOURCES = ("builtin", "extra", "mcp")


def _tcat_role_table(config: Any) -> dict:
    """ROLE_CAPABILITIES parsed, {} when unset or malformed (Config clamps it)."""
    raw = str(getattr(config, "role_capabilities", "") or "").strip()
    if not raw:
        return {}
    try:
        table = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return table if isinstance(table, dict) else {}


def _tcat_cap_tools() -> dict[str, set[str]]:
    """Capability key -> the concrete tools it unlocks (always-on ones excluded).

    tools_for_capabilities() is the function profiles, subagents and the MCP
    endpoint resolve grants with, so MCP's per-server keys resolve here exactly
    as they do at run time.
    """
    always = set(ALWAYS_ON_TOOLS)
    out: dict[str, set[str]] = {}
    for key in list(CAPABILITY_ORDER):
        try:
            out[key] = set(tools_for_capabilities([key])) - always
        except Exception:  # noqa: BLE001 - a key removed mid-sync grants nothing
            out[key] = set()
    return out


def _tcat_mcp_servers() -> dict[str, str]:
    """Local MCP tool name -> the server it comes from."""
    owners: dict[str, str] = {}
    try:
        for server in MCP.status():
            for name in server.get("tools") or []:
                owners[str(name)] = str(server.get("name") or "")
    except Exception:  # noqa: BLE001 - a broken manager must not hide the catalogue
        pass
    return owners


def tool_catalog(config: Any, db: Any, registry: Any, user: dict) -> dict:
    """The catalogue payload for `user` (see the module docstring)."""
    admin = user.get("role") == "admin"
    role = "admin" if admin else str(user.get("role") or "user")
    always = set(ALWAYS_ON_TOOLS)
    cap_tools = _tcat_cap_tools()
    mcp_owner = _tcat_mcp_servers()
    extra = set(EXTRA_TOOL_NAMES)
    try:
        registered = list(registry.names()) if registry is not None else []
    except Exception:  # noqa: BLE001
        registered = []
    try:
        profiles = list(db.list_agents()) if db is not None else []
    except Exception:  # noqa: BLE001
        profiles = []
    if not admin:
        profiles = [p for p in profiles if p.get("enabled")]

    role_caps = set(capabilities_for_role(role, config))
    profile_tools: dict[str, set[str]] = {}
    for p in profiles:
        caps = set(p.get("capabilities") or [])
        if not admin:
            # What this user would actually get from the profile in chat.
            caps &= role_caps
        try:
            profile_tools[str(p.get("id"))] = set(tools_for_capabilities(sorted(caps)))
        except Exception:  # noqa: BLE001
            profile_tools[str(p.get("id"))] = set(always)

    roles: dict[str, list[str]] = {}
    role_tools: dict[str, set[str]] = {}
    if admin:
        names = ["admin", "user"] + sorted(k for k in _tcat_role_table(config)
                                          if k not in ("admin", "user"))
        for name in names:
            caps = capabilities_for_role(name, config)
            roles[name] = list(caps)
            try:
                role_tools[name] = set(tools_for_capabilities(caps))
            except Exception:  # noqa: BLE001
                role_tools[name] = set(always)

    visible = set(registered)
    if admin:
        visible |= always
        for tools in cap_tools.values():
            visible |= tools
        visible |= set(mcp_owner)
    else:
        for tools in profile_tools.values():
            visible |= tools

    order = {name: i for i, name in enumerate(registered)}
    out_tools = []
    for name in visible:
        tool = registry.get(name) if registry is not None else None
        server = mcp_owner.get(name)
        source = "mcp" if name in mcp_owner else ("extra" if name in extra else "builtin")
        granting = [k for k in CAPABILITY_ORDER if name in cap_tools.get(k, set())]
        declared = getattr(tool, "capability", None) if tool is not None else None
        primary = str(declared) if declared else (
            f"mcp:{server}" if server and f"mcp:{server}" in granting
            else (granting[0] if granting else None))
        try:
            risk = classify_tool_risk(name, {}, tool)
        except Exception:  # noqa: BLE001
            risk = "external"
        entry = {
            "name": name,
            "description": (getattr(tool, "description", "") or "") if tool is not None else "",
            "capability": primary,
            "capabilities": granting,
            "always": name in always,
            "risk": risk,
            "source": source,
            "server": server or None,
            "registered": name in order,
            "profiles": [{"id": p.get("id"), "name": p.get("name"),
                          "enabled": bool(p.get("enabled"))}
                         for p in profiles if name in profile_tools.get(str(p.get("id")), ())],
        }
        if admin:
            entry["roles"] = [r for r in roles if name in role_tools.get(r, ())]
        out_tools.append(entry)
    out_tools.sort(key=lambda t: (TOOL_CATALOG_SOURCES.index(t["source"]),
                                  order.get(t["name"], len(order)), t["name"]))

    payload: dict = {
        "tools": out_tools,
        "role": role,
        "admin": admin,
        "capabilities": [
            {**cap, "tools": sorted(cap_tools.get(cap["key"], set()))}
            for cap in public_capabilities()
        ],
        "always_on": list(ALWAYS_ON_TOOLS),
    }
    if admin:
        payload["roles"] = roles
        payload["role_capabilities_set"] = bool(_tcat_role_table(config))
        # The MCP panel shows the editor's Apply button only when the running
        # process can adopt MCP_SERVERS without a restart.
        payload["mcp"] = {
            "servers_mutable": "mcp_servers" in getattr(type(config), "MUTABLE", ()),
            "server_enabled": bool(getattr(config, "mcp_server_enabled", False)),
            "endpoint": MCP_SERVER_PATH,
        }
    return payload


def register_tool_catalog(app: Any, ctx: Any) -> None:
    """GET /api/tools/catalog (any signed-in user; admins see more)."""

    @app.get("/api/tools/catalog")
    async def tools_catalog(user: dict = ctx.USER):
        return await asyncio.to_thread(tool_catalog, ctx.config, ctx.db, ctx.registry, user)


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    "TOOL_CATALOG_SOURCES",
    "_tcat_cap_tools",
    "_tcat_mcp_servers",
    "_tcat_role_table",
    "register_tool_catalog",
    "tool_catalog",
]
