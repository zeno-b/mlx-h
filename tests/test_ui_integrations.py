"""Tests for the integrations Settings panels and the tool catalogue route.

The panels (Connect a client, API tokens, MCP servers, Tool catalogue,
Approvals, Audit, Projects) are static markup plus one script part, so most of
what can break is structural: an id the script looks up that the markup lost,
a route the panel no longer calls, an admin-only block that ships ungated, or
a sub-tab whose loader never runs. GET /api/tools/catalog is the one server
piece these panels added; its tests pin who sees what.
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_llm.config as config_mod  # noqa: E402

config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-ui-integrations-"))

from local_llm.config import Config  # noqa: E402
from local_llm.settings_schema import (  # noqa: E402
    SETTINGS_FIELDS, SETTINGS_SUBTABS, settings_field_sub)
from local_llm.ui import HTML_PAGE, render_ui  # noqa: E402
from local_llm.ui_script_integrations import UI_JS_INTEGRATIONS  # noqa: E402
from local_llm.ui_settings import UI_SETTINGS_HTML  # noqa: E402

PANELS = ("clients", "tokens", "mcp", "mcpedit", "catalogue", "approvals", "policies",
          "audit", "projects")
ADMIN_PANELS = ("mcp", "mcpedit", "audit")
ROUTES = (
    "/api/integrations/snippets", "/api/auth/tokens", "/api/mcp", "/api/config",
    "/api/tools/catalog", "/api/tools", "/api/agents", "/api/agents/capabilities",
    "/api/tool-policies", "/api/approvals/pending", "/api/audit/tool-decisions",
    "/api/projects", "/api/users", "/api/chat/turns/",
)


def _script(page: str) -> str:
    match = re.search(r"<script>(.*?)</script>", page, re.S)
    assert match, "no <script> block"
    return match.group(1)


def _panel(name: str) -> str:
    """The markup of one data-panel block (up to the next panel or section)."""
    start = UI_SETTINGS_HTML.index(f'data-panel="{name}"')
    open_tag = UI_SETTINGS_HTML.rindex("<div", 0, start)
    nxt = [i for i in (UI_SETTINGS_HTML.find('data-panel="', start + 12),
                       UI_SETTINGS_HTML.find("</section>", start)) if i > 0]
    return UI_SETTINGS_HTML[open_tag:min(nxt)]


# ---------------------------------------------------------------- markup --

def test_every_panel_renders_once_with_its_gating():
    for name in PANELS:
        assert UI_SETTINGS_HTML.count(f'data-panel="{name}"') == 1, name
        tag = UI_SETTINGS_HTML[UI_SETTINGS_HTML.rindex("<div", 0, UI_SETTINGS_HTML.index(
            f'data-panel="{name}"')):UI_SETTINGS_HTML.index(f'data-panel="{name}"')]
        assert "panel-block" in tag, name
        assert ("admin-only" in tag) == (name in ADMIN_PANELS), name


def test_panels_sit_in_their_sub_tabs():
    html = UI_SETTINGS_HTML
    tools = [t["key"] for t in SETTINGS_SUBTABS["tools"]]
    assert tools[0] == "catalogue"
    assert {"catalogue", "clients", "tokens", "approvals", "mcp"} <= set(tools)
    assert "audit" in [t["key"] for t in SETTINGS_SUBTABS["admin"]]
    # The MCP sub-tab button and panel are admin-only; the others are not.
    assert re.search(r'<button type="button" class="btn-sm admin-only" data-sub="mcp"', html)
    assert re.search(r'<div class="sub-panel hidden admin-only" data-sub="mcp">', html)
    for sub, panel in (("clients", "clients"), ("tokens", "tokens"), ("mcp", "mcpedit"),
                       ("catalogue", "catalogue"), ("approvals", "policies")):
        opening = html.index(f'<div class="sub-panel', html.index(f'data-subtabs="tools"'))
        sub_at = html.index(f'data-sub="{sub}">', opening)
        panel_at = html.index(f'data-panel="{panel}"')
        later = [html.find(f'data-sub="{t}">', sub_at + 1) for t in tools if t != sub]
        end = min([i for i in later if i > sub_at] + [html.index("</section>", sub_at)])
        assert sub_at < panel_at < end, (sub, panel)
    agent = html[re.search(r'<section[^>]*data-section="agent"', html).start():]
    agent = agent[:agent.index("</section>")]
    assert 'data-panel="projects"' in agent


def test_tools_fields_land_in_the_matching_sub_tab():
    by_key = {f["key"]: f for f in SETTINGS_FIELDS}
    for key, sub in (("tool_approval_mode", "approvals"), ("mcp_server_enabled", "mcp"),
                     ("v1_default_mode", "clients")):
        if key in by_key:
            assert settings_field_sub(by_key[key]) == sub, key
    for f in SETTINGS_FIELDS:
        if f["section"] == "tools":
            assert settings_field_sub(f) in [t["key"] for t in SETTINGS_SUBTABS["tools"]]


def test_ids_the_script_uses_exist_and_are_unique():
    page = render_ui()
    defined = re.findall(r'\bid="([A-Za-z0-9_]+)"', page)
    refs = set(re.findall(r"getElementById\(['\"]([A-Za-z0-9_]+)['\"]\)", UI_JS_INTEGRATIONS))
    assert refs, "the integrations script looks nothing up"
    missing = sorted(refs - set(defined))
    assert not missing, missing
    for ident in refs:
        assert defined.count(ident) == 1, ident
    # The bespoke ids the old loaders used survive.
    for ident in ("toolsList", "mcpStatus"):
        assert page.count(f'id="{ident}"') == 1, ident


def test_every_route_is_referenced():
    for route in ROUTES:
        assert route in UI_JS_INTEGRATIONS, route


def test_no_inline_styles_and_kit_controls_only():
    for name in PANELS:
        block = _panel(name)
        assert 'style="' not in block, name
        assert "<table" not in block, name   # tables are built by the script
    assert "style=" not in UI_JS_INTEGRATIONS.replace(".style.setProperty", "")
    # Every seg in the panels is a kit segmented control.
    for seg in re.findall(r'<div class="([^"]*)" role="radiogroup"', UI_SETTINGS_HTML):
        assert "seg" in seg.split(), seg


def test_admin_only_blocks_inside_user_panels():
    tokens = _panel("tokens")
    assert re.search(r'class="check admin-only"[^>]*>\s*<input type="checkbox" value="control"',
                     tokens)
    assert 'id="integTokAllWrap"' in tokens and "admin-only" in tokens.split(
        'id="integTokAllWrap"')[0].rsplit("<", 1)[1]
    policies = _panel("policies")
    for scope in ("profile", "global"):
        assert re.search(r'<button type="button" class="btn-sm admin-only" data-value="%s"'
                         % scope, policies), scope
    projects = _panel("projects")
    assert re.search(r'class="[^"]*admin-only[^"]*"[^>]*id="integProjSharedWrap"', projects) or \
        re.search(r'id="integProjSharedWrap"[^>]*class="[^"]*admin-only', projects)


def test_script_dialect_and_conventions():
    code = re.sub(r"//[^\n]*", "", UI_JS_INTEGRATIONS)
    assert "=>" not in code, "arrow function"
    assert "`" not in code, "template literal"
    assert not re.search(r"^\s*(let|const|class)\s", code, re.M), "ES6 declaration"
    assert "alert(" not in code
    assert not re.search(r"(?<![\w.])confirm\(", code), "confirm() instead of uiConfirm"
    assert "innerHTML = text" not in code
    assert not re.search(r'\.innerHTML\s*=(?!\s*"")', code), "innerHTML with content"
    for bad in ("<script>", "</script>", "<style>", "</style>"):
        assert bad not in UI_JS_INTEGRATIONS
    assert "\\\\n" not in UI_JS_INTEGRATIONS
    # Lazy loading: each sub-tab registers with the Settings tab registry.
    for key in ("tools/catalogue", "tools/clients", "tools/tokens", "tools/approvals",
                "tools/mcp", "admin/audit", "agent"):
        assert f'"{key}"' in UI_JS_INTEGRATIONS, key
    # Its own names: no clash with another script part.
    names = re.findall(r"^\s*(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(",
                       _script(render_ui()), re.M)
    assert not sorted({n for n in names if names.count(n) > 1})
    mine = re.findall(r"^\s*(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(",
                      UI_JS_INTEGRATIONS, re.M)
    assert mine and all(n.startswith("integ") for n in mine), mine


def test_script_part_is_assembled_after_admin_and_before_settings():
    script = _script(HTML_PAGE)
    assert script.index("var ADMIN_TABS = [") < script.index("function integRegisterTab")
    assert script.index("function integRegisterTab") < script.index("function openSettings")
    assert HTML_PAGE.count("<script>") == 1


# ----------------------------------------------------------------- route --

def _build_app(**overrides):
    from local_llm.api import create_app
    from local_llm.core import ADAPTER_DIR
    from local_llm.database import Database
    from local_llm.model_server import ModelServerManager
    from local_llm.tools import ToolRegistry
    from local_llm.training import RetrainManager

    base = dict(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                allow_test_user=False, node_token="secret-node-token",
                skills_dir=tempfile.mkdtemp(prefix="local-llm-test-skills-"))
    base.update(overrides)
    cfg = Config(**base)
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)  # never started
    rm = RetrainManager(db, mm, cfg)
    app = create_app(cfg, db, mm, rm, ToolRegistry(cfg, db))
    app.state.model_manager = mm
    return app, cfg, db


def _login(app, username, password):
    from fastapi.testclient import TestClient
    client = TestClient(app)
    r = client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return client


def test_catalog_route_admin_view():
    app, cfg, _db = _build_app(role_capabilities='{"user": ["web_search"]}')
    admin = _login(app, "admin", "adminpw123")
    r = admin.post("/api/agents", json={"name": "Researcher", "description": "web",
                                        "capabilities": ["web_api"]})
    assert r.status_code == 200, r.text
    out = admin.get("/api/tools/catalog").json()
    assert out["admin"] is True and out["role"] == "admin"
    tools = {t["name"]: t for t in out["tools"]}
    # Every known tool, including extra tools no agent registers by default.
    assert "browse_page" in tools and tools["browse_page"]["source"] == "extra"
    assert tools["browse_page"]["registered"] is False
    assert tools["web_search"]["source"] == "builtin" and tools["web_search"]["registered"]
    assert tools["calculator"]["always"] is True
    assert tools["run_shell"]["risk"] == "exec"
    assert tools["read_file"]["risk"] == "read"
    assert tools["fetch_url"]["capability"] == "web_api"
    assert [p["name"] for p in tools["fetch_url"]["profiles"]] == ["Researcher"]
    assert tools["read_file"]["profiles"] == []
    # Role grants come from ROLE_CAPABILITIES via capabilities_for_role.
    assert "user" in tools["web_search"]["roles"] and "admin" in tools["web_search"]["roles"]
    assert "user" not in tools["run_shell"]["roles"]
    assert out["roles"]["user"] and out["role_capabilities_set"] is True
    assert out["mcp"] == {"servers_mutable": "mcp_servers" in Config.MUTABLE,
                          "server_enabled": False, "endpoint": "/mcp"}
    caps = {c["key"]: c for c in out["capabilities"]}
    assert "fetch_url" in caps["web_api"]["tools"]


def test_catalog_route_user_view_hides_admin_facts():
    app, cfg, _db = _build_app()
    admin = _login(app, "admin", "adminpw123")
    for name, enabled in (("Visible", True), ("Hidden", False)):
        r = admin.post("/api/agents", json={"name": name, "description": "",
                                            "capabilities": ["file_ops"], "enabled": enabled})
        assert r.status_code == 200, r.text
        if not enabled:
            agent_id = r.json()["agent"]["id"]
            admin.post(f"/api/agents/{agent_id}", json={"enabled": False})
    r = admin.post("/api/users", json={"username": "bob", "password": "bobpw12345",
                                       "role": "user"})
    assert r.status_code == 200, r.text
    bob = _login(app, "bob", "bobpw12345")
    out = bob.get("/api/tools/catalog").json()
    assert out["admin"] is False and out["role"] == "user"
    assert "roles" not in out and "mcp" not in out and "role_capabilities_set" not in out
    tools = {t["name"]: t for t in out["tools"]}
    assert all("roles" not in t for t in out["tools"])
    # Unregistered extra tools stay out of a user's view unless a profile grants them.
    assert "browse_page" not in tools
    names = {p["name"] for t in out["tools"] for p in t["profiles"]}
    assert names == {"Visible"}, names


def test_catalog_route_needs_sign_in():
    from fastapi.testclient import TestClient
    app, _cfg, _db = _build_app()
    assert TestClient(app).get("/api/tools/catalog").status_code == 401


def _run_standalone():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            import traceback; traceback.print_exc(); print(f"FAIL  {name}: {exc}"); failed += 1
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
