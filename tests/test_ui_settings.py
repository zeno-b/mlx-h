"""Tests for the Settings view: schema invariants, rendered markup, page assembly.

The Settings panel is generated from local_llm/settings_schema.py, so most of
what used to be hand-checked per control is now a property of the schema: every
runtime-mutable setting has exactly one control (or a documented reason not
to), the control's id/data attributes are what the generic loader binds by,
the bounds a slider shows are bounds the server actually enforces, and the
type a control sends is the type POST /api/config accepts.
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_llm.config as config_mod  # noqa: E402

config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-ui-settings-"))

from local_llm.config import Config  # noqa: E402
from local_llm.settings_schema import (  # noqa: E402
    SECTION_KEYS, SETTINGS_FIELDS, SETTINGS_SECTIONS, SETTINGS_SUBTABS, UI_HIDDEN,
    USER_PREFS, WIDGETS, schema_for_role, settings_control_id, settings_field_sub,
    settings_field_type,
)
from local_llm.ui import HTML_PAGE, render_ui  # noqa: E402
from local_llm.ui_settings import UI_SETTINGS_HTML  # noqa: E402
from local_llm.ui_script_admin import UI_JS_ADMIN  # noqa: E402
from local_llm.ui_script_kit import UI_JS_KIT  # noqa: E402
from local_llm.ui_script_models import UI_JS_MODELS  # noqa: E402
from local_llm.ui_script_settings import UI_JS_SETTINGS  # noqa: E402

_PY_TYPE = {bool: "bool", int: "int", float: "float", str: "str"}

# Bespoke panel elements moved from the old #modelsView / settings aside /
# History backup panel. Their loaders find them by id, so the ids must survive.
BESPOKE_IDS = (
    "modelCurrent modelTable modelCustom adapterSelect kvSize modelLog datasetStats "
    "exportReviewedOnly approvePendingBtn docsStats dropZone fileInput docsPath docsUrl "
    "docsQuery docScope docsResult cfgProjectDir projectStatus sandboxHint projectDiff "
    "usersList newUserName newUserPass newUserRole agentsList newAgentName newAgentDesc "
    "newAgentCaps agentRunPick agentRunPrompt agentRunResult clusterNodes routingSummary "
    "routingEvents importFile importStatus importList restoreInput retrainRecipe perfStats "
    "prefillCurve prefixWarn memKey memValue memoryList toolsList themeBtn"
).split()


def _script(page: str) -> str:
    match = re.search(r"<script>(.*?)</script>", page, re.S)
    assert match, "no <script> block"
    return match.group(1)


# ------------------------------------------------------------------ schema --

def test_every_mutable_setting_has_a_control_or_a_reason():
    shown = {f["key"] for f in SETTINGS_FIELDS if f["scope"] == "global"}
    missing = set(Config.MUTABLE) - shown - set(UI_HIDDEN)
    assert not missing, f"mutable settings with no control: {sorted(missing)}"
    assert not (shown - set(Config.MUTABLE)), "a control writes a key POST /api/config ignores"
    for key, reason in UI_HIDDEN.items():
        assert key in Config.MUTABLE, f"UI_HIDDEN names {key}, which is not mutable"
        assert key not in shown, f"{key} is both hidden and shown"
        assert len(reason) > 20, f"UI_HIDDEN[{key}] needs a real reason"


def test_one_control_per_setting_and_valid_metadata():
    seen = set()
    for f in SETTINGS_FIELDS:
        slot = (f["scope"], f["key"])
        assert slot not in seen, f"{slot} is rendered twice"
        seen.add(slot)
        assert f["scope"] in ("global", "user", "info"), f
        assert f["section"] in SECTION_KEYS, f
        assert f["widget"] in WIDGETS, f
        assert f["label"] and f["help"] and f["group"], f
        if f["widget"] in ("select", "segmented"):
            values = [o["value"] for o in f.get("options", [])]
            assert len(values) >= 2 and len(set(values)) == len(values), f
        if "min" in f and "max" in f:
            assert f["min"] <= f["max"], f
        if f.get("sub"):
            assert f["sub"] in [t["key"] for t in SETTINGS_SUBTABS[f["section"]]], f
        if f["scope"] == "info":
            assert f["widget"] == "readonly" and hasattr(Config(), f["key"]), f
        if f["scope"] == "user":
            assert f["key"] in USER_PREFS, f
    # Every preference is reachable from the panel.
    user_keys = {f["key"] for f in SETTINGS_FIELDS if f["scope"] == "user"}
    assert user_keys == set(USER_PREFS), set(USER_PREFS) ^ user_keys
    # No two controls share an element id.
    ids = [settings_control_id(f) for f in SETTINGS_FIELDS]
    assert len(ids) == len(set(ids))


def test_control_ids_follow_the_naming_convention():
    by_slot = {(f["scope"], f["key"]): settings_control_id(f) for f in SETTINGS_FIELDS}
    assert by_slot[("global", "model_cutoff")] == "cfgModelCutoff"
    assert by_slot[("global", "date_in_prompt")] == "cfgDateInPrompt"
    assert by_slot[("global", "hedge_escalation")] == "cfgHedgeEscalation"
    assert by_slot[("user", "temperature")] == "prefTemperature"
    assert by_slot[("user", "theme")] == "themeBtn"   # the legacy id survives


def test_the_type_a_control_sends_matches_the_config_field():
    cfg = Config()
    for f in SETTINGS_FIELDS:
        if f["scope"] != "global":
            continue
        want = _PY_TYPE[type(getattr(cfg, f["key"]))]
        assert settings_field_type(f) == want, (f["key"], settings_field_type(f), want)


def test_slider_bounds_are_bounds_the_server_enforces():
    """min/max are display hints, but a hint the clamp contradicts is a lie:
    the slider would offer a value the server silently moves."""
    for f in SETTINGS_FIELDS:
        if f["scope"] != "global" or f["widget"] not in ("range", "number"):
            continue
        if "min" not in f or "max" not in f:
            continue
        kind = settings_field_type(f)
        span = max(1.0, float(f["max"]) - float(f["min"]))
        for probe in (f["min"] - span * 10, f["max"] + span * 10):
            cfg = Config()
            value = int(probe) if kind == "int" else float(probe)
            cfg.apply({f["key"]: value})
            got = getattr(cfg, f["key"])
            assert f["min"] - 1e-9 <= got <= f["max"] + 1e-9, (f["key"], value, got)


def test_schema_for_role_hides_admin_sections_and_installation_values():
    user = schema_for_role(False)
    admin = schema_for_role(True)
    assert {s["key"] for s in user["sections"]} == {s["key"] for s in SETTINGS_SECTIONS
                                                   if not s["admin"]}
    assert all(f["scope"] != "global" for f in user["fields"])
    assert "admin" not in user["subtabs"] and set(admin["subtabs"]) == set(SETTINGS_SUBTABS)
    assert len(admin["fields"]) == len(SETTINGS_FIELDS)


# ------------------------------------------------------------------ markup --

def test_every_field_renders_a_bound_control_with_label_and_hint():
    html = UI_SETTINGS_HTML
    for f in SETTINGS_FIELDS:
        cid = settings_control_id(f)
        assert html.count(f'id="{cid}"') == 1, cid
        assert f'id="{cid}" data-key="{f["key"]}" data-scope="{f["scope"]}"' in html, cid
        assert f'data-widget="{f["widget"]}"' in html
        assert f'id="{cid}Hint"' in html, cid
        if f["widget"] in ("segmented", "readonly"):
            assert f'id="{cid}Label"' in html, cid
        else:
            assert f'for="{cid}"' in html, cid
    switches = sum(1 for f in SETTINGS_FIELDS if f["widget"] == "switch")
    assert html.count('<span class="slider"></span>') >= switches
    ranges = sum(1 for f in SETTINGS_FIELDS if f["widget"] == "range")
    assert html.count('<output class="range-value"') == ranges
    restarts = sum(1 for f in SETTINGS_FIELDS if f.get("restart"))
    assert html.count('<span class="badge">restart</span>') >= restarts


def test_settings_view_structure():
    html = UI_SETTINGS_HTML
    assert html.startswith('<div id="settingsView" class="view hidden"')
    for sec in SETTINGS_SECTIONS:
        assert f'class="rail-item' in html
        assert f'data-section="{sec["key"]}"' in html
        assert html.count(f'<section class="settings-section') == len(SETTINGS_SECTIONS)
    # Admin sections are cosmetically gated, ordinary ones are not.
    for sec in SETTINGS_SECTIONS:
        tag = re.search(r'<section class="([^"]*)" data-section="%s"' % sec["key"], html)
        assert tag and (("admin-only" in tag.group(1)) == bool(sec["admin"])), sec["key"]
    # Exactly one section and one sub-tab per tabbed section ship visible.
    visible = re.findall(r'<section class="settings-section( admin-only)?" ', html)
    assert len(visible) == 1
    for section, tabs in SETTINGS_SUBTABS.items():
        assert f'data-subtabs="{section}"' in html
        for t in tabs:
            assert f'class="sub-panel' in html and f'data-sub="{t["key"]}"' in html
    for needed in ("settingsSearch", "settingsRail", "settingsTitle", "settingsDesc",
                   "settingsBar", "settingsDirty", "settingsApplyBtn", "settingsDiscardBtn"):
        assert f'id="{needed}"' in html, needed
    # No inline styles in the generated view: layout is the kit's job.
    assert 'style="' not in html


def test_every_field_lands_in_a_rendered_sub_tab():
    for f in SETTINGS_FIELDS:
        sub = settings_field_sub(f)
        if f["section"] in SETTINGS_SUBTABS:
            assert sub in [t["key"] for t in SETTINGS_SUBTABS[f["section"]]], f["key"]
        else:
            assert sub == "", f["key"]


def test_bespoke_panel_ids_survive_exactly_once():
    page = render_ui()
    for element in BESPOKE_IDS:
        assert page.count(f'id="{element}"') == 1, element


# ------------------------------------------------------------- page / JS --

def test_page_assembly_invariants():
    page = render_ui()
    assert "<!--SETTINGS_VIEW-->" not in page
    for tag in ("<style>", "</style>", "<script>", "</script>"):
        assert HTML_PAGE.count(tag) == 1, tag
    assert HTML_PAGE.index("<style>") < HTML_PAGE.index("<script>")
    assert len(re.findall(r"<div\b", page)) == len(re.findall(r"</div>", page))
    assert "\\\\n" not in page, "double-escaped newline renders a literal backslash-n"
    script = _script(page)
    defined = set(re.findall(r'\bid="([A-Za-z0-9_]+)"', page))
    refs = set(re.findall(r"getElementById\(['\"]([A-Za-z0-9_]+)['\"]\)", script))
    assert not (refs - defined), sorted(refs - defined)
    # The kit runs first, panels (which calls authBoot) last.
    assert script.index("function uiToast") < script.index("function renderMarkdown")
    assert script.index("function openSettings") < script.index("authBoot();")


def test_settings_entry_points_and_shims_exist():
    script = _script(render_ui())
    for fn in ("openSettings", "closeSettings", "toggleSettings", "loadSettings",
               "applySettings", "discardSettings", "settingsCanLeave", "loadConfig",
               "saveConfig", "applyTheme", "toggleTheme", "initTheme", "applyAppearancePrefs",
               "uiToast", "postJSON", "uiAct", "uiSeg", "wireSeg", "uiRangeFill", "uiDownload",
               "uiConfirm", "loadModels", "showAdminSection", "showAdminTab", "adminTabShown",
               "adminTabStop", "wireRestoreInput", "downloadBackup", "restoreBackup",
               "loadToolsCatalogue", "loadMcpStatus"):
        assert re.search(r"function\s+%s\s*\(" % fn, script), fn
    assert "var ADMIN_TABS = [" in script


def test_no_function_is_declared_twice():
    """One global namespace: a second declaration silently replaces the first,
    so a stale copy left in another script part wins without any error."""
    names = re.findall(r"^\s*(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(", _script(render_ui()),
                       re.M)
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, f"declared more than once: {dupes}"


def test_new_script_parts_keep_the_es5_dialect_and_use_toasts():
    for name, src in (("kit", UI_JS_KIT), ("admin", UI_JS_ADMIN),
                      ("settings", UI_JS_SETTINGS), ("models", UI_JS_MODELS)):
        code = re.sub(r"//[^\n]*", "", src)
        assert "=>" not in code, f"{name}: arrow function"
        assert "`" not in code, f"{name}: template literal"
        assert not re.search(r"^\s*(let|const|class)\s", code, re.M), f"{name}: ES6 declaration"
        assert "alert(" not in code, f"{name}: alert() instead of uiToast"
        assert not re.search(r"(?<![\w.])confirm\(", code), f"{name}: confirm() instead of uiConfirm"
        assert "innerHTML = text" not in code
        for bad in ("<script>", "</script>", "<style>", "</style>"):
            assert bad not in src, f"{name}: {bad} inside script text"


# ---------------------------------------------------------------- HTTP --

def _build_app():
    from fastapi.testclient import TestClient
    from local_llm.api import create_app
    from local_llm.core import ADAPTER_DIR
    from local_llm.database import Database
    from local_llm.model_server import ModelServerManager
    from local_llm.tools import ToolRegistry
    from local_llm.training import RetrainManager
    cfg = Config(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                 allow_test_user=False, node_token="secret-node-token",
                 skills_dir=tempfile.mkdtemp(prefix="llm-test-ui-settings-skills-"))
    db = Database(Path(tempfile.mkdtemp()) / "x.db")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)  # never started
    app = create_app(cfg, db, mm, RetrainManager(db, mm, cfg), ToolRegistry(cfg, db))
    app.state.model_manager = mm
    return TestClient(app)


def test_values_sent_with_their_schema_type_are_accepted_by_the_server():
    """Post every global setting back exactly as the panel would send it (its
    current value, coerced to the control's data-type). The server must take
    all of them: one refused here is a control that can never be saved."""
    with _build_app() as c:
        assert c.post("/api/auth/login",
                      json={"username": "admin", "password": "adminpw123"}).status_code == 200
        current = c.get("/api/config").json()["config"]
        cast = {"int": int, "float": float, "bool": bool, "str": str}
        body = {}
        for f in SETTINGS_FIELDS:
            if f["scope"] != "global":
                continue
            assert f["key"] in current, f"GET /api/config does not report {f['key']}"
            body[f["key"]] = cast[settings_field_type(f)](current[f["key"]])
        r = c.post("/api/config", json=body)
        assert r.status_code == 200, r.text
        assert r.json()["ignored"] == [], r.json()["ignored"]
        # Info fields read their value from the same config payload.
        for f in SETTINGS_FIELDS:
            if f["scope"] == "info":
                assert f["key"] in current, f"GET /api/config does not report {f['key']}"


def test_settings_schema_route_when_present():
    """GET /api/settings/schema (prefs module): fields for the role plus the
    values the panel loads. Skipped quietly on a build without the route."""
    with _build_app() as c:
        c.post("/api/auth/login", json={"username": "admin", "password": "adminpw123"})
        r = c.get("/api/settings/schema")
        if r.status_code == 404:
            return
        assert r.status_code == 200, r.text
        data = r.json()
        assert len(data["fields"]) == len(SETTINGS_FIELDS)
        values = data.get("values") or {}
        for f in SETTINGS_FIELDS:
            if f["scope"] in ("global", "info"):
                assert f["key"] in values, f["key"]
        assert set(USER_PREFS) <= set(data.get("preferences") or {})


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
