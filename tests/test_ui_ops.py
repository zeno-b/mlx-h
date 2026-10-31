"""Tests for the operational Settings panels (ui_script_ops.py) and uiChart.

The panels (Inference engine, Admission, Aliases, Benchmark, Methods and
autopilot, Pair a node, Remote access) are static markup plus one script part
that fills them from admin routes. What can break is mostly structural: an id
the script looks up that the markup lost, a route the script no longer calls,
a panel that moved out of its (admin-only) section, a loader the lazy-load
registry never runs, or a payload field the script reads that the server
renamed. The route tests pin the fields the script reads and that every
route behind these panels refuses a standard user.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_llm.config as config_mod  # noqa: E402

config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-ui-ops-"))

from local_llm.config import Config  # noqa: E402
from local_llm.ui import HTML_PAGE, render_ui  # noqa: E402
from local_llm.ui_script_charts import UI_JS_CHARTS  # noqa: E402
from local_llm.ui_script_ops import UI_JS_OPS  # noqa: E402
from local_llm.ui_settings import UI_SETTINGS_HTML  # noqa: E402
from local_llm.ui_styles import UI_CSS  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# data-panel -> (section, sub-tab or "" for a section without sub-tabs)
PANELS = {
    "engine": ("performance", ""),
    "admission": ("performance", ""),
    "aliases": ("models", "model"),
    "bench": ("models", "model"),
    "trainmethods": ("models", "training"),
    "pairing": ("admin", "cluster"),
    "remote": ("admin", "system"),
}
ROUTES = (
    "/api/backends", "/api/health", "/api/models", "/api/models/aliases",
    "/api/bench/results", "/api/bench/recommendation", "/api/bench/status", "/api/bench/run",
    "/api/train/methods", "/api/train/plan?method=", "/api/train/export?method=",
    "/api/cluster/nodes", "/api/cluster/pairing", "/api/cluster/paired", "/api/routing/events",
    "/api/config",
)
TAB_KEYS = ("performance", "models/model", "models/training", "admin/cluster", "admin/system")
ADMIN_ROUTES = ("/api/backends", "/api/models", "/api/bench/results", "/api/bench/recommendation",
                "/api/bench/status", "/api/train/methods", "/api/train/plan", "/api/cluster/nodes",
                "/api/cluster/paired", "/api/routing/events", "/api/config")


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


def _code(src: str) -> str:
    return re.sub(r"//[^\n]*", "", src)


# ---------------------------------------------------------------- markup --

def test_every_panel_renders_once_in_its_admin_only_tab():
    for name, (section, sub) in PANELS.items():
        assert UI_SETTINGS_HTML.count(f'data-panel="{name}"') == 1, name
        at = UI_SETTINGS_HTML.index(f'data-panel="{name}"')
        sec_at = UI_SETTINGS_HTML.rindex("<section ", 0, at)
        sec_tag = UI_SETTINGS_HTML[sec_at:UI_SETTINGS_HTML.index(">", sec_at)]
        assert f'data-section="{section}"' in sec_tag, (name, sec_tag)
        # Gating is cosmetic (the server enforces it), but it must be there.
        assert "admin-only" in sec_tag, (name, sec_tag)
        if sub:
            sub_at = UI_SETTINGS_HTML.rindex('class="sub-panel', sec_at, at)
            sub_tag = UI_SETTINGS_HTML[sub_at:UI_SETTINGS_HTML.index(">", sub_at)]
            assert f'data-sub="{sub}"' in sub_tag, (name, sub_tag)


def test_ids_the_script_uses_exist_exactly_once():
    page = render_ui()
    defined = re.findall(r'\bid="([A-Za-z0-9_]+)"', page)
    refs = set(re.findall(r"getElementById\(['\"]([A-Za-z0-9_]+)['\"]\)", UI_JS_OPS))
    assert len(refs) > 30, "the ops script looks up fewer elements than its panels have"
    missing = sorted(refs - set(defined))
    assert not missing, missing
    for ident in refs:
        assert defined.count(ident) == 1, ident
    # The bespoke ids the older loaders used survive the cluster rework.
    for ident in ("clusterNodes", "routingSummary", "routingEvents", "modelTable", "modelCustom"):
        assert page.count(f'id="{ident}"') == 1, ident


def test_every_onclick_in_the_panels_has_a_function():
    names = set(re.findall(r"^\s*(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(",
                           _script(render_ui()), re.M))
    calls = set(re.findall(r'onclick="(ops[A-Za-z]+)\(', UI_SETTINGS_HTML))
    assert calls, "no ops buttons found"
    assert not sorted(calls - names), sorted(calls - names)


def test_every_route_is_referenced():
    for route in ROUTES:
        assert route in UI_JS_OPS, route
    # A sweep stops the model, so the script must confirm and send confirm:true.
    assert re.search(r"uiConfirm\([^)]*sweep", UI_JS_OPS, re.S)
    assert "{ confirm: true }" in UI_JS_OPS


def test_no_inline_styles_and_kit_components_only():
    for name in PANELS:
        block = _panel(name)
        assert 'style="' not in block, name
    # Only chart geometry is set inline (a meter's width, a bar's --fill).
    code = _code(UI_JS_OPS)
    assert "style=" not in code
    assert set(re.findall(r"\.style\.(\w+)", code)) <= {"width", "setProperty"}
    assert "style=" not in _code(UI_JS_CHARTS)
    # The one new CSS for these panels lives in its marked block.
    block = UI_CSS[UI_CSS.index("panels-ops: charts */"):UI_CSS.index("panels-ops: charts end")]
    for cls in (".ops-meter", ".ops-code", ".ops-scroll", ".uichart"):
        assert cls in block, cls
    for token in ("--chart-1", "--chart-2", "--chart-3", "--chart-4"):
        assert f"var({token})" in block, token


def test_script_dialect_and_conventions():
    for name, src in (("ops", UI_JS_OPS), ("charts", UI_JS_CHARTS)):
        code = _code(src)
        assert "=>" not in code, f"{name}: arrow function"
        assert "`" not in code, f"{name}: template literal"
        assert not re.search(r"^\s*(let|const|class)\s", code, re.M), f"{name}: ES6 declaration"
        assert "alert(" not in code, name
        assert not re.search(r"(?<![\w.])confirm\(", code), f"{name}: confirm() instead of uiConfirm"
        assert not re.search(r"\.innerHTML\s*=", code), f"{name}: innerHTML"
        for bad in ("<script>", "</script>", "<style>", "</style>"):
            assert bad not in src, f"{name}: {bad}"
        assert "\\\\n" not in src, f"{name}: double-escaped newline"
        assert "—" not in src, f"{name}: em dash"
    # One global namespace: each part declares only its own prefix.
    mine = re.findall(r"^\s*(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(", UI_JS_OPS, re.M)
    assert mine and all(n.startswith("ops") for n in mine), mine
    charts = re.findall(r"^\s*(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(", UI_JS_CHARTS, re.M)
    assert charts and all(n.startswith("uiChart") or n == "uiSvg" for n in charts), charts
    names = re.findall(r"^\s*(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(", _script(render_ui()), re.M)
    assert not sorted({n for n in names if names.count(n) > 1})
    # uiChart's public API, which the dashboards panels call too.
    assert re.search(r"function uiChart\(host, spec\)", UI_JS_CHARTS)
    assert re.search(r"function uiChartPush\(list, value, cap\)", UI_JS_CHARTS)
    for kind in ('"bar"', '"sparkline"'):
        assert kind in UI_JS_CHARTS, kind


def test_parts_are_assembled_after_admin_and_registered_with_the_tab_registry():
    script = _script(HTML_PAGE)
    assert HTML_PAGE.count("<script>") == 1
    assert (script.index("var ADMIN_TABS = [") < script.index("function uiChart(")
            < script.index("function opsTabHook") < script.index("function openSettings"))
    # The hook runs as a top-level statement (it edits ADMIN_TABS at load).
    assert re.search(r"^\s*opsTabHook\(\);", script, re.M)
    hook = UI_JS_OPS[UI_JS_OPS.index("function opsTabHook"):UI_JS_OPS.index("opsTabHook();")]
    for key in TAB_KEYS:
        assert f'opsTabSet("{key}"' in hook, key
    # The switcher reload after a model switch keeps the catalogue columns.
    assert "loadModels = async function" in UI_JS_OPS


def test_bundle_and_selftest_know_both_modules():
    bundle = (REPO / "bundle.py").read_text()
    order = bundle[bundle.index("ORDER = ["):]
    selftest = (REPO / "local_llm" / "selftest.py").read_text()
    for mod in ("ui_script_charts", "ui_script_ops"):
        assert f'"{mod}"' in order[:order.index('"ui", ')], mod
        assert f'"{mod}"' in selftest, mod


def test_assembled_script_parses_when_a_js_engine_is_available():
    """Syntax-check the whole page script with JavaScriptCore (macOS) or node."""
    jsc = Path("/System/Library/Frameworks/JavaScriptCore.framework/Versions/Current/Helpers/jsc")
    node = shutil.which("node")
    if not jsc.exists() and not node:
        return  # no engine on this machine; the structural tests above still ran
    tmp = Path(tempfile.mkdtemp(prefix="llm-test-ui-ops-js-"))
    src = tmp / "page.js"
    src.write_text(_script(render_ui()))
    if jsc.exists():
        checker = tmp / "check.js"
        checker.write_text('try { checkSyntax("%s"); print("ok"); } catch (e) { print("ERR " + e); }' % src)
        out = subprocess.run([str(jsc), str(checker)], capture_output=True, text=True, timeout=60)
    else:
        out = subprocess.run([node, "--check", str(src)], capture_output=True, text=True, timeout=60)
        out.stdout = "ok" if out.returncode == 0 else "ERR " + out.stderr
    assert out.stdout.strip() == "ok", out.stdout + out.stderr


# ----------------------------------------------------------------- routes --

def _build_app():
    from local_llm.api import create_app
    from local_llm.core import ADAPTER_DIR
    from local_llm.database import Database
    from local_llm.model_server import ModelServerManager
    from local_llm.tools import ToolRegistry
    from local_llm.training import RetrainManager

    cfg = Config(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                 allow_test_user=False, node_token="secret-node-token",
                 skills_dir=tempfile.mkdtemp(prefix="llm-test-ui-ops-skills-"))
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)  # never started
    app = create_app(cfg, db, mm, RetrainManager(db, mm, cfg), ToolRegistry(cfg, db))
    app.state.model_manager = mm
    return app, cfg, db


def _login(app, username, password):
    from fastapi.testclient import TestClient
    client = TestClient(app)
    r = client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return client


def _alice(app, admin):
    r = admin.post("/api/users", json={"username": "alice", "password": "alicepw123", "role": "user"})
    assert r.status_code in (200, 201, 409), r.text
    return _login(app, "alice", "alicepw123")


def test_routes_return_the_fields_the_panels_read():
    app, _cfg, _db = _build_app()
    admin = _login(app, "admin", "adminpw123")

    backends = admin.get("/api/backends").json()
    assert {"backends", "configured", "current"} <= set(backends)
    for key in ("backend", "backend_label", "status", "decisions", "prompt_cache", "adapter_served",
                "request_model", "capabilities", "accepted_levers"):
        assert key in backends["current"], key
    for key in ("name", "label", "available", "reason", "capabilities", "current"):
        assert key in backends["backends"][0], key

    queue = admin.get("/api/health").json()["queue"]
    for key in ("running", "waiting", "limit", "max_waiting", "free_mb", "reserved_mb", "min_free_mb",
                "admitted_total", "rejected_total", "deferred_total", "wait_ms_p50", "wait_ms_p95",
                "memory_aware"):
        assert key in queue, key

    models = admin.get("/api/models").json()
    item = models["catalog"][0]
    for key in ("id", "cached", "current", "aliases", "meta", "fits_in_ram", "recommended"):
        assert key in item, key
    aliases = admin.get("/api/models/aliases").json()
    assert {"aliases", "current"} <= set(aliases)
    assert all({"id", "source"} <= set(v) for v in aliases["aliases"].values())

    assert {"runs", "latest"} <= set(admin.get("/api/bench/results").json())
    assert {"applied", "computed", "ram_gb"} <= set(admin.get("/api/bench/recommendation").json())
    status = admin.get("/api/bench/status").json()
    for key in ("running", "done", "total", "current", "error", "file"):
        assert key in status, key
    # Without confirm the sweep refuses, which is why the UI asks first.
    assert admin.post("/api/bench/run", json={}).status_code == 400

    methods = admin.get("/api/train/methods").json()
    assert methods["methods"] and {"name", "label", "available", "reason"} <= set(methods["methods"][0])
    plan = admin.get("/api/train/plan", params={"method": methods["methods"][0]["name"]}).json()
    for key in ("reason", "corpus", "budget_gb", "num_layers", "lora_rank", "batch_size"):
        assert key in plan, key

    nodes = admin.get("/api/cluster/nodes").json()
    node = nodes["nodes"][0]
    for key in ("name", "role", "state", "ram_gb", "free_ram_gb", "queue", "last_latency_ms", "model",
                "backend", "capabilities", "protocol", "shard_capable", "shard_group"):
        assert key in node, key
    pairing = admin.post("/api/cluster/pairing", json={}).json()
    assert pairing["code"] and pairing["expires_at"]
    # The script substitutes these two placeholders into the join command.
    assert "<this-url>" in pairing["join"] and "<code>" in pairing["join"]
    paired = admin.get("/api/cluster/paired").json()
    assert {"nodes", "pending_codes", "cluster_nodes_env"} <= set(paired)
    assert paired["pending_codes"] >= 1

    cfg = admin.get("/api/config").json()["config"]
    for key in ("web_host", "web_port", "public_base_url", "trusted_proxies", "auth_enabled",
                "auth_cookie_secure", "node_token", "node_e2e_key", "node_ca_file",
                "node_client_cert", "node_allow_plaintext", "cluster_nodes", "studio_node_url"):
        assert key in cfg, key
    # Secrets reach the remote-access card only as set / unset.
    assert cfg["node_token"] in ("", "***set***")


def test_panel_routes_refuse_a_standard_user():
    app, _cfg, _db = _build_app()
    admin = _login(app, "admin", "adminpw123")
    alice = _alice(app, admin)
    for route in ADMIN_ROUTES:
        assert alice.get(route).status_code == 403, route
    assert alice.post("/api/cluster/pairing", json={}).status_code == 403
    assert alice.post("/api/bench/run", json={"confirm": True}).status_code == 403
    assert alice.delete("/api/cluster/paired/anything").status_code == 403
    # Everyone sees only the slot counts of the queue.
    queue = alice.get("/api/health").json()["queue"]
    assert set(queue) == {"running", "waiting", "limit", "max_waiting"}


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
