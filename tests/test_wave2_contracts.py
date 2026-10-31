"""Cross-feature contracts that no single feature's tests own.

Each wave-2 feature (mlx-engine, the model catalogue, training methods) was
built and tested on its own; these checks pin the seams between them, which
broke silently before: a registered backend that GET /api/backends did not
list, DRAFT_MODEL=auto finding no catalogue, an alias reaching the model
switcher unresolved, and a new module missing from the bundle or the selftest.
"""
from __future__ import annotations

import ast
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_llm.config as config_mod  # noqa: E402
config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-wave2-"))

from local_llm.backends import _backends_resolve_draft, backend_names, get_backend  # noqa: E402
from local_llm.catalog import BUILTIN_MODEL_ALIASES, draft_for, resolve_model  # noqa: E402
from local_llm.config import ADAPTER_DIR, Config  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "local_llm"


def _build_app():
    from local_llm.api import create_app
    from local_llm.database import Database
    from local_llm.tools import ToolRegistry
    from local_llm.training import RetrainManager
    cfg = Config(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                 allow_test_user=False, node_token="secret-node-token",
                 skills_dir=tempfile.mkdtemp(prefix="local-llm-test-skills-"))
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)  # never started
    app = create_app(cfg, db, mm, RetrainManager(db, mm, cfg), ToolRegistry(cfg, db))
    app.state.model_manager = mm
    return app, cfg


def test_engine_registered_and_listed_by_route():
    from fastapi.testclient import TestClient
    assert "mlx-engine" in backend_names()
    assert get_backend("mlx-engine").request_model(None) == "default_model"
    app, _cfg = _build_app()
    with TestClient(app) as client:
        client.post("/api/auth/login", json={"username": "admin", "password": "adminpw123"})
        r = client.get("/api/backends")
        assert r.status_code == 200, r.text
        rows = {row["name"]: row for row in r.json()["backends"]}
        assert "mlx-engine" in rows, sorted(rows)
        assert rows["mlx-engine"]["current"] is False
        assert r.json()["configured"] == "mlx-lm"  # default unchanged


def test_draft_model_auto_uses_catalog():
    target = next(mid for mid in (resolve_model(a) for a in BUILTIN_MODEL_ALIASES)
                  if draft_for(mid))

    class _M:
        draft_model = "auto"
        model_id = target

    picked, note = _backends_resolve_draft(_M())
    assert picked == draft_for(target) and "catalogue" in note


def test_alias_resolved_by_model_switcher():
    from fastapi.testclient import TestClient
    app, cfg = _build_app()
    mm = app.state.model_manager
    swapped: list = []
    mm.swap = lambda model, adapter=None: swapped.append(model) or True  # never start a server
    # The select route also restarts on a background thread when the model
    # changed; unstubbed, that thread launched a real mlx_lm.server during the
    # next test module (the conftest guard now refuses it).
    mm.restart = lambda *a, **k: True
    with TestClient(app) as client:
        client.post("/api/auth/login", json={"username": "admin", "password": "adminpw123"})
        r = client.post("/api/models/select", json={"model": "coder"})
        assert r.status_code == 200, r.text
    assert swapped == [resolve_model("coder")] and swapped[0] != "coder"
    assert cfg.model == resolve_model("coder")


def _quoted_names(source: str, start: str, end_marker: str) -> set[str]:
    block = source[source.index(start):]
    return set(re.findall(r'"(\w+)"', block[:block.index(end_marker)]))


def test_every_module_in_bundle_order_and_selftest():
    modules = {p.stem for p in PKG.glob("*.py")} - {"__init__", "__main__"}
    order = _quoted_names((ROOT / "bundle.py").read_text(), "ORDER = [", "\n]")
    embed_only = {"tool_rpc_shim"}  # read as text at run time, never imported
    assert modules - order - embed_only == set()
    expected = _quoted_names((PKG / "selftest.py").read_text(), "_expected = ", "}")
    wave2 = {"admission", "appctx", "backends", "catalog", "mlx_engine", "nodesec", "prefs",
             "remote_access", "settings_schema", "sharding", "toolparse", "train_methods",
             "turnctx", "ui_script_admin", "ui_script_kit", "ui_script_settings", "ui_settings"}
    assert wave2 - expected == set()
    for name in wave2:
        tree = ast.parse((PKG / f"{name}.py").read_text())
        assert any(isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "__all__"
                                                     for t in n.targets)
                   for n in tree.body), f"{name} has no __all__"


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
