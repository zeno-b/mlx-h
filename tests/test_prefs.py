"""Tests for per-user preferences (local_llm/prefs.py) and the routes around them."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import local_llm.config as config_mod  # noqa: E402
config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-prefs-"))

from fastapi.testclient import TestClient  # noqa: E402

from local_llm.api import create_app  # noqa: E402
from local_llm.config import Config  # noqa: E402
from local_llm.core import ADAPTER_DIR, SENTINEL_LOCAL_USER  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.prefs import PreferenceStore, user_generation_prefs  # noqa: E402
from local_llm.settings_schema import default_user_prefs  # noqa: E402
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.training import RetrainManager  # noqa: E402


def build_app(**overrides):
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


def login(client, username, password):
    return client.post("/api/auth/login", json={"username": username, "password": password})


def _with_two_users(app):
    """An admin client plus a second, non-admin user created through the API."""
    admin = TestClient(app)
    admin.__enter__()
    assert login(admin, "admin", "adminpw123").status_code == 200
    r = admin.post("/api/users", json={"username": "bob", "password": "bobpw12345",
                                       "role": "user"})
    assert r.status_code in (200, 201), r.text
    bob = TestClient(app)
    bob.__enter__()
    assert login(bob, "bob", "bobpw12345").status_code == 200
    return admin, bob


def _close(*clients):
    for c in clients:
        c.__exit__(None, None, None)


def test_preferences_start_at_the_defaults():
    app, _, _ = build_app()
    with TestClient(app) as c:
        assert login(c, "admin", "adminpw123").status_code == 200
        body = c.get("/api/me/preferences").json()
    assert body["preferences"] == default_user_prefs(), body
    assert body["defaults"] == default_user_prefs()


def test_unauthenticated_callers_get_401():
    app, _, _ = build_app()
    with TestClient(app) as c:
        assert c.get("/api/me/preferences").status_code == 401
        assert c.post("/api/me/preferences", json={"theme": "dark"}).status_code == 401
        assert c.get("/api/settings/schema").status_code == 401


def test_one_users_preferences_are_invisible_and_untouchable_to_another():
    app, _, db = build_app()
    admin, bob = _with_two_users(app)
    try:
        r = admin.post("/api/me/preferences", json={"theme": "dark", "temperature": 0.2})
        assert r.status_code == 200, r.text
        assert sorted(r.json()["changed"]) == ["temperature", "theme"]

        # Bob sees only defaults, and his own write does not touch the admin's.
        assert bob.get("/api/me/preferences").json()["preferences"]["theme"] == "system"
        assert bob.post("/api/me/preferences", json={"theme": "light"}).status_code == 200
        # A body cannot name another user: user_id is just an unknown key.
        r = bob.post("/api/me/preferences", json={"user_id": "admin", "theme": "light"})
        assert "user_id" in r.json()["ignored"]

        mine = admin.get("/api/me/preferences").json()["preferences"]
        assert mine["theme"] == "dark" and mine["temperature"] == 0.2, mine
        assert bob.get("/api/me/preferences").json()["preferences"]["theme"] == "light"
        assert bob.get("/api/settings/schema").json()["preferences"]["temperature"] is None
    finally:
        _close(admin, bob)
    rows = db.execute("SELECT DISTINCT user_id FROM user_preferences").fetchall()
    assert len(rows) == 2, [dict(r) for r in rows]


def test_validation_clamps_ignores_and_null_resets():
    app, _, db = build_app()
    with TestClient(app) as c:
        assert login(c, "admin", "adminpw123").status_code == 200
        r = c.post("/api/me/preferences", json={
            "temperature": 9, "max_tokens": 1, "theme": "neon", "code_wrap": "yes",
            "nonsense": 1, "density": "compact"}).json()
        prefs = r["preferences"]
        assert prefs["temperature"] == 2.0 and prefs["max_tokens"] == 16, prefs
        assert prefs["density"] == "compact"
        assert sorted(r["ignored"]) == ["code_wrap", "nonsense", "theme"], r
        assert prefs["theme"] == "system" and prefs["code_wrap"] is False

        # null puts a key back on the installation default, and storing the
        # default leaves no row behind (so a later default change reaches it).
        r = c.post("/api/me/preferences", json={"temperature": None,
                                                "density": "comfortable"}).json()
        assert r["preferences"]["temperature"] is None
        assert sorted(r["changed"]) == ["density", "temperature"], r
        # A no-op save reports nothing changed.
        assert c.post("/api/me/preferences", json={"max_tokens": 16}).json()["changed"] == []
        assert c.post("/api/me/preferences", json=[1, 2]).status_code in (400, 422)
    keys = {r["key"] for r in db.execute("SELECT key FROM user_preferences").fetchall()}
    assert keys == {"max_tokens"}, keys


def test_a_stored_value_is_revalidated_on_read():
    """A row written before a bound tightened (or for a removed key) never
    reaches a request at face value."""
    _, _, db = build_app()
    store = PreferenceStore(db)
    store.update("u1", {"temperature": 0.5})
    db.execute("UPDATE user_preferences SET value = '7.5' WHERE user_id = 'u1'")
    db.execute("INSERT INTO user_preferences (user_id, key, value) VALUES ('u1', 'gone', '1')")
    db.execute("INSERT INTO user_preferences (user_id, key, value) "
               "VALUES ('u1', 'theme', 'not json')")
    db.commit()
    prefs = store.get("u1")
    assert prefs["temperature"] == 2.0 and "gone" not in prefs and prefs["theme"] == "system"
    assert user_generation_prefs(db, "u1") == {"temperature": 2.0}
    assert user_generation_prefs(db, None) == {}


def test_auth_off_is_a_single_user_install():
    app, _, db = build_app(auth_enabled=False)
    with TestClient(app) as c:
        r = c.post("/api/me/preferences", json={"text_size": "large"})
        assert r.status_code == 200, r.text
        assert c.get("/api/me/preferences").json()["preferences"]["text_size"] == "large"
        schema = c.get("/api/settings/schema").json()
        assert schema["role"] == "admin" and schema["values"], schema.keys()
    assert PreferenceStore(db).get(SENTINEL_LOCAL_USER)["text_size"] == "large"


def test_schema_is_filtered_by_role_and_never_leaks_secrets():
    app, cfg, _ = build_app(oidc_client_secret="s3cr3t-value")
    admin, bob = _with_two_users(app)
    try:
        a = admin.get("/api/settings/schema").json()
        b = bob.get("/api/settings/schema").json()
    finally:
        _close(admin, bob)
    for payload in (a, b):
        assert {"sections", "fields", "user_prefs", "values", "preferences"} <= set(payload)
    admin_sections = {s["key"] for s in a["sections"]}
    user_sections = {s["key"] for s in b["sections"]}
    assert user_sections < admin_sections, (user_sections, admin_sections)
    assert not any(s["admin"] for s in b["sections"])
    assert all(f["scope"] != "global" for f in b["fields"]), [
        f["key"] for f in b["fields"] if f["scope"] == "global"]
    assert any(f["scope"] == "global" for f in a["fields"])
    # Admins get the live values, masked exactly like GET /api/config;
    # non-admins get no installation values at all.
    assert b["values"] == {}
    assert a["values"]["max_tokens"] == cfg.max_tokens
    assert "s3cr3t-value" not in json.dumps(a)
    for secret in cfg.SECRET_FIELDS:
        if secret in a["values"]:
            assert a["values"][secret] in ("", "***set***"), (secret, a["values"][secret])


def _capture_generation(app):
    seen: list[tuple] = []

    async def fake_run(message, history, conversation_id=None, max_tokens=None,
                       temperature=None, cancel=None):
        seen.append((max_tokens, temperature))
        yield {"type": "final", "answer": "ok", "steps": 1, "trace": [],
               "tools_used": [], "elapsed_ms": 1, "prompt_tokens": 1,
               "completion_tokens": 1, "truncated": False, "changed_files": [],
               "diff": ""}

    app.state.agent.run_iterating = fake_run
    return seen


def test_preferences_fill_generation_settings_the_request_omits():
    app, cfg, _ = build_app(knowledge_triage=False, max_tokens=1024, temperature=0.7)
    app.state.model_manager.status = "ready"
    seen = _capture_generation(app)
    admin, bob = _with_two_users(app)
    try:
        message = {"message": "explain what a mutex is", "agent": True}
        # No preference: the installation defaults.
        assert bob.post("/api/chat", json=message).status_code == 200
        assert seen[-1] == (1024, 0.7), seen

        assert bob.post("/api/me/preferences",
                        json={"max_tokens": 300, "temperature": 0.1}).status_code == 200
        assert bob.post("/api/chat", json=message).status_code == 200
        assert seen[-1] == (300, 0.1), seen

        # The request still wins over the preference, including temperature 0.
        r = bob.post("/api/chat", json=dict(message, max_tokens=64, temperature=0.0))
        assert r.status_code == 200
        assert seen[-1] == (64, 0.0), seen

        # The streaming route reads the same preference.
        with bob.stream("POST", "/api/chat/stream", json=message) as r:
            assert r.status_code == 200
            "".join(r.iter_text())
        assert seen[-1] == (300, 0.1), seen

        # Another user's preference never applies to this one.
        assert admin.post("/api/chat", json=message).status_code == 200
        assert seen[-1] == (1024, 0.7), seen
    finally:
        _close(admin, bob)
    assert cfg.max_tokens == 1024 and cfg.temperature == 0.7  # shared config untouched


def test_docs_scope_survives_a_restart():
    """Defect: POST /api/docs/scope applied rag_scope in memory only, so the
    scope silently reset to the whole knowledge base on the next start."""
    app, cfg, _ = build_app(rag_scope="old.md")
    with TestClient(app) as c:
        assert login(c, "admin", "adminpw123").status_code == 200
        r = c.post("/api/docs/scope", params={"paths": ""})
        assert r.status_code == 200, r.text
        assert r.json()["changed"] == ["rag_scope"] and r.json()["saved"] == ["rag_scope"]
    saved = json.loads(cfg.settings_path().read_text())
    assert saved["rag_scope"] == ""


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
