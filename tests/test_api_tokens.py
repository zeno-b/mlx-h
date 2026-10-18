"""Tests for personal API tokens, scopes, CLI login and the OIDC device flow."""
from __future__ import annotations

import asyncio
import base64
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_llm.config as config_mod  # noqa: E402

config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-api-tokens-"))

from fastapi.testclient import TestClient  # noqa: E402

import local_llm.api as api_mod  # noqa: E402
from local_llm.api import create_app  # noqa: E402
from local_llm.api_tokens import (  # noqa: E402
    looks_like_api_token, normalize_scopes, resolve_api_token, token_allows_route)
from local_llm.config import Config  # noqa: E402
from local_llm.core import ADAPTER_DIR  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.obslog import redact_text  # noqa: E402
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


def _admin_client(app):
    client = TestClient(app)
    r = client.post("/api/auth/login", json={"username": "admin", "password": "adminpw123"})
    assert r.status_code == 200, r.text
    return client


def _make_user(admin, name="bob", password="bobpw12345", role="user"):
    r = admin.post("/api/users", json={"username": name, "password": password, "role": role})
    assert r.status_code == 200, r.text
    return r.json()["user"]


def _user_client(app, name="bob", password="bobpw12345"):
    client = TestClient(app)
    r = client.post("/api/auth/login", json={"username": name, "password": password})
    assert r.status_code == 200, r.text
    return client


def _new_token(client, **body):
    r = client.post("/api/auth/tokens", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


def _who(client, headers):
    """The user /api/auth/me resolves for these headers, or None (it never 401s)."""
    r = client.get("/api/auth/me", headers=headers)
    assert r.status_code == 200, r.text
    return r.json().get("user") if r.json().get("authenticated") else None


# --------------------------------------------------------------------------- #
def test_token_lifecycle_create_use_list_revoke():
    app, cfg, db = build_app()
    admin = _admin_client(app)
    made = _new_token(admin, name="laptop")
    token = made["token"]
    assert looks_like_api_token(token) and token.startswith("llm_")
    row = made["api_token"]
    assert row["scopes"] == ["chat"] and row["name"] == "laptop" and row["active"]
    assert row["expires_at"] and "token_hash" not in row
    # Default lifetime is API_TOKEN_DEFAULT_TTL_DAYS (90).
    assert row["expires_at"][:4] in {str(time.gmtime().tm_year), str(time.gmtime().tm_year + 1)}
    # Only the hash is stored.
    stored = db.execute("SELECT token_hash FROM api_tokens").fetchone()["token_hash"]
    assert stored != token and len(stored) == 64
    listing = admin.get("/api/auth/tokens").json()["tokens"]
    assert [t["id"] for t in listing] == [row["id"]]
    assert token not in json.dumps(listing)

    bare = TestClient(app)
    me = bare.get("/api/auth/me", headers=_bearer(token))
    assert me.status_code == 200, me.text
    user = me.json().get("user") or me.json()
    assert user["username"] == "admin" and user["token_scopes"] == ["chat"]
    # last_used_at is touched (rate-limited, the first use always writes).
    assert db.execute("SELECT last_used_at FROM api_tokens").fetchone()["last_used_at"]

    r = admin.delete(f"/api/auth/tokens/{row['id']}")
    assert r.status_code == 200 and r.json()["revoked"] is True
    assert _who(bare, _bearer(token)) is None
    assert admin.get("/api/auth/tokens").json()["tokens"][0]["active"] is False


def test_x_api_key_header_accepts_only_api_tokens():
    app, cfg, db = build_app()
    admin = _admin_client(app)
    token = _new_token(admin)["token"]
    bare = TestClient(app)
    assert _who(bare, {"x-api-key": token}) is not None
    assert _who(bare, {"x-api-key": "sk-not-ours-1234567890"}) is None
    # A session token is a cookie/Bearer credential, never an x-api-key one.
    session = admin.cookies.get(cfg.auth_cookie_name)
    assert session
    assert _who(bare, {"x-api-key": session}) is None


def test_cookie_wins_over_bearer_and_api_key():
    app, cfg, db = build_app()
    admin = _admin_client(app)
    _make_user(admin)
    bob = _user_client(app)
    bob_token = _new_token(bob)["token"]
    r = admin.get("/api/auth/me", headers={**_bearer(bob_token), "x-api-key": bob_token})
    user = r.json().get("user") or r.json()
    assert user["username"] == "admin" and "token_scopes" not in user


def test_scopes_gate_routes_and_never_exceed_role():
    app, cfg, db = build_app()
    admin = _admin_client(app)
    _make_user(admin)
    bob = _user_client(app)
    bare = TestClient(app)

    chat = _new_token(admin, scopes=["chat"])["token"]
    assert bare.get("/api/conversations", headers=_bearer(chat)).status_code == 200
    assert bare.get("/api/users", headers=_bearer(chat)).status_code == 403
    assert bare.get("/api/health", headers=_bearer(chat)).status_code == 403
    assert bare.post("/api/auth/tokens", json={}, headers=_bearer(chat)).status_code == 403

    status = _new_token(admin, scopes=["status"])["token"]
    assert bare.get("/api/health", headers=_bearer(status)).status_code == 200
    assert bare.get("/api/conversations", headers=_bearer(status)).status_code == 403
    assert bare.post("/api/tasks/nope/cancel", headers=_bearer(status)).status_code == 403

    control = _new_token(admin, scopes=["control"])["token"]
    assert bare.get("/api/health", headers=_bearer(control)).status_code == 200  # implies status
    assert bare.post("/api/tasks/nope/cancel", headers=_bearer(control)).status_code != 403

    full = _new_token(admin, scopes=["admin"])["token"]
    assert bare.get("/api/users", headers=_bearer(full)).status_code == 200

    # admin scope on a non-admin is plain user rights; control is refused.
    bob_full = _new_token(bob, scopes="admin")["token"]
    assert bare.get("/api/conversations", headers=_bearer(bob_full)).status_code == 200
    assert bare.get("/api/users", headers=_bearer(bob_full)).status_code == 403
    r = bob.post("/api/auth/tokens", json={"scopes": ["control"]})
    assert r.status_code == 400 and "administrators" in r.json()["error"]
    assert bob.post("/api/auth/tokens", json={"scopes": ["root"]}).status_code == 400


def test_route_table_and_scope_normalisation():
    assert token_allows_route(["chat"], "POST", "/v1/messages")
    assert token_allows_route(["chat"], "POST", "/api/chat/stream")
    assert token_allows_route(["chat"], "GET", "/api/conversation/abc")
    assert not token_allows_route(["chat"], "POST", "/api/models/select")
    assert not token_allows_route(["chat"], "GET", "/api/chatty")
    assert token_allows_route(["status"], "HEAD", "/api/health")
    assert token_allows_route(["control"], "POST", "/api/models/select")
    assert not token_allows_route(["status"], "POST", "/api/models/select")
    assert token_allows_route(["admin"], "DELETE", "/api/users/x")
    assert normalize_scopes(None) == ["chat"]
    assert normalize_scopes("chat, status,chat") == ["chat", "status"]
    try:
        normalize_scopes(["control"], role="user")
        raise AssertionError("control must need admin")
    except ValueError:
        pass


def _fake_request(token, path, method="GET"):
    return SimpleNamespace(cookies={}, headers={"authorization": f"Bearer {token}"},
                           scope={"path": path}, method=method,
                           url=SimpleNamespace(path=path), state=SimpleNamespace())


def test_require_scope_dependency_factory():
    from fastapi import HTTPException
    app, cfg, db = build_app()
    auth = app.state.ctx.auth
    admin = _admin_client(app)
    _make_user(admin)
    bob = _user_client(app)
    chat = _new_token(admin, scopes=["chat"])["token"]
    control = _new_token(admin, scopes=["control"])["token"]
    bob_chat = _new_token(bob)["token"]

    def run(dep, token, path, method="GET"):
        try:
            return asyncio.run(dep(_fake_request(token, path, method)))
        except HTTPException as exc:
            return exc.status_code

    assert run(auth.require_scope("chat"), chat, "/v1/messages", "POST")["username"] == "admin"
    assert run(auth.require_scope("status"), chat, "/v1/messages", "POST") == 403
    assert run(auth.require_scope("status"), control, "/api/health")["username"] == "admin"
    assert run(auth.require_scope("control"), control, "/api/models/select", "POST")["id"]
    # A non-admin holding the chat scope cannot pass a control-only dependency.
    assert run(auth.require_scope("control"), bob_chat, "/v1/x", "POST") == 403
    try:
        auth.require_scope("nonsense")
        raise AssertionError("unknown scope must raise")
    except ValueError:
        pass


def test_revoked_on_role_change_disable_and_password_reset():
    app, cfg, db = build_app()
    admin = _admin_client(app)
    bob_id = _make_user(admin)["id"]
    bob = _user_client(app)
    bare = TestClient(app)

    # Promotion is a role change too (the users route only purges sessions on
    # demotion): the token was issued for a "user" and must not become admin.
    t1 = _new_token(bob)["token"]
    assert admin.post(f"/api/users/{bob_id}", json={"role": "admin"}).status_code == 200
    assert _who(bare, _bearer(t1)) is None
    reason = db.execute("SELECT revoked_reason FROM api_tokens").fetchone()["revoked_reason"]
    assert reason == "role changed"

    bob = _user_client(app)
    t2 = _new_token(bob)["token"]
    assert admin.post(f"/api/users/{bob_id}", json={"disabled": True}).status_code == 200
    assert _who(bare, _bearer(t2)) is None
    assert admin.post(f"/api/users/{bob_id}", json={"disabled": False}).status_code == 200
    # Re-enabling does not resurrect it: the route revoked it for good.
    assert _who(bare, _bearer(t2)) is None

    bob = _user_client(app)
    t3 = _new_token(bob)["token"]
    assert admin.post(f"/api/users/{bob_id}", json={"password": "newpw98765"}).status_code == 200
    assert _who(bare, _bearer(t3)) is None

    # Deleting the user kills its tokens as well.
    bob = _user_client(app, password="newpw98765")
    t4 = _new_token(bob)["token"]
    assert admin.delete(f"/api/users/{bob_id}").status_code == 200
    assert _who(bare, _bearer(t4)) is None


def test_role_change_outside_the_route_revokes_on_use():
    app, cfg, db = build_app()
    admin = _admin_client(app)
    bob_id = _make_user(admin)["id"]
    token = _new_token(_user_client(app))["token"]
    assert resolve_api_token(db, token)["id"] == bob_id
    db.update_user(bob_id, role="admin")        # e.g. Entra refreshing the role
    assert resolve_api_token(db, token) is None
    db.update_user(bob_id, role="user")
    assert resolve_api_token(db, token) is None   # stays revoked


def test_expiry_and_lifetime_ceiling():
    app, cfg, db = build_app(api_token_default_ttl_days=30)
    admin = _admin_client(app)
    assert admin.post("/api/auth/tokens", json={"ttl_days": 31}).status_code == 400
    assert admin.post("/api/auth/tokens", json={"ttl_days": 0}).status_code == 400
    made = _new_token(admin, ttl_days=7)
    bare = TestClient(app)
    assert _who(bare, _bearer(made["token"])) is not None
    db.execute("UPDATE api_tokens SET expires_at = '2000-01-01T00:00:00+00:00'")
    db.commit()
    assert _who(bare, _bearer(made["token"])) is None
    assert admin.get("/api/auth/tokens").json()["tokens"][0]["active"] is False

    app2, _, _ = build_app(api_token_default_ttl_days=0)
    forever = _new_token(_admin_client(app2))
    assert forever["api_token"]["expires_at"] is None


def test_admin_lists_all_and_revokes_any_user_only_own():
    app, cfg, db = build_app()
    admin = _admin_client(app)
    _make_user(admin)
    _make_user(admin, name="carol", password="carolpw12345")
    bob = _user_client(app)
    carol = _user_client(app, "carol", "carolpw12345")
    bob_row = _new_token(bob, name="bob-cli")["api_token"]
    _new_token(carol, name="carol-cli")
    assert len(bob.get("/api/auth/tokens?all=1").json()["tokens"]) == 1
    everyone = admin.get("/api/auth/tokens?all=1").json()
    assert everyone["all"] is True and {t["username"] for t in everyone["tokens"]} == {"bob", "carol"}
    assert carol.delete(f"/api/auth/tokens/{bob_row['id']}").status_code == 404
    r = admin.delete(f"/api/auth/tokens/{bob_row['id']}")
    assert r.status_code == 200 and r.json()["revoked"] is True
    assert bob.get("/api/auth/tokens").json()["tokens"][0]["revoked_reason"] == "revoked by admin"


def test_logout_with_api_token_revokes_it():
    app, cfg, db = build_app()
    token = _new_token(_admin_client(app))["token"]
    bare = TestClient(app)
    assert bare.post("/api/auth/logout", headers=_bearer(token)).status_code == 200
    assert _who(bare, _bearer(token)) is None


def test_api_tokens_are_redacted_in_logs():
    token = "llm_ab12cd34_" + "Zx9" * 13 + "Q"
    assert token not in redact_text(f"Authorization header x-api-key: {token}")
    assert token not in redact_text(f"ANTHROPIC_API_KEY_{token}")
    assert token not in redact_text(f'{{"key": "{token}"}}')
    assert "llm_tokens_total" in redact_text("metric llm_tokens_total=5")


def test_cli_login_issues_token_and_shares_web_throttle():
    app, cfg, db = build_app()
    client = TestClient(app)
    original = api_mod.LOGIN_MAX_FAILURES
    api_mod.LOGIN_MAX_FAILURES = 2
    try:
        bad = {"username": "admin", "password": "wrong"}
        assert client.post("/api/auth/cli/login", json=bad).status_code == 401
        assert client.post("/api/auth/cli/login", json=bad).status_code == 401
        r = client.post("/api/auth/cli/login", json=bad)
        assert r.status_code == 429 and r.headers.get("retry-after")
        # Same buckets as the web login: it is blocked too.
        web = client.post("/api/auth/login", json={"username": "admin", "password": "adminpw123"})
        assert web.status_code == 429
    finally:
        api_mod.LOGIN_MAX_FAILURES = original

    app, cfg, db = build_app()
    client = TestClient(app)
    r = client.post("/api/auth/cli/login",
                    json={"username": "admin", "password": "adminpw123", "name": "my laptop"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert looks_like_api_token(body["token"]) and body["api_token"]["name"] == "my laptop"
    assert body["user"]["username"] == "admin" and "password_hash" not in body["user"]
    # No cookie: a terminal gets a token, not a browser session.
    assert not client.cookies.get(cfg.auth_cookie_name)
    assert TestClient(app).get("/v1/models", headers=_bearer(body["token"])).status_code != 403


def test_auth_disabled_token_routes_explain_themselves():
    app, cfg, db = build_app(auth_enabled=False)
    client = TestClient(app)
    r = client.post("/api/auth/cli/login", json={"username": "x", "password": "y"})
    assert r.status_code == 400 and r.json()["auth_enabled"] is False
    assert client.get("/api/auth/tokens").json() == {"tokens": [], "auth_enabled": False}
    assert client.post("/api/auth/tokens", json={}).status_code == 400
    flags = client.get("/api/auth/config").json()
    assert flags["cli_login"] is False and flags["device_flow"] is False


# --------------------------------------------------------------------------- #
# OIDC device code, with Entra mocked                                         #
# --------------------------------------------------------------------------- #
_DISC = {"issuer": "https://login.example/tenant/v2.0",
         "token_endpoint": "https://login.example/tenant/oauth2/v2.0/token",
         "device_authorization_endpoint": "https://login.example/tenant/oauth2/v2.0/devicecode",
         "authorization_endpoint": "https://login.example/tenant/oauth2/v2.0/authorize"}


def _jwt(claims):
    def seg(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")
    return f"{seg({'alg': 'none'})}.{seg(claims)}.sig"


class _FakeResp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


class _FakeEntra:
    """Stands in for httpx.Client: scripted answers per endpoint."""

    def __init__(self, token_answers):
        self.token_answers = list(token_answers)
        self.calls = []
        self.started = 0

    def __call__(self, *a, **kw):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, data=None):
        self.calls.append((url, dict(data or {})))
        if url == _DISC["device_authorization_endpoint"]:
            self.started += 1
            return _FakeResp(200, {"device_code": f"DEV-CODE-{self.started}",
                                   "user_code": "ABCD-EFGH",
                                   "verification_uri": "https://microsoft.com/devicelogin",
                                   "expires_in": 900, "interval": 1,
                                   "message": "Go to https://microsoft.com/devicelogin"})
        return self.token_answers.pop(0)


def _device_app():
    app, cfg, db = build_app(oidc_enabled=True, oidc_tenant_id="tenant",
                             oidc_client_id="web-client", oidc_client_secret="s",
                             oidc_cli_client_id="cli-client",
                             oidc_admin_emails="boss@example.com")
    auth = app.state.ctx.auth
    auth.oidc_discovery = lambda: _DISC
    return app, cfg, db, auth


def _claims(**over):
    base = {"aud": "cli-client", "iss": _DISC["issuer"], "exp": int(time.time()) + 600,
            "oid": "oid-123", "email": "dev@example.com", "name": "Dev"}
    base.update(over)
    return base


def _make_due(auth):
    for entry in auth._device_pending.values():
        entry["next_poll"] = 0


def test_device_flow_not_configured_is_404_with_reason():
    app, cfg, db = build_app()
    client = TestClient(app)
    r = client.post("/api/auth/device/start")
    assert r.status_code == 404 and "OIDC_CLI_CLIENT_ID" in r.json()["error"]
    assert client.post("/api/auth/device/poll", json={"device_code": "x"}).status_code == 404


def test_device_flow_end_to_end_mocked():
    import httpx
    app, cfg, db, auth = _device_app()
    fake = _FakeEntra([
        _FakeResp(400, {"error": "authorization_pending"}),
        _FakeResp(200, {"id_token": _jwt(_claims()), "access_token": "x"}),
    ])
    original = httpx.Client
    httpx.Client = fake
    try:
        client = TestClient(app)
        assert client.get("/api/auth/config").json()["device_flow"] is True
        start = client.post("/api/auth/device/start")
        assert start.status_code == 200, start.text
        s = start.json()
        assert s["user_code"] == "ABCD-EFGH" and s["device_code"] == "DEV-CODE-1"
        assert fake.calls[0][1]["client_id"] == "cli-client"
        # Polling before the interval is up never reaches Entra.
        r = client.post("/api/auth/device/poll", json={"device_code": s["device_code"]})
        assert r.json()["status"] == "slow_down" and len(fake.calls) == 1
        _make_due(auth)
        r = client.post("/api/auth/device/poll", json={"device_code": s["device_code"]})
        assert r.status_code == 200 and r.json()["status"] == "pending"
        _make_due(auth)
        r = client.post("/api/auth/device/poll",
                        json={"device_code": s["device_code"], "name": "dev laptop"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "ok" and looks_like_api_token(body["token"])
        grant = fake.calls[-1][1]
        assert grant["grant_type"] == "urn:ietf:params:oauth:grant-type:device_code"
        me = TestClient(app).get("/api/auth/me", headers=_bearer(body["token"])).json()
        user = me.get("user") or me
        assert user["source"] == "oidc" and user["role"] == "user"
        # The code is spent: a replay is "expired" without calling Entra.
        before = len(fake.calls)
        r = client.post("/api/auth/device/poll", json={"device_code": s["device_code"]})
        assert r.status_code == 410 and len(fake.calls) == before
        # A device code this server never issued is refused the same way.
        r = client.post("/api/auth/device/poll", json={"device_code": "someone-elses"})
        assert r.status_code == 410
    finally:
        httpx.Client = original


def test_device_flow_rejects_wrong_audience_and_declines():
    import httpx
    app, cfg, db, auth = _device_app()
    fake = _FakeEntra([
        _FakeResp(200, {"id_token": _jwt(_claims(aud="web-client-other"))}),
        _FakeResp(400, {"error": "authorization_declined"}),
    ])
    original = httpx.Client
    httpx.Client = fake
    try:
        client = TestClient(app)
        code = client.post("/api/auth/device/start").json()["device_code"]
        _make_due(auth)
        r = client.post("/api/auth/device/poll", json={"device_code": code})
        assert r.status_code == 400 and "audience" in r.json()["error"]
        code = client.post("/api/auth/device/start").json()["device_code"]
        _make_due(auth)
        r = client.post("/api/auth/device/poll", json={"device_code": code})
        assert r.status_code == 403 and r.json()["status"] == "denied"
        assert db.execute("SELECT COUNT(*) AS c FROM api_tokens").fetchone()["c"] == 0
    finally:
        httpx.Client = original


def test_device_start_is_bounded_per_client():
    import httpx
    app, cfg, db, auth = _device_app()
    original = httpx.Client
    httpx.Client = _FakeEntra([])
    try:
        client = TestClient(app)
        codes = [client.post("/api/auth/device/start").status_code for _ in range(6)]
        assert codes[:5] == [200] * 5 and codes[5] == 429
    finally:
        httpx.Client = original


def _run_standalone():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"FAIL  {name}: {exc}")
            failed += 1
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
