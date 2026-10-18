"""Tests for workspaces and projects (workspaces.py) and the two isolation bugs.

Every test that turns workspaces on patches WORKSPACE_DIR to a temp directory
first: the per-user roots are created under it, and a test must never create
users/<id> inside the checkout's real ./workspace.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_llm.config as config_mod  # noqa: E402

config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-workspaces-"))

from fastapi.testclient import TestClient  # noqa: E402

import local_llm.tools as tools_mod  # noqa: E402
import local_llm.workspaces as ws_mod  # noqa: E402
from local_llm.checkpoints import CheckpointStore  # noqa: E402
from local_llm.config import Config  # noqa: E402
from local_llm.core import ADAPTER_DIR  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.obslog import set_acting_user, set_acting_conversation  # noqa: E402
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.training import RetrainManager  # noqa: E402
from local_llm.turnctx import TurnContext, set_current_turn  # noqa: E402
from local_llm.workspaces import (  # noqa: E402
    context_injected_event, create_project, effective_rag_scope, list_projects,
    resolve_project, set_project_member, update_project, workspace_turn_scope,
)


@contextlib.contextmanager
def _workspace_dir():
    """Point every module's WORKSPACE_DIR binding at a fresh temp directory."""
    tmp = Path(tempfile.mkdtemp(prefix="llm-ws-root-")).resolve()
    saved = (tools_mod.WORKSPACE_DIR, ws_mod.WORKSPACE_DIR)
    tools_mod.WORKSPACE_DIR = tmp
    ws_mod.WORKSPACE_DIR = tmp
    try:
        yield tmp
    finally:
        tools_mod.WORKSPACE_DIR, ws_mod.WORKSPACE_DIR = saved


def _as(user_id, fn, *args, conversation=None, turn=None):
    """Run fn in a fresh context acting for user_id, like one request would."""
    def _inner():
        set_acting_user(user_id)
        set_acting_conversation(conversation)
        # Always set, so a turn another test leaked into this thread's context
        # cannot leak into this one.
        set_current_turn(turn)
        return fn(*args)
    return contextvars.copy_context().run(_inner)


def _db():
    return Database(Path(tempfile.mkdtemp()) / "ws.db")


def _users(db):
    alice = db.create_user("alice", role="user")
    bob = db.create_user("bob", role="user")
    admin = db.create_user("root", role="admin")
    return alice, bob, admin


def build_app(**overrides):
    base = dict(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                allow_test_user=False, node_token="secret-node-token",
                skills_dir=tempfile.mkdtemp(prefix="local-llm-test-skills-"))
    base.update(overrides)
    cfg = Config(**base)
    db = _db()
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)  # never started
    registry = ToolRegistry(cfg, db)
    app = create_app_ref()(cfg, db, mm, RetrainManager(db, mm, cfg), registry)
    app.state.model_manager = mm
    return app, cfg, db, registry


def create_app_ref():
    from local_llm.api import create_app
    return create_app


def login(client, username, password):
    r = client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return r


def _client_for(app, admin_client, username):
    r = admin_client.post("/api/users", json={"username": username,
                                              "password": f"{username}pass12",
                                              "role": "user"})
    assert r.status_code in (200, 201), r.text
    client = TestClient(app)
    login(client, username, f"{username}pass12")
    return client


# --------------------------------------------------------------------------- #
# Defaults: disabled == today                                                  #
# --------------------------------------------------------------------------- #
def test_disabled_is_todays_single_root_for_everyone():
    project = Path(tempfile.mkdtemp()).resolve()
    db = _db()
    alice, _, _ = _users(db)
    reg = ToolRegistry(Config(project_dir=str(project)), db)
    assert _as(None, reg._root) == project
    # A non-admin acting user changes nothing while the feature is off.
    assert _as(alice["id"], reg._root) == project
    out, err = _as(alice["id"], reg.call, "write_file",
                   {"path": "a.txt", "content": "x"}, conversation="c1")
    assert err is None, out
    assert (project / "a.txt").read_text() == "x"
    # Outside a turn the registry attributes are the old shared ones.
    assert "a.txt" in reg.changed_files
    reg.checkpoint_id = "cp-1"
    assert reg.checkpoint_id == "cp-1"
    reg.skills_used = ["one"]
    assert _as(alice["id"], lambda: reg.skills_used) == ["one"]
    db.close()


# --------------------------------------------------------------------------- #
# Per-user roots                                                               #
# --------------------------------------------------------------------------- #
def test_per_user_roots_confine_the_file_tools():
    with _workspace_dir() as wsdir:
        project = Path(tempfile.mkdtemp()).resolve()
        db = _db()
        alice, bob, admin = _users(db)
        cfg = Config(project_dir=str(project), workspaces_enabled=True)
        reg = ToolRegistry(cfg, db)
        alice_root = _as(alice["id"], reg._root)
        assert alice_root == (wsdir / "users" / alice["id"]).resolve()
        assert alice_root.is_dir(), "created on demand"
        # Admins keep the configured project (POST /api/config project_dir).
        assert _as(admin["id"], reg._root) == project
        assert _as("local", reg._root) == project      # auth off: implicit admin
        assert _as(None, reg._root) == project         # no request at all

        out, err = _as(alice["id"], reg.call, "write_file",
                       {"path": "secret.md", "content": "alice only"})
        assert err is None, out
        assert (alice_root / "secret.md").is_file()
        out, err = _as(bob["id"], reg.call, "read_file", {"path": "secret.md"})
        assert "alice only" not in out
        out, err = _as(bob["id"], reg.call, "read_file",
                       {"path": f"../{alice['id']}/secret.md"})
        assert err is not None and "escapes" in out, out
        db.close()


def test_admin_default_workspace_does_not_reach_into_user_roots():
    with _workspace_dir():
        db = _db()
        alice, _, admin = _users(db)
        reg = ToolRegistry(Config(workspaces_enabled=True), db)
        _as(alice["id"], reg.call, "write_file", {"path": "n.md", "content": "a"})
        out, err = _as(admin["id"], reg.call, "read_file",
                       {"path": f"users/{alice['id']}/n.md"})
        assert err is not None and "users/" in out, out
        db.close()


# --------------------------------------------------------------------------- #
# Projects                                                                     #
# --------------------------------------------------------------------------- #
def test_project_membership_and_sharing():
    with _workspace_dir() as wsdir:
        allowed = Path(tempfile.mkdtemp()).resolve()
        db = _db()
        alice, bob, admin = _users(db)
        cfg = Config(workspaces_enabled=True, workspace_roots=str(allowed))

        # A user's project lives under their own root; escapes are refused.
        thesis = create_project(db, cfg, alice, "Thesis")
        assert Path(thesis["root"]).resolve().parent == (wsdir / "users" / alice["id"]).resolve()
        for bad in ("../bob", "/etc", str(allowed)):
            try:
                create_project(db, cfg, alice, "Bad", root_path=bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"user project root {bad!r} was accepted")
        try:
            create_project(db, cfg, alice, "Mine", shared=True)
        except PermissionError:
            pass
        else:
            raise AssertionError("a non-admin created a shared project")

        # Private: bob is out until alice adds him, and out again after removal.
        try:
            resolve_project(db, bob, thesis["id"])
        except PermissionError:
            pass
        else:
            raise AssertionError("bob resolved alice's private project")
        set_project_member(db, alice, thesis["id"], bob["id"], "member")
        assert resolve_project(db, bob, thesis["id"])["access"] == "member"
        try:
            update_project(db, bob, thesis["id"], name="Mine now")
        except PermissionError:
            pass
        else:
            raise AssertionError("a plain member renamed the project")
        set_project_member(db, alice, thesis["id"], bob["id"], None)
        try:
            resolve_project(db, bob, thesis["id"])
        except PermissionError:
            pass
        else:
            raise AssertionError("removal did not revoke access")

        # Admin shared project: under WORKSPACE_ROOTS only, visible to everyone.
        try:
            create_project(db, cfg, admin, "Elsewhere",
                           root_path=str(Path(tempfile.mkdtemp())), shared=True)
        except ValueError:
            pass
        else:
            raise AssertionError("shared root outside WORKSPACE_ROOTS was accepted")
        # Default parents include the workspace dir, which contains users/<id>.
        try:
            create_project(db, Config(workspaces_enabled=True), admin, "Leak",
                           root_path=str(wsdir / "users" / alice["id"]), shared=True)
        except ValueError:
            pass
        else:
            raise AssertionError("a shared project was rooted in a user's workspace")
        team = create_project(db, cfg, admin, "Team", root_path="team", shared=True)
        assert Path(team["root"]).resolve() == (allowed / "team").resolve()
        assert {p["id"] for p in list_projects(db, bob)} == {team["id"]}
        assert {p["id"] for p in list_projects(db, alice)} == {team["id"], thesis["id"]}
        assert resolve_project(db, bob, team["id"])["access"] == "member"
        assert resolve_project(db, admin, thesis["id"])["access"] == "admin"
        try:
            resolve_project(db, bob, "nope")
        except LookupError:
            pass
        else:
            raise AssertionError("a missing project resolved")
        assert resolve_project(db, bob, "") is None
        db.close()


def test_a_turn_bound_to_a_project_is_confined_to_it_and_gets_its_context():
    with _workspace_dir():
        allowed = Path(tempfile.mkdtemp()).resolve()
        db = _db()
        alice, bob, admin = _users(db)
        cfg = Config(workspaces_enabled=True, workspace_roots=str(allowed))
        reg = ToolRegistry(cfg, db)
        team = create_project(db, cfg, admin, "Team", root_path="team", shared=True,
                              rag_scope="docs/a.md, docs/b.md")
        root = Path(team["root"])
        (root / "AGENTS.md").write_text("Always answer in haiku.", encoding="utf-8")
        outside = Path(tempfile.mkdtemp()).resolve()
        (outside / "loot.txt").write_text("secret", encoding="utf-8")
        (root / "link").symlink_to(outside)

        project = resolve_project(db, bob, team["id"])
        turn = TurnContext(user_id=bob["id"], role="user", project=project)
        assert _as(bob["id"], reg._root, turn=turn) == root.resolve()
        out, err = _as(bob["id"], reg.call, "write_file",
                       {"path": "notes.md", "content": "hi"}, turn=turn)
        assert err is None, out
        assert (root / "notes.md").read_text() == "hi"
        for escape in ("../escape.txt", "link/loot.txt", "/etc/passwd"):
            out, err = _as(bob["id"], reg.call, "read_file", {"path": escape}, turn=turn)
            assert "secret" not in out and "root:" not in out, (escape, out)
        out, err = _as(bob["id"], reg.call, "write_file",
                       {"path": "../escape.txt", "content": "x"}, turn=turn)
        assert err is not None and not (root.parent / "escape.txt").exists()

        # Context injection: the project's AGENTS.md, and an event to say so.
        assert "haiku" in _as(bob["id"], reg.project_context, turn=turn)
        event = _as(bob["id"], context_injected_event, reg, turn=turn)
        assert event and event["type"] == "context_injected", event
        assert event["files"] == ["AGENTS.md"] and event["project"] == "Team", event
        assert event["chars"] > 0
        assert _as(bob["id"], effective_rag_scope, cfg, turn=turn) == "docs/a.md,docs/b.md"
        assert effective_rag_scope(Config(rag_scope="x.md")) == "x.md"
        # Without a project, no injected event for an empty root.
        assert _as(bob["id"], context_injected_event, reg) is None
        db.close()


# --------------------------------------------------------------------------- #
# Bug 1: the execute_code bridge lost the acting user                          #
# --------------------------------------------------------------------------- #
def test_execute_code_bridge_keeps_the_acting_user_scope():
    """The bridge serves the script from plain threads, which do not inherit
    contextvars: get_acting_user() was None there, every scoped store read
    unscoped, and a script calling tools.recall_memory() saw every user's
    notes. The bridge now re-binds the caller's user, conversation and root."""
    project = Path(tempfile.mkdtemp()).resolve()
    db = _db()
    alice, bob, _ = _users(db)
    db.remember("alice-key", "alice private note", None, user_id=alice["id"])
    db.remember("bob-key", "bob private note", None, user_id=bob["id"])
    reg = ToolRegistry(Config(project_dir=str(project), allow_python=True,
                              tool_timeout=60), db)
    code = "print(tools.recall_memory(query=''))\n"
    out, err = _as(alice["id"], reg.call, "execute_code", {"code": code}, "conv-a")
    assert err is None, out
    assert "alice private note" in out, out
    assert "bob private note" not in out, out
    # The bridged call is audited as alice's, in alice's conversation.
    bridged = [r for r in db.list_tool_calls(conversation_id="conv-a")
               if r["name"] == "recall_memory"]
    assert bridged and all(r.get("user_id") == alice["id"] for r in bridged), bridged
    db.close()


def test_execute_code_bridge_resolves_paths_in_the_callers_root():
    with _workspace_dir():
        db = _db()
        alice, _, _ = _users(db)
        reg = ToolRegistry(Config(workspaces_enabled=True, allow_python=True,
                                  tool_timeout=60), db)
        _as(alice["id"], reg.call, "write_file", {"path": "mine.txt", "content": "A1"})
        code = "print('got', tools.read_file(path='mine.txt').strip())\n"
        out, err = _as(alice["id"], reg.call, "execute_code", {"code": code})
        assert err is None and "got A1" in out, out
        db.close()


# --------------------------------------------------------------------------- #
# Bug 2: per-turn registry state shared across concurrent turns                #
# --------------------------------------------------------------------------- #
def test_two_interleaved_turns_keep_their_own_registry_state():
    project = Path(tempfile.mkdtemp()).resolve()
    db = _db()
    alice, bob, _ = _users(db)
    reg = ToolRegistry(Config(project_dir=str(project)), db)
    seen: dict[str, dict] = {}

    async def turn(user, name, gate_in, gate_out):
        set_acting_user(user["id"])
        set_current_turn(TurnContext(user_id=user["id"], conversation_id=f"conv-{name}"))
        # What the agent does at the start of a turn.
        reg.checkpoint_id = f"cp-{name}"
        reg.skills_used = []
        reg.documents_written = []
        gate_out.set()
        await gate_in.wait()                    # the other turn starts meanwhile
        out, err = await asyncio.to_thread(
            reg.call, "write_file", {"path": f"{name}.txt", "content": name},
            f"conv-{name}")
        assert err is None, out
        reg.documents_written.append(f"{name}.pdf")
        reg.skills_used.append(f"skill-{name}")
        await asyncio.sleep(0)
        seen[name] = {"checkpoint": reg.checkpoint_id,
                      "documents": list(reg.documents_written),
                      "skills": list(reg.skills_used),
                      "changed": set(reg.changed_files),
                      "conversation": reg.conversation_id}

    async def main():
        a_ready, b_ready = asyncio.Event(), asyncio.Event()
        await asyncio.gather(turn(alice, "a", b_ready, a_ready),
                             turn(bob, "b", a_ready, b_ready))

    asyncio.run(main())
    for name in ("a", "b"):
        assert seen[name] == {"checkpoint": f"cp-{name}",
                              "documents": [f"{name}.pdf"],
                              "skills": [f"skill-{name}"],
                              "changed": {f"{name}.txt"},
                              "conversation": f"conv-{name}"}, (name, seen[name])
    # The admin's session view of this root still lists both turns' edits.
    assert {"a.txt", "b.txt"} <= _as(None, lambda: set(reg.changed_files))
    db.close()


def test_a_child_registry_in_the_same_turn_does_not_wipe_the_parents_state():
    project = Path(tempfile.mkdtemp()).resolve()
    parent = ToolRegistry(Config(project_dir=str(project)), None)
    with workspace_turn_scope():
        parent.documents_written = ["report.pdf"]
        parent.skills_used = ["cv-format"]
        child = ToolRegistry(Config(project_dir=str(project)), None)
        child.documents_written = []             # the child agent's turn reset
        child.skills_used = []
        assert parent.documents_written == ["report.pdf"]
        assert parent.skills_used == ["cv-format"]
    # Leaving the scope returns to the shared default state.
    assert parent.documents_written == []


# --------------------------------------------------------------------------- #
# HTTP: projects, download confinement, checkpoints, browse                    #
# --------------------------------------------------------------------------- #
def test_projects_routes_and_download_confinement_across_users():
    with _workspace_dir():
        app, cfg, db, reg = build_app(workspaces_enabled=True, allow_shell=True)
        with TestClient(app) as admin:
            login(admin, "admin", "adminpw123")
            alice = _client_for(app, admin, "alice")
            bob = _client_for(app, admin, "bob")
            alice_id = db.get_user_by_username("alice")["id"]
            bob_id = db.get_user_by_username("bob")["id"]

            listing = alice.get("/api/projects").json()
            assert listing["enabled"] is True and listing["projects"] == []
            # Local exec with per-user roots: the caveat is loud, not hidden.
            assert "NOT a security boundary" in listing["warning"], listing

            made = alice.post("/api/projects", json={"name": "Thesis"})
            assert made.status_code == 200, made.text
            pid = made.json()["project"]["id"]
            proj_root = Path(made.json()["project"]["root"])
            (proj_root / "chapter.md").write_text("# one", encoding="utf-8")
            assert bob.post("/api/projects", json={"name": "S", "shared": True}).status_code == 403

            # Alice's default-root file: hers to download, invisible to bob.
            _as(alice_id, reg.call, "write_file", {"path": "report.md", "content": "A"})
            r = alice.get("/api/files/download", params={"path": "report.md"})
            assert r.status_code == 200 and r.content == b"A", r.text
            assert bob.get("/api/files/download",
                           params={"path": "report.md"}).status_code == 404
            assert bob.get("/api/files/download",
                           params={"path": f"../{alice_id}/report.md"}).status_code == 400

            # Project download: membership checked, not-yours looks like missing.
            q = {"path": "chapter.md", "project_id": pid}
            assert alice.get("/api/files/download", params=q).status_code == 200
            assert bob.get("/api/files/download", params=q).status_code == 404
            assert bob.get(f"/api/projects/{pid}/context").status_code == 404
            r = alice.post(f"/api/projects/{pid}/members",
                           json={"username": "bob", "role": "member"})
            assert r.status_code == 200, r.text
            assert [m["user_id"] for m in r.json()["members"]] == [bob_id]
            assert bob.get("/api/files/download", params=q).status_code == 200
            assert bob.post(f"/api/projects/{pid}", json={"name": "Bob's"}).status_code == 403
            assert bob.delete(f"/api/projects/{pid}").status_code == 403

            (proj_root / "AGENTS.md").write_text("Cite sources.", encoding="utf-8")
            ctxv = bob.get(f"/api/projects/{pid}/context").json()
            assert ctxv["files"] == ["AGENTS.md"] and "Cite sources." in ctxv["text"], ctxv

            assert alice.post(f"/api/projects/{pid}",
                              json={"name": "Thesis v2"}).json()["project"]["name"] == "Thesis v2"
            gone = alice.delete(f"/api/projects/{pid}")
            assert gone.status_code == 200 and gone.json()["files_kept"] is True
            assert proj_root.is_dir()
            assert bob.get("/api/files/download", params=q).status_code == 404
            assert {p["id"] for p in admin.get("/api/projects").json()["projects"]} == set()
        db.close()


def test_download_with_workspaces_off_is_todays_shared_root():
    project = Path(tempfile.mkdtemp()).resolve()
    app, cfg, db, reg = build_app(project_dir=str(project))
    reg.call("write_file", {"path": "shared.md", "content": "S"}, None)
    with TestClient(app) as admin:
        login(admin, "admin", "adminpw123")
        alice = _client_for(app, admin, "alice")
        r = alice.get("/api/files/download", params={"path": "shared.md"})
        assert r.status_code == 200 and r.content == b"S", r.text
        assert alice.get("/api/projects").json()["enabled"] is False
    db.close()


def test_checkpoints_are_listed_and_rolled_back_only_by_their_owner():
    with _workspace_dir():
        app, cfg, db, reg = build_app(workspaces_enabled=True)
        reg.checkpoints = CheckpointStore(Path(tempfile.mkdtemp()) / "cp")
        with TestClient(app) as admin:
            login(admin, "admin", "adminpw123")
            alice = _client_for(app, admin, "alice")
            bob = _client_for(app, admin, "bob")
            alice_id = db.get_user_by_username("alice")["id"]

            def _turn():
                with workspace_turn_scope():
                    reg.checkpoint_id = reg.checkpoints.new_id("ca")
                    reg.call("write_file", {"path": "keep.txt", "content": "ORIG"}, "ca")
                with workspace_turn_scope():
                    reg.checkpoint_id = reg.checkpoints.new_id("cb")
                    reg.call("write_file", {"path": "keep.txt", "content": "CHANGED"}, "cb")
                return reg._root()
            alice_root = _as(alice_id, _turn)
            mine = alice.get("/api/checkpoints").json()["checkpoints"]
            assert len(mine) == 2 and all(c["user_id"] == alice_id for c in mine), mine
            assert bob.get("/api/checkpoints").json()["checkpoints"] == []
            assert len(admin.get("/api/checkpoints").json()["checkpoints"]) == 2
            newest = mine[0]["id"]
            assert bob.post(f"/api/checkpoints/{newest}/rollback").status_code == 404
            done = alice.post(f"/api/checkpoints/{newest}/rollback").json()
            assert done["restored"] == ["keep.txt"], done
            assert (alice_root / "keep.txt").read_text() == "ORIG"
        db.close()


def test_browse_is_confined_to_workspace_roots_when_enabled():
    with _workspace_dir():
        allowed = Path(tempfile.mkdtemp()).resolve()
        (allowed / "repo").mkdir()
        app, cfg, db, reg = build_app(workspaces_enabled=True, workspace_roots=str(allowed))
        with TestClient(app) as admin:
            login(admin, "admin", "adminpw123")
            r = admin.get("/api/browse")
            assert r.status_code == 200 and r.json()["path"] == str(allowed), r.text
            assert [e["name"] for e in r.json()["entries"]] == ["repo"]
            assert r.json()["parent"] is None
            assert admin.get("/api/browse", params={"path": "/"}).status_code == 403
            assert admin.get("/api/browse",
                             params={"path": str(allowed / "repo")}).status_code == 200
        db.close()


# --------------------------------------------------------------------------- #
# Knowledge base: documents unique per owner, not per path                     #
# --------------------------------------------------------------------------- #
def test_two_owners_can_index_the_same_path_without_taking_over_or_leaking():
    db = _db()
    if not db.fts_enabled:
        return
    db.index_document("notes.md", "alpha zebra crossing for alice", "n", user_id="alice")
    db.index_document("notes.md", "bravo zebra stripes for bob", "n", user_id="bob")
    alice_hits = [h["chunk"] for h in db.search_documents("zebra", user_id="alice")]
    bob_hits = [h["chunk"] for h in db.search_documents("zebra", user_id="bob")]
    assert alice_hits == ["alpha zebra crossing for alice"], alice_hits
    assert bob_hits == ["bravo zebra stripes for bob"], bob_hits
    # Re-indexing replaces only the caller's copy.
    db.index_document("notes.md", "alpha zebra revised", "n", user_id="alice")
    assert [h["chunk"] for h in db.search_documents("zebra", user_id="bob")] == bob_hits
    assert db.document_stats(user_id="bob")["documents"] == 1
    assert db.document_stats()["documents"] == 2
    # Unscoped removal (admin / import removal) still takes every copy.
    db.remove_document("notes.md")
    assert db.document_stats()["documents"] == 0
    db.close()

    # The no-FTS5 fallback, with a chunk table built by hand the way the
    # selftest builds one (no owner column until first use).
    fb = _db()
    fb.search_mode = "fallback"
    fb.execute("CREATE TABLE IF NOT EXISTS doc_chunks_plain "
               "(id INTEGER PRIMARY KEY, path TEXT, chunk TEXT)")
    fb.index_document("notes.md", "alpha zebra crossing for alice", "n", user_id="alice")
    fb.index_document("notes.md", "bravo zebra stripes for bob", "n", user_id="bob")
    assert [h["chunk"] for h in fb.search_documents("zebra", user_id="alice")] == \
        ["alpha zebra crossing for alice"]
    assert len(fb.search_documents("zebra")) == 2
    fb.close()


def test_an_old_knowledge_base_is_migrated_to_per_owner_rows():
    import sqlite3
    path = Path(tempfile.mkdtemp()) / "old.db"
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE VIRTUAL TABLE doc_chunks USING fts5(path, chunk)")
    except sqlite3.OperationalError:
        conn.close()
        return                                   # no FTS5 in this build
    conn.execute("""CREATE TABLE documents (id INTEGER PRIMARY KEY, path TEXT UNIQUE,
                    title TEXT, chars INTEGER, chunks INTEGER,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("INSERT INTO documents (path, title, chars, chunks) "
                 "VALUES ('old.md', 'old', 20, 1)")
    conn.execute("INSERT INTO doc_chunks (path, chunk) VALUES ('old.md', 'legacy quokka text')")
    conn.commit()
    conn.close()
    db = Database(path)
    db.close()
    db = Database(path)                          # idempotent on a second start
    assert [h["chunk"] for h in db.search_documents("quokka", user_id="local")] == \
        ["legacy quokka text"]
    assert db.search_documents("quokka", user_id="someone-else") == []
    db.index_document("old.md", "a different quokka", "old", user_id="carol")
    assert len(db.search_documents("quokka")) == 2
    db.close()


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
