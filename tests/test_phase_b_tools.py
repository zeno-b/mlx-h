"""Phase B tooling (upgrade/ROUTING_AND_TOOLING.md, P3/P4/P5/P8 and the router
hints of P1): the FILE_DENY_GLOBS deny-list, the kb_overview tool, route hints
for the lookups the router could not reach, create_document's outputs folder
and the composer-upload conversation link."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Never the repository's own logs/ or data/: a test run must leave app state
# exactly as it found it.
os.environ.setdefault("LOG_DIR", tempfile.mkdtemp(prefix="llm-test-phaseb-logs-"))
import local_llm.config as config_mod  # noqa: E402
config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-phaseb-data-"))

from fastapi.testclient import TestClient  # noqa: E402

import local_llm.attachments as att_mod  # noqa: E402
from local_llm.agent import Agent  # noqa: E402
from local_llm.api import create_app  # noqa: E402
from local_llm.auth import hash_password  # noqa: E402
from local_llm.config import (CAPABILITY_GROUPS, Config,  # noqa: E402
                              capabilities_for_role, tools_for_capabilities)
from local_llm.core import ADAPTER_DIR, SHARED_OWNER, estimate_tokens  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.llm import GenerationStats  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.obslog import get_acting_user, set_acting_user  # noqa: E402
from local_llm.policy import classify_tool_risk  # noqa: E402
from local_llm.settings_schema import SETTINGS_FIELDS  # noqa: E402
from local_llm.tools import ToolRegistry, file_denied  # noqa: E402
from local_llm.training import RetrainManager  # noqa: E402

SECRET = "SECRET_TOKEN=hunter2-not-for-models"
DEFAULT_GLOBS = Config().file_deny_globs


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _project() -> Path:
    """A project root shaped like this app's own checkout."""
    root = Path(tempfile.mkdtemp(prefix="llm-test-phaseb-proj-"))
    (root / ".env").write_text(SECRET + "\n")
    (root / "data").mkdir()
    (root / "data" / "feedback.db").write_bytes(b"SQLite format 3\x00" + SECRET.encode())
    (root / "logs").mkdir()
    (root / "logs" / "app.log").write_text(SECRET + "\n")
    (root / "keys").mkdir()
    (root / "keys" / "server.pem").write_text(SECRET + "\n")
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("def main():\n    return 'hello'\n")
    (root / "sub" / "data").mkdir(parents=True)
    (root / "sub" / "data" / "rows.csv").write_text("a,b\n1,2\n")
    (root / "notes.txt").write_text("plain notes\n")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]\n")
    return root


def _registry(db: Database | None = None, **overrides) -> tuple[ToolRegistry, Path]:
    root = Path(overrides.pop("project_dir", "") or _project())
    base = dict(project_dir=str(root), checkpoints_enabled=False,
                skills_dir=tempfile.mkdtemp(prefix="llm-test-phaseb-skills-"),
                log_dir=os.environ["LOG_DIR"])
    base.update(overrides)
    return ToolRegistry(Config(**base), db), root


def _db() -> Database:
    return Database(Path(tempfile.mkdtemp(prefix="llm-test-phaseb-db-")) / "kb.db")


def _protected(out: str, err: str | None, shown: str) -> None:
    assert err, out
    assert f"{shown} is protected (FILE_DENY_GLOBS)" in out, out
    assert "hunter2" not in out, out


# --------------------------------------------------------------------------- #
# P8: the deny-list
# --------------------------------------------------------------------------- #
def test_read_file_refuses_secrets_and_app_data():
    reg, root = _registry()
    out, err = reg.call("read_file", {"path": ".env"}, None)
    _protected(out, err, ".env")
    out, err = reg.call("read_file", {"path": "data/feedback.db"}, None)
    _protected(out, err, "data/feedback.db")
    for path, shown in ((".ENV", ".ENV"), ("./.env", ".env"), ("src/../.env", ".env"),
                        ("logs/app.log", "logs/app.log"), ("keys/server.pem", "keys/server.pem"),
                        ("/.env", ".env")):
        out, err = reg.call("read_file", {"path": path}, None)
        assert err and "is protected (FILE_DENY_GLOBS)" in out and "hunter2" not in out, (path, out)
    # A directory under an anchored pattern is refused as a whole.
    out, err = reg.call("list_files", {"path": "data"}, None)
    _protected(out, err, "data")
    out, err = reg.call("file_info", {"path": ".env"}, None)
    _protected(out, err, ".env")
    # Anchored means anchored: a nested data/ folder is ordinary project data.
    out, err = reg.call("read_file", {"path": "sub/data/rows.csv"}, None)
    assert err is None and out.strip(), (out, err)
    out, err = reg.call("read_file", {"path": "src/app.py"}, None)
    assert err is None and "hello" in out
    # The existing .git refusal keeps its own message.
    out, err = reg.call("read_file", {"path": ".git/config"}, None)
    assert err and "refusing to touch the .git directory" in out, out


def test_writes_to_protected_paths_are_refused_and_change_nothing():
    reg, root = _registry()
    out, err = reg.call("write_file", {"path": "logs/x", "content": "pwned"}, None)
    _protected(out, err, "logs/x")
    assert not (root / "logs" / "x").exists()
    out, err = reg.call("write_file", {"path": ".env.local", "content": "A=1"}, None)
    _protected(out, err, ".env.local")
    assert not (root / ".env.local").exists()
    out, err = reg.call("write_file", {"path": "adapters/latest/x.safetensors",
                                       "content": "x"}, None)
    _protected(out, err, "adapters/latest/x.safetensors")
    assert not (root / "adapters").exists()
    out, err = reg.call("edit_file", {"path": ".env", "find": "hunter2", "replace": "x"}, None)
    _protected(out, err, ".env")
    assert (root / ".env").read_text() == SECRET + "\n"
    out, err = reg.call("create_document", {"path": "data/report.pdf",
                                            "content": "# Report"}, None)
    _protected(out, err, "data/report.pdf")
    assert not (root / "data" / "report.pdf").exists()
    # Ordinary writes are untouched.
    out, err = reg.call("write_file", {"path": "src/new.py", "content": "x = 1\n"}, None)
    assert err is None and (root / "src" / "new.py").is_file(), out


def test_walks_never_list_or_read_protected_files():
    reg, root = _registry()
    flat, err = reg.call("list_files", {}, None)
    assert err is None
    names = {line.split(" ", 1)[0] for line in flat.splitlines()}
    assert {"src/", "notes.txt", "sub/"} <= names, flat
    assert not names & {".env", "data/", "logs/", ".git/"}, flat
    deep, _ = reg.call("list_files", {"recursive": "true"}, None)
    assert "src/app.py" in deep and "sub/data/rows.csv" in deep, deep
    for hidden in (".env", "data/feedback.db", "logs/app.log", "keys/server.pem"):
        assert hidden not in deep, (hidden, deep)
    # search_files reads every file it walks: a pattern aimed at the secret
    # must find nothing, not read .env sideways.
    out, err = reg.call("search_files", {"pattern": "hunter2"}, None)
    assert err is None and out == "No matches.", out
    # A symlink is judged by where it lands.
    link = root / "innocent.txt"
    try:
        link.symlink_to(root / ".env")
    except (OSError, NotImplementedError):
        return
    out, err = reg.call("read_file", {"path": "innocent.txt"}, None)
    _protected(out, err, ".env")
    out, _ = reg.call("search_files", {"pattern": "hunter2"}, None)
    assert out == "No matches.", out
    flat, _ = reg.call("list_files", {}, None)
    assert "innocent.txt" not in flat, flat


def test_index_docs_skips_protected_files():
    db = _db()
    reg, root = _registry(db)
    out, err = reg.call("index_docs", {"path": "."}, None)
    assert err is None, out
    paths = {r["path"] for r in db.document_stats()["items"]}
    assert "src/app.py" in paths and "notes.txt" in paths, paths
    assert not {p for p in paths if "hunter2" in p or p.startswith(("data/", "logs/", "keys/"))
                or p == ".env"}, paths
    assert not db.search_documents("hunter2"), "a protected file reached the index"
    db.close()


def test_at_reference_to_a_protected_file_is_refused():
    reg, root = _registry()
    block, notes = asyncio.run(reg.expand_references("what is in @.env and @notes.txt"))
    assert "hunter2" not in block, block
    assert "plain notes" in block, block
    assert any(".env is protected (FILE_DENY_GLOBS)" in n for n in notes), notes
    block, notes = asyncio.run(reg.expand_references("look at @data/feedback.db"))
    assert block == "" and any("protected" in n for n in notes), (block, notes)


def test_at_diff_leaves_protected_files_out():
    import shutil
    import subprocess
    if shutil.which("git") is None:
        return
    reg, root = _registry()
    git = ["git", "-C", str(root), "-c", "user.email=t@example.invalid", "-c", "user.name=t",
           "-c", "commit.gpgsign=false"]
    (root / ".gitignore").write_text("")
    (root / "data" / "settings.json").write_text('{"a": 1}\n')
    (root / "my notes.md").write_text("one\n")
    if subprocess.run(git + ["init", "-q"], capture_output=True).returncode != 0:
        return
    subprocess.run(git + ["add", "-f", "src/app.py", "data/settings.json", "my notes.md",
                          ".env"], check=True, capture_output=True)
    subprocess.run(git + ["commit", "-q", "-m", "x"], check=True, capture_output=True)
    (root / "src" / "app.py").write_text("def main():\n    return 'changed'\n")
    (root / "data" / "settings.json").write_text('{"a": "' + SECRET + '"}\n')
    (root / ".env").write_text(SECRET + "\nMORE=hunter2\n")
    (root / "my notes.md").write_text("two\n")
    diff = reg.git_diff()
    assert "changed" in diff and "my notes.md" in diff, diff
    assert "hunter2" not in diff and "settings.json" not in diff, diff
    assert "[2 protected file(s) left out of this diff (FILE_DENY_GLOBS)]" in diff, diff
    block, notes = asyncio.run(reg.expand_references("review @diff"))
    assert "changed" in block and "hunter2" not in block, block
    # Only protected changes: the note, not "no uncommitted changes".
    (root / "src" / "app.py").write_text("def main():\n    return 'hello'\n")
    (root / "my notes.md").write_text("one\n")
    assert reg.git_diff().startswith("[2 protected file(s)"), reg.git_diff()


def test_the_deny_list_is_gitignore_shaped_and_configurable():
    g = DEFAULT_GLOBS
    for rel in (".env", "sub/.env", ".env.production", "x.db", "a/b/c.DB", "a/k.pem",
                "data", "data/x/y.json", "logs/app.log", ".venv/bin/python",
                "secrets/token", "keys/id.key", "z.db-wal", ".git/HEAD"):
        assert file_denied(rel, g), rel
    for rel in ("notes.txt", ".envrc", "src/data/rows.csv", "database.py", "datafile.txt",
                "a/secrets/b", "outputs/report.pdf", "docs/logs.md", ""):
        assert not file_denied(rel, g), rel
    assert file_denied("deep/nested/x.cfg", "**/nested/**")
    assert file_denied("cfg/prod.yaml", "cfg/*.yaml") and not file_denied("a/cfg/prod.yaml",
                                                                         "cfg/*.yaml")
    assert file_denied("build/out/x", "out/") and file_denied("out", "out/")
    # Blank entries dropped, whitespace trimmed.
    assert Config(file_deny_globs=" .env , ,data/** ,").file_deny_globs == ".env,data/**"
    # An admin who empties the list gets the old behaviour, except .git.
    reg, root = _registry(file_deny_globs="")
    out, err = reg.call("read_file", {"path": ".env"}, None)
    assert err is None and "hunter2" in out
    out, err = reg.call("read_file", {"path": ".git/config"}, None)
    assert err and "refusing to touch the .git directory" in out
    # Restart-only, and documented as such in Settings.
    assert "file_deny_globs" not in Config.MUTABLE and "outputs_dir" not in Config.MUTABLE
    info = {f["key"]: f for f in SETTINGS_FIELDS if f["scope"] == "info"}
    assert info["file_deny_globs"]["env"] == "FILE_DENY_GLOBS"
    assert info["outputs_dir"]["env"] == "OUTPUTS_DIR"


# --------------------------------------------------------------------------- #
# P5: create_document's outputs folder, and the download route
# --------------------------------------------------------------------------- #
def test_create_document_puts_a_bare_name_in_the_outputs_folder():
    reg, root = _registry()
    out, err = reg.call("create_document", {"path": "report.pdf",
                                            "content": "# Report\n\nBody."}, None)
    assert err is None, out
    assert out.startswith("Created outputs/report.pdf: a "), out
    assert "\u2014" not in out
    assert (root / "outputs" / "report.pdf").read_bytes().startswith(b"%PDF")
    assert not (root / "report.pdf").exists()
    assert reg.documents_written == ["outputs/report.pdf"]
    out, err = reg.call("create_document", {"path": "report.pdf", "content": "# Again"}, None)
    assert err is None and out.startswith("Updated outputs/report.pdf"), out
    # A path with a folder in it is a choice, and is kept.
    out, err = reg.call("create_document", {"path": "docs/plan.docx", "content": "# Plan"}, None)
    assert err is None and (root / "docs" / "plan.docx").is_file(), out
    assert reg.documents_written[-1] == "docs/plan.docx"
    # The format check comes first: *.key is protected, but the honest answer
    # to "deck.key" is that this tool cannot write Keynote.
    out, err = reg.call("create_document", {"path": "deck.key", "content": "# Hi"}, None)
    assert err and ".pdf" in out and ".docx" in out and "protected" not in out, out
    # OUTPUTS_DIR empty keeps the old place: the root.
    reg2, root2 = _registry(outputs_dir="")
    out, err = reg2.call("create_document", {"path": "plain.pdf", "content": "# P"}, None)
    assert err is None and (root2 / "plain.pdf").is_file(), out
    reg3, root3 = _registry(outputs_dir="exports/generated")
    out, err = reg3.call("create_document", {"path": "x.xlsx",
                                             "content": "| a | b |\n|---|---|\n| 1 | 2 |"}, None)
    assert err is None and (root3 / "exports" / "generated" / "x.xlsx").is_file(), out


def test_outputs_dir_is_clamped_inside_the_root():
    assert Config().outputs_dir == "outputs"
    for bad in ("/tmp/out", "../elsewhere", "a/../../b", "~/out", "C:/out"):
        assert Config(outputs_dir=bad).outputs_dir == "outputs", bad
    assert Config(outputs_dir="./reports/").outputs_dir == "reports"
    assert Config(outputs_dir="reports\\pdf").outputs_dir == "reports/pdf"
    assert Config(outputs_dir=".").outputs_dir == ""
    assert Config(outputs_dir="").outputs_dir == ""


def _app(project: Path):
    cfg = Config(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                 allow_test_user=False, project_dir=str(project), checkpoints_enabled=False,
                 skills_dir=tempfile.mkdtemp(prefix="llm-test-phaseb-skills-"),
                 log_dir=os.environ["LOG_DIR"], agent_enabled=True, knowledge_triage=False)
    db = _db()
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)   # never started
    registry = ToolRegistry(cfg, db)
    app = create_app(cfg, db, mm, RetrainManager(db, mm, cfg), registry)
    app.state.model_manager = mm
    mm.status = "ready"
    return app, cfg, db, registry


def test_download_serves_the_outputs_folder_and_refuses_protected_paths():
    root = _project()
    app, cfg, db, registry = _app(root)
    out, err = registry.call("create_document", {"path": "report.pdf",
                                                 "content": "# Report\n\nBody."}, None)
    assert err is None, out
    (root / "secrets").mkdir()
    (root / "secrets" / "leak.pdf").write_bytes(b"%PDF-1.4 " + SECRET.encode())
    (root / "data" / "dump.json").write_text('{"k": "' + SECRET + '"}')
    with TestClient(app) as c:
        assert c.post("/api/auth/login", json={"username": "admin",
                                               "password": "adminpw123"}).status_code == 200
        r = c.get("/api/files/download", params={"path": "outputs/report.pdf"})
        assert r.status_code == 200, r.text
        assert r.content.startswith(b"%PDF")
        assert 'filename="report.pdf"' in r.headers["content-disposition"]
        for path in ("secrets/leak.pdf", "data/dump.json", ".env", "DATA/dump.json"):
            r = c.get("/api/files/download", params={"path": path})
            assert r.status_code == 400, (path, r.status_code)
            assert "protected (FILE_DENY_GLOBS)" in r.json()["error"], (path, r.text)
            assert "hunter2" not in r.text
    db.close()


# --------------------------------------------------------------------------- #
# P1 router hints
# --------------------------------------------------------------------------- #
class _RouterClient:
    """Records the router prompt and replies with a canned decision."""

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list = []

    async def complete_with_stats(self, messages, *a, **kw):
        self.prompts.append(messages)
        return self.reply, GenerationStats()


NEW_ROUTABLE = ("search_docs", "recall_memory", "search_skills", "search_past_conversations")

# The budget, stated: each new hint is one line of at most 40 tokens, every
# route hint together (a fully configured registry: project, knowledge base,
# skills) stays under 340 tokens, and the whole router system prompt under 800
# (measured at 615 when this was written; 64 reply tokens go on top). The router
# is a model call on every unclassified turn of a 4096-token, ~15 tok/s model.
HINT_MAX_TOKENS = 40
MENU_MAX_TOKENS = 340
ROUTER_SYSTEM_MAX_TOKENS = 800


def test_lookups_the_router_could_not_reach_are_on_its_menu_and_it_stays_short():
    db = _db()
    reg, _ = _registry(db, skills_enabled=True)
    routable = {t.name: t for t in reg.routable()}
    for name in NEW_ROUTABLE:
        tool = routable.get(name)
        assert tool is not None, f"{name} is not on the router menu"
        hint = tool.route_hint
        assert "\n" not in hint and hint.startswith('{"action":"' + name + '"'), hint
        assert estimate_tokens(hint) <= HINT_MAX_TOKENS, (name, estimate_tokens(hint))
    menu = "\n".join(t.route_hint for t in routable.values())
    assert estimate_tokens(menu) <= MENU_MAX_TOKENS, estimate_tokens(menu)
    # The prompt the router actually sends.
    client = _RouterClient('{"action":"search_docs","query":"vacation policy"}')
    agent = Agent(reg.config, reg, client)
    decision = asyncio.run(agent.route("what do my documents say about the vacation policy", []))
    assert decision == {"action": "search_docs", "query": "vacation policy"}, decision
    system = client.prompts[0][0]["content"]
    for name in NEW_ROUTABLE:
        assert f'"action":"{name}"' in system, name
    assert estimate_tokens(system) <= ROUTER_SYSTEM_MAX_TOKENS, estimate_tokens(system)
    # recall_memory takes no required argument, so a bare action is a decision.
    client = _RouterClient('{"action":"recall_memory"}')
    decision = asyncio.run(Agent(reg.config, reg, client).route("what do you remember about me", []))
    assert decision == {"action": "recall_memory"}, decision
    db.close()


def test_routability_follows_registration():
    # No database: no knowledge base, no history; recall_memory is still there.
    reg, _ = _registry(None, skills_enabled=False)
    names = {t.name for t in reg.routable()}
    assert "recall_memory" in names
    assert not names & {"search_docs", "search_past_conversations", "search_skills",
                        "kb_overview"}, names


# --------------------------------------------------------------------------- #
# P3: kb_overview
# --------------------------------------------------------------------------- #
SKILL_MD = ("---\nname: devils-advocate\ndescription: Argue the other side of a plan "
            "before committing to it.\n---\n\n# Devil's advocate\n\nSteps...\n")
FOLDED_SKILL = ("---\nname: tdd\ndescription: >\n  Write the failing test first,\n"
                "  then the code.\n---\n\nBody\n")


def _kb(db: Database) -> None:
    alice = "alice"
    db.index_document("imports/repo/skills/devils-advocate/SKILL.md", SKILL_MD,
                      "SKILL.md", user_id=alice)
    db.index_document("imports/repo/skills/tdd/SKILL.md", FOLDED_SKILL, "SKILL.md",
                      user_id=alice)
    db.index_document("imports/repo/docs/guide.md", "Intro line\n\n# User Guide\n\nText " * 40,
                      "guide.md", user_id=alice)
    db.index_document("imports/repo/scripts/tool.py", "# a comment, not a title\nx = 1\n",
                      "tool.py", user_id=alice)
    db.index_document("notes/todo.txt", "buy milk", "todo.txt", user_id=alice)
    db.index_document("shared/handbook.md", "# Team Handbook\n\nRules.", "handbook.md",
                      user_id=SHARED_OWNER)
    db.index_document("bob/private.md", "# Bob's diary\n\nSecret.", "private.md",
                      user_id="bob")


def _as(user: str | None):
    class _Bound:
        def __enter__(self):
            self.saved = get_acting_user()
            set_acting_user(user)

        def __exit__(self, *exc):
            set_acting_user(self.saved)
    return _Bound()


def test_kb_overview_lists_totals_folders_and_titles_owner_scoped():
    db = _db()
    _kb(db)
    reg, _ = _registry(db)
    tool = reg.get("kb_overview")
    assert tool is not None and tool.risk == "read" and tool.capability == "knowledge"
    assert classify_tool_risk("kb_overview", {}, tool) == "read"
    assert not tool.required
    with _as("alice"):
        out, err = reg.call("kb_overview", {}, None)
    assert err is None, out
    assert out.startswith("Knowledge base: 6 documents"), out
    assert "Bob" not in out and "private.md" not in out, "another user's document leaked"
    assert "devils-advocate: Argue the other side of a plan before committing to it." in out, out
    assert "tdd: Write the failing test first, then the code." in out, out
    assert "guide.md: User Guide" in out, out
    assert "handbook.md: Team Handbook" in out, out
    # A "# ..." line in code is a comment, not a title.
    assert "tool.py (" in out and "a comment" not in out, out
    assert "Folders" in out and "imports/repo/skills" in out.replace("imports/repo/ (", ""), out
    # Newest first: the last document alice can see was indexed last.
    lines = [ln for ln in out.splitlines() if ln.startswith("- ")]
    assert "handbook.md" in lines[0], lines
    assert len(out) <= reg.config.tool_result_chars
    db.close()


def test_kb_overview_pages_within_tool_result_chars():
    db = _db()
    for i in range(60):
        db.index_document(f"library/vol{i % 3}/chapter-{i:02d}.md",
                          f"# Chapter {i:02d} of the long imported library\n\ntext",
                          f"chapter-{i:02d}.md", user_id="alice")
    reg, _ = _registry(db, tool_result_chars=600)
    seen: list[str] = []
    offset = 0
    with _as("alice"):
        for _round in range(60):
            out, err = reg.call("kb_overview", {"offset": offset}, None)
            assert err is None, out
            assert len(out) <= 600, len(out)
            page = [ln for ln in out.splitlines() if ln.startswith("- ")]
            assert page, out
            seen += page
            if "More: call kb_overview offset=" not in out:
                break
            offset = int(out.rsplit("offset=", 1)[1].split()[0].rstrip("."))
    assert len(seen) == 60 and len(set(seen)) == 60, (len(seen), len(set(seen)))
    # limit bounds a page even when more would fit.
    reg2, _ = _registry(db, tool_result_chars=20000)
    with _as("alice"):
        out, _ = reg2.call("kb_overview", {"limit": 5}, None)
    assert len([ln for ln in out.splitlines() if ln.startswith("- ")]) == 5, out
    assert "(1-5 of 60)" in out and "More: call kb_overview offset=5." in out, out
    db.close()


def test_kb_overview_folder_filter_resolves_relative_names():
    db = _db()
    _kb(db)
    reg, _ = _registry(db)
    with _as("alice"):
        out, err = reg.call("kb_overview", {"folder": "imports/repo/skills"}, None)
        assert err is None and "Documents in imports/repo/skills" in out, out
        assert "(1-2 of 2)" in out and "guide.md" not in out, out
        # A small model names the last folder only; it is found anyway.
        rel, _ = reg.call("kb_overview", {"folder": "skills"}, None)
        assert "Documents in imports/repo/skills" in rel, rel
        upper, _ = reg.call("kb_overview", {"folder": "IMPORTS/Repo/skills/"}, None)
        assert "(1-2 of 2)" in upper, upper
        missing, _ = reg.call("kb_overview", {"folder": "nope"}, None)
        assert "No documents under 'nope'." in missing and "Folders" in missing, missing
        # Bob's folder does not exist for alice.
        hidden, _ = reg.call("kb_overview", {"folder": "bob"}, None)
        assert "No documents under 'bob'." in hidden and "diary" not in hidden, hidden
        past, _ = reg.call("kb_overview", {"offset": 500}, None)
        assert "No documents at offset 500" in past, past
    db.close()


def test_kb_overview_on_an_empty_or_missing_knowledge_base():
    db = _db()
    reg, _ = _registry(db)
    out, err = reg.call("kb_overview", {}, None)
    assert err is None and "empty" in out, out
    db.close()
    reg2, _ = _registry(None)
    assert reg2.get("kb_overview") is None and reg2.get("search_docs") is None


def test_the_knowledge_capability_grants_the_knowledge_tools():
    assert CAPABILITY_GROUPS["knowledge"]["tools"] == ["search_docs", "kb_overview"]
    always = ["final_answer", "calculator", "current_time", "weather"]
    assert tools_for_capabilities(["knowledge"]) == always + ["search_docs", "kb_overview"]
    # A role granted "knowledge" also covers the narrower knowledge_search key.
    cfg = Config(role_capabilities='{"user": ["knowledge"]}')
    assert {"knowledge", "knowledge_search"} <= set(capabilities_for_role("user", cfg))
    cfg2 = Config(role_capabilities='{"user": ["knowledge_search"]}')
    assert "knowledge" not in capabilities_for_role("user", cfg2)
    # A profile limited to "knowledge" gets a registry with both tools.
    db = _db()
    reg, _ = _registry(db, agent_tools=",".join(tools_for_capabilities(["knowledge"])))
    assert {"search_docs", "kb_overview"} <= set(reg.names())
    db.close()


# --------------------------------------------------------------------------- #
# P4: composer uploads are filed under the conversation
# --------------------------------------------------------------------------- #
def test_link_attachments_sets_only_the_callers_unlinked_rows():
    db = _db()
    store = att_mod.AttachmentStore(db, None)
    mine = store.save("alice", b"hello there", "a.txt")
    other_conv = store.save("alice", b"other one", "b.txt", conversation_id="conv-old")
    bobs = store.save("bob", b"bob's file", "c.txt")
    assert mine["conversation_id"] is None
    n = att_mod.link_attachments_to_conversation(
        db, "alice", [mine["id"], other_conv["id"], bobs["id"], "att_bogus", ""], "conv-new")
    assert n == 1
    assert store.get(mine["id"], "alice")["conversation_id"] == "conv-new"
    # Never stolen from another conversation, never across users.
    assert store.get(other_conv["id"], "alice")["conversation_id"] == "conv-old"
    assert store.get(bobs["id"], "bob")["conversation_id"] is None
    # Idempotent, and a no-op without a conversation.
    assert att_mod.link_attachments_to_conversation(db, "alice", [mine["id"]], "conv-x") == 0
    assert store.get(mine["id"], "alice")["conversation_id"] == "conv-new"
    assert att_mod.link_attachments_to_conversation(db, "alice", [bobs["id"]], None) == 0
    # Deleting the conversation now takes the upload with it.
    assert att_mod.delete_conversation_attachments(db, "alice", "conv-new") == 1
    assert store.get(mine["id"], "alice") is None
    db.close()


def _final(answer: str = "ok") -> dict:
    return {"type": "final", "answer": answer, "steps": 1, "trace": [], "tools_used": [],
            "elapsed_ms": 1, "prompt_tokens": 1, "completion_tokens": 1, "truncated": False,
            "changed_files": [], "diff": ""}


def _cookie(c: TestClient, username: str, password: str) -> str:
    c.cookies.clear()
    r = c.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    value = "; ".join(f"{k}={v}" for k, v in c.cookies.items())
    c.cookies.clear()
    return value


def test_a_chat_turn_links_its_composer_uploads_to_the_conversation():
    app, cfg, db, registry = _app(_project())
    db.create_user("bob", password_hash=hash_password("bobpw12345"), role="user")

    async def fake_run(message, history, conversation_id=None, max_tokens=None,
                       temperature=None, cancel=None, attachments=None, **kw):
        yield _final("seen")

    app.state.agent.run_iterating = fake_run
    with TestClient(app) as c:
        admin = _cookie(c, "admin", "adminpw123")
        up = c.post("/api/attachments", content=b"# devils-advocate\n\nArgue.",
                    headers={"cookie": admin, "X-Filename": "devils-advocate.md"})
        assert up.status_code == 200 and up.json()["conversation_id"] is None, up.text
        att = up.json()["id"]
        r = c.post("/api/chat", json={"message": "add this to skills", "attachments": [att]},
                   headers={"cookie": admin})
        assert r.status_code == 200, r.text
        conv = r.json()["conversation_id"]
        listed = c.get("/api/attachments", params={"conversation_id": conv},
                       headers={"cookie": admin}).json()["attachments"]
        assert [a["id"] for a in listed] == [att], listed
        # Re-sending it in another conversation does not move it.
        r = c.post("/api/chat/stream", json={"message": "again", "attachments": [att]},
                   headers={"cookie": admin})
        assert r.status_code == 200, r.text
        row = c.get("/api/attachments", headers={"cookie": admin}).json()["attachments"]
        assert [a["conversation_id"] for a in row if a["id"] == att] == [conv], row
        # The streaming route links too.
        up2 = c.post("/api/attachments", content=b"second file",
                     headers={"cookie": admin, "X-Filename": "two.txt"}).json()["id"]
        r = c.post("/api/chat/stream", json={"message": "and this", "attachments": [up2],
                                             "conversation_id": conv},
                   headers={"cookie": admin})
        assert r.status_code == 200, r.text
        listed = c.get("/api/attachments", params={"conversation_id": conv},
                       headers={"cookie": admin}).json()["attachments"]
        assert {a["id"] for a in listed} == {att, up2}, listed
        # Someone else's id is still refused before anything is linked.
        bob = _cookie(c, "bob", "bobpw12345")
        r = c.post("/api/chat", json={"message": "x", "attachments": [att]},
                   headers={"cookie": bob})
        assert r.status_code == 404
    db.close()
