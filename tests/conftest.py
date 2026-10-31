"""Session-wide isolation for pytest runs (a test module run directly is unaffected).

Two things every test module would otherwise have to remember, and the runs
that forgot did real damage:

1. Logs. obslog.configure_logging writes LOG_DIR/app.log, LOG_DIR defaults to
   the repository's logs/ folder, and create_app() configures logging, so the
   suite appended to the real logs/app.log (upgrade/ROUTING_AND_TOOLING.md P9
   counted 1298 auth.admin_bootstrapped events there). DATA_DIR is redirected
   by each module; LOG_DIR is redirected here, once, for the whole session:
   in the environment (Config.log_dir and obslog._log_dir read it, and child
   processes inherit it) and in every local_llm module that bound the path by
   star import (core, obslog, sysutil, model_server, tasks, training, ...).
   The package is imported here for that, which is what the first test
   module's own import would have done anyway.

   DATA_DIR had the same hole. Each test module points config.DATA_DIR at its
   own temp folder, but tools, agent, claude_import and others bound DATA_DIR
   (and paths built from it: IMPORTS_DIR, SFT_DIR, DB_PATH ...) by star import
   when the package loaded, so agent checkpoints, partial replies and imports
   still landed in the repository's data/ folder. Every such module-level
   path under the repository's data/ is rebased onto a session temp folder
   here too; a module that sets config.DATA_DIR afterwards still wins.

2. Model servers. A full run once left a real mlx_lm.server listening on port
   8090 for days. subprocess.Popen is replaced, for the whole process, by a
   subclass that refuses any command line that would start a model server
   (mlx_lm with server, the mlx-engine entry, an mlx.launch of mlx_lm.server)
   with RuntimeError("tests must not launch a real model server"). Process-
   wide rather than only as local_llm.model_server sees it, because
   subprocess.run, check_output and asyncio's subprocesses all construct
   subprocess.Popen, so one guard covers every launch path. A `--help` probe
   (sysutil.help_cmd sniffing mlx_lm.server's flags) prints and exits, so it
   passes. A test that mocks Popen itself replaces this class for its
   duration and keeps working. Every refusal is listed, with the test that
   caused it, in a "model-server guard" section of the terminal summary,
   since the code under test may swallow the error.

3. Approvals: APPROVAL_DEFAULT_MODE=off for the session (see below).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

_CT_LOG_DIR = Path(tempfile.mkdtemp(prefix="llm-test-logs-"))
os.environ["LOG_DIR"] = str(_CT_LOG_DIR)

# 3. Approvals. Since the "ask before every action" rule, a turn that belongs
#    to a user asks before every tool call unless that user chose otherwise
#    (APPROVAL_DEFAULT_MODE defaults to "all"). The suite predates the rule:
#    hundreds of tests run tools in user turns and assert today's results, so
#    the session runs with the old default ("off") and
#    tests/test_approval_default.py sets "all" explicitly where it tests the
#    rule itself. Set BEFORE local_llm is imported: Config reads it per
#    instance, but a module may build a Config at import time. A module run
#    directly (not under pytest) needs APPROVAL_DEFAULT_MODE=off in its
#    environment for the same reason. Assigned, not setdefault: a value
#    exported in the developer's shell must not change what the suite asserts.
os.environ["APPROVAL_DEFAULT_MODE"] = "off"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_llm.core as _ct_core  # noqa: E402  (imports the whole package)

_CT_REPO_LOG_DIR = Path(_ct_core.LOG_DIR)


def ct_redirect_logs() -> list[str]:
    """Point every loaded local_llm module's LOG_DIR at the session's temp folder."""
    patched = []
    for name, module in list(sys.modules.items()):
        if module is None or not (name == "local_llm" or name.startswith("local_llm.")):
            continue
        value = getattr(module, "LOG_DIR", None)
        if value is not None and Path(value) == _CT_REPO_LOG_DIR:
            module.LOG_DIR = _CT_LOG_DIR
            patched.append(name)
    return patched


ct_redirect_logs()

_CT_DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-data-"))
_CT_REPO_DATA_DIR = Path(_ct_core.DATA_DIR)


def ct_redirect_data() -> list[str]:
    """Rebase every loaded local_llm module's paths under the repo's data/ folder."""
    patched = []
    for name, module in list(sys.modules.items()):
        if module is None or not (name == "local_llm" or name.startswith("local_llm.")):
            continue
        for attr, value in list(vars(module).items()):
            if not attr.isupper() or not isinstance(value, Path):
                continue
            if value == _CT_REPO_DATA_DIR or _CT_REPO_DATA_DIR in value.parents:
                setattr(module, attr, _CT_DATA_DIR / value.relative_to(_CT_REPO_DATA_DIR))
                patched.append(f"{name}.{attr}")
    return patched


ct_redirect_data()


# --------------------------------------------------------------------------- #
# The model-server guard                                                       #
# --------------------------------------------------------------------------- #
_CT_REAL_POPEN = subprocess.Popen
_CT_TRIPS: list[dict] = []
_CT_CURRENT = {"test": ""}


def _ct_argv(args) -> list[str]:
    if isinstance(args, (str, bytes, os.PathLike)):
        return [os.fsdecode(args)]
    try:
        return [os.fsdecode(a) if isinstance(a, (bytes, os.PathLike)) else str(a)
                for a in args]
    except TypeError:
        return [str(args)]


def ct_launches_model_server(args) -> bool:
    """Whether a Popen command line would start a model server."""
    parts = _ct_argv(args)
    text = " ".join(parts)
    words = text.split()
    if "--help" in words or "-h" in words:
        return False
    if "local_llm.mlx_engine" in text or "mlx_engine_cli" in text:
        return True
    # The bundled build starts the engine as `python deploy.py mlx-engine ...`.
    if any(parts[i] == "mlx-engine" and parts[i - 1].endswith(".py")
           for i in range(1, len(parts))):
        return True
    return "mlx_lm" in text and "server" in text


class _CtGuardedPopen(_CT_REAL_POPEN):
    """subprocess.Popen, except that it will not start a model server."""

    def __init__(self, args, *rest, **kwargs):
        if ct_launches_model_server(args):
            _CT_TRIPS.append({"test": _CT_CURRENT["test"],
                              "command": " ".join(_ct_argv(args))[:300]})
            raise RuntimeError("tests must not launch a real model server")
        super().__init__(args, *rest, **kwargs)


subprocess.Popen = _CtGuardedPopen


def ct_guard_trips() -> list[dict]:
    """The launches refused so far: [{"test", "command"}]."""
    return list(_CT_TRIPS)


def pytest_collection_finish(session):
    # A module imported during collection that bound LOG_DIR or DATA_DIR some
    # other way.
    ct_redirect_logs()
    ct_redirect_data()


def pytest_runtest_logstart(nodeid, location):
    _CT_CURRENT["test"] = nodeid


def pytest_runtest_logfinish(nodeid, location):
    _CT_CURRENT["test"] = ""


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _CT_TRIPS:
        return
    terminalreporter.section("model-server guard")
    for trip in _CT_TRIPS:
        terminalreporter.write_line(f"{trip['test'] or '(outside a test)'}: {trip['command']}")
