"""Item 15: the installer. envfile (the .env loader deploy.py runs first),
`deploy.py init` / `service` and the --doctor install checks (setup_cmds.py),
install.sh (syntax, and --dry-run with PATH shims: no network, nothing run)
and the HEAD-only Homebrew formula (ruby -c when ruby exists)."""
from __future__ import annotations

import io
import json
import os
import plistlib
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import local_llm.setup_cmds as setup  # noqa: E402
from local_llm.envfile import (DOTENV_LOADED_VAR, DOTENV_SWITCH_VAR, load_dotenv,  # noqa: E402
                               parse_dotenv)

INSTALL_SH = REPO / "install.sh"
FORMULA = REPO / "packaging" / "homebrew" / "local-llm.rb"
BASH = shutil.which("bash")
RUBY = shutil.which("ruby")


# --------------------------------------------------------------------------- #
# envfile
# --------------------------------------------------------------------------- #

def test_parse_dotenv_rules():
    text = "\n".join([
        "# a comment",
        "",
        "PLAIN=value",
        "export EXPORTED=yes-it-is",
        "SPACED =  padded  ",
        "INLINE=abc   # trailing comment",
        "HASH=abc#123",
        "SINGLE='literal $HOME \\n # not a comment'",
        'DOUBLE="tab\\tquote\\" dollar\\$ newline\\n"',
        "NOEXPAND=$HOME/${USER}",
        "EMPTY=",
        "BROKEN='unclosed",
        "1BAD=x",
        "noequals",
        "DUP=first",
        "DUP=second",
    ])
    values, problems = parse_dotenv(text)
    assert values["PLAIN"] == "value"
    assert values["EXPORTED"] == "yes-it-is"
    assert values["SPACED"] == "padded"
    assert values["INLINE"] == "abc"
    assert values["HASH"] == "abc#123"
    assert values["SINGLE"] == "literal $HOME \\n # not a comment"
    assert values["DOUBLE"] == 'tab\tquote" dollar$ newline\n'
    assert values["NOEXPAND"] == "$HOME/${USER}"
    assert values["EMPTY"] == ""
    assert values["DUP"] == "second"
    assert "BROKEN" not in values and "1BAD" not in values
    joined = " | ".join(problems)
    assert "unclosed" in joined and "not a variable name" in joined
    assert "no '='" in joined and "DUP is set again" in joined


def test_load_dotenv_never_overrides_and_can_be_switched_off(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_ID=from-file\nAPP_NAME='From File'\n", encoding="utf-8")
    environ = {"MODEL_ID": "from-shell"}
    applied = load_dotenv(env_file, environ=environ)
    assert applied == {"APP_NAME": "From File"}
    assert environ["MODEL_ID"] == "from-shell"
    assert environ[DOTENV_LOADED_VAR] == str(env_file)
    forced = {"MODEL_ID": "from-shell"}
    load_dotenv(env_file, override=True, environ=forced)
    assert forced["MODEL_ID"] == "from-file"
    off = {DOTENV_SWITCH_VAR: "0"}
    assert load_dotenv(env_file, environ=off) == {} and "APP_NAME" not in off
    missing = {}
    assert load_dotenv(tmp_path / "nope.env", environ=missing) == {} and missing == {}


@pytest.mark.parametrize("value", ["plain", "with space", "it's", 'say "hi" $5 \\ ok',
                                   "a # b", "two\nlines", "", "x=y:z/w@v"])
def test_env_line_round_trips_through_the_loader(value):
    line = setup.setup_env_line("KEY", value)
    assert parse_dotenv(line)[0] == {"KEY": value}


def test_deploy_py_and_module_load_env_before_core(tmp_path):
    """MODEL_ID only takes effect when .env is applied before core's import."""
    env_file = tmp_path / "custom.env"
    env_file.write_text("MODEL_ID=example/Fake-3B-4bit\nCONTEXT_SIZE=4096\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items()
           if k not in ("MODEL_ID", "CONTEXT_SIZE", DOTENV_LOADED_VAR, DOTENV_SWITCH_VAR)}
    env["LOCAL_LLM_ENV_FILE"] = str(env_file)
    env["CONTEXT_SIZE"] = "2048"  # the environment wins over the file
    for argv in ([sys.executable, str(REPO / "deploy.py"), "--print-config"],
                 [sys.executable, "-m", "local_llm", "--print-config"]):
        done = subprocess.run(argv, cwd=str(REPO), env=env, capture_output=True, text=True,
                              timeout=120)
        assert done.returncode == 0, done.stderr[-2000:]
        data = json.loads(done.stdout[done.stdout.index("{"):])
        assert data["model"] == "example/Fake-3B-4bit", argv
        assert int(data["context_size"]) == 2048, argv


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #

def test_recommend_starter_follows_tiers_and_keeps_headroom(tmp_path):
    rec8 = setup.recommend_starter(8, tmp_path)
    assert rec8["model_id"].endswith("Coder-3B-Instruct-4bit") and rec8["context_size"] == 4096
    assert rec8["admission_min_free_mb"] == 384 and rec8["max_concurrent_generations"] == 2
    assert setup.recommend_starter(16, tmp_path)["model_id"].endswith("7B-Instruct-4bit")
    rec64 = setup.recommend_starter(64, tmp_path)
    assert rec64["model_id"].endswith("32B-Instruct-4bit") and rec64["context_size"] == 32768
    assert rec64["max_concurrent_generations"] == 4 and rec64["admission_min_free_mb"] == 1536
    # 14GB: the tier says 7B, which does not fit the GPU share; step down.
    rec14 = setup.recommend_starter(14, tmp_path)
    assert rec14["model_id"].endswith("3B-Instruct-4bit") and "stepped down" in rec14["reason"]
    for rec in (rec8, rec64, rec14):
        assert setup.setup_model_fits(rec["model_id"], rec["ram_gb"]) in (True, None) or \
            rec is rec14


def test_recommend_starter_prefers_an_applied_benchmark_that_fits(tmp_path):
    big = "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"
    (tmp_path / "recommendation.json").write_text(json.dumps(
        {"model_id": big, "context_size": 12288, "ram_gb": 16.0}), encoding="utf-8")
    rec = setup.recommend_starter(16, tmp_path)
    assert rec["model_id"] == big and rec["context_size"] == 12288
    assert rec["source"] == "benchmark"
    # Measured on another machine size: ignored.
    assert setup.recommend_starter(64, tmp_path)["source"] == "tier"
    # A recommendation that would not fit with headroom: ignored too.
    (tmp_path / "recommendation.json").write_text(json.dumps(
        {"model_id": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit", "context_size": 8192,
         "ram_gb": 16.0}), encoding="utf-8")
    assert setup.recommend_starter(16, tmp_path)["source"] == "tier"


# --------------------------------------------------------------------------- #
# deploy.py init
# --------------------------------------------------------------------------- #

def _init(root, *argv, ram=16, **hooks):
    out = io.StringIO()
    code = setup.init_cli(["--skip-deps", "--non-interactive", *argv], root=root, ram_gb=ram,
                          out=out, **hooks)
    return code, out.getvalue()


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_init_writes_recommended_env_privately(tmp_path):
    code, out = _init(tmp_path)
    assert code == 0, out
    env_file = tmp_path / ".env"
    assert _mode(env_file) == 0o600
    values, problems = parse_dotenv(env_file.read_text())
    assert problems == []
    assert values["MODEL_ID"].endswith("7B-Instruct-4bit")
    assert values["CONTEXT_SIZE"] == "8192"
    assert values["MAX_CONCURRENT_GENERATIONS"] == "2"
    assert values["ADMISSION_MEMORY_AWARE"] == "1" and values["ADMISSION_MIN_FREE_MB"] == "768"
    assert values["WEB_HOST"] == "127.0.0.1" and values["AUTH_ENABLED"] == "0"
    assert "AUTH_ADMIN_PASSWORD" not in values and "NODE_TOKEN" not in values
    assert "Admin sign-in" not in out


def test_init_multi_user_primary_generates_secrets_once(tmp_path):
    code, out = _init(tmp_path, "--multi-user", "--cluster", "primary")
    assert code == 0, out
    values = parse_dotenv((tmp_path / ".env").read_text())[0]
    password = values["AUTH_ADMIN_PASSWORD"]
    assert len(password) >= 20 and out.count(password) == 1
    assert values["AUTH_ENABLED"] == "1" and values["WEB_HOST"] == "0.0.0.0"
    assert values["NODE_ROLE"] == "primary" and len(values["NODE_TOKEN"]) >= 40
    assert values["NODE_TOKEN"] not in out
    # A re-run with --force reuses both secrets instead of rotating them.
    code, out2 = _init(tmp_path, "--multi-user", "--cluster", "primary", "--force")
    again = parse_dotenv((tmp_path / ".env").read_text())[0]
    assert again["AUTH_ADMIN_PASSWORD"] == password and again["NODE_TOKEN"] == values["NODE_TOKEN"]
    assert "Admin sign-in" not in out2


def test_init_refuses_to_overwrite_without_force(tmp_path):
    old = ("# mine\nMODEL_ID=my/model\nAUTH_ENABLED=1\nAUTH_ADMIN_PASSWORD=keep-me\n"
           "OIDC_ENABLED=1   # custom\n")
    env_file = tmp_path / ".env"
    env_file.write_text(old, encoding="utf-8")
    code, out = _init(tmp_path)
    assert code == 0, out
    assert env_file.read_text() == old
    new_file = tmp_path / ".env.new"
    assert _mode(new_file) == 0o600
    new = parse_dotenv(new_file.read_text())[0]
    assert new["OIDC_ENABLED"] == "1"                 # carried over
    assert new["AUTH_ENABLED"] == "1"                 # a re-run never turns login off
    assert new["AUTH_ADMIN_PASSWORD"] == "keep-me"    # never rotated
    assert "~ MODEL_ID: 'my/model' ->" in out and "keep-me" not in out
    assert ".env.new" in out and "--force" in out

    code, out = _init(tmp_path, "--force")
    assert code == 0
    assert (tmp_path / ".env.bak").read_text() == old and _mode(tmp_path / ".env.bak") == 0o600
    assert parse_dotenv(env_file.read_text())[0]["MODEL_ID"].endswith("7B-Instruct-4bit")


def test_init_secondary_runs_the_join_flow_into_the_written_file(tmp_path):
    calls = []

    def join(url, code, path):
        calls.append((url, code, path))
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("NODE_TOKEN=issued-by-primary\n")
        return 0

    code, out = _init(tmp_path, "--cluster", "secondary", "--primary-url",
                      "http://mini.local:8000", "--code", "12-34", join=join)
    assert code == 0, out
    assert calls == [("http://mini.local:8000", "12-34", tmp_path / ".env")]
    values = parse_dotenv((tmp_path / ".env").read_text())[0]
    assert values["NODE_ROLE"] == "secondary" and values["AUTH_ENABLED"] == "1"
    assert values["NODE_TOKEN"] == "issued-by-primary"

    code, out = _init(tmp_path / "b", "--cluster", "secondary", join=join)
    assert code == 2 and not (tmp_path / "b" / ".env").exists()
    code, out = _init(tmp_path / "c", "--cluster", "secondary", "--primary-url", "http://x:1",
                      "--code", "1", join=lambda *a: 1)
    assert code == 1 and "Pairing failed" in out


def test_default_join_uses_cluster_join_with_env_file(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(setup, "cluster_cli", lambda argv: seen.append(argv) or 0)
    assert setup._setup_join("http://mini.local:8000", "77", tmp_path / ".env") == 0
    assert seen == [["join", "http://mini.local:8000", "--code", "77",
                     "--env-file", str(tmp_path / ".env")]]


def test_init_dry_run_writes_nothing_and_masks_secrets(tmp_path):
    code, out = _init(tmp_path, "--dry-run", "--multi-user")
    assert code == 0
    assert not (tmp_path / ".env").exists() and not (tmp_path / ".env.new").exists()
    assert "AUTH_ADMIN_PASSWORD=(secret)" in out and "MODEL_ID=" in out


def test_init_interactive_prompts(tmp_path):
    answers = iter(["", "y"])
    out = io.StringIO()
    args = setup._setup_init_parser().parse_args(["--skip-deps"])
    code = setup.run_init(args, root=tmp_path, ram_gb=16, ask=lambda _q: next(answers), out=out)
    assert code == 0
    values = parse_dotenv((tmp_path / ".env").read_text())[0]
    assert values["AUTH_ENABLED"] == "1" and values["MODEL_ID"].endswith("7B-Instruct-4bit")


def test_init_model_override_warns_when_it_does_not_fit(tmp_path):
    code, out = _init(tmp_path, "--model", "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit", ram=8)
    assert code == 0 and "does not fit" in out
    assert parse_dotenv((tmp_path / ".env").read_text())[0]["MODEL_ID"].endswith("32B-Instruct-4bit")


# --------------------------------------------------------------------------- #
# deploy.py service
# --------------------------------------------------------------------------- #

def test_plist_rendering(tmp_path):
    data = setup.render_service_plist(tmp_path, tmp_path / ".venv/bin/python", args=["--agent"])
    doc = plistlib.loads(data)
    assert doc["Label"] == "local.llm.server"
    assert doc["ProgramArguments"] == [str(tmp_path / ".venv/bin/python"),
                                       str(tmp_path / "deploy.py"), "--agent"]
    assert doc["WorkingDirectory"] == str(tmp_path)
    assert doc["KeepAlive"] == {"SuccessfulExit": False} and doc["RunAtLoad"] is True
    assert doc["StandardOutPath"] == str(tmp_path / "logs" / "service.out.log")
    assert doc["StandardErrorPath"] == str(tmp_path / "logs" / "service.err.log")
    assert "/opt/homebrew/bin" in doc["EnvironmentVariables"]["PATH"]
    assert not any("PASSWORD" in k or "TOKEN" in k for k in doc["EnvironmentVariables"])
    assert setup.service_plist_path(agents_dir=tmp_path).name == "local.llm.server.plist"


class _Done:
    def __init__(self, code=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = code, stdout, stderr


def test_service_install_status_uninstall_with_fake_launchctl(tmp_path):
    root, agents = tmp_path / "app", tmp_path / "LaunchAgents"
    calls = []

    def runner(cmd, **_kw):
        calls.append(cmd)
        if cmd[1] == "print":
            return _Done(0, "\tstate = running\n\tpid = 4242\n\tlast exit code = (never exited)\n")
        return _Done(0)

    out = io.StringIO()
    assert setup.service_cli(["install"], root=root, agents_dir=agents, runner=runner,
                             out=out) == 1
    assert calls == [] and "deploy.py init" in out.getvalue()

    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "python").write_text("")
    assert setup.service_cli(["install", "--", "--agent"], root=root, agents_dir=agents,
                             runner=runner, out=io.StringIO()) == 0
    plist = agents / "local.llm.server.plist"
    assert plistlib.loads(plist.read_bytes())["ProgramArguments"][-1] == "--agent"
    assert (root / "logs").is_dir()
    uid = os.getuid()
    assert calls == [["launchctl", "bootout", f"gui/{uid}/local.llm.server"],
                     ["launchctl", "bootstrap", f"gui/{uid}", str(plist)]]

    calls.clear()
    out = io.StringIO()
    assert setup.service_cli(["status"], root=root, agents_dir=agents, runner=runner,
                             out=out) == 0
    assert calls == [["launchctl", "print", f"gui/{uid}/local.llm.server"]]
    assert "state running, pid 4242" in out.getvalue()

    calls.clear()
    assert setup.service_cli(["uninstall"], root=root, agents_dir=agents, runner=runner,
                             out=io.StringIO()) == 0
    assert not plist.exists() and calls == [["launchctl", "bootout", f"gui/{uid}/local.llm.server"]]

    failing = lambda cmd, **_kw: _Done(5, "", "Bootstrap failed: 5")  # noqa: E731
    out = io.StringIO()
    assert setup.service_cli(["install"], root=root, agents_dir=agents, runner=failing,
                             out=out) == 1
    assert "Bootstrap failed" in out.getvalue()


def test_service_dry_run_changes_nothing(tmp_path):
    def runner(*_a, **_kw):
        raise AssertionError("launchctl must not run in a dry run")
    out = io.StringIO()
    assert setup.service_cli(["install", "--dry-run"], root=tmp_path, agents_dir=tmp_path / "LA",
                             runner=runner, out=out) == 0
    assert not (tmp_path / "LA").exists() and "launchctl bootstrap" in out.getvalue()


# --------------------------------------------------------------------------- #
# --doctor install checks
# --------------------------------------------------------------------------- #

def _healthy_facts(**over):
    facts = {
        "machine": "arm64", "translated": False, "python": (3, 12, 4), "in_venv": True,
        "venv_python": "/x/.venv/bin/python", "venv_exists": True, "venv_machine": "arm64",
        "venv_missing": [], "model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        "model_cached": True, "model_size_gb": 4.3, "model_min_ram_gb": 8, "ram_gb": 16.0,
        "disk_free_gb": 200.0, "disk_path": "/x/hub", "root_free_gb": 200.0,
        "env_path": "/x/.env", "env_text": "MODEL_ID=x\nAUTH_ENABLED=0\n", "env_mode": 0o600,
        "env_loaded": "/x/.env", "environ": {},
        "ports": {"web": (8000, False), "model": (8080, False)}, "service_installed": False,
    }
    facts.update(over)
    return facts


def _levels(rows, topic):
    return [(level, msg) for level, t, msg in rows if t == topic]


def test_doctor_checks_pass_on_a_healthy_install():
    rows = setup.evaluate_install_facts(_healthy_facts())
    assert {level for level, _t, _m in rows} <= {"ok", "info"}
    assert {t for _l, t, _m in rows} >= {"python", "venv", "model", "disk", "ram", ".env", "ports"}


def test_doctor_checks_flag_each_problem():
    ev = setup.evaluate_install_facts
    assert _levels(ev(_healthy_facts(machine="x86_64", translated=True)), "python")[0][0] == "fail"
    assert "Rosetta" in _levels(ev(_healthy_facts(machine="x86_64")), "python")[0][1]
    assert _levels(ev(_healthy_facts(python=(3, 9, 6))), "python")[0][0] == "fail"
    assert _levels(ev(_healthy_facts(venv_exists=False)), "venv")[0][0] == "warn"
    assert _levels(ev(_healthy_facts(venv_missing=None)), "venv")[0][0] == "fail"
    assert _levels(ev(_healthy_facts(venv_machine="x86_64")), "venv")[0][0] == "fail"
    assert "mlx_lm" in _levels(ev(_healthy_facts(venv_missing=["mlx_lm"])), "venv")[0][1]
    assert _levels(ev(_healthy_facts(model_cached=False)), "model")[0][0] == "warn"
    assert _levels(ev(_healthy_facts(disk_free_gb=1.0)), "disk")[0][0] == "fail"
    assert _levels(ev(_healthy_facts(disk_free_gb=8.0, model_cached=False,
                                     model_size_gb=18.4, model_min_ram_gb=32)), "disk")[0][0] \
        == "warn"
    assert _levels(ev(_healthy_facts(ram_gb=8.0, model_size_gb=18.4, model_min_ram_gb=32)),
                   "ram")[0][0] == "fail"
    assert _levels(ev(_healthy_facts(ram_gb=8.0, model_size_gb=5.0, model_min_ram_gb=8)),
                   "ram")[0][0] == "warn"
    env_rows = _levels(ev(_healthy_facts(
        env_text="AUTH_ENABLED=true\nAGENT_ENABLED=off\nAUTH_ADMIN_PASSWORD=s3cret\nBAD LINE\n",
        env_mode=0o644, environ={"AUTH_ENABLED": "1"})), ".env")
    joined = " | ".join(f"{level}:{msg}" for level, msg in env_rows)
    assert "warn:AUTH_ENABLED=true is read as OFF" in joined
    assert "info:AGENT_ENABLED=off" in joined
    assert "chmod 600" in joined and "no '='" in joined
    assert "set in the environment" in joined and "s3cret" not in joined
    assert _levels(ev(_healthy_facts(env_text=None)), ".env")[0][0] == "info"
    ports = _levels(ev(_healthy_facts(ports={"web": (8000, True), "model": (8080, False)})),
                    "ports")
    assert ports[0][0] == "info" and ports[1][0] == "ok"


def test_print_install_checks_counts_failures():
    out = io.StringIO()
    fails = setup.print_install_checks(facts=_healthy_facts(machine="x86_64", disk_free_gb=0.5),
                                       out=out)
    assert fails == 2 and "[FAIL] python" in out.getvalue()


def test_cli_wiring():
    import local_llm.cli as cli
    commands = cli.subcommands()
    assert commands["init"] is setup.init_cli and commands["service"] is setup.service_cli
    source = (REPO / "local_llm" / "cli.py").read_text()
    assert "print_install_checks(args.web_port, args.model_port)" in source


# --------------------------------------------------------------------------- #
# install.sh
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(BASH is None, reason="bash not installed")
def test_install_sh_syntax():
    done = subprocess.run([BASH, "-n", str(INSTALL_SH)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert os.access(INSTALL_SH, os.X_OK)


def _shims(tmp_path, *, os_name="Darwin", arch="arm64", py="3.12 arm64"):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "calls.log"
    scripts = {
        "python3": f'#!/bin/sh\necho "python3 $*" >> "{log}"\necho "{py}"\n',
        "git": f'#!/bin/sh\necho "git $*" >> "{log}"\n',
        "uname": f'#!/bin/sh\ncase "$1" in -s) echo {os_name};; -m) echo {arch};; esac\n',
        "sysctl": "#!/bin/sh\necho 0\n",
    }
    for name, body in scripts.items():
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path / "home"),
           "LANG": "C"}
    return env, log


def _install(tmp_path, *args, **shim):
    env, log = _shims(tmp_path, **shim)
    done = subprocess.run([BASH, str(INSTALL_SH), *args], cwd=str(tmp_path), env=env,
                          capture_output=True, text=True, timeout=60)
    calls = log.read_text().splitlines() if log.exists() else []
    return done, calls


@pytest.mark.skipif(BASH is None, reason="bash not installed")
def test_install_sh_dry_run_fresh_install(tmp_path):
    target = tmp_path / "app"
    done, calls = _install(tmp_path, "--dry-run", "--dir", str(target), "--multi-user",
                           "--service", "--join", "http://mini.local:8000", "--code", "12-34")
    assert done.returncode == 0, done.stderr
    out = done.stdout
    assert "Dry run" in out
    assert f"+ git clone --quiet https://github.com/zeno-b/native-silicon-local-llm.git {target}" \
        in out
    assert ("deploy.py init --multi-user --cluster secondary --primary-url "
            "http://mini.local:8000 --code 12-34 --non-interactive") in out
    assert "deploy.py service install" in out
    # Nothing ran: only the read-only interpreter probe, and no directory exists.
    assert all(c.startswith("python3 -c ") for c in calls), calls
    assert not target.exists()


@pytest.mark.skipif(BASH is None, reason="bash not installed")
def test_install_sh_dry_run_rerun_keeps_env(tmp_path):
    target = tmp_path / "app"
    (target / ".git").mkdir(parents=True)
    (target / ".env").write_text("MODEL_ID=x\n")
    done, calls = _install(tmp_path, "--dry-run", "--dir", str(target))
    assert done.returncode == 0, done.stderr
    assert f"+ git -C {target} pull --ff-only --quiet" in done.stdout
    assert "deploy.py init --deps-only" in done.stdout and "Keeping" in done.stdout
    git_calls = [c for c in calls if c.startswith("git ")]
    assert git_calls and all(("remote get-url" in c) or (" status " in c) for c in git_calls)


@pytest.mark.skipif(BASH is None, reason="bash not installed")
def test_install_sh_refusals(tmp_path):
    done, _ = _install(tmp_path, "--dry-run", "--dir", str(tmp_path / "a"), os_name="Linux")
    assert done.returncode == 1 and "macOS on Apple Silicon" in done.stderr
    done, _ = _install(tmp_path, "--dry-run", "--dir", str(tmp_path / "a"), arch="x86_64")
    assert done.returncode == 1 and "arm64" in done.stderr
    done, _ = _install(tmp_path, "--dry-run", "--join", "http://x:1")
    assert done.returncode == 1 and "--code" in done.stderr
    done, _ = _install(tmp_path, "--dry-run", "--python", "python3", "--dir",
                       str(tmp_path / "a"), py="3.9 arm64")
    assert done.returncode == 1 and "need 3.10" in done.stderr
    assert "sudo" not in INSTALL_SH.read_text().replace("never uses sudo", "")


# --------------------------------------------------------------------------- #
# Homebrew formula
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(RUBY is None, reason="ruby not installed")
def test_formula_ruby_syntax():
    done = subprocess.run([RUBY, "-c", str(FORMULA)], capture_output=True, text=True)
    assert done.returncode == 0 and "Syntax OK" in done.stdout, done.stderr


_RUBY_STUB = """
require "pathname"
class Formula
  def self.desc(*); end
  def self.homepage(*); end
  def self.head(*); end
  def self.depends_on(*); end
  def self.test(*); end
  def self.[](_name)
    Struct.new(:opt_bin).new(Pathname.new(ENV.fetch("STUB_PY_DIR")))
  end
  def opt_libexec
    Pathname.new(ENV.fetch("STUB_KEG"))
  end
end
load ARGV[0]
print LocalLlm.new.wrapper_script
"""


@pytest.mark.skipif(RUBY is None or BASH is None, reason="ruby or bash not installed")
def test_formula_wrapper_copies_the_keg_and_runs_deploy(tmp_path):
    keg = tmp_path / "keg"
    (keg / "local_llm").mkdir(parents=True)
    (keg / "local_llm" / "core.py").write_text("# core\n")
    (keg / "deploy.py").write_text("# deploy\n")
    py_dir = tmp_path / "pybin"
    py_dir.mkdir()
    fake_py = py_dir / "python3.12"
    fake_py.write_text('#!/bin/sh\necho "ran in $(pwd -P): $*"\n')
    fake_py.chmod(0o755)
    stub = tmp_path / "stub.rb"
    stub.write_text(_RUBY_STUB)
    env = {**os.environ, "STUB_KEG": str(keg), "STUB_PY_DIR": str(py_dir)}
    rendered = subprocess.run([RUBY, str(stub), str(FORMULA)], env=env, capture_output=True,
                              text=True, timeout=60)
    assert rendered.returncode == 0, rendered.stderr
    wrapper = tmp_path / "local-llm"
    wrapper.write_text(rendered.stdout)
    wrapper.chmod(0o755)
    assert subprocess.run([BASH, "-n", str(wrapper)]).returncode == 0
    home = tmp_path / "data"
    run_env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "LOCAL_LLM_HOME": str(home)}
    done = subprocess.run([str(wrapper), "init", "--dry-run"], env=run_env, capture_output=True,
                          text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert (home / "deploy.py").exists() and (home / "local_llm" / "core.py").exists()
    assert f"ran in {home.resolve()}: deploy.py init --dry-run" in done.stdout
    (home / "data").mkdir()
    (home / "data" / "keep.db").write_text("x")
    subprocess.run([str(wrapper)], env=run_env, capture_output=True, timeout=60)
    assert (home / "data" / "keep.db").exists()          # data survives a resync
    (home / ".git").mkdir()
    refused = subprocess.run([str(wrapper)], env=run_env, capture_output=True, text=True,
                             timeout=60)
    assert refused.returncode == 2 and "git checkout" in refused.stderr


# --------------------------------------------------------------------------- #
# The single-file build applies .env first too
# --------------------------------------------------------------------------- #

def test_bundle_lists_envfile_first_and_setup_cmds():
    source = (REPO / "bundle.py").read_text()
    order = source[source.index("ORDER = ["):]
    assert order.index('"envfile"') < order.index('"core"')
    assert '"setup_cmds"' in order
