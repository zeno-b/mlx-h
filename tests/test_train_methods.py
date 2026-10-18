"""Tests for training methods (DPO/ORPO/KTO/GRPO) and the training autopilot.

No real training runs: the trainer subprocess is replaced by a fake that
prints validation lines and writes adapter files, exactly like the SFT tests in
test_app.py (whose helpers are copied here, not imported).
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import local_llm.config as config_mod  # noqa: E402
config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-train-methods-"))

from fastapi.testclient import TestClient  # noqa: E402

from local_llm.config import Config  # noqa: E402
from local_llm.database import Database  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.training import RetrainManager  # noqa: E402
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.api import create_app  # noqa: E402
from local_llm import sysutil as sysutil_mod  # noqa: E402
from local_llm import training as training_mod  # noqa: E402
from local_llm import train_methods as tm  # noqa: E402


# A help text shaped like mlx-lm-lora 5.8.6's argparse output (train.py
# build_parser), trimmed to the flags this module passes.
PREF_HELP = (
    "usage: train.py [-h] [--model MODEL] [--train] [--data DATA] "
    "[--train-type {lora,dora,full}] "
    "[--train-mode {sft,dpo,dsla,ftpo,cpo,orpo,grpo,klpo,online_dpo,xpo,rlhf_reinforce,ppo}] "
    "[--num-layers NUM_LAYERS] [--batch-size BATCH_SIZE] [--iters ITERS] "
    "[--val-batches VAL_BATCHES] [--learning-rate LEARNING_RATE] "
    "[--steps-per-eval STEPS_PER_EVAL] [--adapter-path ADAPTER_PATH] "
    "[--save-every SAVE_EVERY] [--max-seq-length MAX_SEQ_LENGTH] [-c CONFIG] "
    "[--grad-checkpoint] [--beta BETA] [--reference-model-path REFERENCE_MODEL_PATH] "
    "[--reward-weights REWARD_WEIGHTS] [--reward-functions REWARD_FUNCTIONS] "
    "[--reward-functions-file REWARD_FUNCTIONS_FILE]\n"
    "  -c, --config CONFIG   A YAML configuration file with the training options\n")

# mlx-lm 0.32.2's flags (mlx_lm.lora --help), trimmed to the ones _build_cmd probes.
SFT_HELP = ("--model --train --data --fine-tune-type --num-layers --batch-size --iters "
            "--val-batches --learning-rate --steps-per-eval --adapter-path --save-every "
            "--max-seq-length -c --config --grad-checkpoint")

# A 3B 4-bit model's shape (Qwen2.5-Coder-3B-Instruct-4bit's config.json, with
# its 1.74 GB of safetensors), so the plan tests do not depend on the HF cache.
META_3B = {"model": "fake/3b", "known": True, "layers": 36, "hidden_size": 2048,
           "intermediate_size": 11008, "vocab_size": 151936, "heads": 16, "kv_heads": 2,
           "head_dim": 128, "bits": 4, "weights_bytes": 1_736_293_090,
           "weights_source": "safetensors on disk"}


# --------------------------------------------------------------------------- #
# Helpers (copied from tests/test_app.py, extended to sandbox train_methods)
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def training_sandbox(**overrides):
    root = Path(tempfile.mkdtemp())
    sft, adapters, backups, logs = (root / "sft", root / "latest",
                                    root / "backups", root / "logs")
    for d in (sft, adapters, backups, logs):
        d.mkdir(parents=True, exist_ok=True)
    patches = {"SFT_DIR": sft, "ADAPTER_DIR": adapters,
               "ADAPTER_BACKUP_DIR": backups, "LOG_DIR": logs,
               "REPLAY_FILE": sft / "replay.jsonl"}
    saved = []
    for mod in (training_mod, sysutil_mod, tm):
        for name, value in patches.items():
            if hasattr(mod, name):
                saved.append((mod, name, getattr(mod, name)))
                setattr(mod, name, value)
    cfg = Config(**overrides)
    db = Database(root / "test.db")
    mm = ModelServerManager(cfg.model, 8090, adapters)   # never started
    rm = RetrainManager(db, mm, cfg)
    try:
        yield rm, cfg, db, patches
    finally:
        for mod, name, value in saved:
            setattr(mod, name, value)
        db.close()
        shutil.rmtree(root, ignore_errors=True)


@contextlib.contextmanager
def patched(module, **values):
    old = {k: getattr(module, k) for k in values}
    for k, v in values.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)


def _fake_trainer(rm, patches, losses, *, exit_code=0, checkpoints=(), extra=""):
    """Stand in for the trainer: emit val lines, write adapter files, record cmd."""
    adapters = patches["ADAPTER_DIR"]
    calls = []

    def fake_run(cmd, stdout=None, stderr=None, timeout=None):
        calls.append(list(cmd))
        for it, loss in losses:
            stdout.write(f"Iter {it}: Val loss {loss:.4f}{extra}, Val took 0.1s\n")
        stdout.flush()
        (adapters / "adapters.safetensors").write_text("final")
        for it in checkpoints:
            (adapters / f"{it:07d}_adapters.safetensors").write_text(f"ckpt{it}")

        class Result:
            returncode = exit_code
        return Result()

    training_mod.subprocess.run = fake_run
    rm._lora_help = SFT_HELP
    rm.model_manager.stop = lambda *a, **k: None
    rm.model_manager.start = lambda *a, **k: None
    rm.model_manager.restart = lambda *a, **k: None
    return calls


def fb(db, prompt, answer, *, rating=1, corrected=None, approved=1, pending=0,
       source="button"):
    cur = db.execute(
        """INSERT INTO feedback (user_prompt, assistant_response, rating,
               corrected_response, approved_for_training, pending_approval, source)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (prompt, answer, rating, corrected, approved, pending, source))
    db.commit()
    return cur.lastrowid


def corrections(db, n, prefix="q"):
    for i in range(n):
        fb(db, f"{prefix}{i}?", f"a wrong answer ({i})", rating=-1,
           corrected=f"the right answer ({i})")


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# SFT stays byte-identical
# --------------------------------------------------------------------------- #
def test_default_config_is_sft_and_every_new_knob_is_neutral():
    cfg = Config()
    assert cfg.train_method == "sft" and tm.train_method_name(cfg) == "sft"
    assert cfg.train_autopilot is False and cfg.train_promote_ungated is False
    assert cfg.train_beta == 0.1 and cfg.train_reference_adapter == ""
    for key in ("train_method", "train_autopilot", "train_beta", "train_reference_adapter",
                "train_grpo_rewards", "train_promote_ungated"):
        assert key in Config.MUTABLE, key
    # Clamps: beta bounded, a typo kept (so the run refuses it by name).
    cfg.apply({"train_beta": 99, "train_method": " DOP "})
    assert cfg.train_beta == 10.0 and cfg.train_method == "dop"


def test_sft_command_is_unchanged_by_default():
    with training_sandbox(train_replay_ratio=0, train_tool_ratio=0) as (rm, cfg, db, patches):
        rm._lora_help = SFT_HELP
        plan = rm.plan(20)
        cmd = rm._build_cmd(plan)
        every = str(plan["eval_every"])
        assert cmd == [
            sys.executable, "-m", "mlx_lm.lora",
            "--model", cfg.model, "--train", "--data", str(patches["SFT_DIR"]),
            "--adapter-path", str(patches["ADAPTER_DIR"]),
            "--iters", str(plan["iters"]), "--batch-size", "1",
            "--fine-tune-type", "lora", "--num-layers", "8", "--learning-rate", "3e-5",
            "--grad-checkpoint", "--max-seq-length", "1024",
            "--steps-per-eval", every, "--save-every", every, "--val-batches", "-1",
            "--config", str(patches["SFT_DIR"] / "lora_config.yaml")], cmd


def test_sft_run_never_probes_the_preference_trainer():
    real_run = training_mod.subprocess.run
    probes = []
    with training_sandbox(train_replay_ratio=0, train_tool_ratio=0) as (rm, cfg, db, patches):
        corrections(db, 20)
        calls = _fake_trainer(rm, patches, [(1, 2.5), (30, 1.8), (60, 1.7)])
        try:
            with patched(training_mod, preference_trainer_help=lambda: probes.append(1) or ""):
                rm.run("test")
        finally:
            training_mod.subprocess.run = real_run
        assert "Retraining complete" in rm.status["message"], rm.status
        assert probes == [] and calls[0][2] == "mlx_lm.lora"
        meta = sysutil_mod.adapter_meta(patches["ADAPTER_DIR"])
        assert meta["method"] == "sft" and meta["autopilot"] is None
        assert meta["val_loss_baseline"] == 2.5


# --------------------------------------------------------------------------- #
# Refusals: never silently SFT
# --------------------------------------------------------------------------- #
def _refused(method, **over):
    real_run = training_mod.subprocess.run
    with training_sandbox(train_method=method, **over) as (rm, cfg, db, patches):
        corrections(db, 20)
        (patches["ADAPTER_DIR"] / "adapters.safetensors").write_text("original")
        calls = _fake_trainer(rm, patches, [(1, 1.0), (10, 0.5)])
        try:
            with patched(training_mod, preference_trainer_help=lambda: ""):
                rm.run("test")
        finally:
            training_mod.subprocess.run = real_run
        assert calls == [], "a refused method must not spawn any trainer"
        assert (patches["ADAPTER_DIR"] / "adapters.safetensors").read_text() == "original"
        assert db.get_untrained_count() == 20
        assert rm.status["running"] is False
        return rm.status["message"]


def test_preference_method_without_mlx_lm_lora_is_refused():
    msg = _refused("dpo")
    assert "mlx-lm-lora" in msg and "SFT" in msg, msg


def test_kto_is_refused_as_export_only():
    msg = _refused("kto")
    assert "KTO" in msg and "export" in msg, msg


def test_unknown_method_is_refused_by_name():
    msg = _refused("dop")
    assert "dop" in msg and "Unknown TRAIN_METHOD" in msg, msg


def test_installed_trainer_without_the_mode_is_refused():
    problem = tm.preference_trainer_problem("orpo", "--train-mode {sft,dpo} --config")
    assert problem and "--train-mode orpo" in problem
    assert tm.preference_trainer_problem("orpo", PREF_HELP) is None
    # Without a config file `fuse` cannot be disabled.
    assert "fuse" in tm.preference_trainer_problem("dpo", "--train-mode {sft,dpo}")


def test_reference_adapter_must_be_a_model_directory():
    root = Path(tempfile.mkdtemp())
    try:
        adapter = root / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapters.safetensors").write_text("x")
        model = root / "fused"
        model.mkdir()
        (model / "config.json").write_text("{}")
        (model / "model.safetensors").write_text("x")
        assert "fuse" in tm.reference_model_problem(Config(train_reference_adapter=str(adapter)))
        assert tm.reference_model_problem(Config(train_reference_adapter=str(model))) is None
        assert tm.reference_model_problem(Config()) is None
        assert tm.reference_model_problem(Config(train_reference_adapter="org/repo")) is None
        assert "does not exist" in tm.reference_model_problem(
            Config(train_reference_adapter=str(root / "missing")))
    finally:
        shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def _mixed_feedback(db):
    ids = {}
    ids["correction"] = fb(db, "p1", "<think>hmm</think>bad answer", rating=-1,
                           corrected="good answer")
    ids["pos"] = fb(db, "p2", "approved good", rating=1)
    ids["neg"] = fb(db, "p2", "approved bad", rating=-1)            # admin-approved negative
    # A non-admin's thumbs-down with a correction: queued, never approved.
    ids["pending"] = fb(db, "p2", "pending bad", rating=-1, corrected="pending fix",
                        approved=0, pending=1)
    # Implicit rejections are recorded unapproved; one was later approved.
    ids["implicit"] = fb(db, "p2", "implicit bad", rating=-1, approved=0, source="implicit")
    ids["implicit_ok"] = fb(db, "p3", "implicit approved bad", rating=-1, source="implicit")
    ids["p3pos"] = fb(db, "p3", "p3 good", rating=1)
    return ids


def test_dpo_pairs_use_only_approved_rows_and_strip_reasoning():
    with training_sandbox(train_val_split=0.05) as (rm, cfg, db, patches):
        ids = _mixed_feedback(db)
        out = tm.export_preferences(db, cfg, "dpo")
        rows = out["train"] + out["valid"]
        pairs = {(r["prompt"], r["chosen"], r["rejected"]) for r in rows}
        assert ("p1", "good answer", "bad answer") in pairs          # reasoning stripped
        assert ("p2", "approved good", "approved bad") in pairs
        assert ("p3", "p3 good", "implicit approved bad") in pairs
        text = json.dumps(rows)
        assert "pending" not in text and "implicit bad" not in text and "<think>" not in text
        assert set(out["feedback_ids"]) == {ids["correction"], ids["pos"], ids["neg"],
                                            ids["implicit_ok"], ids["p3pos"]}
        assert all(set(r) == {"system", "prompt", "chosen", "rejected"} for r in rows)
        assert all(r["system"] == cfg.training_system_prompt for r in rows)


def test_holdout_shares_no_prompt_and_empty_splits_are_not_written():
    with training_sandbox(train_val_split=0.2) as (rm, cfg, db, patches):
        corrections(db, 30)
        for i in range(10):                          # extra pairs on the same prompts
            fb(db, f"q{i}?", f"liked ({i})", rating=1)
        out_dir = patches["SFT_DIR"].parent / "pref" / "orpo"
        out = tm.export_preferences(db, cfg, "orpo", out_dir=out_dir)
        train, valid = jsonl(out_dir / "train.jsonl"), jsonl(out_dir / "valid.jsonl")
        assert train and valid and len(train) + len(valid) == out["counts"]["examples"]
        assert not ({r["prompt"] for r in train} & {r["prompt"] for r in valid})
        assert jsonl(out_dir / "test.jsonl") == valid
        # One prompt cannot be split: no valid.jsonl at all (the trainer's
        # loader crashes on an empty file), and a stale one is removed.
        db.execute("DELETE FROM feedback")
        db.commit()
        corrections(db, 1)
        out = tm.export_preferences(db, cfg, "orpo", out_dir=out_dir)
        assert out["counts"]["train"] == 1 and out["counts"]["valid"] == 0
        assert not (out_dir / "valid.jsonl").exists() and not (out_dir / "test.jsonl").exists()


def test_over_length_pairs_are_dropped():
    with training_sandbox(train_seq_len="400") as (rm, cfg, db, patches):
        fb(db, "short?", "no", rating=-1, corrected="yes")
        fb(db, "long?", "no", rating=-1, corrected="word " * 4000)
        out = tm.export_preferences(db, cfg, "dpo")
        assert out["counts"]["examples"] == 1 and out["counts"]["dropped_long"] == 1


def test_kto_export_labels_each_answer():
    with training_sandbox() as (rm, cfg, db, patches):
        _mixed_feedback(db)
        out = tm.export_preferences(db, cfg, "kto")
        rows = out["train"] + out["valid"]
        labels = {(r["prompt"], r["completion"]): r["label"] for r in rows}
        assert labels[("p1", "good answer")] is True and labels[("p1", "bad answer")] is False
        assert labels[("p2", "approved bad")] is False
        assert ("p2", "pending bad") not in labels and ("p2", "implicit bad") not in labels
        assert out["counts"]["desirable"] + out["counts"]["undesirable"] == len(rows)


def test_grpo_export_and_rewards():
    with training_sandbox(train_tool_ratio=5.0) as (rm, cfg, db, patches):
        fb(db, "what is 2+2?", "4", rating=1)
        tools = [{"messages": [
            {"role": "system", "content": "tool preamble"},
            {"role": "user", "content": "search the news"},
            {"role": "assistant", "content": json.dumps({"tool": "web_search",
                                                         "args": {"query": "news"}})}]}]
        out = tm.export_preferences(db, cfg, "grpo", tool_examples=tools)
        rows = {r["prompt"]: r for r in out["train"] + out["valid"]}
        assert rows["what is 2+2?"]["type"] == "text" and rows["what is 2+2?"]["answer"] == "4"
        assert rows["search the news"]["type"] == "tool"
        assert rows["search the news"]["system"] == "tool preamble"
        assert out["counts"]["human"] == 1 and out["counts"]["tool"] == 1

    call = json.dumps({"tool": "web_search", "args": {"query": "x"}})
    other = json.dumps({"tool": "calculator", "args": {"expression": "1+1"}})
    good = tm.grpo_score(call, call, "tool")
    assert good["tool_call"] == 1.0 and good["format"] == 1.0 and good["length"] > 0.9
    assert tm.grpo_score(other, call, "tool")["tool_call"] == 0.5
    assert tm.grpo_score("I would search for it.", call, "tool")["tool_call"] == 0.0
    assert tm.grpo_score('{"tool": "web_search", "args": ', "x", "text")["format"] == 0.0
    assert tm.grpo_score("", "4", "text") == {"tool_call": 0.0, "format": 0.0, "length": 0.0}
    assert tm.grpo_score("<think>so</think>4", "4", "text")["length"] == 1.0


def test_grpo_reward_spec_and_generated_reward_file():
    assert tm.parse_grpo_reward_spec("") == [("tool_call", 1.0), ("format", 0.5),
                                             ("length", 0.25)]
    assert tm.parse_grpo_reward_spec("format, length:2") == [("format", 1.0), ("length", 2.0)]
    for bad in ("speed:1", "format:-1", "format:x", "format,format"):
        try:
            tm.parse_grpo_reward_spec(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} was accepted")

    # The generated module registers one function per term with mlx_lm_lora's
    # registry. A stand-in registry module is installed so nothing third-party runs.
    registry = {}
    fake = types.ModuleType("mlx_lm_lora.trainer.grpo_reward_functions")
    fake.register_reward_function = lambda name=None: (
        lambda f: registry.setdefault(name or f.__name__, f))
    saved = {k: sys.modules.get(k) for k in ("mlx_lm_lora", "mlx_lm_lora.trainer",
                                             "mlx_lm_lora.trainer.grpo_reward_functions")}
    sys.modules["mlx_lm_lora"] = types.ModuleType("mlx_lm_lora")
    sys.modules["mlx_lm_lora.trainer"] = types.ModuleType("mlx_lm_lora.trainer")
    sys.modules["mlx_lm_lora.trainer.grpo_reward_functions"] = fake
    root = Path(tempfile.mkdtemp())
    try:
        path = tm.write_grpo_reward_file(root / "rewards.py",
                                         tm.parse_grpo_reward_spec("tool_call,length:0.5"))
        spec = importlib.util.spec_from_file_location("custom_rewards_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert set(registry) == {"llm_tool_call", "llm_length"}
        call = json.dumps({"tool": "web_search", "args": {"query": "x"}})
        scores = registry["llm_tool_call"](prompts=["a", "b"], completions=[call, "plain"],
                                           answer=[call, call], types=["tool", "tool"])
        assert scores == [1.0, 0.0]
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Autopilot
# --------------------------------------------------------------------------- #
def test_plan_fits_the_existing_8gb_recipe_and_refuses_what_cannot_fit():
    stats = {"count": 40, "p50": 300, "p95": 700, "max": 900}
    cfg = Config()
    plan = tm.plan_autopilot(8, META_3B, stats, method="sft", config=cfg)
    assert plan["fits"] is True and plan["grad_checkpoint"] is True
    assert plan["num_layers"] == 8 and plan["batch_size"] == 1
    assert plan["max_seq_length"] == 832            # p95 700 * 1.15, rounded up to 64
    assert plan["configured"]["fits"] is True       # today's recipe fits too
    assert plan["layer_streaming"]["supported"] is False
    # ORPO needs no reference and fits; DPO's reference copy does not.
    assert tm.plan_autopilot(8, META_3B, stats, method="orpo", config=cfg)["fits"] is True
    dpo = tm.plan_autopilot(8, META_3B, stats, method="dpo", config=cfg)
    assert dpo["fits"] is False and dpo["knob"] == "TRAIN_SEQ_LEN"
    assert "TRAIN_METHOD=orpo" in dpo["reason"]
    huge = dict(META_3B, weights_bytes=40 * 1024 ** 3)
    refused = tm.plan_autopilot(8, huge, stats, method="sft", config=cfg)
    assert refused["fits"] is False and refused["knob"] == "MODEL"
    # A big Mac gets a bigger recipe and keeps TRAIN_SEQ_LEN as the ceiling.
    big = tm.plan_autopilot(64, META_3B, dict(stats, count=400, p95=3000), config=cfg)
    assert big["fits"] and big["num_layers"] == 16 and big["batch_size"] == 4
    assert big["max_seq_length"] == 1024 and big["grad_checkpoint"] is False


def test_plan_without_a_model_config_keeps_the_recipe():
    plan = tm.plan_autopilot(8, {"model": "x", "known": False, "reason": "missing."},
                             {"count": 0}, config=Config())
    assert plan["estimated"] is False and plan["fits"] is None
    assert tm.autopilot_overrides(plan) == {}


def test_model_meta_reads_the_cached_config_when_present():
    meta = tm.model_train_meta("mlx-community/Qwen2.5-Coder-3B-Instruct-4bit")
    if not meta["known"]:
        return                                      # not cached on this machine
    assert meta["layers"] == 36 and meta["hidden_size"] == 2048 and meta["bits"] == 4
    assert meta["weights_bytes"] > 1_000_000_000
    assert tm.model_train_meta("no-such/model-xyz")["known"] is False


def test_autopilot_run_applies_the_plan_for_that_run_only_and_records_it():
    real_run = training_mod.subprocess.run
    with training_sandbox(train_replay_ratio=0, train_tool_ratio=0, train_autopilot=True,
                          train_num_layers=4, train_lora_rank=32) as (rm, cfg, db, patches):
        corrections(db, 20)
        calls = _fake_trainer(rm, patches, [(1, 2.5), (30, 1.8), (60, 1.7)])
        try:
            with patched(training_mod, model_train_meta=lambda model: dict(META_3B)):
                rm.run("test")
        finally:
            training_mod.subprocess.run = real_run
        assert "Retraining complete" in rm.status["message"], rm.status
        cmd = calls[0]
        assert cmd[cmd.index("--num-layers") + 1] == "8"
        yaml = (patches["SFT_DIR"] / "lora_config.yaml").read_text()
        assert "rank: 8" in yaml
        # The shared config is untouched.
        assert rm.config is cfg and cfg.train_num_layers == 4 and cfg.train_lora_rank == 32
        meta = sysutil_mod.adapter_meta(patches["ADAPTER_DIR"])
        assert meta["autopilot"]["num_layers"] == 8 and meta["method"] == "sft"


def test_autopilot_refuses_a_run_that_will_not_fit():
    real_run = training_mod.subprocess.run
    with training_sandbox(train_autopilot=True) as (rm, cfg, db, patches):
        corrections(db, 20)
        calls = _fake_trainer(rm, patches, [(1, 1.0), (2, 0.5)])
        huge = dict(META_3B, weights_bytes=400 * 1024 ** 3)
        try:
            with patched(training_mod, model_train_meta=lambda model: huge):
                rm.run("test")
        finally:
            training_mod.subprocess.run = real_run
        assert calls == [] and "MODEL" in rm.status["message"], rm.status
        assert db.get_untrained_count() == 20


# --------------------------------------------------------------------------- #
# Preference runs through the (faked) mlx_lm_lora trainer
# --------------------------------------------------------------------------- #
def _pref_run(method, losses, *, checkpoints=(), extra="", **over):
    real_run = training_mod.subprocess.run
    sandbox = training_sandbox(train_method=method, train_replay_ratio=0, **over)
    rm, cfg, db, patches = sandbox.__enter__()
    try:
        corrections(db, 20)
        (patches["ADAPTER_DIR"] / "adapters.safetensors").write_text("original")
        calls = _fake_trainer(rm, patches, losses, checkpoints=checkpoints, extra=extra)
        try:
            with patched(training_mod, preference_trainer_help=lambda: PREF_HELP):
                rm.run("test")
        finally:
            training_mod.subprocess.run = real_run
        return rm, db, patches, calls, sandbox
    except BaseException:
        sandbox.__exit__(None, None, None)
        raise


def test_dpo_run_builds_the_mlx_lm_lora_command_and_is_gated():
    rm, db, patches, calls, sandbox = _pref_run(
        "dpo", [(1, 0.6931), (20, 0.5), (40, 0.45)],
        extra=", Val chosen reward 0.1, Val rejected reward -0.1, Val accuracy 0.750, "
              "Val margin 0.2", train_beta=0.2)
    try:
        assert "DPO retraining complete" in rm.status["message"], rm.status
        cmd = calls[0]
        assert cmd[1:3] == ["-m", "mlx_lm_lora.train"]
        assert cmd[cmd.index("--train-mode") + 1] == "dpo"
        assert cmd[cmd.index("--beta") + 1] == "0.2"
        data_dir = Path(cmd[cmd.index("--data") + 1])
        assert data_dir == patches["SFT_DIR"].parent / "pref" / "dpo"
        assert "--reference-model-path" not in cmd       # base model is the reference
        yaml = Path(cmd[cmd.index("--config") + 1]).read_text()
        assert "fuse: false" in yaml and "lora_parameters:" in yaml
        meta = sysutil_mod.adapter_meta(patches["ADAPTER_DIR"])
        assert meta["method"] == "dpo" and meta["trainer"] == "mlx_lm_lora.train"
        assert meta["val_loss_baseline"] == 0.6931 and meta["val_loss_final"] == 0.45
        assert meta["val_accuracy_final"] == 0.75 and meta["reference"] == "base model"
        assert db.get_untrained_count() == 0
    finally:
        sandbox.__exit__(None, None, None)


def test_orpo_run_that_gets_worse_is_rolled_back():
    rm, db, patches, calls, sandbox = _pref_run("orpo", [(1, 1.0), (20, 1.4), (40, 1.6)],
                                                train_promote_best=False)
    try:
        assert "worse" in rm.status["message"], rm.status
        assert (patches["ADAPTER_DIR"] / "adapters.safetensors").read_text() == "original"
        assert db.get_untrained_count() == 20
    finally:
        sandbox.__exit__(None, None, None)


def test_preference_run_without_metrics_needs_promote_ungated():
    rm, db, patches, calls, sandbox = _pref_run("orpo", [])
    try:
        assert calls and "TRAIN_PROMOTE_UNGATED" in rm.status["message"], rm.status
        assert (patches["ADAPTER_DIR"] / "adapters.safetensors").read_text() == "original"
    finally:
        sandbox.__exit__(None, None, None)
    rm, db, patches, calls, sandbox = _pref_run("orpo", [], train_promote_ungated=True)
    try:
        assert "ORPO retraining complete" in rm.status["message"], rm.status
        meta = sysutil_mod.adapter_meta(patches["ADAPTER_DIR"])
        assert meta["gate"]["ungated"] is True
    finally:
        sandbox.__exit__(None, None, None)


def test_grpo_is_never_gated_on_its_surrogate_loss():
    rm, db, patches, calls, sandbox = _pref_run("grpo", [(1, 0.01), (20, 0.0)])
    try:
        assert "policy-gradient" in rm.status["message"], rm.status
        cmd = calls[0]
        assert cmd[cmd.index("--reward-functions") + 1] == "llm_tool_call,llm_format,llm_length"
        assert cmd[cmd.index("--reward-weights") + 1] == "[1.0,0.5,0.25]"
        assert Path(cmd[cmd.index("--reward-functions-file") + 1]).is_file()
    finally:
        sandbox.__exit__(None, None, None)


def test_train_plan_cli_prints_json():
    import contextlib as _cl
    import io
    buf = io.StringIO()
    missing = Path(tempfile.mkdtemp()) / "absent.db"     # never the real database
    with patched(tm, DB_PATH=missing, help_cmd=lambda module: "",
                 model_train_meta=lambda model: dict(META_3B)), _cl.redirect_stdout(buf):
        code = tm.train_plan_cli(["--method", "orpo"])
    plan = json.loads(buf.getvalue())
    assert plan["method"] == "orpo" and plan["fits"] is True and code == 0


def test_trainer_metric_parsing():
    text = ("Iter 1: Val loss 0.693, Val chosen reward 0.000, Val rejected reward 0.000, "
            "Val accuracy 0.000, Val margin 0.000, Val took 1.0s\n"
            "Iter 10: Val loss 0.512, Val chosen reward 0.3, Val rejected reward -0.2, "
            "Val accuracy 0.800, Val margin -0.100, Val took 1.0s\n")
    m = tm.parse_trainer_metrics(text)
    assert m["val_losses"] == [(1, 0.693), (10, 0.512)]
    assert m["val_accuracy"] == [(1, 0.0), (10, 0.8)]
    assert m["val_margin"][-1] == (10, -0.1)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
def test_training_routes():
    with training_sandbox(auth_enabled=True, admin_username="admin",
                          admin_password="adminpw123", allow_test_user=False,
                          node_token="secret-node-token",
                          skills_dir=tempfile.mkdtemp(prefix="llm-test-skills-")) as (
            rm, cfg, db, patches):
        _mixed_feedback(db)
        rm._lora_help = SFT_HELP
        rm._pref_help = ""                  # mlx-lm-lora absent: never spawned here
        app = create_app(cfg, db, rm.model_manager, rm, ToolRegistry(cfg, db))
        app.state.model_manager = rm.model_manager
        with TestClient(app) as c:
            assert c.get("/api/train/methods").status_code == 401
            assert c.post("/api/auth/login", json={"username": "admin",
                                                   "password": "adminpw123"}).status_code == 200
            r = c.get("/api/train/methods")
            assert r.status_code == 200, r.text
            methods = {m["name"]: m for m in r.json()["methods"]}
            assert list(methods) == ["sft", "dpo", "orpo", "kto", "grpo"]
            assert methods["sft"]["available"] and methods["sft"]["configured"]
            assert not methods["dpo"]["available"] and "mlx-lm-lora" in methods["dpo"]["reason"]
            assert not methods["kto"]["available"]

            r = c.get("/api/train/plan?method=orpo")
            assert r.status_code == 200, r.text
            assert r.json()["method"] == "orpo" and "layer_streaming" in r.json()
            assert c.get("/api/train/plan?method=nope").status_code == 400

            r = c.post("/api/train/export?method=dpo")
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["counts"]["examples"] == 3 and body["trainer_problem"]
            assert Path(body["dir"]) == patches["SFT_DIR"].parent / "pref" / "dpo"
            assert "train" not in body and len(body["sample"]) <= 2
            r = c.post("/api/train/export?method=sft")
            assert r.status_code == 200 and r.json()["method"] == "sft", r.text
            assert (patches["SFT_DIR"] / "train.jsonl").exists()

            rm.lock.acquire()
            try:
                assert c.post("/api/train/export?method=kto").status_code == 409
            finally:
                rm.lock.release()


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
