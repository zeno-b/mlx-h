"""Training methods beyond SFT, and a memory-aware autopilot for the recipe.

The feedback loop has only ever taught by imitation: every approved answer
becomes a supervised example. That throws away half of what the feedback UI
collects. A correction says "this, not that"; a thumbs-down next to a
thumbs-up on the same question says the same thing. Preference methods (DPO,
ORPO, KTO, GRPO) train on exactly that contrast, so this module turns the same
approved rows into the dataset shape each method needs and drives a trainer
that implements it.

Facts this module is built on (read from the sources, not assumed):

  * mlx-lm 0.32.2 (`mlx_lm.lora --help`) trains SFT only. Its memory levers
    are --num-layers, --batch-size, --max-seq-length, --grad-checkpoint,
    --grad-accumulation-steps and --clear-cache-threshold. It has no
    preference mode and no layer-streaming or offload option.
  * Preference training comes from the third-party `mlx-lm-lora` package
    (5.8.6 read without installing it). Its entry point is the module
    `mlx_lm_lora.train` (console script `mlx_lm_lora.train`), whose
    `--train-mode` offers sft, dpo, dsla, ftpo, cpo, orpo, grpo, klpo,
    online_dpo, xpo, rlhf_reinforce and ppo (train.py build_parser). It has NO
    KTO mode, so KTO here is export-only and a KTO run is refused.
  * Its local loader reads {train,valid,test}.jsonl from --data and calls
    create_dataset on each file that exists; an existing EMPTY file crashes on
    `data[0]` (trainer/datasets.py load_local_dataset, create_dataset). Empty
    splits are therefore not written at all.
  * DPO rows are {"system", "prompt", "chosen", "rejected"} with prompt a plain
    user string (DPODataset); ORPO reads the same keys (ORPODataset, system via
    system_feature); GRPO rows are {"prompt", "answer", "system", "type"}, and
    `answer` is required (GRPODataset indexes item["answer"]).
  * DPO and GRPO load a second, frozen copy of the model as the reference
    (train.py run(): load_reference_model for dpo/grpo/...). With no
    --reference-model-path that copy is the base model again; a path is
    loaded with mlx_lm's load() and NO adapter, so it must be a full model
    directory (a fused adapter), never a bare adapter directory. ORPO needs no
    reference, which is why it is the method that fits on 8 GB.
  * `fuse` defaults to True in its CONFIG_DEFAULTS and the --fuse CLI flag is
    store_true with default None, so the CLI cannot turn it off. Left on, a
    successful run writes a de-quantized FUSED MODEL into --adapter-path,
    which is the live adapter directory. The generated YAML config therefore
    always says `fuse: false` (main() applies YAML keys the CLI left None).
  * Its trainers print validation as "Iter N: Val loss X, ..." (DPO and ORPO
    add "Val accuracy" and "Val margin"), evaluate at it == 1 before the first
    update, and save "{it:07d}_adapters.safetensors" checkpoints, the same
    shapes the SFT gate already parses, so the val-loss gate and the
    best-checkpoint promotion carry over to DPO and ORPO unchanged.

The autopilot's memory model is an estimate (inference, not measurement):
resident weights from the safetensors on disk, a reference copy where the
method loads one, LoRA parameters with fp32 Adam state, activations (layer
boundaries only under gradient checkpointing), and the fp16 logits, which
dominate on small models with large vocabularies. It is calibrated so that the
recipe this project already runs (a 3B 4-bit model, 8 layers, 1024 tokens, on
8 GB) fits, and it errs high.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import math
import random
import re
import sys
from dataclasses import dataclass, is_dataclass, replace as _tm_dc_replace
from pathlib import Path

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .database import *  # noqa: F401,F403
from .sysutil import *  # noqa: F401,F403
from .websearch import *  # noqa: F401,F403
from .llm import *  # noqa: F401,F403


PREFERENCE_TRAINER_MODULE = "mlx_lm_lora.train"
TRAIN_METHODS = ("sft", "dpo", "orpo", "kto", "grpo")


# --------------------------------------------------------------------------- #
# Training data paths, resolved when used                                      #
# --------------------------------------------------------------------------- #
# SFT_DIR and REPLAY_FILE are copied into every module by star import when it
# loads, so redirecting DATA_DIR on the config module afterwards (what the test
# template and the UI preview server do, the way Config.settings_path honours
# it) left the export writing into the repo's data/sft. These read the config
# module's DATA_DIR at call time. A caller passes its own module's constant, so
# a sandbox that patches SFT_DIR / REPLAY_FILE directly still wins.
_TM_REPO_DATA_DIR = ROOT / "data"


def train_data_root() -> Path:
    """DATA_DIR as the config module holds it now (the bundle's own global there)."""
    mod = sys.modules.get(getattr(Config, "__module__", "") or "")
    value = getattr(mod, "DATA_DIR", None) if mod is not None else None
    return Path(value) if value is not None else Path(DATA_DIR)


def train_sft_dir(current: Path | None = None) -> Path:
    """The SFT data directory: `current` (default this module's SFT_DIR) when it
    was redirected, else DATA_DIR/sft when DATA_DIR was, else the repo default."""
    base = Path(current) if current is not None else Path(SFT_DIR)
    if base != _TM_REPO_DATA_DIR / "sft":
        return base
    root = train_data_root()
    return root / "sft" if root != _TM_REPO_DATA_DIR else base


def train_replay_file(current: Path | None = None) -> Path:
    """The rehearsal corpus, following a redirected REPLAY_FILE, SFT_DIR or DATA_DIR."""
    base = Path(current) if current is not None else Path(REPLAY_FILE)
    if base != _TM_REPO_DATA_DIR / "sft" / "replay.jsonl":
        return base
    return train_sft_dir() / "replay.jsonl"


@dataclass(frozen=True)
class TrainMethodSpec:
    """One training method: which trainer runs it and how its run is judged."""

    name: str
    label: str
    trainer_module: str | None     # None: no installed trainer implements it
    train_mode: str | None         # mlx_lm_lora --train-mode value
    data_format: str
    needs_reference: bool
    # "val_loss" when the trainer's validation loss is comparable between the
    # first (pre-update) evaluation and the last; None when it is not.
    metric: str | None
    experimental: bool
    description: str


TRAIN_METHOD_SPECS: dict[str, TrainMethodSpec] = {
    "sft": TrainMethodSpec(
        "sft", "SFT", "mlx_lm.lora", None,
        '{"messages": [system, user, assistant]}', False, "val_loss", False,
        "Supervised fine-tuning on approved answers (the original loop)."),
    "dpo": TrainMethodSpec(
        "dpo", "DPO", PREFERENCE_TRAINER_MODULE, "dpo",
        '{"system", "prompt", "chosen", "rejected"}', True, "val_loss", False,
        "Direct preference optimisation on chosen/rejected pairs. Loads a second, "
        "frozen copy of the model as the reference."),
    "orpo": TrainMethodSpec(
        "orpo", "ORPO", PREFERENCE_TRAINER_MODULE, "orpo",
        '{"system", "prompt", "chosen", "rejected"}', False, "val_loss", False,
        "Odds-ratio preference optimisation: SFT on the chosen answer plus a "
        "penalty on the rejected one. No reference model, so the lightest method."),
    "kto": TrainMethodSpec(
        "kto", "KTO", None, None,
        '{"system", "prompt", "completion", "label"}', False, None, False,
        "Unpaired preference data (each answer labelled desirable or not). "
        "Export only: mlx-lm-lora 5.8.6 has no KTO train mode."),
    "grpo": TrainMethodSpec(
        "grpo", "GRPO", PREFERENCE_TRAINER_MODULE, "grpo",
        '{"system", "prompt", "answer", "type"} + reward functions', True, None, True,
        "Group relative policy optimisation against programmatic rewards (tool-call "
        "parse success, format validity, length). Experimental: it generates while "
        "training, is slow on small Macs and reports no comparable metric."),
}

# GRPO reward terms this module implements, in TRAIN_GRPO_REWARDS order.
GRPO_REWARD_NAMES = ("tool_call", "format", "length")
_TM_DEFAULT_REWARDS = "tool_call:1.0,format:0.5,length:0.25"

# One prompt with many answers would otherwise dominate the pair set: every
# positive crossed with every negative grows quadratically.
_TM_MAX_PAIRS_PER_PROMPT = 4
_TM_GB = 1024 ** 3


# --------------------------------------------------------------------------- #
# Method selection and trainer probing
# --------------------------------------------------------------------------- #
def train_method_name(config) -> str:
    """The configured method, normalised. Unknown names are returned as-is so
    the run refuses them: rewriting a typo to "sft" would be a silent SFT."""
    return str(getattr(config, "train_method", "sft") or "sft").strip().lower() or "sft"


def preference_trainer_help() -> str:
    """`mlx_lm_lora.train --help`, or "" when the package is not importable.

    find_spec first: spawning an interpreter that only fails to import costs a
    second or two on every probe of a machine that never installed it.
    """
    try:
        if importlib.util.find_spec("mlx_lm_lora") is None:
            return ""
    except (ImportError, ValueError):
        return ""
    return help_cmd(PREFERENCE_TRAINER_MODULE)


def _tm_mode_offered(help_text: str, mode: str) -> bool:
    """Whether --help lists `mode` among the --train-mode choices."""
    if not help_has_flag(help_text, "--train-mode"):
        return False
    match = re.search(r"--train-mode\s*\{([^}]*)\}", help_text)
    if match:
        return mode in [c.strip() for c in match.group(1).split(",")]
    return re.search(r"(?<![\w-])" + re.escape(mode) + r"(?![\w-])", help_text) is not None


def preference_trainer_problem(method: str, help_text: str) -> str | None:
    """Why `method` cannot be trained here, or None when it can."""
    spec = TRAIN_METHOD_SPECS.get(method)
    if spec is None:
        return (f"Unknown TRAIN_METHOD={method!r}; choose one of "
                f"{', '.join(TRAIN_METHODS)}. Refusing rather than training SFT.")
    if spec.trainer_module is None:
        return (f"{spec.label} has no trainer here: mlx-lm-lora 5.8.6 offers sft, dpo, "
                f"orpo, cpo, grpo and others, but no KTO mode. Export the set with "
                f"POST /api/train/export?method={method} and train it elsewhere, or "
                f"choose TRAIN_METHOD=orpo. Nothing was trained.")
    if method == "sft":
        return None
    if not help_text:
        return (f"TRAIN_METHOD={method} needs the mlx-lm-lora package (pip install "
                f"mlx-lm-lora), and `python -m {PREFERENCE_TRAINER_MODULE} --help` could "
                f"not be read. Refusing rather than falling back to SFT.")
    if not _tm_mode_offered(help_text, spec.train_mode or method):
        return (f"The installed {PREFERENCE_TRAINER_MODULE} does not offer --train-mode "
                f"{spec.train_mode}. Upgrade mlx-lm-lora. Nothing was trained.")
    if not (help_has_flag(help_text, "--config") or help_has_flag(help_text, "-c")):
        # Without a config file `fuse` cannot be turned off, and a fused model
        # would be written over the live adapter directory.
        return (f"The installed {PREFERENCE_TRAINER_MODULE} takes no --config file, so its "
                f"fuse-on-finish default cannot be disabled; refusing to let it write a "
                f"fused model over the live adapter.")
    return None


def reference_model_problem(config) -> str | None:
    """Validate TRAIN_REFERENCE_ADAPTER for the methods that load a reference.

    mlx_lm_lora loads --reference-model-path with mlx_lm's load() and no
    adapter, so the value must be a model directory (or a Hugging Face repo id),
    never a bare LoRA adapter directory.
    """
    raw = str(getattr(config, "train_reference_adapter", "") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.exists():
        looks_like_repo = ("/" in raw and not raw.startswith(("/", ".", "~"))
                           and raw.count("/") == 1)
        if looks_like_repo:
            return None
        return f"TRAIN_REFERENCE_ADAPTER={raw} does not exist."
    if path.is_dir() and (path / "config.json").is_file() and any(path.glob("*.safetensors")):
        return None
    if path.is_dir() and ((path / "adapter_config.json").is_file()
                          or (path / "adapters.safetensors").is_file()):
        return (f"TRAIN_REFERENCE_ADAPTER={raw} is a LoRA adapter directory, but "
                f"mlx_lm_lora loads the reference with mlx_lm's load() and no adapter. "
                f"Fuse it first (python -m mlx_lm fuse --model <base> --adapter-path "
                f"{raw} --save-path <dir>) and point TRAIN_REFERENCE_ADAPTER at <dir>.")
    return (f"TRAIN_REFERENCE_ADAPTER={raw} is not a model directory "
            f"(config.json plus *.safetensors weights).")


def describe_train_methods(config, sft_help: str, pref_help: str) -> list[dict]:
    """Every method, whether it can run on this install, and why not."""
    current = train_method_name(config)
    out = []
    for name in TRAIN_METHODS:
        spec = TRAIN_METHOD_SPECS[name]
        if name == "sft":
            problem = None if sft_help else (
                "mlx_lm.lora --help could not be read; install or repair mlx-lm.")
        else:
            problem = preference_trainer_problem(name, pref_help)
        out.append({
            "name": name, "label": spec.label, "trainer": spec.trainer_module,
            "train_mode": spec.train_mode, "available": problem is None,
            "reason": problem, "data_format": spec.data_format,
            "needs_reference": spec.needs_reference,
            "gate": spec.metric or "none (needs TRAIN_PROMOTE_UNGATED=1)",
            "experimental": spec.experimental, "description": spec.description,
            "configured": name == current,
        })
    return out


# --------------------------------------------------------------------------- #
# GRPO rewards
# --------------------------------------------------------------------------- #
def parse_grpo_reward_spec(text: str | None) -> list[tuple[str, float]]:
    """"tool_call:1.0,format:0.5" -> [("tool_call", 1.0), ("format", 0.5)].

    A name without a weight weighs 1.0; an empty spec is the default trio.
    Unknown names and negative weights are refused, not dropped: a typo that
    silently removed a reward term would change what the model is optimised for.
    """
    raw = str(text or "").strip() or _TM_DEFAULT_REWARDS
    out: list[tuple[str, float]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, weight = part.partition(":")
        name = name.strip().lower()
        if name not in GRPO_REWARD_NAMES:
            raise ValueError(f"Unknown GRPO reward {name!r} in TRAIN_GRPO_REWARDS; "
                             f"choose from {', '.join(GRPO_REWARD_NAMES)}.")
        try:
            value = float(weight) if weight.strip() else 1.0
        except ValueError:
            raise ValueError(f"GRPO reward weight {weight!r} for {name} is not a number.") from None
        if value < 0 or not math.isfinite(value):
            raise ValueError(f"GRPO reward weight for {name} must be a finite number >= 0.")
        if any(n == name for n, _ in out):
            raise ValueError(f"GRPO reward {name} is listed twice in TRAIN_GRPO_REWARDS.")
        out.append((name, value))
    if not out:
        raise ValueError("TRAIN_GRPO_REWARDS names no reward.")
    return out


def grpo_score(completion: str, answer: str, kind: str = "text") -> dict[str, float]:
    """Score one GRPO completion on every reward term, each in [0, 1].

    tool_call: for a prompt whose approved answer was a tool call, 1.0 when the
      completion parses as a call to the same tool (via llm.parse_tool_call, the
      parser the agent loop itself uses), 0.5 for a call to another tool, 0 for
      no call. For a direct-answer prompt, 1.0 when it answered without a call.
    format: 0 for an empty reply or a reply that starts like structured output
      (a JSON object or a <tool_call> wrapper) and does not parse; else 1.
    length: closeness to the approved answer's length, 1 - |len - ref| / ref.
    """
    visible = strip_reasoning(completion or "").strip()
    call = parse_tool_call(visible) if visible else None
    if kind == "tool":
        expected = parse_tool_call(strip_reasoning(answer or "").strip())
        if call is None:
            tool = 0.0
        else:
            tool = 1.0 if expected is None or call[0] == expected[0] else 0.5
    else:
        tool = 1.0 if (visible and call is None) else 0.0
    if not visible:
        fmt = 0.0
    elif visible.startswith(("{", "<tool_call>", "[TOOL_CALLS]")):
        fmt = 1.0 if call is not None else 0.0
    else:
        fmt = 1.0
    ref = len(strip_reasoning(answer or "").strip())
    if ref <= 0:
        length = 1.0 if visible else 0.0
    else:
        length = max(0.0, 1.0 - min(1.0, abs(len(visible) - ref) / ref))
    return {"tool_call": tool, "format": fmt, "length": length}


_TM_REWARD_FILE = '''"""GRPO reward functions, generated by local_llm.train_methods. Do not edit:
this file is rewritten on every GRPO run from TRAIN_GRPO_REWARDS.

mlx_lm_lora imports it through --reward-functions-file and calls each function
as f(prompts=..., completions=..., answer=..., types=...) -> list[float].
"""
import json
import sys

sys.path.insert(0, {root!r})
try:
    from local_llm.train_methods import grpo_score as _score
except Exception:  # bundled single-file install: no package to import
    _score = None

from mlx_lm_lora.trainer.grpo_reward_functions import register_reward_function


def _fallback(completion, answer, kind):
    # The canonical {{"tool": ..., "args": ...}} JSON only; the full parser
    # needs the package.
    text = (completion or "").split("</think>")[-1].strip()
    call = None
    if text.startswith("{{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and isinstance(obj.get("tool"), str):
                call = obj["tool"]
        except ValueError:
            pass
    if kind == "tool":
        tool = 0.0 if call is None else 1.0
    else:
        tool = 1.0 if (text and call is None) else 0.0
    fmt = 0.0 if not text else (1.0 if (not text.startswith("{{") or call) else 0.0)
    ref = len((answer or "").strip())
    length = (1.0 if text else 0.0) if ref <= 0 else max(0.0, 1.0 - min(1.0, abs(len(text) - ref) / ref))
    return {{"tool_call": tool, "format": fmt, "length": length}}


def _scores(completions, answer, types, key):
    out = []
    for i, completion in enumerate(completions or []):
        ref = answer[i] if answer and i < len(answer) else ""
        kind = (types[i] if types and i < len(types) else None) or "text"
        scored = _score(completion, ref, kind) if _score else _fallback(completion, ref, kind)
        out.append(float(scored.get(key, 0.0)))
    return out

{functions}
'''


def write_grpo_reward_file(path: Path, rewards: list[tuple[str, float]]) -> Path:
    """Write the reward module mlx_lm_lora imports via --reward-functions-file."""
    functions = []
    for name, _weight in rewards:
        functions.append(
            f'\n@register_reward_function("llm_{name}")\n'
            f'def llm_{name}(prompts, completions, answer, types=None):\n'
            f'    return _scores(completions, answer, types, "{name}")\n')
    root = str(Path(__file__).resolve().parent.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_TM_REWARD_FILE.format(root=root, functions="".join(functions)),
                    encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Preference export
# --------------------------------------------------------------------------- #
def _tm_seq_budget(config) -> int | None:
    if not getattr(config, "train_drop_over_length", True):
        return None
    try:
        return int(str(config.train_seq_len).strip())
    except (TypeError, ValueError):
        return None


def _tm_row_tokens(row: dict) -> int:
    """Estimated tokens of the longest sequence the trainer builds from a row."""
    targets = [row.get(k) for k in ("chosen", "rejected", "completion", "answer")]
    longest = max((t for t in targets if isinstance(t, str)), key=len, default="")
    return messages_tokens([{"role": "system", "content": row.get("system", "")},
                            {"role": "user", "content": row.get("prompt", "")},
                            {"role": "assistant", "content": longest}])


def _tm_split(groups: dict[str, list[dict]], val_split: float,
              seed: int) -> tuple[list[dict], list[dict]]:
    """Hold out whole prompts, so no feedback row lands in both splits.

    Every row built from a feedback row is keyed by that row's prompt, so
    splitting by prompt is what keeps the held-out set honest: a pair whose
    chosen answer was also trained on in another pair measures nothing.
    """
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) < 2:
        return [r for k in keys for r in groups[k]], []
    holdout = max(1, int(round(len(keys) * val_split)))
    holdout = min(holdout, len(keys) - 1)
    valid_keys = set(keys[len(keys) - holdout:])
    train = [r for k in keys if k not in valid_keys for r in groups[k]]
    valid = [r for k in keys if k in valid_keys for r in groups[k]]
    return train, valid


def export_preferences(db, config, method: str, *, out_dir: Path | None = None,
                       tool_examples: list[dict] | None = None, seed: int = 42) -> dict:
    """Build the method's dataset from approved feedback; write it when out_dir is set.

    Only rows with approved_for_training = 1 (and no pending approval) are
    read, on BOTH sides of a pair: a non-admin's thumbs-down is never approved
    on submission, and an implicit rejection is recorded unapproved, so neither
    enters a preference set until an admin approves it. Targets go through
    strip_reasoning so the model is never taught to emit, or penalised for,
    a reasoning block it was not shown.
    """
    method = str(method or "").strip().lower()
    if method not in ("dpo", "orpo", "kto", "grpo"):
        raise ValueError(f"export_preferences handles dpo, orpo, kto and grpo, not {method!r}.")
    system = config.training_system_prompt
    rows = db.execute(
        "SELECT id, user_prompt, assistant_response, corrected_response, rating "
        "FROM feedback WHERE approved_for_training = 1 "
        "AND COALESCE(pending_approval, 0) = 0 ORDER BY id").fetchall()

    by_prompt: dict[str, list[dict]] = {}
    for row in rows:
        prompt = (row["user_prompt"] or "").strip()
        original = strip_reasoning(row["assistant_response"] or "").strip()
        corrected = strip_reasoning(row["corrected_response"] or "").strip()
        if not prompt:
            continue
        by_prompt.setdefault(prompt, []).append({
            "id": row["id"], "original": original, "corrected": corrected,
            "rating": int(row["rating"] or 0)})

    groups: dict[str, list[dict]] = {}
    seen: set[tuple] = set()
    human_ids: set[int] = set()

    def add(prompt: str, item: dict, ids: list[int]) -> None:
        key = (prompt,) + tuple(str(item.get(k)) for k in
                                ("chosen", "rejected", "completion", "label", "answer"))
        if key in seen:
            return
        seen.add(key)
        item = dict({"system": system, "prompt": prompt}, **item)
        item["_ids"] = ids
        groups.setdefault(prompt, []).append(item)

    for prompt, items in by_prompt.items():
        positives: list[tuple[str, int]] = []      # (text, feedback id)
        negatives: list[tuple[str, int]] = []
        for it in items:
            if it["corrected"] and it["corrected"] != it["original"]:
                positives.append((it["corrected"], it["id"]))
                if it["original"]:
                    negatives.append((it["original"], it["id"]))
            elif it["rating"] > 0 and it["original"]:
                positives.append((it["original"], it["id"]))
            elif it["rating"] < 0 and it["original"]:
                negatives.append((it["original"], it["id"]))
        if method in ("dpo", "orpo"):
            pairs = 0
            # Corrections first: the strongest "this, not that" signal there is.
            ordered = sorted(((c, r) for c in positives for r in negatives),
                             key=lambda pr: (pr[0][1] != pr[1][1], pr[0][1], pr[1][1]))
            for (chosen, cid), (rejected, rid) in ordered:
                if pairs >= _TM_MAX_PAIRS_PER_PROMPT or chosen == rejected:
                    continue
                before = len(seen)
                add(prompt, {"chosen": chosen, "rejected": rejected}, sorted({cid, rid}))
                if len(seen) > before:
                    pairs += 1
        elif method == "kto":
            for text, fid in positives:
                add(prompt, {"completion": text, "label": True}, [fid])
            for text, fid in negatives:
                add(prompt, {"completion": text, "label": False}, [fid])
        else:  # grpo
            if positives:
                answer, fid = positives[0]
                kind = "tool" if parse_tool_call(answer) is not None else "text"
                add(prompt, {"answer": answer, "type": kind}, [fid])

    human = sum(len(v) for v in groups.values())
    tool_rows = 0
    if method == "grpo" and tool_examples:
        cap = int(getattr(config, "train_tool_examples", 0) or 0)
        ratio = float(getattr(config, "train_tool_ratio", 0) or 0)
        if ratio > 0:
            cap = min(cap, int(human * ratio))
        for example in tool_examples[:max(0, cap)]:
            msgs = example.get("messages") or []
            if len(msgs) < 3:
                continue
            prompt = str(msgs[1].get("content") or "").strip()
            call = str(msgs[2].get("content") or "").strip()
            if not prompt or not call or prompt in by_prompt:
                continue
            before = len(seen)
            add(prompt, {"system": str(msgs[0].get("content") or system),
                         "answer": call, "type": "tool"}, [])
            tool_rows += len(seen) - before

    budget = _tm_seq_budget(config)
    dropped_long = 0
    if budget is not None:
        for prompt in list(groups):
            kept = [r for r in groups[prompt] if _tm_row_tokens(r) <= budget]
            dropped_long += len(groups[prompt]) - len(kept)
            if kept:
                groups[prompt] = kept
            else:
                del groups[prompt]

    for items in groups.values():
        for r in items:
            human_ids.update(r["_ids"])
    train, valid = _tm_split(groups, float(getattr(config, "train_val_split", 0.1)), seed)
    random.Random(seed).shuffle(train)

    def clean(chunk: list[dict]) -> list[dict]:
        return [{k: v for k, v in r.items() if k != "_ids"} for r in chunk]

    files: list[str] = []
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, chunk in (("train.jsonl", train), ("valid.jsonl", valid),
                            ("test.jsonl", valid)):
            target = out_dir / name
            if chunk:
                target.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n"
                                          for x in clean(chunk)), encoding="utf-8")
                files.append(str(target))
            elif target.exists():
                # An empty file is a crash in the trainer's loader; a stale one
                # from an earlier export is wrong data. Remove both.
                target.unlink()

    counts = {
        "rows_considered": len(rows), "examples": len(train) + len(valid),
        "human": sum(1 for r in train + valid if r["_ids"]), "tool": tool_rows,
        "train": len(train), "valid": len(valid), "dropped_long": dropped_long,
        "prompts": len(groups),
    }
    if method == "kto":
        counts["desirable"] = sum(1 for r in train + valid if r.get("label") is True)
        counts["undesirable"] = counts["examples"] - counts["desirable"]
    return {"method": method, "format": TRAIN_METHOD_SPECS[method].data_format,
            "counts": counts, "train": clean(train), "valid": clean(valid),
            "feedback_ids": sorted(human_ids), "files": files,
            "dir": str(out_dir) if out_dir is not None else None}


# --------------------------------------------------------------------------- #
# Autopilot: model shape, corpus shape, memory estimate, plan
# --------------------------------------------------------------------------- #
def _tm_model_dir(model_id: str) -> Path | None:
    """Directory holding config.json for a local model or a cached HF repo."""
    try:
        local = Path(str(model_id)).expanduser()
        if local.is_dir():
            return local if (local / "config.json").is_file() else None
        repo = hf_cache_dir() / ("models--" + str(model_id).replace("/", "--"))
        snapshots = repo / "snapshots"
        ref = repo / "refs" / "main"
        if ref.is_file():
            pinned = snapshots / ref.read_text(encoding="utf-8").strip()
            if (pinned / "config.json").is_file():
                return pinned
        if snapshots.is_dir():
            for snap in sorted(snapshots.iterdir()):
                if (snap / "config.json").is_file():
                    return snap
    except (OSError, ValueError):
        return None
    return None


def model_train_meta(model_id: str) -> dict:
    """The model's shape from its config.json on disk; never downloads or loads it."""
    folder = _tm_model_dir(model_id) if model_id else None
    if folder is None:
        return {"model": model_id, "known": False,
                "reason": "config.json is not on disk (local directory or Hugging Face cache)."}
    try:
        cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"model": model_id, "known": False, "reason": f"config.json unreadable: {exc}"}
    src = cfg
    if "num_hidden_layers" not in cfg and isinstance(cfg.get("text_config"), dict):
        src = cfg["text_config"]
    try:
        layers = int(src.get("num_hidden_layers") or 0)
        hidden = int(src.get("hidden_size") or 0)
        inter = int(src.get("intermediate_size") or 4 * hidden)
        vocab = int(src.get("vocab_size") or cfg.get("vocab_size") or 32000)
        heads = int(src.get("num_attention_heads") or 1)
        kv_heads = int(src.get("num_key_value_heads") or heads)
        head_dim = int(src.get("head_dim") or (hidden // max(1, heads)))
    except (TypeError, ValueError):
        return {"model": model_id, "known": False, "reason": "config.json has no usable shape."}
    if layers <= 0 or hidden <= 0:
        return {"model": model_id, "known": False, "reason": "config.json has no layer shape."}
    quant = cfg.get("quantization") or cfg.get("quantization_config") or {}
    bits = int(quant.get("bits") or 16) if isinstance(quant, dict) else 16
    group = int(quant.get("group_size") or 64) if isinstance(quant, dict) else 64
    weights = 0
    try:
        weights = sum(p.stat().st_size for p in folder.glob("*.safetensors"))
    except OSError:
        weights = 0
    source = "safetensors on disk"
    if not weights:
        params = layers * (4 * hidden * hidden + 3 * hidden * inter) + vocab * hidden
        if not cfg.get("tie_word_embeddings", True):
            params += vocab * hidden
        per_param = bits / 8.0 + (4.0 / group if bits < 16 else 0.0)
        weights = int(params * per_param)
        source = "estimated from config.json"
    return {"model": model_id, "known": True, "layers": layers, "hidden_size": hidden,
            "intermediate_size": inter, "vocab_size": vocab, "heads": heads,
            "kv_heads": kv_heads, "head_dim": head_dim, "bits": bits,
            "weights_bytes": int(weights), "weights_source": source}


def _tm_percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(math.ceil(pct * len(ordered))) - 1))
    return int(ordered[index])


def corpus_token_stats(db, config, method: str = "sft") -> dict:
    """Estimated token lengths of the examples the next run would train on."""
    lengths: list[int] = []
    if method in ("dpo", "orpo", "kto", "grpo"):
        export = export_preferences(db, dataclass_replace_safe(config, train_drop_over_length=False),
                                    method)
        lengths = [_tm_row_tokens(r) for r in export["train"] + export["valid"]]
    else:
        system = config.training_system_prompt
        rows = db.execute(
            "SELECT user_prompt, assistant_response, corrected_response FROM feedback "
            "WHERE approved_for_training = 1").fetchall()
        for row in rows:
            answer = (row["corrected_response"] or row["assistant_response"] or "").strip()
            prompt = (row["user_prompt"] or "").strip()
            if prompt and answer:
                lengths.append(messages_tokens([
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer}]))
    return {"count": len(lengths), "p50": _tm_percentile(lengths, 0.5),
            "p95": _tm_percentile(lengths, 0.95), "max": max(lengths) if lengths else 0}


def dataclass_replace_safe(config, **changes):
    """dataclasses.replace for a Config, or a shallow attribute copy for a stub."""
    if is_dataclass(config):
        return _tm_dc_replace(config, **changes)
    clone = type("_TmConfig", (), {})()
    clone.__dict__.update(getattr(config, "__dict__", {}))
    for key, value in changes.items():
        setattr(clone, key, value)
    return clone


def _tm_budget_bytes(ram_gb: float) -> int:
    """Memory one training process may use: ~70% of RAM (macOS keeps the rest
    for itself and caps the GPU working set near there), less ~0.5 GB for the
    web process that stays up while the model server is stopped."""
    return int(max(0.0, float(ram_gb) * 0.70 - 0.5) * _TM_GB)


def estimate_train_memory(meta: dict, *, num_layers: int, rank: int, batch: int,
                          seq: int, grad_checkpoint: bool, method: str = "sft",
                          fine_tune_type: str = "lora") -> dict:
    """Peak bytes of one training step (heuristic; see the module docstring)."""
    L, H, I = meta["layers"], meta["hidden_size"], meta["intermediate_size"]
    V, A = meta["vocab_size"], meta["heads"]
    n = L if num_layers < 0 else max(1, min(num_layers, L))
    spec = TRAIN_METHOD_SPECS.get(method, TRAIN_METHOD_SPECS["sft"])
    weights = int(meta["weights_bytes"])
    reference = weights if spec.needs_reference else 0
    if fine_tune_type == "full":
        trainable = n * (4 * H * H + 3 * H * I) * 16
    else:
        # q, k, v, o (approximately H -> H each) plus gate, up, down (H <-> I),
        # each A (in x r) and B (r x out), fp32 with grad and two Adam moments.
        trainable = n * rank * (4 * 2 * H + 3 * (H + I)) * 16
    # Pair methods run chosen and rejected through the model in one step.
    seqs = batch * (2 if method in ("dpo", "orpo") else 1)
    group = 4 if method == "grpo" else 1          # mlx_lm_lora --group-size default
    seqs *= group
    tokens = seqs * seq
    attention = 5 * A * seq * seq * seqs          # Korthikanti et al., bytes per layer
    if grad_checkpoint:
        activations = n * tokens * H * 2 + tokens * 34 * H + attention
    else:
        activations = n * (tokens * 34 * H + attention)
    logits = tokens * V * 2 * 3                   # fp16 logits, softmax and its grad
    generation = 0
    if method == "grpo":
        kv = 2 * L * meta["kv_heads"] * meta["head_dim"] * 2
        generation = kv * group * batch * (seq + 512)   # --max-completion-length 512
    overhead = int(0.6 * _TM_GB)
    parts = {"weights": weights, "reference": reference, "trainable": trainable,
             "activations": activations, "logits": logits, "generation": generation,
             "overhead": overhead}
    return {"peak_bytes": int(sum(parts.values())), "parts": parts, "layers_tuned": n}


def _tm_gb(value: float) -> float:
    return round(value / _TM_GB, 2)


def layer_streaming_support(help_texts: list[str]) -> dict:
    """Whether any installed trainer offers layer streaming / offload (fact from --help)."""
    for text in help_texts:
        for flag in re.findall(r"--[\w-]*(?:stream|offload)[\w-]*", text or ""):
            return {"supported": True, "flag": flag,
                    "note": f"{flag} was found in a trainer's --help; the autopilot does "
                            f"not pass it automatically because its semantics are unverified."}
    return {"supported": False, "flag": None,
            "note": "Unsupported: neither mlx_lm.lora (mlx-lm 0.32.2) nor mlx_lm_lora.train "
                    "(mlx-lm-lora 5.8.6) exposes a layer-streaming or offload option. Memory "
                    "is held down with --num-layers, --grad-checkpoint, --batch-size and "
                    "--max-seq-length instead."}


def plan_autopilot(ram_gb: float, meta: dict, stats: dict, *, method: str = "sft",
                   config=None, help_texts: list[str] | None = None) -> dict:
    """Choose layers, rank, batch, sequence window and checkpointing that fit.

    Preference order when shrinking: batch first (it scales activations and
    the logits, the largest term), then layers, then rank. Gradient
    checkpointing is turned off only when the run fits in 60% of the budget
    without it, since it costs ~30% speed and nothing else.
    """
    ram_gb = float(ram_gb or 0)
    budget = _tm_budget_bytes(ram_gb)
    ft = str(getattr(config, "train_fine_tune_type", "lora") or "lora").lower()
    count = int(stats.get("count") or 0)
    val_split = float(getattr(config, "train_val_split", 0.1) or 0.1)
    n_train = max(1, int(round(count * (1.0 - val_split)))) if count else 0
    # TRAIN_SEQ_LEN stays the ceiling: the autopilot only shrinks the window to
    # what the corpus needs, so lowering TRAIN_SEQ_LEN is a knob that works.
    cap = 4096 if ram_gb >= 32 else 2048
    try:
        cap = min(cap, max(256, int(str(getattr(config, "train_seq_len", cap)).strip())))
    except (TypeError, ValueError):
        pass
    if stats.get("p95"):
        seq = int(math.ceil(stats["p95"] * 1.15 / 64.0)) * 64
        seq = max(256, min(cap, seq))
    else:
        try:
            seq = int(str(getattr(config, "train_seq_len", "1024")).strip())
        except (TypeError, ValueError):
            seq = 1024
    plan: dict = {
        "method": method, "ram_gb": round(ram_gb, 1), "budget_gb": _tm_gb(budget),
        "model": {k: meta.get(k) for k in ("model", "known", "layers", "hidden_size",
                                           "vocab_size", "bits", "weights_source")},
        "corpus": dict(stats), "max_seq_length": seq,
        "layer_streaming": layer_streaming_support(help_texts or []),
        "heuristic": ("Estimated from config.json and the weights on disk, not measured; "
                      "it errs high."),
        "applied": bool(getattr(config, "train_autopilot", False)),
    }
    if meta.get("weights_bytes"):
        plan["model"]["weights_gb"] = _tm_gb(meta["weights_bytes"])
    if not meta.get("known"):
        plan.update({
            "estimated": False, "fits": None, "knob": None,
            "num_layers": getattr(config, "train_num_layers", 8),
            "lora_rank": getattr(config, "train_lora_rank", 8),
            "batch_size": getattr(config, "train_batch_size", 1),
            "grad_checkpoint": True, "estimated_peak_gb": None,
            "reason": f"Not estimated: {meta.get('reason', 'model shape unknown')} "
                      f"The configured recipe is kept."})
        return plan

    def estimate(layers, rank, batch, gc, seq_len=seq):
        return estimate_train_memory(meta, num_layers=layers, rank=rank, batch=batch,
                                     seq=seq_len, grad_checkpoint=gc, method=method,
                                     fine_tune_type=ft)

    # The recipe as configured, so the plan also says whether a manual run fits.
    try:
        configured_seq = int(str(getattr(config, "train_seq_len", seq)).strip())
    except (TypeError, ValueError):
        configured_seq = seq
    if config is not None:
        mine = estimate(int(config.train_num_layers), int(config.train_lora_rank),
                        int(config.train_batch_size), True, configured_seq)
        plan["configured"] = {
            "num_layers": config.train_num_layers, "lora_rank": config.train_lora_rank,
            "batch_size": config.train_batch_size, "max_seq_length": configured_seq,
            "estimated_peak_gb": _tm_gb(mine["peak_bytes"]),
            "fits": mine["peak_bytes"] <= budget}

    layers_cap = meta["layers"]
    preferred_layers = min(layers_cap, 16 if ram_gb >= 32 else 8)
    layer_opts = []
    value = preferred_layers
    while value >= 2:
        layer_opts.append(value)
        value //= 2
    layer_opts = layer_opts or [max(1, layers_cap)]
    rank0 = 16 if ram_gb >= 32 else 8
    batch0 = 4 if ram_gb >= 32 else (2 if ram_gb >= 16 else 1)
    # Too large a batch on a small corpus leaves too few optimizer steps.
    batch0 = max(1, min(batch0, max(1, n_train // 8))) if n_train else 1
    batch_opts = []
    value = batch0
    while value >= 1:
        batch_opts.append(value)
        value //= 2
    candidates = [(layer_opts[0], rank0, b) for b in batch_opts]
    candidates += [(lay, rank0, 1) for lay in layer_opts[1:]]
    candidates.append((layer_opts[-1], min(rank0, 4), 1))

    chosen = None
    for layers, rank, batch in candidates:
        fast = estimate(layers, rank, batch, False)
        if fast["peak_bytes"] <= budget * 0.6:
            chosen = (layers, rank, batch, False, fast)
            break
        safe = estimate(layers, rank, batch, True)
        if safe["peak_bytes"] <= budget:
            chosen = (layers, rank, batch, True, safe)
            break

    if chosen is None:
        layers, rank, batch = candidates[-1]
        smallest = estimate(layers, rank, batch, True)
        parts = smallest["parts"]
        fixed = parts["weights"] + parts["reference"] + parts["overhead"]
        extra = ""
        if parts["reference"]:
            extra = (f" {method.upper()} also loads a {_tm_gb(parts['reference'])} GB "
                     f"reference copy of the model; TRAIN_METHOD=orpo needs none.")
        plan.update({"estimated": True, "fits": False, "num_layers": layers,
                     "lora_rank": rank, "batch_size": batch, "grad_checkpoint": True,
                     "estimated_peak_gb": _tm_gb(smallest["peak_bytes"]),
                     "breakdown_gb": {k: _tm_gb(v) for k, v in parts.items()}})
        if fixed > budget:
            plan["knob"] = "MODEL" if not parts["reference"] else "TRAIN_METHOD"
            plan["reason"] = (f"Will not fit: the weights alone need "
                              f"{_tm_gb(fixed)} GB of the ~{plan['budget_gb']} GB training "
                              f"budget on this {plan['ram_gb']} GB Mac. Choose a smaller MODEL."
                              + extra)
        else:
            plan["knob"] = "TRAIN_SEQ_LEN"
            plan["reason"] = (f"Will not fit: even at batch 1, {layers} layers and rank {rank}, "
                              f"a {seq}-token window (the corpus p95 plus margin) needs "
                              f"{plan['estimated_peak_gb']} GB of the ~{plan['budget_gb']} GB "
                              f"budget. Lower TRAIN_SEQ_LEN (longer examples are then dropped) "
                              f"or shorten SYSTEM_PROMPT." + extra)
        return plan

    layers, rank, batch, gc, est = chosen
    plan.update({
        "estimated": True, "fits": True, "knob": None, "num_layers": layers,
        "lora_rank": rank, "batch_size": batch, "grad_checkpoint": gc,
        "estimated_peak_gb": _tm_gb(est["peak_bytes"]),
        "breakdown_gb": {k: _tm_gb(v) for k, v in est["parts"].items()},
        "reason": (f"Fits: ~{_tm_gb(est['peak_bytes'])} GB of the ~{plan['budget_gb']} GB "
                   f"budget with {layers} layers, rank {rank}, batch {batch}, {seq} tokens"
                   f"{', gradient checkpointing' if gc else ''}."),
    })
    return plan


def autopilot_overrides(plan: dict) -> dict:
    """Config fields a fitting, estimated plan sets for one run."""
    if not plan.get("estimated") or not plan.get("fits"):
        return {}
    return {"train_num_layers": int(plan["num_layers"]),
            "train_lora_rank": int(plan["lora_rank"]),
            "train_batch_size": int(plan["batch_size"]),
            "train_seq_len": str(int(plan["max_seq_length"]))}


# --------------------------------------------------------------------------- #
# Preference trainer command, metrics and gate
# --------------------------------------------------------------------------- #
def write_preference_config(path: Path, config) -> Path:
    """YAML for mlx_lm_lora: the LoRA shape, and fuse switched off."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Generated by the retrain loop. Edits are overwritten on the next run.\n"
        "# fuse must stay false: mlx_lm_lora fuses by default and would write a\n"
        "# de-quantized model over the live adapter directory.\n"
        "fuse: false\n"
        "lora_parameters:\n"
        f"  rank: {int(config.train_lora_rank)}\n"
        f"  scale: {float(config.train_lora_scale)}\n"
        f"  dropout: {float(config.train_lora_dropout)}\n",
        encoding="utf-8")
    return path


def build_preference_cmd(config, method: str, help_text: str, plan: dict, *,
                         data_dir: Path, adapter_dir: Path, config_path: Path,
                         reward_file: Path | None = None,
                         rewards: list[tuple[str, float]] | None = None,
                         grad_checkpoint: bool = True) -> list[str]:
    """The mlx_lm_lora.train command line, built against its --help like SFT's."""
    spec = TRAIN_METHOD_SPECS[method]
    cmd = [sys.executable, "-m", spec.trainer_module]

    def add(names: list[str], value=None, required: bool = False) -> None:
        if not add_if_supported(cmd, help_text, names, value) and required:
            cmd.append(names[0])
            if value is not None:
                cmd.append(str(value))

    add(["--model"], config.model, True)
    add(["--train"], None, True)
    add(["--train-mode"], spec.train_mode, True)
    add(["--data"], str(data_dir), True)
    add(["--adapter-path"], str(adapter_dir), True)
    ft = (config.train_fine_tune_type or "lora").lower()
    if ft in ("lora", "dora", "full"):
        add(["--train-type", "--fine-tune-type"], ft)
    add(["--iters"], str(plan["iters"]))
    add(["--batch-size"], str(plan["batch_size"]))
    add(["--num-layers"], str(config.train_num_layers))
    add(["--learning-rate"], config.train_lr)
    if grad_checkpoint:
        add(["--grad-checkpoint"])
    add(["--max-seq-length"], config.train_seq_len)
    add(["--steps-per-eval"], str(plan["eval_every"]))
    add(["--save-every"], str(plan["save_every"]))
    add(["--val-batches"], "-1")
    add(["--beta"], str(float(config.train_beta)))
    reference = str(getattr(config, "train_reference_adapter", "") or "").strip()
    if spec.needs_reference and reference:
        add(["--reference-model-path"], str(Path(reference).expanduser())
            if Path(reference).expanduser().exists() else reference)
    if method == "grpo" and reward_file is not None and rewards:
        add(["--reward-functions-file"], str(reward_file))
        add(["--reward-functions"], ",".join(f"llm_{n}" for n, _ in rewards))
        add(["--reward-weights"], "[" + ",".join(str(float(w)) for _, w in rewards) + "]")
    cmd.extend(["--config", str(config_path)])
    return cmd


_TM_VAL_ACC_RE = re.compile(
    r"Iter\s+(\d+)\s*:\s*Val loss\s+[0-9.]+.*?Val accuracy\s+([0-9]*\.?[0-9]+)",
    re.IGNORECASE)
_TM_VAL_MARGIN_RE = re.compile(
    r"Iter\s+(\d+)\s*:\s*Val loss\s+[0-9.]+.*?Val margin\s+(-?[0-9]*\.?[0-9]+)",
    re.IGNORECASE)
_TM_VAL_LOSS_RE = re.compile(r"Iter\s+(\d+)\s*:\s*Val loss\s+(-?[0-9]*\.?[0-9]+)",
                             re.IGNORECASE)


def parse_trainer_metrics(text: str) -> dict:
    """Validation points from a preference trainer's log (dpo_trainer.py prints
    "Iter N: Val loss X, Val chosen reward .., Val rejected reward .., Val
    accuracy .., Val margin .., Val took ..s"; GRPO prints loss and time only)."""
    return {
        "val_losses": [(int(i), float(v)) for i, v in _TM_VAL_LOSS_RE.findall(text or "")],
        "val_accuracy": [(int(i), float(v)) for i, v in _TM_VAL_ACC_RE.findall(text or "")],
        "val_margin": [(int(i), float(v)) for i, v in _TM_VAL_MARGIN_RE.findall(text or "")],
    }


def gate_preference_run(method: str, metrics: dict, config,
                        best: tuple[int, float] | None = None) -> dict:
    """Decide whether a preference run may be promoted.

    DPO and ORPO losses are comparable between the first evaluation (taken
    before any update) and the last: lower means the chosen answers are
    ranked above the rejected ones more strongly. GRPO's validation "loss" is
    a policy-gradient surrogate over group-normalised advantages, not a score
    that falls as the model improves, so it gates nothing. Without a
    comparable metric the run is refused unless TRAIN_PROMOTE_UNGATED=1.
    """
    spec = TRAIN_METHOD_SPECS[method]
    ungated = bool(getattr(config, "train_promote_ungated", False))
    losses = metrics.get("val_losses") or []
    verdict = {"method": method, "metric": spec.metric, "baseline": None, "final": None,
               "ungated": False, "ok": False, "message": ""}
    if spec.metric == "val_loss" and len(losses) >= 2:
        baseline = losses[0][1]
        final = best[1] if best else losses[-1][1]
        verdict.update(baseline=baseline, final=final)
        if config.train_val_check and final > baseline * (1.0 + config.train_val_tolerance):
            verdict["message"] = (f"Held-out {spec.label} loss got worse ({baseline:.4f} -> "
                                  f"{final:.4f}); the adapter was discarded. Collect more "
                                  f"feedback, or lower TRAIN_EPOCHS / TRAIN_LR / TRAIN_BETA.")
            return verdict
        verdict.update(ok=True, message=f"Held-out {spec.label} loss {baseline:.4f} -> {final:.4f}.")
        return verdict
    if ungated:
        verdict.update(ok=True, ungated=True, message=(
            f"{spec.label} reported no comparable metric; promoted unverified because "
            f"TRAIN_PROMOTE_UNGATED=1."))
        return verdict
    if spec.metric is None:
        why = (f"{spec.label} reports no metric comparable across a run (its validation "
               f"loss is a policy-gradient surrogate)")
    else:
        why = (f"{spec.label} reported fewer than two validation points, so the run could "
               f"not be compared with its baseline")
    verdict["message"] = (why + "; the adapter was discarded. Set TRAIN_PROMOTE_UNGATED=1 "
                                "to promote such runs unverified.")
    return verdict


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
def register_training(app, ctx) -> None:
    """GET /api/train/methods, GET /api/train/plan, POST /api/train/export (admin)."""
    from fastapi.responses import JSONResponse
    ADMIN = ctx.ADMIN

    def _manager():
        rm = ctx.retrain_manager
        return rm if rm is not None and hasattr(rm, "method_catalog") else None

    def _method(value: str) -> str:
        return (str(value or "").strip().lower() or train_method_name(ctx.config))

    @app.get("/api/train/methods")
    async def train_methods_list(_a: dict = ADMIN):
        rm = _manager()
        if rm is None:
            return JSONResponse({"error": "Training is not available in this process."}, 503)
        methods = await asyncio.to_thread(rm.method_catalog)
        return {"methods": methods, "configured": train_method_name(ctx.config),
                "autopilot": bool(getattr(ctx.config, "train_autopilot", False)),
                "promote_ungated": bool(getattr(ctx.config, "train_promote_ungated", False))}

    @app.get("/api/train/plan")
    async def train_plan(method: str = "", _a: dict = ADMIN):
        rm = _manager()
        if rm is None:
            return JSONResponse({"error": "Training is not available in this process."}, 503)
        name = _method(method)
        if name not in TRAIN_METHODS:
            return JSONResponse({"error": f"Unknown method {name!r}; choose one of "
                                          f"{', '.join(TRAIN_METHODS)}."}, 400)
        try:
            return await asyncio.to_thread(rm.autopilot_plan, name)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": redact_text(f"{type(exc).__name__}: {exc}")}, 500)

    @app.post("/api/train/export")
    async def train_export(method: str = "", _a: dict = ADMIN):
        rm = _manager()
        if rm is None:
            return JSONResponse({"error": "Training is not available in this process."}, 503)
        name = _method(method)
        if name not in TRAIN_METHODS:
            return JSONResponse({"error": f"Unknown method {name!r}; choose one of "
                                          f"{', '.join(TRAIN_METHODS)}."}, 400)
        try:
            result = await asyncio.to_thread(rm.export_method, name)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": redact_text(f"{type(exc).__name__}: {exc}")}, 500)
        if result.get("busy"):
            return JSONResponse({"error": "A training run is in progress; export after it."}, 409)
        log_event(get_logger("training"), logging.INFO, "training.export",
                  method=name, examples=result.get("counts", {}).get("examples"))
        return result


def train_plan_cli(argv: list[str]) -> int:
    """`deploy.py train-plan [--method dpo]`: print the autopilot plan as JSON.

    Read-only: reads the model's config.json and the feedback database, never
    loads a model and never trains.
    """
    import argparse
    parser = argparse.ArgumentParser(prog="deploy.py train-plan",
                                     description="Print the training autopilot's plan.")
    parser.add_argument("--method", default=None, choices=list(TRAIN_METHODS))
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)
    config = Config(**({"model": args.model} if args.model else {}))
    method = args.method or train_method_name(config)
    stats = {"count": 0, "p50": 0, "p95": 0, "max": 0}
    if DB_PATH.exists():
        db = Database(DB_PATH)
        try:
            stats = corpus_token_stats(db, config, method)
        finally:
            db.close()
    plan = plan_autopilot(TOTAL_RAM_GB, model_train_meta(config.model), stats,
                          method=method, config=config,
                          help_texts=[help_cmd("mlx_lm.lora")])
    print(json.dumps(plan, indent=2, default=str))
    return 0 if plan.get("fits") is not False else 2


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    '_TM_REPO_DATA_DIR',
    'GRPO_REWARD_NAMES',
    'PREFERENCE_TRAINER_MODULE',
    'TRAIN_METHODS',
    'TRAIN_METHOD_SPECS',
    'TrainMethodSpec',
    'autopilot_overrides',
    'build_preference_cmd',
    'corpus_token_stats',
    'dataclass_replace_safe',
    'describe_train_methods',
    'estimate_train_memory',
    'export_preferences',
    'train_data_root',
    'train_replay_file',
    'train_sft_dir',
    'gate_preference_run',
    'grpo_score',
    'layer_streaming_support',
    'model_train_meta',
    'parse_grpo_reward_spec',
    'parse_trainer_metrics',
    'plan_autopilot',
    'preference_trainer_help',
    'preference_trainer_problem',
    'reference_model_problem',
    'register_training',
    'train_method_name',
    'train_plan_cli',
    'write_grpo_reward_file',
    'write_preference_config',
    '_tm_model_dir',
    '_tm_row_tokens',
    '_tm_split',
]
