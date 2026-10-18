"""Model catalogue: per-model metadata, aliases, RAM fit and the benchmark files.

The switcher used to know three things about a model: its repo id, whether the
weights were on disk, and whether it was the one loaded. Everything else (how
big it is, whether a Mac this size can hold it, whether it thinks, which
tool-call format it speaks, which small sibling can draft for it) lived in
comments next to DEFAULT_MODEL_CATALOG, where neither the UI nor the
benchmark could read it. Picking a model was therefore guesswork followed by a
multi-gigabyte download that sometimes did not fit.

This module is the readable version of those comments, plus three things that
need them:

  * aliases ("fast", "coder", "large", and the operator's own MODEL_ALIASES),
    resolved to a repo id BEFORE anything else sees the name. That ordering is
    the point: ClusterRouter.classify matches LARGE_MODEL_MARKERS ("14B",
    "32B", ...) against the model name, so an unresolved "large" would never
    route to the Studio and a user alias called "my14b" pointing at a 3B
    would route there for nothing;
  * draft_for(), which backends.py looks up by name for DRAFT_MODEL=auto;
  * the benchmark result and recommendation files (diagnostics.py measures,
    this module stores, core.py's RAM-tier defaults read the recommendation).

It deliberately imports nothing but core: sysutil (model_catalog) and config
read it, so it must load before both. Config is read duck-typed.

Every size, RAM floor and context figure below is approximate. Entries marked
approximate=True also carry a repo id or capability flag that was not verified
on this machine; the benchmark (deploy.py bench --sweep) is how a number
becomes a measurement.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import *  # noqa: F401,F403

# Mac unified-memory sizes, used to round a computed RAM floor up to a machine
# that exists. A floor of "9.3GB" means nothing to someone choosing a Mac.
_CAT_MAC_RAM_SIZES = (8, 16, 18, 24, 32, 36, 48, 64, 96, 128, 192, 256, 512)
# macOS lets the GPU wire roughly two thirds of RAM on smaller machines by
# default (recommendedMaxWorkingSetSize); the weights plus the KV cache must sit
# inside that, and the web process, the OS and a training run share the rest.
_CAT_GPU_SHARE = 0.66
# Typical KV cache + runtime overhead at the context sizes this app uses.
_CAT_RUNTIME_GB = 1.0

RECOMMENDATION_FILE = "recommendation.json"
BENCH_DIR_NAME = "bench"


def _cat_min_ram(size_gb: float) -> int:
    """Smallest Mac RAM size whose default GPU share holds the weights + runtime."""
    need = (float(size_gb) + _CAT_RUNTIME_GB) / _CAT_GPU_SHARE
    for size in _CAT_MAC_RAM_SIZES:
        if size >= need:
            return size
    return _CAT_MAC_RAM_SIZES[-1]


def _cat(model_id: str, family: str, params_b: float, quant: str, size_gb: float,
         context_max: int, *, thinking: bool = False, tool_format: str = "hermes",
         vision: bool = False, draft: str | None = None, notes: str = "",
         approximate: bool = False, active_params_b: float | None = None) -> dict:
    return {
        "id": model_id,
        "family": family,
        "params_b": params_b,
        "active_params_b": active_params_b if active_params_b is not None else params_b,
        "quant": quant,
        "approx_size_gb": size_gb,
        "min_ram_gb": _cat_min_ram(size_gb),
        "context_max": context_max,
        "thinking": thinking,
        "tool_format": tool_format,
        "vision": vision,
        "draft": draft,
        "notes": notes,
        "approximate": approximate,
    }


_CAT_QC = "mlx-community/Qwen2.5-Coder-"
_CAT_Q = "mlx-community/Qwen2.5-"
_CAT_LLAMA_DRAFT = "mlx-community/Llama-3.2-1B-Instruct-4bit"

# Metadata for every DEFAULT_MODEL_CATALOG id, the RAM-tier defaults
# (core._default_model_for_ram) and a few widely used MLX models. Order is
# irrelevant; lookups are by id. tool_format names a toolparse format, or
# "auto" when the family has no settled one.
MODEL_METADATA: tuple[dict, ...] = (
    # Qwen2.5-Coder: one tokenizer across sizes, so any smaller sibling can draft.
    _cat(_CAT_QC + "32B-Instruct-4bit", "qwen2.5-coder", 32.8, "4bit", 18.4, 32768,
         draft=_CAT_QC + "1.5B-Instruct-4bit",
         notes="RAM-tier default at 48GB and up. Strongest local coder here."),
    _cat(_CAT_QC + "14B-Instruct-4bit", "qwen2.5-coder", 14.7, "4bit", 8.3, 32768,
         draft=_CAT_QC + "0.5B-Instruct-4bit",
         notes="RAM-tier default at 24GB and up."),
    _cat(_CAT_QC + "7B-Instruct-4bit", "qwen2.5-coder", 7.6, "4bit", 4.3, 32768,
         draft=_CAT_QC + "0.5B-Instruct-4bit",
         notes="Best code quality that loads on 8GB, but only after raising the GPU "
               "wired limit; inference only there, do not retrain against it."),
    _cat(_CAT_QC + "3B-Instruct-4bit", "qwen2.5-coder", 3.1, "4bit", 1.9, 32768,
         notes="The 8GB default. Trains fine. A draft model barely pays at this size."),
    _cat(_CAT_QC + "1.5B-Instruct-4bit", "qwen2.5-coder", 1.5, "4bit", 1.0, 32768,
         notes="Fast completion-style coder; also the draft for the 32B."),
    _cat(_CAT_QC + "0.5B-Instruct-4bit", "qwen2.5-coder", 0.5, "4bit", 0.3, 32768,
         notes="Draft model for the larger Qwen2.5-Coder sizes; too weak to serve.",
         approximate=True),
    # Qwen3.5 (thinking). Figures and tool format not verified here.
    _cat("mlx-community/Qwen3.5-4B-OptiQ-4bit", "qwen3.5", 4.0, "mixed-4bit", 2.5, 32768,
         thinking=True, tool_format="auto", approximate=True,
         notes="Mixed precision, agent-calibrated quantization."),
    _cat("mlx-community/Qwen3.5-4B-MLX-4bit", "qwen3.5", 4.0, "4bit", 2.5, 32768,
         thinking=True, tool_format="auto", approximate=True,
         notes="Stock uniform 4-bit."),
    # Qwen2.5 general purpose.
    _cat(_CAT_Q + "3B-Instruct-4bit", "qwen2.5", 3.1, "4bit", 1.7, 32768,
         draft=_CAT_Q + "0.5B-Instruct-4bit", notes="No thinking mode."),
    _cat(_CAT_Q + "1.5B-Instruct-8bit", "qwen2.5", 1.5, "8bit", 1.6, 32768,
         notes="8-bit: better format adherence than the 4-bit at the same size."),
    _cat(_CAT_Q + "1.5B-Instruct-4bit", "qwen2.5", 1.5, "4bit", 0.9, 32768),
    _cat(_CAT_Q + "0.5B-Instruct-4bit", "qwen2.5", 0.5, "4bit", 0.4, 32768,
         notes="Fastest, weakest at tools."),
    # Llama 3.x: 3.1 and 3.2 share a tokenizer, so the 1B drafts for both.
    _cat("mlx-community/Llama-3.2-3B-Instruct-4bit", "llama3.2", 3.2, "4bit", 1.8, 131072,
         tool_format="llama3", draft=_CAT_LLAMA_DRAFT),
    _cat(_CAT_LLAMA_DRAFT, "llama3.2", 1.2, "4bit", 0.8, 131072, tool_format="llama3"),
    _cat("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", "llama3.1", 8.0, "4bit", 4.5,
         131072, tool_format="llama3", draft=_CAT_LLAMA_DRAFT, approximate=True),
    # Widely used current MLX models (repo ids and figures not verified here).
    _cat("mlx-community/Qwen3-8B-4bit", "qwen3", 8.2, "4bit", 4.6, 32768, thinking=True,
         draft="mlx-community/Qwen3-0.6B-4bit", approximate=True),
    _cat("mlx-community/Qwen3-14B-4bit", "qwen3", 14.8, "4bit", 8.3, 32768, thinking=True,
         draft="mlx-community/Qwen3-0.6B-4bit", approximate=True),
    _cat("mlx-community/Qwen3-0.6B-4bit", "qwen3", 0.6, "4bit", 0.4, 32768, thinking=True,
         notes="Draft model for the larger Qwen3 sizes.", approximate=True),
    _cat("mlx-community/Qwen3-30B-A3B-4bit", "qwen3-moe", 30.5, "4bit", 17.2, 32768,
         thinking=True, active_params_b=3.3, approximate=True,
         notes="Mixture of experts: decodes about as fast as a 3B, needs the memory "
               "of a 30B."),
    _cat("mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit", "qwen3-coder", 30.5, "4bit",
         17.2, 262144, tool_format="qwen3_coder", active_params_b=3.3, approximate=True,
         notes="Mixture-of-experts coder with the XML <function=...> tool format."),
    _cat("mlx-community/gemma-3-4b-it-4bit", "gemma3", 4.3, "4bit", 3.0, 131072,
         tool_format="gemma", vision=True, approximate=True,
         notes="Image input needs mlx-vlm; mlx-lm serves it as text only."),
    _cat("mlx-community/gemma-3-12b-it-4bit", "gemma3", 12.2, "4bit", 8.0, 131072,
         tool_format="gemma", vision=True, approximate=True,
         notes="Image input needs mlx-vlm; mlx-lm serves it as text only."),
    _cat("mlx-community/gemma-3-27b-it-4bit", "gemma3", 27.0, "4bit", 16.5, 131072,
         tool_format="gemma", vision=True, approximate=True,
         notes="Image input needs mlx-vlm; mlx-lm serves it as text only."),
    # Vision-language models for MODEL_BACKEND=mlx-vlm (vision.py). The 3B
    # Qwen2.5-VL and the Gemma 3n repo ids appear in mlx-vlm 0.7.8's README; the
    # others follow mlx-community naming. Sizes are estimates including the
    # vision tower.
    _cat("mlx-community/Qwen2.5-VL-3B-Instruct-4bit", "qwen2.5-vl", 3.8, "4bit", 2.4, 32768,
         tool_format="auto", vision=True, approximate=True,
         notes="Image input needs mlx-vlm; the smallest Qwen vision model that reads "
               "documents and screenshots well."),
    _cat("mlx-community/Qwen2.5-VL-7B-Instruct-4bit", "qwen2.5-vl", 8.3, "4bit", 5.0, 32768,
         tool_format="auto", vision=True, approximate=True,
         notes="Image input needs mlx-vlm."),
    _cat("mlx-community/Qwen3-VL-4B-Instruct-4bit", "qwen3-vl", 4.4, "4bit", 2.9, 262144,
         tool_format="auto", vision=True, approximate=True,
         notes="Image input needs mlx-vlm."),
    _cat("mlx-community/gemma-3n-E2B-it-4bit", "gemma3n", 5.4, "4bit", 3.0, 32768,
         tool_format="gemma", vision=True, approximate=True, active_params_b=2.0,
         notes="Image (and audio) input needs mlx-vlm."),
    _cat("mlx-community/Mistral-7B-Instruct-v0.3-4bit", "mistral", 7.25, "4bit", 4.1,
         32768, tool_format="mistral", approximate=True),
    _cat("mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit", "deepseek-r1-distill", 7.6,
         "4bit", 4.3, 131072, thinking=True, tool_format="auto", approximate=True,
         notes="Long visible reasoning; weak at tool calls."),
)

_CAT_BY_ID: dict[str, dict] = {entry["id"]: entry for entry in MODEL_METADATA}

# Built-in aliases. Static on purpose: "coder" means the same repo on every
# machine, so a config that says MODEL_ID=coder is reproducible; fits_in_ram and
# the benchmark say whether this machine can hold it.
BUILTIN_MODEL_ALIASES: dict[str, str] = {
    "tiny": _CAT_Q + "0.5B-Instruct-4bit",
    "fast": _CAT_Q + "1.5B-Instruct-4bit",
    "balanced": _CAT_QC + "3B-Instruct-4bit",
    "coder": _CAT_QC + "7B-Instruct-4bit",
    "reasoning": "mlx-community/Qwen3.5-4B-MLX-4bit",
    "vision": "mlx-community/gemma-3-4b-it-4bit",
    "large": _CAT_QC + "32B-Instruct-4bit",
}

_CAT_PARAMS_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")
_CAT_BITS_RE = re.compile(r"(\d+)\s*-?\s*bit", re.IGNORECASE)


def parse_model_aliases(text: Any) -> tuple[dict[str, str], str]:
    """({alias: model id}, error). error is "" when the value is usable.

    The value is a JSON object of non-empty strings. Names are matched case
    insensitively, and a name containing "/" is refused because a repo id is
    never looked up as an alias (it would be unreachable).
    """
    raw = str(text or "").strip()
    if not raw:
        return {}, ""
    try:
        data = json.loads(raw)
    except ValueError as exc:
        return {}, f"not valid JSON ({exc})"
    if not isinstance(data, dict):
        return {}, "expected a JSON object of alias -> model id"
    out: dict[str, str] = {}
    for name, target in data.items():
        if not isinstance(name, str) or not name.strip() or "/" in name:
            return {}, f"alias name {name!r} must be a non-empty string without '/'"
        if not isinstance(target, str) or not target.strip():
            return {}, f"alias {name!r} must map to a non-empty model id string"
        out[name.strip().lower()] = target.strip()
    return out, ""


def _cat_saved_aliases() -> str:
    """MODEL_ALIASES as saved from the Settings panel, or "".

    cli build_config runs before Config.load_saved(), so an alias defined only
    in the UI would otherwise not resolve for MODEL_ID / --model at boot.
    """
    try:
        saved = json.loads((DATA_DIR / "settings.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return ""
    value = saved.get("model_aliases") if isinstance(saved, dict) else None
    return value if isinstance(value, str) else ""


def user_model_aliases(config: Any = None) -> dict[str, str]:
    """The operator's aliases (config.model_aliases, else MODEL_ALIASES)."""
    text = (getattr(config, "model_aliases", None) if config is not None
            else os.environ.get("MODEL_ALIASES", ""))
    aliases, error = parse_model_aliases(text)
    if error:
        # Config.apply already refused it; this only covers a raw env value.
        log(f"Ignoring MODEL_ALIASES: {error}", logging.WARNING)
        return {}
    return aliases


def model_aliases(config: Any = None) -> dict[str, dict]:
    """Every alias with where it came from: {name: {"id", "source"}}.

    User aliases override built-ins of the same name.
    """
    merged = {name: {"id": target, "source": "builtin"}
              for name, target in BUILTIN_MODEL_ALIASES.items()}
    for name, target in user_model_aliases(config).items():
        merged[name] = {"id": resolve_model(target, None) if target.lower() in
                        BUILTIN_MODEL_ALIASES else target, "source": "user"}
    return merged


def resolve_model(name: Any, config: Any = None) -> str:
    """The repo id (or path) an alias stands for; anything else unchanged.

    A repo id ("org/name") or an existing directory is never treated as an
    alias. A user alias may point at a built-in one ("chat" -> "fast"); chains
    stop after a few hops so a cycle cannot hang a request.
    """
    current = str(name or "").strip()
    if not current:
        return current
    user = user_model_aliases(config)
    for _ in range(4):
        if "/" in current or Path(current).expanduser().is_dir():
            return current
        key = current.lower()
        target = user.get(key) or BUILTIN_MODEL_ALIASES.get(key)
        if not target or target == current:
            return current
        current = target
    return current


def alias_of(model_id: str, config: Any = None) -> str:
    """The first alias (user aliases first) that resolves to model_id, or ""."""
    names = model_aliases(config)
    user_first = sorted(names.items(), key=lambda item: item[1]["source"] != "user")
    for name, info in user_first:
        if info["id"] == model_id:
            return name
    return ""


def catalog_cli_fields(args: Any) -> dict:
    """Config fields cli.build_config must override when --model / MODEL_ID is an alias.

    Returns {} when the name is not an alias, so a plain repo id goes through
    untouched. config.model ends up the resolved id (what the router, the
    manager and every log line see); config.model_alias keeps the name for
    display only.
    """
    requested = str(getattr(args, "model", "") or "").strip()
    if not requested or "/" in requested:
        return {}
    text = _cat_saved_aliases() or os.environ.get("MODEL_ALIASES", "")
    probe = type("_CatAliasSource", (), {"model_aliases": text})()
    resolved = resolve_model(requested, probe)
    if resolved == requested:
        return {}
    return {"model": resolved, "model_alias": requested.lower()}


def _cat_guess(model_id: str) -> dict | None:
    """Rough metadata from the repo name for a model the table does not list."""
    name = model_id.rsplit("/", 1)[-1]
    params = _CAT_PARAMS_RE.search(name)
    if not params:
        return None
    params_b = float(params.group(1))
    bits_match = _CAT_BITS_RE.search(name)
    bits = int(bits_match.group(1)) if bits_match else 16
    size_gb = round(params_b * bits / 8 * 1.1, 1)
    return {
        "id": model_id, "family": "", "params_b": params_b, "active_params_b": params_b,
        "quant": f"{bits}bit" if bits_match else "", "approx_size_gb": size_gb,
        "min_ram_gb": _cat_min_ram(size_gb), "context_max": None, "thinking": None,
        "tool_format": "auto", "vision": None, "draft": None,
        "notes": "Estimated from the repo name; not in the catalogue.",
        "approximate": True, "guessed": True,
    }


def model_metadata(model_id: str, *, guess: bool = True) -> dict | None:
    """Catalogue metadata for a repo id (a copy), a name-based guess, or None."""
    entry = _CAT_BY_ID.get(str(model_id or ""))
    if entry is not None:
        out = dict(entry)
        out["aliases"] = sorted(n for n, t in BUILTIN_MODEL_ALIASES.items() if t == entry["id"])
        out["alias"] = out["aliases"][0] if out["aliases"] else ""
        return out
    return _cat_guess(str(model_id or "")) if guess else None


def draft_for(model_id: str) -> str | None:
    """A smaller same-tokenizer model for speculative decoding, or None.

    backends.py (DRAFT_MODEL=auto) looks this name up at call time.
    """
    entry = _CAT_BY_ID.get(str(model_id or ""))
    draft = entry.get("draft") if entry else None
    return draft or None


def fits_in_ram(entry: dict | str | None, ram_gb: float | None = None,
                headroom_gb: float = 0.0) -> bool | None:
    """Whether a model's RAM floor fits this machine; None when unknown."""
    if isinstance(entry, str):
        entry = model_metadata(entry)
    if not entry or not entry.get("min_ram_gb"):
        return None
    ram = TOTAL_RAM_GB if ram_gb is None else float(ram_gb)
    return ram - max(0.0, float(headroom_gb or 0.0)) >= float(entry["min_ram_gb"])


def quality_key(model_id: str) -> tuple:
    """Sort key: bigger and less quantized is better. Higher sorts later."""
    meta = model_metadata(model_id) or {}
    bits = _CAT_BITS_RE.search(str(meta.get("quant") or ""))
    return (float(meta.get("params_b") or 0.0), int(bits.group(1)) if bits else 4)


# --------------------------------------------------------------------------- #
# Benchmark results and the recommendation. diagnostics.py measures; these
# only read and write files, so the API, the CLI and core's defaults agree on
# where they live.
# --------------------------------------------------------------------------- #

def _cat_dir(data_dir: Path | None) -> Path:
    return Path(data_dir) if data_dir is not None else DATA_DIR


def bench_dir(data_dir: Path | None = None) -> Path:
    return _cat_dir(data_dir) / BENCH_DIR_NAME


def save_bench_run(doc: dict, data_dir: Path | None = None) -> Path:
    """Write one sweep to DATA_DIR/bench/<UTC timestamp>.json; returns the path."""
    folder = bench_dir(data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = folder / f"{stamp}.json"
    n = 1
    while path.exists():
        n += 1
        path = folder / f"{stamp}-{n}.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def list_bench_runs(data_dir: Path | None = None) -> list[Path]:
    """Saved sweeps, newest first."""
    folder = bench_dir(data_dir)
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.glob("*.json") if p.is_file()),
                  key=lambda p: p.name, reverse=True)


def load_bench_run(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log(f"Could not read benchmark results {path}: {exc}", logging.WARNING)
        return None
    return data if isinstance(data, dict) else None


def latest_bench_run(data_dir: Path | None = None) -> dict | None:
    for path in list_bench_runs(data_dir):
        doc = load_bench_run(path)
        if doc is not None:
            doc.setdefault("file", path.name)
            return doc
    return None


def read_recommendation(data_dir: Path | None = None,
                        ram_gb: float | None = None) -> dict | None:
    """The applied recommendation (DATA_DIR/recommendation.json), or None.

    With ram_gb, a file stamped for a different machine size reads as None,
    the same rule core._ram_recommendation applies to the defaults.
    """
    path = _cat_dir(data_dir) / RECOMMENDATION_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001
        log(f"Ignoring unreadable {path}: {exc}", logging.WARNING)
        return None
    if not isinstance(data, dict) or not data.get("model_id"):
        return None
    if ram_gb is not None:
        try:
            if abs(float(data.get("ram_gb") or 0) - float(ram_gb)) >= 1.0:
                return None
        except (TypeError, ValueError):
            return None
    return data


def write_recommendation(rec: dict, data_dir: Path | None = None,
                         ram_gb: float | None = None) -> Path:
    """Persist a recommendation for core's RAM-tier defaults.

    Stamped with this machine's RAM: core ignores a file written on a machine
    of a different size (a copied data directory must not pin a 32B onto an
    8GB Mac). Only the model id and context are read back as defaults, and
    only when MODEL_ID / CONTEXT_SIZE are unset; settings.json is never
    touched, so the environment keeps the last word.
    """
    if not rec or not rec.get("model_id"):
        raise ValueError("nothing to apply: the recommendation has no model")
    doc = dict(rec)
    doc["ram_gb"] = round(float(TOTAL_RAM_GB if ram_gb is None else ram_gb), 2)
    doc["applied_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = _cat_dir(data_dir) / RECOMMENDATION_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'BENCH_DIR_NAME',
    'BUILTIN_MODEL_ALIASES',
    'MODEL_METADATA',
    'RECOMMENDATION_FILE',
    '_cat_guess',
    '_cat_min_ram',
    '_cat_saved_aliases',
    'alias_of',
    'bench_dir',
    'catalog_cli_fields',
    'draft_for',
    'fits_in_ram',
    'latest_bench_run',
    'list_bench_runs',
    'load_bench_run',
    'model_aliases',
    'model_metadata',
    'parse_model_aliases',
    'quality_key',
    'read_recommendation',
    'resolve_model',
    'save_bench_run',
    'user_model_aliases',
    'write_recommendation',
]
