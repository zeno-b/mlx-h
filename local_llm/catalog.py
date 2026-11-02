"""Model catalogue: what a model is, aliases, RAM fit and the benchmark files.

No model is built in. The served model is MODEL_ID in .env (the web UI asks for
one when it is unset and writes the answer there), MODEL_CATALOG lists extra ids
for the switcher, and MODEL_ALIASES gives them short names. Everything else the
switcher, the benchmark and the prompt builders want to know about a model --
how big it is, whether a Mac this size can hold it, whether it thinks, which
tool-call format its chat template speaks, whether it has a vision tower, which
downloaded sibling can draft for it -- is read from the model's OWN files once
they are on disk (config.json, the chat template, the safetensors sizes), and
estimated from the repo name before that. A model nobody listed is therefore
described exactly as well as one somebody did.

Three things here depend on that:

  * aliases, resolved to a repo id BEFORE anything else sees the name. That
    ordering is the point: ClusterRouter.classify matches LARGE_MODEL_MARKERS
    ("14B", "32B", ...) against the model name, so an unresolved alias would
    never route to the Studio, and an alias called "my14b" pointing at a 3B
    would route there for nothing;
  * draft_for(), which backends.py looks up by name for DRAFT_MODEL=auto;
  * the benchmark result and recommendation files (diagnostics.py measures,
    this module stores, `recommend --apply` writes the winner into .env).

It deliberately imports nothing but core: sysutil (model_catalog) and config
read it, so it must load before both. Config is read duck-typed.

Sizes and RAM floors are estimates: the safetensors bytes are exact, the GPU
share and runtime overhead are not. The benchmark (deploy.py bench --sweep) is
how an estimate becomes a measurement.
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

# "3B", "1.5b", "135M" in a repo name; not "8Bit" (a letter follows) or "v0.2".
_CAT_PARAMS_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*([bBmM])(?![a-zA-Z])")
_CAT_BITS_RE = re.compile(r"(\d+)\s*-?\s*bit", re.IGNORECASE)
# Chat-template markers -> the toolparse format that template speaks. First
# match wins, so the specific formats come before the generic <tool_call>.
_CAT_TOOL_MARKERS: tuple[tuple[str, str], ...] = (
    ("<function=", "qwen3_coder"),
    ("<|tool_calls_section_begin|>", "kimi"),
    ("tool▁calls▁begin", "deepseek"),
    ("<arg_key>", "glm"),
    ("<|channel|>", "harmony"),
    ("[TOOL_CALLS]", "mistral"),
    ("<|python_tag|>", "llama3"),
    ("tool_code", "gemma"),
    ("<tool_call>", "hermes"),
)
# Positive results only, keyed by the files they were read from: a model that
# is not downloaded yet must be read again once it is, and a re-download
# (a new snapshot or a rewritten config.json) changes the key.
_CAT_LOCAL_CACHE: dict[tuple[str, float], dict] = {}


def _cat_min_ram(size_gb: float) -> int:
    """Smallest Mac RAM size whose default GPU share holds the weights + runtime."""
    need = (float(size_gb) + _CAT_RUNTIME_GB) / _CAT_GPU_SHARE
    for size in _CAT_MAC_RAM_SIZES:
        if size >= need:
            return size
    return _CAT_MAC_RAM_SIZES[-1]


# --------------------------------------------------------------------------- #
# Downloaded models
# --------------------------------------------------------------------------- #

def _cat_repo_snapshot(repo: Path) -> Path | None:
    """The snapshot folder a huggingface_hub repo folder serves, or None."""
    snapshots = repo / "snapshots"
    ref = repo / "refs" / "main"
    try:
        pinned = snapshots / ref.read_text(encoding="utf-8").strip()
        if (pinned / "config.json").is_file():
            return pinned
    except OSError:
        pass
    try:
        found = [p for p in snapshots.iterdir() if (p / "config.json").is_file()]
    except OSError:
        return None
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def model_snapshot_dir(model_id: str) -> Path | None:
    """Where a model's files are on disk (a local folder or a cached snapshot), or None."""
    name = str(model_id or "").strip()
    if not name:
        return None
    local = Path(name).expanduser()
    if local.is_dir():
        return local if (local / "config.json").is_file() else None
    if "/" not in name:
        return None
    return _cat_repo_snapshot(hf_cache_dir() / ("models--" + name.replace("/", "--")))


def _cat_weight_files(folder: Path) -> list[Path]:
    try:
        return sorted(folder.glob("*.safetensors"))
    except OSError:
        return []


def downloaded_models() -> list[str]:
    """Repo ids in the Hugging Face cache that hold a complete language model.

    Complete means a config.json, safetensors weights and a tokenizer, which
    leaves out half-finished downloads and repos that are only a tokenizer or
    a GGUF file mlx cannot load.
    """
    out: list[str] = []
    try:
        repos = sorted(hf_cache_dir().glob("models--*"))
    except OSError:
        return out
    for repo in repos:
        snap = _cat_repo_snapshot(repo)
        if snap is None or not _cat_weight_files(snap):
            continue
        if not ((snap / "tokenizer.json").exists() or (snap / "tokenizer_config.json").exists()):
            continue
        out.append(repo.name[len("models--"):].replace("--", "/"))
    return out


def _cat_read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _cat_chat_template(folder: Path) -> str:
    """The chat template text a model ships, wherever this repo keeps it."""
    template = _cat_read_json(folder / "tokenizer_config.json").get("chat_template")
    if isinstance(template, list):
        template = " ".join(str(t.get("template") or "") for t in template if isinstance(t, dict))
    if not template:
        template = _cat_read_json(folder / "chat_template.json").get("chat_template")
    if not template:
        try:
            template = (folder / "chat_template.jinja").read_text(encoding="utf-8")
        except OSError:
            template = ""
    return str(template or "")


def _cat_tool_format(template: str) -> str:
    for marker, name in _CAT_TOOL_MARKERS:
        if marker in template:
            return name
    return "auto"


def _cat_params_from_name(model_id: str) -> float | None:
    match = _CAT_PARAMS_RE.search(str(model_id or "").rsplit("/", 1)[-1])
    if not match:
        return None
    value = float(match.group(1))
    return round(value / 1000.0, 3) if match.group(2) in "mM" else value


def _cat_local_metadata(model_id: str) -> dict | None:
    """Metadata read from a downloaded model's own files, or None when not on disk."""
    folder = model_snapshot_dir(model_id)
    if folder is None:
        return None
    try:
        key = (str(folder), (folder / "config.json").stat().st_mtime)
    except OSError:
        return None
    cached = _CAT_LOCAL_CACHE.get(key)
    if cached is not None:
        return dict(cached)
    raw = _cat_read_json(folder / "config.json")
    text_cfg = raw.get("text_config") if isinstance(raw.get("text_config"), dict) else {}
    cfg = {**raw, **text_cfg}
    quant = raw.get("quantization") or raw.get("quantization_config") or {}
    bits = quant.get("bits") if isinstance(quant, dict) else None
    if not bits:
        named = _CAT_BITS_RE.search(str(model_id).rsplit("/", 1)[-1])
        bits = int(named.group(1)) if named else None
    weights = _cat_weight_files(folder)
    size_bytes = 0
    for path in weights:
        try:
            size_bytes += path.stat().st_size      # follows the snapshot symlink to its blob
        except OSError:
            pass
    size_gb = round(size_bytes / 1024 ** 3, 2) if size_bytes else None
    params_b = _cat_params_from_name(model_id)
    estimated_params = False
    if params_b is None and size_bytes and bits:
        # Quantized weights carry scales and biases too (~1/8 extra at group 64).
        params_b = round(size_bytes * 8 / float(bits) / 1.125 / 1e9, 2)
        estimated_params = True
    template = _cat_chat_template(folder)
    context = 0
    for name in ("max_position_embeddings", "max_sequence_length", "seq_length", "n_positions"):
        try:
            context = int(cfg.get(name) or 0)
        except (TypeError, ValueError):
            context = 0
        if context:
            break
    entry = {
        "id": str(model_id),
        "family": str(cfg.get("model_type") or ""),
        "params_b": params_b,
        "active_params_b": params_b,
        "quant": f"{int(bits)}bit" if bits else "",
        "approx_size_gb": size_gb,
        "min_ram_gb": _cat_min_ram(size_gb) if size_gb else None,
        "context_max": context or None,
        "thinking": ("<think>" in template or "enable_thinking" in template) if template else None,
        "tool_format": _cat_tool_format(template) if template else "auto",
        "vision": isinstance(raw.get("vision_config"), dict),
        "vocab_size": cfg.get("vocab_size"),
        "draft": None,
        "notes": "Read from the downloaded files" + (
            "; the parameter count is estimated from the weight size." if estimated_params else "."),
        "approximate": estimated_params,
        "guessed": False,
        "local": True,
    }
    _CAT_LOCAL_CACHE[key] = entry
    return dict(entry)


def _cat_guess(model_id: str) -> dict | None:
    """Rough metadata from the repo name for a model that is not downloaded yet."""
    params_b = _cat_params_from_name(model_id)
    if params_b is None:
        return None
    bits_match = _CAT_BITS_RE.search(str(model_id).rsplit("/", 1)[-1])
    bits = int(bits_match.group(1)) if bits_match else 16
    size_gb = round(params_b * bits / 8 * 1.1, 1)
    return {
        "id": model_id, "family": "", "params_b": params_b, "active_params_b": params_b,
        "quant": f"{bits}bit" if bits_match else "", "approx_size_gb": size_gb,
        "min_ram_gb": _cat_min_ram(size_gb), "context_max": None, "thinking": None,
        "tool_format": "auto", "vision": None, "vocab_size": None, "draft": None,
        "notes": "Estimated from the repo name; not downloaded yet.",
        "approximate": True, "guessed": True, "local": False,
    }


def model_metadata(model_id: str, *, guess: bool = True) -> dict | None:
    """What a model is: read from its files when downloaded, else a name-based
    guess (guess=True) or None. Always a fresh dict the caller may change."""
    name = str(model_id or "")
    local = _cat_local_metadata(name)
    if local is not None:
        return local
    return _cat_guess(name) if guess else None


def draft_for(model_id: str) -> str | None:
    """A downloaded, much smaller model with the same tokenizer, for speculative decoding.

    Same model_type and vocabulary size is the requirement mlx-lm's draft
    decoding actually has; a quarter of the served model's size or less is
    what makes drafting pay. backends.py (DRAFT_MODEL=auto) looks this name
    up at call time. None when nothing on disk qualifies.
    """
    target = _cat_local_metadata(str(model_id or ""))
    if not target or not target.get("vocab_size") or not target.get("approx_size_gb"):
        return None
    best: tuple[float, str] | None = None
    for candidate in downloaded_models():
        if candidate == model_id:
            continue
        meta = _cat_local_metadata(candidate)
        if (not meta or meta.get("family") != target["family"]
                or meta.get("vocab_size") != target["vocab_size"]
                or not meta.get("approx_size_gb")):
            continue
        size = float(meta["approx_size_gb"])
        if size * 4 <= float(target["approx_size_gb"]) and (best is None or size < best[0]):
            best = (size, candidate)
    return best[1] if best else None


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
# Aliases: MODEL_ALIASES only
# --------------------------------------------------------------------------- #

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
    """Every alias and the repo id it resolves to: {name: {"id", "source"}}."""
    return {name: {"id": resolve_model(target, config), "source": "env"}
            for name, target in user_model_aliases(config).items()}


def resolve_model(name: Any, config: Any = None) -> str:
    """The repo id (or path) an alias stands for; anything else unchanged.

    A repo id ("org/name") or an existing directory is never treated as an
    alias. An alias may point at another one ("chat" -> "small"); chains stop
    after a few hops so a cycle cannot hang a request.
    """
    current = str(name or "").strip()
    if not current:
        return current
    aliases = user_model_aliases(config)
    for _ in range(4):
        if "/" in current or Path(current).expanduser().is_dir():
            return current
        target = aliases.get(current.lower())
        if not target or target == current:
            return current
        current = target
    return current


def alias_of(model_id: str, config: Any = None) -> str:
    """The first alias that resolves to model_id, or ""."""
    for name, info in sorted(model_aliases(config).items()):
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
    requested = str(getattr(args, "model", "") or os.environ.get("MODEL_ID", "")).strip()
    if not requested or "/" in requested:
        return {}
    resolved = resolve_model(requested)
    if resolved == requested:
        return {}
    return {"model": resolved, "model_alias": requested.lower()}


# --------------------------------------------------------------------------- #
# Benchmark results and the recommendation. diagnostics.py measures; these
# only read and write files, so the API and the CLI agree on where they live.
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
    so a copied data directory never badges a model this Mac cannot hold.
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
    """Record the applied recommendation (the switcher's "recommended" badge).

    Stamped with this machine's RAM, so a copied data directory never badges a
    model a Mac of another size cannot hold. It is a record, not a setting:
    `recommend --apply` writes MODEL_ID and CONTEXT_SIZE into .env, which is
    the only place the served model comes from.
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
    'RECOMMENDATION_FILE',
    '_cat_guess',
    '_cat_local_metadata',
    '_cat_min_ram',
    'alias_of',
    'bench_dir',
    'catalog_cli_fields',
    'downloaded_models',
    'draft_for',
    'fits_in_ram',
    'latest_bench_run',
    'list_bench_runs',
    'load_bench_run',
    'model_aliases',
    'model_metadata',
    'model_snapshot_dir',
    'parse_model_aliases',
    'quality_key',
    'read_recommendation',
    'resolve_model',
    'save_bench_run',
    'user_model_aliases',
    'write_recommendation',
]
