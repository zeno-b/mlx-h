"""Pluggable inference backends: how the model server child is launched.

The app used to know exactly one way to serve a model: `python -m
mlx_lm.server` with a command line assembled by ModelServerManager. That
coupling made every alternative engine (a throughput-tuned mlx-lm, an external
OpenAI-compatible MLX server, a distributed one) a fork of the manager. A
BackendSpec moves the per-engine knowledge (command line, which --help to
sniff, which log lines mean the generation thread died, which "model" value a
request must carry, what the engine can do) into data, so the manager stays a
supervisor and other modules can add engines through register_backend().

Contract for other modules (mlx_engine.py, sharding.py, ...):

    from .backends import *
    register_backend(BackendSpec(name="mlx-engine", label="...", launcher=fn, ...))

`launcher(manager, use_adapter)` returns the argv list for subprocess.Popen.
It may read any attribute of the ModelServerManager (model_id, model_port,
host, web_port, adapter_path(), _server_help, the lever attributes, the
backend_* strings) and should set `manager._accepted_levers` to the optional
flags it actually passed, so describe() can report them. BACKEND_EXTRA_ARGS
are appended by the manager after the launcher returns. `launch_inputs` names
the manager attributes the argv depends on, so apply_config() can tell whether
a configuration change needs a restart.

Facts about the installed mlx-lm 0.32.2 that shaped this module (read from
.venv/.../mlx_lm/server.py and models/cache.py):

  * ModelProvider.load (server.py:477-486) maps the CLI --adapter-path ONLY
    for the request model name "default_model"; any other "model" value keeps
    the request's own `adapters` field (None when absent). The startup load is
    load("default_model", None, "default_model") (server.py:473-475), so its
    model key is (repo, cli_adapter, cli_draft). A request carrying the repo id
    produces (repo, None, cli_draft), the keys differ, and _load reloads the
    base weights WITHOUT the adapter (server.py:482-484). That is why every
    request to an mlx-lm server sends request_model() == "default_model".
  * Batching is off when a draft model is loaded (server.py:460), when any
    cache layer lacks `merge` (server.py:461-463: QuantizedKVCache,
    ChunkedKVCache and ConcatenateKVCache have none), when --kv-bits is set
    (server.py:768-773), and per request when it carries a `seed`
    (server.py:771) or image parts (server.py:818-820).
  * LRUPromptCache is built with the entry count only (server.py:1914);
    --prompt-cache-bytes is enforced by trim_to() when a request joins the
    batch (server.py:886-889) and nowhere on the sequential path.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import logging
import re
import shlex
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .sysutil import *  # noqa: F401,F403


# Capability vocabulary. Free text would drift between engines and the UI could
# not render it, so registration rejects anything outside this set.
BACKEND_CAPABILITIES = frozenset({
    "batching",            # continuous batching of concurrent requests
    "speculative",         # draft-model speculative decoding
    "kv_quant",            # quantized KV cache
    "prompt_cache",        # server-side prefix cache reused across requests
    "reasoning_field",     # returns reasoning in a separate message field
    "native_tool_calls",   # parses tool calls into tool_calls itself
    "vision",              # accepts image content parts
    "distributed",         # spans more than one machine
    "per_request_adapter", # "adapters" in the request body selects a LoRA
    "mtp",                 # multi-token prediction heads
    "cached_tokens",       # reports usage.prompt_tokens_details.cached_tokens
})

DEFAULT_BACKEND = "mlx-lm"


def _backends_always_available() -> tuple[bool, str]:
    return True, ""


def _backends_model_of(config: Any) -> str:
    """The configured model id from a Config, a manager, or anything similar."""
    for attr in ("model", "model_id"):
        value = getattr(config, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


@dataclass
class BackendSpec:
    """Everything the manager needs to run one inference engine."""

    name: str
    label: str
    # Module whose `python -m <module> --help` output is sniffed for supported
    # flags. Empty means "do not sniff": the launcher builds a fixed command.
    help_module: str = "mlx_lm.server"
    # (manager, use_adapter) -> argv. Required for anything that starts a child.
    launcher: Callable[[Any, bool], list[str]] | None = None
    # Compiled bytes regexes matched against the child's log by the watchdog.
    fatal_patterns: tuple = ()
    # What the engine can do at all (shown before anything is started).
    capabilities: frozenset = frozenset()
    # () -> (usable, reason), or (config) -> (usable, reason). Cheap: called
    # on every GET /api/backends.
    available: Callable[..., tuple[bool, str]] = _backends_always_available
    # (config-like) -> the "model" value a request body must carry so the
    # engine serves the configured model WITH the configured adapter. Called
    # with a Config (model_client) or the manager (probes); both are accepted.
    request_model: Callable[[Any], str] = _backends_model_of
    description: str = ""
    # (manager) -> list of human-readable decisions, shown in describe().
    decide: Callable[[Any], list[str]] | None = None
    # (manager) -> capabilities actually in effect for this manager, when they
    # depend on configuration (batching drops out when a draft model loads).
    effective_capabilities: Callable[[Any], frozenset] | None = None
    # Whether the launcher can serve a LoRA adapter from the command line: a
    # bool or (manager) -> bool. When False the manager never builds the
    # adapter candidate, and describe() reports the adapter as not served.
    supports_adapter: Any = True
    # Listed by GET /api/backends. Internal or experimental specs can opt out.
    public: bool = True
    # (manager) -> the watchdog probe timeout to use, when the engine queues
    # the probe behind real work (sequential serving). None keeps the manager's.
    probe_timeout: Callable[[Any], float] | None = None
    # (manager) -> the prompt-cache plan (prompt_cache_plan() shape) for
    # describe(), or None when the engine's cache is not ours to size.
    cache_plan: Callable[[Any], dict] | None = None
    # Manager attributes the launcher's argv depends on. Empty means "unknown",
    # and apply_config() then treats every launch attribute as relevant.
    launch_inputs: tuple = ()
    extra: dict = field(default_factory=dict)

    def caps_for(self, manager: Any = None) -> frozenset:
        if manager is not None and self.effective_capabilities is not None:
            try:
                return frozenset(self.effective_capabilities(manager))
            except Exception as exc:  # noqa: BLE001
                log(f"Backend {self.name}: capability probe failed ({exc}).", logging.DEBUG)
        return frozenset(self.capabilities)

    def serves_adapter(self, manager: Any = None) -> bool:
        flag = self.supports_adapter
        if callable(flag):
            try:
                return bool(flag(manager))
            except Exception:  # noqa: BLE001
                return False
        return bool(flag)


BACKENDS: dict[str, BackendSpec] = {}
_backends_warned: set[str] = set()


def register_backend(spec: BackendSpec) -> BackendSpec:
    """Add or replace an engine. Unknown capability names are a programming error."""
    if not isinstance(spec, BackendSpec):
        raise TypeError("register_backend expects a BackendSpec")
    if not spec.name or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", spec.name):
        raise ValueError(f"invalid backend name {spec.name!r}")
    unknown = set(spec.capabilities) - BACKEND_CAPABILITIES
    if unknown:
        raise ValueError(
            f"backend {spec.name!r} declares unknown capabilities: {sorted(unknown)}"
        )
    spec.capabilities = frozenset(spec.capabilities)
    BACKENDS[spec.name] = spec
    return spec


def backend_names() -> list[str]:
    return list(BACKENDS)


def get_backend(name: str | BackendSpec | None) -> BackendSpec:
    """Resolve a name to a spec. Unknown names fall back to mlx-lm, loudly.

    Falling back rather than raising keeps a typo in MODEL_BACKEND from leaving
    the app with no model at all: the server still starts, on the engine that
    has always worked, and the log says why (once per name, because this runs
    for every request payload).
    """
    if isinstance(name, BackendSpec):
        return name
    key = (name or DEFAULT_BACKEND).strip().lower()
    spec = BACKENDS.get(key)
    if spec is None:
        if key not in _backends_warned:
            _backends_warned.add(key)
            log(
                f"Unknown MODEL_BACKEND {name!r}; using {DEFAULT_BACKEND}. "
                f"Known backends: {', '.join(BACKENDS) or 'none'}.",
                logging.WARNING,
            )
        spec = BACKENDS[DEFAULT_BACKEND]
    return spec


def request_model_for(config: Any) -> str:
    """The "model" value every request to the local engine must carry.

    One function for every request path (ModelClient.payload, its readiness
    probe, the manager's probes, the benchmark), so they can never disagree:
    on mlx-lm a mismatched value is a weight reload, not a cosmetic field.
    """
    spec = getattr(config, "backend", None)
    if not isinstance(spec, BackendSpec):      # a Config; a manager carries its spec
        spec = get_backend(getattr(config, "model_backend", None) or DEFAULT_BACKEND)
    try:
        value = spec.request_model(config)
    except Exception as exc:  # noqa: BLE001
        log(f"Backend {spec.name}: request_model failed ({exc}); sending the model id.",
            logging.WARNING)
        value = ""
    return value or _backends_model_of(config)


def _backends_call_available(spec: BackendSpec, config: Any) -> tuple[bool, str]:
    try:
        params = inspect.signature(spec.available).parameters
        ok, reason = spec.available(config) if params else spec.available()
    except Exception as exc:  # noqa: BLE001
        ok, reason = False, f"availability check failed: {exc}"
    return bool(ok), str(reason or "")


def public_backends(config: Any = None) -> list[dict]:
    """One row per listed engine, for GET /api/backends and the Settings UI."""
    current = (getattr(config, "model_backend", "") or DEFAULT_BACKEND).strip().lower()
    if current not in BACKENDS:
        current = DEFAULT_BACKEND
    rows = []
    for spec in BACKENDS.values():
        if not spec.public and spec.name != current:
            continue
        ok, reason = _backends_call_available(spec, config)
        rows.append({
            "name": spec.name,
            "label": spec.label,
            "description": spec.description,
            "available": ok,
            "reason": reason,
            "capabilities": sorted(spec.capabilities),
            "current": spec.name == current,
        })
    return rows


# --------------------------------------------------------------------------- #
# Helpers shared by the mlx-lm family
# --------------------------------------------------------------------------- #
# The generation thread's death leaves the HTTP server answering 200 on
# /v1/models and /health while every completion hangs forever. The traceback
# goes to the child's stdout/stderr (the log file) and nowhere else.
MLXLM_FATAL_PATTERNS = (
    re.compile(rb"Exception in thread [^\n]*\(_generate\)"),
    re.compile(rb"Command buffer execution failed"),
    re.compile(rb"kIOGPUCommandBufferCallbackErrorOutOfMemory"),
)
# Metal out-of-memory kills any MLX engine's generation, not only mlx-lm's.
METAL_FATAL_PATTERNS = MLXLM_FATAL_PATTERNS[1:]


def _backends_has_flag(help_text: str, flag: str) -> bool:
    return help_has_flag(help_text, flag)


def _backends_allowed_origins(m: Any) -> str:
    """The web UI's own origins, or "" when the manager was not told the port.

    The browser never talks to the model port (the UI goes through the web
    app), so nothing legitimate needs CORS there. mlx-lm defaults to "*",
    which lets any page read completions from it. Note what this does NOT
    stop: mlx-lm parses the body regardless of Content-Type (server.py
    do_POST), so a cross-site "simple" POST still executes; it only becomes
    unreadable. Binding to 127.0.0.1 remains the real boundary.
    """
    port = int(getattr(m, "web_port", 0) or 0)
    if port <= 0:
        return ""
    return ",".join(f"http://{host}:{port}" for host in ("127.0.0.1", "localhost", "[::1]"))


def _backends_catalog_draft_for() -> Callable[[str], Any] | None:
    """catalog.draft_for if a catalog module provides one, else None.

    Looked up by name at call time rather than imported, because the catalogue
    is another feature's module: in the bundled build it is a global, in the
    package it is local_llm.catalog, and on a build without it this degrades
    to "no draft" instead of an import error.
    """
    fn = globals().get("draft_for")
    if fn is None and __package__:
        try:
            module = importlib.import_module(f"{__package__}.catalog")
            fn = getattr(module, "draft_for", None)
        except Exception:  # noqa: BLE001
            fn = None
    return fn if callable(fn) else None


def _backends_resolve_draft(m: Any, *, prefer_batching: bool = False) -> tuple[str, str]:
    """(draft model repo or "", note). Never raises."""
    raw = str(getattr(m, "draft_model", "") or "").strip()
    if not raw:
        return "", ""
    note = ""
    if raw.lower() == "auto":
        if prefer_batching:
            return "", (
                "DRAFT_MODEL=auto: more than one generation may run at once, and a "
                "loaded draft model turns continuous batching off on mlx-lm 0.32, so "
                "batching wins and no draft model is loaded. Set DRAFT_MODEL to a repo "
                "id to force speculative decoding."
            )
        finder = _backends_catalog_draft_for()
        if finder is None:
            return "", ("DRAFT_MODEL=auto: this build has no catalog.draft_for(), so no "
                        "draft model is loaded. Set DRAFT_MODEL to a repo id instead.")
        try:
            picked = str(finder(m.model_id) or "").strip()
        except Exception as exc:  # noqa: BLE001
            return "", f"DRAFT_MODEL=auto: the catalogue lookup failed ({exc}); no draft model."
        if not picked:
            return "", f"DRAFT_MODEL=auto: the catalogue lists no draft model for {m.model_id}."
        raw = picked
        note = f"DRAFT_MODEL=auto picked {picked} from the catalogue."
    if raw == m.model_id:
        return "", ("DRAFT_MODEL is the served model itself, which doubles the memory "
                    "for no speed-up; ignored.")
    if not model_is_cached(raw):
        note = (note + " " if note else "") + (
            f"Draft model {raw} is not downloaded yet; the server fetches it on start.")
    return raw, note


# --------------------------------------------------------------------------- #
# Prompt-cache sizing (item 5, server level)
# --------------------------------------------------------------------------- #
# Distinct system prefixes one agent turn can send: the router call, knowledge
# triage, the full tool lane, a narrowed tool subset, the prose lane and the
# reasoning step each carry their own system prompt (agent.py, llm.py lanes).
PROMPT_CACHE_LANES = ("router", "triage", "tools", "tools-subset", "prose", "reasoning")
# The readiness/health probe ("hi") and the watchdog probe ("ok") are system-
# less prompts that also land in the same LRU every 15 to 30 seconds.
_BACKENDS_PROBE_ENTRIES = 2

_backends_model_cfg_cache: dict[str, dict] = {}
_backends_hybrid_types_cache: set[str] | None = None


def _backends_model_config(model_id: str) -> dict:
    """config.json of a local or cached model, merged with text_config. {} if unknown."""
    if not model_id:
        return {}
    if model_id in _backends_model_cfg_cache:
        return _backends_model_cfg_cache[model_id]
    found: dict = {}
    candidates: list[Path] = []
    local = Path(model_id).expanduser()
    if local.is_dir():
        candidates.append(local / "config.json")
    else:
        snaps = hf_cache_dir() / ("models--" + model_id.replace("/", "--")) / "snapshots"
        try:
            candidates.extend(sorted(snaps.glob("*/config.json"),
                                     key=lambda p: p.stat().st_mtime, reverse=True))
        except OSError:
            pass
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            nested = data.get("text_config")
            found = {**data, **nested} if isinstance(nested, dict) else data
            break
    _backends_model_cfg_cache[model_id] = found
    return found


def _backends_hybrid_types() -> set[str]:
    """model_type values whose mlx-lm implementation builds a recurrent cache.

    Read from the installed mlx_lm/models sources rather than a hand-kept list,
    so a new hybrid architecture is recognised the day mlx-lm ships it.
    """
    global _backends_hybrid_types_cache
    if _backends_hybrid_types_cache is not None:
        return _backends_hybrid_types_cache
    found: set[str] = set()
    try:
        spec = importlib.util.find_spec("mlx_lm")
        root = Path(spec.origin).parent / "models" if spec and spec.origin else None
        if root is not None and root.is_dir():
            for path in root.glob("*.py"):
                if path.stem in ("cache", "base", "__init__"):
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                if "ArraysCache(" in text or "MambaCache(" in text:
                    found.add(path.stem)
    except Exception:  # noqa: BLE001
        pass
    _backends_hybrid_types_cache = found
    return found


def _backends_is_hybrid(cfg: dict) -> bool | None:
    """True for models with non-trimmable (recurrent) cache layers. None if unknown."""
    if not cfg:
        return None
    if str(cfg.get("model_type", "")) in _backends_hybrid_types():
        return True
    if any(key in cfg for key in ("linear_num_value_heads", "linear_num_key_heads",
                                  "full_attention_interval")):
        return True
    layer_types = cfg.get("layer_types")
    if isinstance(layer_types, list) and any(
            str(kind) not in ("full_attention", "sliding_attention") for kind in layer_types):
        return True
    return False


def _backends_kv_bytes_per_token(cfg: dict, kv_bits: int = 0, kv_group_size: int = 64) -> int | None:
    """KV-cache bytes one token costs, from config.json. None when it cannot be told."""
    try:
        layers = int(cfg.get("num_hidden_layers") or 0)
        heads = int(cfg.get("num_attention_heads") or 0)
        kv_heads = int(cfg.get("num_key_value_heads") or heads)
        head_dim = int(cfg.get("head_dim") or (int(cfg.get("hidden_size") or 0) // max(1, heads)))
    except (TypeError, ValueError):
        return None
    if layers <= 0 or kv_heads <= 0 or head_dim <= 0:
        return None
    interval = cfg.get("full_attention_interval")
    if isinstance(interval, int) and interval > 1:
        layers = max(1, layers // interval)     # only the attention layers keep KV
    if kv_bits and kv_bits > 0:
        # Packed values plus one fp16 scale and bias per group.
        per_elem = kv_bits / 8 + 4 / max(1, kv_group_size)
    else:
        per_elem = 2.0
    return int(2 * layers * kv_heads * head_dim * per_elem)


def lane_aware_cache_size(concurrency: int, hybrid: bool = False) -> int:
    """Prompt-cache entries a turn needs so its system prefixes do not evict each other.

    One entry per lane for an attention-only model: mlx-lm drops an entry's
    prefixes when the cache is trimmable (models/cache.py insert_cache), and
    the surviving full-prompt entry serves the shorter prefix by trimming.
    Two per lane for a hybrid model: its recurrent layers cannot be trimmed, so
    the system-boundary snapshot and the full entry are both kept. Plus one per
    concurrent generation (its in-flight sequence) and the two probe prompts.
    """
    per_lane = 2 if hybrid else 1
    return len(PROMPT_CACHE_LANES) * per_lane + max(1, int(concurrency or 1)) + _BACKENDS_PROBE_ENTRIES


def prompt_cache_plan(m: Any, *, batching: bool, concurrency: int, adopt: bool) -> dict:
    """What the prompt cache should hold, and why. `adopt` lets the size grow."""
    configured = int(getattr(m, "prompt_cache_size", 0) or 0)
    cfg = _backends_model_config(getattr(m, "model_id", ""))
    hybrid = _backends_is_hybrid(cfg)
    per_token = _backends_kv_bytes_per_token(
        cfg, getattr(m, "kv_bits", 0), getattr(m, "kv_group_size", 64)) if cfg else None
    recommended = lane_aware_cache_size(concurrency, hybrid=bool(hybrid))
    byte_cap = int(getattr(m, "prompt_cache_bytes", 0) or 0)
    context = int(getattr(m, "context_size", 0) or 0)
    notes: list[str] = []
    bound: int | None = None
    if not batching:
        # The sequential path never applies --prompt-cache-bytes, so the entry
        # count is the only memory bound there. Assume an average entry of half
        # the context window [inference: real entries range from a ~1k-token
        # system prefix to a full window].
        if per_token and context > 0 and byte_cap > 0:
            bound = max(1, byte_cap // (per_token * max(256, context // 2)))
        else:
            bound = max(1, configured)
        recommended = min(recommended, bound)
    size = min(64, max(configured, recommended)) if adopt else configured
    if configured and configured < recommended and not adopt:
        notes.append(
            f"PROMPT_CACHE_SIZE={configured} is below the {recommended} prefixes a turn "
            f"can create ({len(PROMPT_CACHE_LANES)} system-prompt lanes, "
            f"{max(1, concurrency)} concurrent generation(s), 2 probes), so lanes evict "
            "each other's system prefix. mlx-lm-turbo sizes this automatically; on "
            f"mlx-lm set PROMPT_CACHE_SIZE={recommended}."
        )
    elif adopt and size > configured:
        notes.append(
            f"Prompt cache holds {size} entries (lane-aware: {len(PROMPT_CACHE_LANES)} lanes"
            f"{' x2 for a hybrid model' if hybrid else ''} + {max(1, concurrency)} "
            f"concurrent + 2 probes) instead of PROMPT_CACHE_SIZE={configured}"
            + (f"; bounded by PROMPT_CACHE_BYTES at every batch admission." if batching
               else f"; capped at {bound} so it fits PROMPT_CACHE_BYTES, which the "
                    "sequential path does not enforce.")
        )
    if hybrid:
        notes.append(
            "Hybrid model: its recurrent cache cannot be trimmed, so a cached entry is "
            "reused only when it is an exact prefix of the new prompt. The batched path "
            "snapshots the system-prompt boundary of each request; the sequential path "
            "stores only the full prompt plus reply."
        )
    return {
        "size": size,
        "configured_size": configured,
        "recommended_size": recommended,
        "lanes": list(PROMPT_CACHE_LANES),
        "bytes": byte_cap,
        "bytes_enforced": bool(batching),
        "hybrid_model": hybrid,
        "kv_bytes_per_token": per_token,
        "notes": notes,
    }


def _backends_batching_blockers(m: Any, draft: str) -> list[str]:
    blockers = []
    help_text = getattr(m, "_server_help", "")
    if getattr(m, "kv_bits", 0) > 0 and (not help_text or _backends_has_flag(help_text, "--kv-bits")):
        blockers.append(f"KV_BITS={m.kv_bits} (mlx-lm serves every request one at a time "
                        "when --kv-bits is set)")
    if draft and (not help_text or _backends_has_flag(help_text, "--draft-model")):
        blockers.append(f"draft model {draft} (a loaded draft model makes the model "
                        "non-batchable)")
    return blockers


# --------------------------------------------------------------------------- #
# mlx-lm: today's command line, unchanged
# --------------------------------------------------------------------------- #
def _mlxlm_levers(m: Any) -> dict:
    return {
        "max_kv_size": m.max_kv_size,
        "kv_bits": m.kv_bits,
        "kv_group_size": m.kv_group_size,
        "quantized_kv_start": m.quantized_kv_start,
        "prefill_step_size": m.prefill_step_size,
        "prompt_cache_size": m.prompt_cache_size,
        "prompt_cache_bytes": m.prompt_cache_bytes,
        "decode_concurrency": m.decode_concurrency,
        "prompt_concurrency": m.prompt_concurrency,
        "prompt_cache_dir": m.prompt_cache_dir,
    }


def _mlxlm_command(m: Any, use_adapter: bool, lv: dict, *,
                   draft: str = "", num_draft_tokens: int = 0) -> list[str]:
    """The mlx_lm.server argv. With no draft and no web port, byte-identical to
    the pre-backend ModelServerManager._build_cmd (tests pin this)."""
    help_text = m._server_help
    cmd = [sys.executable, "-m", "mlx_lm.server"]
    levers: list[str] = []
    if not add_if_supported(cmd, help_text, ["--model", "--hf-path", "--mlx-path"], m.model_id):
        cmd.extend(["--model", m.model_id])
    if not add_if_supported(cmd, help_text, ["--port"], str(m.model_port)):
        cmd.extend(["--port", str(m.model_port)])
    if lv["max_kv_size"] > 0:
        # Caps the rotating KV cache. Older mlx-lm builds do not have it, so
        # this is best-effort rather than an error.
        if add_if_supported(cmd, help_text, ["--max-kv-size", "--max_kv_size"],
                            str(lv["max_kv_size"])):
            levers.append("--max-kv-size")
        else:
            log(
                "Installed mlx_lm.server has no --max-kv-size; the KV cache stays unbounded.",
                logging.WARNING,
            )
    adapter_path = m.adapter_path()
    # Count the memory levers that the installed build actually accepted.
    # Zero of them on an 8GB machine is worth saying out loud: it means the
    # server runs with no cap on prefill peak or on resident prompt caches,
    # which is how a marginal machine ends up with a dead generation thread.
    accepted: list[str] = []
    if lv["kv_bits"] > 0:
        # Quantized KV cache roughly halves cache memory at 8 bits, which is
        # the difference between a usable context and swapping on 8GB.
        # Server support is genuinely unsettled across builds, so this is
        # best effort and the run continues without it.
        if add_if_supported(cmd, help_text, ["--kv-bits"], str(lv["kv_bits"])):
            accepted.append("--kv-bits")
            add_if_supported(cmd, help_text, ["--kv-group-size"], str(lv["kv_group_size"]))
            add_if_supported(cmd, help_text, ["--quantized-kv-start"], str(lv["quantized_kv_start"]))
        else:
            log(
                "Installed mlx_lm.server has no --kv-bits; the KV cache stays in fp16. "
                "Support for this flag varies by build.",
                logging.WARNING,
            )
    # --- mlx-lm >= 0.29 levers ----------------------------------------- #
    # --prefill-step-size is the important one: its default of 2048 makes any
    # shorter prompt a single forward pass, and the peak of that pass is what
    # fails the Metal command buffer on 8GB.
    for key, flag in (("prefill_step_size", "--prefill-step-size"),
                      ("prompt_cache_size", "--prompt-cache-size"),
                      ("prompt_cache_bytes", "--prompt-cache-bytes"),
                      ("decode_concurrency", "--decode-concurrency"),
                      ("prompt_concurrency", "--prompt-concurrency")):
        if lv[key] > 0 and add_if_supported(cmd, help_text, [flag], str(lv[key])):
            accepted.append(flag)
    if lv["prompt_cache_dir"]:
        # The agent re-sends an extending prefix every step, which is the
        # best case for prompt caching. 0.32 has no persistent cache flag at
        # all; whole-flag matching keeps "--prompt-cache" from matching
        # "--prompt-cache-size" (argparse then exits "ambiguous option").
        if add_if_supported(cmd, help_text, ["--prompt-cache-dir", "--prompt-cache"],
                            lv["prompt_cache_dir"]):
            levers.append(cmd[-2])
        else:
            log("Installed mlx_lm.server has no prompt cache flag; ignoring PROMPT_CACHE_DIR.",
                logging.WARNING)
    if use_adapter and adapter_path is not None:
        add_if_supported(cmd, help_text, ["--adapter-path", "--adapter"], str(adapter_path))
    # --- Levers added with the backend registry. Each is off unless asked
    # for, and appended after the original flags, so a default configuration
    # keeps the exact command line it always had.
    if draft:
        if add_if_supported(cmd, help_text, ["--draft-model"], draft):
            levers.append("--draft-model")
            if num_draft_tokens > 0 and add_if_supported(
                    cmd, help_text, ["--num-draft-tokens"], str(num_draft_tokens)):
                levers.append("--num-draft-tokens")
        else:
            log("Installed mlx_lm.server has no --draft-model; speculative decoding is off.",
                logging.WARNING)
    origins = _backends_allowed_origins(m)
    if origins and add_if_supported(cmd, help_text, ["--allowed-origins"], origins):
        levers.append("--allowed-origins")
    if accepted:
        log("Model server memory levers in effect: " + ", ".join(accepted), logging.DEBUG)
    elif help_text:
        log(
            "Installed mlx_lm.server accepted none of the memory-limit flags this "
            "build knows about, so prefill peak and resident prompt caches are "
            "uncapped. On a small machine that risks a Metal out-of-memory that "
            "kills the server's generation thread. Check `mlx_lm.server --help` "
            "for the flag names this version uses.",
            logging.WARNING,
        )
    m._accepted_levers = accepted + levers
    return cmd


def _mlxlm_launcher(m: Any, use_adapter: bool) -> list[str]:
    draft, _ = _backends_resolve_draft(m)
    return _mlxlm_command(m, use_adapter, _mlxlm_levers(m), draft=draft,
                          num_draft_tokens=int(getattr(m, "num_draft_tokens", 0) or 0))


def _mlxlm_request_model(_config: Any) -> str:
    # See the module docstring: only this name maps onto the CLI model AND
    # the CLI adapter (server.py:477-486).
    return "default_model"


def _mlxlm_available() -> tuple[bool, str]:
    if importlib.util.find_spec("mlx_lm") is None:
        return False, "mlx-lm is not installed in this environment"
    return True, ""


# What _mlxlm_command reads from the manager besides model, port and adapter.
_MLXLM_LAUNCH_INPUTS = (
    "max_kv_size", "kv_bits", "kv_group_size", "quantized_kv_start", "prefill_step_size",
    "prompt_cache_size", "prompt_cache_bytes", "decode_concurrency", "prompt_concurrency",
    "prompt_cache_dir", "draft_model", "num_draft_tokens", "web_port",
)


def _mlxlm_cache_plan(m: Any) -> dict:
    draft, _ = _backends_resolve_draft(m)
    return prompt_cache_plan(
        m, batching=not _backends_batching_blockers(m, draft),
        concurrency=getattr(m, "max_concurrent_generations", 0) or 1, adopt=False)


_MLXLM_STATIC_CAPS = frozenset({
    "batching", "speculative", "kv_quant", "prompt_cache", "reasoning_field",
    "native_tool_calls", "vision", "per_request_adapter", "cached_tokens",
})


def _mlxlm_effective_caps(m: Any) -> frozenset:
    draft, _ = _backends_resolve_draft(m)
    caps = {"prompt_cache", "reasoning_field", "native_tool_calls", "vision",
            "per_request_adapter", "cached_tokens"}
    if not _backends_batching_blockers(m, draft):
        caps.add("batching")
    if draft:
        caps.add("speculative")
    if getattr(m, "kv_bits", 0) > 0:
        caps.add("kv_quant")
    return frozenset(caps)


def _mlxlm_decide(m: Any) -> list[str]:
    draft, note = _backends_resolve_draft(m)
    blockers = _backends_batching_blockers(m, draft)
    notes = [note] if note else []
    if blockers:
        notes.append("Continuous batching is off: " + "; ".join(blockers) + ".")
    notes.extend(prompt_cache_plan(
        m, batching=not blockers, concurrency=getattr(m, "max_concurrent_generations", 0) or 1,
        adopt=False)["notes"])
    return notes


register_backend(BackendSpec(
    name="mlx-lm",
    label="mlx-lm server",
    help_module="mlx_lm.server",
    launcher=_mlxlm_launcher,
    fatal_patterns=MLXLM_FATAL_PATTERNS,
    capabilities=_MLXLM_STATIC_CAPS,
    available=_mlxlm_available,
    request_model=_mlxlm_request_model,
    description=("The stock mlx_lm.server with the memory levers from the environment, "
                 "exactly as before backends existed. The safe default."),
    decide=_mlxlm_decide,
    effective_capabilities=_mlxlm_effective_caps,
    cache_plan=_mlxlm_cache_plan,
    launch_inputs=_MLXLM_LAUNCH_INPUTS,
))


# --------------------------------------------------------------------------- #
# mlx-lm-turbo: the same server, tuned for throughput
# --------------------------------------------------------------------------- #
# A probe on a sequential server waits for the answer in front of it, and the
# log is quiet while it decodes, so 25 seconds is a false "not generating".
_TURBO_SERIAL_PROBE_TIMEOUT = 90.0


def _turbo_decode_cap(ram_gb: float) -> int:
    # Every decode slot holds a live KV sequence in the same unified memory as
    # the weights; the cap is what each RAM tier can carry next to its default
    # model (core.py tiers) without the batch itself becoming the OOM.
    if ram_gb < 12:
        return 3
    if ram_gb < 24:
        return 4
    if ram_gb < 48:
        return 8
    return 16


def _turbo_prefill_parallel(ram_gb: float) -> int:
    # Parallel prefill multiplies exactly the allocation that fails the Metal
    # command buffer on small machines, so it stays serial below 16 GB.
    if ram_gb < 16:
        return 1
    if ram_gb < 32:
        return 2
    if ram_gb < 64:
        return 4
    return 8


def turbo_plan(m: Any) -> dict:
    """Every lever mlx-lm-turbo will pass, plus the reasoning behind each."""
    ram = TOTAL_RAM_GB
    conc = int(getattr(m, "max_concurrent_generations", 0) or 0) or _default_concurrent_generations(ram)
    decisions: list[str] = []
    draft, note = _backends_resolve_draft(m, prefer_batching=conc > 1 and m.kv_bits <= 0)
    if note:
        decisions.append(note)
    blockers = _backends_batching_blockers(m, draft)
    batching = not blockers
    lv = _mlxlm_levers(m)
    probe_timeout = float(getattr(m, "probe_timeout", 25.0))
    if batching:
        cap = _turbo_decode_cap(ram)
        decode = min(cap, max(int(m.decode_concurrency or 0), conc + 1))
        prompt = min(decode, max(int(m.prompt_concurrency or 0), _turbo_prefill_parallel(ram)))
        lv["decode_concurrency"] = decode
        lv["prompt_concurrency"] = prompt
        decisions.append(
            f"Continuous batching on: decode concurrency {decode} (MAX_CONCURRENT_GENERATIONS "
            f"{conc} plus one slot so the watchdog probe is batched beside a long answer, "
            f"capped at {cap} for {ram:.0f} GB), prompt concurrency {prompt}."
        )
    else:
        probe_timeout = max(probe_timeout, _TURBO_SERIAL_PROBE_TIMEOUT)
        decisions.append(
            "Continuous batching off: " + "; ".join(blockers) + f". Up to {conc} "
            "concurrent generations queue inside the server, so --decode-concurrency and "
            "--prompt-concurrency stay at their configured values (the sequential path "
            f"ignores them). Watchdog probe timeout raised to {probe_timeout:.0f}s because "
            "the probe now waits behind the answer in front of it."
        )
    cache = prompt_cache_plan(m, batching=batching, concurrency=conc, adopt=True)
    lv["prompt_cache_size"] = cache["size"]
    decisions.extend(cache["notes"])
    decisions.append(
        "Still decided inside the server: a model whose cache layers cannot merge "
        "(ChunkedKVCache, QuantizedKVCache) and any request carrying a seed or images "
        "is served sequentially."
    )
    return {
        "levers": lv,
        "draft_model": draft,
        "num_draft_tokens": int(getattr(m, "num_draft_tokens", 0) or 0) if draft else 0,
        "batching": batching,
        "concurrency": conc,
        "probe_timeout": probe_timeout,
        "prompt_cache": cache,
        "decisions": decisions,
    }


def _turbo_launcher(m: Any, use_adapter: bool) -> list[str]:
    plan = turbo_plan(m)
    return _mlxlm_command(m, use_adapter, plan["levers"], draft=plan["draft_model"],
                          num_draft_tokens=plan["num_draft_tokens"])


def _turbo_effective_caps(m: Any) -> frozenset:
    plan = turbo_plan(m)
    caps = {"prompt_cache", "reasoning_field", "native_tool_calls", "vision",
            "per_request_adapter", "cached_tokens"}
    if plan["batching"]:
        caps.add("batching")
    if plan["draft_model"]:
        caps.add("speculative")
    if m.kv_bits > 0:
        caps.add("kv_quant")
    return frozenset(caps)


register_backend(BackendSpec(
    name="mlx-lm-turbo",
    label="mlx-lm server, throughput-tuned",
    help_module="mlx_lm.server",
    launcher=_turbo_launcher,
    fatal_patterns=MLXLM_FATAL_PATTERNS,
    capabilities=_MLXLM_STATIC_CAPS,
    available=_mlxlm_available,
    request_model=_mlxlm_request_model,
    description=("The same mlx_lm.server with continuous batching sized from RAM and "
                 "MAX_CONCURRENT_GENERATIONS, a lane-aware prompt cache, and speculative "
                 "decoding when DRAFT_MODEL is set. describe() explains each choice."),
    decide=lambda m: turbo_plan(m)["decisions"],
    effective_capabilities=_turbo_effective_caps,
    probe_timeout=lambda m: turbo_plan(m)["probe_timeout"],
    cache_plan=lambda m: turbo_plan(m)["prompt_cache"],
    # Sizing reads the concurrency and the context window on top of mlx-lm's.
    launch_inputs=_MLXLM_LAUNCH_INPUTS + ("max_concurrent_generations", "context_size"),
))


# --------------------------------------------------------------------------- #
# openai-external: any OpenAI-compatible MLX engine (Rapid-MLX, vllm-mlx, ...)
# --------------------------------------------------------------------------- #
_EXT_PLACEHOLDERS = ("python", "model", "port", "host", "adapter", "draft_model")
# Without {port} the app probes a port nobody listens on; without {model} the
# request "model" (the configured id) and the served model can disagree.
_EXT_REQUIRED = ("model", "port")
_EXT_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_backend_command(template: str, values: dict) -> list[str]:
    """Expand a BACKEND_COMMAND template into argv. ValueError on a bad template.

    Placeholders are substituted per shlex token, so JSON in an argument
    ('{"enable_thinking": false}') is left alone. A token whose optional
    placeholder ({adapter}, {draft_model}) is empty is dropped, together with
    the flag in front of it ("--adapter-path {adapter}" disappears as a pair).
    """
    if not (template or "").strip():
        raise ValueError("BACKEND_COMMAND is empty: the openai-external backend needs the "
                         "command line that starts the server")
    try:
        tokens = shlex.split(template)
    except ValueError as exc:
        raise ValueError(f"BACKEND_COMMAND does not parse: {exc}") from None
    used: set[str] = set()
    for token in tokens:
        used.update(_EXT_PLACEHOLDER_RE.findall(token))
    unknown = sorted(used - set(_EXT_PLACEHOLDERS))
    if unknown:
        raise ValueError(
            "BACKEND_COMMAND uses unknown placeholder(s) " + ", ".join("{" + u + "}" for u in unknown)
            + "; allowed: " + ", ".join("{" + p + "}" for p in _EXT_PLACEHOLDERS))
    missing = [p for p in _EXT_REQUIRED if p not in used]
    if missing:
        raise ValueError(
            "BACKEND_COMMAND must contain " + " and ".join("{" + p + "}" for p in missing)
            + ": the app has to start the engine on the port it probes, serving the "
              "model it asks for")
    out: list[str] = []
    for token in tokens:
        names = _EXT_PLACEHOLDER_RE.findall(token)
        if any(not str(values.get(name, "") or "") for name in names):
            if not token.startswith("-") and out and out[-1].startswith("-") and "=" not in out[-1]:
                out.pop()
            continue
        out.append(_EXT_PLACEHOLDER_RE.sub(lambda mt: str(values[mt.group(1)]), token))
    return out


def _ext_capabilities(m_or_config: Any) -> frozenset:
    raw = str(getattr(m_or_config, "backend_capabilities", "") or "")
    names = {item.strip().lower() for item in raw.split(",") if item.strip()}
    unknown = names - BACKEND_CAPABILITIES
    if unknown:
        key = "ext-caps:" + ",".join(sorted(unknown))
        if key not in _backends_warned:
            _backends_warned.add(key)
            log("BACKEND_CAPABILITIES has unknown names (ignored): " + ", ".join(sorted(unknown))
                + ". Known: " + ", ".join(sorted(BACKEND_CAPABILITIES)), logging.WARNING)
    return frozenset(names & BACKEND_CAPABILITIES)


def _ext_launcher(m: Any, use_adapter: bool) -> list[str]:
    adapter = m.adapter_path() if use_adapter else None
    draft, _ = _backends_resolve_draft(m)
    values = {
        "python": sys.executable,
        "model": m.model_id,
        "port": str(m.model_port),
        "host": getattr(m, "host", "127.0.0.1") or "127.0.0.1",
        "adapter": str(adapter) if adapter is not None else "",
        "draft_model": draft,
    }
    cmd = expand_backend_command(getattr(m, "backend_command", ""), values)
    m._accepted_levers = sorted(name for name in _EXT_PLACEHOLDER_RE.findall(m.backend_command)
                                if values.get(name))
    return cmd


def _ext_available(config: Any = None) -> tuple[bool, str]:
    template = str(getattr(config, "backend_command", "") or "")
    if not template.strip():
        return False, "set BACKEND_COMMAND to the engine's start command"
    try:
        argv = expand_backend_command(template, {
            "python": sys.executable, "model": "m", "port": "1", "host": "127.0.0.1",
            "adapter": "", "draft_model": ""})
    except ValueError as exc:
        return False, str(exc)
    exe = argv[0] if argv else ""
    if exe != sys.executable and shutil.which(exe) is None and not Path(exe).exists():
        return False, f"{exe} is not on PATH"
    return True, ""


def _ext_decide(m: Any) -> list[str]:
    notes = []
    caps = _ext_capabilities(m)
    try:
        expand_backend_command(m.backend_command, {
            "python": sys.executable, "model": m.model_id, "port": str(m.model_port),
            "host": "127.0.0.1", "adapter": "", "draft_model": ""})
    except ValueError as exc:
        notes.append(f"Will not start: {exc}.")
    if "{adapter}" not in (m.backend_command or ""):
        notes.append("BACKEND_COMMAND has no {adapter} placeholder, so the feedback LoRA "
                     "adapter is not served by this engine.")
    if not caps:
        notes.append("No BACKEND_CAPABILITIES declared; the app assumes a plain "
                     "OpenAI-compatible chat endpoint.")
    return notes


register_backend(BackendSpec(
    name="openai-external",
    label="External OpenAI-compatible engine",
    help_module="",
    launcher=_ext_launcher,
    fatal_patterns=METAL_FATAL_PATTERNS,
    capabilities=frozenset(),
    available=_ext_available,
    request_model=_backends_model_of,
    description=("Launch any OpenAI-compatible MLX server (Rapid-MLX, vllm-mlx, ...) from "
                 "BACKEND_COMMAND with {python} {model} {port} {host} {adapter} "
                 "{draft_model} placeholders; capabilities come from BACKEND_CAPABILITIES."),
    decide=_ext_decide,
    effective_capabilities=_ext_capabilities,
    supports_adapter=lambda m: "{adapter}" in (getattr(m, "backend_command", "") or ""),
    launch_inputs=("backend_command", "draft_model"),
))


# --------------------------------------------------------------------------- #
# Route
# --------------------------------------------------------------------------- #
def register_backends(app: Any, ctx: Any) -> None:
    """GET /api/backends: every engine, whether it can run here, and the live one."""
    ADMIN = ctx.ADMIN

    @app.get("/api/backends")
    def list_backends(_a: dict = ADMIN):
        manager = getattr(app.state, "model_manager", None) or ctx.model_manager
        try:
            current = manager.describe() if manager is not None else None
        except Exception as exc:  # noqa: BLE001
            current = {"error": redact_text(f"{type(exc).__name__}: {exc}")}
        return {
            "backends": public_backends(ctx.config),
            "configured": getattr(ctx.config, "model_backend", DEFAULT_BACKEND),
            "current": current,
            "capability_vocabulary": sorted(BACKEND_CAPABILITIES),
        }


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'BACKENDS',
    'BACKEND_CAPABILITIES',
    'BackendSpec',
    'DEFAULT_BACKEND',
    'METAL_FATAL_PATTERNS',
    'MLXLM_FATAL_PATTERNS',
    'PROMPT_CACHE_LANES',
    '_backends_allowed_origins',
    '_backends_has_flag',
    '_backends_is_hybrid',
    '_backends_kv_bytes_per_token',
    '_backends_model_config',
    '_backends_resolve_draft',
    'backend_names',
    'expand_backend_command',
    'get_backend',
    'lane_aware_cache_size',
    'prompt_cache_plan',
    'public_backends',
    'register_backend',
    'register_backends',
    'request_model_for',
    'turbo_plan',
]
