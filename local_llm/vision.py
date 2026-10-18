"""The "mlx-vlm" engine and the one answer to "can the served model see images?".

Image input used to be a capability nothing could actually exercise: mlx-lm
declares "vision", but its server only builds a processor for the model types in
mlx_lm.processors (one family in mlx-lm 0.32), so Gemma 3 or Qwen2.5-VL behind it
are served as text and an image part makes the request fail with "Only 'text'
content type is supported". mlx-vlm is the engine that serves those models with
their vision towers, so it gets its own BackendSpec here, built from the facts of
its real CLI (mlx-vlm 0.7.8, read from the wheel, not executed):

  * `python -m mlx_vlm.server` (console script `mlx_vlm.server`; the package's
    __main__ calls the same main()). Flags used: --model, --host, --port,
    --max-kv-size, --kv-bits, --kv-group-size, --quantized-kv-start,
    --prefill-step-size, --max-num-seqs. --host defaults to 0.0.0.0 there, so it
    is always passed as 127.0.0.1: the app proxies, nothing else should reach it.
  * OpenAI-compatible POST /v1/chat/completions (and /chat/completions) with
    {"type": "image_url", "image_url": {"url": "data:..."}} content parts; GET
    /v1/models and /health. The request's "model" is loaded by path when it
    differs from the preloaded one (get_cached_model(request.model)), so every
    request must carry exactly the --model value or it triggers a second load.
  * Continuous batching is on by default (a ResponseGenerator per loaded model);
    --max-num-seqs bounds the running batch. Streaming deltas carry separate
    `reasoning` and `tool_calls` fields.
  * --adapter-path exists, but loads adapters in mlx-vlm's own LoRA layout; the
    adapters this app trains are mlx-lm LoRA for text models, so they are not
    served (inference from the two code bases, not a tested load).

served_vision() is what the chat route asks before a turn with attachments: it
sets turn.extras["vision"], which the agent treats as authoritative.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from typing import Any

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .sysutil import *  # noqa: F401,F403
from .backends import *  # noqa: F401,F403
from .catalog import *  # noqa: F401,F403


VLM_BACKEND = "mlx-vlm"
VLM_SERVER_MODULE = "mlx_vlm.server"
VISION_MODES = ("auto", "on", "off")

# Manager attributes the mlx-vlm argv depends on besides model and port.
_VLM_LAUNCH_INPUTS = (
    "max_kv_size", "kv_bits", "kv_group_size", "quantized_kv_start", "prefill_step_size",
    "decode_concurrency", "max_concurrent_generations",
)

_VLM_STATIC_CAPS = frozenset({
    "vision", "batching", "kv_quant", "native_tool_calls", "reasoning_field",
})


def _vlm_available() -> tuple[bool, str]:
    if importlib.util.find_spec("mlx_vlm") is None:
        return False, "mlx-vlm is not installed in this environment (pip install mlx-vlm)"
    return True, ""


def _vlm_int(m: Any, name: str) -> int:
    try:
        return int(getattr(m, name, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _vlm_max_num_seqs(m: Any) -> int:
    """The running-batch bound: DECODE_CONCURRENCY, else the admission limit.

    mlx-vlm's default is unbounded, which on a small Mac turns a burst of
    requests into a burst of KV caches; the app's own concurrency settings are
    the bound the operator already chose for the other engines.
    """
    return _vlm_int(m, "decode_concurrency") or _vlm_int(m, "max_concurrent_generations")


def vlm_command(m: Any, use_adapter: bool = False) -> list[str]:
    """The mlx_vlm.server argv for a manager (help text sniffed for optional flags).

    --model, --host and --port exist in every mlx-vlm server release, so they
    are passed unconditionally. Everything else is passed only when the
    installed build's --help lists it: an unknown flag is an argparse exit, and
    a server that never starts is worse than one without a memory lever.
    """
    help_text = getattr(m, "_server_help", "") or ""
    cmd = [sys.executable, "-m", VLM_SERVER_MODULE,
           "--model", str(getattr(m, "model_id", "") or ""),
           "--host", "127.0.0.1",
           "--port", str(getattr(m, "model_port", 0) or 0)]
    accepted: list[str] = []
    max_kv = _vlm_int(m, "max_kv_size")
    if max_kv > 0 and add_if_supported(cmd, help_text, ["--max-kv-size"], str(max_kv)):
        accepted.append("--max-kv-size")
    kv_bits = _vlm_int(m, "kv_bits")
    if kv_bits > 0 and add_if_supported(cmd, help_text, ["--kv-bits"], str(kv_bits)):
        accepted.append("--kv-bits")
        group = _vlm_int(m, "kv_group_size")
        if group > 0:
            add_if_supported(cmd, help_text, ["--kv-group-size"], str(group))
        start = _vlm_int(m, "quantized_kv_start")
        if start > 0:
            add_if_supported(cmd, help_text, ["--quantized-kv-start"], str(start))
    step = _vlm_int(m, "prefill_step_size")
    if step > 0 and add_if_supported(cmd, help_text, ["--prefill-step-size"], str(step)):
        accepted.append("--prefill-step-size")
    seqs = _vlm_max_num_seqs(m)
    if seqs > 0 and add_if_supported(cmd, help_text, ["--max-num-seqs"], str(seqs)):
        accepted.append("--max-num-seqs")
    try:
        m._accepted_levers = accepted
    except Exception:  # noqa: BLE001
        pass
    return cmd


def _vlm_launcher(m: Any, use_adapter: bool) -> list[str]:
    return vlm_command(m, use_adapter)


def _vlm_request_model(config: Any) -> str:
    # mlx-vlm loads request.model by path when it differs from the preloaded
    # one, so the only safe value is the exact --model id (a Config's `model`,
    # a manager's `model_id`).
    for attr in ("model_id", "model"):
        value = getattr(config, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _vlm_effective_caps(m: Any) -> frozenset:
    caps = {"vision", "native_tool_calls", "reasoning_field"}
    if _vlm_max_num_seqs(m) != 1:
        caps.add("batching")
    if _vlm_int(m, "kv_bits") > 0:
        caps.add("kv_quant")
    return frozenset(caps)


def _vlm_decide(m: Any) -> list[str]:
    out = ["Images: accepted as image_url content parts (mlx-vlm serves the vision tower)."]
    seqs = _vlm_max_num_seqs(m)
    out.append(f"Batching: continuous, at most {seqs} sequences at once." if seqs > 1
               else "Batching: off (one sequence at a time)." if seqs == 1
               else "Batching: continuous, unbounded (set DECODE_CONCURRENCY to bound it).")
    if _vlm_int(m, "kv_bits") > 0:
        out.append(f"KV cache: quantized to {_vlm_int(m, 'kv_bits')} bits.")
    out.append("LoRA adapters: not served (this app trains mlx-lm adapters; mlx-vlm "
               "uses its own adapter layout).")
    return out


register_backend(BackendSpec(
    name=VLM_BACKEND,
    label="mlx-vlm (vision models)",
    help_module=VLM_SERVER_MODULE,
    launcher=_vlm_launcher,
    fatal_patterns=tuple(METAL_FATAL_PATTERNS),
    capabilities=_VLM_STATIC_CAPS,
    available=_vlm_available,
    request_model=_vlm_request_model,
    description=("mlx-vlm's OpenAI-compatible server: serves vision-language models "
                 "(Qwen2.5-VL, Qwen3-VL, Gemma 3, ...) with image input, continuous "
                 "batching and optional KV quantization."),
    decide=_vlm_decide,
    effective_capabilities=_vlm_effective_caps,
    supports_adapter=False,
    launch_inputs=_VLM_LAUNCH_INPUTS,
))


# --------------------------------------------------------------------------- #
# Does the served model see images?
# --------------------------------------------------------------------------- #
def _vlm_mlxlm_processor_types() -> set[str]:
    """Model types the installed mlx-lm can build an image processor for."""
    try:
        import mlx_lm.processors as _procs  # noqa: PLC0415 (heavy, optional)
    except Exception:  # noqa: BLE001
        return set()
    table = getattr(_procs, "PROCESSORS", None)
    return {str(k) for k in table} if isinstance(table, dict) else set()


def model_has_vision_config(model_id: str) -> bool | None:
    """True/False from the cached config.json (vision_config present), None if unknown."""
    cfg = _backends_model_config(str(model_id or ""))
    if not cfg:
        return None
    return bool(cfg.get("vision_config"))


def served_vision(manager: Any = None, config: Any = None) -> tuple[bool, str]:
    """(can the served model read image parts, why) for the chat route.

    VISION_MODE=on/off overrides the detection for setups it cannot see (an
    external backend command, a remote engine). In auto, both the engine and
    the model must agree, because each alone has lied before: mlx-lm declares
    "vision" for every model, and a vision model behind an engine that cannot
    run its vision tower sees nothing.
    """
    mode = str(getattr(config, "vision_mode", "auto") or "auto").strip().lower()
    if mode == "off":
        return False, "image input is switched off (VISION_MODE=off)"
    if mode == "on":
        return True, "VISION_MODE=on"
    spec = getattr(manager, "backend", None)
    if not isinstance(spec, BackendSpec):
        spec = get_backend(getattr(config, "model_backend", None)
                           or getattr(manager, "backend_name", None) or DEFAULT_BACKEND)
    try:
        caps = spec.caps_for(manager)
    except Exception:  # noqa: BLE001
        caps = frozenset(spec.capabilities)
    if "vision" not in caps:
        return False, f"the {spec.name} engine does not accept images"
    model = str(getattr(manager, "model_id", "") or getattr(config, "model", "") or "")
    meta = model_metadata(model, guess=False) if model else None
    listed = meta.get("vision") if isinstance(meta, dict) else None
    if listed is False:
        return False, f"{model} is a text-only model"
    if spec.name == VLM_BACKEND:
        return True, f"{model} on mlx-vlm"
    has_tower = model_has_vision_config(model)
    if spec.name == DEFAULT_BACKEND:
        # mlx-lm runs a vision tower only for the processor types it ships.
        cfg = _backends_model_config(model) if has_tower else {}
        kind = str(cfg.get("processor_type") or cfg.get("model_type") or "")
        if has_tower and kind and kind in _vlm_mlxlm_processor_types():
            return True, f"{model} on mlx-lm ({kind} processor)"
        return False, (f"{model} is served as text by mlx-lm; use MODEL_BACKEND=mlx-vlm "
                       "for image input")
    if listed is True or has_tower:
        return True, f"{model} on {spec.name}"
    return False, f"{model} is not known to accept images"


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'VISION_MODES',
    'VLM_BACKEND',
    'VLM_SERVER_MODULE',
    '_VLM_LAUNCH_INPUTS',
    '_VLM_STATIC_CAPS',
    '_vlm_available',
    '_vlm_decide',
    '_vlm_effective_caps',
    '_vlm_launcher',
    '_vlm_max_num_seqs',
    '_vlm_mlxlm_processor_types',
    '_vlm_request_model',
    'model_has_vision_config',
    'served_vision',
    'vlm_command',
]
