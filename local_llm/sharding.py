"""Layer-sharded serving across Macs (mlx distributed) and `deploy.py cluster`.

Request routing (cluster.py) sends each generation to ONE machine that holds
the whole model, so the largest model the cluster can serve is the largest
model one Mac can hold. Layer sharding (pipeline parallelism) splits a model's
layers across machines instead, so two 64GB Macs can serve what neither fits
alone. This module plans and launches that with the stock tools; it does not
reimplement any of them.

Facts read from the installed packages (mlx 0.32.4, mlx-lm 0.32.2) that shape
this module, with the file and line they come from:

  * Launcher: `mlx.launch` (.venv/bin/mlx.launch -> mlx/_distributed_utils/
    launch.py:main). Options used here: --hostfile, --backend, --python, and
    the command after `--` (launch.py:499-565). The command is prefixed with
    --python on every rank (launch.py:152-154), so the same interpreter path
    must exist on every host.
  * Hostfile format (mlx/_distributed_utils/common.py:37-91): JSON, either a
    list of hosts or {"backend": "...", "envs": [...], "hosts": [...]}; each
    host is {"ssh": "<ssh target>", "ips": ["<ip>", ...], "rdma": [...]}.
    Unknown host keys are ignored (common.py:82), so this module stores each
    host's "ram_gb" there for planning.
  * A host whose ssh is exactly "127.0.0.1" runs locally; every other host is
    reached with `ssh -tt <ssh>` (launch.py:47-55), so passwordless SSH from
    the primary to each other host is required. The ring backend (the default
    on Apple Silicon without CUDA, launch.py:587-588) needs IPs, not just
    hostnames (launch.py:300-304); jaccl needs Thunderbolt RDMA.
  * Rank 0 runs the HTTP server; the other ranks only join the computation
    (mlx_lm/server.py:1905-1920). The primary is therefore always rank 0 with
    ssh "127.0.0.1", so the app's model client keeps talking to
    127.0.0.1:<model_port> exactly as with a single-machine server.
  * `mlx_lm.server --pipeline` selects pipeline parallelism; without it a
    multi-rank group uses tensor parallelism (server.py:380-385).
  * The split is EVEN: sharded_load calls model.model.pipeline(group) with no
    split (mlx_lm/utils.py:762), and PipelineMixin then gives each rank
    len(layers)//size layers, the low ranks one extra (models/pipeline.py:21-24),
    in reverse: rank 0 holds the LAST layers (pipeline.py:17-18, 34-35). Every
    rank also keeps the embeddings, final norm and head (utils.py:772-774).
    PipelineMixin.pipeline accepts a custom split, but the stock server never
    passes one, so plan_shards checks the even split against each host's free
    memory rather than pretending a RAM-weighted split is applied.
  * Pipeline loading reads model.safetensors.index.json (utils.py:766), so a
    single-file checkpoint without an index cannot be pipeline-sharded, and the
    model's class must implement pipeline() (utils.py:739-751).
  * Adapters and draft models are refused in distributed mode: "Loading with
    adapters or draft models not supported in distributed mode"
    (mlx_lm/server.py:410-416). The feedback LoRA is therefore either NOT
    served (base model only) or fused into new weights first:
        python -m mlx_lm.fuse --model <base> --adapter-path <adapters> \\
            --save-path <fused-dir>
    then MODEL_ID=<fused-dir> on every host (mlx_lm/fuse.py:18-60).

NOT verified on this machine: there is one 8GB Mac here, so no multi-host
launch, ring connection, or sharded generation has been run. The planner, the
hostfile writer and the command builder are tested offline; the launch itself
is the stock mlx.launch.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import re
import shlex
import shutil
import socket
import struct
import sys
from pathlib import Path
from typing import Any

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .sysutil import *  # noqa: F401,F403
from .admission import *  # noqa: F401,F403
from .backends import *  # noqa: F401,F403
from .nodesec import *  # noqa: F401,F403
from .cluster import *  # noqa: F401,F403


SHARD_BACKEND = "mlx-distributed"
_SHARD_GIB = 1024 ** 3
_SHARD_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_shard_capable_cache: list = []


# --------------------------------------------------------------------------- #
# Node facts (also reported by /api/node/health)
# --------------------------------------------------------------------------- #
def node_free_ram_gb() -> float | None:
    """Memory available for new allocations now, in GB (None if unmeasurable)."""
    free = free_memory_bytes()
    return None if free is None else round(free / _SHARD_GIB, 2)


def node_backend_name(config: Any) -> str:
    name = (getattr(config, "model_backend", "") or os.environ.get("MODEL_BACKEND", "")
            or DEFAULT_BACKEND)
    return str(name).strip().lower()


def mlx_launch_path() -> str:
    """The mlx.launch console script next to this interpreter, or on PATH."""
    beside = Path(sys.executable).parent / "mlx.launch"
    if beside.exists():
        return str(beside)
    return shutil.which("mlx.launch") or ""


def node_shard_capable() -> bool:
    """Can this machine be a rank of a layer-sharded group (mlx + mlx.launch)?"""
    if not _shard_capable_cache:
        ok = (importlib.util.find_spec("mlx") is not None
              and importlib.util.find_spec("mlx_lm") is not None
              and bool(mlx_launch_path()))
        _shard_capable_cache.append(ok)
    return _shard_capable_cache[0]


# --------------------------------------------------------------------------- #
# Model geometry from the checkpoint itself
# --------------------------------------------------------------------------- #
def _shard_safetensors_header(path: Path) -> dict:
    """The JSON header of a .safetensors file (8-byte LE length + JSON)."""
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path.name}: truncated safetensors file")
        (length,) = struct.unpack("<Q", raw)
        if length > 100 * 1024 * 1024:
            raise ValueError(f"{path.name}: implausible header length {length}")
        return json.loads(fh.read(length))


def _shard_model_dir(model: str) -> Path | None:
    """A local model directory, or the cached HF snapshot holding config.json.

    Its own lookup (admission.py keeps its finder private): the pinned
    refs/main snapshot first, then any snapshot with a config.json.
    """
    try:
        local = Path(model).expanduser()
        if local.is_dir():
            return local if (local / "config.json").is_file() else None
        repo = hf_cache_dir() / ("models--" + str(model).replace("/", "--"))
        snapshots, ref = repo / "snapshots", repo / "refs" / "main"
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


def _shard_model_source_has(model_type: str, needle: str) -> bool | None:
    try:
        spec = importlib.util.find_spec(f"mlx_lm.models.{model_type}")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    try:
        return needle in Path(spec.origin).read_text(encoding="utf-8")
    except OSError:
        return None


def model_shard_info(model: str) -> dict:
    """Per-layer weight bytes, shared bytes and KV cost, read from disk.

    Exact rather than estimated: each tensor's byte range comes from the
    safetensors headers, so quantized and mixed-precision checkpoints are
    measured as they are. Nothing is loaded into memory.
    """
    model_dir = _shard_model_dir(model)
    if model_dir is None:
        raise FileNotFoundError(f"{model}: no local copy (config.json) found; download it "
                                "first or pass a local model directory")
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    src = cfg if "num_hidden_layers" in cfg else (cfg.get("text_config") or cfg)
    num_layers = int(src.get("num_hidden_layers") or 0)
    if num_layers <= 0:
        raise ValueError(f"{model}: config.json has no num_hidden_layers")
    per_layer = [0] * num_layers
    shared = 0
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"{model}: no .safetensors weights in {model_dir}")
    for path in files:
        for name, meta in _shard_safetensors_header(path).items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            start, end = meta.get("data_offsets") or (0, 0)
            size = int(end) - int(start)
            match = _SHARD_LAYER_RE.search(name)
            if match is None:
                shared += size
            elif int(match.group(1)) < num_layers:
                per_layer[int(match.group(1))] += size
            # Layers past the model's own count are MTP heads that sanitize
            # drops (utils.py:782-787): they never reach memory.
    model_type = str(src.get("model_type") or cfg.get("model_type") or "")
    return {
        "model": model, "dir": str(model_dir), "model_type": model_type,
        "num_layers": num_layers, "layer_bytes": per_layer, "shared_bytes": shared,
        "total_bytes": shared + sum(per_layer),
        "kv_bytes_per_token": estimate_kv_bytes_per_token(str(model_dir)) or 0,
        "has_index": (model_dir / "model.safetensors.index.json").exists(),
        # Most models get pipeline() by inheriting PipelineMixin (qwen2.py:12)
        # rather than defining it; utils.py:739 only asks hasattr(pipeline).
        "pipeline_supported": (_shard_model_source_has(model_type, "PipelineMixin")
                               or _shard_model_source_has(model_type, "def pipeline")),
        "tensor_supported": _shard_model_source_has(model_type, "def shard"),
    }


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
def mlx_even_split(num_layers: int, ranks: int) -> list[int]:
    """Layers per rank exactly as PipelineMixin.pipeline computes them."""
    base, extra = divmod(num_layers, ranks)
    return [base + (1 if r < extra else 0) for r in range(ranks)]


def mlx_rank_layers(split: list[int], rank: int) -> tuple[int, int]:
    """[start, end) layer range of a rank: rank 0 holds the LAST layers."""
    start = sum(split[rank + 1:])
    return start, start + split[rank]


def plan_shards(info: dict, hosts: list[dict], *, context_tokens: int = 8192,
                kv_headroom_gb: float | None = None, reserve_gb: float = 0.5,
                reorder: bool = True) -> dict:
    """Check whether `hosts` can serve the model pipeline-sharded, per rank.

    hosts[0] is the primary (rank 0, HTTP server) and keeps its place; with
    reorder the others are sorted by free memory so the ranks that receive
    an extra layer are the roomiest. Each rank needs its layers' weights, the
    shared tensors (embeddings, norm, head), the KV cache for context_tokens
    over its own layers (or a flat kv_headroom_gb) and a runtime reserve, and
    the plan is REFUSED (ok False) when any rank's free memory is short.
    """
    notes: list[str] = []
    if not hosts:
        return {"ok": False, "reason": "no hosts", "ranks": [], "split": [], "notes": notes}
    hosts = list(hosts)
    if reorder and len(hosts) > 2:
        hosts = [hosts[0]] + sorted(hosts[1:], key=lambda h: -(h.get("free_ram_gb") or 0))
    layers = int(info["num_layers"])
    n = len(hosts)
    if n > layers:
        return {"ok": False, "reason": f"{n} hosts for {layers} layers", "ranks": [],
                "split": [], "notes": notes}
    split = mlx_even_split(layers, n)
    kv_per_layer_token = (info.get("kv_bytes_per_token") or 0) / max(1, layers)
    ranks, short = [], []
    for r, host in enumerate(hosts):
        start, end = mlx_rank_layers(split, r)
        weights = (info["shared_bytes"] + sum(info["layer_bytes"][start:end])) / _SHARD_GIB
        if kv_headroom_gb is not None:
            kv = float(kv_headroom_gb)
        else:
            kv = kv_per_layer_token * (end - start) * max(0, int(context_tokens)) / _SHARD_GIB
        need = weights + kv + reserve_gb
        free = host.get("free_ram_gb")
        fits = free is not None and float(free) >= need
        ranks.append({"rank": r, "name": host.get("name") or host.get("ssh") or f"rank{r}",
                      "ssh": host.get("ssh", ""), "free_ram_gb": free,
                      "layers": [start, end], "layer_count": end - start,
                      "weights_gb": round(weights, 2), "kv_gb": round(kv, 2),
                      "need_gb": round(need, 2),
                      "spare_gb": None if free is None else round(float(free) - need, 2),
                      "fits": fits})
        if not fits:
            short.append(f"{ranks[-1]['name']} needs {need:.1f} GB "
                         + ("but its free memory is unknown" if free is None
                            else f"but has {float(free):.1f} GB free"))
    if info.get("pipeline_supported") is False:
        short.append(f"mlx-lm's {info.get('model_type')!r} model has no pipeline() "
                     "(mlx_lm/utils.py:742-745 would refuse --pipeline)")
    if info.get("has_index") is False:
        short.append("the checkpoint has no model.safetensors.index.json, which pipeline "
                     "loading reads (mlx_lm/utils.py:766)")
    whole = info["total_bytes"] / _SHARD_GIB
    if hosts[0].get("free_ram_gb") and whole + reserve_gb <= float(hosts[0]["free_ram_gb"]):
        notes.append(f"the whole model ({whole:.1f} GB) already fits on {ranks[0]['name']}; "
                     "sharding only adds network latency per token")
    notes.append("split is even (mlx_lm/utils.py:762); rank 0 holds the last layers and "
                 "serves HTTP")
    notes.append("adapters are not served in distributed mode (mlx_lm/server.py:410-416); "
                 "fuse them with mlx_lm.fuse to serve the trained model")
    return {"ok": not short, "reason": "; ".join(short) or "fits", "split": split,
            "ranks": ranks, "notes": notes, "model": info.get("model"),
            "model_gb": round(whole, 2), "context_tokens": context_tokens}


# --------------------------------------------------------------------------- #
# Hostfile + launch command
# --------------------------------------------------------------------------- #
def shard_default_hostfile(config: Any = None) -> Path:
    explicit = str(getattr(config, "cluster_hostfile", "") or
                   os.environ.get("CLUSTER_HOSTFILE", "")).strip()
    if explicit:
        return Path(explicit).expanduser()
    if config is not None and hasattr(config, "settings_path"):
        return Path(config.settings_path()).parent / "cluster_hostfile.json"
    return DATA_DIR / "cluster_hostfile.json"


def write_hostfile(path: Path | str, hosts: list[dict], backend: str = "ring",
                   envs: list[str] | None = None) -> dict:
    """Write an mlx.launch hostfile (common.py:37-91 format). Returns the data."""
    if not hosts:
        raise ValueError("at least one host is required")
    clean = []
    for i, host in enumerate(hosts):
        ssh = str(host.get("ssh") or "").strip()
        ips = [str(ip).strip() for ip in (host.get("ips") or []) if str(ip).strip()]
        if not ssh:
            raise ValueError(f"host {i} has no ssh target")
        if backend == "ring" and not ips:
            raise ValueError(f"host {ssh!r} needs an IP: the ring backend connects by IP "
                             "(mlx/_distributed_utils/launch.py:300-304)")
        entry: dict = {"ssh": ssh, "ips": ips}
        if host.get("rdma"):
            entry["rdma"] = host["rdma"]
        if host.get("ram_gb"):
            entry["ram_gb"] = float(host["ram_gb"])
        clean.append(entry)
    data = {"backend": backend, "envs": list(envs or []), "hosts": clean}
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


def read_hostfile(path: Path | str) -> dict:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {"backend": "", "envs": [], "hosts": data}
    if not isinstance(data, dict) or not isinstance(data.get("hosts"), list):
        raise ValueError(f"{path}: not an mlx hostfile (expected a hosts list)")
    return data


def distributed_launch_command(hostfile: Path | str, model: str, port: int, *,
                               host: str = "127.0.0.1", python: str | None = None,
                               backend: str = "", pipeline: bool = True,
                               extra: list[str] | None = None) -> list[str]:
    """argv that starts mlx_lm.server on every rank via mlx.launch."""
    launcher = mlx_launch_path()
    cmd = ([launcher] if launcher else
           [sys.executable, "-c", "from mlx._distributed_utils.launch import main; main()"])
    cmd += ["--hostfile", str(hostfile)]
    if backend:
        cmd += ["--backend", backend]
    cmd += ["--python", python or sys.executable, "--", "-m", "mlx_lm.server",
            "--model", model, "--host", host, "--port", str(port)]
    if pipeline:
        cmd.append("--pipeline")
    return cmd + list(extra or [])


# --------------------------------------------------------------------------- #
# The "mlx-distributed" backend
# --------------------------------------------------------------------------- #
def _shard_launcher(m: Any, use_adapter: bool) -> list[str]:
    hostfile = shard_default_hostfile(getattr(m, "config", None))
    m._accepted_levers = ["pipeline"]
    return distributed_launch_command(
        hostfile, m.model_id, int(m.model_port),
        host=getattr(m, "host", "127.0.0.1") or "127.0.0.1",
        python=os.environ.get("CLUSTER_PYTHON") or None)


def _shard_available(config: Any = None) -> tuple[bool, str]:
    if not node_shard_capable():
        return False, "mlx / mlx-lm / mlx.launch are not installed in this environment"
    path = shard_default_hostfile(config)
    if not path.exists():
        return False, f"no hostfile at {path}; create one with `deploy.py cluster hostfile`"
    try:
        hosts = read_hostfile(path)["hosts"]
    except (OSError, ValueError) as exc:
        return False, str(exc)
    if len(hosts) < 2:
        return False, "the hostfile lists fewer than two hosts"
    return True, ""


def _shard_decide(m: Any) -> list[str]:
    path = shard_default_hostfile(getattr(m, "config", None))
    notes = [f"Hostfile {path}; rank 0 (this machine) serves HTTP on 127.0.0.1:"
             f"{getattr(m, 'model_port', '?')}.",
             "The LoRA adapter and any draft model are NOT served: mlx-lm refuses them in "
             "distributed mode (server.py:410-416). Fuse the adapter with mlx_lm.fuse and "
             "point MODEL_ID at the fused weights to serve the trained model.",
             "Layers are split evenly across hosts (mlx_lm/utils.py:762); check the fit "
             "with `deploy.py cluster plan` first."]
    return notes


register_backend(BackendSpec(
    name=SHARD_BACKEND,
    label="mlx distributed (layer-sharded)",
    help_module="",
    launcher=_shard_launcher,
    fatal_patterns=MLXLM_FATAL_PATTERNS,
    capabilities=frozenset({"distributed", "prompt_cache", "reasoning_field",
                            "native_tool_calls"}),
    available=_shard_available,
    request_model=lambda _config: "default_model",
    description=("mlx_lm.server --pipeline across the Macs in CLUSTER_HOSTFILE, started with "
                 "mlx.launch; for models no single machine can hold. Base weights only "
                 "(or pre-fused adapters). Unverified on real multi-host hardware here."),
    decide=_shard_decide,
    supports_adapter=False,
))


def _shard_registry_hook(registry: Any) -> None:
    """Make the primary the virtual node of its shard group.

    With MODEL_BACKEND=mlx-distributed the local endpoint is backed by every
    host in the hostfile, so the primary advertises the capabilities of the
    group's combined memory (a 2 x 64GB group can take large_model work).
    """
    config = registry.config
    if node_backend_name(config) != SHARD_BACKEND:
        return
    path = shard_default_hostfile(config)
    try:
        hosts = read_hostfile(path)["hosts"]
    except (OSError, ValueError) as exc:
        log(f"mlx-distributed: cannot read hostfile {path}: {exc}", logging.WARNING)
        return
    primary = registry.local_node()
    primary.shard_group = [str(h.get("ssh")) for h in hosts if isinstance(h, dict)]
    primary.backend = SHARD_BACKEND
    primary.shard_capable = True
    total = sum(float(h.get("ram_gb") or 0) for h in hosts if isinstance(h, dict))
    if total > primary.ram_gb:
        primary.capabilities = capabilities_for_machine(total)
        primary.detail = f"shard group of {len(hosts)} hosts, {total:.0f} GB total"


if _shard_registry_hook not in CLUSTER_REGISTRY_HOOKS:
    CLUSTER_REGISTRY_HOOKS.append(_shard_registry_hook)


# --------------------------------------------------------------------------- #
# deploy.py cluster plan|hostfile|status|join
# --------------------------------------------------------------------------- #
def _shard_parse_hosts(items: list[str]) -> list[dict]:
    """NAME=FREE_GB entries for `plan`."""
    hosts = []
    for item in items:
        name, _, gb = item.partition("=")
        try:
            hosts.append({"name": name.strip(), "free_ram_gb": float(gb)})
        except ValueError:
            raise SystemExit(f"--host expects NAME=FREE_GB, got {item!r}")
    return hosts


def _shard_parse_hostfile_hosts(items: list[str]) -> list[dict]:
    """SSH=IP[,IP...][@RAM_GB] entries for `hostfile`."""
    hosts = []
    for item in items:
        ssh, _, rest = item.partition("=")
        ips, _, ram = rest.partition("@")
        try:
            ram_gb = float(ram) if ram else 0.0
        except ValueError:
            raise SystemExit(f"--host expects SSH=IP[,IP][@RAM_GB], got {item!r}")
        hosts.append({"ssh": ssh.strip(), "ips": [i for i in ips.split(",") if i.strip()],
                      "ram_gb": ram_gb})
    return hosts


def _shard_live_hosts(config: Any) -> list[dict]:
    """Free memory of this machine and every configured/paired peer, probed now."""
    registry = NodeRegistry(config)
    registry.load_paired()
    monitor = HealthMonitor(config, registry)
    hosts = [{"name": registry.local_node().name, "free_ram_gb": node_free_ram_gb()}]
    for node in registry.nodes:
        if node.is_local:
            continue
        monitor._probe_remote(node)
        hosts.append({"name": node.name, "free_ram_gb": node.free_ram_gb,
                      "state": node.state, "detail": node.detail})
    return hosts


def _shard_print_plan(plan: dict) -> None:
    print(f"Model {plan.get('model')} ({plan.get('model_gb')} GB of weights), "
          f"split {plan['split']}, context {plan.get('context_tokens')} tokens")
    print(f"  {'rank':<5}{'host':<22}{'layers':<12}{'weights':>9}{'kv':>7}{'need':>8}"
          f"{'free':>8}  fits")
    for r in plan["ranks"]:
        free = "?" if r["free_ram_gb"] is None else f"{float(r['free_ram_gb']):.1f}"
        print(f"  {r['rank']:<5}{r['name'][:21]:<22}{str(r['layers']):<12}"
              f"{r['weights_gb']:>9.2f}{r['kv_gb']:>7.2f}{r['need_gb']:>8.2f}{free:>8}"
              f"  {'yes' if r['fits'] else 'NO'}")
    print(("OK: " if plan["ok"] else "REFUSED: ") + plan["reason"])
    for note in plan["notes"]:
        print(f"  note: {note}")


def _shard_cmd_plan(args: Any, config: Any) -> int:
    try:
        info = model_shard_info(args.model or config.model)
    except (OSError, ValueError) as exc:
        print(f"cannot read the model: {exc}", file=sys.stderr)
        return 2
    hosts = _shard_parse_hosts(args.host) if args.host else _shard_live_hosts(config)
    if len(hosts) < 2:
        print("only one host known: pass --host NAME=FREE_GB for each machine, or "
              "configure CLUSTER_NODES / pair nodes first", file=sys.stderr)
    plan = plan_shards(info, hosts, context_tokens=args.context,
                       kv_headroom_gb=args.kv_headroom_gb, reserve_gb=args.reserve_gb)
    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        _shard_print_plan(plan)
    return 0 if plan["ok"] else 1


def _shard_cmd_hostfile(args: Any, config: Any) -> int:
    hosts = _shard_parse_hostfile_hosts(args.host)
    if not hosts:
        print("pass --host SSH=IP[@RAM_GB] per machine, this machine first as "
              "127.0.0.1=<its LAN or Thunderbolt IP>", file=sys.stderr)
        return 2
    if hosts[0]["ssh"] != "127.0.0.1":
        print(f"note: rank 0 ({hosts[0]['ssh']}) will be reached over ssh; use 127.0.0.1 "
              "for this machine so the HTTP server runs here (launch.py:49)")
    path = Path(args.out).expanduser() if args.out else shard_default_hostfile(config)
    try:
        write_hostfile(path, hosts, backend=args.backend)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Wrote {path}")
    cmd = distributed_launch_command(path, config.model, config.model_port,
                                     backend=args.backend)
    print("The app starts it with MODEL_BACKEND=mlx-distributed"
          + ("" if path == shard_default_hostfile(config) else f" CLUSTER_HOSTFILE={path}")
          + "; by hand it is:")
    print("  " + shlex.join(cmd))
    print("Every host needs passwordless ssh from here, the same python at "
          f"{sys.executable}, mlx-lm, and the model (or the fused model directory).")
    return 0


def _shard_cmd_status(args: Any, config: Any) -> int:
    registry = NodeRegistry(config)
    registry.load_paired()
    monitor = HealthMonitor(config, registry)
    local = registry.local_node()
    print(f"{'node':<20}{'state':<13}{'proto':<6}{'ram':>6}{'free':>7}{'load':>6}"
          f"{'queue':>6}  model / detail")
    print(f"{local.name[:19]:<20}{'local':<13}{'-':<6}{local.ram_gb:>6.0f}"
          f"{(node_free_ram_gb() or 0):>7.1f}{(sample_load_ratio() or 0):>6.2f}{'-':>6}  "
          f"{config.model}  backend={node_backend_name(config)}")
    worst = 0
    for node in registry.nodes:
        if node.is_local:
            continue
        monitor._probe_remote(node)
        free = "-" if node.free_ram_gb is None else f"{node.free_ram_gb:.1f}"
        load = "-" if node.load_ratio is None else f"{node.load_ratio:.2f}"
        queue = "-" if node.queue is None else str(node.queue)
        print(f"{node.name[:19]:<20}{node.state:<13}{node.protocol or '-':<6}"
              f"{node.ram_gb:>6.0f}{free:>7}{load:>6}{queue:>6}  "
              f"{node.model or ''} {node.detail}".rstrip())
        if node.state in (UNAVAILABLE, DEGRADED):
            worst = 1
    path = shard_default_hostfile(config)
    if path.exists():
        try:
            hf = read_hostfile(path)
            print(f"shard hostfile {path}: {len(hf['hosts'])} hosts, backend "
                  f"{hf.get('backend') or 'default'}")
        except (OSError, ValueError) as exc:
            print(f"shard hostfile {path}: unreadable ({exc})")
    return worst


def _shard_update_env_token(env_path: Path, token: str) -> str:
    """Add token to NODE_TOKEN in an env file (comma list). Returns a message."""
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    for i, line in enumerate(lines):
        if line.strip().startswith("NODE_TOKEN="):
            current = [t.strip() for t in line.split("=", 1)[1].strip().strip('"').split(",")
                       if t.strip()]
            if token in current:
                return f"{env_path}: NODE_TOKEN already contains the token"
            lines[i] = "NODE_TOKEN=" + ",".join([token] + current)
            break
    else:
        lines.append(f"NODE_TOKEN={token}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(env_path, 0o600)
    except OSError:
        pass
    return f"{env_path}: NODE_TOKEN updated (restart this node to apply)"


def _shard_cmd_join(args: Any, config: Any) -> int:
    import httpx
    primary = args.primary_url.rstrip("/")
    problem = node_url_problem(config, primary)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    machine = detect_machine()
    host = socket.gethostname().split(".")[0] or "node"
    body = {"code": args.code, "name": args.name or machine["name"],
            "url": args.url or f"http://{host}.local:{config.web_port}",
            "ram_gb": args.ram_gb if args.ram_gb is not None else machine["ram_gb"]}
    try:
        with httpx.Client(verify=node_ssl_context(config), timeout=20.0) as client:
            resp = client.post(primary + "/api/cluster/join", json=body)
    except (httpx.HTTPError, NodeSecurityError) as exc:
        print(f"cannot reach {primary}: {exc}", file=sys.stderr)
        return 1
    data = resp.json() if resp.headers.get("content-type", "").startswith(
        "application/json") else {}
    if resp.status_code != 200:
        print(f"join refused ({resp.status_code}): {data.get('error') or resp.text[:200]}",
              file=sys.stderr)
        return 1
    token = data["node_token"]
    print(f"Paired as {data.get('name')!r} with primary {data.get('primary')!r}.")
    if args.env_file:
        print(_shard_update_env_token(Path(args.env_file).expanduser(), token))
    else:
        print("Add this to the joining node's .env (comma-separate it from any existing "
              "NODE_TOKEN), restart it, then restart the primary:")
        print(f"NODE_TOKEN={token}")
    return 0


def cluster_cli(argv: list[str], config: Any = None) -> int:
    """`deploy.py cluster <plan|hostfile|status|join> ...`."""
    parser = argparse.ArgumentParser(prog="deploy.py cluster",
                                     description="Multi-Mac cluster helpers.")
    sub = parser.add_subparsers(dest="cmd")
    p = sub.add_parser("plan", help="Check whether a model fits layer-sharded across hosts.")
    p.add_argument("--model", default="")
    p.add_argument("--host", action="append", default=[],
                   help="NAME=FREE_GB (repeat; first is rank 0). Default: probe the nodes.")
    p.add_argument("--context", type=int, default=8192,
                   help="KV headroom is sized for this many tokens of context.")
    p.add_argument("--kv-headroom-gb", type=float, default=None,
                   help="Flat KV headroom per host instead of the context-derived one.")
    p.add_argument("--reserve-gb", type=float, default=0.5)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("hostfile", help="Write the mlx.launch hostfile.")
    p.add_argument("--host", action="append", default=[],
                   help="SSH=IP[,IP][@RAM_GB] (repeat; this machine first as 127.0.0.1=IP)")
    p.add_argument("--backend", default="ring", choices=["ring", "jaccl", "jaccl-ring", "mpi"])
    p.add_argument("--out", default="")
    sub.add_parser("status", help="Probe every configured or paired node.")
    p = sub.add_parser("join", help="Pair this machine with a primary.")
    p.add_argument("primary_url")
    p.add_argument("--code", required=True)
    p.add_argument("--name", default="")
    p.add_argument("--url", default="", help="This node's URL as the primary reaches it.")
    p.add_argument("--ram-gb", type=float, default=None)
    p.add_argument("--env-file", default="", help="Write the token into this .env file.")
    args = parser.parse_args(argv)
    config = config or Config()
    handlers = {"plan": _shard_cmd_plan, "hostfile": _shard_cmd_hostfile,
                "status": _shard_cmd_status, "join": _shard_cmd_join}
    if args.cmd not in handlers:
        parser.print_help()
        return 2
    return handlers[args.cmd](args, config)


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'SHARD_BACKEND', 'node_free_ram_gb', 'node_backend_name', 'mlx_launch_path',
    'node_shard_capable', 'model_shard_info', 'mlx_even_split', 'mlx_rank_layers',
    'plan_shards', 'shard_default_hostfile', 'write_hostfile', 'read_hostfile',
    'distributed_launch_command', 'cluster_cli',
    '_shard_safetensors_header', '_shard_registry_hook', '_shard_update_env_token',
    '_shard_launcher', '_shard_available',
]
