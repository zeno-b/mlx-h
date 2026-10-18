"""Automatic Mac Mini (primary) / Mac Studio (secondary) routing and failover.

This is a real scheduler/router subsystem, not an ``if machine == ...`` switch.
Users never pick a machine; work is classified, nodes are evaluated on live
load/health/capability, one is selected with a recorded reason, and execution
fails over to another node without re-running committed work.

Pieces
------
* :class:`Node` — one execution target (the local mlx server, or a remote peer),
  with capabilities, capacity and live health state.
* :class:`NodeRegistry` — builds the node set from config: the primary is always
  the local model server; peers come from ``CLUSTER_NODES`` (any number),
  ``STUDIO_NODE_URL`` (the one-peer shorthand) or, when ``CLUSTER_NODES`` is
  unset, nodes paired with ``deploy.py cluster join`` (DATA_DIR/cluster_nodes.json,
  read when the heartbeat starts). With no peer the registry has one node and
  routing is a no-op.
* :class:`ClusterRouter` — classifies a task, scores candidate nodes against
  configurable factors (active requests, queue depth, CPU/memory, node health,
  model capability, SLA), selects an ordered candidate list with a human reason,
  and records every decision to ``routing_events`` for admin observability.
* :class:`HealthMonitor` — a background heartbeat loop that probes each node and
  moves it between healthy / degraded / overloaded / draining / unavailable /
  starting, so a failed Studio never breaks the Mini and eligible Mini work
  shifts to the Studio when it is overloaded or down.

The router is exercised offline by the self-test with injected node snapshots and
a fake executor: overloaded-primary, unavailable-primary, large-model,
unavailable-secondary and recovery all have deterministic outcomes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import secrets
import re
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .core import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .nodesec import *  # noqa: F401,F403
from .remote_access import *  # noqa: F401,F403

# Request must resolve at MODULE scope for FastAPI (see api.py); a unique name
# because the single-file build shares one namespace with api.py's own import.
try:
    from fastapi import Request as _cluster_Request  # noqa: F401
except Exception:  # pragma: no cover - selftest without FastAPI
    _cluster_Request = Any  # type: ignore


_cluster_log = get_logger("routing")

# Node health states, from best to worst for scheduling.
HEALTHY = "healthy"
DEGRADED = "degraded"       # responding but slow / recent failures
OVERLOADED = "overloaded"   # at/over capacity; avoid unless nothing else
DRAINING = "draining"       # finishing in-flight work, take no new work
STARTING = "starting"       # coming up (loading weights)
UNAVAILABLE = "unavailable"  # not reachable / failing

# Order used when otherwise-equal: lower is preferred.
_STATE_RANK = {HEALTHY: 0, DEGRADED: 1, OVERLOADED: 2, STARTING: 3,
               DRAINING: 4, UNAVAILABLE: 5}


@dataclass
class Node:
    """One routable execution target."""
    name: str
    role: str                    # "primary" | "secondary"
    is_local: bool               # local mlx server vs a remote peer app
    capabilities: set[str] = field(default_factory=set)
    remote_url: str = ""         # base URL for a remote node (empty if local)
    # Live state (guarded by the registry lock).
    state: str = STARTING
    active: int = 0              # in-flight generations dispatched here
    last_latency_ms: float | None = None
    consecutive_failures: int = 0
    last_heartbeat: float = 0.0
    cpu_pct: float | None = None
    mem_pct: float | None = None
    # 1-minute load average divided by core count. A saturation RATIO (1.0 ==
    # fully committed), deliberately NOT reported as a percentage: it routinely
    # exceeds 1.0 on a healthy machine and used to be mislabelled as "cpu 100%",
    # which flagged an idle laptop as overloaded.
    load_ratio: float | None = None
    model: str | None = None
    # Detected hardware, so the cluster panel shows the real machine.
    machine: str = ""
    cores: int = 0
    ram_gb: float = 0.0
    detail: str = ""
    # Circuit breaker: when the node trips to UNAVAILABLE it is skipped until this
    # timestamp, after which one half-open trial request is allowed through.
    cooldown_until: float = 0.0
    # Reported by a protocol-2 peer (/api/node/health) or detected locally.
    free_ram_gb: float | None = None
    queue: int | None = None     # generations waiting in the node's admission queue
    backend: str = ""
    shard_capable: bool = False
    protocol: int = 0            # 0 = not probed yet, 1 = legacy peer, 2 = current
    # Capabilities written in CLUSTER_NODES are kept as given; otherwise they
    # follow the RAM the node reports, so an 8GB peer never attracts 32B work.
    caps_fixed: bool = False
    # A paired node's own bearer token (cluster_nodes.json). Never in snapshot().
    token: str = field(default="", repr=False)
    # Last proof of life (a good heartbeat or a successful dispatch); the
    # router stops trusting a peer silent for longer than heartbeat_timeout.
    last_ok_heartbeat: float = 0.0
    # Hosts of the layer-sharded group this node fronts (sharding.py).
    shard_group: list = field(default_factory=list)

    def can_serve(self, requirements: "Requirements") -> bool:
        if requirements.needs_large_model and "large_model" not in self.capabilities:
            return False
        for cap in requirements.required_capabilities:
            if cap not in self.capabilities:
                return False
        return True

    def snapshot(self) -> dict:
        return {
            "name": self.name, "role": self.role, "is_local": self.is_local,
            "state": self.state, "active": self.active,
            "capabilities": sorted(self.capabilities),
            "last_latency_ms": self.last_latency_ms,
            "consecutive_failures": self.consecutive_failures,
            "cpu_pct": self.cpu_pct, "mem_pct": self.mem_pct,
            "load_ratio": self.load_ratio, "machine": self.machine,
            "cores": self.cores, "ram_gb": self.ram_gb,
            "model": self.model, "detail": self.detail,
            "url": node_safe_url(self.remote_url) or None,
            "free_ram_gb": self.free_ram_gb, "queue": self.queue,
            "backend": self.backend, "shard_capable": self.shard_capable,
            "protocol": self.protocol, "shard_group": list(self.shard_group),
            "paired": bool(self.token),
            "cooldown_remaining_s": (round(self.cooldown_until - time.time(), 1)
                                     if self.cooldown_until > time.time() else None),
            "last_heartbeat_age_s": (round(time.time() - self.last_heartbeat, 1)
                                     if self.last_heartbeat else None),
        }


@dataclass
class Requirements:
    """What a task needs, derived by classification."""
    kind: str = "chat"                       # chat | reasoning | code | task
    needs_large_model: bool = False
    required_capabilities: tuple[str, ...] = ()
    # Soft preference (not a hard requirement): heavy work advertises the
    # capabilities that make a node a better fit (e.g. "reasoning" for the
    # Studio), so the router steers toward it when it is ready and has capacity.
    prefer_capabilities: tuple[str, ...] = ()
    requested_model: str | None = None
    complexity: str = "normal"               # low | normal | high


@dataclass
class RoutingDecision:
    """The result of a selection: an ordered candidate list plus the why."""
    candidates: list[Node]
    reason: str
    requirements: Requirements
    snapshot: list[dict]

    @property
    def primary_choice(self) -> Node | None:
        return self.candidates[0] if self.candidates else None


# Apple model identifiers -> a short, readable node name.
_MODEL_NAMES = (
    ("macbookpro", "macbook-pro"), ("macbookair", "macbook-air"), ("macbook", "macbook"),
    ("macmini", "mac-mini"), ("macstudio", "mac-studio"), ("macpro", "mac-pro"),
    ("imacpro", "imac-pro"), ("imac", "imac"), ("virtualmac", "mac-vm"),
)

_MACHINE: dict | None = None


def friendly_node_name(model: str, hostname: str) -> str:
    """A short node name from the hardware model, falling back to the hostname."""
    key = re.sub(r"[^a-z]", "", (model or "").lower())
    for prefix, nice in _MODEL_NAMES:
        if key.startswith(prefix):
            return nice
    host = re.sub(r"[^a-z0-9-]", "", (hostname or "").lower())
    return host or (key or "local-node")


def detect_machine() -> dict:
    """What machine is this, really? Cached; probed once per process.

    The node used to be hardcoded to "mac-mini" regardless of hardware, which
    is simply wrong on any other Mac. This reads the actual model identifier so
    the registry and the admin panel describe the real machine.
    """
    global _MACHINE
    if _MACHINE is not None:
        return _MACHINE
    model = ""
    try:
        model = subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True,
                               text=True, timeout=2).stdout.strip()
    except Exception:
        model = ""
    if not model:
        try:
            model = platform.machine()
        except Exception:
            model = ""
    host = ""
    try:
        host = socket.gethostname().split(".")[0]
    except Exception:
        host = ""
    _MACHINE = {
        "model": model,
        "hostname": host,
        "cores": os.cpu_count() or 1,
        "ram_gb": float(TOTAL_RAM_GB or 0),
        "name": friendly_node_name(model, host),
    }
    return _MACHINE


def capabilities_for_machine(ram_gb: float) -> set:
    """Capabilities a local node can honestly advertise for its RAM.

    Only a roomy machine may claim the high-memory / large-model / deep-reasoning
    capabilities the router steers heavy work to; an 8GB laptop must not.
    """
    caps = {"chat", "code", "default"}
    if ram_gb >= 32:
        caps |= {"high_memory", "large_model", "reasoning"}
    return caps


def sample_load_ratio() -> float | None:
    """1-minute load average per core. A ratio, not a percentage."""
    try:
        return round(os.getloadavg()[0] / (os.cpu_count() or 1), 2)
    except (OSError, AttributeError):
        return None


def sample_local_load() -> tuple[float | None, float | None]:
    """Best-effort (cpu_pct, mem_pct) for this machine.

    cpu_pct is a REAL utilisation sample and is None when we cannot measure one.
    It used to be derived from the load average (load/cores*100), which is not a
    utilisation figure at all: on an idle 8-core laptop a 1-minute load of 17
    produced "cpu 100%" and tripped the overload threshold. Better to report
    nothing than to report a number that means something else -- the router
    skips a factor it cannot measure (see sample_load_ratio for saturation).
    """
    cpu_pct = None
    mem_pct = None
    try:
        import psutil  # optional
        cpu_pct = float(psutil.cpu_percent(interval=None))
        mem_pct = float(psutil.virtual_memory().percent)
    except Exception:
        pass      # psutil absent: both stay None, i.e. "unmeasured", not "idle"
    return cpu_pct, mem_pct


_CLUSTER_BASE_CAPS = frozenset({"chat", "code", "default"})
_CLUSTER_STUDIO_CAPS = frozenset({"chat", "code", "default", "large_model",
                                  "high_memory", "reasoning"})
# Callables(registry) run at the end of NodeRegistry._build (see sharding.py).
CLUSTER_REGISTRY_HOOKS: list[Callable[[Any], None]] = []


def _cluster_slug(text: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9-]", "-", str(text or "").lower())).strip("-")[:40]


def _cluster_slug_from_url(url: str) -> str:
    try:
        from urllib.parse import urlsplit
        host = urlsplit(url).hostname or ""
    except ValueError:
        host = ""
    if host and not re.fullmatch(r"[0-9.:a-f]+", host):
        host = host.split(".")[0]
    return _cluster_slug(host) or "node"


def cluster_node_specs(config: Any) -> list[dict]:
    """Peer specs from CLUSTER_NODES plus the STUDIO_NODE_URL alias.

    CLUSTER_NODES is a JSON list (objects with url and optional name, ram_gb,
    capabilities; or bare URL strings) or comma-separated URLs. Bad entries
    are logged and skipped rather than failing startup. Tokens are NOT read
    from here (the value is shown in /api/config); paired nodes carry theirs
    in cluster_nodes.json.
    """
    raw = str(getattr(config, "cluster_nodes", "") or "").strip()
    specs: list[dict] = []
    entries: list = []
    if raw.startswith("["):
        try:
            entries = json.loads(raw)
        except ValueError as exc:
            log(f"CLUSTER_NODES is not valid JSON ({exc}); ignoring it.", logging.WARNING)
            entries = []
    elif raw:
        entries = [u.strip() for u in raw.split(",") if u.strip()]
    for entry in entries:
        if isinstance(entry, str):
            entry = {"url": entry}
        if not isinstance(entry, dict) or not str(entry.get("url") or "").strip():
            log(f"CLUSTER_NODES: skipping entry without a url: {entry!r}"[:200], logging.WARNING)
            continue
        caps = entry.get("capabilities")
        try:
            ram = float(entry.get("ram_gb") or 0)
        except (TypeError, ValueError):
            ram = 0.0
        specs.append({"name": str(entry.get("name") or ""), "url": str(entry["url"]).strip(),
                      "ram_gb": max(0.0, ram),
                      "capabilities": [str(c) for c in caps] if isinstance(caps, list) else None,
                      "source": "cluster_nodes"})
    studio = str(getattr(config, "studio_node_url", "") or "").strip()
    if studio and not any(s["url"].rstrip("/") == studio.rstrip("/") for s in specs):
        specs.append({"name": "mac-studio", "url": studio, "legacy_superset": True,
                      "source": "studio_node_url"})
    return specs


# --------------------------------------------------------------------------- #
# Paired nodes (deploy.py cluster join): DATA_DIR/cluster_nodes.json
# --------------------------------------------------------------------------- #
def cluster_paired_path(config: Any) -> Path:
    # Derived from settings_path() so it follows DATA_DIR as config.py sees it
    # (tests repoint config.DATA_DIR before anything is built).
    return Path(config.settings_path()).parent / "cluster_nodes.json"


def read_paired_nodes(config: Any) -> list[dict]:
    path = cluster_paired_path(config)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as exc:
        log(f"Cannot read {path}: {exc}", logging.WARNING)
        return []
    nodes = data.get("nodes") if isinstance(data, dict) else None
    return [n for n in (nodes or []) if isinstance(n, dict) and n.get("url") and n.get("name")]


def write_paired_nodes(config: Any, nodes: list[dict]) -> None:
    """Atomic, owner-only write: the file holds each node's bearer token."""
    path = cluster_paired_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump({"nodes": nodes}, fh, indent=2)
    os.replace(tmp, path)


class NodeRegistry:
    """The set of nodes, built from config, with thread-safe state updates."""

    def __init__(self, config: Config):
        self.config = config
        self._lock = threading.RLock()
        self.nodes: list[Node] = []
        self._build()

    def _build(self) -> None:
        cfg = self.config
        machine = detect_machine()
        primary = Node(
            name=cfg.node_name or machine["name"],
            role="primary", is_local=True,
            capabilities=capabilities_for_machine(machine["ram_gb"]),
            state=STARTING,
            machine=machine["model"], cores=machine["cores"], ram_gb=machine["ram_gb"],
        )
        self.nodes = [primary]
        for spec in cluster_node_specs(cfg):
            self.add_remote(spec)
        # Feature modules (sharding.py) adjust the node set here, e.g. turning
        # the primary into the front of a layer-sharded group.
        for hook in list(CLUSTER_REGISTRY_HOOKS):
            try:
                hook(self)
            except Exception as exc:  # noqa: BLE001 - a hook must never break startup
                log_event(_cluster_log, 30, "cluster.registry_hook_failed", error=str(exc))

    def add_remote(self, spec: dict) -> Node | None:
        """Add one peer from a spec (see cluster_node_specs); None if a duplicate."""
        url = str(spec.get("url") or "").strip().rstrip("/")
        if not url:
            return None
        with self._lock:
            if any(n.remote_url == url for n in self.nodes):
                return None
            taken = {n.name for n in self.nodes}
            base = _cluster_slug(spec.get("name") or "") or _cluster_slug_from_url(url)
            name, i = base, 2
            while name in taken:
                name, i = f"{base}-{i}", i + 1
            ram = float(spec.get("ram_gb") or 0)
            caps_given = spec.get("capabilities")
            if caps_given:
                caps = {str(c) for c in caps_given}
            elif ram > 0:
                caps = capabilities_for_machine(ram)
            elif spec.get("legacy_superset"):
                # STUDIO_NODE_URL has always meant "the high-memory Mac Studio":
                # keep that promise until the node reports its real RAM.
                caps = set(_CLUSTER_STUDIO_CAPS)
            else:
                caps = set(_CLUSTER_BASE_CAPS)
            node = Node(name=name, role="secondary", is_local=False, remote_url=url,
                        capabilities=caps, state=STARTING, ram_gb=ram,
                        caps_fixed=bool(caps_given), token=str(spec.get("token") or ""))
            problem = self.refusal(node)
            if problem:
                node.state, node.detail = UNAVAILABLE, problem[:200]
                log(f"Cluster node {name} ({node_safe_url(url)}) is not usable: {problem}",
                    logging.WARNING)
            self.nodes.append(node)
            return node

    def refusal(self, node: Node) -> str:
        """Why traffic to this peer is refused by policy ("" when allowed)."""
        if node.is_local:
            return ""
        return node_url_problem(self.config, node.remote_url) or node_e2e_problem(self.config)

    def load_paired(self) -> int:
        """Add nodes paired via /api/cluster/join. Only when CLUSTER_NODES is unset.

        Called when the heartbeat starts rather than from the constructor, so a
        throwaway registry (the self-test, the CLI) never picks up this
        machine's paired peers by accident.
        """
        if str(getattr(self.config, "cluster_nodes", "") or "").strip():
            return 0
        added = 0
        for entry in read_paired_nodes(self.config):
            if self.add_remote(dict(entry, source="paired")) is not None:
                added += 1
        if added:
            log(f"Cluster: {added} paired node(s) from {cluster_paired_path(self.config)}")
        return added

    @property
    def multi_node(self) -> bool:
        return len(self.nodes) > 1

    def local_node(self) -> Node:
        for node in self.nodes:
            if node.is_local:
                return node
        return self.nodes[0]

    def by_name(self, name: str) -> Node | None:
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def update(self, name: str, **fields: Any) -> None:
        with self._lock:
            node = self.by_name(name)
            if not node:
                return
            for key, value in fields.items():
                if hasattr(node, key):
                    setattr(node, key, value)

    def begin(self, node: Node) -> None:
        with self._lock:
            node.active += 1

    def end(self, node: Node, ok: bool, latency_ms: float | None = None) -> None:
        with self._lock:
            node.active = max(0, node.active - 1)
            if ok:
                node.last_ok_heartbeat = time.time()
                # A success closes the breaker: a half-open trial that worked, or
                # ordinary healthy traffic, both clear the failure state.
                node.consecutive_failures = 0
                node.cooldown_until = 0.0
                if latency_ms is not None:
                    node.last_latency_ms = round(latency_ms, 1)
                if node.state in (UNAVAILABLE, STARTING, DEGRADED):
                    node.state = HEALTHY
            else:
                node.consecutive_failures += 1
                if node.consecutive_failures >= 2:
                    node.state = UNAVAILABLE
                    # Open the breaker: skip this node until the cooldown elapses,
                    # then allow one half-open trial (see ClusterRouter._eligible).
                    node.cooldown_until = time.time() + max(
                        0.0, getattr(self.config, "route_cooldown_s", 20.0))
                elif node.state == HEALTHY:
                    node.state = DEGRADED

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [n.snapshot() for n in self.nodes]


class ClusterRouter:
    """Classify a task, pick nodes, record the decision, and fail over."""

    def __init__(self, config: Config, registry: NodeRegistry, db: Any = None):
        self.config = config
        self.registry = registry
        self.db = db
        # In-flight task ids, so an accidental re-dispatch of the same unit of
        # work is detected rather than silently duplicated.
        self._inflight: set[str] = set()
        self._inflight_lock = threading.Lock()

    # ---- dispatch accounting ---------------------------------------------- #
    # ModelClient calls router.begin/end around every dispatch. These used to
    # exist only on NodeRegistry, so the moment a second node was configured
    # every generation raised AttributeError before reaching any node.
    def begin(self, node: Node) -> None:
        self.registry.begin(node)

    def end(self, node: Node, ok: bool, latency_ms: float | None = None) -> None:
        self.registry.end(node, ok, latency_ms)

    # ---- classification --------------------------------------------------- #
    def classify(self, *, model: str | None = None, kind: str = "chat",
                 complexity: str = "normal") -> Requirements:
        model = model or self.config.model
        markers = [m.strip().lower() for m in
                   (self.config.large_model_markers or "").split(",") if m.strip()]
        needs_large = any(m in (model or "").lower() for m in markers)
        req_caps: tuple[str, ...] = ()
        prefer: tuple[str, ...] = ()
        if kind == "reasoning" or complexity == "high":
            # Deep reasoning is eligible for (not required on) the Studio: a soft
            # preference, so heavy work steers toward the high-capability node
            # when it is ready, but light chat is never forced off the primary.
            prefer = ("reasoning",)
        return Requirements(kind=kind, needs_large_model=needs_large,
                            required_capabilities=req_caps,
                            prefer_capabilities=prefer,
                            requested_model=model, complexity=complexity)

    # ---- capacity / health helpers --------------------------------------- #
    def _is_overloaded(self, node: Node) -> tuple[bool, str]:
        cfg = self.config
        if node.active >= cfg.route_max_active_per_node:
            return True, f"active {node.active}>={cfg.route_max_active_per_node}"
        # Requests already waiting in the node's own admission queue: past
        # route_queue_depth, new work is better started somewhere idle.
        if (cfg.route_queue_depth and node.queue is not None
                and node.queue >= cfg.route_queue_depth):
            return True, f"queue {node.queue}>={cfg.route_queue_depth}"
        if node.cpu_pct is not None and node.cpu_pct >= cfg.route_cpu_pct:
            return True, f"cpu {node.cpu_pct}%>={cfg.route_cpu_pct}%"
        if node.mem_pct is not None and node.mem_pct >= cfg.route_mem_pct:
            return True, f"mem {node.mem_pct}%>={cfg.route_mem_pct}%"
        if (node.load_ratio is not None and cfg.route_load_ratio
                and node.load_ratio >= cfg.route_load_ratio):
            return True, f"load {node.load_ratio}x>={cfg.route_load_ratio}x per core"
        if (cfg.route_sla_ms and node.last_latency_ms is not None
                and node.last_latency_ms >= cfg.route_sla_ms):
            return True, f"latency {node.last_latency_ms}ms>=SLA {cfg.route_sla_ms}ms"
        return False, ""

    def _is_stale(self, node: Node) -> bool:
        """A peer with no proof of life for heartbeat_timeout is not trusted.

        Guards against a stuck heartbeat thread leaving a dead peer HEALTHY.
        Never applies before the first successful probe (selftest snapshots).
        """
        return (not node.is_local and node.last_ok_heartbeat > 0
                and node.state != UNAVAILABLE
                and time.time() - node.last_ok_heartbeat > self.config.heartbeat_timeout)

    def _eligible(self, node: Node, req: Requirements) -> bool:
        if not node.can_serve(req) or node.state == DRAINING:
            return False
        if self._is_stale(node):
            return False
        if node.state == UNAVAILABLE:
            # Half-open: a tripped node is eligible again only after its cooldown
            # elapses, and even then it scores worst (see _score), so it is used
            # only as a single trial when nothing healthier can serve the work.
            return node.cooldown_until > 0 and time.time() >= node.cooldown_until
        return True

    def _is_half_open(self, node: Node) -> bool:
        return (node.state == UNAVAILABLE and node.cooldown_until > 0
                and time.time() >= node.cooldown_until)

    def _score(self, node: Node) -> float:
        """Lower is better. Orders eligible nodes by health, then live load.

        Health dominates (a healthy node always beats a degraded one); within a
        health tier the least-busy, lowest-latency, lowest-utilisation node wins,
        so a second request does not pile onto a node already working.
        """
        score = _STATE_RANK.get(node.state, 9) * 1000.0
        score += node.active * 50.0
        if node.cpu_pct is not None:
            score += node.cpu_pct
        if node.mem_pct is not None:
            score += node.mem_pct
        if node.last_latency_ms is not None:
            score += node.last_latency_ms / 50.0
        return score

    # ---- selection -------------------------------------------------------- #
    def select(self, req: Requirements) -> RoutingDecision:
        """Return an ordered candidate list (best first) and the reason."""
        nodes = self.registry.nodes
        snapshot = self.registry.snapshot()

        # Single-node install: the local node is the only answer.
        if len(nodes) == 1:
            only = nodes[0]
            reason = "single node (no secondary configured)"
            if not only.can_serve(req):
                reason = ("single node cannot meet requirement "
                          f"(needs_large_model={req.needs_large_model}); using it anyway")
            return RoutingDecision([only], reason, req, snapshot)

        primary = self.registry.local_node()

        eligible = [n for n in nodes if self._eligible(n, req)]
        reasons: list[str] = []

        # Hard capability need (large model): only capable nodes qualify.
        if req.needs_large_model:
            capable = [n for n in eligible if "large_model" in n.capabilities]
            if capable:
                capable.sort(key=self._score)
                order = capable + sorted((n for n in eligible if n not in capable),
                                         key=self._score)
                reasons.append(f"large model {req.requested_model!r} requires "
                               "high-memory node")
                return RoutingDecision(order, "; ".join(reasons), req, snapshot)
            # No capable node up: fall through to best-effort below.
            reasons.append("no high-memory node available for large model; best effort")

        # Soft capability preference (heavy reasoning): steer toward a ready
        # capability node (the Studio) with spare capacity, keeping the primary
        # as the immediate fallback. Light chat has no preference and falls
        # through to the primary-first path below.
        if req.prefer_capabilities:
            preferred = [
                n for n in eligible
                if n is not primary
                and all(c in n.capabilities for c in req.prefer_capabilities)
                and n.state in (HEALTHY, DEGRADED)
                and not self._is_overloaded(n)[0]
            ]
            if preferred:
                preferred.sort(key=self._score)
                chosen = preferred[0]
                rest = sorted((n for n in eligible if n is not chosen), key=self._score)
                reasons.append(f"heavy {req.kind} work -> capability node "
                               f"{sorted(req.prefer_capabilities)}")
                return RoutingDecision([chosen] + rest, "; ".join(reasons), req, snapshot)

        # Ordinary case: prefer the primary unless it is unfit. A node still
        # loading weights (STARTING) is eligible but not ready to be *preferred*
        # over a healthy secondary, so it is treated like an overloaded primary.
        if primary in eligible:
            over, why = self._is_overloaded(primary)
            ready = primary.state in (HEALTHY, DEGRADED)
            if not over and ready:
                order = [primary] + sorted((n for n in eligible if n is not primary),
                                           key=self._score)
                reasons.append("primary healthy and within capacity")
                return RoutingDecision(order, "; ".join(reasons), req, snapshot)
            reasons.append(f"primary overloaded ({why})" if over
                           else f"primary not ready ({primary.state})")
            # Offload to the least-loaded eligible secondary if one can take it.
            others = sorted((n for n in eligible if n is not primary), key=self._score)
            takeable = [n for n in others if not self._is_overloaded(n)[0]]
            if takeable:
                pick = takeable[0]
                reasons.append("secondary has capacity -> offload"
                               + (" (half-open trial)" if self._is_half_open(pick) else ""))
                rest = [primary] + [n for n in others if n is not pick]
                return RoutingDecision([pick] + rest, "; ".join(reasons), req, snapshot)
            if others:
                reasons.append("secondary also loaded; keep on primary")
                return RoutingDecision([primary] + others, "; ".join(reasons), req, snapshot)
            reasons.append("secondary unavailable; keep on primary")
            return RoutingDecision([primary], "; ".join(reasons), req, snapshot)

        # Primary not eligible (down/incapable): use the best eligible secondary.
        others = sorted((n for n in eligible if n is not primary), key=self._score)
        if others:
            pick = others[0]
            reasons.append("primary unavailable -> secondary"
                           + (" (half-open trial)" if self._is_half_open(pick) else ""))
            return RoutingDecision(others, "; ".join(reasons), req, snapshot)

        # Nothing eligible: return the least-bad node so the caller degrades
        # honestly rather than 500ing with no target.
        fallback = sorted(nodes, key=self._score)
        reasons.append("no healthy node; best-effort fallback")
        return RoutingDecision(fallback, "; ".join(reasons), req, snapshot)

    # ---- observability ---------------------------------------------------- #
    def record(self, decision: RoutingDecision, node: Node, *, status: str,
               attempt: int = 0, duration_ms: float | None = None,
               error: str | None = None, correlation_id: str | None = None,
               task_id: str | None = None, user_id: str | None = None,
               conversation_id: str | None = None) -> None:
        log_event(_cluster_log, 20, "routing.decision",
                  node=node.name, status=status, attempt=attempt,
                  kind=decision.requirements.kind,
                  needs_large_model=decision.requirements.needs_large_model,
                  requested_model=decision.requirements.requested_model,
                  reason=decision.reason, duration_ms=duration_ms,
                  correlation_id=correlation_id, task_id=task_id)
        if self.db is not None:
            try:
                self.db.log_routing_event(
                    correlation_id=correlation_id, task_id=task_id, user_id=user_id,
                    conversation_id=conversation_id, kind=decision.requirements.kind,
                    requested_model=decision.requirements.requested_model,
                    selected_model=node.model or decision.requirements.requested_model,
                    selected_node=node.name, reason=decision.reason,
                    candidates=decision.snapshot, status=status, attempt=attempt,
                    duration_ms=duration_ms, error=error)
            except Exception as exc:  # never let telemetry break a request
                log_event(_cluster_log, 30, "routing.record_failed", error=str(exc))

    # ---- idempotency ------------------------------------------------------ #
    def claim(self, task_id: str) -> bool:
        """Register a unit of work. False if it is already in flight (a dup)."""
        if not task_id:
            return True
        with self._inflight_lock:
            if task_id in self._inflight:
                return False
            self._inflight.add(task_id)
            return True

    def release(self, task_id: str) -> None:
        if not task_id:
            return
        with self._inflight_lock:
            self._inflight.discard(task_id)


def _cluster_num(value: Any, kind: type = float) -> Any:
    try:
        return None if value is None or value == "" else kind(value)
    except (TypeError, ValueError):
        return None


def remote_health_state(info: dict) -> tuple[str, str]:
    """Map a peer's /api/node/health status to a node state, with a reason.

    Anything unrecognised (including the model manager's "error: ..." text, or
    an empty status) is DEGRADED. It used to fall through to HEALTHY, so a peer
    whose model had crashed kept attracting work.
    """
    raw = str(info.get("state") or info.get("model_status") or "").strip()
    key = raw.lower()
    if key in ("ready", "healthy", "ok"):
        return HEALTHY, ""
    if key in ("loading", "starting", "restarting"):
        return STARTING, ""
    if key in ("stopped", "unavailable"):
        return UNAVAILABLE, f"remote status {raw!r}"
    if key == "overloaded":
        return OVERLOADED, ""
    if key == "draining":
        return DRAINING, ""
    return DEGRADED, f"remote status {raw or 'missing'!r}"[:200]


def remote_health_fields(info: dict, protocol_header: Any = None) -> dict:
    """Registry fields from a peer's health answer (protocol 1 or 2)."""
    state, detail = remote_health_state(info)
    fields: dict[str, Any] = {
        "state": state, "detail": detail, "model": info.get("model"),
        "cpu_pct": _cluster_num(info.get("cpu_pct")),
        "mem_pct": _cluster_num(info.get("mem_pct")),
    }
    protocol = _cluster_num(protocol_header or info.get("protocol"), int) or 1
    fields["protocol"] = protocol
    if protocol >= 2:
        for key in ("load_ratio", "free_ram_gb"):
            fields[key] = _cluster_num(info.get(key))
        ram = _cluster_num(info.get("ram_gb"))
        if ram:
            fields["ram_gb"] = ram
        fields["queue"] = _cluster_num(info.get("queue"), int)
        fields["cores"] = _cluster_num(info.get("cores"), int) or 0
        fields["machine"] = str(info.get("machine") or "")[:64]
        fields["backend"] = str(info.get("backend") or "")[:64]
        fields["shard_capable"] = bool(info.get("shard_capable"))
    return fields


class HealthMonitor:
    """Background heartbeat loop that keeps node states current."""

    def __init__(self, config: Config, registry: NodeRegistry,
                 local_status: Callable[[], dict] | None = None):
        self.config = config
        self.registry = registry
        # Callable returning {"status": <mlx status>, "model": <id>} for the
        # local model server (wired to ModelServerManager in create_app).
        self.local_status = local_status
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        try:
            self.registry.load_paired()
        except Exception as exc:  # noqa: BLE001 - a bad file must not stop the app
            log_event(_cluster_log, 30, "cluster.paired_load_failed", error=str(exc))
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="cluster-heartbeat")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=2)

    def _loop(self) -> None:
        # Probe once promptly, then on the configured interval.
        while True:
            try:
                self.tick()
            except Exception as exc:  # never let the monitor thread die
                log_event(_cluster_log, 40, "heartbeat.error", error=str(exc))
            if self._stop.wait(max(1.0, self.config.heartbeat_interval)):
                return

    def tick(self) -> None:
        """One heartbeat pass over every node."""
        cpu_pct, mem_pct = sample_local_load()
        load_ratio = sample_load_ratio()
        for node in list(self.registry.nodes):
            if node.is_local:
                self._probe_local(node, cpu_pct, mem_pct, load_ratio)
            else:
                self._probe_remote(node)

    def _probe_local(self, node: Node, cpu_pct, mem_pct, load_ratio=None) -> None:
        state = HEALTHY
        detail = ""
        model = node.model
        if self.local_status is not None:
            try:
                info = self.local_status() or {}
                status = str(info.get("status", ""))
                model = info.get("model") or model
                # Optional extras (api.py): admission queue, free RAM, engine.
                extra = {k: info[k] for k in ("free_ram_gb", "backend")
                         if info.get(k) is not None}
                # "queue" is the admission snapshot (a dict) or a bare count.
                queue = info.get("queue")
                if isinstance(queue, dict):
                    queue = queue.get("waiting")
                if queue is not None:
                    extra["queue"] = _cluster_num(queue, int)
                if extra:
                    self.registry.update(node.name, **extra)
                if status == "ready":
                    state = HEALTHY
                elif status in ("loading", "starting", "restarting"):
                    state = STARTING
                elif status == "stopped":
                    state = UNAVAILABLE
                else:
                    state = DEGRADED
                    detail = status
            except Exception as exc:
                state, detail = DEGRADED, str(exc)
        # Capacity pressure downgrades an otherwise-healthy node. Every factor is
        # skipped when it cannot be measured -- an unmeasured signal must never
        # read as "over the limit" (that is what made an idle laptop overloaded).
        if state == HEALTHY:
            cfg = self.config
            over_cpu = cpu_pct is not None and cpu_pct >= cfg.route_cpu_pct
            over_mem = mem_pct is not None and mem_pct >= cfg.route_mem_pct
            over_load = (load_ratio is not None and cfg.route_load_ratio
                         and load_ratio >= cfg.route_load_ratio)
            over_active = node.active >= cfg.route_max_active_per_node
            over_queue = bool(cfg.route_queue_depth and node.queue is not None
                              and node.queue >= cfg.route_queue_depth)
            if over_cpu or over_mem or over_load or over_active or over_queue:
                state = OVERLOADED
        fields = dict(state=state, cpu_pct=cpu_pct, mem_pct=mem_pct,
                      load_ratio=load_ratio, model=model,
                      detail=detail, last_heartbeat=time.time())
        # A healthy probe closes the breaker, exactly as _probe_remote and
        # end(ok=True) do: otherwise a primary that tripped to UNAVAILABLE and
        # recovered keeps consecutive_failures>=2, so the next single failure
        # re-trips it straight to UNAVAILABLE instead of degrading first.
        if state == HEALTHY:
            fields["consecutive_failures"] = 0
            fields["cooldown_until"] = 0.0
        self.registry.update(node.name, **fields)

    def _probe_remote(self, node: Node) -> None:
        cfg = self.config
        url = node.remote_url.rstrip("/") + "/api/node/health"
        # heartbeat_timeout also caps one probe: a peer that cannot answer
        # within it is, by definition, not heartbeating.
        timeout = max(1.0, min(cfg.node_probe_timeout, cfg.heartbeat_timeout))
        started = time.time()
        try:
            with node_sync_client(cfg, timeout=timeout) as client:
                resp = client.get(url, headers=node_auth_headers(cfg, node))
            latency = (time.time() - started) * 1000
            if resp.status_code != 200:
                self._mark_unreachable(node, f"health {resp.status_code}")
                return
            info = resp.json()
            if not isinstance(info, dict):
                raise ValueError("health answer is not a JSON object")
            fields = remote_health_fields(info, resp.headers.get(NODE_PROTOCOL_HEADER))
            ram = fields.get("ram_gb") or 0
            if ram and not node.caps_fixed:
                fields["capabilities"] = capabilities_for_machine(ram)
            now = time.time()
            fields.update(last_latency_ms=round(latency, 1), consecutive_failures=0,
                          last_heartbeat=now, last_ok_heartbeat=now)
            if fields["state"] != UNAVAILABLE:
                fields["cooldown_until"] = 0.0
            self.registry.update(node.name, **fields)
        except NodeSecurityError as exc:
            self._mark_unreachable(node, f"refused: {exc}")
        except Exception as exc:
            self._mark_unreachable(node, str(exc))

    def _mark_unreachable(self, node: Node, detail: str) -> None:
        failures = node.consecutive_failures + 1
        silent = (node.last_ok_heartbeat > 0
                  and time.time() - node.last_ok_heartbeat > self.config.heartbeat_timeout)
        state = UNAVAILABLE if failures >= 2 or silent else DEGRADED
        fields = dict(state=state, consecutive_failures=failures,
                      last_heartbeat=time.time(), detail=detail[:200])
        # Opening the breaker from the heartbeat too, so the router's half-open
        # trial logic works whether the failure was seen by a request or a probe.
        if state == UNAVAILABLE:
            fields["cooldown_until"] = time.time() + max(
                0.0, getattr(self.config, "route_cooldown_s", 20.0))
        self.registry.update(node.name, **fields)
        log_event(_cluster_log, 30, "heartbeat.node_unreachable",
                  node=node.name, detail=detail[:200], state=state)


# --------------------------------------------------------------------------- #
# Pairing bootstrap: POST /api/cluster/pairing (admin) + POST /api/cluster/join
# --------------------------------------------------------------------------- #
# Adding a node used to mean hand-copying a shared secret into two .env files
# and agreeing on URLs. An admin now issues a short-lived, single-use code on
# the primary; `deploy.py cluster join` on the new node trades it for that
# node's own bearer token. Codes are stored only as hashes; the join endpoint
# is unauthenticated by design (the code IS the credential), so failures are
# rate-limited per client and in total.
_CLUSTER_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # no 0/O, 1/I
_CLUSTER_JOIN_WINDOW_S = 600.0
_CLUSTER_JOIN_MAX_PER_IP = 5
_CLUSTER_JOIN_MAX_TOTAL = 30


def _cluster_code_hash(code: str) -> str:
    norm = re.sub(r"[^A-Z0-9]", "", str(code or "").upper())
    return hashlib.sha256(("cluster-pairing\n" + norm).encode()).hexdigest()


class PairingBook:
    """Single-use pairing codes (hashed, expiring) plus the join throttle."""

    def __init__(self) -> None:
        self._codes: dict[str, float] = {}
        self._fails: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def issue(self, ttl_s: float) -> tuple[str, float]:
        raw = "".join(secrets.choice(_CLUSTER_CODE_ALPHABET) for _ in range(10))
        code = f"{raw[:5]}-{raw[5:]}"
        now = time.time()
        expires = now + ttl_s
        with self._lock:
            for key in [k for k, exp in self._codes.items() if exp < now]:
                del self._codes[key]
            self._codes[_cluster_code_hash(code)] = expires
        return code, expires

    def redeem(self, code: str) -> bool:
        """True once per valid, unexpired code; the code is consumed either way."""
        with self._lock:
            expires = self._codes.pop(_cluster_code_hash(code), None)
        return expires is not None and expires >= time.time()

    def pending(self) -> int:
        now = time.time()
        with self._lock:
            return sum(1 for exp in self._codes.values() if exp >= now)

    def _prune(self, now: float) -> None:
        horizon = now - _CLUSTER_JOIN_WINDOW_S
        for ip in list(self._fails):
            kept = [t for t in self._fails[ip] if t >= horizon]
            if kept:
                self._fails[ip] = kept
            else:
                del self._fails[ip]

    def retry_in(self, ip: str, now: float) -> int:
        with self._lock:
            self._prune(now)
            mine = self._fails.get(ip, [])
            if len(mine) >= _CLUSTER_JOIN_MAX_PER_IP:
                return max(1, int(mine[0] + _CLUSTER_JOIN_WINDOW_S - now) + 1)
            everyone = sorted(t for times in self._fails.values() for t in times)
            if len(everyone) >= _CLUSTER_JOIN_MAX_TOTAL:
                return max(1, int(everyone[0] + _CLUSTER_JOIN_WINDOW_S - now) + 1)
            return 0

    def fail(self, ip: str, now: float) -> None:
        with self._lock:
            self._fails.setdefault(ip, []).append(now)


def register_cluster(app: Any, ctx: Any) -> None:
    """Pairing routes. Paired nodes join the registry at the next restart."""
    from fastapi.responses import JSONResponse

    config, ADMIN = ctx.config, ctx.ADMIN
    book = PairingBook()
    app.state.cluster_pairing = book

    @app.post("/api/cluster/pairing")
    def cluster_pairing_code(body: dict | None = None, _a: dict = ADMIN):
        try:
            ttl = float((body or {}).get("ttl_s") or 600)
        except (TypeError, ValueError):
            return JSONResponse({"error": "ttl_s must be a number"}, status_code=400)
        ttl = min(3600.0, max(60.0, ttl))
        code, expires = book.issue(ttl)
        log_event(_cluster_log, 20, "cluster.pairing_issued", ttl_s=ttl,
                  user_id=(_a or {}).get("id"))
        return {"code": code, "expires_at": round(expires, 3), "ttl_s": ttl,
                "join": "deploy.py cluster join <this-url> --code <code> --url <node-url>"}

    @app.post("/api/cluster/join")
    def cluster_join(body: dict, request: _cluster_Request):
        now = time.time()
        ip = client_ip_for(request, config)
        retry = book.retry_in(ip, now)
        if retry:
            log_event(_cluster_log, 30, "cluster.join_throttled", client=ip, retry_in_s=retry)
            return JSONResponse({"error": f"too many failed join attempts; retry in {retry}s"},
                                status_code=429, headers={"Retry-After": str(retry)})
        if not isinstance(body, dict):
            return JSONResponse({"error": "expected a JSON object"}, status_code=400)
        name = _cluster_slug(str(body.get("name") or ""))
        url = str(body.get("url") or "").strip().rstrip("/")
        try:
            ram = float(body.get("ram_gb") or 0)
        except (TypeError, ValueError):
            ram = -1.0
        if not name:
            return JSONResponse({"error": "name is required (letters, digits, dashes)"},
                                status_code=400)
        if not 0 <= ram <= 8192:
            return JSONResponse({"error": "ram_gb must be a number of GB"}, status_code=400)
        problem = node_url_problem(config, url)
        if problem:
            return JSONResponse({"error": problem}, status_code=400)
        if not book.redeem(str(body.get("code") or "")):
            book.fail(ip, now)
            log_event(_cluster_log, 30, "cluster.join_refused", client=ip, node=name)
            return JSONResponse({"error": "invalid or expired pairing code"}, status_code=403)
        token = secrets.token_urlsafe(32)
        nodes = [n for n in read_paired_nodes(config)
                 if n.get("name") != name and str(n.get("url") or "").rstrip("/") != url]
        nodes.append({"name": name, "url": url, "ram_gb": ram, "token": token,
                      "joined_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))})
        try:
            write_paired_nodes(config, nodes)
        except OSError as exc:
            return JSONResponse({"error": redact_text(f"cannot save the node: {exc}")},
                                status_code=500)
        log_event(_cluster_log, 20, "cluster.node_paired", node=name,
                  url=node_safe_url(url), ram_gb=ram, client=ip)
        return {"node_token": token, "name": name,
                "primary": ctx.node_registry.local_node().name,
                "restart_required": True,
                "next": ("Add the token to NODE_TOKEN on the joining node (comma-separate "
                         "it from any existing token), restart it, then restart the primary.")}

    @app.get("/api/cluster/paired")
    def cluster_paired(_a: dict = ADMIN):
        live = {n.remote_url for n in ctx.node_registry.nodes}
        return {"nodes": [{"name": n.get("name"), "url": node_safe_url(str(n.get("url"))),
                           "ram_gb": n.get("ram_gb"), "joined_at": n.get("joined_at"),
                           "active": str(n.get("url") or "").rstrip("/") in live}
                          for n in read_paired_nodes(config)],
                "pending_codes": book.pending(),
                "cluster_nodes_env": bool(str(config.cluster_nodes or "").strip())}

    @app.delete("/api/cluster/paired/{name}")
    def cluster_unpair(name: str, _a: dict = ADMIN):
        nodes = read_paired_nodes(config)
        kept = [n for n in nodes if n.get("name") != name]
        if len(kept) == len(nodes):
            return JSONResponse({"error": "no such paired node"}, status_code=404)
        write_paired_nodes(config, kept)
        log_event(_cluster_log, 20, "cluster.node_unpaired", node=name)
        return {"removed": name, "restart_required": True}


__all__ = [
    "register_cluster",
    "PairingBook",
    "CLUSTER_REGISTRY_HOOKS",
    "cluster_node_specs",
    "cluster_paired_path",
    "read_paired_nodes",
    "write_paired_nodes",
    "remote_health_state",
    "remote_health_fields",
    "_cluster_slug",
    "_cluster_slug_from_url",
    "_cluster_code_hash",
    "_cluster_Request",
    "_CLUSTER_BASE_CAPS",
    "_CLUSTER_STUDIO_CAPS",
    "Node",
    "NodeRegistry",
    "Requirements",
    "RoutingDecision",
    "ClusterRouter",
    "HealthMonitor",
    "sample_local_load",
    "sample_load_ratio",
    "detect_machine",
    "friendly_node_name",
    "capabilities_for_machine",
    "HEALTHY", "DEGRADED", "OVERLOADED", "DRAINING", "STARTING", "UNAVAILABLE",
]
