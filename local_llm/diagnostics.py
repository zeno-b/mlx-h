"""Doctor, benchmark, the hardware benchmark sweep and CSV export.

Split out of the original single-file deploy.py. The sweep (bench_matrix,
recommend, deploy.py bench --sweep / recommend / models and the /api/bench
routes) answers "which MODEL_ID, context and backend should this Mac run?"
with measurements instead of the RAM-tier guess in core.py; catalog.py holds
the metadata and the files it reads and writes.
"""

from __future__ import annotations

import csv
import json
import logging
import asyncio
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .core import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .database import *  # noqa: F401,F403
from .sysutil import *  # noqa: F401,F403
from .model_client import *  # noqa: F401,F403
from .model_server import *  # noqa: F401,F403
from .backends import *  # noqa: F401,F403
from .obslog import *  # noqa: F401,F403
from .catalog import *  # noqa: F401,F403


def export_to_csv(db: Database, path: Path) -> int:
    """Export all feedback to CSV."""
    rows = db.execute("SELECT * FROM feedback ORDER BY created_at DESC").fetchall()
    if not rows:
        return 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return len(rows)


def doctor(web_port: int, model_port: int) -> None:
    """Probe the loopback ports and report what is actually answering on each."""
    import urllib.error
    import urllib.request

    def probe(port: int, path: str) -> str:
        url = f"http://127.0.0.1:{port}{path}"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return f"{resp.status} {resp.headers.get('Content-Type')} {resp.read(90)!r}"
        except urllib.error.HTTPError as exc:
            return f"{exc.code} {exc.headers.get('Content-Type')} {exc.read(90)!r}"
        except Exception as exc:
            return f"no response ({exc})"

    print(f"python      : {sys.version.split()[0]} ({platform.machine()}, {platform.system()})")
    # Report what the user actually ran, not whichever module this line lives in.
    _entry = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else Path(__file__).resolve()
    print(f"entry point : {_entry}")
    print(f"code        : {Path(__file__).resolve().parent}")
    print(f"ui build    : {UI_BUILD}")
    print(f"in venv     : {in_venv()}  ({sys.prefix})")

    print(f"\nweb UI port {web_port}: {'LISTENING' if port_open(web_port) else 'CLOSED'}")
    if port_open(web_port):
        for path in ["/", "/api/health"]:
            print(f"  GET {path:14} -> {probe(web_port, path)}")

    print(f"\nmodel port {model_port}: {'LISTENING' if port_open(model_port) else 'CLOSED'}")
    if port_open(model_port):
        print(f"  GET {'/v1/models':14} -> {probe(model_port, '/v1/models')}")
        print(f"  GET {'/':14} -> {probe(model_port, '/')}")

    others = [p for p in range(8000, 8101) if p not in (web_port, model_port) and port_open(p)]
    print(f"\nother loopback listeners in 8000-8100: {others or 'none'}")
    print("\nA healthy web UI answers GET / with 200 text/html and")
    print("GET /api/health with 200 application/json. Anything else is the wrong port.")


def _bench_listeners(port: int) -> dict | None:
    """The model server listening on `port`, read from its own command line.

    {"pid", "model", "adapter"} or None. The command line is the one source
    that is right after both kinds of drift --bench used to miss: the app
    moving the model server off MODEL_PORT because the port was busy, and the
    UI switching models after boot. A listener whose argv does not carry
    `--port <port>` and `--model` (the web UI, anything unrelated) is not it.
    """
    lsof = shutil.which("lsof")
    if not lsof:
        return None
    try:
        found = subprocess.run([lsof, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                               capture_output=True, text=True, timeout=5)
    except Exception:  # noqa: BLE001
        return None
    for pid in (int(x) for x in found.stdout.split() if x.isdigit()):
        try:
            command = subprocess.run(["ps", "-o", "command=", "-p", str(pid)],
                                     capture_output=True, text=True, timeout=5).stdout
        except Exception:  # noqa: BLE001
            continue
        argv = command.split()

        def value(flag: str) -> str:
            return argv[argv.index(flag) + 1] if flag in argv[:-1] else ""

        if value("--port") == str(port) and value("--model"):
            return {"pid": pid, "model": value("--model"), "adapter": value("--adapter-path")}
    return None


def locate_model_server(preferred_port: int, fallback_model: str) -> dict:
    """Where the running app's model server is: {"port", "model", "adapter", "pid", "how"}.

    Scans the same 100-port window get_free_port() shifts into. Falls back to
    the old behaviour (the configured port and model) when no listener can be
    identified, e.g. without lsof; {} when nothing listens at all.
    """
    for port in range(preferred_port, preferred_port + 100):
        if not port_open(port):
            continue
        found = _bench_listeners(port)
        if found:
            return {"port": port, "how": "process", **found}
    if port_open(preferred_port):
        return {"port": preferred_port, "model": fallback_model, "adapter": "",
                "pid": None, "how": "configured"}
    return {}


def _bench_sizes(prompts: int, prompt_tokens: int) -> list[int]:
    """64 tokens, then prompt_tokens growing 4x per step: prompts=3 -> 64/512/2048."""
    count = min(6, max(1, int(prompts or 1)))
    return [64] + [int(prompt_tokens) * (4 ** i) for i in range(count - 1)]


def benchmark(
    config: Config,
    prompts: int = 3,
    prompt_tokens: int = 512,
    baseline_path: Path | None = None,
    save: bool = False,
    *,
    discover: bool = True,
    sizes: list[int] | None = None,
    collect: list | None = None,
    max_tokens: int = 64,
    out: Any = print,
) -> int:
    """Measure prefill and decode against a running model server.

    Reports time to first token separately from decode throughput, because on
    this hardware they respond to completely different fixes: TTFT is prompt
    processing and scales with how much context the agent re-sends, decode is
    memory bandwidth and scales with model size. A change that helps one often
    does nothing for the other.

    `prompts` is how many prompt sizes are measured (64 tokens, then
    prompt_tokens growing 4x each step); it used to be accepted and ignored.
    With discover=True (deploy.py --bench) the target is the model server the
    running app actually started, found by its command line, rather than
    MODEL_PORT and MODEL_ID: the app shifts the port when it is busy and the
    UI can switch models, and either way the old benchmark measured, or
    labelled, the wrong thing. bench_matrix passes discover=False because it
    started the server itself. `collect` receives every sample, failures
    included, for callers that keep results.
    """
    import dataclasses
    import urllib.request

    model_label = config.model
    port = config.model_port
    request_config = config
    if discover:
        where = locate_model_server(config.model_port, config.model)
        if not where:
            out(f"No model server on ports {config.model_port}-{config.model_port + 99}. "
                "Start the app first.")
            return 1
        port = int(where["port"])
        model_label = where.get("model") or config.model
        if where.get("adapter"):
            model_label = f"{model_label} + adapter {where['adapter']}"
        if port != config.model_port:
            out(f"(the model server is on port {port}, not MODEL_PORT {config.model_port}: "
                "the app moved it because the port was busy)")
        if where.get("model") and where["model"] != config.model:
            # Backends that send the repo id (not mlx-lm's "default_model") must
            # name the model that is loaded, or the request is a reload.
            request_config = dataclasses.replace(config, model=where["model"], model_port=port)
    elif not port_open(port):
        out(f"No model server on port {port}. Start the app first.")
        return 1

    sizes = [int(s) for s in (sizes or _bench_sizes(prompts, prompt_tokens)) if int(s) > 0]
    # Size the filler for the LARGEST prompt, or the biggest sample silently
    # runs short and the prefill curve looks flatter than it is.
    sentence = "The quick brown fox jumps over the lazy dog. "
    repeats = (max(sizes) * CHARS_PER_TOKEN) // len(sentence) + 2
    filler = sentence * repeats
    out(f"model : {model_label}")
    out(f"port  : {port}")
    out("")
    out(f"{'prompt tok':>11}  {'ttft ms':>8}  {'decode tok/s':>12}  {'total ms':>9}")

    failures = 0
    samples: list[dict] = []
    for size in sizes:
        text = filler[: size * CHARS_PER_TOKEN]
        body = json.dumps({
            # Same request name the app sends, so the benchmark measures the
            # adapter-carrying model rather than forcing a base-model reload.
            "model": request_model_for(request_config),
            "messages": [
                {"role": "user", "content": text + "\n\nReply with exactly one short sentence."}
            ],
            "max_tokens": int(max_tokens),
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json"}, method="POST",
        )
        started = time.time()
        first_at: float | None = None
        completion = 0
        prompt_reported = 0
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data: "):
                        continue
                    chunk = line[6:]
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                    except Exception:
                        continue
                    if data.get("usage"):
                        prompt_reported = int(data["usage"].get("prompt_tokens") or 0)
                        completion = int(data["usage"].get("completion_tokens") or completion)
                    choices = data.get("choices") or []
                    if choices and (choices[0].get("delta") or {}).get("content"):
                        if first_at is None:
                            first_at = time.time()
                        completion += 1
        except Exception as exc:
            out(f"{size:>11}  request failed: {exc}")
            failures += 1
            if collect is not None:
                collect.append({"requested": size, "error": redact_text(str(exc))})
            continue

        total_ms = (time.time() - started) * 1000
        ttft_ms = ((first_at - started) * 1000) if first_at else total_ms
        decode_ms = max(1.0, total_ms - ttft_ms)
        tps = completion / (decode_ms / 1000.0)
        prompt_count = prompt_reported or size
        sample = {
            "requested": size,
            "prompt_tokens": prompt_count,
            "completion_tokens": completion,
            "ttft_ms": round(ttft_ms, 1),
            "tps": round(tps, 2),
            "prompt_tps": round(prompt_count / max(0.001, ttft_ms / 1000.0), 1),
            "total_ms": round(total_ms, 1),
        }
        samples.append(sample)
        if collect is not None:
            collect.append(sample)
        out(f"{prompt_count:>11}  {ttft_ms:>8.0f}  {tps:>12.1f}  {total_ms:>9.0f}")

    out("")
    if baseline_path is not None:
        _report_against_baseline(baseline_path, model_label, samples, save)

    if discover:
        out("TTFT rising steeply with prompt size is re-prefill cost. That is what")
        out("prompt caching, smaller tool results and a tighter tool catalogue attack.")
        out("Flat TTFT with low tok/s is memory bandwidth: use a smaller model.")
    return 1 if failures == len(sizes) else 0


def _report_against_baseline(path: Path, model: str, samples: list[dict], save: bool) -> None:
    """Compare this run to a saved one. Optimizing without this is guessing."""
    previous: dict = {}
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log(f"Could not read the benchmark baseline: {exc}", logging.WARNING)

    prior = previous.get("samples") if previous.get("model") == model else None
    if prior:
        print(f"vs baseline from {previous.get('recorded_at', 'unknown')}:")
        print(f"{'prompt tok':>11}  {'ttft delta':>12}  {'tok/s delta':>12}")
        by_size = {sample["requested"]: sample for sample in prior}
        for sample in samples:
            was = by_size.get(sample["requested"])
            if not was:
                continue
            ttft_delta = sample["ttft_ms"] - was["ttft_ms"]
            tps_delta = sample["tps"] - was["tps"]
            print(f"{sample['prompt_tokens']:>11}  {ttft_delta:>+11.0f}ms  {tps_delta:>+12.1f}")
        print()
    elif previous:
        print(f"(baseline is for {previous.get('model')}, not comparing)\n")

    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "model": model,
            "recorded_at": iso(utc_now()),
            "samples": samples,
        }, indent=2), encoding="utf-8")
        print(f"Baseline saved to {path}. Re-run --bench after a change to compare.")


# --------------------------------------------------------------------------- #
# Hardware benchmark sweep (item 9). One model server at a time on a free
# port, measured with benchmark(), its peak memory sampled from outside, the
# whole sweep written to DATA_DIR/bench/<timestamp>.json, and recommend()
# picking the best model, context and backend this machine actually runs.
# --------------------------------------------------------------------------- #

def _bench_tree_rss_mb(pid: int) -> float | None:
    """Resident memory of pid plus its descendants, in MB (one ps call).

    Descendants count because a BACKEND_COMMAND wrapper (sh -c, a launcher
    script) puts the real engine one level down. On Apple Silicon the Metal
    buffers live in unified memory and show up in RSS, so this approximates
    what the model costs the machine; it is not the GPU wired figure.
    """
    try:
        listing = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="],
                                 capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return None
    children: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for row in listing.splitlines():
        parts = row.split()
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            continue
        child, parent, kb = (int(p) for p in parts)
        rss[child] = kb
        children.setdefault(parent, []).append(child)
    if pid not in rss:
        return None
    total, stack, seen = 0, [pid], set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        total += rss.get(current, 0)
        stack.extend(children.get(current, []))
    return round(total / 1024.0, 1)


class _BenchPeakSampler:
    """Polls a process tree's RSS on a thread and keeps the maximum."""

    def __init__(self, pid: int | None, sampler: Any = None, interval: float = 0.5):
        self.pid = pid
        self.sampler = sampler or _bench_tree_rss_mb
        self.interval = interval
        self.peak_mb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self) -> None:
        if not self.pid:
            return
        try:
            value = self.sampler(self.pid)
        except Exception:  # noqa: BLE001
            value = None
        if value is not None and (self.peak_mb is None or value > self.peak_mb):
            self.peak_mb = float(value)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()

    def __enter__(self) -> "_BenchPeakSampler":
        self.sample()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="bench-rss")
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.sample()


def _bench_default_manager(model_id: str, port: int, context: int, backend: str,
                           config: Config) -> Any:
    """A ModelServerManager for one sweep cell: base weights, no adapter, no watchdog.

    The adapter is left out because adapters belong to one base model and the
    sweep compares bases; the watchdog interval is pushed out so its probe
    generations never land inside a measurement.
    """
    return ModelServerManager(
        model_id, port, ADAPTER_DIR, int(context), "none",
        kv_bits=config.kv_bits, kv_group_size=config.kv_group_size,
        quantized_kv_start=config.quantized_kv_start,
        prefill_step_size=config.prefill_step_size,
        prompt_cache_size=config.prompt_cache_size,
        prompt_cache_bytes=config.prompt_cache_bytes,
        decode_concurrency=config.decode_concurrency,
        prompt_concurrency=config.prompt_concurrency,
        probe_interval=86400.0,
        backend=backend,
        draft_model=config.draft_model,
        num_draft_tokens=config.num_draft_tokens,
        backend_command=config.backend_command,
        backend_capabilities=config.backend_capabilities,
        backend_extra_args=config.backend_extra_args,
        max_concurrent_generations=config.max_concurrent_generations,
        context_size=int(context),
    )


def _bench_default_candidates(config: Config) -> list[str]:
    """Cached catalogue models this machine can hold, current model first."""
    picked = []
    for entry in model_catalog(config):
        if entry["current"] or (entry["cached"] and entry.get("fits_in_ram") is not False):
            picked.append(entry["id"])
    return picked


def _bench_median(values: list[float]) -> float | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def _bench_skip_reason(model_id: str, context: int, backend: str, config: Config,
                       allow_download: bool) -> str:
    meta = model_metadata(model_id)
    if fits_in_ram(meta) is False:
        return (f"needs about {meta['min_ram_gb']}GB RAM; this machine has "
                f"{TOTAL_RAM_GB:.0f}GB")
    if meta and meta.get("context_max") and int(context) > int(meta["context_max"]):
        return f"context {context} exceeds the model's {meta['context_max']}"
    if not allow_download and not model_is_cached(model_id):
        return "not downloaded (run the sweep with --download to fetch it)"
    if backend not in backend_names():
        return f"unknown backend {backend!r}"
    row = next((r for r in public_backends(config) if r["name"] == backend), None)
    if row is not None and not row["available"]:
        return f"backend {backend} unavailable: {row['reason']}"
    return ""


def bench_matrix(config: Config, candidates: list[str] | None = None,
                 contexts: list[int] | None = None, backends: list[str] | None = None,
                 *, data_dir: Path | None = None, manager_factory: Any = None,
                 port_finder: Any = None, rss_sampler: Any = None,
                 allow_download: bool = False, progress: Any = None,
                 cancel: threading.Event | None = None, out: Any = print) -> dict:
    """Benchmark every (model, context, backend) cell and save the sweep.

    Strictly one model server at a time, each on a port from get_free_port and
    stopped before the next starts: two models resident at once is exactly the
    memory spike the sweep exists to avoid recommending. Cells that cannot run
    (does not fit, not downloaded, backend missing) are recorded as skipped
    with the reason rather than attempted. Returns the saved document, with
    "file" naming it under DATA_DIR/bench/.
    """
    import dataclasses

    factory = manager_factory or _bench_default_manager
    find_port = port_finder or get_free_port
    names = [n for n in (candidates or _bench_default_candidates(config)) if str(n).strip()]
    models = []
    for name in names:
        resolved = resolve_model(name, config)
        if resolved not in models:
            models.append(resolved)
    ctx_list = [int(c) for c in (contexts or [config.context_size])]
    engine_list = [str(b).strip().lower() for b in (backends or [config.model_backend or DEFAULT_BACKEND])]
    cells = [(m, c, b) for m in models for c in ctx_list for b in engine_list]
    results: list[dict] = []
    started_at = iso(utc_now())
    for index, (model_id, context, backend) in enumerate(cells, 1):
        if cancel is not None and cancel.is_set():
            break
        label = f"[{index}/{len(cells)}] {model_id} ctx={context} backend={backend}"
        if progress is not None:
            progress(index - 1, len(cells), label)
        row: dict = {"model_id": model_id, "alias": alias_of(model_id, config),
                     "context_size": context, "backend": backend,
                     "kv_bits": int(config.kv_bits or 0), "success": False}
        skip = _bench_skip_reason(model_id, context, backend, config, allow_download)
        if skip:
            row.update(skipped=True, error=skip)
            out(f"{label}: skipped ({skip})")
            results.append(row)
            continue
        out(f"{label}: loading")
        port = find_port(config.model_port)
        manager = factory(model_id, port, context, backend, config)
        samples: list[dict] = []
        load_started = time.time()
        try:
            manager.start()
            row["load_s"] = round(time.time() - load_started, 1)
            pid = getattr(getattr(manager, "proc", None), "pid", None)
            cell_config = dataclasses.replace(
                config, model=model_id, model_port=int(getattr(manager, "model_port", port)),
                model_backend=backend, context_size=context)
            # Prompts sized to the window: a tiny prompt (decode speed), a
            # quarter window and most of it (prefill speed and peak memory).
            sizes = sorted({64, max(128, context // 4), max(256, int(context * 0.75) - 128)})
            with _BenchPeakSampler(pid, rss_sampler) as peak:
                benchmark(cell_config, baseline_path=None, discover=False, sizes=sizes,
                          collect=samples, out=lambda *_a, **_k: None)
            good = [s for s in samples if "error" not in s]
            row["samples"] = samples
            row["peak_rss_mb"] = peak.peak_mb
            if good:
                row.update(
                    success=True,
                    ttft_ms=good[0]["ttft_ms"],
                    decode_tps=_bench_median([s["tps"] for s in good]),
                    prompt_tps=good[-1]["prompt_tps"],
                )
            else:
                row["error"] = "; ".join(s.get("error", "") for s in samples)[:500] or "no samples"
        except Exception as exc:  # noqa: BLE001
            row["error"] = redact_text(f"{type(exc).__name__}: {exc}")[:500]
        finally:
            try:
                manager.stop()
            except Exception as exc:  # noqa: BLE001
                log(f"bench: stopping {model_id} failed: {exc}", logging.WARNING)
        if row["success"]:
            out(f"{label}: decode {row['decode_tps']:.1f} tok/s, prompt "
                f"{row['prompt_tps']:.0f} tok/s, ttft {row['ttft_ms']:.0f} ms, peak "
                + (f"{row['peak_rss_mb']:.0f} MB" if row.get("peak_rss_mb") else "unknown"))
        else:
            out(f"{label}: failed ({row.get('error')})")
        results.append(row)
    if progress is not None:
        progress(len(cells), len(cells), "done")
    doc = {
        "recorded_at": iso(utc_now()),
        "started_at": started_at,
        "ram_gb": round(TOTAL_RAM_GB, 2),
        "machine": f"{platform.machine()} {platform.system()} {platform.mac_ver()[0]}".strip(),
        "results": results,
        "recommendation": recommend(results, TOTAL_RAM_GB,
                                    min_decode_tps=config.bench_min_decode_tps,
                                    headroom_gb=config.bench_headroom_gb),
    }
    path = save_bench_run(doc, data_dir)
    doc["file"] = path.name
    log_event(get_logger("bench"), logging.INFO, "bench.sweep",
              cells=len(cells), succeeded=sum(1 for r in results if r["success"]),
              file=path.name)
    return doc


def _bench_memory_mb(row: dict) -> float | None:
    """Measured peak, else the catalogue's weight size plus runtime overhead."""
    if row.get("peak_rss_mb"):
        return float(row["peak_rss_mb"])
    meta = model_metadata(row.get("model_id", ""))
    if meta and meta.get("approx_size_gb"):
        return (float(meta["approx_size_gb"]) + 1.0) * 1024.0
    return None


def recommend(results: list[dict], ram_gb: float, *, min_decode_tps: float = 10.0,
              headroom_gb: float = 0.0) -> dict:
    """The best (model, context, backend) from sweep results for this machine.

    Highest quality first (bigger, then less quantized: catalog.quality_key)
    among the runs that succeeded, fit in RAM minus headroom at their measured
    peak, and decoded at least min_decode_tps; for that model the largest
    context that passed, then the fastest backend. A machine where nothing is
    fast enough gets the fastest run that fits, with a reason that says so.
    """
    ram = float(ram_gb)
    headroom = float(headroom_gb) if headroom_gb and float(headroom_gb) > 0 else max(2.5, ram * 0.25)
    budget_mb = max(0.0, ram - headroom) * 1024.0
    empty = {"model_id": None, "alias": "", "context_size": None, "kv_bits": None,
             "backend": None, "ram_gb": round(ram, 2), "headroom_gb": round(headroom, 2),
             "min_decode_tps": float(min_decode_tps)}
    ok = [r for r in results or [] if r.get("success")]
    if not ok:
        return {**empty, "reason": "no benchmark run succeeded"}
    fitting = [r for r in ok if (_bench_memory_mb(r) or 0) and _bench_memory_mb(r) <= budget_mb]
    if not fitting:
        return {**empty, "reason": (f"nothing fit in {budget_mb / 1024:.1f}GB "
                                    f"({ram:.0f}GB RAM minus {headroom:.1f}GB headroom)")}
    fast = [r for r in fitting if float(r.get("decode_tps") or 0) >= float(min_decode_tps)]
    if fast:
        pick = max(fast, key=lambda r: (quality_key(r["model_id"]), int(r["context_size"]),
                                        float(r.get("decode_tps") or 0)))
        reason = (f"highest-quality model that fit in {budget_mb / 1024:.1f}GB "
                  f"(peak {(_bench_memory_mb(pick) or 0) / 1024:.1f}GB) and decoded at "
                  f"{float(pick['decode_tps']):.1f} tok/s (minimum {float(min_decode_tps):g})")
    else:
        pick = max(fitting, key=lambda r: float(r.get("decode_tps") or 0))
        reason = (f"no run reached {float(min_decode_tps):g} tok/s; this is the fastest that "
                  f"fit ({float(pick.get('decode_tps') or 0):.1f} tok/s)")
    return {
        **empty,
        "model_id": pick["model_id"],
        "alias": pick.get("alias") or alias_of(pick["model_id"]),
        "context_size": int(pick["context_size"]),
        "kv_bits": int(pick.get("kv_bits") or 0),
        "backend": pick.get("backend") or DEFAULT_BACKEND,
        "decode_tps": pick.get("decode_tps"),
        "prompt_tps": pick.get("prompt_tps"),
        "ttft_ms": pick.get("ttft_ms"),
        "peak_rss_mb": pick.get("peak_rss_mb"),
        "reason": reason,
    }


# --------------------------------------------------------------------------- #
# CLI verbs: deploy.py bench / recommend / models (cli.py subcommands).
# --------------------------------------------------------------------------- #

def _bench_csv(text: str | None, cast: Any = str) -> list:
    return [cast(part.strip()) for part in str(text or "").split(",") if part.strip()]


def _bench_cli_config() -> Config:
    """Environment defaults plus what Settings saved (the bench thresholds, aliases)."""
    cfg = Config()
    cfg.load_saved()
    return cfg


def _bench_print_recommendation(rec: dict) -> None:
    if not rec.get("model_id"):
        print(f"No recommendation: {rec.get('reason')}")
        return
    alias = f" ({rec['alias']})" if rec.get("alias") else ""
    print(f"Recommended: {rec['model_id']}{alias}")
    print(f"  context {rec['context_size']}, backend {rec['backend']}, kv_bits {rec['kv_bits']}")
    print(f"  why: {rec['reason']}")


def _bench_apply(rec: dict) -> int:
    if not rec.get("model_id"):
        print("Nothing applied.")
        return 1
    path = write_recommendation(rec)
    print(f"Wrote {path}. It becomes the default MODEL_ID / CONTEXT_SIZE on the next "
          "start, unless those are set in the environment.")
    extras = []
    if rec.get("backend") and rec["backend"] != DEFAULT_BACKEND:
        extras.append(f"MODEL_BACKEND={rec['backend']}")
    if rec.get("kv_bits"):
        extras.append(f"KV_BITS={rec['kv_bits']}")
    if extras:
        print("Not applied automatically (set in .env if wanted): " + " ".join(extras))
    return 0


def bench_cli(argv: list[str]) -> int:
    """deploy.py bench [--sweep [--models a,b] [--contexts 4096,8192] [--backends ...] [--apply]]."""
    import argparse
    parser = argparse.ArgumentParser(
        prog="deploy.py bench",
        description="Benchmark the running model server, or with --sweep load each "
                    "candidate in turn and recommend the best for this Mac.")
    parser.add_argument("--sweep", action="store_true",
                        help="Start each model/context/backend in turn (stop the app first).")
    parser.add_argument("--models", default="", help="Comma list of repo ids or aliases.")
    parser.add_argument("--contexts", default="", help="Comma list of context sizes.")
    parser.add_argument("--backends", default="", help="Comma list of backend names.")
    parser.add_argument("--apply", action="store_true",
                        help="Write the recommendation as this machine's default.")
    parser.add_argument("--download", action="store_true",
                        help="Allow the sweep to download models that are not cached.")
    parser.add_argument("--save", action="store_true",
                        help="Without --sweep: save the result as the comparison baseline.")
    parser.add_argument("--model-port", type=int, default=None)
    args = parser.parse_args(argv)
    ensure_dirs()
    cfg = _bench_cli_config()
    if args.model_port:
        cfg.model_port = args.model_port
    if not args.sweep:
        return benchmark(cfg, baseline_path=DATA_DIR / "bench_baseline.json", save=args.save)
    running = locate_model_server(cfg.model_port, cfg.model)
    if running:
        print(f"A model server is running on port {running['port']} "
              f"({running.get('model') or 'unknown model'}). Stop the app first: the sweep "
              "loads each model in turn and needs the memory. From the web UI, use "
              "POST /api/bench/run instead.")
        return 1
    try:
        contexts = _bench_csv(args.contexts, int)
    except ValueError:
        print("--contexts must be a comma list of integers.")
        return 2
    doc = bench_matrix(cfg, _bench_csv(args.models) or None, contexts or None,
                       _bench_csv(args.backends) or None, allow_download=args.download)
    print(f"\nSaved {bench_dir() / doc['file']}")
    _bench_print_recommendation(doc["recommendation"])
    return _bench_apply(doc["recommendation"]) if args.apply else 0


def recommend_cli(argv: list[str]) -> int:
    """deploy.py recommend [--apply] [--file NAME]: from the latest saved sweep."""
    import argparse
    parser = argparse.ArgumentParser(prog="deploy.py recommend",
                                     description="Recommend a model from saved sweeps.")
    parser.add_argument("--apply", action="store_true",
                        help="Write it as this machine's default MODEL_ID / CONTEXT_SIZE.")
    parser.add_argument("--file", default="", help="A sweep file under DATA_DIR/bench.")
    args = parser.parse_args(argv)
    doc = load_bench_run(bench_dir() / args.file) if args.file else latest_bench_run()
    if not doc:
        print("No saved sweep. Run: deploy.py bench --sweep")
        return 1
    cfg = _bench_cli_config()
    rec = recommend(doc.get("results") or [], TOTAL_RAM_GB,
                    min_decode_tps=cfg.bench_min_decode_tps, headroom_gb=cfg.bench_headroom_gb)
    _bench_print_recommendation(rec)
    applied = read_recommendation(ram_gb=TOTAL_RAM_GB)
    if applied:
        print(f"Currently applied: {applied['model_id']} (context {applied.get('context_size')})")
    return _bench_apply(rec) if args.apply else 0


def models_cli(argv: list[str]) -> int:
    """deploy.py models [--json]: the catalogue with aliases, RAM fit and cache state."""
    import argparse
    parser = argparse.ArgumentParser(prog="deploy.py models",
                                     description="List catalogue models, aliases and fit.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    cfg = _bench_cli_config()
    rows = model_catalog(cfg)
    aliases = model_aliases(cfg)
    if args.json:
        print(json.dumps({"ram_gb": round(TOTAL_RAM_GB, 1), "models": rows,
                          "aliases": aliases}, indent=2, default=str))
        return 0
    print(f"RAM: {TOTAL_RAM_GB:.0f}GB   (* current, R recommended, ~ approximate figures)")
    print(f"{'':2} {'model':58} {'size':>6} {'min RAM':>7} {'fits':>5}  {'cache':8} aliases")
    for row in rows:
        meta = row.get("meta") or {}
        mark = ("*" if row["current"] else " ") + ("R" if row["recommended"] else " ")
        size = f"{meta['approx_size_gb']:.1f}G" if meta.get("approx_size_gb") else "?"
        floor = f"{meta['min_ram_gb']}G" if meta.get("min_ram_gb") else "?"
        fits = {True: "yes", False: "no", None: "?"}[row.get("fits_in_ram")]
        approx = "~" if meta.get("approximate") else " "
        print(f"{mark} {row['id'][:58]:58} {size:>6} {floor:>7} {fits:>5}{approx} "
              f"{'cached' if row['cached'] else '-':8} {', '.join(row['aliases'])}")
    print("\naliases:")
    for name, info in sorted(aliases.items()):
        print(f"  {name:12} -> {info['id']}  ({info['source']})")
    return 0


# --------------------------------------------------------------------------- #
# Routes: /api/models/aliases and /api/bench/* (register_catalog).
# --------------------------------------------------------------------------- #

_BENCH_FILE_RE = re.compile(r"^[0-9A-Za-z_-]{1,64}\.json$")


class _BenchJob:
    """State of the one sweep a server may run at a time (shared across requests)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.state: dict = {"running": False, "started_at": None, "finished_at": None,
                            "done": 0, "total": 0, "current": "", "error": "",
                            "file": "", "recommendation": None, "applied": False,
                            "log": []}

    def progress(self, done: int, total: int, current: str) -> None:
        with self.lock:
            self.state.update(done=done, total=total, current=current)

    def line(self, text: str = "", *_a: Any, **_k: Any) -> None:
        with self.lock:
            self.state["log"] = (self.state["log"] + [str(text)])[-50:]

    def snapshot(self) -> dict:
        with self.lock:
            return {**self.state, "running": self.running, "log": list(self.state["log"])}


def _bench_busy(ctx: Any) -> str:
    """Why a sweep must not start now, or "" when it may."""
    if (getattr(ctx.retrain_manager, "status", None) or {}).get("running"):
        return "A training run is in progress and owns the model server."
    try:
        queue = ctx.generation_gate.snapshot()
    except Exception:  # noqa: BLE001
        queue = {}
    if int(queue.get("running") or 0) or int(queue.get("waiting") or 0):
        return (f"{int(queue.get('running') or 0)} generation(s) running and "
                f"{int(queue.get('waiting') or 0)} waiting. Let them finish first.")
    runs = list(getattr(ctx.tasks, "by_task", {}) or {})
    if runs:
        return f"{len(runs)} task run(s) in flight. Cancel them first."
    return ""


def _bench_job_body(job: _BenchJob, ctx: Any, opts: dict) -> None:
    manager = ctx.model_manager
    was_serving = bool(getattr(manager, "is_alive", lambda: False)())
    try:
        if was_serving:
            job.line("Stopping the serving model to free its memory.")
            manager.stop()
        manager.status = "benchmarking"
        doc = bench_matrix(
            ctx.config, opts["models"] or None, opts["contexts"] or None,
            opts["backends"] or None, allow_download=opts["download"],
            manager_factory=ctx.extras.get("bench_manager_factory"),
            port_finder=ctx.extras.get("bench_port_finder"),
            rss_sampler=ctx.extras.get("bench_rss_sampler"),
            progress=job.progress, out=job.line)
        applied = False
        if opts["apply"] and (doc.get("recommendation") or {}).get("model_id"):
            write_recommendation(doc["recommendation"])
            applied = True
        with job.lock:
            job.state.update(file=doc.get("file", ""), recommendation=doc.get("recommendation"),
                             applied=applied)
    except Exception as exc:  # noqa: BLE001
        log(f"Benchmark sweep failed: {exc}", logging.WARNING)
        with job.lock:
            job.state["error"] = redact_text(f"{type(exc).__name__}: {exc}")
    finally:
        manager.status = "stopped"
        if was_serving:
            job.line("Restarting the serving model.")
            try:
                manager.start()
            except Exception as exc:  # noqa: BLE001
                log(f"Could not restart the model server after the sweep: {exc}",
                    logging.ERROR)
                with job.lock:
                    job.state["error"] = (job.state["error"] + " " if job.state["error"] else "") \
                        + redact_text(f"restart failed: {exc}")
        with job.lock:
            job.running = False
            job.state.update(running=False, finished_at=iso(utc_now()))


def _bench_list_arg(value: Any, cast: Any, field: str) -> list:
    if value in (None, "", []):
        return []
    items = value if isinstance(value, list) else str(value).split(",")
    out = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        if len(text) > 200:
            raise ValueError(f"{field}: entry too long")
        out.append(cast(text))
    if len(out) > 32:
        raise ValueError(f"{field}: at most 32 entries")
    return out


def register_catalog(app: Any, ctx: Any) -> None:
    """GET /api/models/aliases, GET /api/bench/{results,recommendation,status}, POST /api/bench/run."""
    from fastapi.responses import JSONResponse

    config = ctx.config
    job = _BenchJob()
    # select_model reads this to refuse a model switch while a sweep owns
    # the model server.
    app.state.bench_job = job
    ctx.extras["bench_job"] = job

    @app.get("/api/models/aliases")
    def catalog_aliases(_u: dict = ctx.USER):
        return {"aliases": model_aliases(config), "current": config.model,
                "model_alias": config.model_alias}

    @app.get("/api/bench/results")
    async def bench_results(file: str = "", _a: dict = ctx.ADMIN):
        def load() -> dict | JSONResponse:
            runs = list_bench_runs()
            if file:
                if not _BENCH_FILE_RE.match(file):
                    return JSONResponse({"error": "invalid file name"}, status_code=400)
                doc = load_bench_run(bench_dir() / file) if (bench_dir() / file).is_file() else None
                if doc is None:
                    return JSONResponse({"error": "no such benchmark file"}, status_code=404)
                return {"file": file, "run": doc}
            summary = []
            for path in runs[:50]:
                doc = load_bench_run(path) or {}
                rows = doc.get("results") or []
                summary.append({"file": path.name, "recorded_at": doc.get("recorded_at"),
                                "ram_gb": doc.get("ram_gb"), "cells": len(rows),
                                "succeeded": sum(1 for r in rows if r.get("success")),
                                "recommendation": doc.get("recommendation")})
            latest = load_bench_run(runs[0]) if runs else None
            return {"runs": summary, "latest": latest}
        return await asyncio.to_thread(load)

    @app.get("/api/bench/recommendation")
    async def bench_recommendation(_a: dict = ctx.ADMIN):
        def load() -> dict:
            latest = latest_bench_run()
            computed = None
            if latest:
                computed = recommend(latest.get("results") or [], TOTAL_RAM_GB,
                                     min_decode_tps=config.bench_min_decode_tps,
                                     headroom_gb=config.bench_headroom_gb)
            return {"applied": read_recommendation(ram_gb=TOTAL_RAM_GB),
                    "computed": computed, "from_file": (latest or {}).get("file"),
                    "ram_gb": round(TOTAL_RAM_GB, 2)}
        return await asyncio.to_thread(load)

    @app.get("/api/bench/status")
    def bench_status(_a: dict = ctx.ADMIN):
        return job.snapshot()

    @app.post("/api/bench/run")
    def bench_run(body: dict, _a: dict = ctx.ADMIN):
        if body.get("confirm") is not True:
            return JSONResponse({"error": "The sweep stops the serving model for its whole "
                                          "duration. Send {\"confirm\": true} to proceed."},
                                status_code=400)
        try:
            opts = {
                "models": _bench_list_arg(body.get("models"), str, "models"),
                "contexts": _bench_list_arg(body.get("contexts"), int, "contexts"),
                "backends": _bench_list_arg(body.get("backends"), str, "backends"),
                "apply": body.get("apply") is True,
                "download": body.get("download") is True,
            }
        except (TypeError, ValueError) as exc:
            return JSONResponse({"error": f"invalid request: {exc}"}, status_code=400)
        if any(not 512 <= c <= 131072 for c in opts["contexts"]):
            return JSONResponse({"error": "contexts must be between 512 and 131072"},
                                status_code=400)
        busy = _bench_busy(ctx)
        if busy:
            return JSONResponse({"error": busy}, status_code=409)
        with job.lock:
            if job.running:
                return JSONResponse({"error": "A benchmark sweep is already running."},
                                    status_code=409)
            job.running = True
            job.state.update(running=True, started_at=iso(utc_now()), finished_at=None,
                             done=0, total=0, current="", error="", file="",
                             recommendation=None, applied=False, log=[])
        log_event(get_logger("bench"), logging.INFO, "bench.start",
                  models=len(opts["models"]), contexts=opts["contexts"],
                  backends=opts["backends"], apply=opts["apply"])
        threading.Thread(target=_bench_job_body, args=(job, ctx, opts),
                         daemon=True, name="bench-sweep").start()
        return JSONResponse({"started": True, "status": job.snapshot()}, status_code=202)


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    '_BenchJob',
    '_BenchPeakSampler',
    '_bench_busy',
    '_bench_default_manager',
    '_bench_listeners',
    '_bench_memory_mb',
    '_bench_sizes',
    '_bench_tree_rss_mb',
    '_report_against_baseline',
    'bench_cli',
    'bench_matrix',
    'benchmark',
    'doctor',
    'export_to_csv',
    'locate_model_server',
    'models_cli',
    'recommend',
    'recommend_cli',
    'register_catalog',
]
