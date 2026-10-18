"""Tests for layer-sharded serving (sharding.py) and `deploy.py cluster`.

Everything here is offline: there is one Mac in this environment, so the
planner, hostfile, launch command and backend wiring are checked against the
installed mlx / mlx-lm sources, not against a real multi-host run.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_llm.config as config_mod  # noqa: E402

config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-sharding-"))

from local_llm.config import Config  # noqa: E402
from local_llm.backends import get_backend, BACKENDS  # noqa: E402
from local_llm.cluster import NodeRegistry  # noqa: E402
from local_llm.sharding import (  # noqa: E402
    SHARD_BACKEND, cluster_cli, distributed_launch_command, mlx_even_split,
    mlx_rank_layers, model_shard_info, plan_shards, read_hostfile, write_hostfile,
    _shard_update_env_token)

GIB = 1024 ** 3


def _info(layers=32, layer_gb=1.0, shared_gb=2.0, kv_per_token=0):
    return {"model": "fake", "model_type": "llama", "num_layers": layers,
            "layer_bytes": [int(layer_gb * GIB)] * layers, "shared_bytes": int(shared_gb * GIB),
            "total_bytes": int(shared_gb * GIB) + int(layer_gb * GIB) * layers,
            "kv_bytes_per_token": kv_per_token, "has_index": True,
            "pipeline_supported": True}


def _cli(argv, config=None):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        code = cluster_cli(argv, config or Config())
    return code, out.getvalue()


def test_even_split_matches_mlx_pipeline_mixin():
    from mlx_lm.models.pipeline import PipelineMixin

    class _Group:
        def __init__(self, rank, size):
            self._r, self._s = rank, size

        def rank(self):
            return self._r

        def size(self):
            return self._s

    class _Model(PipelineMixin):
        def __init__(self, n):
            super().__init__()
            self.layers = list(range(n))

    for layers, size in ((36, 2), (37, 2), (28, 3), (80, 4)):
        split = mlx_even_split(layers, size)
        for rank in range(size):
            model = _Model(layers)
            model.pipeline(_Group(rank, size))
            assert mlx_rank_layers(split, rank) == (model.start_idx, model.end_idx)
    # Rank 0 (the HTTP server, this machine) holds the LAST layers.
    assert mlx_rank_layers(mlx_even_split(37, 2), 0) == (18, 37)


def test_hostfile_round_trips_through_mlx_parser():
    from mlx._distributed_utils.common import Hostfile
    path = Path(tempfile.mkdtemp()) / "hosts.json"
    hosts = [{"ssh": "127.0.0.1", "ips": ["192.168.1.10"], "ram_gb": 64},
             {"ssh": "studio.local", "ips": ["192.168.1.11"], "ram_gb": 128}]
    write_hostfile(path, hosts, backend="ring", envs=["MLX_METAL_FAST_SYNCH=1"])
    parsed = Hostfile.from_file(path)
    assert [h.ssh_hostname for h in parsed.hosts] == ["127.0.0.1", "studio.local"]
    assert parsed.hosts[1].ips == ["192.168.1.11"]
    assert parsed.backend == "ring" and parsed.envs == ["MLX_METAL_FAST_SYNCH=1"]
    assert read_hostfile(path)["hosts"][1]["ram_gb"] == 128
    try:
        write_hostfile(path, [{"ssh": "studio.local"}], backend="ring")
    except ValueError as exc:
        assert "IP" in str(exc)
    else:
        raise AssertionError("ring needs IPs")


def test_launch_command_shape():
    cmd = distributed_launch_command("/tmp/h.json", "mlx-community/X", 8090,
                                     python="/venv/bin/python", backend="ring")
    joined = " ".join(cmd)
    assert "--hostfile /tmp/h.json" in joined and "--backend ring" in joined
    assert "--python /venv/bin/python -- -m mlx_lm.server" in joined
    assert cmd[-1] == "--pipeline" and "--port 8090" in joined
    # The adapter is never on the command line (mlx-lm refuses it distributed).
    assert "--adapter-path" not in joined


def test_plan_fits_and_refuses():
    info = _info(layers=32, layer_gb=1.0, shared_gb=2.0)       # 34 GB model
    ok = plan_shards(info, [{"name": "mini", "free_ram_gb": 24},
                            {"name": "studio", "free_ram_gb": 24}],
                     kv_headroom_gb=1.0, reserve_gb=0.5)
    # 16 layers + 2 GB shared + 1 GB KV + 0.5 reserve = 19.5 GB per rank.
    assert ok["ok"] and ok["split"] == [16, 16]
    assert [r["need_gb"] for r in ok["ranks"]] == [19.5, 19.5]
    refused = plan_shards(info, [{"name": "mini", "free_ram_gb": 24},
                                 {"name": "studio", "free_ram_gb": 12}],
                          kv_headroom_gb=1.0)
    assert not refused["ok"] and "studio needs 19.5 GB" in refused["reason"]
    unknown = plan_shards(info, [{"name": "a", "free_ram_gb": 40}, {"name": "b"}],
                          kv_headroom_gb=1.0)
    assert not unknown["ok"] and "unknown" in unknown["reason"]
    # KV headroom derived from context: 32 layers x 1 MB/token over 8192 tokens.
    kv = plan_shards(_info(kv_per_token=32 * 1024 * 1024),
                     [{"name": "a", "free_ram_gb": 64}, {"name": "b", "free_ram_gb": 64}],
                     context_tokens=8192)
    assert kv["ranks"][0]["kv_gb"] == 128.0 and not kv["ok"]
    no_index = plan_shards(dict(info, has_index=False),
                           [{"name": "a", "free_ram_gb": 64}, {"name": "b", "free_ram_gb": 64}])
    assert not no_index["ok"] and "index" in no_index["reason"]


def test_model_shard_info_from_cached_checkpoint():
    try:
        info = model_shard_info("mlx-community/Qwen2.5-Coder-3B-Instruct-4bit")
    except FileNotFoundError:
        return      # not cached on this machine: nothing to read
    assert info["num_layers"] == 36 and len(info["layer_bytes"]) == 36
    assert all(b > 0 for b in info["layer_bytes"]) and info["shared_bytes"] > 0
    assert info["kv_bytes_per_token"] > 0


def test_backend_is_registered_without_adapter_support():
    spec = get_backend(SHARD_BACKEND)
    assert spec.name == SHARD_BACKEND and SHARD_BACKEND in BACKENDS
    assert spec.supports_adapter is False
    ok, reason = spec.available(Config(cluster_hostfile="/nonexistent/hosts.json"))
    assert not ok and reason


def test_registry_hook_makes_primary_a_shard_group():
    path = Path(tempfile.mkdtemp()) / "hosts.json"
    write_hostfile(path, [{"ssh": "127.0.0.1", "ips": ["10.0.0.1"], "ram_gb": 64},
                          {"ssh": "studio.local", "ips": ["10.0.0.2"], "ram_gb": 64}])
    plain = NodeRegistry(Config(cluster_hostfile=str(path))).local_node()
    assert plain.shard_group == []
    cfg = Config(model_backend=SHARD_BACKEND, cluster_hostfile=str(path))
    primary = NodeRegistry(cfg).local_node()
    assert primary.shard_group == ["127.0.0.1", "studio.local"]
    assert primary.backend == SHARD_BACKEND and "large_model" in primary.capabilities


def test_cluster_cli_plan_and_hostfile():
    try:
        model_shard_info("mlx-community/Qwen2.5-Coder-3B-Instruct-4bit")
        cached = True
    except FileNotFoundError:
        cached = False
    if cached:
        code, out = _cli(["plan", "--model", "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
                          "--host", "mini=6", "--host", "studio=6", "--json"])
        plan = json.loads(out)
        assert code == 0 and plan["ok"] and plan["split"] == [18, 18]
        code, out = _cli(["plan", "--model", "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
                          "--host", "mini=0.5", "--host", "studio=0.5"])
        assert code == 1 and "REFUSED" in out
    out_path = Path(tempfile.mkdtemp()) / "h.json"
    code, out = _cli(["hostfile", "--host", "127.0.0.1=10.0.0.1@64",
                      "--host", "studio.local=10.0.0.2@128", "--out", str(out_path)])
    assert code == 0 and "mlx_lm.server" in out and "--pipeline" in out
    assert read_hostfile(out_path)["hosts"][1] == {"ssh": "studio.local", "ips": ["10.0.0.2"],
                                                   "ram_gb": 128.0}
    assert _cli([])[0] == 2


def test_join_writes_token_into_env_file():
    env = Path(tempfile.mkdtemp()) / ".env"
    env.write_text("MODEL_ID=x\nNODE_TOKEN=old\n")
    _shard_update_env_token(env, "new")
    assert "NODE_TOKEN=new,old" in env.read_text()
    assert "already" in _shard_update_env_token(env, "new")
    fresh = Path(tempfile.mkdtemp()) / ".env"
    _shard_update_env_token(fresh, "tok")
    assert fresh.read_text() == "NODE_TOKEN=tok\n"
    assert (fresh.stat().st_mode & 0o777) == 0o600


def _run_standalone():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            import traceback; traceback.print_exc(); print(f"FAIL  {name}: {exc}"); failed += 1
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
