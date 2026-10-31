"""Tests for the admin dashboards (dashboards.py), the event sink (obslog.py)
and the adapter ELO evaluation (elo.py)."""
from __future__ import annotations

import logging
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_llm.config as config_mod  # noqa: E402

config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-dashboards-"))

from fastapi.testclient import TestClient  # noqa: E402

import local_llm.elo as elo_mod  # noqa: E402
from local_llm.api import create_app  # noqa: E402
from local_llm.config import Config  # noqa: E402
from local_llm.dashboards import (  # noqa: E402
    DASH_SCHEMA_NAME, _dash_epoch_sql, dash_bind_runtime, dash_reset_active_cache,
    dashboard_summary, dashboard_timeseries)
from local_llm.database import Database  # noqa: E402
from local_llm.elo import (  # noqa: E402
    ELO_INITIAL, ELO_SCHEMA_NAME, elo_create_pair, elo_expected, elo_parse_verdict,
    elo_ratings, elo_recompute, elo_record_match, elo_sync_implicit, elo_update)
from local_llm.llm import GenerationStats  # noqa: E402
from local_llm.model_server import ModelServerManager  # noqa: E402
from local_llm.obslog import EVENT_SINK, EventSink, log_event  # noqa: E402
from local_llm.core import ADAPTER_DIR  # noqa: E402
from local_llm.tools import ToolRegistry  # noqa: E402
from local_llm.training import RetrainManager  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def build_app(**overrides):
    """A fresh app + temp DB (copied from tests/test_app.py:build_app)."""
    base = dict(auth_enabled=True, admin_username="admin", admin_password="adminpw123",
                allow_test_user=False, node_token="secret-node-token",
                skills_dir=tempfile.mkdtemp(prefix="local-llm-test-skills-"))
    base.update(overrides)
    cfg = Config(**base)
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    mm = ModelServerManager(cfg.model, 8090, ADAPTER_DIR)  # never started
    rm = RetrainManager(db, mm, cfg)
    app = create_app(cfg, db, mm, rm, ToolRegistry(cfg, db))
    app.state.model_manager = mm
    return app, cfg, db


def login(client, username, password):
    return client.post("/api/auth/login", json={"username": username, "password": password})


def admin_and_user(c):
    assert login(c, "admin", "adminpw123").status_code == 200
    r = c.post("/api/users", json={"username": "bob", "password": "bobpass123", "role": "user"})
    assert r.status_code in (200, 201), r.text
    # Not entered: the admin's client already ran the app's lifespan.
    user = TestClient(c.app)
    assert login(user, "bob", "bobpass123").status_code == 200
    return user


def fresh_db() -> Database:
    db = Database(Path(tempfile.mkdtemp()) / "dash.db")
    db.ensure_schema(DASH_SCHEMA_NAME)
    db.ensure_schema(ELO_SCHEMA_NAME)
    return db


def insert_metric(db, ts, endpoint="chat_stream", duration=100.0, status=200, error=None,
                  prompt=None, completion=None, ttft=None, tps=None, model="m",
                  user="u1", adapter=None, backend=None, cached=None, queue=None, conv="c1"):
    db.execute(
        "INSERT INTO metrics (timestamp, endpoint, duration_ms, status_code, error, "
        "prompt_tokens, completion_tokens, ttft_ms, decode_tps, model, conversation_id, "
        "user_id, adapter_id, backend, cached_prompt_tokens, queue_ms) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (ts, endpoint, duration, status, error, prompt, completion, ttft, tps, model, conv,
         user, adapter, backend, cached, queue))
    db.commit()


def sql_ts(epoch: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(epoch))


class _FakeManager:
    """Just enough of ModelServerManager for active_adapter()."""
    model_id = "org/base-model"
    adapter_choice = "none"
    status = "ready"
    backend = None

    def adapter_path(self):
        return None


# --------------------------------------------------------------------------- #
# Aggregations                                                                 #
# --------------------------------------------------------------------------- #
def test_epoch_sql_normalises_mixed_timestamp_formats():
    db = fresh_db()
    now = 1_760_000_000
    for ts in (sql_ts(now - 60), "2025-10-09T08:52:20+00:00", now - 120,
               (now - 180) * 1000, str(now - 240)):
        insert_metric(db, ts)
    rows = db.execute(f"SELECT {_dash_epoch_sql('timestamp')} AS t FROM metrics ORDER BY id"
                      ).fetchall()
    got = [r["t"] for r in rows]
    assert got[0] == now - 60
    assert got[1] == 1_759_999_940          # 2025-10-09T08:52:20Z
    assert got[2] == now - 120
    assert got[3] == now - 180              # epoch milliseconds
    assert got[4] == now - 240              # epoch seconds stored as text


def test_summary_aggregates_seeded_rows():
    db = fresh_db()
    db.ensure_schema("policy")
    dash_bind_runtime(model_manager=_FakeManager())
    now = time.time()
    recent = sql_ts(now - 300)
    # Two streaming requests (one failed), each with its generation row.
    insert_metric(db, recent, "chat_stream", 1000, 200, user="u1", queue=50.0)
    insert_metric(db, recent, "chat_stream_gen", 900, 200, prompt=100, completion=50,
                  ttft=200, tps=20.0, user="u1", adapter="ad-x", backend="mlx-lm", cached=40)
    insert_metric(db, recent, "chat_stream", 3000, 200, user="u2", queue=150.0)
    insert_metric(db, recent, "chat_stream_gen", 2800, 200, prompt=300, completion=100,
                  ttft=400, tps=10.0, user="u2", adapter="ad-y", backend="mlx-lm", cached=60)
    insert_metric(db, recent, "chat_stream", 5, 503, error="4 generations queued", user="u2")
    # Two plain-lane non-streaming turns: no agent_step rows, so their own
    # token totals count (the agent-lane case is the next test).
    insert_metric(db, recent, "chat", 500, 200, prompt=70, completion=30, user="u1", conv="c9")
    insert_metric(db, recent, "chat", 400, 200, prompt=10, completion=5, user="u1", conv="c10")
    # Old row outside the 1h window.
    insert_metric(db, sql_ts(now - 7200), "chat_stream", 99999, 500, error="boom", user="u3")
    db.execute("INSERT INTO routing_events (created_at, selected_node, status, duration_ms) "
               "VALUES (?, 'node-a', 'ok', 120), (?, 'node-a', 'failed', 80), "
               "(?, 'node-b', 'ok', 50)", (recent, recent, recent))
    db.execute("INSERT INTO tool_calls (created_at, name, duration_ms, error) VALUES "
               "(?, 'web_search', 300, NULL), (?, 'web_search', 500, 'timeout'), "
               "(?, 'calculator', 2, NULL)", (recent, recent, recent))
    db.execute("INSERT INTO tool_decisions (created_at, tool, decision, approver_id, latency_ms) "
               "VALUES (?, 'shell', 'allow', 'admin', 4000), (?, 'shell', 'deny', 'admin', 2000), "
               "(?, 'web_search', 'auto', NULL, NULL)", (recent, recent, recent))
    db.execute("INSERT INTO feedback (created_at, user_prompt, assistant_response, rating, "
               "adapter_id) VALUES (?, 'q', 'a', 1, 'ad-x'), (?, 'q', 'a', -1, 'ad-y'), "
               "(?, 'q2', 'a', 1, 'ad-x')", (recent, recent, recent))
    db.commit()

    out = dashboard_summary(db, "1h", now=now, sink=EventSink())
    k = out["kpis"]
    assert k["requests"] == 5                      # 3 chat_stream + 2 chat, not gens
    assert k["errors"] == 1 and abs(k["error_rate"] - 0.2) < 1e-9
    assert k["latency_ms"]["p50"] == 1000.0        # nearest rank of 400, 500, 1000, 3000
    assert k["latency_ms"]["p95"] == 3000.0
    assert k["ttft_ms"]["p50"] in (200.0, 400.0)
    assert k["decode_tps"]["avg"] == 15.0
    # 100+300 from the generations, 70+10 from the plain chat rows.
    assert k["prompt_tokens"] == 480 and k["completion_tokens"] == 185
    assert k["cached_prompt_tokens"] == 100
    assert abs(k["cache_hit_ratio"] - 0.25) < 1e-9  # 100 of the 400 tagged prompt tokens
    assert k["queue_ms"]["samples"] == 2 and k["queue_ms"]["p95"] == 150.0
    assert k["tool_calls"] == 3 and k["tool_errors"] == 1
    assert k["approvals"] == 3
    assert k["feedback"]["up"] == 2 and k["feedback"]["down"] == 1
    assert k["active_users"] == 2
    assert out["errors_by_type"][0] == {"status": 503, "type": "# generations queued",
                                        "count": 1}
    users = {u["user_id"]: u for u in out["per_user"]}
    assert users["u2"]["requests"] == 2 and users["u2"]["errors"] == 1
    assert users["u1"]["prompt_tokens"] == 180
    models = {(m["adapter_id"], m["backend"]): m for m in out["per_model"]}
    assert models[("ad-y", "mlx-lm")]["prompt_tokens"] == 300
    nodes = {n["node"]: n for n in out["per_node"]}
    assert nodes["node-a"]["requests"] == 2 and nodes["node-a"]["failed"] == 1
    tools = {t["name"]: t for t in out["tools"]}
    assert tools["web_search"]["error_rate"] == 0.5
    assert out["approvals"]["human_latency_p50_ms"] in (2000.0, 4000.0)
    by_adapter = {a["adapter_id"]: a for a in out["feedback_by_adapter"]}
    assert by_adapter["ad-x"]["up"] == 2 and by_adapter["ad-y"]["down"] == 1
    # The 24h window includes the old failure.
    assert dashboard_summary(db, "24h", now=now, sink=EventSink())["kpis"]["requests"] == 6


def test_summary_does_not_double_count_agent_steps():
    db = fresh_db()
    now = time.time()
    ts = sql_ts(now - 60)
    insert_metric(db, ts, "agent_step", 100, 200, prompt=50, completion=10, conv="cz")
    insert_metric(db, ts, "agent_step", 100, 200, prompt=60, completion=20, conv="cz")
    insert_metric(db, ts, "chat", 300, 200, prompt=110, completion=30, conv="cz")
    k = dashboard_summary(db, "1h", now=now, sink=EventSink())["kpis"]
    assert k["prompt_tokens"] == 110 and k["completion_tokens"] == 30
    assert k["requests"] == 1 and k["generations"] == 2


def test_timeseries_buckets_and_validation():
    db = fresh_db()
    now = 1_760_000_000
    insert_metric(db, sql_ts(now - 30), "chat_stream", 100, 200)
    insert_metric(db, sql_ts(now - 30), "chat_stream", 300, 500, error="x")
    insert_metric(db, sql_ts(now - 1800), "chat_stream", 200, 200)
    out = dashboard_timeseries(db, "requests,errors,error_rate,latency_p50", "1h",
                               "5m", now=now)
    assert out["bucket_s"] == 300
    series = {s["metric"]: s["values"] for s in out["series"]}
    assert sum(series["requests"]) == 3 and sum(series["errors"]) == 1
    assert series["error_rate"][-1] == 50.0
    assert len(out["labels"]) == len(series["requests"])
    try:
        dashboard_timeseries(db, "nope", "1h", now=now)
        raise AssertionError("unknown metric accepted")
    except ValueError:
        pass


# --------------------------------------------------------------------------- #
# Event sink and http.request                                                  #
# --------------------------------------------------------------------------- #
def test_event_sink_captures_structured_events_and_redacts():
    sink = EventSink(capacity=50, problem_capacity=10)
    logger = logging.getLogger("local_llm.dashtest")
    logger.setLevel(logging.INFO)
    logger.addHandler(sink)
    try:
        log_event(logger, logging.INFO, "dash.test", api_key="sk-live-abcdef1234567890",
                  note="Authorization: Bearer abcdefghijklmnop1234", count=3)
        log_event(logger, logging.WARNING, "dash.warn", reason="disk low")
        log_event(logger, logging.INFO, "dash.test", count=4)
    finally:
        logger.removeHandler(sink)
    counts = sink.counts()
    assert counts["events"]["dash.test"] == 2 and counts["events"]["dash.warn"] == 1
    newest = sink.records(limit=10, event="dash.test")
    assert len(newest) == 2 and newest[0]["fields"]["count"] == 4
    oldest = newest[1]
    text = repr(oldest)
    assert "sk-live-abcdef1234567890" not in text and "abcdefghijklmnop1234" not in text
    problems = sink.records(level="WARNING")
    assert [p["event"] for p in problems] == ["dash.warn"]


def test_http_request_event_carries_user_id():
    app, _, db = build_app()
    logging.getLogger("local_llm").setLevel(logging.INFO)
    with TestClient(app) as c:
        assert login(c, "admin", "adminpw123").status_code == 200
        admin_id = db.get_user_by_username("admin")["id"]
        assert c.get("/api/dashboards/events?limit=5").status_code == 200
        recs = EVENT_SINK.records(event="http.request", limit=20)
        mine = [r for r in recs if r["fields"].get("route") == "/api/dashboards/events"]
        assert mine, recs[:3]
        assert mine[0]["fields"].get("user_id") == admin_id


# --------------------------------------------------------------------------- #
# Attribution                                                                  #
# --------------------------------------------------------------------------- #
def test_metric_and_feedback_rows_are_tagged_with_the_active_adapter():
    db = fresh_db()
    dash_bind_runtime(model_manager=_FakeManager())
    dash_reset_active_cache()
    db.log_metric("chat_stream_gen", 10.0, 200, stats=GenerationStats(prompt_tokens=5),
                  conversation_id="cv1")
    row = db.execute("SELECT adapter_id FROM metrics ORDER BY id DESC LIMIT 1").fetchone()
    assert row["adapter_id"] == "base@base-model"
    db.record_feedback("p", "a", 1, 1, source="implicit")
    row = db.execute("SELECT adapter_id FROM feedback ORDER BY id DESC LIMIT 1").fetchone()
    assert row["adapter_id"] == "base@base-model"


def test_feedback_route_tags_adapter_from_the_conversation():
    app, _, db = build_app()
    with TestClient(app) as c:
        assert login(c, "admin", "adminpw123").status_code == 200
        db.ensure_schema(DASH_SCHEMA_NAME)
        insert_metric(db, sql_ts(time.time()), "chat_stream_gen", adapter="ad-conv01",
                      conv="conv-77")
        r = c.post("/api/feedback", json={"user_prompt": "hi", "assistant_response": "hello",
                                          "rating": 1, "conversation_id": "conv-77"})
        assert r.status_code == 200, r.text
        row = db.execute("SELECT adapter_id FROM feedback ORDER BY id DESC LIMIT 1").fetchone()
        assert row["adapter_id"] == "ad-conv01"


# --------------------------------------------------------------------------- #
# ELO                                                                          #
# --------------------------------------------------------------------------- #
def test_elo_math():
    assert abs(elo_expected(1500, 1500) - 0.5) < 1e-12
    assert abs(elo_expected(1900, 1500) - 1 / (1 + 10 ** (-1))) < 1e-12
    a, b = elo_update(1500, 1500, "a")
    assert (a, b) == (1512.0, 1488.0)                      # K=24
    a, b = elo_update(1600, 1400, "tie")
    assert a < 1600 and b > 1400 and abs((a + b) - 3000) < 1e-9
    a, b = elo_update(1400, 1600, "a")                     # upset: bigger swing
    assert a - 1400 > 12
    try:
        elo_update(1500, 1500, "x")
        raise AssertionError("bad winner accepted")
    except ValueError:
        pass
    assert elo_parse_verdict("<think>A seems fine</think>\nB") == "b"
    assert elo_parse_verdict("Tie.") == "tie"
    assert elo_parse_verdict("Answer A is clearer") == "a"
    assert elo_parse_verdict("both are fine") is None


def test_elo_records_matches_and_recompute_is_identical():
    db = fresh_db()
    assert elo_record_match(db, "ad-1", "ad-2", "a", prompt="p1", ref="r1") is not None
    assert elo_record_match(db, "ad-1", "ad-2", "a", prompt="p1", ref="r1") is None  # dup
    elo_record_match(db, "ad-2", "ad-3", "tie", prompt="p2")
    elo_record_match(db, "ad-3", "ad-1", "a", prompt="p3", judge="model", source="tournament")

    def snapshot():
        return {r["adapter_id"]: (round(r["elo"], 3), r["games"], r["wins"], r["losses"], r["ties"])
                for r in db.execute("SELECT * FROM adapter_ratings")}

    before = snapshot()
    assert before["ad-1"][1:] == (2, 1, 1, 0)
    assert before["ad-2"][1:] == (2, 0, 1, 1)
    # Zero-sum up to the 4-decimal storage rounding.
    assert abs(sum(v[0] for v in before.values()) - 3 * ELO_INITIAL) < 0.01
    assert elo_recompute(db) == 3
    assert snapshot() == before
    # Prompt text is kept only at LOG_CHAT_CONTENT=full (default: metadata).
    row = db.execute("SELECT prompt, prompt_hash FROM adapter_matches LIMIT 1").fetchone()
    assert row["prompt"] is None and len(row["prompt_hash"]) == 16


def test_implicit_matches_from_feedback():
    db = fresh_db()
    rows = [("What is MLX?", 1, "ad-new"), ("  what is mlx ", -1, "ad-old"),
            ("What is MLX", -1, "ad-new"),           # same adapter both ways: no self-match
            ("Unrelated prompt", 1, "ad-old"), ("hello", 1, None)]
    for prompt, rating, adapter in rows:
        db.execute("INSERT INTO feedback (user_prompt, assistant_response, rating, adapter_id) "
                   "VALUES (?, 'x', ?, ?)", (prompt, rating, adapter))
    db.commit()
    assert elo_sync_implicit(db) == 1
    assert elo_sync_implicit(db) == 0                      # idempotent
    match = db.execute("SELECT * FROM adapter_matches").fetchone()
    assert (match["adapter_a"], match["adapter_b"], match["winner"]) == ("ad-new", "ad-old", "a")
    assert match["source"] == "implicit"
    ratings = {r["adapter_id"]: r for r in elo_ratings(db, include_unrated=False)}
    assert ratings["ad-new"]["elo"] > ratings["ad-old"]["elo"]
    assert ratings["ad-new"]["rank"] == 1


def test_eval_and_dashboard_routes_rbac():
    app, _, _ = build_app()
    with TestClient(app) as anon:
        assert anon.get("/api/dashboards/summary").status_code == 401
        assert anon.get("/api/eval/compare/next").status_code == 401
    with TestClient(app) as c:
        user = admin_and_user(c)
        for path in ("/api/dashboards/summary?window=1h", "/api/dashboards/timeseries",
                     "/api/dashboards/events", "/api/eval/ratings", "/api/eval/matches",
                     "/api/eval/tournament"):
            assert user.get(path).status_code == 403, path
            assert c.get(path).status_code == 200, (path, c.get(path).text)
        assert user.post("/api/eval/tournament", json={"confirm": True}).status_code == 403
        assert user.post("/api/eval/ratings/recompute").status_code == 403
        assert user.get("/api/eval/compare/next").status_code == 200
        assert c.get("/api/dashboards/summary?window=2y").status_code == 400
        assert c.get("/api/dashboards/timeseries?metric=bogus").status_code == 400
        assert c.get("/api/dashboards/events?level=LOUD").status_code == 400


def test_pair_votes_are_blind_once_per_user_and_free_form_is_admin_only():
    app, _, db = build_app()
    with TestClient(app) as c:
        user = admin_and_user(c)
        pair_id = elo_create_pair(db, "Say hi", "Hi!", "Hello there.", "ad-p1", "ad-p2", "run1")
        nxt = user.get("/api/eval/compare/next").json()
        assert nxt["remaining"] == 1 and nxt["pair"]["id"] == pair_id
        assert "adapter_a" not in nxt["pair"] and "adapter_b" not in nxt["pair"]
        r = user.post("/api/eval/compare/vote", json={"pair_id": pair_id, "winner": "b"})
        assert r.status_code == 200, r.text
        assert set(r.json()["match"]["revealed"]) == {"a", "b"}
        assert user.post("/api/eval/compare/vote",
                         json={"pair_id": pair_id, "winner": "a"}).status_code == 409
        assert user.get("/api/eval/compare/next").json()["pair"] is None
        assert c.get("/api/eval/compare/next").json()["pair"]["id"] == pair_id
        assert user.post("/api/eval/compare/vote",
                         json={"pair_id": "nope", "winner": "a"}).status_code == 404
        assert user.post("/api/eval/compare/vote",
                         json={"pair_id": pair_id, "winner": "x"}).status_code == 400
        free = {"prompt": "q", "answer_a": "1", "answer_b": "2", "adapter_a": "none",
                "adapter_b": "ad-p1", "winner": "a"}
        assert user.post("/api/eval/compare/vote", json=free).status_code == 403
        r = c.post("/api/eval/compare/vote", json=free)
        assert r.status_code == 200, r.text
        assert r.json()["match"]["adapter_a"].startswith("base@")
        assert c.post("/api/eval/compare/vote", json=free).status_code == 409
        bad = dict(free, adapter_b="ad-never-seen")
        assert c.post("/api/eval/compare/vote", json=bad).status_code == 400
        ratings = c.get("/api/eval/ratings").json()
        ids = {r["adapter_id"] for r in ratings["ratings"]}
        assert {"ad-p1", "ad-p2"} <= ids and ratings["matches"] == 2
        assert c.get("/api/eval/matches?limit=5").json()["matches"][0]["source"] == "vote"


def _fake_catalog(model_id=None):
    return [{"choice": "none", "key": "base@test", "label": "base model"},
            {"choice": "latest", "key": "ad-latest", "label": "latest"},
            {"choice": "20250101-000000", "key": "ad-backup", "label": "20250101-000000"}]


def _wait_idle(c, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = c.get("/api/eval/tournament").json()
        if not snap["running"]:
            return snap
        time.sleep(0.05)
    raise AssertionError("tournament did not finish")


def test_tournament_model_judge_and_human_pairs():
    app, _, db = build_app()
    original = elo_mod.adapter_catalog
    elo_mod.adapter_catalog = _fake_catalog
    loads: list[str] = []

    def fake_load(ctx, choice):
        loads.append(choice)
        ctx.model_manager.adapter_choice = choice

    def fake_generate(ctx, messages, max_tokens, temperature):
        if messages[0]["role"] == "system":           # the judge: prefer the "latest" answer
            user_text = messages[-1]["content"]
            a_part = user_text.split("Answer A:")[1].split("Answer B:")[0]
            return "A" if "from latest" in a_part else ("B" if "from latest" in user_text
                                                        else "TIE")
        return f"answer from {ctx.model_manager.adapter_choice}"

    try:
        with TestClient(app) as c:
            assert login(c, "admin", "adminpw123").status_code == 200
            ctx = app.state.ctx
            ctx.extras["eval_load_adapter"] = fake_load
            ctx.extras["eval_generate"] = fake_generate
            ctx.model_manager.adapter_choice = "latest"
            assert c.post("/api/eval/tournament", json={}).status_code == 400     # no confirm
            assert c.post("/api/eval/tournament", json={"confirm": True,
                                                        "adapters": ["ghost"]}).status_code == 400
            ctx.tasks.chat_in_flight = 1
            assert c.post("/api/eval/tournament", json={"confirm": True}).status_code == 409
            ctx.tasks.chat_in_flight = 0
            r = c.post("/api/eval/tournament", json={"confirm": True, "prompts": ["p1", "p2"],
                                                     "seed": 7})
            assert r.status_code == 202, r.text
            snap = _wait_idle(c)
            assert snap["error"] == "" and snap["phase"] == "done", snap
            assert snap["matches"] == 6                   # 3 pairs x 2 prompts
            # base model answered last, judged without another load, then restored.
            assert loads[-2] == "none" and loads[-1] == "latest"
            assert ctx.model_manager.adapter_choice == "latest"
            ratings = {r["adapter_id"]: r for r in elo_ratings(db, include_unrated=False)}
            assert ratings["ad-latest"]["wins"] == 4
            assert ratings["ad-latest"]["rank"] == 1
            judged = db.execute("SELECT DISTINCT judge, source FROM adapter_matches").fetchall()
            assert [tuple(j) for j in judged] == [("model", "tournament")]

            r = c.post("/api/eval/tournament", json={"confirm": True, "judge": "human",
                                                     "adapters": ["none", "latest"],
                                                     "prompts": ["only one"]})
            assert r.status_code == 202, r.text
            snap = _wait_idle(c)
            assert snap["pairs"] == 1 and snap["error"] == ""
            nxt = c.get("/api/eval/compare/next").json()["pair"]
            assert nxt["prompt"] == "only one"
            assert {nxt["answer_a"], nxt["answer_b"]} == {"answer from none",
                                                          "answer from latest"}
    finally:
        elo_mod.adapter_catalog = original


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
