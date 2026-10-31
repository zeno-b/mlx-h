"""Front-end: Admin > Dashboards (operations overview and adapter ratings).

The server already records one metrics row per request and generation, one
row per tool call, approval decision, routing decision and rating, and keeps
the recent structured log events in memory (obslog.EVENT_SINK). Until now an
admin could see none of it except as raw JSON from /api/metrics. These loaders
render it into the cards ui_settings.py adds (the "dashboards" block):

  * Overview: KPI tiles for the chosen window (1h, 24h, 7d) from
    GET /api/dashboards/summary: request rate, error rate, latency, TTFT,
    decode speed, tokens, prompt cache hits, queue wait, tools, approvals,
    feedback and active users.
  * Over time: uiChart line charts from GET /api/dashboards/timeseries (two
    calls, the route takes at most eight metrics each).
  * Breakdown: per user, per model/adapter/engine, errors by type, per node,
    per tool, approvals and feedback by adapter (tables and a bar chart).
  * Events: the most frequent structured events and the recent warnings and
    errors (GET /api/dashboards/events); already redacted on the server.
  * Adapter ratings: the ELO table and recent matches (GET /api/eval/ratings,
    /api/eval/matches), a tournament launcher with live progress
    (POST /api/eval/tournament, polled through GET /api/eval/tournament), and
    the blind A/B voting card (GET /api/eval/compare/next,
    POST /api/eval/compare/vote). The voting card never shows which adapter
    wrote which answer until the vote is stored; model output is set with
    textContent only.

Wiring: dashTabHook() runs once when this part executes (after
ui_script_admin.py, so ADMIN_TABS exists) and registers the "admin/dashboards"
tab: the loader runs when the tab is shown, and a 3-second poll follows a
running tournament (and refreshes everything once a minute) only while the tab
stays visible. Charts go through uiChart (ui_script_charts.py); when that part
is missing the chart hosts say so instead of breaking the tab.

Every route behind these cards is admin-only on the server except the vote and
next-pair routes; the Admin section is admin-only in the rail, which is
cosmetic. Results and failures are toasts.
"""

from __future__ import annotations

UI_JS_DASHBOARDS = r"""   // -------------------------------------------------------- dashboards ---

   var dashState = { win: "24h", level: "WARNING", pair: null, ticks: 0, tourney: false };

   // ---------------------------------------------------------- helpers ---

   function dashEl(tag, cls, text) {
     var el = document.createElement(tag);
     if (cls) el.className = cls;
     if (text != null) el.textContent = String(text);
     return el;
   }

   function dashIsNum(v) {
     return v !== null && v !== undefined && v !== "" && isFinite(Number(v));
   }

   // 1234567 -> "1.2M", 23456 -> "23.5k", 12.345 -> "12.3". A tile is read
   // at a glance; the exact figure is in the tables and the JSON routes.
   function dashNum(v) {
     if (!dashIsNum(v)) return "-";
     var n = Number(v);
     var a = Math.abs(n);
     if (a >= 1e6) return (Math.round(n / 1e5) / 10) + "M";
     if (a >= 1e4) return (Math.round(n / 100) / 10) + "k";
     if (a >= 100) return String(Math.round(n));
     return String(Math.round(n * 10) / 10);
   }

   function dashMs(v) {
     if (!dashIsNum(v)) return "-";
     var n = Number(v);
     if (n >= 1000) return (Math.round(n / 100) / 10) + " s";
     return Math.round(n) + " ms";
   }

   function dashRate(v) {
     if (!dashIsNum(v)) return "-";
     return (Math.round(Number(v) * 1000) / 10) + "%";
   }

   // SQLite's CURRENT_TIMESTAMP ("2026-10-24 21:05:00") is UTC without a zone;
   // Date() would read it as local time (or refuse it, in Safari).
   function dashDate(value) {
     if (!value) return null;
     var text = String(value);
     if (/^\d{4}-\d\d-\d\d \d\d:\d\d(:\d\d)?$/.test(text)) text = text.replace(" ", "T") + "Z";
     var d = new Date(text);
     return isNaN(d.getTime()) ? null : d;
   }

   function dashPad(n) {
     return n < 10 ? "0" + n : String(n);
   }

   function dashTime(value) {
     var d = dashDate(value);
     if (!d) return value ? String(value) : "-";
     var today = new Date();
     var hm = dashPad(d.getHours()) + ":" + dashPad(d.getMinutes());
     if (d.toDateString() === today.toDateString()) return hm + ":" + dashPad(d.getSeconds());
     return d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) + " " + hm;
   }

   // A chart's x label: the time of day, plus the weekday on the 7-day window.
   function dashTick(value, win) {
     var d = dashDate(value);
     if (!d) return "";
     var hm = dashPad(d.getHours()) + ":" + dashPad(d.getMinutes());
     if (win === "7d") return d.toLocaleDateString(undefined, { weekday: "short" }) + " " + hm;
     return hm;
   }

   function dashShortModel(name) {
     var text = String(name || "-");
     var parts = text.split("/");
     return parts[parts.length - 1] || text;
   }

   function dashChart(id, spec) {
     var host = document.getElementById(id);
     if (!host) return;
     if (typeof uiChart === "function") {
       uiChart(host, spec);
     } else {
       host.textContent = "Charts are unavailable in this build.";
     }
   }

   // Fill a tbody with rows of cells (text or DOM nodes); an empty list shows
   // one muted row instead of a bare header.
   function dashTable(id, rows, cols, empty) {
     var body = document.getElementById(id);
     if (!body) return;
     body.textContent = "";
     if (!rows.length) {
       var tr0 = document.createElement("tr");
       var td0 = dashEl("td", "muted", empty || "Nothing in this window.");
       td0.colSpan = cols;
       tr0.appendChild(td0);
       body.appendChild(tr0);
       return;
     }
     rows.forEach(function(cells) {
       var tr = document.createElement("tr");
       cells.forEach(function(cell) {
         var td = document.createElement("td");
         if (cell && cell.nodeType) td.appendChild(cell);
         else td.textContent = cell == null ? "-" : String(cell);
         tr.appendChild(td);
       });
       body.appendChild(tr);
     });
   }

   function dashLevelPill(level) {
     var name = String(level || "INFO").toUpperCase();
     var kind = (name === "ERROR" || name === "CRITICAL") ? "error"
              : (name === "WARNING" ? "warn" : "muted");
     return dashEl("span", "pill " + kind, name.toLowerCase());
   }

   // ---------------------------------------------------------- overview ---

   function dashTile(host, label, value, sub) {
     var tile = dashEl("div", "dash-kpi");
     tile.appendChild(dashEl("span", "dash-kpi-label", label));
     tile.appendChild(dashEl("span", "dash-kpi-value", value));
     if (sub) tile.appendChild(dashEl("span", "dash-kpi-sub", sub));
     tile.title = label + ": " + value + (sub ? " (" + sub + ")" : "");
     host.appendChild(tile);
   }

   function dashRenderSummary(d) {
     var k = d.kpis || {};
     var lat = k.latency_ms || {};
     var ttft = k.ttft_ms || {};
     var tps = k.decode_tps || {};
     var queue = k.queue_ms || {};
     var fb = k.feedback || {};
     var host = document.getElementById("dashKpis");
     host.textContent = "";
     dashTile(host, "Requests", dashNum(k.requests), dashNum(k.requests_per_min) + " per minute");
     dashTile(host, "Error rate", dashRate(k.error_rate), dashNum(k.errors) + " failed");
     dashTile(host, "Latency p50", dashMs(lat.p50), "p95 " + dashMs(lat.p95));
     dashTile(host, "First token p50", dashMs(ttft.p50), "p95 " + dashMs(ttft.p95));
     dashTile(host, "Decode speed", dashIsNum(tps.avg) ? dashNum(tps.avg) + " tok/s" : "-",
              dashNum(k.generations) + " generations");
     dashTile(host, "Tokens in / out", dashNum(k.prompt_tokens) + " / " + dashNum(k.completion_tokens),
              "prompt / completion");
     dashTile(host, "Prompt cache hits", dashRate(k.cache_hit_ratio),
              dashNum(k.cached_prompt_tokens) + " cached tokens");
     dashTile(host, "Queue wait p95", dashMs(queue.p95), "p50 " + dashMs(queue.p50));
     // No calls means no rate: "- failed" read like a broken value.
     dashTile(host, "Tool calls", dashNum(k.tool_calls),
       dashIsNum(k.tool_error_rate) ? dashRate(k.tool_error_rate) + " failed" : "none failed");
     dashTile(host, "Approvals", dashNum(k.approvals), "policy decisions");
     dashTile(host, "Thumbs up", dashRate(fb.up_rate),
              dashNum(fb.up) + " up, " + dashNum(fb.down) + " down");
     dashTile(host, "Active users", dashNum(k.active_users), "made a request");

     var active = d.active_adapter || {};
     var range = document.getElementById("dashRange");
     range.textContent = "From " + dashTime(d.since) + " to " + dashTime(d.until) +
       (active.key ? ". Serving: " + (active.choice || "none") + " (" + active.key + ")" +
                     (active.backend ? " on " + active.backend : "") : "") + ".";

     dashTable("dashUsers", (d.per_user || []).map(function(u) {
       return [u.username || u.user_id, dashNum(u.requests), dashNum(u.errors), dashMs(u.latency_p50),
               dashNum(u.prompt_tokens) + " / " + dashNum(u.completion_tokens)];
     }), 5);
     dashTable("dashModels", (d.per_model || []).map(function(m) {
       return [dashShortModel(m.model), m.adapter_label || "-", m.backend || "-",
               dashNum(m.generations), dashMs(m.ttft_p50),
               dashIsNum(m.decode_tps_avg) ? dashNum(m.decode_tps_avg) + " tok/s" : "-"];
     }), 6);
     var errs = d.errors_by_type || [];
     dashChart("dashErrors", {
       type: "bar", title: "Errors by type",
       labels: errs.map(function(e) { return e.type + " (HTTP " + e.status + ")"; }),
       series: [{ label: "Errors", values: errs.map(function(e) { return e.count; }) }],
       empty: "No failed requests in this window."
     });
     dashTable("dashNodes", (d.per_node || []).map(function(n) {
       return [n.node, dashNum(n.requests), dashNum(n.failed), dashMs(n.duration_p50), dashMs(n.duration_p95)];
     }), 5, "No routed requests in this window.");
     dashTable("dashTools", (d.tools || []).map(function(t) {
       return [t.name, dashNum(t.calls), dashRate(t.error_rate), dashMs(t.duration_avg), dashMs(t.duration_p95)];
     }), 5, "No tool calls in this window.");
     var appr = d.approvals || {};
     var kv = document.getElementById("dashApprovals");
     kv.textContent = "";
     function dashKvPair(label, value) {
       kv.appendChild(dashEl("dt", null, label));
       kv.appendChild(dashEl("dd", null, value));
     }
     dashKvPair("Decisions", dashNum(appr.total));
     (appr.by_decision || []).forEach(function(b) { dashKvPair(b.decision, dashNum(b.count)); });
     dashKvPair("Human answer time p50", dashMs(appr.human_latency_p50_ms));
     dashTable("dashFeedback", (d.feedback_by_adapter || []).map(function(a) {
       return [a.label, dashNum(a.total), dashNum(a.up), dashNum(a.down), dashRate(a.up_rate)];
     }), 5, "No ratings in this window.");
   }

   async function dashLoadSeries(win) {
     var base = "/api/dashboards/timeseries?window=" + encodeURIComponent(win) + "&metric=";
     var a = await uiAct("Dashboard charts", function() {
       return fetchJSON(base + "requests,errors,latency_p50,latency_p95,ttft_p50,ttft_p95,decode_tps,prompt_tokens");
     });
     var b = await uiAct("Dashboard charts", function() {
       return fetchJSON(base + "completion_tokens,tool_calls,tool_errors,cache_hit_ratio,queue_p95");
     });
     if (!a || !b) return;
     var by = {};
     [a.data, b.data].forEach(function(doc) {
       (doc.series || []).forEach(function(s) { by[s.metric] = s; });
     });
     var labels = (a.data.labels || []).map(function(t) { return dashTick(t, win); });
     function dashSeriesLine(id, metrics, unit, title, max) {
       var series = metrics.filter(function(m) { return by[m]; }).map(function(m) {
         return { label: by[m].label, values: by[m].values };
       });
       var spec = { type: "line", series: series, labels: labels, unit: unit, title: title,
                    empty: "Nothing recorded in this window." };
       if (max) spec.max = max;
       dashChart(id, spec);
     }
     dashSeriesLine("dashChartRequests", ["requests", "errors"], "", "Requests and errors per interval");
     dashSeriesLine("dashChartLatency", ["latency_p50", "latency_p95"], "ms", "Request latency");
     dashSeriesLine("dashChartTtft", ["ttft_p50", "ttft_p95"], "ms", "Time to first token");
     dashSeriesLine("dashChartTps", ["decode_tps"], "tok/s", "Decode speed");
     dashSeriesLine("dashChartTokens", ["prompt_tokens", "completion_tokens"], "tok", "Tokens per interval");
     dashSeriesLine("dashChartTools", ["tool_calls", "tool_errors"], "", "Tool calls per interval");
     dashSeriesLine("dashChartCache", ["cache_hit_ratio"], "%", "Prompt cache hit ratio", 100);
     dashSeriesLine("dashChartQueue", ["queue_p95"], "ms", "Queue wait p95");
   }

   // ------------------------------------------------------------ events ---

   async function dashLoadEvents() {
     var lvl = dashState.level;
     var r = await uiAct("Events", function() {
       return fetchJSON("/api/dashboards/events?limit=50" + (lvl ? "&level=" + encodeURIComponent(lvl) : ""));
     });
     if (!r) return;
     var counts = (r.data.counts || {}).events || {};
     var names = Object.keys(counts).sort(function(x, y) { return counts[y] - counts[x]; }).slice(0, 10);
     dashChart("dashEventCounts", {
       type: "bar", title: "Most frequent events", labels: names,
       series: [{ label: "Events", values: names.map(function(n) { return counts[n]; }) }],
       empty: "No structured events yet."
     });
     var since = (r.data.counts || {}).since;
     document.getElementById("dashEventSince").textContent =
       "Structured log events since this process started (" + dashTime(since) +
       "), kept in memory whatever LOG_FORMAT says. Secrets are redacted.";
     dashTable("dashEvents", (r.data.events || []).map(function(e) {
       return [dashTime(e.ts), dashLevelPill(e.level), e.event || e.logger || "-", e.msg || ""];
     }), 4, lvl ? "No warnings or errors recorded." : "No events recorded.");
   }

   // ----------------------------------------------------------- ratings ---

   async function dashLoadRatings() {
     var r = await uiAct("Adapter ratings", function() { return fetchJSON("/api/eval/ratings"); });
     if (!r) return;
     var d = r.data || {};
     dashTable("dashRatings", (d.ratings || []).map(function(x) {
       var name = dashEl("span", null, x.label || x.adapter_id);
       name.title = x.adapter_id;
       var cell = dashEl("span", "row nowrap");
       cell.appendChild(name);
       if (x.active) cell.appendChild(dashEl("span", "badge", "serving"));
       if (x.provisional) cell.appendChild(dashEl("span", "muted", "provisional"));
       return [String(x.rank), cell, String(Math.round(Number(x.elo) || 0)), dashNum(x.games),
               x.wins + " / " + x.losses + " / " + x.ties, dashRate(x.win_rate)];
     }), 6, "No adapters yet.");
     var src = d.by_source || {};
     var parts = Object.keys(src).map(function(k) { return src[k] + " " + k; });
     document.getElementById("dashRatingsNote").textContent =
       dashNum(d.matches) + " matches" + (parts.length ? " (" + parts.join(", ") + ")" : "") +
       ". " + dashNum(d.open_pairs) + " prepared pairs open for votes.";
     var m = await uiAct("Matches", function() { return fetchJSON("/api/eval/matches?limit=10"); });
     if (!m) return;
     dashTable("dashMatches", (m.data.matches || []).map(function(x) {
       var winner = x.winner === "tie" ? "tie" : (x.winner === "a" ? x.label_a : x.label_b);
       return [dashTime(x.created_at), x.label_a, x.label_b, winner, x.judge, x.source];
     }), 6, "No matches yet.");
   }

   async function dashRecompute() {
     var r = await uiAct("Recompute", function() { return postJSON("/api/eval/ratings/recompute", {}); });
     if (!r) return;
     uiToast("Replayed " + dashNum(r.data.replayed) + " matches (" + dashNum(r.data.implicit_added) +
             " new from feedback).", "ok");
     dashLoadRatings();
   }

   // -------------------------------------------------------- tournament ---

   function dashRenderTourney(s) {
     s = s || {};
     var box = document.getElementById("dashTourneyBox");
     var btn = document.getElementById("dashTourneyBtn");
     var cancel = document.getElementById("dashTourneyCancel");
     var pill = document.getElementById("dashTourneyPill");
     var text = document.getElementById("dashTourneyText");
     var was = dashState.tourney;
     dashState.tourney = !!s.running;
     btn.disabled = !!s.running;
     cancel.hidden = !s.running;
     if (!s.running && !s.finished_at) { box.hidden = true; return; }
     box.hidden = false;
     if (s.running) {
       pill.className = "pill running";
       pill.textContent = s.phase || "running";
       text.textContent = dashNum(s.done) + " of " + dashNum(s.total) + (s.current ? ": " + s.current : "");
     } else if (s.error) {
       pill.className = "pill error";
       pill.textContent = "failed";
       text.textContent = s.error;
     } else {
       pill.className = "pill ok";
       pill.textContent = "done";
       text.textContent = s.judge === "human"
         ? dashNum(s.pairs) + " pairs ready to vote on"
         : dashNum(s.matches) + " matches judged" +
           (s.skipped ? ", " + dashNum(s.skipped) + " unclear verdicts skipped" : "");
     }
     if (was && !s.running) {
       dashLoadRatings();
       dashLoadPair();
     }
   }

   async function dashPollTourney() {
     var r = null;
     try { r = await fetchJSON("/api/eval/tournament"); } catch (err) { r = null; }
     if (r) dashRenderTourney(r.data);
   }

   async function dashStartTournament() {
     var judge = uiSegValue(document.getElementById("dashJudge")) || "model";
     if (!uiConfirm("Run a tournament? Every adapter answers the same prompt set in turn on the " +
                    "shared model server, so chats get 'Model not ready' until it ends.")) return;
     var r = await uiAct("Tournament", function() {
       return postJSON("/api/eval/tournament", { confirm: true, judge: judge });
     });
     if (!r) return;
     uiToast("Tournament started.", "ok");
     dashRenderTourney(r.data.status);
   }

   async function dashCancelTournament() {
     var r = await uiAct("Cancel tournament", function() {
       return postJSON("/api/eval/tournament/cancel", {});
     });
     if (r) uiToast("Stopping after the current answer.", "info");
   }

   // ------------------------------------------------------- A/B voting ---

   async function dashLoadPair() {
     var r = await uiAct("Compare answers", function() { return fetchJSON("/api/eval/compare/next"); });
     if (!r) return;
     var p = r.data.pair;
     dashState.pair = p ? p.id : null;
     document.getElementById("dashVoteLeft").textContent = dashNum(r.data.remaining || 0) + " waiting";
     document.getElementById("dashVoteEmpty").hidden = !!p;
     document.getElementById("dashVoteBox").hidden = !p;
     if (!p) return;
     // Model output: text only, never markup.
     document.getElementById("dashVotePrompt").textContent = p.prompt || "";
     document.getElementById("dashVoteA").textContent = p.answer_a || "";
     document.getElementById("dashVoteB").textContent = p.answer_b || "";
   }

   async function dashVote(winner) {
     var id = dashState.pair;
     if (!id) return;
     dashState.pair = null;
     var r = await uiAct("Vote", function() {
       return postJSON("/api/eval/compare/vote", { pair_id: id, winner: winner });
     });
     if (!r) { dashState.pair = id; return; }
     var rev = (r.data.match || {}).revealed || {};
     uiToast("Vote saved. A was " + (rev.a || "?") + ", B was " + (rev.b || "?") + ".", "ok");
     dashLoadPair();
     dashLoadRatings();
   }

   // ------------------------------------------------------------ wiring ---

   async function dashLoad() {
     var win = dashState.win;
     var r = await uiAct("Dashboards", function() {
       return fetchJSON("/api/dashboards/summary?window=" + encodeURIComponent(win));
     });
     if (r) dashRenderSummary(r.data);
     await Promise.all([dashLoadSeries(win), dashLoadEvents(), dashLoadRatings(),
                        dashLoadPair(), dashPollTourney()]);
   }

   // Every 3 s while the tab is visible: follow a running tournament; once a
   // minute, refresh the whole tab.
   async function dashPoll() {
     dashState.ticks += 1;
     if (dashState.ticks % 20 === 0) return dashLoad();
     return dashPollTourney();
   }

   function dashWire() {
     wireSeg(document.getElementById("dashWindow"), function(value) {
       dashState.win = value || "24h";
       dashLoad();
     });
     wireSeg(document.getElementById("dashEventLevel"), function(value) {
       dashState.level = value || "";
       dashLoadEvents();
     });
     wireSeg(document.getElementById("dashJudge"), null);
   }

   function dashTabHook() {
     if (typeof ADMIN_TABS === "undefined") return;
     var entry = typeof adminTabEntry === "function" ? adminTabEntry("admin/dashboards") : null;
     if (!entry) {
       entry = { key: "admin/dashboards" };
       ADMIN_TABS.push(entry);
     }
     entry.load = function() {
       dashWire();
       dashState.ticks = 0;
       return dashLoad();
     };
     entry.poll = [function() { return dashPoll(); }, 3000];
   }

   dashTabHook();
"""

__all__ = ["UI_JS_DASHBOARDS"]
