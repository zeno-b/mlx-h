"""Front-end: Admin > Evals (the evaluation and regression suites, evals.py).

Fills the cards ui_settings.py adds (the "evals" block):

  * Run evaluations: a mode seg (scripted or live model), the suites to run
    and "save as baseline", POST /api/evals/run; a running run is followed
    through GET /api/evals/runs (its `job`) and can be cancelled with
    POST /api/evals/cancel. The model and adapter a live run would measure,
    and whether the promotion gate is on, come from the same route.
  * Scores against the baseline: for the selected run (the newest by
    default), a uiChart bar chart of each suite's score next to the
    baseline's, and a table with the change, latency and tokens. "Make
    baseline" keeps the run as the reference for its model, adapter and
    mode (POST /api/evals/baseline).
  * Runs: the recent runs (GET /api/evals/runs); View selects one
    (GET /api/evals/runs/{id}).
  * Cases: the selected run's failed (or all) cases with the checks that
    failed. Case details come from model output and are set as text only.

Wiring: evalsTabHook() runs once when this part executes (after
ui_script_admin.py, so ADMIN_TABS exists) and registers "admin/evals": the
loader runs when the tab is shown and a 3-second poll follows a running
evaluation only while the tab stays visible. Every route is admin-only.
"""

from __future__ import annotations

UI_JS_EVALS = r"""   // ------------------------------------------------------------- evals ---

   var evalsState = { selected: null, run: null, filter: "failed", running: false, ticks: 0 };

   var EVALS_LABELS = {
     tool_calling: "Tool calling", rag: "Knowledge base", multi_turn: "Multi-turn",
     routing: "Routing", parsers: "Parsers", api: "API shapes"
   };

   function evalsEl(tag, cls, text) {
     var el = document.createElement(tag);
     if (cls) el.className = cls;
     if (text != null) el.textContent = String(text);
     return el;
   }

   function evalsIsNum(v) {
     return v !== null && v !== undefined && v !== "" && isFinite(Number(v));
   }

   function evalsPct(v) {
     if (!evalsIsNum(v)) return "-";
     return (Math.round(Number(v) * 1000) / 10) + "%";
   }

   // A change in percentage points, signed: "+2.5", "-10".
   function evalsDelta(v) {
     if (!evalsIsNum(v)) return "-";
     var n = Math.round(Number(v) * 1000) / 10;
     return (n > 0 ? "+" : "") + n;
   }

   function evalsMs(v) {
     if (!evalsIsNum(v)) return "-";
     var n = Number(v);
     if (n >= 1000) return (Math.round(n / 100) / 10) + " s";
     return Math.round(n) + " ms";
   }

   function evalsTime(value) {
     if (!value) return "-";
     var d = new Date(String(value));
     if (isNaN(d.getTime())) return String(value);
     return d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) + " " +
       d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
   }

   function evalsLabel(suite) {
     return EVALS_LABELS[suite] || String(suite || "-");
   }

   function evalsShort(name) {
     var parts = String(name || "-").split("/");
     return parts[parts.length - 1] || String(name || "-");
   }

   // Fill a tbody with rows of cells (text or DOM nodes); an empty list shows
   // one muted row instead of a bare header.
   function evalsTable(id, rows, cols, empty) {
     var body = document.getElementById(id);
     if (!body) return;
     body.textContent = "";
     if (!rows.length) {
       var tr0 = document.createElement("tr");
       var td0 = evalsEl("td", "muted", empty || "Nothing yet.");
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

   function evalsPill(text, kind) {
     return evalsEl("span", "pill " + (kind || "muted"), text);
   }

   function evalsSuites() {
     var box = document.getElementById("evalsSuites");
     if (!box) return [];
     var out = [];
     Array.prototype.forEach.call(box.querySelectorAll("input[type=checkbox]"), function(cb) {
       if (cb.checked) out.push(cb.value);
     });
     return out;
   }

   // ---------------------------------------------------------- loading ---

   async function evalsLoad() {
     var r = await uiAct("Evaluations", function() { return fetchJSON("/api/evals/runs?limit=30"); });
     if (!r) return;
     var data = r.data || {};
     evalsRenderActive(data);
     evalsRenderJob(data.job || {});
     evalsRenderRuns(data.runs || []);
     var runs = data.runs || [];
     var keep = evalsState.selected && runs.some(function(x) { return x.id === evalsState.selected; });
     if (!keep && runs.length) evalsState.selected = runs[0].id;
     if (evalsState.selected) return evalsShow(evalsState.selected);
     evalsRenderRun(null);
   }

   function evalsRenderActive(data) {
     var dl = document.getElementById("evalsActive");
     if (!dl) return;
     dl.textContent = "";
     var a = data.active || {};
     var pairs = [
       ["Live model", evalsShort(a.model) + (a.status && a.status !== "ready" ? " (" + a.status + ")" : "")],
       ["Adapter", a.adapter_label || a.adapter_key || "-"],
       ["Regression gate", data.on_promote ? "on: promoted adapters are evaluated" : "off"],
       ["Allowed drop", evalsPct(data.max_regression)]
     ];
     pairs.forEach(function(p) {
       dl.appendChild(evalsEl("dt", null, p[0]));
       dl.appendChild(evalsEl("dd", null, p[1]));
     });
   }

   function evalsRenderJob(job) {
     var box = document.getElementById("evalsJobBox");
     var btn = document.getElementById("evalsRunBtn");
     var cancel = document.getElementById("evalsCancelBtn");
     var running = !!job.running;
     var was = evalsState.running;
     evalsState.running = running;
     if (btn) btn.disabled = running;
     if (cancel) cancel.hidden = !running;
     if (box) box.hidden = !running && !job.error;
     var pill = document.getElementById("evalsJobPill");
     var text = document.getElementById("evalsJobText");
     if (pill) {
       pill.className = "pill " + (running ? "running" : (job.error ? "error" : "muted"));
       pill.textContent = running ? "running" : (job.status || "idle");
     }
     if (text) {
       var progress = job.total ? " " + job.done + " of " + job.total : "";
       text.textContent = running
         ? (job.mode || "") + " run" + progress + (job.current ? ": " + job.current : "")
         : (job.error || "");
     }
     if (was && !running && job.run_id) {
       evalsState.selected = job.run_id;
       if (job.error) uiToast("Evaluation: " + job.error, "error");
       else uiToast(job.regressed ? "Evaluation finished with a regression." : "Evaluation finished.",
                    job.regressed ? "error" : "ok");
     }
   }

   function evalsRenderRuns(runs) {
     var rows = runs.map(function(run) {
       var result;
       if (run.status !== "done") result = evalsPill(run.status, run.status === "error" ? "error" : "warn");
       else if (run.regressed) result = evalsPill("regressed", "error");
       else if (run.baseline_id) result = evalsPill("no regression", "ok");
       else result = evalsPill("no baseline", "muted");
       var view = evalsEl("button", "btn btn-sm", run.id === evalsState.selected ? "Shown" : "View");
       view.type = "button";
       view.title = "Show this run's scores and cases";
       view.addEventListener("click", function() {
         evalsState.selected = run.id;
         evalsShow(run.id);
       });
       return [evalsTime(run.created_at), run.mode, evalsShort(run.model) + " / " +
               (run.adapter_label || run.adapter_key || "-"), evalsPct(run.score),
               (run.passed || 0) + " / " + (run.total || 0), result, run.trigger || "-", view];
     });
     evalsTable("evalsRuns", rows, 8, "No evaluation has run yet.");
   }

   async function evalsShow(id) {
     var r = await uiAct("Evaluation run", function() {
       return fetchJSON("/api/evals/runs/" + encodeURIComponent(id));
     });
     if (!r) return;
     evalsRenderRun(r.data);
   }

   function evalsRenderRun(run) {
     evalsState.run = run;
     var summary = document.getElementById("evalsSummary");
     var btn = document.getElementById("evalsBaselineBtn");
     if (btn) btn.disabled = !run || run.status !== "done";
     if (!run || !run.suites) {
       if (summary) summary.textContent = run && run.status === "running"
         ? "This run is still going." : "No run yet. Run the scripted suites to check the harness.";
       evalsTable("evalsCompare", [], 7, "No run selected.");
       evalsChart([], [], []);
       evalsRenderCases();
       return;
     }
     var scores = run.scores || {};
     var comparison = run.comparison || {};
     var base = run.baseline;
     var text = run.mode + " run on " + evalsShort(run.model) + " / " +
       (run.adapter_label || run.adapter_key || "-") + ": " + evalsPct(run.score) + " overall, " +
       evalsTime(run.created_at || run.started_at) + ".";
     if (run.status !== "done") text += " Status: " + run.status + (run.error ? " (" + run.error + ")" : "") + ".";
     else if (!base && !run.baseline_id) text += " No baseline for this model and adapter when it ran.";
     else if (run.regressed) text += " Regressed: " + (run.regressions || []).map(evalsLabel).join(", ") + ".";
     else text += " No suite dropped more than the allowed amount.";
     if (summary) summary.textContent = text;
     var suites = run.suites || [];
     var current = [], before = [];
     var rows = suites.map(function(suite) {
       var s = scores[suite] || {};
       var c = comparison[suite] || {};
       current.push(evalsIsNum(s.score) ? Math.round(Number(s.score) * 1000) / 10 : null);
       before.push(evalsIsNum(c.baseline) ? Math.round(Number(c.baseline) * 1000) / 10 : null);
       var change = evalsIsNum(c.delta)
         ? evalsPill(evalsDelta(c.delta), c.regressed ? "error" : (Number(c.delta) < 0 ? "warn" : "ok"))
         : "-";
       var tokens = (s.prompt_tokens || 0) + (s.completion_tokens || 0);
       return [evalsLabel(suite) + (s.error ? " (error)" : ""), evalsPct(s.score),
               (s.passed || 0) + " / " + (s.total || 0), evalsPct(c.baseline), change,
               evalsMs(s.latency_ms), tokens ? String(tokens) : "-"];
     });
     evalsTable("evalsCompare", rows, 7, "This run has no suites.");
     evalsChart(suites.map(evalsLabel), current, before);
     evalsRenderCases();
   }

   function evalsChart(labels, current, before) {
     var host = document.getElementById("evalsChart");
     if (!host) return;
     if (typeof uiChart !== "function") {
       host.textContent = "Charts are unavailable in this build.";
       return;
     }
     var series = [{ label: "This run", values: current }];
     if (before.some(function(v) { return v !== null; })) series.push({ label: "Baseline", values: before });
     uiChart(host, { type: "bar", title: "Suite scores", labels: labels, series: series,
                     unit: "%", max: 100, empty: "No scores yet." });
   }

   function evalsRenderCases() {
     var run = evalsState.run;
     var title = document.getElementById("evalsCasesTitle");
     var results = (run && run.results) || [];
     var shown = evalsState.filter === "all" ? results : results.filter(function(r) { return !r.passed; });
     if (title) title.textContent = run && run.id ? "Cases of " + run.id : "Cases";
     var rows = shown.map(function(r) {
       var bad = Object.keys(r.checks || {}).filter(function(k) { return !r.checks[k]; });
       var why = bad.join(", ") || (r.error ? r.error : (r.passed ? "none" : "-"));
       var cell = evalsEl("span", r.passed ? "muted" : null, why);
       if (r.detail) cell.title = JSON.stringify(r.detail).slice(0, 600);
       var tokens = (r.prompt_tokens || 0) + (r.completion_tokens || 0);
       return [evalsLabel(r.suite), r["case"], evalsPct(r.score), cell, evalsMs(r.latency_ms),
               tokens ? String(tokens) : "-"];
     });
     evalsTable("evalsCases", rows, 6, run ? (evalsState.filter === "all" ? "No cases." : "Every case passed.")
                                         : "No run selected.");
   }

   // ---------------------------------------------------------- actions ---

   async function evalsRun() {
     var suites = evalsSuites();
     if (!suites.length) {
       uiToast("Pick at least one suite.", "error");
       return;
     }
     var mode = uiSegValue(document.getElementById("evalsMode")) || "scripted";
     if (mode === "live" && !uiConfirm("Run the evaluations against the loaded model? It takes a few " +
                                       "minutes and refuses while anything else is generating.")) return;
     var save = document.getElementById("evalsSaveBaseline");
     var r = await uiAct("Run evaluations", function() {
       return postJSON("/api/evals/run", { suites: suites, mode: mode, save_baseline: !!(save && save.checked) });
     });
     if (!r) return;
     evalsState.selected = r.data.run_id;
     uiToast("Evaluation started.", "ok");
     evalsRenderJob(r.data.status || { running: true, mode: mode });
   }

   async function evalsCancel() {
     var r = await uiAct("Cancel evaluation", function() { return postJSON("/api/evals/cancel", {}); });
     if (r) uiToast("Stopping after the current case.", "info");
   }

   async function evalsSaveBaseline() {
     var run = evalsState.run;
     if (!run || !run.id) return;
     if (!uiConfirm("Make " + run.id + " the baseline for " + evalsShort(run.model) + " / " +
                    (run.adapter_label || run.adapter_key) + " (" + run.mode + ")?")) return;
     var r = await uiAct("Make baseline", function() {
       return postJSON("/api/evals/baseline", { run_id: run.id });
     });
     if (r) uiToast("Baseline saved. Later runs of this model and adapter compare against it.", "ok");
   }

   // Every 3 s while the tab is visible: follow a running evaluation; once a
   // minute, refresh the tab.
   async function evalsPoll() {
     evalsState.ticks += 1;
     if (!evalsState.running && evalsState.ticks % 20 !== 0) return;
     return evalsLoad();
   }

   function evalsWire() {
     wireSeg(document.getElementById("evalsMode"), null);
     wireSeg(document.getElementById("evalsCaseFilter"), function(value) {
       evalsState.filter = value || "failed";
       evalsRenderCases();
     });
   }

   function evalsTabHook() {
     if (typeof ADMIN_TABS === "undefined") return;
     var entry = typeof adminTabEntry === "function" ? adminTabEntry("admin/evals") : null;
     if (!entry) {
       entry = { key: "admin/evals" };
       ADMIN_TABS.push(entry);
     }
     entry.load = function() {
       evalsWire();
       evalsState.ticks = 0;
       return evalsLoad();
     };
     entry.poll = [function() { return evalsPoll(); }, 3000];
   }

   evalsTabHook();
"""

__all__ = ["UI_JS_EVALS"]
