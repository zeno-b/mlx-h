"""Front-end: the operational Settings panels (engine, admission, catalogue,
benchmark, training autopilot, cluster nodes and pairing, remote access).

Wave 3 gave the server facts only it knows at run time: which inference engine
really runs and why it was configured that way (GET /api/backends), how full
the admission queue is (GET /api/health .queue), what each catalogue model is
and whether it fits this Mac (GET /api/models, GET /api/models/aliases), what
the last benchmark measured and recommends (GET /api/bench/*), what the
training autopilot would do per method (GET /api/train/methods, /plan,
POST /export) and which nodes are paired (GET/DELETE /api/cluster/paired,
POST /api/cluster/pairing). Until now the UI showed none of it, or showed it
as a text dump in a <pre> (the cluster view). These loaders render it into the
cards ui_settings.py adds (the "panels-ops" block), with kit classes only:
.table, .kv, .pill, .badge, .meter, .readonly, and uiChart for the queue
sparklines, the benchmark's decode speeds, the autopilot's memory breakdown
and free memory per shard-capable node.

Wiring, and why it is done this way:

  * opsTabHook() runs once when this part executes (after ui_script_admin.py,
    so ADMIN_TABS exists) and points the lazy-load registry entries of the
    tabs these cards live in at the loaders below, adding entries for tabs
    that had none (admin/system). Nothing loads or polls while its tab is
    hidden, exactly like the other tabs.
  * The model switcher table (#modelTable) is rendered by loadModels() in
    ui_script_admin.py, which every switch reloads through postModel(). The
    catalogue columns (aliases, fits in RAM, recommended, metadata) must
    survive that reload, so loadModels is wrapped once here rather than
    copied: the base loader keeps the current-model box and the adapter
    select, then opsLoadCatalog re-renders the table with the extra columns.
  * Restart-only facts (MODEL_BACKEND, WEB_HOST, ...) stay read-only schema
    fields; these cards explain them and print the exact command to change
    them, they never write them.

Every route behind these cards is admin-only on the server; the sections are
admin-only in the rail, which is cosmetic. Results and failures are toasts.
No proxy route exists for mlx-engine's own /v1/engine/stats, so the engine
card does not show those counters (it says where they live instead).
"""

from __future__ import annotations

UI_JS_OPS = r"""   // --------------------------------------------------------------- ops ---

   var opsAdmHistory = { run: [], wait: [] };
   var opsPerfTicks = 0;
   var opsBenchActive = false;
   var opsTrainState = { methods: [], plans: {}, pick: null };
   var opsPairTimer = null;

   // ---------------------------------------------------------- helpers ---

   function opsEl(tag, cls, text) {
     var el = document.createElement(tag);
     if (cls) el.className = cls;
     if (text != null) el.textContent = String(text);
     return el;
   }

   function opsClear(el) {
     if (el) el.textContent = "";
     return el;
   }

   function opsNum(v) {
     return typeof v === "number" && isFinite(v);
   }

   function opsPill(text, kind) {
     return opsEl("span", "pill " + (kind || "muted"), text);
   }

   function opsBadge(text, kind) {
     return opsEl("span", "badge" + (kind ? " " + kind : ""), text);
   }

   function opsBtn(label, cls, onClick, title) {
     var b = opsEl("button", cls || "btn btn-sm", label);
     b.type = "button";
     if (title) b.setAttribute("title", title);
     b.onclick = onClick;
     return b;
   }

   // "12.5 tok/s", "-" for a value the server did not report.
   function opsFmt(v, unit, digits) {
     if (!opsNum(v)) return "-";
     var d = digits == null ? (Math.abs(v) >= 100 ? 0 : 1) : digits;
     var n = d ? v.toFixed(d) : String(Math.round(v));
     return unit ? n + " " + unit : n;
   }

   function opsGB(mb) {
     return opsNum(mb) ? opsFmt(mb / 1024, "GB", 1) : "-";
   }

   function opsCtx(tokens) {
     var n = Number(tokens) || 0;
     if (!n) return "-";
     return n >= 1024 && n % 1024 === 0 ? (n / 1024) + "k" : String(n);
   }

   function opsHuman(key) {
     return String(key || "").replace(/_/g, " ");
   }

   function opsShortId(id) {
     var s = String(id || "");
     return s.indexOf("/") >= 0 ? s.slice(s.lastIndexOf("/") + 1) : s;
   }

   function opsIsLoopback(host) {
     var h = String(host || "").toLowerCase().replace(/^\[|\]$/g, "");
     return h === "localhost" || h === "::1" || h.indexOf("127.") === 0;
   }

   // A settled fetch: {data} or {error}, so Promise.all over several routes
   // renders what arrived instead of failing the whole card on one 503.
   function opsTry(promise) {
     return promise.then(function(out) { return { data: out.data }; },
                         function(err) { return { error: (err && err.message) || String(err) }; });
   }

   // <dl class="kv"> from [label, value, ddClass] rows. A null or empty value
   // is skipped, so a fact the server did not report is absent rather than
   // printed as "undefined".
   function opsKv(dl, rows) {
     if (!dl) return;
     opsClear(dl);
     rows.forEach(function(r) {
       if (!r || r[1] == null || r[1] === "") return;
       dl.appendChild(opsEl("dt", "", r[0]));
       var dd = opsEl("dd", r[2] || "");
       if (r[1].nodeType) dd.appendChild(r[1]);
       else dd.textContent = String(r[1]);
       dl.appendChild(dd);
     });
   }

   function opsCell(row, content, cls) {
     var td = opsEl("td", cls || "");
     if (content == null) td.textContent = "-";
     else if (content.nodeType) td.appendChild(content);
     else td.textContent = String(content);
     row.appendChild(td);
     return td;
   }

   function opsEmptyRow(tbody, cols, text) {
     if (!tbody) return;
     opsClear(tbody);
     var row = opsEl("tr");
     var td = opsEl("td", "muted", text);
     td.setAttribute("colspan", String(cols));
     row.appendChild(td);
     tbody.appendChild(row);
   }

   // A cell's main line plus a muted second line (reason, metadata).
   function opsTwoLine(main, sub, mainCls) {
     var box = opsEl("div", "");
     box.appendChild(main && main.nodeType ? main : opsEl("div", mainCls || "", main == null ? "-" : main));
     if (sub) box.appendChild(opsEl("div", "hint", sub));
     return box;
   }

   // A meter with its figure under it. The width is chart geometry, the one
   // inline style the kit allows.
   function opsMeter(pct, caption, kind) {
     var wrap = opsEl("div", "ops-meter");
     var meter = opsEl("div", "meter");
     var fill = opsEl("div", kind || "");
     var p = opsNum(pct) ? Math.max(0, Math.min(100, pct)) : 0;
     fill.style.width = p.toFixed(1) + "%";
     meter.appendChild(fill);
     wrap.appendChild(meter);
     wrap.appendChild(opsEl("span", "hint mono", caption));
     return wrap;
   }

   function opsGauge(title, pct, caption, kind) {
     var box = opsEl("div", "stack tight grow");
     box.appendChild(opsEl("span", "field-label", title));
     box.appendChild(opsMeter(pct, caption, kind));
     return box;
   }

   function opsStatusKind(status) {
     var s = String(status || "");
     if (s === "ready" || s === "running" || s === "healthy") return "ok";
     if (s === "starting" || s === "loading" || s === "restarting") return "running";
     if (s === "stopped" || !s) return "muted";
     return "error";
   }

   // ------------------------------------------------------- tab wiring ---

   function opsTabSet(key, load, poll) {
     if (typeof ADMIN_TABS === "undefined") return;
     var entry = typeof adminTabEntry === "function" ? adminTabEntry(key) : null;
     if (!entry) {
       entry = { key: key };
       ADMIN_TABS.push(entry);
     }
     entry.load = load;
     if (poll) entry.poll = poll;
   }

   function opsTabHook() {
     opsTabSet("performance", function() {
       opsPerfTicks = 0;
       if (typeof loadPerf === "function") loadPerf();
       opsLoadEngine();
       return opsLoadAdmission();
     }, [function() { return opsPerfPoll(); }, 3000]);
     opsTabSet("models/model", function() {
       opsLoadBench();
       return opsModelTab();
     }, [function() { return opsBenchPoll(); }, 3000]);
     opsTabSet("models/training", function() {
       if (typeof loadDatasetStats === "function") loadDatasetStats();
       return opsLoadTraining();
     });
     opsTabSet("admin/cluster", function() {
       opsPairReset();
       return opsLoadCluster();
     }, [function() { return opsLoadCluster(); }, 5000]);
     opsTabSet("admin/system", function() { return opsLoadRemote(); });
   }

   opsTabHook();

   // loadModels (ui_script_admin.py) renders the plain switcher table and is
   // what postModel reloads after a switch; wrap it once so every caller gets
   // the catalogue columns too (see the module docstring).
   var opsBaseLoadModels = typeof loadModels === "function" ? loadModels : null;
   if (opsBaseLoadModels && !opsBaseLoadModels.opsWrapped) {
     loadModels = async function() {
       await opsBaseLoadModels();
       return opsLoadCatalog();
     };
     loadModels.opsWrapped = true;
   }

   // ------------------------------------------- performance: the engine ---

   async function opsLoadEngine() {
     var state = document.getElementById("opsEngineState");
     var table = document.getElementById("opsEngineTable");
     var out;
     try {
       out = (await fetchJSON("/api/backends")).data || {};
     } catch (err) {
       opsClear(state).appendChild(opsEl("span", "hint warn", "Could not load the engine state: " + err.message));
       return;
     }
     var cur = out.current || {};
     var configured = out.configured || "";
     var running = cur.backend || configured;
     opsClear(state);
     state.appendChild(opsPill(cur.backend_label || running || "unknown engine", cur.error ? "error" : "ok"));
     state.appendChild(opsPill("model " + (cur.status || "unknown"), opsStatusKind(cur.status)));
     if (cur.backend_requested && cur.backend && cur.backend_requested !== cur.backend) {
       state.appendChild(opsPill("asked for " + cur.backend_requested, "warn"));
     }
     if (configured && running && configured !== running) {
       state.appendChild(opsPill(configured + " after a restart", "warn"));
     }
     if (cur.adapter_active) {
       state.appendChild(opsPill(cur.adapter_served ? "LoRA served" : "LoRA not served",
                                 cur.adapter_served ? "ok" : "warn"));
     }
     if (cur.error) state.appendChild(opsEl("span", "hint warn", cur.error));

     var adapter = null;
     if (cur.adapter) {
       adapter = cur.adapter + (cur.adapter_active
         ? (cur.adapter_served ? ", answering with it" : ", not given to this engine")
         : ", none active");
     }
     opsKv(document.getElementById("opsEngineFacts"), [
       ["Running", running ? (cur.backend_label ? cur.backend_label + " (" + running + ")" : running) : null],
       ["MODEL_BACKEND", configured, "mono"],
       ["Model", cur.model, "mono"],
       ["Request model", cur.request_model, "mono"],
       ["Adapter", adapter],
       ["Draft model", cur.draft_model, "mono"],
       ["Capabilities", (cur.capabilities || []).join(", ")],
       ["Accepted levers", (cur.accepted_levers || []).join(", ")]
     ]);

     var decisions = cur.decisions || [];
     var dhost = opsClear(document.getElementById("opsEngineDecisions"));
     decisions.forEach(function(d) { dhost.appendChild(opsEl("p", "hint mt-0", d)); });
     document.getElementById("opsEngineDecisionsBox").hidden = !decisions.length;

     var plan = cur.prompt_cache || null;
     document.getElementById("opsEngineCacheBox").hidden = !plan;
     if (plan) {
       var rows = [
         ["Entries", plan.size != null ? plan.size + (plan.size !== plan.configured_size
           ? " (PROMPT_CACHE_SIZE is " + plan.configured_size + ")" : "") : null],
         ["Recommended", plan.recommended_size],
         ["Byte cap", plan.bytes ? opsGB(plan.bytes / 1048576) + (plan.bytes_enforced
           ? ", enforced at every batch admission" : ", not enforced on the sequential path") : "none"],
         ["Lanes", (plan.lanes || []).join(", ")],
         ["KV per token", opsNum(plan.kv_bytes_per_token) ? Math.round(plan.kv_bytes_per_token / 1024) + " KB" : null],
         ["Hybrid model", plan.hybrid_model ? "yes: only exact prefixes are reused" : null]
       ];
       (plan.notes || []).forEach(function(n) { rows.push(["Note", n]); });
       opsKv(document.getElementById("opsEngineCache"), rows);
     }

     var engines = out.backends || [];
     if (!engines.length) { opsEmptyRow(table, 3, "No engines registered."); return; }
     opsClear(table);
     engines.forEach(function(b) {
       var tr = opsEl("tr");
       var top = opsEl("div", "row");
       top.appendChild(opsEl("span", "", b.label || b.name));
       top.appendChild(opsEl("span", "mono muted", b.name));
       if (b.name === running) top.appendChild(opsBadge("running", "ok"));
       else if (b.current) top.appendChild(opsBadge("configured", "warn"));
       opsCell(tr, opsTwoLine(top, b.description));
       opsCell(tr, opsTwoLine(opsPill(b.available ? "available" : "unavailable", b.available ? "ok" : "error"),
                              b.available ? "" : (b.reason || "")));
       var caps = opsEl("div", "row");
       (b.capabilities || []).forEach(function(c) { caps.appendChild(opsBadge(c)); });
       opsCell(tr, caps);
       table.appendChild(tr);
     });
   }

   // ---------------------------------------- performance: admission ---

   function opsPerfPoll() {
     opsPerfTicks += 1;
     // The live-performance card used to poll every 10 s on its own; the
     // queue needs 3 s, so the slower card rides every third tick.
     if (opsPerfTicks % 3 === 0 && typeof loadPerf === "function") loadPerf();
     return opsLoadAdmission();
   }

   async function opsLoadAdmission() {
     var pill = document.getElementById("opsAdmPill");
     var meters = document.getElementById("opsAdmMeters");
     var q = null;
     try {
       q = ((await fetchJSON("/api/health")).data || {}).queue || null;
     } catch (err) {
       pill.className = "pill error";
       pill.textContent = "unreachable";
       return;
     }
     if (!q) {
       pill.className = "pill muted";
       pill.textContent = "no data";
       return;
     }
     var running = Number(q.running) || 0;
     var waiting = Number(q.waiting) || 0;
     var limit = Number(q.limit) || 0;
     var depth = Number(q.max_waiting) || 0;
     var kind = "muted";
     var label = "idle";
     if (waiting && depth && waiting >= depth) { kind = "error"; label = "queue full"; }
     else if (waiting) { kind = "warn"; label = waiting + " waiting"; }
     else if (running) { kind = "running"; label = running + " running"; }
     pill.className = "pill " + kind;
     pill.textContent = label;

     opsClear(meters);
     meters.appendChild(opsGauge("Slots", limit ? running / limit * 100 : 0,
                                 running + " of " + limit + " running", running >= limit && limit ? "warn" : ""));
     meters.appendChild(opsGauge("Queue", depth ? waiting / depth * 100 : 0,
                                 waiting + " of " + depth + " waiting",
                                 waiting >= depth && depth ? "error" : (waiting ? "warn" : "")));

     uiChartPush(opsAdmHistory.run, running, 60);
     uiChartPush(opsAdmHistory.wait, waiting, 60);
     uiChart(document.getElementById("opsAdmSparkRun"), {
       type: "sparkline", series: [{ label: "running", values: opsAdmHistory.run }],
       max: Math.max(1, limit), title: "Running generations, last 3 minutes", empty: "-"
     });
     uiChart(document.getElementById("opsAdmSparkWait"), {
       type: "sparkline", series: [{ label: "waiting", values: opsAdmHistory.wait }],
       max: Math.max(1, depth), title: "Waiting requests, last 3 minutes", empty: "-"
     });

     var free = null;
     if (opsNum(q.free_mb)) free = opsGB(q.free_mb) + (q.min_free_mb ? ", floor " + opsGB(q.min_free_mb) : "");
     else if (q.memory_aware === false) free = "not probed: a plain slot gate";
     opsKv(document.getElementById("opsAdmFacts"), [
       ["Slots", running + " running of " + limit + " (MAX_CONCURRENT_GENERATIONS)"],
       ["Queue depth", waiting + " waiting of " + depth + (q.queue_multiplier ? " (" + q.queue_multiplier + " per slot)" : "")],
       ["Free memory", free],
       ["Reserved", opsNum(q.reserved_mb) ? opsGB(q.reserved_mb) + " for running generations" : null],
       ["Admitted", q.admitted_total],
       ["Rejected", q.rejected_total],
       ["Deferred for memory", q.deferred_total],
       ["Wait p50 / p95", opsFmt(q.wait_ms_p50, "ms") + " / " + opsFmt(q.wait_ms_p95, "ms")],
       ["Memory probe", q.memory_aware ? (q.memory_probe || "on") : "off"]
     ]);
   }

   // --------------------------------------------- models: catalogue ---

   async function opsModelTab() {
     opsLoadAliases();
     return loadModels();
   }

   function opsModelMeta(meta) {
     if (!meta) return "";
     var parts = [];
     if (meta.family) parts.push(meta.family);
     if (opsNum(meta.params_b)) {
       parts.push(meta.params_b + "B" + (opsNum(meta.active_params_b) && meta.active_params_b !== meta.params_b
         ? " (" + meta.active_params_b + "B active)" : ""));
     }
     if (meta.quant) parts.push(meta.quant);
     if (opsNum(meta.approx_size_gb)) parts.push("~" + meta.approx_size_gb + " GB");
     if (meta.context_max) parts.push(opsCtx(meta.context_max) + " context");
     if (meta.thinking) parts.push("thinking");
     if (meta.vision) parts.push("vision");
     if (meta.tool_format && meta.tool_format !== "auto") parts.push(meta.tool_format + " tools");
     if (meta.guessed) parts.push("estimated from the name");
     return parts.join(", ");
   }

   async function opsLoadCatalog() {
     var table = document.getElementById("modelTable");
     var d;
     try {
       d = (await fetchJSON("/api/models")).data || {};
     } catch (err) {
       return;   // the base loader already said why in the current-model box
     }
     var items = d.catalog || [];
     if (!items.length) { opsEmptyRow(table, 4, "No models in the catalogue."); return; }
     opsClear(table);
     items.forEach(function(item) {
       var meta = item.meta || {};
       var tr = opsEl("tr");
       var top = opsEl("div", "row");
       top.appendChild(opsEl("span", "mono", item.id));
       (item.aliases || []).forEach(function(a) { top.appendChild(opsBadge(a)); });
       if (item.recommended) top.appendChild(opsBadge("recommended", "ok"));
       opsCell(tr, opsTwoLine(top, opsModelMeta(meta)));
       var fits = item.fits_in_ram;
       var fit = fits === true ? opsBadge("fits", "ok") : (fits === false ? opsBadge("too big", "error") : opsBadge("unknown"));
       opsCell(tr, opsTwoLine(fit, meta.min_ram_gb ? "needs a Mac with " + meta.min_ram_gb + " GB" : ""));
       var pill = opsPill(item.current ? "in use" : (item.cached ? "cached" : "download"),
                          item.current ? "running" : (item.cached ? "ok" : "muted"));
       opsCell(tr, pill, "cell-status");
       var use = opsBtn("Use", "btn btn-sm", function() { useModel(item.id); }, "Switch to " + item.id);
       use.disabled = !!item.current;
       opsCell(tr, use, "cell-action");
       table.appendChild(tr);
     });
   }

   async function opsLoadAliases() {
     var table = document.getElementById("opsAliasTable");
     var list = document.getElementById("opsAliasList");
     var d;
     try {
       d = (await fetchJSON("/api/models/aliases")).data || {};
     } catch (err) {
       opsEmptyRow(table, 4, "Could not load aliases: " + err.message);
       return;
     }
     var aliases = d.aliases || {};
     var names = Object.keys(aliases).sort();
     opsClear(list);
     names.forEach(function(n) {
       var o = document.createElement("option");
       o.value = n;
       o.label = (aliases[n] || {}).id || n;
       list.appendChild(o);
     });
     if (!names.length) { opsEmptyRow(table, 4, "No aliases defined."); return; }
     opsClear(table);
     names.forEach(function(n) {
       var a = aliases[n] || {};
       var tr = opsEl("tr");
       opsCell(tr, n, "mono");
       opsCell(tr, a.id, "mono");
       opsCell(tr, opsBadge(a.source === "user" ? "yours" : "built-in"), "cell-status");
       var inUse = d.current === n || d.current === a.id;
       var use = opsBtn(inUse ? "In use" : "Use", "btn btn-sm", function() { useModel(n); }, "Switch to " + n + " (" + a.id + ")");
       use.disabled = inUse;
       opsCell(tr, use, "cell-action");
       table.appendChild(tr);
     });
   }

   // --------------------------------------------- models: benchmark ---

   async function opsLoadBench() {
     var calls = await Promise.all([
       opsTry(fetchJSON("/api/bench/results")),
       opsTry(fetchJSON("/api/bench/recommendation")),
       opsTry(fetchJSON("/api/bench/status"))
     ]);
     opsRenderBenchStatus(calls[2].data || null);
     opsRenderBenchRec(calls[1]);
     opsRenderBenchRuns(calls[0]);
   }

   function opsRenderBenchRuns(res) {
     var table = document.getElementById("opsBenchTable");
     var file = document.getElementById("opsBenchFile");
     var chart = document.getElementById("opsBenchChart");
     if (res.error) {
       opsEmptyRow(table, 5, "Could not load benchmark results: " + res.error);
       file.textContent = "";
       opsClear(chart);
       return;
     }
     var d = res.data || {};
     var latest = d.latest || null;
     var rows = (latest && latest.results) || [];
     if (!rows.length) {
       opsEmptyRow(table, 5, "No benchmark has run on this machine yet.");
       file.textContent = "";
       opsClear(chart);
       return;
     }
     opsClear(table);
     var labels = [];
     var tps = [];
     rows.forEach(function(r) {
       var name = r.alias || opsShortId(r.model_id);
       var tr = opsEl("tr");
       opsCell(tr, opsTwoLine(opsEl("span", "mono", name),
                              opsCtx(r.context_size) + " context, " + (r.backend || "default engine")));
       opsCell(tr, r.success ? opsFmt(r.decode_tps, "tok/s") : "-", "nowrap");
       opsCell(tr, r.success ? opsFmt(r.ttft_ms, "ms") : "-", "nowrap");
       opsCell(tr, opsNum(r.peak_rss_mb) ? opsGB(r.peak_rss_mb) : "-", "nowrap");
       var pill = r.success ? opsPill("ok", "ok") : (r.skipped ? opsPill("skipped", "muted") : opsPill("failed", "error"));
       opsCell(tr, opsTwoLine(pill, r.success ? "" : (r.error || "")));
       table.appendChild(tr);
       if (r.success) {
         labels.push(name + " " + opsCtx(r.context_size) + (r.backend ? " " + r.backend : ""));
         tps.push(r.decode_tps);
       }
     });
     uiChart(chart, {
       type: "bar", series: [{ label: "decode", values: tps }], labels: labels, unit: "tok/s",
       title: "Decode speed per benchmarked model", empty: "No cell of the latest run succeeded."
     });
     var runs = d.runs || [];
     var head = runs[0] || {};
     var text = "Latest run " + (head.file || latest.file || "");
     if (head.recorded_at) text += ", " + relativeTime(head.recorded_at);
     text += head.cells != null ? ": " + head.succeeded + " of " + head.cells + " cells succeeded" : "";
     text += runs.length > 1 ? ". " + (runs.length - 1) + " earlier run(s) kept." : ".";
     file.textContent = text;
   }

   function opsRenderBenchRec(res) {
     var box = opsClear(document.getElementById("opsBenchRec"));
     if (res.error) {
       box.appendChild(opsEl("p", "hint warn", "Could not load the recommendation: " + res.error));
       return;
     }
     var d = res.data || {};
     var c = d.computed || null;
     var applied = d.applied || null;
     var head = opsEl("div", "row");
     head.appendChild(opsEl("h3", "h3 mt-0", "Recommendation"));
     box.appendChild(head);
     if (!c) {
       box.appendChild(opsEl("p", "muted mt-0", "None yet: run a sweep to measure what this "
                             + (d.ram_gb ? Math.round(d.ram_gb) + " GB " : "") + "Mac serves well."));
       if (applied && applied.model_id) {
         box.appendChild(opsEl("p", "hint mt-0", "Applied earlier: " + (applied.alias || applied.model_id)
                               + (applied.context_size ? " at " + opsCtx(applied.context_size) + " context." : ".")));
       }
       return;
     }
     if (!c.model_id) {
       head.appendChild(opsBadge("nothing fits", "warn"));
       box.appendChild(opsEl("p", "hint mt-0", c.reason || "No benchmarked model qualified."));
       return;
     }
     var isApplied = !!(applied && applied.model_id === c.model_id
                        && Number(applied.context_size) === Number(c.context_size));
     if (isApplied) head.appendChild(opsBadge("applied", "ok"));
     else if (applied && applied.model_id) head.appendChild(opsBadge("differs from the applied one", "warn"));
     else head.appendChild(opsBadge("not applied"));
     var dl = opsEl("dl", "kv");
     box.appendChild(dl);
     opsKv(dl, [
       ["Model", (c.alias ? c.alias + ": " : "") + c.model_id, "mono"],
       ["Context", opsCtx(c.context_size) + " tokens"],
       ["Engine", c.backend],
       ["KV cache", c.kv_bits ? c.kv_bits + "-bit" : "full precision"],
       ["Decode", opsFmt(c.decode_tps, "tok/s")],
       ["First token", opsFmt(c.ttft_ms, "ms")],
       ["Peak memory", opsNum(c.peak_rss_mb) ? opsGB(c.peak_rss_mb) + " of " + d.ram_gb + " GB" : null]
     ]);
     if (c.reason) box.appendChild(opsEl("p", "hint mt-0", c.reason));
     var actions = opsEl("div", "row");
     actions.appendChild(opsBtn("Use this model", "btn btn-sm", function() { useModel(c.alias || c.model_id); },
                                "Switch the served model to " + c.model_id));
     if (!isApplied) {
       actions.appendChild(opsEl("span", "hint", "To make it this Mac's default (model, context and engine): "));
       actions.appendChild(opsEl("span", "mono hint", "python deploy.py recommend --apply"));
     }
     box.appendChild(actions);
   }

   // Returns true when a sweep this page watched has just finished.
   function opsRenderBenchStatus(s) {
     var was = opsBenchActive;
     opsBenchActive = !!(s && s.running);
     document.getElementById("opsBenchRunBtn").disabled = opsBenchActive;
     document.getElementById("opsBenchProgress").hidden = !opsBenchActive;
     if (opsBenchActive) {
       var done = Number(s.done) || 0;
       var total = Number(s.total) || 0;
       document.getElementById("opsBenchPill").textContent = total ? done + " of " + total : "starting";
       document.getElementById("opsBenchCurrent").textContent = s.current || "";
       document.getElementById("opsBenchMeter").style.width = (total ? Math.round(done / total * 100) : 0) + "%";
     }
     if (was && !opsBenchActive && s) {
       if (s.error) uiToast("Benchmark sweep failed: " + s.error, "error");
       else uiToast("Benchmark sweep finished" + (s.file ? ": " + s.file : "") + ". The model server is back.", "ok");
       return true;
     }
     return false;
   }

   async function opsBenchPoll() {
     if (!opsBenchActive) return;
     var r = await opsTry(fetchJSON("/api/bench/status"));
     if (r.error) return;
     if (opsRenderBenchStatus(r.data)) {
       opsLoadBench();
       loadModels();
     }
   }

   async function opsRunBench() {
     if (!uiConfirm("Run a benchmark sweep now?\n\nIt stops the serving model for the whole run: "
                    + "each cached model is loaded at several context sizes on each engine, which "
                    + "takes minutes per model, and chat is unavailable until it finishes. "
                    + "Nothing is downloaded.")) return;
     await uiAct("Could not start the sweep", async function() {
       var out = await postJSON("/api/bench/run", { confirm: true });
       uiToast("Benchmark sweep started. The model server stays stopped until it finishes.", "warn");
       opsRenderBenchStatus((out.data || {}).status || { running: true });
     });
   }

   // ---------------------------------------------- models: training ---

   function opsTrainFind(name) {
     for (var i = 0; i < opsTrainState.methods.length; i++) {
       if (opsTrainState.methods[i].name === name) return opsTrainState.methods[i];
     }
     return null;
   }

   async function opsLoadTraining() {
     var seg = document.getElementById("opsTrainSeg");
     var counts = document.getElementById("opsTrainCounts");
     var d;
     try {
       d = (await fetchJSON("/api/train/methods")).data || {};
     } catch (err) {
       opsClear(seg);
       opsEmptyRow(counts, 3, "Training methods unavailable: " + err.message);
       document.getElementById("opsTrainPlanReason").textContent = "";
       return;
     }
     var methods = d.methods || [];
     opsTrainState.methods = methods;
     opsTrainState.plans = {};
     if (!opsTrainFind(opsTrainState.pick)) opsTrainState.pick = d.configured || (methods[0] || {}).name || null;
     opsClear(seg);
     methods.forEach(function(m) {
       var b = opsEl("button", "", m.label || m.name);
       b.type = "button";
       b.setAttribute("data-value", m.name);
       if (!m.available) b.setAttribute("title", "Unavailable here: " + (m.reason || "see the reason below"));
       seg.appendChild(b);
     });
     uiSeg(seg, opsTrainState.pick);
     wireSeg(seg, function(value) {
       opsTrainState.pick = value;
       opsTrainShowMethod();
       opsLoadPlan(value);
     });
     opsTrainShowMethod();

     // One plan per method: its corpus count is that method's dataset size.
     opsEmptyRow(counts, 3, "counting...");
     var plans = await Promise.all(methods.map(function(m) {
       return opsTry(fetchJSON("/api/train/plan?method=" + encodeURIComponent(m.name)));
     }));
     opsClear(counts);
     methods.forEach(function(m, i) {
       var p = plans[i];
       opsTrainState.plans[m.name] = p.data || null;
       var tr = opsEl("tr");
       var top = opsEl("div", "row");
       top.appendChild(opsEl("span", "", m.label || m.name));
       if (m.configured) top.appendChild(opsBadge("configured", "ok"));
       if (m.experimental) top.appendChild(opsBadge("experimental", "warn"));
       opsCell(tr, opsTwoLine(top, m.data_format ? "data: " + m.data_format : ""));
       var count = p.data && p.data.corpus ? p.data.corpus.count : null;
       opsCell(tr, count == null ? "-" : String(count), "nowrap");
       opsCell(tr, opsTwoLine(opsPill(m.available ? "ready" : "unavailable", m.available ? "ok" : "error"),
                              m.available ? "" : (m.reason || "")));
       counts.appendChild(tr);
     });
     var picked = opsTrainState.pick;
     opsRenderPlan(opsTrainState.plans[picked] || null, picked ? (plans[methods.indexOf(opsTrainFind(picked))] || {}).error : null);
   }

   function opsTrainShowMethod() {
     var m = opsTrainFind(opsTrainState.pick) || {};
     var hint = document.getElementById("opsTrainMethodHint");
     var parts = [];
     if (m.description) parts.push(m.description);
     if (m.gate) parts.push("Promotion gate: " + m.gate + ".");
     if (m.available === false && m.reason) parts.push("Unavailable here: " + m.reason);
     hint.textContent = parts.join(" ");
     hint.classList.toggle("warn", m.available === false);
     document.getElementById("opsTrainExportBtn").disabled = !m.name;
   }

   async function opsLoadPlan(method) {
     if (opsTrainState.plans[method]) {
       opsRenderPlan(opsTrainState.plans[method], null);
       return;
     }
     document.getElementById("opsTrainPlanReason").textContent = "loading...";
     var r = await opsTry(fetchJSON("/api/train/plan?method=" + encodeURIComponent(method)));
     if (opsTrainState.pick !== method) return;
     if (r.data) opsTrainState.plans[method] = r.data;
     opsRenderPlan(r.data || null, r.error);
   }

   function opsRenderPlan(plan, error) {
     var reason = document.getElementById("opsTrainPlanReason");
     var dl = document.getElementById("opsTrainPlan");
     var chart = document.getElementById("opsTrainPlanChart");
     if (!plan) {
       reason.textContent = error ? "No plan: " + error : "No plan for this method.";
       reason.classList.toggle("warn", !!error);
       opsClear(dl);
       opsClear(chart);
       return;
     }
     var problem = plan.trainer_problem;
     reason.textContent = (plan.reason || "") + (problem ? " Trainer: " + problem : "");
     reason.classList.toggle("warn", plan.fits === false || !!problem);
     var model = plan.model || {};
     var corpus = plan.corpus || {};
     var stream = plan.layer_streaming;
     // What Retrain would do with the TRAIN_* values as set, when the
     // autopilot is off or picked something else.
     var mine = plan.configured || null;
     var mineText = null;
     if (mine && typeof mine === "object") {
       mineText = mine.num_layers + " layers, rank " + mine.lora_rank + ", batch " + mine.batch_size
         + (mine.max_seq_length ? ", " + mine.max_seq_length + " tokens" : "")
         + (opsNum(mine.estimated_peak_gb) ? ": ~" + mine.estimated_peak_gb + " GB" : "")
         + (mine.fits === true ? ", fits" : (mine.fits === false ? ", does not fit" : ""));
     }
     opsKv(dl, [
       ["Model", model.model ? model.model + (model.layers ? ", " + model.layers + " layers" : "") : null, "mono"],
       ["Memory budget", opsNum(plan.budget_gb) ? plan.budget_gb + " GB of " + plan.ram_gb + " GB" : null],
       ["Estimated peak", opsNum(plan.estimated_peak_gb) ? plan.estimated_peak_gb + " GB"
         + (plan.fits === true ? ", fits" : (plan.fits === false ? ", does not fit" : "")) : null],
       ["Layers / rank / batch", [plan.num_layers, plan.lora_rank, plan.batch_size].join(" / ")],
       ["Sequence window", plan.max_seq_length ? plan.max_seq_length + " tokens" : null],
       ["Gradient checkpointing", plan.grad_checkpoint ? "on" : "off"],
       ["Knob", plan.knob],
       ["Your TRAIN_* values", mineText],
       ["Corpus", corpus.count != null ? corpus.count + " examples; tokens p50 " + corpus.p50
         + ", p95 " + corpus.p95 + ", max " + corpus.max : null],
       ["Layer streaming", stream && typeof stream === "object" ? (stream.supported ? "supported" : "not supported") : null],
       ["Autopilot", plan.applied ? "on: Retrain uses this plan" : "off: Retrain uses the TRAIN_* values below"]
     ]);
     // Components the method does not use (no reference model for SFT, no
     // generation outside GRPO) are 0 and would only add empty rows.
     var parts = plan.breakdown_gb || null;
     var keys = parts ? Object.keys(parts).filter(function(k) { return opsNum(parts[k]) && parts[k] > 0; }) : [];
     if (keys.length) {
       uiChart(chart, {
         type: "bar", labels: keys.map(opsHuman),
         series: [{ label: "GB", values: keys.map(function(k) { return parts[k]; }) }],
         unit: "GB", max: opsNum(plan.budget_gb) ? plan.budget_gb : undefined,
         title: "Estimated training memory by component, against the budget"
       });
     } else {
       opsClear(chart);
     }
   }

   async function opsExportTraining() {
     var method = opsTrainState.pick;
     if (!method) return;
     var btn = document.getElementById("opsTrainExportBtn");
     btn.disabled = true;
     try {
       await uiAct("Export failed", async function() {
         var out = await postJSON("/api/train/export?method=" + encodeURIComponent(method), {});
         var r = out.data || {};
         var c = r.counts || {};
         var rows = [["Exported", (r.method || method) + (r.format ? " (" + r.format + ")" : "")]];
         Object.keys(c).forEach(function(k) { rows.push([opsHuman(k), c[k]]); });
         rows.push(["Written to", r.dir || null, "mono"]);
         rows.push(["Files", (r.files || []).join(", ") || null, "mono"]);
         rows.push(["Feedback rows", r.feedback_ids]);
         if (r.trainer_problem) rows.push(["Trainer", r.trainer_problem]);
         var dl = document.getElementById("opsTrainExport");
         opsKv(dl, rows);
         dl.hidden = false;
         uiToast("Exported the " + (r.method || method) + " training set"
                 + (c.examples != null ? ": " + c.examples + " examples." : "."), "ok");
       });
     } finally {
       btn.disabled = false;
     }
   }

   // ------------------------------------------------- admin: cluster ---

   function opsNodeRow(n) {
     var tr = opsEl("tr");
     var top = opsEl("div", "row");
     top.appendChild(opsEl("strong", "", n.name));
     top.appendChild(opsBadge(n.role || "node"));
     if (n.is_local) top.appendChild(opsBadge("this Mac"));
     if (n.paired) top.appendChild(opsBadge("paired"));
     var sub = [n.machine || "", n.cores ? n.cores + " cores" : "",
                n.protocol ? "protocol v" + n.protocol : "", n.url || ""].filter(Boolean).join(", ");
     opsCell(tr, opsTwoLine(top, sub));

     var kinds = { healthy: "ok", degraded: "warn", overloaded: "warn", starting: "running",
                   draining: "muted", unavailable: "error" };
     var why = [];
     if (n.consecutive_failures) why.push(n.consecutive_failures + " failure(s)");
     if (n.cooldown_remaining_s != null) why.push("retry in " + Math.ceil(n.cooldown_remaining_s) + " s");
     if (n.detail) why.push(n.detail);
     opsCell(tr, opsTwoLine(opsPill(n.state || "unknown", kinds[n.state] || "muted"), why.join(", ")));

     var ram = Number(n.ram_gb) || 0;
     var free = opsNum(n.free_ram_gb) ? n.free_ram_gb : null;
     var used = free != null && ram ? (ram - free) / ram * 100 : (opsNum(n.mem_pct) ? n.mem_pct : null);
     var cap = free != null ? free.toFixed(1) + " of " + ram + " GB free"
               : (ram ? ram + " GB" + (opsNum(n.mem_pct) ? ", " + Math.round(n.mem_pct) + "% used" : "") : "-");
     opsCell(tr, used == null ? cap : opsMeter(used, cap, used >= 90 ? "error" : (used >= 75 ? "warn" : "")));

     var load = [];
     load.push("latency " + (opsNum(n.last_latency_ms) ? Math.round(n.last_latency_ms) + " ms" : "-"));
     if (n.queue != null) load.push("queue " + n.queue);
     if (opsNum(n.cpu_pct)) load.push("cpu " + Math.round(n.cpu_pct) + "%");
     if (opsNum(n.load_ratio)) load.push("load " + n.load_ratio + "x per core");
     if (n.active) load.push(n.active + " active");
     opsCell(tr, opsTwoLine(load[0], load.slice(1).join(", ")));

     opsCell(tr, opsTwoLine(opsEl("span", "mono", n.model || "-"), n.backend ? "engine " + n.backend : ""));
     var caps = opsEl("div", "row");
     (n.capabilities || []).forEach(function(c) { caps.appendChild(opsBadge(c)); });
     if (n.shard_capable) caps.appendChild(opsBadge("shard", "ok"));
     opsCell(tr, caps);
     return tr;
   }

   async function opsLoadCluster() {
     var host = document.getElementById("clusterNodes");
     // The paired list rides the same 5 s poll, so a Mac that runs the join
     // command shows up here while the pairing code is still on screen.
     opsLoadPaired();
     var d;
     try {
       d = (await fetchJSON("/api/cluster/nodes")).data || {};
     } catch (err) {
       opsClear(host).appendChild(opsEl("p", "hint warn", "Could not load the cluster: " + err.message));
       return;
     }
     var nodes = d.nodes || [];
     document.getElementById("opsClusterTopology").textContent =
       (d.multi_node ? "Multi-node: " + nodes.length + " nodes." : "Single node.")
       + " This machine is the " + (d.node_role || "primary") + ".";
     opsClear(host);
     if (!nodes.length) {
       host.appendChild(opsEl("p", "muted", "No nodes registered."));
     } else {
       var wrap = opsEl("div", "ops-scroll");
       var table = opsEl("table", "table");
       var thead = opsEl("thead");
       var hr = opsEl("tr");
       ["Node", "State", "Memory", "Load", "Model", "Capabilities"].forEach(function(h) {
         hr.appendChild(opsEl("th", "", h));
       });
       thead.appendChild(hr);
       table.appendChild(thead);
       var tbody = opsEl("tbody");
       nodes.forEach(function(n) { tbody.appendChild(opsNodeRow(n)); });
       table.appendChild(tbody);
       wrap.appendChild(table);
       host.appendChild(wrap);
     }
     opsRenderShard(nodes);
     opsRenderRoutingSummary(d.routing_summary);
     return opsLoadRouting();
   }

   function opsRenderShard(nodes) {
     var host = opsClear(document.getElementById("opsShard"));
     var group = [];
     nodes.forEach(function(n) {
       (n.shard_group || []).forEach(function(h) { if (group.indexOf(h) < 0) group.push(h); });
     });
     if (group.length) host.appendChild(opsEl("p", "mt-0", "Shard group, in hostfile order: " + group.join(", ")));
     var capable = nodes.filter(function(n) { return n.shard_capable; });
     if (!capable.length) {
       host.appendChild(opsEl("p", "muted mt-0", "No node reports a shard-capable engine (mlx-distributed)."));
       return;
     }
     var total = 0;
     var free = 0;
     capable.forEach(function(n) {
       total += Number(n.ram_gb) || 0;
       free += opsNum(n.free_ram_gb) ? n.free_ram_gb : 0;
     });
     host.appendChild(opsEl("p", "mt-0", capable.length + " shard-capable node(s): "
       + capable.map(function(n) { return n.name; }).join(", ") + ". "
       + (free ? free.toFixed(1) + " GB free of " : "") + total + " GB across them."));
     var box = opsEl("div", "");
     host.appendChild(box);
     uiChart(box, {
       type: "bar", labels: capable.map(function(n) { return n.name; }),
       series: [{ label: "free", values: capable.map(function(n) { return opsNum(n.free_ram_gb) ? n.free_ram_gb : null; }) }],
       unit: "GB", title: "Free memory per shard-capable node", empty: "No node reports its free memory yet."
     });
   }

   function opsRenderRoutingSummary(summary) {
     var sum = document.getElementById("routingSummary");
     var byNode = (summary && summary.by_node) || {};
     var parts = Object.keys(byNode).map(function(node) {
       var st = byNode[node] || {};
       return node + " (" + Object.keys(st).map(function(k) { return k + " " + st[k]; }).join(", ") + ")";
     });
     sum.textContent = parts.length ? "Routing totals: " + parts.join("; ") : "";
   }

   async function opsLoadRouting() {
     var box = document.getElementById("routingEvents");
     try {
       var events = ((await fetchJSON("/api/routing/events?limit=40")).data || {}).events || [];
       if (!events.length) { box.textContent = "No routing decisions recorded yet."; return; }
       box.textContent = events.map(function(e) {
         return (e.created_at || "") + "  " + (e.selected_node || "?")
                + "  [" + (e.status || "") + "]  " + (e.kind || "")
                + (e.requested_model ? "  " + e.requested_model : "")
                + (e.reason ? "  : " + e.reason : "");
       }).join("\n");
     } catch (err) {
       box.textContent = "Could not load routing events: " + err.message;
     }
   }

   // -------------------------------------------------- admin: pairing ---

   function opsPairReset() {
     if (opsPairTimer) { clearInterval(opsPairTimer); opsPairTimer = null; }
     document.getElementById("opsPairBox").hidden = true;
     document.getElementById("opsPairCode").textContent = "";
     document.getElementById("opsPairCmd").textContent = "";
   }

   function opsPairTick(expiresAt) {
     var pill = document.getElementById("opsPairExpiry");
     var left = Math.round(expiresAt - Date.now() / 1000);
     if (left <= 0) {
       pill.className = "pill error";
       pill.textContent = "expired";
       if (opsPairTimer) { clearInterval(opsPairTimer); opsPairTimer = null; }
       return;
     }
     var m = Math.floor(left / 60);
     var s = left % 60;
     pill.className = "pill " + (left < 60 ? "warn" : "muted");
     pill.textContent = "expires in " + m + ":" + (s < 10 ? "0" : "") + s;
   }

   async function opsCreatePairing() {
     await uiAct("Could not create a pairing code", async function() {
       var r = (await postJSON("/api/cluster/pairing", {})).data || {};
       if (!r.code) throw new Error("the server returned no code");
       opsPairReset();
       var origin = window.location.origin;
       var join = String(r.join || "deploy.py cluster join <this-url> --code <code> --url <node-url>")
         .replace("<this-url>", origin).replace("<code>", r.code);
       document.getElementById("opsPairCode").textContent = r.code;
       document.getElementById("opsPairCmd").textContent = "python " + join;
       document.getElementById("opsPairBox").hidden = false;
       var expires = Number(r.expires_at) || (Date.now() / 1000 + (Number(r.ttl_s) || 600));
       opsPairTick(expires);
       opsPairTimer = setInterval(function() { opsPairTick(expires); }, 1000);
       if (opsIsLoopback(window.location.hostname)) {
         uiToast("This page is open on a loopback address: replace " + origin
                 + " in the command with an address the new Mac can reach.", "warn");
       }
       opsLoadPaired();
     });
   }

   function opsCopyPairCmd() {
     var text = document.getElementById("opsPairCmd").textContent;
     if (!text) return;
     copyText(text).then(function() { uiToast("Join command copied.", "ok"); },
                         function() { uiToast("Copy failed: select the command and copy it by hand.", "warn"); });
   }

   async function opsLoadPaired() {
     var table = document.getElementById("opsPairedTable");
     var hint = document.getElementById("opsPairedHint");
     var d;
     try {
       d = (await fetchJSON("/api/cluster/paired")).data || {};
     } catch (err) {
       opsEmptyRow(table, 5, "Could not load paired nodes: " + err.message);
       return;
     }
     var notes = [];
     if (d.pending_codes) notes.push(d.pending_codes + " unused pairing code(s) still valid.");
     if (d.cluster_nodes_env) notes.push("CLUSTER_NODES is also set: those nodes appear in the table above, not here.");
     hint.textContent = notes.join(" ");
     var nodes = d.nodes || [];
     if (!nodes.length) { opsEmptyRow(table, 5, "No node has paired with this primary."); return; }
     opsClear(table);
     nodes.forEach(function(n) {
       var tr = opsEl("tr");
       opsCell(tr, opsTwoLine(opsEl("strong", "", n.name), n.url || ""));
       opsCell(tr, opsNum(n.ram_gb) && n.ram_gb ? n.ram_gb + " GB" : "-", "nowrap");
       opsCell(tr, n.joined_at ? relativeTime(n.joined_at) : "-", "nowrap");
       opsCell(tr, opsPill(n.active ? "serving" : "restart pending", n.active ? "ok" : "warn"), "cell-status");
       opsCell(tr, opsBtn("Unpair", "btn btn-sm btn-danger", function() { opsUnpair(n.name); },
                          "Remove " + n.name + " from the paired nodes"), "cell-action");
       table.appendChild(tr);
     });
   }

   async function opsUnpair(name) {
     if (!uiConfirm("Unpair " + name + "? It is removed from the paired nodes, and the primary "
                    + "stops routing to it after its next restart.")) return;
     await uiAct("Unpair failed", async function() {
       await fetchJSON("/api/cluster/paired/" + encodeURIComponent(name), { method: "DELETE" });
       uiToast(name + " unpaired. Restart the primary for it to take effect.", "ok");
       opsLoadPaired();
     });
   }

   // ------------------------------------------- admin: remote access ---

   function opsCheck(host, kind, text) {
     var row = opsEl("div", "row nowrap");
     row.appendChild(opsPill(kind === "ok" ? "ok" : (kind === "error" ? "problem" : "check"), kind));
     row.appendChild(opsEl("span", "hint grow", text));
     host.appendChild(row);
   }

   function opsCommand(host, cmd, what) {
     var box = opsEl("div", "stack tight");
     box.appendChild(opsEl("span", "hint", what));
     var row = opsEl("div", "row nowrap");
     row.appendChild(opsEl("div", "readonly mono grow", cmd));
     row.appendChild(opsBtn("Copy", "btn btn-sm", function() {
       copyText(cmd).then(function() { uiToast("Command copied.", "ok"); },
                          function() { uiToast("Copy failed: select the command and copy it by hand.", "warn"); });
     }, "Copy this command"));
     box.appendChild(row);
     host.appendChild(box);
   }

   async function opsLoadRemote() {
     var pills = document.getElementById("opsRemotePills");
     var c;
     try {
       c = ((await fetchJSON("/api/config")).data || {}).config || {};
     } catch (err) {
       opsClear(pills).appendChild(opsEl("span", "hint warn", "Could not read the configuration: " + err.message));
       return;
     }
     var host = String(c.web_host || "127.0.0.1");
     var loop = opsIsLoopback(host) || host === "";
     var base = String(c.public_base_url || "").trim();
     var https = base.toLowerCase().indexOf("https://") === 0;
     var proxies = String(c.trusted_proxies || "").trim();
     var auth = !!c.auth_enabled;
     var secure = !!c.auth_cookie_secure;
     var e2e = !!c.node_e2e_key;
     var tls = [c.node_ca_file ? "CA " + c.node_ca_file : "",
                c.node_client_cert ? "client certificate " + c.node_client_cert : ""].filter(Boolean).join(", ");
     var remoteNodes = !!(String(c.cluster_nodes || "").trim() || String(c.studio_node_url || "").trim());

     opsClear(pills);
     pills.appendChild(loop ? opsPill("loopback only", "ok") : opsPill("listening on " + host, "warn"));
     pills.appendChild(auth ? opsPill("sign-in on", "ok") : opsPill("sign-in off", loop && !base ? "muted" : "error"));
     pills.appendChild(secure ? opsPill("secure cookie", "ok") : opsPill("plain-HTTP cookie", loop && !base ? "muted" : "warn"));
     if (base) pills.appendChild(opsPill(https ? "public URL" : "public URL not HTTPS", https ? "ok" : "warn"));
     if (proxies) pills.appendChild(opsPill("trusted proxies", "ok"));
     if (tls) pills.appendChild(opsPill("node TLS", "ok"));
     if (e2e) pills.appendChild(opsPill("node E2E", "ok"));

     opsKv(document.getElementById("opsRemoteFacts"), [
       ["Listens on", host + ":" + (c.web_port || ""), "mono"],
       ["Public URL", base || "not set: links and the sign-in redirect use the request's own origin"],
       ["Trusted proxies", proxies || "none: X-Forwarded-For is ignored"],
       ["Sign-in", auth ? (c.oidc_enabled ? "on, with Entra ID" : "on, local accounts") : "off: every visitor is the local admin"],
       ["Session cookie", secure ? "Secure: sent over HTTPS only" : "also sent over plain HTTP"],
       ["Node token", c.node_token ? "set" : "not set"],
       ["Node E2E key", e2e ? "set: node payloads are sealed end to end" : "not set"],
       ["Node TLS", tls || "system trust store, no client certificate"],
       ["Plain HTTP to nodes", c.node_allow_plaintext ? "allowed to any host (NODE_ALLOW_PLAINTEXT=1)"
                                                      : "only to loopback, private and tailnet hosts"]
     ]);

     var checks = opsClear(document.getElementById("opsRemoteChecks"));
     var issues = 0;
     if (!loop && !auth) {
       opsCheck(checks, "error", "WEB_HOST=" + host + " with AUTH_ENABLED=0: the server refuses to start like "
                + "this, because every visitor would be an unauthenticated admin.");
       issues++;
     }
     if (!loop && auth && !secure) {
       opsCheck(checks, "warn", "Listening on " + host + " with AUTH_COOKIE_SECURE=0: the session cookie travels "
                + "over plain HTTP. Put TLS in front and set AUTH_COOKIE_SECURE=1.");
       issues++;
     }
     if (base && !auth) {
       opsCheck(checks, "error", "PUBLIC_BASE_URL is set while AUTH_ENABLED=0: anyone who reaches that URL is an admin.");
       issues++;
     }
     if (base && !https) {
       opsCheck(checks, "warn", "PUBLIC_BASE_URL is not https://: browsers send the password and the session in the clear.");
       issues++;
     }
     if (base && https && !secure) {
       opsCheck(checks, "warn", "Served over HTTPS with AUTH_COOKIE_SECURE=0: set it to 1 so the cookie never leaves over HTTP.");
       issues++;
     }
     if (base && !proxies) {
       opsCheck(checks, "warn", "PUBLIC_BASE_URL is set but TRUSTED_PROXIES is empty: behind a proxy or tunnel every "
                + "request arrives from 127.0.0.1, so the login throttle treats all users as one client.");
       issues++;
     }
     if (proxies && !base && c.oidc_enabled) {
       opsCheck(checks, "warn", "TRUSTED_PROXIES is set but PUBLIC_BASE_URL is empty: the Entra ID redirect is built "
                + "from the internal origin.");
       issues++;
     }
     if (remoteNodes && !e2e && !tls) {
       opsCheck(checks, "warn", "Remote nodes are configured with neither TLS settings nor NODE_E2E_KEY: prompts to them "
                + "are protected only if their URLs are https://.");
       issues++;
     }
     if (!issues) opsCheck(checks, "ok", "No problem found in the current configuration.");

     var hostname = "chat.example.com";
     if (base) {
       try { hostname = new URL(base).hostname || hostname; } catch (err) { hostname = "chat.example.com"; }
     }
     var cmds = opsClear(document.getElementById("opsRemoteCmds"));
     opsCommand(cmds, "python deploy.py remote status", "What is bound, proxied and secured, and which tools are installed.");
     opsCommand(cmds, "python deploy.py remote tailscale", "Publish on your tailnet with Tailscale Serve (HTTPS, tailnet only).");
     opsCommand(cmds, "python deploy.py remote funnel", "Publish to the internet with Tailscale Funnel.");
     opsCommand(cmds, "python deploy.py remote cloudflare --hostname " + hostname,
                "Publish through a Cloudflare Tunnel on your own hostname.");
     opsCommand(cmds, "python deploy.py remote gen-key", "Print a fresh NODE_TOKEN and NODE_E2E_KEY for the cluster.");
   }
"""

__all__ = ["UI_JS_OPS"]
