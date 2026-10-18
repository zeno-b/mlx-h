"""Front-end: the admin and Models & Training panels inside Settings.

These functions used to live in ui_script_panels.py behind a separate admin
"Models" view whose opening fired every loader at once (models, dataset,
documents, project, users, agents, cluster) whether or not the panel was ever
looked at, and whose results came back through alert(). They now serve the
sub-tabs of Settings > Models & Training and Settings > Admin, and the
Performance / Memory / Tools sections:

  * ADMIN_TABS is the lazy-load registry: `{key, load, poll: [fn, ms]}` per
    section or "section/sub". adminTabShown(key) runs the loader when that tab
    becomes visible and starts its poll; adminTabStop() ends the poll when the
    tab is hidden or the view changes, so nothing polls off screen.
  * Results and failures are toasts (uiToast), never alert() and never lines
    in the chat transcript.
  * Element ids are unchanged (ui_settings.py renders the same ids), so the
    markup moved without the loaders noticing.

Admin routes are enforced server-side; gating here is cosmetic.
"""

from __future__ import annotations

UI_JS_ADMIN = r"""   // ------------------------------------------------------- admin tabs ---

   // Lazy-load registry. key is "section" or "section/sub" of Settings; load
   // runs each time that tab is shown; poll [fn, ms] repeats while it stays
   // visible. Loaders are called through closures so a part of the page that
   // is missing a loader fails one tab, not the whole script.
   var ADMIN_TABS = [
     { key: "models/model", load: function() { return loadModels(); } },
     { key: "models/training", load: function() { return loadDatasetStats(); } },
     { key: "models/knowledge", load: function() { return loadDocsStats(); } },
     { key: "performance", load: function() { return loadPerf(); },
       poll: [function() { return loadPerf(); }, 10000] },
     { key: "memory", load: function() { return loadMemory(); } },
     { key: "tools", load: function() { return loadToolsTab(); } },
     { key: "admin/users", load: function() { return loadUsers(); } },
     { key: "admin/agents", load: function() { return loadAgents(); } },
     { key: "admin/cluster", load: function() { return loadCluster(); },
       poll: [function() { return loadCluster(); }, 5000] },
     { key: "admin/import", load: function() { return loadImports(); } },
     { key: "admin/codebase", load: function() { return loadProjectStatus(); } },
     { key: "admin/logs", load: function() { return loadModelLog(); },
       poll: [function() { return loadModelLog(); }, 3000] }
   ];
   var _adminTabKey = null;
   var _adminPollTimer = null;

   function adminTabEntry(key) {
     for (var i = 0; i < ADMIN_TABS.length; i++) {
       if (ADMIN_TABS[i].key === key) return ADMIN_TABS[i];
     }
     return null;
   }

   function adminRunSafely(fn) {
     try {
       var result = fn();
       if (result && typeof result.catch === "function") {
         result.catch(function(err) { if (window.console) console.warn("settings loader:", err); });
       }
     } catch (err) {
       if (window.console) console.warn("settings loader:", err);
     }
   }

   // A Settings tab became visible: run its loader, start its poll.
   function adminTabShown(key) {
     adminTabStop();
     _adminTabKey = key;
     var entry = adminTabEntry(key);
     if (!entry) return;
     if (entry.load) adminRunSafely(entry.load);
     if (entry.poll && entry.poll[0]) {
       var fn = entry.poll[0];
       _adminPollTimer = setInterval(function() {
         // A background tab does not need a live log or cluster view.
         if (document.hidden) return;
         adminRunSafely(fn);
       }, Math.max(1000, entry.poll[1] || 5000));
     }
   }

   function adminTabStop() {
     if (_adminPollTimer) { clearInterval(_adminPollTimer); _adminPollTimer = null; }
     _adminTabKey = null;
   }

   // Old entry points (palette commands, buttons in other panels, bookmarks)
   // named the tabs of the retired Models view. Map them onto Settings.
   function adminSectionFor(name) {
     var models = { models: "model", model: "model", data: "training", training: "training",
                    knowledge: "knowledge" };
     if (models[name]) return ["models", models[name]];
     return ["admin", name || "users"];
   }

   function showAdminSection(name) {
     var target = adminSectionFor(name);
     openSettings(target[0], target[1]);
   }

   function showAdminTab(name) {
     showAdminSection(name);
   }

   // ---------------------------------------------------------- backup ---

   async function downloadBackup() {
     // Blob download (uiDownload), so a 403/500 is a toast rather than the
     // whole app replaced by an error body.
     var ok = await uiDownload("/api/backup", "backup-" + Date.now() + ".json");
     if (ok) uiToast("Backup downloaded.", "ok");
   }

   async function restoreBackup(file) {
     if (!file) return;
     await uiAct("Restore failed", async function() {
       var text = await file.text();
       var data;
       try { data = JSON.parse(text); } catch (err) { throw new Error("that file is not a JSON backup"); }
       var r = await postJSON("/api/backup/restore", data);
       var d = (r.data || {}).restored || {};
       uiToast("Restored " + (d.messages || 0) + " messages, " + (d.prompts || 0) + " prompts, "
               + (d.feedback || 0) + " feedback rows, " + (d.documents || 0) + " documents.", "ok");
       if (typeof loadHistory === "function") loadHistory();
     });
   }

   // Idempotent: the restore picker must fire restoreBackup once per file even
   // if more than one boot path wires it.
   function wireRestoreInput() {
     var input = document.getElementById("restoreInput");
     if (!input || input.getAttribute("data-wired") === "1") return;
     input.setAttribute("data-wired", "1");
     input.addEventListener("change", function() {
       if (input.files && input.files[0]) restoreBackup(input.files[0]);
       input.value = "";
     });
   }

   // ----------------------------------------------------------- models ---

   // Loads the Model sub-tab only. It used to fan out to every admin panel;
   // each tab now loads itself when shown (ADMIN_TABS).
   async function loadModels() {
     var currentBox = document.getElementById("modelCurrent");
     try {
       var out = await fetchJSON("/api/models");
       var current = out.data.current;
       currentBox.textContent =
         "model    : " + current.model + (current.cached ? "  (cached)" : "  (not downloaded)") +
         "\nadapter  : " + current.adapter + (current.adapter_path ? "  " + current.adapter_path : "") +
         "\nkv cache : " + (current.max_kv_size || "unbounded") +
         "\nstatus   : " + current.status +
         "\ncache dir: " + out.data.cache_dir;

       var table = document.getElementById("modelTable");
       table.innerHTML = "";
       (out.data.catalog || []).forEach(function(item) {
         var row = document.createElement("tr");
         var id = document.createElement("td");
         id.className = "id mono";
         id.textContent = item.id;
         var state = document.createElement("td");
         state.className = "cell-status";
         var pill = statusPill(item.current ? "running" : (item.cached ? "ok" : "muted"));
         pill.textContent = item.current ? "in use" : (item.cached ? "cached" : "download");
         state.appendChild(pill);
         var action = document.createElement("td");
         action.className = "cell-action";
         var button = document.createElement("button");
         button.type = "button";
         button.className = "btn btn-sm";
         button.textContent = "Use";
         button.disabled = !!item.current;
         button.onclick = function() { useModel(item.id); };
         action.appendChild(button);
         row.appendChild(id);
         row.appendChild(state);
         row.appendChild(action);
         table.appendChild(row);
       });

       var select = document.getElementById("adapterSelect");
       select.innerHTML = "";
       (out.data.adapters || []).forEach(function(item) {
         var option = document.createElement("option");
         option.value = item.id;
         option.textContent = item.id + (item.modified ? "  (" + relativeTime(item.modified) + ")" : "");
         select.appendChild(option);
       });
       // Only select an adapter that is actually offered. `current.adapter` is the
       // REQUESTED choice (often "latest"), which is not in the list on an install
       // that has never trained one -- assigning it left the dropdown blank and a
       // subsequent Apply submitted an empty adapter.
       var wanted = current.adapter;
       var offered = Array.prototype.map.call(select.options, function(o) { return o.value; });
       if (offered.indexOf(wanted) < 0) {
         var opt = document.createElement("option");
         opt.value = wanted;
         opt.textContent = wanted + "  (not built yet)";
         select.insertBefore(opt, select.firstChild);
       }
       select.value = wanted;
       document.getElementById("kvSize").value = current.max_kv_size || 0;
     } catch (err) {
       if (currentBox) currentBox.textContent = "Could not load models: " + err.message;
     }
   }

   async function postModel(body) {
     await uiAct("Could not switch", async function() {
       var out = await postJSON("/api/models/select", body);
       if (out.data.error) { uiToast(out.data.error, "error"); return; }
       uiToast("Model server restarting. Admin > Logs shows its progress.", "ok");
       loadModels();
       loadModelLog();
     });
   }

   function useModel(modelId) {
     if (!modelId) { uiToast("Enter a Hugging Face repo id first.", "warn"); return; }
     if (!uiConfirm("Switch to " + modelId + "? The model server restarts, " +
                    "and an uncached model downloads first.")) return;
     postModel({ model: modelId, restart: true });
   }

   function applyModel() {
     postModel({
       adapter: document.getElementById("adapterSelect").value,
       max_kv_size: Number(document.getElementById("kvSize").value) || 0,
       restart: true
     });
   }

   async function loadModelLog() {
     var box = document.getElementById("modelLog");
     if (!box) return;
     try {
       var out = await fetchJSON("/api/logs/model?lines=120");
       // Follow the tail only when the reader is already at the bottom, so a
       // poll does not yank the view away from the line being read.
       var atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 8;
       box.textContent = out.data.text || "(no output yet)";
       if (atBottom) box.scrollTop = box.scrollHeight;
     } catch (err) {
       box.textContent = "Log unavailable: " + err.message;
     }
   }

   // ------------------------------------------------------------ tools ---

   function loadToolsTab() {
     loadToolsCatalogue();
     if (document.body.getAttribute("data-role") === "admin") loadMcpStatus();
   }

   async function loadToolsCatalogue() {
     var host = document.getElementById("toolsList");
     if (!host) return;
     try {
       var out = await fetchJSON("/api/tools");
       var tools = out.data.tools || [];
       host.innerHTML = "";
       if (!tools.length) { host.textContent = "No tools are available."; return; }
       var table = document.createElement("table");
       table.className = "table";
       var body = document.createElement("tbody");
       tools.forEach(function(t) {
         var row = document.createElement("tr");
         var name = document.createElement("td");
         name.className = "mono";
         name.textContent = t.name;
         var desc = document.createElement("td");
         desc.className = "muted";
         desc.textContent = t.description || "";
         row.appendChild(name);
         row.appendChild(desc);
         body.appendChild(row);
       });
       table.appendChild(body);
       host.appendChild(table);
     } catch (err) {
       host.textContent = "Could not load tools: " + err.message;
     }
   }

   async function loadMcpStatus() {
     var host = document.getElementById("mcpStatus");
     if (!host) return;
     try {
       var out = await fetchJSON("/api/mcp");
       var servers = out.data.servers || [];
       host.innerHTML = "";
       if (!servers.length) {
         host.textContent = out.data.configured ? "Configured, but no server is running."
                                                : "No MCP servers configured.";
         return;
       }
       servers.forEach(function(s) {
         var row = document.createElement("div");
         row.className = "row list-row";
         var name = document.createElement("span");
         name.className = "mono grow";
         name.textContent = s.name + "  (" + ((s.tools || []).length) + " tools)";
         var pill = statusPill(s.alive ? "ok" : "error");
         pill.textContent = s.alive ? "running" : "stopped";
         row.appendChild(name);
         row.appendChild(pill);
         host.appendChild(row);
         if (s.error) {
           var err = document.createElement("p");
           err.className = "hint error";
           err.textContent = s.error;
           host.appendChild(err);
         }
       });
     } catch (err) {
       host.textContent = "Could not load MCP status: " + err.message;
     }
   }

   // ------------------------------------------------------- user admin ---

   async function loadUsers() {
     var host = document.getElementById("usersList");
     if (!host) return;
     try {
       var out = await fetchJSON("/api/users");
       var users = out.data.users || [];
       host.innerHTML = "";
       if (!users.length) { host.textContent = "No users yet."; return; }
       users.forEach(function(u) {
         var row = document.createElement("div");
         row.className = "row list-row";
         var who = document.createElement("div");
         who.className = "grow";
         who.textContent = u.username +
           (u.display_name && u.display_name !== u.username ? " (" + u.display_name + ")" : "") +
           (u.source && u.source !== "local" ? "  · " + u.source : "");
         var pill = statusPill(u.disabled ? "muted" : "ok");
         pill.textContent = u.role + (u.disabled ? " · disabled" : "");
         var actions = document.createElement("div");
         actions.className = "row card-actions";
         var roleBtn = document.createElement("button");
         roleBtn.type = "button";
         roleBtn.className = "btn btn-sm";
         roleBtn.textContent = u.role === "admin" ? "Make user" : "Make admin";
         roleBtn.onclick = function() { updateUser(u.id, { role: u.role === "admin" ? "user" : "admin" }); };
         var disBtn = document.createElement("button");
         disBtn.type = "button";
         disBtn.className = "btn btn-sm";
         disBtn.textContent = u.disabled ? "Enable" : "Disable";
         disBtn.onclick = function() { updateUser(u.id, { disabled: !u.disabled }); };
         var delBtn = document.createElement("button");
         delBtn.type = "button";
         delBtn.className = "btn btn-sm btn-danger";
         delBtn.textContent = "Delete";
         delBtn.onclick = function() {
           if (uiConfirm("Delete user " + u.username + "? Their data stays but they can "
               + "no longer sign in.")) deleteUser(u.id);
         };
         actions.appendChild(roleBtn);
         actions.appendChild(disBtn);
         actions.appendChild(delBtn);
         row.appendChild(who);
         row.appendChild(pill);
         row.appendChild(actions);
         host.appendChild(row);
       });
     } catch (err) {
       host.textContent = "Could not load users: " + err.message;
     }
   }

   async function createUser() {
     var name = (document.getElementById("newUserName").value || "").trim();
     var pass = document.getElementById("newUserPass").value || "";
     var role = document.getElementById("newUserRole").value || "user";
     if (!name || !pass) { uiToast("Enter a username and a password for the new user.", "warn"); return; }
     await uiAct("Could not create user", async function() {
       await postJSON("/api/users", { username: name, password: pass, role: role });
       document.getElementById("newUserName").value = "";
       document.getElementById("newUserPass").value = "";
       uiToast("User " + name + " created.", "ok");
       loadUsers();
     });
   }

   async function updateUser(id, patch) {
     await uiAct("Could not update user", async function() {
       await postJSON("/api/users/" + encodeURIComponent(id), patch);
       loadUsers();
     });
   }

   async function deleteUser(id) {
     await uiAct("Could not delete user", async function() {
       await fetchJSON("/api/users/" + encodeURIComponent(id), { method: "DELETE" });
       uiToast("User deleted.", "ok");
       loadUsers();
     });
   }

   // --------------------------------------------------- cluster / routing ---

   async function loadCluster() {
     var host = document.getElementById("clusterNodes");
     if (host) {
       try {
         var out = await fetchJSON("/api/cluster/nodes");
         var d = out.data;
         var nodes = d.nodes || [];
         var lines = [];
         lines.push("Topology: " + (d.multi_node ? "multi-node" : "single node")
                    + "   (this node's role: " + (d.node_role || "primary") + ")");
         nodes.forEach(function(n) {
           lines.push("");
           lines.push(n.name + "  [" + n.role + "]  " + n.state + (n.is_local ? "  (local)" : ""));
           if (n.machine || n.ram_gb) {
             lines.push("  hardware " + (n.machine || "?")
                        + (n.cores ? "  " + n.cores + " cores" : "")
                        + (n.ram_gb ? "  " + n.ram_gb + "GB RAM" : ""));
           }
           lines.push("  active " + n.active
                      + "   latency " + (n.last_latency_ms != null ? n.last_latency_ms + "ms" : "-")
                      + "   load " + (n.load_ratio != null ? n.load_ratio + "x/core" : "-")
                      + "   cpu " + (n.cpu_pct != null ? n.cpu_pct + "%" : "n/a")
                      + "   mem " + (n.mem_pct != null ? n.mem_pct + "%" : "n/a"));
           lines.push("  caps " + ((n.capabilities || []).join(", ") || "-")
                      + (n.consecutive_failures ? "   failures " + n.consecutive_failures : "")
                      + (n.cooldown_remaining_s != null ? "   cooldown " + n.cooldown_remaining_s + "s" : ""));
           if (n.model) lines.push("  model " + n.model);
         });
         host.textContent = lines.join("\n");
         var sum = document.getElementById("routingSummary");
         if (sum) {
           var byNode = (d.routing_summary && d.routing_summary.by_node) || {};
           var parts = Object.keys(byNode).map(function(node) {
             var st = byNode[node];
             var inner = Object.keys(st).map(function(k) { return k + " " + st[k]; }).join(", ");
             return node + " (" + inner + ")";
           });
           sum.textContent = parts.length ? ("Routing totals: " + parts.join("; ")) : "";
         }
       } catch (err) {
         host.textContent = "Could not load cluster: " + err.message;
       }
     }
     var evbox = document.getElementById("routingEvents");
     if (!evbox) return;
     try {
       var eo = await fetchJSON("/api/routing/events?limit=40");
       var events = eo.data.events || [];
       if (!events.length) { evbox.textContent = "No routing decisions recorded yet."; return; }
       evbox.textContent = events.map(function(e) {
         return (e.created_at || "") + "  " + (e.selected_node || "?")
                + "  [" + (e.status || "") + "]  " + (e.kind || "")
                + (e.requested_model ? "  " + e.requested_model : "")
                + (e.reason ? "  : " + e.reason : "");
       }).join("\n");
     } catch (err) {
       evbox.textContent = "Could not load routing events: " + err.message;
     }
   }

   // -------------------------------------------------------- agents ------ #

   var _agentCaps = [];
   var _agentRuntime = {};

   async function loadAgentCaps() {
     try {
       var out = await fetchJSON("/api/agents/capabilities");
       _agentCaps = out.data.capabilities || [];
       _agentRuntime = out.data.runtime || {};
     } catch (err) { _agentCaps = []; _agentRuntime = {}; }
   }

   // One styled checkbox row (.check), shared by the capability grid and the
   // agent run picker so both look and behave the same.
   function adminCheckRow(value, labelText, descText, checked) {
     var row = document.createElement("label");
     row.className = "check cap-item";
     var cb = document.createElement("input");
     cb.type = "checkbox";
     cb.value = value;
     cb.checked = !!checked;
     var text = document.createElement("div");
     text.className = "cap-text";
     var lab = document.createElement("div");
     lab.className = "cap-label";
     lab.textContent = labelText;
     var desc = document.createElement("div");
     desc.className = "cap-desc muted";
     desc.textContent = descText || "";
     text.appendChild(lab);
     text.appendChild(desc);
     row.appendChild(cb);
     row.appendChild(text);
     return { row: row, box: cb };
   }

   // Render the capability checkboxes into `host`, pre-checking `selected` keys.
   // Returns a reader for the currently-checked capability keys.
   function renderCaps(host, selected) {
     host.innerHTML = "";
     var sel = {};
     (selected || []).forEach(function(k) { sel[k] = true; });
     var boxes = [];
     if (!_agentCaps.length) {
       // The catalogue failed to load; say so instead of showing an empty grid
       // that looks like "this agent has no capabilities to choose from".
       host.textContent = "Could not load the capability list. Refresh to try again.";
       return function() { return (selected || []).slice(); };
     }
     _agentCaps.forEach(function(c) {
       var note = "";
       if (c.key === "code_exec" && !(_agentRuntime.allow_shell || _agentRuntime.allow_python)) {
         note = " (server execution is off)";
       }
       if (c.key === "office365" && !_agentRuntime.office365_configured) {
         note = " (not connected)";
       }
       var item = adminCheckRow(c.key, c.label + note, c.description, sel[c.key]);
       host.appendChild(item.row);
       boxes.push(item.box);
     });
     return function() {
       return boxes.filter(function(b) { return b.checked; })
                   .map(function(b) { return b.value; });
     };
   }

   async function loadAgents() {
     if (!_agentCaps.length) await loadAgentCaps();
     var host = document.getElementById("agentsList");
     if (!host) return;
     try {
       var out = await fetchJSON("/api/agents");
       var agents = out.data.agents || [];
       host.innerHTML = "";
       if (!agents.length) {
         var empty = document.createElement("p");
         empty.className = "empty";
         empty.textContent = "No agents yet. Create one below.";
         host.appendChild(empty);
       }
       agents.forEach(function(a) {
         var card = document.createElement("div");
         card.className = "card agent-card";
         var head = document.createElement("div");
         head.className = "card-header agent-head";
         var name = document.createElement("input");
         name.type = "text";
         name.value = a.name;
         name.className = "input agent-name grow";
         name.setAttribute("aria-label", "Agent name");
         var sw = document.createElement("label");
         sw.className = "switch";
         sw.title = "Enabled";
         var enc = document.createElement("input");
         enc.type = "checkbox";
         enc.checked = !!a.enabled;
         enc.setAttribute("aria-label", "Enabled");
         var sl = document.createElement("span");
         sl.className = "slider";
         sw.appendChild(enc);
         sw.appendChild(sl);
         head.appendChild(name);
         head.appendChild(sw);
         var desc = document.createElement("input");
         desc.type = "text";
         desc.value = a.description || "";
         desc.placeholder = "description";
         desc.className = "input agent-desc";
         desc.setAttribute("aria-label", "Agent description");
         var caps = document.createElement("div");
         caps.className = "cap-grid";
         var readCaps = renderCaps(caps, a.capabilities);
         var actions = document.createElement("div");
         actions.className = "row card-actions";
         var save = document.createElement("button");
         save.type = "button";
         save.className = "btn btn-primary btn-sm";
         save.textContent = "Save";
         save.onclick = function() { saveAgent(a.id, name.value, desc.value, readCaps(), enc.checked); };
         var del = document.createElement("button");
         del.type = "button";
         del.className = "btn btn-danger btn-sm";
         del.textContent = "Delete";
         del.onclick = function() { if (uiConfirm("Delete agent " + a.name + "?")) deleteAgent(a.id); };
         actions.appendChild(save);
         actions.appendChild(del);
         card.appendChild(head);
         card.appendChild(desc);
         card.appendChild(caps);
         card.appendChild(actions);
         host.appendChild(card);
       });
       var np = document.getElementById("newAgentCaps");
       if (np) window._newAgentReadCaps = renderCaps(np, []);
       var pick = document.getElementById("agentRunPick");
       if (pick) {
         pick.innerHTML = "";
         window._agentRunBoxes = [];
         var enabled = agents.filter(function(a) { return a.enabled; });
         if (!enabled.length) { pick.textContent = "Enable at least one agent to run."; return; }
         enabled.forEach(function(a) {
           // Show the same human labels as the capability grid above, not the
           // internal keys (file_ops, web_api, ...).
           var labels = (a.capabilities || []).map(function(k) {
             for (var i = 0; i < _agentCaps.length; i++) {
               if (_agentCaps[i].key === k) return _agentCaps[i].label;
             }
             return k;
           });
           var item = adminCheckRow(a.id, a.name, labels.join(", ") || "no capabilities", false);
           pick.appendChild(item.row);
           window._agentRunBoxes.push(item.box);
         });
       }
     } catch (err) {
       host.textContent = "Could not load agents: " + err.message;
     }
   }

   async function createAgent() {
     var name = (document.getElementById("newAgentName").value || "").trim();
     if (!name) { uiToast("Enter a name for the agent.", "warn"); return; }
     var desc = (document.getElementById("newAgentDesc").value || "").trim();
     var caps = window._newAgentReadCaps ? window._newAgentReadCaps() : [];
     await uiAct("Could not create agent", async function() {
       await postJSON("/api/agents", { name: name, description: desc, capabilities: caps });
       document.getElementById("newAgentName").value = "";
       document.getElementById("newAgentDesc").value = "";
       uiToast("Agent " + name + " created.", "ok");
       loadAgents();
     });
   }

   async function saveAgent(id, name, description, capabilities, enabled) {
     await uiAct("Could not save agent", async function() {
       await postJSON("/api/agents/" + encodeURIComponent(id),
                      { name: name, description: description,
                        capabilities: capabilities, enabled: enabled });
       uiToast("Agent saved.", "ok");
       loadAgents();
     });
   }

   async function deleteAgent(id) {
     await uiAct("Could not delete agent", async function() {
       await fetchJSON("/api/agents/" + encodeURIComponent(id), { method: "DELETE" });
       loadAgents();
     });
   }

   async function runAgents() {
     var ids = (window._agentRunBoxes || []).filter(function(b) { return b.checked; })
                 .map(function(b) { return b.value; });
     var prompt = (document.getElementById("agentRunPrompt").value || "").trim();
     var out = document.getElementById("agentRunResult");
     if (!ids.length) { uiToast("Select at least one agent to run.", "warn"); return; }
     if (!prompt) { uiToast("Enter a task for the agents.", "warn"); return; }
     out.textContent = "Running " + ids.length + " agent(s)... this can take a while.";
     try {
       var r = await postJSON("/api/agents/run", { prompt: prompt, agent_ids: ids });
       var d = r.data || {};
       out.innerHTML = "";
       if (d.skipped) {
         var warn = document.createElement("p");
         warn.className = "hint warn";
         warn.textContent = d.skipped + " selected agent(s) were not run: at most "
           + (d.max_agents || "a few") + " run together.";
         out.appendChild(warn);
       }
       if (d.merged) {
         var m = document.createElement("div");
         m.className = "card agent-merged";
         var h = document.createElement("div");
         h.className = "cap-label";
         h.textContent = "Merged answer";
         var b = document.createElement("div");
         b.className = "agent-answer";
         renderMarkdown(d.merged, b);
         m.appendChild(h);
         m.appendChild(b);
         out.appendChild(m);
       }
       (d.results || []).forEach(function(res) {
         var box = document.createElement("details");
         box.className = "agent-result";
         var sum = document.createElement("summary");
         sum.textContent = res.name + ": " + ((res.tools_used || []).join(", ") || "no tools used");
         var body = document.createElement("div");
         body.className = "agent-answer";
         renderMarkdown(res.answer || "(no answer)", body);
         box.appendChild(sum);
         box.appendChild(body);
         out.appendChild(box);
       });
       if (!d.merged && !(d.results || []).length) out.textContent = "No results.";
     } catch (err) { out.textContent = "Run failed: " + err.message; }
   }

   wireRestoreInput();
"""

__all__ = ["UI_JS_ADMIN"]
