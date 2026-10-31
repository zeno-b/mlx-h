"""Front-end: the integrations panels inside Settings (Wave 4a items 2, 10, 16, 17).

Seven cards share this script part:

  * Tools > Catalogue: GET /api/tools/catalog (tool_catalog.py), one table of
    every tool with its capability, risk class, source and who reaches it.
  * Tools > Approvals: the caller's pending approvals (answered through the
    same /api/chat/turns/{turn}/approvals/{id} route the chat uses), the
    /api/tool-policies rules and the caller's own recent tool decisions.
  * Tools > Connect a client: /api/integrations/snippets per client, plus the
    `deploy.py wire <client> --login` command that mints a token for it.
  * Tools > API tokens: list, create (the plaintext is shown once), revoke.
  * Tools > MCP (admin): live /api/mcp status, this app's own /mcp endpoint,
    and a structured MCP_SERVERS editor. It writes the setting through
    /api/config only when the running process can adopt it (Config.MUTABLE);
    otherwise it prints the .env line with a restart note, because a POST the
    server refuses would look like a save that silently did nothing.
  * Admin > Audit: /api/audit/tool-decisions with filters and paging.
  * Agent > Projects: /api/projects, members, sharing and the context preview.

Why one part and not seven: the cards share the table, pill, date and copy
helpers and the tab registration, and each one is small. Loaders run only
while their tab is visible (integRegisterTab adds them to ADMIN_TABS, whose
poll stops when the tab hides), so opening Settings fetches nothing for a tab
nobody looks at. Server text reaches the DOM through textContent only. Role
gating here is cosmetic: every route enforces its own access and scoping.
"""

from __future__ import annotations

UI_JS_INTEGRATIONS = r"""   // ------------------------------------------------- integrations ---

   var _integCat = null;
   var _integClient = "claude-code";
   var _integClientData = null;
   var _integPol = { conversations: [], agents: [], users: [], caps: [], me: null,
                     risks: ["read", "write", "external", "exec"] };
   var _integUserNames = {};
   var _integAud = { rows: [], oldest: null };
   var _integMyAud = { rows: [], oldest: null };
   var _integMcpState = { rows: [], mutable: false, loaded: false, unparsed: false, json: "" };
   var _integProjState = { list: [], open: null, current: null };
   var INTEG_PAGE = 100;
   var INTEG_MY_PAGE = 20;
   var INTEG_RISK_PILL = { read: "ok", write: "warn", external: "running", exec: "error" };
   var INTEG_DECISION_PILL = { allow: "ok", ask: "warn", deny: "error" };
   var INTEG_SOURCE_LABEL = { builtin: "built in", extra: "extra", mcp: "MCP" };
   var INTEG_MODE_LABEL = { off: "never (off)", risky: "for risky tools", all: "for every tool" };

   // Add a loader (and an optional [fn, ms] poll) to the Settings tab registry
   // in ui_script_admin.py. A key that already has a loader keeps it and runs
   // this one after it, so two parts can serve one tab.
   function integRegisterTab(key, load, poll) {
     if (typeof ADMIN_TABS === "undefined") return;
     for (var i = 0; i < ADMIN_TABS.length; i++) {
       if (ADMIN_TABS[i].key !== key) continue;
       var before = ADMIN_TABS[i].load;
       ADMIN_TABS[i].load = function() {
         if (before) adminRunSafely(before);
         return load();
       };
       if (poll && !ADMIN_TABS[i].poll) ADMIN_TABS[i].poll = poll;
       return;
     }
     ADMIN_TABS.push({ key: key, load: load, poll: poll || null });
   }

   // ---------------------------------------------------------- helpers ---

   function integIsAdmin() {
     return document.body.getAttribute("data-role") === "admin";
   }

   function integNode(tag, cls, text) {
     var el = document.createElement(tag);
     if (cls) el.className = cls;
     if (text != null) el.textContent = String(text);
     return el;
   }

   function integButton(label, cls, onClick) {
     var b = integNode("button", "btn btn-sm" + (cls ? " " + cls : ""), label);
     b.type = "button";
     b.onclick = onClick;
     return b;
   }

   function integClear(host, message) {
     if (!host) return;
     host.innerHTML = "";
     if (message) host.appendChild(integNode("p", "muted", message));
   }

   // A kit .table with a header row appended to host; returns its tbody.
   function integTable(host, heads) {
     var table = integNode("table", "table");
     var head = document.createElement("thead");
     var row = document.createElement("tr");
     heads.forEach(function(h) { row.appendChild(integNode("th", null, h)); });
     head.appendChild(row);
     table.appendChild(head);
     var body = document.createElement("tbody");
     table.appendChild(body);
     host.appendChild(table);
     return body;
   }

   function integCell(row, value, cls) {
     var td = integNode("td", cls || null, value == null ? "" : value);
     row.appendChild(td);
     return td;
   }

   function integPill(kind, text) {
     return integNode("span", "pill " + (kind || "muted"), text);
   }

   // One dt/dd pair; value may be text or an element.
   function integKv(dl, key, value) {
     if (!dl) return;
     dl.appendChild(integNode("dt", null, key));
     var dd = integNode("dd");
     if (value && value.nodeType) dd.appendChild(value);
     else dd.textContent = value == null ? "" : String(value);
     dl.appendChild(dd);
   }

   function integOn(el, type, fn) {
     if (!el || el.getAttribute("data-integ-" + type) === "1") return;
     el.setAttribute("data-integ-" + type, "1");
     el.addEventListener(type, fn);
   }

   // SQLite timestamps are UTC without a zone ("2026-10-24 09:00:00"); ISO
   // ones carry it. Both parse to the same instant here.
   function integParseDate(value) {
     if (!value) return null;
     var s = String(value);
     if (s.indexOf("T") < 0) s = s.replace(" ", "T");
     if (!/(Z|[+-]\d\d:?\d\d)$/i.test(s)) s += "Z";
     var d = new Date(s);
     return isNaN(d.getTime()) ? null : d;
   }

   function integDate(value, withTime) {
     var d = integParseDate(value);
     if (!d) return value ? String(value) : "";
     return withTime ? d.toLocaleString() : d.toLocaleDateString();
   }

   function integAgo(value) {
     if (!value) return "never";
     var d = integParseDate(value);
     if (!d) return String(value);
     var s = Math.round((Date.now() - d.getTime()) / 1000);
     if (s < 0) return integDate(value, true);
     if (s < 60) return s + "s ago";
     if (s < 3600) return Math.round(s / 60) + "m ago";
     if (s < 86400) return Math.round(s / 3600) + "h ago";
     return Math.round(s / 86400) + "d ago";
   }

   function integCopyFrom(button, text) {
     if (!text) { uiToast("Nothing to copy yet.", "warn"); return; }
     if (typeof copyWithFeedback === "function" && button) { copyWithFeedback(button, text); return; }
     copyText(text).then(function() { uiToast("Copied.", "ok"); }, function() {
       uiToast("Could not copy. Select the text and copy it by hand.", "warn");
     });
   }

   async function integMe() {
     try {
       return ((await fetchJSON("/api/auth/me")).data || {}).user || null;
     } catch (err) {
       return null;
     }
   }

   // Admin only (the route is ADMIN): id -> username for audit and rules.
   async function integLoadUsers() {
     if (!integIsAdmin()) return;
     try {
       var users = ((await fetchJSON("/api/users")).data || {}).users || [];
       _integPol.users = users;
       users.forEach(function(u) { _integUserNames[u.id] = u.username; });
     } catch (err) {
       // Names stay ids; the rest of the card still works.
     }
   }

   function integUserName(id) {
     if (!id) return "local";
     if (_integPol.me && id === _integPol.me.id) return "you";
     return _integUserNames[id] || id;
   }

   // Bind the cards' controls once (each binder is idempotent).
   function integWire() {
     wireSeg(document.getElementById("integCatSource"), function() { integRenderCatalogue(); });
     integOn(document.getElementById("integCatFilter"), "input", function() { integRenderCatalogue(); });
     wireSeg(document.getElementById("integClientSeg"), function(value) { integLoadClient(value); });
     integOn(document.getElementById("integTokAll"), "change", function() { integLoadTokens(); });
     wireSeg(document.getElementById("integPolScope"), function(value) { integPolScopeChanged(value); });
     wireSeg(document.getElementById("integPolDecision"), null);
     integOn(document.getElementById("integAudUser"), "change", function() { integAudReload(); });
     integOn(document.getElementById("integAudTool"), "change", function() { integAudReload(); });
     integOn(document.getElementById("integAudDecision"), "change", function() { integAudReload(); });
     integOn(document.getElementById("integAudFrom"), "change", function() { integAudReload(); });
     integOn(document.getElementById("integAudTo"), "change", function() { integAudReload(); });
     integOn(document.getElementById("integMcpRows"), "input", function() { integMcpRender(); });
     integOn(document.getElementById("integMcpRows"), "change", function() { integMcpRender(); });
   }

   // -------------------------------------------------------- catalogue ---

   async function integLoadCatalogue() {
     integWire();
     var host = document.getElementById("toolsList");
     if (!host) return;
     try {
       _integCat = (await fetchJSON("/api/tools/catalog")).data || {};
     } catch (err) {
       integClear(host, "Could not load the tool catalogue: " + err.message);
       return;
     }
     integRenderCatalogue();
   }

   function integRenderCatalogue() {
     var host = document.getElementById("toolsList");
     var summary = document.getElementById("integCatSummary");
     if (!host || !_integCat) return;
     var q = String(document.getElementById("integCatFilter").value || "").trim().toLowerCase();
     var source = uiSegValue(document.getElementById("integCatSource")) || "";
     var admin = !!_integCat.admin;
     var all = _integCat.tools || [];
     var tools = all.filter(function(t) {
       if (source && t.source !== source) return false;
       if (!q) return true;
       return [t.name, t.description, t.capability, t.risk, t.server].join(" ").toLowerCase().indexOf(q) >= 0;
     });
     integClear(host);
     if (!tools.length) {
       host.appendChild(integNode("p", "muted", all.length ? "No tool matches the filter." : "No tools are available to you."));
     } else {
       var heads = ["Tool", "Capability", "Risk", "Source", "Agent profiles"];
       if (admin) heads.push("Roles");
       var body = integTable(host, heads);
       tools.forEach(function(t) {
         var tr = document.createElement("tr");
         var name = integCell(tr, "");
         name.appendChild(integNode("div", "mono", t.name));
         if (t.description) name.appendChild(integNode("div", "hint", t.description));
         integCell(tr, t.always ? "always on" : (t.capability || "none"), "mono");
         integCell(tr, "", "cell-status").appendChild(integPill(INTEG_RISK_PILL[t.risk], t.risk || "unknown"));
         var src = integCell(tr, "", "cell-status");
         src.appendChild(integNode("span", "badge", (INTEG_SOURCE_LABEL[t.source] || t.source) + (t.server ? ": " + t.server : "")));
         if (admin && !t.registered) src.appendChild(integNode("div", "hint", "not on the default agent"));
         var profiles = (t.profiles || []).map(function(p) { return p.name + (p.enabled ? "" : " (off)"); });
         integCell(tr, profiles.length ? profiles.join(", ") : (t.always ? "every agent" : "none"));
         // nowrap: the table's overflow-wrap:anywhere let the auto layout
         // squeeze this column until ", user" broke onto its own line.
         if (admin) integCell(tr, (t.roles || []).join(", ") || "none", "nowrap");
         body.appendChild(tr);
       });
     }
     if (summary) {
       var text = tools.length + " of " + all.length + " tools shown. ";
       if (admin) {
         text += _integCat.role_capabilities_set ? "Role grants come from ROLE_CAPABILITIES."
                                                 : "Role grants are the defaults (ROLE_CAPABILITIES is not set).";
       } else {
         text += "These are the default agent's tools and those of the agent profiles you may pick.";
       }
       summary.textContent = text;
     }
   }

   // ---------------------------------------------------------- clients ---

   async function integLoadClient(client) {
     integWire();
     if (client) _integClient = client;
     var wanted = _integClient;
     uiSeg(document.getElementById("integClientSeg"), wanted);
     var pre = document.getElementById("integClientText");
     var d;
     try {
       d = (await fetchJSON("/api/integrations/snippets?client=" + encodeURIComponent(wanted))).data || {};
     } catch (err) {
       pre.textContent = "";
       uiToast("Could not load the client settings: " + err.message, "error");
       return;
     }
     if (wanted !== _integClient) return;   // a newer pick is loading
     _integClientData = d;
     document.getElementById("integClientTitle").textContent = d.title || wanted;
     pre.textContent = d.text || "";
     document.getElementById("integClientCmd").textContent = d.command || "";
     var facts = document.getElementById("integClientFacts");
     facts.innerHTML = "";
     integKv(facts, "Server", d.url || "");
     integKv(facts, "Model", d.model || "");
     integKv(facts, "Token", d.auth_enabled
       ? "replace " + (d.token_placeholder || "the placeholder") + " with an API token from the API tokens tab, or let --login mint one"
       : "not needed: sign-in is off on this server");
     var notes = document.getElementById("integClientNotes");
     integClear(notes);
     (d.notes || []).forEach(function(n) { notes.appendChild(integNode("p", "hint", n)); });
   }

   function integCopyClientCmd(button) {
     integCopyFrom(button, (_integClientData || {}).command || "");
   }

   function integCopyClientText(button) {
     integCopyFrom(button, (_integClientData || {}).text || "");
   }

   // ----------------------------------------------------------- tokens ---

   async function integLoadTokens() {
     integWire();
     var host = document.getElementById("integTokList");
     if (!host) return;
     var all = document.getElementById("integTokAll");
     var wantAll = !!(all && all.checked && integIsAdmin());
     var d;
     try {
       d = (await fetchJSON("/api/auth/tokens" + (wantAll ? "?all=1" : ""))).data || {};
     } catch (err) {
       integClear(host, "Could not load tokens: " + err.message);
       return;
     }
     var off = d.auth_enabled === false;
     document.getElementById("integTokAuthOff").hidden = !off;
     document.getElementById("integTokForm").hidden = off;
     integRenderTokens(d.tokens || [], !!d.all);
   }

   function integRenderTokens(tokens, all) {
     var host = document.getElementById("integTokList");
     integClear(host);
     if (!tokens.length) { host.appendChild(integNode("p", "muted", "No tokens yet.")); return; }
     var heads = ["Name", "Token", "Scopes", "Created", "Last used", "Expires", "State", ""];
     if (all) heads.splice(1, 0, "Owner");
     var body = integTable(host, heads);
     tokens.forEach(function(t) {
       var tr = document.createElement("tr");
       integCell(tr, t.name || "token");
       if (all) integCell(tr, t.username || t.user_id || "");
       integCell(tr, t.hint || "", "mono");
       integCell(tr, (t.scopes || []).join(", "));
       integCell(tr, integDate(t.created_at));
       integCell(tr, integAgo(t.last_used_at));
       integCell(tr, t.expires_at ? integDate(t.expires_at) : "never");
       var state = t.active ? "active" : (t.revoked_at ? "revoked" : "expired");
       var pill = integCell(tr, "", "cell-status").appendChild(integPill(t.active ? "ok" : "muted", state));
       if (t.revoked_reason) pill.title = t.revoked_reason;
       var act = integCell(tr, "", "cell-action");
       if (t.active) act.appendChild(integButton("Revoke", "btn-danger", function() { integRevokeToken(t); }));
       body.appendChild(tr);
     });
   }

   async function integCreateToken() {
     var nameEl = document.getElementById("integTokName");
     var scopes = [];
     Array.prototype.forEach.call(document.getElementById("integTokScopes").querySelectorAll("input[type=checkbox]"), function(c) {
       if (c.checked) scopes.push(c.value);
     });
     var body = { name: nameEl.value.trim(), scopes: scopes };
     var days = document.getElementById("integTokDays").value.trim();
     if (days !== "") body.ttl_days = Number(days);
     var btn = document.getElementById("integTokCreateBtn");
     btn.disabled = true;
     try {
       var out = await postJSON("/api/auth/tokens", body);
       var field = document.getElementById("integTokNew");
       field.value = (out.data || {}).token || "";
       document.getElementById("integTokNewBox").hidden = false;
       field.focus();
       field.select();
       nameEl.value = "";
       uiToast("Token created. Copy it now: it is shown only once.", "ok");
       integLoadTokens();
     } catch (err) {
       uiToast("Could not create the token: " + err.message, "error");
     } finally {
       btn.disabled = false;
     }
   }

   function integCopyNewToken(button) {
     integCopyFrom(button, document.getElementById("integTokNew").value);
   }

   // The plaintext leaves the page as soon as it is no longer needed.
   function integHideNewToken() {
     var field = document.getElementById("integTokNew");
     if (field) field.value = "";
     var box = document.getElementById("integTokNewBox");
     if (box) box.hidden = true;
   }

   async function integRevokeToken(t) {
     if (!uiConfirm("Revoke the token " + (t.name || t.hint) + "? Clients using it stop working at once.")) return;
     await uiAct("Could not revoke the token", async function() {
       await fetchJSON("/api/auth/tokens/" + encodeURIComponent(t.id), { method: "DELETE" });
       uiToast("Token revoked.", "ok");
       await integLoadTokens();
     });
   }

   // -------------------------------------------------------- approvals ---

   async function integLoadApprovalsTab() {
     integWire();
     integLoadPending();
     await integLoadPolicyChoices();
     await integLoadPolicies();
     integLoadMyAudit(true);
   }

   async function integLoadPending() {
     var host = document.getElementById("integApprList");
     var count = document.getElementById("integApprCount");
     if (!host) return;
     var items;
     try {
       items = ((await fetchJSON("/api/approvals/pending")).data || {}).approvals || [];
     } catch (err) {
       integClear(host, "Could not load approvals: " + err.message);
       return;
     }
     count.textContent = String(items.length);
     count.className = "pill " + (items.length ? "warn" : "muted");
     integClear(host);
     if (!items.length) { host.appendChild(integNode("p", "muted", "Nothing is waiting for you.")); return; }
     items.forEach(function(a) { host.appendChild(integApprovalRow(a)); });
   }

   function integApprovalRow(a) {
     var box = integNode("div", "stack tight list-row");
     var head = integNode("div", "row");
     head.appendChild(integNode("span", "mono grow", a.name));
     head.appendChild(integPill(INTEG_RISK_PILL[a.risk], a.risk || "unknown"));
     head.appendChild(integNode("span", "hint", (a.expires_in || 0) + "s left"));
     box.appendChild(head);
     if (a.reason) box.appendChild(integNode("p", "hint", a.reason));
     box.appendChild(integNode("pre", "logbox mono", JSON.stringify(a.args || {}, null, 2)));
     var scopes = a.scopes || [];
     var actions = integNode("div", "row wrap");
     actions.appendChild(integButton("Allow once", "btn-primary", function() { integAnswerApproval(a, "allow", "once"); }));
     if (scopes.indexOf("conversation") >= 0) {
       actions.appendChild(integButton("Allow in this conversation", "", function() { integAnswerApproval(a, "allow", "conversation"); }));
     }
     if (scopes.indexOf("tool") >= 0) {
       actions.appendChild(integButton("Always allow " + a.name, "", function() { integAnswerApproval(a, "allow", "tool"); }));
     }
     actions.appendChild(integButton("Deny", "btn-danger", function() { integAnswerApproval(a, "deny", "once"); }));
     box.appendChild(actions);
     return box;
   }

   async function integAnswerApproval(a, decision, scope) {
     var ok = await uiAct("Could not answer the approval", async function() {
       await postJSON("/api/chat/turns/" + encodeURIComponent(a.turn_id) + "/approvals/" + encodeURIComponent(a.id),
                      { decision: decision, scope: scope });
       return true;
     });
     if (ok) uiToast(decision === "allow" ? "Allowed." : "Denied.", decision === "allow" ? "ok" : "info");
     integLoadPending();
     if (ok && scope !== "once") integLoadPolicies();
   }

   // What the rule form offers: tools, capabilities, risk classes, and the
   // targets of each scope (own conversations; profiles and users for admins).
   async function integLoadPolicyChoices() {
     var jobs = [
       fetchJSON("/api/conversations?limit=100").then(function(o) {
         _integPol.conversations = (o.data || {}).conversations || [];
       }, function() {}),
       fetchJSON("/api/agents/capabilities").then(function(o) {
         _integPol.caps = (o.data || {}).capabilities || [];
       }, function() {}),
       integMe().then(function(me) { _integPol.me = me; })
     ];
     if (!_integCat) {
       jobs.push(fetchJSON("/api/tools/catalog").then(function(o) { _integCat = o.data || {}; }, function() {}));
     }
     if (integIsAdmin()) {
       jobs.push(fetchJSON("/api/agents").then(function(o) {
         _integPol.agents = (o.data || {}).agents || [];
       }, function() {}));
       jobs.push(integLoadUsers());
     }
     await Promise.all(jobs);
     integFillToolPicker();
     integPolScopeChanged(uiSegValue(document.getElementById("integPolScope")) || "user");
   }

   function integOptions(select, items) {
     items.forEach(function(it) {
       var o = document.createElement("option");
       o.value = it[0];
       o.textContent = it[1];
       select.appendChild(o);
     });
   }

   function integOptGroup(select, label, items) {
     if (!items.length) return;
     var group = document.createElement("optgroup");
     group.label = label;
     integOptions(group, items);
     select.appendChild(group);
   }

   function integFillToolPicker() {
     var select = document.getElementById("integPolTool");
     if (!select) return;
     var keep = select.value;
     select.innerHTML = "";
     var names = ((_integCat || {}).tools || []).map(function(t) { return t.name; }).sort();
     integOptGroup(select, "Tools", names.map(function(n) { return [n, n]; }));
     // Rule keys accept cap:<a-z0-9_> only, so per-server mcp:<name> keys are
     // left out (the MCP tools themselves are listed above).
     integOptGroup(select, "Capabilities", (_integPol.caps || []).filter(function(c) {
       return /^[a-z0-9_]+$/.test(c.key);
     }).map(function(c) { return ["cap:" + c.key, "cap:" + c.key + (c.label ? " (" + c.label + ")" : "")]; }));
     integOptGroup(select, "Risk classes", _integPol.risks.map(function(r) { return ["risk:" + r, "risk:" + r]; }));
     integOptGroup(select, "Every tool", [["*", "* (every tool)"]]);
     if (keep) select.value = keep;
   }

   function integPolScopeChanged(scope) {
     var row = document.getElementById("integPolTargetRow");
     var label = document.getElementById("integPolTargetLabel");
     var select = document.getElementById("integPolTarget");
     if (!row || !select) return;
     select.innerHTML = "";
     var me = _integPol.me || {};
     var items = [];
     if (scope === "conversation") {
       label.textContent = "Conversation";
       items = _integPol.conversations.map(function(c) {
         var id = c.conversation_id || c.id;
         return [id, c.title || id];
       });
       if (!items.length) items = [["", "No conversations yet"]];
     } else if (scope === "profile") {
       label.textContent = "Agent profile";
       items = _integPol.agents.map(function(a) { return [a.id, a.name + (a.enabled ? "" : " (off)")]; });
       if (!items.length) items = [["", "No agent profiles yet"]];
     } else if (scope === "user" && integIsAdmin() && _integPol.users.length) {
       label.textContent = "User";
       items = _integPol.users.map(function(u) { return [u.id, u.username + (u.id === me.id ? " (you)" : "")]; });
     }
     integOptions(select, items);
     if (scope === "user" && me.id) select.value = me.id;
     row.hidden = !items.length;
   }

   async function integLoadPolicies() {
     var host = document.getElementById("integPolList");
     var facts = document.getElementById("integPolFacts");
     if (!host) return;
     var d;
     try {
       d = (await fetchJSON("/api/tool-policies")).data || {};
     } catch (err) {
       integClear(host, "Could not load the rules: " + err.message);
       return;
     }
     if (d.risk_classes && d.risk_classes.length) _integPol.risks = d.risk_classes;
     facts.innerHTML = "";
     integKv(facts, "Ask before running", INTEG_MODE_LABEL[d.mode] || d.mode || "");
     integKv(facts, "When nobody can answer", d.unattended || "");
     integKv(facts, "Approval timeout", (d.timeout_s == null ? "" : d.timeout_s + " s"));
     integKv(facts, "Re-ask after reading the web", d.taint ? "on" : "off");
     var rows = d.policies || [];
     integClear(host);
     if (!rows.length) { host.appendChild(integNode("p", "muted", "No rules yet: the installation default applies.")); return; }
     var me = _integPol.me || {};
     var admin = integIsAdmin();
     var body = integTable(host, ["Applies to", "Tool", "Decision", "Expires", ""]);
     rows.forEach(function(p) {
       var tr = document.createElement("tr");
       integCell(tr, integPolTargetText(p));
       integCell(tr, p.tool, "mono");
       integCell(tr, "", "cell-status").appendChild(integPill(INTEG_DECISION_PILL[p.decision], p.decision));
       integCell(tr, p.expired ? "expired" : (p.expires_at ? integDate(p.expires_at, true) : "never"));
       var act = integCell(tr, "", "cell-action");
       // Users remove only rules they set themselves (the server says 403 otherwise).
       if (admin || (p.created_by && p.created_by === me.id)) {
         act.appendChild(integButton("Remove", "btn-ghost btn-danger", function() { integDeletePolicy(p); }));
       } else {
         act.appendChild(integNode("span", "hint", "set by an administrator"));
       }
       body.appendChild(tr);
     });
   }

   function integPolTargetText(p) {
     if (p.scope === "global") return "everyone";
     if (p.scope === "user") return integUserName(p.scope_id) === "you" ? "you" : "user " + integUserName(p.scope_id);
     if (p.scope === "profile") {
       var agent = _integPol.agents.filter(function(a) { return a.id === p.scope_id; })[0];
       return "agent " + (agent ? agent.name : p.scope_id);
     }
     if (p.scope === "conversation") {
       var conv = _integPol.conversations.filter(function(c) { return (c.conversation_id || c.id) === p.scope_id; })[0];
       return "conversation " + (conv && conv.title ? conv.title : p.scope_id);
     }
     return p.scope + " " + (p.scope_id || "");
   }

   async function integSavePolicy() {
     var scope = uiSegValue(document.getElementById("integPolScope")) || "user";
     var body = {
       scope: scope,
       tool: document.getElementById("integPolTool").value,
       decision: uiSegValue(document.getElementById("integPolDecision")) || "ask"
     };
     var target = document.getElementById("integPolTarget").value;
     if (scope === "conversation" || scope === "profile") {
       if (!target) { uiToast(scope === "conversation" ? "Pick a conversation first." : "Pick an agent profile first.", "warn"); return; }
       body.scope_id = target;
     } else if (scope === "user" && target) {
       body.scope_id = target;
     }
     if (!body.tool) { uiToast("Pick a tool first.", "warn"); return; }
     var expires = document.getElementById("integPolExpires").value;
     if (expires) body.expires_in_s = Number(expires);
     await uiAct("Could not save the rule", async function() {
       await postJSON("/api/tool-policies", body);
       uiToast("Rule saved: " + body.decision + " " + body.tool + ".", "ok");
       await integLoadPolicies();
     });
   }

   async function integDeletePolicy(p) {
     if (!uiConfirm("Remove the rule " + p.decision + " " + p.tool + " for " + integPolTargetText(p) + "?")) return;
     await uiAct("Could not remove the rule", async function() {
       await fetchJSON("/api/tool-policies/" + encodeURIComponent(p.id), { method: "DELETE" });
       uiToast("Rule removed.", "ok");
       await integLoadPolicies();
     });
   }

   // The caller's own decisions (the route scopes a user to their rows; an
   // administrator asks for their own id explicitly).
   async function integLoadMyAudit(reset) {
     var host = document.getElementById("integMyAudit");
     if (!host) return;
     if (reset) _integMyAud = { rows: [], oldest: null };
     var url = "/api/audit/tool-decisions?limit=" + INTEG_MY_PAGE;
     if (integIsAdmin()) {
       var me = _integPol.me || await integMe();
       if (me && me.id) url += "&user_id=" + encodeURIComponent(me.id);
     }
     if (_integMyAud.oldest) url += "&before_id=" + encodeURIComponent(_integMyAud.oldest);
     var rows;
     try {
       rows = ((await fetchJSON(url)).data || {}).decisions || [];
     } catch (err) {
       integClear(host, "Could not load your decisions: " + err.message);
       return;
     }
     _integMyAud.rows = _integMyAud.rows.concat(rows);
     if (rows.length) _integMyAud.oldest = rows[rows.length - 1].id;
     document.getElementById("integMyAuditMore").hidden = rows.length < INTEG_MY_PAGE;
     integRenderDecisions(host, _integMyAud.rows, false, "No tool decisions recorded for you yet.");
   }

   function integRenderDecisions(host, rows, showUser, emptyText) {
     integClear(host);
     if (!rows.length) { host.appendChild(integNode("p", "muted", emptyText)); return; }
     var heads = ["When", "Tool", "Risk", "Decision", "Rule", "Arguments"];
     if (showUser) heads.splice(1, 0, "User");
     var body = integTable(host, heads);
     rows.forEach(function(r) {
       var tr = document.createElement("tr");
       integCell(tr, integDate(r.created_at, true), "nowrap");
       if (showUser) integCell(tr, integUserName(r.user_id));
       integCell(tr, r.tool, "mono");
       integCell(tr, "", "cell-status").appendChild(integPill(INTEG_RISK_PILL[r.risk], r.risk || "unknown"));
       var decision = integCell(tr, "", "cell-status");
       decision.appendChild(integPill(INTEG_DECISION_PILL[r.decision], r.decision));
       if (r.latency_ms != null) decision.appendChild(integNode("div", "hint", (r.latency_ms / 1000).toFixed(1) + " s"));
       integCell(tr, (r.rule || "") + (r.approver_id ? " by " + integUserName(r.approver_id) : ""), "mono");
       var args = JSON.stringify(r.args || {});
       var cell = integCell(tr, args.length > 160 ? args.slice(0, 160) + "..." : args, "mono");
       cell.title = args;
       body.appendChild(tr);
     });
   }

   // ------------------------------------------------------------ audit ---

   async function integLoadAudit(reset) {
     integWire();
     var host = document.getElementById("integAudList");
     if (!host) return;
     if (reset) {
       _integAud = { rows: [], oldest: null };
       _integPol.me = _integPol.me || await integMe();
       await integLoadUsers();
       integFillAuditUsers();
     }
     var url = "/api/audit/tool-decisions?limit=" + INTEG_PAGE;
     var user = document.getElementById("integAudUser").value;
     var tool = document.getElementById("integAudTool").value.trim();
     var decision = document.getElementById("integAudDecision").value;
     var from = document.getElementById("integAudFrom").value;
     var to = document.getElementById("integAudTo").value;
     if (user) url += "&user_id=" + encodeURIComponent(user);
     if (tool) url += "&tool=" + encodeURIComponent(tool);
     if (decision) url += "&decision=" + encodeURIComponent(decision);
     // The date inputs are local days; the server compares UTC instants.
     if (from) url += "&since=" + encodeURIComponent(new Date(from + "T00:00:00").toISOString());
     if (to) url += "&until=" + encodeURIComponent(new Date(to + "T23:59:59.999").toISOString());
     if (_integAud.oldest) url += "&before_id=" + encodeURIComponent(_integAud.oldest);
     var data;
     try {
       data = (await fetchJSON(url)).data || {};
     } catch (err) {
       integClear(host, "Could not load the audit log: " + err.message);
       return;
     }
     var rows = data.decisions || [];
     _integAud.rows = _integAud.rows.concat(rows);
     if (rows.length) _integAud.oldest = rows[rows.length - 1].id;
     document.getElementById("integAudMore").hidden = data.has_more === undefined
       ? rows.length < INTEG_PAGE : !data.has_more;
     integRenderAudit();
   }

   // Every filter runs on the server; a change starts from the newest.
   function integAudReload() {
     _integAud = { rows: [], oldest: null };
     integLoadAudit(false);
   }

   function integFillAuditUsers() {
     var select = document.getElementById("integAudUser");
     if (!select) return;
     var keep = select.value;
     select.innerHTML = "";
     integOptions(select, [["", "All users"]].concat(_integPol.users.map(function(u) {
       return [u.id, u.username];
     })));
     select.value = keep;
   }

   // The rows the server matched so far; Older loads the next page.
   function integRenderAudit() {
     var host = document.getElementById("integAudList");
     if (!host) return;
     var filtered = ["integAudUser", "integAudTool", "integAudDecision", "integAudFrom", "integAudTo"]
       .some(function(id) { var el = document.getElementById(id); return el && String(el.value || "").trim(); });
     integRenderDecisions(host, _integAud.rows, true, filtered
       ? "No tool decision matches these filters."
       : "No tool decisions recorded yet.");
     document.getElementById("integAudCount").textContent = _integAud.rows.length
       ? _integAud.rows.length + " row" + (_integAud.rows.length === 1 ? "" : "s") + " shown." : "";
   }

   // -------------------------------------------------------------- MCP ---

   async function integLoadMcpTab() {
     integWire();
     integLoadMcp();
     if (!_integMcpState.loaded) await integMcpLoadConfig();
   }

   async function integLoadMcp() {
     var host = document.getElementById("mcpStatus");
     if (!host) return;
     try {
       var d = (await fetchJSON("/api/mcp")).data || {};
       var servers = d.servers || [];
       integClear(host);
       if (!servers.length) {
         host.appendChild(integNode("p", "muted", d.configured
           ? "MCP_SERVERS is set, but no server is running: check the entries below and the app log."
           : "No MCP servers configured."));
       } else {
         var body = integTable(host, ["Server", "Transport", "Protocol", "Tools", "State"]);
         servers.forEach(function(s) {
           var tr = document.createElement("tr");
           var name = integCell(tr, "");
           name.appendChild(integNode("div", "mono", s.name));
           if (s.command) name.appendChild(integNode("div", "hint mono", s.command));
           if (s.error) name.appendChild(integNode("div", "hint error", s.error));
           integCell(tr, s.transport || "stdio");
           integCell(tr, s.protocol_version || "unknown", "mono");
           var tools = s.tools || [];
           var cell = integCell(tr, String(tools.length));
           if (tools.length) cell.title = tools.join(", ");
           integCell(tr, "", "cell-status").appendChild(integPill(s.alive ? "ok" : "error", s.alive ? "running" : "stopped"));
           body.appendChild(tr);
         });
       }
     } catch (err) {
       integClear(host, "Could not load MCP status: " + err.message);
     }
     await integMcpServeFacts();
   }

   async function integMcpServeFacts() {
     var dl = document.getElementById("integMcpServe");
     if (!dl) return;
     try {
       _integCat = (await fetchJSON("/api/tools/catalog")).data || {};
     } catch (err) {
       return;
     }
     var m = _integCat.mcp || {};
     dl.innerHTML = "";
     integKv(dl, "Serving tools", integPill(m.server_enabled ? "ok" : "muted", m.server_enabled ? "on" : "off"));
     integKv(dl, "URL", window.location.origin + (m.endpoint || "/mcp"));
   }

   // MCP_SERVERS accepts a list, {"mcpServers": {...}} or {name: {...}}.
   function integMcpEntries(text) {
     var data;
     try { data = JSON.parse(text); } catch (err) { return null; }
     var out = [];
     if (Array.isArray(data)) {
       data.forEach(function(e) { if (e && typeof e === "object") out.push(e); });
       return out;
     }
     if (!data || typeof data !== "object") return null;
     var map = data.mcpServers && typeof data.mcpServers === "object" ? data.mcpServers : data;
     Object.keys(map).forEach(function(name) {
       var body = map[name];
       if (!body || typeof body !== "object") return;
       var entry = { name: name };
       Object.keys(body).forEach(function(k) { if (k !== "name") entry[k] = body[k]; });
       out.push(entry);
     });
     return out;
   }

   function integMcpPairs(obj, sep) {
     if (!obj || typeof obj !== "object") return "";
     return Object.keys(obj).map(function(k) { return k + sep + obj[k]; }).join("\n");
   }

   function integMcpRowFrom(entry) {
     var kind = String(entry.type || entry.transport || "").toLowerCase();
     var http = kind === "http" || (!entry.command && !!entry.url);
     return {
       type: http ? "http" : "stdio",
       name: String(entry.name || ""),
       command: String(entry.command || ""),
       args: (Array.isArray(entry.args) ? entry.args : []).map(String).join("\n"),
       env: integMcpPairs(entry.env, "="),
       cwd: String(entry.cwd || ""),
       url: String(entry.url || ""),
       headers: integMcpPairs(entry.headers, ": "),
       bearer: String(entry.bearer || entry.token || ""),
       enabled: entry.enabled !== false
     };
   }

   async function integMcpLoadConfig() {
     integWire();
     var cfg;
     try {
       cfg = ((await fetchJSON("/api/config")).data || {}).config || {};
     } catch (err) {
       uiToast("Could not read MCP_SERVERS: " + err.message, "error");
       return;
     }
     var raw = String(cfg.mcp_servers || "").trim();
     var entries = raw ? integMcpEntries(raw) : [];
     _integMcpState.mutable = (cfg.mutable || []).indexOf("mcp_servers") >= 0;
     _integMcpState.unparsed = entries === null;
     _integMcpState.rows = (entries || []).map(integMcpRowFrom);
     _integMcpState.loaded = true;
     integMcpDraw();
   }

   function integMcpDraw() {
     var host = document.getElementById("integMcpRows");
     integClear(host, _integMcpState.rows.length ? "" : "No servers yet: add one with the buttons above.");
     _integMcpState.rows.forEach(function(r) { host.appendChild(integMcpRowEl(r)); });
     integMcpRender();
   }

   function integMcpAdd(type) {
     integWire();
     var host = document.getElementById("integMcpRows");
     if (!host.querySelector("[data-mcp-row]")) integClear(host);
     var el = integMcpRowEl(integMcpRowFrom(type === "http" ? { type: "http" } : {}));
     host.appendChild(el);
     var first = el.querySelector('[data-mcp="name"]');
     if (first) first.focus();
     integMcpRender();
   }

   function integMcpInput(tag, key, value, placeholder) {
     var el = document.createElement(tag);
     el.className = tag === "textarea" ? "input textarea mono" : "input mono";
     if (tag === "textarea") el.rows = 2;
     else el.type = "text";
     el.value = value || "";
     el.placeholder = placeholder || "";
     el.setAttribute("data-mcp", key);
     el.setAttribute("autocomplete", "off");
     el.setAttribute("spellcheck", "false");
     return el;
   }

   function integMcpField(label, control) {
     var wrap = integNode("label", "field-row");
     wrap.appendChild(integNode("span", "field-label", label));
     var holder = integNode("div", "field-control");
     holder.appendChild(control);
     wrap.appendChild(holder);
     return wrap;
   }

   function integMcpRowEl(row) {
     var box = integNode("div", "card stack tight");
     box.setAttribute("data-mcp-row", row.type);
     var head = integNode("div", "row nowrap");
     head.appendChild(integNode("span", "badge", row.type === "http" ? "HTTP" : "stdio"));
     var name = integMcpInput("input", "name", row.name, "Server name");
     name.className = "input mono grow";
     name.setAttribute("aria-label", "Server name");
     head.appendChild(name);
     var sw = integNode("label", "switch");
     sw.title = "Enabled";
     var on = document.createElement("input");
     on.type = "checkbox";
     on.checked = row.enabled !== false;
     on.setAttribute("data-mcp", "enabled");
     on.setAttribute("aria-label", "Enabled");
     sw.appendChild(on);
     sw.appendChild(integNode("span", "slider"));
     head.appendChild(sw);
     head.appendChild(integButton("Remove", "btn-ghost btn-danger", function() {
       if (box.parentNode) box.parentNode.removeChild(box);
       integMcpRender();
     }));
     box.appendChild(head);
     if (row.type === "http") {
       box.appendChild(integMcpField("URL", integMcpInput("input", "url", row.url, "https://example.com/mcp")));
       box.appendChild(integMcpField("Headers", integMcpInput("textarea", "headers", row.headers, "One per line: Name: value")));
       box.appendChild(integMcpField("Bearer token", integMcpInput("input", "bearer", row.bearer, "${MY_SERVER_TOKEN}")));
     } else {
       box.appendChild(integMcpField("Command", integMcpInput("input", "command", row.command, "npx")));
       box.appendChild(integMcpField("Arguments", integMcpInput("textarea", "args", row.args, "One argument per line")));
       box.appendChild(integMcpField("Environment", integMcpInput("textarea", "env", row.env, "One per line: NAME=value or NAME=${NAME}")));
       box.appendChild(integMcpField("Working folder", integMcpInput("input", "cwd", row.cwd, "Optional")));
     }
     return box;
   }

   function integMcpCollect() {
     var rows = [];
     var host = document.getElementById("integMcpRows");
     Array.prototype.forEach.call(host.querySelectorAll("[data-mcp-row]"), function(box) {
       var row = { type: box.getAttribute("data-mcp-row") };
       Array.prototype.forEach.call(box.querySelectorAll("[data-mcp]"), function(el) {
         row[el.getAttribute("data-mcp")] = el.type === "checkbox" ? el.checked : el.value;
       });
       rows.push(row);
     });
     return rows;
   }

   function integMcpLines(text) {
     return String(text || "").split("\n").map(function(s) { return s.trim(); })
       .filter(function(s) { return s.length > 0; });
   }

   function integMcpMap(text, sep, problems, what, label) {
     var out = {};
     var count = 0;
     integMcpLines(text).forEach(function(line, i) {
       var at = line.indexOf(sep);
       if (at <= 0) {
         problems.push(label + ": " + what + " line " + (i + 1) + " needs " + (sep === "=" ? "NAME=value." : "Name: value."));
         return;
       }
       out[line.slice(0, at).trim()] = line.slice(at + 1).trim();
       count += 1;
     });
     return count ? out : null;
   }

   function integMcpBuild(rows, problems) {
     var seen = {};
     return rows.map(function(r, i) {
       var name = String(r.name || "").trim();
       var label = name || "server " + (i + 1);
       if (!name) problems.push(label + ": a name is required.");
       else if (seen[name]) problems.push(label + ": that name is used twice.");
       seen[name] = true;
       var e = { name: name };
       if (r.type === "http") {
         e.type = "http";
         e.url = String(r.url || "").trim();
         if (!/^https?:\/\//i.test(e.url)) problems.push(label + ": the URL must start with http:// or https://.");
         var headers = integMcpMap(r.headers, ":", problems, "header", label);
         if (headers) e.headers = headers;
         if (String(r.bearer || "").trim()) e.bearer = String(r.bearer).trim();
       } else {
         e.command = String(r.command || "").trim();
         if (!e.command) problems.push(label + ": a command is required.");
         var args = integMcpLines(r.args);
         if (args.length) e.args = args;
         var env = integMcpMap(r.env, "=", problems, "environment", label);
         if (env) e.env = env;
         if (String(r.cwd || "").trim()) e.cwd = String(r.cwd).trim();
       }
       if (r.enabled === false) e.enabled = false;
       return e;
     });
   }

   // The edited list as JSON, the .env line (single-quoted for a shell), the
   // problems that would make the server drop an entry, and the masked-value
   // warning: GET /api/config shows header and env values as ***.
   function integMcpRender() {
     var problems = [];
     var entries = integMcpBuild(integMcpCollect(), problems);
     var json = entries.length ? JSON.stringify(entries) : "";
     _integMcpState.json = json;
     var q = "'";
     var quoted = json ? q + json.split(q).join(q + '"' + q + '"' + q) + q : "";
     document.getElementById("integMcpJson").textContent = "MCP_SERVERS=" + quoted;
     if (_integMcpState.unparsed) {
       problems.unshift("The current MCP_SERVERS is not JSON this editor can read, so it started empty; its result replaces the old value.");
     }
     var box = document.getElementById("integMcpProblems");
     box.textContent = problems.join(" ");
     box.hidden = !problems.length;
     document.getElementById("integMcpMasked").hidden = json.indexOf("***") < 0;
     var apply = document.getElementById("integMcpApply");
     apply.hidden = !_integMcpState.mutable;
     apply.disabled = problems.length > 0;
     document.getElementById("integMcpRestart").hidden = _integMcpState.mutable;
     document.getElementById("integMcpHow").textContent = _integMcpState.mutable
       ? "This server adopts MCP_SERVERS while running: Apply saves it."
       : "MCP_SERVERS is read when the app starts: replace the MCP_SERVERS line in .env with this one and restart the app.";
   }

   function integMcpCopy(button) {
     integMcpRender();
     integCopyFrom(button, _integMcpState.json ? document.getElementById("integMcpJson").textContent : "");
   }

   async function integMcpApply() {
     integMcpRender();
     if (!_integMcpState.mutable) return;
     if (_integMcpState.json.indexOf("***") >= 0) {
       uiToast("Replace every *** with the real value or a ${NAME} reference first.", "warn");
       return;
     }
     await uiAct("Could not save MCP_SERVERS", async function() {
       var out = await postJSON("/api/config", { mcp_servers: _integMcpState.json });
       var changed = (out.data || {}).changed || [];
       uiToast(changed.indexOf("mcp_servers") >= 0 ? "MCP_SERVERS saved." : "Nothing changed.", "ok");
       _integMcpState.loaded = false;
       await integLoadMcpTab();
     });
   }

   // --------------------------------------------------------- projects ---

   async function integLoadProjects() {
     integWire();
     var host = document.getElementById("integProjList");
     if (!host) return;
     var d;
     try {
       d = (await fetchJSON("/api/projects")).data || {};
     } catch (err) {
       integClear(host, "Could not load projects: " + err.message);
       return;
     }
     document.getElementById("integProjOff").hidden = !!d.enabled;
     document.getElementById("integProjBody").hidden = !d.enabled;
     var warn = document.getElementById("integProjWarn");
     warn.textContent = d.warning || "";
     warn.hidden = !d.warning;
     document.getElementById("integProjRootHint").textContent = d.default_root
       ? "Leave the folder empty for a new one in your workspace (" + d.default_root + "); a relative folder is taken from there."
       : "Leave the folder empty for a new one in the workspace; a folder must sit under WORKSPACE_ROOTS.";
     _integProjState.list = d.projects || [];
     integRenderProjects();
     var open = _integProjState.open;
     if (open) {
       var still = _integProjState.list.filter(function(p) { return p.id === open; })[0];
       if (still) integOpenProject(still);
       else integCloseProject();
     }
   }

   function integRenderProjects() {
     var host = document.getElementById("integProjList");
     var admin = integIsAdmin();
     integClear(host);
     var list = _integProjState.list;
     if (!list.length) { host.appendChild(integNode("p", "muted", "No projects yet.")); return; }
     var body = integTable(host, ["Project", "Folder", "Your access", "Everyone", ""]);
     list.forEach(function(p) {
       var tr = document.createElement("tr");
       integCell(tr, p.name);
       integCell(tr, p.root || p.root_path || "", "mono");
       integCell(tr, "", "cell-status").appendChild(integNode("span", "badge", p.access || "member"));
       var shared = integCell(tr, "", "cell-status");
       if (admin) {
         // Only administrators open a project to everyone (the server refuses others).
         var sw = integNode("label", "switch");
         var box = document.createElement("input");
         box.type = "checkbox";
         box.checked = !!p.shared;
         box.setAttribute("aria-label", "Open " + p.name + " to every signed-in user");
         box.onchange = function() { integShareProject(p, box.checked, box); };
         sw.appendChild(box);
         sw.appendChild(integNode("span", "slider"));
         shared.appendChild(sw);
       } else {
         shared.textContent = p.shared ? "yes" : "no";
       }
       var act = integCell(tr, "", "cell-action");
       act.appendChild(integButton("Open", "", function() { integOpenProject(p); }));
       if (p.access === "owner" || p.access === "admin") {
         act.appendChild(integButton("Delete", "btn-ghost btn-danger", function() { integDeleteProject(p); }));
       }
       body.appendChild(tr);
     });
   }

   async function integCreateProject() {
     var nameEl = document.getElementById("integProjName");
     var rootEl = document.getElementById("integProjRoot");
     var shared = document.getElementById("integProjShared");
     var name = nameEl.value.trim();
     if (!name) { uiToast("Give the project a name.", "warn"); return; }
     var body = { name: name, root: rootEl.value.trim() };
     if (integIsAdmin() && shared.checked) body.shared = true;
     await uiAct("Could not create the project", async function() {
       var out = await postJSON("/api/projects", body);
       var p = (out.data || {}).project || {};
       nameEl.value = "";
       rootEl.value = "";
       shared.checked = false;
       uiToast("Project created in " + (p.root || p.root_path || "its folder") + ".", "ok");
       await integLoadProjects();
     });
   }

   async function integShareProject(p, on, box) {
     try {
       await postJSON("/api/projects/" + encodeURIComponent(p.id), { shared: !!on });
       p.shared = !!on;
       uiToast(on ? p.name + " is open to every signed-in user." : p.name + " is members only again.", "ok");
     } catch (err) {
       if (box) box.checked = !on;
       uiToast("Could not change sharing: " + err.message, "error");
     }
   }

   async function integDeleteProject(p) {
     if (!uiConfirm("Delete the project " + p.name + "? Its files stay on disk.")) return;
     await uiAct("Could not delete the project", async function() {
       var out = await fetchJSON("/api/projects/" + encodeURIComponent(p.id), { method: "DELETE" });
       var root = (out.data || {}).root;
       uiToast("Project deleted" + (root ? "; its files are still in " + root : "") + ".", "ok");
       if (_integProjState.open === p.id) integCloseProject();
       await integLoadProjects();
     });
   }

   async function integOpenProject(p) {
     _integProjState.open = p.id;
     _integProjState.current = p;
     document.getElementById("integProjDetailTitle").textContent = p.name;
     document.getElementById("integProjDetail").hidden = false;
     var pre = document.getElementById("integProjContext");
     var d;
     try {
       d = (await fetchJSON("/api/projects/" + encodeURIComponent(p.id) + "/context")).data || {};
     } catch (err) {
       pre.textContent = "";
       uiToast("Could not load the project: " + err.message, "error");
       return;
     }
     if (_integProjState.open !== p.id) return;
     var facts = document.getElementById("integProjFacts");
     facts.innerHTML = "";
     integKv(facts, "Folder", d.root || p.root || "");
     integKv(facts, "Your access", p.access || "member");
     integKv(facts, "Open to everyone", p.shared ? "yes" : "no, members only");
     integKv(facts, "Context files", (d.files || []).length ? d.files.join(", ") : "none found");
     integKv(facts, "Context size", (d.chars || 0) + " characters"
       + (d.enabled === false ? " (context files are off on this server)" : ""));
     if (p.rag_scope) integKv(facts, "Knowledge scope", p.rag_scope);
     pre.textContent = d.text || "No context file: the agent gets no project instructions.";
     integRenderMembers(p, d.members || []);
   }

   function integCloseProject() {
     _integProjState.open = null;
     _integProjState.current = null;
     var box = document.getElementById("integProjDetail");
     if (box) box.hidden = true;
   }

   function integRenderMembers(p, members) {
     var host = document.getElementById("integProjMembers");
     var manage = ["admin", "owner", "manager"].indexOf(p.access) >= 0;
     document.getElementById("integProjAddRow").hidden = !manage;
     integClear(host);
     if (!members.length) {
       host.appendChild(integNode("p", "muted", "Only the owner" + (p.shared ? ", and everyone through sharing." : ".")));
       return;
     }
     var body = integTable(host, ["User", "Role", ""]);
     members.forEach(function(m) {
       var tr = document.createElement("tr");
       integCell(tr, m.username || m.user_id);
       integCell(tr, m.role || "member");
       var act = integCell(tr, "", "cell-action");
       if (manage) act.appendChild(integButton("Remove", "btn-ghost btn-danger", function() { integRemoveMember(p, m); }));
       body.appendChild(tr);
     });
   }

   async function integAddMember() {
     var p = _integProjState.current;
     var input = document.getElementById("integProjMember");
     var username = input.value.trim();
     if (!p || !username) { uiToast("Type the username to add.", "warn"); return; }
     var role = document.getElementById("integProjMemberRole").value;
     await uiAct("Could not add the member", async function() {
       var out = await postJSON("/api/projects/" + encodeURIComponent(p.id) + "/members",
                                { username: username, role: role });
       input.value = "";
       integRenderMembers(p, (out.data || {}).members || []);
       uiToast(username + " added as " + role + ".", "ok");
     });
   }

   async function integRemoveMember(p, m) {
     if (!uiConfirm("Remove " + (m.username || m.user_id) + " from " + p.name + "?")) return;
     await uiAct("Could not remove the member", async function() {
       var out = await postJSON("/api/projects/" + encodeURIComponent(p.id) + "/members",
                                { user_id: m.user_id, remove: true });
       integRenderMembers(p, (out.data || {}).members || []);
       uiToast("Member removed.", "ok");
     });
   }

   // ------------------------------------------------------------- boot ---

   integRegisterTab("tools/catalogue", function() { return integLoadCatalogue(); });
   integRegisterTab("tools/approvals", function() { return integLoadApprovalsTab(); },
                    [function() { return integLoadPending(); }, 5000]);
   integRegisterTab("tools/clients", function() { return integLoadClient(); });
   integRegisterTab("tools/tokens", function() { return integLoadTokens(); });
   integRegisterTab("tools/mcp", function() { return integLoadMcpTab(); });
   integRegisterTab("admin/audit", function() { return integLoadAudit(true); });
   integRegisterTab("agent", function() { return integLoadProjects(); });
   integWire();
   document.addEventListener("viewchange", function(e) {
     var d = e.detail || {};
     if (d.from === "settings") integHideNewToken();
   });
"""

__all__ = ["UI_JS_INTEGRATIONS"]
