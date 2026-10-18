"""Front-end: training data, knowledge base, codebase and chat-export helpers.

Part of the embedded UI, split out of the single HTML_PAGE string for
readability. Assembled once at import time in ui.py, so there is no
runtime cost and no extra file I/O.

The panels these functions fill now live in Settings (Models & Training >
Training / Knowledge, Admin > Codebase; markup in ui_settings.py, lazy loading
in ui_script_admin.py). Results and failures are toasts (uiToast from the
kit), not alert(): a blocking dialog for a routine "Indexed." froze the tab.
"""

from __future__ import annotations

UI_JS_MODELS = r"""   // ------------------------------------------------------------ models ---

   async function loadDatasetStats() {
     var box = document.getElementById("datasetStats");
     if (!box) return;
     try {
       var s = (await fetchJSON("/api/dataset/stats")).data;
       var bySource = Object.keys(s.by_source || {})
         .map(function(k) { return k + ": " + s.by_source[k]; }).join(", ") || "none";
       var pending = s.pending || 0;
       // Pad to the longest label instead of hand-counting spaces, which had
       // already drifted out of alignment by one column.
       var rows = [
         ["Total examples", s.total],
         ["Approved (to imitate)", s.approved],
         ["Awaiting approval", pending],
         ["Rejected (bad answers)", s.rejected],
         ["Corrections", s.corrected],
         ["Preference pairs", s.preference_pairs],
         ["Reviewed (curated)", s.reviewed],
         ["By source", bySource]
       ];
       var width = rows.reduce(function(w, r) { return Math.max(w, r[0].length); }, 0) + 2;
       box.textContent = rows.map(function(r) {
         return (r[0] + ":" + " ".repeat(width - r[0].length)) + r[1];
       }).join("\n");
       // The queue holds ratings from non-admin accounts, which never enter the
       // shared adapter on their own. Only offer the button when it has work.
       var btn = document.getElementById("approvePendingBtn");
       if (btn) {
         btn.hidden = pending === 0;
         btn.textContent = "approve " + pending + " queued";
       }
     } catch (err) {
       box.textContent = "Could not load dataset stats: " + err.message;
     }
   }

   async function approvePendingFeedback() {
     var btn = document.getElementById("approvePendingBtn");
     if (btn) btn.disabled = true;
     try {
       var out = (await fetchJSON("/api/feedback/approve-pending", { method: "POST" })).data;
       uiToast("Approved " + out.approved + " example(s) for training.", "ok");
     } catch (err) {
       uiToast("Could not approve: " + err.message, "error");
     } finally {
       if (btn) btn.disabled = false;
       loadDatasetStats();
     }
   }

   async function exportDataset(format) {
     var reviewed = document.getElementById("exportReviewedOnly");
     var url = "/api/dataset/export?format=" + encodeURIComponent(format)
       + "&reviewed_only=" + (reviewed && reviewed.checked ? "true" : "false");
     // Not uiDownload: the example count comes back in a response header.
     try {
       var resp = await fetch(url);
       if (resp.status === 401 && typeof onUnauthorized === "function") { onUnauthorized(); return; }
       if (!resp.ok) {
         var e = {};
         try { e = await resp.json(); } catch (x) { e = {}; }
         throw new Error(typeof httpErrorText === "function" ? httpErrorText(e, resp.status)
                                                             : ("HTTP " + resp.status));
       }
       var count = resp.headers.get("X-Example-Count") || "?";
       var blob = await resp.blob();
       var a = document.createElement("a");
       a.href = URL.createObjectURL(blob);
       a.download = "dataset_" + format + ".jsonl";
       document.body.appendChild(a); a.click(); a.remove();
       setTimeout(function() { URL.revokeObjectURL(a.href); }, 1500);
       uiToast(count + " examples exported as " + format + " (also saved to data/exports/).", "ok");
     } catch (err) {
       uiToast("Export failed: " + err.message, "error");
     }
   }

   async function loadProjectStatus() {
     var box = document.getElementById("projectStatus");
     if (!box) return;
     try {
       var s = (await fetchJSON("/api/project/status")).data;
       if (!s.project_dir) {
         box.textContent = "No project set: file tools use the sandbox workspace.\n"
           + "Set a project directory above to work on a local codebase.";
         var inp = document.getElementById("cfgProjectDir");
         if (inp && !inp.value) inp.value = "";
         return;
       }
       var lines = [];
       lines.push("Directory: " + s.root);
       lines.push("Git repo:  " + (s.is_git_repo ? ("yes (branch " + (s.branch || "?") + ")")
         : "NO: you will have no easy way to review or revert edits"));
       var changed = s.changed_this_session || [];
       lines.push("Changed this session: " + (changed.length ? changed.join(", ") : "none"));
       if (s.is_git_repo) {
         var st = (s.status || "").trim();
         lines.push("");
         lines.push(st ? ("Uncommitted (git status):\n" + st) : "Working tree clean.");
       }
       box.textContent = lines.join("\n");
       var hint = document.getElementById("sandboxHint");
       if (hint) {
         var exec = [];
         if (s.allow_shell) exec.push("shell");
         if (s.allow_python) exec.push("python");
         if (exec.length) {
           hint.textContent = "Execution enabled (" + exec.join(" + ")
             + "), confined to this directory with a timeout. The agent can run "
             + (s.test_command ? ("tests (" + s.test_command + "), ") : "")
             + "read the output, and fix the code. Not a security sandbox: it runs "
             + "with your privileges, so use it only on code you trust and review.";
         } else {
           hint.textContent = "Execution disabled. Start the app with --allow-shell "
             + "(and/or --allow-python) to let the agent run and test the code, then "
             + "iterate on failures. Confined to this directory; review with git.";
         }
       }
       var inp2 = document.getElementById("cfgProjectDir");
       if (inp2 && !inp2.value) inp2.value = s.project_dir;
     } catch (err) {
       box.textContent = "Could not load project status: " + err.message;
     }
   }

   async function loadProjectDiff() {
     var pre = document.getElementById("projectDiff");
     if (!pre) return;
     try {
       var d = (await fetchJSON("/api/project/diff")).data;
       modelsShow(pre);
       if (!d.project_dir) {
         pre.textContent = "No project set, so nothing to diff. Set a project directory above.";
         return;
       }
       if (!d.is_git_repo) { pre.textContent = "Not a git repository: no diff available."; return; }
       pre.textContent = (d.diff && d.diff.trim()) ? d.diff : "No uncommitted changes.";
     } catch (err) {
       modelsShow(pre);
       pre.textContent = "Could not load diff: " + err.message;
     }
   }

   async function saveProjectDir() {
     var val = (document.getElementById("cfgProjectDir").value || "").trim();
     try {
       var out = await postJSON("/api/config", { project_dir: val });
       // Confirm the server actually took it. This used to alert success
       // unconditionally, which hid the fact that the field was being dropped.
       var d = out.data || {};
       var applied = (d.config && d.config.project_dir) || "";
       // Compare on "did the server end up with a project set", not on string
       // equality: the server resolves ~ and symlinks, so /var/... legitimately
       // comes back as /private/var/... .
       if (val && !applied) {
         uiToast("The server did not apply that project directory. It reported: "
                 + (applied || "(none)"), "error");
       } else {
         uiToast(val ? ("Project set to " + applied)
                     : "Project cleared: using the sandbox workspace.", "ok");
       }
       loadProjectStatus();
     } catch (err) {
       uiToast("Could not set project directory: " + err.message, "error");
     }
   }

   async function revertChanges() {
     if (!uiConfirm("Undo this session's edits? Modified files are restored to the "
       + "last commit and files created this session are deleted. This cannot be undone.")) return;
     try {
       var r = await fetchJSON("/api/project/revert", { method: "POST" });
       if (r.data && r.data.error) { uiToast(r.data.error, "error"); return; }
       var d = r.data || r;
       uiToast("Restored " + (d.restored || []).length + ", removed " + (d.removed || []).length
         + (d.failed && d.failed.length ? (", failed " + d.failed.length) : "") + ".",
         d.failed && d.failed.length ? "warn" : "ok");
       loadProjectStatus();
     } catch (err) {
       uiToast("Revert failed: " + err.message, "error");
     }
   }

   async function loadDocsStats() {
     var box = document.getElementById("docsStats");
     if (!box) return;
     try {
       var s = (await fetchJSON("/api/docs/stats")).data;
       if (!s.enabled) { box.textContent = "Knowledge base unavailable."; return; }
       var lines = ["Documents: " + s.documents, "Passages:  " + s.chunks];
       if (s.search_mode && s.search_mode !== "fts5") {
         lines.push("Search:    built-in ranking (this Python's SQLite has no FTS5)");
       }
       if (s.items && s.items.length) {
         lines.push("");
         lines.push("Indexed:");
         s.items.slice(0, 15).forEach(function(i) {
           lines.push("  " + i.path + "  (" + i.chunks + " passages)");
         });
         // Count from the authoritative document total, not from `items` (which
         // the API caps), or the two numbers on screen contradict each other.
         var total = (typeof s.documents === "number") ? s.documents : s.items.length;
         if (total > 15) lines.push("  ... and " + (total - 15) + " more");
       } else {
         lines.push("");
         lines.push("Nothing indexed yet: answers use the model's own knowledge.");
       }
       box.textContent = lines.join("\n");
       renderScope(s.items || []);
     } catch (err) {
       box.textContent = "Could not load knowledge base stats: " + err.message;
     }
   }

   async function renderScope(items) {
     var host = document.getElementById("docScope");
     if (!host) return;
     var current = [];
     try {
       var sc = await fetchJSON("/api/docs/scope");
       current = ((sc.data || sc).scope || "").split(",").map(function(x) { return x.trim(); })
                   .filter(Boolean);
     } catch (err) { current = []; }
     if (!items.length) { host.textContent = "Nothing indexed yet."; return; }
     host.innerHTML = "";
     items.forEach(function(it) {
       // The kit's checkbox row (.check), same as the capability grid.
       var row = document.createElement("label");
       row.className = "check";
       var cb = document.createElement("input");
       cb.type = "checkbox";
       cb.value = it.path;
       cb.checked = current.indexOf(it.path) >= 0;
       var text = document.createElement("span");
       text.className = "mono";
       text.textContent = it.path;
       row.appendChild(cb);
       row.appendChild(text);
       host.appendChild(row);
     });
   }

   async function applyScope() {
     var boxes = document.querySelectorAll("#docScope input[type=checkbox]");
     var picked = [];
     boxes.forEach(function(b) { if (b.checked) picked.push(b.value); });
     try {
       await fetchJSON("/api/docs/scope?paths=" + encodeURIComponent(picked.join(",")),
                       { method: "POST" });
       uiToast(picked.length
         ? ("Answers will use only these " + picked.length + " document(s).")
         : "No documents ticked: using the whole knowledge base.", "ok");
     } catch (err) {
       uiToast("Could not set scope: " + err.message, "error");
     }
   }

   async function clearScope() {
     try {
       await fetchJSON("/api/docs/scope?paths=", { method: "POST" });
       loadDocsStats();
       uiToast("Using the whole knowledge base.", "ok");
     } catch (err) {
       uiToast("Could not clear scope: " + err.message, "error");
     }
   }

   async function indexDocs() {
     var path = (document.getElementById("docsPath").value || "").trim();
     if (!path) { uiToast("Enter a file or folder path relative to the project.", "warn"); return; }
     try {
       var r = await fetchJSON("/api/docs/index?path=" + encodeURIComponent(path), { method: "POST" });
       var d = r.data || r;
       if (d.error) { uiToast(d.error, "error"); return; }
       uiToast(d.result || "Indexed.", "ok");
       loadDocsStats();
     } catch (err) {
       uiToast("Indexing failed: " + err.message, "error");
     }
   }

   async function searchDocs() {
     var q = (document.getElementById("docsQuery").value || "").trim();
     var pre = document.getElementById("docsResult");
     if (!q) { uiToast("Type a question to test retrieval.", "warn"); return; }
     try {
       var r = await fetchJSON("/api/docs/search?q=" + encodeURIComponent(q));
       var hits = (r.data || r).hits || [];
       modelsShow(pre);
       pre.textContent = hits.length
         ? hits.map(function(h) { return "[" + h.path + "]\n" + h.chunk; }).join("\n\n")
         : "No matching passages.";
     } catch (err) {
       modelsShow(pre);
       pre.textContent = "Search failed: " + err.message;
     }
   }

   async function clearDocs() {
     if (!uiConfirm("Remove every indexed document? Your actual files are not touched.")) return;
     try {
       var r = await fetchJSON("/api/docs/clear", { method: "POST" });
       uiToast("Cleared " + ((r.data || r).cleared || 0) + " document(s).", "ok");
       loadDocsStats();
     } catch (err) {
       uiToast("Clear failed: " + err.message, "error");
     }
   }

   async function indexUrl() {
     var url = (document.getElementById("docsUrl").value || "").trim();
     if (!url) { uiToast("Enter a URL to fetch and index.", "warn"); return; }
     try {
       var r = await fetchJSON("/api/docs/index_url?url=" + encodeURIComponent(url), { method: "POST" });
       var d = r.data || r;
       if (d.error) { uiToast(d.error, "error"); return; }
       uiToast(d.result || "Indexed.", "ok");
       loadDocsStats();
     } catch (err) {
       uiToast("Indexing failed: " + err.message, "error");
     }
   }

   async function regenerateLast() {
     if (busy) { uiToast("Wait for the current answer to finish.", "warn"); return; }
     try {
       var r = await fetchJSON("/api/conversation/" + encodeURIComponent(conversationId) + "/regenerate",
                               { method: "POST" });
       var d = r.data || r;
       if (d.error) { uiToast(d.error, "error"); return; }
       // The server's regenerate drops BOTH sides of the exchange, so the page
       // must too: removing only the answer left the old question on screen and
       // send() then rendered it a second time (plus a stranded feedback bar).
       var kids = Array.prototype.slice.call(chat.children);
       var droppedAnswer = false;
       for (var i = kids.length - 1; i >= 0; i--) {
         var el = kids[i];
         if (el.classList.contains("feedback")) { el.remove(); continue; }
         if (!droppedAnswer && (el.classList.contains("turn")
                                || el.classList.contains("assistant"))) {
           el.remove(); droppedAnswer = true; continue;
         }
         if (droppedAnswer && el.classList.contains("user")) { el.remove(); break; }
       }
       input.value = d.prompt;
       send();
     } catch (err) {
       uiToast("Could not regenerate: " + err.message, "error");
     }
   }

   // Reveal a result box that ships hidden (class + attribute, so neither a
   // stylesheet rule nor the UA default can keep it hidden).
   function modelsShow(el) {
     if (!el) return;
     el.classList.remove("hidden");
     el.hidden = false;
   }

   function exportChat() {
     downloadFile("/api/conversation/" + encodeURIComponent(conversationId) + "/export?format=markdown",
                  "conversation-" + conversationId + ".md");
   }
"""

__all__ = ["UI_JS_MODELS"]
