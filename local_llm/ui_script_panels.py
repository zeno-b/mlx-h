"""Front-end: history, prompt library, uploads, command palette, folder picker and boot.

Part of the embedded UI, split out of the single HTML_PAGE string for
readability. Assembled once at import time in ui.py, so there is no
runtime cost and no extra file I/O.

Theme, backup, models, users, cluster and agents used to live here too. They
are Settings sections now (ui_script_settings.py, ui_script_admin.py,
ui_script_models.py), which is where a user looks for them. What stays is the
chat-side furniture plus the boot sequence, which must run last because it
calls functions from every other script part.

Every result and error goes to a toast (uiToast, from ui_script_kit.py)
instead of alert(): an alert blocks the whole tab, and several of these fire
from background work the user did not just click.
"""

from __future__ import annotations

UI_JS_PANELS = r"""   // ---------------------------------------------------- history panel ---

   function renderConversations(items, isSearch) {
     var host = document.getElementById("historyList");
     host.textContent = "";
     if (!items.length) {
       var empty = document.createElement("div");
       empty.className = "empty";
       empty.textContent = isSearch ? "No conversations matched." : "No conversations yet.";
       host.appendChild(empty);
       return;
     }
     items.forEach(function(c) {
       var id = c.conversation_id || c.id;
       var row = document.createElement("div");
       row.className = "conv-item card";
       var title = document.createElement("div");
       title.className = "ctitle";
       title.textContent = (c.pinned ? "★ " : "") + (c.title || id).toString().slice(0, 90);
       var meta = document.createElement("div");
       meta.className = "cmeta muted mono";
       var when = c.last_at || c.created_at || "";
       var count = (c.messages !== undefined) ? c.messages : null;
       meta.textContent = id.slice(0, 8) + (when ? " · " + when : "")
         + (count !== null ? " · " + count + " messages" : "");
       row.appendChild(title);
       row.appendChild(meta);
       if (c.snippet) {
         var sn = document.createElement("div");
         sn.className = "csnip hint";
         sn.textContent = c.snippet;
         row.appendChild(sn);
       }
       var convUrl = "/api/conversation/" + encodeURIComponent(id);
       var bar = document.createElement("div");
       bar.className = "msg-actions card-actions";
       bar.appendChild(makeBtn("Open", "Reopen this conversation and continue it",
         function() { openConversation(id); }));
       bar.appendChild(makeBtn("Fork", "Copy this conversation so you can take it a different direction",
         async function() {
           try {
             var fr = await fetchJSON(convUrl + "/fork", { method: "POST" });
             uiToast("Forked into a new conversation.", "ok");
             openConversation((fr.data || fr).conversation_id);
           } catch (err) { uiToast("Could not fork: " + err.message, "error"); }
         }));
       bar.appendChild(makeBtn(c.pinned ? "Unpin" : "Pin", "Keep this conversation at the top of the list",
         async function() {
           try {
             await fetchJSON(convUrl + "/pin?pinned=" + (c.pinned ? "false" : "true"), { method: "POST" });
             loadHistory();
           } catch (err) { uiToast("Could not pin: " + err.message, "error"); }
         }));
       bar.appendChild(makeBtn("Rename", "Give this conversation a name", async function() {
         var name = window.prompt("Name this conversation:", c.title || "");
         if (!name) return;
         try {
           await fetchJSON(convUrl + "/title?title=" + encodeURIComponent(name), { method: "POST" });
           loadHistory();
         } catch (err) { uiToast("Could not rename: " + err.message, "error"); }
       }));
       bar.appendChild(makeBtn("Export", "Download as Markdown", function() {
         downloadFile(convUrl + "/export?format=markdown", "conversation-" + id + ".md");
       }));
       bar.appendChild(makeBtn("Delete", "Delete this conversation permanently", async function() {
         if (!(await uiConfirm("Delete this conversation? This cannot be undone."))) return;
         try {
           await fetchJSON(convUrl, { method: "DELETE" });
           // Deleting the open conversation must not leave the chat pointed at
           // an id the server no longer has.
           if (id === conversationId) {
             stopStream();
             conversationId = randomId();
             safeStorageSet("llm_conversation", conversationId);
             chat.innerHTML = "";
             usedTokens = 0;
             updateContextMeter(0);
             showChatEmpty();
           }
           loadHistory();
         } catch (err) { uiToast("Delete failed: " + err.message, "error"); }
       }, "btn-danger"));
       row.appendChild(bar);
       host.appendChild(row);
     });
   }

   async function loadHistory() {
     var host = document.getElementById("historyList");
     if (!host) return;
     host.textContent = "loading...";
     try {
       var r = await fetchJSON("/api/conversations");
       renderConversations((r.data || r).conversations || [], false);
     } catch (err) {
       host.textContent = "Could not load conversations: " + err.message;
     }
   }

   async function runHistorySearch() {
     var q = (document.getElementById("historySearch").value || "").trim();
     if (!q) { loadHistory(); return; }
     var host = document.getElementById("historyList");
     host.textContent = "searching...";
     try {
       var r = await fetchJSON("/api/conversations/search?q=" + encodeURIComponent(q));
       renderConversations((r.data || r).results || [], true);
     } catch (err) {
       host.textContent = "Search failed: " + err.message;
     }
   }

   function clearHistorySearch() {
     var box = document.getElementById("historySearch");
     if (box) box.value = "";
     loadHistory();
   }

   // "Search" lives in one place now: the History view's search box (the
   // header button, the chat menu and the palette all land here).
   function openHistorySearch() {
     showView("history");
     if (currentView !== "history") return;
     var box = document.getElementById("historySearch");
     if (box) { box.focus(); box.select(); }
   }

   // Render a stored conversation into #chat. Reopening from History and the
   // boot replay share this path. Boot used to show an empty screen while the
   // composer silently continued the stored conversation, so the model
   // answered with context the user could not see.
   async function openConversation(id, opts) {
     // Same reason as newChat: a turn still streaming into the chat we are
     // about to clear keeps running against detached nodes, leaves Send
     // disabled, and charges its tokens and its feedback bar to whichever
     // conversation happens to be open when it finishes.
     stopStream();
     var quiet = !!(opts && opts.quiet);
     try {
       var r = await fetchJSON("/api/conversation/" + encodeURIComponent(id));
       var d = r.data || r;
       var msgs = d.messages || [];
       conversationId = id;
       safeStorageSet("llm_conversation", id);
       chat.innerHTML = "";
       var lastAnswer = null;
       msgs.forEach(function(m) {
         if (m.role !== "user" && m.role !== "assistant") return;
         var div = addMessage(m.role, m.content || "");
         if (m.role === "assistant") lastAnswer = div;
       });
       if (lastAnswer) setRegenerateTarget(lastAnswer.querySelector(".msg-actions"));
       if (!msgs.length) showChatEmpty();
       if (typeof d.context_size === "number" && d.context_size > 0) contextSize = d.context_size;
       usedTokens = Number(d.est_tokens) || 0;
       updateContextMeter(0);
       if (!quiet) showView("chat");
       scrollDown(true);
     } catch (err) {
       if (!quiet) { uiToast("Could not open conversation: " + err.message, "error"); return; }
       // Boot replay of an id the server does not know for this user (another
       // account signed in on this browser, or the row was deleted): start a
       // fresh conversation rather than posting into someone else's id.
       if (err.status === 404 || err.status === 403) {
         conversationId = randomId();
         safeStorageSet("llm_conversation", conversationId);
       }
       chat.innerHTML = "";
       showChatEmpty();
     }
   }

   // ---------------------------------------------------- prompt library ---

   function togglePrompts() {
     var panel = document.getElementById("promptsPanel");
     panel.classList.toggle("hidden");
     var btn = document.getElementById("promptsBtn");
     var open = !panel.classList.contains("hidden");
     if (btn) btn.setAttribute("aria-pressed", open ? "true" : "false");
     if (open) loadPrompts();
   }

   async function loadPrompts() {
     var host = document.getElementById("promptList");
     if (!host) return;
     host.textContent = "loading...";
     try {
       var r = await fetchJSON("/api/prompts");
       var items = (r.data || r).prompts || [];
       host.textContent = "";
       if (!items.length) {
         var empty = document.createElement("div");
         empty.className = "empty";
         empty.textContent = "No saved prompts yet.";
         host.appendChild(empty);
         return;
       }
       items.forEach(function(pr) {
         var row = document.createElement("div");
         row.className = "prompt-item card";
         var n = document.createElement("div");
         n.className = "pname";
         n.textContent = pr.name;
         var b = document.createElement("div");
         b.className = "pbody muted";
         b.textContent = pr.body;
         var bar = document.createElement("div");
         bar.className = "msg-actions card-actions";
         bar.appendChild(makeBtn("Insert", "Put this prompt in the message box", function() {
           showView("chat");
           input.value = pr.body;
           input.dispatchEvent(new Event("input"));
           input.focus();
         }));
         bar.appendChild(makeBtn("Delete", "Delete this saved prompt", async function() {
           try {
             await fetchJSON("/api/prompts/" + encodeURIComponent(pr.name), { method: "DELETE" });
             loadPrompts();
           } catch (err) { uiToast("Delete failed: " + err.message, "error"); }
         }, "btn-danger"));
         row.appendChild(n); row.appendChild(b); row.appendChild(bar);
         host.appendChild(row);
       });
     } catch (err) {
       host.textContent = "Could not load prompts: " + err.message;
     }
   }

   async function savePrompt() {
     var nameBox = document.getElementById("promptName");
     var name = (nameBox.value || "").trim();
     var body = (input.value || "").trim();
     if (!name) { uiToast("Give the prompt a name.", "warn"); nameBox.focus(); return; }
     if (!body) { uiToast("Type the prompt in the message box first, then save it.", "warn"); return; }
     try {
       await fetchJSON("/api/prompts?name=" + encodeURIComponent(name) + "&body=" + encodeURIComponent(body),
                       { method: "POST" });
       nameBox.value = "";
       uiToast("Saved prompt “" + name + "”.", "ok");
       loadPrompts();
     } catch (err) {
       uiToast("Could not save: " + err.message, "error");
     }
   }

   // ------------------------------------------------- file drag & drop ---

   async function uploadFiles(files) {
     if (!files || !files.length) return;
     for (var i = 0; i < files.length; i++) {
       var form = new FormData();
       form.append("file", files[i]);
       try {
         var resp = await fetch("/api/docs/upload", { method: "POST", body: form });
         if (resp.status === 401 && typeof onUnauthorized === "function") { onUnauthorized(); return; }
         var data = {};
         try { data = await resp.json(); } catch (e) { data = {}; }
         if (!resp.ok || data.error) {
           uiToast(files[i].name + ": " + httpErrorText(data, resp.status), "error");
           continue;
         }
         uiToast(data.result || ("Indexed " + files[i].name), "ok");
       } catch (err) {
         uiToast("Upload failed for " + files[i].name + ": " + err.message, "error");
       }
     }
     loadDocsStats();
   }

   function wireDropZone() {
     var zone = document.getElementById("dropZone");
     var picker = document.getElementById("fileInput");
     if (!zone || !picker) return;
     zone.onclick = function() { picker.click(); };
     zone.onkeydown = function(e) {
       if (e.key === "Enter" || e.key === " ") { e.preventDefault(); picker.click(); }
     };
     picker.onchange = function() { uploadFiles(picker.files); picker.value = ""; };
     ["dragenter", "dragover"].forEach(function(ev) {
       zone.addEventListener(ev, function(e) {
         e.preventDefault(); e.stopPropagation(); zone.classList.add("dragging");
       });
     });
     ["dragleave", "drop"].forEach(function(ev) {
       zone.addEventListener(ev, function(e) {
         e.preventDefault(); e.stopPropagation(); zone.classList.remove("dragging");
       });
     });
     zone.addEventListener("drop", function(e) {
       if (e.dataTransfer && e.dataTransfer.files) uploadFiles(e.dataTransfer.files);
     });
   }

   // ------------------------------------------------- command palette ---

   // Every Settings destination is reachable by name, so nobody has to know
   // which section or sub-tab a control lives in. Admin rows are filtered out
   // for other roles (the server enforces it anyway).
   var PALETTE_COMMANDS = [
     { label: "New chat", hint: "start a fresh conversation", run: function() { newChat(); } },
     { label: "Regenerate last answer", hint: "same question, new answer", run: function() { regenerateLast(); } },
     { label: "Search conversations", hint: "history", run: function() { openHistorySearch(); } },
     { label: "Export this conversation", hint: "markdown", run: function() { exportChat(); } },
     { label: "Prompt library", hint: "saved prompts", run: function() { togglePrompts(); } },
     { label: "Mode: Auto", hint: "server decides when to use tools", run: function() { setChatMode("auto"); } },
     { label: "Mode: Chat", hint: "plain reply, no tools", run: function() { setChatMode("chat"); } },
     { label: "Mode: Agent", hint: "tools and step-by-step reasoning", run: function() { setChatMode("agent"); } },
     { label: "Chat", hint: "back to the conversation", run: function() { showView("chat"); } },
     { label: "History", hint: "past conversations", run: function() { showView("history"); } },
     { label: "Tasks", hint: "scheduled jobs", admin: true, run: function() { showView("tasks"); } },
     { label: "Settings", hint: "every option in one place", run: function() { openSettings(); } },
     { label: "Appearance", hint: "settings: theme, density, text size", run: function() { openSettings("appearance"); } },
     { label: "Toggle theme", hint: "light / dark", run: function() { toggleTheme(); } },
     { label: "Generation", hint: "settings: temperature, reply length", run: function() { openSettings("generation"); } },
     { label: "Agent", hint: "settings: default mode, steps", run: function() { openSettings("agent"); } },
     { label: "Memory", hint: "settings: notes the assistant keeps", run: function() { openSettings("memory"); } },
     { label: "Tools", hint: "settings: tool catalogue", run: function() { openSettings("tools"); } },
     { label: "Model", hint: "settings: switch model, adapter", admin: true, run: function() { openSettings("models", "model"); } },
     { label: "Training", hint: "settings: LoRA retrain, dataset", admin: true, run: function() { openSettings("models", "training"); } },
     { label: "Knowledge base", hint: "settings: index documents", admin: true, run: function() { openSettings("models", "knowledge"); } },
     { label: "Performance", hint: "settings: speed levers, stats", admin: true, run: function() { openSettings("performance"); } },
     { label: "Users", hint: "admin: accounts and roles", admin: true, run: function() { openSettings("admin", "users"); } },
     { label: "Agents", hint: "admin: what each agent can do", admin: true, run: function() { openSettings("admin", "agents"); } },
     { label: "Cluster & routing", hint: "admin: node health", admin: true, run: function() { openSettings("admin", "cluster"); } },
     { label: "Import", hint: "admin: chat export archive", admin: true, run: function() { openSettings("admin", "import"); } },
     { label: "Codebase", hint: "admin: project folder", admin: true, run: function() { openSettings("admin", "codebase"); } },
     { label: "Backup & restore", hint: "admin: save or restore everything", admin: true, run: function() { openSettings("admin", "backup"); } },
     { label: "Download backup", hint: "admin: save everything now", admin: true, run: function() { downloadBackup(); } },
     { label: "Logs", hint: "admin: model server log", admin: true, run: function() { openSettings("admin", "logs"); } },
     { label: "Restart model server", hint: "admin: reload the model", admin: true, run: async function() {
         if (await uiConfirm("Restart the model server? Answers in progress stop.")) restartModel();
       } }
   ];
   var paletteSel = 0;

   function renderPalette() {
     var q = (document.getElementById("paletteInput").value || "").toLowerCase();
     var list = document.getElementById("paletteList");
     var isAdmin = roleIsAdmin();
     var matches = PALETTE_COMMANDS.filter(function(c) {
       if (c.admin && !isAdmin) return false;  // never expose admin actions to a non-admin
       return !q || c.label.toLowerCase().indexOf(q) >= 0 || c.hint.toLowerCase().indexOf(q) >= 0;
     });
     if (paletteSel >= matches.length) paletteSel = 0;
     list.textContent = "";
     matches.forEach(function(c, i) {
       var row = document.createElement("div");
       row.className = "pal-item" + (i === paletteSel ? " sel" : "");
       row.setAttribute("role", "option");
       row.setAttribute("aria-selected", i === paletteSel ? "true" : "false");
       row.textContent = c.label;
       var hint = document.createElement("span");
       hint.className = "palhint muted";
       hint.textContent = c.hint;
       row.appendChild(hint);
       row.onclick = function() { closePalette(); c.run(); };
       list.appendChild(row);
     });
     if (!matches.length) {
       var none = document.createElement("div");
       none.className = "empty";
       none.textContent = "No matching command.";
       list.appendChild(none);
     }
     list.dataset.count = matches.length;
     return matches;
   }

   function openPalette() {
     document.getElementById("palette").classList.remove("hidden");
     var box = document.getElementById("paletteInput");
     box.value = ""; paletteSel = 0; renderPalette(); box.focus();
   }

   function closePalette() {
     document.getElementById("palette").classList.add("hidden");
   }

   // --------------------------------------------------- folder picker ---

   var browsePath = "";

   async function openBrowser(start) {
     document.getElementById("browser").classList.remove("hidden");
     var dirBox = document.getElementById("cfgProjectDir");
     await loadBrowse(start || (dirBox ? (dirBox.value || "").trim() : ""));
   }

   function closeBrowser() {
     document.getElementById("browser").classList.add("hidden");
   }

   function browseRow(label, onclick, tag) {
     var row = document.createElement("button");
     row.type = "button";
     row.className = "dir-item btn-ghost";
     var name = document.createElement("span");
     name.textContent = label;
     row.appendChild(name);
     if (tag) {
       var badge = document.createElement("span");
       badge.className = "badge repo";
       badge.textContent = tag;
       row.appendChild(badge);
     }
     if (onclick) row.onclick = onclick; else row.disabled = true;
     return row;
   }

   async function loadBrowse(path) {
     var list = document.getElementById("browserList");
     var label = document.getElementById("browserPath");
     list.textContent = "loading...";
     try {
       var r = await fetchJSON("/api/browse?path=" + encodeURIComponent(path || ""));
       var d = r.data || r;
       if (d.error) { list.textContent = d.error; return; }
       browsePath = d.path;
       label.textContent = d.path + (d.is_git_repo ? "   (git repository)" : "");
       list.textContent = "";
       if (d.parent) list.appendChild(browseRow("../", function() { loadBrowse(d.parent); }));
       if (!d.entries.length) list.appendChild(browseRow("(no sub-folders)", null));
       d.entries.forEach(function(e) {
         list.appendChild(browseRow(e.name + "/", function() { loadBrowse(e.path); },
                                    e.is_git_repo ? "git repo" : ""));
       });
     } catch (err) {
       list.textContent = "Could not browse: " + err.message;
     }
   }

   function chooseCurrentFolder() {
     if (!browsePath) return;
     var dirBox = document.getElementById("cfgProjectDir");
     if (dirBox) dirBox.value = browsePath;
     closeBrowser();
     saveProjectDir();
   }

   // ------------------------------------------------------------- boot ---

   wireDropZone();
   // Theme and the backup restore picker belong to Settings (Appearance and
   // Admin > Backup); they still have to be wired before anything renders.
   if (typeof initTheme === "function") initTheme();
   if (typeof wireRestoreInput === "function") wireRestoreInput();

   // Cmd/Ctrl+K opens the command palette; arrows and Enter drive it, Esc closes.
   document.addEventListener("keydown", function(e) {
     var pal = document.getElementById("palette");
     var open = pal && !pal.classList.contains("hidden");
     if ((e.metaKey || e.ctrlKey) && e.key && e.key.toLowerCase() === "k") {
       e.preventDefault();
       if (open) closePalette(); else openPalette();
       return;
     }
     if (!open) {
       // Esc also closes the other overlays, so no panel needs its own button
       // to get out of it.
       if (e.key === "Escape") {
         var browser = document.getElementById("browser");
         if (browser && !browser.classList.contains("hidden")) { closeBrowser(); return; }
         var prompts = document.getElementById("promptsPanel");
         if (prompts && !prompts.classList.contains("hidden")) togglePrompts();
       }
       return;
     }
     if (e.key === "Escape") { e.preventDefault(); closePalette(); return; }
     var matches = renderPalette();
     if (e.key === "ArrowDown") {
       e.preventDefault(); paletteSel = Math.min(paletteSel + 1, matches.length - 1); renderPalette();
     } else if (e.key === "ArrowUp") {
       e.preventDefault(); paletteSel = Math.max(paletteSel - 1, 0); renderPalette();
     } else if (e.key === "Enter") {
       e.preventDefault();
       var chosen = matches[paletteSel];
       closePalette();
       if (chosen) chosen.run();
     }
   });
   var _palInput = document.getElementById("paletteInput");
   if (_palInput) _palInput.addEventListener("input", function() { paletteSel = 0; renderPalette(); });
   var _palOverlay = document.getElementById("palette");
   if (_palOverlay) _palOverlay.addEventListener("click", function(e) {
     if (e.target === _palOverlay) closePalette();
   });
   var _browserOverlay = document.getElementById("browser");
   if (_browserOverlay) _browserOverlay.addEventListener("click", function(e) {
     if (e.target === _browserOverlay) closeBrowser();
   });

   // Resolve auth, show the login screen if required, then load the data that
   // suits the signed-in user's role. Defined in the auth script part. It runs
   // last because it calls into every other part.
   authBoot();
 """

__all__ = ["UI_JS_PANELS"]
