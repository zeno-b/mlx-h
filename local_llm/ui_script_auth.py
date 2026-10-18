"""Front-end auth: login gate, role gating, logout, and the admin import panel.

Part of the single embedded <script>. It owns application boot: authBoot() runs
last (called from the panels boot), decides whether to show the login overlay or
the app, applies the signed-in user's role to the DOM (so admin-only chrome is
hidden for non-admins), and then loads the data appropriate to that role.

Boot fails CLOSED. It used to treat an unreachable /api/auth/me as "auth off"
and reveal the full admin chrome to whoever opened the tab while the server was
restarting; now the app stays behind the lock with an error and retries.

Session auth is cookie-based, so every fetch/EventSource/download already carries
credentials; there is nothing to add to individual requests here.
"""

from __future__ import annotations

UI_JS_AUTH = r"""
   // ---------------------------------------------------------------- auth ---
   var _authRetryTimer = null;

   async function authBoot() {
     if (_authRetryTimer) { clearTimeout(_authRetryTimer); _authRetryTimer = null; }
     var info;
     try {
       var out = await fetchJSON("/api/auth/me");
       info = (out && out.data) || {};
     } catch (err) {
       authBootFailed("Cannot reach the server (" + err.message + ").");
       return;
     }
     window.AUTH_ENABLED = !!info.auth_enabled;
     window.OIDC_ENABLED = !!info.oidc_enabled;
     if (info.auth_enabled && !info.authenticated) {
       showLogin(info);
       return;
     }
     // Without a user only an explicit "auth is off" from the server means the
     // single local admin. Anything else is a malformed answer, not a licence
     // to show admin controls.
     var user = info.user || (info.auth_enabled === false ? { role: "admin", username: "local" } : null);
     if (!user) {
       authBootFailed("The server did not say who is signed in.");
       return;
     }
     var err0 = document.getElementById("loginError");
     if (err0) err0.textContent = "";
     // Apply the role BEFORE revealing the app, so admin-only chrome does not
     // pop in a frame after the rest of the UI.
     applyRole(user);
     hideLogin();
     runDataBoot(user.role === "admin");
   }

   // Stay locked, say why, and try again: a server restart is the common
   // cause and resolves itself within seconds.
   function authBootFailed(message) {
     showLogin({ oidc_enabled: window.OIDC_ENABLED });
     var status = document.getElementById("loginError");
     if (status) status.textContent = message + " Retrying in 5 seconds...";
     _authRetryTimer = setTimeout(authBoot, 5000);
   }

   function showLogin(info) {
     // Leave the boot state (nothing shown) for the login box, in one step.
     document.body.classList.remove("booting");
     document.body.classList.add("locked");
     var oidc = document.getElementById("oidcBtn");
     if (oidc) oidc.classList.toggle("hidden", !(info && info.oidc_enabled));
     var u = document.getElementById("loginUser");
     if (u) u.focus();
   }

   function hideLogin() {
     document.body.classList.remove("booting");
     document.body.classList.remove("locked");
   }

   function applyRole(user) {
     document.body.dataset.role = user.role || "user";
     var chip = document.getElementById("userChip");
     if (chip) chip.textContent = user.username || "local";
     var info = document.getElementById("acctInfo");
     if (info) {
       info.textContent = "";
       var line = document.createElement("div");
       var who = document.createElement("strong");
       who.textContent = user.username || "local";
       line.appendChild(who);
       var role = document.createElement("span");
       role.className = "badge";
       role.textContent = user.role || "user";
       line.appendChild(document.createTextNode(" "));
       line.appendChild(role);
       info.appendChild(line);
       if (!window.AUTH_ENABLED) {
         var note = document.createElement("p");
         note.className = "hint";
         note.textContent = "Single-user mode: sign-in is off. Start the server with "
           + "AUTH_ENABLED=1 to turn on accounts, roles and logout.";
         info.appendChild(note);
       }
     }
     // With auth disabled there is no session to end, so hide the logout action
     // (the account menu still explains the single-user state above).
     var logout = document.getElementById("logoutBtn");
     if (logout) logout.classList.toggle("hidden", !window.AUTH_ENABLED);
   }

   function toggleAcct(force) {
     var pop = document.getElementById("acctPop");
     var btn = document.getElementById("acctBtn");
     if (!pop) return;
     var open = (typeof force === "boolean") ? force : pop.classList.contains("hidden");
     pop.classList.toggle("hidden", !open);
     if (btn) btn.setAttribute("aria-expanded", open ? "true" : "false");
   }

   // Reveal/hide a password field, flipping the little show/hide button's label.
   function togglePw(id, btn) {
     var el = document.getElementById(id);
     if (!el) return;
     var reveal = el.type === "password";
     el.type = reveal ? "text" : "password";
     if (btn) btn.textContent = reveal ? "hide" : "show";
   }

   // Close the account menu on an outside click or Escape.
   document.addEventListener("click", function(e) {
     var pop = document.getElementById("acctPop");
     if (!pop || pop.classList.contains("hidden")) return;
     if (!e.target.closest || !e.target.closest("#acctMenu")) toggleAcct(false);
   });
   document.addEventListener("keydown", function(e) {
     if (e.key !== "Escape") return;
     var pop = document.getElementById("acctPop");
     if (pop && !pop.classList.contains("hidden")) toggleAcct(false);
   });

   // What every role needs at boot. Admin data (config, perf, imports, the
   // model list) is NOT fetched here any more: each Settings section loads
   // its own data when it is opened, so a non-admin never fires admin routes
   // and an admin who only chats does not pay for panels nobody looks at.
   function runDataBoot(isAdmin) {
     refreshHealth();
     if (window._healthTimer) clearInterval(window._healthTimer);
     window._healthTimer = setInterval(refreshHealth, 3000);
     loadChatPrefs();
     // Show the conversation the composer is about to continue.
     openConversation(conversationId, { quiet: true });
     loadMemory();
   }

   async function doLogin() {
     var status = document.getElementById("loginError");
     var user = document.getElementById("loginUser").value.trim();
     var pass = document.getElementById("loginPass").value;
     if (!user || !pass) { if (status) status.textContent = "Enter a username and password."; return; }
     if (status) status.textContent = "";
     var btn = document.getElementById("loginBtn");
     if (btn) btn.disabled = true;
     try {
       var res = await fetch("/api/auth/login", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify({ username: user, password: pass })
       });
       if (!res.ok) {
         var data = {};
         try { data = await res.json(); } catch (e) { data = {}; }
         var msg;
         if (res.status === 429) {
           // The throttle used to read as "Invalid username or password", so a
           // locked-out user kept retrying the right password and stayed locked.
           var wait = res.headers.get("Retry-After");
           msg = "Too many failed attempts. Try again in "
             + (wait ? wait + " seconds" : "a little while") + ".";
         } else if (res.status === 401 || res.status === 403) {
           msg = "Invalid username or password.";
         } else {
           msg = "Sign-in failed: " + httpErrorText(data, res.status);
         }
         if (status) status.textContent = msg;
         return;
       }
       // Reload so the whole app re-initialises cleanly with the new session.
       window.location.reload();
     } catch (err) {
       if (status) status.textContent = "Login error: " + err.message;
     } finally {
       if (btn) btn.disabled = false;
     }
   }

   // A protected request returned 401 while we thought we were signed in: the
   // session expired, was revoked by an admin (disable / role change / password
   // reset all drop sessions), or was ended in another tab. Return to the login
   // screen instead of leaving a dead app on screen with silently failing calls.
   function onUnauthorized() {
     if (!window.AUTH_ENABLED) return;                       // auth off: nothing to do
     if (document.body.classList.contains("locked")) return; // already on login
     if (window._healthTimer) { clearInterval(window._healthTimer); window._healthTimer = null; }
     var err = document.getElementById("loginError");
     if (err) err.textContent = "Your session ended. Please sign in again.";
     showLogin({ oidc_enabled: window.OIDC_ENABLED });
   }

   function loginKey(e) {
     if (e.key === "Enter" && !e.isComposing) { e.preventDefault(); doLogin(); }
   }

   function oidcLogin() { window.location = "/api/auth/oidc/login"; }

   async function doLogout() {
     try { await fetch("/api/auth/logout", { method: "POST" }); } catch (err) { /* reload anyway */ }
     window.location.reload();
   }

   // -------------------------------------------------- Claude import panel ---
   function importSummary(it) {
     var counts = it.counts;
     if (typeof counts === "string") { try { counts = JSON.parse(counts); } catch (e) { counts = null; } }
     if (!counts) return "";
     return counts.conversations + " conversations, " + counts.messages + " messages, " +
            counts.knowledge_docs + " knowledge docs, " + counts.skills + " skills" +
            (counts.duplicates ? ", " + counts.duplicates + " duplicates skipped" : "");
   }

   // Status word -> the kit's pill modifier.
   function importPillClass(status) {
     if (status === "completed") return "ok";
     if (status === "failed") return "error";
     if (status === "pending") return "muted";
     return "running";
   }

   var _importPoll = null;

   async function loadImports() {
     var box = document.getElementById("importList");
     if (!box) return;
     if (_importPoll) { clearTimeout(_importPoll); _importPoll = null; }
     try {
       var out = await fetchJSON("/api/imports");
       var items = (out.data && out.data.imports) || [];
       box.textContent = "";
       if (!items.length) {
         var empty = document.createElement("div");
         empty.className = "empty";
         empty.textContent = "No imports yet.";
         box.appendChild(empty);
         return;
       }
       var active = false;
       items.forEach(function(it) {
         var status = it.status || "pending";
         var row = document.createElement("div");
         row.className = "import-row card";
         var head = document.createElement("div");
         head.className = "import-head card-header row";
         var title = document.createElement("strong");
         title.className = "grow";
         title.textContent = it.filename || "(export)";
         var badge = document.createElement("span");
         badge.className = "pill " + importPillClass(status);
         badge.textContent = status + " " + (it.progress || 0) + "%";
         head.appendChild(title); head.appendChild(badge);
         row.appendChild(head);
         var detail = document.createElement("div");
         detail.className = "import-detail hint";
         detail.textContent = importSummary(it) || (it.error ? ("error: " + it.error) : "");
         row.appendChild(detail);
         var actions = document.createElement("div");
         actions.className = "card-actions";
         if (status === "failed") {
           actions.appendChild(makeBtn("Retry", "Run this import again", function() { retryImportJob(it.id); }));
         }
         actions.appendChild(makeBtn("Remove", "Remove this import and what it created",
           function() { deleteImportJob(it.id); }, "btn-danger"));
         row.appendChild(actions);
         box.appendChild(row);
         if (status !== "completed" && status !== "failed") active = true;
       });
       // Poll while an import is still running and the list is on screen.
       if (active && box.offsetParent !== null) _importPoll = setTimeout(loadImports, 1500);
     } catch (err) {
       box.textContent = "Could not load imports: " + err.message;
     }
   }

   async function startImport() {
     var inp = document.getElementById("importFile");
     var status = document.getElementById("importStatus");
     if (!inp || !inp.files || !inp.files[0]) {
       uiToast("Choose a chat export .zip first.", "warn");
       return;
     }
     var file = inp.files[0];
     if (status) status.textContent = "Uploading " + file.name + " (" +
       Math.round(file.size / 1024) + " KB)...";
     try {
       var res = await fetch("/api/import", {
         method: "POST", body: file, headers: { "X-Filename": file.name }
       });
       if (res.status === 401 && typeof onUnauthorized === "function") { onUnauthorized(); return; }
       var data = {};
       try { data = await res.json(); } catch (e) { data = {}; }
       if (!res.ok) {
         if (status) status.textContent = "";
         uiToast("Import failed: " + httpErrorText(data, res.status), "error");
         return;
       }
       if (status) status.textContent = "Processing import " + (data.import_id || "") + "...";
       inp.value = "";
       loadImports();
     } catch (err) {
       if (status) status.textContent = "";
       uiToast("Import error: " + err.message, "error");
     }
   }

   async function retryImportJob(id) {
     try {
       await fetchJSON("/api/imports/" + encodeURIComponent(id) + "/retry", { method: "POST" });
       uiToast("Import restarted.", "ok");
     } catch (err) { uiToast("Could not retry: " + err.message, "error"); }
     loadImports();
   }

   async function deleteImportJob(id) {
     if (!(await uiConfirm("Remove this import and the conversations and knowledge it created?"))) return;
     try {
       await fetchJSON("/api/imports/" + encodeURIComponent(id), { method: "DELETE" });
       uiToast("Import removed.", "ok");
     } catch (err) { uiToast("Could not remove: " + err.message, "error"); }
     loadImports();
   }
"""

__all__ = ["UI_JS_AUTH"]
