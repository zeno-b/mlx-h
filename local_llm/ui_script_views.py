"""Front-end: view switching.

Part of the embedded UI, split out of the single HTML_PAGE string for
readability. Assembled once at import time in ui.py, so there is no
runtime cost and no extra file I/O.

Four views share #main: chat, history, tasks (admin) and settings. The old
admin "Models" view is now the Models & Training section of Settings, so
showView("models") survives only as a shim for bookmarks, the palette and
older code paths. Settings content, its loaders and its sub-tab polling
belong to ui_script_settings.py / ui_script_admin.py; this file only owns
which view is on screen, the composer's visibility and the Tasks timer.
"""

from __future__ import annotations

UI_JS_VIEWS = r"""   // ------------------------------------------------------------- views ---

   var currentView = "chat";
   var selectedTask = null;
   var runSource = null;
   var currentRunId = null;
   var tasksTimer = null;
   var VIEW_NAMES = ["chat", "history", "tasks", "settings"];

   // Leaving Settings with unsaved edits must ask first. The settings code
   // owns the dirty state, so it can answer through settingsCanLeave(); until
   // it does, the sticky Apply bar being visible is the same signal (it is
   // shown exactly while something is unsaved).
   function viewsMayLeaveSettings() {
     if (typeof settingsCanLeave === "function") return settingsCanLeave() !== false;
     var bar = document.getElementById("settingsBar");
     if (!bar || bar.hidden || bar.classList.contains("hidden")) return true;
     return window.confirm("Leave Settings? Your unsaved changes will be lost.");
   }

   function showView(name) {
     // The old admin Models view is a Settings section now.
     if (name === "models") { openSettings("models", "model"); return; }
     if (VIEW_NAMES.indexOf(name) < 0) name = "chat";
     // Tasks is admin-only; a stray call (deep link, old code path) from a
     // user account would open an empty view and earn a 403 from /api/tasks.
     if (name === "tasks" && document.body.getAttribute("data-role") !== "admin") name = "chat";
     var previous = currentView;
     if (previous === "settings" && name !== "settings" && !viewsMayLeaveSettings()) return;
     currentView = name;
     document.body.setAttribute("data-view", name);
     VIEW_NAMES.forEach(function(view) {
       var el = document.getElementById(view + "View");
       if (el) el.classList.toggle("hidden", view !== name);
       var nav = document.getElementById("nav" + view.charAt(0).toUpperCase() + view.slice(1));
       if (nav) {
         nav.classList.toggle("active", view === name);
         nav.setAttribute("aria-pressed", view === name ? "true" : "false");
       }
     });
     var gear = document.getElementById("settingsBtn");
     if (gear) {
       gear.classList.toggle("active", name === "settings");
       gear.setAttribute("aria-pressed", name === "settings" ? "true" : "false");
     }
     // The composer belongs to the conversation. It used to be hidden with an
     // inline display style, which no stylesheet rule could ever override.
     var composer = document.getElementById("composer");
     if (composer) composer.classList.toggle("hidden", name !== "chat");
     if (typeof closeChatMenu === "function") closeChatMenu();

     if (tasksTimer) { clearInterval(tasksTimer); tasksTimer = null; }
     if (name === "tasks") {
       loadTasks();
       tasksTimer = setInterval(loadTasks, 3000);
     } else if (name === "history") {
       loadHistory();
     }
     // Settings sub-tabs poll (model log, cluster, perf). They listen for
     // this to stop when the view goes away, so nothing polls off screen.
     if (previous !== name) {
       try {
         document.dispatchEvent(new CustomEvent("viewchange", { detail: { from: previous, to: name } }));
       } catch (err) { /* very old browser: no CustomEvent constructor */ }
     }
   }
"""

__all__ = ["UI_JS_VIEWS"]
