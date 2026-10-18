"""Front-end: the Settings view (navigation, search, load, dirty tracking, save).

The old panel had one hand-written line per control in loadConfig() and
another in saveConfig(); a control missing from either looked fine and did
nothing. Controls are now bound generically by the attributes ui_settings.py
renders on them (data-key, data-scope, data-widget, data-type):

  * loadSettings() reads GET /api/settings/schema (values for admins, the
    caller's preferences for everyone), falling back to GET /api/config and
    GET /api/me/preferences on a server without it, and writes every control.
  * Editing marks the field dirty and shows the sticky settings bar. Apply
    posts ONLY dirty global keys to POST /api/config and ONLY dirty user keys
    to POST /api/me/preferences, reports changed / ignored / clamped values
    in a toast (never in the chat transcript) and re-reads the values from
    the responses. Discard restores what was loaded.
  * Appearance preferences apply live and save the moment they change.
  * Theme: System / Light / Dark. html.light / html.dark carry the choice,
    neither means "follow the OS". localStorage llm_theme caches it so the
    page paints in the right scheme before the preferences request returns.

openSettings(section, sub) is the one way in (gear button, palette, buttons
in other panels); toggleSettings(), loadConfig() and saveConfig() remain as
shims for older callers. Sub-tab loaders and polling live in
ui_script_admin.py (ADMIN_TABS).
"""

from __future__ import annotations

UI_JS_SETTINGS = r"""   // ---------------------------------------------------------- settings ---

   var settingsSection = null;        // section on screen (or to show on open)
   var settingsSubs = {};             // section -> sub-tab last shown
   var settingsLoaded = {};           // "scope:key" -> value as the server reported it
   var settingsReturnView = "chat";   // where closeSettings() goes back to
   var settingsRestartKeys = [];      // labels of applied settings that need a restart
   var settingsSearching = false;
   // Slots ("scope:key") the user changed while loadSettings() was in flight.
   // Opening Settings and clicking Light before the schema arrived used to
   // let the late (pre-click) values flip the theme back to System on screen
   // while the account had saved Light. null when no load is running.
   var settingsTouched = null;

   function settingsTouch(el) {
     if (settingsTouched && el) settingsTouched[settingsSlot(el)] = true;
   }

   function settingsIsAdmin() {
     return document.body.getAttribute("data-role") === "admin";
   }

   function settingsRoot() {
     return document.getElementById("settingsView");
   }

   function settingsEach(selector, fn) {
     var root = settingsRoot();
     if (!root) return;
     Array.prototype.forEach.call(root.querySelectorAll(selector), fn);
   }

   // Every bound control: the elements carrying data-key + data-scope.
   function settingsControls() {
     var out = [];
     settingsEach("[data-key][data-scope][data-widget]", function(el) { out.push(el); });
     return out;
   }

   function settingsSlot(el) {
     return el.getAttribute("data-scope") + ":" + el.getAttribute("data-key");
   }

   function settingsFieldOf(el) {
     return el.closest ? el.closest(".field") : null;
   }

   function settingsLabelOf(el) {
     var field = settingsFieldOf(el);
     var label = field ? field.querySelector(".field-label") : null;
     var text = label ? label.textContent.replace(/\s*restart\s*$/, "") : "";
     return text || el.getAttribute("data-key");
   }

   // ------------------------------------------------- value read / write ---

   // The value a control holds, as the JSON type its setting takes.
   // undefined = nothing valid to send (empty required number, read-only, an
   // untouched secret).
   function settingsRead(el) {
     var widget = el.getAttribute("data-widget");
     if (widget === "switch") return !!el.checked;
     if (widget === "segmented") return uiSegValue(el);
     if (widget === "readonly") return undefined;
     if (widget === "secret") return el.value === "" ? undefined : el.value;
     var raw = el.value == null ? "" : String(el.value);
     var type = el.getAttribute("data-type");
     if (type === "int" || type === "float") {
       if (raw.trim() === "") return el.getAttribute("data-nullable") === "1" ? null : undefined;
       var n = Number(raw);
       if (!isFinite(n)) return undefined;
       return type === "int" ? Math.round(n) : n;
     }
     return raw;
   }

   function settingsBytes(n) {
     if (!isFinite(n)) return String(n);
     if (n >= 1073741824) return (Math.round(n / 107374182.4) / 10) + " GB";
     if (n >= 1048576) return Math.round(n / 1048576) + " MB";
     if (n >= 1024) return Math.round(n / 1024) + " KB";
     return n + " B";
   }

   // How a value reads in a range label, a read-only fact or a toast.
   function settingsFormat(el, value) {
     if (value === null || value === undefined || value === "") return "not set";
     if (typeof value === "boolean") return value ? "on" : "off";
     if (Object.prototype.toString.call(value) === "[object Array]") return value.join(", ") || "none";
     if (typeof value === "object") return JSON.stringify(value);
     var format = el ? el.getAttribute("data-format") : null;
     var unit = el ? (el.getAttribute("data-unit") || "") : "";
     if (format === "percent") return Math.round(Number(value) * 100) + "%";
     if (format === "bytes") return settingsBytes(Number(value));
     return String(value) + (unit ? " " + unit : "");
   }

   function settingsRangeLabel(input) {
     uiRangeFill(input);
     var out = input.parentNode ? input.parentNode.querySelector(".range-value") : null;
     if (out) out.textContent = settingsFormat(input, settingsRead(input));
   }

   function settingsWrite(el, value) {
     var widget = el.getAttribute("data-widget");
     if (widget === "switch") { el.checked = !!value; return; }
     if (widget === "segmented") { uiSeg(el, value); return; }
     if (widget === "readonly") { el.textContent = settingsFormat(el, value); return; }
     if (widget === "secret") {
       // Never echo a secret back; say whether one is set.
       el.value = "";
       el.placeholder = value ? "(set)" : "(not set)";
       return;
     }
     el.value = (value === null || value === undefined) ? "" : String(value);
     if (widget === "range") settingsRangeLabel(el);
   }

   function settingsSame(a, b) {
     if (a === b) return true;
     if ((a === null || a === undefined) && (b === null || b === undefined)) return true;
     if (typeof a === "number" && typeof b === "number") return Math.abs(a - b) < 1e-9;
     if (a === null || a === undefined || b === null || b === undefined) return false;
     return String(a) === String(b);
   }

   // ------------------------------------------------------- dirty state ---

   function settingsIsDirty(el) {
     var widget = el.getAttribute("data-widget");
     if (widget === "readonly" || el.getAttribute("data-autosave") === "1") return false;
     var value = settingsRead(el);
     if (widget === "secret") return value !== undefined;
     return !settingsSame(value, settingsLoaded[settingsSlot(el)]);
   }

   function settingsMark(el) {
     var field = settingsFieldOf(el);
     if (!field) return;
     var dirty = settingsIsDirty(el);
     field.classList.toggle("dirty", dirty);
     var widget = el.getAttribute("data-widget");
     field.classList.toggle("invalid", dirty && widget !== "secret" && settingsRead(el) === undefined);
   }

   function settingsDirtyControls() {
     return settingsControls().filter(function(el) {
       var field = settingsFieldOf(el);
       return !!(field && field.classList.contains("dirty"));
     });
   }

   function settingsUpdateBar() {
     var dirty = settingsDirtyControls();
     var n = dirty.length;
     var label = document.getElementById("settingsDirty");
     if (label) label.textContent = n ? (n + " unsaved change" + (n === 1 ? "" : "s")) : "No unsaved changes";
     var restart = document.getElementById("settingsRestart");
     var restartText = document.getElementById("settingsRestartText");
     if (restartText) {
       restartText.textContent = settingsRestartKeys.length
         ? "Applied, but read when the app starts: " + settingsRestartKeys.join(", ")
           + ". Restart the app to use the new values."
         : "";
     }
     if (restart) restart.classList.toggle("hidden", !settingsRestartKeys.length);
     var apply = document.getElementById("settingsApplyBtn");
     var discard = document.getElementById("settingsDiscardBtn");
     if (apply) apply.disabled = !n;
     if (discard) discard.disabled = !n;
     var bar = document.getElementById("settingsBar");
     if (bar) bar.classList.toggle("hidden", !n && !settingsRestartKeys.length);
     // A dot on each rail item whose section holds unsaved edits.
     var bySection = {};
     dirty.forEach(function(el) {
       var sec = el.closest(".settings-section");
       if (sec) bySection[sec.getAttribute("data-section")] = true;
     });
     settingsEach(".rail-item[data-section]", function(item) {
       item.classList.toggle("dirty", !!bySection[item.getAttribute("data-section")]);
     });
   }

   function settingsDismissRestart() {
     settingsRestartKeys = [];
     settingsUpdateBar();
   }

   // Leaving Settings (views script asks through this) with unsaved edits.
   function settingsCanLeave() {
     var n = settingsDirtyControls().length;
     if (!n) return true;
     if (!uiConfirm("Leave Settings? " + n + " unsaved change" + (n === 1 ? "" : "s")
                    + " will be lost.")) return false;
     discardSettings();
     return true;
   }

   // ------------------------------------------------------ load / adopt ---

   // Write server values into the controls of one scope and make them the
   // new clean baseline. Only keys present in `values` are touched, so a
   // non-admin (no installation values) keeps empty, hidden controls.
   function settingsAdopt(scopes, values) {
     values = values || {};
     settingsControls().forEach(function(el) {
       var scope = el.getAttribute("data-scope");
       if (scopes.indexOf(scope) < 0) return;
       var key = el.getAttribute("data-key");
       if (!Object.prototype.hasOwnProperty.call(values, key)) return;
       settingsLoaded[settingsSlot(el)] = values[key];
       // A field being typed into keeps its text; it is still compared to
       // the new baseline below.
       var field = settingsFieldOf(el);
       if (document.activeElement !== el || !field || !field.classList.contains("dirty")) {
         settingsWrite(el, values[key]);
       }
       settingsMark(el);
     });
     settingsUpdateBar();
   }

   function settingsAdoptValues(values) {
     values = values || {};
     settingsAdopt(["global", "info"], values);
     // The context meter's denominator lives in the chat script.
     if (typeof values.context_size === "number" && typeof contextSize !== "undefined") {
       contextSize = values.context_size;
       if (typeof updateContextMeter === "function") updateContextMeter(0);
     }
     // project_dir is edited by the Codebase panel (folder browser + Set), not
     // by a bound control, but it still shows the value in use.
     var pd = document.getElementById("cfgProjectDir");
     if (pd && Object.prototype.hasOwnProperty.call(values, "project_dir") && document.activeElement !== pd) {
       pd.value = values.project_dir || "";
     }
   }

   function settingsAdoptPrefs(prefs) {
     settingsAdopt(["user"], prefs || {});
   }

   async function loadSettings() {
     var admin = settingsIsAdmin();
     var values = null;
     var prefs = null;
     settingsTouched = {};
     try {
       var out = await fetchJSON("/api/settings/schema");
       values = out.data.values || null;
       prefs = out.data.preferences || null;
     } catch (err) {
       // A server without the schema route: read the two routes it combines.
       if (admin) {
         try {
           var cfg = await fetchJSON("/api/config");
           values = cfg.data.config || null;
         } catch (err2) {
           uiToast("Could not load settings: " + err2.message, "error");
         }
       }
     }
     if (!prefs) {
       try {
         var p = await fetchJSON("/api/me/preferences");
         prefs = p.data.preferences || null;
       } catch (err3) { prefs = null; }
     }
     var touched = settingsTouched || {};
     settingsTouched = null;
     function untouched(obj, scopes) {
       var keep = {};
       Object.keys(obj).forEach(function(k) {
         var hit = scopes.some(function(sc) { return touched[sc + ":" + k]; });
         if (!hit) keep[k] = obj[k];
       });
       return keep;
     }
     if (values) settingsAdoptValues(untouched(values, ["global", "info"]));
     if (prefs) {
       var fresh = untouched(prefs, ["user"]);
       settingsAdoptPrefs(fresh);
       applyAppearancePrefs(fresh);
     }
     return { values: values, preferences: prefs };
   }

   // ------------------------------------------------------ apply / discard ---

   function settingsControlFor(scope, key) {
     var found = null;
     settingsControls().forEach(function(el) {
       if (!found && el.getAttribute("data-scope") === scope && el.getAttribute("data-key") === key) found = el;
     });
     return found;
   }

   async function applySettings() {
     var dirty = settingsDirtyControls();
     if (!dirty.length) { uiToast("Nothing to apply.", "info"); return; }
     var globals = {};
     var prefs = {};
     var invalid = [];
     dirty.forEach(function(el) {
       var value = settingsRead(el);
       if (value === undefined) { invalid.push(settingsLabelOf(el)); return; }
       var key = el.getAttribute("data-key");
       if (el.getAttribute("data-scope") === "global") globals[key] = value;
       else if (el.getAttribute("data-scope") === "user") prefs[key] = value;
     });
     if (invalid.length) {
       uiToast("Enter a valid value for: " + invalid.join(", "), "warn");
       return;
     }
     var apply = document.getElementById("settingsApplyBtn");
     if (apply) apply.disabled = true;
     var saved = [];
     var ignored = [];
     var clamped = [];
     var failures = [];
     if (Object.keys(globals).length) {
       try {
         var out = await postJSON("/api/config", globals);
         var d = out.data || {};
         var cfg = d.config || {};
         (d.changed || []).forEach(function(key) {
           var el = settingsControlFor("global", key);
           var label = el ? settingsLabelOf(el) : key;
           saved.push(label);
           // The server's clamps are the authority; say when one moved the value.
           if (Object.prototype.hasOwnProperty.call(globals, key) &&
               Object.prototype.hasOwnProperty.call(cfg, key) && !settingsSame(cfg[key], globals[key])) {
             clamped.push(label + " = " + settingsFormat(el, cfg[key]));
           }
           if (el && el.getAttribute("data-restart") === "1" && settingsRestartKeys.indexOf(label) < 0) {
             settingsRestartKeys.push(label);
           }
         });
         (d.ignored || []).forEach(function(key) {
           var el = settingsControlFor("global", key);
           ignored.push(el ? settingsLabelOf(el) : key);
         });
         settingsAdoptValues(cfg);
       } catch (err) {
         failures.push("installation settings: " + err.message);
       }
     }
     if (Object.keys(prefs).length) {
       try {
         var po = await postJSON("/api/me/preferences", prefs);
         var pd = po.data || {};
         (pd.changed || []).forEach(function(key) {
           var el = settingsControlFor("user", key);
           saved.push(el ? settingsLabelOf(el) : key);
         });
         (pd.ignored || []).forEach(function(key) {
           var el = settingsControlFor("user", key);
           ignored.push(el ? settingsLabelOf(el) : key);
         });
         settingsAdoptPrefs(pd.preferences || {});
         applyAppearancePrefs(pd.preferences || {});
       } catch (err) {
         failures.push("your preferences: " + err.message);
       }
     }
     settingsUpdateBar();
     if (failures.length) uiToast("Could not save " + failures.join("; "), "error");
     if (ignored.length) uiToast("Not applied (invalid value): " + ignored.join(", "), "warn");
     if (clamped.length) uiToast("Adjusted to the allowed range: " + clamped.join(", "), "warn");
     if (saved.length) uiToast("Saved: " + saved.join(", "), "ok");
     else if (!failures.length && !ignored.length) uiToast("No settings changed.", "info");
   }

   function discardSettings() {
     settingsDirtyControls().forEach(function(el) {
       settingsWrite(el, settingsLoaded[settingsSlot(el)]);
       settingsMark(el);
     });
     settingsUpdateBar();
   }

   // ------------------------------------------------- appearance / theme ---

   function applyTheme(name) {
     if (name !== "light" && name !== "dark") name = "system";
     var root = document.documentElement;
     // Neither class = System: the stylesheet's prefers-color-scheme block
     // decides. html.light forces light, html.dark forces dark.
     root.classList.toggle("light", name === "light");
     root.classList.toggle("dark", name === "dark");
     root.setAttribute("data-theme", name);
     uiSeg(document.getElementById("themeBtn"), name);
     try { localStorage.setItem("llm_theme", name); } catch (err) { /* private mode */ }
   }

   function settingsEffectiveTheme() {
     var root = document.documentElement;
     if (root.classList.contains("dark")) return "dark";
     if (root.classList.contains("light")) return "light";
     try {
       return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
     } catch (err) { return "light"; }
   }

   // Flip between light and dark (palette command). Saved like the control.
   function toggleTheme() {
     var next = settingsEffectiveTheme() === "dark" ? "light" : "dark";
     applyTheme(next);
     settingsSavePrefQuietly("theme", next);
   }

   function applyAppearancePrefs(prefs) {
     prefs = prefs || {};
     if (prefs.theme) applyTheme(prefs.theme);
     var body = document.body;
     var cache = {};
     try { cache = JSON.parse(localStorage.getItem("llm_appearance") || "{}") || {}; } catch (err) { cache = {}; }
     if (prefs.density) {
       body.classList.toggle("density-compact", prefs.density === "compact");
       cache.density = prefs.density;
     }
     if (prefs.text_size) {
       body.classList.toggle("text-small", prefs.text_size === "small");
       body.classList.toggle("text-large", prefs.text_size === "large");
       cache.text_size = prefs.text_size;
     }
     try { localStorage.setItem("llm_appearance", JSON.stringify(cache)); } catch (err) { /* private mode */ }
     // Chat-side preferences (enter to send, code wrap, reasoning display,
     // metrics, default mode) belong to the chat script.
     if (typeof applyChatPrefs === "function") applyChatPrefs(prefs);
   }

   // Pre-paint: the cached choice, before any request returns.
   function initTheme() {
     var saved = null;
     try { saved = localStorage.getItem("llm_theme"); } catch (err) { saved = null; }
     applyTheme(saved || "system");
     var cache = null;
     try { cache = JSON.parse(localStorage.getItem("llm_appearance") || "null"); } catch (err) { cache = null; }
     if (cache) {
       document.body.classList.toggle("density-compact", cache.density === "compact");
       document.body.classList.toggle("text-small", cache.text_size === "small");
       document.body.classList.toggle("text-large", cache.text_size === "large");
     }
   }

   async function settingsSavePrefQuietly(key, value) {
     var patch = {};
     patch[key] = value;
     try {
       var out = await postJSON("/api/me/preferences", patch);
       settingsAdoptPrefs((out.data || {}).preferences || {});
     } catch (err) {
       // Applied and cached in this browser either way; only the account copy failed.
       if (window.console) console.warn("preference not saved:", err);
     }
   }

   // Appearance controls: apply live, save at once (no Apply needed).
   async function settingsAutosave(el) {
     var key = el.getAttribute("data-key");
     var value = settingsRead(el);
     if (value === undefined) return;
     var patch = {};
     patch[key] = value;
     applyAppearancePrefs(patch);
     settingsLoaded[settingsSlot(el)] = value;
     try {
       var out = await postJSON("/api/me/preferences", patch);
       var d = out.data || {};
       if ((d.ignored || []).length) uiToast("Not saved: " + settingsLabelOf(el), "warn");
       settingsAdoptPrefs(d.preferences || {});
     } catch (err) {
       uiToast(settingsLabelOf(el) + " applied in this browser but not saved to your account: "
               + err.message, "warn");
     }
   }

   // ------------------------------------------------------- navigation ---

   function settingsSectionEl(section) {
     var root = settingsRoot();
     return root ? root.querySelector('.settings-section[data-section="' + section + '"]') : null;
   }

   function settingsAllowed(secEl) {
     return !!secEl && (settingsIsAdmin() || !secEl.classList.contains("admin-only"));
   }

   // Show settingsSection (and its sub-tab), then hand the tab to the
   // lazy-load registry so its loader runs and its poll starts.
   function settingsActivate() {
     var root = settingsRoot();
     if (!root) return;
     if (settingsSearching) settingsClearSearch(false);
     var section = settingsSection;
     var secEl = settingsSectionEl(section);
     if (!settingsAllowed(secEl)) {
       section = "appearance";
       secEl = settingsSectionEl(section);
     }
     settingsSection = section;
     settingsEach(".settings-section", function(s) { s.classList.toggle("hidden", s !== secEl); });
     settingsEach(".rail-item[data-section]", function(item) {
       var on = item.getAttribute("data-section") === section;
       item.classList.toggle("active", on);
       item.setAttribute("aria-selected", on ? "true" : "false");
     });
     var title = document.getElementById("settingsTitle");
     var desc = document.getElementById("settingsDesc");
     if (title && secEl) title.textContent = secEl.getAttribute("data-title") || "";
     if (desc && secEl) desc.textContent = secEl.getAttribute("data-desc") || "";
     var tabKey = section;
     var panels = secEl ? secEl.querySelectorAll(".sub-panel[data-sub]") : [];
     if (panels.length) {
       var sub = settingsSubs[section];
       var known = Array.prototype.some.call(panels, function(p) { return p.getAttribute("data-sub") === sub; });
       if (!known) sub = panels[0].getAttribute("data-sub");
       settingsSubs[section] = sub;
       Array.prototype.forEach.call(panels, function(p) {
         p.classList.toggle("hidden", p.getAttribute("data-sub") !== sub);
       });
       Array.prototype.forEach.call(secEl.querySelectorAll(".seg-tabs button[data-sub]"), function(b) {
         var on = b.getAttribute("data-sub") === sub;
         b.classList.toggle("active", on);
         b.setAttribute("aria-pressed", on ? "true" : "false");
       });
       tabKey = section + "/" + sub;
     }
     try { localStorage.setItem("llm_settings_section", section); } catch (err) { /* private mode */ }
     if (typeof adminTabShown === "function") adminTabShown(tabKey);
   }

   // The one way in: gear button, palette, deep links from other panels.
   // openSettings("users") (an old admin tab name) also works.
   function openSettings(section, sub) {
     if (section && !settingsSectionEl(section) && typeof adminSectionFor === "function") {
       var mapped = adminSectionFor(section);
       section = mapped[0];
       sub = sub || mapped[1];
     }
     if (!section) {
       var stored = null;
       try { stored = localStorage.getItem("llm_settings_section"); } catch (err) { stored = null; }
       section = settingsSection || stored || "appearance";
     }
     settingsSection = section;
     if (sub) settingsSubs[section] = sub;
     if (typeof currentView !== "undefined" && currentView === "settings") {
       settingsActivate();
       return;
     }
     // showView fires "viewchange", whose listener below activates and loads.
     showView("settings");
   }

   function closeSettings() {
     showView(settingsReturnView && settingsReturnView !== "settings" ? settingsReturnView : "chat");
   }

   // Old entry point (the aside's toggle): open or close the Settings view.
   function toggleSettings() {
     if (typeof currentView !== "undefined" && currentView === "settings") closeSettings();
     else openSettings();
   }

   // Thin wrappers kept for older callers (auth boot, old buttons).
   async function loadConfig() {
     return loadSettings();
   }

   async function saveConfig() {
     return applySettings();
   }

   document.addEventListener("viewchange", function(e) {
     var d = e.detail || {};
     if (d.to === "settings") {
       settingsReturnView = d.from && d.from !== "settings" ? d.from : "chat";
       if (!settingsSection) openSettingsDefault();
       settingsActivate();
       loadSettings();
     } else if (d.from === "settings") {
       if (typeof adminTabStop === "function") adminTabStop();
       if (settingsSearching) settingsClearSearch(false);
     }
   });

   function openSettingsDefault() {
     var stored = null;
     try { stored = localStorage.getItem("llm_settings_section"); } catch (err) { stored = null; }
     settingsSection = stored || "appearance";
   }

   // ----------------------------------------------------------- search ---

   function settingsSearch(query) {
     var q = String(query || "").trim().toLowerCase();
     if (!q) { settingsClearSearch(true); return; }
     var root = settingsRoot();
     if (!root) return;
     if (!settingsSearching && typeof adminTabStop === "function") adminTabStop();
     settingsSearching = true;
     root.classList.add("searching");
     var admin = settingsIsAdmin();
     var total = 0;
     settingsEach(".settings-section", function(sec) {
       var hits = 0;
       if (sec.classList.contains("admin-only") && !admin) { sec.classList.add("hidden"); return; }
       Array.prototype.forEach.call(sec.querySelectorAll(".field[data-field]"), function(field) {
         var text = (field.textContent + " " + field.getAttribute("data-field")).toLowerCase();
         var gated = field.classList.contains("admin-only") && !admin;
         var hit = !gated && text.indexOf(q) >= 0;
         field.classList.toggle("hidden", !hit);
         if (hit) hits += 1;
       });
       Array.prototype.forEach.call(sec.querySelectorAll("details.sgroup, details.sgroup-advanced"), function(group) {
         var visible = group.querySelector(".field[data-field]:not(.hidden)");
         group.classList.toggle("hidden", !visible);
         if (visible && !group.open) { group.open = true; group.setAttribute("data-search-opened", "1"); }
       });
       Array.prototype.forEach.call(sec.querySelectorAll(".sub-panel"), function(panel) {
         panel.classList.toggle("hidden", !panel.querySelector(".field[data-field]:not(.hidden)"));
       });
       Array.prototype.forEach.call(sec.querySelectorAll(".panel-block, .seg-tabs"), function(block) {
         block.classList.add("hidden");
       });
       var head = sec.querySelector(".settings-section-title");
       if (head) head.classList.toggle("hidden", !hits);
       sec.classList.toggle("hidden", !hits);
       total += hits;
     });
     var none = document.getElementById("settingsNoMatch");
     if (none) none.classList.toggle("hidden", total > 0);
     var title = document.getElementById("settingsTitle");
     var desc = document.getElementById("settingsDesc");
     if (title) title.textContent = "Search";
     if (desc) desc.textContent = total + " matching setting" + (total === 1 ? "" : "s");
   }

   function settingsClearSearch(reactivate) {
     var root = settingsRoot();
     if (!root) return;
     settingsSearching = false;
     root.classList.remove("searching");
     settingsEach(".field[data-field], details.sgroup, details.sgroup-advanced, .panel-block, .seg-tabs", function(el) {
       el.classList.remove("hidden");
       if (el.getAttribute("data-search-opened") === "1") { el.open = false; el.removeAttribute("data-search-opened"); }
     });
     settingsEach(".settings-section-title", function(h) { h.classList.add("hidden"); });
     var none = document.getElementById("settingsNoMatch");
     if (none) none.classList.add("hidden");
     var input = document.getElementById("settingsSearch");
     if (input && input.value && reactivate !== true) input.value = "";
     if (reactivate) settingsActivate();
   }

   // --------------------------------------------------------- wiring ---

   function wireSettings() {
     var root = settingsRoot();
     if (!root || root.getAttribute("data-wired") === "1") return;
     root.setAttribute("data-wired", "1");
     settingsEach('input[type="range"][data-key]', function(el) { settingsRangeLabel(el); });

     function bound(target) {
       var el = target && target.closest ? target.closest("[data-key][data-scope][data-widget]") : null;
       return el && root.contains(el) ? el : null;
     }
     root.addEventListener("input", function(e) {
       var el = bound(e.target);
       if (!el) return;
       if (el.getAttribute("data-widget") === "range") settingsRangeLabel(el);
       settingsTouch(el);
       if (el.getAttribute("data-autosave") === "1") return;
       settingsMark(el);
       settingsUpdateBar();
     });
     root.addEventListener("change", function(e) {
       var el = bound(e.target);
       if (!el) return;
       settingsTouch(el);
       if (el.getAttribute("data-autosave") === "1") { settingsAutosave(el); return; }
       settingsMark(el);
       settingsUpdateBar();
     });
     root.addEventListener("click", function(e) {
       var target = e.target;
       // Segmented controls bound to a setting.
       var btn = target && target.closest ? target.closest("button[data-value]") : null;
       var seg = btn ? btn.closest(".seg[data-key]") : null;
       if (btn && seg && root.contains(seg) && !btn.disabled) {
         uiSeg(seg, btn.getAttribute("data-value"));
         settingsTouch(seg);
         if (seg.getAttribute("data-autosave") === "1") { settingsAutosave(seg); return; }
         settingsMark(seg);
         settingsUpdateBar();
         return;
       }
       // Sub-tabs.
       var tab = target && target.closest ? target.closest(".seg-tabs button[data-sub]") : null;
       if (tab && root.contains(tab)) {
         var sec = tab.closest(".settings-section");
         if (sec) openSettings(sec.getAttribute("data-section"), tab.getAttribute("data-sub"));
       }
     });
     var search = document.getElementById("settingsSearch");
     if (search) {
       search.addEventListener("input", function() { settingsSearch(search.value); });
       search.addEventListener("keydown", function(e) {
         if (e.key === "Escape") { search.value = ""; settingsClearSearch(true); }
       });
     }
     // Ctrl/Cmd+S applies while Settings is open (instead of the browser
     // offering to save the page).
     document.addEventListener("keydown", function(e) {
       if (typeof currentView === "undefined" || currentView !== "settings") return;
       if ((e.metaKey || e.ctrlKey) && e.key && e.key.toLowerCase() === "s") {
         e.preventDefault();
         applySettings();
       }
     });
     settingsUpdateBar();
   }

   initTheme();
   wireSettings();
"""

__all__ = ["UI_JS_SETTINGS"]
