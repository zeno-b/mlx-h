"""Front-end: the shared JS kit (toasts, JSON posts, segmented controls, ranges).

Every panel used to hand-roll the same five things: a fetch with JSON headers,
a try/catch that called alert(), a pressed-state loop over a button group, a
slider label, and a blob download. Each copy drifted (alert() blocking the tab
for a routine "Saved", downloads re-wrapped as text that corrupted .docx/.pdf).
These helpers are the one implementation; ui_script_settings.py,
ui_script_admin.py, ui_script_models.py and the shell scripts call them.

Function declarations hoist across the single assembled <script>, so this part
may sit anywhere in the order; ui.py puts it first because it has no
top-level statements that depend on anything else.
"""

from __future__ import annotations

UI_JS_KIT = r"""   // --------------------------------------------------------------- kit ---

   // One toast implementation: stacked in #toastHost, auto-dismissed, never
   // blocking (alert() froze the whole tab for a routine "Saved").
   // kind: ok | error | warn | info.
   function uiToast(message, kind) {
     kind = kind || "info";
     var host = document.getElementById("toastHost");
     if (!host) {
       // A page without the host (an old cached shell): still say something.
       if (kind === "error" && window.console) console.error(message);
       return null;
     }
     var toast = document.createElement("div");
     toast.className = "toast " + kind;
     toast.setAttribute("role", kind === "error" ? "alert" : "status");
     var text = document.createElement("span");
     text.className = "toast-text";
     text.textContent = String(message == null ? "" : message);
     var close = document.createElement("button");
     close.type = "button";
     close.className = "btn-icon btn-ghost btn-sm toast-close";
     close.setAttribute("aria-label", "Dismiss");
     close.textContent = "×";
     toast.appendChild(text);
     toast.appendChild(close);
     host.appendChild(toast);
     // Errors stay longer: they are the ones somebody has to read twice.
     var ms = kind === "error" ? 9000 : (kind === "warn" ? 7000 : 4000);
     var timer = setTimeout(dismiss, ms);
     function dismiss() {
       clearTimeout(timer);
       if (toast.parentNode) toast.parentNode.removeChild(toast);
     }
     close.onclick = dismiss;
     // Keep the stack short: the oldest goes first.
     while (host.children.length > 4) host.removeChild(host.firstChild);
     return toast;
   }

   // fetchJSON with a JSON body. Same return shape ({res, data}) and the same
   // throw-on-non-2xx behaviour, so callers keep one error path.
   async function postJSON(url, body) {
     return fetchJSON(url, {
       method: "POST",
       headers: { "Content-Type": "application/json" },
       body: JSON.stringify(body == null ? {} : body)
     });
   }

   // Run fn; on failure show "<label>: <reason>" as an error toast instead of
   // an alert or a line in the chat transcript. Returns fn's result or null.
   async function uiAct(label, fn) {
     try {
       return await fn();
     } catch (err) {
       uiToast((label ? label + ": " : "") + ((err && err.message) || err), "error");
       return null;
     }
   }

   // Press the button of a .seg whose data-value equals value.
   function uiSeg(el, value) {
     if (!el) return;
     var want = value == null ? "" : String(value);
     Array.prototype.forEach.call(el.querySelectorAll("button[data-value]"), function(b) {
       var on = b.getAttribute("data-value") === want;
       b.setAttribute("aria-pressed", on ? "true" : "false");
       b.classList.toggle("active", on);
     });
   }

   // The value of the pressed button of a .seg, or null when none is.
   function uiSegValue(el) {
     if (!el) return null;
     var on = el.querySelector('button[data-value][aria-pressed="true"]');
     return on ? on.getAttribute("data-value") : null;
   }

   // Wire a .seg: a click presses the button and calls onChange(value, el).
   function wireSeg(el, onChange) {
     if (!el || el.getAttribute("data-wired") === "1") return;
     el.setAttribute("data-wired", "1");
     el.addEventListener("click", function(e) {
       var b = e.target.closest ? e.target.closest("button[data-value]") : null;
       if (!b || !el.contains(b) || b.disabled) return;
       var value = b.getAttribute("data-value");
       uiSeg(el, value);
       if (onChange) onChange(value, el);
     });
   }

   // A range's filled track is a CSS custom property, the one thing a class
   // cannot carry. Call after every programmatic value change.
   function uiRangeFill(input) {
     if (!input) return;
     var min = Number(input.min || 0);
     var max = Number(input.max || 100);
     var val = Number(input.value);
     var pct = max > min ? ((val - min) / (max - min)) * 100 : 0;
     pct = Math.max(0, Math.min(100, pct));
     input.style.setProperty("--fill", pct + "%");
   }

   // Download a URL as a file via fetch + blob, so a 401/403/500 becomes a
   // toast instead of replacing the app with an error body. Reads the body as a
   // Blob, never as text: re-wrapping binary as text corrupted .docx and .pdf.
   async function uiDownload(url, filename) {
     try {
       var resp = await fetch(url);
       if (resp.status === 401 && typeof onUnauthorized === "function") { onUnauthorized(); return false; }
       if (!resp.ok) {
         var e = {};
         try { e = await resp.json(); } catch (x) { e = {}; }
         var why = typeof httpErrorText === "function" ? httpErrorText(e, resp.status)
                                                        : (e.error || e.detail || ("HTTP " + resp.status));
         uiToast("Download failed: " + why, "error");
         return false;
       }
       var blob = await resp.blob();
       var a = document.createElement("a");
       a.href = URL.createObjectURL(blob);
       a.download = filename || "download";
       document.body.appendChild(a);
       a.click();
       a.remove();
       // Revoking in the same tick can cancel the download in some browsers.
       setTimeout(function() { URL.revokeObjectURL(a.href); }, 1500);
       return true;
     } catch (err) {
       uiToast("Download failed: " + err.message, "error");
       return false;
     }
   }

   // One place to replace with a styled dialog later; window.confirm for now.
   function uiConfirm(message) {
     return window.confirm(message);
   }
"""

__all__ = ["UI_JS_KIT"]
