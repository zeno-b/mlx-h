"""Front-end: shared state, helpers, markdown rendering, the composer and the streaming chat trace.

Part of the embedded UI, split out of the single HTML_PAGE string for
readability. Assembled once at import time in ui.py, so there is no
runtime cost and no extra file I/O.

This part runs first, so it owns the globals every later part reads (chat,
input, conversationId, busy, contextSize, fetchJSON, downloadFile, addSystem,
renderMarkdown, stopStream). Settings load/save moved to ui_script_settings.py;
what stays here is the chat surface itself: the composer (mode switch, Enter
handling, context meter), the per-user display preferences it honours, the
overflow menu, the glass-box trace and the health pill.
"""

from __future__ import annotations

UI_JS_CHAT = r"""
   var chat = document.getElementById("chat");
   var input = document.getElementById("input");
   var sendBtn = document.getElementById("sendBtn");
   var stopBtn = document.getElementById("stopBtn");
   var statusEl = document.getElementById("status");
   var modeSeg = document.getElementById("modeSeg");

   // localStorage can throw (Safari private mode, a sandboxed iframe, storage
   // disabled by policy). An uncaught throw on the very first line that
   // touches it used to stop this whole script, which is every button on the
   // page, so every read and write goes through these two.
   function safeStorageGet(key) {
     try { return window.localStorage.getItem(key); } catch (err) { return null; }
   }
   function safeStorageSet(key, value) {
     try { window.localStorage.setItem(key, value); } catch (err) { /* private mode */ }
   }

   var conversationId = safeStorageGet("llm_conversation") || randomId();
   safeStorageSet("llm_conversation", conversationId);
   var contextSize = 4096;
   var usedTokens = 0;
   var busy = false;
   var controller = null;

   // Per-user display preferences (GET /api/me/preferences). The defaults are
   // the behaviour this page had before preferences existed, so an older
   // server without the route renders exactly as it used to.
   var chatPrefs = {
     enter_to_send: true,
     code_wrap: false,
     reasoning_display: "collapsed",
     show_metrics: true,
     default_mode: "auto"
   };
   var _chatPrefsMode = null;
   // The composer's mode switch: "auto" lets the server decide (and honours
   // AGENT_ENABLED), "chat" forces a plain reply, "agent" forces the tool loop.
   var chatMode = "auto";

   function randomId() {
     return Math.random().toString(36).slice(2, 12);
   }

   function estimateTokens(text) {
     return Math.max(1, Math.ceil((text || "").length / 4));
   }

   function roleIsAdmin() {
     return document.body.dataset.role === "admin";
   }

   input.addEventListener("keydown", function(e) {
     if (e.key !== "Enter") return;
     // An IME (Japanese, Chinese, Korean input) uses Enter to CONFIRM the
     // candidate being composed. Treating that Enter as send posted half-typed
     // messages; keyCode 229 covers browsers that do not set isComposing.
     if (e.isComposing || e.keyCode === 229) return;
     var modified = e.metaKey || e.ctrlKey;
     var sends = chatPrefs.enter_to_send ? (!e.shiftKey && !e.altKey) : modified;
     if (sends) {
       e.preventDefault();
       send();
     }
   });

   input.addEventListener("input", function() {
     // Auto-grow is geometry measured from the content, not a look, so it is
     // the one style this element gets from script.
     input.style.height = "auto";
     input.style.height = Math.min(200, input.scrollHeight) + "px";
   });

   function stopStream() {
     if (controller) controller.abort();
   }

   // ------------------------------------------------------- composer ---

   function setChatMode(mode) {
     if (["auto", "chat", "agent"].indexOf(mode) < 0) mode = "auto";
     chatMode = mode;
     if (!modeSeg) return;
     Array.prototype.forEach.call(modeSeg.querySelectorAll("button[data-value]"), function(b) {
       var on = b.getAttribute("data-value") === mode;
       b.setAttribute("aria-pressed", on ? "true" : "false");
       b.classList.toggle("active", on);
     });
   }

   // What send() puts in the request's `agent` field. null is not "off": it
   // tells the server to use its own default (config.agent_enabled plus the
   // automatic promotions), which is what every non-admin silently lost when
   // this was the admin-only checkbox and they always sent false.
   function modeAgentFlag() {
     if (chatMode === "agent") return true;
     if (chatMode === "chat") return false;
     return null;
   }

   if (modeSeg) modeSeg.addEventListener("click", function(e) {
     var b = e.target.closest ? e.target.closest("button[data-value]") : null;
     if (!b) return;
     setChatMode(b.getAttribute("data-value"));
     input.focus();
   });

   function applyChatPrefs(prefs) {
     if (prefs) {
       ["enter_to_send", "code_wrap", "reasoning_display", "show_metrics", "default_mode"]
         .forEach(function(k) {
           if (prefs[k] !== undefined && prefs[k] !== null) chatPrefs[k] = prefs[k];
         });
     }
     document.body.classList.toggle("code-wrap", !!chatPrefs.code_wrap);
     document.body.classList.toggle("hide-metrics", !chatPrefs.show_metrics);
     document.body.setAttribute("data-reasoning", chatPrefs.reasoning_display || "collapsed");
     input.placeholder = chatPrefs.enter_to_send
       ? "Send a message. Enter sends, Shift+Enter adds a line."
       : "Send a message. Ctrl/Cmd+Enter sends, Enter adds a line.";
     var tip = chatPrefs.enter_to_send ? "Send your message (or press Enter)."
                                       : "Send your message (or press Ctrl/Cmd+Enter).";
     sendBtn.setAttribute("data-tip", tip);
     // The default mode seeds the switch once, and again only when the
     // preference itself changes. Re-applying it on every reload of the
     // preferences would undo the switch the user just flipped for this chat.
     if (chatPrefs.default_mode !== _chatPrefsMode) {
       _chatPrefsMode = chatPrefs.default_mode;
       setChatMode(chatPrefs.default_mode);
     }
   }

   async function loadChatPrefs() {
     try {
       var out = await fetchJSON("/api/me/preferences");
       applyChatPrefs((out.data || {}).preferences || {});
     } catch (err) {
       // A server without per-user preferences, or a transient failure: keep
       // the current values. The chat must never depend on this request.
       applyChatPrefs(null);
     }
   }

   // Scrolling used to be one forced layout per token. `chat.scrollTop =
   // chat.scrollHeight` reads a geometry property, so the browser has to flush
   // pending layout before it can answer -- at 20-40 tok/s over a tree holding
   // six live reasoning panes and a markdown subtree rebuilt four times a
   // second, that is 20-40 synchronous reflows per second and it is most of the
   // jank. Coalescing to one per animation frame costs nothing visually: the
   // screen only repaints that often anyway.
   //
   // It also stopped fighting the user. Scrolling up to read step 2 used to be
   // undone by the very next token, which made the trace unreadable while it
   // streamed -- the opposite of a glass box. Once the view is more than a
   // little way off the bottom the follow stops until the next turn re-arms it.
   var stickToBottom = true, scrollQueued = false, pinEl = null;
   chat.addEventListener("scroll", function() {
     stickToBottom = chat.scrollHeight - chat.scrollTop - chat.clientHeight < 48;
   });
   // Keep an inner scroller (a reasoning pane) showing its newest line. Without
   // this a step that runs past its 160px box looked frozen while the page
   // kept jerking underneath it.
   function pinBottom(el) { pinEl = el; scrollDown(); }
   function scrollDown(force) {
     if (force) stickToBottom = true;
     if (scrollQueued) return;
     scrollQueued = true;
     requestAnimationFrame(function() {
       scrollQueued = false;
       if (pinEl) { pinEl.scrollTop = pinEl.scrollHeight; pinEl = null; }
       if (stickToBottom) chat.scrollTop = chat.scrollHeight;
     });
   }

   // --------------------------------------------------- small helpers ---

   // One factory for every button script builds, so a JS-made button carries
   // the same kit classes as one written in the markup.
   function makeBtn(label, title, onclick, extra) {
     var b = document.createElement("button");
     b.type = "button";
     b.className = "btn btn-sm btn-ghost" + (extra ? " " + extra : "");
     b.textContent = label;
     if (title) b.title = title;
     if (onclick) b.onclick = onclick;
     return b;
   }

   // navigator.clipboard only exists on secure origins (https or localhost).
   // Opened over plain http on a LAN or Tailscale address it is undefined, and
   // every Copy button threw a TypeError before its error handler could run,
   // so copying silently did nothing. Fall back to a selected text area.
   function copyText(text) {
     if (navigator.clipboard && window.isSecureContext) {
       return navigator.clipboard.writeText(text);
     }
     return new Promise(function(resolve, reject) {
       var prev = document.activeElement;
       var ta = document.createElement("textarea");
       ta.value = text;
       ta.setAttribute("readonly", "");
       ta.setAttribute("aria-hidden", "true");
       ta.className = "clip-buffer";
       document.body.appendChild(ta);
       try { ta.focus({ preventScroll: true }); } catch (e) { ta.focus(); }
       ta.select();
       var ok = false;
       try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
       ta.remove();
       if (prev && prev.focus) { try { prev.focus({ preventScroll: true }); } catch (e) { /* gone */ } }
       if (ok) resolve(); else reject(new Error("copy refused"));
     });
   }

   function copyWithFeedback(button, text) {
     copyText(text).then(function() {
       button.textContent = "Copied";
       setTimeout(function() { button.textContent = "Copy"; }, 1200);
     }, function() {
       uiToast("Could not copy. Select the text and copy it by hand.", "warn");
     });
   }

   // Only these link targets are rendered as live links. Model output can carry
   // prompt-injected web content, and a markdown link to `javascript:...`
   // (or `data:`) used to become a real anchor one click away from running.
   // Same-origin document downloads are the one relative target the app emits.
   function safeHref(raw) {
     var href = String(raw || "").trim();
     if (/^(https?:\/\/|mailto:)/i.test(href)) return href;
     if (/^\/api\/files\/download\?/.test(href)) return href;
     return "";
   }

   // ------------------------------------------------------- markdown ---
   // Builds DOM nodes rather than assigning innerHTML, so model output can never
   // inject markup or scripts into the page. Handles fenced code (with a copy
   // button and language label), headings, lists, blockquotes, inline code,
   // bold/italic and links (scheme-checked by safeHref).

   function renderInline(text, host) {
     // Order matters: code spans first so their contents are never re-parsed.
     var pattern = /(`[^`\n]+`)|(\*\*[^*\n]+\*\*)|(\*[^*\n]+\*|_[^_\n]+_)|(\[[^\]\n]+\]\([^)\s]+\))/;
     var rest = text;
     while (rest) {
       var m = rest.match(pattern);
       if (!m) { host.appendChild(document.createTextNode(rest)); break; }
       if (m.index > 0) host.appendChild(document.createTextNode(rest.slice(0, m.index)));
       var tok = m[0];
       if (tok[0] === "`") {
         var code = document.createElement("code");
         code.className = "inline-code";
         code.textContent = tok.slice(1, -1);
         host.appendChild(code);
       } else if (tok.slice(0, 2) === "**") {
         var b = document.createElement("strong");
         b.textContent = tok.slice(2, -2);
         host.appendChild(b);
       } else if (tok[0] === "[") {
         var close = tok.indexOf("](");
         var href = safeHref(tok.slice(close + 2, -1));
         if (href) {
           var a = document.createElement("a");
           a.textContent = tok.slice(1, close);
           a.href = href;
           a.target = "_blank";
           a.rel = "noopener noreferrer";
           host.appendChild(a);
         } else {
           // Not a scheme we open: show what the model wrote, inert.
           host.appendChild(document.createTextNode(tok));
         }
       } else {
         var i = document.createElement("em");
         i.textContent = tok.slice(1, -1);
         host.appendChild(i);
       }
       rest = rest.slice(m.index + tok.length);
     }
   }

   // Minimal offline syntax highlighting. A real grammar is overkill here and a
   // CDN library would break the offline-first promise, so this tokenises the
   // few things that carry most of the visual signal: comments, strings,
   // numbers and keywords. Everything is inserted as text nodes, never HTML.
   var HL_KEYWORDS = {
     python: "def class return if elif else for while import from as try except finally raise with lambda yield pass break continue in is not and or None True False async await global nonlocal assert del",
     javascript: "function return if else for while var let const class new try catch finally throw typeof instanceof in of do switch case break continue default null undefined true false async await import export from extends this",
     json: "true false null",
     bash: "if then else fi for while do done case esac function return export local echo cd set source",
     sql: "select from where group by order having insert update delete create table drop alter join left right inner outer on as values set distinct limit"
   };
   HL_KEYWORDS.js = HL_KEYWORDS.javascript;
   HL_KEYWORDS.ts = HL_KEYWORDS.javascript;
   HL_KEYWORDS.py = HL_KEYWORDS.python;
   HL_KEYWORDS.sh = HL_KEYWORDS.bash;

   function highlightInto(code, lang, host) {
     var words = HL_KEYWORDS[(lang || "").toLowerCase()];
     if (!words) { host.textContent = code; return; }
     var keywords = {};
     words.split(" ").forEach(function(w) { keywords[w] = true; });
     var lineComment = (lang === "python" || lang === "py" || lang === "bash" || lang === "sh") ? "#" : "//";
     var i = 0;
     function emit(text, cls) {
       if (!text) return;
       if (!cls) { host.appendChild(document.createTextNode(text)); return; }
       var span = document.createElement("span");
       span.className = cls;
       span.textContent = text;
       host.appendChild(span);
     }
     while (i < code.length) {
       var ch = code[i];
       // Comment to end of line
       if (code.startsWith(lineComment, i)) {
         var nl = code.indexOf("\n", i);
         if (nl < 0) nl = code.length;
         emit(code.slice(i, nl), "hl-com");
         i = nl;
         continue;
       }
       // String literal
       if (ch === '"' || ch === "'" || ch === "`") {
         var j = i + 1;
         while (j < code.length && code[j] !== ch) {
           if (code[j] === "\\") j++;
           j++;
         }
         emit(code.slice(i, Math.min(j + 1, code.length)), "hl-str");
         i = j + 1;
         continue;
       }
       // Number
       if (ch >= "0" && ch <= "9") {
         var k = i;
         while (k < code.length && /[0-9._xa-fA-F]/.test(code[k])) k++;
         emit(code.slice(i, k), "hl-num");
         i = k;
         continue;
       }
       // Word (keyword or plain identifier)
       if (/[A-Za-z_$]/.test(ch)) {
         var w = i;
         while (w < code.length && /[A-Za-z0-9_$]/.test(code[w])) w++;
         var word = code.slice(i, w);
         emit(word, keywords[word] ? "hl-kw" : null);
         i = w;
         continue;
       }
       emit(ch, null);
       i++;
     }
   }

   function makeCodeBlock(code, lang) {
     var wrap = document.createElement("div");
     wrap.className = "codeblock";
     var bar = document.createElement("div");
     bar.className = "codebar";
     var label = document.createElement("span");
     label.textContent = lang || "code";
     var copy = makeBtn("Copy", "Copy this code", null);
     copy.onclick = function() { copyWithFeedback(copy, code); };
     bar.appendChild(label);
     bar.appendChild(copy);
     var pre = document.createElement("pre");
     var el = document.createElement("code");
     highlightInto(code, lang, el);
     pre.appendChild(el);
     wrap.appendChild(bar);
     wrap.appendChild(pre);
     return wrap;
   }

   // Re-render streaming text as markdown, but only every ~250ms: parsing on
   // every token would be wasteful and would make half-typed code fences flicker.
   function scheduleMarkdown(node) {
     if (node.mdTimer) return;
     // A long answer re-renders half as often. The tail is still visibly
     // growing at 2Hz, and by the time a reply is four thousand characters the
     // parse is walking hundreds of lines and re-highlighting every code block
     // in it, four times a second, for no gain the eye can see.
     var wait = (node.raw && node.raw.length > 4000) ? 500 : 250;
     node.mdTimer = setTimeout(function() {
       node.mdTimer = null;
       // Nothing new arrived since the last pass (the model stalled, or the
       // reply ended between ticks): rebuilding identical DOM would throw away
       // the user's text selection and every code block's scroll position for
       // no change on screen.
       if (node.lastRendered === node.raw) return;
       node.lastRendered = node.raw;
       // An unclosed code fence used to revert the WHOLE message to its raw
       // markdown source -- every heading back to "## ", every earlier code
       // block losing its bar and its highlighting -- and then flip back when
       // the fence closed. Two full-message swaps per code block. It is also
       // unnecessary: renderMarkdown consumes to the end of the text when it
       // finds no closing fence, so a half-written block already renders as a
       // code block that grows. Render it.
       renderMarkdown(node.raw, node.text);
       // Rendered now: the token handler stops overwriting textContent with raw
       // text every token, so the message no longer oscillates plain<->markdown
       // (the flicker). Subsequent tokens accumulate and re-render on the next
       // debounce, swapped in atomically by renderMarkdown.
       node.rendered = true;
       scrollDown();
     }, wait);
   }

   function renderMarkdown(text, host) {
     // Build into a detached fragment and swap it in atomically at the end, so
     // the message is never momentarily empty (clearing innerHTML then refilling
     // on every debounced streaming pass is what made the answer flicker).
     var frag = document.createDocumentFragment();
     var lines = String(text == null ? "" : text).split("\n");
     var i = 0;
     var list = null;
     function endList() { list = null; }
     while (i < lines.length) {
       var line = lines[i];
       var fence = line.match(/^\s*```(\w+)?\s*$/);
       if (fence) {
         endList();
         var lang = fence[1] || "";
         var buf = [];
         i++;
         while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) { buf.push(lines[i]); i++; }
         i++;
         frag.appendChild(makeCodeBlock(buf.join("\n"), lang));
         continue;
       }
       var heading = line.match(/^(#{1,4})\s+(.*)$/);
       if (heading) {
         endList();
         var h = document.createElement("h" + Math.min(4, heading[1].length + 2));
         h.className = "md-h";
         renderInline(heading[2], h);
         frag.appendChild(h);
         i++;
         continue;
       }
       var item = line.match(/^\s*[-*+]\s+(.*)$/) || line.match(/^\s*(\d+)\.\s+(.*)$/);
       if (item) {
         var ordered = /^\s*\d+\./.test(line);
         if (!list || list.ordered !== ordered) {
           var el = document.createElement(ordered ? "ol" : "ul");
           el.className = "md-list";
           frag.appendChild(el);
           list = { el: el, ordered: ordered };
         }
         var li = document.createElement("li");
         renderInline(item.length === 2 ? item[1] : item[2], li);
         list.el.appendChild(li);
         i++;
         continue;
       }
       if (/^\s*>\s?/.test(line)) {
         endList();
         var q = document.createElement("blockquote");
         q.className = "md-quote";
         renderInline(line.replace(/^\s*>\s?/, ""), q);
         frag.appendChild(q);
         i++;
         continue;
       }
       if (!line.trim()) { endList(); i++; continue; }
       endList();
       var p = document.createElement("p");
       p.className = "md-p";
       var block = [line];
       i++;
       while (i < lines.length && lines[i].trim() && !/^\s*(```|#{1,4}\s|[-*+]\s|\d+\.\s|>)/.test(lines[i])) {
         block.push(lines[i]);
         i++;
       }
       renderInline(block.join("\n"), p);
       frag.appendChild(p);
     }
     host.replaceChildren(frag);
     return host;
   }

   // ------------------------------------------------------- messages ---

   // An empty conversation shows one quiet line instead of a blank pane.
   function showChatEmpty() {
     if (chat.querySelector(".chat-empty")) return;
     var empty = document.createElement("div");
     empty.className = "empty chat-empty";
     empty.textContent = "Ask anything. Paste code or documents freely; long input is processed in chunks.";
     chat.appendChild(empty);
   }

   function clearChatEmpty() {
     var empty = chat.querySelector(".chat-empty");
     if (empty) empty.remove();
   }

   function addMessage(role, text) {
     clearChatEmpty();
     var div = document.createElement("div");
     div.className = "msg " + role;
     if (role === "assistant") {
       var body = document.createElement("div");
       body.className = "msg-body";
       renderMarkdown(text, body);
       div.appendChild(body);
     } else {
       div.textContent = text;
     }
     addMessageActions(div, role, text);
     chat.appendChild(div);
     scrollDown();
     return div;
   }

   // Copy / edit-and-resend controls under each message.
   function addMessageActions(div, role, text) {
     var bar = document.createElement("div");
     bar.className = "msg-actions";
     var copy = makeBtn("Copy", "Copy this message to the clipboard", null);
     copy.onclick = function() {
       copyWithFeedback(copy, div.dataset.raw || text || div.textContent || "");
     };
     bar.appendChild(copy);
     if (role === "user") {
       bar.appendChild(makeBtn("Edit & resend", "Put this question back in the box to change and ask again",
         function() {
           input.value = div.dataset.raw || text || div.textContent || "";
           input.focus();
           showView("chat");
         }));
     }
     div.dataset.raw = text || "";
     div.appendChild(bar);
     return bar;
   }

   // Regenerate lives on the LAST answer only (it used to be a header button
   // that acted on whatever happened to be last). The server drops the final
   // exchange, so offering it anywhere else would regenerate a different
   // answer than the one the button sits under.
   function clearRegenerate() {
     Array.prototype.forEach.call(chat.querySelectorAll(".regen-btn"), function(b) { b.remove(); });
   }

   function setRegenerateTarget(bar) {
     clearRegenerate();
     if (!bar) return;
     bar.appendChild(makeBtn("Regenerate", "Discard this answer and generate a new one for the same question",
       function() { regenerateLast(); }, "regen-btn"));
   }

   function addSystem(text) {
     clearChatEmpty();
     var div = document.createElement("div");
     div.className = "system-msg";
     div.textContent = text;
     chat.appendChild(div);
     scrollDown();
     return div;
   }

   // --- Glass box: build one trace timeline per assistant turn. ---
   function startTrace() {
     clearChatEmpty();
     var turn = document.createElement("div");
     turn.className = "turn";
     var trace = document.createElement("div");
     trace.className = "trace";
     turn.appendChild(trace);
     // Always-visible activity line: a spinner, the current state, and a clock
     // that keeps ticking so a slow step never looks frozen.
     var activity = document.createElement("div");
     activity.className = "gactivity";
     var spin = document.createElement("span"); spin.className = "gspin";
     var label = document.createElement("span"); label.className = "glabel"; label.textContent = "starting…";
     var elapsed = document.createElement("span"); elapsed.className = "gelapsed";
     activity.appendChild(spin); activity.appendChild(label); activity.appendChild(elapsed);
     turn.appendChild(activity);
     chat.appendChild(turn);
     scrollDown();
     var t = { turn: turn, trace: trace, answer: null, tool: null,
               activity: activity, label: label, elapsed: elapsed,
               started: Date.now(), stepLabel: "", timer: null };
     // Tick the clock four times a second. This is the liveness proof: as long
     // as this number moves, the turn is not dead.
     t.timer = setInterval(function() {
       var secs = (Date.now() - t.started) / 1000;
       // Tenths for the first ten seconds, whole seconds after that. A turn on
       // this hardware runs a minute or more, and a tenths digit flickering
       // through ten values a second for that long is noise, not information --
       // by then the number is there to prove the turn is alive, which whole
       // seconds do just as well. The 4Hz tick itself stays: that is what
       // notices a stall promptly.
       var next = secs < 10 ? secs.toFixed(1) + "s" : Math.round(secs) + "s";
       if (t.elapsed.textContent !== next) t.elapsed.textContent = next;
       // If a single step runs long, flag it visually so a real stall is
       // obvious. Latched rather than toggled: toggle() re-evaluated the class
       // four times a second forever, and the state only ever goes one way.
       if (secs > 25 && !t.answered && !t.stalledOn) {
         t.stalledOn = true;
         activity.classList.add("stalled");
       }
     }, 250);
     return t;
   }

   function setActivity(t, text) {
     if (!t.activity) return;
     // One line, and short. A notice carries up to 400 characters (the plan
     // line is "plan (5 steps): 1) ... 2) ..."), and writing that into a
     // one-line monospace row grew the footer to four wrapped lines and then
     // snapped it back on the next event -- 50px of the page moving under
     // text the user was reading, twice per step.
     var next = String(text || "").split("\n")[0].slice(0, 72);
     next = t.stepLabel ? (t.stepLabel + " · " + next) : next;
     // The token handler calls this on every token. Assigning the same string
     // 30 times a second is 30 needless style recalculations.
     if (t.label.textContent !== next) t.label.textContent = next;
   }

   function stopActivity(t) {
     if (t.timer) { clearInterval(t.timer); t.timer = null; }
     t.answered = true;
     // Close any reasoning step still open. `.done` was added in exactly one
     // place, the reason_done event, so a step interrupted by a cancel or an
     // error kept a full-height pane AND a caret whose click handler checks
     // for `.done` and therefore did nothing -- a visible affordance that
     // silently no-ops, in a state the clamping introduced. Every exit from a
     // turn comes through here, which is the point.
     if (t.reason) { t.reason.node.classList.add("done"); t.reason = null; }
     if (t.activity && t.activity.parentNode) t.activity.parentNode.removeChild(t.activity);
   }

   function bumpActivity(t) {
     // Keep the activity line as the last child of the turn as nodes are added.
     // The guard is not an optimisation: re-appending a node that is ALREADY
     // last still removes and re-inserts it, and a CSS animation restarts from
     // zero when its element is re-inserted. This runs on every token, so the
     // spinner was being reset ~30 times a second and sat visibly frozen for
     // the whole stream -- the one element on screen whose only job is to prove
     // the turn is alive.
     if (t.activity && t.turn.lastChild !== t.activity) t.turn.appendChild(t.activity);
   }

   function traceRouter(t, name, args) {
     var node = document.createElement("div");
     node.className = "gnode router";
     var head = document.createElement("div");
     head.className = "ghead";
     var loc = args && (args.location || args.query);
     head.textContent = "router → " + name + (loc ? " (" + String(loc).slice(0, 60) + ")" : "");
     node.appendChild(head);
     t.trace.appendChild(node);
     scrollDown();
   }

   function traceTool(t, name, args) {
     var node = document.createElement("div");
     node.className = "gnode tool";
     var head = document.createElement("div");
     head.className = "ghead";
     var label = document.createElement("span");
     label.className = "gtool";
     label.textContent = name;
     var pill = document.createElement("span");
     pill.className = "gpill";
     pill.textContent = "running";
     var caret = document.createElement("span");
     caret.className = "gcaret";
     caret.textContent = "▸";
     head.appendChild(label);
     head.appendChild(pill);
     head.appendChild(caret);
     var argsEl = document.createElement("div");
     argsEl.className = "gargs";
     argsEl.textContent = JSON.stringify(args || {});
     var body = document.createElement("pre");
     body.className = "gbody";
     body.textContent = "";
     node.appendChild(head);
     node.appendChild(argsEl);
     node.appendChild(body);
     head.onclick = function() { node.classList.toggle("open"); };
     t.trace.appendChild(node);
     scrollDown();
     return { node: node, pill: pill, body: body };
   }

   function traceThinking(t) {
     if (t.thinking) return t.thinking;
     var node = document.createElement("div");
     node.className = "gnode thinking open";
     var head = document.createElement("div");
     head.className = "ghead";
     var tag = document.createElement("span");
     tag.className = "gtool";
     tag.textContent = "thinking";
     var caret = document.createElement("span");
     caret.className = "gcaret";
     caret.textContent = "▸";
     head.appendChild(tag);
     head.appendChild(caret);
     var body = document.createElement("pre");
     body.className = "gbody";
     body.textContent = "";
     node.appendChild(head);
     node.appendChild(body);
     head.onclick = function() { node.classList.toggle("open"); };
     t.trace.appendChild(node);
     scrollDown();
     t.thinking = { node: node, body: body };
     return t.thinking;
   }

   function traceReasonStep(t, step, total, label) {
     var node = document.createElement("div");
     node.className = "gnode reason";
     var head = document.createElement("div");
     head.className = "ghead";
     var tag = document.createElement("span");
     tag.className = "gtool";
     tag.textContent = "reasoning " + step + "/" + total;
     var lab = document.createElement("span");
     lab.className = "glabel muted";
     lab.textContent = label || "";
     var caret = document.createElement("span");
     caret.className = "gcaret";
     caret.textContent = "▸";
     head.appendChild(tag); head.appendChild(lab); head.appendChild(caret);
     var body = document.createElement("pre");
     body.className = "gbody";
     body.textContent = "";
     node.appendChild(head); node.appendChild(body);
     // A finished step clamps to a couple of lines and can be opened again.
     // Six steps each holding an open 160px scroller pushed the answer a
     // screen and a half down the page and gave the per-token scroll six more
     // boxes to lay out; clamping the finished ones leaves exactly one live
     // pane on screen at a time. The click only does anything once the step is
     // done, so it never collapses a step that is still streaming.
     head.onclick = function() {
       if (node.classList.contains("done")) node.classList.toggle("open");
     };
     t.trace.appendChild(node);
     scrollDown();
     return { node: node, body: body };
   }

   function traceDetail(t, message) {
     var el = document.createElement("div");
     el.className = "gdetail";
     el.textContent = message;
     t.turn.appendChild(el);
     scrollDown();
   }

   function traceNotice(t, message, info) {
     var node = document.createElement("div");
     node.className = "gnode notice" + (info ? " info" : "");
     var msg = document.createElement("div");
     msg.className = "gmsg";
     msg.textContent = message;
     node.appendChild(msg);
     t.trace.appendChild(node);
     scrollDown();
   }

   function traceAnswer(t) {
     if (t.answer) return t.answer;
     var node = document.createElement("div");
     node.className = "gnode answer pending";
     var text = document.createElement("div");
     text.className = "gtext";
     node.appendChild(text);
     t.turn.appendChild(node);
     t.answer = { node: node, text: text };
     scrollDown();
     return t.answer;
   }

   function settleAnswer(t, failed) {
     // Finish a turn that ended badly. Every exit has to agree on what a dead
     // turn looks like, and they did not: `final` cleared pending, the abort
     // handler cleared pending and marked it failed, and the error branch did
     // neither -- so a stream that died mid-answer left the text it had at
     // 75% opacity for ever, looking like it was still coming.
     //
     // An empty bubble is removed rather than settled: .gtext carries a
     // border, a background and 10px of padding, so an empty one is a visible
     // rounded box that says nothing.
     if (!t.answer) return;
     if (t.answer.mdTimer) { clearTimeout(t.answer.mdTimer); t.answer.mdTimer = null; }
     if (!t.answer.text.textContent) { dropAnswer(t); return; }
     t.answer.node.classList.remove("pending");
     if (failed) t.answer.node.classList.add("failed");
   }

   function dropAnswer(t) {
     // Remove a dead draft, don't just forget it. The tool loop streams the
     // model's tokens before it knows whether they are an answer or a tool
     // call, so what is usually in this bubble at tool_result is the raw
     // {"tool": ..., "args": ...} JSON. Clearing only the reference left the
     // node in the turn, still classed `pending` -- so a three-tool turn ended
     // with three permanently half-transparent bubbles of protocol JSON
     // stacked above the real answer, and traceAnswer built a fresh one each
     // time because it only checks the reference.
     if (!t.answer) return;
     if (t.answer.mdTimer) { clearTimeout(t.answer.mdTimer); t.answer.mdTimer = null; }
     if (t.answer.node.parentNode) t.answer.node.parentNode.removeChild(t.answer.node);
     t.answer = null;
   }

   function addFeedbackBar(userText, botText) {
     var bar = document.createElement("div");
     bar.className = "feedback";
     var up = makeBtn("Good", "Good answer: keep it as an example to learn from", null);
     var down = makeBtn("Bad", "Bad answer: learn to avoid it", null);
     var correct = makeBtn("Correct it", "Provide a corrected answer", null);
     function rated(btn) {
       up.disabled = true; down.disabled = true;
       btn.classList.add("active");
       btn.setAttribute("aria-pressed", "true");
     }
     up.onclick = function() { vote(userText, botText, 1); rated(up); };
     down.onclick = function() { vote(userText, botText, -1); rated(down); };
     correct.onclick = function() { correctAnswer(userText, botText); };
     bar.appendChild(up);
     bar.appendChild(down);
     bar.appendChild(correct);
     chat.appendChild(bar);
     scrollDown();
     return bar;
   }

   // ---------------------------------------------------- http helpers ---

   // A readable message out of an error body. FastAPI validation errors are a
   // `detail` ARRAY of {loc, msg} objects, which used to reach the user as
   // "[object Object]".
   function httpErrorText(data, status) {
     if (data && data.error) {
       if (typeof data.error === "string") return data.error;
       return data.error.message || JSON.stringify(data.error);
     }
     var detail = data && data.detail;
     if (Array.isArray(detail)) {
       return detail.map(function(item) {
         if (!item || typeof item !== "object") return String(item);
         var loc = (item.loc || []).filter(function(p) {
           return p !== "body" && p !== "query" && p !== "path";
         }).join(".");
         return (loc ? loc + ": " : "") + (item.msg || JSON.stringify(item));
       }).join("; ");
     }
     if (detail) return typeof detail === "string" ? detail : JSON.stringify(detail);
     return "HTTP " + status;
   }

   async function fetchJSON(url, options) {
     var res = await fetch(url, options);
     // A 401 on any authenticated request means the session ended (expired,
     // revoked by an admin, or logged out in another tab): return to login.
     if (res.status === 401 && typeof onUnauthorized === "function") onUnauthorized();
     var raw = await res.text();
     var data = {};
     if (raw) {
       try {
         data = JSON.parse(raw);
       } catch (err) {
         throw new Error(
           "HTTP " + res.status + " from " + (res.url || url) +
           " returned non-JSON, so this page is probably talking to the model " +
           "server instead of the web UI. Body: " + raw.slice(0, 200)
         );
       }
     }
     // Throw on any non-2xx so callers' catch blocks surface the real error
     // (a 403/400/500/503) instead of silently reading fields off an error body
     // and rendering an empty, healthy-looking state.
     if (!res.ok) {
       var e = new Error(httpErrorText(data, res.status));
       e.status = res.status;
       throw e;
     }
     return { res: res, data: data };
   }

   // Download a URL as a file via fetch+blob (not window.location), so a
   // 401/403/404/500 shows a message instead of replacing the whole app tab.
   // It used to read the body as TEXT and re-wrap it as text/plain, which is
   // fine for Markdown and silently corrupts a .docx, .pdf or .xlsx. The kit's
   // uiDownload is the one implementation; the fallback below is the same
   // thing for a page assembled without the kit.
   async function downloadFile(url, filename) {
     if (typeof uiDownload === "function") return uiDownload(url, filename);
     try {
       var resp = await fetch(url);
       if (resp.status === 401 && typeof onUnauthorized === "function") { onUnauthorized(); return; }
       if (!resp.ok) {
         var e = {}; try { e = await resp.json(); } catch (x) { e = {}; }
         uiToast("Download failed: " + httpErrorText(e, resp.status), "error");
         return;
       }
       var blob = await resp.blob();
       var a = document.createElement("a");
       a.href = URL.createObjectURL(blob);
       a.download = filename;
       document.body.appendChild(a); a.click(); a.remove();
       // Revoking in the same tick can cancel the download in some browsers.
       setTimeout(function() { URL.revokeObjectURL(a.href); }, 1500);
     } catch (err) {
       uiToast("Download failed: " + err.message, "error");
     }
   }

   function compactTokens(n) {
     n = Number(n) || 0;
     if (n >= 10000) return Math.round(n / 1000) + "k";
     if (n >= 1000) return (n / 1000).toFixed(1) + "k";
     return String(n);
   }

   function updateContextMeter(delta) {
     usedTokens = Math.max(0, usedTokens + (delta || 0));
     var size = contextSize > 0 ? contextSize : 1;
     var pct = Math.min(100, Math.round((usedTokens / size) * 100));
     var bar = document.getElementById("ctxBar");
     if (bar) {
       // The fill is data, not a look: the width is the one value a class
       // cannot carry. --fill is the same number for a stylesheet that draws
       // the track with a gradient instead.
       bar.style.width = pct + "%";
       bar.style.setProperty("--fill", pct + "%");
       if (bar.parentNode) {
         bar.parentNode.classList.toggle("warn", pct >= 75 && pct < 95);
         bar.parentNode.classList.toggle("error", pct >= 95);
       }
     }
     var hint = document.getElementById("ctxHint");
     if (hint) hint.textContent = compactTokens(usedTokens) + " / " + compactTokens(contextSize);
     var meter = document.getElementById("ctxMeter");
     if (meter) {
       meter.setAttribute("data-tip", "About " + usedTokens + " of " + contextSize
         + " tokens of context used in this conversation (" + pct + "%). "
         + "The oldest turns drop out automatically.");
     }
   }

   async function send() {
     if (busy) return;
     var message = input.value.trim();
     if (!message) return;

     input.value = "";
     input.style.height = "auto";
     busy = true;
     sendBtn.disabled = true;
     stopBtn.disabled = false;
     controller = new AbortController();
     clearRegenerate();
     addMessage("user", message);
     updateContextMeter(estimateTokens(message));

     var trace = startTrace();
     // Re-arm the follow: having scrolled up to read the last turn must not
     // leave the next one streaming off-screen.
     scrollDown(true);
     var answered = false;
     var firstTool = true;

     try {
       var res = await fetch("/api/chat/stream", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         signal: controller.signal,
         // max_tokens and temperature are deliberately absent: the server
         // fills them from this user's preferences, so the web UI, the CLI
         // and API clients all get the same per-user defaults.
         body: JSON.stringify({
           message: message,
           conversation_id: conversationId,
           agent: modeAgentFlag()
         })
       });

       if (!res.ok || !res.body) {
         var text = await res.text();
         // Session ended mid-chat (this is a raw fetch, not fetchJSON): return
         // to the login screen instead of showing a cryptic error.
         if (res.status === 401 && typeof onUnauthorized === "function") {
           onUnauthorized(); stopActivity(trace); return;
         }
         // Show the clean message, not the raw {"error": ...} JSON envelope.
         var msg = text;
         try { msg = httpErrorText(JSON.parse(text), res.status); } catch (e) { msg = text || ("HTTP " + res.status); }
         var a0 = traceAnswer(trace);
         a0.text.textContent = "Error: " + String(msg).slice(0, 400);
         a0.node.classList.remove("pending");
         a0.node.classList.add("failed");
         stopActivity(trace);
         return;
       }

       var reader = res.body.getReader();
       var decoder = new TextDecoder();
       var buffer = "";

       while (true) {
         var chunk = await reader.read();
         if (chunk.done) break;
         buffer += decoder.decode(chunk.value, { stream: true });
         var parts = buffer.split("\n\n");
         buffer = parts.pop();

         for (var i = 0; i < parts.length; i++) {
           var line = parts[i].trim();
           if (line.indexOf("data: ") !== 0) continue;
           var payload = line.slice(6);
           if (payload === "[DONE]") continue;
           var event;
           try {
             event = JSON.parse(payload);
           } catch (err) {
             continue;
           }
           handleEvent(event);
         }
       }

       function handleEvent(event) {
         if (event.type === "start") {
           conversationId = event.conversation_id || conversationId;
           safeStorageSet("llm_conversation", conversationId);
         } else if (event.type === "detail") {
           traceDetail(trace, event.message || "");
           bumpActivity(trace);
         } else if (event.type === "phase") {
           setActivity(trace, event.label || "working…");
           bumpActivity(trace);
         } else if (event.type === "reason_step") {
           trace.reason = traceReasonStep(trace, event.step, event.total, event.label);
           bumpActivity(trace);
           setActivity(trace, "reasoning " + event.step + "/" + event.total + "…");
         } else if (event.type === "reason_token") {
           if (trace.reason) {
             trace.reason.body.textContent += event.token;
             // Follow the newest line inside the step's own box as well as the
             // page: the box caps at 160px, so without this the step froze
             // visibly after eight lines while the page went on jerking.
             pinBottom(trace.reason.body);
           }
         } else if (event.type === "reason_done") {
           if (trace.reason) {
             // conclusion is a 200-character digest of text that has ALREADY
             // streamed into this body. Swapping it in collapsed a full pane
             // to two lines, mid-word and without an ellipsis, and yanked
             // everything below it up the page -- once per step. Keep what the
             // user watched arrive; the digest is only needed when nothing
             // streamed, which is the non-streaming fallback in the agent.
             if (!trace.reason.body.textContent && event.conclusion) {
               trace.reason.body.textContent = event.conclusion;
             }
             trace.reason.node.classList.add("done");
             trace.reason = null;
           }
         } else if (event.type === "context") {
           // "kept 1 of 16" is the number that matters: "trimmed 15" reads as
           // routine housekeeping right up to the point where nothing of the
           // conversation is left in front of the model.
           var kept = (typeof event.kept === "number") ? event.kept : null;
           var total = (typeof event.total === "number") ? event.total : null;
           var line = (kept !== null && total !== null)
             ? ("context window full: kept " + kept + " of " + total + " earlier messages")
             : ("trimmed " + event.dropped + " old messages to fit the context window");
           traceNotice(trace, line, kept !== 0);
           bumpActivity(trace);
         } else if (event.type === "step") {
           // Only label the step once the turn is genuinely multi-step. Four
           // out of five turns end at step 1, so a permanent "step 1/6" prefix
           // carries no information and reads like a counter that is stuck.
           trace.stepLabel = event.step > 1
             ? "step " + event.step + (event.max_steps ? "/" + event.max_steps : "")
             : "";
           setActivity(trace, "thinking…");
           // A new step supersedes whatever streamed during the last one: a
           // draft about to be revised, or a tool call the model printed
           // before deciding. Remove the bubble rather than emptying it --
           // emptying left a bordered, padded, background-filled box standing
           // empty on the page while the next step spun up, which reads worse
           // than no box at all, and an empty styled box is exactly what the
           // eye reads as something going wrong.
           if (trace.answer && trace.answer.node.classList.contains("pending")) {
             dropAnswer(trace);
           }
         } else if (event.type === "think_token") {
           // reasoning_display "hidden": no reasoning pane at all, only the
           // activity line says the model is thinking.
           if (chatPrefs.reasoning_display !== "hidden") {
             traceThinking(trace).body.textContent += event.token;
             bumpActivity(trace);
           }
           setActivity(trace, "thinking…");
           scrollDown();
         } else if (event.type === "token") {
           // First answer token: the thinking phase is over, collapse it so the
           // answer is the focus but the reasoning stays one click away
           // (unless the user asked for reasoning to stay expanded).
           if (trace.thinking && !trace.thinking.done) {
             trace.thinking.done = true;
             if (chatPrefs.reasoning_display !== "expanded") trace.thinking.node.classList.remove("open");
           }
           var ansNode = traceAnswer(trace);
           ansNode.raw = (ansNode.raw || "") + event.token;
           // Only write the growing raw text while markdown has not rendered yet
           // (or a code fence is open). Once rendered, leave the formatted DOM in
           // place and let the debounced renderMarkdown update it: writing raw
           // text every token here is what made the answer flicker.
           if (!ansNode.rendered) ansNode.text.textContent = ansNode.raw;
           scheduleMarkdown(ansNode);
           bumpActivity(trace);
           setActivity(trace, "generating…");
           scrollDown();
         } else if (event.type === "tool_call") {
           if (firstTool) { traceRouter(trace, event.name, event.args); firstTool = false; }
           trace.tool = traceTool(trace, event.name, event.args);
           trace.tool.node.classList.add("running");
           bumpActivity(trace);
           setActivity(trace, "running " + event.name + "…");
         } else if (event.type === "tool_result") {
           if (trace.tool) {
             trace.tool.node.classList.remove("running");
             trace.tool.body.textContent = event.result || "";
             if (event.error) {
               trace.tool.node.classList.add("failed");
               trace.tool.pill.textContent = "failed";
             } else {
               trace.tool.node.classList.add("ok");
               trace.tool.pill.textContent = "ok";
             }
             trace.tool = null;
           }
           dropAnswer(trace);
           trace.thinking = null;
           setActivity(trace, "thinking…");
         } else if (event.type === "notice") {
           traceNotice(trace, event.message || "", !!event.info);
           bumpActivity(trace);
           setActivity(trace, event.message || "working…");
         } else if (event.type === "final") {
           var ans = traceAnswer(trace);
           if (ans.mdTimer) { clearTimeout(ans.mdTimer); ans.mdTimer = null; }
           // Streaming re-renders on a debounce; the final pass is
           // authoritative. But when the bubble ALREADY shows exactly the
           // final answer -- now the common case, since the plain lane splits
           // <think> out as it streams instead of dumping it into this bubble
           // -- rebuilding the whole message buys nothing and costs a visible
           // re-layout at the moment the user starts reading.
           //
           // The test is what the DOM last rendered, not what the raw buffer
           // holds. Comparing raw against the final text looked equivalent and
           // is not: the line above just cancelled a pending debounce, so when
           // the last tokens arrived inside that 250ms window the raw buffer
           // matches while the DOM is still a render or two behind, and
           // skipping here would drop the tail of every answer that finished
           // mid-debounce.
           var finalText = event.answer || "(no answer)";
           if (ans.lastRendered !== finalText) {
             renderMarkdown(finalText, ans.text);
             ans.lastRendered = finalText;
             ans.raw = event.answer || "";
             ans.rendered = true;
           }
           ans.node.classList.remove("pending");
           answered = true;
           updateContextMeter(estimateTokens(event.answer || ""));
           var bits = [];
           if (event.tools_used && event.tools_used.length) bits.push(event.tools_used.join(", "));
           if (chatPrefs.show_metrics) {
             bits.push((event.steps || 1) + " step" + ((event.steps || 1) === 1 ? "" : "s"));
             // Cost of the turn, in the shape a local runner prints it:
             // "2048 tok - 100.4s - 20 tok/s". tok is what the model generated
             // across every step, and the rate is against wall-clock, so on an
             // agent turn it includes the time tools took -- which is the
             // number you actually waited, not the decode speed of one step
             // (that is in the usage rows under Performance).
             var secs = (event.elapsed_ms || 0) / 1000;
             var tok = event.completion_tokens || 0;
             if (tok) bits.push(tok + " tok");
             bits.push(secs.toFixed(1) + "s");
             if (tok && secs > 0) {
               var tps = tok / secs;
               bits.push((tps >= 10 ? Math.round(tps) : tps.toFixed(1)) + " tok/s");
             }
           }
           if (event.truncated) bits.push("continued from summary");
           stopActivity(trace);
           if (bits.length) {
             var meta = document.createElement("div");
             meta.className = "gmeta";
             meta.textContent = bits.join(" · ");
             trace.turn.appendChild(meta);
           }
           if (event.documents && event.documents.length) {
             // A generated document is the deliverable, not a diff to review, so
             // it gets its own row of download links above the changed-files
             // node -- which would otherwise present a .docx as an edit to
             // inspect with git before pushing.
             var docs = document.createElement("div");
             docs.className = "docrow";
             event.documents.forEach(function(rel) {
               var name = rel.split("/").pop();
               var a = document.createElement("a");
               a.className = "docdl";
               a.href = "/api/files/download?path=" + encodeURIComponent(rel);
               a.setAttribute("download", name);
               a.textContent = "⬇︎  " + name;
               a.title = "Download " + rel;
               // Fetched, not navigated: an expired session or a moved file
               // shows a toast instead of replacing the app with an error body.
               a.onclick = function(e) { e.preventDefault(); downloadFile(a.href, name); };
               docs.appendChild(a);
             });
             trace.turn.appendChild(docs);
           }
           // The document files are already offered above as downloads; listing
           // them again as "changed files (review before push)" reads as a
           // warning about the very thing the user asked for.
           var changed = (event.changed_files || []).filter(function(f) {
             return (event.documents || []).indexOf(f) === -1;
           });
           if (changed.length) {
             var wrap = document.createElement("div");
             wrap.className = "gnode diff open";
             var head = document.createElement("div");
             head.className = "ghead";
             var tag = document.createElement("span");
             tag.className = "gtool";
             tag.textContent = "changed " + changed.length + " file"
               + (changed.length === 1 ? "" : "s") + " (review before push)";
             var caret = document.createElement("span");
             caret.className = "gcaret";
             caret.textContent = "▸";
             head.appendChild(tag); head.appendChild(caret);
             var body = document.createElement("pre");
             body.className = "gbody gdiff";
             body.textContent = event.diff && event.diff.trim()
               ? event.diff
               : (changed.join("\n") + "\n\n(new files, not yet tracked by git)");
             head.onclick = function() { wrap.classList.toggle("open"); };
             wrap.appendChild(head); wrap.appendChild(body);
             trace.turn.appendChild(wrap);
           }
           setRegenerateTarget(addFeedbackBar(message, event.answer || ""));
           // Both routes are ADMIN. Every non-admin answer used to fire two
           // requests that could only ever come back 403.
           if (roleIsAdmin()) {
             loadRunCurve(conversationId);
             loadPerf();
           }
         } else if (event.type === "cancelled") {
           // settleAnswer, not a bare notice. This branch and the one below
           // it were both `event.type === "cancelled"`, so the second was
           // unreachable and the settling it was added to do never ran: a
           // server-side cancel (agent.py yields `cancelled` from five
           // places, and api.py forwards every event) left the partial
           // answer classed `pending` for good -- greyed out at 0.75 opacity
           // with no `.failed` marking, the same permanently-unsettled
           // bubble the error branch below was fixed for.
           settleAnswer(trace, false);
           traceNotice(trace, "stopped", false);
           stopActivity(trace);
           answered = true;
         } else if (event.type === "error") {
           traceNotice(trace, "error: " + event.error, false);
           // Settle whatever streamed before the error. Marking the turn
           // answered skips the cleanup below, so without this the partial
           // text sat at the pending opacity for good -- a half-written answer
           // permanently greyed out, with none of the .failed styling the
           // AbortError path twenty lines down applies for the same situation.
           // The two failure paths disagreed about what a dead turn looks like.
           settleAnswer(trace, true);
           stopActivity(trace);
           answered = true;
         }
       }

       if (!answered) {
         var a = traceAnswer(trace);
         a.node.classList.remove("pending");
         if (!a.text.textContent) a.text.textContent = "(no response)";
       }
       stopActivity(trace);
     } catch (err) {
       if (err.name === "AbortError") {
         // Stop pressed. Settle whatever streamed; if nothing did, leave no
         // bubble at all. traceAnswer used to be called unconditionally here,
         // so stopping before the first token CREATED an answer node and then
         // left it empty -- a bordered, padded, background-filled box with
         // nothing in it, which reads as something having gone wrong.
         settleAnswer(trace, false);
         traceNotice(trace, "stopped", false);
       } else {
         var a2 = traceAnswer(trace);
         a2.node.classList.remove("pending");
         // A broken stream on a local single-box setup almost always means the
         // model server ran out of memory and dropped the connection. Say that
         // plainly instead of showing a raw stream error.
         if (!a2.text.textContent) {
           a2.text.textContent = "The connection dropped, most likely the local "
             + "model server ran low on memory. Try a shorter request, or lower "
             + "AUTO_FETCH_RESULTS / the context size.";
         }
         a2.node.classList.add("failed");
       }
       stopActivity(trace);
     } finally {
       busy = false;
       controller = null;
       sendBtn.disabled = false;
       stopBtn.disabled = true;
       input.focus();
     }
   }

   // ------------------------------------------------- chat overflow menu ---

   function toggleChatMenu(force) {
     var menu = document.getElementById("chatMenu");
     var btn = document.getElementById("chatMenuBtn");
     if (!menu) return;
     var open = (typeof force === "boolean") ? force : menu.classList.contains("hidden");
     menu.classList.toggle("hidden", !open);
     if (btn) btn.setAttribute("aria-expanded", open ? "true" : "false");
     if (open) {
       var first = menu.querySelector("button");
       if (first) first.focus();
     }
   }

   function closeChatMenu() { toggleChatMenu(false); }

   function chatMenuAction(action) {
     closeChatMenu();
     if (action === "export") exportChat();
     else if (action === "regenerate") regenerateLast();
     else if (action === "search") openHistorySearch();
     else if (action === "prompts") togglePrompts();
   }

   // Close the menus on an outside click or Escape.
   document.addEventListener("click", function(e) {
     var menu = document.getElementById("chatMenu");
     if (!menu || menu.classList.contains("hidden")) return;
     if (e.target.closest && e.target.closest("#chatMenu, #chatMenuBtn")) return;
     closeChatMenu();
   });
   document.addEventListener("keydown", function(e) {
     if (e.key !== "Escape") return;
     var menu = document.getElementById("chatMenu");
     if (menu && !menu.classList.contains("hidden")) {
       closeChatMenu();
       var btn = document.getElementById("chatMenuBtn");
       if (btn) btn.focus();
     }
   });

   // ----------------------------------------------------------- memory ---

   async function loadMemory() {
     var list = document.getElementById("memoryList");
     if (!list) return;
     try {
       var out = await fetchJSON("/api/memory?limit=50");
       var items = out.data.memories || [];
       list.innerHTML = "";
       if (!items.length) {
         list.textContent = "No stored notes.";
         return;
       }
       items.forEach(function(item) {
         var row = document.createElement("div");
         row.className = "row memory-item";
         var label = document.createElement("span");
         label.className = "grow mono";
         label.textContent = item.key + ": " + item.value;
         row.appendChild(label);
         row.appendChild(makeBtn("Forget", "Forget this note", function() { forgetMemory(item.key); },
                                 "btn-danger"));
         list.appendChild(row);
       });
     } catch (err) {
       list.textContent = "Memory unavailable: " + err.message;
     }
   }

   async function saveMemory() {
     var keyEl = document.getElementById("memKey");
     var valueEl = document.getElementById("memValue");
     if (!keyEl || !valueEl) return;
     var key = keyEl.value.trim();
     var value = valueEl.value.trim();
     if (!key || !value) { uiToast("Enter both a key and a value.", "warn"); return; }
     try {
       await fetchJSON("/api/memory", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify({ key: key, value: value })
       });
       keyEl.value = "";
       valueEl.value = "";
       uiToast("Stored " + key + ".", "ok");
       loadMemory();
     } catch (err) {
       uiToast("Memory error: " + err.message, "error");
     }
   }

   async function forgetMemory(key) {
     try {
       await fetchJSON("/api/memory/" + encodeURIComponent(key), { method: "DELETE" });
       loadMemory();
     } catch (err) {
       uiToast("Memory error: " + err.message, "error");
     }
   }

   // ---------------------------------------------- feedback and training ---

   function reportLearning(learning) {
     // Skill authoring happens in the background AFTER this response, so this
     // reports only what is known synchronously: which procedures the answer
     // used, and any that were retired for a bad track record.
     if (!learning) return;
     var used = learning.skills_used || [];
     if (used.length) uiToast("Credited skill(s): " + used.join(", "), "info");
     var gone = learning.retired || [];
     if (gone.length) uiToast("Retired skill(s) for a poor record: " + gone.join(", "), "warn");
   }

   async function vote(userPrompt, assistantResponse, rating) {
     try {
       var out = await fetchJSON("/api/feedback", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify({
           user_prompt: userPrompt,
           assistant_response: assistantResponse,
           rating: rating,
           corrected_response: null,
           // So the server can credit the skills this answer actually used.
           conversation_id: conversationId
         })
       });
       uiToast(out.data.status || "Feedback saved.", "ok");
       reportLearning(out.data.learning);
     } catch (err) {
       uiToast("Feedback error: " + err.message, "error");
     }
   }

   async function correctAnswer(userPrompt, assistantResponse) {
     var corrected = window.prompt("Corrected answer:", assistantResponse);
     if (corrected === null) return;
     try {
       var out = await fetchJSON("/api/feedback", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify({
           user_prompt: userPrompt,
           assistant_response: assistantResponse,
           rating: 1,
           corrected_response: corrected,
           conversation_id: conversationId
         })
       });
       uiToast(out.data.status || "Correction saved.", "ok");
       reportLearning(out.data.learning);
     } catch (err) {
       uiToast("Correction error: " + err.message, "error");
     }
   }

   async function retrain() {
     if (!(await uiConfirm("Start retraining? The model server will be stopped temporarily."))) return;
     try {
       var out = await fetchJSON("/api/retrain", { method: "POST" });
       uiToast(out.data.status || "Retraining started.", "ok");
     } catch (err) {
       uiToast("Retrain error: " + err.message, "error");
     }
   }

   async function newChat() {
     // Abort a turn still in flight first. Without this the old stream keeps
     // running against nodes that have just been removed from the page: the
     // Send button stays disabled for up to a minute so the new chat looks
     // broken, the final event appends a feedback bar into the cleared chat
     // with nothing above it, and that answer's tokens are charged to the new
     // conversation's context meter, which newChat has just reset to zero.
     //
     // Switch to the chat first: leaving Settings with unsaved changes asks
     // for confirmation, and a "stay" answer must not cost the user the
     // conversation they had open.
     if (currentView !== "chat") {
       showView("chat");
       if (currentView !== "chat") return;
     }
     stopStream();
     closeChatMenu();
     conversationId = randomId();
     safeStorageSet("llm_conversation", conversationId);
     chat.innerHTML = "";
     usedTokens = 0;
     updateContextMeter(0);
     showChatEmpty();
     input.focus();
   }

   async function restartModel() {
     try {
       var out = await fetchJSON("/api/model/restart", { method: "POST" });
       uiToast("Model server restarting. KV cache: " + out.data.max_kv_size, "ok");
     } catch (err) {
       uiToast("Restart error: " + err.message, "error");
     }
   }

   // ----------------------------------------------------- status pill ---

   function healthState(status) {
     status = String(status || "unknown");
     if (status === "ready") return { word: "Ready", cls: "ok" };
     if (status === "starting" || status === "loading" || status === "restarting") {
       return { word: status.charAt(0).toUpperCase() + status.slice(1) + "…", cls: "warn" };
     }
     if (status.indexOf("error") === 0) return { word: "Model error", cls: "error" };
     if (status === "stopped") return { word: "Model stopped", cls: "warn" };
     return { word: status, cls: "muted" };
   }

   async function refreshHealth() {
     try {
       var out = await fetchJSON("/api/health");
       var data = out.data;
       // Health carries the context size for every role, so the composer's
       // meter is right for non-admins too.
       if (typeof data.context_size === "number" && data.context_size !== contextSize) {
         contextSize = data.context_size;
         updateContextMeter(0);
       }
       // The pill says one thing at a glance; everything else goes to the
       // tooltip, which has room to wrap.
       var state = healthState(data.model_status);
       var brief = [state.word];
       var cls = state.cls;
       var detail = [];
       var stale = data.ui_build && data.ui_build !== "{{UI_BUILD}}";
       if (stale) {
         brief = ["Stale page: reload"];
         cls = "error";
         detail.push("Server is build " + data.ui_build + ", this tab is {{UI_BUILD}}.");
       }
       detail.push("Model status: " + (data.model_status || "unknown"));
       detail.push("agent " + (data.agent_enabled ? "on" : "off") + " by default");
       detail.push("Max reply " + data.max_tokens + " tokens");
       detail.push((data.tools ? data.tools.length : 0) + " tools");
       var retrain = (data.retrain && data.retrain.message) || "idle";
       if (retrain !== "idle") { brief.push("retrain: " + retrain); if (cls === "ok") cls = "running"; }
       else if (data.retrain) detail.push("Retrain idle");
       // What the live adapter was actually trained on, and whether the run
       // improved the held-out loss. Without this a bad adapter is invisible.
       var recipe = document.getElementById("retrainRecipe");
       if (recipe) {
         var meta = data.adapter_meta;
         if (meta && meta.trained_at) {
           var c = meta.counts || {};
           var line = "Live adapter: " + (c.feedback || 0) + " feedback, "
                    + (c.tool || 0) + " tool, " + (c.replay || 0) + " rehearsal, "
                    + ((meta.plan && meta.plan.iters) || "?") + " iterations, "
                    + meta.trained_at.slice(0, 10) + ".";
           if (meta.val_loss_baseline != null && meta.val_loss_final != null) {
             line += " Held-out loss " + meta.val_loss_baseline.toFixed(3)
                   + " → " + meta.val_loss_final.toFixed(3) + ".";
           }
           recipe.textContent = line;
         } else if (data.retrain) {
           recipe.textContent = "No trained adapter yet: the base model is live.";
         }
       }
       if (data.stats) {
         detail.push("Feedback: " + data.stats.total + " total, "
                     + data.stats.approved + " approved, "
                     + data.stats.untrained + " untrained");
         if (data.stats.pending) detail.push(data.stats.pending + " awaiting approval");
       }
       if (typeof data.memories === "number") {
         detail.push(data.memories + " memory notes");
       }
       var warn = document.getElementById("prefixWarn");
       if (warn) {
         var churn = !!(data.prefix && data.prefix.generation > 1);
         warn.classList.toggle("hidden", !churn);
         warn.hidden = !churn;
         if (churn) {
           warn.textContent = "Prompt prefix changed " + (data.prefix.generation - 1) +
             " time(s) this session (last " + (data.prefix.changed_at || "") +
             "). Each change invalidates any cached prefix.";
         }
       }
       if (data.tasks) {
         var running = data.tasks.running || [];
         if (running.length) {
           brief.push(running.length + " task" + (running.length > 1 ? "s" : ""));
           if (cls === "ok") cls = "running";
           detail.push("Running: " + running.map(function(item) {
             return "step " + item.step + "/" + item.max_steps;
           }).join(", "));
         }
         detail.push(data.tasks.total + " tasks defined");
       }
       var text = brief.join(" · ");
       if (statusEl.textContent !== text) statusEl.textContent = text;
       var pillClass = "pill status-pill " + cls;
       if (statusEl.className !== pillClass) statusEl.className = pillClass;
       statusEl.setAttribute("data-tip-below", "");
       statusEl.setAttribute("data-tip",
         "Model " + (data.model || "?")
         + " · context " + (data.context_size || contextSize) + " tokens"
         + (data.ram_gb ? " · " + data.ram_gb + "GB RAM" : "")
         + (detail.length ? " · " + detail.join(" · ") : ""));
     } catch (err) {
       statusEl.textContent = "Offline";
       statusEl.className = "pill status-pill error";
       statusEl.setAttribute("data-tip", "Status unavailable: " + err.message);
     }
   }
"""

__all__ = ["UI_JS_CHAT"]
