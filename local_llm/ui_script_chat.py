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

The stream is one run object per turn (chatRun) read by readSSE, a parser
that follows the SSE grammar (CRLF, multi-line data, id: lines, keepalive
comments). Every request sends X-Stream-Ids: 1, so frames are numbered and a
turn survives a dropped connection: chatFollow re-attaches from the last
sequence number, and boot re-attaches to a turn still running for the open
conversation (GET /api/chat/active, then /api/chat/turns/{id}/events?after=N).
Stop cancels the turn on the server before it aborts the connection. The
trace renders tool calls as they stream (keyed by call id), inline decision
cards for tool approvals and cloud consent, queue position, injected project
context and live speed figures. The composer carries the project and agent
pickers, attachments (paperclip, drop, paste) and dictation; the header
carries the notifications bell.
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

   // Let go of the turn this tab is streaming, now. New chat, opening another
   // conversation and deleting this one call it before clearing the page, so
   // the late frames of the old turn must not land in the new one: the run
   // is marked detached (its frames are ignored) and the controls are reset
   // here rather than when the aborted fetch gets round to failing. The turn
   // is also cancelled on the server, because a stream opened with
   // X-Stream-Ids only detaches on abort and would keep generating; with
   // {detach: true} it is left running (reopening the same conversation
   // re-attaches to it). The Stop button goes through chatStop() instead.
   function stopStream(opts) {
     var run = chatRun;
     if (run) {
       run.detached = true;
       if (!(opts && opts.detach) && !run.ended) chatCancelTurn(run);
       chatRun = null;
     }
     if (controller) controller.abort();
     controller = null;
     busy = false;
     sendBtn.disabled = false;
     stopBtn.disabled = true;
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
     // The rest of the per-user composer state rides on the same boot call:
     // the project and agent pickers, the paperclip and mic, the bell.
     chatLoadComposerOptions();
     chatPollNotifications(true);
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

   function traceRouter(t, name, args, before) {
     var node = document.createElement("div");
     node.className = "gnode router";
     var head = document.createElement("div");
     head.className = "ghead";
     var loc = args && (args.location || args.query);
     head.textContent = "router → " + name + (loc ? " (" + String(loc).slice(0, 60) + ")" : "");
     node.appendChild(head);
     // Above the tool card it routed to: with streamed tool calls the card
     // exists before the call is final.
     if (before && before.parentNode === t.trace) t.trace.insertBefore(node, before);
     else t.trace.appendChild(node);
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
     return { node: node, pill: pill, body: body, label: label, argsEl: argsEl };
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

   // ------------------------------------------------------- streaming ---

   // The turn this tab is streaming (null when idle). One object per turn
   // instead of closure variables inside send(), because the same turn can
   // now be picked up again by a later request: after a dropped connection,
   // or after a reload, from GET /api/chat/turns/{id}/events?after=N.
   var chatRun = null;

   // Read a text/event-stream body and call onFrame({id, data}) per event.
   // The old reader split on a blank line and kept only lines starting with
   // "data: ", which is right for exactly one server: a proxy that rewrites
   // line ends to CRLF, a frame split over several data: lines, or the id:
   // line X-Stream-Ids puts in front of every frame all lost events without
   // a trace. This follows the SSE grammar: CR, LF or CRLF end a line, a
   // leading ":" is a comment (the server's keepalive), data lines join with
   // a newline, and a blank line dispatches the event.
   async function readSSE(res, onFrame) {
     var reader = res.body.getReader();
     var decoder = new TextDecoder();
     var buf = "";
     var data = [];
     var hasData = false;
     var id = null;
     function chatSseLine(text) {
       if (text === "") {
         if (hasData) onFrame({ id: id, data: data.join("\n") });
         data = []; hasData = false; id = null;
         return;
       }
       if (text.charAt(0) === ":") return;
       var colon = text.indexOf(":");
       var field = colon < 0 ? text : text.slice(0, colon);
       var value = colon < 0 ? "" : text.slice(colon + 1);
       if (value.charAt(0) === " ") value = value.slice(1);
       if (field === "data") { data.push(value); hasData = true; }
       else if (field === "id" && value.indexOf("\u0000") < 0) id = value;
     }
     function chatSseDrain(atEnd) {
       while (buf) {
         var cut = buf.search(/[\r\n]/);
         if (cut < 0) return;
         // A CR as the very last character may be the first half of a CRLF
         // whose LF is still in flight: wait for the next chunk.
         if (buf.charAt(cut) === "\r" && cut === buf.length - 1 && !atEnd) return;
         var width = (buf.charAt(cut) === "\r" && buf.charAt(cut + 1) === "\n") ? 2 : 1;
         chatSseLine(buf.slice(0, cut));
         buf = buf.slice(cut + width);
       }
     }
     while (true) {
       var chunk = await reader.read();
       if (chunk.done) break;
       buf += decoder.decode(chunk.value, { stream: true });
       chatSseDrain(false);
     }
     buf += decoder.decode();
     chatSseDrain(true);
     // An event without its closing blank line is incomplete by the spec (and
     // the old reader dropped it too): nothing more to dispatch.
   }

   function chatAbortError() {
     var err = new Error("aborted");
     err.name = "AbortError";
     return err;
   }

   function chatSleep(ms, signal) {
     return new Promise(function(resolve, reject) {
       if (signal && signal.aborted) { reject(chatAbortError()); return; }
       var timer = setTimeout(resolve, ms);
       if (signal) signal.addEventListener("abort", function() {
         clearTimeout(timer);
         reject(chatAbortError());
       });
     });
   }

   function chatBeginRun(message, opts) {
     opts = opts || {};
     busy = true;
     sendBtn.disabled = true;
     stopBtn.disabled = false;
     controller = new AbortController();
     var run = {
       trace: startTrace(), message: message || "", conversationId: conversationId,
       turnId: opts.turnId || null, projectId: opts.projectId || "", lastSeq: 0,
       controller: controller, answered: false, ended: false, done: false, gone: false,
       firstTool: true, tools: {}, lastTool: null, sawThinking: false,
       firstUsage: null, usage: null, stopping: false, detached: false,
       cancelSent: false, cancelAccepted: false, resumed: !!opts.resumed, reattached: false, events: 0
     };
     chatRun = run;
     // Re-arm the follow: having scrolled up to read the last turn must not
     // leave the next one streaming off-screen.
     scrollDown(true);
     return run;
   }

   function chatEndRun(run) {
     // A run stopStream() already let go of must not reset the controls a
     // newer run now owns.
     if (chatRun !== run) return;
     chatRun = null;
     busy = false;
     controller = null;
     sendBtn.disabled = false;
     stopBtn.disabled = true;
   }

   // One SSE response into the run. With X-Stream-Ids every frame carries its
   // sequence number, so a frame at or below the last one rendered is the
   // overlap of a resume and is skipped: nothing renders twice.
   async function chatPump(run, res) {
     await readSSE(res, function(frame) {
       if (run.detached) return;
       if (frame.data === "[DONE]") { run.done = true; return; }
       if (frame.id !== null && frame.id !== "") {
         var seq = parseInt(frame.id, 10);
         if (!isNaN(seq)) {
           if (seq <= run.lastSeq) return;
           run.lastSeq = seq;
         }
       }
       var event;
       try { event = JSON.parse(frame.data); } catch (err) { return; }
       if (!event || typeof event !== "object") return;
       run.events += 1;
       handleEvent(run, event);
     });
   }

   // The turn a conversation is running on the server, for a run that lost
   // its connection before the start event named it (or for a Stop pressed
   // while the turn was still queued).
   async function chatFindTurn(run) {
     try {
       var out = await fetchJSON("/api/chat/active?conversation_id=" + encodeURIComponent(run.conversationId));
       var snap = out.data && out.data.turn;
       if (snap && snap.turn_id) run.turnId = snap.turn_id;
     } catch (err) { /* an older server, or offline: nothing to find */ }
     return run.turnId;
   }

   // Re-attach to a running turn from the event after the last one rendered.
   // null (and run.gone) when the turn no longer exists: it finished long
   // enough ago to be pruned, or the server restarted.
   async function chatOpenEvents(run) {
     var res = await fetch("/api/chat/turns/" + encodeURIComponent(run.turnId)
                           + "/events?after=" + run.lastSeq,
                           { headers: { "X-Stream-Ids": "1" }, signal: run.controller.signal });
     if (res.status === 401 && typeof onUnauthorized === "function") {
       onUnauthorized(); run.gone = true; return null;
     }
     if (res.status === 404) { run.gone = true; return null; }
     if (!res.ok || !res.body) return null;
     run.reattached = true;
     return res;
   }

   // Follow a turn to its end. A connection that drops before the turn ended
   // (no final, cancelled or error, no [DONE]) is picked up again from the
   // last sequence number instead of being reported as a failure: the server
   // keeps a turn streamed with X-Stream-Ids running for a grace period so a
   // flaky network or a laptop lid does not cost the answer.
   async function chatFollow(run, res) {
     var attempts = 0;
     var lastErr = null;
     while (true) {
       if (res) {
         var seen = run.events;
         try {
           await chatPump(run, res);
         } catch (err) {
           if (err && err.name === "AbortError") throw err;
           lastErr = err;
         }
         if (run.events > seen) attempts = 0;
       }
       if (run.ended || run.done || run.detached) return;
       if (run.gone || attempts >= 6) {
         if (lastErr) throw lastErr;
         return;
       }
       attempts += 1;
       setActivity(run.trace, "connection lost, reconnecting (try " + attempts + ")…");
       await chatSleep(Math.min(8000, 500 * Math.pow(2, attempts - 1)), run.controller.signal);
       res = null;
       try {
         if (!run.turnId && !(await chatFindTurn(run))) { run.gone = true; continue; }
         res = await chatOpenEvents(run);
       } catch (err) {
         if (err && err.name === "AbortError") throw err;
         lastErr = err;
       }
     }
   }

   // The stream ended without an error: settle what is on screen.
   function chatSettleRun(run) {
     chatDropStartedTools(run);
     if (run.resumed && !run.events && !run.answered) {
       // Re-attached to a turn that finished in between: its answer is in
       // the conversation already, so an empty trace would only add noise.
       stopActivity(run.trace);
       if (run.trace.turn.parentNode) run.trace.turn.parentNode.removeChild(run.trace.turn);
       return;
     }
     if (!run.answered) {
       var a = traceAnswer(run.trace);
       a.node.classList.remove("pending");
       if (!a.text.textContent) {
         a.text.textContent = run.gone
           ? "The connection dropped and this turn is no longer running. Reopen the conversation from History to see what was stored."
           : "(no response)";
       }
     }
     stopActivity(run.trace);
   }

   function chatFailRun(run, err) {
     chatDropStartedTools(run);
     if (err && err.name === "AbortError") {
       // Stop pressed. Settle whatever streamed; if nothing did, leave no
       // bubble at all. traceAnswer used to be called unconditionally here,
       // so stopping before the first token CREATED an answer node and then
       // left it empty: a bordered, padded box with nothing in it, which
       // reads as something having gone wrong. A turn whose own `cancelled`
       // event already arrived has said "stopped" once.
       if (!run.ended) {
         settleAnswer(run.trace, false);
         traceNotice(run.trace, "stopped", false);
       }
     } else {
       var a2 = traceAnswer(run.trace);
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
     stopActivity(run.trace);
   }

   async function send() {
     if (busy) return;
     var message = input.value.trim();
     if (chatAttBusy()) { uiToast("Wait for the attachment upload to finish.", "warn"); return; }
     var atts = chatAttReady();
     if (!message) {
       if (atts.length) uiToast("Add a message to send with the attachment.", "warn");
       return;
     }

     input.value = "";
     input.style.height = "auto";
     clearRegenerate();
     var userDiv = addMessage("user", message);
     if (atts.length) chatRenderMsgAtts(userDiv, atts);
     chatAttClear(false);
     updateContextMeter(estimateTokens(message));

     var projectId = chatSelectValue("projectSel");
     var agentId = chatSelectValue("agentSel");
     var run = chatBeginRun(message, { projectId: projectId });
     // max_tokens and temperature are deliberately absent: the server fills
     // them from this user's preferences, so the web UI, the CLI and API
     // clients all get the same per-user defaults. The optional fields go
     // out only when chosen, so a turn without a project, an agent profile
     // or attachments is exactly the request it always was.
     var body = { message: message, conversation_id: conversationId, agent: modeAgentFlag() };
     if (projectId) body.project_id = projectId;
     if (agentId) body.agent_id = agentId;
     if (atts.length) body.attachments = atts.map(function(a) { return a.id; });

     try {
       var res = await fetch("/api/chat/stream", {
         method: "POST",
         // X-Stream-Ids: 1 numbers every frame and keeps the turn running
         // through a dropped connection, which is what makes Stop (an
         // explicit cancel) and resume (events?after=N) possible.
         headers: { "Content-Type": "application/json", "X-Stream-Ids": "1" },
         signal: run.controller.signal,
         body: JSON.stringify(body)
       });

       if (!res.ok || !res.body) {
         var text = await res.text();
         // Session ended mid-chat (this is a raw fetch, not fetchJSON): return
         // to the login screen instead of showing a cryptic error.
         if (res.status === 401 && typeof onUnauthorized === "function") {
           onUnauthorized(); stopActivity(run.trace); return;
         }
         // Show the clean message, not the raw {"error": ...} JSON envelope.
         var msg = text;
         try { msg = httpErrorText(JSON.parse(text), res.status); } catch (e) { msg = text || ("HTTP " + res.status); }
         var a0 = traceAnswer(run.trace);
         a0.text.textContent = "Error: " + String(msg).slice(0, 400);
         a0.node.classList.remove("pending");
         a0.node.classList.add("failed");
         stopActivity(run.trace);
         return;
       }

       await chatFollow(run, res);
       chatSettleRun(run);
     } catch (err) {
       chatFailRun(run, err);
     } finally {
       chatEndRun(run);
       input.focus();
     }
   }

   // After a reload (or reopening the conversation) a turn may still be
   // running on the server. Re-attach to it from its first buffered event,
   // so the page shows the same live trace instead of a question with no
   // answer that silently appears in History later.
   async function chatResumeActive(cid, lastUserText) {
     if (busy || chatRun) return;
     var snap = null;
     try {
       var out = await fetchJSON("/api/chat/active?conversation_id=" + encodeURIComponent(cid));
       snap = out.data && out.data.turn;
     } catch (err) { return; }
     if (!snap || !snap.turn_id || snap.done || cid !== conversationId || busy || chatRun) return;
     var run = chatBeginRun(lastUserText || "", { turnId: snap.turn_id, resumed: true });
     setActivity(run.trace, "reconnecting to the running turn…");
     try {
       var res = await chatOpenEvents(run);
       await chatFollow(run, res);
       chatSettleRun(run);
     } catch (err) {
       chatFailRun(run, err);
     } finally {
       chatEndRun(run);
     }
   }

   // POST /api/chat/turns/{turn_id}/cancel. True when the server took it.
   async function chatCancelTurn(run) {
     if (run.cancelSent) return run.cancelAccepted;
     run.cancelSent = true;
     try {
       var tid = run.turnId || (await chatFindTurn(run));
       if (!tid) return false;
       var out = await postJSON("/api/chat/turns/" + encodeURIComponent(tid) + "/cancel", {});
       run.cancelAccepted = !!(out.data && out.data.cancelled);
     } catch (err) {
       run.cancelAccepted = false;
     }
     return run.cancelAccepted;
   }

   // The Stop button. Cancel the turn on the server first, then give its
   // `cancelled` event (which carries the partial answer the server stored)
   // a moment to arrive; abort the connection only if it does not. Aborting
   // first no longer stops anything: a stream opened with X-Stream-Ids only
   // detaches, and the turn would go on generating for the grace period.
   async function chatStop() {
     var run = chatRun;
     if (!run || run.ended || run.stopping) { stopStream(); return; }
     run.stopping = true;
     setActivity(run.trace, "stopping…");
     var accepted = await chatCancelTurn(run);
     if (accepted) {
       var waited = 0;
       while (chatRun === run && !run.ended && !run.done && waited < 2500) {
         await chatSleep(100);
         waited += 100;
       }
     }
     if (chatRun === run && !run.ended) stopStream();
   }

   // ------------------------------------------------------ stream events ---

   function handleEvent(run, event) {
     var trace = run.trace;
     if (event.type === "start") {
       conversationId = event.conversation_id || conversationId;
       run.conversationId = conversationId;
       if (event.turn_id) run.turnId = event.turn_id;
       safeStorageSet("llm_conversation", conversationId);
     } else if (event.type === "queued") {
       // Every generation slot is busy: say where this turn is in the line
       // instead of a spinner that looks hung.
       setActivity(trace, chatQueuedLabel(event));
       bumpActivity(trace);
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
         // everything below it up the page, once per step. Keep what the
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
     } else if (event.type === "context_injected") {
       // The project's instruction files the agent reads this turn: the
       // model follows them, so the user gets to see that they are there.
       traceNotice(trace, chatInjectedLabel(event), true);
       if (event.project_id && !run.projectId) run.projectId = event.project_id;
       bumpActivity(trace);
     } else if (event.type === "step") {
       // A tool card whose call never materialised (the model started
       // printing one, then answered instead) is noise once the step is over.
       chatDropStartedTools(run);
       // Only label the step once the turn is genuinely multi-step. Four
       // out of five turns end at step 1, so a permanent "step 1/6" prefix
       // carries no information and reads like a counter that is stuck.
       trace.stepLabel = event.step > 1
         ? "step " + event.step + (event.max_steps ? "/" + event.max_steps : "")
         : "";
       setActivity(trace, "thinking…");
       // A new step supersedes whatever streamed during the last one: a
       // draft about to be revised, or a tool call the model printed
       // before deciding. Remove the bubble rather than emptying it: an
       // empty bordered, padded box standing on the page while the next
       // step spins up is exactly what the eye reads as something wrong.
       if (trace.answer && trace.answer.node.classList.contains("pending")) {
         dropAnswer(trace);
       }
     } else if (event.type === "think_token") {
       // reasoning_display "hidden": no reasoning pane at all, only the
       // activity line says the model is thinking.
       if (chatPrefs.reasoning_display !== "hidden") {
         traceThinking(trace).body.textContent += event.token;
         run.sawThinking = true;
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
     } else if (event.type === "tool_call_start") {
       chatToolStart(run, event);
       bumpActivity(trace);
       setActivity(trace, "calling " + (event.name || "a tool") + "…");
     } else if (event.type === "tool_call_delta") {
       chatToolDelta(run, event);
     } else if (event.type === "tool_call") {
       var called = chatToolCall(run, event);
       if (run.firstTool) { traceRouter(trace, event.name, event.args, called.node); run.firstTool = false; }
       bumpActivity(trace);
       setActivity(trace, "running " + event.name + "…");
     } else if (event.type === "tool_result") {
       chatToolResult(run, event);
       dropAnswer(trace);
       trace.thinking = null;
       setActivity(trace, "thinking…");
     } else if (event.type === "approval_required") {
       chatApprovalCard(run, event);
       bumpActivity(trace);
       setActivity(trace, "waiting for your approval: " + (event.name || "tool"));
     } else if (event.type === "approval_resolved") {
       chatApprovalResolved(run, event);
       setActivity(trace, "thinking…");
     } else if (event.type === "consent_required") {
       chatConsentCard(run, event);
       bumpActivity(trace);
       setActivity(trace, "waiting for your consent to use the cloud model");
     } else if (event.type === "consent_resolved") {
       chatDecisionResolved("consent", event);
       setActivity(trace, "thinking…");
     } else if (event.type === "usage") {
       chatNoteUsage(run, event);
     } else if (event.type === "notice") {
       traceNotice(trace, event.message || "", !!event.info);
       bumpActivity(trace);
       setActivity(trace, event.message || "working…");
     } else if (event.type === "final") {
       chatDropStartedTools(run);
       if (event.turn_id) run.turnId = event.turn_id;
       // The reasoning of a turn whose thinking did not stream (the plain
       // lane splits <think> out at the end, a cloud model returns it whole)
       // still gets its collapsed pane, per the reasoning_display preference.
       if (event.reasoning && !run.sawThinking && chatPrefs.reasoning_display !== "hidden") {
         var th = traceThinking(trace);
         th.body.textContent = event.reasoning;
         th.done = true;
         if (chatPrefs.reasoning_display !== "expanded") th.node.classList.remove("open");
       }
       var ans = traceAnswer(trace);
       if (ans.mdTimer) { clearTimeout(ans.mdTimer); ans.mdTimer = null; }
       // Streaming re-renders on a debounce; the final pass is
       // authoritative. But when the bubble ALREADY shows exactly the
       // final answer (now the common case, since the plain lane splits
       // <think> out as it streams instead of dumping it into this bubble)
       // rebuilding the whole message buys nothing and costs a visible
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
       run.answered = true;
       run.ended = true;
       updateContextMeter(estimateTokens(event.answer || ""));
       var bits = [];
       if (event.tools_used && event.tools_used.length) bits.push(event.tools_used.join(", "));
       if (chatPrefs.show_metrics) {
         bits.push((event.steps || 1) + " step" + ((event.steps || 1) === 1 ? "" : "s"));
         // Cost of the turn, in the shape a local runner prints it:
         // "2048 tok - 100.4s - 20 tok/s". tok is what the model generated
         // across every step, and the rate is against wall-clock, so on an
         // agent turn it includes the time tools took: the number you
         // actually waited. The decode speed of the model itself and the
         // time to the first token come from the usage events after it.
         var secs = (event.elapsed_ms || 0) / 1000;
         var tok = event.completion_tokens || 0;
         if (tok) bits.push(tok + " tok");
         bits.push(secs.toFixed(1) + "s");
         if (tok && secs > 0) bits.push(chatRate(tok / secs) + " tok/s");
         var speed = chatUsageText(run.firstUsage, run.usage);
         if (speed) bits.push(speed);
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
         // node, which would otherwise present a .docx as an edit to inspect
         // with git before pushing. A project turn wrote into the project's
         // root, so the download names the project to resolve it there.
         var docs = document.createElement("div");
         docs.className = "docrow";
         event.documents.forEach(function(rel) {
           var name = rel.split("/").pop();
           var a = document.createElement("a");
           a.className = "docdl";
           a.href = "/api/files/download?path=" + encodeURIComponent(rel)
             + (run.projectId ? "&project_id=" + encodeURIComponent(run.projectId) : "");
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
         // The undo for exactly this turn: the checkpoint the final event
         // names holds every file as it was before the turn wrote it.
         if (event.checkpoint) {
           var undoRow = document.createElement("div");
           undoRow.className = "row mt-1";
           var undo = makeBtn("Undo these changes",
             "Put every file this turn changed back the way it was before the turn", null);
           undo.onclick = function() { chatUndoTurn(event.checkpoint, event.changed_files || changed, undo, undoRow); };
           undoRow.appendChild(undo);
           wrap.appendChild(undoRow);
         }
         trace.turn.appendChild(wrap);
       }
       setRegenerateTarget(addFeedbackBar(run.message, event.answer || ""));
       // Both routes are ADMIN. Every non-admin answer used to fire two
       // requests that could only ever come back 403.
       if (roleIsAdmin()) {
         loadRunCurve(conversationId);
         loadPerf();
       }
     } else if (event.type === "cancelled") {
       chatShowPartial(trace, event.partial);
       settleAnswer(trace, false);
       // settleAnswer, not a bare notice: a server-side cancel used to leave
       // the partial answer classed `pending` for good, greyed out with no
       // `.failed` marking. The partial is the text the server stored, shown
       // when nothing had streamed into the bubble yet (a resumed turn).
       traceNotice(trace, chatStoppedLabel(event), false);
       stopActivity(trace);
       run.answered = true;
       run.ended = true;
     } else if (event.type === "error") {
       traceNotice(trace, "error: " + event.error, false);
       // Settle whatever streamed before the error. Marking the turn
       // answered skips the cleanup after the stream, so without this the
       // partial text sat at the pending opacity for good: a half-written
       // answer permanently greyed out, with none of the .failed styling the
       // AbortError path applies for the same situation.
       settleAnswer(trace, true);
       stopActivity(trace);
       run.answered = true;
       run.ended = true;
     }
   }

   // ----------------------------------------------------- event helpers ---

   // "Undo these changes" under a turn's changed files: after a confirm, roll
   // the turn's checkpoint back (POST /api/checkpoints/{id}/rollback, which
   // lets the owner undo their own turn and an admin any) and say what came
   // back. Files the turn created are deleted; files it changed are restored.
   async function chatUndoTurn(checkpointId, files, btn, row) {
     var list = (files || []).filter(function(f) { return !!f; });
     var what = list.length === 1 ? list[0] : list.length + " files";
     var ask = "Undo this turn's changes to " + what + "? Files it created are deleted and "
       + "files it changed go back to how they were before the turn.";
     if (!(await uiConfirm(ask))) return;
     btn.disabled = true;
     btn.textContent = "Undoing…";
     var old = row.querySelector(".hint");
     if (old) row.removeChild(old);
     var note = document.createElement("span");
     try {
       var out = await postJSON("/api/checkpoints/" + encodeURIComponent(checkpointId) + "/rollback", {});
       var data = out.data || {};
       var restored = data.restored || [];
       var problems = data.problems || [];
       note.className = "hint " + (problems.length ? "warn" : "ok");
       note.textContent = "Undone: put back " + restored.length + " file"
         + (restored.length === 1 ? "" : "s")
         + (restored.length ? " (" + restored.join(", ") + ")" : "")
         + (problems.length ? ". Not restored: " + problems.join("; ") : ".");
       btn.textContent = "Undone";
       uiToast(problems.length ? "Undo finished with problems." : "This turn's changes were undone.",
               problems.length ? "warn" : "ok");
     } catch (err) {
       note.className = "hint error";
       note.textContent = "Undo failed: " + ((err && err.message) || err);
       btn.disabled = false;
       btn.textContent = "Undo these changes";
       uiToast("Undo failed: " + ((err && err.message) || err), "error");
     }
     row.appendChild(note);
   }

   function chatRate(tps) {
     tps = Number(tps) || 0;
     return tps >= 10 ? String(Math.round(tps)) : tps.toFixed(1);
   }

   function chatSecs(ms) {
     ms = Number(ms) || 0;
     return ms >= 1000 ? (ms / 1000).toFixed(1) + "s" : Math.round(ms) + "ms";
   }

   // "TTFT 180ms · decode 31 tok/s": the first step's time to first token
   // (what the user waited before anything moved) and the latest step's
   // decode speed (what the model itself manages on this machine).
   function chatUsageText(first, last) {
     var bits = [];
     if (first && first.ttft_ms) bits.push("TTFT " + chatSecs(first.ttft_ms));
     if (last && last.decode_tps) bits.push("decode " + chatRate(last.decode_tps) + " tok/s");
     return bits.join(" · ");
   }

   function chatNoteUsage(run, event) {
     var u = (event.usage && typeof event.usage === "object") ? event.usage : event;
     if (!run.firstUsage) run.firstUsage = u;
     run.usage = u;
     if (!chatPrefs.show_metrics) return;
     var meta = document.getElementById("composerMeta");
     if (!meta) return;
     var text = chatUsageText(run.firstUsage, u);
     if (meta.textContent !== text) meta.textContent = text;
     meta.setAttribute("data-tip", "Latest turn: time to first token and decode speed"
       + (u.cached_tokens ? ", " + u.cached_tokens + " prompt tokens served from cache" : "") + ".");
   }

   function chatQueuedLabel(event) {
     var label = "waiting for a slot: #" + (event.position || 1);
     if (event.limit) label += " (" + (event.running || 0) + " of " + event.limit + " busy)";
     return label;
   }

   function chatInjectedLabel(event) {
     var files = (event.files || []).map(function(f) {
       return typeof f === "string" ? f : ((f && (f.path || f.name)) || "");
     }).filter(function(f) { return !!f; });
     return "project " + (event.project || event.project_id || "") + ": read "
       + (files.length ? files.join(", ") : "its context files")
       + (event.chars ? " (" + compactTokens(event.chars) + " chars)" : "");
   }

   function chatStoppedLabel(event) {
     var why = String(event.reason || "");
     if (!why || why === "cancelled" || why.indexOf("by the user") >= 0) return "stopped";
     return "stopped: " + why.slice(0, 120);
   }

   function chatShowPartial(trace, partial) {
     if (!partial) return;
     var a = traceAnswer(trace);
     if (a.text.textContent) return;
     renderMarkdown(partial, a.text);
     a.raw = partial;
     a.lastRendered = partial;
     a.rendered = true;
   }

   // Tool cards are keyed by call id. The trace used to keep one `tool` slot,
   // so two calls in one step (or a result arriving after the next call
   // started) wrote the second result into the first card.
   function chatToolKey(event) {
     if (event.id) return String(event.id);
     if (event.call_id) return String(event.call_id);
     if (event.index !== undefined && event.index !== null) return "i" + event.index;
     return "";
   }

   function chatToolStart(run, event) {
     var key = chatToolKey(event);
     if (key && run.tools[key]) return run.tools[key];
     var card = traceTool(run.trace, event.name || "tool", null);
     card.argsEl.textContent = "";
     card.argsRaw = "";
     card.called = false;
     card.key = key;
     card.node.classList.add("running");
     card.pill.textContent = "calling";
     if (key) run.tools[key] = card;
     run.lastTool = card;
     return card;
   }

   function chatToolDelta(run, event) {
     var key = chatToolKey(event);
     var card = key ? run.tools[key] : run.lastTool;
     // A delta without its start: a resume that began mid-call.
     if (!card) card = chatToolStart(run, event);
     if (card.called) return;
     card.argsRaw = (card.argsRaw || "") + String(event.args_delta || "");
     card.argsEl.textContent = card.argsRaw;
     scrollDown();
   }

   function chatToolCall(run, event) {
     var key = chatToolKey(event);
     var card = key ? run.tools[key] : null;
     if (!card) {
       card = traceTool(run.trace, event.name || "tool", event.args);
       card.key = key;
       if (key) run.tools[key] = card;
     } else {
       // The authoritative arguments replace whatever streamed in, and an
       // `edited` call (the user changed them in the approval card) says so.
       card.argsEl.textContent = JSON.stringify(event.args || {});
       if (event.name) card.label.textContent = event.name;
     }
     card.called = true;
     card.node.classList.add("running");
     card.node.classList.remove("awaiting");
     card.pill.textContent = event.edited ? "running (edited)" : "running";
     if (event.edited) card.node.classList.add("edited");
     run.lastTool = card;
     return card;
   }

   function chatToolResult(run, event) {
     var key = chatToolKey(event);
     var card = (key && run.tools[key]) || run.lastTool;
     if (!card) return;
     card.called = true;
     card.done = true;
     card.node.classList.remove("running");
     card.node.classList.remove("awaiting");
     card.body.textContent = event.result || (event.error && event.error !== true ? String(event.error) : "");
     if (event.error) {
       card.node.classList.add("failed");
       card.pill.textContent = "failed";
     } else {
       card.node.classList.add("ok");
       card.pill.textContent = "ok";
     }
     if (run.lastTool === card) run.lastTool = null;
   }

   // A tool_call_start with no tool_call by the next step (or the end of the
   // turn): the model began printing a call and then did something else.
   function chatDropStartedTools(run) {
     Object.keys(run.tools).forEach(function(key) {
       var card = run.tools[key];
       if (card.called) return;
       if (card.node.parentNode) card.node.parentNode.removeChild(card.node);
       if (run.lastTool === card) run.lastTool = null;
       delete run.tools[key];
     });
   }

   // ---------------------------------------------- approvals and consent ---

   // One inline decision card for both prompts a turn can stop on: a tool
   // call the policy wants a person to approve, and sending the turn to a
   // cloud model. Keyed "<kind>:<id>" so a replayed event (resume, or the
   // boot list of pending approvals) replaces its card instead of adding one.
   var chatDecisions = {};
   var CHAT_SCOPE_LABELS = {
     approval: { once: "Once", conversation: "This conversation", tool: "Always for this tool" },
     consent: { once: "Allow once", conversation: "Allow for this conversation", always: "Always allow" }
   };

   function chatRiskClass(risk) {
     risk = String(risk || "").toLowerCase();
     if (risk === "read" || risk === "low") return "ok";
     if (risk === "exec" || risk === "high" || risk === "critical") return "error";
     return risk ? "warn" : "muted";
   }

   function chatPendingHost() {
     var host = chat.querySelector(".decision-pending");
     if (host) return host;
     clearChatEmpty();
     host = document.createElement("div");
     host.className = "decision-pending";
     host.setAttribute("aria-label", "Waiting for your answer");
     chat.appendChild(host);
     return host;
   }

   function chatDecisionCard(spec) {
     var key = spec.kind + ":" + spec.id;
     var old = chatDecisions[key];
     if (old) chatDecisionDiscard(old);
     var node = document.createElement("div");
     node.className = (spec.host ? "gnode " : "") + "decision " + spec.kind;
     var card = document.createElement("div");
     card.className = "card stack decision-card";
     card.setAttribute("role", "group");
     card.setAttribute("aria-label", spec.title);
     node.appendChild(card);

     var head = document.createElement("div");
     head.className = "card-header";
     var title = document.createElement("span");
     title.className = "gtool";
     title.textContent = spec.title;
     head.appendChild(title);
     if (spec.risk) {
       var risk = document.createElement("span");
       risk.className = "pill " + chatRiskClass(spec.risk);
       risk.textContent = spec.risk;
       head.appendChild(risk);
     }
     var timer = document.createElement("span");
     timer.className = "decision-timer muted mono";
     head.appendChild(timer);
     card.appendChild(head);

     (spec.lines || []).forEach(function(text) {
       if (!text) return;
       var el = document.createElement("div");
       el.className = "hint";
       el.textContent = text;
       card.appendChild(el);
     });

     var state = { key: key, kind: spec.kind, id: spec.id, turnId: spec.turnId || "",
                   callId: spec.callId || "", run: spec.run || null, node: node, card: card,
                   head: head, timer: timer, buttons: [], editor: null, actions: null,
                   tick: null, deadline: 0, closed: false, sending: false };

     if (spec.args !== undefined) {
       var shown = JSON.stringify(spec.args == null ? {} : spec.args, null, 2);
       var pre = document.createElement("pre");
       pre.className = "decision-args mono";
       pre.textContent = shown;
       card.appendChild(pre);
       var editor = document.createElement("textarea");
       editor.className = "input textarea decision-edit hidden";
       editor.setAttribute("aria-label", "Tool arguments as JSON");
       editor.spellcheck = false;
       editor.value = shown;
       var edit = makeBtn("Edit arguments", "Change the arguments before allowing the call", null);
       edit.onclick = function() {
         var opening = editor.classList.contains("hidden");
         editor.classList.toggle("hidden", !opening);
         pre.classList.toggle("hidden", opening);
         edit.textContent = opening ? "Discard edits" : "Edit arguments";
         if (opening) editor.focus(); else editor.value = shown;
       };
       card.appendChild(edit);
       card.appendChild(editor);
       state.editor = editor;
       state.buttons.push(edit);
     }

     var row = document.createElement("div");
     row.className = "row wrap decision-actions";
     var labels = CHAT_SCOPE_LABELS[spec.kind] || {};
     var scopes = (spec.scopes && spec.scopes.length) ? spec.scopes : ["once"];
     if (spec.kind === "approval") {
       var seg = document.createElement("div");
       // wrap: on a phone "Always for this tool" ran past the card's edge.
       seg.className = "seg wrap";
       seg.setAttribute("role", "radiogroup");
       seg.setAttribute("aria-label", "Remember the answer for");
       scopes.forEach(function(s) {
         var b = document.createElement("button");
         b.type = "button";
         b.setAttribute("data-value", s);
         b.textContent = labels[s] || s;
         seg.appendChild(b);
         state.buttons.push(b);
       });
       uiSeg(seg, scopes[0]);
       wireSeg(seg);
       row.appendChild(seg);
       var gap = document.createElement("div");
       gap.className = "spacer";
       row.appendChild(gap);
       var deny = makeBtn("Deny", "Refuse this tool call; the agent is told and carries on", null, "btn-danger");
       var allow = document.createElement("button");
       allow.type = "button";
       allow.className = "btn btn-sm btn-primary";
       allow.textContent = "Allow";
       deny.onclick = function() { chatDecide(state, "deny", uiSegValue(seg) || "once"); };
       allow.onclick = function() { chatDecide(state, "allow", uiSegValue(seg) || "once"); };
       row.appendChild(deny);
       row.appendChild(allow);
       state.buttons.push(deny, allow);
     } else {
       scopes.forEach(function(s, i) {
         var b = document.createElement("button");
         b.type = "button";
         b.className = "btn btn-sm" + (i === 0 ? " btn-primary" : "");
         b.textContent = labels[s] || ("Allow (" + s + ")");
         b.onclick = function() { chatDecide(state, "allow", s); };
         row.appendChild(b);
         state.buttons.push(b);
       });
       var no = makeBtn("Deny", "Keep this turn on the local model", null, "btn-danger");
       no.onclick = function() { chatDecide(state, "deny", "once"); };
       row.appendChild(no);
       state.buttons.push(no);
     }
     card.appendChild(row);
     state.actions = row;

     if (spec.note) {
       var note = document.createElement("div");
       note.className = "row wrap hint";
       var noteText = document.createElement("span");
       noteText.textContent = spec.note.text;
       note.appendChild(noteText);
       if (spec.note.action) note.appendChild(makeBtn(spec.note.label, spec.note.label, spec.note.action));
       card.appendChild(note);
     }

     var host = spec.host || chatPendingHost();
     host.appendChild(node);
     chatDecisions[key] = state;
     // The countdown starts once the card is on the page: the tick stops
     // itself for a card that is not, so starting it first stopped it at once.
     var secs = Number(spec.expiresIn) || 0;
     if (secs > 0) {
       state.deadline = Date.now() + secs * 1000;
       state.tick = setInterval(function() { chatDecisionTick(state); }, 1000);
       chatDecisionTick(state);
     }
     scrollDown();
     return state;
   }

   function chatDecisionTick(state) {
     if (state.closed) return;
     if (!document.body.contains(state.node)) {
       // The chat was cleared under the card (new chat, another conversation).
       if (state.tick) { clearInterval(state.tick); state.tick = null; }
       return;
     }
     var left = Math.max(0, Math.round((state.deadline - Date.now()) / 1000));
     state.timer.textContent = "expires in " + Math.floor(left / 60) + ":" + (left % 60 < 10 ? "0" : "") + (left % 60);
     if (left <= 0) chatDecisionClose(state, { decision: "timeout" });
   }

   function chatDecisionDiscard(state) {
     if (state.tick) { clearInterval(state.tick); state.tick = null; }
     state.closed = true;
     var parent = state.node.parentNode;
     if (parent) parent.removeChild(state.node);
     // The card moved into a live trace: an emptied holder for prompts of
     // other turns would stay behind as a blank gap in the chat.
     if (parent && parent.classList.contains("decision-pending") && !parent.firstChild && parent.parentNode) {
       parent.parentNode.removeChild(parent);
     }
     if (chatDecisions[state.key] === state) delete chatDecisions[state.key];
   }

   function chatDecisionResult(kind, result) {
     var decision = String((result && result.decision) || "");
     var scope = String((result && result.scope) || "once");
     var label = (CHAT_SCOPE_LABELS[kind] || {})[scope] || scope;
     if (result && result.by === "cancel") return { text: "stopped", cls: "muted" };
     if (decision === "allow") return { text: "allowed" + (scope !== "once" ? " (" + label.toLowerCase() + ")" : ""), cls: "ok" };
     if (decision === "deny") return { text: "denied", cls: "error" };
     if (decision === "timeout") return { text: "timed out", cls: "warn" };
     if (decision === "gone") return { text: "no longer waiting", cls: "muted" };
     return { text: decision || "answered", cls: "muted" };
   }

   function chatDecisionClose(state, result) {
     if (state.tick) { clearInterval(state.tick); state.tick = null; }
     state.closed = true;
     state.node.classList.add("resolved");
     if (state.actions && state.actions.parentNode) state.actions.parentNode.removeChild(state.actions);
     state.buttons.forEach(function(b) {
       if (b.parentNode && b.parentNode !== state.actions) b.parentNode.removeChild(b);
     });
     if (state.editor && state.editor.parentNode) state.editor.parentNode.removeChild(state.editor);
     var shown = state.card.querySelector(".decision-args");
     if (shown) shown.classList.remove("hidden");
     var r = chatDecisionResult(state.kind, result);
     state.timer.className = "pill " + r.cls + " decision-timer";
     state.timer.textContent = r.text;
   }

   async function chatDecide(state, decision, scope) {
     if (state.closed || state.sending) return;
     var body = { decision: decision, scope: scope };
     if (state.kind === "approval" && decision === "allow" && state.editor
         && !state.editor.classList.contains("hidden")) {
       var parsed;
       try { parsed = JSON.parse(state.editor.value); } catch (err) {
         uiToast("The arguments are not valid JSON: " + err.message, "error");
         return;
       }
       if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
         uiToast("The arguments must be a JSON object.", "error");
         return;
       }
       body.args = parsed;
     }
     var turnId = state.turnId || (state.run && state.run.turnId) || "";
     if (!turnId) { uiToast("This request is not attached to a running turn any more.", "warn"); return; }
     state.sending = true;
     state.buttons.forEach(function(b) { b.disabled = true; });
     var path = state.kind === "approval" ? "/approvals/" : "/consent/";
     try {
       await postJSON("/api/chat/turns/" + encodeURIComponent(turnId) + path + encodeURIComponent(state.id), body);
       chatDecisionClose(state, body);
     } catch (err) {
       state.sending = false;
       if (err.status === 404) {
         chatDecisionClose(state, { decision: "gone" });
         uiToast("That request was already answered or has expired.", "warn");
       } else {
         state.buttons.forEach(function(b) { b.disabled = false; });
         uiToast("Could not send the answer: " + err.message, "error");
       }
     }
   }

   // approval_resolved / consent_resolved: the server's word is final (it
   // knows about a timeout, a cancel, or an answer given in another tab).
   function chatDecisionResolved(kind, event) {
     var state = chatDecisions[kind + ":" + event.id];
     if (state) chatDecisionClose(state, event);
   }

   // A card rebuilt from a replayed event (a re-attach re-sends the turn's
   // buffered events) carries the expiry the prompt had when it was first
   // sent, which can be minutes stale. The pending list has the time left.
   async function chatSyncDeadline(state) {
     if (!state || !state.tick) return;
     var approval = state.kind === "approval";
     try {
       var out = await fetchJSON(approval ? "/api/approvals/pending" : "/api/cloud/consents");
       var rows = (out.data && (approval ? out.data.approvals : out.data.pending)) || [];
       rows.forEach(function(row) {
         if (state.closed || String(row.id) !== String(state.id)) return;
         var left = Number(row.expires_in) || 0;
         if (left > 0) { state.deadline = Date.now() + left * 1000; chatDecisionTick(state); }
       });
     } catch (err) { /* keep the event's own countdown */ }
   }

   function chatApprovalCard(run, event) {
     var tool = event.call_id ? run.tools[String(event.call_id)] : null;
     if (tool) {
       tool.node.classList.add("awaiting");
       tool.pill.textContent = "awaiting approval";
     }
     var state = chatDecisionCard({
       kind: "approval", id: event.id, run: run, turnId: run.turnId, callId: event.call_id,
       host: run.trace.trace, title: "Allow " + (event.name || "this tool") + "?",
       risk: event.risk, lines: [event.reason], args: event.args === undefined ? {} : event.args,
       scopes: event.scopes, expiresIn: event.expires_in
     });
     if (run.reattached) chatSyncDeadline(state);
   }

   function chatApprovalResolved(run, event) {
     chatDecisionResolved("approval", event);
     var tool = event.call_id ? run.tools[String(event.call_id)] : null;
     if (!tool) return;
     tool.node.classList.remove("awaiting");
     if (event.decision === "allow") {
       tool.pill.textContent = "running";
     } else {
       tool.node.classList.remove("running");
       tool.node.classList.add("failed");
       tool.pill.textContent = event.decision === "timeout" ? "timed out" : "denied";
     }
   }

   function chatConsentLines(event) {
     var where = [event.provider, event.model].filter(function(x) { return !!x; }).join(" / ");
     var size = event.est_prompt_tokens ? "about " + compactTokens(event.est_prompt_tokens) + " prompt tokens" : "";
     return [event.reason, [where, size].filter(function(x) { return !!x; }).join(", ")];
   }

   function chatConsentCard(run, event) {
     var state = chatDecisionCard({
       kind: "consent", id: event.id, run: run, turnId: run.turnId,
       host: run.trace.trace, title: "Send this turn to the cloud model?",
       lines: chatConsentLines(event), scopes: event.scopes, expiresIn: event.expires_in
     });
     if (run.reattached) chatSyncDeadline(state);
   }

   // Approvals (and cloud consents) still waiting for this user, for turns
   // this tab is not streaming: started before a reload, in another tab, or
   // by a scheduled task that asks. The live stream replays its own.
   async function chatLoadPendingDecisions() {
     var approvals = [];
     var consents = [];
     try {
       var out = await fetchJSON("/api/approvals/pending");
       approvals = (out.data && out.data.approvals) || [];
     } catch (err) { approvals = []; }
     try {
       var out2 = await fetchJSON("/api/cloud/consents");
       consents = (out2.data && out2.data.pending) || [];
     } catch (err) { consents = []; }
     function chatPendingSkip(kind, row) {
       if (chatRun && row.turn_id && row.turn_id === chatRun.turnId) return true;
       var have = chatDecisions[kind + ":" + row.id];
       return !!(have && !have.closed && document.body.contains(have.node));
     }
     function chatPendingNote(row) {
       if (!row.conversation_id || row.conversation_id === conversationId) return null;
       return { text: "From another conversation.", label: "Open it",
                action: function() { openConversation(row.conversation_id); } };
     }
     approvals.forEach(function(row) {
       if (chatPendingSkip("approval", row)) return;
       chatDecisionCard({
         kind: "approval", id: row.id, turnId: row.turn_id, callId: row.call_id,
         title: "Allow " + (row.name || "this tool") + "?", risk: row.risk,
         lines: [row.reason], args: row.args === undefined ? {} : row.args,
         scopes: row.scopes, expiresIn: row.expires_in, note: chatPendingNote(row)
       });
     });
     consents.forEach(function(row) {
       if (chatPendingSkip("consent", row)) return;
       chatDecisionCard({
         kind: "consent", id: row.id, turnId: row.turn_id,
         title: "Send this turn to the cloud model?", lines: chatConsentLines(row),
         scopes: row.scopes, expiresIn: row.expires_in, note: chatPendingNote(row)
       });
     });
   }

   // ------------------------------------------------- history replay ---

   function chatToolNames(tools) {
     return (Array.isArray(tools) ? tools : []).map(function(t) {
       if (typeof t === "string") return t;
       return (t && (t.name || t.tool)) || "";
     }).filter(function(t) { return !!t; });
   }

   // A stored answer's meta: its reasoning (collapsed per the
   // reasoning_display preference), the tools it used, and whether it was
   // stopped. These used to exist only in the live stream, so a reopened
   // conversation showed bare answers.
   function chatReplayMeta(div, meta) {
     if (!meta || typeof meta !== "object") return;
     var actions = div.querySelector(".msg-actions");
     if (meta.reasoning && chatPrefs.reasoning_display !== "hidden") {
       var det = document.createElement("details");
       det.className = "msg-reasoning";
       if (chatPrefs.reasoning_display === "expanded") det.open = true;
       var sum = document.createElement("summary");
       sum.textContent = "Reasoning";
       var pre = document.createElement("pre");
       pre.className = "msg-reasoning-body";
       pre.textContent = String(meta.reasoning);
       det.appendChild(sum);
       det.appendChild(pre);
       div.insertBefore(det, div.firstChild);
     }
     var tools = chatToolNames(meta.tools);
     if (tools.length || meta.cancelled) {
       var line = document.createElement("div");
       line.className = "msg-meta";
       if (tools.length) {
         var names = document.createElement("span");
         names.className = "mono";
         names.textContent = "tools: " + tools.join(", ");
         line.appendChild(names);
       }
       if (meta.cancelled) {
         var pill = document.createElement("span");
         pill.className = "pill warn";
         pill.textContent = "stopped";
         line.appendChild(pill);
       }
       div.insertBefore(line, actions);
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
     // The bell rides on this poll (self-throttled to every 15 seconds).
     chatPollNotifications(false);
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

   // ------------------------------------------------ composer: pickers ---

   // Project and agent-profile pickers. Each stays hidden until the server
   // says there is something to pick, so an installation without projects or
   // agent profiles keeps exactly the toolbar it had.
   function chatSelectValue(id) {
     var sel = document.getElementById(id);
     if (!sel || sel.classList.contains("hidden")) return "";
     return sel.value || "";
   }

   function chatFillSelect(sel, items, emptyLabel, labelOf) {
     var keep = sel.value;
     while (sel.firstChild) sel.removeChild(sel.firstChild);
     var none = document.createElement("option");
     none.value = "";
     none.textContent = emptyLabel;
     sel.appendChild(none);
     items.forEach(function(item) {
       if (!item || !item.id) return;
       var opt = document.createElement("option");
       opt.value = item.id;
       opt.textContent = labelOf(item);
       sel.appendChild(opt);
     });
     sel.value = keep;
     if (sel.value !== keep) sel.value = "";
   }

   async function chatLoadComposerOptions() {
     var psel = document.getElementById("projectSel");
     try {
       var out = await fetchJSON("/api/projects");
       var d = out.data || {};
       var projects = d.projects || [];
       chatFillSelect(psel, projects, "No project", function(p) { return p.name || p.id; });
       psel.classList.toggle("hidden", d.enabled === false || !projects.length);
     } catch (err) {
       psel.value = "";
       psel.classList.add("hidden");
     }
     var asel = document.getElementById("agentSel");
     try {
       var out2 = await fetchJSON("/api/agents");
       // Admins are sent every profile (to manage them); only enabled ones
       // can run a turn, which is all a non-admin is sent anyway.
       var agents = ((out2.data && out2.data.agents) || []).filter(function(a) { return a.enabled; });
       chatFillSelect(asel, agents, "Default agent", function(a) { return a.name || a.id; });
       asel.classList.toggle("hidden", !agents.length);
     } catch (err) {
       asel.value = "";
       asel.classList.add("hidden");
     }
     // Attachments: the paperclip and the mic only where the route exists.
     var attOk = false;
     try {
       await fetchJSON("/api/attachments");
       attOk = true;
     } catch (err) { attOk = false; }
     document.getElementById("attBtn").classList.toggle("hidden", !attOk);
     document.getElementById("micBtn").classList.toggle("hidden", !(attOk && chatMicSupported() && !chatMicOff));
   }

   // --------------------------------------------- composer: attachments ---

   // Uploaded before sending (POST /api/attachments, the raw file as the
   // body), sent as ids with the message. A chip per file; removing one also
   // deletes the upload so nothing is left behind that no message uses.
   var chatAtts = [];
   var CHAT_ATT_MAX = 16;

   function chatAttPick() {
     var picker = document.getElementById("attInput");
     if (picker) picker.click();
   }

   function chatAttBusy() {
     return chatAtts.some(function(a) { return a.status === "uploading"; });
   }

   function chatAttReady() {
     return chatAtts.filter(function(a) { return a.status === "ready" && a.id; });
   }

   function chatAttClear(deleteUploads) {
     if (deleteUploads) chatAtts.forEach(function(a) { if (a.id) chatAttForget(a.id); });
     chatAtts = [];
     chatAttRender();
   }

   function chatAttForget(id) {
     fetch("/api/attachments/" + encodeURIComponent(id), { method: "DELETE" }).catch(function() {});
   }

   function chatAttRemove(item) {
     item.removed = true;
     chatAtts = chatAtts.filter(function(a) { return a !== item; });
     if (item.id) chatAttForget(item.id);
     chatAttRender();
   }

   function chatAttRender() {
     var host = document.getElementById("attChips");
     if (!host) return;
     while (host.firstChild) host.removeChild(host.firstChild);
     chatAtts.forEach(function(item) {
       var chip = document.createElement("span");
       chip.className = "att-chip" + (item.status === "uploading" ? " uploading" : "");
       if (item.kind === "image" && item.id) {
         var img = document.createElement("img");
         img.className = "att-thumb";
         img.alt = "";
         img.src = "/api/attachments/" + encodeURIComponent(item.id);
         chip.appendChild(img);
       }
       var name = document.createElement("span");
       name.className = "att-name";
       name.textContent = item.name + (item.status === "uploading" ? " (uploading…)" : "");
       name.title = item.name + (item.bytes ? " · " + chatBytes(item.bytes) : "");
       chip.appendChild(name);
       var x = document.createElement("button");
       x.type = "button";
       x.className = "btn-icon btn-ghost btn-sm att-remove";
       x.setAttribute("aria-label", "Remove " + item.name);
       x.textContent = "×";
       x.onclick = function() { chatAttRemove(item); };
       chip.appendChild(x);
       host.appendChild(chip);
     });
     host.classList.toggle("hidden", !chatAtts.length);
   }

   function chatBytes(n) {
     n = Number(n) || 0;
     if (n >= 1048576) return (n / 1048576).toFixed(1) + " MB";
     if (n >= 1024) return Math.round(n / 1024) + " KB";
     return n + " B";
   }

   // POST one blob as an attachment. Resolves to the stored row; throws an
   // Error carrying the server's own reason (too large, type refused, quota).
   async function chatUploadBlob(blob, filename) {
     var res = await fetch("/api/attachments", {
       method: "POST",
       headers: { "Content-Type": blob.type || "application/octet-stream",
                  "X-Filename": encodeURIComponent(filename || "file") },
       body: blob
     });
     if (res.status === 401 && typeof onUnauthorized === "function") onUnauthorized();
     var data = {};
     try { data = await res.json(); } catch (e) { data = {}; }
     if (!res.ok || !data.id) {
       var err = new Error(httpErrorText(data, res.status));
       err.status = res.status;
       throw err;
     }
     return data;
   }

   async function chatAttUpload(files) {
     if (!files || !files.length) return;
     var list = Array.prototype.slice.call(files);
     for (var i = 0; i < list.length; i++) {
       if (chatAtts.length >= CHAT_ATT_MAX) {
         uiToast("At most " + CHAT_ATT_MAX + " attachments per message.", "warn");
         break;
       }
       var file = list[i];
       var item = { name: file.name || "pasted file", status: "uploading", id: null, kind: "", bytes: file.size };
       chatAtts.push(item);
       chatAttRender();
       try {
         var row = await chatUploadBlob(file, item.name);
         item.id = row.id;
         item.kind = row.kind || "";
         item.mime = row.mime || "";
         item.bytes = row.bytes || item.bytes;
         item.status = "ready";
         // Removed while it was still uploading: drop the upload as well.
         if (item.removed) chatAttForget(row.id);
       } catch (err) {
         chatAtts = chatAtts.filter(function(a) { return a !== item; });
         uiToast(item.name + ": " + err.message, "error");
       }
       chatAttRender();
     }
   }

   // Thumbnails (images) and name chips (other files) under a user message.
   // Accepts upload rows or bare ids (a stored message's meta may only keep
   // the ids): an id whose image fails to load falls back to a chip.
   function chatRenderMsgAtts(div, atts) {
     if (!div || !atts || !atts.length) return;
     var row = document.createElement("div");
     row.className = "msg-atts";
     atts.forEach(function(a) {
       var id = typeof a === "string" ? a : (a && a.id);
       if (!id) return;
       var name = (a && a.name) || (a && a.filename) || "attachment";
       var kind = typeof a === "string" ? "" : (a.kind || "");
       function chatAsChip() {
         var chip = document.createElement("span");
         chip.className = "att-chip";
         chip.textContent = name;
         return chip;
       }
       if (kind === "image" || !kind) {
         var img = document.createElement("img");
         img.className = "msg-thumb";
         img.alt = name;
         img.loading = "lazy";
         img.src = "/api/attachments/" + encodeURIComponent(id);
         img.onclick = function() { window.open(img.src, "_blank", "noopener"); };
         img.onerror = function() { if (img.parentNode) img.parentNode.replaceChild(chatAsChip(), img); };
         row.appendChild(img);
       } else {
         row.appendChild(chatAsChip());
       }
     });
     div.insertBefore(row, div.querySelector(".msg-actions"));
   }

   // ---------------------------------------------------- composer: mic ---

   // Dictation: record with MediaRecorder, upload the clip as an audio
   // attachment, transcribe it on the server, put the text in the box.
   var chatRec = null;
   var chatMicOff = false;

   function chatMicSupported() {
     return !!(window.MediaRecorder && navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
   }

   function chatMicUi(state) {
     var mic = document.getElementById("micBtn");
     if (!mic) return;
     mic.classList.toggle("recording", state === "recording");
     mic.setAttribute("aria-pressed", state === "recording" ? "true" : "false");
     mic.disabled = state === "working";
     mic.setAttribute("data-tip", state === "recording" ? "Recording: click again to stop and transcribe."
       : (state === "working" ? "Transcribing…" : "Dictate: record a voice note and insert its transcript."));
   }

   async function chatMicToggle() {
     if (chatRec) { chatRec.recorder.stop(); return; }
     if (!chatMicSupported()) { uiToast("This browser cannot record audio.", "warn"); return; }
     var stream;
     try {
       stream = await navigator.mediaDevices.getUserMedia({ audio: true });
     } catch (err) {
       uiToast("Microphone unavailable: " + ((err && err.message) || err), "error");
       return;
     }
     var recorder;
     try { recorder = new MediaRecorder(stream); } catch (err) {
       stream.getTracks().forEach(function(t) { t.stop(); });
       uiToast("Recording failed: " + err.message, "error");
       return;
     }
     var chunks = [];
     recorder.ondataavailable = function(e) { if (e.data && e.data.size) chunks.push(e.data); };
     recorder.onstop = function() {
       stream.getTracks().forEach(function(t) { t.stop(); });
       chatRec = null;
       chatMicUi("idle");
       chatMicTranscribe(new Blob(chunks, { type: recorder.mimeType || "audio/webm" }));
     };
     chatRec = { recorder: recorder };
     recorder.start();
     chatMicUi("recording");
   }

   async function chatMicTranscribe(blob) {
     if (!blob || !blob.size) return;
     var type = blob.type || "";
     var ext = type.indexOf("ogg") >= 0 ? "ogg" : (type.indexOf("mp4") >= 0 || type.indexOf("aac") >= 0 ? "m4a" : "webm");
     chatMicUi("working");
     var row = null;
     try {
       row = await chatUploadBlob(blob, "voice-note." + ext);
       var res = await fetch("/api/attachments/" + encodeURIComponent(row.id) + "/transcribe", { method: "POST" });
       var data = {};
       try { data = await res.json(); } catch (e) { data = {}; }
       if (res.status === 503) {
         // Speech to text is not installed on the server: say how to get it.
         // The server's error usually quotes the hint already.
         var why = data.error || "Speech to text is not available.";
         var hint = data.hint && why.indexOf(data.hint) < 0 ? " " + data.hint : "";
         uiToast(why + hint, "warn");
       } else if (res.status === 404 && !data.error) {
         // The server has attachments but no transcription route.
         chatMicOff = true;
         document.getElementById("micBtn").classList.add("hidden");
         uiToast("Voice input is not available on this server.", "warn");
       } else if (!res.ok) {
         uiToast("Transcription failed: " + httpErrorText(data, res.status), "error");
       } else {
         chatInsertText(data.text || "");
       }
     } catch (err) {
       uiToast("Voice note failed: " + err.message, "error");
     } finally {
       // The clip was only the way to the text.
       if (row && row.id) chatAttForget(row.id);
       chatMicUi("idle");
     }
   }

   function chatInsertText(text) {
     text = String(text || "").trim();
     if (!text) { uiToast("No speech was recognised.", "warn"); return; }
     var value = input.value;
     var start = typeof input.selectionStart === "number" ? input.selectionStart : value.length;
     var end = typeof input.selectionEnd === "number" ? input.selectionEnd : value.length;
     var before = value.slice(0, start);
     var sep = before && !/\s$/.test(before) ? " " : "";
     input.value = before + sep + text + value.slice(end);
     var pos = (before + sep + text).length;
     input.focus();
     try { input.setSelectionRange(pos, pos); } catch (err) { /* not focusable yet */ }
     input.dispatchEvent(new Event("input"));
   }

   // ---------------------------------------------------- notifications ---

   // The header bell: what the agent and scheduled tasks told this user
   // (GET /api/notifications). Polled from the health poll, at most every
   // 15 seconds; a server without the route hides the bell for good.
   var chatNotifs = [];
   var chatNotifAt = 0;
   var chatNotifOff = false;

   async function chatPollNotifications(force) {
     if (chatNotifOff) return;
     var now = Date.now();
     if (!force && now - chatNotifAt < 15000) return;
     chatNotifAt = now;
     try {
       var out = await fetchJSON("/api/notifications?limit=30");
       chatNotifs = (out.data && out.data.notifications) || [];
       document.getElementById("notifMenu").classList.remove("hidden");
       chatNotifRender();
     } catch (err) {
       if (err.status === 404) {
         chatNotifOff = true;
         document.getElementById("notifMenu").classList.add("hidden");
       }
     }
   }

   function chatNotifRender() {
     var unread = chatNotifs.filter(function(n) { return !n.read_at; }).length;
     var badge = document.getElementById("notifBadge");
     badge.textContent = unread > 9 ? "9+" : String(unread);
     badge.classList.toggle("hidden", !unread);
     var btn = document.getElementById("notifBtn");
     btn.setAttribute("aria-label", unread ? "Notifications, " + unread + " unread" : "Notifications");
     var list = document.getElementById("notifList");
     while (list.firstChild) list.removeChild(list.firstChild);
     if (!chatNotifs.length) {
       var empty = document.createElement("div");
       empty.className = "empty";
       empty.textContent = "Nothing yet.";
       list.appendChild(empty);
       return;
     }
     chatNotifs.forEach(function(n) {
       var item = document.createElement("button");
       item.type = "button";
       item.setAttribute("role", "menuitem");
       item.className = "notif-item" + (n.read_at ? "" : " unread") + (n.level ? " " + n.level : "");
       if (n.title) {
         var title = document.createElement("span");
         title.className = "notif-title";
         title.textContent = n.title;
         item.appendChild(title);
       }
       var body = document.createElement("span");
       body.className = "notif-body";
       body.textContent = n.body || "";
       item.appendChild(body);
       var when = document.createElement("span");
       when.className = "notif-when";
       when.textContent = String(n.created_at || "").replace("T", " ").slice(0, 16)
         + (n.source ? " · " + n.source : "");
       item.appendChild(when);
       item.onclick = function() { chatNotifRead(n); };
       list.appendChild(item);
     });
   }

   async function chatNotifRead(n) {
     if (n.read_at) return;
     n.read_at = new Date().toISOString();
     chatNotifRender();
     try {
       await postJSON("/api/notifications/" + encodeURIComponent(n.id) + "/read", {});
     } catch (err) { /* the next poll shows the server's view */ }
   }

   async function chatNotifReadAll() {
     var stamp = new Date().toISOString();
     chatNotifs.forEach(function(n) { if (!n.read_at) n.read_at = stamp; });
     chatNotifRender();
     await uiAct("Mark all read", function() { return postJSON("/api/notifications/read-all", {}); });
   }

   function chatNotifToggle(force) {
     var pop = document.getElementById("notifPop");
     var btn = document.getElementById("notifBtn");
     var open = (typeof force === "boolean") ? force : pop.classList.contains("hidden");
     pop.classList.toggle("hidden", !open);
     btn.setAttribute("aria-expanded", open ? "true" : "false");
     if (open) chatPollNotifications(true);
   }

   // ------------------------------------------------ composer wiring ---

   (function chatWireComposer() {
     var picker = document.getElementById("attInput");
     if (picker) picker.addEventListener("change", function() {
       chatAttUpload(picker.files);
       picker.value = "";
     });
     // Paste an image (a screenshot) or a file straight into the message box.
     input.addEventListener("paste", function(e) {
       var files = e.clipboardData && e.clipboardData.files;
       if (!files || !files.length || document.getElementById("attBtn").classList.contains("hidden")) return;
       e.preventDefault();
       chatAttUpload(files);
     });
     // Drop files anywhere on the conversation or the composer.
     var box = document.getElementById("composerBox");
     function chatHasFiles(e) {
       var types = e.dataTransfer && e.dataTransfer.types;
       return !!types && Array.prototype.indexOf.call(types, "Files") >= 0;
     }
     ["chatView", "composer"].forEach(function(id) {
       var zone = document.getElementById(id);
       if (!zone) return;
       zone.addEventListener("dragover", function(e) {
         if (!chatHasFiles(e) || document.getElementById("attBtn").classList.contains("hidden")) return;
         e.preventDefault();
         box.classList.add("dragging");
       });
       zone.addEventListener("dragleave", function(e) {
         if (e.target === zone) box.classList.remove("dragging");
       });
       zone.addEventListener("drop", function(e) {
         box.classList.remove("dragging");
         if (!chatHasFiles(e) || document.getElementById("attBtn").classList.contains("hidden")) return;
         e.preventDefault();
         chatAttUpload(e.dataTransfer.files);
       });
     });
     // The bell's popover closes on an outside click or Escape, like the menus.
     document.addEventListener("click", function(e) {
       var pop = document.getElementById("notifPop");
       if (!pop || pop.classList.contains("hidden")) return;
       if (e.target.closest && e.target.closest("#notifMenu")) return;
       chatNotifToggle(false);
     });
     document.addEventListener("keydown", function(e) {
       if (e.key !== "Escape") return;
       var pop = document.getElementById("notifPop");
       if (pop && !pop.classList.contains("hidden")) {
         chatNotifToggle(false);
         document.getElementById("notifBtn").focus();
       }
     });
   })();
"""

__all__ = ["UI_JS_CHAT"]
