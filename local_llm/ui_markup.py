"""The HTML skeleton: header, views, side panels, overlays and the composer.

Part of the embedded UI, split out of the single HTML_PAGE string for
readability. Assembled once at import time in ui.py, so there is no
runtime cost and no extra file I/O.

The page is deliberately thin on the chat surface. The header used to carry
six action buttons and four nav buttons (Regenerate, Search and Export each
duplicated somewhere else, and Settings existed only for admins), so every
option lived in two places and none of them in the place you looked. Now:

  * header: brand, a segmented view switcher, one status pill, New chat,
    Settings (every role) and the account menu;
  * chat: a single overflow menu for conversation actions, plus Regenerate on
    the last answer itself;
  * composer: the message box with one toolbar (mode, attach, dictate, the
    project and agent pickers, prompts, context meter, Stop, Send); the
    extras stay hidden until the server says they apply;
  * header: a notifications bell, shown once GET /api/notifications answers;
  * header: the Approvals control, on every view, role and width: the user's
    tool approval mode at a glance (Ask every action / Ask for changes /
    Don't ask) and a menu that changes it at once. It is the "always visible"
    setting the ask-before-every-action rule depends on (policy.py);

  * Settings: one full view rendered by ui_settings.py in place of the
    SETTINGS_VIEW marker below, which also holds what used to be the admin
    "Models" view and the History > Backup panel.

No inline style attributes: every look comes from the class kit in
ui_styles.py, so a control looks the same wherever it appears.
"""

from __future__ import annotations

UI_HEAD = r"""
<!doctype html>
<html>
<head>
 <meta charset="utf-8">
 <meta name="viewport" content="width=device-width, initial-scale=1">
 <title>{{APP_NAME}}</title>
 <!-- The one interface typeface. Loaded when online; the CSS font stack falls
      back to the system sans offline, so the app never blocks on this. -->
 <link rel="preconnect" href="https://fonts.googleapis.com">
 <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
 <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap">
 <style>"""

UI_BODY = r"""</style>
</head>
<body class="booting">
 <header>
   <div class="brand"><span class="brand-slot">{{APP_LOGO}}</span><strong>{{APP_NAME}}</strong> <span class="build" title="UI build version">build {{UI_BUILD}}</span></div>
   <nav class="seg views" id="viewSeg" aria-label="Views">
     <button type="button" id="navChat" class="active" aria-pressed="true" onclick="showView('chat')" data-tip-below data-tip="Talk to the model. Ask questions, paste code or documents, run tools.">Chat</button>
     <button type="button" id="navHistory" aria-pressed="false" onclick="showView('history')" data-tip-below data-tip="Browse, search, reopen and manage past conversations.">History</button>
     <button type="button" id="navTasks" class="admin-only" aria-pressed="false" onclick="showView('tasks')" data-tip-below data-tip="Scheduled or saved jobs the agent runs on demand or on a timer.">Tasks</button>
   </nav>
   <div id="status" class="pill status-pill muted" role="status" aria-live="polite" data-tip-below data-tip="Connecting to the server.">Starting...</div>
   <div class="spacer"></div>
   <div class="actions">
     <!-- approvals: the always-visible Approvals control (ui_script_settings.py apprLoad/apprChoose, policy.py) -->
     <div class="appr" id="apprMenu">
       <button type="button" id="apprBtn" class="btn-ghost appr-btn" data-mode="all" onclick="apprToggle()" aria-haspopup="menu" aria-expanded="false" aria-controls="apprPop" aria-label="Approvals: Ask every action" data-tip-below data-tip="Approvals: Ask every action. Every tool call, even reading a file or listing a folder, waits for your yes."><svg class="icon appr-icon" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path class="appr-g appr-g-all" d="m9 12 2 2 4-4"/><path class="appr-g appr-g-risky" d="M12 22V2"/><path class="appr-g appr-g-off" d="m3 3 18 18"/></svg><span class="appr-prefix">Approvals:</span> <span id="apprLabel" class="appr-label">Ask every action</span><span id="apprShort" class="appr-short">Ask all</span><svg class="icon appr-lock" viewBox="0 0 24 24" width="12" height="12" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg><span class="caret" aria-hidden="true">&#9662;</span></button>
       <div class="menu appr-pop hidden" id="apprPop" role="menu" aria-label="Tool approvals">
         <div class="appr-head">
           <strong class="h3">Tool approvals</strong>
           <p class="hint" id="apprNote">When the assistant must ask you before it uses a tool. Applies to every chat at once.</p>
         </div>
         <button type="button" role="menuitemradio" aria-checked="true" class="appr-opt active" id="apprOptAll" data-value="all" onclick="apprChoose('all')"><span class="appr-opt-title">Ask every action</span><span class="appr-opt-help">Every tool call, even reading a file or listing a folder, waits for your yes.</span></button>
         <button type="button" role="menuitemradio" aria-checked="false" class="appr-opt" id="apprOptRisky" data-value="risky" onclick="apprChoose('risky')"><span class="appr-opt-title">Ask for changes</span><span class="appr-opt-help">Reading and searching run on their own; writing or deleting files, running code and sending anything wait for your yes.</span></button>
         <button type="button" role="menuitemradio" aria-checked="false" class="appr-opt" id="apprOptOff" data-value="off" onclick="apprChoose('off')"><span class="appr-opt-title">Don't ask</span><span class="appr-opt-help">Tools run without asking you; administrator rules still apply.</span></button>
         <p class="hint warn appr-locknote hidden" id="apprLockNote" role="note"></p>
         <p class="hint appr-unattended" id="apprUnattended">Scheduled tasks, API and MCP clients cannot show a prompt: what would ask is refused there.</p>
         <div class="menu-sep"></div>
         <button type="button" role="menuitem" id="apprMore" onclick="apprOpenSettings()">Approval rules and waiting approvals</button>
       </div>
     </div>
     <!-- approvals end -->
     <div class="notif hidden" id="notifMenu">
       <button type="button" id="notifBtn" class="btn-icon btn-ghost notif-btn" onclick="chatNotifToggle()" aria-haspopup="menu" aria-expanded="false" aria-controls="notifPop" aria-label="Notifications" data-tip-below data-tip="Notifications from the agent and from scheduled tasks."><svg class="icon" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/></svg><span id="notifBadge" class="badge notif-badge hidden" aria-hidden="true"></span></button>
       <div class="menu notif-pop hidden" id="notifPop" role="menu" aria-label="Notifications">
         <div class="row between notif-head">
           <strong class="h3">Notifications</strong>
           <button type="button" id="notifReadAll" class="btn-ghost btn-sm" onclick="chatNotifReadAll()">Mark all read</button>
         </div>
         <div id="notifList" class="stack notif-list"></div>
       </div>
     </div>
     <button type="button" id="newChatBtn" class="btn-icon btn-ghost" onclick="newChat()" aria-label="New chat" data-tip-below data-tip="Start a fresh conversation. The current one stays in History."><svg class="icon" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.4 2.6a2 2 0 0 1 3 3L12 15l-4 1 1-4Z"/></svg></button>
     <button type="button" id="settingsBtn" class="btn-icon btn-ghost" onclick="openSettings()" aria-label="Settings" data-tip-below data-tip="Every option in one place: appearance, generation, agent, memory and tools (plus models, performance and admin for administrators)."><svg class="icon" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg></button>
     <div class="acct" id="acctMenu">
       <button type="button" id="acctBtn" class="btn-ghost acct-btn" onclick="toggleAcct()" aria-haspopup="menu" aria-expanded="false" aria-controls="acctPop" data-tip-below data-tip="Your account and session."><span id="userChip" class="user-chip"></span> <span class="caret" aria-hidden="true">&#9662;</span></button>
       <div class="menu acct-pop hidden" id="acctPop" role="menu">
         <div class="acct-info" id="acctInfo"></div>
         <button type="button" id="logoutBtn" role="menuitem" onclick="doLogout()">Log out</button>
       </div>
     </div>
   </div>
 </header>

 <div id="loginOverlay">
   <div id="loginBox" class="stack">
     <div class="brand login-brand"><span class="brand-slot">{{APP_LOGO_LOGIN}}</span><strong>{{APP_NAME}}</strong></div>
     <p class="hint login-sub">Sign in to continue.</p>
     <label class="field-label" for="loginUser">Username</label>
     <input id="loginUser" class="input" type="text" autocomplete="username" onkeydown="loginKey(event)" placeholder="username">
     <label class="field-label" for="loginPass">Password</label>
     <div class="pw-field">
       <input id="loginPass" class="input" type="password" autocomplete="current-password" onkeydown="loginKey(event)" placeholder="password">
       <button type="button" class="btn-ghost btn-sm pw-reveal" onclick="togglePw('loginPass', this)" aria-label="Show or hide the password">show</button>
     </div>
     <div id="loginError" class="login-error" role="alert"></div>
     <button type="button" id="loginBtn" class="btn-primary login-submit" onclick="doLogin()">Sign in</button>
     <button type="button" id="oidcBtn" class="login-submit hidden" onclick="oidcLogin()">Sign in with Microsoft</button>
   </div>
 </div>

 <div id="main">
   <div id="chatView" class="view">
     <button type="button" id="chatMenuBtn" class="btn-icon btn-ghost" onclick="toggleChatMenu()" aria-haspopup="menu" aria-expanded="false" aria-controls="chatMenu" aria-label="Conversation actions" data-tip-below data-tip="Export, regenerate, search history, prompt library."><svg class="icon" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" fill="currentColor" stroke="none"><circle cx="5" cy="12" r="1.8"/><circle cx="12" cy="12" r="1.8"/><circle cx="19" cy="12" r="1.8"/></svg></button>
     <div id="chatMenu" class="menu hidden" role="menu" aria-label="Conversation actions">
       <button type="button" role="menuitem" onclick="chatMenuAction('export')">Export as Markdown</button>
       <button type="button" role="menuitem" onclick="chatMenuAction('regenerate')">Regenerate last answer</button>
       <button type="button" role="menuitem" onclick="chatMenuAction('search')">Search history</button>
       <button type="button" role="menuitem" onclick="chatMenuAction('prompts')">Prompts library</button>
     </div>
     <div id="chat" aria-live="polite"></div>
   </div>

   <div id="historyView" class="view hidden">
     <div class="panel stack">
       <h2 class="h2" data-tip="Every conversation stored locally. Reopen one to continue it.">Conversations</h2>
       <div class="row">
         <input id="historySearch" class="input grow" type="search" placeholder="Search all conversations..." aria-label="Search conversations" onkeydown="if(event.key==='Enter')runHistorySearch()">
         <button type="button" onclick="runHistorySearch()" data-tip="Find conversations containing this text.">Search</button>
         <button type="button" class="btn-ghost" onclick="clearHistorySearch()" data-tip="Clear the search and show all recent conversations.">Show all</button>
       </div>
       <div id="historyList" class="stack">loading...</div>
     </div>
   </div>

   <div id="tasksView" class="view hidden">
     <div class="task-list stack">
       <div class="row">
         <h2 class="h3 grow">Tasks</h2>
         <button type="button" class="btn-primary btn-sm" onclick="toggleTaskForm()">New task</button>
       </div>

       <div id="taskForm" class="card stack hidden">
         <div class="field">
           <label class="field-label" for="tfName">Name</label>
           <input id="tfName" class="input" type="text" placeholder="Morning news sweep">
         </div>
         <div class="field">
           <label class="field-label" for="tfGoal">Goal, written as an instruction to the agent</label>
           <textarea id="tfGoal" class="input textarea" placeholder="Search for news about MLX released in the last day and summarise anything new in three bullets."></textarea>
         </div>
         <div class="row">
           <div class="field grow">
             <label class="field-label" for="tfInterval">Repeat every (seconds)</label>
             <input id="tfInterval" class="input" type="number" min="0" step="60" value="0">
           </div>
           <div class="field grow">
             <label class="field-label" for="tfSteps" data-tip="Tool/reasoning steps per turn before the agent must answer. Big tasks continue past this from a summary.">Max steps</label>
             <input id="tfSteps" class="input" type="number" min="1" max="20" value="6">
           </div>
         </div>
         <div class="field">
           <label class="field-label" for="tfTools">Tools (comma separated, blank for all)</label>
           <input id="tfTools" class="input" type="text" placeholder="web_search, fetch_url, remember">
         </div>
         <div class="field">
           <label class="field-label" for="tfSystem">System prompt override (optional)</label>
           <textarea id="tfSystem" class="input textarea" placeholder="Leave blank to use the global system prompt."></textarea>
         </div>
         <div class="field">
           <label class="field-label" for="tfHistory">Keep conversation history</label>
           <select id="tfHistory" class="input select">
             <option value="false">no, each run is fresh</option>
             <option value="true">yes, runs build on each other</option>
           </select>
         </div>
         <div class="row">
           <button type="button" class="btn-primary" onclick="createTask()">Create</button>
           <button type="button" class="btn-ghost" onclick="toggleTaskForm()">Cancel</button>
         </div>
         <p class="hint">
           Interval 0 means the task only runs when you press Run. A repeating
           task fires once as soon as it is created, so you find out quickly
           whether the goal is worded well.
         </p>
       </div>

       <div id="taskCards" class="stack">loading...</div>
     </div>

     <div class="task-monitor stack">
       <div id="monitorHeader" class="muted">Select a task to watch its runs, or create one.</div>
       <div id="runControls" class="row hidden">
         <div class="field grow">
           <label class="field-label" for="runPicker">Run</label>
           <select id="runPicker" class="input select" onchange="openRun(this.value)"></select>
         </div>
         <button type="button" onclick="runSelectedTask()">Run now</button>
         <button type="button" class="btn-ghost" onclick="cancelSelectedTask()">Cancel run</button>
       </div>
       <div id="runFeed"></div>
     </div>
   </div>

   <!--SETTINGS_VIEW-->

   <div id="browser" class="hidden" role="dialog" aria-modal="true" aria-label="Choose a project folder">
     <div id="browserBox">
       <div id="browserPath" class="mono">~</div>
       <div id="browserList"></div>
       <div class="row browser-actions">
         <button type="button" class="btn-primary" onclick="chooseCurrentFolder()" data-tip="Use the folder shown above as the project directory.">Use this folder</button>
         <button type="button" class="btn-ghost" onclick="closeBrowser()">Cancel</button>
       </div>
     </div>
   </div>

   <div id="palette" class="hidden" role="dialog" aria-modal="true" aria-label="Command palette">
     <div id="paletteBox">
       <input id="paletteInput" class="input" type="text" placeholder="Type a command..." autocomplete="off" aria-label="Command">
       <div id="paletteList" role="listbox"></div>
     </div>
   </div>

   <aside id="promptsPanel" class="hidden stack" aria-label="Prompt library">
     <div class="row">
       <h2 class="h3 grow">Prompt library</h2>
       <button type="button" class="btn-icon btn-ghost" onclick="togglePrompts()" aria-label="Close the prompt library">&times;</button>
     </div>
     <p class="hint">Save prompts you type often, then insert one into the message box with a click.</p>
     <div id="promptList" class="stack">loading...</div>
     <div class="field">
       <label class="field-label" for="promptName">Save the current message box as</label>
       <div class="row">
         <input id="promptName" class="input grow" type="text" placeholder="name, e.g. code-review">
         <button type="button" class="btn-primary" onclick="savePrompt()" data-tip="Save whatever is in the message box under this name.">Save</button>
       </div>
     </div>
   </aside>
 </div>

 <footer id="composer">
   <div class="composer-box" id="composerBox">
     <textarea id="input" class="input textarea" rows="1" placeholder="Send a message. Enter sends, Shift+Enter adds a line." aria-label="Message"></textarea>
     <div id="attChips" class="att-chips hidden" aria-label="Attachments"></div>
     <input id="attInput" class="hidden" type="file" multiple aria-label="Attach files">
     <div class="composer-toolbar">
       <div class="seg" id="modeSeg" role="radiogroup" aria-label="Mode">
         <button type="button" data-value="auto" aria-pressed="true" class="active" data-tip="Auto: the server decides when a turn needs tools (the installation default).">Auto</button>
         <button type="button" data-value="chat" aria-pressed="false" data-tip="Chat: one plain reply, no tools.">Chat</button>
         <button type="button" data-value="agent" aria-pressed="false" data-tip="Agent: tools (search, files, code) and step-by-step reasoning.">Agent</button>
       </div>
       <button type="button" id="attBtn" class="btn-icon btn-ghost btn-sm hidden" onclick="chatAttPick()" aria-label="Attach files" data-tip="Attach images or files. You can also drop or paste them into the message box."><svg class="icon" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.4 11.1-9.2 9.2a6 6 0 0 1-8.5-8.5l9.2-9.2a4 4 0 0 1 5.7 5.7l-9.2 9.2a2 2 0 0 1-2.8-2.8l8.5-8.5"/></svg></button>
       <button type="button" id="micBtn" class="btn-icon btn-ghost btn-sm mic-btn hidden" onclick="chatMicToggle()" aria-label="Dictate" aria-pressed="false" data-tip="Dictate: record a voice note and insert its transcript."><svg class="icon" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M19 10v1a7 7 0 0 1-14 0v-1"/><path d="M12 18v4"/></svg></button>
       <select id="projectSel" class="select composer-select hidden" aria-label="Project" title="Run this turn inside one of your projects: its files, instructions and knowledge."><option value="">No project</option></select>
       <select id="agentSel" class="select composer-select hidden" aria-label="Agent profile" title="Run this turn as a saved agent profile (its tools and instructions)."><option value="">Default agent</option></select>
       <button type="button" id="promptsBtn" class="btn-ghost btn-sm" onclick="togglePrompts()" data-tip="Insert a saved prompt, or save the message box as one.">Prompts</button>
       <div id="ctxMeter" class="ctx-meter" data-tip="Context usage in this conversation.">
         <div class="meter"><div id="ctxBar"></div></div>
         <span id="ctxHint" class="hint mono">0 / 4096</span>
       </div>
       <span id="composerMeta" class="composer-meta muted mono"></span>
       <div class="spacer"></div>
       <button type="button" id="stopBtn" onclick="chatStop()" disabled data-tip="Stop the current response. Keeps whatever streamed so far.">Stop</button>
       <button type="button" id="sendBtn" class="btn-primary" onclick="send()" data-tip="Send your message.">Send</button>
     </div>
   </div>
 </footer>

 <div id="toastHost" class="toast-host" aria-live="polite"></div>

 <script>"""

UI_TAIL = r"""</script>
</body>
</html>
"""

__all__ = ["UI_HEAD", "UI_BODY", "UI_TAIL"]
