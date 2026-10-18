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
  * composer: the message box with one toolbar (mode, prompts, context meter,
    Stop, Send);
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
 <!-- Texcel brand typeface. Loaded when online; the CSS font stack falls back to
      a clean geometric system sans offline, so the app never blocks on this. -->
 <link rel="preconnect" href="https://fonts.googleapis.com">
 <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
 <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Raleway:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400&display=swap">
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
   <div class="composer-box">
     <textarea id="input" class="input textarea" rows="1" placeholder="Send a message. Enter sends, Shift+Enter adds a line." aria-label="Message"></textarea>
     <div class="composer-toolbar">
       <div class="seg" id="modeSeg" role="radiogroup" aria-label="Mode">
         <button type="button" data-value="auto" aria-pressed="true" class="active" data-tip="Auto: the server decides when a turn needs tools (the installation default).">Auto</button>
         <button type="button" data-value="chat" aria-pressed="false" data-tip="Chat: one plain reply, no tools.">Chat</button>
         <button type="button" data-value="agent" aria-pressed="false" data-tip="Agent: tools (search, files, code) and step-by-step reasoning.">Agent</button>
       </div>
       <button type="button" id="promptsBtn" class="btn-ghost btn-sm" onclick="togglePrompts()" data-tip="Insert a saved prompt, or save the message box as one.">Prompts</button>
       <div id="ctxMeter" class="ctx-meter" data-tip="Context usage in this conversation.">
         <div class="meter"><div id="ctxBar"></div></div>
         <span id="ctxHint" class="hint mono">0 / 4096</span>
       </div>
       <span id="composerMeta" class="composer-meta muted mono"></span>
       <div class="spacer"></div>
       <button type="button" id="stopBtn" onclick="stopStream()" disabled data-tip="Stop the current response. Keeps whatever streamed so far.">Stop</button>
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
