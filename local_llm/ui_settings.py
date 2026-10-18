"""Front-end: the Settings view, rendered from settings_schema at import time.

Settings used to be an aside with hand-written markup per control, plus a
separate admin "Models" view with its own sub-navigation, plus a Backup panel
under History. Options lived in three places, two controls wrote the same key,
and most runtime-mutable settings had no control at all. This module renders
ONE view, #settingsView, from SETTINGS_SECTIONS / SETTINGS_FIELDS: a rail of
sections, a search box, collapsible groups of fields, sub-tabs for the two
big sections, and the bespoke panels (model switcher, users, agents, cluster,
import, codebase, backup, knowledge base, notes, tool catalogue) moved here
with their element ids unchanged so every loader keeps finding its host.

It is rendered once at import time, so the page is still one static string:
every control id is a literal the selftest can resolve, and there is no
runtime templating. ui.py substitutes UI_SETTINGS_HTML for the
<!--SETTINGS_VIEW--> marker in UI_BODY.

Control ids: a global (Config) key renders as cfg + PascalCase(key)
(model_cutoff -> cfgModelCutoff), a user preference as pref + PascalCase(key).
Every control carries data-key, data-scope and data-widget, which is all
ui_script_settings.py needs to load, track and save it: no per-field JS.
"""

from __future__ import annotations

import html as _uis_html
from typing import Any

from .core import *  # noqa: F401,F403
from .settings_schema import *  # noqa: F401,F403


def _uis_esc(text: Any) -> str:
    return _uis_html.escape(str(text), quote=True)


def _uis_num(value: Any) -> str:
    """A bound as an HTML attribute: 0.05 stays 0.05, 64.0 becomes 64."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _uis_control(f: dict[str, Any], cid: str, hint_id: str) -> str:
    """The control for one field (one implementation per widget kind)."""
    widget = f["widget"]
    attrs = [
        f'id="{cid}"',
        f'data-key="{_uis_esc(f["key"])}"',
        f'data-scope="{_uis_esc(f["scope"])}"',
        f'data-widget="{widget}"',
        f'data-type="{settings_field_type(f)}"',
        f'aria-describedby="{hint_id}"',
    ]
    if f.get("format"):
        attrs.append(f'data-format="{_uis_esc(f["format"])}"')
    if f.get("unit"):
        attrs.append(f'data-unit="{_uis_esc(f["unit"])}"')
    if f.get("nullable"):
        attrs.append('data-nullable="1"')
    if f.get("autosave"):
        attrs.append('data-autosave="1"')
    if f.get("restart"):
        attrs.append('data-restart="1"')
    bounds = "".join(f' {k}="{_uis_num(f[k])}"' for k in ("min", "max", "step") if k in f)
    placeholder = (f' placeholder="{_uis_esc(f["placeholder"])}"'
                   if f.get("placeholder") else "")
    common = " ".join(attrs)
    unit = (f'<span class="unit muted">{_uis_esc(f["unit"])}</span>'
            if f.get("unit") and widget in ("number", "text") else "")

    if widget == "switch":
        return (f'<label class="switch"><input type="checkbox" {common}>'
                f'<span class="slider"></span></label>')
    if widget == "range":
        return (f'<div class="range"><input type="range" {common}{bounds}>'
                f'<output class="range-value" for="{cid}"></output></div>')
    if widget == "number":
        return f'<input class="input" type="number" {common}{bounds}{placeholder}>{unit}'
    if widget == "text":
        return (f'<input class="input" type="text" {common}{placeholder}'
                f' autocomplete="off" spellcheck="false">{unit}')
    if widget == "textarea":
        rows = int(f.get("rows", 4))
        return (f'<textarea class="input textarea" rows="{rows}" {common}{placeholder}'
                f' spellcheck="false"></textarea>')
    if widget == "secret":
        return (f'<input class="input" type="password" placeholder="(set)" {common}'
                f' autocomplete="new-password">')
    if widget == "select":
        options = "".join(f'<option value="{_uis_esc(o["value"])}">{_uis_esc(o["label"])}</option>'
                          for o in f.get("options", []))
        return f'<select class="input select" {common}>{options}</select>'
    if widget == "segmented":
        buttons = "".join(
            f'<button type="button" class="btn-sm" data-value="{_uis_esc(o["value"])}"'
            f' aria-pressed="false">{_uis_esc(o["label"])}</button>'
            for o in f.get("options", []))
        return (f'<div class="seg" role="radiogroup" aria-labelledby="{cid}Label" {common}>'
                f'{buttons}</div>')
    if widget == "readonly":
        return f'<div class="readonly mono" {common}>...</div>'
    raise ValueError(f"settings field {f['key']}: unknown widget {widget!r}")


def _uis_field(f: dict[str, Any]) -> str:
    cid = settings_control_id(f)
    hint_id = cid + "Hint"
    classes = ["field"]
    if f["widget"] == "switch":
        classes.append("field-row")
    if f["scope"] != "user":
        # Cosmetic gating only (the server enforces): non-admins never get
        # installation values, so an empty control would just mislead.
        classes.append("admin-only")
    badge = ' <span class="badge">restart</span>' if f.get("restart") else ""
    label_text = _uis_esc(f["label"]) + badge
    if f["widget"] in ("segmented", "readonly"):
        # Not a labelable element: name it by id instead of label[for].
        label = f'<span class="field-label" id="{cid}Label">{label_text}</span>'
    else:
        label = f'<label class="field-label" for="{cid}">{label_text}</label>'
    return (f'<div class="{" ".join(classes)}" data-field="{_uis_esc(f["key"])}"'
            f' data-scope="{f["scope"]}">'
            f'{label}'
            f'<div class="field-control">{_uis_control(f, cid, hint_id)}</div>'
            f'<p class="field-hint" id="{hint_id}">{_uis_esc(f.get("help", ""))}</p>'
            f'</div>')


def _uis_groups(fields: list[dict[str, Any]]) -> str:
    """Collapsible groups in first-appearance order; advanced fields fold away."""
    order: list[str] = []
    by_group: dict[str, list[dict[str, Any]]] = {}
    for f in fields:
        if f["group"] not in by_group:
            order.append(f["group"])
            by_group[f["group"]] = []
        by_group[f["group"]].append(f)
    out = []
    for group in order:
        items = by_group[group]
        plain = "".join(_uis_field(f) for f in items if not f.get("advanced"))
        advanced = "".join(_uis_field(f) for f in items if f.get("advanced"))
        if advanced:
            gated = all(f["scope"] != "user" for f in items if f.get("advanced"))
            plain += (f'<details class="sgroup-advanced{" admin-only" if gated else ""}">'
                      f'<summary>Advanced</summary><div class="sbody">{advanced}</div></details>')
        admin_only = all(f["scope"] != "user" for f in items)
        out.append(
            f'<details class="sgroup{" admin-only" if admin_only else ""}" open'
            f' data-group="{_uis_esc(group)}"><summary>{_uis_esc(group)}</summary>'
            f'<div class="sbody">{plain}</div></details>')
    return "".join(out)


# --------------------------------------------------------------------------- #
# Bespoke panels: markup moved from the old #modelsView, the settings aside and
# the History view's Backup panel. Element ids are preserved verbatim because
# the loaders (ui_script_admin.py, ui_script_models.py, ui_script_auth.py,
# ui_script_tasks.py, ui_script_chat.py) look them up by id. Inline styles are
# gone: layout comes from the kit classes in ui_styles.py.
# --------------------------------------------------------------------------- #

_UIS_PANEL_MODEL = r"""
<div class="panel-block card" data-panel="model">
  <div class="card-header"><h3 class="h3">Current model</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm" onclick="opsModelTab()" data-tip="Reload the list of available and cached models." title="Refresh model list">Refresh</button>
      <button type="button" class="btn btn-sm" onclick="restartModel()" data-tip="Restart the local model server. Use it if the server stops responding." title="Restart model server">Restart server</button>
    </div>
  </div>
  <pre id="modelCurrent" class="logbox mono">loading...</pre>
  <h3 class="h3">Switch model</h3>
  <table class="table models"><tbody id="modelTable"></tbody></table>
  <div class="field">
    <label class="field-label" for="modelCustom">Or any Hugging Face repo id or alias</label>
    <div class="field-control row">
      <input id="modelCustom" class="input grow" type="text" list="opsAliasList" placeholder="mlx-community/Qwen2.5-7B-Instruct-4bit or coder" autocomplete="off" spellcheck="false">
      <datalist id="opsAliasList"></datalist>
      <button type="button" class="btn" onclick="useModel(document.getElementById('modelCustom').value.trim())" data-tip="Queue this model id (Hugging Face / mlx-community) to load." title="Use this model">Use</button>
    </div>
  </div>
  <h3 class="h3">Adapter and cache</h3>
  <div class="row">
    <div class="field grow">
      <label class="field-label" for="adapterSelect">LoRA adapter</label>
      <div class="field-control"><select id="adapterSelect" class="input select"></select></div>
    </div>
    <div class="field grow">
      <label class="field-label" for="kvSize">KV cache cap <span class="badge">restart</span></label>
      <div class="field-control"><input id="kvSize" class="input" type="number" min="0" step="512"><span class="unit muted">tokens, 0 = unbounded</span></div>
    </div>
  </div>
  <div class="row">
    <button type="button" class="btn btn-primary" onclick="applyModel()" data-tip="Load the selected model and adapter. Restarts the model server." title="Apply and restart">Apply and restart</button>
    <button type="button" class="btn btn-ghost" onclick="openSettings('admin', 'logs')" title="Model server log">Show server log</button>
  </div>
  <p class="hint">Switching to a model that is not cached downloads it on first use, which can take minutes and several gigabytes. Admin &gt; Logs shows the model server's own output, including download progress.</p>
</div>
"""

_UIS_PANEL_TRAINING = r"""
<div class="panel-block card" data-panel="training">
  <div class="card-header"><h3 class="h3">Retrain on feedback</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-primary btn-sm" onclick="retrain()" data-tip="Fine-tune the model on your thumbs-up/down feedback so far (LoRA)." title="Retrain on feedback">Retrain now</button>
    </div>
  </div>
  <p class="hint">Runs a LoRA fine-tune from your approved answers. Ratings from non-admin accounts wait for approval below and never train the shared adapter until you approve them.</p>
  <p class="hint mono" id="retrainRecipe">&nbsp;</p>
</div>
<div class="panel-block card" data-panel="dataset">
  <div class="card-header"><h3 class="h3">Training data</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm" onclick="loadDatasetStats()" data-tip="Refresh the dataset counts." title="Refresh stats">Refresh</button>
      <button type="button" id="approvePendingBtn" class="btn btn-sm" hidden onclick="approvePendingFeedback()" data-tip="Ratings from non-admin accounts wait here. They never train the shared adapter until you approve them." title="Approve queued examples">Approve queued</button>
    </div>
  </div>
  <pre id="datasetStats" class="logbox mono">loading...</pre>
  <p class="hint">Your feedback is stored as portable data, not tied to the current model. Export it to train this assistant now, or any model later. Approved answers become examples to imitate; rejected answers and your corrections become preference pairs for DPO-style tuning.</p>
  <div class="field field-row">
    <label class="field-label" for="exportReviewedOnly">Export reviewed rows only</label>
    <div class="field-control"><label class="switch"><input id="exportReviewedOnly" type="checkbox"><span class="slider"></span></label></div>
    <p class="field-hint">Curated data trains better on any model.</p>
  </div>
  <div class="row wrap">
    <button type="button" class="btn" onclick="exportDataset('chat')" data-tip="Chat messages WITH this assistant's system prompt. Trains a model to be this assistant. Used by the built-in LoRA loop." title="Export chat JSONL">Chat (this assistant)</button>
    <button type="button" class="btn" onclick="exportDataset('bare')" data-tip="Chat messages with NO system prompt. Model-neutral: train a different base model or persona." title="Export bare JSONL">Bare Q&amp;A (any model)</button>
    <button type="button" class="btn" onclick="exportDataset('preference')" data-tip="{prompt, chosen, rejected} pairs from corrections and good/bad answers. For DPO-style preference tuning later." title="Export preference pairs">Preference pairs (DPO)</button>
    <button type="button" class="btn" onclick="exportDataset('raw')" data-tip="Every column as JSONL. A lossless archive you can reshape into any format in the future." title="Export raw archive">Raw archive</button>
  </div>
</div>
"""

_UIS_PANEL_KNOWLEDGE = r"""
<div class="panel-block card" data-panel="knowledge">
  <div class="card-header"><h3 class="h3">Knowledge base</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm btn-danger" onclick="clearDocs()" data-tip="Remove every indexed document. Your files are not touched." title="Clear">Clear all</button>
    </div>
  </div>
  <pre id="docsStats" class="logbox mono">loading...</pre>
  <p class="hint">Indexed documents are searched on every question and the best passages are added to the prompt with their source path. SQLite full-text ranking needs no embedding model and no extra memory. PDFs, Word documents, notebooks, spreadsheets and text are all read.</p>
  <div id="dropZone" class="drop-zone" data-tip="Drag files here, or click to choose, to add them to the knowledge base.">
    <strong>Drop files here</strong> or click to choose: PDFs, Word docs, spreadsheets, notebooks, text.
  </div>
  <input id="fileInput" type="file" multiple hidden>
  <div class="field">
    <label class="field-label" for="docsPath">File or folder to index (relative to the project)</label>
    <div class="field-control row">
      <input id="docsPath" class="input grow" type="text" placeholder="docs" autocomplete="off">
      <button type="button" class="btn" onclick="indexDocs()" data-tip="Read and index this file or folder into the knowledge base." title="Index">Index</button>
    </div>
  </div>
  <div class="field">
    <label class="field-label" for="docsUrl">Web page to index</label>
    <div class="field-control row">
      <input id="docsUrl" class="input grow" type="text" placeholder="https://example.com/docs" autocomplete="off">
      <button type="button" class="btn" onclick="indexUrl()" data-tip="Fetch a web page and add it to the knowledge base so you can ask about it later without re-fetching." title="Index URL">Index URL</button>
    </div>
  </div>
  <div class="field">
    <label class="field-label" for="docsQuery">Test retrieval</label>
    <div class="field-control row">
      <input id="docsQuery" class="input grow" type="text" placeholder="test a search..." autocomplete="off">
      <button type="button" class="btn" onclick="searchDocs()" data-tip="Preview what the model would retrieve for this question." title="Search">Search</button>
    </div>
  </div>
  <pre id="docsResult" class="gbody gdiff hidden" hidden></pre>
  <div class="field">
    <span class="field-label" data-tip="Limit retrieval to specific documents, so answers come only from what you choose.">Answer only from these documents</span>
    <div id="docScope" class="scope-list stack"></div>
  </div>
  <div class="row">
    <button type="button" class="btn" onclick="applyScope()" data-tip="Restrict retrieval to the ticked documents." title="Apply scope">Apply scope</button>
    <button type="button" class="btn btn-ghost" onclick="clearScope()" data-tip="Search the whole knowledge base again." title="Use all documents">Use all</button>
  </div>
</div>
"""

_UIS_PANEL_PERF = r"""
<div class="panel-block card" data-panel="perf">
  <div class="card-header"><h3 class="h3">Live performance</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm" onclick="loadPerf()" title="Refresh stats">Refresh</button>
    </div>
  </div>
  <div class="tools-list mono" id="perfStats">no samples yet</div>
  <div id="prefillCurve" class="chart"></div>
  <p class="hint">Prompt tokens climbing step over step within one run is re-prefill cost. Flat means the prefix is being reused.</p>
  <p class="hint warn hidden" id="prefixWarn" hidden></p>
</div>
"""

_UIS_PANEL_MEMORY = r"""
<div class="panel-block card" data-panel="notes">
  <div class="card-header"><h3 class="h3">My notes</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm" onclick="loadMemory()" title="Refresh notes">Refresh</button>
    </div>
  </div>
  <p class="hint">Notes the agent stores with the remember tool, and anything you add here. They persist across restarts and are private to your account.</p>
  <div class="row">
    <input id="memKey" class="input" type="text" placeholder="key" aria-label="Note key" autocomplete="off">
    <input id="memValue" class="input grow" type="text" placeholder="value" aria-label="Note value" autocomplete="off">
    <button type="button" class="btn btn-primary" onclick="saveMemory()">Store</button>
  </div>
  <div class="tools-list" id="memoryList">loading...</div>
</div>
"""

_UIS_PANEL_TOOLS = r"""
<div class="panel-block card" data-panel="catalogue">
  <div class="card-header"><h3 class="h3">Tool catalogue</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm" onclick="loadToolsCatalogue()" title="Refresh tools">Refresh</button>
    </div>
  </div>
  <p class="hint">The tools the agent can call right now. Which of them a named agent may use is set per agent in Admin &gt; Agents.</p>
  <div id="toolsList" class="tools-list">loading...</div>
</div>
"""

_UIS_PANEL_MCP = r"""
<div class="panel-block card admin-only" data-panel="mcp">
  <div class="card-header"><h3 class="h3">MCP servers</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm" onclick="loadMcpStatus()" title="Refresh MCP status">Refresh</button>
    </div>
  </div>
  <p class="hint">External tool servers from MCP_SERVERS. A server that failed to start is listed with its error; its tools are simply missing otherwise.</p>
  <div id="mcpStatus" class="tools-list">loading...</div>
</div>
"""

_UIS_PANEL_USERS = r"""
<div class="panel-block card" data-panel="users">
  <div class="card-header"><h3 class="h3">Users</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm" onclick="loadUsers()" title="Refresh user list">Refresh</button>
    </div>
  </div>
  <p class="hint">Admins reach every section and manage users; standard users get the chat, their own history, notes and knowledge base. Disabling or changing a user signs them out immediately. The last remaining admin cannot be removed.</p>
  <div id="usersList" class="stack">loading...</div>
  <h3 class="h3">Add a user</h3>
  <div class="row wrap">
    <input id="newUserName" class="input" type="text" placeholder="username" aria-label="Username" autocomplete="off">
    <input id="newUserPass" class="input" type="password" placeholder="password" aria-label="Password" autocomplete="new-password">
    <select id="newUserRole" class="input select" aria-label="Role">
      <option value="user">user</option>
      <option value="admin">admin</option>
    </select>
    <button type="button" class="btn btn-primary" onclick="createUser()" data-tip="Create this local user." title="Add user">Add user</button>
  </div>
</div>
"""

_UIS_PANEL_AGENTS = r"""
<div class="panel-block card" data-panel="agents">
  <div class="card-header"><h3 class="h3">Agents</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm" onclick="loadAgents()" title="Refresh agents">Refresh</button>
    </div>
  </div>
  <p class="hint">An agent is a named profile that switches on capabilities. It only gets the tools you enable here: file operations, running code, web and APIs, your knowledge base, memory, or Office&nbsp;365. Create a few specialised agents, then run several together on one task below.</p>
  <div id="agentsList" class="stack">loading...</div>
</div>
<div class="panel-block card" data-panel="newagent">
  <div class="card-header"><h3 class="h3">New agent</h3></div>
  <div class="stack">
    <input id="newAgentName" class="input" type="text" placeholder="name, e.g. Coder" aria-label="Agent name" autocomplete="off">
    <input id="newAgentDesc" class="input" type="text" placeholder="what this agent is for (optional)" aria-label="Agent description" autocomplete="off">
    <div id="newAgentCaps" class="cap-grid"></div>
  </div>
  <div class="row">
    <button type="button" class="btn btn-primary" onclick="createAgent()" data-tip="Create this agent profile." title="Add agent">Add agent</button>
  </div>
</div>
<div class="panel-block card" data-panel="agentrun">
  <div class="card-header"><h3 class="h3">Run several agents on one task</h3></div>
  <p class="hint">Each selected agent works the task with only its own tools; their answers are merged into one.</p>
  <div id="agentRunPick" class="cap-grid"></div>
  <textarea id="agentRunPrompt" class="input textarea" rows="3" placeholder="describe the task for the selected agents..." aria-label="Task for the agents"></textarea>
  <div class="row">
    <button type="button" class="btn btn-primary" onclick="runAgents()" data-tip="Run the selected agents together and merge their answers." title="Run together">Run together</button>
  </div>
  <div id="agentRunResult" class="stack"></div>
</div>
"""

_UIS_PANEL_CLUSTER = r"""
<div class="panel-block card" data-panel="cluster">
  <div class="card-header"><h3 class="h3">Cluster and routing</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm" onclick="opsLoadCluster()" data-tip="Refresh node health and routing history." title="Refresh cluster">Refresh</button>
    </div>
  </div>
  <p class="hint">Work is classified and routed automatically: the primary handles light chat, heavy reasoning and large models prefer the high-memory secondary, and a node that fails is skipped (with a cooldown and a half-open retry) so the other keeps serving. Nothing here is a machine you pick by hand.</p>
  <p class="hint" id="opsClusterTopology"></p>
  <div id="clusterNodes" class="stack tight"><p class="muted">loading...</p></div>
  <h3 class="h3">Layer sharding</h3>
  <div id="opsShard" class="stack tight"></div>
  <p class="hint">The mlx-distributed engine splits one model's layers across Macs. No route computes a shard plan from here; on the primary, <span class="mono">python deploy.py cluster plan --model &lt;repo id&gt;</span> probes the nodes and says whether the model fits, and <span class="mono">python deploy.py cluster hostfile</span> writes the mlx.launch hostfile.</p>
  <p id="routingSummary" class="hint"></p>
  <h3 class="h3">Recent routing decisions</h3>
  <pre id="routingEvents" class="logbox mono">loading...</pre>
</div>
"""

_UIS_PANEL_IMPORT = r"""
<div class="panel-block card" data-panel="import">
  <div class="card-header"><h3 class="h3">Import chat history</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm" onclick="loadImports()" title="Refresh import list">Refresh</button>
    </div>
  </div>
  <p class="hint">Upload an &ldquo;Export data&rdquo; .zip from Claude, ChatGPT, DeepSeek or xAI. Conversations become browsable history, their text is indexed into your knowledge base for retrieval, and project instructions are saved as reusable prompts. The archive is validated and sandboxed (path-traversal, zip-bomb and size limits). Retrieval surfaces only relevant passages; nothing is pasted into every prompt. Imported data is private to your account.</p>
  <div class="row">
    <input id="importFile" class="input grow" type="file" accept=".zip,application/zip" aria-label="Chat export archive">
    <button type="button" class="btn btn-primary" onclick="startImport()" data-tip="Upload and process the selected chat export." title="Import">Import</button>
  </div>
  <p id="importStatus" class="hint"></p>
  <div id="importList" class="stack">loading...</div>
</div>
"""

_UIS_PANEL_CODEBASE = r"""
<div class="panel-block card" data-panel="codebase">
  <div class="card-header"><h3 class="h3">Codebase</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm" onclick="loadProjectStatus()" data-tip="Refresh git status and the list of files changed this session." title="Refresh">Refresh</button>
    </div>
  </div>
  <div class="field">
    <label class="field-label" for="cfgProjectDir">Project directory (PROJECT_DIR)</label>
    <div class="field-control row">
      <input id="cfgProjectDir" class="input grow mono" type="text" placeholder="/Users/you/path/to/repo" autocomplete="off" spellcheck="false">
      <button type="button" class="btn" onclick="openBrowser()" data-tip="Browse your folders and pick the project directory." title="Browse for a folder">Browse...</button>
      <button type="button" class="btn btn-primary" onclick="saveProjectDir()" data-tip="Point the file tools at this local codebase. Empty = sandbox workspace only." title="Set project directory">Set</button>
    </div>
    <p class="field-hint">When set, the agent reads and edits files in this directory in place. It never touches .git and cannot write outside the directory. Changes are yours to review with git and push manually; nothing is committed for you.</p>
  </div>
  <pre id="projectStatus" class="logbox mono">loading...</pre>
  <div class="row">
    <button type="button" class="btn" onclick="loadProjectDiff()" data-tip="Show the uncommitted git diff so you can review before pushing." title="Show diff">Show diff</button>
    <button type="button" class="btn btn-danger" onclick="revertChanges()" data-tip="Undo this session's edits: restore modified files and delete files created this session. Cannot be undone." title="Revert session changes">Revert session changes</button>
  </div>
  <p class="hint" id="sandboxHint"></p>
  <pre id="projectDiff" class="gbody gdiff hidden" hidden></pre>
</div>
"""

_UIS_PANEL_BACKUP = r"""
<div class="panel-block card" data-panel="backup">
  <div class="card-header"><h3 class="h3">Backup</h3></div>
  <p class="hint">Downloads your conversations, prompts, feedback and indexed documents as one JSON file. Restoring merges a backup back in without deleting what is already here. Model weights are not included; they are re-downloadable.</p>
  <div class="row">
    <button type="button" class="btn btn-primary" onclick="downloadBackup()" data-tip="Download everything as a JSON file." title="Download backup">Download backup</button>
    <button type="button" class="btn" onclick="document.getElementById('restoreInput').click()" data-tip="Merge a previously downloaded backup back in." title="Restore backup">Restore backup</button>
  </div>
  <input id="restoreInput" type="file" accept="application/json" hidden>
</div>
"""

_UIS_PANEL_LOGS = r"""
<div class="panel-block card" data-panel="modellog">
  <div class="card-header"><h3 class="h3">Model server log</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm" onclick="loadModelLog()" title="Refresh log">Refresh</button>
    </div>
  </div>
  <pre id="modelLog" class="logbox mono">loading...</pre>
</div>
"""

# (section, sub) -> (markup before the schema groups, markup after them).
_UIS_PANELS: dict[tuple[str, str], tuple[str, str]] = {
    ("models", "model"): (_UIS_PANEL_MODEL, ""),
    ("models", "training"): (_UIS_PANEL_TRAINING, ""),
    ("models", "knowledge"): (_UIS_PANEL_KNOWLEDGE, ""),
    ("performance", ""): ("", _UIS_PANEL_PERF),
    ("memory", ""): (_UIS_PANEL_MEMORY, ""),
    ("tools", ""): (_UIS_PANEL_TOOLS, _UIS_PANEL_MCP),
    ("admin", "users"): (_UIS_PANEL_USERS, ""),
    ("admin", "agents"): (_UIS_PANEL_AGENTS, ""),
    ("admin", "cluster"): (_UIS_PANEL_CLUSTER, ""),
    ("admin", "import"): (_UIS_PANEL_IMPORT, ""),
    ("admin", "codebase"): (_UIS_PANEL_CODEBASE, ""),
    ("admin", "backup"): (_UIS_PANEL_BACKUP, ""),
    ("admin", "logs"): ("", _UIS_PANEL_LOGS),
}


# ---- panels-ops (Wave 4a: items 1, 4, 6, 7, 9, 20) ------------------------ #
# Live operational panels next to the schema fields they explain. The schema
# renders the restart-only facts (MODEL_BACKEND, WEB_HOST, ...) as read-only
# fields; these cards add what only the server knows at run time: which engine
# actually runs and why, the admission queue, the catalogue and benchmark, the
# training autopilot's plan, the node table and pairing, and whether remote
# access is configured safely. Loaders live in ui_script_ops.py (opsTabHook
# wires them into ADMIN_TABS). Every route behind them is admin-only on the
# server; the sections are admin-only in the rail, which is cosmetic.

_UIS_OPS_ENGINE = r"""
<div class="panel-block card" data-panel="engine">
  <div class="card-header"><h3 class="h3">Inference engine</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm" onclick="opsLoadEngine()" title="Refresh engine state">Refresh</button>
    </div>
  </div>
  <div class="row wrap" id="opsEngineState"><span class="muted">loading...</span></div>
  <dl class="kv" id="opsEngineFacts"></dl>
  <div class="stack tight" id="opsEngineDecisionsBox" hidden>
    <span class="field-label">Why it runs this way</span>
    <div class="stack tight" id="opsEngineDecisions"></div>
  </div>
  <div class="stack tight" id="opsEngineCacheBox" hidden>
    <span class="field-label">Prompt cache plan</span>
    <dl class="kv" id="opsEngineCache"></dl>
  </div>
  <span class="field-label">Engines on this machine</span>
  <table class="table"><thead><tr><th>Engine</th><th>Status</th><th>Capabilities</th></tr></thead><tbody id="opsEngineTable"></tbody></table>
  <p class="hint">Switching engines is restart-only: set <span class="mono">MODEL_BACKEND</span> in .env (or start with <span class="mono">python deploy.py --backend mlx-engine</span>) and restart the app. An engine marked unavailable says why; fix that first. mlx-engine keeps its own counters on the model port (<span class="mono">/v1/engine/stats</span>), which this app does not proxy.</p>
</div>
"""

_UIS_OPS_ADMISSION = r"""
<div class="panel-block card" data-panel="admission">
  <div class="card-header"><h3 class="h3">Admission</h3>
    <div class="card-actions">
      <span class="pill muted" id="opsAdmPill">idle</span>
      <button type="button" class="btn btn-sm" onclick="opsLoadAdmission()" title="Refresh the queue">Refresh</button>
    </div>
  </div>
  <div class="row wrap" id="opsAdmMeters"></div>
  <div class="stack tight">
    <div class="row nowrap"><span class="hint shrink-0">Running</span><div class="grow" id="opsAdmSparkRun"></div></div>
    <div class="row nowrap"><span class="hint shrink-0">Waiting</span><div class="grow" id="opsAdmSparkWait"></div></div>
  </div>
  <dl class="kv" id="opsAdmFacts"></dl>
  <p class="hint">Live every 3 seconds while this section is open. A request beyond the free slots waits in line (the chat shows its position); beyond the queue depth, or under the free-memory floor, it is refused with a retry hint. The slot count is restart-only (MAX_CONCURRENT_GENERATIONS); the Admission group below holds the live levers.</p>
</div>
"""

_UIS_OPS_CATALOG = r"""
<div class="panel-block card" data-panel="aliases">
  <div class="card-header"><h3 class="h3">Aliases</h3></div>
  <p class="hint">Short names the switcher, MODEL_ID and the API's model field accept. Add your own under Switcher &gt; Model aliases below.</p>
  <table class="table"><thead><tr><th>Alias</th><th>Model</th><th>Source</th><th></th></tr></thead><tbody id="opsAliasTable"></tbody></table>
</div>
<div class="panel-block card" data-panel="bench">
  <div class="card-header"><h3 class="h3">Benchmark</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm" onclick="opsLoadBench()" title="Refresh benchmark results">Refresh</button>
      <button type="button" class="btn btn-sm btn-primary" id="opsBenchRunBtn" onclick="opsRunBench()" title="Run a benchmark sweep">Run sweep</button>
    </div>
  </div>
  <div class="stack tight" id="opsBenchProgress" hidden>
    <div class="row nowrap"><span class="pill running" id="opsBenchPill">running</span><span class="hint mono grow" id="opsBenchCurrent"></span></div>
    <div class="meter"><div id="opsBenchMeter"></div></div>
  </div>
  <div class="stack tight" id="opsBenchRec"><p class="muted">loading...</p></div>
  <div id="opsBenchChart"></div>
  <table class="table"><thead><tr><th>Run</th><th>Decode</th><th>TTFT</th><th>Peak RAM</th><th>Result</th></tr></thead><tbody id="opsBenchTable"></tbody></table>
  <p class="hint" id="opsBenchFile"></p>
  <p class="hint">A sweep loads each cached model at several context sizes on each engine, measures decode speed, time to first token and peak memory, and recommends the best that fits this Mac. It stops the serving model for the whole run. From a shell: <span class="mono">python deploy.py bench --sweep</span> then <span class="mono">python deploy.py recommend --apply</span>.</p>
</div>
"""

_UIS_OPS_TRAINING = r"""
<div class="panel-block card" data-panel="trainmethods">
  <div class="card-header"><h3 class="h3">Methods and autopilot</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm" onclick="opsLoadTraining()" title="Refresh methods and plans">Refresh</button>
      <button type="button" class="btn btn-sm" id="opsTrainExportBtn" onclick="opsExportTraining()" data-tip="Write the selected method's training set to disk without training." title="Export dataset">Export dataset</button>
    </div>
  </div>
  <div class="seg" role="radiogroup" id="opsTrainSeg" aria-label="Training method to preview"></div>
  <p class="hint" id="opsTrainMethodHint"></p>
  <table class="table"><thead><tr><th>Method</th><th>Examples</th><th>Status</th></tr></thead><tbody id="opsTrainCounts"></tbody></table>
  <span class="field-label">Autopilot plan</span>
  <p class="hint" id="opsTrainPlanReason">loading...</p>
  <dl class="kv" id="opsTrainPlan"></dl>
  <div id="opsTrainPlanChart"></div>
  <dl class="kv" id="opsTrainExport" hidden></dl>
  <p class="hint">Previewing a method changes nothing: Retrain uses the Training method setting below. The plan is estimated from the model's config.json and the corpus, not measured, and errs high.</p>
</div>
"""

_UIS_OPS_PAIRING = r"""
<div class="panel-block card" data-panel="pairing">
  <div class="card-header"><h3 class="h3">Pair a node</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm btn-primary" onclick="opsCreatePairing()" title="Create a one-time pairing code">Create pairing code</button>
    </div>
  </div>
  <p class="hint">A pairing code works once and expires (10 minutes by default). Run the join command on the new Mac with the URL this primary reaches it at; the node receives its own token, and both machines need a restart before it serves.</p>
  <div class="stack tight" id="opsPairBox" hidden>
    <div class="row"><span class="ops-code" id="opsPairCode"></span><span class="pill muted" id="opsPairExpiry"></span></div>
    <div class="row nowrap"><div class="readonly mono grow" id="opsPairCmd"></div><button type="button" class="btn btn-sm" id="opsPairCopy" onclick="opsCopyPairCmd()">Copy</button></div>
    <p class="hint warn">Shown once: the server keeps only a hash of the code. Close this section and it is gone; create a new one if needed.</p>
  </div>
  <h3 class="h3">Paired nodes</h3>
  <table class="table"><thead><tr><th>Node</th><th>RAM</th><th>Joined</th><th>State</th><th></th></tr></thead><tbody id="opsPairedTable"></tbody></table>
  <p class="hint" id="opsPairedHint"></p>
</div>
"""

_UIS_OPS_REMOTE = r"""
<div class="panel-block card" data-panel="remote">
  <div class="card-header"><h3 class="h3">Remote access</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm" onclick="opsLoadRemote()" title="Refresh remote access status">Refresh</button>
    </div>
  </div>
  <div class="row wrap" id="opsRemotePills"><span class="muted">loading...</span></div>
  <dl class="kv" id="opsRemoteFacts"></dl>
  <div class="stack tight" id="opsRemoteChecks"></div>
  <span class="field-label">Publish it</span>
  <div class="stack tight" id="opsRemoteCmds"></div>
  <p class="hint">Each command prints what it would do; add <span class="mono">--apply</span> to run it. Tailscale Serve keeps the app on your tailnet, Funnel and Cloudflare Tunnel publish it to the internet (AUTH_ENABLED=1 required). Behind any of them set <span class="mono">TRUSTED_PROXIES=127.0.0.1,::1</span>, <span class="mono">PUBLIC_BASE_URL</span> and <span class="mono">AUTH_COOKIE_SECURE=1</span>, then restart.</p>
  <p class="hint">Identity-aware proxies (Cloudflare Access, Tailscale identity headers) are an extra gate in front of the login, not a replacement for it: the app does not trust their identity headers, so keep AUTH_ENABLED=1 (ideally with Entra ID) behind them.</p>
</div>
"""

# (section, sub) -> (markup appended before the schema groups, after them).
_UIS_OPS_PANELS: dict[tuple[str, str], tuple[str, str]] = {
    ("performance", ""): (_UIS_OPS_ENGINE + _UIS_OPS_ADMISSION, ""),
    ("models", "model"): (_UIS_OPS_CATALOG, ""),
    ("models", "training"): (_UIS_OPS_TRAINING, ""),
    ("admin", "cluster"): (_UIS_OPS_PAIRING, ""),
    ("admin", "system"): ("", _UIS_OPS_REMOTE),
}


def _uis_ops_merge() -> None:
    """Append the ops cards to _UIS_PANELS without rewriting its entries."""
    for key, (before, after) in _UIS_OPS_PANELS.items():
        old_before, old_after = _UIS_PANELS.get(key, ("", ""))
        _UIS_PANELS[key] = (old_before.strip() + before.strip(), old_after.strip() + after.strip())


_uis_ops_merge()
# ---- panels-ops end --------------------------------------------------------- #


def _uis_body(section: str, sub: str, fields: list[dict[str, Any]]) -> str:
    before, after = _UIS_PANELS.get((section, sub), ("", ""))
    return before.strip() + _uis_groups(fields) + after.strip()


def _uis_section(sec: dict[str, Any], first: bool) -> str:
    key = sec["key"]
    fields = [f for f in SETTINGS_FIELDS if f["section"] == key]
    classes = ["settings-section"]
    if sec.get("admin"):
        classes.append("admin-only")
    if not first:
        classes.append("hidden")
    parts = [
        f'<section class="{" ".join(classes)}" data-section="{key}"'
        f' data-title="{_uis_esc(sec["label"])}" data-desc="{_uis_esc(sec["description"])}"'
        f' aria-label="{_uis_esc(sec["label"])}">',
        # Shown only while searching, so a match says which section it is in.
        f'<h3 class="h3 settings-section-title hidden">{_uis_esc(sec["label"])}</h3>',
    ]
    tabs = SETTINGS_SUBTABS.get(key)
    if tabs:
        buttons = "".join(
            f'<button type="button" class="btn-sm{" active" if i == 0 else ""}"'
            f' data-sub="{t["key"]}" role="tab"'
            f' aria-pressed="{"true" if i == 0 else "false"}">{_uis_esc(t["label"])}</button>'
            for i, t in enumerate(tabs))
        parts.append(f'<div class="seg seg-tabs" role="tablist" data-subtabs="{key}"'
                     f' aria-label="{_uis_esc(sec["label"])} sections">{buttons}</div>')
        for i, t in enumerate(tabs):
            sub_fields = [f for f in fields if settings_field_sub(f) == t["key"]]
            hidden = "" if i == 0 else " hidden"
            parts.append(f'<div class="sub-panel{hidden}" data-sub="{t["key"]}">'
                         f'{_uis_body(key, t["key"], sub_fields)}</div>')
    else:
        parts.append(_uis_body(key, "", fields))
    parts.append("</section>")
    return "".join(parts)


def render_settings_html() -> str:
    """The whole #settingsView: rail, search, sections, settings bar."""
    rail = "".join(
        f'<button type="button" class="rail-item{" admin-only" if s.get("admin") else ""}'
        f'{" active" if i == 0 else ""}" data-section="{s["key"]}" role="tab"'
        f' aria-selected="{"true" if i == 0 else "false"}"'
        f' onclick="openSettings(\'{s["key"]}\')">{_uis_esc(s["label"])}</button>'
        for i, s in enumerate(SETTINGS_SECTIONS))
    sections = "".join(_uis_section(s, i == 0) for i, s in enumerate(SETTINGS_SECTIONS))
    first = SETTINGS_SECTIONS[0]
    return (
        '<div id="settingsView" class="view hidden" aria-label="Settings">'
        '<nav class="settings-rail" id="settingsRail" aria-label="Settings sections">'
        '<input type="search" id="settingsSearch" class="input" placeholder="Search settings"'
        ' aria-label="Search settings" autocomplete="off" spellcheck="false">'
        f'<div class="rail-items" role="tablist">{rail}</div>'
        '</nav>'
        '<div class="settings-main">'
        '<div class="settings-head">'
        f'<h2 class="h2" id="settingsTitle">{_uis_esc(first["label"])}</h2>'
        f'<p class="muted" id="settingsDesc">{_uis_esc(first["description"])}</p>'
        '</div>'
        '<p class="empty hidden" id="settingsNoMatch">No setting matches that search.</p>'
        f'{sections}'
        '<div class="settings-bar hidden" id="settingsBar" role="region" aria-label="Unsaved settings">'
        '<span id="settingsDirty" class="muted">No unsaved changes</span>'
        # The restart-flagged settings are read when the APP starts (tool
        # registration, stores), so restarting only the model server would not
        # apply them; the notice names them instead of offering that button.
        '<span id="settingsRestart" class="settings-restart hidden">'
        '<span class="badge">restart</span> <span id="settingsRestartText"></span>'
        '<button type="button" class="btn btn-sm btn-ghost" onclick="settingsDismissRestart()">Dismiss</button>'
        '</span>'
        '<span class="spacer"></span>'
        '<button type="button" class="btn" id="settingsDiscardBtn" onclick="discardSettings()">Discard</button>'
        '<button type="button" class="btn btn-primary" id="settingsApplyBtn" onclick="applySettings()">Apply</button>'
        '</div>'
        '</div>'
        '</div>'
    )


UI_SETTINGS_HTML = render_settings_html()


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    "UI_SETTINGS_HTML",
    "render_settings_html",
    "_uis_esc",
    "_uis_num",
    "_uis_control",
    "_uis_field",
    "_uis_groups",
    "_uis_body",
    "_uis_section",
]
