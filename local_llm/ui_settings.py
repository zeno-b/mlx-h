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
  <table class="table models"><thead><tr><th>Model</th><th>Fits this Mac</th><th>Status</th><th></th></tr></thead><tbody id="modelTable"></tbody></table>
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
      <button type="button" class="btn btn-sm" onclick="integLoadCatalogue()" title="Refresh the tool catalogue">Refresh</button>
    </div>
  </div>
  <p class="hint">Every tool the agent can call: the capability that grants it, how risky the approval policy rates it, where it comes from and which agent profiles reach it. Which tools a named agent may use is set per agent in Admin &gt; Agents; when to ask first is set under Approvals.</p>
  <div class="row wrap">
    <input type="search" class="input grow" id="integCatFilter" placeholder="Filter by name, capability or risk" aria-label="Filter tools" autocomplete="off" spellcheck="false">
    <div class="seg" role="radiogroup" id="integCatSource" aria-label="Where the tool comes from">
      <button type="button" class="btn-sm active" data-value="" aria-pressed="true">All</button>
      <button type="button" class="btn-sm" data-value="builtin" aria-pressed="false">Built in</button>
      <button type="button" class="btn-sm" data-value="extra" aria-pressed="false">Extra</button>
      <button type="button" class="btn-sm" data-value="mcp" aria-pressed="false">MCP</button>
    </div>
  </div>
  <div id="toolsList">loading...</div>
  <p class="hint" id="integCatSummary"></p>
</div>
"""

_UIS_PANEL_MCP = r"""
<div class="panel-block card admin-only" data-panel="mcp">
  <div class="card-header"><h3 class="h3">MCP servers</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm" onclick="integLoadMcp()" title="Refresh MCP status">Refresh</button>
    </div>
  </div>
  <p class="hint">External tool servers from MCP_SERVERS, as they are right now. A server that failed to start is listed with its error; its tools are simply missing otherwise. The capability mcp:&lt;server&gt; grants one server's tools, mcp grants them all.</p>
  <div id="mcpStatus">loading...</div>
  <h3 class="h3">This app as an MCP server</h3>
  <dl class="kv" id="integMcpServe"></dl>
  <p class="hint">With Serve tools over MCP on (below), MCP clients such as Claude Code or Codex reach this app's tools at that URL, signed in with an API token (chat scope) sent as a Bearer header. They run under the same roles, approvals and audit as chat.</p>
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
    ("tools", "catalogue"): (_UIS_PANEL_TOOLS, ""),
    ("tools", "mcp"): (_UIS_PANEL_MCP, ""),
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


# ---- panels-integrations (Wave 4a: items 2, 10, 16, 17) -------------------- #
# The Tools sub-tabs (Connect a client, API tokens, Approvals, MCP), Admin >
# Audit and Agent > Projects. The catalogue and MCP status cards above keep
# their ids (toolsList, mcpStatus) and are upgraded in place; these are new.
# Loaders live in ui_script_integrations.py and register with ADMIN_TABS, so a
# card loads only while its tab is visible. Tables are built by the script.
# Every route behind them enforces its own access (owner scoping for tokens,
# rules, projects and decisions; ADMIN for MCP and the full audit), so the
# admin-only classes here are cosmetic.

_UIS_INTEG_CLIENTS = r"""
<div class="panel-block card" data-panel="clients">
  <div class="card-header"><h3 class="h3">Connect a client</h3></div>
  <p class="hint">Point a coding agent or an SDK at this server's OpenAI and Anthropic compatible API. Pick the client to see what it needs.</p>
  <div class="seg wrap" role="radiogroup" id="integClientSeg" aria-label="Client">
    <button type="button" class="btn-sm active" data-value="claude-code" aria-pressed="true">Claude Code</button>
    <button type="button" class="btn-sm" data-value="aider" aria-pressed="false">Aider</button>
    <button type="button" class="btn-sm" data-value="codex" aria-pressed="false">Codex</button>
    <button type="button" class="btn-sm" data-value="continue" aria-pressed="false">Continue</button>
    <button type="button" class="btn-sm" data-value="opencode" aria-pressed="false">OpenCode</button>
    <button type="button" class="btn-sm" data-value="hermes" aria-pressed="false">Hermes</button>
    <button type="button" class="btn-sm" data-value="generic" aria-pressed="false">Generic</button>
  </div>
  <dl class="kv" id="integClientFacts"></dl>
  <span class="field-label">From a terminal</span>
  <div class="row nowrap"><div class="readonly mono grow" id="integClientCmd"></div><button type="button" class="btn btn-sm" onclick="integCopyClientCmd(this)">Copy</button></div>
  <p class="hint">Run it in this app's folder on the machine the client runs on. --login signs in with your username and password and mints a token for the client (--device uses Microsoft sign-in instead); add --write to merge the settings into the client's config file, which is backed up first. Without --write it only prints them.</p>
  <span class="field-label">Or set it up by hand</span>
  <div class="codeblock"><div class="codebar"><span id="integClientTitle">settings</span><button type="button" class="btn btn-sm" onclick="integCopyClientText(this)">Copy</button></div><pre id="integClientText">loading...</pre></div>
  <div class="stack tight" id="integClientNotes"></div>
</div>
"""

_UIS_INTEG_TOKENS = r"""
<div class="panel-block card" data-panel="tokens">
  <div class="card-header"><h3 class="h3">API tokens</h3>
    <div class="card-actions">
      <span class="hint admin-only" aria-hidden="true">All users</span>
      <label class="switch admin-only" id="integTokAllWrap" title="Show every user's tokens"><input type="checkbox" id="integTokAll" aria-label="Show every user's tokens"><span class="slider"></span></label>
      <button type="button" class="btn btn-sm" onclick="integLoadTokens()" title="Refresh tokens">Refresh</button>
    </div>
  </div>
  <p class="hint">A token lets a script, a coding agent or the CLI call this server as you (Authorization: Bearer). Give each client its own token, with the narrowest scopes it needs, so one can be revoked without breaking the others.</p>
  <p class="hint warn" id="integTokAuthOff" hidden>Sign-in is off on this server (AUTH_ENABLED=0), so clients need no token: any value works.</p>
  <div class="stack" id="integTokForm">
    <div class="field-row"><label class="field-label" for="integTokName">Name</label><div class="field-control"><input class="input" type="text" id="integTokName" maxlength="80" placeholder="laptop claude-code" autocomplete="off" spellcheck="false"></div></div>
    <div class="field-row"><span class="field-label" id="integTokScopesLabel">Scopes</span>
      <div class="field-control cap-grid" id="integTokScopes" role="group" aria-labelledby="integTokScopesLabel">
        <label class="check"><input type="checkbox" value="chat" checked><span>chat: conversations, the /v1 API and MCP</span></label>
        <label class="check"><input type="checkbox" value="status"><span>status: read-only health, models and tasks</span></label>
        <label class="check admin-only"><input type="checkbox" value="control"><span>control: switch models, run and cancel tasks</span></label>
        <label class="check"><input type="checkbox" value="admin"><span>admin: everything your account may do</span></label>
      </div>
    </div>
    <div class="field-row"><label class="field-label" for="integTokDays">Lifetime</label><div class="field-control"><input class="input" type="number" id="integTokDays" min="0" max="3650" step="1" placeholder="server default"><span class="unit muted">days</span></div></div>
    <div class="row"><button type="button" class="btn btn-sm btn-primary" id="integTokCreateBtn" onclick="integCreateToken()">Create token</button></div>
  </div>
  <div class="stack tight" id="integTokNewBox" hidden>
    <span class="field-label">Your new token</span>
    <div class="row nowrap">
      <input class="input mono grow" type="text" id="integTokNew" readonly aria-label="New API token" autocomplete="off" spellcheck="false">
      <button type="button" class="btn btn-sm" onclick="integCopyNewToken(this)">Copy</button>
      <button type="button" class="btn btn-sm btn-ghost" onclick="integHideNewToken()">Done</button>
    </div>
    <p class="hint warn">Shown once: the server keeps only a hash of it. Copy it into the client now; if it is lost, revoke it and create another.</p>
  </div>
  <div id="integTokList"><p class="muted">loading...</p></div>
</div>
"""

_UIS_INTEG_APPROVALS = r"""
<div class="panel-block card" data-panel="approvals">
  <div class="card-header"><h3 class="h3">Waiting for you</h3>
    <div class="card-actions">
      <span class="pill muted" id="integApprCount">0</span>
      <button type="button" class="btn btn-sm" onclick="integLoadPending()" title="Refresh pending approvals">Refresh</button>
    </div>
  </div>
  <p class="hint">Tool calls paused until you answer, from any of your chats or API clients. Answer here or in the chat that asked; a call nobody answers is refused when its time runs out.</p>
  <div class="stack" id="integApprList"><p class="muted">loading...</p></div>
</div>
<div class="panel-block card" data-panel="policies">
  <div class="card-header"><h3 class="h3">Tool rules</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm" onclick="integLoadPolicies()" title="Refresh rules">Refresh</button>
    </div>
  </div>
  <dl class="kv" id="integPolFacts"></dl>
  <p class="hint">A rule allows, asks first or denies one tool, a capability (cap:), a risk class (risk:) or every tool (*). Administrator rules (everyone, agent profile) come first and their deny is final; your own rules (you, one conversation) can tighten anything and turn an administrator's ask into allow.</p>
  <div class="stack" id="integPolForm">
    <div class="field-row"><span class="field-label" id="integPolScopeLabel">Applies to</span>
      <div class="field-control">
        <div class="seg" role="radiogroup" id="integPolScope" aria-labelledby="integPolScopeLabel">
          <button type="button" class="btn-sm active" data-value="user" aria-pressed="true">Me</button>
          <button type="button" class="btn-sm" data-value="conversation" aria-pressed="false">One conversation</button>
          <button type="button" class="btn-sm admin-only" data-value="profile" aria-pressed="false">Agent profile</button>
          <button type="button" class="btn-sm admin-only" data-value="global" aria-pressed="false">Everyone</button>
        </div>
      </div>
    </div>
    <div class="field-row" id="integPolTargetRow" hidden><label class="field-label" for="integPolTarget" id="integPolTargetLabel">Which</label><div class="field-control"><select class="input select" id="integPolTarget"></select></div></div>
    <div class="field-row"><label class="field-label" for="integPolTool">Tool</label><div class="field-control"><select class="input select" id="integPolTool"></select></div></div>
    <div class="field-row"><span class="field-label" id="integPolDecisionLabel">Decision</span>
      <div class="field-control">
        <div class="seg" role="radiogroup" id="integPolDecision" aria-labelledby="integPolDecisionLabel">
          <button type="button" class="btn-sm" data-value="allow" aria-pressed="false">Allow</button>
          <button type="button" class="btn-sm active" data-value="ask" aria-pressed="true">Ask first</button>
          <button type="button" class="btn-sm" data-value="deny" aria-pressed="false">Deny</button>
        </div>
      </div>
    </div>
    <div class="field-row"><label class="field-label" for="integPolExpires">Expires</label><div class="field-control"><select class="input select" id="integPolExpires">
      <option value="">Never</option><option value="3600">In 1 hour</option><option value="86400">In 1 day</option><option value="604800">In 7 days</option>
    </select></div></div>
    <div class="row"><button type="button" class="btn btn-sm btn-primary" onclick="integSavePolicy()">Save rule</button></div>
  </div>
  <div id="integPolList"><p class="muted">loading...</p></div>
  <h3 class="h3">Your recent tool decisions</h3>
  <div id="integMyAudit"><p class="muted">loading...</p></div>
  <div class="row"><span class="spacer"></span><button type="button" class="btn btn-sm" id="integMyAuditMore" onclick="integLoadMyAudit(false)" hidden>Older</button></div>
</div>
"""

_UIS_INTEG_MCPEDIT = r"""
<div class="panel-block card admin-only" data-panel="mcpedit">
  <div class="card-header"><h3 class="h3">Configure MCP servers</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm" onclick="integMcpAdd('stdio')" title="Add a server this machine starts as a process">Add stdio server</button>
      <button type="button" class="btn btn-sm" onclick="integMcpAdd('http')" title="Add a server reached over HTTP">Add HTTP server</button>
      <button type="button" class="btn btn-sm btn-ghost" onclick="integMcpLoadConfig()" title="Discard edits and reload the current setting">Reset</button>
    </div>
  </div>
  <p class="hint">Builds MCP_SERVERS. A stdio server is a command this machine starts; it receives only the environment listed here, never the app's own secrets. An HTTP server is a Streamable HTTP endpoint. Write a secret as ${NAME} to read it from the app's environment when it connects instead of storing it in the setting.</p>
  <div class="stack" id="integMcpRows"></div>
  <p class="hint error" id="integMcpProblems" hidden></p>
  <p class="hint warn" id="integMcpMasked" hidden>Values shown as *** are secrets the server masks when it shows settings. Replace each with the real value or a ${NAME} reference before using the result, or the server receives a literal ***.</p>
  <div class="codeblock"><div class="codebar"><span>.env</span><button type="button" class="btn btn-sm" onclick="integMcpCopy(this)">Copy</button></div><pre id="integMcpJson"></pre></div>
  <div class="row">
    <span class="badge" id="integMcpRestart">restart</span>
    <span class="hint grow" id="integMcpHow">MCP_SERVERS is read when the app starts: replace the MCP_SERVERS line in .env with this one and restart the app.</span>
    <button type="button" class="btn btn-sm btn-primary" id="integMcpApply" onclick="integMcpApply()" hidden>Apply</button>
  </div>
</div>
"""

_UIS_INTEG_AUDIT = r"""
<div class="panel-block card admin-only" data-panel="audit">
  <div class="card-header"><h3 class="h3">Tool decisions</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm" onclick="integLoadAudit(true)" title="Reload from the newest">Refresh</button>
    </div>
  </div>
  <p class="hint">Every tool call the approval policy did not silently allow: who, which tool, the decision, the rule or person behind it and how long the answer took. Arguments are stored redacted. Rows are kept when a conversation is deleted.</p>
  <div class="row wrap">
    <select class="input select" id="integAudUser" aria-label="User"><option value="">All users</option></select>
    <input class="input" type="search" id="integAudTool" placeholder="Tool name" aria-label="Tool name" autocomplete="off" spellcheck="false">
    <select class="input select" id="integAudDecision" aria-label="Decision">
      <option value="">Any decision</option><option value="allow">Allowed</option><option value="ask">Asked</option><option value="deny">Denied</option><option value="timeout">Timed out</option><option value="cancelled">Cancelled</option>
    </select>
    <input class="input" type="date" id="integAudFrom" aria-label="From date">
    <input class="input" type="date" id="integAudTo" aria-label="To date">
  </div>
  <div id="integAudList"><p class="muted">loading...</p></div>
  <div class="row"><span class="hint" id="integAudCount"></span><span class="spacer"></span><button type="button" class="btn btn-sm" id="integAudMore" onclick="integLoadAudit(false)" hidden>Older</button></div>
</div>
"""

_UIS_INTEG_PROJECTS = r"""
<div class="panel-block card" data-panel="projects">
  <div class="card-header"><h3 class="h3">Projects</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm" onclick="integLoadProjects()" title="Refresh projects">Refresh</button>
    </div>
  </div>
  <p class="hint">A project is a folder the agent works in, with its own context files (AGENTS.md and the like) and knowledge scope. Share it with people by adding them as members.</p>
  <p class="hint warn" id="integProjOff" hidden>Projects are off on this server. An administrator turns them on with Per-user workspaces (WORKSPACES_ENABLED) under Tools, then restarts the app.</p>
  <div class="stack" id="integProjBody">
    <p class="hint warn admin-only" id="integProjWarn" hidden></p>
    <div class="row wrap">
      <input class="input grow" type="text" id="integProjName" maxlength="80" placeholder="Project name" aria-label="Project name" autocomplete="off" spellcheck="false">
      <input class="input grow mono" type="text" id="integProjRoot" placeholder="Folder (optional)" aria-label="Project folder" autocomplete="off" spellcheck="false">
      <span class="hint admin-only" aria-hidden="true">Shared</span>
      <label class="switch admin-only" id="integProjSharedWrap" title="Open to every signed-in user"><input type="checkbox" id="integProjShared" aria-label="Open to every signed-in user"><span class="slider"></span></label>
      <button type="button" class="btn btn-sm btn-primary" onclick="integCreateProject()">Create</button>
    </div>
    <p class="hint" id="integProjRootHint"></p>
    <div id="integProjList"><p class="muted">loading...</p></div>
    <div class="stack tight" id="integProjDetail" hidden>
      <div class="row"><h3 class="h3 grow" id="integProjDetailTitle"></h3><button type="button" class="btn btn-sm btn-ghost" onclick="integCloseProject()">Close</button></div>
      <dl class="kv" id="integProjFacts"></dl>
      <span class="field-label">Members</span>
      <div id="integProjMembers"></div>
      <div class="row wrap" id="integProjAddRow">
        <input class="input grow" type="text" id="integProjMember" placeholder="Username" aria-label="Username to add" autocomplete="off" spellcheck="false">
        <select class="input select" id="integProjMemberRole" aria-label="Member role"><option value="member">Member</option><option value="manager">Manager</option></select>
        <button type="button" class="btn btn-sm" onclick="integAddMember()">Add member</button>
      </div>
      <span class="field-label">Context the agent gets</span>
      <pre class="logbox mono" id="integProjContext"></pre>
    </div>
  </div>
</div>
"""

# (section, sub) -> (markup before the schema groups, markup after them).
_UIS_INTEG_PANELS: dict[tuple[str, str], tuple[str, str]] = {
    ("tools", "clients"): (_UIS_INTEG_CLIENTS, ""),
    ("tools", "tokens"): (_UIS_INTEG_TOKENS, ""),
    ("tools", "approvals"): (_UIS_INTEG_APPROVALS, ""),
    ("tools", "mcp"): (_UIS_INTEG_MCPEDIT, ""),
    ("admin", "audit"): (_UIS_INTEG_AUDIT, ""),
    ("agent", ""): ("", _UIS_INTEG_PROJECTS),
}


def _uis_integ_merge() -> None:
    """Append the integrations cards to _UIS_PANELS without rewriting its entries."""
    for key, (before, after) in _UIS_INTEG_PANELS.items():
        old_before, old_after = _UIS_PANELS.get(key, ("", ""))
        _UIS_PANELS[key] = (old_before.strip() + before.strip(), old_after.strip() + after.strip())


_uis_integ_merge()
# ---- panels-integrations end ----------------------------------------------- #


# ---- dashboards (Wave 4a: item 14) ------------------------------------------ #
# Admin > Dashboards: KPI tiles, charts per window, breakdown tables and the
# recent warnings from the structured log feed (dashboards.py), then the
# adapter ratings and the blind A/B voting card (elo.py). Loaders live in
# ui_script_dashboards.py (dashTabHook wires them into ADMIN_TABS). Every route
# behind them is admin-only on the server except the vote and next-pair routes,
# which any signed-in user may call; the section is admin-only in the rail.
_UIS_DASH_PANEL = r"""
<div class="panel-block card" data-panel="dashboards">
  <div class="card-header"><h3 class="h3">Overview</h3>
    <div class="card-actions">
      <div class="seg" role="radiogroup" id="dashWindow" aria-label="Time window">
        <button type="button" class="btn-sm" data-value="1h" aria-pressed="false">1 hour</button>
        <button type="button" class="btn-sm active" data-value="24h" aria-pressed="true">24 hours</button>
        <button type="button" class="btn-sm" data-value="7d" aria-pressed="false">7 days</button>
      </div>
      <button type="button" class="btn btn-sm" onclick="dashLoad()" title="Refresh the dashboards">Refresh</button>
    </div>
  </div>
  <p class="hint" id="dashRange">Requests, latency and tokens from the metrics this server records for every request.</p>
  <div class="dash-kpis" id="dashKpis"><span class="muted">loading...</span></div>
</div>
<div class="panel-block card" data-panel="dashcharts">
  <div class="card-header"><h3 class="h3">Over time</h3></div>
  <div class="dash-charts">
    <div class="dash-chart"><span class="field-label">Requests and errors</span><div id="dashChartRequests"></div></div>
    <div class="dash-chart"><span class="field-label">Latency</span><div id="dashChartLatency"></div></div>
    <div class="dash-chart"><span class="field-label">Time to first token</span><div id="dashChartTtft"></div></div>
    <div class="dash-chart"><span class="field-label">Decode speed</span><div id="dashChartTps"></div></div>
    <div class="dash-chart"><span class="field-label">Tokens</span><div id="dashChartTokens"></div></div>
    <div class="dash-chart"><span class="field-label">Tool calls</span><div id="dashChartTools"></div></div>
    <div class="dash-chart"><span class="field-label">Prompt cache hits</span><div id="dashChartCache"></div></div>
    <div class="dash-chart"><span class="field-label">Queue wait</span><div id="dashChartQueue"></div></div>
  </div>
  <p class="hint">Latency counts successful requests only. Cache hits are shown only for engines that report cached prompt tokens.</p>
</div>
<div class="panel-block card" data-panel="dashbreakdown">
  <div class="card-header"><h3 class="h3">Breakdown</h3></div>
  <span class="field-label">Per user</span>
  <div class="ops-scroll"><table class="table"><thead><tr><th>User</th><th>Requests</th><th>Errors</th><th>Latency p50</th><th>Tokens in / out</th></tr></thead><tbody id="dashUsers"></tbody></table></div>
  <span class="field-label">Per model, adapter and engine</span>
  <div class="ops-scroll"><table class="table"><thead><tr><th>Model</th><th>Adapter</th><th>Engine</th><th>Generations</th><th>TTFT p50</th><th>Decode</th></tr></thead><tbody id="dashModels"></tbody></table></div>
  <span class="field-label">Errors by type</span>
  <div id="dashErrors"></div>
  <span class="field-label">Per node</span>
  <div class="ops-scroll"><table class="table"><thead><tr><th>Node</th><th>Requests</th><th>Failed</th><th>p50</th><th>p95</th></tr></thead><tbody id="dashNodes"></tbody></table></div>
  <span class="field-label">Tools</span>
  <div class="ops-scroll"><table class="table"><thead><tr><th>Tool</th><th>Calls</th><th>Error rate</th><th>Average</th><th>p95</th></tr></thead><tbody id="dashTools"></tbody></table></div>
  <span class="field-label">Approvals</span>
  <dl class="kv" id="dashApprovals"></dl>
  <span class="field-label">Feedback by adapter</span>
  <div class="ops-scroll"><table class="table"><thead><tr><th>Adapter</th><th>Ratings</th><th>Up</th><th>Down</th><th>Up rate</th></tr></thead><tbody id="dashFeedback"></tbody></table></div>
</div>
<div class="panel-block card" data-panel="dashevents">
  <div class="card-header"><h3 class="h3">Events</h3>
    <div class="card-actions">
      <div class="seg" role="radiogroup" id="dashEventLevel" aria-label="Which events">
        <button type="button" class="btn-sm active" data-value="WARNING" aria-pressed="true">Warnings and errors</button>
        <button type="button" class="btn-sm" data-value="" aria-pressed="false">All</button>
      </div>
    </div>
  </div>
  <p class="hint" id="dashEventSince">Structured log events this process has seen (kept in memory, whatever LOG_FORMAT says; secrets are redacted).</p>
  <div id="dashEventCounts"></div>
  <div class="ops-scroll"><table class="table"><thead><tr><th>Time</th><th>Level</th><th>Event</th><th>Message</th></tr></thead><tbody id="dashEvents"></tbody></table></div>
</div>
<div class="panel-block card" data-panel="adapterratings">
  <div class="card-header"><h3 class="h3">Adapter ratings</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm" onclick="dashRecompute()" title="Rebuild the ratings from every stored match">Recompute</button>
    </div>
  </div>
  <p class="hint">ELO ratings (K=24, everyone starts at 1500) from pairwise comparisons on the same prompt: your votes below, opposite thumbs on one prompt across adapters, and tournaments. Fewer than 10 games is provisional.</p>
  <div class="ops-scroll"><table class="table"><thead><tr><th>Rank</th><th>Adapter</th><th>ELO</th><th>Games</th><th>W / L / T</th><th>Win rate</th></tr></thead><tbody id="dashRatings"></tbody></table></div>
  <p class="hint" id="dashRatingsNote"></p>
  <span class="field-label">Tournament</span>
  <div class="row wrap">
    <div class="seg" role="radiogroup" id="dashJudge" aria-label="Who judges">
      <button type="button" class="btn-sm active" data-value="model" aria-pressed="true">Model judges</button>
      <button type="button" class="btn-sm" data-value="human" aria-pressed="false">People vote</button>
    </div>
    <button type="button" class="btn btn-sm btn-primary" id="dashTourneyBtn" onclick="dashStartTournament()" title="Answer a fixed prompt set with every adapter and compare">Run tournament</button>
    <button type="button" class="btn btn-sm btn-ghost" id="dashTourneyCancel" onclick="dashCancelTournament()" hidden>Cancel</button>
  </div>
  <div class="row nowrap" id="dashTourneyBox" hidden><span class="pill running" id="dashTourneyPill">running</span><span class="hint mono grow ellipsis" id="dashTourneyText"></span></div>
  <p class="hint">A tournament loads each adapter in turn on the shared model server, so chats get "Model not ready" until it ends. It refuses to start while anything is generating.</p>
  <span class="field-label">Recent matches</span>
  <div class="ops-scroll"><table class="table"><thead><tr><th>When</th><th>A</th><th>B</th><th>Winner</th><th>Judge</th><th>Source</th></tr></thead><tbody id="dashMatches"></tbody></table></div>
</div>
<div class="panel-block card" data-panel="abvote">
  <div class="card-header"><h3 class="h3">Compare answers</h3>
    <div class="card-actions">
      <span class="pill muted" id="dashVoteLeft">0 waiting</span>
      <button type="button" class="btn btn-sm" onclick="dashLoadPair()" title="Load the next comparison">Next</button>
    </div>
  </div>
  <p class="hint">Two adapters answered the same prompt. Pick the better answer; which adapter wrote which is shown after you vote.</p>
  <p class="empty" id="dashVoteEmpty">Nothing to compare. Run a tournament with "People vote" to prepare pairs.</p>
  <div class="stack tight" id="dashVoteBox" hidden>
    <span class="field-label">Prompt</span>
    <div class="readonly" id="dashVotePrompt"></div>
    <div class="dash-ab">
      <div class="stack tight"><span class="field-label">Answer A</span><div class="dash-answer" id="dashVoteA"></div></div>
      <div class="stack tight"><span class="field-label">Answer B</span><div class="dash-answer" id="dashVoteB"></div></div>
    </div>
    <div class="row wrap center">
      <button type="button" class="btn" onclick="dashVote('a')">A is better</button>
      <button type="button" class="btn" onclick="dashVote('tie')">About the same</button>
      <button type="button" class="btn" onclick="dashVote('b')">B is better</button>
    </div>
  </div>
</div>
"""


def _uis_dash_merge() -> None:
    """Put the dashboard cards in Admin > Dashboards without rewriting _UIS_PANELS."""
    old_before, old_after = _UIS_PANELS.get(("admin", "dashboards"), ("", ""))
    _UIS_PANELS[("admin", "dashboards")] = (old_before.strip() + _UIS_DASH_PANEL.strip(),
                                            old_after.strip())


_uis_dash_merge()
# ---- dashboards end --------------------------------------------------------- #


# ---- evals (Wave 4b: item 19) ----------------------------------------------- #
# Admin > Evals: start a run of the evaluation suites (scripted or live), follow
# it, compare its per-suite scores with the baseline of the model and adapter it
# ran against, keep a run as the new baseline, and read the failed cases. The
# regression-gate settings (EVAL_ON_PROMOTE, EVAL_MAX_REGRESSION) render below
# these cards. Loaders live in ui_script_evals.py (evalsTabHook wires them into
# ADMIN_TABS); every route behind them is admin-only.
_UIS_EVALS_PANEL = r"""
<div class="panel-block card" data-panel="evals">
  <div class="card-header"><h3 class="h3">Run evaluations</h3>
    <div class="card-actions">
      <div class="seg" role="radiogroup" id="evalsMode" aria-label="Which model answers">
        <button type="button" class="btn-sm active" data-value="scripted" aria-pressed="true">Scripted</button>
        <button type="button" class="btn-sm" data-value="live" aria-pressed="false">Live model</button>
      </div>
      <button type="button" class="btn btn-sm btn-primary" id="evalsRunBtn" onclick="evalsRun()" title="Run the selected suites">Run</button>
      <button type="button" class="btn btn-sm btn-ghost" id="evalsCancelBtn" onclick="evalsCancel()" hidden>Cancel</button>
    </div>
  </div>
  <p class="hint">Scripted replays canned model answers: it checks the harness, the parsers and the API, and needs no model. Live sends the same prompts to the loaded model and adapter; it refuses while anything else is generating.</p>
  <div class="field-row"><span class="field-label" id="evalsSuitesLabel">Suites</span>
    <div class="field-control cap-grid" id="evalsSuites" role="group" aria-labelledby="evalsSuitesLabel">
      <label class="check"><input type="checkbox" value="tool_calling" checked><span>Tool calling: picks the right tool with valid arguments</span></label>
      <label class="check"><input type="checkbox" value="rag" checked><span>Knowledge base: answers from the documents and cites them</span></label>
      <label class="check"><input type="checkbox" value="multi_turn" checked><span>Multi-turn: edits, continue after a cut-off, no language drift</span></label>
      <label class="check"><input type="checkbox" value="routing" checked><span>Routing: each message gets the right intent, lane, tools and retrieval</span></label>
      <label class="check"><input type="checkbox" value="parsers" checked><span>Parsers: every model family's tool-call format (no model)</span></label>
      <label class="check"><input type="checkbox" value="api" checked><span>API: /v1 OpenAI and Anthropic response shapes (no model)</span></label>
    </div>
  </div>
  <div class="field-row"><label class="field-label" for="evalsSaveBaseline">Save as baseline</label>
    <div class="field-control"><label class="switch"><input type="checkbox" id="evalsSaveBaseline" aria-label="Save the run as the baseline"><span class="slider"></span></label><span class="hint">Keep this run as the reference for its model and adapter.</span></div>
  </div>
  <div class="row nowrap" id="evalsJobBox" hidden><span class="pill running" id="evalsJobPill">running</span><span class="hint mono grow ellipsis" id="evalsJobText"></span></div>
  <dl class="kv" id="evalsActive"></dl>
</div>
<div class="panel-block card" data-panel="evalscores">
  <div class="card-header"><h3 class="h3">Scores against the baseline</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm" id="evalsBaselineBtn" onclick="evalsSaveBaseline()" title="Keep the selected run as the baseline for its model and adapter" disabled>Make baseline</button>
    </div>
  </div>
  <p class="hint" id="evalsSummary">No run yet.</p>
  <div id="evalsChart"></div>
  <div class="ops-scroll"><table class="table"><thead><tr><th>Suite</th><th>Score</th><th>Passed</th><th>Baseline</th><th>Change</th><th>Latency</th><th>Tokens</th></tr></thead><tbody id="evalsCompare"></tbody></table></div>
</div>
<div class="panel-block card" data-panel="evalruns">
  <div class="card-header"><h3 class="h3">Runs</h3>
    <div class="card-actions">
      <button type="button" class="btn btn-sm" onclick="evalsLoad()" title="Refresh the runs">Refresh</button>
    </div>
  </div>
  <div class="ops-scroll"><table class="table"><thead><tr><th>When</th><th>Mode</th><th>Model and adapter</th><th>Score</th><th>Passed</th><th>Result</th><th>Trigger</th><th></th></tr></thead><tbody id="evalsRuns"></tbody></table></div>
</div>
<div class="panel-block card" data-panel="evalcases">
  <div class="card-header"><h3 class="h3" id="evalsCasesTitle">Cases</h3>
    <div class="card-actions">
      <div class="seg" role="radiogroup" id="evalsCaseFilter" aria-label="Which cases">
        <button type="button" class="btn-sm active" data-value="failed" aria-pressed="true">Failed</button>
        <button type="button" class="btn-sm" data-value="all" aria-pressed="false">All</button>
      </div>
    </div>
  </div>
  <div class="ops-scroll"><table class="table"><thead><tr><th>Suite</th><th>Case</th><th>Score</th><th>Failed checks</th><th>Latency</th><th>Tokens</th></tr></thead><tbody id="evalsCases"></tbody></table></div>
</div>
"""


def _uis_evals_merge() -> None:
    """Put the evaluation cards in Admin > Evals, above its settings."""
    old_before, old_after = _UIS_PANELS.get(("admin", "evals"), ("", ""))
    _UIS_PANELS[("admin", "evals")] = (old_before.strip() + _UIS_EVALS_PANEL.strip(),
                                       old_after.strip())


_uis_evals_merge()
# ---- evals end -------------------------------------------------------------- #


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
        # A sub-tab flagged admin (Tools > MCP) is gated like admin-only
        # sections: cosmetic here, its routes are ADMIN on the server.
        buttons = "".join(
            f'<button type="button" class="btn-sm{" active" if i == 0 else ""}'
            f'{" admin-only" if t.get("admin") else ""}"'
            f' data-sub="{t["key"]}" role="tab"'
            f' aria-pressed="{"true" if i == 0 else "false"}">{_uis_esc(t["label"])}</button>'
            for i, t in enumerate(tabs))
        parts.append(f'<div class="seg seg-tabs" role="tablist" data-subtabs="{key}"'
                     f' aria-label="{_uis_esc(sec["label"])} sections">{buttons}</div>')
        for i, t in enumerate(tabs):
            sub_fields = [f for f in fields if settings_field_sub(f) == t["key"]]
            hidden = "" if i == 0 else " hidden"
            hidden += " admin-only" if t.get("admin") else ""
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
