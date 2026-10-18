"""Settings metadata: the single description of every option the UI shows.

Settings used to be hand-wired three times over: markup per field, one line in
loadConfig(), one line in saveConfig(). The panel exposed 22 of the 138
runtime-mutable config keys, two controls wrote the same key (and one of them
silently lost its edits), and none of them knew a range the server did not
already enforce somewhere else. This module is now the one place that says, for
every option: which section and group it lives in, how it is labelled, which
control renders it, the bounds a slider shows, whether it needs a restart and
whether it is the user's own preference or an installation-wide setting.

Three scopes:

  * "global" -- a Config field in Config.MUTABLE. Admin only; read from
    GET /api/config and written through POST /api/config, whose clamps stay the
    authority (min/max/step here are display hints, never validation).
  * "user"   -- a per-user preference stored server-side (user_preferences
    table, GET/POST /api/me/preferences). Validated by USER_PREFS below.
  * "info"   -- read-only facts (restart-only levers, the search provider) shown
    so the panel tells the truth about the running process.

The markup is rendered from this module once, at import time, by ui_settings.py,
so the page is still one static string and every control id is a literal the
selftest can check.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------- #
# Sections, in the order the Settings rail shows them. `admin` sections are
# hidden from non-admins in the UI and filtered out of the schema the API
# returns to them.
# --------------------------------------------------------------------------- #
SETTINGS_SECTIONS: list[dict[str, Any]] = [
    {"key": "appearance", "label": "Appearance", "admin": False,
     "description": "Theme, density and how answers are displayed. Changes apply "
                    "and save immediately, no Apply needed."},
    {"key": "generation", "label": "Generation", "admin": False,
     "description": "System prompt, reply length, temperature and context."},
    {"key": "agent", "label": "Agent", "admin": False,
     "description": "When the assistant uses tools and how hard it thinks."},
    {"key": "models", "label": "Models & Training", "admin": True,
     "description": "The served model, adapters, LoRA training and the knowledge base."},
    {"key": "performance", "label": "Performance", "admin": True,
     "description": "Throughput, timeouts, prefix reuse and admission control."},
    {"key": "memory", "label": "Memory", "admin": False,
     "description": "Notes the assistant keeps, retrieval and learned skills."},
    {"key": "tools", "label": "Tools", "admin": False,
     "description": "The tool catalogue, sandboxed execution and integrations."},
    {"key": "admin", "label": "Admin", "admin": True,
     "description": "Users, agents, cluster, imports, codebase, backup and logs."},
]

SECTION_KEYS = [s["key"] for s in SETTINGS_SECTIONS]

# Widget kinds the renderer understands. Keep this list closed: a new kind is a
# new control in ui_settings.py and ui_script_settings.py, not a free string.
WIDGETS = ("switch", "range", "number", "select", "segmented", "text",
           "textarea", "secret", "readonly")

# --------------------------------------------------------------------------- #
# Per-user preferences. Every key here is validated server-side by
# validate_user_prefs(); the UI renders them from SETTINGS_FIELDS entries with
# scope "user". `None` as a default means "use the installation default".
# --------------------------------------------------------------------------- #
USER_PREFS: dict[str, dict[str, Any]] = {
    "theme": {"type": "enum", "options": ["system", "light", "dark"], "default": "system"},
    "density": {"type": "enum", "options": ["comfortable", "compact"], "default": "comfortable"},
    "text_size": {"type": "enum", "options": ["small", "medium", "large"], "default": "medium"},
    "reasoning_display": {"type": "enum", "options": ["collapsed", "expanded", "hidden"],
                          "default": "collapsed"},
    "code_wrap": {"type": "bool", "default": False},
    "enter_to_send": {"type": "bool", "default": True},
    "show_metrics": {"type": "bool", "default": True},
    "default_mode": {"type": "enum", "options": ["auto", "chat", "agent"], "default": "auto"},
    # Generation overrides applied to this user's requests when the request
    # itself does not carry a value. None = the installation default.
    "temperature": {"type": "float", "min": 0.0, "max": 2.0, "default": None, "nullable": True},
    "max_tokens": {"type": "int", "min": 16, "max": 32768, "default": None, "nullable": True},
    # ---- extension user prefs: above this line ---- #
}


def validate_user_prefs(updates: dict) -> tuple[dict, list[str]]:
    """Coerce and bound a preferences patch. Returns (accepted, ignored_keys).

    Unknown keys and values of the wrong shape are ignored, never guessed at:
    a preference the server cannot interpret must not become a stored value
    that every later request trips over.
    """
    accepted: dict[str, Any] = {}
    ignored: list[str] = []
    for key, value in (updates or {}).items():
        spec = USER_PREFS.get(key)
        if spec is None:
            ignored.append(key)
            continue
        if value is None:
            if spec.get("nullable") or spec.get("default") is None:
                accepted[key] = None
            else:
                ignored.append(key)
            continue
        kind = spec["type"]
        try:
            if kind == "enum":
                if str(value) not in spec["options"]:
                    raise ValueError(value)
                accepted[key] = str(value)
            elif kind == "bool":
                if not isinstance(value, bool):
                    raise ValueError(value)
                accepted[key] = value
            elif kind == "int":
                if isinstance(value, bool):
                    raise ValueError(value)
                number = int(value)
                accepted[key] = min(spec["max"], max(spec["min"], number))
            elif kind == "float":
                if isinstance(value, bool):
                    raise ValueError(value)
                number = float(value)
                accepted[key] = min(spec["max"], max(spec["min"], number))
            elif kind == "str":
                text = str(value)[: int(spec.get("max_len", 200))]
                accepted[key] = text
            else:
                raise ValueError(kind)
        except (TypeError, ValueError):
            ignored.append(key)
    return accepted, ignored


def default_user_prefs() -> dict[str, Any]:
    return {key: spec.get("default") for key, spec in USER_PREFS.items()}


# --------------------------------------------------------------------------- #
# Fields. One dict per option:
#   key       Config field name (scope global) or USER_PREFS key (scope user)
#   scope     "global" | "user" | "info"
#   section   one of SECTION_KEYS
#   group     group title inside the section (rendered as a collapsible group)
#   label     short label, sentence case
#   help      one or two sentences: what it does and the trade-off
#   widget    one of WIDGETS
#   min/max/step/unit   for range/number (display hints only)
#   options   list of {"value", "label"} for select/segmented
#   restart   True when the running process only adopts it after a restart
#   advanced  True to collapse it under the group's "Advanced" disclosure
# Optional keys the renderer also understands:
#   sub         sub-tab inside a section that has them (SETTINGS_SUBTABS);
#               missing or unknown lands in that section's first sub-tab
#   type        "int" | "float" | "bool" | "str", only when settings_field_type()
#               would guess wrong (a float setting with whole-number bounds)
#   format      "percent" (a 0..1 ratio shown as %) or "bytes" (shown as MB/GB)
#   placeholder text shown in an empty text or number box
#   rows        textarea height in lines
#   autosave    user prefs applied and saved the moment they change (Appearance)
#   env         environment variable behind an "info" field, named in its help
#   id          control id override (the theme switch keeps the legacy themeBtn)
#
# Bounds are copied from the clamps in Config.apply() and tests/test_ui_settings
# applies out-of-range values to prove they still agree. A bound the clamp does
# not enforce is left out rather than invented, so the slider never promises a
# limit the server would not hold.
# --------------------------------------------------------------------------- #


def _ss(key: str, section: str, group: str, label: str, text: str, widget: str,
        **extra: Any) -> dict[str, Any]:
    """One SETTINGS_FIELDS entry. scope defaults to "global" (a Config field)."""
    entry: dict[str, Any] = {"key": key, "scope": extra.pop("scope", "global"),
                             "section": section, "group": group, "label": label,
                             "help": text, "widget": widget}
    entry.update(extra)
    return entry


def _ss_opts(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"value": value, "label": label} for value, label in pairs]


def _ss_info(key: str, section: str, group: str, label: str, text: str,
             env: str, **extra: Any) -> dict[str, Any]:
    """A read-only fact about the running process (restart-only levers)."""
    return _ss(key, section, group, label, f"{text} Set {env} and restart to change it.",
               "readonly", scope="info", env=env, restart=True, **extra)


SETTINGS_FIELDS: list[dict[str, Any]] = [
    # ================================================================ #
    # Appearance: per-user, applied live and saved on change.
    # ================================================================ #
    _ss("theme", "appearance", "Theme", "Colour scheme",
        "System follows your operating system's light or dark mode. Applied "
        "immediately and remembered for your account.",
        "segmented", scope="user", autosave=True, id="themeBtn",
        options=_ss_opts(("system", "System"), ("light", "Light"), ("dark", "Dark"))),
    _ss("density", "appearance", "Layout", "Density",
        "Compact tightens spacing and control heights so more fits on screen.",
        "segmented", scope="user", autosave=True,
        options=_ss_opts(("comfortable", "Comfortable"), ("compact", "Compact"))),
    _ss("text_size", "appearance", "Layout", "Text size",
        "Scales every text size in the interface together.",
        "segmented", scope="user", autosave=True,
        options=_ss_opts(("small", "Small"), ("medium", "Medium"), ("large", "Large"))),
    _ss("reasoning_display", "appearance", "Chat", "Reasoning display",
        "How visible reasoning appears in an answer's trace: folded until you open "
        "it, open while it streams, or not shown. This only changes the display; "
        "whether the model reasons is set under Generation.",
        "segmented", scope="user", autosave=True,
        options=_ss_opts(("collapsed", "Collapsed"), ("expanded", "Expanded"),
                         ("hidden", "Hidden"))),
    _ss("code_wrap", "appearance", "Chat", "Wrap long code lines",
        "Wrap code blocks at the edge of the column instead of scrolling them sideways.",
        "switch", scope="user", autosave=True),
    _ss("show_metrics", "appearance", "Chat", "Show answer metrics",
        "Show tokens, time and speed under each answer.",
        "switch", scope="user", autosave=True),
    _ss("enter_to_send", "appearance", "Chat", "Enter sends the message",
        "When off, Enter inserts a new line and the Send button sends.",
        "switch", scope="user", autosave=True),

    # ================================================================ #
    # Generation
    # ================================================================ #
    _ss("temperature", "generation", "My defaults", "My temperature",
        "Used for your messages when a request does not set its own. Leave it empty "
        "to follow the installation default.",
        "number", scope="user", min=0.0, max=2.0, step=0.05, nullable=True,
        placeholder="Installation default"),
    _ss("max_tokens", "generation", "My defaults", "My max reply tokens",
        "Longest reply for your messages when a request does not set one, still "
        "capped by the context size. Leave it empty to follow the installation default.",
        "number", scope="user", min=16, max=32768, step=16, unit="tokens", nullable=True,
        placeholder="Installation default"),

    _ss("system_prompt", "generation", "Prompt", "System prompt",
        "The instructions every reply starts from, placed after the identity line. "
        "A change invalidates any cached prompt prefix, and the live adapter was "
        "trained against the old prompt, so retrain after a large rewrite.",
        "textarea", rows=6),
    _ss("identity", "generation", "Prompt", "Identity line",
        "Put in front of the system prompt so the model states who it is instead of "
        "the base model's pretrained identity. Leave it empty to drop the line.",
        "textarea", rows=3),
    _ss("model_cutoff", "generation", "Prompt", "Model training cutoff",
        "Your model's training cutoff as YYYY-MM. A question naming a later year "
        "needs a lookup because the weights cannot hold the answer. Leave it empty "
        "if unknown: any recent year then counts as needing one. A malformed value "
        "is refused, not half-parsed.",
        "text", placeholder="YYYY-MM"),
    _ss("date_in_prompt", "generation", "Prompt", "Tell the model today's date",
        "Put today's date, and the cutoff above, in the system prompt. Without it the "
        "model has no anchor for judging whether its own knowledge is still current.",
        "switch"),

    _ss("max_tokens", "generation", "Reply", "Max reply tokens",
        "Longest reply the model may generate. Higher allows longer answers and costs "
        "memory and time; a value at or above the context size is cut to half of it.",
        "number", min=16, max=32768, step=16, unit="tokens"),
    _ss("temperature", "generation", "Reply", "Temperature",
        "Randomness of the final answer. 0 is deterministic and focused, higher is "
        "more varied. Tool-choosing steps use the tool temperature instead.",
        "range", min=0.0, max=2.0, step=0.05),
    _ss("code_max_tokens", "generation", "Reply", "Max reply tokens for code",
        "Reply budget when the request is for code, so a program is not cut off "
        "mid-function at the general limit. Still bounded by the context window.",
        "number", min=256, max=32768, step=64, unit="tokens"),
    _ss("auto_continue", "generation", "Reply", "Finish cut-off answers",
        "When a reply stops at the token limit, continue it automatically instead of "
        "waiting for a 'go on'. Every continuation re-prefills the prompt.",
        "switch"),
    _ss("auto_continue_max", "generation", "Reply", "Continuations per answer",
        "How many continuation passes one turn may spend. Each is a full generation, "
        "so this is a wall-clock budget as much as a quality one. 0 disables it.",
        "range", min=0, max=6, step=1),

    _ss("context_size", "generation", "Context", "Context size",
        "Working window this process enforces for prompt plus reply, applied on the "
        "next message. Larger holds more history and uses more memory; the model "
        "server's own KV cap only changes on restart (Models & Training).",
        "number", min=512, max=131072, step=512, unit="tokens"),
    _ss("history_turns", "generation", "Context", "History turns",
        "Past messages carried into each request. Fewer means less memory and less "
        "continuity; the oldest turns drop out automatically when the window is full.",
        "range", min=0, max=200, step=1),

    _ss("reasoning_visible", "generation", "Reasoning", "Reason out loud",
        "Ask the model to reason step by step before answering and show that "
        "reasoning in the trace. Costs tokens on every turn; turn it off for speed.",
        "switch"),
    _ss("disable_thinking", "generation", "Reasoning", "Skip the thinking phase",
        "Ask reasoning models (Qwen3.5 and later) not to emit their hidden <think> "
        "block. Faster and lighter on the KV cache; hard questions may suffer.",
        "switch"),

    _ss("repetition_penalty_enabled", "generation", "Repetition", "Penalise repetition",
        "Off by default: some mlx-lm builds crash the generation thread when the "
        "repetition fields are sent. Turn it on once your build is fixed.",
        "switch"),
    _ss("repetition_penalty", "generation", "Repetition", "Repetition penalty",
        "Multiplier against tokens already in the window, which breaks the verbatim "
        "loops small quantised models fall into. 1.0 disables it.",
        "number", step=0.05, advanced=True),
    _ss("repetition_context_size", "generation", "Repetition", "Repetition window",
        "How many recent tokens the penalty looks back over.",
        "number", step=1, unit="tokens", advanced=True),

    # ================================================================ #
    # Agent
    # ================================================================ #
    _ss("default_mode", "agent", "Mode", "Default mode",
        "Where the composer's mode switch starts. Auto lets the server decide per "
        "message, Chat answers in one reply, Agent may call tools and reason in steps.",
        "segmented", scope="user",
        options=_ss_opts(("auto", "Auto"), ("chat", "Chat"), ("agent", "Agent"))),
    _ss("agent_enabled", "agent", "Mode", "Auto uses tools",
        "What Auto means on this installation. On, Auto turns may call tools and "
        "reason in steps; off, Auto answers in one reply unless triage or escalation "
        "decides a lookup is needed. Chat and Agent override it per message.",
        "switch"),

    _ss("agent_max_steps", "agent", "Steps", "Max steps",
        "Hard ceiling on steps in one turn. The agent stops there and salvages what "
        "it has.",
        "range", min=1, max=20, step=1),
    _ss("agent_min_steps", "agent", "Steps", "Min steps",
        "Steps the agent takes before it may settle. At 2 or more the first reply is "
        "reviewed as a draft, which catches hasty answers and costs one generation "
        "per extra step. Never above Max steps.",
        "range", min=1, max=20, step=1),
    _ss("hard_step_cap", "agent", "Steps", "Absolute step cap",
        "Lets a long task keep going past Max steps by compacting its progress into a "
        "running summary, so it slows down instead of stopping. Never below Max steps.",
        "range", min=1, max=60, step=1),

    _ss("knowledge_triage", "agent", "Routing", "Knowledge triage",
        "Before a substantive answer, ask the model in one word whether it needs a "
        "lookup and seed the search when it does. Biased toward searching, since a "
        "needless search is cheaper than a confident wrong answer.",
        "switch"),
    _ss("agent_escalation", "agent", "Routing", "Escalate to tools when needed",
        "Let a tool-free turn switch into agent mode once the model reports it "
        "lacked a capability, at most once per turn.",
        "switch"),
    _ss("hedge_escalation", "agent", "Routing", "Look up hedged answers",
        "When an answer hedges about external facts, look it up once and answer "
        "again. Costs a search and a second generation on a turn that looked "
        "finished; never fires against /no-search.",
        "switch"),
    _ss("fast_path", "agent", "Routing", "Deterministic fast path",
        "Answer arithmetic and bare URLs without a model round trip.",
        "switch"),

    _ss("incremental_reasoning", "agent", "Deep reasoning", "Decompose hard questions",
        "Break a hard question with no tool to call into a plan, solve it one step at "
        "a time and synthesise. The most expensive path here: one model call per step.",
        "switch"),
    _ss("reasoning_signals", "agent", "Deep reasoning", "Cues needed",
        "Analytical cues (compare, trade-offs, why does) that make a question worth "
        "decomposing. 1 decomposes anything with a single cue, which is much slower "
        "and more thorough.",
        "range", min=1, max=6, step=1),
    _ss("reasoning_min_chars", "agent", "Deep reasoning", "Single-cue length",
        "Question length at which one cue is enough on its own. Lower it to "
        "decompose shorter questions.",
        "number", min=0, max=2000, step=10, unit="chars"),
    _ss("reasoning_max_steps", "agent", "Deep reasoning", "Plan steps",
        "Most sub-steps a decomposed question is split into.",
        "range", min=2, max=10, step=1),
    _ss("reasoning_tokens", "agent", "Deep reasoning", "Tokens per reasoning step",
        "Generation budget for each reasoning, chunk and source-extraction pass. "
        "Larger passes think more and hold more memory.",
        "number", min=64, max=2048, step=64, unit="tokens"),

    _ss("delegation_enabled", "agent", "Delegation", "Allow delegation",
        "Let the agent hand a self-contained sub-task to a helper with its own clean "
        "context and get back only the conclusion. The delegate tool is registered "
        "at startup, so a change applies after a restart.",
        "switch", restart=True),
    _ss("subagent_max_concurrent", "agent", "Delegation", "Helpers at once",
        "Each helper holds its own KV cache next to the parent's, so keep this at 1 "
        "on an 8 GB machine.",
        "range", min=1, max=8, step=1),
    _ss("subagent_max_per_turn", "agent", "Delegation", "Helpers per turn",
        "Total helpers one turn may spawn, so a parent that delegates in a loop "
        "still finishes.",
        "range", min=1, max=20, step=1),
    _ss("subagent_max_steps", "agent", "Delegation", "Steps per helper",
        "Step budget for each helper.",
        "range", min=1, max=20, step=1),
    _ss("subagent_result_chars", "agent", "Delegation", "Helper answer limit",
        "Characters of a helper's answer that come back into the parent's context: "
        "the parent pays for the conclusion, not for the work.",
        "number", min=200, max=20000, step=100, unit="chars"),

    _ss("task_state_enabled", "agent", "Task continuity", "Track the task across turns",
        "Keep a structured record of the task and its artifact per conversation and "
        "show it to the model every turn, so a follow-up is not answered as a new "
        "question once old messages are trimmed.",
        "switch"),
    _ss("drift_check_enabled", "agent", "Task continuity", "Catch task drift",
        "Check a finished answer against the active task and regenerate once on a "
        "strong contradiction, such as Python where the task is Bash.",
        "switch"),
    _ss("task_artifact_chars", "agent", "Task continuity", "Artifact in the prompt",
        "Ceiling on characters of the current artifact put in front of the model. A "
        "small context shrinks it further.",
        "number", min=0, max=200000, step=500, unit="chars"),
    _ss("artifact_reply_headroom", "agent", "Task continuity", "Rewrite headroom",
        "Extra reply tokens on top of the artifact's own size when a turn has to "
        "reproduce the whole file.",
        "number", min=0, max=8192, step=64, unit="tokens", advanced=True),
    _ss("artifact_patch_chars", "agent", "Task continuity", "Patch above",
        "Above this size a change asks for the changed sections only and splices "
        "them back in, instead of re-emitting the whole file. 0 disables patching.",
        "number", min=0, max=200000, step=100, unit="chars", advanced=True),

    _ss("code_check_enabled", "agent", "Code checks", "Check generated code",
        "Look for unbalanced braces, blocks spliced in twice, self-shadowing "
        "functions and common literal mistakes before showing code. Text only, no "
        "tools.",
        "switch"),
    _ss("code_check_deep", "agent", "Code checks", "Parse code with real tools",
        "Also run the language's own parser (bash -n, py_compile, cc) when the host "
        "has one. Starts a subprocess, so it also needs shell execution enabled.",
        "switch"),

    _ss("show_internals", "agent", "Diagnostics", "Show pipeline internals",
        "Add detail events to the trace: prompt sizes, per-step timing, extracted "
        "lengths and fallback reasons.",
        "switch"),
    _ss("debug_prompts", "agent", "Diagnostics", "Log assembled prompts",
        "Log the full prompt, task state, budget and lane of every model call at "
        "DEBUG. Verbose; the prompt text still obeys the chat-content log level.",
        "switch"),

    # ================================================================ #
    # Models & Training (admin)
    # ================================================================ #
    _ss_info("model_catalog", "models", "Switcher", "Extra models offered",
             "Model ids added to the switcher on top of the built-in catalogue.",
             "MODEL_CATALOG", sub="model"),

    _ss("auto_retrain_threshold", "models", "Retraining", "Retrain automatically after",
        "Start a retrain on its own once this many new approved examples have "
        "accumulated. 0 keeps retraining manual.",
        "number", step=1, unit="examples", sub="training"),
    _ss("train_min_examples", "models", "Retraining", "Minimum approved examples",
        "Retraining refuses to run below this many approved examples: a tiny set "
        "overfits and makes the model forget more than it learns.",
        "number", min=1, step=1, unit="examples", sub="training"),

    _ss("train_epochs", "models", "Recipe", "Epochs",
        "Passes over the training set when iterations are derived. More epochs fit "
        "your feedback harder and risk memorising it.",
        "number", min=0.1, max=50.0, step=0.1, sub="training"),
    _ss("train_iters", "models", "Recipe", "Iterations",
        "0 derives the count from the corpus size and the epochs. Any other value "
        "pins it, still held between the minimum and maximum below.",
        "number", min=0, step=10, sub="training"),
    _ss("train_min_iters", "models", "Recipe", "Minimum iterations",
        "Floor on the derived count, so a small corpus still moves the optimiser.",
        "number", min=1, step=10, sub="training"),
    _ss("train_max_iters", "models", "Recipe", "Maximum iterations",
        "Ceiling on the derived count, so a large corpus cannot run for hours "
        "unattended. Never below the minimum.",
        "number", min=1, step=100, sub="training"),
    _ss("train_lr", "models", "Recipe", "Learning rate",
        "Passed to mlx-lm as written, for example 3e-5. Higher learns faster and "
        "forgets more of what the base model knew.",
        "text", placeholder="3e-5", sub="training"),
    _ss("train_batch_size", "models", "Recipe", "Batch size",
        "Examples per optimiser step. 1 is what fits a 3B model on 8 GB; larger "
        "batches need proportionally more memory.",
        "number", min=1, max=64, step=1, sub="training"),
    _ss("train_seq_len", "models", "Recipe", "Sequence length",
        "Longest training example in tokens. mlx-lm truncates longer ones, which "
        "cuts the answer and teaches the model to stop mid-reply.",
        "text", placeholder="1024", sub="training"),
    _ss("train_drop_over_length", "models", "Recipe", "Drop over-length examples",
        "Leave examples longer than the sequence length out at export instead of "
        "letting mlx-lm truncate them.",
        "switch", sub="training"),

    _ss("train_fine_tune_type", "models", "Adapter", "Method",
        "LoRA trains small adapters and fits on 8 GB. DoRA is a slightly heavier "
        "variant. Full fine-tunes every weight and needs far more memory than an "
        "8 GB Mac has.",
        "segmented", sub="training",
        options=_ss_opts(("lora", "LoRA"), ("dora", "DoRA"), ("full", "Full"))),
    _ss("train_num_layers", "models", "Adapter", "Layers to tune",
        "How many of the top layers get adapters. Fewer layers keep a 3B model "
        "trainable on 8 GB; -1 tunes them all (full fine-tuning on big machines).",
        "number", step=1, sub="training"),
    _ss("train_lora_rank", "models", "Adapter", "LoRA rank",
        "Adapter width. A higher rank can learn more and costs memory and time.",
        "number", min=1, max=256, step=1, sub="training"),
    _ss("train_lora_scale", "models", "Adapter", "LoRA scale",
        "How strongly the adapter's update is applied on top of the base weights.",
        "number", min=0.1, max=1000.0, step=0.5, sub="training"),
    _ss("train_lora_dropout", "models", "Adapter", "LoRA dropout",
        "Share of adapter activations dropped while training. A little helps "
        "against overfitting a small corpus.",
        "range", min=0.0, max=0.9, step=0.05, sub="training"),

    _ss("train_on_tool_calls", "models", "Tool traces", "Train on tool traces",
        "Include the agent's own tool-call traces as examples. They teach tool use, "
        "but they are the model's output fed back as ground truth.",
        "switch", sub="training"),
    _ss("train_tool_quality", "models", "Tool traces", "Trace quality",
        "Rated keeps only traces from conversations whose answer a person approved. "
        "All takes anything that did not raise, which compounds the model's "
        "existing tool-selection bias with every retrain.",
        "segmented", sub="training", options=_ss_opts(("rated", "Rated"), ("all", "All"))),
    _ss("train_tool_examples", "models", "Tool traces", "Tool trace limit",
        "Most tool traces one run may include.",
        "number", min=0, step=10, sub="training"),
    _ss("train_tool_ratio", "models", "Tool traces", "Traces per approved example",
        "Caps tool traces as a multiple of the human-approved rows so your feedback "
        "is not drowned out. 0 removes the cap.",
        "number", min=0.0, step=0.5, sub="training"),

    _ss("train_val_check", "models", "Safeguards", "Roll back worse adapters",
        "Refuse to promote an adapter whose held-out loss ended worse than its "
        "starting baseline.",
        "switch", sub="training"),
    _ss("train_val_split", "models", "Safeguards", "Held-out share",
        "Share of the corpus never trained on and used to judge the run.",
        "range", min=0.05, max=0.5, step=0.05, format="percent", sub="training"),
    _ss("train_val_tolerance", "models", "Safeguards", "Loss tolerance",
        "How much worse than the baseline still counts as not worse: the noise band.",
        "range", min=0.0, max=1.0, step=0.01, sub="training"),
    _ss("train_promote_best", "models", "Safeguards", "Promote the best checkpoint",
        "Use the best-scoring periodic checkpoint instead of the last one: early "
        "stopping after the fact, at no extra cost.",
        "switch", sub="training"),
    _ss("train_replay_ratio", "models", "Safeguards", "Rehearsal share",
        "Share of the final training set drawn from the general rehearsal file, so "
        "the adapter keeps its general abilities. 0 disables rehearsal.",
        "range", min=0.0, max=0.5, step=0.05, format="percent", sub="training"),
    _ss("train_timeout", "models", "Safeguards", "Training time limit",
        "Hard ceiling on one run. A hung trainer otherwise leaves the model server "
        "stopped with the status stuck on training.",
        "number", min=60, max=86400, step=60, unit="s", type="float", sub="training"),
    _ss("train_max_backups", "models", "Safeguards", "Adapter backups kept",
        "Each backup is a full copy of the adapter and nothing else deletes them.",
        "number", min=0, max=1000, step=1, sub="training"),

    _ss("rag_enabled", "models", "Retrieval", "Use the knowledge base",
        "Search indexed documents on every question and put the best passages, with "
        "their sources, in front of the model. No effect while nothing is indexed.",
        "switch", sub="knowledge"),
    _ss("rag_passages", "models", "Retrieval", "Passages per question",
        "More passages give the model more to cite and cost context on every turn.",
        "range", min=1, max=20, step=1, sub="knowledge"),

    # ================================================================ #
    # Performance (admin)
    # ================================================================ #
    _ss("tool_result_chars", "performance", "Tool results", "Tool result into context",
        "Characters of a tool result that enter the model's context. A result is "
        "re-sent on every later step, so its cost multiplies with the steps that "
        "follow.",
        "number", min=200, max=40000, step=100, unit="chars"),
    _ss("tool_raw_chars", "performance", "Tool results", "Raw tool output limit",
        "How much a tool may return at all: what is logged, shown in the trace and "
        "read by the summariser. Never below the context limit above.",
        "number", min=200, max=200000, step=1000, unit="chars"),
    _ss("summarise_tool_results", "performance", "Tool results", "Summarise long tool results",
        "Give an oversized result one cheap summarisation pass before it enters the "
        "context. Worth a few seconds because of the per-step multiplier.",
        "switch"),
    _ss("summarise_over_chars", "performance", "Tool results", "Summarise above",
        "Results longer than this are summarised.",
        "number", min=500, step=100, unit="chars"),

    _ss("stable_prefix", "performance", "Long prompts", "Stable prompt prefix",
        "When the trace outgrows the context, fold the oldest steps into one summary "
        "instead of dropping them, so the server's prefix cache stays valid.",
        "switch"),
    _ss("chunk_large_prompts", "performance", "Long prompts", "Chunk oversized prompts",
        "Process a prompt larger than the context part by part into bounded notes, "
        "then synthesise, instead of prefilling it in one pass.",
        "switch"),
    _ss("chunk_trigger_ratio", "performance", "Long prompts", "Chunk above",
        "Share of the context above which a prompt is chunked. It must leave room "
        "for the reply.",
        "range", min=0.2, max=0.9, step=0.05, format="percent"),
    _ss("chunk_size_ratio", "performance", "Long prompts", "Chunk size",
        "Share of the context each chunk targets. Never above the trigger.",
        "range", min=0.1, max=0.9, step=0.05, format="percent"),

    _ss("stall_timeout", "performance", "Timeouts", "Stall timeout",
        "Seconds of silence mid-stream before a generation counts as stalled and "
        "is retried with a smaller budget, instead of hanging on a wedged server.",
        "number", min=10, max=600, step=5, unit="s"),
    _ss("max_generation_timeout", "performance", "Timeouts", "Longest generation",
        "Ceiling on the time budget a non-streaming call derives from its reply "
        "size. Never below the stall timeout.",
        "number", min=10, max=7200, step=30, unit="s", type="float"),
    _ss("decode_floor_tps", "performance", "Timeouts", "Slowest expected decode",
        "Pessimistic decode rate used to size non-streaming time budgets. Lower "
        "waits longer before calling a slow reply stalled.",
        "number", min=0.5, max=1000.0, step=0.5, unit="tok/s"),
    _ss("ready_wait_timeout", "performance", "Timeouts", "Wait for a restarting server",
        "How long a retry waits for a restarting model server to be ready. Longer "
        "helps slow cold starts.",
        "number", min=2, max=300, step=1, unit="s", type="float"),
    _ss("retrieval_deadline", "performance", "Timeouts", "Retrieval deadline",
        "Wall-clock ceiling on fetching and reading all sources for one answer.",
        "number", min=15, max=600, step=5, unit="s", type="float"),
    _ss("reasoning_step_timeout", "performance", "Timeouts", "Reasoning step limit",
        "Wall-clock cap on one reasoning step that streams slowly but never "
        "finishes.",
        "number", min=10, max=300, step=5, unit="s"),
    _ss("agent_run_timeout", "performance", "Timeouts", "Per-agent run limit",
        "Cap for one agent inside a multi-agent run. On timeout that agent reports "
        "a failure and the others still return.",
        "number", min=10, max=3600, step=10, unit="s", type="float"),

    _ss("resilient_retries", "performance", "Resilience", "Retries on failure",
        "Times a failed generation is retried with a smaller budget before the "
        "error is shown. Covers out-of-memory kills and watchdog restarts.",
        "range", min=0, max=6, step=1),
    _ss("min_max_tokens", "performance", "Resilience", "Smallest retry budget",
        "Floor the reply budget shrinks to across retries.",
        "range", min=32, max=512, step=16, unit="tokens"),

    _ss("max_concurrent_generations", "performance", "Admission", "Generations at once",
        "Chat and agent generations that run in parallel. Each holds its own KV "
        "cache in the same memory as the weights, so more slots means more "
        "memory, not just more throughput. Resizes the live controller.",
        "range", min=1, max=64, step=1),
    _ss("generation_queue_depth", "performance", "Admission", "Queue depth multiplier",
        "How many requests may wait, as a multiple of the slot count above "
        "(waiting room = slots x this), before new requests get an immediate 503.",
        "range", min=1, max=64, step=1, unit="x slots"),
    _ss("admission_memory_aware", "performance", "Admission", "Memory-aware admission",
        "Hold a request back while starting it would leave less free memory than "
        "the floor below. A request is always admitted when nothing else runs. "
        "Switches itself off when free memory cannot be read.",
        "switch"),
    _ss("admission_min_free_mb", "performance", "Admission", "Free memory floor",
        "Free memory a new generation must leave after its own KV reservation. "
        "Higher avoids swapping and out-of-memory kills at the cost of queueing.",
        "number", min=0, max=1048576, step=64, unit="MB"),
    _ss_info("max_concurrent_tasks", "performance", "Admission", "Scheduled runs at once",
             "Background task runs that may execute together.", "MAX_CONCURRENT_TASKS"),
    _ss_info("chat_idle_seconds", "performance", "Admission", "Quiet before a scheduled run",
             "Seconds without chat activity before a scheduled run may start.",
             "CHAT_IDLE_SECONDS", unit="s"),

    _ss_info("prefill_step_size", "performance", "Model server", "Prefill step",
             "Tokens prefilled per step. Smaller steps avoid Metal out-of-memory "
             "crashes on long prompts and prefill slower.", "PREFILL_STEP_SIZE",
             unit="tokens"),
    _ss_info("prompt_cache_size", "performance", "Model server", "Cached prefixes",
             "Distinct prompt prefixes the server keeps; each is a live KV sequence.",
             "PROMPT_CACHE_SIZE"),
    _ss_info("prompt_cache_bytes", "performance", "Model server", "Prompt cache memory",
             "Memory ceiling for the server's prompt cache.", "PROMPT_CACHE_BYTES",
             format="bytes"),
    _ss_info("decode_concurrency", "performance", "Model server", "Decode slots",
             "Two slots let the watchdog's liveness probe run beside a long answer.",
             "DECODE_CONCURRENCY"),
    _ss_info("prompt_concurrency", "performance", "Model server", "Parallel prefills",
             "Kept at 1: parallel prefill multiplies the allocation that runs out "
             "of memory.", "PROMPT_CONCURRENCY"),
    _ss_info("kv_bits", "performance", "Model server", "KV cache bits",
             "0 keeps the cache in fp16. Inert on mlx-lm 0.29 and later, which "
             "dropped the flag.", "KV_BITS", advanced=True),
    _ss_info("kv_group_size", "performance", "Model server", "KV quantisation group",
             "Group size for a quantised KV cache.", "KV_GROUP_SIZE", advanced=True),
    _ss_info("quantized_kv_start", "performance", "Model server", "Quantise KV after",
             "Tokens kept in full precision before KV quantisation starts.",
             "QUANTIZED_KV_START", advanced=True, unit="tokens"),
    _ss_info("prompt_cache_dir", "performance", "Model server", "Prompt cache directory",
             "Where a persisted prompt cache lives; empty keeps it in memory only.",
             "PROMPT_CACHE_DIR", advanced=True),

    _ss_info("watchdog_probe_interval", "performance", "Watchdog", "Probe interval",
             "How often the watchdog runs a one-token completion once the server "
             "log has gone quiet.", "WATCHDOG_PROBE_INTERVAL", unit="s"),
    _ss_info("watchdog_probe_failures", "performance", "Watchdog", "Failures before restart",
             "Consecutive failed probes before the model server is restarted.",
             "WATCHDOG_PROBE_FAILURES"),
    _ss_info("watchdog_max_restarts", "performance", "Watchdog", "Restarts allowed",
             "Automatic restarts inside the window before the watchdog gives up and "
             "reports an error.", "WATCHDOG_MAX_RESTARTS"),
    _ss_info("watchdog_probe_timeout", "performance", "Watchdog", "Probe timeout",
             "How long one probe may take.", "WATCHDOG_PROBE_TIMEOUT", unit="s",
             advanced=True),
    _ss_info("watchdog_restart_window", "performance", "Watchdog", "Restart window",
             "Window the restart allowance is counted over.",
             "WATCHDOG_RESTART_WINDOW", unit="s", advanced=True),

    # ================================================================ #
    # Memory
    # ================================================================ #
    _ss("recall_results", "memory", "Recall", "Past exchanges recalled",
        "How many earlier exchanges a search of past conversations returns.",
        "range", min=1, max=50, step=1),
    _ss("recall_snippet_chars", "memory", "Recall", "Recall snippet length",
        "Characters of each recalled exchange: a reminder, not a transcript.",
        "number", min=40, max=2000, step=20, unit="chars"),

    _ss("skills_enabled", "memory", "Skills", "Use skills",
        "Skills are short procedures the model loads by name; only a one-line "
        "description of each sits in the prompt. The skill tool is registered at "
        "startup, so a change applies fully after a restart.",
        "switch", restart=True),
    _ss("skill_autoload", "memory", "Skills", "Load matching skills automatically",
        "Load the matching skill without waiting for the model to ask: a 3B model "
        "rarely reaches for it on its own.",
        "switch"),
    _ss("skills_autolearn", "memory", "Skills", "Learn skills from rated turns",
        "Draft a new skill from a turn you rate well and record outcomes against the "
        "skills a turn used. Off keeps the library unchanged.",
        "switch"),
    _ss("skills_in_prompt", "memory", "Skills", "Skills listed in the prompt",
        "Catalogue lines that enter the prompt, about 15 tokens each. Above this the "
        "ones most relevant to the message win.",
        "range", min=1, max=100, step=1),
    _ss("skill_autoload_overlap", "memory", "Skills", "Words a skill must share",
        "Significant words a message and a skill must share before it autoloads. "
        "Higher misses more matches; lower risks an irrelevant procedure in context.",
        "range", min=1, max=10, step=1),
    _ss("skill_body_chars", "memory", "Skills", "Skill size limit",
        "Hard cap on a loaded skill body. A skill that does not fit the window it "
        "exists to save is worse than none.",
        "number", min=400, max=40000, step=100, unit="chars"),
    _ss("skills_max", "memory", "Skills", "Library size",
        "Most skills the library keeps. Read when the library is opened at startup.",
        "number", min=1, max=5000, step=10, restart=True),
    _ss("skill_author_tokens", "memory", "Skills", "Skill drafting budget",
        "Reply budget for the one small call that drafts a skill.",
        "number", min=96, max=4096, step=32, unit="tokens", advanced=True),
    _ss("skill_retire_min_rated", "memory", "Skills", "Ratings before retiring",
        "Rated uses a skill needs before it can be retired, so one bad day does not "
        "retire a good skill.",
        "number", min=1, step=1, advanced=True),
    _ss("skill_retire_loss_rate", "memory", "Skills", "Retire above this failure rate",
        "Share of rated uses that went badly above which a skill is retired.",
        "range", min=0.1, max=1.0, step=0.05, format="percent", advanced=True),
    _ss_info("skills_dir", "memory", "Skills", "Skill library folder",
             "Where skills are stored; empty means data/skills.", "SKILLS_DIR",
             advanced=True),

    _ss("context_files_enabled", "memory", "Project context", "Load project instructions",
        "Read AGENTS.md, CLAUDE.md, .hermes.md, SOUL.md and .cursorrules from the "
        "project root into the system prompt.",
        "switch"),
    _ss("context_files_chars", "memory", "Project context", "Project instructions limit",
        "Cap across all of those files together, so a file written for a 200k "
        "window cannot fill a 4k one.",
        "number", min=0, max=40000, step=100, unit="chars"),
    _ss("reference_expansion", "memory", "Project context", "Expand @-references",
        "Inline @path, @folder, @url and @diff from a message before the turn "
        "starts, so referring to a file is not a tool call.",
        "switch"),
    _ss("reference_max", "memory", "Project context", "References per message",
        "Most @-references expanded in one message.",
        "range", min=1, max=20, step=1),
    _ss("reference_chars", "memory", "Project context", "Reference size limit",
        "Characters taken from each expanded reference.",
        "number", min=200, max=60000, step=100, unit="chars"),

    _ss("checkpoints_enabled", "memory", "Checkpoints", "Record file checkpoints",
        "Keep each file's original bytes before a turn overwrites it, so an edit "
        "can be rolled back.",
        "switch"),
    _ss("checkpoint_keep", "memory", "Checkpoints", "Checkpoints kept",
        "Older checkpoints are pruned. Read when the store opens at startup.",
        "number", min=1, max=500, step=1, restart=True),
    _ss("checkpoint_max_file_bytes", "memory", "Checkpoints", "Largest file captured",
        "A larger file is recorded as not captured rather than copied on every "
        "write. Read at startup.",
        "number", min=0, step=100000, format="bytes", unit="bytes", restart=True),

    # ================================================================ #
    # Tools
    # ================================================================ #
    _ss("search_results", "tools", "Web", "Search results",
        "Results the search tool returns per query.",
        "range", min=1, max=10, step=1),
    _ss("auto_fetch_results", "tools", "Web", "Pages read after a search",
        "Top result pages fetched automatically so the model sees their text, not "
        "just the snippet. 0 relies on the snippet and the model's own fetches.",
        "number", step=1),
    _ss("auto_fetch_char_cap", "tools", "Web", "Fetched page limit",
        "Most characters of a fetched page that may enter the prompt, so one large "
        "page cannot exhaust the context during prefill.",
        "number", min=1000, max=60000, step=500, unit="chars"),
    _ss_info("search_backend", "tools", "Web", "Search provider",
             "Locked to DuckDuckGo Lite; no other provider exists.", "SEARCH_BACKEND"),
    _ss_info("allow_local_fetch", "tools", "Web", "Fetch local addresses",
             "Off refuses loopback and private-network targets, so an injected page "
             "cannot make the agent read this machine's own services.",
             "ALLOW_LOCAL_FETCH"),

    _ss("tool_temperature", "tools", "Tool calls", "Tool step temperature",
        "Temperature only while the model is choosing a tool. 0 keeps tool calls "
        "deterministic and well-formed.",
        "range", min=0.0, max=2.0, step=0.05),
    _ss("execute_code_max_calls", "tools", "Tool calls", "Tool calls per script",
        "Tool calls one execute_code script may make. That tool also needs Python "
        "execution enabled on the server.",
        "number", min=1, max=1000, step=1),

    _ss_info("allow_shell", "tools", "Code execution", "Shell execution",
             "Whether run_shell and run_tests may run at all.", "ALLOW_SHELL"),
    _ss_info("allow_python", "tools", "Code execution", "Python execution",
             "Whether run_python and execute_code may run at all.", "ALLOW_PYTHON"),
    _ss("exec_backend", "tools", "Code execution", "Run commands in",
        "Local is a confined subprocess with your privileges: a guardrail, not a "
        "boundary. Docker runs each command in a container with no network, which "
        "is a real boundary.",
        "segmented", options=_ss_opts(("local", "Local"), ("docker", "Docker"))),
    _ss("docker_image", "tools", "Code execution", "Docker image",
        "Image the Docker backend runs commands in. A missing image is reported, "
        "not pulled.",
        "text", placeholder="python:3.12-slim"),
    _ss("exec_guardrail", "tools", "Code execution", "Refuse catastrophic commands",
        "Block rm -rf /, mkfs, dd to a device, fork bombs, sudo and git push before "
        "they run. Narrow on purpose, and not a security boundary.",
        "switch"),
    _ss("exec_env_passthrough", "tools", "Code execution", "Environment passed through",
        "Extra variable names, comma separated, that executed code may see. "
        "Everything else is stripped so secrets never reach model-written code.",
        "text", placeholder="DATABASE_URL,FIXTURE_DIR"),
    _ss("exec_memory_mb", "tools", "Code execution", "Memory limit",
        "Memory ceiling for one command. Enforced by Docker; macOS refuses it for "
        "local commands, which report it as unenforced.",
        "number", min=128, max=65536, step=128, unit="MB"),
    _ss("exec_max_file_mb", "tools", "Code execution", "Largest file written",
        "Largest file one command may write, on both backends: the limit that stops "
        "a runaway loop filling the disk.",
        "number", min=1, max=65536, step=1, unit="MB"),
    _ss("exec_pids_limit", "tools", "Code execution", "Process limit",
        "Process ceiling inside a container. Not applied locally.",
        "number", min=16, max=8192, step=16, advanced=True),
    _ss_info("tool_timeout", "tools", "Code execution", "Tool timeout",
             "Wall clock for one shell or Python call.", "TOOL_TIMEOUT", unit="s"),

    _ss("test_command", "tools", "Tests", "Test command",
        "Command run_tests runs in the project. Empty detects pytest, npm test, "
        "cargo test or go test from the project's files.",
        "text", placeholder="Auto-detect"),
    _ss("test_timeout", "tools", "Tests", "Test timeout",
        "Its own, longer clock so a slow but healthy suite is not reported as "
        "failing. Never below the tool timeout.",
        "number", min=30, max=3600, step=10, unit="s"),
    _ss("auto_iterate_rounds", "tools", "Tests", "Fix-and-retest rounds",
        "After an agent turn edits files, run the tests and let it fix failures up "
        "to this many times. 0 turns it off; needs shell execution.",
        "range", min=0, max=5, step=1),

    _ss_info("mcp_timeout", "tools", "MCP", "MCP call timeout",
             "Deadline for one call to an MCP server; the tool reports a timeout and "
             "the agent carries on.", "MCP_TIMEOUT", unit="s"),

    # ================================================================ #
    # Admin
    # ================================================================ #
    _ss("route_max_active_per_node", "admin", "Routing", "Active requests per node",
        "A node with this many requests in flight counts as busy and new work goes "
        "elsewhere.",
        "number", min=1, max=256, step=1, sub="cluster"),
    _ss("route_cpu_pct", "admin", "Routing", "CPU saturation",
        "CPU use at which a node counts as saturated.",
        "range", min=1, max=100, step=1, unit="%", type="float", sub="cluster"),
    _ss("route_mem_pct", "admin", "Routing", "Memory saturation",
        "Memory use at which a node counts as saturated.",
        "range", min=1, max=100, step=1, unit="%", type="float", sub="cluster"),
    _ss("route_load_ratio", "admin", "Routing", "Load per core",
        "Load average per core at which a node counts as saturated. A ratio, not a "
        "percentage: healthy Macs often sit above 1.",
        "number", min=0.0, max=64.0, step=0.5, sub="cluster"),
    _ss("route_sla_ms", "admin", "Routing", "Latency SLA",
        "When the primary's recent latency exceeds this, eligible work spills to "
        "the secondary. 0 disables the factor.",
        "number", min=0, max=3600000, step=100, unit="ms", sub="cluster"),
    _ss("route_cooldown_s", "admin", "Routing", "Failed node cooldown",
        "How long a failed node is skipped before one trial request may close the "
        "breaker again.",
        "number", min=0, max=3600, step=1, unit="s", type="float", sub="cluster"),
    _ss("large_model_markers", "admin", "Routing", "Large model markers",
        "Comma-separated substrings of a model id that mark it as large, so it "
        "prefers the high-memory secondary.",
        "text", placeholder="14B,32B,70B,72B", sub="cluster"),
    _ss("route_queue_depth", "admin", "Routing", "Route queue depth",
        "Reserved: the current router does not read it yet.",
        "number", min=0, max=100000, step=1, sub="cluster", advanced=True),
    _ss("heartbeat_interval", "admin", "Heartbeat", "Heartbeat interval",
        "How often node health is refreshed.",
        "number", min=1, max=3600, step=1, unit="s", type="float", sub="cluster"),
    _ss("heartbeat_timeout", "admin", "Heartbeat", "Heartbeat timeout",
        "Reserved: the current heartbeat loop does not read it yet.",
        "number", min=2, max=86400, step=1, unit="s", type="float", sub="cluster",
        advanced=True),
    _ss_info("node_probe_timeout", "admin", "Heartbeat", "Node probe timeout",
             "Timeout for one health probe of another node.", "NODE_PROBE_TIMEOUT",
             unit="s", sub="cluster", advanced=True),

    _ss_info("import_max_zip_bytes", "admin", "Limits", "Largest archive",
             "Biggest chat export upload accepted.", "IMPORT_MAX_ZIP_BYTES",
             format="bytes", sub="import"),
    _ss_info("import_max_uncompressed_bytes", "admin", "Limits", "Largest unpacked size",
             "Total unpacked size before an archive counts as a zip bomb.",
             "IMPORT_MAX_UNCOMPRESSED_BYTES", format="bytes", sub="import"),
    _ss_info("import_max_file_bytes", "admin", "Limits", "Largest file inside",
             "Biggest single file inside an archive.", "IMPORT_MAX_FILE_BYTES",
             format="bytes", sub="import"),
    _ss_info("import_max_files", "admin", "Limits", "Most files inside",
             "Most entries an archive may hold.", "IMPORT_MAX_FILES", sub="import"),

    _ss("log_level", "admin", "Logging", "Log level",
        "DEBUG and TRACE are verbose and useful while diagnosing; INFO is the normal "
        "level. Applies immediately.",
        "select", sub="logs",
        options=_ss_opts(("CRITICAL", "Critical"), ("ERROR", "Error"),
                         ("WARNING", "Warning"), ("INFO", "Info"), ("DEBUG", "Debug"),
                         ("TRACE", "Trace"))),
    _ss("log_format", "admin", "Logging", "Log format",
        "JSON for aggregation (Loki, ELK, Datadog), text for reading locally. Text "
        "drops the structured fields.",
        "segmented", sub="logs", options=_ss_opts(("json", "JSON"), ("text", "Text"))),
    _ss("log_chat_content", "admin", "Logging", "Chat content in logs",
        "Metadata logs only length and a fingerprint, the production default. Full "
        "logs redacted, truncated text. Disabled logs nothing about content.",
        "segmented", sub="logs",
        options=_ss_opts(("disabled", "Disabled"), ("metadata", "Metadata"),
                         ("full", "Full"))),
    _ss_info("log_dir", "admin", "Log files", "Log folder",
             "Where app.log and the model, training and task logs are written; empty "
             "means ./logs.", "LOG_DIR", sub="logs"),
    _ss_info("log_max_bytes", "admin", "Log files", "Rotate at",
             "Size at which app.log rotates.", "LOG_MAX_BYTES", format="bytes",
             sub="logs"),
    _ss_info("log_backup_count", "admin", "Log files", "Rotations kept",
             "Rotated files kept beside app.log.", "LOG_BACKUP_COUNT", sub="logs"),
    _ss_info("log_retention_days", "admin", "Log files", "Retention",
             "Rotations older than this are deleted at startup.", "LOG_RETENTION_DAYS",
             unit="days", sub="logs"),

    _ss_info("auth_enabled", "admin", "Sign-in", "Accounts and sign-in",
             "Off means one synthetic local admin owns everything; on, every request "
             "needs a session.", "AUTH_ENABLED", sub="system"),
    _ss_info("oidc_enabled", "admin", "Sign-in", "Microsoft Entra ID sign-in",
             "Offers Sign in with Microsoft next to local accounts.", "OIDC_ENABLED",
             sub="system"),
    _ss_info("allow_test_user", "admin", "Sign-in", "Test account",
             "A convenience non-admin account for development. Remove it in "
             "production.", "AUTH_ALLOW_TEST_USER", sub="system"),
    _ss_info("auth_session_ttl_hours", "admin", "Sign-in", "Session lifetime",
             "How long a sign-in lasts.", "AUTH_SESSION_TTL_HOURS", unit="hours",
             sub="system"),
    _ss_info("auth_cookie_secure", "admin", "Sign-in", "Secure cookie",
             "Marks the session cookie HTTPS-only; needed behind a TLS proxy, wrong "
             "on plain HTTP.", "AUTH_COOKIE_SECURE", sub="system"),
    _ss_info("node_role", "admin", "This node", "Node role",
             "Primary routes work; a secondary only serves generations.", "NODE_ROLE",
             sub="system"),
    _ss_info("node_name", "admin", "This node", "Node name",
             "Name shown in routing decisions.", "NODE_NAME", sub="system"),
    _ss_info("studio_node_url", "admin", "This node", "Secondary node",
             "Generation URL of the high-memory secondary; empty runs one node.",
             "STUDIO_NODE_URL", sub="system"),
    _ss_info("web_port", "admin", "This node", "Web port",
             "Port this interface and the API listen on.", "WEB_PORT", sub="system"),
    _ss_info("model_port", "admin", "This node", "Model server port",
             "Port of the local mlx_lm.server.", "MODEL_PORT", sub="system"),
    # --- Inference backend (item 1, backends.py): restart-only, so read-only.
    # GET /api/backends lists the engines and the live one's decisions.
    _ss_info("model_backend", "performance", "Inference engine", "Backend",
             "Engine serving the model: mlx-lm (the stock server), mlx-lm-turbo (the "
             "same server with batching, prompt cache and drafts sized for this Mac), "
             "mlx-engine (this app's own server: continuous batching, speculative "
             "decoding for a lone request, byte-bounded prefix cache), mlx-distributed "
             "(layer sharding across nodes) or openai-external (BACKEND_COMMAND).",
             "MODEL_BACKEND"),
    _ss_info("draft_model", "performance", "Inference engine", "Draft model",
             "Small model of the same family for speculative decoding; empty is off, "
             "auto asks the catalogue. On mlx-lm it disables continuous batching.",
             "DRAFT_MODEL"),
    _ss_info("num_draft_tokens", "performance", "Inference engine", "Draft tokens",
             "Tokens the draft proposes per step; 0 keeps the server default.",
             "NUM_DRAFT_TOKENS", advanced=True),
    _ss_info("backend_command", "performance", "Inference engine", "External command",
             "Start command for openai-external, with {model} and {port} placeholders.",
             "BACKEND_COMMAND", advanced=True),
    _ss_info("backend_capabilities", "performance", "Inference engine",
             "External capabilities",
             "Comma-separated capabilities the external engine has (batching, vision, ...).",
             "BACKEND_CAPABILITIES", advanced=True),
    _ss_info("backend_extra_args", "performance", "Inference engine", "Extra arguments",
             "Appended to any backend's command line, for flags the app does not model.",
             "BACKEND_EXTRA_ARGS", advanced=True),
    # Tool-call parsing (toolparse.py).
    _ss("tool_call_format", "tools", "Tool calls", "Tool-call formats",
        "Which model formats are read as tool calls. auto accepts all of them; or a "
        "comma list of json, openai, hermes, qwen3_coder, glm, llama3, mistral, "
        "pythonic, gemma, deepseek, kimi, harmony. The JSON protocol is always on.",
        "text", placeholder="auto", advanced=True),
    _ss("tool_call_repair", "tools", "Tool calls", "Repair broken tool calls",
        "Fix the malformed JSON small quantized models produce (missing quotes, "
        "trailing commas, unclosed braces) instead of showing it as the answer. "
        "Tool names are never guessed.",
        "switch"),
    # --- Cluster, sharding, remote access (items 4, 15, 20): all restart-only,
    # so read-only facts. NODE_E2E_KEY is a secret and never listed.
    _ss_info("cluster_nodes", "admin", "This node", "Cluster nodes",
             "Peer nodes as a JSON list or comma-separated URLs; empty uses the nodes "
             "paired with deploy.py cluster join.", "CLUSTER_NODES", sub="system"),
    _ss_info("cluster_hostfile", "admin", "This node", "Shard hostfile",
             "mlx.launch hostfile for the mlx-distributed backend; empty is "
             "DATA_DIR/cluster_hostfile.json.", "CLUSTER_HOSTFILE", sub="system",
             advanced=True),
    _ss_info("web_host", "admin", "Remote access", "Bind interface",
             "Interface the web server listens on; anything but loopback needs "
             "AUTH_ENABLED=1.", "WEB_HOST", sub="system"),
    _ss_info("public_base_url", "admin", "Remote access", "Public URL",
             "Origin browsers use behind a proxy or tunnel; the sign-in redirect is "
             "built from it.", "PUBLIC_BASE_URL", sub="system"),
    _ss_info("trusted_proxies", "admin", "Remote access", "Trusted proxies",
             "Proxy IPs or CIDRs whose X-Forwarded-For names the real client.",
             "TRUSTED_PROXIES", sub="system", advanced=True),
    _ss_info("node_ca_file", "admin", "Node security", "Node CA bundle",
             "CA certificates that sign peer node TLS certificates.", "NODE_CA_FILE",
             sub="system", advanced=True),
    _ss_info("node_client_cert", "admin", "Node security", "Node client certificate",
             "Client certificate presented to peer nodes (mutual TLS).",
             "NODE_CLIENT_CERT", sub="system", advanced=True),
    _ss_info("node_client_key", "admin", "Node security", "Node client key file",
             "Private key file for the node client certificate.", "NODE_CLIENT_KEY",
             sub="system", advanced=True),
    _ss_info("node_allow_plaintext", "admin", "Node security", "Allow plain HTTP peers",
             "Permits http:// to peers outside private and tailnet ranges.",
             "NODE_ALLOW_PLAINTEXT", sub="system", advanced=True),
    # --- Model catalogue and benchmark (catalog.py, item 9).
    _ss("model_aliases", "models", "Switcher", "Model aliases",
        "Your own short names for models, as a JSON object of alias to repo id. They "
        "add to (and can override) the built-in fast, balanced, coder, reasoning, "
        "vision, tiny and large, and work in the model switcher and MODEL_ID. An "
        "invalid value is refused and the previous aliases are kept.",
        "textarea", rows=3, sub="model", advanced=True,
        placeholder='{"chat": "mlx-community/Qwen2.5-3B-Instruct-4bit"}'),
    _ss_info("model_alias", "models", "Switcher", "Started from alias",
             "The alias MODEL_ID or --model named at startup, if any; the model "
             "field always shows the repo id it resolved to.", "MODEL_ID", sub="model",
             advanced=True),
    _ss("bench_min_decode_tps", "models", "Benchmark", "Minimum decode speed",
        "The benchmark recommendation (deploy.py recommend) never picks a model that "
        "generated slower than this, in tokens per second.",
        "number", min=1, max=200, step=0.5, unit="tok/s", sub="model", advanced=True),
    _ss("bench_headroom_gb", "models", "Benchmark", "Memory headroom",
        "RAM the recommendation keeps free beyond a model's measured peak, for the OS, "
        "this app and training. 0 is automatic: the larger of 2.5GB and a quarter "
        "of RAM.",
        "number", min=0, max=256, step=0.5, unit="GB", sub="model", advanced=True),
    # --- Training methods and autopilot (train_methods.py) --- #
    _ss("train_method", "models", "Method", "Training method",
        "SFT imitates approved answers. DPO and ORPO learn from corrections and from "
        "a thumbs-up next to a thumbs-down on the same prompt; GRPO optimises "
        "programmatic rewards (experimental). DPO, ORPO and GRPO need the mlx-lm-lora "
        "package and are refused without it, never run as SFT. KTO is export only.",
        "select", sub="training",
        options=_ss_opts(("sft", "SFT"), ("dpo", "DPO"), ("orpo", "ORPO"),
                         ("kto", "KTO (export only)"), ("grpo", "GRPO (experimental)"))),
    _ss("train_beta", "models", "Method", "Preference strength (beta)",
        "How far DPO, ORPO and GRPO may move from the reference behaviour. Higher "
        "follows the preferences harder.",
        "number", min=0.0, max=10.0, step=0.05, sub="training"),
    _ss("train_reference_adapter", "models", "Method", "Reference model",
        "Model DPO and GRPO compare against. Empty uses the base model. A path must be "
        "a full model directory: fuse an adapter first, the trainer loads none.",
        "text", placeholder="base model", sub="training", advanced=True),
    _ss("train_grpo_rewards", "models", "Method", "GRPO rewards",
        "Reward terms and weights: tool_call (the reply parses as the expected tool "
        "call), format (no broken structured output), length (close to the approved "
        "answer's length).",
        "text", placeholder="tool_call:1.0,format:0.5,length:0.25", sub="training",
        advanced=True),
    _ss("train_autopilot", "models", "Autopilot", "Size the recipe automatically",
        "Choose layers, rank, batch, sequence window and gradient checkpointing from "
        "this Mac's RAM, the model and the corpus, and refuse a run that will not fit. "
        "Off keeps the recipe above exactly.",
        "switch", sub="training"),
    _ss("train_promote_ungated", "models", "Safeguards", "Promote unverified preference runs",
        "Keep a DPO, ORPO or GRPO adapter even when the run produced no metric the "
        "rollback check can compare (GRPO never does). Off rolls such runs back.",
        "switch", sub="training", advanced=True),
    # --- Chat turns (streams.py) ---
    _ss("turn_detach_grace_s", "performance", "Resilience", "Keep a turn after disconnect",
        "Seconds a streaming answer keeps running after the browser disconnects, so "
        "reloading the page or a dropped connection picks it back up instead of "
        "losing it. 0 stops the answer as soon as the connection drops.",
        "range", min=0.0, max=600.0, step=5.0, unit="s"),
    # --- Workspaces and projects (workspaces.py) --- #
    _ss("workspaces_enabled", "tools", "Workspaces", "Per-user workspaces",
        "Give every non-admin user their own folder for the file tools, and let a "
        "chat work inside a project. Off, everyone shares one workspace. Local "
        "shell and Python are not confined by this: use the Docker backend.",
        "switch"),
    _ss("workspace_roots", "tools", "Workspaces", "Shared project locations",
        "Folders an admin may place a shared project under, comma separated. Empty "
        "allows the workspace folder and the codebase folder.",
        "text", placeholder="/Users/shared/projects", advanced=True),
    # --- Tool approval policy (policy.py) --- #
    _ss("tool_approval_mode", "tools", "Approvals", "Ask before running tools",
        "Off runs every tool call as before. Risky asks you before anything that "
        "writes files, runs code or sends something out; reading stays automatic. "
        "All asks before every call.",
        "segmented", options=_ss_opts(("off", "Off"), ("risky", "Risky"), ("all", "All"))),
    _ss("tool_approval_timeout_s", "tools", "Approvals", "Approval timeout",
        "How long a prompt waits for an answer before the call counts as refused.",
        "number", min=5, max=3600, step=5, unit="s"),
    _ss("tool_approval_unattended", "tools", "Approvals", "When nobody can answer",
        "Scheduled tasks and API calls cannot show a prompt. Deny refuses what "
        "would have needed approval; allow runs it anyway.",
        "segmented", options=_ss_opts(("deny", "Deny"), ("allow", "Allow"))),
    _ss("tool_approval_taint", "tools", "Approvals", "Re-ask after reading the web",
        "After a turn reads a web page, an MCP result or mail, its writes, code "
        "runs and sends need a fresh approval even where you said always allow: "
        "the request may have come from that content, not from you.",
        "switch"),
    _ss("tool_approval_roles", "tools", "Approvals", "Rules per role",
        "JSON rules by role, keyed by tool name, cap:<capability>, risk:<class> or *. "
        'Example: {"user": {"risk:exec": "deny"}, "admin": {"*": "allow"}}. An '
        "administrator deny cannot be loosened by a user's own rules.",
        "textarea", rows=3, placeholder='{"user": {"risk:exec": "ask"}}', advanced=True),
    # --- OpenAI / Anthropic compatible API (openai_compat.py, anthropic_compat.py) --- #
    _ss("v1_default_mode", "tools", "Integrations", "API clients run as",
        "What /v1/chat/completions, /v1/responses and /v1/messages do when the model "
        "is a plain name. Agent runs this app's agent with its own tools. Raw hands "
        "the model the client's tools, for coding clients (Claude Code, Codex, Aider) "
        "that run their own tool loop. A client can always ask for raw/<model> or "
        "agent/<profile> by model name.",
        "segmented", options=_ss_opts(("agent", "Agent"), ("raw", "Raw"))),
    _ss("v1_raw_enabled", "tools", "Integrations", "Allow raw model access",
        "Let API clients use the model directly with their own tools. Off, every "
        "API request runs the agent, whatever model name it asks for.",
        "switch"),
    # API tokens and terminal sign-in (api_tokens.py, auth.py device flow).
    _ss_info("api_token_default_ttl_days", "admin", "Sign-in", "API token lifetime",
             "Days a new API token for coding tools and the CLI stays valid, and the "
             "longest a user may request; 0 means tokens never expire.",
             "API_TOKEN_DEFAULT_TTL_DAYS", unit="days", sub="system"),
    _ss_info("oidc_cli_client_id", "admin", "Sign-in", "Terminal sign-in app",
             "Client id of an Entra app with public client flows enabled; turns on "
             "Microsoft sign-in from the terminal (device code). Empty turns it off.",
             "OIDC_CLI_CLIENT_ID", sub="system"),
    # --- Multimodal input (attachments.py, audio.py, vision.py) --- #
    _ss("attachments_enabled", "tools", "Attachments", "Allow attachments",
        "Let users attach images, voice notes and documents to a chat message. "
        "Images reach the model only when it can see; documents are read as text "
        "and voice notes are transcribed.",
        "switch"),
    _ss("attachment_max_bytes", "tools", "Attachments", "Largest upload",
        "The biggest single file a user may attach or send for transcription.",
        "number", min=1024, max=536870912, step=1048576, unit="bytes", advanced=True),
    _ss("attachment_quota_mb", "tools", "Attachments", "Storage per user",
        "Total attachment storage per user. 0 means no limit.",
        "number", min=0, max=1048576, step=64, unit="MB", advanced=True),
    _ss("vision_mode", "models", "Vision", "Image input",
        "Auto sends images only when the engine and the model can both read them "
        "(use the mlx-vlm engine for vision models). On always sends them, off "
        "never does.",
        "segmented", sub="model",
        options=_ss_opts(("auto", "Auto"), ("on", "On"), ("off", "Off"))),
    _ss("whisper_model", "tools", "Voice", "Speech-to-text model",
        "The mlx-whisper model that transcribes voice notes and "
        "/v1/audio/transcriptions. Downloaded on first use.",
        "text", placeholder="mlx-community/whisper-tiny", advanced=True),
    _ss("tts_model", "tools", "Voice", "Text-to-speech model",
        "The mlx-audio model behind /v1/audio/speech. Downloaded on first use.",
        "text", placeholder="mlx-community/Kokoro-82M-bf16", advanced=True),
    _ss("tts_voice", "tools", "Voice", "Voice",
        "The voice text-to-speech uses when a request names none (or an OpenAI "
        "voice such as alloy).",
        "text", placeholder="af_heart", advanced=True),
    # --- MCP server endpoint, role capabilities, named tools (mcp_server.py, tools_extra.py) --- #
    _ss("mcp_server_enabled", "tools", "Integrations", "Serve tools over MCP",
        "Let MCP clients (Claude Code, IDE agents) use this app's tools at /mcp after "
        "signing in. Each call runs as that user, with their role's tools and the "
        "approval rules for unattended use.",
        "switch"),
    _ss("role_capabilities", "tools", "Capabilities", "Capabilities per role",
        "JSON lists of capability keys per role; a role that is not listed keeps every "
        'capability. Example: {"user": ["web_search", "file_ops", "mcp"], "admin": ["*"]}. '
        "Narrows agent profiles in chat and the tools offered over MCP.",
        "textarea", rows=3, placeholder='{"user": ["web_search", "file_ops"]}', advanced=True),
    _ss_info("extra_tools", "tools", "Capabilities", "Extra tools for the default agent",
             "Newer tools (browse_page, git, http_request, ...) the default agent gets "
             "without a profile: empty for none, all, or a comma list. Profiles get them "
             "from their capabilities either way.", "EXTRA_TOOLS", advanced=True),
    _ss_info("database_urls", "tools", "Capabilities", "Read-only databases",
             'SQLite databases the database tool may query, as JSON {"name": "/path.db"}.',
             "DATABASE_URLS", advanced=True),
    _ss_info("image_gen_command", "tools", "Capabilities", "Image generator",
             "Command template for image_generate with {prompt} and {output}; empty "
             "uses mflux-generate when installed.", "IMAGE_GEN_COMMAND", advanced=True),
    _ss_info("image_gen_timeout_s", "tools", "Capabilities", "Image generation timeout",
             "Seconds one image may take before it is stopped.", "IMAGE_GEN_TIMEOUT_S",
             unit="s", advanced=True),
    _ss_info("notify_macos", "tools", "Capabilities", "macOS notifications",
             "Show an administrator's notify messages as a macOS banner when the server "
             "runs on this Mac's own screen.", "NOTIFY_MACOS", advanced=True),
    _ss_info("o365_user", "tools", "Capabilities", "Fallback mailbox",
             "Mailbox the email, calendar and contacts tools use for a local run or an "
             "administrator without an address; users always get their own.",
             "O365_USER", advanced=True),
    # --- Cloud fallback for oversized prefill (cloud.py) --- #
    _ss("cloud_fallback_enabled", "admin", "Cloud fallback", "Offer a cloud model for huge prompts",
        "When a prompt is too large or too slow for the local model, ask the user whether "
        "it may go to the configured cloud model instead. Nothing leaves this machine "
        "without that consent; declining runs locally as before. Needs CLOUD_MODEL and "
        "CLOUD_API_KEY.",
        "switch", sub="system"),
    _ss("cloud_consent", "admin", "Cloud fallback", "Consent",
        "Per request asks on every turn that would go to the cloud. Per user also offers "
        "this conversation and always; only an always consent lets scheduled tasks and "
        "API clients use the cloud.",
        "segmented", sub="system",
        options=_ss_opts(("per_request", "Per request"), ("per_user", "Per user"))),
    _ss("cloud_min_prefill_s", "admin", "Cloud fallback", "Slow prefill threshold",
        "Offer the cloud when reading the prompt locally would take at least this long "
        "at the measured prompt speed. 0 turns this trigger off; a prompt larger than "
        "the context window is always offered.",
        "number", min=0, max=86400, step=10, unit="s", sub="system"),
    _ss("cloud_min_prompt_tokens", "admin", "Cloud fallback", "Large prompt threshold",
        "Offer the cloud for any prompt of at least this many tokens. 0 turns this "
        "trigger off.",
        "number", min=0, max=10000000, step=1000, unit="tokens", sub="system"),
    _ss("cloud_allowed_roles", "admin", "Cloud fallback", "Who may use it",
        "Roles and users who may be offered the cloud, comma separated: role names, "
        "user:<name> entries, or * for everyone.",
        "text", placeholder="admin, user:alice", sub="system"),
    _ss("cloud_max_tokens_per_user_month", "admin", "Cloud fallback", "Monthly tokens per user",
        "Cloud tokens (prompt and reply) one user may use per calendar month; past it "
        "their prompts stay local. 0 means no cap.",
        "number", min=0, max=1000000000000, step=100000, unit="tokens", sub="system"),
    _ss_info("cloud_provider", "admin", "Cloud fallback", "Provider",
             "openai for any OpenAI-compatible endpoint, anthropic for the Messages API.",
             "CLOUD_PROVIDER", sub="system"),
    _ss_info("cloud_base_url", "admin", "Cloud fallback", "Endpoint",
             "Where cloud prompts are sent; empty means the provider's public API. The "
             "key is CLOUD_API_KEY and is never shown.", "CLOUD_BASE_URL", sub="system"),
    _ss_info("cloud_model", "admin", "Cloud fallback", "Cloud model",
             "The model cloud prompts run on; empty keeps the fallback unavailable.",
             "CLOUD_MODEL", sub="system"),
    # ---- extension settings fields: one block per feature, above this line ---- #
]

# Config.MUTABLE keys deliberately not shown as a control, with the reason.
UI_HIDDEN: dict[str, str] = {
    "project_dir": ("Edited in Admin > Codebase, which has a folder browser and a Set "
                    "button whose request the server resolves and refuses unless it "
                    "is an existing directory."),
    "rag_scope": ("Edited in Models & Training > Knowledge as a list of indexed "
                  "documents to tick; /api/docs/scope validates every path against "
                  "the index, which a free-text box would bypass."),
    # ---- extension hidden settings: above this line ---- #
}


# Sub-tabs of the sections that hold more than one screenful. A field's `sub`
# picks its tab; the bespoke panels (model switcher, users, cluster, ...) are
# rendered into the tab of the same key by ui_settings.py.
SETTINGS_SUBTABS: dict[str, list[dict[str, str]]] = {
    "models": [
        {"key": "model", "label": "Model"},
        {"key": "training", "label": "Training"},
        {"key": "knowledge", "label": "Knowledge"},
    ],
    "admin": [
        {"key": "users", "label": "Users"},
        {"key": "agents", "label": "Agents"},
        {"key": "cluster", "label": "Cluster"},
        {"key": "import", "label": "Import"},
        {"key": "codebase", "label": "Codebase"},
        {"key": "backup", "label": "Backup"},
        {"key": "logs", "label": "Logs"},
        {"key": "system", "label": "System"},
    ],
}


def settings_field_type(field: dict[str, Any]) -> str:
    """The JSON type a control's value is sent as: int, float, bool or str.

    Guessed from the widget and its bounds; an explicit `type` wins. The guess
    matters because POST /api/config coerces by the Config field's type, and a
    float sent for an int field (or "0.5" as text) is refused, not rounded.
    """
    if field.get("type"):
        return str(field["type"])
    widget = field.get("widget")
    if widget == "switch":
        return "bool"
    if widget in ("range", "number"):
        if any(isinstance(field.get(k), float) for k in ("min", "max", "step")):
            return "float"
        return "int"
    return "str"


def settings_field_sub(field: dict[str, Any]) -> str:
    """The sub-tab a field renders in ("" for a section without sub-tabs)."""
    tabs = SETTINGS_SUBTABS.get(field.get("section", ""))
    if not tabs:
        return ""
    keys = [t["key"] for t in tabs]
    sub = field.get("sub")
    return sub if sub in keys else keys[0]


def settings_control_id(field: dict[str, Any]) -> str:
    """Element id of a field's control: cfg<Pascal> (global/info), pref<Pascal> (user)."""
    if field.get("id"):
        return str(field["id"])
    pascal = "".join(part[:1].upper() + part[1:] for part in str(field["key"]).split("_"))
    return ("pref" if field.get("scope") == "user" else "cfg") + pascal


def fields_for(section: str, *, admin: bool) -> list[dict[str, Any]]:
    """The fields of one section that a user with this role may see."""
    sec = next((s for s in SETTINGS_SECTIONS if s["key"] == section), None)
    if sec is None or (sec["admin"] and not admin):
        return []
    out = []
    for f in SETTINGS_FIELDS:
        if f["section"] != section:
            continue
        if f["scope"] == "global" and not admin:
            continue
        out.append(f)
    return out


def schema_for_role(admin: bool) -> dict[str, Any]:
    """The schema served by GET /api/settings/schema for this role."""
    sections = [s for s in SETTINGS_SECTIONS if admin or not s["admin"]]
    return {
        "sections": sections,
        "fields": [f for s in sections for f in fields_for(s["key"], admin=admin)],
        "subtabs": {k: v for k, v in SETTINGS_SUBTABS.items()
                    if any(s["key"] == k for s in sections)},
        "user_prefs": {k: {kk: vv for kk, vv in v.items()} for k, v in USER_PREFS.items()},
    }


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'SETTINGS_SECTIONS',
    'SECTION_KEYS',
    'WIDGETS',
    'USER_PREFS',
    'SETTINGS_FIELDS',
    'UI_HIDDEN',
    'SETTINGS_SUBTABS',
    'settings_field_type',
    'settings_field_sub',
    'settings_control_id',
    'validate_user_prefs',
    'default_user_prefs',
    'fields_for',
    'schema_for_role',
]
