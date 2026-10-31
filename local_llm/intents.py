"""The app's own vocabulary, read as instructions before any model decides.

The UI names its features: Knowledge base, Codebase, Skills, Memory, History,
Prompts, Tasks, Tools. People type those words in the chat and mean them. The
router did not: it is one short call to the main model with a menu of six
tools, so "check the codebase for design mistakes" came back `answer` (a prompt
with no tools at all), "add this to skills" with an attached SKILL.md came back
`answer` because the router never saw the attachment, and "use the knowledge
base to craft a pdf" ran a document turn whose only tool was create_document.
All three are reviewed in upgrade/ROUTING_AND_TOOLING.md.

classify() turns those words into an Intent deterministically, before the
router runs: which lane the turn takes, which tools it may see, whether
knowledge-base retrieval is injected, and whether the action needs the user's
confirmation. The first matching rule wins; no match returns None and the turn
goes down the existing pipeline unchanged. It never calls a model.

Precision over recall, everywhere below. A rule that fires wrongly takes a turn
away from the router and the model with no way back inside that turn, while a
rule that stays silent only costs what the turn cost before this module
existed. So every rule needs clear phrasing, each one carries the guards its
false positives taught it (a pasted code block, a question about how to do
something elsewhere, a CV that lists skills), and anything ambiguous is left to
the router.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from .config import *  # noqa: F401,F403
from .textutil import *  # noqa: F401,F403
from .tools_extra import EXTRA_TOOL_NAMES

# Lanes an intent can select.
#   tools          the tool loop with `tool_subset` (router skipped)
#   prose          no tools (the router's "answer" lane)
#   reply          a deterministic answer in `reply`, no model call
#   clarify        ask `clarify_question`, no model call
#   document       the document lane, with lookups from `research_sources`
INTENT_LANES = ("tools", "prose", "reply", "clarify", "document")

# Retrieval rules for automatic knowledge-base injection (with_retrieved_context).
#   off   never inject   kb   inject, KB only   auto   today's behaviour
INTENT_RETRIEVAL = ("off", "kb", "auto")

# The intent names. classify() evaluates its rules in _IT_RULES order, which
# differs from this listing in two places on purpose: document_create runs
# before the history, knowledge-base and codebase rules because a document
# request absorbs those as research sources ("use the knowledge base to craft a
# pdf" is a PDF built from the KB, not a KB question), and schedule runs before
# disabled_capability because "remind me every morning to check my email" asks
# for a scheduled task, not for the mailbox now.
INTENT_NAMES = (
    "conv_action", "admin_action", "self_describe", "disabled_capability",
    "skill_save", "skill_use", "skill_list",
    "memory_write", "memory_forget", "memory_read",
    "history",
    "kb_write", "kb_overview", "kb_query",
    "codebase_write", "codebase_lookup", "codebase_review",
    "schedule",
    "document_create",
)

# Tools each tools-lane intent may see. Names not registered under the running
# config are dropped at use time (like Agent.document_tool_subset), so a subset
# can name a tool that only exists in a later phase.
INTENT_TOOL_SUBSETS: dict[str, tuple[str, ...]] = {
    "skill_save": ("save_skill", "search_skills", "final_answer"),
    "skill_use": ("load_skill", "search_skills", "final_answer"),
    "skill_list": ("list_skills", "search_skills", "final_answer"),
    "memory_write": ("remember", "final_answer"),
    "memory_forget": ("recall_memory", "forget", "final_answer"),
    "memory_read": ("recall_memory", "search_docs", "final_answer"),
    "history": ("search_past_conversations", "final_answer"),
    "kb_write": ("index_docs", "index_url", "final_answer"),
    "kb_overview": ("kb_overview", "search_docs", "final_answer"),
    "kb_query": ("search_docs", "kb_overview", "final_answer"),
    "codebase_lookup": ("search_files", "read_file", "list_files", "final_answer"),
    "codebase_review": ("repo_map", "list_files", "search_files", "read_file",
                        "file_info", "delegate_task", "final_answer"),
    # Documentation written from the project's own files. No web tools: the
    # README turn that motivated this intent ran on the full catalogue and
    # fetched a raw.githubusercontent URL the model made up, for a project
    # that sits on disk. Both write tools are confirmed (confirm_tools).
    "codebase_write": ("list_files", "search_files", "read_file", "file_info",
                       "edit_file", "write_file", "final_answer"),
    # A recurring job is created by the task scheduler and nothing else; the
    # confirm card shows the name, goal, interval and tools before it exists.
    "schedule": ("task_scheduler", "final_answer"),
}

# Lookup tools a document turn gets per research source (the other targets
# named in the same message). "web" is today's DOCUMENT_LOOKUP_TOOLS behaviour.
DOCUMENT_RESEARCH_TOOLS: dict[str, tuple[str, ...]] = {
    "web": ("web_search", "fetch_url"),
    "knowledge_base": ("kb_overview", "search_docs"),
    "codebase": ("repo_map", "list_files", "search_files", "read_file"),
    "history": ("search_past_conversations",),
    "files": ("read_file",),
}


@dataclass(frozen=True)
class Intent:
    """What a message asks for, in the app's own terms."""

    name: str                                   # one of INTENT_NAMES
    lane: str                                   # one of INTENT_LANES
    tool_subset: tuple[str, ...] = ()           # tools lane only
    retrieval: str = "auto"                     # one of INTENT_RETRIEVAL
    research_sources: tuple[str, ...] = ()      # document lane: keys of DOCUMENT_RESEARCH_TOOLS
    targets: tuple[str, ...] = ()               # codebase, knowledge_base, skills, memory, history, web, files, document, attachment
    confirm_tools: tuple[str, ...] = ()         # tool calls that need the user's OK in an interactive turn
    reply: str = ""                             # reply lane: the deterministic answer
    clarify_question: str = ""                  # clarify lane: the one question to ask
    referent: str = ""                          # none | attachment | previous_answer | named_path | inline
    source: str = "lexicon"                     # lexicon | slash
    confidence: float = 1.0
    matched: str = ""                           # the text that matched, for the trace and logs
    extras: dict = field(default_factory=dict)

    def as_log(self) -> dict:
        """Metadata-only fields for router.decision logging (no message text)."""
        return {"intent": self.name, "lane": self.lane, "intent_source": self.source,
                "confidence": self.confidence, "tool_subset": list(self.tool_subset),
                "retrieval": self.retrieval, "research_sources": list(self.research_sources)}


# --------------------------------------------------------------------------- #
# What the rules need to know besides the text, gathered once per call.
# --------------------------------------------------------------------------- #
class _ItCtx:
    __slots__ = ("message", "body", "has_attachment", "attachment_names", "has_history",
                 "basename_rx", "registered", "fenced", "source")

    def __init__(self, message: str, *, has_attachment: bool, attachment_names: tuple[str, ...],
                 has_history: bool, project_basename: str, registered: frozenset[str]):
        self.message = message
        self.body = ""
        self.has_attachment = bool(has_attachment or attachment_names)
        self.attachment_names = tuple(str(n) for n in attachment_names if n)
        self.has_history = bool(has_history)
        self.basename_rx = _it_basename_rx(project_basename)
        self.registered = registered
        self.fenced = "```" in message
        self.source = "lexicon"


# Curly quotes typed on a phone must not defeat "don't" and "what's".
_IT_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
# How much of a paragraph counts as the instruction. Rules read the request,
# not the material pasted after it.
_IT_HEAD_CHARS = 400
# Project folder names too generic to be evidence of anything: "review the app"
# is already a project term, and "check main" is not a request about a repo.
_IT_GENERIC_BASENAMES = {
    "app", "apps", "src", "code", "project", "projects", "repo", "repos", "test",
    "tests", "main", "home", "documents", "desktop", "downloads", "github", "workspace",
    "work", "dev", "tmp", "temp", "source", "lib", "server", "client", "web", "api",
    "data", "docs", "site", "backend", "frontend",
}


def _it_basename_rx(basename: str) -> re.Pattern | None:
    name = (basename or "").strip().strip("/").rsplit("/", 1)[-1]
    if len(name) < 3 or name.lower() in _IT_GENERIC_BASENAMES:
        return None
    # Hyphens and dots are part of a folder name ("mlx-h"), so the boundary is
    # "not a name character" rather than \b, which would match "mlx" in "mlx-h".
    return re.compile(rf"(?<![\w.-]){re.escape(name)}(?![\w-])", re.I)


def _it_clean(text: str) -> str:
    return " ".join((text or "").translate(_IT_QUOTES).split())


def _it_heads(message: str) -> list[tuple[str, str]]:
    """The instruction candidates of a message, each with the rest as its body.

    The first paragraph is the usual place for the request ("add this to skills"
    followed by a pasted SKILL.md); the last paragraph is the other ("<pasted
    text>" then "save this as a skill"). A code fence ends a paragraph, so the
    lexicon never reads code as an instruction.
    """
    raw = (message or "").translate(_IT_QUOTES).strip()
    if not raw:
        return []
    fence = raw.find("```")
    prose = raw[:fence] if fence >= 0 else raw
    paras = [p for p in re.split(r"\n\s*\n", prose) if p.strip()]
    if not paras:
        return []
    out: list[tuple[str, str]] = []
    first = _it_clean(paras[0])
    rest = raw[len(paras[0]):].strip()
    out.append((first[:_IT_HEAD_CHARS], f"{first[_IT_HEAD_CHARS:]}\n{rest}".strip()))
    if len(paras) > 1 and fence < 0:
        last = _it_clean(paras[-1])
        if len(last) <= _IT_HEAD_CHARS:
            out.append((last, raw[:raw.rfind(paras[-1])].strip()))
    return out


def _it_rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.I)


def _it_any(patterns: tuple[re.Pattern, ...], text: str) -> re.Match | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match
    return None


# --------------------------------------------------------------------------- #
# Shared guards.
# --------------------------------------------------------------------------- #
# A question about how something is done is a request for an explanation, even
# when it names a feature: "how do I add a user in linux", "what does git
# status do". The action rules (admin, disabled capability) stay silent on it.
_IT_HOWTO = _it_rx(
    r"^(?:please\s+)?(?:how\s+(?:do|can|could|should|would|to|does|is|are)\b|how's\b|why\b|"
    r"what(?:'s|\s+is|\s+are)\s+(?:the\s+)?(?:best\s+|right\s+|correct\s+|proper\s+)?"
    r"(?:way|command|syntax|difference)\b|is\s+it\s+possible\b|explain\b|"
    r"tell\s+me\s+(?:how|why|about)\b|can\s+i\b|should\s+i\b|what\s+does\b|when\s+should\b)")
# The narrow form for rules whose own phrasing is a question ("what does the
# knowledge base say", "tell me about the chat we had last week"): only a
# request for instructions.
_IT_ASK_HOW = _it_rx(r"^(?:please\s+)?(?:how\s+(?:do|can|could|should|would)\s+(?:i|we|you)\b|"
                     r"how\s+to\b|why\b|can\s+i\b|should\s+i\b)")
# Another system named in the request: the user is asking about their own
# stack, not about this app ("create a new user in postgres", "generate an api
# key for openai", "make a backup script in bash").
_IT_FOREIGN = _it_rx(
    r"\b(?:python|javascript|typescript|golang|c\+\+|c#|ruby|php|kotlin|"
    r"bash|zsh|powershell|sql|django|flask|fastapi|react|vue|angular|node\.?js|"
    r"laravel|postgres(?:ql)?|mysql|sqlite|mongo(?:db)?|redis|linux|ubuntu|"
    r"debian|windows|macos|ios|android|aws|azure|gcp|docker|kubernetes|k8s|github|gitlab|"
    r"bitbucket|jira|keycloak|auth0|okta|firebase|supabase|ldap|active\s+directory|"
    r"openai|anthropic|stripe|slack|discord|telegram|twilio|hugging\s?face|pytorch|"
    r"tensorflow|transformers|sklearn|scikit|keras|nginx|apache|wordpress|shopify|"
    r"vs\s?code|cursor|claude\s+desktop|chatgpt\s+app|iphone|ipad)\b")
# The user's own product: "add a user called bob in my app" is a coding task.
_IT_THEIR_APP = _it_rx(r"\b(?:in|to|for|on)\s+(?:my|our|the\s+client's)\s+(?:app|application|site|"
                       r"website|project|service|product|backend|frontend|api)\b")
# Writing code is the model's own job and always available; a request that
# asks for code next to an action is not a request for the action.
_IT_WRITE = _it_rx(r"\b(?:write|implement|create|generate|add|build|draft|refactor)\b")


# --------------------------------------------------------------------------- #
# Tool subsets, honest about what is switched off.
# --------------------------------------------------------------------------- #
# The tools without which an intent cannot do its job. When none of them is
# registered, the turn would run a tool lane with nothing but final_answer, so
# the intent becomes a reply that says what is off instead.
_IT_REQUIRED: dict[str, tuple[str, ...]] = {
    "skill_save": ("save_skill",),
    "skill_use": ("load_skill",),
    "skill_list": ("list_skills", "search_skills"),
    "memory_write": ("remember",),
    "memory_forget": ("forget",),
    "memory_read": ("recall_memory",),
    "history": ("search_past_conversations",),
    "kb_write": ("index_docs", "index_url"),
    "kb_overview": ("kb_overview", "search_docs"),
    "kb_query": ("search_docs", "kb_overview"),
    "codebase_lookup": ("search_files", "read_file", "list_files"),
    "codebase_review": ("repo_map", "list_files", "search_files", "read_file"),
    # Writing is the job: with neither write tool registered the turn could
    # only describe a README it cannot save, so it says what is off instead.
    "codebase_write": ("write_file", "edit_file"),
    "schedule": ("task_scheduler",),
}

_IT_INTENT_WHAT = {
    "skill_save": "Saving skills", "skill_use": "Loading a skill",
    "skill_list": "Listing your skills", "memory_write": "Saving notes to memory",
    "memory_forget": "Forgetting saved notes", "memory_read": "Reading your saved notes",
    "history": "Searching past conversations", "kb_write": "Adding to the knowledge base",
    "kb_overview": "Browsing the knowledge base", "kb_query": "Searching the knowledge base",
    "codebase_lookup": "Searching the codebase", "codebase_review": "Reading the codebase",
    "codebase_write": "Writing files in the project", "schedule": "Scheduling tasks",
}

_IT_O365 = "the Office 365 settings (O365_TENANT_ID, O365_CLIENT_ID, O365_CLIENT_SECRET)"
_IT_CATALOGUE = "Settings > Tools > Catalogue"


def _it_switch(tool: str) -> str:
    """The lever that registers a tool, in the words the Settings page uses."""
    if tool in ("run_shell", "run_tests"):
        return "ALLOW_SHELL=1"
    if tool in ("run_python", "execute_code"):
        return "ALLOW_PYTHON=1"
    if tool == "code_interpreter":
        return "ALLOW_PYTHON=1 and code_interpreter in EXTRA_TOOLS"
    if tool.startswith("o365_"):
        return _IT_O365
    if tool in ("email", "email_send", "calendar", "calendar_create", "contacts"):
        return f"{tool} in EXTRA_TOOLS plus {_IT_O365}"
    if tool == "image_generate":
        return "image_generate in EXTRA_TOOLS (it needs mflux or IMAGE_GEN_COMMAND)"
    if tool == "image_analyze":
        return "image_analyze in EXTRA_TOOLS (it needs a vision model)"
    if tool == "database":
        return "database in EXTRA_TOOLS with DATABASE_URLS"
    if tool in ("load_skill", "search_skills", "save_skill", "list_skills"):
        return "SKILLS_ENABLED=1 (Settings > Memory > Skills)"
    if tool == "delegate_task":
        return "DELEGATION_ENABLED=1 (Settings > Agent > Delegation)"
    if tool in ("search_docs", "index_docs", "index_url", "kb_overview"):
        return ("a knowledge base with SQLite full-text search, and the tool allowed for "
                "this agent (AGENT_TOOLS or its profile)")
    if tool in EXTRA_TOOL_NAMES:
        return f"{tool} in EXTRA_TOOLS"
    return "the agent's tool list (AGENT_TOOLS, or the agent profile in Settings > Admin > Agents)"


def _it_tool_list(tools: tuple[str, ...]) -> str:
    names = [f"`{t}`" for t in tools]
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


# Tools INTENT_TOOL_SUBSETS names ahead of their implementation (P3).
# kb_overview shipped in phase B, so a reply about it being off names it.
_IT_NOT_YET = frozenset({"repo_map", "list_skills"})


def _it_unavailable(name: str, ctx: _ItCtx, matched: str, targets: tuple[str, ...]) -> Intent:
    """A tools-lane intent whose tools are all switched off, as an honest reply."""
    required = _IT_REQUIRED.get(name, ())
    what = _IT_INTENT_WHAT.get(name, "That")
    if name == "schedule":
        # Scheduling itself is not off: the Tasks view does it without the chat
        # tool. So the intent keeps its name and only the lane changes, and the
        # trace reads "schedule" rather than a missing capability.
        reply = ("Scheduled tasks are not enabled as a chat tool here, so I have not scheduled "
                 "anything. You can create one in the Tasks view with New task; an admin can "
                 f"also enable `task_scheduler` with EXTRA_TOOLS in {_IT_CATALOGUE}.")
        return Intent(name="schedule", lane="reply", retrieval="off", reply=reply,
                      targets=targets, source=ctx.source, matched=matched[:80],
                      extras={"requested": name, "missing_tools": list(required)})
    else:
        # Tools from a later phase are not named: "repo_map is not enabled"
        # would send an admin looking for a switch that does not exist yet.
        named = tuple(t for t in required if t not in _IT_NOT_YET) or required
        verb = "is" if len(named) == 1 else "are"
        reply = (f"{what} is not available in this chat because {_it_tool_list(named)} {verb} "
                 f"not enabled. It needs {_it_switch(named[0])}; an admin can check which tools "
                 f"are on in {_IT_CATALOGUE}.")
    return Intent(name="disabled_capability", lane="reply", retrieval="off", reply=reply,
                  targets=targets, source=ctx.source, matched=matched[:80],
                  extras={"requested": name, "missing_tools": list(required)})


def _it_tools(name: str, ctx: _ItCtx, matched: str, *, retrieval: str = "off",
              referent: str = "", confirm: tuple[str, ...] = (),
              targets: tuple[str, ...] = (), extras: dict | None = None) -> Intent:
    """A tools-lane intent, its subset narrowed to what is registered."""
    subset = INTENT_TOOL_SUBSETS[name]
    if ctx.registered:
        required = _IT_REQUIRED.get(name, ())
        if required and not any(t in ctx.registered for t in required):
            return _it_unavailable(name, ctx, matched, targets)
        # final_answer is the loop's exit and always registered; keep it even
        # when a caller passes a registered set that leaves it out.
        subset = tuple(t for t in subset if t in ctx.registered or t == "final_answer")
    return Intent(name=name, lane="tools", tool_subset=subset, retrieval=retrieval,
                  targets=targets, confirm_tools=tuple(t for t in confirm if t in subset),
                  referent=referent, source=ctx.source, matched=matched[:80],
                  extras=dict(extras or {}))


def _it_reply(name: str, ctx: _ItCtx, matched: str, reply: str, **extras) -> Intent:
    return Intent(name=name, lane="reply", retrieval="off", reply=reply, source=ctx.source,
                  matched=matched[:80], extras=extras)


def _it_clarify(name: str, ctx: _ItCtx, matched: str, question: str,
                targets: tuple[str, ...] = ()) -> Intent:
    return Intent(name=name, lane="clarify", retrieval="off", clarify_question=question,
                  referent="none", targets=targets, source=ctx.source, matched=matched[:80])


# --------------------------------------------------------------------------- #
# conv_action: the conversation's own buttons. They act on the client and on
# rows the model never sees, so the model must not pretend to do them ("export
# this conversation as markdown" used to reach the document lane, which wrote a
# .md from memory instead of the real transcript).
# --------------------------------------------------------------------------- #
_IT_CONV = _it_rx(
    r"^(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?"
    r"(?P<verb>export|download|regenerate|redo|retry|pin|unpin|rename|fork|duplicate|delete|"
    r"remove|archive)\s+(?:(?:this|the|our|my|current|last|previous|whole|entire)\s+){1,3}"
    r"(?P<noun>chat|conversation|thread|answer|response|reply)\b")
_IT_NEW_CHAT = _it_rx(r"^(?:please\s+)?(?:start|open|begin)\s+a\s+(?:new|fresh)\s+"
                      r"(?:chat|conversation)\s*[.!]?\s*$")
_IT_MENU = "the conversation menu (the ... button at the top of the chat)"


def _it_conv_action(text: str, ctx: _ItCtx) -> Intent | None:
    if _IT_NEW_CHAT.search(text):
        return _it_reply("conv_action", ctx, text, "Use the New chat button in the header. "
                         "The current conversation stays in History.", action="new")
    match = _IT_CONV.search(text)
    if not match:
        return None
    verb = match.group("verb").lower()
    if match.group("noun").lower() in ("answer", "response", "reply") and verb not in (
            "export", "download", "regenerate", "redo", "retry"):
        return None  # "delete the last answer" is not deleting the conversation
    if verb in ("export", "download"):
        fmt = document_request(text)
        if fmt and fmt != ".md":
            # "export this answer as a pdf" is a document to produce, which the
            # document lane can do; only the Markdown transcript is a button.
            return None
        reply = (f"Use Export as Markdown in {_IT_MENU}; History also has an Export button on "
                 "every conversation. That exports the real transcript, which I cannot "
                 "reproduce from here.")
    elif verb in ("regenerate", "redo", "retry"):
        reply = (f"Use Regenerate on the last answer, or Regenerate last answer in {_IT_MENU}.")
    elif verb in ("pin", "unpin", "rename", "fork", "duplicate"):
        label = {"unpin": "Unpin", "duplicate": "Fork"}.get(verb, verb.capitalize())
        reply = (f"Open History and use {label} on this conversation's card. I cannot change "
                 "conversations from the chat.")
    else:
        reply = ("Open History and use Delete on the conversation's card; it asks you to "
                 "confirm because it cannot be undone. I cannot delete conversations from the "
                 "chat.")
    return _it_reply("conv_action", ctx, match.group(0), reply, action=verb)


# --------------------------------------------------------------------------- #
# admin_action: settings and access control. Never acted on from the chat
# (credentials, other people's access, a model restart that drops answers in
# progress); the reply says where the setting lives. Each pattern ties the
# change verb to its object, so "create a user model class" is not a request
# to create a user.
# --------------------------------------------------------------------------- #
_IT_LOC = {
    "users": "Settings > Admin > Users",
    "tokens": "Settings > Tools > API tokens",
    "approvals": "Settings > Tools > Approvals",
    "clients": "Settings > Tools > Connect a client",
    "mcp": "Settings > Tools > MCP",
    "cluster": "Settings > Admin > Cluster",
    "backup": "Settings > Admin > Backup",
    "model": "Settings > Models & Training > Model",
    "training": "Settings > Models & Training > Training",
    "knowledge": "Settings > Models & Training > Knowledge",
    "system": "Settings > Admin > System",
    "import": "Settings > Admin > Import",
    "dashboards": "Settings > Admin > Dashboards",
    "evals": "Settings > Admin > Evals",
    "appearance": "Settings > Appearance",
    "generation": "Settings > Generation",
    "tools": _IT_CATALOGUE,
    "memory": "Settings > Memory",
    "agent": "Settings > Agent",
    "admin": "Settings > Admin",
    "settings": "Settings",
}
_IT_ADMIN_ONLY = {"users", "mcp", "cluster", "backup", "model", "training", "knowledge",
                  "system", "import", "dashboards", "evals", "admin"}

_IT_ADMIN_RULES: tuple[tuple[str, re.Pattern, str], ...] = (
    ("users", _it_rx(
        r"^(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?(?:add|create|invite|delete|remove|"
        r"disable|enable|block|unblock)\s+(?:a\s+|an\s+|the\s+|new\s+|another\s+)*"
        r"(?:admin\s+|user\s+)?(?:user|account)s?\b"
        r"(?=\s*$|\s*[,:.!]|\s+(?:called|named|for|with|whose)\b|\s+[\w.+-]+@)"),
     "I cannot manage user accounts from the chat. An admin adds, changes and removes "
     "users in {loc}."),
    ("tokens", _it_rx(
        r"\b(?:create|generate|make|issue|mint|revoke|delete|rotate|give\s+me|get\s+me)\s+"
        r"(?:a\s+|an\s+|the\s+|my\s+|new\s+|another\s+)*(?:api|access|bearer|personal)[- ]?"
        r"(?:access\s+)?(?:token|key)s?\b(?!\s+(?:generator|class|function|validation|"
        r"validator|middleware|field|column|table|model|rotation))"),
     "I cannot create or revoke API tokens from the chat, since they are credentials. "
     "Create one in {loc}; the token is shown once there, so copy it then."),
    ("approvals", _it_rx(
        r"^(?:please\s+)?(?:always|auto(?:matically)?)\s+(?:allow|approve)\s+(?:the\s+)?(?:[a-z]+_\w+|"
        r"[\w-]+\s+tool\b|tool\s+[\w-]+|everything\b|all\s+tools\b|(?:it|this|that)\s*[.!]?\s*$)"
        r"|\b(?:stop|don't|dont|never)\s+ask(?:ing)?\s+(?:me\s+)?(?:for\s+)?(?:approval|"
        r"permission|confirmation|to\s+(?:approve|confirm))\b"
        r"|\b(?:turn|switch)\s+off\s+(?:the\s+)?(?:tool\s+)?approvals?\b"
        r"|\b(?:add|create|change|edit|set)\s+(?:a\s+|an\s+|the\s+)?(?:tool|approval)\s+rules?\b"),
     "Tool approvals are not changed from the chat. The scope buttons on an approval card "
     "grant standing permission for a tool, and admins edit the rules in {loc}."),
    ("clients", _it_rx(
        r"\bconnect\s+(?:claude(?:\s+(?:code|desktop))?|cursor|vs\s?code|codex|cline|zed|"
        r"windsurf|continue|an?\s+(?:mcp\s+)?client|my\s+(?:editor|ide))\b.{0,40}?\b(?:to|with)"
        r"\s+(?:this\s+(?:app|server)|you|the\s+(?:app|server))\b"),
     "Connecting a client is set up outside the chat: {loc} has the steps and the "
     "configuration to copy, and an admin manages the MCP endpoint in Settings > Tools > MCP."),
    ("mcp", _it_rx(
        r"\b(?:add|configure|connect|set\s+up|enable|disable|remove|install|register)\b.{0,30}?"
        r"\bmcp(?:\s+(?:servers?|config(?:uration)?|tools?))?\b"),
     "MCP servers are configured by an admin in {loc}; I cannot change that configuration "
     "from the chat."),
    ("cluster", _it_rx(
        r"\b(?:add|join|remove|connect|pair|attach)\s+(?:a\s+|an\s+|another\s+|the\s+|this\s+|"
        r"my\s+|new\s+)*(?:node|mac(?:\s+(?:mini|studio))?|machine|worker|computer)s?\s+"
        r"(?:to|from|with)\s+(?:the\s+|this\s+|my\s+)?cluster\b"
        r"|\b(?:enable|disable|set\s+up|configure|turn\s+on|turn\s+off)\s+(?:the\s+)?"
        r"(?:cluster(?:ing)?|cluster\s+mode|layer\s+sharding)\b"),
     "Cluster nodes are managed by an admin in {loc}; I cannot change the cluster from the "
     "chat."),
    ("backup", _it_rx(
        r"^(?:please\s+)?(?:can\s+you\s+)?(?:make|create|take|run|do|start)\s+(?:a\s+|an?\s+full\s+|"
        r"the\s+)?backup\b(?!\s+(?:script|strategy|plan|policy|of\s+my\s+(?:phone|mac|laptop|"
        r"computer|files|photos)))"
        r"|^(?:please\s+)?back\s+(?:it\s+)?up\s+(?:everything|the\s+(?:app|data|database|server|"
        r"settings|conversations)|my\s+(?:data|conversations|settings)|all)\b"
        r"|^(?:please\s+)?restore\s+(?:from\s+)?(?:the\s+|a\s+|my\s+|last\s+|latest\s+)*backup\b"),
     "Backups are made and restored by an admin in {loc}; I do not run them from the chat, "
     "and a restore replaces the current data."),
    ("model", _it_rx(
        r"^(?:please\s+)?(?:can\s+you\s+)?(?:switch|change|swap|move)\s+(?:over\s+)?to\s+"
        r"(?:the\s+|a\s+|another\s+|a\s+different\s+)?(?:[\w.:/-]+\s+){0,3}model\b"
        r"|^(?:please\s+)?(?:restart|reload|reboot|stop|start|kill)\s+(?:the\s+|your\s+)?"
        r"(?:model(?:\s+server)?|llm(?:\s+server)?|mlx\s+server|inference\s+server)\b"
        r"|^(?:please\s+)?(?:load|unload|serve)\s+(?:the\s+)?(?:[\w.:/-]+\s+){0,3}model\s*"
        r"(?:instead|now)?\s*[.!]?\s*$"),
     "I cannot switch or restart the model from the chat: it would drop answers in "
     "progress. An admin does this in {loc}."),
    ("training", _it_rx(
        r"^(?:please\s+)?re-?train\s+(?:yourself|the\s+model|your\s+model|now)\b"
        r"|^(?:please\s+)?(?:re-?train|fine[- ]?tune|train)\b.{0,40}?\b(?:on|with|from)\s+"
        r"(?:my|our)\s+(?:feedback|ratings|corrections|conversations|chats)\b"),
     "Retraining is started by an admin in {loc}; I do not start it from the chat."),
    ("system", _it_rx(
        r"\b(?:enable|disable|turn\s+(?:on|off)|set\s+up|open|configure|allow)\s+(?:the\s+)?"
        r"remote\s+access\b"
        r"|^(?:please\s+)?(?:enable|disable|turn\s+(?:on|off)|set\s+up|configure|switch\s+"
        r"(?:on|off))\s+(?:the\s+)?(?:sign[- ]?in|log[- ]?in|oidc|sso|entra(?:\s+id)?|"
        r"microsoft\s+sign[- ]?in|authentication)\b"
        r"|^(?:please\s+)?(?:can\s+you\s+)?(?:change|reset|set)\s+(?:the\s+)?admin\s+password\b"),
     "Sign-in and remote access are server settings an admin changes in {loc}; I cannot "
     "change them from the chat."),
    ("knowledge", _it_rx(
        r"^(?:please\s+)?(?:clear|wipe|delete|empty|reset|purge|erase)\s+(?:the\s+|my\s+)?"
        r"(?:whole\s+|entire\s+)?(?:knowledge[- ]?base|kb|document\s+index)\b"),
     "I do not clear the knowledge base from the chat because it cannot be undone. An admin "
     "can clear or re-index it in {loc}."),
    ("import", _it_rx(
        r"^(?:please\s+)?import\s+(?:my\s+|our\s+|the\s+)?(?:chatgpt|claude|openai)\s+"
        r"(?:chats?|conversations?|history|export|data)\b"
        r"|^(?:please\s+)?import\s+(?:my\s+)?(?:chat|conversation)\s+history\b"),
     "Importing chat history needs the export file, so it is done in {loc}."),
    ("dashboards", _it_rx(
        r"^(?:please\s+)?(?:open|show(?:\s+me)?|go\s+to|take\s+me\s+to)\s+(?:the\s+|my\s+)?"
        r"(?:admin\s+)?dashboards?\s*[.!]?\s*$"),
     "The dashboards are in {loc}."),
    ("evals", _it_rx(
        r"^(?:please\s+)?run\s+(?:the\s+)?(?:evals?|eval(?:uation)?\s+suites?|evaluations?)"
        r"\s*[.!]?\s*$"),
     "The evaluation suites run from {loc}; I do not start them from the chat, since a run "
     "holds the model for minutes."),
    ("appearance", _it_rx(
        r"^(?:please\s+)?(?:turn\s+on|turn\s+off|switch\s+(?:on|off|to)|enable|disable|use|"
        r"change\s+to|go)\s+(?:the\s+)?(?:dark|light)\s+(?:mode|theme)\s*[.!]?\s*$"),
     "The theme is your own setting in {loc}; it applies as soon as you change it there."),
    ("generation", _it_rx(
        r"^(?:please\s+)?(?:set|change|lower|raise|increase|decrease|put)\s+(?:the\s+|your\s+|"
        r"my\s+)?(?:sampling\s+)?temperature\s+(?:to\s+)?(?:[01](?:\.\d+)?|2(?:\.0+)?|\.\d+)"
        r"\s*[.!]?\s*$"),
     "Temperature is set in {loc} (your own default); I cannot change settings from the "
     "chat."),
)
# "open the knowledge base settings": where a settings page is, by its name.
_IT_SETTINGS_NAV = _it_rx(
    r"^(?:please\s+)?(?:open|show(?:\s+me)?|go\s+to|take\s+me\s+to|bring\s+up|where\s+(?:is|are))"
    r"\s+(?:the\s+|my\s+)?(?:(?P<what>[\w&-]+(?:\s+[\w&-]+){0,2})\s+)?(?:settings|preferences)"
    r"(?:\s+(?:page|panel|tab|screen))?\s*[.!?]?\s*$")
_IT_SETTINGS_WORDS = (
    (re.compile(r"knowledge|kb|rag|retrieval", re.I), "knowledge"),
    (re.compile(r"tools?|catalogue|catalog", re.I), "tools"),
    (re.compile(r"approvals?", re.I), "approvals"),
    (re.compile(r"mcp", re.I), "mcp"),
    (re.compile(r"tokens?", re.I), "tokens"),
    (re.compile(r"models?|training", re.I), "model"),
    (re.compile(r"memory|skills?|notes", re.I), "memory"),
    (re.compile(r"appearance|theme|display", re.I), "appearance"),
    (re.compile(r"generation|temperature", re.I), "generation"),
    (re.compile(r"agent|routing", re.I), "agent"),
    (re.compile(r"users?|accounts?", re.I), "users"),
    (re.compile(r"admin", re.I), "admin"),
)


def _it_admin_location(area: str) -> str:
    loc = _IT_LOC.get(area, "Settings")
    return loc + (" (visible to admins)" if area in _IT_ADMIN_ONLY else "")


def _it_admin_action(text: str, ctx: _ItCtx) -> Intent | None:
    if _IT_HOWTO.search(text) or ctx.fenced:
        return None
    nav = _IT_SETTINGS_NAV.search(text)
    if nav:
        what = nav.group("what") or ""
        area = next((a for rx, a in _IT_SETTINGS_WORDS if rx.search(what)), "settings")
        if what and area == "settings":
            return None  # "show me the wifi settings" is not this app
        reply = ("Settings opens from the header, next to New chat." if area == "settings"
                 else f"You will find that in {_it_admin_location(area)}.")
        return _it_reply("admin_action", ctx, nav.group(0), reply, area=area,
                         location=_IT_LOC[area])
    if _IT_FOREIGN.search(text) or _IT_THEIR_APP.search(text):
        return None
    for area, pattern, template in _IT_ADMIN_RULES:
        match = pattern.search(text)
        if match:
            if area == "mcp" and re.search(r"\b(?:to|in|into)\s+(?:claude|cursor|vs\s?code|zed|"
                                           r"windsurf|my\s+(?:editor|ide))\b", text, re.I):
                return None  # configuring another app's MCP client, not this one
            return _it_reply("admin_action", ctx, match.group(0),
                             template.format(loc=_it_admin_location(area)),
                             area=area, location=_IT_LOC[area])
    return None


# --------------------------------------------------------------------------- #
# self_describe: "what can you do". The prose lane has no tool list in its
# prompt, so the model used to invent capabilities; the registry knows.
# --------------------------------------------------------------------------- #
_IT_SELF = _it_rx(
    r"\bwhat\s+(?:tools|capabilities|abilities|functions|features)\s+do\s+you\s+(?:have|support)\b"
    r"|\bwhat\s+can\s+you\s+(?:do|help(?:\s+me)?\s+with)(?:\s+for\s+me|\s+here|\s+in\s+this\s+"
    r"(?:app|chat))?\s*[?.!]*\s*$"
    r"|\bwhich\s+tools\s+(?:do\s+you\s+have|can\s+you\s+use|are\s+(?:available|enabled|on))\b"
    r"|\bwhat\s+are\s+your\s+(?:capabilities|tools|abilities)\b"
    r"|\bwhat\s+are\s+you\s+(?:capable\s+of|able\s+to\s+do)\s*[?.!]*\s*$"
    r"|\b(?:list|show(?:\s+me)?)\s+(?:your|the|all)\s+(?:available\s+|enabled\s+)?tools\b"
    r"|^(?:help|capabilities|tools)\s*[?.!]?\s*$")

# Tools the capability groups do not name yet, with the label they belong under.
_IT_EXTRA_LABELS = {
    "load_skill": "Skills", "search_skills": "Skills", "save_skill": "Skills",
    "list_skills": "Skills", "search_past_conversations": "Conversation history",
    "delegate_task": "Delegation to helper agents", "index_docs": "Knowledge base",
    "index_url": "Knowledge base", "kb_overview": "Knowledge base",
    "repo_map": "File operations", "calculator": "Utilities", "current_time": "Utilities",
    "weather": "Utilities",
}


def _it_group_switch(key: str, tools: list[str]) -> str:
    if key == "code_exec":
        return "ALLOW_SHELL / ALLOW_PYTHON"
    if key in ("shell",):
        return "ALLOW_SHELL"
    if key == "code_interpreter":
        return "ALLOW_PYTHON and EXTRA_TOOLS"
    if key in ("office365",):
        return "the Office 365 settings"
    if key in ("email", "calendar", "contacts"):
        return "EXTRA_TOOLS and the Office 365 settings"
    if tools and all(t in EXTRA_TOOL_NAMES for t in tools):
        return "EXTRA_TOOLS"
    return "AGENT_TOOLS or the agent profile"


def _it_capabilities_text(registered: frozenset[str]) -> str:
    """A deterministic description of what this chat can do, from the registry."""
    groups = CAPABILITY_GROUPS if isinstance(CAPABILITY_GROUPS, dict) else {}
    assigned: set[str] = set()
    lines: list[tuple[str, list[str]]] = []
    by_label: dict[str, list[str]] = {}

    def _add(label: str, names: list[str]) -> None:
        if label in by_label:
            by_label[label].extend(names)
        else:
            by_label[label] = list(names)
            lines.append((label, by_label[label]))
        assigned.update(names)

    for key, group in groups.items():
        names = [t for t in (group.get("tools") or []) if t in registered and t not in assigned]
        if names:
            _add(str(group.get("label") or key), names)
    mcp = sorted(t for t in registered if t.startswith("mcp_") and t not in assigned)
    if mcp:
        _add(str((groups.get("mcp") or {}).get("label") or "MCP servers"), mcp)
    for tool in sorted(registered):
        if tool in assigned or tool == "final_answer":
            continue
        _add(_IT_EXTRA_LABELS.get(tool, "Other tools"), [tool])

    off: list[str] = []
    off_tools: list[set[str]] = []
    for key, group in groups.items():
        tools = list(group.get("tools") or [])
        if not tools or any(t in registered for t in tools):
            continue
        if any(set(tools) <= seen for seen in off_tools):
            continue  # "Shell" is already covered by "Run code & shell"
        off_tools.append(set(tools))
        off.append(f"{group.get('label') or key} ({_it_group_switch(key, tools)})")

    parts = ["Here is what I can do in this chat, from the tools that are switched on:", ""]
    parts += [f"- {label}: {', '.join(f'`{t}`' for t in names)}" for label, names in lines]
    parts += ["", "Without a tool I can also answer from my own knowledge, explain things, and "
                  "write code or text."]
    if off:
        parts += ["", f"Switched off here: {', '.join(off)}. An admin can turn these on in "
                      f"{_IT_CATALOGUE}."]
    return "\n".join(parts)


def _it_self_describe(text: str, ctx: _ItCtx) -> Intent | None:
    match = _IT_SELF.search(text)
    if not match or not ctx.registered:
        return None  # without the registry there is nothing true to say
    return _it_reply("self_describe", ctx, match.group(0),
                     _it_capabilities_text(ctx.registered))


# --------------------------------------------------------------------------- #
# schedule: recurring or timed work. Needs a scheduling verb or a task noun
# together with a recurrence; "what do people eat every day" and "I run every
# morning" are not requests.
# --------------------------------------------------------------------------- #
_IT_RECUR = (r"(?:every\s+(?:day|morning|evening|night|afternoon|hour|week|weekday|month|"
             r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
             r"\d+\s+(?:minutes?|hours?|days?|weeks?))|each\s+(?:day|morning|evening|night|week|"
             r"monday|tuesday|wednesday|thursday|friday)|daily|hourly|weekly|nightly|monthly)")
_IT_SCHEDULE = (
    _it_rx(r"\bremind\s+me\s+(?:to\b|at\b|in\s+\d|on\s+\w+day\b|tomorrow\b|tonight\b|every\b|"
           r"each\b|next\b|later\b)"),
    _it_rx(r"\bschedule\s+(?:a\s+|an\s+|the\s+|this\s+|my\s+)?(?:(?:daily|weekly|hourly|nightly|"
           r"monthly|recurring|regular)\s+)?(?:task|job|run|search|check|report|reminder|summary|"
           r"digest|scan|backup)\b"),
    _it_rx(r"\bschedule\s+(?:this|it|that)\b(?!\s+(?:meeting|call|appointment))"),
    _it_rx(r"\b(?:create|set\s+up|make|add|start)\s+(?:a\s+|an\s+)?(?:new\s+)?(?:recurring\s+|"
           r"scheduled\s+|daily\s+|weekly\s+|hourly\s+|nightly\s+)?(?:task|job|reminder)\b.{0,60}?"
           rf"(?:{_IT_RECUR}|\bat\s+\d|\btomorrow\b|\btonight\b|\bon\s+\w+day\b)"),
    _it_rx(rf"^(?:please\s+)?{_IT_RECUR}\b,?\s+(?:please\s+)?(?:search|check|look\s+up|fetch|"
           r"summari[sz]e|send|collect|give\s+me|compile|gather|run|post|notify|tell\s+me)\b"),
    _it_rx(r"^(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?(?:search|check|look\s+up|fetch|"
           r"summari[sz]e|send|collect|give\s+me|compile|gather|post|notify)\b.{0,80}?\b"
           rf"{_IT_RECUR}\b"),
)
# Scheduling inside another system is a coding or ops question ("a cron job
# that runs every hour", "a python script that runs daily").
_IT_SCHEDULE_FOREIGN = _it_rx(r"\b(?:cron(?:tab)?|celery|airflow|systemd|launchd|github\s+actions|"
                              r"kubernetes|k8s|lambda|windows|linux|macos|ubuntu|script|function|"
                              r"code|meeting|appointment)\b")


# "a daily workout routine", "my weekly report": the recurrence describes a
# thing, it does not ask for one to run.
_IT_RECUR_ANY = _it_rx(rf"\b{_IT_RECUR}\b")
_IT_RECUR_ADJ = _it_rx(r"\b(?:a|an|the|my|your|our|their|this|that|some)\s+"
                       r"(?:daily|hourly|weekly|nightly|monthly)\b")


def _it_schedule(text: str, ctx: _ItCtx) -> Intent | None:
    if _IT_HOWTO.search(text) or ctx.fenced:
        return None
    match = _it_any(_IT_SCHEDULE, text)
    if not match or _IT_SCHEDULE_FOREIGN.search(text):
        return None
    if match.re in _IT_SCHEDULE[-2:] and \
            len(_IT_RECUR_ADJ.findall(text)) >= len(_IT_RECUR_ANY.findall(text)):
        # The two verb-plus-recurrence rules only: "schedule a daily report"
        # names the scheduling itself and stays a schedule.
        return None
    return _it_tools("schedule", ctx, match.group(0), confirm=("task_scheduler",),
                     targets=("tasks",))


# --------------------------------------------------------------------------- #
# disabled_capability: requests for a tool this server has switched off. The
# honest answer is "not enabled, here is the switch", never a guess and never a
# claim to have run something. Only when the registry is known: an empty
# registered_tools means "not told", not "nothing is on".
# --------------------------------------------------------------------------- #
_IT_CAPABILITIES: tuple[tuple[str, str, tuple[str, ...], str, bool, re.Pattern], ...] = (
    # key, label, tools (any registered = enabled), what was not done, write-guarded, pattern
    ("tests", "Running the test suite", ("run_tests", "run_shell"), "run anything", True, _it_rx(
        r"^(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?(?:run|execute|rerun|re-run)\s+"
        r"(?:the\s+|my\s+|all\s+(?:the\s+)?|our\s+|these\s+|unit\s+|integration\s+)*"
        r"(?:tests?|test\s+suite|specs?|pytest|unit\s+tests?)\b"
        r"|^(?:please\s+)?pytest\b"
        r"|\b(?:do|does|did)\s+(?:the\s+|all\s+(?:the\s+)?|my\s+)?tests?\s+(?:still\s+)?pass\b")),
    ("shell", "Running shell commands", ("run_shell",), "run anything", True, _it_rx(
        r"^(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?(?:run|execute)\s+(?:this\s+|the\s+|"
        r"these\s+|that\s+|the\s+following\s+|a\s+|my\s+)?(?:shell|terminal|bash|zsh)\s+"
        r"(?:commands?|script|one-?liner)\b"
        r"|^(?:please\s+)?(?:can\s+you\s+)?(?:run|execute)\s+(?:this|the\s+following|these)\s+"
        r"commands?\b"
        r"|^(?:please\s+)?(?:can\s+you\s+)?(?:run|execute)\s+`[^`]+`"
        r"|^(?:please\s+)?(?:can\s+you\s+)?(?:run|execute)\b.{1,60}?\b(?:in|on|from)\s+"
        r"(?:the|my|a)\s+(?:shell|terminal|command\s+line)\b")),
    ("python", "Running Python", ("run_python", "execute_code", "code_interpreter"),
     "run anything", True, _it_rx(
         r"^(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?(?:run|execute|eval(?:uate)?)\s+"
         r"(?:this|the|my|that|the\s+following|these)\s+(?:python|py)(?:\s+(?:code|script|"
         r"snippet|file|lines?))?\b"
         r"|\brun\s+(?:it|this|that)\s+(?:in|with|using)\s+python\b"
         r"|^(?:please\s+)?(?:can\s+you\s+)?(?:run|execute)\s+[\w./-]+\.py\b")),
    ("code", "Running code", ("run_python", "execute_code", "code_interpreter", "run_shell"),
     "run anything", True, _it_rx(
         r"^(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?(?:run|execute)\s+(?:this|the\s+"
         r"following|that|my)\s+(?:code|script|snippet|program)\b")),
    ("git_commit", "Committing with git", ("git_commit",), "committed anything", False, _it_rx(
        r"^(?:please\s+)?(?:can\s+you\s+)?(?:git\s+)?commit\s+(?:my|the|these|all|this|your|our)"
        r"\s+(?:changes|edits|work|files?|fix|code)\b"
        r"|^(?:please\s+)?(?:make|create)\s+a\s+(?:git\s+)?commit\b(?!\s+message)")),
    ("git", "Git", ("git",), "touched the repository", False, _it_rx(
        r"^(?:please\s+)?(?:can\s+you\s+)?(?:(?:run|show(?:\s+me)?|check|give\s+me|do)\s+"
        r"(?:the\s+|a\s+|my\s+)?)?git\s+(?:status|diff|log|blame|branch(?:es)?)\b"
        r"|^(?:please\s+)?(?:show(?:\s+me)?|give\s+me)\s+(?:the\s+)?(?:commit\s+log|commit\s+"
        r"history|recent\s+commits|last\s+commits?)\b")),
    ("email_send", "Sending email", ("email_send", "o365_mail"), "sent anything", False, _it_rx(
        r"^(?:please\s+)?(?:can\s+you\s+)?(?:send|forward)\s+(?:an?\s+)?e-?mail\s+to\b"
        r"|^(?:please\s+)?(?:can\s+you\s+)?e-?mail\s+(?:it|this|that|these|them)\s+to\b"
        r"|\bsend\s+(?:it|this|that)\s+(?:by|via|as\s+an?)\s+e-?mail\b")),
    ("email", "Email", ("email", "o365_mail"), "read any mail", False, _it_rx(
        r"\b(?:check|read|search|open|show(?:\s+me)?|summari[sz]e|go\s+through|scan)\s+my\s+"
        r"(?:e-?mails?|mail(?:box)?|inbox|outlook)\b"
        r"|\b(?:any|new|unread)\s+(?:new\s+)?e-?mails\b"
        r"|\bwhat(?:'s|\s+is)\s+in\s+my\s+inbox\b")),
    ("calendar_create", "Creating calendar events", ("calendar_create",), "created any event",
     False, _it_rx(
         r"^(?:please\s+)?(?:can\s+you\s+)?(?:schedule|book|set\s+up|add|create|put)\s+(?:a\s+|"
         r"an\s+)?(?:meeting|appointment|call|event)\b(?!\s*(?:listener|handler|emitter|loop|"
         r"bus|driven|sourcing|type|class|log|logger|object|model))")),
    ("calendar", "Calendar", ("calendar", "o365_calendar"), "read your calendar", False, _it_rx(
        r"\bmy\s+(?:calendar|agenda)\b(?!\s+(?:app|application|component|widget|ui|view|page|"
        r"project|code|class))"
        r"|\bmy\s+schedule\s+(?:for\s+)?(?:today|tomorrow|this\s+week)\b")),
    ("contacts", "Contacts", ("contacts",), "looked anything up", False, _it_rx(
        r"\bmy\s+(?:outlook\s+)?contacts\b(?!\s+(?:app|page|component|list\s+component))")),
    # read_file reads .xlsx too (openpyxl), so reading is off only when both are.
    ("spreadsheet_read", "Reading spreadsheets", ("spreadsheet", "read_file"), "read the file",
     False, _it_rx(
        r"^(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?(?:read|open|analy[sz]e|summari[sz]e|"
        r"parse|load|look\s+at|go\s+through)\s+(?:this|the|my|that)\s+(?:spreadsheet|workbook|"
        r"excel\s+(?:file|sheet|workbook)|xlsx\s+file)\b"
        r"|^(?:please\s+)?(?:can\s+you\s+)?(?:read|open|analy[sz]e|summari[sz]e|parse|load)\s+"
        r"[\w./~-]+\.(?:xlsx|xlsm|xls)\b")),
    ("spreadsheet", "Writing into a spreadsheet", ("spreadsheet_write",), "changed the file",
     False, _it_rx(
         r"\b(?:write|put|set|update|fill)\b.{0,30}?\b(?:cells?|range)\s+[A-Z]{1,3}\d+\b"
         r".{0,40}?\.(?:xlsx|xlsm)\b")),
    ("image_generate", "Image generation", ("image_generate",), "generated anything", False,
     _it_rx(
         r"^(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?(?:generate|create|make|draw|render|"
         r"paint|produce)\s+(?:me\s+)?(?:an?\s+|some\s+)?(?:\w+\s+)?(?:image|picture|photo|"
         r"illustration|drawing|painting|artwork)s?\s+(?:of|showing|with|for|that)\b")),
    ("image_analyze", "Image analysis", ("image_analyze",), "looked at the image", False, _it_rx(
        r"\b(?:describe|analy[sz]e|caption|what(?:'s|\s+is)\s+in|read\s+the\s+text\s+in|ocr)\s+"
        r"(?:the\s+(?:image|picture|photo|screenshot)\s+)?[\w./~-]+\.(?:png|jpe?g|gif|webp|"
        r"heic)\b")),
    ("database", "Database queries", ("database",), "queried anything", True, _it_rx(
        r"\b(?:query|search|inspect|look\s+(?:in|into)|check|read)\s+(?:the|my|our|this)\s+"
        r"(?:sqlite\s+)?(?:database|db)\b(?!\s+(?:schema\s+design|design|model))"
        r"|\brun\s+(?:this|the|my)\s+(?:sql|query)\b.{0,30}?\b(?:on|against|in)\b")),
    ("http", "HTTP requests other than GET", ("http_request",), "sent any request", True,
     _it_rx(
         r"^(?:please\s+)?(?:can\s+you\s+)?(?:send|make|do|issue|fire)\s+(?:an?\s+)?(?:http\s+)?"
         r"(?:post|put|patch|delete)\s+request\b")),
)
_IT_ALREADY_SENT = _it_rx(r"\b(?:i|we)\s+(?:just\s+)?(?:uploaded|attached|sent|shared|dropped)\b"
                          r"|\b(?:uploaded|attached)\s+(?:file|spreadsheet|image|picture|photo|"
                          r"workbook)\b")
_IT_CAP_NO_IMAGE = _it_rx(r"\b(?:svg|ascii|mermaid|diagram|chart|plot|graph|matplotlib|html|css|"
                          r"tikz|latex)\b")
_IT_CAP_EXTRA = {
    "git": " For the working-tree diff you can type @diff in the composer.",
}


def _it_disabled_capability(text: str, ctx: _ItCtx) -> Intent | None:
    if not ctx.registered or _IT_HOWTO.search(text) or ctx.fenced:
        return None
    for key, label, tools, not_done, write_guard, pattern in _IT_CAPABILITIES:
        match = pattern.search(text)
        if not match:
            continue
        if any(t in ctx.registered for t in tools):
            return None  # it is on: the normal pipeline handles the request
        if write_guard and _IT_WRITE.search(text):
            return None  # "write and run the tests": the writing part needs no tool
        if key == "image_generate" and _IT_CAP_NO_IMAGE.search(text):
            return None  # an SVG or a chart is code the model can write
        if key in ("image_analyze", "spreadsheet", "spreadsheet_read") and (
                ctx.has_attachment or _IT_ALREADY_SENT.search(text)):
            return None  # the attachment pipeline reads what was sent
        if key in ("calendar", "calendar_create", "contacts") and _IT_FOREIGN.search(text):
            return None  # "my calendar app in react" is a coding request
        if key in ("calendar", "contacts") and _IT_WRITE.search(text):
            return None
        reply = (f"{label} is not enabled on this server, so I have not {not_done}. An admin "
                 f"can turn it on in {_IT_CATALOGUE}; it needs {_it_switch(tools[0])}."
                 + _IT_CAP_EXTRA.get(key, ""))
        return Intent(name="disabled_capability", lane="reply", retrieval="off", reply=reply,
                      source=ctx.source, matched=match.group(0)[:80],
                      extras={"capability": key, "missing_tools": list(tools)})
    return None


# --------------------------------------------------------------------------- #
# Referents: what "this" is. Attachment text first, then the previous answer;
# with neither, one question instead of letting retrieval supply a referent
# (turn 1 of the review saved nothing and re-typed a KB file instead).
# --------------------------------------------------------------------------- #
_IT_INLINE_MIN = 80


def _it_referent(ctx: _ItCtx, named: str = "") -> str:
    if len(ctx.body) >= _IT_INLINE_MIN:
        return "inline"
    if ctx.has_attachment:
        return "attachment"
    if named:
        return "named_path"
    if ctx.has_history:
        return "previous_answer"
    return ""


def _it_skill_md(ctx: _ItCtx) -> str:
    """The attachment that looks like a skill file, if any."""
    for name in ctx.attachment_names:
        if name.lower().endswith(".md"):
            return name
    return ""


# --------------------------------------------------------------------------- #
# Skills.
# --------------------------------------------------------------------------- #
_IT_DEICTIC = (r"(?:this|that|it|these|those|the\s+above|above|the\s+(?:attached|uploaded)\s+"
               r"(?:file|document|text|skill|procedure)|the\s+(?:file|document|procedure|workflow|"
               r"process|steps|instructions|answer|reply|response|text|playbook|checklist|method|"
               r"approach|prompt)|(?:the\s+)?(?P<named>[\w.-]+\.md)|the\s+[\w-]+\s+(?:procedure|"
               r"workflow|process|playbook|checklist))")
_IT_SKILL_SAVE = (
    _it_rx(rf"\b(?:add|save|turn|make|store|keep|put|convert|file|record)\s+{_IT_DEICTIC}\s+"
           r"(?:to|as|into|in|among)\s+(?:(?:a|an|my|the|your|our|new|reusable)\s+)*"
           r"(?:[\w-]+\s+)?skills?\b(?!\s+(?:section|list|tree|set|matrix|gap|level|on\s+my|"
           r"in\s+my))"),
    _it_rx(rf"\b(?:make|create|build|write|draft)\s+(?:(?:a|an|new|reusable)\s+)*skill\s+"
           rf"(?:out\s+of|from|based\s+on)\s+{_IT_DEICTIC}"),
    _it_rx(r"^(?:please\s+)?(?:add|save)\s+(?:to|as)\s+(?:a\s+|my\s+|the\s+)?skills?\s*[.!]?\s*$"),
    _it_rx(r"^(?:please\s+)?(?:make|turn)\s+(?:this|that|it)\s+(?:into\s+)?(?:a|an|my)\s+"
           r"(?:new\s+|reusable\s+)?skill(?![\w-])\s*(?:please)?\s*[.!]?\s*$"),
)
_IT_SKILL_USE = _it_rx(
    r"\b(?:use|apply|follow|load|invoke)\s+(?:the\s+|my\s+|our\s+|your\s+)?"
    r"(?P<name>[\w][\w.-]*(?:\s+[\w][\w.-]*)?)\s+skill\b(?!s)")
_IT_SKILL_USE_NOT = {
    "right", "best", "correct", "appropriate", "same", "new", "other", "relevant", "proper",
    "a", "an", "any", "each", "this", "that", "one", "my", "your", "the", "social", "soft",
    "hard", "people", "communication", "language", "special", "certain", "every",
}
_IT_SKILL_LIST = (
    _it_rx(r"\b(?:list|show(?:\s+me)?|display|what\s+are)\s+(?:all\s+)?(?:of\s+)?(?:my|our|"
           r"(?:the\s+|your\s+)?(?:saved|available|installed|stored))\s+(?:saved\s+)?skills\b"
           r"(?!\s+(?:section|on\s+my|in\s+my))"),
    # "the skills" alone is generic ("list the skills a data engineer needs");
    # it means the library only when nothing qualifies it.
    _it_rx(r"\b(?:list|show(?:\s+me)?|display)\s+(?:all\s+)?(?:of\s+)?the\s+skills\s*"
           r"(?:you\s+have|in\s+(?:the|my|your)\s+(?:skill\s+)?library)?\s*[.!?]?\s*$"),
    _it_rx(r"\b(?:what|which|how\s+many)\s+skills\s+(?:do\s+(?:you|i)\s+have|are\s+(?:saved|"
           r"installed|available|in\s+(?:my|the|your)\s+(?:skill\s+)?library))\b"),
    _it_rx(r"\b(?:my|the|your)\s+skills?\s+library\b"),
    _it_rx(r"^(?:my\s+)?skills\s*\??$"),
)
# A CV lists skills too: "add python to my skills on my resume".
_IT_SKILL_NOT = _it_rx(r"\b(?:resume|r[ée]sum[ée]|cv|linkedin|curriculum|job\s+(?:ad|posting|"
                       r"description)|candidate|hiring|skill\s+tree|character|rpg)\b")
# Importing skills in bulk is an admin action of its own (skills_import, not
# built yet); it must not read as saving one skill or as a KB question.
_IT_SKILL_IMPORT = _it_rx(
    r"\b(?:import|load|sync|copy|bring\s+in)\b.{0,60}?\bskills\b.{0,60}?\b(?:from|out\s+of)\b"
    r"|\b(?:import|load|sync)\s+(?:the\s+)?skills?\s+(?:repo|repository|folder|library)\b")


def _it_skill(text: str, ctx: _ItCtx) -> Intent | None:
    if _IT_SKILL_NOT.search(text) or _IT_SKILL_IMPORT.search(text):
        return None
    match = _it_any(_IT_SKILL_SAVE, text)
    if match:
        named = match.groupdict().get("named") or ""
        if named and any(n.lower() == named.lower() for n in ctx.attachment_names):
            named = ""
        referent = _it_referent(ctx, named)
        if not referent:
            return _it_clarify(
                "skill_save", ctx, match.group(0),
                "What should I save as a skill: attach the file, paste the text, or tell me "
                "which earlier answer to use?", targets=("skills",))
        extras: dict = {}
        if referent == "attachment" and _it_skill_md(ctx):
            extras["attachment"] = _it_skill_md(ctx)
        if named:
            extras["path"] = named
        targets = ("skills", "attachment") if referent == "attachment" else ("skills",)
        return _it_tools("skill_save", ctx, match.group(0), referent=referent,
                         confirm=("save_skill",), targets=targets, extras=extras)
    match = _IT_SKILL_USE.search(text)
    if match:
        name = match.group("name").strip()
        if name.lower() not in _IT_SKILL_USE_NOT and name.split()[0].lower() not in (
                "a", "an", "any", "this", "that", "your", "my"):
            return _it_tools("skill_use", ctx, match.group(0), targets=("skills",),
                             extras={"skill": name})
    match = _it_any(_IT_SKILL_LIST, text)
    if match:
        return _it_tools("skill_list", ctx, match.group(0), targets=("skills",))
    return None


# --------------------------------------------------------------------------- #
# Memory. "remember that X" in the prose lane got "I'll remember" and no write,
# a false claim; these go straight to the memory tools.
# --------------------------------------------------------------------------- #
_IT_MEM_WRITE = (
    _it_rx(r"^(?:please\s+)?(?:remember|don't\s+forget|dont\s+forget|do\s+not\s+forget|keep\s+in\s+"
           r"mind|make\s+a\s+note|take\s+(?:a\s+)?note|note\s+down|note)\s*(?:that\b|:)\s*\S"),
    _it_rx(r"^(?:please\s+)?(?:remember|don't\s+forget|dont\s+forget|do\s+not\s+forget)\s+(?:my|i|"
           r"i'm|i\s+am|we|we're|our|to\s+always|to\s+never|always|never)\b"),
)
_IT_MEM_WRITE_DEICTIC = _it_rx(
    r"^(?:please\s+)?(?:save|add|store|put|write)\s+(?:this|that|it)\s+(?:to|in|into)\s+"
    r"(?:my\s+|your\s+)?(?:notes|memory|memories)\b"
    r"|^(?:please\s+)?remember\s+(?:this|that)\s*[.!]?\s*$")
_IT_MEM_FORGET = (
    _it_rx(r"^(?:please\s+)?forget\s+(?:about\s+)?(?:my\b|what\s+i\s+(?:told|said)\b|the\s+note\b|"
           r"that\s+note\b|everything\s+(?:you\s+know\s+)?about\s+me\b|all\s+(?:my\s+)?"
           r"(?:notes|memories)\b|that\s+(?:i|i'm|i\s+am|my|we|we're|our)\b)"),
    _it_rx(r"^(?:please\s+)?(?:delete|remove|clear|erase)\s+(?:the\s+|my\s+|that\s+|all\s+(?:my\s+)?)?"
           r"(?:note|notes|memory|memories)(?:\s+(?:about|on|for|called|named)\b|\s*[.!]?\s*$)"),
    _it_rx(r"^(?:please\s+)?(?:delete|remove|clear|erase)\b.{0,40}?\bfrom\s+(?:your\s+|my\s+)?"
           r"(?:memory|notes)\b"),
)
_IT_MEM_READ = (
    _it_rx(r"\bwhat\s+do\s+you\s+(?:remember|know)\s+about\s+me\b"),
    _it_rx(r"\bwhat\s+(?:have\s+you|did\s+you)\s+(?:remembered|saved|noted|stored)\b"),
    _it_rx(r"\b(?:show|list|read|summari[sz]e|search|check|open)\s+(?:me\s+)?(?:all\s+)?(?:my|your)"
           r"\s+(?:saved\s+)?(?:notes|memories)\b"),
    _it_rx(r"\bwhat(?:'s|\s+is|\s+are)\s+in\s+my\s+notes\b"),
    _it_rx(r"\bwhat\s+(?:are|were)\s+my\s+(?:saved\s+)?(?:preferences|notes)\b"),
    _it_rx(r"\bdo\s+you\s+remember\s+(?:my|what\s+i|that\s+i|me)\b"),
    _it_rx(r"^(?:my\s+notes|my\s+memories)\s*\??$"),
)
# A second sentence after the note is usually the real request ("keep in mind
# that I'm on Windows. How do I install X?"), so a write needs a single one.
_IT_SECOND_SENTENCE = re.compile(r"[.!;]\s+\w")


def _it_memory(text: str, ctx: _ItCtx) -> Intent | None:
    match = _IT_MEM_WRITE_DEICTIC.search(text)
    if match:
        referent = _it_referent(ctx)
        if not referent:
            return _it_clarify("memory_write", ctx, match.group(0),
                               "What should I remember: paste the text or tell me the fact to "
                               "keep?", targets=("memory",))
        return _it_tools("memory_write", ctx, match.group(0), referent=referent,
                         targets=("memory",))
    match = _it_any(_IT_MEM_WRITE, text)
    if match and "?" not in text and not _IT_SECOND_SENTENCE.search(text.rstrip(" .!")):
        return _it_tools("memory_write", ctx, match.group(0), referent="inline",
                         targets=("memory",))
    match = _it_any(_IT_MEM_FORGET, text)
    if match:
        return _it_tools("memory_forget", ctx, match.group(0), targets=("memory",))
    match = _it_any(_IT_MEM_READ, text)
    if match:
        return _it_tools("memory_read", ctx, match.group(0), targets=("memory",))
    return None


# --------------------------------------------------------------------------- #
# Targets shared by the document, history, knowledge-base and codebase rules.
# --------------------------------------------------------------------------- #
_IT_KB_TERM = _it_rx(
    r"\b(?:the|my|our|your|this)\s+knowledge[- ]?bases?\b"
    r"|\b(?:the|my|our|your)\s+kb\b"
    r"|\b(?:search|query)\s+(?:the\s+)?kb\b"
    r"|\bmy\s+(?:indexed\s+)?(?:docs|documents)\b(?!\s+folder)"
    r"|\bindexed\s+(?:docs|documents|files|pages)\b"
    r"|\b(?:according\s+to|based\s+on)\s+(?:my|our)\s+(?:docs|documents)\b"
    # "my library" is also a code library or a building; it is the document
    # library only where it is asked what it says.
    r"|\bwhat\s+does\s+(?:my|our)\s+(?:document\s+)?library\s+say\b"
    r"|\b(?:according\s+to|search|check)\s+(?:my|our)\s+document\s+library\b")
_IT_PROJECT_STRONG = _it_rx(
    r"\b(?:the|this|my|our|your)\s+(?:code\s?base|repo(?:sitory)?|source\s+code|source\s+tree|"
    r"project\s+(?:folder|directory|dir|files))\b|\bcode\s?base\b")
_IT_PROJECT_WEAK = _it_rx(
    r"\b(?:the|this|my|our)\s+(?:project|app|application)(?![\w'-])(?!\s+(?:plan|proposal|idea|"
    r"ideas|brief|description|timeline|budget|manager|management|store|name|scope|charter|"
    r"status|update|report|deadline|meeting|team|goals?|requirements|summary))"
    r"|\b(?P<code>(?:the|my|our)\s+code)(?![\w'-])(?!\s+(?:below|above|here|snippet|block|sample|"
    r"example|you|i|we|that\s+you|which\s+you|from\s+(?:your|the)\s+(?:answer|reply|last)|"
    r"in\s+your\s+(?:answer|reply)|of\s+conduct))")
_IT_HISTORY = (
    _it_rx(r"\b(?:past|previous|earlier|old|older|other|prior|last)\s+(?:chats?|conversations?)\b"
           r"(?!\s+(?:i|we)\s+had\b|\s+with\s+(?!you\b))"),
    _it_rx(r"\bsearch\s+(?:through\s+)?(?:my\s+|our\s+|the\s+)?(?:chat\s+|conversation\s+)?history"
           r"\b(?!\s+of\b)"),
    # "chat history" as the place to look, not as the subject ("is my chat
    # history private", "my conversation history disappeared").
    _it_rx(r"\b(?:in|from|through|across)\s+(?:my\s+|our\s+|the\s+)?(?:chat|conversation)\s+"
           r"history\b"),
    _it_rx(r"\bfind\s+(?:the|a|my|that|our)\s+(?:chat|conversation|discussion|thread)\s+(?:about|"
           r"where|on|in\s+which|from)\b"),
    _it_rx(r"\blast\s+time\s+(?:we|i|you)\b"),
    _it_rx(r"\b(?:what|when)\s+did\s+(?:we|i)\s+(?:discuss|talk\s+about|ask\s+(?:you\s+)?(?:about)?|"
           r"say\s+about|decide)\b.{0,60}?\b(?:yesterday|last\s+(?:week|month|night|time)|earlier|"
           r"before|previously|the\s+other\s+day|ago)\b"),
    _it_rx(r"\b(?:we|i)\s+(?:discussed|talked\s+about|asked\s+you\s+about)\b.{0,60}?\b(?:yesterday|"
           r"last\s+(?:week|month|night|time)|the\s+other\s+day|ago|in\s+(?:a|another|an\s+earlier|"
           r"the\s+previous)\s+(?:chat|conversation))\b"),
    _it_rx(r"\byou\s+(?:told|showed|gave)\s+me\b.{0,40}?\b(?:yesterday|last\s+(?:week|month|night|"
           r"time)|the\s+other\s+day|in\s+(?:a|another|an\s+earlier|the\s+previous)\s+"
           r"(?:chat|conversation))\b"),
)
# Web as a document source: an explicit web word, a URL or a domain.
_IT_WEB_WORDS = _it_rx(r"\b(?:the\s+web|web\s+search|online|internet|google|websites?|"
                       r"search\s+engine|latest\s+news|news\s+about)\b")
_IT_NAMED_FILE = re.compile(
    r"(?<![\w./-])@?(?:[\w.-]+/)*(?P<file>[\w-]+\.(?:py|js|ts|tsx|jsx|md|txt|json|ya?ml|toml|csv|"
    r"pdf|docx|xlsx|pptx|html?|ipynb|rs|go|java|rb|sh|sql|log|ini|cfg|xml))\b", re.I)
_IT_OUTPUT_NAME = re.compile(r"(?:\bas|\bto|\binto|\bcalled|\bnamed|\bfile)\s+$", re.I)


# "the source" is a citation as often as it is code ("check the source for this
# claim"); next to words only code has, it is the project's source.
_IT_SOURCE = _it_rx(r"\b(?:the|this|my|our)\s+source\b(?!\s+(?:of|for\s+(?:this|that|the)\s+"
                    r"(?:claim|quote|statement|article|story|number|figure)))")
_IT_CODE_ONLY_WORDS = _it_rx(r"\b(?:code\s+smells?|smells?|anti-?patterns?|tech(?:nical)?\s+debt|"
                             r"code\s+quality|dead\s+code|unused\s+(?:imports|code|functions)|"
                             r"refactor\w*|bugs?|architecture|modules?|functions?|classes)\b")


def _it_has_project(text: str, ctx: _ItCtx, *, strong_only: bool = False) -> re.Match | None:
    match = _IT_PROJECT_STRONG.search(text)
    if match:
        return match
    if ctx.basename_rx is not None:
        match = ctx.basename_rx.search(text)
        if match:
            return match
    if strong_only:
        return None
    match = _IT_PROJECT_WEAK.search(text)
    if match:
        return match
    match = _IT_SOURCE.search(text)
    return match if match and _IT_CODE_ONLY_WORDS.search(text) else None


def _it_named_files(text: str, ctx: _ItCtx) -> list[str]:
    """Files named as material to read, not the output name and not an attachment."""
    scan = _URL_IN_TEXT.sub(" ", text)
    attached = {n.lower() for n in ctx.attachment_names}
    out = []
    for match in _IT_NAMED_FILE.finditer(scan):
        if _IT_OUTPUT_NAME.search(scan[:match.start()]):
            continue
        name = match.group("file")
        if name.lower() not in attached:
            out.append(name)
    return out


# --------------------------------------------------------------------------- #
# document_create: a file to produce. document_request() keeps today's line
# between producing and reading ("summarise this pdf" reads one), and the
# other targets in the same message become research sources, so a PDF "from
# the knowledge base" can search it (turn 2 of the review could not).
# --------------------------------------------------------------------------- #
_IT_DOC_FORMATS = (".pdf", ".docx", ".xlsx", ".pptx")
_IT_PROBLEM = _it_rx(r"\b(?:broken|not\s+working|doesn't\s+work|does\s+not\s+work|fails?|failing|"
                     r"corrupt\w*|malformed|garbled|(?:won't|can't|cannot|doesn't|does\s+not)\s+open|"
                     r"(?:is|was)\s+not\s+(?:a\s+)?(?:valid|well-formed|readable|proper)|"
                     r"crash\w*|bug|error|exception|issue\s+with)\b")
_IT_DOC_VERB = _it_rx(
    r"\b(?:creat\w*|generat\w*|mak\w*|writ\w*|produc\w*|export\w*|sav\w*|build\w*|draft\w*|"
    r"prepar\w*|render\w*|compil\w*|craft\w*|compos\w*|put\s+together|turn\s+(?:it|this|that|"
    r"them)\s+into|convert\w*|give\s+me|send\s+me|i\s+(?:need|want)|output)\b")
_IT_DOC_DEICTIC = _it_rx(r"\b(?:this|that|it|the\s+(?:above|answer|last\s+answer|summary))\b")
_IT_HISTORY_SOURCE = _it_rx(r"\bour\s+(?:chats|conversations)\b|\bwhat\s+we\s+(?:discussed|talked\s+"
                            r"about)\b")


# Sources whose material is not in the knowledge base: injecting KB passages
# into a document built from them is noise (P2: auto-RAG is off for codebase,
# history and attachment turns).
_IT_NOT_KB_SOURCES = frozenset({"codebase", "history", "files"})


def _it_document_retrieval(sources: list[str], referent: str) -> str:
    """A document's retrieval follows its sources, as the matching intent's would."""
    if "knowledge_base" in sources:
        return "kb"
    if "web" in sources:
        return "auto"
    if sources and set(sources) <= _IT_NOT_KB_SOURCES:
        return "off"
    if not sources and referent == "attachment":
        return "off"
    return "auto"


def _it_document(text: str, ctx: _ItCtx, *, fmt: str = "") -> Intent | None:
    if not fmt:
        fmt = document_request(text) or ""
        if fmt not in _IT_DOC_FORMATS and not _DOC_READ.search(text):
            # document_request lets a spelled-out filename win, so "create a pdf
            # report from notes.md" says .md; the format asked for is the PDF.
            fmt = next((suffix for suffix, pattern in _DOC_OTHER
                        if suffix in _IT_DOC_FORMATS and pattern.search(text)), fmt)
        if fmt not in _IT_DOC_FORMATS:
            return None
        if is_code_request(text) or _IT_PROBLEM.search(text) or _IT_HOWTO.search(text):
            # "create a pdf parser in python" asks for code, "export to pdf is
            # broken" reports a bug and "how do I save a pdf in word" asks for
            # instructions; all three stay on today's pipeline.
            return None
        target = any(pattern.search(text) for _, pattern in _DOC_TARGET)
        if not (target or _IT_DOC_VERB.search(text)):
            return None  # "where is the pdf file" names a format but asks for nothing
    sources: list[str] = []
    if _IT_KB_TERM.search(text):
        sources.append("knowledge_base")
    if _it_has_project(text, ctx, strong_only=True):
        sources.append("codebase")
    if _it_any(_IT_HISTORY, text) or _IT_HISTORY_SOURCE.search(text):
        sources.append("history")
    explicit_web = bool(_URL_IN_TEXT.search(text) or _BARE_DOMAIN.search(text)
                        or _IT_WEB_WORDS.search(text))
    # A bare "research" or "look up" meant the web before this module existed
    # (document_needs_research), and still does when nothing else is named; with
    # the knowledge base named, "research my knowledge base" is not a web search.
    if explicit_web or (not sources and _LOOKUP_ASK.search(text)):
        sources.append("web")
    if _it_named_files(text, ctx):
        sources.append("files")
    referent = ("attachment" if ctx.has_attachment
                else "previous_answer" if ctx.has_history and _IT_DOC_DEICTIC.search(text) else "")
    targets = tuple(dict.fromkeys(["document", *sources]
                                  + (["attachment"] if ctx.has_attachment else [])))
    return Intent(name="document_create", lane="document",
                  retrieval=_it_document_retrieval(sources, referent),
                  research_sources=tuple(sources), targets=targets, referent=referent,
                  source=ctx.source, matched=text[:80], extras={"format": fmt})


# --------------------------------------------------------------------------- #
# History and the knowledge base.
# --------------------------------------------------------------------------- #
def _it_history(text: str, ctx: _ItCtx) -> Intent | None:
    match = _it_any(_IT_HISTORY, text)
    if not match or _IT_ASK_HOW.search(text):
        return None  # "how do I export my chat history" asks for instructions
    return _it_tools("history", ctx, match.group(0), targets=("history",))


_IT_KB_WRITE = (
    _it_rx(r"^(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?(?:index|re-?index|ingest|add|import|"
           r"upload|put|save|load)\b.{0,80}?\b(?:to|into|in)\s+(?:the\s+|my\s+|our\s+)?"
           r"(?:kb|knowledge[- ]?base)\b"),
    _it_rx(r"^(?:please\s+)?(?:re-?index|index|ingest)\s+(?:the\s+|my\s+|this\s+|that\s+|these\s+|"
           r"our\s+|all\s+(?:the\s+)?)?(?:[\w./~-]+\s+){0,2}(?:folder|directory|dir|files?|docs|"
           r"documents|pages?|site|website|url|links?|pdfs?|notes|repo(?:sitory)?)\b"),
    _it_rx(r"^(?:please\s+)?(?:re-?index|index|ingest)\s+(?:https?://\S+|[~./][\w./~-]+|"
           r"[\w-]+/[\w./-]*)"),
)
_IT_KB_WRITE_DEICTIC = _it_rx(r"^(?:please\s+)?(?:add|save|put|index|upload)\s+(?:this|that|it|these|"
                              r"those)\s+(?:to|into|in)\b")
_IT_KB_OVERVIEW = _it_rx(
    r"\b(?:what(?:'s|\s+is|\s+are)\s+(?:in|inside|stored\s+in)|overview|list(?:\s+(?:all|the|my))?|"
    r"inventory|how\s+many|which\s+(?:docs|documents|files|topics)|(?:useful|key|main|interesting)"
    r"\s+facts|main\s+topics|topics|table\s+of\s+contents|contents|stats|statistics|size\s+of)\b")
_IT_KB_SUMMARY = _it_rx(r"\bsummar\w+")
# The knowledge base as the subject of a question about the feature itself,
# not as the place to look ("what does the knowledge base say" still looks).
_IT_KB_ABOUT = _it_rx(r"\b(?:how\s+(?:does|do|is|are|can|should)|what\s+(?:is|are)|why\s+(?:does|is))"
                      r"\s+(?:the|a|my|your)\s+(?:knowledge[- ]?base|kb)\b(?!\s+say)")
# Someone else's knowledge base product.
_IT_KB_ELSEWHERE = _it_rx(r"\b(?:notion|confluence|obsidian|zendesk|sharepoint|gitbook|freshdesk|"
                          r"helpjuice|guru|wiki(?:pedia)?)\b")
_IT_KB_TOPIC = _it_rx(r"\b(?:about|regarding|concerning|on\s+how|says?\s+about|for)\b")
# One document in the knowledge base is a search, not an overview.
_IT_KB_ONE_DOC = _it_rx(r"\b(?:knowledge[- ]?base|kb)\s+(?:article|document|doc|page|entry|note|"
                        r"section|post|file)\b")
# The knowledge base as the subject of a remark: "the knowledge base is slow",
# "clearing the knowledge base did not empty it". The user is reporting
# something, not asking to search it.
_IT_KB_REMARK = _it_rx(
    r"\b(?:the|my|our)\s+(?:knowledge[- ]?base|kb)\s+(?:is|was|seems|looks|keeps|did|didn't|"
    r"doesn't|isn't|won't|wasn't|has\s+been|still|never|always|got|became)\b"
    r"|\b(?:clearing|indexing|re-?indexing|importing|emptying|resetting)\s+(?:the|my|our)\s+"
    r"(?:knowledge[- ]?base|kb)\b")


def _it_kb(text: str, ctx: _ItCtx) -> Intent | None:
    if _IT_KB_ELSEWHERE.search(text):
        return None
    match = _it_any(_IT_KB_WRITE, text)
    if match:
        url = _URL_IN_TEXT.search(text)
        if url:
            referent = "inline"
        elif _IT_KB_WRITE_DEICTIC.search(text):
            referent = "attachment" if ctx.has_attachment else ""
            if not referent:
                # The previous answer cannot be indexed: index_docs takes a path
                # and index_url a URL. Ask for one of those.
                return _it_clarify("kb_write", ctx, match.group(0),
                                   "What should I add to the knowledge base: attach the file, give "
                                   "me a folder path, or paste a URL?",
                                   targets=("knowledge_base",))
        else:
            referent = "named_path"
        return _it_tools("kb_write", ctx, match.group(0), referent=referent,
                         targets=("knowledge_base",))
    match = _IT_KB_TERM.search(text)
    if not match:
        return None
    if _IT_KB_ABOUT.search(text) or _IT_ASK_HOW.search(text) or _IT_KB_REMARK.search(text):
        return None  # "how does the knowledge base work" asks about the feature
    overview = _IT_KB_OVERVIEW.search(text) or _IT_KB_SUMMARY.search(text)
    if overview and not (_IT_KB_TOPIC.search(text) or _IT_KB_ONE_DOC.search(text)):
        return _it_tools("kb_overview", ctx, match.group(0), retrieval="kb",
                         targets=("knowledge_base",))
    return _it_tools("kb_query", ctx, match.group(0), retrieval="kb",
                     targets=("knowledge_base",))


# --------------------------------------------------------------------------- #
# The codebase. "Codebase" is a Settings term (Admin > Codebase, PROJECT_DIR),
# and the project folder's own name means the same thing. Retrieval is off for
# both intents: the knowledge base is a different corpus, and turn 3 of the
# review was answered from an unrelated imported repo it injected.
# --------------------------------------------------------------------------- #
_IT_IDENT = (r"(?:[A-Za-z_][A-Za-z0-9]*_[A-Za-z0-9_]+|[a-z]+[A-Z][A-Za-z0-9]*|"
             r"[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*|[A-Za-z_]\w*\(\)|"
             r"[A-Za-z_]\w*\.[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)")
# Case-sensitive on purpose: under re.I the camelCase branches match any word.
_IT_IDENT_RX = re.compile(rf"(?<![\w/@:.-])(?!(?:[eE]\.g|[iI]\.e|etc)\b){_IT_IDENT}(?![\w-])")
_IT_LOOKUP = (
    _it_rx(r"\bwhere\s+(?:is|are|do\s+we|does\s+\w+|did\s+we|was|were)\b.{0,40}?\b(?:defined|"
           r"declared|implemented|used|called|invoked|imported|referenced|instantiated|raised|"
           r"registered|handled|configured|set)\b"),
    _it_rx(r"\b(?:find|show|list)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?(?:usages?|uses|references|"
           r"callers|call\s+sites|definitions?|occurrences)\s+(?:of|to|for)\b"),
    _it_rx(r"\bwho\s+calls\b"),
)
_IT_LOOKUP_DIRECT = (
    _it_rx(r"\bwhere\s+in\s+(?:the|this|my|our)\s+(?:code\s?base|repo(?:sitory)?|project|code)\b"),
    _it_rx(r"^(?:please\s+)?grep\s+(?:for\s+)?(?!(?:is|was|does|command|stands|means)\b)\S+"
           r"(?!.*\?)"),
)
_IT_REVIEW = _it_rx(
    r"\b(?:review|audit|check|inspect|analy[sz]e|analysis|summari[sz]e|summary|overview|"
    r"architecture|structure|organi[sz]ation|organi[sz]ed|smells?|mistakes?|issues|problems|bugs?|"
    r"flaws|weaknesses|anti-?patterns?|tech(?:nical)?\s+debt|code\s+quality|improvements|explore|"
    r"walk\s+(?:me\s+)?through|explain|describe|understand|look\s+(?:at|through|over)|"
    r"go\s+(?:through|over)|assess|evaluate|critique|feedback\s+on)\b"
    r"|\bdesign\s+(?:mistakes|issues|flaws|problems|review|choices|decisions|patterns|smells|"
    r"quality)\b|\b(?:the|its|overall)\s+design\b"
    r"|\bwhat\s+does\s+(?:the|this|my|our)\s+(?:project|app|application|code\s?base|"
    r"repo(?:sitory)?|code)\s+do\b")
# A request to change code: the review subset has no edit tools, so these stay
# on today's pipeline. The verb needs an object, so "suggest what to refactor"
# is still a review.
_IT_MODIFY = _it_rx(
    r"\b(?:fix|edit|change|modify|refactor|rename|add|implement|write|update|patch|delete|remove|"
    r"create|build|rewrite|migrate|upgrade|install|run|deploy|commit|push|format|lint|convert|"
    r"port|translate)\s+(?:it|them|this|that|these|those|the|a|an|my|our|all|any|some|every|each|"
    r"new|[\w-]+\.\w+)\b")
_IT_HOSTED = _it_rx(r"\b(?:github|gitlab|bitbucket|pull\s+requests?|merge\s+requests?|prs?|"
                    r"app\s+store|play\s+store)\b")


def _it_codebase(text: str, ctx: _ItCtx) -> Intent | None:
    if ctx.fenced or _IT_HOSTED.search(text):
        return None
    project = _it_has_project(text, ctx)
    direct = _it_any(_IT_LOOKUP_DIRECT, text)
    if direct:
        return _it_tools("codebase_lookup", ctx, direct.group(0), targets=("codebase",))
    lookup = _it_any(_IT_LOOKUP, text)
    if lookup and (project or _IT_IDENT_RX.search(text)):
        return _it_tools("codebase_lookup", ctx, lookup.group(0), targets=("codebase",))
    if not project:
        return None
    strong = _it_has_project(text, ctx, strong_only=True)
    if not strong:
        # "the code" or "my code" with an attachment or an earlier answer most
        # likely means that code, not the project folder.
        code_only = bool(project.re is _IT_PROJECT_WEAK and project.group("code"))
        if ctx.has_attachment or (code_only and ctx.has_history):
            return None
    review = _IT_REVIEW.search(text)
    if not review or _IT_MODIFY.search(text):
        return None
    return _it_tools("codebase_review", ctx, project.group(0), targets=("codebase",))


# --------------------------------------------------------------------------- #
# codebase_write: documentation written from the project's own files ("check
# the codebase and write a readme.md"). The message that taught this rule
# matched nothing (the review rule rightly stays silent on "write a"), the
# router picked list_files, and the full catalogue ran on a 3B model: it read
# the first 1,500 characters of a 100 KB README.md, wrote 841 characters back
# over the whole file, fetched a GitHub URL it invented and never said it had
# replaced anything. Here the turn gets the file tools and nothing on the web,
# both write tools need the user's OK, and the guidance below tells the model
# to read before it writes and to put a new version beside an existing file
# unless the user asked to replace it.
#
# The verb must govern the target ("write a readme", "add a section to the
# README"), so a code request that only mentions documentation ("write a python
# script that generates the docs") stays on today's pipeline, as do PDF and
# Word requests, which document_create has already taken by the time this runs.
# --------------------------------------------------------------------------- #
_IT_CW_VERB = (r"(?:write|re-?write|create|generate|draft|update|improve|add|make|produce|"
               r"prepare|document|refresh|expand|extend|fill\s+in|flesh\s+out|overwrite|"
               r"replace)")
# Words that end the verb's reach: a clause boundary, or a thing to write that
# is not documentation (code, or a reply about the docs rather than the docs).
_IT_CW_GAP = (r"(?!(?:that|which|who|if|when|because|so|but|then|and|or|script|scripts|"
              r"function|functions|program|programs|class|classes|method|methods|tests?|"
              r"parser|generator|bot|cli|summary|summaries|review|critique|feedback|"
              r"explanation|analysis|opinion|thoughts|questions?|answer|tl;?dr|report|"
              r"translation|comparison|email|letter|tweet|post)\b)[\w'./-]+")
_IT_CW_TARGET = (
    r"(?P<readme>read[ _-]?me(?:\.(?:md|markdown|rst|txt))?)"
    r"|(?P<changelog>change[ _-]?log(?:\.md)?)"
    r"|(?P<contributing>contributing(?:\.md)?)"
    r"|(?P<arch>(?:architecture|design)\s+(?:docs?|documents?|documentation))"
    r"|(?P<file>[\w./-]*[\w-]\.(?:md|markdown|rst|txt|adoc))"
    r"|(?P<docs>(?<!google\s)docs?|documentation|guide|(?:markdown|md)\s+(?:file|doc|document))")
_IT_CW_ASK = _it_rx(
    rf"\b{_IT_CW_VERB}\s+(?:{_IT_CW_GAP}\s+){{0,5}}?(?:{_IT_CW_TARGET})(?![\w-])")
# The verb under a negation: "review the codebase but don't update the readme".
_IT_CW_NOT = _it_rx(r"\b(?:don't|do\s+not|never|without|no\s+need\s+to|not)\s+(?:\w+\s+)?$")
# A request for code that writes documentation is a code request.
_IT_CW_CODE = _it_rx(r"\b(?:write|create|generate|build|make|implement|code)\s+(?:me\s+)?"
                     r"(?:a|an|the|some)?\s*(?:[\w+#.-]+\s+){0,2}(?:script|function|program|"
                     r"class|cli|parser|generator|bot|tool|hook|workflow|action|plugin)\b")
# Words that ask for the existing file to be replaced, not kept beside the new
# one. "replace" counts only with the file as its object: "replace the install
# section of the readme" is an edit, and the file stays.
_IT_CW_OVERWRITE = _it_rx(
    r"\b(?:replacing|overwriting|over-writing|from\s+scratch|start\s+over)\b"
    r"|\b(?:throw\s+away|discard)\s+(?:the\s+)?(?:old|existing|current)\b"
    r"|\b(?:replace|overwrite)\s+(?:it|the\s+(?:old|existing|current)\s+(?:one|file|version))\b"
    r"|\brewrite\s+(?:\w+\s+){0,3}?(?:completely|entirely|fully)\b")
_IT_CW_REPLACE_FILE = _it_rx(
    r"\b(?:replace|overwrite|over-write)\s+(?:(?:the|my|our|this|its|existing|current|old|"
    rf"whole|entire)\s+)*(?:{_IT_CW_TARGET})(?![\w-])")
# Conventional names, so "a readme" and "readme.md" both mean the README.md a
# repository already has (on a case-sensitive disk "readme.md" is a new file).
_IT_CW_CANONICAL = {"readme": "README", "changelog": "CHANGELOG",
                    "contributing": "CONTRIBUTING"}


def _it_cw_target(match: re.Match) -> tuple[str, str]:
    """(file name, kind) the request is about; the name is "" for docs in general."""
    groups = match.groupdict()
    for kind in ("readme", "changelog", "contributing"):
        raw = groups.get(kind)
        if raw:
            suffix = raw[raw.rfind("."):].lower() if "." in raw else ".md"
            return f"{_IT_CW_CANONICAL[kind]}{suffix}", kind
    if groups.get("arch"):
        stem = "ARCHITECTURE" if groups["arch"].lower().startswith("arch") else "DESIGN"
        return f"{stem}.md", "architecture"
    if groups.get("file"):
        name = groups["file"].lstrip("./")
        stem = name.rsplit("/", 1)[-1].rsplit(".", 1)[0].replace("-", "").replace("_", "").lower()
        if "/" not in name and stem in _IT_CW_CANONICAL:
            return f"{_IT_CW_CANONICAL[stem]}{name[name.rfind('.'):].lower()}", stem
        return name, "file"
    return "", "docs"


def codebase_write_guidance(target: str = "", *, overwrite: bool = False,
                            outputs_dir: str = "outputs") -> str:
    """What the model is told on a codebase_write turn (Agent injects it).

    Short numbered steps, because the model it is written for is a 3B one that
    followed none of this unprompted. `outputs_dir` is the configured
    OUTPUTS_DIR; with none ("" when it is the project root) a new version goes
    next to the original under a .new name instead.
    """
    name = str(target or "").strip()
    outputs = str(outputs_dir or "").strip().strip("/")
    if name:
        if outputs:
            beside = f"{outputs}/{name}"
        else:
            stem, dot, ext = name.rpartition(".")
            beside = f"{stem}.new.{ext}" if dot and stem else f"{name}.new"
        look = f"read_file {name} if it exists"
        if overwrite:
            keep = (f"3. The user asked to replace {name}: write_file it with the complete new "
                    "text, never a shortened copy of a file you saw only part of.")
        else:
            keep = (f"3. If {name} already exists, do not replace it: the user did not ask to "
                    "overwrite it. Make targeted edit_file changes to parts you have read, or "
                    f"write the complete new version to {beside} and say where it is. Never "
                    "write_file over a file you saw only part of.")
    else:
        # No file named ("write docs for the repo"): any existing file is one
        # the user did not ask to have replaced.
        beside = f"{outputs}/<file name>" if outputs else "a new file with .new in its name"
        look = "read README.md and docs/ if they exist"
        if overwrite:
            keep = ("3. The user asked to replace the existing documentation: write_file each "
                    "file with its complete new text, never a shortened copy of a file you "
                    "saw only part of.")
        else:
            keep = ("3. Do not replace an existing file: the user did not ask to overwrite "
                    f"one. Put new text in {beside}, or make targeted edit_file changes to "
                    "parts you have read. Never write_file over a file you saw only part of.")
    return "\n".join((
        f"The user wants {name or 'documentation'} written from this project's own files. "
        "Use the file tools only: nothing on the web describes this project, so do not "
        "search or fetch it.",
        f"1. Look first: list_files, then {look}, and the files that say what the "
        "project is (entry points, configuration, the main package). file_info gives a "
        "file's size; read_file shows only the start of a long file.",
        "2. Write only what the files you read show. Do not invent features, commands, "
        "options, badges or URLs; leave a section out rather than guess.",
        keep,
        "4. Each write_file and edit_file may be put to the user for approval first; if one "
        "is declined, do not send it again unchanged.",
        "5. In the final answer, name the file you wrote or changed (or say you changed "
        "nothing) and list the files you read.",
    ))


# "explain" opens a question only when a question word follows it: "explain the
# codebase and write a README" is an instruction with two steps.
_IT_CW_EXPLAIN_HOW = _it_rx(r"^(?:please\s+)?explain\s+(?:to\s+me\s+)?(?:how|why|what|when|"
                            r"whether|which)\b")


def _it_cw_question(text: str) -> bool:
    howto = _IT_HOWTO.search(text)
    if howto is None:
        return False
    if howto.group(0).lower().endswith("explain"):
        return bool(_IT_CW_EXPLAIN_HOW.search(text))
    return True


def _it_codebase_write(text: str, ctx: _ItCtx, *, project_named: bool = False) -> Intent | None:
    if ctx.fenced or _IT_HOSTED.search(text) or _it_cw_question(text):
        return None  # pasted code, a GitHub page, or "how do I write a good README"
    ask = None
    for match in _IT_CW_ASK.finditer(text):
        if not _IT_CW_NOT.search(text[:match.start()]):
            ask = match
            break
    if ask is None or _IT_CW_CODE.search(text):
        return None
    project = None if project_named else _it_has_project(text, ctx)
    if not project_named:
        if not project:
            return None  # "write a readme" alone: whose? The pipeline asks or guesses.
        if not _it_has_project(text, ctx, strong_only=True):
            # The same yield as the review rule: "my code" next to an attachment
            # or an earlier answer is that code, not the project folder.
            code_only = bool(project.re is _IT_PROJECT_WEAK and project.group("code"))
            if ctx.has_attachment or (code_only and ctx.has_history):
                return None
    target, kind = _it_cw_target(ask)
    overwrite = bool(_IT_CW_OVERWRITE.search(text) or _IT_CW_REPLACE_FILE.search(text))
    return _it_tools("codebase_write", ctx, ask.group(0), confirm=("write_file", "edit_file"),
                     targets=("codebase",),
                     extras={"target": target, "target_kind": kind, "overwrite": overwrite,
                             "guidance": codebase_write_guidance(target, overwrite=overwrite)})


# --------------------------------------------------------------------------- #
# Prompt library phrasing ("use my code-review prompt", "save this to the prompt
# library") and bulk skill imports: features with no tool yet. Left to the
# pipeline rather than misread as a codebase, skill or knowledge-base request.
# --------------------------------------------------------------------------- #
_IT_PROMPT_LIBRARY = _it_rx(
    r"\b(?:use|run|apply|load|open)\s+(?:my|the|our)\s+(?:saved\s+)?[\w-]+(?:\s+[\w-]+)?\s+prompt\b"
    r"|\bprompts?\s+library\b|\bsave\s+(?:this|that|it)\s+as\s+a\s+prompt\b")


def _it_rules(text: str, ctx: _ItCtx) -> Intent | None:
    """The lexicon proper, in evaluation order; the first rule that fires wins."""
    if _IT_PROMPT_LIBRARY.search(text) or _IT_SKILL_IMPORT.search(text):
        return None
    for rule in _IT_RULES:
        found = rule(text, ctx)
        if found is not None:
            return found
    return None


_IT_RULES = (
    _it_conv_action, _it_admin_action, _it_self_describe, _it_schedule,
    _it_disabled_capability, _it_skill, _it_memory, _it_document, _it_history, _it_kb,
    _it_codebase_write, _it_codebase,
)


# --------------------------------------------------------------------------- #
# Slash commands: the explicit form of the same intents. routing_override owns
# /search, /web, /nosearch, /answer, /local, /agent, /tools and /act, so those
# return None here; /kb and /docs belong to both (routing_override strips the
# command, classify says what it asks for).
# --------------------------------------------------------------------------- #
_IT_SLASH = re.compile(r"^\s*/(kb|docs|code|skills?|history|mem|memory|pdf)(?=\s|$)\s*", re.I)
_IT_SLASH_FAMILY = {"kb": "kb", "docs": "kb", "code": "code", "skill": "skills",
                    "skills": "skills", "history": "history", "mem": "memory",
                    "memory": "memory", "pdf": "pdf"}


def _it_slash(family: str, rest: str, ctx: _ItCtx) -> Intent:
    text = _it_clean(rest)
    matched = f"/{family}"
    found: Intent | None = None
    if family == "kb":
        found = _it_kb(text, ctx) if text else None
        if found is None or found.name not in ("kb_write", "kb_overview", "kb_query"):
            name = "kb_query" if text else "kb_overview"
            found = _it_tools(name, ctx, matched, retrieval="kb", targets=("knowledge_base",))
    elif family == "code":
        # Under /code the project is already named, so "/code write a readme"
        # is documentation for it rather than a read-only review.
        written = _it_codebase_write(text, ctx, project_named=True) if text else None
        found = written or (_it_codebase(text, ctx) if text else None)
        lookup = _it_any(_IT_LOOKUP, text) or _it_any(_IT_LOOKUP_DIRECT, text) if text else None
        if written is None and (found is None or found.name != "codebase_lookup") and lookup:
            # Under /code the project is already named, so "where is classify
            # used" is a lookup even when "classify" is a plain word.
            found = _it_tools("codebase_lookup", ctx, matched, targets=("codebase",))
        if written is None and (found is None or found.name != "codebase_lookup"):
            found = _it_tools("codebase_review", ctx, matched, targets=("codebase",))
    elif family == "skills":
        found = _it_skill(text, ctx) if text else None
        if found is None:
            found = _it_tools("skill_list", ctx, matched, targets=("skills",))
    elif family == "history":
        found = _it_tools("history", ctx, matched, targets=("history",))
    elif family == "memory":
        found = _it_memory(text, ctx) if text else None
        if found is None:
            found = _it_tools("memory_read", ctx, matched, targets=("memory",))
    else:  # pdf
        if text:
            found = _it_document(text, ctx, fmt=".pdf")
        elif ctx.has_attachment or ctx.has_history:
            found = Intent(name="document_create", lane="document", retrieval="auto",
                           targets=("document",), source=ctx.source, matched=matched,
                           referent="attachment" if ctx.has_attachment else "previous_answer",
                           extras={"format": ".pdf"})
        else:
            found = _it_clarify("document_create", ctx, matched,
                                "What should the PDF contain?", targets=("document",))
    extras = dict(found.extras)
    extras["cleaned_message"] = rest.strip()
    return replace(found, extras=extras, source="slash")


def classify(message: str, *, has_attachment: bool = False,
             attachment_names: tuple[str, ...] = (), has_history: bool = False,
             project_basename: str = "", registered_tools: frozenset[str] | set[str] = frozenset(),
             enabled: bool = True) -> Intent | None:
    """Map a message to an Intent, or None to leave the turn to the existing pipeline.

    Deterministic and cheap. `registered_tools` lets the rules answer honestly
    when a capability is switched off (disabled_capability) and keeps a tools
    lane from being chosen when none of its tools exist. An empty set means
    the caller did not say, so every tool is assumed available and the
    switched-off replies stay silent.
    """
    if not enabled:
        return None
    text = (message or "").strip()
    if not text:
        return None
    override = routing_override(text)
    if override and override[0] != "kb":
        return None  # /search, /answer, /agent ...: routing_override decides
    ctx = _ItCtx(text, has_attachment=has_attachment,
                 attachment_names=tuple(attachment_names or ()), has_history=has_history,
                 project_basename=project_basename,
                 registered=frozenset(registered_tools or ()))
    slash = _IT_SLASH.match(text)
    if slash:
        ctx.source = "slash"
        return _it_slash(_IT_SLASH_FAMILY[slash.group(1).lower()], text[slash.end():], ctx)
    for instruction, body in _it_heads(text):
        ctx.body = body
        found = _it_rules(instruction, ctx)
        if found is not None:
            return found
    return None


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'INTENT_LANES',
    'INTENT_RETRIEVAL',
    'INTENT_NAMES',
    'INTENT_TOOL_SUBSETS',
    'DOCUMENT_RESEARCH_TOOLS',
    'Intent',
    'classify',
    'codebase_write_guidance',
]
