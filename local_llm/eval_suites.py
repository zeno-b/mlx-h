"""Evaluation suite data (item 19): cases as Python literals, so they bundle.

evals.py runs these. Every model-facing case carries a `script`: the canned
model outputs the deterministic "scripted" mode replays in order (a string, or
{"text", "finish"} where finish "length" simulates a reply cut off at its
token budget). Scripted mode proves the harness end to end and runs in CI; the
"live" mode sends the same prompts to the real local model and ignores the
scripts.

Suites
------
tool_calling  prompt -> the tool the agent should pick, predicates over its
              arguments, canned results for the tools it may call (side
              effecting tools are always stubbed in the sandbox), and facts
              the final answer must contain. expect_tool None means "answer
              without a tool" (and, for the bait case, without inventing one).
rag           fixture documents indexed into a throwaway per-user knowledge
              base; each question names the fact(s) the answer must contain
              and the source path it must cite.
multi_turn    scripted conversations (create an artifact, follow-up edits,
              continue after truncation, language drift); every turn lists
              the checks its answer must pass.
routing       messages -> the intent, lane, tool set and retrieval rule they
              must get (upgrade/ROUTING_AND_TOOLING.md P1 and P9): the three
              reviewed failures plus phrasings for every intent and ordinary
              questions the lexicon must leave alone. Scripted mode calls
              intents.classify() directly (no model) and runs the three
              reviewed messages through the Agent as well; live mode runs
              every case through the Agent against the configured model.
parsers       toolparse fixtures across every model family format, the 4-bit
              JSON repairs and prose that must stay prose (no model needed).
api           /v1 compat request/response shape checks; the cases live in
              evals.py because each one is a request sequence, not data.

Argument predicates (tool_calling "args"): {"contains": s} (case-insensitive
substring), {"contains_all": [..]}, {"regex": r} (case-insensitive search),
{"equals": v}, {"one_of": [..]}, {"present": True}, {"number": True}.
"""

from __future__ import annotations


# Files the sandbox workspace starts with (paths relative to the project).
EVAL_WORKSPACE_FILES: dict = {
    "config/settings.yaml": (
        "service:\n  name: invoicer\n  host: 0.0.0.0\n  port: 9137\n"
        "logging:\n  level: info\n"),
    "src/billing.py": (
        "TAX_RATE = 0.20\n\n\n"
        "def parse_invoice(text):\n"
        "    \"\"\"Split an invoice line into (description, amount).\"\"\"\n"
        "    description, _, amount = text.rpartition(\" \")\n"
        "    return description.strip(), float(amount)\n\n\n"
        "def total_with_tax(amount):\n"
        "    return round(amount * (1 + TAX_RATE), 2)\n"),
    "README.md": "# Invoicer\n\nA tiny invoicing service the evaluation suite works on.\n",
}


EVAL_TOOL_CASES: list = [
    {"id": "weather_city",
     "prompt": "What's the weather like in Ghent right now?",
     "expect_tool": "weather", "args": {"location": {"contains": "ghent"}},
     "canned": {"weather": "Ghent, Belgium: 14 C, light rain, wind 20 km/h from the south-west."},
     "answer_contains": ["14"],
     "script": ['{"tool": "weather", "args": {"location": "Ghent"}}',
                "It is 14 C with light rain in Ghent right now."]},
    {"id": "calculator_expression",
     "prompt": "Use the calculator to work out (17 * 23) + sqrt(144).",
     "expect_tool": "calculator", "args": {"expression": {"contains_all": ["17", "23", "144"]}},
     "answer_contains": ["403"],
     "script": ['{"tool": "calculator", "args": {"expression": "(17 * 23) + sqrt(144)"}}',
                "(17 * 23) + sqrt(144) = 403."]},
    {"id": "read_project_file",
     "prompt": "Open config/settings.yaml in the project and tell me which port the service listens on.",
     "expect_tool": "read_file", "args": {"path": {"contains": "settings.yaml"}},
     "answer_contains": ["9137"],
     "script": ['{"tool": "read_file", "args": {"path": "config/settings.yaml"}}',
                "The service listens on port 9137."]},
    {"id": "search_project_symbol",
     "prompt": "Find where the function parse_invoice is defined in the project.",
     "expect_tool": "search_files", "args": {"pattern": {"contains": "parse_invoice"}},
     "answer_contains": ["billing.py"],
     "script": ['{"tool": "search_files", "args": {"pattern": "def parse_invoice"}}',
                "parse_invoice is defined in src/billing.py."]},
    {"id": "web_search_current",
     "prompt": "Search the web for the latest stable release of the Rust compiler.",
     "expect_tool": "web_search", "args": {"query": {"contains": "rust"}},
     "canned": {"web_search": ("1. Announcing Rust 1.91.0 - The Rust Programming Language Blog\n"
                               "   The Rust team is happy to announce a new version of Rust, "
                               "1.91.0, the latest stable release.")},
     "answer_contains": ["1.91"],
     "script": ['{"tool": "web_search", "args": {"query": "latest stable Rust compiler release"}}',
                "According to the search results the latest stable Rust release is 1.91.0."]},
    {"id": "fetch_named_url",
     "prompt": "Read https://example.org/changelog and tell me what the newest entry added.",
     "expect_tool": "fetch_url", "args": {"url": {"contains": "example.org/changelog"}},
     "canned": {"fetch_url": "Changelog\n\n## 2.4.0\n- Added CSV export.\n\n## 2.3.1\n- Fixed a crash on empty invoices."},
     "answer_contains": ["csv"],
     "script": ['{"tool": "fetch_url", "args": {"url": "https://example.org/changelog"}}',
                "The newest entry, 2.4.0, added CSV export."]},
    {"id": "write_new_file",
     "prompt": "Save a file named notes/todo.txt with two lines: buy milk, call Anna.",
     "expect_tool": "write_file",
     "args": {"path": {"contains": "todo.txt"}, "content": {"contains_all": ["milk", "anna"]}},
     "script": ['{"tool": "write_file", "args": {"path": "notes/todo.txt", '
                '"content": "buy milk\\ncall Anna\\n"}}',
                "Saved notes/todo.txt with the two items."]},
    {"id": "edit_constant",
     "prompt": "In src/billing.py change TAX_RATE from 0.20 to 0.21.",
     "expect_tool": "edit_file",
     "args": {"path": {"contains": "billing.py"}, "find": {"contains": "0.20"},
              "replace": {"contains": "0.21"}},
     "script": ['{"tool": "edit_file", "args": {"path": "src/billing.py", '
                '"find": "TAX_RATE = 0.20", "replace": "TAX_RATE = 0.21"}}',
                "TAX_RATE is now 0.21 in src/billing.py."]},
    {"id": "remember_preference",
     "prompt": "Remember that my preferred timezone is Europe/Brussels.",
     "expect_tool": "remember",
     "args": {"key": {"present": True}, "value": {"contains": "brussels"}},
     "script": ['{"tool": "remember", "args": {"key": "user.timezone", "value": "Europe/Brussels"}}',
                "Noted: your preferred timezone is Europe/Brussels."]},
    {"id": "time_in_zone",
     "prompt": "What time is it in Tokyo right now? Use the clock tool.",
     "expect_tool": "current_time", "args": {"timezone": {"contains": "tokyo"}},
     "script": ['{"tool": "current_time", "args": {"timezone": "Asia/Tokyo"}}',
                "That is the current time in Tokyo."]},
    {"id": "no_tool_needed",
     "prompt": "In two sentences, what is a hash map?",
     "expect_tool": None,
     "answer_contains": ["key"],
     "script": ["A hash map stores key-value pairs and finds a value by hashing its key. "
                "Lookups, inserts and deletes take constant time on average."]},
    {"id": "no_invented_tool",
     "prompt": "Send an SMS to my colleague saying I will be ten minutes late.",
     "expect_tool": None,
     "script": ["I cannot send text messages from here: none of my tools can send an SMS. "
                "You could send it from your phone: \"I will be ten minutes late.\""]},
]


# --------------------------------------------------------------------------- #
# RAG: fixture documents and questions                                         #
# --------------------------------------------------------------------------- #
EVAL_RAG_DOCS: dict = {
    "kb/handbook/leave-policy.md": (
        "# Leave policy\n\nEmployees accrue 26 days of paid annual leave per calendar year. "
        "Unused leave carries over until 31 March of the following year. Requests longer "
        "than ten working days need approval from a department head.\n"),
    "kb/it/vpn-setup.md": (
        "# VPN setup\n\nThe corporate VPN gateway is vpn.example.internal and listens on UDP "
        "port 51820 (WireGuard). New devices are enrolled through the self-service portal; "
        "enrolment codes expire after 48 hours.\n"),
    "kb/product/release-notes.md": (
        "# Release notes\n\nVersion 3.2, codename Juniper, introduced offline sync and a dark "
        "theme. Version 3.3 removed the legacy XML export; use the JSON export instead.\n"),
    "kb/finance/expenses.md": (
        "# Expenses\n\nMeal expenses are reimbursed up to 35 euro per day during business "
        "travel. Receipts must be uploaded within 30 days of the expense.\n"),
}

EVAL_RAG_CASES: list = [
    {"id": "leave_days",
     "prompt": "How many days of paid annual leave do employees accrue per year?",
     "facts": ["26 days", "26"], "source": "kb/handbook/leave-policy.md",
     "script": ["Employees accrue 26 days of paid annual leave per calendar year "
                "(source: kb/handbook/leave-policy.md)."]},
    {"id": "leave_carry_over",
     "prompt": "Until when does unused annual leave carry over?",
     "facts": ["31 march", "march 31"], "source": "kb/handbook/leave-policy.md",
     "script": ["Unused leave carries over until 31 March of the following year "
                "(source: kb/handbook/leave-policy.md)."]},
    {"id": "vpn_port",
     "prompt": "Which UDP port does the corporate VPN gateway listen on?",
     "facts": ["51820"], "source": "kb/it/vpn-setup.md",
     "script": ["The VPN gateway listens on UDP port 51820, it is WireGuard "
                "(source: kb/it/vpn-setup.md)."]},
    {"id": "release_codename",
     "prompt": "What was the codename of version 3.2?",
     "facts": ["juniper"], "source": "kb/product/release-notes.md",
     "script": ["Version 3.2 was codenamed Juniper (source: kb/product/release-notes.md)."]},
    {"id": "meal_limit",
     "prompt": "What is the daily limit for meal expenses during business travel?",
     "facts": ["35 euro", "35"], "source": "kb/finance/expenses.md",
     "script": ["Meals are reimbursed up to 35 euro per day while travelling for business "
                "(source: kb/finance/expenses.md)."]},
]


# --------------------------------------------------------------------------- #
# Multi-turn task continuity                                                   #
# --------------------------------------------------------------------------- #
# Turn checks: language (the code block's language), no_drift (no block in
# another language), defines (names the code must define), clean (no error
# from codecheck), contains (substrings of the code), truncated (the reply was
# cut off at its budget), continued (the agent resumed instead of restarting).
EVAL_MULTI_TURN_CASES: list = [
    {"id": "artifact_then_edits",
     "turns": [
         {"user": "Write a Python function slugify(title) that lowercases a title and "
                  "replaces every run of non-alphanumeric characters with a single hyphen.",
          "checks": {"language": "python", "defines": ["slugify"], "clean": True}},
         {"user": "Now add a max_length parameter that trims the slug without cutting a "
                  "word in half.",
          "checks": {"language": "python", "no_drift": True, "defines": ["slugify"],
                     "contains": ["max_length"], "clean": True}},
         {"user": "Add a pytest test for it.",
          "checks": {"language": "python", "no_drift": True,
                     "defines": ["slugify", "test_slugify"], "clean": True}},
     ],
     "script": [
         "```python\nimport re\n\n\ndef slugify(title):\n"
         "    return re.sub(r\"[^a-z0-9]+\", \"-\", title.lower()).strip(\"-\")\n```",
         "```python\nimport re\n\n\ndef slugify(title, max_length=None):\n"
         "    slug = re.sub(r\"[^a-z0-9]+\", \"-\", title.lower()).strip(\"-\")\n"
         "    if max_length is None or len(slug) <= max_length:\n"
         "        return slug\n"
         "    cut = slug[:max_length + 1]\n"
         "    return cut.rsplit(\"-\", 1)[0] if \"-\" in cut else slug[:max_length]\n```",
         "```python\nimport re\n\n\ndef slugify(title, max_length=None):\n"
         "    slug = re.sub(r\"[^a-z0-9]+\", \"-\", title.lower()).strip(\"-\")\n"
         "    if max_length is None or len(slug) <= max_length:\n"
         "        return slug\n"
         "    cut = slug[:max_length + 1]\n"
         "    return cut.rsplit(\"-\", 1)[0] if \"-\" in cut else slug[:max_length]\n\n\n"
         "def test_slugify():\n"
         "    assert slugify(\"Hello, World!\") == \"hello-world\"\n"
         "    assert slugify(\"one two three\", max_length=8) == \"one-two\"\n```",
     ]},
    {"id": "continue_after_truncation",
     "turns": [
         {"user": "Write a JavaScript module with two functions, debounce(fn, ms) and "
                  "throttle(fn, ms), and export both.",
          "checks": {"language": "javascript", "truncated": True}},
         {"user": "continue",
          "checks": {"language": "javascript", "continued": True,
                     "defines": ["debounce", "throttle"], "contains": ["module.exports"],
                     "clean": True}},
     ],
     "script": [
         {"text": "```javascript\nfunction debounce(fn, ms) {\n  let timer;\n"
                  "  return function (...args) {\n    clearTimeout(timer);\n"
                  "    timer = setTimeout(() => fn.apply(this, args), ms);\n  };\n}\n\n"
                  "function throttle(fn, ms) {\n  let last = 0;\n",
          "finish": "length"},
         "  return function (...args) {\n    const now = Date.now();\n"
         "    if (now - last >= ms) {\n      last = now;\n      fn.apply(this, args);\n"
         "    }\n  };\n}\n\nmodule.exports = { debounce, throttle };\n```",
     ]},
    {"id": "language_drift",
     "turns": [
         {"user": "Write a Go function Reverse(s string) string that reverses a string rune "
                  "by rune.",
          "checks": {"language": "go", "defines": ["Reverse"], "clean": True}},
         {"user": "Make it keep combining marks attached to their base letter.",
          "checks": {"language": "go", "no_drift": True, "defines": ["Reverse"],
                     "contains": ["unicode"], "clean": True}},
     ],
     "script": [
         "```go\npackage text\n\nfunc Reverse(s string) string {\n\tr := []rune(s)\n"
         "\tfor i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 {\n\t\tr[i], r[j] = r[j], r[i]\n"
         "\t}\n\treturn string(r)\n}\n```",
         "```go\npackage text\n\nimport \"unicode\"\n\n"
         "func Reverse(s string) string {\n\tvar clusters [][]rune\n"
         "\tfor _, r := range s {\n"
         "\t\tif unicode.Is(unicode.Mn, r) && len(clusters) > 0 {\n"
         "\t\t\tlast := len(clusters) - 1\n"
         "\t\t\tclusters[last] = append(clusters[last], r)\n\t\t\tcontinue\n\t\t}\n"
         "\t\tclusters = append(clusters, []rune{r})\n\t}\n"
         "\tout := make([]rune, 0, len(s))\n"
         "\tfor i := len(clusters) - 1; i >= 0; i-- {\n"
         "\t\tout = append(out, clusters[i]...)\n\t}\n\treturn string(out)\n}\n```",
     ]},
]


# --------------------------------------------------------------------------- #
# Routing: which intent, lane, tools and retrieval a message gets              #
# --------------------------------------------------------------------------- #
# Every case states what the turn must get before the model writes a word:
#   intent     an intents.INTENT_NAMES name, or None for "the lexicon stays out
#              of it" (the existing pipeline and the router decide)
#   lane       the Intent's lane (tools, prose, reply, clarify, document), or
#              "pipeline" when intent is None
#   offer      tools the turn must be offered; a name the sandbox registry does
#              not have (a P3 tool not built yet) is reported, not required
#   not_offer  tools the turn must not be offered
#   retrieval  the knowledge-base injection rule: off, kb, or auto (today's)
# Optional: attachments (names from EVAL_ROUTING_ATTACHMENTS), history (a
# previous exchange exists), agent (scripted mode also runs the Agent and
# checks the lane it took and the tools its system prompt lists) and script
# (that run's canned model outputs).
#
# Why "the agent is told the project folder is mlx-h": the third reviewed
# message was typed in this repository, and the folder name is itself a
# codebase term ("check mlx-h for design mistakes").
EVAL_ROUTING_PROJECT = "mlx-h"

# The reviewed turn 1 attachment: a markdown skill with SKILL.md frontmatter.
EVAL_ROUTING_ATTACHMENTS: dict = {
    "devils-advocate.md": (
        "---\nname: devils-advocate\n"
        "description: Argue against a plan to find its weakest assumption before anyone "
        "commits to it.\n---\n\n# Devil's advocate\n\n"
        "- Name the one assumption that sinks the plan if it is false, and argue that it is "
        "false.\n"
        "- End with the strongest counter-proposal, not a list of nitpicks.\n"),
}

# The previous exchange for cases with history (a referent for "this").
EVAL_ROUTING_HISTORY: list = [
    {"role": "user", "content": "Give me a short checklist for reviewing a pull request."},
    {"role": "assistant", "content": (
        "1. Read the description and the linked issue.\n2. Run the tests locally.\n"
        "3. Check error handling and naming.\n4. Leave one summary comment.")},
]

# The sandbox project and knowledge base. Each KB document shares words with
# one reviewed message, so a turn whose retrieval rule is "off" would show the
# document's path in its prompt if auto-RAG ran anyway (reviewed turns 1 and 3
# got exactly such an unrelated passage injected).
EVAL_ROUTING_WORKSPACE_FILES: dict = {
    "README.md": "# mlx-h\n\nA local model server, agent and trainer.\n",
    "local_llm/router.py": (
        "def route(message, tools):\n"
        "    \"\"\"Pick a lane for a message.\"\"\"\n"
        "    return \"answer\" if not tools else tools[0]\n"),
}
EVAL_ROUTING_KB_DOCS: dict = {
    "kb/claude-skills/devils-advocate/SKILL.md": (
        "# Devil's advocate skill\n\nAdd this skill to your skills library to argue "
        "against a plan. Skills are saved procedures.\n"),
    "kb/notes/codebase-design-review.md": (
        "# Codebase design review\n\nCheck the codebase for design mistakes and summarize "
        "them: layering, naming, error handling.\n"),
    "kb/facts/useful-facts.md": (
        "# Useful facts\n\nAn overview of useful facts in the knowledge base: the office "
        "opens at 8, the VPN port is 51820, leave carries over until 31 March.\n"),
}

_EVAL_RT_SKILL_NOT = ["web_search", "search_docs", "write_file", "create_document"]
_EVAL_RT_KB_NOT = ["web_search", "fetch_url", "write_file", "edit_file"]
_EVAL_RT_CODE_NOT = ["web_search", "fetch_url", "search_docs", "write_file", "edit_file",
                "create_document"]
_EVAL_RT_HISTORY_NOT = ["search_docs", "web_search"]
# codebase_write reads the project and writes into it; the web knows nothing
# about a folder on this disk, and a README is not a create_document job.
_EVAL_RT_WRITE_OFFER = ["list_files", "read_file", "write_file", "edit_file"]
_EVAL_RT_WRITE_NOT = ["web_search", "fetch_url", "index_url", "search_docs", "create_document"]

EVAL_ROUTING_CASES: list = [
    # -- The three reviewed failures (upgrade/ROUTING_AND_TOOLING.md section 1).
    {"id": "reviewed_add_to_skills", "message": "add this to skills",
     "attachments": ["devils-advocate.md"],
     "intent": "skill_save", "lane": "tools", "offer": ["save_skill"],
     "not_offer": _EVAL_RT_SKILL_NOT, "retrieval": "off", "agent": True,
     "script": ['{"tool": "save_skill", "args": {"name": "devils-advocate", "description": '
                '"Argue against a plan to find its weakest assumption before anyone commits '
                'to it.", "body": "# Devil\'s advocate"}}',
                "Saved the devils-advocate skill."]},
    {"id": "reviewed_kb_pdf",
     "message": "use the knowledge base to craft a pdf showing an overview of usefull facts "
                "in the knowledge base",
     "intent": "document_create", "lane": "document",
     "offer": ["create_document", "search_docs", "kb_overview"],
     "not_offer": ["web_search", "fetch_url", "write_file", "edit_file"],
     "retrieval": "kb", "agent": True,
     "script": ['{"tool": "search_docs", "args": {"query": "useful facts"}}',
                '{"tool": "create_document", "args": {"path": "knowledge-base-facts.pdf", '
                '"content": "# Useful facts\\n\\n- The office opens at 8.\\n- The VPN port '
                'is 51820."}}',
                "Created knowledge-base-facts.pdf from the knowledge base."]},
    {"id": "reviewed_codebase_review",
     "message": "check the codebase for design mistakes and summarize",
     "intent": "codebase_review", "lane": "tools",
     "offer": ["list_files", "search_files", "read_file", "repo_map"],
     "not_offer": _EVAL_RT_CODE_NOT, "retrieval": "off", "agent": True,
     "script": ['{"tool": "list_files", "args": {"path": "."}}',
                "The project has one module, local_llm/router.py; its route() returns a tool "
                "name or \"answer\" from the same function, which mixes two return types."]},

    # -- skill_save / skill_use / skill_list
    {"id": "skill_save_after_answer", "message": "save this as a skill", "history": True,
     "intent": "skill_save", "lane": "tools", "offer": ["save_skill"],
     "not_offer": _EVAL_RT_SKILL_NOT, "retrieval": "off"},
    {"id": "skill_save_attached_file", "message": "add the attached file to my skills",
     "attachments": ["devils-advocate.md"],
     "intent": "skill_save", "lane": "tools", "offer": ["save_skill"],
     "not_offer": _EVAL_RT_SKILL_NOT, "retrieval": "off"},
    {"id": "skill_save_no_referent", "message": "keep this as a skill",
     "intent": "skill_save", "lane": "clarify", "offer": [], "not_offer": ["save_skill"],
     "retrieval": "off"},
    {"id": "skill_use_named", "message": "use the devils-advocate skill to review my plan",
     "intent": "skill_use", "lane": "tools", "offer": ["load_skill"],
     "not_offer": ["save_skill", "web_search", "search_docs"], "retrieval": "off"},
    {"id": "skill_use_apply", "message": "apply the devils-advocate skill to this plan",
     "intent": "skill_use", "lane": "tools", "offer": ["load_skill"],
     "not_offer": ["save_skill", "web_search"], "retrieval": "off"},
    {"id": "skill_list_mine", "message": "list my skills",
     "intent": "skill_list", "lane": "tools", "offer": ["list_skills", "search_skills"],
     "not_offer": ["save_skill", "web_search", "search_docs"], "retrieval": "off"},
    {"id": "skill_list_installed", "message": "which skills are installed",
     "intent": "skill_list", "lane": "tools", "offer": ["list_skills", "search_skills"],
     "not_offer": ["save_skill", "web_search"], "retrieval": "off"},
    {"id": "skill_list_slash", "message": "/skills",
     "intent": "skill_list", "lane": "tools", "offer": ["list_skills", "search_skills"],
     "not_offer": ["save_skill"], "retrieval": "off"},

    # -- memory_write / memory_read / memory_forget
    {"id": "memory_write_that", "message": "remember that my timezone is Europe/Brussels",
     "intent": "memory_write", "lane": "tools", "offer": ["remember"],
     "not_offer": ["forget", "web_search", "search_docs"], "retrieval": "off"},
    {"id": "memory_write_plain", "message": "remember my dog's name is Pixel",
     "intent": "memory_write", "lane": "tools", "offer": ["remember"],
     "not_offer": ["forget", "web_search"], "retrieval": "off"},
    {"id": "memory_read_about_me", "message": "what do you remember about me",
     "intent": "memory_read", "lane": "tools", "offer": ["recall_memory"],
     "not_offer": ["remember", "forget", "web_search"], "retrieval": "off"},
    {"id": "memory_read_notes", "message": "summarize my notes",
     "intent": "memory_read", "lane": "tools", "offer": ["recall_memory"],
     "not_offer": ["remember", "forget", "web_search"], "retrieval": "off"},
    {"id": "memory_read_slash", "message": "/mem",
     "intent": "memory_read", "lane": "tools", "offer": ["recall_memory"],
     "not_offer": ["remember", "forget"], "retrieval": "off"},
    {"id": "memory_forget_key", "message": "forget my timezone",
     "intent": "memory_forget", "lane": "tools", "offer": ["forget", "recall_memory"],
     "not_offer": ["remember", "web_search"], "retrieval": "off"},

    # -- history
    {"id": "history_past_chats", "message": "search past chats for the docker discussion",
     "intent": "history", "lane": "tools", "offer": ["search_past_conversations"],
     "not_offer": _EVAL_RT_HISTORY_NOT, "retrieval": "off"},
    {"id": "history_discussed", "message": "what did we discuss yesterday about mlx",
     "intent": "history", "lane": "tools", "offer": ["search_past_conversations"],
     "not_offer": _EVAL_RT_HISTORY_NOT, "retrieval": "off"},
    {"id": "history_last_time",
     "message": "last time we talked about quantization, what did you suggest",
     "intent": "history", "lane": "tools", "offer": ["search_past_conversations"],
     "not_offer": _EVAL_RT_HISTORY_NOT, "retrieval": "off"},
    {"id": "history_slash", "message": "/history docker",
     "intent": "history", "lane": "tools", "offer": ["search_past_conversations"],
     "not_offer": _EVAL_RT_HISTORY_NOT, "retrieval": "off"},

    # -- kb_write / kb_overview / kb_query
    {"id": "kb_write_folder", "message": "index the docs folder",
     "intent": "kb_write", "lane": "tools", "offer": ["index_docs"],
     "not_offer": ["web_search", "write_file", "search_docs"], "retrieval": "off"},
    {"id": "kb_write_url", "message": "add https://example.com/docs to the knowledge base",
     "intent": "kb_write", "lane": "tools", "offer": ["index_url"],
     "not_offer": ["web_search", "fetch_url", "write_file"], "retrieval": "off"},
    {"id": "kb_write_ingest", "message": "ingest the reports folder into the knowledge base",
     "intent": "kb_write", "lane": "tools", "offer": ["index_docs"],
     "not_offer": ["web_search", "write_file"], "retrieval": "off"},
    {"id": "kb_overview_whats_in", "message": "what is in my knowledge base",
     "intent": "kb_overview", "lane": "tools", "offer": ["kb_overview", "search_docs"],
     "not_offer": _EVAL_RT_KB_NOT, "retrieval": "kb"},
    {"id": "kb_overview_how_many", "message": "how many documents are in my knowledge base",
     "intent": "kb_overview", "lane": "tools", "offer": ["kb_overview", "search_docs"],
     "not_offer": _EVAL_RT_KB_NOT, "retrieval": "kb"},
    {"id": "kb_overview_summarize", "message": "summarize the knowledge base",
     "intent": "kb_overview", "lane": "tools", "offer": ["kb_overview", "search_docs"],
     "not_offer": _EVAL_RT_KB_NOT, "retrieval": "kb"},
    {"id": "kb_query_search", "message": "search the knowledge base for onboarding",
     "intent": "kb_query", "lane": "tools", "offer": ["search_docs"],
     "not_offer": _EVAL_RT_KB_NOT, "retrieval": "kb"},
    {"id": "kb_query_my_documents", "message": "search my documents for pricing",
     "intent": "kb_query", "lane": "tools", "offer": ["search_docs"],
     "not_offer": _EVAL_RT_KB_NOT, "retrieval": "kb"},
    {"id": "kb_query_according", "message": "according to my docs, what is book-to-skill",
     "intent": "kb_query", "lane": "tools", "offer": ["search_docs"],
     "not_offer": _EVAL_RT_KB_NOT, "retrieval": "kb"},
    {"id": "kb_query_indexed", "message": "check the indexed docs for the VPN port",
     "intent": "kb_query", "lane": "tools", "offer": ["search_docs"],
     "not_offer": _EVAL_RT_KB_NOT, "retrieval": "kb"},
    {"id": "kb_query_slash", "message": "/kb what does the knowledge base say about tdd",
     "intent": "kb_query", "lane": "tools", "offer": ["search_docs"],
     "not_offer": _EVAL_RT_KB_NOT, "retrieval": "kb"},

    # -- codebase_lookup / codebase_review
    {"id": "codebase_lookup_snake", "message": "where is routing_override defined",
     "intent": "codebase_lookup", "lane": "tools", "offer": ["search_files"],
     "not_offer": _EVAL_RT_CODE_NOT, "retrieval": "off"},
    {"id": "codebase_lookup_camel", "message": "where is ToolRegistry defined",
     "intent": "codebase_lookup", "lane": "tools", "offer": ["search_files"],
     "not_offer": _EVAL_RT_CODE_NOT, "retrieval": "off"},
    {"id": "codebase_lookup_usages", "message": "find usages of build_agent_system_prompt",
     "intent": "codebase_lookup", "lane": "tools", "offer": ["search_files"],
     "not_offer": _EVAL_RT_CODE_NOT, "retrieval": "off"},
    {"id": "codebase_review_overview", "message": "give me an overview of this repo",
     "intent": "codebase_review", "lane": "tools",
     "offer": ["list_files", "search_files", "read_file", "repo_map"],
     "not_offer": _EVAL_RT_CODE_NOT, "retrieval": "off"},
    {"id": "codebase_review_project", "message": "review this project for design mistakes",
     "intent": "codebase_review", "lane": "tools",
     "offer": ["list_files", "search_files", "read_file", "repo_map"],
     "not_offer": _EVAL_RT_CODE_NOT, "retrieval": "off"},
    {"id": "codebase_review_basename", "message": "check mlx-h for design mistakes",
     "intent": "codebase_review", "lane": "tools",
     "offer": ["list_files", "search_files", "read_file", "repo_map"],
     "not_offer": _EVAL_RT_CODE_NOT, "retrieval": "off"},
    {"id": "codebase_review_audit", "message": "audit our codebase for security issues",
     "intent": "codebase_review", "lane": "tools",
     "offer": ["list_files", "search_files", "read_file"],
     "not_offer": _EVAL_RT_CODE_NOT, "retrieval": "off"},
    {"id": "codebase_review_summarize_code", "message": "summarize the code",
     "intent": "codebase_review", "lane": "tools",
     "offer": ["list_files", "search_files", "read_file"],
     "not_offer": _EVAL_RT_CODE_NOT, "retrieval": "off"},

    # -- codebase_write. The first message is the turn that matched nothing,
    #    ran the full catalogue, overwrote a 100 KB README.md with 841
    #    characters and fetched a GitHub URL the model made up. Each case
    #    needs the file tools with both writes, and no web tool at all.
    {"id": "codebase_write_check_and_readme",
     "message": "check the codebase and write a readme.md",
     "intent": "codebase_write", "lane": "tools", "offer": _EVAL_RT_WRITE_OFFER,
     "not_offer": _EVAL_RT_WRITE_NOT, "retrieval": "off"},
    {"id": "codebase_write_check_and_readme_caps",
     "message": "Check the codebase and write a README.md.",
     "intent": "codebase_write", "lane": "tools", "offer": _EVAL_RT_WRITE_OFFER,
     "not_offer": _EVAL_RT_WRITE_NOT, "retrieval": "off"},
    {"id": "codebase_write_readme_project", "message": "write a readme for this project",
     "intent": "codebase_write", "lane": "tools", "offer": _EVAL_RT_WRITE_OFFER,
     "not_offer": _EVAL_RT_WRITE_NOT, "retrieval": "off"},
    {"id": "codebase_write_update_readme",
     "message": "update the README.md for the codebase",
     "intent": "codebase_write", "lane": "tools", "offer": _EVAL_RT_WRITE_OFFER,
     "not_offer": _EVAL_RT_WRITE_NOT, "retrieval": "off"},
    {"id": "codebase_write_docs_basename", "message": "generate docs for mlx-h",
     "intent": "codebase_write", "lane": "tools", "offer": _EVAL_RT_WRITE_OFFER,
     "not_offer": _EVAL_RT_WRITE_NOT, "retrieval": "off"},
    {"id": "codebase_write_contributing", "message": "add a CONTRIBUTING.md to the repo",
     "intent": "codebase_write", "lane": "tools", "offer": _EVAL_RT_WRITE_OFFER,
     "not_offer": _EVAL_RT_WRITE_NOT, "retrieval": "off"},
    {"id": "codebase_write_architecture",
     "message": "draft an architecture doc for the codebase",
     "intent": "codebase_write", "lane": "tools", "offer": _EVAL_RT_WRITE_OFFER,
     "not_offer": _EVAL_RT_WRITE_NOT, "retrieval": "off"},
    {"id": "codebase_write_improve_readme", "message": "improve the readme of the codebase",
     "intent": "codebase_write", "lane": "tools", "offer": _EVAL_RT_WRITE_OFFER,
     "not_offer": _EVAL_RT_WRITE_NOT, "retrieval": "off"},
    {"id": "codebase_write_look_then_generate",
     "message": "look at the repo and generate a README",
     "intent": "codebase_write", "lane": "tools", "offer": _EVAL_RT_WRITE_OFFER,
     "not_offer": _EVAL_RT_WRITE_NOT, "retrieval": "off"},
    {"id": "codebase_write_add_section",
     "message": "add an installation section to the README of this repo",
     "intent": "codebase_write", "lane": "tools", "offer": _EVAL_RT_WRITE_OFFER,
     "not_offer": _EVAL_RT_WRITE_NOT, "retrieval": "off"},
    {"id": "codebase_write_changelog", "message": "write a CHANGELOG for the repo",
     "intent": "codebase_write", "lane": "tools", "offer": _EVAL_RT_WRITE_OFFER,
     "not_offer": _EVAL_RT_WRITE_NOT, "retrieval": "off"},
    {"id": "codebase_write_slash", "message": "/code write a readme",
     "intent": "codebase_write", "lane": "tools", "offer": _EVAL_RT_WRITE_OFFER,
     "not_offer": _EVAL_RT_WRITE_NOT, "retrieval": "off"},
    # Its neighbours keep their lanes: a review without a write, a PDF about
    # the codebase, and a code request that only mentions documentation.
    {"id": "codebase_review_check_summarize", "message": "check the codebase and summarize",
     "intent": "codebase_review", "lane": "tools",
     "offer": ["list_files", "search_files", "read_file"],
     "not_offer": _EVAL_RT_CODE_NOT, "retrieval": "off"},
    {"id": "document_pdf_about_codebase", "message": "create a pdf about the codebase",
     "intent": "document_create", "lane": "document",
     "offer": ["create_document", "list_files", "read_file"],
     "not_offer": ["web_search", "fetch_url", "search_docs", "write_file"], "retrieval": "off"},
    {"id": "plain_script_writes_docs",
     "message": "write a python script that generates the docs for this repo",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},
    {"id": "plain_readme_howto", "message": "how do I write a good readme for my repo",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},
    {"id": "plain_readme_no_project", "message": "write a readme",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},

    # -- schedule (task_scheduler is not registered in the sandbox, so the
    #    intent answers how to create the task instead of pretending to)
    {"id": "schedule_every_morning",
     "message": "create a task that runs every morning to search news about MLX",
     "intent": "schedule", "lane": "reply", "offer": [], "not_offer": ["web_search"],
     "retrieval": "off"},
    {"id": "schedule_remind", "message": "remind me every morning at 8 to check the build",
     "intent": "schedule", "lane": "reply", "offer": [], "not_offer": [],
     "retrieval": "off"},
    {"id": "schedule_weekly", "message": "every Friday send me a summary of the week's chats",
     "intent": "schedule", "lane": "reply", "offer": [], "not_offer": [],
     "retrieval": "off"},

    # -- document_create, with and without a research source
    {"id": "document_kb_spreadsheet",
     "message": "create an excel sheet of the documents in the knowledge base",
     "intent": "document_create", "lane": "document",
     "offer": ["create_document", "search_docs", "kb_overview"],
     "not_offer": ["web_search", "fetch_url", "write_file"], "retrieval": "kb"},
    {"id": "document_kb_deck", "message": "make a slide deck about the knowledge base",
     "intent": "document_create", "lane": "document",
     "offer": ["create_document", "search_docs", "kb_overview"],
     "not_offer": ["web_search", "fetch_url", "write_file"], "retrieval": "kb"},
    {"id": "document_plain_pdf", "message": "write a pdf report about the history of Rome",
     "intent": "document_create", "lane": "document", "offer": ["create_document"],
     "not_offer": ["search_docs", "kb_overview", "search_past_conversations", "write_file"],
     "retrieval": "auto"},
    {"id": "document_previous_answer", "message": "export the summary as a docx",
     "history": True, "intent": "document_create", "lane": "document",
     "offer": ["create_document"], "not_offer": ["search_docs", "write_file"],
     "retrieval": "auto"},
    {"id": "document_slash_pdf", "message": "/pdf a one page summary of our roadmap",
     "intent": "document_create", "lane": "document", "offer": ["create_document"],
     "not_offer": ["write_file"], "retrieval": "auto"},

    # -- self_describe / disabled_capability / admin_action / conv_action:
    #    deterministic replies, no model call and no tools
    {"id": "self_describe_tools", "message": "what tools do you have",
     "intent": "self_describe", "lane": "reply", "offer": [], "not_offer": [],
     "retrieval": "off"},
    {"id": "self_describe_can", "message": "what can you do",
     "intent": "self_describe", "lane": "reply", "offer": [], "not_offer": [],
     "retrieval": "off"},
    {"id": "disabled_tests", "message": "run the tests",
     "intent": "disabled_capability", "lane": "reply", "offer": [], "not_offer": [],
     "retrieval": "off"},
    {"id": "disabled_shell", "message": "run ls -la in the shell",
     "intent": "disabled_capability", "lane": "reply", "offer": [], "not_offer": [],
     "retrieval": "off"},
    {"id": "disabled_git_commit", "message": "commit my changes",
     "intent": "disabled_capability", "lane": "reply", "offer": [], "not_offer": [],
     "retrieval": "off"},
    {"id": "disabled_email", "message": "check my email",
     "intent": "disabled_capability", "lane": "reply", "offer": [], "not_offer": [],
     "retrieval": "off"},
    {"id": "disabled_image", "message": "generate an image of a cat",
     "intent": "disabled_capability", "lane": "reply", "offer": [], "not_offer": [],
     "retrieval": "off"},
    {"id": "disabled_database",
     "message": "query the database for the number of conversations",
     "intent": "disabled_capability", "lane": "reply", "offer": [], "not_offer": [],
     "retrieval": "off"},
    {"id": "admin_switch_model", "message": "switch to the coder model",
     "intent": "admin_action", "lane": "reply", "offer": [], "not_offer": [],
     "retrieval": "off"},
    {"id": "admin_api_token", "message": "create an api token for my script",
     "intent": "admin_action", "lane": "reply", "offer": [], "not_offer": [],
     "retrieval": "off"},
    {"id": "admin_always_allow", "message": "always allow write_file",
     "intent": "admin_action", "lane": "reply", "offer": [], "not_offer": [],
     "retrieval": "off"},
    {"id": "admin_add_user", "message": "add a user called bob",
     "intent": "admin_action", "lane": "reply", "offer": [], "not_offer": [],
     "retrieval": "off"},
    {"id": "admin_remote_access", "message": "enable remote access",
     "intent": "admin_action", "lane": "reply", "offer": [], "not_offer": [],
     "retrieval": "off"},
    {"id": "admin_mcp", "message": "configure an MCP server",
     "intent": "admin_action", "lane": "reply", "offer": [], "not_offer": [],
     "retrieval": "off"},
    {"id": "conv_export", "message": "export this conversation as markdown",
     "intent": "conv_action", "lane": "reply", "offer": [], "not_offer": [],
     "retrieval": "off"},
    {"id": "conv_regenerate", "message": "regenerate the last answer",
     "intent": "conv_action", "lane": "reply", "offer": [], "not_offer": [],
     "retrieval": "off"},
    {"id": "conv_rename", "message": "rename this chat to MLX notes",
     "intent": "conv_action", "lane": "reply", "offer": [], "not_offer": [],
     "retrieval": "off"},
    {"id": "conv_delete", "message": "delete this conversation",
     "intent": "conv_action", "lane": "reply", "offer": [], "not_offer": [],
     "retrieval": "off"},

    # -- Ordinary questions the lexicon must leave to the pipeline, many of
    #    them carrying a UI word in another sense (a CV's skills, a cron job,
    #    a notes app, someone else's knowledge base, a pasted code block).
    {"id": "plain_hash_map", "message": "In two sentences, what is a hash map?",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},
    {"id": "plain_capital", "message": "What is the capital of Belgium?",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},
    {"id": "plain_python_function",
     "message": "Write a Python function that reverses a string",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},
    {"id": "plain_resume_skill", "message": "How do I add a skill to my resume?",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},
    {"id": "plain_engineer_skills", "message": "what are good skills for a data engineer",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},
    {"id": "plain_kb_concept",
     "message": "Explain how a knowledge base differs from a database",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},
    {"id": "plain_zendesk_kb",
     "message": "Explain the knowledge base article format used by Zendesk",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},
    {"id": "plain_cron", "message": "how do I schedule a cron job on Linux",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},
    {"id": "plain_web_search", "message": "search the web for mlx 0.32 release notes",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},
    {"id": "plain_weather", "message": "what is the weather in Brussels tomorrow",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},
    {"id": "plain_fix_bug", "message": "fix the bug in agent.py",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},
    {"id": "plain_prompt_library", "message": "use my code-review prompt",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},
    {"id": "plain_remember_when",
     "message": "remember when we used floppy disks? write a short poem about it",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},
    {"id": "plain_forget_it", "message": "Forget it, just tell me a joke",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},
    {"id": "plain_pandas_index", "message": "How do I index a pandas DataFrame?",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},
    {"id": "plain_roman_history", "message": "Summarize the history of the Roman empire",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},
    {"id": "plain_notes_app", "message": "My notes app keeps crashing on iOS, any ideas?",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},
    {"id": "plain_birdhouse_tools", "message": "What tools do I need to build a birdhouse?",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},
    {"id": "plain_pasted_code",
     "message": "Can you check my code for bugs?\n```python\nprint(1)\n```",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},
    {"id": "plain_github_review", "message": "How do I review a pull request on GitHub?",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},
    {"id": "plain_daily_routine", "message": "Give me a daily workout routine",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},
    # A GET is what fetch_url does, and read_file reads .xlsx, so with those
    # two registered neither request is a switched-off capability: saying "not
    # enabled" would be false. They stay with the router, which offers both.
    {"id": "plain_http_get", "message": "make an http request to https://api.example.com",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},
    {"id": "plain_read_spreadsheet", "message": "read this spreadsheet",
     "intent": None, "lane": "pipeline", "offer": [], "not_offer": [], "retrieval": "auto"},
    # -- Phrasings that were gaps in the first cut of the lexicon (P1 lists
    #    each trigger); scored since classify agrees.
    {"id": "memory_note_that", "message": "note that I prefer metric units",
     "intent": "memory_write", "lane": "tools", "offer": ["remember"], "not_offer": [],
     "retrieval": "off"},
    {"id": "memory_dont_forget", "message": "don't forget my standup is at 9:30",
     "intent": "memory_write", "lane": "tools", "offer": ["remember"], "not_offer": [],
     "retrieval": "off"},
    {"id": "memory_forget_that", "message": "forget that I live in Ghent",
     "intent": "memory_forget", "lane": "tools", "offer": ["forget"], "not_offer": [],
     "retrieval": "off"},
    {"id": "kb_my_library", "message": "what does my library say about vacation policy",
     "intent": "kb_query", "lane": "tools", "offer": ["search_docs"], "not_offer": [],
     "retrieval": "kb"},
    {"id": "codebase_review_source", "message": "review the source for code smells",
     "intent": "codebase_review", "lane": "tools", "offer": ["list_files"],
     "not_offer": ["search_docs"], "retrieval": "off"},
    {"id": "codebase_slash_lookup", "message": "/code where is classify used",
     "intent": "codebase_lookup", "lane": "tools", "offer": ["search_files"],
     "not_offer": [], "retrieval": "off"},
    {"id": "document_from_codebase",
     "message": "craft a pdf report of the codebase architecture",
     "intent": "document_create", "lane": "document",
     "offer": ["create_document", "list_files", "read_file"],
     "not_offer": ["web_search", "search_docs"], "retrieval": "off"},
    {"id": "document_from_history",
     "message": "make a word document summarizing our past chats about MLX",
     "intent": "document_create", "lane": "document",
     "offer": ["create_document", "search_past_conversations"],
     "not_offer": ["web_search", "search_docs"], "retrieval": "off"},
]

# Phrasings where intents.classify() disagrees with the P1 table today. They
# are findings for the lexicon, kept out of the scored suite so CI stays a
# regression signal; tests/test_routing_suite.py runs them as expected
# failures, and a case moves into EVAL_ROUTING_CASES once classify agrees.
# Empty since phase B integration: nine of the first eleven now classify as P1
# says and were promoted, and the http and spreadsheet ones were wrong while
# fetch_url and read_file are registered (see plain_http_get).
EVAL_ROUTING_GAPS: list = []


# --------------------------------------------------------------------------- #
# Parsers: toolparse fixtures                                                  #
# --------------------------------------------------------------------------- #
EVAL_PARSER_KNOWN = ("web_search", "get_weather", "calculator", "read_file")

# (id, text, expected [[name, args], ...], format of the first call)
EVAL_PARSER_FORMATS: list = [
    ("canonical", '{"tool": "web_search", "args": {"query": "mlx"}}',
     [["web_search", {"query": "mlx"}]], "json"),
    ("name_parameters", '{"name": "web_search", "parameters": {"query": "mlx"}}',
     [["web_search", {"query": "mlx"}]], "json"),
    ("string_args", '{"tool": "web_search", "args": "{\\"query\\": \\"mlx\\"}"}',
     [["web_search", {"query": "mlx"}]], "json"),
    ("openai_function",
     '{"type": "function", "function": {"name": "web_search", "arguments": "{\\"query\\": \\"mlx\\"}"}}',
     [["web_search", {"query": "mlx"}]], "openai"),
    ("openai_tool_calls",
     '{"tool_calls": [{"id": "c1", "function": {"name": "web_search", "arguments": "{\\"query\\": \\"a\\"}"}},'
     ' {"function": {"name": "calculator", "arguments": {"expression": "1+1"}}}]}',
     [["web_search", {"query": "a"}], ["calculator", {"expression": "1+1"}]], "openai"),
    ("hermes", 'Let me look.\n<tool_call>\n{"name": "web_search", "arguments": {"query": "mlx"}}\n</tool_call>',
     [["web_search", {"query": "mlx"}]], "hermes"),
    ("hermes_two",
     '<tool_call>{"name": "web_search", "arguments": {"query": "a"}}</tool_call>\n'
     '<tool_call>{"name": "get_weather", "arguments": {"city": "Ghent"}}</tool_call>',
     [["web_search", {"query": "a"}], ["get_weather", {"city": "Ghent"}]], "hermes"),
    ("qwen3_coder",
     "<tool_call>\n<function=get_weather>\n<parameter=city>\nGhent\n</parameter>\n</function>\n</tool_call>",
     [["get_weather", {"city": "Ghent"}]], "qwen3_coder"),
    ("llama3_python_tag", '<|python_tag|>{"name": "web_search", "parameters": {"query": "mlx"}}<|eom_id|>',
     [["web_search", {"query": "mlx"}]], "llama3"),
    ("mistral_array", '[TOOL_CALLS] [{"name": "web_search", "arguments": {"query": "mlx"}}]',
     [["web_search", {"query": "mlx"}]], "mistral"),
    ("mistral_args", '[TOOL_CALLS]web_search[ARGS]{"query": "mlx"}',
     [["web_search", {"query": "mlx"}]], "mistral"),
    ("pythonic", '[web_search(query="mlx"), get_weather(city="Ghent", days=3)]',
     [["web_search", {"query": "mlx"}], ["get_weather", {"city": "Ghent", "days": 3}]], "pythonic"),
    ("pythonic_tokens", '<|tool_call_start|>[web_search(query="mlx")]<|tool_call_end|>',
     [["web_search", {"query": "mlx"}]], "pythonic"),
    ("gemma_tool_code", '```tool_code\nweb_search(query="mlx")\n```',
     [["web_search", {"query": "mlx"}]], "gemma"),
    ("function_gemma",
     "<start_function_call>call:get_weather{city:<escape>Ghent<escape>}<end_function_call>",
     [["get_weather", {"city": "Ghent"}]], "gemma"),
    ("deepseek_v3",
     "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function"
     "<｜tool▁sep｜>web_search\n"
     '```json\n{"query": "mlx"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
     [["web_search", {"query": "mlx"}]], "deepseek"),
    ("kimi_k2",
     "<|tool_calls_section_begin|><|tool_call_begin|>functions.web_search:0"
     '<|tool_call_argument_begin|>{"query": "mlx"}<|tool_call_end|><|tool_calls_section_end|>',
     [["web_search", {"query": "mlx"}]], "kimi"),
    ("glm45", "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Ghent</arg_value>\n</tool_call>",
     [["get_weather", {"city": "Ghent"}]], "glm"),
    ("gpt_oss_harmony",
     "<|channel|>analysis<|message|>Need search.<|end|><|start|>assistant<|channel|>commentary "
     'to=functions.web_search <|constrain|>json<|message|>{"query": "mlx"}<|call|>',
     [["web_search", {"query": "mlx"}]], "harmony"),
]

# (id, broken text, repair tag that must be recorded, expected args)
EVAL_PARSER_REPAIRS: list = [
    ("single_quotes", "{'tool': 'web_search', 'args': {'query': 'mlx',}}", "single_quotes",
     {"query": "mlx"}),
    ("trailing_commas", "{'tool': 'web_search', 'args': {'query': 'mlx',}}", "trailing_commas",
     {"query": "mlx"}),
    ("smart_quotes", "{“tool”: “web_search”, “args”: {“query”: “mlx”}}",
     "smart_quotes", {"query": "mlx"}),
    ("python_literals",
     '{"tool": "calculator", "args": {"expression": "1+1", "exact": True, "x": None}}',
     "python_literals", {"expression": "1+1", "exact": True, "x": None}),
    ("unquoted_keys", '{tool: "web_search", args: {query: "mlx"}}', "unquoted_keys",
     {"query": "mlx"}),
    ("raw_control_chars", '{"tool": "read_file", "args": {"text": "a\nb"}}', "raw_control_chars",
     {"text": "a\nb"}),
    ("unclosed_brackets", '{"tool": "web_search", "args": {"query": "mlx"', "unclosed_brackets",
     {"query": "mlx"}),
    ("name_noise", '{"tool": "functions.web_search()", "args": {"query": "mlx"}}', "name_noise",
     {"query": "mlx"}),
]

# Text that must NOT parse as a call (prose, data, nameless or unknown calls).
EVAL_PARSER_PROSE: list = [
    ("config_example", 'Here is a config example: {"name": "my-app", "version": 2}'),
    ("code_snippet", "Use [x(1)] in your code."),
    ("plain_list", "[1, 2, 3] is a list"),
    ("plain_answer", "The answer is 42."),
    ("unknown_implicit", '{"name": "launch_rockets", "arguments": {}}'),
    ("nameless", '<tool_call>{"arguments": {"query": "mlx"}}</tool_call>'),
    ("reasoning_only", '<think>{"tool": "calculator", "args": {}}</think>The answer is 2.'),
]

# Formats the streaming detector is checked on at several chunk sizes.
EVAL_PARSER_STREAMED = ("canonical", "hermes", "qwen3_coder", "mistral_args", "pythonic",
                        "llama3_python_tag")
EVAL_PARSER_CHUNKS = (1, 3, 7, 0)   # 0 = the whole text in one chunk


__all__ = [
    "EVAL_WORKSPACE_FILES",
    "EVAL_TOOL_CASES",
    "EVAL_RAG_DOCS",
    "EVAL_RAG_CASES",
    "EVAL_MULTI_TURN_CASES",
    "EVAL_ROUTING_PROJECT",
    "EVAL_ROUTING_ATTACHMENTS",
    "EVAL_ROUTING_HISTORY",
    "EVAL_ROUTING_WORKSPACE_FILES",
    "EVAL_ROUTING_KB_DOCS",
    "EVAL_ROUTING_CASES",
    "EVAL_ROUTING_GAPS",
    "EVAL_PARSER_KNOWN",
    "EVAL_PARSER_FORMATS",
    "EVAL_PARSER_REPAIRS",
    "EVAL_PARSER_PROSE",
    "EVAL_PARSER_STREAMED",
    "EVAL_PARSER_CHUNKS",
]
