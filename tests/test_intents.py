"""Tests for the UI-term intent lexicon (intents.py) and the detector fixes it
relies on in textutil.py.

The three conversations reviewed in upgrade/ROUTING_AND_TOOLING.md come first;
then every intent with phrasings that must fire and phrasings that must not,
switched-off tools, slash commands, and ordinary questions that must be left
to the router."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_llm.config as config_mod  # noqa: E402

config_mod.DATA_DIR = Path(tempfile.mkdtemp(prefix="llm-test-intents-"))

from local_llm.config import Config  # noqa: E402
from local_llm.intents import (  # noqa: E402
    DOCUMENT_RESEARCH_TOOLS, INTENT_LANES, INTENT_NAMES, INTENT_RETRIEVAL,
    INTENT_TOOL_SUBSETS, Intent, classify)
from local_llm.settings_schema import SETTINGS_FIELDS  # noqa: E402
from local_llm.textutil import (  # noqa: E402
    is_code_request, refers_to_project, routing_override)

# The tools registered under the configuration the three failed turns ran with
# (default agent, ALLOW_SHELL=0, ALLOW_PYTHON=0, EXTRA_TOOLS empty).
REG = frozenset(
    "web_search fetch_url weather calculator current_time list_files read_file file_info "
    "index_url search_docs index_docs write_file create_document edit_file search_files "
    "recall_feedback remember recall_memory forget load_skill search_skills save_skill "
    "search_past_conversations final_answer delegate_task".split())

CASE_1 = "add this to skills"
CASE_2 = ("use the knowledge base to craft a pdf showing an overview of usefull facts in "
          "the knowledge base")
CASE_3 = "check the codebase for design mistakes and summarize"


def _c(message: str, **kw) -> Intent | None:
    kw.setdefault("registered_tools", REG)
    kw.setdefault("project_basename", "mlx-h")
    return classify(message, **kw)


def _assert_contract(intent: Intent, registered: frozenset[str] = REG) -> None:
    assert intent.name in INTENT_NAMES
    assert intent.lane in INTENT_LANES
    assert intent.retrieval in INTENT_RETRIEVAL
    assert set(intent.research_sources) <= set(DOCUMENT_RESEARCH_TOOLS)
    if registered:
        assert set(intent.tool_subset) <= registered | {"final_answer"}
    assert set(intent.confirm_tools) <= set(intent.tool_subset)
    if intent.lane == "tools":
        assert intent.tool_subset and "final_answer" in intent.tool_subset
    if intent.lane == "reply":
        assert intent.reply.strip()
    if intent.lane == "clarify":
        assert intent.clarify_question.strip()
    for text in (intent.reply, intent.clarify_question):
        assert "—" not in text and "–" not in text


# --------------------------------------------------------------------------- #
# The three failed conversations
# --------------------------------------------------------------------------- #
def test_case_1_add_attachment_to_skills():
    intent = _c(CASE_1, has_attachment=True, attachment_names=("devils-advocate.md",))
    assert intent is not None
    _assert_contract(intent)
    assert intent.name == "skill_save"
    assert intent.lane == "tools"
    assert intent.referent == "attachment"
    assert intent.confirm_tools == ("save_skill",)
    assert intent.retrieval == "off"
    assert "save_skill" in intent.tool_subset
    assert intent.extras.get("attachment") == "devils-advocate.md"
    assert "attachment" in intent.targets


def test_case_1_without_attachment_or_history_asks_one_question():
    intent = _c(CASE_1)
    assert intent is not None
    _assert_contract(intent)
    assert intent.name == "skill_save"
    assert intent.lane == "clarify"
    question = intent.clarify_question
    assert question.endswith("?")
    assert sum(question.count(ch) for ch in ".?!") == 1, "one sentence"
    assert intent.retrieval == "off"


def test_case_1_with_history_saves_the_previous_answer():
    intent = _c(CASE_1, has_history=True)
    assert intent.name == "skill_save" and intent.lane == "tools"
    assert intent.referent == "previous_answer"
    assert intent.confirm_tools == ("save_skill",)


def test_case_2_kb_sourced_pdf():
    intent = _c(CASE_2)
    assert intent is not None
    _assert_contract(intent)
    assert intent.name == "document_create"
    assert intent.lane == "document"
    assert "knowledge_base" in intent.research_sources
    assert "web" not in intent.research_sources
    assert intent.retrieval == "kb"
    assert intent.extras["format"] == ".pdf"


def test_case_3_codebase_review():
    intent = _c(CASE_3)
    assert intent is not None
    _assert_contract(intent)
    assert intent.name == "codebase_review"
    assert intent.lane == "tools"
    assert intent.retrieval == "off"
    for tool in ("list_files", "search_files", "read_file"):
        assert tool in intent.tool_subset
    assert "repo_map" not in intent.tool_subset, "not registered, so dropped"
    # Unknown registry: the full subset, including tools of a later phase.
    assert _c(CASE_3, registered_tools=frozenset()).tool_subset == \
        INTENT_TOOL_SUBSETS["codebase_review"]


def test_the_project_folder_name_is_a_codebase_term():
    intent = _c("review mlx-h for design mistakes")
    assert intent.name == "codebase_review"
    assert _c("review mlx-h for design mistakes", project_basename="") is None
    # A generic folder name is no evidence.
    assert _c("review main for mistakes", project_basename="main") is None
    # "mlx" is not "mlx-h".
    assert _c("what's new in mlx this month") is None


def test_disabled_lexicon_returns_none():
    for message in (CASE_1, CASE_2, CASE_3, "/kb what is tdd"):
        assert classify(message, registered_tools=REG, enabled=False) is None


# --------------------------------------------------------------------------- #
# Every intent: phrasings that fire it and phrasings that must not
# --------------------------------------------------------------------------- #
POSITIVE: dict[str, list[tuple[str, dict]]] = {
    "conv_action": [
        ("export this conversation as markdown", {}), ("regenerate the last answer", {}),
        ("pin this conversation", {}), ("rename this chat", {}), ("delete this chat", {}),
        ("start a new chat", {}),
    ],
    "admin_action": [
        ("add a user called bob", {}), ("create an api token for my script", {}),
        ("always allow write_file", {}), ("switch to the coder model", {}),
        ("restart the model server", {}), ("retrain the model on my feedback", {}),
        ("enable remote access", {}), ("make a backup", {}),
        ("connect claude code to this app over mcp", {}), ("add a node to the cluster", {}),
        ("clear the knowledge base", {}), ("import my chatgpt history", {}),
        ("open the knowledge base settings", {}), ("turn on dark mode", {}),
        ("set the temperature to 0.2", {}), ("run the evals", {}),
    ],
    "self_describe": [
        ("what tools do you have", {}), ("what can you do", {}),
        ("what are your capabilities", {}), ("which tools are available", {}),
        ("what are you capable of", {}),
    ],
    "disabled_capability": [
        ("run the tests", {}), ("run this python: print(1)", {}), ("show the git diff", {}),
        ("commit my changes", {}), ("check my email", {}), ("what is on my calendar today", {}),
        ("generate an image of a cat", {}),
        ("query the database for the number of conversations", {}),
        ("send an email to bob about the release", {}), ("my contacts", {}),
    ],
    "skill_save": [
        (CASE_1, {"has_attachment": True}), ("save this as a skill", {"has_history": True}),
        ("turn that into a skill", {"has_history": True}),
        ("make a skill out of this", {"has_attachment": True}),
        ("save devils-advocate.md as a skill", {}), ("make this a skill", {"has_history": True}),
    ],
    "skill_use": [
        ("use the devils-advocate skill to review my plan", {}),
        ("apply the code-review skill", {}), ("follow the release-checklist skill", {}),
        ("load the pdf-tools skill", {}),
    ],
    "skill_list": [
        ("list my skills", {}), ("show me the saved skills", {}),
        ("what skills do you have", {}), ("my skill library", {}),
    ],
    "memory_write": [
        ("remember that my timezone is Europe/Brussels", {}),
        ("don't forget that I prefer metric units", {}),
        ("keep in mind that we deploy on Fridays", {}), ("remember my name is Zeno", {}),
        ("note that I prefer metric units", {}), ("don't forget my standup is at 9:30", {}),
    ],
    "memory_forget": [
        ("forget my timezone", {}), ("forget what I told you about my address", {}),
        ("delete the note about my car", {}), ("remove my timezone from your memory", {}),
        ("forget that I live in Ghent", {}),
    ],
    "memory_read": [
        ("what do you remember about me", {}), ("summarize my notes", {}),
        ("show me my notes", {}), ("what are my saved preferences", {}),
    ],
    "history": [
        ("search past chats for the docker discussion", {}),
        ("what did we discuss yesterday about mlx", {}),
        ("find the conversation about LoRA training", {}), ("search history for docker", {}),
        ("in a previous conversation you gave me a regex", {}),
        ("find the docker command in my chat history", {}),
    ],
    "kb_write": [
        ("index the docs folder", {}), ("add https://example.com/docs to the knowledge base", {}),
        ("ingest ~/notes into the kb", {}), ("index https://example.com/guide", {}),
        ("add this to the knowledge base", {"has_attachment": True}),
    ],
    "kb_overview": [
        ("what is in my knowledge base", {}), ("what is in the kb", {}),
        ("list the documents in my knowledge base", {}),
        ("give me an overview of the knowledge base", {}),
        ("how many documents are in my knowledge base", {}),
    ],
    "kb_query": [
        ("search the knowledge base for onboarding", {}),
        ("according to my docs, what is book-to-skill", {}),
        ("use the knowledge base to answer what is tdd", {}),
        ("search my documents for pricing", {}),
        ("what does the knowledge base say about failover", {}),
        ("summarize the knowledge base article on onboarding", {}),
        ("what is in the knowledge base about vacation policy", {}),
        ("tell me about onboarding according to my docs", {}),
        ("what does my library say about vacation policy", {}),
    ],
    "codebase_lookup": [
        ("where is routing_override defined", {}), ("find usages of classify_question", {}),
        ("who calls build_agent_system_prompt", {}),
        ("where in the codebase do we handle auth", {}), ("grep for TODO", {}),
    ],
    "codebase_write": [
        ("check the codebase and write a readme.md", {}), ("write a readme for this project", {}),
        ("update the README.md for the codebase", {}), ("generate docs for mlx-h", {}),
        ("add a CONTRIBUTING.md to the repo", {}),
        ("draft an architecture doc for the codebase", {}),
        ("improve the readme of the codebase", {}),
        ("add an installation section to the README of this repo", {}),
    ],
    "codebase_review": [
        (CASE_3, {}), ("check the codebase", {}), ("review this project for design mistakes", {}),
        ("give me an overview of this repo", {}), ("explain the repo structure", {}),
        ("summarize the code", {}), ("what does this project do", {}),
        ("audit our codebase for security issues", {}),
        ("review the source for code smells", {}),
    ],
    "schedule": [
        ("create a task that runs every morning to search news about MLX", {}),
        ("remind me to check the build tomorrow", {}),
        ("every morning, summarize the AI news", {}),
        ("schedule a daily report of new documents", {}),
        ("search for mlx releases every week", {}),
    ],
    "document_create": [
        (CASE_2, {}), ("create an excel sheet of the documents in the knowledge base", {}),
        ("make a slide deck about the knowledge base", {}),
        ("write a word document summarising the codebase", {}),
        ("make a pdf about texcel.be", {}), ("create a pdf report from notes.md", {}),
    ],
}

NEGATIVE: dict[str, list[str]] = {
    "conv_action": ["export to pdf is broken in my app", "how do I delete a git branch",
                    "delete the last answer",
                    "regenerate the api docs from the openapi spec",
                    "the chat feature in my app needs a rename button"],
    "admin_action": ["how do I add a user in linux", "create a user model class in python",
                     "create a new user in postgres", "generate an api key for openai",
                     "add a user story for the login page", "make a backup script in bash",
                     "train the model on mnist", "load the model in pytorch",
                     "always allow users to edit their profile in my app",
                     "re-running bootstrap reset the admin password",
                     "show me the wifi settings", "restart nginx", "switch to dark chocolate"],
    "self_describe": ["what can you do about the error in my config",
                      "which tools do I need to fix a bike",
                      "what are the best tools for woodworking", "what tools does a plumber use"],
    "disabled_capability": ["how do I run the tests in django", "explain what git status does",
                            "write an email to my boss asking for a day off",
                            "write a sql query that counts users",
                            "write a commit message for these changes",
                            "generate an svg image of a circle", "make a logo for my startup",
                            "read the email below and reply politely",
                            "build my calendar app in react",
                            "what's the best way to run tests in parallel"],
    "skill_save": ["add teamwork to my skills section of my cv", "make it a skill-based game",
                   "add python to my skills on my resume", "create a skill tree for my rpg",
                   "import the skills from the knowledge base into my skills"],
    "skill_use": ["use the right skill for this", "what are your skills",
                  "use your communication skills", "how do I use a skill saw"],
    "skill_list": ["what are your skills", "list the skills a data engineer needs",
                   "show my skills on the resume page", "skills needed for a nurse"],
    "memory_write": ["remember to be concise",
                     "keep in mind that I'm on Windows. How do I install brew?",
                     "Note that this is a draft. Can you improve it?",
                     "do you remember how recursion works"],
    "memory_forget": ["forget that, what time is it", "forget it",
                      "delete the notes column from my dataframe",
                      "I always forget my keys, any tips?"],
    "memory_read": ["what notes are in a c major chord", "my notes app crashes on startup",
                    "take notes on this lecture", "what do you know about rust"],
    "history": ["when was the last time the cubs won", "search the history of rome",
                "the last conversation I had with my boss was weird",
                "as we discussed, write the function", "tell me about the history of the internet",
                "how do I export my chat history", "is my chat history private?",
                "my conversation history disappeared"],
    "kb_write": ["what is in the kb", "is the onboarding guide in the knowledge base?",
                 "what is the index of the array", "how do I index a pandas dataframe"],
    "kb_overview": ["what is a knowledge base", "how does the knowledge base work",
                    "how does the knowledge base work in notion", "build a knowledge base app"],
    "kb_query": ["what is a knowledge base", "my documents folder is missing on windows",
                 "my library fails to import", "the library closes at 5",
                 "the knowledge base is slow", "clearing the knowledge base did not empty it",
                 "how do I add files to the knowledge base", "why is the knowledge base empty",
                 "how does the knowledge base work in notion",
                 "according to the docs, map() returns an iterator"],
    "codebase_lookup": ["where is the eiffel tower", "where is consciousness defined in philosophy",
                        "find usages of the word love in the bible",
                        "grep is a unix command, what does it do"],
    "codebase_write": ["write a readme",
                       "write a python script that generates the docs for this repo",
                       "how do I write a good readme for my repo",
                       "review the codebase but don't update the readme",
                       "write a summary of the readme for the repo",
                       "create a pdf about the codebase", "update the readme on github for my repo",
                       "write a blog post about the codebase", "add docstrings to the codebase",
                       "write a readme for the knowledge base", "write a readme for the project plan"],
    "codebase_review": ["check the source for this claim", "review my project plan",
                        "check the source of the river for pollution",
                        "explain the code below", "summarize the code you wrote",
                        "design the app's login page", "fix the bug in agent.py",
                        "review the codebase and fix the bugs", "check the app store reviews",
                        "use my code-review prompt", "is this app free?"],
    "schedule": ["what is the daily recommended intake of vitamin c",
                 "Give me a daily workout routine", "send me my weekly report",
                 "I run every morning, any stretching tips?",
                 "remind me how list comprehensions work", "add a task to my todo list",
                 "what do people eat every day in japan",
                 "write a cron job that runs every hour",
                 "every morning I drink coffee, is that bad?"],
    "document_create": ["summarize this pdf", "export to pdf is broken in my app",
                        "can you create a pdf parser in python", "where is the pdf file",
                        "read the excel file I uploaded",
                        "the generated PDF is not a well-formed PDF file",
                        "how do I save a pdf in word"],
}


@pytest.mark.parametrize("name,message,kw", [
    (name, message, kw) for name, cases in POSITIVE.items() for message, kw in cases])
def test_intent_fires(name, message, kw):
    assert len(POSITIVE[name]) >= 4
    intent = _c(message, **kw)
    assert intent is not None, f"{message!r} matched nothing"
    _assert_contract(intent)
    assert intent.name == name, f"{message!r} -> {intent.name}"
    assert intent.source == "lexicon"


@pytest.mark.parametrize("name,message", [
    (name, message) for name, cases in NEGATIVE.items() for message in cases])
def test_intent_does_not_fire(name, message):
    assert len(NEGATIVE[name]) >= 4
    intent = _c(message)
    assert intent is None or intent.name != name, f"{message!r} -> {intent.name}"


def test_every_intent_has_examples_both_ways():
    assert set(POSITIVE) == set(INTENT_NAMES)
    assert set(NEGATIVE) == set(INTENT_NAMES)


ORDINARY = [
    "what is the capital of france", "explain recursion",
    "write a python script that parses csv", "search the web for mlx news",
    "search the web for mlx 0.32 release notes", "what is the weather in Brussels tomorrow",
    "hi", "thanks!", "how do I reverse a list in python", "translate this sentence to french",
    "fix the bug in agent.py", "what changed in the repo", "check the logs for errors",
    "notify me when this is done", "use agent mode for this", "summarize this pdf",
    "save this prompt to the prompt library", "use my code-review prompt",
    "write a haiku about autumn", "https://example.com", "17*23",
    "/usr/bin/python is missing, what now?",
]


@pytest.mark.parametrize("message", ORDINARY)
def test_ordinary_messages_are_left_to_the_router(message):
    assert _c(message) is None


# --------------------------------------------------------------------------- #
# Referents and pasted material
# --------------------------------------------------------------------------- #
def test_pasted_body_is_an_inline_referent():
    body = "---\nname: devils-advocate\ndescription: argue the other side\n---\n" + "x " * 80
    intent = _c(f"add this to skills:\n\n{body}")
    assert intent.name == "skill_save" and intent.referent == "inline"
    intent = _c(f"{body}\n\nsave this as a skill")
    assert intent.name == "skill_save" and intent.referent == "inline"


def test_code_in_the_message_is_never_read_as_an_instruction():
    assert _c("review the code\n```python\nprint('check the codebase')\n```") is None
    assert _c("review this code: ```def f(): pass```") is None


def test_weak_project_terms_yield_to_an_attachment_or_an_earlier_answer():
    assert _c("summarize the code", has_history=True) is None
    assert _c("review the app for mistakes", has_attachment=True) is None
    # A strong term still means the project folder.
    assert _c("check the codebase", has_attachment=True).name == "codebase_review"


def test_memory_and_kb_writes_without_a_referent_ask():
    intent = _c("remember this")
    assert intent.name == "memory_write" and intent.lane == "clarify"
    intent = _c("add this to the knowledge base")
    assert intent.name == "kb_write" and intent.lane == "clarify"
    assert _c("remember this", has_history=True).referent == "previous_answer"


# --------------------------------------------------------------------------- #
# Switched-off tools become honest replies
# --------------------------------------------------------------------------- #
def test_tools_lane_with_nothing_registered_becomes_a_reply():
    bare = frozenset({"final_answer", "web_search", "fetch_url"})
    cases = {
        "search the knowledge base for onboarding": ("kb_query", "search_docs"),
        CASE_3: ("codebase_review", "read_file"),
        "list my skills": ("skill_list", "search_skills"),
        "remember that my timezone is Europe/Brussels": ("memory_write", "remember"),
        "search past chats for the docker discussion": ("history", "search_past_conversations"),
    }
    for message, (requested, tool) in cases.items():
        intent = _c(message, registered_tools=bare)
        assert intent is not None, message
        _assert_contract(intent, bare)
        assert intent.name == "disabled_capability" and intent.lane == "reply", message
        assert intent.extras["requested"] == requested
        assert tool in intent.reply
        assert "Settings > Tools > Catalogue" in intent.reply
        # A later-phase tool is not named as something to switch on. kb_overview
        # shipped in phase B, so it may be named next to search_docs.
        assert "repo_map" not in intent.reply and "list_skills" not in intent.reply


def test_skills_switched_off_names_the_switch():
    no_skills = REG - {"load_skill", "search_skills", "save_skill"}
    intent = _c(CASE_1, has_attachment=True, registered_tools=no_skills)
    assert intent.name == "disabled_capability" and intent.lane == "reply"
    assert "SKILLS_ENABLED" in intent.reply


def test_disabled_capability_names_the_switch_and_never_claims_to_have_run():
    expected = {
        "run the tests": "ALLOW_SHELL", "run this python: print(1)": "ALLOW_PYTHON",
        "show the git diff": "EXTRA_TOOLS", "check my email": "O365",
        "generate an image of a cat": "EXTRA_TOOLS",
    }
    for message, switch in expected.items():
        intent = _c(message)
        assert intent.name == "disabled_capability" and intent.lane == "reply", message
        assert switch in intent.reply
        assert "Settings > Tools > Catalogue" in intent.reply
        assert "I have not" in intent.reply
    assert "@diff" in _c("show the git diff").reply


def test_disabled_capability_needs_a_known_registry_and_a_missing_tool():
    assert _c("run the tests", registered_tools=frozenset()) is None
    assert _c("run the tests", registered_tools=REG | {"run_tests", "run_shell"}) is None
    assert _c("check my email", registered_tools=REG | {"o365_mail"}) is None


def test_spreadsheet_reading_is_off_only_without_read_file():
    # read_file reads .xlsx, so with it registered the request is left alone.
    assert _c("read this spreadsheet") is None
    no_read = REG - {"read_file"}
    intent = _c("read this spreadsheet", registered_tools=no_read)
    assert intent.name == "disabled_capability" and "spreadsheet in EXTRA_TOOLS" in intent.reply
    # What was uploaded is read by the attachment pipeline.
    assert _c("read this spreadsheet", registered_tools=no_read, has_attachment=True) is None
    assert _c("read the excel file I uploaded", registered_tools=no_read) is None


def test_schedule_with_and_without_the_scheduler():
    message = "create a task that runs every morning to search news about MLX"
    intent = _c(message)
    assert intent.name == "schedule" and intent.lane == "reply"
    assert "Tasks view" in intent.reply and "I have not scheduled" in intent.reply
    assert not intent.tool_subset and not intent.confirm_tools
    intent = _c(message, registered_tools=REG | {"task_scheduler"})
    assert intent.name == "schedule" and intent.lane == "tools"
    assert intent.confirm_tools == ("task_scheduler",)
    assert intent.tool_subset == ("task_scheduler", "final_answer")


def test_self_describe_is_built_from_the_registry():
    intent = _c("what can you do")
    assert intent.name == "self_describe" and intent.lane == "reply"
    for tool in ("read_file", "web_search", "search_docs", "save_skill"):
        assert f"`{tool}`" in intent.reply
    assert "run_shell" not in intent.reply.split("Switched off")[0]
    assert "Run code & shell" in intent.reply and "ALLOW_SHELL" in intent.reply
    assert "Settings > Tools > Catalogue" in intent.reply
    # Without the registry there is nothing true to say.
    assert _c("what can you do", registered_tools=frozenset()) is None


def test_admin_and_conversation_actions_only_explain():
    for message, _ in POSITIVE["admin_action"] + POSITIVE["conv_action"]:
        intent = _c(message)
        assert intent.lane == "reply" and not intent.tool_subset, message
    assert "Settings > Admin > Users" in _c("add a user called bob").reply
    assert "Settings > Tools > API tokens" in _c("create an api token for my script").reply
    assert "Settings > Tools > Approvals" in _c("always allow write_file").reply
    assert "Settings > Models & Training > Model" in _c("switch to the coder model").reply
    assert "History" in _c("pin this conversation").reply
    assert "Export as Markdown" in _c("export this conversation as markdown").reply


# --------------------------------------------------------------------------- #
# Document sources
# --------------------------------------------------------------------------- #
def test_document_research_sources_come_from_the_other_targets():
    def sources(message: str) -> tuple[str, ...]:
        intent = _c(message)
        assert intent is not None and intent.name == "document_create", message
        return intent.research_sources

    assert sources("write a word document summarising the codebase") == ("codebase",)
    assert sources("make a pdf of what we discussed last time") == ("history",)
    assert sources("make a pdf about texcel.be") == ("web",)
    assert sources("create a pdf report from notes.md") == ("files",)
    # Today's semantics: a lookup ask with no other source is a web lookup...
    assert sources("research the market and write a pdf report") == ("web",)
    # ...but "research my knowledge base" is not.
    assert sources("research my knowledge base and write a pdf report") == ("knowledge_base",)
    # Retrieval follows the sources, as the matching intent's would (P1, P2):
    # the knowledge base is injected for a KB document, never for one built
    # from the codebase, past chats, a named file or an attachment.
    def retrieval(message: str, **kw) -> str:
        return _c(message, **kw).retrieval

    assert retrieval(CASE_2) == "kb"
    assert retrieval("write a word document summarising the codebase") == "off"
    assert retrieval("make a word document summarizing our past chats about MLX") == "off"
    assert retrieval("create a pdf report from notes.md") == "off"
    assert retrieval("turn this into a pdf", has_attachment=True) == "off"
    assert retrieval("make a pdf about texcel.be") == "auto"
    assert retrieval("create a pdf for construction invoice") == "auto"


def test_document_referent():
    assert _c("turn this into a pdf", has_attachment=True).referent == "attachment"
    assert _c("turn this into a pdf", has_history=True).referent == "previous_answer"


# --------------------------------------------------------------------------- #
# Slash commands
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("message,name", [
    ("/kb what does the runbook say about failover", "kb_query"),
    ("/docs pricing", "kb_query"),
    ("/kb", "kb_overview"),
    ("/code", "codebase_review"),
    ("/code where is routing_override defined", "codebase_lookup"),
    ("/code where is classify used", "codebase_lookup"),
    ("/skills", "skill_list"),
    ("/history docker", "history"),
    ("/mem", "memory_read"),
    ("/mem remember that I like tea", "memory_write"),
    ("/pdf overview of my knowledge base", "document_create"),
])
def test_slash_commands_map_to_intents(message, name):
    intent = _c(message)
    assert intent is not None and intent.name == name
    assert intent.source == "slash"
    _assert_contract(intent)
    assert intent.extras["cleaned_message"] == message.split(None, 1)[1] if " " in message \
        else intent.extras["cleaned_message"] == ""


def test_pdf_slash_is_a_pdf_with_sources():
    intent = _c("/pdf overview of my knowledge base")
    assert intent.extras["format"] == ".pdf"
    assert intent.research_sources == ("knowledge_base",) and intent.retrieval == "kb"
    assert _c("/pdf").lane == "clarify"
    assert _c("/pdf", has_history=True).referent == "previous_answer"


@pytest.mark.parametrize("message", [
    "/search who won the 2025 tour de france", "/web mlx news", "/websearch x",
    "/nosearch explain tail latency", "/no-search x", "/answer x", "/local x",
    "/agent find the bug in main.py", "/tools what changed since friday", "/act do it",
])
def test_routing_override_commands_are_left_alone(message):
    assert _c(message) is None
    assert routing_override(message) is not None


def test_routing_override_is_unchanged():
    assert routing_override("/kb what does our runbook say") == ("kb", "what does our runbook say")
    assert routing_override("/code review it") is None
    assert routing_override("/pdf x") is None


# --------------------------------------------------------------------------- #
# Detector false positives found by the probe (textutil)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("message", [
    "summarize the code", "use my code-review prompt", "create an api token for my script",
    "connect claude code to this app over mcp",
])
def test_is_code_request_false_positives(message):
    assert not is_code_request(message)


@pytest.mark.parametrize("message", [
    "fix this code", "write a python script attempting 5 os recon techniques",
    "python script to pull the newest CVEs", "implement a binary search in rust",
    "write a swift function that uses drive api without api key",
    "generate an rsa key in python", "write a function to generate an api key",
    "create a password reset flow for my app", "add password validation to my script",
    "code a function that sorts a list", "write 10 lines of code",
])
def test_is_code_request_still_recognises_code(message):
    assert is_code_request(message)


def test_refers_to_project_ignores_web_addresses():
    assert not refers_to_project("add https://example.com/docs to the knowledge base")
    assert not refers_to_project("see example.com/docs/api for details")
    assert not refers_to_project("check https://github.com/a/b/blob/main/x.py")
    assert refers_to_project("fix the bug in local_llm/agent.py")
    assert refers_to_project("read ~/projects/app/main")
    assert refers_to_project("open ./src/components/Button")
    assert not refers_to_project("write a basic c++ crud program")
    # "my code-review prompt" is a compound noun, not the user's code.
    assert not refers_to_project("use my code-review prompt")
    assert refers_to_project("fix my code")
    assert refers_to_project("in my code-base")


# --------------------------------------------------------------------------- #
# Contract, config and settings
# --------------------------------------------------------------------------- #
def test_as_log_carries_no_message_text():
    intent = _c(CASE_3)
    logged = intent.as_log()
    assert logged["intent"] == "codebase_review" and logged["intent_source"] == "lexicon"
    assert "matched" not in logged and "reply" not in logged
    flat = " ".join(str(v) for v in logged.values())
    for word in ("check", "design", "mistakes", "summarize"):
        assert word not in flat


def test_every_tool_subset_ends_with_final_answer():
    for name, subset in INTENT_TOOL_SUBSETS.items():
        assert name in INTENT_NAMES
        assert subset[-1] == "final_answer"


def test_config_switch_defaults_on_and_is_mutable(monkeypatch):
    assert Config().intent_lexicon_enabled is True
    assert "intent_lexicon_enabled" in Config.MUTABLE
    monkeypatch.setenv("INTENT_LEXICON_ENABLED", "0")
    assert Config().intent_lexicon_enabled is False
    field = next(f for f in SETTINGS_FIELDS if f["key"] == "intent_lexicon_enabled")
    assert field["section"] == "agent" and field["group"] == "Routing"
    assert field["widget"] == "switch"
    assert os.environ.get("INTENT_LEXICON_ENABLED") == "0"


# --------------------------------------------------------------------------- #
# codebase_write: "check the codebase and write a readme.md" matched nothing,
# ran on the full catalogue, overwrote a 100 KB README.md with 841 characters
# and fetched a GitHub URL the model made up. These pin the intent that turn
# now gets: file tools only, both writes confirmed, and the guidance that keeps
# an existing file unless the user asked to replace it.
# --------------------------------------------------------------------------- #
CASE_README = "check the codebase and write a readme.md"
_WEB_TOOLS = {"web_search", "fetch_url", "index_url"}


def test_codebase_write_the_readme_turn():
    intent = _c(CASE_README)
    assert intent is not None
    _assert_contract(intent)
    assert intent.name == "codebase_write" and intent.lane == "tools"
    assert intent.tool_subset == ("list_files", "search_files", "read_file", "file_info",
                                  "edit_file", "write_file", "final_answer")
    assert not _WEB_TOOLS & set(intent.tool_subset)
    assert intent.retrieval == "off" and intent.targets == ("codebase",)
    assert intent.confirm_tools == ("write_file", "edit_file")
    assert intent.extras["target"] == "README.md"
    assert intent.extras["overwrite"] is False
    guidance = intent.extras["guidance"]
    assert "read_file README.md if it exists" in guidance
    assert "do not replace it" in guidance and "outputs/README.md" in guidance
    assert "list the files you read" in guidance
    assert "do not search or fetch" in guidance
    assert "\u2014" not in guidance and "\u2013" not in guidance


@pytest.mark.parametrize("message,name", [
    (CASE_README, "codebase_write"),
    ("Check the codebase and write a README.md.", "codebase_write"),
    ("check the codebase and summarize", "codebase_review"),
    (CASE_3, "codebase_review"),
    ("write a python script that parses csv", None),
    ("write a python script that generates the docs for this repo", None),
    ("create a pdf about the codebase", "document_create"),
    ("write a readme for the repo and export it as pdf", "document_create"),
    ("add the readme to the knowledge base", "kb_write"),
    ("review the codebase and fix the bugs", None),
])
def test_codebase_write_ordering_against_the_neighbouring_rules(message, name):
    intent = _c(message)
    assert (intent.name if intent else None) == name, message


def test_a_codebase_pdf_researches_the_codebase():
    intent = _c("create a pdf about the codebase")
    assert intent.lane == "document" and intent.research_sources == ("codebase",)
    assert intent.extras["format"] == ".pdf"


@pytest.mark.parametrize("message,target,kind", [
    ("write a readme for this project", "README.md", "readme"),
    ("update readme.rst in the repo", "README.rst", "readme"),
    ("write a CHANGELOG for the repo", "CHANGELOG.md", "changelog"),
    ("add a contributing.md to the codebase", "CONTRIBUTING.md", "contributing"),
    ("draft an architecture doc for the codebase", "ARCHITECTURE.md", "architecture"),
    ("write a design document for mlx-h", "DESIGN.md", "architecture"),
    ("create docs/setup.md for the project", "docs/setup.md", "file"),
    ("check the codebase and write the findings to review.md", "review.md", "file"),
    ("generate docs for mlx-h", "", "docs"),
    ("write documentation for the codebase", "", "docs"),
    ("create a user guide for this project", "", "docs"),
])
def test_codebase_write_names_its_target(message, target, kind):
    intent = _c(message)
    assert intent is not None and intent.name == "codebase_write", message
    assert intent.extras["target"] == target and intent.extras["target_kind"] == kind


def test_codebase_write_keeps_the_file_unless_told_to_replace_it():
    kept = _c("update the readme of the codebase")
    assert kept.extras["overwrite"] is False
    assert "do not replace it" in kept.extras["guidance"]
    for message in ("rewrite the readme for mlx-h from scratch",
                    "write a readme for my project, replacing the old one",
                    "overwrite the README.md of this repo with a new one"):
        intent = _c(message)
        assert intent.name == "codebase_write" and intent.extras["overwrite"] is True, message
        assert "The user asked to replace README.md" in intent.extras["guidance"]
        assert "do not replace it" not in intent.extras["guidance"]
    # "rewrite" alone is a new version, not permission to drop the old one,
    # and replacing a section of the file is an edit of it.
    assert _c("rewrite the readme of the codebase").extras["overwrite"] is False
    section = _c("replace the install section of the readme of this repo")
    assert section.name == "codebase_write" and section.extras["overwrite"] is False


def test_codebase_write_guidance_follows_outputs_dir():
    from local_llm.intents import codebase_write_guidance
    assert "write the complete new version to out/README.md" in codebase_write_guidance(
        "README.md", outputs_dir="out/")
    # OUTPUTS_DIR "." is the project root: the new version goes beside it.
    assert "README.new.md" in codebase_write_guidance("README.md", outputs_dir="")
    general = codebase_write_guidance("")
    assert "outputs/<file name>" in general and "Do not replace an existing file" in general
    assert "replace the existing documentation" in codebase_write_guidance("", overwrite=True)


def test_codebase_write_needs_a_project_term_and_a_governed_target():
    for message in ("write a readme", "update the readme", "write a readme for the project plan",
                    "review the codebase but don't update the readme",
                    "write a summary of the readme for the repo",
                    "write a blog post about the codebase",
                    "explain how to write a readme for my project",
                    "can you write a readme for this repo?\n```\nprint(1)\n```"):
        intent = _c(message)
        assert intent is None or intent.name != "codebase_write", message
    # "explain the codebase and ..." is an instruction, not a question.
    assert _c("explain the codebase and write a readme").name == "codebase_write"


def test_codebase_write_weak_term_yields_to_an_earlier_answer():
    assert _c("write a readme for my code").name == "codebase_write"
    found = _c("write a readme for my code", has_history=True)
    assert found is None or found.name != "codebase_write"
    found = _c("write a readme for this project", has_attachment=True,
               attachment_names=("notes.txt",))
    assert found is None or found.name != "codebase_write"
    assert _c("write a readme for the codebase", has_attachment=True).name == "codebase_write"


def test_codebase_write_without_write_tools_says_what_is_off():
    reg = REG - {"write_file", "edit_file"}
    intent = _c(CASE_README, registered_tools=reg)
    _assert_contract(intent, reg)
    assert intent.name == "disabled_capability" and intent.lane == "reply"
    assert intent.extras["requested"] == "codebase_write"
    assert "`write_file` and `edit_file` are not enabled" in intent.reply
    # One write tool is enough to do the job; the subset keeps what exists.
    only_edit = REG - {"write_file"}
    intent = _c(CASE_README, registered_tools=only_edit)
    assert intent.name == "codebase_write" and "write_file" not in intent.tool_subset
    assert intent.confirm_tools == ("edit_file",)


def test_code_slash_writes_when_asked_to_write():
    intent = _c("/code write a readme")
    assert intent.name == "codebase_write" and intent.source == "slash"
    assert intent.extras["target"] == "README.md"
    assert intent.extras["cleaned_message"] == "write a readme"
    assert _c("/code where is classify used").name == "codebase_lookup"
    assert _c("/code review the docs").name == "codebase_review"


def test_codebase_write_guidance_reaches_the_prompt_and_the_web_does_not(monkeypatch, tmp_path):
    """The real classify and Agent: narrowed catalogue, guidance in the system prompt."""
    import asyncio
    import json

    import local_llm.agent as agent_mod
    from local_llm.agent import Agent
    from local_llm.database import Database
    from local_llm.llm import GenerationStats
    from local_llm.tools import ToolRegistry

    class _Client:
        def __init__(self, replies):
            self.replies, self.prompts = list(replies), []

        def _next(self):
            return self.replies.pop(0) if self.replies else "done."

        def stream(self, messages, *a, **kw):
            self.prompts.append(messages)
            reply = self._next()

            async def gen():
                yield reply
            return gen()

        async def complete_with_stats(self, messages, *a, **kw):
            self.prompts.append(messages)
            return self._next(), GenerationStats()

        async def complete(self, messages, *a, **kw):
            return (await self.complete_with_stats(messages))[0]

        def reply_budget(self, messages, max_tokens=None, quiet=False):
            return max_tokens or 512

        async def wait_until_ready(self, timeout=0.0):
            return True

        @staticmethod
        def classify_error(exc):
            return "the model stalled"

    async def online(*a, **k):
        return True

    monkeypatch.setattr(agent_mod, "has_internet", online)
    project = tmp_path / "proj"
    project.mkdir()
    (project / "README.md").write_text("# proj\n\nA long README.\n", encoding="utf-8")
    cfg = Config(agent_enabled=True, knowledge_triage=False, fast_path=False,
                 incremental_reasoning=False, chunk_large_prompts=False, agent_max_steps=3,
                 agent_min_steps=1, resilient_retries=1, ready_wait_timeout=0.01,
                 auto_iterate_rounds=0, show_internals=False, rag_enabled=False,
                 project_dir=str(project), outputs_dir="drafts",
                 skills_dir=str(tmp_path / "skills"))
    db = Database(tmp_path / "agent.db")
    reg = ToolRegistry(cfg, db)
    calls = []
    reg.call = lambda name, args, cid=None: (calls.append(name) or "tool output", None)
    client = _Client([json.dumps({"tool": "list_files", "args": {}}),
                      "I read README.md and changed nothing."])
    agent = Agent(cfg, reg, client)

    async def main():
        return [ev async for ev in agent.run(CASE_README, [], "t", 256, None)]

    try:
        events = asyncio.run(main())
    finally:
        db.close()
    notes = [e.get("message") for e in events if e.get("type") == "notice"]
    assert "Understood as: codebase_write (lexicon)" in notes
    assert not any(str(p[0].get("content", "")).startswith("You are a router")
                   for p in client.prompts), "the router was asked anyway"
    system = client.prompts[0][0]["content"]
    for name in ("list_files", "read_file", "file_info", "edit_file", "write_file"):
        assert f"- {name}:" in system, name
    for name in ("web_search", "fetch_url", "create_document", "remember"):
        assert f"- {name}:" not in system, name
    # The guidance rides on the user turn (build_base's extra_context), with
    # the configured OUTPUTS_DIR as the place for a new version.
    asked = client.prompts[0][-1]["content"]
    assert asked.endswith(CASE_README)
    assert "drafts/README.md" in asked and "do not replace it" in asked
    assert "outputs/README.md" not in asked
    assert calls == ["list_files"]
