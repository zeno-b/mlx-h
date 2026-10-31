# Routing, retrieval and tooling: review of the last three conversations and proposed changes

Status: phase B implemented (P1 lexicon with 18 intents, P2 retrieval fixes, P3 `kb_overview` only,
P4, P5, P6, P8, P9; routing eval suite 97/97 in scripted mode). Still open: phase A settings
(section 2), phase C tools (`repo_map`, `list_skills`, `skills_import`, `capabilities_overview`,
`prompt_library`) and intent-labelled router menu, phase D assessor (P10), phase E composer and
intent chip (P7).
Evidence: `data/feedback.db` (messages, tool_calls, metrics, attachments), `logs/app.log.1`
(LOG_LEVEL=TRACE, so `turn.start` and `router.decision` are present), and a replay of all three
turns through the real `Agent` with a recording fake model and a copy of the database.

## 1. Verdict

None of the three conversations made sense, and in every case the failure happened before the
model wrote a word: the turn was sent down the wrong lane with the wrong tools and the wrong
context. The model (Llama-3.2-3B-Instruct-uncensored-6bit, 4096-token context) made it worse,
but a better model in the same pipeline would have failed the same way on two of the three.

| # | Message | What the user got | Root causes (ranked) |
|---|---|---|---|
| 1 | "add this to skills" + attached `devils-advocate.md` | The attachment re-typed as prose. No skill saved. 2 steps, 2067 tokens, ~140 s. | (a) The router saw only the 8-token text, not the attachment, and answered `action=answer`, which rebuilds the prompt with zero tools. (b) "skills" is a UI term with no route: `save_skill` exists but is not on the router menu. (c) RAG matched "skills" against 3467 of 3473 KB paths and injected another copy of the same file. |
| 2 | "use the knowledge base to craft a pdf showing an overview of usefull facts in the knowledge base" | One PDF written 5 times to the repo root from a single passage; reply "The PDF document has been created". 6 steps, ~110 s. | (a) The document lane narrows tools to `create_document` and only adds lookups when `document_needs_research` sees a web signal, so `search_docs` was never available. (b) No tool can give an overview of a KB; FTS returns 4 passages for any question. (c) Nothing stops repeated writes to the same path; one write was cut off at the reply limit and retried. (d) The model mistook the one passage (`book-to-skill.md`) for a "converter tool". |
| 3 | "check the codebase for design mistakes and summarize" | "The skills listed in the product_goal_router.py file are not verifiable…". No tool call. 3 steps, step 2 spent 2048 tokens of reasoning (126 s). | (a) The router answered `action=answer` although `list_files`/`read_file`/`search_files` are on its menu and `project_dir` is this repo. (b) "codebase" is a UI term (Settings > Admin > Codebase) that nothing maps to the file tools; `refers_to_project` misses "check the codebase" and only gates RAG anyway. (c) RAG injected three files from the unrelated imported skills repo. (d) `REASONING_INSTRUCTION` asks a non-thinking 3B model to think before answering; it reasoned until the reply cap. |

Two cross-cutting causes affect all three turns:

- **The context budget does not fit the tool lane.** `data/settings.json` sets `max_tokens: 2048` on
  a 4096 context, which leaves 1792 tokens for the prompt. The logs show the tool-lane prompt floor
  at 2350 to 3850 tokens: `prompt floor is 3850 tokens against a 1792-token budget … history is
  fully evicted`. Every agent turn starts already over budget.
- **The knowledge base is one imported repository.** All 3473 documents come from
  `claude-files/local/claude-skills-main`. Auto-RAG injects it into any substantive question, so it
  is noise for everything that is not about that repository.

## 2. Quick wins that need no code (do these first)

1. **Context budget.** Either set `CONTEXT_SIZE=8192` (KV cost for Llama-3.2-3B at fp16 is about
   112 KB per token, so roughly 0.9 GB for 8192 tokens: fits on 8 GB when nothing else is loaded) or
   lower Settings > Generation > Max tokens to 768 to 1024. Today the tool lane cannot fit at all.
2. **Model.** `Qwen2.5-Coder-3B-Instruct-4bit` is already cached and follows the JSON tool protocol
   far more reliably than the uncensored Llama 3.2 3B. Compare them with the eval suite before
   deciding: `deploy.py eval --live --suite tool_calling,rag,multi_turn` once with each `MODEL_ID`.
3. **Knowledge base scope.** Set Settings > Models & Training > Knowledge > scope to the folders you
   actually want auto-injected, or clear and re-index only what you need. The imported skills repo is
   better served by the skills import proposed in P3.
4. **Logging.** `LOG_LEVEL=TRACE` in `.env` writes every prompt fingerprint and debug event; use
   `INFO` in daily use. Also `chmod 600 .env` (it is 664 and holds secrets).

## 3. Proposals

Each proposal lists the change, the files it touches and how to verify it. P1 to P4 fix the three
observed failures; P5 to P9 make "every UI term in the chat is an instruction" true in general; P10
is the small-model assessor.

### P1. A UI-term intent lexicon that runs before the router (fixes 1, 2, 3)

The app's own vocabulary (Knowledge base, Codebase, Skills, Memory, History, Prompts, Tasks, Tools,
Projects, …) should be read as an instruction, deterministically, before any model decides
anything. Add `local_llm/intents.py` with an ordered table; the first match wins and returns an
`Intent(name, lane, tool_subset, retrieval, confirm, source="lexicon", confidence)`.

| Intent | Triggers (examples, case-insensitive) | Lane and tools | Retrieval | Confirmation |
|---|---|---|---|---|
| `skill_save` | add / save / make / keep … to / as / into (a / my) skill(s) | tool lane: `save_skill`, `final_answer` | off | confirm card with drafted name, description, body |
| `skill_use` | use (the / my) X skill | `load_skill` by fuzzy slug (library, then KB `SKILL.md`) | off | none |
| `skill_list` | list / show my skills, what skills | `list_skills` (new) | off | none |
| `skill_import` | import / load / sync … skills … from | `skills_import` (new, dry run) | off | admin + confirm |
| `kb_overview` | KB term + overview / what's in / list / how many / useful facts / summarize | `kb_overview` (new) then answer | KB | none |
| `kb_query` | knowledge base, kb, my docs/documents/library, indexed docs, according to my docs | `search_docs`, `kb_overview`, `final_answer` | KB only, web off | none |
| `kb_write` | index / ingest / add … to the knowledge base | `index_docs` / `index_url` | off | confirm for large folders |
| `codebase_review` | (the / this / my / our) codebase, repo, project, code, source, app, or the project folder name + review / audit / check / analyse / summarise / design / architecture / mistakes | `repo_map` (new), `list_files`, `search_files`, `read_file`, `file_info`, `delegate_task`, `final_answer` | off (the KB is a different corpus) | none |
| `codebase_lookup` | where is X defined / used / called, identifier-shaped token | `search_files` directly | off | none |
| `memory_write` | remember (that) X, note that, don't forget | `remember` directly | off | none |
| `memory_read` | what do you remember / know about me, my notes | `recall_memory` | off | none |
| `memory_forget` | forget X | `forget` after showing the matched key | off | ask if ambiguous |
| `history` | past / previous chats, last time, we discussed, search my history | `search_past_conversations` | off | none |
| `document_create` | make / craft / export … pdf, docx, xlsx, pptx, report | doc lane; research source from the other targets in the same message (KB, codebase, history, web) | as the source intent | none |
| `self_describe` | what can you do, what tools do you have | `capabilities_overview` (new, deterministic) | off | none |
| `disabled_capability` | run the tests, shell, git, email, calendar, spreadsheet, image, database, http, when not registered | deterministic "not enabled" reply + deep link to Settings > Tools > Catalogue | off | none |
| `admin_action` | add a user, create an API token, always allow X, restart / switch the model, retrain, remote access, MCP config | never act: explain + deep link (admins only) | off | n/a |
| `conv_action` | export / regenerate / pin / rename / fork / delete this chat | client action button via a `ui_action` event; delete behind uiConfirm | n/a | yes for delete |
| `schedule` | every morning, daily, remind me, schedule | `task_scheduler` if registered, else deep link to Tasks prefilled | off | confirm card |

The full inventory behind this table (41 UI terms, synonyms, which tool each needs, whether it is
registered and routable today) was produced during this review; the important gaps are in P3.

Code changes:

- `local_llm/intents.py` (new): the table, `classify(message, has_attachment, has_history,
  project_basename) -> Intent | None`, and `INTENT_TOOL_SUBSETS` generalising
  `Agent.document_tool_subset` (`agent.py:1984`).
- `local_llm/agent.py` `run()`: call `classify` right after `routing_override` (`textutil.py:1790`)
  and before knowledge triage (`agent.py:3980`); when it returns an intent, skip `route()`
  (`agent.py:1738`), build the base with `tool_subset=INTENT_TOOL_SUBSETS[intent]` (the existing
  narrowing path in `build_agent_system_prompt`, `llm.py:159`), and apply the intent's retrieval rule.
- `local_llm/textutil.py`: fold `refers_to_project` (`:1612`) into the codebase intent and fix the
  probe-found false positives (`is_code_request` on "summarize the code", "use my code-review prompt",
  "create an api token for my script"; `refers_to_project` on any URL with a two-segment path).
- Router fallback: change the router's menu from tool names to intent labels (the same names as the
  table plus `answer` and `web`), so the existing 64-token call picks a lane, not a tool. Add
  `route_hint`s for `search_docs`, `search_past_conversations`, `recall_memory` and `search_skills`
  (`tools.py`) so the router can reach what the lexicon misses.
- Slash commands become the same intents (`/kb`, `/code`, `/skills`, `/history`, `/mem`, `/pdf`),
  and `/kb` actually calls `search_docs` (today it only changes a notice).

Verify: a `routing` eval suite (P9) with the three failed messages and the 75-phrase probe; target
100% intent accuracy on lexicon cases.

### P2. Retrieval gating and relevance (fixes 1 and 3)

- **Overlap filter bug.** `agent.py:1952` counts tokens found in the chunk **or the path**. "skills"
  appears in 3467 of 3473 paths, so every hit passes. Count chunk tokens only; give a path match
  credit only when the user named the file. Ignore query tokens that occur in more than ~20% of
  documents (an IDF floor), and keep at most two passages per document.
- **Intent-gated RAG.** Auto-RAG runs only for `chat`/`kb_query`/`document_create`-from-KB intents.
  It is off for codebase, skills, memory, history and attachments-as-referent turns.
- **Collections.** Tag documents with their import id or folder at index time and let the admin mark
  a collection "search on request only". Imported chat histories and third-party repos default to
  on-request; they remain reachable through `search_docs` and `@kb:<path>`.
- **Query rewriting.** For KB intents, rewrite the user message into 1 to 3 search queries (by the
  task model in P10, or a keyword extractor until then) instead of OR-ing every word.

### P3. Missing tools behind UI terms (fixes 1 and 2)

| Tool | Purpose | Notes |
|---|---|---|
| `kb_overview` | Totals, folders with counts, titles, `SKILL.md` descriptions, newest documents; paged | Read-only; answers "what is in my knowledge base", feeds KB-sourced documents |
| `repo_map` | Directory tree with sizes, languages, entry points, import fan-in, top-level symbols, honouring `.gitignore` | Read-only; the first step of every codebase review; makes map-reduce through `delegate_task` possible on a 4096 context |
| `list_skills` | List the library (name, description, uses, reliability) | `search_skills` needs a query today |
| `skills_import` | Import `SKILL.md` files from a KB path prefix or a folder into the library; dry run with counts, collisions and the 200-skill cap | Admin + confirm; alternative: let `SkillLibrary` read KB `SKILL.md` as a read-only second source |
| `capabilities_overview` | Deterministic list of registered tools, disabled capabilities with the Settings switch, KB size, skills count, project dir | No model call |
| `prompt_library` | List / get / save prompts | Or client-side `/p name` expansion in the composer |
| `app_status` | Read-only status for admins (model, queue, perf) | Optional |

Capability groups (`config.py` `CAPABILITY_GROUPS`): add `skills` (load/search/save/list) and
`history` keys, and fold `search_docs` and `kb_overview` into `knowledge`, which today grants RAG but
not `search_docs`.

### P4. Referent resolution and clarification (fixes 1)

"this", "that", "it", "the file", "above" must resolve before anything runs: attachment text first,
then the previous assistant answer, then a path the user named. When nothing resolves, ask one
question instead of letting RAG supply a referent. Concretely:

- Show the router (and the P10 assessor) a one-line attachment summary
  (`[attached: devils-advocate.md, markdown, 6 KB, frontmatter name=devils-advocate]`), not just the
  8-token text. Today the router decides blind.
- `skill_save` with an attachment that already has `SKILL.md` frontmatter saves it as-is after the
  confirm card.
- Bug: composer uploads are stored with `conversation_id = NULL` (`attachments` row
  `att_10f56503054ed064b5cc4fbf`). Link the attachment to the conversation when the turn starts so
  history replay and "the file I sent earlier" work.

### P5. Document lane: research sources, one write, outputs folder (fixes 2)

- Research source comes from the message's other targets, not only web signals
  (`textutil.py:701`): KB terms add `search_docs` + `kb_overview`, codebase terms add `repo_map` +
  `read_file`, history terms add `search_past_conversations`. Extend `DOCUMENT_LOOKUP_TOOLS`
  (`agent.py:1982`) per source instead of one fixed tuple.
- After the first successful `create_document`, the turn is terminal unless the user asked for
  revisions; a second write to the same path in one turn needs an explicit revision intent. Skip the
  `agent_min_steps` revise pass for document turns.
- Write generated files to `outputs/` (or the project's outputs folder), never the repo root:
  `knowledge-base-craft-pdf.pdf` from turn 2 is sitting in the repository.

### P6. Reasoning budget (fixes 3)

- Append `REASONING_INSTRUCTION` (`llm.py:65`) only when the served model is a thinking model
  (`catalog.py` already has a `thinking` flag per entry). For other models use a one-line "plan in at
  most three lines" or nothing.
- Cap reasoning per step (`THINK_MAX_TOKENS`, e.g. 384): when the `ThinkSplitter` has emitted that
  many think tokens, stop the stream and re-ask for the answer only. Step 2 of turn 3 spent its whole
  2048-token budget thinking.

### P7. Composer modes mean what they say

Today "Chat" is upgraded to the agent pipeline by knowledge triage for any message of 8+ characters
(`api.py:2304`) and "Agent" can still land in the zero-tool prose lane. Proposed: Chat = prose lane,
no triage upgrade; Agent = skip the router, use the intent's tool subset (or the full catalogue);
Auto = lexicon, then assessor, then router. Show slash commands in the composer placeholder and the
palette, and show an "Understood as: Codebase review" chip in the trace with a one-click override;
that chip is what makes the UI-term mapping visible and correctable.

### P8. Safety around `project_dir` pointing at the app itself

`project_dir` is this repository. Add a deny-list to `ToolRegistry._resolve` for read and write
(`.env`, `.env.*`, `data/`, `logs/`, `adapters/`, `.venv/`, `*.db`) so `read_file` cannot read secrets
and `write_file`/`edit_file` cannot modify the running app's data. Generated documents go to
`outputs/` (P5).

### P9. Observability and regression tests

- Log `router.decision` at INFO with `intent`, `source` (lexicon / exemplar / assessor / router),
  `confidence` and the tool subset, metadata only.
- Add a `routing` suite to `local_llm/eval_suites.py`: the three failed messages (with the attachment
  for case 1), the 75-phrase probe, and expected intent, tool subset and RAG on/off per case. It runs
  in scripted mode in CI and live with `deploy.py eval --live --suite routing`.
- Test isolation bug: the test suite writes into the real `logs/app.log` (1298 `auth.admin_bootstrapped`
  events). Point `LOG_DIR` at a temp folder in every test module, as `DATA_DIR` already is.

### P10. A small "task model" that assesses each prompt

Other projects split this job off the main model. From my knowledge (the research step of this
review was cut off, so treat specifics as unverified): Open WebUI has a separate "Task Model" setting
used for titles, tags, search-query generation and autocomplete; Aider uses a "weak model" for commit
messages and history summaries; RouteLLM (LMSYS) trains small routers on preference data to choose
between a strong and a weak model; Katanemo's Arch-Router-1.5B maps a query to user-defined route
policies; aurelio's semantic-router routes by embedding similarity to example utterances; NVIDIA's
prompt-task-and-complexity classifier labels task type and complexity with a DeBERTa model.

For this app the right shape is tiered, cheapest first, and the first confident tier wins:

| Tier | What | Cost | Decides when |
|---|---|---|---|
| 0 | Lexicon (P1) + slash commands | ~0 ms | an explicit UI term or command matches |
| 1 | Exemplar matcher: word + character n-gram TF-IDF against ~10 examples per intent, no dependencies | ~1 ms | top score ≥ 0.6 and margin ≥ 0.25 |
| 2 | Small LLM assessor returning strict JSON (below) | ~0.2 to 0.6 s (estimate, 0.5B to 1.5B on an M1) | otherwise, when enabled |
| 3 | Today's router call on the main model, returning an intent label | ~3 s measured in these logs | assessor disabled, timed out or unparsable |

A scratch prototype of tiers 0 and 1 classified all three failed messages correctly with high
margin (codebase review 0.88, margin 0.64) and correctly reported low confidence on ambiguous text
such as "is there a race condition in the streaming code" (0.12), which is where tier 2 earns its
keep.

**Assessor output** (parsed with `toolparse` JSON repair; any parse failure falls through to tier 3):

```json
{
  "intent": "codebase_review",
  "targets": ["codebase"],
  "referent": "none | attachment | previous_answer | named_path | inline",
  "clarify_question": "",
  "retrieval": {"scope": "off | kb | history | codebase", "queries": ["..."]},
  "output": {"format": "chat | pdf | docx | xlsx | pptx | md | code", "language": "en"},
  "complexity": "low | medium | high",
  "confidence": 0.0
}
```

**Model and hosting on an 8 GB M1** (sizes approximate, latencies are estimates to be measured):

- Candidates: `Qwen2.5-0.5B-Instruct-4bit` (~0.3 GB, fastest, weakest), `Qwen2.5-1.5B-Instruct-4bit`
  (~0.9 GB, more reliable JSON), `Llama-3.2-1B-Instruct-4bit` (~0.7 GB). Start with Qwen2.5-0.5B and
  let the routing eval suite decide whether 1.5B is worth the RAM.
- Hosting: a second `ModelServerManager` on its own port managed like the main one (watchdog,
  status, `describe()`), with a tiny prompt cache and `decode_concurrency 1`. The admission
  controller must count its memory. On a Mini/Studio cluster the assessor runs on the node that
  receives the request (the Mini) and the main generation is routed as today.
- If the main model and the assessor share a tokenizer family, the same small model can also be the
  speculative-decoding draft (`DRAFT_MODEL`), so the RAM is spent once. With the current Llama 3.2
  3B main model that would be Llama-3.2-1B; with Qwen2.5-3B it would be Qwen2.5-0.5B.

**Other jobs for the same task model**: conversation titles (`maybe_autotitle`), KB and history
search-query rewriting (P2), tool-result summaries (`summarise_tool_results`), and the referent
summary of attachments (P4). Each of these runs on the 3B main model today and takes its slot.

**Config**: `ASSESSOR_ENABLED` (default 0), `ASSESSOR_MODEL` (alias, default `tiny`),
`ASSESSOR_TIMEOUT_MS` (800), `ASSESSOR_MIN_CONFIDENCE` (0.6), `INTENT_LEXICON_ENABLED` (default 1).
Settings: a "Routing" group under Agent.

**Evaluation**: the `routing` suite (P9) measures intent accuracy, tool-subset correctness, RAG
gating and latency per tier; compare "lexicon + exemplar" against "+ 0.5B assessor" against
"+ 1.5B assessor" before turning tier 2 on by default.

## 4. Order of work

| Phase | Proposals | Effort (estimate) | Risk |
|---|---|---|---|
| A | Section 2 settings changes | minutes | none |
| B | P1 lexicon + intent tool subsets, P2 overlap fix + intent-gated RAG, P4 referent + attachment link bug, P5 document lane, P6 reasoning gate, P8 deny-list, P9 logging + routing suite | 1 to 2 days | low: deterministic, covered by the routing suite |
| C | P3 tools (`kb_overview`, `repo_map`, `list_skills`, `capabilities_overview`, `skills_import`) and capability groups | 1 to 2 days | low |
| D | P10 tier 1 exemplar matcher, then tier 2 assessor and task-model jobs | 2 to 3 days | medium: RAM on 8 GB, needs live evaluation |
| E | P7 composer modes, intent chip, slash hints, `ui_action` deep links | 1 day | low |

Phase B alone fixes all three conversations reviewed here.
