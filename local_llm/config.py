"""The Config dataclass: every setting, its env var, and its clamps.

Split out of the original single-file deploy.py; behaviour is unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import re
import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

from .core import *  # noqa: F401,F403
from .sysutil import *  # noqa: F401,F403
from .toolparse import *  # noqa: F401,F403


# The search provider is locked to DuckDuckGo Lite (see websearch.SearchBackend).
# These are the only spellings accepted for the SEARCH_BACKEND env var / CLI flag;
# every one of them resolves to the same single provider. Anything else is a
# request for a different engine and is refused.
_DDG_LITE_ALIASES = {
    "duckduckgo_lite", "duckduckgo-lite", "duckduckgo", "ddg", "ddg-lite",
    "ddg_lite", "ddglite", "lite", "",
}


def _normalize_search_backend(value: str | None) -> str:
    """Force the search provider to DuckDuckGo Lite.

    Historical aliases (ddg, duckduckgo, lite, ...) are accepted silently; a
    value that names any other engine (google, bing, brave, tavily, searxng, ...)
    is warned about once and still normalised to duckduckgo_lite. This is the
    config-layer enforcement of the "DuckDuckGo Lite only" constraint: no other
    provider can be selected here, so nothing downstream ever has to choose one.
    """
    name = str(value or "").strip().lower()
    if name and name not in _DDG_LITE_ALIASES:
        log(f"Search provider is locked to DuckDuckGo Lite; ignoring requested "
            f"backend {value!r}.", logging.WARNING)
    return "duckduckgo_lite"


def is_usable_cutoff(value) -> bool:
    """True if value is a training cutoff the freshness test can compare on.

    ONE definition, used by both the settings-update path and the sanitiser.
    Two copies drifted the moment one of them learned about ranges: a shape-
    only check accepts "9999-99", so an edit that the sanitiser would reject
    was adopted by the update loop and then blanked -- the exact
    adopt-then-clobber this pair exists to prevent, for the one value most
    likely to be a typo.
    """
    m = re.fullmatch(r"(\d{4})(?:-(\d{2}))?", str(value or "").strip())
    return bool(m and 2000 <= int(m.group(1)) <= 2100
                and (m.group(2) is None or 1 <= int(m.group(2)) <= 12))


@dataclass
class Config:
    model: str = DEFAULT_MODEL
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    # Identity line prepended to the system prompt so the model states who it is
    # rather than falling back on the base model's pretrained identity (e.g.
    # "created by Alibaba Cloud" for Qwen). Overriding identity is a prompt job,
    # not a weights job. Set ASSISTANT_IDENTITY to "" to disable, or to your own
    # text. Defaults to the app's brand name.
    identity: str = field(default_factory=lambda: os.environ.get(
        "ASSISTANT_IDENTITY",
        f"You are {APP_NAME}, a private local AI assistant running on the user's "
        f"Apple Silicon Mac. If asked who or what you are, or who made you, say "
        f"you are {APP_NAME}, a local assistant; do not claim to be built by any "
        f"particular company or name an underlying base model.").strip())
    # Refuse to fine-tune on fewer than this many approved examples: a tiny set
    # overfits and causes catastrophic forgetting rather than a useful shift.
    train_min_examples: int = field(default_factory=lambda: int(os.environ.get("TRAIN_MIN_EXAMPLES", "16")))
    # Knowledge base (RAG): when documents are indexed, the best-matching passages
    # are prepended to the question so answers come from your own material with
    # citations. Uses SQLite FTS5 (BM25) — no embedding model, so it costs no
    # extra memory on an 8GB machine. No-op while the index is empty.
    rag_enabled: bool = field(default_factory=lambda: os.environ.get("RAG_ENABLED", "1") == "1")
    rag_passages: int = field(default_factory=lambda: int(os.environ.get("RAG_PASSAGES", "4")))
    # Comma-separated document paths to restrict retrieval to. Empty = search the
    # whole knowledge base. Set from the UI to "chat with this document".
    rag_scope: str = field(default_factory=lambda: os.environ.get("RAG_SCOPE", "").strip())
    # Local codebase the file tools read and edit. When empty, tools stay in the
    # sandboxed ./workspace. When set to a project directory, the agent can read
    # and change that project's files in place; you review with git and decide
    # what to push. Confined to this directory; .git is never written.
    project_dir: str = field(default_factory=lambda: os.environ.get("PROJECT_DIR", "").strip())
    # Command run by the run_tests tool (in the project dir). Empty = auto-detect
    # (pytest / npm test / cargo test / go test) from the project's files.
    test_command: str = field(default_factory=lambda: os.environ.get("TEST_COMMAND", "").strip())
    # Execution backend for run_shell/run_python/run_tests. "local" is the
    # confined subprocess (fast, but runs with your privileges). "docker" runs
    # the command inside a container with the project mounted, which IS a real
    # boundary: the agent can iterate freely and the blast radius stays in the
    # container. You still review the diff on the host and push manually.
    # After an agent turn edits files, optionally run the tests and, if they fail,
    # let the agent see the failures and try again, up to this many rounds (0 =
    # off). Needs execution enabled (--allow-shell) and a detectable test command.
    auto_iterate_rounds: int = field(default_factory=lambda: int(os.environ.get("AUTO_ITERATE_ROUNDS", "2")))
    exec_backend: str = field(default_factory=lambda: os.environ.get("EXEC_BACKEND", "local").strip().lower())
    docker_image: str = field(default_factory=lambda: os.environ.get("DOCKER_IMAGE", "python:3.12-slim").strip())
    # --- Sandbox limits, applied to every command the agent runs ------------
    # Extra environment variables model-written code is allowed to see, by name.
    # Everything not on the built-in allowlist is dropped: _exec used to hand
    # the whole server environment to the subprocess, so a one-line script could
    # print AUTH_ADMIN_PASSWORD, O365_CLIENT_SECRET and NODE_TOKEN straight into
    # the transcript -- and the transcript is what the feedback table harvests
    # for LoRA training. Deny by default, opt in by name for the few a project's
    # tests genuinely need (e.g. "DATABASE_URL,STRIPE_TEST_KEY").
    exec_env_passthrough: str = field(default_factory=lambda: os.environ.get(
        "EXEC_ENV_PASSTHROUGH", "").strip())
    # Memory ceiling for a sandboxed command. Enforced by the docker backend;
    # macOS refuses RLIMIT_AS/DATA outright, so the local backend reports it as
    # unenforced rather than pretending (see sandbox_limits()).
    exec_memory_mb: int = field(default_factory=lambda: int(
        os.environ.get("EXEC_MEMORY_MB", "2048")))
    # Largest file a sandboxed command may write. Enforced on both backends, and
    # the one limit that stops a runaway loop filling the disk the model server
    # also needs.
    exec_max_file_mb: int = field(default_factory=lambda: int(
        os.environ.get("EXEC_MAX_FILE_MB", "256")))
    # Process ceiling inside a container (docker --pids-limit). Not applied
    # locally: RLIMIT_NPROC counts every process the whole UID owns, so a small
    # number there would stop a command forking at all on a busy desktop.
    exec_pids_limit: int = field(default_factory=lambda: int(
        os.environ.get("EXEC_PIDS_LIMIT", "512")))
    # Refuse a handful of unmistakably catastrophic shell commands (rm -rf /,
    # mkfs, dd to a device, fork bombs, sudo, git push). A guardrail against a
    # 3B model's mistake, which is the actual threat model on a personal
    # machine -- NOT a security boundary, and it cannot be one: the local
    # backend runs with your privileges by design.
    exec_guardrail: bool = field(default_factory=lambda: os.environ.get(
        "EXEC_GUARDRAIL", "1") == "1")
    # Tests get their own, longer clock. A suite that legitimately takes two
    # minutes was being killed by the 30-second tool timeout and reported as a
    # failure, which sent the auto-iterate loop off fixing code that was fine.
    test_timeout: int = field(default_factory=lambda: int(
        os.environ.get("TEST_TIMEOUT", "180")))
    model_port: int = field(default_factory=lambda: int(os.environ.get("MODEL_PORT", "8080")))
    web_port: int = field(default_factory=lambda: int(os.environ.get("WEB_PORT", "8000")))
    # Iteration count is DERIVED from the corpus, not fixed: at batch size 1 a
    # constant 300 iters is ~19 epochs over the 16-example minimum (memorisation)
    # and well under one epoch once tool traces are included (some rows never
    # seen). Set TRAIN_ITERS to a non-zero value to pin it manually anyway.
    train_iters: int = field(default_factory=lambda: int(os.environ.get("TRAIN_ITERS", "0")))
    train_epochs: float = field(default_factory=lambda: float(os.environ.get("TRAIN_EPOCHS", "3")))
    # Floor and ceiling on the derived count: enough steps for the optimiser to
    # move at all, few enough that a big corpus cannot run for hours unattended.
    train_min_iters: int = field(default_factory=lambda: int(os.environ.get("TRAIN_MIN_ITERS", "40")))
    train_max_iters: int = field(default_factory=lambda: int(os.environ.get("TRAIN_MAX_ITERS", "2000")))
    train_lr: str = field(default_factory=lambda: os.environ.get("TRAIN_LR", "3e-5"))
    train_batch_size: int = field(default_factory=lambda: int(os.environ.get("TRAIN_BATCH_SIZE", "1")))
    # 512 truncates the target, not just the prompt: the system prompt alone is
    # ~210 tokens, and mlx-lm cuts an over-long sequence rather than dropping it,
    # so a short window trains the model to stop mid-answer. Examples that still
    # do not fit are dropped at export instead (see train_drop_over_length).
    train_seq_len: str = field(default_factory=lambda: os.environ.get("TRAIN_SEQ_LEN", "1024"))
    train_drop_over_length: bool = field(default_factory=lambda: os.environ.get("TRAIN_DROP_OVER_LENGTH", "1") == "1")
    # Tuning only the top layers is what keeps a 3B trainable on 8GB while the
    # web process and the page cache are also resident.
    train_num_layers: int = field(default_factory=lambda: int(os.environ.get("TRAIN_NUM_LAYERS", "8")))
    # Fine-tuning method. "lora" (default) trains small adapters and fits on 8GB.
    # "dora" is a slightly heavier LoRA variant. "full" fine-tunes all weights and
    # needs far more memory than an 8GB Mac has — set it (and TRAIN_NUM_LAYERS=-1)
    # only once you move to bigger hardware. The dataset format is identical
    # across all three, so the training set you build now is already future-proof.
    train_fine_tune_type: str = field(default_factory=lambda: os.environ.get("TRAIN_FINE_TUNE_TYPE", "lora"))
    # LoRA shape. mlx-lm takes these through a config file rather than flags, so
    # they are written to data/sft/lora_config.yaml and passed with --config.
    train_lora_rank: int = field(default_factory=lambda: int(os.environ.get("TRAIN_LORA_RANK", "8")))
    train_lora_scale: float = field(default_factory=lambda: float(os.environ.get("TRAIN_LORA_SCALE", "20.0")))
    train_lora_dropout: float = field(default_factory=lambda: float(os.environ.get("TRAIN_LORA_DROPOUT", "0.0")))
    train_on_tool_calls: bool = field(default_factory=lambda: os.environ.get("TRAIN_ON_TOOL_CALLS", "1") == "1")
    train_tool_examples: int = field(default_factory=lambda: int(os.environ.get("TRAIN_TOOL_EXAMPLES", "400")))
    # Tool traces are the model's own output fed back as ground truth, so they
    # are capped as a MULTIPLE of the human-approved rows rather than taken at
    # face value: at the old defaults 400 traces against a 16-example minimum
    # made the human feedback 4% of the gradient. 0 disables the cap.
    train_tool_ratio: float = field(default_factory=lambda: float(os.environ.get("TRAIN_TOOL_RATIO", "3.0")))
    # "rated" keeps only traces from conversations whose answer a human approved;
    # "all" is the old behaviour (anything that did not raise). "did not raise"
    # is not "was correct", and training on unfiltered self-output compounds the
    # model's existing tool-selection bias with every retrain.
    train_tool_quality: str = field(default_factory=lambda: os.environ.get("TRAIN_TOOL_QUALITY", "rated").strip().lower())
    # Rehearsal against catastrophic forgetting: this share of the FINAL mixed
    # set is drawn from data/sft/replay.jsonl and interleaved throughout, so the
    # adapter keeps a general-capability anchor. 0 disables it.
    train_replay_ratio: float = field(default_factory=lambda: float(os.environ.get("TRAIN_REPLAY_RATIO", "0.15")))
    # Fraction of the corpus held out. Never trained on, and the run is judged
    # against it (see train_val_check).
    train_val_split: float = field(default_factory=lambda: float(os.environ.get("TRAIN_VAL_SPLIT", "0.1")))
    # Roll back rather than promote an adapter whose held-out loss ended worse
    # than it started. mlx-lm measures the first validation loss BEFORE any
    # update, so the first reading is the pre-training baseline. Without this the
    # only failure the backup protects against is a non-zero exit code.
    train_val_check: bool = field(default_factory=lambda: os.environ.get("TRAIN_VAL_CHECK", "1") == "1")
    # How much worse than the baseline still counts as "not worse" (noise band).
    train_val_tolerance: float = field(default_factory=lambda: float(os.environ.get("TRAIN_VAL_TOLERANCE", "0.02")))
    # Promote the best-scoring periodic checkpoint instead of the last one, which
    # is early stopping after the fact and costs nothing: mlx-lm has already
    # written the checkpoints.
    train_promote_best: bool = field(default_factory=lambda: os.environ.get("TRAIN_PROMOTE_BEST", "1") == "1")
    # Hard ceiling on one training run. A hung trainer otherwise leaves the model
    # server stopped indefinitely with the UI stuck on "Training LoRA adapter".
    train_timeout: float = field(default_factory=lambda: float(os.environ.get("TRAIN_TIMEOUT", "7200")))
    # Adapter backups to keep. Each is a full copy; nothing else deletes them.
    train_max_backups: int = field(default_factory=lambda: int(os.environ.get("TRAIN_MAX_BACKUPS", "5")))
    max_tokens: int = field(default_factory=lambda: int(os.environ.get("MAX_TOKENS", "512")))
    auto_retrain_threshold: int = field(default_factory=lambda: int(os.environ.get("AUTO_RETRAIN_THRESHOLD", "0")))

    # Context window. context_size is the budget this process enforces when it
    # assembles a request; max_kv_size is what the model server is told to
    # allocate. They are separate because the server flag is optional and
    # changing it needs a restart, while context_size takes effect immediately.
    context_size: int = field(default_factory=lambda: int(
        os.environ.get("CONTEXT_SIZE") or _default_context_for_ram(TOTAL_RAM_GB)))
    max_kv_size: int = field(default_factory=lambda: int(os.environ.get("MAX_KV_SIZE", "0")))
    temperature: float = field(default_factory=lambda: float(os.environ.get("TEMPERATURE", "0.7")))
    history_turns: int = field(default_factory=lambda: int(os.environ.get("HISTORY_TURNS", "20")))

    # Agent
    agent_enabled: bool = field(default_factory=lambda: os.environ.get("AGENT_ENABLED", "0") == "1")
    # Before answering a substantive question, ask the model in one word whether
    # it needs to look the answer up. This is the general form of the fast-path
    # regexes: instead of enumerating every phrasing, let the model judge, but
    # in a shape a small model handles well (a single SEARCH/ANSWER token) and
    # then seed the search deterministically so it does not depend on the model
    # emitting tool-call JSON. Biased toward SEARCH when unsure, since a needless
    # search is cheaper than a confident wrong answer or a refusal.
    knowledge_triage: bool = field(default_factory=lambda: os.environ.get("KNOWLEDGE_TRIAGE", "1") == "1")
    # Let a tool-free turn switch into agent mode once it is clear it needed
    # tools after all. The router decides answer-vs-tool before the model has
    # seen how hard the question is, from one 64-token call on a small model,
    # and when it guesses wrong the turn is locked out of every tool for the
    # rest of its life: the reply that came back was "I don't have access to
    # real-time data", with a working web_search sitting one step away. The
    # switch fires on the model's OWN report that it lacked a capability, which
    # is evidence the router never had, and at most once per turn.
    agent_escalation: bool = field(default_factory=lambda: os.environ.get("AGENT_ESCALATION", "1") == "1")
    # Finish an answer that stopped at the reply limit without being asked. The
    # reply budget is a property of the context window, not of the request: a
    # program that needs 2000 tokens on a 4096-token context gets cut off
    # however it is phrased. Leaving it there and printing "ask me to continue"
    # made finishing the work the user's job, one "go on" at a time, and every
    # one of those round trips re-prefills the whole prompt. The continuation
    # lane already existed for the manual case; this stops it waiting to be
    # asked. 0 restores the old behaviour.
    auto_continue: bool = field(default_factory=lambda: os.environ.get("AUTO_CONTINUE", "1") == "1")
    # How many continuation passes one turn may spend. Each is a full
    # generation, so this is a wall-clock budget as much as a quality one.
    auto_continue_max: int = field(default_factory=lambda: int(
        os.environ.get("AUTO_CONTINUE_MAX", "3")))
    # After a lookup search, automatically fetch this many of the top result
    # pages and give the model their full text, not just the snippet. This is
    # what makes one generic search path answer domain-specific questions (a
    # stock price, a score, a forecast, a version number): the answer is usually
    # on the page even when the snippet omits it. 0 disables auto-fetch and falls
    # back to snippet-only plus the model choosing to call fetch_url itself.
    auto_fetch_results: int = field(default_factory=lambda: int(os.environ.get("AUTO_FETCH_RESULTS", "2")))
    # Hardest ceiling on characters of a fetched page that may enter the prompt.
    # A large docs page would otherwise dominate context and OOM prefill on an
    # 8GB machine. Roughly char_cap/4 tokens.
    auto_fetch_char_cap: int = field(default_factory=lambda: int(
        os.environ.get("AUTO_FETCH_CHAR_CAP") or _default_fetch_cap(TOTAL_RAM_GB)))
    agent_max_steps: int = field(default_factory=lambda: int(os.environ.get("AGENT_MAX_STEPS", "6")))
    # How hard the agent thinks about an ordinary question. A small model answers
    # the instant it can, not the instant it should: four out of five turns used
    # to end at step 1 with whatever the first pass produced. Below
    # agent_min_steps the loop refuses to settle on a plain-text reply -- it
    # hands the draft back with a critique instruction and spends another step
    # improving it. 1 restores the old one-shot behaviour; every step above it
    # costs one more generation per turn, which on 8GB is seconds, not
    # milliseconds. Never exceeds agent_max_steps.
    agent_min_steps: int = field(default_factory=lambda: int(os.environ.get("AGENT_MIN_STEPS", "2")))
    # Resilience under memory pressure. When a generation errors (a model-server
    # OOM kill and watchdog restart look like a dropped connection from here), the
    # turn retries with a smaller token budget rather than surfacing an error.
    # resilient_retries is how many times; min_max_tokens is the floor it shrinks
    # to. hard_step_cap lets a big task keep going past agent_max_steps by
    # compacting progress into a running summary, so it slows down but does not
    # stop. These exist to satisfy "never error or stop; step down and continue".
    resilient_retries: int = field(default_factory=lambda: int(os.environ.get("RESILIENT_RETRIES", "3")))
    min_max_tokens: int = field(default_factory=lambda: int(os.environ.get("MIN_MAX_TOKENS", "128")))
    # Reply budget for a CODE request. The general max_tokens default (512) cuts a
    # program off mid-function, which is what happened to a C++ CRUD answer: the
    # text simply stopped at "std::string newTitle, new". Code gets more room,
    # bounded by the context window at use time.
    code_max_tokens: int = field(default_factory=lambda: int(os.environ.get("CODE_MAX_TOKENS", "1536")))
    # ----------------------------------------------------------------- #
    # Multi-turn task continuity (see taskstate.py). The failure these
    # exist for: task identity lived only in the transcript, so trimming
    # the oldest messages threw away the language, the platform and the
    # only copy of the artifact, and a follow-up like "add more checks"
    # was answered as if it were a brand-new question.
    # ----------------------------------------------------------------- #
    # Keep a structured task/artifact record per conversation and put it in
    # front of the model on every turn. Off restores the old behaviour exactly.
    task_state_enabled: bool = field(default_factory=lambda: os.environ.get("TASK_STATE_ENABLED", "1") == "1")
    # Characters of the current artifact allowed into the prompt. Sized as a
    # share of the context at use time as well, so a small window shrinks it
    # rather than overflowing; this is the absolute ceiling.
    task_artifact_chars: int = field(default_factory=lambda: int(
        os.environ.get("TASK_ARTIFACT_CHARS", "8000")))
    # Check the finished answer against the active task and, on a strong
    # contradiction (Python where the task is Bash), regenerate once with a
    # corrected prompt instead of returning it.
    drift_check_enabled: bool = field(default_factory=lambda: os.environ.get("DRIFT_CHECK_ENABLED", "1") == "1")
    # Extra reply tokens granted on top of the artifact's own size when a turn
    # has to reproduce a whole file. Without it a 900-token script asked to grow
    # gets a budget smaller than the script it must re-emit.
    artifact_reply_headroom: int = field(default_factory=lambda: int(
        os.environ.get("ARTIFACT_REPLY_HEADROOM", "640")))
    # Above this many characters, a turn that changes the artifact is asked for
    # the changed sections only and the reply is spliced back in, instead of the
    # model re-emitting the whole file. Re-emitting a 3600-character module to
    # add a try/catch cost 45 seconds a turn and put the whole file at risk on
    # every one of them. 0 disables the patch lane.
    artifact_patch_chars: int = field(default_factory=lambda: int(
        os.environ.get("ARTIFACT_PATCH_CHARS", "2400")))
    # Check generated code before showing it: unbalanced braces, blocks spliced
    # in twice, a function shadowing a builtin and then recursing into itself,
    # and a short list of literal mistakes per language. Pure text, no tools.
    code_check_enabled: bool = field(default_factory=lambda: os.environ.get("CODE_CHECK_ENABLED", "1") == "1")
    # Additionally parse the code with a real parser (bash -n, py_compile, pwsh,
    # cc -fsyntax-only) when the host has one. Starts a subprocess, so it is
    # gated on allow_shell as well and is off unless explicitly enabled.
    code_check_deep: bool = field(default_factory=lambda: os.environ.get("CODE_CHECK_DEEP", "0") == "1")
    # Log the fully assembled prompt (redacted, and only when log_chat_content
    # is not "disabled") plus the task state, budget and lane for every model
    # call. Off by default: it is verbose and it writes conversation text.
    debug_prompts: bool = field(default_factory=lambda: os.environ.get("DEBUG_PROMPTS", "0") == "1")

    # If the model server sends nothing for this many seconds mid-generation, the
    # request is treated as stalled: it raises, and the resilient loop retries
    # with a smaller budget instead of hanging. This is what turns "stuck" into
    # visible "retrying" rather than minutes of dead air waiting on a wedged or
    # OOM-killed server.
    stall_timeout: int = field(default_factory=lambda: int(os.environ.get("STALL_TIMEOUT", "60")))
    # stall_timeout is a SILENCE detector, and it only means that on a stream,
    # where the HTTP read timeout is the gap between chunks. A non-streaming
    # completion has no chunks, so the same value becomes a cap on TOTAL
    # generation time: at this machine's ~16 tok/s a healthy 1536-token reply
    # takes ~96s and was being cut off at 60 and reported as "the model
    # stalled". Non-streaming calls therefore get a budget scaled to the work
    # requested, against a deliberately pessimistic floor decode rate.
    decode_floor_tps: float = field(default_factory=lambda: float(
        os.environ.get("DECODE_FLOOR_TPS", "4")))
    # Hard ceiling on that derived budget, so a huge reply request cannot mean
    # an unbounded wait.
    max_generation_timeout: float = field(default_factory=lambda: float(
        os.environ.get("MAX_GENERATION_TIMEOUT", "900")))
    # How long a retry waits for a restarting model server to become ready again
    # before giving up on that attempt. Longer helps slow cold-start reloads.
    ready_wait_timeout: float = field(default_factory=lambda: float(os.environ.get("READY_WAIT_TIMEOUT", "40")))
    hard_step_cap: int = field(default_factory=lambda: int(os.environ.get("HARD_STEP_CAP", "24")))
    # Incremental reasoning: for a hard analytical question with no tool to call,
    # decompose it into sub-steps and solve them one at a time, carrying only
    # short conclusions forward. Each pass is small, so the working set stays
    # inside 8GB no matter how deep the reasoning goes, and decomposition makes a
    # small model reason better than one shot. It is slower (several small calls)
    # by design: it takes its time instead of failing or answering shallowly.
    incremental_reasoning: bool = field(default_factory=lambda: os.environ.get("INCREMENTAL_REASONING", "1") == "1")
    # Chunk an oversized prompt (a pasted file or long document) and process it
    # part by part, so a single input larger than the context never has to be
    # prefilled in one pass. This is what lets an 8GB machine handle a large
    # prompt: split it, extract findings per chunk into bounded notes, then
    # synthesise. Chunk size and trigger are derived from context_size.
    chunk_large_prompts: bool = field(default_factory=lambda: os.environ.get("CHUNK_LARGE_PROMPTS", "1") == "1")
    # Emit verbose "under the hood" detail events (prompt sizes, per-step timing,
    # extracted lengths, fallback reasons) so the whole pipeline is visible.
    show_internals: bool = field(default_factory=lambda: os.environ.get("SHOW_INTERNALS", "1") == "1")
    # Hard wall-clock ceiling for the whole multi-source retrieval phase (fetch +
    # read all sources). Prevents a stalling extraction from grinding for minutes.
    retrieval_deadline: float = field(default_factory=lambda: float(os.environ.get("RETRIEVAL_DEADLINE", "90")))
    # Fraction of the context above which a prompt is chunked, and the fraction
    # of the context each chunk targets. Derived from context_size so they scale
    # with RAM; exposed so the thresholds themselves can be tuned per machine.
    chunk_trigger_ratio: float = field(default_factory=lambda: float(os.environ.get("CHUNK_TRIGGER_RATIO", "0.6")))
    chunk_size_ratio: float = field(default_factory=lambda: float(os.environ.get("CHUNK_SIZE_RATIO", "0.4")))
    reasoning_max_steps: int = field(default_factory=lambda: int(os.environ.get("REASONING_MAX_STEPS", "6")))
    # When a question is worth decomposing rather than answering in one pass.
    # reasoning_signals is how many analytical cues ("compare", "trade-offs",
    # "why does") make it automatic; reasoning_min_chars is the length at which
    # a single cue is enough on its own. Lower either one and more questions get
    # the full treatment -- a plan, a model call per step, then a synthesis --
    # which is the most expensive path here, so these are the latency-for-depth
    # dial. reasoning_signals=1 decomposes anything with one cue in it.
    reasoning_signals: int = field(default_factory=lambda: int(os.environ.get("REASONING_SIGNALS", "2")))
    reasoning_min_chars: int = field(default_factory=lambda: int(os.environ.get("REASONING_MIN_CHARS", "80")))
    # The model's training cutoff, as "YYYY-MM". Nothing in this pipeline used
    # to know what year the weights stopped at, and nothing put today's date in
    # front of the model either -- so "who won the 2026 final" and "who won the
    # 1998 final" were the same question to the freshness check and to the
    # model, and the first was answered confidently out of weights that could
    # not contain the answer. A year in the question is the most reliable
    # freshness signal there is; this is what makes it readable. Empty means
    # unknown, which is treated conservatively: any recent year counts as
    # needing a lookup, because guessing silently is the worse failure.
    model_cutoff: str = field(default_factory=lambda: os.environ.get("MODEL_CUTOFF", ""))
    # Put today's date in the system prompt. One line, and it is what lets the
    # model reason about recency at all -- without it "is this still current"
    # has no anchor and the answer is whatever the weights remember.
    date_in_prompt: bool = field(default_factory=lambda: os.environ.get("DATE_IN_PROMPT", "1") == "1")
    # Escalate an answer that hedges about external facts ("I'm not sure",
    # "I don't have specific information about that") into one lookup and a
    # second attempt. This is the case the refusal detector misses: a refusal
    # says it CANNOT answer and is short, a hedge answers anyway at length and
    # the guess reads as fact. It costs real seconds when it fires -- a search,
    # up to two page reads and another generation on a turn the user thought
    # had finished -- so it is one per turn, never against /no-search, and off
    # with HEDGE_ESCALATION=0.
    hedge_escalation: bool = field(default_factory=lambda: os.environ.get("HEDGE_ESCALATION", "1") == "1")
    # Hard wall-clock cap per reasoning step. Distinct from stall_timeout (which
    # only fires on zero output): this bounds a step that streams slowly but
    # never finishes, so a single step can never wedge the whole chain.
    reasoning_step_timeout: int = field(default_factory=lambda: int(os.environ.get("REASONING_STEP_TIMEOUT", "45")))
    # Per-step generation budget for reasoning, chunk and source-extraction
    # passes. RAM-scaled default; small on 8GB, larger on roomy machines.
    reasoning_tokens: int = field(default_factory=lambda: int(
        os.environ.get("REASONING_TOKENS") or _default_reasoning_tokens(TOTAL_RAM_GB)))
    allow_python: bool = field(default_factory=lambda: os.environ.get("ALLOW_PYTHON", "0") == "1")
    allow_shell: bool = field(default_factory=lambda: os.environ.get("ALLOW_SHELL", "0") == "1")
    # Comma-separated allowlist. Empty means every registered tool is offered.
    agent_tools: str = field(default_factory=lambda: os.environ.get("AGENT_TOOLS", ""))
    # Which LoRA adapter the model server loads: latest, none, or a backup id.
    adapter: str = field(default_factory=lambda: os.environ.get("ADAPTER", "latest"))
    # Extra model ids to offer in the switcher, on top of DEFAULT_MODEL_CATALOG.
    model_catalog: str = field(default_factory=lambda: os.environ.get("MODEL_CATALOG", ""))
    # How many background task runs may execute at once. The model server
    # serves one request at a time, so more than one mostly adds queueing.
    max_concurrent_tasks: int = field(default_factory=lambda: int(os.environ.get("MAX_CONCURRENT_TASKS", "1")))
    # How many chat/agent generations may run at once, and how deep the queue
    # behind them goes. Each in-flight generation holds its own KV cache in the
    # same unified memory as the model weights, so this is a memory limit as much
    # as a fairness one; the default scales with RAM.
    max_concurrent_generations: int = field(default_factory=lambda: int(
        os.environ.get("MAX_CONCURRENT_GENERATIONS")
        or _default_concurrent_generations(TOTAL_RAM_GB)))
    generation_queue_depth: int = field(default_factory=lambda: int(
        os.environ.get("GENERATION_QUEUE_DEPTH", "4")))
    task_poll_seconds: int = field(default_factory=lambda: int(os.environ.get("TASK_POLL_SECONDS", "2")))
    # Wall-clock cap for ONE agent inside a multi-agent run (/api/agents/run).
    # Without it a wedged or unreachable model server holds the whole request
    # open; on timeout that agent reports a failure and the others still return.
    agent_run_timeout: float = field(default_factory=lambda: float(
        os.environ.get("AGENT_RUN_TIMEOUT", "300")))
    # Seconds of interactive quiet before a scheduled run is allowed to start.
    chat_idle_seconds: int = field(default_factory=lambda: int(os.environ.get("CHAT_IDLE_SECONDS", "45")))
    # fetch_url refuses loopback and RFC1918 targets unless this is on, so a
    # prompt-injected page cannot make the agent read the machine's own
    # services (including this app's API) and hand the result back.
    allow_local_fetch: bool = field(default_factory=lambda: os.environ.get("ALLOW_LOCAL_FETCH", "0") == "1")

    # Reasoning-mode models (Qwen3.5 and later) emit a <think> block by default.
    # In a tool loop that is pure cost: the reasoning is discarded by the
    # protocol, it inflates the KV cache, and JSON inside it confuses parsing.
    disable_thinking: bool = field(default_factory=lambda: os.environ.get("DISABLE_THINKING", "0") == "1")
    # Make the model reason out loud before answering, and show that reasoning in
    # the trace. Adds a <think> instruction to the prompt so even non-reasoning
    # models produce visible step-by-step thinking. Costs tokens; turn off for
    # speed. On by default per request for maximum transparency.
    reasoning_visible: bool = field(default_factory=lambda: os.environ.get("REASONING_VISIBLE", "1") == "1")
    # Tool-selection steps want deterministic JSON; only the final answer wants
    # the configured temperature. One value for both costs malformed calls.
    tool_temperature: float = field(default_factory=lambda: float(os.environ.get("TOOL_TEMPERATURE", "0.0")))
    # Multiplicative penalty on tokens already in the window. Small quantised
    # models fall into verbatim repetition loops, and greedy decoding
    # (tool_temperature 0.0) has no way out of one: the argmax that produced the
    # loop keeps producing it until max_tokens runs out. 1.0 disables it.
    repetition_penalty: float = field(default_factory=lambda: float(os.environ.get("REPETITION_PENALTY", "1.1")))
    repetition_context_size: int = field(default_factory=lambda: int(os.environ.get("REPETITION_CONTEXT_SIZE", "64")))
    # OFF by default. Recent mlx-lm builds (see generate.py _step) crash with
    # "TypeError: 'NoneType' object is not iterable" over self.logits_processors
    # when the repetition_penalty request fields are present, killing the
    # generation thread so every request stalls. Sending no logits-processor
    # fields avoids the null list entirely. Set REPETITION_PENALTY_ENABLED=1 to
    # re-enable once upstream is fixed.
    repetition_penalty_enabled: bool = field(default_factory=lambda: os.environ.get("REPETITION_PENALTY_ENABLED", "0") == "1")
    # Answer arithmetic and bare URLs without a model round trip at all.
    fast_path: bool = field(default_factory=lambda: os.environ.get("FAST_PATH", "1") == "1")
    # When the trace outgrows the context, collapse the oldest steps into one
    # summary message instead of dropping them off the front. Dropping shifts
    # every following token and invalidates any server-side prefix cache at the
    # exact point a run is longest.
    stable_prefix: bool = field(default_factory=lambda: os.environ.get("STABLE_PREFIX", "1") == "1")

    # KV cache quantization, passed through to mlx_lm.server when the installed
    # build accepts the flags. 0 leaves the cache in fp16. NOTE: mlx-lm dropped
    # --kv-bits and --max-kv-size around 0.29; on those builds these are inert
    # and the memory levers below are the ones that apply. Both sets are still
    # attempted, because which flags exist depends on the installed version.
    kv_bits: int = field(default_factory=lambda: int(os.environ.get("KV_BITS", "0")))
    kv_group_size: int = field(default_factory=lambda: int(os.environ.get("KV_GROUP_SIZE", "64")))
    quantized_kv_start: int = field(default_factory=lambda: int(os.environ.get("QUANTIZED_KV_START", "1024")))
    prompt_cache_dir: str = field(default_factory=lambda: os.environ.get("PROMPT_CACHE_DIR", ""))

    # --- Model-server memory levers (mlx-lm >= 0.29 flag names) ------------- #
    # These are the flags that actually exist on a current mlx_lm.server, and
    # they are the difference between a marginal 8GB machine and one that dies
    # of a Metal command-buffer OOM mid-prefill. They are best-effort: an older
    # build that does not know a flag simply does not get it.
    prefill_step_size: int = field(default_factory=lambda: int(
        os.environ.get("PREFILL_STEP_SIZE") or _default_prefill_step(TOTAL_RAM_GB)))
    # How many distinct prefixes the server's prompt cache may hold. Small,
    # because a turn creates one per lane and each is a live KV sequence.
    prompt_cache_size: int = field(default_factory=lambda: int(
        os.environ.get("PROMPT_CACHE_SIZE", "2")))
    prompt_cache_bytes: int = field(default_factory=lambda: int(
        os.environ.get("PROMPT_CACHE_BYTES") or _default_prompt_cache_bytes(TOTAL_RAM_GB)))
    # Decode slots. 2 rather than 1 so the watchdog's one-token liveness probe
    # is batched alongside a long answer instead of queueing behind it: with a
    # single slot the probe cannot distinguish "busy for 90 seconds" from
    # "generation thread is dead", and a false restart would kill a live answer.
    # The extra slot costs one KV sequence of a handful of tokens.
    decode_concurrency: int = field(default_factory=lambda: int(
        os.environ.get("DECODE_CONCURRENCY", "2")))
    # Prefill stays serial: parallel prompt processing multiplies exactly the
    # allocation that OOMs.
    prompt_concurrency: int = field(default_factory=lambda: int(
        os.environ.get("PROMPT_CONCURRENCY", "1")))

    # --- Model-server watchdog --------------------------------------------- #
    # mlx_lm.server runs generation on one background thread. When that thread
    # raises (a Metal OOM does), the process stays alive and keeps answering
    # /v1/models and /health with 200 while every completion hangs forever. A
    # liveness check that only looks at process exit never sees this, so the
    # watchdog does two extra things: it scans the backend's own log for the
    # thread's traceback, and when the log has gone quiet it runs a real
    # one-token completion.
    watchdog_probe_interval: float = field(default_factory=lambda: float(
        os.environ.get("WATCHDOG_PROBE_INTERVAL", "30")))
    watchdog_probe_timeout: float = field(default_factory=lambda: float(
        os.environ.get("WATCHDOG_PROBE_TIMEOUT", "25")))
    # Consecutive probe failures before a restart. Guards against restarting a
    # server that is merely slow.
    watchdog_probe_failures: int = field(default_factory=lambda: int(
        os.environ.get("WATCHDOG_PROBE_FAILURES", "3")))
    # Restarting into an immediate OOM forever is worse than stopping and
    # saying so. More than this many auto-restarts inside the window and the
    # watchdog gives up, leaving an honest error status for the UI.
    watchdog_max_restarts: int = field(default_factory=lambda: int(
        os.environ.get("WATCHDOG_MAX_RESTARTS", "3")))
    watchdog_restart_window: float = field(default_factory=lambda: float(
        os.environ.get("WATCHDOG_RESTART_WINDOW", "600")))

    # --- Skills: procedural memory with progressive disclosure -------------- #
    # A skill is a short markdown document describing HOW to do one kind of
    # task. Only its one-line description sits in the prompt; the body enters
    # context when the model asks for it by name. That split is the whole point
    # on a 3B model in a 4096-token window: the catalogue costs ~15 tokens per
    # skill and the procedure costs nothing until it is needed.
    skills_enabled: bool = field(default_factory=lambda: os.environ.get("SKILLS_ENABLED", "1") == "1")
    # How many catalogue lines may enter the prompt. Above this, the ones most
    # relevant to the current message win and the rest stay invisible, so a
    # library of 200 skills still costs a fixed ~180 tokens.
    skills_in_prompt: int = field(default_factory=lambda: int(os.environ.get("SKILLS_IN_PROMPT", "12")))
    # Hard cap on a loaded skill body. A skill that does not fit the window it
    # exists to economise on is worse than no skill.
    skill_body_chars: int = field(default_factory=lambda: int(
        os.environ.get("SKILL_BODY_CHARS") or _default_fetch_cap(TOTAL_RAM_GB) // 2))
    skills_max: int = field(default_factory=lambda: int(os.environ.get("SKILLS_MAX", "200")))
    # Where skills live. Empty means DATA_DIR/skills. Explicit rather than
    # derived so a test, a second instance, or a shared library on another disk
    # can point somewhere else without patching module globals.
    skills_dir: str = field(default_factory=lambda: os.environ.get("SKILLS_DIR", "").strip())
    # Load the matching skill WITHOUT waiting for the model to ask. A 3B model
    # does not reach for load_skill on its own: given a house-format question
    # with the procedure one tool call away, it answers generically instead.
    # This is the same reasoning as the deterministic read_url and web_search
    # shortcuts -- the router second-guessing an obvious case is worse than not
    # asking it. Off means skills are only ever loaded when the model asks.
    skill_autoload: bool = field(default_factory=lambda: os.environ.get("SKILL_AUTOLOAD", "1") == "1")
    # Significant words a message and a skill must share before autoloading. The
    # cost of a false positive is an irrelevant procedure in a 4096-token
    # window, so this errs towards missing a match.
    skill_autoload_overlap: int = field(default_factory=lambda: int(
        os.environ.get("SKILL_AUTOLOAD_OVERLAP", "2")))
    # The closed loop: write a skill from a turn the user rated well, and record
    # an outcome against every skill the turn loaded. Off means skills still
    # work, they just never change on their own.
    skills_autolearn: bool = field(default_factory=lambda: os.environ.get("SKILLS_AUTOLEARN", "1") == "1")
    # Reply budget for the one small model call that drafts a skill. Small on
    # purpose: a skill that cannot be said in a page is not a skill.
    skill_author_tokens: int = field(default_factory=lambda: int(
        os.environ.get("SKILL_AUTHOR_TOKENS", "384")))
    # A skill is retired once it has this many RATED uses and more than this
    # share of them went badly. Both guards matter: without the minimum, one bad
    # day retires a good skill; without the rate, nothing is ever retired.
    skill_retire_min_rated: int = field(default_factory=lambda: int(
        os.environ.get("SKILL_RETIRE_MIN_RATED", "4")))
    skill_retire_loss_rate: float = field(default_factory=lambda: float(
        os.environ.get("SKILL_RETIRE_LOSS_RATE", "0.6")))

    # --- Project context and @-references ----------------------------------- #
    # Instructions found in the project root (.hermes.md, AGENTS.md, CLAUDE.md,
    # SOUL.md, .cursorrules) are loaded into the system prompt, so a repo that
    # already documents how to work in it does not have to be re-explained.
    context_files_enabled: bool = field(default_factory=lambda: os.environ.get("CONTEXT_FILES", "1") == "1")
    # Hard cap across ALL of those files together. A CLAUDE.md written for a
    # frontier model with a 200k window would otherwise eat this machine's
    # entire 4096-token context before the question is even asked.
    context_files_chars: int = field(default_factory=lambda: int(
        os.environ.get("CONTEXT_FILES_CHARS") or _default_fetch_cap(TOTAL_RAM_GB) // 2))
    # @path, @folder, @url and @diff in a message are expanded inline before the
    # turn starts, so referring to a file is not a tool call the model has to
    # decide to make.
    reference_expansion: bool = field(default_factory=lambda: os.environ.get("REFERENCE_EXPANSION", "1") == "1")
    reference_max: int = field(default_factory=lambda: int(os.environ.get("REFERENCE_MAX", "5")))
    reference_chars: int = field(default_factory=lambda: int(
        os.environ.get("REFERENCE_CHARS") or _default_fetch_cap(TOTAL_RAM_GB)))
    # Cross-session recall: how many past exchanges search_past_conversations
    # returns, and how much of each. Small, because these are snippets meant to
    # remind the model, not transcripts.
    recall_results: int = field(default_factory=lambda: int(os.environ.get("RECALL_RESULTS", "6")))
    recall_snippet_chars: int = field(default_factory=lambda: int(
        os.environ.get("RECALL_SNIPPET_CHARS", "200")))

    # --- Delegation and scripted tool use ----------------------------------- #
    # A subagent is a second agent loop with its own context and its own KV
    # cache. On 8GB that is the binding constraint, so the default is ONE at a
    # time: the value of delegation here is an isolated context that does not
    # pollute the parent's 4096 tokens, not parallelism.
    delegation_enabled: bool = field(default_factory=lambda: os.environ.get("DELEGATION_ENABLED", "1") == "1")
    subagent_max_concurrent: int = field(default_factory=lambda: int(
        os.environ.get("SUBAGENT_MAX_CONCURRENT", "1")))
    # How many subagents one turn may spawn in total, and how many steps each
    # gets. A parent that delegates in a loop would otherwise never finish.
    subagent_max_per_turn: int = field(default_factory=lambda: int(
        os.environ.get("SUBAGENT_MAX_PER_TURN", "3")))
    subagent_max_steps: int = field(default_factory=lambda: int(
        os.environ.get("SUBAGENT_MAX_STEPS", "4")))
    # Characters of a subagent's answer that come back into the parent's
    # context. The whole point is that the parent pays for the CONCLUSION, not
    # for the work.
    subagent_result_chars: int = field(default_factory=lambda: int(
        os.environ.get("SUBAGENT_RESULT_CHARS", "2000")))
    # execute_code: one script that calls the tools itself, collapsing a
    # multi-step pipeline into a single turn. Gated behind ALLOW_PYTHON, which
    # is the same trust decision -- it runs model-written Python as your user.
    execute_code_max_calls: int = field(default_factory=lambda: int(
        os.environ.get("EXECUTE_CODE_MAX_CALLS", "40")))

    # --- Checkpoints -------------------------------------------------------- #
    # Record a file's original bytes before a turn overwrites it, so an edit can
    # be undone. Copy-on-write per file, not a snapshot of the tree: snapshotting
    # a real repository on every turn that changes one line is unusable.
    checkpoints_enabled: bool = field(default_factory=lambda: os.environ.get("CHECKPOINTS", "1") == "1")
    checkpoint_keep: int = field(default_factory=lambda: int(os.environ.get("CHECKPOINT_KEEP", "20")))
    # A file over this size is recorded as "not captured" rather than copied.
    # Silently duplicating a large artefact on every write would be worse than
    # admitting one file is not covered, which the manifest then says.
    checkpoint_max_file_bytes: int = field(default_factory=lambda: int(
        os.environ.get("CHECKPOINT_MAX_FILE_BYTES", "4000000")))

    # --- MCP ---------------------------------------------------------------- #
    # External Model Context Protocol servers, whose tools join the registry
    # alongside the built-ins. JSON, either as an array of
    # {name, command, args, env} or the {"mcpServers": {...}} object form other
    # clients use, so a config file can be copied across unchanged. Empty means
    # no MCP, and nothing is started.
    mcp_servers: str = field(default_factory=lambda: os.environ.get("MCP_SERVERS", "").strip())
    # Per-request deadline for an MCP call. A server that hangs must not hold a
    # turn open; the tool reports the timeout and the agent carries on.
    mcp_timeout: float = field(default_factory=lambda: float(os.environ.get("MCP_TIMEOUT", "30")))

    # Tools
    # Search provider is LOCKED to DuckDuckGo Lite. SEARCH_BACKEND is honoured
    # only insofar as it names a DuckDuckGo-Lite alias; any request for another
    # engine is ignored (with a warning) and normalised back to duckduckgo_lite.
    search_backend: str = field(default_factory=lambda: _normalize_search_backend(
        os.environ.get("SEARCH_BACKEND", "duckduckgo_lite")))
    search_results: int = field(default_factory=lambda: int(os.environ.get("SEARCH_RESULTS", "5")))
    tool_timeout: int = field(default_factory=lambda: int(os.environ.get("TOOL_TIMEOUT", "30")))
    # Two different caps, and the difference matters.
    #
    # tool_raw_chars is how much a tool may return at all. It is what gets
    # logged and shown in the UI, and what the summariser reads.
    #
    # tool_result_chars is how much may enter the model's context. A result is
    # re-sent on every later step, so a 4000-character page is not a 4000-token
    # cost, it is that times the number of steps that follow.
    #
    # Collapsing these two into one number means the raw result is destroyed
    # before anything can summarise it, and the summariser becomes dead code.
    tool_raw_chars: int = field(default_factory=lambda: int(os.environ.get("TOOL_RAW_CHARS", "20000")))
    tool_result_chars: int = field(default_factory=lambda: int(os.environ.get("TOOL_RESULT_CHARS", "1500")))
    # Results longer than this get one cheap summarisation pass before they
    # enter the context. Worth it even at 15 tok/s because of the multiplier.
    summarise_tool_results: bool = field(default_factory=lambda: os.environ.get("SUMMARISE_TOOL_RESULTS", "1") == "1")
    summarise_over_chars: int = field(default_factory=lambda: int(os.environ.get("SUMMARISE_OVER_CHARS", "2500")))

    # ----------------------------------------------------------------- #
    # Structured logging (see obslog.py). All env-overridable.
    # ----------------------------------------------------------------- #
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))
    # json for aggregation (Loki/ELK/Datadog), text for local reading.
    log_format: str = field(default_factory=lambda: os.environ.get("LOG_FORMAT", "json"))
    # Override the log directory; empty uses ./logs under the project root.
    log_dir: str = field(default_factory=lambda: os.environ.get("LOG_DIR", "").strip())
    # How much chat content is written: disabled | metadata (length+fingerprint,
    # never the text) | full (redacted, truncated). Production default: metadata.
    log_chat_content: str = field(default_factory=lambda: os.environ.get("LOG_CHAT_CONTENT", "metadata"))
    log_max_bytes: int = field(default_factory=lambda: int(os.environ.get("LOG_MAX_BYTES", str(10 * 1024 * 1024))))
    log_backup_count: int = field(default_factory=lambda: int(os.environ.get("LOG_BACKUP_COUNT", "10")))
    log_retention_days: int = field(default_factory=lambda: int(os.environ.get("LOG_RETENTION_DAYS", "14")))

    # ----------------------------------------------------------------- #
    # Authentication and multi-user (see auth.py). OFF by default so a
    # single-user local install behaves exactly as before (a synthetic
    # 'local' admin owns everything). Turn on with AUTH_ENABLED=1.
    # ----------------------------------------------------------------- #
    auth_enabled: bool = field(default_factory=lambda: os.environ.get("AUTH_ENABLED", "0") == "1")
    auth_session_ttl_hours: int = field(default_factory=lambda: int(os.environ.get("AUTH_SESSION_TTL_HOURS", "168")))
    auth_cookie_name: str = field(default_factory=lambda: os.environ.get("AUTH_COOKIE_NAME", "llm_session"))
    # Set 1 when served over HTTPS (behind a reverse proxy) so the cookie is
    # marked Secure. Leave 0 for plain-HTTP local dev or the cookie won't be set.
    auth_cookie_secure: bool = field(default_factory=lambda: os.environ.get("AUTH_COOKIE_SECURE", "0") == "1")
    # First-run admin bootstrap. The password is used ONCE to create the admin
    # then must not persist; it is never stored in the DB in plaintext and is
    # redacted from any config dump. If unset and no admin exists, the app prints
    # a one-time generated password to the log at startup.
    admin_username: str = field(default_factory=lambda: os.environ.get("AUTH_ADMIN_USERNAME", "admin").strip())
    admin_password: str = field(default_factory=lambda: os.environ.get("AUTH_ADMIN_PASSWORD", ""))
    # A convenience non-admin account for dev/testing. NOT a backdoor: it exists
    # only when explicitly enabled and with an explicit password, and the README
    # documents removing it for production.
    allow_test_user: bool = field(default_factory=lambda: os.environ.get("AUTH_ALLOW_TEST_USER", "0") == "1")
    test_username: str = field(default_factory=lambda: os.environ.get("AUTH_TEST_USERNAME", "test").strip())
    test_password: str = field(default_factory=lambda: os.environ.get("AUTH_TEST_PASSWORD", ""))

    # ----------------------------------------------------------------- #
    # Microsoft Entra ID / OpenID Connect (authorization-code flow).
    # ----------------------------------------------------------------- #
    oidc_enabled: bool = field(default_factory=lambda: os.environ.get("OIDC_ENABLED", "0") == "1")
    oidc_tenant_id: str = field(default_factory=lambda: os.environ.get("OIDC_TENANT_ID", "").strip())
    oidc_client_id: str = field(default_factory=lambda: os.environ.get("OIDC_CLIENT_ID", "").strip())
    oidc_client_secret: str = field(default_factory=lambda: os.environ.get("OIDC_CLIENT_SECRET", ""))
    oidc_redirect_uri: str = field(default_factory=lambda: os.environ.get("OIDC_REDIRECT_URI", "").strip())
    # Defaults to https://login.microsoftonline.com/{tenant}/v2.0 when empty.
    oidc_authority: str = field(default_factory=lambda: os.environ.get("OIDC_AUTHORITY", "").strip())
    oidc_scopes: str = field(default_factory=lambda: os.environ.get("OIDC_SCOPES", "openid profile email").strip())
    # Role mapping. Any of these that matches an incoming token grants admin.
    oidc_admin_emails: str = field(default_factory=lambda: os.environ.get("OIDC_ADMIN_EMAILS", "").strip())
    oidc_admin_groups: str = field(default_factory=lambda: os.environ.get("OIDC_ADMIN_GROUPS", "").strip())
    oidc_admin_roles: str = field(default_factory=lambda: os.environ.get("OIDC_ADMIN_ROLES", "").strip())
    oidc_default_role: str = field(default_factory=lambda: os.environ.get("OIDC_DEFAULT_ROLE", "user").strip())

    # ----------------------------------------------------------------- #
    # Office 365 / Microsoft Graph (agent "office365" capability). These are the
    # connection settings; with them unset the capability's tools return a clear
    # "not configured" message instead of calling Graph, so the framework ships
    # now and can be connected later by supplying an Azure AD app.
    # ----------------------------------------------------------------- #
    o365_tenant_id: str = field(default_factory=lambda: os.environ.get("O365_TENANT_ID", "").strip())
    o365_client_id: str = field(default_factory=lambda: os.environ.get("O365_CLIENT_ID", "").strip())
    o365_client_secret: str = field(default_factory=lambda: os.environ.get("O365_CLIENT_SECRET", ""))
    o365_scopes: str = field(default_factory=lambda: os.environ.get(
        "O365_SCOPES", "https://graph.microsoft.com/.default").strip())

    # ----------------------------------------------------------------- #
    # Mac Mini (primary) / Mac Studio (secondary) cluster + routing
    # (see cluster.py). Single-node by default: with no STUDIO_NODE_URL the
    # router has one node (the local model server) and behaves as before.
    # ----------------------------------------------------------------- #
    node_role: str = field(default_factory=lambda: os.environ.get("NODE_ROLE", "primary").strip().lower())
    node_name: str = field(default_factory=lambda: os.environ.get("NODE_NAME", "").strip())
    # The Studio's OpenAI-compatible generation base URL, e.g.
    # http://studio.local:8080 . Empty disables the secondary node entirely.
    studio_node_url: str = field(default_factory=lambda: os.environ.get("STUDIO_NODE_URL", "").strip())
    # A secondary advertises where its router/primary is (used for heartbeats).
    # Shared secret for inter-node calls (sent as a bearer token). SECRET.
    node_token: str = field(default_factory=lambda: os.environ.get("NODE_TOKEN", ""))
    # Configurable routing factors. No arbitrary hard-coded thresholds.
    route_max_active_per_node: int = field(default_factory=lambda: int(os.environ.get("ROUTE_MAX_ACTIVE", "2")))
    route_queue_depth: int = field(default_factory=lambda: int(os.environ.get("ROUTE_QUEUE_DEPTH", "4")))
    route_cpu_pct: float = field(default_factory=lambda: float(os.environ.get("ROUTE_CPU_PCT", "85")))
    # Load average per core at which a node counts as saturated. This is a RATIO
    # (1.0 == fully committed), not a percentage: a healthy Mac routinely sits
    # above 1.0, so the bar is deliberately well clear of normal operation.
    route_load_ratio: float = field(default_factory=lambda: float(os.environ.get("ROUTE_LOAD_RATIO", "4")))
    route_mem_pct: float = field(default_factory=lambda: float(os.environ.get("ROUTE_MEM_PCT", "85")))
    # If the primary's recent latency exceeds this SLA (ms), eligible work spills
    # to the Studio. 0 disables the SLA factor.
    route_sla_ms: int = field(default_factory=lambda: int(os.environ.get("ROUTE_SLA_MS", "0")))
    # Circuit breaker: after a node trips to UNAVAILABLE it is skipped for this
    # long, then allowed a single half-open trial request. A success closes the
    # breaker (back to healthy); a failure re-opens it for another cooldown. This
    # lets a briefly-flapping Studio recover between heartbeats without hammering
    # a genuinely-down node on every request.
    route_cooldown_s: float = field(default_factory=lambda: float(os.environ.get("ROUTE_COOLDOWN_S", "20")))
    # Substrings that mark a model as "large" (needs the high-memory Studio).
    large_model_markers: str = field(default_factory=lambda: os.environ.get(
        "LARGE_MODEL_MARKERS", "14B,32B,70B,72B").strip())
    heartbeat_interval: float = field(default_factory=lambda: float(os.environ.get("HEARTBEAT_INTERVAL", "10")))
    heartbeat_timeout: float = field(default_factory=lambda: float(os.environ.get("HEARTBEAT_TIMEOUT", "30")))
    # Timeout for a node health probe / heartbeat request.
    node_probe_timeout: float = field(default_factory=lambda: float(os.environ.get("NODE_PROBE_TIMEOUT", "8")))

    # ----------------------------------------------------------------- #
    # Claude-history ZIP import limits (see claude_import.py). The upload is
    # untrusted: these bound the damage a hostile archive can do.
    # ----------------------------------------------------------------- #
    import_max_zip_bytes: int = field(default_factory=lambda: int(os.environ.get("IMPORT_MAX_ZIP_BYTES", str(200 * 1024 * 1024))))
    import_max_files: int = field(default_factory=lambda: int(os.environ.get("IMPORT_MAX_FILES", "20000")))
    import_max_uncompressed_bytes: int = field(default_factory=lambda: int(os.environ.get("IMPORT_MAX_UNCOMPRESSED_BYTES", str(1024 * 1024 * 1024))))
    import_max_file_bytes: int = field(default_factory=lambda: int(os.environ.get("IMPORT_MAX_FILE_BYTES", str(50 * 1024 * 1024))))

    # ----------------------------------------------------------------- #
    # Feature settings. Each feature adds its own commented block of fields
    # ABOVE the marker line below (never edit another feature's block). Keep
    # defaults scalar (bool/int/float/str): the /api/config schema is generated
    # from MUTABLE by default-value type and skips anything else. Structured
    # values are JSON strings.
    # ----------------------------------------------------------------- #
    # --- Admission control (item 6, admission.py) --------------------------- #
    # max_concurrent_generations / generation_queue_depth (above) stay the slot
    # count and the waiting-room MULTIPLIER (max waiting = slots x depth); these
    # two add the memory half. With memory awareness on, a request that would
    # leave less than admission_min_free_mb of free memory (after the KV
    # reserved by everything already running, plus its own) waits for a running
    # generation to finish instead of starting; a request is still always
    # admitted when nothing else is running. Free memory is read from psutil or
    # vm_stat; when neither works the check switches itself off. The floor
    # scales with RAM because the backend's own working set (prompt cache,
    # Metal buffer cache) does: 384MB on 8GB, 768MB up to 24GB, 1.5GB above.
    # Off by default so an upgrade admits exactly what the slot gate did; turn
    # it on (env or Settings) once the floor suits the machine.
    admission_memory_aware: bool = field(default_factory=lambda: os.environ.get(
        "ADMISSION_MEMORY_AWARE", "0") == "1")
    admission_min_free_mb: int = field(default_factory=lambda: int(
        os.environ.get("ADMISSION_MIN_FREE_MB")
        or (384 if TOTAL_RAM_GB < 14 else 768 if TOTAL_RAM_GB <= 24 else 1536)))
    # --- Inference backend (item 1, backends.py) --------------------------- #
    # Which engine serves the model. "mlx-lm" is today's mlx_lm.server command
    # line, unchanged; "mlx-lm-turbo" is the same server with batching, prompt
    # cache and speculative decoding sized from RAM and concurrency;
    # "openai-external" launches BACKEND_COMMAND. Restart-only: the engine is a
    # process, and the UI cannot swap it under a running generation.
    model_backend: str = field(default_factory=lambda: os.environ.get(
        "MODEL_BACKEND", "mlx-lm").strip().lower())
    # Speculative decoding: a small model of the same family proposes tokens
    # the served model verifies. "" is off; "auto" asks the model catalogue.
    # On mlx-lm 0.32 a loaded draft model turns continuous batching off.
    draft_model: str = field(default_factory=lambda: os.environ.get("DRAFT_MODEL", "").strip())
    # Tokens the draft proposes per step. 0 keeps the server's default (3).
    num_draft_tokens: int = field(default_factory=lambda: int(
        os.environ.get("NUM_DRAFT_TOKENS", "0")))
    # openai-external only: the start command, with {python} {model} {port}
    # {host} {adapter} {draft_model} placeholders, and the comma-separated
    # capability names the engine has (backends.BACKEND_CAPABILITIES).
    backend_command: str = field(default_factory=lambda: os.environ.get("BACKEND_COMMAND", ""))
    backend_capabilities: str = field(default_factory=lambda: os.environ.get(
        "BACKEND_CAPABILITIES", ""))
    # Appended verbatim (shell-split) to any backend's command line: an escape
    # hatch for a flag this app does not model yet.
    backend_extra_args: str = field(default_factory=lambda: os.environ.get(
        "BACKEND_EXTRA_ARGS", ""))
    # --- Cluster, sharding, remote access (items 4, 15, 20: cluster.py, ------ #
    # sharding.py, nodesec.py, remote_access.py). All restart-only: they shape
    # the node set, the listening socket or the peer clients, which are built
    # once at startup.
    # Peer nodes: a JSON list ([{"name": "studio", "url": "https://...",
    # "ram_gb": 64}] or ["https://..."]) or comma-separated URLs. Merged with
    # STUDIO_NODE_URL (the one-node shorthand). Empty: nodes paired with
    # `deploy.py cluster join` (DATA_DIR/cluster_nodes.json) are used instead.
    cluster_nodes: str = field(default_factory=lambda: os.environ.get("CLUSTER_NODES", "").strip())
    # mlx.launch hostfile for MODEL_BACKEND=mlx-distributed (layer-sharded
    # serving). Empty means DATA_DIR/cluster_hostfile.json.
    cluster_hostfile: str = field(default_factory=lambda: os.environ.get("CLUSTER_HOSTFILE", "").strip())
    # Interface the web server binds. Anything but loopback is refused unless
    # AUTH_ENABLED=1 (and warned about unless AUTH_COOKIE_SECURE=1).
    web_host: str = field(default_factory=lambda: os.environ.get("WEB_HOST", "127.0.0.1").strip())
    # Reverse proxies whose X-Forwarded-For is believed (comma IPs/CIDRs).
    # Empty: the socket peer is the client, exactly as before.
    trusted_proxies: str = field(default_factory=lambda: os.environ.get("TRUSTED_PROXIES", "").strip())
    # The externally visible origin (https://llm.example.com) behind a proxy or
    # tunnel; the OIDC redirect URI is derived from it.
    public_base_url: str = field(default_factory=lambda: os.environ.get("PUBLIC_BASE_URL", "").strip())
    # TLS for peer node URLs: a private CA bundle and an optional client cert.
    node_ca_file: str = field(default_factory=lambda: os.environ.get("NODE_CA_FILE", "").strip())
    node_client_cert: str = field(default_factory=lambda: os.environ.get("NODE_CLIENT_CERT", "").strip())
    node_client_key: str = field(default_factory=lambda: os.environ.get("NODE_CLIENT_KEY", "").strip())
    # Plain http:// to a non-private, non-tailnet peer is refused unless this is 1.
    node_allow_plaintext: bool = field(default_factory=lambda: os.environ.get("NODE_ALLOW_PLAINTEXT", "0") == "1")
    # Optional end-to-end sealing key for node traffic (32 bytes, hex or
    # base64; needs the optional `cryptography` package). SECRET.
    node_e2e_key: str = field(default_factory=lambda: os.environ.get("NODE_E2E_KEY", "").strip())
    # Tool-call parsing (toolparse.py). "auto" accepts every model family's
    # format (Hermes <tool_call>, Qwen3-Coder <function=...>, Llama 3.x,
    # Mistral [TOOL_CALLS], pythonic, Gemma, DeepSeek, Kimi, GLM, gpt-oss);
    # a comma list narrows it. The canonical {"tool","args"} JSON is always on.
    tool_call_format: str = field(default_factory=lambda: os.environ.get("TOOL_CALL_FORMAT", "auto").strip())
    # Recover broken structured output (4-bit models drop quotes, leave
    # trailing commas, stop before the closing brace). Every repair is
    # recorded on the parsed call; tool names are never invented.
    tool_call_repair: bool = field(default_factory=lambda: os.environ.get("TOOL_CALL_REPAIR", "1") == "1")
    # Model catalogue (catalog.py). MODEL_ALIASES is a JSON object of
    # {"alias": "repo/id"} merged over the built-in aliases (fast, balanced,
    # coder, reasoning, vision, tiny, large). model_alias is informational: the
    # alias --model / MODEL_ID named, while `model` always holds the resolved id.
    model_aliases: str = field(default_factory=lambda: os.environ.get("MODEL_ALIASES", "").strip())
    model_alias: str = ""
    # Benchmark recommendation (deploy.py recommend): the slowest decode speed a
    # recommended model may have, and RAM kept free beyond its measured peak
    # (0 = automatic: the larger of 2.5GB and a quarter of RAM).
    bench_min_decode_tps: float = field(default_factory=lambda: float(
        os.environ.get("BENCH_MIN_DECODE_TPS", "10")))
    bench_headroom_gb: float = field(default_factory=lambda: float(
        os.environ.get("BENCH_HEADROOM_GB", "0")))
    # Training methods and autopilot (train_methods.py). sft is the original
    # mlx_lm.lora loop, unchanged; dpo/orpo/grpo run through the optional
    # mlx-lm-lora package and are refused (never silently SFT) without it; kto
    # is export-only (no installed trainer implements it).
    train_method: str = field(default_factory=lambda: os.environ.get("TRAIN_METHOD", "sft").strip().lower())
    # 1: derive layers, rank, batch, sequence window and gradient checkpointing
    # from RAM, the model's config.json and the corpus, and refuse a run that
    # will not fit. 0 keeps the configured recipe exactly.
    train_autopilot: bool = field(default_factory=lambda: os.environ.get("TRAIN_AUTOPILOT", "0") == "1")
    # Preference strength for DPO/ORPO/GRPO (mlx_lm_lora --beta; its default 0.1).
    train_beta: float = field(default_factory=lambda: float(os.environ.get("TRAIN_BETA", "0.1")))
    # Reference model for DPO/GRPO. Empty: the base model. A path must be a full
    # model directory (fuse an adapter first); mlx_lm_lora loads it without one.
    train_reference_adapter: str = field(default_factory=lambda: os.environ.get("TRAIN_REFERENCE_ADAPTER", "").strip())
    # GRPO reward terms and weights (experimental): tool_call, format, length.
    train_grpo_rewards: str = field(default_factory=lambda: os.environ.get("TRAIN_GRPO_REWARDS", "tool_call:1.0,format:0.5,length:0.25").strip())
    # Promote a preference run that produced no comparable metric (GRPO always,
    # DPO/ORPO without validation points). Off: such runs are rolled back.
    train_promote_ungated: bool = field(default_factory=lambda: os.environ.get("TRAIN_PROMOTE_UNGATED", "0") == "1")
    # --- Chat turns (streams.py) ---
    # Seconds a streaming chat turn keeps running after its client disconnects,
    # so a page reload or a dropped connection can resume it (GET
    # /api/chat/turns/{id}/events) instead of losing the work in flight. 0
    # cancels on disconnect, as before turns were resumable.
    turn_detach_grace_s: float = field(default_factory=lambda: float(os.environ.get("TURN_DETACH_GRACE_S", "30")))
    # Workspaces and projects (workspaces.py). 1: a non-admin's file tools work
    # in their own WORKSPACE_DIR/users/<id> (created on first use) and a turn
    # can be confined to a project root; admins keep PROJECT_DIR. 0 keeps the
    # single installation root everyone shared before.
    workspaces_enabled: bool = field(default_factory=lambda: os.environ.get("WORKSPACES_ENABLED", "0") == "1")
    # Parent directories an admin may put a shared project's root under (comma
    # list). Empty: the workspace directory and PROJECT_DIR. With workspaces on
    # it also confines the admin folder browser (/api/browse).
    workspace_roots: str = field(default_factory=lambda: os.environ.get("WORKSPACE_ROOTS", "").strip())
    # Tool approval policy (policy.py). off = every call runs as before (only
    # explicit override rows apply); risky = write/exec/external calls wait for
    # the user's approval; all = every call does.
    tool_approval_mode: str = field(default_factory=lambda: os.environ.get(
        "TOOL_APPROVAL_MODE", "off").strip().lower())
    # What "ask" becomes when nobody can answer (scheduled task, API call):
    # deny (default) or allow.
    tool_approval_unattended: str = field(default_factory=lambda: os.environ.get(
        "TOOL_APPROVAL_UNATTENDED", "deny").strip().lower())
    # Seconds an approval prompt waits before it counts as a refusal.
    tool_approval_timeout_s: int = field(default_factory=lambda: int(
        os.environ.get("TOOL_APPROVAL_TIMEOUT_S", "120")))
    # Prompt-injection defence: after a turn read web/MCP/mail content, its
    # write/exec/external calls need a fresh approval (when the mode is not off).
    tool_approval_taint: bool = field(default_factory=lambda: os.environ.get(
        "TOOL_APPROVAL_TAINT", "1") == "1")
    # Per-role rules, JSON: {"user": {"risk:exec": "deny"}, "admin": {"*": "allow"}}.
    tool_approval_roles: str = field(default_factory=lambda: os.environ.get(
        "TOOL_APPROVAL_ROLES", "").strip())
    # Per-user approvals (policy.py, the header's Approvals control): the mode
    # a user gets until they choose one. all = ask before every tool call, risky
    # = ask before writes, code, sends and deletes, off = never ask. The rule
    # this implements is "always ask for every action unless the user changes
    # the setting the header always shows", so the default is all.
    # TOOL_APPROVAL_MODE above stays the administrator's floor: when it is
    # stricter than a user's choice, it wins.
    approval_default_mode: str = field(default_factory=lambda: os.environ.get(
        "APPROVAL_DEFAULT_MODE", "all").strip().lower())
    # OpenAI / Anthropic compatible API (openai_compat.py, anthropic_compat.py).
    # What a /v1 request whose model is a plain alias or repo id runs as:
    # "agent" (this app's agent with its server-side tools, today's
    # /v1/chat/completions behaviour) or "raw" (the model with the client's own
    # tools, for coding clients that run their own tool loop). "raw/<alias>"
    # and "agent/<profile>" model names choose per request either way.
    v1_default_mode: str = field(default_factory=lambda: os.environ.get(
        "V1_DEFAULT_MODE", "agent").strip().lower())
    # 0 refuses raw mode entirely (every /v1 request then runs the agent).
    v1_raw_enabled: bool = field(default_factory=lambda: os.environ.get(
        "V1_RAW_ENABLED", "1") == "1")
    # API tokens and CLI sign-in (api_tokens.py, auth.py device flow).
    # Lifetime of a new API token in days, and the longest one a request may
    # ask for (a request can shorten it, never extend it); 0 = no expiry.
    api_token_default_ttl_days: int = field(default_factory=lambda: int(
        os.environ.get("API_TOKEN_DEFAULT_TTL_DAYS", "90")))
    # Entra app registration with "Allow public client flows" for terminal
    # sign-in (OIDC device code). Empty = device sign-in off. A public client
    # id is not a secret; it may equal OIDC_CLIENT_ID when that app allows it.
    oidc_cli_client_id: str = field(default_factory=lambda: os.environ.get(
        "OIDC_CLI_CLIENT_ID", "").strip())
    # MCP server endpoint and the gated tool set (mcp_server.py, tools_extra.py).
    # Capability keys per role, JSON: {"user": ["web_search", "file_ops"],
    # "admin": ["*"]}. A role that is not listed keeps every capability, so the
    # empty default is today's rule (the profile alone decides). Narrows agent
    # profiles in chat and the tools this app offers as an MCP server.
    role_capabilities: str = field(default_factory=lambda: os.environ.get(
        "ROLE_CAPABILITIES", "").strip())
    # Serve this app's tools to MCP clients (Claude Code, IDEs) at POST /mcp,
    # behind the same sign-in. Off by default: it is a new way in.
    mcp_server_enabled: bool = field(default_factory=lambda: os.environ.get(
        "MCP_SERVER_ENABLED", "0") == "1")
    # Which of the newer tools (browse_page, git, http_request, ...) the default
    # agent registers when it has no allowlist: "" none (today's prompt), "all",
    # or a comma list. A profile that grants a capability gets its tools anyway.
    extra_tools: str = field(default_factory=lambda: os.environ.get(
        "EXTRA_TOOLS", "").strip())
    # Read-only databases for the database tool, JSON {"name": "/path/to.db"}
    # (SQLite). SQLite files inside the acting root are always reachable.
    database_urls: str = field(default_factory=lambda: os.environ.get(
        "DATABASE_URLS", "").strip())
    # notify tool channels: a webhook (ntfy / Slack-compatible JSON POST; the
    # URL usually embeds a token, so it is a secret) and a macOS notification
    # when the server runs on this Mac's own console session.
    notify_webhook_url: str = field(default_factory=lambda: os.environ.get(
        "NOTIFY_WEBHOOK_URL", "").strip())
    notify_macos: bool = field(default_factory=lambda: os.environ.get(
        "NOTIFY_MACOS", "1") == "1")
    # image_generate: a command template run without a shell, with {prompt},
    # {output}, {width}, {height}, {steps} filled per argument. Empty uses
    # mflux-generate when it is installed; otherwise the tool says it is
    # unavailable.
    image_gen_command: str = field(default_factory=lambda: os.environ.get(
        "IMAGE_GEN_COMMAND", "").strip())
    image_gen_timeout_s: int = field(default_factory=lambda: int(
        os.environ.get("IMAGE_GEN_TIMEOUT_S", "600")))
    # Mailbox the email/calendar/contacts tools fall back to when the acting
    # user has no address of their own (auth off, or an admin). A signed-in
    # user only ever reaches their own mailbox.
    o365_user: str = field(default_factory=lambda: os.environ.get("O365_USER", "").strip())
    # Multimodal input (attachments.py, audio.py, vision.py). Uploads a chat
    # turn can attach: images, audio (transcribed) and documents (read as text).
    attachments_enabled: bool = field(default_factory=lambda: os.environ.get(
        "ATTACHMENTS_ENABLED", "1") == "1")
    # Largest single upload, in bytes (also caps /v1/audio/transcriptions).
    attachment_max_bytes: int = field(default_factory=lambda: int(
        os.environ.get("ATTACHMENT_MAX_BYTES", str(20 * 1024 * 1024))))
    # Total stored per user, in MB; 0 = unlimited.
    attachment_quota_mb: int = field(default_factory=lambda: int(
        os.environ.get("ATTACHMENT_QUOTA_MB", "1024")))
    # Where uploads are stored. Empty: DATA_DIR/attachments (next to the database).
    attachments_dir: str = field(default_factory=lambda: os.environ.get(
        "ATTACHMENTS_DIR", "").strip())
    # Can the served model see images? auto asks the engine and the model
    # (vision.served_vision); on/off override it for setups auto cannot see.
    vision_mode: str = field(default_factory=lambda: os.environ.get(
        "VISION_MODE", "auto").strip().lower())
    # Speech-to-text model (mlx-whisper) and text-to-speech model and voice
    # (mlx-audio). Loaded on first use only; never downloaded unless used.
    whisper_model: str = field(default_factory=lambda: os.environ.get(
        "WHISPER_MODEL", "mlx-community/whisper-tiny").strip())
    tts_model: str = field(default_factory=lambda: os.environ.get(
        "TTS_MODEL", "mlx-community/Kokoro-82M-bf16").strip())
    tts_voice: str = field(default_factory=lambda: os.environ.get(
        "TTS_VOICE", "af_heart").strip())
    # Cloud fallback for oversized prefill (cloud.py). Off by default, which is
    # today's behaviour: every prompt stays on this machine. On, a prompt the
    # local model cannot prefill well (larger than the context window even after
    # trimming, slower than CLOUD_MIN_PREFILL_S at the measured prompt speed, or
    # at least CLOUD_MIN_PROMPT_TOKENS) may go to a cloud model, but only for an
    # allowed role and only after that user's consent.
    cloud_fallback_enabled: bool = field(default_factory=lambda: os.environ.get(
        "CLOUD_FALLBACK_ENABLED", "0") == "1")
    # openai: any OpenAI-compatible /chat/completions endpoint; anthropic: /v1/messages.
    cloud_provider: str = field(default_factory=lambda: os.environ.get(
        "CLOUD_PROVIDER", "openai").strip().lower())
    # Empty: the provider's public endpoint (https://api.openai.com/v1 or
    # https://api.anthropic.com). Restart-only, like the key: it is where
    # user content goes.
    cloud_base_url: str = field(default_factory=lambda: os.environ.get(
        "CLOUD_BASE_URL", "").strip())
    cloud_model: str = field(default_factory=lambda: os.environ.get(
        "CLOUD_MODEL", "").strip())
    cloud_api_key: str = field(default_factory=lambda: os.environ.get(
        "CLOUD_API_KEY", "").strip())
    # per_request asks on every turn that would go to the cloud; per_user also
    # offers "this conversation" and "always", which unattended runs may use.
    cloud_consent: str = field(default_factory=lambda: os.environ.get(
        "CLOUD_CONSENT", "per_request").strip().lower())
    # Offer the cloud when the estimated local prefill takes at least this many
    # seconds at the measured prompt speed; 0 turns this trigger off.
    cloud_min_prefill_s: float = field(default_factory=lambda: float(
        os.environ.get("CLOUD_MIN_PREFILL_S", "120")))
    # Offer the cloud for any prompt of at least this many tokens; 0 = off.
    cloud_min_prompt_tokens: int = field(default_factory=lambda: int(
        os.environ.get("CLOUD_MIN_PROMPT_TOKENS", "0")))
    # Who may be offered the cloud: role names and user:<id or username>
    # entries, comma separated; * is everyone.
    cloud_allowed_roles: str = field(default_factory=lambda: os.environ.get(
        "CLOUD_ALLOWED_ROLES", "admin").strip())
    # Cloud tokens (prompt + reply) one user may spend per calendar month (UTC);
    # 0 = no cap.
    cloud_max_tokens_per_user_month: int = field(default_factory=lambda: int(
        os.environ.get("CLOUD_MAX_TOKENS_PER_USER_MONTH", "0")))
    # Evaluation and regression suite (evals.py). EVAL_ON_PROMOTE=1 runs the
    # live tool-calling, RAG and multi-turn suites after a retrain promoted an
    # adapter and restarted the server, and rolls the adapter back when any
    # suite scores more than EVAL_MAX_REGRESSION (a 0..1 fraction) below the
    # previous adapter's baseline. Off by default: training is unchanged.
    eval_on_promote: bool = field(default_factory=lambda: os.environ.get(
        "EVAL_ON_PROMOTE", "0") == "1")
    eval_max_regression: float = field(default_factory=lambda: float(
        os.environ.get("EVAL_MAX_REGRESSION", "0.05")))
    # File-tool safety and generated documents (tools.py, Phase B of
    # upgrade/ROUTING_AND_TOOLING.md). PROJECT_DIR may be this app's own
    # checkout, where read_file could read .env and write_file could rewrite
    # data/feedback.db under the running server. FILE_DENY_GLOBS is the list of
    # paths, relative to the active root, that no file tool, @reference,
    # download or create_document may read or write: gitignore-style, a
    # pattern with a slash is anchored at the root, one without matches at any
    # depth, and a matching directory protects everything below it. Matched
    # case-insensitively (APFS would otherwise open .ENV as .env). Restart-only:
    # a web setting that can lift the guard is a guard one stolen admin session
    # removes. OUTPUTS_DIR is where create_document writes a bare file name
    # (relative to the root, created on demand; empty keeps the root), so a
    # generated report never lands among the source files.
    file_deny_globs: str = field(default_factory=lambda: os.environ.get(
        "FILE_DENY_GLOBS", ".env,.env.*,data/**,logs/**,adapters/**,.venv/**,*.db,"
        "*.db-wal,*.db-shm,*.key,*.pem,secrets/**,.git/**").strip())
    outputs_dir: str = field(default_factory=lambda: os.environ.get(
        "OUTPUTS_DIR", "outputs").strip())
    # Agent routing, retrieval relevance and reasoning budget (agent.py, llm.py;
    # upgrade/ROUTING_AND_TOOLING.md P2 and P6).
    # A knowledge-base query word found in more than this fraction of the
    # indexed documents says nothing about which one is relevant ("skills" was
    # in 3467 of 3473 paths) and is ignored by the retrieval overlap test.
    # Applies once the index holds at least 20 documents; 1.0 turns it off.
    rag_df_ceiling: float = field(default_factory=lambda: float(
        os.environ.get("RAG_DF_CEILING", "0.2")))
    # When the "think before you answer" instruction is appended (under
    # REASONING_VISIBLE): auto = only for a model the catalogue marks as a
    # thinking model, always = every model (the old behaviour), never = none.
    reasoning_prompt: str = field(default_factory=lambda: os.environ.get(
        "REASONING_PROMPT", "auto").strip().lower())
    # Reasoning tokens one agent step may spend before the stream is stopped
    # and the model is asked for the answer only (once, then what it has is
    # salvaged). 0 = no cap.
    think_max_tokens: int = field(default_factory=lambda: int(
        os.environ.get("THINK_MAX_TOKENS", "384")))
    # UI-term intent lexicon (intents.py): read the app's own vocabulary ("add
    # this to skills", "check the codebase", "use the knowledge base to make a
    # pdf") as an instruction before the router runs, so those turns get the
    # right tools and retrieval. 0 leaves every turn to the router as before.
    intent_lexicon_enabled: bool = field(default_factory=lambda: os.environ.get(
        "INTENT_LEXICON_ENABLED", "1") == "1")
    # ---- extension fields: feature blocks above this line ---- #

    seed_demo: bool = False
    retrain_now: bool = False
    export_only: bool = False
    list_feedback: bool = False
    export_format: Literal["jsonl", "csv"] = "jsonl"

    # Fields whose value must never be returned by public() or written to a log.
    SECRET_FIELDS = (
        "admin_password", "test_password", "oidc_client_secret", "node_token",
        "o365_client_secret",
        "node_e2e_key",   # cluster: E2E sealing key for node traffic
        "notify_webhook_url",   # notify tool: webhook URLs carry their token
        "cloud_api_key",   # cloud fallback: the provider's API key
        # ---- extension secrets: above this line ---- #
    )
    # JSON settings shown by public() with their credential-like values masked
    # (see mask_structured_secrets), so GET /api/config and --print-config
    # never print an MCP bearer token or a server env secret.
    MASKED_STRUCTURED_FIELDS = ("mcp_servers", "database_urls")

    # Settings the web UI is allowed to change at runtime. Anything not listed
    # here needs a process restart and is rejected by /api/config.
    MUTABLE = (
        "system_prompt", "identity", "train_min_examples", "project_dir",
        # The training recipe, tunable live so a run can be corrected without a
        # restart. train_iters=0 derives the count from train_epochs.
        "train_iters", "train_epochs", "train_min_iters", "train_max_iters",
        "train_lr", "train_batch_size", "train_seq_len", "train_drop_over_length",
        "train_num_layers", "train_fine_tune_type",
        "train_lora_rank", "train_lora_scale", "train_lora_dropout",
        "train_on_tool_calls", "train_tool_examples", "train_tool_ratio",
        "train_tool_quality", "train_replay_ratio", "train_val_split",
        "train_val_check", "train_val_tolerance", "train_promote_best",
        "train_timeout", "train_max_backups", "auto_retrain_threshold",
        "rag_enabled", "rag_passages", "rag_scope",
        "skills_enabled", "skills_in_prompt", "skill_body_chars", "skills_max",
        "skills_autolearn", "skill_author_tokens", "skill_retire_min_rated",
        "skill_autoload", "skill_autoload_overlap",
        "checkpoints_enabled", "checkpoint_keep", "checkpoint_max_file_bytes",
        "delegation_enabled", "subagent_max_concurrent", "subagent_max_per_turn",
        "subagent_max_steps", "subagent_result_chars", "execute_code_max_calls",
        "context_files_enabled", "context_files_chars", "reference_expansion",
        "reference_max", "reference_chars", "recall_results", "recall_snippet_chars",
        "skill_retire_loss_rate",
        "max_tokens", "temperature", "repetition_penalty",
        "repetition_context_size", "repetition_penalty_enabled", "context_size",
        "history_turns", "agent_enabled", "agent_max_steps", "agent_min_steps",
        # search_backend is intentionally NOT mutable: the provider is locked to
        # DuckDuckGo Lite and cannot be changed from the UI or the API.
        "search_results", "tool_result_chars", "tool_raw_chars", "auto_fetch_results",
        "disable_thinking", "reasoning_visible", "tool_temperature", "fast_path", "stable_prefix", "knowledge_triage", "agent_escalation",
        "auto_continue", "auto_continue_max",
        "summarise_tool_results", "summarise_over_chars",
        # Safeguards, all tunable live so a machine can be dialled in without a
        # restart or an env edit.
        "incremental_reasoning", "reasoning_max_steps", "reasoning_step_timeout",
        "reasoning_signals", "reasoning_min_chars",
        "model_cutoff", "date_in_prompt", "hedge_escalation",
        "reasoning_tokens", "chunk_large_prompts", "chunk_trigger_ratio",
        "chunk_size_ratio", "auto_fetch_char_cap", "stall_timeout", "ready_wait_timeout",
        "decode_floor_tps", "max_generation_timeout",
        "resilient_retries", "min_max_tokens", "hard_step_cap",
        "show_internals", "retrieval_deadline", "exec_backend", "docker_image",
        "test_command", "auto_iterate_rounds", "agent_run_timeout", "code_max_tokens",
        "exec_env_passthrough", "exec_memory_mb", "exec_max_file_mb",
        "exec_pids_limit", "exec_guardrail", "test_timeout",
        # Multi-turn task continuity.
        "task_state_enabled", "task_artifact_chars", "drift_check_enabled",
        "artifact_reply_headroom", "artifact_patch_chars", "code_check_enabled",
        "code_check_deep", "debug_prompts",
        # Logging: enabling DEBUG/TRACE and disabling content logs at runtime.
        "log_level", "log_format", "log_chat_content",
        # Routing factors, tunable live so a two-Mac cluster can be dialled in.
        # (studio_node_url is NOT here: adding/removing a node is a topology
        # change that rebuilds the registry, so it needs a restart.)
        "route_max_active_per_node", "route_queue_depth", "route_cpu_pct",
        "route_mem_pct", "route_load_ratio", "route_sla_ms", "route_cooldown_s",
        "large_model_markers",
        "heartbeat_interval", "heartbeat_timeout",
        # Admission control: the slot count and queue multiplier resize the
        # live controller (POST /api/config calls resize), the memory floor and
        # switch are re-read by it immediately.
        "max_concurrent_generations", "generation_queue_depth",
        "admission_memory_aware", "admission_min_free_mb",
        # Tool-call parsing: read on every parse, so a change applies to the
        # next reply.
        "tool_call_format", "tool_call_repair",
        # Model catalogue: aliases resolve on the next switch; the bench
        # thresholds apply to the next recommendation.
        "model_aliases", "bench_min_decode_tps", "bench_headroom_gb",
        # Training method and autopilot: read when a run starts.
        "train_method", "train_autopilot", "train_beta", "train_reference_adapter",
        "train_grpo_rewards", "train_promote_ungated",
        # Chat turns: read when a client disconnects, so it applies to the
        # next detach.
        "turn_detach_grace_s",
        # Workspaces: read on every tool call (the root resolves per context),
        # so a change applies to the next call.
        "workspaces_enabled", "workspace_roots",
        # Tool approval policy: read on every tool call, so a change applies
        # to the next call.
        "tool_approval_mode", "tool_approval_unattended", "tool_approval_timeout_s",
        "tool_approval_taint", "tool_approval_roles",
        # Per-user approvals: the default for users who never chose a mode,
        # read on every tool call, so a change applies to the next call.
        "approval_default_mode",
        # OpenAI / Anthropic compatible API: read per request, so a change
        # applies to the next /v1 call.
        "v1_default_mode", "v1_raw_enabled",
        # MCP server endpoint and role capabilities: read per request.
        "mcp_server_enabled", "role_capabilities",
        # Multimodal input: limits are read per upload, the vision mode per
        # turn, the speech models on their next load.
        "attachments_enabled", "attachment_max_bytes", "attachment_quota_mb",
        "vision_mode", "whisper_model", "tts_model", "tts_voice",
        # Cloud fallback: read on every model call, so a change applies to the
        # next one. Provider, endpoint, model and key stay restart-only.
        "cloud_fallback_enabled", "cloud_consent", "cloud_min_prefill_s",
        "cloud_min_prompt_tokens", "cloud_allowed_roles", "cloud_max_tokens_per_user_month",
        # Evaluation suite: read when a retrain promotes and when a run is
        # compared, so a change applies to the next one.
        "eval_on_promote", "eval_max_regression",
        # Agent routing, retrieval relevance and reasoning budget: read per
        # turn, so a change applies to the next message.
        "rag_df_ceiling", "reasoning_prompt", "think_max_tokens",
        # Intent lexicon: read at the start of every turn, so a change applies
        # to the next message.
        "intent_lexicon_enabled",
        # ---- extension mutable settings: one commented group per feature, above this line ---- #
    )

    # The only supported search provider. DuckDuckGo Lite, exclusively.
    SEARCH_BACKENDS = ("duckduckgo_lite",)

    @property
    def system_prompt_with_identity(self) -> str:
        """The system prompt with the identity line prepended, if set.

        Today's date goes here too, which is the one place every lane reads --
        the tool prompt, the prose prompt, the plain lane in the API and the
        reasoning synthesis all come through this property, so putting it
        anywhere else would mean four copies and three of them going stale.

        The date had never been in any prompt at all, and its absence was not
        cosmetic: with no anchor the model cannot tell a question about this
        year from a question about a settled one, cannot judge whether its own
        knowledge is current, and answers "is that still true" out of weights
        that stopped years ago with no sign that anything is missing. Stating
        the cutoff alongside it turns "I might be out of date" from something
        the model has to guess at into something it can read.

        It changes once a day, so the cached prompt prefix survives a session
        (see stable_prefix); it is not a per-turn value.
        """
        parts = [self.training_system_prompt]
        if self.date_in_prompt:
            today = datetime.date.today().isoformat()
            line = f"Today's date is {today}."
            if self.model_cutoff:
                line += (f" Your training data ends around {self.model_cutoff}. "
                         "For anything that may have changed since then, say so "
                         "plainly or look it up rather than answering from memory.")
            # After the identity, before the instructions: the identity is the
            # most stable thing in the prompt and belongs at the front of the
            # cached prefix.
            parts.insert(1 if self.identity else 0, line)
        return "\n\n".join(p for p in parts if p)

    @property
    def training_system_prompt(self) -> str:
        """The system prompt WITHOUT today's date, for anything that persists.

        Training examples are conditioned on the system prompt they were
        exported with, and the adapter records a hash of it so a prompt change
        after the fact is caught. Today's date is in the served prompt on
        purpose and must stay out of both: hashed, it changes the fingerprint
        every midnight and makes the "the system prompt has changed" warning
        fire on essentially every run started on a different day; exported, it
        conditions every example on a date that is already wrong by the time
        the adapter is used, and teaches the model that date.
        """
        if self.identity:
            return f"{self.identity}\n\n{self.system_prompt}"
        return self.system_prompt

    def settings_path(self) -> Path:
        """Where settings changed from the UI are stored."""
        return DATA_DIR / "settings.json"

    def load_saved(self) -> list[str]:
        """Apply settings previously saved from the UI. Returns what was applied.

        Without this, "Save" in Settings only held until the process ended:
        every mutable field went back to its environment default on the next
        start, so a user who raised Max tokens to 4000, restarted for an
        unrelated reason, and saw 512 again had no way to tell a silent revert
        from a UI bug. Saved values deliberately win over environment defaults,
        because the UI change is the later and more explicit act; the caller logs
        what was restored so the precedence is visible rather than mysterious.
        """
        path = self.settings_path()
        try:
            raw = json.loads(path.read_text())
        except FileNotFoundError:
            return []
        except Exception as exc:
            log(f"Ignoring unreadable {path} ({exc}); using environment defaults. "
                "Delete the file to clear saved settings.", logging.WARNING)
            return []
        if not isinstance(raw, dict):
            return []
        # Only mutable, non-secret fields, so a stale file can never resurrect a
        # credential or a setting that needs a restart to take effect anyway.
        wanted = {key: value for key, value in raw.items()
                  if key in self.MUTABLE and key not in self.SECRET_FIELDS}
        return self.apply(wanted) if wanted else []

    def save_settings(self, fields: list[str]) -> list[str]:
        """Merge these fields into the saved settings file. Returns what was written.

        Called with whatever apply() reported as CHANGED, so a value the
        guardrails rejected or clamped is never written back at face value.
        """
        keep = [name for name in fields
                if name in self.MUTABLE and name not in self.SECRET_FIELDS]
        if not keep:
            return []
        path = self.settings_path()
        try:
            existing = json.loads(path.read_text())
            if not isinstance(existing, dict):
                existing = {}
        except Exception:
            existing = {}
        existing.update({name: getattr(self, name) for name in keep})
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename: a crash mid-write leaves the old file intact
            # rather than a truncated one that reverts every setting.
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(existing, indent=2, sort_keys=True))
            tmp.replace(path)
        except Exception as exc:
            log(f"Could not save settings to {path}: {exc}", logging.WARNING)
            return []
        return keep

    def __post_init__(self) -> None:
        # Clamp at construction too, not only on live edits, so a bad value from
        # an environment variable at startup is corrected the same way the UI's
        # live edits are. apply({}) runs every guardrail with no other effect.
        self.apply({})

    def public(self) -> dict:
        data = {k: v for k, v in asdict(self).items()}
        # Never expose secrets through the API/UI or a config dump: report only
        # whether each is set. This is what makes GET /api/config safe to serve.
        for secret in self.SECRET_FIELDS:
            if secret in data:
                data[secret] = "***set***" if data.get(secret) else ""
        # Structured settings that may EMBED credentials (MCP server headers and
        # env, a DSN with a password). Not SECRET_FIELDS: those are blanked on
        # load from saved settings and reported only as set/unset, and the
        # shape of these (which servers, which databases) is worth showing.
        # Only the values that could be credentials are masked.
        for name in self.MASKED_STRUCTURED_FIELDS:
            if data.get(name):
                data[name] = mask_structured_secrets(str(data[name]))
        data["mutable"] = list(self.MUTABLE)
        return data

    def apply(self, updates: dict) -> list[str]:
        """Apply a settings patch. Returns the names of the fields changed.

        Values are clamped afterwards, and any field the clamp moved is reported
        as changed too, so the UI never shows a setting the process did not
        actually adopt.
        """
        changed = []
        for key, value in updates.items():
            if key not in self.MUTABLE or value is None:
                continue
            current = getattr(self, key)
            try:
                if isinstance(current, bool):
                    value = bool(value)
                elif isinstance(current, int):
                    value = int(value)
                elif isinstance(current, float):
                    value = float(value)
                else:
                    value = str(value)
            except (TypeError, ValueError):
                # A malformed value (e.g. "abc" for a number). Skip this field
                # rather than crash the whole settings update.
                log(f"Ignoring invalid value for {key}: {value!r}", logging.WARNING)
                continue
            # model_cutoff is validated HERE and not only by the sanitiser
            # below, because the two do different jobs. The sanitiser exists
            # for a bad value arriving from the environment, where the only
            # honest outcome is "unknown". A bad value arriving over
            # POST /api/config is a rejected EDIT, and adopting-then-blanking
            # it reported the field as changed, wrote "" into the settings
            # file, and so pinned the cutoff to empty for every future boot --
            # overriding a perfectly good MODEL_CUTOFF in the environment
            # because of one typo in the Settings panel. Leaving the current
            # value alone puts the field in the endpoint's `ignored` list,
            # which is what tells the user their value was refused.
            if (key == "model_cutoff" and str(value).strip()
                    and not is_usable_cutoff(value)):
                log(f"Ignoring invalid value for {key}: {value!r} "
                    "(expected YYYY-MM or YYYY)", logging.WARNING)
                continue
            if value != current:
                setattr(self, key, value)
                changed.append(key)

        before = {name: getattr(self, name) for name in
                  ("context_size", "max_tokens", "temperature", "agent_max_steps",
                   "agent_min_steps",
                   "history_turns", "search_results", "tool_result_chars",
                   "tool_temperature", "summarise_over_chars")}
        self.context_size = min(131072, max(512, self.context_size))
        self.max_tokens = max(16, self.max_tokens)
        if self.max_tokens >= self.context_size:
            self.max_tokens = max(16, self.context_size // 2)
        self.temperature = min(2.0, max(0.0, self.temperature))
        self.tool_temperature = min(2.0, max(0.0, self.tool_temperature))
        self.summarise_over_chars = max(500, self.summarise_over_chars)
        self.agent_max_steps = min(20, max(1, self.agent_max_steps))
        # A minimum above the maximum would loop until the step budget ran out
        # and then salvage, which is the opposite of thinking harder.
        self.agent_min_steps = min(self.agent_max_steps, max(1, self.agent_min_steps))
        self.resilient_retries = min(6, max(0, self.resilient_retries))
        self.min_max_tokens = min(512, max(32, self.min_max_tokens))
        self.stall_timeout = min(600, max(10, self.stall_timeout))
        # A floor rate at or below zero would make the derived budget infinite.
        self.decode_floor_tps = min(1000.0, max(0.5, self.decode_floor_tps))
        self.max_generation_timeout = min(
            7200.0, max(float(self.stall_timeout), self.max_generation_timeout))
        self.auto_fetch_char_cap = min(60000, max(1000, self.auto_fetch_char_cap))
        # The cap can never be below the ordinary step budget.
        self.hard_step_cap = min(60, max(self.agent_max_steps, self.hard_step_cap))
        self.reasoning_max_steps = min(10, max(2, self.reasoning_max_steps))
        # 1 is as low as it goes: a message with no analytical cue at all is
        # never decomposed, so 0 would mean the same thing as 1 while reading
        # like "always".
        self.reasoning_signals = min(6, max(1, self.reasoning_signals))
        self.reasoning_min_chars = min(2000, max(0, self.reasoning_min_chars))
        # "YYYY-MM" or empty. Anything else is dropped rather than
        # half-parsed: a malformed cutoff compared as a string would silently
        # make every year look recent, or none of them, and neither failure
        # announces itself. Empty is the honest value for "not known".
        #
        # The range check is not pedantry. A shape-only test accepts 9999-99,
        # which is a plausible typo for a year and silently switches the whole
        # freshness signal OFF -- no question is ever newer than the cutoff, so
        # nothing is ever looked up, and the only symptom is answers quietly
        # going stale. Out-of-range reads as unknown, which is conservative in
        # the right direction: unknown means any recent year counts.
        _cut = str(self.model_cutoff or "").strip()
        if is_usable_cutoff(_cut):
            self.model_cutoff = _cut
        else:
            if _cut:
                log(f"MODEL_CUTOFF {_cut!r} is not a usable YYYY-MM date; "
                    "treating the cutoff as unknown, which means any recent "
                    "year counts as needing a lookup.", logging.WARNING)
            self.model_cutoff = ""
        self.reasoning_step_timeout = min(300, max(10, self.reasoning_step_timeout))
        self.retrieval_deadline = min(600.0, max(15.0, self.retrieval_deadline))
        self.auto_iterate_rounds = min(5, max(0, self.auto_iterate_rounds))
        # Each pass is a full generation on a local model, so the ceiling is a
        # wall-clock guard: six passes of a slow 3B is several minutes of a
        # turn the user cannot interrupt except by stopping it.
        self.auto_continue_max = min(6, max(0, self.auto_continue_max))
        self.exec_memory_mb = min(65536, max(128, self.exec_memory_mb))
        self.exec_max_file_mb = min(65536, max(1, self.exec_max_file_mb))
        self.exec_pids_limit = min(8192, max(16, self.exec_pids_limit))
        # At least the tool timeout: a test command given less time than an
        # ordinary shell call is the wrong way round.
        self.test_timeout = min(3600, max(self.tool_timeout, self.test_timeout))
        self.rag_passages = min(20, max(1, self.rag_passages))

        # --- Skills ----------------------------------------------------------- #
        self.skills_in_prompt = min(100, max(1, self.skills_in_prompt))
        # 400 characters is about the shortest body that can still hold a
        # procedure; below that a skill is a description, not a skill.
        self.skill_body_chars = min(40000, max(400, self.skill_body_chars))
        self.skills_max = min(5000, max(1, self.skills_max))
        self.skill_autoload_overlap = min(10, max(1, self.skill_autoload_overlap))
        self.context_files_chars = min(40000, max(0, self.context_files_chars))
        self.reference_max = min(20, max(1, self.reference_max))
        self.reference_chars = min(60000, max(200, self.reference_chars))
        self.recall_results = min(50, max(1, self.recall_results))
        self.recall_snippet_chars = min(2000, max(40, self.recall_snippet_chars))
        self.subagent_max_concurrent = min(8, max(1, self.subagent_max_concurrent))
        self.subagent_max_per_turn = min(20, max(1, self.subagent_max_per_turn))
        self.subagent_max_steps = min(20, max(1, self.subagent_max_steps))
        self.subagent_result_chars = min(20000, max(200, self.subagent_result_chars))
        self.execute_code_max_calls = min(1000, max(1, self.execute_code_max_calls))
        self.checkpoint_keep = min(500, max(1, self.checkpoint_keep))
        self.checkpoint_max_file_bytes = max(0, self.checkpoint_max_file_bytes)
        self.mcp_timeout = min(300.0, max(1.0, self.mcp_timeout))
        self.skill_author_tokens = min(4096, max(96, self.skill_author_tokens))
        self.skill_retire_min_rated = max(1, self.skill_retire_min_rated)
        self.skill_retire_loss_rate = min(1.0, max(0.1, self.skill_retire_loss_rate))

        # --- Model server ----------------------------------------------------- #
        # A prefill step of 0 or a negative one would be rejected by the server;
        # anything above the context window is the same as no chunking at all.
        self.prefill_step_size = min(8192, max(64, self.prefill_step_size))
        self.prompt_cache_size = min(64, max(1, self.prompt_cache_size))
        # 64MB is the smallest cap that still holds a single useful prefix.
        self.prompt_cache_bytes = max(64 * 1024 ** 2, self.prompt_cache_bytes)
        self.decode_concurrency = min(16, max(1, self.decode_concurrency))
        self.prompt_concurrency = min(8, max(1, self.prompt_concurrency))
        self.watchdog_probe_interval = min(600.0, max(5.0, self.watchdog_probe_interval))
        self.watchdog_probe_timeout = min(120.0, max(2.0, self.watchdog_probe_timeout))
        self.watchdog_probe_failures = min(10, max(1, self.watchdog_probe_failures))
        self.watchdog_max_restarts = min(100, max(1, self.watchdog_max_restarts))
        self.watchdog_restart_window = min(86400.0, max(30.0, self.watchdog_restart_window))

        # --- Training -------------------------------------------------------- #
        self.train_min_examples = max(1, self.train_min_examples)
        self.train_epochs = min(50.0, max(0.1, self.train_epochs))
        self.train_batch_size = min(64, max(1, self.train_batch_size))
        self.train_min_iters = max(1, self.train_min_iters)
        self.train_max_iters = max(self.train_min_iters, self.train_max_iters)
        # 0 means "derive from train_epochs"; anything else is a manual pin and
        # is still held inside the floor/ceiling so a typo cannot run for a day.
        if self.train_iters:
            self.train_iters = min(self.train_max_iters, max(1, self.train_iters))
        else:
            self.train_iters = 0
        self.train_tool_examples = max(0, self.train_tool_examples)
        self.train_tool_ratio = max(0.0, self.train_tool_ratio)
        if str(self.train_tool_quality).strip().lower() not in ("rated", "all"):
            self.train_tool_quality = "rated"
        else:
            self.train_tool_quality = str(self.train_tool_quality).strip().lower()
        # Above ~0.5 the "rehearsal" set is the training set; below 0 is nonsense.
        self.train_replay_ratio = min(0.5, max(0.0, self.train_replay_ratio))
        # A split of 0 would leave nothing to judge the run by; 0.5 would leave
        # nothing to train on.
        self.train_val_split = min(0.5, max(0.05, self.train_val_split))
        self.train_val_tolerance = min(1.0, max(0.0, self.train_val_tolerance))
        self.train_timeout = min(86400.0, max(60.0, self.train_timeout))
        self.train_max_backups = min(1000, max(0, self.train_max_backups))
        self.train_lora_rank = min(256, max(1, self.train_lora_rank))
        self.train_lora_scale = min(1000.0, max(0.1, self.train_lora_scale))
        self.train_lora_dropout = min(0.9, max(0.0, self.train_lora_dropout))
        if str(self.train_fine_tune_type).strip().lower() not in ("lora", "dora", "full"):
            self.train_fine_tune_type = "lora"
        else:
            self.train_fine_tune_type = str(self.train_fine_tune_type).strip().lower()
        # A scope of unusable entries would silently return nothing; normalise it.
        self.rag_scope = ",".join(
            part.strip() for part in str(self.rag_scope or "").split(",") if part.strip())
        self.reasoning_tokens = min(2048, max(64, self.reasoning_tokens))
        self.ready_wait_timeout = min(300.0, max(2.0, self.ready_wait_timeout))
        # Ratios kept in sane bands so a bad value cannot break chunking: the
        # trigger must leave room for a reply, and a chunk must be smaller than
        # the trigger or it could never fit.
        self.chunk_trigger_ratio = min(0.9, max(0.2, self.chunk_trigger_ratio))
        self.chunk_size_ratio = min(self.chunk_trigger_ratio, max(0.1, self.chunk_size_ratio))
        self.history_turns = min(200, max(0, self.history_turns))
        self.search_results = min(10, max(1, self.search_results))
        self.tool_result_chars = min(40000, max(200, self.tool_result_chars))
        self.tool_raw_chars = min(200000, max(self.tool_result_chars, self.tool_raw_chars))
        # The search provider is immutable: whatever arrived (env, CLI, a stale
        # constructor value), it is forced back to DuckDuckGo Lite here.
        self.search_backend = _normalize_search_backend(self.search_backend)

        # Anything that is not "docker" means the local sandbox at the two call
        # sites that read this, so junk was already safe -- but, as with
        # log_level, it must not be stored and displayed as if it were in use.
        self.exec_backend = (str(self.exec_backend or "local").strip().lower()
                             if str(self.exec_backend or "").strip().lower()
                             in ("local", "docker") else "local")

        # --- Logging -------------------------------------------------------- #
        # Held to a level the logger will actually adopt. level_from_name()
        # already falls back to INFO on junk, so an unrecognised name was
        # harmless at the consumer -- but it was still stored and shown in
        # Settings, which is the one thing this pass is not allowed to do:
        # report a setting the process is not using.
        self.log_level = str(self.log_level or "INFO").strip().upper() or "INFO"
        if self.log_level not in ("CRITICAL", "ERROR", "WARNING", "INFO",
                                  "DEBUG", "TRACE", "NOTSET"):
            self.log_level = "INFO"
        if str(self.log_format).strip().lower() not in ("json", "text"):
            self.log_format = "json"
        else:
            self.log_format = str(self.log_format).strip().lower()
        if str(self.log_chat_content).strip().lower() not in ("disabled", "metadata", "full"):
            self.log_chat_content = "metadata"
        else:
            self.log_chat_content = str(self.log_chat_content).strip().lower()
        self.log_max_bytes = max(0, self.log_max_bytes)
        self.log_backup_count = min(1000, max(0, self.log_backup_count))
        self.log_retention_days = min(3650, max(0, self.log_retention_days))

        # --- Auth ----------------------------------------------------------- #
        self.auth_session_ttl_hours = min(8760, max(1, self.auth_session_ttl_hours))
        if str(self.oidc_default_role).strip().lower() not in ("user", "admin"):
            self.oidc_default_role = "user"

        # --- Cluster / routing ---------------------------------------------- #
        if str(self.node_role).strip().lower() not in ("primary", "secondary"):
            self.node_role = "primary"
        else:
            self.node_role = str(self.node_role).strip().lower()
        self.route_max_active_per_node = min(256, max(1, self.route_max_active_per_node))
        self.route_queue_depth = min(100000, max(0, self.route_queue_depth))
        self.route_cpu_pct = min(100.0, max(1.0, self.route_cpu_pct))
        self.route_mem_pct = min(100.0, max(1.0, self.route_mem_pct))
        self.route_load_ratio = min(64.0, max(0.0, self.route_load_ratio))
        self.code_max_tokens = min(32768, max(256, self.code_max_tokens))
        self.task_artifact_chars = min(200_000, max(0, self.task_artifact_chars))
        self.artifact_reply_headroom = min(8192, max(0, self.artifact_reply_headroom))
        self.artifact_patch_chars = min(200_000, max(0, self.artifact_patch_chars))
        self.route_sla_ms = min(3600000, max(0, self.route_sla_ms))
        self.route_cooldown_s = min(3600.0, max(0.0, self.route_cooldown_s))
        self.agent_run_timeout = min(3600.0, max(10.0, self.agent_run_timeout))
        self.heartbeat_interval = min(3600.0, max(1.0, self.heartbeat_interval))
        self.heartbeat_timeout = min(86400.0, max(2.0, self.heartbeat_timeout))
        self.node_probe_timeout = min(120.0, max(1.0, self.node_probe_timeout))

        # --- Import limits -------------------------------------------------- #
        self.import_max_zip_bytes = max(1024, self.import_max_zip_bytes)
        self.import_max_files = min(10_000_000, max(1, self.import_max_files))
        self.import_max_uncompressed_bytes = max(1024, self.import_max_uncompressed_bytes)
        self.import_max_file_bytes = max(1024, self.import_max_file_bytes)

        # --- Admission control ----------------------------------------------- #
        # GenerationGate already floored both at 1 internally, so the clamp only
        # makes the stored value match what the gate uses. 64 slots is beyond
        # any Mac's KV budget; the multiplier caps the waiting room at 64x that.
        self.max_concurrent_generations = min(64, max(1, self.max_concurrent_generations))
        self.generation_queue_depth = min(64, max(1, self.generation_queue_depth))
        self.admission_min_free_mb = min(1_048_576, max(0, self.admission_min_free_mb))

        # --- Inference backend ----------------------------------------------- #
        # An unknown backend name is NOT rewritten here: backends.get_backend
        # falls back to mlx-lm with a warning, and /api/backends shows what was
        # asked for. Only the shape is normalised.
        self.model_backend = str(self.model_backend or "mlx-lm").strip().lower() or "mlx-lm"
        self.draft_model = str(self.draft_model or "").strip()
        self.num_draft_tokens = min(64, max(0, int(self.num_draft_tokens or 0)))
        self.backend_command = str(self.backend_command or "").strip()
        self.backend_capabilities = ",".join(
            part.strip().lower() for part in str(self.backend_capabilities or "").split(",")
            if part.strip())
        self.backend_extra_args = str(self.backend_extra_args or "").strip()

        # --- Cluster / remote access ---------------------------------------- #
        # An empty bind host would make uvicorn listen on every interface,
        # the opposite of what "unset" should mean.
        self.web_host = str(self.web_host or "").strip() or "127.0.0.1"
        self.public_base_url = str(self.public_base_url or "").strip().rstrip("/")

        # --- Tool-call parsing --- #
        # An unknown format name falls back to "auto": a typo must not
        # silently stop every non-JSON tool call from parsing.
        self.tool_call_format = normalise_tool_call_format(self.tool_call_format)
        self.tool_call_repair = bool(self.tool_call_repair)

        # --- Model catalogue --- #
        # MODEL_ALIASES must be a JSON object of non-empty strings. A malformed
        # value is refused, not adopted, for the same reason as model_cutoff: a
        # typo in the Settings panel must not wipe aliases that MODEL_ID or a
        # client already depends on. The last good value is remembered on the
        # instance (apply({}) runs at construction, so it always exists);
        # dropping the key from `changed` puts it in /api/config's `ignored`.
        _cat_raw = str(self.model_aliases or "").strip()
        try:
            _cat_obj = json.loads(_cat_raw) if _cat_raw else {}
            _cat_ok = isinstance(_cat_obj, dict) and all(
                isinstance(k, str) and k.strip() and "/" not in k
                and isinstance(v, str) and v.strip() for k, v in _cat_obj.items())
        except ValueError:
            _cat_ok = False
        if _cat_ok:
            self.model_aliases = _cat_raw
            self.__dict__["_catalog_aliases_ok"] = _cat_raw
        else:
            _cat_prev = self.__dict__.get("_catalog_aliases_ok", "")
            log(f"Ignoring invalid MODEL_ALIASES {_cat_raw[:80]!r} (expected a JSON "
                'object like {"chat": "mlx-community/Qwen2.5-3B-Instruct-4bit"}); '
                "keeping the previous aliases.", logging.WARNING)
            self.model_aliases = _cat_prev
            if "model_aliases" in changed:
                changed.remove("model_aliases")
        self.bench_min_decode_tps = min(200.0, max(1.0, float(self.bench_min_decode_tps)))
        self.bench_headroom_gb = min(256.0, max(0.0, float(self.bench_headroom_gb)))

        # --- Training methods --- #
        # An unknown method is kept (normalised) rather than rewritten to sft:
        # the run then refuses it by name, where a rewrite would train SFT
        # silently on a typo.
        self.train_method = str(self.train_method or "sft").strip().lower() or "sft"
        self.train_autopilot = bool(self.train_autopilot)
        try:
            beta = float(self.train_beta)
        except (TypeError, ValueError):
            beta = 0.1
        self.train_beta = min(10.0, max(0.0, beta if beta == beta else 0.1))
        self.train_reference_adapter = str(self.train_reference_adapter or "").strip()
        self.train_grpo_rewards = str(self.train_grpo_rewards or "").strip()
        self.train_promote_ungated = bool(self.train_promote_ungated)

        # --- Chat turns ---
        try:
            grace = float(self.turn_detach_grace_s)
        except (TypeError, ValueError):
            grace = 30.0
        # NaN fails every comparison, so it would survive min/max untouched.
        self.turn_detach_grace_s = min(600.0, max(0.0, grace if grace == grace else 30.0))

        # --- Workspaces --- #
        self.workspaces_enabled = bool(self.workspaces_enabled)
        self.workspace_roots = ",".join(
            part.strip() for part in str(self.workspace_roots or "").split(",") if part.strip())

        # --- Tool approval policy --- #
        # An unknown mode becomes "risky", not "off": a typo in a safety
        # setting must fail towards asking, never towards silently allowing.
        _pol_mode = str(self.tool_approval_mode or "").strip().lower()
        _pol_aliases = {"": "off", "0": "off", "false": "off", "none": "off",
                        "1": "risky", "true": "risky", "on": "risky", "ask": "risky"}
        _pol_mode = _pol_aliases.get(_pol_mode, _pol_mode)
        if _pol_mode not in ("off", "risky", "all"):
            log(f"Unknown TOOL_APPROVAL_MODE {_pol_mode[:40]!r}; using 'risky'.",
                logging.WARNING)
            _pol_mode = "risky"
        self.tool_approval_mode = _pol_mode
        self.tool_approval_unattended = (
            "allow" if str(self.tool_approval_unattended or "").strip().lower() == "allow"
            else "deny")
        try:
            _pol_timeout = int(self.tool_approval_timeout_s)
        except (TypeError, ValueError):
            _pol_timeout = 120
        self.tool_approval_timeout_s = min(3600, max(5, _pol_timeout))
        self.tool_approval_taint = bool(self.tool_approval_taint)
        # TOOL_APPROVAL_ROLES must be a JSON object of objects; a malformed value
        # is refused (kept at the last good one), like MODEL_ALIASES above.
        _pol_raw = str(self.tool_approval_roles or "").strip()
        try:
            _pol_obj = json.loads(_pol_raw) if _pol_raw else {}
            _pol_ok = isinstance(_pol_obj, dict) and all(
                isinstance(v, dict) for v in _pol_obj.values())
        except (TypeError, ValueError):
            _pol_ok = False
        if _pol_ok:
            self.tool_approval_roles = _pol_raw
            self.__dict__["_policy_roles_ok"] = _pol_raw
        else:
            log(f"Ignoring invalid TOOL_APPROVAL_ROLES {_pol_raw[:80]!r} (expected a JSON "
                'object like {"user": {"risk:exec": "ask"}}).', logging.WARNING)
            self.tool_approval_roles = self.__dict__.get("_policy_roles_ok", "")
            if "tool_approval_roles" in changed:
                changed.remove("tool_approval_roles")
        # --- Per-user approvals --- #
        # Same aliases as TOOL_APPROVAL_MODE, but an unknown value becomes
        # "all": this is the setting that promises users are asked, so a typo
        # must fail towards asking every time.
        _apr_mode = str(self.approval_default_mode or "").strip().lower()
        _apr_mode = {"0": "off", "false": "off", "none": "off", "never": "off",
                     "1": "all", "true": "all", "on": "all", "ask": "all",
                     "changes": "risky"}.get(_apr_mode, _apr_mode)
        if _apr_mode not in ("off", "risky", "all"):
            log(f"Unknown APPROVAL_DEFAULT_MODE {_apr_mode[:40]!r}; using 'all'.",
                logging.WARNING)
            _apr_mode = "all"
        self.approval_default_mode = _apr_mode

        # --- OpenAI / Anthropic compatible API ---
        _v1_mode = str(self.v1_default_mode or "").strip().lower()
        if _v1_mode not in ("agent", "raw"):
            log(f"Ignoring invalid V1_DEFAULT_MODE {_v1_mode[:40]!r} (expected agent or raw); "
                "using agent.", logging.WARNING)
            _v1_mode = "agent"
        self.v1_default_mode = _v1_mode
        self.v1_raw_enabled = bool(self.v1_raw_enabled)

        # --- API tokens / CLI sign-in ---
        try:
            _tok_ttl = int(self.api_token_default_ttl_days)
        except (TypeError, ValueError):
            _tok_ttl = 90
        self.api_token_default_ttl_days = min(3650, max(0, _tok_ttl))
        self.oidc_cli_client_id = str(self.oidc_cli_client_id or "").strip()

        # --- MCP server + gated tools ---
        self.mcp_server_enabled = bool(self.mcp_server_enabled)
        # ROLE_CAPABILITIES must be a JSON object of lists; a malformed value is
        # refused (kept at the last good one) rather than locking everyone out
        # or, worse, silently granting everything a typo was meant to narrow.
        _rc_raw = str(self.role_capabilities or "").strip()
        try:
            _rc_obj = json.loads(_rc_raw) if _rc_raw else {}
            _rc_ok = isinstance(_rc_obj, dict) and all(
                isinstance(v, list) and all(isinstance(c, str) for c in v)
                for v in _rc_obj.values())
        except (TypeError, ValueError):
            _rc_ok = False
        if _rc_ok:
            self.role_capabilities = _rc_raw
            self.__dict__["_role_caps_ok"] = _rc_raw
        else:
            log(f"Ignoring invalid ROLE_CAPABILITIES {_rc_raw[:80]!r} (expected a JSON "
                'object like {"user": ["web_search"], "admin": ["*"]}).', logging.WARNING)
            self.role_capabilities = self.__dict__.get("_role_caps_ok", "")
            if "role_capabilities" in changed:
                changed.remove("role_capabilities")
        try:
            _ig_timeout = int(self.image_gen_timeout_s)
        except (TypeError, ValueError):
            _ig_timeout = 600
        self.image_gen_timeout_s = min(3600, max(10, _ig_timeout))
        # --- Multimodal input ---
        self.attachments_enabled = bool(self.attachments_enabled)
        self.attachment_max_bytes = min(512 * 1024 * 1024,
                                        max(1024, int(self.attachment_max_bytes)))
        self.attachment_quota_mb = min(1024 * 1024, max(0, int(self.attachment_quota_mb)))
        _mm_vision = str(self.vision_mode or "").strip().lower()
        if _mm_vision not in ("auto", "on", "off"):
            log(f"Ignoring invalid VISION_MODE {_mm_vision[:40]!r} (expected auto, on or "
                "off); using auto.", logging.WARNING)
            _mm_vision = "auto"
        self.vision_mode = _mm_vision
        # Repo ids and voice names only: these reach a model loader, so a path
        # or URL typed into Settings must not.
        for _mm_name, _mm_default in (("whisper_model", "mlx-community/whisper-tiny"),
                                      ("tts_model", "mlx-community/Kokoro-82M-bf16"),
                                      ("tts_voice", "af_heart")):
            _mm_value = str(getattr(self, _mm_name) or "").strip()
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*(/[A-Za-z0-9][A-Za-z0-9_.-]*)?",
                                _mm_value or "-") or len(_mm_value) > 200:
                if _mm_value:
                    log(f"Ignoring invalid {_mm_name.upper()} {_mm_value[:60]!r}; using "
                        f"{_mm_default}.", logging.WARNING)
                _mm_value = _mm_default
            setattr(self, _mm_name, _mm_value)
        # --- Cloud fallback --- #
        self.cloud_fallback_enabled = bool(self.cloud_fallback_enabled)
        _cl_provider = str(self.cloud_provider or "").strip().lower()
        if _cl_provider not in ("openai", "anthropic"):
            log(f"Ignoring invalid CLOUD_PROVIDER {_cl_provider[:40]!r} (expected openai "
                "or anthropic); using openai.", logging.WARNING)
            _cl_provider = "openai"
        self.cloud_provider = _cl_provider
        # An endpoint that is not http(s) is refused outright: this URL receives
        # user content, so a typo must not turn into some other scheme handler.
        _cl_url = str(self.cloud_base_url or "").strip().rstrip("/")
        if _cl_url and not re.match(r"^https?://[^\s/]+", _cl_url):
            log(f"Ignoring invalid CLOUD_BASE_URL {_cl_url[:60]!r} (expected http(s)://...); "
                "using the provider default.", logging.WARNING)
            _cl_url = ""
        self.cloud_base_url = _cl_url
        self.cloud_model = str(self.cloud_model or "").strip()[:200]
        # An unknown consent mode becomes per_request, the stricter one.
        _cl_consent = str(self.cloud_consent or "").strip().lower().replace("-", "_")
        if _cl_consent not in ("per_request", "per_user"):
            log(f"Ignoring invalid CLOUD_CONSENT {_cl_consent[:40]!r} (expected per_request "
                "or per_user); using per_request.", logging.WARNING)
            _cl_consent = "per_request"
        self.cloud_consent = _cl_consent
        self.cloud_min_prefill_s = min(86400.0, max(0.0, float(self.cloud_min_prefill_s)))
        self.cloud_min_prompt_tokens = min(10_000_000, max(0, int(self.cloud_min_prompt_tokens)))
        self.cloud_allowed_roles = ",".join(
            part.strip() for part in str(self.cloud_allowed_roles or "").split(",")
            if part.strip())[:2000]
        self.cloud_max_tokens_per_user_month = min(
            1_000_000_000_000, max(0, int(self.cloud_max_tokens_per_user_month)))
        # --- Evaluation suite --- #
        self.eval_on_promote = bool(self.eval_on_promote)
        self.eval_max_regression = min(1.0, max(0.0, float(self.eval_max_regression)))
        # --- File-tool deny-list and generated documents --- #
        # Blank entries dropped and whitespace trimmed, so ".env, data/**" is
        # two patterns; capped so a pasted file cannot become the regex.
        self.file_deny_globs = ",".join(
            part.strip() for part in str(self.file_deny_globs or "").split(",")
            if part.strip())[:4000]
        # Relative to the active root and confined to it: an absolute path or a
        # ".." would put generated documents outside every root's resolver, so
        # either falls back to the default with a warning.
        _out_dir = str(self.outputs_dir or "").strip().replace("\\", "/")
        while _out_dir.startswith("./"):
            _out_dir = _out_dir[2:]
        _out_dir = _out_dir.rstrip("/")
        if (_out_dir.startswith(("/", "~")) or ".." in _out_dir.split("/")
                or re.match(r"^[A-Za-z]:", _out_dir) or len(_out_dir) > 200):
            log(f"Ignoring invalid OUTPUTS_DIR {_out_dir[:60]!r} (expected a folder "
                "relative to the project root); using outputs.", logging.WARNING)
            _out_dir = "outputs"
        self.outputs_dir = "" if _out_dir == "." else _out_dir
        # --- Agent routing, retrieval relevance and reasoning budget --- #
        self.rag_df_ceiling = min(1.0, max(0.01, float(self.rag_df_ceiling)))
        # An unknown mode becomes auto: the instruction then follows the
        # catalogue, which is the setting this field exists to make the default.
        _rp_mode = str(self.reasoning_prompt or "").strip().lower()
        if _rp_mode not in ("auto", "always", "never"):
            log(f"Ignoring invalid REASONING_PROMPT {_rp_mode[:40]!r} (expected auto, "
                "always or never); using auto.", logging.WARNING)
            _rp_mode = "auto"
        self.reasoning_prompt = _rp_mode
        self.think_max_tokens = min(8192, max(0, int(self.think_max_tokens)))
        # --- Intent lexicon --- #
        self.intent_lexicon_enabled = bool(self.intent_lexicon_enabled)
        # ---- extension clamps: one `# --- <Feature> ---` block per feature, above this line ---- #

        for name, old in before.items():
            if getattr(self, name) != old and name not in changed:
                changed.append(name)
        return changed



# --------------------------------------------------------------------------- #
# Agent capabilities. An "agent" is a named profile that switches on a set of
# capabilities; each capability maps to the concrete tools it unlocks, so the
# admin picks *what an agent can do* in plain terms and the registry gates the
# actual tools. Capabilities are the single source of truth for both the UI
# (the checkboxes on the Agents screen) and the runtime (the tool allowlist).
# --------------------------------------------------------------------------- #
CAPABILITY_GROUPS = {
    "file_ops": {
        "label": "File operations",
        "description": "Read, search, create and edit files in the project directory, "
                       "and produce documents (PDF, Word, Excel, PowerPoint).",
        "tools": ["read_file", "write_file", "edit_file", "list_files",
                  "search_files", "file_info", "create_document"],
    },
    "code_exec": {
        "label": "Run code & shell",
        "description": "Execute shell commands, run Python, and run the test suite "
                       "(needs --allow-shell / --allow-python to be enabled on the server).",
        "tools": ["run_shell", "run_python", "run_tests"],
    },
    "web_api": {
        "label": "Web & APIs",
        "description": "Search the web and fetch URLs / call HTTP APIs (DuckDuckGo Lite).",
        "tools": ["web_search", "fetch_url"],
    },
    "knowledge": {
        "label": "Knowledge base",
        "description": "Retrieve answers from the user's indexed documents (RAG).",
        # RAG itself is a retrieval flag (rag_for_capabilities()); the tools are
        # what lets a "Knowledge base" profile actually look things up and
        # answer "what is in my knowledge base", which RAG alone never could.
        "tools": ["search_docs", "kb_overview"],
    },
    "memory": {
        "label": "Memory",
        "description": "Remember and recall notes and prior feedback across turns.",
        "tools": ["remember", "recall_memory", "forget", "recall_feedback"],
    },
    "office365": {
        "label": "Office 365",
        "description": "Read/send mail, browse OneDrive/SharePoint files and read the "
                       "calendar via Microsoft Graph (needs an Azure AD app; configured "
                       "in Settings).",
        "tools": ["o365_mail", "o365_files", "o365_calendar"],
    },
    # --- The named tool capabilities (tools_extra.py) and MCP. The six keys
    # above stay for stored profiles and expand exactly as before; these are
    # finer-grained and several overlap them on purpose (web_search is inside
    # web_api). New tools register only when a capability (or EXTRA_TOOLS)
    # asks for them, so the default agent's prompt is unchanged.
    "web_search": {"label": "Web search", "description": "Search the web (DuckDuckGo Lite).",
                   "tools": ["web_search"]},
    "browse_page": {"label": "Browse pages",
                    "description": "Open a web page as readable text with its links, "
                                   "and follow one link on the same site.",
                    "tools": ["browse_page", "fetch_url"]},
    "code_interpreter": {"label": "Code interpreter",
                         "description": "A Python session whose variables persist across "
                                        "calls in a conversation (needs --allow-python).",
                         "tools": ["code_interpreter"]},
    "shell": {"label": "Shell", "description": "Run shell commands and the test suite "
                                               "(needs --allow-shell).",
              "tools": ["run_shell", "run_tests"]},
    "git": {"label": "Git", "description": "Status, diff, log, show and branches of the "
                                           "project repository; stage and commit (never push).",
            "tools": ["git", "git_commit"]},
    "create_document": {"label": "Create documents",
                        "description": "Write PDF, Word, Excel, PowerPoint and other documents.",
                        "tools": ["create_document"]},
    "spreadsheet": {"label": "Spreadsheets",
                    "description": "Read cells and ranges of .xlsx/.csv files and write sheets.",
                    "tools": ["spreadsheet", "spreadsheet_write"]},
    "pdf_extract": {"label": "PDF extraction",
                    "description": "Page count and the text of chosen pages of a PDF.",
                    "tools": ["pdf_extract"]},
    "image_analyze": {"label": "Image analysis",
                      "description": "Describe or answer questions about an image (needs a "
                                     "vision model).",
                      "tools": ["image_analyze"]},
    "image_generate": {"label": "Image generation",
                       "description": "Generate an image with a local generator (mflux or "
                                      "IMAGE_GEN_COMMAND).",
                       "tools": ["image_generate"]},
    "email": {"label": "Email", "description": "Search and read your Outlook mail; sending "
                                               "asks first (Microsoft Graph).",
              "tools": ["email", "email_send"]},
    "calendar": {"label": "Calendar", "description": "Read your Outlook calendar; creating "
                                                     "events asks first (Microsoft Graph).",
                 "tools": ["calendar", "calendar_create"]},
    "contacts": {"label": "Contacts", "description": "Search your Outlook contacts "
                                                     "(Microsoft Graph).",
                 "tools": ["contacts"]},
    "http_request": {"label": "HTTP requests",
                     "description": "Call public HTTP APIs with any method, headers and "
                                    "body; anything but GET asks first.",
                     "tools": ["http_request"]},
    "database": {"label": "Databases (read-only)",
                 "description": "Run read-only SQL against configured SQLite databases "
                                "or .db files in the project.",
                 "tools": ["database"]},
    "knowledge_search": {"label": "Search indexed documents",
                         "description": "Full-text search of the documents you indexed.",
                         "tools": ["search_docs"]},
    "task_scheduler": {"label": "Scheduled tasks",
                       "description": "Create, list and cancel your own scheduled agent "
                                      "tasks (never with more tools than you have).",
                       "tools": ["task_scheduler"]},
    "notify": {"label": "Notifications",
               "description": "Send you a short notification (in-app, webhook, macOS).",
               "tools": ["notify"]},
    "mcp": {"label": "MCP servers",
            "description": "Every tool of every configured MCP server. Grant a single "
                           "server with its own MCP: key instead.",
            "tools": []},   # dynamic: see register_capability_resolver / mcp.py
    # ---- extension capabilities: above this line ---- #
}

# Tools every agent always has: the loop's exit condition plus harmless utilities
# that need no permission. Never gated by a capability.
ALWAYS_ON_TOOLS = ["final_answer", "calculator", "current_time", "weather"]

# Order capabilities are shown in the UI.
CAPABILITY_ORDER = ["file_ops", "code_exec", "web_api", "knowledge", "memory", "office365"]
# The named tool capabilities and MCP (one "mcp:<server>" key per configured
# server is inserted after "mcp" by mcp.mcp_sync_capabilities).
CAPABILITY_ORDER += [
    "web_search", "browse_page", "code_interpreter", "shell", "git", "create_document",
    "spreadsheet", "pdf_extract", "image_analyze", "image_generate", "email", "calendar",
    "contacts", "http_request", "database", "knowledge_search", "task_scheduler",
    "notify", "mcp",
]

# Capabilities whose tools are only known at run time (MCP servers). Each
# resolver maps a capability list to extra concrete tool names.
_CAPABILITY_RESOLVERS: list = []


def register_capability_resolver(fn) -> None:
    """Add a fn(capabilities) -> [tool names] for dynamic capabilities."""
    if fn not in _CAPABILITY_RESOLVERS:
        _CAPABILITY_RESOLVERS.append(fn)


def tools_for_capabilities(capabilities) -> list[str]:
    """The concrete tool allowlist unlocked by a set of capability keys.

    Always includes ALWAYS_ON_TOOLS so the agent can still answer and do basic
    utility work even with no capability enabled.
    """
    caps = set(capabilities or [])
    allowed = list(ALWAYS_ON_TOOLS)
    for key in CAPABILITY_ORDER:
        if key in caps:
            allowed.extend(CAPABILITY_GROUPS[key]["tools"])
    for resolver in list(_CAPABILITY_RESOLVERS):
        try:
            allowed.extend(resolver(caps) or [])
        except Exception as exc:  # noqa: BLE001 - a dynamic capability is optional
            log(f"capability resolver failed: {exc}", logging.WARNING)
    # De-dupe, preserve order.
    seen: set[str] = set()
    out: list[str] = []
    for name in allowed:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def rag_for_capabilities(capabilities) -> bool:
    """Whether the knowledge-base (RAG) retrieval should be on for these caps."""
    return "knowledge" in set(capabilities or [])


def public_capabilities() -> list[dict]:
    """Capability catalogue for the UI (key, label, description), in display order."""
    return [
        {"key": key,
         "label": CAPABILITY_GROUPS[key]["label"],
         "description": CAPABILITY_GROUPS[key]["description"]}
        for key in CAPABILITY_ORDER
    ]


def capabilities_for_role(role: str | None, config: object = None) -> list[str]:
    """The capability keys a role may use, from ROLE_CAPABILITIES.

    A role missing from the table (or no table at all, the default) keeps every
    capability: today's rule, where the profile alone decides. "*" is every
    key; "mcp" also grants every per-server "mcp:<server>" key. A key whose
    tools are all granted by other keys counts as granted too, so a role given
    the legacy "code_exec" may use a profile that asks for "shell" (the same
    tools), but not one that asks for "code_interpreter".
    """
    every = list(CAPABILITY_ORDER)
    raw = str(getattr(config, "role_capabilities", "") or "").strip() if config is not None else ""
    try:
        table = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        table = {}
    if not isinstance(table, dict) or str(role or "user") not in table:
        return every
    entry = table[str(role or "user")]
    if not isinstance(entry, list):
        return []           # fail closed on a shape the clamp would have refused
    granted = {c for c in entry if isinstance(c, str)}
    if "*" in granted:
        return every
    if "mcp" in granted:
        granted |= {k for k in every if k.startswith("mcp:")}
    tools: set[str] = set()
    for key in granted:
        tools.update((CAPABILITY_GROUPS.get(key) or {}).get("tools") or [])
    for key in every:
        group_tools = (CAPABILITY_GROUPS.get(key) or {}).get("tools") or []
        if group_tools and set(group_tools) <= tools:
            granted.add(key)
    return [k for k in every if k in granted]


_MASK_KEYS = re.compile(r"(?i)(secret|password|passwd|token|api[-_]?key|bearer|"
                        r"authorization|credential|cookie|private[-_]?key)")


def mask_structured_secrets(text: str) -> str:
    """A JSON setting with its credential-like values replaced by "***".

    Every value under "env" or "headers" is masked (that is where MCP servers
    take their tokens), as is any key that names a credential and any
    user:password@ in a URL. Text that is not JSON is masked as a whole: an
    unparseable value cannot be shown safely.
    """
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return "***set***" if text else ""

    def scrub_url(value: str) -> str:
        return re.sub(r"(?i)(://)([^/@\s]+)@", r"\1***@", value)

    def walk(node: object, parent: str = "") -> object:
        if isinstance(node, dict):
            out = {}
            for key, value in node.items():
                if parent in ("env", "headers") or _MASK_KEYS.search(str(key)):
                    out[key] = "***" if not isinstance(value, (dict, list)) else walk(value, key)
                else:
                    out[key] = walk(value, str(key).lower())
            return out
        if isinstance(node, list):
            return [walk(item, parent) for item in node]
        if isinstance(node, str):
            return scrub_url(node)
        return node

    return json.dumps(walk(data))


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'Config',
    '_DDG_LITE_ALIASES',
    '_normalize_search_backend',
    'is_usable_cutoff',
    'CAPABILITY_GROUPS',
    'CAPABILITY_ORDER',
    'ALWAYS_ON_TOOLS',
    'tools_for_capabilities',
    'rag_for_capabilities',
    'public_capabilities',
    'register_capability_resolver',
    'capabilities_for_role',
    'mask_structured_secrets',
]
