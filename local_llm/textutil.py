"""Message classifiers, chunking and routing heuristics.

Split out of the original single-file deploy.py; behaviour is unchanged.
"""

from __future__ import annotations

import json
import re
from typing import NamedTuple

from .core import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .tools import *  # noqa: F401,F403
from .model_client import *  # noqa: F401,F403


def _is_trivial(text: str) -> bool:
    """True if the whole message is just a greeting or acknowledgement."""
    match = TRIVIAL_MESSAGE.match(text)
    # The pattern matches a leading greeting; require that only punctuation and
    # whitespace follow it, so "hi there" is trivial but "hi, fix this bug" is not.
    return bool(match and _TRIVIAL_TAIL.fullmatch(text[match.end():]))


def is_substantive(message: str) -> bool:
    """True if a message is a real question or request worth routing.

    Used by the chat handlers to keep chit-chat on the cheap plain-reply path
    while sending anything substantive through the model router.
    """
    text = (message or "").strip()
    if len(text) < 8 or _is_trivial(text):
        return False
    return True


def _last_user_turn(history: list[dict] | None) -> str | None:
    """The most recent user message in a history list.

    The router is given recent history, but this is also used to log or inspect
    the previous turn when resolving follow-ups like "look it up".
    """
    for turn in reversed(history or []):
        if turn.get("role") == "user" and (turn.get("content") or "").strip():
            return turn["content"]
    return None


# Requests to write or modify code. These are answered from the model's own
# knowledge and must never be routed to a web search: a code-generation prompt
# ("write a python script...") sent to a search engine returns tutorials at best
# and, as with "os recon" being read as OCR, irrelevant junk at worst. Detected
# deterministically so it also skips the router call entirely.
CODE_INTENT = re.compile(
    r"\b(write|create|generate|build|implement|code|program|fix|debug|"
    r"refactor|optimi[sz]e|complete|extend|port|convert|translate|add|modify|"
    r"update|rewrite|snippet)\b",
    re.I,
)
CODE_OBJECT = re.compile(
    r"\b(script|scripts|function|functions|program|programme|code|class|classes|"
    r"method|methods|module|snippet|regex|regexp|cli|parser|app|application|"
    r"component|query|loop|algorithm|algorithms|unit ?tests?|test|tests|api|"
    r"endpoint|schema|decorator|generator|command|one[- ]?liner|"
    # Systems-programming objects. "write a mouse driver ... in c" matched no
    # object and no language, so it was not treated as code: it got the 512-token
    # chat budget and was cut off mid-function. Deliberately excludes ambiguous
    # business words (service, interface, library) whose false positives would
    # skip the router on a genuine lookup question.
    r"driver|drivers|daemon|makefile|header|headers|struct|shader|firmware|"
    r"binding|bindings|wrapper|middleware|dockerfile|kernel module)\b",
    re.I,
)
# Programming languages and runtimes worth treating as a code signal on their own
# when paired with a code verb.
CODE_LANGUAGE = re.compile(
    r"\b(python|py|javascript|js|typescript|ts|rust|go|golang|c\+\+|cpp|c#|java|"
    r"bash|shell|sh|zsh|ruby|php|perl|sql|html|css|react|node|node\.js|"
    r"swift|swiftui|kotlin|dart|flutter|scala|objective-c|objc|lua|r|matlab|"
    r"powershell|ps1|julia|haskell|elixir|clojure|solidity)\b",
    re.I,
)
# C, which cannot go in CODE_LANGUAGE as a bare \bc\b without matching "vitamin
# c" and every stray initial. These are the phrasings that unambiguously mean
# the language.
CODE_C_LANGUAGE = re.compile(
    r"(?i)\b(?:in|using|with|of)\s+c\b(?!\+|#|\w)|\bansi\s+c\b|\bpure\s+c\b|"
    r"\bc\s*(?:89|90|99|11|17|23)\b|"
    r"\bc\s+(?:program|programme|code|source|function|header|library|driver|app)\b")


# A code request that plausibly depends on external or current information, so
# the model should look it up before writing. Without this, "write a script
# using the latest X API" or "a recon script covering current techniques" would
# be answered from stale weights.
CODE_NEEDS_LOOKUP = re.compile(
    r"\b(latest|current|recent|newest|modern|up[- ]?to[- ]?date|today|this year|"
    r"as of|version|changelog|"
    # Security-research nouns: these move fast and benefit from current sources.
    r"recon|reconnaissance|exploit|exploits|vulnerabilit|cve|payload|"
    r"enumeration|privilege escalation|pentest|attack)\b",
    re.I,
)

# Time-sensitive signal: the same notion the router is told to search on (today's
# events, prices, scores, the latest version of something, a named current
# holder). Used to let a single self-issued search through on an answer-routed
# turn when the question genuinely needs fresh, external facts, while still
# blocking reflexive searches for general knowledge.
TIME_SENSITIVE = re.compile(
    r"\b(today|todays|tonight|yesterday|right now|currently|current|latest|"
    r"newest|recent|recently|this (?:week|month|year|morning|season)|"
    r"as of|so far this|"
    r"price|prices|cost|stock|shares|exchange rate|rate today|"
    r"news|headline|breaking|score|scores|standings|fixture|"
    r"weather|forecast|temperature|"
    r"release|released|version|update|changelog|"
    r"who is the (?:current|new)|latest version|out yet|release date)\b",
    re.I,
)


# Implicit chat feedback. A short reply that is essentially praise or a plain
# rejection is a rating of the previous answer: "good job" approves it as a
# training example, "no, wrong" marks it as a bad answer. Kept conservative so a
# substantive message that merely starts with "no" is not misread; longer
# messages must contain an explicit multi-word phrase.
_POS_LEAD = re.compile(
    r"^\s*(?:that'?s\s+)?(great|perfect|excellent|awesome|amazing|nice|good|"
    r"correct|exactly|right|yes+|yep|yeah|thanks?|thank\s+you|ty|helpful|"
    r"brilliant|love\s+it|works|spot\s+on)\b", re.I)
_NEG_LEAD = re.compile(
    r"^\s*(?:no+|nope|nah|wrong|incorrect|bad|false|not\s+(?:right|correct|quite|good))\b",
    re.I)
_POS_PHRASE = re.compile(
    r"\b(good\s+job|well\s+done|that'?s\s+(?:right|correct|perfect)|exactly\s+right|"
    r"perfect\s+answer|that\s+works|great\s+answer|that'?s\s+it)\b", re.I)
_NEG_PHRASE = re.compile(
    r"\b(that'?s\s+(?:wrong|incorrect|not\s+right|false)|wrong\s+answer|"
    r"you'?re\s+wrong|not\s+(?:what\s+i|correct)|that'?s\s+not\s+it|bad\s+answer)\b",
    re.I)
# Guards: leading tokens that look negative/positive but are not feedback.
_NOT_FEEDBACK = re.compile(
    r"^\s*(?:no\s+(?:way|idea|one|problem|worries|clue)|not\s+sure|"
    r"right\s+(?:now|away)|yes\s+(?:and|but|please)\b|"
    # "no" as a determiner, not a verdict: "no timeout is needed here" is a
    # statement about the design, and treating it as a rejection re-asks the
    # previous request instead of acting on it.
    r"no\s+\w+\s+(?:is|are|was|were|will|would|should|needs?|needed|required)\b)",
    re.I)

# What may follow a lead word and still leave the message pure feedback. Anything
# else means the word was doing grammatical work -- "GOOD, now add retries",
# "CORRECT the typo on line 4", "NO timeout is needed here" -- rather than
# judging the answer. A lead-word verdict therefore has to consume the WHOLE
# message, the same reasoning as the continuation pattern: a verdict that is
# only a prefix is a guess about a sentence that went on to say something else.
_FEEDBACK_TAIL = re.compile(
    r"(?i)^[\s,.!]*(?:that|this|it|thanks?|thank\s+you|ty|cheers|"
    r"mate|man|perfect|great|now|indeed|much\s+better|"
    r"very\s+much|a\s+lot|so\s+much|"
    r"answer|one|work[s]?|"
    # A second verdict word is still just a verdict: "yes exactly", "no, wrong".
    r"exactly|right|correct|wrong|incorrect|nope|no+|yes+|"
    r"is\s+(?:right|correct|wrong|incorrect)|"
    r"was\s+(?:right|correct|wrong|incorrect))*[\s,.!?]*$")

# Praise that is immediately withdrawn. "thanks, but it still fails" was being
# stored as a POSITIVE training example of the very answer the user had just
# said did not work -- and positive/negative labels here are not advisory, they
# become the LoRA dataset.
_WITHDRAWN = re.compile(
    r"(?i)\b(?:but|however|though|except|although)\b|"
    r"\b(?:still|again)\s+(?:\w+\s+){0,2}"
    r"(?:fail|fails|failing|wrong|broken|error|errors|crash|crashes|off)\b|"
    r"\b(?:does\s*n[o']?t|doesn'?t|did\s*n[o']?t|didn'?t|is\s*n[o']?t|isn'?t|"
    r"won'?t|can'?t|cannot)\b")


def classify_implicit_feedback(message: str) -> int | None:
    """+1 if the message praises the prior answer, -1 if it rejects it, else None.

    Conservative by design, because this is not advisory: a verdict here is
    written to the feedback table and becomes a training example. A wrong label
    teaches the model that a bad answer was good, and a wrong NEGATIVE also
    makes the chat handler re-ask the previous request and throw away what the
    user actually just said. None is always the safe answer.
    """
    m = (message or "").strip()
    if not m:
        return None
    if _NOT_FEEDBACK.match(m):
        return None
    withdrawn = bool(_WITHDRAWN.search(m))
    if _POS_PHRASE.search(m):
        return None if withdrawn else 1
    if _NEG_PHRASE.search(m):
        return -1
    short = len(m.split()) <= 6
    if not short:
        return None
    # Asymmetric on purpose. A false POSITIVE teaches the model that a bad
    # answer was good and there is nothing downstream to catch it, so praise has
    # to consume the whole message: "good, now add retries" is an instruction
    # with a courtesy on the front, not an endorsement.
    match = _POS_LEAD.match(m)
    if match and _FEEDBACK_TAIL.match(m[match.end():]):
        return None if withdrawn else 1
    # A rejection is left looser. "no c++ just an empty json?" names what was
    # wrong in the same breath, and the retry path carries the user's own words
    # to the model as the complaint, so the instruction is not lost the way a
    # mislabelled positive is.
    if _NEG_LEAD.match(m):
        return -1
    return None


def rejection_retry_message(prompt: str, answer: str, complaint: str) -> str:
    """What to send the model after the user rejects an answer.

    A rejection is about the PREVIOUS request, so the request is re-asked with
    the complaint attached. Answering the complaint as a fresh question is how
    "no c++ just an empty json?" produced a second empty JSON object: the model
    was asked about the remark, not about the C++ program, while the rejected
    answer sat in the history as the only worked example in sight. Naming it as
    rejected is the point.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return (complaint or "").strip()
    rejected = " ".join((answer or "").split())[:300] or "(empty)"
    return (f"{prompt}\n\n[Your previous answer to this was REJECTED by the user, who "
            f"said: \"{(complaint or '').strip()}\". The rejected answer was: {rejected}\n"
            "Answer the original request again, in full, in plain text, fixing what "
            "they complained about. Do not repeat the rejected answer, and do not "
            "reply with JSON.]")


def build_reusable_dataset(rows: list[dict], fmt: str, system_prompt: str = "") -> tuple[str, int]:
    """Turn feedback rows into one of several reusable dataset formats.

    The point is portability: the data you collect should train ANY model later,
    not just this one. Formats:

    - "chat": messages with this assistant's system prompt. Trains a model to be
      THIS assistant. What the built-in LoRA loop uses.
    - "bare": messages with NO system prompt, just user/assistant. Model-neutral;
      use it to train a different base model or a different persona.
    - "preference": {prompt, chosen, rejected} triples for DPO-style preference
      tuning, built from corrected answers and from good/bad answers to the same
      prompt. This is what makes the rejected ("no, wrong") examples pay off.
    - "raw": every column as JSONL. A lossless archive you can reshape into any
      format in the future.

    Returns (text, example_count).
    """
    lines: list[str] = []

    if fmt == "raw":
        for r in rows:
            lines.append(json.dumps(r, ensure_ascii=False, default=str))
        return "\n".join(lines) + ("\n" if lines else ""), len(lines)

    if fmt == "preference":
        seen = set()
        # 1) corrected answers: original is rejected, correction is chosen.
        for r in rows:
            corrected = (r.get("corrected_response") or "").strip()
            original = (r.get("assistant_response") or "").strip()
            prompt = (r.get("user_prompt") or "").strip()
            if corrected and prompt and corrected != original:
                key = (prompt, corrected, original)
                if key in seen:
                    continue
                seen.add(key)
                lines.append(json.dumps(
                    {"prompt": prompt, "chosen": corrected, "rejected": original},
                    ensure_ascii=False))
        # 2) same prompt with an approved answer and a rejected answer.
        approved_by_prompt: dict[str, str] = {}
        rejected_by_prompt: dict[str, str] = {}
        for r in rows:
            prompt = (r.get("user_prompt") or "").strip()
            ans = (r.get("assistant_response") or "").strip()
            if not prompt or not ans:
                continue
            if r.get("approved_for_training"):
                approved_by_prompt.setdefault(prompt, ans)
            elif (r.get("rating") or 0) < 0:
                rejected_by_prompt.setdefault(prompt, ans)
        for prompt, chosen in approved_by_prompt.items():
            rejected = rejected_by_prompt.get(prompt)
            if rejected and rejected != chosen:
                key = (prompt, chosen, rejected)
                if key in seen:
                    continue
                seen.add(key)
                lines.append(json.dumps(
                    {"prompt": prompt, "chosen": chosen, "rejected": rejected},
                    ensure_ascii=False))
        return "\n".join(lines) + ("\n" if lines else ""), len(lines)

    # "chat" or "bare": supervised messages. Prefer the corrected answer as the
    # target when present; skip rejected-only rows (nothing good to imitate).
    seen_msgs = set()
    for r in rows:
        prompt = (r.get("user_prompt") or "").strip()
        target = (r.get("corrected_response") or r.get("assistant_response") or "").strip()
        if not prompt or not target:
            continue
        if not r.get("approved_for_training") and not (r.get("corrected_response") or "").strip():
            continue  # a bad answer with no correction is not a target to imitate
        key = (prompt, target)
        if key in seen_msgs:
            continue
        seen_msgs.add(key)
        messages = []
        if fmt == "chat" and system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": target})
        lines.append(json.dumps({"messages": messages}, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else ""), len(lines)


# A short remark ABOUT the previous answer rather than a new question:
# "doesnt have a main function", "no", "i meant ...", "its not ...". These refer
# to the conversation, so retrieving documents for them is wrong -- the terms
# match unrelated indexed material and the model answers from that instead.
_FOLLOWUP_LEAD = re.compile(
    r"^\s*(?:no+|nope|nah|yes+|yeah|ok|okay|wrong|incorrect|still|again|but|also|"
    r"i\s+meant|i\s+mean|that'?s|thats|it'?s|its|you\s+(?:just|didn'?t|did\s+not)|"
    r"doesn'?t|does\s+not|didn'?t|did\s+not|isn'?t|is\s+not|aren'?t|won'?t|can'?t|"
    r"cannot|reconsider|redo|retry|nevermind|never\s+mind)\b",
    re.I,
)


def is_followup_remark(message: str) -> bool:
    """True for a short comment on the previous turn, not a standalone question.

    Used to keep knowledge-base retrieval from firing on conversational repair
    ("doesnt have a main function"), where the useful context is the previous
    exchange, never an indexed document.
    """
    text = (message or "").strip()
    if not text or len(text.split()) > 20:
        return False
    return bool(_FOLLOWUP_LEAD.match(text))


def is_time_sensitive(message: str) -> bool:
    """True if a question plausibly needs current/external facts to answer well.

    Deliberately errs toward the router's own wording so the gate and the router
    agree on what 'needs a lookup' means.
    """
    return bool(TIME_SENSITIVE.search(message or ""))


# The model saying it cannot do the job without something it was not given: the
# web, the user's files, a shell. This is the model itself reporting that the
# tool-free lane was the wrong lane, and it is far stronger evidence than the
# router's one-shot guess -- the router decides before seeing the question's
# difficulty, this arrives after the model has tried.
#
# Split in two because the two halves want different tools. NEEDS_LOOKUP is
# answered by web_search/fetch_url; NEEDS_FILES by the file and shell tools.
_NEEDS_LOOKUP = re.compile(
    r"(?i)\b(?:"
    r"(?:i\s+)?(?:do\s*n[o']?t|don'?t|cannot|can'?t|could\s+not|couldn'?t|"
    r"am\s+(?:un|not\s+)able|'?m\s+(?:un|not\s+)able|unable)\s+"
    r"(?:to\s+)?(?:\w+\s+){0,3}?"
    r"(?:access|browse|reach|retrieve|look\s+(?:it|that|this)?\s*up|search|"
    r"fetch|download|open\s+(?:the\s+)?(?:link|url|page|website)|"
    r"real[- ]?time|live|current|up[- ]?to[- ]?date|internet|web)"
    r"|"
    r"(?:my|the)\s+(?:knowledge|training)\s+(?:cut[- ]?off|cutoff|data)\b"
    r"|"
    r"as\s+of\s+my\s+(?:last\s+)?(?:knowledge|training|update)"
    r"|"
    r"i\s+(?:do\s*n[o']?t|don'?t)\s+have\s+(?:access\s+to\s+)?"
    r"(?:real[- ]?time|live|current|up[- ]?to[- ]?date|internet|the\s+web)"
    r"|"
    r"(?:you|you'?ll|please)\s+(?:should\s+|will\s+|need\s+to\s+|can\s+)?"
    r"(?:check|visit|consult|see)\s+(?:the\s+)?(?:official|vendor|their)\s+"
    r"(?:website|site|docs|documentation|page)"
    r")\b"
)
_NEEDS_FILES = re.compile(
    r"(?i)\b(?:"
    r"(?:i\s+)?(?:do\s*n[o']?t|don'?t|cannot|can'?t|could\s+not|couldn'?t|"
    r"am\s+(?:un|not\s+)able|'?m\s+(?:un|not\s+)able|unable)\s+"
    r"(?:to\s+)?(?:\w+\s+){0,3}?"
    r"(?:read|open|see|view|inspect|list|run|execute)\s+"
    r"(?:your|the|any|this|that|these|those)?\s*"
    r"(?:file|files|code|codebase|repo|repository|project|directory|folder|"
    r"script|source|command|tests?)"
    r"|"
    r"without\s+(?:seeing|reading|access\s+to)\s+(?:your|the)\s+"
    r"(?:file|files|code|codebase|repo|repository|project|source)"
    r"|"
    r"(?:please\s+)?(?:paste|share|show|send)\s+(?:me\s+)?(?:the\s+)?"
    r"(?:file|files|code|contents|source|error|output)\b"
    r")"
)

# How much text may surround a refusal before it stops being a refusal. A real
# answer that opens "I don't have real-time access, but as of 2024 ..." and then
# answers is NOT a refusal, and escalating it would spend a search on a question
# already answered. A refusal is short because there was nothing to say.
REFUSAL_MAX_CHARS = 700


# The revision pass asking for the draft instead of improving it. Handed
# REVISE_DRAFT with its own draft immediately above, a small model sometimes
# answers "I don't see any draft or previous response to compare against" --
# and that reply, being non-empty, then REPLACED a finished 1800-character
# answer with a question the user cannot act on.
_NO_DRAFT_REPLY = re.compile(
    r"(?i)(?:"
    r"(?:do\s*n[o']?t|don'?t|cannot|can'?t|could\s+not|couldn'?t|unable\s+to)\s+"
    r"(?:see|find|locate|access)\s+(?:any\s+|a\s+|the\s+|your\s+)?"
    r"(?:draft|previous\s+(?:response|answer|message)|prior\s+\w+|text\s+to\s+review)"
    r"|"
    r"(?:please\s+)?(?:provide|share|paste|post|send)\s+(?:me\s+)?(?:the|your|a)\s+"
    r"(?:draft|text|response|answer|content)\s+(?:you\s+)?(?:want|for|to)\b"
    r"|"
    r"there\s+(?:is|was)\s+no\s+(?:draft|previous\s+(?:response|answer))"
    r")")
# A meta-reply is short by nature: it has no content of its own. Past this it is
# an answer that happens to mention a draft.
NO_DRAFT_MAX_CHARS = 400


def is_missing_draft_reply(text: str) -> bool:
    """True when a reply asks for the draft rather than being the improved one."""
    body = (text or "").strip()
    if not body or len(body) > NO_DRAFT_MAX_CHARS or "```" in body:
        return False
    return bool(_NO_DRAFT_REPLY.search(body))


# The explicit channel. A reasoning step is asked to say "NEED LOOKUP: <what>"
# when it cannot settle its sub-question from what it knows, and that is a far
# better signal than any phrasing heuristic because the model is answering the
# question directly rather than being caught apologising. It goes through
# missing_capability with everything else so there stays exactly ONE place that
# decides a turn was in the wrong lane -- two gates would drift, and the whole
# point of this one is that the escalation rules (a tool exists, the user did
# not say /no-search, there is a step left) are applied uniformly.
NEED_MARKER = re.compile(r"(?i)^\s*NEED\s+(LOOKUP|FILES)\s*[:\-]")

# Hedging: the model answering, but reporting low confidence in what it just
# said. Distinct from a refusal, which says it CANNOT answer. This is the
# characteristic 3B failure the refusal patterns miss entirely -- it does not
# say "I have no web access", it says "I don't have specific information about
# that product" and then guesses anyway, and the guess reads as fact.
_HEDGED = re.compile(
    r"(?i)\b(?:"
    r"i'?m\s+not\s+(?:sure|certain|confident)|"
    r"i\s+(?:am\s+)?not\s+(?:sure|certain|confident)|"
    r"i\s+(?:do\s*n[o']?t|don'?t)\s+have\s+(?:any\s+|reliable\s+|detailed\s+|"
    r"specific\s+|much\s+)?(?:information|data|details|knowledge)\b|"
    r"i'?m\s+not\s+aware\s+of|"
    r"(?:my|this)\s+information\s+may\s+be\s+(?:out\s*dated|outdated|stale|old)|"
    r"may\s+(?:no\s+longer|not)\s+be\s+(?:accurate|current|up[- ]to[- ]date)|"
    r"i\s+(?:believe|think)\s+(?:it|this|that|they)\s+(?:is|was|were|are)|"
    r"as\s+far\s+as\s+i\s+know|to\s+the\s+best\s+of\s+my\s+knowledge|"
    r"outside\s+(?:of\s+)?what\s+i\s+know|"
    r"you\s+(?:may|might|should)\s+want\s+to\s+(?:confirm|verify|double[- ]check)"
    r")\b")


# The model asking the USER something, rather than reporting a gap in its own
# knowledge. This is the most useful thing a small model can say about an
# ambiguous request, and it must never be mistaken for not knowing an answer.
_ASKS_THE_USER = re.compile(
    r"(?i)(?:\b(?:could|can|would|will)\s+you\s+"
    r"(?:please\s+)?(?:paste|share|send|show|tell|clarify|confirm|specify|"
    r"provide|give|point|let\s+me\s+know)\b"
    r"|\bplease\s+(?:paste|share|send|show|clarify|specify|provide|tell)\b"
    r"|\bwhich\s+(?:one|file|version|option|of\s+these)\b"
    r"|\bdid\s+you\s+mean\b|\bwhat\s+do\s+you\s+mean\b"
    r"|\blet\s+me\s+know\b)")
# A caveat about the user's OWN material is a careful answer, not a knowledge
# gap: there is nothing on the web that settles whether THEIR edge case holds.
_HEDGE_OWN_WORK = re.compile(
    # Nouns that are the user's work whatever determiner precedes them.
    r"(?i)\b(?:your|the|this|that)\s+(?:\w+\s+){0,2}?"
    r"(?:edge\s+cases?|codebase|repo|repository|script|function|"
    r"config(?:uration)?|setup|environment|test\s+suite|use\s+case)\b"
    # ... and the generic ones ONLY when they are explicitly the user's.
    # "data", "input", "file" and "project" were on the list above with a
    # bare "the" allowed, so an answer about an external fact that happened
    # to contain "the data suggests" or "the input from analysts" had its
    # escalation silently cancelled -- two ordinary words deciding whether a
    # knowledge gap gets looked up.
    r"|\b(?:your|my|our)\s+(?:\w+\s+){0,2}?"
    r"(?:data|input|file|files|project|code|tests?|requirements?)\b")


def is_hedged_answer(answer: str) -> bool:
    """True if a reply reports low confidence in what it just said.

    Deliberately NOT folded into missing_capability: a hedge is not a refusal,
    and the two want different treatment. A refusal means the turn produced
    nothing and the tools have to be handed over; a hedge means the turn
    produced an answer that happens to be untrustworthy, so a lookup is worth
    ONE attempt and the existing answer is the fallback if it fails.

    Bounded by the same REFUSAL_MAX_CHARS as a refusal, and blind to anything
    carrying a code block or a table: a long, worked answer with a caveat in it
    is a careful answer, and re-running the turn over a caveat would punish
    exactly the replies that deserve it least.
    """
    text = (answer or "").strip()
    if not text or len(text) > REFUSAL_MAX_CHARS or "```" in text:
        return False
    if not _HEDGED.search(text):
        return False
    # The model asking the USER something is not a gap in its knowledge, it is
    # the most useful thing a small model can say about an ambiguous request.
    # "I'm not sure which file you mean -- could you paste the path?" was being
    # read as a knowledge gap: the turn ran a web search and answered from it,
    # and the question the user needed to see was never shown.
    if _ASKS_THE_USER.search(text):
        return False
    # A caveat about the user's OWN material is a careful answer. "You may want
    # to double-check the edge case" is advice about their code, and there is
    # nothing on the web that settles it.
    if _HEDGE_OWN_WORK.search(text):
        return False
    return True


def missing_capability(answer: str) -> str | None:
    """Which capability an answer says it lacked: "lookup", "files", or None.

    Used to switch a tool-free turn into agent mode after the fact. Returns None
    for any answer long enough to be a real answer, so a hedged-but-complete
    reply is never mistaken for a refusal.
    """
    text = (answer or "").strip()
    # The explicit marker is checked before the length cap, because it is a
    # protocol token rather than a plea: a reasoning step that says NEED
    # LOOKUP: and then explains at length what is missing is still a step
    # asking for the web, and truncating that request at 700 characters would
    # be silently dropping the clearest signal there is.
    marked = NEED_MARKER.match(text)
    if marked:
        return "lookup" if marked.group(1).lower() == "lookup" else "files"
    if not text or len(text) > REFUSAL_MAX_CHARS:
        return None
    # A reply carrying a code block or a table did work; the caveat in it is a
    # caveat, not a refusal.
    if "```" in text:
        return None
    if _NEEDS_FILES.search(text):
        return "files"
    if _NEEDS_LOOKUP.search(text):
        return "lookup"
    return None


# A request for a document FILE, and which format it asks for. Detected
# deterministically, for the same reason is_code_request is: "write me a PDF
# report on the Q3 numbers" reaches the router as an ordinary request, the router
# quite reasonably answers "answer", and the turn is then rebuilt with no tools
# at all -- so the model writes a lovely report into the chat and no file is ever
# produced. The user asked for a file and got a message.
#
# Order matters: the first pattern that matches wins, so the more specific
# formats are listed before the ones whose words appear inside them.
_DOC_FORMAT_WORDS = (
    (".docx", r"word\s+(?:document|doc|file|version)|docx|ms\s*word|"
              r"microsoft\s+word"),
    (".xlsx", r"excel|spread\s?sheet|xlsx|workbook"),
    (".pptx", r"power\s?point|pptx|slide\s*deck|slides|presentation|deck"),
    (".pdf", r"pdf"),
    (".csv", r"csv"),
    (".html", r"html|web\s*page"),
    (".md", r"markdown|\.md\b"),
    (".txt", r"(?:plain\s*)?text\s+file|\.txt\b"),
)
# Verbs that mean "produce one", as opposed to reading one the user already has.
_DOC_MAKE = (r"(?:creat\w*|generat\w*|mak\w*|writ\w*|produc\w*|export\w*|sav\w*|"
             r"build\w*|draft\w*|prepar\w*|render\w*|render|put|compile|"
             r"give\s+me|send\s+me|i\s+(?:need|want)|turn\s+(?:it|this|that|them)\s+into|"
             r"convert\s+(?:it|this|that|them)\s+(?:in)?to)")
# A filename the user spelled out wins over any phrasing: "save it as notes.docx"
# names both the format and the file.
# Format words that are safe ONLY inside a target phrase. "to word" means Word;
# a bare "word" anywhere else is an English word.
_DOC_TARGET_EXTRA = {".docx": r"word", ".xlsx": r"sheets?", ".pptx": r"slides?"}
_DOC_FILENAME = re.compile(
    r"(?i)\b[\w][\w \-.]{0,60}?(\.(?:pdf|docx|xlsx|pptx|csv|html?|md|txt))\b")


# Verbs that mean the user already HAS the file and wants it read. A format word
# under one of these is not a request to produce anything: "summarise this pdf",
# "read the excel file I uploaded". Only an explicit output phrase ("... into a
# word document") overrides it.
_DOC_READ = re.compile(
    r"(?i)\b(?:read|open|parse|extract|summari[sz]e|analy[sz]e|review|index|"
    r"import|uploaded?|attached?|look\s+at|check|what(?:'s|\s+is)\s+in)\b")


def _doc_patterns() -> tuple[list, list]:
    """Two matchers per format, built once.

    `target` is the unambiguous one: "as/into/to a <fmt>" states the OUTPUT and
    therefore beats a filename mentioned elsewhere in the same sentence, which is
    what makes "convert report.pdf to word" a Word request rather than a PDF one.
    `other` covers "<fmt> file/report/..." and "<make verb> ... a <fmt>", both of
    which are requests to produce one but neither of which can outrank a
    filename the user spelled out.
    """
    target, other = [], []
    for suffix, words in _DOC_FORMAT_WORDS:
        extra = _DOC_TARGET_EXTRA.get(suffix)
        target.append((suffix, re.compile(
            r"(?i)\b(?:as|in-?to|to)\s+(?:an?\s+)?(?:\w+\s+){0,2}?"
            rf"(?:{words}{'|' + extra if extra else ''})\b")))
        other.append((suffix, re.compile(
            r"(?i)(?:"
            rf"\b(?:in)\s+(?:an?\s+)?(?:new\s+)?(?:{words})\b"
            rf"|\b(?:{words})\s+(?:file|document|doc|report|version|copy|export|"
            r"deck|presentation|summary|output)\b"
            rf"|\b{_DOC_MAKE}\s+(?:me\s+)?(?:\w+\s+){{0,3}}?(?:an?|the)?\s*(?:{words})\b"
            r")")))
    return target, other


_DOC_TARGET, _DOC_OTHER = _doc_patterns()


# The subject a document request is ABOUT, which is how a new document is told
# apart from an edit to the current one. Anchored at the end of the message so
# it takes the operative phrase: "create a powerpoint about kubernetes" is a new
# document, "make it a word document" names no subject at all and is a format
# change to the one in hand.
_DOC_SUBJECT = re.compile(
    r"(?i)\b(?:about|regarding|concerning|covering|detailing|for|of|on)\s+(.{3,90})$")
# Words too generic to be evidence of a different subject.
_SUBJECT_STOP = {
    "document", "documents", "file", "files", "report", "reports", "version",
    "copy", "export", "deck", "presentation", "summary", "output", "page",
    "pages", "word", "excel", "powerpoint", "spreadsheet", "workbook",
    "markdown", "please", "thanks", "again", "instead", "same", "thing",
    "this", "that", "them", "these", "those", "with", "from", "into", "onto",
    "style", "format", "layout", "template", "design", "branding",
}


def document_subject(message: str) -> str:
    """The trailing "about/for/of X" phrase of a message, or ""."""
    match = _DOC_SUBJECT.search(" ".join((message or "").split()))
    return match.group(1).strip(" .?!,;:") if match else ""


def _subject_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", (text or "").lower())
            if w not in _SUBJECT_STOP}


def starts_new_document(message: str, objective: str) -> bool:
    """True when a document request is for a DIFFERENT document, not an edit.

    Deliberately hard to satisfy, and the caller pairs it with "does not refer
    back to the active task". Resetting wrongly throws away the document source
    the user has been building up, which is the worse of the two mistakes -- the
    codebase makes the same call for code tasks in should_reset -- so it needs a
    subject of its own that shares nothing with what is already being made.
    """
    fresh = _subject_words(document_subject(message))
    if not fresh:
        return False
    return not (fresh & _subject_words(objective or ""))


# "look up as much as you can find about them, then draft the document" -- an
# explicit instruction to research before writing. Without it the document
# directive sends the model straight to create_document and the lookup the user
# asked for never happens.
_LOOKUP_ASK = re.compile(
    r"(?i)\b(?:look\s+(?:it|them|this|that|these|those)?\s*up|look\s+into|"
    r"research|find\s+out|search\s+(?:for|the\s+web|online|it)|check\s+online|"
    r"google\s+|find\s+(?:as\s+much|everything|all\s+you|what\s+you\s+can|"
    r"information|info|details)|"
    r"(?:from|based\s+on|using)\s+(?:their|the|its)\s+(?:website|site|page))\b")
# A bare domain the user typed without a scheme ("www.texcel.be", "texcel.be").
# Restricted to real TLDs so "main.py" and "v1.2" are not read as web addresses.
_BARE_DOMAIN = re.compile(
    r"(?i)\b(?:www\.[a-z0-9-]+\.[a-z]{2,}"
    r"|[a-z0-9][a-z0-9-]{1,}\.(?:com|be|nl|fr|de|org|net|io|co|eu|uk|dev|ai|"
    r"info|biz|es|it|lu|ch|at|se|dk|no|fi|pl|pt|ie)\b)")


def document_needs_research(message: str) -> bool:
    """True when a document request also asks for something to be looked up.

    Either explicitly ("look up as much as you can find about them") or by
    naming a site the document is supposed to be about. A document turn that
    ignores this produces a confident page of invented facts, which is worse
    than no document.
    """
    text = (message or "").strip()
    if not text:
        return False
    return bool(_LOOKUP_ASK.search(text) or _URL_IN_TEXT.search(text)
                or _BARE_DOMAIN.search(text))


def document_request(message: str) -> str | None:
    """The file extension a message asks to be produced, or None.

    Returns ".pdf", ".docx", ... so the caller can name the format back to the
    user and steer the tool. None for a message that merely mentions a format
    ("summarise this pdf"), which is a request to READ one -- acting on that
    would have the agent overwrite the very file it was asked to look at.
    """
    text = (message or "").strip()
    if not text:
        return None
    for suffix, pattern in _DOC_TARGET:
        if pattern.search(text):
            return suffix
    if _DOC_READ.search(text):
        return None
    named = _DOC_FILENAME.search(text)
    if named:
        suffix = named.group(1).lower()
        return ".html" if suffix == ".htm" else suffix
    for suffix, pattern in _DOC_OTHER:
        if pattern.search(text):
            return suffix
    return None


# Framing words to strip so the search topic is the subject, not "write a python
# script to ...". Applied only when building a query for a code lookup.
_CODE_FRAMING = re.compile(
    r"^(?:please\s+)?(?:can you\s+|could you\s+)?"
    r"(?:write|create|generate|build|implement|make|code|program|give me|show me)\s+"
    r"(?:me\s+)?(?:a|an|the)?\s*"
    r"(?:python|py|javascript|js|typescript|ts|rust|go|golang|bash|shell|sh|ruby|"
    r"php|perl|sql|c\+\+|cpp|java)?\s*"
    r"(?:script|program|function|snippet|tool|cli|code|app|module|class)?\s*"
    r"(?:that|which|to|for|attempting|covering|using|demonstrating|showing)?\s*",
    re.I,
)


def code_search_topic(message: str) -> str:
    """Strip 'write a python script to ...' framing down to the search subject."""
    topic = _CODE_FRAMING.sub("", (message or "").strip()).strip(" .?!\t")
    return topic or (message or "").strip()[:200]


# A hard analytical question that benefits from being broken into steps. Requires
# an analytical signal (compare, why, how would, evaluate, design, tradeoffs...)
# AND some heft (length, several clauses, or an explicit "step by step"), so it
# does not fire on simple factual or definitional questions that answer in one
# shot. Lookups and code are excluded upstream, so this only sees "answer"-class
# questions.
REASONING_SIGNAL = re.compile(
    r"\b(compare|contrast|versus|vs\.?|trade[- ]?offs?|pros and cons|"
    r"why (?:is|are|does|do|would|should|did)|how would|how do i|how should|"
    r"analy[sz]e|evaluate|assess|weigh|design|architect|derive|prove|"
    r"implications|consequences|reason through|think through|step by step|"
    r"walk me through|work out|figure out|explain why|justify|"
    r"what (?:would|if) )\b",
    re.I,
)


def is_reasoning_question(message: str, signals: int = 2, min_chars: int = 80) -> bool:
    """True if a question is worth decomposing into incremental reasoning steps.

    `signals` is how many analytical cues make it automatic, and `min_chars` the
    length at which a single cue is enough. Both are exposed as config so the
    bar can be moved: decomposition is the most expensive thing this pipeline
    does (one model call per step plus a synthesis), so the right threshold
    depends on the machine and on how much latency the user will trade for
    depth. A message with no cue at all never decomposes, whatever they are set
    to -- the cue list is what distinguishes a question with work in it from
    one that just needs an answer.
    """
    text = (message or "").strip()
    found = REASONING_SIGNAL.findall(text)
    if not found:
        return False
    # Enough analytical cues (e.g. "compare ... tradeoffs ... vs") means a
    # genuinely multi-faceted question regardless of length.
    if len(found) >= max(1, signals):
        return True
    # A single cue plus an explicit step-by-step request, length, or several
    # clauses. A lone short cue ("why is the sky blue") stays one-shot.
    if re.search(r"step by step|think through|walk me through", text, re.I):
        return True
    clauses = text.count(" and ") + text.count(", ") + text.count("?")
    return len(text) >= min_chars or clauses >= 2


# --------------------------------------------------------------------------
# What a question IS, as opposed to how long it is.
#
# is_reasoning_question above answers a single question -- "is this worth
# decomposing?" -- from a single family of cues, and everything that does real
# work without happening to use one of those words was answered in one shallow
# pass. Measured here against a corpus of ordinary messages, these ALL came
# back False: a percentage calculation, a train word problem, "who directed the
# film that won best picture the year Obama was elected", "plan a three day
# trip to Lisbon", "my docker build hangs at step 4" and "should I use postgres
# or sqlite". None of them contains "compare", "trade-offs" or "why does", and
# every one of them is a question a 3B model measurably improves on when it is
# allowed to take the problem apart first.
#
# It is wrong in the other direction too, because the cue list is base forms
# only and several of the words have a second, commoner sense. "Comparing
# Postgres and MySQL, which wins?" scores zero while the imperative form scores
# one. "How much does a 747 weigh" matched `weigh`, written for "weigh the
# options", and a single-fact question became a six-step chain with no tools.
# "Design a birthday card message" matched `design`. "Justify the text to both
# margins in LaTeX" matched `justify`. "I work out three times a week" matched
# `work out`.
#
# So the shape is classified first and the depth is derived from the shape. The
# shape is also what picks the treatment: "why does TCP use a three-way
# handshake" and "plan a three day trip" are both work and want completely
# different handling -- one a short causal chain, the other an itemised plan --
# and a single boolean cannot say that.
#
# Order is the whole design. Real messages carry cues for several kinds at once
# ("summarise this and explain why it matters" is both), so the ladder runs
# from the most structurally distinctive to the most general and the first
# match wins. Nothing here calls the model: this runs on every turn, including
# all the ones that will never decompose, so it has to be free.

QUESTION_KINDS = (
    "code", "summarize", "creative", "debug", "math", "multi_hop", "planning",
    "decision", "comparative", "analytical", "causal", "procedural",
    "definition", "chitchat", "factual",
)
# How much work the answer gets. "direct" answers in one pass, "short" gets a
# fuller budget and the model's own think-then-answer pass, "deep" is planned
# and decomposed into steps.
QUESTION_DEPTHS = ("direct", "short", "deep")

# How much of a message the kind patterns look at. A message's INTENT lives
# in its opening lines; everything after that is the material it is asking
# about, and a pasted file runs to tens of thousands of characters.
CLASSIFY_SCAN_CHARS = 2000

# Reshaping text the user already has. Outranks everything below it: "summarise
# this and explain why it matters" is a summary request, and decomposing it
# fed the synthesis six 300-character notes in place of the material.
_Q_TRANSFORM = re.compile(
    r"(?i)\b(?:summari[sz]\w*|tl;?dr|recap|condense|shorten|"
    r"translat\w*|rewrit\w*|reword|rephras\w*|proofread|"
    r"key\s+(?:points?|takeaways?)|gist\s+of|boil\s+(?:it|this)\s+down|"
    r"convert\s+(?:this|it|the\s+following))\b")

# Writing something for pleasure or for a person, where a plan then a worse
# poem is strictly worse than the poem.
_Q_CREATIVE = re.compile(
    r"(?i)\b(?:poem|haiku|sonnet|limerick|story|tale|song|lyrics?|joke|riddle|"
    r"toast|eulogy|caption|slogan|tagline|blurb|"
    r"(?:birthday|christmas|greeting|thank[- ]you)\s+card|"
    r"message\s+for\s+my|note\s+for\s+my|speech\s+for|"
    r"in\s+the\s+(?:voice|style)\s+of)\b"
    # Correspondence, but only where the user is asking for it to be WRITTEN.
    # A bare "email" is far more often one they already have, and _Q_TRANSFORM
    # runs first, so "summarise this email" never reaches here.
    r"|\b(?:writ\w+|draft\w*|compos\w+|pen)\s+(?:me\s+)?(?:a|an|the|some)?\s*"
    r"(?:\w+\s+){0,3}?(?:e-?mail|letter|memo|reply|note|message|post|tweet|"
    r"bio|pitch|essay|press\s+release|cover\s+letter|blog\s+post)\b")

# Something is broken. Near the top because a bug report is full of numbers
# that are not quantities: "hangs at step 4", "exits with code 137".
_Q_SYMPTOM = re.compile(
    r"(?i)\b(?:error|errors|exception|traceback|stack\s*trace|segfault|panic|"
    r"crash(?:es|ing|ed)?|fails?|failing|failed|failure|broken|breaks?|"
    r"hangs?|hanging|freezes?|frozen|stuck|times?\s*out|timing\s*out|timeout|"
    r"bugs?\s+in|the\s+bug|a\s+bug|"
    # "leak" and "regression" only in their fault sense. A bare "regression"
    # is the statistical one far more often here ("explain linear
    # regression"), and it would otherwise send a textbook question down the
    # decomposition path.
    r"memory\s+leak|leak(?:s|ing)\s+(?:memory|file|connections?|fds?)|"
    r"(?:a|the|caused\s+a|introduced\s+a)\s+regression|regression\s+in|"
    r"(?:does\s*n[o']?t|doesn'?t|won'?t|isn'?t|not)\s+work(?:ing)?|"
    r"stopped\s+working|no\s+longer\s+works?|returns?\s+nothing)\b"
    # An HTTP status, not a quantity. A bare [45]xx used to be enough, so "we
    # have 500 employees and need a payroll policy" and "a laptop for 450
    # euros" were both read as bug reports and sent down the six-generation
    # decomposition path. The number has to be doing a status's job.
    r"|\b(?:status|code|error|response|returns?|returned|got|getting|gives?|"
    r"throwing|with)\s+(?:an?\s+)?[45]\d\d\b"
    r"|\b[45]\d\d\s+(?:error|response|status|from|on|when)\b"
    r"|\bwrong\s+with\b")

# Asking for something to be WRITTEN, as opposed to reporting something that
# happened. Deliberately not CODE_INTENT, which lists "code" as a bare word
# and so matches the noun in "exit code 137" -- the very phrase this is here
# to let through as a bug report.
_CODE_PRODUCE = re.compile(
    r"(?i)\b(?:write|create|generate|implement|build|make|draft|"
    r"show\s+me|give\s+me|add\s+a|refactor|rewrite|port)\b")

# A machine-emitted error token. Nobody writes these about anything but their
# own running system, so one is enough on its own.
_Q_ERROR_TOKEN = re.compile(
    r"(?:[A-Z][A-Za-z]*(?:Error|Exception)\b|ECONN\w+|ENOENT|EACCES|"
    r"exit(?:ed|s)?\s+(?:with\s+)?(?:code\s+)?\d+|segmentation fault|"
    r"\bsegfault\b|\btraceback\b|\bstack\s*trace\b|\bcore dumped\b)")

# A thing that only exists in a system that is running. This is the other half
# of the possessive test above: "the deploy failed again" and "build is broken"
# name no owner at all, and they are unmistakably bug reports because a deploy
# and a build are not subjects anyone discusses in the abstract. Together the
# two cover the cases a length test was standing in for, without its
# arbitrariness.
_Q_SYSTEM_SUBJECT = re.compile(
    r"(?i)\b(?:build|builds|server|deploy(?:ment)?|test|tests|suite|ci|cd|"
    r"pipeline|container|docker|image|pod|cluster|node|job|cron|worker|daemon|"
    r"script|app|application|service|endpoint|api|route|query|request|"
    r"import|install|compile|compiler|linker|migration|database|db|cache|"
    r"function|method|module|package|dependency|venv|env|process|thread|"
    r"connection|socket|port|binary|model|gpu|cpu|memory|disk)\b")
_Q_HELP_SHAPE = re.compile(
    r"(?i)\b(?:where\s+do\s+i|what(?:'?s| is)\s+(?:going\s+)?wrong|why\s+"
    r"(?:is|does|did|would|won'?t|isn'?t|doesn'?t)|"
    r"how\s+do\s+i\s+(?:fix|debug|diagnose|solve)|any\s+ideas?|"
    r"what\s+(?:should|do)\s+i\s+do|help|fix\s+(?:this|it)|debug|"
    r"find\s+(?:the\s+)?(?:bug|issue|problem|cause|fault))\b")
# Whose thing is broken. The symptom gate below also accepted a message of 40
# characters or more, and a bug report is routinely shorter than that: "my
# docker build hangs at step 4" is 31, so the very example this classifier was
# written for fell through the symptom branch and came out `factual`/`direct`
# -- no plan, no steps, the one-shot answer decomposition exists to avoid.
# Length cannot be the test, because "why did the roman empire fail" is 29 and
# is not a bug report, and "what is a memory leak" is a definition. What
# separates them is whose thing is failing: a bug report says it is the user's.
_Q_SYMPTOM_MINE = re.compile(
    r"(?i)(?:^|\b)(?:my|our|i'?m|i\s+am|we'?re|we\s+are|"
    r"i\s+(?:get|got|keep|see|saw|ran|run|tried)|"
    r"we\s+(?:get|got|keep|see|saw|ran|run|tried))\b")

# Quantitative work. Separate from the analytical cues because a word problem
# contains none of them, and because a small model made to show its arithmetic
# one step at a time gets the answer right far more often than one asked for it
# in a single pass. "estimate" is explicit only: a bare "how many" is the shape
# of "how many days in a week", which is a fact, not a Fermi problem.
_Q_MATH_ASK = re.compile(
    r"(?i)\b(?:calculat\w+|comput\w+|work\s+(?:it|this)\s+out|"
    r"solve\s+(?:for|this|the)|prov(?:e|ing)\b|proof\s+that|disprove|"
    r"deriv(?:e|ing|ation)|integral|derivative|factorial|"
    r"permutations?|combinations?|"
    # "difference" is NOT in this list. "What is the difference between a list
    # and a tuple" is the commonest comparative phrasing in the language, and
    # with it here the maths branch ran first and answered it with a plan, a
    # generation per step and a synthesis. The arithmetic sense is rare enough
    # that _Q_QUANTITY picks it up when real numbers are present.
    r"what(?:'?s|\s+is|\s+are)\s+the\s+(?:sum|total|average|mean|median|"
    r"product|percentage|percent|ratio|probability|odds|area|volume|"
    r"expected\s+value)|"
    # Arithmetic stated rather than named: "what is 17% of 4823 plus the
    # square root of 900" asks for a calculation in every word except the
    # ones a cue list looks for.
    r"\d+\s*(?:%|per\s?cent)\s+of\b|square\s+root|cube\s+root|"
    r"raised\s+to\s+the|to\s+the\s+power\s+of)\b")
_Q_PROOF = re.compile(
    r"(?i)\b(?:prov(?:e|ing)\b|proof\s+that|disprove|deriv(?:e|ing|ation)|solve\s+(?:for|this))\b")
_Q_ESTIMATION = re.compile(
    r"(?i)\b(?:estimat\w+|ballpark|roughly\s+how|approximately\s+how|"
    r"order\s+of\s+magnitude|back[- ]of[- ]the[- ]envelope|fermi)\b")
# The shape of a word problem: quantities that interact, and a question.
_Q_QUANTITY = re.compile(
    r"(?i)\d[\d,.]*\s*(?:%|percent|km/?h?|kmh|mph|mi(?:les)?|kg|lbs?|"
    r"[$€£]|dollars?|euros?|pounds?|[ap]m|"
    r"sec(?:onds?)?|min(?:utes?)?|hours?|hrs?|days?|weeks?|months?|years?)\b")
_Q_QUANTITY_VERB = re.compile(
    r"(?i)\b(?:leaves?|left|travels?|arrives?|meets?|costs?|gives?|each|per|"
    r"total|split|share[sd]?|rate|speed|faster|slower|apart|combined|"
    r"altogether|remaining|how\s+(?:much|many|long|far|fast))\b")
_DIGIT_GROUP = re.compile(r"\d[\d,.]*")

# A question whose answer needs another answer first: a relative clause
# standing in for an entity the user has not named, or two wh-questions chained
# through a shared subject. Structurally invisible to any cue list, and the
# biggest source of confidently wrong one-shot answers here -- the model
# resolves the inner clause wrongly and never notices, because it never
# separated the two lookups.
_Q_MULTIHOP = re.compile(
    r"(?i)\b(?:the|that|which|whose|whatever|whoever)\s+(?:\w+\s+){0,3}?"
    r"(?:that|who|which|where)\s+(?:\w+\s+){0,6}?"
    r"(?:won|wrote|made|built|directed|founded|invented|discovered|created|"
    r"holds?|held|owns?|owned|runs?|ran|leads?|led|borders?|contains?|"
    r"produced|released|published|designed|starred|beat|replaced|preceded)\b"
    # A wh-question anchored to an EVENT that has to be dated first: "who was
    # president when the berlin wall fell". The past-tense auxiliary is what
    # makes it one. Without it this alternation was "any wh-question
    # containing the word `when`", and `when` is overwhelmingly a plain
    # conditional in the questions people actually ask here -- "what happens
    # when you divide by zero", "what should i do when the disk fills up",
    # "where do i put the config when using docker compose" all came out
    # multi_hop/deep and bought a plan, a generation per step and a synthesis
    # for a one-line answer. A present-tense clause after `when` describes a
    # situation; only a past one names an event with a date to look up.
    r"|\b(?:who|what|which|where)\s+(?:\w+\s+){0,2}?(?:was|were|did|had)\b"
    r"[^?]{5,}\b(?:when|while|during|"
    r"after|before|at\s+the\s+time)\b[^?]{5,}"
    r"|\b(?:who|what|which|where|when)\b[^?]{5,}\band\b[^?]{5,}"
    r"\b(?:who|what|which|his|her|their|its|that\s+same)\b")

# Producing a plan: an ordered set of choices where the later ones depend on
# the earlier. Exactly the shape decomposition is for. "design" needs a
# technical object, or "design a birthday card message" lands here instead of
# on creative writing.
_Q_PLANNING = re.compile(
    r"(?i)\b(?:plan(?:ning)?\s+(?:a|an|the|my|our|out)|itinerary|roadmap|"
    r"strategy\s+for|approach\s+(?:for|to)|game\s*plan|"
    r"design(?:ing|ed)?\s+(?:a|an|the)\s+(?:\w+\s+){0,2}?"
    r"(?:system|schema|api|architecture|pipeline|service|database|db|"
    r"migration|index|cache|queue|scheduler|protocol|algorithm|"
    r"rate\s*limiter|data\s*model|layout|workflow)|"
    r"architect(?:ure|ing|ed)\b|migration\s+plan|rollout|"
    r"break\s+(?:this|it)\s+down|step[- ]by[- ]step|"
    r"think\s+(?:through|it\s+through)|reason\s+through|walk\s+me\s+through)\b")

# Asking to be told which to pick. Different from a comparison: a comparison is
# finished once both sides are described, a decision is not finished until one
# is chosen, and the two used to be handled identically.
_Q_DECISION = re.compile(
    r"(?i)(?:\b(?:should|shall)\s+(?:i|we|you|they)\b|"
    r"\bwhich\s+(?:one\s+)?(?:is|are|would|should|do|does|makes)\b|"
    r"\bwhich\s+(?:\w+\s+){0,2}?(?:should|would|is\s+bett|is\s+best)|"
    r"\b(?:is|are)\s+it\s+worth\b|\bworth\s+(?:it|the)\b|"
    r"\b(?:what|which)\s+(?:would\s+you|do\s+you)\s+recommend\b|"
    r"\brecommend(?:ation)?\s+(?:for|between)\b|"
    r"\bbetter\s+(?:choice|option|fit|bet)\b|\bpick\s+(?:one|between)\b)")
_Q_ALTERNATIVES = re.compile(r"(?i)\s(?:or|vs\.?|versus)\s")

_Q_COMPARISON = re.compile(
    r"(?i)\b(?:compar(?:e|es|ed|ing|ison|isons)|contrast(?:s|ed|ing)?|"
    r"versus|vs\.?|trade[- ]?offs?|pros\s+and\s+cons|"
    r"difference[s]?\s+between|differ\s+from|"
    r"(?:better|faster|cheaper|safer|simpler)\s+than|"
    r"advantages?\s+(?:and\s+disadvantages?\s+)?(?:of|over)|"
    r"instead\s+of)\b")

_Q_ANALYTICAL = re.compile(
    r"(?i)\b(?:analy[sz]\w+|evaluat\w+|assess(?:es|ed|ing|ment)?|"
    r"implications?|consequences?|ramifications?|"
    r"critique|criticis\w+|weigh\s+(?:up\s+)?the|"
    r"what\s+(?:would|will)\s+happen\s+if|what\s+if\s+(?:i|we|you|they|it))\b")

_Q_CAUSAL = re.compile(
    r"(?i)\b(?:why\s+(?:is|are|was|were|does|do|did|would|should|can'?t|"
    r"cannot|don'?t|doesn'?t|didn'?t|not|the|my|it|this|that)|"
    r"what\s+cause[sd]?|root\s+cause|how\s+come|"
    r"reasons?\s+(?:why|for|behind)|explain\s+why|"
    r"justif(?:y|ies|ied|ication)\s+(?:the\s+)?(?:choice|decision|use|cost|"
    r"approach|design|claim)|account\s+for\s+the)\b")

_Q_PROCEDURAL = re.compile(
    r"(?i)\b(?:how\s+(?:do|would|can|should|to)\s+(?:i|we|you|one)?|"
    r"how\s+to\b|steps?\s+to\b|guide\s+(?:to|for)|tutorial|"
    r"instructions?\s+for|what(?:'?s|\s+is)\s+the\s+(?:best\s+)?way\s+to|"
    r"how\s+(?:does|do|did)\s+\w+\s+work|"
    # "explain how tcp handshakes work" -- the same question with the
    # auxiliary dropped, which the does/do/did form cannot see.
    r"how\s+(?:\w+\s+){1,3}?works?\b)")

_Q_DEFINITION = re.compile(
    r"(?i)^(?:\s*(?:so|ok(?:ay)?|and|but|hey)[,\s]+)?"
    r"(?:define\b|definition\s+of\b|meaning\s+of\b|"
    r"what\s+does\s+[\w\-]+\s+mean\b|"
    r"what(?:'?s|\s+is|\s+are)\s+(?:a|an|the)?\s*[\w\-]+(?:\s+[\w\-]+){0,2}\s*\??$)")


class QuestionShape(NamedTuple):
    """What kind of question this is, and how much work to spend on it.

    `cues` carries the detectors that fired. It is not decoration: the trace
    has to be able to say WHY a turn was decomposed, because that is the
    difference between a user moving the thresholds and a user turning the
    whole feature off.
    """
    kind: str
    depth: str
    cues: tuple

    @property
    def deep(self) -> bool:
        return self.depth == "deep"


def _reasoning_heft(text: str, signals: int, min_chars: int) -> bool:
    """The existing depth bar, unchanged, reusable by kind.

    Lifted verbatim out of is_reasoning_question so the two cannot drift: what
    reasoning_signals and reasoning_min_chars mean must not depend on which
    kind the message was sorted into first.
    """
    found = REASONING_SIGNAL.findall(text)
    if found and len(found) >= max(1, signals):
        return True
    if re.search(r"step by step|think through|walk me through", text, re.I):
        return True
    clauses = text.count(" and ") + text.count(", ") + text.count("?")
    return bool(found) and (len(text) >= min_chars or clauses >= 2)


def classify_question(message: str, signals: int = 2,
                      min_chars: int = 80) -> QuestionShape:
    """Sort a message into a kind and a depth. Pure, and no model call.

    `signals` and `min_chars` are the same two dials is_reasoning_question
    exposes and mean the same thing, so a machine tuned for depth gets more
    depth out of every kind rather than only out of the analytical ones.

    The heft test applies to the kinds that are only SOMETIMES worth the full
    treatment -- a comparison of two things in six words is not -- and is
    skipped for the kinds that essentially always are. A word problem, a
    chained lookup, a plan and a bug report are structurally multi-step however
    tersely they are put.
    """
    full = " ".join((message or "").strip().split())
    if not full:
        return QuestionShape("chitchat", "direct", ())
    if _is_trivial(full):
        return QuestionShape("chitchat", "direct", ("trivial",))
    # Match against a bounded prefix, but measure heft against the whole
    # message. What a message ASKS FOR is in its opening lines; what follows is
    # the material it asks about, and a pasted file can be tens of thousands of
    # characters. Scanning all of it with a dozen patterns costs real
    # milliseconds on every single turn -- measured at 210ms for one pattern on
    # a 6000-character input, on a path that runs before anything else happens.
    # The length and clause counts below still read the full text, because that
    # is exactly what reasoning_min_chars is asking about.
    text = full[:CLASSIFY_SCAN_CHARS]

    # A pasted traceback before anything else. is_code_request fires on one --
    # it is full of filenames, "module" and "line" -- so the single commonest
    # bug report there is came out `code`/`direct` and was answered as though
    # the user had asked for a program. A machine error token means something
    # broke; only an explicit request to WRITE something outranks it, which is
    # what keeps "write a function that raises ValueError" a code request.
    if (_Q_ERROR_TOKEN.search(text) and not _CODE_PRODUCE.search(text)
            and not _Q_DEFINITION.match(text)):
        # The definition guard is the same one the symptom gate below carries:
        # "what is a segfault" names an error token and is asking what one IS.
        return QuestionShape("debug", "deep", ("error_token",))
    if is_code_request(text):
        # Code has its own lane, budget and task state, and a program is never
        # improved by a model this size planning it in prose first.
        return QuestionShape("code", "direct", ("code",))
    if _Q_TRANSFORM.search(text):
        return QuestionShape("summarize", "direct", ("transform",))
    if _Q_CREATIVE.search(text):
        return QuestionShape("creative", "direct", ("creative",))
    if not _Q_DEFINITION.match(text) and (
            _Q_ERROR_TOKEN.search(text)
            or (_Q_SYMPTOM.search(text)
                and (_Q_SYMPTOM_MINE.search(text)
                     or (_Q_SYSTEM_SUBJECT.search(text)
                         and not _Q_PROCEDURAL.search(text))))):
        # A fault signal plus evidence that the failing thing is the user's.
        # Neither half works alone. A fault word on its own makes "why did the
        # roman empire fail" and "what is a memory leak" into bug reports; the
        # length test that used to stand in for this made the flip pure
        # character count, so "my docker build hangs at step 4" (31 chars) was
        # a plain factual question while the same sentence nine characters
        # longer was a bug report -- and that sentence is the example this
        # classifier was written for. Ownership answers it: a bug report says
        # whose thing broke, either with a possessive or by naming something
        # that only exists in a running system. An error token needs neither,
        # because nobody pastes ECONNREFUSED about the Roman empire.
        #
        # The definition guard is for "what is a memory leak": a fault word
        # and a system noun, and yet plainly a request for a definition
        # rather than a report that one happened.
        #
        # And the procedural guard is the same idea one step along. The
        # system-subject half is a stand-in for ownership, but its word list
        # is broad on purpose ("request", "function", "test", "memory",
        # "disk"), so a fault word anywhere in a how-to matched it: "how do i
        # fix a memory leak", "how do i retry a request when it times out"
        # and "how to stop docker hanging on the build cache" were all read
        # as reports of a live failure and decomposed. A how-to is a request
        # for instructions, not a report -- the procedural branch below says
        # so in as many words and never gets the chance. It only guards the
        # system-subject arm: "how do i fix MY docker build that hangs" still
        # names whose thing is broken, and is still a bug report.
        return QuestionShape("debug", "deep", ("symptom",))

    stripped = text.rstrip("=?").strip()
    if (ARITHMETIC_ONLY.match(stripped)
            and any(c.isdigit() for c in stripped)
            and any(op in stripped for op in "+-*/^%")):
        # Bare arithmetic is the calculator's job, not a five-step plan.
        return QuestionShape("math", "direct", ("arithmetic",))
    if _Q_MATH_ASK.search(text) or _Q_ESTIMATION.search(text):
        # Deep only when there is real work in it. "What's 17% of 340" is one
        # operation, and planning it, spending a generation per step and then
        # synthesising is slower than answering AND more likely to be wrong.
        # Three or more quantities, an estimation or proof, or simply enough
        # sentence to hold several moving parts.
        worth_it = (len(_DIGIT_GROUP.findall(text)) >= 3
                    or bool(_Q_ESTIMATION.search(text))
                    or bool(_Q_PROOF.search(text))
                    or len(text) >= 40)
        return QuestionShape("math", "deep" if worth_it else "direct",
                             ("math_ask",))
    if (len(_DIGIT_GROUP.findall(text)) >= 2 and _Q_QUANTITY.search(text)
            and _Q_QUANTITY_VERB.search(text)):
        return QuestionShape("math", "deep", ("word_problem",))

    if _Q_MULTIHOP.search(text):
        return QuestionShape("multi_hop", "deep", ("multi_hop",))
    if _Q_PLANNING.search(text):
        return QuestionShape("planning", "deep", ("planning",))
    if _Q_DECISION.search(text):
        # Naming the alternatives is what makes it a real deliberation.
        # "Should I restart it?" is a question; "should I use postgres or
        # sqlite" is a decision with two branches to work through.
        deep = bool(_Q_ALTERNATIVES.search(text)) or _reasoning_heft(
            full, signals, min_chars)
        return QuestionShape("decision", "deep" if deep else "short", ("decision",))

    heavy = _reasoning_heft(full, signals, min_chars)
    for name, pattern in (("comparative", _Q_COMPARISON),
                          ("analytical", _Q_ANALYTICAL),
                          ("causal", _Q_CAUSAL),
                          ("procedural", _Q_PROCEDURAL)):
        if pattern.search(text):
            if name == "procedural":
                # A how-to is a list of steps, not a chain of conclusions.
                # "How do I install Homebrew, and do I need Xcode first?" used
                # to clear the bar on length alone and got a plan, six
                # generations and a synthesis for a three-line answer.
                return QuestionShape("procedural", "short", (name,))
            return QuestionShape(name, "deep" if heavy else "short", (name,))

    if _Q_ALTERNATIVES.search(text) and len(text.split()) <= 8:
        # "postgres or mysql" -- no verb, no cue, and unmistakably a request
        # to weigh two things against each other.
        return QuestionShape("comparative", "short", ("alternatives",))
    if _Q_DEFINITION.match(text):
        return QuestionShape("definition", "direct", ("definition",))
    return QuestionShape("factual", "direct", ())


# Protocol scaffolding a model wraps its answer in, which must never reach the
# user. Two shapes, both measured on this hardware rather than guessed at:
#
#   <final_answer>...</final_answer>     an XML tag invented by symmetry with the
#                                        <think>...</think> tags the reasoning
#                                        instruction asks for
#   **Answer**: ...  /  Final answer: ...  a label, sometimes dressed as markdown
#
# Both come from the same place: an instruction that describes XML tags and then
# says "give your final answer". The prompt no longer phrases it that way, but a
# 3B model treats prompt wording as a suggestion, so the output is cleaned too.
_ANSWER_TAG = re.compile(
    r"(?is)\A\s*<\s*(final[_ ]?answer|answer|response|output)\s*>\s*(.*?)"
    r"(?:\s*<\s*/\s*\1\s*>\s*)?\Z")
# A label at the very START of the reply only. A mid-text "Answer:" is somebody
# formatting their prose, and rewriting that would be worse than the leak.
#
# Two forms, because the model produces both: an inline label ending in a colon,
# and a markdown heading on its own line. The heading form requires the word
# "final", since a bare "## Answer" section heading is plausibly the user's own
# structure whereas "## Final Answer" is the protocol talking.
_ANSWER_LABEL = re.compile(
    r"(?i)\A\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*final[_ ]?answer\s*(?:\*\*|__)?\s*[:\-]\s*"
    r"|\A\s*(?:#{1,6}\s*)?(?:\*\*|__)\s*answer\s*(?:\*\*|__)\s*[:\-]\s*")
_ANSWER_HEADING = re.compile(
    r"(?i)\A\s*(?:#{1,6}\s*|\*\*|__)\s*final[_ ]?answer\s*(?:\*\*|__)?\s*:?[ \t]*\r?\n+")


def unwrap_answer(text: str) -> str:
    """Strip protocol scaffolding a model wrapped its answer in.

    Applied where an answer becomes the reply, not where it is generated, so
    every lane is covered by one rule: the tool loop, the prose lane, the
    continuation, a subagent's result and the plain non-agent chat all end up
    here.

    Conservative by design. Tags are removed only when one WRAPS the whole
    reply, and a label only at the very start, because "<final_answer>" inside
    an answer about XML, or an "Answer:" halfway down a formatted reply, are the
    user's content rather than protocol noise.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned
    for _ in range(3):        # a tag inside a label, or the reverse
        before = cleaned
        match = _ANSWER_TAG.match(cleaned)
        if match:
            cleaned = match.group(2).strip()
        cleaned = _ANSWER_HEADING.sub("", cleaned, count=1)
        cleaned = _ANSWER_LABEL.sub("", cleaned, count=1).strip()
        # A JSON object whose only content is the answer. parse_tool_call
        # ignores it (no tool name) so it used to be shown to the user raw.
        if cleaned.startswith("{") and cleaned.endswith("}"):
            try:
                parsed = json.loads(cleaned)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                keys = {str(k).lower() for k in parsed}
                if keys and keys <= {"answer", "final_answer", "response", "text"}:
                    values = [str(v) for v in parsed.values() if isinstance(v, str)]
                    if values:
                        cleaned = "\n".join(values).strip()
        # A dangling close tag left by a wrapper whose open tag was in the
        # reasoning block, or by truncation.
        cleaned = re.sub(
            r"(?is)\s*<\s*/\s*(final[_ ]?answer|answer|response|output)\s*>\s*\Z",
            "", cleaned).strip()
        if cleaned == before:
            break
    return cleaned or (text or "").strip()


# The visible marker appended to a reply that hit the token budget. It is the
# ONLY durable record that an answer was cut off: the conversation stores the
# text, not the finish_reason, so a "continue" arriving in a later request has
# nothing else to go on. Generation and detection therefore share one definition,
# and the pattern is loose about the punctuation so notes written by an older
# build still match.
TRUNCATION_NOTE = re.compile(r"\n*\[cut off at the \d+-token reply limit[^\]]*\]\s*$")


def truncation_note(limit: int, requested: int = 0, context_size: int = 0) -> str:
    """The marker note_if_cut appends. Keep in sync with TRUNCATION_NOTE.

    The advice has to be true. "Raise Max tokens in Settings" was printed on
    turns where the harness had already widened the user's 512 to 1536 on its
    own and the real ceiling was the context window, so following it changed
    nothing. When the effective limit is not the user's setting, say which one
    actually bit.
    """
    if requested and limit > requested:
        reason = (f" \u2014 your Max tokens is {requested}, already widened to "
                  f"{limit} for code; ask me to continue")
    elif context_size and limit + 256 >= context_size:
        reason = (f" \u2014 the {context_size}-token context window is the limit "
                  "here, not Max tokens; ask me to continue")
    else:
        reason = " \u2014 raise \"Max tokens\" in Settings, or ask me to continue"
    return f"\n\n[cut off at the {limit}-token reply limit{reason}]"


def was_truncated(text: str) -> bool:
    """True if this stored assistant message ends in the truncation marker."""
    return bool(TRUNCATION_NOTE.search(text or ""))


def append_below_answer(text: str, extra: str) -> str:
    """Append a footnote while keeping the truncation note last.

    was_truncated anchors to the END of the string, and it is the only evidence
    the continuation lane has that an answer stopped at the budget rather than
    at its natural end. Anything appended after it therefore does not merely
    look untidy: it turns "continue" back into an ordinary question.
    """
    body = text or ""
    if not (extra or "").strip():
        return body
    match = TRUNCATION_NOTE.search(body)
    if not match:
        return body.rstrip() + extra
    return body[:match.start()].rstrip() + extra + "\n" + match.group(0).strip()


def strip_truncation_note(text: str) -> str:
    """Drop the marker.

    Called on the way INTO every prompt as well as on the continuation path: the
    marker is UI copy addressed to the user ("raise Max tokens in Settings"), and
    feeding it back as conversation invites the model to imitate it.
    """
    return TRUNCATION_NOTE.sub("", text or "").rstrip()


# A short "carry on" follow-up. Loose about wording, strict about length: the
# caller requires that the previous answer was ACTUALLY truncated as well, so
# both signals must agree before a turn is treated as a continuation and a long
# instruction that merely opens with "continue" is never hijacked.
# A continuation request carries NO new instruction, and the whole string has to
# be consumed to prove it. As a prefix match this hijacked real work: "more
# tests please", "finish the parser" and "go ahead and add retries" were all
# read as "resume the previous answer", so whenever the previous reply had been
# cut off -- which on a small local model is most of them -- the user's actual
# instruction was dropped on the floor and never reached the model.
CONTINUE_REQUEST = re.compile(
    r"(?i)^\s*(?:please\s+|ok(?:ay)?[,\s]+|yes[,\s]+|and\s+|just\s+)*"
    r"(?:continue|carry\s+on|keep\s+going|go\s+on|go\s+ahead|resume|"
    r"finish\s+(?:it|that|this|up)?|(?:the\s+)?rest|more)"
    # Only continuation-shaped tails, never a new object.
    r"(?:\s+(?:of\s+(?:it|that|this)|from\s+where\s+(?:you|it)\s+\w+|"
    r"where\s+you\s+left\s+off|with\s+(?:it|that|this)|for\s+me|now|"
    r"pl(?:ease|z)|then))*"
    r"[\s,.!]*$")


def is_continue_request(message: str) -> bool:
    """True for a short "continue" style follow-up. Pair with was_truncated.

    A full match, not a prefix one: anything trailing the continue word is a new
    instruction, and resuming instead of reading it silently discards it.
    """
    text = (message or "").strip()
    if not text or len(text) > 80:
        return False
    return bool(CONTINUE_REQUEST.match(text))


# A question ABOUT code rather than a request to change it. The failure this
# exists for: mid-task, "whats the logic behind SetHardwareInfo(...)" is a
# request for an explanation, but every gate upstream saw an active code task
# and briefed the model to re-emit the whole file -- into a 512-token chat
# budget, so it was cut off mid-function and the question was never answered.
_EXPLAIN_LEAD = re.compile(
    r"(?i)^\s*(?:and\s+|so\s+|but\s+|ok(?:ay)?[,\s]+|please\s+|"
    r"can\s+you\s+|could\s+you\s+)*"
    r"(?:what(?:'?s| is| are| does| do| was| were)?|why|how\s+(?:does|do|did|is|are|"
    r"come)|explain|describe|walk\s+me\s+through|tell\s+me\s+(?:about|why|how|what)|"
    r"where\s+(?:does|do|is|are)|when\s+(?:does|do|is|are)|which\s+\w+\s+(?:does|do|is|are)|"
    r"is\s+(?:that|this|it|there)|are\s+(?:those|these|they)|does\s+(?:that|this|it)|"
    r"do\s+(?:those|these|they)|should\s+(?:i|we|that|this|it))\b")

# Asking for a change, even when the sentence opens like a question. "can you
# add error handling" is a modification; "what does this do" is not.
_EXPLAIN_NOT = re.compile(
    r"(?i)\b(add|append|extend|include|implement|rewrite|refactor|port|convert|"
    r"translate|fix|correct|change|modify|update|remove|delete|drop|replace|"
    r"rename|split|merge|optimi[sz]e|simplify|harden|improve|make\s+it|"
    r"give\s+me|show\s+me\s+the\s+(?:code|file|script)|write)\b")


def is_explanation_request(message: str) -> bool:
    """True for a question ABOUT the work rather than a change to it.

    Gates two things that have to agree: whether the reply budget is the chat
    budget, and whether the task brief demands the whole file back. They
    disagreed, and a question mid-task got a "reply with the complete updated
    file" instruction on a 512-token budget.

    Deliberately conservative: a message that also asks for a change ("why is
    that slow, can you fix it") is a change, because answering it as prose only
    would drop half the request.
    """
    text = " ".join((message or "").split())
    if not text or len(text) > 400:
        return False
    if not _EXPLAIN_LEAD.match(text) and "?" not in text:
        return False
    if _EXPLAIN_NOT.search(text):
        return False
    # A quoted code fragment is normal in "what does this do: <code>", so a
    # fenced block is not by itself evidence of a code request here.
    return bool(_EXPLAIN_LEAD.match(text) or text.rstrip().endswith("?"))


# Pointing at work already in progress: a pronoun, or a definite reference to
# the thing being built. Used to keep the router's hands off a follow-up.
_ACTIVE_TASK_REFERENCE = re.compile(
    r"(?i)(?:^|\b)(?:it|its|it'?s|that|this|these|those|them|"
    r"the\s+(?:\w+\s+){0,2}(?:script|code|file|program|programme|module|"
    r"function|port|thing|one|artifact|above)|"
    r"(?:the|your|my)\s+(?:last|previous|earlier|first|second)\b|"
    r"what\s+(?:you|we)\s+(?:wrote|produced|made|had|did))\b")


def refers_to_active_task(message: str) -> bool:
    """True when a short message is about work already under way.

    Length-bounded: a long message carries enough of its own subject that the
    router can judge it, while "how about the c port" carries none at all and
    becomes a literal two-word search query the moment it leaves this context.
    """
    text = " ".join((message or "").split())
    if not text or len(text.split()) > 14:
        return False
    if _URL_IN_TEXT.search(text):
        return False                    # a link is a lookup, whatever else it says
    if is_time_sensitive(text):
        # "the latest version of X" points at the world, not at the artifact,
        # however definite the article is.
        return False
    return bool(_ACTIVE_TASK_REFERENCE.search(text))


# "code" is both a verb in CODE_INTENT and a noun in CODE_OBJECT, so "summarize
# the code", "use my code-review prompt" and "connect Claude Code to this app"
# each counted as a code verb plus a code object and were answered as requests
# to write code. In a product name, in a hyphenated compound and as the object
# of a summary it is the noun, so the VERB check ignores it there; the object
# check still sees it ("fix this code"). Narrow on purpose: "show me the code"
# and "review this code:" before an unfenced paste kept their old reading,
# since nothing reported them as wrong.
_CODE_NOUN = re.compile(
    r"(?i)\b(?:claude|vs|visual\s+studio)\s+code\b"
    r"|\bcode(?=-[a-z])"
    r"|\b(?:summari[sz]e|summary\s+of|overview\s+of)\s+(?:the|my|our|your)\s+code\b")
# A credential is created with a verb CODE_INTENT knows, often for a script
# CODE_OBJECT knows: "create an api token for my script" is account admin, not
# code. Masked only when no language is named ("generate an rsa key in python"
# stays code) and only when the credential is the object itself, so "create a
# password reset flow for my app" still is.
_CREDENTIAL_ASK = re.compile(
    r"(?i)\b(?:create|generate|make|issue|revoke|rotate)\s+(?:(?:an?|the|my|new|another)\s+)*"
    r"(?:(?:api|access|bearer|auth|personal)[- ]?)?(?:tokens?|keys?|passwords?)\b"
    r"(?!\s+(?:validation|validator|field|check|checker|hashing|hash|generator|reset|strength|"
    r"policy|input|form|manager|rotation|class|function|method|column|table|model|handling|"
    r"logic|support|auth\w*|parser|store|storage)\b)")


def is_code_request(message: str) -> bool:
    """True if the message asks to write, fix, or modify code.

    Requires a code verb plus either a code object (script, function, ...) or a
    language name, or a fenced code block in the message. Kept deliberately
    strict so "write up the latest news" (verb, but no code object or language)
    does not match and can still be routed to a search.
    """
    text = message or ""
    if "```" in text:
        return True
    has_language = bool(CODE_LANGUAGE.search(text) or CODE_C_LANGUAGE.search(text))
    if not has_language:
        text = _CREDENTIAL_ASK.sub(" ", text)
    has_verb = bool(CODE_INTENT.search(_CODE_NOUN.sub(" ", text)))
    has_object = bool(CODE_OBJECT.search(text))
    # A code verb plus an object or a language ("write a python script"), or a
    # language and an object together even without an imperative verb ("python
    # script to pull CVEs"), both count as a code request.
    return (has_verb and (has_object or has_language)) or (has_language and has_object)


# Result lines from _web_search look like "N. Title\n   URL\n   snippet". Pull
# the result URLs in order so the retrieval pipeline can fetch the top ones.
_RESULT_URL = re.compile(r"^\s*(https?://\S+)\s*$", re.M)


# Aggregator, listing and JS-shell URLs whose fetched HTML is mostly navigation
# and script, not article text. Extracting from them wastes a fetch-and-read
# cycle and, on 8GB, the junk-filled prompt is what stalls. Skip them and use the
# next result (or the search snippet) instead.
_LOW_VALUE_HOST = re.compile(
    r"(?:^|\.)(?:news\.google\.|news\.yahoo\.|flipboard\.|reddit\.com|"
    r"twitter\.com|x\.com|facebook\.com|pinterest\.|quora\.com)", re.I)
_LOW_VALUE_PATH = re.compile(
    r"/(?:category|categories|tag|tags|topics?|section|sections|feed|feeds|"
    r"latest|trending|search|archive)(?:/|$|\?)", re.I)


def is_low_value_url(url: str) -> bool:
    """True for aggregator/listing/JS-shell pages unlikely to yield article text."""
    try:
        from urllib.parse import urlparse
        parts = urlparse(url)
    except Exception:
        return False
    host = parts.netloc.lower()
    path = parts.path or "/"
    if _LOW_VALUE_HOST.search(host):
        return True
    if _LOW_VALUE_PATH.search(path):
        return True
    # A bare domain root (no real path) is a homepage/shell, not an article.
    if path in ("", "/") and not parts.query:
        return True
    return False


def extractable_text_len(page: str) -> int:
    """Rough count of article-like text in a fetched page (already tag-stripped).

    Aggregator shells strip down to a pile of short link fragments; a real
    article has long prose lines. Count characters only from lines that read like
    prose (long, or sentence-punctuated) so a wall of two-word nav links scores
    near zero even when the raw length is large.
    """
    total = 0
    for line in (page or "").splitlines():
        line = line.strip()
        if len(line) >= 60 or (len(line) >= 30 and any(c in line for c in ".!?")):
            total += len(line)
    return total


def is_thin_page(page: str, min_chars: int = 400) -> bool:
    """True if a fetched page has too little real prose to be worth extracting."""
    return extractable_text_len(page) < min_chars


def top_result_urls(search_text: str, limit: int) -> list[str]:
    """The first `limit` result URLs from a web_search result block, de-duped."""
    seen: list[str] = []
    for match in _RESULT_URL.finditer(search_text or ""):
        url = match.group(1)
        if url not in seen:
            seen.append(url)
        if len(seen) >= limit:
            break
    return seen


# Result blocks look like "N. Title\n   URL\n   snippet" (see _web_search). This
# splits a result block into (title, url, snippet) records in engine order.
_RESULT_HEAD = re.compile(r"^\s*\d+\.\s+(.*)$")
# The user pointing at their own material: their project, a repo, a path, a
# filename. A code request WITHOUT any of these ("write a basic c++ crud
# program") wants code from the model's own knowledge, and pulling indexed
# documents into it only spends context and derails the answer.
# "my code-review prompt" is a compound noun, not "my code": the bare "code"
# branch refuses a following hyphen or word character.
_PROJECT_REFERENCE = re.compile(
    r"(\bthis (?:project|repo|repository|codebase|file|function|class|module|"
    r"script|code|bug|error|test|package)\b"
    r"|\b(?:my|our) (?:project|repo|repository|codebase|code[ -]?base|code(?![\w-])|file|"
    r"files|script|scripts|module|function|class|tests?)\b"
    r"|\b(?:in|from|of|open|read|edit|patch|review) (?:the )?(?:file|files|repo|"
    r"repository|codebase|project)\b"
    r"|\b[\w-]+\.(?:py|js|ts|tsx|jsx|rs|go|c|h|cc|cpp|hpp|cs|java|rb|php|swift|"
    r"kt|sh|zsh|sql|html|css|json|toml|yaml|yml|md|txt|ipynb)\b"
    r"|[~.]?/[\w.-]+/[\w.-]+)",
    re.I,
)


# A web address is somebody else's page, never the user's own files: "add
# https://example.com/docs to the knowledge base" matched the two-segment path
# branch on "//example.com/docs", and "example.com/a/b" without a scheme the
# same way. Addresses are blanked before the project patterns run.
_PROJECT_NOT_A_PATH = re.compile(_BARE_DOMAIN.pattern + r"(?:/[^\s]*)?")


def refers_to_project(message: str) -> bool:
    """True if the message points at the user's own files, repo or project."""
    text = _PROJECT_NOT_A_PATH.sub(" ", _URL_IN_TEXT.sub(" ", message or ""))
    return bool(_PROJECT_REFERENCE.search(text))


def _parse_search_results(search_text: str) -> list[tuple[str, str, str]]:
    """Parse a web_search result block into (title, url, snippet) records."""
    records: list[tuple[str, str, str]] = []
    title = url = ""
    snippet_lines: list[str] = []

    def flush() -> None:
        if url:
            records.append((title, url, " ".join(snippet_lines).strip()))

    for raw in (search_text or "").splitlines():
        head = _RESULT_HEAD.match(raw)
        if head:
            flush()
            title, url, snippet_lines = head.group(1).strip(), "", []
            continue
        urlm = _RESULT_URL.match(raw)
        if urlm and not url:
            url = urlm.group(1)
            continue
        if raw.strip():
            snippet_lines.append(raw.strip())
    flush()
    return records


def rank_result_urls(search_text: str, query: str, limit: int) -> list[str]:
    """Rerank a web_search result block by relevance before auto-fetching.

    Engine order is a weak prior; this promotes results whose title/snippet
    actually overlap the question and demotes aggregator/listing hosts, so the
    bounded number of pages we spend a fetch-and-read on are the most likely to
    carry the answer. Falls back to engine order when there is nothing to score
    (no query tokens, or an unparseable block).
    """
    records = _parse_search_results(search_text)
    if not records:
        return top_result_urls(search_text, limit)
    tokens = _query_tokens(query)

    def score(title: str, url: str, snippet: str, rank: int) -> float:
        s = -0.1 * rank  # keep engine order as the tiebreak
        if tokens:
            tl = title.lower()
            sn = snippet.lower()
            s += 2.0 * sum(1 for t in tokens if t in tl)
            s += 1.0 * sum(1 for t in tokens if t in sn)
        if is_low_value_url(url):
            s -= 5.0
        return s

    # Sort on (-score, rank) so a higher score wins and engine order breaks ties.
    scored: list[tuple[float, int, str]] = []
    for rank, (title, url, snippet) in enumerate(records):
        scored.append((-score(title, url, snippet, rank), rank, url))
    scored.sort()
    out: list[str] = []
    for _neg, _rank, url in scored:
        if url not in out:
            out.append(url)
        if len(out) >= limit:
            break
    return out


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into ~size-character chunks with a little overlap.

    Overlap keeps a sentence that straddles a boundary from being lost to both
    chunks. Prefers to break on a newline or space near the boundary so chunks
    fall on natural seams rather than mid-word.
    """
    text = text or ""
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Back off to the nearest newline or space within the last 15%.
            window = text.rfind("\n", start + int(size * 0.85), end)
            if window == -1:
                window = text.rfind(" ", start + int(size * 0.85), end)
            if window != -1:
                end = window
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def quick_tool(message: str) -> tuple[str, dict] | None:
    """Deterministic shortcuts that need no model call at all.

    Only two cases qualify: a message that is purely a URL, and one that is
    purely arithmetic. Both are unambiguous and common enough that spending a
    router call on them would be wasteful. Everything else returns None and is
    handled by the model router.
    """
    text = (message or "").strip()
    if not text or len(text) > 400:
        return None
    url = BARE_URL.match(text)
    if url:
        return "fetch_url", {"url": url.group(1)}
    expression = text.rstrip("=?").strip()
    # Require an operator and a digit so a bare number or a year ("2026") is not
    # mistaken for arithmetic.
    if (ARITHMETIC_ONLY.match(expression) and any(op in expression for op in "+-*/^%")
            and any(char.isdigit() for char in expression)):
        return "calculator", {"expression": expression.replace("^", "**").replace(",", "")}
    return None


# A URL anywhere in a message, and the words that signal "go read this page and
# tell me about it" (as opposed to merely mentioning a link). Used to send
# "summarise https://…" to fetch_url instead of letting the router guess.
_URL_IN_TEXT = re.compile(r'https?://[^\s<>()\[\]{}"\']+', re.I)
# Prefix stems with a LEADING word boundary only (no trailing \b): "summar"
# must match summary/summarize/summarise, so a trailing boundary would break it.
URL_READ_INTENT = re.compile(
    r"\b(?:summar|tl;?dr|read|open|fetch|retriev|scrap|extract|explain|describ|"
    r"analy[sz]e|review|brief|digest|go to|look at|check|"
    r"what(?:'s| is| are| does| say)|contents? of|summary of)",
    re.I)


def url_read_request(message: str) -> str | None:
    """The single URL to fetch when a message asks to read/summarise a page.

    A message that is nothing but a URL is handled by quick_tool (it returns the
    page verbatim). This covers the very common "summarise <url>", "read this:
    <url>", "what does <url> say", "tl;dr <url>", where the user wants the model
    to fetch the page and then answer *from* it. Returns None unless there is
    exactly one URL and a clear read/summarise intent in a short instruction, so
    an ordinary message that merely mentions a link falls through to the router.
    """
    text = (message or "").strip()
    if not text:
        return None
    urls = _URL_IN_TEXT.findall(text)
    if len(urls) != 1:
        return None
    url = urls[0].rstrip('.,;:!?)\'"')
    rest = _URL_IN_TEXT.sub(" ", text).strip()
    if not rest:
        return None  # a bare URL: quick_tool handles it
    if len(rest.split()) <= 20 and URL_READ_INTENT.search(rest):
        return url
    return None


# Power-user routing overrides: a leading slash command that forces a lane and
# bypasses the model router entirely. Deterministic and explicit, so the user is
# never surprised by the router's judgement when they have already made the call.
# Lanes: "web_search" (force a search), "answer" (never search; own knowledge +
# knowledge base), "kb" (answer, but say the knowledge base is the source),
# "agent" (skip the router and go straight to the tool loop).
_OVERRIDE = re.compile(
    r"^\s*/(search|web|websearch|nosearch|no-search|answer|local|kb|docs|"
    r"agent|tools|act)\b[ \t]*",
    re.I)
_OVERRIDE_LANE = {
    "search": "web_search", "web": "web_search", "websearch": "web_search",
    "nosearch": "answer", "no-search": "answer", "answer": "answer", "local": "answer",
    "kb": "kb", "docs": "kb",
    "agent": "agent", "tools": "agent", "act": "agent",
}


def routing_override(message: str) -> tuple[str, str] | None:
    """Parse a leading /command override. Returns (lane, cleaned_message) or None.

    The command is stripped from the message so the downstream tool or the model
    sees only the real request ("/search foo bar" -> ("web_search", "foo bar")).
    """
    m = _OVERRIDE.match(message or "")
    if not m:
        return None
    lane = _OVERRIDE_LANE[m.group(1).lower().replace("-", "")]
    rest = (message[m.end():]).strip()
    return lane, rest


# Unambiguous "search the web" imperatives. Unlike TIME_SENSITIVE (a soft hint
# that a lookup might help), these are direct commands to search, so they route
# straight to web_search with no model-router call. Kept strict: a bare
# "search X" often means the attached codebase, so a web marker or a
# search-engine name is required -- except when the verb IS a search engine.
_SEARCH_LEAD = re.compile(
    r"^\s*(?:please\s+)?(?:can|could|would)?\s*(?:you\s+)?"
    r"(?:"
    r"web[- ]?search(?:\s+for)?|"
    r"search\s+(?:the\s+)?(?:web|internet|online)(?:\s+for)?|"
    r"search\s+up|"
    r"google|bing|duckduckgo|ddg"
    r")\b[:,]?\s+",
    re.I)
# A trailing "... online" / "on the web" that turns a plain lookup into a search.
_ONLINE_TRAIL = re.compile(r"\b(?:online|on the web|on the internet)\b\s*[.?!]*$", re.I)
_LOOKUP_LEAD = re.compile(r"^\s*(?:please\s+)?(?:find|look\s+up|search\s+for)\b\s+", re.I)


def web_search_request(message: str) -> str | None:
    """The query to search for when a message is an explicit web-search command.

    Covers "search the web for X", "google X", "web search X", and "find/look up
    X online". Returns None for anything not unambiguously a web search, so an
    ordinary question still goes through the model router.
    """
    text = (message or "").strip()
    if not text or len(text) > 400:
        return None
    m = _SEARCH_LEAD.match(text)
    if m:
        return text[m.end():].strip(" .?!\t") or None
    # "find X online" / "look up X on the web": a lookup verb with a web marker.
    if _LOOKUP_LEAD.match(text) and _ONLINE_TRAIL.search(text):
        q = _LOOKUP_LEAD.sub("", text, count=1)
        return _ONLINE_TRAIL.sub("", q).strip(" .?!\t") or None
    return None


# ---------------------------------------------------------------- searching
#
# web_search_request above is the STRICT form and stays exactly as it is: a
# search imperative at the start of the sentence, or a lookup verb with a
# trailing web marker. Measured against twenty phrasings people actually type
# to ask for a search, it caught three.
#
# The misses were not exotic. "check online for the price" puts the web marker
# at the FRONT, where the trailing-marker rule cannot see it. "look it up",
# "do a web search", "cite your sources", "verify that" and "any recent news
# on this" carry no query at all, and a function whose contract is "return the
# query or None" has no way to say "yes, search, and the terms are in the
# previous turn" -- so it said None, the router answered from the weights, and
# the word the user typed went unanswered above the reply. The user asking in
# plain English is the least ambiguous signal this pipeline ever gets, and it
# was the one being dropped.
#
# So: a wider matcher, and a third return value. None means not an explicit
# ask; a string means search for this; the EMPTY string means search, and the
# subject is whatever we were just talking about.
_SEARCH_VERB = re.compile(
    r"(?i)\b(?:web[- ]?search|search\s+(?:the\s+)?(?:web|internet|online)|"
    r"search\s+up|search\s+for|google|bing|duckduckgo|ddg|"
    r"look\s+(?:it|this|that|them)\s+up|look\s+up|look\s+online|"
    r"check\s+online|check\s+the\s+(?:web|internet|news)|"
    r"browse|go\s+(?:and\s+)?(?:find|look))\b")
# Asking for evidence rather than for a search, which amounts to the same
# instruction: the model cannot cite what it has not fetched.
_SEARCH_EVIDENCE = re.compile(
    r"(?i)\b(?:sources?|citations?|references?|"
    r"fact[- ]?check|"
    r"is\s+(?:that|this|it)\s+still\s+(?:true|right|correct|accurate|current)|"
    r"any\s+(?:recent\s+)?news|what\s+does\s+the\s+internet\s+say|"
    r"according\s+to\s+the\s+(?:web|internet))\b")
# "verify" and "double-check" ask for a check, and only sometimes for a check
# against the world. "Can you double-check my math?" and "verify my
# understanding of closures" are asking about work already on the screen, and
# reading them as search commands sent the PREVIOUS turn's text to DuckDuckGo
# and skipped the router entirely -- the reflexive-search leak, reintroduced
# through the one door that bypasses every other guard. So they count only when
# what is to be checked is not the user's own material.
_SEARCH_CHECK = re.compile(r"(?i)\b(?:verify|double[- ]?check|confirm)\b")
_CHECK_OWN_WORK = re.compile(
    r"(?i)\b(?:my|our|your|this|that|the)\s+(?:\w+\s+){0,2}?"
    r"(?:math|maths|arithmetic|calculation|sums?|logic|reasoning|understanding|"
    r"work|answer|reply|code|script|function|draft|writing|spelling|grammar|"
    r"assumption|interpretation|approach|solution|proof)\b")
# The one direction this must stay strict in. A lookup verb pointed at the
# user's OWN material is not a web search: "search my project for the auth
# handler" and "look up the price in my spreadsheet" both name where to look,
# and it is not the internet.
_SEARCH_OWN = re.compile(
    r"(?i)\b(?:my|our|your|this|these|the\s+attached|the\s+uploaded|"
    r"the\s+above)\s+(?:\w+\s+){0,2}?"
    r"(?:codebase|repo|repository|project|file|files|folder|directory|draft|"
    r"document|docs|notes|kb|knowledge\s*base|code|source|transcript|"
    r"spreadsheet|conversation|history|inbox|emails?)\b"
    # ... or at a file the user named. "look up the value of model_cutoff in
    # config.py" reached DuckDuckGo: a filename is as clear a statement of
    # where to look as "my repo" is, and it is the form people actually use.
    r"|\b(?:in|from|inside)\s+[\w./-]+\.[a-z]{1,5}\b"
    r"|\b[\w-]+\.(?:py|js|ts|tsx|jsx|go|rs|rb|java|c|h|cpp|cs|swift|kt|php|sh|"
    r"sql|json|ya?ml|toml|ini|cfg|conf|env|md|txt|csv|log)\b")
# What is left of "browse for it" once the verb is stripped: a pronoun, not a
# query. The subject lives in the previous turn.
#
# Pronouns and politeness only. "the <word>" was in this list and should never
# have been: "look up the weather" and "search for the president" name exactly
# what to search for, and fullmatching them threw the user's own query away and
# searched the previous turn instead.
_BARE_REFERENT = re.compile(
    r"(?i)\s*(?:it|this|that|these|those|them|"
    r"please|for\s+me|again|too|as\s+well)?\s*")

# The user asking whether or how THEY can do something, with a lookup verb as
# the verb of that question rather than as an instruction. "can you look this
# up" is a request; "can i search up a key in a dict" is a question about
# dictionaries. Deliberately "i" and not "i|we": "can we look up the price"
# includes the assistant in the actor and reads as a request, so widening the
# subject would refuse a search the user did ask for -- the worse failure of
# the two, and the one _WEATHER_TOPIC's comment already argues about.
_SEARCH_ASKED_ABOUT = re.compile(
    r"(?i)\s*(?:so|ok(?:ay)?|hey|and|but)?[,\s]*"
    r"(?:how|where|why|when|can|could|should|would|do|did|is\s+there)\s+"
    r"(?:\w+\s+){0,2}?i\b")


# Weather has its own tool, and its spec says to use it for any weather
# question because search snippets for weather describe weather WEBSITES
# rather than the forecast. "Look up the weather in Paris tomorrow" is a
# lookup verb in front of a weather question, and answering it here took the
# decision away from the router, which would have picked the right tool.
# Deliberately only the words that essentially never mean anything else in a
# lookup. "temperature", "rain" and "wind" were here too and blocked "the
# temperature coefficient of copper", "the melting temperature of tungsten"
# and "the rain man cast" -- refusing a legitimate search is a worse failure
# than occasionally searching for a forecast, because the weather tool is a
# nicety and the search is the whole request.
_WEATHER_TOPIC = re.compile(
    r"(?i)\b(?:weather|forecast)\b"
    r"|\bhow\s+(?:hot|cold|warm)\s+(?:is|will)\s+it\b"
    r"|\b(?:temperature|humidity|wind\s+speed)\b(?=[^.?!]*\b(?:today|tomorrow|"
    r"tonight|outside|right\s+now|this\s+(?:week|weekend|morning|afternoon))\b)")


def explicit_search_request(message: str) -> str | None:
    """The query when the user has plainly asked for a lookup, else None.

    Returns the empty string -- not None -- when the ask carries no query of
    its own ("look it up", "sources?", "verify that"). Empty means "search,
    and I have no terms for you"; the caller resolves the subject from the
    previous turn. That is a different answer from "do not search", and
    collapsing the two is what made half these phrasings unroutable.

    The strict matcher runs first, so nothing that worked before changes, and
    the wider rules only ever add.
    """
    text = " ".join((message or "").strip().split())
    if not text or len(text) > 400:
        return None
    # A question about DOING something is not an instruction to search for
    # it. _SEARCH_VERB lists "browse", "search up" and "google" as bare
    # verbs, and matching them anywhere in the first 24 characters turned
    # "how do i browse the filesystem in python" and "can i search up a key
    # in a dict in O(1)" into web searches of their own leftovers -- a fetch
    # spent, and the question the user asked never answered. The test is the
    # SUBJECT: "can you look up X" and "could you google X" are requests and
    # stay requests; "can I", "how do I", "should I" are the user asking
    # about their own action.
    if _SEARCH_ASKED_ABOUT.match(text):
        return None
    strict = web_search_request(text)
    if strict:
        # The same bare-referent test the verb path below applies. It lived
        # only there, so "look it up" correctly returned "" and asked the
        # caller to resolve the subject from the previous turn while "google
        # it" -- which the strict matcher catches first and returns from --
        # sent the literal string "it" to DuckDuckGo. Two spellings of one
        # request, two different answers.
        if _BARE_REFERENT.fullmatch(strict):
            return ""
        return strict
    # Looking in the user's own material is not looking on the web.
    if _SEARCH_OWN.search(text):
        return None
    if _WEATHER_TOPIC.search(text):
        return None      # the router picks the weather tool for these
    verb = _SEARCH_VERB.search(text)
    if verb:
        # Near the start only. "I googled it already and it said X" is a
        # report of a search, not an instruction to run one, and treating the
        # two alike turns every mention of the web into a fetch.
        if verb.start() > 24:
            return None
        rest = text[verb.end():].strip(" .,:;?!\t")
        rest = re.sub(r"^(?:for|about|on|up|into)\s+", "", rest, flags=re.I).strip()
        rest = _ONLINE_TRAIL.sub("", rest).strip(" .,:;?!\t")
        # "browse for it", "look up that" -- the residual is a pronoun, which
        # is not a query. Sending it to a search engine returns nothing useful
        # and costs a fetch; the subject is in the previous turn, which is
        # exactly what the empty string tells the caller.
        if _BARE_REFERENT.fullmatch(rest):
            return ""
        return rest
    # "sources?", "is that still true?" -- an instruction only when it is the
    # whole message. Buried in a paragraph it is a turn of phrase.
    if len(text) <= 120:
        if _SEARCH_EVIDENCE.search(text):
            return ""
        if _SEARCH_CHECK.search(text) and not _CHECK_OWN_WORK.search(text):
            return ""
    return None


# What may stand in for a query the user did not type.
#
# explicit_search_request returns "" for "look it up" / "sources?" / "verify
# that", and the caller fills the subject in from the previous turn. That turn
# was being used VERBATIM, and it is the one string in the conversation that
# nobody has checked: every guard in this module -- _SEARCH_OWN, the code test,
# the project test -- is applied to the message the user just typed, and none
# of them to the message being sent in its place. So a turn that pasted a
# config, a traceback or a private document and a follow-up of "can you
# double-check that?" put the paste, secrets and all, into a DuckDuckGo query.
# The guards have to follow the text, not the turn it arrived on.
_PASTED_MATERIAL = re.compile(
    r"```"                                   # a fenced block
    r"|^\s*(?:[A-Za-z_][A-Za-z0-9_]*\s*[=:]\s*\S)"  # key=value / key: value
    r"|^\s*(?:Traceback|File \"|\s+at\s+\S+\()"     # a stack trace
    r"|^\s*(?:def |class |import |from \w+ import|\$ |> |#!/)",
    re.M)


def search_subject(previous: str | None) -> str | None:
    """The previous turn reduced to something safe to send to a search engine.

    None when it is not safe to send at all. A search query is a short subject
    in natural language; anything that reads as the user's OWN material --
    pasted code or output, a request about their project, a file, a document --
    is not a query and must never leave the machine just because the follow-up
    happened to be two words long. Answering from the weights is the correct
    behaviour there, and the caller treats None exactly as "not an explicit
    search", which is what it was before this substitution existed.
    """
    raw = previous or ""
    # The length bar first, and deliberately: it is the cheapest test, it
    # rules out the most, and everything below it is a regex. A previous turn
    # is a whole user message and can be a megabyte of pasted file, which cost
    # half a second here before this ran first.
    #
    # A long turn has no single subject to search for anyway, and its first
    # 200 characters are an arbitrary slice of the user's prose rather than a
    # query they would recognise.
    if len(raw) > 4000:
        return None
    text = " ".join(raw.strip().split())
    if not text or len(text) > 200:
        return None
    if _PASTED_MATERIAL.search(raw):
        return None
    if _SEARCH_OWN.search(text) or refers_to_project(text) or is_code_request(text):
        return None
    return text


# Freshness, with a calendar rather than a word list.
#
# TIME_SENSITIVE is a keyword list and has no idea what year it is, so "who won
# the 2026 champions league final" and "who won the 1998 world cup" were the
# same message to it -- and since today's date has never appeared in any prompt
# here, the model could not tell them apart either. A year the weights cannot
# have seen is the single most reliable freshness signal there is, and it was
# being thrown away.
_RECENT_YEAR = re.compile(r"\b(20[2-9]\d)\b")
# Any four-digit year, so a HISTORICAL one can rule freshness out as firmly as
# a recent one rules it in.
#
# 2000-2039, not 2000-2099. The whole 20xx range makes a year out of every
# four-digit round number this application is full of: "set max tokens to
# 2048", "will a 2080 ti be fast enough", "the nfs mount is on port 2049",
# "i get a timeout after 2050 ms". Each one read as a year at or after the
# cutoff, which is the branch that hard-returns True, so the freshness guard
# silently switched itself off and permitted a web lookup on a question about
# a config value. Nobody asks a chat assistant about the year 2080; the range
# only has to cover years a person might actually name.
_ANY_YEAR = re.compile(r"\b(1[89]\d{2}|20[0-3]\d)\b")
# A version string, which almost always means "the state of this thing now".
_VERSION_TOKEN = re.compile(
    r"(?i)\b(?:v|ver\.?|version\s*)?\d+\.\d+(?:\.\d+)?\b"
    r"|\b[a-z][a-z0-9]*\d+(?:[.-][a-z0-9]+){1,3}\b")
_VERSION_CONTEXT = re.compile(
    r"(?i)\b(?:release[ds]?|chang\w*|release\s*notes|out|"
    r"support|supported|breaking|deprecat\w*|new\s+in|what'?s\s+(?:in|new)|"
    r"migrat\w*|upgrad\w*|available|ship(?:ped|ping)?|announc\w*|"
    r"added|removed|fixed|differ\w*|compare[ds]?|vs\.?)\b")
# Questions about who or what currently holds a position, and whether
# something has happened yet. None of these need the word "latest" in them.
_HOLDER_QUESTION = re.compile(
    r"(?i)\b(?:who\s+(?:is|are|'?s)\s+(?:the\s+)?(?:current|new)?\s*\w*\s*"
    r"(?:ceo|cto|president|chair|chairman|prime\s+minister|champion|holder|"
    r"leader|manager|owner|winner)|"
    r"who\s+won\b|who\s+holds\b|"
    r"is\s+\w+\s+still\b|are\s+\w+\s+still\b|"
    r"did\s+\w+\s+(?:release|ship|launch|announce|win|buy|acquire|merge)\b|"
    r"when\s+(?:did|does|will)\s+\w+\s+(?:come\s+out|release|ship|launch)|"
    r"what\s+happened\s+(?:at|to|with)\b|"
    r"out\s+yet\b|release\s+date|"
    r"how\s+much\s+(?:is|does|did)\s+\w+\s+(?:cost|charge))\b")
# The technical senses of TIME_SENSITIVE's own keywords, every one of which
# points at the user's code rather than at the world. Without this the word
# "current" in "the current working directory" spent a web search.
#
# Two lists, because the two need different strength. The first are senses that
# simply are not about the world, whatever else the sentence says. The second
# are words with a perfectly good world sense that only become technical when
# they are pointed at something the user owns -- "the current state of my
# draft" is code, "the current state of the Ukraine conflict" is news, and the
# first version of this suppressed BOTH because `state` was on one flat list.
_FRESH_TECHNICAL = re.compile(
    r"(?i)\b(?:current\s+(?:working\s+)?(?:directory|dir|path|branch|commit|"
    r"keyword|node|element|scope|thread)|"
    r"release\s+(?:candidate|build|mode)|"
    r"current\s+through|electric\s+current|alternating\s+current|direct\s+current|"
    r"version\s+control)\b")
_FRESH_TECHNICAL_OWN = re.compile(
    r"(?i)\bcurrent\s+(?:\w+\s+){0,2}?"
    r"(?:value|user|context|state|implementation|design|behaviou?r|approach|"
    r"setup|code|version|config(?:uration)?|schema|layout)\s+"
    r"(?:of|in|for)\s+(?:my|our|this|the|your)\s+"
    r"(?:\w+\s+){0,2}?"
    r"(?:code|codebase|draft|file|files|script|project|repo|repository|document|"
    r"function|app|program|branch|config|module|class|page|sheet|test|tests)\b")


def needs_fresh_facts(message: str, cutoff: str = "") -> bool:
    """True when answering well needs facts newer than the model's weights.

    `cutoff` is the model's training cutoff as "YYYY-MM" (config.model_cutoff),
    or "" when it is not known. Unknown is treated conservatively: any recent
    year counts, because the alternative is answering a question about this
    year out of weights that stopped two years ago while giving the user no
    sign that anything is missing.

    A superset of is_time_sensitive, minus its technical false positives. The
    thin wrapper is kept and unchanged for every existing caller.
    """
    text = " ".join((message or "").strip().split())
    if not text:
        return False
    if _FRESH_TECHNICAL.search(text) or _FRESH_TECHNICAL_OWN.search(text):
        return False
    # A year at or after the cutoff settles it: the weights cannot hold it.
    #
    # A year BEFORE the cutoff only demotes, and only when nothing else in the
    # message asks for something current. It is strong evidence on its own --
    # "who won the 1998 world cup" reads as a recency question to every
    # keyword test and the answer has not moved in decades -- but it is not
    # evidence at all when the year is a quantity or a vintage rather than the
    # subject. "How much does a 1965 Mustang cost today" and "today's price for
    # 2000 shares of AAPL" both name an old number and ask about right now, and
    # deciding on the year alone refused the lookup on both.
    years = _ANY_YEAR.findall(text)
    floor = str(cutoff)[:4] if cutoff else "2000"
    if years and max(years) >= floor:
        return True
    if years and not TIME_SENSITIVE.search(text):
        return False
    if _HOLDER_QUESTION.search(text):
        return True
    if _VERSION_TOKEN.search(text) and _VERSION_CONTEXT.search(text):
        return True
    return bool(TIME_SENSITIVE.search(text))


# Backwards-compatible alias. Older call sites and tests refer to fast_path_call;
# it now covers only the deterministic shortcuts. The prev_user parameter is kept
# for signature compatibility but is unused, because follow-up resolution ("look
# it up") is now handled by the model router, which sees the conversation.
def fast_path_call(message: str, prev_user: str | None = None) -> tuple[str, dict] | None:
    return quick_tool(message)



# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'CODE_C_LANGUAGE',
    'CODE_INTENT',
    'CODE_LANGUAGE',
    'CONTINUE_REQUEST',
    'TRUNCATION_NOTE',
    'is_continue_request',
    'unwrap_answer',
    'truncation_note',
    'was_truncated',
    'strip_truncation_note',
    'append_below_answer',
    'CODE_NEEDS_LOOKUP',
    'CODE_OBJECT',
    'REASONING_SIGNAL',
    'TIME_SENSITIVE',
    'URL_READ_INTENT',
    '_URL_IN_TEXT',
    'url_read_request',
    'routing_override',
    'web_search_request',
    '_OVERRIDE',
    '_SEARCH_LEAD',
    '_ONLINE_TRAIL',
    '_LOOKUP_LEAD',
    '_CODE_FRAMING',
    '_LOW_VALUE_HOST',
    '_LOW_VALUE_PATH',
    '_NEG_LEAD',
    '_NEG_PHRASE',
    '_NOT_FEEDBACK',
    '_FEEDBACK_TAIL',
    '_WITHDRAWN',
    '_POS_LEAD',
    '_POS_PHRASE',
    '_RESULT_URL',
    '_is_trivial',
    '_last_user_turn',
    'build_reusable_dataset',
    'chunk_text',
    'classify_implicit_feedback',
    'rejection_retry_message',
    'code_search_topic',
    'extractable_text_len',
    'fast_path_call',
    'is_code_request',
    'is_explanation_request',
    'refers_to_active_task',
    '_ACTIVE_TASK_REFERENCE',
    '_EXPLAIN_LEAD',
    '_EXPLAIN_NOT',
    'is_low_value_url',
    'is_reasoning_question',
    'is_substantive',
    'is_thin_page',
    'is_time_sensitive',
    'document_request',
    'document_needs_research',
    '_LOOKUP_ASK',
    '_BARE_DOMAIN',
    'document_subject',
    'starts_new_document',
    '_DOC_SUBJECT',
    '_SUBJECT_STOP',
    '_subject_words',
    '_DOC_FORMAT_WORDS',
    '_DOC_TARGET',
    '_DOC_OTHER',
    '_DOC_READ',
    '_DOC_FILENAME',
    '_DOC_TARGET_EXTRA',
    '_DOC_MAKE',
    'missing_capability',
    'classify_question',
    'QuestionShape',
    'QUESTION_KINDS',
    'QUESTION_DEPTHS',
    'explicit_search_request',
    'search_subject',
    'needs_fresh_facts',
    'is_hedged_answer',
    'NEED_MARKER',
    'is_missing_draft_reply',
    'NO_DRAFT_MAX_CHARS',
    '_NO_DRAFT_REPLY',
    'REFUSAL_MAX_CHARS',
    '_NEEDS_LOOKUP',
    '_NEEDS_FILES',
    'is_followup_remark',
    '_FOLLOWUP_LEAD',
    'quick_tool',
    'top_result_urls',
    'rank_result_urls',
    '_parse_search_results',
    '_query_tokens',
    'rag_query_tokens',
    '_RAG_GENERIC',
    'refers_to_project',
    '_PROJECT_REFERENCE',
    '_PROJECT_NOT_A_PATH',
    '_CODE_NOUN',
    '_CREDENTIAL_ASK',
    '_RESULT_HEAD',
]
