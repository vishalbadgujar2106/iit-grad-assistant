"""
rag.py
Core retrieval-augmented generation logic:
  1. Rank the knowledge-base chunks against the student's question (BM25)
  2. Build a grounded prompt from the best matches
  3. Call Claude and return the answer plus the sources used, so the
     frontend can show "answered from: ..."

Retrieval is plain BM25 keyword ranking in pure Python. With a knowledge base
this small (a few dozen chunks) it matches what the old embedding model +
ChromaDB setup retrieved, without shipping PyTorch or a vector database,
which is what makes the backend small enough to run serverless.
"""

import logging
import math
import os
import re
from collections import Counter
from functools import lru_cache

from dotenv import load_dotenv

from ingest import build_chunks, load_documents

load_dotenv()

log = logging.getLogger(__name__)

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = 1024
TOP_K = 5  # how many chunks to send to the model per question
# Chunks scoring below this fraction of the best match are dropped, so a
# sharp question doesn't drag in loosely related sections.
RELATIVE_SCORE_CUTOFF = 0.35
HISTORY_TURNS = 6  # most recent messages from the conversation to include

SYSTEM_PROMPT = """You are GradGuide, a graduate student assistant for \
Illinois Institute of Technology (Illinois Tech). You are an independent \
student project, not an official Illinois Tech service. Students ask you about OPT \
and STEM OPT, CPT, on-campus jobs, course registration, full-time status, and \
financial aid. Most of them \
are international graduate students on F-1 or J-1 visas, often checking \
something quickly between classes.

Each question arrives with reference notes taken from Illinois Tech's Office \
of Global Services (OGS), Graduate Academic Affairs, Student Employment, \
Student Accounting, and Financial Aid pages. Those notes are \
your only verified source. State facts about Illinois Tech, immigration rules, \
deadlines, fees, and procedures only when the notes support them. If the notes \
cover only part of the question, answer that part and say plainly what you \
don't have reliable information on. If they don't cover it at all, say so in \
a sentence and point the student to the right office: OGS for anything about \
visas, OPT, or CPT; the Registrar for registration; Student Accounting or \
Financial Aid for money. Never fill gaps from general knowledge, and never make \
up URLs, phone numbers, email addresses, dates, or fee amounts. That includes \
helpful-sounding extras: don't add tips, consequences, job titles to contact, \
or how a process "usually" works unless the notes say it, and don't stretch a \
rule to cover more than the notes state. A shorter answer that sticks to the \
notes is better than a fuller one that guesses. If the notes don't mention a \
rule, say you don't have information on it rather than saying the rule doesn't \
exist. Read the notes carefully before saying something isn't covered.

Never estimate or speculate about dates or timing in a student's own case, \
such as when an office entered, processed, or approved something for them, \
how many days they have left, or how much earlier a clock "may have" \
started. You can't see their records. State the rule as the notes give it, \
and if the answer depends on a date in the student's record, tell them to \
confirm that exact date with the office that holds it (for SEVIS and OPT \
dates, OGS) before they act.

Write the way a knowledgeable senior student would explain it to a classmate: \
warm, plain, and direct. Put the direct answer in the first sentence, then add \
only the details that matter for this student's situation. Most answers should \
be under about 150 words. Talk as if you simply know Illinois Tech's guidance; \
don't refer to "the notes", "the context", "documents", or "sources".

The chat window shows plain text, so keep formatting minimal. Use short \
paragraphs. Use a numbered list only for steps that happen in order, and a \
short hyphen list only for three or more parallel items. Don't use headings, \
bold, italics, horizontal rules, block quotes, tables, or emojis. Don't use \
em dashes or en dashes; use commas, periods, colons, or parentheses instead.

When a question touches on keeping visa status (dropping below full time, \
starting or changing work, deadlines that affect status), end with one short \
sentence suggesting they confirm with OGS before acting, and only one: don't \
repeat the OGS suggestion elsewhere in the answer. Skip that for \
questions that don't affect status, such as how to use the portal or what a \
TA does. The app already shows a general "not legal advice" disclaimer, so \
don't add one. When a date changes every semester, mention that the exact \
date should be checked on the current OGS or Academic Calendar page.

Earlier messages in the conversation may be included; use them to understand \
follow-ups like "what about OPT?". For greetings or questions outside these \
topics, reply briefly and mention what you can help with."""

# Words too common to say anything about which chunk is relevant.
STOPWORDS = frozenset("""
a about after all also am an and any are as at be been before being but by
can could did do does doing for from get got had has have how i if in into
is it its just me my of on or our should so than that the their them then
there these they this to up us was we were what where which while who
why will with would you your yours im dont need want
""".split())

# A few everyday phrasings mapped onto the vocabulary the docs actually use.
SYNONYMS = {
    "cost": ["fee"], "costs": ["fee"], "price": ["fee"], "pay": ["fee"],
    "expensive": ["fee"], "money": ["funding", "fee"],
    "job": ["employment", "work"], "jobs": ["employment", "work"],
    "internship": ["cpt"], "intern": ["cpt"],
    "class": ["course"], "classes": ["course"],
    "credit": ["credit", "hour"], "units": ["credit"],
    "scholarship": ["funding"], "assistantship": ["ta", "ga"],
    "ta": ["teaching", "assistantship"], "tas": ["teaching", "assistantship"],
    "ra": ["research", "assistantship"], "ras": ["research", "assistantship"],
    "ga": ["assistantship"],
    # Ties a bare "OPT" to the OPT doc's title (Optional Practical Training);
    # many docs mention OPT in passing, so the acronym alone ranks weakly.
    "opt": ["optional"],
    "visa": ["status", "sevis"], "deadline": ["deadline", "date"],
    "drop": ["drop", "below"], "enroll": ["register"], "enrol": ["register"],
    "apply": ["request", "submit"], "when": ["date", "earliest", "timeline"],
    "approve": ["processing"], "approval": ["processing"],
}


def _stem(word):
    """Very light suffix stripping so 'credits'/'credit' and 'dropping'/'drop' match."""
    for suffix in ("ing", "ies", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            if suffix == "ies":
                return word[:-3] + "y"
            if suffix == "s" and word.endswith("ss"):
                return word
            stem = word[: -len(suffix)]
            # "dropping" -> "dropp" -> "drop", but keep "enroll", "pass".
            if suffix in ("ing", "ed") and stem[-1] == stem[-2] and stem[-1] not in "lsz":
                stem = stem[:-1]
            return stem
    return word


def tokenize(text, expand=False):
    # Hyphens split words, so "full-time" in the docs matches "full time".
    words = re.findall(r"[a-z0-9]+", text.lower())
    tokens = []
    for w in words:
        # Single letters are leftovers like the "s" in "what's"; digits stay
        # because "9 credits" or "691" matter.
        if w in STOPWORDS or (len(w) == 1 and not w.isdigit()):
            continue
        tokens.append(_stem(w))
        if expand:
            tokens.extend(_stem(s) for s in SYNONYMS.get(w, []))
    return tokens


class BM25Index:
    """Okapi BM25 over a list of chunks. Small and dependency-free."""

    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        # Index the doc title and section header alongside the body so a
        # question like "CPT application steps" hits the CPT 'Application
        # Steps' section rather than the OPT one.
        self.doc_tokens = [
            tokenize(f"{c['title']} {c['section']} {c['text']}") for c in chunks
        ]
        self.doc_freqs = [Counter(toks) for toks in self.doc_tokens]
        self.avg_len = sum(len(t) for t in self.doc_tokens) / max(len(chunks), 1)
        df = Counter(term for toks in self.doc_tokens for term in set(toks))
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query_tokens, i):
        freqs, length = self.doc_freqs[i], len(self.doc_tokens[i])
        total = 0.0
        for term in query_tokens:
            tf = freqs.get(term)
            if not tf:
                continue
            norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
            total += self.idf[term] * tf * (self.k1 + 1) / norm
        return total

    def search(self, query, top_k):
        query_tokens = tokenize(query, expand=True)
        scored = [(self.score(query_tokens, i), i) for i in range(len(self.chunks))]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(reverse=True)
        if not scored:
            return []
        floor = scored[0][0] * RELATIVE_SCORE_CUTOFF
        return [
            {**self.chunks[i], "score": round(s, 3)}
            for s, i in scored[:top_k]
            if s >= floor
        ]


@lru_cache(maxsize=1)
def get_index():
    chunks = build_chunks(load_documents())
    if not chunks:
        raise RuntimeError("No knowledge base documents found in data/.")
    return BM25Index(chunks)


def retrieve(question: str, top_k: int = TOP_K, history=None):
    """
    Return the top_k most relevant chunks. For follow-ups ("what about
    OPT?") the student's previous message is added to the search so the
    topic carries over.
    """
    query = question
    if history:
        prior_user = [m["content"] for m in history if m["role"] == "user"]
        if prior_user:
            # The current question is repeated so it outweighs the (often
            # longer) previous one; BM25 sums repeated query terms.
            query = f"{question} {question} {prior_user[-1]}"
    return get_index().search(query, top_k)


def build_prompt(question: str, chunks: list):
    if chunks:
        notes = "\n\n".join(
            f'<note title="{c["title"]}: {c["section"]}" '
            f'last_verified="{c["verified"]}">\n{c["text"]}\n</note>'
            for c in chunks
        )
    else:
        notes = "(No reference notes matched this question.)"

    return f"""<reference_notes>
{notes}
</reference_notes>

Student's question: {question}"""


def clean_history(history):
    """
    Keep only well-formed recent turns, starting with a user message, since
    the API requires the conversation to open with the user.
    """
    turns = [
        {"role": m["role"], "content": m["content"]}
        for m in (history or [])
        if m.get("role") in ("user", "assistant") and m.get("content", "").strip()
    ][-HISTORY_TURNS:]
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    return turns


@lru_cache(maxsize=1)
def get_client():
    """
    Requires ANTHROPIC_API_KEY to be set in the environment (see
    .env.example). Created once and reused across requests.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    import anthropic

    # Worst case is 2 attempts x 25s = 50s, which stays under the browser's
    # 60s timeout (frontend/index.html) and the function's maxDuration
    # (vercel.json), so the server never keeps paying for an answer the
    # student has already given up on. Typical answers take 4-8s.
    return anthropic.Anthropic(timeout=25.0, max_retries=1)


def generate_answer(question: str, chunks: list, history=None) -> str:
    client = get_client()
    messages = clean_history(history)
    messages.append({"role": "user", "content": build_prompt(question, chunks)})

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    answer = "".join(b.text for b in response.content if b.type == "text").strip()
    if response.stop_reason == "max_tokens":
        log.warning("Answer hit max_tokens (%s) for question: %r", MAX_TOKENS, question)
        answer += "\n\n(That answer got cut off. Ask me to continue if you need the rest.)"
    return answer


def answer_question(question: str, history=None, top_k: int = TOP_K):
    """
    Full pipeline used by the API: retrieve -> generate -> return answer +
    sources, so the frontend can display what the answer was grounded in.
    """
    history = clean_history(history)
    chunks = retrieve(question, top_k=top_k, history=history)
    answer = generate_answer(question, chunks, history)
    sources = sorted({c["source"] for c in chunks})
    return {"answer": answer, "sources": sources, "chunks_used": chunks}


if __name__ == "__main__":
    # Quick manual test: python rag.py "your question here"
    import sys

    q = " ".join(sys.argv[1:]) or "How many credits do I need to stay full-time?"
    result = answer_question(q)
    print("\nQ:", q)
    print("\nA:", result["answer"])
    print("\nSources:", result["sources"])
