"""
eval/evaluate.py
Runs the full RAG pipeline against eval/test_questions.json and scores every
answer. This is what turns the project from "a chatbot that seemed to work
when I tried it" into "a chatbot with measured accuracy."

Each answer gets three checks:
  - grounded: every factual claim is supported by the retrieved notes (or the
    bot correctly says it doesn't know). Graded by Claude.
  - correct:  the answer agrees with the ground truth. Graded by Claude.
  - style:    no em/en dashes, headings, bold, or emojis, and a reasonable
    length. Checked with plain code, no LLM needed.

Usage (needs ANTHROPIC_API_KEY; costs a few cents per run):
    python eval/evaluate.py

Exits with a non-zero status code if any pass rate falls below the
thresholds below, so it can gate a deploy (see .github/workflows/eval.yml).
"""

import json
import os
import re
import sys

# Allow running as `python eval/evaluate.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import answer_question, build_prompt, get_client  # noqa: E402

JUDGE_MODEL = "claude-sonnet-5-5"

# Minimum acceptable pass rates (0-1). Tune these as you gather more real
# data on what "good enough" means for this project.
THRESHOLDS = {
    "grounded": 0.9,
    "correct": 0.8,
    "style": 0.9,
}

MAX_WORDS = 250
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")

JUDGE_PROMPT = """You are grading an answer from a chatbot that helps \
Illinois Tech international graduate students. The chatbot may only use the \
reference notes it was given.

<reference_notes>
{notes}
</reference_notes>

<question>{question}</question>

<ground_truth>{ground_truth}</ground_truth>

<chatbot_answer>
{answer}
</chatbot_answer>

Grade two things:
- grounded: true if every factual claim in the answer (rules, numbers, \
deadlines, offices, URLs) is supported by the reference notes, or the answer \
correctly says it doesn't have that information. Generic advice like "check \
with OGS" counts as grounded. False if it states any fact the notes don't \
support.
- correct: true if the answer agrees with the ground truth and doesn't \
contradict it. For a question the ground truth says is out of scope, true \
only if the chatbot declines rather than answering from general knowledge.

Give a one-sentence reason that names any unsupported or wrong claim."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "grounded": {"type": "boolean"},
        "correct": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["grounded", "correct", "reason"],
    "additionalProperties": False,
}


def style_problems(answer):
    problems = []
    if "—" in answer or "–" in answer:
        problems.append("em/en dash")
    if re.search(r"^\s*#{1,6}\s", answer, re.MULTILINE):
        problems.append("heading")
    if "**" in answer:
        problems.append("bold")
    if EMOJI.search(answer):
        problems.append("emoji")
    if len(answer.split()) > MAX_WORDS:
        problems.append(f"over {MAX_WORDS} words")
    return problems


def judge(question, ground_truth, answer, chunks):
    # Give the judge exactly the notes the chatbot saw, including each note's
    # title and "last verified" line, so it doesn't flag facts that came from
    # those headers as unsupported.
    notes = build_prompt(question, chunks).split("</reference_notes>")[0]
    notes = notes.replace("<reference_notes>", "").strip()
    response = get_client().messages.create(
        model=JUDGE_MODEL,
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                notes=notes, question=question,
                ground_truth=ground_truth, answer=answer,
            ),
        }],
        output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def main():
    test_path = os.path.join(os.path.dirname(__file__), "test_questions.json")
    with open(test_path, "r", encoding="utf-8") as f:
        test_questions = json.load(f)

    print(f"Running {len(test_questions)} test questions ...\n")
    totals = {"grounded": 0, "correct": 0, "style": 0}

    for item in test_questions:
        question = item["question"]
        result = answer_question(question, history=item.get("history"))
        answer = result["answer"]
        verdict = judge(question, item["ground_truth"], answer, result["chunks_used"])
        style = style_problems(answer)

        totals["grounded"] += verdict["grounded"]
        totals["correct"] += verdict["correct"]
        totals["style"] += not style

        flags = "".join([
            "G" if verdict["grounded"] else "-",
            "C" if verdict["correct"] else "-",
            "S" if not style else "-",
        ])
        print(f"[{flags}] {question}")
        if not (verdict["grounded"] and verdict["correct"]):
            print(f"       judge: {verdict['reason']}")
        if style:
            print(f"       style: {', '.join(style)}")
        if flags != "GCS":
            # Show the answer too, so a too-strict judge is easy to spot.
            print("       answer: " + answer.replace("\n", "\n               "))

    n = len(test_questions)
    print("\nPass rates (G = grounded, C = correct, S = style):")
    failed = False
    for metric, threshold in THRESHOLDS.items():
        rate = totals[metric] / n
        ok = rate >= threshold
        failed |= not ok
        print(f"  {metric:9} {rate:.2f}  (threshold {threshold})  {'ok' if ok else 'BELOW'}")

    if failed:
        print("\nEvaluation FAILED: at least one score is below its threshold.")
        sys.exit(1)
    print("\nEvaluation PASSED.")


if __name__ == "__main__":
    main()
