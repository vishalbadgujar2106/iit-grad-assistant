"""
eval/evaluate.py
Runs the full RAG pipeline against eval/test_questions.json and scores every
answer with RAGAS. This is what turns the project from "a chatbot that
seemed to work when I tried it" into "a chatbot with measured accuracy."

Usage:
    python eval/evaluate.py

Exits with a non-zero status code if scores fall below the thresholds below,
so this script can be dropped straight into a CI pipeline (see
.github/workflows/eval.yml) to block a deploy on regressions.
"""

import json
import os
import sys

# Allow running as `python eval/evaluate.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import retrieve, generate_answer  # noqa: E402

# Minimum acceptable average scores (0-1). Tune these as you gather more
# real data on what "good enough" means for this project.
THRESHOLDS = {
    "faithfulness": 0.75,
    "answer_relevancy": 0.70,
}


def run_pipeline_on_testset(test_questions):
    """Run retrieval + generation for every test question, collecting the
    fields RAGAS needs: question, answer, retrieved contexts, ground truth."""
    records = []
    for item in test_questions:
        question = item["question"]
        print(f"  Running: {question}")
        chunks = retrieve(question)
        contexts = [c["text"] for c in chunks]
        try:
            answer = generate_answer(question, chunks)
        except RuntimeError as e:
            print(f"    Skipping generation ({e})")
            answer = ""
        records.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
        })
    return records


def score_with_ragas(records):
    """
    Scores records with RAGAS. Requires `pip install ragas datasets` and an
    LLM API key (RAGAS uses an LLM internally to judge faithfulness/relevancy).
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy

    dataset = Dataset.from_list(records)
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    return result.to_pandas()


def main():
    test_path = os.path.join(os.path.dirname(__file__), "test_questions.json")
    with open(test_path, "r", encoding="utf-8") as f:
        test_questions = json.load(f)

    print(f"Loaded {len(test_questions)} test questions.")
    print("Running pipeline (retrieval + generation) on each ...")
    records = run_pipeline_on_testset(test_questions)

    print("\nScoring with RAGAS ...")
    df = score_with_ragas(records)
    print(df[["question", "faithfulness", "answer_relevancy"]].to_string(index=False))

    avg_faithfulness = df["faithfulness"].mean()
    avg_relevancy = df["answer_relevancy"].mean()

    print(f"\nAverage faithfulness: {avg_faithfulness:.2f} (threshold: {THRESHOLDS['faithfulness']})")
    print(f"Average answer_relevancy: {avg_relevancy:.2f} (threshold: {THRESHOLDS['answer_relevancy']})")

    failed = (
        avg_faithfulness < THRESHOLDS["faithfulness"]
        or avg_relevancy < THRESHOLDS["answer_relevancy"]
    )

    if failed:
        print("\n❌ Evaluation FAILED — scores below threshold. Blocking deploy.")
        sys.exit(1)
    else:
        print("\n✅ Evaluation PASSED.")
        sys.exit(0)


if __name__ == "__main__":
    main()
