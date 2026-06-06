from __future__ import annotations

import argparse
import json
from pathlib import Path


def compute_rougel(predictions: list[str], references: list[str]) -> float:
    try:
        import evaluate

        rouge = evaluate.load("rouge")
        result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
        return float(result["rougeL"])
    except Exception:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [
            scorer.score(reference, prediction)["rougeL"].fmeasure
            for prediction, reference in zip(predictions, references)
        ]
        return float(sum(scores) / max(len(scores), 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute RougeL from a JSONL file.")
    parser.add_argument("--jsonl", required=True, help="Rows must contain prediction and reference fields.")
    args = parser.parse_args()

    predictions: list[str] = []
    references: list[str] = []
    with open(Path(args.jsonl), "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            predictions.append(row["prediction"])
            references.append(row["reference"])

    print({"rougeL": compute_rougel(predictions, references)})


if __name__ == "__main__":
    main()
