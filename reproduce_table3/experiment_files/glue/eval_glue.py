from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def compute_accuracy(predictions: Sequence[int], references: Sequence[int]) -> float:
    correct = sum(int(p == r) for p, r in zip(predictions, references))
    return float(correct / max(len(references), 1))


def compute_f1(predictions: Sequence[int], references: Sequence[int]) -> float:
    tp = sum(int(p == 1 and r == 1) for p, r in zip(predictions, references))
    fp = sum(int(p == 1 and r == 0) for p, r in zip(predictions, references))
    fn = sum(int(p == 0 and r == 1) for p, r in zip(predictions, references))
    denom = 2 * tp + fp + fn
    return float(0.0 if denom == 0 else (2 * tp) / denom)


def compute_matthews_corrcoef(predictions: Sequence[int], references: Sequence[int]) -> float:
    tp = sum(int(p == 1 and r == 1) for p, r in zip(predictions, references))
    tn = sum(int(p == 0 and r == 0) for p, r in zip(predictions, references))
    fp = sum(int(p == 1 and r == 0) for p, r in zip(predictions, references))
    fn = sum(int(p == 0 and r == 1) for p, r in zip(predictions, references))
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom == 0:
        return 0.0
    return float((tp * tn - fp * fn) / np.sqrt(denom))


def compute_pearson(predictions: Sequence[float], references: Sequence[float]) -> float:
    preds = np.asarray(predictions, dtype=np.float64)
    refs = np.asarray(references, dtype=np.float64)
    if len(refs) < 2 or np.std(preds) == 0.0 or np.std(refs) == 0.0:
        return 0.0
    return float(np.corrcoef(preds, refs)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def compute_spearman(predictions: Sequence[float], references: Sequence[float]) -> float:
    preds = np.asarray(predictions, dtype=np.float64)
    refs = np.asarray(references, dtype=np.float64)
    if len(refs) < 2:
        return 0.0
    return compute_pearson(_rankdata(preds), _rankdata(refs))


def compute_glue_metrics(task: str, predictions: Sequence[float], references: Sequence[float]) -> dict[str, float]:
    task = task.lower()
    if task == "stsb":
        pearson = compute_pearson(predictions, references)
        spearman = compute_spearman(predictions, references)
        return {
            "pearson": pearson,
            "spearmanr": spearman,
            "glue_score": (pearson + spearman) / 2.0,
        }

    preds = [int(p) for p in predictions]
    refs = [int(r) for r in references]
    accuracy = compute_accuracy(preds, refs)

    if task == "cola":
        matthews = compute_matthews_corrcoef(preds, refs)
        return {"matthews_correlation": matthews, "glue_score": matthews}

    if task in {"mrpc", "qqp"}:
        f1 = compute_f1(preds, refs)
        return {"accuracy": accuracy, "f1": f1, "glue_score": (accuracy + f1) / 2.0}

    return {"accuracy": accuracy, "glue_score": accuracy}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute GLUE metrics from a JSONL file.")
    parser.add_argument("--jsonl", required=True, help="Rows must contain prediction and label fields.")
    parser.add_argument("--task", default="sst2", help="GLUE task name.")
    args = parser.parse_args()

    predictions: list[float] = []
    labels: list[float] = []
    with open(Path(args.jsonl), "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            predictions.append(float(row["prediction"]))
            labels.append(float(row["label"]))

    print(compute_glue_metrics(args.task, predictions, labels))


if __name__ == "__main__":
    main()
