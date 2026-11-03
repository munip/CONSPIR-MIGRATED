#!/usr/bin/env python3
"""Evaluate one CONSPIRed-trained candidate labeler against a labeled split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.label_loco import TRAITS, known_targets, qwen_predictor, read_jsonl, sha256_file, train_supervised
except ModuleNotFoundError:  # direct invocation as `python scripts/evaluate_trait_labeler.py`
    from label_loco import TRAITS, known_targets, qwen_predictor, read_jsonl, sha256_file, train_supervised


def metrics(targets: list[list[int]], predictions: list[list[int]]) -> dict:
    per_trait = {}
    tp_all = fp_all = fn_all = 0
    exact = 0
    for target, predicted in zip(targets, predictions):
        exact += target == predicted
    for i, trait in enumerate(TRAITS):
        tp = sum(y[i] == 1 and p[i] == 1 for y, p in zip(targets, predictions))
        fp = sum(y[i] == 0 and p[i] == 1 for y, p in zip(targets, predictions))
        fn = sum(y[i] == 1 and p[i] == 0 for y, p in zip(targets, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_trait[trait] = {"precision": precision, "recall": recall, "f1": f1}
        tp_all += tp
        fp_all += fp
        fn_all += fn
    micro_p = tp_all / (tp_all + fp_all) if tp_all + fp_all else 0.0
    micro_r = tp_all / (tp_all + fn_all) if tp_all + fn_all else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    return {
        "macro_f1": sum(row["f1"] for row in per_trait.values()) / len(per_trait),
        "micro_f1": micro_f1,
        "exact_match": exact / len(targets) if targets else 0.0,
        "per_trait": per_trait,
    }


def evaluate(method: str, train_path: Path, test_path: Path, output: Path,
             model_id: str | None = None, threshold: float = 0.5,
             device: str | None = None, max_new_tokens: int = 96) -> dict:
    train_rows, test_rows = list(read_jsonl(train_path)), list(read_jsonl(test_path))
    if not train_rows or not test_rows:
        raise ValueError("training and evaluation files must both be non-empty")
    train_docs = {r.get("source", {}).get("document_id") for r in train_rows}
    test_docs = {r.get("source", {}).get("document_id") for r in test_rows}
    overlap = (train_docs & test_docs) - {None, ""}
    if overlap:
        raise ValueError(f"training and evaluation splits share {len(overlap)} source documents")
    if any(r.get("source", {}).get("dataset", "").lower() != "conspired" for r in train_rows + test_rows):
        raise ValueError("both files must be normalized CONSPIRed records")

    targets = [known_targets(row) for row in test_rows]
    invalid = 0
    if method == "supervised":
        model = train_supervised(train_rows)
        scores = model.predict_proba([row["text"] for row in test_rows])
        predictions = [[int(value >= threshold) for value in row] for row in scores]
    else:
        if not model_id:
            raise ValueError("--model is required for qwen")
        predict = qwen_predictor(train_rows, model_id, device, max_new_tokens)
        predictions = []
        for row in test_rows:
            try:
                labels, _raw = predict(row["text"])
                predictions.append([int(trait in labels) for trait in TRAITS])
            except (ValueError, json.JSONDecodeError):
                invalid += 1
                predictions.append([0] * len(TRAITS))
    result = {
        "method": method,
        "model_id": model_id if method == "qwen" else "tfidf-char-word-ovr-logistic",
        "training_file": str(train_path),
        "training_sha256": sha256_file(train_path),
        "evaluation_file": str(test_path),
        "evaluation_sha256": sha256_file(test_path),
        "records": len(test_rows),
        "invalid_outputs": invalid,
        "threshold": threshold,
        "metrics": metrics(targets, predictions),
        "gold_labels": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("supervised", "qwen"))
    parser.add_argument("--train", type=Path, default=Path("datasets/conspired/train.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("datasets/conspired/test.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--device")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(evaluate(args.method, args.train, args.test, args.output,
                                  args.model, args.threshold, args.device,
                                  args.max_new_tokens), indent=2))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
