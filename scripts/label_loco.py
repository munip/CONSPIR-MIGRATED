#!/usr/bin/env python3
"""Create LOCO training-pool candidate trait labels from CONSPIRed supervision.

This writes pseudo-labels only. It never reads or labels the frozen LOCO holdout.
The methods are a TF-IDF multi-label classifier and local Hugging Face chat model
with balanced CONSPIRed training examples. Candidate records remain explicitly
marked as machine-generated and are not gold evaluation labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRAITS = (
    "contradictory", "overriding_suspicion", "nefarious_intent",
    "persecuted_victim", "immune_to_evidence", "reinterpreting_randomness",
)
UNLABELED = "something_must_be_wrong"
SYSTEM_PROMPT = """Classify the text for the six CONSPIRed cognitive traits. Multiple traits may apply.
Trait IDs: contradictory (internally inconsistent claims); overriding_suspicion (overriding suspicion of institutions);
nefarious_intent (attributing hidden malicious intent); persecuted_victim (self/group framed as targeted victim);
immune_to_evidence (contrary evidence dismissed or made impossible to accept);
reinterpreting_randomness (random events treated as intentionally connected).
Return only a JSON array of present trait IDs. Use [] if none are clearly present."""


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no}: expected an object")
                yield row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def known_targets(row: dict) -> list[int]:
    ann = row.get("trait_annotations", {})
    if any(ann.get(trait) not in ("present", "absent") for trait in TRAITS):
        raise ValueError(f"{row.get('id')}: expected six known CONSPIRed labels")
    return [int(ann[trait] == "present") for trait in TRAITS]


def parse_prediction(text: str) -> list[str]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, list) or any(not isinstance(x, str) or x not in TRAITS for x in value):
        raise ValueError("expected a JSON array containing only the six CONSPIRed trait IDs")
    if len(value) != len(set(value)):
        raise ValueError("duplicate trait ID")
    return value


def word_windows(text: str, size: int = 140, overlap: int = 30) -> list[str]:
    words = re.findall(r"\S+", text)
    if len(words) <= size:
        return [text]
    step = size - overlap
    return [" ".join(words[start:start + size]) for start in range(0, len(words), step)]


def train_supervised(train_rows: list[dict]):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise SystemExit("Install optional dependencies: pip install -r requirements-loco-labeling.txt") from exc
    texts, targets = [], []
    for row in train_rows:
        if row.get("source", {}).get("dataset", "").lower() != "conspired":
            raise ValueError("supervised training input must be the CONSPIRed training split")
        texts.append(row["text"])
        targets.append(known_targets(row))
    if not texts:
        raise ValueError("CONSPIRed training split is empty")
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=120_000,
                                 strip_accents="unicode", sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2,
                                 max_features=160_000, sublinear_tf=True)),
    ])
    model = Pipeline([
        ("features", features),
        ("labels", OneVsRestClassifier(LogisticRegression(
            class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42
        ))),
    ])
    model.fit(texts, targets)
    return model


def qwen_predictor(train_rows: list[dict], model_id: str, device: str | None,
                   max_new_tokens: int):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install optional dependencies: pip install -r requirements-loco-labeling.txt") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    model_kwargs: dict[str, Any] = {"torch_dtype": "auto"}
    if device:
        model_kwargs["device_map"] = device
    else:
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    # Deterministic, compact demonstrations drawn only from the official train split.
    examples = []
    selected = set()
    for trait in TRAITS:
        row = next((r for r in train_rows if r["id"] not in selected
                    and r["trait_annotations"].get(trait) == "present"), None)
        if row:
            selected.add(row["id"])
            examples.append((row["text"][:700], [t for t in TRAITS if row["trait_annotations"].get(t) == "present"]))
    negative = next((r for r in train_rows if r["id"] not in selected
                     and not any(r["trait_annotations"].get(t) == "present" for t in TRAITS)), None)
    if negative:
        examples.append((negative["text"][:700], []))

    def predict(text: str) -> tuple[list[str], str]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for example_text, labels in examples:
            messages.extend([
                {"role": "user", "content": f"Text:\n{example_text}"},
                {"role": "assistant", "content": json.dumps(labels)},
            ])
        messages.append({"role": "user", "content": f"Text:\n{text}"})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=min(8192, getattr(model.config, "max_position_embeddings", 8192)))
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                    pad_token_id=tokenizer.eos_token_id)
        generated = output[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
        return parse_prediction(answer), answer
    return predict


def label_loco(method: str, conspired_train: Path, loco_train: Path, output: Path,
               model_id: str | None = None, device: str | None = None,
               threshold: float = 0.5, max_new_tokens: int = 96,
               qwen_aggregation: str = "any") -> dict:
    if qwen_aggregation not in ("any", "majority"):
        raise ValueError("qwen_aggregation must be any or majority")
    train_rows = list(read_jsonl(conspired_train))
    count = 0
    for row in read_jsonl(loco_train):
        if row.get("source", {}).get("dataset", "").lower() != "loco":
            raise ValueError("input must be the imported LOCO train pool, never the holdout")
        if row.get("source", {}).get("upstream_split") != "train":
            raise ValueError("refusing to label rows outside the LOCO training partition")
        count += 1
    if not count:
        raise ValueError("LOCO training pool is empty")

    model, predict = None, None
    if method == "supervised":
        model = train_supervised(train_rows)
    elif method == "qwen":
        if not model_id:
            raise ValueError("--model is required for --method qwen")
        predict = qwen_predictor(train_rows, model_id, device, max_new_tokens)
    else:
        raise ValueError("method must be supervised or qwen")

    output.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    invalid = 0
    written = 0
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in read_jsonl(loco_train):
            chunks = word_windows(row["text"])
            raw_outputs = []
            if model is not None:
                probabilities = model.predict_proba(chunks)
                # OneVsRest returns one probability vector per label.
                if hasattr(probabilities, "shape"):
                    probabilities = [probabilities[:, i] for i in range(len(TRAITS))]
                trait_scores = {trait: float(max(probabilities[i])) for i, trait in enumerate(TRAITS)}
                predicted = [trait for trait in TRAITS if trait_scores[trait] >= threshold]
                valid_windows = len(chunks)
            else:
                votes = {trait: 0 for trait in TRAITS}
                valid_windows = 0
                for chunk in chunks:
                    try:
                        labels, raw = predict(chunk)
                        valid_windows += 1
                        raw_outputs.append(raw)
                        for trait in labels:
                            votes[trait] += 1
                    except (ValueError, json.JSONDecodeError) as exc:
                        invalid += 1
                        raw_outputs.append(f"<invalid: {exc}>")
                trait_scores = {trait: votes[trait] / max(1, valid_windows) for trait in TRAITS}
                predicted = [
                    trait for trait in TRAITS
                    if (votes[trait] > 0 if qwen_aggregation == "any"
                        else trait_scores[trait] >= threshold)
                ]
                if valid_windows == 0:
                    predicted = []
            anns = {
                trait: ("unannotated" if method == "qwen" and not valid_windows
                        else "present" if trait in predicted else "absent")
                for trait in TRAITS
            }
            anns[UNLABELED] = "unannotated"
            row["trait_annotations"] = anns
            row["present_traits"] = predicted
            known_traits = [] if method == "qwen" and not valid_windows else list(TRAITS)
            row["known_traits"] = known_traits
            row["metadata"]["label_status"] = "invalid" if not known_traits else "pseudo_labeled"
            row["metadata"]["candidate_annotation"] = {
                "method": "conspired_tfidf_ovr_logistic" if method == "supervised" else "conspired_few_shot_qwen",
                "model_id": model_id if method == "qwen" else "tfidf-char-word-ovr-logistic",
                "training_split": "ConspirED/train.jsonl",
                "training_sha256": sha256_file(conspired_train),
                "threshold": threshold,
                "window_words": 140,
                "window_overlap_words": 30,
                "window_aggregation": qwen_aggregation if method == "qwen" else "max_probability",
                "trait_scores": trait_scores,
                "raw_outputs": raw_outputs if method == "qwen" else [],
                "created_at": timestamp,
                "gold_label": False,
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return {"method": method, "input_records": written, "output": str(output),
            "invalid_qwen_windows": invalid, "gold_labels": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("supervised", "qwen"))
    parser.add_argument("--conspired-train", type=Path, default=Path("datasets/conspired/train.jsonl"))
    parser.add_argument("--loco-train", type=Path, default=Path("datasets/loco/imported/train_pool.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", help="Local Hugging Face model path or model ID (Qwen method only)")
    parser.add_argument("--device", help="Transformers device_map, e.g. auto, cuda:0, or cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--qwen-aggregation", choices=("any", "majority"), default="any")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args(argv)
    if not 0 < args.threshold <= 1:
        parser.error("--threshold must be in (0, 1]")
    try:
        result = label_loco(args.method, args.conspired_train, args.loco_train, args.output,
                            args.model, args.device, args.threshold, args.max_new_tokens,
                            args.qwen_aggregation)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
