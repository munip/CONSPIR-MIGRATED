#!/usr/bin/env python3
"""Replay a cached QA Inspect log through legacy and v2 schema representations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inspect_eval.cb_dataset import load_items  # noqa: E402
from inspect_eval.qa_schema_parity import (  # noqa: E402
    EXPECTED_CORPUS_RECORD_COUNT,
    EXPECTED_CORPUS_SHA256,
    EXPECTED_LEGACY_SCHEMA_SHA256,
    EXPECTED_V2_SCHEMA_SHA256,
    build_conversion_manifest,
    build_parity_report,
    build_record_pairs,
    cache_entries_from_eval_log,
    sha256_file,
    validate_eval_log_settings,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "questions" / "corpus.jsonl",
    )
    parser.add_argument(
        "--legacy-schema",
        type=Path,
        default=ROOT / "schema.json",
    )
    parser.add_argument(
        "--v2-schema",
        type=Path,
        default=ROOT / "schemas" / "conspir" / "v2_schema.json",
    )
    parser.add_argument("--eval-log", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Expected Inspect eval model identifier")
    parser.add_argument("--grader", required=True, help="Expected grader-role model identifier")
    parser.add_argument("--rubric", required=True, help="Expected logged rubric, e.g. rubric_v2")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "schema-parity",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    corpus = args.corpus.resolve()
    root = corpus.parent.parent
    expected_corpus = (root / "questions" / "corpus.jsonl").resolve()
    if corpus != expected_corpus:
        raise ValueError("--corpus must point to a questions\\corpus.jsonl file")

    legacy_schema = _read_json(args.legacy_schema)
    v2_schema = _read_json(args.v2_schema)
    hashes = {
        "corpus": (sha256_file(corpus), EXPECTED_CORPUS_SHA256),
        "legacy_schema": (sha256_file(args.legacy_schema), EXPECTED_LEGACY_SCHEMA_SHA256),
        "v2_schema": (sha256_file(args.v2_schema), EXPECTED_V2_SCHEMA_SHA256),
    }
    for artifact, (actual, expected) in hashes.items():
        if actual != expected:
            raise ValueError(
                f"Pinned {artifact} SHA-256 mismatch: expected {expected}, got {actual}"
            )
    records = [
        item
        for _, _, item in load_items(root=root, schema=legacy_schema)
    ]
    if len(records) != EXPECTED_CORPUS_RECORD_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CORPUS_RECORD_COUNT} corpus records, found {len(records)}"
        )
    pairs = build_record_pairs(records, v2_schema)
    eval_log = read_eval_log(args.eval_log)
    logged_settings = validate_eval_log_settings(
        eval_log,
        model=args.model,
        grader=args.grader,
        rubric=args.rubric,
    )
    cache_entries = cache_entries_from_eval_log(eval_log)
    report = build_parity_report(
        pairs,
        cache_entries,
        model=args.model,
        grader=args.grader,
        rubric=args.rubric,
    )
    report["settings"].update({
        "model_revision": logged_settings["model_revision"],
        "grader_revision": logged_settings["grader_revision"],
        "grader_role": logged_settings["grader_role"],
    })
    report["artifacts"] = {
        "corpus": str(corpus),
        "corpus_sha256": hashes["corpus"][0],
        "legacy_schema": str(args.legacy_schema.resolve()),
        "legacy_schema_sha256": hashes["legacy_schema"][0],
        "v2_schema": str(args.v2_schema.resolve()),
        "v2_schema_sha256": hashes["v2_schema"][0],
        "eval_log": str(args.eval_log.resolve()),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = (
        args.output_dir / "conversion_manifest.json",
        args.output_dir / "response_cache.jsonl",
        args.output_dir / "parity_report.json",
    )
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs:
        raise FileExistsError(
            "Parity outputs already exist: "
            + ", ".join(str(path) for path in existing_outputs)
        )
    manifest = build_conversion_manifest(
        pairs,
        corpus_sha256=report["artifacts"]["corpus_sha256"],
        legacy_schema_sha256=report["artifacts"]["legacy_schema_sha256"],
        v2_schema_sha256=report["artifacts"]["v2_schema_sha256"],
    )
    manifest_path, cache_path, report_path = output_paths
    _write_json_new(manifest_path, manifest)
    with cache_path.open("x", encoding="utf-8") as stream:
        for entry in cache_entries:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    report["artifacts"]["response_cache"] = str(cache_path.resolve())
    _write_json_new(report_path, report)
    return report


def main() -> None:
    report = run(_arguments())
    print(json.dumps({
        "parity_pass": report["parity_pass"],
        "record_count": report["record_count"],
        "cached_eval_sample_count": report["cached_eval_sample_count"],
        "missing_record_ids": report["missing_record_ids"],
        "unknown_record_ids": report["unknown_record_ids"],
    }, indent=2))
    if not report["parity_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
