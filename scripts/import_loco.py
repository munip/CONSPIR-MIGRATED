#!/usr/bin/env python3
"""Stream LOCO documents into provenance-preserving, group-split CONSPIR records.

This importer does not infer trait labels. Both partitions are deliberately emitted
with all traits unannotated; label the train pool separately and keep the holdout
frozen for human-adjudicated evaluation.

Install the streaming parser with: python -m pip install ijson
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

try:
    import ijson
except ImportError:
    raise SystemExit("Missing dependency: python -m pip install ijson")

ROOT = Path(__file__).resolve().parent.parent
DOMAIN_REGISTRY_PATH = ROOT / "schemas" / "conspir" / "domains.json"
TRAIT_IDS = (
    "contradictory", "overriding_suspicion", "nefarious_intent",
    "something_must_be_wrong", "persecuted_victim", "immune_to_evidence",
    "reinterpreting_randomness",
)


def iter_documents(path: Path):
    """Yield documents from LOCO's top-level JSON array without loading it all."""
    with path.open("rb") as stream:
        yield from ijson.items(stream, "item")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_sidecar(path: Path | None, key: str) -> dict[str, dict]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise ValueError(f"LOCO sidecar {path} must contain a JSON array")
    index = {}
    for row in rows:
        if not isinstance(row, dict) or row.get(key) in (None, ""):
            continue
        value = str(row[key])
        if value in index:
            raise ValueError(f"LOCO sidecar {path} has duplicate {key} {value!r}")
        index[value] = row
    return index


def _load_domain_mapping(path: Path | None,
                         topic_description_path: Path | None = None) -> tuple[dict[str, dict], set[str]]:
    registry = json.loads(DOMAIN_REGISTRY_PATH.read_text(encoding="utf-8"))
    allowed = {domain["id"] for domain in registry["domains"]}
    if path is None:
        return {}, allowed
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("topics"), list):
        raise ValueError(f"LOCO domain mapping {path} must contain a topics array")
    registry_hash = data.get("registry_sha256")
    if registry_hash is not None and registry_hash != sha256_file(DOMAIN_REGISTRY_PATH):
        raise ValueError(f"LOCO domain mapping {path} was created for a different domain registry")
    topic_hash = data.get("topic_description_sha256")
    if topic_hash is not None:
        if topic_description_path is None:
            raise ValueError(
                f"LOCO domain mapping {path} requires its --topic-description source file"
            )
        if topic_hash != sha256_file(Path(topic_description_path)):
            raise ValueError(f"LOCO domain mapping {path} was created for different topic descriptions")
    index = {}
    for row in data["topics"]:
        if not isinstance(row, dict) or not row.get("topic_id"):
            raise ValueError(f"LOCO domain mapping {path} contains an entry without topic_id")
        topic_id = str(row["topic_id"])
        if topic_id in index:
            raise ValueError(f"LOCO domain mapping {path} has duplicate topic_id {topic_id!r}")
        status = row.get("status")
        if not isinstance(status, str) or status not in {"approved", "pending_review", "rejected"}:
            raise ValueError(f"LOCO domain mapping for {topic_id!r} has invalid status")
        suggested_domain = row.get("suggested_domain_id")
        if (suggested_domain is not None
                and (not isinstance(suggested_domain, str) or suggested_domain not in allowed)):
            raise ValueError(f"LOCO domain mapping for {topic_id!r} has invalid suggested_domain_id")
        if status == "approved":
            domain_id = row.get("domain_id")
            if not isinstance(domain_id, str) or domain_id not in allowed:
                raise ValueError(f"Approved LOCO domain mapping for {topic_id!r} has invalid domain_id")
        elif row.get("domain_id") is not None:
            raise ValueError(f"Unapproved LOCO domain mapping for {topic_id!r} must not set domain_id")
        index[topic_id] = row
    return index, allowed


def json_compatible(value):
    """Convert ijson Decimal values and nested containers to JSON-safe values."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    return value


def canonical_group(value) -> str:
    if value is None or value == "":
        raise ValueError("LOCO row is missing topic_k100; cannot make a leakage-safe split")
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def choose_holdout_groups(group_counts: Counter, fraction: float, seed: int) -> set[str]:
    if not 0 < fraction < 1:
        raise ValueError("holdout fraction must be between 0 and 1")
    if len(group_counts) < 2:
        raise ValueError("At least two distinct topic groups are required for a holdout")
    total = sum(group_counts.values())
    target = min(total - 1, max(1, round(total * fraction)))
    ranked = sorted(
        group_counts,
        key=lambda group: hashlib.sha256(f"{seed}\0{group}".encode("utf-8")).hexdigest(),
    )
    holdout, count = set(), 0
    for group in ranked:
        if len(holdout) == len(ranked) - 1:
            break
        if not holdout or abs(count + group_counts[group] - target) < abs(count - target):
            holdout.add(group)
            count += group_counts[group]
        if count >= target:
            break
    if not holdout or len(holdout) == len(ranked):
        raise ValueError("Unable to create non-empty train and holdout topic partitions")
    return holdout


def _doc_id(row: dict) -> str:
    value = row.get("doc_id")
    if value is None or str(value).strip() == "":
        raise ValueError("LOCO row is missing doc_id")
    return str(value)


def _normalized_record(row: dict, source_revision: str, split: str, group: str,
                      source_license: str, website_info: dict | None = None,
                      topic_info: dict | None = None,
                      domain_mapping: dict | None = None) -> dict:
    doc_id = _doc_id(row)
    text = row.get("txt")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"LOCO document {doc_id!r} has empty or invalid txt")
    url = row.get("URL", row.get("url"))
    mapping = domain_mapping or {}
    approved_domain = mapping.get("domain_id") if mapping.get("status") == "approved" else None
    return {
        "schema_version": "2.0",
        "task_type": "trait_classification",
        "id": f"loco-{doc_id}",
        "text": text,
        "domain_id": approved_domain or "unmapped",
        "trait_annotations": {trait: "unannotated" for trait in TRAIT_IDS},
        "source": {
            "dataset": "LOCO",
            "dataset_revision": source_revision,
            "source_row_id": doc_id,
            "document_id": doc_id,
            "upstream_split": split,
            "license": source_license,
            **({"url": str(url)} if url else {}),
        },
        "metadata": {
            "website": row.get("website"),
            "date": row.get("date"),
            "subcorpus": row.get("subcorpus"),
            "seeds": row.get("seeds"),
            "title": row.get("title"),
            "topic_k100": row.get("topic_k100"),
            "topic_k200": row.get("topic_k200"),
            "topic_k300": row.get("topic_k300"),
            "split_group_key": "topic_k100",
            "split_group": group,
            "label_status": "unannotated",
            "domain_mapping_status": mapping.get("status", "unmapped"),
            **({"suggested_domain_id": mapping["suggested_domain_id"]}
               if mapping.get("suggested_domain_id") else {}),
            "source_metadata": {key: value for key, value in row.items() if key not in ("txt", "URL", "url")},
            **({"website_metadata": website_info} if website_info else {}),
            **({"topic_description": topic_info} if topic_info else {}),
        },
    }


def import_loco(input_path: Path, output_dir: Path, seed: int = 42,
                holdout_fraction: float = 0.30, source_revision: str = "unspecified",
                source_license: str = "unverified", topic_description_path: Path | None = None,
                website_metadata_path: Path | None = None,
                domain_mapping_path: Path | None = None) -> dict:
    input_path, output_dir = Path(input_path), Path(output_dir)
    topic_descriptions = _load_sidecar(topic_description_path, "Topic_ID")
    website_metadata = _load_sidecar(website_metadata_path, "website")
    domain_mappings, _allowed_domains = _load_domain_mapping(
        domain_mapping_path, topic_description_path
    )
    group_counts = Counter()
    seen_ids = set()
    for index, row in enumerate(iter_documents(input_path), start=1):
        if not isinstance(row, dict):
            raise ValueError(f"LOCO item {index} is not a JSON object")
        doc_id = _doc_id(row)
        if doc_id in seen_ids:
            raise ValueError(f"Duplicate LOCO doc_id {doc_id!r}")
        seen_ids.add(doc_id)
        group_counts[canonical_group(row.get("topic_k100"))] += 1
    if not group_counts:
        raise ValueError("LOCO input contains no documents")

    holdout_groups = choose_holdout_groups(group_counts, holdout_fraction, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_pool.jsonl"
    holdout_path = output_dir / "frozen_holdout.jsonl"
    counts = Counter()
    with train_path.open("w", encoding="utf-8", newline="\n") as train_file, \
            holdout_path.open("w", encoding="utf-8", newline="\n") as holdout_file:
        for row in iter_documents(input_path):
            group = canonical_group(row.get("topic_k100"))
            split = "holdout" if group in holdout_groups else "train"
            record = _normalized_record(
                row, source_revision, split, group, source_license,
                website_info=website_metadata.get(str(row.get("website", ""))),
                topic_info=topic_descriptions.get(str(row.get("topic_k100", ""))),
                domain_mapping=domain_mappings.get(str(row.get("topic_k100", ""))),
            )
            counts["website_metadata_joined"] += int("website_metadata" in record["metadata"])
            counts["topic_description_joined"] += int("topic_description" in record["metadata"])
            counts["domain_mapped"] += int(record["domain_id"] != "unmapped")
            counts["domain_suggested"] += int("suggested_domain_id" in record["metadata"])
            mapping_status = record["metadata"]["domain_mapping_status"]
            counts[f"domain_status_{mapping_status}"] += 1
            destination = holdout_file if split == "holdout" else train_file
            destination.write(json.dumps(json_compatible(record), ensure_ascii=False) + "\n")
            counts[split] += 1

    total_documents = counts["train"] + counts["holdout"]
    manifest = {
        "dataset": "LOCO",
        "source_revision": source_revision,
        "source_license": source_license,
        "input_file": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "split_seed": seed,
        "group_key": "topic_k100",
        "requested_holdout_fraction": holdout_fraction,
        "actual_holdout_fraction": counts["holdout"] / total_documents,
        "documents": total_documents,
        "train_pool_documents": counts["train"],
        "frozen_holdout_documents": counts["holdout"],
        "train_pool_sha256": sha256_file(train_path),
        "frozen_holdout_sha256": sha256_file(holdout_path),
        "topic_description_file": str(Path(topic_description_path).resolve()) if topic_description_path else None,
        "topic_description_sha256": sha256_file(Path(topic_description_path)) if topic_description_path else None,
        "website_metadata_file": str(Path(website_metadata_path).resolve()) if website_metadata_path else None,
        "website_metadata_sha256": sha256_file(Path(website_metadata_path)) if website_metadata_path else None,
        "documents_with_topic_description": counts["topic_description_joined"],
        "documents_with_website_metadata": counts["website_metadata_joined"],
        "documents_with_approved_domain_mapping": counts["domain_mapped"],
        "documents_with_domain_suggestion": counts["domain_suggested"],
        "domain_mapping_file": str(Path(domain_mapping_path).resolve()) if domain_mapping_path else None,
        "domain_mapping_sha256": sha256_file(Path(domain_mapping_path)) if domain_mapping_path else None,
        "domain_mapping_status_counts": {
            "approved": counts["domain_mapped"],
            "pending_review": counts["domain_status_pending_review"],
            "rejected": counts["domain_status_rejected"],
            "unmapped": total_documents - counts["domain_mapped"]
                - counts["domain_status_pending_review"] - counts["domain_status_rejected"],
        },
        "train_groups": sorted(set(group_counts) - holdout_groups),
        "holdout_groups": sorted(holdout_groups),
        "annotations": "all seven traits unannotated; no source-status labels converted to traits",
        "source_status_field": "metadata.subcorpus",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to LOCO.json (top-level JSON array)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets" / "loco" / "imported")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.30)
    parser.add_argument("--source-revision", default="unspecified")
    parser.add_argument("--source-license", default="unverified")
    parser.add_argument("--topic-description", type=Path)
    parser.add_argument("--website-metadata", type=Path)
    parser.add_argument("--domain-mapping", type=Path,
                        help="Reviewed mapping from topic_k100 to approved domain registry IDs")
    args = parser.parse_args(argv)
    try:
        manifest = import_loco(args.input, args.output_dir, args.seed,
                               args.holdout_fraction, args.source_revision, args.source_license,
                               args.topic_description, args.website_metadata, args.domain_mapping)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
