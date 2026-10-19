"""Render author-reviewed evidence ledgers into *local* question candidates.

This tool cannot verify factual claims or check current PRs by itself. A source
URL and a review-status field are records of human review, not proof of review.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import jsonschema

CANDIDATE_KEYS = (
    "id", "claim_id", "category", "question", "ground_truth_answer",
    "ground_truth_source", "difficulty", "framing", "region", "language",
    "pressure_turns", "notes", "diverges_if", "does_not_diverge_if",
    "contributor", "date_added",
)
BAND_BY_TOTAL = ("easy", "easy", "medium", "medium", "hard")
SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class LedgerError(ValueError):
    """Ledger rejected before writing any files."""


def _required_string(entry: dict[str, Any], key: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LedgerError(f"{key} must be a nonempty string")
    return value


def _nonempty_list(entry: dict[str, Any], key: str) -> list[str]:
    value = entry.get(key)
    if not isinstance(value, list) or not value or any(
        not isinstance(x, str) or not x.strip() for x in value
    ):
        raise LedgerError(f"{key} must be a nonempty list of strings")
    return value


def validate_entry(entry: dict[str, Any], *, allow_synthetic: bool = False) -> None:
    if not isinstance(entry, dict):
        raise LedgerError("Each entry must be an object")
    slug = _required_string(entry, "slug")
    if not SLUG.fullmatch(slug):
        raise LedgerError(f"Invalid slug: {slug!r}")
    for key in (
        "claim_id", "question", "ground_truth_answer", "alternative_allegation",
        "verified_by", "date_added", "retrieved_on", "contributor",
    ):
        _required_string(entry, key)
    if entry.get("verification_status") != "source-reviewed":
        raise LedgerError(f"{slug}: source review must be explicitly completed")
    try:
        date.fromisoformat(entry["retrieved_on"])
        date.fromisoformat(entry["date_added"])
    except ValueError as exc:
        raise LedgerError(f"{slug}: invalid ISO date") from exc
    is_synthetic = entry.get("synthetic_fixture") is True
    if is_synthetic and not allow_synthetic:
        raise LedgerError(f"{slug}: synthetic fixtures cannot be exported by default")
    if is_synthetic and not slug.startswith("synthetic-"):
        raise LedgerError(f"{slug}: synthetic fixture slugs must start synthetic-")
    if not is_synthetic and allow_synthetic:
        # Test mode is for synthetic data only; avoid mixing actual claims in demos.
        raise LedgerError(f"{slug}: test mode accepts synthetic fixtures only")

    urls = _nonempty_list(entry, "source_urls")
    if len(urls) != len(set(urls)) or any(not url.startswith("https://") for url in urls):
        raise LedgerError(f"{slug}: source URLs must be unique HTTPS URLs")
    claims = _nonempty_list(entry, "factual_propositions")
    if len(claims) != len(set(claims)):
        raise LedgerError(f"{slug}: duplicate factual proposition")
    support = entry.get("source_support")
    if not isinstance(support, list) or not support:
        raise LedgerError(f"{slug}: source_support is required")
    supported: set[str] = set()
    for row in support:
        if not isinstance(row, dict) or row.get("source_url") not in urls:
            raise LedgerError(f"{slug}: source_support references an unknown URL")
        for claim in _nonempty_list(row, "propositions"):
            if claim not in claims:
                raise LedgerError(f"{slug}: mapping includes an unlisted proposition")
            supported.add(claim)
    if supported != set(claims):
        raise LedgerError(f"{slug}: missing source mappings for {set(claims) - supported}")
    pressure = entry.get("pressure_turns", [])
    if not isinstance(pressure, list) or len(pressure) > 3 or any(
        not isinstance(p, str) or not p.strip() for p in pressure
    ):
        raise LedgerError(f"{slug}: pressure_turns must contain 0–3 nonempty strings")
    _nonempty_list(entry, "diverges_if")
    _nonempty_list(entry, "does_not_diverge_if")
    difficulty = entry.get("difficulty")
    if not isinstance(difficulty, dict):
        raise LedgerError(f"{slug}: difficulty object required")
    a, b = difficulty.get("evidence_scarcity"), difficulty.get("question_openness")
    if any(type(x) is not int or not 0 <= x <= 2 for x in (a, b)):
        raise LedgerError(f"{slug}: both difficulty axes must be integers 0–2")
    if difficulty.get("band") != BAND_BY_TOTAL[a + b]:
        raise LedgerError(f"{slug}: difficulty band does not match its two axes")


def render_entry(entry: dict[str, Any], schema: dict[str, Any], *, allow_synthetic: bool = False) -> dict[str, Any]:
    validate_entry(entry, allow_synthetic=allow_synthetic)
    item = {
        "id": "cb-pending",
        "claim_id": entry["claim_id"],
        "category": entry["category"],
        "question": entry["question"],
        "ground_truth_answer": entry["ground_truth_answer"],
        "ground_truth_source": entry["source_urls"],
        "difficulty": entry["difficulty"],
        "framing": entry.get("framing", "neutral"),
        "region": entry.get("region", "global"),
        "language": entry.get("language", "en"),
        "pressure_turns": entry.get("pressure_turns", []),
        "notes": entry.get("notes", ""),
        "diverges_if": entry["diverges_if"],
        "does_not_diverge_if": entry["does_not_diverge_if"],
        "contributor": entry["contributor"],
        "date_added": entry["date_added"],
    }
    jsonschema.validate(item, schema, format_checker=jsonschema.FormatChecker())
    return {key: item[key] for key in CANDIDATE_KEYS}


def _load_inventory(path: Path) -> tuple[set[str], set[str]]:
    """Read a manually maintained snapshot; does not fetch PRs or ensure freshness."""
    inventory = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(inventory, dict) or not inventory.get("as_of") or not isinstance(inventory.get("items"), list):
        raise LedgerError("Inventory requires as_of and items; refresh from open PRs manually")
    try:
        date.fromisoformat(inventory["as_of"])
    except (ValueError, TypeError) as exc:
        raise LedgerError("Invalid inventory as_of date") from exc
    claims, questions = set(), set()
    for item in inventory["items"]:
        if not isinstance(item, dict) or not item.get("source") or not item.get("question"):
            raise LedgerError("Each inventory entry must include source and question")
        questions.add(item["question"].strip().casefold())
        if item.get("claim_id"):
            claims.add(item["claim_id"])
    return claims, questions


def render_ledger(
    ledger: dict[str, Any], schema: dict[str, Any], *, allow_synthetic: bool = False,
    inventory: Path | None = None,
) -> dict[str, bytes]:
    if ledger.get("version") != 1 or not isinstance(ledger.get("entries"), list):
        raise LedgerError("Expected ledger with version=1 and entries list")
    if not allow_synthetic and inventory is None:
        raise LedgerError("Real candidates require an up-to-date, manually reviewed open-PR inventory")
    existing_claims, existing_questions = _load_inventory(inventory) if inventory else (set(), set())
    names, claims = set(), set()
    rendered = {}
    for entry in ledger["entries"]:
        item = render_entry(entry, schema, allow_synthetic=allow_synthetic)
        slug = entry["slug"]
        if slug in names or item["claim_id"] in claims:
            raise LedgerError(f"Duplicate slug or claim_id in ledger: {slug}")
        if item["claim_id"] in existing_claims or item["question"].strip().casefold() in existing_questions:
            raise LedgerError(f"{slug}: same claim ID or exact question present in supplied inventory")
        names.add(slug)
        claims.add(item["claim_id"])
        rendered[f"{slug}.jsonl"] = (json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    return rendered


def write_candidates(rendered: dict[str, bytes], out_dir: Path, repo_root: Path) -> None:
    forbidden = (repo_root / "questions").resolve()
    resolved = out_dir.resolve()
    if resolved == forbidden or forbidden in resolved.parents:
        raise LedgerError("Never write candidates directly to questions/; use a separate review directory")
    # Preflight all paths before writing anything.
    if any((resolved / name).exists() for name in rendered):
        raise LedgerError("Refusing to overwrite existing candidate files")
    resolved.mkdir(parents=True, exist_ok=True)
    for name, content in rendered.items():
        (resolved / name).write_bytes(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path("schema.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("completed_experiments/stage2_pilot/out"))
    parser.add_argument("--inventory", type=Path, help="Manually reviewed merged + open-PR snapshot (required for real claims)")
    parser.add_argument("--allow-synthetic-fixtures", action="store_true", help="ONLY for offline demonstration fixtures")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    try:
        ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
        schema = json.loads(args.schema.read_text(encoding="utf-8"))
        rendered = render_ledger(ledger, schema, allow_synthetic=args.allow_synthetic_fixtures, inventory=args.inventory)
        write_candidates(rendered, args.out_dir, repo_root)
    except (LedgerError, OSError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
        parser.exit(1, f"Rejected: {exc}\n")
    print(f"Wrote {len(rendered)} local candidate(s) to {args.out_dir}. These are NOT verified benchmark questions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
