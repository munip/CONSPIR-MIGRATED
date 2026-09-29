#!/usr/bin/env python3
"""Identify possible near-duplicate questions in questions/.

Run before opening a pull request, after validate_questions.py:

    python scripts/check_duplicates.py

Only pending items are reviewed. Existing corpus items are the set that a
submission is checked against, and corpus-vs-corpus pairs are not reported.

The review applies one of two methods per pending item, selected by its
`category` field.

Defined categories
    For the categories defined in the schema, the script prints every question
    already filed under that category and the contributor reviews the list
    directly.

The `other` category
    `other` is the fallback value used when no defined category applies, so it
    provides no meaningful grouping. Pending items in this category are
    evaluated using the embedding-based similarity check.

Cross-category check
    `category` is a topic label, not a claim identity, so the same claim can be
    filed under neighbouring categories. As a safety net, pending items are
    also compared by embedding against items outside their own category, and
    reported only at or above CROSS_CATEGORY_THRESHOLD.

All methods are advisory. Two questions may concern the same event and remain
individually valid if they test different claims. The script reports
information for the contributor to assess and does not block a submission.

Pass --all to print the entire corpus grouped by category.

The embedding paths download a small (~64MB) local model on first use and
cache it. No API key is required and no data leaves the local machine.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_DIR = ROOT / "questions"
PENDING_DIR = QUESTIONS_DIR / "pending"
PENDING_ID = "cb-pending"
UNGROUPED_CATEGORY = "other"

CATEGORY_THRESHOLD = 0.6
CROSS_CATEGORY_THRESHOLD = 0.8


def load_questions() -> list[dict]:
    """Load every item under questions/, tagging each with its source file."""
    items = []
    for path in sorted(QUESTIONS_DIR.rglob("*.jsonl")):
        if path.name == "template.jsonl":
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["_path"] = path
            item["_pending"] = PENDING_DIR in path.parents
            items.append(item)
    return items


def is_pending(item: dict) -> bool:
    return bool(item.get("_pending")) or item.get("id") == PENDING_ID


def label(item: dict) -> str:
    """Mark the contributor's own items so they are distinguishable in a long list."""
    return "-> " if is_pending(item) else "   "


def category_of(item: dict) -> str:
    return item.get("category", UNGROUPED_CATEGORY)


def display_id(item: dict) -> str:
    """cb-pending is shared by every pending item, so show the file too."""
    item_id = item.get("id", "?")
    if is_pending(item):
        return f"{item_id} ({item['_path'].name})"
    return item_id


def embed(items: list[dict], required: bool):
    """Return a normalized embedding matrix, or None if the model is absent.

    `required` distinguishes the two callers. The `other` check cannot work
    without embeddings. The cross-category check is a safety net on top of
    a review that has already happened, so it degrades to a warning 
    instead of blocking the run.
    """
    try:
        from fastembed import TextEmbedding
        import numpy as np
    except ImportError:
        if required:
            sys.exit("Missing dependency: pip install fastembed numpy")
        print(
            "  skipped — fastembed/numpy not installed.\n"
            "  Install with 'pip install fastembed numpy' to enable the\n"
            "  cross-category check for miscategorised duplicates."
        )
        return None

    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    vectors = np.array(list(model.embed([i["question"] for i in items])))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def report_pairs(pairs: list, heading: str, note: str) -> None:
    print(f"\n{heading}\n")
    for score, a, b in pairs:
        same_claim = (
            a.get("claim_id")
            and b.get("claim_id")
            and a["claim_id"] == b["claim_id"]
        )
        suffix = "  (same claim_id — expected)" if same_claim else ""
        print(f"  similarity {score:.2f}{suffix}")
        for item in (a, b):
            marker = "->" if is_pending(item) else "  "
            cat = category_of(item)
            print(f"    {marker} [{cat}] {display_id(item)}: {item['question']}")
        print()
    print(note)


def review_by_category(items: list[dict], categories: set[str]) -> None:
    """Report every item in the given categories for manual review."""
    by_category = defaultdict(list)
    for item in items:
        by_category[category_of(item)].append(item)

    for category in sorted(categories):
        group = by_category.get(category, [])
        print(f"\n{category} — {len(group)} question(s)")
        print("-" * (len(category) + 20))
        for item in group:
            print(f"  {label(item)}{display_id(item)}: {item['question']}")
            claim_id = item.get("claim_id")
            if claim_id:
                print(f"       [claim: {claim_id}]")


def review_other_by_similarity(items: list[dict]) -> int:
    """Embedding check within `other`, which has no meaningful grouping."""
    other = [i for i in items if category_of(i) == UNGROUPED_CATEGORY]
    pending = [i for i in other if is_pending(i)]
    if not pending or len(other) < 2:
        print(f"\nFewer than 2 questions in '{UNGROUPED_CATEGORY}', nothing to compare.")
        return 0

    print(f"\n{UNGROUPED_CATEGORY} — embedding {len(other)} question(s)...")
    normalized = embed(other, required=True)
    similarity = normalized @ normalized.T

    flagged = []
    for i in range(len(other)):
        for j in range(i + 1, len(other)):
            if not (is_pending(other[i]) or is_pending(other[j])):
                continue
            score = float(similarity[i, j])
            if score >= CATEGORY_THRESHOLD:
                flagged.append((score, other[i], other[j]))
    flagged.sort(key=lambda x: -x[0])

    if not flagged:
        print(f"No pairs above similarity {CATEGORY_THRESHOLD}. Looks distinct.")
        return 0

    report_pairs(
        flagged,
        f"{len(flagged)} pair(s) in '{UNGROUPED_CATEGORY}' worth a second look:",
        "These are informational, not errors. 'other' has no grouping to read,\n"
        "so similarity is the only signal available there.",
    )
    return len(flagged)


def review_across_categories(items: list[dict]) -> int:
    """Safety net for claims filed under different categories."""
    pending = [i for i in items if is_pending(i)]
    if not pending or len(items) < 2:
        return 0

    print(
        f"\ncross-category — embedding {len(items)} question(s) "
        f"at threshold {CROSS_CATEGORY_THRESHOLD}..."
    )
    normalized = embed(items, required=False)
    if normalized is None:
        return 0
    similarity = normalized @ normalized.T

    index = {id(item): n for n, item in enumerate(items)}
    flagged = []
    for a in pending:
        for b in items:
            if a is b or category_of(a) == category_of(b):
                continue
            # Avoid reporting a pending-pending pair twice.
            if is_pending(b) and index[id(b)] < index[id(a)]:
                continue
            score = float(similarity[index[id(a)], index[id(b)]])
            if score >= CROSS_CATEGORY_THRESHOLD:
                flagged.append((score, a, b))
    flagged.sort(key=lambda x: -x[0])

    if not flagged:
        print(
            f"No cross-category pairs above {CROSS_CATEGORY_THRESHOLD}."
        )
        return 0

    report_pairs(
        flagged,
        f"{len(flagged)} cross-category pair(s) worth checking:",
        "These sit in different categories but are worded very similarly. Check\n"
        "whether one of them is miscategorised, or whether they test genuinely\n"
        "different claims about related topics.",
    )
    return len(flagged)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="print the whole corpus grouped by category, not just your categories",
    )
    args = parser.parse_args()

    items = load_questions()
    if not items:
        print("No questions found in questions/.")
        return 0

    pending = [i for i in items if is_pending(i)]

    if args.all:
        categories = {category_of(i) for i in items}
    elif pending:
        categories = {category_of(i) for i in pending}
        print(
            f"{len(pending)} pending item(s) in {len(categories)} "
            f"categor{'y' if len(categories) == 1 else 'ies'}. "
            "Your items are marked with ->."
        )
    else:
        print("No pending items found. Nothing to check.")
        return 0

    grouped = sorted(categories - {UNGROUPED_CATEGORY})
    if grouped:
        review_by_category(items, set(grouped))
        print(
            "\nRead the list(s) above and check whether your question is already\n"
            "covered. Same event but a different claim is fine. Same question\n"
            "reworded is not — reuse the existing item, or set claim_id to group\n"
            "them if you are deliberately adding another framing."
        )

    if UNGROUPED_CATEGORY in categories:
        review_other_by_similarity(items)

    if pending:
        review_across_categories(items)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
