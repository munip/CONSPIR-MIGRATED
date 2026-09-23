#!/usr/bin/env python3
"""Flag possible near-duplicate questions in questions/*.jsonl. Run before
opening a PR, after validate_questions.py:

    python scripts/check_duplicates.py

This is a similarity check, not a hard rule. Two questions can be about the
same event and both be valid if they test different claims. Read the flagged
pairs yourself and use your judgment.

First run downloads a small (~64MB) local embedding model and caches it.
No API key needed, nothing leaves your machine.
"""
import json
import sys
from pathlib import Path

try:
    from fastembed import TextEmbedding
    import numpy as np
except ImportError:
    sys.exit("Missing dependency: pip install fastembed numpy")

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_DIR = ROOT / "questions"

# Below this, questions are treated as unrelated. Above it, they're printed
# for a human to look at — calibrated against real near-duplicate pairs
# ("What happened to JFK on the day he died?" vs "Who killed President
# Kennedy?" scores 0.74; unrelated topics score well under 0.5).
FLAG_THRESHOLD = 0.6


def load_questions() -> list[dict]:
    items = []
    sources = []
    # Walk recursively so we include pending/*.jsonl and corpus.jsonl
    for path in sorted(QUESTIONS_DIR.rglob("*.jsonl")):
        if path.name == "template.jsonl":
            continue
        # Classify files in questions/pending/ as 'pending', others as 'corpus'
        try:
            is_pending = path.match("**/pending/*.jsonl")
        except Exception:
            is_pending = False
        source = "pending" if is_pending else "corpus"
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            sources.append((source, path.name))
    return items, sources


def main() -> int:
    items, sources = load_questions()
    if len(items) < 2:
        print("Fewer than 2 questions in the corpus, nothing to compare.")
        return 0

    print(f"Embedding {len(items)} question(s)...")
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    texts = [item["question"] for item in items]
    embeddings = np.array(list(model.embed(texts)))
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarity = normalized @ normalized.T

    flagged = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            # Only flag pairs where at least one item is from pending
            src_i, _ = sources[i]
            src_j, _ = sources[j]
            if src_i == "corpus" and src_j == "corpus":
                continue
            score = float(similarity[i, j])
            if score >= FLAG_THRESHOLD:
                flagged.append((score, items[i], items[j], sources[i], sources[j]))
    flagged.sort(key=lambda x: -x[0])

    if not flagged:
        print(f"No pairs above similarity {FLAG_THRESHOLD}. Looks distinct.")
        return 0

    print(f"\n{len(flagged)} pair(s) worth a second look:\n")
    for score, a, b, sa, sb in flagged:
        a_src, a_file = sa
        b_src, b_file = sb
        print(f"  similarity {score:.2f}")
        print(f"    {a['id']} ({a_src}/{a_file}): {a['question']}")
        print(f"    {b['id']} ({b_src}/{b_file}): {b['question']}")
        print()
    print(
        "This is informational, not an error. If your question tests a different\n"
        "claim than the one it's paired with here, it's fine. If it's the same\n"
        "question reworded, pick one and drop the other."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
