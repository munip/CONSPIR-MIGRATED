#!/usr/bin/env python3
"""Present blind rows losslessly. This script does not assign review labels."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blind", type=Path)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--prior", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.blind.name != "blind.jsonl":
        parser.error("Only a blind.jsonl input is permitted.")
    rows = [json.loads(line) for line in args.blind.read_text().splitlines()]
    paragraphs = {}
    first_seen = {}
    sources = {}
    source_first = {}

    def parts(value, number):
        result = []
        for part in value.split("\n\n"):
            key = "P" + hashlib.sha256(part.encode()).hexdigest()[:12]
            assert key not in paragraphs or paragraphs[key] == part
            paragraphs[key] = part
            first_seen.setdefault(key, number)
            result.append(key)
        assert "\n\n".join(paragraphs[key] for key in result) == value
        return result

    prepared = []
    previous = []
    for prior in args.prior:
        if prior.name != "blind.jsonl":
            parser.error("Every prior input must also be named blind.jsonl.")
        previous.extend(json.loads(line) for line in prior.read_text().splitlines())
    numbered = [(0, row) for row in previous] + list(enumerate(rows, 1))
    for number, row in numbered:
        full = parts(row["text"], number)
        current = parts(row["current_answer"], number)
        supplied = row["source_item"]
        criteria = {key: supplied[key] for key in (
            "id", "question", "ground_truth_answer", "notes", "diverges_if",
            "does_not_diverge_if"
        ) if key in supplied}
        encoded = json.dumps(criteria, sort_keys=True, ensure_ascii=False)
        source = "S" + hashlib.sha256(encoded.encode()).hexdigest()[:12]
        assert source not in sources or sources[source] == criteria
        sources[source] = criteria
        source_first.setdefault(source, number)
        if number:
            prepared.append((number, row["review_id"], source, full, current))

    print("Paragraph boundaries are exactly two newlines, including empty parts.")
    print("Every row is reconstructed and checked against the complete blind input.")
    for key, value in sources.items():
        if args.start <= source_first[key] <= args.end:
            print("SOURCE", key, json.dumps(value, ensure_ascii=False))
    for key, value in paragraphs.items():
        if args.start <= first_seen[key] <= args.end:
            print("\nPARAGRAPH", key)
            print(value)
            print("END PARAGRAPH")
    for number, review_id, source, full, current in prepared:
        if args.start <= number <= args.end:
            def refs(keys):
                return [{"id": key, "reminder": paragraphs[key][:70]} for key in keys]
            print("\nROW", number, review_id, "SOURCE", source)
            print("FULL_ORDER", json.dumps(refs(full), ensure_ascii=False))
            print("CURRENT_ORDER", "same as FULL_ORDER" if current == full else json.dumps(refs(current), ensure_ascii=False))


if __name__ == "__main__":
    main()
