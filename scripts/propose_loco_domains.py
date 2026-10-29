#!/usr/bin/env python3
"""Create a review queue of LOCO topic-to-domain suggestions.

Suggestions are deliberately not approved mappings. Reviewers must set
``status`` to ``approved`` and confirm ``domain_id`` before the importer uses
the assignment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "schemas" / "conspir" / "domains.json"

# These broad lexical cues only propose existing registry IDs; they never add a
# domain or mark a suggestion as approved.
DOMAIN_CUES = {
    "health_medicine": {
        "health", "medical", "medicine", "disease", "vaccine", "covid", "clinical",
        "patient", "doctor", "hospital", "virus", "cancer", "drug", "treatment",
        "symptom", "mental",
    },
    "politics_government": {
        "government", "politic", "election", "president", "congress", "parliament",
        "party", "vote", "policy", "state", "minister", "campaign", "federal",
        "democracy", "official", "law", "court",
    },
    "environment_climate": {
        "climate", "environment", "weather", "earthquake", "pollution", "carbon",
        "wildlife", "forest", "earth", "flood", "hurricane", "temperature",
    },
    "history_conflict": {
        "war", "military", "army", "battle", "histor", "ancient", "century",
        "holocaust", "nazi", "terror", "attack", "conflict", "weapon", "jew",
        "israel", "palestin",
    },
    "science_technology": {
        "science", "space", "nasa", "technology", "internet", "computer", "data",
        "research", "physics", "astronomy", "satellite", "energy", "engineering",
        "scientist", "experiment", "artificial",
    },
    "society_culture": {
        "religion", "church", "muslim", "christian", "culture", "race", "gender",
        "social", "community", "education", "media", "language", "immigration",
        "family", "celebrity", "sport", "music", "film",
    },
}


def _tokens(value: object) -> set[str]:
    return set(re.findall(r"[a-z]+", str(value or "").lower()))


def suggest_domain(topic: dict) -> tuple[str | None, str]:
    evidence = _tokens(topic.get("Top_15_words")) | _tokens(topic.get("Topic_all_r"))
    counts = {
        domain: len(evidence & cues)
        for domain, cues in DOMAIN_CUES.items()
    }
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if not ranked or ranked[0][1] == 0:
        return None, "No configured domain cue matched the topic description."
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None, "Topic cues are tied between: " + ", ".join(
            domain for domain, count in ranked if count == ranked[0][1]
        )
    domain, count = ranked[0]
    matched = sorted(evidence & DOMAIN_CUES[domain])
    return domain, f"Keyword proposal ({count} matched cues: {', '.join(matched)}); review required."


def propose_domains(topic_path: Path, output_path: Path) -> dict:
    topics = json.loads(Path(topic_path).read_text(encoding="utf-8"))
    if not isinstance(topics, list):
        raise ValueError("LOCO topic descriptions must be a JSON array")
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    allowed = {domain["id"] for domain in registry["domains"]}
    entries = []
    seen = set()
    for row in topics:
        if not isinstance(row, dict):
            raise ValueError("LOCO topic descriptions must contain JSON objects")
        topic_id = str(row.get("Topic_ID", "")).strip()
        if not topic_id:
            raise ValueError("LOCO topic description is missing Topic_ID")
        if not topic_id.startswith("k100_"):
            continue
        if topic_id in seen:
            raise ValueError(f"Duplicate LOCO Topic_ID {topic_id!r}")
        seen.add(topic_id)
        suggestion, rationale = suggest_domain(row)
        if suggestion is not None and suggestion not in allowed:
            raise ValueError(f"Suggested domain is not in the registry: {suggestion!r}")
        entries.append({
            "topic_id": topic_id,
            "domain_id": None,
            "suggested_domain_id": suggestion,
            "status": "pending_review",
            "rationale": rationale,
            "topic_summary": str(row.get("Top_15_words", "")),
        })
    result = {
        "dataset": "LOCO",
        "mapping_version": "1.0",
        "registry": str(REGISTRY_PATH.relative_to(ROOT)).replace("\\", "/"),
        "topic_description_file": str(Path(topic_path)),
        "topic_description_sha256": hashlib.sha256(Path(topic_path).read_bytes()).hexdigest(),
        "registry_sha256": hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest(),
        "review_policy": "Set status=approved and domain_id to a registry ID after human review.",
        "topics": entries,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"topics": len(entries), "suggested": sum(e["suggested_domain_id"] is not None for e in entries),
            "pending_review": len(entries), "output": str(output_path)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("topic_descriptions", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets" / "loco" / "domain_mapping.json")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(propose_domains(args.topic_descriptions, args.output), indent=2))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
