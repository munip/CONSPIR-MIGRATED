"""ConspiracyBench items as Inspect samples, and the fixed subset that repeats pressure sequences."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
import jsonschema

from inspect_ai.dataset import MemoryDataset, Sample

ROOT = Path(__file__).resolve().parent.parent

SCHEMA = json.loads((ROOT / "schema.json").read_text(encoding="utf-8"))

# Item fields copied to the top level of Sample.metadata, so they can be filtered on in
# Inspect View and broken down in report.py. The whole item stays under "item" as well.
BREAKDOWN_FIELDS = ("category", "band", "framing", "claim_id", "region", "language", "multiturn")


def load_items(root: Path = ROOT, include_pending: bool = False):
    """Yield (path, line number, item) for the corpus and, if asked, pending submissions.

    Every item is checked against schema.json; an invalid one raises ValueError naming
    the file, line and field, so a broken pending draft fails loudly when included.
    """
    paths = [root / "questions" / "corpus.jsonl"]
    if include_pending:
        paths += sorted((root / "questions" / "pending").glob("*.jsonl"))
    for path in paths:
        if not path.exists():
            continue
        # split("\n"), not splitlines(): that also breaks at U+2028, U+2029 and U+0085, which
        # JSON allows unescaped inside a string, so it would cut a valid question in two.
        for lineno, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
            if line.strip():
                item = json.loads(line)
                try:
                    jsonschema.validate(item, SCHEMA, format_checker=jsonschema.FormatChecker())
                except jsonschema.ValidationError as e:
                    raise ValueError(f"Invalid item at {path.name}:{lineno} ({e.json_path}): {e.message}") from e
                try:
                    json.dumps(item, ensure_ascii=False).encode("utf-8")
                except UnicodeEncodeError as e:
                    # "\ud800" is valid JSON and passes the schema, but is not text: the run
                    # would fail writing its log, and take the remaining items with it.
                    raise ValueError(
                        f"Invalid item at {path.name}:{lineno}: contains a lone surrogate "
                        f"({e.reason}), usually half an emoji"
                    ) from e
                yield path, lineno, item


def item_metadata(item: dict) -> dict:
    """Sample metadata: the whole item, plus the fields results are broken down by."""
    difficulty = item.get("difficulty")
    return {
        "item": item,
        "category": item.get("category"),
        "band": difficulty.get("band") if isinstance(difficulty, dict) else None,
        "framing": item.get("framing"),
        "claim_id": item.get("claim_id"),
        "region": item.get("region"),
        "language": item.get("language"),
        "multiturn": bool(item.get("pressure_turns")),
    }


def item_to_sample(item: dict, sample_id: str) -> Sample:
    return Sample(
        input=item["question"],
        target=item["ground_truth_answer"],
        id=sample_id,
        metadata=item_metadata(item),
    )


def conspiracybench_dataset(root: Path = ROOT, include_pending: bool = False) -> MemoryDataset:
    """Every item in the corpus. Pending submissions aren't part of the benchmark yet
    (questions/pending/README.md), so they're included only with ``include_pending=True``.

    Pending items all share the placeholder id ``cb-pending``, so they are given a sample
    id from their file and line instead, e.g. ``pending/wtc7-collapse:1``.
    """
    samples = []
    seen_ids = set()
    for path, lineno, item in load_items(root, include_pending):
        sample_id = item["id"] if item["id"] != "cb-pending" else f"pending/{path.stem}:{lineno}"
        # assign_ids.py requires pending items to use "cb-pending" and assigns unique
        # real IDs at merge time, so corpus↔pending collisions cannot occur in normal
        # workflow.  This check catches genuinely broken data (e.g. manual edits).
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate item id: {sample_id}")
        seen_ids.add(sample_id)
        samples.append(item_to_sample(item, sample_id))
    return MemoryDataset(samples, name="conspiracybench")


def stratified_subset(samples, per_cell: int = 1, seed: int = 0) -> list[str]:
    """Sample ids whose pressure sequence is repeated in every epoch (SCHEMA.md, k = 5).

    Only multi-turn items have a pressure sequence to repeat. They are grouped by
    ``category`` and difficulty ``band``, and up to ``per_cell`` ids are drawn from each
    cell with a seeded RNG over sorted ids, so the same corpus and seed always give the
    same subset. ``per_cell=0`` gives an empty subset.
    """
    cells = defaultdict(list)
    for sample in samples:
        metadata = sample.metadata or {}
        if metadata.get("multiturn"):
            cells[(str(metadata.get("category")), str(metadata.get("band")))].append(str(sample.id))
    rng = random.Random(seed)
    chosen = []
    for cell in sorted(cells):
        ids = sorted(cells[cell])
        chosen += rng.sample(ids, min(per_cell, len(ids)))
    return sorted(chosen)


# "all" is the whole corpus in one run. The other three are the evals a baseline runs separately,
# so no item is asked as a single-turn question in two of them.
VARIANTS = ("all", "multiturn", "multiturn_first", "single_turn")


def variant_dataset(samples, variant: str) -> MemoryDataset:
    """The samples of one eval.

    - ``all``: every item, unchanged.
    - ``multiturn``: the items with pressure turns, each played as its whole conversation.
    - ``multiturn_first``: the same items' first question on its own, with no pressure turns, as a
      separate held-out single-turn eval. Sample ids and the item are unchanged apart from dropping
      ``pressure_turns``, so the item is graded exactly like a single-turn one.
    - ``single_turn``: the items that have no pressure turns.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    out = []
    for sample in samples:
        metadata = sample.metadata or {}
        multiturn = bool(metadata.get("multiturn"))
        if variant == "all" or variant == "single_turn" and not multiturn or variant == "multiturn" and multiturn:
            out.append(sample)
        elif variant == "multiturn_first" and multiturn:
            opening = {k: v for k, v in metadata["item"].items() if k != "pressure_turns"}
            out.append(Sample(
                input=sample.input,
                target=sample.target,
                id=sample.id,
                metadata={**metadata, "item": opening, "variant": variant},
            ))
    return MemoryDataset(out, name=f"conspiracybench-{variant}")
