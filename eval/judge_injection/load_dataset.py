"""Load the frozen judge-injection release and reproduce scorer scan inputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RELEASE_PATH = Path(__file__).with_name("release-v1.0.0.jsonl")


def load_cases(path: str | Path = RELEASE_PATH) -> list[dict[str, Any]]:
    """Return release rows in their frozen experiment order."""
    cases = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return cases


def project_scan_input(case: dict[str, Any]) -> str:
    """Match ``conspiracy_judge``: cumulative assistant answers, joined by ``\n\n``.

    User turns (including the question and pressure prompts), the source item, and
    trusted grading guidance are intentionally excluded.
    """
    assistant_answers = [
        message["content"]
        for message in (case.get("transcript") or [])
        if message.get("role") == "assistant"
    ]
    assistant_answers.append(case["model_answer"])
    return "\n\n".join(assistant_answers)
