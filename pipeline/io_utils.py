"""Draft-only output helpers. Nothing here may write into questions/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DRAFT_STATUS = "needs_human_verification"


class OutputError(ValueError):
    """Refused to write the requested output."""


def ensure_safe_output(path: Path | str, repo_root: Path | str = REPO_ROOT) -> Path:
    resolved = Path(path).resolve()
    forbidden = (Path(repo_root) / "questions").resolve()
    if resolved == forbidden or forbidden in resolved.parents:
        raise OutputError("Pipeline drafts must never be written into questions/")
    return resolved


def write_jsonl(records, path, *, overwrite: bool = False, repo_root: Path | str = REPO_ROOT) -> int:
    resolved = ensure_safe_output(path, repo_root)
    records = list(records)
    if resolved.exists() and not overwrite:
        raise OutputError(f"{resolved} exists; pass --overwrite to replace it")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def read_jsonl(path: Path | str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def make_console_safe() -> None:
    """Escape characters the console cannot show instead of crashing (Windows cp1252/cp949)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(errors="backslashreplace")
