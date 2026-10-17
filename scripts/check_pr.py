#!/usr/bin/env python3
"""Minimum-bar check on a drafted pull request description.

Write your description to a file, then run:

    python scripts/check_pr.py pr_description.md

It checks that the description is non-empty, past a minimum length, and not
placeholder text (including the task page's template left unfilled). It can't
judge whether a description is *good*, only whether one exists. See
CONTRIBUTING.md step 8.
"""
import re
import sys

MIN_LENGTH = 40
PLACEHOLDER_PHRASES = ["todo", "tbd", "fill in", "n/a", "add a description", "lorem ipsum"]
PLACEHOLDER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in PLACEHOLDER_PHRASES) + r")(?!\w)", re.IGNORECASE
)
# Template markers from the task page's example: cb-XXXX and <...> slots.
# <https://...> autolinks are real content, not slots.
TEMPLATE_RE = re.compile(r"cb-XXXX|<(?!https?://)[^<>\n]{1,120}>", re.IGNORECASE)


def check(text: str) -> list[str]:
    problems = []
    stripped = text.strip()
    if not stripped:
        return ["Empty."]
    if len(stripped) < MIN_LENGTH:
        problems.append(f"Too short ({len(stripped)} chars, need at least {MIN_LENGTH}).")
    if PLACEHOLDER_RE.search(stripped):
        problems.append("Looks like placeholder text, not a real description.")
    leftovers = TEMPLATE_RE.findall(stripped)
    if leftovers:
        problems.append(
            "Template text left in: " + ", ".join(dict.fromkeys(leftovers))
            + ". Replace each one with your own words."
        )
    return problems


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_pr.py path/to/description.md")
        sys.exit(1)

    path = sys.argv[1]
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"No such file: {path}")
        print("Write your PR description to a file first, e.g.:")
        print("  cat > pr_description.md << 'EOF'")
        print("  Adds a question on ...: what it tests and why it's a good probe")
        print("  Ground truth source: https://...")
        print("  EOF")
        sys.exit(1)

    problems = check(text)
    if problems:
        print("FAIL — this description isn't ready yet:")
        for p in problems:
            print(f"  - {p}")
        print()
        print("A good PR description covers what you're proposing and why — e.g. what")
        print("the question tests, and the source behind its ground truth. If you're an")
        print("AI agent, see CONTRIBUTING.md's \"For AI coding agents\" section.")
        sys.exit(1)

    print(f'PASS — use this file with --commit-description "$(cat {path})"')


if __name__ == "__main__":
    main()
