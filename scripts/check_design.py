"""Checks the design of experiment requests: every experiment_requests/**/EXPERIMENT.md needs all six
stages filled in (Claim, Unit and metric, Controls, Data, Noise, Readout), and Controls has to name a
baseline. A request that fails isn't merged. Errors fail the check; warnings are printed.

  python scripts/check_design.py experiment_requests/<task>/<slug>/EXPERIMENT.md
  python scripts/check_design.py           every EXPERIMENT.md under experiment_requests/

See CONTRIBUTING.md, "Designing an experiment". The rules are the same as `stationhouse design check`
(stationhouse-cli, src/stationhouse_cli/design.py). Keep the two in sync.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (key, heading as written, other headings accepted for it)
STAGES = [
    ("claim", "Claim", ("hypothesis", "research question", "question")),
    ("metric", "Unit and metric", ("unit", "metric", "metrics", "measure")),
    ("controls", "Controls", ("baselines", "baselines and controls", "control")),
    ("data", "Data", ("dataset", "data and cleaning", "inputs")),
    ("noise", "Noise", ("noise floor", "variance", "seeds")),
    ("readout", "Readout", ("pre-registered readout", "pass rule", "success criteria", "what counts as a result")),
]

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
COMMENT = re.compile(r"<!--.*?-->", re.S)
PLACEHOLDER = re.compile(r"\b(TODO|TBD|FIXME)\b|^\s*\.\.\.\s*$", re.M)

BASELINE = re.compile(r"baseline|random|shuffl|constant|majority|chance|carry[- ]forward|copy|untrained|base model|"
                      r"null model|placebo|no[- ]op|ablat|prior|naive|always[- ]", re.I)
VERSION = re.compile(r"\b[0-9a-f]{7,64}\b|commit|sha|hash|version|v\d|revision|@\s*\w|\d{4}-\d{2}-\d{2}|"
                     r"pilot/|prefix", re.I)  # an atlas prefix is a fixed snapshot
SPLIT = re.compile(r"split|held[- ]out|hold[- ]out|test set|train|leak|contaminat|fresh|unseen|not copied|none copied", re.I)
FILTERS = re.compile(r"filter|dedup|duplicate|drop|remov|clean|exclud|kept|\bnone\b|as is", re.I)
REPEATS = re.compile(r"seed|rerun|re-run|repeat|bootstrap|confidence interval|\bci\b|std|standard (?:error|deviation)|"
                     r"variance|k\s*=\s*\d|\d+\s*(?:runs|samples|draws)|permutation|random[- ]neuron|"
                     r"threshold|margin|q99|percentile", re.I)  # distance from a threshold is the neuron hunt's version
REFUTE = re.compile(r"refut|falsif|null|no effect|fail|reject|inconclusive|would mean|wrong if|not support|negative", re.I)
NUMBER = re.compile(r"\d")


def _stage_for(title: str):
    t = re.sub(r"^[\d.)\s]+|[*_`:]", "", title).strip().lower()
    for key, name, others in STAGES:
        if any(t == n or t.startswith(n + " ") or t.startswith(n + "(") for n in (name.lower(), *others)):
            return key
    return None


def sections(text: str) -> dict:
    """{stage key: section text, comments removed}. A section runs to the next heading at its own
    level or above, so sub-headings inside it stay with it, and the first heading for a stage wins."""
    # Comments go first (keeping their line breaks), so a heading inside one doesn't count.
    lines = COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text).splitlines()
    found = {}
    for i, line in enumerate(lines):
        m = HEADING.match(line)
        if not m:
            continue
        key = _stage_for(m.group(2))
        if not key or key in found:
            continue
        level, body = len(m.group(1)), []
        for nxt in lines[i + 1:]:
            h = HEADING.match(nxt)
            if h and len(h.group(1)) <= level:
                break
            body.append(nxt)
        found[key] = "\n".join(body).strip()
    return found


def check(text: str) -> list[dict]:
    """[{level: error|warning, stage, message}]. Any error means the design isn't ready to run."""
    secs = sections(text)
    out = []

    def add(level, stage, message):
        out.append({"level": level, "stage": stage, "message": message})

    names = {key: name for key, name, _ in STAGES}
    for key, name, _ in STAGES:
        body = secs.get(key)
        if body is None:
            add("error", key, f"no `{name}` section")
        elif not body:
            add("error", key, f"`{name}` is empty")
        elif PLACEHOLDER.search(body):
            add("error", key, f"`{name}` still has a TODO/TBD or a bare `...` line")

    def filled(key):
        return bool(secs.get(key)) and not PLACEHOLDER.search(secs[key])

    if filled("controls") and not BASELINE.search(secs["controls"]):
        add("error", "controls", "Controls doesn't name a baseline: add at least one trivial one (constant, majority, "
                                 "copy the input, carry forward, base model, random units) and say what it scores")
    if filled("metric") and not NUMBER.search(secs["metric"]):
        add("warning", "metric", "Unit and metric gives no sample size: say how many units")
    if filled("data"):
        d = secs["data"]
        if not VERSION.search(d):
            add("warning", "data", "Data doesn't pin a version (commit, hash, revision or date)")
        if not SPLIT.search(d):
            add("warning", "data", "Data doesn't say how the evaluation set is kept apart (split, held-out, leakage)")
        if not FILTERS.search(d):
            add("warning", "data", "Data doesn't list its filters or cleaning steps (say 'none' if there are none)")
    if filled("noise") and not REPEATS.search(secs["noise"]):
        add("warning", "noise", "Noise doesn't say how it's estimated (seeds, reruns, bootstrap, random-unit baseline)")
    if filled("readout"):
        r = secs["readout"]
        if not REFUTE.search(r):
            add("warning", "readout", "Readout doesn't say which result would refute the claim")
        if not NUMBER.search(r):
            add("warning", "readout", "Readout has no number: give the threshold or effect size that counts")
    order = {k: i for i, (k, _, _) in enumerate(STAGES)}
    out.sort(key=lambda f: (f["level"] != "error", order[f["stage"]]))
    for f in out:
        f["section"] = names[f["stage"]]
    return out


def report(label, text) -> int:
    """Prints the findings for one design; returns how many are errors."""
    findings = check(text)
    errors = sum(f["level"] == "error" for f in findings)
    print(f"{label}: " + ("ready" if not errors else f"{errors} error(s)")
          + (f", {len(findings) - errors} warning(s)" if len(findings) > errors else ""))
    for f in findings:
        print(f"  {f['level']}: {f['section']}: {f['message']}")
    return errors


def main(argv):
    paths = [Path(p) for p in argv] or sorted((ROOT / "experiment_requests").rglob("EXPERIMENT.md"))
    if not paths:
        print("check_design: no EXPERIMENT.md under experiment_requests/; nothing to check.")
        return 0
    errors = sum(report(str(p), p.read_text(encoding="utf-8")) for p in paths)
    if errors:
        print("\nSettle the design before the request is merged: see CONTRIBUTING.md, \"Designing an experiment\", "
              "and experiment_requests/TEMPLATE.md.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
