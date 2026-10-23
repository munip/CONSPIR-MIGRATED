"""Data for the combined judge page: validity-v1 rerun on final labels (v1.2) + Experiment 005 injection study."""
import csv, json
from pathlib import Path
WT = Path.home() / "ConspiracyBench/.claude/worktrees/reliability-v1.2-rerun/completed_experiments/judge_reliability"
NEW = WT / "results/rerun_v1.2_2026-10-22"
OLD = WT / "results"
L = ["hold", "diverge", "abstain"]

units = [json.loads(x) for x in (WT / "validity_v1/units.jsonl").read_text().splitlines() if x.strip()]
key = {r["unit_id"]: r["model"].split("/")[-1].replace("cb-", "") for r in csv.DictReader(open(WT / "validity_v1/key.csv"))}
ref = {r["unit_id"]: r for r in csv.DictReader(open(WT / "validity_v1/labels/reference_v1.2.csv"))}

def preds(path):
    out = {"no_context": {}, "preceding_exchange": {}}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r["status"] != "ok": continue
        p = r["probabilities"]
        out[r["variant"]][r["unit_id"]] = [r["choice"]] + [round(p[l], 4) for l in L]
    return out

rel = {"units": [], "preds": {}, "prev": {}}
for u in units:
    r = ref[u["case_id"]]
    rel["units"].append({
        "id": u["case_id"], "turn": u["turn"], "framing": u.get("framing"), "solver": key.get(u["case_id"], ""),
        "item": u["item"]["id"], "q": u["item"]["question"], "gt": u["item"]["ground_truth_answer"],
        "div_if": u["item"].get("diverges_if") or [], "transcript": [[m["role"], m["content"]] for m in u["transcript"] or []],
        "reply": u["model_answer"], "v12": r["label"],
    })
for j in ["pplx", "clef", "kev", "laya"]:
    rel["preds"][j] = preds(NEW / f"{j}_validity_v1_rubric_v2/results.jsonl")
    prev = preds(OLD / f"{j}_validity_v1_rubric_v2/results.jsonl")
    pairs = [(rel["preds"][j][v][u], prev[v][u]) for v in prev for u in prev[v] if u in rel["preds"][j][v]]
    rel["prev"][j] = {"compared": len(pairs), "same": sum(a[0] == b[0] for a, b in pairs),
                      "max_dp": max(abs(a[i] - b[i]) for a, b in pairs for i in (1, 2, 3))}
info = [json.loads(l) for l in (NEW / "pplx_validity_v1_rubric_v2/run_info.jsonl").read_text().splitlines()]
rel["run"] = {"commit": info[-1]["suite_commit"], "started": info[-1]["started"], "gpu": "H100 80GB (Modal)"}
exp5 = json.loads((Path.home() / "exp005-viz/data.json").read_text())
data = {"rel": rel, "inj": exp5}
(Path(__file__).parent / "data.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
print("units", len(rel["units"]), "| bytes", len(json.dumps(data)))
