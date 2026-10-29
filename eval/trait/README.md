# Response-robustness judge validation

`scripts/export_response_judgments.py` extracts per-turn judge outputs from an Inspect
`.eval` log. `scripts/validate_response_judge.py` compares those outputs with an
independently human-adjudicated reference set. It reports per-axis exact agreement,
Cohen's kappa, confusion counts, and coverage. Invalid judge outputs and unpaired units
are reported separately; missing labels are not treated as negatives.

Both `--human` and `--judge` inputs are UTF-8 JSONL with one row per assistant
turn:

```json
{"sample_id":"cb-0001","epoch":1,"turn_index":0,"factual_alignment":"matches","conspiracy_endorsement":"does_not_endorse","abstention":false}
```

`turn_index` starts at zero and follows the assistant response order in the
Inspect transcript. `epoch` distinguishes repeated samples (default `1`). The reference file must contain human labels; do not copy
model-judge outputs into it. These files are not populated with example labels
because no human-adjudicated response-robustness set has been collected yet.

```powershell
python scripts/export_response_judgments.py logs/response-robustness.eval `
  --output datasets/review/response-robustness-judge.jsonl
python scripts/validate_response_judge.py `
  --human datasets/review/response-robustness-human.jsonl `
  --judge datasets/review/response-robustness-judge.jsonl `
  --output results/response-robustness-judge-validation.json
```
