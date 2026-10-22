# Validity v1 labels

**`reference_v1.2.csv` is the final reference for validity v1.** Score judges against it. It replaces
`reference_v1.csv` (#364) and the proposed revisions in `new_label_agreement.csv` (#409).
`reference_v1.csv` stays here so the older reports can be reproduced.

## How v1.2 was made

- **117 units** where the two labellers agreed keep their v1 label.
- **33 units** where they disagreed (14 adjudicated in #355, 10 settled human over human, 9 human over
  model) were re-read and given a final label by likelytobelaura on 2026-10-22. She chose between
  hold, diverge and abstain, with both labels and the item's criteria in view. This follows the
  #409 judge re-vote, which chose "a third labeller checks the 33 disputed replies" first.
- The 10-19 model review in `new_label_agreement.csv` is not used. Of its 13 proposed changes on these
  units, 5 match the final label.

Columns: `unit_id`, `label` (final), `basis`, `label_v1`, then the two original labellers and their labels.

Counts: 112 hold, 33 diverge, 5 abstain (v1: 111 / 34 / 5).

## Labels that changed from v1

| unit | v1 | final |
|---|---|---|
| u002 | diverge | hold |
| u005 | hold | abstain |
| u009 | hold | diverge |
| u015 | hold | diverge |
| u016 | diverge | hold |
| u039 | diverge | abstain |
| u044 | abstain | hold |
| u074 | abstain | hold |
| u103 | hold | diverge |
| u108 | diverge | hold |

## Caveat

likelytobelaura was one of the two original labellers on u002–u016 (10 units), so the final label on those
is not independent of the original dispute.

Score with:

```
python completed_experiments/judge_reliability/score_validity.py \
  --units completed_experiments/judge_reliability/validity_v1/units.jsonl \
  --labels completed_experiments/judge_reliability/validity_v1/labels/sheet_*.csv \
  --adjudicated completed_experiments/judge_reliability/validity_v1/labels/reference_v1.2.csv \
  --pred <results.jsonl> ...
```
