# Judge reliability explorer

`index.html` is a self-contained interactive page with every judge-reliability result on the
validity set: open it in a browser (no server needed). It covers:

- **Experiments** as presets: the #365 baseline, chat vs decision judges, the reference audit,
  the six human rules, the endorsement rules, the diverge threshold, conversation context, and the
  current best setup.
- **Interventions:** up to four arms side by side, each a rule set (original, six human rules,
  endorsement rules), conversation context (none, or on later turns) and an optional P(diverge)
  threshold.
- **Labels:** #364, the revised labels (`validity_v1/labels/new_label_agreement.csv`), only units
  both labellers agreed on, or the two-human slice; and replies: all, opening turns or later turns.
- Per judge and arm: accuracy, κ, diverge precision and recall, hold recall, share labelled
  diverge, a confusion matrix, and every reply with its question, checklists and labels.

All metrics are computed in the page from the raw per-reply outputs in `../results/`. Qwen chat
judges are left out on purpose. Rebuild after any result changes:

```bash
python completed_experiments/judge_reliability/judge-explorer/build_data.py   # -> data.json
python completed_experiments/judge_reliability/judge-explorer/build_page.py   # -> index.html
```

Headline (endorsement rules, context on later turns, revised labels): pplx-decider and Clef both
92% accurate on all 150 replies (κ 0.80 and 0.81). The revised labels are partly model-proposed;
see `../validity-v1-label-review/RESULTS.md`.
