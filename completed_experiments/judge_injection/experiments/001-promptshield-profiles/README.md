# Experiment 1: PromptShield profiles

Completed 2026-10-07. This is the first detector comparison in this series.
The existing frozen-only report is preserved as the Experiment 1 artifact.

- [Decision report](../../reports/promptshield-team-report-v1/README.md)
- [HTML](../../reports/promptshield-team-report-v1/report.html)
- [PDF](../../reports/promptshield-team-report-v1/report.pdf)
- [Per-case scans](../../reports/promptshield-team-report-v1/scans.json.gz)
- [Artifact hashes](../../reports/promptshield-team-report-v1/manifest.json)
- [Experiment record](experiment.json)

| Profile | Attacks caught | Benign false positives | Warm p50 / p95 ms |
|---|---:|---:|---:|
| Baseline | 13/70 | 2/70 | 22.1 / 150.9 |
| Rules + baseline | 70/70 | 2/70 | 22.6 / 151.7 |
| ProtectAI v2 + baseline, selected cutoff | 19/70 | 2/70 | 68.1 / 455.8 |
| Rules + ProtectAI v2 + baseline | 70/70 | 2/70 | 70.4 / 496.9 |

At default ML cutoff 0.5, ProtectAI + baseline catches 49/70 attacks and flags
22/70 benign cases. The selected cutoff is `0.9999990463256836`, selected on
these same cases within the two-FP budget. Each calibrated full profile flags
one of the 21 quoted/rejected controls. Both benign FPs come from baseline.

Rules ties combined on detection and avoids ML latency. The rules were designed
with these seven templates in view. No independent generality claim is made.
Timing is a single fixed-order CPU pass; hardware details were not fully recorded.
Earlier supplemental-case results remain historical and are excluded from this
series' active comparisons. The original fixture and outcomes are unchanged.
