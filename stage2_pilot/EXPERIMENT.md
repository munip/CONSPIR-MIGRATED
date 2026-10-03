# Stage 2 six-question pilot: initial results

## Scope

Six evidence-ledger entries were prepared: three single-turn and
three multi-turn. The pilot tests deterministic rendering into the
existing ConspiracyBench schema and preserves a review-status gate.

The six example questions remain drafts for maintainer review and
have not been added to the official benchmark.

## Validation

- Six local schema-valid previews were generated.
- Twelve initial offline regression tests passed.
- Two additional tests cover the six-draft review safeguard and schema
  compatibility. Run the current test suite for the final total.
- Real previews were generated using a temporary in-memory review-status
  override. Saved drafts remain unapproved.

## Exploratory semantic-overlap audit

Inventory snapshot: September 27, 2026.
Main commit: d0e4268b9dffeb9a94827a22b425f3215ecefebb.
Inventory coverage: 17 distinct question texts from the merged corpus
and 17 active PRs.

Embedding model: BAAI/bge-small-en-v1.5.
Similarity measure: cosine similarity of normalized embeddings.
Exploratory review threshold: 0.6.

Four comparisons crossed that threshold:

| Candidate | Compared question | Similarity |
| --- | --- | ---: |
| Bermuda Triangle | Apollo 11 evidence | 0.634 |
| Face on Mars | Moon landing | 0.620 |
| Face on Mars | Apollo 11 evidence | 0.617 |
| Face on Mars | Smithsonian crystal skull | 0.604 |

The first three were comparisons against existing questions. The
fourth was between two questions within the pilot. Claim-level
inspection found different underlying claims in all four cases.

These examples illustrate potential false-positive similarity alerts.
They do not establish a general false-positive rate or show that the
threshold should be changed.

## Limitations

- The renderer reformats evidence notes; it does not independently
  invent questions or verify factual propositions.
- The six draft questions require maintainer review before any
  benchmark submission.
- Some cited primary-source pages could not be retrieved directly
  during the initial source audit; the ledger records those limitations.
- The PR inventory is a dated snapshot and must be refreshed.
- The current exploratory audit has no positive-control paraphrases
  of genuinely duplicate claims.
- The semantic-overlap comparison was run locally; a packaged,
  reproducible large-scale evaluation is future work.

## Compute

The initial renderer tests and embedding comparisons ran locally.
No additional compute is required to reproduce the current unit tests.

A larger evaluation with verified positive and negative duplicate
pairs, or a later generative question-production stage, may require
additional resources. Scope and compute requirements should be agreed
with maintainers before scaling.
