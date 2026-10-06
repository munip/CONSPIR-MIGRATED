# Automated question pipeline (`pipeline.auto`)

> **New contributor? Start with [`GUIDE.md`](./GUIDE.md)**: step by step, guardrails, and data-creation tips. This file is the reference.

Turns **any sources** (URLs, saved HTML, text files) into a draft ConspiracyBench claim: **three items
(neutral, leading, adversarial) that share a `claim_id` and are all multi-turn** (2-3 `pressure_turns` each).
A local model drafts; code checks; **a person verifies** before anything can be exported.

```
sources (any URL / .html / .txt)
  -> propositions, each with a VERBATIM quote          gate: quote must be in the stored source text,
                                                              and every number/name in the claim must be in the quote
  -> claim label (claim_id, category, region, allegation)
  -> ground-truth answer from the propositions only    gate: no number/name that its cited propositions don't contain
  -> 3 CANDIDATES per framing (neutral / leading / adversarial), each multi-turn
       deterministic gates mark a candidate BLOCKED (2-3 turns, schema-valid, band matches axes, adversarial question
       uses the allegation's own terms and does not hedge it); BLOCKED candidates cannot be chosen until edited clean
       model audits are advice only (flags): framing reads as intended; a turn names a new fact; a turn does not push;
       a turn asserts the state of the evidence
  -> ledger.json + review.md                           status: needs_human_verification
  -> YOU: review (approve / reject / edit each claim and the answer; PICK one candidate per framing, or edit one)
  -> finalize: re-run every gate, refuse anything unapproved -> final/<claim_id>.jsonl (+ provenance)
```

It is **URL-agnostic**: no domain list decides what is trustworthy. What it proves is "this exact sentence is in
this exact text, hash-stamped at retrieval"; *you* decide whether the source deserves belief.

## Setup (local only, no API key)

```bash
python3 -m venv .venv && .venv/bin/pip install jsonschema
ollama pull qwen2.5:7b            # any 3B-8B instruct model; or point --backend openai at llama.cpp / vLLM
```

Source text and drafts only go to a loopback model server. A non-local `--base-url` is refused unless you pass
`--allow-remote-llm`.

## Use

```bash
# 1. draft (minutes on a laptop). Mix any sources; `path | citation` is required for local files.
.venv/bin/python -m pipeline.auto build --topic "Harrow Point blackout" \
    --source https://example.org/inquiry-report \
    --source "notes/report.txt | Northern Lights Board, inquiry report, 1987"

# 2. read <run>/review.md (all candidates side by side), then verify and choose in the terminal
#    (--candidates N changes how many drafts per framing; default 3)
.venv/bin/python -m pipeline.auto review pipeline/out/auto/<claim-id> --reviewer <your-hf-username>

# 3. (after any hand edit of ledger.json) re-run all deterministic gates
.venv/bin/python -m pipeline.auto check pipeline/out/auto/<claim-id>

# 4. export what you approved
.venv/bin/python -m pipeline.auto finalize pipeline/out/auto/<claim-id>
```

`finalize` writes `final/<claim_id>.jsonl` (3 lines, `id: "cb-pending"`) and `<claim_id>.provenance.json` (the
quotes, spans and hashes). It never writes into `questions/`. You then copy the `.jsonl` into `questions/pending/`
and follow [`CONTRIBUTING.md`](../../CONTRIBUTING.md) steps 6-7 (`validate_questions.py`, `check_duplicates.py`) and open the PR.
Attach the provenance file so maintainers can re-check the quotes.

Feeding from the discovery layer: `build --from-candidates pipeline/out/candidates_with_sources.jsonl --topic "<candidate topic>"`.

## Choosing candidates

For each framing `review` lists the drafts with their pressure turns, judge conditions, and any BLOCKED reason or flag:

```
choose: <number> pick, e<number> edit, s skip framing, q quit
```

Picking a candidate is your approval of it. `e2` lets you rewrite candidate 2's question and turns (blank keeps the
current text); the deterministic gates re-run on your edit, so you can fix a hedged adversarial question ("allegedly...")
in seconds instead of rewriting it from scratch. The model's own audit no longer applies to an edited candidate.
The ground-truth answer is not offered as candidates: it is built only from approved claims, and you can edit it in `ledger.json`.

## What you must verify (the tool cannot)

1. **Each claim is true and its quote supports it.** The gate proves the quote exists, not that it means what the
   claim says or that the source is right. Open the source.
2. **The allegation is a real belief.** It is model-written and unsourced; it only describes the narrative.
3. **Framing.** neutral = asks openly; leading = presupposes something favouring the theory but a truthful answer
   exists in its frame; adversarial = states the false claim as fact. See `SCHEMA.md#choosing-framing`.
4. **Pressure turns press without inventing facts.** Reviewer flags in `review.md` list terms in questions/turns
   that appear in no source; check those first.
5. **Settled, not contested** (inclusion rule). The pipeline cannot judge this. Check `consensus_as_of`.

## When a build ends `needs_edit` / `failed`

`report.json` and `review.md` say why. Typical causes: a script-rendered or login-walled page (extracts too little
text: save the page's text to a `.txt` file), a topic the sources barely cover (fewer than 2 verified claims), or a model too
small to satisfy the framing audit (try a bigger `--model`). If every draft for a framing is BLOCKED the status is `needs_edit`: open `review` and edit one (or write it in
`ledger.json`). Nothing is exported until you have chosen an approved item for all three framings.

### Writing a framing yourself

You can also add a candidate by hand: in `ledger.json` copy one under `candidates.<framing>`, give it a new `id` and
`"blocking": []`, and edit its fields. It goes through the same gates and the same approval as a drafted one.

## Limits

- English text only. PDFs must be converted to text first.
- Quote matching is exact after whitespace/quote/dash/case normalisation; a quote spanning a table or footnote may fail.
- Entity grounding is a heuristic (numbers and capitalised words). It catches invented dates and names, not subtle
  misstatements; that is what step 1 of your review is for.
- Model audits are a second opinion from the same small model class, not ground truth. Only the high-precision ones
  block (framing, a *named* new fact); noisier ones become flags in `review.md`. Pressure turns often assert things like
  "there was no physical evidence" as fact: that can be false against your sources, so read every flag.
- Difficulty axes are model-estimated; check them against the `SCHEMA.md` rubric.
- With a 7B model, adversarial drafts are often hedged ("allegedly...") and arrive BLOCKED; plan to edit one. A larger
  local model should improve this (untested). More `--candidates` costs one model call plus audits per extra draft.
- Semantic duplicate detection stays with `scripts/check_duplicates.py`; the pipeline only blocks exact repeats.

Tests: `.venv/bin/python -B -m unittest discover -s pipeline/tests -p "test_*.py"` (offline; the model is scripted).
