# Candidate-discovery pipeline (draft-only prototype)

Upstream layer for the "automated question-creation pipeline" task (Discussion #61).
It finds *candidate* topics and *candidate* sources for a human to verify. It does not
write questions, does not call a model, does not touch `questions/`, and never marks
anything verified. Every record has `status: "needs_human_verification"`.

> **Want finished questions, not just candidates?** See [`auto/README.md`](auto/README.md): `python -m pipeline.auto`
> takes any sources to three human-verified, multi-turn framings.

## Where it sits

```
Wikipedia "List of conspiracy theories"
  -> discover_topics.py      (this PR)  topics + list-page context
  -> harvest_references.py   (this PR)  each article's own cited references
  -> [later] propositions -> quote-containment check -> human verification
  -> [later] question/metadata generation -> #62 renderer -> validate / #64 duplicates
```

Wikipedia is a map, not a source: only the articles' *cited external references* are
emitted as candidate sources. Wikipedia/Wikimedia URLs are filtered out, and the `claim`
field is list-page context for orientation, not a verified claim.

## Run (from the repo root; standard library only)

```
python -m pipeline.discover_topics --limit 20
python -m pipeline.harvest_references --limit 5
python -m pipeline.harvest_references --topic chemtrail --topic "Flight 370"   # chosen topics only
```

Set a real contact in the User-Agent (Wikimedia policy):
`--user-agent "ConspiracyBench-pipeline/0.1 (HF: newsss; you@example.com)"`

Outputs go to `pipeline/out/` (git-ignored); fetched pages are cached in `pipeline/cache/`.
Existing outputs are never overwritten without `--overwrite`.

Offline: `--list-html saved.html` for discovery; `--html-dir DIR --offline` for harvesting
(files named `<wiki.safe_stem(title)>.html`).

## Tests

```
python -B -m unittest discover -s pipeline/tests -p "test_*.py" -v
```

Fixtures in `pipeline/fixtures/` are hand-written to imitate MediaWiki's HTML structure;
their text is invented.

## Output shape

`candidate_topics.jsonl`: `topic`, `claim`, `claim_origin`, `source_article_url`,
`list_section`, `link_origin` (`hatnote`|`inline`), `mentions`, `status`, `discovered_from`.
`claim_origin` is `wikipedia_list_context`, `enumeration_sentence` (one sentence listing many
topics, so `claim` is left empty rather than copying a shared blob) or `none`.

`candidates_with_sources.jsonl`: the above plus `candidate_sources[]`
(`url`, `archive_url` (Wayback, if the article cited one), `link_text`, `citation_text`,
`source_hint`, `ref_ids`), `sources_meta`, and `article_lead` (the article's lead paragraph, for
reviewer orientation only; never a citation). `source_hint` (government / academic /
fact_check / other) is a domain heuristic for ordering only, not a quality judgement.

## Reading the output

- `claim` is list-page *context*, not the allegation: sometimes it states the theory, sometimes
  the debunking. The allegation and verified propositions are produced in a later step.
- `lead_signal: false` means the article's lead never mentions conspiracy/claims, so it is
  probably a generic article (a team, a historical period) that the list merely links to.
- `catalog` sources (WorldCat/ADS lookups, mirrors) are dropped when the same reference has a
  real link, and ranked last otherwise. `sources_meta.resolved_title` shows redirect targets.
- `duplicate_of` marks a topic whose link redirects to an article already harvested (e.g. an event
  article and its "conspiracy theories" redirect), so the same sources aren't reviewed twice.
- `source_hint` ordering is a weak domain heuristic, not relevance: an irrelevant `.gov` page can
  outrank a relevant news report. Relevance comes from the later proposition-matching step.
- Sources are attached to the *article*, not to a specific claim; matching propositions to
  sources is the later quote-containment step.

## Known limits

- Pages reachable under different hostnames (e.g. `nla.gov.au` vs `trove.nla.gov.au`, `www2.` vs `www.`)
  are not merged, so a few sources appear twice.

- Topic filter: titles with strong cues (conspiracy, hoax, theory, cover-up ...) are kept; weaker
  cues ("Death of ...", "... incident", any "Main article" hatnote) are kept only when the
  surrounding sentence mentions conspiracy/allegation. Expect misses and some noise, plus
  related duplicates (e.g. an event article and its "conspiracy theories" article). Humans review.
- Some candidates are historically verified conspiracies or very recent events; flag those in the
  contested-vs-settled step rather than here.
- Encoding: every text read/write names `encoding="utf-8"` and the CLIs escape characters the
  console can't show (see Discussion #84 / PR #88); tests enforce both.
- Windows console: `type` may show garbled characters (e.g. `M─üori`). The files are UTF-8; run
  `chcp 65001` first or open them in an editor.
- Link liveness, quote-containment and contested-vs-settled checks are not built yet.
- Not yet checked against the live Wikipedia HTML at scale.
