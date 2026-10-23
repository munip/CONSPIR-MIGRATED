# Combined judge page

Source of https://spacesheep.dev/@likelytobelaura/judge-trust, which combines this rerun with the
injection study in `../../../judge_injection/experiments/005-injection-resistance/`.

`extract.py` → `data.json` (reads this rerun's results, `validity_v1/`, and the Experiment 005 page
data); `build.py` inlines it into `template.html` → `site/index.html`. All metrics on the page are
computed in the browser from the raw per-reply outputs; bootstrap intervals use a fixed seed.
`extract.py` currently reads from a local checkout path and the Experiment 005 page's data.json;
adjust the paths at the top to rebuild elsewhere.
