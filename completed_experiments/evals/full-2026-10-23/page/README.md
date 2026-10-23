# Results page

Source of <https://spacesheep.dev/@likelytobelaura/conspiracybench-baseline>.

```bash
python build_data.py   # reads ../outputs/<model>/{units.jsonl, pplx/results.jsonl, report/} -> data.json
python build.py        # data.json + template.html -> conspiracybench-baseline/ (index.html + data-*.js)
```

`build_data.py` needs each model's `units.jsonl`, which `inspect_eval/export_units.py` regenerates from
the Inspect logs. Paths in both scripts point at the machine the page was built on; adjust `RUN` and
`H` to rebuild elsewhere. The data is split into several files because spacesheep rejects a deploy
that uploads too much new data in one call: add the files one deploy at a time.
