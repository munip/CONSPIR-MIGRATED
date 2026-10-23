"""Build site/: index.html (page + small data) and data-*.js files, each kept small for the uploader."""
import json
from pathlib import Path

H = Path(__file__).parent
data = json.load(open(H / "data.json"))
site = H / "conspiracybench-baseline"
site.mkdir(exist_ok=True)
for old in site.glob("data-*.js"):
    old.unlink()

dump = lambda x: json.dumps(x, separators=(",", ":"), ensure_ascii=False)
items, transcripts = data.pop("items"), data.pop("transcripts")
files = {"data-items.js": f"window.CB_ITEMS={dump(items)};"}
keys = list(transcripts)
half = len(keys) // 2
for i, part in enumerate((keys[:half], keys[half:]), 1):
    files[f"data-transcripts-{i}.js"] = f"Object.assign(window.CB_TRANSCRIPTS=window.CB_TRANSCRIPTS||{{}},{dump({k: transcripts[k] for k in part})});"
for name, body in files.items():
    (site / name).write_text(body)

scripts = "".join(f'<script src="{name}"></script>' for name in files)
html = (H / "template.html").read_text()
html = html.replace("/*DATA*/null", dump(data)).replace("<!--DATA-SCRIPTS-->", scripts)
(site / "index.html").write_text(html)
for f in sorted(site.iterdir()):
    if f.is_file() and not f.name.startswith("."):
        print(f"{f.name}: {f.stat().st_size / 1e3:.0f} KB")
