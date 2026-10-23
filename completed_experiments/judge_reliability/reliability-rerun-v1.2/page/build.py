import json
from pathlib import Path

H = Path(__file__).parent
data = json.load(open(H / "data.json"))
html = (H / "template.html").read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
(H / "site").mkdir(exist_ok=True)
(H / "site/index.html").write_text(html)
print(len(html), "bytes")
