#!/usr/bin/env python3
"""Inline data.json into template.html -> index.html (a self-contained page for spacesheep)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / 'data.json').read_text(encoding='utf-8'))
page = (HERE / 'template.html').read_text(encoding='utf-8').replace(
    '__DATA__', json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/'))
(HERE / 'index.html').write_text(page, encoding='utf-8')
print(len(page.encode('utf-8')), 'bytes')
