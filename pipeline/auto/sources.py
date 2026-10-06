"""Load source documents from any URL, local text/HTML file, or pasted text.

No domain rules live here: a source is whatever the contributor points at. Trust comes later,
from verbatim quote containment (gates.py) and the human review step, not from the hostname.
Standard library only.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable

USER_AGENT = "ConspiracyBench-pipeline/0.2 (research prototype; pass --user-agent with your contact)"
MAX_BYTES = 5_000_000
TEXT_TYPES = ("text/plain", "text/markdown", "application/json", "application/xml", "text/xml")
HTML_TYPES = ("text/html", "application/xhtml+xml")
WIKI_HOSTS = ("wikipedia.org", "wikimedia.org", "wikidata.org", "wikiquote.org")


class SourceError(RuntimeError):
    """A source could not be loaded or turned into text."""


_SKIP = {"script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form", "template"}
_BLOCK = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table",
          "section", "article", "main", "blockquote", "pre", "dd", "dt", "figcaption"}
_MAIN = {"article", "main"}
_CITE_MARK = re.compile(r"\[(?:\d{1,3}|[a-z]|citation needed|note \d+)\]", re.I)


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.main_depth = 0
        self.all: list[str] = []
        self.main: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in _SKIP:
            self.skip += 1
        if tag in _MAIN:
            self.main_depth += 1
        if tag in _BLOCK:
            self._put("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in _SKIP and self.skip:
            self.skip -= 1
        if tag in _MAIN and self.main_depth:
            self.main_depth -= 1
        if tag in _BLOCK:
            self._put("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if not self.skip:
            self._put(data)

    def _put(self, s):
        self.all.append(s)
        if self.main_depth:
            self.main.append(s)


def clean_text(text: str) -> str:
    text = _CITE_MARK.sub("", text.replace("\xa0", " "))
    lines = (re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in text.split("\n"))
    return re.sub(r"\n{2,}", "\n\n", "\n".join(lines)).strip()


def html_to_text(html: str) -> tuple[str, str]:
    """Return (text, title). Prefers <article>/<main> content when it is substantial."""
    parser = _Text()
    parser.feed(html)
    parser.close()
    main, whole = clean_text("".join(parser.main)), clean_text("".join(parser.all))
    return (main if len(main) >= 500 else whole), clean_text(parser.title)


def parse_spec(spec: str) -> tuple[str, str | None]:
    """'<url-or-path>' or '<url-or-path> | <citation text>'."""
    left, sep, right = spec.partition(" | ")
    return left.strip(), (right.strip() or None) if sep else None


def _get(url: str, user_agent: str, timeout: int = 30) -> tuple[bytes, str, str]:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "text/html,text/plain,*/*;q=0.5"})
    last: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                body = resp.read(MAX_BYTES + 1)
                if len(body) > MAX_BYTES:
                    raise SourceError(f"{url}: larger than {MAX_BYTES // 1_000_000} MB; save the relevant text to a file")
                return body, resp.headers.get_content_type(), resp.headers.get_content_charset() or ""
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            break
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            break
    raise SourceError(f"GET {url} failed: {last}")


def _decode(body: bytes, charset: str) -> str:
    if not charset:
        m = re.search(rb"<meta[^>]+charset=[\"']?([\w-]+)", body[:4096], re.I)
        charset = m.group(1).decode("ascii") if m else "utf-8"
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def _to_text(body: bytes, ctype: str, charset: str, label: str) -> tuple[str, str]:
    raw = _decode(body, charset)
    if ctype in HTML_TYPES or (ctype not in TEXT_TYPES and re.search(r"<html|<body|<p[ >]", raw[:2000], re.I)):
        return html_to_text(raw)
    if ctype in TEXT_TYPES or ctype.startswith("text/"):
        return clean_text(raw), ""
    raise SourceError(f"{label}: unsupported content type {ctype!r} (PDF/other: extract the text to a .txt file and pass that)")


def load_source(spec: str, *, cache_dir: Path | str, user_agent: str = USER_AGENT, refresh: bool = False,
                fetcher: Callable[[str, str], tuple[bytes, str, str]] = _get) -> dict:
    """Return {uri, kind, citation, title, text, sha256, retrieved_on}. Never executes anything it fetches."""
    uri, citation = parse_spec(spec)
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme in ("http", "https"):
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / (hashlib.sha1(uri.encode("utf-8")).hexdigest() + ".json")
        if path.exists() and not refresh:
            rec = json.loads(path.read_text(encoding="utf-8"))
            if rec.get("uri") == uri:
                rec["citation"] = citation
                return rec
        body, ctype, charset = fetcher(uri, user_agent)
        text, title = _to_text(body, ctype, charset, uri)
        kind, retrieved = "url", date.today().isoformat()
    elif parsed.scheme in ("", ) or re.match(r"^[A-Za-z]:[\\/]", uri):
        p = Path(uri).expanduser()
        if not p.is_file():
            raise SourceError(f"{uri}: not a file (sources are http(s) URLs or local text/HTML files)")
        body = p.read_bytes()[:MAX_BYTES]
        if body.startswith(b"%PDF") or b"\x00" in body[:4096]:
            raise SourceError(f"{uri}: binary file (PDF/other: extract the text to a .txt file and pass that)")
        text, title = _to_text(body, "text/html" if p.suffix.lower() in (".html", ".htm") else "text/plain", "", uri)
        uri, kind, retrieved = str(p.resolve()), "file", date.today().isoformat()
        if not title:
            title = p.name
    else:
        raise SourceError(f"{uri}: only http(s) URLs and local files are supported")
    if len(text) < 200:
        raise SourceError(f"{uri}: extracted only {len(text)} characters of text; is it a login wall or script-rendered page?")
    rec = {"uri": uri, "kind": kind, "citation": citation, "title": title, "text": text,
           "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "retrieved_on": retrieved}
    if kind == "url":
        path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return rec


def is_wikimedia(uri: str) -> bool:
    host = urllib.parse.urlparse(uri).hostname or ""
    return any(host == h or host.endswith("." + h) for h in WIKI_HOSTS)


def sources_from_candidates(path: Path | str, topic: str, limit: int = 4) -> list[str]:
    """Bridge from discover_topics/harvest_references output: source URLs for one candidate topic."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line) if line.strip() else {}
        if rec.get("topic", "").casefold() == topic.casefold():
            ordered = [s["url"] for s in rec.get("candidate_sources", []) if s.get("source_hint") != "catalog"]
            return ordered[:limit]
    raise SourceError(f"topic {topic!r} not found in {path}")
