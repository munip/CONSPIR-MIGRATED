"""Step 2: fetch candidate source pages, with a Wayback fallback. Draft-only.

Given a topic's `candidate_sources` (from harvest_references), download each page's text so a later
step can check, by string containment, that a proposition's supporting quote really appears in the
source. HTTP status alone is not trusted: a 403 to a script can be a live page (see #61 / #51), so a
failed fetch falls back to the article's own Wayback snapshot, then to the Wayback availability API.
This module records HOW each text was obtained and never marks anything verified.

Run from the repo root (a topic filter is required, to avoid fetching hundreds of pages by accident):
    python -m pipeline.fetch_sources --topic chemtrail --per-topic 6
PDF text needs the optional `pypdf` package; without it PDFs are reported as `unsupported_pdf`.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Iterator, NamedTuple

from . import wiki
from .harvest_references import select_candidates
from .io_utils import OutputError, make_console_safe, read_jsonl, write_jsonl

logging.getLogger("pypdf").setLevel(logging.ERROR)  # silence per-font "fontTools is required" warnings

MAX_BYTES = 8_000_000
AVAILABILITY_API = "https://archive.org/wayback/available?url="


class Response(NamedTuple):
    status: int
    headers: dict
    body: bytes
    final_url: str


class FetchFailure(RuntimeError):
    """Network-level failure (DNS, TLS, timeout, ...) rather than an HTTP status."""


def http_get(url: str, user_agent: str = wiki.DEFAULT_USER_AGENT, timeout: int = 30) -> Response:
    request = urllib.request.Request(
        url, headers={"User-Agent": user_agent, "Accept": "text/html,application/pdf,*/*;q=0.8"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return Response(resp.status, headers, resp.read(MAX_BYTES), resp.geturl())
    except urllib.error.HTTPError as exc:
        headers = {k.lower(): v for k, v in exc.headers.items()} if exc.headers else {}
        return Response(exc.code, headers, b"", exc.geturl() or url)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise FetchFailure(str(exc)) from exc


def wayback_lookup(url: str, get: Callable[[str], Response] = http_get) -> str | None:
    """Closest Wayback snapshot for `url` via the availability API, or None."""
    try:
        resp = get(AVAILABILITY_API + urllib.parse.quote(url, safe=""))
        if resp.status != 200:
            return None
        closest = (json.loads(resp.body.decode("utf-8")).get("archived_snapshots") or {}).get("closest") or {}
    except (FetchFailure, ValueError, AttributeError):
        return None
    if closest.get("available") and str(closest.get("status")) == "200" and closest.get("url"):
        return closest["url"]
    return None


_SNAPSHOT = re.compile(r"^(https?://web\.archive\.org/web/\d{4,14})(?=/)", re.IGNORECASE)


def raw_snapshot_url(url: str) -> str:
    """Ask Wayback for the original bytes (`id_`), without its toolbar or rewritten links."""
    return _SNAPSHOT.sub(r"\1id_", url, count=1)


class _TextParser(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "template", "head"}
    _BLOCK = {"p", "div", "br", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6",
              "section", "article", "table", "ul", "ol", "blockquote", "pre"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1
        elif tag in self._BLOCK:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in self._SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in self._BLOCK:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextParser()
    parser.feed(html)
    parser.close()
    return re.sub(r"\s+", " ", "".join(parser.parts)).strip()


def pdf_to_text(data: bytes) -> str | None:
    """Text of a PDF, "" if unreadable, or None when `pypdf` is not installed."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    try:
        reader = PdfReader(io.BytesIO(data))
        return re.sub(r"\s+", " ", " ".join((page.extract_text() or "") for page in reader.pages)).strip()
    except Exception:  # noqa: BLE001 - damaged or encrypted PDFs raise many different errors
        return ""


def text_source_url(result: dict) -> str | None:
    """The exact URL the cached text was downloaded from (`final_url` is the ORIGINAL attempt)."""
    if result.get("via") == "wayback" and result.get("snapshot_url"):
        return raw_snapshot_url(result["snapshot_url"])
    return result.get("final_url") if result.get("via") == "original" else None


def _decode(resp: Response) -> str:
    match = re.search(r"charset=([\w-]+)", resp.headers.get("content-type", ""), re.IGNORECASE)
    for encoding in ([match.group(1)] if match else []) + ["utf-8"]:
        try:
            return resp.body.decode(encoding, errors="replace")
        except LookupError:
            continue
    return resp.body.decode("utf-8", errors="replace")


def extract_text(resp: Response) -> tuple[str | None, str]:
    """Return (text, kind). text is None for a PDF we cannot read or an unsupported type."""
    ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
    if resp.body[:5] == b"%PDF-" or ctype == "application/pdf":
        return pdf_to_text(resp.body), "pdf"
    if not ctype or ctype.startswith("text/") or "html" in ctype or "xml" in ctype:
        return html_to_text(_decode(resp)), "html"
    return None, "unsupported"


def _attempt(url: str, get: Callable[[str], Response]) -> dict:
    try:
        resp = get(url)
    except FetchFailure as exc:
        return {"http_status": None, "error": str(exc), "text": None, "kind": None, "final_url": url}
    if resp.status != 200:
        return {"http_status": resp.status, "error": f"HTTP {resp.status}", "text": None,
                "kind": None, "final_url": resp.final_url}
    text, kind = extract_text(resp)
    return {"http_status": 200, "error": None, "text": text, "kind": kind, "final_url": resp.final_url}


def _snapshots(url: str, archive_url: str | None, lookup: Callable[[str], str | None]) -> Iterator[str]:
    if archive_url:
        yield archive_url
    found = lookup(url)  # lazy: only reached if the article's own snapshot is missing or failed
    if found and found != archive_url:
        yield found


def fetch_source(url: str, *, archive_url: str | None = None,
                 get: Callable[[str], Response] = http_get,
                 lookup: Callable[[str], str | None] = wayback_lookup) -> dict:
    """Fetch one source. Statuses: ok, unavailable, unsupported_type, unsupported_pdf, empty_text."""
    first = _attempt(url, get)
    record = {"url": url, "final_url": first["final_url"], "http_status": first["http_status"],
              "via": None, "snapshot_url": None, "kind": first["kind"], "text": None,
              "status": None, "note": ""}
    if first["error"] is None:
        if first["kind"] == "unsupported":
            record["status"] = "unsupported_type"
        elif first["text"] is None:
            record.update(status="unsupported_pdf", note="install pypdf to read PDF text")
        elif not first["text"]:
            record["status"] = "empty_text"
        else:
            record.update(status="ok", via="original", text=first["text"])
        return record
    problems: list[str] = []
    for snapshot in _snapshots(url, archive_url, lookup):
        second = _attempt(raw_snapshot_url(snapshot), get)
        record["snapshot_url"] = snapshot
        if second["error"] is None and second["text"]:
            record.update(status="ok", via="wayback", text=second["text"], kind=second["kind"],
                          note=f"original failed: {first['error']}")
            return record
        problems.append(second["error"] or "snapshot had no text")
    record["status"] = "unavailable"
    record["note"] = f"original failed: {first['error']}; " + (
        "snapshot failed: " + "; ".join(problems) if problems else "no snapshot found")
    return record


def fetch_cached(url: str, *, archive_url: str | None = None, text_dir: Path, refresh: bool = False,
                 delay: float = 1.0, get: Callable[[str], Response] = http_get,
                 lookup: Callable[[str], str | None] = wayback_lookup,
                 sleep: Callable[[float], None] = time.sleep) -> dict:
    """fetch_source plus an on-disk cache: <text_dir>/<key>.json (metadata) and <key>.txt (text)."""
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    meta_path, text_path = text_dir / f"{key}.json", text_dir / f"{key}.txt"
    if meta_path.exists() and not refresh:
        record = json.loads(meta_path.read_text(encoding="utf-8"))
        record["from_cache"] = True
        record["text_path"] = str(text_path) if text_path.exists() else None
        return record
    if delay:
        sleep(delay)
    record = fetch_source(url, archive_url=archive_url, get=get, lookup=lookup)
    text = record.pop("text")
    record["retrieved_on"] = date.today().isoformat()
    record["chars"] = len(text) if text else 0
    record["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None
    record["text_path"] = None
    text_dir.mkdir(parents=True, exist_ok=True)
    if text:
        text_path.write_text(text, encoding="utf-8")
        record["text_path"] = str(text_path)
    meta_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    record["from_cache"] = False
    return record


def main(argv: list[str] | None = None, *, get=None, lookup=None, sleep=time.sleep) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", type=Path, default=Path("pipeline/out/candidates_with_sources.jsonl"))
    p.add_argument("--out", type=Path, default=Path("pipeline/out/source_manifest.jsonl"))
    p.add_argument("--text-dir", type=Path, default=Path("pipeline/cache/source_text"))
    p.add_argument("--topic", action="append", required=True,
                   help="Only topics containing this text (repeatable, case-insensitive)")
    p.add_argument("--per-topic", type=int, default=8, help="Max sources fetched per topic")
    p.add_argument("--hint", action="append", choices=["government", "academic", "fact_check", "other"],
                   help="Only sources with this source_hint (repeatable); catalog links are always skipped")
    p.add_argument("--user-agent", default=wiki.DEFAULT_USER_AGENT)
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    make_console_safe()
    get = get or (lambda u: http_get(u, args.user_agent))
    lookup = lookup or (lambda u: wayback_lookup(u, get))
    try:
        topics = select_candidates(read_jsonl(args.sources), args.topic)
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not topics:
        print("Error: no candidate topics matched.", file=sys.stderr)
        return 1
    rows: list[dict] = []
    for topic in topics:
        chosen = [s for s in topic.get("candidate_sources", [])
                  if s["source_hint"] != "catalog" and (not args.hint or s["source_hint"] in args.hint)]
        for source in chosen[: args.per_topic]:
            result = fetch_cached(source["url"], archive_url=source.get("archive_url"),
                                  text_dir=args.text_dir, refresh=args.refresh, delay=args.delay,
                                  get=get, lookup=lookup, sleep=sleep)
            rows.append({"topic": topic["topic"], "source_hint": source["source_hint"],
                         "link_text": source.get("link_text", ""), "article_archive_url": source.get("archive_url"),
                         **{k: result.get(k) for k in ("status", "via", "snapshot_url", "http_status", "final_url",
                                                       "kind", "chars", "sha256", "text_path", "retrieved_on",
                                                       "note", "from_cache")},
                         "text_url": text_source_url(result), "url": source["url"]})
    try:
        write_jsonl(rows, args.out, overwrite=args.overwrite)
    except OutputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    counts = Counter(r["status"] for r in rows)
    print(f"{len(rows)} sources from {len(topics)} topic(s) -> {args.out}")
    for status, n in counts.most_common():
        print(f"  {n:3d}  {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())