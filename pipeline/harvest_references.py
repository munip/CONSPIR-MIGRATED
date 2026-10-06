"""Step 1b: harvest each candidate article's OWN cited references as candidate sources.

Reads candidate_topics.jsonl, fetches each linked Wikipedia article, and lists the
external URLs in its reference list. Wikipedia/Wikimedia URLs are never emitted as
sources. Wayback snapshots are paired with their original URL. `source_hint` is a
cheap domain heuristic for ordering only; a human decides what counts as primary.

Run from the repo root:
    python -m pipeline.harvest_references --limit 5
    python -m pipeline.harvest_references --html-dir pipeline/fixtures/articles --offline
"""
from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import wiki
from .io_utils import OutputError, make_console_safe, read_jsonl, write_jsonl

WIKI_HOSTS = (
    "wikipedia.org", "wikimedia.org", "wikidata.org", "mediawiki.org", "wikisource.org",
    "wikiquote.org", "wikibooks.org", "wiktionary.org", "wikinews.org", "wikiversity.org",
    "wikivoyage.org",
)
_WAYBACK = re.compile(r"^https?://web\.archive\.org/web/\d{4,14}(?:[a-z_]{2,3})?/(https?:/{1,2}.+)$", re.I)
_GOV = (".gov", ".mil", ".gov.uk", ".gov.au", ".gc.ca", ".gov.in", ".nic.in")
_GOV_HOSTS = ("europa.eu", "un.org", "who.int", "parliament.uk", "archives.gov")
_ACADEMIC = (".edu", ".ac.uk", ".edu.au")
_ACADEMIC_HOSTS = (
    "doi.org", "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "jstor.org", "sciencedirect.com",
    "nature.com", "springer.com", "wiley.com", "tandfonline.com", "sagepub.com", "arxiv.org",
    "science.org", "thelancet.com", "bmj.com", "jamanetwork.com",
)
_FACT_CHECK = ("snopes.com", "politifact.com", "factcheck.org", "fullfact.org")
_CATALOG = (
    "worldcat.org", "adsabs.harvard.edu", "ghostarchive.org", "archive.ph", "archive.today",
    "archive.is", "isbnsearch.org", "openlibrary.org", "lccn.loc.gov",
)  # lookup/duplicate/mirror links, not sources of their own
_RANK = {"government": 0, "academic": 1, "fact_check": 2, "other": 3, "catalog": 4}
_GENERIC_LINK_TEXT = {"", "the original", "original", "archived", "archive", "link"}
LEAD_SIGNAL_RE = re.compile(
    r"conspir|theor|hoax|alleg|false|myth|pseudo|debunk|misinformation|disinformation|"
    r"cover[- ]?up|denial|erroneous|unfounded|baseless|claim", re.IGNORECASE)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _host_matches(host: str, suffixes) -> bool:
    return any(host == s.lstrip(".") or host.endswith(s if s.startswith(".") else "." + s) for s in suffixes)


def normalize_url(href: str) -> str | None:
    href = (href or "").strip()
    if href.startswith("//"):
        href = "https:" + href
    return href if re.match(r"^https?://", href, re.I) else None


def split_wayback(url: str) -> tuple[str, str | None]:
    """Return (original_url, archive_url_or_None)."""
    m = _WAYBACK.match(url)
    if not m:
        return url, None
    return re.sub(r"^(https?):/(?!/)", r"\1://", m.group(1)), url


def is_wiki_host(url: str) -> bool:
    return _host_matches((urlparse(url).hostname or "").lower(), WIKI_HOSTS)


def lead_signal(lead: str) -> bool:
    """True if the article's lead talks about conspiracy/claims; False flags generic articles."""
    return bool(LEAD_SIGNAL_RE.search(lead or ""))


def source_hint(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if _host_matches(host, _CATALOG):
        return "catalog"
    if _host_matches(host, _ACADEMIC_HOSTS):  # before .gov: PubMed/PMC live under nih.gov
        return "academic"
    if _host_matches(host, _GOV) or _host_matches(host, _GOV_HOSTS):
        return "government"
    if _host_matches(host, _ACADEMIC):
        return "academic"
    if _host_matches(host, _FACT_CHECK):
        return "fact_check"
    return "other"


def _url_key(url: str) -> str:
    """Identity of a page ignoring http/https, a leading www., trailing slash and fragment."""
    p = urlparse(url)
    host = (p.hostname or "").lower()
    host = host[4:] if host.startswith("www.") else host
    return f"{host}{unquote(p.path).rstrip('/')}?{unquote(p.query)}"


def _generic(text: str) -> bool:
    return text.strip().lower() in _GENERIC_LINK_TEXT


class _RefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.refs: list[dict] = []
        self._ol_depth = 0
        self._li: dict | None = None
        self._li_depth = 0
        self._skip: tuple[str, int] | None = None
        self._link: dict | None = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        classes = (a.get("class") or "").split()
        if self._skip:
            if tag == self._skip[0]:
                self._skip = (tag, self._skip[1] + 1)
            return
        if tag in ("style", "script", "sup") or (tag == "span" and "mw-cite-backlink" in classes):
            self._skip = (tag, 1)
        elif tag == "ol":
            if self._ol_depth:
                self._ol_depth += 1
            elif "references" in classes:
                self._ol_depth = 1
        elif tag == "li" and self._ol_depth:
            if self._li is None:
                self._li, self._li_depth = {"id": a.get("id", ""), "text": [], "links": []}, 1
            else:
                self._li_depth += 1
        elif tag == "a" and self._li is not None:
            self._link = {"href": a.get("href", ""), "buf": []}

    def handle_endtag(self, tag):
        if self._skip:
            if tag == self._skip[0]:
                depth = self._skip[1] - 1
                self._skip = (tag, depth) if depth > 0 else None
            return
        if tag == "a" and self._link is not None:
            if self._li is not None:
                self._li["links"].append((self._link["href"], _clean("".join(self._link["buf"]))))
            self._link = None
        elif tag == "li" and self._li is not None:
            self._li_depth -= 1
            if self._li_depth == 0:
                self.refs.append(self._li)
                self._li = None
        elif tag == "ol" and self._ol_depth:
            self._ol_depth -= 1

    def handle_data(self, data):
        if self._skip or self._li is None:
            return
        self._li["text"].append(data)
        if self._link is not None:
            self._link["buf"].append(data)


class _LeadParser(HTMLParser):
    """First substantial paragraph before the first section heading."""

    _SKIP_DIV = ("hatnote", "shortdescription", "thumb", "infobox", "sidebar", "navbox", "metadata")

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[str] = []
        self._p: list[str] | None = None
        self._skip: tuple[str, int] | None = None
        self._done = False

    def handle_starttag(self, tag, attrs):
        if self._done:
            return
        classes = (dict(attrs).get("class") or "").split()
        if self._skip:
            if tag == self._skip[0]:
                self._skip = (tag, self._skip[1] + 1)
            return
        if tag in ("style", "script", "sup", "table") or (
            tag == "div" and any(c.startswith(self._SKIP_DIV) for c in classes)
        ):
            self._skip = (tag, 1)
        elif tag in ("h2", "h3"):
            self._done = True
        elif tag == "p":
            self._p = []

    def handle_endtag(self, tag):
        if self._done:
            return
        if self._skip:
            if tag == self._skip[0]:
                depth = self._skip[1] - 1
                self._skip = (tag, depth) if depth > 0 else None
            return
        if tag == "p" and self._p is not None:
            text, self._p = _clean("".join(self._p)), None
            if len(text) >= 60:
                self.paragraphs.append(text)

    def handle_data(self, data):
        if not self._done and not self._skip and self._p is not None:
            self._p.append(data)


def extract_lead(html: str, limit: int = 600) -> str:
    """Wikipedia lead paragraph, for reviewer orientation only (never a citation)."""
    parser = _LeadParser()
    parser.feed(html)
    parser.close()
    if not parser.paragraphs:
        return ""
    text = parser.paragraphs[0]
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def extract_sources(html: str, *, max_sources: int | None = None) -> tuple[list[dict], int]:
    """Return (sources, reference_count) from an article's reference list only."""
    parser = _RefParser()
    parser.feed(html)
    parser.close()
    sources: dict[str, dict] = {}  # keyed by _url_key
    for ref in parser.refs:
        citation = re.sub(r"\s*\{\{cite.*$", "", _clean("".join(ref["text"])))[:300]
        plain: dict[str, str] = {}
        archives: dict[str, str] = {}
        archive_text: dict[str, str] = {}
        for href, text in ref["links"]:
            url = normalize_url(href)
            if not url:
                continue
            original, archive = split_wayback(url)
            if is_wiki_host(original):
                continue
            if archive:
                archives.setdefault(original, archive)
                archive_text.setdefault(original, text)
            else:
                plain.setdefault(original, text)
        if any(source_hint(u) != "catalog" for u in [*plain, *archives]):
            plain = {u: t for u, t in plain.items() if source_hint(u) != "catalog"}
            archives = {u: a for u, a in archives.items() if source_hint(u) != "catalog"}
        # Pair each snapshot with the plain link to the same page, even if one is http and the
        # other https; prefer the https spelling as the canonical URL.
        texts: dict[str, str] = dict(plain)
        plain_by_key = {_url_key(u): u for u in plain}
        snapshot_for: dict[str, str] = {}
        for original, snap in archives.items():
            key = _url_key(original)
            target = plain_by_key.get(key)
            if target is None:
                target = original
                texts.setdefault(target, "")
                plain_by_key[key] = target
            elif target.startswith("http://") and original.startswith("https://"):
                texts[original] = texts.pop(target)
                plain_by_key[key], target = original, original
            snapshot_for.setdefault(target, snap)
            if _generic(texts[target]):
                alt = archive_text.get(original, "")
                texts[target] = "" if _generic(alt) else alt
        for url, text in texts.items():
            text = "" if _generic(text) else text
            key = _url_key(url)
            entry = sources.get(key)
            if entry is None:
                entry = sources[key] = {
                    "url": url,
                    "archive_url": None,
                    "link_text": text,
                    "citation_text": citation,
                    "source_hint": source_hint(url),
                    "ref_ids": [],
                }
            if entry["archive_url"] is None:
                entry["archive_url"] = snapshot_for.get(url)
            if not entry["link_text"] and text:
                entry["link_text"] = text
            if ref["id"] and ref["id"] not in entry["ref_ids"]:
                entry["ref_ids"].append(ref["id"])
    ordered = sorted(sources.values(), key=lambda s: _RANK[s["source_hint"]])
    return (ordered[:max_sources] if max_sources else ordered), len(parser.refs)


def select_candidates(candidates: list[dict], topics: list[str] | None = None,
                      limit: int | None = None) -> list[dict]:
    """Optionally keep only topics containing any of `topics` (case-insensitive), then cap at `limit`."""
    if topics:
        needles = [t.casefold() for t in topics]
        candidates = [c for c in candidates if any(n in c["topic"].casefold() for n in needles)]
    return candidates[:limit] if limit else candidates


def mark_duplicate_articles(records: list[dict]) -> None:
    """Flag topics whose links redirect to an article already harvested (same resolved title)."""
    first: dict[str, str] = {}
    for rec in records:
        resolved = (rec.get("sources_meta") or {}).get("resolved_title")
        if not resolved:
            continue
        if resolved in first and first[resolved] != rec["topic"]:
            rec["duplicate_of"] = first[resolved]
        else:
            first.setdefault(resolved, rec["topic"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", type=Path, default=Path("pipeline/out/candidate_topics.jsonl"))
    p.add_argument("--out", type=Path, default=Path("pipeline/out/candidates_with_sources.jsonl"))
    p.add_argument("--html-dir", type=Path, help="Directory of saved <stem>.html articles (see wiki.safe_stem)")
    p.add_argument("--offline", action="store_true", help="Never use the network")
    p.add_argument("--cache-dir", type=Path, default=Path("pipeline/cache"))
    p.add_argument("--user-agent", default=wiki.DEFAULT_USER_AGENT)
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--topic", action="append",
                   help="Only harvest topics containing this text (repeatable, case-insensitive)")
    p.add_argument("--limit", type=int, help="Process only the first N (matching) candidates")
    p.add_argument("--max-sources", type=int, default=25)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    make_console_safe()
    try:
        candidates = select_candidates(read_jsonl(args.candidates), args.topic, args.limit)
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not candidates:
        print("Error: no candidate topics matched.", file=sys.stderr)
        return 1
    results, with_sources, errors = [], 0, 0
    for cand in candidates:
        out = dict(cand)
        title = cand["topic"]
        try:
            html = revid = retrieved = resolved = None
            if args.html_dir and (args.html_dir / f"{wiki.safe_stem(title)}.html").exists():
                html = (args.html_dir / f"{wiki.safe_stem(title)}.html").read_text(encoding="utf-8")
            elif args.offline:
                raise wiki.FetchError("offline and no saved HTML for this topic")
            else:
                rec = wiki.fetch_page(title, cache_dir=args.cache_dir, user_agent=args.user_agent,
                                      delay=args.delay, refresh=args.refresh)
                html, revid, retrieved = rec["html"], rec["revid"], rec["retrieved_on"]
                resolved = rec["title"]
            sources, ref_count = extract_sources(html, max_sources=args.max_sources)
            out["candidate_sources"] = sources
            out["article_lead"] = extract_lead(html)
            out["lead_signal"] = lead_signal(out["article_lead"])
            out["article_lead_origin"] = "wikipedia_article_lead (orientation only, not a citation)"
            out["sources_meta"] = {"article_revid": revid, "retrieved_on": retrieved, "resolved_title": resolved,
                                   "reference_count": ref_count, "sources_kept": len(sources)}
            with_sources += bool(sources)
        except wiki.FetchError as exc:
            out["candidate_sources"] = []
            out["sources_error"] = str(exc)
            errors += 1
        results.append(out)
    mark_duplicate_articles(results)
    try:
        write_jsonl(results, args.out, overwrite=args.overwrite)
    except OutputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"{len(results)} topics: {with_sources} with sources, {errors} errors -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
