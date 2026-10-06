"""Polite MediaWiki fetcher with an on-disk cache. Standard library only.

Wikipedia is used ONLY to discover topics and to find the articles' own cited
references. Wikipedia text is never a citation for a benchmark ground truth.
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
from pathlib import Path
from typing import Callable

API_URL = "https://en.wikipedia.org/w/api.php"
LIST_PAGE_TITLE = "List of conspiracy theories"
DEFAULT_USER_AGENT = (
    "ConspiracyBench-pipeline/0.1 (research prototype; "
    "pass --user-agent with your contact details)"
)


class FetchError(RuntimeError):
    """A page could not be fetched or parsed."""


def article_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + urllib.parse.quote(
        title.replace(" ", "_"), safe="_(),'!*-.:"
    )


def safe_stem(title: str) -> str:
    """Windows-safe file stem for cache files and --html-dir fixtures.

    Plain titles keep a readable name. If sanitizing would lose information (non-ASCII letters,
    punctuation, very long titles) a short hash of the full title is appended, so different titles
    never share a file (e.g. "Éire" vs "Ēire").
    """
    plain = title.replace(" ", "_")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", plain).strip("_")
    if stem == plain and len(plain) <= 80:
        return stem
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:8]
    return f"{stem[:60] or 'page'}-{digest}"


def _http_get(url: str, user_agent: str, timeout: int = 30) -> str:
    request = urllib.request.Request(
        url, headers={"User-Agent": user_agent, "Accept": "application/json"}
    )
    last: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
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
    raise FetchError(f"GET {url} failed: {last}")


def fetch_page(
    title: str,
    *,
    cache_dir: Path | str,
    user_agent: str = DEFAULT_USER_AGENT,
    delay: float = 1.0,
    refresh: bool = False,
    fetcher: Callable[[str, str], str] = _http_get,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Return {title, revid, html, retrieved_on, from_cache} for a Wikipedia page."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{safe_stem(title)}.json"
    if path.exists() and not refresh:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("requested_title") == title:  # never serve another title's cached page
            record["from_cache"] = True
            return record
    params = {
        "action": "parse", "page": title, "prop": "text|revid", "redirects": 1,
        "format": "json", "formatversion": 2,
        "disableeditsection": 1, "disablelimitreport": 1,
    }
    if delay:
        sleep(delay)
    payload = json.loads(fetcher(API_URL + "?" + urllib.parse.urlencode(params), user_agent))
    if "error" in payload:
        raise FetchError(f"{title}: {payload['error'].get('info', payload['error'])}")
    parsed = payload.get("parse")
    if not parsed or not isinstance(parsed.get("text"), str):
        raise FetchError(f"{title}: unexpected API response shape")
    record = {
        "requested_title": title,
        "title": parsed.get("title", title),
        "revid": parsed.get("revid"),
        "html": parsed["text"],
        "retrieved_on": date.today().isoformat(),
        "from_cache": False,
    }
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return record