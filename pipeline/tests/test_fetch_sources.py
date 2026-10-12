import contextlib
import io
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline import fetch_sources as fs
from pipeline import io_utils

HTML = (b"<html><head><title>T</title><style>.x{}</style></head><body><nav>Menu</nav>"
        b"<p>The USAF says contrails are condensation.</p><script>var a=1;</script>"
        b"<div>Second   line</div></body></html>")
URL = "https://a.example/x"
SNAP = "https://web.archive.org/web/2020/https://a.example/x"
RAW = "https://web.archive.org/web/2020id_/https://a.example/x"


def ok(body=HTML, ctype="text/html; charset=utf-8"):
    return fs.Response(200, {"content-type": ctype}, body, "")


def make_get(routes):
    calls = []

    def get(url):
        calls.append(url)
        result = routes.get(url, fs.Response(404, {}, b"", url))
        if isinstance(result, Exception):
            raise result
        return result
    get.calls = calls
    return get


def no_lookup(url):
    raise AssertionError("availability lookup should not be called")


class ParsingTests(unittest.TestCase):
    def test_html_to_text_skips_scripts_and_collapses_space(self):
        text = fs.html_to_text(HTML.decode())
        self.assertIn("The USAF says contrails are condensation.", text)
        self.assertIn("Second line", text)
        self.assertNotIn("var a", text)
        self.assertNotIn(".x{}", text)

    def test_raw_snapshot_url(self):
        self.assertEqual(fs.raw_snapshot_url("https://web.archive.org/web/20200101000000/https://x.org/a"),
                         "https://web.archive.org/web/20200101000000id_/https://x.org/a")
        already = "https://web.archive.org/web/20200101000000id_/https://x.org/a"
        self.assertEqual(fs.raw_snapshot_url(already), already)
        self.assertEqual(fs.raw_snapshot_url("https://x.org/a"), "https://x.org/a")

    def test_text_source_url(self):
        self.assertEqual(fs.text_source_url({"via": "wayback", "snapshot_url": SNAP, "final_url": "x"}), RAW)
        self.assertEqual(fs.text_source_url({"via": "original", "final_url": URL}), URL)
        self.assertIsNone(fs.text_source_url({"via": None, "final_url": URL}))

    def test_pypdf_font_warnings_are_silenced(self):
        self.assertEqual(logging.getLogger("pypdf").level, logging.ERROR)

    def test_wayback_lookup_parsing(self):
        good = json.dumps({"archived_snapshots": {"closest": {
            "available": True, "status": "200", "url": SNAP}}}).encode()
        route = lambda body, status=200: (lambda u: fs.Response(status, {}, body, u))
        self.assertEqual(fs.wayback_lookup(URL, route(good)), SNAP)
        self.assertIsNone(fs.wayback_lookup(URL, route(b'{"archived_snapshots": {}}')))
        self.assertIsNone(fs.wayback_lookup(URL, route(good, 503)))
        self.assertIsNone(fs.wayback_lookup(URL, route(b"not json")))

        def boom(u):
            raise fs.FetchFailure("down")
        self.assertIsNone(fs.wayback_lookup(URL, boom))


class FetchSourceTests(unittest.TestCase):
    def test_ok_from_original(self):
        get = make_get({URL: ok()})
        rec = fs.fetch_source(URL, get=get, lookup=no_lookup)
        self.assertEqual((rec["status"], rec["via"]), ("ok", "original"))
        self.assertIn("condensation", rec["text"])
        self.assertEqual(get.calls, [URL])

    def test_403_falls_back_to_the_articles_own_snapshot(self):
        get = make_get({URL: fs.Response(403, {}, b"", URL), RAW: ok()})
        rec = fs.fetch_source(URL, archive_url=SNAP, get=get, lookup=no_lookup)
        self.assertEqual((rec["status"], rec["via"]), ("ok", "wayback"))
        self.assertEqual(rec["snapshot_url"], SNAP)
        self.assertEqual(get.calls[-1], RAW)
        self.assertIn("HTTP 403", rec["note"])

    def test_network_error_uses_availability_lookup(self):
        get = make_get({URL: fs.FetchFailure("timed out"), RAW: ok()})
        rec = fs.fetch_source(URL, get=get, lookup=lambda u: SNAP)
        self.assertEqual((rec["status"], rec["via"]), ("ok", "wayback"))
        self.assertIn("timed out", rec["note"])

    def test_unavailable_when_no_snapshot_exists(self):
        rec = fs.fetch_source(URL, get=make_get({}), lookup=lambda u: None)
        self.assertEqual(rec["status"], "unavailable")
        self.assertIsNone(rec["text"])
        self.assertIn("no snapshot found", rec["note"])

    def test_snapshot_without_text_is_unavailable(self):
        get = make_get({RAW: ok(b"")})
        rec = fs.fetch_source(URL, archive_url=SNAP, get=get, lookup=lambda u: None)
        self.assertEqual(rec["status"], "unavailable")
        self.assertIn("snapshot had no text", rec["note"])

    def test_pdf_needs_pypdf_or_reads_text(self):
        pdf = fs.Response(200, {"content-type": "application/pdf"}, b"%PDF-1.4 ...", URL)
        with mock.patch.object(fs, "pdf_to_text", return_value=None):
            self.assertEqual(fs.fetch_source(URL, get=make_get({URL: pdf}), lookup=no_lookup)["status"],
                             "unsupported_pdf")
        with mock.patch.object(fs, "pdf_to_text", return_value="Contrails facts text"):
            rec = fs.fetch_source(URL, get=make_get({URL: pdf}), lookup=no_lookup)
        self.assertEqual((rec["status"], rec["kind"], rec["text"]), ("ok", "pdf", "Contrails facts text"))

    def test_unsupported_type(self):
        img = fs.Response(200, {"content-type": "image/png"}, b"\x89PNG", URL)
        self.assertEqual(fs.fetch_source(URL, get=make_get({URL: img}), lookup=no_lookup)["status"],
                         "unsupported_type")


class CacheAndCliTests(unittest.TestCase):
    def test_cache_hits_network_once(self):
        get, sleeps = make_get({URL: ok()}), []
        with tempfile.TemporaryDirectory() as d:
            first = fs.fetch_cached(URL, text_dir=Path(d), get=get, lookup=no_lookup, sleep=sleeps.append)
            second = fs.fetch_cached(URL, text_dir=Path(d), get=get, lookup=no_lookup, sleep=sleeps.append)
            self.assertEqual(len(get.calls), 1)
            self.assertEqual(sleeps, [1.0])
            self.assertFalse(first["from_cache"])
            self.assertTrue(second["from_cache"])
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertIn("condensation", Path(second["text_path"]).read_text(encoding="utf-8"))
            self.assertNotIn("text", first)

    def _candidates(self, path):
        src = lambda url, hint: {"url": url, "archive_url": None, "link_text": "t", "source_hint": hint}
        rows = [
            {"topic": "Chemtrail conspiracy theory", "candidate_sources": [
                src("https://gov.example/a", "government"), src("https://lookup.example/c", "catalog"),
                src("https://news.example/b", "other")]},
            {"topic": "Flat Earth theory", "candidate_sources": [src("https://flat.example/z", "other")]},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def test_main_end_to_end_is_scoped_and_skips_catalog(self):
        gov = fs.Response(200, {"content-type": "text/html"}, HTML, "https://gov.example/a")
        get = make_get({"https://gov.example/a": gov, "https://news.example/b": fs.Response(404, {}, b"", "")})
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self._candidates(d / "c.jsonl")
            argv = ["--sources", str(d / "c.jsonl"), "--out", str(d / "m.jsonl"),
                    "--text-dir", str(d / "t"), "--topic", "chemtrail", "--delay", "0"]
            self.assertEqual(fs.main(argv, get=get, lookup=lambda u: None, sleep=lambda s: None), 0)
            rows = io_utils.read_jsonl(d / "m.jsonl")
            self.assertEqual([(r["url"], r["status"]) for r in rows],
                             [("https://gov.example/a", "ok"), ("https://news.example/b", "unavailable")])
            self.assertTrue(Path(rows[0]["text_path"]).exists())
            self.assertEqual((rows[0]["text_url"], rows[1]["text_url"]), ("https://gov.example/a", None))
            self.assertNotIn("https://flat.example/z", get.calls)
            self.assertNotIn("https://lookup.example/c", get.calls)
            self.assertEqual(fs.main(argv, get=get, lookup=lambda u: None, sleep=lambda s: None), 1)  # no overwrite

    def test_main_requires_a_topic_and_reports_no_match(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                fs.main([])
            with tempfile.TemporaryDirectory() as d:
                self._candidates(Path(d) / "c.jsonl")
                self.assertEqual(fs.main(["--sources", str(Path(d) / "c.jsonl"), "--topic", "nothing"]), 1)


if __name__ == "__main__":
    unittest.main()