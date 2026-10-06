import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline import discover_topics as dt
from pipeline import harvest_references as hr
from pipeline import io_utils, wiki

HERE = Path(__file__).resolve().parents[1]
LIST_HTML = (HERE / "fixtures/list_page.html").read_text(encoding="utf-8")
ARTICLE_HTML = (HERE / "fixtures/articles/Moon_landing_conspiracy_theories.html").read_text(encoding="utf-8")


class DiscoverTests(unittest.TestCase):
    def setUp(self):
        self.cands = {c["topic"]: c for c in dt.parse_list_page(LIST_HTML, retrieved_on="2026-10-02")}

    def test_expected_topics_only(self):
        self.assertEqual(set(self.cands), {
            "Moon landing conspiracy theories", "Roswell incident", "Assassination of Abraham Lincoln",
            "Conspiracy theories about the death of Diana, Princess of Wales",
            "Death of Wolfgang Amadeus Mozart", "Murder of John Lennon", "Death of Jimi Hendrix",
            "Murder of Tupac Shakur", "Death of Michael Jackson", "Suicide of Kurt Cobain",
        })

    def test_weak_title_without_conspiracy_context_is_dropped(self):
        self.assertNotIn("Death of Jesus Christ", self.cands)

    def test_hatnote_with_no_claim_is_dropped(self):
        self.assertNotIn("Pseudohistory", self.cands)

    def test_enumeration_sentence_gives_empty_claim_not_a_shared_blob(self):
        for t in ("Murder of John Lennon", "Suicide of Kurt Cobain"):
            self.assertEqual(self.cands[t]["claim"], "")
            self.assertEqual(self.cands[t]["claim_origin"], "enumeration_sentence")

    def test_initials_do_not_split_sentences(self):
        parts = dt.split_sentences("Figures such as George S. Patton died. Another sentence follows.")
        self.assertEqual(len(parts), 2)
        self.assertTrue(parts[0].endswith("died."))

    def test_all_drafts_point_at_articles(self):
        for c in self.cands.values():
            self.assertEqual(c["status"], "needs_human_verification")
            self.assertTrue(c["source_article_url"].startswith("https://en.wikipedia.org/wiki/"))
            self.assertIn(c["claim_origin"], {"wikipedia_list_context", "enumeration_sentence", "none"})

    def test_hatnote_claim_comes_from_following_paragraph(self):
        c = self.cands["Moon landing conspiracy theories"]
        self.assertEqual(c["link_origin"], "hatnote")
        self.assertTrue(c["claim"].startswith("Some people say the Apollo landings were staged"))
        self.assertNotIn("[2]", c["claim"])
        self.assertEqual(c["mentions"], 2)

    def test_inline_claim_is_the_sentence_with_the_link(self):
        c = self.cands["Roswell incident"]
        self.assertEqual(c["claim"], "Others point to the Roswell incident as evidence of a cover-up.")

    def test_section_path_and_title_decoding(self):
        self.assertEqual(self.cands["Roswell incident"]["list_section"], "Space")
        diana = self.cands["Conspiracy theories about the death of Diana, Princess of Wales"]
        self.assertEqual(diana["list_section"], "Deaths and disappearances > Assassinations")
        self.assertIn("Diana,_Princess_of_Wales", diana["source_article_url"])

    def test_wiki_title_rejects_namespaces_and_externals(self):
        self.assertIsNone(dt.wiki_title("/wiki/File:Moon.jpg"))
        self.assertIsNone(dt.wiki_title("https://example.org/x"))
        self.assertEqual(dt.wiki_title("/wiki/Dag_Hammarskj%C3%B6ld#Death"), "Dag Hammarskjöld")


class HarvestTests(unittest.TestCase):
    def setUp(self):
        self.sources, self.count = hr.extract_sources(ARTICLE_HTML)
        self.by_url = {s["url"]: s for s in self.sources}

    def test_only_reference_list_and_no_wikipedia(self):
        self.assertEqual(self.count, 6)
        self.assertEqual(set(self.by_url), {
            "https://www.example.gov/report.pdf", "https://doi.org/10.1000/xyz",
            "https://www.snopes.com/fact-check/x/", "https://news.example.com/a",
        })

    def test_wayback_paired_with_original(self):
        gov = self.by_url["https://www.example.gov/report.pdf"]
        self.assertTrue(gov["archive_url"].startswith("https://web.archive.org/web/2020"))
        self.assertEqual(gov["ref_ids"], ["cite_note-a-1", "cite_note-f-6"])

    def test_wayback_only_reference_recovers_original(self):
        news = self.by_url["https://news.example.com/a"]
        self.assertIsNotNone(news["archive_url"])

    def test_hints_and_ordering(self):
        self.assertEqual([s["source_hint"] for s in self.sources],
                         ["government", "academic", "fact_check", "other"])

    def test_split_wayback_repairs_collapsed_scheme(self):
        orig, arch = hr.split_wayback("https://web.archive.org/web/2019/http:/old.example.com/p")
        self.assertEqual(orig, "http://old.example.com/p")
        self.assertIsNotNone(arch)

    def test_article_lead_skips_hatnote_and_citations(self):
        lead = hr.extract_lead(ARTICLE_HTML)
        self.assertTrue(lead.startswith("This invented lead paragraph"))
        self.assertNotIn("[1]", lead)

    def test_archive_title_catalog_links_and_cs1_noise(self):
        html = ('<ol class="references"><li id="cite_note-1">Story. '
                '<a class="external text" href="https://web.archive.org/web/2020/https://news.example.com/x">"Real Title"</a> '
                '<a class="external text" href="https://news.example.com/x">the original</a> '
                '{{cite web}}: CS1 maint: url-status '
                '<a class="external text" href="https://search.worldcat.org/issn/1234">1234</a></li>'
                '<li id="cite_note-2"><a class="external text" href="https://search.worldcat.org/issn/9999">9999</a></li></ol>')
        sources, _ = hr.extract_sources(html)
        self.assertEqual([s["url"] for s in sources],
                         ["https://news.example.com/x", "https://search.worldcat.org/issn/9999"])
        self.assertEqual(sources[0]["link_text"], '"Real Title"')
        self.assertIsNotNone(sources[0]["archive_url"])
        self.assertNotIn("{{cite", sources[0]["citation_text"])
        self.assertEqual(sources[1]["source_hint"], "catalog")

    def test_http_and_https_spellings_merge_into_one_source(self):
        html = ('<ol class="references"><li id="cite_note-1">'
                '<a class="external text" href="http://www.aim.org/a/">the original</a> '
                '<a class="external text" href="https://web.archive.org/web/2020/https://www.aim.org/a">"Title"</a></li>'
                '<li id="cite_note-2"><a class="external text" href="https://aim.org/a">again</a></li></ol>')
        sources, _ = hr.extract_sources(html)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["url"], "https://www.aim.org/a")
        self.assertEqual(sources[0]["link_text"], '"Title"')
        self.assertIsNotNone(sources[0]["archive_url"])
        self.assertEqual(sources[0]["ref_ids"], ["cite_note-1", "cite_note-2"])

    def test_pubmed_is_academic_not_government(self):
        self.assertEqual(hr.source_hint("https://pubmed.ncbi.nlm.nih.gov/123"), "academic")
        self.assertEqual(hr.source_hint("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1"), "academic")
        self.assertEqual(hr.source_hint("https://www.archives.gov/research/jfk"), "government")

    def test_mark_duplicate_articles(self):
        recs = [{"topic": "A conspiracy theories", "sources_meta": {"resolved_title": "X"}},
                {"topic": "X", "sources_meta": {"resolved_title": "X"}},
                {"topic": "Y", "sources_meta": {"resolved_title": None}}]
        hr.mark_duplicate_articles(recs)
        self.assertNotIn("duplicate_of", recs[0])
        self.assertEqual(recs[1]["duplicate_of"], "A conspiracy theories")
        self.assertNotIn("duplicate_of", recs[2])

    def test_percent_encoding_does_not_split_a_page(self):
        self.assertEqual(hr._url_key("http://www.stanford.edu/a/I've_b.pdf"),
                         hr._url_key("https://stanford.edu/a/I%27ve_b.pdf"))

    def test_library_catalog_records_are_catalog_not_government(self):
        self.assertEqual(hr.source_hint("https://lccn.loc.gov/38005920"), "catalog")
        self.assertEqual(hr.source_hint("https://www.loc.gov/item/11009937/"), "government")

    def test_select_candidates_by_topic_then_limit(self):
        cands = [{"topic": "Chemtrail conspiracy theory"}, {"topic": "Flat Earth theory"},
                 {"topic": "Malaysia Airlines Flight 370 disappearance theories"}]
        self.assertEqual([c["topic"] for c in hr.select_candidates(cands, ["chemtrail", "mh"])],
                         ["Chemtrail conspiracy theory"])
        self.assertEqual(len(hr.select_candidates(cands, ["theor"], 2)), 2)
        self.assertEqual(hr.select_candidates(cands, ["nothing matches"]), [])
        self.assertEqual(len(hr.select_candidates(cands)), 3)

    def test_lead_signal(self):
        self.assertTrue(hr.lead_signal("A theory claims the landings were staged."))
        self.assertFalse(hr.lead_signal("The Tanzimat was a period of reform in the Ottoman Empire."))

    def test_max_sources(self):
        self.assertEqual(len(hr.extract_sources(ARTICLE_HTML, max_sources=2)[0]), 2)


class InfraTests(unittest.TestCase):
    def test_fetch_caches(self):
        calls = []
        def fake(url, ua):
            calls.append(url)
            return json.dumps({"parse": {"title": "T", "revid": 7, "text": "<p>x</p>"}})
        with tempfile.TemporaryDirectory() as d:
            a = wiki.fetch_page("T", cache_dir=d, delay=0, fetcher=fake)
            b = wiki.fetch_page("T", cache_dir=d, delay=0, fetcher=fake)
        self.assertEqual(len(calls), 1)
        self.assertFalse(a["from_cache"])
        self.assertTrue(b["from_cache"])
        self.assertEqual(b["revid"], 7)

    def test_safe_stem_never_merges_different_titles(self):
        self.assertNotEqual(wiki.safe_stem("Éire"), wiki.safe_stem("Ēire"))
        self.assertTrue(wiki.safe_stem("Éire").startswith("ire-"))
        self.assertNotEqual(wiki.safe_stem("9/11 conspiracy theories"), wiki.safe_stem("9_11 conspiracy theories"))
        self.assertNotEqual(wiki.safe_stem("A" * 100), wiki.safe_stem("A" * 100 + "B"))
        self.assertLessEqual(len(wiki.safe_stem("A" * 300)), 70)
        # plain titles stay readable, so --html-dir fixtures keep their simple names
        self.assertEqual(wiki.safe_stem("Moon landing conspiracy theories"), "Moon_landing_conspiracy_theories")
        self.assertEqual(wiki.safe_stem("List of conspiracy theories"), "List_of_conspiracy_theories")

    def test_cache_never_returns_another_titles_page(self):
        payload = lambda title: json.dumps({"parse": {"title": title, "revid": 1, "text": f"<p>{title}</p>"}})
        calls = []

        def fake(url, ua):
            calls.append(url)
            return payload("Ēire" if "%C4%92" in url else "Éire")
        with tempfile.TemporaryDirectory() as d, mock.patch.object(wiki, "safe_stem", return_value="same"):
            first = wiki.fetch_page("Éire", cache_dir=d, delay=0, fetcher=fake)
            second = wiki.fetch_page("Ēire", cache_dir=d, delay=0, fetcher=fake)
        self.assertEqual(len(calls), 2)
        self.assertEqual((first["title"], second["title"]), ("Éire", "Ēire"))
        self.assertFalse(second["from_cache"])

    def test_api_error_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(wiki.FetchError):
                wiki.fetch_page("X", cache_dir=d, delay=0,
                                fetcher=lambda u, ua: json.dumps({"error": {"info": "missing"}}))

    def test_never_writes_into_questions(self):
        with self.assertRaises(io_utils.OutputError):
            io_utils.write_jsonl([{"a": 1}], io_utils.REPO_ROOT / "questions" / "pending" / "x.jsonl")

    def test_write_jsonl_utf8_lf_and_no_silent_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "o.jsonl"
            io_utils.write_jsonl([{"t": "Hammarskjöld"}], p, repo_root=d)
            self.assertEqual(p.read_bytes(), '{"t": "Hammarskjöld"}\n'.encode("utf-8"))
            with self.assertRaises(io_utils.OutputError):
                io_utils.write_jsonl([{"t": 1}], p, repo_root=d)

    def test_cli_end_to_end_offline(self):
        with tempfile.TemporaryDirectory() as d:
            lst = Path(d) / "list.html"; lst.write_text(LIST_HTML, encoding="utf-8")
            out1, out2 = Path(d) / "c.jsonl", Path(d) / "s.jsonl"
            self.assertEqual(dt.main(["--list-html", str(lst), "--out", str(out1)]), 0)
            arts = Path(d) / "arts"; arts.mkdir()
            (arts / "Moon_landing_conspiracy_theories.html").write_text(ARTICLE_HTML, encoding="utf-8")
            self.assertEqual(hr.main(["--candidates", str(out1), "--out", str(out2),
                                      "--html-dir", str(arts), "--offline"]), 0)
            rows = io_utils.read_jsonl(out2)
            moon = next(r for r in rows if r["topic"].startswith("Moon"))
            self.assertEqual(len(moon["candidate_sources"]), 4)
            self.assertTrue(all(r["status"] == "needs_human_verification" for r in rows))
            self.assertTrue(all("sources_error" in r for r in rows if r is not moon))


class EncodingTests(unittest.TestCase):
    """Windows defaults to cp1252/cp949; see Discussions #84 / PR #88."""

    def test_every_text_io_call_names_an_encoding(self):
        for path in HERE.glob("*.py"):
            src = path.read_text(encoding="utf-8")
            for m in re.finditer(r"\.(?:read_text|write_text|open)\(.*", src):  # rest of the line
                self.assertIn("encoding=", m.group(0), f"{path.name}: {m.group(0)}")

    def test_cli_survives_ascii_console_and_strict_encoding_warnings(self):
        html = ('<h2>Z\u00fcrich affairs</h2><div class="hatnote">Main article: '
                '<a href="/wiki/Zurich_incident">x</a></div><p>A theory claims it was staged.</p>')
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d) / "l.html", Path(d) / "o.jsonl"
            src.write_text(html, encoding="utf-8")
            r = subprocess.run(
                [sys.executable, "-X", "warn_default_encoding", "-W", "error::EncodingWarning",
                 "-m", "pipeline.discover_topics", "--list-html", str(src), "--out", str(out)],
                cwd=HERE.parent, env=dict(os.environ, PYTHONIOENCODING="ascii", PYTHONDONTWRITEBYTECODE="1"), capture_output=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr.decode("ascii", "replace"))
            self.assertIn("Z\\xfcrich", r.stdout.decode("ascii"))
            self.assertEqual(io_utils.read_jsonl(out)[0]["list_section"], "Z\u00fcrich affairs")


if __name__ == "__main__":
    unittest.main()