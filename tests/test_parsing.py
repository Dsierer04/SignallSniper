"""Feed parsing: the EDGAR atom shape and RSS/Atom newswire shapes."""

from __future__ import annotations

import httpx
import pytest

from signalsniper.bus import EventBus
from signalsniper.feeds.base import TokenBucket
from signalsniper.feeds.edgar import EdgarCurrentFeed, TickerResolver
from signalsniper.feeds.newswire import RssFeed

EDGAR_ATOM = b"""<?xml version="1.0" encoding="ISO-8859-1"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Latest Filings</title>
  <entry>
    <title>8-K - Apple Inc. (0000320193) (Filer)</title>
    <link rel="alternate" type="text/html"
      href="https://www.sec.gov/Archives/edgar/data/320193/000032019326000077-index.htm"/>
    <summary type="html">Item 2.02 Results of Operations and Financial Condition.
      Item 9.01 Financial Statements and Exhibits.</summary>
    <updated>2026-07-30T16:05:12-04:00</updated>
    <category scheme="https://www.sec.gov/" label="form type" term="8-K"/>
    <id>urn:tag:sec.gov,2008:accession-number=0000320193-26-000077</id>
  </entry>
  <entry>
    <title>424B5 - TINYBIO THERAPEUTICS INC (0001800001) (Filer)</title>
    <link rel="alternate" type="text/html"
      href="https://www.sec.gov/Archives/edgar/data/1800001/000180000126000012-index.htm"/>
    <summary type="html">424B5 filing.</summary>
    <updated>2026-07-30T16:31:44-04:00</updated>
    <category scheme="https://www.sec.gov/" label="form type" term="424B5"/>
    <id>urn:tag:sec.gov,2008:accession-number=0001800001-26-000012</id>
  </entry>
</feed>
"""

RSS_20 = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>Wire</title>
  <item>
    <title>Acme Corp (NASDAQ: ACME) Raises Full-Year Guidance</title>
    <link>https://example.com/acme-1</link>
    <guid>acme-1</guid>
    <pubDate>Thu, 30 Jul 2026 20:05:00 GMT</pubDate>
    <description>Acme now expects revenue above consensus estimates.</description>
  </item>
</channel></rss>
"""

ATOM_WIRE = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Beta Inc Announces Pricing of Public Offering</title>
    <link href="https://example.com/beta-1"/>
    <id>beta-1</id>
    <published>2026-07-30T20:10:00Z</published>
    <summary>Beta Inc announced the pricing of an underwritten public offering of
      12,000,000 shares for gross proceeds of $60 million.</summary>
  </entry>
</feed>
"""


def _feed(cls, url="https://example.invalid/f", **kw):
    return cls(url, EventBus(), None, TokenBucket(10.0), **kw)


class TestEdgarParsing:
    def test_extracts_accession_form_items(self):
        feed = EdgarCurrentFeed(EventBus(), None, TokenBucket(10.0), form="8-K")
        docs = list(feed.parse(EDGAR_ATOM, httpx.Headers()))
        assert len(docs) == 2

        aapl = docs[0]
        assert aapl.doc_id == "0000320193-26-000077"
        assert aapl.meta["form"] == "8-K"
        assert aapl.meta["cik"] == "320193"
        assert aapl.meta["company"] == "Apple Inc."
        assert aapl.meta["items"] == ("2.02", "9.01")
        assert aapl.published is not None

    def test_form_from_category_overrides_default(self):
        feed = EdgarCurrentFeed(EventBus(), None, TokenBucket(10.0), form="8-K")
        docs = list(feed.parse(EDGAR_ATOM, httpx.Headers()))
        assert docs[1].meta["form"] == "424B5"
        assert docs[1].doc_id == "0001800001-26-000012"

    def test_dedupes_by_accession(self):
        feed = EdgarCurrentFeed(EventBus(), None, TokenBucket(10.0), form="8-K")
        first = [d for d in feed.parse(EDGAR_ATOM, httpx.Headers()) if feed._remember(d.doc_id)]
        second = [d for d in feed.parse(EDGAR_ATOM, httpx.Headers()) if feed._remember(d.doc_id)]
        assert len(first) == 2
        assert second == []

    def test_seen_set_is_bounded(self):
        feed = EdgarCurrentFeed(EventBus(), None, TokenBucket(10.0), form="8-K", seen_max=100)
        for i in range(500):
            feed._remember(f"acc-{i}")
        assert len(feed._seen) <= 100


class TestNewswireParsing:
    def test_rss_20(self):
        feed = _feed(RssFeed, label="wire")
        docs = list(feed.parse(RSS_20, httpx.Headers()))
        assert len(docs) == 1
        assert docs[0].doc_id == "acme-1"
        assert "Raises Full-Year Guidance" in docs[0].title
        assert docs[0].published is not None

    def test_atom(self):
        feed = _feed(RssFeed, label="wire")
        docs = list(feed.parse(ATOM_WIRE, httpx.Headers()))
        assert len(docs) == 1
        assert docs[0].doc_id == "beta-1"
        assert "$60 million" in docs[0].body

    def test_ir_feed_preattributes_tickers(self):
        feed = _feed(RssFeed, label="acme-ir", tickers=("ACME",))
        docs = list(feed.parse(RSS_20, httpx.Headers()))
        assert docs[0].meta["tickers"] == ("ACME",)

    def test_missing_guid_falls_back_to_stable_hash(self):
        body = b"""<rss version="2.0"><channel><item>
            <title>No guid here</title><description>x</description>
        </item></channel></rss>"""
        feed = _feed(RssFeed)
        a = list(feed.parse(body, httpx.Headers()))[0].doc_id
        b = list(feed.parse(body, httpx.Headers()))[0].doc_id
        assert a == b and a


class TestTickerResolver:
    @pytest.fixture
    def resolver(self):
        r = TickerResolver()
        r.load_mapping({
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            "1": {"cik_str": 1018724, "ticker": "AMZN", "title": "AMAZON COM INC"},
            "2": {"cik_str": 772406, "ticker": "CRUS", "title": "CIRRUS LOGIC INC"},
        })
        return r

    def test_by_cik_ignores_leading_zeros(self, resolver):
        assert resolver.resolve(cik="0000320193") == "AAPL"
        assert resolver.resolve(cik="320193") == "AAPL"

    def test_by_name_case_insensitive(self, resolver):
        assert resolver.resolve(company="Apple Inc.") == "AAPL"
        assert resolver.resolve(company="cirrus logic inc") == "CRUS"

    def test_unknown_returns_empty(self, resolver):
        assert resolver.resolve(cik="999999999") == ""
        assert resolver.resolve(company="Nonexistent Holdings") == ""

    def test_short_name_does_not_prefix_match(self, resolver):
        # Guard against a 3-char query joining to an unrelated issuer.
        assert resolver.resolve(company="APP") == ""
