"""SEC EDGAR real-time filing feed -- the primary edge.

Why EDGAR first and not a news API: EDGAR is the *source*. A newswire summary of
an 8-K is downstream of the 8-K. Acceptance-to-dissemination on the public
current-filings index is seconds, and critically, the long tail of small-cap
filings is not on anybody's algo radar. A 424B5 dropping intraday means a dilutive
offering is pricing; a SC 13D means an activist just showed a hand. Those are
public, structured, timestamped, and routinely under-watched.

Rate limits: SEC allows 10 req/s per IP and *requires* a declaring User-Agent
with contact info. We stay well under via the shared token bucket. Getting IP
blocked the morning of a big print is the single dumbest way to lose this edge.

For sub-second dissemination SEC sells the PDS (Public Dissemination Service)
feed; that is the paid tier the pros use. This module targets the free path,
which is fast enough for the minutes-scale plays this system is built around.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Iterable
from xml.etree import ElementTree as ET

import httpx

from ..models import RawDoc
from .base import PollingFeed

log = logging.getLogger("signalsniper.feeds.edgar")

ATOM = "{http://www.w3.org/2005/Atom}"

CURRENT_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type={form}&company=&dateb=&owner=include&count={count}&output=atom"
)

#: Accession numbers look like 0001234567-26-000123.
_ACCESSION_RE = re.compile(r"(\d{10}-\d{2}-\d{6})")
#: "8-K - ACME CORP (0001234567) (Filer)". The separator is space-hyphen-space,
#: which matters because form types contain hyphens themselves ("8-K", "25-NSE",
#: "NT 10-K") -- splitting on a bare hyphen truncates the form and corrupts the
#: company name.
_TITLE_RE = re.compile(r"^\s*(?P<form>.+?)\s+-\s+(?P<name>.+?)\s*\((?P<cik>\d{4,10})\)")
#: 8-K item codes, e.g. "Item 2.02".
#:
#: IMPORTANT: these do NOT appear in the getcurrent Atom feed. That feed's
#: <summary> is only "Filed: <date> AccNo: <accession> Size: <n> KB". An earlier
#: version of this parser scanned the summary for item codes and always found
#: none, which silently made every 8-K fall back to the generic 0.35 form prior
#: -- below the 0.40 materiality floor, so the engine would have rejected
#: essentially every 8-K and the system would have run nearly silent.
#:
#: The unit tests did not catch it because the fixture was hand-written with
#: item codes in the summary: it tested the assumption, not the feed.
#:
#: Item codes live on the per-filing index page the entry already links to, so
#: `EdgarEnricher` fetches that. It costs one extra request per filing of
#: interest, which is why enrichment is a separate, rate-limited stage rather
#: than part of the poll.
_ITEM_RE = re.compile(r"\bItem\s+(\d{1,2}\.\d{2})\b")


def _text(node: ET.Element | None) -> str:
    return (node.text or "").strip() if node is not None else ""


def _parse_dt(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


class EdgarCurrentFeed(PollingFeed):
    """Polls EDGAR's current-filings index for one or more form types.

    Pass `form=""` to watch every form (heaviest, most complete). Pass a specific
    form to get a tighter, faster stream -- one instance per form type is the
    right call, since the per-form endpoints are small and 304 cheaply.
    """

    def __init__(self, *args, form: str = "8-K", count: int = 40, **kwargs) -> None:
        self.form = form
        url = CURRENT_URL.format(form=form, count=count)
        super().__init__(url, *args, **kwargs)
        self.name = f"edgar[{form or 'ALL'}]"

    def parse(self, body: bytes, headers: httpx.Headers) -> Iterable[RawDoc]:
        root = ET.fromstring(body)
        for entry in root.iter(f"{ATOM}entry"):
            doc = self._entry_to_doc(entry)
            if doc is not None:
                yield doc

    def _entry_to_doc(self, entry: ET.Element) -> RawDoc | None:
        title = _text(entry.find(f"{ATOM}title"))
        raw_id = _text(entry.find(f"{ATOM}id"))
        summary = _text(entry.find(f"{ATOM}summary"))
        updated = _text(entry.find(f"{ATOM}updated"))

        link_el = entry.find(f"{ATOM}link")
        url = link_el.get("href", "") if link_el is not None else ""

        # Accession number is the only globally stable dedupe key. It may live in
        # the <id>, the link, or (rarely) only in the summary.
        m = _ACCESSION_RE.search(raw_id) or _ACCESSION_RE.search(url) or _ACCESSION_RE.search(summary)
        accession = m.group(1) if m else (raw_id or url)
        if not accession:
            return None

        form = self.form
        company = ""
        cik = ""
        tm = _TITLE_RE.match(title)
        if tm:
            form = tm.group("form").strip() or form
            company = tm.group("name").strip()
            cik = tm.group("cik").lstrip("0")

        # Category element carries the authoritative form type when present.
        for cat in entry.iter(f"{ATOM}category"):
            if cat.get("label", "").lower() == "form type" and cat.get("term"):
                form = cat.get("term", form)

        # The getcurrent summary does not carry item codes; scan anyway in case a
        # future feed shape adds them, but expect () and let EdgarEnricher fill
        # them in from the filing index page.
        items = tuple(sorted(set(_ITEM_RE.findall(summary + " " + title))))

        return RawDoc(
            source="edgar",
            doc_id=accession,
            title=title or f"{form} {company}",
            url=url,
            published=_parse_dt(updated),
            body=summary,
            meta={
                "form": form,
                "company": company,
                "cik": cik,
                "items": items,
                "accession": accession,
            },
        )


#: "Items" block on a filing index page:
#:   <div class="infoHead">Items</div>
#:   <div class="info">Item 2.02: Results of Operations...<br>Item 9.01: ...</div>
_INDEX_ITEMS_RE = re.compile(
    r"Items?\s*</div>\s*<div[^>]*>(.*?)</div>", re.IGNORECASE | re.DOTALL
)


class EdgarEnricher:
    """Fetches item codes for a filing from its index page.

    Kept out of the polling loop deliberately. The current-filings feed is one
    cheap conditional GET; enrichment is one request *per filing*, and at 16:05
    with hundreds of filings landing at once, doing that inline would blow the
    SEC rate budget and stall the feed. Callers should enrich selectively --
    filings that resolved to a ticker, or forms where item codes change the
    classification.

    NOT VERIFIED AGAINST LIVE SEC. The egress policy in the build environment
    blocks sec.gov, so the index-page shape here comes from documented structure
    rather than a live fetch. Confirm with `doctor` from a host that can reach
    sec.gov before relying on it.
    """

    def __init__(self, client: httpx.AsyncClient, bucket, cost: float = 1.0) -> None:
        self.client = client
        self.bucket = bucket
        self.cost = cost
        self.fetched = 0
        self.failed = 0

    @staticmethod
    def parse_items(html: str) -> tuple[str, ...]:
        block = _INDEX_ITEMS_RE.search(html)
        text = block.group(1) if block else html
        return tuple(sorted(set(_ITEM_RE.findall(text))))

    async def enrich(self, doc: RawDoc) -> RawDoc:
        """Populate doc.meta['items']. Returns the same doc, mutated.

        Failure is non-fatal: an un-enriched 8-K classifies on its form prior
        alone, which is weak but not wrong. Blocking the pipeline on a slow
        index fetch would be worse.
        """
        if doc.meta.get("items") or not doc.url:
            return doc
        await self.bucket.take(self.cost)
        try:
            resp = await self.client.get(doc.url)
            if resp.status_code >= 400:
                self.failed += 1
                return doc
            items = self.parse_items(resp.text)
            if items:
                doc.meta["items"] = items
            self.fetched += 1
        except Exception as exc:
            self.failed += 1
            log.debug("enrich failed for %s: %r", doc.doc_id, exc)
        return doc


class TickerResolver:
    """CIK -> ticker, from SEC's own mapping file.

    Loaded once at startup. ~10k issuers, a few hundred KB. Doing this lookup at
    event time from a dict is nanoseconds; doing it over the network would cost
    more than the entire rest of the pipeline.
    """

    URL = "https://www.sec.gov/files/company_tickers.json"

    def __init__(self) -> None:
        self.by_cik: dict[str, str] = {}
        self.by_name: dict[str, str] = {}

    def load_mapping(self, payload: dict) -> int:
        """Accepts SEC's {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}} shape."""
        n = 0
        for row in payload.values():
            try:
                cik = str(int(row["cik_str"]))
                ticker = str(row["ticker"]).upper().strip()
                title = str(row.get("title", "")).upper().strip()
            except (KeyError, TypeError, ValueError):
                continue
            if not ticker:
                continue
            # First entry wins: SEC lists the primary class first.
            self.by_cik.setdefault(cik, ticker)
            if title:
                self.by_name.setdefault(title, ticker)
            n += 1
        return n

    async def load(self, client: httpx.AsyncClient) -> int:
        resp = await client.get(self.URL)
        resp.raise_for_status()
        return self.load_mapping(resp.json())

    def resolve(self, cik: str = "", company: str = "") -> str:
        if cik:
            hit = self.by_cik.get(str(cik).lstrip("0"))
            if hit:
                return hit
        if company:
            key = company.upper().strip()
            hit = self.by_name.get(key)
            if hit:
                return hit
            # Issuer names drift ("Apple Inc." vs "APPLE INC"); try a prefix match
            # only for reasonably distinctive names to avoid false joins.
            if len(key) >= 8:
                for name, tick in self.by_name.items():
                    if name.startswith(key) or key.startswith(name):
                        return tick
        return ""
