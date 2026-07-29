"""Generic RSS/Atom newswire poller.

Newswires are *slower* than EDGAR for filings but faster for things that never
get filed: pre-announcements, product news, analyst days, FDA letters, and
company IR pages that post before the 8-K clears. Run them alongside EDGAR, not
instead of it.

Company IR feeds are the underrated entry here. A press release hits the issuer's
own IR page at the same instant it hits the wire, and almost nobody polls the IR
page directly -- they wait for the aggregator to pick it up.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable
from xml.etree import ElementTree as ET

import httpx

from ..models import RawDoc
from .base import PollingFeed

log = logging.getLogger("signalsniper.feeds.newswire")

ATOM = "{http://www.w3.org/2005/Atom}"


def _text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def _parse_date(raw: str) -> datetime | None:
    if not raw:
        return None
    try:  # RFC 822, the RSS convention
        return parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        pass
    try:  # ISO 8601, the Atom convention
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


class RssFeed(PollingFeed):
    """Handles RSS 2.0 and Atom in one parser -- both shapes appear in the wild."""

    def __init__(self, url: str, *args, label: str = "", tickers: tuple[str, ...] = (), **kwargs) -> None:
        super().__init__(url, *args, **kwargs)
        self.name = f"rss[{label or url.split('/')[2]}]"
        #: If set, every doc from this feed is pre-attributed to these tickers.
        #: That is the point of a company IR feed -- zero ambiguity, zero lookup.
        self.tickers = tickers

    def parse(self, body: bytes, headers: httpx.Headers) -> Iterable[RawDoc]:
        root = ET.fromstring(body)
        items = list(root.iter("item")) or list(root.iter(f"{ATOM}entry"))
        for item in items:
            doc = self._item_to_doc(item)
            if doc is not None:
                yield doc

    def _item_to_doc(self, item: ET.Element) -> RawDoc | None:
        is_atom = item.tag.startswith(ATOM)

        title = _text(item.find(f"{ATOM}title" if is_atom else "title"))
        if is_atom:
            link_el = item.find(f"{ATOM}link")
            url = link_el.get("href", "") if link_el is not None else ""
            guid = _text(item.find(f"{ATOM}id"))
            published = _parse_date(
                _text(item.find(f"{ATOM}published")) or _text(item.find(f"{ATOM}updated"))
            )
            body = _text(item.find(f"{ATOM}summary")) or _text(item.find(f"{ATOM}content"))
        else:
            url = _text(item.find("link"))
            guid = _text(item.find("guid"))
            published = _parse_date(_text(item.find("pubDate")))
            body = _text(item.find("description"))

        if not (title or url):
            return None

        # Prefer a publisher-supplied id; fall back to a content hash so an item
        # with a rotating URL does not re-fire every poll.
        doc_id = guid or url or hashlib.blake2b(title.encode(), digest_size=16).hexdigest()

        return RawDoc(
            source="newswire",
            doc_id=doc_id,
            title=title,
            url=url,
            published=published,
            body=body,
            meta={"feed": self.name, "tickers": self.tickers},
        )


#: Wire feeds that publish company press releases. All free, all pollable.
#: Company IR feeds beat these -- add your watchlist's own IR RSS URLs to
#: config and poll them at a tighter interval.
PUBLIC_WIRES: dict[str, str] = {
    "globenewswire": "https://www.globenewswire.com/RssFeed/orgclass/1/feedTitle/GlobeNewswire%20-%20News%20about%20Public%20Companies",
    "accesswire": "https://www.accesswire.com/users/rss.aspx",
    "prnewswire_financial": "https://www.prnewswire.com/rss/financial-services-latest-news/financial-services-latest-news-list.rss",
    "businesswire_tech": "https://feed.businesswire.com/rss/home/?rss=G1QFDERJXkJeEFpRVQ==",
    "fda_press": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml",
    "sec_litigation": "https://www.sec.gov/rss/litigation/litreleases.xml",
}


def utc_age_seconds(doc: RawDoc) -> float:
    """How stale a doc was when we got it. High values mean the feed is lagging."""
    if doc.published is None:
        return float("nan")
    pub = doc.published
    if pub.tzinfo is None:
        pub = pub.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - pub).total_seconds()
