from __future__ import annotations

import hashlib
import re
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

from pipeline.patient_education.sources import SourceScope

# Reuse the product's externally-identifying UA for approved evidence ingestion.
from src.api.services.evidence.source_fetcher import USER_AGENT

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "mc_cid", "mc_eid", "intcmp",
}

BUCKET_PATTERNS: Dict[str, Sequence[str]] = {
    "cancer_type": [
        r"/cancer-types?(?:/|$)", r"/types(?:/|$)", r"/cancer/types",
        r"breast-cancer|lung-cancer|colorectal|prostate|melanoma|leukemia|lymphoma|myeloma",
    ],
    "diagnosis_testing": [
        r"diagnos", r"staging|stages-", r"biomarker|tumou?r-marker",
        r"tests?-and-procedures|biopsy|pathology|genetic-testing",
    ],
    "treatment": [
        r"/treatment", r"how-cancer-treated", r"chemotherapy", r"radiation",
        r"immunotherapy", r"targeted-therap", r"surgery", r"hormone-therap",
        r"stem-cell|car-t|clinical-trial",
    ],
    "side_effects": [
        r"side-effect", r"fatigue", r"nausea", r"vomit", r"diarrhea",
        r"constipation", r"neuropathy", r"mucositis|mouth-sore", r"pain",
        r"hair-loss|lymphedema|infection|neutropenia",
    ],
    "nutrition_lifestyle": [
        r"nutrition|diet|eating|appetite|taste|food", r"physical-activity|exercise",
        r"sleep", r"sexual|fertility", r"wellness|healthy-living",
    ],
    "supportive_care": [
        r"supportive-care|palliative", r"coping-with-cancer", r"mental-health",
        r"anxiety|depression|distress|counsel", r"support-group|social-support",
        r"patient-navigation",
    ],
    "financial_practical": [
        r"financial|cost|insurance|employment|transport|lodging|legal|practical",
    ],
    "caregiver": [r"caregiver|family-and-friends|caring-for"],
    "survivorship": [r"survivorship|after-treatment|life-after-cancer|recurrence|follow-up-care"],
    "prevention_screening": [r"screening|early-detection|prevention|reduce.*risk"],
}

BLOCKED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".css", ".js", ".xml", ".zip", ".mp3", ".mp4", ".webm",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
}


def normalize_url(url: str) -> str:
    p = urlsplit(url)
    host = (p.hostname or "").lower().rstrip(".")
    path = re.sub(r"/{2,}", "/", p.path or "/")
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    query = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if k.lower() not in TRACKING_PARAMS]
    return urlunsplit(((p.scheme or "https").lower(), host, path, urlencode(query), ""))


def host_matches(host: str, domain: str) -> bool:
    host = (host or "").lower().rstrip(".")
    domain = domain.lower().rstrip(".")
    return host == domain or host.endswith("." + domain)


def path_allowed(url: str, scope: SourceScope, include_pdfs: bool = True) -> bool:
    p = urlsplit(url)
    if not host_matches(p.hostname or "", scope.domain):
        return False
    path = p.path or "/"
    if scope.include_prefixes and not any(path.startswith(x) for x in scope.include_prefixes):
        return False
    if any(path.startswith(x) for x in scope.exclude_prefixes):
        return False
    if any(re.search(rx, url, re.I) for rx in scope.exclude_regexes):
        return False
    ext = Path(path).suffix.lower()
    if ext in BLOCKED_EXTENSIONS:
        return False
    if ext == ".pdf" and not include_pdfs:
        return False
    return True


def bucket_for(url: str, title: str = "") -> str:
    text = f"{url} {title}".lower()
    for bucket, patterns in BUCKET_PATTERNS.items():
        if any(re.search(pattern, text, re.I) for pattern in patterns):
            return bucket
    return "general"


def importance_score(url: str, bucket: str, is_seed: bool) -> float:
    score = 5.0 if is_seed else 0.0
    if bucket != "general":
        score += 3.0
    if bucket in {"treatment", "side_effects", "nutrition_lifestyle", "supportive_care"}:
        score += 1.0
    depth = len([x for x in urlsplit(url).path.split("/") if x])
    if 2 <= depth <= 6:
        score += 1.0
    if url.lower().endswith(".pdf"):
        score += 0.25
    return score


class SourceDiscoverer:
    """Sitemap-first + bounded same-domain crawl for a curated SourceScope."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        delay_seconds: float = 0.35,
        max_sitemaps: int = 300,
        max_crawl_pages: int = 2500,
        max_depth: int = 3,
        include_pdfs: bool = True,
    ) -> None:
        self.delay_seconds = delay_seconds
        self.max_sitemaps = max_sitemaps
        self.max_crawl_pages = max_crawl_pages
        self.max_depth = max_depth
        self.include_pdfs = include_pdfs
        self.client = httpx.Client(
            follow_redirects=True,
            timeout=timeout_seconds,
            headers={"User-Agent": USER_AGENT},
        )

    def close(self) -> None:
        self.client.close()

    def _robots(self, scope: SourceScope) -> Tuple[RobotFileParser, List[str]]:
        url = f"https://{scope.domain}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(url)
        sitemap_urls: List[str] = []
        try:
            response = self.client.get(url)
            if response.status_code < 400:
                lines = response.text.splitlines()
                rp.parse(lines)
                sitemap_urls = [
                    line.split(":", 1)[1].strip()
                    for line in lines
                    if line.lower().startswith("sitemap:")
                ]
                return rp, sitemap_urls
        except httpx.HTTPError:
            pass
        # No readable robots file: RobotFileParser with empty rules permits by default.
        rp.parse([])
        return rp, sitemap_urls

    @staticmethod
    def _robots_allows(rp: RobotFileParser, url: str) -> bool:
        try:
            return rp.can_fetch(USER_AGENT, url)
        except Exception:
            return False

    @staticmethod
    def _common_sitemaps(scope: SourceScope) -> List[str]:
        base = f"https://{scope.domain}"
        return [f"{base}/sitemap.xml", f"{base}/sitemap_index.xml", f"{base}/sitemap-index.xml"]

    def _parse_sitemap(self, url: str) -> Tuple[Optional[str], List[str]]:
        response = self.client.get(url)
        if response.status_code >= 400:
            return None, []
        root = ET.fromstring(response.content)
        tag = root.tag.rsplit("}", 1)[-1].lower()
        locs = []
        for el in root.iter():
            if el.tag.rsplit("}", 1)[-1].lower() == "loc" and el.text:
                locs.append(el.text.strip())
        return tag, locs

    def discover_sitemaps(self, scope: SourceScope, rp: RobotFileParser, declared: Sequence[str]) -> Set[str]:
        queue = deque(dict.fromkeys([*declared, *self._common_sitemaps(scope)]))
        seen_maps: Set[str] = set()
        pages: Set[str] = set()
        while queue and len(seen_maps) < self.max_sitemaps:
            sm = normalize_url(queue.popleft())
            if sm in seen_maps:
                continue
            seen_maps.add(sm)
            try:
                tag, locs = self._parse_sitemap(sm)
            except (httpx.HTTPError, ET.ParseError):
                continue
            if tag == "sitemapindex":
                for loc in locs:
                    if host_matches(urlsplit(loc).hostname or "", scope.domain):
                        queue.append(loc)
            else:
                for loc in locs:
                    u = normalize_url(loc)
                    if path_allowed(u, scope, self.include_pdfs) and self._robots_allows(rp, u):
                        pages.add(u)
            if self.delay_seconds:
                time.sleep(self.delay_seconds)
        return pages

    def _links(self, url: str) -> List[str]:
        try:
            response = self.client.get(url)
            if response.status_code >= 400:
                return []
            ctype = response.headers.get("content-type", "").lower()
            if "html" not in ctype:
                return []
            soup = BeautifulSoup(response.content, "lxml")
            out: List[str] = []
            for anchor in soup.find_all("a", href=True):
                href = anchor.get("href")
                if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
                    continue
                out.append(urljoin(str(response.url), href))
            return out
        except httpx.HTTPError:
            return []

    def crawl(self, scope: SourceScope, rp: RobotFileParser) -> Set[str]:
        queue = deque((normalize_url(u), 0) for u in scope.seed_urls)
        seen: Set[str] = set()
        found: Set[str] = set()
        while queue and len(seen) < self.max_crawl_pages:
            url, depth = queue.popleft()
            if url in seen:
                continue
            seen.add(url)
            if not path_allowed(url, scope, self.include_pdfs) or not self._robots_allows(rp, url):
                continue
            found.add(url)
            if depth >= self.max_depth or url.lower().endswith(".pdf"):
                continue
            for href in self._links(url):
                candidate = normalize_url(href)
                if candidate not in seen and path_allowed(candidate, scope, self.include_pdfs):
                    queue.append((candidate, depth + 1))
            if self.delay_seconds:
                time.sleep(self.delay_seconds)
        return found

    def discover(self, scope: SourceScope) -> List[dict]:
        rp, declared = self._robots(scope)
        sitemap_urls = self.discover_sitemaps(scope, rp, declared)
        crawl_urls = self.crawl(scope, rp)
        seeds = {normalize_url(u) for u in scope.seed_urls}
        merged = sitemap_urls | crawl_urls | seeds
        rows = []
        for url in sorted(merged):
            bucket = bucket_for(url)
            rows.append({
                "source_key": scope.source_key,
                "domain": scope.domain,
                "url": url,
                "bucket": bucket,
                "from_sitemap": url in sitemap_urls,
                "from_root_crawl": url in crawl_urls,
                "is_seed": url in seeds,
                "importance": importance_score(url, bucket, url in seeds),
            })
        return rows


def coverage_gaps(rows: Sequence[dict], scope: SourceScope) -> List[str]:
    buckets = {r.get("bucket") for r in rows}
    return [bucket for bucket in scope.required_buckets if bucket not in buckets]
