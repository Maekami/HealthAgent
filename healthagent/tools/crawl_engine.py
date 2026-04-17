from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import httpx

from .exceptions import HTTPToolError
from .url_utils import canonicalize_url


@dataclass(slots=True)
class FirecrawlScrapeResult:
    url: str
    markdown: Optional[str]
    html: Optional[str]
    raw_html: Optional[str]
    links: list[str]
    metadata: dict[str, Any]
    raw_response: dict[str, Any]


class CrawlEngine(Protocol):
    def crawl(self, url: str) -> FirecrawlScrapeResult:
        ...


class FirecrawlCrawlEngine:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.firecrawl.dev",
        timeout_s: float = 60.0,
        formats: list[str] | None = None,
        only_main_content: bool = True,
        wait_for_ms: int = 0,
        scrape_timeout_ms: int = 60_000,
        proxy: str = "auto",
        store_in_cache: bool = False,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.formats = formats or ["markdown", "html"]
        self.only_main_content = only_main_content
        self.wait_for_ms = wait_for_ms
        self.scrape_timeout_ms = scrape_timeout_ms
        self.proxy = proxy
        self.store_in_cache = store_in_cache

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, url: str) -> dict[str, Any]:
        return {
            "url": url,
            "formats": self.formats,
            "onlyMainContent": self.only_main_content,
            "waitFor": self.wait_for_ms,
            "timeout": self.scrape_timeout_ms,
            "proxy": self.proxy,
            "storeInCache": self.store_in_cache,
        }

    def scrape_raw(self, url: str) -> dict[str, Any]:
        endpoint = f"{self.base_url}/v2/scrape"
        payload = self._payload(url)

        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(
                endpoint,
                headers=self._headers(),
                json=payload,
            )

        if resp.status_code >= 400:
            raise HTTPToolError(
                "Firecrawl scrape request failed",
                status_code=resp.status_code,
                response_body=resp.text,
            )

        data = resp.json()
        if not isinstance(data, dict):
            raise HTTPToolError("Firecrawl response is not a JSON object", response_body=data)

        return data

    def crawl(self, url: str) -> FirecrawlScrapeResult:
        raw = self.scrape_raw(url)
        if not raw.get("success", False):
            raise HTTPToolError("Firecrawl scrape returned success=false", response_body=raw)

        data = raw.get("data") or {}
        if not isinstance(data, dict):
            raise HTTPToolError("Firecrawl scrape 'data' field is missing or invalid", response_body=raw)

        return FirecrawlScrapeResult(
            url=canonicalize_url(data.get("metadata", {}).get("url") or url),
            markdown=data.get("markdown"),
            html=data.get("html"),
            raw_html=data.get("rawHtml"),
            links=list(data.get("links") or []),
            metadata=dict(data.get("metadata") or {}),
            raw_response=raw,
        )