from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol
from urllib.parse import quote

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
        wait_for_ms: int = 5000,
        scrape_timeout_ms: int = 60_000,
        proxy: str = "auto",
        store_in_cache: bool = False,
        max_retries: int = 3,
        retry_backoff_s: float = 1.0,
        max_backoff_s: float = 8.0,
        jina_api_key: str | None = None,
        jina_base_url: str = "https://r.jina.ai",
        jina_engine: str = "browser",
        jina_timeout_s: float | None = None,
        jina_use_post: bool = True,
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
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.max_backoff_s = max_backoff_s

        self.jina_api_key = jina_api_key
        self.jina_base_url = jina_base_url.rstrip("/")
        self.jina_engine = jina_engine
        self.jina_timeout_s = jina_timeout_s or timeout_s
        self.jina_use_post = jina_use_post

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(
        self,
        url: str,
        *,
        proxy_override: str | None = None,
    ) -> dict[str, Any]:
        return {
            "url": url,
            "formats": self.formats,
            "onlyMainContent": self.only_main_content,
            "waitFor": self.wait_for_ms,
            "timeout": self.scrape_timeout_ms,
            "proxy": proxy_override or self.proxy,
            "storeInCache": self.store_in_cache,
            "maxAge": 172800000,
        }

    def _jina_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "X-Engine": self.jina_engine,
        }

        if self.jina_api_key:
            headers["Authorization"] = f"Bearer {self.jina_api_key}"

        return headers

    def _compute_backoff(self, attempt_idx: int) -> float:
        """
        Compute exponential backoff.

        attempt 1 -> base
        attempt 2 -> base * 2
        attempt 3 -> base * 4
        ...
        """
        delay = self.retry_backoff_s * (2 ** (attempt_idx - 1))
        return min(delay, self.max_backoff_s)

    def _title_has_captcha(self, data: dict[str, Any]) -> bool:
        metadata = data.get("metadata") or {}
        title = metadata.get("title")
        if not isinstance(title, str):
            return False
        return "captcha" in title.lower()

    def _build_fallback_result(
        self,
        url: str,
        errors: list[str],
        *,
        used_stealth_retry: bool,
    ) -> FirecrawlScrapeResult:
        normalized_url = canonicalize_url(url)

        error_block = (
            "\n".join(f"- {err}" for err in errors[-self.max_retries :])
            if errors
            else "- Unknown error"
        )

        fallback_markdown = (
            "Unable to extract content from this URL after multiple attempts. "
            "Please try another URL.\n\n"
            f"Target URL: {normalized_url or url}\n\n"
            f"Used stealth proxy on retry: {used_stealth_retry}\n\n"
            f"Recent errors:\n{error_block}"
        )

        return FirecrawlScrapeResult(
            url=normalized_url or url,
            markdown=fallback_markdown,
            html=None,
            raw_html=None,
            links=[],
            metadata={
                "title": "Content extraction failed",
                "url": normalized_url or url,
                "extraction_failed": True,
                "error_count": len(errors),
                "used_stealth_retry": used_stealth_retry,
            },
            raw_response={
                "success": False,
                "fallback": True,
                "errors": errors,
                "used_stealth_retry": used_stealth_retry,
            },
        )

    def scrape_raw(
        self,
        url: str,
        *,
        proxy_override: str | None = None,
    ) -> dict[str, Any]:
        endpoint = f"{self.base_url}/v2/scrape"
        payload = self._payload(url, proxy_override=proxy_override)

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
            raise HTTPToolError(
                "Firecrawl response is not a JSON object",
                response_body=data,
            )

        return data

    def scrape_raw_jina(self, url: str) -> dict[str, Any]:
        if not self.jina_api_key:
            raise HTTPToolError("Jina API key is not configured")

        headers = self._jina_headers()

        with httpx.Client(timeout=self.jina_timeout_s) as client:
            if self.jina_use_post:
                # POST is safer for URLs with hash-based routes, because URL fragments
                # are not sent to the server in normal GET requests.
                resp = client.post(
                    f"{self.jina_base_url}/",
                    headers=headers,
                    data={"url": url},
                )
            else:
                # This matches the common Reader API prefix style:
                # https://r.jina.ai/https://example.com
                encoded_url = quote(url, safe=":/?#[]@!$&'()*+,;=%")
                resp = client.get(
                    f"{self.jina_base_url}/{encoded_url}",
                    headers=headers,
                )

        if resp.status_code >= 400:
            raise HTTPToolError(
                "Jina reader request failed",
                status_code=resp.status_code,
                response_body=resp.text,
            )

        data = resp.json()
        if not isinstance(data, dict):
            raise HTTPToolError(
                "Jina response is not a JSON object",
                response_body=data,
            )

        code = data.get("code")
        status = data.get("status")

        if isinstance(code, int) and code >= 400:
            raise HTTPToolError(
                "Jina reader returned an error code",
                status_code=code,
                response_body=data,
            )

        if isinstance(status, int) and status >= 400:
            raise HTTPToolError(
                "Jina reader returned an error status",
                status_code=status,
                response_body=data,
            )

        return data

    def _extract_jina_data(self, raw: dict[str, Any]) -> dict[str, Any]:
        # Newer JSON responses may wrap content in the "data" field.
        data = raw.get("data")
        if isinstance(data, dict):
            return data

        # Some JSON responses expose url/title/content at the root level.
        root_like_data = {
            key: raw.get(key)
            for key in (
                "url",
                "title",
                "content",
                "description",
                "links",
                "images",
                "usage",
            )
            if key in raw
        }

        if root_like_data:
            return root_like_data

        raise HTTPToolError(
            "Jina reader 'data' field is missing or invalid",
            response_body=raw,
        )

    def _links_from_jina_data(self, data: dict[str, Any]) -> list[str]:
        links_raw = data.get("links")

        if isinstance(links_raw, dict):
            return [
                link
                for link in links_raw.values()
                if isinstance(link, str)
            ]

        if isinstance(links_raw, list):
            return [
                link
                for link in links_raw
                if isinstance(link, str)
            ]

        return []

    def scrape_with_jina(self, url: str) -> FirecrawlScrapeResult:
        raw = self.scrape_raw_jina(url)
        data = self._extract_jina_data(raw)

        content = data.get("content")
        if not isinstance(content, str) or not content.strip():
            raise HTTPToolError(
                "Jina reader returned empty content",
                response_body=raw,
            )

        extracted_url = data.get("url") or url
        normalized_url = canonicalize_url(extracted_url) or extracted_url

        metadata: dict[str, Any] = {
            "title": data.get("title"),
            "description": data.get("description"),
            "url": normalized_url,
            "source": "jina",
            "extraction_provider": "jina",
        }

        usage = data.get("usage")
        if usage is not None:
            metadata["usage"] = usage

        images = data.get("images")
        if images is not None:
            metadata["images"] = images

        return FirecrawlScrapeResult(
            url=normalized_url,
            markdown=content,
            html=None,
            raw_html=None,
            links=self._links_from_jina_data(data),
            metadata=metadata,
            raw_response=raw,
        )

    def crawl(self, url: str) -> FirecrawlScrapeResult:
        errors: list[str] = []
        attempts = max(self.max_retries, 1)
        force_stealth_on_retry = False

        for attempt_idx in range(1, attempts + 1):
            proxy_override = "stealth" if force_stealth_on_retry else None

            try:
                raw = self.scrape_raw(url, proxy_override=proxy_override)

                if not raw.get("success", False):
                    raise HTTPToolError(
                        "Firecrawl scrape returned success=false",
                        response_body=raw,
                    )

                data = raw.get("data") or {}
                if not isinstance(data, dict):
                    raise HTTPToolError(
                        "Firecrawl scrape 'data' field is missing or invalid",
                        response_body=raw,
                    )

                if self._title_has_captcha(data):
                    force_stealth_on_retry = True
                    raise HTTPToolError(
                        "Firecrawl scrape returned a CAPTCHA page title",
                        response_body=data,
                    )

                return FirecrawlScrapeResult(
                    url=canonicalize_url(data.get("metadata", {}).get("url") or url),
                    markdown=data.get("markdown"),
                    html=data.get("html"),
                    raw_html=data.get("rawHtml"),
                    links=list(data.get("links") or []),
                    metadata=dict(data.get("metadata") or {}),
                    raw_response=raw,
                )

            except Exception as exc:
                errors.append(
                    f"attempt {attempt_idx} (proxy={proxy_override or self.proxy}): "
                    f"{type(exc).__name__}: {exc}"
                )

                if attempt_idx < attempts:
                    time.sleep(self._compute_backoff(attempt_idx))
                    continue

        # Final rescue attempt before returning the fallback result.
        if self.jina_api_key:
            try:
                return self.scrape_with_jina(url)
            except Exception as exc:
                errors.append(
                    "jina fallback: "
                    f"{type(exc).__name__}: {exc}"
                )

        return self._build_fallback_result(
            url,
            errors,
            used_stealth_retry=force_stealth_on_retry,
        )