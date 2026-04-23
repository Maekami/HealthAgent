from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, Sequence

import httpx

from healthagent.schemas import SearchObservation, SearchResultItem

from .exceptions import HTTPToolError
from .url_utils import canonicalize_url, extract_domain


def clean_query(query: str) -> str:
    return " ".join((query or "").split())


def append_exclusions_to_query(
    query: str,
    exclusions: Sequence[str] | None = None,
) -> str:
    """
    Append simple negative filters to the query string.
    This is intentionally conservative for debugging clarity.
    """
    query = clean_query(query)
    if not exclusions:
        return query

    parts = [query]
    for item in exclusions:
        item = item.strip()
        if not item:
            continue
        if " " in item:
            parts.append(f'-"{item}"')
        elif "." in item:
            parts.append(f"-site:{item}")
        else:
            parts.append(f"-{item}")
    return " ".join(parts)


def build_serper_tbs(created_at_ms: int) -> str:
    dt = datetime.fromtimestamp(created_at_ms / 1000, tz=timezone.utc)
    cd_max = dt.strftime("%m/%d/%Y")
    return f"cdr:1,cd_min:01/01/1970,cd_max:{cd_max}"


class SearchEngine(Protocol):
    def search(
        self,
        query: str,
        *,
        created_at_ms: Optional[int] = None,
    ) -> SearchObservation:
        ...


class SerperSearchEngine:
    """
    Minimal Serper wrapper with retry + fallback behavior.

    Fallback behavior:
    - If all retries fail, return SearchObservation(results=[], error_message=...).
    - This avoids breaking the whole pipeline while still exposing failure information.
    """

    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = "https://google.serper.dev/search",
        api_key_header: str = "X-API-KEY",
        timeout_s: float = 30.0,
        num_results: int = 10,
        exclusions: Sequence[str] | None = None,
        max_retries: int = 3,
        retry_backoff_s: float = 1.0,
        max_backoff_s: float = 8.0,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.api_key_header = api_key_header
        self.timeout_s = timeout_s
        self.num_results = num_results
        self.exclusions = list(exclusions or [])
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.max_backoff_s = max_backoff_s

    def _build_payload(self, query: str, created_at_ms: Optional[int]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "q": clean_query(append_exclusions_to_query(query, self.exclusions)),
            "num": self.num_results,
        }
        if created_at_ms is not None:
            payload["tbs"] = build_serper_tbs(created_at_ms)
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            self.api_key_header: self.api_key,
        }

    def _compute_backoff(self, attempt_idx: int) -> float:
        """
        Exponential backoff:
        attempt 1 -> base
        attempt 2 -> base * 2
        attempt 3 -> base * 4
        ...
        """
        delay = self.retry_backoff_s * (2 ** (attempt_idx - 1))
        return min(delay, self.max_backoff_s)

    def _build_fallback_observation(
        self,
        query: str,
        errors: list[str],
    ) -> SearchObservation:
        error_message = (
            f"Search failed after multiple attempts for query: {query}. "
            f"Recent errors: {' | '.join(errors[-self.max_retries:])}"
        )
        return SearchObservation(
            results=[],
            error_message=error_message,
        )

    def search_raw(
        self,
        query: str,
        *,
        created_at_ms: Optional[int] = None,
    ) -> dict[str, Any]:
        payload = self._build_payload(query, created_at_ms)

        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(
                self.endpoint,
                headers=self._headers(),
                json=payload,
            )

        if resp.status_code >= 400:
            raise HTTPToolError(
                "Serper search request failed",
                status_code=resp.status_code,
                response_body=resp.text,
            )

        data = resp.json()
        if not isinstance(data, dict):
            raise HTTPToolError("Serper response is not a JSON object", response_body=data)
        return data

    def _parse_search_response(self, data: dict[str, Any]) -> SearchObservation:
        organic = data.get("organic") or []
        results: list[SearchResultItem] = []
        seen_urls: set[str] = set()

        for item in organic:
            raw_url = item.get("link") or ""
            url = canonicalize_url(raw_url)
            if not url or url in seen_urls:
                continue

            seen_urls.add(url)
            results.append(
                SearchResultItem(
                    url=url,
                    title=(item.get("title") or url).strip(),
                    domain=extract_domain(url),
                    snippet=(item.get("snippet") or "").strip(),
                    source_type="organic",
                )
            )
            if len(results) >= self.num_results:
                break

        if not results:
            answer_box = data.get("answerBox") or {}
            raw_url = answer_box.get("link") or ""
            url = canonicalize_url(raw_url)
            if url:
                results.append(
                    SearchResultItem(
                        url=url,
                        title=(answer_box.get("title") or url).strip(),
                        domain=extract_domain(url),
                        snippet=(answer_box.get("snippet") or answer_box.get("answer") or "").strip(),
                        source_type="answerBox",
                    )
                )

        return SearchObservation(
            results=results,
            error_message=None,
        )

    def search(
        self,
        query: str,
        *,
        created_at_ms: Optional[int] = None,
    ) -> SearchObservation:
        errors: list[str] = []
        attempts = max(self.max_retries, 1)

        for attempt_idx in range(1, attempts + 1):
            try:
                data = self.search_raw(query, created_at_ms=created_at_ms)
                return self._parse_search_response(data)

            except Exception as exc:
                errors.append(f"attempt {attempt_idx}: {type(exc).__name__}: {exc}")

                if attempt_idx < attempts:
                    time.sleep(self._compute_backoff(attempt_idx))
                    continue

        return self._build_fallback_observation(query, errors)