from __future__ import annotations

from healthagent.schemas import (
    CompressedHistory,
    SearchObservation,
    SearchRecord,
    VisitObservation,
    VisitedPageRecord,
)
from healthagent.tools.url_utils import canonicalize_url


class HistoryBuilder:
    """
    Maintains a minimal compressed history for the agent.

    Current design:
    - searches: query + returned candidate results + optional error message
    - visited_pages: visited url + task-focused summary
    """

    def update_with_search(
        self,
        history: CompressedHistory,
        query: str,
        observation: SearchObservation,
    ) -> CompressedHistory:
        query = " ".join((query or "").split()).strip()

        normalized_results = []
        seen_urls = set()
        for item in observation.results:
            normalized_url = canonicalize_url(item.url)
            if not normalized_url or normalized_url in seen_urls:
                continue

            seen_urls.add(normalized_url)
            normalized_results.append(
                item.model_copy(update={"url": normalized_url})
            )

        new_record = SearchRecord(
            query=query,
            results=normalized_results,
            error_message=(
                " ".join((observation.error_message or "").split()).strip()
                or None
            ),
        )

        searches = list(history.searches)
        if searches:
            last = searches[-1]
            if (
                last.query == new_record.query
                and last.results == new_record.results
                and last.error_message == new_record.error_message
            ):
                return history

        searches.append(new_record)

        return history.model_copy(
            update={
                "searches": searches,
            }
        )

    def update_with_visit(
        self,
        history: CompressedHistory,
        observation: VisitObservation,
    ) -> CompressedHistory:
        normalized_url = canonicalize_url(observation.url)

        new_record = VisitedPageRecord(
            url=normalized_url,
            summary=" ".join((observation.summary or "").split()).strip(),
        )

        visited_pages = list(history.visited_pages)

        replaced = False
        for idx, record in enumerate(visited_pages):
            if canonicalize_url(record.url) == normalized_url:
                visited_pages[idx] = new_record
                replaced = True
                break

        if not replaced:
            visited_pages.append(new_record)

        return history.model_copy(
            update={
                "visited_pages": visited_pages,
            }
        )