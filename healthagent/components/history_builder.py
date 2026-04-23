from __future__ import annotations

from healthagent.schemas import (
    CompressedHistory,
    RetryFeedbackRecord,
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
    - searches: query + returned candidate results
    - visited_pages: visited url + task-focused summary
    - retry_feedback: write-time retry hints such as text-too-long
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
        )

        searches = list(history.searches)
        if searches:
            last = searches[-1]
            if last.query == new_record.query and last.results == new_record.results:
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

    def update_with_write_text_too_long(
        self,
        history: CompressedHistory,
        *,
        attempted_text: str,
        text_length: int,
        max_text_chars: int,
    ) -> CompressedHistory:
        feedback = RetryFeedbackRecord(
            kind="write_text_too_long",
            attempted_text=" ".join((attempted_text or "").split()).strip(),
            text_length=text_length,
            max_text_chars=max_text_chars,
            message=(
                f"The attempted write text was too long: {text_length} characters, "
                f"but the maximum allowed is {max_text_chars}. Retry with a shorter text."
            ),
        )

        retry_feedback = list(history.retry_feedback)

        if retry_feedback:
            last = retry_feedback[-1]
            if (
                last.kind == feedback.kind
                and last.attempted_text == feedback.attempted_text
                and last.text_length == feedback.text_length
                and last.max_text_chars == feedback.max_text_chars
            ):
                return history

        retry_feedback.append(feedback)

        return history.model_copy(
            update={
                "retry_feedback": retry_feedback,
            }
        )