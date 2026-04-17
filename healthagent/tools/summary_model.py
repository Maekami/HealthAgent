from __future__ import annotations

from typing import Optional

from healthagent.schemas import VisitObservation

from .crawl_engine import FirecrawlScrapeResult
from .llm_client import ChatClient
from .url_utils import extract_domain, truncate_text


class GoalConditionedSummaryModel:
    def __init__(
        self,
        chat_client: ChatClient,
        *,
        model: Optional[str] = None,
        max_input_chars: int = 20_000,
        max_output_tokens: int = 8192,
        temperature: float = 0.0,
    ) -> None:
        self.chat_client = chat_client
        self.model = model
        self.max_input_chars = max_input_chars
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    def _build_messages(self, scrape_result: FirecrawlScrapeResult, goal: str) -> list[dict[str, str]]:
        page_content = (
            scrape_result.markdown
            or scrape_result.html
            or scrape_result.raw_html
            or ""
        )
        page_content = truncate_text(page_content, self.max_input_chars)

        title = scrape_result.metadata.get("title") or ""
        domain = extract_domain(scrape_result.url)

        system = (
            "You summarize crawled webpages for an evidence-grounded note-writing agent. "
            "Follow the user's goal exactly. Use only the provided page content. "
            "Do not add outside knowledge. If the page does not provide the needed information, "
            "say so briefly and clearly."
        )

        user = (
            f"URL: {scrape_result.url}\n"
            f"Domain: {domain}\n"
            f"Title: {title}\n\n"
            f"Goal:\n{goal}\n\n"
            f"Page content:\n{page_content}\n\n"
            "Write a concise task-focused summary."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def summarize(self, scrape_result: FirecrawlScrapeResult, goal: str) -> VisitObservation:
        messages = self._build_messages(scrape_result, goal)
        generation = self.chat_client.chat(
            messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )

        return VisitObservation(
            url=scrape_result.url,
            page_title=scrape_result.metadata.get("title"),
            domain=extract_domain(scrape_result.url),
            summary=generation.text,
        )