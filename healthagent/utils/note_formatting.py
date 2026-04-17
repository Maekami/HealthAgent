from __future__ import annotations

from typing import Iterable, List

from healthagent.schemas import ClaimSupport
from healthagent.tools.url_utils import canonicalize_url


def dedupe_support_urls(support: Iterable[ClaimSupport]) -> List[str]:
    urls: List[str] = []
    seen: set[str] = set()

    for item in support:
        url = canonicalize_url(item.url)
        if not url or url in seen:
            continue
        urls.append(url)
        seen.add(url)

    return urls


def render_note_text_with_urls(text: str, urls: list[str]) -> str:
    text = " ".join((text or "").split()).strip()
    if not urls:
        return text
    return text + " " + " ".join(urls)


def effective_note_length(text: str, urls: list[str]) -> int:
    """
    Platform-specific effective length:
    - text counts by normal character length
    - each appended URL counts as 1 character
    """
    text = " ".join((text or "").split()).strip()
    return len(text) + len(urls)