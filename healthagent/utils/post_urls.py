from __future__ import annotations

import re
from typing import Dict, List, Set

import httpx

from healthagent.schemas import PostPackage
from healthagent.tools.url_utils import canonicalize_url


_URL_PATTERN = re.compile(r"https?://[^\s<>\"]+")


def _collect_text_fields(post: PostPackage) -> List[str]:
    texts: List[str] = []

    if post.tweet_text:
        texts.append(post.tweet_text)

    if post.caption:
        texts.extend(
            text
            for text in post.caption.values()
            if text
        )

    return texts


def extract_urls_from_post(post: PostPackage) -> List[str]:
    found: List[str] = []
    seen: Set[str] = set()

    for text in _collect_text_fields(post):
        for match in _URL_PATTERN.findall(text or ""):
            url = match.strip().rstrip(".,);]")
            if url and url not in seen:
                found.append(url)
                seen.add(url)

    return found


def resolve_url_aliases(
    url: str,
    *,
    timeout_s: float = 10.0,
) -> Set[str]:
    aliases: Set[str] = set()

    normalized_original = canonicalize_url(url)
    if normalized_original:
        aliases.add(normalized_original)

    try:
        with httpx.Client(
            timeout=timeout_s,
            follow_redirects=True,
            headers={"User-Agent": "health-agent/0.1"},
        ) as client:
            try:
                response = client.head(url)
            except Exception:
                response = client.get(url)

        final_url = str(response.url)
        normalized_final = canonicalize_url(final_url)
        if normalized_final:
            aliases.add(normalized_final)
    except Exception:
        pass

    return aliases


def build_post_url_aliases(
    post: PostPackage,
    *,
    timeout_s: float = 10.0,
) -> Dict[str, List[str]]:
    aliases: Dict[str, List[str]] = {}

    for raw_url in extract_urls_from_post(post):
        alias_set = resolve_url_aliases(raw_url, timeout_s=timeout_s)

        normalized_raw = canonicalize_url(raw_url)
        if normalized_raw:
            alias_set.add(normalized_raw)

        alias_list = sorted({alias for alias in alias_set if alias})
        aliases[raw_url] = alias_list

    return aliases