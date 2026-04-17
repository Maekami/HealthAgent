from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


_TRACKING_PARAMS_PREFIXES = ("utm_",)
_TRACKING_PARAMS_EXACT = {
    "fbclid",
    "gclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "source",
}


def canonicalize_url(url: str) -> str:
    """Normalize URL for dedup/debugging."""
    url = (url or "").strip()
    if not url:
        return ""

    parsed = urlparse(url)
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"

    query_items = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        lower_key = key.lower()
        if lower_key in _TRACKING_PARAMS_EXACT:
            continue
        if any(lower_key.startswith(prefix) for prefix in _TRACKING_PARAMS_PREFIXES):
            continue
        query_items.append((key, value))

    normalized_query = urlencode(query_items, doseq=True)

    return urlunparse(
        (
            scheme,
            netloc,
            path.rstrip("/") or "/",
            "",
            normalized_query,
            "",
        )
    )


def extract_domain(url: str) -> str:
    parsed = urlparse(url or "")
    return parsed.netloc.lower()


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]"