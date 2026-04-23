from __future__ import annotations

from typing import List, Literal

from pydantic import Field

from .base import SchemaBase
from .observations import SearchResultItem


class SearchRecord(SchemaBase):
    query: str
    results: List[SearchResultItem] = Field(default_factory=list)


class VisitedPageRecord(SchemaBase):
    url: str
    summary: str


class RetryFeedbackRecord(SchemaBase):
    kind: Literal["write_text_too_long"]
    attempted_text: str
    text_length: int
    max_text_chars: int
    message: str


class CompressedHistory(SchemaBase):
    searches: List[SearchRecord] = Field(default_factory=list)
    visited_pages: List[VisitedPageRecord] = Field(default_factory=list)
    retry_feedback: List[RetryFeedbackRecord] = Field(default_factory=list)