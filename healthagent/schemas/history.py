from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from .base import SchemaBase
from .observations import SearchResultItem


class SearchRecord(SchemaBase):
    query: str
    results: List[SearchResultItem] = Field(default_factory=list)
    error_message: Optional[str] = None


class VisitedPageRecord(SchemaBase):
    url: str
    summary: str


class CompressedHistory(SchemaBase):
    searches: List[SearchRecord] = Field(default_factory=list)
    visited_pages: List[VisitedPageRecord] = Field(default_factory=list)