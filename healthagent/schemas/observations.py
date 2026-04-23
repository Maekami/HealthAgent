from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from .base import SchemaBase


class SearchResultItem(SchemaBase):
    url: str
    title: str
    domain: str
    snippet: str
    source_type: Optional[str] = None


class SearchObservation(SchemaBase):
    results: List[SearchResultItem] = Field(default_factory=list)
    error_message: Optional[str] = None


class VisitObservation(SchemaBase):
    url: str
    page_title: Optional[str] = None
    domain: Optional[str] = None
    summary: str