from __future__ import annotations

from typing import Optional

from pydantic import Field, field_validator

from .base import SchemaBase


class MemoryQueryOutput(SchemaBase):
    query: str = Field(
        min_length=1,
        max_length=512,
        description="A short retrieval query for memory search.",
    )
    rationale: Optional[str] = Field(
        default=None,
        max_length=512,
        description="Optional short explanation of why this query was generated.",
    )

    @field_validator("query", mode="before")
    @classmethod
    def _normalize_query(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("Expected a string.")
        value = " ".join(value.split()).strip()
        if not value:
            raise ValueError("query must not be empty.")
        return value

    @field_validator("rationale", mode="before")
    @classmethod
    def _normalize_rationale(cls, value):
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("Expected a string.")
        value = " ".join(value.split()).strip()
        return value or None