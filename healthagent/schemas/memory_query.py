from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, field_validator

from .base import SchemaBase


MemoryStage = Literal["core_task_incomplete", "core_task_complete"]


class MemoryQueryOutput(SchemaBase):
    query: str = Field(
        min_length=1,
        max_length=512,
        description="A short generalized retrieval query that identifies the post type or claim archetype.",
    )
    stage: MemoryStage = Field(
        description="The current stage for selecting the right memory pool.",
    )
    rationale: Optional[str] = Field(
        default=None,
        max_length=512,
        description="Short explanation of why this memory query is needed right now.",
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