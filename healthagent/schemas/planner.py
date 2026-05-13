from __future__ import annotations

from pydantic import Field, field_validator

from .base import SchemaBase
from .rubrics import InstanceRubrics


class RubricPlannerOutput(SchemaBase):
    current_assessment: str = Field(
        min_length=1,
        max_length=512,
        description="Concise summary of what the post is mainly doing and what the planner needs to frame.",
    )
    memory_reflection: str = Field(
        min_length=1,
        max_length=512,
        description="Short reflection on whether retrieved planner memory is useful for this post pattern.",
    )
    rubrics: InstanceRubrics

    @field_validator("current_assessment", "memory_reflection", mode="before")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("Expected a string.")
        value = " ".join(value.split()).strip()
        return value