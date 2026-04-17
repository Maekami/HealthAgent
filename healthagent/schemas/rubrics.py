# from __future__ import annotations

# from typing import List, Literal

# from pydantic import Field

# from .base import SchemaBase


# NoteIntent = Literal["correction", "context", "mixed"]


# class InstanceRubrics(SchemaBase):
#     core_checkworthy_claims: List[str] = Field(default_factory=list)
#     priority_questions: List[str] = Field(default_factory=list)
#     multimodal_risks: List[str] = Field(default_factory=list)
#     note_intent: NoteIntent

from __future__ import annotations

from typing import List, Literal

from pydantic import Field, field_validator

from .base import SchemaBase


NoteIntent = Literal["correction", "context", "mixed"]


class InstanceRubrics(SchemaBase):
    core_checkworthy_claims: List[str] = Field(
        default_factory=list,
        min_length=1,
        max_length=4,
        description="Main checkworthy claims in the post package.",
    )
    priority_questions: List[str] = Field(
        default_factory=list,
        min_length=1,
        max_length=4,
        description="Most important questions to verify first.",
    )
    multimodal_risks: List[str] = Field(
        default_factory=list,
        max_length=3,
        description="Potential cross-modal risks if applicable.",
    )
    note_intent: NoteIntent = Field(
        description="Overall note intent: correction, context, or mixed."
    )

    @field_validator(
        "core_checkworthy_claims",
        "priority_questions",
        "multimodal_risks",
        mode="before",
    )
    @classmethod
    def _normalize_string_list(cls, value):
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("Expected a list of strings.")

        normalized: list[str] = []
        seen: set[str] = set()

        for item in value:
            if not isinstance(item, str):
                raise TypeError("All items must be strings.")

            item = " ".join(item.split()).strip()
            if not item:
                continue

            if len(item) > 220:
                item = item[:220].rstrip()

            if item not in seen:
                normalized.append(item)
                seen.add(item)

        return normalized