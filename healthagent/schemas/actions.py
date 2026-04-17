from __future__ import annotations

from typing import Annotated, List, Literal, Union

from pydantic import Field, field_validator

from .base import SchemaBase


class ClaimSupport(SchemaBase):
    claim: str
    url: str

    @field_validator("claim", "url", mode="before")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("Expected a string.")
        return " ".join(value.split()).strip()


class SearchAction(SchemaBase):
    action: Literal["search"]
    query: str
    reason: str

    @field_validator("query", "reason", mode="before")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("Expected a string.")
        return " ".join(value.split()).strip()


class VisitAction(SchemaBase):
    action: Literal["visit"]
    url: str
    goal: str
    reason: str

    @field_validator("url", "goal", "reason", mode="before")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("Expected a string.")
        return " ".join(value.split()).strip()


# class WriteAction(SchemaBase):
#     action: Literal["write"]
#     note: str
#     support: List[ClaimSupport] = Field(default_factory=list)
#     reason: str

#     @field_validator("note", "reason", mode="before")
#     @classmethod
#     def _normalize_text(cls, value: str) -> str:
#         if not isinstance(value, str):
#             raise TypeError("Expected a string.")
#         return " ".join(value.split()).strip()
class WriteAction(SchemaBase):
    action: Literal["write"]
    text: str
    support: List[ClaimSupport] = Field(default_factory=list)
    reason: str

    @field_validator("text", "reason", mode="before")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("Expected a string.")
        return " ".join(value.split()).strip()


class AbstainAction(SchemaBase):
    action: Literal["abstain"]
    reason: str

    @field_validator("reason", mode="before")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("Expected a string.")
        return " ".join(value.split()).strip()


AgentAction = Annotated[
    Union[SearchAction, VisitAction, WriteAction, AbstainAction],
    Field(discriminator="action"),
]


class ActorThinking(SchemaBase):
    current_assessment: str = Field(
        min_length=1,
        max_length=512,
        description="Concise summary of the current evidence state.",
    )
    main_gap: str = Field(
        min_length=1,
        max_length=256,
        description="The main missing piece or unresolved issue.",
    )
    decision_rationale: str = Field(
        min_length=1,
        max_length=512,
        description="Why this next action is the best next step.",
    )

    @field_validator(
        "current_assessment",
        "main_gap",
        "decision_rationale",
        mode="before",
    )
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("Expected a string.")
        value = " ".join(value.split()).strip()
        return value


class ActorDecision(SchemaBase):
    thinking: ActorThinking
    action: AgentAction