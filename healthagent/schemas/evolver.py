from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field, field_validator, model_validator

from .base import SchemaBase


EpisodeMode = Literal["excellent", "satisfactory", "unsatisfactory"]


def _normalize_item(item: str, *, max_len: int = 260) -> str:
    item = " ".join(item.split()).strip()
    item = item.lstrip("-*•.0123456789) ]")
    item = " ".join(item.split()).strip()
    if len(item) > max_len:
        item = item[:max_len].rstrip()
    return item


class EvolverMemoryItem(SchemaBase):
    trigger: str = Field(
        min_length=1,
        max_length=512,
        description="A recurring post or evidence pattern where this memory applies.",
    )
    rule: str = Field(
        min_length=1,
        max_length=512,
        description="What the planner or actor should do for that pattern.",
    )
    why: str = Field(
        min_length=1,
        max_length=512,
        description="Why this pattern matters or why the rule is useful.",
    )

    @field_validator("trigger", "rule", "why", mode="before")
    @classmethod
    def _normalize_fields(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("Expected a string.")
        return _normalize_item(value)


class UtilityModeJudgeOutput(SchemaBase):
    planner_rationale: str = Field(
        min_length=1,
        max_length=1024,
        description="Why the planner did or did not successfully identify the necessary core claims.",
    )
    planner_ok: bool

    actor_task_rationale: Optional[str] = Field(
        default=None,
        description="Why the actor did or did not complete the planner-defined task at the minimally good level.",
    )
    actor_task_completed: Optional[bool] = Field(
        default=None,
        description="Whether the actor completed the planner-defined task at the minimally good level.",
    )

    actor_excellent_rationale: Optional[str] = Field(
        default=None,
        description="Why the actor does or does not meet the stricter excellent standard, evaluated only if actor_task_completed is true.",
    )
    actor_ok: Optional[bool] = Field(
        default=None,
        description="Whether the actor meets the stricter excellent standard. This is only evaluated if actor_task_completed is true.",
    )

    @field_validator("planner_rationale", mode="before")
    @classmethod
    def _normalize_planner_rationale(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("Expected a string.")
        return " ".join(value.split()).strip()

    @field_validator("actor_task_rationale", "actor_excellent_rationale", mode="before")
    @classmethod
    def _normalize_optional_rationales(cls, value):
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("Expected a string.")
        return " ".join(value.split()).strip()

    @model_validator(mode="after")
    def _validate_actor_fields(self):
        if self.planner_ok:
            if self.actor_task_completed is None:
                raise ValueError(
                    "actor_task_completed must be provided when planner_ok is true."
                )
            if not self.actor_task_rationale:
                raise ValueError(
                    "actor_task_rationale must be provided when planner_ok is true."
                )

            if self.actor_task_completed:
                if self.actor_ok is None:
                    raise ValueError(
                        "actor_ok must be provided when actor_task_completed is true."
                    )
                if not self.actor_excellent_rationale:
                    raise ValueError(
                        "actor_excellent_rationale must be provided when actor_task_completed is true."
                    )

        return self


class UtilityEvolverExcellentOutput(SchemaBase):
    episode_assessment: str = Field(
        min_length=1,
        max_length=1024,
        description="Concise assessment of why the episode was excellent.",
    )
    successful_patterns: List[str] = Field(
        default_factory=list,
        min_length=1,
        max_length=4,
        description="Reusable successful patterns from the episode.",
    )
    planner_memory_items: List[EvolverMemoryItem] = Field(
        default_factory=list,
        min_length=1,
        max_length=4,
        description="Generalizable planning guidance for future episodes.",
    )
    actor_memory_items: List[EvolverMemoryItem] = Field(
        default_factory=list,
        min_length=1,
        max_length=4,
        description="Generalizable execution guidance for future episodes.",
    )

    @field_validator("episode_assessment", mode="before")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("Expected a string.")
        return " ".join(value.split()).strip()

    @field_validator("successful_patterns", mode="before")
    @classmethod
    def _normalize_successful_patterns(cls, value):
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("Expected a list of strings.")
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                raise TypeError("All items must be strings.")
            item = _normalize_item(item)
            if not item:
                continue
            if item not in seen:
                normalized.append(item)
                seen.add(item)
        return normalized

    @model_validator(mode="after")
    def _validate_minimum_memory(self):
        if len(self.planner_memory_items) + len(self.actor_memory_items) < 1:
            raise ValueError(
                "Excellent episodes must produce at least one memory item for planner or actor."
            )
        return self


class UtilityEvolverSatisfactoryOutput(SchemaBase):
    episode_assessment: str = Field(
        min_length=1,
        max_length=1024,
        description="Concise assessment of why the episode was satisfactory but not excellent.",
    )
    actor_improvement_needs: List[str] = Field(
        default_factory=list,
        min_length=1,
        max_length=4,
        description="High-impact actor-side weaknesses or missed opportunities.",
    )
    highest_utility_actor_improvements: List[str] = Field(
        default_factory=list,
        min_length=1,
        max_length=3,
        description="The most useful actor-side improvements for future episodes.",
    )
    actor_memory_items: List[EvolverMemoryItem] = Field(
        default_factory=list,
        min_length=1,
        max_length=4,
        description="Generalizable actor guidance for future episodes. Must contain at least one item.",
    )
    planner_memory_items: List[EvolverMemoryItem] = Field(
        default_factory=list,
        max_length=4,
        description="Optional generalizable planning guidance if there is something worth preserving.",
    )

    @field_validator("episode_assessment", mode="before")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("Expected a string.")
        return " ".join(value.split()).strip()

    @field_validator(
        "actor_improvement_needs",
        "highest_utility_actor_improvements",
        mode="before",
    )
    @classmethod
    def _normalize_lists(cls, value):
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("Expected a list of strings.")
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                raise TypeError("All items must be strings.")
            item = _normalize_item(item)
            if not item:
                continue
            if item not in seen:
                normalized.append(item)
                seen.add(item)
        return normalized


class UtilityEvolverUnsatisfactoryOutput(SchemaBase):
    episode_assessment: str = Field(
        min_length=1,
        max_length=1024,
        description="Concise assessment of why the planner failed to frame the task properly.",
    )
    planner_failure_modes: List[str] = Field(
        default_factory=list,
        min_length=1,
        max_length=4,
        description="Planner-side failure patterns from this episode.",
    )
    planner_memory_items: List[EvolverMemoryItem] = Field(
        default_factory=list,
        min_length=1,
        max_length=4,
        description="Generalizable planning guidance for future episodes. Must contain at least one item.",
    )

    @field_validator("episode_assessment", mode="before")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("Expected a string.")
        return " ".join(value.split()).strip()

    @field_validator("planner_failure_modes", mode="before")
    @classmethod
    def _normalize_lists(cls, value):
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("Expected a list of strings.")
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                raise TypeError("All items must be strings.")
            item = _normalize_item(item)
            if not item:
                continue
            if item not in seen:
                normalized.append(item)
                seen.add(item)
        return normalized