from __future__ import annotations

from typing import Mapping, Sequence

from healthagent.schemas import PostPackage
from healthagent.schemas.memory_query import MemoryStage

from .base import (
    BaseMemoryStore,
    BasePlannerMemory,
    EmptyPlannerMemory,
    LexicalMemoryMixin,
    MemoryRecord,
    MemoryScope,
    normalize_text,
)


def _post_to_planner_query(post: PostPackage) -> str:
    chunks: list[str] = [post.tweet_text]
    chunks.extend(post.caption.values())
    return normalize_text(" ".join(chunks))


class SimplePlannerMemory(LexicalMemoryMixin, BasePlannerMemory):
    def __init__(
        self,
        store: BaseMemoryStore,
        *,
        top_k: int = 5,
        min_score: float = 0.15,
        scopes: Sequence[MemoryScope] = ("planner", "shared"),
    ) -> None:
        super().__init__(store, top_k=top_k, min_score=min_score)
        self.scopes = list(scopes)

    def retrieve(
        self,
        *,
        post: PostPackage,
        query: str | None = None,
        stage: MemoryStage | None = None,
    ) -> list[str]:
        effective_query = normalize_text(query or _post_to_planner_query(post))
        effective_stage = stage or "core_task_incomplete"

        records = self._retrieve_records(
            query=effective_query,
            scopes=self.scopes,
            stage=effective_stage,
        )
        return self._records_to_texts(records)


def build_planner_memory_records(
    items: Sequence[Mapping[str, str]],
    *,
    scope: MemoryScope = "planner",
    source: str = "manual",
    priority: float = 1.0,
    default_stage: MemoryStage = "core_task_incomplete",
) -> list[MemoryRecord]:
    records: list[MemoryRecord] = []
    for item in items:
        trigger = normalize_text(item.get("trigger", ""))
        rule = normalize_text(item.get("rule", ""))
        why = normalize_text(item.get("why", ""))
        stage = item.get("stage") or default_stage
        if not trigger or not rule or not why:
            continue

        records.append(
            MemoryRecord(
                scope=scope,
                stage=stage,
                trigger=trigger,
                rule=rule,
                why=why,
                source=source,
                priority=priority,
            )
        )
    return records


__all__ = [
    "BasePlannerMemory",
    "EmptyPlannerMemory",
    "SimplePlannerMemory",
    "build_planner_memory_records",
]