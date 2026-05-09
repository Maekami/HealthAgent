from __future__ import annotations

from typing import Mapping, Sequence

from healthagent.schemas import PostPackage

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

    for image in post.image_views:
        if image.caption:
            chunks.append(image.caption)
        if image.ocr:
            chunks.append(image.ocr)
        if image.description:
            chunks.append(image.description)

    for video in post.video_views:
        if video.transcript:
            chunks.append(video.transcript)
        if video.ocr:
            chunks.append(video.ocr)
        if video.temporal_summary:
            chunks.append(video.temporal_summary)
        chunks.extend(video.keyframe_captions)

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
    ) -> list[str]:
        effective_query = normalize_text(query or _post_to_planner_query(post))
        records = self._retrieve_records(
            query=effective_query,
            scopes=self.scopes,
        )
        return self._records_to_texts(records)


def build_planner_memory_records(
    items: Sequence[Mapping[str, str]],
    *,
    scope: MemoryScope = "planner",
    source: str = "manual",
    priority: float = 1.0,
) -> list[MemoryRecord]:
    records: list[MemoryRecord] = []
    for item in items:
        trigger = normalize_text(item.get("trigger", ""))
        rule = normalize_text(item.get("rule", ""))
        why = normalize_text(item.get("why", ""))
        if not trigger or not rule or not why:
            continue

        records.append(
            MemoryRecord(
                scope=scope,
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