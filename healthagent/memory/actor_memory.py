from __future__ import annotations

from typing import Mapping, Sequence

from healthagent.schemas import CompressedHistory, InstanceRubrics, PostPackage

from .base import (
    BaseActorMemory,
    BaseMemoryStore,
    EmptyActorMemory,
    LexicalMemoryMixin,
    MemoryRecord,
    MemoryScope,
    normalize_text,
)


def _post_to_actor_query(
    post: PostPackage,
    history: CompressedHistory,
    instance_rubrics: InstanceRubrics,
) -> str:
    chunks: list[str] = [post.tweet_text]

    chunks.extend(instance_rubrics.core_checkworthy_claims)
    chunks.extend(instance_rubrics.priority_questions)
    chunks.extend(instance_rubrics.multimodal_risks)
    chunks.append(instance_rubrics.note_intent)

    for search in history.searches:
        chunks.append(search.query)
        if search.error_message:
            chunks.append(search.error_message)
        for result in search.results:
            chunks.append(result.title)
            chunks.append(result.snippet)

    for page in history.visited_pages:
        chunks.append(page.url)
        chunks.append(page.summary)

    return normalize_text(" ".join(chunks))


class SimpleActorMemory(LexicalMemoryMixin, BaseActorMemory):
    def __init__(
        self,
        store: BaseMemoryStore,
        *,
        top_k: int = 5,
        min_score: float = 0.15,
        scopes: Sequence[MemoryScope] = ("actor", "shared"),
    ) -> None:
        super().__init__(store, top_k=top_k, min_score=min_score)
        self.scopes = list(scopes)

    def retrieve(
        self,
        *,
        post: PostPackage,
        history: CompressedHistory,
        instance_rubrics: InstanceRubrics,
        query: str | None = None,
    ) -> list[str]:
        effective_query = normalize_text(
            query or _post_to_actor_query(post, history, instance_rubrics)
        )
        records = self._retrieve_records(
            query=effective_query,
            scopes=self.scopes,
        )
        return self._records_to_texts(records)


def build_actor_memory_records(
    items: Sequence[Mapping[str, str]],
    *,
    scope: MemoryScope = "actor",
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
    "BaseActorMemory",
    "EmptyActorMemory",
    "SimpleActorMemory",
    "build_actor_memory_records",
]