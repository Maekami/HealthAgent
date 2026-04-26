from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence
from uuid import uuid4

from pydantic import Field

from healthagent.schemas import CompressedHistory, InstanceRubrics, PostPackage
from healthagent.schemas.base import SchemaBase


MemoryScope = Literal["planner", "actor", "shared"]


_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]{2,}")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    return _TOKEN_PATTERN.findall(text)


def lexical_overlap_score(query: str, candidate: str) -> float:
    """
    Lightweight lexical scorer for MVP memory retrieval.
    """
    q_tokens = set(tokenize(query))
    c_tokens = set(tokenize(candidate))

    if not q_tokens or not c_tokens:
        return 0.0

    overlap = len(q_tokens & c_tokens)
    if overlap == 0:
        return 0.0

    # slightly favor dense overlap while penalizing overly long candidate text
    return overlap / math.sqrt(len(c_tokens))


class MemoryRecord(SchemaBase):
    record_id: str = Field(default_factory=lambda: str(uuid4()))
    scope: MemoryScope
    text: str
    tags: list[str] = Field(default_factory=list)
    source: str = "manual"
    priority: float = 1.0
    created_at: datetime = Field(default_factory=_utc_now)

    def searchable_text(self) -> str:
        tag_text = " ".join(self.tags)
        return normalize_text(f"{self.text} {tag_text}")


class BaseMemoryStore(ABC):
    @abstractmethod
    def list_records(
        self,
        *,
        scopes: Sequence[MemoryScope] | None = None,
    ) -> list[MemoryRecord]:
        raise NotImplementedError

    @abstractmethod
    def add_records(self, records: Sequence[MemoryRecord]) -> None:
        raise NotImplementedError


class InMemoryStore(BaseMemoryStore):
    def __init__(self, records: Sequence[MemoryRecord] | None = None) -> None:
        self._records: list[MemoryRecord] = list(records or [])

    def list_records(
        self,
        *,
        scopes: Sequence[MemoryScope] | None = None,
    ) -> list[MemoryRecord]:
        if not scopes:
            return list(self._records)
        scope_set = set(scopes)
        return [r for r in self._records if r.scope in scope_set]

    def add_records(self, records: Sequence[MemoryRecord]) -> None:
        self._records.extend(records)


class JsonlMemoryStore(BaseMemoryStore):
    """
    Simple append-only JSONL memory store.

    Good enough for manual seeding now, and later can be used by a self-evolving
    memory writer without changing the retrieval interface.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def list_records(
        self,
        *,
        scopes: Sequence[MemoryScope] | None = None,
    ) -> list[MemoryRecord]:
        scope_set = set(scopes or [])
        records: list[MemoryRecord] = []

        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = MemoryRecord.model_validate_json(line)
            if scope_set and record.scope not in scope_set:
                continue
            records.append(record)

        return records

    def add_records(self, records: Sequence[MemoryRecord]) -> None:
        if not records:
            return

        with self.path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(record.model_dump_json())
                f.write("\n")


class BasePlannerMemory(ABC):
    @abstractmethod
    def retrieve(self, post: PostPackage) -> list[str]:
        raise NotImplementedError


class BaseActorMemory(ABC):
    @abstractmethod
    def retrieve(
        self,
        post: PostPackage,
        history: CompressedHistory,
        instance_rubrics: InstanceRubrics,
    ) -> list[str]:
        raise NotImplementedError


class EmptyPlannerMemory(BasePlannerMemory):
    def retrieve(self, post: PostPackage) -> list[str]:
        return []


class EmptyActorMemory(BaseActorMemory):
    def retrieve(
        self,
        post: PostPackage,
        history: CompressedHistory,
        instance_rubrics: InstanceRubrics,
    ) -> list[str]:
        return []


class LexicalMemoryMixin:
    def __init__(
        self,
        store: BaseMemoryStore,
        *,
        top_k: int = 5,
        min_score: float = 0.15,
    ) -> None:
        self.store = store
        self.top_k = top_k
        self.min_score = min_score

    def _retrieve_records(
        self,
        *,
        query: str,
        scopes: Sequence[MemoryScope],
    ) -> list[MemoryRecord]:
        query = normalize_text(query)
        if not query:
            return []

        records = self.store.list_records(scopes=scopes)

        scored: list[tuple[float, MemoryRecord]] = []
        for record in records:
            score = lexical_overlap_score(query, record.searchable_text()) * record.priority
            if score >= self.min_score:
                scored.append((score, record))

        scored.sort(
            key=lambda item: (
                item[0],
                item[1].priority,
                item[1].created_at.timestamp(),
            ),
            reverse=True,
        )

        return [record for _, record in scored[: self.top_k]]

    def _records_to_texts(self, records: Iterable[MemoryRecord]) -> list[str]:
        return [record.text for record in records]