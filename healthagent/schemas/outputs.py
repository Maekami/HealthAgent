from __future__ import annotations

from typing import List, Literal, Optional

from .actions import ClaimSupport
from .base import SchemaBase


class FinalNote(SchemaBase):
    post_id: str
    text: str
    reference_urls: List[str]
    note: str
    support: List[ClaimSupport]
    reason: str
    effective_length: int


class AbstainDecision(SchemaBase):
    post_id: str
    reason: str


class EpisodeResult(SchemaBase):
    post_id: str
    status: Literal["written", "abstained"]
    final_note: Optional[FinalNote] = None
    abstain: Optional[AbstainDecision] = None