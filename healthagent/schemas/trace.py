from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import Field

from .base import SchemaBase


class StepTrace(SchemaBase):
    step_idx: int

    actor_prompt: str
    actor_raw_output: str
    parsed_action: Dict[str, Any]

    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[Dict[str, Any]] = None

    history_before: Dict[str, Any]
    history_after: Dict[str, Any]

    created_at: datetime = Field(default_factory=datetime.utcnow)


class EpisodeTrace(SchemaBase):
    episode_id: str
    post_id: str

    planner_prompt: str
    planner_raw_output: str
    planner_parsed_output: Dict[str, Any]

    steps: List[StepTrace] = Field(default_factory=list)
    final_output: Dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=datetime.utcnow)