from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional
from json_repair import repair_json

from pydantic import ValidationError

from healthagent.prompts.actor_memory_query import (
    build_actor_memory_query_system_prompt,
    build_actor_memory_query_user_prompt,
)
from healthagent.prompts.planner_memory_query import (
    build_planner_memory_query_system_prompt,
    build_planner_memory_query_user_prompt,
)
from healthagent.schemas import (
    CompressedHistory,
    InstanceRubrics,
    MemoryQueryOutput,
    PostPackage,
)
from healthagent.tools import ChatClient


@dataclass(slots=True)
class MemoryQueryRun:
    output: MemoryQueryOutput
    prompt: str
    raw_output: str


class MemoryQueryBuilderError(RuntimeError):
    """Raised when memory query generation fails."""


def _extract_json_object(text: str) -> str:
    text = text.strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start : end + 1].strip()

    raise MemoryQueryBuilderError("Could not extract a JSON object from memory query output.")


class MemoryQueryBuilder:
    def __init__(
        self,
        chat_client: ChatClient,
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        use_json_mode: bool = True,
    ) -> None:
        self.chat_client = chat_client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_json_mode = use_json_mode

    def _parse_raw_output(self, raw_output: str) -> MemoryQueryOutput:
        try:
            return MemoryQueryOutput.model_validate_json(raw_output)
        except (ValidationError, json.JSONDecodeError):
            pass

        extracted = _extract_json_object(raw_output)

        try:
            parsed = json.loads(extracted)
        except json.JSONDecodeError as exc:
            raise MemoryQueryBuilderError(
                f"Memory query output is not valid JSON. Error: {exc}"
            ) from exc

        try:
            return MemoryQueryOutput.model_validate(parsed)
        except ValidationError as exc:
            raise MemoryQueryBuilderError(
                f"Memory query output does not match schema: {exc}"
            ) from exc

    def _chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> MemoryQueryRun:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        combined_prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"

        response_format: dict[str, Any] | None = None
        if self.use_json_mode:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "memory-query-output",
                    "schema": MemoryQueryOutput.model_json_schema(),
                },
            }

        generation = self.chat_client.chat(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=response_format,
        )
        generation.text = repair_json(generation.text) # FIXME:

        output = self._parse_raw_output(generation.text)

        return MemoryQueryRun(
            output=output,
            prompt=combined_prompt,
            raw_output=generation.text,
        )

    def build_planner_query_run(
        self,
        *,
        post: PostPackage,
    ) -> MemoryQueryRun:
        system_prompt = build_planner_memory_query_system_prompt()
        user_prompt = build_planner_memory_query_user_prompt(
            post=post,
        )
        return self._chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    def build_actor_query_run(
        self,
        *,
        post: PostPackage,
        instance_rubrics: InstanceRubrics,
        history: CompressedHistory,
        budget_state: Mapping[str, Any],
    ) -> MemoryQueryRun:
        system_prompt = build_actor_memory_query_system_prompt()
        user_prompt = build_actor_memory_query_user_prompt(
            post=post,
            instance_rubrics=instance_rubrics,
            history=history,
            budget_state=budget_state,
        )
        return self._chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    def build_planner_query(
        self,
        *,
        post: PostPackage,
    ) -> str:
        return self.build_planner_query_run(
            post=post,
        ).output.query

    def build_actor_query(
        self,
        *,
        post: PostPackage,
        instance_rubrics: InstanceRubrics,
        history: CompressedHistory,
        budget_state: Mapping[str, Any],
    ) -> str:
        return self.build_actor_query_run(
            post=post,
            instance_rubrics=instance_rubrics,
            history=history,
            budget_state=budget_state,
        ).output.query