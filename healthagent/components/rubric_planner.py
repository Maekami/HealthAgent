from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from json_repair import repair_json
from pydantic import ValidationError

from healthagent.prompts.rubric_planner import (
    build_rubric_planner_system_prompt,
    build_rubric_planner_user_prompt,
)
from healthagent.schemas import InstanceRubrics, PostPackage, RubricPlannerOutput
from healthagent.tools import ChatClient


@dataclass(slots=True)
class RubricPlannerRun:
    output: RubricPlannerOutput
    rubrics: InstanceRubrics
    prompt: str
    raw_output: str


class RubricPlannerError(RuntimeError):
    """Raised when rubric planning fails."""


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

    raise RubricPlannerError("Could not extract a JSON object from planner output.")


class RubricPlanner:
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

    def _build_messages(
        self,
        post: PostPackage,
        planner_memory: Sequence[str] | None = None,
    ) -> tuple[list[dict[str, str]], str]:
        system_prompt = build_rubric_planner_system_prompt()
        user_prompt = build_rubric_planner_user_prompt(
            post=post,
            planner_memory=planner_memory,
        )
        combined_prompt = (
            f"[SYSTEM]\n{system_prompt}\n\n"
            f"[USER]\n{user_prompt}"
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return messages, combined_prompt

    def _parse_raw_output(self, raw_output: str) -> RubricPlannerOutput:
        try:
            return RubricPlannerOutput.model_validate_json(raw_output)
        except ValidationError:
            pass
        except json.JSONDecodeError:
            pass

        extracted = _extract_json_object(raw_output)

        try:
            parsed = json.loads(extracted)
        except json.JSONDecodeError as exc:
            raise RubricPlannerError(
                f"Planner output is not valid JSON. Error: {exc}"
            ) from exc

        try:
            return RubricPlannerOutput.model_validate(parsed)
        except ValidationError as exc:
            raise RubricPlannerError(
                f"Planner output does not match RubricPlannerOutput schema: {exc}"
            ) from exc

    def plan_run(
        self,
        post: PostPackage,
        *,
        planner_memory: Sequence[str] | None = None,
    ) -> RubricPlannerRun:
        messages, combined_prompt = self._build_messages(
            post=post,
            planner_memory=planner_memory,
        )

        response_format: dict[str, Any] | None = None
        if self.use_json_mode:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "rubric-planner-output",
                    "schema": RubricPlannerOutput.model_json_schema(),
                },
            }

        generation = self.chat_client.chat(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=response_format,
        )

        generation.text = repair_json(generation.text)
        output = self._parse_raw_output(generation.text)

        return RubricPlannerRun(
            output=output,
            rubrics=output.rubrics,
            prompt=combined_prompt,
            raw_output=generation.text,
        )

    def plan(
        self,
        post: PostPackage,
        *,
        planner_memory: Sequence[str] | None = None,
    ) -> InstanceRubrics:
        return self.plan_run(
            post=post,
            planner_memory=planner_memory,
        ).rubrics