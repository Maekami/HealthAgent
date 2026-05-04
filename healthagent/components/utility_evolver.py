from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional, Union

from pydantic import ValidationError

from healthagent.memory import MemoryRecord
from healthagent.prompts.utility_evolver_excellent import (
    build_utility_evolver_excellent_system_prompt,
    build_utility_evolver_excellent_user_prompt,
)
from healthagent.prompts.utility_evolver_satisfactory import (
    build_utility_evolver_satisfactory_system_prompt,
    build_utility_evolver_satisfactory_user_prompt,
)
from healthagent.prompts.utility_evolver_unsatisfactory import (
    build_utility_evolver_unsatisfactory_system_prompt,
    build_utility_evolver_unsatisfactory_user_prompt,
)
from healthagent.schemas import (
    PostPackage,
    UtilityEvolverExcellentOutput,
    UtilityEvolverSatisfactoryOutput,
    UtilityEvolverUnsatisfactoryOutput,
)
from healthagent.tools import ChatClient

from .utility_mode_judge import UtilityEpisodeMode, UtilityModeJudge, UtilityModeJudgeRun


RoutedEvolverOutput = Union[
    UtilityEvolverExcellentOutput,
    UtilityEvolverSatisfactoryOutput,
    UtilityEvolverUnsatisfactoryOutput,
]


@dataclass(slots=True)
class UtilityEvolverRun:
    mode_judge_run: UtilityModeJudgeRun
    mode: UtilityEpisodeMode
    routed_output: RoutedEvolverOutput
    planner_records: list[MemoryRecord]
    actor_records: list[MemoryRecord]
    prompt: str
    raw_output: str


class UtilityEvolverError(RuntimeError):
    """Raised when utility evolution fails."""


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

    raise UtilityEvolverError("Could not extract a JSON object from utility evolver output.")


class UtilityEvolver:
    def __init__(
        self,
        *,
        mode_judge: UtilityModeJudge,
        chat_client: ChatClient,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        use_json_mode: bool = True,
        planner_priority: float = 1.0,
        actor_priority: float = 1.0,
    ) -> None:
        self.mode_judge = mode_judge
        self.chat_client = chat_client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_json_mode = use_json_mode
        self.planner_priority = planner_priority
        self.actor_priority = actor_priority

    def _build_messages(
        self,
        *,
        post: PostPackage,
        episode_run,
        judge_run: UtilityModeJudgeRun,
    ) -> tuple[list[dict[str, str]], str]:
        if judge_run.mode == "excellent":
            system_prompt = build_utility_evolver_excellent_system_prompt()
            user_prompt = build_utility_evolver_excellent_user_prompt(
                post=post,
                episode_run=episode_run,
                judge_output=judge_run.output,
            )
        elif judge_run.mode == "satisfactory":
            system_prompt = build_utility_evolver_satisfactory_system_prompt()
            user_prompt = build_utility_evolver_satisfactory_user_prompt(
                post=post,
                episode_run=episode_run,
                judge_output=judge_run.output,
            )
        else:
            system_prompt = build_utility_evolver_unsatisfactory_system_prompt()
            user_prompt = build_utility_evolver_unsatisfactory_user_prompt(
                post=post,
                episode_run=episode_run,
                judge_output=judge_run.output,
            )

        combined_prompt = (
            f"[SYSTEM]\n{system_prompt}\n\n"
            f"[USER]\n{user_prompt}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return messages, combined_prompt

    def _response_schema(self, mode: UtilityEpisodeMode) -> dict[str, Any]:
        if mode == "excellent":
            schema = UtilityEvolverExcellentOutput.model_json_schema()
        elif mode == "satisfactory":
            schema = UtilityEvolverSatisfactoryOutput.model_json_schema()
        else:
            schema = UtilityEvolverUnsatisfactoryOutput.model_json_schema()

        return {
            "type": "json_schema",
            "json_schema": {
                "name": "utility-evolver-routed-output",
                "schema": schema,
            },
        }

    def _parse_raw_output(
        self,
        raw_output: str,
        mode: UtilityEpisodeMode,
    ) -> RoutedEvolverOutput:
        target_model = {
            "excellent": UtilityEvolverExcellentOutput,
            "satisfactory": UtilityEvolverSatisfactoryOutput,
            "unsatisfactory": UtilityEvolverUnsatisfactoryOutput,
        }[mode]

        try:
            return target_model.model_validate_json(raw_output)
        except (ValidationError, json.JSONDecodeError):
            pass

        extracted = _extract_json_object(raw_output)

        try:
            parsed = json.loads(extracted)
        except json.JSONDecodeError as exc:
            raise UtilityEvolverError(
                f"Utility evolver output is not valid JSON. Error: {exc}"
            ) from exc

        try:
            return target_model.model_validate(parsed)
        except ValidationError as exc:
            raise UtilityEvolverError(
                f"Utility evolver output does not match schema: {exc}"
            ) from exc

    def _build_records(
        self,
        routed_output: RoutedEvolverOutput,
        *,
        mode: UtilityEpisodeMode,
    ) -> tuple[list[MemoryRecord], list[MemoryRecord]]:
        if mode == "unsatisfactory":
            planner_records = [
                MemoryRecord(
                    scope="planner",
                    text=text,
                    tags=["utility_evolver", "planner", mode],
                    source="utility_evolver",
                    priority=self.planner_priority,
                )
                for text in routed_output.planner_memory_items
            ]
            return planner_records, []

        if mode == "satisfactory":
            planner_records = [
                MemoryRecord(
                    scope="planner",
                    text=text,
                    tags=["utility_evolver", "planner", mode],
                    source="utility_evolver",
                    priority=self.planner_priority,
                )
                for text in routed_output.planner_memory_items
            ]
            actor_records = [
                MemoryRecord(
                    scope="actor",
                    text=text,
                    tags=["utility_evolver", "actor", mode],
                    source="utility_evolver",
                    priority=self.actor_priority,
                )
                for text in routed_output.actor_memory_items
            ]
            return planner_records, actor_records

        planner_records = [
            MemoryRecord(
                scope="planner",
                text=text,
                tags=["utility_evolver", "planner", mode],
                source="utility_evolver",
                priority=self.planner_priority,
            )
            for text in routed_output.planner_memory_items
        ]
        actor_records = [
            MemoryRecord(
                scope="actor",
                text=text,
                tags=["utility_evolver", "actor", mode],
                source="utility_evolver",
                priority=self.actor_priority,
            )
            for text in routed_output.actor_memory_items
        ]
        return planner_records, actor_records

    def evolve_run(
        self,
        *,
        post: PostPackage,
        episode_run,
    ) -> UtilityEvolverRun:
        mode_judge_run = self.mode_judge.judge_run(
            post=post,
            episode_run=episode_run,
        )

        messages, combined_prompt = self._build_messages(
            post=post,
            episode_run=episode_run,
            judge_run=mode_judge_run,
        )

        response_format: dict[str, Any] | None = None
        if self.use_json_mode:
            response_format = self._response_schema(mode_judge_run.mode)

        generation = self.chat_client.chat(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=response_format,
        )

        routed_output = self._parse_raw_output(
            generation.text,
            mode_judge_run.mode,
        )
        planner_records, actor_records = self._build_records(
            routed_output,
            mode=mode_judge_run.mode,
        )

        return UtilityEvolverRun(
            mode_judge_run=mode_judge_run,
            mode=mode_judge_run.mode,
            routed_output=routed_output,
            planner_records=planner_records,
            actor_records=actor_records,
            prompt=combined_prompt,
            raw_output=generation.text,
        )

    def evolve(
        self,
        *,
        post: PostPackage,
        episode_run,
    ) -> tuple[list[MemoryRecord], list[MemoryRecord]]:
        run = self.evolve_run(
            post=post,
            episode_run=episode_run,
        )
        return run.planner_records, run.actor_records