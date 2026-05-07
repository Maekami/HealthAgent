from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional
from json_repair import repair_json

from pydantic import ValidationError

from healthagent.prompts.utility_mode_judge import (
    build_utility_mode_judge_system_prompt,
    build_utility_mode_judge_user_prompt,
)
from healthagent.schemas import PostPackage, UtilityModeJudgeOutput
from healthagent.tools import ChatClient


UtilityEpisodeMode = Literal["excellent", "satisfactory", "unsatisfactory"]


@dataclass(slots=True)
class UtilityModeJudgeRun:
    output: UtilityModeJudgeOutput
    mode: UtilityEpisodeMode
    prompt: str
    raw_output: str


class UtilityModeJudgeError(RuntimeError):
    """Raised when utility mode judging fails."""


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

    raise UtilityModeJudgeError("Could not extract a JSON object from mode judge output.")


class UtilityModeJudge:
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

    def _derive_mode(self, output: UtilityModeJudgeOutput) -> UtilityEpisodeMode:
        if not output.planner_ok:
            return "unsatisfactory"

        if not output.actor_task_completed:
            return "satisfactory"

        if output.actor_ok is True:
            return "excellent"

        return "satisfactory"

    def _build_messages(
        self,
        *,
        post: PostPackage,
        episode_run,
    ) -> tuple[list[dict[str, str]], str]:
        system_prompt = build_utility_mode_judge_system_prompt()
        user_prompt = build_utility_mode_judge_user_prompt(
            post=post,
            episode_run=episode_run,
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

    def _parse_raw_output(self, raw_output: str) -> UtilityModeJudgeOutput:
        try:
            return UtilityModeJudgeOutput.model_validate_json(raw_output)
        except (ValidationError, json.JSONDecodeError):
            pass

        extracted = _extract_json_object(raw_output)

        try:
            parsed = json.loads(extracted)
        except json.JSONDecodeError as exc:
            raise UtilityModeJudgeError(
                f"Mode judge output is not valid JSON. Error: {exc}"
            ) from exc

        try:
            return UtilityModeJudgeOutput.model_validate(parsed)
        except ValidationError as exc:
            raise UtilityModeJudgeError(
                f"Mode judge output does not match schema: {exc}"
            ) from exc

    def judge_run(
        self,
        *,
        post: PostPackage,
        episode_run,
    ) -> UtilityModeJudgeRun:
        messages, combined_prompt = self._build_messages(
            post=post,
            episode_run=episode_run,
        )

        response_format: dict[str, Any] | None = None
        if self.use_json_mode:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "utility-mode-judge-output",
                    "schema": UtilityModeJudgeOutput.model_json_schema(),
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
        print(generation.text)

        output = self._parse_raw_output(generation.text)
        mode = self._derive_mode(output)

        return UtilityModeJudgeRun(
            output=output,
            mode=mode,
            prompt=combined_prompt,
            raw_output=generation.text,
        )

    def judge(
        self,
        *,
        post: PostPackage,
        episode_run,
    ) -> UtilityEpisodeMode:
        return self.judge_run(
            post=post,
            episode_run=episode_run,
        ).mode