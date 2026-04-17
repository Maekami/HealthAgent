from __future__ import annotations

import json
from json_repair import repair_json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from pydantic import ValidationError

from healthagent.prompts.actor import (
    build_actor_system_prompt,
    build_actor_user_prompt,
)
from healthagent.schemas import (
    ActorDecision,
    CompressedHistory,
    InstanceRubrics,
    PostPackage,
    VisitAction,
    WriteAction,
)
from healthagent.tools import ChatClient
from healthagent.tools.url_utils import canonicalize_url
from healthagent.utils.note_formatting import dedupe_support_urls, effective_note_length


@dataclass(slots=True)
class ActorRun:
    decision: ActorDecision
    prompt: str
    raw_output: str


class ActorError(RuntimeError):
    """Raised when actor generation or validation fails."""


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

    raise ActorError("Could not extract a JSON object from actor output.")


class Actor:
    def __init__(
        self,
        chat_client: ChatClient,
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        use_json_mode: bool = True,
        max_text_chars: int = 270,
        max_platform_chars: int = 280,
    ) -> None:
        self.chat_client = chat_client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_json_mode = use_json_mode
        self.max_text_chars = max_text_chars
        self.max_platform_chars = max_platform_chars

    def _build_messages(
        self,
        *,
        post: PostPackage,
        global_rubrics: Sequence[str],
        instance_rubrics: InstanceRubrics,
        history: CompressedHistory,
        budget_state: Mapping[str, Any],
        actor_memory: Sequence[str] | None = None,
    ) -> tuple[list[dict[str, str]], str]:
        system_prompt = build_actor_system_prompt()
        user_prompt = build_actor_user_prompt(
            post=post,
            global_rubrics=global_rubrics,
            instance_rubrics=instance_rubrics,
            history=history,
            budget_state=budget_state,
            actor_memory=actor_memory,
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

    def _parse_raw_output(self, raw_output: str) -> ActorDecision:
        try:
            return ActorDecision.model_validate_json(raw_output)
        except (ValidationError, json.JSONDecodeError):
            pass

        extracted = _extract_json_object(raw_output)

        try:
            parsed = json.loads(extracted)
        except json.JSONDecodeError as exc:
            raise ActorError(f"Actor output is not valid JSON. Error: {exc}") from exc

        try:
            return ActorDecision.model_validate(parsed)
        except ValidationError as exc:
            raise ActorError(
                f"Actor output does not match ActorDecision schema: {exc}"
            ) from exc

    def _candidate_urls_from_history(self, history: CompressedHistory) -> set[str]:
        urls: set[str] = set()
        for search in history.searches:
            for result in search.results:
                normalized = canonicalize_url(result.url)
                if normalized:
                    urls.add(normalized)
        return urls

    def _visited_urls_from_history(self, history: CompressedHistory) -> set[str]:
        return {
            canonicalize_url(item.url)
            for item in history.visited_pages
            if canonicalize_url(item.url)
        }

    def _validate_decision(
        self,
        decision: ActorDecision,
        history: CompressedHistory,
    ) -> ActorDecision:
        candidate_urls = self._candidate_urls_from_history(history)
        visited_urls = self._visited_urls_from_history(history)

        action = decision.action

        if isinstance(action, VisitAction):
            normalized_url = canonicalize_url(action.url)
            if not normalized_url:
                raise ActorError("Visit action has an empty or invalid URL.")
            # FIXME:
            # if normalized_url not in candidate_urls:
            #     raise ActorError(
            #         "Visit action URL must come from URLs available in search history."
            #     )
            decision = decision.model_copy(
                update={
                    "action": action.model_copy(update={"url": normalized_url})
                }
            )

        elif isinstance(action, WriteAction):
            if re.search(r"https?://|www\.", action.text):
                raise ActorError("Write action text must not contain URLs.")

            # if len(action.text) > self.max_text_chars:
            #     raise ActorError(
            #         f"Write action text exceeds max length of {self.max_text_chars} characters."
            #     )

            normalized_support = []
            for item in action.support:
                normalized_url = canonicalize_url(item.url)
                # if normalized_url not in visited_urls:
                #     raise ActorError(
                #         "Write action support URLs must come from previously visited pages."
                #     )
                normalized_support.append(
                    item.model_copy(update={"url": normalized_url})
                )

            unique_urls = dedupe_support_urls(normalized_support)
            platform_len = effective_note_length(action.text, unique_urls)

            # if platform_len > self.max_platform_chars:
            #     raise ActorError(
            #         f"Write action exceeds platform limit: effective_length={platform_len}, "
            #         f"max_allowed={self.max_platform_chars}."
            #     )

            decision = decision.model_copy(
                update={
                    "action": action.model_copy(update={"support": normalized_support})
                }
            )

        return decision

    def act_run(
        self,
        *,
        post: PostPackage,
        global_rubrics: Sequence[str],
        instance_rubrics: InstanceRubrics,
        history: CompressedHistory,
        budget_state: Mapping[str, Any],
        actor_memory: Sequence[str] | None = None,
    ) -> ActorRun:
        messages, combined_prompt = self._build_messages(
            post=post,
            global_rubrics=global_rubrics,
            instance_rubrics=instance_rubrics,
            history=history,
            budget_state=budget_state,
            actor_memory=actor_memory,
        )

        response_format: dict[str, Any] | None = None
        if self.use_json_mode:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "actor-decision",
                    "schema": ActorDecision.model_json_schema(),
                },
            }

        generation = self.chat_client.chat(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=response_format,
        )

        # print(generation.text) # FIXME:
        generation.text = repair_json(generation.text)

        decision = self._parse_raw_output(generation.text)
        decision = self._validate_decision(decision, history)

        return ActorRun(
            decision=decision,
            prompt=combined_prompt,
            raw_output=generation.text,
        )

    def act(
        self,
        *,
        post: PostPackage,
        global_rubrics: Sequence[str],
        instance_rubrics: InstanceRubrics,
        history: CompressedHistory,
        budget_state: Mapping[str, Any],
        actor_memory: Sequence[str] | None = None,
    ) -> ActorDecision:
        return self.act_run(
            post=post,
            global_rubrics=global_rubrics,
            instance_rubrics=instance_rubrics,
            history=history,
            budget_state=budget_state,
            actor_memory=actor_memory,
        ).decision