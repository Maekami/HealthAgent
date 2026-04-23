from __future__ import annotations

import json
from json_repair import repair_json
import re
from dataclasses import dataclass
from typing import Annotated, Any, Mapping, Optional, Sequence, Type, Union

from pydantic import Field, ValidationError, create_model

from healthagent.prompts.actor import (
    build_actor_system_prompt,
    build_actor_user_prompt,
    build_write_refinement_system_prompt,
    build_write_refinement_user_prompt,
)
from healthagent.schemas import (
    AbstainAction,
    ActorDecision,
    ActorThinking,
    CompressedHistory,
    InstanceRubrics,
    PostPackage,
    SearchAction,
    VisitAction,
    WriteAction,
)
from healthagent.schemas.base import SchemaBase
from healthagent.tools import ChatClient
from healthagent.tools.url_utils import canonicalize_url


@dataclass(slots=True)
class ActorRun:
    decision: ActorDecision
    prompt: str
    raw_output: str

class WriteRefinementOutput(SchemaBase):
    text: str
    reason: str


@dataclass(slots=True)
class WriteRefinementRun:
    output: WriteRefinementOutput
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
    ) -> None:
        self.chat_client = chat_client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_json_mode = use_json_mode

    def _build_messages(
        self,
        *,
        post: PostPackage,
        global_rubrics: Sequence[str],
        instance_rubrics: InstanceRubrics,
        history: CompressedHistory,
        budget_state: Mapping[str, Any],
        actor_memory: Sequence[str] | None = None,
        available_actions: Sequence[str] | None = None,
    ) -> tuple[list[dict[str, str]], str]:
        system_prompt = build_actor_system_prompt()
        user_prompt = build_actor_user_prompt(
            post=post,
            global_rubrics=global_rubrics,
            instance_rubrics=instance_rubrics,
            history=history,
            budget_state=budget_state,
            actor_memory=actor_memory,
            available_actions=available_actions,
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

    def _normalized_post_url_aliases(
        self,
        post_url_aliases: Mapping[str, Sequence[str]] | None,
    ) -> list[set[str]]:
        alias_groups: list[set[str]] = []
        for _, aliases in (post_url_aliases or {}).items():
            group = {
                canonicalize_url(url)
                for url in aliases
                if canonicalize_url(url)
            }
            if group:
                alias_groups.append(group)
        return alias_groups

    def _allowed_write_urls(
        self,
        history: CompressedHistory,
        post_url_aliases: Mapping[str, Sequence[str]] | None,
    ) -> set[str]:
        visited_urls = self._visited_urls_from_history(history)
        allowed_urls = set(visited_urls)

        for alias_group in self._normalized_post_url_aliases(post_url_aliases):
            if alias_group & visited_urls:
                allowed_urls |= alias_group

        return allowed_urls

    def _allowed_action_models(
        self,
        available_actions: Sequence[str] | None,
    ) -> list[Type[SchemaBase]]:
        action_map: dict[str, Type[SchemaBase]] = {
            "search": SearchAction,
            "visit": VisitAction,
            "write": WriteAction,
            "abstain": AbstainAction,
        }
        allowed = list(available_actions or ["search", "visit", "write", "abstain"])
        models = [action_map[name] for name in allowed]
        if not models:
            raise ActorError("No available actions were provided.")
        return models

    def _build_response_schema(
        self,
        available_actions: Sequence[str] | None,
    ) -> dict[str, Any]:
        allowed_models = self._allowed_action_models(available_actions)
        action_union = Union.__getitem__(tuple(allowed_models))
        constrained_action_type = Annotated[
            action_union,
            Field(discriminator="action"),
        ]

        DynamicActorDecision = create_model(
            "DynamicActorDecision",
            thinking=(ActorThinking, ...),
            action=(constrained_action_type, ...),
            __base__=SchemaBase,
        )
        return DynamicActorDecision.model_json_schema()

    def _validate_decision(
        self,
        decision: ActorDecision,
        history: CompressedHistory,
        *,
        post_url_aliases: Mapping[str, Sequence[str]] | None = None,
        available_actions: Sequence[str] | None = None,
    ) -> ActorDecision:
        candidate_urls = self._candidate_urls_from_history(history)
        visited_urls = self._visited_urls_from_history(history)
        allowed_write_urls = self._allowed_write_urls(history, post_url_aliases)

        # Expand the URLs present in the tweets into the candidate pool
        for alias_group in self._normalized_post_url_aliases(post_url_aliases):
            candidate_urls |= alias_group

        action = decision.action
        allowed_actions_set = set(available_actions or ["search", "visit", "write", "abstain"])

        if action.action not in allowed_actions_set:
            raise ActorError(
                f"Action '{action.action}' is not allowed in this step. "
                f"Allowed actions: {sorted(allowed_actions_set)}"
            )

        if isinstance(action, VisitAction):
            normalized_url = canonicalize_url(action.url)
            if not normalized_url:
                raise ActorError("Visit action has an empty or invalid URL.")
            if normalized_url not in candidate_urls:
                raise ActorError(
                    "Visit action URL must come from URLs available in search history or post URLs."
                )
            decision = decision.model_copy(
                update={
                    "action": action.model_copy(update={"url": normalized_url})
                }
            )

        elif isinstance(action, WriteAction):
            if re.search(r"https?://|www\.", action.text):
                raise ActorError("Write action text must not contain URLs.")

            normalized_support = []
            for item in action.support:
                normalized_url = canonicalize_url(item.url)
                if normalized_url not in allowed_write_urls:
                    raise ActorError(
                        "Write action support URLs must come from previously visited pages. "
                        "For post URLs, short-link and resolved full-form aliases are both allowed."
                    )
                normalized_support.append(
                    item.model_copy(update={"url": normalized_url})
                )

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
        post_url_aliases: Mapping[str, Sequence[str]] | None = None,
        available_actions: Sequence[str] | None = None,
    ) -> ActorRun:
        messages, combined_prompt = self._build_messages(
            post=post,
            global_rubrics=global_rubrics,
            instance_rubrics=instance_rubrics,
            history=history,
            budget_state=budget_state,
            actor_memory=actor_memory,
            available_actions=available_actions,
        )

        response_format: dict[str, Any] | None = None
        if self.use_json_mode:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "actor-decision",
                    "schema": self._build_response_schema(available_actions),
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

        decision = self._parse_raw_output(generation.text)
        # decision = self._validate_decision(
        #     decision,
        #     history,
        #     post_url_aliases=post_url_aliases,
        #     available_actions=available_actions,
        # )

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
        post_url_aliases: Mapping[str, Sequence[str]] | None = None,
        available_actions: Sequence[str] | None = None,
    ) -> ActorDecision:
        return self.act_run(
            post=post,
            global_rubrics=global_rubrics,
            instance_rubrics=instance_rubrics,
            history=history,
            budget_state=budget_state,
            actor_memory=actor_memory,
            post_url_aliases=post_url_aliases,
            available_actions=available_actions,
        ).decision
    
    def _build_write_refinement_messages(
        self,
        *,
        original_text: str,
        support: Sequence[dict[str, Any]],
        max_text_chars: int,
        previous_failures: Sequence[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, str]], str]:
        system_prompt = build_write_refinement_system_prompt()
        user_prompt = build_write_refinement_user_prompt(
            original_text=original_text,
            support=support,
            max_text_chars=max_text_chars,
            previous_failures=previous_failures,
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

    def _parse_write_refinement_output(
        self,
        raw_output: str,
    ) -> WriteRefinementOutput:
        try:
            return WriteRefinementOutput.model_validate_json(raw_output)
        except (ValidationError, json.JSONDecodeError):
            pass

        extracted = _extract_json_object(raw_output)

        try:
            parsed = json.loads(extracted)
        except json.JSONDecodeError as exc:
            raise ActorError(
                f"Write refinement output is not valid JSON. Error: {exc}"
            ) from exc

        try:
            return WriteRefinementOutput.model_validate(parsed)
        except ValidationError as exc:
            raise ActorError(
                f"Write refinement output does not match schema: {exc}"
            ) from exc

    def refine_write_run(
        self,
        *,
        original_text: str,
        support: Sequence[dict[str, Any]],
        max_text_chars: int,
        previous_failures: Sequence[dict[str, Any]] | None = None,
    ) -> WriteRefinementRun:
        messages, combined_prompt = self._build_write_refinement_messages(
            original_text=original_text,
            support=support,
            max_text_chars=max_text_chars,
            previous_failures=previous_failures,
        )

        response_format: dict[str, Any] | None = None
        if self.use_json_mode:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "write-refinement",
                    "schema": WriteRefinementOutput.model_json_schema(),
                },
            }

        generation = self.chat_client.chat(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=response_format,
        )

        output = self._parse_write_refinement_output(generation.text)

        if re.search(r"https?://|www\.", output.text):
            raise ActorError("Refined write text must not contain URLs.")

        return WriteRefinementRun(
            output=output,
            prompt=combined_prompt,
            raw_output=generation.text,
        )