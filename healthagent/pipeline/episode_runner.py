from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from healthagent.components.actor import Actor
from healthagent.components.history_builder import HistoryBuilder
from healthagent.components.rubric_planner import RubricPlanner
from healthagent.prompts.global_rubrics import build_action_space_rubrics
from healthagent.schemas import (
    AbstainDecision,
    CompressedHistory,
    EpisodeResult,
    EpisodeTrace,
    FinalNote,
    PostPackage,
    StepTrace,
)
from healthagent.tools.crawl_engine import CrawlEngine
from healthagent.tools.search_engine import SearchEngine
from healthagent.tools.summary_model import GoalConditionedSummaryModel
from healthagent.utils.note_formatting import (
    dedupe_support_urls,
    effective_note_length,
    render_note_text_with_urls,
)
from healthagent.utils.post_urls import build_post_url_aliases


@dataclass(slots=True)
class EpisodeRun:
    result: EpisodeResult
    trace: EpisodeTrace
    final_history: CompressedHistory


class EpisodeRunnerError(RuntimeError):
    """Raised when episode execution fails."""


class EpisodeRunner:
    def __init__(
        self,
        *,
        planner: RubricPlanner,
        actor: Actor,
        history_builder: HistoryBuilder,
        search_engine: SearchEngine,
        crawl_engine: CrawlEngine,
        summary_model: GoalConditionedSummaryModel,
        global_rubrics: Sequence[str],
        max_steps: int = 6,
        max_searches: int = 3,
        max_visits: int = 4,
        max_text_chars: int = 270,
        max_platform_chars: int = 280,
        max_write_refinement_attempts: int = 5,
        stream_print: bool = False,
        stream_printer: Optional[Callable[[str], None]] = None,
        post_url_resolve_timeout_s: float = 10.0,
    ) -> None:
        self.planner = planner
        self.actor = actor
        self.history_builder = history_builder
        self.search_engine = search_engine
        self.crawl_engine = crawl_engine
        self.summary_model = summary_model
        self.global_rubrics = list(global_rubrics)
        self.max_steps = max_steps
        self.max_searches = max_searches
        self.max_visits = max_visits
        self.max_text_chars = max_text_chars
        self.max_platform_chars = max_platform_chars
        self.max_write_refinement_attempts = max_write_refinement_attempts
        self.stream_print = stream_print
        self.stream_printer = stream_printer or print
        self.post_url_resolve_timeout_s = post_url_resolve_timeout_s

    def _budget_state(
        self,
        *,
        step_idx: int,
        used_searches: int,
        used_visits: int,
    ) -> dict[str, Any]:
        available_actions = self._available_actions(
            step_idx=step_idx,
            used_searches=used_searches,
            used_visits=used_visits,
        )

        return {
            "current_step": step_idx,
            "max_steps": self.max_steps,
            "remaining_steps": max(self.max_steps - step_idx + 1, 0),
            "used_searches": used_searches,
            "remaining_searches": max(self.max_searches - used_searches, 0),
            "used_visits": used_visits,
            "remaining_visits": max(self.max_visits - used_visits, 0),
            "available_actions": available_actions,
            "is_last_step": step_idx >= self.max_steps,
        }

    def _available_actions(
        self,
        *,
        step_idx: int,
        used_searches: int,
        used_visits: int,
    ) -> list[str]:
        is_last_step = step_idx >= self.max_steps
        if is_last_step:
            return ["write", "abstain"]

        actions = []
        if used_searches < self.max_searches:
            actions.append("search")
        if used_visits < self.max_visits:
            actions.append("visit")
        actions.extend(["write", "abstain"])
        return actions

    def _emit(self, title: str, payload: Any) -> None:
        if not self.stream_print:
            return

        if isinstance(payload, str):
            body = payload
        else:
            body = json.dumps(payload, ensure_ascii=False, indent=2)

        self.stream_printer(f"\n===== {title} =====\n{body}\n")

    def _emit_step_header(self, step_idx: int, budget_state: dict[str, int]) -> None:
        if not self.stream_print:
            return

        self.stream_printer(
            "\n"
            + "=" * 24
            + f" STEP {step_idx} "
            + "=" * 24
            + "\n"
            + json.dumps(budget_state, ensure_ascii=False, indent=2)
            + "\n"
        )

    def _build_written_result(
        self,
        *,
        post_id: str,
        text: str,
        support,
        reason: str,
    ) -> EpisodeResult:
        reference_urls = dedupe_support_urls(support)
        note = render_note_text_with_urls(text, reference_urls)
        note_length = effective_note_length(text, reference_urls)

        if note_length > self.max_platform_chars:
            raise EpisodeRunnerError(
                f"Final note exceeds platform limit: effective_length={note_length}, "
                f"max_allowed={self.max_platform_chars}."
            )

        final_note = FinalNote(
            post_id=post_id,
            text=text,
            reference_urls=reference_urls,
            note=note,
            support=support,
            reason=reason,
            effective_length=note_length,
        )
        return EpisodeResult(
            post_id=post_id,
            status="written",
            final_note=final_note,
            abstain=None,
        )

    def _build_abstained_result(
        self,
        *,
        post_id: str,
        reason: str,
    ) -> EpisodeResult:
        abstain = AbstainDecision(
            post_id=post_id,
            reason=reason,
        )
        return EpisodeResult(
            post_id=post_id,
            status="abstained",
            final_note=None,
            abstain=abstain,
        )

    def run(
        self,
        post: PostPackage,
        *,
        created_at_ms: Optional[int] = None,
        planner_memory: Optional[Sequence[str]] = None,
        actor_memory: Optional[Sequence[str]] = None,
    ) -> EpisodeRun:
        post_url_aliases = build_post_url_aliases(
            post,
            timeout_s=self.post_url_resolve_timeout_s,
        )
        self._emit("POST URL ALIASES", post_url_aliases)

        planner_run = self.planner.plan_run(
            post=post,
            planner_memory=planner_memory,
        )

        self._emit(
            "PLANNER RUBRICS",
            planner_run.rubrics.model_dump(),
        )

        history = CompressedHistory()

        trace = EpisodeTrace(
            episode_id=post.post_id,
            post_id=post.post_id,
            planner_prompt=planner_run.prompt,
            planner_raw_output=planner_run.raw_output,
            planner_parsed_output=planner_run.rubrics.model_dump(),
            steps=[],
            final_output={},
        )

        used_searches = 0
        used_visits = 0
        step_idx = 1

        while step_idx <= self.max_steps:
            budget_state = self._budget_state(
                step_idx=step_idx,
                used_searches=used_searches,
                used_visits=used_visits,
            )

            available_actions = list(budget_state["available_actions"])
            self._emit_step_header(step_idx, budget_state)

            current_global_rubrics = (
                list(self.global_rubrics)
                + build_action_space_rubrics(available_actions)
            )
            self._emit("CURRENT GLOBAL RUBRICS", current_global_rubrics)

            actor_run = self.actor.act_run(
                post=post,
                global_rubrics=current_global_rubrics,
                instance_rubrics=planner_run.rubrics,
                history=history,
                budget_state=budget_state,
                actor_memory=actor_memory,
                post_url_aliases=post_url_aliases,
                available_actions=available_actions,
            )

            self._emit(
                "ACTOR DECISION",
                actor_run.decision.model_dump(),
            )

            action = actor_run.decision.action
            history_before = history.model_dump()

            if action.action == "search":
                if used_searches >= self.max_searches:
                    raise EpisodeRunnerError(
                        f"Search budget exhausted at step {step_idx}."
                    )

                observation = self.search_engine.search(
                    action.query,
                    created_at_ms=created_at_ms,
                )
                self._emit(
                    "SEARCH RESULTS",
                    observation.model_dump(),
                )

                history = self.history_builder.update_with_search(
                    history=history,
                    query=action.query,
                    observation=observation,
                )
                self._emit(
                    "UPDATED HISTORY AFTER SEARCH",
                    history.model_dump(),
                )

                used_searches += 1

                trace.steps.append(
                    StepTrace(
                        step_idx=step_idx,
                        actor_prompt=actor_run.prompt,
                        actor_raw_output=actor_run.raw_output,
                        parsed_action=actor_run.decision.model_dump(),
                        tool_name="search_engine",
                        tool_input={
                            "query": action.query,
                            "created_at_ms": created_at_ms,
                        },
                        tool_output=observation.model_dump(),
                        history_before=history_before,
                        history_after=history.model_dump(),
                    )
                )
                step_idx += 1
                continue

            if action.action == "visit":
                if used_visits >= self.max_visits:
                    raise EpisodeRunnerError(
                        f"Visit budget exhausted at step {step_idx}."
                    )

                scrape_result = self.crawl_engine.crawl(action.url)
                self._emit(
                    "CRAWLED PAGE METADATA",
                    {
                        "url": scrape_result.url,
                        # "metadata": scrape_result.metadata,
                    },
                )

                observation = self.summary_model.summarize(
                    scrape_result=scrape_result,
                    goal=action.goal,
                )
                self._emit(
                    "VISIT SUMMARY",
                    observation.model_dump(),
                )

                history = self.history_builder.update_with_visit(
                    history=history,
                    observation=observation,
                )
                self._emit(
                    "UPDATED HISTORY AFTER VISIT",
                    history.model_dump(),
                )

                used_visits += 1

                trace.steps.append(
                    StepTrace(
                        step_idx=step_idx,
                        actor_prompt=actor_run.prompt,
                        actor_raw_output=actor_run.raw_output,
                        parsed_action=actor_run.decision.model_dump(),
                        tool_name="visit_pipeline",
                        tool_input={
                            "url": action.url,
                            "goal": action.goal,
                        },
                        tool_output={
                            "scrape": {
                                "url": scrape_result.url,
                                "metadata": scrape_result.metadata,
                            },
                            "visit_observation": observation.model_dump(),
                        },
                        history_before=history_before,
                        history_after=history.model_dump(),
                    )
                )
                step_idx += 1
                continue

            if action.action == "write":
                final_text = action.text
                final_reason = action.reason
                support_payload = [item.model_dump() for item in action.support]

                refinement_attempt = 0
                refinement_failures: list[dict[str, Any]] = []

                while (
                    len(final_text) > self.max_text_chars
                    and refinement_attempt < self.max_write_refinement_attempts
                ):
                    refinement_attempt += 1

                    current_failure_record = {
                        "attempt": refinement_attempt,
                        "failure_type": "text_too_long",
                        "text_length": len(final_text),
                        "max_text_chars": self.max_text_chars,
                        "text": final_text,
                        "reason": final_reason,
                    }
                    refinement_failures.append(current_failure_record)

                    self._emit(
                        "WRITE TEXT TOO LONG - REFINEMENT TRIGGERED",
                        current_failure_record,
                    )

                    try:
                        refinement_run = self.actor.refine_write_run(
                            original_text=final_text,
                            support=support_payload,
                            max_text_chars=self.max_text_chars,
                            previous_failures=refinement_failures,
                        )

                        self._emit(
                            "WRITE REFINEMENT OUTPUT",
                            {
                                "text": refinement_run.output.text,
                                "text_length": len(refinement_run.output.text),
                                "reason": refinement_run.output.reason,
                            },
                        )

                        trace.steps.append(
                            StepTrace(
                                step_idx=step_idx,
                                actor_prompt=refinement_run.prompt,
                                actor_raw_output=refinement_run.raw_output,
                                parsed_action={
                                    "action": "write_refinement",
                                    "text": refinement_run.output.text,
                                    "reason": refinement_run.output.reason,
                                },
                                tool_name="write_refinement",
                                tool_input={
                                    "original_text": final_text,
                                    "max_text_chars": self.max_text_chars,
                                    "support": support_payload,
                                    "previous_failures": refinement_failures,
                                },
                                tool_output={
                                    "refined_text": refinement_run.output.text,
                                    "reason": refinement_run.output.reason,
                                    "text_length": len(refinement_run.output.text),
                                },
                                history_before=history_before,
                                history_after=history_before,
                            )
                        )

                        final_text = refinement_run.output.text
                        final_reason = refinement_run.output.reason

                    except Exception as exc:
                        error_record = {
                            "attempt": refinement_attempt,
                            "failure_type": "refinement_error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "text_length": len(final_text),
                            "max_text_chars": self.max_text_chars,
                            "text": final_text,
                        }
                        refinement_failures.append(error_record)

                        self._emit(
                            "WRITE REFINEMENT ERROR",
                            error_record,
                        )

                        trace.steps.append(
                            StepTrace(
                                step_idx=step_idx,
                                actor_prompt="write_refinement_error",
                                actor_raw_output=str(exc),
                                parsed_action={
                                    "action": "write_refinement_error",
                                    "error": f"{type(exc).__name__}: {exc}",
                                },
                                tool_name="write_refinement",
                                tool_input={
                                    "original_text": final_text,
                                    "max_text_chars": self.max_text_chars,
                                    "support": support_payload,
                                    "previous_failures": refinement_failures,
                                },
                                tool_output={
                                    "error": f"{type(exc).__name__}: {exc}",
                                },
                                history_before=history_before,
                                history_after=history_before,
                            )
                        )

                if len(final_text) > self.max_text_chars:
                    result = self._build_abstained_result(
                        post_id=post.post_id,
                        reason=(
                            "Unable to compress the note text to fit the text length limit "
                            "after focused refinement."
                        ),
                    )
                    self._emit(
                        "FINAL ABSTAIN RESULT",
                        result.model_dump(),
                    )

                    trace.steps.append(
                        StepTrace(
                            step_idx=step_idx,
                            actor_prompt=actor_run.prompt,
                            actor_raw_output=actor_run.raw_output,
                            parsed_action=actor_run.decision.model_dump(),
                            tool_name=None,
                            tool_input=None,
                            tool_output=None,
                            history_before=history_before,
                            history_after=history_before,
                        )
                    )
                    trace.final_output = result.model_dump()

                    return EpisodeRun(
                        result=result,
                        trace=trace,
                        final_history=history,
                    )

                result = self._build_written_result(
                    post_id=post.post_id,
                    text=final_text,
                    support=action.support,
                    reason=final_reason,
                )
                self._emit(
                    "FINAL WRITTEN RESULT",
                    result.model_dump(),
                )

                trace.steps.append(
                    StepTrace(
                        step_idx=step_idx,
                        actor_prompt=actor_run.prompt,
                        actor_raw_output=actor_run.raw_output,
                        parsed_action=actor_run.decision.model_dump(),
                        tool_name=None,
                        tool_input=None,
                        tool_output=None,
                        history_before=history_before,
                        history_after=history_before,
                    )
                )
                trace.final_output = result.model_dump()

                return EpisodeRun(
                    result=result,
                    trace=trace,
                    final_history=history,
                )

            if action.action == "abstain":
                result = self._build_abstained_result(
                    post_id=post.post_id,
                    reason=action.reason,
                )

                self._emit(
                    "FINAL ABSTAIN RESULT",
                    result.model_dump(),
                )

                trace.steps.append(
                    StepTrace(
                        step_idx=step_idx,
                        actor_prompt=actor_run.prompt,
                        actor_raw_output=actor_run.raw_output,
                        parsed_action=actor_run.decision.model_dump(),
                        tool_name=None,
                        tool_input=None,
                        tool_output=None,
                        history_before=history_before,
                        history_after=history_before,
                    )
                )
                trace.final_output = result.model_dump()

                return EpisodeRun(
                    result=result,
                    trace=trace,
                    final_history=history,
                )

            raise EpisodeRunnerError(
                f"Unsupported action encountered: {action.action}"
            )

        result = self._build_abstained_result(
            post_id=post.post_id,
            reason="Maximum step budget reached before a final decision was made.",
        )

        self._emit(
            "FINAL ABSTAIN RESULT",
            result.model_dump(),
        )

        trace.final_output = result.model_dump()

        return EpisodeRun(
            result=result,
            trace=trace,
            final_history=history,
        )