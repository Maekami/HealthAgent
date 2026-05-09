from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from healthagent.components.actor import Actor
from healthagent.components.history_builder import HistoryBuilder
from healthagent.components.memory_query_builder import MemoryQueryBuilder
from healthagent.components.rubric_planner import RubricPlanner
from healthagent.memory import (
    BaseActorMemory,
    BasePlannerMemory,
    EmptyActorMemory,
    EmptyPlannerMemory,
)
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
from .write_refinement import run_write_refinement
from healthagent.utils.debug_sink import DebugSink


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
        planner_memory_retriever: BasePlannerMemory | None = None,
        actor_memory_retriever: BaseActorMemory | None = None,
        memory_query_builder: MemoryQueryBuilder | None = None,
        max_steps: int = 6,
        max_searches: int = 3,
        max_visits: int = 4,
        min_searches_before_finalize: int = 1,
        min_visits_before_finalize: int = 1,
        max_text_chars: int = 270,
        max_platform_chars: int = 280,
        max_write_refinement_attempts: int = 5,
        stream_print: bool = False,
        stream_printer: Optional[Callable[[str], None]] = None,
        post_url_resolve_timeout_s: float = 10.0,
        debug_sink: DebugSink | None = None,
        sample_key: str | None = None,
    ) -> None:
        self.planner = planner
        self.actor = actor
        self.history_builder = history_builder
        self.search_engine = search_engine
        self.crawl_engine = crawl_engine
        self.summary_model = summary_model
        self.global_rubrics = list(global_rubrics)

        self.planner_memory_retriever = planner_memory_retriever or EmptyPlannerMemory()
        self.actor_memory_retriever = actor_memory_retriever or EmptyActorMemory()
        self.memory_query_builder = memory_query_builder

        self.max_steps = max_steps
        self.max_searches = max_searches
        self.max_visits = max_visits
        self.min_searches_before_finalize = min_searches_before_finalize
        self.min_visits_before_finalize = min_visits_before_finalize
        self.max_text_chars = max_text_chars
        self.max_platform_chars = max_platform_chars
        self.max_write_refinement_attempts = max_write_refinement_attempts
        self.stream_print = stream_print
        self.stream_printer = stream_printer or print
        self.post_url_resolve_timeout_s = post_url_resolve_timeout_s
        
        self.debug_sink = debug_sink
        self.sample_key = sample_key

    def _finalize_gate_open(
        self,
        *,
        used_searches: int,
        used_visits: int,
    ) -> bool:
        return (
            used_searches >= self.min_searches_before_finalize
            and used_visits >= self.min_visits_before_finalize
        )

    def _available_actions(
        self,
        *,
        step_idx: int,
        used_searches: int,
        used_visits: int,
    ) -> list[str]:
        search_visit_actions: list[str] = []

        if used_searches < self.max_searches:
            search_visit_actions.append("search")
        if used_visits < self.max_visits:
            search_visit_actions.append("visit")

        finalize_gate_open = self._finalize_gate_open(
            used_searches=used_searches,
            used_visits=used_visits,
        )
        is_last_step = step_idx >= self.max_steps

        if finalize_gate_open:
            if is_last_step:
                return ["write", "abstain"]
            return search_visit_actions + ["write", "abstain"]

        return search_visit_actions

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

        finalize_gate_open = self._finalize_gate_open(
            used_searches=used_searches,
            used_visits=used_visits,
        )

        return {
            "current_step": step_idx,
            "max_steps": self.max_steps,
            "remaining_steps": max(self.max_steps - step_idx + 1, 0),
            "used_searches": used_searches,
            "min_searches_before_finalize": self.min_searches_before_finalize,
            "remaining_searches": max(self.max_searches - used_searches, 0),
            "used_visits": used_visits,
            "min_visits_before_finalize": self.min_visits_before_finalize,
            "remaining_visits": max(self.max_visits - used_visits, 0),
            "available_actions": available_actions,
            "finalize_gate_open": finalize_gate_open,
            "is_last_step": step_idx >= self.max_steps,
        }

    def _format_emit_body(self, payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _emit(self, title: str, payload: Any) -> None:
        body = self._format_emit_body(payload)

        if self.debug_sink is not None:
            self.debug_sink.emit(
                title=title,
                payload=payload,   # Original structured data, suitable for saving in JSONL format
                rendered=body,     # Formatted text, convenient for saving as txt
                sample_key=self.sample_key,
                stage="episode",
            )

        if not self.stream_print:
            return

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

    def _resolve_planner_memory(
        self,
        *,
        post: PostPackage,
        planner_memory: Sequence[str] | None,
    ) -> list[str]:
        if planner_memory is not None:
            resolved = list(planner_memory)
            self._emit("PLANNER MEMORY OVERRIDE", resolved)
            return resolved

        query: str | None = None
        if self.memory_query_builder is not None:
            query_run = self.memory_query_builder.build_planner_query_run(
                post=post,
            )
            self._emit("PLANNER MEMORY QUERY", query_run.output.model_dump())
            query = query_run.output.query

        resolved = self.planner_memory_retriever.retrieve(
            post=post,
            query=query,
        )
        self._emit("RESOLVED PLANNER MEMORY", resolved)
        return resolved

    def _resolve_actor_memory(
        self,
        *,
        post: PostPackage,
        history: CompressedHistory,
        instance_rubrics,
        budget_state: dict[str, Any],
        actor_memory: Sequence[str] | None,
    ) -> list[str]:
        if actor_memory is not None:
            resolved = list(actor_memory)
            self._emit("ACTOR MEMORY OVERRIDE", resolved)
            return resolved

        query: str | None = None
        if self.memory_query_builder is not None:
            query_run = self.memory_query_builder.build_actor_query_run(
                post=post,
                instance_rubrics=instance_rubrics,
                history=history,
                budget_state=budget_state,
            )
            self._emit("ACTOR MEMORY QUERY", query_run.output.model_dump())
            query = query_run.output.query

        resolved = self.actor_memory_retriever.retrieve(
            post=post,
            history=history,
            instance_rubrics=instance_rubrics,
            query=query,
        )
        self._emit("RESOLVED ACTOR MEMORY", resolved)
        return resolved

    def _refinement_max_text_chars_from_support(
        self,
        support: Sequence[Any],
    ) -> int:
        """
        When refinement is triggered, dynamically relax/tighten the text budget
        based on the number of unique reference URLs in the initial note.

        Formula:
            max_text_chars = max_platform_chars - url_count
        """
        unique_urls: set[str] = set()

        for item in support:
            url = None
            if isinstance(item, dict):
                url = item.get("url")
            else:
                url = getattr(item, "url", None)

            if isinstance(url, str):
                url = url.strip()
                if url:
                    unique_urls.add(url)

        dynamic_limit = self.max_platform_chars - len(unique_urls)

        # Safety clamp: refinement limit should never be negative, and should not
        # exceed the platform limit.
        return max(0, min(dynamic_limit, self.max_platform_chars))

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
        
        self._emit(
            "INPUT POST",
            post.model_dump(),
        )       

        post_url_aliases = build_post_url_aliases(
            post,
            timeout_s=self.post_url_resolve_timeout_s,
        )
        self._emit("POST URL ALIASES", post_url_aliases)

        resolved_planner_memory = self._resolve_planner_memory(
            post=post,
            planner_memory=planner_memory,
        )

        planner_run = self.planner.plan_run(
            post=post,
            planner_memory=resolved_planner_memory,
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

            if not available_actions:
                result = self._build_abstained_result(
                    post_id=post.post_id,
                    reason=(
                        "No available actions remained before the finalize gate opened. "
                        f"Completed searches: {used_searches}/{self.min_searches_before_finalize}, "
                        f"completed visits: {used_visits}/{self.min_visits_before_finalize}."
                    ),
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

            current_global_rubrics = (
                list(self.global_rubrics)
                + build_action_space_rubrics(available_actions)
            )
            self._emit("CURRENT GLOBAL RUBRICS", current_global_rubrics)

            resolved_actor_memory = self._resolve_actor_memory(
                post=post,
                history=history,
                instance_rubrics=planner_run.rubrics,
                budget_state=budget_state,
                actor_memory=actor_memory,
            )

            actor_run = self.actor.act_run(
                post=post,
                global_rubrics=current_global_rubrics,
                instance_rubrics=planner_run.rubrics,
                history=history,
                budget_state=budget_state,
                actor_memory=resolved_actor_memory,
                post_url_aliases=post_url_aliases,
                available_actions=available_actions,
            )

            self._emit(
                "ACTOR DECISION",
                actor_run.decision.model_dump(),
            )

            action = actor_run.decision.action
            history_before = history.model_dump()

            # FIXME:
            ################################## search ##################################
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

            # FIXME:
            ################################## visit ##################################
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
                        # "raw_response":scrape_result.raw_response,
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

            # FIXME:
            ################################## write ##################################
            if action.action == "write":
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

                support_payload = [item.model_dump() for item in action.support]
                refinement_max_text_chars = self._refinement_max_text_chars_from_support(
                    action.support
                )

                self._emit(
                    "WRITE REFINEMENT BUDGET",
                    {
                        "default_max_text_chars": self.max_text_chars,
                        "max_platform_chars": self.max_platform_chars,
                        "reference_url_count": len({item.url for item in action.support}),
                        "applied_refinement_max_text_chars": refinement_max_text_chars,
                    },
                )

                refinement_outcome = run_write_refinement(
                    actor=self.actor,
                    initial_text=action.text,
                    initial_reason=action.reason,
                    support_payload=support_payload,
                    max_text_chars=refinement_max_text_chars,
                    max_attempts=self.max_write_refinement_attempts,
                    step_idx=step_idx,
                    history_snapshot=history_before,
                    emit=self._emit,
                )

                trace.steps.extend(refinement_outcome.trace_steps)

                if not refinement_outcome.success:
                    raise EpisodeRunnerError("Unable to compress the note text")
                
                    result = self._build_abstained_result(
                        post_id=post.post_id,
                        reason=(
                            "Unable to compress the note text to fit the refinement text length limit "
                            f"({refinement_outcome.applied_max_text_chars}) after focused refinement."
                        ),
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

                result = self._build_written_result(
                    post_id=post.post_id,
                    text=refinement_outcome.final_text,
                    support=action.support,
                    reason=refinement_outcome.final_reason,
                )
                self._emit(
                    "FINAL WRITTEN RESULT",
                    result.model_dump(),
                )
                trace.final_output = result.model_dump()

                return EpisodeRun(
                    result=result,
                    trace=trace,
                    final_history=history,
                )

            # FIXME:
            ################################## abstain ##################################
            if action.action == "abstain":
                # FIXME:
                raise EpisodeRunnerError("Call abstain")

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