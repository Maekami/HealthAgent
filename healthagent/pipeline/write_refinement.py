from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, TYPE_CHECKING

from healthagent.schemas import StepTrace

if TYPE_CHECKING:
    from healthagent.components.actor import Actor


@dataclass(slots=True)
class WriteRefinementOutcome:
    final_text: str
    final_reason: str
    success: bool
    trace_steps: list[StepTrace]
    failure_records: list[dict[str, Any]]


def _noop_emit(title: str, payload: Any) -> None:
    return


def run_write_refinement(
    *,
    actor: "Actor",
    initial_text: str,
    initial_reason: str,
    support_payload: Sequence[dict[str, Any]],
    max_text_chars: int,
    max_attempts: int,
    step_idx: int,
    history_snapshot: dict[str, Any],
    emit: Optional[Callable[[str, Any], None]] = None,
) -> WriteRefinementOutcome:
    emit = emit or _noop_emit

    final_text = initial_text
    final_reason = initial_reason
    refinement_attempt = 0
    refinement_failures: list[dict[str, Any]] = []
    trace_steps: list[StepTrace] = []

    while len(final_text) > max_text_chars and refinement_attempt < max_attempts:
        refinement_attempt += 1

        current_failure_record = {
            "attempt": refinement_attempt,
            "failure_type": "text_too_long",
            "text_length": len(final_text),
            "max_text_chars": max_text_chars,
            "text": final_text,
            "reason": final_reason,
        }
        refinement_failures.append(current_failure_record)

        emit("WRITE TEXT TOO LONG - REFINEMENT TRIGGERED", current_failure_record)

        try:
            refinement_run = actor.refine_write_run(
                original_text=final_text,
                support=support_payload,
                max_text_chars=max_text_chars,
                previous_failures=refinement_failures,
            )

            refined_payload = {
                "text": refinement_run.output.text,
                "text_length": len(refinement_run.output.text),
                "reason": refinement_run.output.reason,
            }
            emit("WRITE REFINEMENT OUTPUT", refined_payload)

            trace_steps.append(
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
                        "max_text_chars": max_text_chars,
                        "support": list(support_payload),
                        "previous_failures": list(refinement_failures),
                    },
                    tool_output={
                        "refined_text": refinement_run.output.text,
                        "reason": refinement_run.output.reason,
                        "text_length": len(refinement_run.output.text),
                    },
                    history_before=history_snapshot,
                    history_after=history_snapshot,
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
                "max_text_chars": max_text_chars,
                "text": final_text,
            }
            refinement_failures.append(error_record)

            emit("WRITE REFINEMENT ERROR", error_record)

            trace_steps.append(
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
                        "max_text_chars": max_text_chars,
                        "support": list(support_payload),
                        "previous_failures": list(refinement_failures),
                    },
                    tool_output={
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    history_before=history_snapshot,
                    history_after=history_snapshot,
                )
            )

    return WriteRefinementOutcome(
        final_text=final_text,
        final_reason=final_reason,
        success=len(final_text) <= max_text_chars,
        trace_steps=trace_steps,
        failure_records=refinement_failures,
    )