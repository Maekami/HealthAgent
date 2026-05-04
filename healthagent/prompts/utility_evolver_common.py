from __future__ import annotations

from typing import Any

from healthagent.schemas import PostPackage


def _filter_step_for_evolver(step) -> dict[str, Any]:
    filtered: dict[str, Any] = {
        "step_idx": step.step_idx,
        "parsed_action": step.parsed_action,
        "tool_name": step.tool_name,
        "tool_input": step.tool_input,
        "tool_output": step.tool_output,
    }

    # Keep the full trace in storage, but simplify the prompt-facing view.
    if step.tool_name == "visit_pipeline" and isinstance(step.tool_output, dict):
        if "visit_observation" in step.tool_output and isinstance(step.tool_output["visit_observation"], dict):
            filtered["tool_output"] = step.tool_output["visit_observation"]

    return filtered


def episode_to_prompt_dict(
    post: PostPackage,
    episode_run,
) -> dict[str, Any]:
    return {
        "post": post.model_dump(),
        "planner_output": episode_run.trace.planner_parsed_output,
        "steps": [
            _filter_step_for_evolver(step)
            for step in episode_run.trace.steps
        ],
    }