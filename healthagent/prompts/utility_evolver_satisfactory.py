from __future__ import annotations

import json

from healthagent.schemas import PostPackage
from healthagent.schemas.evolver import (
    UtilityEvolverSatisfactoryOutput,
    UtilityModeJudgeOutput,
)

from .utility_evolver_common import episode_to_prompt_dict


def build_utility_evolver_satisfactory_system_prompt() -> str:
    return (
        "You are a routed utility evolver for a self-improving evidence-backed note-writing system.\n"
        "This episode has been classified as satisfactory: the planner succeeded, but the actor did not fully complete the task at an excellent standard.\n"
        "This may mean that the actor did not fully complete the planner-defined task, or that the final note was minimally acceptable but still had clear room "
        "for improving perceived social utility.\n"
        "Your job is to focus mainly on actor-side improvements while preserving any planner-side strengths worth reusing.\n"
        "Return valid JSON only."
    )


def build_utility_evolver_satisfactory_user_prompt(
    *,
    post: PostPackage,
    episode_run,
    judge_output: UtilityModeJudgeOutput,
) -> str:
    output_schema = UtilityEvolverSatisfactoryOutput.model_json_schema()

    return (
        "Analyze this completed episode and extract improvement-oriented lessons.\n\n"
        "Requirements:\n"
        "- Focus mainly on what the actor should do better next time.\n"
        "- Treat both of the following as satisfactory cases:\n"
        "  1) the actor did not fully complete the planner-defined task;\n"
        "  2) the actor produced a minimally acceptable note, but there was still clear room to improve perceived social utility.\n"
        "- actor_memory_items must contain at least one reusable actor-side lesson.\n"
        "- planner_memory_items are optional and should only be included if there is something worth preserving.\n"
        "- Prefer high-impact, reusable lessons over exhaustive analysis.\n"
        "- Each list item must be plain text with no leading bullets, numbering, or punctuation.\n"
        "- Return JSON only, with no markdown fence and no extra commentary.\n\n"
        f"Mode judgment:\n{json.dumps(judge_output.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"Episode data:\n{json.dumps(episode_to_prompt_dict(post, episode_run), ensure_ascii=False, indent=2)}\n\n"
        f"Output schema:\n{json.dumps(output_schema, ensure_ascii=False, indent=2)}"
    )