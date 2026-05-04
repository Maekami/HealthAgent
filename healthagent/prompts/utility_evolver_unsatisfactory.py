from __future__ import annotations

import json

from healthagent.schemas import PostPackage
from healthagent.schemas.evolver import (
    UtilityEvolverUnsatisfactoryOutput,
    UtilityModeJudgeOutput,
)

from .utility_evolver_common import episode_to_prompt_dict


def build_utility_evolver_unsatisfactory_system_prompt() -> str:
    return (
        "You are a routed utility evolver for a self-improving evidence-backed note-writing system.\n"
        "This episode has been classified as unsatisfactory because the planner did not complete its task.\n"
        "Your job is to focus only on planner-side improvements.\n"
        "Do not derive actor guidance from this episode, because the task framing itself was insufficient.\n"
        "Return valid JSON only."
    )


def build_utility_evolver_unsatisfactory_user_prompt(
    *,
    post: PostPackage,
    episode_run,
    judge_output: UtilityModeJudgeOutput,
) -> str:
    output_schema = UtilityEvolverUnsatisfactoryOutput.model_json_schema()

    return (
        "Analyze this completed episode and extract planner-oriented lessons.\n\n"
        "Requirements:\n"
        "- Focus only on planner-side failure modes and improvements.\n"
        "- planner_memory_items must contain at least one reusable planning lesson.\n"
        "- Do not derive actor guidance from this episode.\n"
        "- Prefer high-impact, reusable lessons over exhaustive criticism.\n"
        "- Each list item must be plain text with no leading bullets, numbering, or punctuation.\n"
        "- Return JSON only, with no markdown fence and no extra commentary.\n\n"
        f"Mode judgment:\n{json.dumps(judge_output.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"Episode data:\n{json.dumps(episode_to_prompt_dict(post, episode_run), ensure_ascii=False, indent=2)}\n\n"
        f"Output schema:\n{json.dumps(output_schema, ensure_ascii=False, indent=2)}"
    )