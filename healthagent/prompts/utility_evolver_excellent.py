from __future__ import annotations

import json

from healthagent.schemas import PostPackage
from healthagent.schemas.evolver import (
    UtilityEvolverExcellentOutput,
    UtilityModeJudgeOutput,
)

from .utility_evolver_common import episode_to_prompt_dict


def build_utility_evolver_excellent_system_prompt() -> str:
    return (
        "You are a routed utility evolver for a self-improving evidence-backed note-writing system.\n"
        "This episode has been classified as excellent: the planner succeeded, the actor fully completed the planner-defined task, "
        "and there is no clear high-value perceived-social-utility improvement left.\n"
        "Your job is to extract successful, reusable patterns from the episode.\n"
        "Do not default to criticism. Focus on what worked and why it would generalize.\n"
        "Return valid JSON only."
    )


def build_utility_evolver_excellent_user_prompt(
    *,
    post: PostPackage,
    episode_run,
    judge_output: UtilityModeJudgeOutput,
) -> str:
    output_schema = UtilityEvolverExcellentOutput.model_json_schema()

    return (
        "Analyze this completed episode and extract reusable lessons.\n\n"
        "Requirements:\n"
        "- Treat this as a genuinely strong episode, not merely a minimally acceptable one.\n"
        "- Focus on successful patterns that made the planner framing and actor execution work well together.\n"
        "- planner_memory_items should capture reusable planning lessons.\n"
        "- actor_memory_items should capture reusable execution lessons.\n"
        "- At least one memory item must be produced in total.\n"
        "- Prefer 1-3 strong items over filling every slot.\n"
        "- Each list item must be plain text with no leading bullets, numbering, or punctuation.\n"
        "- Return JSON only, with no markdown fence and no extra commentary.\n\n"
        f"Mode judgment:\n{json.dumps(judge_output.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"Episode data:\n{json.dumps(episode_to_prompt_dict(post, episode_run), ensure_ascii=False, indent=2)}\n\n"
        f"Output schema:\n{json.dumps(output_schema, ensure_ascii=False, indent=2)}"
    )