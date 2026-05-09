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
        "Focus only on planner-side improvements.\n"
        "Avoid generic writing advice and avoid entity-specific advice.\n"
        "Each memory item should apply to a recurring class of posts or evidence situations.\n"
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
        "Produce reusable planner memory from this episode.\n\n"
        "For each memory item:\n"
        "- trigger: say when this memory should apply\n"
        "- rule: say what the planner should do\n"
        "- why: say why this helps\n"
        "- avoid universal advice like 'identify the right claim'\n"
        "- avoid entity-specific advice like 'for Covishield posts'\n"
        "- target a recurring class of health misinformation posts or evidence situations\n\n"
        "Requirements:\n"
        "- planner_memory_items must contain at least one item\n"
        "- do not derive actor guidance from this episode\n"
        "- Return JSON only, with no markdown fence and no extra commentary.\n\n"
        f"Mode judgment:\n{json.dumps(judge_output.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"Episode data:\n{json.dumps(episode_to_prompt_dict(post, episode_run), ensure_ascii=False, indent=2)}\n\n"
        f"Output schema:\n{json.dumps(output_schema, ensure_ascii=False, indent=2)}"
    )