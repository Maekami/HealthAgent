from __future__ import annotations

import json

from healthagent.schemas import PostPackage
from healthagent.schemas.evolver import (
    UtilityEvolverUnsatisfactoryOutput,
    UtilityModeJudgeOutput,
)

from .utility_evolver_common import episode_to_prompt_dict


GOOD_TRIGGER_EXAMPLES = [
    "post attributes symptoms of a condition solely to modern life or environmental factors, denying an underlying biological basis",
    "post makes a strong health claim based on a linked study or article",
    "post uses a vivid health example to imply a broader unsupported conclusion",
]

BAD_TRIGGER_EXAMPLES = [
    "planner missed the right claim",
    "note is not good enough",
    "note lacks nuance",
]


def build_utility_evolver_unsatisfactory_system_prompt() -> str:
    return (
        "You are a routed utility evolver for a self-improving evidence-backed note-writing system.\n"
        "This episode has been classified as unsatisfactory because the planner did not complete its task.\n"
        "Focus only on planner-side improvements.\n"
        "Trigger must identify a recurring post type or claim archetype, not a weakness of the current note.\n"
        "Avoid generic writing advice and avoid entity-specific advice.\n"
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
        "For each memory item, output fields in this order:\n"
        "1. trigger\n"
        "2. stage\n"
        "3. rule\n"
        "4. why\n\n"
        "Rules:\n"
        "- trigger must identify a recurring post type or claim archetype\n"
        "- trigger should usually begin with 'post ...'\n"
        "- trigger must not describe a weakness of the current note\n"
        "- stage must be core_task_incomplete\n"
        "- rule should say what the planner should do for that post type\n"
        "- why should explain why that rule helps for that post type\n"
        "- avoid universal advice like 'identify the right claim'\n"
        "- avoid entity-specific advice like 'for Covishield posts'\n"
        "- avoid sample-specific names unless absolutely necessary\n\n"
        "Good trigger examples:\n"
        f"{json.dumps(GOOD_TRIGGER_EXAMPLES, ensure_ascii=False, indent=2)}\n\n"
        "Bad trigger examples:\n"
        f"{json.dumps(BAD_TRIGGER_EXAMPLES, ensure_ascii=False, indent=2)}\n\n"
        "Requirements:\n"
        "- planner_memory_items must contain at least one item\n"
        "- do not derive actor guidance from this episode\n"
        "- Return JSON only, with no markdown fence and no extra commentary.\n\n"
        f"Mode judgment:\n{json.dumps(judge_output.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"Episode data:\n{json.dumps(episode_to_prompt_dict(post, episode_run), ensure_ascii=False, indent=2)}\n\n"
        f"Output schema:\n{json.dumps(output_schema, ensure_ascii=False, indent=2)}"
    )