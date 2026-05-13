from __future__ import annotations

import json

from healthagent.schemas import PostPackage
from healthagent.schemas.evolver import (
    UtilityEvolverExcellentOutput,
    UtilityModeJudgeOutput,
)

from .utility_evolver_common import episode_to_prompt_dict


GOOD_TRIGGER_EXAMPLES = [
    "post makes a claim about the cause or nature of a health condition that could benefit from practical guidance",
    "post attributes symptoms of a condition solely to modern life or environmental factors, denying an underlying biological basis",
    "post claims a condition is not a real disorder or is simply a reaction to lifestyle factors",
]

BAD_TRIGGER_EXAMPLES = [
    "note lacks depth or nuance",
    "note lacks practical implications",
    "note does not calibrate uncertainty",
]

GOOD_RULE_EXAMPLES = [
    "Before finalizing a minimally correct note for posts like this, add one short sentence explaining why the corrected evidence matters for risk interpretation or real-world decision-making.",
    "Before finalizing a minimally correct note for posts like this, consider adding a brief, general recommendation to consult a qualified healthcare professional or a trusted public-health source when personal medical decisions could be affected.",
    "If the core claim has already been corrected, consider adding brief calibrated uncertainty or limits of evidence so the note is more trustworthy and informative.",
]

BAD_RULE_EXAMPLES = [
    "If the note corrects the claim, explain it better.",
    "Add more nuance.",
    "The note should be deeper.",
]


def build_utility_evolver_excellent_system_prompt() -> str:
    return (
        "You are a routed utility evolver for a self-improving evidence-backed note-writing system.\n"
        "This episode has been classified as excellent.\n"
        "Extract reusable planner and actor memory.\n"
        "Trigger must identify a recurring post type or claim archetype, not a weakness of the current note.\n"
        "Rule must be actionable for a future actor before finalizing a note.\n"
        "Avoid generic writing advice and avoid entity-specific advice.\n"
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
        "Produce reusable memory from this episode.\n\n"
        "For each memory item, output fields in this order:\n"
        "1. trigger\n"
        "2. stage\n"
        "3. rule\n"
        "4. why\n\n"
        "Stage definitions:\n"
        "- core_task_incomplete: the planner-defined core task has not yet been sufficiently completed for a minimally good note.\n"
        "- core_task_complete: the planner-defined core task has already been sufficiently completed for a minimally good note, and the remaining opportunity is to improve the note toward an excellent note.\n"
        "- Do not use core_task_complete merely because a note is directionally correct.\n"
        "- Use core_task_complete only when the core claims have already been sufficiently handled for a minimally good note.\n\n"
        "Rules:\n"
        "- trigger must identify a recurring post type or claim archetype\n"
        "- trigger should usually begin with 'post ...'\n"
        "- trigger must not describe a weakness of the current note\n"
        "- stage should capture whether this memory is for core_task_incomplete or core_task_complete\n"
        "- planner memory should normally use core_task_incomplete\n"
        "- actor memory in excellent episodes should normally use core_task_complete\n"
        "- rule must be actionable for a future actor before finalizing a note\n"
        "- do not write rule as commentary on an already-written note\n"
        "- for core_task_complete actor memory, prefer one more high-value improvement step before finalizing\n"
        "- why should explain why that rule helps for that post type\n"
        "- avoid universal advice like 'use simple language'\n"
        "- avoid entity-specific advice like 'for Covishield posts'\n"
        "- avoid sample-specific names unless absolutely necessary\n\n"
        "Good trigger examples:\n"
        f"{json.dumps(GOOD_TRIGGER_EXAMPLES, ensure_ascii=False, indent=2)}\n\n"
        "Bad trigger examples:\n"
        f"{json.dumps(BAD_TRIGGER_EXAMPLES, ensure_ascii=False, indent=2)}\n\n"
        "Good rule examples:\n"
        f"{json.dumps(GOOD_RULE_EXAMPLES, ensure_ascii=False, indent=2)}\n\n"
        "Bad rule examples:\n"
        f"{json.dumps(BAD_RULE_EXAMPLES, ensure_ascii=False, indent=2)}\n\n"
        "Requirements:\n"
        "- At least one memory item must be produced in total.\n"
        "- Prefer 1-3 strong items over filling every slot.\n"
        "- Return JSON only, with no markdown fence and no extra commentary.\n\n"
        f"Mode judgment:\n{json.dumps(judge_output.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"Episode data:\n{json.dumps(episode_to_prompt_dict(post, episode_run), ensure_ascii=False, indent=2)}\n\n"
        f"Output schema:\n{json.dumps(output_schema, ensure_ascii=False, indent=2)}"
    )