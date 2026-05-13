from __future__ import annotations

import json

from healthagent.schemas import PostPackage
from healthagent.schemas.evolver import (
    UtilityEvolverSatisfactoryOutput,
    UtilityModeJudgeOutput,
)

from .utility_evolver_common import episode_to_prompt_dict


GOOD_TRIGGER_EXAMPLES = [
    "Post uses a recent single health incident to imply a broader causal or population-level conclusion",
    "Post uses a linked study or article to support a stronger health claim than the source directly justifies",
    "Post makes a claim about the cause or nature of a health condition that could benefit from practical guidance",
]

BAD_TRIGGER_EXAMPLES = [
    "Note lacks depth",
    "Note lacks practical implications",
    "Note does not offer safe next steps",
]

GOOD_INCOMPLETE_RULE_EXAMPLES = [
    "Before finalizing a note for posts like this, verify whether the post is generalizing from a single incident to a broader causal or population-level conclusion.",
    "For posts like this, do not finalize after checking only the headline claim; also verify whether the linked source supports the broader health conclusion implied by the post.",
]

GOOD_COMPLETE_RULE_EXAMPLES = [
    "Before finalizing a minimally correct note for posts like this, add one short sentence explaining why the corrected evidence matters for risk interpretation or real-world decision-making.",
    "Before finalizing a minimally correct note for posts like this, consider adding a brief, general recommendation to consult a qualified healthcare professional or a trusted public-health source when personal medical decisions could be affected.",
    "If the core claim has already been corrected, consider adding brief calibrated uncertainty or limits of evidence so the note is more trustworthy and informative.",
]

BAD_RULE_EXAMPLES = [
    "If the note corrects the claim, explain it better.",
    "Add more nuance.",
    "The note should be deeper.",
    "The note lacks practical implications.",
]


def build_utility_evolver_satisfactory_system_prompt() -> str:
    return (
        "You are a routed utility evolver for a self-improving evidence-backed note-writing system.\n"
        "This episode has been classified as satisfactory.\n"
        "Focus mainly on reusable actor-side improvements, while preserving planner-side strengths only when worth keeping.\n"
        "Trigger must identify a recurring post type or claim archetype, not a weakness of the current note.\n"
        "Rule must be actionable for a future actor before finalizing a note.\n"
        "Avoid generic writing advice and avoid entity-specific advice.\n"
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
        "Actor memory requirements:\n"
        "- actor_memory_items must include at least one item with stage = core_task_incomplete\n"
        "- actor_memory_items must include at least one item with stage = core_task_complete\n"
        "- core_task_incomplete items should help a future actor reach a minimally good note\n"
        "- core_task_complete items should help a future actor improve a minimally good note toward an excellent note before finalizing\n\n"
        "Rules for trigger:\n"
        "- trigger must identify a recurring post type or claim archetype\n"
        "- trigger should usually begin with 'Post ...'\n"
        "- trigger must not describe a weakness of the current note\n\n"
        "Rules for rule:\n"
        "- rule must be actionable for a future actor before finalizing a note\n"
        "- do not write rule as commentary on an already-written note\n"
        "- prefer rules that can change a future search, visit, or write decision\n"
        "- for core_task_complete, prefer one more high-value improvement step before finalizing\n"
        "- this often means adding practical implication, safe next step, calibrated uncertainty, or one more complementary evidence step\n\n"
        "Rules for why:\n"
        "- explain why that rule helps for that post type\n"
        "- keep it general and reusable\n\n"
        "Good trigger examples:\n"
        f"{json.dumps(GOOD_TRIGGER_EXAMPLES, ensure_ascii=False, indent=2)}\n\n"
        "Bad trigger examples:\n"
        f"{json.dumps(BAD_TRIGGER_EXAMPLES, ensure_ascii=False, indent=2)}\n\n"
        "Good core_task_incomplete rule examples:\n"
        f"{json.dumps(GOOD_INCOMPLETE_RULE_EXAMPLES, ensure_ascii=False, indent=2)}\n\n"
        "Good core_task_complete rule examples:\n"
        f"{json.dumps(GOOD_COMPLETE_RULE_EXAMPLES, ensure_ascii=False, indent=2)}\n\n"
        "Bad rule examples:\n"
        f"{json.dumps(BAD_RULE_EXAMPLES, ensure_ascii=False, indent=2)}\n\n"
        "Other constraints:\n"
        "- avoid universal advice like 'use simple language'\n"
        "- avoid entity-specific advice like 'for Covishield posts'\n"
        "- avoid sample-specific names unless absolutely necessary\n"
        "- planner_memory_items are optional\n"
        "- prefer high-impact, reusable lessons over exhaustive analysis\n"
        "- Return JSON only, with no markdown fence and no extra commentary.\n\n"
        f"Mode judgment:\n{json.dumps(judge_output.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"Episode data:\n{json.dumps(episode_to_prompt_dict(post, episode_run), ensure_ascii=False, indent=2)}\n\n"
        f"Output schema:\n{json.dumps(output_schema, ensure_ascii=False, indent=2)}"
    )