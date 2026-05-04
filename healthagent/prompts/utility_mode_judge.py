from __future__ import annotations

import json

from healthagent.schemas import PostPackage
from healthagent.schemas.evolver import UtilityModeJudgeOutput

from .utility_evolver_common import episode_to_prompt_dict


def build_utility_mode_judge_system_prompt() -> str:
    return (
        "You are a mode judge for a self-improving evidence-backed health note system.\n"
        "Judge the episode in two steps: first planner, then actor.\n"
        "Use a strict standard for excellent episodes.\n"
        "Do not mention benchmark metrics or evaluation procedures.\n"
        "Return valid JSON only."
    )


def build_utility_mode_judge_user_prompt(
    *,
    post: PostPackage,
    episode_run,
) -> str:
    output_schema = UtilityModeJudgeOutput.model_json_schema()

    return (
        "Decide whether the planner and actor completed their tasks.\n\n"
        "Planner:\n"
        "- planner_ok = true only if the planner identified the main checkworthy claims needed for a minimally good note.\n"
        "- planner_ok = false if it missed core claims or framed the task incorrectly.\n\n"
        "Actor (only if planner_ok = true):\n"
        "- First ask whether the actor completed the planner-defined task and produced a minimally good note.\n"
        "- actor_ok = false if important planner-defined claims remain unsupported at the visited-evidence level, or if the actor stopped too early.\n"
        "- A minimally acceptable note is still not enough for actor_ok = true.\n"
        "- actor_ok = true only if the actor completed the planner-defined task and the note is strong enough to count as excellent.\n\n"
        "Excellent notes go beyond direct correction by improving audience understanding and decision-making through most or all of these qualities, when relevant:\n"
        "- plain-language explanation: explains key distinctions or necessary jargon in simple, accessible language.\n"
        "- practical implication: makes clear why the correction matters for health understanding, risk interpretation, or likely behavior.\n"
        "- safe and proportionate next step: gives a cautious, appropriate action suggestion when the misinformation could affect what people do.\n"
        "- calibrated uncertainty: avoids overclaiming and acknowledges relevant limits, uncertainty, or rarity when appropriate.\n\n"
        "Requirements:\n"
        "- Write planner_rationale before planner_ok.\n"
        "- Write actor_rationale before actor_ok.\n"
        "- If planner_ok is false, actor_rationale and actor_ok may be null.\n"
        "- Return JSON only, with no markdown fence and no extra commentary.\n\n"
        f"Episode data:\n{json.dumps(episode_to_prompt_dict(post, episode_run), ensure_ascii=False, indent=2)}\n\n"
        f"Output schema:\n{json.dumps(output_schema, ensure_ascii=False, indent=2)}"
    )