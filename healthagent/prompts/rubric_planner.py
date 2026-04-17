from __future__ import annotations

import json
from typing import Sequence

from healthagent.schemas import PostPackage


def build_rubric_planner_system_prompt() -> str:
    return (
        "You are a rubric planner for a multimodal evidence-backed note-writing system.\n"
        "Your job is to analyze a post package and produce instance-specific rubrics.\n"
        "Do not evaluate truthfulness directly. Do not write the final note.\n"
        "Only identify what should be checked, what should be prioritized, "
        "what multimodal risks may exist, and what the likely note intent should be.\n"
        "Keep all list items concise, specific, and non-redundant.\n"
        "Output valid JSON only."
    )


def _post_package_to_dict(post: PostPackage) -> dict:
    return post.model_dump()


def build_rubric_planner_user_prompt(
    post: PostPackage,
    planner_memory: Sequence[str] | None = None,
) -> str:
    memory_lines = list(planner_memory or [])
    memory_block = (
        "\n".join(f"- {item}" for item in memory_lines)
        if memory_lines
        else "None"
    )

    output_schema = {
        "core_checkworthy_claims": [
            "1 to 4 concise checkworthy claims explicitly stated or strongly implied by the post package."
        ],
        "priority_questions": [
            "1 to 4 concise questions whose answers would most help downstream search decisions."
        ],
        "multimodal_risks": [
            "0 to 3 concise multimodal risks, only if genuinely relevant."
        ],
        "note_intent": "correction | context | mixed",
    }

    return (
        "Analyze the following post package and generate instance-specific rubrics.\n\n"
        "Rules:\n"
        "- Include only claims that are truly checkworthy.\n"
        "- Include only priority questions that would change what a downstream agent searches for.\n"
        "- Include multimodal risks only when there is a real text-image-video mismatch or media-specific concern.\n"
        "- Keep each item concise and specific.\n"
        "- Do not repeat the same idea across fields.\n"
        "- note_intent must be exactly one of: correction, context, mixed.\n"
        "- Return JSON only, with no markdown fence and no extra commentary.\n\n"
        f"Planner memory:\n{memory_block}\n\n"
        f"Post package:\n{json.dumps(_post_package_to_dict(post), ensure_ascii=False, indent=2)}\n\n"
        f"Output schema:\n{json.dumps(output_schema, ensure_ascii=False, indent=2)}"
    )