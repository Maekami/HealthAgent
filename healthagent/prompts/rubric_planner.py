from __future__ import annotations

import json
from typing import Sequence

from healthagent.schemas import PostPackage


def build_rubric_planner_system_prompt() -> str:
    return (
        "You are a rubric planner for an evidence-backed note-writing system.\n"
        "Your job is to analyze a post package and produce instance-specific rubrics.\n"
        "Do not evaluate truthfulness directly. Do not write the final note.\n"
        "Your output must help a downstream agent decide what claims to verify first in order to write a useful note.\n"
        "Treat the post package as the only available input representation.\n"
        "Do not generate rubrics for verifying image or video authenticity, provenance, creation time, source tracing, satire detection, or broader media context.\n"
        "Focus only on the substantive claims expressed or implied by the post package text.\n"
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
            "1 to 4 concise questions whose answers would most directly help a downstream agent verify the post's claims and write a grounded note."
        ],
        "multimodal_risks": [
            "0 to 3 concise risks caused by how text derived from different modalities may be combined or interpreted, only if genuinely relevant."
        ],
        "note_intent": "correction | context | mixed",
    }

    return (
        "Analyze the following post package and generate instance-specific rubrics.\n\n"
        "Rules:\n"
        "- core_checkworthy_claims should capture the main health-related or evidence-relevant claims made or strongly implied by the post package.\n"
        "- priority_questions must focus only on factual questions that help verify those claims.\n"
        "- priority_questions must NOT ask about the source of the image/video, when or where it was created, whether it is authentic, whether it is edited, whether it is satire, or its broader media context.\n"
        "- priority_questions should instead ask whether the post's substantive claims are supported, contradicted, overstated, decontextualized, or causally misleading.\n"
        "- Good priority_questions usually ask whether a medical benefit or harm claim is supported by authoritative evidence, whether a causal statement is valid, whether a quoted statistic is accurate, or whether a risk is misrepresented.\n"
        "- multimodal_risks should only describe interpretation risks that matter for understanding the post's claims after multimodal content has already been converted into text. Do not mention media provenance or authenticity.\n"
        "- Keep each item concise and specific.\n"
        "- Do not repeat the same idea across fields.\n"
        "- note_intent must be exactly one of: correction, context, mixed.\n"
        "- Return JSON only, with no markdown fence and no extra commentary.\n\n"
        f"Planner memory:\n{memory_block}\n\n"
        f"Post package:\n{json.dumps(_post_package_to_dict(post), ensure_ascii=False, indent=2)}\n\n"
        f"Output schema:\n{json.dumps(output_schema, ensure_ascii=False, indent=2)}"
    )