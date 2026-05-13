from __future__ import annotations

import json
from typing import Sequence

from healthagent.schemas import PostPackage, RubricPlannerOutput


def build_rubric_planner_system_prompt() -> str:
    return (
        "You are a rubric planner for an evidence-backed note-writing system.\n"
        "Your job is to analyze a post package and produce instance-specific rubrics.\n"
        "Do not evaluate truthfulness directly.\n"
        "Do not write the final note.\n"
        "Your output must help a downstream agent decide what claims to verify first in order to write a useful note.\n"
        "Treat the post package as the only available input representation.\n"
        "Do not generate rubrics for verifying image or video authenticity, provenance, creation time, source tracing, satire detection, or broader media context.\n"
        "Focus only on the substantive claims expressed or implied by the post package text.\n"
        "Keep all fields concise, specific, and non-redundant.\n"
        "Return valid JSON only."
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
        "current_assessment": "Short summary of what the post is mainly claiming or implying and what needs to be framed.",
        "memory_reflection": "Short statement of whether any retrieved planner memory is useful for this post pattern and how it affects rubric formation. If none is useful, say so briefly.",
        "rubrics": RubricPlannerOutput.model_json_schema()["properties"]["rubrics"],
    }

    return (
        "Analyze the following post package and generate instance-specific rubrics.\n\n"
        "Process:\n"
        "- First summarize the post pattern in current_assessment.\n"
        "- Then briefly review the retrieved planner memory in memory_reflection.\n"
        "- Then produce rubrics.\n\n"
        "Rules for memory_reflection:\n"
        "- Keep it short and specific.\n"
        "- Do not restate the whole post.\n"
        "- Explicitly say whether any retrieved memory helps the planner here.\n"
        "- If useful, say what kind of guidance matters.\n"
        "- If not useful, say that no retrieved memory is currently helpful.\n\n"
        "Rules for rubrics:\n"
        "- core_checkworthy_claims should capture the main health-related or evidence-relevant claims made or strongly implied by the post package.\n"
        "- priority_questions must focus only on factual questions that help verify those claims.\n"
        "- priority_questions must NOT ask about the source of the image/video, when or where it was created, whether it is authentic, whether it is edited, whether it is satire, or its broader media context.\n"
        "- priority_questions should instead ask whether the post's substantive claims are supported, contradicted, overstated, decontextualized, or causally misleading.\n"
        "- Good priority_questions usually ask whether a medical benefit or harm claim is supported by authoritative evidence, whether a causal statement is valid, whether a quoted statistic is accurate, or whether a risk is misrepresented.\n"
        "- multimodal_risks should only describe interpretation risks that matter for understanding the post's claims after multimodal content has already been converted into text.\n"
        "- Do not mention media provenance or authenticity.\n"
        "- Keep each item concise and specific.\n"
        "- Do not repeat the same idea across fields.\n"
        "- note_intent must be exactly one of: correction, context, mixed.\n"
        "- Return JSON only, with no markdown fence and no extra commentary.\n\n"
        f"Planner memory:\n{memory_block}\n\n"
        f"Post package:\n{json.dumps(_post_package_to_dict(post), ensure_ascii=False, indent=2)}\n\n"
        f"Output schema:\n{json.dumps(output_schema, ensure_ascii=False, indent=2)}"
    )