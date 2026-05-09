from __future__ import annotations

import json

from healthagent.schemas import MemoryQueryOutput, PostPackage


def build_planner_memory_query_system_prompt() -> str:
    return (
        "You generate short retrieval queries for planner memory in a health misinformation note system.\n"
        "Your job is to summarize the recurring post pattern and planning challenge in a compact query.\n"
        "Do not write broad universal advice.\n"
        "Return valid JSON only."
    )


def build_planner_memory_query_user_prompt(
    *,
    post: PostPackage,
) -> str:
    output_schema = MemoryQueryOutput.model_json_schema()

    return (
        "Generate a short memory-retrieval query for the planner.\n\n"
        "The query should capture:\n"
        "- the type of post or media pattern\n"
        "- the main health-claim pattern\n"
        "- the planning challenge or checkworthy-claim pattern\n\n"
        "Requirements:\n"
        "- Keep the query short, specific, and reusable.\n"
        "- Avoid entity-specific wording unless absolutely necessary.\n"
        "- Prefer recurring post patterns over one-off details.\n"
        "- Return JSON only, with no markdown fence and no extra commentary.\n\n"
        f"Post:\n{json.dumps(post.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"Output schema:\n{json.dumps(output_schema, ensure_ascii=False, indent=2)}"
    )