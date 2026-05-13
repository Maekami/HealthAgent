from __future__ import annotations

import json

from healthagent.schemas import MemoryQueryOutput, PostPackage


GOOD_QUERY_EXAMPLES = [
    "Post uses a single health incident to imply a broader causal conclusion",
    "Post makes a strong health claim based on a linked study or article",
    "Post denies the biological basis of a condition and reframes it as lifestyle-related",
]

BAD_QUERY_EXAMPLES = [
    "Recent US polio case unvaccinated paralysis causality",
    "Peter Marks FDA pharma favor evidence",
    "Need to verify whether the statistic is true",
]


def build_planner_memory_query_system_prompt() -> str:
    return (
        "You generate short retrieval queries for planner memory in a health misinformation note system.\n"
        "Your query must identify a recurring post type or claim archetype.\n"
        "Do not write sample-specific fact queries.\n"
        "Return valid JSON only."
    )


def build_planner_memory_query_user_prompt(
    *,
    post: PostPackage,
) -> str:
    output_schema = MemoryQueryOutput.model_json_schema()

    return (
        "Generate a short memory-retrieval query for the planner.\n\n"
        "Output order:\n"
        "1. query\n"
        "2. stage\n"
        "3. rationale\n\n"
        "Rules:\n"
        "- query must identify a recurring post type or claim archetype\n"
        "- query should usually begin with 'Post ...'\n"
        "- query must be generalized and reusable\n"
        "- avoid sample-specific facts, proper nouns, and unresolved factual details\n"
        "- stage must be core_task_incomplete\n"
        "- rationale should be short and only explain why this kind of planner memory is needed\n\n"
        "Good query examples:\n"
        f"{json.dumps(GOOD_QUERY_EXAMPLES, ensure_ascii=False, indent=2)}\n\n"
        "Bad query examples:\n"
        f"{json.dumps(BAD_QUERY_EXAMPLES, ensure_ascii=False, indent=2)}\n\n"
        f"Post:\n{json.dumps(post.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"Output schema:\n{json.dumps(output_schema, ensure_ascii=False, indent=2)}"
    )