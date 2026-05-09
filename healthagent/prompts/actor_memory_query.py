from __future__ import annotations

import json
from typing import Any, Mapping

from healthagent.schemas import (
    CompressedHistory,
    InstanceRubrics,
    MemoryQueryOutput,
    PostPackage,
)


def _history_to_prompt_dict(history: CompressedHistory) -> dict[str, Any]:
    return {
        "searches": [
            {
                "query": item.query,
                "results": [
                    {
                        "url": result.url,
                        "title": result.title,
                        "domain": result.domain,
                        "snippet": result.snippet,
                    }
                    for result in item.results
                ],
                "error_message": item.error_message,
            }
            for item in history.searches
        ],
        "visited_pages": [
            {
                "url": page.url,
                "summary": page.summary,
            }
            for page in history.visited_pages
        ],
    }


def build_actor_memory_query_system_prompt() -> str:
    return (
        "You generate short retrieval queries for actor memory in a health misinformation note system.\n"
        "Your job is to summarize the current decision state in a compact query.\n"
        "Focus on the post pattern, what has already been checked, what still seems missing, and the current action pressure.\n"
        "Do not write broad universal advice.\n"
        "Return valid JSON only."
    )


def build_actor_memory_query_user_prompt(
    *,
    post: PostPackage,
    instance_rubrics: InstanceRubrics,
    history: CompressedHistory,
    budget_state: Mapping[str, Any],
) -> str:
    output_schema = MemoryQueryOutput.model_json_schema()

    return (
        "Generate a short memory-retrieval query for the actor.\n\n"
        "The query should capture:\n"
        "- the recurring post or claim pattern\n"
        "- what the planner says must be checked\n"
        "- what has already been searched or visited\n"
        "- what still seems missing or unresolved\n"
        "- the current decision pressure (for example whether write is available or whether the step budget is tight)\n\n"
        "Requirements:\n"
        "- Keep the query short, specific, and reusable.\n"
        "- Prefer the current bottleneck over generic restatement.\n"
        "- Avoid entity-specific wording unless absolutely necessary.\n"
        "- Return JSON only, with no markdown fence and no extra commentary.\n\n"
        f"Post:\n{json.dumps(post.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"Instance rubrics:\n{json.dumps(instance_rubrics.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"Compressed history:\n{json.dumps(_history_to_prompt_dict(history), ensure_ascii=False, indent=2)}\n\n"
        f"Budget state:\n{json.dumps(dict(budget_state), ensure_ascii=False, indent=2)}\n\n"
        f"Output schema:\n{json.dumps(output_schema, ensure_ascii=False, indent=2)}"
    )