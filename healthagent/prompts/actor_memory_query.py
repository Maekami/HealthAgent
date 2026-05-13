from __future__ import annotations

import json
from typing import Any, Mapping

from healthagent.schemas import (
    CompressedHistory,
    InstanceRubrics,
    MemoryQueryOutput,
    PostPackage,
)


GOOD_QUERY_EXAMPLES = [
    "Post uses a recent single health case to imply broader causality",
    "Post uses a vivid adverse event to generalize intervention risk",
    "Post uses a linked study or article to support a stronger health claim than the source directly justifies",
]

BAD_QUERY_EXAMPLES = [
    "Recent US polio case unvaccinated paralysis causality",
    "Peter Marks FDA actions pharma favor evidence",
    "Need to verify the core claims about vaccination status and causality",
]


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
        "Your query must identify a recurring post type or claim archetype.\n"
        "Current state should be reflected in stage, not in the query itself.\n"
        "Do not write sample-specific fact queries.\n"
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
        "Output order:\n"
        "1. query\n"
        "2. stage\n"
        "3. rationale\n\n"
        "Rules for query:\n"
        "- query must identify a recurring post type or claim archetype\n"
        "- query should usually begin with 'Post ...'\n"
        "- query must be generalized and reusable\n"
        "- query must not contain sample-specific facts, proper nouns, or unresolved factual details\n"
        "- current state should be reflected in stage rather than in the query text\n\n"
        "Rules for stage:\n"
        "- core_task_incomplete: the planner-defined core task is not yet fully completed\n"
        "- core_task_complete: the core task is already completed, and the remaining question is whether the note can be improved toward an excellent note\n\n"
        "Rules for rationale:\n"
        "- keep it short\n"
        "- explain why memory for this kind of post is needed now\n"
        "- do not restate sample-specific missing facts\n\n"
        "Good query examples:\n"
        f"{json.dumps(GOOD_QUERY_EXAMPLES, ensure_ascii=False, indent=2)}\n\n"
        "Bad query examples:\n"
        f"{json.dumps(BAD_QUERY_EXAMPLES, ensure_ascii=False, indent=2)}\n\n"
        f"Post:\n{json.dumps(post.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"Instance rubrics:\n{json.dumps(instance_rubrics.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"Compressed history:\n{json.dumps(_history_to_prompt_dict(history), ensure_ascii=False, indent=2)}\n\n"
        f"Budget state:\n{json.dumps(dict(budget_state), ensure_ascii=False, indent=2)}\n\n"
        f"Output schema:\n{json.dumps(output_schema, ensure_ascii=False, indent=2)}"
    )