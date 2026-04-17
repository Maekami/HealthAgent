from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from healthagent.schemas import CompressedHistory, InstanceRubrics, PostPackage


def build_actor_system_prompt() -> str:
    return (
        "You are the core actor in a long-horizon evidence-backed note-writing system.\n"
        "At each step, you must first think briefly about the current state, then choose exactly one action.\n"
        "Your available actions are: search, visit, write, abstain.\n\n"
        "Rules:\n"
        "- Do not invent URLs.\n"
        "- If you choose visit, the URL must come from the current search history.\n"
        "- If you choose write, every supported claim must be backed by a previously visited URL.\n"
        "- Keep the thinking fields concise and operational, not verbose.\n"
        "- Return valid JSON only."
    )


def _history_to_prompt_dict(history: CompressedHistory) -> dict[str, Any]:
    visited_url_set = {item.url for item in history.visited_pages}

    searches = []
    for search in history.searches:
        searches.append(
            {
                "query": search.query,
                "results": [
                    {
                        "url": result.url,
                        "title": result.title,
                        "domain": result.domain,
                        "snippet": result.snippet,
                        "visited": result.url in visited_url_set,
                    }
                    for result in search.results
                ],
            }
        )

    visited_pages = [
        {
            "url": page.url,
            "summary": page.summary,
        }
        for page in history.visited_pages
    ]

    return {
        "searches": searches,
        "visited_pages": visited_pages,
    }


def build_actor_user_prompt(
    *,
    post: PostPackage,
    global_rubrics: Sequence[str],
    instance_rubrics: InstanceRubrics,
    history: CompressedHistory,
    budget_state: Mapping[str, Any],
    actor_memory: Sequence[str] | None = None,
) -> str:
    memory_lines = list(actor_memory or [])
    memory_block = (
        "\n".join(f"- {item}" for item in memory_lines)
        if memory_lines
        else "None"
    )

    output_schema = {
        "thinking": {
            "current_assessment": "Short summary of what is already known from prior searches/visits.",
            "main_gap": "Short statement of the key missing information or unresolved issue.",
            "decision_rationale": "Short statement of why the chosen action is the best next step.",
        },
        "action": {
            "action": "search | visit | write | abstain",
            "query": "Required only if action=search",
            "url": "Required only if action=visit",
            "goal": "Required only if action=visit",
            "text": "Required only if action=write; must not include any URLs",
            "support": [
                {
                    "claim": "Required only if action=write",
                    "url": "Must be a previously visited URL if action=write",
                }
            ],
            "reason": "Required for every action",
        },
    }

    return (
        "Decide the best next step for this post.\n\n"
        "Decision rules:\n"
        "- Choose search if you still need to discover new candidate evidence pages.\n"
        "- Choose visit if there is a promising candidate URL in search history that should be read next.\n"
        "- Choose write only if the visited page summaries are sufficient to support a concise final note.\n"
        "- Choose abstain if the evidence is insufficient and the best next step is to stop.\n"
        "- For visit, write a one-sentence goal telling the summary model what to extract.\n"
        "- For write, the text must stay within 270 characters.\n" # TODO: length constraint can be modified.
        "- For write, the text field must not contain any URLs.\n"
        "- For write, support URLs must come from previously visited pages.\n"
        "- Keep the thinking fields concise.\n"
        "- Return JSON only, with no markdown fence and no extra commentary.\n\n"
        f"Global rubrics:\n{json.dumps(list(global_rubrics), ensure_ascii=False, indent=2)}\n\n"
        f"Actor memory:\n{memory_block}\n\n"
        f"Post package:\n{json.dumps(post.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"Instance-specific rubrics:\n{json.dumps(instance_rubrics.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"Compressed history:\n{json.dumps(_history_to_prompt_dict(history), ensure_ascii=False, indent=2)}\n\n"
        f"Budget state:\n{json.dumps(dict(budget_state), ensure_ascii=False, indent=2)}\n\n"
        f"Output schema:\n{json.dumps(output_schema, ensure_ascii=False, indent=2)}"
    )