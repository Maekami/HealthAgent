from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from healthagent.schemas import CompressedHistory, InstanceRubrics, PostPackage


def build_actor_system_prompt() -> str:
    return (
        "You are the core actor in a long-horizon evidence-backed note-writing system.\n"
        "At each step, you must first think briefly about the current state, then choose exactly one action.\n"
        "Follow the provided rubrics and current context carefully.\n"
        "Do not invent URLs.\n"
        "Keep the thinking fields concise and operational, not verbose.\n"
        "Return valid JSON only."
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
                "error_message": search.error_message,
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


def _build_action_output_schema(
    allowed_actions: Sequence[str],
) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []

    if "search" in allowed_actions:
        variants.append(
            {
                "action": "search",
                "query": "Required",
                "reason": "Required",
            }
        )

    if "visit" in allowed_actions:
        variants.append(
            {
                "action": "visit",
                "url": "Required",
                "goal": "Required",
                "reason": "Required",
            }
        )

    if "write" in allowed_actions:
        variants.append(
            {
                "action": "write",
                "text": "Required; must not include any URLs",
                "support": [
                    {
                        "claim": "Required",
                        "url": "Must refer to an allowed previously visited URL",
                    }
                ],
                "reason": "Required",
            }
        )

    if "abstain" in allowed_actions:
        variants.append(
            {
                "action": "abstain",
                "reason": "Required",
            }
        )

    return variants


def build_actor_user_prompt(
    *,
    post: PostPackage,
    global_rubrics: Sequence[str],
    instance_rubrics: InstanceRubrics,
    history: CompressedHistory,
    budget_state: Mapping[str, Any],
    actor_memory: Sequence[str] | None = None,
    available_actions: Sequence[str] | None = None,
) -> str:
    memory_lines = list(actor_memory or [])
    memory_block = (
        "\n".join(f"- {item}" for item in memory_lines)
        if memory_lines
        else "None"
    )

    allowed_actions = list(available_actions or ["search", "visit", "write", "abstain"])

    output_schema = {
        "thinking": {
            "current_assessment": "Short summary of what is already known from prior searches/visits.",
            "main_gap": "Short statement of the key missing information or unresolved issue.",
            "decision_rationale": "Short statement of why the chosen action is the best next step.",
        },
        "action": _build_action_output_schema(allowed_actions),
    }

    return (
        "Decide the best next step for this post.\n\n"
        f"Available actions for this step:\n{json.dumps(allowed_actions, ensure_ascii=False, indent=2)}\n\n"
        f"Global rubrics:\n{json.dumps(list(global_rubrics), ensure_ascii=False, indent=2)}\n\n"
        f"Actor memory:\n{memory_block}\n\n"
        f"Post package:\n{json.dumps(post.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"Instance-specific rubrics:\n{json.dumps(instance_rubrics.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"Compressed history:\n{json.dumps(_history_to_prompt_dict(history), ensure_ascii=False, indent=2)}\n\n"
        f"Budget state:\n{json.dumps(dict(budget_state), ensure_ascii=False, indent=2)}\n\n"
        f"Output schema:\n{json.dumps(output_schema, ensure_ascii=False, indent=2)}"
    )

def build_write_refinement_system_prompt() -> str:
    return (
        "You are a writing refiner for an evidence-backed note-writing system.\n"
        "Your only job is to shorten an existing note text so that it fits a strict character limit.\n"
        "Do not add new claims. Do not change the factual meaning unless needed for brevity.\n"
        "Preserve the most important evidence-backed content.\n"
        "Do not include any URLs.\n"
        "Learn from previous failed refinement attempts if they are provided.\n"
        "Return valid JSON only."
    )


def build_write_refinement_user_prompt(
    *,
    original_text: str,
    support: Sequence[dict[str, Any]],
    max_text_chars: int,
    previous_failures: Sequence[dict[str, Any]] | None = None,
) -> str:
    output_schema = {
        "text": "A shorter version of the original text, within the character limit, with no URLs.",
        "reason": "A short explanation of how the text was compressed.",
    }

    failures = list(previous_failures or [])

    return (
        "Refine the note text so it fits the required text length limit.\n\n"
        "Rules:\n"
        "- Shorten the text to fit within the limit.\n"
        "- Do not include any URLs.\n"
        "- Do not add new claims.\n"
        "- Keep the most important supported content.\n"
        "- If previous failed attempts are provided, avoid repeating the same ineffective compression pattern.\n"
        "- Return JSON only, with no markdown fence and no extra commentary.\n\n"
        f"Original text:\n{original_text}\n\n"
        f"Support:\n{json.dumps(list(support), ensure_ascii=False, indent=2)}\n\n"
        f"Previous failed attempts:\n{json.dumps(failures, ensure_ascii=False, indent=2)}\n\n"
        f"Character limit:\n{max_text_chars}\n\n"
        f"Output schema:\n{json.dumps(output_schema, ensure_ascii=False, indent=2)}"
    )