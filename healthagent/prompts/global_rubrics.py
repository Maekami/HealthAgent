from __future__ import annotations

from typing import Sequence


def build_action_space_rubrics(
    available_actions: Sequence[str],
) -> list[str]:
    actions = list(available_actions)
    rubrics: list[str] = []

    rubrics.append(
        f"You must choose exactly one action from this action space: {', '.join(actions)}."
    )

    if "search" in actions:
        rubrics.append(
            "Use search only when you need to discover new candidate evidence pages."
        )

    if "visit" in actions:
        rubrics.append(
            "Use visit only when there is a promising candidate URL from search history or post URLs worth reading next."
        )
        rubrics.append(
            "If you choose visit, provide a one-sentence goal for what to extract from the page."
        )

    if "write" in actions:
        rubrics.append(
            "Use write only when the visited page summaries are sufficient to support a concise final note."
        )
        rubrics.append(
            "If you choose write, the text must stay within 270 characters."
        )
        rubrics.append(
            "If you choose write, the text field must not contain any URLs."
        )
        rubrics.append(
            "If you choose write, support URLs must come from previously visited pages; for URLs that appeared in the post, short-link and resolved full-form aliases are both allowed."
        )

    if "abstain" in actions:
        rubrics.append(
            "Use abstain when the evidence is insufficient and the best next step is to stop."
        )

    if actions == ["write", "abstain"] or set(actions) == {"write", "abstain"}:
        rubrics.append(
            "Only write and abstain are available now. Decide whether the currently collected evidence is sufficient for a minimally useful note; if yes, write using only current evidence, otherwise abstain."
        )

    return rubrics