from healthagent.components.actor import Actor, ActorError
from healthagent.schemas import (
    ActorDecision,
    CompressedHistory,
    InstanceRubrics,
    PostPackage,
    SearchObservation,
    SearchResultItem,
    VisitObservation,
)
from healthagent.tools.llm_client import ChatGeneration
from healthagent.components.history_builder import HistoryBuilder


class FakeChatClient:
    def __init__(self, text: str):
        self.text = text

    def chat(self, messages, **kwargs):
        return ChatGeneration(
            text=self.text,
            model="fake-model",
            finish_reason="stop",
            usage={},
            raw_response={},
        )


def _build_history():
    builder = HistoryBuilder()
    history = CompressedHistory()

    search_obs = SearchObservation(
        results=[
            SearchResultItem(
                url="https://www.cdc.gov/flu/treatment/index.html",
                title="CDC Flu Treatment",
                domain="www.cdc.gov",
                snippet="CDC flu treatment guidance",
                source_type="organic",
            )
        ]
    )
    history = builder.update_with_search(
        history=history,
        query="cdc flu treatment vitamin c",
        observation=search_obs,
    )

    visit_obs = VisitObservation(
        url="https://www.cdc.gov/flu/treatment/index.html",
        page_title="CDC Flu Treatment",
        domain="www.cdc.gov",
        summary="This page does not support the claim that vitamin C cures influenza.",
    )
    history = builder.update_with_visit(
        history=history,
        observation=visit_obs,
    )
    return history


def _post_and_rubrics():
    post = PostPackage(
        post_id="p1",
        tweet_text="Vitamin C cures influenza in 24 hours.",
    )
    rubrics = InstanceRubrics(
        core_checkworthy_claims=["Vitamin C cures influenza in 24 hours."],
        priority_questions=["Do authoritative health sources support this treatment claim?"],
        multimodal_risks=[],
        note_intent="correction",
    )
    return post, rubrics


def test_actor_visit_action_parses():
    history = _build_history()
    post, rubrics = _post_and_rubrics()

    raw = """
    {
      "thinking": {
        "current_assessment": "There is a promising CDC source in the search history and one visited summary already exists.",
        "main_gap": "The agent needs to inspect another candidate page or continue from current evidence.",
        "decision_rationale": "Visiting a candidate URL is the best next step."
      },
      "action": {
        "action": "visit",
        "url": "https://www.cdc.gov/flu/treatment/index.html",
        "goal": "Extract only the information relevant to whether the page supports the claim that vitamin C cures influenza.",
        "reason": "This is a high-value source directly relevant to the claim."
      }
    }
    """

    actor = Actor(chat_client=FakeChatClient(raw))
    run = actor.act_run(
        post=post,
        global_rubrics=["Only write supported content."],
        instance_rubrics=rubrics,
        history=history,
        budget_state={"remaining_steps": 3},
    )

    assert isinstance(run.decision, ActorDecision)
    assert run.decision.action.action == "visit"


def test_actor_write_requires_visited_url():
    history = _build_history()
    post, rubrics = _post_and_rubrics()

    raw = """
    {
      "thinking": {
        "current_assessment": "A CDC page has been visited and summarized.",
        "main_gap": "No major gap remains for a concise note.",
        "decision_rationale": "The evidence is sufficient to write."
      },
      "action": {
        "action": "write",
        "note": "Public health guidance does not support the claim that vitamin C cures influenza.",
        "support": [
          {
            "claim": "Public health guidance does not support the claim.",
            "url": "https://example.com/not-visited"
          }
        ],
        "reason": "The evidence is sufficient."
      }
    }
    """

    actor = Actor(chat_client=FakeChatClient(raw))

    try:
        actor.act_run(
            post=post,
            global_rubrics=["Only write supported content."],
            instance_rubrics=rubrics,
            history=history,
            budget_state={"remaining_steps": 3},
        )
        assert False, "Expected ActorError"
    except ActorError:
        assert True