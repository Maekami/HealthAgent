from healthagent.components.rubric_planner import RubricPlanner
from healthagent.schemas import PostPackage
from healthagent.tools.llm_client import ChatGeneration


class FakeChatClient:
    def chat(self, messages, **kwargs):
        return ChatGeneration(
            text="""
            {
              "core_checkworthy_claims": ["Vitamin C cures influenza in 24 hours"],
              "priority_questions": ["Do authoritative health sources support this treatment claim?"],
              "multimodal_risks": [],
              "note_intent": "correction"
            }
            """,
            model="fake-model",
            finish_reason="stop",
            usage={},
            raw_response={},
        )


def test_rubric_planner_basic():
    planner = RubricPlanner(chat_client=FakeChatClient())

    post = PostPackage(
        post_id="p1",
        tweet_text="Vitamin C cures influenza in 24 hours.",
    )

    run = planner.plan_run(post)

    assert run.rubrics.note_intent == "correction"
    assert len(run.rubrics.core_checkworthy_claims) == 1
    assert "Vitamin C" in run.rubrics.core_checkworthy_claims[0]

# from healthagent.components.rubric_planner import RubricPlanner
# from healthagent.schemas import PostPackage
# from healthagent.tools import OpenRouterChatClient,VLLMChatClient

# client = VLLMChatClient(
#     base_url=VLLM_BASE_URL,
#     default_model="google/medgemma-27b-text-it"
# )

# planner = RubricPlanner(
#     chat_client=client,
#     model=None,  # use default_model
# )

# post = PostPackage(
#     post_id="post_001",
#     # tweet_text="Vitamin C cures influenza in 24 hours.",
#     tweet_text="Polio vaccines are not poison. Polio vaccines reduce polio paralysis and disease. The Gaza Strip has been polio-free for the last 25 years. Its reemergence can cause polio outbreaks beyond Gaza."
# )

# run = planner.plan_run(post)