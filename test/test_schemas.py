from healthagent.schemas import (
    CompressedHistory,
    InstanceRubrics,
    PostPackage,
    SearchAction,
    VisitAction,
    WriteAction,
)


def test_post_package():
    post = PostPackage(
        post_id="p1",
        tweet_text="Vitamin C cures flu",
    )
    assert post.post_id == "p1"
    assert post.image_views == []
    assert post.video_views == []


def test_instance_rubrics():
    rubrics = InstanceRubrics(
        core_checkworthy_claims=["Vitamin C cures flu"],
        priority_questions=["Do public health sources support this?"],
        multimodal_risks=[],
        note_intent="correction",
    )
    assert rubrics.note_intent == "correction"


def test_history():
    history = CompressedHistory()
    assert history.searched_queries == []
    assert history.visited_pages == []


def test_actions():
    search = SearchAction(action="search", query="vitamin c flu cdc", reason="Need direct source")
    visit = VisitAction(
        action="visit",
        url="https://www.cdc.gov/",
        goal="Extract whether the page directly addresses the claim.",
        reason="High-value source",
    )
    write = WriteAction(
        action="write",
        note="Public health guidance does not support this claim.",
        support=[],
        reason="Sufficient evidence gathered",
    )

    assert search.action == "search"
    assert visit.action == "visit"
    assert write.action == "write"