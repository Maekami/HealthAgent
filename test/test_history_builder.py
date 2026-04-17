from healthagent.components.history_builder import HistoryBuilder
from healthagent.schemas import (
    CompressedHistory,
    SearchObservation,
    SearchResultItem,
    VisitObservation,
)


def test_update_with_search():
    builder = HistoryBuilder()
    history = CompressedHistory()

    obs = SearchObservation(
        results=[
            SearchResultItem(
                url="https://www.cdc.gov/flu/treatment/index.html?utm_source=test",
                title="Flu Treatment",
                domain="www.cdc.gov",
                snippet="CDC page about treatment",
                source_type="organic",
            ),
            SearchResultItem(
                url="https://www.nih.gov/example",
                title="NIH Example",
                domain="www.nih.gov",
                snippet="NIH page",
                source_type="organic",
            ),
        ]
    )

    updated = builder.update_with_search(
        history=history,
        query="flu treatment cdc",
        observation=obs,
    )

    assert len(updated.searches) == 1
    assert updated.searches[0].query == "flu treatment cdc"
    assert len(updated.searches[0].results) == 2
    assert "utm_source" not in updated.searches[0].results[0].url


def test_update_with_visit():
    builder = HistoryBuilder()
    history = CompressedHistory()

    obs = VisitObservation(
        url="https://www.cdc.gov/flu/treatment/index.html?utm_source=test",
        page_title="Flu Treatment",
        domain="www.cdc.gov",
        summary="This page does not support the claim that vitamin C cures influenza.",
    )

    updated = builder.update_with_visit(
        history=history,
        observation=obs,
    )

    assert len(updated.visited_pages) == 1
    assert "utm_source" not in updated.visited_pages[0].url
    assert "vitamin C cures influenza" in updated.visited_pages[0].summary