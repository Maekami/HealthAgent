from healthagent.tools import (
    FirecrawlCrawlEngine,
    GoalConditionedSummaryModel,
    OpenRouterChatClient,
    SerperSearchEngine,
)

search = SerperSearchEngine(api_key="YOUR_SERPER_KEY")
search_obs = search.search("does vitamin c cure flu", created_at_ms=None)
print(search_obs.model_dump())

crawl = FirecrawlCrawlEngine(api_key="YOUR_FIRECRAWL_KEY")
scrape_result = crawl.crawl("https://www.cdc.gov/flu/treatment/index.html")
print(scrape_result.metadata.get("title"))

llm = OpenRouterChatClient(
    api_key="YOUR_OPENROUTER_KEY",
    default_model="YOUR_MODEL_NAME",
)

summarizer = GoalConditionedSummaryModel(llm)
visit_obs = summarizer.summarize(
    scrape_result,
    goal="Extract only the information relevant to whether this page supports the claim that vitamin C cures influenza.",
)
print(visit_obs.model_dump())