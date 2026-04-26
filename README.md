## Version 1.03

### Enhancements

* **Extracted note refinement into `write_refinement.py`**
  The note refinement logic has been separated into a dedicated `write_refinement.py` module for cleaner structure and easier maintenance.

* **Minimum action requirements before `write` / `abstain`**
  Added stricter episode-level controls so that the model must complete at least `x` searches (default: `1`) and `y` visits (default: `1`) before `write` or `abstain` becomes available. This helps prevent unsupported writing and avoids cases where the model writes after searching only, without actually visiting sources.

* **Stealth proxy support in `crawl_engine.py`**
  Added an optional stealth proxy mode to enhance scraping success rates in restricted environments.

* **Memory module (initial structure)**
  Implemented the foundational structure of the memory component. Further testing and validation are pending.


## Version 1.02

### Enhancements

* **Dynamic action space control**
  When the `search` or `visit` quota is exhausted, those actions are removed from the actor’s available action space. An error is raised only if the actor still attempts to call a removed action.

* **Restricted final-step behavior**
  On the last available step, the actor’s action space is limited to `write` or `abstain` only. The actor must decide whether the collected information is sufficient to produce a minimally useful note. If it is, the actor writes based on the available evidence; otherwise, it abstains.

* **Retry support in `crawl_engine.py`**
  Added a retry mechanism for extraction failures, with up to `n` attempts (`3` by default). If all retries fail, the engine returns a fallback message such as *“Unable to extract content from this URL, please try another one”* to the summary model, ensuring the pipeline continues instead of stopping on error.

* **Fallback handling in `search_engine.py`**
  Added a similar safeguard in `search_engine.py`, including an `error_message` field to pass failure information forward without breaking the pipeline.

* **Refined `max_text_chars` fallback strategy**
  To prevent smaller models from getting stuck in retry loops, the fallback mechanism is refined: after the first `write` attempt exceeds the limit, a dedicated prompt is provided to guide the actor to focus on concise refinement only, instead of re-entering the full retry cycle or appending to history.


## Version 1.01

### Enhancements

* **Extended URL Visiting**
  The `visit` function now supports URLs found both in search results and within post content.

* **Short URL Support**
  URLs in posts can be either shortened (e.g., `https://t.co/...`) or full-length. Both `visit` and `write` now handle and validate both formats.

* **Improved `max_text_chars` Handling**
  When exceeding the limit, the attempt and overflow details are logged into history, allowing the actor to retry without consuming additional steps.


## Version 1.0
Initial version

