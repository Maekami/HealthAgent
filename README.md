## Version 1.05

### Enhancements

* **Separated actor evaluation into two stages**
  The previous single `actor_rationale` evaluation is now split into two layers:

  1. Whether the actor completed the minimum required task.
  2. Whether the result truly meets the `excellent` standard.

* **Refined episode classification logic**
  The updated decision flow is now:

  * `planner_ok = false` → `unsatisfactory`
  * `planner_ok = true` and `actor_task_completed = false` → `satisfactory`
  * `planner_ok = true` and `actor_task_completed = true`:

    * `actor_ok = true` → `excellent`
    * `actor_ok = false` → `satisfactory`

* **More robust evaluation behavior**
  The model is now required to first determine whether the actor actually completed the core task before evaluating output quality. This makes the classification process significantly more reliable than relying on a single `actor_rationale`.


## Version 1.04

### Enhancements

* **Mode judge for episode classification**
  Added a first-stage `mode judge` to classify each episode into one of three modes:

  * `excellent`
  * `satisfactory`
  * `unsatisfactory`

* **Routing-based evolver behavior**
  Added a second-stage `evolver` that routes learning behavior based on the episode mode:

  * `excellent`: extract at least one memory from planner or actor behavior.
  * `satisfactory`: provide at least one actor improvement suggestion, with optional planner memory extraction.
  * `unsatisfactory`: provide at least one planner improvement suggestion; actor behavior is ignored.

* **Planner-first evaluation flow**
  The evaluation now first checks whether the planner identified the core claim that needed verification. If not, the episode is marked `unsatisfactory`. If yes, the actor’s performance is then evaluated to decide between `satisfactory` and `excellent`.


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

