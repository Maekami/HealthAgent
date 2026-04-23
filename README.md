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

