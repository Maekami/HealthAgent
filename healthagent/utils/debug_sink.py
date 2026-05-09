from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(slots=True)
class DebugSink:
    jsonl_path: Optional[str] = None
    text_path: Optional[str] = None
    print_to_console: bool = True
    printer: Any = print

    def _ensure_parent(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def format_event(self, *, title: str, payload: Any) -> str:
        if isinstance(payload, str):
            payload_text = payload
        else:
            payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
        return f"\n===== {title} =====\n{payload_text}\n"

    def emit(
        self,
        *,
        title: str,
        payload: Any,
        rendered: str | None = None,
        sample_key: str | None = None,
        stage: str | None = None,
    ) -> None:
        if rendered is None:
            if isinstance(payload, str):
                rendered = payload
            else:
                rendered = json.dumps(payload, ensure_ascii=False, indent=2)

        pretty_text = f"\n===== {title} =====\n{rendered}\n"

        if self.print_to_console:
            self.printer(pretty_text)

        if self.text_path:
            self._ensure_parent(self.text_path)
            with Path(self.text_path).open("a", encoding="utf-8") as f:
                f.write(pretty_text)

        if self.jsonl_path:
            self._ensure_parent(self.jsonl_path)
            record = {
                "sample_key": sample_key,
                "stage": stage,
                "title": title,
                "payload": payload,
            }
            with Path(self.jsonl_path).open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")