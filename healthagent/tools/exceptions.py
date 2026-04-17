from __future__ import annotations

from typing import Any, Optional


class ToolError(RuntimeError):
    """Base exception for tool-layer failures."""


class HTTPToolError(ToolError):
    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response_body: Optional[Any] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body

    def __str__(self) -> str:
        base = super().__str__()
        if self.status_code is None:
            return base
        return f"{base} (status_code={self.status_code})"