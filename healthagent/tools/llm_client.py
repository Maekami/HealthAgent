from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Protocol, Sequence, TypedDict

import httpx

from .exceptions import HTTPToolError


Role = Literal["system", "user", "assistant", "tool"]


class ChatMessage(TypedDict):
    role: Role
    content: str


@dataclass(slots=True)
class ChatGeneration:
    text: str
    model: Optional[str]
    finish_reason: Optional[str]
    usage: dict[str, Any]
    raw_response: dict[str, Any]


class ChatClient(Protocol):
    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: Optional[dict[str, Any]] = None,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> ChatGeneration:
        ...


def _extract_message_text(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content

    if isinstance(message_content, list):
        chunks: list[str] = []
        for item in message_content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
                elif isinstance(item.get("content"), str):
                    chunks.append(item["content"])
        return "\n".join(chunks).strip()

    return str(message_content)


class BaseOpenAICompatibleChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        default_model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 120.0,
        default_headers: Optional[dict[str, str]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.default_headers = default_headers or {}

    @property
    def max_tokens_field(self) -> str:
        return "max_tokens"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.default_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: Optional[str],
        temperature: float,
        max_tokens: int,
        response_format: Optional[dict[str, Any]],
        extra_body: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        final_model = model or self.default_model
        if not final_model:
            raise ValueError("No model provided and no default_model configured.")

        payload: dict[str, Any] = {
            "model": final_model,
            "messages": list(messages),
            "temperature": temperature,
        }
        payload[self.max_tokens_field] = max_tokens

        if response_format is not None:
            payload["response_format"] = response_format
        if extra_body:
            payload.update(extra_body)

        return payload

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: Optional[dict[str, Any]] = None,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> ChatGeneration:
        payload = self._build_payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            extra_body=extra_body,
        )

        endpoint = f"{self.base_url}/chat/completions"

        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(
                endpoint,
                headers=self._headers(),
                json=payload,
            )

        if resp.status_code >= 400:
            raise HTTPToolError(
                "LLM chat request failed",
                status_code=resp.status_code,
                response_body=resp.text,
            )

        data = resp.json()
        if not isinstance(data, dict):
            raise HTTPToolError("LLM response is not a JSON object", response_body=data)

        choices = data.get("choices") or []
        if not choices:
            raise HTTPToolError("LLM response has no choices", response_body=data)

        first = choices[0]
        message = first.get("message") or {}
        text = _extract_message_text(message.get("content"))

        return ChatGeneration(
            text=text.strip(),
            model=data.get("model"),
            finish_reason=first.get("finish_reason"),
            usage=dict(data.get("usage") or {}),
            raw_response=data,
        )


class OpenRouterChatClient(BaseOpenAICompatibleChatClient):
    def __init__(
        self,
        api_key: str,
        *,
        default_model: Optional[str] = None,
        timeout_s: float = 120.0,
        site_url: Optional[str] = None,
        site_title: Optional[str] = None,
    ) -> None:
        headers: dict[str, str] = {}
        if site_url:
            headers["HTTP-Referer"] = site_url
        if site_title:
            headers["X-OpenRouter-Title"] = site_title

        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            default_model=default_model,
            api_key=api_key,
            timeout_s=timeout_s,
            default_headers=headers,
        )

    @property
    def max_tokens_field(self) -> str:
        # OpenRouter docs recommend max_completion_tokens.
        return "max_completion_tokens"


class VLLMChatClient(BaseOpenAICompatibleChatClient):
    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8001/v1",
        default_model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 120.0,
    ) -> None:
        super().__init__(
            base_url=base_url,
            default_model=default_model,
            api_key=api_key,
            timeout_s=timeout_s,
        )

    @property
    def max_tokens_field(self) -> str:
        return "max_tokens"