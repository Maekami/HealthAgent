from __future__ import annotations

from typing import Dict

from pydantic import Field, field_validator

from .base import SchemaBase


class PostPackage(SchemaBase):
    post_id: str
    tweet_text: str = ""
    caption: Dict[str, str] = Field(default_factory=dict)

    @field_validator("post_id", "tweet_text", mode="before")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return " ".join(value.split()).strip()

    @field_validator("caption", mode="before")
    @classmethod
    def _normalize_caption(cls, value):
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError("caption must be a dict[str, str].")

        normalized: dict[str, str] = {}
        for k, v in value.items():
            key = "" if k is None else str(k).strip()
            val = "" if v is None else " ".join(str(v).split()).strip()
            if key and val:
                normalized[key] = val
        return normalized