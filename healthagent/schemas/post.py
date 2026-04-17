from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from .base import SchemaBase


class ImageTextView(SchemaBase):
    caption: Optional[str] = None
    ocr: Optional[str] = None
    description: Optional[str] = None


class VideoTextView(SchemaBase):
    transcript: Optional[str] = None
    keyframe_captions: List[str] = Field(default_factory=list)
    ocr: Optional[str] = None
    temporal_summary: Optional[str] = None


class PostPackage(SchemaBase):
    post_id: str
    tweet_text: str
    image_views: List[ImageTextView] = Field(default_factory=list)
    video_views: List[VideoTextView] = Field(default_factory=list)