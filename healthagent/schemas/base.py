from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SchemaBase(BaseModel):
    """Base class for all schemas."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=True,
    )