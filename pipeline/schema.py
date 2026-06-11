"""Pydantic event schema for user interaction events."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, field_validator


class UserInteractionEvent(BaseModel):
    user_id: int
    movie_id: int
    event_type: str
    rating: Optional[float] = None
    timestamp: datetime

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        allowed = {"rating", "click", "watch"}
        if v not in allowed:
            raise ValueError(f"event_type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, v: Optional[float], info) -> Optional[float]:
        if v is not None and (v < 0.5 or v > 5.0):
            raise ValueError(f"rating must be between 0.5 and 5.0, got {v}")
        return v

    @field_validator("user_id", "movie_id")
    @classmethod
    def validate_positive_ids(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"ID must be positive, got {v}")
        return v
