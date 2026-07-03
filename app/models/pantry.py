from __future__ import annotations

from datetime import date, datetime
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class PantryItem(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    owner_id: UUID | None = None
    household_id: UUID | None = None
    name: str
    category: str
    expiry_date: date | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    embedding: list[float] | None = None
    embedding_metadata: dict = Field(default_factory=dict)
    embedding_status: str = "pending"
    embedding_updated_at: datetime | None = None
    embedding_error: str | None = None
