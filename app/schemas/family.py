from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ElderlyProfileCreate(BaseModel):
    full_name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=65, le=120)
    gender: str | None = Field(default=None, max_length=20)
    care_tier: str = Field(default="Daily Assistance", max_length=50)
    chronic_conditions: list[str] = Field(default_factory=list)
    emergency_contact_name: str | None = Field(default=None, max_length=100)
    emergency_contact_phone: str | None = Field(default=None, max_length=50)
    residence_address: str | None = Field(default=None, max_length=255)


class ElderlyProfileResponse(ElderlyProfileCreate):
    id: int
    family_user_id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CareSessionCreate(BaseModel):
    elderly_id: int
    scheduled_date: datetime
    service_tier: str = Field(..., min_length=1, max_length=50)
    special_instructions: str | None = None


class CareSessionResponse(BaseModel):
    id: int
    elderly_id: int
    family_user_id: UUID
    caregiver_id: UUID | None
    scheduled_date: datetime
    service_tier: str
    special_instructions: str | None
    status: str
    vitals_recorded: dict[str, Any] | None
    encounter_notes: str | None
    completed_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


class CareSessionComplete(BaseModel):
    vitals_recorded: dict[str, Any] | None = None
    encounter_notes: str | None = None
