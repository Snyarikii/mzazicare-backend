from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


CareLogCategory = Literal[
    "MEDICATION", "VITALS", "MEAL", "NOTE", "INCIDENT", "VISIT_COMPLETION"
]


class ScheduleVisit(BaseModel):
    id: int
    patient_id: str
    patient_name: str
    patient_avatar: str | None = None
    service_type: str
    start_time: datetime
    end_time: datetime | None = None
    location_address: str | None = None
    status: Literal["UPCOMING", "IN_PROGRESS", "COMPLETED"]
    notes: str | None = None


class CareLogCreate(BaseModel):
    patient_id: str = Field(min_length=1, max_length=50)
    category: CareLogCategory
    notes: str = Field(min_length=1)

    @field_validator("patient_id", "notes")
    @classmethod
    def strip_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("This field cannot be blank.")
        return value


class CareLogResponse(BaseModel):
    id: UUID
    caregiver_id: UUID
    patient_id: str
    patient_name: str | None = None
    category: CareLogCategory
    notes: str
    created_at: datetime
