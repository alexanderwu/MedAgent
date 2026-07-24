from typing import List, Optional, Literal
from pydantic import BaseModel, Field

# Centralized definitions of what your model actually supports
SUPPORTED_METRICS = Literal["readmission_rate", "length_of_stay", "mortality_risk"]
SUPPORTED_CHARACTERISTICS = Literal[
    "patient_weight", "history_of_heart_problems", "age", "diabetic_status"
]


class PatientDataExtraction(BaseModel):
    """Pydantic schema enforcing extraction directly into supported literal enums."""

    requested_metrics: List[SUPPORTED_METRICS] = Field(
        default=[],
        description="List of allowed prediction metrics requested by the user.",
    )
    characteristics: List[SUPPORTED_CHARACTERISTICS] = Field(
        default=[], description="List of allowed patient characteristics provided."
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Set ONLY if there is not enough information to identify metrics or characteristics.",
    )
