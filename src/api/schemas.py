"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Input features for injury probability prediction (user-provided)."""

    age: int = Field(..., ge=16, le=45, description="Player age (16-45)")
    position: str = Field(..., min_length=1, max_length=80)
    injury: str = Field(..., min_length=1, max_length=120)
    fifa_rating: float = Field(..., ge=50.0, le=99.0, description="FIFA rating (50-99)")
    absence_days: int = Field(..., ge=1, le=730, description="Last injury absence in days")
    previous_injuries: int = Field(..., ge=0, le=50, description="Number of previous injuries")
    days_since_last_injury: int = Field(
        ...,
        ge=0,
        le=2000,
        description="Days since last return (use 999 if first injury)",
    )


class PredictResponse(BaseModel):
    """API response for POST /predict."""

    injury_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., pattern="^(Low|Medium|High)$")
