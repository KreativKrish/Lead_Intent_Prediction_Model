"""Request and response schemas."""

from typing import Optional

from pydantic import BaseModel, Field


class LeadFeatures(BaseModel):
    """Lead features for prediction."""

    lead_score: float = Field(..., ge=0, le=100, description="Lead quality score")
    company_size: int = Field(..., ge=1, description="Company size")
    engagement_score: float = Field(..., ge=0, le=100, description="Engagement score")
    response_time_hours: float = Field(..., ge=0, description="Response time in hours")
    email_open_rate: float = Field(..., ge=0, le=1, description="Email open rate")
    email_click_rate: float = Field(..., ge=0, le=1, description="Email click rate")
    page_views: int = Field(..., ge=0, description="Number of page views")
    time_since_signup_days: int = Field(..., ge=0, description="Days since signup")
    industry: str = Field(..., description="Industry")
    company_type: str = Field(..., description="Company type")
    location: str = Field(..., description="Location")
    product_interest: str = Field(..., description="Product interest")
    source: str = Field(..., description="Lead source")
    sales_stage: str = Field(..., description="Sales stage")

    class Config:
        json_schema_extra = {
            "example": {
                "lead_score": 75,
                "company_size": 100,
                "engagement_score": 85,
                "response_time_hours": 2,
                "email_open_rate": 0.6,
                "email_click_rate": 0.3,
                "page_views": 15,
                "time_since_signup_days": 30,
                "industry": "Technology",
                "company_type": "SaaS",
                "location": "US",
                "product_interest": "Enterprise",
                "source": "LinkedIn",
                "sales_stage": "Qualification",
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response."""

    prediction_id: Optional[str] = Field(None, description="Unique prediction ID")
    probability: float = Field(..., ge=0, le=1, description="Intent probability score")
    predicted_label: bool = Field(..., description="Predicted intent label (True/False)")
    model_version: Optional[str] = Field(None, description="Model version used")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction_id": "pred_123abc",
                "probability": 0.85,
                "predicted_label": True,
                "model_version": "v1.0.0",
                "confidence": 0.92,
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Model loaded status")
    message: str = Field(..., description="Status message")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
