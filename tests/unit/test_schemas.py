"""Unit tests for API schemas."""

import pytest

from api.schemas import LeadFeatures, PredictionResponse


@pytest.mark.unit
def test_lead_features_validation():
    """Test LeadFeatures validation."""
    valid_data = {
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

    features = LeadFeatures(**valid_data)
    assert features.lead_score == 75
    assert features.company_size == 100


@pytest.mark.unit
def test_lead_features_invalid_score():
    """Test invalid lead score."""
    invalid_data = {
        "lead_score": 150,  # > 100
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

    with pytest.raises(ValueError):
        LeadFeatures(**invalid_data)


@pytest.mark.unit
def test_prediction_response():
    """Test PredictionResponse creation."""
    response = PredictionResponse(
        prediction_id="pred_123",
        probability=0.85,
        predicted_label=True,
        model_version="v1.0.0",
        confidence=0.92,
    )

    assert response.probability == 0.85
    assert response.predicted_label is True
