"""Pytest configuration and fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_lead_data():
    """Generate sample lead data for testing."""
    n_samples = 100
    data = {
        "lead_score": np.random.uniform(0, 100, n_samples),
        "company_size": np.random.randint(1, 1000, n_samples),
        "engagement_score": np.random.uniform(0, 100, n_samples),
        "response_time_hours": np.random.uniform(0, 48, n_samples),
        "email_open_rate": np.random.uniform(0, 1, n_samples),
        "email_click_rate": np.random.uniform(0, 1, n_samples),
        "page_views": np.random.randint(0, 100, n_samples),
        "time_since_signup_days": np.random.randint(0, 365, n_samples),
        "industry": np.random.choice(
            ["Technology", "Finance", "Healthcare", "Retail"],
            n_samples
        ),
        "company_type": np.random.choice(
            ["SaaS", "Enterprise", "Startup", "SMB"],
            n_samples
        ),
        "location": np.random.choice(
            ["US", "EU", "APAC", "LATAM"],
            n_samples
        ),
        "product_interest": np.random.choice(
            ["Enterprise", "SMB", "Startup"],
            n_samples
        ),
        "source": np.random.choice(
            ["LinkedIn", "Direct", "Partner", "Event"],
            n_samples
        ),
        "sales_stage": np.random.choice(
            ["Awareness", "Consideration", "Decision", "Qualification"],
            n_samples
        ),
        "intent_label": np.random.choice([0, 1], n_samples),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_features_df(sample_lead_data):
    """Get sample features without target."""
    return sample_lead_data.drop(columns=["intent_label"])


@pytest.fixture
def sample_target_series(sample_lead_data):
    """Get sample target variable."""
    return sample_lead_data["intent_label"]


@pytest.fixture
def mock_snowflake_connector(monkeypatch, sample_lead_data):
    """Mock Snowflake connector."""
    class MockConnector:
        def fetch_dataframe(self, query):
            return sample_lead_data

    def mock_init(self):
        pass

    monkeypatch.setattr(
        "src.data.snowflake_connector.SnowflakeConnector.__init__",
        mock_init
    )
    monkeypatch.setattr(
        "src.data.snowflake_connector.SnowflakeConnector.fetch_dataframe",
        MockConnector().fetch_dataframe
    )
