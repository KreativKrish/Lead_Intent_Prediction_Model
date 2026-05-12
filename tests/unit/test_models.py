"""Unit tests for models."""

import numpy as np
import pytest

from src.models import LeadIntentModel


@pytest.mark.unit
def test_model_build():
    """Test model building."""
    model = LeadIntentModel()
    xgb_model = model.build_model()
    assert xgb_model is not None


@pytest.mark.unit
def test_model_fit_predict(sample_features_df, sample_target_series):
    """Test model fitting and prediction."""
    model = LeadIntentModel()
    model.fit(sample_features_df, sample_target_series)

    predictions = model.predict(sample_features_df)
    assert len(predictions) == len(sample_features_df)
    assert all(pred in [0, 1] for pred in predictions)


@pytest.mark.unit
def test_model_predict_proba(sample_features_df, sample_target_series):
    """Test probability predictions."""
    model = LeadIntentModel()
    model.fit(sample_features_df, sample_target_series)

    proba = model.predict_proba(sample_features_df)
    assert proba.shape == (len(sample_features_df), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)


@pytest.mark.unit
def test_feature_importance(sample_features_df, sample_target_series):
    """Test feature importance extraction."""
    model = LeadIntentModel()
    model.fit(sample_features_df, sample_target_series)

    importance = model.get_feature_importance(top_n=5)
    assert len(importance) <= 5
    assert all(isinstance(v, (int, float)) for v in importance.values())
