"""Unit tests for feature engineering."""

import pytest

from src.features import FeaturePipeline


@pytest.mark.unit
def test_pipeline_build():
    """Test feature pipeline building."""
    pipeline = FeaturePipeline()
    sklearn_pipeline = pipeline.build_pipeline()
    assert sklearn_pipeline is not None


@pytest.mark.unit
def test_pipeline_fit_transform(sample_features_df):
    """Test feature pipeline fit and transform."""
    pipeline = FeaturePipeline()
    X_transformed = pipeline.fit_transform(sample_features_df)

    assert X_transformed is not None
    assert len(X_transformed) == len(sample_features_df)


@pytest.mark.unit
def test_pipeline_transform_without_fit(sample_features_df):
    """Test transform without fitting raises error."""
    pipeline = FeaturePipeline()

    with pytest.raises(ValueError):
        pipeline.transform(sample_features_df)
