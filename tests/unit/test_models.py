"""Unit tests for models."""

import numpy as np
import pytest

from src.models import GradientBoostingModel, LeadIntentModel, LogisticRegressionModel
from src.training.model_selector import ModelResult, ModelSelector


# ─── Helpers ──────────────────────────────────────────────────────────────────

@pytest.fixture
def numerical_features(sample_features_df, sample_target_series):
    """Numerical-only features — avoids categorical encoding for model unit tests."""
    X = sample_features_df.select_dtypes(include="number")
    return X, sample_target_series


# ─── XGBoost (Champion) ───────────────────────────────────────────────────────

@pytest.mark.unit
def test_model_build():
    model = LeadIntentModel()
    assert model.build_model() is not None


@pytest.mark.unit
def test_xgb_tier():
    assert LeadIntentModel.MODEL_TIER == "champion"


@pytest.mark.unit
def test_model_fit_predict(numerical_features):
    X, y = numerical_features
    model = LeadIntentModel()
    model.fit(X, y)

    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert all(pred in [0, 1] for pred in predictions)


@pytest.mark.unit
def test_model_predict_proba(numerical_features):
    X, y = numerical_features
    model = LeadIntentModel()
    model.fit(X, y)

    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)


@pytest.mark.unit
def test_feature_importance(numerical_features):
    X, y = numerical_features
    model = LeadIntentModel()
    model.fit(X, y)

    importance = model.get_feature_importance(top_n=5)
    assert len(importance) <= 5
    assert all(isinstance(v, (int, float)) for v in importance.values())


# ─── Logistic Regression (Baseline) ──────────────────────────────────────────

@pytest.mark.unit
def test_lr_tier():
    assert LogisticRegressionModel.MODEL_TIER == "baseline"


@pytest.mark.unit
def test_lr_model_build():
    model = LogisticRegressionModel()
    assert model.build_model() is not None


@pytest.mark.unit
def test_lr_fit_predict(numerical_features):
    X_t, y = numerical_features
    model = LogisticRegressionModel()
    model.fit(X_t, y)

    preds = model.predict(X_t)
    assert len(preds) == len(y)
    assert all(p in [0, 1] for p in preds)


@pytest.mark.unit
def test_lr_predict_proba(numerical_features):
    X_t, y = numerical_features
    model = LogisticRegressionModel()
    model.fit(X_t, y)

    proba = model.predict_proba(X_t)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


@pytest.mark.unit
def test_lr_feature_importance(numerical_features):
    X_t, y = numerical_features
    model = LogisticRegressionModel()
    model.fit(X_t, y)

    importance = model.get_feature_importance(top_n=5)
    assert len(importance) <= 5
    assert all(isinstance(v, float) for v in importance.values())


@pytest.mark.unit
def test_lr_not_fitted_raises():
    model = LogisticRegressionModel()
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(np.zeros((5, 3)))


# ─── Gradient Boosting (Challenger) ──────────────────────────────────────────

@pytest.mark.unit
def test_gb_tier():
    assert GradientBoostingModel.MODEL_TIER == "challenger"


@pytest.mark.unit
def test_gb_build_default():
    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingModel()
    clf = model.build_model()
    assert isinstance(clf, GradientBoostingClassifier)


@pytest.mark.unit
def test_gb_build_random_forest(monkeypatch):
    from sklearn.ensemble import RandomForestClassifier

    model = GradientBoostingModel()
    monkeypatch.setattr(model, "_algorithm", "random_forest")
    clf = model.build_model()
    assert isinstance(clf, RandomForestClassifier)


@pytest.mark.unit
def test_gb_fit_predict(numerical_features):
    X_t, y = numerical_features
    model = GradientBoostingModel()
    model.fit(X_t, y)

    preds = model.predict(X_t)
    assert len(preds) == len(y)
    assert all(p in [0, 1] for p in preds)


@pytest.mark.unit
def test_gb_predict_proba(numerical_features):
    X_t, y = numerical_features
    model = GradientBoostingModel()
    model.fit(X_t, y)

    proba = model.predict_proba(X_t)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


@pytest.mark.unit
def test_gb_feature_importance(numerical_features):
    X_t, y = numerical_features
    model = GradientBoostingModel()
    model.fit(X_t, y)

    importance = model.get_feature_importance(top_n=5)
    assert len(importance) <= 5
    assert all(isinstance(v, float) for v in importance.values())


@pytest.mark.unit
def test_gb_not_fitted_raises():
    model = GradientBoostingModel()
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(np.zeros((5, 3)))


# ─── ModelSelector ────────────────────────────────────────────────────────────

@pytest.fixture
def sample_results():
    return [
        ModelResult(tier="baseline",   model=None, metrics={"roc_auc": 0.72}, mlflow_run_id="run_1"),
        ModelResult(tier="challenger", model=None, metrics={"roc_auc": 0.81}, mlflow_run_id="run_2"),
        ModelResult(tier="champion",   model=None, metrics={"roc_auc": 0.85}, mlflow_run_id="run_3"),
    ]


@pytest.mark.unit
def test_model_selector_picks_best(sample_results, monkeypatch):
    monkeypatch.setattr("mlflow.MlflowClient.set_tag", lambda *a, **k: None)
    winner = ModelSelector().select(sample_results)
    assert winner.tier == "champion"
    assert winner.metrics["roc_auc"] == 0.85


@pytest.mark.unit
def test_model_selector_challenger_wins(monkeypatch):
    monkeypatch.setattr("mlflow.MlflowClient.set_tag", lambda *a, **k: None)
    results = [
        ModelResult("baseline",   None, {"roc_auc": 0.72}, "r1"),
        ModelResult("challenger", None, {"roc_auc": 0.91}, "r2"),
        ModelResult("champion",   None, {"roc_auc": 0.85}, "r3"),
    ]
    winner = ModelSelector().select(results)
    assert winner.tier == "challenger"


@pytest.mark.unit
def test_selector_raises_on_empty():
    with pytest.raises(ValueError, match="No model results"):
        ModelSelector().select([])


@pytest.mark.unit
def test_compare_with_registered_champion_promotes(monkeypatch):
    mock_run = type("R", (), {
        "data": type("D", (), {"metrics": {"roc_auc": 0.80}})()
    })()
    monkeypatch.setattr("mlflow.MlflowClient.get_run", lambda self, rid: mock_run)

    selector = ModelSelector()
    assert selector.compare_with_registered_champion({"roc_auc": 0.83}, "old_run") is True


@pytest.mark.unit
def test_compare_with_registered_champion_blocks(monkeypatch):
    mock_run = type("R", (), {
        "data": type("D", (), {"metrics": {"roc_auc": 0.80}})()
    })()
    monkeypatch.setattr("mlflow.MlflowClient.get_run", lambda self, rid: mock_run)

    selector = ModelSelector()
    assert selector.compare_with_registered_champion({"roc_auc": 0.79}, "old_run") is False


@pytest.mark.unit
def test_compare_auto_promotes_on_error(monkeypatch):
    def raise_err(self, rid):
        raise Exception("MLflow unavailable")

    monkeypatch.setattr("mlflow.MlflowClient.get_run", raise_err)
    assert ModelSelector().compare_with_registered_champion({"roc_auc": 0.70}, "bad_run") is True
