"""Seed a trained model into local MLflow for development/testing.

Generates synthetic data — no Snowflake required.
Run once after starting MLflow to make a Production model available for the API.

Usage:
    python scripts/train_local.py
"""

import os
import sys

# Load .env if present (sets MLFLOW_TRACKING_URI, MODEL_NAME, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Sensible defaults so the script works before .env is configured
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "lead_intent_prediction")
os.environ.setdefault("MODEL_NAME", "lead_intent_model")
os.environ.setdefault("ENV", "development")

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SKPipeline

from src.features import FeaturePipeline
from src.models import LeadIntentModel


def generate_synthetic_data(n: int = 600) -> pd.DataFrame:
    """Generate realistic synthetic lead data with a learnable intent signal."""
    np.random.seed(42)
    lead_score       = np.random.uniform(0, 100, n)
    engagement_score = np.random.uniform(0, 100, n)
    response_time    = np.random.uniform(0, 48, n)
    email_click_rate = np.random.uniform(0, 1, n)

    # Intent correlates with lead quality signals — gives model something to learn
    signal = (
        0.40 * lead_score / 100
        + 0.30 * engagement_score / 100
        + 0.20 * (1 - response_time / 48)
        + 0.10 * email_click_rate
    )
    intent = (signal + np.random.normal(0, 0.12, n) > 0.5).astype(int)

    return pd.DataFrame({
        "lead_score":             lead_score,
        "company_size":           np.random.randint(5, 1000, n),
        "engagement_score":       engagement_score,
        "response_time_hours":    response_time,
        "email_open_rate":        np.random.uniform(0, 1, n),
        "email_click_rate":       email_click_rate,
        "page_views":             np.random.randint(0, 100, n),
        "time_since_signup_days": np.random.randint(0, 365, n),
        "industry":       np.random.choice(["Technology", "Finance", "Healthcare", "Retail"], n),
        "company_type":   np.random.choice(["SaaS", "Enterprise", "Startup", "SMB"], n),
        "location":       np.random.choice(["US", "EU", "APAC", "LATAM"], n),
        "product_interest": np.random.choice(["Enterprise", "SMB", "Startup"], n),
        "source":         np.random.choice(["LinkedIn", "Direct", "Partner", "Event"], n),
        "sales_stage":    np.random.choice(["Awareness", "Consideration", "Decision", "Qualification"], n),
        "intent_label":   intent,
    })


def main():
    tracking_uri  = os.environ["MLFLOW_TRACKING_URI"]
    experiment    = os.environ.get("MLFLOW_EXPERIMENT_NAME", "lead_intent_prediction")
    model_name    = os.environ.get("MODEL_NAME", "lead_intent_model")

    print(f"MLflow:     {tracking_uri}")
    print(f"Experiment: {experiment}")
    print(f"Model name: {model_name}")
    print("Generating synthetic training data (600 samples)…")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    df = generate_synthetic_data(600)
    X  = df.drop(columns=["intent_label"])
    y  = df["intent_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    feature_pipeline = FeaturePipeline()
    feature_pipeline.build_pipeline()

    model = LeadIntentModel()
    model.build_model()

    full_pipeline = SKPipeline([
        ("preprocessor", feature_pipeline.pipeline),
        ("classifier",   model.model),
    ])

    print("Training XGBoost champion model…")
    with mlflow.start_run(run_name="local_dev_seed") as run:
        full_pipeline.fit(X_train, y_train)

        y_prob = full_pipeline.predict_proba(X_test)[:, 1]
        y_pred = full_pipeline.predict(X_test)

        metrics = {
            "roc_auc":  round(roc_auc_score(y_test, y_prob), 4),
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1":       round(f1_score(y_test, y_pred), 4),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(full_pipeline, "model")
        run_id = run.info.run_id

    print(f"  Run ID:  {run_id}")
    print(f"  Metrics: {metrics}")

    # Register and promote to Production
    print("Registering model in MLflow registry…")
    client  = mlflow.MlflowClient()
    result  = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    version = result.version

    # Archive any existing Production version
    for v in client.get_latest_versions(model_name, stages=["Production"]):
        client.transition_model_version_stage(model_name, v.version, "Archived")

    client.transition_model_version_stage(model_name, version, "Production")

    print(f"\n  Model v{version} → Production")
    print("\nDone! Next steps:")
    print(f"  MLflow UI:  {tracking_uri}")
    print("  Start API:  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")
    print("  Dashboard:  http://localhost:8000/ui")


if __name__ == "__main__":
    main()
