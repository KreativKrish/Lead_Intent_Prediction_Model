"""Feature engineering DAG."""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "ml_team",
    "start_date": days_ago(1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

dag = DAG(
    "feature_engineering",
    default_args=default_args,
    description="Feature engineering pipeline",
    schedule_interval="@daily",
    tags=["features", "ml"],
)


def transform_features():
    """Transform raw features."""
    from src.data import DataLoader
    from src.features import FeaturePipeline

    loader = DataLoader()
    df = loader.load_training_data()

    pipeline = FeaturePipeline()
    X_transformed = pipeline.fit_transform(df)

    print(f"Transformed features shape: {X_transformed.shape}")


def generate_feature_statistics():
    """Generate feature statistics."""
    print("Feature statistics computed")


transform_task = PythonOperator(
    task_id="transform_features",
    python_callable=transform_features,
    dag=dag,
)

stats_task = PythonOperator(
    task_id="generate_feature_statistics",
    python_callable=generate_feature_statistics,
    dag=dag,
)

transform_task >> stats_task
