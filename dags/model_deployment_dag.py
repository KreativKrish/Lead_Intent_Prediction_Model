"""Model deployment DAG."""

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
    "model_deployment",
    default_args=default_args,
    description="Model deployment and promotion",
    schedule_interval=None,  # Manual trigger
    tags=["deployment", "ml"],
)


def promote_to_staging(**context):
    """Promote model to Staging stage."""
    from src.models import ModelRegistry

    registry = ModelRegistry()
    latest_version = registry.get_latest_version("None")

    if latest_version:
        registry.transition_stage(latest_version, "Staging")
        print(f"Promoted model v{latest_version} to Staging")


def promote_to_production(**context):
    """Promote model to Production stage."""
    from src.models import ModelRegistry

    registry = ModelRegistry()
    staging_version = registry.get_latest_version("Staging")

    if staging_version:
        registry.transition_stage(staging_version, "Production")
        print(f"Promoted model v{staging_version} to Production")


staging_task = PythonOperator(
    task_id="promote_to_staging",
    python_callable=promote_to_staging,
    dag=dag,
)

prod_task = PythonOperator(
    task_id="promote_to_production",
    python_callable=promote_to_production,
    dag=dag,
)

staging_task >> prod_task
