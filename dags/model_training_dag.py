"""Model training DAG."""

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
    "model_training",
    default_args=default_args,
    description="Model training pipeline",
    schedule_interval="@weekly",
    tags=["training", "ml"],
)


def train_model(**context):
    """Train the model."""
    from src.training import Trainer

    trainer = Trainer()
    run_id = trainer.train()

    # Store run_id in XCom for downstream tasks
    context["task_instance"].xcom_push(key="mlflow_run_id", value=run_id)
    print(f"Training complete. Run ID: {run_id}")

    return run_id


def evaluate_model(**context):
    """Evaluate the model."""
    run_id = context["task_instance"].xcom_pull(
        key="mlflow_run_id", task_ids="train_model"
    )
    print(f"Evaluating model from run: {run_id}")


def register_model(**context):
    """Register model to MLflow registry."""
    from src.models import ModelRegistry

    run_id = context["task_instance"].xcom_pull(
        key="mlflow_run_id", task_ids="train_model"
    )

    registry = ModelRegistry()
    version = registry.register_model(run_id)
    print(f"Model registered. Version: {version}")


train_task = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag,
)

eval_task = PythonOperator(
    task_id="evaluate_model",
    python_callable=evaluate_model,
    dag=dag,
)

register_task = PythonOperator(
    task_id="register_model",
    python_callable=register_model,
    dag=dag,
)

train_task >> eval_task >> register_task
