"""Model deployment DAG — challenger vs champion gate.

Triggered by model_training_dag with dag_run.conf containing:
  - winner_run_id: MLflow run ID of the training winner
  - winner_tier:   "baseline" | "challenger" | "champion"

If the winner beats the current Production model on the primary metric by at
least promotion_threshold (config: model_selection.promotion_threshold), it is
registered and promoted to Production.  If no Production model exists yet, the
winner is auto-promoted.
"""

from datetime import timedelta

from airflow import DAG
from airflow.exceptions import AirflowSkipException
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
    description="Challenger vs Champion gate — promotes winner to Production",
    schedule_interval=None,  # triggered by model_training_dag
    tags=["deployment", "ml"],
)


def load_challenger_metrics(**context):
    """Pull winner run metrics from MLflow using run_id in dag_run.conf."""
    import mlflow

    conf = context["dag_run"].conf or {}
    winner_run_id = conf.get("winner_run_id")
    winner_tier   = conf.get("winner_tier", "unknown")

    if not winner_run_id:
        raise ValueError("winner_run_id not found in dag_run.conf.")

    mlflow.set_tracking_uri(None)  # use MLFLOW_TRACKING_URI env var
    client = mlflow.MlflowClient()
    run = client.get_run(winner_run_id)
    metrics = dict(run.data.metrics)

    ti = context["task_instance"]
    ti.xcom_push(key="challenger_metrics",  value=metrics)
    ti.xcom_push(key="challenger_run_id",   value=winner_run_id)
    ti.xcom_push(key="challenger_tier",     value=winner_tier)

    print(f"Challenger: tier={winner_tier}, run_id={winner_run_id}, metrics={metrics}")
    return metrics


def compare_models(**context):
    """Compare challenger vs current Production champion.

    Pushes should_promote=True/False to XCom.
    Auto-promotes if no Production model is registered yet.
    """
    from src.models import ModelRegistry
    from src.training import ModelSelector

    ti = context["task_instance"]
    challenger_metrics = ti.xcom_pull(key="challenger_metrics", task_ids="load_challenger_metrics")

    registry = ModelRegistry()
    champion_version = registry.get_latest_version("Production")

    if champion_version is None:
        print("No Production model found. Auto-promoting challenger.")
        ti.xcom_push(key="should_promote", value=True)
        return True

    # Resolve the champion's MLflow run_id from the registry version
    try:
        client = registry.mlflow_mgr.client
        version_info = client.get_model_version(registry.mlflow_mgr.model_name, champion_version)
        champion_run_id = version_info.run_id
    except Exception as exc:
        print(f"Could not load champion version info: {exc}. Auto-promoting.")
        ti.xcom_push(key="should_promote", value=True)
        return True

    selector = ModelSelector()
    should_promote = selector.compare_with_registered_champion(
        challenger_metrics=challenger_metrics,
        champion_run_id=champion_run_id,
    )

    ti.xcom_push(key="should_promote", value=should_promote)
    return should_promote


def register_and_promote(**context):
    """Register the winning run to MLflow Registry and promote to Production.

    Raises AirflowSkipException if the promotion gate was not passed.
    """
    from src.models import ModelRegistry

    ti = context["task_instance"]
    should_promote    = ti.xcom_pull(key="should_promote",      task_ids="compare_models")
    challenger_run_id = ti.xcom_pull(key="challenger_run_id",   task_ids="load_challenger_metrics")
    challenger_tier   = ti.xcom_pull(key="challenger_tier",     task_ids="load_challenger_metrics")

    if not should_promote:
        print("Challenger did not beat champion. Skipping promotion.")
        raise AirflowSkipException("Challenger did not meet promotion threshold.")

    registry = ModelRegistry()
    version = registry.register_model(challenger_run_id, artifact_uri="model")
    registry.transition_stage(version, "Staging")
    registry.transition_stage(version, "Production")

    print(
        f"Promoted {challenger_tier} model to Production — "
        f"run_id={challenger_run_id}, version={version}"
    )


load_task = PythonOperator(
    task_id="load_challenger_metrics",
    python_callable=load_challenger_metrics,
    dag=dag,
)

compare_task = PythonOperator(
    task_id="compare_models",
    python_callable=compare_models,
    dag=dag,
)

promote_task = PythonOperator(
    task_id="register_and_promote",
    python_callable=register_and_promote,
    dag=dag,
)

load_task >> compare_task >> promote_task
