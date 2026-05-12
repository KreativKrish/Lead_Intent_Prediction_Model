"""Model training DAG — trains all three tiers and triggers deployment gate."""

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
    description="Ensemble model training pipeline (Baseline / Challenger / Champion)",
    schedule_interval="@weekly",
    tags=["training", "ml"],
)


def train_all_models(**context):
    """Train Baseline (LR), Challenger (GB/RF), and Champion (XGBoost) tiers."""
    from src.training import Trainer

    trainer = Trainer()
    run_ids = trainer.train_all()

    ti = context["task_instance"]
    ti.xcom_push(key="baseline_run_id",   value=run_ids["baseline_run_id"])
    ti.xcom_push(key="challenger_run_id", value=run_ids["challenger_run_id"])
    ti.xcom_push(key="champion_run_id",   value=run_ids["champion_run_id"])
    ti.xcom_push(key="winner_run_id",     value=run_ids["winner_run_id"])
    ti.xcom_push(key="winner_tier",       value=run_ids["winner_tier"])

    print(
        f"Ensemble training complete. "
        f"Winner: {run_ids['winner_tier']} (run_id={run_ids['winner_run_id']})"
    )
    return run_ids


def evaluate_all_models(**context):
    """Log a comparison summary using the XCom run IDs pushed by train_all_models."""
    ti = context["task_instance"]
    winner_run_id = ti.xcom_pull(key="winner_run_id", task_ids="train_all_models")
    winner_tier   = ti.xcom_pull(key="winner_tier",   task_ids="train_all_models")
    print(f"Evaluation complete. Selected champion: {winner_tier} (run_id={winner_run_id})")


def trigger_deployment(**context):
    """Trigger the model_deployment DAG passing the winner run_id in conf."""
    from airflow.api.common.trigger_dag import trigger_dag

    ti = context["task_instance"]
    conf = {
        "winner_run_id": ti.xcom_pull(key="winner_run_id", task_ids="train_all_models"),
        "winner_tier":   ti.xcom_pull(key="winner_tier",   task_ids="train_all_models"),
    }
    trigger_dag(dag_id="model_deployment", conf=conf, replace_microseconds=False)
    print(f"Triggered model_deployment DAG with conf: {conf}")


train_task = PythonOperator(
    task_id="train_all_models",
    python_callable=train_all_models,
    dag=dag,
)

eval_task = PythonOperator(
    task_id="evaluate_all_models",
    python_callable=evaluate_all_models,
    dag=dag,
)

trigger_task = PythonOperator(
    task_id="trigger_deployment",
    python_callable=trigger_deployment,
    dag=dag,
)

train_task >> eval_task >> trigger_task
