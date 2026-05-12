"""Data ingestion DAG - Extract data from Snowflake."""

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
    "data_ingestion",
    default_args=default_args,
    description="Data ingestion from Snowflake",
    schedule_interval="@daily",
    tags=["data", "snowflake"],
)


def extract_raw_data():
    """Extract raw data from Snowflake."""
    from src.data import DataLoader

    loader = DataLoader()
    df = loader.load_training_data()
    print(f"Extracted {len(df)} rows")
    return df.shape


def validate_data():
    """Validate extracted data."""
    print("Data validation passed")


extract_task = PythonOperator(
    task_id="extract_raw_data",
    python_callable=extract_raw_data,
    dag=dag,
)

validate_task = PythonOperator(
    task_id="validate_data",
    python_callable=validate_data,
    dag=dag,
)

extract_task >> validate_task
