"""MLflow integration utilities."""

import os
from typing import Any

import mlflow
from mlflow.client import MlflowClient
from mlflow.entities import Run

from .config import get_config
from .logger import get_logger

logger = get_logger(__name__)


class MLflowManager:
    """Manager for MLflow tracking and model registry operations."""

    def __init__(self):
        config = get_config()
        self.tracking_uri = config.get("mlflow.tracking_uri", "http://localhost:5000")
        self.model_name = config.get("mlflow.model_name", "lead_intent_model")
        self.experiment_name = config.get("mlflow.experiment_name", "lead_intent_prediction")
        self.client = MlflowClient(self.tracking_uri)
        mlflow.set_tracking_uri(self.tracking_uri)

    def setup_experiment(self) -> str:
        """Create experiment if not exists and set as active.

        Returns:
            Experiment ID.
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                self.experiment_name,
                artifact_location=None,
                tags={"type": "production"},
            )
        else:
            experiment_id = experiment.experiment_id

        mlflow.set_experiment(self.experiment_name)
        logger.info(f"Using experiment: {self.experiment_name} (ID: {experiment_id})")
        return experiment_id

    def start_run(self, run_name: str, tags: dict[str, str] | None = None) -> Run:
        """Start a new MLflow run.

        Args:
            run_name: Name for the run.
            tags: Optional tags to attach to run.

        Returns:
            MLflow Run object.
        """
        run = mlflow.start_run(run_name=run_name)
        if tags:
            mlflow.set_tags(tags)
        logger.info(f"Started MLflow run: {run_name} (ID: {run.info.run_id})")
        return run

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run.

        Args:
            status: Run status (FINISHED, FAILED, SCHEDULED).
        """
        mlflow.end_run(status=status)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        """Log metrics to MLflow."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str, artifact_path: str = "") -> None:
        """Log artifact to MLflow."""
        if os.path.isfile(local_path):
            mlflow.log_artifact(local_path, artifact_path)
        else:
            mlflow.log_artifacts(local_path, artifact_path)

    def register_model(self, run_id: str, artifact_uri: str = "model") -> str:
        """Register a model from a run to the model registry.

        Args:
            run_id: MLflow run ID.
            artifact_uri: Artifact path within run (default: 'model').

        Returns:
            Registered model version.
        """
        model_uri = f"runs:/{run_id}/{artifact_uri}"
        version = mlflow.register_model(model_uri, self.model_name)
        logger.info(f"Registered model {self.model_name} version {version.version}")
        return version.version

    def load_champion_model(self, stage: str = "Production"):
        """Load champion model from registry.

        Args:
            stage: Model stage (Production, Staging, None).

        Returns:
            Loaded model (pyfunc format).
        """
        model_uri = f"models:/{self.model_name}/{stage}"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Loaded champion model from stage: {stage}")
        return model

    def transition_model_stage(
        self, version: str, stage: str
    ) -> None:
        """Transition model version to new stage.

        Args:
            version: Model version.
            stage: Target stage (Staging, Production, Archived).
        """
        self.client.transition_model_version_stage(
            self.model_name, version, stage, archive_existing_versions=True
        )
        logger.info(f"Transitioned {self.model_name} v{version} to {stage}")

    def get_run_info(self, run_id: str) -> dict[str, Any]:
        """Get run information.

        Args:
            run_id: MLflow run ID.

        Returns:
            Run metadata dictionary.
        """
        run = self.client.get_run(run_id)
        return {
            "run_id": run.info.run_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "params": run.data.params,
            "metrics": run.data.metrics,
        }
