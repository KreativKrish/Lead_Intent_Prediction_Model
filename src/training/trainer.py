"""Model training orchestration."""

import mlflow

from ..data import DataLoader
from ..features import FeaturePipeline
from ..models import LeadIntentModel
from ..utils.config import get_config
from ..utils.logger import get_logger
from ..utils.mlflow_utils import MLflowManager
from .evaluator import Evaluator

logger = get_logger(__name__)


class Trainer:
    """Orchestrate end-to-end model training pipeline."""

    def __init__(self):
        self.config = get_config()
        self.mlflow_mgr = MLflowManager()
        self.data_loader = DataLoader()
        self.evaluator = Evaluator()

    def train(self, run_name: str = "lead_intent_training") -> str:
        """Execute full training pipeline.

        Args:
            run_name: MLflow run name.

        Returns:
            MLflow run ID.
        """
        # Setup MLflow experiment
        self.mlflow_mgr.setup_experiment()

        # Start MLflow run with autolog
        mlflow.autolog()

        with mlflow.start_run(run_name=run_name) as run:
            logger.info(f"Starting training run: {run_name}")

            # Load data
            df = self.data_loader.load_training_data()
            train_df, test_df = self.data_loader.split_data(df)

            # Extract features and target
            target_col = self.config.get("features.target_column")
            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]

            # Feature engineering
            feature_pipeline = FeaturePipeline()
            feature_pipeline.fit(X_train)
            X_train_transformed = feature_pipeline.transform(X_train)
            X_test_transformed = feature_pipeline.transform(X_test)

            logger.info(f"Feature engineered: {X_train_transformed.shape}")

            # Build and train model
            model = LeadIntentModel(feature_pipeline)
            model.build_model()
            model.fit(X_train_transformed, y_train)

            # Evaluate
            metrics = self.evaluator.evaluate(
                model, X_test_transformed, y_test
            )

            # Log metrics and artifacts
            mlflow.log_metrics(metrics)
            mlflow.log_params(
                {
                    "test_size": self.config.get("training.test_size"),
                    "random_state": self.config.get("training.random_state"),
                }
            )

            # Log model
            mlflow.sklearn.log_model(feature_pipeline.pipeline, "preprocessor")
            mlflow.xgboost.log_model(model.model, "model")

            # Log feature importance
            feature_importance = model.get_feature_importance(top_n=10)
            mlflow.log_dict(feature_importance, "feature_importance.json")

            logger.info(f"Training complete. Run ID: {run.info.run_id}")
            logger.info(f"Metrics: {metrics}")

            return run.info.run_id


def main():
    """Entry point for training CLI."""
    trainer = Trainer()
    run_id = trainer.train()
    logger.info(f"Training completed. Run ID: {run_id}")


if __name__ == "__main__":
    main()
