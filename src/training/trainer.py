"""Model training orchestration."""

import mlflow

from ..data import DataLoader
from ..features import FeaturePipeline
from ..models import GradientBoostingModel, LeadIntentModel, LogisticRegressionModel
from ..utils.config import get_config
from ..utils.logger import get_logger
from ..utils.mlflow_utils import MLflowManager
from .evaluator import Evaluator
from .model_selector import ModelResult, ModelSelector

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

    def train_all(self, run_name: str = "lead_intent_ensemble_training") -> dict[str, str]:
        """Train all three model tiers under a shared parent MLflow run.

        Returns:
            Dict with keys: baseline_run_id, challenger_run_id, champion_run_id,
            winner_run_id, winner_tier.
        """
        self.mlflow_mgr.setup_experiment()
        mlflow.autolog(disable=True)

        with mlflow.start_run(run_name=run_name) as parent_run:
            mlflow.set_tag("pipeline_type", "ensemble_training")
            logger.info(f"Starting ensemble training run: {parent_run.info.run_id}")

            # --- Shared data loading and feature engineering ---
            df = self.data_loader.load_training_data()
            train_df, test_df = self.data_loader.split_data(df)

            target_col = self.config.get("features.target_column")
            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]

            feature_pipeline = FeaturePipeline()
            feature_pipeline.fit(X_train)
            X_train_t = feature_pipeline.transform(X_train)
            X_test_t = feature_pipeline.transform(X_test)

            logger.info(f"Feature engineering complete: {X_train_t.shape}")

            # --- Train each tier sequentially ---
            tier_classes = [LogisticRegressionModel, GradientBoostingModel, LeadIntentModel]
            results: list[ModelResult] = []
            run_ids: dict[str, str] = {}

            for ModelClass in tier_classes:
                result = self._train_single_tier(
                    ModelClass, feature_pipeline, X_train_t, y_train, X_test_t, y_test
                )
                results.append(result)
                run_ids[f"{result.tier}_run_id"] = result.mlflow_run_id

            # --- Select champion ---
            selector = ModelSelector()
            winner = selector.select(results)
            run_ids["winner_run_id"] = winner.mlflow_run_id
            run_ids["winner_tier"] = winner.tier

            mlflow.set_tag("winning_tier", winner.tier)
            primary = selector.primary_metric
            mlflow.log_metric(f"winning_{primary}", winner.metrics.get(primary, 0.0))

        logger.info(f"Ensemble training complete. Winner: {winner.tier} ({winner.mlflow_run_id})")
        return run_ids

    def _train_single_tier(
        self,
        ModelClass,
        feature_pipeline,
        X_train,
        y_train,
        X_test,
        y_test,
    ) -> ModelResult:
        """Train one model tier in a nested MLflow child run."""
        tier = ModelClass.MODEL_TIER

        with mlflow.start_run(run_name=f"tier_{tier}", nested=True) as child_run:
            mlflow.set_tag("model_tier", tier)

            model = ModelClass(feature_pipeline)
            model.build_model()
            model.fit(X_train, y_train)

            metrics = self.evaluator.evaluate(model, X_test, y_test)
            mlflow.log_metrics(metrics)
            mlflow.log_params(model.get_params())

            mlflow.sklearn.log_model(feature_pipeline.pipeline, "preprocessor")
            if tier == "champion":
                mlflow.xgboost.log_model(model.model, "model")
            else:
                mlflow.sklearn.log_model(model.model, "model")

            feature_importance = model.get_feature_importance(top_n=10)
            mlflow.log_dict(feature_importance, "feature_importance.json")

            run_id = child_run.info.run_id
            logger.info(f"Tier '{tier}' complete — run_id={run_id}, metrics={metrics}")

        return ModelResult(
            tier=tier,
            model=model,
            metrics=metrics,
            mlflow_run_id=run_id,
            params=model.get_params(),
        )


def main():
    """Entry point for training CLI."""
    trainer = Trainer()
    run_id = trainer.train()
    logger.info(f"Training completed. Run ID: {run_id}")


if __name__ == "__main__":
    main()
