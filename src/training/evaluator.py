"""Model evaluation and metrics computation."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from ..utils.config import get_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Evaluate model performance and compute metrics."""

    def __init__(self):
        self.config = get_config()

    def evaluate(self, model, X_test, y_test) -> dict[str, float]:
        """Evaluate model on test set.

        Args:
            model: Fitted model with predict/predict_proba methods.
            X_test: Test features.
            y_test: Test labels.

        Returns:
            Dictionary of evaluation metrics.
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        logger.info(f"Evaluation metrics: {metrics}")

        # Check thresholds
        self._check_thresholds(metrics)

        # Log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")

        return metrics

    def _check_thresholds(self, metrics: dict) -> None:
        """Check if metrics meet minimum thresholds.

        Args:
            metrics: Computed metrics dictionary.
        """
        thresholds = self.config.get("evaluation.thresholds", {})

        for metric, threshold in thresholds.items():
            if metric in metrics:
                if metrics[metric] < threshold:
                    logger.warning(
                        f"{metric}: {metrics[metric]:.4f} < {threshold} (below threshold)"
                    )
                else:
                    logger.info(f"{metric}: {metrics[metric]:.4f} >= {threshold} (passed)")

    def compute_feature_importance_dataframe(self, model, feature_names):
        """Create feature importance DataFrame.

        Args:
            model: Fitted model.
            feature_names: List of feature names.

        Returns:
            DataFrame with feature importance.
        """
        import pandas as pd

        importance = model.get_feature_importance(top_n=len(feature_names))
        return pd.DataFrame(
            list(importance.items()),
            columns=["feature", "importance"]
        ).sort_values("importance", ascending=False)
