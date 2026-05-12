"""Model selection and comparison across tiers."""

from dataclasses import dataclass, field

import mlflow

from ..utils.config import get_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelResult:
    """Container for a single tier's training output."""

    tier: str
    model: object  # BaseLeadModel — kept as object to avoid circular import
    metrics: dict[str, float]
    mlflow_run_id: str
    params: dict[str, object] = field(default_factory=dict)


class ModelSelector:
    """Compare ModelResult objects across tiers and determine the champion."""

    def __init__(self):
        self.config = get_config()
        self.primary_metric: str = self.config.get(
            "model_selection.primary_metric", "roc_auc"
        )
        self.promotion_threshold: float = self.config.get(
            "model_selection.promotion_threshold", 0.0
        )

    def select(self, results: list[ModelResult]) -> ModelResult:
        """Rank all results by primary_metric and return the winner.

        Also tags each MLflow child run with model_tier and selected_as_champion.
        """
        if not results:
            raise ValueError("No model results to compare.")

        ranked = sorted(
            results,
            key=lambda r: r.metrics.get(self.primary_metric, 0.0),
            reverse=True,
        )

        winner = ranked[0]
        self._log_comparison_table(ranked)

        client = mlflow.MlflowClient()
        for result in ranked:
            is_winner = result is winner
            try:
                client.set_tag(result.mlflow_run_id, "model_tier", result.tier)
                client.set_tag(result.mlflow_run_id, "selected_as_champion", str(is_winner))
            except Exception as exc:
                logger.warning(f"Could not set MLflow tags on run {result.mlflow_run_id}: {exc}")

        logger.info(
            f"Champion selected: tier={winner.tier}, "
            f"{self.primary_metric}={winner.metrics.get(self.primary_metric, 'N/A'):.4f}"
        )
        return winner

    def compare_with_registered_champion(
        self,
        challenger_metrics: dict[str, float],
        champion_run_id: str,
    ) -> bool:
        """Return True if challenger beats the registered champion by at least
        promotion_threshold delta on the primary metric.

        Used by the deployment DAG. If champion_run_id cannot be fetched, auto-promotes.
        """
        try:
            client = mlflow.MlflowClient()
            champion_run = client.get_run(champion_run_id)
            champion_score = champion_run.data.metrics.get(self.primary_metric, 0.0)
        except Exception as exc:
            logger.warning(f"Could not load champion run {champion_run_id}: {exc}. Auto-promoting.")
            return True

        challenger_score = challenger_metrics.get(self.primary_metric, 0.0)
        delta = challenger_score - champion_score

        logger.info(
            f"Challenger {self.primary_metric}={challenger_score:.4f} vs "
            f"Champion {self.primary_metric}={champion_score:.4f} "
            f"(delta={delta:+.4f}, threshold={self.promotion_threshold})"
        )
        return delta >= self.promotion_threshold

    def _log_comparison_table(self, ranked: list[ModelResult]) -> None:
        logger.info("=== Model Comparison ===")
        for rank, result in enumerate(ranked, 1):
            score = result.metrics.get(self.primary_metric, float("nan"))
            logger.info(f"  Rank {rank}: {result.tier:12s} | {self.primary_metric}={score:.4f}")
        logger.info("========================")
