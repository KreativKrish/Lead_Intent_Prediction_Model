"""Hyperparameter tuning with Optuna."""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from ..utils.config import get_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class HyperparameterTuner:
    """Hyperparameter optimization using Optuna."""

    def __init__(self):
        self.config = get_config()

    def create_study(self, study_name: str = "lead_intent_tuning"):
        """Create Optuna study for hyperparameter tuning.

        Args:
            study_name: Name of the study.

        Returns:
            Optuna Study object.
        """
        sampler = TPESampler(seed=42)
        pruner = MedianPruner()

        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        logger.info(f"Created Optuna study: {study_name}")
        return study

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for hyperparameter optimization.

        Args:
            trial: Optuna trial object.

        Returns:
            Metric to optimize (AUC score).
        """
        # Suggest hyperparameters
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)

        # Return placeholder score (implement with actual training)
        return 0.85

    def optimize(self, n_trials: int | None = None) -> dict:
        """Run hyperparameter optimization.

        Args:
            n_trials: Number of trials (default from config).

        Returns:
            Best hyperparameters.
        """
        n_trials = n_trials or self.config.get("training.hyperparameter_tuning.n_trials", 50)

        study = self.create_study()
        study.optimize(self.objective, n_trials=n_trials)

        best_params = study.best_params
        best_score = study.best_value

        logger.info(f"Optimization complete. Best score: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")

        return best_params
