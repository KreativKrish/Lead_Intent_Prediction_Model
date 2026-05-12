"""Hyperparameter tuning with Optuna."""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import xgboost as xgb

from ..utils.config import get_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class HyperparameterTuner:
    """Hyperparameter optimization using Optuna with real cross-validation."""

    def __init__(self, model_type: str = "xgboost"):
        """
        Args:
            model_type: One of "xgboost", "logistic_regression",
                        "gradient_boosting", "random_forest".
        """
        self.config = get_config()
        self.model_type = model_type
        self.cv_folds: int = self.config.get("training.cv_folds", 5)
        self.X_train = None
        self.y_train = None

    def set_data(self, X_train, y_train) -> "HyperparameterTuner":
        """Provide training data used by the cross-validation objective."""
        self.X_train = X_train
        self.y_train = y_train
        return self

    def create_study(self, study_name: str = "lead_intent_tuning"):
        """Create Optuna study for hyperparameter tuning."""
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
        """Cross-validated ROC-AUC objective for the configured model type."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Call set_data(X_train, y_train) before optimize().")

        estimator = self._build_trial_estimator(trial)
        scores = cross_val_score(
            estimator,
            self.X_train,
            self.y_train,
            cv=self.cv_folds,
            scoring="roc_auc",
            n_jobs=-1,
        )
        return float(scores.mean())

    def _build_trial_estimator(self, trial: optuna.Trial):
        if self.model_type == "logistic_regression":
            return LogisticRegression(
                C=trial.suggest_float("C", 1e-3, 10.0, log=True),
                max_iter=trial.suggest_int("max_iter", 200, 2000),
                solver=trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
                class_weight="balanced",
                random_state=42,
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 2, 8),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                random_state=42,
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 3, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        else:  # xgboost (default)
            return xgb.XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                objective="binary:logistic",
                eval_metric="auc",
                random_state=42,
                verbosity=0,
            )

    def optimize(self, n_trials: int | None = None) -> dict:
        """Run hyperparameter optimization.

        Returns:
            Best hyperparameters.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Call set_data(X_train, y_train) before optimize().")

        n_trials = n_trials or self.config.get("training.hyperparameter_tuning.n_trials", 50)

        study = self.create_study(study_name=f"lead_intent_tuning_{self.model_type}")
        study.optimize(self.objective, n_trials=n_trials)

        best_params = study.best_params
        best_score = study.best_value

        logger.info(f"Optimization complete [{self.model_type}]. Best score: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")

        return best_params
