"""Feature engineering pipeline."""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from ..utils.config import get_config
from ..utils.logger import get_logger
from .encoders import CategoricalEncoder
from .selectors import FeatureSelector

logger = get_logger(__name__)


class FeaturePipeline:
    """Feature engineering and transformation pipeline."""

    def __init__(self):
        self.config = get_config()
        self.pipeline = None
        self.encoder = CategoricalEncoder()
        self.selector = FeatureSelector()

    def build_pipeline(self) -> Pipeline:
        """Build scikit-learn feature engineering pipeline.

        Returns:
            Fitted sklearn Pipeline.
        """
        numerical_features = self.config.get("features.numerical_features", [])
        categorical_features = self.config.get("features.categorical_features", [])

        # Numerical transformation
        numerical_transformer = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
            ]
        )

        # Categorical transformation
        categorical_transformer = self.encoder.get_transformer()

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        # Create final pipeline
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("selector", self.selector.get_selector()),
            ]
        )

        logger.info("Built feature engineering pipeline")
        return self.pipeline

    def fit(self, X: pd.DataFrame) -> "FeaturePipeline":
        """Fit the pipeline on training data.

        Args:
            X: Training features.

        Returns:
            Self for method chaining.
        """
        if self.pipeline is None:
            self.build_pipeline()

        self.pipeline.fit(X)
        logger.info("Fitted feature pipeline on training data")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features.

        Args:
            X: Features to transform.

        Returns:
            Transformed features.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        X_transformed = self.pipeline.transform(X)
        logger.info(f"Transformed features: {X_transformed.shape}")
        return X_transformed

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform features.

        Args:
            X: Features to fit and transform.

        Returns:
            Transformed features.
        """
        return self.fit(X).transform(X)
