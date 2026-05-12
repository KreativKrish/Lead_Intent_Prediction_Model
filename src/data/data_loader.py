"""Data loading and preprocessing utilities."""

import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils.config import get_config
from ..utils.logger import get_logger
from .snowflake_connector import SnowflakeConnector
from .validators import DataValidator

logger = get_logger(__name__)


class DataLoader:
    """Load and preprocess data from Snowflake."""

    def __init__(self):
        self.connector = SnowflakeConnector()
        self.config = get_config()
        self.validator = DataValidator()

    def load_training_data(self, table_name: str = "TRAINING_DATA") -> pd.DataFrame:
        """Load training data from Snowflake.

        Args:
            table_name: Name of the training data table.

        Returns:
            DataFrame with training data.
        """
        logger.info(f"Loading training data from {table_name}")
        df = self.connector.fetch_dataframe(f"SELECT * FROM {table_name}")

        # Validate data
        self.validator.validate(df)

        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    def load_features(self, table_name: str = "FEATURES") -> pd.DataFrame:
        """Load feature data from Snowflake.

        Args:
            table_name: Name of the features table.

        Returns:
            DataFrame with features.
        """
        logger.info(f"Loading features from {table_name}")
        df = self.connector.fetch_dataframe(f"SELECT * FROM {table_name}")
        logger.info(f"Loaded {len(df)} feature vectors")
        return df

    def split_data(self, df: pd.DataFrame, test_size: float | None = None):
        """Split data into train and test sets.

        Args:
            df: Input DataFrame.
            test_size: Test set proportion (default from config).

        Returns:
            Tuple of (train_df, test_df).
        """
        test_size = test_size or self.config.get("training.test_size", 0.2)
        random_state = self.config.get("training.random_state", 42)

        target_col = self.config.get("features.target_column")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[target_col],
        )

        logger.info(f"Split data: train={len(train_df)}, test={len(test_df)}")
        return train_df, test_df

    def get_feature_columns(self) -> tuple[list, list]:
        """Get numerical and categorical feature column names.

        Returns:
            Tuple of (numerical_features, categorical_features).
        """
        numerical = self.config.get("features.numerical_features", [])
        categorical = self.config.get("features.categorical_features", [])
        return numerical, categorical

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        """Handle missing values in DataFrame.

        Args:
            df: Input DataFrame.
            strategy: Strategy for imputation (mean, median, ffill).

        Returns:
            DataFrame with missing values handled.
        """
        if df.isnull().sum().sum() == 0:
            logger.info("No missing values found")
            return df

        logger.warning(f"Found {df.isnull().sum().sum()} missing values")

        if strategy == "mean":
            numeric_cols = df.select_dtypes(include=["number"]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == "median":
            numeric_cols = df.select_dtypes(include=["number"]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == "ffill":
            df = df.fillna(method="ffill").fillna(method="bfill")

        logger.info(f"Imputed missing values using {strategy}")
        return df
