"""Data validation utilities."""

import pandas as pd

from ..utils.config import get_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Validate data quality and schema."""

    def __init__(self):
        self.config = get_config()

    def validate(self, df: pd.DataFrame) -> bool:
        """Run all validation checks.

        Args:
            df: DataFrame to validate.

        Returns:
            True if valid, raises exception otherwise.
        """
        if not self.config.get("data_validation.enabled", False):
            logger.info("Data validation disabled")
            return True

        self.check_shape(df)
        self.check_duplicates(df)
        self.check_nulls(df)
        self.check_dtypes(df)

        logger.info("Data validation passed")
        return True

    def check_shape(self, df: pd.DataFrame) -> None:
        """Check DataFrame shape is non-empty."""
        if df.shape[0] == 0:
            raise ValueError("DataFrame is empty")
        if df.shape[1] == 0:
            raise ValueError("DataFrame has no columns")
        logger.info(f"Shape check passed: {df.shape}")

    def check_duplicates(self, df: pd.DataFrame) -> None:
        """Check for duplicate rows."""
        if not self.config.get("data_validation.check_duplicates", False):
            return

        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows")

    def check_nulls(self, df: pd.DataFrame) -> None:
        """Check for null values."""
        if not self.config.get("data_validation.check_nulls", False):
            return

        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Found null values:\n{null_counts[null_counts > 0]}")

    def check_dtypes(self, df: pd.DataFrame) -> None:
        """Check data types are appropriate."""
        numerical_cols = self.config.get("features.numerical_features", [])
        categorical_cols = self.config.get("features.categorical_features", [])

        for col in numerical_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"Expected numeric column '{col}' is not numeric")

        for col in categorical_cols:
            if col in df.columns and not (
                df[col].dtype == "object" or pd.api.types.is_categorical_dtype(df[col])
            ):
                logger.warning(f"Expected categorical column '{col}' is not object/categorical")

    def detect_outliers(self, df: pd.DataFrame, method: str = "iqr") -> pd.Series:
        """Detect outliers in numerical columns.

        Args:
            df: Input DataFrame.
            method: Detection method (iqr, zscore).

        Returns:
            Boolean Series indicating outliers.
        """
        if not self.config.get("data_validation.outlier_detection.enabled", False):
            return pd.Series([False] * len(df))

        numerical_cols = df.select_dtypes(include=["number"]).columns

        if method == "iqr":
            multiplier = self.config.get("data_validation.outlier_detection.multiplier", 1.5)
            outliers = pd.Series([False] * len(df), index=df.index)

            for col in numerical_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers |= (df[col] < Q1 - multiplier * IQR) | (df[col] > Q3 + multiplier * IQR)

            logger.info(f"Detected {outliers.sum()} outliers using IQR method")
            return outliers

        return pd.Series([False] * len(df))
