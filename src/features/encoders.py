"""Custom feature encoders and transformers."""

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


class CategoricalEncoder:
    """Encoder for categorical features."""

    def get_transformer(self) -> Pipeline:
        """Get categorical transformation pipeline.

        Returns:
            sklearn Pipeline for categorical encoding.
        """
        return Pipeline(
            steps=[
                (
                    "onehot",
                    OneHotEncoder(
                        sparse_output=False,
                        handle_unknown="ignore",
                        drop="first",
                    ),
                ),
            ]
        )
