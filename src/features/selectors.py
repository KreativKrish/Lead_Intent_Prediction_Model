"""Feature selection utilities."""

from sklearn.feature_selection import VarianceThreshold

from ..utils.config import get_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FeatureSelector:
    """Select features based on variance and other criteria."""

    def __init__(self):
        self.config = get_config()

    def get_selector(self):
        """Get feature selector.

        Returns:
            Feature selector transformer (e.g., VarianceThreshold).
        """
        threshold = self.config.get("features.feature_selection.threshold", 0.01)
        logger.info(f"Using VarianceThreshold with threshold={threshold}")
        return VarianceThreshold(threshold=threshold)
