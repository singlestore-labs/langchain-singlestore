"""Utility functions for working with vectors and vectorstores."""

from enum import Enum

class DistanceStrategy(str, Enum):
    """Enumerator of the Distance strategies for calculating distances
    between vectors."""

    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
    DOT_PRODUCT = "DOT_PRODUCT"
