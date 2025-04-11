"""Utility functions for working with vectors and vectorstores."""

from enum import Enum
import hashlib

class DistanceStrategy(str, Enum):
    """Enumerator of the Distance strategies for calculating distances
    between vectors."""

    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
    DOT_PRODUCT = "DOT_PRODUCT"

def hash(_input: str) -> str:
    """Use a deterministic hashing approach."""
    return hashlib.md5(_input.encode()).hexdigest()
