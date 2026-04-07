import hashlib
from enum import Enum
from importlib.metadata import PackageNotFoundError, version

# Connection attributes
CONNECTOR_NAME = "langchain python sdk"


def _get_connector_version() -> str:
    """Get connector version by adding 2 to the package major version."""
    try:
        pkg_version = version("langchain-singlestore")
        # Parse version (e.g., "1.1.0" -> ["1", "1", "0"])
        version_parts = pkg_version.split(".")
        # Add 2 to major version: 1.1.0 -> 3.1.0
        major_version = int(version_parts[0]) + 2
        version_parts[0] = str(major_version)
        return ".".join(version_parts)
    except PackageNotFoundError:
        # Fallback if package is not installed
        return "3.0.0"


CONNECTOR_VERSION = _get_connector_version()


def set_connector_attributes(connection_kwargs: dict) -> None:
    """Set connector name and version in connection attributes.

    Args:
        connection_kwargs: Dictionary containing connection keyword arguments.
            Will be modified in-place to add _connector_name and _connector_version.
    """
    if "conn_attrs" not in connection_kwargs:
        connection_kwargs["conn_attrs"] = {}

    connection_kwargs["conn_attrs"]["_connector_name"] = CONNECTOR_NAME
    connection_kwargs["conn_attrs"]["_connector_version"] = CONNECTOR_VERSION


class DistanceStrategy(str, Enum):
    """Distance strategies for calculating similarity between vectors.

    Attributes:
        EUCLIDEAN_DISTANCE: Computes the Euclidean (L2) distance between vectors.
            Lower scores indicate more similar vectors. Not compatible with
            WEIGHTED_SUM search strategy.
        DOT_PRODUCT: Computes the dot product (inner product) between vectors.
            Higher scores indicate more similar vectors. This is the default
            and recommended strategy for most embedding models.
    """

    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
    DOT_PRODUCT = "DOT_PRODUCT"


class FullTextIndexVersion(str, Enum):
    """Full-text index versions supported by SingleStore.

    Attributes:
        V1: Original full-text index implementation. Compatible with all
            SingleStore versions that support full-text search. Only supports
            MATCH scoring mode.
        V2: New full-text index implementation available in SingleStore 8.7+.
            Offers improved performance and supports additional scoring modes
            (BM25, BM25_GLOBAL).
    """

    V1 = "V1"
    V2 = "V2"


class FullTextScoringMode(str, Enum):
    """Scoring algorithms for full-text search ranking.

    Attributes:
        MATCH: Uses SingleStore's native MATCH() AGAINST() function.
            Compatible with both V1 and V2 full-text indexes.
        BM25: Best Matching 25 algorithm with TF-IDF scoring and document
            length normalization. Requires V2 full-text index.
        BM25_GLOBAL: BM25 with global IDF statistics across all partitions.
            Provides consistent scoring in distributed environments.
            Requires V2 full-text index.
    """

    MATCH = "MATCH"
    BM25 = "BM25"
    BM25_GLOBAL = "BM25_GLOBAL"


def hash(_input: str) -> str:
    """Use a deterministic hashing approach."""
    return hashlib.md5(_input.encode()).hexdigest()
