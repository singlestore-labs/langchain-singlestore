"""SingleStore connection helpers and shared enums.

No LangChain / LangGraph imports live in this module so it can be shared by
every SingleStore integration package.
"""

import hashlib
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from typing import Optional

DEFAULT_CONNECTOR_NAME = "langchain python sdk"


def compute_connector_version(package_name: str, *, fallback: str = "3.0.0") -> str:
    """Return the connector version to advertise to SingleStore.

    Historically the connector version is the package version with ``2`` added
    to the major component (``1.5.0`` -> ``3.5.0``).  This preserves the wire
    contract that pre-monorepo releases used.
    """
    try:
        pkg_version = version(package_name)
        version_parts = pkg_version.split(".")
        major_version = int(version_parts[0]) + 2
        version_parts[0] = str(major_version)
        return ".".join(version_parts)
    except (PackageNotFoundError, ValueError):
        return fallback


def set_connector_attributes(
    connection_kwargs: dict,
    *,
    connector_name: str = DEFAULT_CONNECTOR_NAME,
    connector_version: Optional[str] = None,
) -> None:
    """Stamp connector identity onto ``connection_kwargs['conn_attrs']``.

    ``connector_version`` may be ``None`` when the caller has no versioned
    package to report; the attribute is simply omitted in that case.
    """
    if "conn_attrs" not in connection_kwargs:
        connection_kwargs["conn_attrs"] = {}

    connection_kwargs["conn_attrs"]["_connector_name"] = connector_name
    if connector_version is not None:
        connection_kwargs["conn_attrs"]["_connector_version"] = connector_version


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
