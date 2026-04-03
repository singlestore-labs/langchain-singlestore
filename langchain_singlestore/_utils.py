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
    """Enumerator of the Distance strategies for calculating distances
    between vectors."""

    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
    DOT_PRODUCT = "DOT_PRODUCT"


class FullTextIndexVersion(str, Enum):
    """Enumerator of the Full-Text Index versions"""

    V1 = "V1"
    V2 = "V2"

def hash(_input: str) -> str:
    """Use a deterministic hashing approach."""
    return hashlib.md5(_input.encode()).hexdigest()
