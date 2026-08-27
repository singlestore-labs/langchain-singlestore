"""Backwards-compatibility shim.

The implementation now lives in ``singlestore_langchain_core._utils``.
This module keeps the historical import path ``langchain_singlestore._utils``
working and binds ``CONNECTOR_NAME`` / ``CONNECTOR_VERSION`` to values
specific to the ``langchain-singlestore`` package.
"""

from singlestore_langchain_core._utils import (
    DEFAULT_CONNECTOR_NAME,
    DistanceStrategy,
    FullTextIndexVersion,
    FullTextScoringMode,
    compute_connector_version,
    hash,
)
from singlestore_langchain_core._utils import (
    set_connector_attributes as _set_connector_attributes,
)

CONNECTOR_NAME = DEFAULT_CONNECTOR_NAME
CONNECTOR_VERSION = compute_connector_version("langchain-singlestore")


def set_connector_attributes(connection_kwargs: dict) -> None:
    """Set connector name and version in connection attributes."""
    _set_connector_attributes(
        connection_kwargs,
        connector_name=CONNECTOR_NAME,
        connector_version=CONNECTOR_VERSION,
    )


__all__ = [
    "CONNECTOR_NAME",
    "CONNECTOR_VERSION",
    "DistanceStrategy",
    "FullTextIndexVersion",
    "FullTextScoringMode",
    "hash",
    "set_connector_attributes",
]
