"""Shared SingleStore helpers used by ``langchain-singlestore`` and
``langgraph-singlestore``.

This package contains only SingleStore/SQL primitives with no dependency on
``langchain-core`` or ``langgraph``.  It is intended to be an internal
implementation detail; direct use by application code is not supported.
"""

from importlib import metadata

from singlestore_langchain_core._connection import (
    create_connection_pool,
)
from singlestore_langchain_core._filter import (
    FilterTypedDict,
    _get_match_param_function,
    _parse_filter,
)
from singlestore_langchain_core._utils import (
    DEFAULT_CONNECTOR_NAME,
    LANGGRAPH_CONNECTOR_NAME,
    DistanceStrategy,
    FullTextIndexVersion,
    FullTextScoringMode,
    compute_connector_version,
    hash,
    set_connector_attributes,
)

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = ""
del metadata

__all__ = [
    "DEFAULT_CONNECTOR_NAME",
    "LANGGRAPH_CONNECTOR_NAME",
    "DistanceStrategy",
    "FilterTypedDict",
    "FullTextIndexVersion",
    "FullTextScoringMode",
    "_get_match_param_function",
    "_parse_filter",
    "compute_connector_version",
    "hash",
    "set_connector_attributes",
    "create_connection_pool",
    "__version__",
]
