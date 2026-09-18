"""SingleStore integrations for LangGraph.

Top-level re-exports for the checkpoint saver and long-term memory store.
The underlying modules live under
``langgraph_singlestore.checkpoint`` and
``langgraph_singlestore.store``.
"""

from importlib import metadata

from langgraph_singlestore.checkpoint import (
    AsyncSingleStoreSaver,
    SingleStoreSaver,
)
from langgraph_singlestore.store import (
    AsyncSingleStoreStore,
    SingleStoreStore,
)
from langgraph_singlestore.store.base import SingleStoreIndexConfig

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = ""
del metadata

__all__ = [
    "AsyncSingleStoreSaver",
    "AsyncSingleStoreStore",
    "SingleStoreIndexConfig",
    "SingleStoreSaver",
    "SingleStoreStore",
    "__version__",
]
