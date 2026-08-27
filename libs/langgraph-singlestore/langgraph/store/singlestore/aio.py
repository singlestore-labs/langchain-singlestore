"""Asynchronous SingleStore-backed ``BaseStore``.

Placeholder for the async I/O path.
"""

from __future__ import annotations

from langgraph.store.singlestore.base import SingleStoreStore

_NOT_IMPLEMENTED = (
    "AsyncSingleStoreStore is currently a scaffolding placeholder. "
    "Track progress in libs/langgraph-singlestore/CHANGELOG.md."
)


class AsyncSingleStoreStore(SingleStoreStore):
    """Async variant of :class:`SingleStoreStore`."""

    async def setup(self) -> None:  # type: ignore[override]
        raise NotImplementedError(_NOT_IMPLEMENTED)
