"""Asynchronous SingleStore-backed ``BaseCheckpointSaver``.

Placeholder: async I/O will be implemented on top of ``singlestoredb`` with a
dedicated async pool.
"""

from __future__ import annotations

from langgraph.checkpoint.singlestore.base import SingleStoreSaver

_NOT_IMPLEMENTED = (
    "AsyncSingleStoreSaver is currently a scaffolding placeholder. "
    "Track progress in libs/langgraph-singlestore/CHANGELOG.md."
)


class AsyncSingleStoreSaver(SingleStoreSaver):
    """Async variant of :class:`SingleStoreSaver`."""

    async def setup(self) -> None:  # type: ignore[override]
        raise NotImplementedError(_NOT_IMPLEMENTED)
