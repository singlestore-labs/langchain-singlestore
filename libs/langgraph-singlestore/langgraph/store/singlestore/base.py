"""Synchronous SingleStore-backed ``BaseStore``.

Placeholder: key/value + optional vector search will be implemented on top of
``singlestoredb`` and the shared vector primitives.
"""

from __future__ import annotations

from typing import Any, Iterable

from singlestore_langchain_core._utils import (
    DEFAULT_CONNECTOR_NAME,
    compute_connector_version,
    set_connector_attributes,
)

from langgraph.store.base import BaseStore, Op, Result

_NOT_IMPLEMENTED = (
    "SingleStoreStore is currently a scaffolding placeholder. "
    "Track progress in libs/langgraph-singlestore/CHANGELOG.md."
)


class SingleStoreStore(BaseStore):
    """SingleStore-backed store (synchronous)."""

    def __init__(
        self,
        *,
        table_name: str = "store",
        pool_size: int = 5,
        max_overflow: int = 10,
        timeout: float = 30,
        **connection_kwargs: Any,
    ) -> None:
        super().__init__()
        self.table_name = table_name
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.timeout = timeout
        set_connector_attributes(
            connection_kwargs,
            connector_name=DEFAULT_CONNECTOR_NAME,
            connector_version=compute_connector_version("langgraph-singlestore"),
        )
        self.connection_kwargs = connection_kwargs

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def setup(self) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED)
