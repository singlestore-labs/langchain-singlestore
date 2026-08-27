"""Synchronous SingleStore-backed ``BaseCheckpointSaver``.

Placeholder: schema management and checkpoint I/O will be implemented on top
of ``singlestoredb`` using the pooled connection helpers in
``singlestore_langchain_core``.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator, Optional, Sequence

from langchain_core.runnables import RunnableConfig
from singlestore_langchain_core._utils import (
    DEFAULT_CONNECTOR_NAME,
    compute_connector_version,
    set_connector_attributes,
)

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)

_NOT_IMPLEMENTED = (
    "SingleStoreSaver is currently a scaffolding placeholder. "
    "Track progress in libs/langgraph-singlestore/CHANGELOG.md."
)


class SingleStoreSaver(BaseCheckpointSaver):
    """SingleStore-backed checkpoint saver (synchronous)."""

    def __init__(
        self,
        *,
        table_name: str = "checkpoints",
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

    def setup(self) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    # Async fallbacks delegate to the sync API by default; override in
    # AsyncSingleStoreSaver.
    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        raise NotImplementedError(_NOT_IMPLEMENTED)
        # Unreachable — keeps mypy happy about the async-generator return.
        yield  # type: ignore[unreachable]

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED)
