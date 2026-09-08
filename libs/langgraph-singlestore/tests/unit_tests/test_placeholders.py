"""Sanity tests for the langgraph-singlestore scaffolding.

The checkpoint saver is still a placeholder; the store draft is functional
against a mock connection pool.
"""

from unittest.mock import MagicMock

import pytest
from singlestore_langchain_core._connection import CallerOwnedConnectionPool
from sqlalchemy.pool import Pool

from langgraph.checkpoint.singlestore import (
    AsyncSingleStoreSaver,
    SingleStoreSaver,
)
from langgraph.store.base import PutOp
from langgraph.store.singlestore import AsyncSingleStoreStore, SingleStoreStore


def test_saver_placeholder_raises() -> None:
    saver = SingleStoreSaver(host="localhost")
    with pytest.raises(NotImplementedError):
        saver.setup()


def test_async_saver_is_subclass() -> None:
    assert issubclass(AsyncSingleStoreSaver, SingleStoreSaver)


def test_async_store_is_subclass() -> None:
    assert issubclass(AsyncSingleStoreStore, SingleStoreStore)


def _make_store() -> tuple[SingleStoreStore, MagicMock, MagicMock]:
    pool = MagicMock(spec=Pool)
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    pool.connect.return_value = conn
    store = SingleStoreStore(connection_pool=pool)
    return store, conn, cursor


def test_store_uses_injected_pool() -> None:
    pool = MagicMock(spec=Pool)
    store = SingleStoreStore(connection_pool=pool)
    assert isinstance(store.connection_pool, CallerOwnedConnectionPool)
    assert store.connection_pool._connection_pool is pool


def test_store_batch_empty_is_noop() -> None:
    store, _, cursor = _make_store()
    assert store.batch([]) == []
    cursor.execute.assert_not_called()


def test_store_batch_put_emits_upsert() -> None:
    store, _, cursor = _make_store()
    store.batch([PutOp(("users", "1"), "prefs", {"theme": "dark"})])
    assert cursor.execute.called
    sql = cursor.execute.call_args_list[0].args[0]
    assert "INSERT INTO store" in sql
    assert "ON DUPLICATE KEY UPDATE" in sql
