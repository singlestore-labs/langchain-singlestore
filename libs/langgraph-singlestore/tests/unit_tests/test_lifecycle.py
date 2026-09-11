"""Unit tests for :class:`SingleStoreStore` construction and lifecycle.

Covers the connection-injection dispatch, ``close()`` semantics against
caller-owned resources, the ``stop_ttl_sweeper`` timeout branch, and the
connector-attribute injection on the plain-kwargs path.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

import pytest
from singlestore_langchain_core import LANGGRAPH_CONNECTOR_NAME
from singlestore_langchain_core._connection import (
    CallerOwnedConnectionPool,
    SingleConnectionPool,
)
from singlestoredb.connection import Connection
from sqlalchemy.pool import Pool

from langgraph.store.singlestore import SingleStoreStore


class TestConnectionInjectionDispatch:
    def test_connection_and_pool_are_mutually_exclusive(self) -> None:
        conn = MagicMock(spec=Connection)
        pool = MagicMock(spec=Pool)
        with pytest.raises(ValueError, match="both"):
            SingleStoreStore(connection=conn, connection_pool=pool)

    def test_connection_arg_wraps_in_single_connection_pool(self) -> None:
        conn = MagicMock(spec=Connection)
        store = SingleStoreStore(connection=conn)
        assert isinstance(store.connection_pool, SingleConnectionPool)
        assert store.connection_pool._connection is conn

    def test_connection_pool_arg_wraps_in_caller_owned_pool(self) -> None:
        pool = MagicMock(spec=Pool)
        store = SingleStoreStore(connection_pool=pool)
        assert isinstance(store.connection_pool, CallerOwnedConnectionPool)
        assert store.connection_pool._connection_pool is pool


class TestConnectorAttributesInjection:
    """``__init__`` mutates ``connection_kwargs`` with the connector name and
    version so downstream ``singlestoredb.connect`` calls advertise the
    langgraph connector."""

    def test_kwargs_receive_connector_name_and_version(self) -> None:
        # Use a caller-supplied pool so no real connection is opened, then
        # inspect the recorded kwargs.
        pool = MagicMock(spec=Pool)
        store = SingleStoreStore(
            connection_pool=pool,
            host="ignored",
            user="ignored",
        )
        conn_attrs = store.connection_kwargs.get("conn_attrs") or {}
        assert conn_attrs.get("_connector_name") == LANGGRAPH_CONNECTOR_NAME
        assert isinstance(conn_attrs.get("_connector_version"), str)
        assert conn_attrs["_connector_version"]


class TestCloseWithCallerOwnedResources:
    def test_close_does_not_dispose_caller_owned_pool(self) -> None:
        pool = MagicMock(spec=Pool)
        store = SingleStoreStore(connection_pool=pool)
        store.close()
        pool.dispose.assert_not_called()

    def test_close_is_idempotent(self) -> None:
        pool = MagicMock(spec=Pool)
        store = SingleStoreStore(connection_pool=pool)
        store.close()
        store.close()
        pool.dispose.assert_not_called()

    def test_close_does_not_close_caller_owned_connection(self) -> None:
        conn = MagicMock(spec=Connection)
        store = SingleStoreStore(connection=conn)
        store.close()
        conn.close.assert_not_called()


class TestStopTtlSweeperTimeoutBranch:
    """When the sweeper thread doesn't shut down within ``timeout``,
    ``stop_ttl_sweeper`` must return ``False`` and preserve the thread
    reference so the caller can retry."""

    def test_timeout_returns_false_and_preserves_thread(self) -> None:
        pool = MagicMock(spec=Pool)
        store = SingleStoreStore(connection_pool=pool)

        fake_thread = MagicMock(spec=threading.Thread)
        # ``is_alive`` is polled twice: once at the top of stop_ttl_sweeper
        # and once after ``join`` returns. Keep it alive throughout to
        # simulate a thread that ignored the stop event within the timeout.
        fake_thread.is_alive.return_value = True
        fake_future: Any = object()
        store._ttl_sweeper_thread = fake_thread
        store._ttl_sweeper_future = fake_future  # type: ignore[assignment]

        assert store.stop_ttl_sweeper(timeout=0.01) is False
        fake_thread.join.assert_called_once_with(0.01)
        assert store._ttl_sweeper_thread is fake_thread
        assert store._ttl_sweeper_future is fake_future
        assert store._ttl_stop_event.is_set()

    def test_success_clears_thread_and_future(self) -> None:
        pool = MagicMock(spec=Pool)
        store = SingleStoreStore(connection_pool=pool)

        fake_thread = MagicMock(spec=threading.Thread)
        # Alive at the entry check, dead after ``join`` -- the normal
        # shutdown sequence.
        fake_thread.is_alive.side_effect = [True, False]
        store._ttl_sweeper_thread = fake_thread
        store._ttl_sweeper_future = object()  # type: ignore[assignment]

        assert store.stop_ttl_sweeper(timeout=1.0) is True
        assert store._ttl_sweeper_thread is None
        assert store._ttl_sweeper_future is None
