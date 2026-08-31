"""Unit tests for singlestore_langchain_core._connection."""

import unittest
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.pool import Pool, QueuePool

from singlestore_langchain_core._connection import (
    QueueConnectionPool,
    SingleConnectionPool,
    create_connection_pool,
)


class TestSingleConnectionPool(unittest.TestCase):
    def test_connect_forwards_to_wrapped_connection(self) -> None:
        conn = MagicMock(name="connection")
        pool = SingleConnectionPool(conn)
        proxy = pool.connect()
        # attribute access is forwarded
        assert proxy.cursor() is conn.cursor.return_value
        conn.cursor.assert_called_once_with()

    def test_is_pool_subclass(self) -> None:
        assert isinstance(SingleConnectionPool(MagicMock()), Pool)

    def test_close_does_not_close_underlying_connection(self) -> None:
        conn = MagicMock(name="connection")
        pool = SingleConnectionPool(conn)
        proxy = pool.connect()
        proxy.close()
        conn.close.assert_not_called()

    def test_multiple_checkouts_share_underlying_connection(self) -> None:
        conn = MagicMock(name="connection")
        pool = SingleConnectionPool(conn)
        p1 = pool.connect()
        p1.close()
        p2 = pool.connect()
        # Both proxies target the same live connection.
        assert p1.cursor() is conn.cursor.return_value
        assert p2.cursor() is conn.cursor.return_value
        conn.close.assert_not_called()

    def test_context_manager_yields_connection_and_does_not_close(self) -> None:
        conn = MagicMock(name="connection")
        pool = SingleConnectionPool(conn)
        with pool.connect() as entered:
            assert entered is conn
        conn.close.assert_not_called()

    def test_dispose_is_noop(self) -> None:
        conn = MagicMock(name="connection")
        pool = SingleConnectionPool(conn)
        pool.dispose()
        conn.close.assert_not_called()


class TestDefaultConnectionPool(unittest.TestCase):
    def test_stores_configuration(self) -> None:
        kwargs = {"host": "example", "port": 3306}
        pool = QueueConnectionPool(
            pool_size=7,
            max_overflow=3,
            timeout=15.0,
            connection_kwargs=kwargs,
        )
        assert pool._pool_size == 7
        assert pool._max_overflow == 3
        assert pool._timeout == 15.0
        assert pool._connection_kwargs == kwargs

    def test_default_kwargs(self) -> None:
        pool = QueueConnectionPool()
        assert pool._pool_size == 5
        assert pool._max_overflow == 10
        assert pool._timeout == 30
        assert pool._connection_kwargs == {}

    def test_none_connection_kwargs_becomes_empty_dict(self) -> None:
        pool = QueueConnectionPool(connection_kwargs=None)
        assert pool._connection_kwargs == {}

    def test_inner_pool_is_queue_pool(self) -> None:
        pool = QueueConnectionPool()
        assert isinstance(pool._pool, QueuePool)

    def test_is_pool_subclass(self) -> None:
        assert isinstance(QueueConnectionPool(), Pool)

    def test_create_connection_calls_singlestoredb_connect(self) -> None:
        kwargs = {"host": "h", "user": "u"}
        with patch("singlestore_langchain_core._connection.connect") as mock_connect:
            mock_connect.return_value = MagicMock(name="conn")
            pool = QueueConnectionPool(connection_kwargs=kwargs)
            result = pool._open_connection()
            mock_connect.assert_called_once_with(**kwargs)
            assert result is mock_connect.return_value

    def test_connection_kwargs_are_copied(self) -> None:
        kwargs = {"host": "h"}
        pool = QueueConnectionPool(connection_kwargs=kwargs)
        kwargs["host"] = "other"
        assert pool._connection_kwargs == {"host": "h"}

    def test_connect_delegates_to_inner_pool(self) -> None:
        pool = QueueConnectionPool()
        with patch.object(pool._pool, "connect") as mock_connect:
            mock_connect.return_value = MagicMock(name="conn_proxy")
            result = pool.connect()
            mock_connect.assert_called_once_with()
            assert result is mock_connect.return_value

    def test_dispose_delegates_to_inner_pool(self) -> None:
        pool = QueueConnectionPool()
        with patch.object(pool._pool, "dispose") as mock_dispose:
            pool.dispose()
            mock_dispose.assert_called_once_with()


class TestCreateConnectionPool(unittest.TestCase):
    def test_raises_when_both_connection_and_pool_provided(self) -> None:
        conn = MagicMock(name="connection")
        provided_pool = MagicMock(spec=Pool)
        with pytest.raises(ValueError, match="Cannot specify both"):
            create_connection_pool(connection=conn, connection_pool=provided_pool)

    def test_returns_single_connection_pool_when_connection_given(self) -> None:
        conn = MagicMock(name="connection")
        pool = create_connection_pool(connection=conn)
        assert isinstance(pool, SingleConnectionPool)
        # Proxy forwards to the wrapped connection.
        assert pool.connect().cursor() is conn.cursor.return_value

    def test_returns_provided_pool_as_is(self) -> None:
        provided_pool = MagicMock(spec=Pool)
        result = create_connection_pool(connection_pool=provided_pool)
        assert result is provided_pool

    def test_returns_default_pool_with_no_arguments(self) -> None:
        pool = create_connection_pool()
        assert isinstance(pool, QueueConnectionPool)
        assert pool._pool_size == 5
        assert pool._max_overflow == 10
        assert pool._timeout == 30
        assert pool._connection_kwargs == {}

    def test_forwards_pool_parameters_to_default_pool(self) -> None:
        kwargs = {"host": "example", "database": "db"}
        pool = create_connection_pool(
            pool_size=2,
            max_overflow=4,
            timeout=1.5,
            connection_kwargs=kwargs,
        )
        assert isinstance(pool, QueueConnectionPool)
        assert pool._pool_size == 2
        assert pool._max_overflow == 4
        assert pool._timeout == 1.5
        assert pool._connection_kwargs == kwargs
