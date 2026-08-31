"""Integration tests for singlestore_langchain_core._connection.

These tests exercise the connection pool helpers against a real SingleStore
instance started via Docker (see ``conftest.py``).
"""

from typing import Any

from sqlalchemy.pool import QueuePool

from singlestore_langchain_core._connection import (
    QueueConnectionPool,
    SingleConnectionPool,
    create_connection_pool,
)

from .conftest import ConnectionParameters


def _fetch_one(conn: Any) -> Any:
    cur = conn.cursor()
    try:
        cur.execute("select 1")
        return cur.fetchone()
    finally:
        cur.close()


class TestSingleConnectionPoolIntegration:
    def test_returns_wrapped_live_connection(self, raw_connection: Any) -> None:
        pool = SingleConnectionPool(raw_connection)
        conn = pool.connect()
        assert conn is raw_connection
        row = _fetch_one(conn)
        assert row[0] == 1

    def test_via_factory(self, raw_connection: Any) -> None:
        pool = create_connection_pool(connection=raw_connection)
        assert isinstance(pool, SingleConnectionPool)
        assert pool.connect() is raw_connection
        assert _fetch_one(pool.connect())[0] == 1


class TestDefaultConnectionPoolIntegration:
    def test_connect_yields_working_connection(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        pool = QueueConnectionPool(
            pool_size=2,
            max_overflow=1,
            timeout=10,
            connection_kwargs=connection_parameters.as_kwargs(),
        )
        assert isinstance(pool._pool, QueuePool)
        proxy = pool.connect()
        try:
            assert _fetch_one(proxy)[0] == 1
        finally:
            proxy.close()

    def test_pool_reuses_connections(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        pool = QueueConnectionPool(
            pool_size=1,
            max_overflow=0,
            timeout=10,
            connection_kwargs=connection_parameters.as_kwargs(),
        )
        proxy1 = pool.connect()
        assert _fetch_one(proxy1)[0] == 1
        proxy1.close()

        proxy2 = pool.connect()
        try:
            assert _fetch_one(proxy2)[0] == 1
        finally:
            proxy2.close()

    def test_via_factory(self, connection_parameters: ConnectionParameters) -> None:
        pool = create_connection_pool(
            pool_size=2,
            max_overflow=0,
            timeout=10,
            connection_kwargs=connection_parameters.as_kwargs(),
        )
        assert isinstance(pool, QueueConnectionPool)
        proxy = pool.connect()
        try:
            assert _fetch_one(proxy)[0] == 1
        finally:
            proxy.close()


class TestCreateConnectionPoolIntegration:
    def test_forwards_provided_pool(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        inner = QueueConnectionPool(connection_kwargs=connection_parameters.as_kwargs())
        result = create_connection_pool(connection_pool=inner)
        assert result is inner
        proxy = result.connect()
        try:
            assert _fetch_one(proxy)[0] == 1
        finally:
            proxy.close()
