"""
This module provides the connection utilities for SingleStore within the LangChain framework.
"""

from typing import Optional

from singlestoredb.connection import Connection, connect
from sqlalchemy.pool import Pool, QueuePool


class SingleConnectionPool(Pool):
    def __init__(self, connection: Optional[Connection] = None):
        self._connection = connection

    def connect(self):
        return self._connection


class DefaultConnectionPool(Pool):
    def __init__(
        self,
        pool_size: int = 5,
        max_overflow: int = 10,
        timeout: float = 30,
        connection_kwargs: Optional[dict] = None,
    ):
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._timeout = timeout
        self._connection_kwargs = connection_kwargs or {}
        self._pool = QueuePool(
            self._get_connection,
            pool_size=self._pool_size,
            max_overflow=self._max_overflow,
            timeout=self._timeout,
        )

    def _get_connection(self):
        return connect(**self._connection_kwargs)

    def connect(self):
        return self._pool.connect()


def create_connection_pool(
    connection: Optional[Connection] = None,
    connection_pool: Optional[Pool] = None,
    pool_size: int = 5,
    max_overflow: int = 10,
    timeout: float = 30,
    connection_kwargs: Optional[dict] = None,
) -> Pool:
    if connection is not None and connection_pool is not None:
        raise ValueError("Cannot specify both a connection and a connection pool.")
    if connection is not None:
        return SingleConnectionPool(connection)
    if connection_pool is not None:
        return connection_pool
    return DefaultConnectionPool(
        pool_size=pool_size,
        max_overflow=max_overflow,
        timeout=timeout,
        connection_kwargs=connection_kwargs,
    )
