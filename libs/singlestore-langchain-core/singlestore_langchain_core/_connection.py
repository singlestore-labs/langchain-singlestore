"""Connection-pool utilities for SingleStore integrations.

Two SQLAlchemy ``Pool`` implementations are provided:

* :class:`SingleConnectionPool` — always hands out the same, caller-owned
  connection. Useful when the caller manages the connection lifecycle itself
  (tests, notebooks, a shared long-lived connection).
* :class:`QueueConnectionPool` — a thin wrapper around
  :class:`sqlalchemy.pool.QueuePool` that lazily opens
  :func:`singlestoredb.connect` connections using a stored kwargs mapping.

Use :func:`create_connection_pool` as the single entry point; it picks the
right implementation based on the arguments the caller supplied.
"""

from typing import Any, Optional

from singlestoredb.connection import Connection, connect
from sqlalchemy.pool import Pool, QueuePool


class SingleConnectionPool(Pool):
    """Pool that returns the same pre-established connection on every call.

    The connection is owned by the caller; this class does not open or close
    it. Intended for scenarios where a single, long-lived connection is
    reused across operations.
    """

    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def connect(self) -> Any:  # type: ignore[override]
        return self._connection


class QueueConnectionPool(Pool):
    """Lazy, size-bounded pool backed by :class:`sqlalchemy.pool.QueuePool`.

    Connections are opened on demand via :func:`singlestoredb.connect` using
    ``connection_kwargs``. See the ``singlestoredb.connect`` documentation for
    the full list of accepted keyword arguments (``host``, ``user``,
    ``password``, ``port``, ``database``, TLS options, etc.).

    Args:
        pool_size: Number of persistent connections kept in the pool.
        max_overflow: Additional connections that may be opened beyond
            ``pool_size`` under load.
        timeout: Seconds to wait for a free connection before raising.
        connection_kwargs: Keyword arguments forwarded to
            :func:`singlestoredb.connect` when a new connection is opened.
    """

    def __init__(
        self,
        pool_size: int = 5,
        max_overflow: int = 10,
        timeout: float = 30,
        connection_kwargs: Optional[dict] = None,
    ) -> None:
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._timeout = timeout
        self._connection_kwargs: dict = dict(connection_kwargs or {})
        self._pool = QueuePool(
            self._open_connection,
            pool_size=self._pool_size,
            max_overflow=self._max_overflow,
            timeout=self._timeout,
        )

    def _open_connection(self) -> Any:
        return connect(**self._connection_kwargs)

    def connect(self) -> Any:  # type: ignore[override]
        return self._pool.connect()


def create_connection_pool(
    connection: Optional[Connection] = None,
    connection_pool: Optional[Pool] = None,
    pool_size: int = 5,
    max_overflow: int = 10,
    timeout: float = 30,
    connection_kwargs: Optional[dict] = None,
) -> Pool:
    """Return the connection pool that matches the supplied arguments.

    Dispatch rules (checked in order):

    1. If both ``connection`` and ``connection_pool`` are given, raise
       :class:`ValueError` — the caller must pick one.
    2. If ``connection`` is given, wrap it in a :class:`SingleConnectionPool`.
    3. If ``connection_pool`` is given, return it unchanged.
    4. Otherwise build a :class:`QueueConnectionPool` from ``pool_size``,
       ``max_overflow``, ``timeout`` and ``connection_kwargs``.

    See :class:`QueueConnectionPool` and :func:`singlestoredb.connect` for the
    meaning of the remaining arguments.
    """
    if connection is not None and connection_pool is not None:
        raise ValueError("Cannot specify both a connection and a connection pool.")

    if connection is not None:
        return SingleConnectionPool(connection)

    if connection_pool is not None:
        return connection_pool

    return QueueConnectionPool(
        pool_size=pool_size,
        max_overflow=max_overflow,
        timeout=timeout,
        connection_kwargs=connection_kwargs,
    )
