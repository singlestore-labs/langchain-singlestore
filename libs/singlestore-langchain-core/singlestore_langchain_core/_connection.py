"""Connection-pool utilities for SingleStore integrations.

Three SQLAlchemy ``Pool`` implementations are provided:

* :class:`SingleConnectionPool` — always hands out the same, caller-owned
  connection. Useful when the caller manages the connection lifecycle itself
  (tests, notebooks, a shared long-lived connection).
* :class:`QueueConnectionPool` — a thin wrapper around
  :class:`sqlalchemy.pool.QueuePool` that lazily opens
  :func:`singlestoredb.connect` connections using a stored kwargs mapping.
* :class:`CallerOwnedConnectionPool` — delegates to a pool supplied by the
  caller and treats :meth:`dispose` as a no-op, so the caller keeps ownership
  of the wrapped pool's lifecycle.

Use :func:`create_connection_pool` as the single entry point; it picks the
right implementation based on the arguments the caller supplied.
"""

from typing import Any, Optional

from singlestoredb.connection import Connection, connect
from sqlalchemy.pool import Pool, QueuePool


class _CallerOwnedConnection:
    """DBAPI-like proxy over a caller-owned :class:`Connection`.

    Consumers of a pool follow the ``connect()``/``close()`` idiom, but for
    :class:`SingleConnectionPool` the underlying connection is owned by the
    caller and must outlive the checkout. This proxy forwards every attribute
    to the wrapped connection while making ``close()`` a no-op so subsequent
    checkouts keep working.
    """

    __slots__ = ("_connection",)

    def __init__(self, connection: Connection) -> None:
        object.__setattr__(self, "_connection", connection)

    def close(self) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        if name == "_connection":
            raise AttributeError(name)
        return getattr(self._connection, name)

    def __enter__(self) -> Connection:
        return self._connection

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


class SingleConnectionPool(Pool):
    """Pool that returns the same pre-established connection on every call.

    The connection is owned by the caller; this class does not open or close
    it. Intended for scenarios where a single, long-lived connection is
    reused across operations. Each :meth:`connect` returns a lightweight
    proxy so callers may follow the standard ``connect()``/``close()``
    idiom without tearing down the shared connection.
    """

    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def connect(self) -> Any:  # type: ignore[override]
        return _CallerOwnedConnection(self._connection)

    def dispose(self) -> None:  # type: ignore[override]
        # Caller owns the underlying connection; nothing to release here.
        return None


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

    def dispose(self) -> None:  # type: ignore[override]
        self._pool.dispose()


class CallerOwnedConnectionPool(Pool):
    """Pool wrapper that delegates to a caller-owned pool without disposing it.

    Forwards :meth:`connect` to the wrapped pool but treats :meth:`dispose`
    as a no-op, so the caller keeps full ownership of the pool's lifecycle.
    This mirrors the semantics of :class:`_CallerOwnedConnection` at the pool
    level.
    """

    def __init__(self, connection_pool: Pool) -> None:
        self._connection_pool = connection_pool

    def connect(self) -> Any:  # type: ignore[override]
        return self._connection_pool.connect()

    def dispose(self) -> None:  # type: ignore[override]
        # Caller owns the wrapped pool; nothing to release here.
        return None


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
    3. If ``connection_pool`` is given, wrap it in a
       :class:`CallerOwnedConnectionPool` so ``dispose()`` on the returned
       pool doesn't tear down the caller's pool.
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
        return CallerOwnedConnectionPool(connection_pool)

    return QueueConnectionPool(
        pool_size=pool_size,
        max_overflow=max_overflow,
        timeout=timeout,
        connection_kwargs=connection_kwargs,
    )
