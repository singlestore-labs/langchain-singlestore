"""Regression tests for the ``_CursorContext`` lock-leak fix.

If ``pool.connect()`` or ``conn.cursor()`` raises inside ``__enter__``,
Python does not invoke ``__exit__`` -- the context manager must release
``self._lock`` and close any partial connection itself, otherwise the
saver/store deadlocks on the next call.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from singlestoredb.connection import Connection
from sqlalchemy.pool import Pool

from langgraph_singlestore.checkpoint import SingleStoreSaver
from langgraph_singlestore.store import SingleStoreStore


class _Boom(RuntimeError):
    pass


def _make_pool(
    *,
    connect_raises: bool = False,
    cursor_raises: bool = False,
) -> tuple[MagicMock, MagicMock]:
    pool = MagicMock(spec=Pool)
    conn = MagicMock(spec=Connection)
    if connect_raises:
        pool.connect.side_effect = _Boom("connect failed")
    else:
        pool.connect.return_value = conn
    if cursor_raises:
        conn.cursor.side_effect = _Boom("cursor failed")
    else:
        conn.cursor.return_value = MagicMock()
    return pool, conn


class TestSaverCursorContextLockLeak:
    def test_connect_failure_releases_lock(self) -> None:
        pool, _ = _make_pool(connect_raises=True)
        saver = SingleStoreSaver(connection_pool=pool)

        with pytest.raises(_Boom):
            with saver._cursor():
                pass

        assert not saver._lock.locked(), "lock leaked after connect() failure"

    def test_cursor_failure_releases_lock_and_closes_connection(self) -> None:
        pool, conn = _make_pool(cursor_raises=True)
        saver = SingleStoreSaver(connection_pool=pool)

        with pytest.raises(_Boom):
            with saver._cursor():
                pass

        assert not saver._lock.locked(), "lock leaked after cursor() failure"
        conn.close.assert_called_once()

    def test_lock_reusable_after_failure(self) -> None:
        pool, _ = _make_pool(connect_raises=True)
        saver = SingleStoreSaver(connection_pool=pool)

        for _ in range(3):
            with pytest.raises(_Boom):
                with saver._cursor():
                    pass
        assert not saver._lock.locked()


class TestStoreCursorContextLockLeak:
    def test_connect_failure_releases_lock(self) -> None:
        pool, _ = _make_pool(connect_raises=True)
        store = SingleStoreStore(connection_pool=pool)

        with pytest.raises(_Boom):
            with store._cursor():
                pass

        assert not store._lock.locked(), "lock leaked after connect() failure"

    def test_cursor_failure_releases_lock_and_closes_connection(self) -> None:
        pool, conn = _make_pool(cursor_raises=True)
        store = SingleStoreStore(connection_pool=pool)

        with pytest.raises(_Boom):
            with store._cursor():
                pass

        assert not store._lock.locked(), "lock leaked after cursor() failure"
        conn.close.assert_called_once()
