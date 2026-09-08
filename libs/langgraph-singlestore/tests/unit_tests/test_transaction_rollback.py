"""Tests that transactional get/search paths ROLLBACK on error.

The ``_batch_get_ops`` and ``_batch_search_ops`` refresh-TTL branches issue
``BEGIN`` / ``COMMIT`` around a SELECT ... FOR UPDATE plus UPDATE pair. When a
statement between ``BEGIN`` and ``COMMIT`` fails, the code must issue
``ROLLBACK`` before the exception propagates -- otherwise a caller-owned
connection (which ``_CursorContext.__exit__`` cannot close) is left inside an
open transaction.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy.pool import Pool

from langgraph.store.base import GetOp, SearchOp
from langgraph.store.singlestore import SingleStoreStore


def _make_store() -> tuple[SingleStoreStore, MagicMock]:
    pool = MagicMock(spec=Pool)
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    pool.connect.return_value = conn
    store = SingleStoreStore(connection_pool=pool)
    return store, cursor


def _executed_sqls(cursor: MagicMock) -> list[str]:
    return [call.args[0].strip().upper() for call in cursor.execute.call_args_list]


class TestBatchGetOpsRollback:
    def test_select_failure_inside_transaction_rolls_back(self) -> None:
        store, cursor = _make_store()

        def execute(sql: str, *args: Any, **kwargs: Any) -> None:
            if "SELECT" in sql.upper():
                raise RuntimeError("boom")

        cursor.execute.side_effect = execute

        with pytest.raises(RuntimeError, match="boom"):
            store.batch([GetOp(("ns",), "k", refresh_ttl=True)])

        sqls = _executed_sqls(cursor)
        assert "BEGIN" in sqls
        assert "ROLLBACK" in sqls
        assert "COMMIT" not in sqls
        assert sqls.index("BEGIN") < sqls.index("ROLLBACK")

    def test_ttl_update_failure_inside_transaction_rolls_back(self) -> None:
        store, cursor = _make_store()
        cursor.fetchall.return_value = []

        def execute(sql: str, *args: Any, **kwargs: Any) -> None:
            if sql.strip().upper().startswith("UPDATE"):
                raise RuntimeError("update failed")

        cursor.execute.side_effect = execute

        with pytest.raises(RuntimeError, match="update failed"):
            store.batch([GetOp(("ns",), "k", refresh_ttl=True)])

        sqls = _executed_sqls(cursor)
        assert "BEGIN" in sqls
        assert "ROLLBACK" in sqls
        assert "COMMIT" not in sqls

    def test_success_path_commits_without_rollback(self) -> None:
        store, cursor = _make_store()
        cursor.fetchall.return_value = []

        store.batch([GetOp(("ns",), "k", refresh_ttl=True)])

        sqls = _executed_sqls(cursor)
        assert "BEGIN" in sqls
        assert "COMMIT" in sqls
        assert "ROLLBACK" not in sqls

    def test_no_refresh_ttl_does_not_open_transaction(self) -> None:
        store, cursor = _make_store()
        cursor.fetchall.return_value = []

        store.batch([GetOp(("ns",), "k", refresh_ttl=False)])

        sqls = _executed_sqls(cursor)
        assert "BEGIN" not in sqls
        assert "COMMIT" not in sqls
        assert "ROLLBACK" not in sqls


class TestBatchSearchOpsRollback:
    def test_select_failure_inside_transaction_rolls_back(self) -> None:
        store, cursor = _make_store()

        def execute(sql: str, *args: Any, **kwargs: Any) -> None:
            if "SELECT" in sql.upper():
                raise RuntimeError("boom")

        cursor.execute.side_effect = execute

        with pytest.raises(RuntimeError, match="boom"):
            store.batch([SearchOp(("ns",), refresh_ttl=True)])

        sqls = _executed_sqls(cursor)
        assert "BEGIN" in sqls
        assert "ROLLBACK" in sqls
        assert "COMMIT" not in sqls

    def test_ttl_update_failure_inside_transaction_rolls_back(self) -> None:
        store, cursor = _make_store()
        cursor.fetchall.return_value = []

        def execute(sql: str, *args: Any, **kwargs: Any) -> None:
            if sql.strip().upper().startswith("UPDATE"):
                raise RuntimeError("update failed")

        cursor.execute.side_effect = execute

        with pytest.raises(RuntimeError, match="update failed"):
            store.batch([SearchOp(("ns",), refresh_ttl=True)])

        sqls = _executed_sqls(cursor)
        assert "BEGIN" in sqls
        assert "ROLLBACK" in sqls
        assert "COMMIT" not in sqls

    def test_success_path_commits_without_rollback(self) -> None:
        store, cursor = _make_store()
        cursor.fetchall.return_value = []

        store.batch([SearchOp(("ns",), refresh_ttl=True)])

        sqls = _executed_sqls(cursor)
        assert "BEGIN" in sqls
        assert "COMMIT" in sqls
        assert "ROLLBACK" not in sqls

    def test_no_refresh_ttl_does_not_open_transaction(self) -> None:
        store, cursor = _make_store()
        cursor.fetchall.return_value = []

        store.batch([SearchOp(("ns",), refresh_ttl=False)])

        sqls = _executed_sqls(cursor)
        assert "BEGIN" not in sqls
        assert "COMMIT" not in sqls
        assert "ROLLBACK" not in sqls
