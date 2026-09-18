"""Regression tests for the multi-thread pending-sends migration path.

``list()`` results may span multiple threads (e.g. when called without a
``thread_id`` filter). Every legacy checkpoint (``v < 4``) must have its
parent ``TASKS`` writes migrated against its own thread, not just the
thread of the first row.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from singlestoredb.connection import Connection
from sqlalchemy.pool import Pool

from langgraph_singlestore.checkpoint import SingleStoreSaver


def _legacy_value(*, thread_id: str, parent_id: str, ns: str = "") -> dict[str, Any]:
    """Build a minimal row dict that ``_maybe_migrate_pending_sends`` accepts."""
    return {
        "thread_id": thread_id,
        "checkpoint_ns": ns,
        "checkpoint_id": f"ck-{parent_id}-child",
        "parent_checkpoint_id": parent_id,
        "checkpoint": {"v": 3, "channel_values": {}},
        "channel_values": [],
        "metadata": {},
        "pending_writes": [],
    }


def _make_saver_with_cursor() -> tuple[SingleStoreSaver, MagicMock]:
    pool = MagicMock(spec=Pool)
    conn = MagicMock(spec=Connection)
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    pool.connect.return_value = conn
    saver = SingleStoreSaver(connection_pool=pool)
    return saver, cursor


class TestMultiThreadPendingSendsMigration:
    def test_each_thread_gets_its_own_sends_query(self) -> None:
        saver, cursor = _make_saver_with_cursor()
        cursor.fetchall.return_value = []

        values = [
            _legacy_value(thread_id="t-A", parent_id="p1"),
            _legacy_value(thread_id="t-B", parent_id="p2"),
            _legacy_value(thread_id="t-A", parent_id="p3"),
        ]

        saver._maybe_migrate_pending_sends(cursor, values)

        # One SELECT per distinct thread_id (2), not one query pinned to the
        # first row's thread.
        assert cursor.execute.call_count == 2

        thread_ids_queried = {call.args[1][0] for call in cursor.execute.call_args_list}
        assert thread_ids_queried == {"t-A", "t-B"}

        # Thread A gets both of its parent ids; thread B gets its one.
        params_by_thread = {
            call.args[1][0]: call.args[1][1:] for call in cursor.execute.call_args_list
        }
        assert set(params_by_thread["t-A"]) == {"p1", "p3"}
        assert set(params_by_thread["t-B"]) == {"p2"}

    def test_no_migration_when_all_modern_checkpoints(self) -> None:
        saver, cursor = _make_saver_with_cursor()
        modern = _legacy_value(thread_id="t-A", parent_id="p1")
        modern["checkpoint"]["v"] = 4  # not legacy

        saver._maybe_migrate_pending_sends(cursor, [modern])

        cursor.execute.assert_not_called()

    def test_no_migration_when_no_parent(self) -> None:
        saver, cursor = _make_saver_with_cursor()
        orphan = _legacy_value(thread_id="t-A", parent_id="")
        orphan["parent_checkpoint_id"] = None

        saver._maybe_migrate_pending_sends(cursor, [orphan])

        cursor.execute.assert_not_called()

    def test_single_thread_still_one_query(self) -> None:
        saver, cursor = _make_saver_with_cursor()
        cursor.fetchall.return_value = []

        values = [
            _legacy_value(thread_id="t-solo", parent_id="p1"),
            _legacy_value(thread_id="t-solo", parent_id="p2"),
        ]

        saver._maybe_migrate_pending_sends(cursor, values)

        assert cursor.execute.call_count == 1
        args = cursor.execute.call_args.args[1]
        assert args[0] == "t-solo"
        assert set(args[1:]) == {"p1", "p2"}
