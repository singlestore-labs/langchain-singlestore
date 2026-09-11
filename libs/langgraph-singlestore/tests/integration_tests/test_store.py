"""Integration tests for :class:`SingleStoreStore`.

Runs a real SingleStore container (see ``conftest.py``) and exercises the
draft store end-to-end via ``PutOp`` + ``GetOp``.
"""

import json
import time
from typing import Any, cast

import pytest
from singlestoredb.connection import connect

from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    SearchItem,
    SearchOp,
    TTLConfig,
)
from langgraph.store.singlestore import SingleStoreStore

from .conftest import ConnectionParameters


def _as_item(result: object) -> Item | None:
    """Narrow a ``batch()`` result to an ``Item`` for typed access."""
    assert result is None or isinstance(result, Item)
    return cast("Item | None", result)


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert a tuple ``store`` row into a labelled dict for readable asserts."""
    r = cast("tuple[Any, ...]", row)
    raw_value = r[2]
    value = json.loads(raw_value) if isinstance(raw_value, (str, bytes)) else raw_value
    return {
        "prefix": r[0],
        "key": r[1],
        "value": value,
        "created_at": r[3],
        "updated_at": r[4],
        "expires_at": r[5],
        "ttl_minutes": r[6],
    }


def _fetch_store_row(
    params: ConnectionParameters, namespace: tuple[str, ...], key: str
) -> dict[str, Any] | None:
    """Fetch a single ``store`` row via a raw SQL connection (bypasses the store)."""
    prefix = "/".join(namespace)
    conn = connect(**params.as_kwargs())
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT prefix, `key`, value, created_at, updated_at, expires_at, "
            "ttl_minutes FROM store WHERE prefix = %s AND `key` = %s",
            (prefix, key),
        )
        row = cur.fetchone()
        cur.close()
        return None if row is None else _row_to_dict(row)
    finally:
        conn.close()


def _fetch_all_store_rows(params: ConnectionParameters) -> list[dict[str, Any]]:
    """Fetch every row in the ``store`` table via a raw SQL connection."""
    conn = connect(**params.as_kwargs())
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT prefix, `key`, value, created_at, updated_at, expires_at, "
            "ttl_minutes FROM store ORDER BY prefix, `key`"
        )
        rows = cur.fetchall()
        cur.close()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def _count_store_rows(params: ConnectionParameters) -> int:
    """Return ``SELECT COUNT(*) FROM store`` via a raw SQL connection."""
    conn = connect(**params.as_kwargs())
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM store")
        row = cast("tuple[Any, ...] | None", cur.fetchone())
        cur.close()
        assert row is not None
        return int(row[0])
    finally:
        conn.close()


class TestSingleStoreStorePutOp:
    def test_put_then_get_returns_stored_value(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            namespace = ("users", "alice")
            key = "prefs"
            value = {"theme": "dark", "lang": "en"}

            put_results = store.batch([PutOp(namespace, key, value)])
            assert put_results == [None]

            get_results = store.batch([GetOp(namespace, key)])
            assert len(get_results) == 1
            item = _as_item(get_results[0])
            assert item is not None
            assert item.namespace == namespace
            assert item.key == key
            assert item.value == value
        finally:
            store.close()

    def test_multi_insert_persists_all_rows(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """A single ``batch()`` with many ``PutOp``s writes every row."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            ops = [
                PutOp(("users", "alice"), "prefs", {"theme": "dark"}),
                PutOp(("users", "alice"), "profile", {"name": "Alice"}),
                PutOp(("users", "bob"), "prefs", {"theme": "light"}),
                PutOp(("agents", "planner"), "state", {"step": 1, "done": False}),
            ]
            results = store.batch(ops)
            assert results == [None, None, None, None]

            assert _count_store_rows(connection_parameters) == len(ops)
            for op in ops:
                row = _fetch_store_row(connection_parameters, op.namespace, op.key)
                assert row is not None, f"missing row for {op.namespace}/{op.key}"
                assert row["prefix"] == "/".join(op.namespace)
                assert row["key"] == op.key
                assert row["value"] == op.value
                # No TTL requested => both TTL columns must be NULL.
                assert row["expires_at"] is None
                assert row["ttl_minutes"] is None
        finally:
            store.close()

    def test_update_overwrites_value_and_bumps_updated_at(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Re-``PutOp`` on same (namespace, key) updates value and ``updated_at``."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            namespace = ("users", "alice")
            key = "prefs"

            store.batch([PutOp(namespace, key, {"theme": "dark", "lang": "en"})])
            initial = _fetch_store_row(connection_parameters, namespace, key)
            assert initial is not None
            assert initial["value"] == {"theme": "dark", "lang": "en"}
            assert initial["created_at"] == initial["updated_at"]

            # ``TIMESTAMP`` has 1-second granularity — sleep so the update is
            # observably later than the insert.
            time.sleep(1.1)

            store.batch([PutOp(namespace, key, {"theme": "light", "lang": "fr"})])
            updated = _fetch_store_row(connection_parameters, namespace, key)
            assert updated is not None
            assert updated["value"] == {"theme": "light", "lang": "fr"}
            assert updated["created_at"] == initial["created_at"]
            assert updated["updated_at"] > initial["updated_at"]
            # Update without TTL clears any prior TTL columns.
            assert updated["expires_at"] is None
            assert updated["ttl_minutes"] is None

            # Only one row exists for this (namespace, key).
            assert _count_store_rows(connection_parameters) == 1
        finally:
            store.close()

    def test_delete_removes_row(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """``PutOp`` with ``value=None`` deletes the row from ``store``."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            namespace = ("users", "alice")
            key = "prefs"

            store.batch([PutOp(namespace, key, {"theme": "dark"})])
            assert _fetch_store_row(connection_parameters, namespace, key) is not None

            store.batch([PutOp(namespace, key, None)])
            assert _fetch_store_row(connection_parameters, namespace, key) is None
            assert _count_store_rows(connection_parameters) == 0

            # ``GetOp`` after delete must yield ``None``.
            get_results = store.batch([GetOp(namespace, key)])
            assert get_results == [None]
        finally:
            store.close()

    def test_batch_mix_insert_update_delete(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """A single batch may mix inserts, updates, and deletes."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            # Seed two rows so the batch can update one and delete the other.
            store.batch(
                [
                    PutOp(("users", "alice"), "prefs", {"theme": "dark"}),
                    PutOp(("users", "bob"), "prefs", {"theme": "light"}),
                ]
            )

            results = store.batch(
                [
                    PutOp(("users", "alice"), "prefs", {"theme": "solarized"}),
                    PutOp(("users", "bob"), "prefs", None),
                    PutOp(("users", "carol"), "prefs", {"theme": "sepia"}),
                ]
            )
            assert results == [None, None, None]

            rows = _fetch_all_store_rows(connection_parameters)
            by_prefix = {(r["prefix"], r["key"]): r for r in rows}
            assert set(by_prefix) == {
                ("users/alice", "prefs"),
                ("users/carol", "prefs"),
            }
            assert by_prefix[("users/alice", "prefs")]["value"] == {
                "theme": "solarized"
            }
            assert by_prefix[("users/carol", "prefs")]["value"] == {"theme": "sepia"}
        finally:
            store.close()

    def test_duplicate_ops_in_batch_last_write_wins(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """When one batch contains multiple ops for the same key, the last one wins."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            namespace = ("users", "alice")
            key = "prefs"

            store.batch(
                [
                    PutOp(namespace, key, {"v": 1}),
                    PutOp(namespace, key, {"v": 2}),
                    PutOp(namespace, key, {"v": 3}),
                ]
            )

            row = _fetch_store_row(connection_parameters, namespace, key)
            assert row is not None
            assert row["value"] == {"v": 3}
            assert _count_store_rows(connection_parameters) == 1
        finally:
            store.close()

    def test_delete_of_missing_key_is_noop(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Deleting a non-existent (namespace, key) is silently a no-op."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            results = store.batch([PutOp(("nope",), "missing", None)])
            assert results == [None]
            assert _count_store_rows(connection_parameters) == 0
        finally:
            store.close()

    @pytest.mark.parametrize("ttl_minutes", [1, 5, 60, 1440])
    def test_ttl_populates_expires_at_and_ttl_minutes(
        self,
        connection_parameters: ConnectionParameters,
        ttl_minutes: int,
    ) -> None:
        """``PutOp(ttl=X)`` sets ``ttl_minutes`` and ``expires_at = created_at + X``."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            namespace = ("users", "alice")
            key = f"prefs-ttl-{ttl_minutes}"

            store.batch(
                [PutOp(namespace, key, {"theme": "dark"}, ttl=float(ttl_minutes))]
            )

            row = _fetch_store_row(connection_parameters, namespace, key)
            assert row is not None
            # ``ttl_minutes`` is stored numerically; compare as float.
            assert float(row["ttl_minutes"]) == pytest.approx(float(ttl_minutes))
            assert row["expires_at"] is not None

            # All three ``NOW()`` calls in the INSERT execute in the same
            # statement, so the delta must be exactly ``ttl_minutes`` minutes.
            delta_seconds = (row["expires_at"] - row["created_at"]).total_seconds()
            assert delta_seconds == pytest.approx(ttl_minutes * 60, abs=1.0)
        finally:
            store.close()

    def test_ttl_update_replaces_previous_ttl(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """A subsequent PutOp changes ``expires_at``/``ttl_minutes`` accordingly."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            namespace = ("agents", "planner")
            key = "state"

            # Insert with 60-minute TTL.
            store.batch([PutOp(namespace, key, {"step": 1}, ttl=60.0)])
            first = _fetch_store_row(connection_parameters, namespace, key)
            assert first is not None
            assert float(first["ttl_minutes"]) == pytest.approx(60.0)
            assert first["expires_at"] is not None

            # Overwrite with 5-minute TTL — expires_at must move earlier.
            store.batch([PutOp(namespace, key, {"step": 2}, ttl=5.0)])
            second = _fetch_store_row(connection_parameters, namespace, key)
            assert second is not None
            assert second["value"] == {"step": 2}
            assert float(second["ttl_minutes"]) == pytest.approx(5.0)
            assert second["expires_at"] is not None
            assert second["expires_at"] < first["expires_at"]

            # Overwrite without TTL — both TTL columns must be reset to NULL.
            store.batch([PutOp(namespace, key, {"step": 3})])
            third = _fetch_store_row(connection_parameters, namespace, key)
            assert third is not None
            assert third["value"] == {"step": 3}
            assert third["expires_at"] is None
            assert third["ttl_minutes"] is None
        finally:
            store.close()


class TestSingleStoreStoreGetOp:
    def test_get_missing_key_returns_none(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            results = store.batch([GetOp(("users", "alice"), "missing")])
            assert results == [None]
        finally:
            store.close()

    def test_get_returns_item_without_ttl(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            namespace = ("users", "alice")
            key = "prefs"
            value = {"theme": "dark"}
            store.batch([PutOp(namespace, key, value)])

            results = store.batch([GetOp(namespace, key, refresh_ttl=False)])
            item = _as_item(results[0])
            assert item is not None
            assert item.namespace == namespace
            assert item.key == key
            assert item.value == value
        finally:
            store.close()

    def test_get_batches_multiple_keys_across_namespaces(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Batch of ``GetOp``s returns each item at the right result index."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            store.batch(
                [
                    PutOp(("users", "alice"), "prefs", {"theme": "dark"}),
                    PutOp(("users", "alice"), "profile", {"name": "Alice"}),
                    PutOp(("users", "bob"), "prefs", {"theme": "light"}),
                ]
            )

            ops = [
                GetOp(("users", "alice"), "prefs", refresh_ttl=False),
                GetOp(("users", "bob"), "prefs", refresh_ttl=False),
                GetOp(("users", "alice"), "profile", refresh_ttl=False),
                GetOp(("users", "carol"), "prefs", refresh_ttl=False),
            ]
            results = store.batch(ops)
            assert len(results) == 4

            item0 = _as_item(results[0])
            item1 = _as_item(results[1])
            item2 = _as_item(results[2])
            assert item0 is not None and item0.value == {"theme": "dark"}
            assert item1 is not None and item1.value == {"theme": "light"}
            assert item2 is not None and item2.value == {"name": "Alice"}
            assert results[3] is None  # missing key stays None
        finally:
            store.close()

    def test_get_with_refresh_ttl_true_bumps_expires_at(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """``GetOp(refresh_ttl=True)`` must push ``expires_at`` forward."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            namespace = ("users", "alice")
            key = "prefs"

            store.batch([PutOp(namespace, key, {"theme": "dark"}, ttl=60.0)])
            before = _fetch_store_row(connection_parameters, namespace, key)
            assert before is not None
            assert before["expires_at"] is not None

            # ``TIMESTAMP`` has 1-second granularity — wait so the refresh
            # produces an observably later ``expires_at``.
            time.sleep(1.1)

            results = store.batch([GetOp(namespace, key, refresh_ttl=True)])
            item = _as_item(results[0])
            assert item is not None
            assert item.value == {"theme": "dark"}

            after = _fetch_store_row(connection_parameters, namespace, key)
            assert after is not None
            assert after["expires_at"] is not None
            assert after["expires_at"] > before["expires_at"]
            # ``ttl_minutes`` is preserved on refresh — only ``expires_at`` moves.
            assert float(after["ttl_minutes"]) == pytest.approx(
                float(before["ttl_minutes"])
            )
            # The refresh SQL bumps ``updated_at`` alongside ``expires_at``.
            assert after["updated_at"] >= before["updated_at"]
            # ``created_at`` is immutable.
            assert after["created_at"] == before["created_at"]
        finally:
            store.close()

    def test_get_with_refresh_ttl_false_leaves_expires_at_unchanged(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """``GetOp(refresh_ttl=False)`` must NOT modify ``expires_at``."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            namespace = ("users", "alice")
            key = "prefs"

            store.batch([PutOp(namespace, key, {"theme": "dark"}, ttl=60.0)])
            before = _fetch_store_row(connection_parameters, namespace, key)
            assert before is not None
            assert before["expires_at"] is not None

            time.sleep(1.1)

            results = store.batch([GetOp(namespace, key, refresh_ttl=False)])
            item = _as_item(results[0])
            assert item is not None
            assert item.value == {"theme": "dark"}

            after = _fetch_store_row(connection_parameters, namespace, key)
            assert after is not None
            assert after["expires_at"] == before["expires_at"]
            assert after["updated_at"] == before["updated_at"]
            assert after["created_at"] == before["created_at"]
            assert float(after["ttl_minutes"]) == pytest.approx(
                float(before["ttl_minutes"])
            )
        finally:
            store.close()

    def test_get_with_refresh_ttl_true_on_row_without_ttl_is_noop(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Refreshing TTL on a row with no TTL must not populate ``expires_at``."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            namespace = ("users", "alice")
            key = "prefs"

            store.batch([PutOp(namespace, key, {"theme": "dark"})])
            before = _fetch_store_row(connection_parameters, namespace, key)
            assert before is not None
            assert before["expires_at"] is None
            assert before["ttl_minutes"] is None

            time.sleep(1.1)

            results = store.batch([GetOp(namespace, key, refresh_ttl=True)])
            item = _as_item(results[0])
            assert item is not None

            after = _fetch_store_row(connection_parameters, namespace, key)
            assert after is not None
            assert after["expires_at"] is None
            assert after["ttl_minutes"] is None
        finally:
            store.close()

    def test_get_mixed_refresh_ttl_in_single_batch(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Only rows fetched with ``refresh_ttl=True`` are refreshed."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            namespace = ("users", "alice")
            key_refresh = "prefs"
            key_no_refresh = "profile"

            store.batch(
                [
                    PutOp(namespace, key_refresh, {"theme": "dark"}, ttl=60.0),
                    PutOp(namespace, key_no_refresh, {"name": "Alice"}, ttl=60.0),
                ]
            )
            before_refresh = _fetch_store_row(
                connection_parameters, namespace, key_refresh
            )
            before_no_refresh = _fetch_store_row(
                connection_parameters, namespace, key_no_refresh
            )
            assert before_refresh is not None
            assert before_refresh["expires_at"] is not None
            assert before_no_refresh is not None
            assert before_no_refresh["expires_at"] is not None

            time.sleep(1.1)

            results = store.batch(
                [
                    GetOp(namespace, key_refresh, refresh_ttl=True),
                    GetOp(namespace, key_no_refresh, refresh_ttl=False),
                ]
            )
            assert _as_item(results[0]) is not None
            assert _as_item(results[1]) is not None

            after_refresh = _fetch_store_row(
                connection_parameters, namespace, key_refresh
            )
            after_no_refresh = _fetch_store_row(
                connection_parameters, namespace, key_no_refresh
            )
            assert after_refresh is not None
            assert after_no_refresh is not None
            # Only the ``refresh_ttl=True`` row had its ``expires_at`` bumped.
            assert after_refresh["expires_at"] > before_refresh["expires_at"]
            assert after_no_refresh["expires_at"] == before_no_refresh["expires_at"]
        finally:
            store.close()

    def test_get_expired_row_returns_none(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Rows past ``expires_at`` must not be returned by ``GetOp``."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            namespace = ("users", "alice")
            key = "prefs"

            # Insert normally, then force the row to be already expired via a
            # direct UPDATE — avoids waiting a full minute for the smallest
            # supported TTL to elapse.
            store.batch([PutOp(namespace, key, {"theme": "dark"}, ttl=1.0)])
            conn = connect(**connection_parameters.as_kwargs())
            try:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE store SET expires_at = DATE_SUB(NOW(), INTERVAL 1 MINUTE) "
                    "WHERE prefix = %s AND `key` = %s",
                    ("/".join(namespace), key),
                )
                cur.close()
            finally:
                conn.close()

            # Both refresh modes must treat an expired row as absent.
            results = store.batch(
                [
                    GetOp(namespace, key, refresh_ttl=False),
                    GetOp(namespace, key, refresh_ttl=True),
                ]
            )
            assert results == [None, None]
        finally:
            store.close()

    def test_get_refresh_ttl_default_is_true(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """``GetOp`` defaults ``refresh_ttl`` to ``True`` — verify it refreshes."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            namespace = ("users", "alice")
            key = "prefs"

            store.batch([PutOp(namespace, key, {"theme": "dark"}, ttl=60.0)])
            before = _fetch_store_row(connection_parameters, namespace, key)
            assert before is not None and before["expires_at"] is not None

            time.sleep(1.1)

            # No explicit refresh_ttl — rely on the default.
            results = store.batch([GetOp(namespace, key)])
            assert _as_item(results[0]) is not None

            after = _fetch_store_row(connection_parameters, namespace, key)
            assert after is not None and after["expires_at"] is not None
            assert after["expires_at"] > before["expires_at"]
        finally:
            store.close()


# ---------------------------------------------------------------------------
# ``ListNamespacesOp``
# ---------------------------------------------------------------------------

# Fixed seed for ``list_namespaces`` tests. Each entry is a namespace tuple
# stored with a single dummy key ``k``. Chosen to exercise:
#   * multiple top-level roots (``users``, ``agents``, ``docs``);
#   * varying depths (2..4);
#   * shared suffixes (``prefs``, ``state``);
#   * shared middle segment (``planner``).
_LIST_NS_SEED: tuple[tuple[str, ...], ...] = (
    ("users", "alice", "prefs"),
    ("users", "alice", "profile"),
    ("users", "bob", "prefs"),
    ("users", "carol", "prefs"),
    ("agents", "planner", "state"),
    ("agents", "planner", "config"),
    ("agents", "researcher", "state"),
    ("docs", "public", "readme"),
    ("docs", "private", "draft"),
    ("docs", "private", "notes", "v1"),
)


def _seed_list_namespaces(store: SingleStoreStore) -> None:
    store.batch([PutOp(ns, "k", {"i": i}) for i, ns in enumerate(_LIST_NS_SEED)])


def _list(store: SingleStoreStore, op: ListNamespacesOp) -> list[tuple[str, ...]]:
    results = store.batch([op])
    assert len(results) == 1
    return cast("list[tuple[str, ...]]", results[0])


class TestSingleStoreStoreListNamespacesOp:
    def test_no_conditions_returns_all_distinct_namespaces_sorted(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_list_namespaces(store)

            got = _list(store, ListNamespacesOp())
            assert got == sorted(_LIST_NS_SEED)
        finally:
            store.close()

    def test_prefix_condition_filters_by_root(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_list_namespaces(store)

            got = _list(
                store,
                ListNamespacesOp(
                    match_conditions=(
                        MatchCondition(match_type="prefix", path=("users",)),
                    )
                ),
            )
            expected = sorted(ns for ns in _LIST_NS_SEED if ns[0] == "users")
            assert got == expected
        finally:
            store.close()

    def test_suffix_condition_filters_by_tail(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_list_namespaces(store)

            got = _list(
                store,
                ListNamespacesOp(
                    match_conditions=(
                        MatchCondition(match_type="suffix", path=("prefs",)),
                    )
                ),
            )
            expected = sorted(ns for ns in _LIST_NS_SEED if ns[-1] == "prefs")
            assert got == expected
        finally:
            store.close()

    def test_prefix_condition_with_wildcard_segment(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """``("users", "*")`` matches any user + at least one child segment."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_list_namespaces(store)

            got = _list(
                store,
                ListNamespacesOp(
                    match_conditions=(
                        MatchCondition(match_type="prefix", path=("users", "*")),
                    )
                ),
            )
            expected = sorted(
                ns for ns in _LIST_NS_SEED if ns[0] == "users" and len(ns) >= 3
            )
            assert got == expected
        finally:
            store.close()

    def test_multiple_match_conditions_are_anded(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Both conditions must hold — prefix ``docs`` AND suffix ``draft``."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_list_namespaces(store)

            got = _list(
                store,
                ListNamespacesOp(
                    match_conditions=(
                        MatchCondition(match_type="prefix", path=("docs",)),
                        MatchCondition(match_type="suffix", path=("draft",)),
                    )
                ),
            )
            assert got == [("docs", "private", "draft")]
        finally:
            store.close()

    def test_multiple_match_conditions_can_yield_empty(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Contradictory prefix + suffix must return an empty list, not error."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_list_namespaces(store)

            got = _list(
                store,
                ListNamespacesOp(
                    match_conditions=(
                        MatchCondition(match_type="prefix", path=("users",)),
                        MatchCondition(match_type="suffix", path=("readme",)),
                    )
                ),
            )
            assert got == []
        finally:
            store.close()

    def test_max_depth_truncates_and_deduplicates(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """``max_depth=2`` collapses ``users/alice/prefs`` -> ``users/alice``."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_list_namespaces(store)

            got = _list(store, ListNamespacesOp(max_depth=2))
            expected = sorted({ns[:2] for ns in _LIST_NS_SEED})
            assert got == expected
        finally:
            store.close()

    def test_max_depth_one_returns_only_roots(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_list_namespaces(store)

            got = _list(store, ListNamespacesOp(max_depth=1))
            assert got == [("agents",), ("docs",), ("users",)]
        finally:
            store.close()

    def test_max_depth_greater_than_actual_returns_full_namespaces(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """A ``max_depth`` past the deepest namespace is a no-op truncation."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_list_namespaces(store)

            deepest = max(len(ns) for ns in _LIST_NS_SEED)
            got = _list(store, ListNamespacesOp(max_depth=deepest + 5))
            assert got == sorted(_LIST_NS_SEED)
        finally:
            store.close()

    def test_max_depth_combines_with_match_conditions(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Filter first (via LIKE), then truncate — verify final tuple set."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_list_namespaces(store)

            got = _list(
                store,
                ListNamespacesOp(
                    match_conditions=(
                        MatchCondition(match_type="prefix", path=("users",)),
                    ),
                    max_depth=2,
                ),
            )
            expected = sorted({ns[:2] for ns in _LIST_NS_SEED if ns[0] == "users"})
            assert got == expected
        finally:
            store.close()

    def test_pagination_limit_returns_first_n_in_sort_order(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_list_namespaces(store)

            all_sorted = sorted(_LIST_NS_SEED)
            got = _list(store, ListNamespacesOp(limit=3))
            assert got == all_sorted[:3]
        finally:
            store.close()

    def test_pagination_offset_skips_leading_rows(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_list_namespaces(store)

            all_sorted = sorted(_LIST_NS_SEED)
            got = _list(store, ListNamespacesOp(offset=3, limit=3))
            assert got == all_sorted[3:6]
        finally:
            store.close()

    def test_pagination_covers_the_full_set_via_pages(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Concatenating consecutive pages reproduces the sorted namespace list."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_list_namespaces(store)

            all_sorted = sorted(_LIST_NS_SEED)
            page_size = 4
            collected: list[tuple[str, ...]] = []
            offset = 0
            while True:
                page = _list(store, ListNamespacesOp(limit=page_size, offset=offset))
                if not page:
                    break
                collected.extend(page)
                offset += page_size
                # Defensive stop — avoids an infinite loop if pagination misbehaves.
                assert offset <= len(all_sorted) + page_size
            assert collected == all_sorted
        finally:
            store.close()

    def test_pagination_offset_past_end_returns_empty(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_list_namespaces(store)

            got = _list(
                store, ListNamespacesOp(offset=len(_LIST_NS_SEED) + 10, limit=5)
            )
            assert got == []
        finally:
            store.close()

    def test_pagination_applies_after_max_depth_dedup(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """``max_depth=1`` gives 3 roots; ``offset=1, limit=1`` picks the middle."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_list_namespaces(store)

            got = _list(store, ListNamespacesOp(max_depth=1, offset=1, limit=1))
            assert got == [("docs",)]
        finally:
            store.close()

    def test_expired_rows_are_excluded(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Rows whose TTL has elapsed must not surface in ``list_namespaces``."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            store.batch(
                [
                    PutOp(("users", "alice"), "k", {"i": 0}),
                    PutOp(("users", "bob"), "k", {"i": 1}, ttl=1.0),
                ]
            )
            # Force the second row past its TTL.
            conn = connect(**connection_parameters.as_kwargs())
            try:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE store SET expires_at = DATE_SUB(NOW(), INTERVAL 1 MINUTE) "
                    "WHERE prefix = %s",
                    ("users/bob",),
                )
                cur.close()
            finally:
                conn.close()

            got = _list(
                store,
                ListNamespacesOp(
                    match_conditions=(
                        MatchCondition(match_type="prefix", path=("users",)),
                    )
                ),
            )
            assert got == [("users", "alice")]
        finally:
            store.close()


# --- SearchOp -----------------------------------------------------------------

# Seed keyed by (namespace, key) so the same namespace can appear multiple
# Seed keyed by (namespace, key) so the same namespace can appear multiple
# times. Namespaces are deliberately at least 3 segments so that
# ``namespace_prefix=("users",)`` (which the store translates to a
# ``LIKE 'users/%'`` pattern) actually matches. Values carry heterogeneous
# fields so every filter operator has both matching and non-matching rows.
_SEARCH_SEED: tuple[tuple[tuple[str, ...], str, dict[str, Any]], ...] = (
    (
        ("docs", "public", "articles"),
        "readme",
        {
            "type": "readme",
            "score": 5,
            "public": True,
            "tags": ["intro", "docs"],
            "seeded": True,
        },
    ),
    (
        ("docs", "public", "articles"),
        "guide",
        {
            "type": "guide",
            "score": 3,
            "public": True,
            "tags": ["intro", "tutorial"],
            "seeded": True,
        },
    ),
    (
        ("docs", "private", "notes"),
        "draft",
        {
            "type": "draft",
            "score": 1,
            "public": False,
            "tags": ["wip"],
            "seeded": True,
        },
    ),
    (
        ("users", "alice", "profile"),
        "main",
        {"role": "admin", "level": 10, "active": True, "seeded": True},
    ),
    (
        ("users", "bob", "profile"),
        "main",
        {"role": "user", "level": 5, "active": True, "seeded": True},
    ),
    (
        ("users", "carol", "profile"),
        "main",
        # Only carol carries the ``vip`` field — used to test ``$exists``.
        {
            "role": "user",
            "level": 2,
            "active": False,
            "vip": "no",
            "seeded": True,
        },
    ),
)


def _seed_search(store: SingleStoreStore) -> None:
    store.batch([PutOp(ns, key, val) for ns, key, val in _SEARCH_SEED])


def _search(store: SingleStoreStore, op: SearchOp) -> list[SearchItem]:
    results = store.batch([op])
    assert len(results) == 1
    return cast("list[SearchItem]", results[0])


def _keys(items: list[SearchItem]) -> set[tuple[tuple[str, ...], str]]:
    """Reduce ``SearchItem``s to the identifying (namespace, key) pairs."""
    return {(it.namespace, it.key) for it in items}


def _expected_keys(
    predicate: "Any",
) -> set[tuple[tuple[str, ...], str]]:
    return {(ns, key) for ns, key, val in _SEARCH_SEED if predicate(ns, key, val)}


class TestSingleStoreStoreSearchOp:
    def test_no_prefix_no_filter_returns_all_rows(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_search(store)

            got = _search(
                store, SearchOp(namespace_prefix=(), limit=100, refresh_ttl=False)
            )
            assert _keys(got) == _expected_keys(lambda ns, k, v: True)
            for item in got:
                assert isinstance(item, SearchItem)
                assert isinstance(item.value, dict)
        finally:
            store.close()

    def test_empty_store_returns_empty_list(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            got = _search(store, SearchOp(namespace_prefix=(), refresh_ttl=False))
            assert got == []
        finally:
            store.close()

    def test_all_seeded_rows_via_broad_filter(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Substitute for ``test_no_prefix_no_filter_returns_all_rows`` — pin a
        broad filter so ``_search_where`` produces a non-empty clause."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_search(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=(),
                    filter={"seeded": True},
                    limit=100,
                    refresh_ttl=False,
                ),
            )
            assert _keys(got) == _expected_keys(lambda ns, k, v: True)
            for item in got:
                assert isinstance(item, SearchItem)
                assert isinstance(item.value, dict)
        finally:
            store.close()

    def test_namespace_prefix_filters_by_root(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_search(store)

            got = _search(
                store,
                SearchOp(namespace_prefix=("users",), limit=100, refresh_ttl=False),
            )
            assert _keys(got) == _expected_keys(lambda ns, k, v: ns[0] == "users")
        finally:
            store.close()

    def test_namespace_prefix_filters_by_nested_path(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_search(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=("docs", "public"),
                    limit=100,
                    refresh_ttl=False,
                ),
            )
            assert _keys(got) == _expected_keys(
                lambda ns, k, v: ns[:2] == ("docs", "public")
            )
        finally:
            store.close()

    def test_namespace_prefix_no_match_returns_empty(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_search(store)

            got = _search(
                store,
                SearchOp(namespace_prefix=("nope",), refresh_ttl=False),
            )
            assert got == []
        finally:
            store.close()

    # ------------------------------------------------------------------ filters

    def test_filter_exact_match_string(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_search(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=(),
                    filter={"role": "admin"},
                    limit=100,
                    refresh_ttl=False,
                ),
            )
            assert _keys(got) == _expected_keys(
                lambda ns, k, v: v.get("role") == "admin"
            )
        finally:
            store.close()

    def test_filter_exact_match_bool(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_search(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=(),
                    filter={"active": True},
                    limit=100,
                    refresh_ttl=False,
                ),
            )
            assert _keys(got) == _expected_keys(
                lambda ns, k, v: v.get("active") is True
            )
        finally:
            store.close()

    def test_filter_exact_match_int(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_search(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=(),
                    filter={"score": 5},
                    limit=100,
                    refresh_ttl=False,
                ),
            )
            assert _keys(got) == _expected_keys(lambda ns, k, v: v.get("score") == 5)
        finally:
            store.close()

    def test_filter_eq_operator(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_search(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=(),
                    filter={"role": {"$eq": "user"}},
                    limit=100,
                    refresh_ttl=False,
                ),
            )
            assert _keys(got) == _expected_keys(
                lambda ns, k, v: v.get("role") == "user"
            )
        finally:
            store.close()

    def test_filter_ne_operator(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """``$ne`` requires the field to exist AND be different — rows without
        the field are excluded."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_search(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=(),
                    filter={"role": {"$ne": "admin"}},
                    limit=100,
                    refresh_ttl=False,
                ),
            )
            assert _keys(got) == _expected_keys(
                lambda ns, k, v: "role" in v and v["role"] != "admin"
            )
        finally:
            store.close()

    @pytest.mark.parametrize(
        ("operator", "value", "predicate"),
        [
            ("$gt", 3, lambda x: x > 3),
            ("$gte", 3, lambda x: x >= 3),
            ("$lt", 5, lambda x: x < 5),
            ("$lte", 5, lambda x: x <= 5),
        ],
    )
    def test_filter_numeric_comparisons(
        self,
        connection_parameters: ConnectionParameters,
        operator: str,
        value: int,
        predicate: "Any",
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_search(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=(),
                    filter={"score": {operator: value}},
                    limit=100,
                    refresh_ttl=False,
                ),
            )
            assert _keys(got) == _expected_keys(
                lambda ns, k, v: "score" in v and predicate(v["score"])
            )
        finally:
            store.close()

    def test_filter_in_operator(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_search(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=(),
                    filter={"role": {"$in": ["admin", "user"]}},
                    limit=100,
                    refresh_ttl=False,
                ),
            )
            assert _keys(got) == _expected_keys(
                lambda ns, k, v: v.get("role") in ("admin", "user")
            )
        finally:
            store.close()

    def test_filter_nin_operator(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """``$nin`` requires the field to exist and NOT be in the list."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_search(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=(),
                    filter={"role": {"$nin": ["admin"]}},
                    limit=100,
                    refresh_ttl=False,
                ),
            )
            assert _keys(got) == _expected_keys(
                lambda ns, k, v: "role" in v and v["role"] not in ("admin",)
            )
        finally:
            store.close()

    def test_filter_exists_true(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_search(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=(),
                    filter={"vip": {"$exists": True}},
                    limit=100,
                    refresh_ttl=False,
                ),
            )
            assert _keys(got) == _expected_keys(lambda ns, k, v: "vip" in v)
        finally:
            store.close()

    def test_filter_exists_false(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_search(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=(),
                    filter={"vip": {"$exists": False}},
                    limit=100,
                    refresh_ttl=False,
                ),
            )
            assert _keys(got) == _expected_keys(lambda ns, k, v: "vip" not in v)
        finally:
            store.close()

    def test_filter_multiple_keys_are_anded(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_search(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=(),
                    filter={"role": "user", "active": True},
                    limit=100,
                    refresh_ttl=False,
                ),
            )
            assert _keys(got) == _expected_keys(
                lambda ns, k, v: v.get("role") == "user" and v.get("active") is True
            )
        finally:
            store.close()

    def test_filter_mixes_exact_and_operator(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_search(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=(),
                    filter={"role": "user", "level": {"$gte": 5}},
                    limit=100,
                    refresh_ttl=False,
                ),
            )
            assert _keys(got) == _expected_keys(
                lambda ns, k, v: v.get("role") == "user" and v.get("level", 0) >= 5
            )
        finally:
            store.close()

    def test_prefix_and_filter_combined(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_search(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=("docs",),
                    filter={"public": True},
                    limit=100,
                    refresh_ttl=False,
                ),
            )
            assert _keys(got) == _expected_keys(
                lambda ns, k, v: ns[0] == "docs" and v.get("public") is True
            )
        finally:
            store.close()

    def test_filter_with_no_matches_returns_empty(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_search(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=(),
                    filter={"role": "ghost"},
                    refresh_ttl=False,
                ),
            )
            assert got == []
        finally:
            store.close()

    # -------------------------------------------------------------- pagination

    def test_pagination_limit_caps_result_size(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_search(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=(),
                    filter={"seeded": True},
                    limit=2,
                    refresh_ttl=False,
                ),
            )
            assert len(got) == 2
            expected_all = _expected_keys(lambda ns, k, v: True)
            assert _keys(got).issubset(expected_all)
        finally:
            store.close()

    def test_pagination_limit_plus_offset_covers_full_set(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Paging through the whole store returns every seeded row exactly once."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_search(store)

            expected_all = _expected_keys(lambda ns, k, v: True)
            page_size = 2
            collected: set[tuple[tuple[str, ...], str]] = set()
            offset = 0
            while True:
                page = _search(
                    store,
                    SearchOp(
                        namespace_prefix=(),
                        filter={"seeded": True},
                        limit=page_size,
                        offset=offset,
                        refresh_ttl=False,
                    ),
                )
                if not page:
                    break
                page_keys = _keys(page)
                assert page_keys.isdisjoint(collected)
                collected |= page_keys
                offset += page_size
                assert offset <= len(expected_all) + page_size
            assert collected == expected_all
        finally:
            store.close()

    def test_pagination_offset_past_end_returns_empty(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_search(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=(),
                    filter={"seeded": True},
                    offset=len(_SEARCH_SEED) + 10,
                    limit=5,
                    refresh_ttl=False,
                ),
            )
            assert got == []
        finally:
            store.close()

    # -------------------------------------------------------------- ordering

    def test_results_ordered_by_updated_at_desc(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Newer writes come first; ordering is stable across the returned page."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()

            # Space inserts by >1s so ``updated_at`` (TIMESTAMP, 1s granularity)
            # is strictly different between rows. Uses a 2-segment namespace
            # so ``namespace_prefix=("t",)`` (LIKE ``t/%``) actually matches.
            store.batch([PutOp(("t", "sub"), "a", {"i": 0})])
            time.sleep(1.1)
            store.batch([PutOp(("t", "sub"), "b", {"i": 1})])
            time.sleep(1.1)
            store.batch([PutOp(("t", "sub"), "c", {"i": 2})])

            got = _search(
                store,
                SearchOp(namespace_prefix=("t",), limit=10, refresh_ttl=False),
            )
            assert [it.key for it in got] == ["c", "b", "a"]
            for earlier, later in zip(got, got[1:]):
                assert earlier.updated_at >= later.updated_at
        finally:
            store.close()

    def test_update_moves_row_to_front(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Re-writing an existing row bumps ``updated_at`` and reshuffles order."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()

            store.batch([PutOp(("t", "sub"), "a", {"i": 0})])
            time.sleep(1.1)
            store.batch([PutOp(("t", "sub"), "b", {"i": 1})])
            time.sleep(1.1)
            # Overwrite ``a`` — it should now be newest.
            store.batch([PutOp(("t", "sub"), "a", {"i": 99})])

            got = _search(
                store,
                SearchOp(namespace_prefix=("t",), limit=10, refresh_ttl=False),
            )
            assert [it.key for it in got] == ["a", "b"]
            assert got[0].value == {"i": 99}
        finally:
            store.close()

    # -------------------------------------------------------------- TTL / expiry

    def test_expired_rows_are_excluded(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Rows whose TTL has elapsed must not surface in ``search``."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            store.batch(
                [
                    PutOp(("t", "sub"), "fresh", {"i": 0}),
                    PutOp(("t", "sub"), "stale", {"i": 1}, ttl=1.0),
                ]
            )
            conn = connect(**connection_parameters.as_kwargs())
            try:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE store SET expires_at = DATE_SUB(NOW(), INTERVAL 1 MINUTE) "
                    "WHERE `key` = %s",
                    ("stale",),
                )
                cur.close()
            finally:
                conn.close()

            got = _search(
                store,
                SearchOp(namespace_prefix=("t",), limit=10, refresh_ttl=False),
            )
            assert [it.key for it in got] == ["fresh"]
        finally:
            store.close()

    def test_refresh_ttl_true_bumps_expires_at_on_matching_rows(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """``SearchOp(refresh_ttl=True)`` extends ``expires_at`` on rows that
        match the search — and only those rows."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            store.batch(
                [
                    PutOp(("t", "sub"), "match", {"kind": "hit"}, ttl=1.0),
                    PutOp(("t", "sub"), "skip", {"kind": "miss"}, ttl=1.0),
                ]
            )
            before = {
                r["key"]: r["expires_at"]
                for r in _fetch_all_store_rows(connection_parameters)
            }
            assert before["match"] is not None and before["skip"] is not None

            # TIMESTAMP granularity is 1s — sleep so the refresh is observable.
            time.sleep(1.1)

            # Filter matches only ``match``; ``refresh_ttl=True`` (default).
            got = _search(
                store,
                SearchOp(
                    namespace_prefix=("t",),
                    filter={"kind": "hit"},
                    limit=10,
                ),
            )
            assert [it.key for it in got] == ["match"]

            after = {
                r["key"]: r["expires_at"]
                for r in _fetch_all_store_rows(connection_parameters)
            }
            assert after["match"] > before["match"]
            assert after["skip"] == before["skip"]
        finally:
            store.close()

    def test_refresh_ttl_true_with_no_prefix_no_filter(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """The empty-clause path must produce valid SQL for both the SELECT
        and the TTL-refresh UPDATE."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_search(store)

            got = _search(store, SearchOp(namespace_prefix=(), limit=100))
            assert _keys(got) == _expected_keys(lambda ns, k, v: True)
        finally:
            store.close()

    # -------------------------------------------------------------- vector query

    def test_query_raises_not_configured(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Natural-language search is out of scope for this draft."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            _seed_search(store)

            with pytest.raises(
                ValueError, match="Please provide an index configuration"
            ):
                store.batch(
                    [
                        SearchOp(
                            namespace_prefix=(),
                            query="find me something",
                            refresh_ttl=False,
                        )
                    ]
                )
        finally:
            store.close()


# --- TTL sweeping -------------------------------------------------------------


def _expire_rows_by_key(params: ConnectionParameters, keys: list[str]) -> None:
    """Force the given rows past their ``expires_at`` via raw SQL."""
    conn = connect(**params.as_kwargs())
    try:
        cur = conn.cursor()
        placeholders = ",".join(["%s"] * len(keys))
        cur.execute(
            f"UPDATE store SET expires_at = DATE_SUB(NOW(), INTERVAL 1 MINUTE) "
            f"WHERE `key` IN ({placeholders})",
            tuple(keys),
        )
        cur.close()
    finally:
        conn.close()


def _wait_until(predicate: Any, timeout: float = 5.0, interval: float = 0.05) -> bool:
    """Poll ``predicate`` up to ``timeout`` seconds; return the last value."""
    deadline = time.time() + timeout
    result = predicate()
    while not result and time.time() < deadline:
        time.sleep(interval)
        result = predicate()
    return bool(result)


class TestSingleStoreStoreSweepTTL:
    def test_sweep_ttl_deletes_only_expired_rows(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Only rows whose ``expires_at`` has passed are removed; rows with a
        future ``expires_at`` and rows with ``ttl_minutes IS NULL`` survive."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            store.batch(
                [
                    PutOp(("t",), "no_ttl", {"i": 0}),
                    PutOp(("t",), "fresh", {"i": 1}, ttl=60.0),
                    PutOp(("t",), "stale1", {"i": 2}, ttl=1.0),
                    PutOp(("t",), "stale2", {"i": 3}, ttl=1.0),
                ]
            )
            _expire_rows_by_key(connection_parameters, ["stale1", "stale2"])

            deleted = store.sweep_ttl()
            assert deleted == 2

            remaining = {r["key"] for r in _fetch_all_store_rows(connection_parameters)}
            assert remaining == {"no_ttl", "fresh"}
        finally:
            store.close()

    def test_sweep_ttl_returns_zero_when_nothing_expired(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            store.batch(
                [
                    PutOp(("t",), "no_ttl", {"i": 0}),
                    PutOp(("t",), "fresh", {"i": 1}, ttl=60.0),
                ]
            )
            assert store.sweep_ttl() == 0
            assert _count_store_rows(connection_parameters) == 2
        finally:
            store.close()

    def test_sweep_ttl_on_empty_store_returns_zero(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            assert store.sweep_ttl() == 0
        finally:
            store.close()

    def test_sweep_ttl_boundary_expires_at_equal_now_is_deleted(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """A row with ``expires_at == NOW()`` is invisible to reads
        (``_SELECT_BASE`` uses ``expires_at > NOW()``), so the sweeper must
        delete it — otherwise it would leak."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            store.batch([PutOp(("t",), "edge", {"i": 0}, ttl=60.0)])
            conn = connect(**connection_parameters.as_kwargs())
            try:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE store SET expires_at = NOW() WHERE `key` = %s",
                    ("edge",),
                )
                cur.close()
            finally:
                conn.close()

            deleted = store.sweep_ttl()
            assert deleted == 1
            assert _count_store_rows(connection_parameters) == 0
        finally:
            store.close()


class TestSingleStoreStoreTTLSweeperThread:
    def test_start_ttl_sweeper_without_config_returns_resolved_future(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """No ``ttl_config`` → the sweeper is a no-op and the returned future
        is already resolved."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            future = store.start_ttl_sweeper()
            assert future.done()
            assert future.result() is None
            assert store._ttl_sweeper_thread is None
        finally:
            store.close()

    def test_start_ttl_sweeper_runs_initial_sweep(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """The first sweep happens as soon as the background thread starts —
        before the first ``interval`` wait — so callers don't need to wait
        one full interval to see effects."""
        store = SingleStoreStore(
            ttl_config=TTLConfig(sweep_interval_minutes=60),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            store.batch(
                [
                    PutOp(("t",), "fresh", {"i": 0}),
                    PutOp(("t",), "stale", {"i": 1}, ttl=1.0),
                ]
            )
            _expire_rows_by_key(connection_parameters, ["stale"])
            assert _count_store_rows(connection_parameters) == 2

            future = store.start_ttl_sweeper()
            assert not future.done()

            assert _wait_until(lambda: _count_store_rows(connection_parameters) == 1)
            remaining = {r["key"] for r in _fetch_all_store_rows(connection_parameters)}
            assert remaining == {"fresh"}

            assert store.stop_ttl_sweeper(timeout=5.0) is True
            assert future.result(timeout=5.0) is None
        finally:
            store.close()

    def test_stop_ttl_sweeper_when_not_running_returns_true(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            assert store.stop_ttl_sweeper() is True
        finally:
            store.close()

    def test_start_ttl_sweeper_is_idempotent(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """A second ``start_ttl_sweeper`` on an already-running sweeper
        returns the same tracked future without spawning a second thread."""
        store = SingleStoreStore(
            ttl_config=TTLConfig(sweep_interval_minutes=60),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            first = store.start_ttl_sweeper()
            thread = store._ttl_sweeper_thread
            assert thread is not None and thread.is_alive()

            second = store.start_ttl_sweeper()
            assert second is first
            assert store._ttl_sweeper_thread is thread

            assert store.stop_ttl_sweeper(timeout=5.0) is True
            assert first.result(timeout=5.0) is None
        finally:
            store.close()

    def test_start_after_stop_starts_a_new_thread(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """After ``stop_ttl_sweeper``, a subsequent ``start_ttl_sweeper`` must
        launch a fresh thread and return a fresh future."""
        store = SingleStoreStore(
            ttl_config=TTLConfig(sweep_interval_minutes=60),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            first_future = store.start_ttl_sweeper()
            first_thread = store._ttl_sweeper_thread
            assert first_thread is not None

            assert store.stop_ttl_sweeper(timeout=5.0) is True
            assert first_future.result(timeout=5.0) is None
            assert store._ttl_sweeper_thread is None

            second_future = store.start_ttl_sweeper()
            second_thread = store._ttl_sweeper_thread
            assert second_thread is not None
            assert second_thread is not first_thread
            assert second_future is not first_future

            assert store.stop_ttl_sweeper(timeout=5.0) is True
            assert second_future.result(timeout=5.0) is None
        finally:
            store.close()

    def test_sweep_interval_minutes_kwarg_overrides_config(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """The ``sweep_interval_minutes`` argument to ``start_ttl_sweeper``
        wins over the value in ``ttl_config``. We can't measure interval
        directly without waiting, so this test just verifies that the call
        accepts a fractional value and the sweeper starts + stops cleanly."""
        store = SingleStoreStore(
            ttl_config=TTLConfig(sweep_interval_minutes=60),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            future = store.start_ttl_sweeper(sweep_interval_minutes=0.01)
            assert store._ttl_sweeper_thread is not None
            assert store.stop_ttl_sweeper(timeout=5.0) is True
            assert future.result(timeout=5.0) is None
        finally:
            store.close()


# --- Mixed batch --------------------------------------------------------------
# ``SingleStoreStore.batch`` executes ops in a fixed order regardless of the
# caller's ordering: GetOp → SearchOp → ListNamespacesOp → PutOp. Reads
# therefore see the store state that existed *before* the batch, and writes
# take effect only after every read in the batch has run. These tests pin
# that contract and verify that results are still scattered back into the
# caller's original op positions.


class TestSingleStoreStoreMixedBatch:
    def test_empty_batch_returns_empty_list(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            assert store.batch([]) == []
        finally:
            store.close()

    def test_result_shape_matches_caller_op_order(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Each result slot corresponds to the caller's op at the same index,
        regardless of the internal execution order (Get → Search → List → Put)."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            store.batch([PutOp(("users", "alice"), "prefs", {"theme": "dark"})])

            ops: list[Op] = [
                GetOp(("users", "alice"), "prefs"),
                PutOp(("users", "bob"), "prefs", {"theme": "light"}),
                SearchOp(namespace_prefix=("users",), refresh_ttl=False),
                ListNamespacesOp(),
                GetOp(("users", "carol"), "prefs"),
            ]
            results = store.batch(ops)

            assert len(results) == len(ops)
            # 0: existing key -> Item
            got = _as_item(results[0])
            assert got is not None
            assert got.value == {"theme": "dark"}
            # 1: put -> None sentinel
            assert results[1] is None
            # 2: search -> list[SearchItem]
            search_hits = cast("list[SearchItem]", results[2])
            assert isinstance(search_hits, list)
            assert all(isinstance(it, SearchItem) for it in search_hits)
            # 3: list namespaces -> list[tuple[str, ...]]
            listed = cast("list[tuple[str, ...]]", results[3])
            assert isinstance(listed, list)
            assert all(isinstance(ns, tuple) for ns in listed)
            # 4: missing key -> None
            assert results[4] is None
        finally:
            store.close()

    def test_reads_see_pre_batch_state_not_puts_in_same_batch(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """A ``GetOp`` batched with a ``PutOp`` for the same key returns
        ``None`` (or the pre-batch value) — writes run last."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            namespace = ("users", "alice")

            results = store.batch(
                [
                    PutOp(namespace, "prefs", {"theme": "dark"}),
                    GetOp(namespace, "prefs"),
                ]
            )
            assert results[0] is None
            # Get ran BEFORE Put, so it must not observe the pending write.
            assert _as_item(results[1]) is None

            # Follow-up read (new batch) does see the write.
            observed = _as_item(store.batch([GetOp(namespace, "prefs")])[0])
            assert observed is not None
            assert observed.value == {"theme": "dark"}
        finally:
            store.close()

    def test_get_sees_pre_batch_value_when_batched_with_overwrite(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """If a key already exists and the batch both reads and overwrites it,
        the read returns the OLD value; a later batch observes the new one."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            namespace = ("users", "alice")

            store.batch([PutOp(namespace, "prefs", {"theme": "dark"})])

            results = store.batch(
                [
                    GetOp(namespace, "prefs"),
                    PutOp(namespace, "prefs", {"theme": "light"}),
                ]
            )
            got = _as_item(results[0])
            assert got is not None
            assert got.value == {"theme": "dark"}
            assert results[1] is None

            after = _as_item(store.batch([GetOp(namespace, "prefs")])[0])
            assert after is not None
            assert after.value == {"theme": "light"}
        finally:
            store.close()

    def test_get_sees_pre_batch_value_when_batched_with_delete(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """``PutOp(value=None)`` is a delete; a GetOp for the same key in the
        same batch still returns the pre-batch row."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            namespace = ("users", "alice")

            store.batch([PutOp(namespace, "prefs", {"theme": "dark"})])

            results = store.batch(
                [
                    GetOp(namespace, "prefs"),
                    PutOp(namespace, "prefs", None),
                ]
            )
            got = _as_item(results[0])
            assert got is not None
            assert got.value == {"theme": "dark"}
            assert results[1] is None

            # After the batch, the row is gone.
            assert _fetch_store_row(connection_parameters, namespace, "prefs") is None
        finally:
            store.close()

    def test_search_and_list_do_not_see_puts_from_same_batch(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Both ``SearchOp`` and ``ListNamespacesOp`` run before ``PutOp``,
        so a new namespace introduced in the same batch is not visible to
        either read op."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            store.batch([PutOp(("users", "alice"), "prefs", {"theme": "dark"})])

            results = store.batch(
                [
                    PutOp(("agents", "planner"), "state", {"step": 1}),
                    SearchOp(
                        namespace_prefix=(),
                        filter={"theme": "dark"},
                        refresh_ttl=False,
                    ),
                    ListNamespacesOp(),
                ]
            )
            search_hits = cast("list[SearchItem]", results[1])
            listed = cast("list[tuple[str, ...]]", results[2])

            # Search filter matches alice's row; agents row is not yet visible.
            assert {(it.namespace, it.key) for it in search_hits} == {
                (("users", "alice"), "prefs")
            }
            # ListNamespaces sees only pre-batch namespaces.
            assert ("agents", "planner") not in listed
            assert ("users", "alice") in listed

            # After the batch commits, both namespaces are visible.
            after_list = _list(store, ListNamespacesOp())
            assert ("agents", "planner") in after_list
            assert ("users", "alice") in after_list
        finally:
            store.close()

    def test_within_group_order_is_preserved_across_result_slots(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Interleaving many ops of the same type across the caller list —
        each GetOp result must land back at its original slot."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            store.batch(
                [
                    PutOp(("users", "alice"), "prefs", {"i": 1}),
                    PutOp(("users", "bob"), "prefs", {"i": 2}),
                    PutOp(("users", "carol"), "prefs", {"i": 3}),
                ]
            )

            ops: list[Op] = [
                GetOp(("users", "alice"), "prefs"),
                PutOp(("users", "dave"), "prefs", {"i": 4}),
                GetOp(("users", "bob"), "prefs"),
                PutOp(("users", "erin"), "prefs", {"i": 5}),
                GetOp(("users", "carol"), "prefs"),
                GetOp(("users", "ghost"), "prefs"),
            ]
            results = store.batch(ops)

            assert results[1] is None
            assert results[3] is None
            for idx, expected_i in [(0, 1), (2, 2), (4, 3)]:
                item = _as_item(results[idx])
                assert item is not None
                assert item.value == {"i": expected_i}
            assert results[5] is None
        finally:
            store.close()

    def test_deduplicated_puts_in_batch_last_write_wins(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Multiple ``PutOp``s for the same (namespace, key) collapse to the
        last one; a batched ``GetOp`` still sees the pre-batch state (or
        ``None`` for a brand-new key)."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            namespace = ("t",)

            results = store.batch(
                [
                    PutOp(namespace, "k", {"v": 1}),
                    GetOp(namespace, "k"),
                    PutOp(namespace, "k", None),  # delete
                    PutOp(namespace, "k", {"v": 2}),  # final winner
                ]
            )
            # The get slot ran against pre-batch state: key did not exist.
            assert results[0] is None
            assert _as_item(results[1]) is None
            assert results[2] is None
            assert results[3] is None

            # Post-batch: the final PutOp wins.
            final = _as_item(store.batch([GetOp(namespace, "k")])[0])
            assert final is not None
            assert final.value == {"v": 2}
            assert _count_store_rows(connection_parameters) == 1
        finally:
            store.close()

    def test_get_refresh_ttl_and_search_refresh_ttl_in_same_batch(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """A single batch may mix a GetOp with ``refresh_ttl=True`` and a
        SearchOp with ``refresh_ttl=True`` — both must independently bump
        ``expires_at`` on the rows they touched."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            store.batch(
                [
                    PutOp(("users", "alice"), "prefs", {"role": "admin"}, ttl=1.0),
                    PutOp(("users", "bob"), "prefs", {"role": "user"}, ttl=1.0),
                    PutOp(("users", "carol"), "prefs", {"role": "user"}, ttl=1.0),
                ]
            )
            before = {
                r["key"]: r["expires_at"]
                for r in _fetch_all_store_rows(connection_parameters)
            }
            # 1s TIMESTAMP granularity — sleep so the refresh is observable.
            time.sleep(1.1)

            results = store.batch(
                [
                    GetOp(("users", "alice"), "prefs", refresh_ttl=True),
                    SearchOp(
                        namespace_prefix=("users",),
                        filter={"role": "user"},
                        refresh_ttl=True,
                    ),
                ]
            )
            assert _as_item(results[0]) is not None
            hits = cast("list[SearchItem]", results[1])
            assert {it.key for it in hits} == {"prefs"}
            assert {it.namespace for it in hits} == {
                ("users", "bob"),
                ("users", "carol"),
            }

            after = {
                r["key"]: r["expires_at"]
                for r in _fetch_all_store_rows(connection_parameters)
            }
            # All three rows were touched by one of the two reads.
            for key in ("prefs",):
                assert after[key] > before[key]
        finally:
            store.close()

    def test_mixed_batch_all_op_types_end_to_end(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """One batch containing every op type; asserts each result slot and
        the final DB state after the batch commits."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            store.batch(
                [
                    PutOp(("users", "alice"), "prefs", {"theme": "dark"}),
                    PutOp(("docs", "public"), "readme", {"seeded": True}),
                ]
            )

            ops: list[Op] = [
                GetOp(("users", "alice"), "prefs"),  # 0 - hit
                SearchOp(
                    namespace_prefix=("docs",),
                    filter={"seeded": True},
                    refresh_ttl=False,
                ),  # 1 - one hit
                PutOp(("users", "bob"), "prefs", {"theme": "light"}),  # 2
                ListNamespacesOp(),  # 3
                GetOp(("users", "bob"), "prefs"),  # 4 - miss (pre-batch)
                PutOp(("users", "alice"), "prefs", None),  # 5 - delete
            ]
            results = store.batch(ops)

            item0 = _as_item(results[0])
            assert item0 is not None and item0.value == {"theme": "dark"}

            hits = cast("list[SearchItem]", results[1])
            assert {(h.namespace, h.key) for h in hits} == {
                (("docs", "public"), "readme")
            }

            assert results[2] is None

            listed = cast("list[tuple[str, ...]]", results[3])
            assert ("users", "alice") in listed
            assert ("docs", "public") in listed
            assert ("users", "bob") not in listed  # not yet visible

            assert _as_item(results[4]) is None
            assert results[5] is None

            # Post-batch final state: bob written, alice deleted, docs untouched.
            all_rows = _fetch_all_store_rows(connection_parameters)
            keys_by_prefix = {(r["prefix"], r["key"]) for r in all_rows}
            assert keys_by_prefix == {
                ("users/bob", "prefs"),
                ("docs/public", "readme"),
            }
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_abatch_mixed_ops_returns_same_shape_as_batch(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """``abatch`` delegates to ``batch`` via the default executor — the
        result shape and ordering must be identical to the sync path."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            store.batch([PutOp(("users", "alice"), "prefs", {"theme": "dark"})])

            ops: list[Op] = [
                GetOp(("users", "alice"), "prefs"),
                PutOp(("users", "bob"), "prefs", {"theme": "light"}),
                SearchOp(namespace_prefix=("users",), refresh_ttl=False),
                ListNamespacesOp(),
            ]
            results = await store.abatch(ops)

            assert len(results) == len(ops)
            item = _as_item(results[0])
            assert item is not None
            assert item.value == {"theme": "dark"}
            assert results[1] is None
            assert isinstance(results[2], list)
            assert isinstance(results[3], list)
        finally:
            store.close()
