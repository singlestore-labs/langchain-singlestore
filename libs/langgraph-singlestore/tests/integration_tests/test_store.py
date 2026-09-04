"""Integration tests for :class:`SingleStoreStore`.

Runs a real SingleStore container (see ``conftest.py``) and exercises the
draft store end-to-end via ``PutOp`` + ``GetOp``.
"""

import json
import time
from typing import Any, cast

import pytest
from singlestoredb.connection import connect

from langgraph.store.base import GetOp, Item, ListNamespacesOp, MatchCondition, PutOp
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
