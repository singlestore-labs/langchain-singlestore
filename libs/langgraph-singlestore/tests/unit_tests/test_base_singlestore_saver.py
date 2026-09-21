"""Unit tests for :class:`BaseSingleStoreSaver`.

The base class is dialect-agnostic: it holds the SQL constants, the serde
helpers (``_dump_blobs``/``_load_blobs``/``_dump_writes``/``_load_writes``),
``_migrate_pending_sends``, ``get_next_version``, and ``_search_where``.

These tests exercise each of those pieces directly on the base class — no
database connection is required.
"""

from __future__ import annotations

import re
from typing import Any, cast

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import WRITES_IDX_MAP

from langgraph_singlestore.checkpoint._base import (
    INSERT_CHECKPOINT_WRITES_SQL,
    MIGRATIONS,
    SELECT_PENDING_SENDS_SQL,
    SELECT_SQL,
    TASKS,
    UPSERT_CHECKPOINT_BLOBS_SQL,
    UPSERT_CHECKPOINT_WRITES_SQL,
    UPSERT_CHECKPOINTS_SQL,
    BaseSingleStoreSaver,
)

# ---------------------------------------------------------------- fixtures


@pytest.fixture
def saver() -> BaseSingleStoreSaver:
    """A fresh base saver with the default serializer."""
    return BaseSingleStoreSaver()


# ---------------------------------------------------------------- class wiring


class TestClassAttributes:
    """The class-level SQL constants must be wired to the module constants."""

    def test_select_sql_is_module_constant(self) -> None:
        assert BaseSingleStoreSaver.SELECT_SQL is SELECT_SQL

    def test_select_pending_sends_sql_is_module_constant(self) -> None:
        assert BaseSingleStoreSaver.SELECT_PENDING_SENDS_SQL is SELECT_PENDING_SENDS_SQL

    def test_migrations_are_module_constant(self) -> None:
        assert BaseSingleStoreSaver.MIGRATIONS is MIGRATIONS

    def test_upsert_sql_constants_match_module(self) -> None:
        assert (
            BaseSingleStoreSaver.UPSERT_CHECKPOINT_BLOBS_SQL
            is UPSERT_CHECKPOINT_BLOBS_SQL
        )
        assert BaseSingleStoreSaver.UPSERT_CHECKPOINTS_SQL is UPSERT_CHECKPOINTS_SQL
        assert (
            BaseSingleStoreSaver.UPSERT_CHECKPOINT_WRITES_SQL
            is UPSERT_CHECKPOINT_WRITES_SQL
        )
        assert (
            BaseSingleStoreSaver.INSERT_CHECKPOINT_WRITES_SQL
            is INSERT_CHECKPOINT_WRITES_SQL
        )

    def test_select_sql_contains_where_placeholder(self) -> None:
        """``_search_where`` is spliced into ``{{where}}`` by callers."""
        assert "{{where}}" in SELECT_SQL

    def test_migrations_first_entry_creates_migrations_table(self) -> None:
        """Position 0 is the sentinel table used to track schema version."""
        assert "checkpoint_migrations" in MIGRATIONS[0]

    def test_pending_sends_sql_references_tasks_channel(self) -> None:
        """``TASKS`` must be interpolated into the pending-sends query."""
        assert f"channel = '{TASKS}'" in SELECT_PENDING_SENDS_SQL


# ---------------------------------------------------------------- get_next_version


class TestGetNextVersion:
    _VERSION_RE = re.compile(r"^\d{32}\.\d+\.\d+$|^\d{32}\.\d+(?:e[-+]?\d+)?$")

    def test_none_returns_v1(self, saver: BaseSingleStoreSaver) -> None:
        v = saver.get_next_version(None, None)
        assert v.split(".")[0] == "0" * 31 + "1"

    def test_int_current_is_incremented(self, saver: BaseSingleStoreSaver) -> None:
        v = saver.get_next_version(3, None)  # type: ignore[arg-type]
        assert v.split(".")[0] == "0" * 31 + "4"

    def test_string_current_is_incremented(self, saver: BaseSingleStoreSaver) -> None:
        current = "0" * 31 + "5" + ".0000000000000001"
        v = saver.get_next_version(current, None)
        assert v.split(".")[0] == "0" * 31 + "6"

    def test_version_prefix_is_32_zero_padded_digits(
        self, saver: BaseSingleStoreSaver
    ) -> None:
        v = saver.get_next_version(None, None)
        prefix, _, _ = v.partition(".")
        assert len(prefix) == 32
        assert prefix.isdigit()

    def test_versions_are_strictly_increasing(
        self, saver: BaseSingleStoreSaver
    ) -> None:
        v1 = saver.get_next_version(None, None)
        v2 = saver.get_next_version(v1, None)
        v3 = saver.get_next_version(v2, None)
        assert v1 < v2 < v3


# ---------------------------------------------------------------- _dump_blobs


class TestDumpBlobs:
    def test_empty_versions_returns_empty_list(
        self, saver: BaseSingleStoreSaver
    ) -> None:
        assert saver._dump_blobs("t", "ns", {"a": 1}, {}) == []

    def test_present_channel_is_serialized(self, saver: BaseSingleStoreSaver) -> None:
        rows = saver._dump_blobs("t1", "ns1", {"messages": "hi"}, {"messages": "v1"})
        assert len(rows) == 1
        thread_id, ns, channel, version, type_tag, blob = rows[0]
        assert (thread_id, ns, channel, version) == ("t1", "ns1", "messages", "v1")
        # JsonPlusSerializer emits ("msgpack", <bytes>) for simple values.
        assert type_tag == "msgpack"
        assert isinstance(blob, (bytes, bytearray))

    def test_missing_channel_is_marked_empty_with_null_blob(
        self, saver: BaseSingleStoreSaver
    ) -> None:
        """``versions`` may include channels not present in ``values``; the
        base class emits a sentinel ``("empty", None)`` row for those."""
        rows = saver._dump_blobs("t", "ns", {}, {"gone": "v1"})
        assert rows == [("t", "ns", "gone", "v1", "empty", None)]

    def test_mixed_present_and_missing(self, saver: BaseSingleStoreSaver) -> None:
        rows = saver._dump_blobs(
            "t",
            "ns",
            {"present": "x"},
            {"present": "v1", "missing": "v2"},
        )
        assert len(rows) == 2
        by_channel = {row[2]: row for row in rows}
        assert by_channel["missing"][4] == "empty"
        assert by_channel["missing"][5] is None
        assert by_channel["present"][4] == "msgpack"
        assert isinstance(by_channel["present"][5], (bytes, bytearray))


# ---------------------------------------------------------------- _load_blobs


class TestLoadBlobs:
    def test_empty_input_returns_empty_dict(self, saver: BaseSingleStoreSaver) -> None:
        assert saver._load_blobs([]) == {}

    def test_empty_type_rows_are_skipped(self, saver: BaseSingleStoreSaver) -> None:
        rows = [(b"missing", b"empty", b"")]
        assert saver._load_blobs(rows) == {}

    def test_round_trip_through_dump_and_load(
        self, saver: BaseSingleStoreSaver
    ) -> None:
        """Serialize a value with ``_dump_blobs`` then reconstruct it via
        ``_load_blobs``; the value must round-trip exactly."""
        dumped = saver._dump_blobs(
            "t", "ns", {"messages": {"role": "user"}}, {"messages": "v1"}
        )
        # _load_blobs expects (channel, type, blob) triples as bytes; the
        # value channel above is present, so `row[5]` is guaranteed non-None.
        load_input: list[tuple[bytes, bytes, bytes]] = [
            (row[2].encode(), row[4].encode(), cast(bytes, row[5])) for row in dumped
        ]
        loaded = saver._load_blobs(load_input)
        assert loaded == {"messages": {"role": "user"}}


# ---------------------------------------------------------------- _dump_writes


class TestDumpWrites:
    def test_regular_channel_uses_positional_idx(
        self, saver: BaseSingleStoreSaver
    ) -> None:
        rows = saver._dump_writes(
            "t", "ns", "cp1", "task1", "path/1", [("out", 1), ("out", 2)]
        )
        assert [row[5] for row in rows] == [0, 1]
        assert [row[6] for row in rows] == ["out", "out"]

    @pytest.mark.parametrize(
        ("channel", "expected_idx"),
        list(WRITES_IDX_MAP.items()),
    )
    def test_special_channels_use_writes_idx_map(
        self,
        saver: BaseSingleStoreSaver,
        channel: str,
        expected_idx: int,
    ) -> None:
        """``WRITES_IDX_MAP`` overrides the positional idx for reserved
        channels (``__error__``, ``__interrupt__`` etc.)."""
        rows = saver._dump_writes(
            "t", "ns", "cp1", "task1", "path/1", [(channel, "payload")]
        )
        assert rows[0][5] == expected_idx
        assert rows[0][6] == channel

    def test_row_shape_matches_upsert_sql_placeholders(
        self, saver: BaseSingleStoreSaver
    ) -> None:
        """The tuple must have 9 elements to match ``UPSERT_CHECKPOINT_WRITES_SQL``
        (thread, ns, cp, task, path, idx, channel, type, blob)."""
        rows = saver._dump_writes("t", "ns", "cp1", "task1", "path/1", [("out", 1)])
        assert len(rows[0]) == 9
        assert rows[0][:5] == ("t", "ns", "cp1", "task1", "path/1")
        assert rows[0][7] == "msgpack"
        assert isinstance(rows[0][8], (bytes, bytearray))


# ---------------------------------------------------------------- _load_writes


class TestLoadWrites:
    def test_empty_input_returns_empty_list(self, saver: BaseSingleStoreSaver) -> None:
        assert saver._load_writes([]) == []

    def test_falsy_input_returns_empty_list(self, saver: BaseSingleStoreSaver) -> None:
        """The implementation uses truthiness — ``None`` must not crash."""
        assert saver._load_writes(None) == []  # type: ignore[arg-type]

    def test_round_trip_through_dump_and_load(
        self, saver: BaseSingleStoreSaver
    ) -> None:
        dumped = saver._dump_writes(
            "t",
            "ns",
            "cp1",
            "task-abc",
            "path/1",
            [("messages", {"role": "user"})],
        )
        # _load_writes expects (task_id, channel, type, blob) as bytes.
        load_input = [
            (row[3].encode(), row[6].encode(), row[7].encode(), row[8])
            for row in dumped
        ]
        loaded = saver._load_writes(load_input)
        assert loaded == [("task-abc", "messages", {"role": "user"})]


# ------------------------------------------ _migrate_pending_sends


class TestMigratePendingSends:
    def test_empty_input_is_noop(self, saver: BaseSingleStoreSaver) -> None:
        checkpoint: dict[str, Any] = {"channel_versions": {"existing": "1"}}
        channel_values: list[tuple[bytes, bytes, bytes]] = []
        saver._migrate_pending_sends([], checkpoint, channel_values)
        assert channel_values == []
        assert TASKS not in checkpoint["channel_versions"]

    def test_appends_tasks_channel_value(self, saver: BaseSingleStoreSaver) -> None:
        """Migration serializes the send list into one ``channel_values``
        entry keyed by ``TASKS``."""
        # A "send" is a (type, blob) pair as bytes coming from the DB.
        type_tag, blob = saver.serde.dumps_typed("send-1")
        pending = [(type_tag.encode(), blob)]

        checkpoint: dict[str, Any] = {"channel_versions": {"other": "v42"}}
        channel_values: list[tuple[bytes, bytes, bytes]] = []
        saver._migrate_pending_sends(pending, checkpoint, channel_values)

        assert len(channel_values) == 1
        appended_channel, appended_type, _appended_blob = channel_values[0]
        assert appended_channel == TASKS.encode()
        assert appended_type == b"msgpack"

    def test_tasks_version_is_max_of_existing_when_populated(
        self, saver: BaseSingleStoreSaver
    ) -> None:
        type_tag, blob = saver.serde.dumps_typed("s")
        pending = [(type_tag.encode(), blob)]
        checkpoint: dict[str, Any] = {"channel_versions": {"a": "v1", "b": "v9"}}

        saver._migrate_pending_sends(pending, checkpoint, [])
        assert checkpoint["channel_versions"][TASKS] == "v9"

    def test_tasks_version_falls_back_to_get_next_version_when_empty(
        self, saver: BaseSingleStoreSaver
    ) -> None:
        type_tag, blob = saver.serde.dumps_typed("s")
        pending = [(type_tag.encode(), blob)]
        checkpoint: dict[str, Any] = {"channel_versions": {}}

        saver._migrate_pending_sends(pending, checkpoint, [])
        assigned = checkpoint["channel_versions"][TASKS]
        # get_next_version(None, None) yields "...0001.<random>"
        assert assigned.split(".")[0].endswith("1")
        assert len(assigned.split(".")[0]) == 32

    def test_hex_encoded_blob_is_decoded(self, saver: BaseSingleStoreSaver) -> None:
        """The SingleStore driver returns ``HEX(blob)`` as an ASCII string;
        the migration must call ``bytes.fromhex`` on it."""
        type_tag, raw_blob = saver.serde.dumps_typed("send-hex")
        hex_blob = raw_blob.hex()  # str
        # The runtime code accepts str for the blob (HEX(...) returns text);
        # the declared type is bytes, so cast for mypy.
        pending = cast("list[tuple[bytes, bytes]]", [(type_tag.encode(), hex_blob)])

        checkpoint: dict[str, Any] = {"channel_versions": {}}
        channel_values: list[tuple[bytes, bytes, bytes]] = []
        saver._migrate_pending_sends(pending, checkpoint, channel_values)
        assert len(channel_values) == 1


# ---------------------------------------------------------------- _search_where


class TestSearchWhere:
    def test_all_none_yields_empty_where(self, saver: BaseSingleStoreSaver) -> None:
        sql, params = saver._search_where(None, None, None)
        assert sql == ""
        assert params == []

    def test_thread_id_only(self, saver: BaseSingleStoreSaver) -> None:
        config = cast(RunnableConfig, {"configurable": {"thread_id": "t1"}})
        sql, params = saver._search_where(config, None)
        assert "c.thread_id = %s" in sql
        assert params == ["t1"]

    def test_thread_id_and_checkpoint_ns(self, saver: BaseSingleStoreSaver) -> None:
        config = cast(
            RunnableConfig,
            {"configurable": {"thread_id": "t1", "checkpoint_ns": "ns1"}},
        )
        sql, params = saver._search_where(config, None)
        assert "c.thread_id = %s" in sql
        assert "c.checkpoint_ns = %s" in sql
        assert params == ["t1", "ns1"]

    def test_empty_string_checkpoint_ns_is_included(
        self, saver: BaseSingleStoreSaver
    ) -> None:
        """``checkpoint_ns == ''`` is a real value (default namespace) and
        must be included — only ``None`` is skipped."""
        config = cast(
            RunnableConfig,
            {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}},
        )
        sql, params = saver._search_where(config, None)
        assert "c.checkpoint_ns = %s" in sql
        assert params == ["t1", ""]

    def test_checkpoint_id_from_config(self, saver: BaseSingleStoreSaver) -> None:
        config = cast(
            RunnableConfig,
            {"configurable": {"thread_id": "t1", "checkpoint_id": "cp-1"}},
        )
        sql, params = saver._search_where(config, None)
        assert "c.checkpoint_id = %s" in sql
        assert params == ["t1", "cp-1"]

    def test_before_adds_less_than_predicate(self, saver: BaseSingleStoreSaver) -> None:
        before = cast(RunnableConfig, {"configurable": {"checkpoint_id": "cp-99"}})
        sql, params = saver._search_where(None, None, before)
        assert "c.checkpoint_id < %s" in sql
        assert params == ["cp-99"]

    def test_filter_wraps_into_and_and_calls_parse_filter(
        self, saver: BaseSingleStoreSaver
    ) -> None:
        """The metadata filter is wrapped as ``{"$and": [...]}`` and handed to
        ``_parse_filter`` with ``metadata_field="c.metadata"``."""
        sql, params = saver._search_where(None, {"status": "active"})
        assert sql.startswith("WHERE ")
        assert "c.metadata" in sql
        # _parse_filter returns positional params: [value, field].
        assert params == ["active", "status"]

    def test_filter_multi_key_generates_and_conditions(
        self, saver: BaseSingleStoreSaver
    ) -> None:
        sql, params = saver._search_where(None, {"a": 1, "b": 2})
        assert " AND " in sql
        assert params == [1, "a", 2, "b"]

    def test_empty_filter_is_treated_as_no_filter(
        self, saver: BaseSingleStoreSaver
    ) -> None:
        sql, params = saver._search_where(None, {})
        assert sql == ""
        assert params == []

    def test_where_keyword_only_when_predicates_exist(
        self, saver: BaseSingleStoreSaver
    ) -> None:
        config = cast(RunnableConfig, {"configurable": {"thread_id": "t"}})
        sql, _ = saver._search_where(config, None)
        assert sql.startswith("WHERE ")

    def test_predicates_are_joined_with_and(self, saver: BaseSingleStoreSaver) -> None:
        config = cast(
            RunnableConfig,
            {
                "configurable": {
                    "thread_id": "t1",
                    "checkpoint_ns": "ns1",
                    "checkpoint_id": "cp-1",
                }
            },
        )
        before = cast(RunnableConfig, {"configurable": {"checkpoint_id": "cp-99"}})
        sql, params = saver._search_where(config, {"k": "v"}, before)
        # Every predicate present, and joined by AND.
        assert sql.count(" AND ") >= 4
        # Config params come first, then filter (value, field), then before.
        assert params[:3] == ["t1", "ns1", "cp-1"]
        assert params[-1] == "cp-99"
        assert "v" in params and "k" in params
