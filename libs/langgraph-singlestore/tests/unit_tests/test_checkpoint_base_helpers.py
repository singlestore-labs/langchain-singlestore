"""Unit tests for module-level helper functions in
:mod:`langgraph_singlestore.checkpoint.base`.

These helpers translate the raw output of the checkpoint SQL statements
(tuple/dict rows, JSON columns that may or may not have been decoded by the
driver, ``JSON_AGG`` arrays containing ``HEX(blob)`` strings) into the byte
shapes the base saver expects.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from langgraph_singlestore.checkpoint.base import (
    _as_dict,
    _as_list,
    _parse_channel_values,
    _parse_pending_writes,
    _parse_sends,
    _row_get,
    _row_to_checkpoint_dict,
)

# ---------------------------------------------------------------- _row_get


class TestRowGet:
    def test_tuple_row_is_indexed_positionally(self) -> None:
        row = ("t1", "ck-1", "ns")
        assert _row_get(row, 0, "thread_id") == "t1"
        assert _row_get(row, 1, "checkpoint_id") == "ck-1"

    def test_dict_row_is_indexed_by_name(self) -> None:
        row = {"thread_id": "t1", "checkpoint_id": "ck-1"}
        assert _row_get(row, 0, "thread_id") == "t1"
        assert _row_get(row, 999, "checkpoint_id") == "ck-1"

    def test_list_row_falls_back_to_positional(self) -> None:
        """Non-dict sequences must go through positional access."""
        row = ["a", "b", "c"]
        assert _row_get(row, 2, "ignored") == "c"

    def test_dict_row_missing_key_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            _row_get({"x": 1}, 0, "y")


# ---------------------------------------------------------------- _as_dict


class TestAsDict:
    def test_none_yields_empty_dict(self) -> None:
        assert _as_dict(None) == {}

    def test_dict_input_is_returned_unchanged(self) -> None:
        """The driver may already have decoded the JSON column."""
        d = {"k": "v"}
        assert _as_dict(d) is d

    def test_json_string_is_parsed(self) -> None:
        assert _as_dict('{"k": 1, "n": null}') == {"k": 1, "n": None}

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _as_dict("not-json")


# ---------------------------------------------------------------- _as_list


class TestAsList:
    def test_none_yields_empty_list(self) -> None:
        assert _as_list(None) == []

    def test_list_input_is_returned_unchanged(self) -> None:
        raw = [1, 2, 3]
        assert _as_list(raw) is raw

    def test_json_string_array_is_parsed(self) -> None:
        assert _as_list('[1, 2, "x"]') == [1, 2, "x"]

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _as_list("[oops")


# --------------------------------------------------- _parse_channel_values


class TestParseChannelValues:
    def test_none_yields_empty_list(self) -> None:
        assert _parse_channel_values(None) == []

    def test_str_entries_are_encoded_and_hex_is_decoded(self) -> None:
        entry = ["messages", "msgpack", "68656c6c6f"]  # "hello" hex-encoded
        assert _parse_channel_values([entry]) == [(b"messages", b"msgpack", b"hello")]

    def test_accepts_json_encoded_string(self) -> None:
        """Handles the driver-not-decoded JSON case."""
        raw = json.dumps([["ch", "msgpack", "61"]])
        assert _parse_channel_values(raw) == [(b"ch", b"msgpack", b"a")]

    def test_null_channel_entry_is_skipped(self) -> None:
        """``JSON_AGG`` with no matching blobs emits a single null-channel
        row; the parser must drop it."""
        rows = [[None, None, None], ["ch", "msgpack", "00"]]
        assert _parse_channel_values(rows) == [(b"ch", b"msgpack", b"\x00")]

    def test_bytes_inputs_pass_through_without_re_encoding(self) -> None:
        """If the driver already returns bytes, no double-encoding occurs."""
        entry = [b"messages", b"msgpack", b"hello"]
        assert _parse_channel_values([entry]) == [(b"messages", b"msgpack", b"hello")]

    def test_multiple_entries_preserve_order(self) -> None:
        rows = [
            ["a", "msgpack", "01"],
            ["b", "msgpack", "02"],
            ["c", "msgpack", "03"],
        ]
        out = _parse_channel_values(rows)
        assert [t[0] for t in out] == [b"a", b"b", b"c"]
        assert [t[2] for t in out] == [b"\x01", b"\x02", b"\x03"]


# ---------------------------------------------------- _parse_pending_writes


class TestParsePendingWrites:
    def test_none_yields_empty_list(self) -> None:
        assert _parse_pending_writes(None) == []

    def test_str_entries_are_encoded_and_hex_is_decoded(self) -> None:
        entry = ["task-1", "messages", "msgpack", "68656c6c6f"]
        assert _parse_pending_writes([entry]) == [
            (b"task-1", b"messages", b"msgpack", b"hello")
        ]

    def test_accepts_json_encoded_string(self) -> None:
        raw = json.dumps([["task", "ch", "msgpack", "61"]])
        assert _parse_pending_writes(raw) == [(b"task", b"ch", b"msgpack", b"a")]

    def test_bytes_inputs_pass_through(self) -> None:
        entry = [b"task-1", b"messages", b"msgpack", b"hello"]
        assert _parse_pending_writes([entry]) == [
            (b"task-1", b"messages", b"msgpack", b"hello")
        ]

    def test_multiple_entries_preserve_order(self) -> None:
        rows = [
            ["t1", "a", "msgpack", "01"],
            ["t2", "b", "msgpack", "02"],
        ]
        out = _parse_pending_writes(rows)
        assert [t[0] for t in out] == [b"t1", b"t2"]
        assert [t[3] for t in out] == [b"\x01", b"\x02"]


# ---------------------------------------------------------------- _parse_sends


class TestParseSends:
    def test_none_yields_empty_list(self) -> None:
        assert _parse_sends(None) == []

    def test_type_is_encoded_and_hex_str_is_preserved(self) -> None:
        """Sends are handed to ``_migrate_pending_sends`` which itself
        performs the hex→bytes conversion, so we leave the blob untouched."""
        out = _parse_sends([["msgpack", "68656c6c6f"]])
        assert out == [(b"msgpack", "68656c6c6f")]  # type: ignore[comparison-overlap]

    def test_accepts_json_encoded_string(self) -> None:
        raw = json.dumps([["msgpack", "aa"]])
        assert _parse_sends(raw) == [(b"msgpack", "aa")]  # type: ignore[comparison-overlap]

    def test_bytes_type_passes_through(self) -> None:
        out = _parse_sends([[b"msgpack", b"raw"]])
        assert out == [(b"msgpack", b"raw")]


# --------------------------------------------------- _row_to_checkpoint_dict


def _row(
    *,
    thread_id: str = "t1",
    checkpoint: Any = None,
    checkpoint_ns: str = "",
    checkpoint_id: str = "cp-1",
    parent_checkpoint_id: Any = None,
    metadata: Any = None,
    channel_values: Any = None,
    pending_writes: Any = None,
) -> tuple[Any, ...]:
    """Build a raw row in the shape ``SELECT_SQL`` returns."""
    return (
        thread_id,
        checkpoint if checkpoint is not None else {"v": 4, "id": checkpoint_id},
        checkpoint_ns,
        checkpoint_id,
        parent_checkpoint_id,
        metadata if metadata is not None else {},
        channel_values,
        pending_writes,
    )


class TestRowToCheckpointDict:
    def test_tuple_row_full_shape(self) -> None:
        row = _row(
            checkpoint={"v": 4, "id": "cp-1", "channel_values": {"k": 1}},
            metadata={"source": "input"},
            parent_checkpoint_id="cp-0",
            channel_values=[["ch", "msgpack", "00"]],
            pending_writes=[["task-1", "ch", "msgpack", "01"]],
        )
        parsed = _row_to_checkpoint_dict(row)

        assert parsed["thread_id"] == "t1"
        assert parsed["checkpoint"] == {
            "v": 4,
            "id": "cp-1",
            "channel_values": {"k": 1},
        }
        assert parsed["checkpoint_ns"] == ""
        assert parsed["checkpoint_id"] == "cp-1"
        assert parsed["parent_checkpoint_id"] == "cp-0"
        assert parsed["metadata"] == {"source": "input"}
        assert parsed["channel_values"] == [(b"ch", b"msgpack", b"\x00")]
        assert parsed["pending_writes"] == [(b"task-1", b"ch", b"msgpack", b"\x01")]

    def test_dict_row_is_indexed_by_name(self) -> None:
        """When the driver returns dict rows, columns are pulled by name."""
        row = {
            "thread_id": "t2",
            "checkpoint": '{"v": 4, "id": "cp-2"}',
            "checkpoint_ns": "ns",
            "checkpoint_id": "cp-2",
            "parent_checkpoint_id": None,
            "metadata": '{"src": "x"}',
            "channel_values": '[["ch", "msgpack", "00"]]',
            "pending_writes": None,
        }
        parsed = _row_to_checkpoint_dict(row)

        assert parsed["thread_id"] == "t2"
        assert parsed["checkpoint"] == {"v": 4, "id": "cp-2"}
        assert parsed["metadata"] == {"src": "x"}
        assert parsed["channel_values"] == [(b"ch", b"msgpack", b"\x00")]
        assert parsed["pending_writes"] == []

    def test_null_json_columns_default_to_empty(self) -> None:
        """A row with no matching blobs or writes still parses cleanly."""
        row = _row(
            checkpoint=None,
            metadata=None,
            channel_values=None,
            pending_writes=None,
        )
        parsed = _row_to_checkpoint_dict(row)

        assert parsed["metadata"] == {}
        assert parsed["channel_values"] == []
        assert parsed["pending_writes"] == []

    def test_json_string_columns_are_decoded(self) -> None:
        row = _row(
            checkpoint='{"v": 4, "id": "cp-1"}',
            metadata='{"source": "loop"}',
        )
        parsed = _row_to_checkpoint_dict(row)
        assert parsed["checkpoint"] == {"v": 4, "id": "cp-1"}
        assert parsed["metadata"] == {"source": "loop"}
