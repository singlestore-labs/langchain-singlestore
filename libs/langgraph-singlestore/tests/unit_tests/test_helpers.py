"""Unit tests for helper functions in :mod:`langgraph.store.singlestore.base`.

Covers helpers not exercised by ``test_search_where`` or
``test_namespace_escaping``:

* ``_group_ops`` — grouping heterogeneous ops by concrete class
* ``_row_get`` — dict / tuple row column access
* ``_row_to_item`` — row -> :class:`Item`
* ``_row_to_search_item`` — row -> :class:`SearchItem`
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    PutOp,
    SearchItem,
    SearchOp,
)
from langgraph.store.singlestore.base import (
    _group_ops,
    _row_get,
    _row_to_item,
    _row_to_search_item,
)

# ---------------------------------------------------------------- _group_ops


class TestGroupOps:
    def test_empty_iterable_yields_empty_grouping(self) -> None:
        grouped, total = _group_ops([])
        assert grouped == {}
        assert total == 0

    def test_groups_by_concrete_op_type(self) -> None:
        ops = [
            GetOp(("users", "alice"), "prefs"),
            PutOp(("users", "alice"), "prefs", {"theme": "dark"}),
            GetOp(("users", "bob"), "prefs"),
            SearchOp(namespace_prefix=("users",)),
            ListNamespacesOp(),
            PutOp(("users", "bob"), "prefs", None),
        ]
        grouped, total = _group_ops(ops)

        assert total == len(ops)
        assert set(grouped.keys()) == {GetOp, PutOp, SearchOp, ListNamespacesOp}
        assert len(grouped[GetOp]) == 2
        assert len(grouped[PutOp]) == 2
        assert len(grouped[SearchOp]) == 1
        assert len(grouped[ListNamespacesOp]) == 1

    def test_preserves_original_index_within_each_group(self) -> None:
        """The (idx, op) pairs must retain the caller's original position so
        results can be scattered back into the correct slots."""
        ops = [
            GetOp(("a",), "1"),
            PutOp(("a",), "1", {"x": 1}),
            GetOp(("a",), "2"),
            PutOp(("a",), "2", {"x": 2}),
            GetOp(("a",), "3"),
        ]
        grouped, total = _group_ops(ops)

        assert total == 5
        assert [idx for idx, _ in grouped[GetOp]] == [0, 2, 4]
        assert [idx for idx, _ in grouped[PutOp]] == [1, 3]
        # The op payloads must be the exact original objects.
        assert [op for _, op in grouped[GetOp]] == [ops[0], ops[2], ops[4]]
        assert [op for _, op in grouped[PutOp]] == [ops[1], ops[3]]

    def test_accepts_generator_input(self) -> None:
        """``_group_ops`` takes ``Iterable[Op]``, so a one-shot generator works."""

        def gen() -> Any:
            yield GetOp(("a",), "1")
            yield PutOp(("a",), "1", {"x": 1})

        grouped, total = _group_ops(gen())
        assert total == 2
        assert len(grouped[GetOp]) == 1
        assert len(grouped[PutOp]) == 1


# ---------------------------------------------------------------- _row_get


class TestRowGet:
    def test_reads_by_index_from_tuple(self) -> None:
        row = ("users/alice", "prefs", '{"theme": "dark"}')
        assert _row_get(row, 0, "prefix") == "users/alice"
        assert _row_get(row, 1, "key") == "prefs"
        assert _row_get(row, 2, "value") == '{"theme": "dark"}'

    def test_reads_by_index_from_list(self) -> None:
        row = ["users/alice", "prefs", '{"theme": "dark"}']
        assert _row_get(row, 0, "prefix") == "users/alice"
        assert _row_get(row, 1, "key") == "prefs"

    def test_reads_by_name_from_dict(self) -> None:
        row = {
            "prefix": "users/alice",
            "key": "prefs",
            "value": '{"theme": "dark"}',
        }
        assert _row_get(row, 0, "prefix") == "users/alice"
        assert _row_get(row, 1, "key") == "prefs"
        assert _row_get(row, 2, "value") == '{"theme": "dark"}'

    def test_dict_ignores_index_argument(self) -> None:
        """Positional index is ignored for dict rows — only the name matters."""
        row = {"prefix": "users/alice", "key": "prefs"}
        # A deliberately wrong index still resolves correctly by name.
        assert _row_get(row, 99, "prefix") == "users/alice"

    def test_tuple_ignores_name_argument(self) -> None:
        """Column name is ignored for tuple rows — only the index matters."""
        row = ("users/alice", "prefs")
        # A deliberately wrong name still resolves correctly by index.
        assert _row_get(row, 0, "not_a_column") == "users/alice"

    def test_missing_dict_key_raises_key_error(self) -> None:
        row = {"prefix": "users/alice"}
        with pytest.raises(KeyError):
            _row_get(row, 1, "key")

    def test_out_of_range_tuple_index_raises_index_error(self) -> None:
        row = ("users/alice",)
        with pytest.raises(IndexError):
            _row_get(row, 5, "prefix")


# ------------------------------------------------------------ row conversion


# Column order used by ``_SELECT_BASE``:
#   0: prefix   1: key   2: value   3: created_at   4: updated_at
#   5: expires_at   6: ttl_minutes
_CREATED_AT = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
_UPDATED_AT = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)


def _tuple_row(value: Any) -> tuple[Any, ...]:
    return ("users/alice", "prefs", value, _CREATED_AT, _UPDATED_AT, None, None)


def _dict_row(value: Any) -> dict[str, Any]:
    return {
        "prefix": "users/alice",
        "key": "prefs",
        "value": value,
        "created_at": _CREATED_AT,
        "updated_at": _UPDATED_AT,
        "expires_at": None,
        "ttl_minutes": None,
    }


class TestRowToItem:
    def test_tuple_row_with_json_string_value_is_parsed(self) -> None:
        raw_value = json.dumps({"theme": "dark", "lang": "en"})
        item = _row_to_item(("users", "alice"), _tuple_row(raw_value))

        assert isinstance(item, Item)
        assert item.namespace == ("users", "alice")
        assert item.key == "prefs"
        assert item.value == {"theme": "dark", "lang": "en"}
        assert item.created_at == _CREATED_AT
        assert item.updated_at == _UPDATED_AT

    def test_tuple_row_with_dict_value_is_passed_through(self) -> None:
        """Some drivers return ``JSON`` columns already deserialised."""
        value = {"theme": "dark", "lang": "en"}
        item = _row_to_item(("users", "alice"), _tuple_row(value))
        # Must be the same content — helper shouldn't double-decode.
        assert item.value == value

    def test_dict_row_with_json_string_value(self) -> None:
        raw_value = json.dumps({"n": 1})
        item = _row_to_item(("t",), _dict_row(raw_value))
        assert item.value == {"n": 1}
        assert item.key == "prefs"

    def test_dict_row_with_dict_value(self) -> None:
        item = _row_to_item(("t",), _dict_row({"n": 2}))
        assert item.value == {"n": 2}

    def test_value_with_json_bytes_is_parsed(self) -> None:
        """``json.loads`` accepts bytes — mirror that for driver flexibility."""
        raw_value = json.dumps({"a": 1}).encode()
        item = _row_to_item(("t",), _tuple_row(raw_value))
        assert item.value == {"a": 1}


class TestRowToSearchItem:
    def test_tuple_row_with_json_string_value_is_parsed(self) -> None:
        raw_value = json.dumps({"score": 5})
        item = _row_to_search_item(("docs",), _tuple_row(raw_value))

        assert isinstance(item, SearchItem)
        assert item.namespace == ("docs",)
        assert item.key == "prefs"
        assert item.value == {"score": 5}
        assert item.created_at == _CREATED_AT
        assert item.updated_at == _UPDATED_AT
        # No vector search yet — score is always None.
        assert item.score is None

    def test_dict_row_with_dict_value(self) -> None:
        item = _row_to_search_item(("docs",), _dict_row({"score": 3}))
        assert isinstance(item, SearchItem)
        assert item.value == {"score": 3}
        assert item.score is None

    def test_namespace_argument_wins_over_row_prefix(self) -> None:
        """The caller-supplied ``namespace`` is authoritative — the row's
        ``prefix`` column is not re-parsed here."""
        item = _row_to_search_item(
            ("caller", "provided"), _tuple_row(json.dumps({"x": 1}))
        )
        assert item.namespace == ("caller", "provided")
