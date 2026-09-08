"""Unit tests for :func:`langgraph.store.singlestore.base._search_where`.

The function is a pure translator from ``SearchOp`` to a ``WHERE`` SQL
fragment plus positional parameters. These tests exercise every branch
of the function (namespace prefix, filter, combinations, empty inputs)
and every filter operator supported by
``singlestore_langchain_core._filter._parse_filter`` so that the wrapping
into ``{"$and": [{k: v}, ...]}`` is validated end to end.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from langgraph.store.base import SearchOp
from langgraph.store.singlestore.base import (
    _namespace_for_exact_search,
    _namespace_for_prefix_search,
    _search_where,
)

# ---------------------------------------------------------------- helpers


def _op(
    namespace_prefix: tuple[str, ...] = (),
    filter: dict[str, Any] | None = None,
) -> SearchOp:
    return SearchOp(namespace_prefix=namespace_prefix, filter=filter)


# ---------------------------------------------------------------- no clauses


def test_empty_op_yields_empty_where() -> None:
    where_sql, params = _search_where(_op())
    assert where_sql == ""
    assert params == []


def test_empty_filter_dict_is_ignored() -> None:
    where_sql, params = _search_where(_op(filter={}))
    assert where_sql == ""
    assert params == []


def test_none_filter_is_ignored() -> None:
    where_sql, params = _search_where(_op(namespace_prefix=(), filter=None))
    assert where_sql == ""
    assert params == []


# ---------------------------------------------------------------- prefix only


@pytest.mark.parametrize(
    "namespace_prefix",
    [
        ("users",),
        ("users", "alice"),
        ("users", "alice", "prefs"),
        # Segments containing the separator, escape char, and LIKE wildcards.
        ("a/b",),
        ("a\\b", "c"),
        ("100%",),
        ("under_score",),
    ],
)
def test_prefix_only_produces_like_or_eq_clause(
    namespace_prefix: tuple[str, ...],
) -> None:
    """A concrete (non-wildcard) prefix matches descendants *or* the exact row."""
    where_sql, params = _search_where(_op(namespace_prefix=namespace_prefix))
    assert where_sql == "(prefix LIKE %s OR prefix = %s)"
    exact_filter, exact_param = _namespace_for_exact_search(namespace_prefix)
    assert exact_filter == "prefix = %s"
    assert params == [_namespace_for_prefix_search(namespace_prefix), exact_param]


@pytest.mark.parametrize(
    "namespace_prefix",
    [
        ("users", "*"),
        ("*", "alice"),
        ("*",),
        ("users", "*", "prefs"),
    ],
)
def test_prefix_with_wildcard_produces_two_like_clauses(
    namespace_prefix: tuple[str, ...],
) -> None:
    """A wildcard prefix uses ``LIKE`` on both sides of the ``OR``."""
    where_sql, params = _search_where(_op(namespace_prefix=namespace_prefix))
    assert where_sql == "(prefix LIKE %s OR prefix LIKE %s)"
    exact_filter, exact_param = _namespace_for_exact_search(namespace_prefix)
    assert exact_filter == "prefix LIKE %s"
    assert params == [_namespace_for_prefix_search(namespace_prefix), exact_param]


# ---------------------------------------------------------------- filter only


def test_filter_exact_match_string() -> None:
    where_sql, params = _search_where(_op(filter={"status": "active"}))
    assert where_sql == (
        "(JSON_MATCH_ANY(MATCH_PARAM_STRING_STRICT() = %s, value, %s))"
    )
    assert params == ["active", "status"]


@pytest.mark.parametrize(
    ("value", "expected_match_func"),
    [
        ("active", "MATCH_PARAM_STRING_STRICT()"),
        (42, "MATCH_PARAM_DOUBLE_STRICT()"),
        (3.14, "MATCH_PARAM_DOUBLE_STRICT()"),
        (True, "MATCH_PARAM_BOOL_STRICT()"),
        (False, "MATCH_PARAM_BOOL_STRICT()"),
    ],
)
def test_filter_exact_match_value_types(value: Any, expected_match_func: str) -> None:
    where_sql, params = _search_where(_op(filter={"field": value}))
    assert where_sql == (f"(JSON_MATCH_ANY({expected_match_func} = %s, value, %s))")
    assert params == [value, "field"]


def test_filter_multiple_keys_are_anded_in_insertion_order() -> None:
    where_sql, params = _search_where(_op(filter={"status": "active", "score": 5}))
    assert where_sql == (
        "(JSON_MATCH_ANY(MATCH_PARAM_STRING_STRICT() = %s, value, %s)"
        " AND "
        "JSON_MATCH_ANY(MATCH_PARAM_DOUBLE_STRICT() = %s, value, %s))"
    )
    assert params == ["active", "status", 5, "score"]


def test_filter_eq_operator() -> None:
    where_sql, params = _search_where(_op(filter={"status": {"$eq": "active"}}))
    assert where_sql == (
        "(JSON_MATCH_ANY(MATCH_PARAM_STRING_STRICT() = %s, value, %s))"
    )
    assert params == ["active", "status"]


def test_filter_ne_operator() -> None:
    where_sql, params = _search_where(_op(filter={"status": {"$ne": "active"}}))
    assert where_sql == (
        "(NOT JSON_MATCH_ANY(MATCH_PARAM_STRING_STRICT() = %s, value, %s)"
        " AND JSON_MATCH_ANY_EXISTS(value, %s))"
    )
    assert params == ["active", "status", "status"]


@pytest.mark.parametrize(
    ("operator", "sql_op"),
    [("$gt", ">"), ("$gte", ">="), ("$lt", "<"), ("$lte", "<=")],
)
def test_filter_numeric_comparison_operators(operator: str, sql_op: str) -> None:
    where_sql, params = _search_where(_op(filter={"score": {operator: 4.99}}))
    assert where_sql == f"(JSON_EXTRACT_DOUBLE(value, %s) {sql_op} %s)"
    assert params == ["score", 4.99]


def test_filter_in_operator() -> None:
    where_sql, params = _search_where(_op(filter={"tag": {"$in": ["a", "b", "c"]}}))
    assert where_sql == (
        "(JSON_MATCH_ANY(JSON_ARRAY_CONTAINS_JSON(%s, MATCH_PARAM_JSON()),"
        " value, %s))"
    )
    assert params == [json.dumps(["a", "b", "c"]), "tag"]


def test_filter_nin_operator() -> None:
    where_sql, params = _search_where(_op(filter={"tag": {"$nin": ["a", "b"]}}))
    assert where_sql == (
        "(NOT JSON_MATCH_ANY(JSON_ARRAY_CONTAINS_JSON(%s, MATCH_PARAM_JSON()),"
        " value, %s) AND JSON_MATCH_ANY_EXISTS(value, %s))"
    )
    assert params == [json.dumps(["a", "b"]), "tag", "tag"]


def test_filter_exists_true() -> None:
    where_sql, params = _search_where(_op(filter={"tag": {"$exists": True}}))
    assert where_sql == "(JSON_MATCH_ANY_EXISTS(value, %s))"
    assert params == ["tag"]


def test_filter_exists_false() -> None:
    where_sql, params = _search_where(_op(filter={"tag": {"$exists": False}}))
    assert where_sql == "(NOT JSON_MATCH_ANY_EXISTS(value, %s))"
    assert params == ["tag"]


def test_filter_mixes_exact_and_operator_conditions() -> None:
    where_sql, params = _search_where(
        _op(filter={"status": "active", "score": {"$gte": 3.0}})
    )
    assert where_sql == (
        "(JSON_MATCH_ANY(MATCH_PARAM_STRING_STRICT() = %s, value, %s)"
        " AND "
        "JSON_EXTRACT_DOUBLE(value, %s) >= %s)"
    )
    assert params == ["active", "status", "score", 3.0]


# ---------------------------------------------------------------- combined


def test_prefix_and_filter_are_anded_in_order() -> None:
    where_sql, params = _search_where(
        _op(
            namespace_prefix=("users", "alice"),
            filter={"status": "active"},
        )
    )
    assert where_sql == (
        "(prefix LIKE %s OR prefix = %s)"
        " AND "
        "(JSON_MATCH_ANY(MATCH_PARAM_STRING_STRICT() = %s, value, %s))"
    )
    _, exact_param = _namespace_for_exact_search(("users", "alice"))
    assert params == [
        _namespace_for_prefix_search(("users", "alice")),
        exact_param,
        "active",
        "status",
    ]


def test_prefix_and_multi_condition_filter() -> None:
    where_sql, params = _search_where(
        _op(
            namespace_prefix=("docs",),
            filter={"type": "report", "score": {"$gt": 4}},
        )
    )
    assert where_sql == (
        "(prefix LIKE %s OR prefix = %s)"
        " AND "
        "(JSON_MATCH_ANY(MATCH_PARAM_STRING_STRICT() = %s, value, %s)"
        " AND "
        "JSON_EXTRACT_DOUBLE(value, %s) > %s)"
    )
    _, exact_param = _namespace_for_exact_search(("docs",))
    assert params == [
        _namespace_for_prefix_search(("docs",)),
        exact_param,
        "report",
        "type",
        "score",
        4,
    ]


def test_wildcard_prefix_and_filter_combined() -> None:
    """Wildcard prefix + filter still ANDs the prefix clause with the filter."""
    where_sql, params = _search_where(
        _op(namespace_prefix=("users", "*"), filter={"active": True})
    )
    assert where_sql == (
        "(prefix LIKE %s OR prefix LIKE %s)"
        " AND "
        "(JSON_MATCH_ANY(MATCH_PARAM_BOOL_STRICT() = %s, value, %s))"
    )
    _, exact_param = _namespace_for_exact_search(("users", "*"))
    assert params == [
        _namespace_for_prefix_search(("users", "*")),
        exact_param,
        True,
        "active",
    ]


# ---------------------------------------------------------------- errors


def test_filter_rejects_unsupported_operator() -> None:
    with pytest.raises(ValueError, match="Unsupported operator"):
        _search_where(_op(filter={"field": {"$bogus": 1}}))


def test_filter_rejects_non_numeric_gt() -> None:
    with pytest.raises(ValueError, match=r"\$gt must be a numeric value"):
        _search_where(_op(filter={"field": {"$gt": "nope"}}))


def test_filter_rejects_non_list_in() -> None:
    with pytest.raises(ValueError, match=r"\$in must be a list"):
        _search_where(_op(filter={"field": {"$in": "nope"}}))


def test_filter_rejects_non_bool_exists() -> None:
    with pytest.raises(ValueError, match=r"\$exists must be a boolean"):
        _search_where(_op(filter={"field": {"$exists": "yes"}}))
