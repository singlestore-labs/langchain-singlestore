"""Tests for the _filter module."""

# mypy: disable-error-code=arg-type
import json

import pytest

from langchain_singlestore._filter import _get_match_param_function, _parse_filter


# Tests for _get_match_param_function
def test_get_match_param_function() -> None:
    # Test string values
    assert _get_match_param_function("test") == "MATCH_PARAM_STRING_STRICT()"

    # Test numeric values
    assert _get_match_param_function(10) == "MATCH_PARAM_DOUBLE_STRICT()"
    assert _get_match_param_function(10.5) == "MATCH_PARAM_DOUBLE_STRICT()"

    # Test boolean values
    assert _get_match_param_function(True) == "MATCH_PARAM_BOOL_STRICT()"
    assert _get_match_param_function(False) == "MATCH_PARAM_BOOL_STRICT()"

    # Test unsupported value
    with pytest.raises(ValueError, match="Unsupported value type"):
        _get_match_param_function([1, 2, 3])
    with pytest.raises(ValueError, match="Unsupported value type"):
        _get_match_param_function({"key": "value"})


# Tests for _parse_filter - Simple Filters
def test_exact_match_filter() -> None:
    filter_dict = {"field": "value"}
    query, params = _parse_filter(filter_dict, "metadata_field")
    assert (
        query == "JSON_MATCH_ANY(MATCH_PARAM_STRING_STRICT() = %s, metadata_field, %s)"
    )
    assert params == ["value", "field"]

    filter_dict = {"field": 123}  # type: ignore[dict-item]
    query, params = _parse_filter(filter_dict, "metadata_field")
    assert (
        query == "JSON_MATCH_ANY(MATCH_PARAM_DOUBLE_STRICT() = %s, metadata_field, %s)"
    )
    assert params == [123, "field"]

    filter_dict = {"field": True}  # type: ignore[dict-item]
    query, params = _parse_filter(filter_dict, "metadata_field")
    assert query == "JSON_MATCH_ANY(MATCH_PARAM_BOOL_STRICT() = %s, metadata_field, %s)"
    assert params == [True, "field"]


def test_eq_filter() -> None:
    filter_dict = {"field": {"$eq": "value"}}
    query, params = _parse_filter(filter_dict, "metadata_field")
    assert (
        query == "JSON_MATCH_ANY(MATCH_PARAM_STRING_STRICT() = %s, metadata_field, %s)"
    )
    assert params == ["value", "field"]


def test_ne_filter() -> None:
    filter_dict = {"field": {"$ne": "value"}}
    query, params = _parse_filter(filter_dict, "metadata_field")
    assert (
        query
        == "NOT JSON_MATCH_ANY(MATCH_PARAM_STRING_STRICT() = %s, metadata_field, %s)"
        " AND JSON_MATCH_ANY_EXISTS(metadata_field, %s)"
    )
    assert params == ["value", "field", "field"]


# Tests for numeric comparison filters
def test_gt_filter() -> None:
    filter_dict = {"field": {"$gt": 10}}
    query, params = _parse_filter(filter_dict, "metadata_field")
    assert query == "JSON_EXTRACT_DOUBLE(metadata_field, %s) > %s"
    assert params == ["field", 10]

    with pytest.raises(ValueError, match=r"\$gt must be a numeric value"):
        _parse_filter({"field": {"$gt": "string"}}, "metadata_field")


def test_gte_filter() -> None:
    filter_dict = {"field": {"$gte": 10}}
    query, params = _parse_filter(filter_dict, "metadata_field")
    assert query == "JSON_EXTRACT_DOUBLE(metadata_field, %s) >= %s"
    assert params == ["field", 10]

    with pytest.raises(ValueError, match=r"\$gte must be a numeric value"):
        _parse_filter({"field": {"$gte": "string"}}, "metadata_field")


def test_lt_filter() -> None:
    filter_dict = {"field": {"$lt": 10}}
    query, params = _parse_filter(filter_dict, "metadata_field")
    assert query == "JSON_EXTRACT_DOUBLE(metadata_field, %s) < %s"
    assert params == ["field", 10]

    with pytest.raises(ValueError, match=r"\$lt must be a numeric value"):
        _parse_filter({"field": {"$lt": "string"}}, "metadata_field")


def test_lte_filter() -> None:
    filter_dict = {"field": {"$lte": 10}}
    query, params = _parse_filter(filter_dict, "metadata_field")
    assert query == "JSON_EXTRACT_DOUBLE(metadata_field, %s) <= %s"
    assert params == ["field", 10]

    with pytest.raises(ValueError, match=r"\$lte must be a numeric value"):
        _parse_filter({"field": {"$lte": "string"}}, "metadata_field")


# Tests for array filters
def test_in_filter() -> None:
    filter_dict = {"field": {"$in": ["value1", "value2"]}}
    query, params = _parse_filter(filter_dict, "metadata_field")
    assert query == (
        "JSON_MATCH_ANY(JSON_ARRAY_CONTAINS_JSON(%s, MATCH_PARAM_JSON()), "
        "metadata_field, %s)"
    )
    assert params == [json.dumps(["value1", "value2"]), "field"]

    with pytest.raises(ValueError, match=r"\$in must be a list"):
        _parse_filter({"field": {"$in": "not_a_list"}}, "metadata_field")


def test_nin_filter() -> None:
    filter_dict = {"field": {"$nin": ["value1", "value2"]}}
    query, params = _parse_filter(filter_dict, "metadata_field")
    assert query == (
        "NOT JSON_MATCH_ANY(JSON_ARRAY_CONTAINS_JSON(%s, "
        "MATCH_PARAM_JSON()), metadata_field, %s)"
        " AND JSON_MATCH_ANY_EXISTS(metadata_field, %s)"
    )
    assert params == [json.dumps(["value1", "value2"]), "field", "field"]

    with pytest.raises(ValueError, match=r"\$nin must be a list"):
        _parse_filter({"field": {"$nin": "not_a_list"}}, "metadata_field")


# Tests for exists filter
def test_exists_filter() -> None:
    filter_dict = {"field": {"$exists": True}}
    query, params = _parse_filter(filter_dict, "metadata_field")
    assert query == "JSON_MATCH_ANY_EXISTS(metadata_field, %s)"
    assert params == ["field"]

    filter_dict = {"field": {"$exists": False}}
    query, params = _parse_filter(filter_dict, "metadata_field")
    assert query == "NOT JSON_MATCH_ANY_EXISTS(metadata_field, %s)"
    assert params == ["field"]

    with pytest.raises(ValueError, match=r"\$exists must be a boolean"):
        _parse_filter({"field": {"$exists": "not_a_bool"}}, "metadata_field")


# Tests for logical operators
def test_and_filter() -> None:
    filter_dict = {"$and": [{"field1": "value1"}, {"field2": "value2"}]}
    query, params = _parse_filter(filter_dict, "metadata_field")
    expected_query = (
        "JSON_MATCH_ANY(MATCH_PARAM_STRING_STRICT() = %s, metadata_field, %s)"
        + " AND JSON_MATCH_ANY(MATCH_PARAM_STRING_STRICT() = %s, metadata_field, %s)"
    )
    assert query == expected_query
    assert params == ["value1", "field1", "value2", "field2"]

    with pytest.raises(ValueError, match=r"\$and must be a list of filters"):
        _parse_filter({"$and": "not_a_list"}, "metadata_field")


def test_or_filter() -> None:
    filter_dict = {"$or": [{"field1": "value1"}, {"field2": "value2"}]}
    query, params = _parse_filter(filter_dict, "metadata_field")
    expected_query = (
        "(JSON_MATCH_ANY(MATCH_PARAM_STRING_STRICT() = %s, metadata_field, %s)"
        + " OR JSON_MATCH_ANY(MATCH_PARAM_STRING_STRICT() = %s, metadata_field, %s))"
    )
    assert query == expected_query
    assert params == ["value1", "field1", "value2", "field2"]

    with pytest.raises(ValueError, match=r"\$or must be a list of filters"):
        _parse_filter({"$or": "not_a_list"}, "metadata_field")


# Tests for nested filters
def test_nested_filters() -> None:
    filter_dict = {
        "$and": [
            {"field1": "value1"},
            {"$or": [{"field2": {"$gt": 10}}, {"field3": {"$exists": True}}]},
        ]
    }
    query, params = _parse_filter(filter_dict, "metadata_field")
    expected_query = (
        "JSON_MATCH_ANY(MATCH_PARAM_STRING_STRICT() = %s, metadata_field, %s)"
        + " AND (JSON_EXTRACT_DOUBLE(metadata_field, %s) > %s"
        + " OR JSON_MATCH_ANY_EXISTS(metadata_field, %s))"
    )
    assert query == expected_query
    assert params == ["value1", "field1", "field2", 10, "field3"]


# Tests for error cases
def test_filter_error_cases() -> None:
    # Test non-dictionary filter
    with pytest.raises(ValueError, match="Filter must be a dictionary"):
        _parse_filter([1, 2, 3], "metadata_field")

    # Test empty filter
    with pytest.raises(ValueError, match="Filter must contain exactly one key"):
        _parse_filter({}, "metadata_field")

    # Test filter with multiple keys
    with pytest.raises(ValueError, match="Filter must contain exactly one key"):
        _parse_filter({"field1": "value1", "field2": "value2"}, "metadata_field")

    # Test field filter with multiple operators
    with pytest.raises(ValueError, match="Field filter must contain exactly one key"):
        _parse_filter({"field": {"$eq": "value1", "$ne": "value2"}}, "metadata_field")

    # Test unsupported operator
    with pytest.raises(ValueError, match="Unsupported operator"):
        _parse_filter({"field": {"$unsupported": "value"}}, "metadata_field")
