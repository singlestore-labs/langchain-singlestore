"""Backwards-compatibility shim.

The implementation now lives in ``singlestore_langchain_core._filter``.
"""

from singlestore_langchain_core._filter import (
    AndFilter,
    EqFilter,
    ExactMatchFilter,
    ExistsFilter,
    FieldFilter,
    FieldValue,
    FilterTypedDict,
    GteFilter,
    GtFilter,
    InFilter,
    LteFilter,
    LtFilter,
    NeFilter,
    NinFilter,
    NumericFieldValue,
    OrFilter,
    SimpleFilter,
    _get_match_param_function,
    _handle_and_filter,
    _handle_operator_filter,
    _handle_or_filter,
    _parse_filter,
)

__all__ = [
    "AndFilter",
    "EqFilter",
    "ExactMatchFilter",
    "ExistsFilter",
    "FieldFilter",
    "FieldValue",
    "FilterTypedDict",
    "GtFilter",
    "GteFilter",
    "InFilter",
    "LtFilter",
    "LteFilter",
    "NeFilter",
    "NinFilter",
    "NumericFieldValue",
    "OrFilter",
    "SimpleFilter",
    "_get_match_param_function",
    "_handle_and_filter",
    "_handle_operator_filter",
    "_handle_or_filter",
    "_parse_filter",
]
