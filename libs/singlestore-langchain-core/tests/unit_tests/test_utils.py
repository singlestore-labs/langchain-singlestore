"""Unit tests for singlestore_langchain_core._utils."""

import unittest

from singlestore_langchain_core._utils import (
    DEFAULT_CONNECTOR_NAME,
    DistanceStrategy,
    FullTextIndexVersion,
    FullTextScoringMode,
    compute_connector_version,
    hash,
    set_connector_attributes,
)


class TestConnectorConstants(unittest.TestCase):
    def test_default_connector_name(self) -> None:
        assert DEFAULT_CONNECTOR_NAME == "langchain python sdk"

    def test_compute_connector_version_fallback(self) -> None:
        v = compute_connector_version("this-package-does-not-exist")
        assert v == "3.0.0"

    def test_compute_connector_version_custom_fallback(self) -> None:
        v = compute_connector_version("nope", fallback="9.9.9")
        assert v == "9.9.9"


class TestSetConnectorAttributes(unittest.TestCase):
    def test_defaults_stamp_name_only(self) -> None:
        kwargs: dict = {}
        set_connector_attributes(kwargs)
        assert kwargs["conn_attrs"]["_connector_name"] == DEFAULT_CONNECTOR_NAME
        assert "_connector_version" not in kwargs["conn_attrs"]

    def test_stamps_custom_name_and_version(self) -> None:
        kwargs: dict = {}
        set_connector_attributes(
            kwargs, connector_name="unit test sdk", connector_version="1.2.3"
        )
        assert kwargs["conn_attrs"]["_connector_name"] == "unit test sdk"
        assert kwargs["conn_attrs"]["_connector_version"] == "1.2.3"

    def test_preserves_existing_conn_attrs(self) -> None:
        kwargs: dict = {"conn_attrs": {"foo": "bar"}}
        set_connector_attributes(kwargs, connector_version="1.0.0")
        assert kwargs["conn_attrs"]["foo"] == "bar"
        assert kwargs["conn_attrs"]["_connector_version"] == "1.0.0"


class TestEnums(unittest.TestCase):
    def test_distance_strategy_values(self) -> None:
        assert DistanceStrategy.DOT_PRODUCT.value == "DOT_PRODUCT"
        assert DistanceStrategy.EUCLIDEAN_DISTANCE.value == "EUCLIDEAN_DISTANCE"

    def test_full_text_index_version_values(self) -> None:
        assert FullTextIndexVersion.V1.value == "V1"
        assert FullTextIndexVersion.V2.value == "V2"

    def test_full_text_scoring_mode_values(self) -> None:
        assert FullTextScoringMode.MATCH.value == "MATCH"
        assert FullTextScoringMode.BM25.value == "BM25"
        assert FullTextScoringMode.BM25_GLOBAL.value == "BM25_GLOBAL"


class TestHash(unittest.TestCase):
    def test_hash_is_deterministic(self) -> None:
        assert hash("foo") == hash("foo")
        assert hash("foo") != hash("bar")
