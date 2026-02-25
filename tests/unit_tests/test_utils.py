"""Unit tests for langchain_singlestore._utils module."""

import unittest

from langchain_singlestore._utils import (
    CONNECTOR_NAME,
    CONNECTOR_VERSION,
    DistanceStrategy,
    hash,
    set_connector_attributes,
)


class TestConnectorConstants(unittest.TestCase):
    """Test connector constants."""

    def test_connector_name(self) -> None:
        """Test that CONNECTOR_NAME is set correctly."""
        assert CONNECTOR_NAME == "langchain python sdk"
        assert isinstance(CONNECTOR_NAME, str)

    def test_connector_version_format(self) -> None:
        """Test that CONNECTOR_VERSION is in valid format."""
        assert isinstance(CONNECTOR_VERSION, str)
        parts = CONNECTOR_VERSION.split(".")
        assert len(parts) == 3, "Version should be in X.Y.Z format"
        # All parts should be numeric
        for part in parts:
            assert part.isdigit(), f"Version part '{part}' should be numeric"


class TestSetConnectorAttributes(unittest.TestCase):
    """Test set_connector_attributes function."""

    def test_sets_attributes_in_empty_kwargs(self) -> None:
        """Test setting attributes in empty connection kwargs."""
        connection_kwargs: dict[str, dict] = {}
        set_connector_attributes(connection_kwargs)

        assert "conn_attrs" in connection_kwargs
        assert connection_kwargs["conn_attrs"]
        ["_connector_name"] == CONNECTOR_NAME
        assert connection_kwargs["conn_attrs"]
        ["_connector_version"] == CONNECTOR_VERSION

    def test_sets_attributes_preserves_existing_attrs(self) -> None:
        """Test that existing conn_attrs are preserved."""
        connection_kwargs: dict[str, dict] = {"conn_attrs": {"custom_attr": "value"}}
        set_connector_attributes(connection_kwargs)

        assert connection_kwargs["conn_attrs"]["custom_attr"] == "value"
        assert connection_kwargs["conn_attrs"]
        ["_connector_name"] == CONNECTOR_NAME
        assert connection_kwargs["conn_attrs"]
        ["_connector_version"] == CONNECTOR_VERSION

    def test_sets_attributes_creates_dict_if_missing(self) -> None:
        """Test that conn_attrs dict is created if missing."""
        connection_kwargs: dict[str, dict] = {}
        set_connector_attributes(connection_kwargs)

        assert isinstance(connection_kwargs.get("conn_attrs"), dict)

    def test_modifies_in_place(self) -> None:
        """Test that function modifies dictionary in place."""
        connection_kwargs: dict[str, dict] = {}
        original_id = id(connection_kwargs)
        set_connector_attributes(connection_kwargs)

        assert id(connection_kwargs) == original_id, "Should modify dict in place"


class TestDistanceStrategy(unittest.TestCase):
    """Test DistanceStrategy enum."""

    def test_distance_strategy_values(self) -> None:
        """Test that distance strategies are defined correctly."""
        assert DistanceStrategy.EUCLIDEAN_DISTANCE.value == "EUCLIDEAN_DISTANCE"
        assert DistanceStrategy.DOT_PRODUCT.value == "DOT_PRODUCT"

    def test_distance_strategy_is_string_enum(self) -> None:
        """Test that DistanceStrategy is a string enum."""
        assert isinstance(DistanceStrategy.DOT_PRODUCT, str)
        assert isinstance(DistanceStrategy.EUCLIDEAN_DISTANCE, str)

    def test_distance_strategy_comparison(self) -> None:
        """Test distance strategy comparison."""
        assert DistanceStrategy.DOT_PRODUCT == "DOT_PRODUCT"
        assert DistanceStrategy.EUCLIDEAN_DISTANCE == "EUCLIDEAN_DISTANCE"


class TestHashFunction(unittest.TestCase):
    """Test hash utility function."""

    def test_hash_returns_hex_string(self) -> None:
        """Test that hash returns a valid hex string."""
        result = hash("test input")
        assert isinstance(result, str)
        # MD5 produces 32-character hex strings
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    def test_hash_is_deterministic(self) -> None:
        """Test that hash produces same output for same input."""
        input_str = "test string"
        hash1 = hash(input_str)
        hash2 = hash(input_str)
        assert hash1 == hash2

    def test_hash_differs_for_different_inputs(self) -> None:
        """Test that hash differs for different inputs."""
        hash1 = hash("input1")
        hash2 = hash("input2")
        assert hash1 != hash2

    def test_hash_handles_empty_string(self) -> None:
        """Test that hash handles empty string."""
        result = hash("")
        assert isinstance(result, str)
        assert len(result) == 32

    def test_hash_handles_special_characters(self) -> None:
        """Test that hash handles special characters."""
        result = hash("test!@#$%^&*()_-+=[]{}|;:,.<>?")
        assert isinstance(result, str)
        assert len(result) == 32


if __name__ == "__main__":
    unittest.main()
