"""Unit tests for langchain_singlestore.document_loaders module."""

import unittest
from unittest.mock import MagicMock, patch

from langchain_singlestore.document_loaders import SingleStoreLoader


class TestSingleStoreLoaderInit(unittest.TestCase):
    """Test SingleStoreLoader initialization."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.patcher = patch("langchain_singlestore.document_loaders.QueuePool")
        self.mock_pool_class = self.patcher.start()
        self.mock_pool = MagicMock()
        self.mock_pool_class.return_value = self.mock_pool

    def tearDown(self) -> None:
        """Clean up patches."""
        self.patcher.stop()

    def test_init_with_required_params(self) -> None:
        """Test initialization with required parameters."""
        loader = SingleStoreLoader(host="localhost")

        assert loader.table_name == "embeddings"
        assert loader.content_field == "content"
        assert loader.metadata_field == "metadata"
        assert loader.id_field == "id"

    def test_init_custom_table_name(self) -> None:
        """Test that custom table name is used."""
        loader = SingleStoreLoader(host="localhost", table_name="my_documents")

        assert loader.table_name == "my_documents"

    def test_init_custom_field_names(self) -> None:
        """Test that custom field names are used."""
        loader = SingleStoreLoader(
            host="localhost",
            content_field="text",
            metadata_field="meta",
            id_field="doc_id",
        )

        assert loader.content_field == "text"
        assert loader.metadata_field == "meta"
        assert loader.id_field == "doc_id"

    def test_init_sets_connector_attributes(self) -> None:
        """Test that connector attributes are set."""
        loader = SingleStoreLoader(host="localhost")

        assert "conn_attrs" in loader.connection_kwargs
        assert "_connector_name" in loader.connection_kwargs["conn_attrs"]
        assert "_connector_version" in loader.connection_kwargs["conn_attrs"]

    def test_init_pool_settings(self) -> None:
        """Test that connection pool settings are configured."""
        SingleStoreLoader(host="localhost", pool_size=10, max_overflow=20, timeout=60)

        # Verify QueuePool was called with correct parameters
        self.mock_pool_class.assert_called_once()
        call_kwargs = self.mock_pool_class.call_args[1]
        assert call_kwargs["pool_size"] == 10
        assert call_kwargs["max_overflow"] == 20
        assert call_kwargs["timeout"] == 60

    def test_init_connection_kwargs(self) -> None:
        """Test that connection kwargs are set correctly."""
        loader = SingleStoreLoader(
            host="localhost", port=3306, user="testuser", password="testpass"
        )

        assert loader.connection_kwargs["host"] == "localhost"
        assert loader.connection_kwargs["port"] == 3306
        assert loader.connection_kwargs["user"] == "testuser"
        assert loader.connection_kwargs["password"] == "testpass"

    def test_sanitize_input_removes_special_chars(self) -> None:
        """Test that _sanitize_input removes special characters."""
        loader = SingleStoreLoader(host="localhost")

        result = loader._sanitize_input("test!@#$%^&*()input")
        assert result == "testinput"

    def test_sanitize_input_keeps_alphanumeric_and_underscore(self) -> None:
        """Test that _sanitize_input keeps alphanumeric and underscore."""
        loader = SingleStoreLoader(host="localhost")

        result = loader._sanitize_input("test_123_input")
        assert result == "test_123_input"

    def test_connection_pool_created(self) -> None:
        """Test that connection pool is created."""
        loader = SingleStoreLoader(host="localhost")

        assert loader.connection_pool is not None


class TestSingleStoreLoaderLoad(unittest.TestCase):
    """Test document loading functionality."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.patcher = patch("langchain_singlestore.document_loaders.QueuePool")
        self.mock_pool_class = self.patcher.start()
        self.mock_pool = MagicMock()
        self.mock_pool_class.return_value = self.mock_pool
        self.mock_conn = MagicMock()
        self.mock_pool.connect.return_value = self.mock_conn

    def tearDown(self) -> None:
        """Clean up patches."""
        self.patcher.stop()

    def test_lazy_load_returns_iterator(self) -> None:
        """Test that lazy_load returns an iterator."""
        loader = SingleStoreLoader(host="localhost")

        result = loader.lazy_load()

        # Should be an iterator
        assert hasattr(result, "__iter__")

    def test_load_returns_list(self) -> None:
        """Test that load returns a list of documents."""
        loader = SingleStoreLoader(host="localhost")

        mock_cursor = MagicMock()
        self.mock_conn.cursor.return_value = mock_cursor

        # Mock database response with one document
        test_content = "test document content"
        test_metadata = {"source": "test"}  # Metadata should be a dict, not string
        test_id = "1"
        mock_cursor.fetchall.return_value = [(test_content, test_metadata, test_id)]

        result = loader.load()

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].page_content == test_content


if __name__ == "__main__":
    unittest.main()
