"""Unit tests for langchain_singlestore.document_loaders module."""

import unittest
from unittest.mock import MagicMock, patch

from sqlalchemy.pool import Pool

from langchain_singlestore.document_loaders import SingleStoreLoader


def _make_loader(**kwargs: object) -> SingleStoreLoader:
    """Build a loader backed by a mock pool so tests never touch the DB."""
    params: dict = {"connection_pool": MagicMock(spec=Pool)}
    params.update(kwargs)
    return SingleStoreLoader(**params)  # type: ignore[arg-type]


class TestSingleStoreLoaderInit(unittest.TestCase):
    def test_init_with_required_params(self) -> None:
        loader = _make_loader()
        assert loader.table_name == "embeddings"
        assert loader.content_field == "content"
        assert loader.metadata_field == "metadata"
        assert loader.id_field == "id"

    def test_init_custom_table_name(self) -> None:
        loader = _make_loader(table_name="my_documents")
        assert loader.table_name == "my_documents"

    def test_init_custom_field_names(self) -> None:
        loader = _make_loader(
            content_field="text",
            metadata_field="meta",
            id_field="doc_id",
        )
        assert loader.content_field == "text"
        assert loader.metadata_field == "meta"
        assert loader.id_field == "doc_id"

    def test_init_sets_connector_attributes(self) -> None:
        loader = _make_loader(host="localhost")
        assert "conn_attrs" in loader.connection_kwargs
        assert "_connector_name" in loader.connection_kwargs["conn_attrs"]
        assert "_connector_version" in loader.connection_kwargs["conn_attrs"]

    def test_init_pool_settings(self) -> None:
        """Pool sizing kwargs are forwarded to create_connection_pool."""
        with patch(
            "langchain_singlestore.document_loaders.create_connection_pool"
        ) as mock_factory:
            mock_factory.return_value = MagicMock(spec=Pool)
            SingleStoreLoader(
                host="localhost", pool_size=10, max_overflow=20, timeout=60
            )
        mock_factory.assert_called_once()
        call_kwargs = mock_factory.call_args.kwargs
        assert call_kwargs["pool_size"] == 10
        assert call_kwargs["max_overflow"] == 20
        assert call_kwargs["timeout"] == 60

    def test_init_forwards_connection(self) -> None:
        conn = MagicMock(name="raw_connection")
        with patch(
            "langchain_singlestore.document_loaders.create_connection_pool"
        ) as mock_factory:
            mock_factory.return_value = MagicMock(spec=Pool)
            SingleStoreLoader(connection=conn)
        assert mock_factory.call_args.kwargs["connection"] is conn

    def test_init_connection_kwargs(self) -> None:
        loader = _make_loader(
            host="localhost", port=3306, user="testuser", password="testpass"
        )
        assert loader.connection_kwargs["host"] == "localhost"
        assert loader.connection_kwargs["port"] == 3306
        assert loader.connection_kwargs["user"] == "testuser"
        assert loader.connection_kwargs["password"] == "testpass"

    def test_sanitize_input_removes_special_chars(self) -> None:
        loader = _make_loader()
        assert loader._sanitize_input("test!@#$%^&*()input") == "testinput"

    def test_sanitize_input_keeps_alphanumeric_and_underscore(self) -> None:
        loader = _make_loader()
        assert loader._sanitize_input("test_123_input") == "test_123_input"

    def test_connection_pool_is_the_injected_mock(self) -> None:
        pool = MagicMock(spec=Pool)
        loader = SingleStoreLoader(connection_pool=pool)
        assert loader.connection_pool is pool


class TestSingleStoreLoaderLoad(unittest.TestCase):
    def _loader(self) -> tuple[SingleStoreLoader, MagicMock]:
        pool = MagicMock(spec=Pool)
        conn = MagicMock()
        pool.connect.return_value = conn
        loader = SingleStoreLoader(connection_pool=pool)
        return loader, conn

    def test_lazy_load_returns_iterator(self) -> None:
        loader, _ = self._loader()
        assert hasattr(loader.lazy_load(), "__iter__")

    def test_load_returns_list(self) -> None:
        loader, conn = self._loader()
        mock_cursor = MagicMock()
        conn.cursor.return_value = mock_cursor

        test_content = "test document content"
        test_metadata = {"source": "test"}
        mock_cursor.fetchall.return_value = [(test_content, test_metadata, "1")]

        result = loader.load()
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].page_content == test_content


if __name__ == "__main__":
    unittest.main()
