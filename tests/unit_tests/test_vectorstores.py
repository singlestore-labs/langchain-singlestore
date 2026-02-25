"""Unit tests for langchain_singlestore.vectorstores module."""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.embeddings import Embeddings

from langchain_singlestore._utils import DistanceStrategy
from langchain_singlestore.vectorstores import SingleStoreVectorStore


class MockEmbeddings(Embeddings):
    """Mock embeddings for testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return mock embeddings."""
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return mock embedding."""
        return [0.1, 0.2, 0.3]


class TestSingleStoreVectorStoreInit(unittest.TestCase):
    """Test SingleStoreVectorStore initialization."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.patcher = patch("langchain_singlestore.vectorstores.QueuePool")
        self.mock_pool_class = self.patcher.start()
        self.mock_pool = MagicMock()
        self.mock_pool_class.return_value = self.mock_pool

    def tearDown(self) -> None:
        """Clean up patches."""
        self.patcher.stop()

    def test_init_with_required_params(self) -> None:
        """Test initialization with required parameters."""
        embeddings = MockEmbeddings()
        vs = SingleStoreVectorStore(embedding=embeddings, host="localhost")

        assert vs.embedding == embeddings
        assert vs.table_name == "embeddings"
        assert vs.distance_strategy == DistanceStrategy.DOT_PRODUCT

    def test_init_sets_default_table_name(self) -> None:
        """Test that default table name is set."""
        embeddings = MockEmbeddings()
        vs = SingleStoreVectorStore(embedding=embeddings, host="localhost")

        assert vs.table_name == "embeddings"

    def test_init_custom_table_name(self) -> None:
        """Test that custom table name is used."""
        embeddings = MockEmbeddings()
        vs = SingleStoreVectorStore(
            embedding=embeddings, host="localhost", table_name="custom_embeddings"
        )

        assert vs.table_name == "custom_embeddings"

    def test_init_sets_field_names(self) -> None:
        """Test that field names are set correctly."""
        embeddings = MockEmbeddings()
        vs = SingleStoreVectorStore(embedding=embeddings, host="localhost")

        assert vs.content_field == "content"
        assert vs.metadata_field == "metadata"
        assert vs.vector_field == "vector"
        assert vs.id_field == "id"

    def test_init_custom_field_names(self) -> None:
        """Test that custom field names are used."""
        embeddings = MockEmbeddings()
        vs = SingleStoreVectorStore(
            embedding=embeddings,
            host="localhost",
            content_field="text",
            metadata_field="meta",
            vector_field="vec",
            id_field="doc_id",
        )

        assert vs.content_field == "text"
        assert vs.metadata_field == "meta"
        assert vs.vector_field == "vec"
        assert vs.id_field == "doc_id"

    def test_init_distance_strategy(self) -> None:
        """Test that distance strategy is set."""
        embeddings = MockEmbeddings()
        vs = SingleStoreVectorStore(
            embedding=embeddings,
            host="localhost",
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        )

        assert vs.distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE

    def test_init_vector_index_disabled_by_default(self) -> None:
        """Test that vector index is disabled by default."""
        embeddings = MockEmbeddings()
        vs = SingleStoreVectorStore(embedding=embeddings, host="localhost")

        assert vs.use_vector_index is False

    def test_init_vector_index_enabled(self) -> None:
        """Test that vector index can be enabled."""
        embeddings = MockEmbeddings()
        vs = SingleStoreVectorStore(
            embedding=embeddings, host="localhost", use_vector_index=True
        )

        assert vs.use_vector_index is True

    def test_init_full_text_search_disabled_by_default(self) -> None:
        """Test that full-text search is disabled by default."""
        embeddings = MockEmbeddings()
        vs = SingleStoreVectorStore(embedding=embeddings, host="localhost")

        assert vs.use_full_text_search is False

    def test_init_full_text_search_enabled(self) -> None:
        """Test that full-text search can be enabled."""
        embeddings = MockEmbeddings()
        vs = SingleStoreVectorStore(
            embedding=embeddings, host="localhost", use_full_text_search=True
        )

        assert vs.use_full_text_search is True

    def test_init_sets_connector_attributes(self) -> None:
        """Test that connector attributes are set."""
        embeddings = MockEmbeddings()
        vs = SingleStoreVectorStore(embedding=embeddings, host="localhost")

        assert "conn_attrs" in vs.connection_kwargs
        assert "_connector_name" in vs.connection_kwargs["conn_attrs"]
        assert "_connector_version" in vs.connection_kwargs["conn_attrs"]

    def test_init_vector_size_default(self) -> None:
        """Test that vector_size defaults to 1536."""
        embeddings = MockEmbeddings()
        vs = SingleStoreVectorStore(embedding=embeddings, host="localhost")

        assert vs.vector_size == 1536

    def test_init_custom_vector_size(self) -> None:
        """Test that custom vector_size is used."""
        embeddings = MockEmbeddings()
        vs = SingleStoreVectorStore(
            embedding=embeddings, host="localhost", vector_size=768
        )

        assert vs.vector_size == 768

    def test_init_pool_settings(self) -> None:
        """Test that connection pool settings are configured."""
        embeddings = MockEmbeddings()
        SingleStoreVectorStore(
            embedding=embeddings,
            host="localhost",
            pool_size=10,
            max_overflow=20,
            timeout=60,
        )

        # Verify QueuePool was called with correct parameters
        self.mock_pool_class.assert_called_once()
        call_kwargs = self.mock_pool_class.call_args[1]
        assert call_kwargs["pool_size"] == 10
        assert call_kwargs["max_overflow"] == 20
        assert call_kwargs["timeout"] == 60


class TestSingleStoreVectorStoreSanitize(unittest.TestCase):
    """Test _sanitize_input method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.patcher = patch("langchain_singlestore.vectorstores.QueuePool")
        self.mock_pool_class = self.patcher.start()
        self.mock_pool = MagicMock()
        self.mock_pool_class.return_value = self.mock_pool

    def tearDown(self) -> None:
        """Clean up patches."""
        self.patcher.stop()

    def test_sanitize_removes_special_chars(self) -> None:
        """Test that sanitize removes special characters."""
        embeddings = MockEmbeddings()
        vs = SingleStoreVectorStore(embedding=embeddings, host="localhost")

        result = vs._sanitize_input("test!@#$%^&*()input")
        assert result == "testinput"

    def test_sanitize_keeps_alphanumeric_and_underscore(self) -> None:
        """Test that sanitize keeps alphanumeric and underscore."""
        embeddings = MockEmbeddings()
        vs = SingleStoreVectorStore(embedding=embeddings, host="localhost")

        result = vs._sanitize_input("test_123_input")
        assert result == "test_123_input"


class TestSingleStoreVectorStoreSearchStrategy(unittest.TestCase):
    """Test SearchStrategy enum."""

    def test_search_strategies_defined(self) -> None:
        """Test that all search strategies are defined."""
        assert hasattr(SingleStoreVectorStore.SearchStrategy, "VECTOR_ONLY")
        assert hasattr(SingleStoreVectorStore.SearchStrategy, "TEXT_ONLY")
        assert hasattr(SingleStoreVectorStore.SearchStrategy, "FILTER_BY_TEXT")
        assert hasattr(SingleStoreVectorStore.SearchStrategy, "FILTER_BY_VECTOR")
        assert hasattr(SingleStoreVectorStore.SearchStrategy, "WEIGHTED_SUM")

    def test_search_strategy_values(self) -> None:
        """Test search strategy values."""
        assert SingleStoreVectorStore.SearchStrategy.VECTOR_ONLY == "VECTOR_ONLY"
        assert SingleStoreVectorStore.SearchStrategy.TEXT_ONLY == "TEXT_ONLY"
        assert SingleStoreVectorStore.SearchStrategy.FILTER_BY_TEXT == "FILTER_BY_TEXT"


class TestSingleStoreVectorStoreEmbeddings(unittest.TestCase):
    """Test embeddings property."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.patcher = patch("langchain_singlestore.vectorstores.QueuePool")
        self.mock_pool_class = self.patcher.start()
        self.mock_pool = MagicMock()
        self.mock_pool_class.return_value = self.mock_pool

    def tearDown(self) -> None:
        """Clean up patches."""
        self.patcher.stop()

    def test_embeddings_property(self) -> None:
        """Test that embeddings property returns the embedding model."""
        embeddings = MockEmbeddings()
        vs = SingleStoreVectorStore(embedding=embeddings, host="localhost")

        assert vs.embeddings is embeddings


if __name__ == "__main__":
    unittest.main()
