"""Unit tests for langchain_singlestore.cache module."""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.embeddings import Embeddings

from langchain_singlestore._utils import DistanceStrategy
from langchain_singlestore.cache import SingleStoreSemanticCache


class MockEmbeddings(Embeddings):
    """Mock embeddings for testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return mock embeddings."""
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return mock embedding."""
        return [0.1, 0.2, 0.3]


class TestSingleStoreSemanticCacheInit(unittest.TestCase):
    """Test SingleStoreSemanticCache initialization."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.patcher = patch("langchain_singlestore.cache.SingleStoreVectorStore")
        self.mock_vs_class = self.patcher.start()
        self.mock_vs = MagicMock()
        self.mock_vs_class.return_value = self.mock_vs

    def tearDown(self) -> None:
        """Clean up patches."""
        self.patcher.stop()

    def test_init_with_required_params(self) -> None:
        """Test initialization with required parameters."""
        embeddings = MockEmbeddings()
        cache = SingleStoreSemanticCache(embedding=embeddings, host="localhost")

        assert cache.embedding is embeddings
        assert cache.cache_table_prefix == "cache_"
        assert cache.search_threshold == 0.2

    def test_init_custom_cache_table_prefix(self) -> None:
        """Test that custom cache_table_prefix is used."""
        embeddings = MockEmbeddings()
        cache = SingleStoreSemanticCache(
            embedding=embeddings,
            host="localhost",
            cache_table_prefix="my_cache_",
        )

        assert cache.cache_table_prefix == "my_cache_"

    def test_init_custom_search_threshold(self) -> None:
        """Test that custom search_threshold is used."""
        embeddings = MockEmbeddings()
        cache = SingleStoreSemanticCache(
            embedding=embeddings, host="localhost", search_threshold=0.5
        )

        assert cache.search_threshold == 0.5

    def test_init_sets_distance_strategy(self) -> None:
        """Test that distance strategy can be stored for later use."""
        embeddings = MockEmbeddings()
        cache = SingleStoreSemanticCache(
            embedding=embeddings,
            host="localhost",
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        )

        # Store connection kwargs for later use when creating vectorstores
        assert "distance_strategy" in cache.connection_kwargs

        assert cache.connection_kwargs
        ["distance_strategy"] == DistanceStrategy.EUCLIDEAN_DISTANCE

    def test_init_initializes_cache_dict(self) -> None:
        """Test that cache dict is initialized for lazy vectorstore creation."""
        embeddings = MockEmbeddings()
        cache = SingleStoreSemanticCache(embedding=embeddings, host="localhost")

        # Vectorstores are created lazily, not on init
        assert hasattr(cache, "_cache_dict")
        assert isinstance(cache._cache_dict, dict)
        assert len(cache._cache_dict) == 0


class TestSingleStoreSemanticCacheVectorStoreParams(unittest.TestCase):
    """Test that vectorstore parameters are passed correctly."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.patcher = patch("langchain_singlestore.cache.SingleStoreVectorStore")
        self.mock_vs_class = self.patcher.start()
        self.mock_vs = MagicMock()
        self.mock_vs_class.return_value = self.mock_vs

    def tearDown(self) -> None:
        """Clean up patches."""
        self.patcher.stop()

    def test_passes_table_name_with_prefix(self) -> None:
        """Test that custom cache_table_prefix is stored."""
        embeddings = MockEmbeddings()
        cache = SingleStoreSemanticCache(
            embedding=embeddings,
            host="localhost",
            cache_table_prefix="my_prefix_",
        )

        # Prefix is stored and used when vectorstores are created lazily
        assert cache.cache_table_prefix == "my_prefix_"

    def test_passes_use_vector_index(self) -> None:
        """Test that use_vector_index is stored in connection_kwargs."""
        embeddings = MockEmbeddings()
        cache = SingleStoreSemanticCache(
            embedding=embeddings,
            host="localhost",
            use_vector_index=True,
        )

        # Parameter is stored in connection_kwargs for lazy vectorstore creation
        assert cache.connection_kwargs["use_vector_index"] is True

    def test_passes_vector_size(self) -> None:
        """Test that vector_size is stored in connection_kwargs."""
        embeddings = MockEmbeddings()
        cache = SingleStoreSemanticCache(
            embedding=embeddings,
            host="localhost",
            vector_size=768,
        )

        # Parameter is stored in connection_kwargs for lazy vectorstore creation
        assert cache.connection_kwargs["vector_size"] == 768

    def test_passes_host(self) -> None:
        """Test that host is stored in connection_kwargs."""
        embeddings = MockEmbeddings()
        test_host = "test.example.com"
        cache = SingleStoreSemanticCache(
            embedding=embeddings,
            host=test_host,
        )

        # Host is stored in connection_kwargs for lazy vectorstore creation
        assert cache.connection_kwargs["host"] == test_host

    def test_passes_custom_connection_params(self) -> None:
        """Test that custom connection parameters are stored."""
        embeddings = MockEmbeddings()
        cache = SingleStoreSemanticCache(
            embedding=embeddings,
            host="localhost",
            port=3306,
            user="testuser",
            password="testpass",
        )

        # Parameters are stored in connection_kwargs for lazy vectorstore creation
        assert cache.connection_kwargs["port"] == 3306
        assert cache.connection_kwargs["user"] == "testuser"
        assert cache.connection_kwargs["password"] == "testpass"


class TestSingleStoreSemanticCacheInheritance(unittest.TestCase):
    """Test that cache inherits from BaseCache correctly."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.patcher = patch("langchain_singlestore.cache.SingleStoreVectorStore")
        self.mock_vs_class = self.patcher.start()
        self.mock_vs = MagicMock()
        self.mock_vs_class.return_value = self.mock_vs

    def tearDown(self) -> None:
        """Clean up patches."""
        self.patcher.stop()

    def test_is_instance_of_base_cache(self) -> None:
        """Test that SingleStoreSemanticCache is a BaseCache."""
        from langchain_core.caches import BaseCache

        embeddings = MockEmbeddings()
        cache = SingleStoreSemanticCache(embedding=embeddings, host="localhost")

        assert isinstance(cache, BaseCache)


if __name__ == "__main__":
    unittest.main()
