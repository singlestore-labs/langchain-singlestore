"""
Integration tests for SingleStoreVectorStore with pre-computed embeddings.

These tests verify that the vectorstore works correctly when users provide
pre-computed embeddings, bypassing the embedding model entirely. The ErrorEmbeddings
class is used to ensure the embedding model is never called.
"""

import math
import os
import tempfile
from typing import Any, Generator, List

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_singlestore._utils import (
    DistanceStrategy,
    FullTextIndexVersion,
    FullTextScoringMode,
)
from langchain_singlestore.vectorstores import SingleStoreVectorStore
from tests.integration_tests.conftest import TEST_DB_NAME


class StoreTracker:
    """Tracks SingleStoreVectorStore instances for automatic cleanup.

    Usage:
        def test_something(self, store_tracker):
            store = store_tracker.create(embedding=..., host=..., database=...)
            # ... test code ...
            # store.drop() is called automatically after the test
    """

    def __init__(self) -> None:
        self._stores: List[SingleStoreVectorStore] = []

    def track(self, store: SingleStoreVectorStore) -> SingleStoreVectorStore:
        """Register a store for automatic cleanup."""
        self._stores.append(store)
        return store

    def create(self, **kwargs: Any) -> SingleStoreVectorStore:
        """Create and track a new SingleStoreVectorStore."""
        store = SingleStoreVectorStore(**kwargs)
        return self.track(store)

    def create_from_texts(self, **kwargs: Any) -> SingleStoreVectorStore:
        """Create from texts and track a new SingleStoreVectorStore."""
        store = SingleStoreVectorStore.from_texts(**kwargs)
        return self.track(store)

    def create_from_documents(self, **kwargs: Any) -> SingleStoreVectorStore:
        """Create from documents and track a new SingleStoreVectorStore."""
        store = SingleStoreVectorStore.from_documents(**kwargs)
        return self.track(store)

    def cleanup(self) -> None:
        """Drop all tracked stores."""
        for store in self._stores:
            try:
                store.drop()
            except Exception:
                pass  # Ignore errors during cleanup
        self._stores.clear()


@pytest.fixture
def store_tracker() -> Generator[StoreTracker, None, None]:
    """Fixture that provides automatic cleanup for vectorstores.

    Example:
        def test_my_test(self, store_tracker, clean_db_url):
            store = store_tracker.create(
                embedding=ErrorEmbeddings(),
                host=clean_db_url,
                database=TEST_DB_NAME,
            )
            store.add_texts(...)  # use the store
            # No need for try/finally - cleanup is automatic
    """
    tracker = StoreTracker()
    try:
        yield tracker
    finally:
        tracker.cleanup()


class ErrorEmbeddings(Embeddings):
    """Fake embeddings that raises an error. For testing purposes.

    This class is used to verify that pre-computed embeddings are being used
    instead of calling the embedding model. If the vectorstore accidentally
    tries to compute embeddings, the test will fail with a clear error.
    """

    def embed_query(self, text: str) -> List[float]:
        raise ValueError(
            "Embedding model was called when it should not have been. "
            "Pre-computed embeddings should be used instead."
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise ValueError(
            "Embedding model was called when it should not have been. "
            "Pre-computed embeddings should be used instead."
        )

    def embed_image(self, uris: List[str]) -> List[List[float]]:
        raise ValueError(
            "Embedding model was called when it should not have been. "
            "Pre-computed embeddings should be used instead."
        )


def generate_embedding(index: int, size: int = 10) -> List[float]:
    """Generate a deterministic embedding for testing.

    Uses trigonometric functions to create embeddings that have
    predictable similarity relationships.

    Args:
        index: The document index (affects the angle used)
        size: The embedding dimension

    Returns:
        A list of floats representing the embedding
    """
    return [math.cos((index + i) * math.pi / size) for i in range(size)]


def generate_embeddings(count: int, size: int = 10) -> List[List[float]]:
    """Generate multiple deterministic embeddings for testing.

    Args:
        count: Number of embeddings to generate
        size: The embedding dimension

    Returns:
        A list of embeddings
    """
    return [generate_embedding(i, size) for i in range(count)]


class TestPrecomputedEmbeddingsVectorStoreCreation:
    """Test vectorstore creation methods with pre-computed embeddings."""

    @pytest.fixture
    def sample_texts(self) -> List[str]:
        """Sample texts for testing."""
        return [
            "The quick brown fox jumps over the lazy dog.",
            "A journey of a thousand miles begins with a single step.",
            "To be or not to be, that is the question.",
        ]

    @pytest.fixture
    def sample_metadatas(self) -> List[dict]:
        """Sample metadata for testing."""
        return [
            {"source": "proverb", "category": "animals"},
            {"source": "proverb", "category": "wisdom"},
            {"source": "shakespeare", "category": "philosophy"},
        ]

    @pytest.fixture
    def sample_documents(
        self, sample_texts: List[str], sample_metadatas: List[dict]
    ) -> List[Document]:
        """Sample documents for testing."""
        return [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(sample_texts, sample_metadatas)
        ]

    @pytest.fixture
    def precomputed_embeddings(self, sample_texts: List[str]) -> List[List[float]]:
        """Pre-computed embeddings for sample texts."""
        return generate_embeddings(len(sample_texts))

    # ============================================================
    # Constructor creation tests
    # ============================================================

    def test_constructor_add_texts_with_precomputed_embeddings(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        sample_texts: List[str],
        sample_metadatas: List[dict],
        precomputed_embeddings: List[List[float]],
    ) -> None:
        """Test creating vectorstore with constructor and add_texts with embeddings."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        ids = store.add_texts(
            texts=sample_texts,
            metadatas=sample_metadatas,
            embeddings=precomputed_embeddings,
        )
        assert len(ids) == len(sample_texts)

        # Verify search works with pre-computed query embedding
        results = store.similarity_search(
            query="fox",
            k=1,
            query_embedding=precomputed_embeddings[0],
        )
        assert len(results) == 1
        assert results[0].page_content == sample_texts[0]

    def test_constructor_add_documents_with_precomputed_embeddings(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        sample_documents: List[Document],
        precomputed_embeddings: List[List[float]],
    ) -> None:
        """Test vectorstore with constructor and add_documents with embeddings."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        ids = store.add_documents(
            documents=sample_documents,
            embeddings=precomputed_embeddings,
        )
        assert len(ids) == len(sample_documents)

        results = store.similarity_search(
            query="any",
            k=3,
            query_embedding=precomputed_embeddings[0],
        )
        assert len(results) == 3

    @pytest.mark.parametrize(
        "distance_strategy",
        [DistanceStrategy.DOT_PRODUCT, DistanceStrategy.EUCLIDEAN_DISTANCE],
    )
    def test_constructor_with_different_distance_strategies(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        sample_texts: List[str],
        precomputed_embeddings: List[List[float]],
        distance_strategy: DistanceStrategy,
    ) -> None:
        """Test vectorstore creation with different distance strategies."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            distance_strategy=distance_strategy,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        store.add_texts(
            texts=sample_texts,
            embeddings=precomputed_embeddings,
        )
        results = store.similarity_search(
            query="any",
            k=1,
            query_embedding=precomputed_embeddings[2],
        )
        assert len(results) == 1
        assert results[0].page_content == sample_texts[2]

    @pytest.mark.parametrize(
        "distance_strategy,vector_size,index_options",
        [
            (DistanceStrategy.DOT_PRODUCT, 10, None),
            (DistanceStrategy.EUCLIDEAN_DISTANCE, 10, None),
            (
                DistanceStrategy.EUCLIDEAN_DISTANCE,
                10,
                {"index_type": "IVF_PQ", "nlist": 256},
            ),
        ],
    )
    def test_constructor_with_vector_index(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        sample_texts: List[str],
        distance_strategy: DistanceStrategy,
        vector_size: int,
        index_options: dict,
    ) -> None:
        """Test vectorstore creation with vector index enabled."""
        embeddings = generate_embeddings(len(sample_texts), vector_size)
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            distance_strategy=distance_strategy,
            use_vector_index=True,
            vector_size=vector_size,
            vector_index_options=index_options,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        store.add_texts(texts=sample_texts, embeddings=embeddings)
        results = store.similarity_search(
            query="any",
            k=1,
            query_embedding=embeddings[2],
        )
        assert len(results) == 1
        assert results[0].page_content == sample_texts[2]

    @pytest.mark.parametrize(
        "full_text_index_version",
        [FullTextIndexVersion.V1, FullTextIndexVersion.V2],
    )
    def test_constructor_with_fulltext_index(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        sample_texts: List[str],
        precomputed_embeddings: List[List[float]],
        full_text_index_version: FullTextIndexVersion,
    ) -> None:
        """Test vectorstore creation with fulltext index enabled."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            use_full_text_search=True,
            full_text_index_version=full_text_index_version,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        store.add_texts(texts=sample_texts, embeddings=precomputed_embeddings)
        results = store.similarity_search(
            query=sample_texts[1],
            k=1,
            search_strategy=SingleStoreVectorStore.SearchStrategy.TEXT_ONLY,
        )
        assert len(results) == 1
        assert results[0].page_content == sample_texts[1]

    # ============================================================
    # from_texts creation tests
    # ============================================================

    def test_from_texts_with_precomputed_embeddings(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        sample_texts: List[str],
        sample_metadatas: List[dict],
        precomputed_embeddings: List[List[float]],
    ) -> None:
        """Test from_texts with pre-computed embeddings."""
        store = store_tracker.create_from_texts(
            texts=sample_texts,
            embedding=ErrorEmbeddings(),
            metadatas=sample_metadatas,
            embeddings=precomputed_embeddings,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        results = store.similarity_search(
            query="any",
            k=3,
            query_embedding=precomputed_embeddings[1],
        )
        assert len(results) == 3

    @pytest.mark.parametrize(
        "distance_strategy",
        [DistanceStrategy.DOT_PRODUCT, DistanceStrategy.EUCLIDEAN_DISTANCE],
    )
    def test_from_texts_with_different_distance_strategies(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        sample_texts: List[str],
        precomputed_embeddings: List[List[float]],
        distance_strategy: DistanceStrategy,
    ) -> None:
        """Test from_texts with different distance strategies."""
        store = store_tracker.create_from_texts(
            texts=sample_texts,
            embedding=ErrorEmbeddings(),
            embeddings=precomputed_embeddings,
            distance_strategy=distance_strategy,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        results = store.similarity_search(
            query="any",
            k=1,
            query_embedding=precomputed_embeddings[0],
        )
        assert len(results) == 1
        assert results[0].page_content == sample_texts[0]

    @pytest.mark.parametrize(
        "distance_strategy,vector_size",
        [
            (DistanceStrategy.DOT_PRODUCT, 10),
            (DistanceStrategy.EUCLIDEAN_DISTANCE, 10),
        ],
    )
    def test_from_texts_with_vector_index(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        sample_texts: List[str],
        distance_strategy: DistanceStrategy,
        vector_size: int,
    ) -> None:
        """Test from_texts with vector index enabled."""
        embeddings = generate_embeddings(len(sample_texts), vector_size)
        store = store_tracker.create_from_texts(
            texts=sample_texts,
            embedding=ErrorEmbeddings(),
            embeddings=embeddings,
            distance_strategy=distance_strategy,
            use_vector_index=True,
            vector_size=vector_size,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        results = store.similarity_search(
            query="any",
            k=1,
            query_embedding=embeddings[2],
        )
        assert len(results) == 1
        assert results[0].page_content == sample_texts[2]

    @pytest.mark.parametrize(
        "full_text_index_version",
        [FullTextIndexVersion.V1, FullTextIndexVersion.V2],
    )
    def test_from_texts_with_fulltext_index(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        sample_texts: List[str],
        precomputed_embeddings: List[List[float]],
        full_text_index_version: FullTextIndexVersion,
    ) -> None:
        """Test from_texts with fulltext index enabled."""
        store = store_tracker.create_from_texts(
            texts=sample_texts,
            embedding=ErrorEmbeddings(),
            embeddings=precomputed_embeddings,
            use_full_text_search=True,
            full_text_index_version=full_text_index_version,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        results = store.similarity_search(
            query=sample_texts[1],
            k=1,
            search_strategy=SingleStoreVectorStore.SearchStrategy.TEXT_ONLY,
        )
        assert len(results) == 1
        assert results[0].page_content == sample_texts[1]

    # ============================================================
    # from_documents creation tests
    # ============================================================

    def test_from_documents_with_precomputed_embeddings(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        sample_documents: List[Document],
        precomputed_embeddings: List[List[float]],
    ) -> None:
        """Test from_documents with pre-computed embeddings."""
        store = store_tracker.create_from_documents(
            documents=sample_documents,
            embedding=ErrorEmbeddings(),
            embeddings=precomputed_embeddings,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        results = store.similarity_search(
            query="any",
            k=3,
            query_embedding=precomputed_embeddings[0],
        )
        assert len(results) == 3

    @pytest.mark.parametrize(
        "distance_strategy",
        [DistanceStrategy.DOT_PRODUCT, DistanceStrategy.EUCLIDEAN_DISTANCE],
    )
    def test_from_documents_with_different_distance_strategies(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        sample_documents: List[Document],
        precomputed_embeddings: List[List[float]],
        distance_strategy: DistanceStrategy,
    ) -> None:
        """Test from_documents with different distance strategies."""
        store = store_tracker.create_from_documents(
            documents=sample_documents,
            embedding=ErrorEmbeddings(),
            embeddings=precomputed_embeddings,
            distance_strategy=distance_strategy,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        results = store.similarity_search(
            query="any",
            k=1,
            query_embedding=precomputed_embeddings[0],
        )
        assert len(results) == 1
        assert results[0].page_content == sample_documents[0].page_content

    @pytest.mark.parametrize(
        "distance_strategy,vector_size",
        [
            (DistanceStrategy.DOT_PRODUCT, 10),
            (DistanceStrategy.EUCLIDEAN_DISTANCE, 10),
        ],
    )
    def test_from_documents_with_vector_index(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        sample_documents: List[Document],
        distance_strategy: DistanceStrategy,
        vector_size: int,
    ) -> None:
        """Test from_documents with vector index enabled."""
        embeddings = generate_embeddings(len(sample_documents), vector_size)
        store = store_tracker.create_from_documents(
            documents=sample_documents,
            embedding=ErrorEmbeddings(),
            embeddings=embeddings,
            distance_strategy=distance_strategy,
            use_vector_index=True,
            vector_size=vector_size,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        results = store.similarity_search(
            query="any",
            k=1,
            query_embedding=embeddings[2],
        )
        assert len(results) == 1
        assert results[0].page_content == sample_documents[2].page_content

    @pytest.mark.parametrize(
        "full_text_index_version",
        [FullTextIndexVersion.V1, FullTextIndexVersion.V2],
    )
    def test_from_documents_with_fulltext_index(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        sample_documents: List[Document],
        precomputed_embeddings: List[List[float]],
        full_text_index_version: FullTextIndexVersion,
    ) -> None:
        """Test from_documents with fulltext index enabled."""
        store = store_tracker.create_from_documents(
            documents=sample_documents,
            embedding=ErrorEmbeddings(),
            embeddings=precomputed_embeddings,
            use_full_text_search=True,
            full_text_index_version=full_text_index_version,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        results = store.similarity_search(
            query=sample_documents[1].page_content,
            k=1,
            search_strategy=SingleStoreVectorStore.SearchStrategy.TEXT_ONLY,
        )
        assert len(results) == 1
        assert results[0].page_content == sample_documents[1].page_content


class TestPrecomputedEmbeddingsAddImages:
    """Test add_images with pre-computed embeddings."""

    @pytest.fixture
    def temp_image_files(self) -> Generator[List[str], None, None]:
        """Create temporary files to simulate images."""
        temp_files = []
        for i in range(3):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            temp_file.write(f"fake image content {i}".encode())
            temp_file.close()
            temp_files.append(temp_file.name)
        yield temp_files
        for file in temp_files:
            try:
                os.remove(file)
            except Exception:
                pass

    def test_add_images_with_precomputed_embeddings(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        temp_image_files: List[str],
    ) -> None:
        """Test adding images with pre-computed embeddings."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        precomputed_embeddings = generate_embeddings(len(temp_image_files))
        ids = store.add_images(
            uris=temp_image_files,
            embeddings=precomputed_embeddings,
        )
        assert len(ids) == len(temp_image_files)

        # Search using pre-computed query embedding
        results = store.similarity_search(
            query="image",
            k=1,
            query_embedding=precomputed_embeddings[0],
        )
        assert len(results) == 1
        assert results[0].page_content == temp_image_files[0]

    def test_add_images_with_metadatas_and_precomputed_embeddings(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        temp_image_files: List[str],
    ) -> None:
        """Test adding images with metadata and pre-computed embeddings."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        precomputed_embeddings = generate_embeddings(len(temp_image_files))
        metadatas = [
            {"index": i, "type": "image"} for i in range(len(temp_image_files))
        ]
        ids = store.add_images(
            uris=temp_image_files,
            metadatas=metadatas,
            embeddings=precomputed_embeddings,
        )
        assert len(ids) == len(temp_image_files)

        results = store.similarity_search(
            query="image",
            k=3,
            query_embedding=precomputed_embeddings[1],
            filter={"type": "image"},
        )
        assert len(results) == 3


class TestPrecomputedEmbeddingsSimilaritySearch:
    """Test similarity search methods with pre-computed embeddings."""

    @pytest.fixture
    def snow_rain_texts(self) -> List[str]:
        """Sample texts about snow and rain."""
        return [
            "In the parched desert, a sudden rainstorm brought relief.",
            "Amidst the bustling cityscape, the rain fell relentlessly.",
            "High in the mountains, the rain transformed into a delicate mist.",
            "Blanketing the countryside, the snowfall painted a serene tableau.",
            "In the urban landscape, snow descended, transforming streets.",
            "Atop the rugged peaks, snow fell with unyielding intensity.",
        ]

    @pytest.fixture
    def snow_rain_metadatas(self) -> List[dict]:
        """Metadata for snow/rain texts."""
        return [
            {"count": "1", "category": "rain", "group": "a"},
            {"count": "2", "category": "rain", "group": "a"},
            {"count": "3", "category": "rain", "group": "b"},
            {"count": "1", "category": "snow", "group": "b"},
            {"count": "2", "category": "snow", "group": "a"},
            {"count": "3", "category": "snow", "group": "a"},
        ]

    @pytest.fixture
    def snow_rain_documents(
        self, snow_rain_texts: List[str], snow_rain_metadatas: List[dict]
    ) -> List[Document]:
        """Sample documents about snow and rain."""
        return [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(snow_rain_texts, snow_rain_metadatas)
        ]

    @pytest.fixture
    def snow_rain_embeddings(self, snow_rain_texts: List[str]) -> List[List[float]]:
        """Pre-computed embeddings for snow/rain texts."""
        return generate_embeddings(len(snow_rain_texts))

    # ============================================================
    # VECTOR_ONLY search strategy tests
    # ============================================================

    def test_similarity_search_vector_only(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        snow_rain_texts: List[str],
        snow_rain_metadatas: List[dict],
        snow_rain_embeddings: List[List[float]],
    ) -> None:
        """Test VECTOR_ONLY search with pre-computed embeddings."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        store.add_texts(
            texts=snow_rain_texts,
            metadatas=snow_rain_metadatas,
            embeddings=snow_rain_embeddings,
        )
        results = store.similarity_search(
            query="rainstorm",
            k=3,
            query_embedding=snow_rain_embeddings[0],
            search_strategy=SingleStoreVectorStore.SearchStrategy.VECTOR_ONLY,
        )
        assert len(results) == 3
        # First result should be the most similar (index 0)
        assert results[0].page_content == snow_rain_texts[0]

    def test_similarity_search_vector_only_with_filter(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        snow_rain_texts: List[str],
        snow_rain_metadatas: List[dict],
        snow_rain_embeddings: List[List[float]],
    ) -> None:
        """Test VECTOR_ONLY search with metadata filter."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        store.add_texts(
            texts=snow_rain_texts,
            metadatas=snow_rain_metadatas,
            embeddings=snow_rain_embeddings,
        )
        results = store.similarity_search(
            query="rainstorm",
            k=3,
            query_embedding=snow_rain_embeddings[0],
            filter={"category": "snow"},
        )
        assert len(results) == 3
        for doc in results:
            assert doc.metadata["category"] == "snow"

    def test_similarity_search_with_score_vector_only(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        snow_rain_texts: List[str],
        snow_rain_metadatas: List[dict],
        snow_rain_embeddings: List[List[float]],
    ) -> None:
        """Test similarity_search_with_score with pre-computed embeddings."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        store.add_texts(
            texts=snow_rain_texts,
            metadatas=snow_rain_metadatas,
            embeddings=snow_rain_embeddings,
        )
        results = store.similarity_search_with_score(
            query="rainstorm",
            k=3,
            query_embedding=snow_rain_embeddings[0],
        )
        assert len(results) == 3
        # Results should be tuples of (Document, score)
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)

    # ============================================================
    # TEXT_ONLY search strategy tests
    # ============================================================

    @pytest.mark.parametrize(
        "full_text_index_version",
        [FullTextIndexVersion.V1, FullTextIndexVersion.V2],
    )
    def test_similarity_search_text_only(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        snow_rain_texts: List[str],
        snow_rain_metadatas: List[dict],
        snow_rain_embeddings: List[List[float]],
        full_text_index_version: FullTextIndexVersion,
    ) -> None:
        """Test TEXT_ONLY search (embedding not needed for search)."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            use_full_text_search=True,
            full_text_index_version=full_text_index_version,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        store.add_texts(
            texts=snow_rain_texts,
            metadatas=snow_rain_metadatas,
            embeddings=snow_rain_embeddings,
        )
        # TEXT_ONLY doesn't need query_embedding
        results = store.similarity_search(
            query="rainstorm parched desert",
            k=3,
            search_strategy=SingleStoreVectorStore.SearchStrategy.TEXT_ONLY,
        )
        assert len(results) >= 1
        # The first text mentions "parched desert" and "rainstorm"
        assert "parched desert" in results[0].page_content

    @pytest.mark.parametrize(
        "full_text_scoring_mode",
        [
            FullTextScoringMode.BM25,
            FullTextScoringMode.BM25_GLOBAL,
            FullTextScoringMode.MATCH,
        ],
    )
    def test_similarity_search_text_only_scoring_modes_v2(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        snow_rain_texts: List[str],
        snow_rain_metadatas: List[dict],
        snow_rain_embeddings: List[List[float]],
        full_text_scoring_mode: FullTextScoringMode,
    ) -> None:
        """Test TEXT_ONLY search with different scoring modes (V2 index)."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            use_full_text_search=True,
            full_text_index_version=FullTextIndexVersion.V2,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        store.add_texts(
            texts=snow_rain_texts,
            metadatas=snow_rain_metadatas,
            embeddings=snow_rain_embeddings,
        )
        results = store.similarity_search(
            query="snowfall countryside",
            k=1,
            search_strategy=SingleStoreVectorStore.SearchStrategy.TEXT_ONLY,
            full_text_scoring_mode=full_text_scoring_mode,
        )
        assert len(results) == 1
        # Should find the document about snowfall in countryside
        assert (
            "countryside" in results[0].page_content
            or "snow" in results[0].page_content
        )

    # ============================================================
    # FILTER_BY_TEXT search strategy tests
    # ============================================================

    @pytest.mark.parametrize(
        "full_text_index_version",
        [FullTextIndexVersion.V1, FullTextIndexVersion.V2],
    )
    def test_similarity_search_filter_by_text(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        snow_rain_texts: List[str],
        snow_rain_metadatas: List[dict],
        snow_rain_embeddings: List[List[float]],
        full_text_index_version: FullTextIndexVersion,
    ) -> None:
        """Test FILTER_BY_TEXT search with pre-computed embeddings."""
        threshold = 1 if full_text_index_version == FullTextIndexVersion.V2 else 0
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            use_full_text_search=True,
            full_text_index_version=full_text_index_version,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )

        store.add_texts(
            texts=snow_rain_texts,
            metadatas=snow_rain_metadatas,
            embeddings=snow_rain_embeddings,
        )
        results = store.similarity_search(
            query="rainstorm parched desert",
            k=1,
            query_embedding=snow_rain_embeddings[0],
            search_strategy=SingleStoreVectorStore.SearchStrategy.FILTER_BY_TEXT,
            filter_threshold=threshold,
        )
        assert len(results) == 1
        assert "parched desert" in results[0].page_content

    @pytest.mark.parametrize(
        "full_text_scoring_mode",
        [
            FullTextScoringMode.BM25,
            FullTextScoringMode.BM25_GLOBAL,
            FullTextScoringMode.MATCH,
        ],
    )
    def test_similarity_search_filter_by_text_scoring_modes_v2(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        snow_rain_texts: List[str],
        snow_rain_metadatas: List[dict],
        snow_rain_embeddings: List[List[float]],
        full_text_scoring_mode: FullTextScoringMode,
    ) -> None:
        """Test FILTER_BY_TEXT with different scoring modes (V2 index)."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            use_full_text_search=True,
            full_text_index_version=FullTextIndexVersion.V2,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        store.add_texts(
            texts=snow_rain_texts,
            metadatas=snow_rain_metadatas,
            embeddings=snow_rain_embeddings,
        )
        results = store.similarity_search(
            query="rainstorm parched desert",
            k=1,
            query_embedding=snow_rain_embeddings[0],
            search_strategy=SingleStoreVectorStore.SearchStrategy.FILTER_BY_TEXT,
            filter_threshold=0.2,
            full_text_scoring_mode=full_text_scoring_mode,
        )
        assert len(results) == 1

    # ============================================================
    # FILTER_BY_VECTOR search strategy tests
    # ============================================================

    @pytest.mark.parametrize(
        "full_text_index_version",
        [FullTextIndexVersion.V1, FullTextIndexVersion.V2],
    )
    def test_similarity_search_filter_by_vector(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        snow_rain_texts: List[str],
        snow_rain_metadatas: List[dict],
        snow_rain_embeddings: List[List[float]],
        full_text_index_version: FullTextIndexVersion,
    ) -> None:
        """Test FILTER_BY_VECTOR search with pre-computed embeddings."""
        threshold = (
            0.2
            if full_text_index_version == FullTextIndexVersion.V2
            else -0.2
        )
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            use_full_text_search=True,
            full_text_index_version=full_text_index_version,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )

        store.add_texts(
            texts=snow_rain_texts,
            metadatas=snow_rain_metadatas,
            embeddings=snow_rain_embeddings,
        )
        results = store.similarity_search(
            query="rainstorm desert rain",
            k=1,
            query_embedding=snow_rain_embeddings[0],
            filter={"category": "rain"},
            search_strategy=SingleStoreVectorStore.SearchStrategy.FILTER_BY_VECTOR,
            filter_threshold=threshold,
        )
        assert len(results) >= 1

    @pytest.mark.parametrize(
        "full_text_scoring_mode",
        [
            FullTextScoringMode.BM25,
            FullTextScoringMode.BM25_GLOBAL,
            FullTextScoringMode.MATCH,
        ],
    )
    def test_similarity_search_filter_by_vector_scoring_modes_v2(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        snow_rain_texts: List[str],
        snow_rain_metadatas: List[dict],
        snow_rain_embeddings: List[List[float]],
        full_text_scoring_mode: FullTextScoringMode,
    ) -> None:
        """Test FILTER_BY_VECTOR with different scoring modes (V2 index)."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            use_full_text_search=True,
            full_text_index_version=FullTextIndexVersion.V2,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )

        store.add_texts(
            texts=snow_rain_texts,
            metadatas=snow_rain_metadatas,
            embeddings=snow_rain_embeddings,
        )
        results = store.similarity_search(
            query="rainstorm desert rain",
            k=1,
            query_embedding=snow_rain_embeddings[0],
            search_strategy=SingleStoreVectorStore.SearchStrategy.FILTER_BY_VECTOR,
            filter_threshold=-10.0,
            full_text_scoring_mode=full_text_scoring_mode,
        )
        assert len(results) == 1

    # ============================================================
    # WEIGHTED_SUM search strategy tests
    # ============================================================

    @pytest.mark.parametrize(
        "full_text_index_version",
        [FullTextIndexVersion.V1, FullTextIndexVersion.V2],
    )
    def test_similarity_search_weighted_sum(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        snow_rain_texts: List[str],
        snow_rain_metadatas: List[dict],
        snow_rain_embeddings: List[List[float]],
        full_text_index_version: FullTextIndexVersion,
    ) -> None:
        """Test WEIGHTED_SUM search with pre-computed embeddings."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            use_full_text_search=True,
            full_text_index_version=full_text_index_version,
            distance_strategy=DistanceStrategy.DOT_PRODUCT,  # Required for WEIGHTED_SUM
            host=clean_db_url,
            database=TEST_DB_NAME,
        )

        store.add_texts(
            texts=snow_rain_texts,
            metadatas=snow_rain_metadatas,
            embeddings=snow_rain_embeddings,
        )
        results = store.similarity_search(
            query="rainstorm desert rain",
            k=1,
            query_embedding=snow_rain_embeddings[0],
            search_strategy=SingleStoreVectorStore.SearchStrategy.WEIGHTED_SUM,
            text_weight=0.3,
            vector_weight=0.7,
        )
        assert len(results) == 1

    def test_similarity_search_weighted_sum_with_filter(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        snow_rain_texts: List[str],
        snow_rain_metadatas: List[dict],
        snow_rain_embeddings: List[List[float]],
    ) -> None:
        """Test WEIGHTED_SUM search with metadata filter."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            use_full_text_search=True,
            full_text_index_version=FullTextIndexVersion.V2,
            distance_strategy=DistanceStrategy.DOT_PRODUCT,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        store.add_texts(
            texts=snow_rain_texts,
            metadatas=snow_rain_metadatas,
            embeddings=snow_rain_embeddings,
        )
        results = store.similarity_search(
            query="rainstorm desert rain",
            k=1,
            query_embedding=snow_rain_embeddings[0],
            filter={"category": "snow"},
            search_strategy=SingleStoreVectorStore.SearchStrategy.WEIGHTED_SUM,
        )
        assert len(results) == 1
        assert results[0].metadata["category"] == "snow"

    @pytest.mark.parametrize(
        "full_text_scoring_mode",
        [
            FullTextScoringMode.BM25,
            FullTextScoringMode.BM25_GLOBAL,
            FullTextScoringMode.MATCH,
        ],
    )
    def test_similarity_search_weighted_sum_scoring_modes_v2(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        snow_rain_texts: List[str],
        snow_rain_metadatas: List[dict],
        snow_rain_embeddings: List[List[float]],
        full_text_scoring_mode: FullTextScoringMode,
    ) -> None:
        """Test WEIGHTED_SUM with different scoring modes (V2 index)."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            use_full_text_search=True,
            full_text_index_version=FullTextIndexVersion.V2,
            distance_strategy=DistanceStrategy.DOT_PRODUCT,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )

        store.add_texts(
            texts=snow_rain_texts,
            metadatas=snow_rain_metadatas,
            embeddings=snow_rain_embeddings,
        )
        results = store.similarity_search(
            query="snowfall countryside",
            k=1,
            query_embedding=snow_rain_embeddings[3],  # Use embedding for snow text
            search_strategy=SingleStoreVectorStore.SearchStrategy.WEIGHTED_SUM,
            text_weight=0.5,
            vector_weight=0.5,
            full_text_scoring_mode=full_text_scoring_mode,
        )
        assert len(results) == 1

    # ============================================================
    # Error case: WEIGHTED_SUM with EUCLIDEAN_DISTANCE
    # ============================================================

    def test_weighted_sum_unsupported_with_euclidean(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        snow_rain_texts: List[str],
        snow_rain_embeddings: List[List[float]],
    ) -> None:
        """Test that WEIGHTED_SUM raises error with EUCLIDEAN_DISTANCE."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            use_full_text_search=True,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        store.add_texts(texts=snow_rain_texts, embeddings=snow_rain_embeddings)
        with pytest.raises(ValueError) as exc_info:
            store.similarity_search(
                query="test",
                k=1,
                query_embedding=snow_rain_embeddings[0],
                search_strategy=SingleStoreVectorStore.SearchStrategy.WEIGHTED_SUM,
            )
        assert "Search strategy SearchStrategy.WEIGHTED_SUM is not" in str(
            exc_info.value
        )


class TestPrecomputedEmbeddingsAdvancedFiltering:
    """Test advanced FilterTypedDict filtering with pre-computed embeddings."""

    @pytest.fixture
    def numeric_texts(self) -> List[str]:
        """Product texts for filter testing."""
        return [
            "Product A with 100 views",
            "Product B with 200 views",
            "Product C with 50 views",
            "Product D with 300 views",
            "Product E with 150 views",
        ]

    @pytest.fixture
    def numeric_metadatas(self) -> List[dict]:
        """Product metadata for filter testing."""
        return [
            {"name": "A", "views": 100, "rating": 4.5, "active": True},
            {"name": "B", "views": 200, "rating": 4.0, "active": True},
            {"name": "C", "views": 50, "rating": 3.5, "active": False},
            {"name": "D", "views": 300, "rating": 5.0, "active": True},
            {"name": "E", "views": 150, "rating": 4.2, "active": False},
        ]

    @pytest.fixture
    def numeric_embeddings(self, numeric_texts: List[str]) -> List[List[float]]:
        """Pre-computed embeddings for products."""
        return generate_embeddings(len(numeric_texts))

    def test_filter_eq_operator(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        numeric_texts: List[str],
        numeric_metadatas: List[dict],
        numeric_embeddings: List[List[float]],
    ) -> None:
        """Test $eq operator with pre-computed embeddings."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        store.add_texts(
            texts=numeric_texts,
            metadatas=numeric_metadatas,
            embeddings=numeric_embeddings,
        )
        results = store.similarity_search(
            query="product",
            k=10,
            query_embedding=numeric_embeddings[0],
            filter={"name": {"$eq": "B"}},
        )
        assert len(results) == 1
        assert results[0].metadata["name"] == "B"

    def test_filter_gt_operator(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        numeric_texts: List[str],
        numeric_metadatas: List[dict],
        numeric_embeddings: List[List[float]],
    ) -> None:
        """Test $gt operator with pre-computed embeddings."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        store.add_texts(
            texts=numeric_texts,
            metadatas=numeric_metadatas,
            embeddings=numeric_embeddings,
        )
        results = store.similarity_search(
            query="product",
            k=10,
            query_embedding=numeric_embeddings[0],
            filter={"views": {"$gt": 150}},
        )
        assert len(results) == 2
        names = [doc.metadata["name"] for doc in results]
        assert "B" in names  # 200 views
        assert "D" in names  # 300 views

    def test_filter_in_operator(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        numeric_texts: List[str],
        numeric_metadatas: List[dict],
        numeric_embeddings: List[List[float]],
    ) -> None:
        """Test $in operator with pre-computed embeddings."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            host=clean_db_url,
            database=TEST_DB_NAME,
        )

        store.add_texts(
            texts=numeric_texts,
            metadatas=numeric_metadatas,
            embeddings=numeric_embeddings,
        )
        results = store.similarity_search(
            query="product",
            k=10,
            query_embedding=numeric_embeddings[0],
            filter={"name": {"$in": ["A", "C"]}},
        )
        assert len(results) == 2
        names = [doc.metadata["name"] for doc in results]
        assert "A" in names
        assert "C" in names

    def test_filter_and_operator(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        numeric_texts: List[str],
        numeric_metadatas: List[dict],
        numeric_embeddings: List[List[float]],
    ) -> None:
        """Test $and operator with pre-computed embeddings."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        store.add_texts(
            texts=numeric_texts,
            metadatas=numeric_metadatas,
            embeddings=numeric_embeddings,
        )
        results = store.similarity_search(
            query="product",
            k=10,
            query_embedding=numeric_embeddings[0],
            filter={
                "$and": [
                    {"views": {"$gt": 100}},
                    {"active": True},
                ]
            },
        )
        # Products with views > 100 AND active=True: B (200), D (300)
        assert len(results) == 2
        names = [doc.metadata["name"] for doc in results]
        assert "B" in names
        assert "D" in names

    def test_filter_or_operator(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        numeric_texts: List[str],
        numeric_metadatas: List[dict],
        numeric_embeddings: List[List[float]],
    ) -> None:
        """Test $or operator with pre-computed embeddings."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        store.add_texts(
            texts=numeric_texts,
            metadatas=numeric_metadatas,
            embeddings=numeric_embeddings,
        )
        results = store.similarity_search(
            query="product",
            k=10,
            query_embedding=numeric_embeddings[0],
            filter={
                "$or": [
                    {"name": "A"},
                    {"rating": {"$gte": 5.0}},
                ]
            },
        )
        # name=A OR rating >= 5.0: A (4.5 but name matches), D (5.0)
        assert len(results) >= 2
        names = [doc.metadata["name"] for doc in results]
        assert "A" in names
        assert "D" in names

    def test_complex_nested_filter(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        numeric_texts: List[str],
        numeric_metadatas: List[dict],
        numeric_embeddings: List[List[float]],
    ) -> None:
        """Test complex nested $and/$or filter with pre-computed embeddings."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        store.add_texts(
            texts=numeric_texts,
            metadatas=numeric_metadatas,
            embeddings=numeric_embeddings,
        )
        results = store.similarity_search(
            query="product",
            k=10,
            query_embedding=numeric_embeddings[0],
            filter={
                "$and": [
                    {
                        "$or": [
                            {"name": "A"},
                            {"name": "B"},
                        ]
                    },
                    {"views": {"$gte": 150}},
                ]
            },
        )
        # (name=A OR name=B) AND views >= 150: B (200)
        assert len(results) == 1
        assert results[0].metadata["name"] == "B"

    def test_filter_with_search_strategy(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        numeric_texts: List[str],
        numeric_metadatas: List[dict],
        numeric_embeddings: List[List[float]],
    ) -> None:
        """Test FilterTypedDict with TEXT_ONLY search and pre-computed embeddings."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            use_full_text_search=True,
            full_text_index_version=FullTextIndexVersion.V2,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        store.add_texts(
            texts=numeric_texts,
            metadatas=numeric_metadatas,
            embeddings=numeric_embeddings,
        )
        # TEXT_ONLY doesn't need query_embedding
        results = store.similarity_search(
            query="product views",
            k=10,
            filter={"views": {"$gt": 100}},
            search_strategy=SingleStoreVectorStore.SearchStrategy.TEXT_ONLY,
        )
        assert len(results) > 0
        for doc in results:
            assert doc.metadata["views"] > 100


class TestPrecomputedEmbeddingsValidation:
    """Test validation of pre-computed embeddings length and vector size."""

    @pytest.fixture
    def sample_texts(self) -> List[str]:
        """Sample texts for testing."""
        return [
            "The quick brown fox jumps over the lazy dog.",
            "A journey of a thousand miles begins with a single step.",
            "To be or not to be, that is the question.",
        ]

    # ============================================================
    # add_texts validation tests
    # ============================================================

    def test_add_texts_validates_embeddings_length_mismatch(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        sample_texts: List[str],
    ) -> None:
        """Test that add_texts raises error when embeddings count doesn't match."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        # Provide fewer embeddings than texts
        embeddings = generate_embeddings(2)  # Only 2 embeddings for 3 texts

        with pytest.raises(ValueError) as exc_info:
            store.add_texts(texts=sample_texts, embeddings=embeddings)

        assert "number of embeddings must match the number of texts" in str(
            exc_info.value
        )

    def test_add_texts_validates_embeddings_length_too_many(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        sample_texts: List[str],
    ) -> None:
        """Test that add_texts raises error when too many embeddings provided."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        # Provide more embeddings than texts
        embeddings = generate_embeddings(5)  # 5 embeddings for 3 texts

        with pytest.raises(ValueError) as exc_info:
            store.add_texts(texts=sample_texts, embeddings=embeddings)

        assert "number of embeddings must match the number of texts" in str(
            exc_info.value
        )

    def test_add_texts_validates_vector_size_with_vector_index(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        sample_texts: List[str],
    ) -> None:
        """Test that add_texts validates vector size matches vector_size setting."""
        vector_size = 10
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            use_vector_index=True,
            vector_size=vector_size,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        # Provide embeddings with wrong dimension (5 instead of 10)
        wrong_size_embeddings = generate_embeddings(len(sample_texts), size=5)

        with pytest.raises(ValueError) as exc_info:
            store.add_texts(texts=sample_texts, embeddings=wrong_size_embeddings)

        assert "does not match the specified vector_size" in str(exc_info.value)

    def test_add_texts_accepts_correct_vector_size_with_vector_index(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        sample_texts: List[str],
    ) -> None:
        """Test that add_texts accepts embeddings with correct vector size."""
        vector_size = 10
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            use_vector_index=True,
            vector_size=vector_size,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        # Provide embeddings with correct dimension
        correct_embeddings = generate_embeddings(len(sample_texts), size=vector_size)

        # Should not raise
        ids = store.add_texts(texts=sample_texts, embeddings=correct_embeddings)
        assert len(ids) == len(sample_texts)

    def test_add_texts_no_vector_size_validation_without_vector_index(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        sample_texts: List[str],
    ) -> None:
        """Test that vector size is not validated when not using vector index."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            use_vector_index=False,  # No vector index
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        # Any embedding size should work without vector index
        embeddings = generate_embeddings(len(sample_texts), size=5)

        # Should not raise
        ids = store.add_texts(texts=sample_texts, embeddings=embeddings)
        assert len(ids) == len(sample_texts)

    # ============================================================
    # add_images validation tests
    # ============================================================

    def test_add_images_validates_embeddings_length_mismatch(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
    ) -> None:
        """Test that add_images raises error when embeddings count doesn't match."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        image_uris = ["image1.jpg", "image2.jpg", "image3.jpg"]
        # Provide fewer embeddings than images
        embeddings = generate_embeddings(2)  # Only 2 embeddings for 3 images

        with pytest.raises(ValueError) as exc_info:
            store.add_images(uris=image_uris, embeddings=embeddings)

        assert "Length of embeddings must match length of uris" in str(exc_info.value)

    def test_add_images_validates_embeddings_length_too_many(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
    ) -> None:
        """Test that add_images raises error when too many embeddings provided."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        image_uris = ["image1.jpg", "image2.jpg", "image3.jpg"]
        # Provide more embeddings than images
        embeddings = generate_embeddings(5)  # 5 embeddings for 3 images

        with pytest.raises(ValueError) as exc_info:
            store.add_images(uris=image_uris, embeddings=embeddings)

        assert "Length of embeddings must match length of uris" in str(exc_info.value)

    def test_add_images_validates_vector_size_with_vector_index(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
    ) -> None:
        """Test that add_images validates vector size matches vector_size setting."""
        vector_size = 10
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            use_vector_index=True,
            vector_size=vector_size,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        image_uris = ["image1.jpg", "image2.jpg", "image3.jpg"]
        # Provide embeddings with wrong dimension (5 instead of 10)
        wrong_size_embeddings = generate_embeddings(len(image_uris), size=5)

        with pytest.raises(ValueError) as exc_info:
            store.add_images(uris=image_uris, embeddings=wrong_size_embeddings)

        assert "does not match the specified vector_size" in str(exc_info.value)

    def test_add_images_accepts_correct_vector_size_with_vector_index(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
    ) -> None:
        """Test that add_images accepts embeddings with correct vector size."""
        vector_size = 10
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            use_vector_index=True,
            vector_size=vector_size,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        image_uris = ["image1.jpg", "image2.jpg", "image3.jpg"]
        # Provide embeddings with correct dimension
        correct_embeddings = generate_embeddings(len(image_uris), size=vector_size)

        # Should not raise
        ids = store.add_images(uris=image_uris, embeddings=correct_embeddings)
        assert len(ids) == len(image_uris)

    # ============================================================
    # add_documents validation tests (via add_texts)
    # ============================================================

    def test_add_documents_validates_embeddings_length_mismatch(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
    ) -> None:
        """Test that add_documents raises error when embeddings count doesn't match."""
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        documents = [
            Document(page_content="doc 1"),
            Document(page_content="doc 2"),
            Document(page_content="doc 3"),
        ]
        # Provide fewer embeddings than documents
        embeddings = generate_embeddings(2)  # Only 2 embeddings for 3 documents

        with pytest.raises(ValueError) as exc_info:
            store.add_documents(documents=documents, embeddings=embeddings)

        assert "number of embeddings must match the number of texts" in str(
            exc_info.value
        )

    # ============================================================
    # from_texts validation tests
    # ============================================================

    def test_from_texts_validates_embeddings_length_mismatch(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        sample_texts: List[str],
    ) -> None:
        """Test that from_texts raises error when embeddings count doesn't match."""
        # Provide fewer embeddings than texts
        embeddings = generate_embeddings(2)  # Only 2 embeddings for 3 texts

        with pytest.raises(ValueError) as exc_info:
            store_tracker.create_from_texts(
                texts=sample_texts,
                embedding=ErrorEmbeddings(),
                embeddings=embeddings,
                host=clean_db_url,
                database=TEST_DB_NAME,
            )

        assert "number of embeddings must match the number of texts" in str(
            exc_info.value
        )

    # ============================================================
    # from_documents validation tests (via from_texts)
    # ============================================================

    def test_from_documents_validates_embeddings_length_mismatch(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
    ) -> None:
        """Test from_documents raises error when embeddings count doesn't match."""
        documents = [
            Document(page_content="doc 1"),
            Document(page_content="doc 2"),
            Document(page_content="doc 3"),
        ]
        # Provide fewer embeddings than documents
        embeddings = generate_embeddings(2)  # Only 2 embeddings for 3 documents

        with pytest.raises(ValueError) as exc_info:
            store_tracker.create_from_documents(
                documents=documents,
                embedding=ErrorEmbeddings(),
                embeddings=embeddings,
                host=clean_db_url,
                database=TEST_DB_NAME,
            )

        assert "number of embeddings must match the number of texts" in str(
            exc_info.value
        )

    # ============================================================
    # similarity_search query_embedding validation tests
    # ============================================================

    def test_similarity_search_validates_query_embedding_size_with_vector_index(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        sample_texts: List[str],
    ) -> None:
        """Test similarity_search validates query embedding size with vector index."""
        vector_size = 10
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            use_vector_index=True,
            vector_size=vector_size,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        # Add texts with correct embedding size
        correct_embeddings = generate_embeddings(len(sample_texts), size=vector_size)
        store.add_texts(texts=sample_texts, embeddings=correct_embeddings)

        # Try to search with wrong query embedding size (5 instead of 10)
        wrong_query_embedding = generate_embedding(0, size=5)

        with pytest.raises(ValueError) as exc_info:
            store.similarity_search(
                query="test",
                k=1,
                query_embedding=wrong_query_embedding,
            )

        assert "does not match vector size" in str(exc_info.value)

    def test_similarity_search_accepts_correct_query_embedding_size(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        sample_texts: List[str],
    ) -> None:
        """Test similarity_search accepts query embedding with correct size."""
        vector_size = 10
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            use_vector_index=True,
            vector_size=vector_size,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        # Add texts with correct embedding size
        correct_embeddings = generate_embeddings(len(sample_texts), size=vector_size)
        store.add_texts(texts=sample_texts, embeddings=correct_embeddings)

        # Search with correct query embedding size
        correct_query_embedding = generate_embedding(0, size=vector_size)

        # Should not raise
        results = store.similarity_search(
            query="test",
            k=1,
            query_embedding=correct_query_embedding,
        )
        assert len(results) == 1

    def test_similarity_search_with_score_validates_query_embedding_size(
        self,
        store_tracker: StoreTracker,
        clean_db_url: str,
        sample_texts: List[str],
    ) -> None:
        """Test similarity_search_with_score validates query embedding size."""
        vector_size = 10
        store = store_tracker.create(
            embedding=ErrorEmbeddings(),
            use_vector_index=True,
            vector_size=vector_size,
            host=clean_db_url,
            database=TEST_DB_NAME,
        )
        # Add texts with correct embedding size
        correct_embeddings = generate_embeddings(len(sample_texts), size=vector_size)
        store.add_texts(texts=sample_texts, embeddings=correct_embeddings)

        # Try to search with wrong query embedding size
        wrong_query_embedding = generate_embedding(0, size=5)

        with pytest.raises(ValueError) as exc_info:
            store.similarity_search_with_score(
                query="test",
                k=1,
                query_embedding=wrong_query_embedding,
            )

        assert "does not match vector size" in str(exc_info.value)
