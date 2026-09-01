"""Unit tests for langchain_singlestore.vectorstores module."""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.embeddings import Embeddings
from sqlalchemy.pool import Pool

from langchain_singlestore._utils import (
    DistanceStrategy,
    FullTextIndexVersion,
    FullTextScoringMode,
)
from langchain_singlestore.vectorstores import SingleStoreVectorStore


class MockEmbeddings(Embeddings):
    """Mock embeddings for testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


def _make_vs(**kwargs: object) -> SingleStoreVectorStore:
    """Build a vector store backed by a mock pool so tests never touch the DB."""
    params: dict = {
        "embedding": MockEmbeddings(),
        "connection_pool": MagicMock(spec=Pool),
    }
    params.update(kwargs)
    return SingleStoreVectorStore(**params)  # type: ignore[arg-type]


class TestSingleStoreVectorStoreInit(unittest.TestCase):
    def test_init_with_required_params(self) -> None:
        vs = _make_vs()
        assert isinstance(vs.embedding, MockEmbeddings)
        assert vs.table_name == "embeddings"
        assert vs.distance_strategy == DistanceStrategy.DOT_PRODUCT

    def test_init_sets_default_table_name(self) -> None:
        vs = _make_vs()
        assert vs.table_name == "embeddings"

    def test_init_custom_table_name(self) -> None:
        vs = _make_vs(table_name="custom_embeddings")
        assert vs.table_name == "custom_embeddings"

    def test_init_sets_field_names(self) -> None:
        vs = _make_vs()
        assert vs.content_field == "content"
        assert vs.metadata_field == "metadata"
        assert vs.vector_field == "vector"
        assert vs.id_field == "id"

    def test_init_custom_field_names(self) -> None:
        vs = _make_vs(
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
        vs = _make_vs(distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)
        assert vs.distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE

    def test_init_vector_index_disabled_by_default(self) -> None:
        vs = _make_vs()
        assert vs.use_vector_index is False

    def test_init_vector_index_enabled(self) -> None:
        vs = _make_vs(use_vector_index=True)
        assert vs.use_vector_index is True

    def test_init_full_text_search_disabled_by_default(self) -> None:
        vs = _make_vs()
        assert vs.use_full_text_search is False

    def test_init_full_text_search_enabled(self) -> None:
        vs = _make_vs(use_full_text_search=True)
        assert vs.use_full_text_search is True

    def test_init_sets_connector_attributes(self) -> None:
        vs = _make_vs(host="localhost")
        assert "conn_attrs" in vs.connection_kwargs
        assert "_connector_name" in vs.connection_kwargs["conn_attrs"]
        assert "_connector_version" in vs.connection_kwargs["conn_attrs"]

    def test_init_vector_size_default(self) -> None:
        vs = _make_vs()
        assert vs.vector_size == 1536

    def test_init_custom_vector_size(self) -> None:
        vs = _make_vs(vector_size=768)
        assert vs.vector_size == 768

    def test_init_pool_settings(self) -> None:
        """Pool sizing kwargs are forwarded to create_connection_pool."""
        with patch(
            "langchain_singlestore.vectorstores.create_connection_pool"
        ) as mock_factory:
            mock_factory.return_value = MagicMock(spec=Pool)
            SingleStoreVectorStore(
                embedding=MockEmbeddings(),
                host="localhost",
                pool_size=10,
                max_overflow=20,
                timeout=60,
            )
        mock_factory.assert_called_once()
        call_kwargs = mock_factory.call_args.kwargs
        assert call_kwargs["pool_size"] == 10
        assert call_kwargs["max_overflow"] == 20
        assert call_kwargs["timeout"] == 60


class TestSingleStoreVectorStoreSanitize(unittest.TestCase):
    def test_sanitize_removes_special_chars(self) -> None:
        vs = _make_vs()
        assert vs._sanitize_input("test!@#$%^&*()input") == "testinput"

    def test_sanitize_keeps_alphanumeric_and_underscore(self) -> None:
        vs = _make_vs()
        assert vs._sanitize_input("test_123_input") == "test_123_input"


class TestSingleStoreVectorStoreSearchStrategy(unittest.TestCase):
    def test_search_strategies_defined(self) -> None:
        assert hasattr(SingleStoreVectorStore.SearchStrategy, "VECTOR_ONLY")
        assert hasattr(SingleStoreVectorStore.SearchStrategy, "TEXT_ONLY")
        assert hasattr(SingleStoreVectorStore.SearchStrategy, "FILTER_BY_TEXT")
        assert hasattr(SingleStoreVectorStore.SearchStrategy, "FILTER_BY_VECTOR")
        assert hasattr(SingleStoreVectorStore.SearchStrategy, "WEIGHTED_SUM")

    def test_search_strategy_values(self) -> None:
        assert SingleStoreVectorStore.SearchStrategy.VECTOR_ONLY == "VECTOR_ONLY"
        assert SingleStoreVectorStore.SearchStrategy.TEXT_ONLY == "TEXT_ONLY"
        assert SingleStoreVectorStore.SearchStrategy.FILTER_BY_TEXT == "FILTER_BY_TEXT"


class TestSingleStoreVectorStoreEmbeddings(unittest.TestCase):
    def test_embeddings_property(self) -> None:
        embeddings = MockEmbeddings()
        vs = _make_vs(embedding=embeddings)
        assert vs.embeddings is embeddings


class TestFulltextScoringModeToSql(unittest.TestCase):
    def test_match_mode_with_v1_index(self) -> None:
        vs = _make_vs(
            use_full_text_search=True,
            full_text_index_version=FullTextIndexVersion.V1,
        )
        sql, query = vs._fulltext_scoring_mode_to_sql(
            FullTextScoringMode.MATCH, "test query"
        )
        assert sql == "MATCH (content) AGAINST (%s)"
        assert query == "test query"

    def test_match_mode_with_v2_index(self) -> None:
        vs = _make_vs(
            use_full_text_search=True,
            full_text_index_version=FullTextIndexVersion.V2,
        )
        sql, query = vs._fulltext_scoring_mode_to_sql(
            FullTextScoringMode.MATCH, "test query"
        )
        assert sql == "MATCH (TABLE embeddings) AGAINST (%s)"
        assert query == "content:(test query)"

    def test_bm25_mode_with_v2_index(self) -> None:
        vs = _make_vs(
            use_full_text_search=True,
            full_text_index_version=FullTextIndexVersion.V2,
        )
        sql, query = vs._fulltext_scoring_mode_to_sql(
            FullTextScoringMode.BM25, "test query"
        )
        assert sql == "BM25(embeddings, %s)"
        assert query == "content:(test query)"

    def test_bm25_global_mode_with_v2_index(self) -> None:
        vs = _make_vs(
            use_full_text_search=True,
            full_text_index_version=FullTextIndexVersion.V2,
        )
        sql, query = vs._fulltext_scoring_mode_to_sql(
            FullTextScoringMode.BM25_GLOBAL, "test query"
        )
        assert sql == "BM25_GLOBAL(embeddings, %s)"
        assert query == "content:(test query)"

    def test_custom_content_field_with_v1(self) -> None:
        vs = _make_vs(
            use_full_text_search=True,
            full_text_index_version=FullTextIndexVersion.V1,
            content_field="text_content",
        )
        sql, query = vs._fulltext_scoring_mode_to_sql(
            FullTextScoringMode.MATCH, "search terms"
        )
        assert sql == "MATCH (text_content) AGAINST (%s)"
        assert query == "search terms"

    def test_custom_content_field_with_v2_match(self) -> None:
        vs = _make_vs(
            use_full_text_search=True,
            full_text_index_version=FullTextIndexVersion.V2,
            content_field="text_content",
            table_name="my_docs",
        )
        sql, query = vs._fulltext_scoring_mode_to_sql(
            FullTextScoringMode.MATCH, "search terms"
        )
        assert sql == "MATCH (TABLE my_docs) AGAINST (%s)"
        assert query == "text_content:(search terms)"

    def test_custom_content_field_with_v2_bm25(self) -> None:
        vs = _make_vs(
            use_full_text_search=True,
            full_text_index_version=FullTextIndexVersion.V2,
            content_field="text_content",
        )
        sql, query = vs._fulltext_scoring_mode_to_sql(
            FullTextScoringMode.BM25, "search terms"
        )
        assert sql == "BM25(embeddings, %s)"
        assert query == "text_content:(search terms)"


if __name__ == "__main__":
    unittest.main()
