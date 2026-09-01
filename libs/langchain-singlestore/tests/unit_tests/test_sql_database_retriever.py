"""Unit tests for SingleStore SQL Database Retriever."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from sqlalchemy.pool import Pool

from langchain_singlestore.sql_database_retriever import (
    SingleStoreSQLDatabaseChain,
    SingleStoreSQLDatabaseRetriever,
)


def _make_retriever(**kwargs: object) -> SingleStoreSQLDatabaseRetriever:
    """Build a retriever backed by an injected mock pool."""
    params: dict = {"connection_pool": MagicMock(spec=Pool)}
    params.update(kwargs)
    return SingleStoreSQLDatabaseRetriever(**params)  # type: ignore[arg-type]


def _retriever_with_cursor(
    mock_cursor: MagicMock,
) -> tuple[SingleStoreSQLDatabaseRetriever, MagicMock, MagicMock]:
    """Return a retriever whose pool checkouts route to ``mock_cursor``."""
    pool = MagicMock(spec=Pool)
    conn = MagicMock()
    conn.cursor.return_value = mock_cursor
    pool.connect.return_value = conn
    retriever = SingleStoreSQLDatabaseRetriever(connection_pool=pool)
    return retriever, pool, conn


class TestSingleStoreSQLDatabaseRetriever:
    def test_initialization_with_host(self) -> None:
        retriever = _make_retriever(host="localhost:3306/test_db")
        assert retriever.pool_size == 5
        assert retriever.max_overflow == 10
        assert retriever.timeout == 30

    def test_initialization_with_custom_pool_settings(self) -> None:
        with patch(
            "langchain_singlestore.sql_database_retriever.create_connection_pool"
        ) as mock_factory:
            mock_factory.return_value = MagicMock(spec=Pool)
            SingleStoreSQLDatabaseRetriever(
                host="localhost:3306/test_db",
                pool_size=10,
                max_overflow=20,
                timeout=60,
            )
        call_kwargs = mock_factory.call_args.kwargs
        assert call_kwargs["pool_size"] == 10
        assert call_kwargs["max_overflow"] == 20
        assert call_kwargs["timeout"] == 60

    def test_initialization_forwards_connection(self) -> None:
        conn = MagicMock(name="raw_connection")
        with patch(
            "langchain_singlestore.sql_database_retriever.create_connection_pool"
        ) as mock_factory:
            mock_factory.return_value = MagicMock(spec=Pool)
            SingleStoreSQLDatabaseRetriever(connection=conn)
        assert mock_factory.call_args.kwargs["connection"] is conn

    def test_initialization_uses_injected_pool(self) -> None:
        pool = MagicMock(spec=Pool)
        retriever = SingleStoreSQLDatabaseRetriever(connection_pool=pool)
        assert retriever.connection_pool is pool

    def test_initialization_with_custom_row_to_document_fn(self) -> None:
        def custom_converter(row_dict: dict, row_index: int) -> Document:
            return Document(
                page_content=f"Custom: {row_dict}", metadata={"index": row_index}
            )

        retriever = _make_retriever(
            host="localhost:3306/test_db", row_to_document_fn=custom_converter
        )
        assert retriever.row_to_document_fn == custom_converter

    def test_default_row_to_document(self) -> None:
        row_dict = {
            "id": 1,
            "name": "John",
            "email": "john@example.com",
            "data": {"key": "value"},
        }
        doc = SingleStoreSQLDatabaseRetriever._default_row_to_document(row_dict, 0)
        assert isinstance(doc, Document)
        assert "id: 1" in doc.page_content
        assert "name: John" in doc.page_content
        assert "email: john@example.com" in doc.page_content
        assert "data:" in doc.page_content
        assert doc.metadata["row_index"] == 0
        assert doc.metadata["source"] == "singlestore_database"
        assert doc.metadata["id"] == 1
        assert doc.metadata["name"] == "John"

    def test_default_row_to_document_with_json_fields(self) -> None:
        row_dict = {
            "id": 1,
            "tags": ["tag1", "tag2"],
            "properties": {"color": "red", "size": "large"},
        }
        doc = SingleStoreSQLDatabaseRetriever._default_row_to_document(row_dict, 1)
        assert "tags:" in doc.page_content
        assert '["tag1", "tag2"]' in doc.page_content
        assert "properties:" in doc.page_content
        assert "color" in doc.page_content

    def test_execute_query_returns_results_as_dicts(self) -> None:
        mock_cursor = MagicMock()
        mock_cursor.description = [("id",), ("name",), ("email",)]
        mock_cursor.fetchall.return_value = [
            (1, "John", "john@example.com"),
            (2, "Jane", "jane@example.com"),
        ]
        retriever, _, _ = _retriever_with_cursor(mock_cursor)

        query = "SELECT id, name, email FROM users"
        results = retriever._execute_query(query)

        assert len(results) == 2
        assert results[0] == {"id": 1, "name": "John", "email": "john@example.com"}
        assert results[1] == {"id": 2, "name": "Jane", "email": "jane@example.com"}
        mock_cursor.execute.assert_called_once_with(query)

    def test_execute_query_empty_result(self) -> None:
        mock_cursor = MagicMock()
        mock_cursor.description = [("id",), ("name",)]
        mock_cursor.fetchall.return_value = []
        retriever, _, _ = _retriever_with_cursor(mock_cursor)

        assert retriever._execute_query("SELECT * FROM users") == []

    def test_get_relevant_documents(self) -> None:
        mock_cursor = MagicMock()
        mock_cursor.description = [("id",), ("name",)]
        mock_cursor.fetchall.return_value = [(1, "Alice"), (2, "Bob")]
        retriever, _, _ = _retriever_with_cursor(mock_cursor)

        docs = retriever._get_relevant_documents(
            "SELECT id, name FROM users", run_manager=MagicMock()
        )
        assert len(docs) == 2
        assert all(isinstance(doc, Document) for doc in docs)
        assert "id: 1" in docs[0].page_content
        assert "name: Alice" in docs[0].page_content
        assert "id: 2" in docs[1].page_content
        assert "name: Bob" in docs[1].page_content

    def test_get_relevant_documents_with_custom_converter(self) -> None:
        mock_cursor = MagicMock()
        mock_cursor.description = [("id",), ("name",)]
        mock_cursor.fetchall.return_value = [(1, "Alice")]

        def custom_converter(row_dict: dict, row_index: int) -> Document:
            return Document(
                page_content=f"Custom-{row_dict['name']}", metadata={"custom": True}
            )

        pool = MagicMock(spec=Pool)
        conn = MagicMock()
        conn.cursor.return_value = mock_cursor
        pool.connect.return_value = conn
        retriever = SingleStoreSQLDatabaseRetriever(
            connection_pool=pool, row_to_document_fn=custom_converter
        )

        docs = retriever._get_relevant_documents(
            "SELECT id, name FROM users", run_manager=MagicMock()
        )
        assert len(docs) == 1
        assert docs[0].page_content == "Custom-Alice"
        assert docs[0].metadata["custom"] is True

    def test_close_connection_pool(self) -> None:
        pool = MagicMock(spec=Pool)
        retriever = SingleStoreSQLDatabaseRetriever(connection_pool=pool)
        retriever.close()
        pool.dispose.assert_called_once()


class TestSingleStoreSQLDatabaseChain:
    def test_from_url_returns_retriever(self) -> None:
        with patch(
            "langchain_singlestore.sql_database_retriever.create_connection_pool"
        ) as mock_factory:
            mock_factory.return_value = MagicMock(spec=Pool)
            mock_llm = MagicMock()
            retriever = SingleStoreSQLDatabaseChain.from_url(
                host="localhost:3306/test_db", llm=mock_llm
            )
            assert isinstance(retriever, SingleStoreSQLDatabaseRetriever)

    def test_query_to_document_executes_query(self) -> None:
        mock_cursor = MagicMock()
        mock_cursor.description = [("count",)]
        mock_cursor.fetchall.return_value = [(42,)]
        pool = MagicMock(spec=Pool)
        conn = MagicMock()
        conn.cursor.return_value = mock_cursor
        pool.connect.return_value = conn

        with patch(
            "langchain_singlestore.sql_database_retriever.create_connection_pool",
            return_value=pool,
        ):
            docs = SingleStoreSQLDatabaseChain.query_to_document(
                query="SELECT COUNT(*) as count FROM users",
                host="localhost:3306/test_db",
            )
        assert len(docs) == 1
        assert "count: 42" in docs[0].page_content

    def test_query_to_document_applies_row_limit(self) -> None:
        mock_cursor = MagicMock()
        mock_cursor.description = [("id",)]
        mock_cursor.fetchall.return_value = [(1,), (2,), (3,)]
        pool = MagicMock(spec=Pool)
        conn = MagicMock()
        conn.cursor.return_value = mock_cursor
        pool.connect.return_value = conn

        with patch(
            "langchain_singlestore.sql_database_retriever.create_connection_pool",
            return_value=pool,
        ):
            SingleStoreSQLDatabaseChain.query_to_document(
                query="SELECT id FROM users",
                host="localhost:3306/test_db",
                row_limit=10,
            )
        called_query = mock_cursor.execute.call_args[0][0]
        assert "LIMIT 10" in str(called_query)
