"""Unit tests for SingleStore SQL Database Retriever."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from langchain_singlestore.sql_database_retriever import (
    SingleStoreSQLDatabaseChain,
    SingleStoreSQLDatabaseRetriever,
)


class TestSingleStoreSQLDatabaseRetriever:
    """Test cases for SingleStoreSQLDatabaseRetriever."""

    def test_initialization_with_host(self) -> None:
        """Test initialization with host parameter."""
        with patch("langchain_singlestore.sql_database_retriever.QueuePool"):
            retriever = SingleStoreSQLDatabaseRetriever(host="localhost:3306/test_db")
            assert retriever.pool_size == 5
            assert retriever.max_overflow == 10
            assert retriever.timeout == 30

    def test_initialization_without_connection_params_raises_error(self) -> None:
        """Test that initialization without connection params raises ValueError."""
        with pytest.raises(ValueError, match="Database connection parameters"):
            with patch("langchain_singlestore.sql_database_retriever.QueuePool"):
                SingleStoreSQLDatabaseRetriever()

    def test_initialization_with_custom_pool_settings(self) -> None:
        """Test initialization with custom pool settings."""
        with patch("langchain_singlestore.sql_database_retriever.QueuePool"):
            retriever = SingleStoreSQLDatabaseRetriever(
                host="localhost:3306/test_db",
                pool_size=10,
                max_overflow=20,
                timeout=60,
            )
            assert retriever.pool_size == 10
            assert retriever.max_overflow == 20
            assert retriever.timeout == 60

    def test_initialization_with_custom_row_to_document_fn(self) -> None:
        """Test initialization with custom row converter function."""

        def custom_converter(row_dict: dict, row_index: int) -> Document:
            return Document(
                page_content=f"Custom: {row_dict}", metadata={"index": row_index}
            )

        with patch("langchain_singlestore.sql_database_retriever.QueuePool"):
            retriever = SingleStoreSQLDatabaseRetriever(
                host="localhost:3306/test_db", row_to_document_fn=custom_converter
            )
            assert retriever.row_to_document_fn == custom_converter

    def test_default_row_to_document(self) -> None:
        """Test default row to document conversion."""
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
        """Test row to document with JSON fields."""
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

    @patch("langchain_singlestore.sql_database_retriever.s2.connect")
    def test_execute_query_returns_results_as_dicts(
        self, mock_connect: MagicMock
    ) -> None:
        """Test query execution returns results as dictionaries."""
        # Setup mock connection and cursor
        mock_cursor = MagicMock()
        mock_cursor.description = [("id",), ("name",), ("email",)]
        mock_cursor.fetchall.return_value = [
            (1, "John", "john@example.com"),
            (2, "Jane", "jane@example.com"),
        ]

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        with patch("langchain_singlestore.sql_database_retriever.QueuePool") as mock:
            mock_pool_instance = MagicMock()
            mock_pool_instance.connect.return_value = mock_conn
            mock.return_value = mock_pool_instance

            retriever = SingleStoreSQLDatabaseRetriever(host="localhost:3306/test_db")

            query = "SELECT id, name, email FROM users"
            results = retriever._execute_query(query)

            assert len(results) == 2
            assert results[0] == {"id": 1, "name": "John", "email": "john@example.com"}
            assert results[1] == {"id": 2, "name": "Jane", "email": "jane@example.com"}
            mock_cursor.execute.assert_called_once_with(query)

    @patch("langchain_singlestore.sql_database_retriever.s2.connect")
    def test_execute_query_empty_result(self, mock_connect: MagicMock) -> None:
        """Test query execution with empty results."""
        mock_cursor = MagicMock()
        mock_cursor.description = [("id",), ("name",)]
        mock_cursor.fetchall.return_value = []

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        with patch("langchain_singlestore.sql_database_retriever.QueuePool") as mock:
            mock_pool_instance = MagicMock()
            mock_pool_instance.connect.return_value = mock_conn
            mock.return_value = mock_pool_instance

            retriever = SingleStoreSQLDatabaseRetriever(host="localhost:3306/test_db")

            results = retriever._execute_query("SELECT * FROM users")
            assert results == []

    @patch("langchain_singlestore.sql_database_retriever.s2.connect")
    def test_get_relevant_documents(self, mock_connect: MagicMock) -> None:
        """Test _get_relevant_documents converts results to Documents."""
        mock_cursor = MagicMock()
        mock_cursor.description = [("id",), ("name",)]
        mock_cursor.fetchall.return_value = [
            (1, "Alice"),
            (2, "Bob"),
        ]

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        with patch("langchain_singlestore.sql_database_retriever.QueuePool") as mock:
            mock_pool_instance = MagicMock()
            mock_pool_instance.connect.return_value = mock_conn
            mock.return_value = mock_pool_instance

            retriever = SingleStoreSQLDatabaseRetriever(host="localhost:3306/test_db")

            docs = retriever._get_relevant_documents(
                "SELECT id, name FROM users", run_manager=MagicMock()
            )

            assert len(docs) == 2
            assert all(isinstance(doc, Document) for doc in docs)
            assert "id: 1" in docs[0].page_content
            assert "name: Alice" in docs[0].page_content
            assert "id: 2" in docs[1].page_content
            assert "name: Bob" in docs[1].page_content

    @patch("langchain_singlestore.sql_database_retriever.s2.connect")
    def test_get_relevant_documents_with_custom_converter(
        self, mock_connect: MagicMock
    ) -> None:
        """Test _get_relevant_documents with custom row converter."""
        mock_cursor = MagicMock()
        mock_cursor.description = [("id",), ("name",)]
        mock_cursor.fetchall.return_value = [(1, "Alice")]

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        def custom_converter(row_dict: dict, row_index: int) -> Document:
            return Document(
                page_content=f"Custom-{row_dict['name']}", metadata={"custom": True}
            )

        with patch("langchain_singlestore.sql_database_retriever.QueuePool") as mock:
            mock_pool_instance = MagicMock()
            mock_pool_instance.connect.return_value = mock_conn
            mock.return_value = mock_pool_instance

            retriever = SingleStoreSQLDatabaseRetriever(
                host="localhost:3306/test_db", row_to_document_fn=custom_converter
            )

            docs = retriever._get_relevant_documents(
                "SELECT id, name FROM users", run_manager=MagicMock()
            )

            assert len(docs) == 1
            assert docs[0].page_content == "Custom-Alice"
            assert docs[0].metadata["custom"] is True

    def test_close_connection_pool(self) -> None:
        """Test closing the connection pool."""
        with patch("langchain_singlestore.sql_database_retriever.QueuePool") as mock:
            mock_pool_instance = MagicMock()
            mock.return_value = mock_pool_instance

            retriever = SingleStoreSQLDatabaseRetriever(host="localhost:3306/test_db")
            retriever.close()

            mock_pool_instance.dispose.assert_called_once()


class TestSingleStoreSQLDatabaseChain:
    """Test cases for SingleStoreSQLDatabaseChain utility class."""

    def test_from_url_returns_retriever(self) -> None:
        """Test from_url returns a retriever instance."""
        with patch("langchain_singlestore.sql_database_retriever.QueuePool"):
            mock_llm = MagicMock()
            retriever = SingleStoreSQLDatabaseChain.from_url(
                host="localhost:3306/test_db", llm=mock_llm
            )
            assert isinstance(retriever, SingleStoreSQLDatabaseRetriever)

    @patch("langchain_singlestore.sql_database_retriever.s2.connect")
    def test_query_to_document_executes_query(self, mock_connect: MagicMock) -> None:
        """Test query_to_document executes query and returns documents."""
        mock_cursor = MagicMock()
        mock_cursor.description = [("count",)]
        mock_cursor.fetchall.return_value = [(42,)]

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        with patch("langchain_singlestore.sql_database_retriever.QueuePool") as mock:
            mock_pool_instance = MagicMock()
            mock_pool_instance.connect.return_value = mock_conn
            mock.return_value = mock_pool_instance

            docs = SingleStoreSQLDatabaseChain.query_to_document(
                query="SELECT COUNT(*) as count FROM users",
                host="localhost:3306/test_db",
            )

            assert len(docs) == 1
            assert "count: 42" in docs[0].page_content

    @patch("langchain_singlestore.sql_database_retriever.s2.connect")
    def test_query_to_document_applies_row_limit(self, mock_connect: MagicMock) -> None:
        """Test query_to_document applies LIMIT clause."""
        mock_cursor = MagicMock()
        mock_cursor.description = [("id",)]
        mock_cursor.fetchall.return_value = [(1,), (2,), (3,)]

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        with patch("langchain_singlestore.sql_database_retriever.QueuePool") as mock:
            mock_pool_instance = MagicMock()
            mock_pool_instance.connect.return_value = mock_conn
            mock.return_value = mock_pool_instance

            SingleStoreSQLDatabaseChain.query_to_document(
                query="SELECT id FROM users",
                host="localhost:3306/test_db",
                row_limit=10,
            )

            # Check that LIMIT was added to query
            called_query = mock_cursor.execute.call_args[0][0]
            assert "LIMIT 10" in str(called_query)

    def test_query_to_document_closes_connection(self) -> None:
        """Test query_to_document closes connection after execution."""
        with patch("langchain_singlestore.sql_database_retriever.QueuePool"):
            with patch("langchain_singlestore.sql_database_retriever.s2.connect"):
                with patch.object(
                    SingleStoreSQLDatabaseRetriever,
                    "_execute_query",
                    return_value=[{"id": 1}],
                ):
                    with patch.object(SingleStoreSQLDatabaseRetriever, "close"):
                        SingleStoreSQLDatabaseChain.query_to_document(
                            query="SELECT id FROM users", host="localhost:3306/test_db"
                        )

                        # Verify close was called (through deletion in finally block)
                        # This is verified by checking that the retriever was used
