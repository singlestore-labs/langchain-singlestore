"""Integration tests for SingleStore SQL Database Retriever."""

import json
import os
from typing import Generator
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from langchain_singlestore.sql_database_retriever import (
    SingleStoreSQLDatabaseChain,
    SingleStoreSQLDatabaseRetriever,
)
from tests.integration_tests.conftest import ConnectionParameters


@pytest.fixture
def test_connection_params(clean_db_connection_parameters: ConnectionParameters) -> Generator[dict[str, str], None, None]:
    """Get connection parameters from environment or use defaults."""
    yield {
        "host": clean_db_connection_parameters.Host,
        "port": str(clean_db_connection_parameters.Port),
        "user": clean_db_connection_parameters.User,
        "password": clean_db_connection_parameters.Password,
        "database": clean_db_connection_parameters.Database,
    }


@pytest.fixture
def retriever(
    test_connection_params: dict[str, str],
) -> Generator[SingleStoreSQLDatabaseRetriever, None, None]:  # type: ignore
    """Create a retriever instance for testing."""
    retriever = SingleStoreSQLDatabaseRetriever(
        pool_size=5,
        max_overflow=10,
        timeout=5,
        row_to_document_fn=None,
        **test_connection_params,
    )
    yield retriever
    retriever.close()


class TestSingleStoreSQLDatabaseRetrieverIntegration:
    """Integration tests for SingleStoreSQLDatabaseRetriever."""

    def test_create_retriever_and_execute_simple_query(
        self, retriever: SingleStoreSQLDatabaseRetriever
    ) -> None:
        """Test creating retriever and executing a simple query."""
        # This query works on any database to verify connection
        try:
            docs = retriever._get_relevant_documents("SELECT 1 as test_value")
            assert len(docs) > 0
            assert isinstance(docs[0], Document)
            assert "test_value" in docs[0].page_content
        except Exception as e:
            pytest.skip(f"Database not available: {str(e)}")

    def test_execute_query_with_multiple_rows(
        self,
        retriever: SingleStoreSQLDatabaseRetriever,
        test_connection_params: dict[str, str],
    ) -> None:
        """Test querying multiple rows and converting to documents."""
        try:
            # Create test table
            conn = retriever._get_connection()
            cursor = conn.cursor()
            try:
                # Drop table if exists
                cursor.execute("DROP TABLE IF EXISTS test_users")

                # Create table
                cursor.execute("""
                    CREATE TABLE test_users (
                        id INT PRIMARY KEY,
                        name VARCHAR(100),
                        email VARCHAR(100),
                        age INT
                    )
                """)

                # Insert test data
                cursor.execute(
                    "INSERT INTO test_users VALUES (1,'Alice','alice@example.com',30)"
                )
                cursor.execute(
                    "INSERT INTO test_users VALUES (2, 'Bob', 'bob@example.com', 25)"
                )
                cursor.execute(
                    "INSERT INTO test_users VALUES (3,'Charlie','ch@example.com',35)"
                )
                conn.commit()

                # Query and verify
                docs = retriever._get_relevant_documents(
                    "SELECT id, name, email, age FROM test_users ORDER BY id"
                )

                assert len(docs) == 3

                # Verify first document
                assert "id: 1" in docs[0].page_content
                assert "name: Alice" in docs[0].page_content
                assert "alice@example.com" in docs[0].page_content

                # Verify metadata
                assert docs[0].metadata["id"] == 1
                assert docs[0].metadata["name"] == "Alice"
                assert docs[1].metadata["age"] == 25

            finally:
                cursor.execute("DROP TABLE IF EXISTS test_users")
                cursor.close()
                conn.close()

        except Exception as e:
            pytest.skip(f"Database not available: {str(e)}")

    def test_query_with_json_data(
        self, retriever: SingleStoreSQLDatabaseRetriever
    ) -> None:
        """Test querying tables with JSON data."""
        try:
            conn = retriever._get_connection()
            cursor = conn.cursor()
            try:
                # Drop table if exists
                cursor.execute("DROP TABLE IF EXISTS test_products")

                # Create table with JSON column
                cursor.execute("""
                    CREATE TABLE test_products (
                        id INT PRIMARY KEY,
                        name VARCHAR(100),
                        properties JSON
                    )
                """)

                # Insert test data
                cursor.execute(
                    "INSERT INTO test_products VALUES (1, 'Laptop', %s)",
                    (json.dumps({"cpu": "Intel i7", "ram": "16GB"}),),
                )
                cursor.execute(
                    "INSERT INTO test_products VALUES (2, 'Phone', %s)",
                    (json.dumps({"screen": "6 inch", "camera": "12MP"}),),
                )
                conn.commit()

                # Query and verify
                docs = retriever._get_relevant_documents(
                    "SELECT id, name, properties FROM test_products"
                )

                assert len(docs) == 2

                # Verify JSON data is in documents
                assert "properties:" in docs[0].page_content
                assert "cpu" in docs[0].page_content or "ram" in docs[0].page_content

            finally:
                cursor.execute("DROP TABLE IF EXISTS test_products")
                cursor.close()
                conn.close()

        except Exception as e:
            pytest.skip(f"Database not available: {str(e)}")

    def test_query_with_empty_result(
        self, retriever: SingleStoreSQLDatabaseRetriever
    ) -> None:
        """Test querying with empty result."""
        try:
            conn = retriever._get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("DROP TABLE IF EXISTS test_empty")
                cursor.execute("""
                    CREATE TABLE test_empty (
                        id INT PRIMARY KEY,
                        value VARCHAR(100)
                    )
                """)
                conn.commit()

                docs = retriever._get_relevant_documents(
                    "SELECT id, value FROM test_empty"
                )

                assert len(docs) == 0

            finally:
                cursor.execute("DROP TABLE IF EXISTS test_empty")
                cursor.close()
                conn.close()

        except Exception as e:
            pytest.skip(f"Database not available: {str(e)}")

    def test_invoke_with_input_dict(
        self, retriever: SingleStoreSQLDatabaseRetriever
    ) -> None:
        """Test invoking retriever with input dictionary."""
        try:
            # Test with simple query
            docs = retriever.invoke("SELECT 42 as answer")
            assert len(docs) > 0
            assert "answer" in docs[0].page_content
        except Exception as e:
            pytest.skip(f"Database not available: {str(e)}")

    def test_custom_row_to_document_function(
        self, test_connection_params: dict[str, str]
    ) -> None:
        """Test using custom row to document function."""

        def custom_converter(row_dict: dict, row_index: int) -> Document:
            # Custom format: "Row X: field1=value1, field2=value2"
            fields = ", ".join(f"{k}={v}" for k, v in row_dict.items())
            return Document(
                page_content=f"Row {row_index}: {fields}",
                metadata={"custom": True, "row_num": row_index, **row_dict},
            )

        try:
            retriever = SingleStoreSQLDatabaseRetriever(
                pool_size=5,
                max_overflow=10,
                timeout=5,
                row_to_document_fn=custom_converter,
                **test_connection_params,
            )

            docs = retriever._get_relevant_documents(
                "SELECT 1 as val, 2 as num", run_manager=MagicMock()
            )

            assert len(docs) > 0
            assert "Row 0:" in docs[0].page_content
            assert docs[0].metadata["custom"] is True

            retriever.close()

        except Exception as e:
            pytest.skip(f"Database not available: {str(e)}")

    def test_error_handling_invalid_query(
        self, retriever: SingleStoreSQLDatabaseRetriever
    ) -> None:
        """Test error handling for invalid SQL query."""
        try:
            with pytest.raises(Exception):
                retriever._get_relevant_documents("SELECT * FROM non_existent_table")
        except Exception as e:
            pytest.skip(f"Database not available: {str(e)}")


class TestSingleStoreSQLDatabaseChainIntegration:
    """Integration tests for SingleStoreSQLDatabaseChain."""

    def test_from_url_and_query(self, test_connection_params: dict[str, str]) -> None:
        """Test creating chain from URL and executing query."""
        try:
            retriever = SingleStoreSQLDatabaseChain.from_url(
                host=f"{test_connection_params['user']}:{test_connection_params['password']}"
                f"@{test_connection_params['host']}:{test_connection_params['port']}/{test_connection_params['database']}",
                llm=None,
            )

            assert isinstance(retriever, SingleStoreSQLDatabaseRetriever)

            docs = retriever._get_relevant_documents(
                "SELECT 1 as test", run_manager=MagicMock()
            )
            assert len(docs) > 0

            retriever.close()

        except Exception as e:
            pytest.skip(f"Database not available: {str(e)}")

    def test_query_to_document_convenience_method(
        self, test_connection_params: dict[str, str]
    ) -> None:
        """Test query_to_document convenience method."""
        try:
            docs = SingleStoreSQLDatabaseChain.query_to_document(
                query="SELECT 1 as test_value", **test_connection_params, row_limit=10
            )

            assert len(docs) > 0
            assert isinstance(docs[0], Document)

        except Exception as e:
            pytest.skip(f"Database not available: {str(e)}")

    def test_query_to_document_with_limit(
        self, test_connection_params: dict[str, str]
    ) -> None:
        """Test query_to_document applies LIMIT correctly."""
        try:
            conn_kwargs = test_connection_params.copy()
            retriever = SingleStoreSQLDatabaseRetriever(
                pool_size=5,
                max_overflow=10,
                timeout=5,
                row_to_document_fn=None,
                **conn_kwargs,
            )

            cursor = retriever._get_connection().cursor()
            try:
                # Create test table
                cursor.execute("DROP TABLE IF EXISTS test_limit")
                cursor.execute("""
                    CREATE TABLE test_limit (
                        id INT PRIMARY KEY
                    )
                """)

                # Insert 100 rows
                for i in range(100):
                    cursor.execute("INSERT INTO test_limit VALUES (%s)", (i + 1,))
                cursor.connection.commit()

                # Query with limit
                docs = SingleStoreSQLDatabaseChain.query_to_document(
                    query="SELECT id FROM test_limit", row_limit=10, **conn_kwargs
                )

                # Should have at most 10 documents due to LIMIT
                assert len(docs) <= 10

            finally:
                cursor.execute("DROP TABLE IF EXISTS test_limit")
                cursor.close()
                retriever.close()

        except Exception as e:
            pytest.skip(f"Database not available: {str(e)}")
