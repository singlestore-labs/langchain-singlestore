"""SingleStore SQL Database Retriever for executing SQL queries
and retrieving results."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, List, Optional

import singlestoredb as s2
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sqlalchemy.pool import QueuePool

from langchain_singlestore._utils import set_connector_attributes

logger = logging.getLogger(__name__)


class SingleStoreSQLDatabaseRetriever(BaseRetriever):
    """Retriever for executing SQL queries against SingleStore and returning
    results as documents.

    This retriever enables LangChain agents and chains to execute SQL queries
    directly against a SingleStore database and retrieve results formatted as
    Document objects. It's useful for building database-aware AI applications
    that need to query structured data.

    Setup:
        Install ``langchain-singlestore``

        .. code-block:: bash

            pip install -U langchain-singlestore

    Instantiate:
        .. code-block:: python

            from langchain_singlestore import SingleStoreSQLDatabaseRetriever

            retriever = SingleStoreSQLDatabaseRetriever(
                host="root:password@localhost:3306/my_database",
            )

    Invoke:
        .. code-block:: python

            # Execute a query and get results as documents
            docs = retriever.invoke({
                "query": "SELECT id, name, email FROM users LIMIT 10"
            })

            # Each row becomes a document with content and metadata
            for doc in docs:
                print(doc.page_content)
                print(doc.metadata)

    Use with agent:
        .. code-block:: python

            from langchain.agents import create_tool_use_agent
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model="gpt-4")
            tools = [
                {
                    "name": "query_database",
                    "description": "Execute SQL queries",
                    "retriever": retriever
                }
            ]

            agent = create_tool_use_agent(llm, tools)
    """

    pool_size: int = 5
    max_overflow: int = 10
    timeout: float = 30
    row_to_document_fn: Optional[Callable[[dict, int], Document]] = None
    connection_kwargs: dict[str, Any] = {}
    connection_pool: Optional[QueuePool] = None

    """
    Attributes:
        connection_kwargs: Keyword arguments for database connection.
        pool_size: Size of the connection pool.
        max_overflow: Maximum connections beyond pool_size.
        timeout: Connection timeout in seconds.
    """

    def __init__(
        self,
        *,
        pool_size: int = 5,
        max_overflow: int = 10,
        timeout: float = 30,
        row_to_document_fn: Optional[Callable[[dict, int], Document]] = None,
        **kwargs: Any,
    ):
        """Initialize SingleStore SQL Database Retriever.

        Args:
            pool_size (int, optional): Determines the number of active connections in
                the pool. Defaults to 5.
            max_overflow (int, optional): Determines the maximum number of connections
                allowed beyond the pool_size. Defaults to 10.
            timeout (float, optional): Specifies the maximum wait time in seconds for
                establishing a connection. Defaults to 30.
            row_to_document_fn (callable, optional): Custom function to convert a row
                to a Document. If None, uses default conversion. The function should
                accept (row_dict, row_index) and return a Document object.

            Following arguments pertain to the database connection:

            host (str, optional): Specifies the hostname, IP address, or URL for the
                database connection. The default scheme is "mysql".
            user (str, optional): Database username.
            password (str, optional): Database password.
            port (int, optional): Database port. Defaults to 3306 for non-HTTP
                connections, 80 for HTTP connections, and 443 for HTTPS connections.
            database (str, optional): Database name.

            Additional optional arguments provide further customization over the
            database connection. See singlestoredb documentation for details.

        Raises:
            ValueError: If database connection parameters are missing.
        """
        super().__init__()

        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.timeout = timeout
        self.row_to_document_fn = row_to_document_fn or self._default_row_to_document

        # Validate connection parameters
        if not kwargs:
            raise ValueError(
                "Database connection parameters must be provided. "
                "Specify at least 'host' parameter."
            )

        # Connection configuration
        self.connection_kwargs = kwargs
        set_connector_attributes(self.connection_kwargs)

        # Create connection pool
        self.connection_pool = QueuePool(
            self._get_connection,
            max_overflow=max_overflow,
            pool_size=pool_size,
            timeout=timeout,
        )

    def _get_connection(self) -> Any:
        """Get a connection from the pool."""
        return s2.connect(**self.connection_kwargs)

    @staticmethod
    def _default_row_to_document(row_dict: dict, row_index: int) -> Document:
        """Convert a database row to a Document object.

        Args:
            row_dict: Dictionary representation of database row.
            row_index: Index of the row in result set.

        Returns:
            Document: Document with row data as content and metadata.
        """
        # Create content from all columns
        content_lines = []
        for key, value in row_dict.items():
            formatted_value = (
                json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            )
            content_lines.append(f"{key}: {formatted_value}")

        page_content = "\n".join(content_lines)

        # Create metadata with row information
        metadata = {
            "row_index": row_index,
            "source": "singlestore_database",
        }
        metadata.update(row_dict)

        return Document(page_content=page_content, metadata=metadata)

    def _execute_query(self, query: str) -> List[dict]:
        """Execute a SQL query and return results as list of dictionaries.

        Args:
            query: SQL query to execute.

        Returns:
            List[dict]: List of rows as dictionaries.

        Raises:
            Exception: If query execution fails.
        """
        if self.connection_pool is None:
            raise RuntimeError(
                "Connection pool not initialized. "
                "Ensure database connection parameters are provided."
            )
        conn = self.connection_pool.connect()
        try:
            # Use namedtuple results for easy dict conversion
            cursor = conn.cursor()
            try:
                cursor.execute(query)
                result = cursor.fetchall()

                # Convert rows to dicts
                if not result:
                    return []

                # Get column names from cursor description
                column_names = [desc[0] for desc in cursor.description]

                # Convert tuples to dicts
                result_dicts = []
                for row in result:
                    row_dict = dict(zip(column_names, row))
                    result_dicts.append(row_dict)

                return result_dicts

            except Exception as e:
                logger.error(f"Error executing query: {query}")
                logger.error(f"Error details: {str(e)}")
                raise
            finally:
                cursor.close()
        finally:
            conn.close()

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Execute query and return results as documents.

        This is the main retriever method called by LangChain.

        Args:
            query: SQL query string to execute.
            run_manager: Callback manager for retriever run.

        Returns:
            List[Document]: List of Document objects containing query results.
        """
        try:
            result_dicts = self._execute_query(query)

            # Convert each row to a Document
            row_converter = self.row_to_document_fn or self._default_row_to_document
            documents = [
                row_converter(row_dict, idx)
                for idx, row_dict in enumerate(result_dicts)
            ]

            logger.info(f"Retrieved {len(documents)} documents from query")
            return documents

        except Exception as e:
            logger.error(f"Failed to retrieve documents from query: {str(e)}")
            raise

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Async version of _get_relevant_documents.

        Note: This implementation uses sync execution. For true async,
        consider using asyncpg or similar async driver.

        Args:
            query: SQL query string to execute.
            run_manager: Async callback manager for retriever run.

        Returns:
            List[Document]: List of Document objects containing query results.
        """
        # For now, use sync implementation in async context
        # In future, this could be enhanced with true async driver
        return self._get_relevant_documents(query)

    def close(self) -> None:
        """Close the connection pool."""
        if self.connection_pool is not None:
            self.connection_pool.dispose()

    def __del__(self) -> None:
        """Ensure connection pool is closed when object is destroyed."""
        try:
            self.close()
        except Exception as e:
            logger.warning(f"Error closing connection pool: {str(e)}")


class SingleStoreSQLDatabaseChain:
    """Helper class for building SQL database chains with SingleStore.

    This is a utility class that provides convenient methods for creating
    SQL query execution chains that can be used with LangChain agents.

    Example:
        .. code-block:: python

            from langchain_singlestore import SingleStoreSQLDatabaseChain
            from langchain_openai import ChatOpenAI

            chain = SingleStoreSQLDatabaseChain.from_url(
                host="root:password@localhost:3306/my_db",
                llm=ChatOpenAI()
            )

            result = chain.run("How many users are in the database?")
            print(result)
    """

    @staticmethod
    def from_url(
        host: str,
        llm: Any,
        **kwargs: Any,
    ) -> Any:
        """Create a SQL database chain from connection URL.

        Args:
            host: Database connection URL or host string.
            llm: Language model to use for query generation.
            **kwargs: Additional arguments for chain configuration.

        Returns:
            Configured SQL database chain ready for use.

        Note:
            Requires langchain sql-agent or similar integration.
        """
        retriever = SingleStoreSQLDatabaseRetriever(host=host, **kwargs)
        return retriever

    @staticmethod
    def query_to_document(
        query: str,
        host: str,
        row_limit: Optional[int] = 100,
        **connection_kwargs: Any,
    ) -> List[Document]:
        """Execute a query and return results as documents.

        Convenience method for one-off query execution.

        Args:
            query: SQL query to execute.
            host: Database host/URL.
            row_limit: Maximum number of rows to return. Defaults to 100.
            **connection_kwargs: Additional database connection arguments.

        Returns:
            List[Document]: Query results as documents.
        """
        if row_limit:
            query = f"{query} LIMIT {row_limit}"

        retriever = SingleStoreSQLDatabaseRetriever(host=host, **connection_kwargs)
        try:
            return retriever.invoke(query)
        finally:
            retriever.close()
