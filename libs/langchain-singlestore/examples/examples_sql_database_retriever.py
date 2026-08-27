"""
Example usage of SingleStore SQL Database Retriever

This script demonstrates various ways to use the SingleStoreSQLDatabaseRetriever
for executing SQL queries and retrieving results as LangChain Document objects.

The retriever can be used standalone, with LangChain agents, or as part of
larger AI applications that need database-aware functionality.
"""

import json
import logging

from langchain_core.documents import Document

from langchain_singlestore import (
    SingleStoreSQLDatabaseChain,
    SingleStoreSQLDatabaseRetriever,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def example_basic_usage() -> None:
    """Basic example: Initialize retriever and execute a simple query."""
    logger.info("\n=== Example 1: Basic Usage ===\n")

    # Initialize retriever with connection parameters
    retriever = SingleStoreSQLDatabaseRetriever(
        host="127.0.0.1:3306",
        user="root",
        password="your_password",
        database="my_database",
    )

    try:
        # Execute a query and get results as documents
        docs = retriever.invoke(input="SELECT 1 as test_value")

        logger.info(f"Retrieved {len(docs)} document(s)")
        for i, doc in enumerate(docs):
            logger.info(f"\nDocument {i}:")
            logger.info(f"Content:\n{doc.page_content}")
            logger.info(f"Metadata: {doc.metadata}")

    finally:
        retriever.close()


def example_with_custom_converter() -> None:
    """Example: Use a custom function to convert rows to documents."""
    logger.info("\n=== Example 2: Custom Row-to-Document Converter ===\n")

    def custom_converter(row_dict: dict, row_index: int) -> Document:
        """
        Custom converter function that formats rows in a specific way.

        Args:
            row_dict: Dictionary representation of the database row
            row_index: Index of the row in the result set

        Returns:
            Document: Formatted document with custom content and metadata
        """
        # Create a formatted string with all fields
        fields = ", ".join(f"{k}={v}" for k, v in row_dict.items())
        content = f"Record #{row_index}: {fields}"

        # Add custom metadata
        metadata = {
            "record_number": row_index,
            "source": "sql_database",
            "field_count": len(row_dict),
        }
        metadata.update(row_dict)

        return Document(page_content=content, metadata=metadata)

    retriever = SingleStoreSQLDatabaseRetriever(
        host="127.0.0.1:3306",
        user="root",
        password="your_password",
        database="my_database",
        row_to_document_fn=custom_converter,
    )

    try:
        # Execute a query using the custom converter
        docs = retriever.invoke(input="SELECT id, name, email FROM users LIMIT 5")

        logger.info(f"Retrieved {len(docs)} document(s) with custom format")
        for doc in docs:
            logger.info(f"\n{doc.page_content}")
            logger.info(f"Metadata: {doc.metadata}")

    finally:
        retriever.close()


def example_chain_convenience_method() -> None:
    """Example: Use the convenience method from SingleStoreSQLDatabaseChain."""
    logger.info("\n=== Example 3: Convenience Method (Query to Document) ===\n")

    # Execute query and automatically convert to documents
    docs = SingleStoreSQLDatabaseChain.query_to_document(
        query="SELECT COUNT(*) as total_users FROM users",
        host="127.0.0.1:3306",
        user="root",
        password="your_password",
        database="my_database",
        row_limit=100,  # Automatically adds LIMIT clause
    )

    logger.info(f"Retrieved {len(docs)} document(s)")
    for doc in docs:
        logger.info(f"\nContent: {doc.page_content}")
        logger.info(f"Metadata: {doc.metadata}")


def example_with_agent_integration() -> None:
    """Example: Integration with LangChain agents (pseudo-code)."""
    logger.info("\n=== Example 4: Agent Integration (Pseudo-code) ===\n")

    # This shows how you would integrate with LangChain agents
    example_code = '''
from langchain_singlestore import SingleStoreSQLDatabaseRetriever
from langchain.agents import Tool
from langchain_openai import ChatOpenAI

# Create retriever
retriever = SingleStoreSQLDatabaseRetriever(
    host="127.0.0.1:3306",
    user="root",
    password="your_password",
    database="my_database",
)

# Create a tool that executes database queries
def query_database(query: str) -> str:
    """Execute SQL query and return formatted results."""
    docs = retriever.invoke(input=query)
    return "\\n\\n".join([
        f"Row {doc.metadata.get('row_index', '')}: {doc.page_content}"
        for doc in docs
    ])

# Create LangChain tool
database_tool = Tool(
    name="query_database",
    func=query_database,
    description="Execute SQL queries against the SingleStore database"
)

# Use with agent (example with AgentType.ZERO_SHOT_REACT_DESCRIPTION)
from langchain.agents import initialize_agent

agent = initialize_agent(
    tools=[database_tool],
    llm=ChatOpenAI(model="gpt-4"),
    agent="zero-shot-react-description",
    verbose=True
)

# Run agent with natural language
result = agent.run(
    "How many customers are in the database? "
    "List their names and email addresses."
)
logger.info(result)
'''
    logger.info(example_code)


def example_with_json_handling() -> None:
    """Example: Handling JSON data in query results."""
    logger.info("\n=== Example 5: JSON Data Handling ===\n")

    def json_aware_converter(row_dict: dict, row_index: int) -> Document:
        """
        Converter that handles JSON fields specially.

        This is useful when your database contains JSON columns.
        """
        content_parts = []

        for key, value in row_dict.items():
            if isinstance(value, (dict, list)):
                # Format JSON data nicely
                formatted = json.dumps(value, indent=2)
                content_parts.append(f"{key} (JSON):\n{formatted}")
            else:
                content_parts.append(f"{key}: {value}")

        content = "\n".join(content_parts)
        metadata = {
            "row_index": row_index,
            "has_json": any(isinstance(v, (dict, list)) for v in row_dict.values()),
        }
        metadata.update(row_dict)

        return Document(page_content=content, metadata=metadata)

    retriever = SingleStoreSQLDatabaseRetriever(
        host="127.0.0.1:3306",
        user="root",
        password="your_password",
        database="my_database",
        row_to_document_fn=json_aware_converter,
    )

    try:
        # Query that returns JSON data
        docs = retriever.invoke(input="SELECT id, name, metadata FROM products LIMIT 3")

        logger.info(f"Retrieved {len(docs)} document(s) with JSON handling")
        for doc in docs:
            logger.info(f"\n--- Document ---\n{doc.page_content}")
            logger.info(f"\nMetadata: {doc.metadata}")

    finally:
        retriever.close()


def example_error_handling() -> None:
    """Example: Proper error handling."""
    logger.info("\n=== Example 6: Error Handling ===\n")

    retriever = SingleStoreSQLDatabaseRetriever(
        host="127.0.0.1:3306",
        user="root",
        password="your_password",
        database="my_database",
    )

    try:
        # Valid query
        docs = retriever.invoke(input="SELECT COUNT(*) as total FROM users")
        logger.info(f"✓ Query successful: {len(docs)} row(s) returned")

    except Exception as e:
        logger.info(f"✗ Error executing query: {type(e).__name__}: {str(e)}")

    try:
        # Invalid query - this should raise an error
        docs = retriever.invoke(input="SELECT * FROM non_existent_table")

    except Exception as e:
        logger.info(f"✓ Caught expected error: {type(e).__name__}")
        logger.info(f"  Error message: {str(e)[:100]}...")

    finally:
        retriever.close()


def example_connection_pool_configuration() -> None:
    """Example: Configure connection pool settings."""
    logger.info("\n=== Example 7: Connection Pool Configuration ===\n")

    # Configure pool for high-performance scenarios
    retriever = SingleStoreSQLDatabaseRetriever(
        host="127.0.0.1:3306",
        user="root",
        password="your_password",
        database="my_database",
        # Connection pool settings
        pool_size=10,  # Number of connections to maintain
        max_overflow=20,  # Additional connections allowed
        timeout=30,  # Connection timeout in seconds
    )

    logger.info("Retriever initialized with custom pool settings:")
    logger.info(f"  Pool size: {retriever.pool_size}")
    logger.info(f"  Max overflow: {retriever.max_overflow}")
    logger.info(f"  Timeout: {retriever.timeout}s")

    try:
        retriever.invoke(input="SELECT 1 as test")
        logger.info("\n✓ Query executed successfully with pooled connection")

    finally:
        retriever.close()


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("SingleStore SQL Database Retriever Examples")
    logger.info("=" * 70)

    # Note: These examples are pseudo-code and require a running SingleStore
    # database. Uncomment and modify with your actual connection details to run.

    logger.info("\nNote: These examples require a running SingleStore database.")
    logger.info("Modify the connection parameters with your actual database details.")

    # Uncomment to run specific examples:
    # example_basic_usage()
    # example_with_custom_converter()
    # example_chain_convenience_method()
    example_with_agent_integration()
    # example_with_json_handling()
    # example_error_handling()
    # example_connection_pool_configuration()

    logger.info("\n" + "=" * 70)
    logger.info("For more information, see the README.md documentation")
    logger.info("=" * 70)
