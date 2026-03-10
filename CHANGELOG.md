# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-03-10

### Added

#### SQL Database Retriever (`SingleStoreSQLDatabaseRetriever`)
- **New component** that enables executing SQL queries directly against SingleStore and retrieving results as LangChain Document objects
- Full integration with LangChain's `BaseRetriever` interface for seamless compatibility with agents and chains
- Flexible row-to-document conversion system:
  - Default row-to-document converter that formats all row fields as document content and metadata
  - Support for custom `row_to_document_fn` parameter to implement specialized document formatting
  - Example custom converters demonstrating various formatting strategies
- Connection pooling with SQLAlchemy:
  - Configurable pool size (default: 5), max overflow (default: 10), and timeout (default: 30s)
  - Automatic connection lifecycle management
  - Proper cleanup through `close()` method and `__del__` destructor

#### SQL Database Chain (`SingleStoreSQLDatabaseChain`)
- **Convenience class** providing utility methods for common database query operations:
  - `from_url()` static method for creating retriever instances from connection URLs
  - `query_to_document()` static method for executing one-off queries without managing retriever lifecycle
  - Built-in LIMIT clause application for result set safety

#### Comprehensive Testing Suite
- **Unit Tests** (`tests/unit_tests/test_sql_database_retriever.py`):
  - 15+ test cases covering initialization, query execution, and connection management
  - Tests for custom row conversion functions
  - Connection pool lifecycle management tests
  - Error handling and edge case scenarios
- **Integration Tests** (`tests/integration_tests/test_sql_database_retriever.py`):
  - Real database scenario tests with table creation and data insertion
  - JSON data handling in query results
  - Multi-row result processing
  - Query limiting and empty result handling
  - Agent integration examples

#### Documentation
- **Jupyter Notebook** (`docs/sql_database_retriever.ipynb`):
  - Complete interactive examples demonstrating all features
  - Setup and initialization guide
  - Basic usage examples with query execution and result inspection
  - Custom row converter implementation patterns
  - Convenience method usage (`from_url()` and `query_to_document()`)
  - Advanced features including JSON handling, connection pool configuration
  - LangChain agent integration examples
  - Complete end-to-end examples combining multiple features

- **Implementation Documentation** (`docs/SQL_DATABASE_RETRIEVER_IMPLEMENTATION.md`):
  - Detailed implementation overview
  - Complete API reference
  - Usage examples and patterns
  - Architecture and design decisions

#### Examples
- **Examples Script** (`examples/examples_sql_database_retriever.py`):
  - 7 comprehensive example functions:
    1. Basic usage showing simple query execution
    2. Custom row-to-document converter patterns
    3. Convenience methods (`SingleStoreSQLDatabaseChain`)
    4. LangChain agent integration patterns
    5. JSON data handling examples
    6. Error handling best practices
    7. Connection pool configuration for various scenarios

#### Code Quality Improvements
- Full type annotations across all code:
  - Function parameters and return types fully annotated
  - Type hints for all public methods and fixtures
  - Proper handling of Optional types for nullable attributes
  - Type guards for None checks before method calls
- Ruff linter compliance:
  - All code passes Ruff format and import checks
  - Clean code following PEP 8 standards
- Mypy type checking:
  - Full type safety with no skipped checks
  - Proper error handling for union types
  - None checks for optional attributes before access

### Technical Details

#### New Classes
- `SingleStoreSQLDatabaseRetriever(BaseRetriever)`: Main retriever class for SQL database queries
- `SingleStoreSQLDatabaseChain`: Utility class with static convenience methods

#### Key Methods
- `invoke(query: str) -> List[Document]`: Execute query synchronously
- `ainvoke(query: str) -> List[Document]`: Execute query asynchronously
- `_execute_query(query: str) -> List[dict]`: Low-level query execution returning raw dicts
- `_get_relevant_documents(query: str, run_manager) -> List[Document]`: Internal retriever method
- `close() -> None`: Close connection pool explicitly
- `SingleStoreSQLDatabaseChain.from_url(host, llm, **kwargs) -> SingleStoreSQLDatabaseRetriever`: Create from URL
- `SingleStoreSQLDatabaseChain.query_to_document(query, host, row_limit, **kwargs) -> List[Document]`: One-off query

#### Connection Parameters
- Direct parameters: `host`, `user`, `password`, `database`
- Pool configuration: `pool_size`, `max_overflow`, `timeout`
- Custom conversion: `row_to_document_fn`
- Additional singlestoredb parameters: `pure_python`, `local_infile`, `charset`, SSL options, etc.

### Dependencies
- **New**: SQLAlchemy `^2.0.40` for connection pooling
  - Note: singlestoredb was already a dependency

### Export Updates
- Added `SingleStoreSQLDatabaseRetriever` to `langchain_singlestore.__init__.py`
- Added `SingleStoreSQLDatabaseChain` to `langchain_singlestore.__init__.py`

### Breaking Changes
- None

### Deprecated
- None

### Fixed
- Type annotation compliance across integration tests
- Proper None checks for optional connection pool attributes
- Long line formatting in function signatures for PEP 8 compliance

### Known Limitations
- Async implementation (`_aget_relevant_documents`) uses sync execution under the hood; true async support can be added in future with async-compatible drivers
- Maximum result set size depends on available memory (no built-in pagination for single query)
- Row-to-document conversion is applied in-memory after query execution

## [1.2.0] - Previous Release

For changes in previous versions, see the git history.
