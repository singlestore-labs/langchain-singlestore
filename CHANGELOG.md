# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2026-04-07

### Added

#### Full-Text Index Version 2 Support
- **New `full_text_index_version` parameter** in `SingleStoreVectorStore` constructor
  - `V1` (default): Original full-text index implementation compatible with all SingleStore versions
  - `V2`: New full-text index implementation available in SingleStore 8.7+ with improved performance and additional features
- Automatic SQL syntax adaptation based on the selected full-text index version

#### Full-Text Scoring Modes
- **New `full_text_scoring_mode` parameter** in `similarity_search` and `similarity_search_with_score` methods
  - `MATCH` (default): Uses SingleStore's native MATCH() AGAINST() function, compatible with V1 and V2
  - `BM25`: Best Matching 25 algorithm with TF-IDF scoring and document length normalization (requires V2)
  - `BM25_GLOBAL`: BM25 with global IDF statistics across all partitions for consistent scoring in distributed environments (requires V2)
- New `FullTextScoringMode` enum exported from `langchain_singlestore`
- Validation to reject BM25/BM25_GLOBAL modes when using V1 full-text index

#### Enhanced Documentation
- Improved docstrings for `DistanceStrategy`, `FullTextIndexVersion`, and `FullTextScoringMode` enums with detailed attribute descriptions
- Updated README with comprehensive documentation for full-text index versions and scoring modes
- Added comparison tables for version features and scoring mode use cases
- Fixed inaccurate "cosine similarity" description in search methods (now correctly describes configurable distance strategy)

#### New Unit Tests
- `TestFulltextScoringModeToSql` test class covering all scoring mode and index version combinations
- Tests for custom content fields and table names with different configurations

### Changed

- Refactored full-text SQL generation into centralized `_fulltext_scoring_mode_to_sql` method
- Improved code examples in docstrings to use correct public imports from `langchain_singlestore`

### Fixed

- Fixed typo in `__init__` docstring example (`SingleStoreVectorStor` → `SingleStoreVectorStore`)
- Fixed missing comma in import statement in `similarity_search_with_score` docstring example
- Fixed missing blank line in code-block directive for full-text index example

### Dependencies

- Updated `aiohttp` from 3.13.3 to 3.13.4
- Updated `pygments` from 2.19.2 to 2.20.0
- Updated `langchain-core` from 1.2.15 to 1.2.22
- Updated `requests` from 2.32.5 to 2.33.0
- Updated `pyjwt` from 2.11.0 to 2.12.0

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
