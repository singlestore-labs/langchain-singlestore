# Unit Tests for langchain-singlestore

This directory contains comprehensive unit tests for the langchain-singlestore library. These tests do not require a real database connection and use mocking to isolate components.

## Test Files

### 1. `test_utils.py`
Tests for utility functions and constants in `langchain_singlestore._utils`.

**Coverage:**
- `CONNECTOR_NAME` constant validation
- `CONNECTOR_VERSION` format and auto-calculation
- `set_connector_attributes()` function
  - Sets connector attributes in empty kwargs
  - Preserves existing attributes
  - Creates `conn_attrs` dict if missing
  - In-place modification verification
- `DistanceStrategy` enum
  - Value validation
  - String enum behavior
  - Comparison operations
- `hash()` utility function
  - Returns valid hex string
  - Deterministic hashing
  - Handles empty strings
  - Handles special characters

**Test Count:** 18 tests

### 2. `test_chat_message_history.py`
Tests for `SingleStoreChatMessageHistory` class with mocked database connections.

**Coverage:**
- Initialization
  - Session ID setting and sanitization
  - Default and custom table names
  - Field name configuration
  - Connector attribute setting
  - Connection pool creation
- Message operations (with mocked database)
  - Table creation on first use
  - Message retrieval
  - Message insertion
  - Session clearing
- Connection parameter handling

**Test Count:** 19 tests

### 3. `test_vectorstores.py`
Tests for `SingleStoreVectorStore` class with mocked database connections.

**Coverage:**
- Initialization
  - Required parameters handling
  - Custom table and field names
  - Distance strategy configuration
  - Vector index enablement
  - Full-text search configuration
  - Connector attributes
  - Vector size configuration
  - Connection pool settings
- `_sanitize_input()` method
  - Special character removal
  - Alphanumeric and underscore preservation
- `SearchStrategy` enum
  - All strategies defined
  - Strategy values
- `embeddings` property

**Test Count:** 19 tests

### 4. `test_document_loaders.py`
Tests for `SingleStoreLoader` class with mocked database connections.

**Coverage:**
- Initialization
  - Default table and field names
  - Custom table and field names
  - Connector attribute setting
  - Connection pool configuration
  - Connection parameters
- `_sanitize_input()` method
- Document loading
  - Iterator from lazy_load()
  - List from load()
  - Document metadata handling

**Test Count:** 11 tests

### 5. `test_cache.py`
Tests for `SingleStoreSemanticCache` class with mocked vectorstore.

**Coverage:**
- Initialization
  - Default and custom cache table prefix
  - Search threshold configuration
  - Distance strategy setting
- VectorStore integration
  - Table naming with prefix
  - Vector index parameter passing
  - Vector size parameter passing
  - Host parameter passing
  - Custom connection parameters
- BaseCache inheritance validation

**Test Count:** 13 tests

## Running the Tests

### Using unittest (built-in)
```bash
# Run all tests
python -m unittest discover -s tests/unit_tests -p "test_*.py" -v

# Run specific test file
python -m unittest tests.unit_tests.test_utils -v

# Run specific test class
python -m unittest tests.unit_tests.test_utils.TestConnectorConstants -v

# Run specific test method
python -m unittest tests.unit_tests.test_utils.TestConnectorConstants.test_connector_name -v
```

### Using pytest (if installed)
```bash
# Run all tests
pytest tests/unit_tests/ -v

# Run with coverage
pytest tests/unit_tests/ --cov=langchain_singlestore --cov-report=html
```

## Test Design Principles

1. **No Database Required**: All tests use mocking via `unittest.mock` to avoid needing a real SingleStore database connection.

2. **Isolated Components**: Each test is independent and doesn't rely on other tests.

3. **Clear Naming**: Test method names clearly describe what is being tested.

4. **Comprehensive Coverage**: Tests cover:
   - Happy path scenarios
   - Edge cases
   - Parameter validation
   - Error conditions
   - Integration between components

5. **Mock Best Practices**: Uses `MagicMock` and `patch` decorators to mock external dependencies like database connections and connection pools.

## Mocking Strategy

### Database Connection Mocking
```python
@patch('langchain_singlestore.chat_message_history.QueuePool')
def setUp(self):
    self.mock_pool = MagicMock()
    self.mock_conn = MagicMock()
    self.mock_pool.connect.return_value = self.mock_conn
```

### Embeddings Mocking
Custom `MockEmbeddings` class extending `langchain_core.embeddings.Embeddings` provides consistent mock embeddings for vector store and cache tests.

## Total Test Count

- **Total Unit Tests: 80**
- **Lines of Test Code: ~900**
- **Coverage Target Modules:**
  - `_utils.py` ✓
  - `chat_message_history.py` ✓
  - `vectorstores.py` ✓
  - `document_loaders.py` ✓
  - `cache.py` ✓

## Future Enhancements

Potential areas for additional testing:
- Mock vectorstore operations (similarity_search, add_texts, delete)
- Mock cache operations (lookup, update, clear)
- Error handling and exception cases
- Concurrent operations
- Large-scale data handling
- Integration tests (when database is available)
