# Contribution Guidelines

Thank you for your interest in contributing to the `langchain-singlestore` project! This document outlines the guidelines for contributing to the repository.

## Table of Contents

- [General Guidelines](#general-guidelines)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Code Standards](#code-standards)
- [Pre-Commit Checklist](#pre-commit-checklist)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Standards](#documentation-standards)
- [GitHub Actions](#github-actions)
- [Creating Pull Requests](#creating-pull-requests)
- [Publishing a New Package](#publish-a-new-package)
- [Troubleshooting](#troubleshooting)

## General Guidelines

- All contributions should adhere to the coding standards and best practices outlined in this repository.
- Each commit to the `main` branch must pass linting and integration tests. A GitHub Action is configured to automatically validate this for every pull request and commit.
- Write clear, descriptive commit messages explaining the purpose of your changes.
- Keep pull requests focused on a single feature or bug fix.

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Poetry (Python dependency manager)
- Docker (for running SingleStore database)

### Local Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/singlestore-labs/langchain-singlestore.git
   cd langchain-singlestore
   ```

2. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install dependencies**:
   ```bash
   poetry install --with lint,typing,test,test_integration
   ```

4. **Set up SingleStore database** (for integration tests):
   ```bash
   docker run -d --name singlestore-dev \
     -e ROOT_PASSWORD="pass" \
     -p 3306:3306 \
     ghcr.io/singlestore-labs/singlestoredb-dev:latest
   
   # Create the test database
   docker exec singlestore-dev mysql -u root -ppass -e "CREATE DATABASE IF NOT EXISTS db;"
   ```

5. **Verify setup**:
   ```bash
   make tests  # Run unit tests
   ```

## Project Structure

```
langchain-singlestore/
├── langchain_singlestore/          # Main package
│   ├── __init__.py                 # Package exports
│   ├── _filter.py                  # Advanced metadata filtering (FilterTypedDict)
│   ├── _utils.py                   # Utility functions and enums
│   ├── cache.py                    # Semantic cache implementation
│   ├── chat_message_history.py     # Chat message persistence
│   ├── document_loaders.py         # Document loading from database
│   ├── embeddings.py               # Embedding utilities
│   └── vectorstores.py             # Vector store implementation
├── tests/
│   ├── unit_tests/                 # Unit tests (no database required)
│   └── integration_tests/          # Integration tests (requires database)
├── docs/                           # Jupyter notebook examples
├── scripts/                        # Utility scripts for development
├── pyproject.toml                  # Project configuration and dependencies
├── Makefile                        # Development commands
└── README.md                       # Project documentation
```

## Code Standards

### Type Hints

- All functions and methods should have type hints for parameters and return types.
- Use `Optional[T]` for optional parameters, not `None` as default.
- Use `Union[T1, T2]` for multiple possible types.

Example:
```python
def similarity_search(
    self,
    query: str,
    k: int = 4,
    filter: Optional[Union[dict, FilterTypedDict]] = None,
) -> List[Document]:
    """Implementation with proper type hints."""
    pass
```

### Code Style

- Follow PEP 8 conventions.
- Use descriptive variable and function names.
- Maximum line length: 88 characters (enforced by ruff).
- Use f-strings for string formatting.

### Import Organization

```python
# Standard library imports
from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

# Third-party imports
import singlestoredb as s2
from langchain_core.documents import Document

# Local imports
from langchain_singlestore._utils import DistanceStrategy
```

### Docstring Format

Use Google-style docstrings:

```python
def add_documents(
    self,
    texts: Iterable[str],
    metadatas: Optional[List[dict]] = None,
) -> List[str]:
    """Add documents to the vector store.
    
    Args:
        texts: Iterable of document texts to add.
        metadatas: Optional list of metadata dictionaries.
        
    Returns:
        List of document IDs that were added.
        
    Raises:
        ValueError: If inputs are invalid.
        
    Examples:
        >>> vs = SingleStoreVectorStore(embeddings)
        >>> ids = vs.add_documents(["text1", "text2"])
    """
    pass
```

## Pre-Commit Checklist

Before committing your changes, ensure the following:

1. **Code Quality (Linting)**:
   ```bash
   make lint
   ```
   This runs:
   - `ruff check` - Python linting and import sorting
   - `ruff format --diff` - Code formatting check
   - `mypy` - Static type checking

2. **Unit Tests** (no database required):
   ```bash
   make tests
   ```
   Verify that your changes don't break existing functionality.

3. **Integration Tests** (requires SingleStore database):
   Make sure a SingleStore instance is running:
   ```bash
   make ci_tests
   ```
   Or run specific test:
   ```bash
   poetry run pytest tests/integration_tests/test_vectorstores.py -v
   ```

4. **Documentation**: 
   - Update docstrings if you modify existing functions
   - Update README.md if adding new features
   - Add examples if introducing new functionality

## Testing Guidelines

### Unit Tests

- Located in `tests/unit_tests/`
- **Do not require** a database connection
- Use `--disable-socket` flag to prevent network calls
- Test individual components in isolation
- Mock external dependencies

Example:
```python
def test_filter_typed_dict_eq_operator():
    """Test $eq operator for exact matching."""
    result = _parse_filter({"name": {"$eq": "value"}}, "metadata_field")
    assert "JSON_MATCH_ANY" in result[0]
```

### Integration Tests

- Located in `tests/integration_tests/`
- **Require** a running SingleStore database
- Test end-to-end workflows
- Test actual database operations
- Use fixtures for common setup

Example:
```python
def test_filter_typed_dict_gt_operator(
    self,
    vectorestore_random: SingleStoreVectorStore,
    numeric_docs: List[Document]
) -> None:
    """Test $gt operator for greater than numeric comparison."""
    vectorestore_random.add_documents(numeric_docs)
    output = vectorestore_random.similarity_search(
        "query",
        k=10,
        filter={"views": {"$gt": 150}},
    )
    assert len(output) == 2
```

### Test Naming Conventions

- Unit test files: `test_<module>.py`
- Test functions: `test_<feature_or_scenario>()`
- Descriptive names that explain what is being tested
- Use docstrings to explain complex test logic

## Documentation Standards

### README.md Updates

When adding new features:
1. Add usage examples
2. Explain configuration options
3. Include any special requirements
4. Update the features section

### Docstring Examples

Include practical examples in docstrings:

```python
def similarity_search(
    self,
    query: str,
    filter: Optional[Union[dict, FilterTypedDict]] = None,
) -> List[Document]:
    """Search for similar documents.
    
    Args:
        query: Search query text.
        filter: Optional metadata filter. Supports:
            - Simple dict: {"field": "value"}
            - FilterTypedDict: {"field": {"$gt": 100}}
    
    Examples:
        >>> # Simple filter
        >>> results = vs.similarity_search("query", filter={"status": "active"})
        
        >>> # Advanced filter with operators
        >>> results = vs.similarity_search(
        ...     "query",
        ...     filter={"$and": [{"views": {"$gt": 100}}, {"active": True}]}
        ... )
    """
    pass
```

## GitHub Actions

The repository includes a GitHub Action that automatically validates:

- **Linting**: Ruff code quality checks
- **Type Checking**: MyPy static type validation
- **Unit Tests**: All unit tests pass
- **Integration Tests**: All integration tests pass

Ensure your changes pass these checks before submitting a pull request. You can run the same checks locally with:

```bash
make lint
make tests
make ci_tests
```

## Creating Pull Requests

### PR Title Format

Use clear, descriptive titles:
- ✅ Good: `Add FilterTypedDict support for advanced metadata filtering`
- ❌ Bad: `Fix stuff` or `Update code`

### PR Description

Include:
1. **What**: Summary of changes
2. **Why**: Context and motivation
3. **How**: Approach taken
4. **Testing**: How to test the changes
5. **Related Issues**: Link to relevant issues (e.g., `Fixes #123`)

Template:
```markdown
## Description
Brief summary of changes

## Motivation
Why this change is needed

## Changes Made
- Specific implementation details
- Key modifications

## Testing
How to verify the changes work:
- Unit tests pass: `make tests`
- Integration tests pass: `make ci_tests`
- New tests added for: [feature]

## Related Issues
Fixes #123
```

### Review Process

1. All PRs require at least one approval
2. All CI checks must pass
3. Code should follow the style guidelines
4. Documentation should be updated
5. Tests should cover new functionality

---

## Publish a New Package

Follow these steps to publish a new version of the `langchain-singlestore` package to PyPI:

1. **Update the Version in `pyproject.toml`**:
   - Open the `pyproject.toml` file.
   - Update the `version` attribute in the `[tool.poetry]` section to the new version.
   - Follow [Semantic Versioning](https://semver.org/) (MAJOR.MINOR.PATCH).

2. **Update CHANGELOG** (if maintained):
   - Document all changes in the new version.
   - Include bug fixes, features, and breaking changes.

3. **Update Documentation**:
   - Ensure the `README.md` file is updated with new features.
   - Update docstrings and examples if behavior changed.

4. **Commit and Merge Changes**:
   - Ensure the `main` branch is in a healthy state, passing all linting and tests.
   - Create a pull request with title like: `Release v<version>`
   - Get approval and merge to `main`.

5. **Create and Push a Version Tag**:
   - Create a new tag for the version:
     ```bash
     git tag v<new_version>
     ```
   - Push the tag to the remote repository:
     ```bash
     git push origin v<new_version>
     ```

Pushing the tag triggers a GitHub Action that automatically builds and publishes the package to PyPI.

6. **Verify Release**:
   - Check [PyPI](https://pypi.org/project/langchain-singlestore/) for the new version.
   - Verify the updated documentation is available.

---

## Troubleshooting

### Common Issues

#### Database Connection Failed

**Issue**: Integration tests fail with "Connection refused"

**Solution**:
1. Ensure SingleStore is running:
   ```bash
   docker ps | grep singlestore
   ```
2. If not running, start the container:
   ```bash
   docker run -d --name singlestore-dev \
     -e ROOT_PASSWORD="pass" \
     -p 3306:3306 \
     ghcr.io/singlestore-labs/singlestoredb-dev:latest
   ```
3. Verify the database exists:
   ```bash
   docker exec singlestore-dev mysql -u root -ppass -e "SHOW DATABASES;"
   ```

#### Linting Fails with Import Errors

**Issue**: MyPy complains about missing types

**Solution**:
1. Ensure all imports have type hints
2. Check that FilterTypedDict is properly imported
3. Use `# type: ignore` comments only when necessary with explanation

#### Tests Pass Locally but Fail in CI

**Issue**: Tests work on local machine but fail in GitHub Actions

**Solution**:
1. Check Python version matches (3.11+)
2. Ensure all dependencies are installed: `poetry install --with lint,typing,test,test_integration`
3. Verify database availability in CI environment
4. Check for hard-coded paths or environment assumptions

### Getting Help

- Check existing issues and pull requests
- Review the [LangChain documentation](https://python.langchain.com/docs/)
- Check [SingleStore documentation](https://docs.singlestore.com/)
- Ask in pull request comments or create a discussion

---

By following these guidelines, you help maintain the quality and reliability of the `langchain-singlestore` project. Thank you for contributing!

