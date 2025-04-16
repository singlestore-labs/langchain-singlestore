# Contribution Guidelines

Thank you for your interest in contributing to the `langchain-singlestore` project! This document outlines the guidelines for contributing to the repository.

## General Guidelines

- All contributions should adhere to the coding standards and best practices outlined in this repository.
- Each commit to the `main` branch must pass linting and integration tests. A GitHub Action is configured to automatically validate this for every pull request and commit.

## Pre-Commit Checklist

Before committing your changes, ensure the following:

1. **Linting**: Run the linter to check for style and formatting issues.
   ```bash
   make lint
   ```

2. **Integration Tests**: Run all integration tests to verify that your changes do not break existing functionality.
   ```bash
   make integration_tests
   ```

3. **Documentation**: Update or add documentation if your changes introduce new features or modify existing ones.

## GitHub Actions

The repository includes a GitHub Action that automatically runs linting and integration tests for every pull request and commit to the `main` branch. Ensure your changes pass these checks before submitting a pull request.

---

## Publish a New Package

Follow these steps to publish a new version of the `langchain-singlestore` package to PyPI:

1. **Update the Version in `pyproject.toml`**:
   - Open the `pyproject.toml` file.
   - Update the `version` attribute in the `[tool.poetry]` section to the new version.

2. **Update the `_connector_version`**:
   - Search the codebase for `_connector_version`.
   - Update the version in the connection attributes if necessary. The `_connector_version` should match the package version +2 (e.g., package version `1.0.0` corresponds to `_connector_version` `3.0.0`).

3. **Update Documentation**:
   - Ensure the `README.md` file and any other relevant documentation are updated to reflect the changes in the new version.

4. **Commit and Merge Changes**:
   - Ensure the `main` branch is in a healthy state, passing all linting and unit tests.
   - Commit your changes and merge them into the `main` branch.

5. **Create and Push a Version Tag**:
   - Create a new tag for the version:
     ```bash
     git tag v<new_version>
     ```
   - Push the tag to the remote repository:
     ```bash
     git push origin v<new_version>
     ```

Pushing the tag will trigger a GitHub Action that automatically builds and publishes the package to PyPI.

---

By following these guidelines, you help maintain the quality and reliability of the `langchain-singlestore` project. Thank you for contributing!
