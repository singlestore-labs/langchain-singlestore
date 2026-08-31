.PHONY: all format lint test tests integration_test integration_tests help

all: help

TEST_FILE ?= tests/unit_tests/
integration_test integration_tests: TEST_FILE = tests/integration_tests/

# unit tests are run with --disable-socket to prevent network calls
test tests:
	poetry run pytest --disable-socket --allow-unix-socket $(TEST_FILE)

# integration tests need network access to talk to the SingleStore docker container
integration_test integration_tests:
	poetry run pytest $(TEST_FILE)

PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_package: PYTHON_FILES=singlestore_langchain_core
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_package lint_tests:
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && poetry run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format:
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff check --select I --fix $(PYTHON_FILES)

help:
	@echo '----'
	@echo 'format              - run code formatters'
	@echo 'lint                - run linters'
	@echo 'test                - run unit tests'
	@echo 'integration_test    - run integration tests (requires docker)'
