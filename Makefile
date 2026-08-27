.PHONY: all help \
	fmt-core fmt-langchain fmt-langgraph fmt \
	lint-core lint-langchain lint-langgraph lint \
	test-core test-langchain test-langgraph test tests \
	integration-langchain integration_tests \
	install

CORE_DIR := libs/singlestore-langchain-core
LANGCHAIN_DIR := libs/langchain-singlestore
LANGGRAPH_DIR := libs/langgraph-singlestore
PACKAGES := $(CORE_DIR) $(LANGCHAIN_DIR) $(LANGGRAPH_DIR)

all: help

# ---------- install ----------
install:
	@for pkg in $(PACKAGES); do \
		echo "== poetry install ($$pkg) =="; \
		( cd $$pkg && poetry install --with lint,typing,test ) || exit $$?; \
	done

# ---------- formatting ----------
fmt: fmt-core fmt-langchain fmt-langgraph
fmt-core:
	$(MAKE) -C $(CORE_DIR) format
fmt-langchain:
	$(MAKE) -C $(LANGCHAIN_DIR) format
fmt-langgraph:
	$(MAKE) -C $(LANGGRAPH_DIR) format

# ---------- linting ----------
lint: lint-core lint-langchain lint-langgraph
lint-core:
	$(MAKE) -C $(CORE_DIR) lint
lint-langchain:
	$(MAKE) -C $(LANGCHAIN_DIR) lint
lint-langgraph:
	$(MAKE) -C $(LANGGRAPH_DIR) lint

# ---------- unit tests ----------
test tests: test-core test-langchain test-langgraph
test-core:
	$(MAKE) -C $(CORE_DIR) test
test-langchain:
	$(MAKE) -C $(LANGCHAIN_DIR) test
test-langgraph:
	$(MAKE) -C $(LANGGRAPH_DIR) test

# ---------- integration tests ----------
integration_tests: integration-langchain
integration-langchain:
	$(MAKE) -C $(LANGCHAIN_DIR) integration_tests

help:
	@echo 'Monorepo targets:'
	@echo '  install               - poetry install every package under libs/'
	@echo '  fmt                   - format every package'
	@echo '  lint                  - lint every package'
	@echo '  test                  - run unit tests for every package'
	@echo '  integration_tests     - run integration tests (langchain-singlestore)'
	@echo ''
	@echo 'Per-package targets: fmt-{core,langchain,langgraph}, lint-{...}, test-{...}'
	@echo 'Or cd libs/<pkg> and use its own Makefile directly.'
