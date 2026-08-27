# SingleStore integrations for LangChain and LangGraph

This monorepo ships three Python packages:

| Package | Path | Purpose |
| --- | --- | --- |
| [`langchain-singlestore`](libs/langchain-singlestore) | `libs/langchain-singlestore/` | LangChain integrations: `SingleStoreVectorStore`, `SingleStoreSemanticCache`, `SingleStoreChatMessageHistory`, `SingleStoreLoader`, `SingleStoreSQLDatabaseRetriever` / `SingleStoreSQLDatabaseChain`. |
| [`langgraph-singlestore`](libs/langgraph-singlestore) | `libs/langgraph-singlestore/` | LangGraph integrations: `SingleStoreSaver` (checkpointing) and `SingleStoreStore` (long-term memory). *Scaffolding — implementations pending.* |
| [`singlestore-langchain-core`](libs/singlestore-langchain-core) | `libs/singlestore-langchain-core/` | Internal shared package: connection helpers, SingleStore enums, and the metadata filter DSL. Depended on by the two packages above. |

## Layout

```
libs/
├── singlestore-langchain-core/    # shared, no langchain/langgraph deps
├── langchain-singlestore/         # LangChain integration
└── langgraph-singlestore/         # LangGraph integration
```

Each package has its own `pyproject.toml`, `Makefile`, tests, and version.
Cross-package dependencies use Poetry path deps with `develop = true` during
development; the publish workflow rewrites them to version constraints before
building wheels.

## Development

```bash
# install every package into its own virtualenv
make install

# lint / test everything, or a single package
make lint
make test
make -C libs/langchain-singlestore lint
make -C libs/langchain-singlestore integration_tests
```

See each package's own README for user-facing documentation.

## Package status

- `langchain-singlestore`: stable, released on PyPI.
- `singlestore-langchain-core`: 0.x, internal.
- `langgraph-singlestore`: pre-release scaffolding.
