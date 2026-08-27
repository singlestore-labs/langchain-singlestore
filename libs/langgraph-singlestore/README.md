# langgraph-singlestore

SingleStore-backed persistence layer for [LangGraph](https://github.com/langchain-ai/langgraph).

> **Status:** scaffolding only. Implementations raise `NotImplementedError`.

## Planned components

- **`langgraph.checkpoint.singlestore.SingleStoreSaver`** — synchronous
  `BaseCheckpointSaver` backed by SingleStore. Persists graph state so runs
  can be resumed and time-travelled.
- **`langgraph.checkpoint.singlestore.AsyncSingleStoreSaver`** — the async
  variant.
- **`langgraph.store.singlestore.SingleStoreStore`** — `BaseStore`
  implementation with optional vector search over SingleStore.

## Installation

```bash
pip install langgraph-singlestore
```

## Layout

This package uses PEP 420 namespace packages under `langgraph.checkpoint.*`
and `langgraph.store.*`, matching the layout used by
`langgraph-checkpoint-postgres` and friends.
