# langgraph-singlestore

SingleStore-backed persistence for [LangGraph](https://github.com/langchain-ai/langgraph):
long-term memory (`BaseStore`) and, in progress, graph checkpointing
(`BaseCheckpointSaver`).

## Status

| Component | Status |
| --- | --- |
| `langgraph.store.singlestore.SingleStoreStore` | Implemented — CRUD, TTL sweeper, namespace listing, optional vector search. |
| `langgraph.store.singlestore.AsyncSingleStoreStore` | Implemented — async wrapper over the sync store via the default executor. |
| `langgraph.checkpoint.singlestore.SingleStoreSaver` | Scaffolding — raises `NotImplementedError`. |
| `langgraph.checkpoint.singlestore.AsyncSingleStoreSaver` | Scaffolding — raises `NotImplementedError`. |

## Installation

```bash
pip install langgraph-singlestore
```

## Store quickstart

```python
from langgraph.store.singlestore import SingleStoreStore

store = SingleStoreStore(
    host="127.0.0.1",
    port=3306,
    user="root",
    password="",
    database="langgraph",
)
store.setup()  # idempotent; applies pending migrations

store.put(("users", "u1"), key="profile", value={"name": "Ada"})
item = store.get(("users", "u1"), key="profile")
```

Callers may pass any one of:

- an existing `singlestoredb.Connection` via `connection=...`,
- an existing SQLAlchemy `Pool` via `connection_pool=...`, or
- plain connection kwargs (`host`, `user`, ...) — an internal
  `QueueConnectionPool` is built lazily.

`connection` and `connection_pool` are mutually exclusive.

### TTL

Pass a `TTLConfig` and start the background sweeper to expire items:

```python
from langgraph.store.base import TTLConfig

store = SingleStoreStore(ttl_config=TTLConfig(...), **conn_kwargs)
store.setup()
store.start_ttl_sweeper()
# ...
store.stop_ttl_sweeper()
```

### Vector search

Pass an `index` config with `dims`, `embed`, and an `ann_index_config`
to enable semantic `search()`:

```python
from langgraph.store.singlestore import SingleStoreStore
from singlestore_langchain_core import ANNIndexConfig

store = SingleStoreStore(
    index={
        "dims": 1536,
        "embed": my_embeddings,
        "ann_index_config": ANNIndexConfig(...),
    },
    **conn_kwargs,
)
store.setup()
results = store.search(("docs",), query="what is singlestore?")
```

### Async

`AsyncSingleStoreStore` exposes the same constructor and shares the sync
class's state; async methods delegate to the sync implementation via the
default executor.

```python
from langgraph.store.singlestore import AsyncSingleStoreStore

store = AsyncSingleStoreStore(**conn_kwargs)
await store.asetup()
await store.aput(("users", "u1"), key="profile", value={"name": "Ada"})
```

## Checkpoint saver

`SingleStoreSaver` / `AsyncSingleStoreSaver` are placeholders today: the
constructors accept configuration but every I/O method raises
`NotImplementedError`. Track progress in [CHANGELOG.md](CHANGELOG.md).

## Layout

This package uses PEP 420 namespace packages under `langgraph.checkpoint.*`
and `langgraph.store.*`, matching the layout used by
`langgraph-checkpoint-postgres` and friends. Shared connection, filter, and
index helpers live in
[`singlestore-langchain-core`](../singlestore-langchain-core).

## Development

```bash
make lint
make test                # unit tests, sockets disabled
make integration_tests   # boots a SingleStore container via singlestoredb
```
