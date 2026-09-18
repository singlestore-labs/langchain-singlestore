# langgraph-singlestore

SingleStore-backed persistence for [LangGraph](https://github.com/langchain-ai/langgraph):
graph checkpointing (`BaseCheckpointSaver`) and long-term memory
(`BaseStore`).

## Status

All public classes are re-exported from the top-level
`langgraph_singlestore` package for convenience:

```python
from langgraph_singlestore import (
    AsyncSingleStoreSaver,
    AsyncSingleStoreStore,
    SingleStoreIndexConfig,
    SingleStoreSaver,
    SingleStoreStore,
)
```

| Component | Status |
| --- | --- |
| `langgraph_singlestore.SingleStoreSaver` | Implemented — `get_tuple`, `list`, `put`, `put_writes`, migrations. |
| `langgraph_singlestore.AsyncSingleStoreSaver` | Implemented — async wrapper over the sync saver via the default executor. |
| `langgraph_singlestore.SingleStoreStore` | Implemented — CRUD, TTL sweeper, namespace listing, optional vector search. |
| `langgraph_singlestore.AsyncSingleStoreStore` | Implemented — async wrapper over the sync store via the default executor. |

## Installation

```bash
pip install langgraph-singlestore
```

## Store quickstart

```python
from langgraph_singlestore import SingleStoreStore

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
from langgraph_singlestore import SingleStoreStore
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
from langgraph_singlestore import AsyncSingleStoreStore

store = AsyncSingleStoreStore(**conn_kwargs)
await store.asetup()
await store.aput(("users", "u1"), key="profile", value={"name": "Ada"})
```

## Checkpoint saver quickstart

`SingleStoreSaver` persists LangGraph checkpoints, pending writes, and
blobs to SingleStore. It implements the full `BaseCheckpointSaver`
interface (`get_tuple`, `list`, `put`, `put_writes`) and their async
counterparts (`aget_tuple`, `alist`, `aput`, `aput_writes`).

```python
from langgraph_singlestore import SingleStoreSaver

saver = SingleStoreSaver(
    host="127.0.0.1",
    port=3306,
    user="root",
    password="",
    database="langgraph",
)
saver.setup()  # idempotent; applies pending migrations

graph = builder.compile(checkpointer=saver)
graph.invoke({"input": "hi"}, config={"configurable": {"thread_id": "t1"}})

saver.close()
```

Or build one from a connection URL:

```python
saver = SingleStoreSaver.from_conn_string(
    "user:password@127.0.0.1:3306/langgraph"
)
saver.setup()
```

Like `SingleStoreStore`, the saver accepts any one of:

- an existing `singlestoredb.Connection` via `connection=...`,
- an existing SQLAlchemy `Pool` via `connection_pool=...`, or
- plain connection kwargs (`host`, `user`, ...) — an internal
  `QueueConnectionPool` is built lazily from `pool_size`, `max_overflow`,
  and `timeout`.

`connection` and `connection_pool` are mutually exclusive. When you pass
your own `connection`, the saver never closes it — lifecycle stays with
the caller.

### Async

`AsyncSingleStoreSaver` shares the sync class’s constructor and state.
Every `a*` method dispatches to the default executor, so the same
SingleStore connection pool serves both sync and async callers.

```python
from langgraph_singlestore import AsyncSingleStoreSaver

saver = AsyncSingleStoreSaver.from_conn_string(
    "user:password@127.0.0.1:3306/langgraph"
)
await saver.asetup()
try:
    graph = builder.compile(checkpointer=saver)
    await graph.ainvoke(
        {"input": "hi"},
        config={"configurable": {"thread_id": "t1"}},
    )
finally:
    await saver.aclose()
```

The async saver adds `asetup()` and `aclose()` helpers for non-blocking
lifecycle management; all read/write methods are inherited from
`SingleStoreSaver`.

## Layout

The top-level `langgraph_singlestore` package re-exports every public
class. Under the hood, implementations live in
`langgraph_singlestore.checkpoint` (the checkpoint saver) and
`langgraph_singlestore.store` (the long-term memory store). Shared
connection, filter, and index helpers live in
[`singlestore-langchain-core`](../singlestore-langchain-core).

## Development

```bash
make lint
make test                # unit tests, sockets disabled
make integration_tests   # boots a SingleStore container via singlestoredb
```
