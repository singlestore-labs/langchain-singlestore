# singlestore-langchain-core

Shared, internal helpers used by the SingleStore integrations for LangChain
and LangGraph:

- Connection pool helpers (`create_connection_pool`, `SingleConnectionPool`,
  `QueueConnectionPool`).
- Connection attribute helpers (`set_connector_attributes`,
  `compute_connector_version`).
- SingleStore capability enums (`DistanceStrategy`, `FullTextIndexVersion`,
  `FullTextScoringMode`).
- Metadata filter DSL (`FilterTypedDict`, `_parse_filter`) shared by vector
  stores and stores.

This package is not intended to be depended on directly by application code.
Install one of the higher-level packages instead:

- [`langchain-singlestore`](../langchain-singlestore)
- [`langgraph-singlestore`](../langgraph-singlestore)

## Connections and pools

Every integration in this repo talks to SingleStore through an SQLAlchemy
[`Pool`](https://docs.sqlalchemy.org/en/20/core/pooling.html) object. The
factory [`create_connection_pool`](singlestore_langchain_core/_connection.py)
picks the right implementation from the arguments you pass. Integrations
forward the same kwargs to it, so the rules below apply everywhere a
SingleStore integration accepts a `connection`, `connection_pool`, or the
usual `host`/`user`/`password`/... parameters.

```python
from singlestore_langchain_core import create_connection_pool
```

### Dispatch rules

`create_connection_pool` picks one of three code paths, in this order:

1. **Both `connection` and `connection_pool` given** — raises `ValueError`.
   Pick one.
2. **`connection` given** — wraps the caller-owned connection in a
   `SingleConnectionPool`. Every checkout returns a proxy over the same
   connection; the pool never opens or closes it.
3. **`connection_pool` given** — returned unchanged. Bring your own pool.
4. **Neither given** — builds a `QueueConnectionPool` from `pool_size`,
   `max_overflow`, `timeout` and `connection_kwargs`. New connections are
   opened lazily on checkout via `singlestoredb.connect(**connection_kwargs)`.

### `SingleConnectionPool` — reuse one caller-owned connection

Use this when you already have a live `singlestoredb` connection and want
every operation to share it (tests, notebooks, or a single long-lived
connection managed by the surrounding application).

```python
import singlestoredb
from singlestore_langchain_core import create_connection_pool

conn = singlestoredb.connect(host="localhost", user="root", database="app")
pool = create_connection_pool(connection=conn)

checkout = pool.connect()
try:
    with checkout.cursor() as cur:
        cur.execute("SELECT 1")
finally:
    # No-op: the pool never closes the caller-owned connection.
    checkout.close()

conn.close()  # you own the lifecycle
```

Notes:

- `pool.connect()` returns a lightweight proxy, not the raw connection.
  Attribute access (`cursor()`, `commit()`, …) is forwarded; `close()` is a
  no-op so the standard `connect()`/`close()` idiom used elsewhere in the
  codebase does not tear down your connection.
- `pool.dispose()` is a no-op for the same reason — the caller owns the
  underlying connection.

### `QueueConnectionPool` — the default, size-bounded pool

Chosen automatically when you don't pass `connection` or `connection_pool`.
It wraps `sqlalchemy.pool.QueuePool` and lazily opens
`singlestoredb.connect(**connection_kwargs)` connections on checkout.

```python
from singlestore_langchain_core import create_connection_pool

pool = create_connection_pool(
    pool_size=5,
    max_overflow=10,
    timeout=30,
    connection_kwargs={
        "host": "localhost",
        "user": "root",
        "password": "...",
        "database": "app",
    },
)

conn = pool.connect()
try:
    with conn.cursor() as cur:
        cur.execute("SELECT 1")
finally:
    conn.close()  # returns the connection to the pool

pool.dispose()  # releases idle connections
```

Parameters:

| Name                | Default | Description                                                                             |
| ------------------- | ------- | --------------------------------------------------------------------------------------- |
| `pool_size`         | `5`     | Persistent connections kept in the pool.                                                |
| `max_overflow`      | `10`    | Extra connections that may be opened beyond `pool_size` under load.                     |
| `timeout`           | `30`    | Seconds to wait for a free connection before raising.                                   |
| `connection_kwargs` | `{}`    | Forwarded verbatim to `singlestoredb.connect` for each new connection. Defensively copied on construction. |

See the [singlestoredb docs](https://singlestoredb-python.labs.singlestore.com/api.html#singlestoredb.connect)
for the full list of connection kwargs (TLS options, credential type,
`autocommit`, `results_type`, etc.).

### Connection parameters from environment variables

`singlestoredb.connect()` reads its parameters from environment variables when
the corresponding kwargs are missing, so `connection_kwargs` may be omitted
entirely (or set to `None`) and the pool will still open working connections.
The variables recognised by `singlestoredb` include:

| Variable                 | Equivalent kwarg                                                         |
| ------------------------ | ------------------------------------------------------------------------ |
| `SINGLESTOREDB_URL`      | Full connection URL: `scheme://user:password@host:port/database`         |
| `SINGLESTOREDB_HOST`     | `host`                                                                   |
| `SINGLESTOREDB_PORT`     | `port`                                                                   |
| `SINGLESTOREDB_USER`     | `user`                                                                   |
| `SINGLESTOREDB_PASSWORD` | `password`                                                               |
| `SINGLESTOREDB_DATABASE` | `database`                                                               |

See the upstream
[singlestoredb PyPI page](https://pypi.org/project/singlestoredb/) and the
[`singlestoredb.connect` reference](https://singlestoredb-python.labs.singlestore.com/generated/singlestoredb.connect.html)
for the authoritative list.

```python
# With SINGLESTOREDB_URL=me:p455w0rd@s2-host.com/my_db in the environment:
from singlestore_langchain_core import create_connection_pool

pool = create_connection_pool()  # no connection_kwargs needed
```

Explicit kwargs passed to `create_connection_pool` (or to the higher-level
integrations) always win over environment variables.

### Bring your own pool

Pass any object that satisfies the SQLAlchemy `Pool` contract — a raw
`QueuePool`, a `StaticPool`, or a custom subclass. The factory returns it
unchanged so integrations pick it up as-is:

```python
from sqlalchemy.pool import StaticPool
from singlestore_langchain_core import create_connection_pool

my_pool = StaticPool(creator=lambda: singlestoredb.connect(...))
pool = create_connection_pool(connection_pool=my_pool)
assert pool is my_pool
```

