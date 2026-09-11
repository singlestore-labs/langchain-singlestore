"""Synchronous SingleStore-backed :class:`~langgraph.store.base.BaseStore`.

Draft implementation modelled after ``langgraph.store.postgres.base``. Uses
``singlestoredb`` through the shared :func:`create_connection_pool` factory
from ``singlestore_langchain_core``, so callers may supply an existing
connection, an existing pool, or plain connection kwargs — identical
semantics to ``langchain-singlestore``.

Vector search (``index=...``) is intentionally out of scope for this draft;
it will be layered on top of ``SingleStoreVectorStore`` in a follow-up.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import threading
from collections import defaultdict
from typing import Any, Iterable, Literal, Optional, Sequence, cast

from singlestore_langchain_core import (
    LANGGRAPH_CONNECTOR_NAME,
    ANNIndexConfig,
    DistanceStrategy,
    FilterTypedDict,
    _parse_filter,
    compute_connector_version,
    create_connection_pool,
    set_connector_attributes,
)
from singlestoredb.connection import Connection
from sqlalchemy.pool import Pool

from langgraph.store.base import (
    BaseStore,
    Embeddings,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)

logger = logging.getLogger(__name__)

# Migrations mirror the layout of the Postgres store, translated to
# SingleStore SQL: ``JSON`` in place of ``jsonb``, inline ``INDEX`` clauses,
# and ``ON DUPLICATE KEY UPDATE`` in place of ``ON CONFLICT``.
MIGRATIONS: Sequence[str] = [
    """CREATE TABLE IF NOT EXISTS store_migrations (
        v INTEGER PRIMARY KEY
    );""",
    """CREATE TABLE IF NOT EXISTS store (
        prefix TEXT NOT NULL,
        `key` TEXT NOT NULL,
        value JSON NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP DEFAULT NULL,
        ttl_minutes INTEGER DEFAULT NULL,
        PRIMARY KEY (prefix(255), `key`(255)),
        INDEX store_prefix_idx (prefix(255)),
        INDEX expires_at_idx (expires_at)
    );""",
    # ``truncate_ns_prefix`` slices an escape-encoded prefix at the Nth
    # *unescaped* "/". Mirrors the Python escape scheme in ``_escape_ns_part``:
    # an escape char consumes the next character, so ``\/`` inside a part is
    # not counted as a boundary. Used by ``list_namespaces`` to push
    # ``max_depth`` truncation into SQL.
    r"""CREATE OR REPLACE FUNCTION truncate_ns_prefix(
        prefix TEXT,
        max_depth INT
    ) RETURNS TEXT AS
    DECLARE
        n INT = CHAR_LENGTH(prefix);
        i INT = 1;
        parts_seen INT = 0;
        ch TEXT;
    BEGIN
        IF prefix IS NULL OR max_depth <= 0 OR n = 0 THEN
            RETURN '';
        END IF;
        WHILE i <= n LOOP
            ch = SUBSTRING(prefix, i, 1);
            IF ch = '\\' AND i < n THEN
                i = i + 2;
            ELSEIF ch = '/' THEN
                parts_seen = parts_seen + 1;
                IF parts_seen = max_depth THEN
                    RETURN SUBSTRING(prefix, 1, i - 1);
                END IF;
                i = i + 1;
            ELSE
                i = i + 1;
            END IF;
        END LOOP;
        RETURN prefix;
    END;""",
]


# --- SQL fragments -----------------------------------------------------------
# ``JSON_EXTRACT_JSON`` returns a JSON value that compares directly to a JSON
# literal; ``JSON_EXTRACT_STRING`` returns the unquoted string form used for
# ordering/comparison of scalar fields.

_VECTOR_INDEX_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS store_vector (
        prefix TEXT NOT NULL,
        `key` TEXT NOT NULL,
        field_name TEXT NOT NULL,
        embedding VECTOR({}, F32) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (prefix(255), `key`(255), field_name(255)),
        INDEX store_vector_prefix_idx (prefix(255), `key`(255)),
        VECTOR INDEX store_vector_embedding_idx (embedding) INDEX_OPTIONS '{}'
    );"""

_UPSERT_BASE_SQL = """
    INSERT INTO store
    (prefix, `key`, value, created_at, updated_at, expires_at, ttl_minutes)
    VALUES """

_ON_DUPLICATE_KEY_UPDATE_SQL = """
    ON DUPLICATE KEY UPDATE
        value = VALUES(value),
        updated_at = CURRENT_TIMESTAMP,
        expires_at = VALUES(expires_at),
        ttl_minutes = VALUES(ttl_minutes)
"""

_UPSERT_BASE_VECTOR_SQL = """
    INSERT INTO store_vector
    (prefix, `key`, field_name, embedding, created_at, updated_at)
    VALUES """

_ON_DUPLICATE_KEY_VECTOR_UPDATE_SQL = """
    ON DUPLICATE KEY UPDATE
        embedding = VALUES(embedding),
        updated_at = CURRENT_TIMESTAMP
"""

_SELECT_BASE = """
    SELECT prefix, `key`, value, created_at, updated_at, expires_at, ttl_minutes
    FROM store WHERE (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP) AND
"""

_ORDER_BY_BASE = """
    ORDER BY updated_at DESC, prefix, `key` LIMIT %s OFFSET %s
"""

_SELECT_WITH_VECTOR_SEARCH_SQL = """
    SELECT store.prefix as prefix, store.`key` as `key`, store.value as value,
    store.created_at as created_at, store.updated_at as updated_at,
    store.expires_at as expires_at, store.ttl_minutes as ttl_minutes,
    {}({}(vector.embedding, JSON_ARRAY_PACK(%s))) as score
    FROM store_vector AS vector JOIN store ON
        vector.prefix = store.prefix AND vector.`key` = store.`key`
    WHERE (store.expires_at IS NULL OR store.expires_at > CURRENT_TIMESTAMP) AND
"""

_GROUP_BY_VECTOR_SEARCH_SQL = """
    GROUP BY store.prefix, store.`key`, store.value, store.created_at,
    store.updated_at, store.expires_at, store.ttl_minutes
"""

_ORDER_BY_VECTOR_SEARCH_SQL = """
    ORDER BY score {}, store.updated_at DESC, store.prefix,
    store.`key` LIMIT %s OFFSET %s
"""

_REFRESH_TTL_SQL_BASE = """
    UPDATE store
    SET expires_at = DATE_ADD(NOW(), INTERVAL ttl_minutes MINUTE),
        updated_at = CURRENT_TIMESTAMP
    WHERE (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP) AND
"""

_DELETE_EXPIRED_FROM_STORE = """
    DELETE FROM store
    WHERE expires_at IS NOT NULL AND expires_at <= NOW()
"""

_DELETE_EXPIRED_FROM_STORE_VECTOR = """
    DELETE FROM store_vector LEFT JOIN store
    ON store_vector.prefix = store.prefix
    AND store_vector.`key` = store.`key`
    WHERE store.prefix is NULL
"""

_DELETE_BASE_FROM_STORE = """
    DELETE FROM store
    WHERE prefix = %s AND `key` IN """


_DELETE_BASE_FROM_STORE_VECTOR_BASE = """
    DELETE FROM store_vector
    WHERE prefix = %s AND `key` IN """

_FLUSH_VECTOR_STORE_SQL = "OPTIMIZE TABLE store_vector FLUSH;"

AGGREGATE_FUNCTIONS_SQL: dict[DistanceStrategy, str] = {
    DistanceStrategy.DOT_PRODUCT: "MAX",
    DistanceStrategy.EUCLIDEAN_DISTANCE: "MIN",
}

SCORE_ORDER_DIRECTION: dict[DistanceStrategy, str] = {
    DistanceStrategy.DOT_PRODUCT: "DESC",
    DistanceStrategy.EUCLIDEAN_DISTANCE: "",
}


def _safe_rollback(cur: Any) -> None:
    # Best-effort ROLLBACK. Swallow driver errors so the original exception
    # from the caller isn't masked by a rollback failure.
    try:
        cur.execute("ROLLBACK")
    except Exception:
        logger.exception("ROLLBACK failed after transactional operation error")


class SingleStoreIndexConfig(IndexConfig):
    ann_index_config: ANNIndexConfig
    _tokenized_fields: list[tuple[str, Literal["$"] | list[str]]]
    _estimated_num_vectors: int


class SingleStoreStore(BaseStore):
    """SingleStore-backed store (synchronous).

    Callers may supply any one of the following:

    * ``connection`` — an existing :class:`singlestoredb.Connection`. The
      store never closes it.
    * ``connection_pool`` — an existing SQLAlchemy :class:`Pool`. The store
      never disposes it.
    * Connection kwargs (``host``, ``user``, ...) — a lazy
      :class:`QueueConnectionPool` is built internally.

    ``connection`` and ``connection_pool`` are mutually exclusive.

    Note:
        If you provide a TTL configuration, you must
        explicitly call `start_ttl_sweeper()` to begin
        the background thread that removes expired items.
        Call `stop_ttl_sweeper()` to properly clean up
        resources when you're done with the store.
    """

    supports_ttl: bool = True

    MIGRATIONS: Sequence[str] = MIGRATIONS

    def __init__(
        self,
        *,
        connection: Optional[Connection] = None,
        connection_pool: Optional[Pool] = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        timeout: float = 30,
        index: Optional[SingleStoreIndexConfig] = None,
        ttl_config: Optional[TTLConfig] = None,
        **connection_kwargs: Any,
    ) -> None:
        super().__init__()
        set_connector_attributes(
            connection_kwargs,
            connector_name=LANGGRAPH_CONNECTOR_NAME,
            connector_version=compute_connector_version("langgraph-singlestore"),
        )
        self.connection_kwargs = connection_kwargs
        self.connection_pool: Pool = create_connection_pool(
            connection=connection,
            connection_pool=connection_pool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            timeout=timeout,
            connection_kwargs=self.connection_kwargs,
        )
        self.index_config = index
        if self.index_config:
            self.embeddings, self.index_config = _ensure_index_config(self.index_config)
        else:
            self.embeddings = None
        self.ttl_config = ttl_config
        self._ttl_sweeper_thread: Optional[threading.Thread] = None
        self._ttl_sweeper_future: Optional[concurrent.futures.Future[None]] = None
        self._ttl_stop_event = threading.Event()
        # Serialise access to the underlying connection when the caller
        # shares a single connection across threads.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        """Run pending migrations. Idempotent; call once before first use."""
        with self._cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS store_migrations (v INTEGER PRIMARY KEY);"
            )
            cur.execute("SELECT v FROM store_migrations ORDER BY v DESC LIMIT 1")
            row = cur.fetchone()
            version = -1 if row is None else int(_row_get(row, 0, "v"))
            for v, sql in enumerate(self.MIGRATIONS[version + 1 :], start=version + 1):
                try:
                    cur.execute(sql)
                    cur.execute("INSERT INTO store_migrations (v) VALUES (%s)", (v,))
                except Exception as exc:
                    logger.error("Failed to apply store migration %s: %s", v, exc)
                    raise
            if self.index_config:
                try:
                    cur.execute(
                        _VECTOR_INDEX_TABLE_SQL.format(
                            int(self.index_config.get("dims", 0)),
                            json.dumps(self.index_config.get("ann_index_config")),
                        ),
                    )
                except Exception as exc:
                    logger.error("Failed to create vector index table: %s", exc)
                    raise

    def close(self) -> None:
        """Release resources; safe to call multiple times.
        No-op on caller-owned connections/pools.
        """
        try:
            if hasattr(self, "_ttl_stop_event") and hasattr(
                self, "_ttl_sweeper_thread"
            ):
                self.stop_ttl_sweeper(timeout=0.1)
        except Exception as exc:
            logger.error("Failed to stop TTL sweeper: %s", exc)
        self.connection_pool.dispose()

    # ---------------------------------------------------------------- batch
    def batch(self, ops: Iterable[Op]) -> list[Result]:
        grouped, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        with self._cursor() as cur:
            if GetOp in grouped:
                self._batch_get_ops(
                    cast("Sequence[tuple[int, GetOp]]", grouped[GetOp]),
                    results,
                    cur,
                )
            if SearchOp in grouped:
                self._batch_search_ops(
                    cast("Sequence[tuple[int, SearchOp]]", grouped[SearchOp]),
                    results,
                    cur,
                )
            if ListNamespacesOp in grouped:
                self._batch_list_namespaces_ops(
                    cast(
                        "Sequence[tuple[int, ListNamespacesOp]]",
                        grouped[ListNamespacesOp],
                    ),
                    results,
                    cur,
                )
            if PutOp in grouped:
                self._batch_put_ops(
                    cast("Sequence[tuple[int, PutOp]]", grouped[PutOp]), cur
                )
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        # SingleStore driver is sync-only; run in the default executor.
        return await asyncio.get_running_loop().run_in_executor(
            None, self.batch, list(ops)
        )

    # ------------------------------------------------------- op implementations
    def _batch_get_ops(
        self,
        get_ops: "Sequence[tuple[int, GetOp]]",
        results: list[Result],
        cur: Any,
    ) -> None:
        by_ns: dict[tuple[str, ...], list[tuple[int, str]]] = defaultdict(list)
        by_ns_ttl: dict[tuple[str, ...], list[tuple[int, str]]] = defaultdict(list)
        for idx, op in get_ops:
            if op.refresh_ttl:
                by_ns_ttl[op.namespace].append((idx, op.key))
            by_ns[op.namespace].append((idx, op.key))

        # One `key IN (...)` per namespace. When any op in the namespace asked
        # for a TTL refresh, wrap the SELECT + UPDATE in a transaction and
        # ``FOR UPDATE`` the read to avoid a lost-update race with a concurrent
        # writer or sweeper.
        for namespace, items in by_ns.items():
            namespace_text = _namespace_to_text(namespace)
            keys = [k for _, k in items]
            placeholders = ",".join(["%s"] * len(keys))
            needs_refresh = namespace in by_ns_ttl

            select_sql = f"{_SELECT_BASE} prefix = %s AND `key` IN ({placeholders})" + (
                " FOR UPDATE" if needs_refresh else ""
            )
            if needs_refresh:
                cur.execute("BEGIN")
                try:
                    cur.execute(select_sql, (namespace_text, *keys))
                    rows_by_key = {
                        _row_get(row, 1, "key"): row for row in cur.fetchall()
                    }
                    for idx, key in items:
                        row = rows_by_key.get(key)
                        results[idx] = _row_to_item(namespace, row) if row else None

                    ttl_keys = [k for _, k in by_ns_ttl[namespace]]
                    ttl_placeholders = ",".join(["%s"] * len(ttl_keys))
                    cur.execute(
                        f"{_REFRESH_TTL_SQL_BASE} prefix = %s "
                        f"AND `key` IN ({ttl_placeholders})",
                        (namespace_text, *ttl_keys),
                    )
                    cur.execute("COMMIT")
                except BaseException:
                    # Caller-owned connections are not closed by __exit__, so
                    # an open transaction would leak onto later operations.
                    _safe_rollback(cur)
                    raise
            else:
                cur.execute(select_sql, (namespace_text, *keys))
                rows_by_key = {_row_get(row, 1, "key"): row for row in cur.fetchall()}
                for idx, key in items:
                    row = rows_by_key.get(key)
                    results[idx] = _row_to_item(namespace, row) if row else None

    def _batch_put_ops(
        self,
        put_ops: "Sequence[tuple[int, PutOp]]",
        cur: Any,
    ) -> None:
        # Deduplicate: last write for a (namespace, key) wins.
        dedupped: dict[tuple[tuple[str, ...], str], PutOp] = {}
        for _, op in put_ops:
            dedupped[(op.namespace, op.key)] = op

        is_updated_vector_index: bool = False
        inserts: list[PutOp] = []
        deletes_by_ns: dict[tuple[str, ...], list[str]] = defaultdict(list)
        inserted_by_ns: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for op in dedupped.values():
            if op.value is None:
                deletes_by_ns[op.namespace].append(op.key)
            else:
                inserts.append(op)
                if self.index_config:
                    inserted_by_ns[op.namespace].append(op.key)

        for namespace, keys in deletes_by_ns.items():
            placeholders = ",".join(["%s"] * len(keys))
            cur.execute(
                _DELETE_BASE_FROM_STORE + f"({placeholders})",
                (_namespace_to_text(namespace), *keys),
            )
            if self.index_config:
                # Delete corresponding entries from the vector index table as well.
                is_updated_vector_index = True
                cur.execute(
                    _DELETE_BASE_FROM_STORE_VECTOR_BASE + f"({placeholders})",
                    (_namespace_to_text(namespace), *keys),
                )
        # Delete entries from the vector index table
        # that correspond to newly inserted base entries.
        for namespace, keys in inserted_by_ns.items():
            placeholders = ",".join(["%s"] * len(keys))
            cur.execute(
                _DELETE_BASE_FROM_STORE_VECTOR_BASE + f"({placeholders})",
                (_namespace_to_text(namespace), *keys),
            )
        insert_values: list[Any] = []
        insert_placeholders: list[str] = []
        insert_vector_key_values: list[tuple[str, str, str]] = []
        embedding_requests: list[str] = []
        insert_vector_placeholders: list[str] = []
        for op in inserts:
            insert_values.extend(
                [
                    _namespace_to_text(op.namespace),
                    op.key,
                    json.dumps(op.value),
                ]
            )
            if op.ttl:
                ttl_minutes = float(op.ttl)
                insert_placeholders.append(
                    "(%s, %s, %s, NOW(), NOW(),"
                    + " DATE_ADD(NOW(), INTERVAL %s MINUTE), %s)"
                )
                # ``ttl_minutes`` column is INTEGER; bind numerics, not strings,
                # so strict-mode SingleStore accepts the value.
                insert_values.extend([ttl_minutes, int(round(ttl_minutes))])
            else:
                insert_placeholders.append("(%s, %s, %s, NOW(), NOW(), NULL, NULL)")

            if self.index_config and op.index is not False:
                value = op.value
                ns = _namespace_to_text(op.namespace)
                k = op.key
                if op.index is None:
                    paths = cast(dict, self.index_config)["__tokenized_fields"]
                else:
                    paths = [(ix, tokenize_path(ix)) for ix in op.index]
                for path, tokenized_path in paths:
                    texts = get_text_at_path(value, tokenized_path)
                    for i, text in enumerate(texts):
                        pathname = f"{path}.{i}" if len(texts) > 1 else path
                        insert_vector_placeholders.append(
                            "(%s, %s, %s, JSON_ARRAY_PACK(%s), "
                            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                        )
                        insert_vector_key_values.append((ns, k, pathname))
                        embedding_requests.append(text)
        if embedding_requests:
            if self.embeddings is None:
                raise ValueError(
                    "Embedding configuration is required for vector operations "
                    "(for semantic search). Please provide an Embeddings when "
                    f"initializing the {self.__class__.__name__}."
                )
            vector_embeddings = self.embeddings.embed_documents(embedding_requests)
            insert_vector_values: list[Any] = []
            for key_values, embedding in zip(
                insert_vector_key_values, vector_embeddings
            ):
                insert_vector_values.extend(
                    [*key_values, "[{}]".format(",".join(map(str, embedding)))]
                )
            cur.execute(
                _UPSERT_BASE_VECTOR_SQL
                + ",".join(insert_vector_placeholders)
                + _ON_DUPLICATE_KEY_VECTOR_UPDATE_SQL,
                tuple(insert_vector_values),
            )
            is_updated_vector_index = True

        if insert_placeholders:
            cur.execute(
                _UPSERT_BASE_SQL
                + ",".join(insert_placeholders)
                + _ON_DUPLICATE_KEY_UPDATE_SQL,
                tuple(insert_values),
            )

        if is_updated_vector_index:
            cur.execute(_FLUSH_VECTOR_STORE_SQL)

    def _batch_search_ops(
        self,
        search_ops: "Sequence[tuple[int, SearchOp]]",
        results: list[Result],
        cur: Any,
    ) -> None:
        for idx, op in search_ops:
            select_params: list[Any] = []
            if op.query:
                if not self.index_config:
                    raise ValueError(
                        "Index configuration is required for search operations. "
                        "Please provide an index configuration "
                        "when initializing the store."
                    )
                if not self.embeddings:
                    raise ValueError(
                        "Embeddings are required for search operations. "
                        "Please provide embeddings "
                        "when initializing the store."
                    )
                embed_query = self.embeddings.embed_query(op.query)
                metric_type = self.index_config["ann_index_config"]["metric_type"]
                base_sql = _SELECT_WITH_VECTOR_SEARCH_SQL.format(
                    AGGREGATE_FUNCTIONS_SQL[metric_type],
                    metric_type.value,
                )
                select_params.append(f"[{','.join(map(str, embed_query))}]")
            else:
                base_sql = _SELECT_BASE

            where_sql, where_params = _search_where(op)
            # ``_SELECT_BASE`` / ``_REFRESH_TTL_SQL`` end with an unconditional
            # ``AND``; supply a truthy tail when the op has no prefix/filter.
            where_sql = where_sql or "TRUE"
            base_sql = f"{base_sql} {where_sql}"
            if op.query:
                order_by_sql = _ORDER_BY_VECTOR_SEARCH_SQL.format(
                    SCORE_ORDER_DIRECTION[metric_type]
                )
                base_sql = f"{base_sql} {_GROUP_BY_VECTOR_SEARCH_SQL} {order_by_sql}"
            else:
                base_sql = f"{base_sql} {_ORDER_BY_BASE}"
            if op.refresh_ttl:
                if op.query:
                    # SingleStore rejects ``FOR UPDATE`` on distributed JOINs
                    # (error 1706), so the vector-search path cannot lock the
                    # SELECT the way the base path does. Instead, run the
                    # ranked SELECT normally and refresh TTLs on the exact
                    # ``(prefix, key)`` pairs returned.
                    cur.execute(
                        f"{base_sql}",
                        (*select_params, *where_params, op.limit, op.offset),
                    )
                    fetched = list(cur.fetchall())
                    results[idx] = [
                        _row_to_search_item(
                            _text_to_namespace(_row_get(row, 0, "prefix")), row
                        )
                        for row in fetched
                    ]
                    by_prefix: dict[str, list[str]] = defaultdict(list)
                    for row in fetched:
                        by_prefix[_row_get(row, 0, "prefix")].append(
                            _row_get(row, 1, "key")
                        )
                    for prefix, keys in by_prefix.items():
                        placeholders = ",".join(["%s"] * len(keys))
                        cur.execute(
                            f"{_REFRESH_TTL_SQL_BASE} prefix = %s "
                            f"AND `key` IN ({placeholders})",
                            (prefix, *keys),
                        )
                    continue
                cur.execute("BEGIN")
                try:
                    cur.execute(
                        f"{base_sql} FOR UPDATE",
                        (*select_params, *where_params, op.limit, op.offset),
                    )
                    results[idx] = [
                        _row_to_search_item(
                            _text_to_namespace(_row_get(row, 0, "prefix")), row
                        )
                        for row in cur.fetchall()
                    ]
                    cur.execute(
                        f"{_REFRESH_TTL_SQL_BASE} {where_sql}",
                        where_params,
                    )
                    cur.execute("COMMIT")
                except BaseException:
                    # Caller-owned connections are not closed by __exit__, so
                    # an open transaction would leak onto later operations.
                    _safe_rollback(cur)
                    raise
            else:
                cur.execute(
                    f"{base_sql}",
                    (*select_params, *where_params, op.limit, op.offset),
                )
                results[idx] = [
                    _row_to_search_item(
                        _text_to_namespace(_row_get(row, 0, "prefix")), row
                    )
                    for row in cur.fetchall()
                ]

    def _batch_list_namespaces_ops(
        self,
        list_ops: "Sequence[tuple[int, ListNamespacesOp]]",
        results: list[Result],
        cur: Any,
    ) -> None:
        for idx, op in list_ops:
            where_clauses: list[str] = []
            params: list[Any] = []
            for cond in op.match_conditions or []:
                exact_filter, exact_match_prefix = _namespace_for_exact_search(
                    cond.path
                )
                where_clauses.append(f"({exact_filter} OR prefix LIKE %s)")
                params.append(exact_match_prefix)
                if cond.match_type == "prefix":
                    params.append(_namespace_for_prefix_search(cond.path))
                elif cond.match_type == "suffix":
                    params.append(_namespace_for_suffix_search(cond.path))
                else:  # pragma: no cover - defensive
                    logger.warning(
                        "Unknown match_type in list_namespaces: %s",
                        cond.match_type,
                    )
            where_sql = (
                "WHERE (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)"
                + (f" AND {' AND '.join(where_clauses)}" if where_clauses else "")
            )
            # ``max_depth`` truncates each returned namespace to the first N
            # parts. SingleStore lacks Postgres' ``unnest``, so the SQL side
            # calls the ``truncate_ns_prefix`` UDF (see migrations) which
            # respects the escape scheme and slices at unescaped ``/``.
            if op.max_depth is None:
                select_expr = "prefix"
                depth_params: tuple[Any, ...] = ()
            else:
                select_expr = "truncate_ns_prefix(prefix, %s)"
                depth_params = (op.max_depth,)
            cur.execute(
                f"SELECT DISTINCT {select_expr} AS trunc_prefix FROM store "
                f"{where_sql} ORDER BY trunc_prefix LIMIT %s OFFSET %s",
                (*depth_params, *params, op.limit, op.offset),
            )
            rows = cur.fetchall()
            seen: dict[tuple[str, ...], None] = {}
            for row in rows:
                ns = _text_to_namespace(_row_get(row, 0, "trunc_prefix"))
                seen[ns] = None
            results[idx] = list(seen.keys())

    def sweep_ttl(self) -> int:
        """Delete expired store items based on TTL.

        Returns:
            int: The number of deleted items.
        """
        with self._cursor() as cur:
            # ``<=`` mirrors the read-path check (``expires_at > NOW()`` means
            # still valid); a row with ``expires_at == NOW()`` is already
            # invisible to reads and must be sweepable.
            cur.execute(_DELETE_EXPIRED_FROM_STORE)
            if self.index_config:
                cur.execute(_DELETE_EXPIRED_FROM_STORE_VECTOR)
            deleted_count = cur.rowcount
            return deleted_count

    def start_ttl_sweeper(
        self, sweep_interval_minutes: float | None = None
    ) -> concurrent.futures.Future[None]:
        """Start a background thread that periodically deletes expired items.

        The first sweep runs synchronously in the background thread as soon as
        it starts, so a caller can ``start_ttl_sweeper`` and immediately
        ``stop_ttl_sweeper`` to force one pass.

        Returns:
            A ``Future`` that resolves when the background loop has exited
            (i.e. after ``stop_ttl_sweeper`` is called or the loop crashes).
            Idempotent: calling this while a sweeper is already running
            returns the *same* future for the running loop.
        """
        if not self.ttl_config:
            future: concurrent.futures.Future[None] = concurrent.futures.Future()
            future.set_result(None)
            return future

        if (
            self._ttl_sweeper_thread
            and self._ttl_sweeper_thread.is_alive()
            and self._ttl_sweeper_future is not None
        ):
            logger.info("TTL sweeper thread is already running")
            return self._ttl_sweeper_future

        self._ttl_stop_event.clear()

        interval = float(
            sweep_interval_minutes or self.ttl_config.get("sweep_interval_minutes") or 5
        )
        logger.info(f"Starting store TTL sweeper with interval {interval} minutes")

        future = concurrent.futures.Future()

        def _sweep_loop() -> None:
            try:
                while not self._ttl_stop_event.is_set():
                    try:
                        expired_items = self.sweep_ttl()
                        if expired_items > 0:
                            logger.info(f"Store swept {expired_items} expired items")
                    except Exception as exc:
                        logger.exception(
                            "Store TTL sweep iteration failed", exc_info=exc
                        )
                    # ``wait`` returns True as soon as ``stop_ttl_sweeper``
                    # fires the event, giving prompt shutdown.
                    if self._ttl_stop_event.wait(interval * 60):
                        break
                future.set_result(None)
            except Exception as exc:
                future.set_exception(exc)

        thread = threading.Thread(target=_sweep_loop, daemon=True, name="ttl-sweeper")
        self._ttl_sweeper_thread = thread
        self._ttl_sweeper_future = future
        thread.start()
        return future

    def stop_ttl_sweeper(self, timeout: float | None = None) -> bool:
        """Stop the TTL sweeper thread if it's running.

        Args:
            timeout: Maximum time to wait for the thread to stop, in seconds.
                If `None`, wait indefinitely.

        Returns:
            bool: True if the thread was successfully stopped or wasn't running,
                False if the timeout was reached before the thread stopped.
        """
        if not self._ttl_sweeper_thread or not self._ttl_sweeper_thread.is_alive():
            return True

        logger.info("Stopping TTL sweeper thread")
        self._ttl_stop_event.set()

        self._ttl_sweeper_thread.join(timeout)
        success = not self._ttl_sweeper_thread.is_alive()

        if success:
            self._ttl_sweeper_thread = None
            self._ttl_sweeper_future = None
            logger.info("TTL sweeper thread stopped")
        else:
            logger.warning("Timed out waiting for TTL sweeper thread to stop")

        return success

    def __del__(self) -> None:
        # Safe during interpreter shutdown: attribute access and logger calls
        # can fail once modules are torn down, so swallow everything.
        try:
            if hasattr(self, "_ttl_stop_event") and hasattr(
                self, "_ttl_sweeper_thread"
            ):
                self.stop_ttl_sweeper(timeout=0.1)
        except Exception:
            pass

    # -------------------------------------------------------------- cursor
    class _CursorContext:
        def __init__(self, pool: Pool, lock: threading.Lock) -> None:
            self._pool = pool
            self._lock = lock
            self._conn: Any = None
            self._cur: Any = None

        def __enter__(self) -> Any:
            self._lock.acquire()
            self._conn = self._pool.connect()
            self._cur = self._conn.cursor()
            return self._cur

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            try:
                if self._cur is not None:
                    self._cur.close()
            finally:
                try:
                    if self._conn is not None:
                        self._conn.close()
                finally:
                    self._lock.release()

    def _cursor(self) -> "SingleStoreStore._CursorContext":
        return SingleStoreStore._CursorContext(self.connection_pool, self._lock)


# ---------------------------------------------------------------- helpers


def _group_ops(
    ops: Iterable[Op],
) -> tuple[dict[type, list[tuple[int, Op]]], int]:
    grouped: dict[type, list[tuple[int, Op]]] = defaultdict(list)
    total = 0
    for idx, op in enumerate(ops):
        grouped[type(op)].append((idx, op))
        total += 1
    return grouped, total


# Namespace parts are joined with "/" for storage in the ``prefix`` column.
# Each part is escaped so joining stays unambiguous *and* so the encoded
# text can be embedded in a LIKE pattern without wildcards leaking in:
#   "\" -> "\\"   escape the escape character
#   "/" -> "\/"   escape the separator
#   "%" -> "\%"   escape the LIKE multi-char wildcard
#   "_" -> "\_"   escape the LIKE single-char wildcard
# ``_namespace_to_text`` is injective, so ``("a/b", "c")`` and
# ``("a", "b", "c")`` serialize to distinct prefixes.
_NS_SEPARATOR = "/"
_NS_ESCAPE = "\\"
_NS_LIKE_WILDCARD_ANY = "%"
_NS_LIKE_WILDCARD_ONE = "_"
_NS_WILDCARD = "*"


def _escape_ns_part(part: str) -> str:
    return (
        part.replace(_NS_ESCAPE, _NS_ESCAPE * 2)
        .replace(_NS_SEPARATOR, _NS_ESCAPE + _NS_SEPARATOR)
        .replace(_NS_LIKE_WILDCARD_ANY, _NS_ESCAPE + _NS_LIKE_WILDCARD_ANY)
        .replace(_NS_LIKE_WILDCARD_ONE, _NS_ESCAPE + _NS_LIKE_WILDCARD_ONE)
    )


def _unescape_ns_part(part: str) -> str:
    out: list[str] = []
    i = 0
    n = len(part)
    while i < n:
        ch = part[i]
        if ch == _NS_ESCAPE and i + 1 < n:
            out.append(part[i + 1])
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _namespace_to_text(namespace: tuple[str, ...]) -> str:
    return _NS_SEPARATOR.join(_escape_ns_part(p) for p in namespace)


def _namespace_with_wildcard_for_search(namespace: tuple[str, ...]) -> str:
    return _NS_SEPARATOR.join(
        _escape_ns_part(p) if p != _NS_WILDCARD else _NS_LIKE_WILDCARD_ANY
        for p in namespace
    )


def _namespace_for_prefix_search(namespace: tuple[str, ...]) -> str:
    return (
        _namespace_with_wildcard_for_search(namespace)
        + _NS_SEPARATOR
        + _NS_LIKE_WILDCARD_ANY
    )


def _namespace_for_suffix_search(namespace: tuple[str, ...]) -> str:
    return (
        _NS_LIKE_WILDCARD_ANY
        + _NS_SEPARATOR
        + _namespace_with_wildcard_for_search(namespace)
    )


def _namespace_for_exact_search(
    namespace: tuple[str, ...], column: str = "prefix"
) -> tuple[str, str]:
    if "*" in namespace:
        return (
            f"{column} LIKE %s",
            _namespace_with_wildcard_for_search(namespace),
        )
    return (
        f"{column} = %s",
        _namespace_to_text(namespace),
    )


def _text_to_namespace(text: str) -> tuple[str, ...]:
    if not text:
        return ()
    parts: list[str] = []
    buf: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == _NS_ESCAPE and i + 1 < n:
            buf.append(ch)
            buf.append(text[i + 1])
            i += 2
        elif ch == _NS_SEPARATOR:
            parts.append(_unescape_ns_part("".join(buf)))
            buf = []
            i += 1
        else:
            buf.append(ch)
            i += 1
    parts.append(_unescape_ns_part("".join(buf)))
    return tuple(parts)


def _search_where(op: SearchOp) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if op.namespace_prefix:
        # Always qualify with ``store.`` so the same WHERE clause works both
        # against ``FROM store`` (base search) and the JOIN with
        # ``store_vector`` used by the query path, where a bare ``prefix``
        # would be ambiguous.
        exact_filter, exact_match_param = _namespace_for_exact_search(
            op.namespace_prefix, column="store.prefix"
        )
        prefix = _namespace_for_prefix_search(op.namespace_prefix)
        clauses.append(f"(store.prefix LIKE %s OR {exact_filter})")
        params.extend([prefix, exact_match_param])
    if op.filter and len(op.filter) > 0:
        adjusted_filter = cast(
            FilterTypedDict,
            {"$and": [{k: v} for k, v in op.filter.items()]},
        )
        filter_clause, filter_params = _parse_filter(
            filter_dict=adjusted_filter, metadata_field="store.value"
        )
        clauses.append(f"({filter_clause})")
        params.extend(filter_params)
    where_sql = " AND ".join(clauses) if clauses else ""
    return where_sql, params


def _row_get(row: Any, index: int, name: str) -> Any:
    """Access a row column by name or index.

    ``singlestoredb.connect(..., results_type="dict")`` yields dicts, whereas
    the default tuple results support only positional access. This helper
    lets the store work with either.
    """
    if isinstance(row, dict):
        return row[name]
    return row[index]


def _row_to_item(namespace: tuple[str, ...], row: Any) -> Item:
    value = _row_get(row, 2, "value")
    if not isinstance(value, dict):
        value = json.loads(value)
    return Item(
        namespace=namespace,
        key=_row_get(row, 1, "key"),
        value=value,
        created_at=_row_get(row, 3, "created_at"),
        updated_at=_row_get(row, 4, "updated_at"),
    )


def _row_to_search_item(namespace: tuple[str, ...], row: Any) -> SearchItem:
    value = _row_get(row, 2, "value")
    if not isinstance(value, dict):
        value = json.loads(value)
    # ``score`` is only present on vector-search rows; the base SELECT has no
    # such column so we fall back to ``None``.
    score: Any = None
    if isinstance(row, dict):
        score = row.get("score")
    elif len(row) > 7:
        score = row[7]
    return SearchItem(
        namespace=namespace,
        key=_row_get(row, 1, "key"),
        value=value,
        created_at=_row_get(row, 3, "created_at"),
        updated_at=_row_get(row, 4, "updated_at"),
        score=float(score) if score is not None else None,
    )


def _ensure_index_config(
    index_config: SingleStoreIndexConfig,
) -> tuple[Optional[Embeddings], SingleStoreIndexConfig]:
    index_config = index_config.copy()
    tokenized: list[tuple[str, Literal["$"] | list[str]]] = []
    tot = 0
    fields = index_config.get("fields") or ["$"]
    if isinstance(fields, str):
        fields = [fields]
    if not isinstance(fields, list):
        raise ValueError(f"Text fields must be a list or a string. Got {fields}")
    for p in fields:
        if p == "$":
            tokenized.append((p, "$"))
            tot += 1
        else:
            toks = tokenize_path(p)
            tokenized.append((p, toks))
            tot += len(toks)
    index_config["__tokenized_fields"] = tokenized  # type: ignore
    index_config["__estimated_num_vectors"] = tot  # type: ignore
    index_config["fields"] = fields  # type: ignore
    embeddings = ensure_embeddings(
        index_config.get("embed"),
    )
    ann_index_config = index_config.get("ann_index_config", {})
    ann_index_config["metric_type"] = ann_index_config.get(
        "metric_type", DistanceStrategy.DOT_PRODUCT
    )
    ann_index_config["index_type"] = ann_index_config.get("index_type", "FLAT")
    index_config["ann_index_config"] = ann_index_config
    return embeddings, index_config
