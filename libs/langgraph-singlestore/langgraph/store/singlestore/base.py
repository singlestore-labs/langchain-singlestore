"""Synchronous SingleStore-backed :class:`~langgraph.store.base.BaseStore`.

Draft implementation modelled after ``langgraph.store.postgres.base``. Uses
``singlestoredb`` through the shared :func:`create_connection_pool` factory
from ``singlestore_langchain_core``, so callers may supply an existing
connection, an existing pool, or plain connection kwargs — identical
semantics to ``langchain-singlestore``.

Vector search (``index=...``) and TTL sweeping are intentionally out of scope
for this draft; they will be layered on top of ``SingleStoreVectorStore`` and
a background sweeper in a follow-up.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import defaultdict
from typing import Any, Iterable, Optional, Sequence, cast

from singlestore_langchain_core import create_connection_pool
from singlestore_langchain_core._utils import (
    LANGGRAPH_CONNECTOR_NAME,
    compute_connector_version,
    set_connector_attributes,
)
from singlestoredb.connection import Connection
from sqlalchemy.pool import Pool

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
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
        INDEX store_prefix_idx (prefix(255))
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

_SELECT_BASE = """
    SELECT prefix, `key`, value, created_at, updated_at, expires_at, ttl_minutes
    FROM store WHERE (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP) AND
"""

_REFRESH_TTL_SQL = """
    UPDATE store
    SET expires_at = DATE_ADD(NOW(), INTERVAL ttl_minutes MINUTE),
        updated_at = CURRENT_TIMESTAMP
    WHERE (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP) AND
"""


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
        self.ttl_config = ttl_config
        self._ttl_sweeper_thread: Optional[threading.Thread] = None
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

    def close(self) -> None:
        """Release resources; safe to call multiple times.

        No-op on caller-owned connections/pools.
        """
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
        # Group `getOps` by whether refreshing ttl or not
        for idx, op in get_ops:
            if op.refresh_ttl:
                by_ns_ttl[op.namespace].append((idx, op.key))
            by_ns[op.namespace].append((idx, op.key))
        # Group by namespace so we can issue one `key IN (...)` per prefix.

        for namespace, items in by_ns.items():
            if namespace in by_ns_ttl:
                keys_ttl = [k for _, k in by_ns_ttl[namespace]]
                placeholders_ttl = ",".join(["%s"] * len(keys_ttl))
                # Handle TTL refresh for this namespace if needed
                cur.execute(
                    f"{_REFRESH_TTL_SQL} prefix = %s AND `key` IN ({placeholders_ttl})",
                    (_namespace_to_text(namespace), *keys_ttl),
                )

            keys = [k for _, k in items]
            placeholders = ",".join(["%s"] * len(keys))
            cur.execute(
                f"{_SELECT_BASE} prefix = %s AND `key` IN ({placeholders})",
                (_namespace_to_text(namespace), *keys),
            )
            rows_by_key: dict[str, Any] = {}
            for row in cur.fetchall():
                rows_by_key[_row_get(row, 1, "key")] = row
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

        inserts: list[PutOp] = []
        deletes_by_ns: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for op in dedupped.values():
            if op.value is None:
                deletes_by_ns[op.namespace].append(op.key)
            else:
                inserts.append(op)

        for namespace, keys in deletes_by_ns.items():
            placeholders = ",".join(["%s"] * len(keys))
            cur.execute(
                f"DELETE FROM store WHERE prefix = %s AND `key` IN ({placeholders})",
                (_namespace_to_text(namespace), *keys),
            )

        insert_values: list[Any] = []
        insert_placeholders: list[str] = []
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

        if insert_placeholders:
            cur.execute(
                _UPSERT_BASE_SQL
                + ",".join(insert_placeholders)
                + _ON_DUPLICATE_KEY_UPDATE_SQL,
                tuple(insert_values),
            )

    def _batch_search_ops(
        self,
        search_ops: "Sequence[tuple[int, SearchOp]]",
        results: list[Result],
        cur: Any,
    ) -> None:
        for idx, op in search_ops:
            if op.query:
                raise NotImplementedError(
                    "Vector search is not yet implemented in this draft. "
                    "Track progress in libs/langgraph-singlestore/CHANGELOG.md."
                )
            where_sql, params = _search_where(op)
            cur.execute(
                f"{_SELECT_BASE} {where_sql} "
                f"ORDER BY updated_at DESC LIMIT %s OFFSET %s",
                (*params, op.limit, op.offset),
            )
            results[idx] = [
                _row_to_search_item(_text_to_namespace(_row_get(row, 0, "prefix")), row)
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
                if cond.match_type == "prefix":
                    where_clauses.append("prefix LIKE %s")
                    params.append(_namespace_for_prefix_search(cond.path))
                elif cond.match_type == "suffix":
                    where_clauses.append("prefix LIKE %s")
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
        prefix = _namespace_to_text(op.namespace_prefix)
        clauses.append("(prefix = %s OR prefix LIKE %s)")
        params.extend([prefix, prefix + ".%"])
    if op.filter:
        pass
        # todo: implement filter handling using singlestore-langchain-core
    where_sql = "WHERE " + " AND ".join(clauses) if clauses else ""
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
    return SearchItem(
        namespace=namespace,
        key=_row_get(row, 1, "key"),
        value=value,
        created_at=_row_get(row, 3, "created_at"),
        updated_at=_row_get(row, 4, "updated_at"),
    )
