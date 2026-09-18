"""Synchronous SingleStore-backed ``BaseCheckpointSaver``.

Implements the checkpoint saver on top of ``singlestoredb`` using the shared
pooled-connection helpers from ``singlestore_langchain_core``. SQL statements
and serde helpers live on :class:`BaseSingleStoreSaver`; this module wires
them up to a real connection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import defaultdict
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Optional, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_serializable_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from singlestore_langchain_core._connection import create_connection_pool
from singlestore_langchain_core._utils import (
    DEFAULT_CONNECTOR_NAME,
    compute_connector_version,
    set_connector_attributes,
)
from singlestoredb.connection import Connection
from sqlalchemy.pool import Pool

from langgraph_singlestore.checkpoint._base import BaseSingleStoreSaver

logger = logging.getLogger(__name__)

# Aliased for use inside ``SingleStoreSaver`` where the ``list`` method
# shadows the builtin.
_builtin_list = list


class SingleStoreSaver(BaseSingleStoreSaver):
    """SingleStore-backed checkpoint saver (synchronous)."""

    def __init__(
        self,
        *,
        connection: Optional[Connection] = None,
        connection_pool: Optional[Pool] = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        timeout: float = 30,
        serde: Optional[SerializerProtocol] = None,
        **connection_kwargs: Any,
    ) -> None:
        """
        Following arguments pertain to the connection pool:

        connection (singlestoredb.Connection, optional): An existing
            caller-owned SingleStoreDB connection. When supplied, every
            database operation shares this connection through an internal
            proxy that never closes it; the caller keeps full ownership of
            the connection lifecycle. Mutually exclusive with
            ``connection_pool``.

        connection_pool (sqlalchemy.pool.Pool, optional): A pre-built
            SQLAlchemy connection pool to use as-is. Useful when the
            surrounding application already manages its own pool (custom
            pool class, shared pool across components, etc.). Mutually
            exclusive with ``connection``.

            When neither ``connection`` nor ``connection_pool`` is passed,
            a default :class:`QueueConnectionPool` is built from
            ``pool_size``, ``max_overflow``, ``timeout``, and the
            connection kwargs described below.

        pool_size (int, optional): Determines the number of active connections in
            the pool. Defaults to 5. Ignored if ``connection`` or
            ``connection_pool`` is supplied.

        max_overflow (int, optional): Determines the maximum number of connections
            allowed beyond the pool_size. Defaults to 10. Ignored if
            ``connection`` or ``connection_pool`` is supplied.

        timeout (float, optional): Specifies the maximum wait time in seconds for
            establishing a connection. Defaults to 30. Ignored if
            ``connection`` or ``connection_pool`` is supplied.


        Following arguments pertain to the database connection:

        host (str, optional): Specifies the hostname, IP address, or URL for the
            database connection. The default scheme is "mysql".

        user (str, optional): Database username.

        password (str, optional): Database password.

        port (int, optional): Database port. Defaults to 3306 for non-HTTP
            connections, 80 for HTTP connections, and 443 for HTTPS connections.

        database (str, optional): Database name.


        Additional optional arguments provide further customization over the
        database connection:

        pure_python (bool, optional): Toggles the connector mode. If True,
            operates in pure Python mode.

        local_infile (bool, optional): Allows local file uploads.

        charset (str, optional): Specifies the character set for string values.

        ssl_key (str, optional): Specifies the path of the file containing the SSL
            key.

        ssl_cert (str, optional): Specifies the path of the file containing the SSL
            certificate.

        ssl_ca (str, optional): Specifies the path of the file containing the SSL
            certificate authority.

        ssl_cipher (str, optional): Sets the SSL cipher list.

        ssl_disabled (bool, optional): Disables SSL usage.

        ssl_verify_cert (bool, optional): Verifies the server's certificate.
            Automatically enabled if ``ssl_ca`` is specified.

        ssl_verify_identity (bool, optional): Verifies the server's identity.

        conv (dict[int, Callable], optional): A dictionary of data conversion
            functions.

        credential_type (str, optional): Specifies the type of authentication to
            use: auth.PASSWORD, auth.JWT, or auth.BROWSER_SSO.

        autocommit (bool, optional): Enables autocommits.

        results_type (str, optional): Determines the structure of the query results:
            tuples, namedtuples, dicts.

        results_format (str, optional): Deprecated. This option has been renamed to
            results_type.
        """
        super().__init__(serde=serde)
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.timeout = timeout
        set_connector_attributes(
            connection_kwargs,
            connector_name=DEFAULT_CONNECTOR_NAME,
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
        # Serialise access to the underlying connection when the caller
        # shares a single connection across threads.
        self._lock = threading.Lock()

    @classmethod
    def from_conn_string(
        cls,
        conn_string: str,
        *,
        serde: SerializerProtocol | None = None,
    ) -> SingleStoreSaver:
        """Create a ``SingleStoreSaver`` from a SingleStore connection URL.

        The URL is forwarded to :func:`singlestoredb.connect` as ``host=`` —
        it accepts the full ``user:password@host:port/database`` form.

        Args:
            conn_string: The SingleStore connection URL.
            serde: Optional serializer.

        Returns:
            A new ``SingleStoreSaver`` — call ``close()`` when done.
        """
        return cls(host=conn_string, serde=serde)

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        """Run pending migrations. Idempotent; call once before first use."""
        with self._cursor() as cur:
            cur.execute(self.MIGRATIONS[0])
            cur.execute("SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1")
            row = cur.fetchone()
            version = -1 if row is None else int(_row_get(row, 0, "v"))
            for v, migration in enumerate(
                self.MIGRATIONS[version + 1 :], start=version + 1
            ):
                try:
                    cur.execute(migration)
                    cur.execute(
                        "INSERT INTO checkpoint_migrations (v) VALUES (%s)",
                        (v,),
                    )
                except Exception as exc:
                    logger.error("Failed to apply checkpoint migration %s: %s", v, exc)
                    raise

    def close(self) -> None:
        """Release resources; safe to call multiple times.

        No-op on caller-owned connections/pools.
        """
        self.connection_pool.dispose()

    # ---------------------------------------------------------------- reads
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Retrieve the checkpoint tuple identified by ``config``.

        If ``configurable.checkpoint_id`` is set, return that specific
        checkpoint; otherwise return the latest checkpoint for the thread.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        if checkpoint_id:
            where = (
                "WHERE c.thread_id = %s AND c.checkpoint_ns = %s "
                "AND c.checkpoint_id = %s"
            )
            args: tuple[Any, ...] = (thread_id, checkpoint_ns, checkpoint_id)
            trailer = ""
        else:
            where = "WHERE c.thread_id = %s AND c.checkpoint_ns = %s"
            args = (thread_id, checkpoint_ns)
            trailer = " ORDER BY c.checkpoint_id DESC LIMIT 1"

        sql = self.SELECT_SQL.replace("{{where}}", where) + trailer

        with self._cursor() as cur:
            cur.execute(sql, args)
            row = cur.fetchone()
            if row is None:
                return None

            value = _row_to_checkpoint_dict(row)
            self._maybe_migrate_pending_sends(cur, [value])
            return self._load_checkpoint_tuple(value)

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """Yield checkpoint tuples matching ``config``/``filter``/``before``.

        Ordered newest-first by ``checkpoint_id``.
        """
        where, args = self._search_where(config, filter, before)
        query = self.SELECT_SQL.replace("{{where}}", where)
        query += " ORDER BY c.checkpoint_id DESC"
        params: list[Any] = list(args)
        if limit is not None:
            query += " LIMIT %s"
            params.append(int(limit))

        with self._cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            if not rows:
                return
            values = [_row_to_checkpoint_dict(row) for row in rows]
            self._maybe_migrate_pending_sends(cur, values)
            for value in values:
                yield self._load_checkpoint_tuple(value)

    # --------------------------------------------------------------- writes
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Persist ``checkpoint`` plus its metadata and new blob versions.

        Primitive channel values stay inline in the checkpoint's
        ``channel_values``; non-primitive values move to ``checkpoint_blobs``
        keyed by ``new_versions``.
        """
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns")
        parent_checkpoint_id = configurable.pop("checkpoint_id", None)

        copy = checkpoint.copy()
        copy["channel_values"] = copy["channel_values"].copy()

        blob_values: dict[str, Any] = {}
        for k, v in list(copy["channel_values"].items()):
            if v is None or isinstance(v, (str, int, float, bool)):
                continue
            blob_values[k] = copy["channel_values"].pop(k)

        next_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        with self._cursor() as cur:
            blob_versions = {k: v for k, v in new_versions.items() if k in blob_values}
            if blob_versions:
                blob_rows = self._dump_blobs(
                    thread_id, checkpoint_ns, blob_values, blob_versions
                )
                cur.executemany(self.UPSERT_CHECKPOINT_BLOBS_SQL, blob_rows)

            cur.execute(
                self.UPSERT_CHECKPOINTS_SQL,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    parent_checkpoint_id,
                    json.dumps(copy),
                    json.dumps(get_serializable_checkpoint_metadata(config, metadata)),
                ),
            )
        return next_config

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes attached to a checkpoint.

        Uses ``UPSERT`` when every write targets a reserved channel in
        ``WRITES_IDX_MAP``; otherwise falls back to ``INSERT IGNORE`` so
        pre-existing writes on the same ``(task, idx)`` are preserved.
        """
        query = (
            self.UPSERT_CHECKPOINT_WRITES_SQL
            if all(w[0] in WRITES_IDX_MAP for w in writes)
            else self.INSERT_CHECKPOINT_WRITES_SQL
        )
        rows = self._dump_writes(
            config["configurable"]["thread_id"],
            config["configurable"]["checkpoint_ns"],
            config["configurable"]["checkpoint_id"],
            task_id,
            task_path,
            writes,
        )
        if not rows:
            return
        with self._cursor() as cur:
            cur.executemany(query, rows)

    def delete_thread(self, thread_id: str) -> None:
        """Delete every checkpoint, blob, and write for ``thread_id``."""
        with self._cursor() as cur:
            cur.execute(
                "DELETE FROM checkpoints WHERE thread_id = %s",
                (str(thread_id),),
            )
            cur.execute(
                "DELETE FROM checkpoint_blobs WHERE thread_id = %s",
                (str(thread_id),),
            )
            cur.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = %s",
                (str(thread_id),),
            )

    # ---------------------------------------------------------------- async
    # SingleStore's driver is sync-only; async methods dispatch to a worker
    # thread. AsyncSingleStoreSaver may override with a native async driver.
    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.get_tuple, config
        )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        # Materialise on the worker to avoid streaming a cursor across threads.
        # ``_builtin_list`` sidesteps the ``self.list`` method shadowing the builtin.
        items = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: _builtin_list(
                self.list(config, filter=filter, before=before, limit=limit)
            ),
        )
        for item in items:
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.put, config, checkpoint, metadata, new_versions
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.get_running_loop().run_in_executor(
            None, self.put_writes, config, writes, task_id, task_path
        )

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.get_running_loop().run_in_executor(
            None, self.delete_thread, thread_id
        )

    # ------------------------------------------------------ internal helpers
    def _load_checkpoint_tuple(self, value: dict[str, Any]) -> CheckpointTuple:
        """Assemble a :class:`CheckpointTuple` from a parsed row dict."""
        checkpoint = value["checkpoint"]
        blob_map = self._load_blobs(value["channel_values"])
        inline_values = checkpoint.get("channel_values") or {}
        merged_checkpoint = {
            **checkpoint,
            "channel_values": {**inline_values, **blob_map},
        }
        parent_config: Optional[RunnableConfig] = None
        if value["parent_checkpoint_id"]:
            parent_config = {
                "configurable": {
                    "thread_id": value["thread_id"],
                    "checkpoint_ns": value["checkpoint_ns"],
                    "checkpoint_id": value["parent_checkpoint_id"],
                }
            }
        return CheckpointTuple(
            {
                "configurable": {
                    "thread_id": value["thread_id"],
                    "checkpoint_ns": value["checkpoint_ns"],
                    "checkpoint_id": value["checkpoint_id"],
                }
            },
            cast(Checkpoint, merged_checkpoint),
            value["metadata"],
            parent_config,
            self._load_writes(value["pending_writes"]),
        )

    def _maybe_migrate_pending_sends(
        self,
        cur: Any,
        values: Sequence[dict[str, Any]],
    ) -> None:
        """Fold TASKS-channel writes from parent checkpoints into channel_values.

        Older graph versions (``checkpoint["v"] < 4``) stored pending sends
        separately; the base loader expects them merged in.
        """
        to_migrate = [
            v
            for v in values
            if v["checkpoint"].get("v", 0) < 4 and v["parent_checkpoint_id"]
        ]
        if not to_migrate:
            return

        # ``list()`` results may span multiple threads; migrate each thread's
        # legacy checkpoints against its own ``checkpoint_writes`` rows.
        by_thread: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for v in to_migrate:
            by_thread[v["thread_id"]].append(v)

        for tid, group in by_thread.items():
            parent_ids = [v["parent_checkpoint_id"] for v in group]
            placeholders = ",".join(["%s"] * len(parent_ids))
            # SELECT_PENDING_SENDS_SQL is a template with a single ``%s`` for
            # the id list; rebuild it here with one placeholder per id.
            sends_sql = self.SELECT_PENDING_SENDS_SQL.replace(
                "(%s)", f"({placeholders})"
            )
            cur.execute(sends_sql, (tid, *parent_ids))
            rows = cur.fetchall()

            grouped_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for v in group:
                grouped_by_parent[v["parent_checkpoint_id"]].append(v)

            for send_row in rows:
                parent_id = _row_get(send_row, 0, "checkpoint_id")
                sends = _parse_sends(_row_get(send_row, 1, "sends"))
                for v in grouped_by_parent[parent_id]:
                    self._migrate_pending_sends(
                        sends, v["checkpoint"], v["channel_values"]
                    )

    # -------------------------------------------------------------- cursor
    class _CursorContext:
        def __init__(self, pool: Pool, lock: threading.Lock) -> None:
            self._pool = pool
            self._lock = lock
            self._conn: Any = None
            self._cur: Any = None

        def __enter__(self) -> Any:
            self._lock.acquire()
            try:
                self._conn = self._pool.connect()
                self._cur = self._conn.cursor()
            except BaseException:
                # __exit__ is not invoked when __enter__ raises; clean up
                # the partial connection and release the lock ourselves.
                if self._conn is not None:
                    try:
                        self._conn.close()
                    except Exception:
                        pass
                    self._conn = None
                self._lock.release()
                raise
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

    def _cursor(self) -> "SingleStoreSaver._CursorContext":
        return SingleStoreSaver._CursorContext(self.connection_pool, self._lock)


# ---------------------------------------------------------------- helpers


def _row_get(row: Any, index: int, name: str) -> Any:
    """Access a row column by name or index.

    Mirrors the store's helper so the saver works with either the default
    tuple cursor or ``results_type="dict"``.
    """
    if isinstance(row, dict):
        return row[name]
    return row[index]


def _as_dict(value: Any) -> dict[str, Any]:
    """Return a dict from a JSON column, whether or not the driver decoded it."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return cast("dict[str, Any]", json.loads(value))


def _as_list(value: Any) -> list[Any]:
    """Return a list from a JSON_AGG column, decoding a string if needed."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return cast("list[Any]", json.loads(value))


def _row_to_checkpoint_dict(row: Any) -> dict[str, Any]:
    """Convert a ``SELECT_SQL`` row into the shape ``_load_checkpoint_tuple``
    and ``_maybe_migrate_pending_sends`` expect. Column order matches
    ``BaseSingleStoreSaver.SELECT_SQL``.
    """
    return {
        "thread_id": _row_get(row, 0, "thread_id"),
        "checkpoint": _as_dict(_row_get(row, 1, "checkpoint")),
        "checkpoint_ns": _row_get(row, 2, "checkpoint_ns"),
        "checkpoint_id": _row_get(row, 3, "checkpoint_id"),
        "parent_checkpoint_id": _row_get(row, 4, "parent_checkpoint_id"),
        "metadata": _as_dict(_row_get(row, 5, "metadata")),
        "channel_values": _parse_channel_values(_row_get(row, 6, "channel_values")),
        "pending_writes": _parse_pending_writes(_row_get(row, 7, "pending_writes")),
    }


def _parse_channel_values(raw: Any) -> list[tuple[bytes, bytes, bytes]]:
    """Decode ``channel_values`` JSON_AGG output into blob triples.

    ``SELECT_SQL`` aggregates ``(channel, type, HEX(blob))`` per row; the
    base loader expects ``(bytes, bytes, bytes)``.
    """
    rows = _as_list(raw)
    result: list[tuple[bytes, bytes, bytes]] = []
    for entry in rows:
        channel, type_tag, hex_blob = entry[0], entry[1], entry[2]
        if channel is None:
            continue
        result.append(
            (
                channel.encode() if isinstance(channel, str) else channel,
                type_tag.encode() if isinstance(type_tag, str) else type_tag,
                bytes.fromhex(hex_blob) if isinstance(hex_blob, str) else hex_blob,
            )
        )
    return result


def _parse_pending_writes(raw: Any) -> list[tuple[bytes, bytes, bytes, bytes]]:
    """Decode ``pending_writes`` JSON_AGG output into write tuples.

    Each entry is ``(task_id, channel, type, HEX(blob))``.
    """
    rows = _as_list(raw)
    result: list[tuple[bytes, bytes, bytes, bytes]] = []
    for entry in rows:
        task_id, channel, type_tag, hex_blob = (
            entry[0],
            entry[1],
            entry[2],
            entry[3],
        )
        result.append(
            (
                task_id.encode() if isinstance(task_id, str) else task_id,
                channel.encode() if isinstance(channel, str) else channel,
                type_tag.encode() if isinstance(type_tag, str) else type_tag,
                bytes.fromhex(hex_blob) if isinstance(hex_blob, str) else hex_blob,
            )
        )
    return result


def _parse_sends(raw: Any) -> list[tuple[bytes, bytes]]:
    """Decode ``SELECT_PENDING_SENDS_SQL`` output into ``(type, blob)`` pairs.

    ``_migrate_pending_sends`` accepts either bytes or a hex string for the
    blob; we hand it the hex string untouched.
    """
    rows = _as_list(raw)
    result: list[tuple[bytes, bytes]] = []
    for entry in rows:
        type_tag, hex_blob = entry[0], entry[1]
        result.append(
            (
                type_tag.encode() if isinstance(type_tag, str) else type_tag,
                cast(bytes, hex_blob),
            )
        )
    return result
