"""Integration tests for :class:`SingleStoreSaver`.

Runs a real SingleStore container (see ``conftest.py``) and exercises the
checkpoint saver end-to-end: schema migrations, ``put``/``get_tuple``,
``put_writes``, ``list``, ``delete_thread``, plus the async variants and
lifecycle behaviour.
"""

from __future__ import annotations

import asyncio
from contextlib import closing
from typing import Any, cast

import pytest
from langchain_core.runnables import RunnableConfig
from singlestore_langchain_core._connection import QueueConnectionPool
from singlestoredb.connection import connect

from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    empty_checkpoint,
)
from langgraph.checkpoint.singlestore import SingleStoreSaver

from .conftest import ConnectionParameters

# ---------------------------------------------------------------- helpers


def _make_checkpoint(
    *,
    checkpoint_id: str,
    channel_values: dict[str, Any] | None = None,
) -> Checkpoint:
    """Build a checkpoint with a deterministic id and versions that match
    every key in ``channel_values`` so ``put`` writes matching blob rows."""
    ck = empty_checkpoint()
    ck["id"] = checkpoint_id
    if channel_values:
        ck["channel_values"] = dict(channel_values)
        ck["channel_versions"] = {k: "1" for k in channel_values}
    return ck


def _config(
    thread_id: str,
    *,
    ns: str = "",
    checkpoint_id: str | None = None,
) -> RunnableConfig:
    cfg: dict[str, Any] = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ns}
    }
    if checkpoint_id is not None:
        cfg["configurable"]["checkpoint_id"] = checkpoint_id
    return cast(RunnableConfig, cfg)


def _count_rows(params: ConnectionParameters, table: str) -> int:
    """Return ``SELECT COUNT(*) FROM <table>`` via a raw SQL connection."""
    conn = connect(**params.as_kwargs())
    try:
        with closing(conn.cursor()) as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            row = cast("tuple[Any, ...] | None", cur.fetchone())
            assert row is not None
            return int(row[0])
    finally:
        conn.close()


def _table_exists(params: ConnectionParameters, table: str) -> bool:
    conn = connect(**params.as_kwargs())
    try:
        with closing(conn.cursor()) as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = %s AND table_name = %s",
                (params.database, table),
            )
            return cur.fetchone() is not None
    finally:
        conn.close()


def _migration_versions(params: ConnectionParameters) -> list[int]:
    conn = connect(**params.as_kwargs())
    try:
        with closing(conn.cursor()) as cur:
            cur.execute("SELECT v FROM checkpoint_migrations ORDER BY v")
            return [int(list(row)[0]) for row in cur.fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------------------- setup


class TestSetup:
    def test_setup_creates_all_checkpoint_tables(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            for table in (
                "checkpoint_migrations",
                "checkpoints",
                "checkpoint_blobs",
                "checkpoint_writes",
            ):
                assert _table_exists(connection_parameters, table), table
        finally:
            saver.close()

    def test_setup_is_idempotent(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Repeated ``setup()`` must be safe: no duplicate migration rows,
        no re-execution errors."""
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            first = _migration_versions(connection_parameters)
            assert first == sorted(set(first)) and first

            saver.setup()
            second = _migration_versions(connection_parameters)
            assert second == first
        finally:
            saver.close()


# ------------------------------------------------------------- put / get


class TestPutGetTuple:
    def test_empty_checkpoint_roundtrip(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            cfg = _config("t1")
            ck = _make_checkpoint(checkpoint_id="cp-1")
            metadata: CheckpointMetadata = {"source": "input"}

            next_cfg = saver.put(cfg, ck, metadata, {})
            assert next_cfg["configurable"]["checkpoint_id"] == "cp-1"

            result = saver.get_tuple(next_cfg)
            assert isinstance(result, CheckpointTuple)
            assert result.checkpoint["id"] == "cp-1"
            assert result.checkpoint["channel_values"] == {}
            assert result.metadata == metadata
            assert result.parent_config is None
            assert result.pending_writes == []
        finally:
            saver.close()

    def test_primitive_channel_values_stay_inline(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Primitives (``None``, ``str``, ``int``, ``float``, ``bool``) must
        be kept in the checkpoint JSON — no blob rows written for them."""
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            cfg = _config("t1")
            ck = _make_checkpoint(
                checkpoint_id="cp-1",
                channel_values={"s": "hi", "i": 1, "f": 1.5, "b": True, "n": None},
            )
            saver.put(cfg, ck, {"source": "input"}, ck["channel_versions"])

            assert _count_rows(connection_parameters, "checkpoint_blobs") == 0

            got = saver.get_tuple(cfg)
            assert got is not None
            assert got.checkpoint["channel_values"] == {
                "s": "hi",
                "i": 1,
                "f": 1.5,
                "b": True,
                "n": None,
            }
        finally:
            saver.close()

    def test_non_primitive_values_go_to_blobs_and_round_trip(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            cfg = _config("t1")
            ck = _make_checkpoint(
                checkpoint_id="cp-1",
                channel_values={"obj": {"nested": [1, 2, 3]}, "prim": "keep-inline"},
            )
            saver.put(cfg, ck, {"source": "input"}, ck["channel_versions"])

            # One blob row for the dict, none for the primitive.
            assert _count_rows(connection_parameters, "checkpoint_blobs") == 1

            got = saver.get_tuple(cfg)
            assert got is not None
            assert got.checkpoint["channel_values"] == {
                "obj": {"nested": [1, 2, 3]},
                "prim": "keep-inline",
            }
        finally:
            saver.close()

    def test_get_tuple_by_checkpoint_id_returns_specific_row(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            saver.put(_config("t1"), _make_checkpoint(checkpoint_id="cp-1"), {}, {})
            # Parent chain: cp-2's parent is cp-1.
            saver.put(
                _config("t1", checkpoint_id="cp-1"),
                _make_checkpoint(checkpoint_id="cp-2"),
                {"step": 2},
                {},
            )

            got = saver.get_tuple(_config("t1", checkpoint_id="cp-1"))
            assert got is not None
            assert got.checkpoint["id"] == "cp-1"
            assert got.parent_config is None

            got_child = saver.get_tuple(_config("t1", checkpoint_id="cp-2"))
            assert got_child is not None
            assert got_child.checkpoint["id"] == "cp-2"
            assert got_child.parent_config is not None
            assert got_child.parent_config["configurable"]["checkpoint_id"] == "cp-1"
        finally:
            saver.close()

    def test_get_tuple_without_id_returns_latest(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            saver.put(_config("t1"), _make_checkpoint(checkpoint_id="cp-1"), {}, {})
            saver.put(
                _config("t1", checkpoint_id="cp-1"),
                _make_checkpoint(checkpoint_id="cp-2"),
                {},
                {},
            )
            saver.put(
                _config("t1", checkpoint_id="cp-2"),
                _make_checkpoint(checkpoint_id="cp-3"),
                {},
                {},
            )

            got = saver.get_tuple(_config("t1"))
            assert got is not None
            assert got.checkpoint["id"] == "cp-3"
        finally:
            saver.close()

    def test_get_tuple_missing_thread_returns_none(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            assert saver.get_tuple(_config("unknown-thread")) is None
        finally:
            saver.close()

    def test_put_same_id_updates_in_place(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """``UPSERT`` on ``(thread_id, ns, checkpoint_id)`` — a second put
        with the same id overwrites, not duplicates."""
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            cfg = _config("t1")
            saver.put(cfg, _make_checkpoint(checkpoint_id="cp-1"), {"step": 1}, {})
            saver.put(cfg, _make_checkpoint(checkpoint_id="cp-1"), {"step": 2}, {})

            assert _count_rows(connection_parameters, "checkpoints") == 1
            got = saver.get_tuple(cfg)
            assert got is not None
            assert cast(dict, got.metadata) == {"step": 2}
        finally:
            saver.close()

    def test_namespace_isolation(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Checkpoints in different ``checkpoint_ns`` never leak across."""
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            saver.put(
                _config("t1", ns="a"), _make_checkpoint(checkpoint_id="a-1"), {}, {}
            )
            saver.put(
                _config("t1", ns="b"), _make_checkpoint(checkpoint_id="b-1"), {}, {}
            )

            got_a = saver.get_tuple(_config("t1", ns="a"))
            got_b = saver.get_tuple(_config("t1", ns="b"))
            assert got_a is not None and got_a.checkpoint["id"] == "a-1"
            assert got_b is not None and got_b.checkpoint["id"] == "b-1"
        finally:
            saver.close()


# ------------------------------------------------------------- put_writes


class TestPutWrites:
    def test_writes_surface_as_pending_writes(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            cfg = _config("t1")
            saver.put(cfg, _make_checkpoint(checkpoint_id="cp-1"), {}, {})

            child_cfg = _config("t1", checkpoint_id="cp-1")
            saver.put_writes(child_cfg, [("out", {"payload": 1})], "task-1")

            got = saver.get_tuple(child_cfg)
            assert got is not None
            assert got.pending_writes == [("task-1", "out", {"payload": 1})]
        finally:
            saver.close()

    def test_reserved_channel_writes_upsert_over_duplicate_idx(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Reserved channels in ``WRITES_IDX_MAP`` land on fixed negative
        idx values; a second ``put_writes`` for the same reserved channel
        must overwrite, not raise or duplicate."""
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            cfg = _config("t1")
            saver.put(cfg, _make_checkpoint(checkpoint_id="cp-1"), {}, {})
            child_cfg = _config("t1", checkpoint_id="cp-1")

            saver.put_writes(child_cfg, [("__error__", "first")], "task-1")
            saver.put_writes(child_cfg, [("__error__", "second")], "task-1")

            assert _count_rows(connection_parameters, "checkpoint_writes") == 1
            got = saver.get_tuple(child_cfg)
            assert got is not None
            assert got.pending_writes == [("task-1", "__error__", "second")]
        finally:
            saver.close()

    def test_non_reserved_writes_are_insert_ignore(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """A second ``put_writes`` for the same ``(task, idx)`` on a
        non-reserved channel must be dropped, preserving the first value."""
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            cfg = _config("t1")
            saver.put(cfg, _make_checkpoint(checkpoint_id="cp-1"), {}, {})
            child_cfg = _config("t1", checkpoint_id="cp-1")

            saver.put_writes(child_cfg, [("out", "first")], "task-1")
            saver.put_writes(child_cfg, [("out", "second")], "task-1")

            got = saver.get_tuple(child_cfg)
            assert got is not None
            # Only the first survives — INSERT IGNORE dropped the duplicate.
            assert got.pending_writes == [("task-1", "out", "first")]
        finally:
            saver.close()

    def test_empty_writes_is_noop(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            cfg = _config("t1")
            saver.put(cfg, _make_checkpoint(checkpoint_id="cp-1"), {}, {})
            saver.put_writes(_config("t1", checkpoint_id="cp-1"), [], "task-1")
            assert _count_rows(connection_parameters, "checkpoint_writes") == 0
        finally:
            saver.close()

    def test_writes_isolated_per_checkpoint(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            saver.put(_config("t1"), _make_checkpoint(checkpoint_id="cp-1"), {}, {})
            saver.put(
                _config("t1", checkpoint_id="cp-1"),
                _make_checkpoint(checkpoint_id="cp-2"),
                {},
                {},
            )

            saver.put_writes(_config("t1", checkpoint_id="cp-1"), [("out", "a")], "t")
            saver.put_writes(_config("t1", checkpoint_id="cp-2"), [("out", "b")], "t")

            got_1 = saver.get_tuple(_config("t1", checkpoint_id="cp-1"))
            got_2 = saver.get_tuple(_config("t1", checkpoint_id="cp-2"))
            assert got_1 is not None and got_1.pending_writes == [("t", "out", "a")]
            assert got_2 is not None and got_2.pending_writes == [("t", "out", "b")]
        finally:
            saver.close()


# ----------------------------------------------------------------- list


class TestList:
    def _seed_three_checkpoints(self, saver: SingleStoreSaver) -> None:
        """Seed cp-1 <- cp-2 <- cp-3 on thread ``t1`` with varying metadata."""
        saver.put(
            _config("t1"),
            _make_checkpoint(checkpoint_id="cp-1"),
            {"step": 1, "source": "input"},
            {},
        )
        saver.put(
            _config("t1", checkpoint_id="cp-1"),
            _make_checkpoint(checkpoint_id="cp-2"),
            {"step": 2, "source": "loop"},
            {},
        )
        saver.put(
            _config("t1", checkpoint_id="cp-2"),
            _make_checkpoint(checkpoint_id="cp-3"),
            {"step": 3, "source": "loop"},
            {},
        )

    def test_list_orders_newest_first(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            self._seed_three_checkpoints(saver)
            got = list(saver.list(_config("t1")))
            assert [t.checkpoint["id"] for t in got] == ["cp-3", "cp-2", "cp-1"]
        finally:
            saver.close()

    def test_list_respects_limit(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            self._seed_three_checkpoints(saver)
            got = list(saver.list(_config("t1"), limit=2))
            assert [t.checkpoint["id"] for t in got] == ["cp-3", "cp-2"]
        finally:
            saver.close()

    def test_list_filters_by_thread_id(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            self._seed_three_checkpoints(saver)
            saver.put(_config("t2"), _make_checkpoint(checkpoint_id="x-1"), {}, {})

            for t in saver.list(_config("t1")):
                assert t.config["configurable"]["thread_id"] == "t1"
        finally:
            saver.close()

    def test_list_filters_by_metadata(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            self._seed_three_checkpoints(saver)
            got = list(saver.list(_config("t1"), filter={"source": "loop"}))
            assert {t.checkpoint["id"] for t in got} == {"cp-2", "cp-3"}
        finally:
            saver.close()

    def test_list_before_excludes_at_and_after(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """``before`` uses strict ``<`` so the anchor itself is excluded."""
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            self._seed_three_checkpoints(saver)
            got = list(
                saver.list(_config("t1"), before=_config("t1", checkpoint_id="cp-3"))
            )
            assert [t.checkpoint["id"] for t in got] == ["cp-2", "cp-1"]
        finally:
            saver.close()

    def test_list_with_no_config_returns_all(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            self._seed_three_checkpoints(saver)
            saver.put(_config("t2"), _make_checkpoint(checkpoint_id="x-1"), {}, {})
            got = list(saver.list(None))
            assert len(got) == 4
        finally:
            saver.close()

    def test_list_empty_thread_yields_nothing(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            assert list(saver.list(_config("t1"))) == []
        finally:
            saver.close()


# ------------------------------------------------------------- delete_thread


class TestDeleteThread:
    def test_delete_removes_all_rows_for_thread(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            saver.put(
                _config("t1"),
                _make_checkpoint(
                    checkpoint_id="cp-1", channel_values={"obj": {"x": 1}}
                ),
                {},
                {"obj": "1"},
            )
            saver.put_writes(
                _config("t1", checkpoint_id="cp-1"), [("out", 1)], "task-1"
            )
            # Second thread should be untouched.
            saver.put(_config("t2"), _make_checkpoint(checkpoint_id="cp-x"), {}, {})

            saver.delete_thread("t1")

            assert _count_rows(connection_parameters, "checkpoints") == 1
            assert _count_rows(connection_parameters, "checkpoint_blobs") == 0
            assert _count_rows(connection_parameters, "checkpoint_writes") == 0
            assert saver.get_tuple(_config("t1")) is None

            got_t2 = saver.get_tuple(_config("t2"))
            assert got_t2 is not None and got_t2.checkpoint["id"] == "cp-x"
        finally:
            saver.close()

    def test_delete_missing_thread_is_noop(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            saver.delete_thread("no-such-thread")
        finally:
            saver.close()


# ----------------------------------------------------------------- async


class TestAsync:
    def test_aput_aget_tuple_round_trip(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()

            async def run() -> CheckpointTuple | None:
                cfg = _config("t1")
                await saver.aput(
                    cfg,
                    _make_checkpoint(checkpoint_id="cp-1"),
                    {"source": "input"},
                    {},
                )
                return await saver.aget_tuple(cfg)

            got = asyncio.run(run())
            assert got is not None
            assert got.checkpoint["id"] == "cp-1"
            assert cast(dict, got.metadata) == {"source": "input"}
        finally:
            saver.close()

    def test_alist_yields_async_iterator(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            saver.put(_config("t1"), _make_checkpoint(checkpoint_id="cp-1"), {}, {})
            saver.put(
                _config("t1", checkpoint_id="cp-1"),
                _make_checkpoint(checkpoint_id="cp-2"),
                {},
                {},
            )

            async def collect() -> list[CheckpointTuple]:
                out: list[CheckpointTuple] = []
                async for item in saver.alist(_config("t1")):
                    out.append(item)
                return out

            items = asyncio.run(collect())
            assert [t.checkpoint["id"] for t in items] == ["cp-2", "cp-1"]
        finally:
            saver.close()

    def test_aput_writes_appears_in_pending_writes(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        try:
            saver.setup()
            saver.put(_config("t1"), _make_checkpoint(checkpoint_id="cp-1"), {}, {})

            async def run() -> CheckpointTuple | None:
                await saver.aput_writes(
                    _config("t1", checkpoint_id="cp-1"),
                    [("out", "async-val")],
                    "task-1",
                )
                return await saver.aget_tuple(_config("t1", checkpoint_id="cp-1"))

            got = asyncio.run(run())
            assert got is not None
            assert got.pending_writes == [("task-1", "out", "async-val")]
        finally:
            saver.close()


# ----------------------------------------------------------------- lifecycle


class TestLifecycle:
    def test_close_is_idempotent_on_owned_pool(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        saver = SingleStoreSaver(**connection_parameters.as_kwargs())
        saver.setup()
        saver.close()
        saver.close()

    def test_close_leaves_injected_pool_usable(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        pool = QueueConnectionPool(
            pool_size=1,
            max_overflow=1,
            timeout=5,
            connection_kwargs=connection_parameters.as_kwargs(),
        )
        try:
            saver = SingleStoreSaver(connection_pool=pool)
            saver.setup()
            saver.close()

            conn = pool.connect()
            try:
                cur = conn.cursor()
                cur.execute("SELECT 1")
                row = cur.fetchone()
                assert row is not None
                assert int(list(row)[0]) == 1
                cur.close()
            finally:
                conn.close()
        finally:
            pool.dispose()

    def test_end_to_end_via_injected_connection(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        conn = connect(**connection_parameters.as_kwargs())
        try:
            saver = SingleStoreSaver(connection=conn)
            try:
                saver.setup()
                cfg = _config("t1")
                saver.put(cfg, _make_checkpoint(checkpoint_id="cp-1"), {}, {})
                got = saver.get_tuple(cfg)
                assert got is not None
                assert got.checkpoint["id"] == "cp-1"
            finally:
                saver.close()
        finally:
            conn.close()

    def test_from_conn_string_builds_working_saver(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """``from_conn_string`` must forward the URL to the driver and yield
        a fully functional saver: setup + put + get_tuple + delete."""
        p = connection_parameters
        conn_string = f"{p.user}:{p.password}@{p.host}:{p.port}/{p.database}"

        saver = SingleStoreSaver.from_conn_string(conn_string)
        try:
            saver.setup()
            cfg = _config("t1")
            saver.put(cfg, _make_checkpoint(checkpoint_id="cp-1"), {}, {})

            got = saver.get_tuple(cfg)
            assert got is not None
            assert got.checkpoint["id"] == "cp-1"

            saver.delete_thread("t1")
            assert saver.get_tuple(cfg) is None
        finally:
            saver.close()

    def test_from_conn_string_accepts_custom_serde(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """The ``serde`` keyword must reach ``BaseCheckpointSaver.__init__``
        so callers can plug a non-default serializer."""
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        p = connection_parameters
        conn_string = f"{p.user}:{p.password}@{p.host}:{p.port}/{p.database}"
        custom_serde = JsonPlusSerializer()

        saver = SingleStoreSaver.from_conn_string(conn_string, serde=custom_serde)
        try:
            assert saver.serde is custom_serde
        finally:
            saver.close()

    def test_from_conn_string_rejects_bad_url(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """A URL that points nowhere must surface as an error at first use,
        not silently succeed."""
        saver = SingleStoreSaver.from_conn_string(
            "no-such-user:no-such-pass@127.0.0.1:1/no-such-db"
        )
        try:
            with pytest.raises(Exception):
                saver.setup()
        finally:
            saver.close()
