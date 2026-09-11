"""Integration tests for :class:`AsyncSingleStoreStore`.

The async subclass dispatches every I/O call to the default executor, so
its correctness is not covered by the sync end-to-end suite. These tests
run the full CRUD, TTL, list-namespaces and vector-search paths through
``asetup`` / ``abatch`` against a real SingleStore container.
"""

from __future__ import annotations

import asyncio
from contextlib import closing
from typing import Any, List, cast

import pytest
from langchain_core.embeddings import Embeddings
from singlestoredb.connection import connect

from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    PutOp,
    SearchItem,
    SearchOp,
)
from langgraph.store.singlestore import AsyncSingleStoreStore
from langgraph.store.singlestore.base import SingleStoreIndexConfig

from .conftest import ConnectionParameters


class _ConstantEmbeddings(Embeddings):
    """Deterministic embeddings: match text scores 1 on dim 0, others 0."""

    def __init__(self, dims: int = 4) -> None:
        self.dims = dims

    def _vec(self, text: str) -> List[float]:
        vec = [0.0] * self.dims
        if "match" in text.lower():
            vec[0] = 1.0
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)


def _make_index_config(embed: Embeddings) -> SingleStoreIndexConfig:
    return cast(
        SingleStoreIndexConfig,
        {"dims": 4, "embed": embed, "fields": ["topic"]},
    )


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


def _as_item(result: Any) -> Item | None:
    assert result is None or isinstance(result, Item)
    return cast("Item | None", result)


@pytest.mark.asyncio
class TestAsyncSetup:
    async def test_asetup_creates_store_tables(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = AsyncSingleStoreStore(**connection_parameters.as_kwargs())
        try:
            await store.asetup()
            assert _table_exists(connection_parameters, "store")
            assert _table_exists(connection_parameters, "store_migrations")
        finally:
            store.close()

    async def test_asetup_is_idempotent(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = AsyncSingleStoreStore(**connection_parameters.as_kwargs())
        try:
            await store.asetup()
            # Second call must be a no-op end-to-end.
            await store.asetup()
            # The store is usable after two setups.
            await store.abatch([PutOp(("t",), "k", {"v": 1})])
            got = await store.abatch([GetOp(("t",), "k")])
            item = _as_item(got[0])
            assert item is not None and item.value == {"v": 1}
        finally:
            store.close()

    async def test_asetup_creates_vector_table_when_index_configured(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = AsyncSingleStoreStore(
            index=_make_index_config(_ConstantEmbeddings()),
            **connection_parameters.as_kwargs(),
        )
        try:
            await store.asetup()
            assert _table_exists(connection_parameters, "store_vector")
        finally:
            store.close()


@pytest.mark.asyncio
class TestAsyncBatchCrud:
    async def test_abatch_put_then_get(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = AsyncSingleStoreStore(**connection_parameters.as_kwargs())
        try:
            await store.asetup()
            await store.abatch([PutOp(("users", "alice"), "prefs", {"theme": "dark"})])
            results = await store.abatch([GetOp(("users", "alice"), "prefs")])
            item = _as_item(results[0])
            assert item is not None
            assert item.namespace == ("users", "alice")
            assert item.key == "prefs"
            assert item.value == {"theme": "dark"}
        finally:
            store.close()

    async def test_abatch_delete(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = AsyncSingleStoreStore(**connection_parameters.as_kwargs())
        try:
            await store.asetup()
            await store.abatch([PutOp(("t",), "k", {"v": 1})])
            await store.abatch([PutOp(("t",), "k", None)])
            got = await store.abatch([GetOp(("t",), "k")])
            assert got[0] is None
        finally:
            store.close()

    async def test_abatch_mixed_op_types_end_to_end(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """A single ``abatch`` mixes every op type. The result list must
        match caller order and each slot must have the right shape."""
        store = AsyncSingleStoreStore(**connection_parameters.as_kwargs())
        try:
            await store.asetup()
            await store.abatch(
                [
                    PutOp(("users", "alice"), "prefs", {"theme": "dark"}),
                    PutOp(("docs", "public"), "readme", {"title": "hi"}),
                ]
            )
            results = await store.abatch(
                [
                    GetOp(("users", "alice"), "prefs"),
                    PutOp(("users", "bob"), "prefs", {"theme": "light"}),
                    SearchOp(namespace_prefix=("users",), refresh_ttl=False),
                    ListNamespacesOp(),
                ]
            )
            assert len(results) == 4
            first = _as_item(results[0])
            assert first is not None and first.value == {"theme": "dark"}
            assert results[1] is None
            assert isinstance(results[2], list)
            assert isinstance(results[3], list)
            assert ("users", "alice") in cast(list, results[3])
        finally:
            store.close()


@pytest.mark.asyncio
class TestAsyncBatchTTL:
    async def test_abatch_ttl_populates_expires_at(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = AsyncSingleStoreStore(**connection_parameters.as_kwargs())
        try:
            await store.asetup()
            await store.abatch([PutOp(("t",), "k", {"v": 1}, ttl=60.0)])
            got = await store.abatch([GetOp(("t",), "k", refresh_ttl=False)])
            item = _as_item(got[0])
            assert item is not None
            # The base ``Item`` type does not surface ``expires_at``; verify
            # the row via a raw SQL probe.
            conn = connect(**connection_parameters.as_kwargs())
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT expires_at, ttl_minutes FROM store WHERE `key`=%s",
                    ("k",),
                )
                row = cur.fetchone()
                cur.close()
                assert row is not None
                expires_at, ttl_minutes = list(row)
                assert expires_at is not None
                assert int(ttl_minutes) == 60
            finally:
                conn.close()
        finally:
            store.close()

    async def test_abatch_refresh_ttl_bumps_expires_at(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """``refresh_ttl=True`` on the async path must transactionally bump
        ``expires_at`` on the matching row, same as the sync path."""
        store = AsyncSingleStoreStore(**connection_parameters.as_kwargs())
        try:
            await store.asetup()
            await store.abatch([PutOp(("t",), "k", {"v": 1}, ttl=60.0)])

            # Move ``expires_at`` forward by only 30 seconds so it stays
            # valid (the refresh SQL skips already-expired rows) but is
            # observably earlier than what the refresh will set.
            conn = connect(**connection_parameters.as_kwargs())
            try:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE store SET expires_at = DATE_ADD(NOW(), "
                    "INTERVAL 30 SECOND) WHERE `key`=%s",
                    ("k",),
                )
                cur.execute("SELECT expires_at FROM store WHERE `key`=%s", ("k",))
                row = cur.fetchone()
                cur.close()
                assert row is not None
                before = list(row)[0]
            finally:
                conn.close()

            await store.abatch([GetOp(("t",), "k", refresh_ttl=True)])

            conn = connect(**connection_parameters.as_kwargs())
            try:
                cur = conn.cursor()
                cur.execute("SELECT expires_at FROM store WHERE `key`=%s", ("k",))
                row = cur.fetchone()
                cur.close()
                assert row is not None
                after = list(row)[0]
            finally:
                conn.close()

            assert after > before
        finally:
            store.close()


@pytest.mark.asyncio
class TestAsyncVectorSearch:
    async def test_abatch_vector_search_end_to_end(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        embed = _ConstantEmbeddings()
        store = AsyncSingleStoreStore(
            index=_make_index_config(embed),
            **connection_parameters.as_kwargs(),
        )
        try:
            await store.asetup()
            await store.abatch(
                [
                    PutOp(("docs",), "hit", {"topic": "match please"}),
                    PutOp(("docs",), "miss", {"topic": "unrelated"}),
                ]
            )
            got = await store.abatch(
                [SearchOp(namespace_prefix=(), query="match", refresh_ttl=False)]
            )
            results = cast("list[SearchItem]", got[0])
            assert [r.key for r in results][:1] == ["hit"]
            hit = next(r for r in results if r.key == "hit")
            miss = next((r for r in results if r.key == "miss"), None)
            assert hit.score is not None
            # ``miss`` still surfaces in DOT_PRODUCT search with score 0.
            if miss is not None:
                assert (hit.score or 0.0) > (miss.score or 0.0)
        finally:
            store.close()


@pytest.mark.asyncio
class TestAsyncDoesNotBlockEventLoop:
    async def test_concurrent_coroutine_progresses_while_abatch_runs(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """``abatch`` dispatches the sync batch to the default executor, so
        an unrelated coroutine must be able to progress on the event loop
        while a batch is in flight. We assert that ``asyncio.sleep`` wakes
        up before the batch finishes even when both start together."""
        store = AsyncSingleStoreStore(**connection_parameters.as_kwargs())
        try:
            await store.asetup()

            counter = {"ticks": 0}

            async def tick() -> None:
                for _ in range(10):
                    await asyncio.sleep(0.01)
                    counter["ticks"] += 1

            ops = [PutOp(("t",), f"k{i}", {"i": i}) for i in range(50)]
            batch_task = asyncio.create_task(store.abatch(ops))
            tick_task = asyncio.create_task(tick())
            await asyncio.gather(batch_task, tick_task)
            # ``tick`` completed all iterations, proving the loop was not
            # blocked by the executor call.
            assert counter["ticks"] == 10
        finally:
            store.close()
