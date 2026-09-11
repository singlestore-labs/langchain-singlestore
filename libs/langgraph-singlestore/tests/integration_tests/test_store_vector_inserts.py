"""Integration tests for inserting vector embeddings via ``SingleStoreStore``.

Covers ``PutOp`` behaviour against the ``store_vector`` table:

* Store initialised **without** an index — vector table absent, no vector
  rows are ever written.
* Store initialised **with** an index, mix of ``PutOp(index=False)`` and
  default ``PutOp`` — only the defaulted puts produce ``store_vector`` rows.
* Store initialised with a fully specified :class:`ANNIndexConfig` (index
  type + tuned parameters) and explicit ``fields`` — expected number of
  vector rows with correct ``(prefix, key, field_name)`` triples.
* ``PutOp(index=[...])`` overrides the store-wide fields for that op.
* Multi-text extraction produces ``field.0``, ``field.1`` ... rows.
* Update-in-place refreshes ``updated_at`` on ``store_vector``.
* ``PutOp(value=None)`` deletes matching ``store_vector`` rows.
"""

from __future__ import annotations

import time
from contextlib import closing
from typing import Any, List, cast

import pytest
from langchain_core.embeddings import Embeddings
from singlestore_langchain_core import IVF_FLATIndexConfig
from singlestoredb.connection import connect

from langgraph.store.base import PutOp
from langgraph.store.singlestore import SingleStoreStore
from langgraph.store.singlestore.base import SingleStoreIndexConfig

from .conftest import ConnectionParameters


class _CountingEmbeddings(Embeddings):
    """Deterministic embeddings that record every request for verification."""

    def __init__(self, dims: int) -> None:
        self.dims = dims
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.calls.append(list(texts))
        return [[float((i + len(t)) % 5) for i in range(self.dims)] for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return [float((i + len(text)) % 5) for i in range(self.dims)]


def _make_index_config(
    embed: Embeddings,
    *,
    dims: int = 8,
    fields: Any = None,
    ann_index_config: Any = None,
) -> SingleStoreIndexConfig:
    cfg: dict[str, Any] = {"dims": dims, "embed": embed}
    if fields is not None:
        cfg["fields"] = fields
    if ann_index_config is not None:
        cfg["ann_index_config"] = ann_index_config
    return cast(SingleStoreIndexConfig, cfg)


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


def _fetch_vector_rows(
    params: ConnectionParameters,
) -> list[tuple[str, str, str]]:
    """Return ``(prefix, key, field_name)`` for every row in ``store_vector``."""
    conn = connect(**params.as_kwargs())
    try:
        with closing(conn.cursor()) as cur:
            cur.execute(
                "SELECT prefix, `key`, field_name FROM store_vector "
                "ORDER BY prefix, `key`, field_name"
            )
            fetched = cast("list[tuple[Any, ...]]", cur.fetchall())
            return [(str(r[0]), str(r[1]), str(r[2])) for r in fetched]
    finally:
        conn.close()


def _fetch_vector_updated_at(
    params: ConnectionParameters, prefix: str, key: str, field_name: str
) -> Any:
    conn = connect(**params.as_kwargs())
    try:
        with closing(conn.cursor()) as cur:
            cur.execute(
                "SELECT updated_at FROM store_vector "
                "WHERE prefix = %s AND `key` = %s AND field_name = %s",
                (prefix, key, field_name),
            )
            row = cast("tuple[Any, ...] | None", cur.fetchone())
            assert row is not None
            return row[0]
    finally:
        conn.close()


def _count_store_rows(params: ConnectionParameters) -> int:
    conn = connect(**params.as_kwargs())
    try:
        with closing(conn.cursor()) as cur:
            cur.execute("SELECT COUNT(*) FROM store")
            row = cast("tuple[Any, ...] | None", cur.fetchone())
            assert row is not None
            return int(row[0])
    finally:
        conn.close()


class TestPutOpsWithoutIndex:
    """Store constructed without an ``index=`` never writes to ``store_vector``."""

    def test_put_without_index_does_not_create_vector_table(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            store.batch([PutOp(("users", "alice"), "prefs", {"summary": "loves jazz"})])
            assert not _table_exists(connection_parameters, "store_vector")
            assert _count_store_rows(connection_parameters) == 1
        finally:
            store.close()

    def test_put_with_explicit_index_list_is_ignored_when_no_config(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """A caller may pass ``index=["summary"]`` on the op even if the store
        has no ``index_config``. It must be silently dropped (no vector rows
        written, no table created)."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            store.batch(
                [
                    PutOp(
                        ("users", "alice"),
                        "prefs",
                        {"summary": "loves jazz"},
                        index=["summary"],
                    )
                ]
            )
            assert not _table_exists(connection_parameters, "store_vector")
        finally:
            store.close()


class TestPutOpsIndexFalse:
    """``PutOp(index=False)`` opts out of vector-index writes."""

    def test_index_false_skips_vector_row(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        embed = _CountingEmbeddings(dims=8)
        store = SingleStoreStore(
            index=_make_index_config(embed, dims=8, fields=["summary"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            store.batch(
                [
                    PutOp(
                        ("users", "alice"),
                        "prefs",
                        {"summary": "loves jazz"},
                        index=False,
                    )
                ]
            )
            assert _fetch_vector_rows(connection_parameters) == []
            assert embed.calls == []
        finally:
            store.close()

    def test_mixed_batch_only_defaulted_puts_are_indexed(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        embed = _CountingEmbeddings(dims=8)
        store = SingleStoreStore(
            index=_make_index_config(embed, dims=8, fields=["summary"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            store.batch(
                [
                    PutOp(
                        ("users", "alice"),
                        "prefs",
                        {"summary": "alice-summary"},
                    ),
                    PutOp(
                        ("users", "bob"),
                        "prefs",
                        {"summary": "bob-summary"},
                        index=False,
                    ),
                    PutOp(
                        ("users", "carol"),
                        "prefs",
                        {"summary": "carol-summary"},
                    ),
                ]
            )
            rows = _fetch_vector_rows(connection_parameters)
            assert rows == [
                ("users/alice", "prefs", "summary"),
                ("users/carol", "prefs", "summary"),
            ]
            # ``embed_documents`` is called once with only the two indexed texts.
            assert len(embed.calls) == 1
            assert sorted(embed.calls[0]) == ["alice-summary", "carol-summary"]
            # All three rows still land in the base ``store`` table.
            assert _count_store_rows(connection_parameters) == 3
        finally:
            store.close()


class TestPutOpsWithFullySpecifiedIndex:
    """Store with a tuned :class:`ANNIndexConfig` and explicit fields."""

    def test_fully_specified_config_writes_expected_vector_rows(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        embed = _CountingEmbeddings(dims=8)
        ann = IVF_FLATIndexConfig(
            index_type="IVF_FLAT",
            nlist=128,
            nprobe=8,
        )
        store = SingleStoreStore(
            index=_make_index_config(
                embed,
                dims=8,
                fields=["summary", "body.text"],
                ann_index_config=ann,
            ),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            store.batch(
                [
                    PutOp(
                        ("docs",),
                        "doc-1",
                        {
                            "summary": "first-summary",
                            "body": {"text": "first-body"},
                        },
                    ),
                    PutOp(
                        ("docs",),
                        "doc-2",
                        {
                            "summary": "second-summary",
                            "body": {"text": "second-body"},
                        },
                    ),
                ]
            )
            rows = _fetch_vector_rows(connection_parameters)
            assert rows == [
                ("docs", "doc-1", "body.text"),
                ("docs", "doc-1", "summary"),
                ("docs", "doc-2", "body.text"),
                ("docs", "doc-2", "summary"),
            ]
            assert len(embed.calls) == 1
            assert sorted(embed.calls[0]) == [
                "first-body",
                "first-summary",
                "second-body",
                "second-summary",
            ]
        finally:
            store.close()

    def test_per_op_index_list_overrides_store_fields(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        embed = _CountingEmbeddings(dims=8)
        store = SingleStoreStore(
            index=_make_index_config(embed, dims=8, fields=["summary"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            store.batch(
                [
                    PutOp(
                        ("docs",),
                        "doc-1",
                        {"summary": "sum-1", "title": "title-1"},
                        index=["title"],
                    ),
                ]
            )
            rows = _fetch_vector_rows(connection_parameters)
            assert rows == [("docs", "doc-1", "title")]
            assert embed.calls == [["title-1"]]
        finally:
            store.close()

    def test_root_field_embeds_full_json_document(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """With ``fields=["$"]`` the entire JSON dump is embedded once."""
        embed = _CountingEmbeddings(dims=8)
        store = SingleStoreStore(
            index=_make_index_config(embed, dims=8, fields=["$"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            store.batch(
                [
                    PutOp(
                        ("docs",),
                        "doc-1",
                        {"summary": "sum", "title": "title"},
                    ),
                ]
            )
            rows = _fetch_vector_rows(connection_parameters)
            assert rows == [("docs", "doc-1", "$")]
            assert len(embed.calls) == 1
            assert len(embed.calls[0]) == 1
            payload = embed.calls[0][0]
            assert "summary" in payload
            assert "title" in payload
        finally:
            store.close()

    def test_multi_text_field_produces_indexed_rows(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """A field that resolves to multiple texts produces ``field.0``,
        ``field.1`` … rows."""
        embed = _CountingEmbeddings(dims=8)
        store = SingleStoreStore(
            index=_make_index_config(embed, dims=8, fields=["items[*].name"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            store.batch(
                [
                    PutOp(
                        ("carts",),
                        "cart-1",
                        {
                            "items": [
                                {"name": "apple"},
                                {"name": "banana"},
                                {"name": "cherry"},
                            ]
                        },
                    ),
                ]
            )
            rows = _fetch_vector_rows(connection_parameters)
            assert rows == [
                ("carts", "cart-1", "items[*].name.0"),
                ("carts", "cart-1", "items[*].name.1"),
                ("carts", "cart-1", "items[*].name.2"),
            ]
            assert len(embed.calls) == 1
            assert sorted(embed.calls[0]) == ["apple", "banana", "cherry"]
        finally:
            store.close()

    def test_missing_field_produces_no_vector_row(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        embed = _CountingEmbeddings(dims=8)
        store = SingleStoreStore(
            index=_make_index_config(
                embed,
                dims=8,
                fields=["summary", "not_present"],
            ),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            store.batch(
                [
                    PutOp(
                        ("docs",),
                        "doc-1",
                        {"summary": "only-summary"},
                    ),
                ]
            )
            rows = _fetch_vector_rows(connection_parameters)
            assert rows == [("docs", "doc-1", "summary")]
            assert embed.calls == [["only-summary"]]
        finally:
            store.close()

    def test_reinsert_updates_vector_row(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        embed = _CountingEmbeddings(dims=8)
        store = SingleStoreStore(
            index=_make_index_config(embed, dims=8, fields=["summary"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            store.batch([PutOp(("docs",), "doc-1", {"summary": "v1"})])
            first_updated = _fetch_vector_updated_at(
                connection_parameters, "docs", "doc-1", "summary"
            )
            # ``updated_at`` has 1s resolution; sleep long enough to observe.
            time.sleep(1.1)
            store.batch([PutOp(("docs",), "doc-1", {"summary": "v2"})])
            rows = _fetch_vector_rows(connection_parameters)
            assert rows == [("docs", "doc-1", "summary")]
            second_updated = _fetch_vector_updated_at(
                connection_parameters, "docs", "doc-1", "summary"
            )
            assert second_updated >= first_updated
            assert embed.calls == [["v1"], ["v2"]]
        finally:
            store.close()

    def test_delete_removes_vector_row(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        embed = _CountingEmbeddings(dims=8)
        store = SingleStoreStore(
            index=_make_index_config(embed, dims=8, fields=["summary"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            store.batch(
                [
                    PutOp(("docs",), "doc-1", {"summary": "keep"}),
                    PutOp(("docs",), "doc-2", {"summary": "gone"}),
                ]
            )
            assert len(_fetch_vector_rows(connection_parameters)) == 2

            store.batch([PutOp(("docs",), "doc-2", None)])

            rows = _fetch_vector_rows(connection_parameters)
            assert rows == [("docs", "doc-1", "summary")]
            assert _count_store_rows(connection_parameters) == 1
        finally:
            store.close()


class TestPutOpsMissingEmbeddingsConfig:
    """Sanity: this scenario is unreachable through construction, but the
    guard clause in ``_batch_put_ops`` is still worth pinning down."""

    def test_missing_embeddings_after_manual_reset_raises(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        embed = _CountingEmbeddings(dims=8)
        store = SingleStoreStore(
            index=_make_index_config(embed, dims=8, fields=["summary"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            store.embeddings = None
            with pytest.raises(ValueError, match="Embedding configuration is required"):
                store.batch([PutOp(("docs",), "doc-1", {"summary": "hello"})])
        finally:
            store.close()


class TestSweepTTLWithVectorIndex:
    """``sweep_ttl`` must also purge ``store_vector`` rows whose base rows
    have expired. The base sweep tests use a non-vector store, so this is
    the only place the ``_DELETE_EXPIRED_FROM_STORE_VECTOR`` branch runs."""

    def test_sweep_ttl_deletes_vector_rows_for_expired_items(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        embed = _CountingEmbeddings(dims=8)
        store = SingleStoreStore(
            index=_make_index_config(embed, dims=8, fields=["summary"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            store.batch(
                [
                    PutOp(("docs",), "keep", {"summary": "still here"}),
                    PutOp(
                        ("docs",),
                        "gone",
                        {"summary": "will expire"},
                        ttl=1.0,
                    ),
                ]
            )
            assert {r[1] for r in _fetch_vector_rows(connection_parameters)} == {
                "keep",
                "gone",
            }

            # Force ``gone`` past its ``expires_at`` and sweep.
            conn = connect(**connection_parameters.as_kwargs())
            try:
                with closing(conn.cursor()) as cur:
                    cur.execute(
                        "UPDATE store SET expires_at = DATE_SUB(NOW(), "
                        "INTERVAL 1 MINUTE) WHERE `key` = %s",
                        ("gone",),
                    )
            finally:
                conn.close()

            deleted = store.sweep_ttl()
            assert deleted >= 1

            surviving_keys = {r[1] for r in _fetch_vector_rows(connection_parameters)}
            assert surviving_keys == {"keep"}
            assert _count_store_rows(connection_parameters) == 1
        finally:
            store.close()

    def test_sweep_ttl_on_vector_store_with_nothing_expired_is_noop(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        embed = _CountingEmbeddings(dims=8)
        store = SingleStoreStore(
            index=_make_index_config(embed, dims=8, fields=["summary"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            store.batch(
                [
                    PutOp(("docs",), "a", {"summary": "one"}),
                    PutOp(("docs",), "b", {"summary": "two"}, ttl=60.0),
                ]
            )
            before_vector = _fetch_vector_rows(connection_parameters)
            before_store = _count_store_rows(connection_parameters)

            assert store.sweep_ttl() == 0
            assert _fetch_vector_rows(connection_parameters) == before_vector
            assert _count_store_rows(connection_parameters) == before_store
        finally:
            store.close()
