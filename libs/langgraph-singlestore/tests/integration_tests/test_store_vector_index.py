"""Integration tests for :class:`SingleStoreStore` with a vector index.

Covers:

* ``setup()`` creates the ``store_vector`` table for a range of
  :class:`ANNIndexConfig` subclasses (AUTO, FLAT, IVF_FLAT, IVF_PQ,
  IVF_PQFS, HNSW_FLAT, HNSW_PQ).
* Custom ``dims``, ``fields`` and ``metric_type`` are honoured.
* Invalid configurations fail loudly at construction or at ``setup()``.
"""

from __future__ import annotations

import json
from contextlib import closing
from typing import Any, List, cast

import pytest
from langchain_core.embeddings import Embeddings
from singlestore_langchain_core import (
    ANNIndexConfig,
    AUTOIndexConfig,
    FLATIndexConfig,
    HNSW_FLATIndexConfig,
    HNSW_PQIndexConfig,
    IVF_FLATIndexConfig,
    IVF_PQFSIIndexConfig,
    IVF_PQIndexConfig,
)
from singlestore_langchain_core._utils import DistanceStrategy
from singlestoredb.connection import connect

from langgraph.store.singlestore import SingleStoreStore
from langgraph.store.singlestore.base import SingleStoreIndexConfig

from .conftest import ConnectionParameters


class _FakeEmbeddings(Embeddings):
    """Deterministic Embeddings implementation for tests."""

    def __init__(self, dims: int) -> None:
        self.dims = dims

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(i % 2) for i in range(self.dims)] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [float(i % 2) for i in range(self.dims)]


def _make_index_config(
    *,
    dims: int = 8,
    fields: Any = None,
    ann_index_config: Any = None,
) -> SingleStoreIndexConfig:
    cfg: dict[str, Any] = {"dims": dims, "embed": _FakeEmbeddings(dims)}
    if fields is not None:
        cfg["fields"] = fields
    if ann_index_config is not None:
        cfg["ann_index_config"] = ann_index_config
    return cast(SingleStoreIndexConfig, cfg)


def _show_create_store_vector(params: ConnectionParameters) -> str:
    conn = connect(**params.as_kwargs())
    try:
        with closing(conn.cursor()) as cur:
            cur.execute("SHOW CREATE TABLE store_vector")
            row = cur.fetchone()
            assert row is not None
            return str(list(row)[1])
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


def _parse_index_options(create_sql: str) -> dict[str, Any]:
    """Extract the JSON payload embedded in ``INDEX_OPTIONS '...'``."""
    marker = 'INDEX_OPTIONS="'
    start = create_sql.index(marker) + len(marker)
    end = create_sql.index('}"', start)
    return cast(
        dict[str, Any], json.loads(create_sql[start : end + 1].replace('\\"', '"'))
    )


class TestSingleStoreStoreVectorIndexCreation:
    """``setup()`` creates ``store_vector`` for every supported ANN config."""

    def test_default_ann_config_creates_flat_index(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(
            index=_make_index_config(dims=8),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            assert _table_exists(connection_parameters, "store_vector")
            create_sql = _show_create_store_vector(connection_parameters)
            assert "vector(8, F32)" in create_sql
            opts = _parse_index_options(create_sql)
            assert opts["index_type"] == "FLAT"
            assert opts["metric_type"] == DistanceStrategy.DOT_PRODUCT
        finally:
            store.close()

    def test_no_index_config_skips_vector_table(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            assert not _table_exists(connection_parameters, "store_vector")
        finally:
            store.close()

    def test_auto_index_config(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        ann = AUTOIndexConfig(index_type="AUTO")
        store = SingleStoreStore(
            index=_make_index_config(dims=16, ann_index_config=ann),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            create_sql = _show_create_store_vector(connection_parameters)
            assert "vector(16, F32)" in create_sql
            opts = _parse_index_options(create_sql)
            assert opts["index_type"] == "AUTO"
        finally:
            store.close()

    def test_flat_index_config(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        ann = FLATIndexConfig(index_type="FLAT")
        store = SingleStoreStore(
            index=_make_index_config(dims=8, ann_index_config=ann),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            opts = _parse_index_options(
                _show_create_store_vector(connection_parameters)
            )
            assert opts["index_type"] == "FLAT"
        finally:
            store.close()

    def test_ivf_flat_index_config(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        ann = IVF_FLATIndexConfig(
            index_type="IVF_FLAT",
            nlist=128,
            nprobe=8,
        )
        store = SingleStoreStore(
            index=_make_index_config(dims=8, ann_index_config=ann),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            opts = _parse_index_options(
                _show_create_store_vector(connection_parameters)
            )
            assert opts["index_type"] == "IVF_FLAT"
            assert opts["nlist"] == 128
            assert opts["nprobe"] == 8
        finally:
            store.close()

    def test_ivf_pq_index_config(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        ann = IVF_PQIndexConfig(
            index_type="IVF_PQ",
            nlist=128,
            m=2,
            nbits=8,
            nprobe=8,
        )
        store = SingleStoreStore(
            index=_make_index_config(dims=8, ann_index_config=ann),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            opts = _parse_index_options(
                _show_create_store_vector(connection_parameters)
            )
            assert opts["index_type"] == "IVF_PQ"
            assert opts["nlist"] == 128
            assert opts["m"] == 2
            assert opts["nbits"] == 8
            assert opts["nprobe"] == 8
        finally:
            store.close()

    def test_ivf_pqfs_index_config(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        ann = IVF_PQFSIIndexConfig(
            index_type="IVF_PQFS",
            nlist=128,
            m=2,
            nprobe=8,
        )
        store = SingleStoreStore(
            index=_make_index_config(dims=8, ann_index_config=ann),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            opts = _parse_index_options(
                _show_create_store_vector(connection_parameters)
            )
            assert opts["index_type"] == "IVF_PQFS"
            assert opts["nlist"] == 128
            assert opts["m"] == 2
            assert opts["nprobe"] == 8
        finally:
            store.close()

    def test_hnsw_flat_index_config(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        ann = HNSW_FLATIndexConfig(
            index_type="HNSW_FLAT",
            M=30,
            efConstruction=40,
            ef=16,
        )
        store = SingleStoreStore(
            index=_make_index_config(dims=8, ann_index_config=ann),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            opts = _parse_index_options(
                _show_create_store_vector(connection_parameters)
            )
            assert opts["index_type"] == "HNSW_FLAT"
            assert opts["M"] == 30
            assert opts["efConstruction"] == 40
            assert opts["ef"] == 16
        finally:
            store.close()

    def test_hnsw_pq_index_config(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        ann = HNSW_PQIndexConfig(
            index_type="HNSW_PQ",
            M=30,
            efConstruction=40,
            m=2,
            nbits=8,
            ef=16,
        )
        store = SingleStoreStore(
            index=_make_index_config(dims=8, ann_index_config=ann),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            opts = _parse_index_options(
                _show_create_store_vector(connection_parameters)
            )
            assert opts["index_type"] == "HNSW_PQ"
            assert opts["M"] == 30
            assert opts["efConstruction"] == 40
            assert opts["m"] == 2
            assert opts["nbits"] == 8
            assert opts["ef"] == 16
        finally:
            store.close()

    def test_euclidean_distance_metric_is_honoured(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        ann = FLATIndexConfig(
            index_type="FLAT",
            metric_type=DistanceStrategy.EUCLIDEAN_DISTANCE,
        )
        store = SingleStoreStore(
            index=_make_index_config(dims=8, ann_index_config=ann),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            opts = _parse_index_options(
                _show_create_store_vector(connection_parameters)
            )
            assert opts["metric_type"] == DistanceStrategy.EUCLIDEAN_DISTANCE
        finally:
            store.close()

    def test_custom_dims_reflected_in_vector_column(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(
            index=_make_index_config(dims=32),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            create_sql = _show_create_store_vector(connection_parameters)
            assert "vector(32, F32)" in create_sql
        finally:
            store.close()

    def test_custom_fields_still_creates_index_table(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(
            index=_make_index_config(dims=8, fields=["$", "summary", "body.text"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            assert _table_exists(connection_parameters, "store_vector")
            assert store.index_config is not None
            assert store.index_config["fields"] == ["$", "summary", "body.text"]
        finally:
            store.close()

    def test_setup_is_idempotent_with_index(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(
            index=_make_index_config(dims=8),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            store.setup()
            assert _table_exists(connection_parameters, "store_vector")
        finally:
            store.close()


class TestSingleStoreStoreVectorIndexInvalidConfig:
    """Invalid configurations must fail fast."""

    def test_missing_embed_raises_at_construction(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        with pytest.raises(ValueError, match="embed must be provided"):
            SingleStoreStore(
                index=cast(SingleStoreIndexConfig, {"dims": 8}),
                **connection_parameters.as_kwargs(),
            )

    @pytest.mark.parametrize("bad_fields", [123, {"a": 1}, 3.14])
    def test_invalid_fields_type_raises_at_construction(
        self,
        connection_parameters: ConnectionParameters,
        bad_fields: Any,
    ) -> None:
        with pytest.raises(ValueError, match="Text fields must be a list or a string"):
            SingleStoreStore(
                index=_make_index_config(dims=8, fields=bad_fields),
                **connection_parameters.as_kwargs(),
            )

    def test_invalid_index_type_fails_at_setup(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """A bogus ``index_type`` string is rejected by SingleStore during DDL."""
        ann: ANNIndexConfig = cast(
            ANNIndexConfig, {"index_type": "NOT_A_REAL_INDEX_TYPE"}
        )
        store = SingleStoreStore(
            index=_make_index_config(dims=8, ann_index_config=ann),
            **connection_parameters.as_kwargs(),
        )
        try:
            with pytest.raises(Exception):
                store.setup()
            assert not _table_exists(connection_parameters, "store_vector")
        finally:
            store.close()

    def test_zero_dims_fails_at_setup(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """A ``VECTOR(0, F32)`` column is not accepted by SingleStore."""
        store = SingleStoreStore(
            index=_make_index_config(dims=0),
            **connection_parameters.as_kwargs(),
        )
        try:
            with pytest.raises(Exception):
                store.setup()
            assert not _table_exists(connection_parameters, "store_vector")
        finally:
            store.close()

    def test_incompatible_hnsw_pq_m_dims_fails_at_setup(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """``dims % m`` must be 0 for product-quantised HNSW indexes."""
        ann = HNSW_PQIndexConfig(
            index_type="HNSW_PQ",
            M=30,
            efConstruction=40,
            m=3,
            nbits=8,
            ef=16,
        )
        store = SingleStoreStore(
            index=_make_index_config(dims=8, ann_index_config=ann),
            **connection_parameters.as_kwargs(),
        )
        try:
            with pytest.raises(Exception):
                store.setup()
        finally:
            store.close()
