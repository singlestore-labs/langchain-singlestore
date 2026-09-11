"""Integration tests for :class:`SingleStoreStore` semantic ``SearchOp``.

Covers ``SearchOp(query=...)`` against the ``store_vector`` table:

* Basic ranking — items whose embedded field matches the query outrank items
  that don't; ``SearchItem.score`` is populated.
* Only items that produced ``store_vector`` rows are candidates: items put
  with ``index=False`` and items with none of the indexed fields present are
  excluded.
* ``namespace_prefix`` narrows the candidate set.
* ``filter`` (exact match and operator forms) narrows the candidate set.
* ``limit`` / ``offset`` paginate through ranked results.
* Multi-field indexes score each item by the MAX field-embedding similarity.
* ``refresh_ttl=True`` extends TTL on the matching rows and only those.
* Every supported :class:`ANNIndexConfig` (FLAT, IVF_FLAT, HNSW_FLAT, AUTO,
  EUCLIDEAN metric) produces the same ranking on small in-memory data.
* Misconfiguration — no ``index_config``, or manually-cleared ``embeddings``
  — raises a clear :class:`ValueError`.
"""

from __future__ import annotations

from typing import Any, List, cast

import pytest
from langchain_core.embeddings import Embeddings
from singlestore_langchain_core import (
    ANNIndexConfig,
    AUTOIndexConfig,
    FLATIndexConfig,
    HNSW_FLATIndexConfig,
    IVF_FLATIndexConfig,
)
from singlestore_langchain_core._utils import DistanceStrategy

from langgraph.store.base import PutOp, SearchItem, SearchOp
from langgraph.store.singlestore import SingleStoreStore
from langgraph.store.singlestore.base import SingleStoreIndexConfig

from .conftest import ConnectionParameters

# --- test embeddings ---------------------------------------------------------


_TOPIC_DIMS = {
    "sports": 0,
    "cooking": 1,
    "tech": 2,
    "history": 3,
}
_EMBED_DIMS = 8


class _TopicEmbeddings(Embeddings):
    """Deterministic embeddings driven by topic keywords in the text.

    ``"sports something"`` -> one-hot at dim 0, ``"cooking recipes"`` at dim
    1, ... Anything else -> zero vector. Combined with ``DOT_PRODUCT``, this
    lets a test assert exact ranking without floating-point tolerances:

    * items whose embedded text contains the query's topic keyword score 1.0
    * every other item scores 0.0
    """

    dims = _EMBED_DIMS

    def __init__(self) -> None:
        self.query_calls: list[str] = []
        self.document_calls: list[list[str]] = []

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        lowered = text.lower()
        for topic, dim in _TOPIC_DIMS.items():
            if topic in lowered:
                vec[dim] = 1.0
                return vec
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.document_calls.append(list(texts))
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        self.query_calls.append(text)
        return self._vec(text)


# --- helpers -----------------------------------------------------------------


def _make_index_config(
    embed: Embeddings,
    *,
    dims: int = _EMBED_DIMS,
    fields: Any = None,
    ann_index_config: Any = None,
) -> SingleStoreIndexConfig:
    cfg: dict[str, Any] = {"dims": dims, "embed": embed}
    if fields is not None:
        cfg["fields"] = fields
    if ann_index_config is not None:
        cfg["ann_index_config"] = ann_index_config
    return cast(SingleStoreIndexConfig, cfg)


def _search(store: SingleStoreStore, op: SearchOp) -> list[SearchItem]:
    results = store.batch([op])
    assert len(results) == 1
    return cast("list[SearchItem]", results[0])


def _keys(items: list[SearchItem]) -> list[tuple[tuple[str, ...], str]]:
    """Ordered ``(namespace, key)`` for each returned item."""
    return [(it.namespace, it.key) for it in items]


# The seed exercises every combination of interest:
#
# * every topic (``sports``/``cooking``/``tech``/``history``) is represented,
#   so a query on any topic has both matches and non-matches.
# * two "sports" items differ on ``public`` — used to test ``filter``.
# * one "sports" item has ``level`` set — used to test operator filters.
# * one item has no ``topic`` field: the ``fields=["topic"]`` config produces
#   no vector row for it, so it must never appear in a query result.
_SEED: tuple[tuple[tuple[str, ...], str, dict[str, Any]], ...] = (
    (
        ("docs", "public"),
        "sports-1",
        {"topic": "latest sports news", "public": True, "level": 10},
    ),
    (
        ("docs", "public"),
        "sports-2",
        {"topic": "sports scores today", "public": False, "level": 3},
    ),
    (
        ("docs", "public"),
        "cooking-1",
        {"topic": "cooking recipes for dinner", "public": True, "level": 7},
    ),
    (
        ("docs", "private"),
        "tech-1",
        {"topic": "tech industry updates", "public": False, "level": 5},
    ),
    (
        ("archive",),
        "history-1",
        {"topic": "history of ancient rome", "public": True, "level": 1},
    ),
    # No ``topic`` field — nothing to embed, never appears in query results.
    (
        ("docs", "public"),
        "no-topic",
        {"summary": "meta-only entry", "public": True, "level": 4},
    ),
)


def _seed(store: SingleStoreStore) -> None:
    store.batch([PutOp(ns, k, v) for ns, k, v in _SEED])


def _seed_keys_with_topic() -> set[tuple[tuple[str, ...], str]]:
    return {(ns, k) for ns, k, v in _SEED if "topic" in v}


def _seed_keys_with_topic_matching(word: str) -> set[tuple[tuple[str, ...], str]]:
    return {
        (ns, k) for ns, k, v in _SEED if "topic" in v and word in v["topic"].lower()
    }


# --- correct usage -----------------------------------------------------------


class TestSearchWithQueryBasics:
    """Query against a store initialised with a typed field index."""

    def test_query_ranks_matching_items_first_and_populates_score(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        embed = _TopicEmbeddings()
        store = SingleStoreStore(
            index=_make_index_config(embed, fields=["topic"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            _seed(store)

            got = _search(
                store,
                SearchOp(namespace_prefix=(), query="sports", refresh_ttl=False),
            )

            # ``no-topic`` never gets a vector row so it's excluded even from
            # the fallback zero-score results.
            assert set(_keys(got)) == _seed_keys_with_topic()
            # First two rows are the two sports items; ordering within the
            # tied "sports" group is by (prefix, key).
            assert _keys(got)[:2] == [
                (("docs", "public"), "sports-1"),
                (("docs", "public"), "sports-2"),
            ]
            assert got[0].score == 1.0
            assert got[1].score == 1.0
            # Non-matching items surface at score 0.0.
            for item in got[2:]:
                assert item.score == 0.0
            assert embed.query_calls == ["sports"]
        finally:
            store.close()

    def test_query_on_unrelated_topic_returns_all_indexed_items_at_zero(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Every indexed item participates in the ranking; unmatched items
        just get score ``0.0``. This is the documented DOT_PRODUCT behaviour
        and tests that no candidate is silently dropped."""
        embed = _TopicEmbeddings()
        store = SingleStoreStore(
            index=_make_index_config(embed, fields=["topic"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            _seed(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=(),
                    query="unrelated",
                    limit=100,
                    refresh_ttl=False,
                ),
            )
            assert set(_keys(got)) == _seed_keys_with_topic()
            assert all(item.score == 0.0 for item in got)
        finally:
            store.close()

    def test_query_excludes_items_with_no_vector_row(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """An item put with ``index=False`` never appears in a query result,
        even when its stored value matches the query topic."""
        embed = _TopicEmbeddings()
        store = SingleStoreStore(
            index=_make_index_config(embed, fields=["topic"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            store.batch(
                [
                    PutOp(("docs",), "indexed", {"topic": "sports weekly"}),
                    PutOp(
                        ("docs",),
                        "opt-out",
                        {"topic": "sports daily"},
                        index=False,
                    ),
                ]
            )

            got = _search(
                store,
                SearchOp(namespace_prefix=(), query="sports", refresh_ttl=False),
            )
            assert _keys(got) == [(("docs",), "indexed")]
            assert got[0].score == 1.0
        finally:
            store.close()

    def test_query_returns_empty_when_vector_table_is_empty(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        embed = _TopicEmbeddings()
        store = SingleStoreStore(
            index=_make_index_config(embed, fields=["topic"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            got = _search(
                store,
                SearchOp(namespace_prefix=(), query="sports", refresh_ttl=False),
            )
            assert got == []
        finally:
            store.close()

    def test_query_with_root_field_embeds_full_json(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """``fields=["$"]`` embeds the JSON dump — the topic keyword ends up
        inside the serialised text regardless of which key it lives on."""
        embed = _TopicEmbeddings()
        store = SingleStoreStore(
            index=_make_index_config(embed, fields=["$"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            store.batch(
                [
                    PutOp(("docs",), "a", {"summary": "sports weekly"}),
                    PutOp(("docs",), "b", {"summary": "cooking classes"}),
                ]
            )
            got = _search(
                store,
                SearchOp(namespace_prefix=(), query="sports", refresh_ttl=False),
            )
            assert _keys(got) == [(("docs",), "a"), (("docs",), "b")]
            assert got[0].score == 1.0
            assert got[1].score == 0.0
        finally:
            store.close()

    def test_query_takes_max_across_multiple_indexed_fields_per_item(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """When a store indexes multiple fields, the score for an item is the
        MAX of the DOT_PRODUCT across its field embeddings — so an item whose
        *any* field matches the query outranks an item with no matching
        field."""
        embed = _TopicEmbeddings()
        store = SingleStoreStore(
            index=_make_index_config(embed, fields=["title", "body"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            store.batch(
                [
                    # Title mismatched, body matches — should still rank first.
                    PutOp(
                        ("docs",),
                        "hit",
                        {"title": "generic headline", "body": "about sports"},
                    ),
                    PutOp(
                        ("docs",),
                        "miss",
                        {"title": "generic headline", "body": "about cooking"},
                    ),
                ]
            )
            got = _search(
                store,
                SearchOp(namespace_prefix=(), query="sports", refresh_ttl=False),
            )
            assert _keys(got) == [(("docs",), "hit"), (("docs",), "miss")]
            assert got[0].score == 1.0
            assert got[1].score == 0.0
        finally:
            store.close()


class TestSearchWithQueryFilters:
    """``filter`` is combined with the query at the SQL WHERE level."""

    def test_query_and_namespace_prefix(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        embed = _TopicEmbeddings()
        store = SingleStoreStore(
            index=_make_index_config(embed, fields=["topic"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            _seed(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=("docs",),
                    query="sports",
                    refresh_ttl=False,
                ),
            )
            expected = {
                (ns, k) for ns, k, v in _SEED if "topic" in v and ns[0] == "docs"
            }
            assert set(_keys(got)) == expected
            # Both "sports-*" items sit under ("docs", "public") — they win.
            assert _keys(got)[:2] == [
                (("docs", "public"), "sports-1"),
                (("docs", "public"), "sports-2"),
            ]
        finally:
            store.close()

    def test_query_and_nested_namespace_prefix_isolates_subtree(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        embed = _TopicEmbeddings()
        store = SingleStoreStore(
            index=_make_index_config(embed, fields=["topic"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            _seed(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=("docs", "private"),
                    query="sports",
                    refresh_ttl=False,
                ),
            )
            # No sports items live under ``docs/private`` — the only indexed
            # entry there is ``tech-1``, and it scores 0 but still surfaces.
            assert _keys(got) == [(("docs", "private"), "tech-1")]
            assert got[0].score == 0.0
        finally:
            store.close()

    def test_query_and_exact_filter(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        embed = _TopicEmbeddings()
        store = SingleStoreStore(
            index=_make_index_config(embed, fields=["topic"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            _seed(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=(),
                    query="sports",
                    filter={"public": True},
                    refresh_ttl=False,
                ),
            )
            expected = {
                (ns, k)
                for ns, k, v in _SEED
                if "topic" in v and v.get("public") is True
            }
            assert set(_keys(got)) == expected
            # ``sports-1`` is the only public sports item and must lead.
            assert _keys(got)[0] == (("docs", "public"), "sports-1")
            assert got[0].score == 1.0
        finally:
            store.close()

    def test_query_and_operator_filter(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        embed = _TopicEmbeddings()
        store = SingleStoreStore(
            index=_make_index_config(embed, fields=["topic"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            _seed(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=(),
                    query="sports",
                    filter={"level": {"$gte": 5}},
                    refresh_ttl=False,
                ),
            )
            expected = {
                (ns, k) for ns, k, v in _SEED if "topic" in v and v.get("level", 0) >= 5
            }
            assert set(_keys(got)) == expected
            assert _keys(got)[0] == (("docs", "public"), "sports-1")
        finally:
            store.close()

    def test_query_and_in_filter(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        embed = _TopicEmbeddings()
        store = SingleStoreStore(
            index=_make_index_config(embed, fields=["topic"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            _seed(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=(),
                    query="tech",
                    filter={"level": {"$in": [5, 7]}},
                    refresh_ttl=False,
                ),
            )
            expected = {
                (ns, k)
                for ns, k, v in _SEED
                if "topic" in v and v.get("level") in (5, 7)
            }
            assert set(_keys(got)) == expected
            # Only ``tech-1`` matches "tech" → its score wins.
            assert _keys(got)[0] == (("docs", "private"), "tech-1")
            assert got[0].score == 1.0
        finally:
            store.close()

    def test_query_and_filter_with_no_matches_returns_empty(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        embed = _TopicEmbeddings()
        store = SingleStoreStore(
            index=_make_index_config(embed, fields=["topic"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            _seed(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=(),
                    query="sports",
                    filter={"public": "definitely-not-a-value"},
                    refresh_ttl=False,
                ),
            )
            assert got == []
        finally:
            store.close()

    def test_query_namespace_prefix_no_match_returns_empty(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        embed = _TopicEmbeddings()
        store = SingleStoreStore(
            index=_make_index_config(embed, fields=["topic"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            _seed(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=("does-not-exist",),
                    query="sports",
                    refresh_ttl=False,
                ),
            )
            assert got == []
        finally:
            store.close()


class TestSearchWithQueryPagination:
    """``limit`` / ``offset`` paginate the ranked candidate set."""

    def test_limit_caps_result_size(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        embed = _TopicEmbeddings()
        store = SingleStoreStore(
            index=_make_index_config(embed, fields=["topic"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            _seed(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=(),
                    query="sports",
                    limit=2,
                    refresh_ttl=False,
                ),
            )
            assert len(got) == 2
            # Highest-scoring items surface first.
            assert set(_keys(got)) == {
                (("docs", "public"), "sports-1"),
                (("docs", "public"), "sports-2"),
            }
            assert all(item.score == 1.0 for item in got)
        finally:
            store.close()

    def test_limit_plus_offset_covers_full_set(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        embed = _TopicEmbeddings()
        store = SingleStoreStore(
            index=_make_index_config(embed, fields=["topic"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            _seed(store)

            page_size = 2
            collected: list[tuple[tuple[str, ...], str]] = []
            offset = 0
            while True:
                page = _search(
                    store,
                    SearchOp(
                        namespace_prefix=(),
                        query="sports",
                        limit=page_size,
                        offset=offset,
                        refresh_ttl=False,
                    ),
                )
                if not page:
                    break
                collected.extend(_keys(page))
                offset += page_size

            assert set(collected) == _seed_keys_with_topic()
            # No page overlap.
            assert len(collected) == len(set(collected))
        finally:
            store.close()

    def test_offset_past_end_returns_empty(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        embed = _TopicEmbeddings()
        store = SingleStoreStore(
            index=_make_index_config(embed, fields=["topic"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            _seed(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=(),
                    query="sports",
                    limit=5,
                    offset=len(_SEED) + 10,
                    refresh_ttl=False,
                ),
            )
            assert got == []
        finally:
            store.close()


class TestSearchWithQueryTTL:
    """``refresh_ttl`` interaction with the query path."""

    def test_expired_rows_are_excluded_from_query(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        from singlestoredb.connection import connect

        embed = _TopicEmbeddings()
        store = SingleStoreStore(
            index=_make_index_config(embed, fields=["topic"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            store.batch(
                [
                    PutOp(("docs",), "fresh", {"topic": "sports weekly"}),
                    PutOp(
                        ("docs",),
                        "stale",
                        {"topic": "sports daily"},
                        ttl=1.0,
                    ),
                ]
            )
            conn = connect(**connection_parameters.as_kwargs())
            try:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE store SET expires_at = DATE_SUB(NOW(), INTERVAL 1 MINUTE) "
                    "WHERE `key` = %s",
                    ("stale",),
                )
                cur.close()
            finally:
                conn.close()

            got = _search(
                store,
                SearchOp(namespace_prefix=(), query="sports", refresh_ttl=False),
            )
            assert _keys(got) == [(("docs",), "fresh")]
        finally:
            store.close()

    def test_refresh_ttl_true_bumps_expires_at_on_matching_rows_only(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        import time

        from singlestoredb.connection import connect

        embed = _TopicEmbeddings()
        store = SingleStoreStore(
            index=_make_index_config(embed, fields=["topic"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            store.batch(
                [
                    PutOp(("docs",), "match", {"topic": "sports weekly"}, ttl=1.0),
                    PutOp(
                        ("other",),
                        "skip",
                        {"topic": "cooking classes"},
                        ttl=1.0,
                    ),
                ]
            )

            def _expiries() -> dict[str, Any]:
                conn = connect(**connection_parameters.as_kwargs())
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT `key`, expires_at FROM store")
                    rows = cur.fetchall()
                    cur.close()
                    return {str(list(r)[0]): list(r)[1] for r in rows}
                finally:
                    conn.close()

            before = _expiries()
            assert before["match"] is not None
            assert before["skip"] is not None

            # TIMESTAMP granularity is 1s.
            time.sleep(1.1)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=("docs",),
                    query="sports",
                    limit=10,
                ),
            )
            assert _keys(got) == [(("docs",), "match")]

            after = _expiries()
            assert after["match"] > before["match"]
            # ``skip`` was filtered out by the namespace prefix — no refresh.
            assert after["skip"] == before["skip"]
        finally:
            store.close()


# --- ANN index configurations ------------------------------------------------


_ANN_CONFIGS: list[tuple[str, ANNIndexConfig]] = [
    ("default_flat", cast(ANNIndexConfig, FLATIndexConfig(index_type="FLAT"))),
    ("auto", cast(ANNIndexConfig, AUTOIndexConfig(index_type="AUTO"))),
    (
        "ivf_flat",
        cast(
            ANNIndexConfig,
            IVF_FLATIndexConfig(index_type="IVF_FLAT", nlist=4, nprobe=4),
        ),
    ),
    (
        "hnsw_flat",
        cast(
            ANNIndexConfig,
            HNSW_FLATIndexConfig(
                index_type="HNSW_FLAT", M=16, efConstruction=32, ef=16
            ),
        ),
    ),
    (
        "flat_euclidean",
        cast(
            ANNIndexConfig,
            FLATIndexConfig(
                index_type="FLAT",
                metric_type=DistanceStrategy.EUCLIDEAN_DISTANCE,
            ),
        ),
    ),
]


class TestSearchWithQueryAcrossIndexConfigs:
    """Ranking must be identical across every supported ANN index config on
    the small in-memory data used here."""

    @pytest.mark.parametrize(
        ("label", "ann"),
        _ANN_CONFIGS,
        ids=[label for label, _ in _ANN_CONFIGS],
    )
    def test_query_ranks_sports_first_across_ann_configs(
        self,
        connection_parameters: ConnectionParameters,
        label: str,
        ann: ANNIndexConfig,
    ) -> None:
        embed = _TopicEmbeddings()
        store = SingleStoreStore(
            index=_make_index_config(embed, fields=["topic"], ann_index_config=ann),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            _seed(store)

            got = _search(
                store,
                SearchOp(namespace_prefix=(), query="sports", refresh_ttl=False),
            )
            assert set(_keys(got)) == _seed_keys_with_topic()
            # Sports items lead regardless of index topology.
            leaders = set(_keys(got)[:2])
            assert leaders == {
                (("docs", "public"), "sports-1"),
                (("docs", "public"), "sports-2"),
            }
            assert got[0].score == 1.0 and got[1].score == 1.0
        finally:
            store.close()

    @pytest.mark.parametrize(
        ("label", "ann"),
        _ANN_CONFIGS,
        ids=[label for label, _ in _ANN_CONFIGS],
    )
    def test_query_with_filter_across_ann_configs(
        self,
        connection_parameters: ConnectionParameters,
        label: str,
        ann: ANNIndexConfig,
    ) -> None:
        embed = _TopicEmbeddings()
        store = SingleStoreStore(
            index=_make_index_config(embed, fields=["topic"], ann_index_config=ann),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            _seed(store)

            got = _search(
                store,
                SearchOp(
                    namespace_prefix=(),
                    query="sports",
                    filter={"public": True},
                    refresh_ttl=False,
                ),
            )
            expected = {
                (ns, k)
                for ns, k, v in _SEED
                if "topic" in v and v.get("public") is True
            }
            assert set(_keys(got)) == expected
            assert _keys(got)[0] == (("docs", "public"), "sports-1")
        finally:
            store.close()


# --- error paths -------------------------------------------------------------


class TestSearchWithQueryErrors:
    def test_query_without_index_config_raises(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            with pytest.raises(ValueError, match="Index configuration is required"):
                store.batch(
                    [
                        SearchOp(
                            namespace_prefix=(),
                            query="anything",
                            refresh_ttl=False,
                        )
                    ]
                )
        finally:
            store.close()

    def test_query_with_manually_cleared_embeddings_raises(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """The construction path always attaches embeddings, but the guard
        clause in ``_batch_search_ops`` is still worth pinning down."""
        embed = _TopicEmbeddings()
        store = SingleStoreStore(
            index=_make_index_config(embed, fields=["topic"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            store.setup()
            store.embeddings = None
            with pytest.raises(ValueError, match="Embeddings are required"):
                store.batch(
                    [
                        SearchOp(
                            namespace_prefix=(),
                            query="sports",
                            refresh_ttl=False,
                        )
                    ]
                )
        finally:
            store.close()

    def test_query_before_setup_fails(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Without ``setup()`` neither ``store`` nor ``store_vector`` exist,
        so a query fails at the database level."""
        embed = _TopicEmbeddings()
        store = SingleStoreStore(
            index=_make_index_config(embed, fields=["topic"]),
            **connection_parameters.as_kwargs(),
        )
        try:
            with pytest.raises(Exception):
                store.batch(
                    [
                        SearchOp(
                            namespace_prefix=(),
                            query="sports",
                            refresh_ttl=False,
                        )
                    ]
                )
        finally:
            store.close()
