"""Unit tests for ``_ensure_index_config``
in :mod:`langgraph.store.singlestore.base`."""

from __future__ import annotations

from typing import Any, List, cast

import pytest
from langchain_core.embeddings import Embeddings
from singlestore_langchain_core import (
    HNSW_FLATIndexConfig,
    IVF_PQIndexConfig,
)
from singlestore_langchain_core._utils import DistanceStrategy

from langgraph.store.singlestore.base import (
    SingleStoreIndexConfig,
    _ensure_index_config,
)


class _FakeEmbeddings(Embeddings):
    """Minimal Embeddings implementation for tests."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.0, 1.0] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.0, 1.0]


def _make_config(**overrides: Any) -> SingleStoreIndexConfig:
    base: dict[str, Any] = {"dims": 2, "embed": _FakeEmbeddings()}
    base.update(overrides)
    return cast(SingleStoreIndexConfig, base)


class TestEnsureIndexConfigFields:
    def test_missing_fields_defaults_to_root(self) -> None:
        embeddings, cfg = _ensure_index_config(_make_config())

        assert isinstance(embeddings, _FakeEmbeddings)
        assert cfg["fields"] == ["$"]
        assert cfg["__tokenized_fields"] == [("$", "$")]  # type: ignore[typeddict-item]
        assert cfg["__estimated_num_vectors"] == 1  # type: ignore[typeddict-item]

    def test_fields_explicit_none_defaults_to_root(self) -> None:
        _, cfg = _ensure_index_config(_make_config(fields=None))
        assert cfg["fields"] == ["$"]
        assert cfg["__tokenized_fields"] == [("$", "$")]  # type: ignore[typeddict-item]
        assert cfg["__estimated_num_vectors"] == 1  # type: ignore[typeddict-item]

    def test_fields_empty_list_defaults_to_root(self) -> None:
        _, cfg = _ensure_index_config(_make_config(fields=[]))
        assert cfg["fields"] == ["$"]
        assert cfg["__estimated_num_vectors"] == 1  # type: ignore[typeddict-item]

    def test_fields_as_string_is_normalized_to_list(self) -> None:
        _, cfg = _ensure_index_config(_make_config(fields="foo.bar"))
        assert cfg["fields"] == ["foo.bar"]
        assert cfg["__tokenized_fields"] == [("foo.bar", ["foo", "bar"])]  # type: ignore[typeddict-item]
        assert cfg["__estimated_num_vectors"] == 2  # type: ignore[typeddict-item]

    def test_fields_list_of_paths_tokenized(self) -> None:
        _, cfg = _ensure_index_config(_make_config(fields=["foo.bar", "baz", "a.b.c"]))
        assert cfg["fields"] == ["foo.bar", "baz", "a.b.c"]
        assert cfg["__tokenized_fields"] == [  # type: ignore[typeddict-item]
            ("foo.bar", ["foo", "bar"]),
            ("baz", ["baz"]),
            ("a.b.c", ["a", "b", "c"]),
        ]
        assert cfg["__estimated_num_vectors"] == 6  # type: ignore[typeddict-item]

    def test_fields_mixed_root_and_paths(self) -> None:
        _, cfg = _ensure_index_config(_make_config(fields=["$", "foo.bar"]))
        assert cfg["__tokenized_fields"] == [  # type: ignore[typeddict-item]
            ("$", "$"),
            ("foo.bar", ["foo", "bar"]),
        ]
        assert cfg["__estimated_num_vectors"] == 3  # type: ignore[typeddict-item]

    @pytest.mark.parametrize("bad_fields", [123, {"a": 1}, 3.14, object()])
    def test_fields_invalid_type_raises(self, bad_fields: Any) -> None:
        with pytest.raises(ValueError, match="Text fields must be a list or a string"):
            _ensure_index_config(_make_config(fields=bad_fields))


class TestEnsureIndexConfigANN:
    def test_ann_config_missing_gets_defaults(self) -> None:
        _, cfg = _ensure_index_config(_make_config())
        ann = cfg["ann_index_config"]
        assert ann["metric_type"] == DistanceStrategy.DOT_PRODUCT
        assert ann["index_type"] == "FLAT"

    def test_ann_config_empty_dict_gets_defaults(self) -> None:
        _, cfg = _ensure_index_config(_make_config(ann_index_config={}))
        ann = cfg["ann_index_config"]
        assert ann["metric_type"] == DistanceStrategy.DOT_PRODUCT
        assert ann["index_type"] == "FLAT"

    def test_ann_config_index_type_preserved(self) -> None:
        ann_in = IVF_PQIndexConfig(
            index_type="IVF_PQ",
            nlist=256,
            m=2,
            nbits=8,
            nprobe=8,
        )
        _, cfg = _ensure_index_config(_make_config(ann_index_config=ann_in))
        ann = cfg["ann_index_config"]
        assert ann["index_type"] == "IVF_PQ"
        assert ann["nlist"] == 256  # type: ignore[typeddict-item]
        assert ann["m"] == 2  # type: ignore[typeddict-item]
        assert ann["nbits"] == 8  # type: ignore[typeddict-item]
        assert ann["nprobe"] == 8  # type: ignore[typeddict-item]
        # metric_type is filled with the default because caller didn't set it.
        assert ann["metric_type"] == DistanceStrategy.DOT_PRODUCT

    def test_ann_config_metric_type_preserved(self) -> None:
        ann_in = HNSW_FLATIndexConfig(
            index_type="HNSW_FLAT",
            metric_type=DistanceStrategy.EUCLIDEAN_DISTANCE,
            M=30,
            efConstruction=40,
            ef=16,
        )
        _, cfg = _ensure_index_config(_make_config(ann_index_config=ann_in))
        ann = cfg["ann_index_config"]
        assert ann["index_type"] == "HNSW_FLAT"
        assert ann["metric_type"] == DistanceStrategy.EUCLIDEAN_DISTANCE
        assert ann["M"] == 30  # type: ignore[typeddict-item]
        assert ann["efConstruction"] == 40  # type: ignore[typeddict-item]
        assert ann["ef"] == 16  # type: ignore[typeddict-item]

    def test_ann_config_only_metric_type_provided(self) -> None:
        _, cfg = _ensure_index_config(
            _make_config(
                ann_index_config={"metric_type": DistanceStrategy.EUCLIDEAN_DISTANCE}
            )
        )
        ann = cfg["ann_index_config"]
        assert ann["metric_type"] == DistanceStrategy.EUCLIDEAN_DISTANCE
        assert ann["index_type"] == "FLAT"


class TestEnsureIndexConfigEmbeddings:
    def test_embeddings_instance_returned_as_is(self) -> None:
        embed = _FakeEmbeddings()
        embeddings, _ = _ensure_index_config(_make_config(embed=embed))
        assert embeddings is embed

    def test_embed_callable_wrapped(self) -> None:
        def fn(texts: List[str]) -> List[List[float]]:
            return [[float(len(t))] for t in texts]

        embeddings, _ = _ensure_index_config(_make_config(embed=fn))
        assert embeddings is not None
        assert embeddings.embed_query("hello") == [5.0]

    def test_missing_embed_raises(self) -> None:
        cfg: SingleStoreIndexConfig = cast(SingleStoreIndexConfig, {"dims": 2})
        with pytest.raises(ValueError, match="embed must be provided"):
            _ensure_index_config(cfg)


class TestEnsureIndexConfigImmutability:
    def test_input_dict_is_not_mutated(self) -> None:
        original: dict[str, Any] = {
            "dims": 2,
            "embed": _FakeEmbeddings(),
            "fields": ["foo.bar"],
            "ann_index_config": {},
        }
        snapshot = {
            "dims": original["dims"],
            "embed": original["embed"],
            "fields": list(original["fields"]),
            "ann_index_config": dict(original["ann_index_config"]),
        }

        _ensure_index_config(cast(SingleStoreIndexConfig, original))

        assert original["dims"] == snapshot["dims"]
        assert original["embed"] is snapshot["embed"]
        assert original["fields"] == snapshot["fields"]
        assert "__tokenized_fields" not in original
        assert "__estimated_num_vectors" not in original

    def test_returned_config_contains_all_derived_keys(self) -> None:
        _, cfg = _ensure_index_config(_make_config(fields=["a.b"]))
        assert "fields" in cfg
        assert "ann_index_config" in cfg
        assert "__tokenized_fields" in cfg  # type: ignore[operator]
        assert "__estimated_num_vectors" in cfg  # type: ignore[operator]
