"""Unit tests for singlestore_langchain_core._index."""

import typing
import unittest
from typing import get_type_hints

from singlestore_langchain_core._index import (
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


class TestANNIndexConfig(unittest.TestCase):
    def test_is_typed_dict(self) -> None:
        assert typing.is_typeddict(ANNIndexConfig)

    def test_is_dict_subclass(self) -> None:
        instance: ANNIndexConfig = ANNIndexConfig()
        assert isinstance(instance, dict)

    def test_total_false_no_required_keys(self) -> None:
        assert ANNIndexConfig.__total__ is False
        assert ANNIndexConfig.__required_keys__ == frozenset()

    def test_optional_keys(self) -> None:
        assert set(ANNIndexConfig.__optional_keys__) == {"index_type", "metric_type"}

    def test_empty_construction(self) -> None:
        cfg: ANNIndexConfig = ANNIndexConfig()
        assert cfg == {}

    def test_full_construction(self) -> None:
        cfg: ANNIndexConfig = ANNIndexConfig(
            index_type="AUTO",
            metric_type=DistanceStrategy.DOT_PRODUCT,
        )
        assert cfg["index_type"] == "AUTO"
        assert cfg["metric_type"] == DistanceStrategy.DOT_PRODUCT

    def test_index_type_literal_values(self) -> None:
        hints = get_type_hints(ANNIndexConfig, include_extras=True)
        literal_args = typing.get_args(hints["index_type"])
        assert set(literal_args) == {
            "AUTO",
            "FLAT",
            "IVF_FLAT",
            "IVF_PQ",
            "IVF_PQFS",
            "HNSW_FLAT",
            "HNSW_PQ",
        }

    def test_metric_type_literal_values(self) -> None:
        hints = get_type_hints(ANNIndexConfig, include_extras=True)
        literal_args = typing.get_args(hints["metric_type"])
        assert set(literal_args) == {
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            DistanceStrategy.DOT_PRODUCT,
        }


class TestAUTOIndexConfig(unittest.TestCase):
    def test_is_typed_dict(self) -> None:
        assert typing.is_typeddict(AUTOIndexConfig)

    def test_total_true(self) -> None:
        assert AUTOIndexConfig.__total__ is True

    def test_required_and_optional_keys(self) -> None:
        assert AUTOIndexConfig.__required_keys__ == frozenset({"index_type"})
        assert AUTOIndexConfig.__optional_keys__ == frozenset({"metric_type"})

    def test_index_type_literal(self) -> None:
        hints = get_type_hints(AUTOIndexConfig, include_extras=True)
        assert typing.get_args(hints["index_type"]) == ("AUTO",)

    def test_construction(self) -> None:
        cfg: AUTOIndexConfig = AUTOIndexConfig(
            index_type="AUTO",
            metric_type=DistanceStrategy.EUCLIDEAN_DISTANCE,
        )
        assert cfg["index_type"] == "AUTO"
        assert cfg["metric_type"] == DistanceStrategy.EUCLIDEAN_DISTANCE


class TestFLATIndexConfig(unittest.TestCase):
    def test_is_typed_dict(self) -> None:
        assert typing.is_typeddict(FLATIndexConfig)

    def test_index_type_literal(self) -> None:
        hints = get_type_hints(FLATIndexConfig, include_extras=True)
        assert typing.get_args(hints["index_type"]) == ("FLAT",)

    def test_construction(self) -> None:
        cfg: FLATIndexConfig = FLATIndexConfig(index_type="FLAT")
        assert cfg["index_type"] == "FLAT"


class TestIVF_FLATIndexConfig(unittest.TestCase):
    def test_is_typed_dict(self) -> None:
        assert typing.is_typeddict(IVF_FLATIndexConfig)

    def test_index_type_literal(self) -> None:
        hints = get_type_hints(IVF_FLATIndexConfig, include_extras=True)
        assert typing.get_args(hints["index_type"]) == ("IVF_FLAT",)

    def test_optional_keys(self) -> None:
        assert IVF_FLATIndexConfig.__required_keys__ == frozenset(
            {"index_type", "nlist", "nprobe"}
        )
        assert IVF_FLATIndexConfig.__optional_keys__ == frozenset({"metric_type"})

    def test_field_types(self) -> None:
        hints = get_type_hints(IVF_FLATIndexConfig)
        assert hints["nlist"] is int
        assert hints["nprobe"] is int

    def test_construction(self) -> None:
        cfg: IVF_FLATIndexConfig = IVF_FLATIndexConfig(
            index_type="IVF_FLAT",
            nlist=128,
            nprobe=8,
        )
        assert cfg["nlist"] == 128
        assert cfg["nprobe"] == 8


class TestIVF_PQIndexConfig(unittest.TestCase):
    def test_is_typed_dict(self) -> None:
        assert typing.is_typeddict(IVF_PQIndexConfig)

    def test_index_type_literal(self) -> None:
        hints = get_type_hints(IVF_PQIndexConfig, include_extras=True)
        assert typing.get_args(hints["index_type"]) == ("IVF_PQ",)

    def test_optional_keys(self) -> None:
        assert IVF_PQIndexConfig.__required_keys__ == frozenset(
            {"index_type", "nlist", "m", "nbits", "nprobe"}
        )
        assert IVF_PQIndexConfig.__optional_keys__ == frozenset({"metric_type"})

    def test_field_types(self) -> None:
        hints = get_type_hints(IVF_PQIndexConfig)
        assert hints["nlist"] is int
        assert hints["m"] is int
        assert hints["nbits"] is int
        assert hints["nprobe"] is int

    def test_construction(self) -> None:
        cfg: IVF_PQIndexConfig = IVF_PQIndexConfig(
            index_type="IVF_PQ",
            nlist=128,
            m=32,
            nbits=8,
            nprobe=8,
        )
        assert cfg["m"] == 32
        assert cfg["nbits"] == 8


class TestIVF_PQFSIIndexConfig(unittest.TestCase):
    def test_is_typed_dict(self) -> None:
        assert typing.is_typeddict(IVF_PQFSIIndexConfig)

    def test_index_type_literal(self) -> None:
        hints = get_type_hints(IVF_PQFSIIndexConfig, include_extras=True)
        assert typing.get_args(hints["index_type"]) == ("IVF_PQFS",)

    def test_optional_keys(self) -> None:
        assert IVF_PQFSIIndexConfig.__required_keys__ == frozenset(
            {"index_type", "nlist", "m", "nprobe"}
        )
        assert IVF_PQFSIIndexConfig.__optional_keys__ == frozenset({"metric_type"})

    def test_field_types(self) -> None:
        hints = get_type_hints(IVF_PQFSIIndexConfig)
        assert hints["nlist"] is int
        assert hints["m"] is int
        assert hints["nprobe"] is int

    def test_construction(self) -> None:
        cfg: IVF_PQFSIIndexConfig = IVF_PQFSIIndexConfig(
            index_type="IVF_PQFS",
            nlist=128,
            m=32,
            nprobe=8,
        )
        assert cfg["index_type"] == "IVF_PQFS"
        assert cfg["m"] == 32


class TestHNSW_FLATIndexConfig(unittest.TestCase):
    def test_is_typed_dict(self) -> None:
        assert typing.is_typeddict(HNSW_FLATIndexConfig)

    def test_index_type_literal(self) -> None:
        hints = get_type_hints(HNSW_FLATIndexConfig, include_extras=True)
        assert typing.get_args(hints["index_type"]) == ("HNSW_FLAT",)

    def test_optional_keys(self) -> None:
        assert HNSW_FLATIndexConfig.__required_keys__ == frozenset(
            {"index_type", "M", "efConstruction", "ef"}
        )
        assert HNSW_FLATIndexConfig.__optional_keys__ == frozenset({"metric_type"})

    def test_field_types(self) -> None:
        hints = get_type_hints(HNSW_FLATIndexConfig)
        assert hints["M"] is int
        assert hints["efConstruction"] is int
        assert hints["ef"] is int

    def test_construction(self) -> None:
        cfg: HNSW_FLATIndexConfig = HNSW_FLATIndexConfig(
            index_type="HNSW_FLAT",
            M=30,
            efConstruction=40,
            ef=16,
        )
        assert cfg["M"] == 30
        assert cfg["efConstruction"] == 40
        assert cfg["ef"] == 16


class TestHNSW_PQIndexConfig(unittest.TestCase):
    def test_is_typed_dict(self) -> None:
        assert typing.is_typeddict(HNSW_PQIndexConfig)

    def test_index_type_literal(self) -> None:
        hints = get_type_hints(HNSW_PQIndexConfig, include_extras=True)
        assert typing.get_args(hints["index_type"]) == ("HNSW_PQ",)

    def test_optional_keys(self) -> None:
        assert HNSW_PQIndexConfig.__required_keys__ == frozenset(
            {"index_type", "M", "efConstruction", "m", "nbits", "ef"}
        )
        assert HNSW_PQIndexConfig.__optional_keys__ == frozenset({"metric_type"})

    def test_field_types(self) -> None:
        hints = get_type_hints(HNSW_PQIndexConfig)
        assert hints["M"] is int
        assert hints["efConstruction"] is int
        assert hints["m"] is int
        assert hints["nbits"] is int
        assert hints["ef"] is int

    def test_construction(self) -> None:
        cfg: HNSW_PQIndexConfig = HNSW_PQIndexConfig(
            index_type="HNSW_PQ",
            M=30,
            efConstruction=40,
            m=32,
            nbits=8,
            ef=16,
        )
        assert cfg["m"] == 32
        assert cfg["nbits"] == 8


class TestIndexConfigInteroperability(unittest.TestCase):
    """All configs are dicts and JSON-serializable with string values."""

    def test_all_are_dicts(self) -> None:
        for cls in (
            ANNIndexConfig,
            AUTOIndexConfig,
            FLATIndexConfig,
            IVF_FLATIndexConfig,
            IVF_PQIndexConfig,
            IVF_PQFSIIndexConfig,
            HNSW_FLATIndexConfig,
            HNSW_PQIndexConfig,
        ):
            assert issubclass(cls, dict)

    def test_configs_accept_plain_dict_literal(self) -> None:
        cfg: ANNIndexConfig = {"index_type": "IVF_PQ"}
        assert cfg["index_type"] == "IVF_PQ"

    def test_json_serializable(self) -> None:
        import json

        cfg: HNSW_FLATIndexConfig = HNSW_FLATIndexConfig(
            index_type="HNSW_FLAT",
            M=30,
            efConstruction=40,
            ef=16,
        )
        payload = json.dumps(cfg)
        assert json.loads(payload) == {
            "index_type": "HNSW_FLAT",
            "M": 30,
            "efConstruction": 40,
            "ef": 16,
        }


if __name__ == "__main__":
    unittest.main()
