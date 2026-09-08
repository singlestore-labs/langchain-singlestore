from typing import Literal, TypedDict

from ._utils import DistanceStrategy


class ANNIndexConfig(TypedDict, total=False):
    """Configuration for vector index in SingleStore store."""

    index_type: Literal[
        "AUTO", "FLAT", "IVF_FLAT", "IVF_PQ", "IVF_PQFS", "HNSW_FLAT", "HNSW_PQ"
    ]
    """
    "AUTO" for automatic selection,
    "FLAT" for Flat index,
    "IVF_FLAT" for Inverted File Flat,
    "IVF_PQ" for Inverted File Product Quantization,
    "IVF_PQFS" for Inverted File Product Quantization with Fine Search,
    "HNSW_FLAT" for Hierarchical Navigable Small World Flat,
    "HNSW_PQ" for Hierarchical Navigable Small World Product Quantization.
    """
    metric_type: Literal[
        DistanceStrategy.EUCLIDEAN_DISTANCE, DistanceStrategy.DOT_PRODUCT
    ]
    """
    Metric type for the vector index.
    "EUCLIDEAN_DISTANCE" for Euclidean distance,
    "DOT_PRODUCT" for dot product.
    Default is "DOT_PRODUCT".
    """


class AUTOIndexConfig(ANNIndexConfig):
    index_type: Literal["AUTO"]  # type: ignore[misc]


class FLATIndexConfig(ANNIndexConfig):
    index_type: Literal["FLAT"]  # type: ignore[misc]


class IVF_FLATIndexConfig(ANNIndexConfig):
    index_type: Literal["IVF_FLAT"]  # type: ignore[misc]
    nlist: int
    """number of inverted lists (number of clusters) created during index build.
    1 <= nlist <= 65536. Default to 128."""
    nprobe: int
    """number of inverted lists to probe during search. 1 <= nprobe <= nlist.
    Default to 8."""


class IVF_PQIndexConfig(ANNIndexConfig):
    index_type: Literal["IVF_PQ"]  # type: ignore[misc]
    nlist: int
    """number of inverted lists (number of clusters) created during index build.
    1 <= nlist <= 65536. Default to 128."""
    m: int
    """number of subquantizers used in product quantization.
    Dimensions % m must equal 0. Default to 32."""
    nbits: int
    """number of bits per quantization index. 1 <= nbits <= 16. Default to 8."""
    nprobe: int
    """number of probes at query time. 1 <= nprobe <= 65536. Default to 8."""


class IVF_PQFSIIndexConfig(ANNIndexConfig):
    index_type: Literal["IVF_PQFS"]  # type: ignore[misc]
    nlist: int
    """number of inverted lists (number of clusters) created during index build.
    1 <= nlist <= 65536. Default to 128."""
    m: int
    """number of subquantizers used in product quantization.
    Dimensions % m must equal 0. Default to 32."""
    nprobe: int
    """number of inverted lists to probe during search. 1 <= nprobe <= nlist.
    Default to 8."""


class HNSW_FLATIndexConfig(ANNIndexConfig):
    index_type: Literal["HNSW_FLAT"]  # type: ignore[misc]
    M: int
    """number of neighbors. 1 <= M <= 2048. Default to 30."""
    efConstruction: int
    """expansion factor at construction time.
    1 <= efConstruction <= 65536. Default to 40."""
    ef: int
    """expansion factor at search time. 1 <= ef <= 65536. Default to 16."""


class HNSW_PQIndexConfig(ANNIndexConfig):
    index_type: Literal["HNSW_PQ"]  # type: ignore[misc]
    M: int
    """number of neighbors. 1 <= M <= 2048. Default to 30."""
    efConstruction: int
    """expansion factor at construction time.
    1 <= efConstruction <= 65536. Default to 40."""
    m: int
    """number of sub-quantizers. dimensions % m must equal 0. Default to 32."""
    nbits: int
    """number of bits per quantization index. 1 <= nbits <= 16. Default to 8."""
    ef: int
    """expansion factor at search time. 1 <= ef <= 65536. Default to 16."""
