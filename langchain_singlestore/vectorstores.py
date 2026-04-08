"""SingleStore vector store"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import singlestoredb as s2
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from sqlalchemy.pool import QueuePool

from langchain_singlestore._filter import FilterTypedDict, _parse_filter
from langchain_singlestore._utils import (
    DistanceStrategy,
    FullTextIndexVersion,
    FullTextScoringMode,
    set_connector_attributes,
)

VST = TypeVar("VST", bound=VectorStore)

DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.DOT_PRODUCT
DEFAULT_FULL_TEXT_INDEX_VERSION = FullTextIndexVersion.V1

ORDERING_DIRECTIVE: dict = {
    DistanceStrategy.EUCLIDEAN_DISTANCE: "",
    DistanceStrategy.DOT_PRODUCT: "DESC",
}


def _is_advanced_filter(filter_dict: dict) -> bool:
    """Detect if filter uses advanced FilterTypedDict syntax (with operators).

    Advanced filters contain keys like $and, $or, or nested dicts with operator keys
    like $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $exists.

    Simple filters just have field names mapping to values.

    Args:
        filter_dict: The filter dictionary to check

    Returns:
        bool: True if filter uses advanced syntax, False if simple nested dict
    """
    if not filter_dict:
        return False

    # Check for direct operators at root level
    if any(key.startswith("$") for key in filter_dict.keys()):
        return True

    # Check for operators in nested dicts
    for value in filter_dict.values():
        if isinstance(value, dict):
            if any(key.startswith("$") for key in value.keys()):
                return True

    return False


def _apply_filter_to_where_clause(
    metadata_field: str,
    filter_dict: Optional[Union[dict, FilterTypedDict]],
    where_clause_values: List[Any],
    arguments: List[str],
) -> None:
    """Apply metadata filter to WHERE clause.

    Supports both simple nested dicts and advanced FilterTypedDict syntax with
    operators like $eq, $gt, $and, $or, etc.

    Args:
        metadata_field: Name of the metadata field in the database

        filter_dict: Filter specification (simple dict or FilterTypedDict)

        where_clause_values: List to accumulate parameter values for SQL

        arguments: List to accumulate SQL WHERE clause conditions
    """
    if not filter_dict:
        return

    if _is_advanced_filter(filter_dict):
        # Use advanced filter parsing with operators
        try:
            filter_query, filter_params = _parse_filter(filter_dict, metadata_field)  # type: ignore[arg-type]
            arguments.append(filter_query)
            where_clause_values.extend(filter_params)
        except (ValueError, KeyError) as e:
            raise ValueError(
                f"Invalid advanced filter specification: {e}. "
                "See documentation for FilterTypedDict syntax."
            ) from e
    else:
        # Use simple nested dict filtering (existing behavior)
        def build_where_clause(
            where_clause_values_inner: List[Any],
            sub_filter: dict,
            prefix_args: Optional[List[str]] = None,
        ) -> None:
            prefix_args = prefix_args or []
            for key in sub_filter.keys():
                if isinstance(sub_filter[key], dict):
                    build_where_clause(
                        where_clause_values_inner, sub_filter[key], prefix_args + [key]
                    )
                else:
                    arguments.append(
                        "JSON_EXTRACT_JSON({}, {}) = %s".format(
                            metadata_field,
                            ", ".join(["%s"] * (len(prefix_args) + 1)),
                        )
                    )
                    where_clause_values_inner += prefix_args + [key]
                    where_clause_values_inner.append(json.dumps(sub_filter[key]))

        build_where_clause(where_clause_values, filter_dict)


class SingleStoreVectorStore(VectorStore):
    """`SingleStore` vector store.

    The prerequisite for using this class is the installation of the ``singlestoredb``
    Python package.

    The SingleStore vectorstore can be created by providing an embedding function and
    the relevant parameters for the database connection, connection pool, and
    optionally, the names of the table and the fields to use.
    """

    class SearchStrategy(str, Enum):
        """Enumerator of the Search strategies for searching in the vectorstore."""

        VECTOR_ONLY = "VECTOR_ONLY"
        TEXT_ONLY = "TEXT_ONLY"
        FILTER_BY_TEXT = "FILTER_BY_TEXT"
        FILTER_BY_VECTOR = "FILTER_BY_VECTOR"
        WEIGHTED_SUM = "WEIGHTED_SUM"

    def _get_connection(self: SingleStoreVectorStore) -> Any:
        return s2.connect(**self.connection_kwargs)

    def __init__(
        self,
        embedding: Embeddings,
        *,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        table_name: str = "embeddings",
        content_field: str = "content",
        metadata_field: str = "metadata",
        vector_field: str = "vector",
        id_field: str = "id",
        use_vector_index: bool = False,
        vector_index_name: str = "",
        vector_index_options: Optional[dict] = None,
        vector_size: int = 1536,
        use_full_text_search: bool = False,
        full_text_index_version: FullTextIndexVersion = DEFAULT_FULL_TEXT_INDEX_VERSION,
        pool_size: int = 5,
        max_overflow: int = 10,
        timeout: float = 30,
        **kwargs: Any,
    ):
        """Initialize with necessary components.

        Args:
            embedding (Embeddings): A text embedding model.

            distance_strategy (DistanceStrategy, optional):
                Determines the strategy employed for calculating
                the distance between vectors in the embedding space.

                Defaults to DOT_PRODUCT.

                Available options are:

                - DOT_PRODUCT: Computes the scalar product of two vectors.
                    This is the default behavior

                - EUCLIDEAN_DISTANCE: Computes the Euclidean distance between
                    two vectors. This metric considers the geometric distance in
                    the vector space, and might be more suitable for embeddings
                    that rely on spatial relationships. This metric is not
                    compatible with the WEIGHTED_SUM search strategy.

            table_name (str, optional): Specifies the name of the table in use.
                Defaults to "embeddings".

            content_field (str, optional): Specifies the field to store the content.
                Defaults to "content".

            metadata_field (str, optional): Specifies the field to store metadata.
                Defaults to "metadata".

            vector_field (str, optional): Specifies the field to store the vector.
                Defaults to "vector".

            id_field (str, optional): Specifies the field to store the id.
                Defaults to "id".

            use_vector_index (bool, optional): Toggles the use of a vector index.
                Works only with SingleStore 8.5 or later. Defaults to False.
                If set to True, vector_size parameter is required to be set to
                a proper value.

            vector_index_name (str, optional): Specifies the name of the vector index.
                Defaults to empty. Will be ignored if use_vector_index is set to False.

            vector_index_options (dict, optional): Specifies the options for
                the vector index. Defaults to {}.
                Will be ignored if use_vector_index is set to False. The options are:
                index_type (str, optional): Specifies the type of the index.

                Defaults to IVF_PQFS.

                For more options, please refer to the SingleStore documentation:
                https://docs.singlestore.com/cloud/reference/sql-reference/vector-functions/vector-indexing/

            vector_size (int, optional): Specifies the size of the vector.
                Defaults to 1536. Required if use_vector_index is set to True.
                Should be set to the same value as the size of the vectors
                stored in the vector_field.

            use_full_text_search (bool, optional): Toggles the use a full-text index
                on the document content. Defaults to False. If set to True, the table
                will be created with a full-text index on the content field,
                and the similarity_search method will allow using TEXT_ONLY,
                FILTER_BY_TEXT, FILTER_BY_VECTOR, and WEIGHTED_SUM search strategies.
                If set to False, the similarity_search method will only allow
                VECTOR_ONLY search strategy.

            full_text_index_version (FullTextIndexVersion, optional): Specifies the
                version of the full-text index to use. Defaults to V1. This parameter
                has effect only if use_full_text_search is set to True.
                Available options are:
                - V1: Uses the original full-text index implementation. This version
                    is compatible with all SingleStore versions that support
                    full-text search.
                - V2: Uses the new full-text index implementation that is available
                    in SingleStore 8.7 and later. This version offers improved
                    performance and additional features, but is not compatible with
                    SingleStore versions prior to 8.7.


            Following arguments pertain to the connection pool:

            pool_size (int, optional): Determines the number of active connections in
                the pool. Defaults to 5.

            max_overflow (int, optional): Determines the maximum number of connections
                allowed beyond the pool_size. Defaults to 10.

            timeout (float, optional): Specifies the maximum wait time in seconds for
                establishing a connection. Defaults to 30.


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

        Examples:
            Basic Usage:

            .. code-block:: python

                from langchain_openai import OpenAIEmbeddings
                from langchain_singlestore import SingleStoreVectorStore

                vectorstore = SingleStoreVectorStore(
                    OpenAIEmbeddings(),
                    host="https://user:password@127.0.0.1:3306/database"
                )

            Advanced Usage:

            .. code-block:: python

                from langchain_openai import OpenAIEmbeddings
                from langchain_singlestore import (
                    SingleStoreVectorStore,
                    DistanceStrategy,
                )

                vectorstore = SingleStoreVectorStore(
                    OpenAIEmbeddings(),
                    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
                    host="127.0.0.1",
                    port=3306,
                    user="user",
                    password="password",
                    database="db",
                    table_name="my_custom_table",
                    pool_size=10,
                    timeout=60,
                )

            Using environment variables:

            .. code-block:: python

                from langchain_openai import OpenAIEmbeddings
                from langchain_singlestore import SingleStoreVectorStore

                os.environ['SingleStore_URL'] = 'me:p455w0rd@s2-host.com/my_db'
                vectorstore = SingleStoreVectorStore(OpenAIEmbeddings())

            Using vector index:

            .. code-block:: python

                from langchain_openai import OpenAIEmbeddings
                from langchain_singlestore import SingleStoreVectorStore

                os.environ['SingleStore_URL'] = 'me:p455w0rd@s2-host.com/my_db'
                vectorstore = SingleStoreVectorStore(
                    OpenAIEmbeddings(),
                    use_vector_index=True,
                )

            Using full-text index:

            .. code-block:: python

                from langchain_openai import OpenAIEmbeddings
                from langchain_singlestore import SingleStoreVectorStore

                os.environ['SingleStore_URL'] = 'me:p455w0rd@s2-host.com/my_db'
                vectorstore = SingleStoreVectorStore(
                    OpenAIEmbeddings(),
                    use_full_text_search=True,
                )
        """

        self.embedding = embedding
        self.distance_strategy = distance_strategy
        self.table_name = self._sanitize_input(table_name)
        self.content_field = self._sanitize_input(content_field)
        self.metadata_field = self._sanitize_input(metadata_field)
        self.vector_field = self._sanitize_input(vector_field)
        self.id_field = self._sanitize_input(id_field)

        self.use_vector_index = bool(use_vector_index)
        self.vector_index_name = self._sanitize_input(vector_index_name)
        self.vector_index_options = dict(vector_index_options or {})
        self.vector_index_options["metric_type"] = self.distance_strategy
        self.vector_size = int(vector_size)

        self.use_full_text_search = bool(use_full_text_search)
        self.full_text_index_version = full_text_index_version

        # Pass the rest of the kwargs to the connection.
        self.connection_kwargs = kwargs

        # Add connection attributes to the connection kwargs.
        set_connector_attributes(self.connection_kwargs)

        # Create connection pool.
        self.connection_pool = QueuePool(
            self._get_connection,
            max_overflow=max_overflow,
            pool_size=pool_size,
            timeout=timeout,
        )
        self._create_table()

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    def _sanitize_input(self, input_str: str) -> str:
        # Remove characters that are not alphanumeric or underscores
        return re.sub(r"[^a-zA-Z0-9_]", "", input_str)

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return self._max_inner_product_relevance_score_fn

    def _create_table(self: SingleStoreVectorStore) -> None:
        """Create table if it doesn't exist."""
        conn = self.connection_pool.connect()
        try:
            cur = conn.cursor()
            try:
                full_text_index = ""
                if self.use_full_text_search:
                    if self.full_text_index_version == FullTextIndexVersion.V2:
                        full_text_index = ", FULLTEXT USING VERSION 2 ({})".format(
                            self.content_field
                        )
                    else:
                        full_text_index = ", FULLTEXT({})".format(self.content_field)
                if self.use_vector_index:
                    index_options = ""
                    if self.vector_index_options and len(self.vector_index_options) > 0:
                        index_options = "INDEX_OPTIONS '{}'".format(
                            json.dumps(self.vector_index_options)
                        )
                    cur.execute(
                        """CREATE TABLE IF NOT EXISTS {}
                        ({} BIGINT AUTO_INCREMENT PRIMARY KEY, {} LONGTEXT CHARACTER
                        SET utf8mb4 COLLATE utf8mb4_general_ci, {} VECTOR({}, F32)
                        NOT NULL, {} JSON, VECTOR INDEX {} ({}) {}{});""".format(
                            self.table_name,
                            self.id_field,
                            self.content_field,
                            self.vector_field,
                            self.vector_size,
                            self.metadata_field,
                            self.vector_index_name,
                            self.vector_field,
                            index_options,
                            full_text_index,
                        ),
                    )
                else:
                    cur.execute(
                        """CREATE TABLE IF NOT EXISTS {}
                        ({} BIGINT AUTO_INCREMENT PRIMARY KEY, {} LONGTEXT CHARACTER
                        SET utf8mb4 COLLATE utf8mb4_general_ci, {} BLOB, {} JSON{});
                        """.format(
                            self.table_name,
                            self.id_field,
                            self.content_field,
                            self.vector_field,
                            self.metadata_field,
                            full_text_index,
                        ),
                    )
            finally:
                cur.close()
        finally:
            conn.close()

    def add_images(
        self,
        uris: List[str],
        metadatas: Optional[List[dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run images through the embeddings and add to the vectorstore.

        Args:
            uris (List[str]): File path to images.
                Each URI will be added to the vectorstore as document content.

            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
                Defaults to None.

            embeddings (Optional[List[List[float]]], optional): Optional pre-generated
                embeddings. Defaults to None.

        Returns:
            List[str]: list of document ids added to the vectorstore
        """
        # Set embeddings
        if (
            embeddings is None
            and self.embedding is not None
            and hasattr(self.embedding, "embed_image")
        ):
            embeddings = self.embedding.embed_image(uris=uris)
        return self.add_texts(uris, metadatas, embeddings, **kwargs)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add more texts to the vectorstore.

        Args:
            texts (Iterable[str]): Iterable of strings/text to add to the vectorstore.

            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
                Defaults to None.

            embeddings (Optional[List[List[float]]], optional): Optional pre-generated
                embeddings. Defaults to None.

        Returns:
            List[str]: list of document ids added to the vectorstore
        """
        result_ids: List[str] = []
        conn = self.connection_pool.connect()
        try:
            cur = conn.cursor()
            try:
                # Write data to singlestore db
                for i, text in enumerate(texts):
                    # Use provided values by default or fallback
                    metadata = metadatas[i] if metadatas else {}
                    embedding = (
                        embeddings[i]
                        if embeddings
                        else self.embedding.embed_documents([text])[0]
                    )
                    if not ids or len(ids) <= i:
                        cur.execute(
                            """INSERT INTO {}({}, {}, {})
                            VALUES (%s, JSON_ARRAY_PACK(%s), %s)""".format(
                                self.table_name,
                                self.content_field,
                                self.vector_field,
                                self.metadata_field,
                            ),
                            (
                                text,
                                "[{}]".format(",".join(map(str, embedding))),
                                json.dumps(metadata),
                            ),
                        )
                        cur.execute("SELECT LAST_INSERT_ID();")
                        row = cur.fetchone()
                        if row:
                            result_ids.append(str(row[0]))
                    else:
                        cur.execute(
                            """INSERT INTO {}({}, {}, {}, {})
                            VALUES (%s, %s, JSON_ARRAY_PACK(%s), %s)
                            ON DUPLICATE KEY UPDATE
                                {} = VALUES({}),
                                {} = VALUES({}),
                                {} = VALUES({})""".format(
                                self.table_name,
                                self.id_field,
                                self.content_field,
                                self.vector_field,
                                self.metadata_field,
                                self.content_field,
                                self.content_field,
                                self.vector_field,
                                self.vector_field,
                                self.metadata_field,
                                self.metadata_field,
                            ),
                            (
                                ids[i],
                                text,
                                "[{}]".format(",".join(map(str, embedding))),
                                json.dumps(metadata),
                            ),
                        )
                        result_ids.append(ids[i])
                if self.use_vector_index or self.use_full_text_search:
                    cur.execute("OPTIMIZE TABLE {} FLUSH;".format(self.table_name))
            finally:
                cur.close()
        finally:
            conn.close()
        return result_ids

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> bool | None:
        """Delete documents from the vectorstore.

        Args:
            ids (List[str], optional): List of document ids to delete.
                If None, all documents will be deleted. Defaults to None.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        if not ids:
            return True

        conn = self.connection_pool.connect()
        try:
            cur = conn.cursor()
            try:
                cur.execute(
                    "DELETE FROM {} WHERE {} IN ({})".format(
                        self.table_name, self.id_field, ",".join(ids)
                    )
                )
                if self.use_vector_index or self.use_full_text_search:
                    cur.execute("OPTIMIZE TABLE {} FLUSH;".format(self.table_name))
            finally:
                cur.close()
        finally:
            conn.close()
        return True

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Union[dict, FilterTypedDict]] = None,
        query_embedding: Optional[List[float]] = None,
        search_strategy: SearchStrategy = SearchStrategy.VECTOR_ONLY,
        filter_threshold: float = 0,
        text_weight: float = 0.5,
        vector_weight: float = 0.5,
        vector_select_count_multiplier: int = 10,
        full_text_scoring_mode: FullTextScoringMode = FullTextScoringMode.MATCH,
        **kwargs: Any,
    ) -> List[Document]:
        """Returns the most similar indexed documents to the query text.

        Uses the configured distance_strategy (DOT_PRODUCT or EUCLIDEAN_DISTANCE)
        to measure similarity between vectors.

        Args:
            query (str): The query text for which to find similar documents.

            k (int): The number of documents to return. Default is 4.

            filter (dict or FilterTypedDict, optional): A dictionary to filter by
                metadata. Can be either:

                1. Simple dict: ``{"field": "value", "status": "active"}``
                   (existing nested dict format)

                2. FilterTypedDict: Advanced filtering with operators:
                   - Comparisons: ``{"field": {"$eq": "value"}}``
                   - Numeric: ``{"age": {"$gt": 18}}``
                   - Collections: ``{"tags": {"$in": ["a", "b"]}}``
                   - Logical: ``{"$and": [{...}, {...}]}``

                Default is None.

            query_embedding (List[float], optional): Pre-computed embedding for
                the query.cIf not provided, the embedding will be computed using
                the configured embedding function.

            search_strategy (SearchStrategy): The search strategy to use.
                Default is SearchStrategy.VECTOR_ONLY.

                Available options are:
                - SearchStrategy.VECTOR_ONLY: Searches only by vector similarity.

                - SearchStrategy.TEXT_ONLY: Searches only by text similarity. This
                    option is only available if use_full_text_search is True.

                - SearchStrategy.FILTER_BY_TEXT: Filters by text similarity and
                    searches by vector similarity. This option is only available if
                    use_full_text_search is True.

                - SearchStrategy.FILTER_BY_VECTOR: Filters by vector similarity and
                    searches by text similarity. This option is only available if
                    use_full_text_search is True.

                - SearchStrategy.WEIGHTED_SUM: Searches by a weighted sum of text and
                    vector similarity. This option is only available if
                    use_full_text_search is True and distance_strategy is DOT_PRODUCT.

            filter_threshold (float): The threshold for filtering by text or vector
                similarity. Default is 0. This option has effect only if search_strategy
                is SearchStrategy.FILTER_BY_TEXT or SearchStrategy.FILTER_BY_VECTOR.

            text_weight (float): The weight of text similarity in the weighted sum
                search strategy. Default is 0.5. This option has effect only if
                search_strategy is SearchStrategy.WEIGHTED_SUM.

            vector_weight (float): The weight of vector similarity in the weighted sum
                search strategy. Default is 0.5. This option has effect only if
                search_strategy is SearchStrategy.WEIGHTED_SUM.

            vector_select_count_multiplier (int): The multiplier for the number of
                vectors to select when using the vector index. Default is 10.
                This parameter has effect only if use_vector_index is True and
                search_strategy is SearchStrategy.WEIGHTED_SUM or
                SearchStrategy.FILTER_BY_TEXT.
                The number of vectors selected will
                be k * vector_select_count_multiplier.
                This is needed due to the limitations of the vector index.

            full_text_scoring_mode (FullTextScoringMode): Specifies the algorithm
                used to calculate text similarity scores. Defaults to
                FullTextScoringMode.MATCH. This parameter only takes effect when
                search_strategy is TEXT_ONLY, FILTER_BY_TEXT, FILTER_BY_VECTOR, or
                WEIGHTED_SUM.

                Available options:

                - MATCH: Uses SingleStore's native MATCH() AGAINST() function.
                    Returns a relevance score based on term frequency in the
                    document. Compatible with both full-text index V1 and V2.

                - BM25: Uses the BM25 (Best Matching 25) ranking algorithm.
                    Provides more accurate relevance scoring by considering
                    term frequency (TF), inverse document frequency (IDF), and
                    document length normalization. Requires full-text index V2.

                - BM25_GLOBAL: Similar to BM25, but computes IDF statistics
                    across the entire dataset rather than per-partition. This
                    can provide more consistent scoring in distributed
                    environments but may have higher computational cost.
                    Requires full-text index V2.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.

        Examples:

            Basic Usage:
            .. code-block:: python

                from langchain_singlestore import SingleStoreVectorStore
                from langchain_openai import OpenAIEmbeddings

                s2 = SingleStoreVectorStore.from_documents(
                    docs,
                    OpenAIEmbeddings(),
                    host="username:password@localhost:3306/database"
                )
                results = s2.similarity_search("query text", 1,
                                    {"metadata_field": "metadata_value"})

            Different Search Strategies:
            .. code-block:: python

                from langchain_singlestore import SingleStoreVectorStore
                from langchain_openai import OpenAIEmbeddings

                s2 = SingleStoreVectorStore.from_documents(
                    docs,
                    OpenAIEmbeddings(),
                    host="username:password@localhost:3306/database",
                    use_full_text_search=True,
                    use_vector_index=True,
                )
                results = s2.similarity_search("query text", 1,
                        search_strategy=SingleStoreVectorStore.SearchStrategy.FILTER_BY_TEXT,
                        filter_threshold=0.5)

            Weighted Sum Search Strategy:
            .. code-block:: python

                from langchain_singlestore import (
                    SingleStoreVectorStore,
                    FullTextScoringMode,
                    FullTextIndexVersion,
                )
                from langchain_openai import OpenAIEmbeddings

                s2 = SingleStoreVectorStore.from_documents(
                    docs,
                    OpenAIEmbeddings(),
                    host="username:password@localhost:3306/database",
                    use_full_text_search=True,
                    use_vector_index=True,
                    full_text_index_version=FullTextIndexVersion.V2,
                )
                results = s2.similarity_search("query text", 1,
                    search_strategy=SingleStoreVectorStore.SearchStrategy.WEIGHTED_SUM,
                    text_weight=0.3,
                    vector_weight=0.7,
                    full_text_scoring_mode=FullTextScoringMode.BM25,
                )
        """
        docs_and_scores = self.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
            query_embedding=query_embedding,
            search_strategy=search_strategy,
            filter_threshold=filter_threshold,
            text_weight=text_weight,
            vector_weight=vector_weight,
            vector_select_count_multiplier=vector_select_count_multiplier,
            full_text_scoring_mode=full_text_scoring_mode,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    def _fulltext_scoring_mode_to_sql(
        self,
        mode: FullTextScoringMode,
        query: str,
    ) -> tuple[str, str]:
        """Convert the full text scoring mode to the corresponding SQL snippet."""
        match_arg = self.content_field
        search_query = query
        if self.full_text_index_version != FullTextIndexVersion.V1:
            search_query = "{}:({})".format(self.content_field, query)
            match_arg = self.table_name
            if mode == FullTextScoringMode.MATCH:
                match_arg = "TABLE {}".format(self.table_name)
        if mode == FullTextScoringMode.MATCH:
            return "MATCH ({}) AGAINST (%s)".format(match_arg), search_query
        else:
            return "{}({}, %s)".format(mode.value, match_arg), search_query

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Union[dict, FilterTypedDict]] = None,
        query_embedding: Optional[List[float]] = None,
        search_strategy: SearchStrategy = SearchStrategy.VECTOR_ONLY,
        filter_threshold: float = 0,
        text_weight: float = 0.5,
        vector_weight: float = 0.5,
        vector_select_count_multiplier: int = 10,
        full_text_scoring_mode: FullTextScoringMode = FullTextScoringMode.MATCH,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Uses the configured distance_strategy (DOT_PRODUCT or EUCLIDEAN_DISTANCE)
        to measure similarity between vectors.

        Args:
            query: Text to look up documents similar to.

            k: Number of Documents to return. Defaults to 4.

            filter: A dictionary to filter by metadata. Can be either:

                1. Simple dict: ``{"field": "value", "status": "active"}``
                   (existing nested dict format)

                2. FilterTypedDict: Advanced filtering with operators:
                   - Comparisons: ``{"field": {"$eq": "value"}}``
                   - Numeric: ``{"age": {"$gt": 18}}``
                   - Collections: ``{"tags": {"$in": ["a", "b"]}}``
                   - Logical: ``{"$and": [{...}, {...}]}``

                Defaults to None.

            query_embedding (List[float], optional): Pre-computed embedding for
                the query. If not provided, the embedding will be computed using
                the configured embedding function.

            search_strategy (SearchStrategy): The search strategy to use.
                Default is SearchStrategy.VECTOR_ONLY.

                Available options are:

                - SearchStrategy.VECTOR_ONLY: Searches only by vector similarity.

                - SearchStrategy.TEXT_ONLY: Searches only by text similarity. This
                    option is only available if use_full_text_search is True.

                - SearchStrategy.FILTER_BY_TEXT: Filters by text similarity and
                    searches by vector similarity. This option is only available if
                    use_full_text_search is True.

                - SearchStrategy.FILTER_BY_VECTOR: Filters by vector similarity and
                    searches by text similarity. This option is only available if
                    use_full_text_search is True.

                - SearchStrategy.WEIGHTED_SUM: Searches by a weighted sum of text and
                    vector similarity. This option is only available if
                    use_full_text_search is True and distance_strategy is DOT_PRODUCT.

            filter_threshold (float): The threshold for filtering by text or vector
                similarity. Default is 0. This option has effect only if search_strategy
                is SearchStrategy.FILTER_BY_TEXT or SearchStrategy.FILTER_BY_VECTOR.

            text_weight (float): The weight of text similarity in the weighted sum
                search strategy. Default is 0.5. This option has effect only if
                search_strategy is SearchStrategy.WEIGHTED_SUM.

            vector_weight (float): The weight of vector similarity in the weighted sum
                search strategy. Default is 0.5. This option has effect only if
                search_strategy is SearchStrategy.WEIGHTED_SUM.

            vector_select_count_multiplier (int): The multiplier for the number of
                vectors to select when using the vector index. Default is 10.
                This parameter has effect only if use_vector_index is True and
                search_strategy is SearchStrategy.WEIGHTED_SUM or
                SearchStrategy.FILTER_BY_TEXT.
                The number of vectors selected will
                be k * vector_select_count_multiplier.
                This is needed due to the limitations of the vector index.

            full_text_scoring_mode (FullTextScoringMode): Specifies the algorithm
                used to calculate text similarity scores. Defaults to
                FullTextScoringMode.MATCH. This parameter only takes effect when
                search_strategy is TEXT_ONLY, FILTER_BY_TEXT, FILTER_BY_VECTOR, or
                WEIGHTED_SUM.

                Available options:

                - MATCH: Uses SingleStore's native MATCH() AGAINST() function.
                    Returns a relevance score based on term frequency in the
                    document. Compatible with both full-text index V1 and V2.

                - BM25: Uses the BM25 (Best Matching 25) ranking algorithm.
                    Provides more accurate relevance scoring by considering
                    term frequency (TF), inverse document frequency (IDF), and
                    document length normalization. Requires full-text index V2.

                - BM25_GLOBAL: Similar to BM25, but computes IDF statistics
                    across the entire dataset rather than per-partition. This
                    can provide more consistent scoring in distributed
                    environments but may have higher computational cost.
                    Requires full-text index V2.

        Returns:
            List of Documents most similar to the query and score for each
            document.

        Raises:
            ValueError: If the search strategy is not supported with the
                distance strategy.

        Examples:
            Basic Usage:
            .. code-block:: python

                from langchain_singlestore import SingleStoreVectorStore
                from langchain_openai import OpenAIEmbeddings

                s2 = SingleStoreVectorStore.from_documents(
                    docs,
                    OpenAIEmbeddings(),
                    host="username:password@localhost:3306/database"
                )
                results = s2.similarity_search_with_score("query text", 1,
                                    {"metadata_field": "metadata_value"})

            Different Search Strategies:

            .. code-block:: python

                from langchain_singlestore import (
                    SingleStoreVectorStore,
                    FullTextIndexVersion,
                    FullTextScoringMode,
                )
                from langchain_openai import OpenAIEmbeddings

                s2 = SingleStoreVectorStore.from_documents(
                    docs,
                    OpenAIEmbeddings(),
                    host="username:password@localhost:3306/database",
                    use_full_text_search=True,
                    use_vector_index=True,
                    vector_size=3,
                    full_text_index_version=FullTextIndexVersion.V2,
                )
                results = s2.similarity_search_with_score(
                        "query text", 1,
                        query_embedding=[0.1, 0.2, 0.3], # Pre-computed embedding
                        search_strategy=SingleStoreVectorStore.SearchStrategy.FILTER_BY_VECTOR,
                        filter_threshold=0.5,
                        full_text_scoring_mode=FullTextScoringMode.BM25,
                )

            Weighted Sum Search Strategy:
            .. code-block:: python

                from langchain_singlestore import SingleStoreVectorStore
                from langchain_openai import OpenAIEmbeddings

                s2 = SingleStoreVectorStore.from_documents(
                    docs,
                    OpenAIEmbeddings(),
                    host="username:password@localhost:3306/database",
                    use_full_text_search=True,
                    use_vector_index=True,
                )
                results = s2.similarity_search_with_score(
                    "query text", 1,
                    search_strategy=SingleStoreVectorStore.SearchStrategy.WEIGHTED_SUM,
                    text_weight=0.3,
                    vector_weight=0.7,
                )
        """

        if (
            search_strategy != SingleStoreVectorStore.SearchStrategy.VECTOR_ONLY
            and not self.use_full_text_search
        ):
            raise ValueError(
                """Search strategy {} is not supported
                when use_full_text_search is False""".format(search_strategy)
            )

        if (
            search_strategy == SingleStoreVectorStore.SearchStrategy.WEIGHTED_SUM
            and self.distance_strategy != DistanceStrategy.DOT_PRODUCT
        ):
            raise ValueError(
                "Search strategy {} is not supported with distance strategy {}".format(
                    search_strategy, self.distance_strategy
                )
            )

        if (
            search_strategy != SingleStoreVectorStore.SearchStrategy.VECTOR_ONLY
            and full_text_scoring_mode != FullTextScoringMode.MATCH
            and self.full_text_index_version != FullTextIndexVersion.V2
        ):
            raise ValueError(
                "Scoring {} is not supported with full-text index version {}".format(
                    full_text_scoring_mode, self.full_text_index_version
                )
            )

        # Creates embedding vector from user query
        embedding = []
        if search_strategy != SingleStoreVectorStore.SearchStrategy.TEXT_ONLY:
            if query_embedding is not None:
                embedding = query_embedding
            else:
                embedding = self.embedding.embed_query(query)

        conn = self.connection_pool.connect()
        result = []
        where_clause: str = ""
        where_clause_values: List[Any] = []
        if filter or search_strategy in [
            SingleStoreVectorStore.SearchStrategy.FILTER_BY_TEXT,
            SingleStoreVectorStore.SearchStrategy.FILTER_BY_VECTOR,
        ]:
            where_clause = "WHERE "
            arguments = []

            if search_strategy == SingleStoreVectorStore.SearchStrategy.FILTER_BY_TEXT:
                function_sql, search_query = self._fulltext_scoring_mode_to_sql(
                    full_text_scoring_mode, query
                )
                arguments.append("{} > %s".format(function_sql))
                where_clause_values.append(search_query)
                where_clause_values.append(float(filter_threshold))

            if (
                search_strategy
                == SingleStoreVectorStore.SearchStrategy.FILTER_BY_VECTOR
            ):
                condition = "{}({}, JSON_ARRAY_PACK(%s)) ".format(
                    self.distance_strategy.name
                    if isinstance(self.distance_strategy, DistanceStrategy)
                    else self.distance_strategy,
                    self.vector_field,
                )
                if self.distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
                    condition += "< %s"
                else:
                    condition += "> %s"
                arguments.append(condition)
                where_clause_values.append("[{}]".format(",".join(map(str, embedding))))
                where_clause_values.append(float(filter_threshold))

            if filter:
                _apply_filter_to_where_clause(
                    self.metadata_field, filter, where_clause_values, arguments
                )
            where_clause += " AND ".join(arguments)

        try:
            cur = conn.cursor()
            try:
                if (
                    search_strategy == SingleStoreVectorStore.SearchStrategy.VECTOR_ONLY
                    or search_strategy
                    == SingleStoreVectorStore.SearchStrategy.FILTER_BY_TEXT
                ):
                    search_options = ""
                    if (
                        self.use_vector_index
                        and search_strategy
                        == SingleStoreVectorStore.SearchStrategy.FILTER_BY_TEXT
                    ):
                        search_options = "SEARCH_OPTIONS '{\"k\":%d}'" % (
                            k * vector_select_count_multiplier
                        )
                    cur.execute(
                        """SELECT {}, {}, {}, {}({}, JSON_ARRAY_PACK(%s)) as __score
                        FROM {} {} ORDER BY __score {}{} LIMIT %s""".format(
                            self.content_field,
                            self.metadata_field,
                            self.id_field,
                            self.distance_strategy.name
                            if isinstance(self.distance_strategy, DistanceStrategy)
                            else self.distance_strategy,
                            self.vector_field,
                            self.table_name,
                            where_clause,
                            search_options,
                            ORDERING_DIRECTIVE[self.distance_strategy],
                        ),
                        ("[{}]".format(",".join(map(str, embedding))),)
                        + tuple(where_clause_values)
                        + (k,),
                    )
                elif (
                    search_strategy
                    == SingleStoreVectorStore.SearchStrategy.FILTER_BY_VECTOR
                    or search_strategy
                    == SingleStoreVectorStore.SearchStrategy.TEXT_ONLY
                ):
                    fulltext_score_query, full_text_query = (
                        self._fulltext_scoring_mode_to_sql(
                            full_text_scoring_mode, query
                        )
                    )
                    cur.execute(
                        """SELECT {}, {}, {}, {} as __score
                        FROM {} {} ORDER BY __score DESC LIMIT %s""".format(
                            self.content_field,
                            self.metadata_field,
                            self.id_field,
                            fulltext_score_query,
                            self.table_name,
                            where_clause,
                        ),
                        (full_text_query,) + tuple(where_clause_values) + (k,),
                    )
                elif (
                    search_strategy
                    == SingleStoreVectorStore.SearchStrategy.WEIGHTED_SUM
                ):
                    fulltext_score_query, full_text_query = (
                        self._fulltext_scoring_mode_to_sql(
                            full_text_scoring_mode, query
                        )
                    )
                    cur.execute(
                        """SELECT {}, {}, r1.{} as {}, __score1 * %s + __score2 * %s
                        as __score FROM (
                            SELECT {}, {}, {}, {} as __score1
                        FROM {} {}) r1 FULL OUTER JOIN (
                            SELECT {}, {}({}, JSON_ARRAY_PACK(%s)) as __score2
                            FROM {} {} ORDER BY __score2 {} LIMIT %s
                        ) r2 ON r1.{} = r2.{} ORDER BY __score {} LIMIT %s""".format(
                            self.content_field,
                            self.metadata_field,
                            self.id_field,
                            self.id_field,
                            self.id_field,
                            self.content_field,
                            self.metadata_field,
                            fulltext_score_query,
                            self.table_name,
                            where_clause,
                            self.id_field,
                            self.distance_strategy.name
                            if isinstance(self.distance_strategy, DistanceStrategy)
                            else self.distance_strategy,
                            self.vector_field,
                            self.table_name,
                            where_clause,
                            ORDERING_DIRECTIVE[self.distance_strategy],
                            self.id_field,
                            self.id_field,
                            ORDERING_DIRECTIVE[self.distance_strategy],
                        ),
                        (text_weight, vector_weight, full_text_query)
                        + tuple(where_clause_values)
                        + ("[{}]".format(",".join(map(str, embedding))),)
                        + tuple(where_clause_values)
                        + (k * vector_select_count_multiplier, k),
                    )
                else:
                    raise ValueError(
                        "Invalid search strategy: {}".format(search_strategy)
                    )

                for row in cur.fetchall():
                    doc = Document(page_content=row[0], metadata=row[1], id=row[2])
                    result.append((doc, float(row[3])))
            finally:
                cur.close()
        finally:
            conn.close()
        return result

    @classmethod
    def from_texts(
        cls: Type[SingleStoreVectorStore],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        table_name: str = "embeddings",
        content_field: str = "content",
        metadata_field: str = "metadata",
        vector_field: str = "vector",
        id_field: str = "id",
        use_vector_index: bool = False,
        vector_index_name: str = "",
        vector_index_options: Optional[dict] = None,
        vector_size: int = 1536,
        use_full_text_search: bool = False,
        full_text_index_version: FullTextIndexVersion = DEFAULT_FULL_TEXT_INDEX_VERSION,
        pool_size: int = 5,
        max_overflow: int = 10,
        timeout: float = 30,
        **kwargs: Any,
    ) -> SingleStoreVectorStore:
        """Create a SingleStoreVectorStore vectorstore from raw documents.

        This is a user-friendly interface that:

            1. Embeds documents.

            2. Creates a new table for the embeddings in SingleStoreVectorStore.

            3. Adds the documents to the newly created table.

        This is intended to be a quick way to get started.
        Args:

            texts (List[str]): List of texts to add to the vectorstore.

            embedding (Embeddings): A text embedding model.

            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
                Defaults to None.

            embeddings (Optional[List[List[float]]], optional): Optional list of
              pre-computed embeddings. If not provided, embeddings will be computed
              using the provided embedding model.

            distance_strategy (DistanceStrategy, optional):
                Determines the strategy employed for calculating
                the distance between vectors in the embedding space.
                Defaults to DOT_PRODUCT.

                Available options are:

                - DOT_PRODUCT: Computes the scalar product of two vectors.
                    This is the default behavior

                - EUCLIDEAN_DISTANCE: Computes the Euclidean distance between
                    two vectors. This metric considers the geometric distance in
                    the vector space, and might be more suitable for embeddings
                    that rely on spatial relationships. This metric is not
                    compatible with the WEIGHTED_SUM search strategy.

            table_name (str, optional): Specifies the name of the table in use.
                Defaults to "embeddings".

            content_field (str, optional): Specifies the field to store the content.
                Defaults to "content".

            metadata_field (str, optional): Specifies the field to store metadata.
                Defaults to "metadata".

            vector_field (str, optional): Specifies the field to store the vector.
                Defaults to "vector".

            id_field (str, optional): Specifies the field to store the id.
                Defaults to "id".

            use_vector_index (bool, optional): Toggles the use of a vector index.
                Works only with SingleStore 8.5 or later. Defaults to False.
                If set to True, vector_size parameter is required to be set to
                a proper value.

            vector_index_name (str, optional): Specifies the name of the vector index.
                Defaults to empty. Will be ignored if use_vector_index is set to False.

            vector_index_options (dict, optional): Specifies the options for
                the vector index. Defaults to {}.
                Will be ignored if use_vector_index is set to False. The options are:
                index_type (str, optional): Specifies the type of the index.
                    Defaults to IVF_PQFS.
                For more options, please refer to the SingleStore documentation:
                https://docs.singlestore.com/cloud/reference/sql-reference/vector-functions/vector-indexing/

            vector_size (int, optional): Specifies the size of the vector.
                Defaults to 1536. Required if use_vector_index is set to True.
                Should be set to the same value as the size of the vectors
                stored in the vector_field.

            use_full_text_search (bool, optional): Toggles the use a full-text index
                on the document content. Defaults to False. If set to True, the table
                will be created with a full-text index on the content field,
                and the similarity_search method will allow using TEXT_ONLY,
                FILTER_BY_TEXT, FILTER_BY_VECTOR, and WEIGHTED_SUM search strategies.
                If set to False, the similarity_search method will only allow
                VECTOR_ONLY search strategy.

            full_text_index_version (FullTextIndexVersion, optional): Determines the
                version of the full-text index to use. Defaults to V1.

                Available options are:

                - V1: Uses the original full-text index implementation. This version
                    is compatible with all versions of SingleStore, but does not
                    support some of the advanced features of the full-text search,
                    such as boolean mode and query expansion.

                - V2: Uses the new full-text index implementation introduced in
                    SingleStore 8.7. This version offers improved performance and
                    additional features, but is only compatible with SingleStore 8.7
                    or later. To use this version, it must be explicitly passed as
                    the value of full_text_index_version.

            pool_size (int, optional): Determines the number of active connections in
                the pool. Defaults to 5.

            max_overflow (int, optional): Determines the maximum number of connections
                allowed beyond the pool_size. Defaults to 10.

            timeout (float, optional): Specifies the maximum wait time in seconds for
                establishing a connection. Defaults to 30.


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

        Example:
            .. code-block:: python

                from langchain_singlestore import SingleStoreVectorStore
                from langchain_openai import OpenAIEmbeddings

                s2 = SingleStoreVectorStore.from_texts(
                    texts,
                    OpenAIEmbeddings(),
                    host="username:password@localhost:3306/database"
                )
        """

        instance = cls(
            embedding,
            distance_strategy=distance_strategy,
            table_name=table_name,
            content_field=content_field,
            metadata_field=metadata_field,
            vector_field=vector_field,
            id_field=id_field,
            pool_size=pool_size,
            max_overflow=max_overflow,
            timeout=timeout,
            use_vector_index=use_vector_index,
            vector_index_name=vector_index_name,
            vector_index_options=vector_index_options,
            vector_size=vector_size,
            use_full_text_search=use_full_text_search,
            full_text_index_version=full_text_index_version,
            **kwargs,
        )
        instance.add_texts(
            texts,
            metadatas,
            embedding.embed_documents(texts) if embeddings is None else embeddings,
            **kwargs,
        )
        return instance

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by their ids.

        Args:
            ids: The ids of the documents to get.

        Returns:
            A list of Document objects.
        """
        documents: list[Document] = []

        if not ids:
            return documents

        conn = self.connection_pool.connect()
        try:
            cur = conn.cursor()
            try:
                cur.execute(
                    "SELECT {}, {}, {} FROM {} WHERE {} IN ({}) ORDER BY {}".format(
                        self.content_field,
                        self.metadata_field,
                        self.id_field,
                        self.table_name,
                        self.id_field,
                        ",".join(ids),
                        self.id_field,
                    )
                )
                for row in cur.fetchall():
                    doc = Document(page_content=row[0], metadata=row[1], id=row[2])
                    documents.append(doc)
            finally:
                cur.close()
        finally:
            conn.close()

        return documents

    def drop(self) -> None:
        """Drop the table and delete all data from the vectorstore.
        Vector store will be unusable after this operation.
        """
        conn = self.connection_pool.connect()
        try:
            cur = conn.cursor()
            try:
                cur.execute("DROP TABLE IF EXISTS {}".format(self.table_name))
            finally:
                cur.close()
        finally:
            conn.close()


# SingleStoreRetriever is not needed, but we keep it for backwards compatibility
SingleStoreRetriever = VectorStoreRetriever
