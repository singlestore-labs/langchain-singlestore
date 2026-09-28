"""Integration tests for SingleStoreEmbeddings.

These tests spin up a SingleStore instance via Docker (see ``conftest.py``)
and register a mock embedding UDF with the signature
``mock_embed(text VARCHAR(100)) RETURNS BLOB`` so ``SingleStoreEmbeddings``
can be exercised end-to-end without needing real AI capabilities enabled
on the cluster.
"""

from typing import Generator, List

import pytest
from singlestoredb import connect

from langchain_singlestore.embeddings import SingleStoreEmbeddings
from tests.integration_tests.conftest import TEST_DB_NAME, ConnectionParameters

MOCK_EMBED_FUNCTION = "mock_embed"

# The mock UDF packs a deterministic 3-dim float32 vector where the first
# component is the input length. That lets tests assert both length and
# per-input variation without depending on external AI services.
_CREATE_MOCK_EMBED_SQL = f"""
CREATE OR REPLACE FUNCTION {MOCK_EMBED_FUNCTION}(input_text VARCHAR(100))
RETURNS BLOB AS
DECLARE
    len_val INT = CHAR_LENGTH(input_text);
BEGIN
    RETURN JSON_ARRAY_PACK(CONCAT('[', len_val, '.0, 0.5, 0.25]'));
END
"""

EXPECTED_DIM = 3


@pytest.fixture(scope="function")
def mock_embed_function(
    clean_db_connection_parameters: ConnectionParameters,
) -> Generator[ConnectionParameters, None, None]:
    """Create the ``mock_embed`` UDF on the test database.

    Yields the connection parameters so tests can build a
    ``SingleStoreEmbeddings`` bound to the same database.
    """
    params = clean_db_connection_parameters
    conn = connect(
        host=params.Host,
        port=params.Port,
        user=params.User,
        password=params.Password,
        database=params.Database,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(_CREATE_MOCK_EMBED_SQL)
        yield params
        with conn.cursor() as cur:
            cur.execute(f"DROP FUNCTION IF EXISTS {MOCK_EMBED_FUNCTION}")
    finally:
        conn.close()


def _build_embeddings(params: ConnectionParameters) -> SingleStoreEmbeddings:
    return SingleStoreEmbeddings(
        function_name=MOCK_EMBED_FUNCTION,
        host=params.Host,
        port=params.Port,
        user=params.User,
        password=params.Password,
        database=params.Database,
    )


def test_mock_embed_function_is_callable_via_sql(
    mock_embed_function: ConnectionParameters,
) -> None:
    """Smoke test: the fixture registered a working UDF."""
    params = mock_embed_function
    conn = connect(
        host=params.Host,
        port=params.Port,
        user=params.User,
        password=params.Password,
        database=params.Database,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT JSON_ARRAY_UNPACK({MOCK_EMBED_FUNCTION}(%s))",
                ("hello",),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    assert row is not None
    unpacked = row[0]  # type: ignore
    assert unpacked is not None


def test_embed_query_returns_vector(
    mock_embed_function: ConnectionParameters,
) -> None:
    """``embed_query`` returns a float vector produced by the mock UDF."""
    embeddings = _build_embeddings(mock_embed_function)
    vector = embeddings.embed_query("hello")

    assert isinstance(vector, list)
    assert len(vector) == EXPECTED_DIM
    assert all(isinstance(v, float) for v in vector)
    # First component encodes the input length (see ``_CREATE_MOCK_EMBED_SQL``).
    assert vector[0] == pytest.approx(len("hello"))
    assert vector[1] == pytest.approx(0.5)
    assert vector[2] == pytest.approx(0.25)


def test_embed_documents_returns_one_vector_per_input(
    mock_embed_function: ConnectionParameters,
) -> None:
    """``embed_documents`` returns one vector per input, in order."""
    embeddings = _build_embeddings(mock_embed_function)
    texts: List[str] = ["a", "abcd", "hello world"]
    vectors = embeddings.embed_documents(texts)

    assert isinstance(vectors, list)
    assert len(vectors) == len(texts)
    for text, vector in zip(texts, vectors):
        assert len(vector) == EXPECTED_DIM
        assert vector[0] == pytest.approx(len(text))
        assert vector[1] == pytest.approx(0.5)
        assert vector[2] == pytest.approx(0.25)


def test_embed_documents_with_empty_input(
    mock_embed_function: ConnectionParameters,
) -> None:
    """Embedding an empty list returns an empty list."""
    embeddings = _build_embeddings(mock_embed_function)
    assert embeddings.embed_documents([]) == []


def test_embed_query_with_qualified_function_name(
    mock_embed_function: ConnectionParameters,
) -> None:
    """A database-qualified function name resolves to the same UDF."""
    params = mock_embed_function
    embeddings = SingleStoreEmbeddings(
        function_name=f"{TEST_DB_NAME}.{MOCK_EMBED_FUNCTION}",
        host=params.Host,
        port=params.Port,
        user=params.User,
        password=params.Password,
        database=params.Database,
    )
    vector = embeddings.embed_query("hi")
    assert len(vector) == EXPECTED_DIM
    assert vector[0] == pytest.approx(len("hi"))
