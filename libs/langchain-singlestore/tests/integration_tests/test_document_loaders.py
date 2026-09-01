from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings

from langchain_singlestore.document_loaders import SingleStoreLoader
from langchain_singlestore.vectorstores import SingleStoreVectorStore
from tests.integration_tests.conftest import TEST_DB_NAME


def test_singlestore_document_loader(clean_db_url: str) -> None:
    # Define test documents
    documents = [
        Document(page_content="Document 1 content", metadata={"author": "Author 1"}),
        Document(page_content="Document 2 content", metadata={"author": "Author 2"}),
    ]

    # Write documents using SingleStoreVectorStore
    vector_store = SingleStoreVectorStore(
        embedding=FakeEmbeddings(size=10),
        host=clean_db_url,
        database=TEST_DB_NAME,
        table_name="test_documents",
    )
    vector_store.add_documents(documents)

    # Read documents using SingleStoreLoader
    loader = SingleStoreLoader(
        host=clean_db_url,
        database=TEST_DB_NAME,
        table_name="test_documents",
        content_field="content",
        metadata_field="metadata",
    )
    loaded_documents = list(loader.lazy_load())
    vector_store.drop()

    # Ensure the list of documents is the same
    assert len(loaded_documents) == len(documents)
    for original, loaded in zip(documents, loaded_documents):
        assert original.page_content == loaded.page_content
        assert original.metadata == loaded.metadata


def test_singlestore_document_loader_with_shared_connection(clean_db_url: str) -> None:
    """SingleStoreLoader works when given a caller-owned connection."""
    import singlestoredb

    documents = [
        Document(page_content="c1", metadata={"a": "1"}),
        Document(page_content="c2", metadata={"a": "2"}),
    ]
    vector_store = SingleStoreVectorStore(
        embedding=FakeEmbeddings(size=10),
        host=clean_db_url,
        database=TEST_DB_NAME,
        table_name="test_documents_conn",
    )
    vector_store.add_documents(documents)

    conn = singlestoredb.connect(host=clean_db_url, database=TEST_DB_NAME)
    try:
        loader = SingleStoreLoader(
            connection=conn,
            table_name="test_documents_conn",
        )
        loaded = list(loader.lazy_load())
        assert len(loaded) == len(documents)
    finally:
        vector_store.drop()
        conn.close()


def test_singlestore_document_loader_with_shared_pool(clean_db_url: str) -> None:
    """SingleStoreLoader works when given a caller-owned connection pool."""
    from singlestore_langchain_core import create_connection_pool

    documents = [Document(page_content="p1", metadata={"a": "1"})]
    vector_store = SingleStoreVectorStore(
        embedding=FakeEmbeddings(size=10),
        host=clean_db_url,
        database=TEST_DB_NAME,
        table_name="test_documents_pool",
    )
    vector_store.add_documents(documents)

    pool = create_connection_pool(
        pool_size=2,
        max_overflow=0,
        timeout=10,
        connection_kwargs={"host": clean_db_url, "database": TEST_DB_NAME},
    )
    try:
        loader = SingleStoreLoader(
            connection_pool=pool,
            table_name="test_documents_pool",
        )
        assert loader.connection_pool is pool
        loaded = list(loader.lazy_load())
        assert len(loaded) == len(documents)
    finally:
        vector_store.drop()
        pool.dispose()
