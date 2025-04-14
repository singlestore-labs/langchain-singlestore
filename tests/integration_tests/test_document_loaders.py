from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings

from langchain_singlestore.document_loaders import SingleStoreLoader
from langchain_singlestore.vectorstores import SingleStoreVectorStore

TEST_SINGLESTOREDB_URL = "root:pass@localhost:3306/db"


def test_singlestore_document_loader() -> None:
    # Define test documents
    documents = [
        Document(page_content="Document 1 content", metadata={"author": "Author 1"}),
        Document(page_content="Document 2 content", metadata={"author": "Author 2"}),
    ]

    # Write documents using SingleStoreVectorStore
    vector_store = SingleStoreVectorStore(
        embedding=FakeEmbeddings(size=10),
        host=TEST_SINGLESTOREDB_URL,
        table_name="test_documents",
    )
    vector_store.add_documents(documents)

    # Read documents using SingleStoreLoader
    loader = SingleStoreLoader(
        host=TEST_SINGLESTOREDB_URL,
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
