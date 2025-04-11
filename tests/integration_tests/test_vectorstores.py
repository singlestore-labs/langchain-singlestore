from typing import Generator, List, cast

import pytest
import numpy as np
import tempfile
import os
from langchain_singlestore.vectorstores import SingleStoreVectorStore
from langchain_singlestore._utils import DistanceStrategy

from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from langchain_experimental.open_clip import OpenCLIPEmbeddings   

from langchain_tests.integration_tests import VectorStoreIntegrationTests

TEST_SINGLESTOREDB_URL = "root:pass@localhost:3306/db"

class RandomEmbeddings(Embeddings):
    """Fake embeddings with random vectors. For testing purposes."""
    def __init__(self, size: int) -> None:
        self.size = size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [cast(list[float], np.random.rand(self.size).tolist()) for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return cast(list[float], np.random.rand(self.size).tolist())

    def embed_image(self, uris: List[str]) -> List[List[float]]:
        return [cast(list[float], np.random.rand(self.size).tolist()) for _ in uris]


class TestSingleStoreVectorStore(VectorStoreIntegrationTests):
    @pytest.fixture(params=[DistanceStrategy.DOT_PRODUCT, DistanceStrategy.EUCLIDEAN_DISTANCE])
    def vectorstore(self, request) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        # note: store should be EMPTY at this point
        # if you need to delete data, you may do so here
        try:
            store = SingleStoreVectorStore(self.get_embeddings(), distance_strategy=request.param, host=TEST_SINGLESTOREDB_URL)
            yield store
            store.drop()
        finally:
            # cleanup operations, or deleting data
            pass

    @pytest.fixture(params=[(DistanceStrategy.DOT_PRODUCT, 10, None),
                            (DistanceStrategy.EUCLIDEAN_DISTANCE, 10, {"index_type": "IVF_PQ", "nlist": 256}),
                            (DistanceStrategy.EUCLIDEAN_DISTANCE, 100, None),])
    def vectorstore_with_vector_index(self, request) -> Generator[VectorStore, None, None]:
        """Get an empty vectorstore with vector index for unit tests."""
        # note: store should be EMPTY at this point
        # if you need to delete data, you may do so here
        try:
            store = SingleStoreVectorStore(distance_strategy=request.param[0],
                                           use_vector_index=True,
                                           vector_size=request.param[1],
                                           vector_index_options=request.param[2],
                                           embedding=RandomEmbeddings(request.param[1]),
                                           host=TEST_SINGLESTOREDB_URL)
            yield store
            store.drop()
        finally:
            # cleanup operations, or deleting data
            pass
    
    @pytest.fixture()
    def vectorestore_random(self) -> Generator[SingleStoreVectorStore, None, None]:
        """Get an empty vectorstore with random embeddings for unit tests."""
        # note: store should be EMPTY at this point
        # if you need to delete data, you may do so here
        try:
            store = SingleStoreVectorStore(embedding=RandomEmbeddings(10), host=TEST_SINGLESTOREDB_URL)
            yield store
            store.drop()
        finally:
            # cleanup operations, or deleting data
            pass

    @pytest.mark.xfail(reason="id should be integer")
    def test_add_documents_with_existing_ids(self, vectorstore: VectorStore) -> None:
        """Test that add_documents with existing IDs is idempotent.

        .. dropdown:: Troubleshooting

            If this test fails, check that ``get_by_ids`` is implemented and returns
            documents in the same order as the IDs passed in.

            This test also verifies that:

            1. IDs specified in the ``Document.id`` field are assigned when adding documents.
            2. If some documents include IDs and others don't string IDs are generated for the latter.

            .. note::
                ``get_by_ids`` was added to the ``VectorStore`` interface in
                ``langchain-core`` version 0.2.11. If difficult to implement, this
                test can be skipped using a pytest ``xfail`` on the test class:

                .. code-block:: python

                    @pytest.mark.xfail(reason=("get_by_ids not implemented."))
                    def test_add_documents_with_existing_ids(self, vectorstore: VectorStore) -> None:
                        super().test_add_documents_with_existing_ids(vectorstore)
        """  # noqa: E501
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        documents = [
            Document(id="1", page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        ids = vectorstore.add_documents(documents)
        assert "1" in ids
        assert vectorstore.get_by_ids(ids) == [
            Document(page_content="foo", metadata={"id": 1}, id="1"),
            Document(page_content="bar", metadata={"id": 2}, id=ids[1]),
        ]

    @pytest.mark.xfail(reason="id should be integer")
    async def test_add_documents_with_existing_ids_async(
        self, vectorstore: VectorStore
    ) -> None:
        """Test that add_documents with existing IDs is idempotent.

        .. dropdown:: Troubleshooting

            If this test fails, check that ``get_by_ids`` is implemented and returns
            documents in the same order as the IDs passed in.

            This test also verifies that:

            1. IDs specified in the ``Document.id`` field are assigned when adding documents.
            2. If some documents include IDs and others don't string IDs are generated for the latter.

            .. note::
                ``get_by_ids`` was added to the ``VectorStore`` interface in
                ``langchain-core`` version 0.2.11. If difficult to implement, this
                test can be skipped using a pytest ``xfail`` on the test class:

                .. code-block:: python

                    @pytest.mark.xfail(reason=("get_by_ids not implemented."))
                    async def test_add_documents_with_existing_ids(self, vectorstore: VectorStore) -> None:
                        await super().test_add_documents_with_existing_ids(vectorstore)
        """  # noqa: E501
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        documents = [
            Document(id="foo", page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        ids = await vectorstore.aadd_documents(documents)
        assert "foo" in ids
        assert await vectorstore.aget_by_ids(ids) == [
            Document(page_content="foo", metadata={"id": 1}, id="foo"),
            Document(page_content="bar", metadata={"id": 2}, id=ids[1]),
        ]

    def test_vector_index(self, vectorstore_with_vector_index: VectorStore) -> None:
        """Test that vector index is created and used correctly."""
        vectorstore_with_vector_index.add_texts(["foo"]*100)
        output = vectorstore_with_vector_index.similarity_search("foo", k=1)
        assert output[0].page_content == "foo"  

    def test_metadata_filtering_1(self, vectorstore: VectorStore) -> None:
        """Test that metadata filtering works correctly."""
        documents = [
            Document(page_content="foo", metadata={"id": 1}),
            Document(page_content="bar", metadata={"id": 2}),
        ]
        vectorstore.add_documents(documents)
        output = vectorstore.similarity_search("foo", k=1, filter={"id": 2})
        assert output[0].page_content == "bar"
        assert output[0].metadata["id"] == 2
        assert len(output) == 1

    def test_metadata_filtering_2(self, vectorstore: VectorStore) -> None:
        """Test that metadata filtering works correctly."""
        documents = [
            Document(page_content="foo", metadata={"id": 1, "category": "budget"}),
            Document(page_content="bar", metadata={"id": 2, "category": "budget"}),
        ]
        vectorstore.add_documents(documents)
        output = vectorstore.similarity_search("foo", k=1, filter={"category": "budget"})
        assert output[0].page_content == "foo"
        assert output[0].metadata["category"] == "budget"
        assert len(output) == 1

    def test_metadata_filtering_3(self, vectorstore: VectorStore) -> None:
        """Test that metadata filtering works correctly."""
        documents = [
            Document(page_content="foo", metadata={"id": 1, "category": "budget"}),
            Document(page_content="bar", metadata={"id": 2, "category": "budget"}),
        ]
        vectorstore.add_documents(documents)
        output = vectorstore.similarity_search("foo", k=1, filter={"category": "budget", "id": 2})
        assert output[0].page_content == "bar"
        assert output[0].metadata["category"] == "budget"
        assert output[0].metadata["id"] == 2
        assert len(output) == 1
    
    def test_metadata_filtering_4(self, vectorstore: VectorStore) -> None:
        """Test that metadata filtering works correctly."""
        documents = [
            Document(page_content="foo", metadata={"id": 1, "category": "budget"}),
            Document(page_content="bar", metadata={"id": 2, "category": "budget"}),
        ]
        vectorstore.add_documents(documents)
        output = vectorstore.similarity_search("foo", k=1, filter={"category": "vacation"})
        assert len(output) == 0
    
    def test_metadata_filtering_5(self, vectorstore: VectorStore) -> None:
        """Test that metadata filtering works correctly."""
        documents = [
            Document(page_content="foo", metadata={"id": 1, "category": "budget", "subfield": {"subfield": {"idx": 1, "other_idx": 2}}}),
            Document(page_content="bar", metadata={"id": 2, "category": "budget", "subfield": {"subfield": {"idx": 2, "other_idx": 3}}}),
        ]
        vectorstore.add_documents(documents)
        output = vectorstore.similarity_search("foo", k=1, filter={"category": "budget", "subfield": {"subfield": {"idx": 2}}})
        assert len(output) == 1
        assert output[0].page_content == "bar"
        assert output[0].metadata["category"] == "budget"
        assert output[0].metadata["subfield"]["subfield"]["idx"] == 2
        assert output[0].metadata["subfield"]["subfield"]["other_idx"] == 3
        assert output[0].metadata["id"] == 2
    
    def test_metadata_filtering_6(self, vectorstore: VectorStore) -> None:
        """Test that metadata filtering works correctly."""
        documents = [
            Document(page_content="foo", metadata={"id": 1, "category": "budget", "is_good": False}),
            Document(page_content="bar", metadata={"id": 2, "category": "budget", "is_good": True}),
        ]
        vectorstore.add_documents(documents)
        output = vectorstore.similarity_search("foo", k=1, filter={"is_good": True})
        assert len(output) == 1
        assert output[0].page_content == "bar"
        assert output[0].metadata["category"] == "budget"
        assert output[0].metadata["is_good"] == True
        assert output[0].metadata["id"] == 2

    def test_metadata_filtering_7(self, vectorstore: VectorStore) -> None:
        """Test that metadata filtering works correctly."""
        documents = [
            Document(page_content="foo", metadata={"id": 1, "category": "budget", "score": 1.5}),
            Document(page_content="bar", metadata={"id": 2, "category": "budget", "score": 2.5}),
        ]
        vectorstore.add_documents(documents)
        output = vectorstore.similarity_search("foo", k=1, filter={"score": 2.5})
        assert len(output) == 1
        assert output[0].page_content == "bar"
        assert output[0].metadata["category"] == "budget"
        assert output[0].metadata["score"] == 2.5
        assert output[0].metadata["id"] == 2

    def test_add_image1(self, vectorestore_random: SingleStoreVectorStore) -> None:
        """Test adding images"""
        temp_files = []
        for _ in range(3):
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(b"foo")
            temp_file.close()
            temp_files.append(temp_file.name)

        vectorestore_random.add_images(temp_files)
        output = vectorestore_random.similarity_search("foo", k=1)
        assert len(output) == 1
        assert output[0].page_content in temp_files

    def test_add_image2(self) -> None:
        docsearch = SingleStoreVectorStore(
            OpenCLIPEmbeddings(), 
            host=TEST_SINGLESTOREDB_URL,
        )
        IMAGES_DIR = "tests/integration_tests/images"
        image_uris = sorted(
            [
                os.path.join(IMAGES_DIR, image_name)
                for image_name in os.listdir(IMAGES_DIR)
                if image_name.endswith(".jpeg")
            ]
        )
        docsearch.add_images(image_uris)
        output = docsearch.similarity_search("horse", k=1)
        docsearch.drop()
        assert len(output) == 1
        assert output[0].page_content in image_uris
        assert output[0].page_content == IMAGES_DIR + "/right.jpeg"