import math
import os
import tempfile
from typing import Generator, List, cast

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_tests.integration_tests import VectorStoreIntegrationTests

from langchain_singlestore._utils import DistanceStrategy, FullTextIndexVersion
from langchain_singlestore.vectorstores import SingleStoreVectorStore

from tests.integration_tests.conftest import TEST_DB_NAME, ConnectionParameters

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


class IncrementalEmbeddings(Embeddings):
    """Fake embeddings with incremental vectors. For testing purposes."""

    def __init__(self) -> None:
        self.counter = 0

    def set_counter(self, counter: int) -> None:
        self.counter = counter

    def embed_query(self, text: str) -> List[float]:
        self.counter += 1
        return [
            math.cos(self.counter * math.pi / 10),
            math.sin(self.counter * math.pi / 10),
        ]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]


class TestSingleStoreVectorStore(VectorStoreIntegrationTests):
    @pytest.fixture(
        params=[DistanceStrategy.DOT_PRODUCT, DistanceStrategy.EUCLIDEAN_DISTANCE]
    )
    def vectorstore(self,
                    request: pytest.FixtureRequest,
                    clean_db_url: str) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        # note: store should be EMPTY at this point
        # if you need to delete data, you may do so here
        try:
            store = SingleStoreVectorStore(
                self.get_embeddings(),
                distance_strategy=request.param,
                host=clean_db_url,
                database=TEST_DB_NAME,
            )
            yield store
            store.drop()
        finally:
            # cleanup operations, or deleting data
            pass

    @pytest.fixture(
        params=[
            (DistanceStrategy.DOT_PRODUCT, 10, None),
            (
                DistanceStrategy.EUCLIDEAN_DISTANCE,
                10,
                {"index_type": "IVF_PQ", "nlist": 256},
            ),
            (DistanceStrategy.EUCLIDEAN_DISTANCE, 100, None),
        ]
    )
    def vectorstore_with_vector_index(
        self, request: pytest.FixtureRequest,
        clean_db_url: str,
    ) -> Generator[VectorStore, None, None]:
        """Get an empty vectorstore with vector index for unit tests."""
        # note: store should be EMPTY at this point
        # if you need to delete data, you may do so here
        try:
            store = SingleStoreVectorStore(
                distance_strategy=request.param[0],
                use_vector_index=True,
                vector_size=request.param[1],
                vector_index_options=request.param[2],
                embedding=RandomEmbeddings(request.param[1]),
                host=clean_db_url,
                database=TEST_DB_NAME,
            )
            yield store
            store.drop()
        finally:
            # cleanup operations, or deleting data
            pass

    @pytest.fixture()
    def vectorestore_random(self, clean_db_url: str) -> Generator[SingleStoreVectorStore, None, None]:
        """Get an empty vectorstore with random embeddings for unit tests."""
        # note: store should be EMPTY at this point
        # if you need to delete data, you may do so here
        try:
            store = SingleStoreVectorStore(
                embedding=RandomEmbeddings(10),
                host=clean_db_url,
                database=TEST_DB_NAME,
            )
            yield store
            store.drop()
        finally:
            # cleanup operations, or deleting data
            pass

    @pytest.fixture(
            params=[FullTextIndexVersion.V1, FullTextIndexVersion.V2]
    )
    def vectorestore_incremental(self, request: pytest.FixtureRequest, clean_db_url: str
                                 ) -> Generator[SingleStoreVectorStore, None, None]:
        """Get an empty vectorstore with incremental embeddings for unit tests."""
        # note: store should be EMPTY at this point
        # if you need to delete data, you may do so here
        try:
            store = SingleStoreVectorStore(
                embedding=IncrementalEmbeddings(),
                host=clean_db_url,
                database=TEST_DB_NAME,
                use_full_text_search=True,
                full_text_index_version=request.param,
            )
            yield store
            store.drop()
        finally:
            # cleanup operations, or deleting data
            pass


    @pytest.fixture()
    def snow_rain_docs(self) -> List[Document]:
        return [
            Document(
                page_content="""In the parched desert, a sudden rainstorm brought
                relief, as the droplets danced upon the thirsty earth, rejuvenating
                the landscape with the sweet scent of petrichor.""",
                metadata={"count": "1", "category": "rain", "group": "a"},
            ),
            Document(
                page_content="""Amidst the bustling cityscape, the rain fell
                relentlessly, creating a symphony of pitter-patter on the pavement,
                while umbrellas bloomed like colorful flowers in a sea of gray.""",
                metadata={"count": "2", "category": "rain", "group": "a"},
            ),
            Document(
                page_content="""High in the mountains, the rain transformed into a
                delicate mist, enveloping the peaks in a mystical veil, where each
                droplet seemed to whisper secrets to the ancient rocks below.""",
                metadata={"count": "3", "category": "rain", "group": "b"},
            ),
            Document(
                page_content="""Blanketing the countryside in a soft, pristine layer,
                the snowfall painted a serene tableau, muffling the world in a tranquil
                hush as delicate flakes settled upon the branches of trees like nature's
                own lacework.""",
                metadata={"count": "1", "category": "snow", "group": "b"},
            ),
            Document(
                page_content="""In the urban landscape, snow descended, transforming
                bustling streets into a winter wonderland, where the laughter of
                children echoed amidst the flurry of snowballs and the twinkle of
                holiday lights.""",
                metadata={"count": "2", "category": "snow", "group": "a"},
            ),
            Document(
                page_content="""Atop the rugged peaks, snow fell with an unyielding
                intensity, sculpting the landscape into a pristine alpine paradise,
                where the frozen crystals shimmered under the moonlight, casting a
                spell of enchantment over the wilderness below.""",
                metadata={"count": "3", "category": "snow", "group": "a"},
            ),
        ]

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
        vectorstore_with_vector_index.add_texts(["foo"] * 100)
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
        output = vectorstore.similarity_search(
            "foo", k=1, filter={"category": "budget"}
        )
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
        output = vectorstore.similarity_search(
            "foo", k=1, filter={"category": "budget", "id": 2}
        )
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
        output = vectorstore.similarity_search(
            "foo", k=1, filter={"category": "vacation"}
        )
        assert len(output) == 0

    def test_metadata_filtering_5(self, vectorstore: VectorStore) -> None:
        """Test that metadata filtering works correctly."""
        documents = [
            Document(
                page_content="foo",
                metadata={
                    "id": 1,
                    "category": "budget",
                    "subfield": {"subfield": {"idx": 1, "other_idx": 2}},
                },
            ),
            Document(
                page_content="bar",
                metadata={
                    "id": 2,
                    "category": "budget",
                    "subfield": {"subfield": {"idx": 2, "other_idx": 3}},
                },
            ),
        ]
        vectorstore.add_documents(documents)
        output = vectorstore.similarity_search(
            "foo",
            k=1,
            filter={"category": "budget", "subfield": {"subfield": {"idx": 2}}},
        )
        assert len(output) == 1
        assert output[0].page_content == "bar"
        assert output[0].metadata["category"] == "budget"
        assert output[0].metadata["subfield"]["subfield"]["idx"] == 2
        assert output[0].metadata["subfield"]["subfield"]["other_idx"] == 3
        assert output[0].metadata["id"] == 2

    def test_metadata_filtering_6(self, vectorstore: VectorStore) -> None:
        """Test that metadata filtering works correctly."""
        documents = [
            Document(
                page_content="foo",
                metadata={"id": 1, "category": "budget", "is_good": False},
            ),
            Document(
                page_content="bar",
                metadata={"id": 2, "category": "budget", "is_good": True},
            ),
        ]
        vectorstore.add_documents(documents)
        output = vectorstore.similarity_search("foo", k=1, filter={"is_good": True})
        assert len(output) == 1
        assert output[0].page_content == "bar"
        assert output[0].metadata["category"] == "budget"
        assert output[0].metadata["is_good"]
        assert output[0].metadata["id"] == 2

    def test_metadata_filtering_7(self, vectorstore: VectorStore) -> None:
        """Test that metadata filtering works correctly."""
        documents = [
            Document(
                page_content="foo",
                metadata={"id": 1, "category": "budget", "score": 1.5},
            ),
            Document(
                page_content="bar",
                metadata={"id": 2, "category": "budget", "score": 2.5},
            ),
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

    def test_add_image2(self, clean_db_connection_parameters: ConnectionParameters) -> None:
        docsearch = SingleStoreVectorStore(
            OpenCLIPEmbeddings(
                model=None,
                preprocess=None,
                tokenizer=None,
            ),
            host=clean_db_connection_parameters.Host,
            port=clean_db_connection_parameters.Port,
            user=clean_db_connection_parameters.User,
            password=clean_db_connection_parameters.Password,
            database=clean_db_connection_parameters.Database,
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

    def test_singlestoredb_text_only_search(
        self, vectorestore_incremental: VectorStore, snow_rain_docs: List[Document]
    ) -> None:
        vectorestore_incremental.add_documents(snow_rain_docs)
        output = vectorestore_incremental.similarity_search(
            "rainstorm in parched desert",
            k=3,
            filter={"count": "1"},
            search_strategy=SingleStoreVectorStore.SearchStrategy.TEXT_ONLY,
        )
        assert len(output) == 2
        assert "In the parched desert" in output[0].page_content
        assert "Blanketing the countryside" in output[1].page_content

        output = vectorestore_incremental.similarity_search(
            "snowfall in countryside",
            k=3,
            search_strategy=SingleStoreVectorStore.SearchStrategy.TEXT_ONLY,
        )
        assert len(output) == 3
        assert "Blanketing the countryside" in output[0].page_content

    def test_singlestoredb_filter_by_text_search(
        self, vectorestore_incremental: VectorStore, snow_rain_docs: List[Document]
    ) -> None:
        vectorestore_incremental.add_documents(snow_rain_docs)
        threshold = 0
        if vectorestore_incremental.full_text_index_version == FullTextIndexVersion.V2:
            threshold = 1
        output = vectorestore_incremental.similarity_search(
            "rainstorm in parched desert",
            k=1,
            search_strategy=SingleStoreVectorStore.SearchStrategy.FILTER_BY_TEXT,
            filter_threshold=threshold,
        )
        assert len(output) == 1
        assert "In the parched desert" in output[0].page_content

    def test_singlestoredb_filter_by_vector_search1(
        self, vectorestore_incremental: VectorStore, snow_rain_docs: List[Document]
    ) -> None:
        vectorestore_incremental.add_documents(snow_rain_docs)
        threshold = -0.2
        if vectorestore_incremental.full_text_index_version == FullTextIndexVersion.V2:
            threshold = 0.2
        output = vectorestore_incremental.similarity_search(
            "rainstorm in parched desert, rain",
            k=1,
            filter={"category": "rain"},
            search_strategy=SingleStoreVectorStore.SearchStrategy.FILTER_BY_VECTOR,
            filter_threshold=threshold,
        )
        assert len(output) == 1
        assert "High in the mountains" in output[0].page_content

    def test_singlestoredb_filter_by_vector_search2(
        self, vectorestore_incremental: VectorStore, snow_rain_docs: List[Document]
    ) -> None:
        vectorestore_incremental.add_documents(snow_rain_docs)
        output = vectorestore_incremental.similarity_search(
            "rainstorm in parched desert, rain",
            k=1,
            filter={"group": "a"},
            search_strategy=SingleStoreVectorStore.SearchStrategy.FILTER_BY_VECTOR,
            filter_threshold=-0.2,
        )
        assert len(output) == 1
        assert "Amidst the bustling cityscape" in output[0].page_content

    def test_singlestoredb_weighted_sum_search_unsupported_strategy(
        self,
        clean_db_url: str,
        snow_rain_docs: List[Document],
    ) -> None:
        docsearch = SingleStoreVectorStore.from_documents(
            snow_rain_docs,
            IncrementalEmbeddings(),
            use_full_text_search=True,
            use_vector_index=True,
            vector_size=2,
            host=clean_db_url,
            database=TEST_DB_NAME,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        )
        try:
            docsearch.similarity_search(
                "rainstorm in parched desert, rain",
                k=1,
                search_strategy=SingleStoreVectorStore.SearchStrategy.WEIGHTED_SUM,
            )
        except ValueError as e:
            assert "Search strategy SearchStrategy.WEIGHTED_SUM is not" in str(e)

    def test_singlestoredb_weighted_sum_search(
        self, vectorestore_incremental: VectorStore, snow_rain_docs: List[Document]
    ) -> None:
        vectorestore_incremental.add_documents(snow_rain_docs)
        output = vectorestore_incremental.similarity_search(
            "rainstorm in parched desert, rain",
            k=1,
            search_strategy=SingleStoreVectorStore.SearchStrategy.WEIGHTED_SUM,
            filter={"category": "snow"},
        )
        assert len(output) == 1
        assert "Atop the rugged peaks" in output[0].page_content

    @pytest.fixture()
    def numeric_docs(self) -> List[Document]:
        """Documents with numeric metadata for testing FilterTypedDict operators."""
        return [
            Document(
                page_content="Product A with 100 views",
                metadata={"name": "A", "views": 100, "rating": 4.5, "active": True},
            ),
            Document(
                page_content="Product B with 200 views",
                metadata={"name": "B", "views": 200, "rating": 4.0, "active": True},
            ),
            Document(
                page_content="Product C with 50 views",
                metadata={"name": "C", "views": 50, "rating": 3.5, "active": False},
            ),
            Document(
                page_content="Product D with 300 views",
                metadata={"name": "D", "views": 300, "rating": 5.0, "active": True},
            ),
            Document(
                page_content="Product E with 150 views",
                metadata={"name": "E", "views": 150, "rating": 4.2, "active": False},
            ),
        ]

    def test_filter_typed_dict_simple_filter_backward_compatibility(
        self, vectorestore_random: SingleStoreVectorStore, numeric_docs: List[Document]
    ) -> None:
        """Test that simple nested dict filters still work (backward compatibility)."""
        vectorestore_random.add_documents(numeric_docs)
        output = vectorestore_random.similarity_search(
            "product views",
            k=10,
            filter={"active": True},
        )
        assert len(output) == 3
        names = [doc.metadata.get("name") for doc in output]
        assert "A" in names
        assert "B" in names
        assert "D" in names

    def test_filter_typed_dict_eq_operator(
        self, vectorestore_random: SingleStoreVectorStore, numeric_docs: List[Document]
    ) -> None:
        """Test $eq operator for exact matching."""
        vectorestore_random.add_documents(numeric_docs)
        output = vectorestore_random.similarity_search(
            "product",
            k=10,
            filter={"name": {"$eq": "B"}},
        )
        assert len(output) == 1
        assert output[0].metadata["name"] == "B"

    def test_filter_typed_dict_ne_operator(
        self, vectorestore_random: SingleStoreVectorStore, numeric_docs: List[Document]
    ) -> None:
        """Test $ne operator for not equal matching."""
        vectorestore_random.add_documents(numeric_docs)
        output = vectorestore_random.similarity_search(
            "product",
            k=10,
            filter={"name": {"$ne": "A"}},
        )
        assert len(output) == 4
        names = [doc.metadata.get("name") for doc in output]
        assert "A" not in names
        assert "B" in names
        assert "C" in names
        assert "D" in names
        assert "E" in names

    def test_filter_typed_dict_gt_operator(
        self, vectorestore_random: SingleStoreVectorStore, numeric_docs: List[Document]
    ) -> None:
        """Test $gt operator for greater than numeric comparison."""
        vectorestore_random.add_documents(numeric_docs)
        output = vectorestore_random.similarity_search(
            "product views",
            k=10,
            filter={"views": {"$gt": 150}},
        )
        assert len(output) == 2
        names = [doc.metadata.get("name") for doc in output]
        assert "B" in names  # 200 views
        assert "D" in names  # 300 views

    def test_filter_typed_dict_gte_operator(
        self, vectorestore_random: SingleStoreVectorStore, numeric_docs: List[Document]
    ) -> None:
        """Test $gte operator for greater than or equal numeric comparison."""
        vectorestore_random.add_documents(numeric_docs)
        output = vectorestore_random.similarity_search(
            "product views",
            k=10,
            filter={"views": {"$gte": 150}},
        )
        assert len(output) == 3
        names = [doc.metadata.get("name") for doc in output]
        assert "B" in names  # 200 views
        assert "D" in names  # 300 views
        assert "E" in names  # 150 views

    def test_filter_typed_dict_lt_operator(
        self, vectorestore_random: SingleStoreVectorStore, numeric_docs: List[Document]
    ) -> None:
        """Test $lt operator for less than numeric comparison."""
        vectorestore_random.add_documents(numeric_docs)
        output = vectorestore_random.similarity_search(
            "product views",
            k=10,
            filter={"views": {"$lt": 100}},
        )
        assert len(output) == 1
        names = [doc.metadata.get("name") for doc in output]
        assert "A" not in names  # 100 views, but this is the boundary
        assert "C" in names  # 50 views

    def test_filter_typed_dict_lte_operator(
        self, vectorestore_random: SingleStoreVectorStore, numeric_docs: List[Document]
    ) -> None:
        """Test $lte operator for less than or equal numeric comparison."""
        vectorestore_random.add_documents(numeric_docs)
        output = vectorestore_random.similarity_search(
            "product views",
            k=10,
            filter={"views": {"$lte": 100}},
        )
        assert len(output) == 2
        names = [doc.metadata.get("name") for doc in output]
        assert "A" in names  # 100 views
        assert "C" in names  # 50 views

    def test_filter_typed_dict_in_operator(
        self, vectorestore_random: SingleStoreVectorStore, numeric_docs: List[Document]
    ) -> None:
        """Test $in operator for membership in list."""
        vectorestore_random.add_documents(numeric_docs)
        output = vectorestore_random.similarity_search(
            "product",
            k=10,
            filter={"name": {"$in": ["A", "C"]}},
        )
        assert len(output) == 2
        names = [doc.metadata.get("name") for doc in output]
        assert "A" in names
        assert "C" in names

    def test_filter_typed_dict_nin_operator(
        self, vectorestore_random: SingleStoreVectorStore, numeric_docs: List[Document]
    ) -> None:
        """Test $nin operator for exclusion from list."""
        vectorestore_random.add_documents(numeric_docs)
        output = vectorestore_random.similarity_search(
            "product",
            k=10,
            filter={"name": {"$nin": ["A", "B"]}},
        )
        assert len(output) == 3
        names = [doc.metadata.get("name") for doc in output]
        assert "A" not in names
        assert "B" not in names
        assert "C" in names
        assert "D" in names
        assert "E" in names

    def test_filter_typed_dict_exists_true(
        self, vectorestore_random: SingleStoreVectorStore, numeric_docs: List[Document]
    ) -> None:
        """Test $exists operator to check field existence (true)."""
        vectorestore_random.add_documents(numeric_docs)
        output = vectorestore_random.similarity_search(
            "product",
            k=10,
            filter={"rating": {"$exists": True}},
        )
        # All docs have rating field
        assert len(output) == 5

    def test_filter_typed_dict_exists_false(
        self, vectorestore_random: SingleStoreVectorStore, numeric_docs: List[Document]
    ) -> None:
        """Test $exists operator to check field absence (false)."""
        # Add a document without 'discount' field
        docs = numeric_docs + [
            Document(
                page_content="Product F",
                metadata={"name": "F", "views": 100},
            )
        ]
        vectorestore_random.add_documents(docs)
        output = vectorestore_random.similarity_search(
            "product",
            k=10,
            filter={"discount": {"$exists": False}},
        )
        # All 6 documents don't have 'discount' field
        assert len(output) == 6

    def test_filter_typed_dict_and_operator_multiple_conditions(
        self, vectorestore_random: SingleStoreVectorStore, numeric_docs: List[Document]
    ) -> None:
        """Test $and operator to combine multiple conditions."""
        vectorestore_random.add_documents(numeric_docs)
        output = vectorestore_random.similarity_search(
            "product views",
            k=10,
            filter={
                "$and": [
                    {"views": {"$gt": 100}},
                    {"active": True},
                ]
            },
        )
        # Documents with views > 100 AND active=True: B (200, True), D (300, True)
        assert len(output) == 2
        names = [doc.metadata.get("name") for doc in output]
        assert "B" in names
        assert "D" in names

    def test_filter_typed_dict_or_operator_multiple_conditions(
        self, vectorestore_random: SingleStoreVectorStore, numeric_docs: List[Document]
    ) -> None:
        """Test $or operator to combine multiple conditions with OR logic."""
        vectorestore_random.add_documents(numeric_docs)
        output = vectorestore_random.similarity_search(
            "product",
            k=10,
            filter={
                "$or": [
                    {"name": "A"},
                    {"rating": {"$gte": 4.5}},
                ]
            },
        )
        # Documents where name=A OR rating >= 4.5: A (4.5), D (5.0), B (4.0 - no)
        assert len(output) >= 2
        names = [doc.metadata.get("name") for doc in output]
        assert "A" in names
        assert "D" in names

    def test_filter_typed_dict_complex_nested_filters(
        self, vectorestore_random: SingleStoreVectorStore, numeric_docs: List[Document]
    ) -> None:
        """Test complex nested $and and $or combinations."""
        vectorestore_random.add_documents(numeric_docs)
        output = vectorestore_random.similarity_search(
            "product views",
            k=10,
            filter={
                "$and": [
                    {
                        "$or": [
                            {"name": "A"},
                            {"name": "B"},
                        ]
                    },
                    {"views": {"$gte": 150}},
                ]
            },
        )
        # Documents where (name=A OR name=B) AND views >= 150: B (200)
        assert len(output) == 1
        assert output[0].metadata["name"] == "B"

    def test_filter_typed_dict_with_search_strategy(
        self,
        vectorestore_incremental: SingleStoreVectorStore,
        numeric_docs: List[Document],
    ) -> None:
        """Test FilterTypedDict with TEXT_ONLY search strategy."""
        vectorestore_incremental.add_documents(numeric_docs)
        output = vectorestore_incremental.similarity_search(
            "product views",
            k=10,
            filter={"views": {"$gt": 100}},
            search_strategy=SingleStoreVectorStore.SearchStrategy.TEXT_ONLY,
        )
        # Should work with FilterTypedDict in any search strategy
        assert len(output) > 0
    
    def test_fulltext_index_version_creation(self,
                                            vectorestore_incremental: VectorStore) -> None:
        """Test that full-text index is created when use_full_text_search is True."""
        conn = vectorestore_incremental._get_connection()
        
        with conn.cursor() as cur:
            cur.execute("SHOW CREATE TABLE embeddings")
            result = cur.fetchone()
            assert result is not None
            create_table_sql = result[1]  # The second column contains the SQL
            if vectorestore_incremental.full_text_index_version == FullTextIndexVersion.V1:
                assert "FULLTEXT USING VERSION 1" in create_table_sql
            elif vectorestore_incremental.full_text_index_version == FullTextIndexVersion.V2:
                assert "FULLTEXT USING VERSION 2" in create_table_sql
            else:
                raise ValueError("Unexpected full text index version")
    
    def test_fulltext_search_korean(self, clean_db_connection_parameters: ConnectionParameters) -> None:
        """Test that full-text search works with Korean text."""
        docsearch = SingleStoreVectorStore(
            embedding=IncrementalEmbeddings(),
            host=clean_db_connection_parameters.Host,
            port=clean_db_connection_parameters.Port,
            user=clean_db_connection_parameters.User,
            password=clean_db_connection_parameters.Password,
            database=clean_db_connection_parameters.Database,
            use_full_text_search=True,
            full_text_index_version=FullTextIndexVersion.V2,
        )
        try:
            docs = [
                Document(
                    page_content="""가뭄이 든 사막에 갑작스러운 폭우가 찾아와 안도감을 선사했습니다.
                    메마른 땅 위로 빗방울이 춤을 추듯 떨어지며, 대지는 감미로운 흙 내음을 풍기며 활력을 되찾았습니다.""",
                    metadata={"category": "비"},
                ),
                Document(
                    page_content="""번화한 도시 한복판에서 비가 끊임없이 쏟아졌습니다.
                    보도에 부딪히는 빗소리가 교향곡처럼 울려 퍼졌고, 회색빛 도심 속에는 알록달록한 우산들이 마치 꽃처럼 피어났습니다.""",
                    metadata={"category": "비"},
                ),
                Document(
                    page_content="""높은 산맥 위로 비가 부드러운 안개로 변해 봉우리들을 신비로운 베일로 감싸 안았습니다.
                    빗방울 하나하나가 아래에 놓인 고대의 바위들에게 비밀을 속삭이는 듯했습니다.""",
                    metadata={"category": "비"},
                ),
                Document(
                    page_content="""눈이 시골 풍경을 하얗고 깨끗하게 덮으며 평온한 장면을 연출했습니다.
                    나뭇가지마다 내려앉은 섬세한 눈송이들은 마치 자연이 만든 레이스 같았고, 세상은 고요한 정적 속에 잠겼습니다.""",
                    metadata={"category": "눈"},
                ),
                Document(
                    page_content="""도심 속으로 눈이 내리며 번잡했던 거리들이 겨울의 동화 속 나라로 변했습니다.
                    흩날리는 눈발 사이로 눈싸움을 하는 아이들의 웃음소리가 울려 퍼졌고, 연말의 전등불은 반짝였습니다.""",
                    metadata={"category": "눈"},
                ),
                Document(
                    page_content="""험준한 산봉우리 위로 눈이 거세게 쏟아지며 대지를 순백의 알프스 낙원으로 빚어냈습니다.
                    얼어붙은 결정체들이 달빛 아래 영롱하게 빛나며, 아래에 펼쳐진 황야에 마법 같은 황홀함을 선사했습니다.""",
                    metadata={"category": "눈"},
                ),
            ]
            docsearch.add_documents(docs)
            textResults = docsearch.similarity_search(
                "메마른 사막의 폭우, 비",
                k=1,
                search_strategy=SingleStoreVectorStore.SearchStrategy.TEXT_ONLY,
            )
            assert len(textResults) == 1
            assert "가뭄이 든 사막에 갑작스러운 폭우가 찾아와 안도감을 선사했습니다." in textResults[0].page_content
        finally:
            docsearch.drop()