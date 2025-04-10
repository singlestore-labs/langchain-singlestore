"""Test SingleStore embeddings."""

from typing import Type

from langchain_singlestore.embeddings import SingleStoreEmbeddings
from langchain_tests.integration_tests import EmbeddingsIntegrationTests


class TestParrotLinkEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[SingleStoreEmbeddings]:
        return SingleStoreEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
