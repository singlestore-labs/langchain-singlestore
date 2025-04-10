"""Test ChatSingleStore chat model."""

from typing import Type

from langchain_singlestore.chat_models import ChatSingleStore
from langchain_tests.integration_tests import ChatModelIntegrationTests


"""class TestChatParrotLinkIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatSingleStore]:
        return ChatSingleStore

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "bird-brain-001",
            "temperature": 0,
            "parrot_buffer_length": 50,
        }
"""