"""Test SingleStoreDB semantic cache. Requires a SingleStore DB database."""
from typing import Dict
from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.outputs import Generation

from langchain_singlestore.cache import SingleStoreSemanticCache
from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.language_models import BaseLLM

TEST_SINGLESTOREDB_URL = "root:pass@localhost:3306/db"

def create_llm_string(llm: BaseLLM) -> str:
    _dict: Dict = llm.dict()
    _dict["stop"] = None
    return str(sorted([(k, v) for k, v in _dict.items()]))

def test_tinglestoredb_semantic_cache() -> None:
    prompt = "How are you?"
    response = "Test response"
    cached_response = "Cached test response"
    set_llm_cache(
        SingleStoreSemanticCache(
            DeterministicFakeEmbedding(size=10),
            host=TEST_SINGLESTOREDB_URL,
        )
    )
    llm = FakeListLLM(responses=[response])
    if llm_cache := get_llm_cache():
        llm_cache.update(
            prompt=prompt,
            llm_string=create_llm_string(llm),
            return_val=[Generation(text=cached_response)],
        )
        assert llm.invoke(prompt) == cached_response
    else:
        raise ValueError(
            "The cache not set. This should never happen, as the pytest fixture "
            "`set_cache_and_teardown` always sets the cache."
        )