"""Sanity tests for the langgraph-singlestore scaffolding.

These verify the import paths and that placeholder classes raise
``NotImplementedError`` from every public method. Real behavior tests will
land alongside the real implementation.
"""

import pytest

from langgraph.checkpoint.singlestore import (
    AsyncSingleStoreSaver,
    SingleStoreSaver,
)
from langgraph.store.singlestore import AsyncSingleStoreStore, SingleStoreStore


def test_saver_placeholder_raises() -> None:
    saver = SingleStoreSaver(host="localhost")
    with pytest.raises(NotImplementedError):
        saver.setup()


def test_async_saver_is_subclass() -> None:
    assert issubclass(AsyncSingleStoreSaver, SingleStoreSaver)


def test_store_placeholder_raises() -> None:
    store = SingleStoreStore(host="localhost")
    with pytest.raises(NotImplementedError):
        store.batch([])


def test_async_store_is_subclass() -> None:
    assert issubclass(AsyncSingleStoreStore, SingleStoreStore)
