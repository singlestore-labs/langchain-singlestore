"""Integration tests for :class:`SingleStoreStore`.

Runs a real SingleStore container (see ``conftest.py``) and exercises the
draft store end-to-end via ``PutOp`` + ``GetOp``.
"""

from typing import cast

from langgraph.store.base import GetOp, Item, PutOp
from langgraph.store.singlestore import SingleStoreStore

from .conftest import ConnectionParameters


def _as_item(result: object) -> Item | None:
    """Narrow a ``batch()`` result to an ``Item`` for typed access."""
    assert result is None or isinstance(result, Item)
    return cast("Item | None", result)


class TestSingleStoreStorePutOp:
    def test_put_then_get_returns_stored_value(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            namespace = ("users", "alice")
            key = "prefs"
            value = {"theme": "dark", "lang": "en"}

            put_results = store.batch([PutOp(namespace, key, value)])
            assert put_results == [None]

            get_results = store.batch([GetOp(namespace, key)])
            assert len(get_results) == 1
            item = _as_item(get_results[0])
            assert item is not None
            assert item.namespace == namespace
            assert item.key == key
            assert item.value == value
        finally:
            store.close()
