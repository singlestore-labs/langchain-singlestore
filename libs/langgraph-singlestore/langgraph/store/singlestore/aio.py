"""Asynchronous SingleStore-backed :class:`~langgraph.store.base.BaseStore`.

Draft implementation. The underlying ``singlestoredb`` driver is
synchronous, so async operations dispatch to the default executor. The class
is API-compatible with :class:`SingleStoreStore` and shares its connection
handling (see the base class docstring for the accepted parameters).
"""

from __future__ import annotations

import asyncio
from typing import Iterable

from langgraph.store.base import Op, Result
from langgraph.store.singlestore.base import SingleStoreStore


class AsyncSingleStoreStore(SingleStoreStore):
    """Async variant of :class:`SingleStoreStore`.

    Delegates all I/O to :class:`SingleStoreStore` via the default executor.
    """

    async def asetup(self) -> None:
        await asyncio.get_running_loop().run_in_executor(None, self.setup)

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        # Materialise the iterable before crossing the executor boundary.
        ops_list = list(ops)
        return await asyncio.get_running_loop().run_in_executor(
            None, self.batch, ops_list
        )
