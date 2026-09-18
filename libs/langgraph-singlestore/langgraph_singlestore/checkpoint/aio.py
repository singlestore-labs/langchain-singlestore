"""Asynchronous SingleStore-backed ``BaseCheckpointSaver``.

The ``singlestoredb`` driver is synchronous, so async operations dispatch to
the default executor. ``SingleStoreSaver`` already exposes every ``a*``
checkpoint method this way; ``AsyncSingleStoreSaver`` adds ``asetup`` /
``aclose`` and exists for API parity with ``AsyncPostgresSaver``.
"""

from __future__ import annotations

import asyncio

from langgraph_singlestore.checkpoint.base import SingleStoreSaver


class AsyncSingleStoreSaver(SingleStoreSaver):
    """Async variant of :class:`SingleStoreSaver`.

    Every ``a*`` method inherited from :class:`SingleStoreSaver` already
    dispatches to the default executor; this subclass adds async lifecycle
    helpers so callers can drive setup and teardown without blocking the
    event loop.
    """

    async def asetup(self) -> None:
        await asyncio.get_running_loop().run_in_executor(None, self.setup)

    async def aclose(self) -> None:
        await asyncio.get_running_loop().run_in_executor(None, self.close)
