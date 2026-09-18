"""Official LangGraph checkpointer conformance suite for :class:`SingleStoreSaver`.

Runs the ``langgraph-checkpoint-conformance`` package against a real
SingleStore container (see ``conftest.py``). See
https://docs.langchain.com/langsmith/custom-checkpointer for the full list
of base and extended capabilities the suite validates.

A fresh saver + fresh schema is materialised for every capability the
suite exercises so cross-test state cannot leak.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import closing

import pytest
from singlestoredb.connection import connect

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.conformance import checkpointer_test, validate
from langgraph.checkpoint.singlestore import SingleStoreSaver

from .conftest import ConnectionParameters

_CHECKPOINT_TABLES = (
    "checkpoint_writes",
    "checkpoint_blobs",
    "checkpoints",
    "checkpoint_migrations",
)


def _drop_checkpoint_tables(params: ConnectionParameters) -> None:
    """Drop every table the saver's migrations own.

    Called at the start of each capability's factory invocation so the
    conformance suite always sees a pristine schema, even across the
    multiple checkpointer instantiations that ``validate`` performs.
    """
    conn = connect(**params.as_kwargs())
    try:
        with closing(conn.cursor()) as cur:
            for table in _CHECKPOINT_TABLES:
                cur.execute(f"DROP TABLE IF EXISTS {table}")
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_conformance(connection_parameters: ConnectionParameters) -> None:
    """Runs the official ``BaseCheckpointSaver`` conformance suite.

    Asserts every base capability (``put``, ``put_writes``, ``get_tuple``,
    ``list``, ``delete_thread``) passes; extended capabilities
    (``delete_for_runs``, ``copy_thread``, ``prune``) are still executed
    and reported but not required to pass.
    """
    kwargs = connection_parameters.as_kwargs()

    async def factory() -> AsyncGenerator[BaseCheckpointSaver, None]:
        _drop_checkpoint_tables(connection_parameters)
        saver = SingleStoreSaver(**kwargs)
        saver.setup()
        try:
            yield saver
        finally:
            saver.close()

    registered = checkpointer_test(name="SingleStoreSaver")(factory)
    report = await validate(registered)
    report.print_report()
    assert report.passed_all_base(), (
        "SingleStoreSaver failed one or more base capabilities; see the "
        "printed report above for details."
    )
