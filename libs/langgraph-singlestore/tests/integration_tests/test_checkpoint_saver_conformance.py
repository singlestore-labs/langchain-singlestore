"""LangGraph checkpoint saver conformance tests for :class:`SingleStoreSaver`.

Ported from the canonical Postgres saver conformance suite in
``langchain-ai/langgraph`` (``libs/checkpoint-postgres/tests/test_sync.py``)
so any behavioural divergence from the reference implementation surfaces
here.

The postgres-specific tests (pipeline / pool modes, ``psycopg`` null-byte
stripping) are omitted; everything else that the base checkpointer contract
promises is exercised end-to-end against a real SingleStore container.
"""

from __future__ import annotations

import re
from typing import Any, cast

import pytest
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    EXCLUDED_METADATA_KEYS,
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.checkpoint.serde.types import TASKS
from langgraph.checkpoint.singlestore import SingleStoreSaver

from .conftest import ConnectionParameters

# ---------------------------------------------------------------- fixtures


@pytest.fixture
def saver(
    connection_parameters: ConnectionParameters,
) -> Any:
    """A fully-migrated saver against a fresh test database."""
    s = SingleStoreSaver(**connection_parameters.as_kwargs())
    s.setup()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def test_data() -> dict[str, list[Any]]:
    """Three configs / checkpoints / metadata rows for search + fetch tests.

    Matches the shape used by the upstream postgres conformance fixture so
    the imported test bodies drop in unchanged.
    """
    config_1: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-1",
            "checkpoint_id": "1",
            "checkpoint_ns": "",
        }
    }
    config_2: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2",
            "checkpoint_id": "2",
            "checkpoint_ns": "",
        }
    }
    config_3: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2",
            "checkpoint_id": "2-inner",
            "checkpoint_ns": "inner",
        }
    }

    chkpnt_1: Checkpoint = empty_checkpoint()
    chkpnt_2: Checkpoint = create_checkpoint(chkpnt_1, {}, 1)
    chkpnt_3: Checkpoint = empty_checkpoint()

    metadata_1 = cast(
        CheckpointMetadata,
        {"source": "input", "step": 2, "score": 1},
    )
    metadata_2 = cast(
        CheckpointMetadata,
        {"source": "loop", "step": 1, "score": None},
    )
    metadata_3 = cast(CheckpointMetadata, {})

    return {
        "configs": [config_1, config_2, config_3],
        "checkpoints": [chkpnt_1, chkpnt_2, chkpnt_3],
        "metadata": [metadata_1, metadata_2, metadata_3],
    }


def _exclude_keys(configurable: dict[str, Any]) -> dict[str, Any]:
    """Strip the langgraph-reserved keys from a ``configurable`` dict.

    ``get_serializable_checkpoint_metadata`` merges every other key into the
    stored metadata, so the tests compare against ``configurable`` minus
    these excluded ones.
    """
    return {k: v for k, v in configurable.items() if k not in EXCLUDED_METADATA_KEYS}


# ---------------------------------------------------------------- migrations


def test_nonnull_migrations() -> None:
    """Every entry in ``SingleStoreSaver.MIGRATIONS`` must be a non-empty
    SQL statement (no accidentally blank string in the list)."""
    _leading_comment = re.compile(r"^/\*.*?\*/")
    for migration in SingleStoreSaver.MIGRATIONS:
        first = _leading_comment.sub("", migration).split()[0]
        assert first.strip()


# ---------------------------------------------------------------- metadata


def test_combined_metadata(
    saver: SingleStoreSaver, test_data: dict[str, list[Any]]
) -> None:
    """A ``config["metadata"]`` sibling must be merged into the stored
    ``CheckpointMetadata`` — this is the contract ``put`` inherits from
    ``get_serializable_checkpoint_metadata``.
    """
    config = cast(
        RunnableConfig,
        {
            "configurable": {"thread_id": "thread-2", "checkpoint_ns": ""},
            "metadata": {"run_id": "my_run_id"},
        },
    )
    chkpnt: Checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
    metadata = cast(
        CheckpointMetadata,
        {"source": "loop", "step": 1, "score": None},
    )
    saver.put(config, chkpnt, metadata, {})

    ckpt_tuple = saver.get_tuple(config)
    assert ckpt_tuple is not None
    assert ckpt_tuple.metadata == {**metadata, "run_id": "my_run_id"}


# ---------------------------------------------------------------- search


def test_search(saver: SingleStoreSaver, test_data: dict[str, list[Any]]) -> None:
    """The base contract for ``list`` filtering:

    * single-key metadata filter
    * multi-key metadata filter
    * empty filter returns everything
    * unmatched filter returns nothing
    * config-only listing returns rows across every namespace of the thread
    """
    configs = test_data["configs"]
    checkpoints = test_data["checkpoints"]
    metadata = test_data["metadata"]

    saver.put(configs[0], checkpoints[0], metadata[0], {})
    saver.put(configs[1], checkpoints[1], metadata[1], {})
    saver.put(configs[2], checkpoints[2], metadata[2], {})

    query_1 = {"source": "input"}
    query_2 = {"step": 1}
    query_3: dict[str, Any] = {}
    query_4 = {"source": "update", "step": 1}

    results_1 = list(saver.list(None, filter=query_1))
    assert len(results_1) == 1
    assert results_1[0].metadata == {
        **_exclude_keys(configs[0]["configurable"]),
        **metadata[0],
    }

    results_2 = list(saver.list(None, filter=query_2))
    assert len(results_2) == 1
    assert results_2[0].metadata == {
        **_exclude_keys(configs[1]["configurable"]),
        **metadata[1],
    }

    results_3 = list(saver.list(None, filter=query_3))
    assert len(results_3) == 3

    results_4 = list(saver.list(None, filter=query_4))
    assert len(results_4) == 0

    # Config-only listing: ``thread_id`` matches both namespaces of thread-2.
    results_5 = list(
        saver.list(cast(RunnableConfig, {"configurable": {"thread_id": "thread-2"}}))
    )
    assert len(results_5) == 2
    assert {
        results_5[0].config["configurable"]["checkpoint_ns"],
        results_5[1].config["configurable"]["checkpoint_ns"],
    } == {"", "inner"}


# ---------------------------------------------------- pending sends migration


def test_pending_sends_migration(saver: SingleStoreSaver) -> None:
    """Writes to the ``TASKS`` channel on checkpoint N must appear as
    ``channel_values[TASKS]`` on checkpoint N+1 when the checkpoint's
    schema version is < 4.

    This is the ``_migrate_pending_sends`` compatibility path exercised by
    both ``get_tuple`` and ``list``.
    """
    config = cast(
        RunnableConfig,
        {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}},
    )

    # First checkpoint + two pending sends attached to it.
    checkpoint_0 = empty_checkpoint()
    config = saver.put(config, checkpoint_0, cast(CheckpointMetadata, {}), {})
    saver.put_writes(
        config,
        [(TASKS, "send-1"), (TASKS, "send-2")],
        task_id="task-1",
    )
    saver.put_writes(
        config,
        [(TASKS, "send-3")],
        task_id="task-2",
    )

    # get_tuple on checkpoint_0 alone must NOT surface the pending sends —
    # they belong to the *next* checkpoint's channel_values.
    tuple_0 = saver.get_tuple(config)
    assert tuple_0 is not None
    assert tuple_0.checkpoint["channel_values"] == {}
    assert tuple_0.checkpoint["channel_versions"] == {}

    # Second checkpoint: the migration attaches the TASKS sends here.
    checkpoint_1 = create_checkpoint(checkpoint_0, {}, 1)
    config = saver.put(config, checkpoint_1, cast(CheckpointMetadata, {}), {})

    tuple_1 = saver.get_tuple(config)
    assert tuple_1 is not None
    assert tuple_1.checkpoint["channel_values"] == {
        TASKS: ["send-1", "send-2", "send-3"]
    }
    assert TASKS in tuple_1.checkpoint["channel_versions"]

    # ``list`` must apply the same migration.
    search_results = list(
        saver.list(cast(RunnableConfig, {"configurable": {"thread_id": "thread-1"}}))
    )
    assert len(search_results) == 2
    assert search_results[-1].checkpoint["channel_values"] == {}
    assert search_results[-1].checkpoint["channel_versions"] == {}
    assert search_results[0].checkpoint["channel_values"] == {
        TASKS: ["send-1", "send-2", "send-3"]
    }
    assert TASKS in search_results[0].checkpoint["channel_versions"]


# ----------------------------------------------- legacy checkpoint compat


def test_get_checkpoint_no_channel_values(
    monkeypatch: pytest.MonkeyPatch,
    saver: SingleStoreSaver,
    test_data: dict[str, list[Any]],
) -> None:
    """Backwards-compat: a checkpoint stored without a ``channel_values`` key
    must still round-trip. ``_load_checkpoint_tuple`` populates the missing
    key with an empty dict."""
    config = cast(
        RunnableConfig,
        {"configurable": {"thread_id": "thread-2", "checkpoint_ns": ""}},
    )
    chkpnt: Checkpoint = create_checkpoint(empty_checkpoint(), {}, 1)
    saver.put(config, chkpnt, cast(CheckpointMetadata, {}), {})

    original = saver._load_checkpoint_tuple

    def patched(value: dict[str, Any]) -> Any:
        value["checkpoint"].pop("channel_values", None)
        return original(value)

    monkeypatch.setattr(saver, "_load_checkpoint_tuple", patched)

    result = saver.get_tuple(config)
    assert result is not None
    assert result.checkpoint["channel_values"] == {}
