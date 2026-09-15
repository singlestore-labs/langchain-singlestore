from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from singlestore_langchain_core import FilterTypedDict, _parse_filter

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.types import TASKS

MetadataInput = dict[str, Any] | None

"""
To add a new migration, add a new string to the MIGRATIONS list.
The position of the migration in the list is the version number.
"""
MIGRATIONS = [
    """CREATE TABLE IF NOT EXISTS checkpoint_migrations (
	v INTEGER PRIMARY KEY
);""",
    """CREATE TABLE IF NOT EXISTS checkpoints (
	thread_id TEXT NOT NULL,
	checkpoint_ns TEXT NOT NULL DEFAULT '',
	checkpoint_id TEXT NOT NULL,
	parent_checkpoint_id TEXT,
	type TEXT,
	checkpoint JSON NOT NULL,
	metadata JSON NOT NULL DEFAULT '{}',
	PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id),
	INDEX checkpoints_thread_id_idx (thread_id)
);""",
    """CREATE TABLE IF NOT EXISTS checkpoint_blobs (
	thread_id TEXT NOT NULL,
	checkpoint_ns TEXT NOT NULL DEFAULT '',
	channel TEXT NOT NULL,
	version TEXT NOT NULL,
	type TEXT NOT NULL,
	`blob` LONGBLOB NULL,
	PRIMARY KEY (thread_id, checkpoint_ns, channel, version),
	INDEX checkpoint_blobs_thread_id_idx (thread_id)
);""",
    """CREATE TABLE IF NOT EXISTS checkpoint_writes (
	thread_id TEXT NOT NULL,
	checkpoint_ns TEXT NOT NULL DEFAULT '',
	checkpoint_id TEXT NOT NULL,
	task_id TEXT NOT NULL,
	idx INTEGER NOT NULL,
	channel TEXT NOT NULL,
	type TEXT,
	`blob` LONGBLOB NOT NULL,
	task_path TEXT NOT NULL DEFAULT '',
	PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx),
	INDEX checkpoint_writes_thread_id_idx (thread_id)
);""",
]

SELECT_SQL = """
SELECT
	c.thread_id,
	c.checkpoint,
	c.checkpoint_ns,
	c.checkpoint_id,
	c.parent_checkpoint_id,
	c.metadata,
	CASE
		WHEN COUNT(bl.channel) > 0
		THEN JSON_AGG(JSON_BUILD_ARRAY(bl.channel, bl.type, HEX(bl.blob)))
		ELSE JSON_BUILD_ARRAY()
	END AS channel_values,
	(
		SELECT JSON_AGG(
			JSON_BUILD_ARRAY(cw.task_id, cw.channel, cw.type, HEX(cw.blob))
		)
		FROM checkpoint_writes cw
		WHERE cw.thread_id = c.thread_id
			AND cw.checkpoint_ns = c.checkpoint_ns
			AND cw.checkpoint_id = c.checkpoint_id
	) AS pending_writes
FROM checkpoints c
LEFT JOIN checkpoint_blobs bl
	ON bl.thread_id = c.thread_id
	AND bl.checkpoint_ns = c.checkpoint_ns
	AND JSON_EXTRACT_STRING(c.checkpoint, 'channel_versions', bl.channel) IS NOT NULL
	AND bl.version = JSON_EXTRACT_STRING(c.checkpoint, 'channel_versions', bl.channel)
{{where}}
GROUP BY
	c.thread_id,
	c.checkpoint,
	c.checkpoint_ns,
	c.checkpoint_id,
	c.parent_checkpoint_id,
	c.metadata
"""

SELECT_PENDING_SENDS_SQL = f"""
select
	checkpoint_id,
	JSON_AGG(JSON_BUILD_ARRAY(type, HEX(`blob`)) order by task_path, task_id, idx)
    as sends
from checkpoint_writes
where thread_id = %s
	and checkpoint_id IN (%s)
	and channel = '{TASKS}'
group by checkpoint_id
"""

UPSERT_CHECKPOINT_BLOBS_SQL = """
	INSERT INTO
    checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type, `blob`)
	VALUES (%s, %s, %s, %s, %s, %s)
	ON DUPLICATE KEY UPDATE
		type = VALUES(type),
		`blob` = VALUES(`blob`)
"""

UPSERT_CHECKPOINTS_SQL = """
	INSERT INTO
    checkpoints (
        thread_id, checkpoint_ns, checkpoint_id,
        parent_checkpoint_id, checkpoint, metadata
    )
	VALUES (%s, %s, %s, %s, %s, %s)
	ON DUPLICATE KEY UPDATE
		checkpoint = VALUES(checkpoint),
		metadata = VALUES(metadata)
"""

UPSERT_CHECKPOINT_WRITES_SQL = """
	INSERT INTO checkpoint_writes (
		thread_id, checkpoint_ns, checkpoint_id,
        task_id, task_path, idx, channel, type, `blob`
	)
	VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
	ON DUPLICATE KEY UPDATE
		channel = VALUES(channel),
		type = VALUES(type),
		`blob` = VALUES(`blob`)
"""

INSERT_CHECKPOINT_WRITES_SQL = """
	INSERT IGNORE INTO checkpoint_writes (
		thread_id, checkpoint_ns, checkpoint_id,
        task_id, task_path, idx, channel, type, `blob`
	)
	VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


class BaseSingleStoreSaver(BaseCheckpointSaver[str]):
    SELECT_SQL = SELECT_SQL
    SELECT_PENDING_SENDS_SQL = SELECT_PENDING_SENDS_SQL
    MIGRATIONS = MIGRATIONS
    UPSERT_CHECKPOINT_BLOBS_SQL = UPSERT_CHECKPOINT_BLOBS_SQL
    UPSERT_CHECKPOINTS_SQL = UPSERT_CHECKPOINTS_SQL
    UPSERT_CHECKPOINT_WRITES_SQL = UPSERT_CHECKPOINT_WRITES_SQL
    INSERT_CHECKPOINT_WRITES_SQL = INSERT_CHECKPOINT_WRITES_SQL

    def _migrate_pending_sends(
        self,
        pending_sends: list[tuple[bytes, bytes]],
        checkpoint: dict[str, Any],
        channel_values: list[tuple[bytes, bytes, bytes]],
    ) -> None:
        if not pending_sends:
            return
        # add to values
        enc, blob = self.serde.dumps_typed(
            [
                self.serde.loads_typed(
                    (c.decode(), bytes.fromhex(b) if isinstance(b, str) else b)
                )
                for c, b in pending_sends
            ],
        )
        channel_values.append((TASKS.encode(), enc.encode(), blob))
        # add to versions
        checkpoint["channel_versions"][TASKS] = (
            max(checkpoint["channel_versions"].values())
            if checkpoint["channel_versions"]
            else self.get_next_version(None, None)
        )

    def _load_blobs(
        self, blob_values: list[tuple[bytes, bytes, bytes]]
    ) -> dict[str, Any]:
        if not blob_values:
            return {}
        return {
            k.decode(): self.serde.loads_typed((t.decode(), v))
            for k, t, v in blob_values
            if t.decode() != "empty"
        }

    def _dump_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        values: dict[str, Any],
        versions: ChannelVersions,
    ) -> list[tuple[str, str, str, str, str, bytes | None]]:
        if not versions:
            return []

        return [
            (
                thread_id,
                checkpoint_ns,
                k,
                cast(str, ver),
                *(
                    self.serde.dumps_typed(values[k])
                    if k in values
                    else ("empty", None)
                ),
            )
            for k, ver in versions.items()
        ]

    def _load_writes(
        self, writes: list[tuple[bytes, bytes, bytes, bytes]]
    ) -> list[tuple[str, str, Any]]:
        return (
            [
                (
                    tid.decode(),
                    channel.decode(),
                    self.serde.loads_typed((t.decode(), v)),
                )
                for tid, channel, t, v in writes
            ]
            if writes
            else []
        )

    def _dump_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        task_path: str,
        writes: Sequence[tuple[str, Any]],
    ) -> list[tuple[str, str, str, str, str, int, str, str, bytes]]:
        return [
            (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                task_path,
                WRITES_IDX_MAP.get(channel, idx),
                channel,
                *self.serde.dumps_typed(value),
            )
            for idx, (channel, value) in enumerate(writes)
        ]

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    def _search_where(
        self,
        config: RunnableConfig | None,
        filter: MetadataInput,
        before: RunnableConfig | None = None,
    ) -> tuple[str, list[Any]]:
        """Return WHERE clause predicates for alist() given config, filter, before.

        This method returns a tuple of a string and a list of values. The string
        is the parametered WHERE clause predicate (including the WHERE keyword):
        "WHERE column1 = %s AND column2 = %s". The list of values contains the
        values for each of the corresponding parameters.
        """
        wheres = []
        param_values = []

        # construct predicate for config filter
        if config:
            wheres.append("c.thread_id = %s ")
            param_values.append(config["configurable"]["thread_id"])
            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                wheres.append("c.checkpoint_ns = %s")
                param_values.append(checkpoint_ns)

            if checkpoint_id := get_checkpoint_id(config):
                wheres.append("c.checkpoint_id = %s ")
                param_values.append(checkpoint_id)

        # construct predicate for metadata filter
        if filter:
            adjusted_filter = cast(
                FilterTypedDict,
                {"$and": [{k: v} for k, v in filter.items()]},
            )
            filter_conditions, filter_params = _parse_filter(
                filter_dict=adjusted_filter, metadata_field="c.metadata"
            )
            if filter_conditions:
                wheres.append(f"({filter_conditions})")
                param_values.extend(filter_params)

        # construct predicate for `before`
        if before is not None:
            wheres.append("c.checkpoint_id < %s ")
            param_values.append(get_checkpoint_id(before))

        return (
            "WHERE " + " AND ".join(wheres) if wheres else "",
            param_values,
        )
