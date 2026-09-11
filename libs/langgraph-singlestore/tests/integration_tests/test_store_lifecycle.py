"""Integration tests for :class:`SingleStoreStore` lifecycle and connection
injection.

Covers behaviours that need a real SingleStore container to observe:

* ``close()`` is idempotent and does not tear down caller-owned resources.
* ``close()`` stops a running TTL sweeper thread.
* Constructing the store with an existing ``Connection`` yields a working
  end-to-end batch.
* ``setup()`` is idempotent against a non-vector store.
"""

from __future__ import annotations

import time
from contextlib import closing

from singlestore_langchain_core._connection import QueueConnectionPool
from singlestoredb.connection import connect

from langgraph.store.base import GetOp, Item, PutOp, TTLConfig
from langgraph.store.singlestore import SingleStoreStore

from .conftest import ConnectionParameters


def _table_exists(params: ConnectionParameters, table: str) -> bool:
    conn = connect(**params.as_kwargs())
    try:
        with closing(conn.cursor()) as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = %s AND table_name = %s",
                (params.database, table),
            )
            return cur.fetchone() is not None
    finally:
        conn.close()


def _migration_versions(params: ConnectionParameters) -> list[int]:
    conn = connect(**params.as_kwargs())
    try:
        with closing(conn.cursor()) as cur:
            cur.execute("SELECT v FROM store_migrations ORDER BY v")
            return [int(list(row)[0]) for row in cur.fetchall()]
    finally:
        conn.close()


class TestCloseIdempotency:
    def test_close_is_idempotent_on_owned_pool(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Calling ``close()`` twice on a store that owns its pool must not
        raise; the second call is a no-op."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        store.setup()
        store.close()
        # Second close on an already-disposed pool must be a no-op.
        store.close()

    def test_close_stops_running_ttl_sweeper(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """``close()`` calls ``stop_ttl_sweeper`` before disposing the pool
        so a live sweeper thread doesn't outlive the store or race the
        pool teardown."""
        store = SingleStoreStore(
            ttl_config=TTLConfig(sweep_interval_minutes=60),
            **connection_parameters.as_kwargs(),
        )
        store.setup()
        future = store.start_ttl_sweeper()
        thread = store._ttl_sweeper_thread
        assert thread is not None and thread.is_alive()

        store.close()

        # ``close()`` invoked ``stop_ttl_sweeper(timeout=0.1)`` -- that
        # should have been enough to trip the stop event and let the loop
        # exit before the pool went away.
        deadline = time.time() + 5.0
        while thread.is_alive() and time.time() < deadline:
            time.sleep(0.05)
        assert not thread.is_alive()
        assert future.result(timeout=5.0) is None


class TestCloseWithCallerOwnedResources:
    def test_close_leaves_injected_pool_usable(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """A pool supplied via ``connection_pool=`` must survive ``store.close()``
        so the caller can keep using it."""
        pool = QueueConnectionPool(
            pool_size=1,
            max_overflow=1,
            timeout=5,
            connection_kwargs=connection_parameters.as_kwargs(),
        )
        try:
            store = SingleStoreStore(connection_pool=pool)
            store.setup()
            store.close()

            # Pool still works after store.close() -- would raise if the
            # store had disposed the caller-owned pool.
            conn = pool.connect()
            try:
                cur = conn.cursor()
                cur.execute("SELECT 1")
                row = cur.fetchone()
                assert row is not None
                assert int(list(row)[0]) == 1
                cur.close()
            finally:
                conn.close()
        finally:
            pool.dispose()

    def test_close_leaves_injected_connection_usable(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """A ``Connection`` supplied via ``connection=`` must survive
        ``store.close()``."""
        conn = connect(**connection_parameters.as_kwargs())
        try:
            store = SingleStoreStore(connection=conn)
            store.setup()
            store.close()

            cur = conn.cursor()
            cur.execute("SELECT 1")
            row = cur.fetchone()
            assert row is not None
            assert int(list(row)[0]) == 1
            cur.close()
        finally:
            conn.close()


class TestConstructWithExistingConnection:
    def test_end_to_end_put_and_get_via_injected_connection(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """The ``connection=`` constructor path must produce a fully
        functional store: setup + put + get end-to-end."""
        conn = connect(**connection_parameters.as_kwargs())
        try:
            store = SingleStoreStore(connection=conn)
            try:
                store.setup()
                store.batch([PutOp(("users", "alice"), "prefs", {"theme": "dark"})])
                results = store.batch([GetOp(("users", "alice"), "prefs")])
                item = results[0]
                assert isinstance(item, Item)
                assert item.value == {"theme": "dark"}
            finally:
                store.close()
        finally:
            conn.close()


class TestSetupIdempotencyNonVector:
    def test_setup_twice_does_not_duplicate_migrations_or_raise(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """Repeated ``setup()`` calls must be safe. Once every migration has
        been applied the second call must be a no-op -- no re-inserted
        ``store_migrations`` rows and no ``CREATE TABLE`` re-executed
        against a populated schema."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            versions_first = _migration_versions(connection_parameters)
            assert versions_first  # at least one migration applied
            assert versions_first == sorted(set(versions_first))

            store.setup()
            versions_second = _migration_versions(connection_parameters)
            assert versions_second == versions_first

            # The store table survived and is usable.
            assert _table_exists(connection_parameters, "store")
            store.batch([PutOp(("t",), "k", {"v": 1})])
            got = store.batch([GetOp(("t",), "k")])
            item = got[0]
            assert isinstance(item, Item)
            assert item.value == {"v": 1}
        finally:
            store.close()

    def test_setup_resumes_from_last_applied_version(
        self, connection_parameters: ConnectionParameters
    ) -> None:
        """After ``setup()`` completes, the highest recorded migration
        version equals ``len(MIGRATIONS) - 1``. This pins the contract
        used by the resume slice ``MIGRATIONS[version + 1 :]``."""
        store = SingleStoreStore(**connection_parameters.as_kwargs())
        try:
            store.setup()
            versions = _migration_versions(connection_parameters)
            assert versions == list(range(len(SingleStoreStore.MIGRATIONS)))
        finally:
            store.close()
