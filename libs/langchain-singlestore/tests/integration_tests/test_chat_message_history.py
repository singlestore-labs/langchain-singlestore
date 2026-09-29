import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, message_to_dict
from singlestore_langchain_core._connection import (
    CallerOwnedConnectionPool,
    QueueConnectionPool,
    SingleConnectionPool,
)

from langchain_singlestore import SingleStoreChatMessageHistory
from tests.integration_tests.conftest import ConnectionParameters


def _connection_kwargs(params: ConnectionParameters) -> dict[str, Any]:
    return {
        "host": params.Host,
        "port": params.Port,
        "user": params.User,
        "password": params.Password,
        "database": params.Database,
    }


def _make_history(
    params: ConnectionParameters, **kwargs: Any
) -> SingleStoreChatMessageHistory:
    return SingleStoreChatMessageHistory(**_connection_kwargs(params), **kwargs)


def test_memory_with_message_store(
    clean_db_connection_parameters: ConnectionParameters,
) -> None:
    """Test the message store with SingleStoreChatMessageHistory."""
    # setup SingleStoreDB as a message store
    message_history = SingleStoreChatMessageHistory(
        session_id="test-session",
        host=clean_db_connection_parameters.Host,
        port=clean_db_connection_parameters.Port,
        user=clean_db_connection_parameters.User,
        password=clean_db_connection_parameters.Password,
        database=clean_db_connection_parameters.Database,
    )

    # add some messages
    message_history.add_message(AIMessage(content="This is me, the AI"))
    message_history.add_message(HumanMessage(content="This is me, the human"))

    # get the message history from the memory store and turn it into a json
    messages = message_history.messages
    messages_json = json.dumps([message_to_dict(msg) for msg in messages])

    assert "This is me, the AI" in messages_json
    assert "This is me, the human" in messages_json

    # remove the record from SingleStoreDB, so the next test run won't pick it up
    message_history.clear()

    assert message_history.messages == []


def test_message_history_with_shared_connection(
    clean_db_connection_parameters: ConnectionParameters,
) -> None:
    """A caller-owned connection is reused across every operation."""
    import singlestoredb

    conn = singlestoredb.connect(
        host=clean_db_connection_parameters.Host,
        port=clean_db_connection_parameters.Port,
        user=clean_db_connection_parameters.User,
        password=clean_db_connection_parameters.Password,
        database=clean_db_connection_parameters.Database,
    )
    try:
        history = SingleStoreChatMessageHistory(
            session_id="shared-conn-session",
            connection=conn,
        )
        history.add_message(AIMessage(content="hello from shared conn"))
        history.add_message(HumanMessage(content="hi back"))
        messages = history.messages
        contents = [m.content for m in messages]
        assert "hello from shared conn" in contents
        assert "hi back" in contents
        history.clear()
        # Underlying connection must survive the pool's dispose semantics.
        history.connection_pool.dispose()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            assert cur.fetchone()[0] == 1  # type: ignore[index]
    finally:
        conn.close()


def test_message_history_with_shared_connection_pool(
    clean_db_connection_parameters: ConnectionParameters,
) -> None:
    """Two histories can share a single caller-managed pool."""
    from singlestore_langchain_core import create_connection_pool

    pool = create_connection_pool(
        pool_size=2,
        max_overflow=0,
        timeout=10,
        connection_kwargs={
            "host": clean_db_connection_parameters.Host,
            "port": clean_db_connection_parameters.Port,
            "user": clean_db_connection_parameters.User,
            "password": clean_db_connection_parameters.Password,
            "database": clean_db_connection_parameters.Database,
        },
    )
    try:
        h1 = SingleStoreChatMessageHistory(session_id="s1", connection_pool=pool)
        h2 = SingleStoreChatMessageHistory(session_id="s2", connection_pool=pool)
        assert isinstance(h1.connection_pool, CallerOwnedConnectionPool)
        assert isinstance(h2.connection_pool, CallerOwnedConnectionPool)
        assert h1.connection_pool._connection_pool is pool
        assert h2.connection_pool._connection_pool is pool

        h1.add_message(AIMessage(content="one"))
        h2.add_message(HumanMessage(content="two"))
        assert [m.content for m in h1.messages] == ["one"]
        assert [m.content for m in h2.messages] == ["two"]
        h1.clear()
        h2.clear()
    finally:
        pool.dispose()


def test_add_messages_bulk_insert(
    clean_db_connection_parameters: ConnectionParameters,
) -> None:
    """``add_messages`` inserts every message and preserves order."""
    history = _make_history(clean_db_connection_parameters, session_id="bulk")
    try:
        batch = [
            HumanMessage(content="q1"),
            AIMessage(content="a1"),
            HumanMessage(content="q2"),
            AIMessage(content="a2"),
        ]
        history.add_messages(batch)

        assert [m.content for m in history.messages] == ["q1", "a1", "q2", "a2"]
    finally:
        history.clear()
        history.close()


def test_add_messages_empty_is_noop(
    clean_db_connection_parameters: ConnectionParameters,
) -> None:
    """An empty batch must not touch the database or fail."""
    history = _make_history(clean_db_connection_parameters, session_id="empty-batch")
    try:
        history.add_messages([])
        assert history.messages == []
    finally:
        history.close()


def test_messages_returned_in_insertion_order(
    clean_db_connection_parameters: ConnectionParameters,
) -> None:
    """Messages must come back in the order they were inserted."""
    history = _make_history(clean_db_connection_parameters, session_id="ordered")
    try:
        expected = [f"msg-{i:02d}" for i in range(20)]
        for text in expected:
            history.add_message(HumanMessage(content=text))
        assert [m.content for m in history.messages] == expected
    finally:
        history.clear()
        history.close()


def test_sessions_are_isolated_within_same_table(
    clean_db_connection_parameters: ConnectionParameters,
) -> None:
    """Two sessions on the same table must not read each other's messages."""
    h_a = _make_history(clean_db_connection_parameters, session_id="alpha")
    h_b = _make_history(clean_db_connection_parameters, session_id="beta")
    try:
        h_a.add_messages(
            [HumanMessage(content="alpha-1"), AIMessage(content="alpha-2")]
        )
        h_b.add_messages([HumanMessage(content="beta-1")])

        assert [m.content for m in h_a.messages] == ["alpha-1", "alpha-2"]
        assert [m.content for m in h_b.messages] == ["beta-1"]

        h_a.clear()
        assert h_a.messages == []
        assert [m.content for m in h_b.messages] == ["beta-1"]
    finally:
        h_a.clear()
        h_b.clear()
        h_a.close()
        h_b.close()


def test_custom_table_and_field_names(
    clean_db_connection_parameters: ConnectionParameters,
) -> None:
    """Custom identifiers are honoured and the table is auto-created."""
    history = _make_history(
        clean_db_connection_parameters,
        session_id="custom",
        table_name="my_messages",
        id_field="msg_id",
        session_id_field="sess",
        message_field="payload",
    )
    try:
        history.add_message(AIMessage(content="hello"))
        assert [m.content for m in history.messages] == ["hello"]

        import singlestoredb

        with singlestoredb.connect(
            **_connection_kwargs(clean_db_connection_parameters)
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("SHOW COLUMNS FROM my_messages")
                columns = {list(row)[0] for row in cur.fetchall()}
        assert {"msg_id", "sess", "payload"} <= columns
    finally:
        history.clear()
        history.close()


def test_add_user_and_ai_message_helpers(
    clean_db_connection_parameters: ConnectionParameters,
) -> None:
    """Inherited convenience helpers still work end-to-end."""
    history = _make_history(clean_db_connection_parameters, session_id="helpers")
    try:
        history.add_user_message("hi")
        history.add_ai_message("hello")
        messages = history.messages
        assert [type(m) for m in messages] == [HumanMessage, AIMessage]
        assert [m.content for m in messages] == ["hi", "hello"]
    finally:
        history.clear()
        history.close()


def test_clear_is_safe_when_empty(
    clean_db_connection_parameters: ConnectionParameters,
) -> None:
    """Calling ``clear`` on an empty session is a no-op."""
    history = _make_history(clean_db_connection_parameters, session_id="empty-clear")
    try:
        history.clear()
        history.clear()
        assert history.messages == []
    finally:
        history.close()


def test_session_id_has_index(
    clean_db_connection_parameters: ConnectionParameters,
) -> None:
    """The created table indexes ``session_id`` to keep lookups cheap."""
    history = _make_history(
        clean_db_connection_parameters,
        session_id="indexed",
        table_name="indexed_messages",
    )
    try:
        history.add_message(AIMessage(content="seed"))

        import singlestoredb

        with singlestoredb.connect(
            **_connection_kwargs(clean_db_connection_parameters)
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("SHOW INDEX FROM indexed_messages")
                indexed_columns = {list(row)[4] for row in cur.fetchall()}
        assert "session_id" in indexed_columns
    finally:
        history.clear()
        history.close()


def test_context_manager_disposes_owned_pool(
    clean_db_connection_parameters: ConnectionParameters,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exiting the ``with`` block disposes the pool the class created."""
    dispose_calls = 0
    original_dispose = QueueConnectionPool.dispose

    def counting_dispose(self: QueueConnectionPool) -> None:
        nonlocal dispose_calls
        dispose_calls += 1
        original_dispose(self)

    monkeypatch.setattr(QueueConnectionPool, "dispose", counting_dispose)

    with SingleStoreChatMessageHistory(
        session_id="ctx-owned",
        **_connection_kwargs(clean_db_connection_parameters),
    ) as history:
        history.add_message(HumanMessage(content="in-context"))
        assert [m.content for m in history.messages] == ["in-context"]
        history.clear()
        assert isinstance(history.connection_pool, QueueConnectionPool)
        assert dispose_calls == 0

    assert dispose_calls >= 1


def test_context_manager_preserves_caller_owned_connection(
    clean_db_connection_parameters: ConnectionParameters,
) -> None:
    """Exiting the ``with`` block must not close a caller-owned connection."""
    import singlestoredb

    conn = singlestoredb.connect(**_connection_kwargs(clean_db_connection_parameters))
    try:
        with SingleStoreChatMessageHistory(
            session_id="ctx-shared",
            connection=conn,
        ) as history:
            assert isinstance(history.connection_pool, SingleConnectionPool)
            history.add_message(AIMessage(content="shared"))
            history.clear()

        # Caller-owned connection must survive the context manager exit.
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            assert cur.fetchone()[0] == 1  # type: ignore[index]
    finally:
        conn.close()


def test_context_manager_preserves_caller_owned_pool(
    clean_db_connection_parameters: ConnectionParameters,
) -> None:
    """Exiting the ``with`` block must not dispose a caller-owned pool."""
    from singlestore_langchain_core import create_connection_pool

    pool = create_connection_pool(
        pool_size=1,
        max_overflow=0,
        timeout=10,
        connection_kwargs=_connection_kwargs(clean_db_connection_parameters),
    )
    try:
        with SingleStoreChatMessageHistory(
            session_id="ctx-pool",
            connection_pool=pool,
        ) as history:
            assert isinstance(history.connection_pool, CallerOwnedConnectionPool)
            history.add_message(HumanMessage(content="pooled"))
            history.clear()

        # The caller's pool must still be usable after the context manager exits.
        checkout = pool.connect()
        try:
            cur = checkout.cursor()
            try:
                cur.execute("SELECT 1")
                assert cur.fetchone()[0] == 1  # type: ignore[index]
            finally:
                cur.close()
        finally:
            checkout.close()
    finally:
        pool.dispose()


def test_close_is_idempotent(
    clean_db_connection_parameters: ConnectionParameters,
) -> None:
    """Calling ``close`` more than once must not raise."""
    history = _make_history(clean_db_connection_parameters, session_id="close-twice")
    history.add_message(AIMessage(content="bye"))
    history.clear()
    history.close()
    history.close()


async def test_async_methods_roundtrip(
    clean_db_connection_parameters: ConnectionParameters,
) -> None:
    """``aadd_messages`` / ``aget_messages`` / ``aclear`` work end-to-end."""
    history = _make_history(clean_db_connection_parameters, session_id="async")
    try:
        await history.aadd_messages(
            [HumanMessage(content="async-q"), AIMessage(content="async-a")]
        )
        messages = await history.aget_messages()
        assert [m.content for m in messages] == ["async-q", "async-a"]

        await history.aclear()
        assert await history.aget_messages() == []
    finally:
        history.close()
