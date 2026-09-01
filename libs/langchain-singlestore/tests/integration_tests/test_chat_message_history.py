import json

from langchain_core.messages import AIMessage, HumanMessage, message_to_dict

from langchain_singlestore import SingleStoreChatMessageHistory
from tests.integration_tests.conftest import ConnectionParameters

# Replace these with your SingleStoreDB connection string


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
        assert h1.connection_pool is pool
        assert h2.connection_pool is pool

        h1.add_message(AIMessage(content="one"))
        h2.add_message(HumanMessage(content="two"))
        assert [m.content for m in h1.messages] == ["one"]
        assert [m.content for m in h2.messages] == ["two"]
        h1.clear()
        h2.clear()
    finally:
        pool.dispose()
