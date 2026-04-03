import json

from langchain_core.messages import AIMessage, HumanMessage, message_to_dict

from langchain_singlestore import SingleStoreChatMessageHistory

from tests.integration_tests.conftest import ConnectionParameters

# Replace these with your SingleStoreDB connection string


def test_memory_with_message_store(clean_db_connection_parameters: ConnectionParameters) -> None:
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
