"""Unit tests for langchain_singlestore.chat_message_history module."""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from langchain_singlestore.chat_message_history import SingleStoreChatMessageHistory


class TestSingleStoreChatMessageHistory(unittest.TestCase):
    """Test SingleStoreChatMessageHistory class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.session_id = "test-session-id"
        self.patcher = patch("langchain_singlestore.chat_message_history.QueuePool")
        self.mock_pool_class = self.patcher.start()
        self.mock_pool = MagicMock()
        self.mock_pool_class.return_value = self.mock_pool

    def tearDown(self) -> None:
        """Clean up patches."""
        self.patcher.stop()

    def test_init_sets_session_id(self) -> None:
        """Test that __init__ sets session_id correctly (sanitized)."""
        chat_history = SingleStoreChatMessageHistory(
            session_id=self.session_id, host="localhost"
        )
        # Session ID gets sanitized: hyphens are removed
        assert chat_history.session_id == "testsessionid"

    def test_init_sanitizes_session_id(self) -> None:
        """Test that session_id is sanitized during init."""
        chat_history = SingleStoreChatMessageHistory(
            session_id="test-session!@#$%", host="localhost"
        )
        # Sanitization removes special characters
        assert chat_history.session_id == "testsession"

    def test_init_sets_default_table_name(self) -> None:
        """Test that default table_name is set."""
        chat_history = SingleStoreChatMessageHistory(
            session_id=self.session_id, host="localhost"
        )
        assert chat_history.table_name == "message_store"

    def test_init_custom_table_name(self) -> None:
        """Test that custom table_name is used."""
        chat_history = SingleStoreChatMessageHistory(
            session_id=self.session_id, host="localhost", table_name="my_messages"
        )
        assert chat_history.table_name == "my_messages"

    def test_init_sets_field_names(self) -> None:
        """Test that field names are set correctly."""
        chat_history = SingleStoreChatMessageHistory(
            session_id=self.session_id, host="localhost"
        )
        assert chat_history.id_field == "id"
        assert chat_history.session_id_field == "session_id"
        assert chat_history.message_field == "message"

    def test_init_custom_field_names(self) -> None:
        """Test that custom field names are used."""
        chat_history = SingleStoreChatMessageHistory(
            session_id=self.session_id,
            host="localhost",
            id_field="custom_id",
            session_id_field="custom_session",
            message_field="custom_message",
        )
        assert chat_history.id_field == "custom_id"
        assert chat_history.session_id_field == "custom_session"
        assert chat_history.message_field == "custom_message"

    def test_init_sets_connector_attributes(self) -> None:
        """Test that connector attributes are set during init."""
        chat_history = SingleStoreChatMessageHistory(
            session_id=self.session_id, host="localhost"
        )
        assert "conn_attrs" in chat_history.connection_kwargs
        assert "_connector_name" in chat_history.connection_kwargs["conn_attrs"]
        assert "_connector_version" in chat_history.connection_kwargs["conn_attrs"]

    def test_sanitize_input_removes_special_chars(self) -> None:
        """Test that _sanitize_input removes special characters."""
        chat_history = SingleStoreChatMessageHistory(
            session_id=self.session_id, host="localhost"
        )
        result = chat_history._sanitize_input("test!@#$%^&*()input")
        assert result == "testinput"

    def test_sanitize_input_keeps_alphanumeric_and_underscore(self) -> None:
        """Test that _sanitize_input keeps alphanumeric and underscore."""
        chat_history = SingleStoreChatMessageHistory(
            session_id=self.session_id, host="localhost"
        )
        result = chat_history._sanitize_input("test_123_input")
        assert result == "test_123_input"

    def test_connection_pool_created(self) -> None:
        """Test that connection pool is created."""
        chat_history = SingleStoreChatMessageHistory(
            session_id=self.session_id, host="localhost"
        )
        assert chat_history.connection_pool is not None

    def test_table_created_flag_initialized_false(self) -> None:
        """Test that table_created flag is initialized as False."""
        chat_history = SingleStoreChatMessageHistory(
            session_id=self.session_id, host="localhost"
        )
        assert chat_history.table_created is False

    def test_connection_kwargs_includes_host(self) -> None:
        """Test that host is included in connection_kwargs."""
        host = "test-host"
        chat_history = SingleStoreChatMessageHistory(
            session_id=self.session_id, host=host
        )
        assert chat_history.connection_kwargs["host"] == host

    def test_connection_kwargs_includes_custom_params(self) -> None:
        """Test that custom parameters are included in connection_kwargs."""
        chat_history = SingleStoreChatMessageHistory(
            session_id=self.session_id,
            host="localhost",
            port=3306,
            user="testuser",
            password="testpass",
        )
        assert chat_history.connection_kwargs["port"] == 3306
        assert chat_history.connection_kwargs["user"] == "testuser"
        assert chat_history.connection_kwargs["password"] == "testpass"


class TestSingleStoreChatMessageHistoryMessages(unittest.TestCase):
    """Test message-related methods of SingleStoreChatMessageHistory."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.patcher = patch("langchain_singlestore.chat_message_history.QueuePool")
        self.mock_pool_class = self.patcher.start()
        self.mock_pool = MagicMock()
        self.mock_pool_class.return_value = self.mock_pool

        self.mock_conn = MagicMock()
        self.mock_pool.connect.return_value = self.mock_conn

    def tearDown(self) -> None:
        """Clean up patches."""
        self.patcher.stop()

    @patch("langchain_singlestore.chat_message_history.messages_from_dict")
    def test_messages_creates_table_if_not_exists(
        self, mock_messages_from_dict: MagicMock
    ) -> None:
        """Test that messages property creates table if needed."""
        chat_history = SingleStoreChatMessageHistory(
            session_id="test-session", host="localhost"
        )
        mock_messages_from_dict.return_value = []

        _ = chat_history.messages

        # _create_table_if_not_exists should be called
        assert chat_history.table_created is True

    @patch("langchain_singlestore.chat_message_history.messages_from_dict")
    def test_messages_retrieves_from_database(
        self, mock_messages_from_dict: MagicMock
    ) -> None:
        """Test that messages property retrieves messages from database."""
        chat_history = SingleStoreChatMessageHistory(
            session_id="test-session", host="localhost"
        )

        mock_cursor = MagicMock()
        self.mock_conn.cursor.return_value = mock_cursor
        test_message_data = ['{"type": "human", "data": {"content": "test"}}']
        mock_cursor.fetchall.return_value = [[msg] for msg in test_message_data]
        mock_messages_from_dict.return_value = [HumanMessage(content="test")]

        messages = chat_history.messages

        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)

    @patch("langchain_singlestore.chat_message_history.message_to_dict")
    def test_add_message_inserts_to_database(
        self, mock_message_to_dict: MagicMock
    ) -> None:
        """Test that add_message inserts message to database."""
        chat_history = SingleStoreChatMessageHistory(
            session_id="test-session", host="localhost"
        )

        mock_cursor = MagicMock()
        self.mock_conn.cursor.return_value = mock_cursor
        mock_message_to_dict.return_value = {"type": "ai", "data": {"content": "test"}}

        message = AIMessage(content="test")
        chat_history.add_message(message)

        # Verify cursor.execute was called
        assert mock_cursor.execute.called

    def test_clear_deletes_from_database(self) -> None:
        """Test that clear method deletes messages from database."""
        chat_history = SingleStoreChatMessageHistory(
            session_id="test-session", host="localhost"
        )

        mock_cursor = MagicMock()
        self.mock_conn.cursor.return_value = mock_cursor

        chat_history.clear()

        # Verify cursor.execute was called with DELETE
        assert mock_cursor.execute.called
        call_args = mock_cursor.execute.call_args
        assert "DELETE" in call_args[0][0]


if __name__ == "__main__":
    unittest.main()
