"""Unit tests for langchain_singlestore.chat_message_history module."""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy.pool import Pool

from langchain_singlestore.chat_message_history import SingleStoreChatMessageHistory


def _make_history(**kwargs: object) -> SingleStoreChatMessageHistory:
    """Build a chat history backed by a mock pool.

    Passing ``connection_pool`` short-circuits real connection setup so unit
    tests never touch the network.
    """
    params: dict = {
        "session_id": kwargs.pop("session_id", "test-session-id"),
        "connection_pool": MagicMock(spec=Pool),
    }
    params.update(kwargs)
    return SingleStoreChatMessageHistory(**params)  # type: ignore[arg-type]


class TestSingleStoreChatMessageHistory(unittest.TestCase):
    def test_init_sets_session_id(self) -> None:
        history = _make_history()
        assert history.session_id == "testsessionid"

    def test_init_sanitizes_session_id(self) -> None:
        history = _make_history(session_id="test-session!@#$%")
        assert history.session_id == "testsession"

    def test_init_sets_default_table_name(self) -> None:
        history = _make_history()
        assert history.table_name == "message_store"

    def test_init_custom_table_name(self) -> None:
        history = _make_history(table_name="my_messages")
        assert history.table_name == "my_messages"

    def test_init_sets_field_names(self) -> None:
        history = _make_history()
        assert history.id_field == "id"
        assert history.session_id_field == "session_id"
        assert history.message_field == "message"

    def test_init_custom_field_names(self) -> None:
        history = _make_history(
            id_field="custom_id",
            session_id_field="custom_session",
            message_field="custom_message",
        )
        assert history.id_field == "custom_id"
        assert history.session_id_field == "custom_session"
        assert history.message_field == "custom_message"

    def test_init_sets_connector_attributes(self) -> None:
        history = _make_history(host="localhost")
        assert "conn_attrs" in history.connection_kwargs
        assert "_connector_name" in history.connection_kwargs["conn_attrs"]
        assert "_connector_version" in history.connection_kwargs["conn_attrs"]

    def test_sanitize_input_removes_special_chars(self) -> None:
        history = _make_history()
        assert history._sanitize_input("test!@#$%^&*()input") == "testinput"

    def test_sanitize_input_keeps_alphanumeric_and_underscore(self) -> None:
        history = _make_history()
        assert history._sanitize_input("test_123_input") == "test_123_input"

    def test_connection_pool_is_the_injected_mock(self) -> None:
        pool = MagicMock(spec=Pool)
        history = SingleStoreChatMessageHistory(
            session_id="test-session", connection_pool=pool
        )
        assert history.connection_pool is pool

    def test_table_created_flag_initialized_false(self) -> None:
        history = _make_history()
        assert history.table_created is False

    def test_connection_kwargs_includes_custom_params(self) -> None:
        history = _make_history(
            host="localhost", port=3306, user="testuser", password="testpass"
        )
        assert history.connection_kwargs["host"] == "localhost"
        assert history.connection_kwargs["port"] == 3306
        assert history.connection_kwargs["user"] == "testuser"
        assert history.connection_kwargs["password"] == "testpass"

    def test_init_pool_settings(self) -> None:
        """Pool sizing kwargs are forwarded to create_connection_pool."""
        with patch(
            "langchain_singlestore.chat_message_history.create_connection_pool"
        ) as mock_factory:
            mock_factory.return_value = MagicMock(spec=Pool)
            SingleStoreChatMessageHistory(
                session_id="s",
                host="localhost",
                pool_size=10,
                max_overflow=20,
                timeout=60,
            )
        mock_factory.assert_called_once()
        call_kwargs = mock_factory.call_args.kwargs
        assert call_kwargs["pool_size"] == 10
        assert call_kwargs["max_overflow"] == 20
        assert call_kwargs["timeout"] == 60

    def test_init_forwards_connection(self) -> None:
        conn = MagicMock(name="raw_connection")
        with patch(
            "langchain_singlestore.chat_message_history.create_connection_pool"
        ) as mock_factory:
            mock_factory.return_value = MagicMock(spec=Pool)
            SingleStoreChatMessageHistory(session_id="s", connection=conn)
        assert mock_factory.call_args.kwargs["connection"] is conn


class TestSingleStoreChatMessageHistoryMessages(unittest.TestCase):
    def _history(self) -> tuple[SingleStoreChatMessageHistory, MagicMock, MagicMock]:
        pool = MagicMock(spec=Pool)
        conn = MagicMock()
        pool.connect.return_value = conn
        history = SingleStoreChatMessageHistory(
            session_id="test-session", connection_pool=pool
        )
        return history, pool, conn

    @patch("langchain_singlestore.chat_message_history.messages_from_dict")
    def test_messages_creates_table_if_not_exists(
        self, mock_messages_from_dict: MagicMock
    ) -> None:
        history, _, _ = self._history()
        mock_messages_from_dict.return_value = []
        _ = history.messages
        assert history.table_created is True

    @patch("langchain_singlestore.chat_message_history.messages_from_dict")
    def test_messages_retrieves_from_database(
        self, mock_messages_from_dict: MagicMock
    ) -> None:
        history, _, conn = self._history()
        mock_cursor = MagicMock()
        conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            ['{"type": "human", "data": {"content": "test"}}']
        ]
        mock_messages_from_dict.return_value = [HumanMessage(content="test")]

        messages = history.messages
        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)

    @patch("langchain_singlestore.chat_message_history.message_to_dict")
    def test_add_message_inserts_to_database(
        self, mock_message_to_dict: MagicMock
    ) -> None:
        history, _, conn = self._history()
        mock_cursor = MagicMock()
        conn.cursor.return_value = mock_cursor
        mock_message_to_dict.return_value = {"type": "ai", "data": {"content": "test"}}

        history.add_message(AIMessage(content="test"))
        assert mock_cursor.execute.called

    def test_clear_deletes_from_database(self) -> None:
        history, _, conn = self._history()
        mock_cursor = MagicMock()
        conn.cursor.return_value = mock_cursor

        history.clear()
        assert mock_cursor.execute.called
        assert "DELETE" in mock_cursor.execute.call_args[0][0]


if __name__ == "__main__":
    unittest.main()
