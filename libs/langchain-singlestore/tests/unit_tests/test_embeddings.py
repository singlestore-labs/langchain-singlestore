"""Unit tests for langchain_singlestore.embeddings module."""

import unittest
from unittest.mock import MagicMock

import pytest
from sqlalchemy.pool import Pool

from langchain_singlestore.embeddings import SingleStoreEmbeddings


def _make_embeddings(**kwargs: object) -> SingleStoreEmbeddings:
    """Build a SingleStoreEmbeddings backed by a mock pool.

    Passing ``connection_pool`` short-circuits real connection setup so unit
    tests never touch the network.
    """
    params: dict = {"connection_pool": MagicMock(spec=Pool)}
    params.update(kwargs)
    return SingleStoreEmbeddings(**params)  # type: ignore[arg-type]


class TestSanitizeFunctionName(unittest.TestCase):
    """Tests for :meth:`SingleStoreEmbeddings._sanitize_function_name`."""

    def setUp(self) -> None:
        self.embeddings = _make_embeddings()

    # Form 1: plain function_name
    def test_plain_identifier(self) -> None:
        assert self.embeddings._sanitize_function_name("my_func") == "my_func"

    def test_plain_identifier_with_digits_and_underscore(self) -> None:
        assert (
            self.embeddings._sanitize_function_name("_Embed_Text_123")
            == "_Embed_Text_123"
        )

    # Form 2: backtick-quoted function_name
    def test_backticked_identifier(self) -> None:
        assert self.embeddings._sanitize_function_name("`my func`") == "`my func`"

    def test_backticked_identifier_with_special_chars(self) -> None:
        assert (
            self.embeddings._sanitize_function_name("`weird-name!@#`")
            == "`weird-name!@#`"
        )

    def test_backticked_identifier_with_escaped_backtick(self) -> None:
        assert (
            self.embeddings._sanitize_function_name("`weird``name`") == "`weird``name`"
        )

    # Form 3: database_name.function_name (both plain)
    def test_qualified_plain_plain(self) -> None:
        assert (
            self.embeddings._sanitize_function_name("db_name.func_name")
            == "db_name.func_name"
        )

    def test_qualified_default_cluster_embed_text(self) -> None:
        assert (
            self.embeddings._sanitize_function_name("cluster.EMBED_TEXT")
            == "cluster.EMBED_TEXT"
        )

    # Form 4: `database_name`.function_name
    def test_qualified_backticked_db_plain_fn(self) -> None:
        assert (
            self.embeddings._sanitize_function_name("`my db`.func_name")
            == "`my db`.func_name"
        )

    # Bonus: plain db + backticked fn, and both backticked
    def test_qualified_plain_db_backticked_fn(self) -> None:
        assert (
            self.embeddings._sanitize_function_name("db_name.`my func`")
            == "db_name.`my func`"
        )

    def test_qualified_backticked_db_backticked_fn(self) -> None:
        assert (
            self.embeddings._sanitize_function_name("`my db`.`my func`")
            == "`my db`.`my func`"
        )

    # Whitespace handling
    def test_leading_and_trailing_whitespace_is_stripped(self) -> None:
        assert self.embeddings._sanitize_function_name("  my_func  ") == "my_func"

    def test_whitespace_around_dot_is_stripped(self) -> None:
        assert (
            self.embeddings._sanitize_function_name("db_name . func_name")
            == "db_name.func_name"
        )

    # Invalid inputs
    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            self.embeddings._sanitize_function_name("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError):
            self.embeddings._sanitize_function_name("   ")

    def test_leading_digit_raises(self) -> None:
        with pytest.raises(ValueError):
            self.embeddings._sanitize_function_name("1func")

    def test_special_chars_without_backticks_raises(self) -> None:
        with pytest.raises(ValueError):
            self.embeddings._sanitize_function_name("my-func")

    def test_dot_only_raises(self) -> None:
        with pytest.raises(ValueError):
            self.embeddings._sanitize_function_name(".")

    def test_trailing_dot_raises(self) -> None:
        with pytest.raises(ValueError):
            self.embeddings._sanitize_function_name("db_name.")

    def test_leading_dot_raises(self) -> None:
        with pytest.raises(ValueError):
            self.embeddings._sanitize_function_name(".func_name")

    def test_more_than_two_parts_raises(self) -> None:
        with pytest.raises(ValueError):
            self.embeddings._sanitize_function_name("a.b.c")

    def test_unclosed_backtick_raises(self) -> None:
        with pytest.raises(ValueError):
            self.embeddings._sanitize_function_name("`unclosed")

    def test_empty_backticked_identifier_raises(self) -> None:
        with pytest.raises(ValueError):
            self.embeddings._sanitize_function_name("``")

    def test_sql_injection_attempt_raises(self) -> None:
        with pytest.raises(ValueError):
            self.embeddings._sanitize_function_name("func; DROP TABLE users;--")

    def test_error_message_includes_input(self) -> None:
        with pytest.raises(ValueError, match="my-func"):
            self.embeddings._sanitize_function_name("my-func")


class TestSingleStoreEmbeddingsInit(unittest.TestCase):
    """Tests that verify ``function_name`` is sanitized during ``__init__``."""

    def test_default_function_name_is_normalized(self) -> None:
        embeddings = _make_embeddings()
        assert embeddings.function_name == "cluster.EMBED_TEXT"

    def test_custom_function_name_is_sanitized(self) -> None:
        embeddings = _make_embeddings(function_name="  my_db . my_func ")
        assert embeddings.function_name == "my_db.my_func"

    def test_invalid_function_name_raises(self) -> None:
        with pytest.raises(ValueError):
            _make_embeddings(function_name="bad-name")


if __name__ == "__main__":
    unittest.main()
