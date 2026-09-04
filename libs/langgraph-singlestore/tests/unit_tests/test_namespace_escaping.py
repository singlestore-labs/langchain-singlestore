"""Unit tests for the namespace <-> prefix-text serialization helpers.

Ensures that ``_namespace_to_text`` escapes the separator so that distinct
namespaces never collide in the ``store.prefix`` column, and that the
roundtrip through ``_text_to_namespace`` recovers the original tuple.
"""

import re

import pytest

from langgraph.store.singlestore.base import (
    _namespace_for_prefix_search,
    _namespace_for_suffix_search,
    _namespace_to_text,
    _text_to_namespace,
)


@pytest.mark.parametrize(
    "namespace",
    [
        (),
        ("users",),
        ("users", "alice"),
        ("users", "alice", "prefs"),
        # Parts containing the separator itself.
        ("a/b",),
        ("a/b", "c"),
        ("a", "b/c"),
        ("a/b", "c/d"),
        # Parts containing the escape character.
        ("a\\b",),
        ("a\\", "b"),
        ("\\", "\\/"),
        # Mix of separator and escape character.
        ("a\\/b", "c"),
        ("a/b\\c", "d/e"),
        # Parts containing the LIKE wildcard ``%``.
        ("100%",),
        ("a%b", "c"),
        ("a", "b%"),
        ("%",),
        ("a%/b", "c%d"),
        # Parts containing the LIKE single-char wildcard ``_``.
        ("user_1",),
        ("a_b", "c"),
        ("_",),
        ("a_%b", "c_/d"),
        # Empty string components are preserved.
        ("", "x"),
        ("x", ""),
        # Unicode passes through untouched.
        ("użytkownik", "小明"),
    ],
)
def test_roundtrip_preserves_namespace(namespace: tuple[str, ...]) -> None:
    text = _namespace_to_text(namespace)
    assert _text_to_namespace(text) == namespace


@pytest.mark.parametrize(
    "left,right",
    [
        # The classic collision the escape scheme must prevent:
        # ``("a/b", "c")`` and ``("a", "b", "c")`` must serialize distinctly.
        (("a/b", "c"), ("a", "b", "c")),
        (("a", "b/c"), ("a", "b", "c")),
        (("a/b/c",), ("a", "b", "c")),
        (("a/b",), ("a", "b")),
        # Escape-char collisions.
        (("a\\", "b"), ("a", "\\b")),
    ],
)
def test_distinct_namespaces_serialize_distinctly(
    left: tuple[str, ...], right: tuple[str, ...]
) -> None:
    assert _namespace_to_text(left) != _namespace_to_text(right)


def test_empty_namespace_serializes_to_empty_string() -> None:
    assert _namespace_to_text(()) == ""
    assert _text_to_namespace("") == ()


def test_separator_in_part_is_escaped() -> None:
    # ``/`` inside a part must be escaped so joining stays unambiguous.
    assert _namespace_to_text(("a/b",)) == "a\\/b"
    assert _namespace_to_text(("a/b", "c")) == "a\\/b/c"


def test_escape_char_in_part_is_escaped() -> None:
    # A literal backslash in a part must be doubled.
    assert _namespace_to_text(("a\\b",)) == "a\\\\b"


def test_percent_wildcard_in_part_is_escaped() -> None:
    # ``%`` is escaped so the encoded text is safe as a LIKE pattern literal.
    assert _namespace_to_text(("100%",)) == "100\\%"
    assert _namespace_to_text(("a%b", "c")) == "a\\%b/c"


def test_underscore_wildcard_in_part_is_escaped() -> None:
    # ``_`` is the LIKE single-char wildcard; escape it for the same reason.
    assert _namespace_to_text(("user_1",)) == "user\\_1"
    assert _namespace_to_text(("a_b", "c")) == "a\\_b/c"


def test_text_to_namespace_handles_trailing_escape() -> None:
    # Never produced by ``_namespace_to_text``, but the decoder must not crash
    # on malformed input; a lone trailing backslash is passed through.
    assert _text_to_namespace("a\\") == ("a\\",)


# --------------------------------------------------------------------- LIKE
# The search-pattern helpers below produce MySQL/SingleStore LIKE patterns. The
# helper ``_like_matches`` reimplements LIKE semantics (default ``\`` escape)
# in Python so we can assert that a given pattern matches / does not match a
# concrete stored ``prefix`` value.


def _like_matches(pattern: str, text: str) -> bool:
    """Return ``True`` iff ``pattern`` matches ``text`` under MySQL LIKE rules."""
    regex_parts: list[str] = []
    i = 0
    n = len(pattern)
    while i < n:
        ch = pattern[i]
        if ch == "\\" and i + 1 < n:
            regex_parts.append(re.escape(pattern[i + 1]))
            i += 2
        elif ch == "%":
            regex_parts.append(".*")
            i += 1
        elif ch == "_":
            regex_parts.append(".")
            i += 1
        else:
            regex_parts.append(re.escape(ch))
            i += 1
    return re.fullmatch("".join(regex_parts), text, re.DOTALL) is not None


class TestNamespaceForPrefixSearch:
    @pytest.mark.parametrize(
        "namespace,expected",
        [
            (("users", "alice"), "users/alice/%"),
            (("users",), "users/%"),
            # ``*`` in the tuple becomes a LIKE wildcard segment.
            (("users", "*"), "users/%/%"),
            (("*", "alice"), "%/alice/%"),
            (("*",), "%/%"),
            # Separators and LIKE metacharacters inside a real part are escaped.
            (("a/b", "c"), "a\\/b/c/%"),
            (("100%",), "100\\%/%"),
            (("user_1",), "user\\_1/%"),
            (("a\\b",), "a\\\\b/%"),
        ],
    )
    def test_pattern_shape(self, namespace: tuple[str, ...], expected: str) -> None:
        assert _namespace_for_prefix_search(namespace) == expected

    def test_matches_descendants_but_not_the_prefix_itself(self) -> None:
        # ``list_namespaces(prefix=("users", "alice"))`` returns descendants of
        # ``users/alice``, not ``users/alice`` itself — the trailing ``/%``
        # requires at least one more segment.
        pattern = _namespace_for_prefix_search(("users", "alice"))
        assert _like_matches(pattern, _namespace_to_text(("users", "alice", "prefs")))
        assert _like_matches(
            pattern, _namespace_to_text(("users", "alice", "prefs", "theme"))
        )
        assert not _like_matches(pattern, _namespace_to_text(("users", "alice")))
        assert not _like_matches(pattern, _namespace_to_text(("users", "bob")))
        assert not _like_matches(pattern, _namespace_to_text(("users",)))

    def test_wildcard_segment_matches_any_single_component(self) -> None:
        # ``("users", "*")`` -> matches any user + at least one child.
        pattern = _namespace_for_prefix_search(("users", "*"))
        assert _like_matches(pattern, _namespace_to_text(("users", "alice", "prefs")))
        assert _like_matches(pattern, _namespace_to_text(("users", "bob", "prefs")))
        # Sibling roots do not match.
        assert not _like_matches(pattern, _namespace_to_text(("agents", "x", "y")))


class TestNamespaceForSuffixSearch:
    @pytest.mark.parametrize(
        "namespace,expected",
        [
            (("prefs",), "%/prefs"),
            (("alice", "prefs"), "%/alice/prefs"),
            # ``*`` becomes a LIKE wildcard segment.
            (("*", "prefs"), "%/%/prefs"),
            (("alice", "*"), "%/alice/%"),
            (("*",), "%/%"),
            # Separators and LIKE metacharacters inside a real part are escaped.
            (("a/b",), "%/a\\/b"),
            (("100%",), "%/100\\%"),
            (("user_1",), "%/user\\_1"),
            (("a\\b",), "%/a\\\\b"),
        ],
    )
    def test_pattern_shape(self, namespace: tuple[str, ...], expected: str) -> None:
        assert _namespace_for_suffix_search(namespace) == expected

    def test_matches_any_namespace_ending_in_suffix(self) -> None:
        # ``list_namespaces(suffix=("prefs",))`` returns namespaces that end
        # in ``.../prefs`` — the leading ``%/`` requires at least one earlier
        # segment (so a bare ``("prefs",)`` does not match).
        pattern = _namespace_for_suffix_search(("prefs",))
        assert _like_matches(pattern, _namespace_to_text(("users", "alice", "prefs")))
        assert _like_matches(pattern, _namespace_to_text(("agents", "prefs")))
        assert not _like_matches(pattern, _namespace_to_text(("prefs",)))
        assert not _like_matches(pattern, _namespace_to_text(("users", "alice")))

    def test_wildcard_segment_matches_any_single_component(self) -> None:
        # ``("*", "prefs")`` -> pattern ``%/%/prefs`` requires >= 2 separators
        # in the target, i.e. namespaces of depth >= 3 ending in ``prefs``.
        pattern = _namespace_for_suffix_search(("*", "prefs"))
        assert _like_matches(pattern, _namespace_to_text(("users", "alice", "prefs")))
        assert _like_matches(pattern, _namespace_to_text(("agents", "x", "prefs")))
        # Depth-2 target has only one ``/`` — not enough separators.
        assert not _like_matches(pattern, _namespace_to_text(("users", "prefs")))
