"""Shared fixtures for langgraph-singlestore integration tests.

Boots a SingleStore server in Docker for the test session, and provides a
clean test database on demand.
"""

from typing import Generator
from urllib.parse import urlparse

import pytest
from singlestoredb import connect
from singlestoredb.server import docker

TEST_DB_NAME = "test_langgraph_singlestore"


class ConnectionParameters:
    """Parsed connection parameters targeting the test database."""

    host: str
    port: int
    user: str
    password: str
    database: str

    def __init__(self, connection_url: str) -> None:
        parsed = urlparse(connection_url)
        self.host = parsed.hostname or "localhost"
        self.port = parsed.port or 3306
        self.user = parsed.username or ""
        self.password = parsed.password or ""
        self.database = TEST_DB_NAME

    def as_kwargs(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "database": self.database,
        }


@pytest.fixture(scope="session")
def docker_server_url() -> Generator[str, None, None]:
    """Start a SingleStore Docker server for the test session."""
    sdb = docker.start(license="")
    conn = sdb.connect()
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE {TEST_DB_NAME}")
    cur.close()
    conn.close()
    yield sdb.connection_url
    sdb.stop()


@pytest.fixture(scope="function")
def clean_db_url(docker_server_url: str) -> Generator[str, None, None]:
    """Provide the docker URL and drop all tables in ``TEST_DB_NAME`` after the test."""
    yield docker_server_url
    conn = connect(host=docker_server_url, database=TEST_DB_NAME)
    cur = conn.cursor()
    cur.execute("SHOW TABLES")
    for row in cur.fetchall():
        cur.execute(f"DROP TABLE {list(row)[0]}")
    cur.close()
    conn.close()


@pytest.fixture(scope="function")
def connection_parameters(
    clean_db_url: str,
) -> Generator[ConnectionParameters, None, None]:
    """Parsed connection parameters against a freshly cleaned database."""
    yield ConnectionParameters(clean_db_url)
