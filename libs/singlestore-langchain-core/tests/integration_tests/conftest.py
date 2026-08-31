"""Shared fixtures for _connection integration tests."""

from typing import Generator
from urllib.parse import urlparse

import pytest
from singlestoredb import connect
from singlestoredb.server import docker

TEST_DB_NAME = "test_singlestore_langchain_core"


class ConnectionParameters:
    """Parsed connection parameters for the SingleStore docker instance."""

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
    cur.execute(f"create database {TEST_DB_NAME}")
    cur.close()
    conn.close()
    yield sdb.connection_url
    sdb.stop()


@pytest.fixture(scope="function")
def connection_parameters(
    docker_server_url: str,
) -> Generator[ConnectionParameters, None, None]:
    """Provide parsed connection parameters targeting the test database."""
    yield ConnectionParameters(docker_server_url)


@pytest.fixture(scope="function")
def raw_connection(
    connection_parameters: ConnectionParameters,
) -> Generator[object, None, None]:
    """Provide a raw singlestoredb connection to the test database."""
    conn = connect(**connection_parameters.as_kwargs())
    yield conn
    conn.close()
