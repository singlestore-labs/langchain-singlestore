"""
Shared fixtures for integration tests.

This module provides fixtures for setting up test environments,
database connections, and index instances used across multiple test files.
"""

from typing import Generator
from urllib.parse import urlparse

import pytest
from singlestoredb import connect
from singlestoredb.server import docker

TEST_DB_NAME = "test_langchain_singlestore"

class ConnectionParameters:
    """Class to hold connection parameters for tests."""
    Host: str
    Port: int
    User: str
    Password: str
    Database: str

    def __init__(self, connection_url: str):
        """Parse connection parameters from connection_url.
        
        Args:
            connection_url: Connection URL in format scheme://user:password@host:port/database
        """
        parsed = urlparse(connection_url)
        self.Host = parsed.hostname or "localhost"
        self.Port = parsed.port or 3306
        self.User = parsed.username or ""
        self.Password = parsed.password or ""
        self.Database = TEST_DB_NAME



@pytest.fixture(scope="session")
def docker_server_url() -> Generator[str, None, None]:
    """Start a SingleStore Docker server for tests."""
    sdb = docker.start(license="")
    conn = sdb.connect()
    curr = conn.cursor()
    curr.execute(f"create database {TEST_DB_NAME}")
    curr.close()
    conn.close()
    yield sdb.connection_url
    sdb.stop()


@pytest.fixture(scope="function")
def clean_db_url(docker_server_url: str) -> Generator[str, None, None]:
    """Provide a clean database URL and clean up tables after test."""
    yield docker_server_url
    conn = connect(host=docker_server_url, database=TEST_DB_NAME)
    curr = conn.cursor()
    curr.execute("show tables")
    results = curr.fetchall()
    for result in results:
        curr.execute(f"drop table {result[0]}")
    curr.close()
    conn.close()

@pytest.fixture(scope="function")
def clean_db_connection_parameters(clean_db_url: str) -> Generator[ConnectionParameters, None, None]:
    """Provide a clean database connection and clean up tables after test."""
    yield ConnectionParameters(connection_url=clean_db_url)
