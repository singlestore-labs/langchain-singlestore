"""SingleStore embeddings integration."""

import re
import struct
from typing import Any, List, Optional

from langchain_core.embeddings import Embeddings
from singlestore_langchain_core import create_connection_pool
from singlestoredb.connection import Connection
from sqlalchemy.pool import Pool

from langchain_singlestore._utils import set_connector_attributes

# Matches an optionally database-qualified SingleStore identifier where each
# part is either an unquoted identifier or a backtick-quoted identifier
# (with `` used to escape a literal backtick inside the quoted form).
_IDENT = r"(?:`(?:[^`]|``)+`|[A-Za-z_][A-Za-z0-9_]*)"
_FUNCTION_NAME_RE = re.compile(
    rf"^\s*(?:(?P<db>{_IDENT})\s*\.\s*)?(?P<fn>{_IDENT})\s*$"
)


class SingleStoreEmbeddings(Embeddings):
    """SingleStore embedding model integration."""

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        function_name: str = "cluster.EMBED_TEXT",
        connection: Optional[Connection] = None,
        connection_pool: Optional[Pool] = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        timeout: float = 30,
        **connection_kwargs: Any,
    ) -> None:
        """Initialize with necessary components.

        Calls the specified SingleStore database function for embedding.
        Default execution intends that user has set up AI capabilities within
        SingleStore Managed Service: https://docs.singlestore.com/cloud/ai/ai-ml-functions/

        Args:
            model (str, optional): The embedding model to use. Defaults to None.

            function_name (str, optional): The name of the database function to call
                for embedding. Defaults to "cluster.EMBED_TEXT".

            Following arguments pertain to the database connection or connection pool:

            connection (singlestoredb.Connection, optional): An existing
                caller-owned SingleStoreDB connection. When supplied, every
                database operation shares this connection through an internal
                proxy that never closes it; the caller keeps full ownership of
                the connection lifecycle. Mutually exclusive with
                ``connection_pool``.

            connection_pool (sqlalchemy.pool.Pool, optional): A pre-built
                SQLAlchemy connection pool to use as-is. Useful when the
                surrounding application already manages its own pool (custom
                pool class, shared pool across components, etc.). Mutually
                exclusive with ``connection``.

                When neither ``connection`` nor ``connection_pool`` is passed,
                a default :class:`QueueConnectionPool` is built from
                ``pool_size``, ``max_overflow``, ``timeout``, and the
                connection kwargs described below.

            Following arguments pertain to the newly created connection pool:

            pool_size (int, optional): Determines the number of active connections in
                the pool. Defaults to 5. Ignored if ``connection`` or
                ``connection_pool`` is supplied.

            max_overflow (int, optional): Determines the maximum number of connections
                allowed beyond the pool_size. Defaults to 10. Ignored if
                ``connection`` or ``connection_pool`` is supplied.

            timeout (float, optional): Specifies the maximum wait time in seconds for
                establishing a connection. Defaults to 30. Ignored if
                ``connection`` or ``connection_pool`` is supplied.


            Following arguments pertain to the database connection:

            host (str, optional): Specifies the hostname, IP address, or URL for the
                database connection. The default scheme is "mysql".

            user (str, optional): Database username.

            password (str, optional): Database password.

            port (int, optional): Database port. Defaults to 3306 for non-HTTP
                connections, 80 for HTTP connections, and 443 for HTTPS connections.

            database (str, optional): Database name.


            Additional optional arguments provide further customization over the
            database connection:

            pure_python (bool, optional): Toggles the connector mode. If True,
                operates in pure Python mode.

            local_infile (bool, optional): Allows local file uploads.

            charset (str, optional): Specifies the character set for string values.

            ssl_key (str, optional): Specifies the path of the file containing the SSL
                key.

            ssl_cert (str, optional): Specifies the path of the file containing the SSL
                certificate.

            ssl_ca (str, optional): Specifies the path of the file containing the SSL
                certificate authority.

            ssl_cipher (str, optional): Sets the SSL cipher list.

            ssl_disabled (bool, optional): Disables SSL usage.

            ssl_verify_cert (bool, optional): Verifies the server's certificate.
                Automatically enabled if ``ssl_ca`` is specified.

            ssl_verify_identity (bool, optional): Verifies the server's identity.

            conv (dict[int, Callable], optional): A dictionary of data conversion
                functions.

            credential_type (str, optional): Specifies the type of authentication to
                use: auth.PASSWORD, auth.JWT, or auth.BROWSER_SSO.

            autocommit (bool, optional): Enables autocommits.

            results_type (str, optional): Determines the structure of the query results:
                tuples, namedtuples, dicts.

            results_format (str, optional): Deprecated. This option has been renamed to
                results_type.
        """
        self.model = model
        self.function_name = self._sanitize_function_name(function_name)
        self.connection_kwargs = connection_kwargs

        # Add connection attributes to the connection kwargs.
        set_connector_attributes(self.connection_kwargs)

        # Create connection pool.
        self.connection_pool = create_connection_pool(
            connection=connection,
            connection_pool=connection_pool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            timeout=timeout,
            connection_kwargs=self.connection_kwargs,
        )

    def _sanitize_function_name(self, function_name: str) -> str:
        """Validate ``function_name`` and return its normalized form.

        Accepted forms:
            1. ``function_name``
            2. ```` `function name` ````
            3. ``database_name.function_name``
            4. ```` `database_name`.function_name ````

        Either part may independently be a plain identifier
        (``[A-Za-z_][A-Za-z0-9_]*``) or a backtick-quoted identifier.
        """
        match = _FUNCTION_NAME_RE.match(function_name)
        if match is None:
            raise ValueError(
                f"Invalid function name: {function_name!r}. Expected one of: "
                "'function_name', '`function name`', "
                "'database_name.function_name', or "
                "'`database_name`.function_name'."
            )
        db, fn = match.group("db"), match.group("fn")
        return f"{db}.{fn}" if db is not None else str(fn)

    def _embed(self, cur: Any, text: str) -> List[float]:
        """Embed a single text string using the database function."""
        if self.model:
            cur.execute(f"SELECT {self.function_name}(%s, %s)", (text, self.model))
        else:
            cur.execute(f"SELECT {self.function_name}(%s)", (text,))
        row = cur.fetchone()
        if row is None or row[0] is None:
            return []  # Return an empty list if no embedding is found
        raw_bytes: bytes = row[0]
        num_floats = len(raw_bytes) // 4
        return list(struct.unpack(f"<{num_floats}f", raw_bytes))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = []
        conn = self.connection_pool.connect()
        try:
            cur = conn.cursor()
            try:
                for text in texts:
                    result.append(self._embed(cur, text))
            finally:
                cur.close()
        finally:
            conn.close()
        return result

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        conn = self.connection_pool.connect()
        try:
            cur = conn.cursor()
            try:
                return self._embed(cur, text)
            finally:
                cur.close()
        finally:
            conn.close()
