"""
Database Client for Cosmic Insights.

Manages Azure SQL connections using pyodbc with proper connection lifecycle,
parameterised query execution, and a health-check helper.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

import pyodbc

from cosmic_insights import config

logger = logging.getLogger(__name__)


class DatabaseClient:
    """Azure SQL connection manager.

    Provides :meth:`execute_query` for read-only queries and
    :meth:`health_check` to verify connectivity.

    Connections are opened lazily and re-used until the client is
    explicitly closed or until a connection error forces a reconnect.
    """

    def __init__(self) -> None:
        self._connection: Optional[pyodbc.Connection] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute a SQL query and return all rows as a list of dicts.

        Args:
            sql: A validated SELECT query string.

        Returns:
            List of row dictionaries keyed by column name.

        Raises:
            pyodbc.Error: On database communication errors.
        """
        logger.debug("Executing SQL: %s", sql[:200])
        with self._get_cursor() as cursor:
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            rows: List[Dict[str, Any]] = []
            for row in cursor.fetchmany(config.MAX_SQL_RESULT_ROWS):
                rows.append(dict(zip(columns, row)))
            logger.info("Query returned %d row(s).", len(rows))
            return rows

    def health_check(self) -> bool:
        """Verify that a database connection can be established.

        Returns:
            ``True`` if the connection succeeds, ``False`` otherwise.
        """
        try:
            with self._get_cursor() as cursor:
                cursor.execute("SELECT 1 AS health")
                result = cursor.fetchone()
                return result is not None and result[0] == 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Database health check failed: %s", exc)
            return False

    def close(self) -> None:
        """Release the underlying pyodbc connection if it is open."""
        if self._connection is not None:
            try:
                self._connection.close()
                logger.debug("Database connection closed.")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error while closing DB connection: %s", exc)
            finally:
                self._connection = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_connection(self) -> pyodbc.Connection:
        """Return an open pyodbc connection, creating one if necessary.

        Returns:
            An active :class:`pyodbc.Connection` instance.
        """
        if self._connection is None:
            self._connection = self._create_connection()
        return self._connection

    def _create_connection(self) -> pyodbc.Connection:
        """Build and return a new pyodbc connection.

        Returns:
            A fresh :class:`pyodbc.Connection`.

        Raises:
            pyodbc.Error: If the connection cannot be established.
        """
        connection_string = (
            f"DRIVER={config.SQL_DRIVER};"
            f"SERVER={config.SQL_SERVER},{config.SQL_PORT};"
            f"DATABASE={config.SQL_DATABASE};"
            f"UID={config.SQL_USERNAME};"
            f"PWD={config.SQL_PASSWORD};"
            "Encrypt=yes;"
            "TrustServerCertificate=no;"
            f"Connection Timeout={config.SQL_CONNECTION_TIMEOUT};"
        )
        logger.debug("Opening new database connection to %s.", config.SQL_SERVER)
        conn = pyodbc.connect(connection_string, timeout=config.SQL_CONNECTION_TIMEOUT)
        conn.timeout = config.SQL_QUERY_TIMEOUT
        return conn

    @contextmanager
    def _get_cursor(self) -> Generator[pyodbc.Cursor, None, None]:
        """Context manager that yields a cursor and handles cleanup.

        If the underlying connection drops, it is reset so the next call
        will establish a fresh connection.

        Yields:
            An open :class:`pyodbc.Cursor`.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
        except pyodbc.Error:
            # Force reconnect on next call.
            self._connection = None
            raise
        finally:
            try:
                cursor.close()
            except Exception:  # noqa: BLE001
                pass
