"""
Schema Processing Layer for Cosmic Insights.

Dynamically reads the Azure SQL database schema (tables, columns, data types,
nullable, primary keys, foreign keys), caches it with TTL-based expiration,
and formats it into a prompt-friendly string for the LLM.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from cosmic_insights import config

logger = logging.getLogger(__name__)


class SchemaProcessor:
    """Reads and caches the Azure SQL database schema.

    Attributes:
        _cache: Cached list of table metadata dictionaries.
        _cache_timestamp: Unix timestamp when the cache was last populated.
    """

    def __init__(self) -> None:
        self._cache: Optional[List[Dict[str, Any]]] = None
        self._cache_timestamp: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_schema(self, db_client: Any, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Return the schema, using the cache when it is still fresh.

        Args:
            db_client: An instance of :class:`~cosmic_insights.database_client.DatabaseClient`.
            force_refresh: When ``True``, bypass the cache and re-read from the database.

        Returns:
            A list of table metadata dictionaries.
        """
        if force_refresh or self._is_cache_stale():
            logger.info("Refreshing schema cache from database.")
            self._cache = self._fetch_schema(db_client)
            self._cache_timestamp = time.time()
        else:
            logger.debug("Returning cached schema (age %.0fs).", time.time() - self._cache_timestamp)

        return self._cache or []

    def format_schema_for_prompt(self, schema: List[Dict[str, Any]]) -> str:
        """Convert the schema list into a human-readable string for the LLM prompt.

        Args:
            schema: List of table metadata dictionaries as returned by :meth:`get_schema`.

        Returns:
            A formatted string representation of the schema.
        """
        if not schema:
            return "No schema information available."

        lines: List[str] = ["Database Schema:\n"]
        for table in schema:
            table_name: str = table.get("table_name", "unknown")
            lines.append(f"Table: {config.SCHEMA_NAME}.{table_name}")
            for col in table.get("columns", []):
                nullable = "NULL" if col.get("is_nullable") else "NOT NULL"
                pk_marker = " [PK]" if col.get("is_primary_key") else ""
                fk_info = ""
                if col.get("foreign_key_table"):
                    fk_info = f" [FK → {col['foreign_key_table']}.{col['foreign_key_column']}]"
                lines.append(
                    f"  - {col['column_name']} ({col['data_type']}, {nullable}){pk_marker}{fk_info}"
                )
            lines.append("")  # blank line between tables

        return "\n".join(lines)

    def invalidate_cache(self) -> None:
        """Force the next :meth:`get_schema` call to re-read from the database."""
        self._cache = None
        self._cache_timestamp = 0.0
        logger.info("Schema cache invalidated.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_cache_stale(self) -> bool:
        """Return ``True`` if the cache is absent or has expired."""
        if self._cache is None:
            return True
        age = time.time() - self._cache_timestamp
        return age > config.SCHEMA_CACHE_TTL_SECONDS

    def _fetch_schema(self, db_client: Any) -> List[Dict[str, Any]]:
        """Read schema information from the database via *db_client*.

        Args:
            db_client: A connected :class:`~cosmic_insights.database_client.DatabaseClient`.

        Returns:
            List of table metadata dictionaries.
        """
        # --- Determine which tables to inspect ---
        if config.SCHEMA_TABLES:
            table_filter = "AND t.TABLE_NAME IN ({})".format(
                ", ".join(f"'{t}'" for t in config.SCHEMA_TABLES)
            )
        else:
            table_filter = ""

        column_sql = f"""
            SELECT
                t.TABLE_NAME,
                c.COLUMN_NAME,
                c.DATA_TYPE,
                c.IS_NULLABLE,
                c.CHARACTER_MAXIMUM_LENGTH,
                c.NUMERIC_PRECISION,
                c.NUMERIC_SCALE,
                c.ORDINAL_POSITION
            FROM INFORMATION_SCHEMA.TABLES t
            JOIN INFORMATION_SCHEMA.COLUMNS c
                ON t.TABLE_NAME = c.TABLE_NAME
                AND t.TABLE_SCHEMA = c.TABLE_SCHEMA
            WHERE t.TABLE_TYPE = 'BASE TABLE'
              AND t.TABLE_SCHEMA = '{config.SCHEMA_NAME}'
              {table_filter}
            ORDER BY t.TABLE_NAME, c.ORDINAL_POSITION
        """

        pk_sql = f"""
            SELECT
                kcu.TABLE_NAME,
                kcu.COLUMN_NAME
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
                ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
                AND tc.TABLE_SCHEMA = kcu.TABLE_SCHEMA
            WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
              AND tc.TABLE_SCHEMA = '{config.SCHEMA_NAME}'
        """

        fk_sql = f"""
            SELECT
                kcu.TABLE_NAME,
                kcu.COLUMN_NAME,
                ccu.TABLE_NAME  AS REFERENCED_TABLE,
                ccu.COLUMN_NAME AS REFERENCED_COLUMN
            FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
                ON rc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
            JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE ccu
                ON rc.UNIQUE_CONSTRAINT_NAME = ccu.CONSTRAINT_NAME
            WHERE kcu.TABLE_SCHEMA = '{config.SCHEMA_NAME}'
        """

        columns_rows = db_client.execute_query(column_sql)
        pk_rows = db_client.execute_query(pk_sql)
        fk_rows = db_client.execute_query(fk_sql)

        # Build lookup sets / dicts for PKs and FKs.
        pk_set = {(r["TABLE_NAME"], r["COLUMN_NAME"]) for r in pk_rows}
        fk_map: Dict[tuple, Dict[str, str]] = {
            (r["TABLE_NAME"], r["COLUMN_NAME"]): {
                "foreign_key_table": r["REFERENCED_TABLE"],
                "foreign_key_column": r["REFERENCED_COLUMN"],
            }
            for r in fk_rows
        }

        # Group columns by table.
        tables: Dict[str, Dict[str, Any]] = {}
        for row in columns_rows:
            tname = row["TABLE_NAME"]
            cname = row["COLUMN_NAME"]

            if cname in config.EXCLUDED_COLUMNS:
                continue

            if tname not in tables:
                tables[tname] = {"table_name": tname, "columns": []}

            col_entry: Dict[str, Any] = {
                "column_name": cname,
                "data_type": row["DATA_TYPE"],
                "is_nullable": row["IS_NULLABLE"] == "YES",
                "is_primary_key": (tname, cname) in pk_set,
                "foreign_key_table": fk_map.get((tname, cname), {}).get("foreign_key_table"),
                "foreign_key_column": fk_map.get((tname, cname), {}).get("foreign_key_column"),
            }
            tables[tname]["columns"].append(col_entry)

        return list(tables.values())
