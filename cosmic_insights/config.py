"""
Configuration module for Cosmic Insights.

All configurable parameters are read from environment variables with sensible defaults.
No secrets are hardcoded here.
"""

import os
import logging
from typing import List

# ---------------------------------------------------------------------------
# Azure SQL Configuration
# ---------------------------------------------------------------------------
SQL_SERVER: str = os.getenv("SQL_SERVER", "your-server.database.windows.net")
SQL_DATABASE: str = os.getenv("SQL_DATABASE", "your-database")
SQL_USERNAME: str = os.getenv("SQL_USERNAME", "your-username")
SQL_PASSWORD: str = os.getenv("SQL_PASSWORD", "")
SQL_DRIVER: str = os.getenv("SQL_DRIVER", "{ODBC Driver 18 for SQL Server}")
SQL_PORT: int = int(os.getenv("SQL_PORT", "1433"))
SQL_CONNECTION_TIMEOUT: int = int(os.getenv("SQL_CONNECTION_TIMEOUT", "30"))
SQL_QUERY_TIMEOUT: int = int(os.getenv("SQL_QUERY_TIMEOUT", "60"))

# ---------------------------------------------------------------------------
# Azure OpenAI Configuration
# ---------------------------------------------------------------------------
OPENAI_ENDPOINT: str = os.getenv("OPENAI_ENDPOINT", "https://your-openai.openai.azure.com/")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_VERSION: str = os.getenv("OPENAI_API_VERSION", "2024-02-01")
OPENAI_DEPLOYMENT_NAME: str = os.getenv("OPENAI_DEPLOYMENT_NAME", "gpt-4o")
OPENAI_MODEL_NAME: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
OPENAI_TOP_P: float = float(os.getenv("OPENAI_TOP_P", "1.0"))

# ---------------------------------------------------------------------------
# Application Settings
# ---------------------------------------------------------------------------
MAX_SQL_RESULT_ROWS: int = int(os.getenv("MAX_SQL_RESULT_ROWS", "500"))

ALLOWED_SQL_OPERATIONS: List[str] = ["SELECT"]

BLOCKED_SQL_KEYWORDS: List[str] = [
    "DROP",
    "DELETE",
    "INSERT",
    "UPDATE",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "EXEC",
    "EXECUTE",
    "GRANT",
    "REVOKE",
]

MAX_SQL_RETRIES: int = int(os.getenv("MAX_SQL_RETRIES", "3"))

# ---------------------------------------------------------------------------
# Schema Settings
# ---------------------------------------------------------------------------
SCHEMA_NAME: str = os.getenv("SCHEMA_NAME", "dbo")

# Comma-separated list of tables to include; empty means all tables.
_TABLES_ENV: str = os.getenv("SCHEMA_TABLES", "")
SCHEMA_TABLES: List[str] = [t.strip() for t in _TABLES_ENV.split(",") if t.strip()] if _TABLES_ENV else []

# Comma-separated list of column names to exclude from the schema context.
_EXCLUDED_COLS_ENV: str = os.getenv("EXCLUDED_COLUMNS", "")
EXCLUDED_COLUMNS: List[str] = [c.strip() for c in _EXCLUDED_COLS_ENV.split(",") if c.strip()] if _EXCLUDED_COLS_ENV else []

SCHEMA_CACHE_TTL_SECONDS: int = int(os.getenv("SCHEMA_CACHE_TTL_SECONDS", "3600"))

# ---------------------------------------------------------------------------
# Visualization Settings
# ---------------------------------------------------------------------------
CHART_WIDTH: int = int(os.getenv("CHART_WIDTH", "12"))
CHART_HEIGHT: int = int(os.getenv("CHART_HEIGHT", "6"))
CHART_DPI: int = int(os.getenv("CHART_DPI", "100"))
CHART_TITLE_FONT_SIZE: int = int(os.getenv("CHART_TITLE_FONT_SIZE", "14"))
CHART_LABEL_FONT_SIZE: int = int(os.getenv("CHART_LABEL_FONT_SIZE", "11"))
CHART_TICK_FONT_SIZE: int = int(os.getenv("CHART_TICK_FONT_SIZE", "9"))

# Comma-separated hex colors for the default palette.
_PALETTE_ENV: str = os.getenv(
    "CHART_COLOR_PALETTE",
    "#2196F3,#4CAF50,#FF9800,#F44336,#9C27B0,#00BCD4,#FF5722,#607D8B",
)
CHART_COLOR_PALETTE: List[str] = [c.strip() for c in _PALETTE_ENV.split(",") if c.strip()]

PIE_CHART_MAX_CATEGORIES: int = int(os.getenv("PIE_CHART_MAX_CATEGORIES", "7"))

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = os.getenv(
    "LOG_FORMAT",
    "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format=LOG_FORMAT)
