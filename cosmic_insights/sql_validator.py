"""
SQL Validation Layer for Cosmic Insights.

Validates that AI-generated SQL queries are safe, SELECT-only,
and free from injection patterns before execution.
"""

from __future__ import annotations

import logging
import re
from typing import List

from cosmic_insights import config
from cosmic_insights.models import ValidationResult

logger = logging.getLogger(__name__)

# Patterns that signal SQL injection attempts.
_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r";\s*\w", re.IGNORECASE),           # Stacked queries: ; followed by another statement
    re.compile(r"--\s*$", re.MULTILINE),             # Inline comment at end of line
    re.compile(r"/\*.*?\*/", re.DOTALL),             # Block comments
    re.compile(r"\bxp_\w+", re.IGNORECASE),          # Extended stored procedures
    re.compile(r"\bsp_\w+", re.IGNORECASE),          # System stored procedures
    re.compile(r"\bOPENROWSET\b", re.IGNORECASE),    # External data source access
    re.compile(r"\bOPENQUERY\b", re.IGNORECASE),     # Linked server query
    re.compile(r"\bBULK\s+INSERT\b", re.IGNORECASE), # Bulk insert
    re.compile(r"\bINTO\s+OUTFILE\b", re.IGNORECASE),# MySQL file write
    re.compile(r"\bLOAD_FILE\b", re.IGNORECASE),     # MySQL file read
]

# Rough heuristic: reject queries exceeding this character count.
MAX_QUERY_LENGTH: int = 4000


def validate_sql(sql: str) -> ValidationResult:
    """Validate a SQL query for safety and correctness.

    Checks:
    - Query starts with an allowed operation (SELECT only).
    - No blocked keywords present.
    - No SQL injection patterns detected.
    - Query length is within threshold.

    Args:
        sql: The SQL query string to validate.

    Returns:
        A :class:`ValidationResult` indicating success or listing all errors.
    """
    errors: List[str] = []

    if not sql or not sql.strip():
        return ValidationResult(is_valid=False, errors=["SQL query is empty."])

    stripped = sql.strip()

    # 1. Must start with an allowed operation.
    first_token = stripped.split()[0].upper()
    if first_token not in config.ALLOWED_SQL_OPERATIONS:
        errors.append(
            f"Only {config.ALLOWED_SQL_OPERATIONS} queries are permitted. "
            f"Received query starting with '{first_token}'."
        )

    # 2. Check for blocked keywords (whole-word matches).
    upper_sql = stripped.upper()
    for keyword in config.BLOCKED_SQL_KEYWORDS:
        # Use word boundaries to avoid false positives (e.g., "CREATES" ≠ "CREATE").
        pattern = re.compile(rf"\b{re.escape(keyword)}\b")
        if pattern.search(upper_sql):
            errors.append(f"Blocked keyword detected: '{keyword}'.")

    # 3. SQL injection pattern checks.
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(stripped):
            errors.append(f"Potential SQL injection pattern detected: {pattern.pattern!r}.")

    # 4. Query length check.
    if len(stripped) > MAX_QUERY_LENGTH:
        errors.append(
            f"Query length ({len(stripped)} chars) exceeds maximum allowed ({MAX_QUERY_LENGTH} chars)."
        )

    is_valid = len(errors) == 0
    if not is_valid:
        logger.warning("SQL validation failed: %s", errors)
    else:
        logger.debug("SQL validation passed.")

    return ValidationResult(is_valid=is_valid, errors=errors)
