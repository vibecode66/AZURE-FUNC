"""
Semantic Mapping Layer for Cosmic Insights.

Maps business-friendly terms (synonyms, abbreviations) to the actual database
column/table names so the LLM receives more precise context.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default mapping dictionary.
# Keys are business-friendly terms (lower-case); values are the actual
# database identifiers they map to.  Extend or override this dictionary at
# runtime by calling :meth:`SemanticMapper.add_mapping`.
# ---------------------------------------------------------------------------
DEFAULT_MAPPINGS: Dict[str, str] = {
    # Financial / sales terms
    "revenue": "total_amount",
    "sales": "total_amount",
    "income": "total_amount",
    "earnings": "total_amount",
    "profit": "net_profit",
    "cost": "total_cost",
    "expense": "total_cost",
    "spend": "total_cost",
    # Customer / entity terms
    "customers": "client_table",
    "clients": "client_table",
    "users": "user_table",
    "accounts": "account_table",
    # Time terms
    "this year": "YEAR(date_column) = YEAR(GETDATE())",
    "last year": "YEAR(date_column) = YEAR(GETDATE()) - 1",
    "this month": "MONTH(date_column) = MONTH(GETDATE()) AND YEAR(date_column) = YEAR(GETDATE())",
    "ytd": "date_column >= DATEFROMPARTS(YEAR(GETDATE()), 1, 1)",
    "year to date": "date_column >= DATEFROMPARTS(YEAR(GETDATE()), 1, 1)",
    # Common abbreviations
    "qty": "quantity",
    "amt": "amount",
    "avg": "average",
    "num": "number",
    "id": "identifier",
}


class SemanticMapper:
    """Applies semantic mappings to enrich natural-language questions.

    Attributes:
        _mappings: Dictionary of term → database identifier mappings.
    """

    def __init__(self, extra_mappings: Dict[str, str] | None = None) -> None:
        self._mappings: Dict[str, str] = dict(DEFAULT_MAPPINGS)
        if extra_mappings:
            self._mappings.update(extra_mappings)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_mapping(self, term: str, db_identifier: str) -> None:
        """Register a new or override an existing semantic mapping.

        Args:
            term: Business-friendly term (case-insensitive).
            db_identifier: Actual database column/table name or SQL fragment.
        """
        self._mappings[term.lower()] = db_identifier
        logger.debug("Mapping added: %r → %r", term, db_identifier)

    def enhance_question(self, question: str) -> str:
        """Enrich a natural-language question with database-specific context.

        Appends a *Semantic hints* section to the question so the LLM can
        generate more accurate SQL.

        Args:
            question: The original user question.

        Returns:
            The question with appended semantic hints (if any were matched).
        """
        hints: List[str] = []
        lower_q = question.lower()

        for term, db_id in self._mappings.items():
            # Whole-word / whole-phrase match (case-insensitive).
            escaped = re.escape(term)
            if re.search(rf"\b{escaped}\b", lower_q):
                hints.append(f"'{term}' refers to '{db_id}'")

        if hints:
            hint_block = "; ".join(hints)
            enhanced = f"{question}\n[Semantic hints: {hint_block}]"
            logger.debug("Question enhanced with %d hint(s).", len(hints))
            return enhanced

        return question

    def get_all_mappings(self) -> Dict[str, str]:
        """Return a copy of the current mapping dictionary.

        Returns:
            Dictionary of term → database identifier mappings.
        """
        return dict(self._mappings)
