"""
Insight Generation Engine for Cosmic Insights.

Delegates to Azure OpenAI to produce structured, business-friendly
insights from SQL query results.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from cosmic_insights.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class InsightEngine:
    """Generates business-friendly textual insights from query data.

    Attributes:
        _openai_client: The shared :class:`~cosmic_insights.openai_client.OpenAIClient` instance.
    """

    def __init__(self, openai_client: OpenAIClient) -> None:
        self._openai_client = openai_client

    def generate(
        self,
        question: str,
        sql_query: str,
        data: List[Dict[str, Any]],
    ) -> str:
        """Produce insights for the given query results.

        Args:
            question: The original user question.
            sql_query: The executed SQL query.
            data: List of result-row dicts returned from the database.

        Returns:
            A string of bullet-point insights.  Returns a generic message
            when *data* is empty.
        """
        if not data:
            logger.info("No data provided; returning empty-data message.")
            return "The query returned no results for the given question."

        logger.info("Generating insights for %d row(s).", len(data))
        try:
            insights = self._openai_client.generate_insights(question, sql_query, data)
            return insights
        except Exception as exc:  # noqa: BLE001
            logger.warning("Insight generation failed: %s", exc)
            return f"Insight generation is currently unavailable. ({exc})"
