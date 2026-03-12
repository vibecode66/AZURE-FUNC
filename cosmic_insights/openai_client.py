"""
Azure OpenAI Integration for Cosmic Insights.

Handles:
- NL → SQL query generation with safety-aware system prompts.
- SQL correction when the generated query fails validation or execution.
- Business-friendly insight generation from query results.
- Plain-English SQL explanation.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI

from cosmic_insights import config

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Wrapper around the Azure OpenAI API for Cosmic Insights operations.

    Attributes:
        _client: Underlying :class:`openai.AzureOpenAI` client instance.
    """

    def __init__(self) -> None:
        self._client = AzureOpenAI(
            azure_endpoint=config.OPENAI_ENDPOINT,
            api_key=config.OPENAI_API_KEY,
            api_version=config.OPENAI_API_VERSION,
        )

    # ------------------------------------------------------------------
    # NL → SQL
    # ------------------------------------------------------------------

    def generate_sql(self, question: str, schema_context: str) -> str:
        """Generate a SQL query from a natural-language question.

        Args:
            question: The (optionally semantic-enriched) user question.
            schema_context: Formatted schema string from :class:`~cosmic_insights.schema_processor.SchemaProcessor`.

        Returns:
            A SQL query string (without surrounding markdown).

        Raises:
            RuntimeError: If the API call fails.
        """
        system_prompt = self._build_sql_system_prompt(schema_context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        logger.debug("Calling OpenAI for SQL generation.")
        response = self._chat_completion(messages)
        sql = self._extract_sql(response)
        logger.info("Generated SQL: %s", sql[:120])
        return sql

    def correct_sql(
        self,
        original_question: str,
        failed_sql: str,
        error_message: str,
        schema_context: str,
    ) -> str:
        """Ask the LLM to fix a SQL query that failed validation or execution.

        Args:
            original_question: The original user question.
            failed_sql: The SQL that failed.
            error_message: The validation/execution error message.
            schema_context: Formatted schema string.

        Returns:
            A corrected SQL query string.
        """
        system_prompt = self._build_sql_system_prompt(schema_context)

        correction_prompt = (
            f"The following SQL query failed with this error:\n\nError: {error_message}\n\n"
            f"Failed SQL:\n```sql\n{failed_sql}\n```\n\n"
            f"Original question: {original_question}\n\n"
            "Please provide a corrected SQL query that fixes the error. "
            "Return only the corrected SQL query, nothing else."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": correction_prompt},
        ]

        logger.debug("Calling OpenAI for SQL correction.")
        response = self._chat_completion(messages)
        corrected_sql = self._extract_sql(response)
        logger.info("Corrected SQL: %s", corrected_sql[:120])
        return corrected_sql

    # ------------------------------------------------------------------
    # Insight Generation
    # ------------------------------------------------------------------

    def generate_insights(
        self,
        question: str,
        sql_query: str,
        data: List[Dict[str, Any]],
    ) -> str:
        """Generate business-friendly insights from query results.

        Args:
            question: The original user question.
            sql_query: The executed SQL query.
            data: List of result rows (dicts).

        Returns:
            A string with structured bullet-point insights.
        """
        if not data:
            return "No data was returned by the query."

        data_preview = json.dumps(data[:50], default=str, indent=2)
        row_count = len(data)

        prompt = (
            f"A user asked: \"{question}\"\n\n"
            f"The following SQL query was executed:\n```sql\n{sql_query}\n```\n\n"
            f"It returned {row_count} row(s). Here is a sample of the results:\n{data_preview}\n\n"
            "Please provide concise, business-friendly insights from these results. "
            "Identify key trends, outliers, and important metrics. "
            "Format your response as structured bullet points. "
            "Include relevant summary statistics where applicable."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a business intelligence analyst. Provide clear, actionable insights "
                    "from data query results. Be concise, specific, and focus on business value."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        logger.debug("Calling OpenAI for insight generation.")
        return self._chat_completion(messages)

    # ------------------------------------------------------------------
    # SQL Explanation
    # ------------------------------------------------------------------

    def explain_sql(self, sql_query: str) -> str:
        """Summarise what a SQL query does in plain English.

        Args:
            sql_query: The SQL query to explain.

        Returns:
            A plain-English description of the query.
        """
        prompt = (
            f"Explain the following SQL query in plain English (one or two sentences, "
            f"suitable for a business user):\n\n```sql\n{sql_query}\n```"
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful data analyst. Explain SQL queries concisely for business users.",
            },
            {"role": "user", "content": prompt},
        ]

        logger.debug("Calling OpenAI for SQL explanation.")
        return self._chat_completion(messages)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Verify connectivity to Azure OpenAI by sending a minimal request.

        Returns:
            ``True`` if the API is reachable, ``False`` otherwise.
        """
        try:
            self._chat_completion(
                [{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAI health check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call the Azure OpenAI chat completion endpoint.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.
            max_tokens: Override the default max-tokens value from config.

        Returns:
            The assistant's reply as a plain string.
        """
        response = self._client.chat.completions.create(
            model=config.OPENAI_DEPLOYMENT_NAME,
            messages=messages,  # type: ignore[arg-type]
            temperature=config.OPENAI_TEMPERATURE,
            max_tokens=max_tokens if max_tokens is not None else config.OPENAI_MAX_TOKENS,
            top_p=config.OPENAI_TOP_P,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _build_sql_system_prompt(schema_context: str) -> str:
        """Compose the system prompt for NL → SQL generation.

        Args:
            schema_context: Formatted database schema string.

        Returns:
            The full system prompt string.
        """
        blocked = ", ".join(config.BLOCKED_SQL_KEYWORDS)
        return (
            "You are an expert SQL analyst for Microsoft Azure SQL Database. "
            "Your task is to convert natural-language questions into valid T-SQL SELECT queries.\n\n"
            f"{schema_context}\n\n"
            "Rules you MUST follow:\n"
            "1. Generate ONLY SELECT queries — no INSERT, UPDATE, DELETE, DROP, or DDL statements.\n"
            f"2. Never use any of these keywords: {blocked}.\n"
            "3. Always use fully qualified table names (schema.table).\n"
            "4. Limit results to a reasonable number of rows using TOP or appropriate filtering.\n"
            "5. Return ONLY the raw SQL query — no markdown fences, no explanations.\n"
            "6. Use proper T-SQL syntax compatible with Azure SQL Database.\n"
            "7. Prefer aggregation queries that answer the business question efficiently."
        )

    @staticmethod
    def _extract_sql(raw_response: str) -> str:
        """Strip markdown code fences from a raw LLM response.

        Args:
            raw_response: Raw text returned by the LLM.

        Returns:
            Clean SQL string.
        """
        text = raw_response.strip()

        # Remove ```sql ... ``` fences.
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop first line (```sql or ```) and last ``` line.
            inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            text = "\n".join(inner).strip()

        return text
