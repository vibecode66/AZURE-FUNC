"""
Pipeline Orchestrator for Cosmic Insights.

Coordinates all layers:
question → semantic mapping → schema context → NL→SQL
→ validation → execution → visualisation → insights → response.

Retry logic is applied when SQL generation or execution fails.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from cosmic_insights import config
from cosmic_insights.database_client import DatabaseClient
from cosmic_insights.insight_engine import InsightEngine
from cosmic_insights.models import AnalyticsResponse
from cosmic_insights.openai_client import OpenAIClient
from cosmic_insights.schema_processor import SchemaProcessor
from cosmic_insights.semantic_mapper import SemanticMapper
from cosmic_insights.sql_validator import validate_sql
from cosmic_insights.visualization_engine import VisualizationEngine

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates the full Cosmic Insights analytics pipeline.

    All component instances are injected at construction time for
    testability and to allow shared connection management.

    Attributes:
        _db: Database client.
        _openai: Azure OpenAI client.
        _schema_processor: Schema cache/formatter.
        _semantic_mapper: Semantic term mapper.
        _visualization: Chart generator.
        _insight_engine: Insight text generator.
    """

    def __init__(
        self,
        db_client: DatabaseClient,
        openai_client: OpenAIClient,
        schema_processor: SchemaProcessor,
        semantic_mapper: SemanticMapper,
        visualization_engine: VisualizationEngine,
        insight_engine: InsightEngine,
    ) -> None:
        self._db = db_client
        self._openai = openai_client
        self._schema_processor = schema_processor
        self._semantic_mapper = semantic_mapper
        self._visualization = visualization_engine
        self._insight_engine = insight_engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, question: str) -> AnalyticsResponse:
        """Run the full analytics pipeline for *question*.

        Args:
            question: Natural-language analytics question from the user.

        Returns:
            An :class:`~cosmic_insights.models.AnalyticsResponse` containing
            the generated SQL, result data, chart, and insights.
        """
        start = time.time()
        logger.info("Starting analytics pipeline for question: %s", question[:100])

        metadata: Dict[str, Any] = {"question_length": len(question)}

        # 1. Semantic mapping.
        enhanced_question = self._semantic_mapper.enhance_question(question)
        logger.debug("Enhanced question: %s", enhanced_question[:120])

        # 2. Fetch and format schema.
        schema = self._schema_processor.get_schema(self._db)
        schema_context = self._schema_processor.format_schema_for_prompt(schema)

        # 3. NL → SQL with retry loop.
        sql: str = ""
        last_error: str = ""
        attempt = 0

        for attempt in range(1, config.MAX_SQL_RETRIES + 1):
            try:
                if attempt == 1:
                    sql = self._openai.generate_sql(enhanced_question, schema_context)
                else:
                    logger.info("Retry %d/%d for SQL generation.", attempt, config.MAX_SQL_RETRIES)
                    sql = self._openai.correct_sql(
                        enhanced_question, sql, last_error, schema_context
                    )

                # 4. SQL Validation.
                validation = validate_sql(sql)
                if not validation.is_valid:
                    last_error = "; ".join(validation.errors)
                    logger.warning("SQL validation failed (attempt %d): %s", attempt, last_error)
                    continue

                # 5. SQL Execution.
                data = self._db.execute_query(sql)
                break

            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                logger.warning("SQL execution error (attempt %d): %s", attempt, last_error)
                data = []
        else:
            # All retries exhausted.
            logger.error("All %d SQL attempts failed. Last error: %s", config.MAX_SQL_RETRIES, last_error)
            return AnalyticsResponse(
                question=question,
                generated_sql=sql,
                success=False,
                error=f"Failed to generate a valid SQL query after {config.MAX_SQL_RETRIES} attempts: {last_error}",
                metadata=metadata,
            )

        metadata["sql_attempts"] = attempt
        metadata["row_count"] = len(data)

        # 6. SQL explanation.
        sql_explanation = ""
        try:
            sql_explanation = self._openai.explain_sql(sql)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SQL explanation failed: %s", exc)

        # 7. Visualisation.
        chart_b64, chart_type = self._visualization.generate_chart(data, title=question[:60])

        # 8. Insight generation.
        insights = self._insight_engine.generate(question, sql, data)

        elapsed = time.time() - start
        metadata["elapsed_seconds"] = round(elapsed, 2)

        logger.info(
            "Pipeline complete in %.2fs: %d rows, chart=%s",
            elapsed,
            len(data),
            chart_type,
        )

        return AnalyticsResponse(
            question=question,
            generated_sql=sql,
            sql_explanation=sql_explanation,
            data=data,
            chart_base64=chart_b64,
            chart_type=chart_type,
            insights=insights,
            metadata=metadata,
            success=True,
        )
