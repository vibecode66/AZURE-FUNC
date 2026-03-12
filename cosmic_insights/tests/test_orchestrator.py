"""
Integration-style tests for the orchestrator.
Uses mocks for external services (OpenAI, Azure SQL).
"""

import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from app.services.orchestrator import AnalyticsOrchestrator
from app.models.response_models import AnalyticsResponse


@pytest.fixture
def mock_settings():
    with patch("app.config.settings.get_settings") as mock:
        settings = MagicMock()
        settings.query.max_result_rows = 50
        settings.query.max_retry_attempts = 2
        settings.openai.endpoint = "https://test.openai.azure.com/"
        settings.openai.api_key = "test-key"
        settings.openai.model = "gpt-4.1"
        settings.openai.api_version = "2024-12-01-preview"
        settings.openai.temperature = 0.0
        settings.openai.max_tokens = 2048
        settings.sql.connection_string = "DRIVER={test};SERVER=test"
        settings.sql.connection_timeout = 5
        settings.sql.query_timeout = 10
        settings.prompts_dir = "app/prompts"
        settings.logging.level = "DEBUG"
        mock.return_value = settings
        yield settings


class TestOrchestrator:

    @patch("app.services.orchestrator.InsightService")
    @patch("app.services.orchestrator.VisualizationService")
    @patch("app.services.orchestrator.SQLExecutor")
    @patch("app.services.orchestrator.SQLValidator")
    @patch("app.services.orchestrator.OpenAIService")
    @patch("app.services.orchestrator.SchemaService")
    @pytest.mark.asyncio
    async def test_successful_pipeline(
        self,
        MockSchema,
        MockOpenAI,
        MockValidator,
        MockExecutor,
        MockViz,
        MockInsight,
        mock_settings,
    ):
        schema_inst = MockSchema.return_value
        schema_inst.get_schema_context.return_value = "Table: tickets -> [ticketnumber, categoryname]"

        openai_inst = MockOpenAI.return_value
        openai_inst.interpret_question.return_value = ("ticket_count", "category")
        openai_inst.generate_sql.return_value = (
            "SELECT categoryname, COUNT(ticketnumber) AS cnt FROM tickets GROUP BY categoryname"
        )

        validator_inst = MockValidator.return_value
        mock_result = MagicMock()
        mock_result.is_valid = True
        validator_inst.validate.return_value = mock_result

        executor_inst = MockExecutor.return_value
        executor_inst.execute.return_value = (
            ["categoryname", "cnt"],
            [{"categoryname": "Network", "cnt": 42}, {"categoryname": "Hardware", "cnt": 31}],
        )

        viz_inst = MockViz.return_value
        viz_inst.select_chart_type.return_value = "bar"
        viz_inst.render_chart.return_value = "base64encodedpng..."

        insight_inst = MockInsight.return_value
        insight_inst.generate.return_value = "Network leads with 42 tickets."

        orchestrator = AnalyticsOrchestrator()
        result = await orchestrator.process("Show tickets by category")

        assert isinstance(result, AnalyticsResponse)
        assert result.sql_status == "success"
        assert result.interpreted_metric == "ticket_count"
        assert result.visualization_type == "bar"
        assert len(result.data) == 2
        assert result.insight == "Network leads with 42 tickets."
        assert result.chart_base64 == "base64encodedpng..."
