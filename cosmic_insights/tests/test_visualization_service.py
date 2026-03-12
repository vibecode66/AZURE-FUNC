"""
Unit tests for visualization type selection.
"""

import pytest
from app.services.visualization_service import VisualizationService


@pytest.fixture
def viz_service():
    return VisualizationService()


class TestChartSelection:

    def test_time_trend_selects_line(self, viz_service):
        result = viz_service.select_chart_type(
            "Show ticket trend over time",
            ["month", "count"],
            [{"month": "2025-01", "count": 10}],
        )
        assert result == "line"

    def test_distribution_few_items_selects_pie(self, viz_service):
        data = [{"status": f"S{i}", "count": i * 5} for i in range(4)]
        result = viz_service.select_chart_type(
            "Show distribution of tickets by status",
            ["status", "count"],
            data,
        )
        assert result == "pie"

    def test_distribution_many_items_selects_bar(self, viz_service):
        data = [{"status": f"S{i}", "count": i * 5} for i in range(15)]
        result = viz_service.select_chart_type(
            "Show distribution of tickets by status",
            ["status", "count"],
            data,
        )
        assert result == "bar"

    def test_ranking_selects_bar(self, viz_service):
        data = [{"cat": f"C{i}", "count": i} for i in range(5)]
        result = viz_service.select_chart_type(
            "Top 5 categories by ticket count",
            ["cat", "count"],
            data,
        )
        assert result == "bar"

    def test_date_like_first_col_selects_line(self, viz_service):
        data = [
            {"month": "2025-01", "count": 10},
            {"month": "2025-02", "count": 20},
        ]
        result = viz_service.select_chart_type(
            "Tickets by month",
            ["month", "count"],
            data,
        )
        assert result == "line"

    def test_multi_numeric_selects_grouped_bar(self, viz_service):
        data = [{"cat": "A", "open": 5, "closed": 3}]
        result = viz_service.select_chart_type(
            "Compare open vs closed tickets",
            ["cat", "open", "closed"],
            data,
        )
        assert result == "grouped_bar"
