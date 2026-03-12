"""
Visualization Engine for Cosmic Insights.

Auto-selects an appropriate chart type based on data shape and generates a
base64-encoded PNG using matplotlib.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from cosmic_insights import config

# Use non-interactive backend so this works inside Azure Functions.
matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Recognised chart types.
CHART_BAR = "bar"
CHART_HORIZONTAL_BAR = "horizontal_bar"
CHART_LINE = "line"
CHART_PIE = "pie"
CHART_TABLE = "table"


class VisualizationEngine:
    """Generates charts from query result data.

    Chart type selection heuristics:
    - **Pie chart**: single label + single numeric column, ≤ ``PIE_CHART_MAX_CATEGORIES`` rows.
    - **Line chart**: a date/time column is detected in the data.
    - **Horizontal bar**: ranked data (one label column, one numeric, > 10 rows).
    - **Bar chart**: categorical vs. numeric comparison (≤ 20 categories).
    - **Table**: everything else.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_chart(
        self,
        data: List[Dict[str, Any]],
        title: str = "",
        chart_type: Optional[str] = None,
    ) -> Tuple[Optional[str], str]:
        """Generate a chart from *data* and return it as a base64 PNG.

        Args:
            data: List of row dicts returned from the database query.
            title: Optional chart title.
            chart_type: Force a specific chart type; when ``None`` the engine
                auto-selects based on the data shape.

        Returns:
            A ``(base64_png_string, chart_type_used)`` tuple.
            Returns ``(None, "table")`` when the data is empty or unsuitable
            for visualisation.
        """
        if not data:
            logger.info("No data to visualise.")
            return None, CHART_TABLE

        selected_type = chart_type or self._select_chart_type(data)
        logger.info("Generating %s chart for %d rows.", selected_type, len(data))

        try:
            if selected_type == CHART_PIE:
                b64 = self._pie_chart(data, title)
            elif selected_type == CHART_LINE:
                b64 = self._line_chart(data, title)
            elif selected_type == CHART_HORIZONTAL_BAR:
                b64 = self._horizontal_bar_chart(data, title)
            elif selected_type == CHART_BAR:
                b64 = self._bar_chart(data, title)
            else:
                return None, CHART_TABLE
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chart generation failed (%s): %s", selected_type, exc)
            return None, CHART_TABLE

        return b64, selected_type

    # ------------------------------------------------------------------
    # Chart-type selector
    # ------------------------------------------------------------------

    def _select_chart_type(self, data: List[Dict[str, Any]]) -> str:
        """Choose the most suitable chart type for *data*.

        Args:
            data: Query result rows.

        Returns:
            One of the ``CHART_*`` constants.
        """
        if not data:
            return CHART_TABLE

        cols = list(data[0].keys())
        label_cols, numeric_cols, date_cols = self._classify_columns(data, cols)

        n_rows = len(data)

        # Pie: single label + single numeric, few categories.
        if (
            len(label_cols) == 1
            and len(numeric_cols) == 1
            and n_rows <= config.PIE_CHART_MAX_CATEGORIES
        ):
            return CHART_PIE

        # Line: time-series data detected.
        if date_cols:
            return CHART_LINE

        # Horizontal bar: ranked list with many rows.
        if label_cols and numeric_cols and n_rows > 10:
            return CHART_HORIZONTAL_BAR

        # Bar: standard categorical comparison.
        if label_cols and numeric_cols and n_rows <= 20:
            return CHART_BAR

        return CHART_TABLE

    # ------------------------------------------------------------------
    # Chart generators
    # ------------------------------------------------------------------

    def _bar_chart(self, data: List[Dict[str, Any]], title: str) -> str:
        cols = list(data[0].keys())
        label_cols, numeric_cols, _ = self._classify_columns(data, cols)

        label_col = label_cols[0] if label_cols else cols[0]
        value_col = numeric_cols[0] if numeric_cols else cols[-1]

        labels = [str(row[label_col]) for row in data]
        values = [float(row[value_col] or 0) for row in data]

        fig, ax = plt.subplots(figsize=(config.CHART_WIDTH, config.CHART_HEIGHT), dpi=config.CHART_DPI)
        colors = self._get_colors(len(values))
        ax.bar(labels, values, color=colors)
        ax.set_title(title or f"{value_col} by {label_col}", fontsize=config.CHART_TITLE_FONT_SIZE)
        ax.set_xlabel(label_col, fontsize=config.CHART_LABEL_FONT_SIZE)
        ax.set_ylabel(value_col, fontsize=config.CHART_LABEL_FONT_SIZE)
        ax.tick_params(axis="x", rotation=45, labelsize=config.CHART_TICK_FONT_SIZE)
        ax.tick_params(axis="y", labelsize=config.CHART_TICK_FONT_SIZE)
        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _horizontal_bar_chart(self, data: List[Dict[str, Any]], title: str) -> str:
        cols = list(data[0].keys())
        label_cols, numeric_cols, _ = self._classify_columns(data, cols)

        label_col = label_cols[0] if label_cols else cols[0]
        value_col = numeric_cols[0] if numeric_cols else cols[-1]

        labels = [str(row[label_col]) for row in data]
        values = [float(row[value_col] or 0) for row in data]

        # Sort descending for ranked view.
        paired = sorted(zip(values, labels), reverse=True)
        values, labels = zip(*paired) if paired else ([], [])  # type: ignore[assignment]

        fig, ax = plt.subplots(
            figsize=(config.CHART_WIDTH, max(config.CHART_HEIGHT, len(labels) * 0.4)),
            dpi=config.CHART_DPI,
        )
        colors = self._get_colors(len(values))
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=config.CHART_TICK_FONT_SIZE)
        ax.set_title(title or f"{value_col} by {label_col}", fontsize=config.CHART_TITLE_FONT_SIZE)
        ax.set_xlabel(value_col, fontsize=config.CHART_LABEL_FONT_SIZE)
        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _line_chart(self, data: List[Dict[str, Any]], title: str) -> str:
        cols = list(data[0].keys())
        label_cols, numeric_cols, date_cols = self._classify_columns(data, cols)

        x_col = date_cols[0] if date_cols else (label_cols[0] if label_cols else cols[0])
        y_cols = numeric_cols if numeric_cols else [cols[-1]]

        x_vals = [str(row[x_col]) for row in data]
        fig, ax = plt.subplots(figsize=(config.CHART_WIDTH, config.CHART_HEIGHT), dpi=config.CHART_DPI)
        colors = self._get_colors(len(y_cols))
        for idx, y_col in enumerate(y_cols):
            y_vals = [float(row[y_col] or 0) for row in data]
            ax.plot(x_vals, y_vals, marker="o", label=y_col, color=colors[idx % len(colors)])

        ax.set_title(title or f"Trend over {x_col}", fontsize=config.CHART_TITLE_FONT_SIZE)
        ax.set_xlabel(x_col, fontsize=config.CHART_LABEL_FONT_SIZE)
        ax.tick_params(axis="x", rotation=45, labelsize=config.CHART_TICK_FONT_SIZE)
        ax.tick_params(axis="y", labelsize=config.CHART_TICK_FONT_SIZE)
        if len(y_cols) > 1:
            ax.legend(fontsize=config.CHART_LABEL_FONT_SIZE)
        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _pie_chart(self, data: List[Dict[str, Any]], title: str) -> str:
        cols = list(data[0].keys())
        label_cols, numeric_cols, _ = self._classify_columns(data, cols)

        label_col = label_cols[0] if label_cols else cols[0]
        value_col = numeric_cols[0] if numeric_cols else cols[-1]

        labels = [str(row[label_col]) for row in data]
        values = [float(row[value_col] or 0) for row in data]

        # Filter out non-positive slices.
        filtered = [(l, v) for l, v in zip(labels, values) if v > 0]
        if not filtered:
            raise ValueError("No positive values for pie chart.")
        labels, values = zip(*filtered)  # type: ignore[assignment]

        fig, ax = plt.subplots(figsize=(config.CHART_WIDTH, config.CHART_HEIGHT), dpi=config.CHART_DPI)
        colors = self._get_colors(len(values))
        ax.pie(
            values,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"fontsize": config.CHART_TICK_FONT_SIZE},
        )
        ax.set_title(title or f"{value_col} distribution", fontsize=config.CHART_TITLE_FONT_SIZE)
        plt.tight_layout()
        return self._fig_to_base64(fig)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_columns(
        data: List[Dict[str, Any]], cols: List[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Classify columns into label, numeric, and date categories.

        Args:
            data: Query result rows.
            cols: List of column names.

        Returns:
            A ``(label_cols, numeric_cols, date_cols)`` tuple.
        """
        label_cols: List[str] = []
        numeric_cols: List[str] = []
        date_cols: List[str] = []

        date_hints = {"date", "time", "year", "month", "week", "day", "period"}

        for col in cols:
            sample_values = [row[col] for row in data[:10] if row[col] is not None]
            if not sample_values:
                label_cols.append(col)
                continue

            col_lower = col.lower()

            # Date detection: column name hint or datetime type.
            if any(hint in col_lower for hint in date_hints):
                date_cols.append(col)
                continue

            # Numeric detection.
            try:
                list(map(float, sample_values))
                numeric_cols.append(col)
            except (ValueError, TypeError):
                label_cols.append(col)

        return label_cols, numeric_cols, date_cols

    @staticmethod
    def _get_colors(n: int) -> List[str]:
        """Return up to *n* colors from the configured palette, cycling if needed.

        Args:
            n: Number of colors required.

        Returns:
            List of hex color strings.
        """
        palette = config.CHART_COLOR_PALETTE
        return [palette[i % len(palette)] for i in range(n)]

    @staticmethod
    def _fig_to_base64(fig: plt.Figure) -> str:
        """Serialize a matplotlib figure to a base64-encoded PNG string.

        Args:
            fig: The matplotlib figure to encode.

        Returns:
            Base64-encoded PNG string.
        """
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
