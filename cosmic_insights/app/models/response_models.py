"""
Pydantic models defining the strict output contract
returned to Copilot on every analytics request.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class AnalyticsRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, description="User analytics question")


class AnalyticsResponse(BaseModel):
    user_question: str
    interpreted_metric: str
    interpreted_dimension: str
    sql_status: str = Field(description="success | fallback | error")
    sql_query: str = ""
    visualization_type: str = ""
    data: list[dict[str, Any]] = Field(default_factory=list)
    chart_base64: Optional[str] = None
    insight: str = ""
    retry_count: int = 0
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    services: dict[str, str] = Field(default_factory=dict)