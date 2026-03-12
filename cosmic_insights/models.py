"""
Pydantic models for Cosmic Insights request/response validation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AnalyticsRequest(BaseModel):
    """Incoming analytics request containing a natural-language question."""

    question: str = Field(..., min_length=1, description="Natural-language analytics question")


class ValidationResult(BaseModel):
    """Result of SQL validation."""

    is_valid: bool
    errors: List[str] = Field(default_factory=list)


class AnalyticsResponse(BaseModel):
    """Full analytics response returned to the caller."""

    question: str
    generated_sql: Optional[str] = None
    sql_explanation: Optional[str] = None
    data: List[Dict[str, Any]] = Field(default_factory=list)
    chart_base64: Optional[str] = None
    chart_type: Optional[str] = None
    insights: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str
    sql_connected: bool
    openai_reachable: bool
    details: Dict[str, Any] = Field(default_factory=dict)


class SchemaResponse(BaseModel):
    """Database schema metadata response."""

    schema_name: str
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    formatted_schema: str = ""
    cached: bool = False
