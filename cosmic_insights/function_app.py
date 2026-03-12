"""
Azure Functions entry point for Cosmic Insights.

Exposes three HTTP-triggered functions:
- POST /api/analyze   — Main analytics endpoint
- GET  /api/health    — Health check
- GET  /api/schema    — Current database schema metadata
"""

from __future__ import annotations

import json
import logging

import azure.functions as func

from cosmic_insights.database_client import DatabaseClient
from cosmic_insights.insight_engine import InsightEngine
from cosmic_insights.models import AnalyticsRequest, HealthResponse, SchemaResponse
from cosmic_insights.openai_client import OpenAIClient
from cosmic_insights.orchestrator import Orchestrator
from cosmic_insights.schema_processor import SchemaProcessor
from cosmic_insights.semantic_mapper import SemanticMapper
from cosmic_insights.visualization_engine import VisualizationEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Function App
# ---------------------------------------------------------------------------
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# ---------------------------------------------------------------------------
# Shared component instances (lazy-initialised per function worker process).
# ---------------------------------------------------------------------------
_db_client: DatabaseClient | None = None
_openai_client: OpenAIClient | None = None
_schema_processor: SchemaProcessor | None = None
_semantic_mapper: SemanticMapper | None = None
_visualization_engine: VisualizationEngine | None = None
_insight_engine: InsightEngine | None = None
_orchestrator: Orchestrator | None = None


def _get_orchestrator() -> Orchestrator:
    """Return the shared :class:`~cosmic_insights.orchestrator.Orchestrator` instance.

    Components are initialised once per worker-process lifetime.

    Returns:
        Fully configured :class:`~cosmic_insights.orchestrator.Orchestrator`.
    """
    global _db_client, _openai_client, _schema_processor  # noqa: PLW0603
    global _semantic_mapper, _visualization_engine, _insight_engine, _orchestrator

    if _orchestrator is None:
        _db_client = DatabaseClient()
        _openai_client = OpenAIClient()
        _schema_processor = SchemaProcessor()
        _semantic_mapper = SemanticMapper()
        _visualization_engine = VisualizationEngine()
        _insight_engine = InsightEngine(_openai_client)

        _orchestrator = Orchestrator(
            db_client=_db_client,
            openai_client=_openai_client,
            schema_processor=_schema_processor,
            semantic_mapper=_semantic_mapper,
            visualization_engine=_visualization_engine,
            insight_engine=_insight_engine,
        )

    return _orchestrator


def _json_response(data: dict, status_code: int = 200) -> func.HttpResponse:
    """Helper to build a JSON HTTP response.

    Args:
        data: Dictionary to serialise as JSON.
        status_code: HTTP status code.

    Returns:
        :class:`azure.functions.HttpResponse` with ``application/json`` content type.
    """
    return func.HttpResponse(
        body=json.dumps(data, default=str),
        status_code=status_code,
        mimetype="application/json",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        },
    )


# ---------------------------------------------------------------------------
# POST /api/analyze
# ---------------------------------------------------------------------------
@app.function_name(name="analyze")
@app.route(route="analyze", methods=["POST", "OPTIONS"])
def analyze(req: func.HttpRequest) -> func.HttpResponse:
    """Main analytics endpoint.

    Accepts a JSON body ``{"question": "..."}`` and orchestrates the full
    pipeline, returning query, data, chart (base64), and insights.

    Args:
        req: The incoming HTTP request.

    Returns:
        JSON response with analytics results or an error message.
    """
    if req.method == "OPTIONS":
        return _json_response({}, 200)

    try:
        body = req.get_json()
    except ValueError:
        return _json_response({"error": "Request body must be valid JSON."}, 400)

    try:
        request_model = AnalyticsRequest(**body)
    except Exception as exc:  # noqa: BLE001
        return _json_response({"error": f"Invalid request: {exc}"}, 400)

    logger.info("analyze: question=%s", request_model.question[:80])

    try:
        orchestrator = _get_orchestrator()
        response = orchestrator.analyze(request_model.question)
        return _json_response(response.model_dump(), 200 if response.success else 500)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in /api/analyze.")
        return _json_response({"error": f"Internal server error: {exc}"}, 500)


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------
@app.function_name(name="health")
@app.route(route="health", methods=["GET", "OPTIONS"])
def health(req: func.HttpRequest) -> func.HttpResponse:
    """Health-check endpoint.

    Verifies connectivity to Azure SQL and Azure OpenAI.

    Args:
        req: The incoming HTTP request.

    Returns:
        JSON health status.
    """
    if req.method == "OPTIONS":
        return _json_response({}, 200)

    logger.info("health: checking dependencies.")

    sql_ok = False
    openai_ok = False
    details: dict = {}

    try:
        db = DatabaseClient()
        sql_ok = db.health_check()
        db.close()
    except Exception as exc:  # noqa: BLE001
        details["sql_error"] = str(exc)

    try:
        oa = OpenAIClient()
        openai_ok = oa.health_check()
    except Exception as exc:  # noqa: BLE001
        details["openai_error"] = str(exc)

    overall_status = "healthy" if sql_ok and openai_ok else "degraded"

    response = HealthResponse(
        status=overall_status,
        sql_connected=sql_ok,
        openai_reachable=openai_ok,
        details=details,
    )

    status_code = 200 if overall_status == "healthy" else 503
    return _json_response(response.model_dump(), status_code)


# ---------------------------------------------------------------------------
# GET /api/schema
# ---------------------------------------------------------------------------
@app.function_name(name="schema")
@app.route(route="schema", methods=["GET", "OPTIONS"])
def schema(req: func.HttpRequest) -> func.HttpResponse:
    """Returns the current database schema metadata.

    Args:
        req: The incoming HTTP request.

    Returns:
        JSON schema representation.
    """
    if req.method == "OPTIONS":
        return _json_response({}, 200)

    logger.info("schema: fetching database schema.")

    force_refresh = req.params.get("refresh", "false").lower() == "true"

    try:
        db = DatabaseClient()
        sp = SchemaProcessor()
        tables = sp.get_schema(db, force_refresh=force_refresh)
        formatted = sp.format_schema_for_prompt(tables)
        db.close()

        from cosmic_insights import config as _cfg

        response = SchemaResponse(
            schema_name=_cfg.SCHEMA_NAME,
            tables=tables,
            formatted_schema=formatted,
            cached=not force_refresh,
        )
        return _json_response(response.model_dump())
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in /api/schema.")
        return _json_response({"error": f"Internal server error: {exc}"}, 500)
