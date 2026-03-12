# Cosmic Insights

## Overview
Cosmic Insights is an **enterprise-grade Microsoft 365 Copilot analytics agent** that retrieves data from Azure SQL, performs aggregations via Azure OpenAI-generated SQL, auto-selects visualizations, and returns charts with insights.

## Architecture

```
User (Microsoft Copilot)
        |
   Cosmic Insights Agent
        |
  Azure Function App (HTTP APIs)
        |
   Schema Processing Layer
        |
 Metadata / Semantic Mapping Layer
        |
 Azure OpenAI (NL -> SQL + Insights)
        |
 SQL Validation Layer
        |
     Azure SQL Database
        |
 Aggregated Result Processing
        |
 Visualization Selection Engine
        |
 Insight Generation Engine
        |
 Chart + Insights returned to Copilot
```

## Project Structure

```
cosmic_insights/
├── host.json
├── local.settings.json
├── requirements.txt
├── function_app.py
├── app/
│   ├── config/
│   │   ├── settings.py
│   │   ├── schema_allowlist.py
│   │   └── semantic_mappings.py
│   ├── routes/
│   │   └── analytics.py
│   ├── services/
│   │   ├── schema_service.py
│   │   ├── openai_service.py
│   │   ├── sql_executor.py
│   │   ├── visualization_service.py
│   │   ├── insight_service.py
│   │   └── orchestrator.py
│   ├── validators/
│   │   └── sql_validator.py
│   ├── models/
│   │   └── response_models.py
│   └── prompts/
│       ├── system_prompt.txt
│       ├── sql_generation_prompt.txt
│       ├── sql_retry_prompt.txt
│       └── insight_prompt.txt
└── tests/
    ├── test_sql_validator.py
    ├── test_visualization_service.py
    └── test_orchestrator.py
```

## Setup

### Prerequisites
- Python 3.10+
- Azure Functions Core Tools v4
- Azure SQL Database
- Azure OpenAI resource
- ODBC Driver 18 for SQL Server

### Installation
```bash
cd cosmic_insights
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration
Edit `local.settings.json` with your real credentials:
- `AZURE_OPENAI_ENDPOINT` - Your Azure OpenAI endpoint
- `AZURE_OPENAI_API_KEY` - Your Azure OpenAI API key
- `AZURE_OPENAI_MODEL` - Model deployment name (default: gpt-4.1)
- `AZURE_SQL_SERVER` - Azure SQL server address
- `AZURE_SQL_DATABASE` - Database name
- `AZURE_SQL_USERNAME` / `AZURE_SQL_PASSWORD` - DB credentials

### Run Locally
```bash
func start
```

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/analyze` | Function key | Submit analytics question |
| GET | `/api/health` | Anonymous | Health check |
| GET | `/api/schema` | Function key | View discovered schema |

### Sample Request
```json
POST /api/analyze
{
  "question": "Show me ticket trend by month"
}
```

### Sample Response
```json
{
  "user_question": "Show me ticket trend by month",
  "interpreted_metric": "ticket_count",
  "interpreted_dimension": "month",
  "sql_status": "success",
  "sql_query": "SELECT FORMAT(createdon, 'yyyy-MM') AS month, COUNT(ticketnumber) AS ticket_count FROM tickets GROUP BY FORMAT(createdon, 'yyyy-MM') ORDER BY month",
  "visualization_type": "line",
  "data": [...],
  "chart_base64": "iVBORw0KGgo...",
  "insight": "Ticket volume increased steadily from January to March...",
  "retry_count": 0
}
```

## Testing
```bash
pip install pytest pytest-asyncio
cd cosmic_insights
python -m pytest tests/ -v
```

## Security
- SQL validation layer blocks destructive commands
- Schema allow-list restricts table/column access
- All credentials externalized to environment variables
- Supports Azure Managed Identity, Key Vault, Entra ID
