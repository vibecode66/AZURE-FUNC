# Cosmic Insights

> A **Microsoft 365 Copilot** analytics agent deployed as an **Azure Function App**.  
> Ask natural-language questions about your Azure SQL database and receive SQL queries, data, visualisations, and AI-generated insights in return.

---

## Architecture

```
User (Microsoft Copilot)
  ↓
Cosmic Insights Agent
  ↓
Azure Function App  (POST /api/analyze | GET /api/health | GET /api/schema)
  ↓
Schema Processing Layer  ←→  Metadata / Semantic Mapping Layer
  ↓
Azure OpenAI  (NL → SQL + Insights)
  ↓
SQL Validation Layer
  ↓
Azure SQL Database
  ↓
Aggregated Result Processing
  ↓
Visualization Selection Engine  +  Insight Generation Engine
  ↓
Chart (base64 PNG) + Insights returned to Copilot
```

---

## Project Structure

```
cosmic_insights/
├── config.py                      # All configuration (reads from env vars)
├── models.py                      # Pydantic request/response models
├── function_app.py                # Azure Functions entry point
├── schema_processor.py            # Schema reading & caching
├── semantic_mapper.py             # Business-term → DB-identifier mapping
├── openai_client.py               # Azure OpenAI wrapper
├── sql_validator.py               # SQL safety validation
├── database_client.py             # Azure SQL connection management
├── visualization_engine.py        # Auto chart generation (matplotlib)
├── insight_engine.py              # Business insight generation
├── orchestrator.py                # Full pipeline coordinator
├── requirements.txt               # Python dependencies
├── host.json                      # Azure Functions host settings
├── local.settings.json.template   # Template for local dev secrets
└── README.md                      # This file
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| Azure Functions Core Tools | v4 |
| ODBC Driver for SQL Server | 18 |
| Azure Subscription | — |

---

## Local Development Setup

### 1 — Clone the repository and navigate to this folder

```bash
git clone <repo-url>
cd AZURE-FUNC/cosmic_insights
```

### 2 — Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Configure local settings

Copy the template and fill in your values:

```bash
cp local.settings.json.template local.settings.json
```

Edit `local.settings.json` and replace every `<placeholder>` with actual values.  
**Never commit `local.settings.json` to source control** (it is in `.gitignore`).

### 5 — Run the Function App locally

```bash
func start
```

The host will print the local URLs, e.g.:

```
analyze: [POST,OPTIONS] http://localhost:7071/api/analyze
health:  [GET,OPTIONS]  http://localhost:7071/api/health
schema:  [GET,OPTIONS]  http://localhost:7071/api/schema
```

---

## API Documentation

### `POST /api/analyze`

Orchestrates the full analytics pipeline and returns a structured response.

**Request body**

```json
{
  "question": "What are the top 10 customers by revenue this year?"
}
```

**Response (200 OK)**

```json
{
  "question": "What are the top 10 customers by revenue this year?",
  "generated_sql": "SELECT TOP 10 ...",
  "sql_explanation": "This query retrieves the top 10 customers ranked by total revenue for the current year.",
  "data": [
    { "customer_name": "Acme Corp", "total_amount": 512000.00 },
    ...
  ],
  "chart_base64": "<base64-encoded PNG>",
  "chart_type": "horizontal_bar",
  "insights": "• Acme Corp leads with $512K in revenue...\n• ...",
  "metadata": {
    "elapsed_seconds": 3.42,
    "row_count": 10,
    "sql_attempts": 1
  },
  "success": true
}
```

**Error response (4xx / 5xx)**

```json
{
  "success": false,
  "error": "Failed to generate a valid SQL query after 3 attempts: ..."
}
```

---

### `GET /api/health`

Returns the status of all downstream dependencies.

**Response (200 — healthy)**

```json
{
  "status": "healthy",
  "sql_connected": true,
  "openai_reachable": true,
  "details": {}
}
```

**Response (503 — degraded)**

```json
{
  "status": "degraded",
  "sql_connected": false,
  "openai_reachable": true,
  "details": {
    "sql_error": "Login failed for user '...'"
  }
}
```

---

### `GET /api/schema`

Returns the current database schema as seen by the agent.

**Query parameters**

| Parameter | Type | Description |
|---|---|---|
| `refresh` | `bool` | Pass `true` to bypass the schema cache |

**Response (200 OK)**

```json
{
  "schema_name": "dbo",
  "tables": [
    {
      "table_name": "orders",
      "columns": [
        { "column_name": "order_id", "data_type": "int", "is_nullable": false, "is_primary_key": true, ... },
        ...
      ]
    }
  ],
  "formatted_schema": "Database Schema:\n\nTable: dbo.orders\n  - order_id ...",
  "cached": true
}
```

---

## Deployment to Azure

### 1 — Create Azure resources

```bash
# Resource group
az group create --name cosmic-insights-rg --location eastus

# Storage account (required by Azure Functions)
az storage account create \
  --name cosmicinsightsstorage \
  --resource-group cosmic-insights-rg \
  --sku Standard_LRS

# Function App (Python 3.11, Consumption plan)
az functionapp create \
  --resource-group cosmic-insights-rg \
  --consumption-plan-location eastus \
  --runtime python \
  --runtime-version 3.11 \
  --functions-version 4 \
  --name cosmic-insights-func \
  --storage-account cosmicinsightsstorage \
  --os-type Linux
```

### 2 — Set application settings

```bash
az functionapp config appsettings set \
  --name cosmic-insights-func \
  --resource-group cosmic-insights-rg \
  --settings \
    SQL_SERVER="<your-server>.database.windows.net" \
    SQL_DATABASE="<your-database>" \
    SQL_USERNAME="<username>" \
    SQL_PASSWORD="<password>" \
    OPENAI_ENDPOINT="https://<resource>.openai.azure.com/" \
    OPENAI_API_KEY="<key>" \
    OPENAI_DEPLOYMENT_NAME="gpt-4o"
```

### 3 — Deploy the code

```bash
func azure functionapp publish cosmic-insights-func
```

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `SQL_SERVER` | `your-server.database.windows.net` | Azure SQL server FQDN |
| `SQL_DATABASE` | `your-database` | Database name |
| `SQL_USERNAME` | `your-username` | SQL login username |
| `SQL_PASSWORD` | *(empty)* | SQL login password |
| `SQL_DRIVER` | `{ODBC Driver 18 for SQL Server}` | ODBC driver string |
| `SQL_PORT` | `1433` | SQL server port |
| `SQL_CONNECTION_TIMEOUT` | `30` | Connection timeout (seconds) |
| `SQL_QUERY_TIMEOUT` | `60` | Query execution timeout (seconds) |
| `OPENAI_ENDPOINT` | — | Azure OpenAI resource endpoint |
| `OPENAI_API_KEY` | — | Azure OpenAI API key |
| `OPENAI_API_VERSION` | `2024-02-01` | API version string |
| `OPENAI_DEPLOYMENT_NAME` | `gpt-4o` | Chat completion deployment name |
| `OPENAI_TEMPERATURE` | `0.0` | Sampling temperature |
| `OPENAI_MAX_TOKENS` | `2000` | Max completion tokens |
| `MAX_SQL_RESULT_ROWS` | `500` | Cap on rows returned per query |
| `MAX_SQL_RETRIES` | `3` | SQL generation retry limit |
| `SCHEMA_NAME` | `dbo` | Target database schema |
| `SCHEMA_TABLES` | *(all)* | Comma-separated table allow-list |
| `EXCLUDED_COLUMNS` | *(none)* | Comma-separated columns to hide |
| `SCHEMA_CACHE_TTL_SECONDS` | `3600` | Schema cache lifetime |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## Security Notes

- SQL queries are validated before execution: only `SELECT` statements are allowed.
- Blocked keywords: `DROP DELETE INSERT UPDATE ALTER CREATE TRUNCATE EXEC EXECUTE GRANT REVOKE`.
- SQL injection patterns are detected and rejected.
- All credentials are read exclusively from environment variables — never hardcoded.
- `local.settings.json` is excluded from source control via `.gitignore`.

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Commit changes following [Conventional Commits](https://www.conventionalcommits.org/).
4. Open a pull request.
