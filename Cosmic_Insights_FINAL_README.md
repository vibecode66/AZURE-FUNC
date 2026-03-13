
# Cosmic Insights

## Overview
Cosmic Insights is a **Microsoft 365 Copilot analytics agent** designed to provide **data analytics and visualization** using enterprise data stored in **Azure SQL**.

The agent accepts natural‑language analytical questions from Copilot users, translates them into **safe SQL aggregation queries**, retrieves summarized results from Azure SQL, automatically selects the most appropriate visualization, and returns **charts with insights**.

The system is designed for **enterprise‑scale analytics**, reliability, schema flexibility, and safe query generation.

---

# User Channel
**User Interface:** Microsoft 365 Copilot only

Users interact with the solution only through **Copilot conversations**. No standalone UI is required.

---

# Agent Name
**Cosmic Insights**

**Description**  
Cosmic Insights is an AI‑powered analytics and visualization agent that retrieves data from Azure SQL, performs aggregations, selects appropriate visualizations, and generates insights to help business users understand operational trends and performance metrics.

---

# Solution Architecture

User (Microsoft Copilot)
        │
   Cosmic Insights Agent
        │
  Azure Function App (HTTP APIs)
        │
   Schema Processing Layer
        │
 Metadata / Semantic Mapping Layer
        │
 Azure OpenAI (NL → SQL + Insights)
        │
 SQL Validation Layer
        │
     Azure SQL Database
        │
 Aggregated Result Processing
        │
 Visualization Selection Engine
        │
 Insight Generation Engine
        │
 Chart + Insights returned to Copilot

---

# Core Components

| Component | Role |
|---|---|
Microsoft Copilot | User interaction interface |
Cosmic Insights Agent | Orchestrates analytics workflow |
Azure Function App | Backend API hosting |
Azure OpenAI | LLM for NL→SQL generation, retries, and insights |
Schema Processing Layer | Dynamically analyzes database schema |
Semantic Mapping Layer | Maps business terms to database schema |
SQL Validation Layer | Ensures query safety |
Azure SQL Database | Enterprise data store |
Visualization Engine | Generates charts |
Insight Engine | Produces textual insights |

---

# Azure Function App Requirements

The backend must be implemented using **Azure Function App with HTTP‑triggered functions**.

Responsibilities:
- Accept Copilot agent requests
- Parse user analytics question
- Process schema metadata
- Generate SQL using Azure OpenAI
- Validate SQL queries
- Execute SQL queries in Azure SQL
- Generate visualizations
- Generate insights
- Return structured response to Copilot

---

# Azure OpenAI Integration

The solution must use **Azure OpenAI** as the Large Language Model layer.

Azure OpenAI is responsible for:

- NL → SQL query generation
- SQL retry / correction
- insight generation
- query interpretation

### Azure OpenAI Configuration

The Azure OpenAI endpoint must be configurable.

Example environment variables:

AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/  
AZURE_OPENAI_API_KEY=<key>  
AZURE_OPENAI_MODEL=gpt-4.1

### LLM Invocation Location

The Azure OpenAI call must be executed from the **Azure Function backend code**, not from Copilot directly.

---

# Prompt Management Requirements

System prompts must be **maintained in code** and version controlled.

Recommended structure:

app/
  prompts/
    system_prompt.txt
    sql_generation_prompt.txt
    sql_retry_prompt.txt
    insight_prompt.txt

### System Prompt Responsibility

The system prompt defines the behavior of the agent including:

- SQL generation rules
- visualization selection
- safety guardrails
- retry logic
- insight generation

---

# Schema Processing Requirements

The system must support **dynamic schema processing** because table structure and metadata may vary.

Responsibilities:

- discover available tables
- identify columns
- determine data types
- detect date fields
- detect numeric measures
- detect categorical dimensions
- generate schema context for LLM

---

# Semantic Mapping Layer

The system must maintain a mapping between **business language and database schema**.

Example:

| Business Term | Database Mapping |
|---|---|
Tickets | COUNT(TicketNumber) |
Month | createdon |
Country | msdyn_countrysubmitteridname |
Priority | prioritycodename |

Responsibilities:

- business terminology mapping
- synonyms
- allowed metrics
- allowed dimensions

---

# SQL Generation Requirements

SQL must be generated using Azure OpenAI but must follow strict rules.

Allowed SQL functions:

COUNT  
SUM  
AVG  
MIN  
MAX

SQL must:

- be aggregation based
- avoid raw dataset retrieval
- avoid destructive commands

---

# SQL Validation Requirements

Before execution, SQL must pass a validation layer that ensures:

- allowed tables only
- allowed columns only
- aggregation presence
- result size limits
- safe query execution

---

# Self‑Correction and Retry Mechanism

If SQL generation fails, the system must enter a **self‑correction loop**.

Retry inputs:

- user question
- schema context
- generated SQL
- database error message

Retry flow:

1. Generate SQL
2. Validate SQL
3. Execute SQL
4. Capture error
5. Regenerate SQL
6. Retry execution

Retry attempts must be **bounded**.

---

# Fallback Analytics Logic

If SQL generation repeatedly fails, predefined analytics templates must be used.

Examples:

- month‑on‑month ticket trend
- tickets by category
- status distribution
- top N categories
- average duration metrics

---

# Visualization Requirements

The system must automatically choose visualization types.

| Query Type | Visualization |
|---|---|
Time trend | Line Chart |
Category comparison | Bar Chart |
Distribution | Pie Chart |
Ranking | Bar Chart |
Structured data | Table |
Multi‑series | Grouped Bar Chart |

Supported chart types:

- Bar Chart
- Line Chart
- Pie Chart
- Table
- Grouped Bar Chart

---

# Insight Generation

Every response must include **insights with the visualization**.

Insights must summarize:

- key trends
- highest / lowest values
- notable patterns
- comparisons

Example:

"Ticket volume increased steadily from January to March with March showing the highest activity."

---

# Output Contract

Each response must include:

- user question
- interpreted metric
- interpreted dimension
- SQL execution status
- visualization type
- aggregated data
- generated insight

---

# Data Processing Rules

Mandatory rules:

1. Never return raw datasets
2. Always aggregate in SQL
3. Limit chart results to 10‑50 rows
4. Perform heavy processing in database

---

# Observability and Logging

Logs must capture:

- user query
- schema used
- generated SQL
- SQL validation results
- database errors
- retry attempts
- final SQL
- visualization selected
- generated insights

---

# Security and Governance

Required controls:

- Copilot → backend authentication
- secure database connection
- SQL validation
- schema allowlist
- input validation

Recommended Azure features:

- Entra ID
- Managed Identity
- Azure Key Vault
- Azure Monitor

---

# Suggested Project Structure

cosmic-insights/

README.md  
requirements.txt  
host.json  
local.settings.json  
function_app.py  

app/

config/  
routes/  
services/  
validators/  
models/  
prompts/  

tests/

---

# Future Enhancements

- semantic metric layer
- predictive analytics
- anomaly detection
- Power BI integration
- caching frequent queries
- scheduled insights

---

# Final Summary

Cosmic Insights is an enterprise‑grade Copilot analytics agent that:

- uses **Microsoft Copilot as the user interface**
- runs backend APIs on **Azure Function App**
- uses **Azure OpenAI for NL→SQL and insights**
- dynamically processes database schemas
- validates SQL queries
- includes self‑correcting retry mechanisms
- automatically selects visualizations
- generates insights for every result
- supports large‑scale enterprise analytics

