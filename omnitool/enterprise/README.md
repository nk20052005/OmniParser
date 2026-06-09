# Omnitool Enterprise

Autonomous Enterprise Operations Platform powered by **Gemma 4**.

## Architecture

```
User Message (Slack / Gradio)
       │
       ▼
┌──────────────────────────────┐
│     Conversation Engine      │
│                              │
│  ┌────────────────────────┐  │
│  │   Intent Detection     │  │  ← Gemma 4 semantic classification
│  │   (NO keyword matching)│  │
│  └──────────┬─────────────┘  │
│             ▼                │
│  ┌────────────────────────┐  │
│  │   Entity Extraction    │  │  ← Gemma 4 entity parsing
│  └──────────┬─────────────┘  │
│             ▼                │
│  ┌────────────────────────┐  │
│  │   Slot Validation      │  │  ← Parameter checking
│  │   & Follow-up Q's      │  │
│  └──────────┬─────────────┘  │
│             ▼                │
│  ┌────────────────────────┐  │
│  │   Approval Check       │  │  ← Human-in-the-loop for dangerous ops
│  └──────────┬─────────────┘  │
│             ▼                │
│  ┌────────────────────────┐  │
│  │   Tool Execution       │  │  ← Azure, ServiceNow, Email, Slack, etc.
│  └──────────┬─────────────┘  │
│             ▼                │
│  ┌────────────────────────┐  │
│  │   Response Generation  │  │  ← Natural language response via Gemma 4
│  └──────────┬─────────────┘  │
│             ▼                │
│  ┌────────────────────────┐  │
│  │   Memory Update        │  │  ← Context persistence (SQLite)
│  └────────────────────────┘  │
└──────────────────────────────┘
```

## Multi-Agent Routing

| Agent | Domain | Intents |
|-------|--------|---------|
| CloudOps Agent | Azure VMs, Monitoring | vm_*, monitor_*, resource_health, cost_analysis |
| ServiceNow Agent | Incidents, RITMs | incident_*, ritm_*, service_request_* |
| Email Agent | Mailbox operations | email_* |
| Slack Agent | Messaging | slack_* |
| RCA Agent | Diagnostics | log_analysis, alert_analysis, rca_generate |
| Mission Control | Dashboards, Runbooks | dashboard_generate, runbook_execute |
| OmniParser Agent | GUI fallback | Any (when APIs unavailable) |

## Quick Start

### 1. Configure

```bash
cd omnitool/enterprise
cp .env.example .env
# Edit .env with your credentials
```

### 2. Install Dependencies

```bash
pip install -r requirements-enterprise.txt
```

### 3. Start Gemma 4

Using Ollama:
```bash
ollama serve
ollama run gemma4
```

Or using vLLM:
```bash
python -m vllm.entrypoints.openai.api_server --model google/gemma-4 --port 8080
```

### 4. Launch

**Both interfaces (recommended):**
```bash
python -m omnitool.enterprise.main --mode both
```

**Gradio only (in-VM):**
```bash
python -m omnitool.enterprise.main --mode gradio --port 7860
```

**Slack only (external):**
```bash
python -m omnitool.enterprise.main --mode slack
```

## Supported Operations

### Azure VM Management
- Start, Stop, Restart, Resize, Deallocate
- Check status, List VMs, Get metrics

### Azure Monitoring
- Check alerts, Get metrics, Resource health, Cost analysis

### ServiceNow
- Create/Update/Close/Assign incidents
- Create RITMs and service requests
- Query tickets

### Email
- Read, Send, Reply, Forward, Search

### Slack
- Send messages, Reply in threads, Broadcast

### Analytics
- Log analysis, Alert analysis, RCA generation
- Dashboard generation, Runbook execution

## Natural Language Examples

All of these work without keyword matching:

| User Says | Intent Detected | Action |
|-----------|----------------|--------|
| "Bring the production web server online" | vm_start | Start VM |
| "Can you shut down the VM that is costing the most?" | cost_analysis → vm_stop | Analyze costs, then stop |
| "Create a Sev2 incident for database latency" | incident_create | Create incident |
| "Send an email to the network team about the outage" | email_send | Send email |
| "Show me all unresolved incidents" | incident_query | Query incidents |
| "Generate RCA for yesterday's outage" | rca_generate | Generate RCA |
| "Check if any VMs are underutilized" | vm_metrics | Get VM metrics |

## Safety Features

### Human Approval Required For:
- Production VM stop/restart/resize
- Runbook execution
- Mass email/broadcast
- Incident closure

### The system will NEVER:
- Use keyword matching for intent detection
- Execute without required parameters
- Skip confirmation for dangerous operations
- Hallucinate execution success
- Expose secrets or credentials

## Project Structure

```
omnitool/enterprise/
├── __init__.py
├── config.py               # Environment-based configuration
├── main.py                 # Application entry point
├── .env.example            # Configuration template
│
├── engine/                 # Core conversation pipeline
│   ├── gemma_client.py     # Gemma 4 LLM client
│   ├── intent.py           # Intent detection
│   ├── entities.py         # Entity extraction
│   ├── slots.py            # Slot filling & follow-up questions
│   ├── response.py         # Response generation
│   └── conversation.py     # Main orchestrator
│
├── tools/                  # Tool implementations
│   ├── base.py             # Base tool class
│   ├── registry.py         # Tool registry & execution
│   ├── azure_vm.py         # Azure VM operations
│   ├── azure_monitor.py    # Azure monitoring
│   ├── servicenow.py       # ServiceNow operations
│   ├── email_tools.py      # Email operations
│   ├── slack_tools.py      # Slack operations
│   └── analytics.py        # Log/Alert analysis, RCA, Dashboard, Runbook
│
├── agents/                 # Multi-agent system
│   ├── router.py           # Intent-to-agent routing
│   └── omniparser_agent.py # OmniParser GUI fallback
│
├── memory/                 # Context persistence
│   └── store.py            # SQLite-based memory
│
├── approval/               # Human approval system
│   └── manager.py          # Approval rules & management
│
├── integrations/           # External integrations
│   └── slack_bot.py        # Slack Socket Mode bot
│
└── interfaces/             # User interfaces
    └── gradio_app.py       # Gradio web UI
```
