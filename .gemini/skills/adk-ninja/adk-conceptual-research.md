# Google Agent Development Kit (ADK) — Conceptual Research

## Table of Contents

1. [Why Use ADK? Key Advantages](#1-why-use-adk-key-advantages)
2. [ADK vs. Alternatives](#2-adk-vs-alternatives)
3. [ADK in the GCP Stack](#3-adk-in-the-gcp-stack)

---

## 1. Why Use ADK? Key Advantages

### What is ADK?

Google's Agent Development Kit (ADK) is an open-source, code-first framework for building, evaluating, and deploying AI agents. It was announced at Google Cloud NEXT 2025 (April 9, 2025) and is the same framework that powers Google's own products, including Agentspace and Customer Engagement Suite (CES). It is available in **four languages**: Python, TypeScript/JavaScript, Go, and Java.

- **Python**: `pip install google-adk`
- **TypeScript**: `npm install @google/adk`
- **Go**: `go get google.golang.org/adk`
- **Java**: Maven/Gradle dependency

### Core Advantages

#### 1. Agent Development That Feels Like Software Development

ADK's core philosophy is to make agent development feel like traditional software development. It brings well-established software engineering principles — version control, testing, modularity, code review — to the agent-building process. Agents are defined in code with standard programming constructs, not dragged and dropped in a visual editor.

#### 3. Model-Agnostic Design

While optimized for Google's Gemini models, ADK is explicitly designed to work with other LLM providers. Through LiteLLM integration and Vertex AI Model Garden, it supports models from Anthropic (Claude), Meta (Llama), Mistral AI, AI21 Labs, Ollama, vLLM, and others.

#### 4. Deployment-Agnostic Architecture

ADK agents can be deployed anywhere: locally via Docker, on Google Cloud Run, on GKE, on Vertex AI Agent Engine (fully managed), or on any custom infrastructure that supports container images. There is no vendor lock-in to Google Cloud for deployment.

#### 5. Complete Lifecycle Coverage

ADK covers the full agent lifecycle:
- **Build**: Code-first development with CLI and Web UI
- **Interact**: CLI, Web UI, API Server, Python API
- **Evaluate**: Built-in evaluation framework
- **Deploy**: Containerize and ship to any target

### Core Features

#### Multi-Agent Architecture

ADK provides three fundamental agent types, all extending a common `BaseAgent` class:

**LlmAgent (also aliased as `Agent`)**: The primary agent type, powered by an LLM. It can understand natural language, reason, plan, generate responses, and dynamically decide how to proceed or which tools to use. It is non-deterministic and flexible.

**Workflow Agents**: Three specialized agent types for deterministic, structured execution:
- **SequentialAgent**: Executes sub-agents in a predefined, linear sequence.
- **ParallelAgent**: Runs sub-agents concurrently for parallel processing.
- **LoopAgent**: Implements iterative execution patterns, running sub-agents in a loop until a condition is met.

**Custom Agents**: Developers can extend `BaseAgent` directly to implement unique operational logic, specific control flows, or specialized integrations.

The key architectural insight is that complex applications **compose** these agent types together. Agents are organized in hierarchies where parent agents can delegate to child (sub) agents. Delegation is **description-driven** — the LLM reads agent descriptions and intelligently routes tasks to the appropriate sub-agent.

#### Rich Tool Ecosystem

ADK has one of the most extensive tool ecosystems of any agent framework:

- **Function Tools**: Any standard function can become a tool. ADK automatically inspects the function signature (name, docstring, parameters, type hints) and generates a schema.
- **Long-Running Function Tools**: For tasks requiring significant processing time (e.g., human approval workflows).
- **Agent-as-a-Tool**: Any agent can be wrapped as a tool — enables cross-framework interoperability (wrap LangGraph, CrewAI, or other ADK agents as tools).
- **MCP (Model Context Protocol) Tools**: Full support for the open MCP standard. ADK can act as an MCP Client or Server.
- **Pre-built Google Cloud Integrations**: BigQuery, Bigtable, Spanner, Pub/Sub, Google Search, Vertex AI Search, Vertex AI RAG Engine, and more.
- **Third-Party Integrations**: GitHub, GitLab, Notion, Linear, Asana, Atlassian, Stripe, PayPal, ElevenLabs, Hugging Face, and many more.
- **Observability Integrations**: AgentOps, Google Cloud Trace, Arize AX, Phoenix, MLflow, W&B Weave, and others.

#### Evaluation and Testing

ADK provides a built-in evaluation framework designed for the probabilistic nature of LLM agents:

**Two Approaches**:
1. **Test Files** (`.test.json`): Single-session, rapid-execution tests for development.
2. **Evalset Files** (`.evalset.json`): Multi-session, multi-turn integration tests with dynamic user simulation.

**Built-in Metrics**:
- `tool_trajectory_avg_score`: Exact match of expected vs. actual tool call sequences
- `response_match_score`: ROUGE-1 text similarity
- `final_response_match_v2`: LLM-judged semantic equivalence
- `hallucinations_v1`: Groundedness checking
- `safety_v1`: Safety verification

**Three Ways to Run**:
1. **Web UI** (`adk web`): Interactive, visual, with trace debugging
2. **Pytest Integration**: `await AgentEvaluator.evaluate(...)`
3. **CLI**: `adk eval <AGENT_MODULE_PATH> <EVAL_SET_PATH>`

#### Session, State, and Memory Management

- **Session**: A single ongoing interaction with a chronological sequence of events.
- **State** (`session.state`): Temporary key-value data scoped to the current session.
- **Memory**: Cross-session knowledge store, searchable archive spanning past sessions.

#### Callbacks and Lifecycle Hooks

Six callback points for observing and controlling agent execution:
- **Before/After Agent**: Hook into agent start and completion
- **Before/After Model**: Hook into LLM request and response
- **Before/After Tool**: Hook into tool invocation and result

Each callback can return `None` (normal execution) or a value to **override** default behavior.

#### Safety and Security

- Identity and Authorization (Agent-Auth and User-Auth patterns)
- Built-in Gemini Safety Filters
- Plugins: Gemini as a Judge, Model Armor, PII Redaction
- Sandboxed Code Execution
- VPC-SC Network Controls

#### Bidirectional Streaming

Low-latency voice and video interaction via the Gemini Live API (experimental):
- Real-time WebSocket communication
- Multimodal inputs (text, audio, video)
- Natural voice conversations with interrupt handling

#### Agent-to-Agent (A2A) Protocol

Open standard for inter-agent communication. Agents can act as both servers and clients, enabling distributed multi-agent systems across different frameworks and organizations.

### What Problems Does ADK Solve?

1. **Fragmented agent development**: Single, unified framework instead of stitching together multiple libraries.
2. **Lack of testability**: Built-in evaluation with trajectory and response quality metrics.
3. **Vendor lock-in**: Model-agnostic and deployment-agnostic design.
4. **Scalability of agent architectures**: Native multi-agent hierarchies with deterministic and dynamic orchestration.
5. **Tool integration complexity**: Massive pre-built tool ecosystem plus MCP support.
6. **Production readiness gap**: Same framework used in Google's own production products.
7. **Interoperability**: Agent-as-a-Tool pattern and A2A protocol for cross-framework collaboration.

---

## 2. ADK vs. Alternatives

### Overview

| Dimension | ADK | Dialogflow CX | LangGraph | CrewAI | LangChain | DIY |
|-----------|-----|---------------|-----------|--------|-----------|-----|
| **Primary Model** | Hierarchical agent composition | Visual flow/state machine | Explicit directed graph | Role-based crews + flows | Component chains + integrations | Whatever you build |
| **Code Required** | Yes (code-first) | No (visual editor) | Yes (graph definition) | Yes (Python + YAML) | Yes (Python/JS) | Yes |
| **Languages** | Python, TS, Go, Java | N/A (managed service) | Python, JS | Python only | Python, JS | Any |
| **Multi-Agent** | Native (sub_agents, workflow agents) | Limited (mega-agents) | Subgraphs, multi-node | Native (Crews, delegation) | Via LangGraph | Build yourself |
| **Cloud Deploy** | Vertex AI, Cloud Run, GKE, any container | Google Cloud managed | LangSmith or self-hosted | Self-hosted or enterprise | LangSmith or self-hosted | Build yourself |
| **Tool Ecosystem** | Function, MCP, OpenAPI, AgentTool | Webhooks | LangChain tools | CrewAI + LangChain tools | Largest integration library | Build yourself |
| **Evaluation** | Built-in (`adk eval`) | Built-in analytics | Via LangSmith | Via enterprise platform | Via LangSmith | Build yourself |
| **Dev Experience** | CLI + dev UI | Visual console | LangGraph Studio | CLI scaffolding | Notebooks + docs | Build yourself |
| **Interoperability** | MCP + A2A protocols | Telephony/messaging connectors | LangChain ecosystem | Standalone | Massive integration library | Whatever you integrate |
| **Best For** | Multi-agent systems with GCP | Structured conversation bots | Complex stateful workflows | Role-based team collaboration | Broad LLM applications | Unique requirements |

### ADK vs. No-Code Platforms (Dialogflow CX)

**Architecture**: Dialogflow CX uses a visual, flow-based design with Flows, Pages, Intents, and Entity Types — a deterministic state machine. ADK is code-first, where agents are defined programmatically and the LLM handles routing and reasoning.

**Ease of Use**: Dialogflow CX is accessible to non-developers (product managers, conversation designers). ADK requires developer skills but a working agent with tools is about 30 lines of Python.

**Flexibility**: Dialogflow CX is constrained by its state-machine paradigm. ADK is far more flexible — agents handle open-ended tasks, chain complex tool calls, and adapt dynamically.

**When to Choose**:
- **Dialogflow CX**: Structured conversation flows (support, IVR, FAQ), native telephony/messaging integrations, non-developer teams.
- **ADK**: LLM-powered reasoning (not just intent matching), dynamic tool/sub-agent composition, code-first with version control and testing.

### ADK vs. LangGraph

**Architecture**: LangGraph uses an explicit directed graph model (StateGraph with Nodes and Edges). You explicitly wire every transition. ADK uses hierarchical agent composition — you define agents with instructions, tools, and sub-agents, and the LLM decides routing.

```python
# LangGraph: explicit graph wiring
graph = StateGraph(State)
graph.add_node("agent", call_model)
graph.add_node("tools", call_tools)
graph.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
```

```python
# ADK: hierarchical composition
root_agent = Agent(
    name="coordinator",
    model="gemini-2.5-flash",
    instruction="Route requests to the appropriate specialist.",
    sub_agents=[research_agent, writing_agent, review_agent]
)
```

**Key Difference**: LangGraph makes control flow explicit in graph edges. ADK makes it implicit via LLM-driven delegation or explicit via workflow agents (Sequential, Parallel, Loop).

**When to Choose**:
- **LangGraph**: Fine-grained control over state transitions, complex conditional branching, visual graph-based reasoning, already in the LangChain/LangSmith ecosystem, mature checkpointing for long-running workflows.
- **ADK**: Higher-level composition model, built-in Google Cloud deployment, multi-language support, native MCP/A2A support, hierarchical delegation patterns.

### ADK vs. CrewAI

**Architecture**: CrewAI uses a role-based paradigm (role, goal, backstory) with Agents, Tasks, and Crews. ADK uses instruction-driven agents (name, model, instruction, description, tools, sub_agents).

```python
# CrewAI: persona-driven
researcher = Agent(
    role="Senior Data Researcher",
    goal="Uncover cutting-edge developments",
    backstory="You're a seasoned researcher with 10 years experience...",
    tools=[SerperDevTool()]
)
```

```python
# ADK: instruction-driven
researcher = Agent(
    name="researcher",
    model="gemini-2.5-flash",
    instruction="You are a research specialist. Search for information and provide detailed findings.",
    tools=[google_search]
)
```

**Key Difference**: CrewAI separates Tasks from Agents and encourages thinking about agents as team members with identities. ADK does not have a formal Task abstraction — work is defined by user queries and instructions.

**When to Choose**:
- **CrewAI**: Naturally modeled as team collaboration (content pipelines, research teams), persona-driven design, Python-only simplicity with YAML config, mature event-driven Flows system.
- **ADK**: Multi-language support, Google Cloud deployment, patterns beyond crew-based collaboration, MCP/A2A interoperability, built-in evaluation and dev UI.

### ADK vs. LangChain

**Architecture**: LangChain is a broad toolkit for LLM applications (chains, RAG, embeddings, agents). ADK is a focused agent framework. LangChain provides standardized interfaces across hundreds of providers; ADK focuses specifically on agent construction, composition, and deployment.

**Key Difference**: LangChain is a general-purpose LLM library. ADK is purpose-built for multi-agent systems. For advanced orchestration, LangChain now directs users to LangGraph.

**When to Choose**:
- **LangChain**: Broad LLM toolkit needs (RAG, document processing, embeddings), largest integration ecosystem, many different providers through standardized interfaces.
- **ADK**: Primary goal is multi-agent systems, focused framework without multiple abstraction layers, Google Cloud deployment, multi-language support.

### ADK vs. DIY / Custom-Built

Building from scratch means implementing: LLM interaction layer, tool execution engine, multi-agent orchestration, session/state management, deployment infrastructure, evaluation, and observability.

ADK provides all of these out of the box, including:
- Automatic schema generation from function signatures
- CLI + dev UI for running/debugging
- Built-in deployment patterns (Dockerfiles, Cloud Run, Agent Engine, GKE)
- Built-in evaluation framework

**When to Choose**:
- **DIY**: Very specific architectural requirements no framework can accommodate, minimize dependencies for regulatory reasons, building a fundamentally different paradigm.
- **ADK**: Focus on agent logic rather than infrastructure, production-quality session/state/evaluation without building it, MCP/A2A interoperability, supported framework with regular releases.

---

## 3. ADK in the GCP Stack

### Vertex AI Integration

ADK has full integration with Vertex AI. Key aspects:

- **Model Access**: Specify model name like `gemini-2.5-flash` and ADK resolves it to the Vertex AI backend. Set `GOOGLE_GENAI_USE_VERTEXAI=True` to switch from AI Studio to Vertex AI endpoints.
- **Vertex AI Hosted Models**: Any model deployed to a Vertex AI endpoint works with ADK.
- **Vertex AI Code Execution Sandbox**: Secure execution of agent-generated code.
- **Vertex AI Search**: Native integration for conversational search and RAG against private data stores.
- **Vertex AI RAG Engine**: Direct integration for retrieval-augmented generation.
- **Gen AI Evaluation Service**: Quality assessment and optimization.

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=us-central1
export GOOGLE_GENAI_USE_VERTEXAI=True
```

### Vertex AI Agent Engine

A managed service within Vertex AI that handles deployment, management, and scaling of AI agents in production. Internally uses the `ReasoningEngine` API resource.

**Sub-Services:**

| Service | Purpose |
|---------|---------|
| **Runtime** | Deploys and auto-scales agents with managed infrastructure, VPC-SC, IAM |
| **Sessions** | Stores individual user-agent interaction threads |
| **Memory Bank** | Long-term memory with semantic search across sessions |
| **Code Execution** (Preview) | Secure sandboxed execution of agent-generated code |
| **Example Store** (Preview) | Dynamic few-shot example retrieval |
| **Quality and Evaluation** (Preview) | Gen AI Evaluation service integration |
| **Observability** | Cloud Trace (OpenTelemetry), Cloud Monitoring, Cloud Logging |

**Security Features:**

| Feature | Runtime | Sessions | Memory Bank |
|---------|---------|----------|-------------|
| VPC Service Controls | Yes | Yes | Yes |
| CMEK | Yes | Yes | Yes |
| Data Residency (DRZ) | Yes | Yes | Yes |
| HIPAA | Yes | Yes | Yes |
| Access Transparency | Yes | Yes | Yes |

Agent Engine also includes **Threat Detection** (Preview) via Security Command Center.

**Deployment Paths:**
1. **Standard Deployment**: For existing GCP projects via Cloud Console and ADK CLI.
2. **Agent Starter Pack (ASP)**: Pre-built templates (ReAct, RAG, multi-agent), automated Terraform infrastructure, CI/CD via Cloud Build.

Currently, deployment to Agent Engine is **Python ADK only**.

### Model Flexibility

ADK is model-agnostic with two integration mechanisms:

| Provider | Wrapper | Example |
|----------|---------|---------|
| Gemini | Direct string | `model="gemini-2.5-flash"` |
| Vertex AI endpoints | Direct string | Endpoint resource string |
| OpenAI (GPT-4o, etc.) | `LiteLlm` | `model=LiteLlm(model="openai/gpt-4o")` |
| Anthropic (Claude) | `LiteLlm` | `model=LiteLlm(model="anthropic/claude-3-haiku-20240307")` |
| Ollama (local) | `LiteLlm` | Points to local Ollama server |
| 100+ LLMs via LiteLLM | `LiteLlm` | Cohere, Mistral, etc. |

```python
from google.adk.models.lite_llm import LiteLlm

agent = LlmAgent(
    model=LiteLlm(model="openai/gpt-4o"),
    name="my_agent",
    instruction="You are a helpful assistant."
)
```

### Google Cloud Services Integration

**Native Integrations:**
- **Data**: BigQuery, Bigtable, Spanner, Pub/Sub, AlloyDB, Cloud SQL
- **Search**: Vertex AI Search, Google Search, RAG Engine
- **Compute**: Cloud Run, GKE, Cloud Functions
- **DevOps**: Cloud Build, Artifact Registry, Terraform
- **Security**: Secret Manager, IAM, VPC-SC
- **Integration**: Application Integration, Apigee API Hub
- **Databases**: MCP Toolbox for Databases (30+ data sources)

### Memory and State in GCP

**Session State** — Key-value store with scoped prefixes:

| Prefix | Scope | Persistence |
|--------|-------|-------------|
| (none) | Current session | With persistent SessionService |
| `user:` | All sessions for a user | With Database/VertexAI services |
| `app:` | All users and sessions | With Database/VertexAI services |
| `temp:` | Current invocation only | Never persisted |

**Session Service Backends:**
- `InMemorySessionService`: Local testing
- `DatabaseSessionService`: Persistent (e.g., SQLite)
- `VertexAiSessionService`: Managed by Agent Engine

**Long-Term Memory:**
- `InMemoryMemoryService`: Basic keyword matching, for prototyping
- `VertexAiMemoryBankService`: LLM-powered extraction, semantic search

### Deployment Options

| Target | Description | Languages |
|--------|-------------|-----------|
| **Vertex AI Agent Engine** | Fully managed, auto-scaling | Python only |
| **Cloud Run** | Serverless containers | Python, TS, Go, Java |
| **GKE** | Managed Kubernetes, full control | Python, TS, Go, Java |
| **Any Container** | Docker/Podman on any host | All |
| **Local** | `adk web`, `adk run`, `adk api_server` | All |

```bash
# Cloud Run deployment
adk deploy cloud_run --project=$PROJECT --region=$REGION ./my_agent

# GKE deployment
adk deploy gke --project=$PROJECT --cluster_name=my-cluster --region=$REGION ./my_agent
```

### Authentication and Security

- **Agent-Auth**: Tools operate under the agent's service account identity.
- **User-Auth**: Tools operate under the user's OAuth identity.
- **IAM**: Full GCP IAM integration across Agent Engine, GKE (Workload Identity Federation), and Cloud Run.
- **VPC-SC**: Agent Engine supports VPC Service Controls perimeters.
- **Safety Plugins**: Gemini as Judge, Model Armor, PII Redaction.

### Monitoring and Observability

**Built-in (Agent Engine):**
- Google Cloud Trace (OpenTelemetry)
- Cloud Monitoring
- Cloud Logging

**Third-Party Integrations:**
- AgentOps, Arize AX / Phoenix, Monocle, MLflow, W&B Weave, Freeplay, BigQuery Agent Analytics

### A2A (Agent-to-Agent) Protocol

Open standard developed by Google, donated to the Linux Foundation. Enables agents built on different frameworks to communicate securely.

- **MCP**: Agent-to-tool communication (how agents connect to tools).
- **A2A**: Agent-to-agent communication (how agents talk to each other).

Philosophy: **"Build with ADK (or any framework), equip with MCP (or any tool), and communicate with A2A."**

Currently experimental, available in Python and Go.

### Overall Architecture

```
                              DEVELOPMENT
                              ===========
                    Developer
                        |
                +-------v--------+
                |   ADK SDK      |     (Python / TypeScript / Go / Java)
                |  - LlmAgent    |
                |  - Tools       |
                |  - Workflows   |
                |  - Multi-Agent |
                +-------+--------+
                        |
            +-----------+-----------+
            |           |           |
       adk web     adk run    adk api_server
       (Dev UI)    (CLI)      (FastAPI)


                              MODELS
                              ======
                +-------------------------------------------+
                |  Gemini (direct)  |  Vertex AI endpoints  |
                |  LiteLLM ---------+-- OpenAI / Anthropic  |
                |  Ollama / vLLM ---+-- Local / self-hosted |
                +-------------------------------------------+


                              DEPLOYMENT
                              ==========
    +-------------------+  +------------------+  +------------------+
    | Vertex AI Agent   |  | Cloud Run        |  | GKE              |
    | Engine            |  |                  |  |                  |
    | - Managed runtime |  | - Serverless     |  | - Kubernetes     |
    | - Sessions svc    |  | - Auto-scaling   |  | - Full control   |
    | - Memory Bank     |  | - FastAPI+Uvicorn|  | - Workload ID    |
    | - Code Execution  |  |                  |  |                  |
    | - Evaluation      |  |                  |  |                  |
    +--------+----------+  +--------+---------+  +--------+---------+
             |                      |                      |
             +----------------------+----------------------+
                                    |
                          GCP INFRASTRUCTURE
                          ==================
    +---------------------------------------------------------------+
    |  IAM  |  VPC-SC  |  CMEK  |  Secret Manager                  |
    +---------------------------------------------------------------+
    |  Cloud Trace  |  Cloud Logging  |  Cloud Monitoring           |
    +---------------------------------------------------------------+
    |  BigQuery | Spanner | Bigtable | Pub/Sub | Cloud SQL          |
    +---------------------------------------------------------------+
    |  Vertex AI Search | RAG Engine | Google Search                |
    +---------------------------------------------------------------+


                              COMMUNICATION
                              =============
    +---------------------+          +---------------------+
    |  Agent A (ADK)      | <--A2A-->|  Agent B (any       |
    |  (Client/Server)    |          |  framework)         |
    +---------------------+          +---------------------+
            |
            +-- MCP --> Tools, APIs, Resources


                              END USERS
                              =========
    +---------------------------------------------------------------+
    |  Web Apps  |  Mobile  |  Chat  |  Voice  |  API               |
    +---------------------------------------------------------------+
```

---

## Sources

- [ADK Official Documentation](https://google.github.io/adk-docs/)
- [ADK Python GitHub Repository](https://github.com/google/adk-python)
- [Google Developers Blog: Agent Development Kit](https://developers.googleblog.com/en/agent-development-kit-easy-to-build-multi-agent-applications/)
- [ADK Tools & Integrations](https://google.github.io/adk-docs/integrations/)
- [ADK Evaluation](https://google.github.io/adk-docs/evaluate/)
- [ADK Sessions & Memory](https://google.github.io/adk-docs/sessions/)
- [ADK Safety](https://google.github.io/adk-docs/safety/)
- [ADK A2A Protocol](https://google.github.io/adk-docs/a2a/)
- [ADK Deployment](https://google.github.io/adk-docs/deploy/)
- [Vertex AI Agent Engine Overview](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview)
- [ADK Models Documentation](https://google.github.io/adk-docs/agents/models/)
- [A2A Protocol](https://a2a-protocol.org/latest/)
- [LangGraph Repository](https://github.com/langchain-ai/langgraph)
- [CrewAI Documentation](https://docs.crewai.com/)
- [Dialogflow CX Docs](https://cloud.google.com/dialogflow/cx/docs/concept/agent)
