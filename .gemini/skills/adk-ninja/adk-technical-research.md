# Google Agent Development Kit (ADK) — Technical Research

## Table of Contents

1. [Getting Started & Project Structure](#1-getting-started--project-structure)
2. [Agent Object & Agent Types](#2-agent-object--agent-types)
3. [Instructions](#3-instructions)
4. [Tools & Tool Types](#4-tools--tool-types)
5. [Callbacks & Callback Types](#5-callbacks--callback-types)
6. [Context Management (Session, State, Memory)](#6-context-management-session-state-memory)
7. [Runner and Events](#7-runner-and-events)
8. [Deployment](#8-deployment)

---

## 1. Getting Started & Project Structure

### Installation

**Prerequisites**: Python 3.10 or later.

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate.bat     # Windows CMD
# .venv\Scripts\Activate.ps1     # Windows PowerShell

# Install ADK
pip install google-adk

# Verify
pip show google-adk
```

### API Key Configuration (`.env` file)

The `.env` file lives **inside the agent directory** (e.g., `my_agent/.env`).

**Option A — Google AI Studio (simplest):**
```
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=PASTE_YOUR_API_KEY_HERE
```

**Option B — Vertex AI:**
```
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
GOOGLE_CLOUD_LOCATION=LOCATION
```
Also requires: `gcloud auth application-default login`

**Option C — Vertex AI Express Mode:**
```
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_API_KEY=PASTE_YOUR_EXPRESS_MODE_API_KEY_HERE
```

### Creating a Project

```bash
adk create my_agent
```

Generates:

```
my_agent/
    __init__.py      # Must import agent module
    agent.py         # Agent definition (must define root_agent)
    .env             # API keys / project configuration
```

### File Details

**`__init__.py`** — Required, must import the agent module:
```python
from . import agent
```

**`agent.py`** — Must define a variable called `root_agent` (ADK's discovery entry point):
```python
from google.adk.agents import Agent

root_agent = Agent(
    name="my_agent",
    model="gemini-3-flash",
    description="A simple helpful assistant.",
    instruction="You are a helpful assistant. Answer user questions.",
)
```

### Minimal Agent with a Tool

```python
from google.adk.agents import Agent

def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The city name.
    """
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": "The weather in New York is sunny, 25°C."
        }
    return {
        "status": "error",
        "error_message": f"Weather information for '{city}' is not available."
    }

root_agent = Agent(
    name="weather_agent",
    model="gemini-3-flash",
    description="Agent to answer questions about weather.",
    instruction="You are a helpful agent who can answer questions about weather.",
    tools=[get_weather],
)
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `adk create <name>` | Scaffold a new agent project |
| `adk run <agent_path>` | Interactive terminal chat with your agent |
| `adk web` | Browser-based dev UI (not for production) |
| `adk api_server` | RESTful API with Swagger docs at `/docs` |
| `adk eval <agent_path> <eval_path>` | Run agent evaluations |
| `adk deploy cloud_run` | Deploy to Cloud Run (see [Deployment](#8-deployment)) |
| `adk deploy gke` | Deploy to GKE (see [Deployment](#8-deployment)) |

#### `adk run` Options

```bash
adk run my_agent                                  # Basic run
adk run --save_session my_agent                   # Save session on exit
adk run --resume my_agent/session.json my_agent   # Resume previous session
adk run --replay input.json my_agent              # Replay from input file
echo "List files" | adk run file_agent            # Pipe input
```

Replay input file format:
```json
{
  "state": {"key": "value"},
  "queries": ["What is 2 + 2?", "What is the capital of France?"]
}
```

#### `adk web` Options

```bash
adk web                                    # Default: localhost:8000
adk web --port 3000                        # Custom port
adk web --session_service_uri "sqlite:///sessions.db"
```

Run from the **parent directory** containing the agent folder. Features: chat interface, session management, state inspection, event history, eval tab, trace tab.

#### `adk api_server` Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/list-apps` | List available agents |
| `POST` | `/apps/{app}/users/{user}/sessions/{session}` | Create session |
| `POST` | `/run` | Run agent (single JSON response) |
| `POST` | `/run_sse` | Run agent (Server-Sent Events stream) |

Request body (uses **camelCase**):
```json
{
  "appName": "my_agent",
  "userId": "u_123",
  "sessionId": "s_abc",
  "newMessage": {
    "role": "user",
    "parts": [{"text": "What is the capital of France?"}]
  }
}
```

### Model Selection & Generation Config

```python
from google.genai.types import GenerateContentConfig

agent = Agent(
    name="my_agent",
    model="gemini-3-flash",
    instruction="...",
    generate_content_config=GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=1024,
        top_p=0.9,
        top_k=40,
    ),
)
```

---

## 2. Agent Object & Agent Types

### BaseAgent

The abstract base class for all agents. Extends Pydantic's `BaseModel`.

**Key properties:**

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Must be a valid Python identifier, unique in agent tree |
| `description` | `str` | Used for routing in multi-agent systems |
| `sub_agents` | `list[BaseAgent]` | Child agents |
| `parent_agent` | `Optional[BaseAgent]` | Set automatically by the framework |
| `before_agent_callback` | `Optional` | Callback before execution |
| `after_agent_callback` | `Optional` | Callback after execution |

**Key methods:**

```python
async def run_async(parent_context) -> AsyncGenerator[Event, None]   # Public entry
async def _run_async_impl(ctx) -> AsyncGenerator[Event, None]        # Override this
async def _run_live_impl(ctx) -> AsyncGenerator[Event, None]         # For streaming

def find_agent(name) -> Optional[BaseAgent]     # Search self + descendants
def find_sub_agent(name) -> Optional[BaseAgent]  # Search descendants only
root_agent -> BaseAgent                           # Property: root of tree
```

### LlmAgent (aliased as `Agent`)

The primary LLM-powered agent. `Agent` and `LlmAgent` are the same class.

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Unique identifier (avoid `"user"`) |
| `model` | `Union[str, BaseLlm]` | `''` | e.g., `"gemini-3-flash"` |
| `description` | `str` | `''` | Used by other agents for routing |
| `instruction` | `Union[str, InstructionProvider]` | `''` | Static string or callable |
| `tools` | `list[ToolUnion]` | `[]` | Functions, BaseTool, AgentTool |
| `sub_agents` | `list[BaseAgent]` | `[]` | Child agents for delegation |
| `output_key` | `str` | `None` | Auto-saves response to session state |
| `output_schema` | | `None` | Enforces JSON output (disables tools) |
| `include_contents` | | `'default'` | `'default'` = history; `'none'` = stateless |
| `planner` | `BasePlanner` | `None` | `BuiltInPlanner` or `PlanReActPlanner` |
| `code_executor` | `BaseCodeExecutor` | `None` | e.g. `BuiltInCodeExecutor()` |
| `generate_content_config` | `Optional` | `None` | Temperature, max_tokens, etc. |

### Workflow Agents

Deterministic orchestration without LLM routing decisions.

#### SequentialAgent

Executes sub-agents in order. Shared `InvocationContext` — earlier agents write state via `output_key`, later agents read via `{variable}` templates.

```python
from google.adk.agents import Agent, SequentialAgent

code_writer = Agent(
    name="CodeWriter",
    model="gemini-3-flash",
    instruction="Write Python code for: {topic}",
    output_key="generated_code",
)

code_reviewer = Agent(
    name="CodeReviewer",
    model="gemini-3-flash",
    instruction="Review this code:\n{generated_code}\nProvide feedback.",
    output_key="review_comments",
)

code_refactorer = Agent(
    name="CodeRefactorer",
    model="gemini-3-flash",
    instruction="Refactor based on feedback.\nCode: {generated_code}\nFeedback: {review_comments}",
    output_key="refactored_code",
)

pipeline = SequentialAgent(
    name="CodePipeline",
    sub_agents=[code_writer, code_reviewer, code_refactorer],
)
```

#### ParallelAgent

Executes all sub-agents concurrently. No sharing of conversation history between branches during execution (they share `session.state`). Event ordering may be non-deterministic.

```python
from google.adk.agents import Agent, SequentialAgent, ParallelAgent

researcher_a = Agent(name="ResearcherA", model="gemini-3-flash",
    instruction="Research renewable energy.", output_key="renewable_result")
researcher_b = Agent(name="ResearcherB", model="gemini-3-flash",
    instruction="Research electric vehicles.", output_key="ev_result")

parallel_research = ParallelAgent(
    name="ParallelResearch",
    sub_agents=[researcher_a, researcher_b],
)

synthesizer = Agent(name="Synthesizer", model="gemini-3-flash",
    instruction="Combine:\n1. {renewable_result}\n2. {ev_result}")

workflow = SequentialAgent(
    name="ResearchWorkflow",
    sub_agents=[parallel_research, synthesizer],
)
```

#### LoopAgent

Repeatedly executes sub-agents in sequence until a termination condition is met.

**Termination conditions:**
1. `max_iterations` parameter (hard limit)
2. Sub-agent calls `tool_context.actions.escalate = True`
3. External logic

```python
from google.adk.agents import Agent, LoopAgent

def exit_loop(tool_context: ToolContext):
    """Call when the critique indicates no further changes needed."""
    tool_context.actions.escalate = True
    return {}

critic = Agent(name="Critic", model="gemini-3-flash",
    instruction="Review: {current_doc}. If good, say 'No major issues found.'",
    output_key="critique")

refiner = Agent(name="Refiner", model="gemini-3-flash",
    instruction="If critique says no issues, call exit_loop. Otherwise improve: {current_doc}",
    tools=[exit_loop], output_key="current_doc")

loop = LoopAgent(
    name="RefinementLoop",
    sub_agents=[critic, refiner],
    max_iterations=5,
)
```

### Custom Agents

Extend `BaseAgent` directly for arbitrary orchestration logic.

```python
class StoryFlowAgent(BaseAgent):
    story_generator: LlmAgent
    critic: LlmAgent
    reviser: LlmAgent

    async def _run_async_impl(self, ctx):
        # Step 1: Generate
        async for event in self.story_generator.run_async(ctx):
            yield event

        # Step 2: Critic-reviser loop
        async for event in self.loop_agent.run_async(ctx):
            yield event

        # Step 3: Conditional logic
        tone = ctx.session.state.get("tone_result")
        if tone == "negative":
            async for event in self.story_generator.run_async(ctx):
                yield event
```

### Multi-Agent Hierarchies

#### Delegation Mechanisms

**A) LLM-Driven (AutoFlow)** — Default when `sub_agents` are present. The parent LLM generates `transfer_to_agent(agent_name='target')` based on sub-agent `description` fields.

```python
coordinator = LlmAgent(
    name="Coordinator",
    model="gemini-3-flash",
    instruction="Route: billing issues to Billing, technical to Support.",
    sub_agents=[
        LlmAgent(name="Billing", description="Handles billing inquiries.", ...),
        LlmAgent(name="Support", description="Handles technical support.", ...),
    ],
)
```

**B) Agent-as-Tool (AgentTool)** — Wrap a target agent as a tool for explicit, synchronous invocation.

```python
from google.adk.tools import agent_tool

image_tool = agent_tool.AgentTool(agent=image_agent)
artist = LlmAgent(name="Artist", tools=[image_tool], ...)
```

**C) Shared Session State** — Simplest: agents write to and read from `session.state`.

#### Common Patterns

| Pattern | Structure | Use Case |
|---------|-----------|----------|
| Coordinator/Dispatcher | Central LlmAgent + sub_agents | Route requests to specialists |
| Sequential Pipeline | SequentialAgent | Ordered processing steps |
| Parallel Fan-Out/Gather | ParallelAgent → Synthesizer | Independent concurrent tasks |
| Generator-Critic | SequentialAgent | Quality improvement |
| Iterative Refinement | LoopAgent | Repeated improvement until convergence |

---

## 3. Instructions

### How Instructions Work

The `instruction` parameter is sent to the LLM as system instruction content. It defines the agent's core task, persona, constraints, tool usage guidance, and output format. Template variables are resolved against session state before each LLM call.

### Static vs Dynamic Instructions

**Static (plain string):**

```python
agent = Agent(
    name="greeter",
    instruction="You are a helpful assistant. The user's name is {user_name}.",
)
```

**Dynamic (callable function):**

```python
def dynamic_instruction(ctx: ReadonlyContext) -> str:
    user_role = ctx.state.get("user_role", "guest")
    if user_role == "admin":
        return "You are an admin assistant with full access to all tools."
    return "You are a guest assistant. Only answer general questions."

agent = Agent(
    name="adaptive_agent",
    model="gemini-3-flash",
    instruction=dynamic_instruction,   # callable, not a string
)
```

The `InstructionProvider` type alias:
```python
InstructionProvider: TypeAlias = Callable[[ReadonlyContext], Union[str, Awaitable[str]]]
```

**ReadonlyContext properties** available in dynamic instruction functions:

| Property | Description |
|----------|-------------|
| `ctx.state` | Read-only view of session state |
| `ctx.user_content` | The user content that started this invocation |
| `ctx.invocation_id` | Current invocation ID |
| `ctx.agent_name` | Name of the currently running agent |
| `ctx.session` | The current session object |

### Template Variables

| Syntax | Description |
|--------|-------------|
| `{var}` | Inserts `session.state['var']` (error if missing) |
| `{var?}` | Optional — silently ignores if missing |
| `{artifact.var}` | Inserts text content of artifact named `var` |

```python
# Agent A writes to state
agent_a = Agent(name="Validator", instruction="Validate input.",
    output_key="validation_status")

# Agent B reads from state
agent_b = Agent(name="Processor",
    instruction="Process only if {validation_status} is 'valid'.")

# Agent C uses optional variable
agent_c = Agent(name="Reporter",
    instruction="Generate report. Previous result: {result?}")
```

### Instruction Best Practices

1. **Be clear and specific** — avoid ambiguity
2. **Use Markdown** — headings, lists, emphasis for readability
3. **Provide examples** — few-shot learning for complex tasks
4. **Guide tool use explicitly** — explain *when* and *why* to use each tool
5. **Define scope boundaries** — state what the agent should NOT do
6. **Reference sub-agents by name** — in coordinator instructions

### Global Instructions

The `global_instruction` parameter is **deprecated**. Use `GlobalInstructionPlugin` instead. Sub-agents do NOT inherit their parent's `instruction` — each agent's instruction is independent.

---

## 4. Tools & Tool Types

### Function Tools

Regular functions become tools automatically when passed to the `tools` list. ADK inspects name, docstring, parameters, type hints, and defaults to generate a schema.

```python
def get_weather(city: str, unit: str = "Celsius") -> dict:
    """Retrieves weather for a city.

    Args:
        city (str): The city name.
        unit (str): Temperature unit, 'Celsius' or 'Fahrenheit'.
    """
    return {"status": "success", "report": f"Sunny in {city}, 25°{unit[0]}"}

agent = Agent(model="gemini-3-flash", tools=[get_weather])
```

**Parameter rules:**
- No default value = **required** parameter
- With default value = **optional** parameter
- `*args` and `**kwargs` are ignored by schema generator
- Prefer `str`, `int`, `float`, `bool` over complex types

**Return types:** Preferred return is `dict`. Non-dict returns auto-wrapped as `{"result": value}`. Include a `"status"` key.

### Long-Running Function Tools

For async operations (human-in-the-loop, external API calls):

```python
from google.adk.tools import LongRunningFunctionTool

def ask_for_approval(purpose: str, amount: float) -> dict:
    """Ask for approval for the reimbursement."""
    return {'status': 'pending', 'approver': 'Manager', 'ticket-id': 'ticket-1'}

long_running_tool = LongRunningFunctionTool(func=ask_for_approval)
```

Resume with updated response when the external process completes:
```python
updated_response = long_running_function_response.model_copy(deep=True)
updated_response.response = {'status': 'approved'}
runner.run_async(session_id=session.id,
    new_message=Content(parts=[Part(function_response=updated_response)]))
```

### Agent-as-a-Tool

Wrap agents as callable tools. The tool agent runs independently with its own session. Parent receives only the final result.

```python
from google.adk.tools import AgentTool

summarizer = LlmAgent(model="gemini-3-flash", name="summarizer",
    instruction="Summarize the provided text concisely")
summarize_tool = AgentTool(agent=summarizer)

main_agent = LlmAgent(model="gemini-3-flash", tools=[summarize_tool])
```

**Cross-framework interop (LangGraph):**

```python
from google.adk.agents import LangGraphAgent

lang_agent = LangGraphAgent(graph=compiled_graph, instruction="You are helpful")
lang_tool = AgentTool(agent=lang_agent)
main_agent = LlmAgent(model="gemini-3-flash", tools=[lang_tool])
```

### MCP Tools (Model Context Protocol)

Connect to MCP servers via `McpToolset`:

**Stdio Transport (local process):**
```python
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command='npx',
            args=["-y", "@modelcontextprotocol/server-filesystem", "/path"],
        ),
    ),
    tool_filter=['read_file', 'list_directory']   # Optional: select specific tools
)
```

**SSE/HTTP Transport (remote server):**
```python
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams

McpToolset(
    connection_params=SseConnectionParams(
        url="https://remote-mcp-server.com/sse",
        headers={"Authorization": "Bearer token"}
    )
)
```

**Using McpToolset in agents:**
```python
root_agent = LlmAgent(
    model='gemini-3-flash',
    name='mcp_agent',
    tools=[
        McpToolset(connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command='npx',
                args=["-y", "@modelcontextprotocol/server-google-maps"],
                env={"GOOGLE_MAPS_API_KEY": api_key}
            )
        ))
    ]
)
```

### OpenAPI Tools

Auto-generate tools from OpenAPI specs:
```python
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

toolset = OpenAPIToolset(spec_str=openapi_spec_json, spec_str_type="json")
agent = LlmAgent(name="api_agent", model="gemini-3-flash", tools=[toolset])
```

### Built-in Tools

| Tool | Import | Notes |
|------|--------|-------|
| Google Search | `from google.adk.tools import google_search` | Gemini 2+ only |
| Code Execution | `BuiltInCodeExecutor()` via `code_executor=` | Gemini 2+ only |
| Vertex AI Search | `VertexAiSearchTool(data_store_id=...)` | Private data stores |
| BigQuery | `BigQueryToolset(...)` | 7 tools (SQL, forecast, insights) |
| Computer Use | `ComputerUseToolset(...)` | Browser automation via Playwright |

**Single-tool limitation**: Google Search, Code Execution, and Vertex AI Search cannot be combined with other tools in a single agent (pre-v1.16.0). Workaround: use `AgentTool` to wrap each in a separate agent.

### ToolContext

Available to tool functions at runtime:

| Property | Description |
|----------|-------------|
| `state` | Mutable session state dictionary |
| `function_call_id` | ID of the specific LLM function call |
| `actions` | Access to `EventActions` (escalate, transfer, etc.) |
| `save_artifact(name, part)` | Save binary data |
| `load_artifact(name)` | Load binary data |
| `list_artifacts()` | List all artifacts |
| `search_memory(query)` | Query memory service |
| `request_credential(auth_config)` | Trigger auth flow |
| `get_auth_response(auth_config)` | Retrieve credentials |

```python
def my_tool(query: str, tool_context: ToolContext) -> dict:
    # Read/write state
    tool_context.state["temp:last_query"] = query

    # Save artifact
    tool_context.save_artifact("report.txt", types.Part(text="Report content"))

    # Search memory
    results = tool_context.search_memory(f"info about {query}")
    return {"status": "success"}
```

### Tool Authentication

**API Key:**
```python
from google.adk.tools.openapi_tool.auth.auth_helpers import token_to_scheme_credential

auth_scheme, auth_credential = token_to_scheme_credential(
    "apikey", "query", "apikey", "YOUR_API_KEY")
toolset = OpenAPIToolset(spec_str="...", auth_scheme=auth_scheme, auth_credential=auth_credential)
```

**OAuth2:**
```python
from fastapi.openapi.models import OAuth2, OAuthFlowAuthorizationCode, OAuthFlows

auth_scheme = OAuth2(flows=OAuthFlows(
    authorizationCode=OAuthFlowAuthorizationCode(
        authorizationUrl="https://accounts.google.com/o/oauth2/auth",
        tokenUrl="https://oauth2.googleapis.com/token",
        scopes={"https://www.googleapis.com/auth/calendar": "calendar"}
    )))
```

### Action Confirmation Pattern

```python
# Simple boolean
FunctionTool(reimburse, require_confirmation=True)

# Dynamic threshold
async def threshold(amount: int, tool_context: ToolContext) -> bool:
    return amount > 1000

FunctionTool(reimburse, require_confirmation=threshold)
```

### Tool Performance: Parallel Execution

As of v1.10.0, ADK runs tools in parallel when possible. Use `async` functions for I/O-bound tools:

```python
async def get_weather(city: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://api.weather.com/{city}") as resp:
            return await resp.json()
```

Synchronous tools **block** parallel execution for all tools.

### Summary of Tool Types

| Tool Type | Class | Use Case |
|-----------|-------|----------|
| Function Tool | `FunctionTool` | Regular functions |
| Long-Running | `LongRunningFunctionTool` | Async ops, human-in-the-loop |
| Agent-as-Tool | `AgentTool` | Wrap agents as callable tools |
| MCP Tool | `McpToolset` | Connect to MCP servers |
| OpenAPI Tool | `OpenAPIToolset` | Auto-generate from API specs |
| Google Search | `google_search` | Web search |
| Code Execution | `BuiltInCodeExecutor` | Run Python code |
| Vertex AI Search | `VertexAiSearchTool` | Search private data |
| BigQuery | `BigQueryToolset` | SQL, forecasting, insights |
| Computer Use | `ComputerUseToolset` | Browser automation |

---

## 5. Callbacks & Callback Types

Callbacks hook into an agent's execution at predefined points. Return `None` to proceed normally, or return a value to **override** default behavior.

### Summary Table

| Callback | Parameters | Return None | Return Value |
|----------|-----------|-------------|--------------|
| `before_agent_callback` | `CallbackContext` | Agent runs normally | `Content` — skip agent |
| `after_agent_callback` | `CallbackContext` | Use agent's output | `Content` — replace output |
| `before_model_callback` | `CallbackContext, LlmRequest` | LLM call proceeds | `LlmResponse` — skip LLM |
| `after_model_callback` | `CallbackContext, LlmResponse` | Use LLM response | `LlmResponse` — replace response |
| `before_tool_callback` | `CallbackContext, ToolCall` | Tool runs normally | `dict` — skip tool |
| `after_tool_callback` | `CallbackContext, dict` | Use tool result | `dict` — replace result |

### before_agent_callback / after_agent_callback

```python
def check_if_agent_should_run(callback_context: CallbackContext) -> Optional[types.Content]:
    if callback_context.state.get("skip_llm_agent", False):
        return types.Content(
            parts=[types.Part(text=f"Agent {callback_context.agent_name} skipped.")],
            role="model"
        )
    return None   # Proceed normally

def modify_output_after_agent(callback_context: CallbackContext) -> Optional[types.Content]:
    if callback_context.state.get("add_note", False):
        return types.Content(
            parts=[types.Part(text="Note added by callback.")], role="model")
    return None
```

### before_model_callback / after_model_callback

**Guardrail pattern — blocking unsafe input:**
```python
def guardrail_before_model(
    callback_context: CallbackContext,
    llm_request: LlmRequest
) -> Optional[LlmResponse]:
    last_message = llm_request.contents[-1].parts[0].text
    if "BLOCK" in last_message.upper():
        return LlmResponse(
            content=types.Content(role="model",
                parts=[types.Part(text="I cannot process this request.")]))
    return None
```

**Caching pattern:**
```python
def caching_before_model(callback_context, llm_request) -> Optional[LlmResponse]:
    cache_key = llm_request.contents[-1].parts[0].text
    cached = my_cache.get(cache_key)
    if cached:
        return LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=cached)]))
    return None
```

**Request modification — injecting system instruction:**
```python
def inject_prefix(callback_context, llm_request) -> Optional[LlmResponse]:
    instruction = llm_request.config.system_instruction
    instruction.parts[0].text = "[Modified] " + (instruction.parts[0].text or "")
    return None   # Continue with modified request
```

### before_tool_callback / after_tool_callback

```python
def before_tool(callback_context: CallbackContext, tool_call: ToolCall) -> Optional[dict]:
    print(f"Tool: {tool_call.name}, Args: {tool_call.args}")
    return None   # Execute normally

def after_tool(callback_context: CallbackContext, tool_result: dict) -> Optional[dict]:
    tool_result["processed_by_callback"] = True
    return None   # Use original (now mutated) result
```

### Wiring Callbacks

```python
my_agent = LlmAgent(
    name="MyAgent",
    model="gemini-3-flash",
    instruction="You are a helpful assistant.",
    before_agent_callback=check_if_agent_should_run,
    after_agent_callback=modify_output_after_agent,
    before_model_callback=guardrail_before_model,
    after_model_callback=None,
    before_tool_callback=before_tool,
    after_tool_callback=after_tool,
)
```

---

## 6. Context Management (Session, State, Memory)

### Session

A `Session` represents a single ongoing interaction.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique conversation identifier |
| `app_name` | `str` | Which agent application owns this |
| `user_id` | `str` | Links to a particular user |
| `events` | `list[Event]` | Chronological sequence of interactions |
| `state` | `dict` | Key-value store for conversation data |
| `last_update_time` | `float` | Timestamp of last activity |

```python
from google.adk.sessions import InMemorySessionService

session_service = InMemorySessionService()
session = await session_service.create_session(
    app_name="my_app",
    user_id="user_123",
    state={"initial_key": "initial_value"}
)
```

### SessionService Implementations

| Service | Backend | Persistence | Use Case |
|---------|---------|-------------|----------|
| `InMemorySessionService` | In-memory | Lost on restart | Local testing |
| `DatabaseSessionService` | SQL (SQLite, PostgreSQL, MySQL) | Persistent | Self-hosted production |
| `VertexAiSessionService` | Vertex AI Agent Engine | Managed persistent | GCP production |

```python
# SQLite
DatabaseSessionService(db_url="sqlite+aiosqlite:///./sessions.db")

# PostgreSQL
DatabaseSessionService(db_url="postgresql+asyncpg://user:pass@host/db")

# Vertex AI
VertexAiSessionService(project="your-project", location="us-central1")
```

### State

`session.state` is a key-value dictionary with scoped prefixes:

| Prefix | Scope | Persistence |
|--------|-------|-------------|
| *(none)* | Current session only | Session lifetime |
| `user:` | All sessions for that user | With DB/Vertex services |
| `app:` | All users and sessions | With DB/Vertex services |
| `temp:` | Current invocation only | Never persisted |

**Writing state:**

1. **Via `output_key`** — auto-saves agent response:
   ```python
   Agent(name="Greeter", output_key="last_greeting", ...)
   ```

2. **Via `ToolContext.state`** (recommended in tools):
   ```python
   def my_tool(tool_context: ToolContext):
       tool_context.state["user_action_count"] = count + 1
       tool_context.state["temp:last_status"] = "success"
   ```

3. **Via `EventActions.state_delta`**:
   ```python
   actions = EventActions(state_delta={"task_status": "active"})
   event = Event(invocation_id="inv_1", actions=actions)
   await session_service.append_event(session, event)
   ```

**Never directly mutate `session.state`** on a retrieved session — it bypasses event tracking.

### Memory

Long-term knowledge across sessions. Distinct from State (per-session).

| Service | Backend | Features |
|---------|---------|----------|
| `InMemoryMemoryService` | In-memory | Basic keyword matching, for prototyping |
| `VertexAiMemoryBankService` | Vertex AI | LLM-powered extraction, semantic search |

**Workflow:**
1. User interacts → session captured
2. `memory_service.add_session_to_memory(session)` → extracts and stores
3. Later session → agent queries with `search_memory(query)`
4. Returns `SearchMemoryResponse` with `MemoryResult` objects

**Built-in memory tools:**
- `PreloadMemoryTool` — always retrieves memory at start of each turn
- `LoadMemoryTool` — retrieves on-demand when agent decides

```python
from google.adk.tools.preload_memory_tool import PreloadMemoryTool

agent = Agent(
    model="gemini-3-flash",
    name="my_agent",
    instruction="Answer using past knowledge.",
    tools=[PreloadMemoryTool()]
)
```

### Artifacts

Named, versioned binary data (files, images, audio) associated with sessions.

```python
from google.adk.artifacts import InMemoryArtifactService
# or: GcsArtifactService(bucket_name="your-bucket")

artifact_service = InMemoryArtifactService()
runner = Runner(agent=agent, artifact_service=artifact_service)
```

**Operations (via ToolContext/CallbackContext):**
```python
# Save (returns version number)
version = await context.save_artifact(filename="report.pdf", artifact=pdf_part)

# Load latest
artifact = await context.load_artifact(filename="report.pdf")

# Load specific version
artifact = await context.load_artifact(filename="report.pdf", version=0)

# List all
filenames = await context.list_artifacts()
```

**Namespacing:** Default is session-scoped. Prefix with `user:` for user-scoped access across sessions.

---

## 7. Runner and Events

### Runner

The central orchestrator of agent execution. Manages the event loop, coordinates agents, LLMs, tools, and services.

```python
from google.adk.runners import Runner

runner = Runner(
    agent=my_agent,                        # Required: root agent
    app_name="my_application",             # Required: application identifier
    session_service=session_service,       # Required: SessionService
    memory_service=memory_service,         # Optional: MemoryService
    artifact_service=artifact_service,     # Optional: ArtifactService
)
```

**Runner loop (conceptual):**
1. Append user query to session as Event
2. Call `agent.run_async(context)` → starts agent event generator
3. For each yielded Event:
   - Commit `state_delta`, `artifact_delta` via SessionService
   - Yield event upstream to application/UI
   - Agent resumes only after Runner finishes processing

### Event

An immutable record representing an occurrence during agent execution.

**Key fields:**

| Field | Type | Description |
|-------|------|-------------|
| `content` | `Optional[Content]` | Message payload (text, function calls) |
| `partial` | `Optional[bool]` | `True` for streaming chunks |
| `author` | `str` | `'user'` or agent name |
| `invocation_id` | `str` | Unique ID for this interaction run |
| `id` | `str` | Unique event ID |
| `timestamp` | `float` | Creation time |
| `actions` | `EventActions` | Side-effects and control signals |
| `error_code` | `Optional[str]` | e.g., `"SAFETY_FILTER_TRIGGERED"` |

**EventActions:**

```python
event.actions.state_delta           # dict: key-value pairs to merge into state
event.actions.artifact_delta        # dict: {filename: version}
event.actions.transfer_to_agent     # str: agent name to transfer to
event.actions.escalate              # bool: terminate the event loop
event.actions.skip_summarization    # bool: skip LLM summary of tool results
```

### Detecting Event Types

```python
async for event in runner.run_async(...):
    # Tool call request
    if event.get_function_calls():
        for fc in event.get_function_calls():
            print(f"Tool: {fc.name}({fc.args})")

    # Tool result
    elif event.get_function_responses():
        for fr in event.get_function_responses():
            print(f"Result: {fr.name} -> {fr.response}")

    # Final response
    elif event.is_final_response():
        if event.content and event.content.parts:
            print(f"Answer: {event.content.parts[0].text}")

    # Streaming chunk
    elif event.partial:
        print(f"Chunk: {event.content.parts[0].text}")
```

### `is_final_response()` Returns True When:

1. Tool result with `skip_summarization == True`, OR
2. Long-running tool call, OR
3. **All of**: no function calls, no function responses, not partial, no pending code execution

### run_async — Programmatic Invocation

```python
from google.genai.types import Content, Part

# Create session
await session_service.create_session(app_name="my_app", user_id="user123")

# Create message
user_message = Content(parts=[Part(text="What is the weather in Paris?")], role="user")

# Run agent — async generator
final_response = ""
async for event in runner.run_async(
    user_id="user123",
    session_id="session456",
    new_message=user_message
):
    if event.is_final_response() and event.content and event.content.parts:
        final_response = event.content.parts[0].text

print(final_response)
```

### Content and Parts

Messages use `Content` (container) and `Part` (individual piece):

```python
from google.genai.types import Content, Part

# User message
Content(parts=[Part(text="Hello")], role="user")

# Model response
Content(parts=[Part(text="Hi there!")], role="model")

# Function call (from LLM)
Part(function_call=FunctionCall(name="get_weather", args={"city": "Paris"}))

# Function response (tool result)
Part(function_response=FunctionResponse(name="get_weather", response={"temp": 22}))

# Binary data
Part.from_bytes(data=image_bytes, mime_type="image/png")
```

### Context Object Hierarchy

| Context | Available In | Key Capabilities |
|---------|-------------|------------------|
| `InvocationContext` | Agent `_run_async_impl` | Full access: session, agent, services |
| `ReadonlyContext` | Instruction providers | Read-only: state, invocation_id, agent_name |
| `CallbackContext` | All callbacks | Read/write state, artifacts, user_content |
| `ToolContext` | Tool functions | Everything in Callback + function_call_id, actions, auth, memory |

### RunConfig

```python
from google.genai.adk import RunConfig, StreamingMode

config = RunConfig(
    streaming_mode=StreamingMode.NONE,    # NONE, SSE, or BIDI
    max_llm_calls=100,                    # Limit total LLM calls (default 500)
    save_input_blobs_as_artifacts=False,
)
```

---

## 8. Deployment

### Overview

| Target | Description | Languages | Use Case |
|--------|-------------|-----------|----------|
| **Local** | `adk web`, `adk run`, `adk api_server` | All | Development & testing |
| **Vertex AI Agent Engine** | Fully managed, auto-scaling | Python only | Production, zero-ops |
| **Cloud Run** | Serverless containers | All | Production, moderate control |
| **GKE** | Managed Kubernetes | All | Production, full control |
| **Any Container** | Docker/Podman on any host | All | On-premise, air-gapped |

### Cloud Run

```bash
adk deploy cloud_run \
    --project=$GOOGLE_CLOUD_PROJECT \
    --region=$GOOGLE_CLOUD_LOCATION \
    --service_name=$SERVICE_NAME \
    --with_ui \
    $AGENT_PATH
```

| Flag | Description |
|------|-------------|
| `--project TEXT` | GCP project ID (required) |
| `--region TEXT` | Deployment location (required) |
| `--service_name TEXT` | Cloud Run service name (optional) |
| `--app_name TEXT` | Application name (defaults to directory name) |
| `--with_ui` | Deploy dev UI alongside API server |
| `--port INTEGER` | Container port (default: 8000) |
| `--agent_engine_id TEXT` | Vertex AI Agent Engine resource ID |

Pass additional gcloud flags after `--`:
```bash
adk deploy cloud_run --project=my-proj --region=us-central1 my_agent \
    -- --no-allow-unauthenticated --min-instances=2
```

The container runs FastAPI + Uvicorn.

### GKE (Google Kubernetes Engine)

```bash
adk deploy gke \
    --project=$GOOGLE_CLOUD_PROJECT \
    --cluster_name=my-cluster \
    --region=$GOOGLE_CLOUD_LOCATION \
    $AGENT_PATH
```

Requires enabling: `container.googleapis.com`, `artifactregistry.googleapis.com`, `cloudbuild.googleapis.com`, `aiplatform.googleapis.com`.

Supports Autopilot clusters with Workload Identity Federation bound to GCP IAM roles (e.g., `roles/aiplatform.user`).

### Vertex AI Agent Engine

Fully managed service within Vertex AI. Handles infrastructure, scaling, governance. Internally uses the `ReasoningEngine` API resource.

**Sub-services included:**

| Service | Purpose |
|---------|---------|
| Runtime | Deploys and auto-scales agents with VPC-SC, IAM |
| Sessions | Stores user-agent interaction threads |
| Memory Bank | Long-term memory with semantic search |
| Code Execution (Preview) | Secure sandboxed code execution |
| Example Store (Preview) | Dynamic few-shot example retrieval |
| Evaluation (Preview) | Gen AI Evaluation service integration |
| Observability | Cloud Trace, Cloud Monitoring, Cloud Logging |

**Deployment paths:**
1. **Standard**: Via Cloud Console and ADK CLI for existing GCP projects
2. **Agent Starter Pack (ASP)**: Pre-built templates (ReAct, RAG, multi-agent), Terraform infrastructure, CI/CD via Cloud Build

**Limitation**: Currently **Python ADK only**.

**Security features**: VPC-SC, CMEK, Data Residency (DRZ), HIPAA, Access Transparency, Threat Detection (Preview).

### Custom Container Deployment

For any container-compatible infrastructure. Use the Cloud Run Dockerfile and FastAPI entry point as templates:

```python
import os
import uvicorn
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))

app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    session_service_uri="sqlite+aiosqlite:///./sessions.db",
    allow_origins=["*"],
    web=True,
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
```

### Authentication for Deployed Agents

- **Agent-Auth**: Tools operate under the agent's service account identity
- **User-Auth**: Tools operate under the user's OAuth identity
- **Cloud Run**: Service accounts + Secret Manager for API keys
- **GKE**: Kubernetes service accounts + Workload Identity Federation + GCP IAM roles
- **Agent Engine**: Full IAM integration with IAM Conditions for Sessions and Memory Bank

---

## Sources

- [ADK Official Documentation](https://google.github.io/adk-docs/)
- [ADK Quickstart](https://google.github.io/adk-docs/get-started/quickstart/)
- [ADK Installation](https://google.github.io/adk-docs/get-started/installation/)
- [ADK LLM Agents](https://google.github.io/adk-docs/agents/llm-agents/)
- [ADK Multi-Agents](https://google.github.io/adk-docs/agents/multi-agents/)
- [ADK Function Tools](https://google.github.io/adk-docs/tools-custom/function-tools/)
- [ADK MCP Tools](https://google.github.io/adk-docs/tools-custom/mcp-tools/)
- [ADK OpenAPI Tools](https://google.github.io/adk-docs/tools-custom/openapi-tools/)
- [ADK Callbacks](https://google.github.io/adk-docs/callbacks/)
- [ADK Sessions](https://google.github.io/adk-docs/sessions/)
- [ADK State](https://google.github.io/adk-docs/sessions/state/)
- [ADK Memory](https://google.github.io/adk-docs/sessions/memory/)
- [ADK Events](https://google.github.io/adk-docs/events/)
- [ADK Runtime/Event Loop](https://google.github.io/adk-docs/runtime/event-loop/)
- [ADK Context](https://google.github.io/adk-docs/context/)
- [ADK Artifacts](https://google.github.io/adk-docs/artifacts/)
- [ADK Python GitHub](https://github.com/google/adk-python)
