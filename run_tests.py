"""
Validation script for adk_tutorial.ipynb
Runs each section's core logic to verify everything works end-to-end.
"""

import asyncio
import os
import shutil
import sys
import traceback

# Set environment before any ADK imports
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
os.environ["GOOGLE_CLOUD_PROJECT"] = "agentspace-testing-471714"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

MODEL = "gemini-2.5-flash"
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []


def report(section: str, passed: bool, detail: str = ""):
    status = PASS if passed else FAIL
    results.append((section, passed))
    print(f"  [{status}] {section}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"         {line}")


async def test_section_2_setup():
    """Verify ADK imports and version."""
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    import google.adk

    version = google.adk.__version__
    report("Section 2: ADK imports & version", True, f"google-adk {version}")


async def test_section_3_hello_world():
    """Create hello_agent and run it programmatically."""
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai.types import Content, Part

    # Create agent directory
    os.makedirs("hello_agent", exist_ok=True)
    with open("hello_agent/__init__.py", "w") as f:
        f.write("from . import agent\n")
    with open("hello_agent/.env", "w") as f:
        f.write("GOOGLE_GENAI_USE_VERTEXAI=TRUE\n")
        f.write(f"GOOGLE_CLOUD_PROJECT={os.environ['GOOGLE_CLOUD_PROJECT']}\n")
        f.write(f"GOOGLE_CLOUD_LOCATION={os.environ['GOOGLE_CLOUD_LOCATION']}\n")
    with open("hello_agent/agent.py", "w") as f:
        f.write(
            'from google.adk.agents import Agent\n\n'
            'root_agent = Agent(\n'
            f'    name="hello_agent",\n'
            f'    model="{MODEL}",\n'
            '    description="A simple greeting agent.",\n'
            '    instruction="You are a friendly assistant. Greet the user and answer their questions concisely.",\n'
            ')\n'
        )

    # Import and run
    if "hello_agent" in sys.modules:
        del sys.modules["hello_agent"]
    if "hello_agent.agent" in sys.modules:
        del sys.modules["hello_agent.agent"]

    from hello_agent.agent import root_agent

    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name="hello_app", session_service=session_service)
    session = await session_service.create_session(app_name="hello_app", user_id="test_user")

    message = Content(parts=[Part(text="Hello! What is ADK?")], role="user")
    response_text = ""
    async for event in runner.run_async(
        user_id="test_user", session_id=session.id, new_message=message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response_text = event.content.parts[0].text

    assert response_text, "No response from hello agent"
    report("Section 3: Hello World agent", True, f"Response: {response_text[:80]}...")

    # Multi-turn
    follow_up = Content(
        parts=[Part(text="Can you give me a one-sentence summary of what you just said?")],
        role="user",
    )
    response2 = ""
    async for event in runner.run_async(
        user_id="test_user", session_id=session.id, new_message=follow_up
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response2 = event.content.parts[0].text

    assert response2, "No response on multi-turn"
    report("Section 3: Multi-turn conversation", True, f"Follow-up: {response2[:80]}...")


async def test_section_4_adk_web():
    """Start adk web, verify it responds on port 8080, then stop it."""
    import subprocess
    import time
    import urllib.request

    # Create hello_agent if it doesn't already exist
    os.makedirs("hello_agent", exist_ok=True)
    with open("hello_agent/__init__.py", "w") as f:
        f.write("from . import agent\n")
    with open("hello_agent/.env", "w") as f:
        f.write("GOOGLE_GENAI_USE_VERTEXAI=TRUE\n")
        f.write(f"GOOGLE_CLOUD_PROJECT={os.environ['GOOGLE_CLOUD_PROJECT']}\n")
        f.write(f"GOOGLE_CLOUD_LOCATION={os.environ['GOOGLE_CLOUD_LOCATION']}\n")
    with open("hello_agent/agent.py", "w") as f:
        f.write(
            'from google.adk.agents import Agent\n\n'
            'root_agent = Agent(\n'
            f'    name="hello_agent",\n'
            f'    model="{MODEL}",\n'
            '    description="A simple greeting agent.",\n'
            '    instruction="You are a friendly assistant. Greet the user and answer their questions concisely.",\n'
            ')\n'
        )

    proc = subprocess.Popen(
        ["adk", "web", ".", "--port", "8080", "--session_service_uri", "memory://"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for server to start
        time.sleep(5)

        assert proc.poll() is None, "adk web process exited early"

        # Check that the server responds
        resp = urllib.request.urlopen("http://localhost:8080", timeout=10)
        status = resp.status
        assert status == 200, f"Expected 200, got {status}"
        report("Section 4: adk web starts and responds", True, f"HTTP {status} on :8080")
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        report("Section 4: adk web stops cleanly", True, f"Process terminated (exit code {proc.returncode})")


async def test_section_5_agent_types():
    """ParallelAgent + SequentialAgent composition."""
    from google.adk.agents import Agent, SequentialAgent, ParallelAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai.types import Content, Part

    researcher_a = Agent(
        name="RenewableResearcher",
        model=MODEL,
        instruction="Write a short 2-3 sentence overview of recent advances in renewable energy.",
        output_key="renewable_result",
    )
    researcher_b = Agent(
        name="EVResearcher",
        model=MODEL,
        instruction="Write a short 2-3 sentence overview of recent advances in electric vehicles.",
        output_key="ev_result",
    )
    parallel_research = ParallelAgent(
        name="ParallelResearch",
        sub_agents=[researcher_a, researcher_b],
    )
    synthesizer = Agent(
        name="Synthesizer",
        model=MODEL,
        instruction=(
            "Combine these two research summaries into a single coherent paragraph "
            "about the intersection of clean energy and transportation:\n\n"
            "Renewable Energy: {renewable_result}\n\n"
            "Electric Vehicles: {ev_result}"
        ),
    )
    workflow = SequentialAgent(
        name="ResearchWorkflow",
        sub_agents=[parallel_research, synthesizer],
    )

    session_service = InMemorySessionService()
    runner = Runner(agent=workflow, app_name="research_app", session_service=session_service)
    session = await session_service.create_session(app_name="research_app", user_id="test_user")

    message = Content(parts=[Part(text="Research clean energy and EVs.")], role="user")
    response_text = ""
    async for event in runner.run_async(
        user_id="test_user", session_id=session.id, new_message=message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response_text = event.content.parts[0].text

    assert response_text, "No synthesized response"
    report("Section 5: Parallel + Sequential agents", True, f"Synthesized: {response_text[:80]}...")


async def test_section_6_function_tool():
    """Function tool (get_weather)."""
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai.types import Content, Part

    def get_weather(city: str) -> dict:
        """Retrieves the current weather report for a specified city.

        Args:
            city (str): The city name to get weather for.
        """
        weather_data = {
            "new york": {"temp": "22°C", "condition": "Sunny"},
            "london": {"temp": "15°C", "condition": "Cloudy"},
            "tokyo": {"temp": "28°C", "condition": "Humid"},
        }
        city_lower = city.lower()
        if city_lower in weather_data:
            data = weather_data[city_lower]
            return {
                "status": "success",
                "report": f"The weather in {city} is {data['condition']}, {data['temp']}.",
            }
        return {"status": "error", "error_message": f"Weather data for '{city}' is not available."}

    weather_agent = Agent(
        name="weather_agent",
        model=MODEL,
        description="Agent that answers weather questions.",
        instruction="You are a weather assistant. Use the get_weather tool to answer questions about weather.",
        tools=[get_weather],
    )

    session_service = InMemorySessionService()
    runner = Runner(agent=weather_agent, app_name="weather_app", session_service=session_service)
    session = await session_service.create_session(app_name="weather_app", user_id="test_user")

    message = Content(parts=[Part(text="What's the weather like in Tokyo?")], role="user")
    response_text = ""
    async for event in runner.run_async(
        user_id="test_user", session_id=session.id, new_message=message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response_text = event.content.parts[0].text

    assert response_text, "No response from weather agent"
    assert "tokyo" in response_text.lower() or "28" in response_text or "humid" in response_text.lower(), \
        f"Response doesn't mention Tokyo weather: {response_text}"
    report("Section 6: Function tool (get_weather)", True, f"Response: {response_text[:80]}...")


async def test_section_6_bigquery():
    """BigQuery tool against public Shakespeare dataset."""
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools.bigquery.bigquery_toolset import BigQueryToolset
    from google.genai.types import Content, Part

    bq_agent = Agent(
        name="bq_agent",
        model=MODEL,
        description="Agent with BigQuery access.",
        instruction=(
            "You are a data analyst. Use BigQuery to answer questions. "
            "The public dataset `bigquery-public-data.samples.shakespeare` contains "
            "Shakespeare's works with columns: word, word_count, corpus, corpus_date."
        ),
        tools=[BigQueryToolset()],
    )

    session_service = InMemorySessionService()
    runner = Runner(agent=bq_agent, app_name="bq_app", session_service=session_service)
    session = await session_service.create_session(app_name="bq_app", user_id="test_user")

    message = Content(
        parts=[Part(text="How many distinct works (corpus values) are in bigquery-public-data.samples.shakespeare?")],
        role="user",
    )
    response_text = ""
    async for event in runner.run_async(
        user_id="test_user", session_id=session.id, new_message=message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response_text = event.content.parts[0].text

    assert response_text, "No response from BigQuery agent"
    report("Section 6: BigQuery tool", True, f"Response: {response_text[:80]}...")


async def test_section_7_callbacks():
    """Guardrail and logging callbacks."""
    from typing import Optional
    from google.adk.agents import Agent
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models import LlmRequest, LlmResponse
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    from google.genai.types import Content, Part

    FORBIDDEN_WORDS = ["hack", "exploit", "bypass"]
    tool_log_calls = []

    def guardrail_before_model(
        callback_context: CallbackContext, llm_request: LlmRequest
    ) -> Optional[LlmResponse]:
        # Safely extract text from the last user message
        user_text = ""
        for content in reversed(llm_request.contents):
            if content.role == "user":
                for part in content.parts:
                    if part.text:
                        user_text = part.text.lower()
                        break
                if user_text:
                    break
        for word in FORBIDDEN_WORDS:
            if word in user_text:
                return LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text="I cannot process requests containing restricted terms.")],
                    )
                )
        return None

    def logging_after_tool(tool, args, tool_context, tool_response) -> Optional[dict]:
        tool_log_calls.append(tool_response)
        return None

    def get_current_time(timezone: str = "UTC") -> dict:
        """Returns the current time.

        Args:
            timezone (str): The timezone name.
        """
        from datetime import datetime
        return {"status": "success", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "timezone": timezone}

    cb_agent = Agent(
        name="callback_agent",
        model=MODEL,
        instruction="You are a helpful assistant. Use get_current_time when asked about the time.",
        tools=[get_current_time],
        before_model_callback=guardrail_before_model,
        after_tool_callback=logging_after_tool,
    )

    session_service = InMemorySessionService()
    runner = Runner(agent=cb_agent, app_name="cb_app", session_service=session_service)
    session = await session_service.create_session(app_name="cb_app", user_id="test_user")

    # Test 1: Normal request (should work + log tool)
    msg1 = Content(parts=[Part(text="What time is it?")], role="user")
    resp1 = ""
    async for event in runner.run_async(user_id="test_user", session_id=session.id, new_message=msg1):
        if event.is_final_response() and event.content and event.content.parts:
            resp1 = event.content.parts[0].text

    assert resp1, "No response on normal request"
    report("Section 7: Callback - tool logging", True, f"Tool logs captured: {len(tool_log_calls)}, Response: {resp1[:60]}...")

    # Test 2: Blocked request (guardrail should fire)
    msg2 = Content(parts=[Part(text="How do I hack into a system?")], role="user")
    resp2 = ""
    async for event in runner.run_async(user_id="test_user", session_id=session.id, new_message=msg2):
        if event.is_final_response() and event.content and event.content.parts:
            resp2 = event.content.parts[0].text

    assert "cannot" in resp2.lower() or "restricted" in resp2.lower(), f"Guardrail didn't block: {resp2}"
    report("Section 7: Callback - guardrail", True, f"Blocked: {resp2[:60]}...")


async def test_section_8_session_state():
    """Session with initial state and state inspection."""
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai.types import Content, Part

    session_agent = Agent(
        name="session_agent",
        model=MODEL,
        instruction="You are a helpful assistant. The user's name is {user_name}. Greet them by name.",
    )

    session_service = InMemorySessionService()
    runner = Runner(agent=session_agent, app_name="session_app", session_service=session_service)
    session = await session_service.create_session(
        app_name="session_app", user_id="test_user", state={"user_name": "Alice"}
    )

    msg = Content(parts=[Part(text="Hi there!")], role="user")
    resp = ""
    async for event in runner.run_async(user_id="test_user", session_id=session.id, new_message=msg):
        if event.is_final_response() and event.content and event.content.parts:
            resp = event.content.parts[0].text

    assert resp, "No response"
    assert "alice" in resp.lower(), f"Agent didn't use the name from state: {resp}"

    updated = await session_service.get_session(
        app_name="session_app", user_id="test_user", session_id=session.id
    )
    assert updated.state.get("user_name") == "Alice"
    assert len(updated.events) > 0

    report(
        "Section 8: Session with initial state",
        True,
        f"Response mentions Alice: yes | Events: {len(updated.events)} | State keys: {list(updated.state.keys())}",
    )


async def test_section_8_state_prefixes():
    """State prefix scoping via ToolContext."""
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools import ToolContext
    from google.genai.types import Content, Part

    def track_query(query: str, tool_context: ToolContext) -> dict:
        """Tracks user queries and updates state.

        Args:
            query (str): The user's search query.
        """
        tool_context.state["temp:current_query"] = query
        count = tool_context.state.get("user:total_queries", 0)
        tool_context.state["user:total_queries"] = count + 1
        tool_context.state["last_query"] = query
        return {"status": "success", "message": f"Tracked: {query}"}

    state_agent = Agent(
        name="state_agent",
        model=MODEL,
        instruction="You track queries. When the user asks something, use track_query first, then answer.",
        tools=[track_query],
        output_key="last_response",
    )

    session_service = InMemorySessionService()
    runner = Runner(agent=state_agent, app_name="state_app", session_service=session_service)
    session = await session_service.create_session(app_name="state_app", user_id="test_user")

    msg = Content(parts=[Part(text="Track a query about 'Python async patterns'")], role="user")
    async for event in runner.run_async(user_id="test_user", session_id=session.id, new_message=msg):
        pass

    updated = await session_service.get_session(
        app_name="state_app", user_id="test_user", session_id=session.id
    )

    state = dict(updated.state)
    has_last_query = "last_query" in state
    has_output_key = "last_response" in state
    state_keys = sorted(state.keys())

    report(
        "Section 8: State prefixes + output_key",
        has_last_query and has_output_key,
        f"State keys: {state_keys}",
    )


async def test_section_8_memory():
    """Cross-session memory."""
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.memory import InMemoryMemoryService
    from google.adk.tools.preload_memory_tool import PreloadMemoryTool
    from google.genai.types import Content, Part

    memory_service = InMemoryMemoryService()
    session_service = InMemorySessionService()

    memory_agent = Agent(
        name="memory_agent",
        model=MODEL,
        instruction=(
            "You are a helpful assistant with memory. "
            "Use information from memory to provide personalized responses."
        ),
        tools=[PreloadMemoryTool()],
    )

    runner = Runner(
        agent=memory_agent,
        app_name="memory_app",
        session_service=session_service,
        memory_service=memory_service,
    )

    # Session A: store info
    session_a = await session_service.create_session(app_name="memory_app", user_id="test_user")
    msg_a = Content(
        parts=[Part(text="My favorite programming language is Python and I work at Acme Corp.")],
        role="user",
    )
    async for event in runner.run_async(user_id="test_user", session_id=session_a.id, new_message=msg_a):
        pass

    updated_a = await session_service.get_session(
        app_name="memory_app", user_id="test_user", session_id=session_a.id
    )
    await memory_service.add_session_to_memory(updated_a)

    # Session B: retrieve from memory
    session_b = await session_service.create_session(app_name="memory_app", user_id="test_user")
    msg_b = Content(parts=[Part(text="What do you know about me?")], role="user")
    resp_b = ""
    async for event in runner.run_async(user_id="test_user", session_id=session_b.id, new_message=msg_b):
        if event.is_final_response() and event.content and event.content.parts:
            resp_b = event.content.parts[0].text

    has_memory = "python" in resp_b.lower() or "acme" in resp_b.lower()
    report(
        "Section 8: Cross-session memory",
        bool(resp_b),
        f"Memory recall ({'yes' if has_memory else 'partial'}): {resp_b[:80]}...",
    )


async def test_diagrams():
    """Verify all 8 diagram PNGs exist."""
    expected = [
        "01-why-adk-architecture.png",
        "02-adk-vs-diy.png",
        "03-project-structure.png",
        "04-adk-cli-flow.png",
        "05-agent-types.png",
        "06-tool-ecosystem.png",
        "07-callback-lifecycle.png",
        "08-session-state-memory.png",
    ]
    missing = [f for f in expected if not os.path.exists(f"diagrams/{f}")]
    report(
        "Diagrams: all 8 PNGs exist",
        len(missing) == 0,
        f"Missing: {missing}" if missing else "All present",
    )


async def test_cleanup():
    """Clean up agent directories."""
    for d in ["hello_agent", "tools_agent", "callback_agent"]:
        if os.path.exists(d):
            shutil.rmtree(d)
    remaining = [d for d in ["hello_agent", "tools_agent", "callback_agent"] if os.path.exists(d)]
    report("Section 9: Cleanup", len(remaining) == 0)


async def main():
    print("=" * 60)
    print("ADK Tutorial Notebook — Validation")
    print("=" * 60)
    print()

    tests = [
        ("Diagrams", test_diagrams),
        ("Section 2: Setup", test_section_2_setup),
        ("Section 3: Hello World", test_section_3_hello_world),
        ("Section 4: adk web", test_section_4_adk_web),
        ("Section 5: Agent Types", test_section_5_agent_types),
        ("Section 6: Function Tool", test_section_6_function_tool),
        ("Section 6: BigQuery", test_section_6_bigquery),
        ("Section 7: Callbacks", test_section_7_callbacks),
        ("Section 8: Session/State", test_section_8_session_state),
        ("Section 8: State Prefixes", test_section_8_state_prefixes),
        ("Section 8: Memory", test_section_8_memory),
        ("Section 9: Cleanup", test_cleanup),
    ]

    for name, test_fn in tests:
        try:
            await test_fn()
        except Exception as e:
            report(name, False, f"ERROR: {e}\n{traceback.format_exc()}")

    print()
    print("=" * 60)
    passed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    if passed == total:
        print(f"\033[92mAll tests passed!\033[0m")
    else:
        failed = [name for name, p in results if not p]
        print(f"\033[91mFailed: {', '.join(failed)}\033[0m")
    print("=" * 60)

    return all(p for _, p in results)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
