from typing import Optional
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools import ToolContext
from google.genai import types

FORBIDDEN_WORDS = ["hack", "exploit", "bypass"]


def guardrail_before_model(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    """Block requests containing forbidden keywords."""
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
            print(f"[GUARDRAIL] Blocked request containing '{word}'")
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="I cannot process requests containing restricted terms.")],
                )
            )
    return None  # Proceed normally


def logging_after_tool(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
    tool_response: dict,
) -> Optional[dict]:
    """Log tool calls and their results."""
    print(f"[TOOL LOG] Tool: {tool.name}")
    print(f"[TOOL LOG] Args: {args}")
    print(f"[TOOL LOG] Result: {tool_response}")
    return None  # Don't modify the result


def get_current_time(timezone: str = "UTC") -> dict:
    """Returns the current time in the specified timezone.

    Args:
        timezone (str): The timezone name (e.g., 'UTC', 'US/Eastern').
    """
    from datetime import datetime
    return {"status": "success", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "timezone": timezone}


root_agent = Agent(
    name="callback_agent",
    model="gemini-2.5-flash",
    description="Agent with guardrail and logging callbacks.",
    instruction="You are a helpful assistant. Use the get_current_time tool when asked about the time.",
    tools=[get_current_time],
    before_model_callback=guardrail_before_model,
    after_tool_callback=logging_after_tool,
)
