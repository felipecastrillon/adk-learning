from google.adk.agents import Agent
from google.adk.tools import google_search

root_agent = Agent(
    name="search_agent",
    model="gemini-2.5-flash",
    description="Searches the web for information.",
    instruction="You are a web search specialist. Use Google Search to find information and provide concise answers.",
    tools=[google_search],
)
