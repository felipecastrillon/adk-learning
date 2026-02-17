from google.adk.agents import Agent

root_agent = Agent(
    name="hello_agent",
    model="gemini-2.5-flash",
    description="A simple greeting agent.",
    instruction="You are a friendly assistant. Greet the user and answer their questions concisely.",
)
