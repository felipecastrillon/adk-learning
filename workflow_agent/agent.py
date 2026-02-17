from google.adk.agents import Agent, SequentialAgent, ParallelAgent

# Two researchers that run concurrently
researcher_a = Agent(
    name="RenewableResearcher",
    model="gemini-2.5-flash",
    instruction="Write a short 2-3 sentence overview of recent advances in renewable energy.",
    output_key="renewable_result",
)

researcher_b = Agent(
    name="EVResearcher",
    model="gemini-2.5-flash",
    instruction="Write a short 2-3 sentence overview of recent advances in electric vehicles.",
    output_key="ev_result",
)

# Synthesizer reads both results via template variables
synthesizer = Agent(
    name="Synthesizer",
    model="gemini-2.5-flash",
    instruction=(
        "Combine these two research summaries into a single coherent paragraph "
        "about the intersection of clean energy and transportation:\n\n"
        "Renewable Energy: {renewable_result}\n\n"
        "Electric Vehicles: {ev_result}"
    ),
)

parallel_research = ParallelAgent(
    name="ParallelResearch",
    sub_agents=[researcher_a, researcher_b],
)

# Sequential wraps: parallel stage → synthesizer
root_agent = SequentialAgent(
    name="ResearchWorkflow",
    sub_agents=[parallel_research, synthesizer],
)
