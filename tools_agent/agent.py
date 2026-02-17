from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.tools.bigquery.bigquery_toolset import BigQueryToolset

# Sub-agent dedicated to Google Search (single-tool limitation workaround)
search_agent = Agent(
    name="search_agent",
    model="gemini-2.5-flash",
    description="Searches the web for information. Delegate web search queries to this agent.",
    instruction="You are a web search specialist. Use Google Search to find information and provide concise answers.",
    tools=[google_search],
)

# Root agent with BigQuery + search delegation
root_agent = Agent(
    name="tools_agent",
    model="gemini-2.5-flash",
    description="Agent with BigQuery and web search capabilities.",
    instruction=(
        "You are a data analyst assistant with access to BigQuery and web search.\n"
        "- For web search queries, delegate to the search_agent.\n"
        "- For data queries, use BigQuery tools to query datasets.\n"
        "- The public dataset `bigquery-public-data.samples.shakespeare` contains "
        "Shakespeare's works with columns: word, word_count, corpus, corpus_date.\n"
        "- Always explain your findings clearly."
    ),
    tools=[BigQueryToolset()],
    sub_agents=[search_agent],
)
