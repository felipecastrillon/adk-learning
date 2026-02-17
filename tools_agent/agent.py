from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.tools.bigquery.bigquery_toolset import BigQueryToolset

# Google Search agent — google_search MUST be the only tool (API limitation)
search_agent = Agent(
    name="search_agent",
    model="gemini-2.5-flash",
    description="Searches the web for information.",
    instruction="You are a web search specialist. Use Google Search to find information and provide concise answers.",
    tools=[google_search],
)

# BigQuery agent — separate agent since google_search can't mix with other tools
bigquery_agent = Agent(
    name="bigquery_agent",
    model="gemini-2.5-flash",
    description="Queries BigQuery datasets.",
    instruction=(
        "You are a data analyst assistant with access to BigQuery.\n"
        "The public dataset `bigquery-public-data.samples.shakespeare` contains "
        "Shakespeare's works with columns: word, word_count, corpus, corpus_date.\n"
        "Always explain your findings clearly."
    ),
    tools=[BigQueryToolset()],
)

# Expose search_agent as the default for CLI usage
root_agent = search_agent
