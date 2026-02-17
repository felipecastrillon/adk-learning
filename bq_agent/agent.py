from google.adk.agents import Agent
from google.adk.tools.bigquery.bigquery_toolset import BigQueryToolset

root_agent = Agent(
    name="bigquery_agent",
    model="gemini-2.5-flash",
    description="Queries BigQuery datasets.",
    instruction=(
        "You are a data analyst assistant with access to BigQuery.\n"
        "The dataset `agentspace-testing-471714.adk_learning.shakespeare` contains "
        "Shakespeare's works with columns: word, word_count, corpus, corpus_date.\n"
        "Always explain your findings clearly."
    ),
    tools=[BigQueryToolset()],
)
