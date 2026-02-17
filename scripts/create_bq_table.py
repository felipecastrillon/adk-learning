
import os
import sys
from google.cloud import bigquery
from google.api_core.exceptions import NotFound, GoogleAPICallError
import google.auth.exceptions

# Configuration
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "agentspace-testing-471714")
DATASET_ID = "adk_learning"
TABLE_ID = "shakespeare"
CSV_FILE_PATH = "data/shakespeare.csv"

def create_bq_table():
    if not os.path.exists(CSV_FILE_PATH):
        print(f"Error: Source file '{CSV_FILE_PATH}' not found.")
        print("Please ensure you have run the data copy step first.")
        return

    print(f"Using project: {PROJECT_ID}")
    
    try:
        client = bigquery.Client(project=PROJECT_ID)
    except Exception as e:
        print(f"Error initializing BigQuery client: {e}")
        print("Please run `gcloud auth application-default login` to authenticate.")
        return

    # 1. Create Dataset if it doesn't exist
    dataset_ref = client.dataset(DATASET_ID)
    try:
        client.get_dataset(dataset_ref)
        print(f"Dataset {DATASET_ID} already exists.")
    except NotFound:
        print(f"Dataset {DATASET_ID} not found. Creating...")
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"  # Adjust location as needed
        try:
            client.create_dataset(dataset)
            print(f"Dataset {DATASET_ID} created.")
        except GoogleAPICallError as e:
             print(f"Failed to create dataset: {e}")
             return

    # 2. Configure the Load Job
    table_ref = dataset_ref.table(TABLE_ID)
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,  # Skip header
        schema=[
            bigquery.SchemaField("word", "STRING"),
            bigquery.SchemaField("word_count", "INTEGER"),
            bigquery.SchemaField("corpus", "STRING"),
            bigquery.SchemaField("corpus_date", "INTEGER"),
        ],
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE, # Overwrite if exists
    )

    # 3. Load the Data
    print(f"Loading data from {CSV_FILE_PATH} into {DATASET_ID}.{TABLE_ID}...")
    try:
        with open(CSV_FILE_PATH, "rb") as source_file:
            job = client.load_table_from_file(source_file, table_ref, job_config=job_config)

        # Wait for the job to complete
        job.result() 

        print(f"Loaded {job.output_rows} rows into {DATASET_ID}.{TABLE_ID}.")
        
        # Verify
        table = client.get_table(table_ref)
        print(f"Table schema: {table.schema}")
        print(f"Total rows: {table.num_rows}")

    except Exception as e:
        print(f"Error loading data: {e}")
        if "403" in str(e):
             print("Please ensure you have authenticated with `gcloud auth application-default login`.")

if __name__ == "__main__":
    create_bq_table()
