
import csv
import os
from google.cloud import bigquery

# Set project ID if not set (fallback to agentspace-testing-471714)
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "agentspace-testing-471714")

def copy_shakespeare_data():
    """Copies the BigQuery public dataset `samples.shakespeare` to a local CSV file."""
    
    print(f"Connecting to BigQuery using project: {project_id}")
    try:
        client = bigquery.Client(project=project_id)
        
        # Query the entire table
        query = """
            SELECT word, word_count, corpus, corpus_date 
            FROM `bigquery-public-data.samples.shakespeare`
        """
        
        print("Executing query (this might take a moment)...")
        query_job = client.query(query)
        rows = query_job.result()  # Waits for job to complete
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        output_file = "data/shakespeare.csv"
        
        print(f"Writing data to {output_file}...")
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(["word", "word_count", "corpus", "corpus_date"])
            
            count = 0
            for row in rows:
                writer.writerow([row.word, row.word_count, row.corpus, row.corpus_date])
                count += 1
                if count % 10000 == 0:
                    print(f"  Processed {count} rows...", end="\r")
            
        print(f"\nSuccessfully saved {count} rows to {output_file}")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Ensure you have authenticated with 'gcloud auth login' and set 'GOOGLE_CLOUD_PROJECT'.")

if __name__ == "__main__":
    copy_shakespeare_data()
