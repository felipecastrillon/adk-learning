# ADK Learning

This repository contains a hands-on guide to building AI agents with Google's Agent Development Kit (ADK).

## Prerequisites

- Python 3.10 or later
- A Google Cloud project with Vertex AI API enabled (recommended) or a Google AI Studio API key.
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed and authenticated (`gcloud auth application-default login`) if using Vertex AI.

## Setup

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd adk-learning
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**

    This project requires `google-adk` and Jupyter Notebook.

    ```bash
    pip install google-adk notebook
    ```

4.  **Set up environment variables:**

    Copy the example environment file to `.env`:

    ```bash
    cp .env.example .env
    ```

    Open `.env` and fill in your configuration:

    *   **For Vertex AI (Recommended):**
        *   Set `GOOGLE_GENAI_USE_VERTEXAI=TRUE`
        *   Set `GOOGLE_CLOUD_PROJECT` to your Google Cloud project ID.
        *   Set `GOOGLE_CLOUD_LOCATION` (e.g., `us-central1`).
    *   **For Google AI Studio:**
        *   Set `GOOGLE_GENAI_USE_VERTEXAI=FALSE`
        *   Set `GOOGLE_API_KEY` to your API key.

## Running the Tutorial

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `adk_tutorial.ipynb` and follow the instructions within the notebook to learn about building agents with ADK.

> **Note:** The notebook contains cells with hardcoded configuration (e.g., `GOOGLE_CLOUD_PROJECT="agentspace-testing-471714"`). Be sure to update these values in the notebook cells to match your own configuration defined in your `.env` file.
