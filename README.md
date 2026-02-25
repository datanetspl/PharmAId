# PharmAId

PharmAId is a full-stack medical AI assistant designed to provide insights into patient medications and potential drug interactions. It leverages large language models and a Retrieval-Augmented Generation (RAG) pipeline to deliver accurate and context-aware information.

## Features

- **Patient Medication Lookup**: Fetches and displays a patient's current medication list.
- **AI-Powered Drug Interaction Analysis**: Utilizes the `google/medgemma-4b-it` model to analyze and identify potential drug interactions.
- **Named Entity Recognition (NER)**: Employs the `tner/xlm-roberta-base-bc5cdr` model to recognize and extract medical entities from text.
- **RAG Pipeline for FDA Data**: Includes a RAG pipeline to ingest, process, and embed FDA drug label information into a ChromaDB vector store for enhanced information retrieval.

## Project Structure

The project is organized into three main components:

- **`frontend/`**: A React-based user interface built with Vite that allows users to interact with the application.
- **`backend/`**: A Flask API that serves the AI models, handles patient data, and performs the core logic of the application.
- **`RAG/`**: A set of Python scripts for building the Retrieval-Augmented Generation pipeline. These scripts download FDA drug data, process it, and create a vector database.

## Setup and Installation

### Backend

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set environment variables:**
    You will need a Hugging Face token to use the models.
    ```bash
    export hf_token="YOUR_HUGGING_FACE_TOKEN"
    ```
    You will also need to sign up for Epic FHIR app `https://fhir.epic.com/Specifications`.
    ```bash
    export CLIENT_ID="YOUR_EPIC_APP_ID"
    ```

4.  **Run the Flask server:**
    ```bash
    python flask_api.py
    ```
    The server will start on `http://localhost:8000`.

### Frontend

1.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```

2.  **Install npm dependencies:**
    ```bash
    npm install
    ```

3.  **Run the development server:**
    ```bash
    npm run dev
    ```
    The application will be available at `http://localhost:5173`.

### RAG Pipeline

1.  **Navigate to the RAG directory:**
    ```bash
    cd RAG
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download FDA data:**
    ```bash
    python fda_download.py
    ```

4.  **Create the vector store:**
    ```bash
    python fda_embed.py
    ```
    This will create a `fda_db` directory containing the ChromaDB vector store.

## Usage

1.  Ensure the backend and frontend servers are running.
2.  Open your web browser and navigate to `http://localhost:5173`.
3.  Use the patient lookup feature to retrieve medication information.
4.  The application will display the AI-generated analysis of potential drug interactions.
