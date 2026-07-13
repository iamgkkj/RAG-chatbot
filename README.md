# Document Knowledge Base — Local RAG Assistant

A clean, production-grade Retrieval-Augmented Generation (RAG) chatbot designed to ingest, index, and query dense PDF documents natively. The system relies on LangChain for pipeline architecture, Google Gemini for semantic embeddings and text synthesis, and ChromaDB for local vector storage.

**Author:** Gopal krishn Khoth
**Registration Number:** 23BCE10669

---

## Key Features

* **Paced Ingestion Pipeline:** Self-throttling ingestion mechanism with automated backoff logic to safely stay within free-tier API Rate Limits (RPM).
* **Persistent Local Storage:** Contextual segments are vectorized and cached locally in an optimized Chroma DB structure to prevent redundant API calls.
* **Stateful UI Interface:** Seamless chat interface built with Streamlit, preserving active session states and message history.
* **Context Optimization:** Fine-tuned token character splitting ($1000$ chunk size, $150$ overlap) paired with an expanded retrieval window ($k=6$) to minimize text fragmentation and eliminate hallucinations.

---

## Directory Architecture

```text
├── data/               # Source PDFs go here for processing
├── .chroma_db/         # Generated persistent vector store database
├── .env                # Local environmental secrets
├── .gitignore          # Git exclusion framework
├── requirements.txt    # Application dependencies
├── ingest.py           # Document chunking & vector ingestion pipeline
└── app.py              # Stateful Streamlit application code

```

---

## Installation & Environment Configuration

### 1. Environment Isolation

Set up an isolated environment using `conda` to maintain dependency integrity:

```bash
# Create the environment
conda create -n rag-bot python=3.11 -y

# Activate the workspace
conda activate rag-bot

```

### 2. Dependency Resolution

Install the required architectural packages compiled in `requirements.txt`:

```bash
pip install -r requirements.txt

```

### 3. Environment Secrets Setup

Create a `.env` file in the root project directory and provide your API configuration:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here

```

---

## Practical Execution Flow

### Step 1: Ingesting Target Documents

1. Drop your sample PDF or text documentation directly inside the `data/` directory.
2. Fire up the paced indexing pipeline from your terminal:

```bash
python ingest.py

```

The script will track reading progress, map out distinct text chunks, compute semantic embedding vectors via `gemini-embedding-001`, and save them down inside the local storage bucket (`.chroma_db/`).

### Step 2: Launching the Interaction Interface

Once the data ingestion routine signals a success, boot up the main application:

```bash
streamlit run app.py

```

Your local browser environment will spin up a fresh tab at `http://localhost:8501`. You can now dynamically converse with your underlying document base.
