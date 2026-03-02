# Mechanic assistant Chat

A Retrieval-Augmented Generation (RAG) chatbot designed to assist field mechanics by providing accurate, source-grounded answers from technical PDF manuals.

## Project Overview

This project enables to ask question and receive answers strictly based on the content of uploaded manuals. It uses:

- **PDF text extraction**
- **Chunking with overlap**
- **Embedding generation**
- **Vectore similarity search**
- **Context-grounded LLM response with citation format**

**Key Technologies:**
- **LLM**: GPT-4o-mini (OpenAI) for response generation
- **Embeddings**: text-embedding-3-large (OpenAI) for semantic search
- **Document Processing**: PyMuPDF/pymupdf4llm for PDF text extraction
- **Vector Storage**: NumPy arrays for efficient similarity search
- **UI**: Streamlit for interactive web interface

---
### Prerequisites

- Python 3.9+
- OpenAI API key
- PDF technical manuals for washing machines

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/kaja98/mechanic_assistant.git
cd mechanic_assistant
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**

Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

4. **Add PDF documents:**

Place your PDF technical manuals in the `documents/` folder:
```bash
mkdir -p documents
cp /path/to/your/manuals/*.pdf documents/
```

### Running the Application

#### Option 1: Command Line Interface (CLI)

```bash
# Build the index (first time or when documents change)
python -m src.main

# The script will:
# 1. Load all PDFs from documents/ folder
# 2. Extract and chunk text
# 3. Generate embeddings
# 4. Save index to src/data/
# 5. Run a test query
```

#### Option 2: Streamlit Web Interface

```bash
streamlit run src/studio/app.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- Interactive chat interface
- PDF viewer with automatic page navigation
- Source citations with direct links to manual pages

---

## Architecture

### Component Overview

| Component | File | Purpose |
|-----------|------|---------|
| **Configuration** | [config.py](src/config.py) | Model settings, chunk sizes, retrieval parameters |
| **Document Processor** | [document_processor.py](src/document_processor.py) | PDF loading, text extraction, chunking |
| **Model Interface** | [model.py](src/model.py) | OpenAI API wrappers for embeddings and chat |
| **Utilities** | [utils.py](src/utils.py) | Index I/O, parsing, timing utilities |
| **Validation Metrics** | [validation_metrics.py](src/validation_metrics.py) | Scoring functions for retrieval quality |
| **Main Pipeline** | [main.py](src/main.py) | Orchestrates document processing and query answering |
| **Web UI** | [studio/app.py](src/studio/app.py) | Streamlit interface with chat and PDF viewer |

---

## Configuration

Edit [src/config.py](src/config.py) to customize behavior:

```python
# Model Configuration
MODEL_NAME = "gpt-4o-mini"              # LLM for response generation
EMBEDDING_MODEL = "text-embedding-3-small"  # Embedding model

# Chunking Configuration
CHUNK_SIZE = 4000                       # Characters per chunk
CHUNK_OVERLAP = 800                     # Overlap between chunks (20%)

# Retrieval Configuration
TOP_K_RESULT = 20                        # Number of chunks to retrieve
DOCS_FOLDER = "documents"               # Path to PDF documents
```
