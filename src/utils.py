from pathlib import Path
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import logging
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, MODEL_NAME
from src.model import generate_embeddings
from src.config import DOCS_FOLDER
import fitz # PyMuPDF library
import time
import json
import tiktoken
import re
from contextlib import contextmanager


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(BASE_DIR / "data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
print("DATA_DIR = ", DATA_DIR)

CHUNK_FILE = DATA_DIR / "chunks.npy"
EMBEDDINGS_FILE = DATA_DIR / "chunks_embeddings.npy"


text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,  # chunk size (characters)
        chunk_overlap=CHUNK_OVERLAP,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )

def load_documents() -> None:
    """Load all PDF documents, build embeddings, and save index."""
    pdf_files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith(".pdf")]

    if not pdf_files:
        msg = (f"No PDF files found in {DOCS_FOLDER}") 
        logger.error(msg)
        return []

    all_chunks = []
    all_embeddings = []
    for file in pdf_files:
        file_path: Path = Path(os.path.join(DOCS_FOLDER, file))
        print(file_path)

        # chunk_text(file_path)
        chunks, embeddings = build_embeddings(file_path)
        all_chunks.extend(chunks)
        all_embeddings.extend(embeddings)

    if all_embeddings:
        all_embeddings = np.vstack(all_embeddings)
        np.save(EMBEDDINGS_FILE, all_embeddings)
        np.save(CHUNK_FILE, np.array(all_chunks))
        msg = f"Chunks and embeddings saved to: {DATA_DIR}"
        logger.info(msg)

    logger.info("Chunks and embeddings saved to: %s", DATA_DIR)
    

def chunk_text(path: Path) -> list[dict]:
    """Split PDF into text chunks with filename and page numbers."""
    file_name: str = path.stem
    chunks: list[dict] = []

    text = ""
    # Open a PDF document
    doc = fitz.open(path)

    # Iterate through pages and extract text
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text() # Extract text as UTF-8

        if text.strip():
            page_chunks = text_splitter.split_text(text)
            for chunk_text in page_chunks:
                chunks.append({
                    "text": chunk_text,
                    "source": file_name,
                    "page": page_number,
                })

    doc.close()
    return chunks


def build_embeddings(file_path: Path) -> tuple[list[dict], np.ndarray]:
    """Generate normalized embeddings for all chunks in a PDF."""
    chunks = chunk_text(file_path)
    texts = [chunk["text"] for chunk in chunks]

    embeddings = generate_embeddings(texts)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    return chunks, embeddings


def load_index() -> tuple[np.ndarray, np.ndarray]:
    """Load saved chunks and embeddings from disk."""
    chunks = np.load(CHUNK_FILE, allow_pickle=True)
    embeddings = np.load(EMBEDDINGS_FILE)
    logger.info("Loaded saved chunks and embeddings from disk.")
    return chunks, embeddings

def save_index(chunks: list, embeddings: list): 
    """ Save text chunks and embeddings to disk.
    
        chunks: list of dicts with keys {text, source, page} 
        embeddings: numpy array of shape (N, D) """ 
    # Convert chunks list → numpy array of objects 
    chunk_array = np.array(chunks, dtype=object) 
    # Save embeddings 
    np.save(EMBEDDINGS_FILE, embeddings) 
    # Save chunks (requires pickle) 
    np.save(CHUNK_FILE, chunk_array, allow_pickle=True) 
    logger.info(f"Chunks and embeddings saved to: {DATA_DIR}")

@contextmanager  
def timer():
    """Context manager for measuring elapsed time within a code block."""
    start = time.perf_counter()
    
    def elapsed() -> float:
        return time.perf_counter() - start
    
    yield elapsed
    
def count_tokens(text: str, model=MODEL_NAME) -> int:
    """Count the number of tokens in a text string for a given model."""
    if not isinstance(text, str):
        text = json.dumps(text, default=str)
    
    try:
        enc = tiktoken.encoding_for_model(model)
    except (KeyError, ValueError):
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def parse_source_info(answer: str) -> dict: 
    """ Extracts source, page, and chunk from an answer string. 
    Example pattern: [source: technical-manual-w11663204-revb, page: 32, chunk: 33] """ 
    pattern =r"\[source:\s*([^,\]]+),\s*page:\s*(\d+),\s*chunk:\s*(\d+)\]?$"
    match = re.search(pattern, answer, re.IGNORECASE) 
    if not match: 
        return None 
    source, page, chunk = match.groups() 
    return { "source": source.strip(), "page": int(page), "chunk": int(chunk) }