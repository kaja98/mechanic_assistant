"""Utils."""
from pathlib import Path
import numpy as np
import logging
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, MODEL_NAME
import time
import json
import tiktoken
import re
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(BASE_DIR / "data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_FILE = DATA_DIR / "chunks.npy"
EMBEDDINGS_FILE = DATA_DIR / "chunks_embeddings.npy"

# ==============================
#  INDEX STORAGE & LOADING
# ==============================

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

# ==============================
# PERFORMANCE UTILITIES
# ==============================
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

# ==============================
# SOURCE PARSING
# ==============================
def parse_source_info(answer: str) -> dict: 
    """ Extracts source, page, and chunk from an answer string. 
    Example pattern: [source: technical-manual-w11663204-revb, page: 32, chunk: 33] """ 
    pattern =r"\[source:\s*([^,\]]+),\s*page:\s*(\d+),\s*chunk:\s*(\d+)\]?$"
    match = re.search(pattern, answer, re.IGNORECASE) 
    if not match: 
        return None 
    source, page, chunk = match.groups() 
    return { "source": source.strip(), "page": int(page), "chunk": int(chunk) }

def save_chunks_txt():
    """Save chunks to txt file"""
    chunks, embeddings = load_index()
    with open("chunks_markdown.txt", "w", encoding="utf-8") as f: 
        for chunk in chunks: 
            f.write(json.dumps(chunk, ensure_ascii=False)) 
            f.write("\n\n---\n\n")