"""Document processor class."""
import os
import logging
from pathlib import Path
from typing import Literal
import numpy as np
import fitz # PyMuPDF library
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from src.config import DOCS_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP
from src.utils import save_index
from src.model import generate_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

class DocumentProcessor:
    """Handles PDF document loading and text extraction."""
    
    def __init__(self, docs_folder: str = DOCS_FOLDER, splitter_type: Literal["recursive", "markdown"] = "markdown"):
        self.docs_folder = docs_folder
        self.splitter_type = splitter_type

        if splitter_type == "markdown":
            self.text_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on
            )
        elif splitter_type == "recursive":
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                add_start_index=True,
            )
        else:
            raise ValueError("splitter_type must be 'recursive' or 'markdown'")
    
    def load_documents(self) -> list[dict[str, any]]:
        if not os.path.exists(self.docs_folder):
            os.mkdir(self.docs_folder)
            logger.info(f"Created {self.docs_folder} folder. Please add PDF documents.")
            return []
        
        pdf_files = [f for f in os. listdir(self.docs_folder) if f.endswith(".pdf")]
        if not pdf_files: 
            logger.error("No PDF files found in docs folder.")
            return []
        
        logger.info(f"Loading {len(pdf_files)} PDF documents...")
        all_chunks = [] 
        for file in pdf_files:
            file_path: Path = Path(os.path.join(DOCS_FOLDER, file))
            chunks = self._process_pdf(file_path)
            all_chunks.extend(chunks)
        logger.debug(f"\n Total chunks created: {len(chunks)}")
        return all_chunks

    def build_index(self, chunks: list[dict[str, any]]): 
        """Generate embeddings and save index.""" 
        texts = [c["text"] for c in chunks] 
        embeddings = generate_embeddings(texts) 
        embeddings = np.array(embeddings) 
        save_index(chunks, embeddings)

    def _process_pdf(self, path: Path) -> list[dict[str, any]]:
        """Split PDF into text chunks with filename and page numbers using pymupdf4llm markdown or Recursive splitter."""
        
        file_name: str = path.stem
        chunks: list[dict[str, any]] = []
        chunk_id = 1
        
        if self.splitter_type == "markdown":
            logger.info(f"Process pdf {file_name} with Markdown splitter.")
            # Extract markdown per page
            pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)

            # pages is a list of dicts when page_chunks=True
            for page in pages:
                text = page.get("text", "")
                metadata = page.get("metadata", {})
                page_number = metadata.get("page", None)

                if text.strip():
                    page_chunks = self.text_splitter.split_text(text)
                    for chunk_text in page_chunks:
                        chunks.append({
                            "id": chunk_id,
                            "text": chunk_text.page_content,
                            "source": file_name,
                            "page": page_number,
                        })
                        chunk_id += 1

            return chunks
        else:
            logger.info("Process pdf with Recursive splitter.")
            # Open a PDF document
            doc = fitz.open(path)

            chunk_id = 1
            # Iterate through pages and extract text
            for page_number, page in enumerate(doc, start=1):
                text = page.get_text() # Extract text as UTF-8
                if text.strip():
                    page_chunks = self.text_splitter.split_text(text)
                    for chunk_text in page_chunks:
                        chunks.append({
                            "id":chunk_id,
                            "text": chunk_text,
                            "source": file_name,
                            "page": page_number,
                        })
                        chunk_id += 1

            doc.close()
            return chunks