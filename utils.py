from pathlib import Path
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import numpy as np
import logging
from config import CHUNK_SIZE, CHUNK_OVERLAP
from model import get_embed_model
from langchain_core.documents import Document

from unstructured.partition.pdf import partition_pdf
from llama_index.core_node_parser import SemanticSplitterNodeParser

DOCS_FOLDER = "documets"
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(BASE_DIR / "data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
print("DATA_DIR = ", DATA_DIR)

CHUNK_FILE = DATA_DIR / "chunks.npy"
EMBEDDINGS_FILE = DATA_DIR / "chunks_embeddings.npy"

path = Path("C:/Users/kajao/OneDrive/Dokumenty/Project/mechanic_assistant/documets/LAD-Front-Loading-Service-Manual-L11.pdf")

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,  # chunk size (characters)
        chunk_overlap=CHUNK_OVERLAP,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )

def load_documents() -> None:

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
    

# def load_pdf(path: Path) -> str:
#     reader = PdfReader(path)
        

def chunk_text(path: Path) -> list[dict]:
    file_name: str = path.stem
    chunks: list[dict] = []

    reader = PdfReader(path)

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        # print(text)

        if text.strip():
            page_chunks = text_splitter.split_text(text)
            for chunk_text in page_chunks:
                chunks.append({
                    "text": chunk_text,
                    "source": file_name,
                    "page": page_num+1,
                })
                # chunks.append(
                #     Document(
                #         page_content=chunk_text,
                #         metadata={
                #             "source": file_name,
                #             "page": page_num + 1,
                #         },
                #     )
                # )
        # print(chunks)
        # break
    return chunks
    # # printing number of pages in pdf file
    # print(len(reader.pages))

    # # creating a page object
    # page = reader.pages[4]

    # # extracting text from page
    # print(page.extract_text())

def test_page(num_page):
    reader = PdfReader(path)
    # creating a page object
    page = reader.pages[num_page]

    # extracting text from page
    print(page.extract_text())

def build_embeddings(file_path: Path) -> tuple:
    chunks = chunk_text(file_path)
    texts = [chunk["text"] for chunk in chunks]
    # texts = [chunk.page_content for chunk in chunks]
    embeddings = get_embed_model(texts)
    return chunks, embeddings

# load_pdf("/...")
# process_file(path)
# chunks, embeddings = build_embeddings()
# print(chunks)
# print("===============")
# print(embeddings)

def load_index():
    chunks = np.load(CHUNK_FILE, allow_pickle=True)
    embeddings = np.load(EMBEDDINGS_FILE)
    return chunks, embeddings

# load_documents()
# test_page(31)

def extract_text():
    elements = partition_pdf(
        path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        infer_table_structure=True
    )
    text = "\n\n".join([str(el) for el in elements])
    print(text)

extract_text()