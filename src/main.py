from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from src.utils import load_documents, load_index, timer, parse_source_info
from src.model import generate_embeddings, generate_chat_response
import numpy as np
import logging
from src.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

def retrieve_top_k(query_embed, chunks,embeddings, k=5): 
    query_embed = query_embed / np.linalg.norm(query_embed) 
    sims = embeddings @ query_embed 
    top_idx = np.argsort(sims)[::-1][:k] 
    return [(i, sims[i], chunks[i]) for i in top_idx]

def run_pipeline(question: str):
    logger.info("Start running pipeline...")
    chunks, embeddings = load_index()
    query_embed = np.array(generate_embeddings([question]))[0]

    retrieved = retrieve_top_k(query_embed, chunks, embeddings, 5)

    context_parts = [] 
    for _, _, chunk in retrieved:
        chunk_id = chunk.get("id", "")
        file_name = chunk.get("source", "unknown_file") 
        page = chunk.get("page", "unknown_page") 
        text = chunk.get("text", "") 
        context_parts.append( f"[Source: {file_name}, Page: {page}, Chunk_id: {chunk_id}]\n{text}" ) 
    context = "\n\n".join(context_parts)
    
    prompt = f""" 
    You are a technical assistant for field mechanics. 
    Answer the question using ONLY the context below.
    Cite the source using the metadata provided in the context.
    Use this exact format: [source: <file_name>, page: <page_number>, chunk:<chunk_id>].
    Context: {context} 
    Question: {question} 
    """ 

    response = generate_chat_response(prompt)
    return response



if __name__ == '__main__':
    
    with timer() as t_buid_idx:
        processor = DocumentProcessor()
        chunks = processor.load_documents()
        processor.build_index(chunks)
    logger.info("Build index completed in %.2f seconds", t_buid_idx)
    
    question: str = "What should be done before servicing electrical components?"
    answer = run_pipeline(question) 
    print(answer)
    print(parse_source_info(answer))
    

