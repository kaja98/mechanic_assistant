from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from utils import load_documents, load_index
from model import generate_embeddings, generate_chat_response
import numpy as np
import logging

logger = logging.getLogger(__name__)

def run_pipeline_faiss(question: str):
    """Return the most relevant FAISS search result for a query."""
    load_documents()
    chunks, embeddings = load_index()
    # query_embed = generate_embeddings(question)

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = FAISS.from_documents(chunks, embedding_model)
    results = vectorstore.similarity_search(question, k=2)

    print(f"\nQuery: '{question}'")
    print("\nTop 2 relevant chunks:")
    for i, doc in enumerate(results, 1):
        print(f"\nChunk {i}: {doc.page_content}")
    
    return results[0]


def retrieve_top_k(query_embed, chunks,embeddings, k=5): 
    query_embed = query_embed / np.linalg.norm(query_embed) 
    sims = embeddings @ query_embed 
    top_idx = np.argsort(sims)[::-1][:k] 
    return [(i, sims[i], chunks[i]) for i in top_idx]

def run_pipeline(question: str):
    load_documents()
    chunks, embeddings = load_index()
    query_embed = np.array(generate_embeddings([question]))[0]

    retrieved = retrieve_top_k(query_embed, chunks, embeddings, 5)

    context_parts = [] 
    for _, _, chunk in retrieved: 
        file_name = chunk.get("source", "unknown_file") 
        page = chunk.get("page", "unknown_page") 
        text = chunk.get("text", "") 
        context_parts.append( f"[Source: {file_name}, Page: {page}]\n{text}" ) 
    context = "\n\n".join(context_parts)
    
    prompt = f""" 
    You are a technical assistant for field mechanics. 
    Answer the question using ONLY the context below.
    Cite the source of your information, e.g. [source, [page_num_1, pge_num_2]]. 
    Context: {context} 
    Question: {question} 
    """ 

    response = generate_chat_response(prompt)
    return response



if __name__ == '__main__':
    question: str = "How does a technician enter service mode on the L11 washing machine?"
    load_documents()
    run_pipeline(question)