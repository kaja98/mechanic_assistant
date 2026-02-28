from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from utils import load_documents, load_index
from model import get_embed_model
from validation_metrics import cosine_similarity, combined_score
import numpy as np

def answer():
    pass

def run_pipeline_faiss(question: str):
    load_documents()
    chunks, embeddings = load_index()
    # query_embed = get_embed_model(question)

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = FAISS.from_documents(chunks, embedding_model)
    results = vectorstore.similarity_search(question, k=2)

    print(f"\nQuery: '{question}'")
    print("\nTop 2 relevant chunks:")
    for i, doc in enumerate(results, 1):
        print(f"\nChunk {i}: {doc.page_content}")
    
    return results[0]

# def retrieve_top_k(query_embedding, chunks, embeddings, k: int) -> list[tuple[int, float, str]]:

#     # scores = [(i, cosine_similarity(query_embedding, emb)) for i, emb in enumerate(embeddings)]
#     # scores.sort(key=lambda x: x[1], reverse=True)
#     # top = scores[:k]

#     # return [(i, sim, chunks[i]) for i, sim in top]
#     query_embedding = query_embedding.reshape(1, -1) 
#     scores = [] 
#     for i, emb in enumerate(embeddings): 
#         emb = emb.reshape(1, -1) # reshape each embedding 
         
#         sims = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0] 
#     top_idx = sims.argsort()[::-1][:k] 
#     return [(i, sims[i], chunks[i]) for i in top_idx]
#     #     sim = cosine_similarity(query_embedding.reshape(1, -1), emb)[0][0]
#     #     scores.append((i, sim)) 
#     # scores.sort(key=lambda x: x[1], reverse=True) 
#     # top = scores[:k] 
#     # return [(i, sim, chunks[i]) for i, sim in top]

def retrieve_top_k(query_text, query_embed, chunks, embeddings, k): 
    query_embed = query_embed / np.linalg.norm(query_embed)  
    embed_sims = embeddings @ query_embed 
    scored = [] 
    for i, sim in enumerate(embed_sims): 
        text = chunks[i]["text"] 
        score = combined_score(query_text, text, sim) 
        scored.append((i, score, chunks[i])) 
    scored.sort(key=lambda x: x[1], reverse=True) 
    return scored[:k]

def run_pipeline(question: str):
    # load_documents()
    chunks, embeddings = load_index()
    query_embed = np.array(get_embed_model([question]))[0]
    # print("query_embed = ", query_embed)
    # print("emmbeddings = ", embeddings[:10])
    # print("Query norm:", np.linalg.norm(query_embed)) 
    # print("First embedding norm:", np.linalg.norm(embeddings[0])) 
    # print("All norms (first 10):", np.linalg.norm(embeddings, axis=1)[:10])
    retrieved = retrieve_top_k(question, query_embed, chunks, embeddings, 3)

    for res in retrieved:
        print(res)



if __name__ == '__main__':
    question: str = "How should user control washing maschine?"
    # answer: str = answer()
    run_pipeline(question)