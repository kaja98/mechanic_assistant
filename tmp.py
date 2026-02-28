import os
import numpy as np
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pypdf import PdfReader
from langchain.docstore import InMemoryDocstore
from model import get_embed_model
# ============ CONFIG ============

OPENAI_API_KEY = ""
DOCS_FOLDER = "./data"
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

client = OpenAI(api_key=OPENAI_API_KEY)

# ============ STEP 1: LOAD + CHUNK ============


def load_documents():
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for file in os.listdir(DOCS_FOLDER):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(DOCS_FOLDER, file))

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    chunks = text_splitter.split_text(text)

                    for chunk in chunks:
                        documents.append(
                            Document(
                                page_content=chunk,
                                metadata={"source": file, "page": page_num + 1},
                            )
                        )

    return documents


# ============ STEP 2: BUILD VECTOR STORE ============


def build_vectorstore():
    docs = load_documents()

    # Extract texts
    texts = [doc.page_content for doc in docs]

    # Generate embeddings using YOUR function
    embeddings = get_embed_model(texts)  # <- list input
    embeddings = np.array(embeddings).astype("float32")

    # Create FAISS index
    dimension = len(embeddings[0])
    index = FAISS.IndexFlatL2(dimension)
    index.add(embeddings)

    # Create docstore mapping
    docstore = InMemoryDocstore({str(i): docs[i] for i in range(len(docs))})
    index_to_docstore_id = {i: str(i) for i in range(len(docs))}

    vectorstore = FAISS(
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        embedding_function=get_embed_model,  # important!
    )

    return vectorstore


# ============ STEP 3: RAG QA ============


def ask_question(vectorstore, question):

    # Retrieve relevant chunks
    results = vectorstore.similarity_search(question, k=3)

    context = "\n\n".join(
        f"[Source: {doc.metadata['source']} | Page {doc.metadata['page']}]\n{doc.page_content}"
        for doc in results
    )

    prompt = f"""
            You are a technical assistant.

            Answer the question using ONLY the context below.

            If the context is insufficient, say:
            "I don't know based on the available documents"

            Context:
            {context}

            Question:
            {question}

            Answer:
            """

    response = client.responses.create(model=MODEL_NAME, input=prompt)

    return response.output_text


# ============ MAIN ============

if __name__ == "__main__":
    print("Loading documents and building index...")
    vectorstore = build_vectorstore()
    print("Ready!\n")

    while True:
        question = input("Ask a question (or type 'exit'): ")

        if question.lower() == "exit":
            break

        answer = ask_question(vectorstore, question)
        print("\nAnswer:\n", answer)
        print("\n" + "-" * 50 + "\n")
