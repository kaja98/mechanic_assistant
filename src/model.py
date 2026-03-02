"""Load OpenAI models."""
from openai import OpenAI
from src.config import MODEL_NAME, EMBEDDING_MODEL
from dotenv import load_dotenv
import os

load_dotenv(override=True)

CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_chat_response(prompt: str) -> str:
    """Return model-generated chat response as plain text."""
    response = CLIENT.responses.create(
    model=MODEL_NAME,
    input=prompt
    )

    print(response.output_text)
    return response.output_text

def generate_embeddings(text: str) -> list[list[float]]:
    """Return list of embedding vectors for the given text."""
    response = CLIENT.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return [item.embedding for item in response.data]
