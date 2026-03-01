# import nltk
# nltk.download("punkt_tab")
# nltk.download("stopwords")

import logging
from nltk import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from model import generate_embeddings

logger = logging.getLogger(__name__)

def stem_words(text: str) -> set[str]:
    """Return a set of stemmed non-stopword tokens."""
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words if word.lower() not in stop_words]
    logger.debug("Stemmed %d words.", len(stemmed_words))
    return set(stemmed_words)

def keyword_match_score(query: str, extracted_text: str) -> float:
    """Return keyword overlap score between query and text."""
    query_stems = stem_words(query)
    extracted_stems = stem_words(extracted_text)
    if not query_stems:
        logger.debug("Query stems empty; returning score 0.0.")
        return 0.0
    matches = query_stems.intersection(extracted_stems)
    score = len(matches) / len(query_stems)
    logger.debug("Keyword match score: %.4f", score)
    return score


def cosine_similarity_score(query: str, extracted_text: str) -> float:
    """Return TF-IDF cosine similarity between query and text."""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([query, extracted_text])
    score = float(cosine_similarity(vectors)[0, 1])
    logger.debug("TF-IDF cosine similarity: %.4f", score) 
    return score

# def contextual_similarity(query: str, extracted_text: str) -> float:
#     query_embedding = generate_embeddings(query)
#     extracted_text_embedding = generate_embeddings(extracted_text)

#     return cosine_similarity([query_embedding, extracted_text_embedding])[0][0]

def contextual_similarity(query: str, extracted_text: str) -> float: 
    """Return embedding-based cosine similarity between query and text."""
    q = np.array(generate_embeddings([query]))[0]
    t = np.array(generate_embeddings([extracted_text]))[0]
    
    q = q / np.linalg.norm(q)
    t = t / np.linalg.norm(t)
    
    score = float(np.dot(q, t)) 
    logger.debug("Contextual similarity: %.4f", score) 
    return score

def reranking(doc_list: list[dict]) -> list[dict]:
    """Return documents sorted by keyword and contextual similarity."""
    sorted_docs = sorted( doc_list, key=lambda x: (float(x["keyword_match"]), float(x["contextual_sim"])), reverse=True)
    logger.info("Reranked %d documents.", len(sorted_docs))
    return sorted_docs


def combined_score(query: str, chunk_text: str, embed_sim: float) -> float: 
    """Return weighted combined similarity score."""
    kw = keyword_match_score(query, chunk_text)
    tfidf = cosine_similarity_score(query, chunk_text)
    ctx = embed_sim
    
    score = 0.2 * kw + 0.2 * tfidf + 0.6 * ctx
    logger.debug( "Combined score: kw=%.4f, tfidf=%.4f, ctx=%.4f → total=%.4f", kw, tfidf, ctx, score)
    return score