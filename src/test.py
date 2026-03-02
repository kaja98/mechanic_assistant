import json
import re
from pathlib import Path
from src.main import run_pipeline

# Project root directory (mechanic_assistant/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
EVAL_DATASET_PATH = DATA_DIR / "eval_dataset.json"
EVAL_RESULTS_PATH = DATA_DIR / "eval_results.json"

def load_json(path: Path):

    with open(path, "r") as f: 
        eval_data = json.load(f)
    print(eval_data)
    return eval_data

def run_test():
    eval_data = load_json(EVAL_DATASET_PATH)
    for item in eval_data: 
        question = item["query"] 
        model_answer = run_pipeline(question) 
        item["model_answer"] = model_answer

    # Save updated dataset 
    with open(EVAL_RESULTS_PATH, "w") as f: 
        json.dump(eval_data, f, indent=4)


def extract_chunks_from_answer(answer: str): 
    """ Extract chunk numbers from model answer using regex. Returns a list of integers. """ 
    # Matches: chunk: 39 or chunk=39 or chunk 39 
    pattern = r"[Cc]hunk[:=\s]+(\d+)" 
    chunks = re.findall(pattern, answer) 
    return [int(c) for c in chunks]


def compare_chunks(predicted, relevant): 
    """ Compare predicted chunks with ground truth. Returns dict with match info. """ 
    predicted_set = set(predicted) 
    relevant_set = set(relevant) 
    return { 
            "predicted_chunks": list(predicted_set), 
            "relevant_chunks": list(relevant_set), 
            "correct": list(predicted_set & relevant_set), 
            "missed": list(relevant_set - predicted_set), 
            "wrong": list(predicted_set - relevant_set), 
            "accuracy": len(predicted_set & relevant_set) / len(relevant_set) if relevant_set else 0 
            }

def evaluate_eval_results(path=EVAL_RESULTS_PATH): 
    with open(path, "r") as f: 
        data = json.load(f) 
        evaluations = [] 
        for item in data: 
            model_answer = item.get("model_answer", "") 
            relevant_chunks = item["relevant_chunks"] 
            predicted_chunks = extract_chunks_from_answer(model_answer) 
            comparison = compare_chunks(predicted_chunks, relevant_chunks) 
            evaluations.append({ "query": item["query"], "expected_chunks": relevant_chunks, "predicted_chunks": predicted_chunks, "chunk_evaluation": comparison, "model_answer": model_answer }) 
    return evaluations

if __name__ == "__main__":
    run_test()
    results = evaluate_eval_results() 
    for r in results: 
        print("QUESTION:", r["query"])
        print("EXPECTED:", r["expected_chunks"]) 
        print("PREDICTED:", r["predicted_chunks"]) 
        print("MATCH:", r["chunk_evaluation"]["correct"]) 
        print("MISSED:", r["chunk_evaluation"]["missed"]) 
        print("WRONG:", r["chunk_evaluation"]["wrong"]) 
        print("ACCURACY:", r["chunk_evaluation"]["accuracy"]) 
        print("-" * 40)