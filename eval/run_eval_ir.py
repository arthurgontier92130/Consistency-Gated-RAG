"""Quick evaluation script — runs the same logic as evaluation_ir.ipynb"""
import os
from sentence_transformers import SentenceTransformer
import faiss, json, random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')

# --- Configuration ---
NB_ROWS = 2000           # must match indexer.py
NB_QUESTIONS_TEST = 200
K_TOP = 20

# --- Load index & metadata ---
print("Loading FAISS index and metadata...")
index = faiss.read_index(os.path.join(ROOT_DIR, "my_rag_db.index"))
index.nprobe = 16

with open(os.path.join(ROOT_DIR, "my_rag_db.json"), "r") as f:
    metadata = json.load(f)

indexed_urls = set(m["url"] for m in metadata)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Index: {index.ntotal} vectors, {len(indexed_urls)} unique URLs")

# --- Load dataset (streaming to avoid downloading all 55GB) ---
print(f"Streaming first {NB_ROWS} examples from Natural Questions...")
dataset = load_dataset("natural_questions", split="train", streaming=True)

candidates = []
for i, example in enumerate(tqdm(dataset, desc="Filtering candidates", total=NB_ROWS)):
    if i >= NB_ROWS:
        break
    url = example["document"]["url"]
    if url in indexed_urls:
        candidates.append({
            "question": example["question"]["text"],
            "ground_truth_url": url
        })

print(f"Candidates with URL in index: {len(candidates)}")

random.seed(42)
eval_set = random.sample(candidates, min(NB_QUESTIONS_TEST, len(candidates)))
print(f"Eval set: {len(eval_set)} questions (random sample)")

# --- Retrieval ---
all_relevance_results = []

for question in tqdm(eval_set, desc="Evaluating retrieval"):
    embedding = embedder.encode([question["question"]], convert_to_numpy=True)
    faiss.normalize_L2(embedding)
    distances, indices = index.search(embedding, K_TOP)

    current_query_relevance = []
    for idx in indices[0]:
        if idx == -1:
            current_query_relevance.append(False)
            continue
        found_url = metadata[idx]["url"]
        is_relevant = (found_url == question["ground_truth_url"])
        current_query_relevance.append(is_relevant)

    all_relevance_results.append(current_query_relevance)

# --- Metrics ---
def calculate_ap(relevance_bools):
    precisions, num_relevant_found = [], 0
    for i, is_relevant in enumerate(relevance_bools):
        if is_relevant:
            num_relevant_found += 1
            precisions.append(num_relevant_found / (i + 1))
    return sum(precisions) / len(precisions) if precisions else 0.0

def calculate_recall_at_k(relevance_bools, k):
    return 1.0 if any(relevance_bools[:k]) else 0.0

def calculate_rr(relevance_bools):
    for i, is_relevant in enumerate(relevance_bools):
        if is_relevant:
            return 1.0 / (i + 1)
    return 0.0

ap_scores = [calculate_ap(res) for res in all_relevance_results]
map_score = np.mean(ap_scores)

recall_at_1 = np.mean([calculate_recall_at_k(res, 1) for res in all_relevance_results])
recall_at_5 = np.mean([calculate_recall_at_k(res, 5) for res in all_relevance_results])
recall_at_20 = np.mean([calculate_recall_at_k(res, 20) for res in all_relevance_results])

rr_scores = [calculate_rr(res) for res in all_relevance_results]
mrr_score = np.mean(rr_scores)

print("\n===== RESULTS =====")
print(f"Questions evaluated: {len(ap_scores)}")
print(f"MAP@{K_TOP}    : {map_score:.3f}")
print(f"MRR        : {mrr_score:.3f}")
print(f"Recall@1   : {recall_at_1:.3f}")
print(f"Recall@5   : {recall_at_5:.3f}")
print(f"Recall@20  : {recall_at_20:.3f}")

# --- 11-Point Interpolated Precision-Recall Curve ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def interpolated_precision_at_recall(relevance_bools, recall_levels):
    precisions_at_ranks, recalls_at_ranks = [], []
    num_relevant_found = 0
    total_relevant = sum(relevance_bools)
    if total_relevant == 0:
        return [0.0] * len(recall_levels)
    for i, is_relevant in enumerate(relevance_bools):
        if is_relevant:
            num_relevant_found += 1
        precisions_at_ranks.append(num_relevant_found / (i + 1))
        recalls_at_ranks.append(num_relevant_found / total_relevant)
    interpolated = []
    for r in recall_levels:
        max_prec = 0.0
        for prec, rec in zip(precisions_at_ranks, recalls_at_ranks):
            if rec >= r:
                max_prec = max(max_prec, prec)
        interpolated.append(max_prec)
    return interpolated

recall_levels = np.linspace(0.0, 1.0, 11)
all_interpolated = [interpolated_precision_at_recall(res, recall_levels) for res in all_relevance_results]
avg_interpolated = np.mean(all_interpolated, axis=0)

plt.figure(figsize=(8, 5))
plt.plot(recall_levels, avg_interpolated, marker='o', linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Interpolated Precision")
plt.title("11-Point Interpolated Precision-Recall Curve")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid(True, alpha=0.3)
plt.xticks(recall_levels)
plt.tight_layout()
pr_path = os.path.join(os.path.dirname(__file__), "pr_curve.png")
plt.savefig(pr_path, dpi=150)
print(f"\nPR curve saved to {pr_path}")
