"""
QA evaluation: measures whether the consistency-gated RAG pipeline
improves answer quality on questions where the LLM is uncertain.

For each question with a gold short answer:
  1. Consistency gate (LLM x3, temp=0.7) → score
  2. If score < 0.9 → compare direct answer vs RAG answer (token F1)
  3. If score >= 0.9 → record as confident, compute F1 of direct answer

Metric: token F1 (bag-of-words overlap between prediction and gold),
the standard evaluation metric for free-text QA (Natural Questions, SQuAD).
"""

import os
import re
import json
import random
import string
import time
import faiss
import numpy as np
from collections import Counter
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from mistralai.client import Mistral
from dotenv import load_dotenv
from tqdm import tqdm

ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')

load_dotenv(os.path.join(ROOT_DIR, '.env'))
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# --- Configuration ---
NB_ROWS = 2000           # must match indexer.py
NB_QUESTIONS = 50
CONSISTENCY_THRESHOLD = 0.9
K_TOP = 3                # chunks to retrieve for RAG

# --- Token F1 (SQuAD-style) ---
def normalize_answer(s):
    """Lowercase, strip articles, punctuation, and extra whitespace."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())

def token_f1(prediction, gold):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

# --- Load index & metadata ---
print("Loading FAISS index and metadata...")
index = faiss.read_index(os.path.join(ROOT_DIR, "my_rag_db.index"))
index.nprobe = 16
with open(os.path.join(ROOT_DIR, "my_rag_db.json"), "r") as f:
    metadata = json.load(f)
indexed_urls = set(m["url"] for m in metadata)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Index: {index.ntotal} vectors")

# --- Collect questions with short answers ---
print(f"Streaming first {NB_ROWS} examples, filtering those with short answers...")
dataset = load_dataset("natural_questions", split="train", streaming=True)

candidates = []
for i, ex in enumerate(tqdm(dataset, desc="Filtering", total=NB_ROWS)):
    if i >= NB_ROWS:
        break
    url = ex["document"]["url"]
    if url not in indexed_urls:
        continue
    short_answers = ex["annotations"]["short_answers"]
    for sa in short_answers:
        if sa["text"] and sa["text"][0]:
            candidates.append({
                "question": ex["question"]["text"],
                "gold_answer": sa["text"][0],
                "url": url
            })
            break

print(f"Questions with short answers in index: {len(candidates)}")
random.seed(42)
eval_set = random.sample(candidates, min(NB_QUESTIONS, len(candidates)))
print(f"Eval set: {len(eval_set)} questions")

# --- Helper functions ---
def api_call_with_retry(messages, temperature=0.7, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.complete(
                model="mistral-small-latest",
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"\n  API error ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

def get_multiple_answers(question, n=3):
    answers = []
    for _ in range(n):
        answer = api_call_with_retry([{"role": "user", "content":
            f"Answer this question in a few words only: {question}"}])
        answers.append(answer)
        time.sleep(1)
    return answers

def consistency_score(answers):
    embeddings = embedder.encode(answers, convert_to_tensor=True)
    scores = util.cos_sim(embeddings, embeddings)
    avg = (scores[0, 1] + scores[0, 2] + scores[1, 2]) / 3
    return float(avg.item())

def rag_answer(question):
    query_vec = embedder.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    distances, indices = index.search(query_vec, K_TOP)

    context_str = ""
    for j in range(K_TOP):
        idx = indices[0][j]
        if idx == -1:
            continue
        doc = metadata[idx]
        context_str += f"Source [{j+1}] ({doc['url']}):\n{doc['text']}\n\n"

    if not context_str:
        return "I don't know."

    return api_call_with_retry([
        {"role": "system", "content":
            "Answer the question using ONLY the provided context. "
            "Be concise: answer in a few words. "
            "If the answer is not in the context, say 'I don't know'."},
        {"role": "user", "content":
            f"CONTEXT:\n{context_str}\nQUESTION: {question}"}
    ], temperature=0.0)

# --- Evaluation loop ---
results_confident = []   # gate passed (score >= 0.9)
results_uncertain = []   # gate triggered RAG (score < 0.9)

for q in tqdm(eval_set, desc="Evaluating QA"):
    answers = get_multiple_answers(q["question"])
    score = consistency_score(answers)
    direct_f1 = token_f1(answers[0], q["gold_answer"])

    if score >= CONSISTENCY_THRESHOLD:
        results_confident.append({
            "question": q["question"],
            "gold": q["gold_answer"],
            "direct_answer": answers[0],
            "consistency_score": score,
            "direct_f1": direct_f1,
        })
    else:
        time.sleep(0.5)
        rag_resp = rag_answer(q["question"])
        rag_f1_val = token_f1(rag_resp, q["gold_answer"])
        results_uncertain.append({
            "question": q["question"],
            "gold": q["gold_answer"],
            "direct_answer": answers[0],
            "rag_answer": rag_resp,
            "consistency_score": score,
            "direct_f1": direct_f1,
            "rag_f1": rag_f1_val,
        })

# --- Report ---
print("\n" + "=" * 60)
print("QA EVALUATION RESULTS")
print("=" * 60)

n_conf = len(results_confident)
n_unc = len(results_uncertain)
print(f"\nGate routing:")
print(f"  Confident (score >= {CONSISTENCY_THRESHOLD}): {n_conf}/{len(eval_set)} ({100*n_conf/len(eval_set):.0f}%)")
print(f"  Uncertain (RAG triggered):  {n_unc}/{len(eval_set)} ({100*n_unc/len(eval_set):.0f}%)")

if results_confident:
    avg_f1_conf = np.mean([r["direct_f1"] for r in results_confident])
    avg_score_conf = np.mean([r["consistency_score"] for r in results_confident])
    print(f"\nConfident questions (direct answer only):")
    print(f"  Avg consistency score: {avg_score_conf:.3f}")
    print(f"  Avg token F1:         {avg_f1_conf:.3f}")

if results_uncertain:
    avg_direct_f1 = np.mean([r["direct_f1"] for r in results_uncertain])
    avg_rag_f1 = np.mean([r["rag_f1"] for r in results_uncertain])
    avg_score_unc = np.mean([r["consistency_score"] for r in results_uncertain])
    improvement = avg_rag_f1 - avg_direct_f1
    print(f"\nUncertain questions (RAG vs direct):")
    print(f"  Avg consistency score: {avg_score_unc:.3f}")
    print(f"  Avg token F1 direct:  {avg_direct_f1:.3f}")
    print(f"  Avg token F1 RAG:     {avg_rag_f1:.3f}")
    print(f"  F1 improvement:       {improvement:+.3f}")

# Save detailed results
all_results = {"confident": results_confident, "uncertain": results_uncertain}
results_path = os.path.join(os.path.dirname(__file__), "qa_eval_results.json")
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nDetailed results saved to {results_path}")
