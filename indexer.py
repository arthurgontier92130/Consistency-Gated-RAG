'''
This script processes a dataset of HTML documents for Retrieval-Augmented Generation (RAG):
 - Cleans and chunks HTML content into text segments
 - Generates embeddings for each chunk using a SentenceTransformer model
 - Builds a FAISS vector index for similarity search
 - Stores chunk metadata (text and source URL) in a JSON file
'''

import json
import faiss
from datasets import load_dataset
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def clean_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def chunk_text(text, chunk_size=256, overlap=30):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        segment = words[i:i+chunk_size]
        reconstitued_chunk = " ".join(segment)
        chunks.append(reconstitued_chunk)
    return chunks

# Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE_WORDS = 128
OVERLAP_WORDS = 20
NB_ROWS = 100000
NLIST = 256  # number of Voronoi cells for IVF index

# Initialize model and dataset
embedder = SentenceTransformer(MODEL_NAME)
print("Downloading dataset (non-streaming for speed)...")
dataset = load_dataset("natural_questions", split=f"train[:{NB_ROWS}]")

# Process documents
all_chunks_text = []
all_urls = []

for i, rows in enumerate(tqdm(dataset, desc="Processing documents")):
    html = rows["document"]["html"]
    url = rows["document"]["url"]
    
    proper_text = clean_html(html)
    list_chunks = chunk_text(proper_text, chunk_size=CHUNK_SIZE_WORDS, overlap=OVERLAP_WORDS)
    
    for chunk in list_chunks:
        if len(chunk.split()) > 20:
            all_chunks_text.append(chunk)
            all_urls.append(url)

print(f"Number of documents treated: {i+1}")
print(f"Number of chunks: {len(all_chunks_text)}")

# Generate embeddings
vectors = embedder.encode(all_chunks_text, convert_to_numpy=True, show_progress_bar=True)
faiss.normalize_L2(vectors)
print(f"Vectors shape: {vectors.shape}")

## Creation of FAISS IVF Index (Approximate Nearest Neighbors)

dimension = vectors.shape[1]
quantizer = faiss.IndexFlatIP(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, NLIST, faiss.METRIC_INNER_PRODUCT)
index.train(vectors)
index.add(vectors)

faiss.write_index(index, "my_rag_db.index")

print(f"FAISS index created.")

metadata = []

for text, url in tqdm(zip(all_chunks_text, all_urls), total=len(all_chunks_text), desc="Creating metadata"):
    metadata.append({"text": text, "url": url})

with open("my_rag_db.json", "w") as f:
    json.dump(metadata, f)
    
print(f"Creation of faiss index and json file on hard disk done.")