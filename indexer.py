import json
import faiss
import numpy as np
from datasets import load_dataset
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

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
CHUNK_SIZE_WORDS = 200
OVERLAP_WORDS = 30
NB_ROWS = 100

# Initialize model and dataset
embedder = SentenceTransformer(MODEL_NAME)
dataset = load_dataset("natural_questions", split="train", streaming=True)

# Process documents
all_chunks_text = []
all_urls = []

for i, rows in enumerate(dataset):
    if i >= NB_ROWS:
        break
    html = rows["document"]["html"]
    url = rows["document"]["url"]
    
    proper_text = clean_html(html)
    list_chunks = chunk_text(proper_text, chunk_size=CHUNK_SIZE_WORDS, overlap=OVERLAP_WORDS)
    
    for chunk in list_chunks:
        if len(chunk.split()) > 20:
            all_chunks_text.append(chunk)
            all_urls.append(url)

print(f"Number of documents treated: {i}")
print(f"Number of chunks: {len(all_chunks_text)}")

# Generate embeddings
vectors = embedder.encode(all_chunks_text, convert_to_numpy=True)
print(f"Vectors shape: {vectors.shape}")

## Creation of FAISS Index

dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

faiss.write_index(index, "my_rag_db.index")

print(f"FAISS index created.")

metadata = []

for text, url in zip(all_chunks_text, all_urls):
    metadata.append({"text": text, "url": url})

with open("my_rag_db.json", "w") as f:
    json.dump(metadata, f)
    
print(f"Creation of faiss index and json file on hard disk done.")