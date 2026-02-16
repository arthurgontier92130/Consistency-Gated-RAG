# Consistency-Gated RAG with Mistral AI

A robust Retrieval-Augmented Generation (RAG) system implementing a **Consistency-Gated** routing mechanism. This project acts as an intelligent assistant that queries an external knowledge base only when the Large Language Model (LLM) exhibits uncertainty, optimizing for both latency and factual accuracy.

## Core Concept

Standard RAG systems retrieve documents for every query, which increases latency and cost. This project implements an **Active RAG** approach:

1.  **Consistency Check**: The system queries the LLM (Mistral) multiple times (Self-Consistency).
2.  **Similarity Analysis**: It computes the cosine similarity between the generated answers.
3.  **Routing**:
    * **High Consistency**: The model is confident. Return the direct answer.
    * **Low Consistency**: The model is hallucinating or unsure. Trigger the **Dense Retrieval** pipeline to ground the answer in factual documents.

This aligns with modern NLP research on calibration and retrieval-augmented generation 

## Architecture

The project consists of three independent modules:

* **`indexer.py` (Offline)**: 
    * Ingests the Google Natural Questions dataset.
    * Cleans and chunks HTML documents.
    * [cite_start]Encodes text using a Bi-Encoder (`all-MiniLM-L6-v2`)[cite: 483].
    * Builds a FAISS Vector Index for efficient similarity search.
* **`rag.py` (Retriever)**:
    * Loads the FAISS index.
    * Performs semantic search to retrieve top-k relevant context.
    * Generates a grounded response using Mistral Small.
* **`main.py` (Router)**:
    * Orchestrates the user interaction.
    * Calculates the **Consistency Score** using pairwise semantic similarity.
    * Decides between Direct Generation vs. RAG.

## Installation

### Prerequisites
* Python 3.9+
* A Mistral AI API Key

### Setup

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/consistency-rag-mistral.git](https://github.com/yourusername/consistency-rag-mistral.git)
    cd consistency-rag-mistral
    ```

2.  **Install dependencies**
    ```bash
    pip install mistralai sentence-transformers faiss-cpu datasets numpy python-dotenv beautifulsoup4
    ```

3.  **Environment Configuration**
    Create a `.env` file in the root directory:
    ```env
    MISTRAL_API_KEY=your_actual_api_key_here
    ```

## Usage

### 1. Build the Knowledge Base
First, download and index the data. This creates `my_rag_db.index` and `my_rag_db.json`.
*Note: Adjust `NB_ROWS` in the script to scale the index size.*

```bash
python indexer.py