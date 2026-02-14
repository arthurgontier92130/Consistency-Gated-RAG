# Consistency-Gated-RAG
Adaptive-Mistral-RAG triggers RAG only when the LLM is unstable. It uses a Self-Consistency router to detect hallucinations. If unstable, it pulls dense vectors from MS MARCO via FAISS for an Open Book answer grounded in facts.
