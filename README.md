# Consistency-Gated-RAG
 
Consistency-Gated RAG is a hybrid Question Answering (QA) framework designed to optimize the balance between a model's parametric memory and external evidence. While large language models (LLMs) store facts in their parameters, they are often poorly calibrated, leading to hallucinations. This project implements an adaptive pipeline that triggers Retrieval-Augmented Generation (RAG) only when the model's internal knowledge is detected as unstable.

## Phase I: The Self-Consistency Router
The system first attempts to answer the query using the model's internal knowledge, a task known as closed book QA.The router queries the model N times with high temperature to measure answer stability. If responses diverge, it indicates the model is not well-calibrated and may be hallucinating. Low stability triggers the transition to open book QA via the retrieval pipeline.

## Phase II: Dense Retrieval Index
To resolve the vocabulary mismatch problem inherent in sparse systems, this project utilizes dense retrieval.Document collections like MS MARCO or Natural Questions are processed into passages. These passages are represented as dense vectors (embeddings) computed by the language model. The system uses FAISS to perform efficient nearest neighbor search.

## Phase III: Grounded Generation
Once relevant passages are retrieved, the generator produces the final response.The prompt includes the original query and the retrieved passages to ensure the answer is grounded in curated facts. The generation includes knowledge citations to provide verifiable evidence and increase user confidence. 

### Evaluation and Metrics
The system is evaluated using metrics defined for ranked retrieval and question answering:Retrieval performance is measured using Mean Average Precision (MAP) or interpolated precision. Answer accuracy for factoid questions is evaluated via exact match or token F1 score.