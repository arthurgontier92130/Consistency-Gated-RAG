import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError("API Key not found in .env file")

client = Mistral(api_key=api_key)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

try:
    index = faiss.read_index("my_rag_db.index")
    with open("my_rag_db.json", "r") as f:
        metadata = json.load(f)
except Exception as e:
    print(f"Error loading database: {e}")
    exit(1)

def retrieve_context(question, k=3):
    query_vector = embedder.encode([question], convert_to_numpy=True)
    distances, indices = index.search(query_vector, k)
    
    results = []
    for i in range(k):
        idx = indices[0][i]
        if idx == -1:
            continue
            
        doc = metadata[idx]
        results.append({
            "text": doc["text"],
            "url": doc["url"],
            "score": float(distances[0][i])
        })
        
    return results

def generate_answer(question):
    print(f"\n--- Searching for: '{question}' ---")
    
    context_results = retrieve_context(question, k=3)
    
    if not context_results:
        return "I did not find relevant information in my documents."

    context_str = ""
    for i, res in enumerate(context_results):
        context_str += f"Source [{i+1}] ({res['url']}):\n{res['text']}\n\n"

    system_prompt = """
    Tu es un assistant factuel et précis. Ta mission est de répondre à la question de l'utilisateur 
    en utilisant UNIQUEMENT les informations fournies dans le contexte ci-dessous.
    
    Règles strictes :
    - Si la réponse n'est pas dans le contexte, dis "Je ne sais pas".
    - Ne jamais inventer d'information.
    - Cite tes sources à la fin de chaque phrase ou paragraphe en utilisant le format [1], [2].
    - Réponds dans la même langue que la question.
    """
    
    user_message = f"""
    CONTEXTE FOURNI :
    {context_str}
    
    QUESTION UTILISATEUR : 
    {question}
    """

    try:
        chat_response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        return chat_response.choices[0].message.content
        
    except Exception as e:
        return f"Error during API call: {e}"

if __name__ == "__main__":
    print("RAG Module Loaded. Type 'q' to quit.")
    
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() in ["q", "quit", "exit"]:
            break
            
        response = generate_answer(user_input)
        
        print("\n--- MISTRAL RESPONSE ---")
        print(response)
        print("-----------------------")