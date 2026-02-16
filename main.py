import os
from sentence_transformers import SentenceTransformer, util
from mistralai import Mistral
from dotenv import load_dotenv
import rag

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

CONSISTENCY_THRESHOLD = 0.9

def get_multiple_answers(question, n=3):
    answers = []
    for k in range(n):
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": question}],
            temperature=0.7 
        )
        answers.append(response.choices[0].message.content)
    return answers

def calculate_consistency_score(answers):
    embeddings = embedder.encode(answers, convert_to_tensor=True)
    scores = util.cos_sim(embeddings, embeddings)
    avg_score = (scores[0, 1] + scores[0, 2] + scores[1, 2]) / 3
    
    return float(avg_score.item())

def main():
    print("=== ASSISTANT INTELLIGENT (CONSISTENCY-GATED RAG) ===")
    print(f"Seuil de confiance : {CONSISTENCY_THRESHOLD}")
    print("Tape 'q' pour quitter")
    
    while True:
        question = input("\n Votre question : ")
        if question.lower() in ['q']:
            break
        
        answers = get_multiple_answers(question)
        score = calculate_consistency_score(answers)
        
        print(f"[ROUTER] Score de confiance: {score:.2f}")
        
        if score > CONSISTENCY_THRESHOLD:
            print(f"VALIDE : Le modele connait la reponse !")
            print(f"\n Reponse: \n{answers[0]}")
        else:
            print(f">>> DOUTE (Score < {CONSISTENCY_THRESHOLD}): Activationd du RAG.")
            print(f"Recherche dans la base de documents...")
            rag_response = rag.generate_answer(question)
            
            print(f"\n Reponse verifiee (RAG) : \n {rag_response}")

if __name__ == "__main__":
    main()