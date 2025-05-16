import json
from pathlib import Path
import openai
import numpy as np

EMBEDDINGS_DIR = Path("data/embeddings")

def load_embeddings():
    # Load all embeddings and texts into memory
    embeddings = []
    texts = []
    for file in EMBEDDINGS_DIR.glob("*.json"):
        data = json.loads(file.read_text(encoding="utf-8"))
        embeddings.append(np.array(data["embedding"]))
        texts.append(data["text"])
    return embeddings, texts

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_similar(text_embedding, embeddings, texts, top_k=3):
    # Calculate similarity between query embedding and all stored embeddings
    similarities = [cosine_similarity(text_embedding, emb) for emb in embeddings]
    # Get indices of top_k most similar embeddings
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    # Return the top_k matching texts
    return [texts[i] for i in top_indices]

def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=text, model=model)
    return np.array(response['data'][0]['embedding'])

def answer_question(question):
    embeddings, texts = load_embeddings()
    question_embedding = get_embedding(question)

    relevant_texts = retrieve_similar(question_embedding, embeddings, texts)
    context = "\n\n".join(relevant_texts)

    prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {question}\nAnswer:"

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=_
