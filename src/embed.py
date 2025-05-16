import os
from pathlib import Path
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROCESSED_DIR = Path("data/processed")
EMBEDDINGS_DIR = Path("data/embeddings")
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def embed_chunks():
    for txt_file in PROCESSED_DIR.glob("*.txt"):
        emb_file = EMBEDDINGS_DIR / f"{txt_file.stem}.json"
        
        if emb_file.exists():
            print(f"Skipping {txt_file.name} - embedding already exists")
            continue
        
        print(f"Embedding {txt_file.name}")
        text = txt_file.read_text(encoding="utf-8")
        embedding = get_embedding(text)
        
        with open(emb_file, "w", encoding="utf-8") as f:
            json.dump({
                "text": text,
                "embedding": embedding
            }, f)

if __name__ == "__main__":
    embed_chunks()
