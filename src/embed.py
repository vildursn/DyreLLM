import os
from pathlib import Path
import openai
import json

from dotenv import load_dotenv

load_dotenv()  # Load .env file

api_key = os.getenv("OPENAI_API_KEY")
print("API key loaded:", api_key is not None)

PROCESSED_DIR = Path("data/processed")
EMBEDDINGS_DIR = Path("data/embeddings")
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

openai.api_key = api_key
if openai.api_key is None:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

def get_embedding(text, model="text-embedding-3-large"):
    response = openai.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def embed_all_chunks():
    for txt_file in PROCESSED_DIR.glob("*.txt"):
        print(f"Embedding {txt_file.name}")
        text = txt_file.read_text(encoding="utf-8")
        embedding = get_embedding(text)
        
        emb_file = EMBEDDINGS_DIR / f"{txt_file.stem}.json"
        with open(emb_file, "w", encoding="utf-8") as f:
            json.dump({
                "text": text,
                "embedding": embedding
            }, f)

if __name__ == "__main__":
    embed_all_chunks()
