import gradio as gr
import numpy as np
from pathlib import Path
import json
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDINGS_DIR = Path("data/embeddings")

def load_embeddings_from_paths(selected_paths):
    embeddings = []
    texts = []
    for folder in selected_paths:
        for file in folder.rglob("*.json"):
            data = json.loads(file.read_text(encoding="utf-8"))
            embeddings.append(np.array(data["embedding"]))
            texts.append(data["text"])
    return embeddings, texts

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_similar(text_embedding, embeddings, texts, top_k=3):
    similarities = [cosine_similarity(text_embedding, emb) for emb in embeddings]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [texts[i] for i in top_indices]

def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)

def list_rulebooks():
    return [f for f in EMBEDDINGS_DIR.iterdir() if f.is_dir()]

def list_extensions(rulebook_path):
    return [f for f in rulebook_path.iterdir() if f.is_dir()]

def select_rulebook_and_extensions():
    rulebooks = list_rulebooks()
    print("Available rulebooks:")
    for idx, rb in enumerate(rulebooks):
        print(f"{idx+1}: {rb.name}")
    rb_idx = int(input("Select a rulebook by number: ")) - 1
    rulebook_path = rulebooks[rb_idx]

    extensions = list_extensions(rulebook_path)
    selected_paths = [rulebook_path]

    if extensions:
        print(f"Extensions found for {rulebook_path.name}:")
        for idx, ext in enumerate(extensions):
            print(f"{idx+1}: {ext.name}")
        use_ext = input("Use extensions? (y/n): ").strip().lower()
        if use_ext == "y":
            ext_idxs = input("Enter extension numbers to use (comma separated, or 'all'): ").strip()
            if ext_idxs == "all":
                selected_paths.extend(extensions)
            else:
                for i in ext_idxs.split(","):
                    selected_paths.append(extensions[int(i)-1])
    return selected_paths

def answer_question(question, embeddings, texts):
    question_embedding = get_embedding(question)
    relevant_texts = retrieve_similar(question_embedding, embeddings, texts)
    context = "\n\n".join(relevant_texts)
    prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {question}\nAnswer:"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# --- Gradio interface

print("Please select rulebooks and extensions before launching the GUI.")
selected_paths = select_rulebook_and_extensions()
print("Loading embeddings. This may take a moment...")
embeddings, texts = load_embeddings_from_paths(selected_paths)
print(f"Loaded {len(embeddings)} embeddings.")

def ask(question):
    if not question.strip():
        return "Please enter a question."
    return answer_question(question, embeddings, texts)

with gr.Blocks() as demo:
    gr.Markdown("# RuleLawyer: Question Answering over Your Documents")
    question_input = gr.Textbox(label="Enter your question here", lines=2, placeholder="Type your question...")
    answer_output = gr.Textbox(label="Answer", interactive=False)
    ask_button = gr.Button("Ask")

    ask_button.click(ask, inputs=question_input, outputs=answer_output)

if __name__ == "__main__":
    demo.launch()
