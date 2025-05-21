import gradio as gr
import numpy as np
from pathlib import Path
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
import rag

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDINGS_DIR = Path("data/embeddings")

def answer_question(question, embeddings, texts):
    # Get the names of the selected rulebooks and extensions
    selected_names = [p.name for p in selected_paths]
    rulebook_info = f"Rulebooks/extensions in use: {', '.join(selected_names)}"
    question_embedding = rag.get_embedding(question)
    relevant_texts = rag.retrieve_similar(question_embedding, embeddings, texts)
    context = "\n\n".join(relevant_texts)
    prompt = (
        f"{rulebook_info}\n\n"
        f"Answer the question based on the following context:\n\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# --- Gradio interface

print("Please select rulebooks and extensions before launching the GUI.")
selected_paths = rag.select_rulebook_and_extensions()
print("Loading embeddings. This may take a moment...")
embeddings, texts = rag.load_embeddings(selected_paths)
print(f"Loaded {len(embeddings)} embeddings.")

# Build dynamic header
selected_names = [p.name for p in selected_paths]
header_md = (
    "# RuleLawyer: Answering your boardgame questions!\n\n"
    f"**Using rulebooks/extensions:** {', '.join(selected_names)}"
)
def ask(question):
    if not question.strip():
        return "Please enter a question."
    return answer_question(question, embeddings, texts)

with gr.Blocks() as demo:
    gr.Markdown(header_md)
    question_input = gr.Textbox(label="Enter your question here", lines=2, placeholder="Type your question...")
    answer_output = gr.Textbox(label="Answer", interactive=False)
    ask_button = gr.Button("Ask")

    ask_button.click(ask, inputs=question_input, outputs=answer_output)

if __name__ == "__main__":
    demo.launch()
