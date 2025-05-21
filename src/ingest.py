import fitz  # PyMuPDF
from pathlib import Path
from typing import List
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CHUNK_SIZE = 500  # words


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extracts all text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        print(f"⚠️ Error reading {pdf_path.name}: {e}")
        return ""


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Splits text into chunks of approximately `chunk_size` words, preserving sentences."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        words = sentence.split()
        if current_length + len(words) > chunk_size and current_chunk:
            # Start a new chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def process_all_pdfs():
    """Processes all PDF files in the raw directory tree and saves text chunks, mirroring the folder structure."""
    pdf_files = list(RAW_DIR.rglob("*.pdf"))
    if not pdf_files:
        print("📂 No PDF files found in 'data/raw/'.")
        return

    for file in pdf_files:
        print(f"📄 Processing: {file.relative_to(RAW_DIR)}")
        text = extract_text_from_pdf(file)
        if not text.strip():
            print(f"⚠️ No text found in {file.name}, skipping.")
            continue

        chunks = split_text_into_chunks(text)

        # Mirror the folder structure in PROCESSED_DIR
        relative_folder = file.parent.relative_to(RAW_DIR)
        out_folder = PROCESSED_DIR / relative_folder
        out_folder.mkdir(parents=True, exist_ok=True)

        for i, chunk in enumerate(chunks):
            out_file = out_folder / f"{file.stem}_chunk_{i:03d}.txt"
            out_file.write_text(chunk, encoding="utf-8")

        print(f"✅ Finished {file.relative_to(RAW_DIR)} – saved {len(chunks)} chunks.")


if __name__ == "__main__":
    process_all_pdfs()
