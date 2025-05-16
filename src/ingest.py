import os
import fitz  # PyMuPDF
from pathlib import Path
from typing import List

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CHUNK_SIZE = 500  # ord


def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


def process_all_pdfs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for file in RAW_DIR.glob("*.pdf"):
        print(f"Processing {file.name}")
        text = extract_text_from_pdf(file)
        chunks = split_text_into_chunks(text)

        for i, chunk in enumerate(chunks):
            out_file = PROCESSED_DIR / f"{file.stem}_chunk_{i:03d}.txt"
            out_file.write_text(chunk, encoding="utf-8")


if __name__ == "__main__":
    process_all_pdfs()