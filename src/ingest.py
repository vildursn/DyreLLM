import fitz  # PyMuPDF
from pathlib import Path
from typing import List

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
    """Splits a string into chunks of approximately `chunk_size` words."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def process_all_pdfs():
    """Processes all PDF files in the raw directory and saves text chunks."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print("📂 No PDF files found in 'data/raw/'.")
        return

    for file in pdf_files:
        print(f"📄 Processing: {file.name}")
        text = extract_text_from_pdf(file)
        if not text.strip():
            print(f"⚠️ No text found in {file.name}, skipping.")
            continue

        chunks = split_text_into_chunks(text)

        for i, chunk in enumerate(chunks):
            out_file = PROCESSED_DIR / f"{file.stem}_chunk_{i:03d}.txt"
            out_file.write_text(chunk, encoding="utf-8")

        print(f"✅ Finished {file.name} – saved {len(chunks)} chunks.")


if __name__ == "__main__":
    process_all_pdfs()
