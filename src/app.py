import sys
from ingest import process_all_pdfs
from embed import embed_all_chunks
from rag import answer_question

def main():
    print("Welcome to DyreLLM!")
    print("Choose an option:")
    print("1. Process PDFs (ingest)")
    print("2. Create embeddings")
    print("3. Ask a question")
    print("4. Exit")

    choice = input("Enter your choice (1-4): ")

    if choice == "1":
        print("Processing PDFs...")
        process_all_pdfs()
        print("Done processing PDFs.")
    elif choice == "2":
        print("Creating embeddings for all chunks...")
        embed_all_chunks()
        print("Done creating embeddings.")
    elif choice == "3":
        question = input("Enter your question: ")
        answer = answer_question(question)
        print("\nAnswer:\n", answer)
    elif choice == "4":
        print("Goodbye!")
        sys.exit()
    else:
        print("Invalid choice, please select 1-4.")

if __name__ == "__main__":
    while True:
        main()
