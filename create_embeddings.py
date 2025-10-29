import os
from utils.pdf_loader import load_and_split_docs
from utils.text_embedder import store_embeddings

DATA_DIR = "data"
DB_DIR = "embeddings/text"

if __name__ == "__main__":
    # Ensure the data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Data directory '{DATA_DIR}' not found.")
        print("Please create it and add your PDF or .txt files.")
    else:
        # Step 1: Load and split the documents
        docs = load_and_split_docs(DATA_DIR)
        
        if docs:
            # Step 2: Create and store embeddings
            store_embeddings(docs, DB_DIR)
        else:
            print("No documents were loaded. Exiting.")
