import os
import fitz  # PyMuPDF
import json
from typing import List, Dict, Any

class Document:
    """Simple document class"""
    # CORRECTED: Changed _init_ to __init__
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Simple text splitter"""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    
    return chunks

def load_and_split_docs(data_dir: str = "data") -> List[Document]:
    """Load and split PDFs and TXT files"""
    docs = []
    print(f"Loading documents from {data_dir}...")

    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)

        if filename.endswith(".pdf"):
            try:
                # Assuming the user has installed PyMuPDF as advised previously: pip install PyMuPDF
                with fitz.open(file_path) as pdf:
                    text = ""
                    for page in pdf:
                        text += page.get_text("text")

                if text:
                    chunks = split_text(text, chunk_size=1000, chunk_overlap=200)
                    for i, chunk in enumerate(chunks):
                        docs.append(Document(
                            page_content=chunk,
                            metadata={"source": filename, "chunk": i}
                        ))
                    print(f"  -> Loaded {filename} ({len(chunks)} chunks)")
            except Exception as e:
                print(f"  [!] Error loading PDF {filename}: {e}")

        elif filename.endswith(".txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as txt_file:
                    text = txt_file.read()

                if text:
                    chunks = split_text(text, chunk_size=1000, chunk_overlap=200)
                    for i, chunk in enumerate(chunks):
                        docs.append(Document(
                            page_content=chunk,
                            metadata={"source": filename, "chunk": i}
                        ))
                    print(f"  -> Loaded {filename} ({len(chunks)} chunks)")
            except Exception as e:
                print(f"  [!] Error loading TXT {filename}: {e}")

    print(f"✅ Total: {len(docs)} chunks")
    return docs

# CORRECTED: Changed _name_ and _file_ to __name__ and __file__
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")
    embeddings_dir = os.path.join(project_root, "embeddings")
    
    # Create embeddings directory if it doesn't exist
    os.makedirs(embeddings_dir, exist_ok=True)
    
    print(f"Looking for data in: {data_dir}")
    
    if os.path.exists(data_dir):
        docs = load_and_split_docs(data_dir)
        
        # Save chunks to the embeddings folder
        output_file = os.path.join(embeddings_dir, "chunks.json")
        chunks_data = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved {len(docs)} chunks to: {output_file}")
    else:
        print(f"Error: Data folder not found at {data_dir}")
        print("Please create a 'data' folder and add PDF/TXT files to it")