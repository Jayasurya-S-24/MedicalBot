from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def store_embeddings(docs, persist_dir="embeddings/"):
    """
    Creates text embeddings using a Sentence Transformer and stores them in ChromaDB.
    """
    if not docs:
        print("No documents found to embed. Exiting.")
        return

    print(f"⚙️ Creating text embeddings using 'sentence-transformers/all-MiniLM-L6-v2'...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # This creates the vector store from our document chunks
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    print("Persisting embeddings to disk...")
    vectordb.persist()
    print(f"💾 Embeddings stored successfully in: {persist_dir}")
