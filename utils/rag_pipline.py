# # from langchain_community.vectorstores import Chroma
# # from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
# # from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
# # import torch
# # import os

# # def get_rag_chain(persist_dir="embeddings/text"):
# #     """
# #     Initializes the complete RAG chain:
# #     1. Loads the local BioMistral-7B model with 4-bit quantization.
# #     2. Loads the ChromaDB vector store.
# #     3. Creates a custom RAG chain to tie them together.
# #     """
# #     print("🔍 Loading BioMistral-7B model (4-bit quantized) and Chroma vector store...")

# #     # --- ACTION REQUIRED: SAFETENSORS CHECK ---
# #     try:
# #         import safetensors
# #     except ImportError:
# #         print("\n\n!! WARNING: 'safetensors' not installed. Please run: pip install safetensors")
# #         print("!! This improves model loading speed and stability for large models.\n")
# #     # ------------------------------------------

# #     # Define the model to load from Hugging Face
# #     model_id = "BioMistral/BioMistral-7B"
    
# #     # Define the offload folder path
# #     offload_folder = os.path.join(os.path.dirname(persist_dir), "offload_cache")
# #     os.makedirs(offload_folder, exist_ok=True)

# #     print(f"  -> Loading tokenizer for {model_id}")
# #     tokenizer = AutoTokenizer.from_pretrained(model_id)
    
# #     # Check system capabilities
# #     print(f"  -> CUDA available: {torch.cuda.is_available()}")
# #     if torch.cuda.is_available():
# #         print(f"  -> GPU: {torch.cuda.get_device_name(0)}")
# #         print(f"  -> GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
# #     print(f"  -> Loading model {model_id} with 4-bit quantization (this may take a while)...")
# #     os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
    
# #     try:
# #         # Configure 4-bit quantization to reduce memory usage by ~75%
# #         quantization_config = BitsAndBytesConfig(
# #             load_in_4bit=True,
# #             bnb_4bit_compute_dtype=torch.float16,
# #             bnb_4bit_use_double_quant=True,
# #             bnb_4bit_quant_type="nf4"
# #         )  # <-- FIXED: Added closing parenthesis here
        
# #         model = AutoModelForCausalLM.from_pretrained(
# #             model_id,
# #             quantization_config=quantization_config,
# #             device_map="auto",
# #             low_cpu_mem_usage=True,
# #             offload_folder=offload_folder,
# #             trust_remote_code=True
# #         )
# #         print("  -> Model loaded successfully with 4-bit quantization!")
# #         print(f"  -> Memory footprint reduced to ~3.5GB (from ~14GB)")
        
# #     except Exception as e:
# #         print(f"\n❌ ERROR loading model: {str(e)}")
# #         print("\n💡 Troubleshooting steps:")
# #         print("   1. Ensure you have at least 8GB RAM available")
# #         print("   2. Close other memory-intensive applications")
# #         print("   3. Check disk space - you need ~15GB free for model download")
# #         print("   4. Clear HuggingFace cache if needed: C:\\Users\\jayas\\.cache\\huggingface")
# #         raise

# #     # Create a transformers pipeline for text generation
# #     pipe = pipeline(
# #         "text-generation",
# #         model=model,
# #         tokenizer=tokenizer,
# #         max_new_tokens=512,
# #         temperature=0.2,
# #         top_p=0.9,
# #         do_sample=True,
# #         pad_token_id=tokenizer.eos_token_id
# #     )

# #     # Wrap the pipeline in a LangChain object
# #     llm = HuggingFacePipeline(pipeline=pipe)

# #     # Load the embedding model and the persistent vector store
# #     print("  -> Loading embeddings and vector store...")
# #     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# #     vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
# #     # Create a retriever (k=3 means it will fetch the top 3 relevant chunks)
# #     retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# #     # Custom RAG Chain class that mimics RetrievalQA behavior
# #     class RAGChain:
# #         """
# #         Custom RAG chain that retrieves documents and generates answers.
# #         Compatible with RetrievalQA interface.
# #         """
# #         def __init__(self, llm, retriever):
# #             self.llm = llm
# #             self.retriever = retriever
        
# #         def __call__(self, inputs):
# #             """
# #             Process a query and return results with source documents.
            
# #             Args:
# #                 inputs: Either a string query or dict with 'query' key
            
# #             Returns:
# #                 dict with 'query', 'result', and 'source_documents'
# #             """
# #             # Handle both string and dict inputs
# #             if isinstance(inputs, dict):
# #                 query = inputs.get("query", "")
# #             else:
# #                 query = inputs
            
# #             # Retrieve relevant documents
# #             source_docs = self.retriever.get_relevant_documents(query)
            
# #             # Combine context from retrieved documents
# #             context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" 
# #                                    for i, doc in enumerate(source_docs)])
            
# #             # Create the prompt
# #             prompt = f"""You are a medical assistant. Use the following context to answer the question accurately and concisely.

# # Context:
# # {context}

# # Question: {query}

# # Answer (provide a clear, accurate response based on the context):"""
            
# #             # Generate response using the LLM
# #             response = self.llm(prompt)
            
# #             return {
# #                 "query": query,
# #                 "result": response,
# #                 "source_documents": source_docs
# #             }
        
# #         def invoke(self, inputs):
# #             """Alias for __call__ to support LangChain-style invocation"""
# #             return self.__call__(inputs)
    
# #     # Create the RAG chain
# #     qa_chain = RAGChain(llm, retriever)
    
# #     print("✅ RAG chain is ready.")
# #     # return qa_chain




from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

def get_rag_chain(persist_dir="embeddings/text"):
    """
    Initializes the complete RAG chain with a CPU-friendly model.
    Uses a smaller medical model optimized for systems without GPU.
    """
    print("🔍 Loading medical model and Chroma vector store...")

    # Use a smaller, CPU-friendly model
    model_id = "microsoft/BioGPT"
    
    print(f"  -> Using CPU-optimized model: {model_id}")
    print(f"  -> This model is much lighter (~1.5GB) and works well on CPU")

    print(f"  -> Loading tokenizer for {model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        print(f"  -> Error loading tokenizer: {e}")
        print(f"  -> Trying with trust_remote_code=True...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"  -> Loading model {model_id}...")
    print(f"  -> CUDA available: {torch.cuda.is_available()}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print("  -> ✅ Model loaded successfully!")
        print(f"  -> Model size: ~1.5GB")
        
    except Exception as e:
        print(f"\n❌ ERROR loading model: {str(e)}")
        print("\n💡 Trying alternative approach...")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            print("  -> ✅ Model loaded successfully (fallback method)!")
        except Exception as e2:
            print(f"\n❌ Still failed: {str(e2)}")
            raise

    # Create a transformers pipeline for text generation
    print("  -> Creating text generation pipeline...")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
    )

    # Wrap the pipeline in a LangChain object
    llm = HuggingFacePipeline(pipeline=pipe)

    # Load the embedding model and the persistent vector store
    print("  -> Loading embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    # Create a retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Custom RAG Chain class
    class RAGChain:
        """
        Custom RAG chain that retrieves documents and generates answers.
        """
        def __init__(self, llm, retriever):
            self.llm = llm
            self.retriever = retriever
        
        def __call__(self, inputs):
            """Process a query and return results with source documents."""
            if isinstance(inputs, dict):
                query = inputs.get("query", "")
            else:
                query = inputs
            
            print(f"  -> Retrieving relevant documents for query...")
            
            # Use invoke() or get_relevant_documents() depending on version
            try:
                source_docs = self.retriever.invoke(query)
            except AttributeError:
                try:
                    source_docs = self.retriever.get_relevant_documents(query)
                except AttributeError:
                    source_docs = self.retriever.vectorstore.similarity_search(query, k=3)
            
            # Combine context from retrieved documents
            context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" 
                                   for i, doc in enumerate(source_docs)])
            
            # Create the prompt
            prompt = f"""Based on the medical context below, answer the question accurately and concisely.

Context:
{context}

Question: {query}

Answer:"""
            
            print(f"  -> Generating response...")
            
            # FIXED: Use invoke() or predict() instead of calling directly
            try:
                # Try invoke() first (newer LangChain)
                response = self.llm.invoke(prompt)
            except AttributeError:
                try:
                    # Try predict() (older LangChain)
                    response = self.llm.predict(prompt)
                except AttributeError:
                    # Fallback: call the underlying pipeline directly
                    response = self.llm.pipeline(prompt)[0]['generated_text']
            
            # Clean up the response if it includes the prompt
            if isinstance(response, str) and prompt in response:
                response = response.replace(prompt, "").strip()
            
            return {
                "query": query,
                "result": response,
                "source_documents": source_docs
            }
        
        def invoke(self, inputs):
            """Alias for __call__ to support LangChain-style invocation"""
            return self.__call__(inputs)
    
    # Create the RAG chain
    qa_chain = RAGChain(llm, retriever)
    
    print("✅ RAG chain is ready!")
    print("=" * 60)
    return qa_chain