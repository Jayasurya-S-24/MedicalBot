from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variable to store the chain
qa_chain = None

def initialize_chain():
    """Initialize the RAG chain at startup"""
    global qa_chain
    try:
        logger.info("Starting RAG chain initialization...")
        
        # Set environment variables for Hugging Face
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        
        from utils.rag_pipline import get_rag_chain
        qa_chain = get_rag_chain()
        logger.info("RAG chain initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {str(e)}")
        logger.error(traceback.format_exc())
        return False

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        "message": "Medical Chatbot API",
        "endpoints": {
            "/health": "GET - Check service health",
            "/ask": "POST - Ask a medical question"
        },
        "version": "1.0.0"
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if qa_chain is None:
        return jsonify({
            "status": "initializing",
            "message": "Model is still loading...",
            "ready": False
        }), 503
    return jsonify({
        "status": "ready",
        "message": "Service is ready",
        "ready": True
    }), 200

@app.route('/ask', methods=['POST'])
def ask_question():
    """Main endpoint for asking questions"""
    try:
        # Check if chain is initialized
        if qa_chain is None:
            logger.warning("Request received but chain not initialized")
            return jsonify({
                "error": "Model is still loading. Please wait and try again.",
                "status": "initializing"
            }), 503
        
        # Get the query from request
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' in request body",
                "example": {"query": "What is diabetes?"}
            }), 400
        
        query = data['query']
        
        # Validate query
        if not query.strip():
            return jsonify({
                "error": "Query cannot be empty"
            }), 400
        
        logger.info(f"Received query: {query[:100]}...")  # Log first 100 chars
        
        # Process the query
        logger.info("Processing query with RAG chain...")
        result = qa_chain({"query": query})
        
        # Extract response
        answer = result.get('result', 'No answer generated')
        source_docs = result.get('source_documents', [])
        
        logger.info("Query processed successfully")
        
        # Format response
        response = {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in source_docs
            ],
            "num_sources": len(source_docs)
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
            "type": type(e).__name__
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested URL was not found on the server"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500

# Render-specific: Initialize on import for gunicorn
if os.environ.get('RENDER'):
    logger.info("=" * 60)
    logger.info("Detected Render environment - Initializing on import")
    logger.info("=" * 60)
    initialize_chain()

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Starting Medical Chatbot Server")
    logger.info("=" * 60)
    
    # Initialize the chain before starting the server
    if not initialize_chain():
        logger.error("Failed to initialize. Exiting...")
        exit(1)
    
    # Get port from environment variable (Render sets this automatically)
    port = int(os.environ.get('PORT', 5000))
    
    logger.info(f"Starting Flask server on http://0.0.0.0:{port}")
    logger.info("=" * 60)
    
    # Use production-ready settings
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )