from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import logging

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
        from utils.rag_pipline import get_rag_chain
        qa_chain = get_rag_chain()
        logger.info("RAG chain initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {str(e)}")
        logger.error(traceback.format_exc())
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if qa_chain is None:
        return jsonify({
            "status": "initializing",
            "message": "Model is still loading..."
        }), 503
    return jsonify({
        "status": "ready",
        "message": "Service is ready"
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
                "error": "Missing 'query' in request body"
            }), 400
        
        query = data['query']
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
                    "content": doc.page_content[:200] + "...",  # First 200 chars
                    "metadata": doc.metadata
                }
                for doc in source_docs
            ]
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("Starting Medical Chatbot Server")
    logger.info("=" * 50)
    
    # Initialize the chain before starting the server
    if not initialize_chain():
        logger.error("Failed to initialize. Exiting...")
        exit(1)
    
    logger.info("Starting Flask server on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)