"""
Flask API for RAG Application with decoupled document ingestion
"""
from flask import Flask, request, jsonify
from rag_engine import RAGEngine
import os
import time
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load configuration
DEBUG = os.environ.get("DEBUG", "True").lower() == "true"
PORT = int(os.environ.get("PORT", 5001))
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", os.path.join(os.getcwd(), "chroma_db"))

# Initialize RAG engine with ChromaDB persistence
rag_engine = RAGEngine(
    base_dir=os.getcwd(), 
    ollama_base_url=OLLAMA_BASE_URL,
    chroma_persist_dir=CHROMA_PERSIST_DIR
)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "timestamp": time.time(),
        "ollama_url": OLLAMA_BASE_URL,
        "document_dir": rag_engine.document_dir,
        "chroma_persist_dir": rag_engine.chroma_persist_dir,
        "index_loaded": rag_engine.index is not None
    })

@app.route('/api/ask', methods=['POST'])
def ask_question():
    
    # Check if RAG engine is initialized
    if rag_engine.index is None:
        try:
            rag_engine.load_data()
        except Exception as e:
            return jsonify({"error": f"Failed to initialize RAG engine: {str(e)}"}), 500
    
    # Get question from request
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' field in request"}), 400
    
    question = data['question']
    if not question or not isinstance(question, str):
        return jsonify({"error": "Question must be a non-empty string"}), 400
    
    # Process question
    try:
        result = rag_engine.answer_question(question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Failed to process question: {str(e)}"}), 500

@app.route('/api/ingest', methods=['POST'])
def ingest_data():
    """Explicitly trigger document ingestion and indexing"""
    try:
        result = rag_engine.ingest_documents()
        return jsonify({
            "status": "success", 
            "message": "Documents ingested successfully",
            "details": result
        })
    except Exception as e:
        return jsonify({"error": f"Failed to ingest documents: {str(e)}"}), 500

@app.route('/api/reload', methods=['POST'])
def reload_data():
    """Reload index from ChromaDB without reingesting documents"""
    try:
        result = rag_engine.load_data()
        return jsonify({
            "status": "success", 
            "message": "Data loaded successfully",
            "details": result
        })
    except Exception as e:
        return jsonify({"error": f"Failed to reload data: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Simple chat interface for direct interaction.
    
    Expected JSON input:
    {
        "message": "Your question here"
    }
    
    Returns the formatted response directly, not in JSON format.
    """
    # Check if RAG engine is initialized
    if rag_engine.index is None:
        try:
            rag_engine.load_data()
        except Exception as e:
            return jsonify({"error": f"Failed to initialize RAG engine: {str(e)}"}), 500
    
    # Get message from request
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "Missing 'message' field in request"}), 400
    
    message = data['message']
    if not message or not isinstance(message, str):
        return jsonify({"error": "Message must be a non-empty string"}), 400
    
    # Exit command (optional)
    if message.lower() == 'exit':
        return "Thank you for reaching out. Feel free to visit us again.. byee"
    
    # Process question
    try:
        result = rag_engine.answer_question(message)
        # Return just the answer, not the full JSON
        return result['answer']
    except Exception as e:
        return jsonify({"error": f"Failed to process message: {str(e)}"}), 500

if __name__ == '__main__':
    # Try to load existing index on startup
    try:
        load_result = rag_engine.load_data()
        print(f"Index loading status: {load_result}")
    except Exception as e:
        print(f"Warning: Failed to load index on startup: {e}")
        print("You can load data later using the /api/reload endpoint")
        print("Or ingest documents using the /api/ingest endpoint")
    
    # Start the Flask app
    app.run(debug=DEBUG, host="0.0.0.0", port=PORT)