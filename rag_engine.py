import os
import uuid
import requests
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from typing import Dict, Any, List, Optional

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAGEngine")

# Important: This ensures OpenAI isn't used by default
Settings.embed_model = None
Settings.llm = None

class RAGEngine:
    def __init__(self, 
             base_dir: str = os.getcwd(), 
             ollama_base_url: str = "http://localhost:11434",
             ollama_model: str = "llama3:latest",  
             chroma_persist_dir: str = None,
             document_dir: str = None):
        
        self.base_dir = base_dir
        self.document_dir = document_dir or os.path.join(base_dir, "DocumentDir")
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.index = None
        self.embed_model = None
        self.llm = None
        
        # Set ChromaDB persistence directory, default to a subdirectory in base_dir
        self.chroma_persist_dir = chroma_persist_dir or os.path.join(base_dir, "chroma_db")
        
        # Dictionary to store chat sessions
        self.chat_sessions = {}
        
        # Create document directory if it doesn't exist
        if not os.path.exists(self.document_dir):
            os.makedirs(self.document_dir)
            logger.info(f"Created document directory: {self.document_dir}")
        
        # Create ChromaDB directory if it doesn't exist
        if not os.path.exists(self.chroma_persist_dir):
            os.makedirs(self.chroma_persist_dir)
            logger.info(f"Created ChromaDB persistence directory: {self.chroma_persist_dir}")
        
        # Verify Ollama is available before proceeding
        if not self._verify_ollama_connection():
            logger.warning(f"Cannot connect to Ollama at {self.ollama_base_url}. Continuing without LLM setup.")
        else:
            # Setup Ollama embedding and LLM
            self._setup_ollama()
            
            # Setup prompt templates
            self._setup_prompts()
            
            # Try to load existing index
            self._load_index()
    
    def _verify_ollama_connection(self) -> bool:
        """Verify connection to Ollama server with improved model checking and fallback"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            # Check if the specified model is available
            models = response.json().get("models", [])
            model_names = [model.get("name") for model in models]
            
            if not model_names:
                logger.warning("No models found in Ollama server response")
                return False
                
            # Extract base model name for comparison (e.g., "llama3" from "llama3:latest")
            base_model = self.ollama_model.split(':')[0] if ':' in self.ollama_model else self.ollama_model
            
            # More flexible model matching - check if any available model starts with our base model name
            if not any(model.lower().startswith(base_model.lower()) for model in model_names):
                logger.warning(f"Warning: Model {self.ollama_model} not found in available models: {', '.join(model_names)}")
                
                # Try to select a reasonable fallback model
                if model_names:
                    # Look for common models in order of preference
                    for preferred_model in ["llama3", "llama2", "mistral", "gemma"]:
                        for model in model_names:
                            if preferred_model.lower() in model.lower():
                                self.ollama_model = model
                                logger.info(f"Using fallback model: {self.ollama_model}")
                                return True
                    
                    # If no preferred model found, use the first available one
                    self.ollama_model = model_names[0]
                    logger.info(f"Using first available model: {self.ollama_model}")
                    return True
                    
                return False
                
            # Found exact or compatible model
            # Find the best matching model from available models
            exact_match = next((model for model in model_names if model.lower() == self.ollama_model.lower()), None)
            if exact_match:
                self.ollama_model = exact_match  # Use exact case from server
                logger.info(f"Using exact model match: {self.ollama_model}")
                return True
                
            # Use first compatible model
            compatible_models = [model for model in model_names if model.lower().startswith(base_model.lower())]
            if compatible_models:
                self.ollama_model = compatible_models[0]
                logger.info(f"Using compatible model: {self.ollama_model}")
                return True
                
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama server: {str(e)}")
            return False
    
    def _setup_ollama(self):
        """Setup Ollama embedding and LLM with improved error handling"""
        try:
            # Setup embedding model
            self.embed_model = OllamaEmbedding(
                model_name=self.ollama_model,
                base_url=self.ollama_base_url,
                embed_batch_size=10,  # Reduce batch size for stability
                ollama_additional_kwargs={"mirostat": 0},
            )
            
            # Important: Set the embedding model in Settings
            Settings.embed_model = self.embed_model
            
            # Setup LLM with increased timeout
            self.llm = Ollama(
                model=self.ollama_model, 
                base_url=self.ollama_base_url,
                request_timeout=180.0  # Increased timeout for longer responses
            )
            
            # Important: Set the LLM model in Settings
            Settings.llm = self.llm
            
            logger.info(f"Successfully set up Ollama embedding and LLM using model: {self.ollama_model}")
        except Exception as e:
            logger.error(f"Failed to setup Ollama models: {str(e)}", exc_info=True)
            logger.warning("Continuing without properly configured Ollama models")
            # Don't raise an exception, allow the application to continue in degraded mode
    
    def _setup_prompts(self):
        """Setup prompt templates for query and refinement"""
        # Text QA template
        text_qa_template_str = (
            "Context information is"
            " below.\n---------------------\n{context_str}\n---------------------\n"
            "Using both the context information and also using your own knowledge, answer"
            " the question: {query_str}\nIf the context isn't helpful, you can also"
            " answer the question on your own.\n"
            " answer using facts but keep it clean and concise so that everyone can understand clearly"
            " ensure you understand the users query and ask follow up questions if required"
            " format the response and ensure it is presentable"
            " Create table structure where needed in the response"
            " Be conversational and maintain continuity with previous messages"
        )
        self.text_qa_template = PromptTemplate(text_qa_template_str)
        
        # Refine template
        refine_template_str = (
            " You are a senior subject matter expert in the banking and finance domain"
            " your speciality is payments. The queries you will get will be related to payments"
            " Your users will be software developers, testers, product owners"
            " Users will need help with Acceptance Criteria Generation"
            " Test Design, Code review etc. Keeping the context in mind answer the question"
            " The original question is as follows:\n {query_str} \n We have provided an"
            " existing answer: {existing_answer}\n We have the opportunity to refine"
            " the existing answer meeting the corporate standards with some more context"
            " \n------------\n{context_msg}\n------------\n Using both the new"
            " context and your own knowledge, update or repeat the existing answer.\n"
            " ensure there is enough space above and below the query to maintain proper document format"
            " Be precise with the answer and ensure answer is in tabular format where needed"
            " Be conversational and maintain continuity with previous messages"
        )
        self.refine_template = PromptTemplate(refine_template_str)
    
    def _load_index(self) -> bool:
        """Attempt to load existing index from ChromaDB with improved error handling"""
        try:
            # Initialize ChromaDB client
            chroma_client = chromadb.PersistentClient(path=self.chroma_persist_dir)
            
            # Check if collection exists
            try:
                collection = chroma_client.get_collection("documents")
                doc_count = collection.count()
                logger.info(f"Found existing ChromaDB collection with {doc_count} documents")
                
                if doc_count == 0:
                    logger.warning("Collection exists but contains no documents")
                    return False
                
                # Create vector store from the existing collection
                vector_store = ChromaVectorStore(chroma_collection=collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                
                # Load index from the vector store
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    storage_context=storage_context,
                    embed_model=self.embed_model  # Explicitly pass the embedding model
                )
                logger.info("Successfully loaded existing index from ChromaDB")
                return True
            except ValueError as e:
                if "Collection not found" in str(e):
                    logger.info("No existing collection found in ChromaDB")
                else:
                    logger.error(f"Error accessing ChromaDB collection: {str(e)}", exc_info=True)
                return False
            except Exception as e:
                logger.error(f"Unexpected error loading collection: {str(e)}", exc_info=True)
                return False
        except Exception as e:
            logger.error(f"Error connecting to ChromaDB: {str(e)}", exc_info=True)
            return False
    
    def ingest_documents(self) -> Dict[str, Any]:
        """Ingest documents and create index, storing in ChromaDB with improved error handling"""
    
        logger.info(f"Ingesting documents from {self.document_dir}...")
        
        try:
            # Load documents
            if not os.path.exists(self.document_dir):
                return {"status": "error", "message": f"Document directory does not exist: {self.document_dir}"}
                
            files = os.listdir(self.document_dir)
            if not files:
                return {"status": "error", "message": f"No files found in document directory: {self.document_dir}"}
            
            documents = SimpleDirectoryReader(input_dir=self.document_dir).load_data()
            logger.info(f"Loaded {len(documents)} documents")
            
            if len(documents) == 0:
                return {"status": "warning", "message": "No documents found to ingest"}
            
            # Initialize ChromaDB
            chroma_client = chromadb.PersistentClient(path=self.chroma_persist_dir)
            
            # Remove existing collection if it exists
            try:
                chroma_client.delete_collection("documents")
                logger.info("Deleted existing collection")
            except ValueError:
                logger.info("No existing collection to delete")
            except Exception as e:
                logger.error(f"Error deleting collection: {str(e)}", exc_info=True)
                logger.warning("Continuing despite collection deletion error")
            
            # Create new collection
            collection = chroma_client.create_collection("documents")
            
            # Create vector store with the collection
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create index with the explicit embedding model
            self.index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=storage_context,
                embed_model=self.embed_model  # Explicitly pass the embedding model
            )
            logger.info("Index created and stored in ChromaDB successfully")
            
            return {"status": "success", "document_count": len(documents)}
        except Exception as e:
            logger.error(f"Error ingesting documents: {str(e)}", exc_info=True)
            return {"status": "error", "message": f"Failed to ingest documents: {str(e)}"}
    
    def load_data(self) -> Dict[str, Any]:
        """
        Load index from ChromaDB if available, otherwise create new index.
        This provides backward compatibility with existing code.
        """
        try:
            if self.index is None:
                if not self._load_index():
                    return self.ingest_documents()
                else:
                    return {"status": "success", "message": "Loaded existing index"}
            else:
                return {"status": "success", "message": "Index already loaded"}
        except Exception as e:
            logger.error(f"Error in load_data: {str(e)}", exc_info=True)
            return {"status": "error", "message": f"Failed to load data: {str(e)}"}
    
    def create_chat_session(self) -> str:
        """Create a new chat session and return session ID"""
        session_id = str(uuid.uuid4())
        self.chat_sessions[session_id] = ChatMemoryBuffer.from_defaults(token_limit=2000)
        logger.info(f"Created new chat session: {session_id}")
        return session_id
    
    def get_chat_sessions(self) -> List[Dict[str, Any]]:
        """Get list of all active chat sessions with metadata"""
        return [
            {
                "session_id": sid,
                "message_count": len(memory.get_chat_history().messages) if hasattr(memory, 'get_chat_history') else 0
            }
            for sid, memory in self.chat_sessions.items()
        ]
    
    def clear_chat_session(self, session_id: str) -> bool:
        """Clear a specific chat session"""
        if session_id in self.chat_sessions:
            try:
                self.chat_sessions[session_id].clear()
                logger.info(f"Cleared chat session: {session_id}")
                return True
            except Exception as e:
                logger.error(f"Error clearing chat session {session_id}: {str(e)}", exc_info=True)
                return False
        logger.warning(f"Attempted to clear non-existent session: {session_id}")
        return False
    
    def delete_chat_session(self, session_id: str) -> bool:
        """Delete a specific chat session"""
        if session_id in self.chat_sessions:
            try:
                del self.chat_sessions[session_id]
                logger.info(f"Deleted chat session: {session_id}")
                return True
            except Exception as e:
                logger.error(f"Error deleting chat session {session_id}: {str(e)}", exc_info=True)
                return False
        logger.warning(f"Attempted to delete non-existent session: {session_id}")
        return False
    
    def answer_question(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Answer a question using the RAG engine with optional chat history and improved error handling"""
        try:
            if not self.index:
                try:
                    # Attempt to load the index automatically
                    load_result = self.load_data()
                    if load_result.get("status") != "success":
                        return {
                            'question': question,
                            'answer': f"I'm sorry, I couldn't access my knowledge base: {load_result.get('message', 'Unknown error')}",
                            'raw_answer': f"Error: {load_result.get('message', 'Index not initialized')}",
                            'sources': [],
                            'session_id': session_id,
                            'error': True
                        }
                except Exception as e:
                    logger.error(f"Failed to load index: {str(e)}", exc_info=True)
                    return {
                        'question': question,
                        'answer': "I'm sorry, I couldn't access my knowledge base. Please try again later or contact support.",
                        'raw_answer': f"Error: Index not initialized and failed to load: {str(e)}",
                        'sources': [],
                        'session_id': session_id,
                        'error': True
                    }
            
            # Use existing session or create a new one if not provided
            memory = None
            is_new_session = False
            
            if session_id:
                if session_id not in self.chat_sessions:
                    # Create a new session with the provided ID
                    try:
                        self.chat_sessions[session_id] = ChatMemoryBuffer.from_defaults(token_limit=2000)
                        is_new_session = True
                        logger.info(f"Created new chat session on-demand: {session_id}")
                    except Exception as e:
                        logger.error(f"Error creating chat session: {str(e)}", exc_info=True)
                        # Continue without memory rather than failing
                memory = self.chat_sessions.get(session_id)
            
            # Create query engine with templates and memory if available
            try:
                # Get chat history as a string
                chat_history = ""
                if memory and hasattr(memory, 'get_chat_history'):
                    chat_history_obj = memory.get_chat_history()
                    if chat_history_obj and hasattr(chat_history_obj, 'messages'):
                        # Format chat history as string
                        history_msgs = []
                        for msg in chat_history_obj.messages:
                            role = getattr(msg, 'role', None) or getattr(msg, 'type', 'unknown')
                            content = getattr(msg, 'content', '')
                            history_msgs.append(f"{role}: {content}")
                        
                        chat_history = "\n".join(history_msgs)
                
                # Create a modified template with chat_history
                current_qa_template = self.text_qa_template.template
                if "{chat_history}" in current_qa_template:
                    # Replace chat_history placeholder with actual history or empty string
                    from copy import deepcopy
                    modified_qa_template = deepcopy(self.text_qa_template)
                    modified_qa_template.template = current_qa_template.replace("{chat_history}", chat_history)
                else:
                    modified_qa_template = self.text_qa_template
                
                # Create query engine with the modified template
                query_engine = self.index.as_query_engine(
                    text_qa_template=modified_qa_template,
                    refine_template=self.refine_template,
                    similarity_top_k=2,
                    chat_memory=memory,
                    llm=self.llm,  # Explicitly use the configured LLM
                    embed_model=self.embed_model  # Explicitly use the configured embedding model
                )
                
                # Get response
                response = query_engine.query(question)
                logger.info(f"Generated response for question: {question[:50]}...")
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}", exc_info=True)
                return {
                    'question': question,
                    'answer': f"I'm sorry, I encountered an error while processing your question. Please try again or rephrase your query.",
                    'raw_answer': f"Error: {str(e)}",
                    'sources': [],
                    'session_id': session_id,
                    'error': True
                }
            
            # Store the interaction in memory if available
            if memory:
                try:
                    if hasattr(response, 'response'):
                        memory.put(question, response.response)
                        if not is_new_session:
                            logger.info(f"Updated chat memory for session: {session_id}")
                except Exception as e:
                    logger.error(f"Error updating chat memory: {str(e)}", exc_info=True)
                    # Continue despite chat memory error
            
            # Format response with source information
            source_documents = []
            final_response = ""
            
            # More robust response handling
            try:
                if hasattr(response, 'response'):
                    final_response = response.response
                elif hasattr(response, 'get_formatted_sources'):
                    # Some llama_index versions may use different response structure
                    final_response = str(response)
                else:
                    final_response = str(response)
                    logger.warning(f"Response object has unexpected structure: {type(response)}")
            except Exception as e:
                logger.error(f"Error extracting response content: {str(e)}", exc_info=True)
                final_response = "I found some information, but had trouble formatting it."
            
            # Extract source information with improved error handling
            try:
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    source_nodes = response.source_nodes
                    
                    # Extract source documents for API response
                    for node in source_nodes:
                        try:
                            metadata = {}
                            if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                                metadata = node.node.metadata
                            
                            source_doc = {
                                'text': node.text[:200] + "..." if len(node.text) > 200 else node.text,
                                'score': float(node.score) if hasattr(node, 'score') else None,
                                'file_name': metadata.get('file_name', 'Unknown'),
                            }
                            
                            if 'page_label' in metadata:
                                source_doc['page_label'] = metadata['page_label']
                            
                            source_documents.append(source_doc)
                        except Exception as e:
                            logger.error(f"Error processing source node: {str(e)}", exc_info=True)
                    
                    # Add source citation to response
                    if source_nodes:
                        source_info = []
                        for i, node in enumerate(source_nodes[:2]):  # Only use first two sources
                            try:
                                if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                                    metadata = node.node.metadata
                                    file_name = metadata.get('file_name', 'Unknown')
                                    page_label = metadata.get('page_label', '')
                                    
                                    if page_label:
                                        source_info.append(f"{file_name} (page: {page_label})")
                                    else:
                                        source_info.append(file_name)
                            except Exception as e:
                                logger.error(f"Error formatting source info: {str(e)}", exc_info=True)
                        
                        if source_info:
                            final_response += '\n\nSources: ' + ', '.join(source_info)
            except Exception as e:
                logger.error(f"Error processing source nodes: {str(e)}", exc_info=True)
                # Continue without source information rather than failing
            
            # Format API response
            result = {
                'question': question,
                'answer': final_response,
                'raw_answer': final_response,
                'sources': source_documents,
                'session_id': session_id
            }
            
            return result
        except Exception as e:
            logger.error(f"Unexpected error in answer_question: {str(e)}", exc_info=True)
            return {
                'question': question,
                'answer': "I apologize, but I encountered an unexpected error. Please try again later.",
                'raw_answer': f"Unexpected error: {str(e)}",
                'sources': [],
                'session_id': session_id,
                'error': True
            }