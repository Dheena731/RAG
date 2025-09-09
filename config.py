import os
from pathlib import Path

class Config:
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data" 
    EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
    CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"
    
      # Set this in environment
    LLM_MODEL = "mistralai/mistral-small-3.2-24b-instruct:free"  # or "openai/gpt-3.5-turbo", "meta-llama/llama-2-7b-chat"
    
    # Alternative: Local LLM settings  
    LOCAL_LLM_URL = "http://localhost:11434"  # Ollama default
    LOCAL_LLM_MODEL = "llama2"  # or "mistral", "codellama"
    
    # Generation settings
    USE_LOCAL_LLM = False  # Set to True to use local LLM instead of OpenRouter
    MAX_CONTEXT_DOCS = 1

    # Embedding model settings
    TEXT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    MULTIMODAL_EMBEDDING_MODEL = "sentence-transformers/all-minilm-l6-v2"  # Using text model for simplicity
    
    # Vector database settings
    COLLECTION_NAME = "multimodal_documents"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Supported file types
    SUPPORTED_TEXT_FORMATS = [".txt", ".pdf", ".docx", ".md"]
    SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    
    # Device settings
    DEVICE = "cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"
    
    def __init__(self):
        # Create necessary directories
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)