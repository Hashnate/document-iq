import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    PROCESSED_FOLDER = os.getenv('PROCESSED_FOLDER', 'processed')
    ALLOWED_EXTENSIONS = {'zip'}
    MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1GB max upload
    
    # Celery
    CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
    
    # AI
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama2')
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    
    # FAISS
    FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', 'faiss_index')
    
    # GPU
    USE_GPU = True