import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    UPLOAD_FOLDER = 'uploads'
    EXTRACT_FOLDER = 'extracted_files'
    ALLOWED_EXTENSIONS = {'zip'}
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
    OLLAMA_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "llama3.2"
    TEMPERATURE = 0.3
    CONTEXT_LENGTH = 15000