from celery import Celery
from ..models.document_processor import DocumentProcessor
from ..models.embedding_model import EmbeddingModel
from ..models.search_engine import SearchEngine
from ..config import Config
import os

config = Config()

celery = Celery(__name__)
celery.conf.broker_url = config.CELERY_BROKER_URL
celery.conf.result_backend = config.CELERY_RESULT_BACKEND

@celery.task(bind=True)
def process_uploaded_files(self, zip_path: str, user_id: str):
    self.update_state(state='PROCESSING', meta={'status': 'Extracting files...'})
    
    # Process the zip file
    processor = DocumentProcessor(config.UPLOAD_FOLDER, config.PROCESSED_FOLDER)
    document_tree, file_paths = processor.process_uploaded_zip(zip_path, user_id)
    
    self.update_state(state='PROCESSING', meta={
        'status': 'Extracting text from documents...',
        'document_tree': document_tree
    })
    
    # Extract text from all files
    documents = processor.extract_text_from_files(file_paths)
    
    self.update_state(state='PROCESSING', meta={
        'status': 'Generating embeddings...',
        'document_tree': document_tree
    })
    
    # Generate embeddings
    embedding_model = EmbeddingModel()
    texts = [doc['text'] for doc in documents]
    embeddings = embedding_model.embed_documents(texts)
    
    # Build search index
    search_engine = SearchEngine()
    search_engine.build_index(embeddings, documents)
    
    return {
        'status': 'Processing complete',
        'document_tree': document_tree,
        'num_documents': len(documents)
    }