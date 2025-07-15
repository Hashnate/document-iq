import os
import zipfile
import PyPDF2
import docx
import shutil
import requests
import uuid
import json
import time
from datetime import datetime, timezone, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, stream_with_context, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
import threading
import re
import hashlib
from pdfminer.high_level import extract_text as pdfminer_extract
import pdfplumber
import markdown
from bs4 import BeautifulSoup
import csv
import xml.etree.ElementTree as ET
from flask import send_file
from flask_migrate import Migrate
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_session import Session

# GPU-optimized imports
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import text, func
from langchain.text_splitter import RecursiveCharacterTextSplitter
from functools import partial

# Load environment variables
load_dotenv()

# GPU environment optimization
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Initialize Flask app
app = Flask(__name__)

# Enhanced Configuration for RTX A6000
class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    UPLOAD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'uploads')
    EXTRACT_FOLDER = os.path.join(os.getcwd(), 'extracted_files')
    CACHE_FOLDER = os.path.join(os.getcwd(), 'file_cache')
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'md', 'csv', 'json', 'xml', 'html'}
    MAX_CONTENT_LENGTH = 1000 * 1024 * 1024  # 1GB
    OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434/api/generate')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3:70b-instruct-q4_K_M')
    OLLAMA_TIMEOUT = 600
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB per file
    MAX_RETRIES = 3
    LOG_FILE = os.path.join(os.getcwd(), 'app.log')
    LOG_LEVEL = logging.INFO
    MAX_CONTEXT_LENGTH = 1000000  # 1M characters
    STREAM_TIMEOUT = 500
    CACHE_EXPIRATION_SECONDS = 3600
    PDF_EXTRACTION_METHOD = 'pdfminer'

    # GPU-optimized RAG configuration
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Better model for GPU
    CHUNK_SIZE = 512  # Optimal for transformer models
    CHUNK_OVERLAP = 50  # Reduced for speed
    SIMILARITY_TOP_K = 10
    VECTOR_STORE_PATH = os.path.join(os.getcwd(), 'vector_store')
    MAX_PARALLEL_PROCESSES = 16  # Utilize 36 cores
    EMBEDDING_BATCH_SIZE = 256  # Utilize 48GB VRAM
    
    # Database optimization
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 20,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'max_overflow': 30
    }

# Apply configuration
app.config.from_object(Config)

# Session configuration - FIXED
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = '/tmp/flask_sessions'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'documentiq:'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)
app.config['SESSION_COOKIE_NAME'] = 'documentiq_session'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///documentiq.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Fatora configuration
app.config['FATORA_API_KEY'] = os.getenv('FATORA_API_KEY')
app.config['FATORA_BASE_URL'] = 'https://api.fatora.io/v1'
app.config['FATORA_WEBHOOK_SECRET'] = os.getenv('FATORA_WEBHOOK_SECRET')

# Proxy configuration
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Create session directory
os.makedirs('/tmp/flask_sessions', exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['EXTRACT_FOLDER'], exist_ok=True)
os.makedirs(app.config['CACHE_FOLDER'], exist_ok=True)

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
migrate = Migrate(app, db)

# Initialize Session
Session(app)

# Configure logging
handler = RotatingFileHandler(
    app.config['LOG_FILE'],
    maxBytes=10 * 1024 * 1024,
    backupCount=3
)
handler.setLevel(app.config['LOG_LEVEL'])
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Subscription Plans
class SubscriptionPlan:
    BASIC = {
        'id': 'basic',
        'name': 'Basic',
        'price': 99,
        'features': ['50 documents/month', '10MB max file size', 'Basic support'],
        'limits': {'max_documents': 50, 'max_file_size': 10 * 1024 * 1024, 'max_conversations': 10}
    }
    
    PRO = {
        'id': 'pro',
        'name': 'Professional',
        'price': 199,
        'features': ['200 documents/month', '50MB max file size', 'Priority support', 'Advanced analytics'],
        'limits': {'max_documents': 200, 'max_file_size': 50 * 1024 * 1024, 'max_conversations': 50}
    }
    
    TRIAL = {
        'id': 'trial',
        'name': 'Free Trial',
        'price': 0,
        'features': ['Unlimited documents for 14 days', '100MB max file size', 'All Professional features', 'No credit card required'],
        'limits': {'max_documents': None, 'max_file_size': 100 * 1024 * 1024, 'max_conversations': None, 'trial_days': 14}
    }

    @classmethod
    def get_plan(cls, plan_id):
        plans = {'basic': cls.BASIC, 'pro': cls.PRO, 'trial': cls.TRIAL}
        return plans.get(plan_id)

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)
    conversations = db.relationship('Conversation', backref='user', lazy=True)
    
    # Subscription fields
    subscription_plan = db.Column(db.String(20), default='free')
    subscription_status = db.Column(db.String(20), default='inactive')
    subscription_id = db.Column(db.String(100))
    payment_method_id = db.Column(db.String(100))
    current_period_end = db.Column(db.DateTime, nullable=True)
    cancel_at_period_end = db.Column(db.Boolean, default=False)
    documents_uploaded = db.Column(db.Integer, default=0)
    last_billing_date = db.Column(db.DateTime)
    trial_start_date = db.Column(db.DateTime)
    trial_end_date = db.Column(db.DateTime)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def _make_aware(self, dt):
        """Convert naive datetime to timezone-aware UTC datetime"""
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    
    @property
    def is_subscribed(self):
        now = datetime.now(timezone.utc)
        
        if self.subscription_status == 'active' and (
            self.current_period_end is None or 
            self._make_aware(self.current_period_end) > now
        ):
            return True
        
        if self.subscription_plan == 'trial' and self.trial_end_date and self._make_aware(self.trial_end_date) > now:
            return True
        
        return False
    
    @property
    def plan_details(self):
        return SubscriptionPlan.get_plan(self.subscription_plan) or {}
    
    def can_upload_file(self, file_size):
        if self.is_admin:
            return True
        
        if self.subscription_plan == 'trial' and self.is_subscribed:
            return True
        
        plan = self.plan_details.get('limits', {})
        max_size = plan.get('max_file_size', 5 * 1024 * 1024)
        if file_size > max_size:
            return False
        
        if plan.get('max_documents') is not None:
            if self.documents_uploaded >= plan['max_documents']:
                if (self.last_billing_date and 
                    self._make_aware(self.last_billing_date).month == datetime.now(timezone.utc).month):
                    return False
                else:
                    self.documents_uploaded = 0
                    db.session.commit()
                
        return True

    @property
    def trial_days_remaining(self):
        """Calculate trial days remaining with timezone-aware handling"""
        if not self.trial_end_date or self.subscription_plan != 'trial':
            return 0
        
        now = datetime.now(timezone.utc)
        trial_end = self._make_aware(self.trial_end_date)
        
        days_left = (trial_end - now).days
        return max(0, days_left)

class Payment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    currency = db.Column(db.String(3), default='QAR')
    payment_intent_id = db.Column(db.String(100))
    payment_method = db.Column(db.String(50))
    status = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    receipt_url = db.Column(db.String(255))

class BillingHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    description = db.Column(db.String(255))
    amount = db.Column(db.Float, nullable=False)
    currency = db.Column(db.String(3), default='QAR')
    status = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    invoice_id = db.Column(db.String(100))

class Conversation(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(100), nullable=False, default='New Query')
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    messages = db.relationship('Message', backref='conversation', lazy=True, cascade='all, delete-orphan')

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.String(36), db.ForeignKey('conversation.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    sources = db.Column(db.JSON)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

class FileCache(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_hash = db.Column(db.String(32), unique=True, nullable=False)
    content = db.Column(db.Text, nullable=False)
    file_metadata = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_path = db.Column(db.String(512), nullable=False)
    file_hash = db.Column(db.String(64), nullable=False)
    file_type = db.Column(db.String(32), nullable=False)
    processed_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    document_metadata = db.Column(db.JSON, name="metadata")

class DocumentChunk(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=False)
    chunk_text = db.Column(db.Text, nullable=False)
    chunk_hash = db.Column(db.String(64), nullable=False)
    chunk_metadata = db.Column(db.JSON, name="metadata")
    vector_id = db.Column(db.String(64))
    document = db.relationship('Document', backref='chunks')

def migrate_existing_datetimes():
    """One-time migration to convert naive datetimes to timezone-aware"""
    try:
        print("Starting datetime migration...")
        
        # Update User table
        users = User.query.all()
        for user in users:
            if user.created_at and user.created_at.tzinfo is None:
                user.created_at = user.created_at.replace(tzinfo=timezone.utc)
            if user.last_login and user.last_login.tzinfo is None:
                user.last_login = user.last_login.replace(tzinfo=timezone.utc)
            if user.current_period_end and user.current_period_end.tzinfo is None:
                user.current_period_end = user.current_period_end.replace(tzinfo=timezone.utc)
            if user.last_billing_date and user.last_billing_date.tzinfo is None:
                user.last_billing_date = user.last_billing_date.replace(tzinfo=timezone.utc)
            if user.trial_start_date and user.trial_start_date.tzinfo is None:
                user.trial_start_date = user.trial_start_date.replace(tzinfo=timezone.utc)
            if user.trial_end_date and user.trial_end_date.tzinfo is None:
                user.trial_end_date = user.trial_end_date.replace(tzinfo=timezone.utc)
        
        # Update other tables
        for payment in Payment.query.all():
            if payment.created_at and payment.created_at.tzinfo is None:
                payment.created_at = payment.created_at.replace(tzinfo=timezone.utc)
                
        for billing in BillingHistory.query.all():
            if billing.created_at and billing.created_at.tzinfo is None:
                billing.created_at = billing.created_at.replace(tzinfo=timezone.utc)
                
        for conv in Conversation.query.all():
            if conv.created_at and conv.created_at.tzinfo is None:
                conv.created_at = conv.created_at.replace(tzinfo=timezone.utc)
            if conv.updated_at and conv.updated_at.tzinfo is None:
                conv.updated_at = conv.updated_at.replace(tzinfo=timezone.utc)
                
        for msg in Message.query.all():
            if msg.timestamp and msg.timestamp.tzinfo is None:
                msg.timestamp = msg.timestamp.replace(tzinfo=timezone.utc)
                
        for cache in FileCache.query.all():
            if cache.created_at and cache.created_at.tzinfo is None:
                cache.created_at = cache.created_at.replace(tzinfo=timezone.utc)
                
        for doc in Document.query.all():
            if doc.processed_at and doc.processed_at.tzinfo is None:
                doc.processed_at = doc.processed_at.replace(tzinfo=timezone.utc)
        
        db.session.commit()
        print("✅ Database datetime migration completed successfully")
        
    except Exception as e:
        db.session.rollback()
        print(f"❌ Database datetime migration failed: {e}")

# Initialize database
with app.app_context():
    db.create_all()
    
    # Run migration only if needed (check if you have any existing data)
    if User.query.count() > 0:
        migrate_existing_datetimes()

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# GPU-Optimized Document Processor
class OptimizedDocumentProcessor:
    def __init__(self, user_id=None):
        self.user_id = user_id
        self.logger = logging.getLogger(__name__)
        self.app = app  # Store app reference for context
        
        # GPU Configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Using device: {self.device}")
        
        if self.device == 'cuda':
            gpu_props = torch.cuda.get_device_properties(0)
            self.logger.info(f"GPU: {gpu_props.name}, Memory: {gpu_props.total_memory / 1e9:.1f}GB")
            torch.cuda.empty_cache()
        
        # Use GPU-optimized model
        self.embedder = SentenceTransformer(
            app.config['EMBEDDING_MODEL'], 
            device=self.device
        )
        
        # Optimize for RTX A6000
        self.batch_size = app.config['EMBEDDING_BATCH_SIZE'] if self.device == 'cuda' else 32
        self.max_workers = app.config['MAX_PARALLEL_PROCESSES']
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=app.config['CHUNK_SIZE'],
            chunk_overlap=app.config['CHUNK_OVERLAP'],
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize GPU vector store
        self.vector_store = self._create_optimized_index()
        self.logger.info(f"Initialized processor: batch_size={self.batch_size}, workers={self.max_workers}")

    def _create_optimized_index(self):
        """Create GPU-accelerated FAISS index"""
        try:
            dim = self.embedder.get_sentence_embedding_dimension()
            
            # Try to load existing index
            index_file = os.path.join(app.config['VECTOR_STORE_PATH'], 'gpu_index.faiss')
            if os.path.exists(index_file):
                index = faiss.read_index(index_file)
                if self.device == 'cuda':
                    res = faiss.StandardGpuResources()
                    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
                    self.logger.info(f"Loaded existing GPU index with {gpu_index.ntotal} vectors")
                    return gpu_index
                else:
                    self.logger.info(f"Loaded existing CPU index with {index.ntotal} vectors")
                    return index
            
            # Create new index
            if self.device == 'cuda':
                res = faiss.StandardGpuResources()
                index = faiss.IndexFlatIP(dim)  # Inner product for normalized vectors
                gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
                self.logger.info(f"Created new GPU FAISS index (dim={dim})")
                return gpu_index
            else:
                index = faiss.IndexFlatIP(dim)
                self.logger.info(f"Created new CPU FAISS index (dim={dim})")
                return index
                
        except Exception as e:
            self.logger.error(f"Failed to create FAISS index: {e}")
            # Fallback to simple CPU index
            dim = self.embedder.get_sentence_embedding_dimension()
            return faiss.IndexFlatL2(dim)

    def process_documents_parallel(self, file_paths):
        """GPU-accelerated parallel document processing"""
        if not file_paths:
            return []
            
        start_time = time.time()
        self.logger.info(f"Starting parallel processing of {len(file_paths)} files")
        
        # Phase 1: Parallel text extraction and chunking
        extraction_start = time.time()
        extracted_data = []
        
        # Create a wrapper function that includes app context
        def process_with_context(file_path):
            with self.app.app_context():
                return self._extract_and_chunk(file_path)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(process_with_context, file_path): file_path 
                for file_path in file_paths
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    chunks, doc_id = future.result()
                    if chunks:
                        extracted_data.extend([(chunk, file_path, doc_id) for chunk in chunks])
                        self.logger.debug(f"Processed {file_path}: {len(chunks)} chunks")
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path}: {e}")
        
        extraction_time = time.time() - extraction_start
        self.logger.info(f"Extraction completed in {extraction_time:.2f}s ({len(extracted_data)} chunks)")
        
        if not extracted_data:
            self.logger.warning("No data extracted from files")
            return []
        
        # Phase 2: GPU batch embedding
        embedding_start = time.time()
        chunk_texts = [item[0] for item in extracted_data]
        
        self.logger.info(f"Starting GPU embedding for {len(chunk_texts)} chunks")
        embeddings = self._batch_embed(chunk_texts)
        
        if len(embeddings) == 0:
            self.logger.error("No embeddings generated")
            return []
        
        embedding_time = time.time() - embedding_start
        self.logger.info(f"GPU embedding completed in {embedding_time:.2f}s ({len(chunk_texts)/embedding_time:.0f} chunks/sec)")
        
        # Phase 3: Bulk storage with app context
        storage_start = time.time()
        with self.app.app_context():
            self._bulk_store_vectors_and_chunks(extracted_data, embeddings)
        storage_time = time.time() - storage_start
        
        total_time = time.time() - start_time
        self.logger.info(f"Total processing: {total_time:.2f}s ({len(extracted_data)/total_time:.0f} chunks/sec)")
        
        # Save vector store
        self.save_vector_store()
        
        return extracted_data

    def _extract_and_chunk(self, file_path):
        """Extract text and create chunks for a single file"""
        try:
            file_hash = get_file_hash(file_path)
            if not file_hash:
                return [], None
            
            # Check if document already exists
            existing_doc = Document.query.filter_by(file_hash=file_hash, user_id=self.user_id).first()
            if existing_doc:
                chunk_count = DocumentChunk.query.filter_by(document_id=existing_doc.id).count()
                if chunk_count > 0:
                    self.logger.info(f"Document already processed: {file_path}")
                    return [], existing_doc.id
            
            # Create document record
            doc = self._store_document(file_path, file_hash, self.user_id)
            if not doc:
                return [], None
            
            # Extract text
            text = read_file(file_path, self.user_id)
            if not text or not text.strip():
                self.logger.warning(f"No text extracted from {file_path}")
                return [], doc.id
            
            # Create chunks
            chunks = self.text_splitter.split_text(text)
            self.logger.debug(f"Created {len(chunks)} chunks from {file_path}")
            return chunks, doc.id
            
        except Exception as e:
            self.logger.error(f"Error extracting {file_path}: {e}")
            return [], None

    def _batch_embed(self, texts):
        """GPU-accelerated batch embedding"""
        if not texts:
            return np.array([])
        
        all_embeddings = []
        
        try:
            # Process in optimized batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                try:
                    batch_embeddings = self.embedder.encode(
                        batch,
                        batch_size=len(batch),
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        device=self.device
                    )
                    all_embeddings.append(batch_embeddings)
                    
                except Exception as e:
                    self.logger.error(f"Batch embedding failed: {e}")
                    # Fallback to individual processing
                    for text in batch:
                        try:
                            embedding = self.embedder.encode([text], normalize_embeddings=True)
                            all_embeddings.append(embedding)
                        except:
                            dim = self.embedder.get_sentence_embedding_dimension()
                            all_embeddings.append(np.zeros((1, dim)))
            
            return np.vstack(all_embeddings) if all_embeddings else np.array([])
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return np.array([])

    def _store_document(self, file_path, file_hash, user_id):
        """Create document record"""
        try:
            if not os.path.exists(file_path):
                return None
            
            file_stats = os.stat(file_path)
            doc_metadata = {
                'size': file_stats.st_size,
                'modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                'original_filename': os.path.basename(file_path),
                'file_type': os.path.splitext(file_path)[1].lower(),
                'processing_time': datetime.now(timezone.utc).isoformat()
            }
            
            doc = Document(
                user_id=user_id,
                file_path=file_path,
                file_hash=file_hash,
                file_type=doc_metadata['file_type'],
                document_metadata=doc_metadata,
                processed_at=datetime.now(timezone.utc)
            )
            
            db.session.add(doc)
            db.session.flush()
            return doc
            
        except Exception as e:
            self.logger.error(f"Failed to store document {file_path}: {e}")
            db.session.rollback()
            return None

    def _bulk_store_vectors_and_chunks(self, extracted_data, embeddings):
        """Bulk store vectors and database records"""
        if len(extracted_data) == 0 or len(embeddings) == 0:
            return
        
        try:
            # Add to vector store
            start_idx = self.vector_store.ntotal
            self.vector_store.add(embeddings.astype(np.float32))
            
            # Prepare bulk database insert
            chunk_records = []
            for i, (chunk_text, file_path, doc_id) in enumerate(extracted_data):
                chunk_records.append({
                    'document_id': doc_id,
                    'chunk_text': chunk_text,
                    'chunk_hash': hashlib.md5(chunk_text.encode()).hexdigest(),
                    'vector_id': str(start_idx + i),
                    'metadata': {
                        'file_path': file_path,
                        'chunk_index': i,
                        'page': i // 10 + 1
                    }
                })
            
            # Bulk insert
            db.session.bulk_insert_mappings(DocumentChunk, chunk_records)
            db.session.commit()
            self.logger.info(f"Bulk stored {len(chunk_records)} chunks")
            
        except Exception as e:
            db.session.rollback()
            self.logger.error(f"Bulk storage failed: {e}")

    def search_documents(self, query, user_id=None, top_k=None):
        """GPU-accelerated document search"""
        top_k = top_k or app.config['SIMILARITY_TOP_K']
        
        if not self.vector_store or self.vector_store.ntotal == 0:
            self.logger.warning("Vector store is empty")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.encode([query], normalize_embeddings=True)
            
            # Search
            distances, indices = self.vector_store.search(query_embedding, top_k)
            
            # Get results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx == -1:  # FAISS returns -1 for invalid results
                    continue
                    
                chunk = DocumentChunk.query.filter_by(vector_id=str(idx)).first()
                if chunk and chunk.document.user_id == user_id:
                    results.append({
                        'chunk': chunk.chunk_text,
                        'score': float(distance),
                        'document': {
                            'id': chunk.document_id,
                            'path': chunk.document.document_metadata.get('original_filename', '')
                        },
                        'metadata': chunk.chunk_metadata or {}
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def save_vector_store(self):
        """Save vector store to disk"""
        try:
            os.makedirs(app.config['VECTOR_STORE_PATH'], exist_ok=True)
            index_file = os.path.join(app.config['VECTOR_STORE_PATH'], 'gpu_index.faiss')
            
            if self.device == 'cuda':
                cpu_index = faiss.index_gpu_to_cpu(self.vector_store)
                faiss.write_index(cpu_index, index_file)
            else:
                faiss.write_index(self.vector_store, index_file)
                
            self.logger.info(f"Saved vector store with {self.vector_store.ntotal} vectors")
            
        except Exception as e:
            self.logger.error(f"Failed to save vector store: {e}")

# Helper Functions
def get_file_hash(file_path):
    """Generate file hash"""
    try:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except:
        return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def read_file(file_path, user_id=None):
    """Read file with caching"""
    if not os.path.exists(file_path):
        return None
    
    try:
        file_hash = get_file_hash(file_path)
        if not file_hash:
            return None
        
        # Check cache
        cached = FileCache.query.filter_by(file_hash=file_hash).first()
        if cached:
            return cached.content
        
        # Read file based on type
        text = None
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            text = read_pdf(file_path)
        elif ext == '.docx':
            text = read_docx(file_path)
        elif ext == '.txt':
            encodings = ['utf-8', 'iso-8859-1', 'windows-1252']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text = f.read()
                    break
                except UnicodeDecodeError:
                    continue
        elif ext in ('.md', '.markdown'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif ext == '.csv':
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                text = "\n".join([", ".join(row) for row in reader])
        elif ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = json.dumps(json.load(f), indent=2)
        elif ext == '.xml':
            text = read_xml(file_path)
        elif ext == '.html':
            text = read_html(file_path)
        
        if not text or not text.strip():
            return None
        
        # Cache the content
        try:
            new_cache = FileCache(
                file_hash=file_hash,
                content=text,
                file_metadata={
                    'path': file_path,
                    'size': os.path.getsize(file_path),
                    'modified': os.path.getmtime(file_path),
                    'type': ext[1:] if ext else 'unknown'
                },
                user_id=user_id
            )
            db.session.add(new_cache)
            db.session.commit()
        except Exception as e:
            logger.error(f"Failed to cache {file_path}: {e}")
            db.session.rollback()
        
        return text
        
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None

def read_pdf(file_path):
    """Read PDF file"""
    try:
        if app.config['PDF_EXTRACTION_METHOD'] == 'pdfminer':
            return pdfminer_extract(file_path)
        elif app.config['PDF_EXTRACTION_METHOD'] == 'pdfplumber':
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text(layout=True)
                    if page_text:
                        text += f"=== PAGE {page.page_number} ===\n{page_text}\n\n"
            return text
        else:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"=== PAGE {i+1} ===\n{page_text}\n\n"
                return text
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return None

def read_docx(file_path):
    """Read DOCX file"""
    try:
        doc = docx.Document(file_path)
        text = []
        for para in doc.paragraphs:
            if para.style.name.startswith('Heading'):
                level = int(para.style.name.split(' ')[1]) if ' ' in para.style.name else 1
                text.append(f"\n{'#' * level} {para.text}\n")
            else:
                text.append(para.text)
        return "\n".join(text)
    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}")
        return None

def read_xml(file_path):
    """Read XML file"""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        def xml_to_text(element, indent=0):
            text = []
            attrs = " ".join(f'{k}="{v}"' for k, v in element.attrib.items())
            tag_text = f"{' ' * indent}<{element.tag}"
            if attrs:
                tag_text += f" {attrs}"
            tag_text += ">"
            text.append(tag_text)
            
            if element.text and element.text.strip():
                text.append(f"{' ' * (indent+2)}{element.text.strip()}")
            
            for child in element:
                text.append(xml_to_text(child, indent+2))
            
            text.append(f"{' ' * indent}</{element.tag}>")
            return "\n".join(text)
        
        return xml_to_text(root)
    except Exception as e:
        logger.error(f"XML extraction failed: {e}")
        return None

def read_html(file_path):
    """Read HTML file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            return soup.get_text('\n')
    except Exception as e:
        logger.error(f"HTML extraction failed: {e}")
        return None

def clean_directory(directory):
    """Clean directory"""
    if not os.path.exists(directory):
        return
        
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.error(f'Failed to delete {file_path}: {e}')

def extract_zip_parallel(zip_path, extract_to):
    """Extract ZIP file in parallel"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_infos = [
                f for f in zip_ref.infolist() 
                if not f.is_dir() 
                and '__MACOSX' not in f.filename
                and allowed_file(f.filename)
            ]
            
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                for file_info in file_infos:
                    futures.append(executor.submit(
                        extract_single_file, zip_ref, file_info, extract_to
                    ))
                
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"File extraction failed: {e}")
                        
        return True
    except Exception as e:
        logger.error(f"ZIP extraction failed: {e}")
        return False

def extract_single_file(zip_ref, file_info, extract_to):
    """Extract single file from ZIP"""
    file_path = file_info.filename
    target_path = os.path.join(extract_to, file_path)
    
    if file_info.file_size > app.config['MAX_FILE_SIZE']:
        logger.warning(f"Skipping large file: {file_path}")
        return
        
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    with zip_ref.open(file_info) as source, open(target_path, 'wb') as target:
        shutil.copyfileobj(source, target)

def build_rag_prompt(context, question):
    """Build RAG prompt"""
    return (
        "Use the following documents to answer the question. "
        "Provide detailed answers with direct quotes when possible. "
        "Always cite sources using the format [^filename||page].\n\n"
        "Documents:\n"
        f"{context}\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "1. Answer precisely based on the documents\n"
        "2. Include relevant quotes with citations\n"
        "3. If unsure, say you couldn't find the information\n"
        "4. Format citations as [^filename||page]\n\n"
        "Answer:"
    )

def get_ollama_response_stream(prompt):
    """Stream response from Ollama"""
    try:
        response = requests.post(
            app.config['OLLAMA_URL'],
            json={
                "model": app.config['OLLAMA_MODEL'],
                "prompt": prompt,
                "stream": True
            },
            stream=True,
            timeout=app.config['OLLAMA_TIMEOUT']
        )
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if 'response' in data:
                    yield data['response']
                    
    except Exception as e:
        yield f"\n\nError: Failed to get response from AI model ({str(e)})"

# Routes
@app.route('/upload', methods=['POST'])
@login_required
def upload_files():
    """GPU-optimized file upload"""
    logger.info(f"Upload request from user {current_user.id}")
    
    if not current_user.is_subscribed and not current_user.is_admin:
        return jsonify({'error': 'Subscription required'}), 403
        
    try:
        # Create directories
        user_id = str(current_user.id)
        user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
        user_extract_dir = os.path.join(app.config['EXTRACT_FOLDER'], user_id)
        
        os.makedirs(user_upload_dir, exist_ok=True)
        os.makedirs(user_extract_dir, exist_ok=True)
        
        # Clean directories
        clean_directory(user_upload_dir)
        clean_directory(user_extract_dir)
        
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
            
        files = request.files.getlist('files')
        saved_files = []
        extracted_files = []
        
        logger.info(f"Processing {len(files)} uploaded files")
        
        for file in files:
            if file.filename == '':
                continue
                
            filename = secure_filename(file.filename)
            is_zip = filename.lower().endswith('.zip')
            
            try:
                if is_zip:
                    # Handle ZIP
                    zip_path = os.path.join(user_upload_dir, filename)
                    file.save(zip_path)
                    
                    if not zipfile.is_zipfile(zip_path):
                        logger.error(f"Invalid ZIP: {zip_path}")
                        os.remove(zip_path)
                        continue
                        
                    # Extract in parallel
                    if extract_zip_parallel(zip_path, user_extract_dir):
                        # Find extracted files
                        for root, _, files_in_zip in os.walk(user_extract_dir):
                            for f in files_in_zip:
                                if allowed_file(f) and not f.startswith('.'):
                                    extracted_files.append(os.path.join(root, f))
                        saved_files.append(filename)
                        
                else:
                    # Handle regular files
                    if not allowed_file(filename):
                        continue
                        
                    file_path = os.path.join(user_extract_dir, filename)
                    file.save(file_path)
                    extracted_files.append(file_path)
                    saved_files.append(filename)
                    
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                continue
                
        if not saved_files:
            return jsonify({'error': 'No valid files were uploaded'}), 400

        # Store in session
        session['current_upload_files'] = [
            os.path.relpath(f, app.config['EXTRACT_FOLDER']) 
            for f in extracted_files
        ]
        session.modified = True
        
        # GPU-accelerated processing
        logger.info(f"Starting GPU processing for {len(extracted_files)} files")
        processor = OptimizedDocumentProcessor(user_id=current_user.id)
        results = processor.process_documents_parallel(extracted_files)
        
        logger.info(f"Processing complete: {len(results)} chunks processed")
        
        return jsonify({
            'message': f'{len(saved_files)} files processed successfully',
            'files': saved_files,
            'chunks': len(results),
            'redirect': url_for('new_chat')
        })
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return jsonify({'error': 'Server error during upload'}), 500

@app.route('/new_chat', methods=['GET', 'POST'])
@login_required
def new_chat():
    """Create new chat"""
    try:
        if 'current_upload_files' in session:
            del session['current_upload_files']
            
        chat_id = str(uuid.uuid4())
        
        conversation = Conversation(
            id=chat_id,
            user_id=current_user.id,
            title='New Query'
        )
        db.session.add(conversation)
        
        # Get uploaded files
        uploaded_files = []
        user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
        if os.path.exists(user_upload_dir):
            for root, _, files in os.walk(user_upload_dir):
                for file in files:
                    if not file.startswith('.'):
                        uploaded_files.append(file)
        
        initial_message = "Files uploaded successfully. Type your search query below."
        if uploaded_files:
            file_list = "\n".join(f"- {file}" for file in uploaded_files[:5])
            if len(uploaded_files) > 5:
                file_list += f"\n- ...and {len(uploaded_files) - 5} more files"
            initial_message = f"{file_list}\n\nUploaded successfully. Type your search query below."
        
        initial_msg = Message(
            conversation_id=chat_id,
            role='assistant',
            content=initial_message,
            timestamp=datetime.now(timezone.utc)
        )
        db.session.add(initial_msg)
        db.session.commit()
        
        return redirect(url_for('chat', chat_id=chat_id))
        
    except Exception as e:
        logger.error(f'Error creating chat: {e}')
        flash('Error creating new chat', 'error')
        return redirect(url_for('index'))

@app.route('/chat/<chat_id>', methods=['GET', 'POST'])
@login_required
def chat(chat_id):
    """GPU-optimized chat interface"""
    try:
        conversation = Conversation.query.filter_by(
            id=chat_id,
            user_id=current_user.id
        ).first_or_404()

        if request.method == 'GET':
            messages = Message.query.filter_by(
                conversation_id=chat_id
            ).order_by(Message.timestamp.asc()).all()

            user_conversations = Conversation.query.filter_by(
                user_id=current_user.id
            ).order_by(Conversation.updated_at.desc()).all()

            uploaded_files = get_uploaded_files(current_user.id)

            return render_template(
                'chat.html',
                chat_id=chat_id,
                conversation=conversation,
                messages=messages,
                conversations=user_conversations,
                chat_title=conversation.title,
                uploaded_files=uploaded_files
            )

        elif request.method == 'POST':
            def generate_response():
                try:
                    if not request.is_json:
                        yield json.dumps({'status': 'error', 'message': 'Invalid request'})
                        return

                    data = request.get_json()
                    question = data.get('question', '').strip()
                    
                    if not question:
                        yield json.dumps({'status': 'error', 'message': 'Please enter a question'})
                        return

                    # Save user message
                    user_msg = Message(
                        conversation_id=chat_id,
                        role='user',
                        content=question,
                        timestamp=datetime.now(timezone.utc)
                    )
                    db.session.add(user_msg)
                    db.session.commit()

                    # Update conversation title
                    if conversation.title == 'New Query':
                        short_title = question[:50] + '...' if len(question) > 50 else question
                        conversation.title = short_title
                        db.session.commit()
                        yield json.dumps({
                            'status': 'title_update',
                            'chat_id': chat_id,
                            'title': short_title
                        }) + "\n"

                    # GPU-accelerated search
                    processor = OptimizedDocumentProcessor(user_id=current_user.id)
                    chunks = processor.search_documents(question, current_user.id)
                    
                    if not chunks:
                        yield json.dumps({
                            'status': 'error',
                            'message': 'No relevant information found'
                        })
                        return

                    # Build context
                    context = ""
                    sources = []
                    for chunk in chunks:
                        context += f"[Document: {chunk['document']['path']}, Score: {chunk['score']:.3f}]\n"
                        context += f"{chunk['chunk']}\n\n"
                        sources.append({
                            'path': chunk['document']['path'],
                            'score': chunk['score'],
                            'text': chunk['chunk'][:200] + '...'
                        })

                    # Stream response
                    yield json.dumps({'status': 'stream_start'}) + "\n"

                    prompt = build_rag_prompt(context, question)
                    full_answer = ""
                    
                    for chunk in get_ollama_response_stream(prompt):
                        yield json.dumps({'status': 'stream_chunk', 'content': chunk}) + "\n"
                        full_answer += chunk

                    # Save response
                    assistant_msg = Message(
                        conversation_id=chat_id,
                        role='assistant',
                        content=full_answer,
                        sources=[s['path'] for s in sources],
                        timestamp=datetime.now(timezone.utc)
                    )
                    db.session.add(assistant_msg)
                    
                    conversation.updated_at = datetime.now(timezone.utc)
                    db.session.commit()

                    yield json.dumps({
                        'status': 'stream_end',
                        'sources': sources
                    }) + "\n"

                except Exception as e:
                    logger.error(f"Chat error: {e}")
                    yield json.dumps({
                        'status': 'error',
                        'message': f"Error: {str(e)}"
                    }) + "\n"

            return Response(stream_with_context(generate_response()), mimetype='application/x-ndjson')

    except Exception as e:
        logger.error(f"Chat route error: {e}")
        if request.method == 'GET':
            flash('Error loading chat', 'error')
            return redirect(url_for('index'))
        else:
            return jsonify({'error': str(e)}), 500

def get_uploaded_files(user_id):
    """Get uploaded files for user"""
    try:
        user_extract_dir = os.path.join(app.config['EXTRACT_FOLDER'], str(user_id))
        uploaded_files = []
        
        if os.path.exists(user_extract_dir):
            for root, _, files in os.walk(user_extract_dir):
                for file in files:
                    if not file.startswith('.'):
                        rel_path = os.path.relpath(os.path.join(root, file), user_extract_dir)
                        uploaded_files.append({
                            'name': file,
                            'path': rel_path,
                            'size': os.path.getsize(os.path.join(root, file))
                        })
        
        return uploaded_files
    except Exception as e:
        logger.error(f"Error getting files: {e}")
        return []

@app.route('/delete_chat/<chat_id>', methods=['POST'])
@login_required
def delete_chat(chat_id):
    """Delete chat"""
    try:
        conversation = Conversation.query.filter_by(id=chat_id, user_id=current_user.id).first()
        if not conversation:
            return jsonify({'success': False, 'error': 'Chat not found'}), 404
            
        db.session.delete(conversation)
        db.session.commit()
        return jsonify({'success': True}), 200
        
    except Exception as e:
        logger.error(f'Error deleting chat: {e}')
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/')
def index():
    """Main index page"""
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not username or not email or not password:
            flash('All fields are required', 'error')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register'))
        
        if User.query.filter_by(username=username).first():
            flash('Username already taken', 'error')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return redirect(url_for('register'))
        
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember') == 'on'

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user, remember=remember)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('Invalid username or password', 'error')

    return render_template('login.html')

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    """User logout"""
    logout_user()
    session.clear()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/get_file_content')
def get_file_content():
    """Get file content"""
    rel_path = request.args.get('path')
    if not rel_path:
        return jsonify({'error': 'File path required'}), 400
    
    try:
        if '..' in rel_path or rel_path.startswith('/'):
            return jsonify({'error': 'Invalid file path'}), 400
        
        rel_path = rel_path.replace('\\', '/').strip('/')
        full_path = os.path.join(app.config['EXTRACT_FOLDER'], *rel_path.split('/'))
        full_path = os.path.normpath(full_path)
        
        if not full_path.startswith(os.path.abspath(app.config['EXTRACT_FOLDER'])):
            return jsonify({'error': 'Access denied'}), 403
        
        if not os.path.exists(full_path):
            return jsonify({'error': 'File not found'}), 404
        
        file_ext = os.path.splitext(full_path)[1].lower()
        
        if file_ext == '.pdf' or file_ext.startswith('.image'):
            return send_file(full_path, as_attachment=False)
        else:
            content = read_file(full_path)
            if not content:
                return jsonify({'error': 'Could not read file'}), 400
                
            return jsonify({
                'file': os.path.basename(full_path),
                'content': content,
                'path': rel_path
            })
    
    except Exception as e:
        logger.error(f"Error serving file: {e}")
        return jsonify({'error': str(e)}), 500

# Subscription routes
@app.route('/start_trial', methods=['POST'])
@login_required
def start_trial():
    """Start trial subscription"""
    now = datetime.now(timezone.utc)
    
    # Check if user already has an active trial
    if (current_user.subscription_plan == 'trial' and 
        current_user.trial_end_date and 
        current_user._make_aware(current_user.trial_end_date) > now):
        return jsonify({'error': 'You already have an active trial'}), 400
        
    current_user.subscription_plan = 'trial'
    current_user.subscription_status = 'active'
    current_user.trial_start_date = now
    current_user.trial_end_date = now + timedelta(days=14)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'Your 14-day free trial has started!',
        'trial_end_date': current_user.trial_end_date.isoformat()
    })

@app.route('/api/check_subscription', methods=['GET'])
@login_required
def check_subscription_api():
    """Check subscription status"""
    try:
        now = datetime.now(timezone.utc)
        trial_days_remaining = 0
        
        if current_user.trial_end_date:
            trial_end = current_user._make_aware(current_user.trial_end_date)
            trial_days_remaining = max(0, (trial_end - now).days)
        
        has_active_trial = (current_user.subscription_plan == 'trial' and 
                           current_user.trial_end_date and 
                           current_user._make_aware(current_user.trial_end_date) > now)
        
        return jsonify({
            'is_logged_in': True,
            'has_subscription': current_user.is_subscribed,
            'has_active_trial': has_active_trial,
            'has_trial_available': (current_user.subscription_plan != 'trial' or 
                                   not current_user.is_subscribed),
            'trial_days_remaining': trial_days_remaining,
            'allow_uploads': current_user.is_subscribed,
            'plan_name': current_user.plan_details.get('name', 'Free')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/pricing')
@login_required
def pricing():
    """Pricing page"""
    plans = {
        'basic': SubscriptionPlan.BASIC,
        'pro': SubscriptionPlan.PRO,
        'enterprise': SubscriptionPlan.TRIAL
    }
    return render_template('pricing.html', plans=plans, now=datetime.now(timezone.utc))

@app.route('/billing')
@login_required
def billing():
    """Billing page"""
    payments = Payment.query.filter_by(user_id=current_user.id).order_by(Payment.created_at.desc()).limit(10).all()
    invoices = BillingHistory.query.filter_by(user_id=current_user.id).order_by(BillingHistory.created_at.desc()).limit(10).all()
    
    return render_template('billing.html', 
                         payments=payments,
                         invoices=invoices,
                         plan=current_user.plan_details,
                         now=datetime.now(timezone.utc))

@app.route('/create_subscription', methods=['POST'])
@login_required
def create_subscription():
    """Create subscription with Fatora"""
    plan_id = request.form.get('plan_id')
    payment_method_id = request.form.get('payment_method_id')
    
    if not plan_id or not payment_method_id:
        return jsonify({'error': 'Missing plan or payment method'}), 400
        
    plan = SubscriptionPlan.get_plan(plan_id)
    if not plan:
        return jsonify({'error': 'Invalid plan'}), 400
    
    try:
        headers = {
            'Authorization': f'Bearer {app.config["FATORA_API_KEY"]}',
            'Content-Type': 'application/json'
        }
        
        # Create customer if not exists
        if not current_user.subscription_id:
            customer_data = {
                'name': current_user.username,
                'email': current_user.email,
                'phone': '',
                'metadata': {
                    'user_id': current_user.id
                }
            }
            
            response = requests.post(
                f'{app.config["FATORA_BASE_URL"]}/customers',
                headers=headers,
                json=customer_data
            )
            
            if response.status_code != 201:
                return jsonify({'error': 'Failed to create customer'}), 400
                
            customer_id = response.json().get('id')
            current_user.subscription_id = customer_id
            db.session.commit()
        
        # Create subscription
        subscription_data = {
            'customer_id': current_user.subscription_id,
            'plan_id': plan_id,
            'payment_method_id': payment_method_id,
            'metadata': {
                'user_id': current_user.id
            }
        }
        
        response = requests.post(
            f'{app.config["FATORA_BASE_URL"]}/subscriptions',
            headers=headers,
            json=subscription_data
        )
        
        if response.status_code != 201:
            return jsonify({'error': 'Failed to create subscription'}), 400
            
        subscription = response.json()
        
        # Update user
        current_user.subscription_plan = plan_id
        current_user.subscription_status = 'active'
        current_user.payment_method_id = payment_method_id
        current_user.current_period_end = datetime.now(timezone.utc) + timedelta(days=30)
        current_user.cancel_at_period_end = False
        db.session.commit()
        
        # Create payment record
        payment = Payment(
            user_id=current_user.id,
            amount=plan['price'],
            currency='QAR',
            payment_intent_id=subscription.get('payment_intent_id'),
            payment_method=payment_method_id,
            status='succeeded',
            receipt_url=subscription.get('receipt_url')
        )
        db.session.add(payment)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'subscription_id': subscription.get('id'),
            'client_secret': subscription.get('client_secret')
        })
        
    except Exception as e:
        logger.error(f'Subscription error: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/cancel_subscription', methods=['POST'])
@login_required
def cancel_subscription():
    """Cancel subscription"""
    try:
        if not current_user.subscription_id:
            return jsonify({'error': 'No active subscription'}), 400
            
        headers = {
            'Authorization': f'Bearer {app.config["FATORA_API_KEY"]}',
            'Content-Type': 'application/json'
        }
        
        response = requests.delete(
            f'{app.config["FATORA_BASE_URL"]}/subscriptions/{current_user.subscription_id}',
            headers=headers
        )
        
        if response.status_code != 200:
            return jsonify({'error': 'Failed to cancel subscription'}), 400
            
        current_user.subscription_status = 'canceled'
        current_user.cancel_at_period_end = True
        db.session.commit()
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f'Cancel subscription error: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/update_payment_method', methods=['POST'])
@login_required
def update_payment_method():
    """Update payment method"""
    payment_method_id = request.form.get('payment_method_id')
    
    if not payment_method_id:
        return jsonify({'error': 'Payment method required'}), 400
        
    try:
        headers = {
            'Authorization': f'Bearer {app.config["FATORA_API_KEY"]}',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            f'{app.config["FATORA_BASE_URL"]}/customers/{current_user.subscription_id}/payment_methods',
            headers=headers,
            json={'payment_method_id': payment_method_id}
        )
        
        if response.status_code != 200:
            return jsonify({'error': 'Failed to update payment method'}), 400
            
        current_user.payment_method_id = payment_method_id
        db.session.commit()
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f'Update payment method error: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/fatora_webhook', methods=['POST'])
def fatora_webhook():
    """Fatora webhook handler"""
    import hmac
    
    payload = request.data
    sig_header = request.headers.get('X-Fatora-Signature')
    
    try:
        # Verify webhook signature
        secret = app.config['FATORA_WEBHOOK_SECRET']
        computed_signature = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        
        if not hmac.compare_digest(computed_signature, sig_header):
            return jsonify({'error': 'Invalid signature'}), 400
            
        event = json.loads(payload)
        
        # Handle different event types
        if event['type'] == 'payment.succeeded':
            handle_payment_succeeded(event['data'])
        elif event['type'] == 'invoice.paid':
            handle_invoice_paid(event['data'])
        elif event['type'] == 'invoice.payment_failed':
            handle_payment_failed(event['data'])
        elif event['type'] == 'customer.subscription.deleted':
            handle_subscription_deleted(event['data'])
            
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f'Webhook error: {e}')
        return jsonify({'error': str(e)}), 500

def handle_payment_succeeded(data):
    """Handle successful payment"""
    payment_intent = data['object']
    
    payment = Payment.query.filter_by(payment_intent_id=payment_intent['id']).first()
    if not payment:
        user = User.query.filter_by(subscription_id=payment_intent['customer']).first()
        if user:
            payment = Payment(
                user_id=user.id,
                amount=payment_intent['amount'] / 100,
                currency=payment_intent['currency'],
                payment_intent_id=payment_intent['id'],
                payment_method=payment_intent['payment_method'],
                status='succeeded',
                receipt_url=payment_intent.get('receipt_url')
            )
            db.session.add(payment)
            db.session.commit()

def handle_invoice_paid(data):
    """Handle paid invoice"""
    invoice = data['object']
    user = User.query.filter_by(subscription_id=invoice['customer']).first()
    
    if user:
        # Update subscription period
        user.current_period_end = datetime.fromtimestamp(invoice['period_end'])
        user.last_billing_date = datetime.now(timezone.utc)
        user.documents_uploaded = 0  # Reset monthly count
        db.session.commit()
        
        # Record invoice
        billing = BillingHistory(
            user_id=user.id,
            description=f"{user.plan_details.get('name', 'Unknown')} Plan",
            amount=invoice['amount_paid'] / 100,
            currency=invoice['currency'],
            status='paid',
            invoice_id=invoice['id']
        )
        db.session.add(billing)
        db.session.commit()

def handle_payment_failed(data):
    """Handle failed payment"""
    invoice = data['object']
    user = User.query.filter_by(subscription_id=invoice['customer']).first()
    
    if user:
        billing = BillingHistory(
            user_id=user.id,
            description=f"Failed payment for {user.plan_details.get('name', 'Unknown')} Plan",
            amount=invoice['amount_due'] / 100,
            currency=invoice['currency'],
            status='failed',
            invoice_id=invoice['id']
        )
        db.session.add(billing)
        db.session.commit()

def handle_subscription_deleted(data):
    """Handle subscription deletion"""
    subscription = data['object']
    user = User.query.filter_by(subscription_id=subscription['customer']).first()
    
    if user:
        user.subscription_status = 'canceled'
        user.current_period_end = None
        db.session.commit()

@app.before_request
def check_subscription():
    """Check subscription status before each request"""
    if current_user.is_authenticated and not current_user.is_admin:
        # Reset document count if new billing month
        if (current_user.last_billing_date and 
            current_user._make_aware(current_user.last_billing_date).month < datetime.now(timezone.utc).month):
            current_user.documents_uploaded = 0
            db.session.commit()

@app.before_request
def check_trial():
    """Check trial status"""
    if current_user.is_authenticated and not current_user.is_admin:
        # Check if trial is ending soon (3 days or less)
        if (current_user.subscription_plan == 'trial' and 
            current_user.trial_end_date and 
            not request.path.startswith('/static') and
            not request.path.startswith('/pricing')):
            
            now = datetime.now(timezone.utc)
            trial_end = current_user._make_aware(current_user.trial_end_date)
            days_left = (trial_end - now).days
            
            if days_left <= 3:
                flash(f'Your free trial ends in {days_left} day(s). Upgrade now to continue uninterrupted service.', 'warning')

# Template filters
@app.template_filter('datetimeformat')
def datetimeformat(value, format='%b %d, %H:%M'):
    """Format datetime for templates"""
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.strftime(format)

@app.template_filter('days_from_now')
def days_from_now(value):
    """Calculate days from now with timezone-aware handling"""
    if not value:
        return 0
    
    now = datetime.now(timezone.utc)
    
    # Handle both naive and timezone-aware datetimes
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    
    days = (value - now).days
    return max(0, days)

@app.template_filter('process_references')
def process_references(text):
    """Process references in text"""
    reference_counter = 1
    references = []
    
    def replace_reference(match):
        nonlocal reference_counter
        filename = match.group(1)
        page = match.group(2) or ''
        highlight = match.group(3) or ''
        
        filename = filename.split(']')[0]
        filename = filename.replace('\\', '/')
        
        ref_id = reference_counter
        reference_counter += 1
        
        references.append({
            'id': ref_id,
            'filename': filename,
            'page': page,
            'highlight': highlight
        })
        
        return (f'<sup class="reference-link" '
                f'data-ref="{ref_id}" '
                f'data-filename="{filename}" '
                f'data-page="{page}" '
                f'data-highlight="{highlight}">'
                f'[{ref_id}]</sup>')
    
    processed_text = re.sub(
        r'\[\^([^\|\]]+)(?:\|\|([^\|\]]+))?(?:\|\|([^\|\]]+))?\]',
        replace_reference,
        text
    )
    
    if references:
        references_section = '<div class="references-section"><h4>References</h4><ol>'
        for ref in sorted(references, key=lambda x: x['id']):
            ref_text = ref['filename'].split('/')[-1]
            if ref['page']:
                ref_text += f", {ref['page']}"
            if ref['highlight']:
                ref_text += f" ({ref['highlight']})"
            references_section += f'<li id="ref-{ref["id"]}">{ref_text}</li>'
        references_section += '</ol></div>'
        
        processed_text += references_section
    
    return processed_text

@app.context_processor
def inject_now():
    """Inject current time into templates"""
    return {'now': datetime.now(timezone.utc)}

@app.after_request
def after_request(response):
    """Add headers to prevent timeouts"""
    response.headers["X-Cloudflare-Time"] = "900"
    response.headers["Connection"] = "keep-alive"
    return response

if __name__ == '__main__':
    try:
        logger.info('Starting GPU-optimized DocumentIQ application')
        app.run(
            host='0.0.0.0',
            port=8000,
            threaded=True,
            debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
        )
    except Exception as e:
        logger.critical(f"Failed to start application: {e}")
        raise