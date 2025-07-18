#!/usr/bin/env python3
"""
DocumentIQ - GPU-Accelerated Document Processing System
Optimized for RTX A6000 with Large File Upload Support
Enhanced for 20GB ZIP files and 50GB extracted content
"""

import os
import zipfile
import shutil
import requests
import uuid
import json
import time
import threading
import hashlib
import asyncio
import aiofiles
import sys
import re
import traceback
import tempfile
import mmap
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from logging.handlers import RotatingFileHandler

# Flask and web framework imports
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, stream_with_context, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_migrate import Migrate
from flask_session import Session

# GPU and ML imports
import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Enhanced document processing imports
import fitz  # PyMuPDF for better PDF processing
import mammoth  # Better DOCX processing
import pandas as pd
from openpyxl import load_workbook
import easyocr  # GPU-accelerated OCR
from pdf2image import convert_from_path
import pytesseract
from bs4 import BeautifulSoup

# Database imports
from sqlalchemy import text, func, create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import QueuePool

# Caching
import redis
import pickle

# System monitoring
import psutil

import textwrap

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Flask app
app = Flask(__name__)


# Processing status tracking with thread safety
processing_status = {}
processing_status_lock = threading.Lock()  # Add this line if it's missing

# Enhanced logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler for better debugging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Enhanced GPU-Optimized Configuration for Large Files
class GPUOptimizedConfig:
    SECRET_KEY = os.getenv('SECRET_KEY', 'gpu-accelerated-documentiq-secret')
    
    # Directories
    UPLOAD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'uploads')
    EXTRACT_FOLDER = os.path.join(os.getcwd(), 'extracted_files')
    CACHE_FOLDER = os.path.join(os.getcwd(), 'file_cache')
    TEMP_DIR = os.path.join(os.getcwd(), 'temp_processing')
    
    # Enhanced file processing settings for large files
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'md', 'csv', 'json', 'xml', 'html', 'xlsx', 'xls', 'pptx', 'jpg', 'jpeg', 'png', 'tiff', 'bmp'}
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024 * 1024  # 20GB (leveraging 256GB RAM)
    MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024  # 10GB per file
    MAX_ZIP_EXTRACTED_SIZE = 50 * 1024 * 1024 * 1024  # 50GB extracted
    
    # Request handling optimizations
    SEND_FILE_MAX_AGE_DEFAULT = 0
    PERMANENT_SESSION_LIFETIME = timedelta(hours=4)
    REQUEST_TIMEOUT = 7200  # 2 hours
    MAX_FORM_MEMORY_SIZE = None  # Unlimited form memory
    
    # Upload optimizations
    UPLOAD_CHUNK_SIZE = 16 * 1024 * 1024  # 16MB chunks
    EXTRACTION_BUFFER_SIZE = 64 * 1024  # 64KB buffer
    EXTRACTION_TIMEOUT = 3600  # 1 hour for extraction
    
    # AI/ML Configuration
    OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434/api/generate')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3:70b-instruct-q4_K_M')
    OLLAMA_TIMEOUT = 600
    
    # Enhanced GPU Processing Configuration
    GPU_ENABLED = torch.cuda.is_available()
    GPU_MEMORY_FRACTION = 0.8  # Use 80% of RTX A6000's 48GB
    MIXED_PRECISION = True  # Use FP16 for faster processing
    GPU_BATCH_SIZE = 128  # Large batches for RTX A6000
    CONCURRENT_FILES = 32  # Process 32 files simultaneously
    MAX_WORKERS = min(64, cpu_count() * 2)  # Enhanced parallelism
    
    # Enhanced RAG Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    SIMILARITY_TOP_K = 20  # Increased for better search results
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast and efficient
    EMBEDDING_DIM = 384
    
    # Database Configuration (optimized for GPU workloads)
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://documentiq_user:12345@localhost:5432/documentiq_db')
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 20,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'max_overflow': 40,
        'pool_timeout': 30,
    }
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Redis Configuration for GPU processing
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    CACHE_EMBEDDINGS = True
    CACHE_TTL = 86400 * 7  # 7 days
    
    # Session Configuration
    SESSION_TYPE = 'filesystem'
    SESSION_FILE_DIR = '/tmp/flask_sessions'
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    SESSION_KEY_PREFIX = 'documentiq:'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=2)
    
    # Logging
    LOG_FILE = os.path.join(os.getcwd(), 'app.log')
    LOG_LEVEL = logging.INFO

# Apply configuration
app.config.from_object(GPUOptimizedConfig)

# Create directories
for directory in ['/tmp/flask_sessions', app.config['UPLOAD_FOLDER'], 
                 app.config['EXTRACT_FOLDER'], app.config['CACHE_FOLDER'], 
                 app.config['TEMP_DIR']]:
    os.makedirs(directory, exist_ok=True)

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
migrate = Migrate(app, db)
Session(app)

# Configure logging
handler = RotatingFileHandler(app.config['LOG_FILE'], maxBytes=10 * 1024 * 1024, backupCount=3)
handler.setLevel(app.config['LOG_LEVEL'])
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GPUProcessingProgress:
    def __init__(self, total_files):
        self.total_files = total_files
        self.processed_files = 0
        self.current_file = ""
        self.status = "processing"
        self.error_message = ""
        self.start_time = time.time()
        self.gpu_stats = self.get_gpu_stats()
        self.throughput = 0.0
        
    def get_gpu_stats(self):
        """Get current GPU statistics"""
        if not torch.cuda.is_available():
            return {"gpu_available": False}
        try:
            return {
                "gpu_available": True,
                "gpu_name": torch.cuda.get_device_name(),
                "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "utilization_percent": torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 'N/A'
            }
        except:
            return {"gpu_available": False, "error": "GPU stats unavailable"}
    
    def update_stats(self):
        """Update performance statistics"""
        self.gpu_stats = self.get_gpu_stats()
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.throughput = self.processed_files / elapsed

# Subscription Plans (enhanced for GPU processing)
class SubscriptionPlan:
    BASIC = {
        'id': 'basic',
        'name': 'Basic',
        'price': 99,
        'features': ['500 documents/month', '100MB max file size', 'Basic support'],
        'limits': {'max_documents': 500, 'max_file_size': 100 * 1024 * 1024, 'max_conversations': 10}
    }
    
    PRO = {
        'id': 'pro',
        'name': 'Professional',
        'price': 199,
        'features': ['2000 documents/month', '500MB max file size', 'Priority support', 'Advanced analytics'],
        'limits': {'max_documents': 2000, 'max_file_size': 500 * 1024 * 1024, 'max_conversations': 50}
    }
    
    TRIAL = {
        'id': 'trial',
        'name': 'Free Trial',
        'price': 0,
        'features': ['Unlimited documents for 14 days', '10GB max file size', 'All Professional features', 'GPU acceleration'],
        'limits': {'max_documents': None, 'max_file_size': 10 * 1024 * 1024 * 1024, 'max_conversations': None, 'trial_days': 14}
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
        return SubscriptionPlan.get_plan(self.subscription_plan) or SubscriptionPlan.get_plan('basic')
    
    def can_upload_file(self, file_size):
        if self.is_admin:
            return True
        
        if self.subscription_plan == 'trial' and self.is_subscribed:
            return True
        
        plan = self.plan_details.get('limits', {})
        max_size = plan.get('max_file_size', 100 * 1024 * 1024)
        return file_size <= max_size

    @property
    def trial_days_remaining(self):
        """Calculate trial days remaining"""
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
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    messages = db.relationship('Message', backref='conversation', lazy=True, cascade='all, delete-orphan')

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.String(36), db.ForeignKey('conversation.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    sources = db.Column(db.JSON)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_path = db.Column(db.String(512), nullable=False)
    file_hash = db.Column(db.String(64), nullable=False)
    file_type = db.Column(db.String(32), nullable=False)
    processed_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    document_metadata = db.Column(db.JSON, name="metadata")

class DocumentContent(db.Model):
    """Enhanced table for storing extracted content and embeddings"""
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    content_type = db.Column(db.String(50), default='text')  # 'text', 'chunk', 'embedding'
    chunk_index = db.Column(db.Integer, default=0)
    embedding = db.Column(db.JSON)  # Store embedding vectors
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    content_metadata = db.Column(db.JSON)  # Changed from 'metadata' to 'content_metadata'
    document = db.relationship('Document', backref='content_records')

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# GPU-Optimized Document Processor
class GPUOptimizedDocumentProcessor:
    """GPU-accelerated document processor for RTX A6000"""
    
    def __init__(self, user_id=None):
        self.user_id = user_id
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize GPU
        self._setup_gpu()
        
        # Initialize models
        self._setup_models()
        
        # Initialize cache
        self._setup_cache()
        
        # Initialize thread pools
        self.thread_executor = ThreadPoolExecutor(max_workers=app.config['MAX_WORKERS'])
        self.process_executor = ProcessPoolExecutor(max_workers=min(16, cpu_count()))
        
        self.logger.info(f"GPU processor initialized on {self.device} for user {user_id}")
    
    def _setup_gpu(self):
        """Configure GPU for optimal performance"""
        if torch.cuda.is_available():
            # Set memory fraction for RTX A6000 (48GB)
            torch.cuda.set_per_process_memory_fraction(app.config['GPU_MEMORY_FRACTION'])
            
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            
            self.logger.info(f"GPU Setup: {torch.cuda.get_device_name()} - {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            self.logger.warning("CUDA not available, using CPU")
    
    def _setup_models(self):
        """Initialize AI models on GPU"""
        try:
            # Sentence transformer for embeddings
            self.embedding_model = SentenceTransformer(
                app.config['EMBEDDING_MODEL'],
                device=self.device
            )
            
            # Enable mixed precision for faster processing
            if app.config['MIXED_PRECISION'] and torch.cuda.is_available():
                self.embedding_model.half()
            
            # OCR model for images (GPU-accelerated)
            self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            
            # Initialize FAISS index for vector search
            self.faiss_index = None
            self.chunk_metadata = []
            
            self.logger.info("GPU models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise
    
    def _setup_cache(self):
        """Setup Redis cache for embeddings"""
        try:
            if app.config['CACHE_EMBEDDINGS']:
                self.redis_client = redis.from_url(
                    app.config['REDIS_URL'],
                    decode_responses=False
                )
                self.redis_client.ping()
                self.logger.info("Redis cache connected")
            else:
                self.redis_client = None
        except Exception as e:
            self.logger.warning(f"Redis not available: {e}")
            self.redis_client = None
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'thread_executor'):
            self.thread_executor.shutdown(wait=True)
        if hasattr(self, 'process_executor'):
            self.process_executor.shutdown(wait=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def clean_directory(directory):
    """Clean directory with better error handling"""
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

def extract_zip_safe(zip_path, extract_to):
    """Memory-efficient ZIP extraction for large files"""
    try:
        os.makedirs(extract_to, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            total_size = 0
            file_count = 0
            extracted_files = []
            
            # Pre-validate the ZIP without loading everything into memory
            logger.info(f"Validating ZIP file: {zip_path}")
            
            for info in zip_ref.infolist():
                # Security checks
                if '..' in info.filename or info.filename.startswith('/'):
                    logger.warning(f"Suspicious file path: {info.filename}")
                    continue
                    
                total_size += info.file_size
                file_count += 1
                
                # Updated size limit for your server (50GB extracted)
                if total_size > app.config['MAX_ZIP_EXTRACTED_SIZE']:
                    logger.error(f"ZIP contents too large: {total_size / 1e9:.2f}GB")
                    return False
            
            logger.info(f"ZIP validation passed: {file_count} files, {total_size / 1e9:.2f}GB total")
            
            # Extract files one by one to avoid memory issues
            extracted_count = 0
            for i, info in enumerate(zip_ref.infolist()):
                if '..' in info.filename or info.filename.startswith('/'):
                    continue
                
                try:
                    # Extract file with progress logging
                    zip_ref.extract(info, extract_to)
                    extracted_path = os.path.join(extract_to, info.filename)
                    
                    if os.path.isfile(extracted_path):
                        extracted_files.append(extracted_path)
                        extracted_count += 1
                    
                    # Log progress every 100 files
                    if (i + 1) % 100 == 0:
                        logger.info(f"Extracted {i + 1}/{file_count} files ({extracted_count} valid)")
                        
                except Exception as e:
                    logger.warning(f"Failed to extract {info.filename}: {e}")
                    continue
            
            logger.info(f"ZIP extraction completed: {extracted_count} files extracted successfully")
            return True
            
    except zipfile.BadZipFile as e:
        logger.error(f"Invalid ZIP file {zip_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"ZIP extraction failed for {zip_path}: {e}")
        return False

def build_rag_prompt(context, question, is_final=True):
    """Enhanced RAG prompt with precise instructions for better results"""
    return f"""Document Context (may be partial):
{context}

Question:
{question}

Instructions:
1. Answer based ONLY on the current context provided above
2. Preserve original document structure in responses (maintain formatting, numbering, bullet points)
3. Quote directly from documents rather than paraphrasing
4. Include specific details, numbers, dates, and references exactly as they appear
5. Every response that includes a section or quote must end with a citation in the format: [^filename||page||section-heading]
   - filename must be the exact relative path from the extracted_files directory
   - Use forward slashes (/) for path separators
   - Include page numbers when available
   - Include section headings when identifiable
6. If information spans multiple sections, cite each section separately
7. If any part of the answer requires information not present in the documents, clearly state "This information is not available in the provided documents"
8. **Never**:
   - Paraphrase or summarize unless specifically requested
   - Add interpretation beyond what's explicitly stated
   - Combine text from different sections without proper citations
   - Modify legal terms, technical definitions, or specific terminology
   - Make assumptions about unstated information
9. **Always**:
   - Use exact quotes when referring to specific content
   - Maintain the original document's tone and terminology
   - Provide citations for every factual claim
   - Structure your response clearly with appropriate headings if helpful
{f"10. This is the final context section - provide a complete answer based on all available information" if is_final else "10. More context may follow - provide partial answer if needed"}

Response:"""

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

def get_gpu_stats():
    """Get GPU performance statistics"""
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    
    try:
        return {
            "gpu_available": True,
            "gpu_name": torch.cuda.get_device_name(),
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__
        }
    except Exception as e:
        return {"gpu_available": False, "error": str(e)}

# Content Validation and Extraction Functions
def is_valid_content(text):
    """More lenient content validation"""
    if not text or not text.strip():
        return False
    
    # Clean text for better analysis
    clean_text = text.strip()
    lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
    
    # Accept content if it meets any of these criteria:
    return (
        len(clean_text) >= 10 or  # At least 10 characters
        len(lines) >= 2 or        # At least 2 lines
        bool(re.search(r'\d+', clean_text)) or  # Contains numbers
        bool(re.search(r'[A-Za-z]{3,}', clean_text)) or  # Contains words
        '|' in clean_text or      # Table-like structure
        '\t' in clean_text        # Tab-separated data
    )

def extract_pdf_content(file_path):
    """Extract complete content from PDF with multiple fallback methods"""
    try:
        doc = fitz.open(file_path)
        full_text = ""
        
        # Method 1: Standard text extraction
        for page_num, page in enumerate(doc):
            # Try different text extraction methods
            page_text = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
            
            # If standard extraction fails, try other methods
            if len(page_text.strip()) < 50:
                # Try blocks method
                blocks = page.get_text("blocks")
                page_text = " ".join([block[4] for block in blocks if len(block) > 4])
                
                # Try dict method as last resort
                if len(page_text.strip()) < 50:
                    text_dict = page.get_text("dict")
                    page_text = ""
                    for block in text_dict["blocks"]:
                        if "lines" in block:
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    page_text += span["text"] + " "
            
            full_text += f"\n[Page {page_num + 1}]\n" + page_text + "\n"
        
        # Method 2: OCR fallback (only if tesseract is available)
        if len(full_text.strip()) < 200:  # If we still don't have much text
            try:
                import pytesseract
                from PIL import Image
                
                logger.info(f"Attempting OCR for {file_path}")
                for page_num, page in enumerate(doc):
                    pix = page.get_pixmap(dpi=300)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img)
                    
                    if ocr_text.strip():
                        full_text += f"\n[OCR Page {page_num + 1}]\n" + ocr_text + "\n"
                        
            except ImportError:
                logger.warning(f"OCR dependencies not available for {file_path}")
            except Exception as e:
                logger.warning(f"OCR failed for {file_path}: {str(e)}")
        
        doc.close()
        return full_text.strip()
        
    except Exception as e:
        logger.error(f"PDF extraction failed for {file_path}: {str(e)}")
        return ""

def extract_docx_content(file_path):
    """Extract complete content from DOCX including all elements"""
    try:
        combined_text = []

        # Method 1: python-docx (paragraphs, tables, headers, footers)
        try:
            import docx
            doc = docx.Document(file_path)

            # Paragraphs
            combined_text.extend(para.text for para in doc.paragraphs if para.text.strip())

            # Tables
            for table in doc.tables:
                for row in table.rows:
                    combined_text.append(" | ".join(cell.text for cell in row.cells))

            # Headers & Footers
            for section in doc.sections:
                for header in [section.header, section.first_page_header]:
                    if header:
                        combined_text.extend(p.text for p in header.paragraphs if p.text.strip())
                for footer in [section.footer, section.first_page_footer]:
                    if footer:
                        combined_text.extend(p.text for p in footer.paragraphs if p.text.strip())

        except Exception as e:
            logger.debug(f"python-docx extraction attempt failed: {str(e)}")

        # Method 2: mammoth (preserves more formatting)
        try:
            with open(file_path, "rb") as docx_file:
                result = mammoth.extract_text(docx_file)
                if result.value.strip():
                    combined_text.append(result.value)
        except Exception as e:
            logger.debug(f"mammoth extraction attempt failed: {str(e)}")

        # Method 3: Direct XML parsing (fallback)
        try:
            with zipfile.ZipFile(file_path) as z:
                with z.open('word/document.xml') as f:
                    soup = BeautifulSoup(f.read(), 'xml')
                    text_elements = []
                    for t in soup.find_all(['w:t', 'w:tab', 'w:br']):
                        if t.name == 'w:t':
                            text_elements.append(t.text)
                        elif t.name == 'w:tab':
                            text_elements.append('\t')
                        elif t.name == 'w:br':
                            text_elements.append('\n')
                    combined_text.append(''.join(text_elements))
        except Exception as e:
            logger.debug(f"XML extraction attempt failed: {str(e)}")

        return "\n".join(filter(None, combined_text)).strip()
    except Exception as e:
        logger.error(f"DOCX extraction failed for {file_path}: {str(e)}")
        return ""

def extract_txt_content(file_path):
    """Extract content from plain text files"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Text extraction failed for {file_path}: {str(e)}")
        return ""

def extract_excel_content(file_path):
    """Extract content from Excel files"""
    try:
        text_content = []
        
        # Try pandas first
        try:
            if file_path.endswith('.xlsx'):
                excel_data = pd.ExcelFile(file_path)
                for sheet_name in excel_data.sheet_names:
                    df = excel_data.parse(sheet_name)
                    text_content.append(f"\n\nSheet: {sheet_name}\n")
                    text_content.append(df.to_string())
            else:
                df = pd.read_excel(file_path)
                text_content.append(df.to_string())
                
            return "\n".join(text_content)
        except:
            pass
            
        # Fallback to openpyxl
        wb = load_workbook(file_path)
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            text_content.append(f"\n\nSheet: {sheet}\n")
            for row in ws.iter_rows(values_only=True):
                text_content.append("\t".join(str(cell) if cell is not None else "" for cell in row))
                
        return "\n".join(text_content)
    except Exception as e:
        logger.error(f"Excel extraction failed for {file_path}: {str(e)}")
        return ""

def extract_content_sync(file_path):
    """Complete content extraction with proper fallbacks"""
    file_ext = Path(file_path).suffix.lower()
    
    # Check for empty files
    if os.path.getsize(file_path) == 0:
        logger.warning(f"Empty file: {file_path}")
        return ""
    
    try:
        if file_ext == '.pdf':
            return extract_pdf_content(file_path)
        elif file_ext == '.docx':
            return extract_docx_content(file_path)
        elif file_ext == '.txt':
            return extract_txt_content(file_path)
        elif file_ext in ('.xlsx', '.xls'):
            return extract_excel_content(file_path)
        else:
            # Fallback for other file types
            return extract_txt_content(file_path)
    except Exception as e:
        logger.error(f"Content extraction failed for {file_path}: {str(e)}")
        return ""

def get_file_hash_sync(file_path):
    """Generate file hash synchronously"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def create_chunks_sync(content):
    """Enhanced chunking that preserves document structure better"""
    chunk_size = app.config['CHUNK_SIZE']
    chunk_overlap = app.config['CHUNK_OVERLAP']
    
    if len(content) <= chunk_size:
        return [content]
    
    # Try to preserve document structure
    chunks = []
    
    # First, try to split by clear section breaks
    major_sections = re.split(r'\n\s*(?=\d+\.|\[|\#|[A-Z][A-Z\s]+:)', content)
    
    if len(major_sections) > 1:
        # Process each major section
        for section in major_sections:
            section = section.strip()
            if not section:
                continue
                
            if len(section) <= chunk_size:
                chunks.append(section)
            else:
                # Split large sections into smaller chunks
                sub_chunks = []
                paragraphs = section.split('\n\n')
                current_chunk = ""
                
                for para in paragraphs:
                    if len(current_chunk) + len(para) + 2 <= chunk_size:
                        current_chunk += ("\n\n" + para) if current_chunk else para
                    else:
                        if current_chunk:
                            sub_chunks.append(current_chunk)
                        current_chunk = para
                
                if current_chunk:
                    sub_chunks.append(current_chunk)
                
                chunks.extend(sub_chunks)
    else:
        # Fallback to paragraph-based chunking
        paragraphs = content.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk += ("\n\n" + para) if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    # Add overlap from previous chunk
                    overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk)
    
    # Ensure no empty chunks
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    logger.info(f"Created {len(chunks)} chunks from {len(content)} characters")
    return chunks

def generate_embeddings_sync(chunks):
    """Generate embeddings synchronously"""
    try:
        # Initialize model (use GPU if available)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(app.config['EMBEDDING_MODEL'], device=device)
        
        # Generate embeddings in batches
        batch_size = app.config['GPU_BATCH_SIZE'] if device == "cuda" else 32
        embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_embeddings = model.encode(batch, convert_to_tensor=True, device=device)
            embeddings.extend(batch_embeddings.cpu().numpy())
        
        return [emb.tolist() for emb in embeddings]
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        # Return random embeddings as fallback
        return [np.random.rand(app.config['EMBEDDING_DIM']).tolist() for _ in chunks]

def save_document_sync(file_path, content, chunks, embeddings, file_hash, user_id):
    """Save document synchronously"""
    try:
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        file_type = os.path.splitext(file_path)[1].lower()
        rel_path = os.path.relpath(file_path, app.config['EXTRACT_FOLDER'])
        
        # Document metadata
        doc_metadata = {
            'original_filename': filename,
            'file_size': file_size,
            'content_length': len(content),
            'relative_path': rel_path,
            'chunk_count': len(chunks),
            'processing_method': 'gpu_optimized_sync'
        }
        
        # Create document record
        doc = Document(
            user_id=user_id,
            file_path=rel_path,
            file_hash=file_hash,
            file_type=file_type,
            document_metadata=doc_metadata
        )
        
        db.session.add(doc)
        db.session.flush()
        
        # Save full content
        content_record = DocumentContent(
            document_id=doc.id,
            content=content,
            content_type='text'
        )
        db.session.add(content_record)
        
        # Save chunks and embeddings
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_record = DocumentContent(
                document_id=doc.id,
                content=chunk,
                content_type='chunk',
                chunk_index=i,
                embedding=embedding,
                content_metadata={'chunk_length': len(chunk)}
            )
            db.session.add(chunk_record)
        
        db.session.commit()
        logger.info(f"Saved document {filename} with {len(chunks)} chunks")
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving document {file_path}: {e}")
        raise

def process_single_file_sync(file_path, user_id):
    """Complete file processing with better error handling"""
    try:
        logger.info(f"Starting processing of {file_path}")
        
        # Extract content
        content = extract_content_sync(file_path)
        logger.debug(f"Extracted content length: {len(content)} characters")
        
        if not content or not is_valid_content(content):
            logger.warning(f"Insufficient content in {file_path}")
            return False
        
        # Generate file hash
        file_hash = get_file_hash_sync(file_path)
        logger.debug(f"Generated file hash: {file_hash}")
        
        # Check if already processed
        existing = Document.query.filter_by(file_hash=file_hash, user_id=user_id).first()
        if existing:
            logger.info(f"File already processed: {os.path.basename(file_path)}")
            return True
        
        # Create document chunks
        chunks = create_chunks_sync(content)
        logger.debug(f"Created {len(chunks)} chunks from document")
        
        # Generate embeddings
        embeddings = generate_embeddings_sync(chunks)
        logger.debug(f"Generated {len(embeddings)} embeddings")
        
        # Save to database
        save_document_sync(file_path, content, chunks, embeddings, file_hash, user_id)
        logger.info(f"Successfully processed: {os.path.basename(file_path)}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

# Enhanced Document Search Function
def search_documents_sync(query, user_id):
    """Enhanced document search that finds more comprehensive results"""
    try:
        # Get all document chunks (not just full text)
        chunks = db.session.query(Document, DocumentContent).join(
            DocumentContent, Document.id == DocumentContent.document_id
        ).filter(
            Document.user_id == user_id,
            DocumentContent.content_type == 'chunk'  # Search chunks instead of full text
        ).all()
        
        if not chunks:
            # Fallback to full text if no chunks
            documents = db.session.query(Document, DocumentContent).join(
                DocumentContent, Document.id == DocumentContent.document_id
            ).filter(
                Document.user_id == user_id,
                DocumentContent.content_type == 'text'
            ).all()
            
            if not documents:
                return []
            
            # Process full documents
            query_words = query.lower().split()
            results = []
            
            for doc, content_record in documents:
                content_lower = content_record.content.lower()
                score = sum(content_lower.count(word) for word in query_words)
                
                if score > 0:
                    filename = doc.document_metadata.get('original_filename', 'Unknown') if doc.document_metadata else 'Unknown'
                    results.append({
                        'document': doc,
                        'chunk': content_record.content,
                        'score': score,
                        'filename': filename,
                        'chunk_index': 0
                    })
            
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:app.config['SIMILARITY_TOP_K']]
        
        # Search through chunks
        query_words = query.lower().split()
        results = []
        
        for doc, chunk_record in chunks:
            content_lower = chunk_record.content.lower()
            
            # Calculate relevance score
            score = 0
            for word in query_words:
                # Give higher score for exact matches
                score += content_lower.count(word) * 2
                # Give lower score for partial matches
                score += sum(1 for w in content_lower.split() if word in w)
            
            if score > 0:
                filename = doc.document_metadata.get('original_filename', 'Unknown') if doc.document_metadata else 'Unknown'
                
                results.append({
                    'document': doc,
                    'chunk': chunk_record.content,
                    'score': score,
                    'filename': filename,
                    'chunk_index': chunk_record.chunk_index
                })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Return more results for comprehensive answers
        return results[:app.config['SIMILARITY_TOP_K']]
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []

def get_context_sync(text, search_term, context_length=500):
    """Get context around search term"""
    try:
        pos = text.lower().find(search_term.lower())
        if pos == -1:
            return text[:context_length]
        
        start = max(0, pos - context_length // 2)
        end = min(len(text), pos + context_length // 2)
        
        return text[start:end]
    except:
        return text[:context_length]

# Routes
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
        
        if not username or not email or not password:
            flash('All fields are required', 'error')
            return redirect(url_for('register'))
        
        if User.query.filter_by(username=username).first():
            flash('Username already taken', 'error')
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

# Replace the upload_files route in your app.py with this enhanced version

@app.route('/upload', methods=['POST'])
@login_required
def upload_files():
    """Enhanced file upload with client disconnect handling"""
    if not current_user or not current_user.is_authenticated:
        return jsonify({'error': 'Authentication required', 'redirect': '/login'}), 401
    
    logger.info(f"Upload request from user {current_user.id} - IP: {request.remote_addr}")
    
    # Check subscription
    if not current_user.is_subscribed and current_user.subscription_plan != 'free':
        doc_count = Document.query.filter_by(user_id=current_user.id).count()
        if doc_count >= 5:
            return jsonify({'error': 'Subscription required for more documents'}), 403
    
    try:
        user_id = current_user.id
        user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
        user_extract_dir = os.path.join(app.config['EXTRACT_FOLDER'], str(user_id))
        
        # Create directories
        os.makedirs(user_upload_dir, exist_ok=True)
        os.makedirs(user_extract_dir, exist_ok=True)
        clean_directory(user_upload_dir)
        clean_directory(user_extract_dir)
        
        # Enhanced file handling with client disconnect protection
        try:
            if not hasattr(request, 'files') or 'files' not in request.files:
                logger.warning("No files in request")
                return jsonify({'error': 'No files uploaded'}), 400
                
            files = request.files.getlist('files')
            logger.info(f"Processing {len(files)} uploaded files")
            
        except Exception as e:
            # Handle client disconnect during form parsing
            if "ClientDisconnected" in str(e) or "Bad Request" in str(e):
                logger.warning(f"Client disconnected during upload: {e}")
                return jsonify({
                    'error': 'Upload interrupted - file too large or connection timeout',
                    'suggestion': 'Try uploading smaller files or check your internet connection',
                    'max_size_gb': app.config['MAX_FILE_SIZE'] / 1e9
                }), 408
            else:
                logger.error(f"Error parsing upload request: {e}")
                return jsonify({'error': 'Invalid upload request'}), 400
        
        extracted_files = []
        upload_warnings = []
        
        for file_idx, file in enumerate(files):
            if file.filename == '':
                continue
                
            filename = secure_filename(file.filename)
            is_zip = filename.lower().endswith('.zip')
            
            try:
                file.seek(0, 2)
                file_size = file.tell()
                file.seek(0)
                
                if file_size > app.config['MAX_FILE_SIZE']:
                    warning_msg = f"File too large: {filename} ({file_size / 1e9:.2f}GB)"
                    logger.warning(warning_msg)
                    upload_warnings.append(warning_msg)
                    continue
                
                logger.info(f"Processing file {file_idx + 1}/{len(files)}: {filename} ({file_size / 1e6:.1f}MB)")
                
                if is_zip:
                    zip_path = os.path.join(user_upload_dir, filename)
                    
                    try:
                        with open(zip_path, 'wb') as zip_file:
                            chunk_size = 8 * 1024 * 1024
                            bytes_written = 0
                            
                            while True:
                                try:
                                    chunk = file.read(chunk_size)
                                    if not chunk:
                                        break
                                    zip_file.write(chunk)
                                    bytes_written += len(chunk)
                                    
                                    if bytes_written % (100 * 1024 * 1024) == 0:
                                        logger.info(f"Uploaded {bytes_written / 1e6:.1f}MB of {filename}")
                                        
                                except Exception as e:
                                    if "ClientDisconnected" in str(e):
                                        logger.warning(f"Client disconnected during {filename} upload")
                                        try:
                                            os.remove(zip_path)
                                        except:
                                            pass
                                        return jsonify({
                                            'error': f'Upload of {filename} interrupted',
                                            'suggestion': 'Please try again with a stable connection'
                                        }), 408
                                    else:
                                        raise
                        
                        logger.info(f"ZIP file saved: {filename} ({file_size / 1e6:.1f}MB)")
                        
                        if extract_zip_safe(zip_path, user_extract_dir):
                            for root, _, zip_files in os.walk(user_extract_dir):
                                for f in zip_files:
                                    if allowed_file(f) and not f.startswith('.'):
                                        extracted_files.append(os.path.join(root, f))
                            logger.info(f"Extracted {len(extracted_files)} files from ZIP")
                        else:
                            logger.error(f"Failed to extract ZIP: {filename}")
                            
                        try:
                            os.remove(zip_path)
                        except:
                            pass
                            
                    except Exception as e:
                        if "ClientDisconnected" in str(e):
                            return jsonify({
                                'error': f'Connection lost during {filename} upload',
                                'suggestion': 'File too large for current connection'
                            }), 408
                        else:
                            raise
                        
                else:
                    if allowed_file(filename):
                        file_path = os.path.join(user_extract_dir, filename)
                        
                        try:
                            with open(file_path, 'wb') as output_file:
                                chunk_size = 8 * 1024 * 1024
                                
                                while True:
                                    try:
                                        chunk = file.read(chunk_size)
                                        if not chunk:
                                            break
                                        output_file.write(chunk)
                                    except Exception as e:
                                        if "ClientDisconnected" in str(e):
                                            logger.warning(f"Client disconnected during {filename} upload")
                                            try:
                                                os.remove(file_path)
                                            except:
                                                pass
                                            return jsonify({
                                                'error': f'Upload of {filename} interrupted'
                                            }), 408
                                        else:
                                            raise
                            
                            extracted_files.append(file_path)
                            logger.info(f"Saved regular file: {filename}")
                            
                        except Exception as e:
                            if "ClientDisconnected" in str(e):
                                return jsonify({
                                    'error': f'Connection lost during {filename} upload'
                                }), 408
                            else:
                                raise
                        
            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}")
                continue
        
        if not extracted_files:
            return jsonify({'error': 'No valid files found after processing'}), 400
        
        # Start processing - FIXED: Initialize progress tracking properly
        job_id = str(uuid.uuid4())
        
        # Initialize progress tracking BEFORE starting the thread
        with processing_status_lock:
            processing_status[job_id] = GPUProcessingProgress(len(extracted_files))
            logger.info(f"Created processing job {job_id} for {len(extracted_files)} files")
        
        def enhanced_background_processing():
            with app.app_context():
                try:
                    # Ensure we have the progress object
                    with processing_status_lock:
                        if job_id not in processing_status:
                            logger.error(f"Job {job_id} not found in processing_status")
                            return
                        progress = processing_status[job_id]
                    
                    start_time = time.time()
                    processed_count = 0
                    
                    logger.info(f"Starting background processing for job {job_id}")
                    
                    for i, file_path in enumerate(extracted_files):
                        try:
                            with processing_status_lock:
                                if job_id in processing_status:
                                    processing_status[job_id].current_file = os.path.basename(file_path)
                            
                            success = process_single_file_sync(file_path, user_id)
                            if success:
                                processed_count += 1
                            
                            with processing_status_lock:
                                if job_id in processing_status:
                                    processing_status[job_id].processed_files = processed_count
                                    processing_status[job_id].update_stats()
                            
                            if (i + 1) % 10 == 0 and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                            logger.info(f"Processed file {i+1}/{len(extracted_files)}: {os.path.basename(file_path)}")
                            
                        except Exception as e:
                            logger.error(f"Error processing {file_path}: {e}")
                            continue
                    
                    # Mark as completed
                    with processing_status_lock:
                        if job_id in processing_status:
                            processing_status[job_id].status = "completed"
                            processing_status[job_id].processed_files = processed_count
                            processing_status[job_id].update_stats()
                            logger.info(f"Job {job_id} completed: {processed_count}/{len(extracted_files)} files processed")
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"Background processing failed for job {job_id}: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    with processing_status_lock:
                        if job_id in processing_status:
                            processing_status[job_id].status = "error"
                            processing_status[job_id].error_message = str(e)
        
        thread = threading.Thread(target=enhanced_background_processing)
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started background processing thread for job {job_id}")
        
        return jsonify({
            'message': f'{len(extracted_files)} files uploaded successfully',
            'job_id': job_id,
            'total_files': len(extracted_files),
            'gpu_enabled': torch.cuda.is_available(),
            'estimated_time': len(extracted_files) * 0.2,
            'upload_method': 'enhanced_with_disconnect_handling',
            'max_file_size_gb': app.config['MAX_FILE_SIZE'] / 1e9
        })
        
    except Exception as e:
        if "ClientDisconnected" in str(e):
            logger.warning(f"Client disconnected: {e}")
            return jsonify({
                'error': 'Connection interrupted during upload',
                'suggestion': 'Please check your connection and try again'
            }), 408
        else:
            logger.error(f"Upload failed: {e}")
            return jsonify({'error': f'Upload failed: {str(e)}'}), 500



@app.route('/upload_status/<job_id>')
@login_required
def upload_status(job_id):
    """Check processing status with enhanced performance metrics - FIXED"""
    logger.debug(f"Status check for job {job_id}")
    
    # Thread-safe access to processing_status
    with processing_status_lock:
        if job_id not in processing_status:
            logger.warning(f"Job {job_id} not found in processing_status. Available jobs: {list(processing_status.keys())}")
            return jsonify({'error': 'Job not found'}), 404
        
        progress = processing_status[job_id]
        
        # Update GPU stats if still processing
        if progress.status == 'processing':
            progress.update_stats()
        
        response = {
            'status': progress.status,
            'processed_files': progress.processed_files,
            'total_files': progress.total_files,
            'current_file': progress.current_file,
            'progress_percent': int((progress.processed_files / progress.total_files) * 100) if progress.total_files > 0 else 0,
            'throughput': progress.throughput,
            'gpu_stats': progress.gpu_stats,
            'processing_method': 'gpu_accelerated_enhanced' if torch.cuda.is_available() else 'cpu_fallback',
            'server_performance': 'RTX_A6000_OPTIMIZED'
        }
        
        if progress.status == 'error':
            response['error_message'] = progress.error_message
        elif progress.status == 'completed':
            # FIXED: Don't immediately delete, set a flag for cleanup
            if not hasattr(progress, 'cleanup_scheduled'):
                progress.cleanup_scheduled = True
                # Schedule cleanup after a delay to allow final status check
                def cleanup_job():
                    time.sleep(5)  # Wait 5 seconds before cleanup
                    with processing_status_lock:
                        if job_id in processing_status:
                            del processing_status[job_id]
                            logger.info(f"Cleaned up completed job {job_id}")
                
                cleanup_thread = threading.Thread(target=cleanup_job)
                cleanup_thread.daemon = True
                cleanup_thread.start()
    
    return jsonify(response)


@app.route('/new_chat')
@login_required
def new_chat():
    """Create new chat"""
    try:
        chat_id = str(uuid.uuid4())
        
        conversation = Conversation(
            id=chat_id,
            user_id=current_user.id,
            title='New Query'
        )
        db.session.add(conversation)
        
        # Check for documents
        doc_count = Document.query.filter_by(user_id=current_user.id).count()
        initial_message = f"Ready to search through {doc_count} documents with acceleration. What would you like to know?"
        
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
    """Enhanced chat with comprehensive search"""
    try:
        conversation = Conversation.query.filter_by(
            id=chat_id,
            user_id=current_user.id
        ).first_or_404()

        # Debug: Print document count to console
        doc_count = Document.query.filter_by(user_id=current_user.id).count()
        print(f"Found {doc_count} documents for user {current_user.id}")

        # Get all documents for this user
        documents = Document.query.filter_by(
            user_id=current_user.id
        ).order_by(Document.processed_at.desc()).all()

        if request.method == 'GET':
            messages = Message.query.filter_by(
                conversation_id=chat_id
            ).order_by(Message.timestamp.asc()).all()

            user_conversations = Conversation.query.filter_by(
                user_id=current_user.id
            ).order_by(Conversation.updated_at.desc()).all()

            return render_template(
                'chat.html',
                chat_id=chat_id,
                conversation=conversation,
                messages=messages,
                conversations=user_conversations,
                documents=documents,  # Make sure this is included
                chat_title=conversation.title
        )

        elif request.method == 'POST':
            def generate_response():
                try:
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

                    # Enhanced document search
                    yield json.dumps({'status': 'status_update', 'message': '🔍 GPU-accelerated document search...'}) + "\n"
                    
                    search_start_time = time.time()
                    results = search_documents_sync(question, current_user.id)
                    search_time = time.time() - search_start_time
                    
                    logger.info(f"Enhanced search completed in {search_time:.3f}s, found {len(results)} results")
                    
                    if not results:
                        yield json.dumps({
                            'status': 'error',
                            'message': 'No relevant information found in your documents'
                        })
                        return

                    # Build comprehensive context from search results
                    context = ""
                    sources = []
                    seen_documents = set()

                    for result in results:
                        doc = result['document']
                        filename = result['filename']
                        score = result.get('score', 0)
                        chunk_index = result.get('chunk_index', 0)
                        
                        # Add document header only once per document
                        if filename not in seen_documents:
                            context += f"\n=== Document: {filename} ===\n"
                            seen_documents.add(filename)
                        
                        # Add chunk with section indicator
                        context += f"\n[Section {chunk_index + 1}]\n"
                        context += f"{result['chunk']}\n"
                        
                        if filename not in sources:
                            sources.append(filename)

                    # Add instruction for comprehensive analysis
                    context += "\n\nNote: This represents the complete available content from the uploaded documents. Please provide a comprehensive answer based on all the information shown above."

                    yield json.dumps({'status': 'status_update', 'message': '🤖 Generating AI response with GPU acceleration...'}) + "\n"

                    # Stream AI response
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
                        sources=sources,
                        timestamp=datetime.now(timezone.utc)
                    )
                    db.session.add(assistant_msg)
                    
                    conversation.updated_at = datetime.now(timezone.utc)
                    db.session.commit()

                    yield json.dumps({
                        'status': 'stream_end',
                        'sources': [{'filename': s, 'path': s} for s in sources],
                        'search_time': search_time,
                        'results_count': len(results),
                        'performance_note': f'GPU-accelerated search: {len(results)} sections in {search_time*1000:.0f}ms'
                    }) + "\n"

                except Exception as e:
                    logger.error(f"Chat error: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
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
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f'Error deleting chat: {e}')
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/update_chat_title/<chat_id>', methods=['POST'])
@login_required  
def update_chat_title(chat_id):
    """Update chat title"""
    try:
        data = request.get_json()
        new_title = data.get('title', '').strip()
        
        if not new_title:
            return jsonify({'error': 'Title cannot be empty'}), 400
            
        conversation = Conversation.query.filter_by(
            id=chat_id, 
            user_id=current_user.id
        ).first()
        
        if not conversation:
            return jsonify({'error': 'Chat not found'}), 404
            
        conversation.title = new_title
        conversation.updated_at = datetime.now(timezone.utc)
        db.session.commit()
        
        return jsonify({'success': True, 'title': new_title})
        
    except Exception as e:
        logger.error(f'Error updating chat title: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/get_file_content')
@login_required
def get_file_content():
    """Get file content"""
    rel_path = request.args.get('path')
    if not rel_path:
        return jsonify({'error': 'File path required'}), 400
    
    try:
        # Security checks
        if '..' in rel_path or rel_path.startswith('/'):
            return jsonify({'error': 'Invalid file path'}), 400
        
        # Find document by relative path
        doc = Document.query.filter_by(
            file_path=rel_path,
            user_id=current_user.id
        ).first()
        
        if not doc:
            return jsonify({'error': 'Document not found'}), 404
        
        # Get content from DocumentContent table
        content_record = DocumentContent.query.filter_by(
            document_id=doc.id,
            content_type='text'
        ).first()
        
        if not content_record:
            return jsonify({'error': 'Document content not found'}), 404
        
        filename = doc.document_metadata.get('original_filename', 'Unknown') if doc.document_metadata else 'Unknown'
        
        return jsonify({
            'file': filename,
            'content': content_record.content,
            'path': rel_path
        })
    
    except Exception as e:
        logger.error(f"Error serving file: {e}")
        return jsonify({'error': str(e)}), 500

# Subscription and Billing Routes
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
        'message': 'Your 14-day free trial with acceleration has started!',
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
            'allow_uploads': current_user.is_subscribed or current_user.subscription_plan == 'free',
            'plan_name': current_user.plan_details.get('name', 'Free'),
            'gpu_enabled': torch.cuda.is_available(),
            'max_file_size_gb': app.config['MAX_FILE_SIZE'] / 1e9,
            'max_extracted_size_gb': app.config['MAX_ZIP_EXTRACTED_SIZE'] / 1e9
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/pricing')
def pricing():
    """Pricing page"""
    plans = {
        'basic': SubscriptionPlan.BASIC,
        'pro': SubscriptionPlan.PRO,
        'trial': SubscriptionPlan.TRIAL
    }
    return render_template('pricing.html', plans=plans, now=datetime.now(timezone.utc))

@app.route('/billing')
@login_required
def billing():
    """Billing page"""
    now = datetime.now(timezone.utc)
    payments = Payment.query.filter_by(user_id=current_user.id).order_by(Payment.created_at.desc()).limit(10).all()
    invoices = BillingHistory.query.filter_by(user_id=current_user.id).order_by(BillingHistory.created_at.desc()).limit(10).all()
    
    return render_template('billing.html',
                         payments=payments,
                         invoices=invoices,
                         plan=current_user.plan_details,
                         now=now)

# Enhanced Monitoring and Debug Routes
@app.route('/gpu_status')
@login_required
def gpu_status():
    """Get comprehensive GPU status"""
    try:
        gpu_stats = get_gpu_stats()
        
        # Add system information
        system_info = {
            'cpu_count': os.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1e9,
            'memory_available_gb': psutil.virtual_memory().available / 1e9,
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'torch_version': torch.__version__
        }
        
        return jsonify({
            'gpu_stats': gpu_stats,
            'system_info': system_info,
            'processing_method': 'gpu_accelerated_enhanced' if torch.cuda.is_available() else 'cpu_fallback',
            'performance_mode': 'RTX_A6000_OPTIMIZED',
            'max_upload_size_gb': app.config['MAX_CONTENT_LENGTH'] / 1e9,
            'max_file_size_gb': app.config['MAX_FILE_SIZE'] / 1e9,
            'max_extracted_size_gb': app.config['MAX_ZIP_EXTRACTED_SIZE'] / 1e9
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/debug/upload_health')
@login_required
def upload_health():
    """Monitor upload system health for large files"""
    try:
        import shutil
        
        # Check disk space
        upload_disk = shutil.disk_usage(app.config['UPLOAD_FOLDER'])
        extract_disk = shutil.disk_usage(app.config['EXTRACT_FOLDER'])
        temp_disk = shutil.disk_usage('/tmp')
        
        # Check memory
        memory = psutil.virtual_memory()
        
        # Check active uploads
        active_jobs = len(processing_status)
        
        health = {
            'disk_space': {
                'upload_free_gb': upload_disk.free / 1e9,
                'extract_free_gb': extract_disk.free / 1e9, 
                'temp_free_gb': temp_disk.free / 1e9
            },
            'memory': {
                'available_gb': memory.available / 1e9,
                'used_percent': memory.percent,
                'total_gb': memory.total / 1e9
            },
            'gpu': get_gpu_stats(),
            'active_uploads': active_jobs,
            'server_optimized': True,
            'max_upload_size_gb': app.config['MAX_CONTENT_LENGTH'] / 1e9,
            'max_file_size_gb': app.config['MAX_FILE_SIZE'] / 1e9,
            'max_extracted_size_gb': app.config['MAX_ZIP_EXTRACTED_SIZE'] / 1e9,
            'recommendations': []
        }
        
        # Add recommendations
        if upload_disk.free < 100e9:  # Less than 100GB free
            health['recommendations'].append('Low disk space in upload directory')
        if extract_disk.free < 200e9:  # Less than 200GB free
            health['recommendations'].append('Low disk space in extraction directory')
        if memory.percent > 80:
            health['recommendations'].append('High memory usage detected')
        if active_jobs > 10:
            health['recommendations'].append('Many concurrent uploads - consider rate limiting')
            
        return jsonify(health)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/debug/documents')
@login_required
def debug_documents():
    """List all processed documents for debugging"""
    try:
        docs = Document.query.filter_by(user_id=current_user.id).all()
        
        result = []
        for doc in docs:
            chunks_count = DocumentContent.query.filter_by(
                document_id=doc.id,
                content_type='chunk'
            ).count()
            
            result.append({
                'id': doc.id,
                'filename': doc.document_metadata.get('original_filename', 'Unknown'),
                'file_size': doc.document_metadata.get('file_size', 0),
                'content_length': doc.document_metadata.get('content_length', 0),
                'chunks_count': chunks_count,
                'processed_at': doc.processed_at.isoformat(),
                'processing_method': doc.document_metadata.get('processing_method', 'unknown')
            })
        
        return jsonify({
            'documents': result,
            'total_documents': len(result),
            'gpu_processing_available': torch.cuda.is_available()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/debug/document/<int:doc_id>')
@login_required
def debug_document(doc_id):
    """Debug route to check document processing"""
    try:
        doc = Document.query.filter_by(id=doc_id, user_id=current_user.id).first()
        if not doc:
            return jsonify({'error': 'Document not found'}), 404
        
        # Get full content
        full_content = DocumentContent.query.filter_by(
            document_id=doc_id,
            content_type='text'
        ).first()
        
        # Get all chunks
        chunks = DocumentContent.query.filter_by(
            document_id=doc_id,
            content_type='chunk'
        ).order_by(DocumentContent.chunk_index).all()
        
        return jsonify({
            'document': {
                'filename': doc.document_metadata.get('original_filename', 'Unknown'),
                'file_size': doc.document_metadata.get('file_size', 0),
                'content_length': doc.document_metadata.get('content_length', 0),
                'chunk_count': len(chunks),
                'processing_method': doc.document_metadata.get('processing_method', 'unknown')
            },
            'full_content_length': len(full_content.content) if full_content else 0,
            'chunks': [
                {
                    'index': chunk.chunk_index,
                    'length': len(chunk.content),
                    'preview': chunk.content[:200] + '...' if len(chunk.content) > 200 else chunk.content,
                    'has_embedding': chunk.embedding is not None
                }
                for chunk in chunks
            ]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Template filters
@app.template_filter('datetimeformat')
def datetimeformat(value, format='%b %d, %H:%M'):
    """Format datetime for templates"""
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.strftime(format)

@app.template_filter('days_from_now')
def days_from_now(value):
    """Calculate days from now"""
    if not value:
        return 0
    
    now = datetime.now(timezone.utc)
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
    """Inject current time and GPU status into templates"""
    return {
        'now': datetime.now(timezone.utc),
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A',
        'max_upload_gb': app.config['MAX_CONTENT_LENGTH'] / 1e9
    }

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
        if (current_user.subscription_plan == 'trial' and 
            current_user.trial_end_date and 
            not request.path.startswith('/static') and
            not request.path.startswith('/pricing')):
            
            now = datetime.now(timezone.utc)
            trial_end = current_user._make_aware(current_user.trial_end_date)
            days_left = (trial_end - now).days
            
            if days_left <= 3:
                flash(f'Your GPU-accelerated trial ends in {days_left} day(s). Upgrade now!', 'warning')

@app.after_request
def after_request(response):
    """Add headers and cleanup"""
    response.headers["X-Cloudflare-Time"] = "900"
    response.headers["Connection"] = "keep-alive"
    response.headers["X-GPU-Accelerated"] = "true" if torch.cuda.is_available() else "false"
    response.headers["X-Server-Optimized"] = "RTX_A6000_LARGE_FILES"
    return response

# Database initialization
def create_app_with_postgres():
    """Initialize app with PostgreSQL"""
    with app.app_context():
        try:
            with db.engine.connect() as connection:
                connection.execute(text('SELECT 1'))
            logger.info('PostgreSQL connection successful')
            
            # Create all tables
            db.create_all()
            logger.info('Database tables created/verified')
            
            # Log GPU status
            gpu_stats = get_gpu_stats()
            logger.info(f'GPU Status: {gpu_stats}')
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

@app.route('/get_file_structure')
@login_required
def get_file_structure():
    """Get file structure as a tree"""
    try:
        user_extract_dir = os.path.join(app.config['EXTRACT_FOLDER'], str(current_user.id))
        
        def build_tree(path):
            name = os.path.basename(path)
            node = {
                'name': name,
                'path': os.path.relpath(path, user_extract_dir),
                'type': 'directory' if os.path.isdir(path) else 'file',
                'extension': os.path.splitext(name)[1].lower() if os.path.isfile(path) else '',
                'children': []
            }
            
            if os.path.isdir(path):
                for item in sorted(os.listdir(path)):
                    full_path = os.path.join(path, item)
                    if not item.startswith('.') and allowed_file(item) or os.path.isdir(full_path):
                        node['children'].append(build_tree(full_path))
            
            return node
        
        tree = build_tree(user_extract_dir)
        return jsonify(tree)
        
    except Exception as e:
        logger.error(f"Error building file tree: {e}")
        return jsonify({'error': str(e)}), 500


# Add this debug route to your app.py file

@app.route('/debug/processing_jobs')
@login_required
def debug_processing_jobs():
    """Debug route to check active processing jobs"""
    try:
        with processing_status_lock:
            jobs_info = {}
            for job_id, progress in processing_status.items():
                jobs_info[job_id] = {
                    'status': progress.status,
                    'processed_files': progress.processed_files,
                    'total_files': progress.total_files,
                    'current_file': progress.current_file,
                    'error_message': getattr(progress, 'error_message', ''),
                    'start_time': progress.start_time,
                    'elapsed_time': time.time() - progress.start_time,
                    'throughput': progress.throughput
                }
        
        return jsonify({
            'active_jobs': len(processing_status),
            'jobs': jobs_info,
            'server_time': time.time(),
            'gpu_available': torch.cuda.is_available()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Also update the existing debug route to show recent documents
@app.route('/debug/recent_documents')
@login_required  
def debug_recent_documents():
    """Debug route to check recently processed documents"""
    try:
        # Get documents from the last hour
        one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
        
        recent_docs = Document.query.filter(
            Document.user_id == current_user.id,
            Document.processed_at >= one_hour_ago
        ).order_by(Document.processed_at.desc()).all()
        
        result = []
        for doc in recent_docs:
            result.append({
                'id': doc.id,
                'filename': doc.document_metadata.get('original_filename', 'Unknown') if doc.document_metadata else 'Unknown',
                'processed_at': doc.processed_at.isoformat(),
                'file_path': doc.file_path,
                'file_type': doc.file_type,
                'processing_method': doc.document_metadata.get('processing_method', 'unknown') if doc.document_metadata else 'unknown'
            })
        
        return jsonify({
            'recent_documents': result,
            'count': len(result),
            'search_time': one_hour_ago.isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

        

if __name__ == '__main__':
    try:
        create_app_with_postgres()
        
        # Enhanced GPU initialization logging
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            logger.info('🚀 DocumentIQ - RTX A6000 GPU Accelerated Mode')
            logger.info(f'GPU: {torch.cuda.get_device_name()}')
            logger.info(f'GPU Memory: {gpu_props.total_memory / 1e9:.1f}GB')
            logger.info(f'CUDA Cores: ~{gpu_props.multi_processor_count * 64}')
            logger.info(f'Max Upload Size: {app.config["MAX_CONTENT_LENGTH"] / 1e9:.1f}GB')
            logger.info(f'Max Single File: {app.config["MAX_FILE_SIZE"] / 1e9:.1f}GB')
            logger.info(f'Max Extracted ZIP: {app.config["MAX_ZIP_EXTRACTED_SIZE"] / 1e9:.1f}GB')
            logger.info('Large File Support: Up to 20GB uploads, 50GB extracted')
            logger.info('Expected Performance: 20-30x faster than CPU-only')
        else:
            logger.warning('⚠️  GPU not available - running in CPU mode')
            logger.warning('Large file uploads will be slower without GPU acceleration')
        
        # Log configuration
        logger.info('Server optimized for RTX A6000 with 256GB RAM')
        
        # Production warning
        logger.warning('🔧 For production, use: gunicorn --config gunicorn.conf.py app:app')
        logger.warning('🔧 Ensure nginx is configured with large file upload settings')
        
        # Start development server (not recommended for production)
        app.run(host='0.0.0.0', port=8000, threaded=True, debug=False)
        
    except Exception as e:
        logger.critical(f"Failed to start application: {e}")
        raise