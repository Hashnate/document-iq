import os
import zipfile
import PyPDF2
import docx
import shutil
import requests
import uuid
import json
import time
import threading
import hashlib
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, stream_with_context, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
import re
from pdfminer.high_level import extract_text as pdfminer_extract
import pdfplumber
from bs4 import BeautifulSoup
import csv
import xml.etree.ElementTree as ET
from flask import send_file
from flask_migrate import Migrate
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_session import Session

# Database imports
from sqlalchemy import text, func
from sqlalchemy.exc import OperationalError

# Simplified imports - no GPU complexity
import numpy as np

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

processing_status = {}

class ProcessingProgress:
    def __init__(self, total_files):
        self.total_files = total_files
        self.processed_files = 0
        self.current_file = ""
        self.status = "processing"
        self.error_message = ""
        self.start_time = time.time()

# Simplified Configuration
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
    LOG_FILE = os.path.join(os.getcwd(), 'app.log')
    LOG_LEVEL = logging.INFO

    # PostgreSQL configuration
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is required")
    
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'max_overflow': 20,
        'pool_timeout': 30,
    }
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Simplified RAG configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    SIMILARITY_TOP_K = 5
    
    # Session configuration
    SESSION_TYPE = 'filesystem'
    SESSION_FILE_DIR = '/tmp/flask_sessions'
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    SESSION_KEY_PREFIX = 'documentiq:'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=1)

# Apply configuration
app.config.from_object(Config)

# Create directories
os.makedirs('/tmp/flask_sessions', exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['EXTRACT_FOLDER'], exist_ok=True)
os.makedirs(app.config['CACHE_FOLDER'], exist_ok=True)

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
        return SubscriptionPlan.get_plan(self.subscription_plan) or SubscriptionPlan.get_plan('basic')
    
    def can_upload_file(self, file_size):
        if self.is_admin:
            return True
        
        if self.subscription_plan == 'trial' and self.is_subscribed:
            return True
        
        plan = self.plan_details.get('limits', {})
        max_size = plan.get('max_file_size', 10 * 1024 * 1024)
        if file_size > max_size:
            return False
                
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
    """Separate table for storing extracted content"""
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    content_type = db.Column(db.String(50), default='text')
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    document = db.relationship('Document', backref='content_records')

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Simplified Document Processor - NO GPU, NO FAISS
class SimpleDocumentProcessor:
    def __init__(self, user_id=None):
        self.user_id = user_id
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized SimpleDocumentProcessor")

    def process_files(self, file_paths):
        """Process files and store content in database"""
        processed_count = 0
        
        for file_path in file_paths:
            try:
                self.logger.info(f"Processing: {file_path}")
                
                # Extract text content
                content = self.extract_text(file_path)
                if not content or not content.strip():
                    self.logger.warning(f"No content extracted from: {file_path}")
                    continue
                
                # Store in database
                file_hash = self.get_file_hash(file_path)
                filename = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)
                file_type = os.path.splitext(file_path)[1].lower()
                
                # Check if already exists
                existing = Document.query.filter_by(file_hash=file_hash, user_id=self.user_id).first()
                if existing:
                    self.logger.info(f"Document already exists: {filename}")
                    continue
                
                # Calculate relative path from EXTRACT_FOLDER
                rel_path = os.path.relpath(file_path, app.config['EXTRACT_FOLDER'])
                
                # Create document metadata
                doc_metadata = {
                    'original_filename': filename,
                    'file_size': file_size,
                    'content_length': len(content),
                    'relative_path': rel_path
                }
                
                # Create new document record
                doc = Document(
                    user_id=self.user_id,
                    file_path=rel_path,
                    file_hash=file_hash,
                    file_type=file_type,
                    document_metadata=doc_metadata
                )
                
                db.session.add(doc)
                db.session.flush()  # Get the document ID
                
                # Store content in separate table
                content_record = DocumentContent(
                    document_id=doc.id,
                    content=content,
                    content_type='text'
                )
                
                db.session.add(content_record)
                db.session.commit()
                
                processed_count += 1
                self.logger.info(f"Successfully processed: {filename}")
                
            except Exception as e:
                db.session.rollback()
                self.logger.error(f"Error processing {file_path}: {e}")
                continue
        
        return processed_count

    def search_documents(self, query, user_id):
        """Simple text search in documents"""
        try:
            # Get all documents for user with their content
            documents = db.session.query(Document, DocumentContent).join(
                DocumentContent, Document.id == DocumentContent.document_id
            ).filter(Document.user_id == user_id).all()
            
            if not documents:
                return []
            
            # Simple keyword matching
            query_words = query.lower().split()
            results = []
            
            for doc, content_record in documents:
                content_lower = content_record.content.lower()
                
                # Calculate relevance score based on keyword matches
                score = 0
                for word in query_words:
                    score += content_lower.count(word)
                
                if score > 0:
                    # Get context around first match
                    context = self.get_context(content_record.content, query_words[0], 500)
                    filename = doc.document_metadata.get('original_filename', 'Unknown')
                    
                    results.append({
                        'document': doc,
                        'content': content_record.content,
                        'score': score,
                        'context': context,
                        'filename': filename
                    })
            
            # Sort by score
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:app.config['SIMILARITY_TOP_K']]
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def get_context(self, text, search_term, context_length=500):
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

    def extract_text(self, file_path):
        """Extract text from various file types"""
        if not os.path.exists(file_path):
            return None
        
        try:
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.pdf':
                return self.extract_pdf_text(file_path)
            elif ext == '.docx':
                return self.extract_docx_text(file_path)
            elif ext == '.txt':
                return self.extract_txt_text(file_path)
            elif ext in ['.md', '.markdown']:
                return self.extract_txt_text(file_path)
            elif ext == '.csv':
                return self.extract_csv_text(file_path)
            elif ext == '.json':
                return self.extract_json_text(file_path)
            elif ext == '.xml':
                return self.extract_xml_text(file_path)
            elif ext == '.html':
                return self.extract_html_text(file_path)
            else:
                self.logger.warning(f"Unsupported file type: {ext}")
                return None
                
        except Exception as e:
            self.logger.error(f"Text extraction failed for {file_path}: {e}")
            return None

    def extract_pdf_text(self, file_path):
        """Extract text from PDF - try multiple methods"""
        text = ""
        
        # Method 1: Try pdfminer
        try:
            text = pdfminer_extract(file_path)
            if text and text.strip():
                return text
        except Exception as e:
            self.logger.warning(f"PDFMiner failed: {e}")
        
        # Method 2: Try pdfplumber
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                return text
        except Exception as e:
            self.logger.warning(f"PDFPlumber failed: {e}")
        
        # Method 3: Try PyPDF2
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                return text
        except Exception as e:
            self.logger.warning(f"PyPDF2 failed: {e}")
        
        return text if text.strip() else None

    def extract_docx_text(self, file_path):
        """Extract text from DOCX"""
        try:
            doc = docx.Document(file_path)
            text = []
            for para in doc.paragraphs:
                text.append(para.text)
            return "\n".join(text)
        except Exception as e:
            self.logger.error(f"DOCX extraction failed: {e}")
            return None

    def extract_txt_text(self, file_path):
        """Extract text from TXT/MD files"""
        encodings = ['utf-8', 'iso-8859-1', 'windows-1252', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        return None

    def extract_csv_text(self, file_path):
        """Extract text from CSV"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                return "\n".join([", ".join(row) for row in rows])
        except Exception as e:
            self.logger.error(f"CSV extraction failed: {e}")
            return None

    def extract_json_text(self, file_path):
        """Extract text from JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return json.dumps(data, indent=2)
        except Exception as e:
            self.logger.error(f"JSON extraction failed: {e}")
            return None

    def extract_xml_text(self, file_path):
        """Extract text from XML"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            return ET.tostring(root, encoding='unicode')
        except Exception as e:
            self.logger.error(f"XML extraction failed: {e}")
            return None

    def extract_html_text(self, file_path):
        """Extract text from HTML"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                return soup.get_text()
        except Exception as e:
            self.logger.error(f"HTML extraction failed: {e}")
            return None

    def get_file_hash(self, file_path):
        """Generate file hash"""
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except:
            return None

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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

def extract_zip_safe(zip_path, extract_to):
    """Safely extract ZIP files"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Check for zip bombs and suspicious files
            total_size = 0
            for info in zip_ref.infolist():
                total_size += info.file_size
                if total_size > 1000 * 1024 * 1024:  # 1GB limit
                    logger.warning("ZIP file too large, extraction stopped")
                    return False
                    
                # Check for path traversal
                if '..' in info.filename or info.filename.startswith('/'):
                    logger.warning(f"Suspicious file path: {info.filename}")
                    continue
            
            zip_ref.extractall(extract_to)
            return True
    except Exception as e:
        logger.error(f"ZIP extraction failed: {e}")
        return False

def build_rag_prompt(context, question):
    """Build RAG prompt"""
    return f"""Use the following documents to answer the question:

Documents:
{context}

Question: {question}

Answer based on the documents provided. If the information is not in the documents, say so."""

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

@app.route('/upload', methods=['POST'])
@login_required
def upload_files():
    """Upload and process files"""
    if not current_user or not current_user.is_authenticated:
        return jsonify({'error': 'Authentication required', 'redirect': '/login'}), 401
    
    logger.info(f"Upload request from user {current_user.id}")
    
    # Check subscription (allow free users to upload a few files)
    if not current_user.is_subscribed and current_user.subscription_plan != 'free':
        doc_count = Document.query.filter_by(user_id=current_user.id).count()
        if doc_count >= 5:  # Free limit
            return jsonify({'error': 'Subscription required for more documents'}), 403
    
    try:
        user_id = current_user.id
        user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
        user_extract_dir = os.path.join(app.config['EXTRACT_FOLDER'], str(user_id))
        
        os.makedirs(user_upload_dir, exist_ok=True)
        os.makedirs(user_extract_dir, exist_ok=True)
        
        # Clean directories
        clean_directory(user_upload_dir)
        clean_directory(user_extract_dir)
        
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
            
        files = request.files.getlist('files')
        extracted_files = []
        
        for file in files:
            if file.filename == '':
                continue
                
            filename = secure_filename(file.filename)
            is_zip = filename.lower().endswith('.zip')
            
            logger.info(f"Processing uploaded file: {filename}")
            
            if is_zip:
                # Handle ZIP files
                zip_path = os.path.join(user_upload_dir, filename)
                file.save(zip_path)
                
                if extract_zip_safe(zip_path, user_extract_dir):
                    # Find all extracted files
                    for root, _, zip_files in os.walk(user_extract_dir):
                        for f in zip_files:
                            if allowed_file(f) and not f.startswith('.'):
                                extracted_files.append(os.path.join(root, f))
                    logger.info(f"Extracted {len(extracted_files)} files from ZIP")
                else:
                    logger.error(f"Failed to extract ZIP: {filename}")
            else:
                # Handle regular files
                if allowed_file(filename):
                    file_path = os.path.join(user_extract_dir, filename)
                    file.save(file_path)
                    extracted_files.append(file_path)
                    logger.info(f"Saved file: {filename}")
        
        if not extracted_files:
            return jsonify({'error': 'No valid files found'}), 400
        
        # Process files immediately (simplified)
        job_id = str(uuid.uuid4())
        processing_status[job_id] = ProcessingProgress(len(extracted_files))
        
        def process_in_background():
            try:
                # CRITICAL: Use app context for database operations
                with app.app_context():
                    progress = processing_status[job_id]
                    processor = SimpleDocumentProcessor(user_id=user_id)
                    
                    processed_count = processor.process_files(extracted_files)
                    
                    progress.status = "completed"
                    progress.processed_files = len(extracted_files)
                    logger.info(f"Processing complete: {processed_count} files processed")
                
            except Exception as e:
                logger.error(f"Background processing failed: {e}")
                if job_id in processing_status:
                    processing_status[job_id].status = "error"
                    processing_status[job_id].error_message = str(e)
        
        # Start background processing
        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'message': f'{len(extracted_files)} files uploaded successfully',
            'job_id': job_id,
            'total_files': len(extracted_files)
        })
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return jsonify({'error': str(e)})

@app.route('/debug/migrate_existing_docs')
@login_required  
def migrate_existing_docs():
    """Migrate existing documents to new schema if needed"""
    try:
        # Find documents that don't have content records
        docs_without_content = db.session.query(Document).outerjoin(
            DocumentContent, Document.id == DocumentContent.document_id
        ).filter(DocumentContent.id.is_(None)).all()
        
        if not docs_without_content:
            return jsonify({
                'message': 'No documents need migration',
                'migrated_count': 0
            })
        
        migrated_count = 0
        processor = SimpleDocumentProcessor(current_user.id)
        
        for doc in docs_without_content:
            try:
                # Reconstruct full file path
                full_path = os.path.join(app.config['EXTRACT_FOLDER'], doc.file_path)
                
                if os.path.exists(full_path):
                    # Extract content again
                    content = processor.extract_text(full_path)
                    
                    if content and content.strip():
                        # Create content record
                        content_record = DocumentContent(
                            document_id=doc.id,
                            content=content,
                            content_type='text'
                        )
                        
                        db.session.add(content_record)
                        migrated_count += 1
                        
                        # Update metadata if needed
                        if not doc.document_metadata:
                            doc.document_metadata = {}
                        
                        doc.document_metadata['original_filename'] = os.path.basename(full_path)
                        doc.document_metadata['content_length'] = len(content)
                        
                        logger.info(f"Migrated document: {doc.file_path}")
                        
            except Exception as e:
                logger.error(f"Failed to migrate document {doc.file_path}: {e}")
                continue
        
        db.session.commit()
        
        return jsonify({
            'message': f'Successfully migrated {migrated_count} documents',
            'migrated_count': migrated_count,
            'total_found': len(docs_without_content)
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/upload_status/<job_id>')
@login_required
def upload_status(job_id):
    """Check processing status"""
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
        
    progress = processing_status[job_id]
    
    response = {
        'status': progress.status,
        'processed_files': progress.processed_files,
        'total_files': progress.total_files,
        'current_file': progress.current_file,
        'progress_percent': int((progress.processed_files / progress.total_files) * 100) if progress.total_files > 0 else 0
    }
    
    if progress.status == 'error':
        response['error_message'] = progress.error_message
    elif progress.status == 'completed':
        # Clean up
        if job_id in processing_status:
            del processing_status[job_id]
    
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
        initial_message = f"Ready to search through {doc_count} documents. What would you like to know?"
        
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
    """Chat interface"""
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

            return render_template(
                'chat.html',
                chat_id=chat_id,
                conversation=conversation,
                messages=messages,
                conversations=user_conversations,
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

                    # Search documents
                    yield json.dumps({'status': 'status_update', 'message': '🔍 Searching documents...'}) + "\n"
                    
                    processor = SimpleDocumentProcessor(user_id=current_user.id)
                    results = processor.search_documents(question, current_user.id)
                    
                    if not results:
                        yield json.dumps({
                            'status': 'error',
                            'message': 'No relevant information found in your documents'
                        })
                        return

                    # Build context
                    context = ""
                    sources = []
                    for result in results:
                        doc = result['document']
                        filename = result['filename']
                        context += f"[Document: {filename}]\n"
                        context += f"{result['context']}\n\n"
                        sources.append(filename)

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
                        'sources': [{'filename': s, 'path': s} for s in sources]
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
        content_record = DocumentContent.query.filter_by(document_id=doc.id).first()
        
        if not content_record:
            return jsonify({'error': 'Document content not found'}), 404
        
        filename = doc.document_metadata.get('original_filename', 'Unknown') if doc.document_metadata else 'Unknown'
        
        # Return the stored content
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
            'allow_uploads': current_user.is_subscribed or current_user.subscription_plan == 'free',
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

@app.after_request
def after_request(response):
    """Add headers to prevent timeouts"""
    response.headers["X-Cloudflare-Time"] = "900"
    response.headers["Connection"] = "keep-alive"
    return response

# Debug routes
@app.route('/debug/status')
@login_required
def debug_status():
    """Show debug status"""
    try:
        user_id = current_user.id
        doc_count = Document.query.filter_by(user_id=user_id).count()
        
        return jsonify({
            'user_id': user_id,
            'documents_processed': doc_count,
            'database_status': 'connected',
            'system_status': 'simplified - no GPU dependencies'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/debug/simple_file_check')
@login_required
def simple_file_check():
    """Very simple file existence check"""
    try:
        user_id = current_user.id
        base_extract_dir = app.config['EXTRACT_FOLDER']
        user_extract_dir = os.path.join(base_extract_dir, str(user_id))
        
        result = {
            'user_id': user_id,
            'base_extract_dir': base_extract_dir,
            'user_extract_dir': user_extract_dir,
            'base_dir_exists': os.path.exists(base_extract_dir),
            'user_dir_exists': os.path.exists(user_extract_dir),
            'files': []
        }
        
        # List all files recursively
        if os.path.exists(user_extract_dir):
            for root, dirs, files in os.walk(user_extract_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    result['files'].append({
                        'name': file,
                        'path': full_path,
                        'size': os.path.getsize(full_path),
                        'extension': os.path.splitext(file)[1].lower(),
                        'allowed': allowed_file(file)
                    })
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/debug/test_single_file')
@login_required
def test_single_file():
    """Test reading a single file"""
    file_path = request.args.get('path', '')
    
    if not file_path:
        return jsonify({'error': 'No file path provided. Use ?path=/full/path/to/file'})
    
    try:
        processor = SimpleDocumentProcessor(current_user.id)
        
        result = {
            'file_path': file_path,
            'exists': os.path.exists(file_path),
            'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            'extension': os.path.splitext(file_path)[1].lower(),
            'allowed': allowed_file(os.path.basename(file_path))
        }
        
        if result['exists'] and result['size'] > 0:
            # Try to extract text
            content = processor.extract_text(file_path)
            result['extraction_success'] = content is not None
            result['content_length'] = len(content) if content else 0
            result['content_preview'] = content[:500] if content else None
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/debug/force_test_processing')
@login_required
def force_test_processing():
    """Force test the processing with a hardcoded file"""
    try:
        # Create a test file directly
        user_id = current_user.id
        user_extract_dir = os.path.join(app.config['EXTRACT_FOLDER'], str(user_id))
        os.makedirs(user_extract_dir, exist_ok=True)
        
        test_file_path = os.path.join(user_extract_dir, 'force_test.txt')
        
        # Write test content
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write("""This is a test document for debugging purposes.
            
It contains multiple paragraphs to test the document processing system.

The system should be able to extract this text and process it properly.

This will help us identify if the issue is with file upload or text processing.""")
        
        # Test the processing
        with app.app_context():
            processor = SimpleDocumentProcessor(user_id=user_id)
            processed_count = processor.process_files([test_file_path])
        
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
        
        return jsonify({
            'success': True,
            'test_file_path': test_file_path,
            'processed_count': processed_count,
            'message': 'Test completed successfully' if processed_count > 0 else 'Test failed'
        })
        
    except Exception as e:
        logger.error(f"Force test failed: {e}")
        return jsonify({'error': str(e)})

@app.route('/debug/test_db')
def test_db():
    """Test database connection"""
    try:
        with db.engine.connect() as connection:
            result = connection.execute(text('SELECT 1'))
            return jsonify({"status": "success", "result": "Database connected"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})

@app.route('/debug/file_paths/<int:user_id>')
@login_required
def debug_file_paths(user_id):
    """Debug endpoint to check file paths for a user"""
    if not current_user.is_admin and current_user.id != user_id:
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        debug_info = {
            'user_id': user_id,
            'extract_folder': app.config['EXTRACT_FOLDER'],
            'files_on_disk': [],
            'documents_in_db': []
        }
        
        # Check files on disk
        user_extract_dir = os.path.join(app.config['EXTRACT_FOLDER'], str(user_id))
        if os.path.exists(user_extract_dir):
            for root, dirs, files in os.walk(user_extract_dir):
                for file in files:
                    if not file.startswith('.'):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, app.config['EXTRACT_FOLDER'])
                        debug_info['files_on_disk'].append({
                            'filename': file,
                            'full_path': full_path,
                            'relative_path': rel_path,
                            'exists': os.path.exists(full_path),
                            'size': os.path.getsize(full_path) if os.path.exists(full_path) else 0
                        })
        
        # Check documents in database
        documents = Document.query.filter_by(user_id=user_id).all()
        for doc in documents:
            filename = doc.document_metadata.get('original_filename', 'Unknown') if doc.document_metadata else 'Unknown'
            content_record = DocumentContent.query.filter_by(document_id=doc.id).first()
            
            debug_info['documents_in_db'].append({
                'id': doc.id,
                'filename': filename,
                'file_path': doc.file_path,
                'file_hash': doc.file_hash,
                'content_length': len(content_record.content) if content_record else 0,
                'file_size': doc.document_metadata.get('file_size', 0) if doc.document_metadata else 0
            })
        
        return jsonify(debug_info)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/debug/clear_user_data')
@login_required
def clear_user_data():
    """Clear all user data for testing"""
    try:
        user_id = current_user.id
        
        # Clear database
        DocumentContent.query.join(Document).filter(Document.user_id == user_id).delete(synchronize_session=False)
        Document.query.filter_by(user_id=user_id).delete()
        Conversation.query.filter_by(user_id=user_id).delete()
        db.session.commit()
        
        # Clear files
        user_extract_dir = os.path.join(app.config['EXTRACT_FOLDER'], str(user_id))
        if os.path.exists(user_extract_dir):
            shutil.rmtree(user_extract_dir)
        
        user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
        if os.path.exists(user_upload_dir):
            shutil.rmtree(user_upload_dir)
        
        return jsonify({
            'success': True,
            'message': 'All user data cleared successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

# Initialize database
def create_app_with_postgres():
    """Initialize app with PostgreSQL"""
    with app.app_context():
        try:
            with db.engine.connect() as connection:
                connection.execute(text('SELECT 1'))
            logger.info('PostgreSQL connection successful')
            
            # Create all tables (including new DocumentContent table)
            db.create_all()
            logger.info('Database tables created/verified')
            
            # Check if we need to migrate existing documents
            try:
                # Check if DocumentContent table exists and has data
                content_count = DocumentContent.query.count()
                doc_count = Document.query.count()
                
                if doc_count > 0 and content_count == 0:
                    logger.info("Found documents without content records - migration may be needed")
                    # Note: Manual migration would be needed for existing documents
                    
            except Exception as e:
                logger.info(f"DocumentContent table check: {e}")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

if __name__ == '__main__':
    try:
        create_app_with_postgres()
        logger.info('Starting DocumentIQ - Simplified Version')
        app.run(host='0.0.0.0', port=8000, threaded=True, debug=False)
    except Exception as e:
        logger.critical(f"Failed to start application: {e}")
        raise