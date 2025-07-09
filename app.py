import os
import zipfile
import PyPDF2
import docx
import shutil
import requests
import uuid
import json
import time
from datetime import datetime, timezone
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, stream_with_context
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

import stripe
from datetime import timedelta
from sqlalchemy import func
from flask_migrate import Migrate

from flask_wtf.csrf import CSRFProtect

# Load environment variables
load_dotenv()



# Initialize Flask app
app = Flask(__name__)
csrf = CSRFProtect(app)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///documentiq.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')

# Fatora configuration
app.config['FATORA_API_KEY'] = os.getenv('FATORA_API_KEY')
app.config['FATORA_BASE_URL'] = 'https://api.fatora.io/v1'
app.config['FATORA_WEBHOOK_SECRET'] = os.getenv('FATORA_WEBHOOK_SECRET')

class SubscriptionPlan:
    BASIC = {
        'id': 'basic',
        'name': 'Basic',
        'price': 99,  # QAR
        'features': [
            '50 documents/month',
            '10MB max file size',
            'Basic support'
        ],
        'limits': {
            'max_documents': 50,
            'max_file_size': 10 * 1024 * 1024,
            'max_conversations': 10
        }
    }
    
    PRO = {
        'id': 'pro',
        'name': 'Professional',
        'price': 199,
        'features': [
            '200 documents/month',
            '50MB max file size',
            'Priority support',
            'Advanced analytics'
        ],
        'limits': {
            'max_documents': 200,
            'max_file_size': 50 * 1024 * 1024,
            'max_conversations': 50
        }
    }
    
    TRIAL = {
        'id': 'trial',
        'name': 'Free Trial',
        'price': 0,
        'features': [
            'Unlimited documents for 14 days',
            '100MB max file size',
            'All Professional features',
            'No credit card required'
        ],
        'limits': {
            'max_documents': None,  # Unlimited
            'max_file_size': 100 * 1024 * 1024,
            'max_conversations': None,
            'trial_days': 14
        }
    }

    @classmethod
    def get_plan(cls, plan_id):
        plans = {
            'basic': cls.BASIC,
            'pro': cls.PRO,
            'trial': cls.TRIAL
        }
        return plans.get(plan_id)
    
# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'


# Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
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
    
    @property
    def is_subscribed(self):
    # Check active paid subscription
        if self.subscription_status == 'active' and (
            self.current_period_end is None or 
            self.current_period_end > datetime.utcnow()
        ):
            return True
        
        # Check trial period
        if self.subscription_plan == 'trial' and self.trial_end_date and self.trial_end_date > datetime.utcnow():
            return True
        
        return False
    
    @property
    def plan_details(self):
        return SubscriptionPlan.get_plan(self.subscription_plan) or {}
    
    def can_upload_file(self, file_size):
        if self.is_admin:
            return True
        
        # Allow unlimited uploads during trial period
        if self.subscription_plan == 'trial' and self.is_subscribed:
            return True
        
        plan = self.plan_details.get('limits', {})
    
        # Check file size limit
        max_size = plan.get('max_file_size', 5 * 1024 * 1024)  # Default 5MB
        if file_size > max_size:
            return False
        
        # Check document count if limited
        if plan.get('max_documents') is not None:
            if self.documents_uploaded >= plan['max_documents']:
                # Check if billing cycle has reset
                if self.last_billing_date and self.last_billing_date.month == datetime.utcnow().month:
                    return False
                else:
                    # Reset counter for new month
                    self.documents_uploaded = 0
                    db.session.commit()
                
        return True
    
class Payment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    currency = db.Column(db.String(3), default='QAR')
    payment_intent_id = db.Column(db.String(100))
    payment_method = db.Column(db.String(50))
    status = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    receipt_url = db.Column(db.String(255))

class BillingHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    description = db.Column(db.String(255))
    amount = db.Column(db.Float, nullable=False)
    currency = db.Column(db.String(3), default='QAR')
    status = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    invoice_id = db.Column(db.String(100))

class Conversation(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(100), nullable=False, default='New Query')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    messages = db.relationship('Message', backref='conversation', lazy=True, cascade='all, delete-orphan')

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.String(36), db.ForeignKey('conversation.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    sources = db.Column(db.JSON)  # Store as JSON array
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class FileCache(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_hash = db.Column(db.String(32), unique=True, nullable=False)
    content = db.Column(db.Text, nullable=False)
    file_metadata = db.Column(db.JSON, nullable=False)  # Renamed here
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)


migrate = Migrate(app, db)

# Initialize database
with app.app_context():
    db.create_all()


# User loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Validate critical environment variables
def check_environment():
    """Validate required environment variables before starting"""
    required_vars = ['OLLAMA_URL', 'OLLAMA_MODEL']
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

# Global progress tracking with thread safety
progress_data = {
    'current': 0,
    'total': 100,
    'message': '',
    'error': None,
    'redirect_url': None
}
progress_lock = threading.Lock()

# Enhanced Configuration with document processing improvements
class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    UPLOAD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'uploads')
    EXTRACT_FOLDER = os.path.join(os.getcwd(), 'extracted_files')
    CACHE_FOLDER = os.path.join(os.getcwd(), 'file_cache')
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'md', 'csv', 'json', 'xml', 'html'}
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
    OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434/api/generate')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3:70b-instruct-q4_K_M')
    OLLAMA_TIMEOUT = 600  # 10 minutes
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB per file
    MAX_RETRIES = 3
    LOG_FILE = os.path.join(os.getcwd(), 'app.log')
    LOG_LEVEL = logging.INFO
    MAX_CONTEXT_LENGTH = 500000  # Characters per file
    STREAM_TIMEOUT = 500  # Seconds
    CACHE_EXPIRATION_SECONDS = 3600  # 1 hour
    CHUNK_SIZE = 30000  # For text chunking
    CHUNK_OVERLAP = 500  # Overlap between chunks
    PDF_EXTRACTION_METHOD = 'pdfminer'  # Options: 'pdfminer', 'pdfplumber', 'pypdf2'

app.config.from_object(Config)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['EXTRACT_FOLDER'], exist_ok=True)
os.makedirs(app.config['CACHE_FOLDER'], exist_ok=True)


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
logging.getLogger().setLevel(app.config['LOG_LEVEL'])

# In-memory storage for conversations
conversations = {}

@app.route('/debug_file_processing')
@login_required
def debug_file_processing():
    """Debug endpoint to verify file processing"""
    try:
        # Get all files in user's extract directory
        user_extract_dir = os.path.join(app.config['EXTRACT_FOLDER'], str(current_user.id))
        all_files = []
        if os.path.exists(user_extract_dir):
            for root, _, files in os.walk(user_extract_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if not file.startswith('.') and not file == '__MACOSX':
                        all_files.append({
                            'path': file_path,
                            'rel_path': os.path.relpath(file_path, user_extract_dir),
                            'exists': os.path.exists(file_path),
                            'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
                        })
        
        # Try reading first 3 files
        sample_results = []
        for file in all_files[:3]:
            content = read_file(file['path'], current_user.id)
            sample_results.append({
                'file': file['rel_path'],
                'content_length': len(content) if content else 0,
                'content_sample': (content[:100] + '...') if content else None,
                'cache_exists': bool(FileCache.query.filter_by(file_hash=get_file_hash(file['path'])).first()) if os.path.exists(file['path']) else False
            })
        
        return jsonify({
            'extract_dir': user_extract_dir,
            'files_found': len(all_files),
            'sample_results': sample_results,
            'cache_entries': FileCache.query.count()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_zip(zip_path, extract_to):
    """Enhanced ZIP extraction with better error handling"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_infos = [
                f for f in zip_ref.infolist() 
                if not f.is_dir() 
                and '__MACOSX' not in f.filename
                and allowed_file(f.filename)
            ]
            total_files = len(file_infos)
            
            for i, file_info in enumerate(file_infos):
                try:
                    file_path = file_info.filename
                    target_path = os.path.join(extract_to, file_path)
                    
                    if file_info.file_size > app.config['MAX_FILE_SIZE']:
                        app.logger.warning(f"Skipping large file: {file_path}")
                        continue
                    
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    
                    with zip_ref.open(file_info) as source, open(target_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
                    
                    if i % 10 == 0:
                        progress = 30 + int((i / total_files) * 60)
                        update_progress(progress, 100, f"Extracting {os.path.basename(file_path)}...")
                        
                except Exception as e:
                    app.logger.error(f"Failed to extract {file_path}: {str(e)}")
                    continue
                    
        return True
    except zipfile.BadZipFile:
        raise ValueError("Invalid ZIP file format")
    except Exception as e:
        raise ValueError(f"ZIP extraction failed: {str(e)}")

def get_file_hash(file_path):
    """Guaranteed file hashing with fallback"""
    try:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except:
        # Fallback to file stats if hashing fails
        stat = os.stat(file_path)
        return f"{stat.st_size}:{stat.st_mtime}"
    
def read_pdf(file_path):
    """Improved PDF reading with multiple extraction methods and structure preservation"""
    try:
        if app.config['PDF_EXTRACTION_METHOD'] == 'pdfminer':
            text = pdfminer_extract(file_path)
        elif app.config['PDF_EXTRACTION_METHOD'] == 'pdfplumber':
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    # Extract text with layout preservation
                    page_text = page.extract_text(
                        layout=True,
                        x_tolerance=1,
                        y_tolerance=1,
                        keep_blank_chars=True
                    )
                    if page_text:
                        text += f"=== PAGE {page.page_number} ===\n{page_text}\n\n"
        else:  # PyPDF2 fallback
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"=== PAGE {i+1} ===\n{page_text}\n\n"
        return text
    except Exception as e:
        app.logger.error(f"PDF extraction failed for {file_path}: {str(e)}")
        return None

def read_docx(file_path):
    """Enhanced DOCX reading with style information and structure preservation"""
    try:
        doc = docx.Document(file_path)
        text = []
        for para in doc.paragraphs:
            # Preserve heading structure
            if para.style.name.startswith('Heading'):
                level = int(para.style.name.split(' ')[1]) if ' ' in para.style.name else 1
                text.append(f"\n{'#' * level} {para.text}\n")
            else:
                text.append(para.text)
            
            # Preserve tables
            if para.tables:
                for table in para.tables:
                    for row in table.rows:
                        row_text = "| " + " | ".join(cell.text for cell in row.cells) + " |"
                        text.append(row_text)
                    text.append("")  # Add empty line after table
                    
        return "\n".join(text)
    except Exception as e:
        app.logger.error(f"DOCX extraction failed for {file_path}: {str(e)}")
        return None

def read_markdown(file_path):
    """Read markdown with original structure preservation"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        app.logger.error(f"Markdown extraction failed for {file_path}: {str(e)}")
        return None

def read_csv(file_path):
    """Read CSV with header detection and structure preservation"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            return "\n".join([", ".join(row) for row in reader])
    except Exception as e:
        app.logger.error(f"CSV extraction failed for {file_path}: {str(e)}")
        return None

def read_xml(file_path):
    """Read XML with tag preservation and structure"""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        def xml_to_text(element, indent=0):
            text = []
            # Add opening tag with attributes
            attrs = " ".join(f'{k}="{v}"' for k, v in element.attrib.items())
            tag_text = f"{' ' * indent}<{element.tag}"
            if attrs:
                tag_text += f" {attrs}"
            tag_text += ">"
            text.append(tag_text)
            
            # Handle text content
            if element.text and element.text.strip():
                text.append(f"{' ' * (indent+2)}{element.text.strip()}")
            
            # Recursively process children
            for child in element:
                text.append(xml_to_text(child, indent+2))
            
            # Add closing tag
            text.append(f"{' ' * indent}</{element.tag}>")
            return "\n".join(text)
        
        return xml_to_text(root)
    except Exception as e:
        app.logger.error(f"XML extraction failed for {file_path}: {str(e)}")
        return None

def read_html(file_path):
    """Read HTML content with structure preservation"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            
            # Preserve basic structure
            for elem in soup.find_all(['div', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                if elem.name.startswith('h'):
                    level = int(elem.name[1])
                    elem.insert_before(f"\n{'#' * level} ")
                    elem.insert_after("\n")
                elif elem.name == 'p':
                    elem.insert_before("\n")
                    elem.insert_after("\n")
                elif elem.name == 'div':
                    elem.insert_before("\n---\n")
                    elem.insert_after("\n---\n")
            
            # Get text with preserved structure
            return soup.get_text('\n')
    except Exception as e:
        app.logger.error(f"HTML extraction failed for {file_path}: {str(e)}")
        return None

# Helper function to read files with user-specific caching
def read_file(file_path, user_id=None):
    """Completely rewritten file reading with guaranteed caching"""
    if not os.path.exists(file_path):
        app.logger.error(f"File not found: {file_path}")
        return None

    try:
        # Generate reliable file hash
        file_hash = get_file_hash(file_path)
        if not file_hash:
            return None

        # DEBUG: Log cache check
        app.logger.debug(f"Checking cache for {file_path} (hash: {file_hash}, user: {user_id})")

        # Check cache with direct database verification
        cached = FileCache.query.filter_by(file_hash=file_hash).first()
        if cached:
            app.logger.debug(f"Cache HIT for {file_path}")
            return cached.content

        app.logger.debug(f"Cache MISS for {file_path}")

        # Read file content based on type
        text = None
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            text = read_pdf(file_path)
        elif ext == '.docx':
            text = read_docx(file_path)
        elif ext in ('.md', '.markdown'):
            text = read_markdown(file_path)
        elif ext == '.csv':
            text = read_csv(file_path)
        elif ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = json.dumps(json.load(f), indent=2)
        elif ext == '.xml':
            text = read_xml(file_path)
        elif ext == '.html':
            text = read_html(file_path)
        elif ext == '.txt':
            text = read_text_file(file_path)

        if not text or not text.strip():
            return None

        # DEBUG: Before caching
        app.logger.debug(f"Attempting to cache {file_path} (size: {len(text)} chars)")

        # Create new cache entry with transaction safety
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
            
            # Verify commit
            db.session.refresh(new_cache)
            if not new_cache.id:
                raise Exception("Cache entry not persisted")
                
            app.logger.debug(f"Successfully cached {file_path} with ID {new_cache.id}")
            
        except Exception as e:
            app.logger.error(f"Failed to cache {file_path}: {str(e)}")
            db.session.rollback()
            # Even if caching fails, still return the text
            return text

        return text

    except Exception as e:
        app.logger.error(f"Fatal error in read_file: {str(e)}")
        return None
    
def get_file_hash_with_verification(file_path):
    """Generate file hash with verification"""
    try:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            while chunk := f.read(65536):  # Read in 64k chunks
                hasher.update(chunk)
        file_hash = hasher.hexdigest()
        
        # Verify hash wasn't empty
        if not file_hash or len(file_hash) != 32:
            raise ValueError(f"Invalid hash generated: {file_hash}")
            
        return file_hash
    except Exception as e:
        app.logger.error(f"Hash generation failed for {file_path}: {str(e)}")
        return None

def get_cached_content(file_hash, user_id):
    """Retrieve cached content with verification"""
    try:
        # Check both user-specific and global cache
        cache_query = FileCache.query.filter_by(file_hash=file_hash)
        
        if user_id is not None:
            cache_query = cache_query.filter(
                (FileCache.user_id == user_id) | 
                (FileCache.user_id.is_(None))
            )

        cached_file = cache_query.order_by(FileCache.user_id.desc()).first()
        
        if cached_file:
            # Verify cached content
            if not cached_file.content or not isinstance(cached_file.content, str):
                app.logger.warning(f"Invalid cache content for hash {file_hash}")
                return None
                
            app.logger.debug(f"Cache hit for hash {file_hash}")
            return cached_file.content
            
        return None
    except Exception as e:
        app.logger.error(f"Cache lookup failed: {str(e)}")
        return None

def cache_file_content(file_hash, text, file_path, file_ext, user_id):
    """Store file content in cache with verification"""
    try:
        # Verify inputs before caching
        if not file_hash or not text or not file_path:
            return False

        # Prepare metadata
        file_metadata = {
            'path': file_path,
            'size': os.path.getsize(file_path),
            'last_modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
            'type': file_ext[1:] if file_ext else 'unknown',
            'extraction_method': app.config['PDF_EXTRACTION_METHOD'] if file_ext == '.pdf' else 'standard'
        }

        # Create cache entry
        cache_entry = FileCache(
            file_hash=file_hash,
            content=text,
            file_metadata=file_metadata,
            user_id=user_id
        )

        # Verify database connection
        if not db.session.is_active:
            db.session.rollback()
            db.session.begin()

        db.session.add(cache_entry)
        db.session.commit()

        # Verify the record was actually saved
        saved_entry = FileCache.query.filter_by(file_hash=file_hash).first()
        if not saved_entry:
            raise ValueError("Cache entry not persisted to database")

        app.logger.debug(f"Successfully cached {file_path}")
        return True

    except Exception as e:
        app.logger.error(f"Failed to cache file {file_path}: {str(e)}")
        db.session.rollback()
        return False

def read_file_by_type(file_path, file_ext):
    """Read file based on its type"""
    try:
        if file_ext == '.pdf':
            return read_pdf(file_path)
        elif file_ext == '.docx':
            return read_docx(file_path)
        elif file_ext in ('.md', '.markdown'):
            return read_markdown(file_path)
        elif file_ext == '.csv':
            return read_csv(file_path)
        elif file_ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.dumps(json.load(f), indent=2)
        elif file_ext == '.xml':
            return read_xml(file_path)
        elif file_ext == '.html':
            return read_html(file_path)
        elif file_ext == '.txt':
            encodings = ['utf-8', 'iso-8859-1', 'windows-1252']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
        return None
    except Exception as e:
        app.logger.error(f"Error reading {file_path}: {str(e)}")
        return None
    

@app.route('/verify_cache')
@login_required
def verify_cache():
    """Route to manually verify caching functionality"""
    try:
        # Create a test file
        test_content = f"Test content at {datetime.now()}"
        test_file = os.path.join(app.config['UPLOAD_FOLDER'], 'cache_test.txt')
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Read file (should trigger caching)
        content = read_file(test_file, current_user.id)
        
        # Check cache directly
        file_hash = get_file_hash(test_file)
        cached_entry = FileCache.query.filter_by(file_hash=file_hash).first()
        
        # Check database connection
        db_status = "OK" if db.session.execute("SELECT 1").scalar() == 1 else "FAILED"
        
        return jsonify({
            'test_file': test_file,
            'file_hash': file_hash,
            'content_matches': content == test_content,
            'cache_exists': bool(cached_entry),
            'cache_content_matches': cached_entry.content == test_content if cached_entry else False,
            'db_connection': db_status,
            'table_exists': 'file_cache' in [t.name for t in db.inspect(db.engine).get_table_names()],
            'cache_entries': FileCache.query.count()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
def chunk_text(text, chunk_size=None, overlap=None):
    """Intelligent chunking that preserves document structure"""
    chunk_size = chunk_size or app.config['CHUNK_SIZE']
    overlap = overlap or app.config['CHUNK_OVERLAP']
    
    # Split by major document sections first
    sections = re.split(r'(\[\w+_SECTION:.+?\])', text)
    sections = [s for s in sections if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for section in sections:
        # If section is a marker, always keep it with its content
        if re.match(r'\[\w+_SECTION:.+?\]', section):
            if current_chunk and len(current_chunk) + len(section) > chunk_size:
                chunks.append(current_chunk)
                current_chunk = current_chunk[-overlap:] + "\n" + section
            else:
                current_chunk += "\n" + section
        else:
            # Regular content - split by paragraphs if possible
            paragraphs = re.split(r'(\n\n+)', section)
            for para in paragraphs:
                if len(current_chunk) + len(para) > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = current_chunk[-overlap:] + para
                    else:
                        # Handle very large paragraphs
                        chunks.append(para[:chunk_size])
                        current_chunk = para[chunk_size-overlap:chunk_size]
                else:
                    current_chunk += para
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks
def clean_directory(directory):
    """Safely clean all files in directory"""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            app.logger.error(f'Failed to delete {file_path}. Reason: {e}')

def update_progress(current, total, message="", redirect_url=None):
    """Update progress information with thread safety"""
    with progress_lock:
        progress_data['current'] = current
        progress_data['total'] = total
        progress_data['message'] = message
        if redirect_url:
            progress_data['redirect_url'] = redirect_url
        if current >= total:
            progress_data['error'] = None

def get_answer_stream(text, question):
    """Process all chunks with context accumulation"""
    # First check if we have any meaningful context
    if not text or text.strip() in ('', '.'):
        app.logger.error(f"No content to analyze. Text length: {len(text) if text else 0}")
        yield "I couldn't find any text content in the uploaded documents to analyze."
        return

    try:
        chunks = chunk_text(text)
        if not chunks:
            app.logger.error("No chunks generated from text")
            yield "Error: Could not process document content."
            return

        context_window = []
        max_context_size = app.config['MAX_CONTEXT_LENGTH'] // 2
        
        full_answer = ""
        
        for i, chunk in enumerate(chunks):
            # Maintain a sliding context window
            context_window.append(chunk)
            context_window = context_window[-3:]  # Keep last 3 chunks
            
            # Build context ensuring we don't exceed limits
            context = "\n\n".join(context_window)[-max_context_size:]
            
            prompt = build_prompt(context, question, is_final=i == len(chunks)-1)
            
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
                            response_text = data['response']
                            
                            # Skip if we get just a dot
                            if response_text.strip() == '.':
                                continue
                                
                            full_answer += response_text
                            yield response_text
                                
            except Exception as e:
                app.logger.error(f"Error processing chunk {i}: {str(e)}")
                yield f"Error processing content: {str(e)}"
                break
        
        # Final check to ensure we don't return empty
        if not full_answer.strip():
            app.logger.error("Empty response generated despite having content")
            yield "I couldn't generate a response based on the provided documents."

    except Exception as e:
        app.logger.error(f"Error in get_answer_stream: {str(e)}")
        yield "An error occurred while processing your request."


def build_prompt(context, question, is_final=False):
    """Construct a prompt that considers multi-chunk context"""
    return (
        "Document Context (may be partial):\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Instructions:\n"
        "1. Answer based on the current context\n"
        # "3. Mark complete answers with [ANSWER_COMPLETE]\n"
        "4. Preserve original document structure in responses\n"
        "5. Every response that includes a section or quote must end with a citation in the format: [^filename||page||section-heading]\n"
        "   - filename must be the exact relative path from the extracted_files directory\n"
        "   - Use forward slashes (/) for path separators\n"
        "6. If no matching section is found, output a single dot (.) as the result: '.'\n"
        "7. **Never**:\n"
        "   - Paraphrase or summarize\n"
        "   - Add interpretation\n"
        "   - Combine text from different sections\n"
        "   - Modify legal terms or definitions\n\n"
        "   - Mention that something was not found or unrelated\n"
        f"{'8. This is the final context section' if is_final else ''}\n\n"
        "Response:"
    )

@app.template_filter('datetimeformat')
def datetimeformat(value, format='%b %d, %H:%M'):
    """Format datetime for templates"""
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.strftime(format)


# Add before first route
@app.before_request
def check_subscription():
    if current_user.is_authenticated and not current_user.is_admin:
        # Reset document count if new billing month
        if current_user.last_billing_date and current_user.last_billing_date.month < datetime.utcnow().month:
            current_user.documents_uploaded = 0
            db.session.commit()

@app.route('/check_subscription')
@login_required
def check_subscription():
    return jsonify({
        'has_subscription': current_user.is_subscribed,
        'has_active_trial': current_user.subscription_plan == 'trial' and current_user.is_subscribed,
        'has_trial_available': current_user.subscription_plan != 'trial' or not current_user.is_subscribed,
        'trial_days_remaining': (current_user.trial_end_date - datetime.utcnow()).days if current_user.trial_end_date else 0,
        'allow_uploads': current_user.is_subscribed  # This will be true for both paid and trial subscriptions
    })

@app.route('/start_trial', methods=['POST'])
@login_required
def start_trial():
    if current_user.subscription_plan == 'trial' and current_user.trial_end_date > datetime.utcnow():
        return jsonify({'error': 'You already have an active trial'}), 400
        
    current_user.subscription_plan = 'trial'
    current_user.subscription_status = 'active'
    current_user.trial_start_date = datetime.utcnow()
    current_user.trial_end_date = datetime.utcnow() + timedelta(days=14)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'Your 14-day free trial has started!',
        'trial_end_date': current_user.trial_end_date.isoformat()
    })

@app.before_request
def check_trial():
    if current_user.is_authenticated and not current_user.is_admin:
        # Check if trial is ending soon (3 days or less)
        if (current_user.subscription_plan == 'trial' and 
            current_user.trial_end_date and 
            (current_user.trial_end_date - datetime.utcnow()).days <= 3 and
            not request.path.startswith('/static') and
            not request.path.startswith('/pricing')):
            
            days_left = (current_user.trial_end_date - datetime.utcnow()).days
            flash(f'Your free trial ends in {days_left} day(s). Upgrade now to continue uninterrupted service.', 'warning')


@app.route('/pricing')
@login_required
def pricing():
    plans = {
        'basic': SubscriptionPlan.BASIC,
        'pro': SubscriptionPlan.PRO,
        'enterprise': SubscriptionPlan.TRIAL
    }
    return render_template('pricing.html', plans=plans, now=datetime.utcnow())

@app.context_processor
def inject_now():
    return {'now': datetime.utcnow()}

@app.route('/create_subscription', methods=['POST'])
@login_required
def create_subscription():
    plan_id = request.form.get('plan_id')
    payment_method_id = request.form.get('payment_method_id')
    
    if not plan_id or not payment_method_id:
        return jsonify({'error': 'Missing plan or payment method'}), 400
        
    plan = SubscriptionPlan.get_plan(plan_id)
    if not plan:
        return jsonify({'error': 'Invalid plan'}), 400
    
    try:
        # Fatora API integration
        headers = {
            'Authorization': f'Bearer {app.config["FATORA_API_KEY"]}',
            'Content-Type': 'application/json'
        }
        
        # Create customer if not exists
        if not current_user.subscription_id:
            customer_data = {
                'name': current_user.username,
                'email': current_user.email,
                'phone': '',  # Can collect during signup
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
        current_user.current_period_end = datetime.utcnow() + timedelta(days=30)
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
        app.logger.error(f'Subscription error: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/cancel_subscription', methods=['POST'])
@login_required
def cancel_subscription():
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
        app.logger.error(f'Cancel subscription error: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/update_payment_method', methods=['POST'])
@login_required
def update_payment_method():
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
        app.logger.error(f'Update payment method error: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/billing')
@login_required
def billing():
    payments = Payment.query.filter_by(user_id=current_user.id).order_by(Payment.created_at.desc()).limit(10).all()
    invoices = BillingHistory.query.filter_by(user_id=current_user.id).order_by(BillingHistory.created_at.desc()).limit(10).all()
    
    return render_template('billing.html', 
                         payments=payments,
                         invoices=invoices,
                         plan=current_user.plan_details,
                         now=datetime.utcnow())


@app.route('/fatora_webhook', methods=['POST'])
def fatora_webhook():
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
        app.logger.error(f'Webhook error: {str(e)}')
        return jsonify({'error': str(e)}), 500

def handle_payment_succeeded(data):
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
    invoice = data['object']
    user = User.query.filter_by(subscription_id=invoice['customer']).first()
    
    if user:
        # Update subscription period
        user.current_period_end = datetime.fromtimestamp(invoice['period_end'])
        user.last_billing_date = datetime.utcnow()
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
    subscription = data['object']
    user = User.query.filter_by(subscription_id=subscription['customer']).first()
    
    if user:
        user.subscription_status = 'canceled'
        user.current_period_end = None
        db.session.commit()



# Authentication routes
@app.route('/register', methods=['GET', 'POST'])
def register():
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
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember') == 'on'
        
        user = User.query.filter_by(username=username).first()
        
        if not user or not user.check_password(password):
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))
        
        login_user(user, remember=remember)
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        flash('Logged in successfully!', 'success')
        return redirect(url_for('index'))
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))



@app.route('/new_chat', methods=['GET', 'POST'])
@login_required
def new_chat():
    app.logger.info('Received request to create new query')
    try:
        chat_id = str(uuid.uuid4())
        
        # Create new conversation in database
        conversation = Conversation(
            id=chat_id,
            user_id=current_user.id,
            title='New Query'
        )
        db.session.add(conversation)
        
        # Get uploaded files information
        uploaded_files = []
        user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
        if os.path.exists(user_upload_dir):
            for root, _, files in os.walk(user_upload_dir):
                for file in files:
                    if not file.startswith('.') and not file == '__MACOSX':
                        uploaded_files.append(file)
        
        # Create initial message
        initial_message = "Files uploaded successfully. 'Type your search query' in the Query bar below."
        if uploaded_files:
            file_list = "\n".join(f"- {file}" for file in uploaded_files[:5])
            if len(uploaded_files) > 5:
                file_list += f"\n- ...and {len(uploaded_files) - 5} more files"
            initial_message = f"{file_list}\n\nUploaded. 'Type your search query' in the Query bar below."
        
        # Add initial message to database
        initial_msg = Message(
            conversation_id=chat_id,
            role='assistant',
            content=initial_message,
            timestamp=datetime.now(timezone.utc)
        )
        db.session.add(initial_msg)
        db.session.commit()
        
        app.logger.info(f'Created new query with ID: {chat_id}')
        return redirect(url_for('chat', chat_id=chat_id))
    except Exception as e:
        app.logger.error(f'Error creating new query: {str(e)}', exc_info=True)
        flash('Error creating new query session', 'error')
        return redirect(url_for('index'))


@csrf.exempt
@app.route('/chat/<chat_id>', methods=['GET', 'POST'])
@login_required
def chat(chat_id):
    # Get the current conversation or return 404 if not found or doesn't belong to user
    conversation = Conversation.query.filter_by(id=chat_id, user_id=current_user.id).first_or_404()
    
    # Get all conversations for this user for the sidebar
    user_conversations = Conversation.query.filter_by(user_id=current_user.id)\
        .order_by(Conversation.updated_at.desc()).all()
    
    # Convert to dictionary format for template compatibility
    chats = {
        conv.id: {
            'title': conv.title,
            'created_at': conv.created_at
        }
        for conv in user_conversations
    }
    
    if request.method == 'POST':
        def generate_response():
            try:
                # Get the question from request
                if request.is_json:
                    data = request.get_json()
                    question = data.get('question', '').strip()
                else:
                    question = request.form.get('question', '').strip()

                if not question:
                    yield json.dumps({'status': 'error', 'message': 'Please enter a question'})
                    return

                # Save user message to database
                user_msg = Message(
                    conversation_id=chat_id,
                    role='user',
                    content=question,
                    timestamp=datetime.now(timezone.utc)
                )
                db.session.add(user_msg)
                db.session.commit()

                # Initial status update
                yield json.dumps({
                    'status': 'status_update',
                    'message': '🔍 Preparing to process files...',
                    'progress': 0
                }) + "\n"

                # Process files from user's directory
                user_extract_dir = os.path.join(app.config['EXTRACT_FOLDER'], str(current_user.id))
                all_files = []
                if os.path.exists(user_extract_dir):
                    for root, _, files in os.walk(user_extract_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            if not file.startswith('.') and not file == '__MACOSX':
                                all_files.append(file_path)
                
                total_files = len(all_files)
                processed_files = 0
                file_contents = []
                
                yield json.dumps({
                    'status': 'status_update',
                    'message': f'📂 Found {total_files} files to process...',
                    'progress': 5
                }) + "\n"

                # Process each file
                for i, file_path in enumerate(all_files):
                    try:
                        text = read_file(file_path, current_user.id)
                        if text:
                            file_contents.append({
                                'filename': os.path.basename(file_path),
                                'path': os.path.relpath(file_path, user_extract_dir),
                                'text': text
                            })
                        
                        processed_files += 1
                        progress = 5 + int((i / total_files) * 25)
                        
                        # Update progress
                        if i % 5 == 0 or i == total_files - 1:
                            yield json.dumps({
                                'status': 'status_update',
                                'message': f'📄 Processing {os.path.basename(file_path)} ({processed_files}/{total_files})',
                                'progress': progress
                            }) + "\n"
                            
                    except Exception as e:
                        app.logger.error(f"Error processing {file_path}: {str(e)}")
                        continue

                # Combine context
                combined_context = "\n\n".join(
                    f"[FILE: {file['path']}]\n{file['text'][:app.config['MAX_CONTEXT_LENGTH']]}"
                    for file in file_contents
                ) if file_contents else "No files available for context"

                yield json.dumps({
                    'status': 'status_update',
                    'message': '🧠 Analyzing content...',
                    'progress': 60
                }) + "\n"

                # Stream AI response
                full_answer = ""
                start_time = time.time()
                yield json.dumps({'status': 'stream_start'}) + "\n"
                
                for chunk in get_answer_stream(combined_context, question):
                    if time.time() - start_time > app.config['STREAM_TIMEOUT']:
                        raise TimeoutError("Response generation timed out")
                        
                    yield json.dumps({'status': 'stream_chunk', 'content': chunk}) + "\n"
                    full_answer += chunk
                    
                    # Update progress during streaming
                    current_progress = 60 + int((len(full_answer) / 10000) * 30)
                    if current_progress > 90:
                        current_progress = 90
                    
                    if int(time.time() - start_time) % 2 == 0:
                        yield json.dumps({
                            'status': 'status_update',
                            'message': '✍️ Generating response...',
                            'progress': current_progress
                        }) + "\n"

                # Final updates
                yield json.dumps({
                    'status': 'status_update',
                    'message': '✅ Processing complete',
                    'progress': 100
                }) + "\n"

                yield json.dumps({
                    'status': 'stream_end',
                    'sources': [f['path'] for f in file_contents],
                    'conversation_id': chat_id
                }) + "\n"

                # Save assistant reply to database
                assistant_msg = Message(
                    conversation_id=chat_id,
                    role='assistant',
                    content=full_answer,
                    sources=[f['path'] for f in file_contents],
                    timestamp=datetime.now(timezone.utc)
                )
                db.session.add(assistant_msg)
                
                # Update conversation title if it's still the default
                if conversation.title == 'New Query':
                    new_title = question[:50] + ("..." if len(question) > 50 else "")
                    conversation.title = new_title
                
                db.session.commit()

            except TimeoutError:
                error_msg = "⏱️ Error: Response generation timed out"
                yield json.dumps({
                    'status': 'status_update',
                    'message': error_msg,
                    'progress': 100,
                    'error': True
                }) + "\n"
                yield json.dumps({'status': 'error', 'message': error_msg}) + "\n"
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                yield json.dumps({
                    'status': 'status_update',
                    'message': error_msg,
                    'progress': 100,
                    'error': True
                }) + "\n"
                yield json.dumps({'status': 'error', 'message': str(e)}) + "\n"

        return Response(stream_with_context(generate_response()), mimetype='application/x-ndjson')

    # For GET requests
    messages = Message.query.filter_by(conversation_id=chat_id).order_by(Message.timestamp).all()
    
    # Get uploaded files for this user
    uploaded_files = []
    user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
    if os.path.exists(user_upload_dir):
        for root, _, files in os.walk(user_upload_dir):
            for file in files:
                if not file.startswith('.') and not file == '__MACOSX':
                    uploaded_files.append(file)
    
    return render_template('chat.html',
                         chat_id=chat_id,
                         chats=chats,
                         conversation={
                             'title': conversation.title,
                             'created_at': conversation.created_at,
                             'messages': [{
                                 'role': msg.role,
                                 'content': msg.content,
                                 'sources': msg.sources,
                                 'timestamp': msg.timestamp
                             } for msg in messages]
                         },
                         uploaded_files=uploaded_files)


@app.route('/delete_chat/<chat_id>', methods=['POST'])
@login_required
def delete_chat(chat_id):
    """Delete a chat session"""
    app.logger.info(f'Received request to delete chat ID: {chat_id}')
    try:
        conversation = Conversation.query.filter_by(id=chat_id, user_id=current_user.id).first()
        if not conversation:
            app.logger.warning(f'Chat not found for deletion: {chat_id}')
            return jsonify({'success': False, 'error': 'Chat not found'}), 404
            
        db.session.delete(conversation)
        db.session.commit()
        app.logger.info(f'Successfully deleted chat ID: {chat_id}')
        return jsonify({'success': True}), 200
        
    except Exception as e:
        app.logger.error(f'Error deleting chat {chat_id}: {str(e)}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if current_user.is_authenticated and not current_user.is_subscribed and not current_user.is_admin:
            return jsonify({'error': 'Subscription required'}), 403
            
        try:
            # Create user-specific directories
            user_id = str(current_user.id) if current_user.is_authenticated else 'anonymous'
            user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
            user_extract_dir = os.path.join(app.config['EXTRACT_FOLDER'], user_id)
            
            os.makedirs(user_upload_dir, exist_ok=True)
            os.makedirs(user_extract_dir, exist_ok=True)
            
            # Clean directories
            clean_directory(user_upload_dir)
            clean_directory(user_extract_dir)
            
        except Exception as e:
            app.logger.error(f'Cleanup error: {str(e)}')
            return jsonify({'error': 'Failed to prepare directories'}), 500

        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
            
        files = request.files.getlist('files')
        saved_files = []
        
        for file in files:
            if file.filename == '':
                continue
                
            filename = secure_filename(file.filename)
            is_zip = filename.lower().endswith('.zip')
            
            try:
                if is_zip:
                    # Save ZIP to user's upload directory
                    zip_path = os.path.join(user_upload_dir, filename)
                    file.save(zip_path)
                    
                    # Extract to user's extract directory
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            for file_info in zip_ref.infolist():
                                if file_info.is_dir() or '__MACOSX' in file_info.filename:
                                    continue
                                    
                                if not allowed_file(file_info.filename):
                                    continue
                                    
                                # Extract file
                                extracted_path = os.path.join(user_extract_dir, file_info.filename)
                                os.makedirs(os.path.dirname(extracted_path), exist_ok=True)
                                with zip_ref.open(file_info) as source, open(extracted_path, 'wb') as target:
                                    shutil.copyfileobj(source, target)
                                    
                        saved_files.append(filename)
                    except Exception as e:
                        app.logger.error(f'Failed to extract {filename}: {str(e)}')
                        continue
                        
                else:
                    # Handle regular files
                    if not allowed_file(filename):
                        continue
                        
                    # Save directly to extract directory
                    file_path = os.path.join(user_extract_dir, filename)
                    file.save(file_path)
                    saved_files.append(filename)
                    
            except Exception as e:
                app.logger.error(f'Failed to process {filename}: {str(e)}')
                continue
                
        if not saved_files:
            return jsonify({'error': 'No valid files were uploaded'}), 400
            
        return jsonify({
            'message': f'{len(saved_files)} files processed',
            'files': saved_files,
            'extract_dir': user_extract_dir
        })
    
    return render_template('index.html', chats=conversations)

@app.route('/verify_upload')
@login_required
def verify_upload():
    """Verify upload directory structure"""
    try:
        user_id = str(current_user.id)
        results = {
            'upload_dir': {
                'path': os.path.join(app.config['UPLOAD_FOLDER'], user_id),
                'exists': os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], user_id)),
                'files': []
            },
            'extract_dir': {
                'path': os.path.join(app.config['EXTRACT_FOLDER'], user_id),
                'exists': os.path.exists(os.path.join(app.config['EXTRACT_FOLDER'], user_id)),
                'files': []
            }
        }
        
        if results['upload_dir']['exists']:
            for root, _, files in os.walk(results['upload_dir']['path']):
                for file in files:
                    results['upload_dir']['files'].append({
                        'name': file,
                        'path': os.path.join(root, file),
                        'size': os.path.getsize(os.path.join(root, file))
                    })
                    
        if results['extract_dir']['exists']:
            for root, _, files in os.walk(results['extract_dir']['path']):
                for file in files:
                    results['extract_dir']['files'].append({
                        'name': file,
                        'path': os.path.join(root, file),
                        'size': os.path.getsize(os.path.join(root, file))
                    })
                    
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/update_chat_title/<chat_id>', methods=['POST'])
def update_chat_title(chat_id):
    try:
        data = request.get_json()
        new_title = data.get('title', '')
        
        # Update your database/store here
        # chats[chat_id]['title'] = new_title
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    

@app.route('/get_file_structure')
def get_file_structure():
    """Return the accurate file structure with multi-level hierarchy"""
    base_path = app.config['EXTRACT_FOLDER']
    file_structure = []

    def build_tree(current_path, relative_path=""):
        name = os.path.basename(current_path)
        item = {
            'name': name,
            'path': os.path.join(relative_path, name) if relative_path else name,
            'type': 'directory' if os.path.isdir(current_path) else 'file',
            'extension': os.path.splitext(name)[1].lower() if os.path.isfile(current_path) else None
        }
        
        if os.path.isdir(current_path):
            item['children'] = []
            try:
                for entry in sorted(os.listdir(current_path)):
                    full_path = os.path.join(current_path, entry)
                    if not entry.startswith('.') and not entry == '__MACOSX':
                        child_relative = os.path.join(relative_path, name) if relative_path else name
                        item['children'].append(build_tree(full_path, child_relative))
            except Exception as e:
                app.logger.error(f"Error reading directory {current_path}: {str(e)}")
        return item

    if os.path.exists(base_path):
        for entry in sorted(os.listdir(base_path)):
            full_path = os.path.join(base_path, entry)
            if not entry.startswith('.') and not entry == '__MACOSX':
                file_structure.append(build_tree(full_path))
    
    return jsonify(file_structure)

@app.route('/get_file_content')
def get_file_content():
    """Enhanced file content endpoint with robust path handling"""
    rel_path = request.args.get('path')
    page_num = request.args.get('page', '')
    highlight = request.args.get('highlight', '')
    
    if not rel_path:
        return jsonify({'error': 'File path required'}), 400
    
    try:
        # Security check - prevent directory traversal
        if '..' in rel_path or rel_path.startswith('/'):
            return jsonify({'error': 'Invalid file path'}), 400
        
        # Normalize path (handle different OS path separators)
        rel_path = rel_path.replace('\\', '/').strip('/')  # Convert to Unix-style and remove leading/trailing slashes
        
        # Build the full path safely
        try:
            full_path = os.path.join(app.config['EXTRACT_FOLDER'], *rel_path.split('/'))
            full_path = os.path.normpath(full_path)
        except Exception as e:
            app.logger.error(f"Path joining failed: {rel_path} - {str(e)}")
            return jsonify({'error': 'Invalid file path'}), 400
        
        # Verify the path is within the extract folder
        if not full_path.startswith(os.path.abspath(app.config['EXTRACT_FOLDER'])):
            app.logger.error(f"Security violation attempt: {full_path}")
            return jsonify({'error': 'Access denied'}), 403
        
        # Case-insensitive file search if needed
        if not os.path.exists(full_path):
            dirname, filename = os.path.split(full_path)
            if os.path.exists(dirname):
                for f in os.listdir(dirname):
                    if f.lower() == filename.lower():
                        full_path = os.path.join(dirname, f)
                        break
            
            if not os.path.exists(full_path):
                app.logger.error(f"File not found: {full_path} (resolved from {rel_path})")
                app.logger.error(f"Extract folder contents: {os.listdir(app.config['EXTRACT_FOLDER'])}")
                return jsonify({'error': 'File not found'}), 404
        
        # Get file stats for debugging
        file_stats = os.stat(full_path)
        app.logger.info(f"Serving file: {full_path} (Size: {file_stats.st_size} bytes)")
        
        # Determine content type
        file_ext = os.path.splitext(full_path)[1].lower()
        mime_types = {
            '.pdf': 'application/pdf',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp',
            '.svg': 'image/svg+xml',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.csv': 'text/csv',
            '.json': 'application/json',
            '.xml': 'application/xml',
            '.html': 'text/html'
        }
        
        content_type = mime_types.get(file_ext, 'application/octet-stream')
        
        # Handle different response types
        if content_type.startswith('image/') or file_ext == '.pdf':
            return send_file(
                full_path,
                mimetype=content_type,
                as_attachment=False,
                last_modified=datetime.fromtimestamp(file_stats.st_mtime),
                etag=str(file_stats.st_mtime)
            )
        else:
            # For text files, read and return content
            content = read_file(full_path)
            if not content:
                return jsonify({'error': 'Could not read file content'}), 400
                
            return jsonify({
                'file': os.path.basename(full_path),
                'content': content,
                'path': rel_path,
                'size': file_stats.st_size,
                'last_modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
            })
    
    except Exception as e:
        app.logger.error(f"Error serving file {rel_path}: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
        
           

@app.route('/progress_stream')
def progress_stream():
    """Server-Sent Events for progress updates"""
    def event_stream():
        last_progress = -1
        try:
            while True:
                with progress_lock:
                    current = progress_data['current']
                    total = progress_data['total']
                    message = progress_data['message']
                    error = progress_data['error']
                    redirect_url = progress_data.get('redirect_url')
                    progress = int((current / total) * 100) if total > 0 else 0
                
                if redirect_url or progress != last_progress or error:
                    data = {
                        'progress': progress,
                        'message': message,
                        'error': error,
                        'redirect': redirect_url
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    last_progress = progress
                
                if redirect_url:
                    yield "event: redirect\ndata: {}\n\n"
                    break
                
                if progress >= 100 or error:
                    yield "event: complete\ndata: {}\n\n"
                    break
                
                time.sleep(0.5)
        except GeneratorExit:
            app.logger.info("Client disconnected from progress stream")
        except Exception as e:
            app.logger.error(f"Error in progress stream: {str(e)}")
    
    return Response(event_stream(), mimetype="text/event-stream")

@app.template_filter('process_references')
def process_references(text):
    """Template filter to process references in text"""
    reference_counter = 1
    references = []
    
    def replace_reference(match):
        nonlocal reference_counter
        filename = match.group(1)
        page = match.group(2) or ''
        highlight = match.group(3) or ''
        
        # Clean and normalize the filename path
        filename = filename.split(']')[0]  # Remove any trailing content
        filename = filename.replace('\\', '/')  # Normalize to forward slashes
        
        ref_id = reference_counter
        reference_counter += 1
        
        # Store reference data
        references.append({
            'id': ref_id,
            'filename': filename,
            'page': page,
            'highlight': highlight
        })
        
        # Return the reference link with proper data attributes
        return (f'<sup class="reference-link" '
                f'data-ref="{ref_id}" '
                f'data-filename="{filename}" '
                f'data-page="{page}" '
                f'data-highlight="{highlight}">'
                f'[{ref_id}]</sup>')
    
    # Process all references in the text
    processed_text = re.sub(
        r'\[\^([^\|\]]+)(?:\|\|([^\|\]]+))?(?:\|\|([^\|\]]+))?\]',
        replace_reference,
        text
    )
    
    # Add references section if any references exist
    if references:
        references_section = '<div class="references-section"><h4>References</h4><ol>'
        for ref in sorted(references, key=lambda x: x['id']):
            ref_text = ref['filename'].split('/')[-1]  # Show only filename
            if ref['page']:
                ref_text += f", {ref['page']}"
            if ref['highlight']:
                ref_text += f" ({ref['highlight']})"
            references_section += f'<li id="ref-{ref["id"]}">{ref_text}</li>'
        references_section += '</ol></div>'
        
        processed_text += references_section
    
    return processed_text


@app.after_request
def after_request(response):
    """Add headers to prevent timeouts"""
    response.headers["X-Cloudflare-Time"] = "900"
    response.headers["Connection"] = "keep-alive"
    return response

if __name__ == '__main__':
    try:
        app.logger.info('Starting application on port 8000')
        app.run(
            host='0.0.0.0',
            port=8000,
            threaded=True,
            debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
        )
    except Exception as e:
        app.logger.critical(f"Failed to start application: {str(e)}")
        raise