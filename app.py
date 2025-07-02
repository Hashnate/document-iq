import os
import zipfile
import PyPDF2
import docx
import shutil
import requests
import uuid
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, stream_with_context
import time
from werkzeug.utils import secure_filename
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

# Load environment variables
load_dotenv()

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
    MAX_CONTEXT_LENGTH = 1000000  # Characters per file
    STREAM_TIMEOUT = 500  # Seconds
    CACHE_EXPIRATION_SECONDS = 3600  # 1 hour
    CHUNK_SIZE = 15000  # For text chunking
    CHUNK_OVERLAP = 500  # Overlap between chunks
    PDF_EXTRACTION_METHOD = 'pdfminer'  # Options: 'pdfminer', 'pdfplumber', 'pypdf2'

# Initialize Flask app
app = Flask(__name__)
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
    """Generate a hash for file caching"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def read_pdf(file_path):
    """Improved PDF reading with multiple extraction methods"""
    try:
        if app.config['PDF_EXTRACTION_METHOD'] == 'pdfminer':
            text = pdfminer_extract(file_path)
        elif app.config['PDF_EXTRACTION_METHOD'] == 'pdfplumber':
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        else:  # PyPDF2 fallback
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        return text
    except Exception as e:
        app.logger.error(f"PDF extraction failed for {file_path}: {str(e)}")
        return None

def read_docx(file_path):
    """Enhanced DOCX reading with style information"""
    try:
        doc = docx.Document(file_path)
        text = []
        for para in doc.paragraphs:
            if para.style.name.startswith('Heading'):
                text.append(f"\n\n{para.text.upper()}\n")
            else:
                text.append(para.text)
        return "\n".join(text)
    except Exception as e:
        app.logger.error(f"DOCX extraction failed for {file_path}: {str(e)}")
        return None

def read_markdown(file_path):
    """Read markdown with HTML conversion for better structure"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            md_text = f.read()
            html = markdown.markdown(md_text)
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text()
    except Exception as e:
        app.logger.error(f"Markdown extraction failed for {file_path}: {str(e)}")
        return None

def read_csv(file_path):
    """Read CSV with header detection"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            return "\n".join([", ".join(row) for row in reader])
    except Exception as e:
        app.logger.error(f"CSV extraction failed for {file_path}: {str(e)}")
        return None

def read_xml(file_path):
    """Read XML with tag preservation"""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        return ET.tostring(root, encoding='utf-8', method='text').decode('utf-8')
    except Exception as e:
        app.logger.error(f"XML extraction failed for {file_path}: {str(e)}")
        return None

def read_html(file_path):
    """Read HTML content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            return soup.get_text()
    except Exception as e:
        app.logger.error(f"HTML extraction failed for {file_path}: {str(e)}")
        return None

def read_file(file_path):
    """Improved file reading with caching and better format support"""
    # Check cache first
    file_hash = get_file_hash(file_path)
    cache_path = os.path.join(app.config['CACHE_FOLDER'], f"{file_hash}.txt")
    
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Skip system files
    if '__MACOSX' in file_path or file_path.endswith('._.DS_Store'):
        return None
        
    if os.path.getsize(file_path) > app.config['MAX_FILE_SIZE']:
        app.logger.warning(f"File too large, skipping: {file_path}")
        return None
        
    file_ext = os.path.splitext(file_path)[1].lower()
    text = None
    
    try:
        if file_ext == '.pdf':
            text = read_pdf(file_path)
        elif file_ext == '.docx':
            text = read_docx(file_path)
        elif file_ext in ('.md', '.markdown'):
            text = read_markdown(file_path)
        elif file_ext == '.csv':
            text = read_csv(file_path)
        elif file_ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = json.dumps(json.load(f), indent=2)
        elif file_ext == '.xml':
            text = read_xml(file_path)
        elif file_ext == '.html':
            text = read_html(file_path)
        elif file_ext == '.txt':
            encodings = ['utf-8', 'iso-8859-1', 'windows-1252']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text = f.read(app.config['MAX_CONTEXT_LENGTH'])
                    break
                except UnicodeDecodeError:
                    continue
    
        # Cache the result
        if text:
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(text)
                
    except Exception as e:
        app.logger.error(f"Error reading {file_path}: {str(e)}")
        return None
    
    return text if text and text.strip() else None

def chunk_text(text, chunk_size=None, overlap=None):
    """Split text into manageable chunks with overlap"""
    chunk_size = chunk_size or app.config['CHUNK_SIZE']
    overlap = overlap or app.config['CHUNK_OVERLAP']
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        
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
    """Enhanced streaming with better chunk handling"""
    chunks = chunk_text(text)
    
    prompt = (
        "You are an AI assistant analyzing documents. Use only the provided context.\n\n"
        f"Question: {question}\n\n"
        "Context:\n"
        f"{chunks[0][:500000]}\n\n"  # Only send first chunk initially
        "Instructions:\n"
        "1. Answer using only the provided context\n"
        "2. Use direct quotes with \"quotes\" or > blockquotes\n"
        "3. Cite sources as [^filename||page||highlight]\n"
        "4. If unsure, say 'Not found in context'\n"
        "5. Use markdown formatting (**bold**, *italics*, bullet points, tables when appropriate)\n\n"
        "Answer:"
    )

    try:
        response = requests.post(
            app.config['OLLAMA_URL'],
            json={
                "model": app.config['OLLAMA_MODEL'],
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.3,
                    "num_ctx": 8000,
                    "num_predict": 1000,
                    "format": "markdown"
                }
            },
            stream=True,
            timeout=app.config['OLLAMA_TIMEOUT']
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if 'response' in chunk:
                    yield chunk['response']

    except requests.exceptions.Timeout:
        yield "Error: The request timed out. Please try again with a simpler query."
    except requests.exceptions.RequestException as e:
        yield f"Error: Failed to get streaming response. {str(e)}"

@app.template_filter('datetimeformat')
def datetimeformat(value, format='%b %d, %H:%M'):
    """Format datetime for templates"""
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.strftime(format)

@app.route('/new_chat', methods=['GET', 'POST'])
def new_chat():
    """Create a new query session"""
    app.logger.info('Received request to create new query')
    try:
        chat_id = str(uuid.uuid4())
        conversations[chat_id] = {
            'created_at': datetime.now().isoformat(),
            'title': 'New Query',
            'messages': []
        }
        app.logger.info(f'Created new query with ID: {chat_id}')
        return redirect(url_for('chat', chat_id=chat_id))
    except Exception as e:
        app.logger.error(f'Error creating new query: {str(e)}', exc_info=True)
        flash('Error creating new query session', 'error')
        return redirect(url_for('index'))

@app.route('/chat/<chat_id>', methods=['GET', 'POST'])
def chat(chat_id):
    """Handle both displaying and streaming processing of chat messages"""
    app.logger.info(f'Received request for chat ID: {chat_id}')
    
    if chat_id not in conversations:
        app.logger.warning(f'Chat session not found: {chat_id}')
        if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'status': 'error', 'message': 'Chat session not found'}), 404
        return redirect(url_for('index'))

    if request.method == 'POST':
        def generate_response():
            try:
                # Get the question from request
                if request.is_json:
                    data = request.get_json()
                    question = data.get('question', '').strip()
                    app.logger.debug(f'Received JSON data: {data}')
                else:
                    question = request.form.get('question', '').strip()
                    app.logger.debug(f'Received form data: {request.form}')

                if not question:
                    yield json.dumps({'status': 'error', 'message': 'Please enter a question'})
                    return

                # Save user message
                conversations[chat_id]['messages'].append({
                    'role': 'user',
                    'content': question,
                    'timestamp': datetime.now().isoformat()
                })

                # Check if we have cached file contents and if they're still valid
                use_cached = False
                if 'file_contents' in conversations[chat_id]:
                    last_updated = conversations[chat_id].get('file_contents_updated')
                    if last_updated:
                        cache_age = (datetime.now() - datetime.fromisoformat(last_updated)).total_seconds()
                        if cache_age < app.config['CACHE_EXPIRATION_SECONDS']:
                            use_cached = True
                            app.logger.info('Using valid cached file contents')

                if not use_cached:
                    # Process files only if not already cached or cache expired
                    file_contents = []
                    app.logger.info('Processing files in extraction folder (first time or cache expired)')
                    for root, _, files in os.walk(app.config['EXTRACT_FOLDER']):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                text = read_file(file_path)
                                if text:
                                    file_contents.append({
                                        'filename': file,
                                        'path': os.path.relpath(file_path, app.config['EXTRACT_FOLDER']),
                                        'text': text
                                    })
                                    app.logger.debug(f'Processed file: {file}')
                            except Exception as e:
                                app.logger.error(f"Error processing {file_path}: {str(e)}")
                                continue
                    
                    # Cache the file contents for future queries
                    conversations[chat_id]['file_contents'] = file_contents
                    conversations[chat_id]['file_contents_updated'] = datetime.now().isoformat()
                    app.logger.info(f'Cached file contents for chat {chat_id}')
                else:
                    # Use cached file contents
                    file_contents = conversations[chat_id]['file_contents']
                    app.logger.info('Using cached file contents for faster response')

                combined_context = "\n\n".join(
                    f"[FILE: {file['path']}]\n{file['text'][:app.config['MAX_CONTEXT_LENGTH']]}"
                    for file in file_contents
                ) if file_contents else "No files available for context"
                app.logger.debug(f'Combined context length: {len(combined_context)}')

                # Stream model response with timeout handling
                full_answer = ""
                start_time = time.time()
                yield json.dumps({'status': 'stream_start'}) + "\n"
                
                for chunk in get_answer_stream(combined_context, question):
                    if time.time() - start_time > app.config['STREAM_TIMEOUT']:
                        raise TimeoutError("Response generation timed out")
                        
                    yield json.dumps({'status': 'stream_chunk', 'content': chunk}) + "\n"
                    full_answer += chunk

                # Finalize response
                yield json.dumps({
                    'status': 'stream_end',
                    'sources': [f['path'] for f in file_contents],
                    'conversation_id': chat_id
                }) + "\n"

                # Save assistant reply after stream
                conversations[chat_id]['messages'].append({
                    'role': 'assistant',
                    'content': full_answer,
                    'sources': [f['path'] for f in file_contents],
                    'timestamp': datetime.now().isoformat()
                })

                if len(conversations[chat_id]['messages']) == 2:
                    new_title = question[:30] + ("..." if len(question) > 30 else "")
                    conversations[chat_id]['title'] = new_title
                    app.logger.info(f'Updated chat title to: {new_title}')

            except TimeoutError:
                error_msg = "Error: Response generation timed out. Please try again with a simpler query."
                app.logger.warning(error_msg)
                yield json.dumps({'status': 'error', 'message': error_msg}) + "\n"
            except Exception as e:
                app.logger.error(f"Streaming error: {str(e)}", exc_info=True)
                yield json.dumps({'status': 'error', 'message': str(e)}) + "\n"

        return Response(stream_with_context(generate_response()), mimetype='application/x-ndjson')

    # GET request: render chat interface
    app.logger.debug(f'Rendering chat interface for chat ID: {chat_id}')
    return render_template('chat.html',
                           chat_id=chat_id,
                           conversation=conversations[chat_id],
                           chats=conversations)

@app.route('/delete_chat/<chat_id>', methods=['POST'])
def delete_chat(chat_id):
    """Delete a chat session"""
    app.logger.info(f'Received request to delete chat ID: {chat_id}')
    try:
        if chat_id in conversations:
            # Clear any cached file contents
            if 'file_contents' in conversations[chat_id]:
                del conversations[chat_id]['file_contents']
            if 'file_contents_updated' in conversations[chat_id]:
                del conversations[chat_id]['file_contents_updated']
            del conversations[chat_id]
            app.logger.info(f'Successfully deleted chat ID: {chat_id}')
            return jsonify({'success': True}), 200
        app.logger.warning(f'Chat not found for deletion: {chat_id}')
        return jsonify({'success': False, 'error': 'Chat not found'}), 404
    except Exception as e:
        app.logger.error(f'Error deleting chat {chat_id}: {str(e)}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            clean_directory(app.config['UPLOAD_FOLDER'])
            clean_directory(app.config['EXTRACT_FOLDER'])
            # Clear all cached file contents in conversations
            for chat_id in conversations:
                if 'file_contents' in conversations[chat_id]:
                    del conversations[chat_id]['file_contents']
                if 'file_contents_updated' in conversations[chat_id]:
                    del conversations[chat_id]['file_contents_updated']
            app.logger.debug('Cleaned upload and extract directories and cleared file caches')
        except Exception as e:
            app.logger.error(f'Cleanup error: {str(e)}')
            flash('Error cleaning previous files', 'error')
            return redirect(request.url)

        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('files')
        upload_folder = app.config['UPLOAD_FOLDER']
        extract_folder = app.config['EXTRACT_FOLDER']
        
        saved_files = []
        for file in files:
            if file.filename == '':
                continue
            
            filename = secure_filename(file.filename)
            
            # Check if file is a ZIP file
            is_zip = filename.lower().endswith('.zip')
            
            if is_zip:
                # Save ZIP to uploads folder
                save_path = os.path.join(upload_folder, filename)
                file.save(save_path)
                saved_files.append(filename)
                
                # Extract ZIP to extracted_files folder (only text files)
                try:
                    extract_zip(save_path, extract_folder)
                    app.logger.info(f'Successfully extracted text files from ZIP: {filename}')
                except Exception as e:
                    app.logger.error(f'Failed to extract ZIP file {filename}: {str(e)}')
                    return jsonify({
                        'error': f'Failed to extract ZIP file: {str(e)}',
                        'files': saved_files
                    }), 400
            else:
                # For non-ZIP files, only allow text files
                if not allowed_file(filename):
                    app.logger.warning(f'Skipping non-text file upload: {filename}')
                    continue
                    
                # Save directly to extracted_files folder
                path_parts = filename.split('/')
                
                if len(path_parts) > 1:
                    # Handle folder structure
                    dir_path = os.path.join(extract_folder, *path_parts[:-1])
                    os.makedirs(dir_path, exist_ok=True)
                    save_path = os.path.join(dir_path, path_parts[-1])
                else:
                    save_path = os.path.join(extract_folder, filename)
                
                file.save(save_path)
                saved_files.append(filename)
        
        if not saved_files:
            return jsonify({'error': 'No valid text files were uploaded'}), 400
        
        return jsonify({
            'message': 'Text files uploaded successfully',
            'files': saved_files
        })
    
    return render_template('index.html', chats=conversations)

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
    """Enhanced file content endpoint with chunking support"""
    rel_path = request.args.get('path')
    page_num = request.args.get('page', '')
    highlight = request.args.get('highlight', '')
    
    if not rel_path:
        return jsonify({'error': 'File path required'}), 400
    
    full_path = os.path.join(app.config['EXTRACT_FOLDER'], rel_path)
    
    if not os.path.exists(full_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        content = read_file(full_path)
        if not content:
            return jsonify({'error': 'Could not read file content'}), 400
            
        # Chunk the content for large files
        chunks = chunk_text(content)
        
        return jsonify({
            'file': os.path.basename(full_path),
            'content': chunks[0],  # First chunk
            'total_chunks': len(chunks),
            'path': rel_path
        })
    
    except Exception as e:
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
        
        ref_id = reference_counter
        reference_counter += 1
        
        # Store reference data
        references.append({
            'id': ref_id,
            'filename': filename,
            'page': page,
            'highlight': highlight
        })
        
        # Return the reference link
        return f'<sup class="reference-link" data-ref="{ref_id}">[{ref_id}]</sup>'
    
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
            ref_text = ref['filename']
            if ref['page']:
                ref_text += f", page {ref['page']}"
            if ref['highlight']:
                ref_text += f" ({ref['highlight']})"
            references_section += f'<li id="ref-{ref["id"]}">{ref_text}</li>'
        references_section += '</ol></div>'
        
        processed_text += references_section
    
    return processed_text

@app.after_request
def after_request(response):
    """Add headers to prevent timeouts"""
    response.headers["X-Cloudflare-Time"] = "600"
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