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
from flask import send_file

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
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'phi4')
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

def read_file(file_path):
    """Improved file reading with caching and original structure preservation"""
    # Skip system files
    if '__MACOSX' in file_path or file_path.endswith('._.DS_Store'):
        return None
        
    if os.path.getsize(file_path) > app.config['MAX_FILE_SIZE']:
        app.logger.warning(f"File too large, skipping: {file_path}")
        return None
        
    # Check cache first
    file_hash = get_file_hash(file_path)
    cache_path = os.path.join(app.config['CACHE_FOLDER'], f"{file_hash}.txt")
    
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            # Read just the content part (after metadata)
            content_started = False
            content_lines = []
            for line in f:
                if line.strip() == "=== CONTENT BEGIN ===":
                    content_started = True
                    continue
                if line.strip() == "=== CONTENT END ===":
                    break
                if content_started:
                    content_lines.append(line)
            return "".join(content_lines)
    
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
                        text = f.read()  # ✅ Read FULL content (no truncation)
                    break
                except UnicodeDecodeError:
                    continue
    
        # Cache the result with structured format
        if text:
            # Create structured cache content with original structure
            cache_content = (
                f"=== FILE METADATA ===\n"
                f"Path: {file_path}\n"
                f"Size: {os.path.getsize(file_path)} bytes\n"
                f"Last Modified: {datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()}\n"
                f"Hash: {file_hash}\n"
                f"Type: {file_ext[1:] if file_ext else 'unknown'}\n"
                f"Extraction Method: {app.config['PDF_EXTRACTION_METHOD'] if file_ext == '.pdf' else 'standard'}\n"
                f"=== CONTENT BEGIN ===\n\n"
                f"{text}\n\n"  # ✅ Now stores FULL content
                f"=== CONTENT END ===\n"
                f"Processed at: {datetime.now().isoformat()}\n"
            )
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(cache_content)
                
            return text
    except Exception as e:
        app.logger.error(f"Error reading {file_path}: {str(e)}")
        return None
    
    return text if text and text.strip() else None


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
    chunks = chunk_text(text)
    context_window = []
    max_context_size = app.config['MAX_CONTEXT_LENGTH'] // 2  # Half for context
    
    for chunk in chunks:
        # Maintain a sliding context window
        context_window.append(chunk)
        context_window = context_window[-3:]  # Keep last 3 chunks
        
        # Build context ensuring we don't exceed limits
        context = "\n\n".join(context_window)[-max_context_size:]
        
        prompt = build_prompt(context, question, is_final=chunk == chunks[-1])
        
        try:
            response = requests.post(
                app.config['OLLAMA_URL'],
                json={
                    "model": app.config['OLLAMA_MODEL'],
                    "prompt": prompt,
                    "stream": True
                },
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'response' in data:
                        yield data['response']
                        
                        # Early termination if we get a complete answer
                        if "[ANSWER_COMPLETE]" in data['response']:
                            return
                            
        except Exception as e:
            yield f"Error processing chunk: {str(e)}"
            break

def build_prompt(context, question, is_final=False):
    """Construct a prompt that considers multi-chunk context"""
    return (
        "Document Context (may be partial):\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Instructions:\n"
        "1. Answer based on the current context\n"
        "3. Mark complete answers with [ANSWER_COMPLETE]\n"
        "4. Preserve original document structure in responses\n"
        "5. Every response that includes a section or quote must end with a citation in the format: [^filename||page||section-heading]\n"
        "   - filename must be the exact relative path from the extracted_files directory\n"
        "   - Use forward slashes (/) for path separators\n"
        "6. If no matching section is found, respond ONLY with: 'Not found in context'\n"
        "7. **Never**:\n"
        "   - Paraphrase or summarize\n"
        "   - Add interpretation\n"
        "   - Combine text from different sections\n"
        "   - Modify legal terms or definitions\n\n"
        f"{'8. This is the final context section' if is_final else ''}\n\n"
        "Response:"
    )

@app.template_filter('datetimeformat')
def datetimeformat(value, format='%b %d, %H:%M'):
    """Format datetime for templates"""
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.strftime(format)

@app.route('/new_chat', methods=['GET', 'POST'])
def new_chat():
    """Create a new query session with initial message"""
    app.logger.info('Received request to create new query')
    try:
        chat_id = str(uuid.uuid4())
        
        # Get uploaded files information
        uploaded_files = []
        for root, _, files in os.walk(app.config['UPLOAD_FOLDER']):
            for file in files:
                if not file.startswith('.') and not file == '__MACOSX':
                    uploaded_files.append(file)
        
        # Create initial message based on uploaded files
        initial_message = "Files uploaded successfully. 'Type your search query' in the Query bar below."
        if uploaded_files:
            file_list = "\n".join(f"- {file}" for file in uploaded_files[:5])  # Show first 5 files
            if len(uploaded_files) > 5:
                file_list += f"\n- ...and {len(uploaded_files) - 5} more files"
            initial_message = f"{file_list}\n\nUploaded. 'Type your search query' in the Query bar below."
        
        conversations[chat_id] = {
            'created_at': datetime.now().isoformat(),
            'title': 'New Query',
            'messages': [
                {
                    'role': 'assistant',
                    'content': initial_message,
                    'timestamp': datetime.now().isoformat()
                }
            ]
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

                # Initial status update
                yield json.dumps({
                    'status': 'status_update',
                    'message': '🔍 Preparing to process files...',
                    'progress': 0
                }) + "\n"

                # Check if we have cached file contents
                use_cached = False
                if 'file_contents' in conversations[chat_id]:
                    last_updated = conversations[chat_id].get('file_contents_updated')
                    if last_updated:
                        cache_age = (datetime.now() - datetime.fromisoformat(last_updated)).total_seconds()
                        if cache_age < app.config['CACHE_EXPIRATION_SECONDS']:
                            use_cached = True
                            file_contents = conversations[chat_id]['file_contents']
                            yield json.dumps({
                                'status': 'status_update',
                                'message': f'⚡ Using cached content from {len(file_contents)} files',
                                'progress': 30
                            }) + "\n"

                if not use_cached:
                    # Count all files first
                    all_files = []
                    for root, _, files in os.walk(app.config['EXTRACT_FOLDER']):
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
                            text = read_file(file_path)
                            if text:
                                file_contents.append({
                                    'filename': os.path.basename(file_path),
                                    'path': os.path.relpath(file_path, app.config['EXTRACT_FOLDER']),
                                    'text': text
                                })
                            
                            processed_files += 1
                            progress = 5 + int((i / total_files) * 25)
                            
                            # Update every 5 files or when done
                            if i % 5 == 0 or i == total_files - 1:
                                yield json.dumps({
                                    'status': 'status_update',
                                    'message': f'📄 Processing {os.path.basename(file_path)} ({processed_files}/{total_files})',
                                    'progress': progress
                                }) + "\n"
                                
                        except Exception as e:
                            app.logger.error(f"Error processing {file_path}: {str(e)}")
                            continue

                    # Cache the results
                    conversations[chat_id]['file_contents'] = file_contents
                    conversations[chat_id]['file_contents_updated'] = datetime.now().isoformat()

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
                    current_progress = 60 + int((len(full_answer) / 10000) * 30)  # Estimate based on answer length
                    if current_progress > 90:
                        current_progress = 90
                    
                    if int(time.time() - start_time) % 2 == 0:  # Update every 2 seconds
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

                # Save assistant reply
                conversations[chat_id]['messages'].append({
                    'role': 'assistant',
                    'content': full_answer,
                    'sources': [f['path'] for f in file_contents],
                    'timestamp': datetime.now().isoformat()
                })

                # Update chat title if first response
                if len(conversations[chat_id]['messages']) == 2:
                    new_title = question[:30] + ("..." if len(question) > 30 else "")
                    conversations[chat_id]['title'] = new_title
                    yield json.dumps({
                        'status': 'title_update',
                        'title': new_title,
                        'chat_id': chat_id
                    }) + "\n"

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

    # GET request remains the same
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
            clean_directory(app.config['CACHE_FOLDER'])
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