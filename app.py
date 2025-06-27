import os
import zipfile
import PyPDF2
import docx
import shutil
import requests
import uuid
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
import time
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
import threading

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

# Enhanced Configuration with RunPod defaults
class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    UPLOAD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'uploads')
    EXTRACT_FOLDER = os.path.join(os.getcwd(), 'extracted_files')
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'zip'}
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
    OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434/api/generate')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3:70b-instruct-q4_K_M')
    OLLAMA_TIMEOUT = 300  # 5 minutes timeout
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 100MB per file in ZIP
    MAX_RETRIES = 3
    LOG_FILE = os.path.join(os.getcwd(), 'app.log')
    LOG_LEVEL = logging.INFO
    SERVER_NAME = os.getenv('RUNPOD_ENDPOINT_URL', '0.0.0.0:8000')
    PREFERRED_URL_SCHEME = 'https'



# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging with rotation
handler = RotatingFileHandler(
    app.config['LOG_FILE'],
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=3
)
handler.setLevel(app.config['LOG_LEVEL'])
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
logging.getLogger().setLevel(app.config['LOG_LEVEL'])

# Ensure directories exist with proper permissions
def ensure_directories():
    """Create required directories with proper permissions"""
    dirs = [app.config['UPLOAD_FOLDER'], app.config['EXTRACT_FOLDER']]
    for dir_path in dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
            os.chmod(dir_path, 0o777)  # Ensure write permissions
        except Exception as e:
            app.logger.error(f"Failed to create directory {dir_path}: {str(e)}")
            raise

ensure_directories()

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
            file_infos = [f for f in zip_ref.infolist() if not f.is_dir() and '__MACOSX' not in f.filename]
            total_files = len(file_infos)
            
            for i, file_info in enumerate(file_infos):
                try:
                    file_path = file_info.filename
                    target_path = os.path.join(extract_to, file_path)
                    
                    # Ensure target directory exists
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    
                    # Extract file
                    with zip_ref.open(file_info) as source, open(target_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
                    
                    # Update progress every 10 files
                    if i % 10 == 0:
                        progress = 30 + int((i / total_files) * 60)  # 30-90% range
                        update_progress(progress, 100, f"Extracting {os.path.basename(file_path)}...")
                        
                except Exception as e:
                    app.logger.error(f"Failed to extract {file_path}: {str(e)}")
                    continue
                    
        return True
    except zipfile.BadZipFile:
        raise ValueError("Invalid ZIP file format")
    except Exception as e:
        raise ValueError(f"ZIP extraction failed: {str(e)}")

def read_file(file_path):
    """Improved file reading with better encoding handling"""
    text = ""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if '__MACOSX' in file_path or file_path.endswith('._.DS_Store'):
            return None
            
        # PDF files
        if file_ext == '.pdf':
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        
        # Word documents
        elif file_ext == '.docx':
            doc = docx.Document(file_path)
            text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        
        # Text files with encoding fallback
        elif file_ext in ('.txt', '.md', '.csv', '.json', '.xml'):
            encodings = ['utf-8', 'iso-8859-1', 'windows-1252']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        text = file.read(100000)  # Read first 100KB
                    break
                except UnicodeDecodeError:
                    continue
        
    except Exception as e:
        app.logger.error(f"Error reading {file_path}: {str(e)}")
        return None
    
    return text if text.strip() else None

def get_answer_from_text(text, question):
    """Enhanced Ollama API integration with better formatting instructions"""
    prompt = (
        f"Context: {text[:1500000]}\n\n"
        f"Question: {question}\n"
        "Provide a detailed, well-structured answer with the following formatting:\n"
        "1. Use markdown formatting (**bold**, *italics*, bullet points, tables when appropriate)\n"
        "2. For references, use this exact format: [^filename||page||highlight]\n"
        "   - filename: source document name with extension\n"
        "   - page: page number (if applicable)\n"
        "   - highlight: key phrase from source (if applicable)\n"
        "3. Place each reference marker immediately after the relevant claim or statement\n\n"
        "Answer:"
    )
    
    for attempt in range(app.config['MAX_RETRIES']):
        try:
            response = requests.post(
                app.config['OLLAMA_URL'],
                json={
                    "model": app.config['OLLAMA_MODEL'],
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_ctx": 15000,
                        "format": "markdown"  # Request markdown formatted output
                    }
                },
                timeout=app.config['OLLAMA_TIMEOUT']
            )
            response.raise_for_status()
            return response.json().get("response", "No answer generated")
        except requests.exceptions.RequestException as e:
            if attempt == app.config['MAX_RETRIES'] - 1:
                return f"Error: Failed after {app.config['MAX_RETRIES']} attempts. {str(e)}"
            time.sleep(1)
    
    return "Error: Unexpected error getting answer"
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
    """Handle both displaying and processing chat messages"""
    app.logger.info(f'Received request for chat ID: {chat_id}')
    
    if chat_id not in conversations:
        app.logger.warning(f'Chat session not found: {chat_id}')
        if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'status': 'error', 'message': 'Chat session not found'}), 404
        return redirect(url_for('index'))

    if request.method == 'POST':
        try:
            app.logger.debug(f'Processing POST request for chat ID: {chat_id}')
            
            if request.is_json:
                data = request.get_json()
                question = data.get('question', '').strip()
                app.logger.debug(f'Received JSON data: {data}')
            else:
                question = request.form.get('question', '').strip()
                app.logger.debug(f'Received form data: {request.form}')

            if not question:
                app.logger.warning("Empty question received")
                return jsonify({'status': 'error', 'message': 'Please enter a question'}), 400

            conversations[chat_id]['messages'].append({
                'role': 'user',
                'content': question,
                'timestamp': datetime.now().isoformat()
            })
            app.logger.debug(f'Added user message to conversation: {question[:50]}...')

            file_contents = []
            app.logger.info('Processing files in extraction folder')
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

            combined_context = "\n\n".join(
                f"[FILE: {file['path']}]\n{file['text'][:10000]}"
                for file in file_contents
            ) if file_contents else "No files available for context"
            app.logger.debug(f'Combined context length: {len(combined_context)}')

            app.logger.info('Sending request to Ollama API')
            answer = get_answer_from_text(combined_context, question)
            app.logger.debug(f'Received answer from Ollama: {answer[:100]}...')

            conversations[chat_id]['messages'].append({
                'role': 'assistant',
                'content': answer,
                'sources': [f['path'] for f in file_contents],
                'timestamp': datetime.now().isoformat()
            })

            if len(conversations[chat_id]['messages']) == 2:
                new_title = question[:30] + ("..." if len(question) > 30 else "")
                conversations[chat_id]['title'] = new_title
                app.logger.info(f'Updated chat title to: {new_title}')

            app.logger.info('Successfully processed chat request')
            return jsonify({
                'status': 'success',
                'answer': answer,
                'sources': [f['path'] for f in file_contents],
                'conversation_id': chat_id
            })

        except Exception as e:
            app.logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': 'An error occurred while processing your request'
            }), 500

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
            app.logger.debug('Cleaned upload and extract directories')
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
                
                # Extract ZIP to extracted_files folder
                try:
                    extract_zip(save_path, extract_folder)
                    app.logger.info(f'Successfully extracted ZIP file: {filename}')
                except Exception as e:
                    app.logger.error(f'Failed to extract ZIP file {filename}: {str(e)}')
                    return jsonify({
                        'error': f'Failed to extract ZIP file: {str(e)}',
                        'files': saved_files
                    }), 400
            else:
                # For non-ZIP files, save directly to extracted_files folder
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
        
        return jsonify({
            'message': 'Files uploaded successfully',
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
    """Return file content using the full relative path"""
    rel_path = request.args.get('path')
    page_num = request.args.get('page', '')
    highlight = request.args.get('highlight', '')
    
    if not rel_path:
        return jsonify({'error': 'File path required'}), 400
    
    full_path = os.path.join(app.config['EXTRACT_FOLDER'], rel_path)
    
    if not os.path.exists(full_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        file_ext = os.path.splitext(full_path)[1].lower()
        content = ""
        
        if file_ext == '.pdf':
            with open(full_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # If specific page requested
                if page_num and page_num.isdigit():
                    page_idx = int(page_num) - 1
                    if 0 <= page_idx < len(reader.pages):
                        content = reader.pages[page_idx].extract_text() or ""
                    else:
                        content = f"Page {page_num} not found in document\n\n"
                        content += "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
                else:
                    content = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        
        elif file_ext == '.docx':
            doc = docx.Document(full_path)
            content = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        
        elif file_ext in ('.txt', '.md', '.csv', '.json', '.xml'):
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
        
        return jsonify({
            'file': os.path.basename(full_path),
            'content': content,
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
    
    return processed_text
    

if __name__ == '__main__':
    try:
        # check_environment()
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