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


# Global progress tracking with thread safety
progress_data = {
    'current': 0,
    'total': 100,
    'message': '',
    'error': None,
    'redirect_url': None  # Add this to track where to redirect
}
progress_lock = threading.Lock()

# Configuration
class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    UPLOAD_FOLDER = 'uploads'
    EXTRACT_FOLDER = 'extracted_files'
    ALLOWED_EXTENSIONS = {'zip'}
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
    OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434/api/generate')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2:latest')
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB per file in ZIP
    MAX_RETRIES = 3  # For Ollama API calls
    LOG_FILE = 'app.log'
    LOG_LEVEL = logging.INFO
    SERVER_NAME = '127.0.0.1:5000'  # Or your actual domain
    # APPLICATION_ROOT = '/'
    PREFERRED_URL_SCHEME = 'http'  # or 'https' if using SSL

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Configure logging
handler = RotatingFileHandler(app.config['LOG_FILE'], maxBytes=10000000, backupCount=3)
handler.setLevel(app.config['LOG_LEVEL'])
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
logging.getLogger().setLevel(app.config['LOG_LEVEL'])

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['EXTRACT_FOLDER'], exist_ok=True)

# In-memory storage for conversations (replace with database in production)
conversations = {}

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_zip(zip_path, extract_to):
    """Extract ZIP file while preserving the exact folder structure and handling edge cases"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get all file information
            file_infos = zip_ref.infolist()
            total_files = len(file_infos)
            
            for i, file_info in enumerate(file_infos):
                # Skip directory entries and macOS metadata
                if file_info.is_dir() or '__MACOSX' in file_info.filename:
                    continue
                    
                # Preserve the exact path structure from the ZIP
                file_path = file_info.filename
                
                # Create target directory if needed
                target_dir = os.path.join(extract_to, os.path.dirname(file_path))
                os.makedirs(target_dir, exist_ok=True)
                
                # Extract the file
                try:
                    with zip_ref.open(file_info) as source, open(os.path.join(extract_to, file_path), 'wb') as target:
                        shutil.copyfileobj(source, target)
                except Exception as e:
                    app.logger.error(f"Failed to extract {file_path}: {str(e)}")
                    continue
                
                # Update progress if needed
                if i % 10 == 0:  # Update every 10 files
                    progress = int((i / total_files) * 100)
                    update_progress(progress, 100, f"Extracting {file_path}...")
                    
        return True
    except zipfile.BadZipFile:
        raise ValueError("The uploaded file is not a valid ZIP file")
    except Exception as e:
        raise ValueError(f"Error extracting ZIP file: {str(e)}")
        
def read_file(file_path):
    """Safely extract text from various file types with error handling"""
    text = ""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        # Skip macOS metadata files
        if '__MACOSX' in file_path or file_path.endswith('._.DS_Store'):
            return None
            
        # PDF files
        if file_ext == '.pdf':
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:  # <-- Now properly iterating through pages
                    text += page.extract_text() + "\n"
        
        # Word documents
        elif file_ext == '.docx':
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        
        # Text-based files
        elif file_ext in ('.txt', '.md'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read(100000)  # Read first 100KB
        
        # Other supported formats
        elif file_ext in ('.csv', '.json', '.xml'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read(50000)  # Read first 50KB
        
    except Exception as e:
        app.logger.error(f"Error reading {file_path}: {str(e)}")
        return None
    
    return text if text.strip() else None
def get_answer_from_text(text, question):
    """Get answer from Ollama API with retry logic"""
    for attempt in range(app.config['MAX_RETRIES']):
        try:
            prompt = (
               f"Context: {text[:1500000]}\n\n"
                f"Question: {question}\n"
                "Provide a detailed, in-depth answer with inline citations like [[ref:filename.ext||page||highlight text]] "
                "referencing the file names from the context. At the end, list all references with page number "
                "in the format:\n\n"
                "References:\n"
                "[[ref:filename1.ext||page_number||highlight text]] - relevant excerpt\n"
                "[[ref:filename2.ext||page_number||highlight text]] - relevant excerpt\n\n"
                "Answer:"
            )
         
            response = requests.post(
                app.config['OLLAMA_URL'],
                json={
                    "model": app.config['OLLAMA_MODEL'],
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_ctx": 15000
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json().get("response", "No answer generated")
        
        except requests.exceptions.RequestException as e:
            if attempt == app.config['MAX_RETRIES'] - 1:  # Last attempt
                return f"Error: Failed to get answer after {app.config['MAX_RETRIES']} attempts. {str(e)}"
            continue
    
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

@app.template_filter('datetimeformat')
def datetimeformat(value, format='%b %d, %H:%M'):
    """Format datetime for templates"""
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.strftime(format)

@app.route('/new_chat', methods=['GET', 'POST'])
def new_chat():
    """Create a new chat session"""
    app.logger.info('Received request to create new chat')
    try:
        chat_id = str(uuid.uuid4())
        conversations[chat_id] = {
            'created_at': datetime.now().isoformat(),
            'title': 'New Chat',
            'messages': []
        }
        app.logger.info(f'Created new chat with ID: {chat_id}')
        return redirect(url_for('chat', chat_id=chat_id))
    except Exception as e:
        app.logger.error(f'Error creating new chat: {str(e)}', exc_info=True)
        flash('Error creating new chat session', 'error')
        return redirect(url_for('index'))

@app.route('/chat/<chat_id>', methods=['GET', 'POST'])
def chat(chat_id):
    """Handle both displaying and processing chat messages"""
    app.logger.info(f'Received request for chat ID: {chat_id}')
    
    # Check if chat exists
    if chat_id not in conversations:
        app.logger.warning(f'Chat session not found: {chat_id}')
        if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'status': 'error', 'message': 'Chat session not found'}), 404
        return redirect(url_for('index'))

    # Handle POST requests (new messages)
    if request.method == 'POST':
        try:
            app.logger.debug(f'Processing POST request for chat ID: {chat_id}')
            
            # Get question from either form or JSON
            if request.is_json:
                data = request.get_json()
                question = data.get('question', '').strip()
                app.logger.debug(f'Received JSON data: {data}')
            else:
                question = request.form.get('question', '').strip()
                app.logger.debug(f'Received form data: {request.form}')

            # Validate question
            if not question:
                app.logger.warning("Empty question received")
                return jsonify({'status': 'error', 'message': 'Please enter a question'}), 400

            # Add user message to conversation history
            conversations[chat_id]['messages'].append({
                'role': 'user',
                'content': question,
                'timestamp': datetime.now().isoformat()
            })
            app.logger.debug(f'Added user message to conversation: {question[:50]}...')

            # Process files in the extraction folder
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

            # Combine all file contents for context
            combined_context = "\n\n".join(
                f"[FILE: {file['path']}]\n{file['text'][:10000]}"
                for file in file_contents
            ) if file_contents else "No files available for context"
            app.logger.debug(f'Combined context length: {len(combined_context)}')

            # Get answer from Ollama
            app.logger.info('Sending request to Ollama API')
            answer = get_answer_from_text(combined_context, question)
            app.logger.debug(f'Received answer from Ollama: {answer[:100]}...')

            # Add assistant response to conversation
            conversations[chat_id]['messages'].append({
                'role': 'assistant',
                'content': answer,
                'sources': [f['path'] for f in file_contents],
                'timestamp': datetime.now().isoformat()
            })

            # Update chat title if this is the first message
            if len(conversations[chat_id]['messages']) == 2:
                new_title = question[:30] + ("..." if len(question) > 30 else "")
                conversations[chat_id]['title'] = new_title
                app.logger.info(f'Updated chat title to: {new_title}')

            # Return success response
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
                'message': 'An error occurred while processing your question'
            }), 500

    # Handle GET requests (display chat interface)
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
    """Handle file uploads and progress tracking"""
    if request.method == 'POST':
        # Initialize progress tracking
        with progress_lock:
            progress_data.update({
                'current': 0,
                'total': 100,
                'message': 'Starting upload...',
                'error': None,
                'redirect_url': None
            })

        # Clean previous files
        try:
            clean_directory(app.config['UPLOAD_FOLDER'])
            clean_directory(app.config['EXTRACT_FOLDER'])
            app.logger.debug('Cleaned upload and extract directories')
        except Exception as e:
            app.logger.error(f'Cleanup error: {str(e)}')
            flash('Error cleaning previous files', 'error')
            return redirect(request.url)

        # Check if file was uploaded
        if 'file' not in request.files:
            app.logger.warning('No file part in request')
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            app.logger.warning('Empty filename in request')
            flash('No file selected', 'error')
            return redirect(request.url)

        if not allowed_file(file.filename):
            app.logger.warning(f'Invalid file type attempted: {file.filename}')
            flash('Only ZIP files are allowed', 'error')
            return redirect(request.url)

        try:
            # Save file immediately in main thread
            filename = secure_filename(file.filename)
            zip_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(zip_path)
            app.logger.info(f'Saved uploaded file to: {zip_path}')

            # Verify file was saved correctly
            if not os.path.exists(zip_path):
                raise ValueError("File failed to save properly")

            # Push application context for the thread
            ctx = app.app_context()
            ctx.push()

            # Start background processing
            thread = threading.Thread(
                target=process_upload_task,
                args=(zip_path,)
            )
            thread.daemon = True
            thread.start()

            # Return immediately - progress will be handled via SSE
            return '', 202  # HTTP 202 Accepted

        except ValueError as e:
            app.logger.error(f'Validation error: {str(e)}')
            try:
                if os.path.exists(zip_path):
                    os.remove(zip_path)
            except:
                pass
            flash(str(e), 'error')
            return redirect(request.url)

        except Exception as e:
            app.logger.error(f"Upload error: {str(e)}", exc_info=True)
            try:
                if os.path.exists(zip_path):
                    os.remove(zip_path)
            except:
                pass
            flash('An error occurred during upload', 'error')
            return redirect(request.url)

    # GET request - show upload form
    app.logger.debug('Rendering index page')
    return render_template('index.html')

@app.route('/get_file_structure')
def get_file_structure():
    """Return the accurate file structure with multi-level hierarchy"""
    base_path = app.config['EXTRACT_FOLDER']
    file_structure = []

    def build_tree(current_path, relative_path=""):
        """Recursively build the accurate file tree structure"""
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
                    # Skip hidden files and system files
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
    
    if not rel_path:
        return jsonify({'error': 'File path required'}), 400
    
    full_path = os.path.join(app.config['EXTRACT_FOLDER'], rel_path)
    
    if not os.path.exists(full_path):
        return jsonify({'error': 'File not found'}), 404
    
    # Rest of your file reading logic...
    try:
        file_ext = os.path.splitext(full_path)[1].lower()
        content = ""
        
        if file_ext == '.pdf':
            with open(full_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                content = "\n".join(page.extract_text() for page in reader.pages)
        
        elif file_ext == '.docx':
            doc = docx.Document(full_path)
            content = "\n".join(para.text for para in doc.paragraphs)
        
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

# Add this helper function at the top
def format_file_size(size_bytes):
    """Convert bytes to MB with 2 decimal places"""
    return round(size_bytes / (1024 * 1024), 2)

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
                
                # Force send when redirect_url is available
                if redirect_url or progress != last_progress or error:
                    data = {
                        'progress': progress,
                        'message': message,
                        'error': error,
                        'redirect': redirect_url
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    last_progress = progress
                
                # Exit conditions
                if progress >= 100 or error:
                    # Send explicit completion event
                    yield "event: complete\ndata: {}\n\n"
                    break
                
                time.sleep(0.5)
        except GeneratorExit:
            app.logger.info("Client disconnected from progress stream")
    
    return Response(event_stream(), mimetype="text/event-stream")
def process_upload_task(zip_path):
    """Background task for processing upload with progress updates"""
    # Create application context for this thread
    with app.app_context():
        try:
            # Initialize progress
            update_progress(0, 100, "Starting processing...")
            app.logger.info(f"Starting processing of {zip_path}")

            # Validate ZIP exists and is accessible
            if not os.path.exists(zip_path):
                error_msg = "Uploaded file not found"
                app.logger.error(error_msg)
                raise ValueError(error_msg)

            # Validate file size
            file_size = os.path.getsize(zip_path)
            if file_size == 0:
                error_msg = "Uploaded file is empty"
                app.logger.error(error_msg)
                raise ValueError(error_msg)

            update_progress(10, 100, "Validating ZIP structure...")
            app.logger.info("Validating ZIP structure")

            # Validate ZIP file
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Get file list and validate
                    file_list = zip_ref.infolist()
                    if not file_list:
                        error_msg = "ZIP file contains no files"
                        app.logger.error(error_msg)
                        raise ValueError(error_msg)

                    # Check for any files that exceed size limit
                    for file_info in file_list:
                        if not file_info.is_dir() and file_info.file_size > app.config['MAX_FILE_SIZE']:
                            error_msg = f"File '{file_info.filename}' exceeds maximum size limit of {app.config['MAX_FILE_SIZE']//(1024*1024)}MB"
                            app.logger.error(error_msg)
                            raise ValueError(error_msg)

            except zipfile.BadZipFile:
                error_msg = "Invalid ZIP file format"
                app.logger.error(error_msg)
                raise ValueError(error_msg)
            except zipfile.LargeZipFile:
                error_msg = "ZIP file requires ZIP64 support"
                app.logger.error(error_msg)
                raise ValueError(error_msg)
            except Exception as e:
                error_msg = f"Error validating ZIP file: {str(e)}"
                app.logger.error(error_msg)
                raise ValueError(error_msg)

            # Clean extraction directory
            try:
                clean_directory(app.config['EXTRACT_FOLDER'])
                update_progress(20, 100, "Cleaned extraction directory")
            except Exception as e:
                error_msg = f"Error cleaning extraction directory: {str(e)}"
                app.logger.error(error_msg)
                raise ValueError(error_msg)

            # Extract files using the improved extract_zip function
            update_progress(30, 100, "Extracting files...")
            app.logger.info(f"Extracting ZIP contents to {app.config['EXTRACT_FOLDER']}")

            try:
                success = extract_zip(zip_path, app.config['EXTRACT_FOLDER'])
                if not success:
                    raise ValueError("Failed to extract ZIP file")
                
                # Get list of extracted files for processing
                extracted_files = []
                for root, _, files in os.walk(app.config['EXTRACT_FOLDER']):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Skip system files and hidden files
                        if not any(part.startswith('.') or part == '__MACOSX' 
                                for part in file_path.split(os.sep)):
                            extracted_files.append(file_path)
                
                if not extracted_files:
                    error_msg = "No valid files extracted from ZIP"
                    app.logger.error(error_msg)
                    raise ValueError(error_msg)
                
                update_progress(50, 100, f"Extracted {len(extracted_files)} files")
            except Exception as e:
                error_msg = f"Extraction failed: {str(e)}"
                app.logger.error(error_msg)
                raise ValueError(error_msg)

            # Process extracted files (50-90%)
            update_progress(50, 100, "Processing extracted files...")
            app.logger.info(f"Processing {len(extracted_files)} extracted files")
            
            processed_count = 0
            for i, file_path in enumerate(extracted_files):
                try:
                    # Try to read the file content
                    if read_file(file_path) is not None:
                        processed_count += 1
                    
                    # Update progress (50-90%)
                    progress = 50 + int((i / len(extracted_files)) * 40)
                    update_progress(progress, 100, f"Processing {os.path.basename(file_path)}...")

                except Exception as e:
                    app.logger.error(f"Error processing {file_path}: {str(e)}")
                    continue

            if processed_count == 0:
                error_msg = "No readable text files found in the extracted files"
                app.logger.error(error_msg)
                raise ValueError(error_msg)

            # Finalize
            update_progress(95, 100, "Finalizing...")
            app.logger.info(f"Successfully processed {processed_count} files")
            
            # Get redirect URL
            redirect_url = url_for('new_chat')
            update_progress(100, 100, "Processing complete!", redirect_url=redirect_url)

        except zipfile.BadZipFile:
            error_msg = "Invalid ZIP file format"
            app.logger.error(error_msg)
            update_progress(100, 100, f"Error: {error_msg}")
        except zipfile.LargeZipFile:
            error_msg = "ZIP file requires ZIP64 support"
            app.logger.error(error_msg)
            update_progress(100, 100, f"Error: {error_msg}")
        except Exception as e:
            app.logger.error(f"Upload processing failed: {str(e)}", exc_info=True)
            update_progress(100, 100, f"Error: {str(e)}")
            
            # Cleanup on failure
            try:
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                if os.path.exists(app.config['EXTRACT_FOLDER']):
                    shutil.rmtree(app.config['EXTRACT_FOLDER'], ignore_errors=True)
            except Exception as cleanup_error:
                app.logger.error(f"Cleanup failed: {str(cleanup_error)}")
if __name__ == '__main__':
    app.logger.info('Starting application')
    app.run(debug=True)