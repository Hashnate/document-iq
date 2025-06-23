import os
import zipfile
import PyPDF2
import docx
import shutil
import requests
import uuid
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler

# Load environment variables
load_dotenv()

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
    """Safely extract ZIP file to directory with validation"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Validate ZIP contents before extraction
            for file in zip_ref.infolist():
                if file.file_size > app.config['MAX_FILE_SIZE']:
                    raise ValueError(f"File {file.filename} exceeds maximum size limit")
                if file.filename.startswith(('/', '..')) or '\\' in file.filename:
                    raise ValueError("Invalid file path in ZIP")
            
            zip_ref.extractall(extract_to)
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
    """Handle file uploads"""
    if request.method == 'POST':
        app.logger.info('Received file upload request')
        # Clean previous files
        clean_directory(app.config['UPLOAD_FOLDER'])
        clean_directory(app.config['EXTRACT_FOLDER'])
        app.logger.debug('Cleaned upload and extract directories')
        
        # Check if file was uploaded
        if 'file' not in request.files:
            app.logger.warning('No file part in request')
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        app.logger.debug(f'Received file: {file.filename}')
        
        if file.filename == '':
            app.logger.warning('Empty filename in request')
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                zip_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(zip_path)
                app.logger.info(f'Saved uploaded file to: {zip_path}')
                
                # Extract ZIP with validation
                extract_zip(zip_path, app.config['EXTRACT_FOLDER'])
                app.logger.info(f'Extracted ZIP file to: {app.config["EXTRACT_FOLDER"]}')
                
                # Count readable files
                file_count = 0
                for root, _, files in os.walk(app.config['EXTRACT_FOLDER']):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if read_file(file_path) is not None:
                            file_count += 1
                app.logger.info(f'Found {file_count} readable files in ZIP')
                
                if file_count == 0:
                    app.logger.warning('No readable text files found in the ZIP')
                    flash('No readable text files found in the ZIP', 'error')
                    return redirect(request.url)
                
                flash(f'Successfully processed {file_count} files', 'success')
                app.logger.info('File upload and processing completed successfully')
                return redirect(url_for('new_chat'))
            
            except ValueError as e:
                app.logger.error(f'Validation error processing file: {str(e)}')
                flash(str(e), 'error')
                return redirect(request.url)
            except Exception as e:
                app.logger.error(f"File processing error: {str(e)}", exc_info=True)
                flash('An error occurred while processing your file', 'error')
                return redirect(request.url)
        
        app.logger.warning(f'Invalid file type attempted: {file.filename}')
        flash('Only ZIP files are allowed', 'error')
        return redirect(request.url)
    
    app.logger.debug('Rendering index page')
    return render_template('index.html')

@app.route('/get_file_structure')
def get_file_structure():
    """Return the file structure of the extracted files"""
    app.logger.info('Received request for file structure')
    base_path = app.config['EXTRACT_FOLDER']
    file_structure = []

    def build_tree(path, name):
        item = {
            'name': name,
            'path': os.path.relpath(path, base_path),
            'type': 'directory' if os.path.isdir(path) else 'file',
            'extension': os.path.splitext(name)[1].lower() if os.path.isfile(path) else None
        }
        
        if os.path.isdir(path):
            item['children'] = []
            try:
                for entry in os.listdir(path):
                    full_path = os.path.join(path, entry)
                    if os.path.isdir(full_path) or entry.lower().endswith(('.pdf', '.docx', '.txt', '.md', '.csv', '.json', '.xml')):
                        item['children'].append(build_tree(full_path, entry))
            except Exception as e:
                app.logger.error(f"Error reading directory {path}: {str(e)}")
        
        return item

    if os.path.exists(base_path):
        for entry in os.listdir(base_path):
            full_path = os.path.join(base_path, entry)
            file_structure.append(build_tree(full_path, entry))
    
    app.logger.debug(f'Returning file structure with {len(file_structure)} items')
    return jsonify(file_structure)

@app.route('/get_file_content')
def get_file_content():
    """Return file content with context around the reference"""
    file_path = request.args.get('path')
    page = request.args.get('page', type=int)
    highlight = request.args.get('highlight', '')
    
    app.logger.info(f'Received request for file content: {file_path}, page: {page}, highlight: {highlight}')
    
    if not file_path:
        app.logger.warning('No file path provided in request')
        return jsonify({'error': 'File path required'}), 400
    
    # Search for the file in the extracted directory
    full_path = None
    for root, _, files in os.walk(app.config['EXTRACT_FOLDER']):
        # Compare filenames without paths
        if os.path.basename(file_path) in files:
            full_path = os.path.join(root, os.path.basename(file_path))
            break
    
    if not full_path or not os.path.exists(full_path):
        app.logger.error(f'File not found: {file_path}')
        return jsonify({'error': 'File not found'}), 404
    
    try:
        file_ext = os.path.splitext(full_path)[1].lower()
        content = ""
        
        if file_ext == '.pdf':
            with open(full_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                if page is not None and 0 <= page < len(reader.pages):
                    content = reader.pages[page].extract_text()
                elif len(reader.pages) > 0:
                    content = reader.pages[0].extract_text()
        
        elif file_ext == '.docx':
            doc = docx.Document(full_path)
            if page is not None and 0 <= page < len(doc.paragraphs):
                content = doc.paragraphs[page].text
            elif len(doc.paragraphs) > 0:
                content = doc.paragraphs[0].text
        
        elif file_ext in ('.txt', '.md', '.csv', '.json', '.xml'):
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
        
        # Find highlight in content
        highlight_index = content.lower().find(highlight.lower()) if highlight else -1
        context = ""
        
        if highlight_index >= 0:
            start = max(0, highlight_index - 100)
            end = min(len(content), highlight_index + len(highlight) + 100)
            context = content[start:end]
            context = context.replace(highlight, f"<mark>{highlight}</mark>", 1)
        
        app.logger.debug(f'Successfully retrieved content from {file_path}')
        return jsonify({
            'file': os.path.basename(full_path),
            'page': page,
            'content': context if context else content,
            'full_path': full_path
        })
    
    except Exception as e:
        app.logger.error(f'Error reading file {file_path}: {str(e)}', exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.logger.info('Starting application')
    app.run(host="0.0.0.0", port=8000)