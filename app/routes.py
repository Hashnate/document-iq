from flask import Blueprint, render_template, request, jsonify, Response, stream_with_context
from ..models.document_processor import DocumentProcessor
from ..models.embedding_model import EmbeddingModel
from ..models.search_engine import SearchEngine
from ..models.llm_model import LLMModel
from ..tasks import process_uploaded_files
import os
from ..config import Config
import uuid
import json
import time

main = Blueprint('main', __name__)
config = Config()

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload_files():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.zip'):
        user_id = str(uuid.uuid4())
        user_folder = os.path.join(config.UPLOAD_FOLDER, user_id)
        os.makedirs(user_folder, exist_ok=True)
        
        # Save file with progress tracking
        file_path = os.path.join(user_folder, file.filename)
        file.save(file_path)
        
        # Start processing
        task = process_uploaded_files.delay(file_path, user_id)
        
        return jsonify({
            'message': 'File upload complete, processing started',
            'task_id': task.id,
            'user_id': user_id
        }), 202
    else:
        return jsonify({'error': 'Invalid file type. Only ZIP files are allowed.'}), 400

@main.route('/progress')
def upload_progress():
    """Server-side progress tracking"""
    def generate():
        progress = 0
        while progress < 100:
            progress += 10
            yield f"data: {json.dumps({'progress': progress})}\n\n"
            time.sleep(0.1)
        yield "data: {}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@main.route('/query', methods=['POST'])
def query_documents():
    data = request.get_json()
    query = data.get('query')
    user_id = data.get('user_id')
    
    if not query or not user_id:
        return jsonify({'error': 'Missing query or user_id'}), 400
    
    # Initialize models
    embedding_model = EmbeddingModel()
    search_engine = SearchEngine()
    llm_model = LLMModel()
    
    # Get query embedding
    query_embedding = embedding_model.embed_query(query)
    
    # Search for relevant documents
    search_results = search_engine.search(query_embedding)
    
    # Prepare context
    context = "\n\n".join([
        f"Document {i+1} ({result[1]['filename']}): {result[1]['text'][:1000]}..."
        for i, result in enumerate(search_results[:5])  # Limit to top 5
    ])
    
    # Stream response
    def generate():
        # First send references
        references = [
            {
                'id': i+1,
                'filename': result[1]['filename'],
                'path': result[1]['relative_path'],
                'score': float(result[0]),
                'excerpt': result[1]['text'][:200] + '...'
            }
            for i, result in enumerate(search_results[:5])
        ]
        yield f"data: {json.dumps({'type': 'references', 'content': references})}\n\n"
        
        # Stream LLM response
        for chunk in llm_model.generate_response(query, context):
            yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"
        
        yield "data: {}\n\n"  # End of stream
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@main.route('/status/<task_id>')
def task_status(task_id):
    from ..tasks import process_uploaded_files
    task = process_uploaded_files.AsyncResult(task_id)
    
    response = {
        'state': task.state,
        'status': task.info.get('status', '') if task.info else '',
        'progress': task.info.get('progress', 0) if task.info else 0,
        'current': task.info.get('current', '') if task.info else '',
        'total': task.info.get('total', '') if task.info else '',
        'document_tree': task.info.get('document_tree', {}) if task.info else {}
    }
    return jsonify(response)