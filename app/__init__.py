from flask import Flask
from flask_celery import make_celery
from .config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Create required directories
    import os
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    
    # Initialize Celery
    celery = make_celery(app)
    app.celery = celery
    
    # Register blueprints
    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)
    
    return app

app = create_app()