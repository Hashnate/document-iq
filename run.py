from app import create_app
from app.models.llm_model import LLMModel

app = create_app()

if __name__ == '__main__':
    # Initialize LLM model at startup
    llm_model = LLMModel()
    app.run(host='0.0.0.0', port=5000, threaded=True)