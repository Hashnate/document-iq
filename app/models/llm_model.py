import ollama
from typing import Generator
from ..config import Config

class LLMModel:
    def __init__(self):
        self.config = Config()
        self.client = ollama.Client(host=self.config.OLLAMA_BASE_URL)
        
        # Ensure model is pulled
        try:
            self.client.show(self.config.OLLAMA_MODEL)
        except ollama.ResponseError:
            print(f"Downloading model {self.config.OLLAMA_MODEL}...")
            self.client.pull(self.config.OLLAMA_MODEL)
    
    def generate_response(self, prompt: str, context: str = None) -> Generator[str, None, None]:
        """Generate streaming response from Ollama"""
        full_prompt = f"Context: {context}\n\nQuestion: {prompt}" if context else prompt
        
        stream = self.client.generate(
            model=self.config.OLLAMA_MODEL,
            prompt=full_prompt,
            stream=True,
            options={
                'temperature': 0.3,
                'num_ctx': 8000  # Larger context window
            }
        )
        
        for chunk in stream:
            yield chunk['response']
    
    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Ollama's embedding endpoint"""
        embeddings = []
        for text in texts:
            response = self.client.embeddings(
                model=self.config.OLLAMA_MODEL,
                prompt=text
            )
            embeddings.append(response['embedding'])
        return embeddings