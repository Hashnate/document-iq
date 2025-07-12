import faiss
import numpy as np
import os
import pickle
from typing import List, Tuple, Dict
from pathlib import Path
from ..config import Config

class SearchEngine:
    def __init__(self):
        self.config = Config()
        self.index = None
        self.documents = []
        self.load_index()
    
    def load_index(self):
        """Load existing FAISS index if available"""
        if os.path.exists(self.config.FAISS_INDEX_PATH):
            self.index = faiss.read_index(self.config.FAISS_INDEX_PATH)
            with open(f"{self.config.FAISS_INDEX_PATH}_docs.pkl", 'rb') as f:
                self.documents = pickle.load(f)
    
    def save_index(self):
        """Save the current index to disk"""
        if self.index is not None:
            os.makedirs(os.path.dirname(self.config.FAISS_INDEX_PATH), exist_ok=True)
            faiss.write_index(self.index, self.config.FAISS_INDEX_PATH)
            with open(f"{self.config.FAISS_INDEX_PATH}_docs.pkl", 'wb') as f:
                pickle.dump(self.documents, f)
    
    def build_index(self, embeddings: np.ndarray, documents: List[Dict]):
        """Build a new FAISS index"""
        dimension = embeddings.shape[1]
        
        # Use GPU if available
        if self.config.USE_GPU:
            res = faiss.StandardGpuResources()
            self.index = faiss.GpuIndexFlatIP(res, dimension)
        else:
            self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.documents = documents
        self.save_index()
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[float], List[Dict]]:
        """Search the index for similar documents"""
        if self.index is None:
            return [], []
        
        # Normalize query vector
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        
        # Search the index
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        # Get the corresponding documents
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= 0:  # FAISS returns -1 for invalid indices
                results.append((distance, self.documents[idx]))
        
        return results