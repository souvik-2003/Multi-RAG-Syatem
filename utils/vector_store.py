import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import pickle
import os
from config.settings import settings

class VectorStore:
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.dimension = 384  # MiniLM dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []
        self.metadatas = []
        self.index_file = os.path.join(settings.VECTOR_DB_PATH, "faiss_index.bin")
        self.metadata_file = os.path.join(settings.VECTOR_DB_PATH, "metadata.pkl")
        self._load_index()
    
    def _load_index(self):
        """Load existing index if available"""
        try:
            if os.path.exists(self.index_file):
                self.index = faiss.read_index(self.index_file)
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self.texts = data.get('texts', [])
                    self.metadatas = data.get('metadatas', [])
        except Exception as e:
            print(f"Could not load existing index: {e}")
    
    def _save_index(self):
        """Save index and metadata"""
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump({'texts': self.texts, 'metadatas': self.metadatas}, f)
    
    def add_document_chunks(self, chunks: List[Dict[str, Any]], document_id: str):
        """Add document chunks to vector store"""
        texts = [chunk['content'] for chunk in chunks]
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            metadatas.append({
                'document_id': document_id,
                'chunk_index': i,
                'type': chunk.get('type', 'text'),
                'page': chunk.get('page'),
                'paragraph': chunk.get('paragraph'),
                'has_images': chunk.get('has_images', False),
                'confidence': chunk.get('confidence', 1.0)
            })
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        
        # Save index
        self._save_index()
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform similarity search"""
        if len(self.texts) == 0:
            return []
        
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, min(k, len(self.texts)))
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.texts) and idx >= 0:
                results.append({
                    'content': self.texts[idx],
                    'metadata': self.metadatas[idx],
                    'distance': float(distance),
                    'relevance_score': 1 / (1 + float(distance))
                })
        
        return results
