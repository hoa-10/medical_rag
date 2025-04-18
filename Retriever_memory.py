from typing import List, Dict
import numpy as np
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

class SemanticMemoryRetriever:
    def __init__(self, model_name = "all-MiniLM-L6-v2", cache_dir = None):
        self.embedding = HuggingFaceEmbeddings(
            model_name=model_name
        )
        self.memory_embedding = []
        self.memory_entries = []

    def add_memory(self, memory_data: List[Dict]):
        for entry in memory_data:
            context = f"User: {entry.get('user', '')}\nAssistant: {entry.get('bot', '')}"
            try:
                embedding = self.embedding.embed_query(context)
                self.memory_embedding.append(embedding)
                self.memory_entries.append(entry)
            except Exception as e:
                print(f"Errors embedding memory: {e}")
    def retrieve_relevant_memories(self, query: str, top_k : int = 3)  -> List[Document]:
        if not self.memory_embedding:
            return []

        query_embedding = self.embedding.embed_query(query)
        similarities = []
        for memory_embedding in self.memory_embedding: 
            similarity = self._cosine_similarity(query_embedding, memory_embedding)
            similarities.append(similarity)
        if not similarity:
            return []
        top_indices = np.argsort(similarity)[-top_k:][::-1]
        documents = []
        for idx in top_indices:
            if idx < len(self.memory_entries):
                entry = self.memory_entries[idx]
                content = f"Previous interaction (Similarity: {similarities[idx]:.2f}):\nUser: {entry.get('user', '')}\nAssistant: {entry.get('bot', '')}"
                metadata = {
                    "timestamp": entry.get("timestamp", ""),
                    "session_id": entry.get("session_id", ""),
                    "similarity": similarities[idx]
                }
                documents.append(Document(page_content=content, metadata=metadata))
        
        return documents
    def _cosine_similarity(self, vec1, vec2):
    
        if not vec1 or not vec2:
            return 0.0
        try:
            v1 = np.asarray(vec1, dtype=np.float64)
            v2 = np.asarray(vec2, dtype=np.float64)
        except (ValueError, TypeError):
            return 0.0
        if v1.shape != v2.shape:
            raise ValueError(f"Incompatible vector dimensions: {v1.shape} vs {v2.shape}")
        norm1_sq = np.einsum('i,i->', v1, v1)
        norm2_sq = np.einsum('i,i->', v2, v2)
        
        epsilon = np.finfo(float).eps 
        if norm1_sq < epsilon or norm2_sq < epsilon:
            return 0.0
        dot_product = np.einsum('i,i->', v1, v2)
        return dot_product / (np.sqrt(norm1_sq) * np.sqrt(norm2_sq))