from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class JinaEmbedder:
    """
    Wrapper for Jina AI embeddings-v4 model using sentence-transformers.
    Supports retrieval task with proper query/passage prompts.
    """
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v4", task: str = "retrieval"):
        """
        Initialize the Jina embeddings model.
        
        Args:
            model_name: HuggingFace model name
            task: Task type ('retrieval', 'text-matching', 'code')
        """
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.task = task
    
    def encode(self, texts: List[str], prompt_name: Optional[str] = None) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of texts to encode
            prompt_name: 'query' or 'passage' for retrieval task, None for others
            
        Returns:
            numpy array of embeddings
        """
        if prompt_name:
            embeddings = self.model.encode(
                sentences=texts,
                task=self.task,
                prompt_name=prompt_name
            )
        else:
            embeddings = self.model.encode(
                sentences=texts,
                task=self.task
            )
        return embeddings


class DummyEmbedder:
    """Dummy embedder for testing purposes - generates random but deterministic embeddings."""
    def encode(self, texts: List[str], prompt_name: Optional[str] = None) -> np.ndarray:
        vecs = []
        for t in texts:
            rs = np.random.RandomState(abs(hash(t)) % (2**32))
            # Generate random vector in reasonable range
            vec = rs.randn(384)  # Use randn for normal distribution
            # Normalize to unit length to avoid numerical issues
            norm = np.linalg.norm(vec)
            if norm > 1e-10:  # Avoid division by zero
                vec = vec / norm
            else:
                # Fallback for edge case: create a simple unit vector
                vec = np.zeros(384)
                vec[0] = 1.0
            vecs.append(vec)
        return np.vstack(vecs)


class EmbeddingRetriever:
    """
    Embedding-based retriever that indexes documents and retrieves relevant ones for queries.
    """
    def __init__(self, embedder=None):
        """
        Initialize the retriever.
        
        Args:
            embedder: An embedder instance (JinaEmbedder or DummyEmbedder). 
                     Defaults to DummyEmbedder if None.
        """
        self.embedder = embedder or DummyEmbedder()
        self.doc_ids = []
        self.doc_vecs = None

    def fit(self, corpus: List[Dict[str, str]]):
        """
        Index the corpus documents.
        
        Args:
            corpus: List of documents with 'doc_id' and 'text' fields
        """
        self.doc_ids = [d["doc_id"] for d in corpus]
        texts = [d["text"] for d in corpus]
        # Encode as passages for retrieval task
        self.doc_vecs = self.embedder.encode(texts, prompt_name="passage")

    def rank(self, query: str, k: int = 5):
        """
        Retrieve and rank documents for a query.
        
        Args:
            query: Query string
            k: Number of top documents to return
            
        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        # Encode as query for retrieval task
        qv = self.embedder.encode([query], prompt_name="query")[0]
        
        # Compute similarity scores
        # Note: Jina embeddings are pre-normalized, so dot product = cosine similarity
        # DummyEmbedder also produces normalized vectors, so we can use dot product
        # Without normalization:
        # similarity = (vec1 @ vec2) / (||vec1|| * ||vec2||)  # Need to divide by norms
        doc_vecs_torch = torch.from_numpy(self.doc_vecs).float()
        qv_torch = torch.from_numpy(qv).float()
        
        # Compute dot product (equivalent to cosine similarity for normalized vectors)
        sims = torch.matmul(doc_vecs_torch, qv_torch)
        
        # Convert back to numpy and handle edge cases
        sims = sims.numpy()
        
        pairs = list(zip(self.doc_ids, sims.tolist()))
        return sorted(pairs, key=lambda x: x[1], reverse=True)[:k]
