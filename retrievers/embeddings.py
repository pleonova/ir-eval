from typing import List, Dict, Tuple
import numpy as np

class DummyEmbedder:
    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            rs = np.random.RandomState(abs(hash(t)) % (2**32))
            vecs.append(rs.rand(384))
        return np.vstack(vecs)

class EmbeddingRetriever:
    def __init__(self, embedder=None):
        self.embedder = embedder or DummyEmbedder()
        self.doc_ids = []
        self.doc_vecs = None

    def fit(self, corpus: List[Dict[str, str]]):
        self.doc_ids = [d["doc_id"] for d in corpus]
        texts = [d["text"] for d in corpus]
        self.doc_vecs = self.embedder.encode(texts)

    def rank(self, query: str, k: int = 5):
        qv = self.embedder.encode([query])[0]
        sims = (self.doc_vecs @ qv) / (np.linalg.norm(self.doc_vecs, axis=1) * (np.linalg.norm(qv) + 1e-9))
        pairs = list(zip(self.doc_ids, sims.tolist()))
        return sorted(pairs, key=lambda x: x[1], reverse=True)[:k]
