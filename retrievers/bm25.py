from typing import List, Dict, Tuple
import math
from collections import Counter

class BM25:
    """
    BM25 probabilistic ranking function.
    
    - Term frequency saturation (controlled by k1): diminishing returns for repeated terms
    - Length normalization (controlled by b): adjusts scores based on document length
    - Probabilistic IDF component: proper inverse document frequency weighting
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 ranker.
        
        Args:
            k1: Controls term frequency saturation (typical: 1.2-2.0)
                Higher k1 = less saturation, term frequency has more impact
            b: Controls length normalization (typical: 0.75)
               b=1: full length normalization, b=0: no length normalization
        """
        self.k1 = k1
        self.b = b
        self.docs = {}
        self.avgdl = 0
        self.df = Counter()  # Document frequency for each term
        self.N = 0  # Total number of documents

    def fit(self, corpus: List[Dict[str, str]]):
        """
        Index the corpus for retrieval.
        
        Args:
            corpus: List of documents with 'doc_id' and 'text' fields
        """
        self.docs = {d["doc_id"]: d["text"].split() for d in corpus}
        self.N = len(self.docs)
        lengths = [len(tokens) for tokens in self.docs.values()]
        self.avgdl = sum(lengths) / max(1, len(lengths))  # Average document length
        
        # Calculate document frequency for each term
        for tokens in self.docs.values():
            for t in set(tokens):
                self.df[t] += 1

    def score(self, query: str, doc_id: str) -> float:
        """
        Calculate BM25 score for a query-document pair.
        
        BM25 formula: Σ IDF(qi) × (f(qi,D) × (k1+1)) / (f(qi,D) + k1 × (1-b + b × |D|/avgdl))
        where:
        - IDF(qi): Inverse document frequency of query term qi
        - f(qi,D): Frequency of qi in document D
        - |D|: Length of document D
        - avgdl: Average document length
        
        Args:
            query: Query string
            doc_id: Document identifier
            
        Returns:
            BM25 relevance score (higher = more relevant)
        """
        tokens = query.split()
        doc_tokens = self.docs[doc_id]
        tf = Counter(doc_tokens)
        score = 0.0
        
        for t in tokens:
            if t not in tf:
                continue
            
            df = self.df.get(t, 0)
            
            # BM25 IDF component: log((N - df + 0.5) / (df + 0.5) + 1)
            # This is NOT simple TF-IDF; it's a probabilistic IDF formula
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            
            # Length normalization component: adjusts for document length vs average
            # When b=0: no normalization; b=1: full normalization
            length_norm = 1 - self.b + self.b * len(doc_tokens) / self.avgdl
            
            # Term frequency saturation: diminishing returns controlled by k1
            # Unlike linear TF-IDF, repeated terms have less impact as count increases
            denom = tf[t] + self.k1 * length_norm
            tf_component = (tf[t] * (self.k1 + 1)) / denom
            
            score += idf * tf_component
            
        return score

    def rank(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Rank all documents by relevance to the query.
        
        Args:
            query: Query string
            k: Number of top results to return
            
        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        scored = [(doc_id, self.score(query, doc_id)) for doc_id in self.docs.keys()]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:k]
