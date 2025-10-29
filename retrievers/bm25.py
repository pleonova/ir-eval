from typing import List, Dict, Tuple
import math
from collections import Counter

class BM25:
    """
    BM25 probabilistic ranking function.
    
    BM25 is a term-based (keyword-based) ranking algorithm, NOT TF-IDF. Key features:
    - Term frequency saturation (controlled by k1): diminishing returns for repeated terms
    - Length normalization (controlled by b): adjusts scores based on document length
    - IDF (Inverse Document Frequency): measures how rare/common a term is across documents
    
    For a concrete example of how BM25 works and why rare terms score higher,
    see the "How BM25 Works" section in the README.md file.
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
        Calculate BM25 score for a query-document pair by matching query terms to document terms.
        
        BM25 formula: Σ IDF(qi) × (f(qi,D) × (k1+1)) / (f(qi,D) + k1 × (1-b + b × |D|/avgdl))
        where:
        - IDF(qi): Inverse Document Frequency - how rare term qi is across all documents
        - f(qi,D): Term frequency - how many times query term qi appears in document D
        - |D|: Length of document D (number of terms)
        - avgdl: Average document length across the corpus
        
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
            
            # IDF (Inverse Document Frequency): measures how rare this term is
            # Formula: log((N - df + 0.5) / (df + 0.5) + 1)
            # where N = total documents, df = documents containing this term
            # This is NOT simple TF-IDF; it's BM25's probabilistic IDF formula
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            
            # Length normalization component: adjusts for document length vs average
            # When b=0: no normalization; b=1: full normalization
            length_norm = 1 - self.b + self.b * len(doc_tokens) / self.avgdl
            
            # Term frequency (TF) saturation: diminishing returns controlled by k1
            # Unlike linear TF-IDF, BM25's TF component saturates as term count increases
            # This prevents over-weighting documents that spam keywords
            denom = tf[t] + self.k1 * length_norm
            tf_component = (tf[t] * (self.k1 + 1)) / denom
            
            score += idf * tf_component
            
        return score

    def rank(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Rank all documents by relevance to the query using BM25 scoring.
        
        This performs exhaustive search: scores every document in the corpus against
        the query terms and returns the top-k highest scoring documents.
        
        For large corpora, this O(N) exhaustive search can be slow. In production systems,
        BM25 is often combined with:
        - Inverted indexes to only score documents containing query terms
        - Approximate methods (e.g., HNSW, LSH) for faster retrieval
        - Two-stage ranking: fast first-pass retrieval + slower reranking
        
        Args:
            query: Query string (will be tokenized by splitting on whitespace)
            k: Number of top results to return (default: 5)
            
        Returns:
            List of (doc_id, score) tuples, sorted by score descending (highest first)
            Returns at most k results, or fewer if corpus has fewer documents
        """
        # Score every document in the corpus against the query
        # For each document, calculate BM25 score based on matching query terms
        scored = [(doc_id, self.score(query, doc_id)) for doc_id in self.docs.keys()]
        
        # Sort by score (descending) and take top-k results
        # Higher scores = more relevant documents
        return sorted(scored, key=lambda x: x[1], reverse=True)[:k]
