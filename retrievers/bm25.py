from typing import List, Dict, Tuple
import math
from collections import Counter

class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = {}
        self.avgdl = 0
        self.df = Counter()
        self.N = 0

    def fit(self, corpus: List[Dict[str, str]]):
        self.docs = {d["doc_id"]: d["text"].split() for d in corpus}
        self.N = len(self.docs)
        lengths = [len(tokens) for tokens in self.docs.values()]
        self.avgdl = sum(lengths) / max(1, len(lengths))
        for tokens in self.docs.values():
            for t in set(tokens):
                self.df[t] += 1

    def score(self, query: str, doc_id: str) -> float:
        tokens = query.split()
        doc_tokens = self.docs[doc_id]
        tf = Counter(doc_tokens)
        score = 0.0
        for t in tokens:
            if t not in tf:
                continue
            df = self.df.get(t, 0)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            denom = tf[t] + self.k1 * (1 - self.b + self.b * len(doc_tokens) / self.avgdl)
            score += idf * ((tf[t] * (self.k1 + 1)) / denom)
        return score

    def rank(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        scored = [(doc_id, self.score(query, doc_id)) for doc_id in self.docs.keys()]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:k]
