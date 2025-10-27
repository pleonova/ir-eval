from typing import List, Dict, Tuple

class LLMJudge:
    def rescore(self, query: str, ranked: List[Tuple[str, float]], doc_lookup: Dict[str, str]):
        adjusted = []
        q = query.lower().split()
        for doc_id, base in ranked:
            text = doc_lookup[doc_id].lower()
            bonus = sum(text.count(tok) for tok in q) * 0.02
            adjusted.append((doc_id, base + bonus))
        return sorted(adjusted, key=lambda x: x[1], reverse=True)
