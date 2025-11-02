from typing import Dict, List
import math

def precision_at_k(ranked_ids: List[str], qrels: Dict[str, int], k: int) -> float:
    if k == 0: return 0.0
    hits = sum(1 for doc_id in ranked_ids[:k] if qrels.get(doc_id, 0) > 0)
    return hits / k

def mrr(ranked_ids: List[str], qrels: Dict[str, int]) -> float:
    for i, d in enumerate(ranked_ids, start=1):
        if qrels.get(d, 0) > 0:
            return 1.0 / i
    return 0.0

def dcg_at_k(rels: List[float], k: int, method: str = "exponential") -> float:
    """
    Calculate Discounted Cumulative Gain at rank k.
    
    Two formulations are available:
    
    Formula structure: DCG = Σ(relevance_gain / position_discount)
    - Position discount is SAME for both: log2(i+2) where i=0,1,2,...
    - Relevance gain is DIFFERENT:
      * Linear: gain = rel_i (proportional)
      * Exponential: gain = 2^rel_i - 1 (emphasizes high relevance)
    
    1. Exponential (default): DCG@k = Σ((2^rel_i - 1) / log2(i+2))
       - Strongly emphasizes highly relevant documents
       - Used by major search engines, Kaggle competitions, and TREC
       - Best for graded relevance (0, 1, 2, 3, etc.)
       - Relevance gains: rel=0→0, rel=1→1, rel=2→3, rel=3→7
    
    2. Linear: DCG@k = Σ(rel_i / log2(i+2))
       - Simpler, proportional gain (rel=3 is exactly 3x rel=1)
       - Same as exponential when relevance is binary (0 or 1)
       - Use when you want linear scaling of relevance grades
       - Relevance gains: rel=0→0, rel=1→1, rel=2→2, rel=3→3
    
    Position discounts (same for both): pos 1→1.0, pos 2→0.63, pos 3→0.5, pos 4→0.43
    
    Args:
        rels: List of relevance scores for ranked documents (e.g., [3, 2, 0, 1]).
              Typically 0 (not relevant), 1 (somewhat relevant), 2 (relevant), 3 (highly relevant).
        k: Number of top results to consider.
        method: "exponential" (default, stricter) or "linear" (simpler).
    
    Returns:
        DCG score as a float. Returns 0.0 if k <= 0 or rels is empty.
    
    Example - How both parts work together:
        >>> # Document with rel=3 at position 2 (i=1):
        >>> # Linear:      gain=3,       discount=log2(3)=1.585  →  contribution=3/1.585=1.89
        >>> # Exponential: gain=2^3-1=7, discount=log2(3)=1.585  →  contribution=7/1.585=4.42
        >>> # Exponential gives 2.3x more weight due to gain (7 vs 3), same position discount!
        
        >>> # With graded relevance [3, 2, 1]
        >>> dcg_at_k([3, 2, 1], k=3, method="exponential")  # Gains: [7, 3, 1]
        >>> dcg_at_k([3, 2, 1], k=3, method="linear")       # Gains: [3, 2, 1]
        
        >>> # With binary relevance [1, 1, 0] - both formulas give same result
        >>> dcg_at_k([1, 1, 0], k=3, method="exponential")  # Gains: [1, 1, 0] - same!
    """
    k = min(k, len(rels))
    if k <= 0:
        return 0.0
    
    if method == "exponential":
        # Exponential gains: emphasizes highly relevant documents (industry standard)
        return sum((2**rels[i] - 1) / math.log2(i + 2) for i in range(k))
    elif method == "linear":
        # Linear gains: proportional scaling (rel=3 contributes 3x what rel=1 does)
        return sum(rels[i] / math.log2(i + 2) for i in range(k))
    else:
        raise ValueError(f"Invalid method '{method}'. Use 'exponential' or 'linear'.")

def ndcg_at_k(ranked_ids: List[str], qrels: Dict[str, int], k: int, method: str = "exponential") -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at rank k.
    
    NDCG normalizes DCG by the ideal DCG (best possible ranking), producing a score
    between 0 and 1. A score of 1.0 means perfect ranking of relevant documents.
    
    Args:
        ranked_ids: List of document IDs in ranked order (e.g., ['doc1', 'doc3', 'doc2']).
        qrels: Dictionary mapping document IDs to relevance scores (e.g., {'doc1': 3, 'doc2': 1}).
        k: Number of top results to consider.
        method: DCG calculation method - "exponential" (default) or "linear".
                See dcg_at_k() docstring for details on when to use each.
    
    Returns:
        NDCG score between 0.0 and 1.0. Returns 0.0 if no relevant documents exist.
    
    Example:
        >>> ranked_ids = ['doc1', 'doc2', 'doc3']
        >>> qrels = {'doc1': 3, 'doc2': 2, 'doc3': 0}
        >>> ndcg_at_k(ranked_ids, qrels, k=2)  # Uses exponential by default
        # Returns 1.0 because top 2 docs are in ideal order
        >>> ndcg_at_k(ranked_ids, qrels, k=2, method="linear")  # Use linear formula
    """
    # Get relevance scores for the ranked documents
    rels = [qrels.get(d, 0) for d in ranked_ids]
    
    # Calculate DCG for the actual ranking
    num = dcg_at_k(rels, k, method=method)
    
    # Calculate IDCG (ideal DCG) using best possible ranking
    ideal = sorted(qrels.values(), reverse=True)
    den = dcg_at_k(ideal, k, method=method)
    
    return num / den if den > 0 else 0.0

def evaluate_all(run: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]], k: int = 5, method: str = "exponential"):
    """
    Evaluate all queries and return aggregate metrics.
    
    Args:
        run: Dictionary mapping query IDs to ranked lists of document IDs.
        qrels: Dictionary mapping query IDs to relevance judgments (doc_id -> relevance score).
        k: Number of top results to consider for metrics.
        method: DCG calculation method - "exponential" (default) or "linear".
    
    Returns:
        Dictionary with aggregate metrics: queries count, P@k, MRR, and NDCG@k.
    """
    p_sum = mrr_sum = ndcg_sum = 0.0
    qs = 0
    for qid, ranked_ids in run.items():
        if qid not in qrels: continue
        qs += 1
        p_sum += precision_at_k(ranked_ids, qrels[qid], k)
        mrr_sum += mrr(ranked_ids, qrels[qid])
        ndcg_sum += ndcg_at_k(ranked_ids, qrels[qid], k, method=method)
    return {
        "queries": qs, 
        f"P@{k}": round(p_sum/max(1,qs), 2), 
        "MRR": round(mrr_sum/max(1,qs), 2), 
        f"NDCG@{k}": round(ndcg_sum/max(1,qs), 2)
    }
