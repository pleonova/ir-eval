from typing import Dict, List, Tuple
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

def dcg_at_k(rels: List[int], k: int) -> float:
    return sum((rel / math.log2(i+2)) for i, rel in enumerate(rels[:k]))

def ndcg_at_k(ranked_ids: List[str], qrels: Dict[str, int], k: int) -> float:
    rels = [qrels.get(d, 0) for d in ranked_ids]
    dcg = dcg_at_k(rels, k)
    ideal = sorted(qrels.values(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    return 0.0 if idcg == 0 else dcg / idcg

def evaluate_all(run: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]], k: int = 5):
    p_sum = mrr_sum = ndcg_sum = 0.0
    qs = 0
    for qid, ranked_ids in run.items():
        if qid not in qrels: continue
        qs += 1
        p_sum += precision_at_k(ranked_ids, qrels[qid], k)
        mrr_sum += mrr(ranked_ids, qrels[qid])
        ndcg_sum += ndcg_at_k(ranked_ids, qrels[qid], k)
    return {"queries": qs, f"P@{k}": p_sum/max(1,qs), "MRR": mrr_sum/max(1,qs), f"NDCG@{k}": ndcg_sum/max(1,qs)}

def evaluate_by_query(run: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]], k: int = 5) -> List[Dict]:
    """Evaluate each query individually and return detailed results."""
    results = []
    for qid, ranked_ids in run.items():
        if qid not in qrels: continue
        
        # Calculate metrics
        ndcg = ndcg_at_k(ranked_ids, qrels[qid], k)
        p_at_k = precision_at_k(ranked_ids, qrels[qid], k)
        mrr_score = mrr(ranked_ids, qrels[qid])
        
        # Get ranked relevance scores
        rels = [qrels[qid].get(d, 0) for d in ranked_ids[:k]]
        
        results.append({
            'qid': qid,
            f'P@{k}': p_at_k,
            'MRR': mrr_score,
            f'NDCG@{k}': ndcg,
            'ranked_relevance': rels,
            'num_relevant': sum(1 for rel in qrels[qid].values() if rel > 0),
            'num_retrieved': sum(1 for r in ranked_ids[:k] if qrels[qid].get(r, 0) > 0)
        })
    
    return results

def get_query_details(qid: str, ranked_ids: List[str], qrels: Dict[str, int], 
                     k: int = 5) -> Dict:
    """Get detailed information about a specific query's results."""
    rels = [qrels.get(d, 0) for d in ranked_ids]
    
    return {
        'qid': qid,
        'ndcg@k': ndcg_at_k(ranked_ids, qrels, k),
        'precision@k': precision_at_k(ranked_ids, qrels, k),
        'mrr': mrr(ranked_ids, qrels),
        'num_relevant_docs': sum(1 for v in qrels.values() if v > 0),
        'num_retrieved_relevant': sum(1 for r in ranked_ids[:k] if qrels.get(r, 0) > 0),
        'top_k_relevance': rels[:k],
        'top_k_doc_ids': ranked_ids[:k],
        'all_relevance_scores': rels
    }
