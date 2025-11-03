"""
Multi-stage information retrieval ranking system.

This module implements a cascading retrieval pipeline that combines:
1. BM25 for initial candidate retrieval
2. Embedding-based reranking
3. LLM judge for final scoring

The pipeline progressively refines results for better precision.
"""

import json
from retrievers.bm25 import BM25
from retrievers.embeddings import EmbeddingRetriever, JinaEmbedder
from retrievers.llm_judge import LLMJudge
from metrics import evaluate_all

def load_jsonl(path):
    """
    Load JSONL (JSON Lines) file where each line is a separate JSON object.
    
    Args:
        path: Path to the JSONL file
        
    Returns:
        List of dictionaries parsed from each line
    """
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def build_qrels(items):
    """
    Build query relevance judgments (qrels) data structure.
    
    Converts flat list of relevance judgments into nested dict format:
    {query_id: {doc_id: relevance_score}}
    
    Args:
        items: List of dicts with keys 'qid', 'doc_id', and 'rel'
        
    Returns:
        Nested dict mapping query IDs to dicts of doc IDs and relevance scores
    """
    qrels = {}
    for it in items:
        # Create nested dict: qrels[qid][doc_id] = relevance_score
        qrels.setdefault(it["qid"], {})[it["doc_id"]] = int(it["rel"])
    return qrels

def run_system(queries, corpus, mode="bm25->embed->judge", k=3, jina_embedder=None):
    """
    Run a multi-stage retrieval pipeline on queries.
    
    Pipeline stages:
    1. BM25 retrieves top-20 candidates (fast, lexical matching)
    2. Embedding reranks candidates (semantic similarity)
    3. LLM judge final scoring (optional, most expensive but highest quality)
    
    Args:
        queries: List of query dicts with 'qid' and 'query' keys
        corpus: List of document dicts with 'doc_id' and 'text' keys
        mode: Pipeline mode - can be "bm25", "embed", "bm25->embed", or "bm25->embed->judge"
        k: Number of final results to return per query
        jina_embedder: Optional pre-initialized JinaEmbedder instance to reuse across calls
        
    Returns:
        Dict mapping query IDs to lists of top-k document IDs
    """
    # Initialize retrievers based on mode
    doc_lookup = {d["doc_id"]: d["text"] for d in corpus}  # Fast document text lookup
    
    # Check for single-method modes
    if mode == "bm25":
        # Pure BM25 retrieval
        bm25 = BM25()
        bm25.fit(corpus)
        results = {}
        for q in queries:
            qid, qtext = q["qid"], q["query"]
            r_bm25 = bm25.rank(qtext, k=k)
            results[qid] = [d for d, _ in r_bm25]
        return results
    
    # For modes that need embeddings, create or reuse embedder
    if jina_embedder is None:
        jina_embedder = JinaEmbedder(model_name="jinaai/jina-embeddings-v4", task="retrieval")
    
    if mode == "embed":
        # Pure embedding retrieval with Jina
        embed = EmbeddingRetriever(embedder=jina_embedder)
        embed.fit(corpus)
        results = {}
        for q in queries:
            qid, qtext = q["qid"], q["query"]
            r_embed = embed.rank(qtext, k=k)
            results[qid] = [d for d, _ in r_embed]
        return results
    
    # For multi-stage modes, initialize all retrievers
    bm25 = BM25(); bm25.fit(corpus)
    embed = EmbeddingRetriever(embedder=jina_embedder); embed.fit(corpus)
    judge = LLMJudge()

    results = {}
    for q in queries:
        qid, qtext = q["qid"], q["query"]
        
        # Stage 1: BM25 retrieves top-20 candidates (cast wide net)
        r1 = bm25.rank(qtext, k=20)
        cand_ids = [d for d, _ in r1]
        
        # Stage 2: Rerank candidates using embeddings (semantic understanding)
        sub_corpus = [{"doc_id": d, "text": doc_lookup[d]} for d in cand_ids]
        embed.fit(sub_corpus)  # Fit on smaller candidate set for efficiency
        r2 = embed.rank(qtext, k=len(cand_ids))
        
        # Stage 3 (optional): LLM judge for final scoring
        if "judge" in mode:
            r2_map = dict(r2)  # Convert to dict for sorting
            r2_sorted = sorted(r2_map.items(), key=lambda x: x[1], reverse=True)
            r3 = judge.rescore(qtext, r2_sorted, doc_lookup)
            final = [d for d, _ in r3[:k]]  # Take top-k after LLM rescoring
        else:
            # Without judge, use embedding results directly
            final = [d for d, _ in r2[:k]]
            
        results[qid] = final
    return results

def main():
    """
    Main evaluation script comparing four retrieval approaches.
    
    Loads test data and evaluates four different pipelines:
    1. BM25 only (baseline lexical search)
    2. Embeddings only (pure semantic search)
    3. BM25 + Embeddings (hybrid: lexical retrieval + semantic reranking)
    4. BM25 + Embeddings + LLM Judge (full pipeline with LLM scoring)
    
    Prints evaluation metrics for each approach to compare effectiveness.
    """
    # Load test data
    queries = load_jsonl("data/queries.jsonl")
    corpus = load_jsonl("data/corpus.jsonl")
    qrels = build_qrels(load_jsonl("data/qrels.jsonl"))

    # Initialize Jina embedder once and reuse across all modes that need it
    print("Loading Jina embeddings model (one-time initialization)...")
    jina_embedder = JinaEmbedder(model_name="jinaai/jina-embeddings-v4", task="retrieval")
    print("✓ Model loaded!\n")

    # Run four different pipeline configurations (reusing the same embedder)
    run1 = run_system(queries, corpus, mode="bm25", k=3, jina_embedder=jina_embedder)
    run2 = run_system(queries, corpus, mode="embed", k=3, jina_embedder=jina_embedder)
    run3 = run_system(queries, corpus, mode="bm25->embed", k=3, jina_embedder=jina_embedder)
    run4 = run_system(queries, corpus, mode="bm25->embed->judge", k=3, jina_embedder=jina_embedder)

    # Evaluate and compare all approaches
    for name, run in [("BM25", run1), ("EMBED", run2), ("BM25+EMB", run3), ("BM25+EMB+JUDGE", run4)]:
        print(name, evaluate_all(run, qrels, k=3))

if __name__ == "__main__":
    main()
