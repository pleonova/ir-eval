import json
import pandas as pd
import math
from retrievers.bm25 import BM25
from retrievers.embeddings import EmbeddingRetriever
from retrievers.llm_judge import LLMJudge
from metrics import evaluate_all, evaluate_by_query

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def build_qrels(items):
    qrels = {}
    for it in items:
        qrels.setdefault(it["qid"], {})[it["doc_id"]] = int(it["rel"])
    return qrels

def run_system(queries, corpus, mode="bm25->embed->judge", k=5):
    bm25 = BM25(); bm25.fit(corpus)
    embed = EmbeddingRetriever(); embed.fit(corpus)
    doc_lookup = {d["doc_id"]: d["text"] for d in corpus}
    judge = LLMJudge()

    results = {}
    for q in queries:
        qid, qtext = q["qid"], q["query"]
        r1 = bm25.rank(qtext, k=20)
        cand_ids = [d for d, _ in r1]
        sub_corpus = [{"doc_id": d, "text": doc_lookup[d]} for d in cand_ids]
        embed.fit(sub_corpus)
        r2 = embed.rank(qtext, k=len(cand_ids))
        if "judge" in mode:
            r2_map = dict(r2)
            r2_sorted = sorted(r2_map.items(), key=lambda x: x[1], reverse=True)
            r3 = judge.rescore(qtext, r2_sorted, doc_lookup)
            final = [d for d, _ in r3[:k]]
        else:
            final = [d for d, _ in r2[:k]]
        results[qid] = final
    return results

def explore_ndcg_results(run: dict, qrels: dict, system_name: str, k: int = 5):
    """Explore NDCG results for each query."""
    query_results = evaluate_by_query(run, qrels, k)
    
    print(f"\n=== NDCG@{k} Results for {system_name} ===\n")
    
    # Create DataFrame
    df = pd.DataFrame(query_results)
    df = df[['qid', f'P@{k}', 'MRR', f'NDCG@{k}', 'num_relevant', 'num_retrieved']]
    df = df.round(3)
    
    # Sort by NDCG
    df = df.sort_values(f'NDCG@{k}', ascending=False)
    
    print(df.to_string(index=False))
    
    # Statistics
    print(f"\n=== Statistics ===")
    print(f"Mean NDCG@{k}: {df[f'NDCG@{k}'].mean():.3f}")
    print(f"Median NDCG@{k}: {df[f'NDCG@{k}'].median():.3f}")
    print(f"Min NDCG@{k}: {df[f'NDCG@{k}'].min():.3f} (Query: {df.loc[df[f'NDCG@{k}'].idxmin(), 'qid']})")
    print(f"Max NDCG@{k}: {df[f'NDCG@{k}'].max():.3f} (Query: {df.loc[df[f'NDCG@{k}'].idxmax(), 'qid']})")
    
    # Queries with zero NDCG
    zero_ndcg = df[df[f'NDCG@{k}'] == 0]
    if len(zero_ndcg) > 0:
        print(f"\nQueries with NDCG@{k} = 0 ({len(zero_ndcg)}): {', '.join(zero_ndcg['qid'].tolist())}")
    
    return df

def compare_systems(systems: list, qrels: dict, k: int = 5):
    """Compare NDCG across multiple systems for each query."""
    print(f"\n=== Comparing Systems by NDCG@{k} ===\n")
    
    comparison_data = []
    for name, run in systems:
        for qid, ranked_ids in run.items():
            if qid not in qrels: continue
            
            from metrics import ndcg_at_k
            ndcg = ndcg_at_k(ranked_ids, qrels[qid], k)
            comparison_data.append({
                'qid': qid,
                'system': name,
                f'NDCG@{k}': ndcg
            })
    
    # Create comparison DataFrame
    comp_df = pd.DataFrame(comparison_data)
    pivot_df = comp_df.pivot(index='qid', columns='system', values=f'NDCG@{k}')
    pivot_df = pivot_df.round(3)
    
    # Sort by best average
    pivot_df['avg'] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values('avg', ascending=False)
    pivot_df = pivot_df.drop('avg', axis=1)
    
    print(pivot_df.to_string())
    
    print("\n=== System Summary ===")
    for name, run in systems:
        print(f"{name}: Mean NDCG@{k} = {pivot_df[name].mean():.3f}")
    
    return pivot_df

def main():
    queries = load_jsonl("data/queries.jsonl")
    corpus = load_jsonl("data/corpus.jsonl")
    qrels = build_qrels(load_jsonl("data/qrels.jsonl"))

    run1 = run_system(queries, corpus, mode="bm25", k=5)
    run2 = run_system(queries, corpus, mode="bm25->embed", k=5)
    run3 = run_system(queries, corpus, mode="bm25->embed->judge", k=5)

    systems = [("BM25", run1), ("BM25+EMB", run2), ("BM25+EMB+JUDGE", run3)]
    results = []
    for name, run in systems:
        metrics = evaluate_all(run, qrels, k=5)
        metrics['System'] = name
        results.append(metrics)

    df = pd.DataFrame(results)
    df = df[['System', 'queries', 'P@5', 'MRR', 'NDCG@5']]
    df = df.round(2)
    print("=== Overall Results ===")
    print(df.to_string(index=False))
    
    # Add exploration capabilities
    print("\n" + "="*50)
    
    # Uncomment one of the following to explore results:
    
    # Option 1: Explore NDCG results for a specific system
    explore_ndcg_results(run3, qrels, "BM25+EMB+JUDGE", k=5)
    
    # Option 2: Compare all systems by NDCG
    compare_systems(systems, qrels, k=5)

if __name__ == "__main__":
    main()
