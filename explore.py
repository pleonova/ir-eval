"""
Script for exploring NDCG results in detail.
Usage: python explore.py [options]
"""

import json
import pandas as pd
import argparse
from metrics import evaluate_by_query, get_query_details
from rank import load_jsonl, build_qrels, run_system, compare_systems

def explore_single_system(run, qrels, system_name, k=5):
    """Explore results for a single system."""
    print(f"\n{'='*60}")
    print(f"Exploring NDCG@{k} for {system_name}")
    print('='*60)
    
    query_results = evaluate_by_query(run, qrels, k)
    df = pd.DataFrame(query_results)
    df = df[['qid', f'P@{k}', 'MRR', f'NDCG@{k}', 'num_relevant', 'num_retrieved']]
    df = df.round(3)
    df = df.sort_values(f'NDCG@{k}', ascending=False)
    
    print("\nPer-Query Results (sorted by NDCG@k):")
    print(df.to_string(index=False))
    
    print(f"\n=== Summary Statistics ===")
    print(f"Mean NDCG@{k}: {df[f'NDCG@{k}'].mean():.3f}")
    print(f"Median NDCG@{k}: {df[f'NDCG@{k}'].median():.3f}")
    print(f"Std Dev NDCG@{k}: {df[f'NDCG@{k}'].std():.3f}")
    print(f"Min NDCG@{k}: {df[f'NDCG@{k}'].min():.3f}")
    print(f"Max NDCG@{k}: {df[f'NDCG@{k}'].max():.3f}")
    
    # Top queries
    print(f"\n=== Top 5 Queries by NDCG@{k} ===")
    top5 = df.head(5)
    for _, row in top5.iterrows():
        print(f"Query {row['qid']}: NDCG@{k}={row[f'NDCG@{k}']:.3f}, "
              f"P@{k}={row[f'P@{k}']:.3f}, Relevant docs: {row['num_relevant']}, "
              f"Retrieved: {row['num_retrieved']}")
    
    # Bottom queries
    print(f"\n=== Bottom 5 Queries by NDCG@{k} ===")
    bottom5 = df.tail(5)
    for _, row in bottom5.iterrows():
        print(f"Query {row['qid']}: NDCG@{k}={row[f'NDCG@{k}']:.3f}, "
              f"P@{k}={row[f'P@{k}']:.3f}, Relevant docs: {row['num_relevant']}, "
              f"Retrieved: {row['num_retrieved']}")
    
    # Zero NDCG queries
    zero_ndcg = df[df[f'NDCG@{k}'] == 0]
    if len(zero_ndcg) > 0:
        print(f"\n=== Queries with NDCG@{k} = 0 ({len(zero_ndcg)}) ===")
        for _, row in zero_ndcg.iterrows():
            print(f"  Query {row['qid']}: No relevant docs retrieved in top {k}")
    
    return df

def explore_all_systems(systems, qrels, k=5):
    """Compare all systems side by side."""
    print(f"\n{'='*60}")
    print(f"Comparing All Systems by NDCG@{k}")
    print('='*60)
    
    comparison_data = []
    for name, run in systems:
        for qid, ranked_ids in run.items():
            if qid not in qrels: continue
            from metrics import ndcg_at_k, precision_at_k, mrr
            comparison_data.append({
                'qid': qid,
                'system': name,
                f'NDCG@{k}': ndcg_at_k(ranked_ids, qrels[qid], k),
                f'P@{k}': precision_at_k(ranked_ids, qrels[qid], k),
                'MRR': mrr(ranked_ids, qrels[qid])
            })
    
    comp_df = pd.DataFrame(comparison_data)
    pivot_df = comp_df.pivot(index='qid', columns='system', values=f'NDCG@{k}')
    pivot_df = pivot_df.round(3)
    
    # Sort by best average across systems
    pivot_df['best'] = pivot_df.max(axis=1)
    pivot_df = pivot_df.sort_values('best', ascending=False)
    pivot_df = pivot_df.drop('best', axis=1)
    
    print("\nPer-Query NDCG Comparison:")
    print(pivot_df.to_string())
    
    print(f"\n=== System Statistics ===")
    for name, run in systems:
        col = pivot_df[name]
        print(f"{name}: Mean={col.mean():.3f}, Median={col.median():.3f}, "
              f"Min={col.min():.3f}, Max={col.max():.3f}")
    
    return pivot_df

def query_detail(qid, queries, runs, qrels, k=5):
    """Show detailed information about a specific query across systems."""
    queries_dict = {q['qid']: q for q in queries}
    
    if qid not in queries_dict:
        print(f"Query {qid} not found")
        return
    
    query_text = queries_dict[qid]['query']
    print(f"\n{'='*60}")
    print(f"Query Details: {qid}")
    print(f"Query: {query_text}")
    print('='*60)
    
    for name, run in runs:
        if qid not in run:
            continue
        
        ranked_ids = run[qid]
        details = get_query_details(qid, ranked_ids, qrels[qid], k)
        
        print(f"\n--- {name} ---")
        print(f"NDCG@{k}: {details['ndcg@k']:.3f}")
        print(f"P@{k}: {details['precision@k']:.3f}")
        print(f"MRR: {details['mrr']:.3f}")
        print(f"Total relevant docs: {details['num_relevant_docs']}")
        print(f"Retrieved relevant: {details['num_retrieved_relevant']}")
        print(f"Top {k} relevance scores: {details['top_k_relevance']}")
        print(f"Top {k} doc IDs: {details['top_k_doc_ids'][:5]}")

def main():
    parser = argparse.ArgumentParser(description='Explore NDCG results')
    parser.add_argument('--mode', choices=['single', 'compare', 'query'], 
                       default='single', help='Exploration mode')
    parser.add_argument('--system', choices=['bm25', 'embed', 'judge'], 
                       default='judge', help='Which system to explore (single mode)')
    parser.add_argument('--query', type=str, help='Query ID for detailed view (query mode)')
    parser.add_argument('-k', type=int, default=5, help='K value for NDCG@k')
    
    args = parser.parse_args()
    
    # Load data
    queries = load_jsonl("data/queries.jsonl")
    corpus = load_jsonl("data/corpus.jsonl")
    qrels = build_qrels(load_jsonl("data/qrels.jsonl"))
    
    # Run systems
    if args.mode in ['single', 'compare', 'query']:
        run1 = run_system(queries, corpus, mode="bm25", k=args.k)
        run2 = run_system(queries, corpus, mode="bm25->embed", k=args.k)
        run3 = run_system(queries, corpus, mode="bm25->embed->judge", k=args.k)
    
    if args.mode == 'single':
        if args.system == 'bm25':
            explore_single_system(run1, qrels, "BM25", k=args.k)
        elif args.system == 'embed':
            explore_single_system(run2, qrels, "BM25+EMB", k=args.k)
        else:
            explore_single_system(run3, qrels, "BM25+EMB+JUDGE", k=args.k)
    
    elif args.mode == 'compare':
        systems = [("BM25", run1), ("BM25+EMB", run2), ("BM25+EMB+JUDGE", run3)]
        explore_all_systems(systems, qrels, k=args.k)
    
    elif args.mode == 'query':
        if not args.query:
            print("Please provide --query argument")
            return
        systems = [("BM25", run1), ("BM25+EMB", run2), ("BM25+EMB+JUDGE", run3)]
        query_detail(args.query, queries, systems, qrels, k=args.k)

if __name__ == "__main__":
    main()
