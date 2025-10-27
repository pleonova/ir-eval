# Search Relevancy Evaluation

A Python project for evaluating and exploring search relevancy algorithms using IR metrics.

## Overview

This project implements and evaluates multiple retrieval systems:
- **BM25**: Classic term-based retrieval
- **BM25 + Embeddings**: BM25 followed by embedding-based reranking
- **BM25 + Embeddings + LLM Judge**: Full pipeline with LLM-based relevance judging

Evaluation metrics include Precision@k, MRR, and NDCG@k.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Evaluation

Run all systems and see aggregate metrics:

```bash
python rank.py
```

This will output a comparison table showing overall performance metrics for each system.

### Exploring NDCG Results

The project includes powerful exploration tools for analyzing NDCG results:

#### 1. Explore Single System

Get detailed per-query analysis for a specific system:

```bash
# Explore BM25 results
python explore.py --mode single --system bm25 -k 5

# Explore BM25+Embeddings results
python explore.py --mode single --system embed -k 5

# Explore BM25+Embeddings+Judge results
python explore.py --mode single --system judge -k 5
```

This shows:
- Per-query NDCG@k, P@k, and MRR scores
- Summary statistics (mean, median, std dev)
- Top 5 and bottom 5 performing queries
- Queries with zero NDCG

#### 2. Compare All Systems

Compare NDCG performance across all systems for each query:

```bash
python explore.py --mode compare -k 5
```

This creates a pivot table showing NDCG@k for each query across all systems, making it easy to identify where each system excels or fails.

#### 3. Query Details

Get detailed information about a specific query across all systems:

```bash
python explore.py --mode query --query <qid> -k 5
```

This shows:
- Query text
- Per-system NDCG@k, P@k, MRR scores
- Top-k relevance scores for each system
- Top-k document IDs

### Programmatic Usage

You can also use the exploration functions directly in Python:

```python
from rank import explore_ndcg_results, compare_systems
from rank import load_jsonl, build_qrels, run_system

# Load data
queries = load_jsonl("data/queries.jsonl")
corpus = load_jsonl("data/corpus.jsonl")
qrels = build_qrels(load_jsonl("data/qrels.jsonl"))

# Run system
run = run_system(queries, corpus, mode="bm25->embed->judge", k=5)

# Explore results
df = explore_ndcg_results(run, qrels, "BM25+EMB+JUDGE", k=5)
```

## API Reference

### Metrics (`metrics.py`)

- `ndcg_at_k(ranked_ids, qrels, k)`: Calculate NDCG@k for a query
- `precision_at_k(ranked_ids, qrels, k)`: Calculate Precision@k
- `mrr(ranked_ids, qrels)`: Calculate Mean Reciprocal Rank
- `evaluate_all(run, qrels, k)`: Evaluate all queries and return aggregate metrics
- `evaluate_by_query(run, qrels, k)`: Evaluate each query and return per-query results
- `get_query_details(qid, ranked_ids, qrels, k)`: Get detailed info for a specific query

### Exploration (`explore.py`)

- `explore_single_system(run, qrels, system_name, k)`: Detailed analysis of one system
- `explore_all_systems(systems, qrels, k)`: Compare multiple systems
- `query_detail(qid, queries, runs, qrels, k)`: Show detailed info for a specific query

## Project Structure

```
eval-project/
├── data/
│   ├── corpus.jsonl      # Document corpus
│   ├── queries.jsonl     # Test queries
│   └── qrels.jsonl        # Relevance judgments
├── retrievers/
│   ├── bm25.py           # BM25 retriever
│   ├── embeddings.py     # Embedding-based retriever
│   └── llm_judge.py      # LLM-based judge
├── metrics.py            # Evaluation metrics
├── rank.py               # Main ranking and evaluation script
├── explore.py            # Result exploration tool
└── tests/                # Unit tests
```
