# Search Relevancy Evaluation

A Python project for evaluating search relevancy algorithms using IR metrics.

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

Run all systems and see aggregate metrics:

```bash
python rank.py
```

This will output performance metrics for each system.

## API Reference

### Metrics (`metrics.py`)

- `ndcg_at_k(ranked_ids, qrels, k)`: Calculate NDCG@k for a query
- `precision_at_k(ranked_ids, qrels, k)`: Calculate Precision@k
- `mrr(ranked_ids, qrels)`: Calculate Mean Reciprocal Rank
- `evaluate_all(run, qrels, k)`: Evaluate all queries and return aggregate metrics

### Ranking (`rank.py`)

- `load_jsonl(path)`: Load data from JSONL file
- `build_qrels(items)`: Build qrels dictionary from relevance judgments
- `run_system(queries, corpus, mode, k)`: Run a retrieval system with specified mode

## Project Structure

```
ir-eval/
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
└── tests/                # Unit tests
```
