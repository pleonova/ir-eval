# Search Relevancy Evaluation

A Python project for evaluating search relevancy algorithms using IR metrics.

## Overview

This project implements and evaluates multiple retrieval systems:
- **BM25**: Classic probabilistic ranking function with term frequency saturation and length normalization
- **BM25 + Embeddings**: BM25 followed by dummy embedding-based reranking (placeholder implementation using deterministic random vectors)
- **BM25 + Embeddings + Keyword Judge**: Full pipeline with keyword-counting based rescoring

**Note**: This is a prototype/skeleton implementation. The embedding retriever uses dummy vectors (not real semantic embeddings), and the "judge" component uses simple keyword matching rather than an actual LLM.

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

## Demo Files

The `demos/` folder contains example scripts and notebooks to help you explore the retrieval systems:

- **`demo_jina_embeddings.py`**: Python script demonstrating how to use the Jina embedding retriever with real semantic embeddings
- **`demo_retrievers.ipynb`**: Interactive Jupyter notebook walking through different retrieval approaches, comparing BM25 and embedding-based methods

These demos provide hands-on examples of using the retrieval systems and are great starting points for experimentation.

## How BM25 Works: A Concrete Example

To understand BM25's behavior, let's walk through a simple example:

**Corpus:**
- Document d1: "a a b" (term 'a' appears twice)
- Document d2: "b c"

**Query:** "a b"

**Results:**
1. d1: score 1.0977
2. d2: score 0.2004

### Why d1 Scores Much Higher

**Document d1 (1.0977)** ✅
- Matches BOTH query terms: "a" and "b"
- Contains "a" (twice!) - **RARE term** (only in 1/2 docs) → high IDF → big score boost (~0.89)
- Contains "b" (once) - common term (in 2/2 docs) → low IDF → small contribution (~0.20)

**Document d2 (0.2004)** ⚠️
- Matches only ONE query term: "b"
- Does NOT contain "a" → no contribution (0.0)
- Contains "b" (once) - common term → low IDF → small score (~0.20)

### Key BM25 Principles

1. **IDF (Inverse Document Frequency)**: Rare terms are weighted more heavily
   - Term "a": appears in 1/2 documents → HIGH IDF
   - Term "b": appears in 2/2 documents → LOW IDF

2. **Term Frequency Saturation**: Multiple occurrences help, but with diminishing returns
   - d1 has "a" twice, but doesn't get full 2x credit (controlled by parameter k1=1.5)

3. **Matching Multiple Query Terms**: Documents matching more query terms rank higher
   - d1 matches 2/2 query terms
   - d2 matches 1/2 query terms

**The Big Insight**: BM25 correctly ranks d1 first because it matches MORE query terms and contains the RARE term "a" which has high discriminative power. Common terms like "b" that appear in many documents contribute less to the final score.

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
│   └── qrels.jsonl        # Relevance judgments (3-point scale)
├── demos/
│   ├── demo_jina_embeddings.py  # Example script using Jina embeddings
│   └── demo_retrievers.ipynb     # Interactive notebook for exploring retrievers
├── retrievers/
│   ├── bm25.py           # BM25 retriever implementation
│   ├── embeddings.py     # Dummy embedding retriever (deterministic random vectors) 
│   └── llm_judge.py      # Keyword-counting rescorer (not an actual LLM - placeholder)
├── metrics.py            # Evaluation metrics (Precision@k, MRR, NDCG@k)
├── rank.py               # Main ranking and evaluation script
└── tests/                # Unit tests
```
