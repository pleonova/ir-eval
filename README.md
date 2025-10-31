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
- **`demo_dcg_comparison.py`**: Comparison of exponential vs linear DCG formulas with concrete examples showing position discounts and relevance gains

Run demos from the project root:
```bash
python demos/demo_dcg_comparison.py
python demos/demo_jina_embeddings.py
```

These demos provide hands-on examples of using the retrieval systems and metrics, and are great starting points for experimentation.

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

## Understanding Asymmetric Embeddings

Modern embedding models (like Jina v3/v4, E5, BGE) use **asymmetric encoding** to dramatically improve retrieval performance. This approach recognizes that queries and documents play fundamentally different roles.

### Why Asymmetric Encoding?

**Queries and documents are different:**
- **Queries**: Short, incomplete, question-like ("python tutorial", "reset password")
- **Documents**: Long, complete, statement-like ("Python Programming Tutorial: Learn Python basics...")

**Traditional (Symmetric) Approach:**
```python
# Same encoding for everything - treats queries and docs identically
query_emb = model.encode("python tutorial")
doc_emb = model.encode("Python Programming Tutorial...")
# Result: ~65% retrieval accuracy
```

**Modern (Asymmetric) Approach:**
```python
# Different prompts for different roles
query_emb = model.encode("python tutorial", prompt_name="query")
doc_emb = model.encode("Python Programming Tutorial...", prompt_name="passage")
# Result: ~80% retrieval accuracy (+23% improvement!)
```

### How It Works

The `prompt_name` parameter tells the model what role the text is playing:

```python
# During indexing - encode documents as "passages"
retriever.fit(corpus)  # Uses prompt_name="passage" internally

# During search - encode query as "query"  
results = retriever.rank("your query")  # Uses prompt_name="query" internally
```

Internally, the model prepends task-specific instructions:
- **Query**: "Represent this query for retrieving relevant documents: [your query]"
- **Passage**: "Represent this document for retrieval: [document text]"

### Performance Impact

Research shows asymmetric encoding provides:
- **+20-40% improvement** in retrieval metrics (NDCG@10, Recall@100)
- Better semantic understanding of search intent
- Optimized embeddings for the query-document matching task

**Key Takeaway**: Always use `prompt_name="query"` for queries and `prompt_name="passage"` for documents when using modern embedding models like Jina v4!

## API Reference

### Metrics (`metrics.py`)

- `ndcg_at_k(ranked_ids, qrels, k, method="exponential")`: Calculate NDCG@k for a query
- `dcg_at_k(rels, k, method="exponential")`: Calculate DCG@k with exponential or linear gains
- `precision_at_k(ranked_ids, qrels, k)`: Calculate Precision@k
- `mrr(ranked_ids, qrels)`: Calculate Mean Reciprocal Rank
- `evaluate_all(run, qrels, k, method="exponential")`: Evaluate all queries and return aggregate metrics

#### Choosing the Right Metric

**Relevance scores** are ground truth labels needed for evaluation. They can be:
- **Binary**: 0 (not relevant) or 1 (relevant)
- **Graded**: Numerical scores like 0-3 or 1-5

**Which metric should you use?**

| Your Data | Metric | When to Use | What It Measures |
|-----------|--------|-------------|------------------|
| **Graded relevance** (0-3 scale) | `ndcg_at_k()` | ✅ **Best for this project** | Rewards highly relevant docs more; normalized score 0-1 |
| Binary (0/1) or Graded | `precision_at_k()` | Any relevant docs in top k? | % of top-k that are relevant (treats any rel > 0 as relevant) |
| Binary (0/1) or Graded | `mrr()` | Where's first relevant result? | Reciprocal rank of first relevant doc (1/position) |

**This project uses graded relevance** (see `data/qrels.jsonl` with `rel: 3`, `rel: 1`, etc.), so **NDCG@k is the most appropriate metric** as it properly weights the difference between highly relevant and somewhat relevant documents.

#### DCG Formula: Exponential vs Linear

The `dcg_at_k()` and `ndcg_at_k()` functions support two formulations (defaults to exponential):

**Formula Structure:**
```
DCG = Σ(relevance_gain / position_discount)
```
- **Position discount**: SAME for both formulas (log2(i+2))
- **Relevance gain**: DIFFERENT between formulas (this is what changes!)

**1. Exponential (default, stricter)** - `method="exponential"`
```
DCG = Σ((2^rel_i - 1) / log2(i+2))   where i=0,1,2,... (0-indexed position)
```
- ✅ **Use this for graded relevance** (recommended for this project)
- Used by Google, Kaggle competitions, TREC benchmarks
- Strongly emphasizes highly relevant documents
- **Relevance gains**: `rel=0→0`, `rel=1→1`, `rel=2→3`, `rel=3→7` ⭐

**2. Linear (simpler)** - `method="linear"`
```
DCG = Σ(rel_i / log2(i+2))   where i=0,1,2,... (0-indexed position)
```
- Use when you want proportional scaling of relevance grades
- **Relevance gains**: `rel=0→0`, `rel=1→1`, `rel=2→2`, `rel=3→3` (proportional)
- **Same as exponential when relevance is binary** (0 or 1)

**Position discounts** (same for both): pos 1→1.0, pos 2→0.63, pos 3→0.5, pos 4→0.43, pos 5→0.39

#### Concrete Example: How Both Parts Work Together

Document with **rel=3** at **position 2** (i=1):

**Linear DCG:**
```
relevance_gain = 3 (just the relevance score)
position_discount = log2(1+2) = log2(3) = 1.585
contribution = 3 / 1.585 = 1.89
```

**Exponential DCG:**
```
relevance_gain = 2^3 - 1 = 7 (exponential boost!)
position_discount = log2(1+2) = log2(3) = 1.585 (same as linear!)
contribution = 7 / 1.585 = 4.42
```

**Key insight**: Exponential gives this document **2.3x more weight** (4.42 vs 1.89) because of the gain formula (7 vs 3), even though the position discount is identical. Both formulas penalize lower positions the same way; they differ only in how they reward relevance.

**Example comparison**:
```python
# Query results with graded relevance
ranked_ids = ['doc1', 'doc2', 'doc3']
qrels = {'doc1': 3, 'doc2': 1, 'doc3': 0}  # Graded: highly, somewhat, not relevant

# Using exponential DCG (default - stricter, emphasizes high relevance)
ndcg_at_k(ranked_ids, qrels, k=3)                      # 1.0 - Perfect ranking!
ndcg_at_k(ranked_ids, qrels, k=3, method="exponential") # 1.0 - Same (explicit)

# Using linear DCG (simpler, equal weighting)
ndcg_at_k(ranked_ids, qrels, k=3, method="linear")     # 1.0 - Also perfect

# Other metrics (binary treatment of relevance)
precision_at_k(ranked_ids, qrels, k=3)  # 0.67 - 2 out of 3 are relevant (any rel > 0)
mrr(ranked_ids, qrels)                  # 1.0 - First result is relevant

# For binary relevance data, both DCG methods give identical results
binary_qrels = {'doc1': 1, 'doc2': 1, 'doc3': 0}  # Binary relevance
ndcg_at_k(ranked_ids, binary_qrels, k=3, method="exponential")  # Same result
ndcg_at_k(ranked_ids, binary_qrels, k=3, method="linear")       # Same result
```

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
