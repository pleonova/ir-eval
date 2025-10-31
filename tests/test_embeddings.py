import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from retrievers.embeddings import DummyEmbedder, JinaEmbedder, EmbeddingRetriever


# ============================================================================
# DummyEmbedder Tests
# ============================================================================

def test_dummy_embedder_basic():
    """Test that DummyEmbedder can encode texts without crashing."""
    embedder = DummyEmbedder()
    texts = ["hello world", "foo bar"]
    embeddings = embedder.encode(texts)
    
    assert embeddings.shape == (2, 384)
    assert isinstance(embeddings, np.ndarray)


def test_dummy_embedder_deterministic():
    """Test that DummyEmbedder produces consistent embeddings for same text."""
    embedder = DummyEmbedder()
    text = ["hello world"]
    
    emb1 = embedder.encode(text)
    emb2 = embedder.encode(text)
    
    # Same text should produce identical embeddings
    np.testing.assert_array_equal(emb1, emb2)


def test_dummy_embedder_different_texts():
    """Test that different texts produce different embeddings."""
    embedder = DummyEmbedder()
    
    emb1 = embedder.encode(["hello"])
    emb2 = embedder.encode(["world"])
    
    # Different texts should have different embeddings
    assert not np.allclose(emb1, emb2)


def test_dummy_embedder_with_prompt_name():
    """Test that DummyEmbedder accepts prompt_name parameter (for compatibility)."""
    embedder = DummyEmbedder()
    
    # Should work with prompt_name parameter
    query_emb = embedder.encode(["query text"], prompt_name="query")
    passage_emb = embedder.encode(["passage text"], prompt_name="passage")
    
    assert query_emb.shape == (1, 384)
    assert passage_emb.shape == (1, 384)


# ============================================================================
# EmbeddingRetriever Tests
# ============================================================================

def test_embedding_retriever_smoke():
    """
    Basic smoke test: EmbeddingRetriever can fit and rank without crashing.
    
    Uses DummyEmbedder for fast, deterministic testing.
    """
    corpus = [
        {"doc_id": "d1", "text": "climate change impacts"},
        {"doc_id": "d2", "text": "machine learning algorithms"}
    ]
    
    retriever = EmbeddingRetriever()
    retriever.fit(corpus)
    results = retriever.rank("climate", k=2)
    
    assert len(results) == 2
    assert all(isinstance(doc_id, str) and isinstance(score, float) 
               for doc_id, score in results)


def test_embedding_retriever_returns_k_results():
    """Test that k parameter correctly limits number of results."""
    corpus = [
        {"doc_id": "d1", "text": "apple"},
        {"doc_id": "d2", "text": "banana"},
        {"doc_id": "d3", "text": "cherry"},
        {"doc_id": "d4", "text": "date"}
    ]
    
    retriever = EmbeddingRetriever()
    retriever.fit(corpus)
    
    assert len(retriever.rank("fruit", k=2)) == 2
    assert len(retriever.rank("fruit", k=10)) == 4  # returns all when k > corpus size


def test_embedding_retriever_sorted_descending():
    """Test that results are sorted by score in descending order."""
    corpus = [
        {"doc_id": "d1", "text": "python programming"},
        {"doc_id": "d2", "text": "java coding"},
        {"doc_id": "d3", "text": "javascript development"}
    ]
    
    retriever = EmbeddingRetriever()
    retriever.fit(corpus)
    results = retriever.rank("software", k=3)
    
    scores = [score for _, score in results]
    # Scores should be in descending order
    assert scores == sorted(scores, reverse=True)


def test_embedding_retriever_returns_all_docs():
    """Test that all document IDs from corpus are present."""
    corpus = [
        {"doc_id": "doc1", "text": "first document"},
        {"doc_id": "doc2", "text": "second document"},
        {"doc_id": "doc3", "text": "third document"}
    ]
    
    retriever = EmbeddingRetriever()
    retriever.fit(corpus)
    results = retriever.rank("document", k=10)
    
    returned_ids = {doc_id for doc_id, _ in results}
    expected_ids = {"doc1", "doc2", "doc3"}
    
    assert returned_ids == expected_ids


def test_embedding_retriever_single_document():
    """Test edge case with single document corpus."""
    corpus = [{"doc_id": "only", "text": "hello world"}]
    
    retriever = EmbeddingRetriever()
    retriever.fit(corpus)
    results = retriever.rank("hello", k=1)
    
    assert len(results) == 1
    assert results[0][0] == "only"
    assert isinstance(results[0][1], float)


def test_embedding_retriever_empty_query():
    """Test behavior with empty query string."""
    corpus = [
        {"doc_id": "d1", "text": "hello world"},
        {"doc_id": "d2", "text": "foo bar"}
    ]
    
    retriever = EmbeddingRetriever()
    retriever.fit(corpus)
    results = retriever.rank("", k=2)
    
    # Should still return results (DummyEmbedder will create embedding for empty string)
    assert len(results) == 2
    assert all(isinstance(score, float) for _, score in results)


def test_embedding_retriever_deterministic():
    """Test that retriever produces consistent results for same query."""
    corpus = [
        {"doc_id": "d1", "text": "machine learning"},
        {"doc_id": "d2", "text": "deep learning"}
    ]
    
    retriever = EmbeddingRetriever()
    retriever.fit(corpus)
    
    results1 = retriever.rank("learning", k=2)
    results2 = retriever.rank("learning", k=2)
    
    # Same query should produce identical results
    assert results1 == results2


def test_embedding_retriever_scores_are_valid():
    """Test that similarity scores are in reasonable range."""
    corpus = [
        {"doc_id": "d1", "text": "text one"},
        {"doc_id": "d2", "text": "text two"}
    ]
    
    retriever = EmbeddingRetriever()
    retriever.fit(corpus)
    results = retriever.rank("query", k=2)
    
    for _, score in results:
        # For normalized embeddings, cosine similarity is in [-1, 1]
        # For unnormalized (DummyEmbedder), should still be reasonable
        assert isinstance(score, float)
        assert not np.isnan(score)
        assert not np.isinf(score)


# ============================================================================
# EmbeddingRetriever with Custom Embedder Tests
# ============================================================================

def test_embedding_retriever_custom_embedder():
    """Test that retriever works with custom embedder."""
    # Create a mock embedder
    mock_embedder = Mock()
    mock_embedder.encode = Mock(return_value=np.random.rand(2, 128))
    
    corpus = [
        {"doc_id": "d1", "text": "doc one"},
        {"doc_id": "d2", "text": "doc two"}
    ]
    
    retriever = EmbeddingRetriever(embedder=mock_embedder)
    retriever.fit(corpus)
    
    # Verify embedder was called during fit with passage prompt
    mock_embedder.encode.assert_called_once()
    call_args = mock_embedder.encode.call_args
    assert call_args[0][0] == ["doc one", "doc two"]
    assert call_args[1]["prompt_name"] == "passage"


def test_embedding_retriever_with_normalized_embeddings():
    """Test that retriever works correctly with normalized embeddings (dot product)."""
    # Create a mock embedder that returns normalized vectors
    mock_embedder = Mock()
    
    # Create normalized embeddings (unit vectors)
    # d1 points in x direction, d2 points in y direction
    doc_vecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # Query points more toward x than y
    query_vec = np.array([[0.8, 0.6, 0.0]])
    
    mock_embedder.encode = Mock(side_effect=[doc_vecs, query_vec])
    
    corpus = [
        {"doc_id": "d1", "text": "doc one"},
        {"doc_id": "d2", "text": "doc two"}
    ]
    
    retriever = EmbeddingRetriever(embedder=mock_embedder)
    retriever.fit(corpus)
    results = retriever.rank("query", k=2)
    
    # Should successfully compute similarity scores using dot product
    assert len(results) == 2
    assert all(isinstance(score, float) for _, score in results)
    
    # With normalized vectors, scores should be cosine similarities
    # query [0.8, 0.6, 0] should be closer to d1 [1,0,0] (score=0.8) than d2 [0,1,0] (score=0.6)
    scores_dict = {doc_id: score for doc_id, score in results}
    assert abs(scores_dict["d1"] - 0.8) < 0.01
    assert abs(scores_dict["d2"] - 0.6) < 0.01
    assert scores_dict["d1"] > scores_dict["d2"]


def test_embedding_retriever_handles_large_corpus():
    """Test that retriever can handle larger corpus efficiently."""
    # Create corpus with 100 documents
    corpus = [
        {"doc_id": f"d{i}", "text": f"document {i} with some text"}
        for i in range(100)
    ]
    
    retriever = EmbeddingRetriever()
    retriever.fit(corpus)
    results = retriever.rank("text", k=10)
    
    assert len(results) == 10
    # Verify all scores are valid
    assert all(isinstance(score, float) for _, score in results)


# ============================================================================
# JinaEmbedder Tests (Unit tests without actual model loading)
# ============================================================================

def test_jina_embedder_initialization():
    """Test JinaEmbedder initialization parameters."""
    # We can't actually load the model in tests, but we can test the interface
    # This test would require mocking SentenceTransformer
    pass  # Skip actual model loading in unit tests


def test_jina_embedder_encode_with_prompts():
    """Test that JinaEmbedder properly passes prompt_name parameters."""
    # Mock test - would require mocking SentenceTransformer
    pass  # Skip actual model loading in unit tests


# ============================================================================
# Integration-style Tests
# ============================================================================

def test_embedding_retriever_semantic_matching():
    """
    Test that retriever shows some semantic understanding.
    Even with DummyEmbedder (deterministic hashing), identical texts should match.
    """
    corpus = [
        {"doc_id": "d1", "text": "python programming language"},
        {"doc_id": "d2", "text": "cooking recipes"},
        {"doc_id": "d3", "text": "python programming language"}  # Duplicate of d1
    ]
    
    retriever = EmbeddingRetriever()
    retriever.fit(corpus)
    results = retriever.rank("python programming language", k=3)
    
    # Exact matches (d1 and d3) should score higher than non-match (d2)
    scores_dict = {doc_id: score for doc_id, score in results}
    
    # d1 and d3 should have same score (identical text)
    assert abs(scores_dict["d1"] - scores_dict["d3"]) < 1e-6
    
    # Both should score higher than d2
    assert scores_dict["d1"] > scores_dict["d2"]
    assert scores_dict["d3"] > scores_dict["d2"]


def test_embedding_retriever_k_equals_zero():
    """Test edge case where k=0."""
    corpus = [{"doc_id": "d1", "text": "test"}]
    
    retriever = EmbeddingRetriever()
    retriever.fit(corpus)
    results = retriever.rank("query", k=0)
    
    assert len(results) == 0
    assert results == []


def test_embedding_retriever_k_negative():
    """Test that negative k behaves according to Python slicing (returns from end)."""
    corpus = [
        {"doc_id": "d1", "text": "doc1"},
        {"doc_id": "d2", "text": "doc2"},
        {"doc_id": "d3", "text": "doc3"}
    ]
    
    retriever = EmbeddingRetriever()
    retriever.fit(corpus)
    results = retriever.rank("query", k=-1)
    
    # Negative k with slicing [:k] returns all but the last element
    # [:−1] returns all elements except the last one
    assert len(results) == 2

