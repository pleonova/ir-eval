from retrievers.bm25 import BM25

def test_bm25_smoke():
    """
    Basic smoke test: BM25 can fit and rank without crashing.
    
    This test demonstrates core BM25 principles (see README "How BM25 Works"):
    - Rare terms (appearing in fewer documents) get higher IDF scores
    - Documents matching more query terms rank higher
    - Term frequency saturation (multiple occurrences have diminishing returns)
    """
    corpus = [{"doc_id":"d1","text":"a a b"}, {"doc_id":"d2","text":"b c"}]
    bm = BM25()
    bm.fit(corpus)
    out = bm.rank("a b", k=2)
    assert len(out) == 2
    assert all(isinstance(s, float) for _, s in out)

def test_bm25_returns_k_results():
    """Test that k parameter correctly limits number of results."""
    corpus = [
        {"doc_id":"d1","text":"apple"},
        {"doc_id":"d2","text":"banana"},
        {"doc_id":"d3","text":"cherry"},
        {"doc_id":"d4","text":"date"}
    ]
    bm = BM25(); bm.fit(corpus)
    assert len(bm.rank("fruit", k=2)) == 2
    assert len(bm.rank("fruit", k=10)) == 4  # returns all when k > corpus size

def test_bm25_sorted_descending():
    """Test that results are sorted by score in descending order."""
    corpus = [
        {"doc_id":"d1","text":"apple apple apple"},
        {"doc_id":"d2","text":"apple apple"},
        {"doc_id":"d3","text":"apple"}
    ]
    bm = BM25(); bm.fit(corpus)
    results = bm.rank("apple", k=3)
    scores = [score for _, score in results]
    # Scores should be in descending order
    assert scores == sorted(scores, reverse=True)

def test_bm25_term_matching():
    """Test that documents with matching terms score higher than those without."""
    corpus = [
        {"doc_id":"relevant","text":"python programming language"},
        {"doc_id":"irrelevant","text":"cooking recipes food"}
    ]
    bm = BM25(); bm.fit(corpus)
    results = bm.rank("python", k=2)
    # Document with "python" should rank first
    assert results[0][0] == "relevant"
    assert results[0][1] > results[1][1]

def test_bm25_term_frequency_matters():
    """Test that higher term frequency increases relevance score."""
    corpus = [
        {"doc_id":"high_tf","text":"apple apple apple apple"},
        {"doc_id":"low_tf","text":"apple orange banana"}
    ]
    bm = BM25(); bm.fit(corpus)
    results = bm.rank("apple", k=2)
    # Document with more "apple" occurrences should score higher
    high_tf_score = [score for doc_id, score in results if doc_id == "high_tf"][0]
    low_tf_score = [score for doc_id, score in results if doc_id == "low_tf"][0]
    assert high_tf_score > low_tf_score

def test_bm25_idf_rare_terms_matter_more():
    """Test that rare terms (high IDF) have more impact than common terms."""
    corpus = [
        {"doc_id":"d1","text":"common common common rare"},
        {"doc_id":"d2","text":"common common common"},
        {"doc_id":"d3","text":"common"},
        {"doc_id":"d4","text":"common"}
    ]
    bm = BM25(); bm.fit(corpus)
    
    # Query for rare term - should strongly prefer d1
    rare_results = bm.rank("rare", k=4)
    assert rare_results[0][0] == "d1"
    # d1 should have much higher score than others for the rare term
    assert rare_results[0][1] > rare_results[1][1]

def test_bm25_empty_query():
    """Test behavior with empty query."""
    corpus = [{"doc_id":"d1","text":"hello world"}]
    bm = BM25(); bm.fit(corpus)
    results = bm.rank("", k=1)
    # Empty query should return zero scores
    assert results[0][1] == 0.0

def test_bm25_no_matching_terms():
    """Test that documents without matching query terms get zero score."""
    corpus = [
        {"doc_id":"d1","text":"apple banana"},
        {"doc_id":"d2","text":"cherry date"}
    ]
    bm = BM25(); bm.fit(corpus)
    results = bm.rank("zebra elephant", k=2)
    # All documents should have zero score (no matching terms)
    assert all(score == 0.0 for _, score in results)

def test_bm25_single_document():
    """Test edge case with single document corpus."""
    corpus = [{"doc_id":"only","text":"hello world"}]
    bm = BM25(); bm.fit(corpus)
    results = bm.rank("hello", k=1)
    assert len(results) == 1
    assert results[0][0] == "only"
    assert results[0][1] > 0.0

