from retrievers.bm25 import BM25

def test_bm25_smoke():
    corpus = [{"doc_id":"d1","text":"a a b"}, {"doc_id":"d2","text":"b c"}]
    bm = BM25(); bm.fit(corpus)
    out = bm.rank("a b", k=2)
    assert len(out) == 2
    assert all(isinstance(s, float) for _, s in out)
