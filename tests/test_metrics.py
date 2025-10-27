from metrics import precision_at_k, mrr, ndcg_at_k

def test_precision_at_k_basic():
    ranked = ["a","b","c","d"]
    qrels = {"a":1, "d":1}
    assert precision_at_k(ranked, qrels, 2) == 0.5

def test_mrr_basic():
    ranked = ["x","b","a"]
    qrels = {"a":1}
    assert abs(mrr(ranked, qrels) - (1/3)) < 1e-9

def test_ndcg_at_k_zero_when_no_rels():
    ranked = ["a","b"]
    qrels = {}
    assert ndcg_at_k(ranked, qrels, 2) == 0.0
