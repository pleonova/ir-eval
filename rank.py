import json
from retrievers.bm25 import BM25
from retrievers.embeddings import EmbeddingRetriever
from retrievers.llm_judge import LLMJudge
from metrics import evaluate_all

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

def main():
    queries = load_jsonl("data/queries.jsonl")
    corpus = load_jsonl("data/corpus.jsonl")
    qrels = build_qrels(load_jsonl("data/qrels.jsonl"))

    run1 = run_system(queries, corpus, mode="bm25", k=5)
    run2 = run_system(queries, corpus, mode="bm25->embed", k=5)
    run3 = run_system(queries, corpus, mode="bm25->embed->judge", k=5)

    for name, run in [("BM25", run1), ("BM25+EMB", run2), ("BM25+EMB+JUDGE", run3)]:
        print(name, evaluate_all(run, qrels, k=5))

if __name__ == "__main__":
    main()
