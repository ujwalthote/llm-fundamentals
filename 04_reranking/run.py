"""04 - Reranking: how much does a cross-encoder add on top of dense search, and at what cost?

Dense search (bge-small, from 02) returns the top candidates for each FiQA question. A
cross-encoder then reads the question and each candidate *together* and re-scores them.
We vary the reranker and how many candidates it reads (the depth).

Run (after 02_retrieval/run.py has cached the post vectors):
    uv run python 04_reranking/run.py
"""

import sys
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent / "02_retrieval"))
from run import CACHE as VEC_CACHE  # noqa: E402
from run import EMBED_MODEL, QUERY_PREFIX, K, bootstrap, load_fiqa, score, tokenize  # noqa: E402

# Every candidate is one full model run on CPU (about 33 ms for L6, 75 ms for L12 at 512 tokens),
# so depth is kept modest. The ceiling is still printed for deeper lists.
DEPTHS = (10, 20, 30)  # how many dense candidates the reranker reads
MAX_DEPTH = max(DEPTHS)
RERANKERS = {
    "MiniLM-L6 (22M)": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "MiniLM-L12 (33M)": "cross-encoder/ms-marco-MiniLM-L12-v2",
}
TIMING_QUESTIONS = 50  # latency is measured one question at a time on this many questions
CACHE = Path(__file__).parent / "cache"


def dense_candidates(n_docs, questions):
    """Top 100 post indices per question, best first (only the first MAX_DEPTH get reranked)."""
    doc_vecs = np.load(VEC_CACHE / f"{EMBED_MODEL.split('/')[-1]}_{n_docs}.npy")
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    q_vecs = model.encode([QUERY_PREFIX + q for q in questions], normalize_embeddings=True)
    return np.array([np.argsort(-(doc_vecs @ qv))[:100] for qv in q_vecs])


def bm25_candidates(docs, questions):
    """Top MAX_DEPTH post indices per question from BM25, best first."""
    bm25 = BM25Okapi([tokenize(d) for d in docs])
    return np.array([np.argsort(-bm25.get_scores(tokenize(q)))[:MAX_DEPTH] for q in questions])


def rerank_scores(name, model, docs, questions, candidates, source="dense"):
    """Cross-encoder score for every (question, candidate) pair, shape (questions, MAX_DEPTH)."""
    path = CACHE / f"{RERANKERS[name].split('/')[-1]}_{source}_{MAX_DEPTH}.npy"
    if path.exists():
        return np.load(path)
    pairs = [
        (q, docs[i])
        for q, cands in zip(questions, candidates, strict=True)
        for i in cands[:MAX_DEPTH]
    ]
    flat = model.predict(pairs, batch_size=32, show_progress_bar=True)
    scores = np.asarray(flat).reshape(len(questions), MAX_DEPTH)
    CACHE.mkdir(exist_ok=True)
    np.save(path, scores)
    return scores


def ms_per_question(model, docs, questions, candidates, depth):
    """Rerank `depth` candidates one question at a time, like a live request."""
    model.predict([(questions[0], docs[i]) for i in candidates[0][:depth]])  # warm-up
    start = time.perf_counter()
    for q, cands in zip(questions[:TIMING_QUESTIONS], candidates, strict=False):
        model.predict([(q, docs[i]) for i in cands[:depth]], batch_size=depth)
    return (time.perf_counter() - start) / TIMING_QUESTIONS * 1000


def evaluate(rankings, doc_ids, qids, relevant):
    """Per-question (recall@10, MRR@10, nDCG@10) for a list of rankings."""
    return np.array(
        [
            score([doc_ids[i] for i in r[:K]], relevant[q])
            for r, q in zip(rankings, qids, strict=True)
        ]
    )


def main():
    doc_ids, docs, qids, questions, relevant = load_fiqa()
    print(f"{len(docs):,} posts, {len(questions)} test questions\n")
    candidates = dense_candidates(len(docs), questions)

    # Ceiling: share of correct posts that are among the candidates at all.
    print("Ceiling (recall of the candidate list; a reranker cannot go above it):")
    for d in (*DEPTHS, 50, 100):
        rec = np.mean(
            [
                len({doc_ids[i] for i in c[:d]} & relevant[q]) / len(relevant[q])
                for c, q in zip(candidates, qids, strict=True)
            ]
        )
        print(f"  top {d:>3}: {rec:.3f}")

    base = evaluate(candidates, doc_ids, qids, relevant)
    base_found = base[:, 0] > 0
    rows = [("dense only", "-", base, 0.0)]
    for name, model_id in RERANKERS.items():
        print(f"\n> {name}")
        model = CrossEncoder(model_id, device="cpu", max_length=512)
        scores = rerank_scores(name, model, docs, questions, candidates)
        for d in DEPTHS:
            order = np.argsort(-scores[:, :d], axis=1)  # rerank only the first d candidates
            reranked = [c[o] for c, o in zip(candidates, order, strict=True)]
            ms = ms_per_question(model, docs, questions, candidates, d)
            rows.append((name, d, evaluate(reranked, doc_ids, qids, relevant), ms))

    print(
        f"\n| Reranker | Depth | recall@{K} | MRR@{K} | nDCG@{K} | nDCG vs dense (95% CI) "
        "| Rescued | Lost | ms / question |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for name, d, per_q, ms in rows:
        r, m, n = per_q.mean(axis=0)
        if d == "-":
            print(f"| {name} | - | {r:.3f} | {m:.3f} | {n:.3f} | - | - | - | - |")
            continue
        diff = per_q[:, 2] - base[:, 2]
        lo, hi = bootstrap(diff)
        found = per_q[:, 0] > 0
        rescued = (found & ~base_found).sum()  # nothing in dense top 10, something after
        lost = (~found & base_found).sum()
        print(
            f"| {name} | {d} | {r:.3f} | {m:.3f} | {n:.3f} | "
            f"{diff.mean():+.3f} ({lo:+.3f} - {hi:+.3f}) | {rescued} | {lost} | {ms:.0f} |"
        )

    # Cross-check: the same reranker on a weaker first stage. The BEIR paper reports that this
    # cross-encoder lifts BM25 on FiQA from 0.236 to 0.347 nDCG@10 (reranking BM25's top 100).
    name = "MiniLM-L6 (22M)"
    print(f"\nCross-check: {name} reranking BM25's top {MAX_DEPTH}")
    bm25_cands = bm25_candidates(docs, questions)
    model = CrossEncoder(RERANKERS[name], device="cpu", max_length=512)
    scores = rerank_scores(name, model, docs, questions, bm25_cands, source="bm25")
    bm25_base = evaluate(bm25_cands, doc_ids, qids, relevant)
    order = np.argsort(-scores, axis=1)
    reranked = evaluate(
        [c[o] for c, o in zip(bm25_cands, order, strict=True)], doc_ids, qids, relevant
    )
    diff = reranked[:, 2] - bm25_base[:, 2]
    lo, hi = bootstrap(diff)
    print(
        f"  BM25 nDCG@{K} {bm25_base[:, 2].mean():.3f} -> reranked {reranked[:, 2].mean():.3f}"
        f"  ({diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}])"
    )


if __name__ == "__main__":
    main()
