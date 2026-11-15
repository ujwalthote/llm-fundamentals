"""02 - Retrieval: keyword search (BM25) vs meaning search (dense embeddings).

For each FiQA test question, both methods return their top 10 posts out of 57,638.
We check how many of the known correct answers are in that top 10.

Run:  uv run python 02_retrieval/run.py
"""

import re
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

K = 10
SEED = 42
CACHE = Path(__file__).parent / "cache"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # best quality-for-speed model from 01
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge's instruction


# ---------- 1. Data ----------
def load_fiqa():
    corpus = load_dataset("mteb/fiqa", "corpus", split="corpus")
    queries = load_dataset("mteb/fiqa", "queries", split="queries")
    qrels = load_dataset("mteb/fiqa", "default", split="test")

    relevant = {}  # question id -> set of correct post ids
    for row in qrels:
        relevant.setdefault(row["query-id"], set()).add(row["corpus-id"])
    question_text = {row["_id"]: row["text"] for row in queries}
    qids = sorted(relevant)
    return corpus["_id"], corpus["text"], qids, [question_text[q] for q in qids], relevant


def top_k(scores):
    """Indices of the K highest scores, best first."""
    best = np.argpartition(-scores, K)[:K]
    return best[np.argsort(-scores[best])]


# ---------- 2a. Keyword search: BM25 ----------
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_search(docs, questions):
    bm25 = BM25Okapi([tokenize(d) for d in docs])
    start = time.perf_counter()
    results = [top_k(bm25.get_scores(tokenize(q))) for q in questions]
    return results, (time.perf_counter() - start) / len(questions)


# ---------- 2b. Meaning search: dense embeddings ----------
def dense_search(docs, questions):
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    path = CACHE / f"{EMBED_MODEL.split('/')[-1]}_{len(docs)}.npy"
    if path.exists():
        doc_vecs = np.load(path)
    else:  # one-time cost: embed every post (about 15-20 minutes on a laptop CPU)
        doc_vecs = model.encode(
            docs, batch_size=32, normalize_embeddings=True, show_progress_bar=True
        )
        CACHE.mkdir(exist_ok=True)
        np.save(path, doc_vecs)

    start = time.perf_counter()
    q_vecs = model.encode([QUERY_PREFIX + q for q in questions], normalize_embeddings=True)
    results = [top_k(doc_vecs @ qv) for qv in q_vecs]  # cosine similarity (vectors normalized)
    return results, (time.perf_counter() - start) / len(questions)


# ---------- 3. Scoring ----------
def score(ranked_ids, correct):
    """recall@10, MRR@10 and nDCG@10 for one question."""
    hits = [doc_id in correct for doc_id in ranked_ids]
    recall = sum(hits) / len(correct)
    mrr = next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0)
    dcg = sum(1 / np.log2(rank + 1) for rank, hit in enumerate(hits, 1) if hit)
    ideal = sum(1 / np.log2(rank + 1) for rank in range(1, min(len(correct), K) + 1))
    return recall, mrr, dcg / ideal


def bootstrap(values, rounds=1000):
    """95% CI of the mean, redrawing questions with repeats (same draws for every input)."""
    rng = np.random.default_rng(SEED)
    n = len(values)
    means = [values[rng.integers(0, n, n)].mean() for _ in range(rounds)]
    return np.percentile(means, [2.5, 97.5])


def main():
    doc_ids, docs, qids, questions, relevant = load_fiqa()
    print(f"{len(docs):,} posts, {len(questions)} test questions\n")

    runs = {}
    print("> BM25")
    runs["BM25"] = bm25_search(docs, questions)
    print("> dense (bge-small)")
    runs["dense"] = dense_search(docs, questions)

    per_q = {}  # method -> array of shape (questions, 3 metrics)
    print(f"\n| Method | recall@{K} | MRR@{K} | nDCG@{K} | ms / question |")
    print("|---|---|---|---|---|")
    for name, (results, sec) in runs.items():
        per_q[name] = np.array(
            [
                score([doc_ids[i] for i in res], relevant[qid])
                for res, qid in zip(results, qids, strict=True)
            ]
        )
        r, m, n = per_q[name].mean(axis=0)
        print(f"| {name} | {r:.3f} | {m:.3f} | {n:.3f} | {sec * 1000:.0f} |")

    print("\n95% CIs (questions redrawn 1,000 times):")
    for col, metric in enumerate(["recall@10", "MRR@10", "nDCG@10"]):
        cis = {name: bootstrap(per_q[name][:, col]) for name in per_q}
        diff = per_q["dense"][:, col] - per_q["BM25"][:, col]
        lo, hi = bootstrap(diff)
        verdict = "real" if lo > 0 or hi < 0 else "could be luck"
        print(
            f"  {metric:<9}  BM25 [{cis['BM25'][0]:.3f}, {cis['BM25'][1]:.3f}]"
            f"  dense [{cis['dense'][0]:.3f}, {cis['dense'][1]:.3f}]"
            f"  dense - BM25: {diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] -> {verdict}"
        )

    # Where does each method fail? "found" = at least one correct post in the top 10.
    found = {name: per_q[name][:, 0] > 0 for name in per_q}
    both = (found["BM25"] & found["dense"]).sum()
    neither = (~found["BM25"] & ~found["dense"]).sum()
    only_bm25 = np.where(found["BM25"] & ~found["dense"])[0]
    only_dense = np.where(~found["BM25"] & found["dense"])[0]
    print(
        f"\nFound a correct post in the top {K}: both {both}, only BM25 {len(only_bm25)}, "
        f"only dense {len(only_dense)}, neither {neither}"
    )

    text_of = dict(zip(doc_ids, docs, strict=True))
    for title, idxs in [("BM25 wins", only_bm25), ("dense wins", only_dense)]:
        print(f"\n--- {title} (first 4) ---")
        for i in idxs[:4]:
            answer = text_of[sorted(relevant[qids[i]])[0]]
            print(f"Q: {questions[i]}")
            print(f"   correct: {answer[:160].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()
