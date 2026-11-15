"""02b - HNSW: how much speed do we gain, and how much accuracy do we lose?

Exact search compares the question with every post. HNSW builds a graph of nearby posts and
walks it, comparing with only a few hundred. We use the same cached bge-small vectors as run.py,
so the only thing that changes is the search method.

Run (after run.py has cached the vectors):  uv run python 02_retrieval/hnsw.py
"""

import time

import faiss
import numpy as np
from run import CACHE, EMBED_MODEL, QUERY_PREFIX, K, load_fiqa, score
from sentence_transformers import SentenceTransformer

M = 32  # graph edges per post: more edges = better recall, more memory
EF_CONSTRUCTION = 200  # how carefully the graph is built
EF_SEARCH = [8, 16, 32, 64, 128, 256]  # how many candidates a search explores

faiss.omp_set_num_threads(1)  # one question at a time on one core, like a single request


def timed_search(index, q_vecs):
    """Search one question at a time. Returns (top-10 ids per question, ms per question)."""
    start = time.perf_counter()
    results = [index.search(qv[None, :], K)[1][0] for qv in q_vecs]
    return results, (time.perf_counter() - start) / len(q_vecs) * 1000


def main():
    doc_ids, _, qids, questions, relevant = load_fiqa()
    doc_vecs = np.load(CACHE / f"{EMBED_MODEL.split('/')[-1]}_{len(doc_ids)}.npy")
    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    q_vecs = model.encode([QUERY_PREFIX + q for q in questions], normalize_embeddings=True)
    dims = doc_vecs.shape[1]
    print(f"{len(doc_ids):,} vectors of {dims} dims, {len(questions)} questions\n")

    def ndcg(results):
        return np.mean(
            [
                score([doc_ids[i] for i in r], relevant[q])[2]
                for r, q in zip(results, qids, strict=True)
            ]
        )

    # Exact search: inner product of normalized vectors = cosine similarity.
    exact = faiss.IndexFlatIP(dims)
    exact.add(doc_vecs)
    truth, exact_ms = timed_search(exact, q_vecs)

    # HNSW: build the graph once.
    hnsw = faiss.IndexHNSWFlat(dims, M, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = EF_CONSTRUCTION
    start = time.perf_counter()
    hnsw.add(doc_vecs)
    print(f"HNSW graph built in {time.perf_counter() - start:.1f} s (M={M})\n")

    print("| Search | ANN recall@10 | nDCG@10 | ms / question | Speed-up |")
    print("|---|---|---|---|---|")
    print(f"| exact | 1.000 | {ndcg(truth):.3f} | {exact_ms:.2f} | 1.0x |")
    for ef in EF_SEARCH:
        hnsw.hnsw.efSearch = ef
        found, ms = timed_search(hnsw, q_vecs)
        # ANN recall: share of the exact top 10 that HNSW also returned.
        overlap = np.mean([len(set(f) & set(t)) / K for f, t in zip(found, truth, strict=True)])
        print(
            f"| HNSW ef={ef} | {overlap:.3f} | {ndcg(found):.3f} | {ms:.2f} | "
            f"{exact_ms / ms:.1f}x |"
        )

    # Exact search cost grows with the number of posts; HNSW grows roughly with its log.
    # Show it by timing exact search on copies of the data (2x, 4x, 8x the posts).
    print("\nExact search time as the collection grows:")
    for times in (1, 2, 4, 8):
        big = faiss.IndexFlatIP(dims)
        for _ in range(times):
            big.add(doc_vecs)
        _, ms = timed_search(big, q_vecs[:200])
        print(f"  {len(doc_ids) * times:>9,} posts: {ms:.2f} ms / question")


if __name__ == "__main__":
    main()
