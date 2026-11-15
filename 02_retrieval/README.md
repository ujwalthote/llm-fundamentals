# 02 · Retrieval: keyword search (BM25) vs meaning search (dense)

## The question

In RAG, the LLM can only answer from what retrieval finds. So: given a question, which method
finds the correct documents more often, and where does each one fail?

- **BM25 (keyword search):** scores a document by the question words it contains. Rare words
  ("SEC", "401k") are worth much more than common ones ("should", "a"), repeats help less and less,
  and long documents get a small penalty. It knows exact words only: "cheque" and "check" are
  unrelated to it.
- **Dense (meaning search):** embeds the question and every document with
  [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (the best quality-for-speed
  model in [01](../01_embeddings/)) and ranks documents by cosine similarity.

## How we measured it

- **Data:** [FiQA](https://huggingface.co/datasets/mteb/fiqa) from the BEIR benchmark. 648 test
  questions from a personal-finance forum, searched against **57,638 forum posts**. Each question
  has 1 to 15 posts marked as correct answers (2.6 on average).
- **Each method returns its top 10 posts** for every question, scored with three metrics:

| Metric | What it asks | Example: 3 correct posts, found at ranks 2 and 7 |
|---|---|---|
| **recall@10** | What share of the correct posts are in the top 10? | 2 of 3 = 0.67 |
| **MRR@10** | How high is the *first* correct post? (1 / its rank) | 1/2 = 0.50 |
| **nDCG@10** | How close is the ranking to the perfect ranking? | 0.45 |

**nDCG in one minute:** each position in the top 10 is worth points, and higher positions are worth
more (rank 1 = 1.00, rank 2 = 0.63, rank 3 = 0.50 … rank 10 = 0.29), because people read from the
top. Add up the points where correct posts appear, then divide by the best possible score (all
correct posts at the very top). The result is a mark out of 1. Recall only counts *what* was found;
nDCG also rewards finding it *higher*. It is BEIR's official metric, so we can compare with
published results.

- **95% CI:** paired bootstrap over questions (1,000 redraws), as in 01.
- **Speed:** milliseconds per question on a laptop CPU (Intel i7-11370H, 4 cores).

## Results

| Method | recall@10 | MRR@10 | nDCG@10 | ms / question |
|---|---|---|---|---|
| BM25 | 0.278 | 0.271 | 0.218 | 323 |
| **Dense (bge-small)** | **0.464** | **0.488** | **0.403** | **8** |
| Dense − BM25 (95% CI) | +0.185 (+0.156 – +0.214) | +0.217 (+0.187 – +0.246) | +0.186 (+0.162 – +0.211) | |

All three differences are real: every confidence interval is far above zero.

**Cross-check against published numbers**

| | Ours | Published | |
|---|---|---|---|
| bge-small nDCG@10 | 0.403 | 0.403 ([model card](https://huggingface.co/BAAI/bge-small-en-v1.5), MTEB FiQA2018: 40.34) | exact match |
| BM25 nDCG@10 | 0.218 | 0.236 (BEIR paper, Elasticsearch BM25) | close; ours has no stemming ("investing" ≠ "invest") |

**Which questions does each method solve?** (a correct post in the top 10)

| Both | Only dense | Only BM25 | Neither |
|---|---|---|---|
| 266 | 171 | 25 | 186 |

## What I learned

1. **Dense search wins clearly on this data**, by about +0.19 nDCG@10. Forum questions are usually
   about an idea ("Why diversify stocks?"), and the answers use different words for it.
2. **BM25 still solves questions dense misses.** It wins on specific names and terms that appear
   word for word in the answer: *"credit union … BOA, Chase"*, *"Physical Checks - Mailing"*,
   *"tax software … filing status"*. That is the argument for **hybrid search**: the two methods
   fail on different questions.
3. **29% of questions are missed by both.** Even the better method finds no correct post in the
   top 10 for 186 of 648 questions. Retrieval is often the weakest stage of a RAG system, and no
   prompt can fix a document that was never retrieved.
4. **The speed gap is about implementation, not the method.** Our BM25 (`rank-bm25`, pure Python)
   scores all 57,638 posts for every question: 323 ms. Production BM25 (Elasticsearch, OpenSearch,
   Postgres full-text) uses an **inverted index** (word → posts that contain it) and only scores
   posts that share a word with the question, in a few milliseconds. Dense was fast here because
   it is one matrix multiply; at millions of documents it would need an approximate index such as
   HNSW.
5. **Embedding is the expensive part.** Embedding the 57,638 posts once took about **50 minutes**
   on a 4-core laptop CPU, with all cores busy. The vectors are cached, so later runs take seconds.
   At that point the limit is the hardware: a GPU would do it in minutes.

## Add-on: HNSW, speed vs accuracy

Exact search compares the question with all 57,638 posts. **HNSW** (the index behind pgvector,
FAISS, Qdrant and most vector databases) links each post to its nearest neighbours in a graph and
walks that graph toward the question, comparing with only a few hundred posts. It is
**approximate**: the walk can miss some of the true top 10. It never makes results more relevant
than exact search; it makes them faster.

[`hnsw.py`](hnsw.py) uses the same cached vectors with FAISS (`IndexHNSWFlat`, M = 32,
efConstruction = 200), one question at a time on one CPU core. **efSearch** is how many candidates
the walk keeps exploring: a dial between speed and accuracy that can be changed per query.
**ANN recall@10** is the share of the exact top 10 that HNSW also returns.

| Search | ANN recall@10 | nDCG@10 | ms / question | Speed-up |
|---|---|---|---|---|
| Exact | 1.000 | 0.403 | 5.60 | 1× |
| HNSW ef=8 | 0.758 | 0.352 | 0.07 | 75× |
| HNSW ef=16 | 0.875 | 0.384 | 0.09 | 64× |
| HNSW ef=32 | 0.940 | 0.393 | 0.15 | 38× |
| **HNSW ef=64** | **0.975** | **0.396** | **0.25** | **23×** |
| HNSW ef=128 | 0.990 | 0.400 | 0.44 | 13× |
| HNSW ef=256 | 0.996 | 0.402 | 0.81 | 7× |

Building the graph took 33 s, once. Exact search time grows in line with the collection: 5.5 ms at
57k posts, 11 ms at 115k, 22 ms at 231k, 43 ms at 461k (timed on copies of the data), so about
1 second per question at 10 million. HNSW grows roughly with the logarithm of the size.

**What I learned**

1. **HNSW approaches exact search but never beats it** (nDCG 0.352 → 0.402, vs 0.403 exact).
2. **ef = 64 is the sweet spot here:** 97.5% of the exact top 10, −0.007 nDCG, 23× faster.
3. **A too-small ef silently hurts quality.** At ef = 8, nDCG drops by 0.05, about as much as
   switching to a much weaker embedding model. pgvector's default `hnsw.ef_search` is 40, so it is
   worth measuring ANN recall instead of trusting defaults.
4. **Under about 100k documents, exact search is fast enough** (single-digit ms). The index pays
   off from a few hundred thousand documents upward.

## Run it

```bash
uv sync
uv run python 02_retrieval/run.py    # BM25 vs dense
uv run python 02_retrieval/hnsw.py   # HNSW vs exact (needs the vectors cached by run.py)
```

The first run embeds all posts (about 50 minutes on a laptop CPU) and caches them in
`02_retrieval/cache/`. Later runs take about 4 minutes, almost all of it BM25.

## Notes

- BM25 uses `rank-bm25`'s `BM25Okapi` with default settings (k1 = 1.5, b = 0.75), lowercase
  alphanumeric tokens, and no stemming or stop-word removal.
- Dense uses bge's recommended query prefix: *"Represent this sentence for searching relevant
  passages: "*. Documents are embedded without a prefix, up to 512 tokens (4% of posts are longer
  and get cut).
- All marked answers in FiQA's test set count equally (binary relevance).
