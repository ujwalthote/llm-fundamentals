# 04 · Reranking: does a cross-encoder improve dense search?

## The question

The dense search in [02](../02_retrieval/) is a **bi-encoder**: the question and each post are
embedded *separately* into 384 numbers, then compared. That is fast (post vectors are computed
once), but the model never sees the question and the post together.

A **cross-encoder** reads them *together* as one input and outputs a single relevance score:

```
[CLS] Is it worth paying off my mortgage early? [SEP] Pay down the loan if the rate is > 5% ... [SEP]
                                      -> score 7.3
```

Every question word can look at every post word, so it is usually more accurate, but nothing can be
precomputed: every (question, post) pair is a full model run. So production systems
**retrieve, then rerank**: a fast retriever picks a few dozen candidates, the cross-encoder reorders
them, and the top 10 go to the LLM.

Two questions: **how much does reranking add, and what does it cost?**

## How we measured it

- **Data:** FiQA, the same 648 test questions and 57,638 posts as [02](../02_retrieval/).
- **First stage:** dense search with bge-small (nDCG@10 = 0.403, from 02).
- **Rerankers:** [ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
  (22M parameters) and [ms-marco-MiniLM-L12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2)
  (33M), the standard open cross-encoders, trained on MS MARCO (Bing web searches). Posts are read
  up to 512 tokens.
- **Depth:** the reranker reorders dense's top 10, 20 or 30 candidates.
- **Ceiling:** a reranker can only reorder what it is given. The share of correct posts that are
  among the candidates at all is the best recall it could reach.
- **Rescued / lost:** questions with no correct post in dense's top 10 that have one after
  reranking, and the reverse.
- **Latency:** milliseconds to rerank one question's candidates, one question at a time, on a laptop
  CPU (Intel i7-11370H, 4 cores).
- **95% CI:** paired bootstrap over questions (1,000 redraws), as in 01 to 03.

## Results

**Ceiling:** share of correct posts in dense's top *n*

| top 10 | top 20 | top 30 | top 50 | top 100 |
|---|---|---|---|---|
| 0.464 | 0.543 | 0.571 | 0.621 | 0.696 |

**Reranking dense's candidates**

| Reranker | Depth | recall@10 | MRR@10 | nDCG@10 | nDCG vs dense (95% CI) | Rescued | Lost | ms / question |
|---|---|---|---|---|---|---|---|---|
| **dense only** | – | 0.464 | **0.488** | **0.403** | – | – | – | **8** |
| MiniLM-L6 | 10 | 0.464 | 0.476 | 0.398 | −0.005 (−0.016 – +0.007) | 0 | 0 | 502 |
| MiniLM-L6 | 20 | 0.470 | 0.465 | 0.394 | −0.010 (−0.024 – +0.005) | 26 | 27 | 1,157 |
| MiniLM-L6 | 30 | 0.462 | 0.455 | 0.386 | **−0.018 (−0.033 – −0.003)** | 30 | 36 | 1,781 |
| MiniLM-L12 | 10 | 0.464 | 0.480 | 0.400 | −0.003 (−0.014 – +0.008) | 0 | 0 | 1,078 |
| MiniLM-L12 | 20 | 0.474 | 0.468 | 0.397 | −0.006 (−0.021 – +0.008) | 26 | 22 | 2,338 |
| MiniLM-L12 | 30 | 0.470 | 0.460 | 0.392 | −0.012 (−0.027 – +0.003) | 28 | 29 | 4,419 |

**Cross-check: the same reranker on BM25's candidates**

| | nDCG@10 |
|---|---|
| BM25 top 30 | 0.218 |
| BM25 top 30, reranked by MiniLM-L6 | **0.294** (+0.077, 95% CI +0.061 – +0.093) |
| Published (BEIR paper: BM25 top 100, reranked) | 0.236 → 0.347 (+0.11) |

## What I learned

1. **Reranking did not help dense search here; at depth 30 it made it worse.** Every reranked
   setting scored below dense alone, and the L6 loss at depth 30 is real (the CI excludes zero).
   Deeper reranking brought in more correct posts (26–30 questions rescued) but pushed out about as
   many (22–36 lost).
2. **A reranker only helps if it is a better judge than your retriever.** These cross-encoders were
   trained on web searches; on finance forum posts they are out of their domain. bge-small was
   trained on a much broader mix and already orders these posts better.
3. **The code is not the problem.** The same reranker lifts BM25 from 0.218 to 0.294, in line with
   the published +0.11 (which reranked 100 candidates instead of 30). Rerankers rescue a *weak* first
   stage. Even reranked, BM25 (0.294) stays well below plain dense (0.403).
4. **Reranking is expensive on a CPU.** One candidate costs about 50 ms (L6) to 150 ms (L12).
   Reranking 30 candidates adds 1.8 to 4.4 seconds per question, against 8 ms for the whole dense
   search. In production, rerankers run on GPUs or behind a short candidate list.
5. **"Add a reranker" is not a free win: measure it on your data.** A stronger reranker trained on
   broader data (bge-reranker-v2, Cohere Rerank, an LLM as reranker) may well help, but they are too
   slow on a laptop CPU or paid, so they are not tested here.

## Run it

```bash
uv sync
uv run python 02_retrieval/run.py    # once, to cache the post vectors
uv run python 04_reranking/run.py
```

About 70 minutes on a laptop CPU the first time, almost all of it running the cross-encoders.
Scores are cached in `04_reranking/cache/`, so later runs take about 15 minutes (the latency
measurement and BM25).

## Notes

- Larger rerankers were planned (bge-reranker-base, 278M parameters), but at the measured speed it
  would take about 7 hours on this CPU, so MiniLM-L12 is used instead.
- Speed-ups tried and not used: sorting pairs by length (no gain), cutting posts to 256 tokens
  (17 to 36% faster, but changes what the model reads), and ONNX / OpenVINO (incompatible with the
  current sentence-transformers release).
- All marked answers in FiQA's test set count equally (binary relevance), as in 02.
