# LLM fundamentals, measured

Small experiments, one per stage of an LLM pipeline. Each one uses a public dataset with human
(gold) labels, so every claim comes with a number and a confidence interval.

```
question -> [embed] -> [retrieve] -> [rerank] -> [generate] -> [judge]
              01          02            04           05           06
```

| # | Stage | Question | Dataset | Status |
|---|---|---|---|---|
| [01](01_embeddings/) | Embeddings | Do embedding models capture meaning the way humans do? | STS-B | done |
| [02](02_retrieval/) | Retrieval | Keyword (BM25) vs dense search: where does each fail? | BEIR FiQA | done |
| [03](03_chunking/) | Chunking | How does chunk size change retrieval? | SQuAD | done |
| [04](04_reranking/) | Reranking | How much does a cross-encoder add, at what latency? | BEIR FiQA | done |
| [05](05_generation/) | Generation | Does the LLM answer correctly and return valid JSON? | SQuAD | done |
| [06](06_judge/) | LLM-as-judge | Does an LLM judge agree with human experts? | MT-Bench human judgments | done |

**Headlines so far**

- **01:** a 33M-parameter model on a laptop reaches about 97% of the Gemini embedding API's
  quality, and above 110M parameters the models are statistically tied.
- **02:** dense search beats BM25 by +0.19 nDCG@10 on finance questions (matching the published
  score exactly), but BM25 still solves 25 questions that dense misses, and 29% are missed by both.
  HNSW at ef=64 keeps 97.5% of the exact top 10 at 23× the speed.
- **03:** at a fixed context budget, paragraph and small chunks win; 400-word chunks collapse for
  dense search because half of them exceed the model's 512-token limit. Comparing chunk sizes at a
  fixed top-k is misleading.
- **04:** standard MS MARCO cross-encoders did *not* improve dense search on finance questions
  (−0.018 nDCG at depth 30) while adding 0.5 to 4.4 s per question on CPU. The same reranker lifts
  BM25 by +0.077, so a reranker only helps when it is a better judge than the retriever.
- **05:** gpt-oss returned valid JSON 480 of 480 times. Given the right paragraph it answers 92%
  exactly; through real retrieval, 75%. With the answer removed it invented one anyway 15 to 25% of
  the time, using real names and numbers from the wrong part of the context.
- **06:** a free Gemini Flash Lite judge agrees with MT-Bench experts 61% of the time (experts agree
  with each other 64%, GPT-4 with them 67%), but every judge flips its verdict on 16 to 33% of pairs
  when only the answer order changes. Re-grading 05 with it by meaning instead of spelling lifts
  gpt-oss-120b from 75% to 88% through real retrieval.

## Principles

- **Free to run:** local open models, plus free API tiers.
- **Readable:** one short script per experiment, top to bottom.
- **Honest numbers:** fixed samples, cached results, and paired bootstrap confidence intervals.

## How this was built

Step by step with [Claude Code](https://claude.com/claude-code) as a pair programmer. Each
experiment was worked through with small examples first, then run, checked against published
results where they exist, and reviewed before it was committed. Commits carry a co-author line to
show this.

## Setup

```bash
uv sync                  # Python 3.12, CPU-only PyTorch
uv run ruff check .      # lint
```

## License

MIT for the code. Datasets and models keep their original licenses.
