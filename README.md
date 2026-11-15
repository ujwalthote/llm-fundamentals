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
| 02 | Retrieval | Keyword (BM25) vs dense search: where does each fail? | BEIR FiQA | next |
| 03 | Chunking | How does chunk size change retrieval? | SQuAD | planned |
| 04 | Reranking | How much does a cross-encoder add, at what latency? | BEIR FiQA | planned |
| 05 | Generation | Does the LLM answer correctly and return valid JSON? | SQuAD | planned |
| 06 | LLM-as-judge | Does an LLM judge agree with human experts? | MT-Bench human judgments | planned |

**Headline from 01:** a 33M-parameter model on a laptop reaches about 97% of the Gemini embedding
API's quality, and above 110M parameters the models are statistically tied.

## Principles

- **Free to run:** local open models, plus free API tiers.
- **Readable:** one short script per experiment, top to bottom.
- **Honest numbers:** fixed samples, cached results, and paired bootstrap confidence intervals.

## Setup

```bash
uv sync                  # Python 3.12, CPU-only PyTorch
uv run ruff check .      # lint
```

## License

MIT for the code. Datasets and models keep their original licenses.
