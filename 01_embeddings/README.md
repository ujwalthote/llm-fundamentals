# 01 · Embeddings: do they capture meaning the way humans do?

## What is an embedding?

An embedding model turns a sentence into a list of numbers (a vector). The model is useful when
sentences that **mean** the same thing get similar vectors, even if they use different words:

| | Sentence A | Sentence B | Humans say |
|---|---|---|---|
| Different words, same meaning | A man is **cutting up** a cucumber. | A man is **slicing** a cucumber. | very similar |
| Same words, different meaning | A man is **speaking**. | A man is **cooking**. | not similar |

Keyword search fails the first case and is fooled by the second. Embeddings are what retrieval,
semantic caching, deduplication and clustering are built on, so if this stage is weak, nothing
later in an LLM pipeline can recover.

## How we measured it

- **Data:** [STS-B](https://huggingface.co/datasets/sentence-transformers/stsb) test set. Every
  sentence pair has a human similarity score from 0 (unrelated) to 1 (same meaning). We use a fixed
  random sample of **400 pairs** (seed 42), so every model sees exactly the same pairs and the
  Gemini free tier is enough to run them.
- **Model score:** cosine similarity between the two sentence vectors.
- **Metric:** **Spearman correlation** between model scores and human scores. It asks one
  question: *does the model put the pairs in the same order as humans?* 1.0 means the same order,
  0 means no relation.
- **Uncertainty:** a paired bootstrap (1,000 resamples of the 400 pairs, every model scored on the
  same resample). This tells us whether a gap between two models is real or luck of the sample.
- **Speed:** sentences per second on a laptop CPU (Intel i7-11370H), timing only the encoding, after
  a warm-up batch.

## Results

| Model | Runs | Params | Vector size | Spearman | 95% CI | Sentences/s |
|---|---|---|---|---|---|---|
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | laptop | 22M | 384 | 0.825 | 0.781 – 0.861 | **270** |
| [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | laptop | 33M | 384 | 0.877 | 0.847 – 0.900 | 135 |
| [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | laptop | 110M | 768 | 0.894 | 0.868 – 0.912 | 42 |
| [EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m) | laptop | 300M | 768 | 0.898 | 0.869 – 0.918 | 19 |
| Gemini `gemini-embedding-001` | Google API | not published | 3072 | **0.903** | 0.874 – 0.922 | n/a¹ |

¹ On the free tier the speed is set by the rate limit (100 texts per minute), not by the model.

**How to read the columns**

- **95% CI (confidence interval):** the range where the model's true score probably lies. We
  tested on 400 pairs, and a different 400 would give a slightly different score. To estimate how
  much, we redraw 400 pairs from our sample (allowing repeats) 1,000 times and score the model on
  each draw. The middle 95% of those 1,000 scores is the CI. A narrow range means a stable number.
  If two models' difference has a CI that includes 0, we can't say which one is better.
- **Sentences/s:** how many sentences the model turns into vectors per second (775 sentences ÷
  seconds taken), on a laptop CPU, after a warm-up batch. It decides indexing time (at 270/s,
  1 million documents take about 1 hour; at 19/s, about 15 hours) and the delay added to every
  user query. Treat it as relative: a GPU would be 10–50× faster for every model.

**Is the bigger model really better?** (paired difference, 95% CI)

| Comparison | Gain | 95% CI | Verdict |
|---|---|---|---|
| bge-small vs MiniLM | +0.052 | +0.024 – +0.084 | real |
| bge-base vs bge-small | +0.017 | +0.006 – +0.028 | real |
| EmbeddingGemma vs bge-base | +0.004 | −0.014 – +0.019 | could be luck |
| Gemini vs EmbeddingGemma | +0.005 | −0.009 – +0.018 | could be luck |
| Gemini vs bge-small | +0.025 | +0.008 – +0.042 | real |

## What I learned

1. **Size helps, until it doesn't.** Going from 22M to 110M parameters gave real, measurable gains.
   Above that, bge-base, EmbeddingGemma and the Gemini API are statistically tied on this task, even
   though Gemini's vectors are 4× larger.
2. **A 33M model gets about 97% of the API's quality.** bge-small runs free on a laptop at 135
   sentences/s and keeps data local. Gemini's lead over it is real (+0.025) but small. Whether that
   is worth an API call, network latency and sending data to a third party depends on the product.
3. **Compare models on the same examples.** The separate confidence intervals of bge-small and
   Gemini overlap, which would suggest "no difference". The paired comparison shows the gap is
   real. Without pairing, you can easily draw the wrong conclusion.
4. **Gold labels are noisy too.** Where Gemini disagrees most with humans, sometimes the model is
   wrong, and sometimes the label looks wrong:

   | Pair | Humans | Gemini | Comment |
   |---|---|---|---|
   | "World Cup live: France 0-0 Germany" vs "World Cup live: Germany 0-0 Ghana" | low | high | Model fooled by the shared template |
   | "It's also a matter of taste." vs "It's definitely just a matter of preference." | high | medium | Model misses the synonym |
   | "Three sheep on a dirt road viewed from a passenger vehicle." vs "… shot from a car." | medium | high | The human label looks low |

## Run it

```bash
uv sync
uv run python 01_embeddings/run.py
```

The Gemini model needs a free API key from [Google AI Studio](https://aistudio.google.com/) in a
`.env` file (`GEMINI_API_KEY=...`). Vectors are cached in `01_embeddings/cache/`, so the API is
called only once.

## Notes

- 400 of the 1,379 test pairs, to stay inside Gemini's free daily limit. Results on the full set
  may differ slightly.
- EmbeddingGemma is loaded from [`unsloth/embeddinggemma-300m`](https://huggingface.co/unsloth/embeddinggemma-300m),
  an ungated copy of Google's weights. The Gemma terms of use apply. It is run with its built-in
  `STS` prompt, and Gemini with `task_type=SEMANTIC_SIMILARITY`, as their authors recommend.
- Speeds are from a single run on one laptop. Treat them as relative, not absolute.
