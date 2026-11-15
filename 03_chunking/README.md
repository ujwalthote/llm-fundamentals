# 03 · Chunking: how does the way we cut documents change retrieval?

## The question

Embedding models read a limited number of tokens (512 for bge-small), and one vector for a whole
5,000-word article would blur many topics together. So long documents are cut into **chunks**, and
each chunk gets its own vector. How big should chunks be, and does it matter *where* we cut?

## How we measured it

- **Data:** the [SQuAD v1.1](https://huggingface.co/datasets/rajpurkar/squad) validation set:
  48 Wikipedia articles (254k words, median 5,037 words per article). Each question has its answer
  marked at an exact position in the article. We use a fixed random sample of **2,000 questions**
  and search the chunks of **all 48 articles** together.
- **Five ways to chunk** (each article's paragraphs are first joined into one text, as if it came
  from a PDF):

| Chunking | Size | Overlap |
|---|---|---|
| 50 words | 50 words | 10 |
| 100 words | 100 words | 20 |
| 200 words | 200 words | 40 |
| 400 words | 400 words | 80 |
| paragraph | Wikipedia's own paragraphs (123 words on average) | none |

- **Two retrievers:** dense ([bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5))
  and BM25, as in [02](../02_retrieval/).
- **A chunk counts as correct only if it covers the answer's marked position.** A short answer
  such as "1893" can appear several times in an article; matching the text alone would count wrong
  chunks as hits.
- **Two ways to score:**
  - **hit@k:** is the answer in the top *k* chunks?
  - **Hit within 1,000 words:** take chunks from the top until 1,000 words are used (20 chunks of
    50 words, 5 of 200, 2 of 400), as if an LLM could read 1,000 words of context. This is the
    fair comparison: at a fixed *k*, bigger chunks simply contain more text.
- **95% CI:** paired bootstrap over questions (1,000 redraws), as in 01 and 02.

## Results

| Method | Chunking | Chunks | Over 512 tokens | hit@1 | hit@3 | hit@5 | Hit within 1,000 words (95% CI) |
|---|---|---|---|---|---|---|---|
| dense | 50 words | 6,357 | 0 | 0.573 | 0.778 | 0.832 | **0.928** (0.916 – 0.940) |
| dense | 100 words | 3,181 | 0 | 0.525 | 0.753 | 0.838 | 0.897 (0.884 – 0.911) |
| dense | 200 words | 1,599 | 0 | 0.475 | 0.740 | 0.809 | 0.809 (0.792 – 0.826) |
| dense | 400 words | 803 | **395** | 0.443 | 0.702 | 0.787 | 0.617 (0.595 – 0.638) |
| dense | paragraph | 2,067 | 9 | **0.665** | **0.834** | **0.880** | 0.907 (0.894 – 0.919) |
| BM25 | 50 words | 6,357 | – | 0.598 | 0.768 | 0.821 | 0.894 (0.880 – 0.907) |
| BM25 | 100 words | 3,181 | – | 0.697 | 0.843 | 0.876 | 0.914 (0.901 – 0.925) |
| BM25 | 200 words | 1,599 | – | 0.726 | 0.875 | 0.907 | 0.907 (0.895 – 0.920) |
| BM25 | 400 words | 803 | – | **0.753** | **0.891** | **0.930** | 0.855 (0.839 – 0.869) |
| BM25 | paragraph | 2,067 | – | 0.745 | 0.870 | 0.897 | **0.914** (0.901 – 0.926) |

Every difference discussed below is real: the paired 95% CIs exclude zero (see the script's output).

## What I learned

1. **Comparing chunk sizes at a fixed k is misleading.** BM25's hit@5 rises steadily with chunk
   size (0.82 → 0.93), which suggests "bigger is better". But 5 chunks of 400 words are 2,000 words
   of text, against 250 for 50-word chunks. At a fixed 1,000-word budget, 400-word chunks are BM25's
   **worst** setting (0.855). Bigger chunks look better only because they contain more text.
2. **Dense retrieval collapses when chunks exceed the model's limit.** 395 of the 803 400-word
   chunks (49%) are longer than bge-small's 512 tokens, and the model silently ignores the rest.
   Dense drops to 0.617 within 1,000 words, against 0.928 for 50-word chunks. BM25 reads the whole
   chunk, so it does not collapse. **Chunk size must fit the embedding model.**
3. **Cutting at natural boundaries is a large win.** Paragraphs (123 words on average) beat fixed
   100-word windows by **+0.14 dense hit@1** (0.665 vs 0.525) at about the same size. Fixed windows
   cut across paragraphs and mix two topics into one vector; a paragraph keeps one idea per chunk.
   In practice: split at headings, paragraphs and sections, not every N tokens.
4. **Dense prefers small chunks, BM25 prefers medium ones.** Within 1,000 words, dense is best at
   50 words (0.928): one small idea gives a sharp vector. BM25 is best from 100 words to paragraph
   size (0.914): it needs enough words to match on.
5. **On SQuAD, BM25 beats dense, the opposite of [02](../02_retrieval/).** SQuAD's questions were
   written by people reading the paragraph, so they reuse its exact words (dense hit@1 0.475 vs
   BM25 0.726 at 200 words). FiQA's questions are real user questions in their own words, and there
   dense won by +0.19 nDCG. **The best retriever depends on the data, so measure on your own data.**

**Not measured here:** whether an LLM answers better from twenty 50-word fragments or a few whole
paragraphs. Retrieval finding the answer is necessary, not sufficient. That is the subject of 05.

## Run it

```bash
uv sync
uv run python 03_chunking/run.py
```

About 15 minutes on a laptop CPU, almost all of it embedding the chunks (cached in
`03_chunking/cache/`).

## Notes

- Words are whitespace-separated tokens; chunk boundaries are exact character positions in the
  joined article text.
- The answer must fit entirely inside one chunk. Overlap (20% of the chunk) makes it rare for an
  answer to be split across two chunks.
- Dense uses bge's query prefix, as in 02. BM25 uses `rank-bm25` defaults, lowercase alphanumeric
  tokens, no stemming.
- SQuAD questions overlap heavily with their paragraph's wording, so absolute numbers here are
  higher than you would see with real user questions.
