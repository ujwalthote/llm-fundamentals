"""03 - Chunking: how does the way we cut documents change retrieval?

Each SQuAD Wikipedia article (median 5,000 words) is cut into chunks in several ways. For a
sample of questions we search all chunks of all 48 articles and check whether the retrieved
chunks contain the answer at its marked position (not just the same words somewhere else).

Run:  uv run python 03_chunking/run.py
"""

import re
from pathlib import Path

import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

N_QUESTIONS = 2000
SEED = 42
KS = (1, 3, 5)
WORD_BUDGET = 1000  # how many words of context we could give an LLM
CACHE = Path(__file__).parent / "cache"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# name -> (words per chunk, words of overlap); None = split at paragraph breaks
CHUNKERS = {
    "50 words": (50, 10),
    "100 words": (100, 20),
    "200 words": (200, 40),
    "400 words": (400, 80),
    "paragraph": None,
}


# ---------- 1. Data: whole articles, with each answer's exact character position ----------
def load_squad():
    ds = load_dataset("rajpurkar/squad", split="validation")
    articles = {}  # title -> list of paragraphs, in order
    for row in ds:
        paragraphs = articles.setdefault(row["title"], [])
        if not paragraphs or paragraphs[-1] != row["context"]:
            paragraphs.append(row["context"])

    texts, para_spans, para_start = {}, {}, {}
    for title, paragraphs in articles.items():
        spans, pos = [], 0
        for p in paragraphs:
            spans.append((pos, pos + len(p)))
            para_start[(title, p)] = pos
            pos += len(p) + 2  # paragraphs joined with a blank line
        texts[title] = "\n\n".join(paragraphs)
        para_spans[title] = spans

    rng = np.random.default_rng(SEED)
    questions = []  # (question, title, answer start, answer end) in article coordinates
    for i in sorted(rng.choice(len(ds), N_QUESTIONS, replace=False)):
        row = ds[int(i)]
        start = para_start[(row["title"], row["context"])] + row["answers"]["answer_start"][0]
        questions.append(
            (row["question"], row["title"], start, start + len(row["answers"]["text"][0]))
        )
    return texts, para_spans, questions


# ---------- 2. Chunking: every chunk is (article, start char, end char) ----------
def chunk_by_words(title, text, size, overlap):
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    step = size - overlap
    chunks = []
    for first in range(0, len(words), step):
        last = min(first + size, len(words)) - 1
        chunks.append((title, words[first][0], words[last][1]))
        if last == len(words) - 1:
            break
    return chunks


def make_chunks(name, texts, para_spans):
    chunks = []
    for title, text in texts.items():
        if CHUNKERS[name] is None:
            chunks += [(title, s, e) for s, e in para_spans[title]]
        else:
            chunks += chunk_by_words(title, text, *CHUNKERS[name])
    return chunks


# ---------- 3. Search ----------
def dense_rankings(name, chunk_texts, questions, model):
    path = CACHE / f"{name.replace(' ', '')}_{len(chunk_texts)}.npy"
    if path.exists():
        vecs = np.load(path)
    else:
        vecs = model.encode(chunk_texts, batch_size=32, normalize_embeddings=True)
        CACHE.mkdir(exist_ok=True)
        np.save(path, vecs)
    q_vecs = model.encode([QUERY_PREFIX + q[0] for q in questions], normalize_embeddings=True)
    return [np.argsort(-(vecs @ qv))[:50] for qv in q_vecs]


def bm25_rankings(chunk_texts, questions):
    def tokenize(t):
        return re.findall(r"[a-z0-9]+", t.lower())

    bm25 = BM25Okapi([tokenize(c) for c in chunk_texts])
    return [np.argsort(-bm25.get_scores(tokenize(q[0])))[:50] for q in questions]


# ---------- 4. Scoring ----------
def evaluate(ranking, chunks, chunk_words, question):
    """Per question: hit@k for each k, and hit within the word budget."""
    _, title, a_start, a_end = question
    hits = [
        chunks[i][0] == title and chunks[i][1] <= a_start and a_end <= chunks[i][2] for i in ranking
    ]
    row = [float(any(hits[:k])) for k in KS]
    used, found = 0, False
    for i, hit in zip(ranking, hits, strict=True):
        used += chunk_words[i]
        if used > WORD_BUDGET and used != chunk_words[i]:  # always allow at least one chunk
            break
        found = found or hit
    return row + [float(found)]


def bootstrap(values, rounds=1000):
    rng = np.random.default_rng(SEED)
    n = len(values)
    return np.percentile([values[rng.integers(0, n, n)].mean() for _ in range(rounds)], [2.5, 97.5])


def main():
    texts, para_spans, questions = load_squad()
    total_words = sum(len(t.split()) for t in texts.values())
    print(f"{len(texts)} articles, {total_words:,} words, {len(questions)} sampled questions\n")
    model = SentenceTransformer(EMBED_MODEL, device="cpu")

    results = {}  # (method, chunker) -> per-question array
    stats = {}
    for name in CHUNKERS:
        chunks = make_chunks(name, texts, para_spans)
        chunk_texts = [texts[t][s:e] for t, s, e in chunks]
        chunk_words = [len(c.split()) for c in chunk_texts]
        too_long = sum(len(ids) > 512 for ids in model.tokenizer(chunk_texts)["input_ids"])
        stats[name] = (len(chunks), np.mean(chunk_words), too_long)
        print(f"> {name}: {len(chunks):,} chunks, avg {np.mean(chunk_words):.0f} words")
        for method in ("dense", "BM25"):
            if method == "dense":
                rankings = dense_rankings(name, chunk_texts, questions, model)
            else:
                rankings = bm25_rankings(chunk_texts, questions)
            results[(method, name)] = np.array(
                [
                    evaluate(r, chunks, chunk_words, q)
                    for r, q in zip(rankings, questions, strict=True)
                ]
            )

    header = " | ".join(f"hit@{k}" for k in KS)
    print(
        f"\n| Method | Chunking | Chunks | Avg words | Cut at 512 tokens | {header} | "
        f"hit within {WORD_BUDGET} words (95% CI) |"
    )
    print("|---|---|---|---|---|" + "---|" * len(KS) + "---|")
    for (method, name), per_q in results.items():
        n, avg, too_long = stats[name]
        hits = " | ".join(f"{v:.3f}" for v in per_q[:, : len(KS)].mean(axis=0))
        lo, hi = bootstrap(per_q[:, -1])
        print(
            f"| {method} | {name} | {n:,} | {avg:.0f} | {too_long} | {hits} | "
            f"{per_q[:, -1].mean():.3f} ({lo:.3f} - {hi:.3f}) |"
        )

    # Paired comparisons against 200-word chunks, at a fixed word budget.
    print(f"\nPaired difference vs dense 200 words, hit within {WORD_BUDGET} words:")
    base = results[("dense", "200 words")][:, -1]
    for (method, name), per_q in results.items():
        if (method, name) == ("dense", "200 words"):
            continue
        diff = per_q[:, -1] - base
        lo, hi = bootstrap(diff)
        verdict = "real" if lo > 0 or hi < 0 else "could be luck"
        print(f"  {method} {name:<10} {diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] -> {verdict}")


if __name__ == "__main__":
    main()
