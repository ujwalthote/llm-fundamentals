"""01 - Do embedding models capture meaning the way humans do?

For each sentence pair in STS-B, humans gave a similarity score (0 = unrelated,
1 = same meaning). We embed both sentences, take the cosine similarity, and check
whether the model ranks the pairs in the same order as the humans (Spearman).

Run:  uv run python 01_embeddings/run.py
"""

import hashlib
import os
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer

N_PAIRS = 400  # same fixed sample for every model (keeps Gemini inside its free daily limit)
SEED = 42
CACHE = Path(__file__).parent / "cache"

# (label, model id, where it runs)
MODELS = [
    ("MiniLM-L6", "sentence-transformers/all-MiniLM-L6-v2", "local"),
    ("bge-small", "BAAI/bge-small-en-v1.5", "local"),
    ("bge-base", "BAAI/bge-base-en-v1.5", "local"),
    # Ungated mirror of google/embeddinggemma-300m (same weights, Gemma terms apply).
    ("EmbeddingGemma", "unsloth/embeddinggemma-300m", "local"),
    ("Gemini API", "gemini-embedding-001", "api"),
]


# ---------- 1. Data ----------
def load_pairs():
    ds = load_dataset("sentence-transformers/stsb", split="test").shuffle(seed=SEED)
    ds = ds.select(range(N_PAIRS))
    return ds["sentence1"], ds["sentence2"], np.array(ds["score"])


# ---------- 2. Embedding ----------
def embed_local(model_id, sentences):
    """Return (vectors, seconds). Times only the encoding, not the download or loading."""
    model = SentenceTransformer(model_id, device="cpu")
    # EmbeddingGemma was trained with task prompts; it ships an "STS" prompt for this task.
    kwargs = {"prompt_name": "STS"} if "STS" in model.prompts else {}
    model.encode(sentences[:32], **kwargs)  # warm-up, so one-time setup is not timed
    start = time.perf_counter()
    vectors = model.encode(sentences, batch_size=32, **kwargs)
    return vectors, time.perf_counter() - start


def embed_gemini(sentences):
    """Return (vectors, None). Speed is not reported: on the free tier it is set by the
    rate limit (100 texts/minute), not by the model."""
    from google import genai
    from google.genai import types

    load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
    vectors = []
    for i in range(0, len(sentences), 100):  # free tier: 100 texts per minute
        if i:
            time.sleep(61)
        batch = sentences[i : i + 100]
        result = client.models.embed_content(
            model="gemini-embedding-001", contents=batch, config=config
        )
        vectors += [e.values for e in result.embeddings]
        print(f"    gemini: {len(vectors)}/{len(sentences)}")
    return np.array(vectors), None


def embed(label, model_id, where, sentences):
    """Embed once, then reuse the saved vectors (so the API is never called twice)."""
    key = hashlib.sha1("\n".join(sentences).encode()).hexdigest()[:10]
    path = CACHE / f"{label}_{key}.npz"
    if path.exists():
        saved = np.load(path)
        seconds = float(saved["seconds"])
        return saved["vectors"], None if np.isnan(seconds) else seconds
    if where == "api":
        vectors, seconds = embed_gemini(sentences)
    else:
        vectors, seconds = embed_local(model_id, sentences)
    CACHE.mkdir(exist_ok=True)
    np.savez(path, vectors=vectors, seconds=np.nan if seconds is None else seconds)
    return vectors, seconds


# ---------- 3. Scoring ----------
def cosine(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(axis=1)


def main():
    s1, s2, human = load_pairs()
    sentences = sorted(set(s1) | set(s2))  # embed each unique sentence once
    index = {s: i for i, s in enumerate(sentences)}
    print(f"{N_PAIRS} pairs, {len(sentences)} unique sentences\n")

    rows, sims = [], {}
    for label, model_id, where in MODELS:
        print(f"> {label}")
        vectors, seconds = embed(label, model_id, where, sentences)
        a = vectors[[index[s] for s in s1]]
        b = vectors[[index[s] for s in s2]]
        sims[label] = cosine(a, b)
        rho = spearmanr(sims[label], human).statistic
        speed = f"{len(sentences) / seconds:,.0f}" if seconds else "n/a"
        rows.append((label, where, vectors.shape[1], rho, speed))

    print("\n| Model | Runs | Dims | Spearman | Sentences/s (CPU) |")
    print("|---|---|---|---|---|")
    for label, where, dims, rho, speed in rows:
        print(f"| {label} | {where} | {dims} | {rho:.3f} | {speed} |")

    # Is the gap between two models real, or luck of which 400 pairs we drew?
    # Bootstrap: redraw 400 pairs (with replacement) 1,000 times and recompute.
    # Paired: each redraw scores every model on the same pairs, so we can compare them.
    rng = np.random.default_rng(SEED)
    boot = {label: [] for label in sims}
    for _ in range(1000):
        idx = rng.integers(0, N_PAIRS, N_PAIRS)
        for label in sims:
            boot[label].append(spearmanr(sims[label][idx], human[idx]).statistic)
    boot = {label: np.array(v) for label, v in boot.items()}

    print("\n95% confidence intervals:")
    for label in sims:
        lo, hi = np.percentile(boot[label], [2.5, 97.5])
        print(f"  {label:<15} [{lo:.3f}, {hi:.3f}]")

    print("\nIs each model better than the next smaller one? (paired difference, 95% CI)")
    labels = list(sims)
    for small, big in [*zip(labels, labels[1:], strict=False), ("bge-small", "Gemini API")]:
        diff = boot[big] - boot[small]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        verdict = "real" if lo > 0 else "could be luck"
        print(f"  {big} - {small}: {diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] -> {verdict}")

    # Where does the best model disagree most with humans? Compare ranks, not raw
    # values, because each model uses the cosine range differently.
    best = max(rows, key=lambda r: r[3])[0]
    model_rank = sims[best].argsort().argsort() / (N_PAIRS - 1)
    human_rank = human.argsort().argsort() / (N_PAIRS - 1)
    gap = model_rank - human_rank
    print(f"\nBiggest disagreements for {best} (rank percentile, 0 = least similar):")
    for i in np.argsort(-np.abs(gap))[:6]:
        print(f"  human {human_rank[i]:.2f} vs model {model_rank[i]:.2f} | {s1[i]}  ||  {s2[i]}")


if __name__ == "__main__":
    main()
