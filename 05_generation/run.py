"""05 - Generation: given retrieved context, does the LLM answer correctly and stay grounded?

SQuAD questions are answered by gpt-oss (via Groq's free tier) from four kinds of context:
the gold paragraph, retrieved paragraphs, retrieved 50-word chunks, and retrieved paragraphs with
the answer removed. Answers come back as JSON checked by a Pydantic schema and are scored with
SQuAD's exact match and F1.

Run (after 03_chunking/run.py has cached its chunk vectors; needs GROQ_API_KEY in .env):
    uv run python 05_generation/run.py
"""

import importlib.util
import json
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    BadRequestError,
    Groq,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, ValidationError
from sentence_transformers import SentenceTransformer

# 03's script is also called run.py, so load it by path.
_spec = importlib.util.spec_from_file_location(
    "chunking", Path(__file__).parent.parent / "03_chunking" / "run.py"
)
chunking = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chunking)

N_QUESTIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
CONDITIONS = ["gold", "paragraphs", "50-word chunks", "answer removed"]
WORD_BUDGET = 400  # words of context for the retrieved conditions
SEED = 42
CACHE = Path(__file__).parent / "cache" / "responses.jsonl"

SYSTEM = (
    "Answer the question using only the numbered context passages. "
    "Reply with the shortest exact phrase from the context that answers it (a name, number, date "
    "or short phrase, not a sentence). If the context does not contain the answer, set found to "
    "false and answer to an empty string. Do not use outside knowledge."
)


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    found: bool


SCHEMA = {
    "name": "answer",
    "strict": True,
    "schema": {**Answer.model_json_schema(), "additionalProperties": False},
}


# ---------- 1. Contexts ----------
def build_contexts(n):
    """For each of the first n questions in 03's sample: question, gold answers, four contexts."""
    texts, para_spans, questions = chunking.load_squad()
    gold_answers = load_gold_answers(questions)
    model = SentenceTransformer(chunking.EMBED_MODEL, device="cpu")
    rng = np.random.default_rng(SEED)
    pick = rng.permutation(len(questions))[:n]  # the first n of a fixed shuffle: n=60 extends n=10
    sample = [questions[i] for i in pick]

    retrieved = {}
    for name in ("paragraph", "50 words"):
        chunks = chunking.make_chunks(name, texts, para_spans)
        chunk_texts = [texts[t][s:e] for t, s, e in chunks]
        rankings = chunking.dense_rankings(name, chunk_texts, sample, model)
        retrieved[name] = (chunks, chunk_texts, rankings)

    items = []
    for qi, (question, title, a_start, a_end) in enumerate(sample):
        answers = gold_answers[(question, title, a_start)]
        para = next(texts[title][s:e] for s, e in para_spans[title] if s <= a_start and a_end <= e)
        ctx = {"gold": [para]}
        hit = {}
        for cond, name in (("paragraphs", "paragraph"), ("50-word chunks", "50 words")):
            chunks, chunk_texts, rankings = retrieved[name]
            picked = take_budget([chunk_texts[i] for i in rankings[qi]])
            ctx[cond] = picked
            covered = [
                chunks[i][0] == title and chunks[i][1] <= a_start and a_end <= chunks[i][2]
                for i in rankings[qi][: len(picked)]
            ]
            hit[cond] = any(covered)
        # Drop every passage that mentions any gold answer, so the answer is truly absent.
        chunks, chunk_texts, rankings = retrieved["paragraph"]
        clean = [
            chunk_texts[i]
            for i in rankings[qi]
            if not any(normalize(a) in normalize(chunk_texts[i]) for a in answers)
        ]
        ctx["answer removed"] = take_budget(clean)
        items.append(
            {"id": int(pick[qi]), "question": question, "answers": answers, "ctx": ctx, "hit": hit}
        )
    return items


def take_budget(passages):
    """Passages from the top until WORD_BUDGET words are used (always at least one)."""
    out, used = [], 0
    for p in passages:
        used += len(p.split())
        if out and used > WORD_BUDGET:
            break
        out.append(p)
    return out


def load_gold_answers(questions):
    """All annotator answers per question (SQuAD validation has up to 3+ per question)."""
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad", split="validation")
    wanted = {(q, t) for q, t, _, _ in questions}
    by_q = {}
    for row in ds:
        if (row["question"], row["title"]) in wanted:
            by_q[(row["question"], row["title"])] = row["answers"]["text"]
    return {(q, t, s): by_q[(q, t)] for q, t, s, _ in questions}


# ---------- 2. Calling the LLM ----------
def ask(client, model, question, passages):
    """One call. Returns (raw text, tokens used)."""
    context = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    server_errors = 0
    while True:
        try:
            r = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_schema", "json_schema": SCHEMA},
                reasoning_effort="low",
                temperature=0,
                max_completion_tokens=1024,
            )
            return r.choices[0].message.content, r.usage.total_tokens
        except RateLimitError as e:
            wait = float(e.response.headers.get("retry-after", 10))
            if wait > 600:  # daily token limit: stop and resume tomorrow from the cache
                raise SystemExit(f"Daily limit reached for {model}; rerun later to resume.") from e
            time.sleep(wait + 1)
        except (InternalServerError, APIConnectionError):  # transient: retry with backoff
            server_errors += 1
            if server_errors > 5:
                raise
            time.sleep(5 * server_errors)
        except BadRequestError as e:  # e.g. the model's reply failed Groq's JSON check
            return f"ERROR: {e.message}", 0


def parse(raw):
    """Validate the reply against the schema. Returns an Answer, or None if invalid."""
    try:
        return Answer.model_validate_json(raw)
    except (ValidationError, TypeError):
        return None


# ---------- 3. Scoring (SQuAD's official normalisation) ----------
def normalize(text):
    text = "".join(ch for ch in text.lower() if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def bootstrap(values, rounds=1000):
    rng = np.random.default_rng(SEED)
    n = len(values)
    return np.percentile([values[rng.integers(0, n, n)].mean() for _ in range(rounds)], [2.5, 97.5])


def main():
    load_dotenv()
    client = Groq(max_retries=0)
    items = build_contexts(N_QUESTIONS)
    print(f"{len(items)} questions x {len(CONDITIONS)} conditions x {len(MODELS)} models\n")

    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cache[(row["model"], row["condition"], row["id"])] = row
    CACHE.parent.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        for cond in CONDITIONS:
            todo = [it for it in items if (model, cond, it["id"]) not in cache]
            if todo:
                print(f"> {model} / {cond}: {len(todo)} calls")
            for it in todo:
                raw, tokens = ask(client, model, it["question"], it["ctx"][cond])
                row = {
                    "model": model,
                    "condition": cond,
                    "id": it["id"],
                    "raw": raw,
                    "tokens": tokens,
                }
                cache[(model, cond, it["id"])] = row
                with CACHE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")

    report(items, cache)


def report(items, cache):
    print(
        "\n| Model | Context | Answer in context | Valid JSON | Exact match (95% CI) | F1 "
        "| Said 'not found' | Tokens / call |"
    )
    print("|---|---|---|---|---|---|---|---|")
    per_q = {}  # (model, condition) -> exact match per question
    for model in MODELS:
        for cond in CONDITIONS:
            em, f1s, found, valid, tokens = [], [], [], [], []
            for it in items:
                row = cache[(model, cond, it["id"])]
                ans = parse(row["raw"])
                valid.append(ans is not None)
                tokens.append(row["tokens"])
                pred = ans.answer if ans and ans.found else ""
                found.append(bool(ans and ans.found))
                em.append(max(float(normalize(pred) == normalize(a)) for a in it["answers"]))
                f1s.append(max(f1(pred, a) for a in it["answers"]))
            em = np.array(em)
            per_q[(model, cond)] = em
            lo, hi = bootstrap(em)
            in_ctx = (
                "yes (gold)"
                if cond == "gold"
                else "no (removed)"
                if cond == "answer removed"
                else f"{np.mean([it['hit'][cond] for it in items]):.0%}"
            )
            print(
                f"| {model.split('/')[-1]} | {cond} | {in_ctx} | {np.mean(valid):.0%} | "
                f"{em.mean():.3f} ({lo:.3f} - {hi:.3f}) | {np.mean(f1s):.3f} | "
                f"{1 - np.mean(found):.0%} | {np.mean(tokens):.0f} |"
            )

    print("\nPaired differences in exact match (same questions):")
    short = {m: m.split("/")[-1] for m in MODELS}
    pairs = [((m, "gold"), (m, "paragraphs")) for m in MODELS]
    pairs += [((m, "50-word chunks"), (m, "paragraphs")) for m in MODELS]
    pairs += [((MODELS[0], c), (MODELS[1], c)) for c in CONDITIONS[:3]]
    for a, b in pairs:
        diff = per_q[a] - per_q[b]
        lo, hi = bootstrap(diff)
        verdict = "real" if lo > 0 or hi < 0 else "could be luck"
        print(
            f"  {short[a[0]]} {a[1]} - {short[b[0]]} {b[1]}: "
            f"{diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] -> {verdict}"
        )

    # Split retrieved conditions by whether retrieval found the answer at all.
    print("\nExact match, split by whether the answer was in the retrieved context:")
    for model in MODELS:
        for cond in ("paragraphs", "50-word chunks"):
            hit = np.array([it["hit"][cond] for it in items])
            em = per_q[(model, cond)]
            print(
                f"  {short[model]} {cond}: answer retrieved {em[hit].mean():.3f} "
                f"(n={hit.sum()}), not retrieved {em[~hit].mean():.3f} (n={(~hit).sum()})"
            )

    # Hallucination: with the answer removed, which answers were still "found"?
    print("\nAnswer removed, but the model still answered:")
    for model in MODELS:
        shown = 0
        for it in items:
            ans = parse(cache[(model, "answer removed", it["id"])]["raw"])
            if ans and ans.found and shown < 10:
                correct = max(f1(ans.answer, a) for a in it["answers"]) > 0.5
                print(f"  [{model.split('/')[-1]}] Q: {it['question']}")
                print(
                    f"      said: {ans.answer!r}  gold: {it['answers'][0]!r}"
                    f"  ({'close to gold' if correct else 'wrong'})"
                )
                shown += 1


if __name__ == "__main__":
    main()
