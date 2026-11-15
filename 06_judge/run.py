"""06 - LLM-as-judge: does an LLM judge agree with human experts, and is it biased?

MT-Bench human judgments: experts compared two chatbot answers to the same question and voted
A, B or tie. We ask LLM judges the same question, twice per pair with the answers swapped, and
measure agreement with the experts, position bias and length bias. GPT-4's verdicts (published
with the dataset) are included as a reference judge.

Run (needs GROQ_API_KEY and GEMINI_API_KEY in .env):
    uv run python 06_judge/run.py
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from groq import (
    APIConnectionError,
    BadRequestError,
    Groq,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, ValidationError

N_PAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
ONLY = sys.argv[2] if len(sys.argv) > 2 else None  # run just this judge (others are still reported)
SEED = 42
# name -> (provider, model id). Each Groq model has its own free daily token budget.
JUDGES = {
    "gemini-3.5-flash-lite": ("gemini", "gemini-3.5-flash-lite"),
    # Free-tier qwen allows only 1,000 output tokens per minute, so it runs without reasoning.
    "qwen3.8-27b (no thinking)": ("groq", "qwen/qwen3.8-27b"),
    "gpt-oss-120b": ("groq", "openai/gpt-oss-120b"),
}
CACHE_DIR = Path(__file__).parent / "cache"  # one file per judge, so judges can run in parallel


def cache_file(judge):
    return CACHE_DIR / f"verdicts_{judge.split()[0]}.jsonl"


PROMPT = """You are an impartial judge. Compare two AI assistants' answers to the user's question.
Consider helpfulness, correctness, depth and clarity. Do not let the order of the answers or their
length influence you. Give a one-sentence reason, then the winner: "A", "B" or "tie".

[User question]
{question}

[Answer A]
{a}

[Answer B]
{b}"""


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str
    winner: Literal["A", "B", "tie"]


# Gemini's schema format does not accept "additionalProperties"; Pydantic still validates replies.
GEMINI_SCHEMA = {
    k: v for k, v in Verdict.model_json_schema().items() if k != "additionalProperties"
}


# ---------- 1. Data ----------
def load_pairs(n):
    """n expert votes on first-turn answers, plus GPT-4's verdict on the same pair if published."""
    ds = load_dataset("lmsys/mt_bench_human_judgments")
    gpt4 = {}  # (question, model_a, model_b) -> GPT-4's winner (both orders aggregated)
    flip = {"model_a": "model_b", "model_b": "model_a", "tie": "tie"}
    gpt4_consistent = []
    for row in ds["gpt4_pair"]:
        if row["turn"] == 1:
            q, a, b, w = row["question_id"], row["model_a"], row["model_b"], row["winner"]
            w = "tie" if w.startswith("tie") else w  # "tie (inconsistent)": flipped with order
            gpt4_consistent.append(row["winner"] != "tie (inconsistent)")
            gpt4[(q, a, b)] = w
            gpt4[(q, b, a)] = flip.get(w, w)  # the same pair stored in the other order

    human = [row for row in ds["human"] if row["turn"] == 1]
    votes = defaultdict(list)  # every expert vote per pair, for human-human agreement
    for row in human:
        votes[(row["question_id"], row["model_a"], row["model_b"])].append(row["winner"])

    rng = np.random.default_rng(SEED)
    pairs = []
    for i in rng.permutation(len(human))[:n]:
        row = human[int(i)]
        key = (row["question_id"], row["model_a"], row["model_b"])
        pairs.append(
            {
                "id": int(i),
                "question": row["conversation_a"][0]["content"],
                "a": row["conversation_a"][1]["content"],
                "b": row["conversation_b"][1]["content"],
                "human": row["winner"],  # model_a / model_b / tie
                "gpt4": gpt4.get(key),
            }
        )
    print(
        f"GPT-4 (published): same verdict in both orders on {np.mean(gpt4_consistent):.0%} "
        f"of {len(gpt4_consistent)} first-turn pairs"
    )
    return pairs, votes


# ---------- 2. Judges ----------
class DailyLimit(Exception):
    """The free tier's daily quota for this model is used up."""


FORMAT_REMINDER = 'Reply only with a JSON object: {"reason": "...", "winner": "A" | "B" | "tie"}.'


def judge_groq(client, model, prompt):
    """Returns (raw reply, tokens, whether a format retry was needed)."""
    schema = {**Verdict.model_json_schema(), "additionalProperties": False}
    messages = [{"role": "user", "content": prompt}]
    tries, retried = 0, False
    while True:
        try:
            r = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "verdict", "strict": True, "schema": schema},
                },
                reasoning_effort="none" if "qwen" in model else "low",
                temperature=0,
                # Groq reserves this against qwen's 1,000 output tokens/minute: keep it small.
                max_completion_tokens=200 if "qwen" in model else 1024,
            )
            return r.choices[0].message.content, r.usage.total_tokens, retried
        except RateLimitError as e:
            wait = float(e.response.headers.get("retry-after", 10))
            if wait > 600:
                raise DailyLimit(model) from e
            time.sleep(wait + 1)
        except (InternalServerError, APIConnectionError):
            tries += 1
            if tries > 5:
                raise
            time.sleep(5 * tries)
        except BadRequestError as e:
            # The reply failed Groq's JSON check (the model wrote plain text). Retry once with a
            # format-only reminder; if it fails again, record it as invalid.
            if retried:
                return f"ERROR: {e.message}", 0, retried
            retried = True
            messages = [{"role": "system", "content": FORMAT_REMINDER}, *messages]


def judge_gemini(client, model, prompt):
    tries = 0
    while True:
        try:
            r = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": GEMINI_SCHEMA,
                    "temperature": 0,
                },
            )
            time.sleep(4.5)  # free tier: 15 requests per minute
            return r.text, r.usage_metadata.total_token_count, False
        except genai_errors.APIError as e:
            tries += 1
            if e.code == 429 and "PerDay" in str(e):
                raise DailyLimit(model) from e
            if tries > 8:
                raise
            time.sleep(15 * tries)


def parse(raw):
    try:
        return Verdict.model_validate_json(raw).winner
    except (ValidationError, TypeError):
        return None


# ---------- 3. Scoring ----------
def combine(first, second):
    """Verdict from both orders. first: A/B shown as-is; second: answers swapped.
    Consistent -> that answer wins; otherwise the judge contradicted itself -> tie."""
    swapped = {"A": "model_b", "B": "model_a", "tie": "tie", None: None}[second]
    normal = {"A": "model_a", "B": "model_b", "tie": "tie", None: None}[first]
    return normal if normal == swapped else "tie"


def agreement(judge, human):
    """(with ties: all votes, 3 classes) and (without ties: both picked a winner)."""
    judge, human = np.array(judge), np.array(human)
    with_tie = (judge == human).mean()
    mask = (judge != "tie") & (human != "tie")
    return with_tie, (judge[mask] == human[mask]).mean(), mask.sum()


def bootstrap(values, rounds=1000):
    rng = np.random.default_rng(SEED)
    n = len(values)
    return np.percentile([values[rng.integers(0, n, n)].mean() for _ in range(rounds)], [2.5, 97.5])


def human_human(votes):
    """Agreement between two experts who voted on the same pair (all pairs with 2+ votes)."""
    same, total, same_nt, total_nt = 0, 0, 0, 0
    for vs in votes.values():
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                same += vs[i] == vs[j]
                total += 1
                if vs[i] != "tie" and vs[j] != "tie":
                    same_nt += vs[i] == vs[j]
                    total_nt += 1
    return same / total, same_nt / total_nt, total


def main():
    load_dotenv()
    clients = {
        "groq": Groq(max_retries=0),
        "gemini": genai.Client(api_key=os.environ["GEMINI_API_KEY"]),
    }
    pairs, votes = load_pairs(N_PAIRS)
    print(f"{len(pairs)} expert votes x 2 orders x {len(JUDGES)} judges\n")

    cache = {}
    CACHE_DIR.mkdir(exist_ok=True)
    for judge in JUDGES:
        if cache_file(judge).exists():
            for line in cache_file(judge).read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                cache[(row["judge"], row["id"], row["order"])] = row

    for judge, (provider, model_id) in JUDGES.items():
        if ONLY and ONLY not in judge:
            continue
        todo = [(p, o) for p in pairs for o in (1, 2) if (judge, p["id"], o) not in cache]
        if todo:
            print(f"> {judge}: {len(todo)} calls")
        fn = judge_groq if provider == "groq" else judge_gemini
        try:
            for p, order in todo:
                a, b = (p["a"], p["b"]) if order == 1 else (p["b"], p["a"])
                prompt = PROMPT.format(question=p["question"], a=a, b=b)
                raw, tokens, retried = fn(clients[provider], model_id, prompt)
                row = {
                    "judge": judge,
                    "id": p["id"],
                    "order": order,
                    "raw": raw,
                    "tokens": tokens,
                    "json_retry": retried,
                }
                cache[(judge, p["id"], order)] = row
                with cache_file(judge).open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
        except DailyLimit:
            print(f"  daily limit reached for {judge}; reporting the pairs it finished")
        except KeyboardInterrupt:
            break

    report(pairs, votes, cache)


def report(pairs, votes, cache):
    hh = human_human(votes)
    print(
        f"\nExpert vs expert (all {hh[2]} pairs of votes on the same answers): "
        f"{hh[0]:.0%} with ties, {hh[1]:.0%} without ties"
    )

    print(
        "\n| Judge | Pairs | Agreement with experts, with ties (95% CI) | Without ties "
        "| n without ties | Same verdict both orders | Picks first answer when inconsistent "
        "| Picks the longer answer | Valid JSON | Tokens / call |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")

    def longer(winner, p):
        return (winner == "model_a") == (len(p["a"]) > len(p["b"]))

    # GPT-4 reference (published verdicts, already combined over both orders).
    with_gpt4 = [p for p in pairs if p["gpt4"]]
    g = agreement([p["gpt4"] for p in with_gpt4], [p["human"] for p in with_gpt4])
    lo, hi = bootstrap(np.array([p["gpt4"] == p["human"] for p in with_gpt4], dtype=float))
    print(
        f"| GPT-4 (published) | {len(with_gpt4)} | {g[0]:.0%} ({lo:.0%} - {hi:.0%}) | {g[1]:.0%} "
        f"| {g[2]} | - | - | - | - | - |"
    )

    for judge in JUDGES:
        # Only pairs this judge finished in both orders (a daily limit can stop it early).
        done = [p for p in pairs if (judge, p["id"], 1) in cache and (judge, p["id"], 2) in cache]
        if not done:
            continue
        raws = [cache[(judge, p["id"], o)]["raw"] for p in done for o in (1, 2)]
        valid = np.mean([parse(r) is not None for r in raws])
        retry = np.mean(
            [cache[(judge, p["id"], o)].get("json_retry", False) for p in done for o in (1, 2)]
        )
        first = [parse(cache[(judge, p["id"], 1)]["raw"]) for p in done]
        second = [parse(cache[(judge, p["id"], 2)]["raw"]) for p in done]
        final = [combine(f, s) for f, s in zip(first, second, strict=True)]
        human = [p["human"] for p in done]
        with_tie, no_tie, n = agreement(final, human)
        lo, hi = bootstrap((np.array(final) == np.array(human)).astype(float))
        consistent = np.mean(
            [
                combine(f, s) != "tie" or (f == s == "tie")
                for f, s in zip(first, second, strict=True)
            ]
        )
        # Position bias: among contradictory pairs, how often did it pick whatever was shown first?
        flips = [
            (f, s)
            for f, s in zip(first, second, strict=True)
            if combine(f, s) == "tie" and not (f == s == "tie")
        ]
        first_pref = np.mean([f == "A" and s == "A" for f, s in flips]) if flips else 0.0
        decided = [(w, p) for w, p in zip(final, done, strict=True) if w != "tie"]
        long_pref = np.mean([longer(w, p) for w, p in decided])
        tokens = np.mean([cache[(judge, p["id"], o)]["tokens"] for p in done for o in (1, 2)])
        print(
            f"| {judge} | {len(done)} | {with_tie:.0%} ({lo:.0%} - {hi:.0%}) | {no_tie:.0%} | {n} "
            f"| {consistent:.0%} | {first_pref:.0%} of {len(flips)} | {long_pref:.0%} "
            f"| {valid:.0%} ({retry:.0%} needed a retry) | {tokens:.0f} |"
        )

    decided = [p for p in pairs if p["human"] != "tie"]
    print(
        f"\nExperts picked the longer answer in "
        f"{np.mean([longer(p['human'], p) for p in decided]):.0%} of {len(decided)} decided votes."
    )


if __name__ == "__main__":
    main()
