"""06b - Using the validated judge on our own pipeline: re-grading 05's answers.

Exact match marked "Economists" wrong when the gold answer is "Economist". Here the judge decides
whether each answer that exact match rejected actually means the same as a gold answer, so we get
05's accuracy by meaning, not by spelling. Only rejected answers are sent to the judge: an exact
match is already correct, and "not found" is already wrong.

Run (after 05_generation/run.py and 06_judge/run.py):
    uv run python 06_judge/grade_05.py
"""

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
from dotenv import load_dotenv
from google import genai
from groq import Groq
from pydantic import BaseModel, ConfigDict, ValidationError

ROOT = Path(__file__).parent.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The judge chosen from 06's results.
JUDGE = sys.argv[1] if len(sys.argv) > 1 else "gemini-3.5-flash-lite"
sys.argv = sys.argv[:1]  # the imported scripts read their own command-line arguments
judging = load("judging", ROOT / "06_judge" / "run.py")
generation = load("generation", ROOT / "05_generation" / "run.py")
PROVIDER, MODEL_ID = judging.JUDGES[JUDGE]
CONDITIONS = ["gold", "paragraphs", "50-word chunks"]
CACHE = Path(__file__).parent / "cache" / f"grades_{JUDGE}.jsonl"

PROMPT = """You are grading a reading-comprehension answer.
Question: {question}
Correct answers (any of these is right): {gold}
Student answer: {pred}

Does the student answer mean the same as one of the correct answers? Ignore spelling, plurals,
number formats ("3" vs "three") and extra words that do not change the meaning. It is wrong if
it names a different thing, a different number, or only part of a multi-part answer.
Give a one-sentence reason, then the verdict "correct" or "wrong"."""


class Grade(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str
    verdict: Literal["correct", "wrong"]


def ask(client, prompt):
    """Reuses 06's judge calls with this task's schema."""
    judging.Verdict = Grade  # same call code, different reply schema
    judging.GEMINI_SCHEMA = {
        k: v for k, v in Grade.model_json_schema().items() if k != "additionalProperties"
    }
    fn = judging.judge_groq if PROVIDER == "groq" else judging.judge_gemini
    raw, _, _ = fn(client, MODEL_ID, prompt)
    try:
        return Grade.model_validate_json(raw)
    except (ValidationError, TypeError):
        return None


def main():
    load_dotenv()
    client = (
        Groq(max_retries=0)
        if PROVIDER == "groq"
        else genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    )
    items = generation.build_contexts(60)
    answers = {}
    for line in (
        (ROOT / "05_generation" / "cache" / "responses.jsonl").read_text("utf-8").splitlines()
    ):
        row = json.loads(line)
        answers[(row["model"], row["condition"], row["id"])] = generation.parse(row["raw"])

    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cache[(row["model"], row["condition"], row["id"])] = row
    CACHE.parent.mkdir(parents=True, exist_ok=True)

    results = {}  # (model, condition) -> list of (exact match, judged correct)
    examples = []
    for model in generation.MODELS:
        for cond in CONDITIONS:
            rows = []
            for it in items:
                ans = answers[(model, cond, it["id"])]
                pred = ans.answer if ans and ans.found else ""
                em = any(
                    generation.normalize(pred) == generation.normalize(a) for a in it["answers"]
                )
                if em or not pred:
                    rows.append((em, em))
                    continue
                key = (model, cond, it["id"])
                if key not in cache:
                    prompt = PROMPT.format(question=it["question"], gold=it["answers"], pred=pred)
                    grade = ask(client, prompt)
                    cache[key] = {
                        "model": model,
                        "condition": cond,
                        "id": it["id"],
                        "verdict": grade.verdict if grade else None,
                        "reason": grade.reason if grade else None,
                    }
                    with CACHE.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(cache[key]) + "\n")
                judged = cache[key]["verdict"] == "correct"
                rows.append((False, judged))
                examples.append((judged, it["question"], pred, it["answers"][0]))
            results[(model, cond)] = np.array(rows, dtype=float)

    print(f"Judge: {JUDGE}\n")
    print("| Model | Context | Exact match | Judged correct (95% CI) | Rescued by the judge |")
    print("|---|---|---|---|---|")
    for (model, cond), r in results.items():
        lo, hi = judging.bootstrap(r[:, 1])
        print(
            f"| {model.split('/')[-1]} | {cond} | {r[:, 0].mean():.3f} | "
            f"{r[:, 1].mean():.3f} ({lo:.3f} - {hi:.3f}) | {int(r[:, 1].sum() - r[:, 0].sum())} |"
        )

    for label, want in (("Judged correct", True), ("Judged wrong", False)):
        print(f"\n{label} (exact match said wrong):")
        for _, q, pred, gold in [e for e in examples if e[0] == want][:8]:
            print(f"  Q: {q}\n     said {pred!r}, gold {gold!r}")


if __name__ == "__main__":
    main()
