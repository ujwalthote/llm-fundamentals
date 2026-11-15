# 06 · LLM-as-judge: can an LLM grade answers the way human experts do?

## The question

Most LLM outputs have no single correct answer ("write a travel blog", "explain this error"), so
exact match is useless, and human grading is too slow to run on every change. The usual answer is
an **LLM judge**: show a strong model two answers and ask which is better.

But a judge is only useful if it agrees with people. So this experiment has two parts:

1. **Validate the judge** against expert votes (06).
2. **Use the validated judge** on our own pipeline's answers from 05 (06b).

## How we measured it

- **Data:** [MT-Bench human judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)
  from the paper *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (Zheng et al., 2023).
  - 80 questions (writing, roleplay, reasoning, maths, coding, extraction, STEM, humanities).
  - Answers from six 2023 chatbots: GPT-4, Claude-v1, GPT-3.5, Vicuna-13B, Alpaca-13B, LLaMA-13B.
  - **3,355 votes by 65 people** (the paper's authors and domain experts): answer A better, B
    better, or tie. These votes are the golden labels; the answers are just material to judge.
  - We use a fixed random sample of **150 first-turn votes** (seed 42).
- **Judges:**
  - **Gemini 3.5 Flash Lite** (Gemini API free tier).
  - **qwen3.8-27b**, without reasoning (Groq free tier; its 1,000 output tokens per minute limit
    makes reasoning too slow).
  - **gpt-oss-120b** (Groq free tier). Its daily token limit was mostly used by 05, so it finished
    only 34 of the 150 pairs.
  - **GPT-4**, as a reference: its verdicts are published with the dataset.
- **Every pair is judged twice, with the answers swapped.** If the verdict follows the position
  rather than the content, the judge contradicted itself and the pair counts as a tie (the
  paper's rule). Replies are JSON `{reason, winner}` checked with Pydantic.
- **Agreement with experts**, as in the paper:
  - *with ties:* the judge's verdict (A, B or tie) equals the expert's.
  - *without ties:* only votes where both the judge and the expert picked a winner.
- **Biases:**
  - **Position:** the share of verdicts that change when only the order changes, and among those,
    how often the judge picked whichever answer it saw first.
  - **Length:** how often the judge's winner is the longer answer, compared with how often the
    experts' winner is.
- **95% CI:** bootstrap over votes (1,000 redraws), as in 01 to 05.

## Results

**The bar: how often two experts agree with each other** (606 pairs of votes on the same answers):
**64% with ties, 83% without**. The paper reports 63% and 81%.

| Judge | Pairs | Agreement with experts, with ties (95% CI) | Without ties | Same verdict both orders | Picks the longer answer | Valid JSON |
|---|---|---|---|---|---|---|
| *Expert vs expert* | *606* | *64%* | *83%* | – | *71%* | – |
| GPT-4 (published) | 150 | **67%** (60% – 73%) | **85%** | 84%¹ | – | – |
| **Gemini 3.5 Flash Lite** | 150 | 61% (53% – 69%) | 78% | **83%** | **72%** | **100%** |
| qwen3.8-27b (no thinking) | 110 | 57% (48% – 67%) | 81% | 67% | 74% | 88%² |
| gpt-oss-120b | 34 | 62% (47% – 79%) | 77% | 76% | 83% | 100% |

¹ Over all 1,200 first-turn pairs in the dataset. ² After one retry; 29% of replies needed it.

- "Picks the longer answer" counts decided verdicts only; for the experts it is their own votes on
  the same 150 pairs (71% of 118 decided votes).
- qwen and gpt-oss stopped at their free daily token limits (110 and 34 pairs), so their intervals
  are wider. Their numbers are reported on the pairs they finished.

## Part 2 (06b): grading our own pipeline

In [05](../05_generation/), exact match marked *"Economists"* wrong when the gold answer is
*"Economist"*. Here Gemini, the most complete and reliable judge above, decides whether each answer
that exact match rejected means the same as a gold answer. Answers that matched exactly are already
correct and "not found" is already wrong, so only the rejected answers are sent to the judge
(54 calls).

| Model | Context | Exact match | Judged by meaning (95% CI) | Rescued |
|---|---|---|---|---|
| gpt-oss-120b | gold | 0.917 | **1.000** (1.000 – 1.000) | 5 |
| gpt-oss-120b | paragraphs | 0.750 | **0.883** (0.800 – 0.967) | 8 |
| gpt-oss-120b | 50-word chunks | 0.783 | **0.850** (0.750 – 0.933) | 4 |
| gpt-oss-20b | gold | 0.850 | **0.950** (0.883 – 1.000) | 6 |
| gpt-oss-20b | paragraphs | 0.750 | **0.867** (0.783 – 0.950) | 7 |
| gpt-oss-20b | 50-word chunks | 0.733 | **0.833** (0.733 – 0.933) | 6 |

| Judged correct (exact match said wrong) | Judged wrong |
|---|---|
| "Economists" for *Economist* | "Welsh" for *German* |
| "public schools" for *public* | "34%" for *some 30% of the city* |
| "about 400 AD to 1914" for *from about 400 AD to 1914* | "nationwide backbone networks" for *a major part of the Internet backbone* |
| "Shoushi Li (授時暦)" for *Shoushi Li* | "a function of n" for *the time taken* |

## What I learned

1. **A small free judge comes close to the experts.** Gemini Flash Lite agrees with the experts
   61% of the time (78% without ties); two experts agree with each other 64% (83%), and GPT-4 67%
   (85%). The intervals overlap: on this data a cheap judge is statistically close to both. The
   experts' own disagreement is the real ceiling: no judge can agree with "the humans" more than
   the humans agree with each other.
2. **Every judge contradicts itself when the order changes.** Gemini gave the same verdict in both
   orders 83% of the time, GPT-4 84%, gpt-oss 76%, qwen 67%. So one in six verdicts depends on which
   answer is shown first. **Always judge both orders** and count disagreement as a tie; a single
   pass silently builds this noise into your eval.
3. **Length bias depends on the judge.** Experts preferred the longer answer in 71% of decided
   votes (longer answers here were often genuinely more complete). Gemini matches that (72%), so it
   adds no extra length bias; gpt-oss picked the longer answer 83% of the time, though on only 34
   pairs. Check this before trusting a judge on changes that make answers longer.
4. **Structured output depends on the model.** Gemini and gpt-oss returned valid JSON every time.
   qwen, under the same strict JSON mode, wrote plain text 29% of the time, and 12% still failed
   after a retry. In 05, gpt-oss was 480 of 480. Test the format on every model you plan to use.
5. **The validated judge changes our 05 conclusions.** Graded by meaning instead of exact spelling,
   gpt-oss-120b answers **100%** correctly from the gold paragraph (exact match said 91.7%) and
   **88%** through real retrieval (exact match said 75%). Most "errors" were wording: "Economists"
   for "Economist", "public schools" for "public". The real errors it kept, such as "Welsh" for
   "German", came from retrieval bringing the wrong paragraph, which confirms 05's finding that
   retrieval, not the model, is the weak point.
6. **Validate the judge, then use it.** The order matters: 06 shows how far Gemini can be trusted,
   and only then does 06b use it to grade our own answers. Without the first step, a judge's grade
   is a number nobody can interpret.
7. **Free tiers shape the design.** Separate daily quotas per model, an output-token limit that
   counts the *maximum* reply length, and a judge that stops halfway: in practice, LLM evals need
   per-model caching, resumable runs and reporting on whatever finished.

## Run it

```bash
uv sync
# GEMINI_API_KEY and GROQ_API_KEY in .env (both free)
uv run python 06_judge/run.py 150              # all judges; or add a name, e.g. "qwen", to run one
uv run python 06_judge/grade_05.py             # 06b, after 05_generation/run.py
```

Every verdict is cached per judge in `06_judge/cache/`, so a run stopped by a daily limit resumes
where it left off, and judges with separate quotas can run in parallel.

## Notes

- Free-tier limits shaped this experiment: Gemini Flash Lite allows 500 requests a day; each Groq
  model allows 200K tokens a day; qwen also allows only 1,000 *output* tokens a minute, and Groq
  counts the maximum reply length against it, so qwen ran with reasoning off and a 200-token cap.
- When qwen's reply failed Groq's JSON check, it was retried once with a format-only reminder
  ("reply only with a JSON object"). Replies that failed twice count as no verdict, which the
  combination rule turns into a tie.
- The published GPT-4 verdicts used the paper's own prompt; ours is a shorter prompt in the same
  spirit, so the comparison with GPT-4 is indicative.
- The answers being judged are from 2023 chatbots. That does not matter here: we test whether the
  judge agrees with the experts, not how good those chatbots were.
