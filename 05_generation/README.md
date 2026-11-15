# 05 · Generation: does the LLM answer correctly, and does it admit when it can't?

## The question

Retrieval (02 to 04) only finds passages. The LLM then has to read them, pick out the answer,
return it in a format the application can use, and **say "not found" when the passages do not
contain it**. This experiment measures each of those, and answers the question left open in
[03](../03_chunking/): does an LLM answer better from whole paragraphs or from small fragments?

## How we measured it

- **Data:** 60 questions from the same SQuAD sample as 03 (fixed shuffle, seed 42), searched
  against all 48 articles.
- **Models:** [gpt-oss-120b and gpt-oss-20b](https://huggingface.co/openai/gpt-oss-120b) (OpenAI's
  open-weight models) on [Groq](https://console.groq.com)'s free tier, `reasoning_effort="low"`,
  temperature 0.
- **Four kinds of context** for every question:

| Context | What the LLM gets | What it tests |
|---|---|---|
| **gold** | The paragraph that contains the answer | Reading only (upper bound) |
| **paragraphs** | Dense search's top paragraphs, up to 400 words | Real RAG, 03's best chunking |
| **50-word chunks** | Dense search's top 50-word chunks, up to 400 words | Small fragments vs whole paragraphs |
| **answer removed** | The retrieved paragraphs, minus every one that contains a gold answer | Does it admit "not found"? |

- **Structured output:** the reply must match a Pydantic schema, enforced with Groq's strict JSON
  mode:

```python
class Answer(BaseModel):
    answer: str   # the shortest exact phrase from the context
    found: bool   # false if the context does not contain the answer
```

- **Scoring:** SQuAD's official **exact match** (after lowercasing and removing punctuation and
  a/an/the, the answer equals one of the human answers) and **F1** (word overlap with the closest
  human answer, so "Denver Broncos" vs "the Broncos" gets partial credit).
- **95% CI:** paired bootstrap over questions (1,000 redraws), as in 01 to 04. With 60 questions
  the intervals are wide; see the notes.

## Results

| Model | Context | Answer in context | Valid JSON | Exact match (95% CI) | F1 | Said "not found" | Tokens / call |
|---|---|---|---|---|---|---|---|
| gpt-oss-120b | gold | yes | 100% | **0.917** (0.850 – 0.983) | 0.965 | 0% | 452 |
| gpt-oss-120b | paragraphs | 90% | 100% | 0.750 (0.633 – 0.867) | 0.831 | 8% | 714 |
| gpt-oss-120b | 50-word chunks | 90% | 100% | 0.783 (0.683 – 0.883) | 0.838 | 10% | 838 |
| gpt-oss-120b | answer removed | no | 100% | – | – | **85%** | 701 |
| gpt-oss-20b | gold | yes | 100% | 0.850 (0.767 – 0.933) | 0.907 | 2% | 445 |
| gpt-oss-20b | paragraphs | 90% | 100% | 0.750 (0.650 – 0.850) | 0.819 | 8% | 707 |
| gpt-oss-20b | 50-word chunks | 90% | 100% | 0.733 (0.617 – 0.833) | 0.815 | 3% | 839 |
| gpt-oss-20b | answer removed | no | 100% | – | – | **75%** | 701 |

**Paired differences in exact match** (same questions)

| Comparison | Difference (95% CI) | |
|---|---|---|
| 120b: gold − retrieved paragraphs | +0.167 (+0.083 – +0.267) | real |
| 20b: gold − retrieved paragraphs | +0.100 (0.000 – +0.200) | could be luck |
| 120b: 50-word chunks − paragraphs | +0.033 (−0.050 – +0.133) | could be luck |
| 20b: 50-word chunks − paragraphs | −0.017 (−0.133 – +0.100) | could be luck |
| gold: 120b − 20b | +0.067 (−0.017 – +0.150) | could be luck |
| paragraphs: 120b − 20b | 0.000 (−0.083 – +0.083) | could be luck |

**When retrieval found the answer** (54 of 60 questions), exact match was 0.82 to 0.87. When it
did not (6 questions), it was **0 for every model and context**.

**Answer removed, but the model answered anyway** (9 of 60 for 120b, 15 of 60 for 20b), for example:

| Question | Model said | Correct answer |
|---|---|---|
| How much did Tesla sell his AC patents to Westinghouse for? | $60,000 | $216,000 |
| In what language were the classes given? | Welsh | German |
| How many points did the Panthers defense surrender? | 24 | 308 |
| How much of Warsaw's population was Jewish? | 35.7% | some 30% of the city |

## What I learned

1. **Structured output is solved for this task:** 480 of 480 replies were valid JSON matching the
   schema. With strict schema enforcement, parsing failures stop being the problem; the
   *content* of the fields is.
2. **Retrieval is the ceiling.** Given the right paragraph, gpt-oss-120b gets 92% exact match;
   through real retrieval it drops to 75%. Every question where retrieval missed the answer was
   answered wrongly or not at all. Improving retrieval (02 to 04) matters more than the model.
3. **The LLM sometimes makes up an answer instead of saying "not found"**: 15% of the time for
   120b, 25% for 20b. The invented answers are not random: they are real numbers and names
   *from the context* that answer a nearby question ($60,000 for a different Tesla deal, 24 points
   from a different game). They look grounded, which makes them dangerous. Production RAG needs a
   check for this, such as asking for the supporting sentence and verifying it.
4. **Fragments vs paragraphs made no measurable difference here** (+0.03 and −0.02, both within
   noise). Whether an LLM can piece the answer together from 50-word fragments was the open
   question from 03; on these short factual questions it can. Questions that need reasoning across
   a whole section might behave differently.
5. **Exact match underrates LLMs.** Most "wrong" gold-paragraph answers were correct but worded
   differently: *"Economists"* vs *"Economist"*, *"3"* vs *"three"*, *"public schools"* vs
   *"public"*. That is why F1 is higher (0.97 for 120b on gold), and why free-text answers need a
   smarter grader, such as an LLM judge. That is the subject of [06](../06_judge/).
6. **The smaller model is close.** gpt-oss-20b matched 120b on retrieved paragraphs (0.750 each)
   and was not significantly worse anywhere, but it invented answers more often (25% vs 15%).

## Run it

```bash
uv sync
uv run python 03_chunking/run.py         # once, to cache the chunk vectors
# add GROQ_API_KEY=... to .env (free key from console.groq.com)
uv run python 05_generation/run.py 60    # number of questions
```

480 calls (60 questions × 4 contexts × 2 models), about 160K tokens per model, which fits the
free tier's 200K tokens per model per day. Calls are paced by the 8K tokens/minute limit, so a run
takes about 45 minutes. Every reply is cached in `05_generation/cache/responses.jsonl`; if a
limit is hit, rerun the same command to continue, and rerunning with a larger number adds
questions without repeating earlier ones.

## Notes

- **60 questions is small.** The confidence intervals are about ±0.1, so only large differences
  show up as real. The free tier's daily token limit sets the size; the sample can be extended on
  later days.
- "Answer removed" drops every retrieved paragraph that contains any of the human answers as text.
  A paragraph can still contain *part* of an answer (one model answered "Murray Gold" when the gold
  answer is "Murray Gold and Ben Foster"), so a few "found" answers there may be partly grounded.
- Transient API errors (one HTTP 500 in this run) are retried with backoff; a reply rejected by the
  strict JSON check would be recorded as invalid JSON, not crash the run (none were).
- 400 words of context is about 3 paragraphs or 8 fifty-word chunks.
