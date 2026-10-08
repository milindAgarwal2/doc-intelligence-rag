# RAG Evaluation Harness

Offline, maintainer-facing quality checks for the retrieval layer. **Embedding-only
— no LLM calls, no API cost.** It drives the same functions the chatbot uses
(`chatBOT_engine.retrieve_docs` / `confidence_probe`), so the numbers reflect real
app behaviour.

## Run

```bash
# from the project root, with the venv active
.venv/bin/python -m tests.eval.run

# also dump raw results as JSON (for tooling / CI artifacts)
.venv/bin/python -m tests.eval.run --json report.json

# use a different dataset
.venv/bin/python -m tests.eval.run --dataset path/to/dataset.json
```

## What it measures

1. **Retrieval recall@k** — for each in-corpus question, is the ground-truth page
   among the top-k retrieved chunks? Reports the rank per question and the overall
   recall. This is the primary "does the right page get found" number.
2. **Confidence calibration** — the distribution of `confidence_probe()` scores for
   in-corpus vs negative (off-topic) questions, the separating gap, and a
   **suggested threshold** for the #3 low-confidence banner.

## Exit code (CI gating)

Exits `0` if `recall@k >= recall_floor`, else `1`. Run it in CI (or before a commit
that touches chunking, the embedding model, retrieval params, or prompts) to catch
retrieval regressions before they reach users.

## Reading the report

- `FOUND @rank N` — the expected page was the Nth retrieved chunk (rank 1 = best).
- `MISS` — the expected page was not in the top-k.
- `SKIPPED` — the question's document isn't currently embedded, so it wasn't scored
  (not a failure — see below).
- `SUGGESTED low_confidence_threshold` — plug this into the GUI slider (or
  `config.json`'s `low_confidence_threshold`) to make the #3 banner meaningful. If
  the report says **NO clean gap**, embedding scores alone can't separate answerable
  from unanswerable on this corpus — keep the banner conservative / non-blocking.

## The dataset (`dataset.json`)

Plain JSON. Each question carries its **human-asserted** ground-truth source — this
is deliberate: the eval is only trustworthy because the expected pages are asserted
independently of the model being tested.

```json
{
  "k": 8,
  "recall_floor": 0.75,
  "questions": [
    {"q": "current consumption in dormant mode",
     "expect_file": "RP-008307-DS-2-pico-datasheet.pdf", "expect_page": 17},
    {"q": "medieval French poetry", "expect_file": null}
  ]
}
```

- `expect_file` — matched by **basename** (the `pdf/`/`pptx/` prefix and path don't
  matter). Set to `null` for a **negative** case (not in any document) — these feed
  confidence calibration and need no embedded doc.
- `expect_page` — the ground-truth page/slide index.
- `k`, `recall_floor` — retrieval depth and the pass/fail recall threshold.

## Adapting to a changing store

The harness reads the currently-embedded documents and **skips** any question whose
`expect_file` isn't present (reported under `SKIPPED`, not counted as a failure).
So you can keep questions for documents that come and go:

- **Add a document** → add its question(s) to `dataset.json`. They start being
  evaluated automatically once the doc is embedded.
- **Remove a document** → its questions auto-skip; no need to delete them.

> The authoritative question→page pairs are written/reviewed by a human (that's what
> keeps the eval honest). For a large new document you can draft candidates however
> you like, but confirm the expected pages against the source before trusting the
> numbers.

## Caveat on sample size

With only a few documents embedded the dataset is small, so **recall is directional,
not statistically strong** — it becomes more meaningful as documents and questions
grow. The **calibration** half is useful immediately: it directly answers "what
threshold should the #3 banner use" for the current corpus.
