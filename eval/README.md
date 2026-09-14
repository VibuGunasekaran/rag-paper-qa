# eval/

- `gold_set.jsonl` — Day 3. Written by hand via `scripts/build_gold.py add`.
- `run_eval.py` — Day 4. Recall@{1,3,5,10}, MRR@10, nDCG@10, Cov@5 for
  comparative rows, and p50/p95 latency over the 18-config sweep (3 chunk sizes
  x dense/BM25/hybrid x rerank on/off). A retrieved chunk is relevant when it
  contains a gold quote, so every chunk size is scored on the same sentences.
  It calls `Retriever.warmup()` once per chunk size before timing anything.
- `results.json` — per-configuration summaries and per-query ranks behind
  `results.md`, for digging into individual misses.
- `results.md` — Day 4 output. The artefact the whole project exists to produce.

Do not cut this directory. Cut Gradio, then Docker, then the reranker, then the
grid down to six rows. The irreducible core is a gold set, retrieval metrics
against it, and an honest README.
