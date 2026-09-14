# eval/

- `gold_set.jsonl` — Day 3. Written by hand via `scripts/build_gold.py add`.
- `run_eval.py` — Day 4. Recall@{1,3,5,10}, MRR@10, nDCG@10, plus the 18-config
  ablation sweep and p50/p95 latency.
- `results.md` — Day 4 output. The artefact the whole project exists to produce.

Do not cut this directory. Cut Gradio, then Docker, then the reranker, then the
grid down to six rows. The irreducible core is a gold set, retrieval metrics
against it, and an honest README.
