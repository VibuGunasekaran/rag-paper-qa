# Retrieval results

Generated 2026-09-14 15:21 by `eval/run_eval.py` on macOS-15.7.7-arm64-arm-64bit (embedding device: mps).

- Questions scored: 135 answerable (15 unanswerable excluded; abstention is scored with generation).
- Embedding: `BAAI/bge-small-en-v1.5`; reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Candidate depth 20 per retriever, RRF k=60, top 10 scored.
- A retrieved chunk is relevant when it contains a gold quote. Latency is one run per query after warm-up.
- One question moves R@k by 0.7 points. Intervals are 95% percentile bootstraps over questions (5000 resamples, seed 0).

## All configurations

| Chunk | Retriever | Rerank | R@1 | R@3 | R@5 | R@5 95% CI | R@10 | MRR@10 | MRR@10 95% CI | nDCG@10 | Cov@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256 | Dense | No | 0.30 | 0.47 | 0.59 | [0.51, 0.67] | 0.69 | 0.417 | [0.35, 0.49] | 0.362 | 0.00 | 15 | 25 |
| 256 | Dense | Yes | 0.44 | 0.64 | 0.70 | [0.61, 0.77] | 0.76 | 0.551 | [0.48, 0.62] | 0.459 | 0.04 | 142 | 167 |
| 256 | BM25 | No | 0.37 | 0.56 | 0.70 | [0.62, 0.78] | 0.84 | 0.508 | [0.44, 0.58] | 0.472 | 0.08 | 3 | 4 |
| 256 | BM25 | Yes | 0.47 | 0.70 | 0.79 | [0.72, 0.86] | 0.84 | 0.604 | [0.54, 0.67] | 0.530 | 0.12 | 132 | 142 |
| 256 | Hybrid | No | 0.33 | 0.56 | 0.68 | [0.61, 0.76] | 0.80 | 0.469 | [0.40, 0.54] | 0.436 | 0.04 | 19 | 22 |
| 256 | Hybrid | Yes | 0.47 | 0.70 | 0.78 | [0.70, 0.84] | 0.85 | 0.602 | [0.54, 0.67] | 0.530 | 0.08 | 149 | 162 |
| 512 | Dense | No | 0.28 | 0.50 | 0.60 | [0.52, 0.68] | 0.78 | 0.425 | [0.36, 0.49] | 0.403 | 0.08 | 14 | 15 |
| 512 | Dense | Yes | 0.44 | 0.65 | 0.73 | [0.66, 0.81] | 0.79 | 0.557 | [0.49, 0.63] | 0.500 | 0.08 | 240 | 284 |
| 512 | BM25 | No | 0.47 | 0.68 | 0.79 | [0.71, 0.85] | 0.87 | 0.598 | [0.53, 0.66] | 0.551 | 0.08 | 2 | 4 |
| 512 | BM25 | Yes | 0.46 | 0.70 | 0.78 | [0.71, 0.84] | 0.87 | 0.598 | [0.53, 0.66] | 0.563 | 0.08 | 236 | 272 |
| 512 | Hybrid | No | 0.38 | 0.64 | 0.76 | [0.69, 0.84] | 0.84 | 0.527 | [0.46, 0.59] | 0.495 | 0.08 | 16 | 19 |
| 512 | Hybrid | Yes | 0.46 | 0.70 | 0.79 | [0.72, 0.85] | 0.87 | 0.603 | [0.54, 0.67] | 0.565 | 0.17 | 236 | 270 |
| 1024 | Dense | No | 0.33 | 0.50 | 0.59 | [0.51, 0.67] | 0.72 | 0.441 | [0.37, 0.51] | 0.389 | 0.08 | 15 | 19 |
| 1024 | Dense | Yes | 0.41 | 0.59 | 0.68 | [0.60, 0.76] | 0.75 | 0.523 | [0.45, 0.59] | 0.470 | 0.21 | 236 | 259 |
| 1024 | BM25 | No | 0.50 | 0.72 | 0.81 | [0.74, 0.87] | 0.89 | 0.623 | [0.56, 0.69] | 0.596 | 0.17 | 1 | 3 |
| 1024 | BM25 | Yes | 0.42 | 0.66 | 0.73 | [0.66, 0.81] | 0.84 | 0.562 | [0.49, 0.63] | 0.536 | 0.12 | 231 | 254 |
| 1024 | Hybrid | No | 0.36 | 0.64 | 0.71 | [0.63, 0.79] | 0.84 | 0.519 | [0.45, 0.59] | 0.494 | 0.21 | 16 | 17 |
| 1024 | Hybrid | Yes | 0.43 | 0.66 | 0.76 | [0.69, 0.83] | 0.87 | 0.573 | [0.50, 0.64] | 0.547 | 0.17 | 237 | 262 |

## Paired comparisons

Difference A - B on the same questions. Only intervals that exclude zero support a claim that A and B differ.

| A | B | Metric | A - B | 95% CI | Excludes 0 |
|---|---|---|---|---|---|
| 512 / BM25 / No | 512 / Dense / No | R@5 | +0.185 | [+0.096, +0.274] | yes |
| 512 / BM25 / No | 512 / Dense / No | MRR@10 | +0.173 | [+0.105, +0.242] | yes |
| 512 / Hybrid / No | 512 / BM25 / No | R@5 | -0.022 | [-0.089, +0.052] | no |
| 512 / Hybrid / No | 512 / BM25 / No | MRR@10 | -0.071 | [-0.130, -0.012] | yes |
| 512 / Hybrid / Yes | 512 / Hybrid / No | R@5 | +0.022 | [-0.037, +0.081] | no |
| 512 / Hybrid / Yes | 512 / Hybrid / No | MRR@10 | +0.076 | [+0.016, +0.138] | yes |
| 512 / Hybrid / Yes | 512 / BM25 / No | R@5 | +0.000 | [-0.067, +0.067] | no |
| 512 / Hybrid / Yes | 512 / BM25 / No | MRR@10 | +0.005 | [-0.055, +0.063] | no |
| 256 / Hybrid / Yes | 512 / Hybrid / Yes | R@5 | -0.007 | [-0.067, +0.052] | no |
| 256 / Hybrid / Yes | 512 / Hybrid / Yes | MRR@10 | -0.000 | [-0.051, +0.050] | no |
| 1024 / Hybrid / Yes | 512 / Hybrid / Yes | R@5 | -0.022 | [-0.096, +0.052] | no |
| 1024 / Hybrid / Yes | 512 / Hybrid / Yes | MRR@10 | -0.030 | [-0.082, +0.021] | no |
| 1024 / BM25 / Yes | 1024 / BM25 / No | R@5 | -0.074 | [-0.148, +0.007] | no |
| 1024 / BM25 / Yes | 1024 / BM25 / No | MRR@10 | -0.061 | [-0.128, +0.010] | no |
| 1024 / Hybrid / Yes | 1024 / Hybrid / No | R@5 | +0.052 | [-0.015, +0.119] | no |
| 1024 / Hybrid / Yes | 1024 / Hybrid / No | MRR@10 | +0.054 | [-0.009, +0.115] | no |

## Recall@5 by question type

| Chunk | Retriever | Rerank | single_fact (n=60) | multi_hop (n=30) | comparative (n=24) | paraphrased (n=21) |
|---|---|---|---|---|---|---|
| 256 | Dense | No | 0.70 | 0.73 | 0.38 | 0.33 |
| 256 | Dense | Yes | 0.80 | 0.87 | 0.54 | 0.33 |
| 256 | BM25 | No | 0.87 | 0.83 | 0.54 | 0.24 |
| 256 | BM25 | Yes | 0.88 | 0.93 | 0.83 | 0.29 |
| 256 | Hybrid | No | 0.80 | 0.83 | 0.50 | 0.33 |
| 256 | Hybrid | Yes | 0.87 | 0.93 | 0.67 | 0.43 |
| 512 | Dense | No | 0.75 | 0.63 | 0.46 | 0.29 |
| 512 | Dense | Yes | 0.85 | 0.77 | 0.67 | 0.43 |
| 512 | BM25 | No | 0.92 | 0.93 | 0.75 | 0.24 |
| 512 | BM25 | Yes | 0.90 | 0.87 | 0.75 | 0.33 |
| 512 | Hybrid | No | 0.87 | 0.80 | 0.79 | 0.38 |
| 512 | Hybrid | Yes | 0.90 | 0.87 | 0.71 | 0.43 |
| 1024 | Dense | No | 0.65 | 0.67 | 0.54 | 0.38 |
| 1024 | Dense | Yes | 0.73 | 0.70 | 0.67 | 0.52 |
| 1024 | BM25 | No | 0.93 | 0.90 | 0.83 | 0.29 |
| 1024 | BM25 | Yes | 0.80 | 0.83 | 0.75 | 0.38 |
| 1024 | Hybrid | No | 0.77 | 0.73 | 0.79 | 0.43 |
| 1024 | Hybrid | Yes | 0.82 | 0.83 | 0.75 | 0.52 |
