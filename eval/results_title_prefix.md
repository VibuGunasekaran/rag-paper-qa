# Retrieval results

Generated 2026-09-14 15:27 by `eval/run_eval.py` on macOS-15.7.7-arm64-arm-64bit (embedding device: mps).

- Questions scored: 135 answerable (15 unanswerable excluded; abstention is scored with generation).
- Embedding: `BAAI/bge-small-en-v1.5`; reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Candidate depth 20 per retriever, RRF k=60, top 10 scored.
- A retrieved chunk is relevant when it contains a gold quote. Latency is one run per query after warm-up.
- One question moves R@k by 0.7 points. Intervals are 95% percentile bootstraps over questions (5000 resamples, seed 0).

## All configurations

| Chunk | Retriever | Rerank | R@1 | R@3 | R@5 | R@5 95% CI | R@10 | MRR@10 | MRR@10 95% CI | nDCG@10 | Cov@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256 | Dense | No | 0.21 | 0.49 | 0.57 | [0.48, 0.65] | 0.71 | 0.370 | [0.31, 0.43] | 0.347 | 0.08 | 15 | 25 |
| 256 | Dense | Yes | 0.41 | 0.62 | 0.72 | [0.64, 0.79] | 0.78 | 0.534 | [0.46, 0.60] | 0.467 | 0.00 | 157 | 181 |
| 256 | BM25 | No | 0.44 | 0.60 | 0.75 | [0.67, 0.82] | 0.84 | 0.557 | [0.49, 0.63] | 0.510 | 0.12 | 3 | 8 |
| 256 | BM25 | Yes | 0.47 | 0.69 | 0.80 | [0.73, 0.87] | 0.87 | 0.605 | [0.54, 0.67] | 0.560 | 0.08 | 146 | 155 |
| 256 | Hybrid | No | 0.36 | 0.59 | 0.70 | [0.63, 0.78] | 0.83 | 0.497 | [0.43, 0.56] | 0.463 | 0.04 | 19 | 22 |
| 256 | Hybrid | Yes | 0.45 | 0.67 | 0.79 | [0.73, 0.86] | 0.87 | 0.593 | [0.53, 0.66] | 0.545 | 0.04 | 162 | 173 |
| 512 | Dense | No | 0.28 | 0.50 | 0.59 | [0.51, 0.67] | 0.70 | 0.410 | [0.34, 0.48] | 0.379 | 0.04 | 15 | 17 |
| 512 | Dense | Yes | 0.44 | 0.65 | 0.74 | [0.67, 0.81] | 0.81 | 0.563 | [0.49, 0.64] | 0.518 | 0.00 | 253 | 259 |
| 512 | BM25 | No | 0.50 | 0.69 | 0.81 | [0.75, 0.88] | 0.86 | 0.617 | [0.55, 0.68] | 0.563 | 0.08 | 2 | 3 |
| 512 | BM25 | Yes | 0.49 | 0.74 | 0.83 | [0.76, 0.89] | 0.88 | 0.630 | [0.56, 0.70] | 0.601 | 0.08 | 238 | 244 |
| 512 | Hybrid | No | 0.39 | 0.63 | 0.77 | [0.70, 0.84] | 0.86 | 0.541 | [0.47, 0.61] | 0.514 | 0.08 | 18 | 20 |
| 512 | Hybrid | Yes | 0.49 | 0.72 | 0.82 | [0.76, 0.88] | 0.88 | 0.628 | [0.56, 0.69] | 0.595 | 0.04 | 257 | 264 |
| 1024 | Dense | No | 0.23 | 0.49 | 0.59 | [0.50, 0.67] | 0.70 | 0.381 | [0.32, 0.45] | 0.366 | 0.04 | 15 | 17 |
| 1024 | Dense | Yes | 0.40 | 0.67 | 0.72 | [0.64, 0.79] | 0.81 | 0.543 | [0.48, 0.61] | 0.501 | 0.12 | 254 | 259 |
| 1024 | BM25 | No | 0.51 | 0.72 | 0.81 | [0.75, 0.87] | 0.87 | 0.624 | [0.56, 0.69] | 0.588 | 0.17 | 1 | 3 |
| 1024 | BM25 | Yes | 0.42 | 0.70 | 0.76 | [0.69, 0.83] | 0.82 | 0.569 | [0.50, 0.64] | 0.545 | 0.04 | 238 | 242 |
| 1024 | Hybrid | No | 0.40 | 0.61 | 0.70 | [0.61, 0.78] | 0.84 | 0.534 | [0.47, 0.60] | 0.505 | 0.12 | 17 | 20 |
| 1024 | Hybrid | Yes | 0.42 | 0.71 | 0.77 | [0.70, 0.84] | 0.88 | 0.576 | [0.51, 0.64] | 0.553 | 0.00 | 258 | 264 |

## Paired comparisons

Difference A - B on the same questions. Only intervals that exclude zero support a claim that A and B differ.

| A | B | Metric | A - B | 95% CI | Excludes 0 |
|---|---|---|---|---|---|
| 512 / BM25 / No | 512 / Dense / No | R@5 | +0.222 | [+0.133, +0.311] | yes |
| 512 / BM25 / No | 512 / Dense / No | MRR@10 | +0.207 | [+0.131, +0.281] | yes |
| 512 / Hybrid / No | 512 / BM25 / No | R@5 | -0.044 | [-0.111, +0.022] | no |
| 512 / Hybrid / No | 512 / BM25 / No | MRR@10 | -0.076 | [-0.138, -0.015] | yes |
| 512 / Hybrid / Yes | 512 / Hybrid / No | R@5 | +0.052 | [-0.015, +0.119] | no |
| 512 / Hybrid / Yes | 512 / Hybrid / No | MRR@10 | +0.087 | [+0.024, +0.147] | yes |
| 512 / Hybrid / Yes | 512 / BM25 / No | R@5 | +0.007 | [-0.052, +0.067] | no |
| 512 / Hybrid / Yes | 512 / BM25 / No | MRR@10 | +0.011 | [-0.052, +0.072] | no |
| 256 / Hybrid / Yes | 512 / Hybrid / Yes | R@5 | -0.030 | [-0.074, +0.015] | no |
| 256 / Hybrid / Yes | 512 / Hybrid / Yes | MRR@10 | -0.034 | [-0.078, +0.013] | no |
| 1024 / Hybrid / Yes | 512 / Hybrid / Yes | R@5 | -0.052 | [-0.111, +0.007] | no |
| 1024 / Hybrid / Yes | 512 / Hybrid / Yes | MRR@10 | -0.052 | [-0.099, -0.005] | yes |
| 1024 / BM25 / Yes | 1024 / BM25 / No | R@5 | -0.052 | [-0.126, +0.022] | no |
| 1024 / BM25 / Yes | 1024 / BM25 / No | MRR@10 | -0.055 | [-0.120, +0.009] | no |
| 1024 / Hybrid / Yes | 1024 / Hybrid / No | R@5 | +0.074 | [+0.007, +0.141] | yes |
| 1024 / Hybrid / Yes | 1024 / Hybrid / No | MRR@10 | +0.043 | [-0.013, +0.097] | no |

## Recall@5 by question type

| Chunk | Retriever | Rerank | single_fact (n=60) | multi_hop (n=30) | comparative (n=24) | paraphrased (n=21) |
|---|---|---|---|---|---|---|
| 256 | Dense | No | 0.70 | 0.70 | 0.33 | 0.29 |
| 256 | Dense | Yes | 0.85 | 0.87 | 0.54 | 0.33 |
| 256 | BM25 | No | 0.88 | 0.93 | 0.62 | 0.24 |
| 256 | BM25 | Yes | 0.92 | 0.97 | 0.67 | 0.38 |
| 256 | Hybrid | No | 0.82 | 0.80 | 0.58 | 0.38 |
| 256 | Hybrid | Yes | 0.93 | 0.97 | 0.58 | 0.38 |
| 512 | Dense | No | 0.72 | 0.67 | 0.42 | 0.33 |
| 512 | Dense | Yes | 0.87 | 0.87 | 0.58 | 0.38 |
| 512 | BM25 | No | 0.95 | 0.93 | 0.79 | 0.29 |
| 512 | BM25 | Yes | 0.90 | 1.00 | 0.79 | 0.43 |
| 512 | Hybrid | No | 0.90 | 0.83 | 0.67 | 0.43 |
| 512 | Hybrid | Yes | 0.92 | 1.00 | 0.71 | 0.43 |
| 1024 | Dense | No | 0.68 | 0.67 | 0.42 | 0.38 |
| 1024 | Dense | Yes | 0.80 | 0.80 | 0.62 | 0.48 |
| 1024 | BM25 | No | 0.95 | 0.90 | 0.83 | 0.29 |
| 1024 | BM25 | Yes | 0.80 | 0.93 | 0.79 | 0.38 |
| 1024 | Hybrid | No | 0.82 | 0.77 | 0.58 | 0.38 |
| 1024 | Hybrid | Yes | 0.83 | 0.93 | 0.75 | 0.38 |
