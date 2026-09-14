# Retrieval results

Generated 2026-09-14 14:48 by `eval/run_eval.py` on macOS-15.7.7-arm64-arm-64bit (embedding device: mps).

- Questions scored: 45 answerable (5 unanswerable excluded; abstention is scored with generation).
- Embedding: `BAAI/bge-small-en-v1.5`; reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Candidate depth 20 per retriever, RRF k=60, top 10 scored.
- A retrieved chunk is relevant when it contains a gold quote. Latency is one run per query after warm-up.
- One question moves R@k by 2.2 points.

## All configurations

| Chunk | Retriever | Rerank | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 | Cov@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 256 | Dense | No | 0.31 | 0.51 | 0.60 | 0.78 | 0.445 | 0.407 | 0.00 | 15 | 62 |
| 256 | Dense | Yes | 0.51 | 0.64 | 0.71 | 0.76 | 0.596 | 0.494 | 0.12 | 145 | 167 |
| 256 | BM25 | No | 0.33 | 0.58 | 0.69 | 0.87 | 0.498 | 0.444 | 0.12 | 3 | 5 |
| 256 | BM25 | Yes | 0.53 | 0.80 | 0.82 | 0.87 | 0.667 | 0.564 | 0.25 | 132 | 143 |
| 256 | Hybrid | No | 0.36 | 0.58 | 0.78 | 0.82 | 0.508 | 0.455 | 0.00 | 19 | 23 |
| 256 | Hybrid | Yes | 0.51 | 0.76 | 0.80 | 0.89 | 0.647 | 0.553 | 0.25 | 147 | 165 |
| 512 | Dense | No | 0.33 | 0.53 | 0.71 | 0.84 | 0.482 | 0.461 | 0.12 | 15 | 24 |
| 512 | Dense | Yes | 0.53 | 0.71 | 0.78 | 0.80 | 0.634 | 0.547 | 0.25 | 255 | 258 |
| 512 | BM25 | No | 0.49 | 0.78 | 0.84 | 0.93 | 0.644 | 0.552 | 0.12 | 2 | 4 |
| 512 | BM25 | Yes | 0.51 | 0.82 | 0.84 | 0.91 | 0.670 | 0.597 | 0.25 | 237 | 241 |
| 512 | Hybrid | No | 0.38 | 0.73 | 0.80 | 0.84 | 0.546 | 0.503 | 0.12 | 18 | 22 |
| 512 | Hybrid | Yes | 0.53 | 0.78 | 0.84 | 0.93 | 0.678 | 0.602 | 0.38 | 259 | 263 |
| 1024 | Dense | No | 0.42 | 0.67 | 0.76 | 0.84 | 0.549 | 0.503 | 0.00 | 15 | 19 |
| 1024 | Dense | Yes | 0.58 | 0.71 | 0.84 | 0.87 | 0.678 | 0.606 | 0.25 | 255 | 257 |
| 1024 | BM25 | No | 0.53 | 0.80 | 0.82 | 0.93 | 0.673 | 0.592 | 0.12 | 2 | 3 |
| 1024 | BM25 | Yes | 0.56 | 0.80 | 0.87 | 0.91 | 0.696 | 0.638 | 0.25 | 237 | 241 |
| 1024 | Hybrid | No | 0.47 | 0.76 | 0.80 | 0.89 | 0.614 | 0.563 | 0.25 | 18 | 20 |
| 1024 | Hybrid | Yes | 0.56 | 0.78 | 0.91 | 0.96 | 0.699 | 0.649 | 0.38 | 259 | 262 |

## Recall@5 by question type

| Chunk | Retriever | Rerank | single_fact (n=20) | multi_hop (n=10) | comparative (n=8) | paraphrased (n=7) |
|---|---|---|---|---|---|---|
| 256 | Dense | No | 0.60 | 0.70 | 0.62 | 0.43 |
| 256 | Dense | Yes | 0.70 | 0.80 | 0.88 | 0.43 |
| 256 | BM25 | No | 0.75 | 0.70 | 0.75 | 0.43 |
| 256 | BM25 | Yes | 0.85 | 0.90 | 0.88 | 0.57 |
| 256 | Hybrid | No | 0.80 | 0.80 | 0.88 | 0.57 |
| 256 | Hybrid | Yes | 0.80 | 0.90 | 0.88 | 0.57 |
| 512 | Dense | No | 0.80 | 0.50 | 0.88 | 0.57 |
| 512 | Dense | Yes | 0.85 | 0.60 | 0.88 | 0.71 |
| 512 | BM25 | No | 0.90 | 0.90 | 1.00 | 0.43 |
| 512 | BM25 | Yes | 0.90 | 0.80 | 0.88 | 0.71 |
| 512 | Hybrid | No | 0.85 | 0.60 | 1.00 | 0.71 |
| 512 | Hybrid | Yes | 0.90 | 0.80 | 0.88 | 0.71 |
| 1024 | Dense | No | 0.80 | 0.60 | 0.88 | 0.71 |
| 1024 | Dense | Yes | 0.90 | 0.60 | 1.00 | 0.86 |
| 1024 | BM25 | No | 0.90 | 0.80 | 1.00 | 0.43 |
| 1024 | BM25 | Yes | 0.85 | 0.80 | 1.00 | 0.86 |
| 1024 | Hybrid | No | 0.85 | 0.50 | 1.00 | 0.86 |
| 1024 | Hybrid | Yes | 0.90 | 0.80 | 1.00 | 1.00 |
