# rag-paper-qa

Retrieval augmented question answering over a corpus of arXiv papers on vision
language models and efficient inference, with a measured retrieval and
faithfulness evaluation.

> **Status: Days 1-4 complete (ingestion, indexing, retrieval, 50-row gold
> set, 18-configuration retrieval evaluation). Next: Day 5, generation and
> faithfulness.**
> Numbers below are placeholders until the gold set exists. Nothing in this
> README is a claim until it has a number next to it.

---

## Findings

Retrieval only, over 45 answerable questions: one question moves R@k by 2.2
points, so gaps of a few points are noise.

The table shows 6 of the 18 configurations the sweep runs (3 chunk sizes x 3
retrievers x rerank on/off); `eval/results.md` has the full grid.

| Chunk | Retriever | Rerank | R@1 | R@5 | MRR@10 | nDCG@10 | p95 latency |
|---|---|---|---|---|---|---|---|
| 512 | Dense | No | 0.33 | 0.71 | 0.482 | 0.461 | 24 ms |
| 512 | BM25 | No | 0.49 | 0.84 | 0.644 | 0.552 | 4 ms |
| 512 | Hybrid | No | 0.38 | 0.80 | 0.546 | 0.503 | 22 ms |
| 512 | Hybrid | Yes | 0.53 | 0.84 | 0.678 | 0.602 | 263 ms |
| 256 | Hybrid | Yes | 0.51 | 0.80 | 0.647 | 0.553 | 165 ms |
| 1024 | Hybrid | Yes | 0.56 | 0.91 | 0.699 | 0.649 | 262 ms |

Recall@5 by question type — this breakdown is where the interesting result
lives, not in the aggregate.

| Chunk | Retriever | Rerank | Single fact (20) | Multi hop (10) | Comparative (8) | Paraphrased (7) |
|---|---|---|---|---|---|---|
| 512 | Dense | No | 0.80 | 0.50 | 0.88 | 0.57 |
| 512 | BM25 | No | 0.90 | 0.90 | 1.00 | 0.43 |
| 512 | Hybrid | No | 0.85 | 0.60 | 1.00 | 0.71 |
| 512 | Hybrid | Yes | 0.90 | 0.80 | 0.88 | 0.71 |
| 256 | Hybrid | Yes | 0.80 | 0.90 | 0.88 | 0.57 |
| 1024 | Hybrid | Yes | 0.90 | 0.80 | 1.00 | 1.00 |

What the numbers show:

- **Reranking raised MRR@10 in all 9 chunk-size x retriever pairs:** by
  0.13-0.15 for dense, 0.09-0.14 for hybrid, and 0.02-0.17 for BM25, at a p95
  cost of 140-170 ms (256) to 240-265 ms (512, 1024).
- **Without reranking, BM25 beat dense at every chunk size** on both R@5 and
  MRR@10 (512: R@5 0.84 vs 0.71, MRR@10 0.644 vs 0.482). Rank fusion without
  reranking did not beat BM25 on MRR@10 at 512 (0.546 vs 0.644) or 1024 (0.614
  vs 0.673); only at 256 did it edge ahead (0.508 vs 0.498).
- **BM25 alone at 512 matched hybrid + rerank on R@5 (0.84 each)** at 4 ms p95
  instead of 263 ms; reranking bought ranking quality (MRR@10 0.678 vs 0.644),
  not recall.
- **The best configuration was 1024 / hybrid / rerank:** R@5 0.91 (41 of 45),
  MRR@10 0.699. Hybrid + rerank R@5 went 0.80, 0.84, 0.91 across 256, 512, 1024,
  a spread of five questions, and larger chunks are favoured by the relevance
  rule itself (see Limitations).
- **Paraphrased questions were BM25's weak spot:** R@5 0.43 at 512 against 0.57
  for dense. With n = 7 that is 3 questions against 4.
- **Comparative questions were rarely fully covered:** the best Cov@5 was 0.38
  (3 of 8), so the top 5 usually held evidence for only one of the two papers.
- **The most persistent miss (q017, missed in 15 of 18 configurations) is a
  naming gap:** the evidence chunk never mentions "ST3"; the name appears only
  in the paper title, which is not part of chunk text. q012 (LLaVA, missed in
  11 of 18) has the same shape. Prepending the paper title to each chunk before
  indexing is the obvious next experiment; it has not been run.

---

## Method

**Corpus.** 40-60 arXiv papers fetched via the arXiv API from a curated seed
list plus a keyword top-up. Parsed with PyMuPDF in native reading order
(position-sorted extraction interleaves the columns of two-column papers), split
on detected section headings, with everything from the references onward
discarded. The arXiv API abstract replaces the PDF's copy, so no paper has two
near-identical abstract chunks competing in retrieval.

**Chunking.** Section-aware, token-windowed at 256 / 512 / 1024 with 15%
overlap. Chunk text is a verbatim slice of the source, cut at tokenizer
offsets rather than decoded from token ids (the bge tokenizer is uncased), so
gold quotes match it exactly. Chunks carry `paper_id`, `section` and a stable
id (`2403.06764::s03::c02`), which is what makes hand-labelling the gold set
survivable.

**Retrieval.** Three retrievers behind one interface:

- Dense: `BAAI/bge-small-en-v1.5`, FAISS `IndexFlatIP` over L2-normalised
  vectors. Exact search — at this corpus size an approximate index costs recall
  and buys nothing.
- Lexical: `rank_bm25` BM25Okapi over the same chunks.
- Hybrid: reciprocal rank fusion, k=60. Rank-based, so no score normalisation
  between a cosine similarity and a BM25 score is needed.

**Reranking.** `cross-encoder/ms-marco-MiniLM-L-6-v2` over the top 20, down to
the top 5. Optional, and measured with latency so the cost is visible.

**Evaluation.** 50 question/answer pairs, each labelled with every chunk that
actually contains the answer: 20 single fact, 10 multi hop, 8 comparative, 7
paraphrased with no keyword overlap, and 5 deliberately unanswerable. The
unanswerable five exist to measure abstention. The set was drafted by Claude
and reviewed by the author; see Limitations.

A retrieved chunk is relevant when it contains a gold quote. R@k is the share
of answerable questions with a relevant chunk in the top k; MRR@10 and nDCG@10
use binary relevance; Cov@5 is the share of comparative questions whose top 5
includes evidence from every paper being compared. Latency is end-to-end
`retrieve()` time after a warm-up call.

---

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Day 1 --------------------------------------------------------------
python -m src.ingest --fetch-only     # resolve titles, prune papers.txt, rerun
python -m src.ingest                  # download, parse, chunk at 512
python -m src.index                   # FAISS + BM25 at 512

# Day 2 --------------------------------------------------------------
python -m src.retrieve "how does FastV decide which visual tokens to drop?" \
    --method hybrid --rerank
python -m src.retrieve "same question" --method bm25 --no-rerank   # compare

# Day 3 --------------------------------------------------------------
python scripts/build_gold.py browse -q "visual token pruning"   # explore first
python scripts/build_gold.py add
python scripts/build_gold.py stats
python scripts/build_gold.py validate

# Day 4: re-chunk and re-index for the ablation grid ------------------
python -m src.ingest --chunk-only --chunk-size 256
python -m src.ingest --chunk-only --chunk-size 1024
python -m src.index --all
python eval/run_eval.py               # 18 configs -> eval/results.md, results.json

# Tests (no models, no network) ---------------------------------------
python tests/test_pipeline.py
```

Everything the ablation sweeps lives in `config.yaml`, not in the code.

---

## A note on the gold set schema

Each gold row records `gold_chunks` (chunk ids) **and** `gold_quotes` (the exact
sentence in each chunk that carries the answer).

The quotes are not redundant. Chunk ids only mean something within one chunk
size, and Day 4 sweeps three of them — a gold set recorded as 512-token ids
alone cannot be scored at 256 or 1024, because those ids do not exist there.
Scoring against the quote makes recall comparable across the whole grid: a
retrieved chunk counts as a hit when it contains the gold sentence.

That only works if the quote fits inside a single chunk at every size, including
256 tokens with its 38-token overlap. `build_gold.py validate` checks each quote
against every chunk size that has an index and flags any that straddle a chunk
boundary; the fix is a shorter span.

---

## Repository layout

```
config.yaml            every knob the ablation grid varies
papers.txt             seed arXiv IDs
src/config.py          config loading
src/ingest.py          fetch, parse, section split, chunk
src/index.py           FAISS and BM25 builders
src/retrieve.py        dense, bm25, hybrid RRF, cross-encoder rerank
scripts/build_gold.py  gold set authoring and validation
tests/test_pipeline.py logic tests, no models required
eval/                  gold_set.jsonl, run_eval.py, results.md
data/                  PDFs, chunks, indexes (gitignored)
```

---

## Limitations

*(Fill this in honestly as you go. Seeds already known:)*

- PDF parsing is heuristic. Two-column academic PDFs, tables and equations all
  degrade to imperfect text, and heading detection is regex-based. Some chunks
  will contain figure captions and table fragments.
- The corpus is one narrow domain, so the results do not generalise to
  heterogeneous document sets.
- The gold set was drafted by Claude (Opus 5) and reviewed and approved as-is
  by the author, rather than written by the author. Gold chunks were found by
  exact-text search over the 512-token chunks rather than by the retrievers
  under test, so no retriever chose its own answers, but the questions were
  written with the corpus text in view, it is one judgement of what "contains
  the answer" means, and inter-annotator agreement is unmeasured.
- The gold set is small: 45 answerable questions, so one question moves R@k by
  about 2.2 points, and 8 comparative questions make Cov@5 very coarse.
  Differences of a few points between configurations are within that noise.
- The relevance rule (a chunk counts if it contains a gold quote) favours larger
  chunks: a 1024-token chunk is more likely to contain the sentence than a
  256-token one. The chunk-size comparison therefore overstates the benefit of
  large chunks, and it ignores that a generator must read four times as much
  text per retrieved 1024-token chunk.
- Latency is one run per query on one laptop (M1, MPS), after warm-up. p95 over
  45 queries is the third-slowest query, not a stable tail estimate.
- Heading detection still admits a few figure and table labels as section
  titles, and misses some real headings. Section labels are metadata only; they
  do not affect retrieval or scoring, but a missed heading merges two sections.
- Appendices that follow the references are discarded with them. Anything a
  paper states only in its appendix is outside the corpus.
- Front matter (title, authors, affiliations) is indexed as ordinary chunks.
- The first dense query and the first reranked query each load a model (5-7 s
  on an M1). Latency is only meaningful after `Retriever.warmup()`.
