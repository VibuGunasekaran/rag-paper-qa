# rag-paper-qa

Retrieval augmented question answering over a corpus of arXiv papers on vision
language models and efficient inference, with a measured retrieval and
faithfulness evaluation.

> **Status: Days 1-2 complete (ingestion, indexing, retrieval).**
> Numbers below are placeholders until the gold set exists. Nothing in this
> README is a claim until it has a number next to it.

---

## Findings

*(Day 4 fills this in. Findings go first, before the method, because the
findings are the point. Leave it empty rather than filling it with adjectives.)*

| Chunk | Retriever | Rerank | R@1 | R@5 | MRR@10 | nDCG@10 | p95 latency |
|---|---|---|---|---|---|---|---|
| 512 | Dense | No | | | | | |
| 512 | BM25 | No | | | | | |
| 512 | Hybrid | No | | | | | |
| 512 | Hybrid | Yes | | | | | |
| 256 | Hybrid | Yes | | | | | |
| 1024 | Hybrid | Yes | | | | | |

Recall@5 by question type — this breakdown is where the interesting result
lives, not in the aggregate.

---

## Method

**Corpus.** 40-60 arXiv papers fetched via the arXiv API from a curated seed
list plus a keyword top-up. Parsed with PyMuPDF, split on detected section
headings, with everything from the references onward discarded.

**Chunking.** Section-aware, token-windowed at 256 / 512 / 1024 with 15%
overlap. Chunks carry `paper_id`, `section` and a stable id
(`2403.06764::s03::c02`), which is what makes hand-labelling the gold set
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

**Evaluation.** 50 hand-written question/answer pairs, each labelled with the
chunk that actually contains the answer: 20 single fact, 10 multi hop, 8
comparative, 7 paraphrased with no keyword overlap, and 5 deliberately
unanswerable. The unanswerable five exist to measure abstention.

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
python scripts/build_gold.py add
python scripts/build_gold.py validate

# Day 4: re-chunk and re-index for the ablation grid ------------------
python -m src.ingest --chunk-only --chunk-size 256
python -m src.ingest --chunk-only --chunk-size 1024
python -m src.index --all

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
- The gold set is written by one person, which means one person's judgement of
  what "contains the answer" means. Inter-annotator agreement is unmeasured.
