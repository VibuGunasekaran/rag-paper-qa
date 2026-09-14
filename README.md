# rag-paper-qa

Retrieval augmented question answering over a corpus of arXiv papers on vision
language models and efficient inference, with a measured retrieval and
faithfulness evaluation.

> **Status: Days 1-5 complete (ingestion, indexing, retrieval, 150-question
> gold set, 18-configuration retrieval evaluation, cited generation with
> abstention and an LLM-judged faithfulness evaluation). Next: Day 6, serving.**
> Every claim below has a number and an interval next to it.

---

## Findings

Retrieval only, over the 135 answerable questions of the 150-question gold set
(one question moves R@k by 0.7 points).

The table shows 8 of the 18 configurations the sweep runs (3 chunk sizes x 3
retrievers x rerank on/off); `eval/results.md` has the full grid.

| Chunk | Retriever | Rerank | R@1 | R@5 | R@5 95% CI | MRR@10 | MRR@10 95% CI | nDCG@10 | p95 latency |
|---|---|---|---|---|---|---|---|---|---|
| 512 | Dense | No | 0.28 | 0.60 | [0.52, 0.68] | 0.425 | [0.36, 0.49] | 0.403 | 15 ms |
| 512 | BM25 | No | 0.47 | 0.79 | [0.71, 0.85] | 0.598 | [0.53, 0.66] | 0.551 | 4 ms |
| 512 | Hybrid | No | 0.38 | 0.76 | [0.69, 0.84] | 0.527 | [0.46, 0.59] | 0.495 | 19 ms |
| 512 | Hybrid | Yes | 0.46 | 0.79 | [0.72, 0.85] | 0.603 | [0.54, 0.67] | 0.565 | 270 ms |
| 256 | Hybrid | Yes | 0.47 | 0.78 | [0.70, 0.84] | 0.602 | [0.54, 0.67] | 0.530 | 162 ms |
| 1024 | Hybrid | Yes | 0.43 | 0.76 | [0.69, 0.83] | 0.573 | [0.50, 0.64] | 0.547 | 262 ms |
| 1024 | BM25 | No | 0.50 | 0.81 | [0.74, 0.87] | 0.623 | [0.56, 0.69] | 0.596 | 3 ms |
| 1024 | BM25 | Yes | 0.42 | 0.73 | [0.66, 0.81] | 0.562 | [0.49, 0.63] | 0.536 | 254 ms |

Intervals are 95% bootstraps over questions. Claims below rest on the paired
comparisons in `eval/results.md`, which resample both configurations on the same
questions.

Recall@5 by question type — this breakdown is where the interesting result
lives, not in the aggregate.

| Chunk | Retriever | Rerank | Single fact (60) | Multi hop (30) | Comparative (24) | Paraphrased (21) |
|---|---|---|---|---|---|---|
| 512 | Dense | No | 0.75 | 0.63 | 0.46 | 0.29 |
| 512 | BM25 | No | 0.92 | 0.93 | 0.75 | 0.24 |
| 512 | Hybrid | No | 0.87 | 0.80 | 0.79 | 0.38 |
| 512 | Hybrid | Yes | 0.90 | 0.87 | 0.71 | 0.43 |
| 256 | Hybrid | Yes | 0.87 | 0.93 | 0.67 | 0.43 |
| 1024 | Hybrid | Yes | 0.82 | 0.83 | 0.75 | 0.52 |
| 1024 | BM25 | No | 0.93 | 0.90 | 0.83 | 0.29 |
| 1024 | BM25 | Yes | 0.80 | 0.83 | 0.75 | 0.38 |

Supported by paired intervals:

- **BM25 beats dense retrieval on both recall and ranking** (512, no rerank):
  R@5 +0.19 [+0.10, +0.27], MRR@10 +0.17 [+0.11, +0.24]. On the first 45
  questions only the ranking gap was resolved; with 135 both are.
- **Rank fusion without reranking ranks below BM25 alone** (512): MRR@10 -0.07
  [-0.13, -0.01], with recall unchanged (R@5 -0.02 [-0.09, +0.05]). Adding a
  weaker dense ranking dilutes a stronger lexical one.
- **The reranker rescues weak first-stage rankings but adds nothing to a strong
  one.** It raises R@5 for dense retrieval at every chunk size (+0.09 to +0.13)
  and for every retriever at 256, but for BM25 at 512 it changes nothing
  (R@5 -0.01 [-0.07, +0.06]).
- **The reranker cannot read most of a long chunk.** It accepts 512 tokens, 42%
  of 1024-token chunks are longer than that, and for 19 of 135 questions every
  gold sentence lies past the cut-off at 1024 (none do at 256 or 512). Reranking
  BM25 at 1024 moved R@5 by -0.07 [-0.15, +0.01], not resolved, but 9 of the 20
  questions it lost had evidence only past the cut-off, against a 14% base rate.

Not supported (intervals include zero):

- **Hybrid + rerank vs plain BM25** (512): R@5 +0.00 [-0.07, +0.07], MRR@10
  +0.01 [-0.06, +0.06]. BM25 alone finds and ranks evidence as well, at 4 ms p95
  instead of 270 ms.
- **Chunk size** (hybrid + rerank): 256 vs 512 R@5 -0.01 [-0.07, +0.05], 1024 vs
  512 R@5 -0.02 [-0.10, +0.05]. Any real effect is small.
- **Dense beating BM25 on paraphrased questions:** dense minus BM25 R@5 is +0.05
  to +0.14 across chunk sizes and rerank settings, and every interval includes
  zero (21 questions).

Observations:

- **Paraphrased questions are the weak spot for every configuration:** R@5 from
  0.24 to 0.52.
- **Comparative questions are rarely fully covered:** the best Cov@5 is 0.21 (5
  of 24); the top 5 usually holds evidence for only one of the papers compared.
- **Naming gap:** on the first 45 questions the most persistent miss (q017,
  missed in 15 of 18 configurations) was an evidence chunk that never names the
  method ("ST3"); the name is only in the paper title, which is not part of
  chunk text.

### Experiment: prepending the paper title to every chunk

Retrievers and the reranker saw `"<paper title>\n\n<chunk>"`; the stored chunk
text and gold matching were unchanged. Built with
`python -m src.index --all --title-prefix`, scored with
`python eval/run_eval.py --title-prefix`, compared with
`python eval/run_eval.py --compare eval/results.json eval/results_title_prefix.json`
(`eval/compare_results_title_prefix.md`).

- **It does not fix the miss it was built for:** q017 (ST3) is retrieved in 4
  of 18 configurations with the title and 3 without.
- **Across the grid the effect is small and mixed.** Of 18 paired comparisons
  on 135 questions, two intervals exclude zero, in opposite directions: BM25 at
  256 ranks better (MRR@10 +0.05 [+0.02, +0.08]) and dense at 1024 ranks worse
  (MRR@10 -0.06 [-0.11, -0.01]). BM25 + rerank at 512 gains R@5 +0.05
  [0.00, +0.10], at the edge of noise.
- **Not adopted:** `title_prefix` stays off. One plausible reading: the repeated
  title helps lexical matching on paper names but pulls every chunk of a paper
  toward the same point in embedding space.

### Generation: cited answers with abstention

All 150 questions, answered by `claude-opus-5` from the top 5 chunks of 512-token
hybrid + rerank retrieval, and judged by `claude-opus-5` against the sources it
was shown (faithfulness) and the gold answer (correctness). Full tables:
`eval/results_generation.md`. One run cost about $7.40, with no refusals and no
malformed responses.

| Metric | Rate [95% CI] | n |
|---|---|---|
| Gold evidence in the top 5 (answerable) | 0.79 [0.72, 0.85] | 135 |
| Correct (answerable) | 0.82 [0.76, 0.88] | 135 |
| Correct, when gold evidence was retrieved | 0.95 [0.91, 0.99] | 106 |
| Every claim supported by the sources (answered) | 1.00 | 124 |
| Abstained on unanswerable questions | 1.00 | 15 |
| Abstained although gold evidence was retrieved | 0.01 [0.00, 0.03] | 106 |
| Abstained when gold evidence was not retrieved | 0.34 [0.17, 0.52] | 29 |

| Type | Correct [95% CI] |
|---|---|
| Single fact (60) | 0.93 [0.87, 0.98] |
| Multi hop (30) | 0.90 [0.80, 1.00] |
| Comparative (24) | 0.58 [0.38, 0.79] |
| Paraphrased (21) | 0.67 [0.48, 0.86] |
| Unanswerable (15, abstention counts as correct) | 1.00 |

- **Retrieval is the bottleneck, not generation.** Of the 24 answerable
  questions not judged correct, 19 had no gold evidence in the top 5. With the
  evidence in hand, 101 of 106 answers are correct and the other 5 are partial:
  a comparative or multi-hop question where the top 5 held evidence for only one
  side, and the answer said so.
- **The weak question types match the Day 4 weak spots.** Comparative (Cov@5 was
  0.21) and paraphrased questions are the two types whose correctness intervals
  sit below single-fact's.
- **No fabricated claims were found.** The judge marked all 124 answers fully
  supported. As a check that does not rely on the judge, every number in every
  answer (335) appears in the sources that answer was shown; the only mismatch
  was a checker artefact.
- **Abstention is well calibrated at both ends.** 15 of 15 unanswerable
  questions abstained, and only 1 of 106 answerable questions with evidence did
  (q126, a comparison where only one method was retrieved; it answered that half).
- **On a retrieval miss the model abstains, answers correctly from other
  chunks, or answers a neighbouring question.** Of the 29 misses: 10 abstained,
  10 were correct anyway, 5 partial, 4 incorrect. The 10 correct ones show the
  gold-quote rule undercounts evidence (q040's LoRA merge is also stated in a
  chunk the gold set did not list). The 4 incorrect answers are not invented:
  each states that the sources don't cover the question, then reports a related
  figure (q083 gives the memory saved by prompt sharing, 6.1-9.8%, when asked
  for the prompt's 12% share). A stricter prompt could turn these into
  abstentions.

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

**Evaluation.** 150 question/answer pairs over all 50 papers, each labelled
with every chunk that actually contains the answer: 60 single fact, 30 multi
hop, 24 comparative, 21 paraphrased with little keyword overlap, and 15
deliberately unanswerable. The unanswerable fifteen exist to measure
abstention. The set was drafted by Claude; see Limitations.

A retrieved chunk is relevant when it contains a gold quote. R@k is the share
of answerable questions with a relevant chunk in the top k; MRR@10 and nDCG@10
use binary relevance; Cov@5 is the share of comparative questions whose top 5
includes evidence from every paper being compared. Latency is end-to-end
`retrieve()` time after a warm-up call.

**Generation.** `src/generate.py` gives Claude the top 5 chunks as numbered
sources and asks for a short answer with inline `[S#]` citations, or an
explicit abstention when the sources do not contain the answer. Output is
structured (`abstained`, `answer`, `cited_sources`); citations are checked
against the source list and the inline markers. Server-side refusal fallbacks
are on, and each answer records which model served it. `eval/judge.py` scores
faithfulness against the sources only and correctness against the gold answer,
as separate verdicts. Unanswerable questions are scored on the abstention flag
itself. Every API response is cached by request hash
(`eval/generation_cache.jsonl`), so re-running the evaluation is free.

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

# Day 5: generation and faithfulness (Claude API) ---------------------
cp .env.example .env                  # then add ANTHROPIC_API_KEY
python eval/run_generation.py --dry-run   # retrieval + cost estimate, no API calls
python eval/run_generation.py --limit 6   # smoke test across question types (~$0.30)
python eval/run_generation.py             # all 150 -> eval/results_generation.md (~$7.40; cached re-runs free)

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
src/generate.py        cited answers with abstention (Claude API, refusal fallbacks)
scripts/build_gold.py  gold set authoring and validation
tests/test_pipeline.py logic tests, no models or network required
eval/                  gold_set.jsonl, run_eval.py (retrieval), run_generation.py +
                       judge.py (generation, faithfulness), results*.md
data/                  PDFs, chunks, indexes (gitignored)
```

---

## Limitations

- PDF parsing is heuristic. Two-column academic PDFs, tables and equations all
  degrade to imperfect text, and heading detection is regex-based. Some chunks
  will contain figure captions and table fragments.
- The corpus is one narrow domain, so the results do not generalise to
  heterogeneous document sets. The keyword top-up also pulled in a few
  off-topic papers (underwater segmentation, text-image retrieval, SAM-based
  referring segmentation); they are kept as realistic distractors.
- The gold set was drafted by Claude (Opus 5): the first 50 questions were
  reviewed and approved by the author, the 100 added later were drafted the
  same way. Gold chunks were found by exact-text search over the 512-token
  chunks rather than by the retrievers under test, so no retriever chose its
  own answers, but the questions were written with the corpus text in view, it
  is one judgement of what "contains the answer" means, and inter-annotator
  agreement is unmeasured.
- 135 answerable questions give intervals of roughly +/-7 points of R@5 on a
  single configuration. Per-type samples (21 paraphrased, 24 comparative) are
  still too small for type-level claims.
- The relevance rule (a chunk counts if it contains a gold quote) favours larger
  chunks: a 1024-token chunk is more likely to contain the sentence than a
  256-token one. The chunk-size comparison therefore overstates the benefit of
  large chunks, and it ignores that a generator must read four times as much
  text per retrieved 1024-token chunk.
- The judge is the same model as the generator, and its verdicts have not been
  checked against human grading, so self-preference could inflate faithfulness
  and correctness. The independent number check covers figures only, not
  wording. The judge also labels some abstentions "partial" rather than
  "abstained", which puts 3 abstentions into the "correct or partial" rate.
- Generation was run once; answers at a different time or with a different
  model version would vary. The 29 retrieval misses are too few to compare
  abstention against answering a neighbouring question with any precision.
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
