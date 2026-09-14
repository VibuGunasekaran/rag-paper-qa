"""Logic tests that need no models, no network and no index.

These cover the parts most likely to be quietly wrong: heading detection,
section splitting, the chunk window arithmetic, and RRF. Run them before
trusting a number that came out of the eval.

    python tests/test_pipeline.py
"""
from __future__ import annotations

import math
import re
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# tqdm is a progress bar, not logic. Stub it so tests run on a bare interpreter.
if "tqdm" not in sys.modules:
    try:
        import tqdm  # noqa: F401
    except ImportError:
        mod = types.ModuleType("tqdm")
        mod.tqdm = lambda it, **kw: it
        sys.modules["tqdm"] = mod

from src.ingest import _is_heading, chunk_section, split_sections  # noqa: E402
from src.index import bm25_tokenize, indexed_text  # noqa: E402
from src.retrieve import reciprocal_rank_fusion  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from run_eval import bootstrap_ci, mrr, ndcg, percentile, recall_at, relevant_flags  # noqa: E402


class WordCounter:
    """Stand-in for the bge tokenizer: one token per word."""

    def encode(self, text): return text.split()
    def spans(self, text): return [m.span() for m in re.finditer(r"\S+", text)]


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    return bool(cond)


def test_headings():
    ok = True
    ok &= check("numbered heading", _is_heading("3.2 Query-Aware Token Selection") is not None)
    ok &= check("named heading", _is_heading("Abstract") == "Abstract")
    ok &= check("body text rejected",
                _is_heading("We evaluate on POPE and report accuracy.") is None)
    ok &= check("sentence-like number rejected",
                _is_heading("3 of the models were trained on ImageNet and then fine tuned.") is None)
    ok &= check("long line rejected", _is_heading("A" * 120) is None)
    ok &= check("figure axis + body text rejected",
                _is_heading("40 To address the above limitations, we propose PagedAt-") is None)
    ok &= check("table row rejected", _is_heading("65 CLIP 98.4 76.2 58.5") is None)
    ok &= check("uppercase subsection accepted",
                _is_heading("4.2 APPLYING LORA TO TRANSFORMER") is not None)
    ok &= check("table captions rejected",
                _is_heading("TABLE VI") is None and _is_heading("8 TABLE VIII") is None)
    ok &= check("numbered sentence rejected",
                _is_heading("7 We obverse that language conversations are often longer") is None)
    ok &= check("title fragment ending mid-phrase rejected",
                _is_heading("PARALLEL VISION TOKEN SCHEDULING FOR FAST AND") is None)
    ok &= check("ordinary headings still accepted",
                _is_heading("2.3 Speculative Sampling") is not None
                and _is_heading("5 EXPERIMENTAL VALIDATION") is not None)
    return ok


def test_split_sections():
    text = "\n".join([
        "Introduction",
        "Vision language models are large.",
        "",
        "2 Method",
        "We prune visual tokens using attention.",
        "",
        "4.2",
        "APPLYING LORA TO TRANSFORMER",
        "In principle, we can apply LoRA to any subset of weight matrices.",
        "40",
        "To address the above limitations, we propose PagedAt-",
        "2.47",
        "GPT-2 L (LoRA)",
        "2",
        "WikiSQL (±0.5%)",
        "References",
        "[1] Someone et al.",
    ])
    sections = split_sections(text, {"references"})
    titles = [t for t, _ in sections]
    bodies = " ".join(b for _, b in sections)
    ok = True
    ok &= check("intro captured", any("Introduction" in t for t in titles))
    ok &= check("method captured", any("Method" in t for t in titles))
    ok &= check("section number on its own line rejoined",
                "4.2 APPLYING LORA TO TRANSFORMER" in titles)
    ok &= check("bare number before body text stays body",
                "To address the above limitations" in bodies
                and not any(t.startswith("40") for t in titles))
    ok &= check("table cells split across lines are not headings",
                not any(t.startswith(("2.47", "2 WikiSQL")) for t in titles)
                and "GPT-2 L (LoRA)" in bodies)
    ok &= check("references dropped", "Someone et al" not in bodies)
    ok &= check("body text preserved", "prune visual tokens" in bodies)
    return ok


def test_chunking():
    counter = WordCounter()
    text = " ".join(f"w{i}" for i in range(1000))
    chunks = chunk_section(text, counter, chunk_size=100, overlap=15, min_tokens=40)
    ok = True
    ok &= check("multiple chunks produced", len(chunks) > 1)
    ok &= check("no chunk exceeds the window",
                all(len(c.split()) <= 100 for c in chunks[:-1]))
    ok &= check("first token present", chunks[0].split()[0] == "w0")
    ok &= check("last token present", chunks[-1].split()[-1] == "w999")
    ok &= check("consecutive chunks overlap",
                chunks[0].split()[-15:] == chunks[1].split()[:15])
    # The tail-merge path: a length that leaves a stub at the end.
    stubby = " ".join(f"w{i}" for i in range(190))
    merged = chunk_section(stubby, counter, chunk_size=100, overlap=15, min_tokens=40)
    flat = " ".join(merged).split()
    ok &= check("merged tail does not duplicate the overlap",
                flat.count("w180") == 1 or len(merged) == 1)
    ok &= check("merged tail keeps the final token", merged[-1].split()[-1] == "w189")
    cased = "QLoRA's 4-bit NormalFloat (NF4) type. " * 60
    kept = chunk_section(cased, counter, chunk_size=100, overlap=15, min_tokens=40)
    ok &= check("chunks are verbatim slices of the source text",
                len(kept) > 1 and all(c in cased for c in kept) and "(NF4)" in kept[0])
    short = chunk_section("too short", counter, 100, 15, min_tokens=40)
    ok &= check("sub-minimum section dropped", short == [])
    return ok


def test_rrf():
    # doc_b is second in both lists; doc_a is first in one and absent from the
    # other. Agreement across retrievers is what RRF is for, so b must win.
    a = ["doc_a", "doc_b", "doc_c"]
    b = ["doc_d", "doc_b", "doc_e"]
    fused = reciprocal_rank_fusion([a, b], k=60)
    ranked = [d for d, _ in fused]
    ok = True
    ok &= check("consensus doc ranks first", ranked[0] == "doc_b")
    ok &= check("all docs retained", len(ranked) == 5)
    expected = 1 / 62 + 1 / 62   # rank 2 in both lists
    ok &= check("score matches the RRF formula",
                abs(dict(fused)["doc_b"] - expected) < 1e-9)
    ok &= check("scores descend",
                all(fused[i][1] >= fused[i + 1][1] for i in range(len(fused) - 1)))
    return ok


def test_bm25_tokenize():
    toks = bm25_tokenize("The FastV method prunes visual tokens at layer-2.")
    ok = True
    ok &= check("stopwords removed", "the" not in toks and "at" not in toks)
    ok &= check("lowercased", "fastv" in toks)
    ok &= check("hyphenated term kept whole", "layer-2" in toks)
    return ok


def test_generation_helpers():
    from types import SimpleNamespace as NS

    from src.generate import _to_answer, build_user_message, inline_citations

    hits = [NS(chunk_id=f"p::s00::c0{i}", title="Paper", section="Method", text=f"text {i}") for i in (1, 2, 3)]
    msg = build_user_message("What is X?", hits)
    ok = True
    ok &= check("sources are numbered S1..Sk", all(f'id="S{i}"' in msg for i in (1, 2, 3)))
    ok &= check("question follows the sources", msg.endswith("Question: What is X?"))
    ok &= check("inline citations parsed in both styles",
                inline_citations("A [S1][S3]. B [S12, S3]. Not a citation [1].") == [1, 3, 12])

    def response(stop_reason, parsed):
        return NS(stop_reason=stop_reason, parsed_output=parsed, model="claude-opus-5",
                  usage=NS(to_dict=lambda: {"input_tokens": 10, "output_tokens": 5}))

    ans = _to_answer(response("end_turn", NS(abstained=False, answer="X is Y [S2][S9].", cited_sources=[9, 2])),
                     "What is X?", hits, 12.3, "claude-opus-5")
    ok &= check("citations map to chunk ids, out-of-range numbers dropped", ans.cited_chunk_ids == ["p::s00::c02"])
    ok &= check("usage is recorded", ans.usage == {"input_tokens": 10, "output_tokens": 5})
    refused = _to_answer(response("refusal", None), "q", hits, 1.0, "claude-opus-5")
    ok &= check("refusals are recorded, not counted as abstentions", refused.refused and not refused.abstained)
    truncated = _to_answer(response("max_tokens", None), "q", hits, 1.0, "claude-opus-5")
    ok &= check("missing structured output is a parse failure", truncated.parse_failed and not truncated.refused)

    from run_generation import summarise

    def record(qtype, abstained, correctness):
        gen = {"refused": False, "parse_failed": False, "abstained": abstained,
               "model": "claude-opus-5", "requested_model": "claude-opus-5", "usage": {}}
        judge = {"refused": False, "parse_failed": False, "faithfulness": "full",
                 "correctness": correctness, "model": "claude-opus-5", "usage": {}}
        return {"type": qtype, "generation": gen, "judge": judge, "evidence_retrieved": True,
                "citations_valid": True, "cites_gold": True}

    s = summarise([record("unanswerable", True, "abstained"), record("unanswerable", False, "incorrect"),
                   record("single_fact", False, "correct"), record("single_fact", True, "abstained")])
    ok &= check("abstaining on an unanswerable question counts as correct",
                s["correct_unanswerable"]["rate"] == 0.5 and s["by_type"]["unanswerable"]["correct"]["rate"] == 0.5)
    ok &= check("abstaining on an answerable question does not", s["by_type"]["single_fact"]["correct"]["rate"] == 0.5)
    return ok


def test_serving():
    from types import SimpleNamespace as NS

    from fastapi.testclient import TestClient

    from src.generate import Answer
    from src.serve import Pipeline, create_app

    hits = [NS(chunk_id=f"2309.06180::s03::c0{i}", paper_id="2309.06180", title="vLLM", section="Method",
               text=f"text {i}") for i in (1, 2)]

    class FakeRetriever:
        chunks = [{}] * 7
        def warmup(self, **kw): pass
        def retrieve(self, question, **kw): return NS(hits=hits, timings_ms={"total_ms": 12.5})

    class FakeGenerator:
        model = "claude-opus-5"
        parse_failed = False
        def answer(self, question, hits):
            return Answer(question=question, abstained=False, answer="Block size is 16 [S2].", cited_sources=[2],
                          cited_chunk_ids=[hits[1].chunk_id], source_chunk_ids=[h.chunk_id for h in hits],
                          requested_model=self.model, model=self.model, stop_reason="end_turn", refused=False,
                          parse_failed=self.parse_failed, latency_ms=900.0)

    gen = FakeGenerator()
    pipeline = Pipeline(retriever=FakeRetriever(), generator=gen)
    ok = True
    with TestClient(create_app(pipeline)) as client:
        ok &= check("UI page served", "Ask the papers" in client.get("/").text)
        health = client.get("/health").json()
        ok &= check("health reports the evaluated configuration",
                    health["chunks"] == 7 and health["model"] == "claude-opus-5" and "method" in health)
        body = client.post("/ask", json={"question": "  What is the block size?  "}).json()
        ok &= check("answer returned with its sources numbered from 1",
                    body["answer"] == "Block size is 16 [S2]." and [s["number"] for s in body["sources"]] == [1, 2])
        ok &= check("only cited sources are flagged", [s["cited"] for s in body["sources"]] == [False, True])
        ok &= check("sources link to arXiv", body["sources"][0]["url"] == "https://arxiv.org/abs/2309.06180")
        ok &= check("latency split by stage",
                    body["timings_ms"] == {"retrieval_ms": 12.5, "generation_ms": 900.0})
        ok &= check("blank question rejected", client.post("/ask", json={"question": "   "}).status_code == 422)
        gen.parse_failed = True
        ok &= check("missing structured answer is a 502, not an empty answer",
                    client.post("/ask", json={"question": "q"}).status_code == 502)
    return ok


def test_indexed_text():
    chunk = {"title": "ST3: Accelerating MLLMs", "text": "we avoid pruning tokens in the first three layers"}
    ok = True
    ok &= check("title prefix leads the indexed text",
                indexed_text(chunk, True)
                == "ST3: Accelerating MLLMs\n\nwe avoid pruning tokens in the first three layers")
    ok &= check("without the prefix the chunk text is unchanged", indexed_text(chunk, False) == chunk["text"])
    ok &= check("a missing title adds nothing", indexed_text({"text": "body"}, True) == "body")
    return ok


def test_metrics():
    flags = [False, True, False, True]   # relevant at ranks 2 and 4
    ok = True
    ok &= check("recall@1 misses", recall_at(flags, 1) == 0.0)
    ok &= check("recall@2 hits", recall_at(flags, 2) == 1.0)
    ok &= check("MRR uses the first relevant rank", mrr(flags) == 0.5)
    ok &= check("MRR is zero past the cutoff", mrr(flags, k=1) == 0.0)
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    ok &= check("nDCG matches the binary-gain formula", abs(ndcg(flags, 2) - expected) < 1e-9)
    ok &= check("perfect ranking scores 1", ndcg([True, True, False], 2) == 1.0)
    ok &= check("nDCG is zero when nothing relevant exists", ndcg([False, False], 0) == 0.0)
    ok &= check("quote match ignores case and whitespace",
                relevant_flags(["The  KV\ncache takes 800 KB", "unrelated"],
                               ["kv cache takes 800 kb"]) == [True, False])
    ok &= check("nearest-rank p95 of 1..100 is 95", percentile(list(range(1, 101)), 95) == 95)
    ok &= check("p50 of one value is that value", percentile([7.0], 50) == 7.0)
    ok &= check("bootstrap CI of a constant is that constant",
                bootstrap_ci([0.5] * 20, n_resamples=200) == (0.5, 0.5))
    ok &= check("bootstrap CI of identical paired runs is zero",
                bootstrap_ci([1.0 - 1.0] * 20, n_resamples=200) == (0.0, 0.0))
    lo, hi = bootstrap_ci([0.0] * 10 + [1.0] * 10, n_resamples=500)
    ok &= check("bootstrap CI brackets the mean", lo < 0.5 < hi and 0.0 <= lo and hi <= 1.0)
    ok &= check("bootstrap CI is reproducible with a seed",
                bootstrap_ci([0.0, 1.0, 1.0], n_resamples=300) == bootstrap_ci([0.0, 1.0, 1.0], n_resamples=300))
    return ok


def main() -> int:
    results = []
    for name, fn in [
        ("heading detection", test_headings),
        ("section splitting", test_split_sections),
        ("chunking", test_chunking),
        ("reciprocal rank fusion", test_rrf),
        ("bm25 tokenisation", test_bm25_tokenize),
        ("indexed text", test_indexed_text),
        ("generation helpers", test_generation_helpers),
        ("serving", test_serving),
        ("retrieval metrics", test_metrics),
    ]:
        print(f"\n{name}")
        results.append(fn())
    print(f"\n{'ALL PASS' if all(results) else 'FAILURES ABOVE'}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
