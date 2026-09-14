"""Logic tests that need no models, no network and no index.

These cover the parts most likely to be quietly wrong: heading detection,
section splitting, the chunk window arithmetic, and RRF. Run them before
trusting a number that came out of the eval.

    python tests/test_pipeline.py
"""
from __future__ import annotations

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
from src.index import bm25_tokenize  # noqa: E402
from src.retrieve import reciprocal_rank_fusion  # noqa: E402


class WordCounter:
    """Stand-in for the bge tokenizer: one token per word."""

    def encode(self, text): return text.split()
    def decode(self, tokens): return " ".join(tokens)


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
    return ok


def test_split_sections():
    text = "\n".join([
        "Introduction",
        "Vision language models are large.",
        "",
        "2 Method",
        "We prune visual tokens using attention.",
        "",
        "References",
        "[1] Someone et al.",
    ])
    sections = split_sections(text, {"references"})
    titles = [t for t, _ in sections]
    bodies = " ".join(b for _, b in sections)
    ok = True
    ok &= check("intro captured", any("Introduction" in t for t in titles))
    ok &= check("method captured", any("Method" in t for t in titles))
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


def main() -> int:
    results = []
    for name, fn in [
        ("heading detection", test_headings),
        ("section splitting", test_split_sections),
        ("chunking", test_chunking),
        ("reciprocal rank fusion", test_rrf),
        ("bm25 tokenisation", test_bm25_tokenize),
    ]:
        print(f"\n{name}")
        results.append(fn())
    print(f"\n{'ALL PASS' if all(results) else 'FAILURES ABOVE'}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
