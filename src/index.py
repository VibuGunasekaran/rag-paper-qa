"""Day 1: build the dense (FAISS) and lexical (BM25) indexes over the same chunks.

Both indexes are built from one chunks_{size}.jsonl and written to a
self-contained directory per chunk size:

    data/index/512/faiss.index    IndexFlatIP over L2-normalised bge vectors
    data/index/512/bm25.pkl       rank_bm25 BM25Okapi
    data/index/512/chunks.jsonl   the exact chunks this index was built from
    data/index/512/meta.json      model, dim, count, build time

Copying the chunks in means an index directory is never ambiguous about what it
contains -- which matters on Day 4 when three chunk sizes exist side by side.

Usage:
    python -m src.index                    # build at the config chunk size
    python -m src.index --chunk-size 256
    python -m src.index --all              # 256, 512 and 1024
    python -m src.index --all --title-prefix   # same, with paper titles prepended
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

from .config import load_config, resolve

# Small, deliberate stop list. BM25 handles common words via IDF, but stripping
# these still measurably sharpens short academic queries.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "how", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "what", "when", "which", "who", "why", "with", "does", "do",
}
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-_.]*")


def bm25_tokenize(text: str) -> list[str]:
    """Shared by index build and query time. Do not let these two drift.

    Trailing punctuation is stripped: without it "layer-2." is indexed as a
    different term from the "layer-2" a user types, and exact-terminology
    queries -- the ones BM25 is supposed to win -- quietly miss.
    """
    out = []
    for tok in TOKEN_RE.findall(text.lower()):
        tok = tok.strip("._-")
        if tok and tok not in STOPWORDS:
            out.append(tok)
    return out


def index_dir(cfg, chunk_size: int) -> Path:
    """data/index/512, or data/index/512-title when chunk titles are indexed too."""
    suffix = "-title" if cfg["index"].get("title_prefix") else ""
    return resolve(cfg["index"]["dir"]) / f"{chunk_size}{suffix}"


def indexed_text(chunk: dict, title_prefix: bool) -> str:
    """The text a retriever sees for a chunk.

    With title_prefix the paper title leads the chunk. Body chunks often never
    name the method they describe ("ST3", "LLaVA"), so a question that names it
    cannot match them. The stored chunk text, and so gold-quote matching, is
    the same either way.
    """
    title = chunk.get("title", "").strip()
    return f"{title}\n\n{chunk['text']}" if title_prefix and title else chunk["text"]


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.ingest --chunk-size N` first."
        )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_index(cfg, chunk_size: int) -> Path:
    import faiss
    from sentence_transformers import SentenceTransformer

    chunk_path = resolve(cfg["corpus"]["chunk_dir"]) / f"chunks_{chunk_size}.jsonl"
    chunks = load_chunks(chunk_path)
    print(f"[{chunk_size}] {len(chunks)} chunks from {chunk_path.name}")

    title_prefix = bool(cfg["index"].get("title_prefix"))
    texts = [indexed_text(c, title_prefix) for c in chunks]
    out_dir = index_dir(cfg, chunk_size)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- dense ---
    model_name = cfg["embedding"]["model"]
    device = cfg["embedding"]["device"]
    print(f"[{chunk_size}] embedding with {model_name} on {device}")
    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {device} unavailable ({exc}); falling back to cpu", file=sys.stderr)
        model = SentenceTransformer(model_name, device="cpu")

    t0 = time.perf_counter()
    # No query prefix here: bge wants it on the query side only.
    vectors = model.encode(
        texts,
        batch_size=cfg["embedding"]["batch_size"],
        normalize_embeddings=True,     # so inner product == cosine similarity
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    embed_s = time.perf_counter() - t0

    dim = vectors.shape[1]
    # Exact search. At ~5k chunks an ANN index buys nothing and costs recall.
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(vectors)
    faiss.write_index(faiss_index, str(out_dir / "faiss.index"))

    # --- lexical ---
    from rank_bm25 import BM25Okapi

    print(f"[{chunk_size}] building BM25")
    bm25 = BM25Okapi([bm25_tokenize(t) for t in texts])
    with open(out_dir / "bm25.pkl", "wb") as fh:
        pickle.dump(bm25, fh)

    # --- chunk store + provenance ---
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")

    meta = {
        "chunk_size": chunk_size,
        "title_prefix": title_prefix,
        "n_chunks": len(chunks),
        "n_papers": len({c["paper_id"] for c in chunks}),
        "embedding_model": model_name,
        "dim": dim,
        "embed_seconds": round(embed_s, 1),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mean_tokens": round(float(np.mean([c["n_tokens"] for c in chunks])), 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[{chunk_size}] done in {embed_s:.1f}s -> {out_dir}")
    print(json.dumps(meta, indent=2))
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Build FAISS and BM25 indexes.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="Build 256, 512 and 1024.")
    ap.add_argument("--title-prefix", action="store_true",
                    help="Index each chunk with its paper title prepended (data/index/<size>-title).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.title_prefix:
        cfg["index"]["title_prefix"] = True
    sizes = [256, 512, 1024] if args.all else [args.chunk_size or cfg["chunking"]["chunk_size"]]
    for size in sizes:
        build_index(cfg, size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
