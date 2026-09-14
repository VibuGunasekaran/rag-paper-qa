"""Day 2: three retrievers behind one interface, plus an optional reranker.

    r = Retriever(chunk_size=512)
    out = r.retrieve("how does FastV decide which visual tokens to drop?",
                     method="hybrid", rerank=True)

`method` is one of dense | bm25 | hybrid. Everything the Day 4 ablation grid
varies -- chunk size, method, candidate depth, RRF k, reranking -- is an
argument here or a key in config.yaml, never a literal in the code.

Timings are recorded per stage on every call, because Day 4 reports p50/p95
latency alongside recall and the fastest configuration is not always the best
one. Models are loaded lazily: a BM25-only sweep never imports torch.

Usage:
    python -m src.retrieve "your question" --method hybrid --rerank
    python -m src.retrieve "your question" --method bm25 --top-k 10
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import load_config, resolve
from .index import bm25_tokenize

METHODS = ("dense", "bm25", "hybrid")


@dataclass
class Hit:
    chunk_id: str
    score: float
    rank: int
    paper_id: str
    title: str
    section: str
    text: str

    def preview(self, n: int = 220) -> str:
        body = " ".join(self.text.split())
        return body[:n] + ("..." if len(body) > n else "")


@dataclass
class RetrievalResult:
    query: str
    method: str
    reranked: bool
    hits: list[Hit] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)

    @property
    def chunk_ids(self) -> list[str]:
        return [h.chunk_id for h in self.hits]

    def to_json(self) -> dict:
        return {
            "query": self.query,
            "method": self.method,
            "reranked": self.reranked,
            "timings_ms": self.timings_ms,
            "hits": [
                {"rank": h.rank, "chunk_id": h.chunk_id, "score": round(h.score, 4),
                 "paper_id": h.paper_id, "section": h.section}
                for h in self.hits
            ],
        }


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    """RRF: score(d) = sum over lists of 1 / (k + rank(d)), rank starting at 1.

    Rank-based, so it needs no score normalisation between a cosine similarity
    and a BM25 score -- which is exactly why it beats naive score blending.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class Retriever:
    def __init__(self, chunk_size: int | None = None, config=None):
        self.cfg = config or load_config()
        self.chunk_size = chunk_size or self.cfg["chunking"]["chunk_size"]
        self.dir = resolve(self.cfg["index"]["dir"]) / str(self.chunk_size)
        if not self.dir.exists():
            raise FileNotFoundError(
                f"No index at {self.dir}. Run "
                f"`python -m src.index --chunk-size {self.chunk_size}` first."
            )
        self.chunks = [
            json.loads(line)
            for line in (self.dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self.ids = [c["chunk_id"] for c in self.chunks]
        self._faiss = None
        self._encoder = None
        self._bm25 = None
        self._reranker = None

    # -- lazy resources ----------------------------------------------------
    @property
    def faiss_index(self):
        if self._faiss is None:
            import faiss
            self._faiss = faiss.read_index(str(self.dir / "faiss.index"))
        return self._faiss

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            try:
                self._encoder = SentenceTransformer(
                    self.cfg["embedding"]["model"], device=self.cfg["embedding"]["device"]
                )
            except Exception:  # noqa: BLE001
                self._encoder = SentenceTransformer(self.cfg["embedding"]["model"], device="cpu")
        return self._encoder

    @property
    def bm25(self):
        if self._bm25 is None:
            with open(self.dir / "bm25.pkl", "rb") as fh:
                self._bm25 = pickle.load(fh)
        return self._bm25

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.cfg["retrieval"]["reranker_model"])
        return self._reranker

    # -- individual retrievers --------------------------------------------
    def dense_search(self, query: str, k: int) -> list[str]:
        prefixed = self.cfg["embedding"]["query_prefix"] + query
        vec = self.encoder.encode(
            [prefixed], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        _, idx = self.faiss_index.search(vec, min(k, len(self.ids)))
        return [self.ids[i] for i in idx[0] if i >= 0]

    def bm25_search(self, query: str, k: int) -> list[str]:
        scores = self.bm25.get_scores(bm25_tokenize(query))
        top = np.argsort(scores)[::-1][: min(k, len(self.ids))]
        return [self.ids[i] for i in top if scores[i] > 0]

    def warmup(self, method: str = "hybrid", rerank: bool = True) -> None:
        """Load models with one throwaway query so timed calls start warm.

        The first dense call loads the encoder and the first reranked call loads
        the cross-encoder, 5-7 s each on an M1. Without this, whichever
        configuration an eval runs first carries that cost in its p95 latency.
        """
        self.retrieve("warm up", method=method, rerank=rerank)

    # -- one interface -----------------------------------------------------
    def retrieve(
        self,
        query: str,
        method: str | None = None,
        rerank: bool | None = None,
        top_k: int | None = None,
        candidate_k: int | None = None,
        rrf_k: int | None = None,
    ) -> RetrievalResult:
        rcfg = self.cfg["retrieval"]
        method = (method or rcfg["method"]).lower()
        rerank = rcfg["rerank"] if rerank is None else rerank
        top_k = top_k or rcfg["top_k"]
        candidate_k = candidate_k or rcfg["candidate_k"]
        rrf_k = rrf_k or rcfg["rrf_k"]
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")

        timings: dict[str, float] = {}
        t_start = time.perf_counter()
        # Retrieve candidate_k deep even when only top_k is returned, so the
        # reranker has something to work with and the two paths stay comparable.
        depth = max(candidate_k, top_k)

        scored: list[tuple[str, float]]
        if method == "dense":
            t = time.perf_counter()
            ranked = self.dense_search(query, depth)
            timings["dense_ms"] = (time.perf_counter() - t) * 1000
            scored = [(cid, 1.0 / (rrf_k + r)) for r, cid in enumerate(ranked, 1)]
        elif method == "bm25":
            t = time.perf_counter()
            ranked = self.bm25_search(query, depth)
            timings["bm25_ms"] = (time.perf_counter() - t) * 1000
            scored = [(cid, 1.0 / (rrf_k + r)) for r, cid in enumerate(ranked, 1)]
        else:
            t = time.perf_counter()
            dense_ranked = self.dense_search(query, depth)
            timings["dense_ms"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            bm25_ranked = self.bm25_search(query, depth)
            timings["bm25_ms"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            scored = reciprocal_rank_fusion([dense_ranked, bm25_ranked], k=rrf_k)
            timings["fuse_ms"] = (time.perf_counter() - t) * 1000

        candidates = scored[:depth]

        if rerank and candidates:
            t = time.perf_counter()
            pairs = [(query, self.by_id[cid]["text"]) for cid, _ in candidates]
            ce_scores = self.reranker.predict(pairs)
            candidates = sorted(
                zip([c for c, _ in candidates], [float(s) for s in ce_scores]),
                key=lambda kv: kv[1],
                reverse=True,
            )
            timings["rerank_ms"] = (time.perf_counter() - t) * 1000

        timings["total_ms"] = (time.perf_counter() - t_start) * 1000
        timings = {k: round(v, 2) for k, v in timings.items()}

        hits = [
            Hit(
                chunk_id=cid,
                score=score,
                rank=i,
                paper_id=self.by_id[cid]["paper_id"],
                title=self.by_id[cid].get("title", ""),
                section=self.by_id[cid].get("section", ""),
                text=self.by_id[cid]["text"],
            )
            for i, (cid, score) in enumerate(candidates[:top_k], start=1)
        ]
        return RetrievalResult(
            query=query, method=method, reranked=bool(rerank),
            hits=hits, timings_ms=timings,
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Query the index from the command line.")
    ap.add_argument("query")
    ap.add_argument("--config", default=None)
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--method", default=None, choices=list(METHODS))
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--candidate-k", type=int, default=None)
    rr = ap.add_mutually_exclusive_group()
    rr.add_argument("--rerank", dest="rerank", action="store_true")
    rr.add_argument("--no-rerank", dest="rerank", action="store_false")
    ap.set_defaults(rerank=None)
    ap.add_argument("--json", action="store_true", help="Machine-readable output.")
    args = ap.parse_args()

    r = Retriever(chunk_size=args.chunk_size, config=load_config(args.config))
    out = r.retrieve(
        args.query, method=args.method, rerank=args.rerank,
        top_k=args.top_k, candidate_k=args.candidate_k,
    )

    if args.json:
        print(json.dumps(out.to_json(), indent=2))
        return 0

    print(f"\nQ: {out.query}")
    print(f"   method={out.method}  rerank={out.reranked}  chunks={r.chunk_size}  {out.timings_ms}\n")
    for h in out.hits:
        print(f"[{h.rank}] {h.chunk_id}   score={h.score:.4f}")
        print(f"    {h.title[:70]}")
        print(f"    section: {h.section}")
        print(f"    {h.preview()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
