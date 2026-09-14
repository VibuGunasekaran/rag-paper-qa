"""Day 4: score every retrieval configuration against the gold set.

    python eval/run_eval.py                                  # 3 sizes x 3 methods x rerank on/off
    python eval/run_eval.py --sizes 512 --methods hybrid --rerank on
    python eval/run_eval.py --report-only                    # re-render results.md from results.json

A retrieved chunk is relevant when it contains one of the row's gold quotes
(whitespace- and case-normalised, exactly as `build_gold.py validate` checks).
Scoring on quotes rather than chunk ids is what makes 256, 512 and 1024
comparable: the ids differ per size, the sentences do not. Unanswerable rows
have no evidence to retrieve and are left out; abstention is a Day 5 metric.

Per configuration, over answerable rows:
    R@k      share of rows with at least one relevant chunk in the top k
    MRR@10   mean reciprocal rank of the first relevant chunk (0 if none in top 10)
    nDCG@10  binary gains; the ideal ranking puts every relevant chunk that exists
             at that chunk size first, capped at 10
    Cov@5    comparative rows only: share whose top 5 holds relevant chunks from
             every paper the gold quotes come from
    p50/p95  end-to-end retrieve() latency, measured after a warm-up call

With a few dozen questions most differences between configurations are noise, so
every headline number carries a 95% bootstrap interval over questions, and the
comparisons the README makes are tested as paired bootstraps (both configurations
resampled on the same questions).

Writes eval/results.md (tables) and eval/results.json (per-query detail).
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from build_gold import TYPES, load_gold, normalise   # noqa: E402
from src.config import load_config, resolve          # noqa: E402

DEPTH = 10
KS = (1, 3, 5, 10)
SIZES = (256, 512, 1024)
METHODS = ("dense", "bm25", "hybrid")
BOOTSTRAP_RESAMPLES = 5000

# (A, B) pairs reported as paired differences A - B. Each key is
# (chunk_size, method, rerank). Pairs whose configs were not run are skipped.
COMPARISONS = [
    ((512, "bm25", False), (512, "dense", False)),
    ((512, "hybrid", False), (512, "bm25", False)),
    ((512, "hybrid", True), (512, "hybrid", False)),
    ((512, "hybrid", True), (512, "bm25", False)),
    ((256, "hybrid", True), (512, "hybrid", True)),
    ((1024, "hybrid", True), (512, "hybrid", True)),
]


# --------------------------------------------------------------------------
# Metrics (pure functions; covered by tests/test_pipeline.py)
# --------------------------------------------------------------------------

def relevant_flags(texts: list[str], quotes: list[str]) -> list[bool]:
    """For each text, whether it contains any quote after normalisation."""
    nq = [normalise(q) for q in quotes if q.strip()]
    return [any(q in normalise(t) for q in nq) for t in texts]


def recall_at(flags: list[bool], k: int) -> float:
    return float(any(flags[:k]))


def mrr(flags: list[bool], k: int = DEPTH) -> float:
    for rank, is_rel in enumerate(flags[:k], start=1):
        if is_rel:
            return 1.0 / rank
    return 0.0


def ndcg(flags: list[bool], n_relevant: int, k: int = DEPTH) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, is_rel in enumerate(flags[:k], start=1) if is_rel)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(n_relevant, k) + 1))
    return dcg / ideal if ideal else 0.0


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: no interpolation, always an observed value."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(p / 100 * len(ordered)) - 1))
    return ordered[idx]


def bootstrap_ci(values: list[float], n_resamples: int = BOOTSTRAP_RESAMPLES,
                 seed: int = 0) -> tuple[float, float]:
    """95% percentile-bootstrap interval for the mean. Seeded, so reports are reproducible.

    For a paired comparison pass the per-question differences A - B: resampling
    the differences resamples both configurations on the same questions.
    """
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(n_resamples))
    return means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples) - 1]


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def quotes_of(row: dict) -> list[str]:
    return [gq["quote"] for gq in row.get("gold_quotes", []) if gq.get("quote", "").strip()]


def gold_papers(row: dict) -> set[str]:
    return {gq["chunk_id"].split("::")[0] for gq in row.get("gold_quotes", [])}


def evaluate_size(cfg, size: int, rows: list[dict], methods, reranks) -> tuple[list[dict], list[dict]]:
    from src.retrieve import Retriever

    r = Retriever(chunk_size=size, config=cfg)
    norm_text = {c["chunk_id"]: normalise(c["text"]) for c in r.chunks}

    # Every chunk at this size that contains a gold quote. This is the ideal set
    # for nDCG and the membership test for every retrieved hit.
    relevant: dict[str, set[str]] = {}
    for row in rows:
        nq = [normalise(q) for q in quotes_of(row)]
        relevant[row["id"]] = {cid for cid, t in norm_text.items() if any(q in t for q in nq)}
    missing = [row["id"] for row in rows if not relevant[row["id"]]]
    if missing:
        print(f"  ! [{size}] no chunk contains the gold quotes for: {', '.join(missing)}", file=sys.stderr)

    t0 = time.perf_counter()
    r.warmup()
    print(f"[{size}] {len(r.chunks)} chunks, warm-up {time.perf_counter() - t0:.1f}s")

    configs, queries = [], []
    for method in methods:
        for rerank in reranks:
            per_row, latencies = [], []
            for row in rows:
                out = r.retrieve(row["question"], method=method, rerank=rerank, top_k=DEPTH)
                latencies.append(out.timings_ms["total_ms"])
                ids = out.chunk_ids
                rel = relevant[row["id"]]
                flags = [cid in rel for cid in ids]
                hit_papers = {cid.split("::")[0] for cid in ids[:5] if cid in rel}
                rec = {
                    "chunk_size": size, "method": method, "rerank": rerank,
                    "id": row["id"], "type": row["type"],
                    "relevant_ranks": [i for i, f in enumerate(flags, start=1) if f],
                    "n_relevant_at_size": len(rel),
                    "latency_ms": out.timings_ms["total_ms"],
                    **{f"R@{k}": recall_at(flags, k) for k in KS},
                    "MRR@10": mrr(flags),
                    "nDCG@10": ndcg(flags, len(rel)),
                }
                if row["type"] == "comparative":
                    rec["Cov@5"] = float(gold_papers(row) <= hit_papers)
                per_row.append(rec)
                queries.append(rec)

            summary = {"chunk_size": size, "method": method, "rerank": rerank, "n": len(per_row)}
            for key in [f"R@{k}" for k in KS] + ["MRR@10", "nDCG@10"]:
                summary[key] = statistics.fmean(q[key] for q in per_row)
            comp = [q["Cov@5"] for q in per_row if "Cov@5" in q]
            summary["Cov@5"] = statistics.fmean(comp) if comp else None
            summary["p50_ms"] = percentile(latencies, 50)
            summary["p95_ms"] = percentile(latencies, 95)
            summary["by_type"] = {
                t: {
                    "n": sum(q["type"] == t for q in per_row),
                    "R@5": statistics.fmean(q["R@5"] for q in per_row if q["type"] == t),
                    "MRR@10": statistics.fmean(q["MRR@10"] for q in per_row if q["type"] == t),
                }
                for t in TYPES if any(q["type"] == t for q in per_row)
            }
            configs.append(summary)
            print(f"  {method:6} rerank={str(rerank):5}  R@5 {summary['R@5']:.3f}  "
                  f"MRR@10 {summary['MRR@10']:.3f}  nDCG@10 {summary['nDCG@10']:.3f}  "
                  f"p95 {summary['p95_ms']:.0f} ms")
    return configs, queries


# --------------------------------------------------------------------------
# Intervals and paired comparisons
# --------------------------------------------------------------------------

def key_of(c: dict) -> tuple[int, str, bool]:
    return (c["chunk_size"], c["method"], c["rerank"])


def add_intervals(configs: list[dict], queries: list[dict]) -> None:
    """Attach 95% bootstrap intervals for R@5 and MRR@10 to each config summary."""
    by_key: dict[tuple, list[dict]] = {}
    for q in queries:
        by_key.setdefault(key_of(q), []).append(q)
    for c in configs:
        rows = by_key.get(key_of(c), [])
        for metric in ("R@5", "MRR@10"):
            c[f"{metric}_ci"] = list(bootstrap_ci([q[metric] for q in rows]))


def paired_comparisons(queries: list[dict]) -> list[dict]:
    by_key: dict[tuple, dict[str, dict]] = {}
    for q in queries:
        by_key.setdefault(key_of(q), {})[q["id"]] = q
    out = []
    for a, b in COMPARISONS:
        if a not in by_key or b not in by_key:
            continue
        ids = sorted(set(by_key[a]) & set(by_key[b]))
        for metric in ("R@5", "MRR@10"):
            diffs = [by_key[a][i][metric] - by_key[b][i][metric] for i in ids]
            lo, hi = bootstrap_ci(diffs)
            out.append({"a": list(a), "b": list(b), "metric": metric, "n": len(ids),
                        "diff": statistics.fmean(diffs), "ci": [lo, hi],
                        "excludes_zero": lo > 0 or hi < 0})
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def label(key) -> tuple[str, str, str]:
    size, method, rerank = key
    return str(size), {"dense": "Dense", "bm25": "BM25", "hybrid": "Hybrid"}[method], \
        "Yes" if rerank else "No"


def fmt_ci(ci) -> str:
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def render_markdown(cfg, configs: list[dict], comparisons: list[dict], n_rows: int, n_skipped: int) -> str:
    types = [t for t in TYPES if any(t in c["by_type"] for c in configs)]
    lines = [
        "# Retrieval results",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `eval/run_eval.py` on {platform.platform()} "
        f"(embedding device: {cfg['embedding']['device']}).",
        "",
        f"- Questions scored: {n_rows} answerable ({n_skipped} unanswerable excluded; abstention is scored with generation).",
        f"- Embedding: `{cfg['embedding']['model']}`; reranker: `{cfg['retrieval']['reranker_model']}`.",
        f"- Candidate depth {cfg['retrieval']['candidate_k']} per retriever, RRF k={cfg['retrieval']['rrf_k']}, "
        f"top {DEPTH} scored.",
        "- A retrieved chunk is relevant when it contains a gold quote. Latency is one run per query after warm-up.",
        f"- One question moves R@k by {100 / n_rows:.1f} points. Intervals are 95% percentile bootstraps over "
        f"questions ({BOOTSTRAP_RESAMPLES} resamples, seed 0).",
        "",
        "## All configurations",
        "",
        "| Chunk | Retriever | Rerank | R@1 | R@3 | R@5 | R@5 95% CI | R@10 | MRR@10 | MRR@10 95% CI "
        "| nDCG@10 | Cov@5 | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in configs:
        cov = f"{c['Cov@5']:.2f}" if c["Cov@5"] is not None else "-"
        lines.append(
            "| " + " | ".join(label(key_of(c))) + " | "
            f"{c['R@1']:.2f} | {c['R@3']:.2f} | {c['R@5']:.2f} | {fmt_ci(c['R@5_ci'])} | {c['R@10']:.2f} | "
            f"{c['MRR@10']:.3f} | {fmt_ci(c['MRR@10_ci'])} | {c['nDCG@10']:.3f} | {cov} | "
            f"{c['p50_ms']:.0f} | {c['p95_ms']:.0f} |"
        )
    if comparisons:
        lines += [
            "",
            "## Paired comparisons",
            "",
            "Difference A - B on the same questions. Only intervals that exclude zero support a claim that A and B differ.",
            "",
            "| A | B | Metric | A - B | 95% CI | Excludes 0 |",
            "|---|---|---|---|---|---|",
        ]
        for p in comparisons:
            lines.append(f"| {' / '.join(label(p['a']))} | {' / '.join(label(p['b']))} | {p['metric']} | "
                         f"{p['diff']:+.3f} | [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}] | "
                         f"{'yes' if p['excludes_zero'] else 'no'} |")
    lines += [
        "",
        "## Recall@5 by question type",
        "",
        "| Chunk | Retriever | Rerank | " + " | ".join(f"{t} (n={configs[0]['by_type'][t]['n']})" for t in types) + " |",
        "|---|---|---|" + "---|" * len(types),
    ]
    for c in configs:
        lines.append("| " + " | ".join(label(key_of(c))) + " | "
                     + " | ".join(f"{c['by_type'][t]['R@5']:.2f}" for t in types) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Score retrieval configurations against the gold set.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--rerank", choices=["both", "on", "off"], default="both")
    ap.add_argument("--report-only", action="store_true",
                    help="Skip retrieval; recompute intervals and re-render from results.json.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    md_path = resolve(cfg["eval"]["results_path"])
    json_path = md_path.with_suffix(".json")

    if args.report_only:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        configs, queries = data["configs"], data["queries"]
        n_rows = len({q["id"] for q in queries})
        n_skipped = data.get("n_skipped", 0)
    else:
        gold = load_gold(resolve(cfg["eval"]["gold_path"]))
        if not gold:
            print("No gold set. Run scripts/build_gold.py first.", file=sys.stderr)
            return 1
        rows = [g for g in gold if g["type"] != "unanswerable" and quotes_of(g)]
        n_rows, n_skipped = len(rows), len(gold) - len(rows)
        reranks = {"both": [False, True], "on": [True], "off": [False]}[args.rerank]
        configs, queries = [], []
        for size in args.sizes:
            c, q = evaluate_size(cfg, size, rows, args.methods, reranks)
            configs += c
            queries += q

    add_intervals(configs, queries)
    comparisons = paired_comparisons(queries)
    md_path.write_text(render_markdown(cfg, configs, comparisons, n_rows, n_skipped), encoding="utf-8")
    json_path.write_text(json.dumps({"n_skipped": n_skipped, "configs": configs, "comparisons": comparisons,
                                     "queries": queries}, indent=1), encoding="utf-8")
    for p in comparisons:
        print(f"  {'/'.join(map(str, p['a']))} - {'/'.join(map(str, p['b']))} {p['metric']:6} "
              f"{p['diff']:+.3f} [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]"
              f"{'  *' if p['excludes_zero'] else ''}")
    print(f"\nWrote {md_path} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
