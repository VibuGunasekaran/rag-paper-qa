"""Day 5: generate cited answers for the gold questions and score them.

    python eval/run_generation.py --dry-run          # retrieval + cost estimate, no API calls
    python eval/run_generation.py --limit 6          # smoke test across question types
    python eval/run_generation.py                    # all 150 questions
    python eval/run_generation.py --report-only      # re-render from results_generation.json

One retrieval configuration (`generation.retrieval` in config.yaml) gives the
generator its top-k chunks. Every API response is cached in
`generation.cache_path`, keyed by a hash of the exact request, so re-running or
re-scoring never pays twice; editing a prompt changes the key.

Scored per question:
  retrieval   evidence_retrieved: some top-k chunk contains a gold quote
  abstention  abstained on unanswerable questions (wanted) and on answerable ones
  citations   cited numbers are valid and match the inline [S#] markers;
              whether the answer cites a chunk that holds the gold evidence
  judge       faithfulness (full/partial/none) against the sources the model saw;
              correctness (correct/partial/incorrect/abstained) against the gold answer
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO / "scripts"))

from run_eval import bootstrap_ci, quotes_of                             # noqa: E402
from build_gold import TYPES, load_gold, normalise                       # noqa: E402
from judge import JUDGE_SYSTEM, build_judge_message, judge_answer        # noqa: E402
from src.config import load_config, resolve                              # noqa: E402
from src.generate import SYSTEM_PROMPT, Generator, build_user_message, inline_citations  # noqa: E402

# USD per million tokens (input, output). Fallback-served requests bill at the
# fallback model's rates, so treat totals as estimates.
PRICES_PER_MTOK = {"claude-opus-5": (5.0, 25.0), "claude-opus-4-8": (5.0, 25.0)}
EST_OUTPUT_TOKENS = 1500   # per request, thinking included; only used by --dry-run


class ResponseCache:
    """Append-only JSONL cache of API results keyed by request hash."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec["value"]

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        return self.data.get(key)

    def put(self, key: str, value: dict) -> None:
        with self.lock:
            self.data[key] = value
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


def select_rows(gold: list[dict], ids: list[str] | None, limit: int | None) -> list[dict]:
    rows = [g for g in gold if not ids or g["id"] in ids]
    if not limit:
        return rows
    # Round-robin across question types so a small smoke test covers every type.
    by_type = [[r for r in rows if r["type"] == t] for t in TYPES]
    picked = []
    while len(picked) < limit and any(by_type):
        for group in by_type:
            if group and len(picked) < limit:
                picked.append(group.pop(0))
    return picked


def retrieve_all(cfg, rows: list[dict]) -> dict:
    from src.retrieve import Retriever

    rc = cfg["generation"]["retrieval"]
    cfg["index"]["title_prefix"] = bool(rc.get("title_prefix"))
    retriever = Retriever(chunk_size=rc["chunk_size"], config=cfg)
    top_k = cfg["generation"]["top_k"]
    return {row["id"]: retriever.retrieve(row["question"], method=rc["method"], rerank=rc["rerank"],
                                          top_k=top_k).hits
            for row in rows}


def process_row(row: dict, hits, gen: Generator, cache: ResponseCache, judge_model: str,
                do_judge: bool) -> dict:
    nq = [normalise(q) for q in quotes_of(row)]
    gold_numbers = [i for i, h in enumerate(hits, start=1) if nq and any(q in normalise(h.text) for q in nq)]
    rec = {
        "id": row["id"], "type": row["type"], "question": row["question"], "reference": row["answer"],
        "source_chunk_ids": [h.chunk_id for h in hits],
        "gold_source_numbers": gold_numbers,
        "evidence_retrieved": bool(gold_numbers),
    }

    user = build_user_message(row["question"], hits)
    gkey = ResponseCache.key("generate", gen.model, SYSTEM_PROMPT, user)
    ans = cache.get(gkey)
    if ans is None:
        ans = asdict(gen.answer(row["question"], hits))
        cache.put(gkey, ans)
    rec["generation"] = ans

    answered = not (ans["refused"] or ans["parse_failed"] or ans["abstained"])
    if answered:
        cited = ans["cited_sources"]
        rec["citations_valid"] = (bool(cited) and all(1 <= c <= len(hits) for c in cited)
                                  and set(cited) == set(inline_citations(ans["answer"])))
        rec["cites_gold"] = bool(set(cited) & set(gold_numbers)) if gold_numbers else None
    else:
        rec["citations_valid"] = rec["cites_gold"] = None

    if do_judge and not (ans["refused"] or ans["parse_failed"]):
        msg = build_judge_message(row["question"], hits, ans["answer"], ans["abstained"], row["answer"],
                                  row["type"] == "unanswerable")
        jkey = ResponseCache.key("judge", judge_model, JUDGE_SYSTEM, msg)
        verdict = cache.get(jkey)
        if verdict is None:
            verdict = asdict(judge_answer(gen.client, judge_model, msg))
            cache.put(jkey, verdict)
        rec["judge"] = verdict
    return rec


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------

def rate(values) -> dict | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    lo, hi = bootstrap_ci(vals)
    return {"n": len(vals), "rate": statistics.fmean(vals), "ci": [lo, hi]}


def summarise(records: list[dict]) -> dict:
    def gen(r):
        return r["generation"]

    def judged(r):
        return r.get("judge") and not (r["judge"]["refused"] or r["judge"]["parse_failed"])

    answerable = [r for r in records if r["type"] != "unanswerable"]
    unanswerable = [r for r in records if r["type"] == "unanswerable"]
    usable = [r for r in records if not (gen(r)["refused"] or gen(r)["parse_failed"])]
    answered = [r for r in usable if not gen(r)["abstained"]]

    def scored(r):
        # Unanswerable rows are scored on the abstention itself: the judge labels a
        # correct abstention "abstained", not "correct", so its verdict can't be used.
        return r in usable if r["type"] == "unanswerable" else judged(r)

    def correct(r):
        if r["type"] == "unanswerable":
            return gen(r)["abstained"]
        return r["judge"]["correctness"] == "correct"

    summary = {
        "n_questions": len(records),
        "refusals": sum(gen(r)["refused"] for r in records),
        "parse_failures": sum(gen(r)["parse_failed"] for r in records),
        "fallback_served": sum(gen(r)["model"] != gen(r)["requested_model"] for r in records),
        "evidence_retrieved": rate(r["evidence_retrieved"] for r in answerable),
        "abstain_on_unanswerable": rate(gen(r)["abstained"] for r in unanswerable if r in usable),
        "abstain_answerable_with_evidence": rate(gen(r)["abstained"] for r in answerable
                                                 if r in usable and r["evidence_retrieved"]),
        "abstain_answerable_without_evidence": rate(gen(r)["abstained"] for r in answerable
                                                    if r in usable and not r["evidence_retrieved"]),
        "citations_valid": rate(r["citations_valid"] for r in answered),
        "cites_gold_evidence": rate(r["cites_gold"] for r in answered if r["type"] != "unanswerable"),
        "faithful_full": rate(r["judge"]["faithfulness"] == "full" for r in answered if judged(r)),
        "correct": rate(r["judge"]["correctness"] == "correct" for r in answerable if judged(r)),
        "correct_or_partial": rate(r["judge"]["correctness"] in ("correct", "partial")
                                   for r in answerable if judged(r)),
        "correct_with_evidence": rate(r["judge"]["correctness"] == "correct"
                                      for r in answerable if judged(r) and r["evidence_retrieved"]),
        "correct_unanswerable": rate(correct(r) for r in unanswerable if scored(r)),
    }
    summary["by_type"] = {
        t: {
            "n": sum(r["type"] == t for r in records),
            "correct": rate(correct(r) for r in records if r["type"] == t and scored(r)),
            "faithful_full": rate(r["judge"]["faithfulness"] == "full" for r in answered
                                  if r["type"] == t and judged(r)),
            "abstained": rate(gen(r)["abstained"] for r in usable if r["type"] == t),
        }
        for t in TYPES if any(r["type"] == t for r in records)
    }
    tokens = {"generate": [0, 0], "judge": [0, 0]}
    dollars = 0.0
    for r in records:
        for stage, res in (("generate", r["generation"]), ("judge", r.get("judge"))):
            if not res:
                continue
            u = res.get("usage") or {}
            tin, tout = u.get("input_tokens", 0) or 0, u.get("output_tokens", 0) or 0
            tokens[stage][0] += tin
            tokens[stage][1] += tout
            pin, pout = PRICES_PER_MTOK.get(res["model"], PRICES_PER_MTOK["claude-opus-5"])
            dollars += (tin * pin + tout * pout) / 1e6
    summary["tokens"] = tokens
    summary["estimated_cost_usd"] = round(dollars, 2)
    return summary


def fmt(m: dict | None) -> str:
    return "-" if m is None else f"{m['rate']:.2f} [{m['ci'][0]:.2f}, {m['ci'][1]:.2f}] (n={m['n']})"


def render_markdown(cfg, summary: dict) -> str:
    g = cfg["generation"]
    rc = g["retrieval"]
    s = summary
    lines = [
        "# Generation results",
        "",
        f"- Generator and judge: `{g['model']}` / `{g['judge_model']}`, server-side refusal fallbacks on.",
        f"- Retrieval feeding generation: {rc['chunk_size']} tokens, {rc['method']}, rerank={rc['rerank']}, "
        f"title_prefix={bool(rc.get('title_prefix'))}, top {g['top_k']} chunks.",
        f"- Questions: {s['n_questions']}. Refusals: {s['refusals']}. Parse failures: {s['parse_failures']}. "
        f"Served by a fallback model: {s['fallback_served']}.",
        f"- Tokens (input/output): generation {s['tokens']['generate'][0]:,}/{s['tokens']['generate'][1]:,}, "
        f"judge {s['tokens']['judge'][0]:,}/{s['tokens']['judge'][1]:,}; estimated cost ${s['estimated_cost_usd']} "
        "(includes cached responses from earlier runs).",
        "- Rates carry 95% bootstrap intervals over questions.",
        "",
        "| Metric | Rate [95% CI] (n) | Wanted |",
        "|---|---|---|",
        f"| Gold evidence in the top {g['top_k']} (answerable) | {fmt(s['evidence_retrieved'])} | high |",
        f"| Correct (judge, answerable) | {fmt(s['correct'])} | high |",
        f"| Correct or partial (judge, answerable) | {fmt(s['correct_or_partial'])} | high |",
        f"| Correct when evidence was retrieved | {fmt(s['correct_with_evidence'])} | high |",
        f"| Faithful: every claim supported (judge, answered) | {fmt(s['faithful_full'])} | high |",
        f"| Abstained on unanswerable questions | {fmt(s['abstain_on_unanswerable'])} | high |",
        f"| Abstained on answerable, evidence retrieved | {fmt(s['abstain_answerable_with_evidence'])} | low |",
        f"| Abstained on answerable, evidence not retrieved | {fmt(s['abstain_answerable_without_evidence'])} | high |",
        f"| Citations valid and consistent with inline markers | {fmt(s['citations_valid'])} | high |",
        f"| Answer cites a chunk holding the gold evidence | {fmt(s['cites_gold_evidence'])} | high |",
        "",
        "## By question type",
        "",
        "| Type | n | Correct | Faithful (full) | Abstained |",
        "|---|---|---|---|---|",
    ]
    for t, v in s["by_type"].items():
        lines.append(f"| {t} | {v['n']} | {fmt(v['correct'])} | {fmt(v['faithful_full'])} | {fmt(v['abstained'])} |")
    lines.append("")
    return "\n".join(lines)


def dry_run(rows: list[dict], hits: dict, cache: ResponseCache, cfg) -> None:
    g = cfg["generation"]
    n_cached, in_tokens = 0, 0
    for row in rows:
        user = build_user_message(row["question"], hits[row["id"]])
        if cache.get(ResponseCache.key("generate", g["model"], SYSTEM_PROMPT, user)):
            n_cached += 1
            continue
        in_tokens += (len(SYSTEM_PROMPT) + len(user)) // 4                      # generator
        in_tokens += (len(JUDGE_SYSTEM) + len(user) + len(row["answer"]) + 400) // 4   # judge
    calls = 2 * (len(rows) - n_cached)
    pin, pout = PRICES_PER_MTOK[g["model"]]
    cost = (in_tokens * pin + calls * EST_OUTPUT_TOKENS * pout) / 1e6
    print(f"{len(rows)} questions, {n_cached} already cached; {calls} API calls to make")
    print(f"~{in_tokens:,} input tokens, ~{calls * EST_OUTPUT_TOKENS:,} output tokens (estimate)")
    print(f"estimated cost: ${cost:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate cited answers and score them.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ids", nargs="+", help="Only these gold question ids.")
    ap.add_argument("--limit", type=int, help="At most N questions, spread across types.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Retrieve and estimate cost; no API calls.")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    g = cfg["generation"]
    md_path = resolve(g["results_path"])
    json_path = md_path.with_suffix(".json")

    if args.report_only:
        records = json.loads(json_path.read_text(encoding="utf-8"))["records"]
    else:
        rows = select_rows(load_gold(resolve(cfg["eval"]["gold_path"])), args.ids, args.limit)
        hits = retrieve_all(cfg, rows)
        cache = ResponseCache(resolve(g["cache_path"]))
        if args.dry_run:
            dry_run(rows, hits, cache, cfg)
            return 0

        import anthropic

        gen = Generator(cfg)
        try:
            first = process_row(rows[0], hits[rows[0]["id"]], gen, cache, g["judge_model"], not args.no_judge)
        except anthropic.AuthenticationError:
            print("Authentication failed. Put ANTHROPIC_API_KEY=... in .env at the repo root "
                  "(see .env.example) or export it, then re-run.", file=sys.stderr)
            return 1
        records, errors = [first], []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_row, row, hits[row["id"]], gen, cache, g["judge_model"],
                                   not args.no_judge): row["id"] for row in rows[1:]}
            for fut in as_completed(futures):
                try:
                    records.append(fut.result())
                except anthropic.APIError as exc:
                    errors.append(f"{futures[fut]}: {type(exc).__name__}: {exc}")
                print(f"  {len(records) + len(errors)}/{len(rows)}", end="\r", flush=True)
        print()
        for e in errors:
            print(f"  ! {e}", file=sys.stderr)
        records.sort(key=lambda r: r["id"])

    summary = summarise(records)
    md_path.write_text(render_markdown(cfg, summary), encoding="utf-8")
    json_path.write_text(json.dumps({"summary": summary, "records": records}, indent=1, ensure_ascii=False),
                         encoding="utf-8")
    print(render_markdown(cfg, summary))
    print(f"Wrote {md_path} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
