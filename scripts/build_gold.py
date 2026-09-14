"""Day 3 helper: write the gold set without copying chunk ids by hand.

Day 3 is the day the project lives or dies, and the thing that makes it
miserable is transcription -- reading a chunk id off the screen, typing it into
a JSONL file, and getting one character wrong so the question silently scores
zero recall forever. This does the transcription for you and validates as it
goes. It does NOT write your questions. That part is the project.

    python scripts/build_gold.py add        # interactive authoring loop
    python scripts/build_gold.py validate   # schema + chunk ids + type mix
    python scripts/build_gold.py stats      # progress against the target mix
    python scripts/build_gold.py browse -q "visual token pruning"

Row schema (eval/gold_set.jsonl):
    {"id", "question", "answer", "gold_chunks": [chunk_id, ...],
     "gold_quotes": [{"chunk_id", "quote"}, ...], "type", "notes"}
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve          # noqa: E402
from src.index import bm25_tokenize                  # noqa: E402
from src.retrieve import Retriever                   # noqa: E402

TYPES = ["single_fact", "multi_hop", "comparative", "paraphrased", "unanswerable"]
TYPE_HELP = {
    "single_fact": "answer sits in one chunk",
    "multi_hop": "needs two or more chunks combined",
    "comparative": "compares across papers",
    "paraphrased": "no keyword overlap with the source wording",
    "unanswerable": "plausible, but the corpus cannot answer it",
}
# A paraphrased question sharing this share of its terms with the evidence is
# not paraphrased: BM25 would find it, and the dense-vs-lexical comparison
# these rows exist for would measure nothing.
PARAPHRASE_MAX_OVERLAP = 0.5


def load_gold(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def next_id(gold: list[dict]) -> str:
    """One past the highest existing id, so hand-deleted rows never cause a clash."""
    nums = [int(m.group(1)) for g in gold if (m := re.match(r"q(\d+)$", g.get("id", "")))]
    return f"q{max(nums, default=0) + 1:03d}"


def print_progress(gold: list[dict], target: dict) -> None:
    counts = {t: sum(1 for g in gold if g.get("type") == t) for t in TYPES}
    total_target = sum(target.values())
    print(f"\n  progress: {len(gold)}/{total_target}")
    for t in TYPES:
        want = target.get(t, 0)
        bar = "#" * counts[t] + "." * max(0, want - counts[t])
        flag = "  <-- done" if counts[t] >= want else ""
        print(f"    {t:<14} {counts[t]:>2}/{want:<2} {bar}{flag}")
    print()


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\\[])")


def normalise(text: str) -> str:
    return " ".join(text.split()).lower()


def nth(items: list, raw: str):
    """1-based lookup that rejects 0 and negatives instead of wrapping around."""
    n = int(raw)
    if not 1 <= n <= len(items):
        raise IndexError(n)
    return items[n - 1]


def pick_quote(chunk_text: str) -> str:
    """Pick the answer-bearing sentence from a chunk, by number, not by typing.

    Why this exists: gold_chunks are chunk ids, and chunk ids only mean anything
    inside one chunk size. Day 4 sweeps 256/512/1024, so a gold set recorded
    only as 512-token ids cannot be scored against the other two -- the ids do
    not exist there. Recording the evidence *sentence* alongside makes recall
    computable at any chunk size: a retrieved chunk counts as a hit if it
    contains the quote. Retrofitting this after 50 questions is a bad evening.
    """
    sentences = [s.strip() for s in SENT_SPLIT.split(chunk_text) if len(s.strip()) > 25]
    print("\n      which sentence actually contains the answer? (shorter spans survive "
          "smaller chunk sizes)")
    for i, s in enumerate(sentences, 1):
        print(f"        ({i}) {s[:160]}{'...' if len(s) > 160 else ''}")
    while True:
        raw = input("      number | 'p' paste a span from this chunk | 'n' no quote: ").strip()
        if raw.lower() == "n":
            print("      ! no quote: this row can only be scored at this chunk size")
            return ""
        if raw.lower() == "p":
            pasted = input("      paste: ").strip()
            if pasted and normalise(pasted) in normalise(chunk_text):
                return pasted
            print("      ! that text is not in this chunk verbatim, try again")
            continue
        try:
            return nth(sentences, raw)
        except (ValueError, IndexError):
            print("      ! not one of the numbers above")


def show_candidates(hits, offset: int = 0) -> None:
    for h in hits:
        print(f"  [{h.rank + offset}] {h.chunk_id}")
        print(f"      {h.paper_id} | {h.section[:48]}")
        print(f"      {h.preview(240)}\n")


def cmd_add(args) -> int:
    cfg = load_config()
    gold_path = resolve(cfg["eval"]["gold_path"])
    target = dict(cfg["eval"]["target_mix"])
    gold = load_gold(gold_path)
    retriever = Retriever(chunk_size=args.chunk_size, config=cfg)

    print("Gold set authoring. Blank question to quit; Ctrl-C is safe (rows are")
    print("appended as you go, nothing is buffered).")
    print_progress(gold, target)

    while True:
        try:
            question = input("Question (blank to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break

        print("\n  type:  " + "  ".join(f"{i+1}={t}" for i, t in enumerate(TYPES)))
        for t in TYPES:
            print(f"         {t:<14} {TYPE_HELP[t]}")
        raw = input("  choose 1-5: ").strip()
        try:
            qtype = TYPES[int(raw) - 1]
        except (ValueError, IndexError):
            print("  ! not a valid choice, skipping this question\n")
            continue

        gold_chunks: list[str] = []
        gold_quotes: list[dict] = []
        if qtype == "unanswerable":
            print("\n  Unanswerable: no gold chunks. The correct behaviour is abstention.")
            answer = input("  What should it say / why is this unanswerable? ").strip()
        else:
            search_q = question
            out = None
            while True:
                if out is None:
                    out = retriever.retrieve(search_q, method="hybrid", rerank=True, top_k=10,
                                             candidate_k=30)
                    print()
                    show_candidates(out.hits)
                    print("  pick: numbers (1,4) | 'id <chunk_id> ...' | 'v <n>' full text | "
                          "'s <query>' search again, keeps picks | 'clear' | 'skip'")
                    print("  Empty line when done.")
                if gold_chunks:
                    print(f"  picked: {', '.join(gold_chunks)}")
                sel = input("  > ").strip()
                low = sel.lower()
                if not sel:
                    if gold_chunks:
                        break
                    continue
                if low == "skip":
                    gold_chunks = []
                    break
                if low == "clear":
                    gold_chunks = []
                    continue
                if low.startswith("s "):
                    search_q = sel[2:].strip()
                    out = None
                    continue
                if low.startswith("v "):
                    try:
                        hit = nth(out.hits, sel[2:])
                        print(f"\n  --- {hit.chunk_id} | {hit.section}\n{hit.text}\n  ---\n")
                    except (ValueError, IndexError):
                        print("  ! 'v' takes one of the numbers above")
                    continue
                if low.startswith("id "):
                    new = sel[3:].replace(",", " ").split()
                    unknown = [c for c in new if c not in retriever.by_id]
                    if unknown:
                        print(f"  ! not chunk ids in this index: {', '.join(unknown)}")
                        continue
                else:
                    try:
                        new = [nth(out.hits, p).chunk_id
                               for p in sel.replace(" ", "").split(",") if p]
                    except (ValueError, IndexError):
                        print("  ! use numbers from the list above, or one of the commands")
                        continue
                gold_chunks += [c for c in new if c not in gold_chunks]
            if not gold_chunks:
                print("  ! no chunks selected, skipping this question\n")
                continue
            if qtype == "multi_hop" and len(gold_chunks) < 2:
                print("  ! multi_hop needs 2+ gold chunks. Recorded anyway -- fix it")
                print("    later or validate will flag it.")
            if qtype == "comparative" and len({c.split("::")[0] for c in gold_chunks}) < 2:
                print("  ! comparative needs chunks from 2+ papers. Recorded anyway -- fix it")
                print("    later or validate will flag it.")
            for cid in gold_chunks:
                q = pick_quote(retriever.by_id[cid]["text"])
                if q:
                    gold_quotes.append({"chunk_id": cid, "quote": q})
            answer = input("\n  Answer (your own words, from those chunks): ").strip()

        notes = input("  Notes (optional, Enter to skip): ").strip()
        row = {
            "id": next_id(gold),
            "question": question,
            "answer": answer,
            "gold_chunks": gold_chunks,
            "gold_quotes": gold_quotes,
            "type": qtype,
            "notes": notes,
        }
        append_row(gold_path, row)
        gold.append(row)
        print(f"  saved {row['id']}")
        print_progress(gold, target)

    print(f"Gold set: {len(gold)} rows in {gold_path}")
    return 0


def cmd_validate(args) -> int:
    cfg = load_config()
    gold_path = resolve(cfg["eval"]["gold_path"])
    target = dict(cfg["eval"]["target_mix"])
    gold = load_gold(gold_path)
    if not gold:
        print(f"No gold set at {gold_path}.", file=sys.stderr)
        return 1

    retriever = Retriever(chunk_size=args.chunk_size, config=cfg)
    known = set(retriever.by_id)
    # Day 4 scores every chunk size against the same quotes, so a quote has to
    # fit inside one chunk at each size, not only the size it was picked at.
    index_root = resolve(cfg["index"]["dir"])
    other_sizes: dict[int, dict[str, list[str]]] = {}
    for size in (256, 512, 1024):
        path = index_root / str(size) / "chunks.jsonl"
        if size == retriever.chunk_size or not path.exists():
            continue
        by_paper: dict[str, list[str]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                c = json.loads(line)
                by_paper.setdefault(c["paper_id"], []).append(normalise(c["text"]))
        other_sizes[size] = by_paper
    problems: list[str] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()

    for i, row in enumerate(gold, 1):
        rid = row.get("id", f"row{i}")
        for field_name in ("question", "answer", "type", "gold_chunks"):
            if field_name not in row:
                problems.append(f"{rid}: missing field '{field_name}'")
        if row.get("type") not in TYPES:
            problems.append(f"{rid}: unknown type {row.get('type')!r}")
        if rid in seen_ids:
            problems.append(f"{rid}: duplicate id")
        seen_ids.add(rid)
        q = row.get("question", "").strip().lower()
        if q in seen_questions:
            problems.append(f"{rid}: duplicate question")
        seen_questions.add(q)

        chunks = row.get("gold_chunks", [])
        if row.get("type") == "unanswerable":
            if chunks:
                problems.append(f"{rid}: unanswerable but has gold_chunks")
        else:
            if not chunks:
                problems.append(f"{rid}: no gold_chunks")
            for cid in chunks:
                if cid not in known:
                    problems.append(
                        f"{rid}: chunk_id {cid} not in the {retriever.chunk_size}-token index"
                    )
            if row.get("type") == "multi_hop" and len(chunks) < 2:
                problems.append(f"{rid}: multi_hop with only {len(chunks)} gold chunk")
            if row.get("type") == "comparative" and len({c.split("::")[0] for c in chunks}) < 2:
                problems.append(f"{rid}: comparative but every gold chunk is from one paper")
            if row.get("type") == "paraphrased":
                q_terms = set(bm25_tokenize(row.get("question", "")))
                quote_terms = set(bm25_tokenize(
                    " ".join(gq.get("quote", "") for gq in row.get("gold_quotes", []))
                ))
                shared = q_terms & quote_terms
                if q_terms and len(shared) / len(q_terms) >= PARAPHRASE_MAX_OVERLAP:
                    problems.append(
                        f"{rid}: paraphrased but shares {len(shared)}/{len(q_terms)} terms "
                        f"with its quotes ({', '.join(sorted(shared))})"
                    )
            quotes = row.get("gold_quotes", [])
            if not quotes:
                problems.append(
                    f"{rid}: no gold_quotes -- this row cannot be scored at any "
                    f"chunk size other than {retriever.chunk_size}"
                )
            for gq in quotes:
                cid, quote = gq.get("chunk_id"), gq.get("quote", "")
                if cid in known and normalise(quote) not in normalise(retriever.by_id[cid]["text"]):
                    problems.append(f"{rid}: quote not found verbatim in {cid}")
                paper = (cid or "").split("::")[0]
                for size, by_paper in other_sizes.items():
                    if not any(normalise(quote) in text for text in by_paper.get(paper, [])):
                        problems.append(
                            f"{rid}: quote from {cid} does not fit inside any {size}-token "
                            "chunk -- pick a shorter span"
                        )

    counts = {t: sum(1 for g in gold if g.get("type") == t) for t in TYPES}
    checked = ", ".join(str(s) for s in other_sizes) or "none built"
    print(f"Validating {len(gold)} rows against the {retriever.chunk_size}-token index "
          f"(quotes also checked at: {checked})\n")
    for t in TYPES:
        want = target.get(t, 0)
        status = "ok" if counts[t] == want else ("over" if counts[t] > want else "short")
        print(f"  {t:<14} {counts[t]:>2} / {want:<2}  {status}")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  ! {p}")
        return 1
    print("\nNo problems. Gold set is clean.")
    return 0


def cmd_stats(args) -> int:
    cfg = load_config()
    gold = load_gold(resolve(cfg["eval"]["gold_path"]))
    print_progress(gold, dict(cfg["eval"]["target_mix"]))
    if gold:
        papers = {c.split("::")[0] for g in gold for c in g.get("gold_chunks", [])}
        multi = sum(1 for g in gold if len(g.get("gold_chunks", [])) > 1)
        print(f"  distinct papers covered: {len(papers)}")
        print(f"  rows with 2+ gold chunks: {multi}\n")
    return 0


def cmd_browse(args) -> int:
    cfg = load_config()
    r = Retriever(chunk_size=args.chunk_size, config=cfg)
    out = r.retrieve(args.q, method="hybrid", rerank=True, top_k=args.n, candidate_k=30)
    show_candidates(out.hits)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Build and validate the gold set.")
    ap.add_argument("--chunk-size", type=int, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("add")
    sub.add_parser("validate")
    sub.add_parser("stats")
    b = sub.add_parser("browse")
    b.add_argument("-q", required=True)
    b.add_argument("-n", type=int, default=10)
    args = ap.parse_args()

    return {"add": cmd_add, "validate": cmd_validate,
            "stats": cmd_stats, "browse": cmd_browse}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
