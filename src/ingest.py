"""Day 1: fetch arXiv papers, parse them section-aware, chunk them.

Pipeline:
    arXiv API (seed IDs + keyword top-up) -> PDFs -> per-section text
      -> token-window chunks -> data/chunks/chunks_{size}.jsonl

Every chunk carries paper_id, section and a stable, human-readable chunk_id
(``2403.06764::s03::c02``) because on Day 3 you will be reading chunk ids off
the screen and typing them into the gold set by hand. Make them legible.

Usage:
    python -m src.ingest                     # full run at the config chunk size
    python -m src.ingest --fetch-only        # resolve metadata, print titles, stop
    python -m src.ingest --chunk-size 256    # re-chunk (reuses downloaded PDFs)
    python -m src.ingest --chunk-only        # skip network, re-chunk existing PDFs
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import urlencode

import requests
from tqdm import tqdm

from .config import load_config, resolve

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM = {"atom": "http://www.w3.org/2005/Atom"}
# arXiv asks for one request every 3 seconds. Be a good citizen.
ARXIV_DELAY_S = 3.0
USER_AGENT = "rag-paper-qa/0.1 (research portfolio project; contact via GitHub)"


# --------------------------------------------------------------------------
# arXiv metadata + PDF download
# --------------------------------------------------------------------------

@dataclass
class Paper:
    arxiv_id: str          # version-stripped, e.g. 2403.06764
    title: str
    authors: list[str] = field(default_factory=list)
    published: str = ""
    categories: list[str] = field(default_factory=list)
    abstract: str = ""
    pdf_url: str = ""

    def to_json(self) -> dict:
        return {
            "paper_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "published": self.published,
            "categories": self.categories,
            "abstract": self.abstract,
        }


def _strip_version(raw: str) -> str:
    return re.sub(r"v\d+$", "", raw.strip())


def read_seed_ids(seed_file: Path) -> list[str]:
    ids: list[str] = []
    if not seed_file.exists():
        return ids
    for line in seed_file.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(_strip_version(line))
    return ids


def _parse_entries(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers: list[Paper] = []
    for entry in root.findall("atom:entry", ATOM):
        raw_id = entry.findtext("atom:id", default="", namespaces=ATOM)
        arxiv_id = _strip_version(raw_id.rsplit("/abs/", 1)[-1])
        if not arxiv_id:
            continue
        title = " ".join(
            (entry.findtext("atom:title", default="", namespaces=ATOM) or "").split()
        )
        abstract = " ".join(
            (entry.findtext("atom:summary", default="", namespaces=ATOM) or "").split()
        )
        authors = [
            (a.findtext("atom:name", default="", namespaces=ATOM) or "").strip()
            for a in entry.findall("atom:author", ATOM)
        ]
        categories = [
            c.attrib.get("term", "") for c in entry.findall("atom:category", ATOM)
        ]
        pdf_url = ""
        for link in entry.findall("atom:link", ATOM):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                published=entry.findtext("atom:published", default="", namespaces=ATOM) or "",
                categories=categories,
                abstract=abstract,
                pdf_url=pdf_url,
            )
        )
    return papers


def _api_get(params: dict) -> str:
    url = f"{ARXIV_API}?{urlencode(params)}"
    resp = requests.get(url, timeout=60, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    return resp.text


def fetch_by_ids(ids: Iterable[str], batch: int = 25) -> list[Paper]:
    ids = list(ids)
    out: list[Paper] = []
    for i in range(0, len(ids), batch):
        chunk = ids[i : i + batch]
        xml_text = _api_get({"id_list": ",".join(chunk), "max_results": len(chunk)})
        out.extend(_parse_entries(xml_text))
        time.sleep(ARXIV_DELAY_S)
    return out


def fetch_by_query(query: str, want: int, exclude: set[str]) -> list[Paper]:
    """Top the corpus up with a relevance-sorted keyword search."""
    out: list[Paper] = []
    start = 0
    page = 50
    while len(out) < want and start < 400:
        xml_text = _api_get(
            {
                "search_query": query,
                "start": start,
                "max_results": page,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
        )
        page_papers = _parse_entries(xml_text)
        if not page_papers:
            break
        for p in page_papers:
            if p.arxiv_id not in exclude and all(p.arxiv_id != q.arxiv_id for q in out):
                out.append(p)
                if len(out) >= want:
                    break
        start += page
        time.sleep(ARXIV_DELAY_S)
    return out


def download_pdfs(papers: list[Paper], pdf_dir: Path) -> list[Paper]:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    kept: list[Paper] = []
    for p in tqdm(papers, desc="downloading", unit="pdf"):
        dest = pdf_dir / f"{p.arxiv_id}.pdf"
        if dest.exists() and dest.stat().st_size > 10_000:
            kept.append(p)
            continue
        try:
            resp = requests.get(
                p.pdf_url, timeout=120, headers={"User-Agent": USER_AGENT}
            )
            resp.raise_for_status()
            if not resp.content.startswith(b"%PDF"):
                print(f"  ! {p.arxiv_id}: not a PDF, skipping", file=sys.stderr)
                continue
            dest.write_bytes(resp.content)
            kept.append(p)
            time.sleep(1.0)
        except Exception as exc:  # noqa: BLE001 - report and carry on
            print(f"  ! {p.arxiv_id}: download failed ({exc})", file=sys.stderr)
    return kept


# --------------------------------------------------------------------------
# PDF -> section-aware text
# --------------------------------------------------------------------------

# Section numbers look like 4, 4.2 or 3.1.2. Table values like 0.92 or 2.47 do not.
NUMBERED_HEADING = re.compile(r"^\s*([1-9]\d?(?:\.\d){0,2})\.?\s+([A-Z][^\n]{2,70})$")
NAMED_HEADINGS = {
    "abstract", "introduction", "related work", "background", "preliminaries",
    "method", "methods", "methodology", "approach", "our approach", "model",
    "experiments", "experimental setup", "experiments and results", "results",
    "evaluation", "analysis", "ablation", "ablation study", "ablation studies",
    "discussion", "limitations", "conclusion", "conclusions",
    "conclusion and future work", "future work", "references", "bibliography",
    "acknowledgment", "acknowledgments", "acknowledgement", "acknowledgements",
    "appendix",
}
NUMBER_ONLY = re.compile(r"^\s*[1-9]\d?(?:\.\d){0,2}\.?\s*$")
MAX_SECTION_NUMBER = 12
NUMERIC_TOKEN = re.compile(r"^[+\-]?\d+(?:\.\d+)?%?$")
NUMERIC_CELL = re.compile(r"\d\.\d|%|±")
TRAILING_FUNCTION_WORDS = {
    "a", "an", "and", "as", "by", "for", "in", "of", "on", "or", "the", "to", "we", "with",
}
ALLCAPS_HEADING = re.compile(r"^\s*([A-Z][A-Z0-9 \-&]{3,50})\s*$")
DEHYPHEN = re.compile(r"(\w)-\n(\w)")


def _is_heading(line: str) -> str | None:
    """Return a normalised heading title, or None if this is body text."""
    stripped = line.strip()
    if not stripped or len(stripped) > 80 or stripped.endswith("."):
        return None
    low = stripped.lower().strip(":")
    if low in NAMED_HEADINGS:
        return stripped.strip(":")
    m = NUMBERED_HEADING.match(stripped)
    if m:
        title = m.group(2).strip()
        words = title.split()
        top_level = int(m.group(1).split(".")[0])
        # Reject "3 dogs were counted" style false positives: headings are short
        # and do not read as sentences. Figure axes and table rows also match
        # the pattern ("40 To address the above limitations, we propose PagedAt-",
        # "65 CLIP 98.4 76.2", "2 WikiSQL (±0.5%)"), so reject implausible
        # section numbers, numeric cells, and titles that break off mid-sentence.
        if (
            len(words) <= 8
            and top_level <= MAX_SECTION_NUMBER
            and not any(NUMERIC_TOKEN.match(w) for w in words)
            and not NUMERIC_CELL.search(title)
            and not title.endswith(("-", ","))
            and words[-1].lower() not in TRAILING_FUNCTION_WORDS
        ):
            return f"{m.group(1)} {title}"
    m = ALLCAPS_HEADING.match(stripped)
    if m and len(m.group(1).split()) <= 8:
        return m.group(1).title()
    return None


def pdf_to_text(pdf_path: Path) -> str:
    import fitz  # PyMuPDF

    parts: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            # Native order, not sort=True: sorting orders lines by page position,
            # which reads straight across both columns of a two-column paper.
            parts.append(page.get_text("text", sort=False))
    text = "\n".join(parts)
    text = DEHYPHEN.sub(r"\1\2", text)          # join words split across lines
    text = re.sub(r"[ \t]+", " ", text)
    return text


def split_sections(text: str, stop_sections: set[str]) -> list[tuple[str, str]]:
    """Split raw paper text into [(section_title, section_text), ...].

    Everything from the first stop section (References etc.) onward is dropped.
    """
    sections: list[tuple[str, list[str]]] = [("Front matter", [])]
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw_line = lines[i]
        heading = _is_heading(raw_line)
        # Native reading order often puts a section number on its own line
        # ("4.2" then "APPLYING LORA TO TRANSFORMER"). Rejoin the pair when it
        # reads as a heading; otherwise the number stays as body text.
        if NUMBER_ONLY.match(raw_line) and i + 1 < len(lines):
            joined = _is_heading(f"{raw_line.strip()} {lines[i + 1].strip()}")
            if joined:
                heading = joined
                i += 1
        i += 1
        if heading:
            base = re.sub(r"^\d[\d.]*\s+", "", heading).lower().strip()
            if base in stop_sections:
                break
            sections.append((heading, []))
        else:
            sections[-1][1].append(raw_line)

    out: list[tuple[str, str]] = []
    for title, lines in sections:
        body = "\n".join(lines)
        # Collapse intra-paragraph newlines, keep blank-line paragraph breaks.
        body = re.sub(r"\n{2,}", "␟", body)
        body = body.replace("\n", " ")
        body = body.replace("␟", "\n\n")
        body = re.sub(r" {2,}", " ", body).strip()
        if body:
            out.append((title, body))
    return out


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

class TokenCounter:
    """bge tokenizer when available, whitespace approximation when not."""

    def __init__(self, model_name: str):
        self._tok = None
        try:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(model_name)
        except Exception as exc:  # noqa: BLE001
            print(
                f"  ! tokenizer '{model_name}' unavailable ({exc}); "
                "falling back to a whitespace estimate. Token counts will be "
                "approximate -- note this in the README limitations.",
                file=sys.stderr,
            )

    def encode(self, text: str) -> list:
        if self._tok is not None:
            return self._tok.encode(text, add_special_tokens=False)
        return text.split()

    def spans(self, text: str) -> list[tuple[int, int]]:
        """Character (start, end) of every token in ``text``.

        Chunks are cut as slices of the original text at these offsets, never
        decoded from token ids: bge's tokenizer is uncased, so decoding returns
        lowercased text with spaces around punctuation ("4 - bit ( nf4 )").
        """
        if self._tok is not None:
            return self._tok(
                text, add_special_tokens=False, return_offsets_mapping=True
            )["offset_mapping"]
        return [m.span() for m in re.finditer(r"\S+", text)]


def chunk_section(
    text: str,
    counter: TokenCounter,
    chunk_size: int,
    overlap: int,
    min_tokens: int,
) -> list[str]:
    spans = counter.spans(text)
    n = len(spans)
    if n == 0:
        return []
    if n <= chunk_size:
        return [text] if n >= min_tokens else []

    stride = max(1, chunk_size - overlap)
    windows: list[tuple[int, int]] = []   # [start, end) token indices
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        windows.append((start, end))
        if end >= n:
            break
        start += stride
    # A stubby tail carries no independent meaning: fold it into its predecessor.
    # Its first `overlap` tokens are already in that window, so extending the
    # predecessor to the end of the text adds only the genuinely new remainder.
    if len(windows) > 1 and windows[-1][1] - windows[-1][0] < min_tokens:
        windows.pop()
        windows[-1] = (windows[-1][0], n)
    return [text[spans[s][0] : spans[e - 1][1]].strip() for s, e in windows]


def build_chunks(
    papers: list[Paper],
    pdf_dir: Path,
    counter: TokenCounter,
    chunk_size: int,
    overlap: int,
    min_tokens: int,
    stop_sections: set[str],
) -> Iterator[dict]:
    for paper in tqdm(papers, desc="chunking", unit="paper"):
        pdf_path = pdf_dir / f"{paper.arxiv_id}.pdf"
        if not pdf_path.exists():
            continue
        try:
            text = pdf_to_text(pdf_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {paper.arxiv_id}: parse failed ({exc})", file=sys.stderr)
            continue
        sections = split_sections(text, stop_sections)
        # The API abstract is cleaner than the PDF's, so lead with it.
        if paper.abstract:
            sections.insert(0, ("Abstract", paper.abstract))
        for s_idx, (title, body) in enumerate(sections):
            for c_idx, chunk_text in enumerate(
                chunk_section(body, counter, chunk_size, overlap, min_tokens)
            ):
                yield {
                    "chunk_id": f"{paper.arxiv_id}::s{s_idx:02d}::c{c_idx:02d}",
                    "paper_id": paper.arxiv_id,
                    "title": paper.title,
                    "section": title,
                    "text": chunk_text,
                    "n_tokens": len(counter.encode(chunk_text)),
                }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def load_manifest(path: Path) -> list[Paper]:
    papers: list[Paper] = []
    if not path.exists():
        return papers
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        papers.append(
            Paper(
                arxiv_id=d["paper_id"],
                title=d.get("title", ""),
                authors=d.get("authors", []),
                published=d.get("published", ""),
                categories=d.get("categories", []),
                abstract=d.get("abstract", ""),
            )
        )
    return papers


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch, parse and chunk the corpus.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--fetch-only", action="store_true",
                    help="Resolve metadata and print titles, then stop.")
    ap.add_argument("--chunk-only", action="store_true",
                    help="Skip the network; re-chunk PDFs already downloaded.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    chunk_size = args.chunk_size or cfg["chunking"]["chunk_size"]
    overlap = int(chunk_size * cfg["chunking"]["overlap_ratio"])
    pdf_dir = resolve(cfg["corpus"]["pdf_dir"])
    manifest_path = resolve(cfg["corpus"]["manifest_path"])
    chunk_dir = resolve(cfg["corpus"]["chunk_dir"])
    stop_sections = {s.lower() for s in cfg["corpus"]["stop_sections"]}

    if args.chunk_only:
        papers = load_manifest(manifest_path)
        if not papers:
            print("No manifest found. Run without --chunk-only first.", file=sys.stderr)
            return 1
    else:
        seed_ids = read_seed_ids(resolve(cfg["corpus"]["seed_file"]))
        print(f"Seed IDs: {len(seed_ids)}")
        papers = fetch_by_ids(seed_ids) if seed_ids else []
        resolved = {p.arxiv_id for p in papers}
        missing = [i for i in seed_ids if i not in resolved]
        if missing:
            print(f"  ! unresolved seed IDs (check them): {', '.join(missing)}",
                  file=sys.stderr)

        target = cfg["corpus"]["target_count"]
        if len(papers) < target:
            need = target - len(papers)
            print(f"Topping up with {need} keyword hits...")
            papers.extend(fetch_by_query(cfg["corpus"]["keyword_query"], need, resolved))

        print(f"\nResolved {len(papers)} papers:")
        for p in papers:
            print(f"  {p.arxiv_id}  {p.title[:78]}")

        if args.fetch_only:
            print("\n--fetch-only: stopping before download. "
                  "Prune papers.txt and rerun when the list looks right.")
            return 0

        papers = download_pdfs(papers, pdf_dir)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as fh:
            for p in papers:
                fh.write(json.dumps(p.to_json()) + "\n")

    counter = TokenCounter(cfg["chunking"]["tokenizer"])
    chunk_dir.mkdir(parents=True, exist_ok=True)
    out_path = chunk_dir / f"chunks_{chunk_size}.jsonl"
    n = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for chunk in build_chunks(
            papers, pdf_dir, counter, chunk_size, overlap,
            cfg["chunking"]["min_chunk_tokens"], stop_sections,
        ):
            fh.write(json.dumps(chunk) + "\n")
            n += 1

    print(f"\nWrote {n} chunks ({chunk_size} tokens, {overlap} overlap) -> {out_path}")
    print(f"Papers: {len(papers)}  |  mean chunks/paper: {n / max(1, len(papers)):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
