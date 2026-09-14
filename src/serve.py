"""Day 6: serve the pipeline over HTTP, with a one-page UI.

    python -m src.serve                 # http://127.0.0.1:8000

POST /ask {"question": "..."} retrieves with the configuration Day 5 evaluated
(`generation.retrieval` in config.yaml), answers with citations or abstains, and
returns the answer, the sources behind it and per-stage latency. GET /health
reports the configuration. Models load at startup, so the first question is not
the slow one.

Every question is a paid Claude API call on the key in .env, and there is no
authentication or rate limit, so the server binds to localhost by default.
"""
from __future__ import annotations

import argparse
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import REPO_ROOT, load_config
from .generate import Generator

MAX_QUESTION_CHARS = 1000
PAGE = Path(__file__).resolve().parent / "static" / "index.html"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class Source(BaseModel):
    number: int
    cited: bool
    chunk_id: str
    paper_id: str
    title: str
    section: str
    url: str
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    abstained: bool
    refused: bool
    model: str
    sources: list[Source]
    timings_ms: dict[str, float]


def arxiv_url(paper_id: str) -> str:
    return f"https://arxiv.org/abs/{paper_id}"


def api_key_configured() -> bool:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


class Pipeline:
    """Retrieve, then generate, with the retrieval settings generation was evaluated on."""

    def __init__(self, config=None, retriever=None, generator=None):
        self.cfg = config or load_config()
        self.rc = self.cfg["generation"]["retrieval"]
        self.top_k = self.cfg["generation"]["top_k"]
        if retriever is None:
            from .retrieve import Retriever

            self.cfg["index"]["title_prefix"] = bool(self.rc.get("title_prefix"))
            retriever = Retriever(chunk_size=self.rc["chunk_size"], config=self.cfg)
        self.retriever = retriever
        self.generator = generator or Generator(config=self.cfg)
        # The encoder and cross-encoder are not guaranteed thread-safe; the API call is.
        self._retrieve_lock = threading.Lock()

    def warmup(self) -> None:
        self.retriever.warmup(method=self.rc["method"], rerank=self.rc["rerank"])

    def describe(self) -> dict:
        return {"chunk_size": self.rc["chunk_size"], "method": self.rc["method"], "rerank": self.rc["rerank"],
                "top_k": self.top_k, "model": self.generator.model, "chunks": len(self.retriever.chunks)}

    def ask(self, question: str) -> AskResponse:
        with self._retrieve_lock:
            result = self.retriever.retrieve(question, method=self.rc["method"], rerank=self.rc["rerank"],
                                             top_k=self.top_k)
        ans = self.generator.answer(question, result.hits)
        if ans.parse_failed:
            raise RuntimeError(f"The model returned no structured answer (stop_reason={ans.stop_reason}).")
        cited = set(ans.cited_sources)
        return AskResponse(
            question=question,
            answer=ans.answer,
            abstained=ans.abstained,
            refused=ans.refused,
            model=ans.model,
            sources=[
                Source(number=i, cited=i in cited, chunk_id=h.chunk_id, paper_id=h.paper_id, title=h.title,
                       section=h.section, url=arxiv_url(h.paper_id), text=h.text)
                for i, h in enumerate(result.hits, start=1)
            ],
            timings_ms={"retrieval_ms": result.timings_ms["total_ms"], "generation_ms": ans.latency_ms},
        )


def create_app(pipeline: Pipeline | None = None) -> FastAPI:
    """Build the app. Pass a pipeline to skip model loading (tests do)."""
    state = {"pipeline": pipeline, "key_missing": False}

    @asynccontextmanager
    async def lifespan(_app):
        if state["pipeline"] is None:
            state["key_missing"] = not api_key_configured()
            if state["key_missing"]:
                print("ANTHROPIC_API_KEY is not set: copy .env.example to .env and add it. /ask will return 503.")
            state["pipeline"] = Pipeline()
            state["pipeline"].warmup()
        yield

    app = FastAPI(title="rag-paper-qa", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE.read_text(encoding="utf-8")

    @app.get("/health")
    def health():
        return {"status": "ok", "api_key_configured": not state["key_missing"], **state["pipeline"].describe()}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest):
        question = req.question.strip()
        if not question:
            raise HTTPException(422, "The question is empty.")
        if state["key_missing"]:
            raise HTTPException(503, "ANTHROPIC_API_KEY is not set: copy .env.example to .env and add it.")
        try:
            return state["pipeline"].ask(question)
        except anthropic.AuthenticationError as exc:
            raise HTTPException(503, "The Claude API rejected the key in .env.") from exc
        except anthropic.APIError as exc:
            raise HTTPException(502, f"Claude API error: {type(exc).__name__}.") from exc
        except RuntimeError as exc:
            raise HTTPException(502, str(exc)) from exc

    return app


app = create_app()


def main() -> int:
    import uvicorn

    ap = argparse.ArgumentParser(description="Serve the question answering pipeline.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
