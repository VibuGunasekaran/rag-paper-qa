"""Day 5: answer a question from retrieved chunks, with citations and abstention.

    from src.generate import Generator
    gen = Generator()
    ans = gen.answer(question, hits)        # hits from Retriever.retrieve(...).hits

The model sees numbered sources [S1]..[Sk] and must cite every claim with the
source numbers it relies on, or abstain when the sources do not contain the
answer. Output is validated against a Pydantic schema, so the eval never scrapes
citations out of prose.

Requests use server-side refusal fallbacks (`fallbacks="default"`): if Claude
Opus 5 declines, the API re-runs the request on Anthropic's recommended fallback
model inside the same call, and `Answer.model` records which model answered.

The system prompt is shorter than the 512-token minimum for prompt caching on
Claude Opus 5, and the retrieved sources differ per question, so no cache
markers are set.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .config import REPO_ROOT, load_config

FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """You answer questions about research papers using only the numbered sources provided with each question.

- Use only information stated in the sources, even when you know more about the topic.
- Support every factual claim with the numbers of the sources it comes from, written inline like [S2] or [S1][S3].
- If the sources do not contain enough information to answer, abstain: set abstained to true and say briefly what is missing. Do not guess.
- If the sources answer only part of the question, answer that part with citations and say what the sources do not cover.
- Keep the answer to one to four sentences.
- List in cited_sources every source number you cite."""


class GeneratedAnswer(BaseModel):
    abstained: bool = Field(description="True when the sources do not contain the answer.")
    answer: str = Field(description="The answer with inline [S#] citations, or a brief explanation of what is missing.")
    cited_sources: list[int] = Field(description="Every source number cited in the answer.")


@dataclass
class Answer:
    question: str
    abstained: bool
    answer: str
    cited_sources: list[int]
    cited_chunk_ids: list[str]
    source_chunk_ids: list[str]
    requested_model: str
    model: str                  # the model that produced the answer; differs after a fallback
    stop_reason: str
    refused: bool
    parse_failed: bool
    latency_ms: float
    usage: dict = field(default_factory=dict)


def format_sources(hits) -> str:
    return "\n\n".join(
        f'<source id="S{i}" paper="{h.title}" section="{h.section}">\n{h.text}\n</source>'
        for i, h in enumerate(hits, start=1)
    )


def build_user_message(question: str, hits) -> str:
    """Sources first, question last: long documents before the query."""
    return f"{format_sources(hits)}\n\nQuestion: {question}"


def inline_citations(text: str) -> list[int]:
    """Source numbers cited inline, e.g. "[S1][S3]" or "[S1, S3]" -> [1, 3]."""
    numbers = set()
    for group in re.findall(r"\[([^\]]*)\]", text):
        numbers.update(int(n) for n in re.findall(r"S(\d+)", group))
    return sorted(numbers)


def _usage_dict(usage) -> dict:
    if usage is None:
        return {}
    to_dict = getattr(usage, "to_dict", None)
    return to_dict() if callable(to_dict) else dict(vars(usage))


def _to_answer(response, question: str, hits, latency_ms: float, requested_model: str) -> Answer:
    refused = response.stop_reason == "refusal"
    parsed = None if refused else getattr(response, "parsed_output", None)
    cited = sorted(set(parsed.cited_sources)) if parsed else []
    return Answer(
        question=question,
        abstained=bool(parsed.abstained) if parsed else False,
        answer=parsed.answer if parsed else "",
        cited_sources=cited,
        cited_chunk_ids=[hits[c - 1].chunk_id for c in cited if 1 <= c <= len(hits)],
        source_chunk_ids=[h.chunk_id for h in hits],
        requested_model=requested_model,
        model=response.model,
        stop_reason=response.stop_reason,
        refused=refused,
        parse_failed=parsed is None and not refused,
        latency_ms=round(latency_ms, 1),
        usage=_usage_dict(response.usage),
    )


def make_client():
    """Anthropic client; reads ANTHROPIC_API_KEY from the environment or a repo-root .env."""
    import anthropic

    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    return anthropic.Anthropic()


class Generator:
    def __init__(self, config=None, client=None):
        self.cfg = config or load_config()
        g = self.cfg["generation"]
        self.model = g["model"]
        self.max_tokens = g["max_tokens"]
        self._client = client

    @property
    def client(self):
        if self._client is None:
            self._client = make_client()
        return self._client

    def answer(self, question: str, hits) -> Answer:
        t = time.perf_counter()
        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_message(question, hits)}],
            output_format=GeneratedAnswer,
        )
        return _to_answer(response, question, hits, (time.perf_counter() - t) * 1000, self.model)
