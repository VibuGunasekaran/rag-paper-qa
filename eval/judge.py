"""LLM judge for Day 5: faithfulness to the retrieved sources and correctness against the gold answer.

Faithfulness is judged only against the sources the generator saw, so a claim
that happens to be true but is not in those sources counts as unsupported.
Correctness is judged against the gold answer. The two are kept separate
because a faithful answer can be wrong (the retrieved sources were wrong) and a
correct answer can be unfaithful (the model answered from memory).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from src.generate import FALLBACK_BETA, _usage_dict, format_sources

JUDGE_SYSTEM = """You grade answers from a retrieval-augmented question answering system over research papers. You get the numbered sources the system saw, the question, a reference answer written by someone who read the papers, and the system's answer.

Judge two things separately.

Faithfulness: is every factual claim in the system answer stated in, or directly implied by, the sources? Only the sources count, not what is true elsewhere or what the reference says. A claim the sources do not contain is unsupported even if the reference agrees with it.
- full: every claim is supported by the sources.
- partial: at least one claim is supported and at least one is not.
- none: no claim is supported.
An abstention that makes no factual claims beyond what the sources show is full.

Correctness: does the system answer agree with the reference answer?
- correct: it gives the reference's key facts; extra supported detail is fine.
- partial: it gives some of the key facts, or is imprecise about them.
- incorrect: it contradicts the reference or misses its key facts.
- abstained: the system declined to answer.
When the reference says the question cannot be answered from the corpus, an abstention is correct and a substantive answer is incorrect.

List each unsupported claim in a few words; leave the list empty when there are none."""


class Judgement(BaseModel):
    unsupported_claims: list[str]
    faithfulness: Literal["full", "partial", "none"]
    correctness: Literal["correct", "partial", "incorrect", "abstained"]


@dataclass
class JudgeResult:
    faithfulness: str | None
    correctness: str | None
    unsupported_claims: list[str]
    model: str
    stop_reason: str
    refused: bool
    parse_failed: bool
    latency_ms: float
    usage: dict = field(default_factory=dict)


def build_judge_message(question: str, hits, answer: str, abstained: bool, reference: str,
                        unanswerable: bool) -> str:
    note = " (The reference marks this question as not answerable from the corpus.)" if unanswerable else ""
    return (
        f"{format_sources(hits)}\n\n"
        f"<question>{question}</question>\n\n"
        f"<reference_answer>{reference}{note}</reference_answer>\n\n"
        f'<system_answer abstained="{str(abstained).lower()}">{answer}</system_answer>'
    )


def judge_answer(client, model: str, message: str, max_tokens: int = 16000) -> JudgeResult:
    t = time.perf_counter()
    response = client.beta.messages.parse(
        model=model,
        max_tokens=max_tokens,
        betas=[FALLBACK_BETA],
        fallbacks="default",
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": message}],
        output_format=Judgement,
    )
    refused = response.stop_reason == "refusal"
    parsed = None if refused else getattr(response, "parsed_output", None)
    return JudgeResult(
        faithfulness=parsed.faithfulness if parsed else None,
        correctness=parsed.correctness if parsed else None,
        unsupported_claims=list(parsed.unsupported_claims) if parsed else [],
        model=response.model,
        stop_reason=response.stop_reason,
        refused=refused,
        parse_failed=parsed is None and not refused,
        latency_ms=round((time.perf_counter() - t) * 1000, 1),
        usage=_usage_dict(response.usage),
    )
