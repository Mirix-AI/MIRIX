"""QA operator adapter for the benchmark-agnostic graph v7.22 policy."""

from __future__ import annotations

from typing import Optional

from v721_policy import aggregation_instruction, normalize_count_answer
from mirix.services.retrieval_policy_v722 import plan_query


def is_v722(version: Optional[str]) -> bool:
    return (version or "").strip().lower() == "v7.22"


def question_operator(question: str) -> str:
    return plan_query(question).operator


__all__ = [
    "aggregation_instruction",
    "is_v722",
    "normalize_count_answer",
    "question_operator",
]
