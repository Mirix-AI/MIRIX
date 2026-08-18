"""QA adapter for the general graph v7.23 evidence policy."""

from __future__ import annotations

from typing import Optional

from v721_policy import aggregation_instruction, normalize_count_answer
from mirix.services.retrieval_policy_v723 import plan_query, verification_decision


def is_v723(version: Optional[str]) -> bool:
    return (version or "").strip().lower() == "v7.23"


def question_operator(question: str) -> str:
    return plan_query(question).operator


__all__ = [
    "aggregation_instruction",
    "is_v723",
    "normalize_count_answer",
    "question_operator",
    "verification_decision",
]
