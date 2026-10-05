"""Error answers of the eval gateway (``evals/skill_bench/infra/gateway.py``), as MIRIX
receives them through it: for the tests of MIRIX's retry-policy classification.

Built exactly as the gateway builds them (``_error_body``, ``_deadline`` and the
``blame`` its classifier adds); the harness's own tests check the gateway side.
"""

from __future__ import annotations

from typing import Any, Dict

#: The 504 when the model's single attempt used the whole 600 s deadline.
GATEWAY_MODEL_TIMEOUT: Dict[str, Any] = {
    "error": {
        "source": "gateway",
        "type": "deadline_exceeded",
        "code": 504,
        "message": "call exceeded the 600s deadline (deadline_during_attempt)",
        "reason": "deadline_during_attempt",
        "retries": 0,
        "upstream_attempts": 1,
        "deadline_s": 600.0,
        "attempt_budget_s": 599.998,
        "queue_wait_s": 0.0,
        "pre_attempt_s": 0.002,
        "blame": "model",
    }
}
#: The same 504 after the gateway's retries ate the deadline: infrastructure.
GATEWAY_LOST_DEADLINE: Dict[str, Any] = {
    "error": {
        **GATEWAY_MODEL_TIMEOUT["error"],
        "message": "call exceeded the 600s deadline (backoff_exceeds_deadline)",
        "reason": "backoff_exceeds_deadline",
        "retries": 3,
        "upstream_attempts": 4,
        "blame": "infra",
    }
}
MODERATION_MESSAGE = (
    "example/chat-model requires moderation on ExampleProvider. Your input was "
    'flagged for "violence". No credits were charged.'
)
#: OpenRouter's moderation refusal as the gateway forwards it.
GATEWAY_MODERATION_403: Dict[str, Any] = {
    "error": {
        "source": "upstream",
        "type": "upstream_error",
        "code": 403,
        "message": MODERATION_MESSAGE,
        "upstream": {"code": 403, "type": None, "message": MODERATION_MESSAGE},
        "blame": "model",
    }
}
#: A 403 that is a real configuration problem.
FORBIDDEN_403: Dict[str, Any] = {
    "error": {"code": 403, "message": "this key may not use the model"}
}
