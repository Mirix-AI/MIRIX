"""
Failover error classification for MetaClaw.

Mirrors OpenClaw's error classification system (failover-matches.ts,
errors.ts, failover-error.ts) to distinguish different error types and
return appropriate HTTP status codes instead of a blanket 502.

Error types and corresponding HTTP codes:
- auth:           401 — token expired, invalid key, re-authenticate needed
- auth_permanent: 403 — key revoked/deactivated, permission denied
- billing:        402 — insufficient credits, payment required
- rate_limit:     429 — rate limited, quota exceeded, too many requests
- overloaded:     503 — provider overloaded, high demand
- timeout:        408 — request timed out, network error, connection reset
- format:         400 — invalid request format, tool call errors
- model_not_found:404 — model does not exist
- session_expired: 410 — CLI session no longer valid
- unknown:        502 — unrecognized error
"""

from __future__ import annotations

import re
from typing import Literal

FailoverReason = Literal[
    "auth",
    "auth_permanent",
    "billing",
    "rate_limit",
    "overloaded",
    "timeout",
    "format",
    "model_not_found",
    "session_expired",
    "unknown",
]

# ── Pattern definitions (from OpenClaw failover-matches.ts) ───────

_RATE_LIMIT_PATTERNS: list[re.Pattern | str] = [
    re.compile(r"rate[_ ]limit|too many requests|429"),
    "model_cooldown",
    "exceeded your current quota",
    "resource has been exhausted",
    "quota exceeded",
    "resource_exhausted",
    "usage limit",
    re.compile(r"\btpm\b", re.I),
    "tokens per minute",
    "tokens per day",
]

_OVERLOADED_PATTERNS: list[re.Pattern | str] = [
    re.compile(r'overloaded_error|"type"\s*:\s*"overloaded_error"', re.I),
    "overloaded",
    re.compile(
        r"service[_ ]unavailable.*(?:overload|capacity|high[_ ]demand)"
        r"|(?:overload|capacity|high[_ ]demand).*service[_ ]unavailable",
        re.I,
    ),
    "high demand",
]

_SERVER_ERROR_PATTERNS: list[re.Pattern | str] = [
    "an error occurred while processing",
    "internal server error",
    "internal_error",
    "server_error",
    "service temporarily unavailable",
    "service_unavailable",
    "bad gateway",
    "gateway timeout",
    "upstream error",
    "upstream connect error",
    "connection reset",
]

_TIMEOUT_PATTERNS: list[re.Pattern | str] = [
    "timeout",
    "timed out",
    "service unavailable",
    "deadline exceeded",
    "context deadline exceeded",
    "connection error",
    "network error",
    "network request failed",
    "fetch failed",
    "socket hang up",
    re.compile(r"\beconn(?:refused|reset|aborted)\b", re.I),
    re.compile(r"\benetunreach\b", re.I),
    re.compile(r"\behostunreach\b", re.I),
    re.compile(r"\betimedout\b", re.I),
    re.compile(r"\bepipe\b", re.I),
    re.compile(r"\benotfound\b", re.I),
    re.compile(r"\beai_again\b", re.I),
]

_BILLING_PATTERNS: list[re.Pattern | str] = [
    re.compile(
        r"""["']?(?:status|code)["']?\s*[:=]\s*402\b"""
        r"""|\bhttp\s*402\b"""
        r"""|\berror(?:\s+code)?\s*[:=]?\s*402\b"""
        r"""|\b(?:got|returned|received)\s+(?:a\s+)?402\b"""
        r"""|^\s*402\s+payment""",
        re.I,
    ),
    "payment required",
    "insufficient credits",
    re.compile(r"insufficient[_ ]quota", re.I),
    "credit balance",
    "plans & billing",
    "insufficient balance",
    re.compile(r"requires?\s+more\s+credits", re.I),
]

_AUTH_PERMANENT_PATTERNS: list[re.Pattern | str] = [
    re.compile(r"api[_ ]?key[_ ]?(?:revoked|invalid|deactivated|deleted)", re.I),
    "invalid_api_key",
    "key has been disabled",
    "key has been revoked",
    "account has been deactivated",
    re.compile(r"could not (?:authenticate|validate).*(?:api[_ ]?key|credentials)", re.I),
    "permission_error",
    "not allowed for this organization",
]

_AUTH_PATTERNS: list[re.Pattern | str] = [
    re.compile(r"invalid[_ ]?api[_ ]?key"),
    "incorrect api key",
    "invalid token",
    "invalid bearer token",
    "authentication",
    "authentication_error",
    "authentication_failed",
    "re-authenticate",
    "oauth token refresh failed",
    "unauthorized",
    "forbidden",
    "access denied",
    "insufficient permissions",
    re.compile(r"missing scopes?:", re.I),
    "expired",
    "token has expired",
    re.compile(r"\b401\b"),
    re.compile(r"\b403\b"),
    "no credentials found",
    "no api key found",
    re.compile(r"\bfailed to (?:extract|parse|validate|decode)\b.*\btoken\b"),
]

_MODEL_NOT_FOUND_PATTERNS: list[re.Pattern | str] = [
    "model not found",
    "model_not_found",
    "does not exist",
    "is not available",
    "unknown model",
    "invalid model",
]

_SESSION_EXPIRED_PATTERNS: list[re.Pattern | str] = [
    "session not found",
    "session expired",
    "conversation invalid",
    "no such session",
    "invalid session",
    "session id not found",
    "conversation id not found",
]


# ── Matching engine ───────────────────────────────────────────────

def _matches(raw: str, patterns: list[re.Pattern | str]) -> bool:
    """Check if raw error text matches any of the given patterns."""
    if not raw:
        return False
    lower = raw.lower()
    for p in patterns:
        if isinstance(p, re.Pattern):
            if p.search(lower):
                return True
        elif p in lower:
            return True
    return False


def is_rate_limit_error(raw: str) -> bool:
    return _matches(raw, _RATE_LIMIT_PATTERNS)


def is_overloaded_error(raw: str) -> bool:
    return _matches(raw, _OVERLOADED_PATTERNS)


def is_billing_error(raw: str) -> bool:
    return _matches(raw, _BILLING_PATTERNS)


def is_auth_permanent_error(raw: str) -> bool:
    return _matches(raw, _AUTH_PERMANENT_PATTERNS)


def is_auth_error(raw: str) -> bool:
    return _matches(raw, _AUTH_PATTERNS)


def is_timeout_error(raw: str) -> bool:
    return _matches(raw, _TIMEOUT_PATTERNS)


def is_server_error(raw: str) -> bool:
    return _matches(raw, _SERVER_ERROR_PATTERNS)


def is_model_not_found_error(raw: str) -> bool:
    return _matches(raw, _MODEL_NOT_FOUND_PATTERNS)


def is_session_expired_error(raw: str) -> bool:
    return _matches(raw, _SESSION_EXPIRED_PATTERNS)


# ── Main classifier (mirrors OpenClaw classifyFailoverReason) ─────

def classify_failover_reason(raw: str) -> FailoverReason:
    """Classify an error message into a failover reason.

    Order matters — more specific classifiers run first, matching
    OpenClaw's classifyFailoverClassificationFromMessage priority.
    """
    if not raw or not raw.strip():
        return "unknown"

    # Session expired (most specific for CLI sessions)
    if is_session_expired_error(raw):
        return "session_expired"

    # Model not found
    if is_model_not_found_error(raw):
        return "model_not_found"

    # Rate limit (before billing — some overlap in patterns)
    if is_rate_limit_error(raw):
        return "rate_limit"

    # Overloaded
    if is_overloaded_error(raw):
        return "overloaded"

    # Billing (after rate limit to avoid misclassification)
    if is_billing_error(raw):
        return "billing"

    # Auth permanent (before general auth)
    if is_auth_permanent_error(raw):
        return "auth_permanent"

    # Auth (general)
    if is_auth_error(raw):
        return "auth"

    # Server error (before timeout — more specific)
    if is_server_error(raw):
        return "timeout"

    # Timeout / network
    if is_timeout_error(raw):
        return "timeout"

    return "unknown"


# ── HTTP status mapping (mirrors OpenClaw resolveFailoverStatus) ──

def resolve_failover_status(reason: FailoverReason) -> int:
    """Map a failover reason to an HTTP status code."""
    _STATUS_MAP: dict[FailoverReason, int] = {
        "billing": 402,
        "rate_limit": 429,
        "overloaded": 503,
        "auth": 401,
        "auth_permanent": 403,
        "timeout": 408,
        "format": 400,
        "model_not_found": 404,
        "session_expired": 410,
        "unknown": 502,
    }
    return _STATUS_MAP.get(reason, 502)


# ── User-facing error messages ────────────────────────────────────

def format_failover_detail(reason: FailoverReason, raw_error: str) -> str:
    """Format a user-friendly error detail string."""
    _MESSAGES: dict[FailoverReason, str] = {
        "auth": "Authentication failed. Token may be expired or invalid. "
                "Run: metaclaw auth paste-token --provider anthropic",
        "auth_permanent": "Authentication permanently denied. API key may be revoked or "
                          "account deactivated.",
        "billing": "Billing error. Insufficient credits or payment required.",
        "rate_limit": "Rate limited. Too many requests or quota exceeded. Retrying shortly.",
        "overloaded": "Provider overloaded. High demand — please retry.",
        "timeout": "Request timed out or network error.",
        "format": "Invalid request format.",
        "model_not_found": "Model not found or not available.",
        "session_expired": "CLI session expired. A new session will be started automatically.",
        "unknown": f"Claude CLI error: {raw_error[:300]}",
    }
    return _MESSAGES.get(reason, raw_error[:300])
