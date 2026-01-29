"""
Utilities for issuing and validating Mirix API keys.
"""

import hashlib
import hmac
import secrets
from typing import Optional


def generate_api_key() -> str:
    """Generate a random API key string."""
    return secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    """Return a SHA-256 hex digest for the given API key."""
    if not api_key:
        raise ValueError("api_key is required to generate hash")
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def verify_api_key(api_key: str, api_key_hash: Optional[str]) -> bool:
    """Constant-time check that api_key matches the stored hash."""
    if not api_key or not api_key_hash:
        return False
    candidate = hash_api_key(api_key)
    return hmac.compare_digest(candidate, api_key_hash)
