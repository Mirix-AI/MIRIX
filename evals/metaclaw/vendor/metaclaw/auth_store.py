"""
Auth profile store for MetaClaw.

Manages OAuth tokens and API keys for model providers.
Storage: ~/.metaclaw/auth-profiles.json

Inspired by OpenClaw's auth profile rotation system.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

AUTH_PROFILES_FILE = Path.home() / ".metaclaw" / "auth-profiles.json"


@dataclass
class AuthProfile:
    """A single authentication profile for a model provider."""

    provider: str                    # e.g. "anthropic", "openai"
    profile_id: str                  # e.g. "anthropic:manual", "anthropic:user@example.com"
    method: str                      # "token" | "api_key"

    # --- Token fields (method=token) ---
    access_token: str = ""
    refresh_token: str = ""
    expires_at: str = ""             # ISO 8601 timestamp

    # --- API key field (method=api_key) ---
    api_key: str = ""

    # --- Status tracking ---
    cooldown_until: float = 0.0      # unix timestamp; 0 = not in cooldown
    error_count: int = 0
    disabled_until: float = 0.0
    disabled_reason: str = ""
    last_used: float = 0.0

    @property
    def is_token(self) -> bool:
        return self.method == "token"

    @property
    def is_api_key(self) -> bool:
        return self.method == "api_key"

    @property
    def is_available(self) -> bool:
        """True if not in cooldown or disabled."""
        now = time.time()
        if self.disabled_until and now < self.disabled_until:
            return False
        if self.cooldown_until and now < self.cooldown_until:
            return False
        return True

    @property
    def is_expired(self) -> bool:
        """Check if the token has expired (only for token method)."""
        if not self.is_token or not self.expires_at:
            return False
        try:
            from datetime import datetime, timezone
            exp = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            return datetime.now(timezone.utc) > exp
        except (ValueError, TypeError):
            return False

    @property
    def credential(self) -> str:
        """Return the usable credential string."""
        if self.is_token:
            return self.access_token
        return self.api_key

    def mark_used(self) -> None:
        self.last_used = time.time()

    def mark_error(self, billing: bool = False) -> None:
        """Apply cooldown after an error."""
        self.error_count += 1
        if billing:
            backoff = min(5 * 3600 * (2 ** (self.error_count - 1)), 24 * 3600)
            self.disabled_until = time.time() + backoff
            self.disabled_reason = "billing"
        else:
            # Exponential backoff: 60s -> 300s -> 1500s -> 3600s (cap)
            backoff = min(60 * (5 ** (self.error_count - 1)), 3600)
            self.cooldown_until = time.time() + backoff

    def reset_errors(self) -> None:
        self.error_count = 0
        self.cooldown_until = 0.0
        self.disabled_until = 0.0
        self.disabled_reason = ""


class AuthStore:
    """Read/write ~/.metaclaw/auth-profiles.json."""

    def __init__(self, path: Path | str = AUTH_PROFILES_FILE):
        self.path = Path(path) if isinstance(path, str) else path
        self._profiles: list[AuthProfile] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def load(self) -> None:
        """Load profiles from disk."""
        self._profiles = []
        self._loaded = True
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for item in data.get("profiles", []):
                self._profiles.append(AuthProfile(**{
                    k: v for k, v in item.items()
                    if k in AuthProfile.__dataclass_fields__
                }))
            logger.debug("[AuthStore] Loaded %d profiles from %s", len(self._profiles), self.path)
        except Exception as e:
            logger.warning("[AuthStore] Failed to read %s: %s", self.path, e)

    def save(self) -> None:
        """Persist profiles to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"profiles": [asdict(p) for p in self._profiles]}
        self.path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.debug("[AuthStore] Saved %d profiles to %s", len(self._profiles), self.path)

    @property
    def profiles(self) -> list[AuthProfile]:
        self._ensure_loaded()
        return list(self._profiles)

    def get_profiles_for_provider(self, provider: str) -> list[AuthProfile]:
        """Return all profiles for a given provider, sorted by availability."""
        self._ensure_loaded()
        matching = [p for p in self._profiles if p.provider == provider]
        # Sort: available first, then tokens before api_keys, then oldest-used first
        return sorted(matching, key=lambda p: (
            not p.is_available,
            not p.is_token,
            p.last_used or 0,
        ))

    def get_best_profile(self, provider: str) -> AuthProfile | None:
        """Return the best available profile for a provider."""
        profiles = self.get_profiles_for_provider(provider)
        for p in profiles:
            if p.is_available and not p.is_expired:
                return p
        return None

    def add_or_update(self, profile: AuthProfile) -> None:
        """Add a new profile or update existing one by profile_id."""
        self._ensure_loaded()
        for i, existing in enumerate(self._profiles):
            if existing.profile_id == profile.profile_id:
                self._profiles[i] = profile
                self.save()
                return
        self._profiles.append(profile)
        self.save()

    def remove(self, profile_id: str) -> bool:
        """Remove a profile by ID."""
        self._ensure_loaded()
        before = len(self._profiles)
        self._profiles = [p for p in self._profiles if p.profile_id != profile_id]
        if len(self._profiles) < before:
            self.save()
            return True
        return False

    # ------------------------------------------------------------------ #
    # Convenience: paste an OAuth token (Claude Code / Codex / Gemini)   #
    # ------------------------------------------------------------------ #

    # Maps provider → env var name used by the corresponding CLI
    OAUTH_ENV_VARS: dict[str, str] = {
        "anthropic": "CLAUDE_CODE_OAUTH_TOKEN",
        "openai-codex": "CODEX_OAUTH_TOKEN",
        "gemini": "GEMINI_OAUTH_TOKEN",
    }

    # Maps provider → CLI binary name
    CLI_BINARIES: dict[str, str] = {
        "anthropic": "claude",
        "openai-codex": "codex",
        "gemini": "gemini",
    }

    def paste_oauth_token(self, provider: str, token_input: str) -> AuthProfile:
        """Parse an OAuth token JSON string and store it.

        Works for any CLI-backed provider (anthropic, openai-codex, gemini).

        Accepts either:
          - Raw JSON: {"accessToken":"...", "refreshToken":"...", "expiresAt":"..."}
          - Just the access token string
        """
        token_input = token_input.strip()

        if token_input.startswith("{"):
            # Full JSON format
            try:
                data = json.loads(token_input)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid token JSON: {e}") from e

            # Support both camelCase (Claude) and snake_case (Codex/Gemini)
            access = data.get("accessToken", "") or data.get("access_token", "")
            refresh = data.get("refreshToken", "") or data.get("refresh_token", "")
            expires = data.get("expiresAt", "") or data.get("expires_at", "")

            if not access:
                raise ValueError("Token JSON missing 'accessToken'/'access_token' field")
        else:
            # Raw token string
            access = token_input
            refresh = ""
            expires = ""

        profile = AuthProfile(
            provider=provider,
            profile_id=f"{provider}:manual",
            method="token",
            access_token=access,
            refresh_token=refresh,
            expires_at=str(expires) if expires else "",
        )

        self.add_or_update(profile)
        logger.info(
            "[AuthStore] Stored %s token profile: %s (expires: %s)",
            provider,
            profile.profile_id,
            expires or "static",
        )
        return profile

    def paste_anthropic_token(self, token_json: str) -> AuthProfile:
        """Backward compat: paste a CLAUDE_CODE_OAUTH_TOKEN."""
        return self.paste_oauth_token("anthropic", token_json)

    def paste_api_key(self, provider: str, api_key: str, profile_id: str = "") -> AuthProfile:
        """Store a plain API key for any provider."""
        if not profile_id:
            profile_id = f"{provider}:default"

        profile = AuthProfile(
            provider=provider,
            profile_id=profile_id,
            method="api_key",
            api_key=api_key.strip(),
        )

        self.add_or_update(profile)
        logger.info("[AuthStore] Stored API key profile: %s", profile.profile_id)
        return profile

    def describe(self) -> str:
        """Return a human-readable summary of all profiles."""
        self._ensure_loaded()
        if not self._profiles:
            return "No auth profiles configured."

        lines = [
            f"Auth store: {self.path}",
            f"Profiles ({len(self._profiles)}):",
        ]
        for p in self._profiles:
            cred_preview = p.credential[:12] + "..." + p.credential[-4:] if len(p.credential) > 20 else p.credential
            status_parts = []
            if p.is_expired:
                status_parts.append("EXPIRED")
            elif not p.is_available:
                status_parts.append("COOLDOWN")
            else:
                status_parts.append("ok")
            if p.expires_at:
                status_parts.append(f"expires={p.expires_at}")
            else:
                status_parts.append("static")
            status = ", ".join(status_parts)

            lines.append(
                f"  - {p.profile_id} ({p.method}): {cred_preview} [{status}]"
            )
        return "\n".join(lines)
