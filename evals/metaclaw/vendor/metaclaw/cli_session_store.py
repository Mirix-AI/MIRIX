"""
CLI session store for MetaClaw.

Tracks Claude CLI session IDs so multi-turn conversations can resume
via `claude -p --resume <sessionId>` instead of replaying the full
message history each time.

Mirrors OpenClaw's cli-session.ts session binding approach:
- Each MetaClaw session (X-Session-Id) maps to a Claude CLI session UUID
- Sessions are invalidated when system prompt or auth profile changes
- Persisted to ~/.metaclaw/cli-sessions.json for restart resilience

Session flow (matching OpenClaw):
1. First request in a session → new UUID, pass --session-id <uuid>
2. Subsequent requests → --resume <uuid> (CLI replays its own history)
3. System prompt change → invalidate, start fresh session
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CLI_SESSIONS_FILE = Path.home() / ".metaclaw" / "cli-sessions.json"

# Claude CLI JSONL output fields that may contain the session ID
SESSION_ID_FIELDS = ("session_id", "sessionId", "conversation_id", "conversationId")


def _hash_text(text: str | None) -> str | None:
    """SHA-256 hash of a string, or None if empty."""
    if not text or not text.strip():
        return None
    return hashlib.sha256(text.strip().encode()).hexdigest()


@dataclass
class CliSessionBinding:
    """Binds a MetaClaw session to a Claude CLI session."""

    cli_session_id: str              # UUID passed to --session-id / --resume
    auth_profile_id: str | None = None
    system_prompt_hash: str | None = None
    created_at: float = 0.0


class CliSessionStore:
    """Manages Claude CLI session bindings keyed by MetaClaw session ID."""

    def __init__(self, path: Path | str | None = None):
        self._path = Path(path) if path else CLI_SESSIONS_FILE
        # metaclaw_session_id → CliSessionBinding
        self._bindings: dict[str, CliSessionBinding] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for sid, entry in raw.get("bindings", {}).items():
                self._bindings[sid] = CliSessionBinding(
                    cli_session_id=entry["cli_session_id"],
                    auth_profile_id=entry.get("auth_profile_id"),
                    system_prompt_hash=entry.get("system_prompt_hash"),
                    created_at=entry.get("created_at", 0.0),
                )
        except Exception as e:
            logger.warning("[CliSessionStore] failed to load %s: %s", self._path, e)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "bindings": {
                sid: asdict(binding) for sid, binding in self._bindings.items()
            }
        }
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ── session resolution (mirrors OpenClaw's resolveCliSessionReuse) ───

    def resolve_session(
        self,
        metaclaw_session_id: str,
        auth_profile_id: str | None = None,
        system_prompt: str | None = None,
    ) -> tuple[str, bool]:
        """Resolve the CLI session ID for a given MetaClaw session.

        Returns (cli_session_id, is_new).
        If the session is invalidated (auth or system prompt changed),
        a new session is started.
        """
        current_sp_hash = _hash_text(system_prompt)
        existing = self._bindings.get(metaclaw_session_id)

        if existing:
            # Check if session is still valid
            invalidated = self._check_invalidation(
                existing, auth_profile_id, current_sp_hash
            )
            if not invalidated:
                return existing.cli_session_id, False
            else:
                logger.info(
                    "[CliSessionStore] session %s invalidated: %s",
                    metaclaw_session_id,
                    invalidated,
                )

        # Create new CLI session
        import time

        cli_sid = str(uuid.uuid4())
        self._bindings[metaclaw_session_id] = CliSessionBinding(
            cli_session_id=cli_sid,
            auth_profile_id=auth_profile_id,
            system_prompt_hash=current_sp_hash,
            created_at=time.time(),
        )
        self.save()
        return cli_sid, True

    def _check_invalidation(
        self,
        binding: CliSessionBinding,
        auth_profile_id: str | None,
        system_prompt_hash: str | None,
    ) -> str | None:
        """Check if a session binding should be invalidated.

        Returns the reason string, or None if still valid.
        """
        if binding.auth_profile_id != auth_profile_id:
            return "auth-profile"
        if binding.system_prompt_hash != system_prompt_hash:
            return "system-prompt"
        return None

    def update_cli_session_id(
        self,
        metaclaw_session_id: str,
        cli_session_id: str,
    ) -> None:
        """Update the CLI session ID (e.g. from CLI output)."""
        existing = self._bindings.get(metaclaw_session_id)
        if existing:
            existing.cli_session_id = cli_session_id
            self.save()

    def clear_session(self, metaclaw_session_id: str) -> None:
        """Remove the CLI session binding for a MetaClaw session."""
        if metaclaw_session_id in self._bindings:
            del self._bindings[metaclaw_session_id]
            self.save()

    def clear_all(self) -> None:
        """Remove all CLI session bindings."""
        self._bindings.clear()
        self.save()

    # ── helpers ───────────────────────────────────────────────────

    def extract_session_id_from_jsonl(self, stdout: str) -> str | None:
        """Extract the CLI session ID from stream-json JSONL output.

        Mirrors OpenClaw's pickCliSessionId — checks known field names
        in each JSON line.
        """
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                for field_name in SESSION_ID_FIELDS:
                    val = obj.get(field_name)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
                # Also check thread_id (used by some CLI versions)
                tid = obj.get("thread_id")
                if isinstance(tid, str) and tid.strip():
                    return tid.strip()
            except json.JSONDecodeError:
                continue
        return None
