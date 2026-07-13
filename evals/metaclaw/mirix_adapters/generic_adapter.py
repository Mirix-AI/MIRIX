"""MirixGenericMemoryAdapter — drop-in SkillEvolver duck-type that drives MIRIX
through the production memory path.

The vendored MetaClaw proxy calls a duck-typed ``SkillEvolver.distill_round(...)``
at every graded round (api_server.py:1182). This adapter implements the SAME
public surface so it is a clean drop-in, but inside it:

  1. INGESTS each MetaClaw turn via ``POST /memory/add_sync`` (SYNC — messages are
     persisted before the call returns, so MIRIX's own automatic trigger can see them).
     Each turn is ONE MIRIX session: ``session_id = f"{day}-r{round_index}"``.
     The body carries ONLY agent-visible content (query → user, answer →
     assistant); NEVER any grade / oracle field (leakage discipline, mirrors the
     ``DISTILL_FORBIDDEN_KEYS`` guard).
  2. Does NOT call any evolution endpoint. Procedural evolution is owned by the
     production MIRIX trigger chain: the meta agent's ``trigger_memory_update``
     checks ``SKILL_TRIGGER_SESSION_THRESHOLD`` and schedules the procedural
     auto-dream internally.
  3. RETRIEVAL is delegated to :class:`MirixSkillsAdapter` (wired separately by
     ``METACLAW_SKILLS_PROVIDER=mirix``) — this adapter performs NO retrieval, so
     retrieval stays on the unified ``/memory/search`` interface.
"""

from __future__ import annotations

import logging
import json
from typing import Any, Dict, List, Optional

import httpx

import asyncio

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 1800.0
DEFAULT_CLIENT_ID = "client-00000000-0000-4000-8000-000000000000"
DEFAULT_EXPECTED_TRIGGER_SESSIONS = 5

ADD_SYNC_ENDPOINT_PATH = "/memory/add_sync"
AGENTS_ENDPOINT_PATH = "/agents"
HEALTH_ENDPOINT_PATH = "/health"
SEARCH_ENDPOINT_PATH = "/memory/search"

# Treat thresholds at/above this sentinel as effectively disabled. A production
# MIRIX eval should leave the in-band trigger enabled; it must not rely on this
# adapter to call an explicit dream endpoint.
IN_BAND_TRIGGER_DISABLED_MIN = 1_000_000

DEFAULT_RETRY_SLEEP_S = 2.0
DEFAULT_MAX_HTTP_RETRIES = 4  # initial + 4 retries = 5 attempts
DEFAULT_RETRY_BACKOFF_MAX_S = 30.0
TRANSCRIPT_CONTENT_MAX_CHARS = 60_000

META_AGENT_TYPE = "meta_memory_agent"
EVOLVE_FAILURE_MARKER = "EVOLVE_FAILURE"

# Keys that, if they ever appeared in an outgoing eval payload, would be an
# oracle leak. The add_sync body is built field-by-field from query/answer only,
# so these can never appear unless a future change widens the interface.
DISTILL_FORBIDDEN_KEYS = (
    "inline_score",
    "reward",
    "eval",
    "answer",
    "command",
    "expect_exit",
    "options",
    "passed",
    "feedback",
    "correct",
    "incorrect",
    "score",
    "selected",
    "format_valid",
)

_TRANSCRIPT_ALLOWED_ROLES = {"user", "assistant", "tool"}


def _json_compact(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    except TypeError:
        return str(value)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
                else:
                    parts.append(_json_compact(item))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return _json_compact(content)


def _clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    marker = "\n[... transcript truncated ...]\n"
    if max_chars <= len(marker) + 16:
        return text[:max_chars]
    head_chars = max(1, (max_chars - len(marker)) // 3)
    tail_chars = max_chars - len(marker) - head_chars
    return text[:head_chars] + marker + text[-tail_chars:]


def _render_tool_calls(tool_calls: Any) -> str:
    if not isinstance(tool_calls, list) or not tool_calls:
        return ""
    safe_calls: list[dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        safe: dict[str, Any] = {}
        for key in ("id", "type", "name", "tool_name"):
            if key in call:
                safe[key] = call[key]
        function = call.get("function")
        if isinstance(function, dict):
            safe_function: dict[str, Any] = {}
            for key in ("name", "arguments"):
                if key in function:
                    safe_function[key] = function[key]
            if safe_function:
                safe["function"] = safe_function
        elif "arguments" in call:
            safe["arguments"] = call["arguments"]
        if safe:
            safe_calls.append(safe)
    return _json_compact(safe_calls) if safe_calls else ""


def _render_visible_transcript(transcript: Any) -> str:
    """Render agent-visible OpenClaw messages without forwarding oracle fields.

    Accepted inputs are the sidecar proxy capture
    ``{"messages": [...], "assistant": {...}}`` or a benchmark llm_log dict with
    a ``messages`` list. Only user/assistant/tool message content and assistant
    tool_calls are rendered. All surrounding result/scoring fields are ignored.
    """
    if transcript is None:
        return ""
    if isinstance(transcript, list):
        messages = transcript
        assistant_msg = None
    elif isinstance(transcript, dict):
        messages = transcript.get("messages")
        assistant_msg = transcript.get("assistant") or transcript.get("assistant_message")
    else:
        return ""
    if not isinstance(messages, list) or not messages:
        return ""

    last_user_idx: int | None = None
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user_idx = idx
            break
    visible_messages = messages[last_user_idx:] if last_user_idx is not None else messages[-12:]

    chunks: list[str] = []
    for raw_msg in visible_messages:
        if not isinstance(raw_msg, dict):
            continue
        role = str(raw_msg.get("role", "")).lower()
        if role not in _TRANSCRIPT_ALLOWED_ROLES:
            continue
        if role == "tool":
            label = "tool result"
            name = raw_msg.get("name") or raw_msg.get("toolName")
            call_id = raw_msg.get("tool_call_id") or raw_msg.get("toolCallId")
            if name:
                label += f" {name}"
            if call_id:
                label += f" ({call_id})"
        else:
            label = role
        content = _content_to_text(raw_msg.get("content")).strip()
        if content:
            chunks.append(f"[{label}]\n{content}")
        tool_calls = _render_tool_calls(raw_msg.get("tool_calls"))
        if tool_calls:
            chunks.append(f"[assistant tool_calls]\n{tool_calls}")

    if isinstance(assistant_msg, dict):
        tool_calls = _render_tool_calls(assistant_msg.get("tool_calls"))
        if tool_calls:
            chunks.append(f"[assistant response tool_calls]\n{tool_calls}")

    return "\n\n".join(chunks).strip()


def _compose_user_content(query: str, transcript: Any = None) -> str:
    query_text = query or ""
    transcript_text = _render_visible_transcript(transcript)
    if not transcript_text:
        return query_text

    prefix = (
        f"{query_text}\n\n"
        "[MetaClaw agent-visible transcript]\n"
        "The following messages/tool results were visible to the benchmark agent "
        "during this same round:\n"
    )
    budget = TRANSCRIPT_CONTENT_MAX_CHARS - len(prefix)
    if budget <= 0:
        return _clip_text(query_text, TRANSCRIPT_CONTENT_MAX_CHARS)
    return prefix + _clip_text(transcript_text, budget)


def build_add_sync_payload(
    *,
    meta_agent_id: str,
    user_id: str,
    query: str,
    answer: str,
    session_id: str,
    transcript: Any = None,
) -> Dict[str, Any]:
    """Build the EXACT body POSTed to ``/memory/add_sync`` for one MetaClaw turn.

    Leakage discipline (assert on the RETURN VALUE in tests): the body carries
    ONLY agent-visible content — ``query`` plus an optional sanitized OpenClaw
    tool transcript → user, ``answer`` → assistant — plus the routing fields
    (``meta_agent_id``, ``user_id``, ``session_id``). NO ``eval`` /
    ``answer``-oracle / ``inline_score`` / ``reward`` / ``feedback`` /
    ``score`` / ``passed`` / ``options`` field is ever read here, so none of
    :data:`DISTILL_FORBIDDEN_KEYS` can appear as outgoing structured keys.

    ``filter_tags`` is deliberately NOT set — the server mirrors ``session_id``
    into ``filter_tags`` itself, and setting both would trip the
    session_id-agreement model validator (rest_api.py:2062-2073).
    """
    return {
        "meta_agent_id": meta_agent_id,
        "user_id": user_id,
        "session_id": session_id,
        "chaining": True,
        "use_cache": True,
        "messages": [
            {"role": "user", "content": _compose_user_content(query, transcript)},
            {"role": "assistant", "content": answer or ""},
        ],
    }


def _payload_reports_failure(payload: Any) -> bool:
    """True iff a 2xx response body explicitly reports a semantic failure."""
    if not isinstance(payload, dict):
        return False
    if payload.get("success") is False:
        return True
    if payload.get("ok") is False:
        return True
    return False


class MirixGenericMemoryAdapter:
    """Drop-in :class:`SkillEvolver` duck-type backed by MIRIX's production memory
    path (``/memory/add_sync`` plus MIRIX's automatic procedural trigger).

    Public surface mirrors the minimal SkillEvolver methods the proxy expects:

      * ``distill_round(*, day, round_id, round_index, query, answer,
        session_id=None, session_done=False) -> dict``  (async) — the method the
        proxy calls, implemented as a production memory ingest.
      * ``evolve(failed_samples, current_skills) -> []``  (async) — no-op; this
        arm is never driven through the raw-transcript path.
      * ``should_evolve(batch, threshold=0.0) -> True``
      * ``.update_history`` / ``.history_path`` — paper-compat attrs.
      * ``close()``.
    """

    def __init__(
        self,
        base_url: str,
        user_id: str,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        client_id: str = DEFAULT_CLIENT_ID,
        *,
        transport: Optional[httpx.BaseTransport] = None,
        retry_sleep_s: float = DEFAULT_RETRY_SLEEP_S,
        max_http_retries: int = DEFAULT_MAX_HTTP_RETRIES,
        retry_backoff_max_s: float = DEFAULT_RETRY_BACKOFF_MAX_S,
        evolve_every_n_turns: int = DEFAULT_EXPECTED_TRIGGER_SESSIONS,
        # Compatibility alias for older launcher config. The value is
        # informational only now; the server's SKILL_TRIGGER_SESSION_THRESHOLD
        # owns the real production cadence.
        evolve_every_n_rounds: Optional[int] = None,
        # Paper-launcher kwargs accepted-and-ignored for signature compat.
        max_new_skills: Optional[int] = None,  # noqa: ARG002
        history_path: Optional[str] = None,  # noqa: ARG002
        **_paper_kwargs: Any,  # noqa: ARG002
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_id = user_id
        self.client_id = client_id
        self.retry_sleep_s = retry_sleep_s
        self.max_http_retries = max(0, max_http_retries)
        self.retry_backoff_max_s = retry_backoff_max_s

        # Paper-compat attrs.
        self.update_history: List[dict] = []
        self.history_path: Optional[str] = None

        # Informational only: production cadence lives on the MIRIX server.
        effective_every_n = (
            evolve_every_n_rounds if evolve_every_n_rounds is not None else evolve_every_n_turns
        )
        self.expected_trigger_sessions = max(1, int(effective_every_n))

        # Telemetry / health gate.
        self.evolve_failures = 0
        self.turns_ingested = 0

        # Lazily-resolved meta agent id (the agent /memory/add_sync writes to,
        # matching the server-side owner the automatic trigger resolves).
        self._meta_agent_id: Optional[str] = None

        client_kwargs: Dict[str, Any] = {
            "base_url": self.base_url,
            "timeout": timeout_s,
            "headers": {"X-Client-Id": self.client_id},
        }
        if transport is not None:
            client_kwargs["transport"] = transport
        self._http = httpx.Client(**client_kwargs)

        # Preflight: refuse to construct if the production memory endpoint is not
        # exposed, or if the server's in-band procedural trigger is disabled.
        self._verify_endpoints_or_raise()
        self._verify_server_config_or_raise()

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:  # pragma: no cover — defensive
            pass

    # ------------------------------------------------------------------ #
    # Preflight                                                           #
    # ------------------------------------------------------------------ #

    def _verify_endpoints_or_raise(self) -> None:
        """GET ``/openapi.json`` and require ``/memory/add_sync`` to be registered.
        Raise loudly if absent.

        On HTTP / parse error we WARN but do NOT raise — we'd rather attempt the
        real calls and surface the actual server error than fail startup on an
        openapi-spec hiccup.
        """
        try:
            resp = self._http.get("/openapi.json")
            resp.raise_for_status()
            spec = resp.json()
            paths = spec.get("paths") if isinstance(spec, dict) else None
            if not isinstance(paths, dict):
                logger.warning(
                    "[MirixGenericMemoryAdapter] /openapi.json has no 'paths' "
                    "object; skipping preflight check"
                )
                return
            missing = [p for p in (ADD_SYNC_ENDPOINT_PATH,) if p not in paths]
            if missing:
                raise RuntimeError(
                    f"MIRIX server at {self.base_url} lacks {missing}; switch to a "
                    f"branch exposing the production memory path or merge it in."
                )
            logger.info(
                "[MirixGenericMemoryAdapter] preflight ok: %s exposes %s",
                self.base_url,
                ADD_SYNC_ENDPOINT_PATH,
            )
        except RuntimeError:
            raise
        except (httpx.HTTPError, ValueError) as e:
            logger.warning(
                "[MirixGenericMemoryAdapter] preflight check skipped (%s); will "
                "attempt memory calls anyway",
                e,
            )

    def _verify_server_config_or_raise(self) -> None:
        """GET ``/health`` and validate the server's skill-trigger config.

        REFUSE to run if the in-band procedural trigger is effectively disabled.
        The production eval path submits turns to ``/memory/add_sync`` and lets
        MIRIX's own meta-agent trigger schedule procedural auto-dream; this
        adapter must not compensate by calling an evolution endpoint.

        If the server is too old to report the field, WARN rather than blocking.
        """
        try:
            resp = self._http.get(HEALTH_ENDPOINT_PATH)
            resp.raise_for_status()
            body = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            logger.warning(
                "[MirixGenericMemoryAdapter] in-band-trigger preflight skipped "
                "(%s); ensure the server was started with an enabled "
                "SKILL_TRIGGER_SESSION_THRESHOLD.",
                e,
            )
            return
        threshold = body.get("skill_trigger_session_threshold") if isinstance(body, dict) else None
        if threshold is None:
            logger.warning(
                "[MirixGenericMemoryAdapter] server /health does not report "
                "skill_trigger_session_threshold; cannot verify the in-band "
                "trigger is enabled.",
            )
            return
        if int(threshold) >= IN_BAND_TRIGGER_DISABLED_MIN:
            raise RuntimeError(
                f"MIRIX server at {self.base_url} has SKILL_TRIGGER_SESSION_THRESHOLD"
                f"={threshold} (in-band procedural trigger effectively DISABLED). "
                f"The mirix-generic arm now follows the production path and only "
                f"submits /memory/add_sync; restart the server with a normal trigger "
                f"threshold, e.g.:\n  SKILL_TRIGGER_SESSION_THRESHOLD={DEFAULT_EXPECTED_TRIGGER_SESSIONS} "
                f"python scripts/start_server.py --port <port>"
            )
        logger.info(
            "[MirixGenericMemoryAdapter] in-band trigger enabled "
            "(skill_trigger_session_threshold=%s); expected eval cadence=%s",
            threshold,
            self.expected_trigger_sessions,
        )

    # ------------------------------------------------------------------ #
    # Public interface (paper SkillEvolver duck-typed surface)            #
    # ------------------------------------------------------------------ #

    def should_evolve(self, batch: list, threshold: float = 0.0) -> bool:  # noqa: ARG002
        """Always True (paper-compat; the proxy calls ``distill_round`` directly)."""
        return True

    async def evolve(
        self,
        failed_samples: list,  # noqa: ARG002
        current_skills: Dict[str, Any],  # noqa: ARG002
    ) -> List[dict]:
        """Raw-transcript path: no-op. The generic arm is driven exclusively
        through ``distill_round``; returning ``[]`` keeps paper's downstream
        ``add_skills`` a no-op (MIRIX is the sole writer)."""
        return []

    # ------------------------------------------------------------------ #
    # meta_agent_id resolution                                            #
    # ------------------------------------------------------------------ #

    def _resolve_meta_agent_id(self) -> str:
        """GET ``/agents`` and pick the single ``meta_memory_agent`` row's id.

        Cached on first success. Raise loudly if no meta agent exists — the
        runner's mirix prelude ensures one, so its absence is a real
        misconfiguration.
        """
        if self._meta_agent_id is not None:
            return self._meta_agent_id
        resp = self._http.get(AGENTS_ENDPOINT_PATH, params={"limit": 100})
        resp.raise_for_status()
        agents = resp.json()
        if not isinstance(agents, list):
            raise RuntimeError(
                f"[MirixGenericMemoryAdapter] GET {AGENTS_ENDPOINT_PATH} returned "
                f"non-list: {type(agents)!r}"
            )
        for a in agents:
            if not isinstance(a, dict):
                continue
            if a.get("agent_type") == META_AGENT_TYPE or a.get("name") == META_AGENT_TYPE:
                aid = a.get("id")
                if aid:
                    self._meta_agent_id = aid
                    logger.info(
                        "[MirixGenericMemoryAdapter] resolved meta agent id=%s", aid
                    )
                    return aid
        raise RuntimeError(
            f"[MirixGenericMemoryAdapter] no '{META_AGENT_TYPE}' found at "
            f"{self.base_url}{AGENTS_ENDPOINT_PATH}; the runner's mirix prelude "
            f"must initialize one before this arm runs."
        )

    # ------------------------------------------------------------------ #
    # The method the proxy calls                                          #
    # ------------------------------------------------------------------ #

    async def distill_round(
        self,
        *,
        day: str,
        round_id: str,
        round_index: int,
        query: str,
        answer: str,
        transcript: Any = None,
        session_id: Optional[str] = None,  # noqa: ARG002 — overridden per-turn below
        session_done: bool = False,
    ) -> Dict[str, Any]:
        """Ingest ONE MetaClaw turn into MIRIX's production memory endpoint.

        PER-TURN session_id: ``f"{day}-r{round_index}"`` (1 MetaClaw turn = 1
        MIRIX session). The bench's incoming ``session_id`` (a whole-day session)
        is deliberately IGNORED for the add_sync body so MIRIX's conversation
        store sees one sealed session per visible turn.

        ``session_done`` is an end-of-session marker from the MetaClaw bench. It
        carries no graded query/answer, so this adapter forwards NOTHING to
        MIRIX. Production procedural evolution is triggered inside MIRIX by the
        meta agent after enough real session_ids have accumulated.
        """
        if session_done:
            return {
                "ok": True,
                "evolved": False,
                "flush": True,
                "turns_ingested": self.turns_ingested,
            }

        # Per-turn session id (1 turn = 1 MIRIX session). Ignore the incoming
        # day-level session_id for the add_sync session_id field.
        per_turn_sid = f"{day}-r{round_index}"
        try:
            # Run the (sync, cached-after-first) GET /agents off the event loop so
            # the first turn does not block the proxy loop (codex P2 2026-06-23).
            meta_agent_id = await asyncio.to_thread(self._resolve_meta_agent_id)
        except Exception as e:  # noqa: BLE001 — surface loudly, keep the bench alive
            self.evolve_failures += 1
            logger.error(
                "[MirixGenericMemoryAdapter] %s could not resolve meta agent id "
                "for %s/%s: %r (evolve_failures=%d) — turn NOT ingested",
                EVOLVE_FAILURE_MARKER,
                day,
                round_id,
                e,
                self.evolve_failures,
            )
            return {"ok": False, "evolved": False}

        body = build_add_sync_payload(
            meta_agent_id=meta_agent_id,
            user_id=self.user_id,
            query=query,
            answer=answer,
            transcript=transcript,
            session_id=per_turn_sid,
        )
        result: Dict[str, Any] = {"ok": True, "evolved": False, "session_id": per_turn_sid}
        try:
            ingested = await self._post_with_retry(ADD_SYNC_ENDPOINT_PATH, body)
            result["ingest"] = ingested
            if _payload_reports_failure(ingested):
                self.evolve_failures += 1
                logger.error(
                    "[MirixGenericMemoryAdapter] %s add_sync %s/%s -> 2xx but body "
                    "reports failure: %r (evolve_failures=%d) — turn NOT persisted",
                    EVOLVE_FAILURE_MARKER,
                    day,
                    round_id,
                    ingested,
                    self.evolve_failures,
                )
                return {"ok": False, "evolved": False}
        except httpx.HTTPError as e:
            self.evolve_failures += 1
            logger.error(
                "[MirixGenericMemoryAdapter] %s add_sync %s/%s -> HTTP error: %r "
                "(evolve_failures=%d) — turn NOT persisted",
                EVOLVE_FAILURE_MARKER,
                day,
                round_id,
                e,
                self.evolve_failures,
            )
            return {"ok": False, "evolved": False}
        except Exception as e:  # noqa: BLE001 — surface any failure LOUDLY
            self.evolve_failures += 1
            logger.error(
                "[MirixGenericMemoryAdapter] %s add_sync %s/%s -> unexpected "
                "error: %r (evolve_failures=%d) — turn NOT persisted",
                EVOLVE_FAILURE_MARKER,
                day,
                round_id,
                e,
                self.evolve_failures,
                exc_info=True,
            )
            return {"ok": False, "evolved": False}

        self.turns_ingested += 1
        result["turns_ingested"] = self.turns_ingested
        return result

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    async def _post_with_retry(
        self,
        path: str,
        body: Dict[str, Any],
        *,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """POST ``path`` with retries on transient 5xx, run off the event loop.

        The sync ``httpx.Client.post`` runs via ``asyncio.to_thread`` so a
        multi-second server-side extraction does not block the proxy event loop.
        """

        def _do_post() -> httpx.Response:
            return self._http.post(path, json=body, params=params)

        attempts = self.max_http_retries + 1
        resp: Optional[httpx.Response] = None
        for attempt in range(attempts):
            try:
                resp = await asyncio.to_thread(_do_post)
            except httpx.HTTPError as exc:
                if attempt >= attempts - 1:
                    raise
                backoff = min(self.retry_sleep_s * (2**attempt), self.retry_backoff_max_s)
                logger.warning(
                    "[MirixGenericMemoryAdapter] %s -> transport error %s, retry "
                    "%d/%d in %.1fs",
                    path,
                    exc,
                    attempt + 1,
                    attempts - 1,
                    backoff,
                )
                await asyncio.sleep(backoff)
                continue
            if 500 <= resp.status_code < 600 and attempt < attempts - 1:
                backoff = min(self.retry_sleep_s * (2**attempt), self.retry_backoff_max_s)
                logger.warning(
                    "[MirixGenericMemoryAdapter] %s -> HTTP %d, retry %d/%d in %.1fs",
                    path,
                    resp.status_code,
                    attempt + 1,
                    attempts - 1,
                    backoff,
                )
                await asyncio.sleep(backoff)
                continue
            break
        assert resp is not None  # attempts >= 1 always binds resp
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            logger.warning(
                "[MirixGenericMemoryAdapter] %s response not JSON: %r",
                path,
                resp.text[:200],
            )
            return {}

__all__ = [
    "MirixGenericMemoryAdapter",
    "ADD_SYNC_ENDPOINT_PATH",
    "AGENTS_ENDPOINT_PATH",
    "HEALTH_ENDPOINT_PATH",
    "SEARCH_ENDPOINT_PATH",
    "IN_BAND_TRIGGER_DISABLED_MIN",
    "DEFAULT_EXPECTED_TRIGGER_SESSIONS",
    "DISTILL_FORBIDDEN_KEYS",
    "EVOLVE_FAILURE_MARKER",
    "build_add_sync_payload",
]
