"""MirixGenericMemoryAdapter — drop-in SkillEvolver duck-type that drives MIRIX
through the GENERIC PRODUCTION memory path (``POST /memory/add_sync`` +
``POST /memory/auto_dream``), a SIBLING to
:class:`evals.metaclaw.mirix_adapters.evolver_adapter.MirixEvolverAdapter`
(which uses the records distill endpoints).

The vendored MetaClaw proxy calls a duck-typed ``SkillEvolver.distill_round(...)``
at every graded round (api_server.py:1182). This adapter implements the SAME
public surface so it is a clean drop-in, but inside it:

  1. INGESTS each MetaClaw turn via ``POST /memory/add_sync`` (SYNC — messages are
     persisted before the call returns, so the evolution barrier below sees them).
     Each turn is ONE MIRIX session: ``session_id = f"{day}-r{round_index}"``.
     The body carries ONLY agent-visible content (query → user, answer →
     assistant); NEVER any grade / oracle field (leakage discipline, mirrors the
     records adapter's ``DISTILL_FORBIDDEN_KEYS`` spirit).
  2. Fires a SYNCHRONOUS BLOCKING evolution barrier every N turns (global cadence,
     reusing :func:`should_evolve_at`): ``POST /memory/auto_dream`` with
     ``{"mode":"procedural","last_n_sessions":N}``. On end-of-session / end-of-run
     flush, it evolves the REMAINDER with ``last_n_sessions = (turns since the last
     barrier)`` so already-evolved sessions are not re-distilled (respects
     MESSAGE_RETAIN_LAST_N_SESSIONS=5).
  3. RETRIEVAL is delegated to :class:`MirixSkillsAdapter` (wired separately by
     ``METACLAW_SKILLS_PROVIDER=mirix``) — this adapter performs NO retrieval, so
     both the records arm and the generic arm retrieve byte-identically.

The in-band fire-and-forget procedural-dream trigger inside MIRIX's meta agent
(``SKILL_TRIGGER_SESSION_THRESHOLD``) MUST be disabled on the SERVER process for
this arm (set it to a very large value) so that ONLY these explicit barriers drive
evolution. If it is not disabled, evolutions double-fire and our barriers may
report 0 experiences (already consumed) — the :pyattr:`degenerate_run` health gate
is the safety net that surfaces that.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx

# Reuse the shared cadence + leakage + diff-count helpers from the records
# adapter (do NOT duplicate them — both arms must agree on the cadence and the
# forbidden-key list).
from evals.metaclaw.mirix_adapters.evolver_adapter import (
    DISTILL_FORBIDDEN_KEYS,
    EVOLVE_FAILURE_MARKER,
    _payload_reports_failure,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 1800.0
DEFAULT_CLIENT_ID = "client-00000000-0000-4000-8000-000000000000"
DEFAULT_EVOLVE_EVERY_N_TURNS = 5

ADD_SYNC_ENDPOINT_PATH = "/memory/add_sync"
AUTO_DREAM_ENDPOINT_PATH = "/memory/auto_dream"
AGENTS_ENDPOINT_PATH = "/agents"
HEALTH_ENDPOINT_PATH = "/health"

# The generic arm REQUIRES the server's in-band fire-and-forget procedural
# trigger (SKILL_TRIGGER_SESSION_THRESHOLD) to be disabled: otherwise it
# double-evolves alongside our explicit barrier (and the degenerate_run gate
# only catches ZERO evolution, never DOUBLE). We treat any threshold at/above
# this sentinel as "disabled" — no real eval ingests a million sessions.
IN_BAND_TRIGGER_DISABLED_MIN = 1_000_000

DEFAULT_RETRY_SLEEP_S = 2.0
DEFAULT_MAX_HTTP_RETRIES = 4  # initial + 4 retries = 5 attempts
DEFAULT_RETRY_BACKOFF_MAX_S = 30.0

META_AGENT_TYPE = "meta_memory_agent"


def build_add_sync_payload(
    *,
    meta_agent_id: str,
    user_id: str,
    query: str,
    answer: str,
    session_id: str,
) -> Dict[str, Any]:
    """Build the EXACT body POSTed to ``/memory/add_sync`` for one MetaClaw turn.

    Leakage discipline (assert on the RETURN VALUE in tests): the body carries
    ONLY the two agent-visible messages — ``query`` → user, ``answer`` →
    assistant — plus the routing fields (``meta_agent_id``, ``user_id``,
    ``session_id``). NO ``eval`` / ``answer``-oracle / ``inline_score`` /
    ``reward`` / ``feedback`` / ``score`` / ``passed`` / ``options`` field is ever
    read here, so none of :data:`DISTILL_FORBIDDEN_KEYS` can appear.

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
            {"role": "user", "content": query or ""},
            {"role": "assistant", "content": answer or ""},
        ],
    }


class MirixGenericMemoryAdapter:
    """Drop-in :class:`SkillEvolver` duck-type backed by MIRIX's production memory
    path (``/memory/add_sync`` + ``/memory/auto_dream``).

    Public surface mirrors :class:`MirixEvolverAdapter`:

      * ``distill_round(*, day, round_id, round_index, query, answer,
        session_id=None, session_done=False) -> dict``  (async) — THE method the
        proxy calls.
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
        evolve_every_n_turns: int = DEFAULT_EVOLVE_EVERY_N_TURNS,
        # Drop-in alias for MirixEvolverAdapter's kwarg name: a caller mirroring
        # the records adapter's signature passes ``evolve_every_n_rounds`` — accept
        # it instead of silently swallowing it into ``**_paper_kwargs`` (codex P2
        # 2026-06-23).
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

        # Paper-compat attrs (parity with MirixEvolverAdapter).
        self.update_history: List[dict] = []
        self.history_path: Optional[str] = None

        # Cadence: fire a barrier once N UN-EVOLVED turns have accrued.
        # ``_turns_since_last_barrier`` is the SINGLE source of truth for both
        # "when to fire" and "how many sessions to evolve" — see distill_round.
        effective_every_n = (
            evolve_every_n_rounds if evolve_every_n_rounds is not None else evolve_every_n_turns
        )
        self.evolve_every_n_turns = max(1, int(effective_every_n))
        # Turns ingested since the last SUCCESSFUL barrier — drives both the
        # periodic-barrier trigger and the remainder flush on session_done, so
        # already-evolved sessions are never re-distilled and a failed barrier's
        # window is retried (not skipped) before retention prunes it.
        self._turns_since_last_barrier = 0

        # Telemetry / health gate.
        self.records_evolution_events = 0  # successful barriers
        self.evolve_failures = 0
        self.barriers_fired = 0
        self.total_experiences = 0
        self.total_skills_changed = 0
        self._degenerate_marker_emitted = False

        # Lazily-resolved meta agent id (the agent /memory/add_sync writes to,
        # and the agent /memory/auto_dream independently re-resolves server-side).
        self._meta_agent_id: Optional[str] = None

        client_kwargs: Dict[str, Any] = {
            "base_url": self.base_url,
            "timeout": timeout_s,
            "headers": {"X-Client-Id": self.client_id},
        }
        if transport is not None:
            client_kwargs["transport"] = transport
        self._http = httpx.Client(**client_kwargs)

        # Preflight: refuse to construct if the production memory endpoints are
        # not exposed (best signal that the server lacks the auto-dream branch),
        # or if the server's in-band procedural trigger is still enabled (which
        # would double-evolve alongside our explicit barrier).
        self._verify_endpoints_or_raise()
        self._verify_server_config_or_raise()

    def close(self) -> None:
        # Backstop health-gate read at teardown. The PRIMARY active hook is the
        # per-session `session_done` flush in distill_round (which the vendored
        # proxy DOES call); close() is not called by the proxy, so this is only a
        # safety net for callers that do teardown explicitly (codex P1
        # 2026-06-23). The marker is idempotent, so a double read is harmless.
        try:
            _ = self.degenerate_run
        except Exception:  # pragma: no cover — defensive
            pass
        try:
            self._http.close()
        except Exception:  # pragma: no cover — defensive
            pass

    # ------------------------------------------------------------------ #
    # Preflight                                                           #
    # ------------------------------------------------------------------ #

    def _verify_endpoints_or_raise(self) -> None:
        """GET ``/openapi.json`` and require both ``/memory/add_sync`` and
        ``/memory/auto_dream`` to be registered paths. Raise loudly if absent.

        On HTTP / parse error we WARN but do NOT raise — we'd rather attempt the
        real calls and surface the actual server error than fail startup on an
        openapi-spec hiccup (mirrors MirixEvolverAdapter preflight).
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
            missing = [
                p
                for p in (ADD_SYNC_ENDPOINT_PATH, AUTO_DREAM_ENDPOINT_PATH)
                if p not in paths
            ]
            if missing:
                raise RuntimeError(
                    f"MIRIX server at {self.base_url} lacks {missing}; switch to a "
                    f"branch exposing the production memory path (add_sync + "
                    f"auto_dream) or merge it in."
                )
            logger.info(
                "[MirixGenericMemoryAdapter] preflight ok: %s exposes %s + %s",
                self.base_url,
                ADD_SYNC_ENDPOINT_PATH,
                AUTO_DREAM_ENDPOINT_PATH,
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

        (1) REFUSE to run if the in-band procedural trigger
        (``SKILL_TRIGGER_SESSION_THRESHOLD``) is not disabled. The generic arm
        feeds turns through ``/memory/add_sync`` (the production path), which makes
        the meta agent's ``trigger_memory_update`` count sessions and fire a
        fire-and-forget auto-dream every N sessions. If that in-band trigger is
        live it double-evolves alongside our explicit barrier — and the
        :pyattr:`degenerate_run` gate only catches ZERO evolution, never DOUBLE —
        so we FAIL FAST here (codex P1 2026-06-23).

        (2) WARN if ``message_retain_last_n_sessions`` is not strictly greater than
        the barrier cadence: healthy runs are lossless even at equality, but a
        FAILED barrier cannot be retried losslessly because the oldest un-evolved
        session is pruned before the retry (codex P2 2026-06-23).

        If the server is too old to report a field (absent / endpoint error) the
        corresponding check WARNs (cannot verify) rather than blocking.
        """
        try:
            resp = self._http.get(HEALTH_ENDPOINT_PATH)
            resp.raise_for_status()
            body = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            logger.warning(
                "[MirixGenericMemoryAdapter] in-band-trigger preflight skipped "
                "(%s); relying on the degenerate_run health gate. Ensure the "
                "server was started with SKILL_TRIGGER_SESSION_THRESHOLD>=%d.",
                e,
                IN_BAND_TRIGGER_DISABLED_MIN,
            )
            return
        threshold = body.get("skill_trigger_session_threshold") if isinstance(body, dict) else None
        if threshold is None:
            logger.warning(
                "[MirixGenericMemoryAdapter] server /health does not report "
                "skill_trigger_session_threshold; cannot verify the in-band "
                "trigger is disabled. Ensure the server was started with "
                "SKILL_TRIGGER_SESSION_THRESHOLD>=%d.",
                IN_BAND_TRIGGER_DISABLED_MIN,
            )
            return
        if int(threshold) < IN_BAND_TRIGGER_DISABLED_MIN:
            raise RuntimeError(
                f"MIRIX server at {self.base_url} has SKILL_TRIGGER_SESSION_THRESHOLD"
                f"={threshold} (in-band procedural trigger ENABLED). The generic arm "
                f"feeds /memory/add_sync, so the in-band trigger would double-evolve "
                f"alongside the explicit barrier. Restart the server with the trigger "
                f"disabled:\n  SKILL_TRIGGER_SESSION_THRESHOLD={IN_BAND_TRIGGER_DISABLED_MIN} "
                f"python scripts/start_server.py --port <port>"
            )
        logger.info(
            "[MirixGenericMemoryAdapter] in-band trigger disabled "
            "(skill_trigger_session_threshold=%s) — only explicit barriers evolve",
            threshold,
        )
        # Retention slack: only the last ``message_retain_last_n_sessions``
        # sessions' raw messages survive. At equality with the cadence a healthy
        # run is lossless, but a FAILED barrier drops the oldest un-evolved session
        # before it can retry — warn so the operator can raise retention for
        # failure tolerance (codex P2 2026-06-23).
        retain = body.get("message_retain_last_n_sessions")
        if isinstance(retain, int) and retain <= self.evolve_every_n_turns:
            logger.warning(
                "[MirixGenericMemoryAdapter] message_retain_last_n_sessions=%d <= "
                "barrier cadence=%d: healthy runs are lossless, but a FAILED barrier "
                "will silently drop un-evolved sessions before it can retry. Set "
                "MESSAGE_RETAIN_LAST_N_SESSIONS > %d on the server for failure "
                "tolerance.",
                retain,
                self.evolve_every_n_turns,
                self.evolve_every_n_turns,
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

        Cached on first success. This mirrors exactly how ``auto_dream_handler``
        resolves the meta agent server-side, so add_sync (which carries this id)
        and auto_dream (which re-resolves it) target the same agent. Raise loudly
        if no meta agent exists — the runner's mirix prelude ensures one, so its
        absence is a real misconfiguration.
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
        session_id: Optional[str] = None,  # noqa: ARG002 — overridden per-turn below
        session_done: bool = False,
    ) -> Dict[str, Any]:
        """Ingest ONE MetaClaw turn into MIRIX, then fire the evolution barrier
        every N turns (and flush the remainder on session_done).

        PER-TURN session_id: ``f"{day}-r{round_index}"`` (1 MetaClaw turn = 1
        MIRIX session). The bench's incoming ``session_id`` (a whole-day session)
        is deliberately IGNORED for the add_sync body so the distiller enumerates
        one session per turn.

        ``session_done`` is the end-of-session FLUSH: it carries no graded
        query/answer — forward NOTHING to add_sync; instead evolve the REMAINDER
        (turns since the last barrier) so already-evolved sessions aren't
        re-distilled. It MUST NOT count as a graded turn or shift the cadence.
        """
        if session_done:
            remainder = self._turns_since_last_barrier
            if remainder > 0:
                evolved = await self._fire_barrier(last_n_sessions=remainder)
                # Only advance the watermark on a SUCCESSFUL barrier. A failed
                # flush must NOT reset the counter (so a later flush/run can retry
                # the un-evolved window) and must report ok=False so the bench's
                # ok-contract counts it as a failure and the proxy.log gate trips
                # (codex P1 2026-06-23 — barrier failures were swallowed as ok).
                if evolved:
                    self._turns_since_last_barrier = 0
                # Read the health gate at the session boundary so the
                # EVOLVE_FAILURE marker reaches proxy.log DURING the run — the
                # vendored proxy never calls close(), so deferring the gate to
                # teardown would make it dead code (codex P1 2026-06-23).
                _ = self.degenerate_run
                return {"ok": bool(evolved), "evolved": evolved, "flush": True, "remainder": remainder}
            _ = self.degenerate_run
            return {"ok": True, "evolved": False, "flush": True, "remainder": 0}

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

        self._turns_since_last_barrier += 1

        # Cadence keys off the count of UN-EVOLVED turns (the single source of
        # truth), NOT the global round_index. After a session_done flush resets
        # the counter, the next periodic barrier waits for N genuinely-new turns
        # instead of re-distilling already-flushed sessions (codex P1/P2
        # 2026-06-23 — the round_index cadence double-counted across a flush). On
        # a FAILED barrier the counter is not reset, so the window widens and the
        # next turn retries it before retention prunes the un-evolved sessions.
        if self._turns_since_last_barrier >= self.evolve_every_n_turns:
            evolved = await self._fire_barrier(
                last_n_sessions=self._turns_since_last_barrier
            )
            result["evolved"] = evolved
            result["watermark"] = round_index
            if evolved:
                # Advance the watermark only on success. A FAILED periodic barrier
                # leaves the remainder counter intact (so the next barrier widens
                # its window to retry the un-evolved turns) and surfaces ok=False
                # so the bench counts it as a failure (codex P1 2026-06-23 —
                # barrier failures were swallowed as ok:True).
                self._turns_since_last_barrier = 0
            else:
                result["ok"] = False
        return result

    # ------------------------------------------------------------------ #
    # Evolution barrier                                                   #
    # ------------------------------------------------------------------ #

    async def _fire_barrier(self, *, last_n_sessions: int) -> bool:
        """POST ``/memory/auto_dream`` (BLOCKING) for the procedural mode and
        accumulate health-gate telemetry.

        ``user_id`` is a QUERY param (rest_api.py:6629), not a body field. The
        body is ``{"mode":"procedural","last_n_sessions":<n>}``. Returns True on a
        healthy 2xx; False (and logs the EVOLVE_FAILURE marker) on any error.
        """
        body = {"mode": "procedural", "last_n_sessions": int(last_n_sessions)}
        try:
            payload = await self._post_with_retry(
                AUTO_DREAM_ENDPOINT_PATH, body, params={"user_id": self.user_id}
            )
        except httpx.HTTPError as e:
            self.evolve_failures += 1
            self.barriers_fired += 1
            logger.error(
                "[MirixGenericMemoryAdapter] %s auto_dream (last_n=%s) -> HTTP "
                "error: %r (evolve_failures=%d) — skills did NOT evolve this barrier",
                EVOLVE_FAILURE_MARKER,
                last_n_sessions,
                e,
                self.evolve_failures,
            )
            return False
        except Exception as e:  # noqa: BLE001 — surface any failure LOUDLY
            self.evolve_failures += 1
            self.barriers_fired += 1
            logger.error(
                "[MirixGenericMemoryAdapter] %s auto_dream (last_n=%s) -> "
                "unexpected error: %r (evolve_failures=%d) — skills did NOT "
                "evolve this barrier",
                EVOLVE_FAILURE_MARKER,
                last_n_sessions,
                e,
                self.evolve_failures,
                exc_info=True,
            )
            return False

        self.barriers_fired += 1
        if _payload_reports_failure(payload):
            self.evolve_failures += 1
            logger.error(
                "[MirixGenericMemoryAdapter] %s auto_dream (last_n=%s) -> 2xx but "
                "body reports failure: %r (evolve_failures=%d)",
                EVOLVE_FAILURE_MARKER,
                last_n_sessions,
                payload,
                self.evolve_failures,
            )
            return False

        experiences, skills_changed = _parse_auto_dream_counts(payload)
        self.total_experiences += experiences
        self.total_skills_changed += skills_changed
        self.records_evolution_events += 1
        if experiences == 0 and skills_changed == 0:
            logger.info(
                "[MirixGenericMemoryAdapter] auto_dream (last_n=%s) -> NO-OP "
                "barrier (0 experiences, 0 skills_changed) — event #%d",
                last_n_sessions,
                self.records_evolution_events,
            )
        else:
            logger.info(
                "[MirixGenericMemoryAdapter] auto_dream (last_n=%s) -> "
                "experiences=%d skills_changed=%d (event #%d)",
                last_n_sessions,
                experiences,
                skills_changed,
                self.records_evolution_events,
            )
        return True

    # ------------------------------------------------------------------ #
    # Health gate                                                         #
    # ------------------------------------------------------------------ #

    @property
    def degenerate_run(self) -> bool:
        """True iff at least one barrier fired but EVERY barrier no-op'd (0
        experiences AND 0 skills_changed across the whole run).

        On first True, emit ONE ``EVOLVE_FAILURE`` marker at ERROR so the post-run
        proxy.log grep gate trips (mirrors the records arm's failure marker).
        """
        degenerate = (
            self.barriers_fired > 0
            and self.total_experiences == 0
            and self.total_skills_changed == 0
        )
        if degenerate and not self._degenerate_marker_emitted:
            self._degenerate_marker_emitted = True
            logger.error(
                "[MirixGenericMemoryAdapter] %s degenerate run: %d barrier(s) "
                "fired but total_experiences=0 AND total_skills_changed=0 — the "
                "generic memory path never evolved any skill. Verify "
                "SKILL_TRIGGER_SESSION_THRESHOLD is disabled on the server (so the "
                "in-band trigger isn't double-consuming) and that messages "
                "persisted via /memory/add_sync.",
                EVOLVE_FAILURE_MARKER,
                self.barriers_fired,
            )
        return degenerate

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

        Mirrors :meth:`MirixEvolverAdapter._post_with_retry`: the sync
        ``httpx.Client.post`` runs via ``asyncio.to_thread`` so a multi-second
        server-side extraction does not block the proxy event loop.
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


def _parse_auto_dream_counts(payload: Any) -> tuple[int, int]:
    """Extract ``(experiences, skills_changed)`` from an AutoDreamResponse body.

    ``experiences`` = ``processed["procedural"]["total"]`` (per-barrier sessions
    distilled). ``skills_changed`` = the top-level ``skills_changed`` int added by
    the Part-B schema change. Tolerant of older shapes → ``(0, 0)``.
    """
    if not isinstance(payload, dict):
        return 0, 0
    experiences = 0
    processed = payload.get("processed")
    if isinstance(processed, dict):
        proc = processed.get("procedural")
        if isinstance(proc, dict):
            t = proc.get("total")
            if isinstance(t, int):
                experiences = t
    skills_changed = payload.get("skills_changed")
    if not isinstance(skills_changed, int):
        skills_changed = 0
    return experiences, skills_changed


__all__ = [
    "MirixGenericMemoryAdapter",
    "ADD_SYNC_ENDPOINT_PATH",
    "AUTO_DREAM_ENDPOINT_PATH",
    "AGENTS_ENDPOINT_PATH",
    "HEALTH_ENDPOINT_PATH",
    "IN_BAND_TRIGGER_DISABLED_MIN",
    "DEFAULT_EVOLVE_EVERY_N_TURNS",
    "build_add_sync_payload",
    "_parse_auto_dream_counts",
]
