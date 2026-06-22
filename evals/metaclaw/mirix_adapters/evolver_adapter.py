"""MirixEvolverAdapter — paper-SkillEvolver-shaped facade over MIRIX's
``POST /v1/skills/evolve`` endpoint.

Used by the D6 dispatch in ``evals/metaclaw/vendor/metaclaw/launcher.py`` when
``METACLAW_EVOLVER_PROVIDER=mirix``.  Paper's MetaClaw proxy calls a duck-typed
``SkillEvolver`` at session_done with a list of
``SimpleNamespace(prompt_text, response_text, reward=0.0)`` samples; this
adapter serializes those into bounded message strings and POSTs them to MIRIX,
which runs its procedural-memory agent server-side to create / edit / delete
skills.

Design rationale (see PRD user-stories #15, #16, #27, #28):

  * **``evolve()`` always returns ``[]``** — MIRIX is the sole writer; paper's
    downstream ``manager.add_skills(new_skills)`` must therefore be a no-op.
    Returning a non-empty list would cause MIRIX skills to be re-POSTed to
    the SkillManager adapter (double-write + drift risk).
  * **No fake quality label** — paper hardcodes ``reward=0.0`` for every
    skills_only sample; we honor that signal and do NOT inject ``outcome=FAIL``
    or similar.  MIRIX's procedural agent decides what to extract.
  * **No ``super().__init__()``** — the parent's constructor demands
    ``OPENAI_API_KEY`` + ``SKILL_EVOLVER_MODEL`` env vars MIRIX does not need
    (LLM work happens server-side inside MIRIX).
  * **Bounded payload** — per-turn message is built from
    ``prompt_text[-1500:]`` + ``response_text[:1500]`` so cumulative-history
    prompts do not blow up the request body.
  * **Endpoint preflight** — at construction, GET ``/openapi.json`` and verify
    ``/v1/skills/evolve`` is exposed.  Raising loud here keeps the eval from
    silently producing a no-skills run when the MIRIX server lacks the
    endpoint (e.g. wrong branch checked out).
  * **One retry on 5xx** — PG pool occasionally returns transient
    ConnectionRefusedError / ConnectionResetError surfacing as HTTP 5xx;
    a single retry after 2 s mitigates without masking real failures.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# FIX7 — defense-in-depth timeout. evolve-from-records runs MIRIX's procedural
# curator agent server-side; a slow-but-working curator must NOT be cut off and
# have its failure swallowed into a degenerate run. The ROOT cause of the >600s
# stall (the curator lacking skill tools → chaining-cap spin) is fixed separately
# in product code (FIX8); this generous ceiling is the safety margin so the eval
# never loses a legitimately-slow evolution. A request that genuinely hangs past
# 1800s still surfaces LOUDLY (see evolve_from_records / distill_round error
# logging) rather than silently returning False.
DEFAULT_TIMEOUT_S = 1800.0
DEFAULT_CLIENT_ID = "client-00000000-0000-4000-8000-000000000000"
DEFAULT_RETRY_SLEEP_S = 2.0
DEFAULT_MAX_HTTP_RETRIES = 4  # initial + 4 retries = 5 attempts
DEFAULT_RETRY_BACKOFF_MAX_S = 30.0
PROMPT_TAIL_CHARS = 1500
RESPONSE_HEAD_CHARS = 1500
EVOLVE_ENDPOINT_PATH = "/v1/skills/evolve"

# FIX7 — silent-swallow defense. A fixed, greppable marker emitted at ERROR
# level whenever an evolution (raw-transcript or records) fails. The post-run
# sanity gate greps proxy.log for this token to detect a run whose skills never
# actually evolved (the failure used to be swallowed into ``return False`` /
# ``return []`` with only a WARNING). Keep the literal stable — the gate depends
# on it.
EVOLVE_FAILURE_MARKER = "EVOLVE_FAILURE"

# C5 (new harness) — the records-evolution endpoints. The old raw-transcript
# path above (`/v1/skills/evolve`) is left UNTOUCHED so the regression-baseline
# arm keeps behaving byte-identically.
DISTILL_ROUND_ENDPOINT_PATH = "/v1/skills/distill-round"
EVOLVE_FROM_RECORDS_ENDPOINT_PATH = "/v1/skills/evolve-from-records"

# Evolve every N completed graded rounds in the new mode (DESIGN §0/§C5). Counts
# fire at {5,10,15,…} — the watermark passed to evolve-from-records is the current
# round index, so an in-progress round (== the round just distilled) is consumed
# by the lag (its record exists with round_index < watermark only after its
# successor arrives), never while its own round is being graded (§3 guard #5).
DEFAULT_EVOLVE_EVERY_N_ROUNDS = 5


def _sample_to_message(sample: Any, idx: int) -> str:
    """Serialize a paper-shape sample into a single bounded message string.

    Paper passes ``SimpleNamespace(prompt_text, response_text, reward=0.0)``
    (see ``api_server.py:2126-2133``).  The reward is uniformly ``0.0`` in
    skills_only mode; we do NOT translate that into a FAIL label because doing
    so would mislead MIRIX's procedural-memory agent into treating every turn
    as a failure case (per PRD user-story #16).
    """
    prompt = getattr(sample, "prompt_text", "") or ""
    response = getattr(sample, "response_text", "") or ""
    prompt_tail = prompt[-PROMPT_TAIL_CHARS:]
    response_head = response[:RESPONSE_HEAD_CHARS]
    # Triple-backtick fences let the procedural agent see prompt vs response
    # as distinct regions without HTML escaping or JSON quoting noise.
    return (
        f"### Turn {idx + 1}\n\n"
        f"**Conversation context (tail {PROMPT_TAIL_CHARS}):**\n"
        f"```\n{prompt_tail}\n```\n\n"
        f"**Assistant response (head {RESPONSE_HEAD_CHARS}):**\n"
        f"```\n{response_head}\n```"
    )


# --------------------------------------------------------------------------- #
# C5 (new harness) — per-round ingestion payload builder + evolve cadence       #
# These are PURE functions (no I/O) so the §3 leakage guards (G1/G3) and the    #
# evolve-every-5-rounds counter (G5) are unit-testable without a live server.   #
# --------------------------------------------------------------------------- #

# The ONLY keys the harness may put on the per-round turn it sends to MIRIX. The
# round's question (which already carries the round-(t-1) "[Previous Feedback]"
# block, see benchmark/src/infer/prompts.py::with_feedback) goes in prompt_text;
# the agent's FINAL graded answer goes in response_text. NOTHING else — never an
# oracle field (eval.answer/eval.command/inline_score/reward/feedback.options).
_DISTILL_TURN_ALLOWED_KEYS = ("prompt_text", "response_text")

# Keys that, if they ever appeared in a distill payload, would be an oracle leak
# (§3 guard #1). The builder constructs the body field-by-field from the
# allow-list so these can never appear; the tuple is exported for tests to assert
# their absence against the actual built payload.
DISTILL_FORBIDDEN_KEYS = (
    "inline_score",
    "reward",
    "eval",
    "answer",  # eval.answer — the MC oracle set
    "command",  # eval.command — the file_check oracle
    "expect_exit",
    "options",  # feedback.options{} — per-option oracle hints
    "passed",
    "feedback",
    "correct",
    "incorrect",
    "score",
    "selected",
    "format_valid",
)


def build_distill_round_payload(
    *,
    day: str,
    round_id: str,
    round_index: int,
    query: str,
    answer: str,
    user_id: str,
    session_id: Optional[str] = None,
    session_done: bool = False,
) -> Dict[str, Any]:
    """Build the EXACT body POSTed to ``/v1/skills/distill-round`` for one round.

    Leakage guards (assert on the RETURN VALUE in tests):

      * **G1 (no oracle field).** The ``turn`` is built only from ``query`` →
        ``prompt_text`` and ``answer`` → ``response_text``. The caller passes the
        agent-visible ``query`` (which the bench already built via
        ``with_feedback`` so it carries the round-(t-1) feedback) and the agent's
        FINAL graded ``answer``. No ``eval.*`` / ``inline_score`` / ``reward`` /
        ``feedback.options`` is ever read here, so none can appear in the body.
      * **G3 (one-round lag preserved).** This carries round t's question +
        answer. The round-(t-1) feedback is INSIDE ``query`` (the bench put it
        there). Round t's OWN grade is NOT sent — the distiller derives t's
        outcome only when round t+1 arrives carrying t's feedback. The harness
        therefore never sends a round's own correctness here.

    ``query`` is mapped to ``prompt_text`` verbatim (NOT tail-truncated): the
    leading ``[Previous Feedback]`` block is the only legitimate pass/fail source
    and must survive (the distiller's ``sanitize_turn`` bounds the middle/tail
    server-side if needed).
    """
    return {
        "day": str(day),
        "round_id": str(round_id),
        "round_index": int(round_index),
        "turn": {
            "prompt_text": query or "",
            "response_text": answer or "",
        },
        "user_id": user_id,
        "session_id": session_id,
        "session_done": bool(session_done),
    }


def should_evolve_at(completed_rounds: int, every_n: int) -> bool:
    """True iff an evolution should fire after ``completed_rounds`` graded rounds.

    Fires at {N, 2N, 3N, …} (e.g. {5,10,15,…} for N=5), NEVER at {N+1, 2N+1, …}.
    A non-positive ``every_n`` disables periodic evolution (only session_done
    would evolve, if wired) so a misconfig can't divide-by-zero.
    """
    if every_n <= 0:
        return False
    return completed_rounds > 0 and completed_rounds % every_n == 0


class RoundEvolutionTracker:
    """Per-session counter of COMPLETED GRADED rounds + evolve-cadence decisions.

    The counter lives here (not in the bench) so the cadence is owned by the
    MIRIX side and unit-testable without the bench subprocess. ``note_round``
    returns ``True`` exactly on the rounds whose 1-based completed-count is a
    multiple of ``every_n`` (G5: fires at {5,10,15,…}). The watermark to pass to
    evolve-from-records is the round's own ``round_index`` (only records with
    ``round_index < watermark`` are consumed server-side, so the just-completed
    round — whose record isn't finalized until its successor — is safely excluded).
    """

    def __init__(self, every_n: int = DEFAULT_EVOLVE_EVERY_N_ROUNDS):
        self.every_n = every_n
        self._counts: Dict[str, int] = {}

    def note_round(self, session_id: str) -> bool:
        """Record one completed graded round for ``session_id``; return whether
        an evolution should fire now."""
        n = self._counts.get(session_id, 0) + 1
        self._counts[session_id] = n
        return should_evolve_at(n, self.every_n)

    def completed(self, session_id: str) -> int:
        return self._counts.get(session_id, 0)

    def reset(self, session_id: str) -> None:
        self._counts.pop(session_id, None)


class MirixEvolverAdapter:
    """Drop-in replacement for paper's :class:`SkillEvolver` backed by
    ``POST /v1/skills/evolve`` on the MIRIX REST server.

    Surface mirrors ``evals/metaclaw/vendor/metaclaw/skill_evolver.py``
    (the duck-typed subset used by ``api_server._evolve_skills_for_session``):

      * ``evolve(failed_samples, current_skills) -> list[dict]``  (async)
      * ``should_evolve(batch, threshold=0.0) -> bool``
      * ``.update_history``  (no-op attr — paper's ``get_update_summary``
        inheritance helper reads it; safe to keep empty)
      * ``.history_path``    (always ``None`` — MIRIX persists server-side)
    """

    # ------------------------------------------------------------------ #
    # Construction                                                        #
    # ------------------------------------------------------------------ #

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
        # C5: evolve every N completed graded rounds in the records (new) mode.
        evolve_every_n_rounds: int = DEFAULT_EVOLVE_EVERY_N_ROUNDS,
        # Paper-launcher kwargs that this adapter accepts-and-ignores so the
        # D6 dispatch can pass through any future cfg keys without crashing.
        max_new_skills: Optional[int] = None,  # noqa: ARG002 — paper signature compat
        history_path: Optional[str] = None,  # noqa: ARG002 — paper signature compat
        **_paper_kwargs: Any,  # noqa: ARG002 — paper signature compat
    ) -> None:
        """
        Args:
            base_url:  MIRIX REST root (e.g. ``http://127.0.0.1:8531``).
            user_id:   The pre-minted user id for this run (runner mints it
                       via ``/users/create_or_get`` before constructing the
                       adapter; we accept it as-is).
            timeout_s: httpx timeout per request.  Default 1800s (FIX7) is a
                       generous defense-in-depth ceiling so a slow-but-working
                       server-side curator is never cut off and silently
                       swallowed into a no-skills run.
            client_id: ``X-Client-Id`` header value.
            transport: Optional httpx transport — tests inject
                       :class:`httpx.MockTransport`.
            retry_sleep_s: Backoff between the (single) 5xx retry attempt.
            max_new_skills, history_path: Accepted for paper-launcher
                signature compatibility.  MIRIX decides server-side how
                many skills to create.

        Raises:
            RuntimeError: if the MIRIX server at ``base_url`` does not expose
                ``/v1/skills/evolve`` (caught at startup → fail fast).

        IMPORTANT: We deliberately do NOT call ``super().__init__()`` because
        the paper SkillEvolver constructor demands ``OPENAI_API_KEY`` +
        ``SKILL_EVOLVER_MODEL`` env vars that this adapter does not use.
        MIRIX runs its own LLM call server-side via the procedural agent.
        """
        self.base_url = base_url.rstrip("/")
        self.user_id = user_id
        self.client_id = client_id
        self.retry_sleep_s = retry_sleep_s
        # Clamp to >= 0 so a misconfigured negative never yields attempts == 0,
        # which would leave `resp` unbound before raise_for_status (codex review
        # 2026-05-28, LOW).  0 means "one attempt, no retries".
        self.max_http_retries = max(0, max_http_retries)
        self.retry_backoff_max_s = retry_backoff_max_s

        # Paper-compat attrs.  ``get_update_summary`` (paper helper) reads
        # ``update_history`` if anything inherits from this class.  Empty
        # list is structurally valid and conveys "MIRIX is authoritative".
        self.update_history: List[dict] = []
        self.history_path: Optional[str] = None

        # C5 (new harness) — per-session completed-graded-round counter that
        # drives evolve-every-N-rounds. Only consulted on the records path
        # (distill_round); the raw-transcript evolve() path never touches it.
        self.evolve_every_n_rounds = evolve_every_n_rounds
        self._round_tracker = RoundEvolutionTracker(every_n=evolve_every_n_rounds)
        # Count of evolution events fired on the records path (per-arm telemetry
        # for the new-harness − old-harness delta, DESIGN §C5 P1-10).
        self.records_evolution_events = 0
        # FIX7 — silent-swallow defense. Count of evolution attempts that FAILED
        # (HTTP error / unexpected exception) on either path. Each increment is
        # accompanied by a LOUD logger.error carrying EVOLVE_FAILURE_MARKER so
        # the post-run sanity gate can detect a run whose skills never evolved.
        self.evolve_failures = 0

        client_kwargs: Dict[str, Any] = {
            "base_url": self.base_url,
            "timeout": timeout_s,
            "headers": {"X-Client-Id": self.client_id},
        }
        if transport is not None:
            client_kwargs["transport"] = transport
        self._http = httpx.Client(**client_kwargs)

        # Preflight: refuse to construct if /v1/skills/evolve isn't exposed.
        # This is the single best signal that the eval is pointed at a
        # MIRIX server lacking the feat/skill-evolve branch.
        self._verify_evolve_endpoint_or_raise()

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:  # pragma: no cover — defensive
            pass

    # ------------------------------------------------------------------ #
    # Preflight                                                           #
    # ------------------------------------------------------------------ #

    def _verify_evolve_endpoint_or_raise(self) -> None:
        """GET ``/openapi.json`` and check that ``/v1/skills/evolve`` is a
        registered path.  Raise :class:`RuntimeError` with a clear remediation
        message if not.

        On HTTP / parse error we log a warning but do NOT raise — we'd rather
        attempt the evolve call and surface the actual server error than fail
        startup on an openapi-spec hiccup.
        """
        try:
            resp = self._http.get("/openapi.json")
            resp.raise_for_status()
            spec = resp.json()
            paths = spec.get("paths") if isinstance(spec, dict) else None
            if not isinstance(paths, dict):
                logger.warning(
                    "[MirixEvolverAdapter] /openapi.json has no 'paths' object; "
                    "skipping preflight check"
                )
                return
            if EVOLVE_ENDPOINT_PATH not in paths:
                raise RuntimeError(
                    f"MIRIX server at {self.base_url} lacks {EVOLVE_ENDPOINT_PATH}; "
                    f"switch to feat/skill-evolve branch or merge it in."
                )
            logger.info(
                "[MirixEvolverAdapter] preflight ok: %s exposes %s",
                self.base_url,
                EVOLVE_ENDPOINT_PATH,
            )
        except RuntimeError:
            raise
        except (httpx.HTTPError, ValueError) as e:
            logger.warning(
                "[MirixEvolverAdapter] preflight check skipped (%s); will attempt "
                "evolve calls anyway",
                e,
            )

    # ------------------------------------------------------------------ #
    # Public interface (paper SkillEvolver duck-typed surface)            #
    # ------------------------------------------------------------------ #

    def should_evolve(self, batch: list, threshold: float = 0.0) -> bool:  # noqa: ARG002
        """Always True.

        ``api_server._evolve_skills_for_session`` does NOT call this in
        skills_only mode — it calls ``evolve`` directly.  Defined defensively
        so any future caller that does check this gate still triggers evolve.
        """
        return True

    async def evolve(
        self,
        failed_samples: list,
        current_skills: Dict[str, Any],  # noqa: ARG002 — MIRIX queries its own bank server-side
    ) -> List[dict]:
        """Forward the session's turns to MIRIX's procedural agent.

        Args:
            failed_samples: List of ``SimpleNamespace(prompt_text,
                response_text, reward=0.0)`` per ``api_server.py:2126``.
            current_skills: Paper's existing skill-bank dict.  Unused —
                MIRIX consults its own store server-side.

        Returns:
            Always ``[]``.  The MIRIX server already wrote any
            created/edited/deleted skills; returning ``[]`` makes paper's
            downstream ``manager.add_skills(new_skills)`` a no-op
            (``api_server.py:2140-2145``) so MIRIX remains the sole writer.
        """
        if not failed_samples:
            logger.info(
                "[MirixEvolverAdapter] evolve 0 samples -> short-circuit (no HTTP call)"
            )
            return []

        messages = [_sample_to_message(s, i) for i, s in enumerate(failed_samples)]
        body = {"messages": messages, "user_id": self.user_id}
        n = len(messages)

        try:
            payload = await self._post_with_retry(body)
        except httpx.HTTPError as e:
            # FIX7 — LOUD. Old-baseline (raw-transcript) arm: a swallowed evolve
            # failure here degrades the OLD arm to retrieval-only, which would
            # inflate the load-bearing (new − old) delta in the WRONG direction.
            # Emit at ERROR with the marker + count it.
            self.evolve_failures += 1
            logger.error(
                "[MirixEvolverAdapter] %s evolve %d samples -> HTTP error after "
                "retry: %r (evolve_failures=%d) — skills did NOT evolve this "
                "session",
                EVOLVE_FAILURE_MARKER,
                n,
                e,
                self.evolve_failures,
            )
            return []
        except Exception as e:  # noqa: BLE001 — surface any failure LOUDLY
            self.evolve_failures += 1
            logger.error(
                "[MirixEvolverAdapter] %s evolve %d samples -> unexpected error: "
                "%r (evolve_failures=%d) — skills did NOT evolve this session",
                EVOLVE_FAILURE_MARKER,
                n,
                e,
                self.evolve_failures,
                exc_info=True,
            )
            return []

        created, edited, deleted = _parse_evolve_diff_counts(payload)
        logger.info(
            "[MirixEvolverAdapter] evolve %d samples -> MIRIX created=%d edited=%d "
            "deleted=%d (paper add_skills will be no-op by design)",
            n,
            created,
            edited,
            deleted,
        )
        # Always [] — see docstring + PRD user-story #15.
        return []

    # ------------------------------------------------------------------ #
    # C5 (new harness) — message-by-message records evolution            #
    # ------------------------------------------------------------------ #

    async def distill_round(
        self,
        *,
        day: str,
        round_id: str,
        round_index: int,
        query: str,
        answer: str,
        session_id: Optional[str] = None,
        session_done: bool = False,
    ) -> Dict[str, Any]:
        """Push EXACTLY ONE graded round to MIRIX's per-round distiller, then
        evolve from records every N completed graded rounds.

        Flow (DESIGN §C5):

          1. POST ``{day, round_id, round_index, turn:{prompt_text=query,
             response_text=answer}}`` to ``/v1/skills/distill-round``. The server
             buffers this round and distills the PREVIOUS one (one-round lag), so
             we NEVER send a round's own grade (§3 G3). ``query`` is the
             agent-visible message (already carrying round-(t-1) feedback); we
             pass it verbatim (no tail-truncation — the feedback head must
             survive).
          2. Increment the per-session completed-round counter. When it hits a
             multiple of ``evolve_every_n_rounds`` (≙ {5,10,15,…}, §3 G5), POST to
             ``/v1/skills/evolve-from-records`` with ``before_round_index =
             round_index`` (the watermark; only records strictly before it are
             consumed, so the just-distilled round — whose record isn't finalized
             until its successor — is safely excluded).

        Returns a small status dict; on any HTTP error returns ``{"ok": False}``
        and keeps the bench loop alive (never raises into the bench).
        """
        sid = session_id if session_id is not None else day
        body = build_distill_round_payload(
            day=day,
            round_id=round_id,
            round_index=round_index,
            query=query,
            answer=answer,
            user_id=self.user_id,
            session_id=sid,
            session_done=session_done,
        )
        result: Dict[str, Any] = {"ok": True, "evolved": False}
        try:
            distilled = await self._post_with_retry(DISTILL_ROUND_ENDPOINT_PATH, body)
            result["distill"] = distilled
            # FIX7 / codex P1: a 2xx distill body that EXPLICITLY reports failure
            # means the round's record was NOT written — it can never be consumed
            # by evolve. Treat it like a transport failure so the bench's
            # ok-contract counts it as a distill_failure and the sanity gate sees
            # the EVOLVE_FAILURE marker.
            if _payload_reports_failure(distilled):
                self.evolve_failures += 1
                logger.error(
                    "[MirixEvolverAdapter] %s distill-round %s/%s -> 2xx but body "
                    "reports failure: %r (evolve_failures=%d) — record NOT written",
                    EVOLVE_FAILURE_MARKER,
                    day,
                    round_id,
                    distilled,
                    self.evolve_failures,
                )
                return {"ok": False, "evolved": False}
        except httpx.HTTPError as e:
            # FIX7 — LOUD. A failed distill means this round's record never gets
            # written, so it can never be consumed by evolve. Emit at ERROR with
            # the marker; the {"ok": False} return is what the bench's tightened
            # success-contract counts as a distill_failure.
            self.evolve_failures += 1
            logger.error(
                "[MirixEvolverAdapter] %s distill-round %s/%s -> HTTP error: %r "
                "(evolve_failures=%d) — record NOT written",
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
                "[MirixEvolverAdapter] %s distill-round %s/%s -> unexpected "
                "error: %r (evolve_failures=%d) — record NOT written",
                EVOLVE_FAILURE_MARKER,
                day,
                round_id,
                e,
                self.evolve_failures,
                exc_info=True,
            )
            return {"ok": False, "evolved": False}

        # Cadence: evolve at session-global rounds {N,2N,3N,…}.
        #
        # RESUME-SAFE (codex P1-B): the evolve decision keys off the DETERMINISTIC
        # session-global ``round_index`` (``should_evolve_at(round_index, N)``), NOT
        # an in-process incrementing counter. The bench's ``round_index`` is a
        # function of the round's fixed position in the session (it advances even
        # for resume-skipped rounds), so a RESUMED run fires evolve at exactly the
        # same global rounds {5,10,15,…} as a fresh run — and a round skipped on
        # resume (already distilled in the original run) simply never re-POSTs, so
        # it can't re-fire its own evolution. An internal counter, by contrast,
        # resets to 0 on every fresh proxy process and would count only the
        # *newly executed* rounds, firing evolve at the wrong global boundary (e.g.
        # local-5 == global-12) and consuming records with a non-monotonic
        # watermark. ``note_round`` is still called for the per-session
        # completed-round telemetry (``completed()``), but it no longer gates the
        # evolution.
        #
        # session_done is the end-of-session FLUSH (codex HIGH #2): it carries no
        # query/answer and is NOT a graded round, so it MUST NOT increment the
        # completed-round counter or fire an evolution (doing so would shift the
        # {5,10,15} cadence by one per session). The flush only clears the
        # server-side lag buffer (handled by the distill-round endpoint above).
        #
        # FORGETTING-GUARD SCOPING (DESIGN §C5 forgetting-guard, v1): we do NOT run
        # the heavy staging→adopt + non-regression re-eval here. Under this
        # corrected cadence (evolve every 5 rounds) a bad edit self-corrects within
        # ~5 rounds, so the blast radius is already bounded by C4's
        # ALREADY-IMPLEMENTED soft-delete + hard per-edit size gate (char-delta 800
        # / major-ratio 0.4) + B_max=6. Full staging→adopt remains a documented P1
        # (see DESIGN §C5 / V1-prereqs); it is intentionally out of scope for v1.
        if not session_done:
            self._round_tracker.note_round(sid)  # telemetry only (completed())
            if should_evolve_at(round_index, self.evolve_every_n_rounds):
                evolved = await self.evolve_from_records(before_round_index=round_index)
                result["evolved"] = evolved
                result["watermark"] = round_index
        return result

    async def evolve_from_records(self, *, before_round_index: Optional[int]) -> bool:
        """POST ``/v1/skills/evolve-from-records`` with the watermark.

        The body carries NO transcript — the curator reads the window's distilled
        records server-side. ``before_round_index`` is the watermark (§3 G5):
        only records with ``round_index < watermark`` are consumed. Returns True
        on a 2xx; False (and logs) on any error, so a transient DB blip never
        poisons the run.
        """
        body: Dict[str, Any] = {
            "before_round_index": before_round_index,
            "user_id": self.user_id,
            # Validity run is deterministic formula-only (DESIGN §C4); the
            # autonomous LLM reducer is held out as an ablation arm.
            "use_autonomous_budget": False,
        }
        try:
            payload = await self._post_with_retry(
                EVOLVE_FROM_RECORDS_ENDPOINT_PATH, body
            )
        except httpx.HTTPError as e:
            # FIX7 — LOUD. This used to be a WARNING swallowed into return False,
            # which silently degraded the records arm to retrieval-only. Emit at
            # ERROR with the greppable marker + count it so the post-run sanity
            # gate refuses to trust the delta.
            self.evolve_failures += 1
            logger.error(
                "[MirixEvolverAdapter] %s evolve-from-records (watermark=%s) -> "
                "HTTP error: %r (evolve_failures=%d) — skills did NOT evolve "
                "this window",
                EVOLVE_FAILURE_MARKER,
                before_round_index,
                e,
                self.evolve_failures,
            )
            return False
        except Exception as e:  # noqa: BLE001 — surface any failure LOUDLY
            self.evolve_failures += 1
            logger.error(
                "[MirixEvolverAdapter] %s evolve-from-records (watermark=%s) -> "
                "unexpected error: %r (evolve_failures=%d) — skills did NOT "
                "evolve this window",
                EVOLVE_FAILURE_MARKER,
                before_round_index,
                e,
                self.evolve_failures,
                exc_info=True,
            )
            return False

        # FIX7 / codex P1: a 2xx body that EXPLICITLY reports failure
        # (``success: false`` / ``ok: false``) must NOT count as a successful
        # evolution event — that would let a degenerate window masquerade as
        # healthy. Surface it LOUDLY like a transport failure.
        if _payload_reports_failure(payload):
            self.evolve_failures += 1
            logger.error(
                "[MirixEvolverAdapter] %s evolve-from-records (watermark=%s) -> "
                "2xx but body reports failure: %r (evolve_failures=%d) — skills "
                "did NOT evolve this window",
                EVOLVE_FAILURE_MARKER,
                before_round_index,
                payload,
                self.evolve_failures,
            )
            return False

        self.records_evolution_events += 1
        created, edited, deleted = _parse_evolve_diff_counts(payload)
        skipped = payload.get("skipped") if isinstance(payload, dict) else None
        logger.info(
            "[MirixEvolverAdapter] evolve-from-records (watermark=%s) -> "
            "created=%d edited=%d deleted=%d skipped=%s (event #%d)",
            before_round_index,
            created,
            edited,
            deleted,
            skipped,
            self.records_evolution_events,
        )
        return True

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    async def _post_with_retry(
        self, path: str = EVOLVE_ENDPOINT_PATH, body: Optional[Dict[str, Any]] = None
    ) -> Any:
        """POST ``path`` with retries on transient 5xx.

        We run the sync ``httpx.Client.post`` via ``asyncio.to_thread`` to
        avoid blocking the proxy event loop during the (potentially
        multi-second) procedural-agent server-side extraction.

        ``path`` defaults to the raw-transcript evolve endpoint so existing
        callers (``evolve``, which passes the body positionally) keep working;
        the records-path callers pass an explicit endpoint.
        """
        import asyncio

        # Back-compat: `evolve()` calls `_post_with_retry(body_dict)` positionally.
        if isinstance(path, dict) and body is None:
            path, body = EVOLVE_ENDPOINT_PATH, path
        if body is None:
            body = {}

        def _do_post() -> httpx.Response:
            return self._http.post(path, json=body)

        # MIRIX's PG pool can emit a sustained run of 5xx (or refuse the
        # connection outright) while it reconnects — the 2026-05-28 both-run
        # saw 13 consecutive 500s, silently degrading the mirix arm to
        # retrieval-only.  Retry several times with exponential backoff so a
        # transient DB blip does not poison the comparison.
        attempts = self.max_http_retries + 1
        resp = None
        for attempt in range(attempts):
            try:
                resp = await asyncio.to_thread(_do_post)
            except httpx.HTTPError as exc:
                if attempt >= attempts - 1:
                    raise
                backoff = min(
                    self.retry_sleep_s * (2**attempt), self.retry_backoff_max_s
                )
                logger.warning(
                    "[MirixEvolverAdapter] %s -> transport error %s, retry "
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
                backoff = min(
                    self.retry_sleep_s * (2**attempt), self.retry_backoff_max_s
                )
                logger.warning(
                    "[MirixEvolverAdapter] %s -> HTTP %d, retry %d/%d in %.1fs",
                    path,
                    resp.status_code,
                    attempt + 1,
                    attempts - 1,
                    backoff,
                )
                await asyncio.sleep(backoff)
                continue
            break
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            logger.warning(
                "[MirixEvolverAdapter] evolve response not JSON: %r",
                resp.text[:200],
            )
            return {}


def _payload_reports_failure(payload: Any) -> bool:
    """True iff a 2xx response body EXPLICITLY reports a semantic failure.

    FIX7 / codex P1 defense-in-depth: HTTP 2xx is necessary but not sufficient —
    MIRIX's distill / evolve endpoints answer ``{"success": false, ...}`` (and
    the proxy / adapter use ``{"ok": false}``) on a downstream failure while
    still returning 200. We must NOT treat such a body as a successful evolution.

    Deliberately CONSERVATIVE to stay backward-compatible with the tolerant
    ``_parse_evolve_diff_counts`` shape handling: only an EXPLICIT ``success is
    False`` or ``ok is False`` counts as failure. A MISSING ``success``/``ok``
    key is treated as success (older / alternate shapes that only carry
    ``changes`` must keep working), so this can never false-FAIL a healthy run.
    """
    if not isinstance(payload, dict):
        return False
    if payload.get("success") is False:
        return True
    if payload.get("ok") is False:
        return True
    return False


def _parse_evolve_diff_counts(payload: Any) -> tuple[int, int, int]:
    """Extract ``(created, edited, deleted)`` counts from MIRIX's evolve
    response.

    Expected shape (per server contract):

        {"success": bool,
         "changes": {"created": [...], "edited": [...], "deleted": [...]},
         "summary": {...}}

    Falls back to ``(0, 0, 0)`` if the shape is unrecognised so the log
    line still renders cleanly (the evolve still completed server-side).
    """
    if not isinstance(payload, dict):
        return 0, 0, 0
    changes = payload.get("changes")
    if not isinstance(changes, dict):
        # Tolerate older / alternate shapes — look for top-level counts.
        for k_alt in ("created", "edited", "deleted"):
            if k_alt in payload:
                changes = payload
                break
        else:
            return 0, 0, 0

    def _count(key: str) -> int:
        v = changes.get(key)
        if isinstance(v, list):
            return len(v)
        if isinstance(v, int):
            return v
        return 0

    return _count("created"), _count("edited"), _count("deleted")


__all__ = [
    "MirixEvolverAdapter",
    "EVOLVE_ENDPOINT_PATH",
    "EVOLVE_FAILURE_MARKER",
    "DISTILL_ROUND_ENDPOINT_PATH",
    "EVOLVE_FROM_RECORDS_ENDPOINT_PATH",
    "DEFAULT_EVOLVE_EVERY_N_ROUNDS",
    "DISTILL_FORBIDDEN_KEYS",
    "PROMPT_TAIL_CHARS",
    "RESPONSE_HEAD_CHARS",
    "build_distill_round_payload",
    "should_evolve_at",
    "RoundEvolutionTracker",
    "_sample_to_message",
    "_parse_evolve_diff_counts",
]
