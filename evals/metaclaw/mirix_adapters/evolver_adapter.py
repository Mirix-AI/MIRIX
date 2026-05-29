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

DEFAULT_TIMEOUT_S = 600.0
DEFAULT_CLIENT_ID = "client-00000000-0000-4000-8000-000000000000"
DEFAULT_RETRY_SLEEP_S = 2.0
DEFAULT_MAX_HTTP_RETRIES = 4  # initial + 4 retries = 5 attempts
DEFAULT_RETRY_BACKOFF_MAX_S = 30.0
PROMPT_TAIL_CHARS = 1500
RESPONSE_HEAD_CHARS = 1500
EVOLVE_ENDPOINT_PATH = "/v1/skills/evolve"


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
            timeout_s: httpx timeout per request.  Default 600s matches
                       MIRIX's procedural-agent worst-case extraction cost.
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
            logger.warning(
                "[MirixEvolverAdapter] evolve %d samples -> HTTP error after retry: %s "
                "(returning [] to keep paper loop alive)",
                n,
                e,
            )
            return []
        except Exception as e:  # pragma: no cover — defensive
            logger.warning(
                "[MirixEvolverAdapter] evolve %d samples -> unexpected error: %s "
                "(returning [])",
                n,
                e,
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
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    async def _post_with_retry(self, body: Dict[str, Any]) -> Any:
        """POST ``/v1/skills/evolve`` with one retry on transient 5xx.

        We run the sync ``httpx.Client.post`` via ``asyncio.to_thread`` to
        avoid blocking the proxy event loop during the (potentially
        multi-second) procedural-agent server-side extraction.
        """
        import asyncio

        def _do_post() -> httpx.Response:
            return self._http.post(EVOLVE_ENDPOINT_PATH, json=body)

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
                    EVOLVE_ENDPOINT_PATH,
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
                    EVOLVE_ENDPOINT_PATH,
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
    "PROMPT_TAIL_CHARS",
    "RESPONSE_HEAD_CHARS",
    "_sample_to_message",
    "_parse_evolve_diff_counts",
]
