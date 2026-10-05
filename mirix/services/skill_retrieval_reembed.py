"""Re-embedding skills that have no retrieval vector.

A skill's ``retrieval_embedding`` is NULL after
``scripts/migrate_skill_retrieval_fields.sql`` (which clears the renamed
column: its old content was the description's vector, not the retrieval
text's), and when an embedding call still failed after its retries (the
skill is then stored without a vector rather than not at all). Until it has
one, the skill fails every vector gate. Two ways fill them:

* **at startup** — :class:`SkillRetrievalReembedder`, started by the FastAPI
  lifespan: one pass in a background task, at most
  ``MIRIX_SKILL_REEMBED_CONCURRENCY`` (default 4) embedding calls at a time,
  cancelled cleanly on shutdown;
* **after every learning run** — the run's user only
  (``AutoDreamManager``), before the run is recorded: a skill the curator
  stored without a vector during an embedding outage gets it as soon as the
  provider answers again, in the same process (a benchmark learns and then
  retrieves in one server). A circuit breaker ends this pass after
  ``MIRIX_SKILL_REEMBED_MAX_CONSECUTIVE_FAILURES`` (default 3) skills in a
  row failed to embed — an outage is not paid for skill by skill;
* **on demand** — ``scripts/reembed_skill_retrieval.py``.

Both run :func:`reembed_missing_retrieval_embeddings`: the live skills whose
vector IS NULL, page by page in id order, each re-embedded by
:meth:`ProceduralMemoryManager.reembed_skill` — the retrieval text (name,
triggers, description) embedded with the row's OWN ``embedding_config``,
written through ``apply_skill_change`` under an optimistic check that the text
is unchanged. A pass is idempotent (a skill that got its vector meanwhile is
skipped) and always ends: a skill whose embedding fails is left for the next
pass instead of being retried in a loop. Soft-deleted skills are not
re-embedded (nothing retrieves them); one that is restored is picked up by the
next pass. Two passes that meet — the startup pass and a learning run's,
right after the retrieval-field migration — embed each skill once:
``reembed_skill`` shares the outcome of the call already in flight for that
skill instead of making its own.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from mirix.log import get_logger

logger = get_logger(__name__)

DEFAULT_REEMBED_BATCH_SIZE = 100


#: Outcome of a skill the pass did not try because its circuit breaker was
#: open (``max_consecutive_failures`` skills in a row had failed).
SKIPPED = "skipped"
# Outcomes that prove the provider answered (the breaker's count restarts)
# and outcomes that are failures of the embedding call (they count).
_PROVIDER_ANSWERED = frozenset({"filled", "changed"})
_EMBEDDING_FAILED = frozenset({"failed", "error"})


@dataclass
class ReembedReport:
    """What a pass did: skills scanned, and how many of each outcome of
    ``ProceduralMemoryManager.reembed_skill`` (``filled``, ``present``,
    ``gone``, ``changed``, ``no_config``, ``failed``, ``disabled``),
    ``error`` (an unexpected exception, logged) or ``skipped`` (not tried:
    the circuit breaker was open). ``stopped_early``: the breaker ended the
    pass — the skills of the pages it did not reach were not even scanned."""

    scanned: int = 0
    outcomes: Dict[str, int] = field(default_factory=dict)
    dry_run: bool = False
    stopped_early: bool = False

    def count(self, outcome: str) -> None:
        self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1

    @property
    def filled(self) -> int:
        return self.outcomes.get("filled", 0)

    @property
    def failed(self) -> int:
        return self.outcomes.get("failed", 0) + self.outcomes.get("error", 0)

    @property
    def no_config(self) -> int:
        """Skills that can never be embedded: their row has no embedding config."""
        return self.outcomes.get("no_config", 0)

    @property
    def skipped(self) -> int:
        """Skills scanned but not tried: the circuit breaker was open."""
        return self.outcomes.get(SKIPPED, 0)

    @property
    def still_missing(self) -> int:
        """Skills the pass left without a vector: failed, errored, skipped or
        without an embedding config (``changed`` / ``gone`` ones brought their
        own vector or no longer matter). A pass that stopped early may have
        left more that it never scanned."""
        return self.failed + self.no_config + self.skipped

    def summary(self) -> str:
        if self.dry_run:
            return f"{self.scanned} live skill(s) without a retrieval vector"
        details = ", ".join(f"{k} {v}" for k, v in sorted(self.outcomes.items()))
        stopped = " (stopped early: embedding outage)" if self.stopped_early else ""
        return (
            f"scanned {self.scanned} live skill(s) without a retrieval vector: "
            f"{details or 'nothing to do'}{stopped}"
        )


class _CircuitBreaker:
    """Opens after ``limit`` embedding failures in a row (None: never)."""

    def __init__(self, limit: Optional[int]):
        self.limit = limit
        self.consecutive = 0
        self.open = False

    def record(self, outcome: str) -> None:
        if outcome in _PROVIDER_ANSWERED:
            self.consecutive = 0
        elif outcome in _EMBEDDING_FAILED:
            self.consecutive += 1
            if self.limit is not None and self.consecutive >= self.limit:
                self.open = True


async def reembed_missing_retrieval_embeddings(
    *,
    manager: Any = None,
    session_maker: Any = None,
    organization_id: Optional[str] = None,
    user_id: Optional[str] = None,
    batch_size: int = DEFAULT_REEMBED_BATCH_SIZE,
    concurrency: Optional[int] = None,
    dry_run: bool = False,
    max_consecutive_failures: Optional[int] = None,
) -> ReembedReport:
    """One pass over the live skills whose ``retrieval_embedding`` IS NULL.

    ``manager`` (a ``ProceduralMemoryManager``) or ``session_maker`` (default:
    the server's database) selects the store; ``organization_id`` limits the
    pass to one organization, ``user_id`` to one user. At most ``concurrency``
    (``MIRIX_SKILL_REEMBED_CONCURRENCY``) skills are re-embedded at a time.
    ``dry_run`` only counts them. Cancellation stops the pass.

    ``max_consecutive_failures``: the circuit breaker. Once that many skills
    in a row failed to embed (a skill whose provider answered restarts the
    count), no further skill is tried: the ones already listed are counted
    ``skipped``, no further page is read, and ``report.stopped_early`` is
    set. At most ``concurrency - 1`` more skills — those already started —
    are tried after it opens. None: every skill is tried.
    """
    from mirix import constants
    from mirix.services import procedural_memory_manager as pmm

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if concurrency is None:
        concurrency = constants.SKILL_REEMBED_CONCURRENCY
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if max_consecutive_failures is not None and max_consecutive_failures < 1:
        raise ValueError("max_consecutive_failures must be at least 1 (or None)")
    manager = manager or pmm.ProceduralMemoryManager(session_maker=session_maker)
    report = ReembedReport(dry_run=dry_run)
    if not dry_run and not pmm.BUILD_EMBEDDINGS_FOR_MEMORY:
        logger.info("Embeddings are off (BUILD_EMBEDDINGS_FOR_MEMORY): no re-embedding")
        return report
    gate = asyncio.Semaphore(concurrency)
    breaker = _CircuitBreaker(max_consecutive_failures)

    async def one(skill_id: str) -> None:
        async with gate:
            if breaker.open:
                report.count(SKIPPED)
                return
            try:
                outcome = await manager.reembed_skill(skill_id)
            except Exception:  # noqa: BLE001 — one skill never stops the pass
                logger.exception("Re-embedding skill %s failed", skill_id)
                outcome = "error"
            breaker.record(outcome)
        report.count(outcome)

    after_id: Optional[str] = None
    while True:
        skill_ids = await manager.list_skill_ids_missing_retrieval_embedding(
            organization_id=organization_id,
            user_id=user_id,
            after_id=after_id,
            limit=batch_size,
        )
        if not skill_ids:
            break
        report.scanned += len(skill_ids)
        after_id = skill_ids[-1]
        if dry_run:
            continue
        await asyncio.gather(*(one(skill_id) for skill_id in skill_ids))
        if breaker.open:
            # Early only if something was left: a skill of this page not
            # tried, or a full page (more may follow). Opening on the very
            # last skill ends the pass as it would have ended anyway.
            if not report.skipped and len(skill_ids) < batch_size:
                break
            report.stopped_early = True
            logger.warning(
                "Skill re-embedding stopped after %d embedding failures in a row "
                "(an embedding outage?): %s",
                breaker.consecutive,
                report.summary(),
            )
            break
    return report


class SkillRetrievalReembedder:
    """The startup re-embedding pass as a background task.

    :meth:`start` schedules one pass and keeps a reference to it;
    :meth:`stop` cancels it and waits for it to end. A failing pass is logged,
    never raised: startup never depends on it.
    """

    def __init__(
        self,
        *,
        manager: Any = None,
        session_maker: Any = None,
        concurrency: Optional[int] = None,
        batch_size: int = DEFAULT_REEMBED_BATCH_SIZE,
    ):
        self.manager = manager
        self.session_maker = session_maker
        self.concurrency = concurrency
        self.batch_size = batch_size
        self.report: Optional[ReembedReport] = None
        self._task: Optional[asyncio.Task] = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def task(self) -> Optional[asyncio.Task]:
        return self._task

    def start(self) -> asyncio.Task:
        if self._task is None:
            self._task = asyncio.ensure_future(self._run())
            try:
                self._task.set_name("mirix-skill-retrieval-reembed")
            except Exception:  # noqa: BLE001
                pass
        return self._task

    async def _run(self) -> Optional[ReembedReport]:
        try:
            report = await reembed_missing_retrieval_embeddings(
                manager=self.manager,
                session_maker=self.session_maker,
                concurrency=self.concurrency,
                batch_size=self.batch_size,
            )
        except asyncio.CancelledError:
            logger.info("Skill re-embedding stopped (shutdown)")
            raise
        except Exception:  # noqa: BLE001 — see the class docstring
            logger.exception("Skill re-embedding pass failed; the next start retries")
            return None
        self.report = report
        if report.scanned:
            logger.info("Skill re-embedding: %s", report.summary())
        return report

    async def stop(self, timeout: float = 10.0) -> None:
        task = self._task
        if task is None or task.done():
            return
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=timeout)
        if not done:
            logger.warning(
                "The skill re-embedding task did not stop within %.0f s", timeout
            )
