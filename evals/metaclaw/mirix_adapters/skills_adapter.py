"""MirixSkillsAdapter — paper-SkillManager-shaped facade over MIRIX REST.

Used by the D6 dispatch in ``evals/metaclaw/vendor/metaclaw/launcher.py`` when
``METACLAW_SKILLS_PROVIDER=mirix``.  The MetaClaw paper proxy calls a
duck-typed ``SkillManager``; this adapter routes the surface to MIRIX's
``/v1/skills`` REST endpoints so retrieval (and, in slice #4, evolution) is
backed by MIRIX procedural memory.

Surface mirrors ``evals/metaclaw/vendor/metaclaw/skill_manager.py``:

    retrieve, retrieve_relevant, format_for_conversation, add_skill,
    add_skills, .skills (property), .generation (attr), get_skill_count,
    reload, save, close.

Category round-trip
-------------------
MIRIX restricts ``entry_type`` to ``{guide, script, workflow}``; paper's
SkillManager uses free-form ``category`` strings (``general``, ``coding``,
``common_mistakes`` …).  We map paper-category → MIRIX entry_type per the
table in ``_map_paper_category_to_entry_type``, and prefix ``description``
with ``[paper-category=<original>] `` for any category that doesn't survive
that coercion.  ``.skills`` and ``retrieve`` recover the original category
from that prefix so paper's dedup / bucketing logic stays correct.

Errors degrade
--------------
``retrieve`` / ``retrieve_relevant`` swallow HTTP errors and return ``[]``
— the paper proxy must not raise on retrieve.  ``add_skill`` returns False
on any non-success response (409 included).  ``.skills`` returns an empty
paper-shape bank on error.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 600.0
DEFAULT_TOP_K = 6
DEFAULT_CLIENT_ID = "client-00000000-0000-4000-8000-000000000000"
PAPER_CATEGORY_PREFIX = "[paper-category="

# MIRIX procedural_memory.entry_type is enum-restricted to these three values.
MIRIX_ENTRY_TYPES = {"guide", "script", "workflow"}


def _map_paper_category_to_entry_type(category: str) -> str:
    """paper-category → MIRIX entry_type. Default 'guide' is the most generic."""
    cat = (category or "").strip().lower()
    if cat in MIRIX_ENTRY_TYPES:
        return cat
    if cat in ("task", "deployment", "ingestion", "pipeline"):
        return "workflow"
    return "guide"


def _restore_paper_category(
    description: str, fallback_category: str
) -> tuple[str, str]:
    """If *description* begins with ``[paper-category=X] ``, strip the prefix and
    return ``(clean_description, X)``.  Otherwise return ``(description, fallback_category)``.
    """
    if description.startswith(PAPER_CATEGORY_PREFIX):
        end = description.find("]")
        if end > 0:
            cat = description[len(PAPER_CATEGORY_PREFIX) : end]
            tail = description[end + 1 :]
            # Drop the single space we inserted after "]"
            if tail.startswith(" "):
                tail = tail[1:]
            return tail, cat
    return description, fallback_category


def _mirix_skill_to_paper(row: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a MIRIX /v1/skills row → paper SkillManager dict.

    MIRIX field ``instructions`` → paper field ``content``.
    Restores the original paper category from the ``[paper-category=X]``
    prefix we set in ``add_skill`` when needed.
    """
    raw_desc = row.get("description", "") or ""
    fallback_cat = row.get("entry_type") or "general"
    clean_desc, category = _restore_paper_category(raw_desc, fallback_cat)
    return {
        "name": row.get("name", ""),
        "description": clean_desc,
        "content": row.get("instructions", "") or "",
        "category": category,
    }


class MirixSkillsAdapter:
    """Drop-in replacement for paper's SkillManager backed by MIRIX REST API.

    Synchronous interface — paper invokes retrieve() inside its async request
    handler without ``await``, so we use a sync httpx.Client to keep the fork
    a one-line dispatch swap rather than a refactor.
    """

    # ------------------------------------------------------------------ #
    # Construction / teardown                                             #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        base_url: str,
        user_id: str,
        top_k: int = DEFAULT_TOP_K,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        client_id: str = DEFAULT_CLIENT_ID,
        *,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        """
        Args:
            base_url:  MIRIX REST API root (e.g. ``http://127.0.0.1:8531``).
            user_id:   Human-readable user id; resolved lazily on first use via
                       ``POST /users/create_or_get`` (idempotent).  Adapter
                       caches the server-resolved id.
            top_k:     Default page size for ``retrieve()``.
            timeout_s: httpx timeout in seconds.
            client_id: Value for the ``X-Client-Id`` auth header.
            transport: Optional httpx transport — used by tests to inject
                       :class:`httpx.MockTransport`.
        """
        self.base_url = base_url.rstrip("/")
        self.user_name = user_id
        self._resolved_user_id: Optional[str] = None
        self.client_id = client_id
        self.top_k = top_k

        client_kwargs: Dict[str, Any] = {
            "base_url": self.base_url,
            "timeout": timeout_s,
            "headers": {"X-Client-Id": self.client_id},
        }
        if transport is not None:
            client_kwargs["transport"] = transport
        self._http = httpx.Client(**client_kwargs)

        # Paper SkillManager exposes these attributes; AsyncRolloutWorker reads
        # .skills and .generation directly (api_server.py:2134, 2334).
        # `.skills` is a @property below (live MIRIX backend snapshot).
        self.generation: int = 0

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:  # pragma: no cover — defensive
            pass

    # ------------------------------------------------------------------ #
    # User resolution                                                     #
    # ------------------------------------------------------------------ #

    def _ensure_user_id(self) -> str:
        """Resolve *user_name* to a stable server-side user.id (cached).

        Idempotent: ``POST /users/create_or_get`` with ``user_id=user_name``
        — MIRIX treats user_id as the lookup key, returning the existing
        user or creating one with exactly that id.  Adapter.user_name ==
        server.user.id, so the same arm across process restarts → same
        user → skills accumulate across runs.
        """
        if self._resolved_user_id:
            return self._resolved_user_id
        try:
            resp = self._http.post(
                "/users/create_or_get",
                json={"user_id": self.user_name, "name": self.user_name},
            )
            resp.raise_for_status()
            self._resolved_user_id = resp.json()["id"]
            logger.info(
                "[MirixSkillsAdapter] resolved user_name=%s -> %s",
                self.user_name,
                self._resolved_user_id,
            )
            return self._resolved_user_id
        except (httpx.HTTPError, KeyError, ValueError) as e:
            logger.warning(
                "[MirixSkillsAdapter] _ensure_user_id failed (%s) — falling back to user_name",
                e,
            )
            return self.user_name

    # ------------------------------------------------------------------ #
    # Retrieval                                                           #
    # ------------------------------------------------------------------ #

    def retrieve(
        self, task_description: str, top_k: Optional[int] = None
    ) -> List[dict]:
        """GET /v1/skills?query=&limit=&user_id=. Errors degrade to []."""
        k = top_k if top_k is not None else self.top_k
        try:
            uid = self._ensure_user_id()
            resp = self._http.get(
                "/v1/skills",
                params={"query": task_description, "limit": k, "user_id": uid},
            )
            resp.raise_for_status()
            payload = resp.json()
            rows = payload.get("skills") if isinstance(payload, dict) else payload
            out = [_mirix_skill_to_paper(s) for s in (rows or [])]
            logger.info(
                "[MirixSkillsAdapter] retrieve q=%r k=%d returned %d skills",
                task_description[:120],
                k,
                len(out),
            )
            return out
        except (httpx.HTTPError, ValueError) as e:
            logger.warning("[MirixSkillsAdapter] retrieve failed: %s", e)
            return []

    def retrieve_relevant(
        self,
        task_description: str,
        top_k: int = 6,
        min_relevance: float = 0.07,  # noqa: ARG002 — MIRIX ranks server-side
    ) -> List[dict]:
        """Alias for :meth:`retrieve` — MIRIX does its own ranking server-side."""
        return self.retrieve(task_description, top_k=top_k)

    # ------------------------------------------------------------------ #
    # Conversation rendering — byte-copy of paper's reference             #
    # ------------------------------------------------------------------ #

    def format_for_conversation(self, skills: List[dict]) -> str:
        """Render skills into a system-prompt block.

        Byte-identical to ``metaclaw.skill_manager.SkillManager.format_for_conversation``
        so the prompt injection block is identical across arms.
        """
        if not skills:
            return ""
        lines = ["## Active Skills"]
        for skill in skills:
            name = skill.get("name", "")
            description = skill.get("description", "")
            content = skill.get("content", "")
            lines.append(f"\n### {name}")
            if description:
                lines.append(f"_{description}_")
            if content:
                lines.append("")
                lines.append(content)
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Mutation                                                            #
    # ------------------------------------------------------------------ #

    def add_skill(self, skill: dict) -> bool:
        """POST /v1/skills.  Returns True on 2xx, False on 409 / any error.

        Maps paper's free-form ``category`` → MIRIX ``entry_type`` enum
        (``{guide, script, workflow}``).  For categories that don't round-
        trip cleanly, prefixes ``description`` with ``[paper-category=<orig>] ``
        so :attr:`skills` and :meth:`retrieve` can recover the original on read.
        """
        name = (skill.get("name") or "").strip()
        if not name:
            logger.warning("[MirixSkillsAdapter] add_skill called with missing name")
            return False

        paper_cat = (skill.get("category") or "general").strip()
        entry_type = _map_paper_category_to_entry_type(paper_cat)

        desc = skill.get("description", "") or ""
        if paper_cat and paper_cat.lower() not in MIRIX_ENTRY_TYPES:
            desc = f"{PAPER_CATEGORY_PREFIX}{paper_cat}] {desc}"

        body = {
            "name": name,
            "description": desc,
            "instructions": skill.get("content", "") or "",
            "entry_type": entry_type,
            "user_id": self._ensure_user_id(),
        }
        try:
            resp = self._http.post("/v1/skills", json=body)
        except httpx.HTTPError as e:
            logger.warning(
                "[MirixSkillsAdapter] add_skill HTTP error for %s: %s", name, e
            )
            return False
        if resp.status_code == 409:
            logger.info("[MirixSkillsAdapter] duplicate skill %s (server 409)", name)
            return False
        if resp.status_code >= 400:
            logger.warning(
                "[MirixSkillsAdapter] add_skill failed for %s: HTTP %d %s",
                name,
                resp.status_code,
                resp.text[:200],
            )
            return False
        logger.info(
            "[MirixSkillsAdapter] added skill %s (entry_type=%s)", name, entry_type
        )
        return True

    def add_skills(self, new_skills: List[dict], category: str = "general") -> int:
        """Batch add.  Increments :attr:`generation` once when ≥1 skill added."""
        added = 0
        for skill in new_skills:
            payload = skill if "category" in skill else {**skill, "category": category}
            if self.add_skill(payload):
                added += 1
        if added > 0:
            self.generation += 1
        return added

    # ------------------------------------------------------------------ #
    # .skills property — live MIRIX-backed paper-shape bank               #
    # ------------------------------------------------------------------ #

    @property
    def skills(self) -> Dict[str, Any]:
        """Live snapshot of the MIRIX skill bank, bucketed into paper's
        ``SkillManager.skills`` shape (general_skills / task_specific_skills /
        common_mistakes) so paper's dedup logic sees real backend state.
        """
        bucketed: Dict[str, Any] = {
            "general_skills": [],
            "task_specific_skills": {},
            "common_mistakes": [],
        }
        try:
            rows = self._fetch_all_skills_paper_shape()
        except (httpx.HTTPError, ValueError, KeyError) as e:
            logger.warning("[MirixSkillsAdapter] .skills backend query failed: %s", e)
            return bucketed

        for s in rows:
            cat = (s.get("category") or "general").strip()
            if cat == "general":
                bucketed["general_skills"].append(s)
            elif cat == "common_mistakes":
                bucketed["common_mistakes"].append(s)
            else:
                bucketed["task_specific_skills"].setdefault(cat, []).append(s)
        return bucketed

    @skills.setter
    def skills(self, _value):
        """No-op — paper code occasionally does ``manager.skills = ...``
        defensively; we're backend-backed so silently ignore rather than raise."""
        return

    def _fetch_all_skills_paper_shape(self) -> List[dict]:
        """GET /v1/skills?limit=500&user_id= and translate each row."""
        uid = self._ensure_user_id()
        resp = self._http.get(
            "/v1/skills",
            params={"query": "", "limit": 500, "user_id": uid},
        )
        resp.raise_for_status()
        payload = resp.json()
        rows = payload.get("skills") if isinstance(payload, dict) else payload
        return [_mirix_skill_to_paper(s) for s in (rows or [])]

    # ------------------------------------------------------------------ #
    # Introspection / lifecycle                                           #
    # ------------------------------------------------------------------ #

    def get_skill_count(self) -> Dict[str, int]:
        """Live-counted via :attr:`skills`."""
        snap = self.skills
        general = len(snap.get("general_skills", []))
        task_specific = sum(
            len(v) for v in snap.get("task_specific_skills", {}).values()
        )
        common_mistakes = len(snap.get("common_mistakes", []))
        return {
            "general": general,
            "task_specific": task_specific,
            "common_mistakes": common_mistakes,
            "total": general + task_specific + common_mistakes,
        }

    def reload(self) -> None:
        """No-op — every retrieve is live."""
        return None

    def save(self, path: Optional[str] = None) -> None:  # noqa: ARG002
        """No-op — MIRIX persists server-side."""
        return None


__all__ = [
    "MirixSkillsAdapter",
    "MIRIX_ENTRY_TYPES",
    "PAPER_CATEGORY_PREFIX",
    "DEFAULT_CLIENT_ID",
    "_map_paper_category_to_entry_type",
    "_mirix_skill_to_paper",
    "_restore_paper_category",
]
