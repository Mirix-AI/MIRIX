"""Adapter that exposes MIRIX procedural memory as paper's SkillManager.

This is the heart of arms A and B (PRD D5). The paper proxy
(`metaclaw.launcher._start_skills_only`) instantiates `SkillManager` from
`metaclaw.skill_manager`; we substitute this class via a single fork point
in launcher.py (PRD D6). All other paper proxy logic — prompt template,
evolve trigger timing, buffer-turn protocol, request/response routing —
stays untouched.

Duck-typed interface (what `metaclaw.api_server.AsyncRolloutWorker` reads):
  - retrieve(task_description, top_k) -> list[dict]
  - format_for_conversation(skills) -> str
  - add_skills(new_skills, category) -> int
  - add_skill(skill) -> bool
  - skills: dict
  - generation: int
  - get_skill_count() -> dict

The two variants:
  - "skill-evolve": MIRIX server on :8531, /v1/skills/{evolve,?query=}
  - "legacy":       MIRIX server on :8532, /memory/{add_sync, search?...}
"""
from __future__ import annotations

import logging
import time
from typing import Any, Literal

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 600.0
DEFAULT_TOP_K = 6
DEFAULT_CLIENT_ID = "client-00000000-0000-4000-8000-000000000000"
DEFAULT_META_AGENT_NAME = "meta_memory_agent"

# MIRIX procedural_memory.entry_type is enum-restricted to these three values
# (see ProceduralMemoryItem schema). paper's SkillManager.category is a free
# string (e.g. "general", "common_mistakes", "communication", "deployment").
# We map paper category -> MIRIX enum, preserving original category by
# prefixing the description so search can still surface it.
MIRIX_ENTRY_TYPES = {"guide", "script", "workflow"}

def _map_paper_category_to_entry_type(category: str) -> str:
    """paper-category → MIRIX entry_type. Default 'guide' is the most generic."""
    cat = (category or "").strip().lower()
    if cat in MIRIX_ENTRY_TYPES:
        return cat
    # workflow is the right bucket for multi-step task procedures
    if cat in ("workflow", "task", "deployment", "ingestion", "pipeline"):
        return "workflow"
    # everything else (general, common_mistakes, communication, etc.) → guide
    return "guide"


def _mirix_skill_to_paper(skill: dict[str, Any]) -> dict[str, Any]:
    """MIRIX /v1/skills row -> paper SkillManager dict shape."""
    return {
        "name": skill.get("name", ""),
        "description": skill.get("description", ""),
        "content": skill.get("instructions", ""),
        "category": skill.get("entry_type") or "general",
    }


def _legacy_procedural_to_paper(row: dict[str, Any]) -> dict[str, Any]:
    """MIRIX /memory procedural_memory row -> paper SkillManager dict shape."""
    entry_type = row.get("entry_type") or "procedure"
    steps = row.get("steps")
    if isinstance(steps, list):
        content = "\n".join(str(s) for s in steps if s is not None)
    else:
        content = steps or ""
    return {
        "name": entry_type,
        "description": row.get("summary") or "",
        "content": content,
        "category": entry_type,
    }


def _round_to_message(round_record: dict[str, Any]) -> str:
    """Serialize a round outcome into a single evolve-message string.

    Mirrors archive `format_adapter.round_to_message` but takes a raw dict
    rather than a RoundResult dataclass so it can ingest paper's own
    `infer_result.json` shape directly.
    """
    rid = round_record.get("round_id") or round_record.get("id", "?")
    rtype = round_record.get("round_type") or round_record.get("type", "?")
    passed = round_record.get("passed")
    if passed is None:
        passed = round_record.get("reward", 0) >= 1.0
    status = "PASS" if passed else "FAIL"
    question = round_record.get("question", "").strip()
    answer = (round_record.get("final_answer")
              or round_record.get("answer")
              or "").strip()
    feedback = (round_record.get("feedback") or "").strip()
    parts = [
        f"### Round {rid}  [{rtype}]  outcome={status}",
        "",
        "**Question:**",
        question,
        "",
        "**Agent final answer:**",
        answer or "(empty)",
        "",
        "**Bench feedback:**",
        feedback or "(no feedback)",
    ]
    err = round_record.get("error")
    if err:
        parts += ["", f"**Error:** {err}"]
    return "\n".join(parts)


class MirixSkillsAdapter:
    """Drop-in replacement for paper's SkillManager backed by MIRIX REST API.

    Sync interface, matching paper's SkillManager. Internally uses a sync
    httpx.Client (not async) because paper invokes retrieve() inside its
    async request handler without await — we preserve that semantic so the
    fork remains a one-line swap rather than a refactor.

    Latency note: paper's retrieve() reads local files in ~1ms; our HTTP
    call is ~10-50ms. This is a known wallclock cost, not a correctness
    deviation.
    """

    def __init__(
        self,
        variant: Literal["skill-evolve", "legacy"],
        base_url: str,
        user_id: str,
        client_id: str = DEFAULT_CLIENT_ID,
        top_k: int = DEFAULT_TOP_K,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        if variant not in ("skill-evolve", "legacy"):
            raise ValueError(f"variant must be 'skill-evolve' or 'legacy', got {variant!r}")
        self.variant = variant
        self.base_url = base_url.rstrip("/")
        # user_id passed in is the human-readable name; server assigns its own
        # internal id. _ensure_user_id() does a /users/create_or_get on first
        # use to resolve human-name -> server.user.id and caches it.
        self.user_name = user_id
        self._resolved_user_id: str | None = None
        self.client_id = client_id
        self.top_k = top_k
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=timeout_s,
            headers={"X-Client-Id": self.client_id},
        )
        self._meta_agent_id: str | None = None

        # Paper SkillManager exposes these attributes; AsyncRolloutWorker
        # reads .skills and .generation directly (api_server.py:2134, 2334).
        # `.skills` is exposed as a @property below — it queries MIRIX
        # backend live each access so the paper evolver's dedup context
        # reflects real backend state (codex parity review confound #2).
        self.generation: int = 0
        self._added_names: set[str] = set()

    def close(self) -> None:
        self._http.close()

    # -- paper's .skills surface: live view of MIRIX backend ----------

    @property
    def skills(self) -> dict[str, Any]:
        """Live snapshot of the MIRIX skill bank, bucketed into paper's
        SkillManager.skills shape so paper's evolver dedup logic
        (api_server.py:2134 `existing = self.skill_manager.skills`)
        sees real backend state, not a stub. Codex parity review fix #2.

        Each access queries the backend. Cheap enough for the evolve
        path (once per test); for the retrieve path paper does NOT read
        .skills, only .retrieve(query, top_k).
        """
        bucketed: dict[str, Any] = {
            "general_skills": [],
            "task_specific_skills": {},
            "common_mistakes": [],
        }
        try:
            rows = self._fetch_all_skills_paper_shape()
        except (httpx.HTTPError, KeyError) as e:
            logger.warning("[MirixSkillsAdapter:%s] .skills backend query failed: %s",
                           self.variant, e)
            return bucketed

        for s in rows:
            cat = s.get("category", "general")
            if cat == "general":
                bucketed["general_skills"].append(s)
            elif cat == "common_mistakes":
                bucketed["common_mistakes"].append(s)
            else:
                bucketed["task_specific_skills"].setdefault(cat, []).append(s)
        return bucketed

    @skills.setter
    def skills(self, _value):
        """Paper SkillManager has self.skills = dict; some downstream code
        may try to assign. We're backend-backed so just no-op the setter
        rather than raise, to keep duck-type compatibility."""
        pass

    def _fetch_all_skills_paper_shape(self) -> list[dict]:
        """Return every skill in the MIRIX bank as a list of paper-shape
        dicts {name, description, content, category}. Decodes original
        paper category from the description prefix we set in add_skill."""
        uid = self._ensure_user_id()
        if self.variant == "skill-evolve":
            # Unified search interface (GET /v1/skills removed). search_method=""
            # -> server per-type default (procedural -> hybrid, env-overridable).
            # Procedural rows are a superset of the old skill rows, so
            # _mirix_skill_to_paper + the category restore below are unchanged.
            resp = self._http.get(
                "/memory/search",
                params={
                    "memory_type": "procedural",
                    "query": "",
                    "limit": 500,
                    "search_field": "description",
                    "search_method": "",
                    "user_id": uid,
                },
            )
            resp.raise_for_status()
            payload = resp.json()
            raw = payload.get("results") if isinstance(payload, dict) else payload
            out = []
            for s in raw or []:
                paper = _mirix_skill_to_paper(s)
                # Restore original paper category from description prefix
                desc = paper.get("description", "")
                if desc.startswith("[paper-category="):
                    end = desc.find("]")
                    if end > 0:
                        cat = desc[len("[paper-category="):end]
                        paper["category"] = cat
                        paper["description"] = desc[end + 2:] if end + 2 < len(desc) else ""
                out.append(paper)
            return out
        # legacy
        resp = self._http.get(
            "/memory/search",
            params={
                "memory_type": "procedural",
                "search_method": "bm25",
                "search_field": "summary",
                "query": "",
                "limit": 500,
                "user_id": uid,
            },
        )
        resp.raise_for_status()
        payload = resp.json()
        raw = payload.get("results") if isinstance(payload, dict) else payload
        return [_legacy_procedural_to_paper(r) for r in (raw or [])]

    # -- paper SkillManager interface ----------------------------------

    def retrieve(self, task_description: str, top_k: int | None = None) -> list[dict]:
        """Retrieve skills relevant to *task_description*. Sync.

        Errors degrade to []. paper proxy must not raise on retrieve.
        """
        k = top_k or self.top_k
        try:
            uid = self._ensure_user_id()
            if self.variant == "skill-evolve":
                # Unified search interface (GET /v1/skills removed);
                # search_method="" -> server per-type default (hybrid).
                resp = self._http.get(
                    "/memory/search",
                    params={
                        "memory_type": "procedural",
                        "query": task_description,
                        "limit": k,
                        "search_field": "description",
                        "search_method": "",
                        "user_id": uid,
                    },
                )
                resp.raise_for_status()
                payload = resp.json()
                rows = payload.get("results") if isinstance(payload, dict) else payload
                return [_mirix_skill_to_paper(s) for s in (rows or [])]
            else:  # legacy
                resp = self._http.get(
                    "/memory/search",
                    params={
                        "memory_type": "procedural",
                        "search_method": "bm25",
                        "search_field": "summary",
                        "query": task_description,
                        "limit": k,
                        "user_id": uid,
                    },
                )
                resp.raise_for_status()
                payload = resp.json()
                rows = payload.get("results") if isinstance(payload, dict) else payload
                return [_legacy_procedural_to_paper(r) for r in (rows or [])]
        except (httpx.HTTPError, httpx.HTTPStatusError) as e:
            logger.warning("[MirixSkillsAdapter:%s] retrieve failed: %s", self.variant, e)
            return []

    def retrieve_relevant(self, task_description: str, top_k: int = 6, min_relevance: float = 0.07) -> list[dict]:
        """Alias for retrieve(). paper's retrieve_relevant() does keyword
        filtering on top of retrieve(); MIRIX backends do their own ranking
        server-side so we just call retrieve() and pass through."""
        return self.retrieve(task_description, top_k=top_k)

    def format_for_conversation(self, skills: list[dict]) -> str:
        """Render skills into a system-prompt block. Identical to paper's
        implementation (we copy the template verbatim to keep prompt-level
        behavior aligned across arms A/B and D)."""
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

    def add_skill(self, skill: dict) -> bool:
        """Ingest a single round outcome (paper calls add_skill on each
        skill returned by the LLM evolver). We POST to MIRIX evolve."""
        name = (skill.get("name") or "").strip()
        if not name:
            return False
        if name in self._added_names:
            return False
        try:
            uid = self._ensure_user_id()
            if self.variant == "skill-evolve":
                # POST /v1/skills (create_skill, status 201) — paper's
                # SkillManager.add_skill() takes an already-evolved skill dict
                # and stores it. MIRIX's /v1/skills/evolve takes raw messages
                # and runs server-side LLM evolve, which is a different
                # contract; we'd lose paper's own evolver output if we used
                # that. So we map add_skill -> POST /v1/skills directly.
                paper_cat = skill.get("category", "general")
                entry_type = _map_paper_category_to_entry_type(paper_cat)
                # Preserve original paper category in description so it
                # remains searchable / restorable after the enum coercion.
                desc = skill.get("description", "")
                if paper_cat and paper_cat not in MIRIX_ENTRY_TYPES:
                    desc = f"[paper-category={paper_cat}] {desc}"
                resp = self._http.post(
                    "/v1/skills",
                    json={
                        "name": name,
                        "description": desc,
                        "instructions": skill.get("content", ""),
                        "entry_type": entry_type,
                        "user_id": uid,
                    },
                )
                # 409 = skill with same name exists; treat as no-op (paper's
                # add_skill returns False for duplicates anyway).
                if resp.status_code == 409:
                    logger.info("[MirixSkillsAdapter:skill-evolve] dup skill %s (server 409)", name)
                    return False
                resp.raise_for_status()
            else:  # legacy
                # archive R1 mitigation: meta_agent will silently drop
                # synthesized messages it doesn't read as "task to extract".
                # Explicit reflection prefix tells the procedural_memory_agent
                # to actually call procedural_memory_insert. Without this,
                # add_sync returns 200 but writes 0 rows (the failure mode
                # that produced 0 procedural rows in the first gating run).
                self._ensure_meta_agent()
                payload_text = (
                    "Reflect on the following skill and store it as a "
                    "procedural memory entry. Call procedural_memory_insert "
                    "with the summary and steps below.\n\n"
                    f"### Skill: {name}\n\n"
                    f"**Description (summary):**\n{skill.get('description', '')}\n\n"
                    f"**Instructions (steps):**\n{skill.get('content', '')}\n\n"
                    f"**Category (entry_type):** {skill.get('category', 'general')}"
                )
                self._http.post(
                    "/memory/add_sync",
                    json={
                        "meta_agent_id": self._meta_agent_id,
                        "messages": [{"role": "user", "content": payload_text}],
                        "user_id": uid,
                        "chaining": True,
                    },
                ).raise_for_status()
            self._added_names.add(name)
            return True
        except (httpx.HTTPError, httpx.HTTPStatusError) as e:
            logger.warning("[MirixSkillsAdapter:%s] add_skill failed for %s: %s",
                           self.variant, name, e)
            return False

    def add_skills(self, new_skills: list[dict], category: str = "general") -> int:
        """Batch ingest. Mirrors paper SkillManager.add_skills semantics:
        increments generation when ≥1 skill added."""
        added = 0
        for skill in new_skills:
            payload = skill if "category" in skill else {**skill, "category": category}
            if self.add_skill(payload):
                added += 1
        if added > 0:
            self.generation += 1
        return added

    def get_skill_count(self) -> dict:
        return {
            "general": len(self.skills.get("general_skills", [])),
            "task_specific": sum(
                len(v) for v in self.skills.get("task_specific_skills", {}).values()
            ),
            "common_mistakes": len(self.skills.get("common_mistakes", [])),
            "total": (
                len(self.skills.get("general_skills", []))
                + sum(len(v) for v in self.skills.get("task_specific_skills", {}).values())
                + len(self.skills.get("common_mistakes", []))
            ),
        }

    def reload(self) -> None:
        """No-op. paper's SkillManager.reload() re-scans skills_dir; we have
        no local store to re-scan."""
        return None

    def save(self, path: str | None = None) -> None:
        """No-op. paper's save() flushes skill bank to disk; MIRIX persists
        server-side automatically."""
        return None

    # -- internal ------------------------------------------------------

    def _ensure_user_id(self) -> str:
        """Resolve human-readable user name to a stable MIRIX user.id.

        Idempotent contract: POST /users/create_or_get with `user_id` set to
        our human-readable name. MIRIX server treats user_id as the lookup
        key (CreateOrGetUserRequest.user_id) — if a user with that id exists,
        return it; otherwise create with exactly that id. This makes
        adapter.user_name == server.user.id, so:

          - same arm across multiple process restarts → same user → skills
            accumulate across runs
          - 30-day main run can resume from a crash without losing the
            evolved skill bank

        Caching the resolved id locally is still correct (avoids the round-trip)
        but it's now also safe because the id is deterministic from user_name.
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
            logger.info("[MirixSkillsAdapter:%s] resolved user_name=%s -> %s",
                        self.variant, self.user_name, self._resolved_user_id)
            return self._resolved_user_id
        except (httpx.HTTPError, KeyError) as e:
            logger.warning("[MirixSkillsAdapter:%s] _ensure_user_id failed: %s",
                           self.variant, e)
            return self.user_name

    def _ensure_meta_agent(self) -> None:
        """Resolve meta_memory_agent id once (legacy variant only)."""
        if self._meta_agent_id:
            return
        for attempt in range(5):
            try:
                resp = self._http.get("/agents")
                resp.raise_for_status()
                agents = resp.json()
                if isinstance(agents, dict):
                    agents = agents.get("agents") or agents.get("results") or []
                for a in agents:
                    if a.get("name") == DEFAULT_META_AGENT_NAME:
                        self._meta_agent_id = a.get("id")
                        return
                raise RuntimeError(f"{DEFAULT_META_AGENT_NAME} not found in /agents")
            except (httpx.HTTPError, RuntimeError) as e:
                if attempt == 4:
                    raise
                logger.warning("[MirixSkillsAdapter:legacy] _ensure_meta_agent retry %d: %s",
                               attempt, e)
                time.sleep(1.0 * (attempt + 1))
