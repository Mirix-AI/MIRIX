"""
AutoDreamManager: orchestrates the auto_dream self-reflection pipeline.

Flow:
  1. Resolve time window (last dream checkpoint → now)
  2. Fetch memories for each requested type
  3. Format memories into a structured message
  4. Invoke AutoDreamAgent via step()
  5. Write a checkpoint episodic memory entry
  6. Return stats
"""

import datetime as dt
import json
import logging
import os
from typing import List, Optional

from mirix.schemas.agent import AgentState, AgentType, CreateAgent
from mirix.schemas.auto_dream import AutoDreamRequest, AutoDreamResponse, MemoryTypeStats
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.message import MessageCreate
from mirix.schemas.enums import MessageRole
from mirix.schemas.user import User as PydanticUser

logger = logging.getLogger(__name__)

# event_type used to tag auto_dream checkpoint records in episodic memory
_CHECKPOINT_EVENT_TYPE = "auto_dream_checkpoint"

# Generic "fetch current memories of these components -> one consolidation agent"
# modes. mode='procedural' is intentionally NOT here: it is handled by the
# dedicated session-experience path (_run_procedural_experience), which returns
# before this mapping is ever consulted.
_MODE_COMPONENTS = {
    "core": ["core"],
    "episodic": ["episodic"],
    "semantic": ["semantic"],
    "resource": ["resource"],
    "knowledge": ["knowledge"],
    "experience": ["episodic", "semantic", "knowledge"],
}


def _load_mode_system_prompt(mode: str) -> str:
    prompt_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "prompts",
        "system",
        "base",
        "auto_dream_agent",
        f"{mode}.txt",
    )
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


class AutoDreamManager:
    # ------------------------------------------------------------------ #
    # Checkpoint helpers                                                   #
    # ------------------------------------------------------------------ #

    async def get_last_dream_time(
        self,
        user: PydanticUser,
        actor: PydanticClient,
        agent_state: AgentState,
    ) -> Optional[dt.datetime]:
        """Return occurred_at of the most recent auto_dream checkpoint, or None."""
        from mirix.services.episodic_memory_manager import EpisodicMemoryManager

        mgr = EpisodicMemoryManager()
        events = await mgr.list_episodic_memory(
            user=user,
            agent_state=agent_state,
            query=_CHECKPOINT_EVENT_TYPE,
            search_method="string_match",
            search_field="event_type",
            limit=1,
            use_cache=False,
        )
        for ev in events:
            if ev.event_type == _CHECKPOINT_EVENT_TYPE:
                return ev.occurred_at
        return None

    async def write_checkpoint(
        self,
        user: PydanticUser,
        actor: PydanticClient,
        agent_state: AgentState,
        dream_time: dt.datetime,
    ) -> None:
        from mirix.schemas.episodic_memory import EpisodicEvent
        from mirix.services.episodic_memory_manager import EpisodicMemoryManager

        import uuid

        mgr = EpisodicMemoryManager()
        checkpoint = EpisodicEvent(
            id=f"ep_{uuid.uuid4().hex[:12]}",
            occurred_at=dream_time,
            actor="system",
            event_type=_CHECKPOINT_EVENT_TYPE,
            summary="Auto dream completed",
            details=f"Auto dream run finished at {dream_time.isoformat()}",
            filter_tags={"type": "system", "source": "auto_dream"},
            user_id=user.id,
            organization_id=actor.organization_id,
        )
        await mgr.create_episodic_memory(episodic_memory=checkpoint, actor=actor)

    # ------------------------------------------------------------------ #
    # Memory fetching                                                      #
    # ------------------------------------------------------------------ #

    async def _fetch_episodic(
        self,
        user: PydanticUser,
        agent_state: AgentState,
        start_date: dt.datetime,
        end_date: dt.datetime,
    ) -> list:
        from mirix.services.episodic_memory_manager import EpisodicMemoryManager

        mgr = EpisodicMemoryManager()
        events = await mgr.list_episodic_memory(
            user=user,
            agent_state=agent_state,
            # Auto-dream fetches ALL current memories regardless of date;
            # the passed window is only recorded in the response for reference.
            start_date=None,
            end_date=None,
            search_method="string_match",
            query="",
            limit=500,
            use_cache=False,
        )
        # exclude system checkpoints
        return [e for e in events if e.event_type != _CHECKPOINT_EVENT_TYPE]

    async def _fetch_core(
        self,
        user: PydanticUser,
        agent_state: AgentState,
        start_date: dt.datetime,
        end_date: dt.datetime,
        actor: PydanticClient,
    ) -> list:
        from mirix.services.block_manager import BlockManager

        mgr = BlockManager()
        return await mgr.get_blocks(
            user=user,
            any_scopes=actor.read_scopes,
            limit=500,
            auto_create_from_default=False,
        )

    async def _fetch_semantic(
        self,
        user: PydanticUser,
        agent_state: AgentState,
        start_date: dt.datetime,
        end_date: dt.datetime,
    ) -> list:
        from mirix.services.semantic_memory_manager import SemanticMemoryManager

        mgr = SemanticMemoryManager()
        return await mgr.list_semantic_items(
            user=user,
            agent_state=agent_state,
            search_method="string_match",
            query="",
            limit=500,
            use_cache=False,
        )

    async def _fetch_procedural(
        self,
        user: PydanticUser,
        agent_state: AgentState,
        start_date: dt.datetime,
        end_date: dt.datetime,
    ) -> list:
        from mirix.services.procedural_memory_manager import ProceduralMemoryManager

        mgr = ProceduralMemoryManager()
        return await mgr.list_procedures(
            user=user,
            agent_state=agent_state,
            search_method="string_match",
            query="",
            limit=500,
            use_cache=False,
        )

    async def _fetch_resource(
        self,
        user: PydanticUser,
        agent_state: AgentState,
        start_date: dt.datetime,
        end_date: dt.datetime,
    ) -> list:
        from mirix.services.resource_memory_manager import ResourceMemoryManager

        mgr = ResourceMemoryManager()
        return await mgr.list_resources(
            user=user,
            agent_state=agent_state,
            search_method="string_match",
            query="",
            limit=500,
            use_cache=False,
        )

    async def _fetch_knowledge_vault(
        self,
        user: PydanticUser,
        agent_state: AgentState,
        start_date: dt.datetime,
        end_date: dt.datetime,
    ) -> list:
        from mirix.services.knowledge_vault_manager import KnowledgeVaultManager

        mgr = KnowledgeVaultManager()
        return await mgr.list_knowledge(
            user=user,
            agent_state=agent_state,
            search_method="string_match",
            query="",
            limit=500,
            use_cache=False,
        )

    # ------------------------------------------------------------------ #
    # Agent management                                                     #
    # ------------------------------------------------------------------ #

    async def get_or_create_dream_agent_state(
        self,
        actor: PydanticClient,
        meta_agent_state: AgentState,
    ) -> AgentState:
        """Return the auto_dream_agent state for this client, creating it if needed."""
        from mirix.server.rest_api import get_server

        server = get_server()
        children = await server.agent_manager.list_agents(
            actor=actor,
            parent_id=meta_agent_state.id,
        )
        for child in children:
            if child.agent_type == AgentType.auto_dream_agent:
                await server.agent_manager.update_agent_tools_and_system_prompts(child.id, actor=actor)
                child = await server.agent_manager.get_agent_by_id(agent_id=child.id, actor=actor)
                return child

        agent_create = CreateAgent(
            name=f"{meta_agent_state.name}_auto_dream_agent",
            agent_type=AgentType.auto_dream_agent,
            llm_config=meta_agent_state.llm_config,
            embedding_config=meta_agent_state.embedding_config,
            parent_id=meta_agent_state.id,
        )
        return await server.agent_manager.create_agent(agent_create=agent_create, actor=actor)

    # ------------------------------------------------------------------ #
    # Goal 2/3 — general session-experience procedural path                #
    # ------------------------------------------------------------------ #

    async def _run_procedural_experience(
        self,
        *,
        request: AutoDreamRequest,
        user: PydanticUser,
        actor: PydanticClient,
        meta_agent_state: AgentState,
        now: dt.datetime,
    ) -> AutoDreamResponse:
        """Goal-2 distillation (+ Goal-3 evolution unless dry_run).

        This is the explicit "program call" entry point for general
        session-experience distillation: calling
        ``AutoDreamManager().run(AutoDreamRequest(mode='procedural', ...))``
        runs it directly. It is ALSO fired every N sessions by the
        ``trigger_memory_update`` claim path (see
        ``functions/function_sets/memory_tools.py``).

        Steps:
          1. Resolve N (``request.last_n_sessions`` or
             ``MESSAGE_RETAIN_LAST_N_SESSIONS``).
          2. Enumerate the meta agent's most-recent N retained sessions.
          3. Distill each session IN PARALLEL into pending SkillExperience rows.
          4. dry_run → return counts; else run Goal-3 skill evolution over the
             freshly-pending experiences, then write the dream checkpoint.
        """
        from mirix.constants import MESSAGE_RETAIN_LAST_N_SESSIONS
        from mirix.services.session_experience_distiller import (
            SessionExperienceDistiller,
        )

        n = request.last_n_sessions or MESSAGE_RETAIN_LAST_N_SESSIONS

        # The dream agent inherits the meta agent's llm_config; allow a model
        # override for testing (same convention as the generic path).
        llm_config = meta_agent_state.llm_config
        if request.model:
            from copy import deepcopy

            llm_config = deepcopy(llm_config)
            llm_config.model = request.model

        distiller = SessionExperienceDistiller(llm_config=llm_config)
        session_ids = await distiller.enumerate_last_n_sessions(
            agent_id=meta_agent_state.id,
            user_id=user.id,
            n=n,
        )
        logger.info(
            "Auto dream (procedural): meta_agent=%s, last_n=%d, sessions=%s",
            meta_agent_state.id,
            n,
            session_ids,
        )

        existing_skills = await self._existing_skill_summaries(user, meta_agent_state)
        experiences = await distiller.distill_sessions(
            meta_agent_state=meta_agent_state,
            user=user,
            actor=actor,
            session_ids=session_ids,
            existing_skills=existing_skills,
        )

        stats = {
            "procedural": MemoryTypeStats(total=len(experiences)),
        }

        if request.dry_run:
            return AutoDreamResponse(
                start_date=None,
                end_date=None,
                processed=stats,
                last_dream_at=now,
                dry_run=True,
                message=(
                    f"Dry run — distilled {len(experiences)} experience(s) from "
                    f"{len(session_ids)} session(s); Goal-3 evolution skipped."
                ),
            )

        # -- Goal 3: evolve skills from the freshly-pending experiences. --
        skills_changed = 0
        evolution_changes: dict = {}
        # Nothing distilled this round -> nothing to evolve. Skip outright so we
        # never resolve/refresh the procedural agent for an empty batch (scoping
        # to [] would B_min=0-skip anyway, but the agent plumbing would still run).
        if experiences:
            try:
                from mirix.services.skill_experience_curator import (
                    run_experience_evolution,
                )

                evolution = await run_experience_evolution(
                    user=user,
                    actor=actor,
                    meta_agent_state=meta_agent_state,
                    # Scope evolution to THIS round's freshly-distilled experiences
                    # only — never the accumulated cross-round pending pool. Earlier
                    # rounds' leftover/low-signal experiences would dilute the curator
                    # prompt and hurt skill quality; the docstring's "freshly-pending"
                    # contract is now enforced by passing exactly the ids we just made.
                    experience_ids=[e.id for e in experiences],
                )
                skills_changed = (evolution or {}).get("skills_changed", 0)
                evolution_changes = (evolution or {}).get("changes", {}) or {}
            except ImportError:
                # Goal-3 curator not yet present in this build — distillation alone
                # is still valuable, so do not fail the run.
                logger.info(
                    "Goal-3 experience curator unavailable; experiences persisted as pending."
                )
            except Exception as e:  # noqa: BLE001 — evolution failure mustn't lose experiences
                logger.warning("Goal-3 experience evolution failed: %s", e)

        await self.write_checkpoint(user, actor, meta_agent_state, now)

        return AutoDreamResponse(
            start_date=None,
            end_date=None,
            processed=stats,
            last_dream_at=now,
            dry_run=False,
            skills_changed=skills_changed,
            changes=evolution_changes,
            message=(
                f"Auto dream (procedural) completed: {len(experiences)} experience(s) "
                f"from {len(session_ids)} session(s); {skills_changed} skill(s) changed."
            ),
        )

    async def _existing_skill_summaries(
        self,
        user: PydanticUser,
        meta_agent_state: AgentState,
    ) -> list:
        """Return [{name, description}] of existing procedural skills (dedup context).

        Best-effort: a failure here just yields an empty list (the distiller
        prompt treats it as "(none)").
        """
        try:
            procedures = await self._fetch_procedural(user, meta_agent_state, None, None)
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not fetch existing skills for dedup context: %s", e)
            return []
        summaries = []
        for p in procedures:
            summaries.append(
                {
                    "name": getattr(p, "name", "") or "",
                    "description": getattr(p, "description", "") or "",
                }
            )
        return summaries

    # ------------------------------------------------------------------ #
    # Main entry point                                                     #
    # ------------------------------------------------------------------ #

    async def run(
        self,
        request: AutoDreamRequest,
        user: PydanticUser,
        actor: PydanticClient,
        meta_agent_state: AgentState,
    ) -> AutoDreamResponse:
        from mirix.agent.auto_dream_agent import AutoDreamAgent
        from mirix.server.rest_api import get_server

        server = get_server()
        now = dt.datetime.now(dt.timezone.utc)

        # ----------------------------------------------------------------- #
        # Goal 2/3 — mode='procedural' is general per-session experience    #
        # distillation (NOT the legacy procedure-consolidation pass). It    #
        # reads the meta agent's last-N RETAINED sessions, distills each    #
        # session's transcript IN PARALLEL into pending SkillExperience     #
        # rows, then (Goal 3, unless dry_run) drives skill self-evolution   #
        # from those experiences. Other modes fall through to the generic   #
        # fetch → format → single-step path below, unchanged.               #
        # ----------------------------------------------------------------- #
        if request.mode == "procedural":
            return await self._run_procedural_experience(
                request=request,
                user=user,
                actor=actor,
                meta_agent_state=meta_agent_state,
                now=now,
            )

        # -- resolve time window --
        end_date = request.end_date or now
        if request.start_date:
            start_date = request.start_date
        else:
            last = await self.get_last_dream_time(user, actor, meta_agent_state)
            start_date = last or (now - dt.timedelta(days=30))

        # DB columns are TIMESTAMP WITHOUT TIME ZONE; strip tzinfo
        if start_date.tzinfo is not None:
            start_date = start_date.astimezone(dt.timezone.utc).replace(tzinfo=None)
        if end_date.tzinfo is not None:
            end_date = end_date.astimezone(dt.timezone.utc).replace(tzinfo=None)

        components = _MODE_COMPONENTS[request.mode]
        logger.info("Auto dream: window %s → %s, mode=%s, components=%s", start_date, end_date, request.mode, components)

        # -- fetch memories --
        fetch_map = {
            "core": self._fetch_core,
            "episodic": self._fetch_episodic,
            "semantic": self._fetch_semantic,
            "procedural": self._fetch_procedural,
            "resource": self._fetch_resource,
            "knowledge": self._fetch_knowledge_vault,
        }
        memories: dict = {}
        for component in components:
            fetcher = fetch_map[component]
            if component == "core":
                items = await fetcher(user, meta_agent_state, start_date, end_date, actor)
            else:
                items = await fetcher(user, meta_agent_state, start_date, end_date)
            memories[component] = items
            logger.info("  %s: fetched %d items", component, len(items))

        # -- dry_run: just return counts without invoking agent --
        if request.dry_run:
            processed = {t: MemoryTypeStats(total=len(items)) for t, items in memories.items()}
            return AutoDreamResponse(
                start_date=start_date,
                end_date=end_date,
                processed=processed,
                last_dream_at=now,
                dry_run=True,
                message="Dry run — no changes applied.",
            )

        # -- build input message for the agent --
        payload = _format_memories_as_message(memories, start_date, end_date, request.mode)

        # -- get or create agent state --
        dream_agent_state = await self.get_or_create_dream_agent_state(actor, meta_agent_state)
        mode_system_prompt = _load_mode_system_prompt(request.mode)
        if dream_agent_state.system != mode_system_prompt:
            dream_agent_state = await server.agent_manager.update_system_prompt(
                agent_id=dream_agent_state.id,
                system_prompt=mode_system_prompt,
                actor=actor,
            )

        # override model if caller requested it (e.g. for testing)
        if request.model:
            from copy import deepcopy

            dream_agent_state = deepcopy(dream_agent_state)
            dream_agent_state.llm_config.model = request.model

        # -- load and run agent --
        dream_agent = await server.load_agent(
            agent_id=dream_agent_state.id,
            actor=actor,
            user=user,
            use_cache=False,
        )
        input_msg = MessageCreate(role=MessageRole.user, content=payload)
        await dream_agent.step(input_messages=input_msg, actor=actor, user=user)

        # -- write checkpoint --
        await self.write_checkpoint(user, actor, meta_agent_state, now)

        # -- build response (stats are approximate: we report totals fetched) --
        processed = {t: MemoryTypeStats(total=len(items)) for t, items in memories.items()}
        return AutoDreamResponse(
            # Auto-dream fetches ALL current memories regardless of date;
            # the passed window is only recorded in the response for reference.
            start_date=None,
            end_date=None,
            processed=processed,
            last_dream_at=now,
            dry_run=False,
            message="Auto dream completed.",
        )


# ------------------------------------------------------------------ #
# Formatting helper                                                    #
# ------------------------------------------------------------------ #

def _serialize_item(item) -> dict:
    """Convert a Pydantic memory item to a compact dict for the LLM."""
    data = item.model_dump(exclude_none=True)
    # drop heavy embedding vectors
    for key in list(data.keys()):
        if key.endswith("_embedding"):
            del data[key]
    return data


def _format_memories_as_message(
    memories: dict,
    start_date: dt.datetime,
    end_date: dt.datetime,
    mode: str,
) -> str:
    component_labels = {
        "core": "CORE MEMORY",
        "episodic": "EPISODIC MEMORY",
        "semantic": "SEMANTIC MEMORY",
        "resource": "RESOURCE MEMORY",
        "procedural": "PROCEDURAL MEMORY",
        "knowledge": "KNOWLEDGE VAULT",
    }
    component_order = " → ".join(component_labels[mem_type] for mem_type in memories.keys())
    lines = [
        f"Mode: {mode}",
        f"Time window: {start_date.isoformat()} → {end_date.isoformat()}",
        "",
        f"Provided component(s), in order: {component_order}.",
        "",
    ]
    for mem_type, items in memories.items():
        label = component_labels[mem_type]
        lines.append(f"=== {label} ({len(items)} items) ===")
        if not items:
            lines.append("(empty)")
        else:
            serialized = [_serialize_item(item) for item in items]
            lines.append(json.dumps(serialized, ensure_ascii=False, default=str, indent=2))
        lines.append("")
    return "\n".join(lines)
