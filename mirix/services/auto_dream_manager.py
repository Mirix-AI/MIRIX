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
        # Pass user_id/client_id explicitly — create_episodic_memory does NOT read them
        # off the EpisodicEvent object; without these, the checkpoint is written under
        # ADMIN_USER_ID and per-user get_last_dream_time() can never find it (causing
        # the dream window to silently fall back to "now - 30 days" every run).
        await mgr.create_episodic_memory(
            episodic_memory=checkpoint,
            actor=actor,
            client_id=actor.id,
            user_id=user.id,
        )

    # ------------------------------------------------------------------ #
    # Memory fetching                                                      #
    # ------------------------------------------------------------------ #
    #
    # Window semantics: the dream window expresses "memories ingested since
    # the last dream finished". That maps to the row's `created_at` (real
    # wall-clock time the entry was inserted), NOT `occurred_at` (the business
    # timestamp the LLM extracted from the source content — for LoComo data
    # this is e.g. "8 May 2023" even though wall-clock today is 2026).
    #
    # The underlying list_* APIs treat their start_date/end_date as filters on
    # occurred_at, so we do NOT pass them here. Instead we fetch the per-user
    # candidate set and filter by created_at in Python.

    @staticmethod
    def _within_created_window(
        item, start_date: Optional[dt.datetime], end_date: Optional[dt.datetime]
    ) -> bool:
        ts = getattr(item, "created_at", None)
        if ts is None:
            return True  # no timestamp → don't drop
        # Strip tzinfo for naive vs naive comparison (DB stores naive UTC)
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.astimezone(dt.timezone.utc).replace(tzinfo=None)
        if start_date is not None and ts < start_date:
            return False
        if end_date is not None and ts > end_date:
            return False
        return True

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
            search_method="string_match",
            query="",
            limit=500,
            use_cache=False,
        )
        return [
            e for e in events
            if e.event_type != _CHECKPOINT_EVENT_TYPE
            and self._within_created_window(e, start_date, end_date)
        ]

    async def _fetch_semantic(
        self,
        user: PydanticUser,
        agent_state: AgentState,
        start_date: dt.datetime,
        end_date: dt.datetime,
    ) -> list:
        from mirix.services.semantic_memory_manager import SemanticMemoryManager

        mgr = SemanticMemoryManager()
        items = await mgr.list_semantic_items(
            user=user,
            agent_state=agent_state,
            search_method="string_match",
            query="",
            limit=500,
            use_cache=False,
        )
        return [i for i in items if self._within_created_window(i, start_date, end_date)]

    async def _fetch_procedural(
        self,
        user: PydanticUser,
        agent_state: AgentState,
        start_date: dt.datetime,
        end_date: dt.datetime,
    ) -> list:
        from mirix.services.procedural_memory_manager import ProceduralMemoryManager

        mgr = ProceduralMemoryManager()
        items = await mgr.list_procedures(
            user=user,
            agent_state=agent_state,
            search_method="string_match",
            query="",
            limit=500,
            use_cache=False,
        )
        return [i for i in items if self._within_created_window(i, start_date, end_date)]

    async def _fetch_resource(
        self,
        user: PydanticUser,
        agent_state: AgentState,
        start_date: dt.datetime,
        end_date: dt.datetime,
    ) -> list:
        from mirix.services.resource_memory_manager import ResourceMemoryManager

        mgr = ResourceMemoryManager()
        items = await mgr.list_resources(
            user=user,
            agent_state=agent_state,
            search_method="string_match",
            query="",
            limit=500,
            use_cache=False,
        )
        return [i for i in items if self._within_created_window(i, start_date, end_date)]

    async def _fetch_knowledge_vault(
        self,
        user: PydanticUser,
        agent_state: AgentState,
        start_date: dt.datetime,
        end_date: dt.datetime,
    ) -> list:
        from mirix.services.knowledge_vault_manager import KnowledgeVaultManager

        mgr = KnowledgeVaultManager()
        items = await mgr.list_knowledge(
            user=user,
            agent_state=agent_state,
            search_method="string_match",
            query="",
            limit=500,
            use_cache=False,
        )
        return [i for i in items if self._within_created_window(i, start_date, end_date)]

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
            if child.name == "auto_dream_agent":
                return child

        agent_create = CreateAgent(
            name="auto_dream_agent",
            agent_type=AgentType.auto_dream_agent,
            llm_config=meta_agent_state.llm_config,
            embedding_config=meta_agent_state.embedding_config,
            parent_id=meta_agent_state.id,
        )
        return await server.agent_manager.create_agent(agent_create=agent_create, actor=actor)

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

        logger.info("Auto dream: window %s → %s, types=%s", start_date, end_date, request.memory_types)

        # -- fetch memories --
        fetch_map = {
            "episodic": self._fetch_episodic,
            "semantic": self._fetch_semantic,
            "procedural": self._fetch_procedural,
            "resource": self._fetch_resource,
            "knowledge_vault": self._fetch_knowledge_vault,
        }
        memories: dict = {}
        for mem_type in request.memory_types:
            if mem_type in fetch_map:
                items = await fetch_map[mem_type](user, meta_agent_state, start_date, end_date)
                memories[mem_type] = items
                logger.info("  %s: fetched %d items", mem_type, len(items))

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
        payload = _format_memories_as_message(
            memories, start_date, end_date, raw_sessions=request.raw_sessions
        )

        # -- get or create agent state --
        dream_agent_state = await self.get_or_create_dream_agent_state(actor, meta_agent_state)

        # -- load and run agent --
        dream_agent = await server.load_agent(
            agent_id=dream_agent_state.id,
            actor=actor,
            user=user,
            use_cache=False,
        )

        # override model if caller requested it (e.g. for testing). Must happen
        # AFTER load_agent: load_agent re-reads agent state from DB, so any
        # pre-load mutation would be discarded. Also update Agent.self.model
        # (used for logging / message recording) — LLM call itself reads
        # self.agent_state.llm_config.
        if request.model:
            dream_agent.agent_state.llm_config.model = request.model
            dream_agent.model = request.model
        if request.temperature is not None:
            dream_agent.agent_state.llm_config.temperature = request.temperature
        input_msg = MessageCreate(role=MessageRole.user, content=payload)
        usage_stats = await dream_agent.step(input_messages=input_msg, actor=actor, user=user)

        # -- write checkpoint --
        # IMPORTANT: use post-dream wall clock so the checkpoint occurred_at sits AFTER
        # any items the dream agent created/merged during this run. Next dream then uses
        # this as start_date and skips the dream's own outputs (prevents window self-pollution).
        post_dream_now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        await self.write_checkpoint(user, actor, meta_agent_state, post_dream_now)

        # -- build response (stats are approximate: we report totals fetched) --
        processed = {t: MemoryTypeStats(total=len(items)) for t, items in memories.items()}
        return AutoDreamResponse(
            start_date=start_date,
            end_date=end_date,
            processed=processed,
            last_dream_at=post_dream_now,
            dry_run=False,
            message="Auto dream completed.",
            usage=usage_stats,
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
    raw_sessions: Optional[List[str]] = None,
) -> str:
    lines = [
        f"Time window: {start_date.isoformat()} → {end_date.isoformat()}",
        "",
        "Please review the following memories for redundancy and conflicts.",
        "Process each type in order: episodic → semantic → procedural → resource → knowledge_vault.",
        "Call finish_memory_update when done.",
        "",
    ]

    # First-dream cheating: when raw_sessions are supplied (typically only on the
    # very first dream, before consolidated memories accrue), prepend the raw
    # conversation text so the agent can ground its consolidation on source dialogue.
    if raw_sessions:
        lines.append(f"=== RAW CONVERSATIONS ({len(raw_sessions)} sessions, reference only) ===")
        lines.append(
            "These are the source conversations behind the memories below. "
            "Use them to disambiguate or enrich merge decisions. "
            "Do NOT call insert tools on the raw text directly — operate only on the memory items."
        )
        for idx, raw in enumerate(raw_sessions, start=1):
            lines.append(f"--- raw_session_{idx} ---")
            lines.append(raw)
        lines.append("")

    for mem_type, items in memories.items():
        lines.append(f"=== {mem_type.upper()} MEMORY ({len(items)} items) ===")
        if not items:
            lines.append("(empty)")
        else:
            serialized = [_serialize_item(item) for item in items]
            lines.append(json.dumps(serialized, ensure_ascii=False, default=str, indent=2))
        lines.append("")
    return "\n".join(lines)
