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
            # Auto-dream fetches ALL current memories regardless of date;
            # the passed window is only recorded in the response for reference.
            start_date=None,
            end_date=None,
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
            # Auto-dream fetches ALL current memories regardless of date;
            # the passed window is only recorded in the response for reference.
            start_date=None,
            end_date=None,
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
            # Auto-dream fetches ALL current memories regardless of date;
            # the passed window is only recorded in the response for reference.
            start_date=None,
            end_date=None,
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
            # Auto-dream fetches ALL current memories regardless of date;
            # the passed window is only recorded in the response for reference.
            start_date=None,
            end_date=None,
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
        payload = _format_memories_as_message(memories, start_date, end_date)

        # -- get or create agent state --
        dream_agent_state = await self.get_or_create_dream_agent_state(actor, meta_agent_state)

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
) -> str:
    lines = [
        f"Time window: {start_date.isoformat()} → {end_date.isoformat()}",
        "",
        "Please review the following memories for redundancy and conflicts.",
        "Process each type in order: episodic → semantic → procedural → resource → knowledge_vault.",
        "Call finish_memory_update when done.",
        "",
    ]
    for mem_type, items in memories.items():
        lines.append(f"=== {mem_type.upper()} MEMORY ({len(items)} items) ===")
        if not items:
            lines.append("(empty)")
        else:
            serialized = [_serialize_item(item) for item in items]
            lines.append(json.dumps(serialized, ensure_ascii=False, default=str, indent=2))
        lines.append("")
    return "\n".join(lines)
