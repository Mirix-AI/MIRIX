"""A real skill curator setup for tests: one organization, client, user and
meta agent (with its procedural child, real tools and real system prompt) in
the database of ``db_isolation.isolated_database`` — a fresh SQLite file for
unit tests, or the fresh Postgres database of
``scripts/run_pg_integration_tests.sh`` — every model call going to the
offline fake LLM server. Nothing here reaches a model provider.
"""

from __future__ import annotations

import datetime as dt
import json
import random
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

_WORDS = (
    "open check queue retry verify record report merge filter parse read write "
    "table column value error step result answer file page field option list "
    "count total search match update create delete index cache limit order"
).split()


_EPOCH = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


def words(n_chars: int, seed: int = 0) -> str:
    """Deterministic, varied text of about ``n_chars`` characters."""
    rng = random.Random(seed)
    out: List[str] = []
    size = 0
    while size < n_chars:
        word = rng.choice(_WORDS)
        out.append(word)
        size += len(word) + 1
    return " ".join(out)[:n_chars]


async def make_world(
    fake,
    *,
    context_window: int = 32_000,
    max_output_tokens: int = 4_000,
    write_scope: str = "curator-scope",
    model: str = "fake-model",
) -> SimpleNamespace:
    """Organization, client, user and a meta agent with a procedural child."""
    from mirix.orm.client import Client as ClientORM
    from mirix.orm.organization import Organization as OrganizationORM
    from mirix.orm.user import User as UserORM
    from mirix.schemas.agent import AgentType, CreateMetaAgent
    from mirix.schemas.client import Client as PydanticClient
    from mirix.schemas.embedding_config import EmbeddingConfig
    from mirix.schemas.llm_config import LLMConfig
    from mirix.schemas.user import User as PydanticUser
    from mirix.server.server import db_context, get_server

    suffix = uuid.uuid4().hex[:8]
    org_id = f"cur-org-{suffix}"
    client_id = f"cur-client-{suffix}"
    user_id = f"cur-user-{suffix}"
    async with db_context() as db:
        db.add(OrganizationORM(id=org_id, name=org_id))
        await db.commit()
    async with db_context() as db:
        db.add(
            ClientORM(
                id=client_id,
                organization_id=org_id,
                name=client_id,
                status="active",
                write_scope=write_scope,
                read_scopes=[write_scope],
            )
        )
        db.add(
            UserORM(
                id=user_id,
                name=user_id,
                organization_id=org_id,
                status="active",
                timezone="UTC",
            )
        )
        await db.commit()
    actor = PydanticClient(
        id=client_id,
        organization_id=org_id,
        name=client_id,
        write_scope=write_scope,
        read_scopes=[write_scope],
    )
    user = PydanticUser(
        id=user_id, name=user_id, organization_id=org_id, timezone="UTC"
    )
    llm_config = LLMConfig(
        model=model,
        model_endpoint_type="openai",
        model_endpoint=fake.base_url,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        api_key="fake-key",
    )
    embedding_config = EmbeddingConfig(
        embedding_model="text-embedding-3-small",
        embedding_endpoint_type="openai",
        embedding_endpoint=fake.base_url,
        embedding_dim=1536,
        api_key="fake-key",
    )
    server = get_server()
    meta = await server.agent_manager.create_meta_agent(
        CreateMetaAgent(
            agents=["procedural_memory_agent"],
            llm_config=llm_config,
            embedding_config=embedding_config,
        ),
        actor=actor,
    )
    children = await server.agent_manager.list_agents(actor=actor, parent_id=meta.id)
    [procedural] = [
        c for c in children if c.agent_type == AgentType.procedural_memory_agent
    ]
    return SimpleNamespace(
        org=org_id,
        actor=actor,
        user=user,
        meta=meta,
        procedural=procedural,
        server=server,
        scope=write_scope,
        llm_config=llm_config,
    )


_WORLD_SCOPE = object()


def procedural_agent(world, *, scope=_WORLD_SCOPE):
    """A procedural agent instance on the world's procedural child, working in
    ``scope`` (default: the world's; None: no scope restriction) — outside any
    learning run unless the caller attaches a curator batch."""
    from mirix.agent import ProceduralMemoryAgent

    scope = world.scope if scope is _WORLD_SCOPE else scope
    return ProceduralMemoryAgent(
        agent_state=world.procedural,
        interface=world.server.default_interface_factory(),
        actor=world.actor,
        user=world.user,
        filter_tags={"scope": scope} if scope else None,
    )


async def add_experience(
    world,
    *,
    title: str,
    content: str = "apply the lesson",
    quote: str = "the user confirmed it",
    signal_type: str = "user_confirmation",
    experience_type: str = "worth_learning",
    session_id: Optional[str] = None,
    importance: float = 0.7,
    credibility: float = 0.8,
):
    """A pending experience of the world. Experiences are curated in
    ``(created_at, id)`` order; SQLite stamps ``created_at`` to the second, so
    each one gets a distinct, increasing ``created_at`` here (as Postgres'
    microsecond timestamps give in practice): creation order is curation order.
    """
    from sqlalchemy import update

    from mirix.orm.skill_experience import SkillExperience as SkillExperienceORM
    from mirix.server.server import db_context
    from mirix.services.skill_experience_manager import SkillExperienceManager

    experience = await SkillExperienceManager().create_experience(
        agent_id=world.meta.id,
        user_id=world.user.id,
        organization_id=world.org,
        session_id=session_id or f"session-{uuid.uuid4().hex[:6]}",
        experience_type=experience_type,
        title=title,
        content=content,
        importance=importance,
        credibility=credibility,
        evidence=json.dumps({"quote": quote, "signal_type": signal_type}),
        created_by_id=world.actor.id,
    )
    world.clock = getattr(world, "clock", _EPOCH) + dt.timedelta(seconds=1)
    async with db_context() as db:
        await db.execute(
            update(SkillExperienceORM)
            .where(SkillExperienceORM.id == experience.id)
            .values(created_at=world.clock)
        )
        await db.commit()
    return experience


async def start_run(world) -> str:
    """A learning run of the world's scope, queued -> running."""
    from mirix.services.skill_learning_run_manager import SkillLearningRunManager

    runs = SkillLearningRunManager()
    run = await runs.create_standalone_run(
        organization_id=world.org,
        user_id=world.user.id,
        agent_id=world.meta.id,
        mode="procedural",
        actor_id=world.actor.id,
    )
    assert await runs.mark_running(run.id)
    return run.id


async def experiences(world, ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """The world's experiences by id (any status)."""
    from sqlalchemy import select

    from mirix.orm.skill_experience import SkillExperience as SkillExperienceORM
    from mirix.server.server import db_context

    async with db_context() as db:
        stmt = select(SkillExperienceORM).where(
            SkillExperienceORM.agent_id == world.meta.id,
            SkillExperienceORM.user_id == world.user.id,
        )
        if ids is not None:
            stmt = stmt.where(SkillExperienceORM.id.in_(ids))
        rows = (await db.execute(stmt)).scalars().all()
        return {row.id: row.to_pydantic() for row in rows}


async def skills(world, *, include_deleted: bool = False) -> Dict[str, Any]:
    """The world user's skills by name."""
    from sqlalchemy import select

    from mirix.orm.procedural_memory import ProceduralMemoryItem
    from mirix.server.server import db_context

    async with db_context() as db:
        stmt = select(ProceduralMemoryItem).where(
            ProceduralMemoryItem.user_id == world.user.id
        )
        if not include_deleted:
            stmt = stmt.where(~ProceduralMemoryItem.is_deleted)
        rows = (await db.execute(stmt)).scalars().all()
        return {row.name: row.to_pydantic() for row in rows}


async def versions(skill_id: str) -> List[tuple]:
    """``(version_no, change_type, actor_kind, run_id, reason)`` of a skill."""
    from sqlalchemy import select

    from mirix.orm.skill_version import SkillVersion as SkillVersionORM
    from mirix.server.server import db_context

    async with db_context() as db:
        rows = (
            (
                await db.execute(
                    select(SkillVersionORM)
                    .where(SkillVersionORM.skill_id == skill_id)
                    .order_by(SkillVersionORM.version_no)
                )
            )
            .scalars()
            .all()
        )
    return [
        (r.version_no, r.change_type, r.actor_kind, r.run_id, r.reason) for r in rows
    ]


def tool_results(request_json: Dict[str, Any]) -> List[str]:
    """The tool-result contents a chat request carried."""
    out = []
    for message in request_json.get("messages", []):
        if message.get("role") == "tool":
            content = message.get("content")
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            out.append(content or "")
    return out


def user_texts(request_json: Dict[str, Any]) -> List[str]:
    out = []
    for message in request_json.get("messages", []):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            out.append(content or "")
    return out
