"""End to end: a skill retrieved with a session id is shown to that
session's distiller as served, and its feedback row carries the version served.

Nothing hand-built stands between the pieces: MIRIX's own retrieval ranks the
skill and its diagnostics become the retrieval log's procedural trace exactly
as ``/memory/retrieve/conversation`` writes it; the session's turns and agent
context are recorded through the conversation store; a claimed learning run is
executed by ``AutoDreamManager`` with the production distiller (its
``SessionContextLoader`` included) and the production OpenAI-compatible client
talking to the offline fake model server; the commit is the learning run's own.
Only the curator, the dream checkpoint and background scheduling are stubbed.
A hermetic SQLite database; nothing leaves localhost.
"""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import mirix.orm  # noqa: F401 -- register every ORM class before create_all
import mirix.services.procedural_memory_manager as pmm
from fakes.fake_llm_server import ChatResponse
from mirix import constants
from mirix.orm.agent import Agent as AgentORM
from mirix.orm.base import Base
from mirix.orm.client import Client as ClientORM
from mirix.orm.organization import Organization as OrganizationORM
from mirix.orm.skill_session_feedback import SkillSessionFeedback as FeedbackORM
from mirix.orm.user import User as UserORM
from mirix.schemas.auto_dream import AutoDreamRequest
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.user import User as PydanticUser
from mirix.services import context_budget, skill_history, skill_learning_trigger
from mirix.services.agent_trigger_state_manager import AgentTriggerStateManager
from mirix.services.conversation_message_manager import ConversationMessageManager
from mirix.services.conversation_session_manager import ConversationSessionManager
from mirix.services.retrieval_log_manager import RetrievalLogManager
from mirix.services.skill_learning_run_manager import SkillLearningRunManager
from mirix.settings import settings

pytestmark = pytest.mark.asyncio(loop_scope="module")

SCOPE = "p16-wiring"


@pytest.fixture(autouse=True)
def _force_sqlite_branch(monkeypatch):
    monkeypatch.setattr(
        type(settings), "mirix_pg_uri_no_default", property(lambda self: None)
    )


class _FakeEmbedModel:
    async def get_text_embedding(self, text):
        return [0.1] * 8


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch):
    async def fake_embedding_model(config, *args, **kwargs):
        return _FakeEmbedModel()

    monkeypatch.setattr(pmm, "embedding_model", fake_embedding_model)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sm(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("p16_wiring") / "test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    local = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def _ctx():
        async with local() as session:
            try:
                yield session
            finally:
                await session.close()

    yield _ctx
    await engine.dispose()


@pytest.fixture
def llm_config(fake_llm_server):
    return LLMConfig(
        model="fake-model",
        model_endpoint_type="openai",
        model_endpoint=fake_llm_server.base_url,
        context_window=200_000,
        max_output_tokens=16_000,
        api_key="fake-key",
    )


@pytest_asyncio.fixture(loop_scope="module")
async def scope(sm, llm_config):
    org = f"org-{uuid.uuid4().hex[:8]}"
    user_id = f"user-{uuid.uuid4().hex[:8]}"
    agent_id = f"agent-{uuid.uuid4()}"
    client_id = f"client-{uuid.uuid4().hex[:8]}"
    async with sm() as db:
        db.add(OrganizationORM(id=org, name=org))
        await db.flush()
        db.add(
            ClientORM(
                id=client_id,
                organization_id=org,
                name=client_id,
                status="active",
                write_scope=SCOPE,
                read_scopes=[SCOPE],
            )
        )
        db.add(
            UserORM(
                id=user_id,
                name=user_id,
                organization_id=org,
                status="active",
                timezone="UTC",
            )
        )
        agent = AgentORM(
            id=agent_id,
            organization_id=org,
            agent_type="meta_memory_agent",
            name="meta",
        )
        agent._created_by_id = client_id
        db.add(agent)
        await db.commit()
    return SimpleNamespace(
        org=org,
        agent=agent_id,
        user=PydanticUser(
            id=user_id, name=user_id, organization_id=org, timezone="UTC"
        ),
        actor=PydanticClient(
            id=client_id,
            organization_id=org,
            name="test client",
            write_scope=SCOPE,
            read_scopes=[SCOPE],
        ),
        meta=SimpleNamespace(id=agent_id, llm_config=llm_config),
    )


@pytest.fixture
def env(sm, monkeypatch):
    """Route MIRIX's managers to the hermetic database; stub only what is not
    the distiller. The distiller — with its session context loader — and the
    LLM client are the production ones."""
    import mirix.server.server as server_module
    from mirix.services.auto_dream_manager import AutoDreamManager
    from mirix.services.skill_experience_curator import CuratorOutcome

    monkeypatch.setattr(server_module, "db_context", sm)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setattr(constants, "LLM_MAX_RETRIES", 0)
    context_budget._reset_tokenizer_state_for_tests()

    async def no_curation(self, **kwargs):
        return CuratorOutcome()

    async def no_checkpoint(self, *args, **kwargs):
        return None

    monkeypatch.setattr(AutoDreamManager, "_curate", no_curation)
    monkeypatch.setattr(AutoDreamManager, "write_checkpoint", no_checkpoint)
    monkeypatch.setattr(
        skill_learning_trigger, "schedule_learning_run", lambda *a, **k: None
    )
    yield
    context_budget._reset_tokenizer_state_for_tests()


SERVED_INSTRUCTIONS = "1. Read the parcel label. 2. Weigh the parcel before sealing."
EDITED_INSTRUCTIONS = "1. Weigh first. 2. Then read the label."


async def _create_skill(sm, scope):
    manager = pmm.ProceduralMemoryManager(session_maker=sm)
    result = await manager.apply_skill_change(
        skill_history.CHANGE_CREATE,
        actor=scope.actor,
        user_id=scope.user.id,
        fields={
            "name": "parcel-label-check",
            "entry_type": "workflow",
            "description": "Check a parcel's label before shipping it.",
            "instructions": SERVED_INSTRUCTIONS,
            "triggers": ["a parcel to ship"],
            "examples": [],
            "organization_id": scope.org,
            "filter_tags": {"scope": SCOPE},
        },
    )
    return manager, result.skill.id


async def _retrieve_and_log(sm, scope, manager, session_id):
    """What /memory/retrieve/conversation does for the procedural type: the
    skill ranking, then its bounded diagnostics as the log's trace."""
    ranked = await manager.retrieve_skills(
        "ship a parcel and check its label", scope.user, [SCOPE], 5
    )
    assert [r.skill.name for r in ranked.skills] == ["parcel-label-check"]
    trace = {
        "procedural": ranked.diagnostics.to_dict(
            max_candidates=constants.RETRIEVAL_LOG_TRACE_CANDIDATES
        )
    }
    log_id = await RetrievalLogManager(session_maker=sm).create_log(
        organization_id=scope.org,
        user_id=scope.user.id,
        client_id=scope.actor.id,
        session_id=session_id,
        memory_types=["procedural"],
        query_text="ship a parcel",
        extracted_topics=None,
        extraction_status=None,
        embedding_status=trace["procedural"].get("embedding_status"),
        per_type=trace,
        llm_usage=None,
        latency_ms=1,
    )
    assert log_id is not None
    return trace["procedural"]


async def _write_session(sm, scope, session_id):
    messages = ConversationMessageManager(session_maker=sm)
    await messages.record_turns(
        session_id=session_id,
        user_id=scope.user.id,
        organization_id=scope.org,
        turns=[
            {"role": "user", "content": "Ship the parcel to the warehouse."},
            {
                "role": "assistant",
                "content": "I read the parcel label first, as the skill says.",
            },
            {"role": "tool", "content": "[scale] the parcel weighs 2.4 kg"},
            {"role": "assistant", "content": "Sealing and shipping the parcel."},
            {"role": "tool", "content": "[ship] shipment accepted: tracking 77"},
        ],
        actor=scope.actor,
        session_tag="task",
        agent_id=scope.agent,
        agent_context="PROMPT TEMPLATE: You ship parcels. Skills: {skills}",
    )
    await ConversationSessionManager(session_maker=sm).close_session(
        organization_id=scope.org, user_id=scope.user.id, session_id=session_id
    )


async def _learn(sm, scope, session_id):
    from mirix.services.auto_dream_manager import AutoDreamManager

    claim = await AgentTriggerStateManager(session_maker=sm).check_and_claim_fire(
        agent_id=scope.agent,
        user_id=scope.user.id,
        organization_id=scope.org,
        threshold=5,
        force=True,
    )
    assert session_id in claim.session_ids
    await AutoDreamManager().run(
        request=AutoDreamRequest(mode="procedural"),
        user=scope.user,
        actor=scope.actor,
        meta_agent_state=scope.meta,
        run_id=claim.run_id,
    )
    return await SkillLearningRunManager(session_maker=sm).get_run(claim.run_id)


def _sent_text(request_json):
    return "\n".join(
        part.get("text", "") if isinstance(part, dict) else str(part)
        for message in request_json["messages"]
        if message["role"] == "user"
        for part in (
            message["content"]
            if isinstance(message["content"], list)
            else [{"text": message["content"]}]
        )
    )


class TestServedSkillsReachTheFeedbackRows:
    async def test_retrieval_to_feedback_row_with_the_version_served(
        self, sm, env, scope, fake_llm_server
    ):
        manager, skill_id = await _create_skill(sm, scope)
        session_id = f"s-{uuid.uuid4().hex[:8]}"
        trace = await _retrieve_and_log(sm, scope, manager, session_id)
        assert trace["returned_versions"] == {skill_id: 1}
        await _write_session(sm, scope, session_id)
        # The skill changes after it was served: the distiller is shown the
        # version the agent got, not today's text.
        await manager.apply_skill_change(
            skill_history.CHANGE_EDIT,
            actor=scope.actor,
            user_id=scope.user.id,
            skill_id=skill_id,
            fields={"instructions": EDITED_INSTRUCTIONS},
        )
        fake_llm_server.queue_response(
            ChatResponse(
                content=json.dumps(
                    {
                        "outcome_evidence": 'the tool said "shipment accepted: tracking 77"',
                        "outcome": "succeeded",
                        "skill_feedback": [
                            {
                                "skill": "parcel-label-check",
                                "followed": "yes",
                                "followed_evidence": '"I read the parcel label first, as the skill says"',
                                "effect": "helped",
                                "effect_evidence": '"shipment accepted: tracking 77"',
                            }
                        ],
                        "experiences": [],
                    }
                )
            )
        )

        run = await _learn(sm, scope, session_id)

        [request] = fake_llm_server.chat_requests()
        sent = _sent_text(request.json)
        assert "<skills_served>" in sent
        assert "[1] name: parcel-label-check (version 1; served 1 time)" in sent
        assert SERVED_INSTRUCTIONS in sent and EDITED_INSTRUCTIONS not in sent
        assert "    When to use: a parcel to ship" in sent
        assert "PROMPT TEMPLATE: You ship parcels. Skills: {skills}" in sent
        async with sm() as db:
            [row] = (
                (
                    await db.execute(
                        select(FeedbackORM).where(FeedbackORM.session_id == session_id)
                    )
                )
                .scalars()
                .all()
            )
        assert row.skill_id == skill_id and row.skill_version_no == 1
        assert (row.followed, row.effect) == ("yes", "helped")
        assert row.session_outcome == "succeeded"
        assert row.learning_run_id == run.id
        assert row.organization_id == scope.org and row.user_id == scope.user.id
        [entry] = [e for e in run.distill_metrics if e["session_id"] == session_id]
        assert entry["served_skills"] == 1 and entry["agent_context_chars"] > 0
        assert entry["skill_feedback"]["given"] == 1
