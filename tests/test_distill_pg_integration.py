"""Postgres integration: the whole distill path of a learning run.

Run through the fresh-database runner (MIRIX_PG_URI is read at import time):

    scripts/run_pg_integration_tests.sh tests/test_distill_pg_integration.py

Real managers on Postgres (sessions, messages, files, learning runs, the row
locks and the atomic commit), the production distiller and the production
OpenAI-compatible LLM client, talking to the offline fake model server — no
paid call, nothing leaves localhost. Only the curator stage and the dream
checkpoint are stubbed (the curator has its own Postgres test,
tests/test_curator_pg_integration.py). Covered:

* images stored through ``store_image_part`` reach the model as base64 at
  their original position in ONE user message; experiences land and
  exactly the rows up to the watermark are marked, in one transaction; the
  per-session metrics and the run's ``llm_usage`` are recorded;
* a long session under a small context window is chunked, every chunk
  within the budget, every chunk's experiences kept;
* a counted failure three times retires the session (``distill_failed_at``);
  an infrastructure error backs it off;
* a repeated commit of the same session inserts nothing twice.
"""

from __future__ import annotations

import base64
import io
import json
import uuid
from types import SimpleNamespace

import pytest
import pytest_asyncio
from PIL import Image
from sqlalchemy import select

from mirix.settings import settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.skipif(
        not settings.mirix_pg_uri_no_default,
        reason="needs Postgres: run via scripts/run_pg_integration_tests.sh",
    ),
]


def _png(color, size=(12, 8)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, "PNG")
    return buffer.getvalue()


def _user_parts(request_json):
    return next(m for m in request_json["messages"] if m["role"] == "user")["content"]


def _user_text(request_json):
    content = _user_parts(request_json)
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if p.get("type") == "text")


def _session_of(request_json):
    return _user_text(request_json).split("session_id: ", 1)[1].split("\n", 1)[0]


def _chunk_label(request_json):
    text = _user_text(request_json)
    if "covering rounds" not in text:
        return "whole"
    return text.split("\nchunk ", 1)[1].split(" covering", 1)[0]


def _answer(request_json):
    from fakes.fake_llm_server import ChatResponse

    title = f"{_session_of(request_json)} {_chunk_label(request_json)}"
    return ChatResponse(
        content=json.dumps(
            [
                {
                    "experience_type": "worth_learning",
                    "title": title,
                    "content": "reuse it",
                    "importance": 0.6,
                    "credibility": 0.7,
                    "evidence": {
                        "quote": "correct",
                        "signal_type": "user_confirmation",
                    },
                }
            ]
        ),
        usage={"prompt_tokens": 300, "completion_tokens": 20, "total_tokens": 320},
    )


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def tables():
    from mirix.server import server

    await server.engine.dispose(close=False)
    await server.ensure_tables_created()
    yield
    await server.engine.dispose()


@pytest_asyncio.fixture(loop_scope="module")
async def world(tables, monkeypatch, tmp_path, fake_llm_server):
    from mirix import constants
    from mirix.orm.agent import Agent as AgentORM
    from mirix.orm.client import Client as ClientORM
    from mirix.orm.organization import Organization as OrganizationORM
    from mirix.orm.user import User as UserORM
    from mirix.schemas.client import Client as PydanticClient
    from mirix.schemas.llm_config import LLMConfig
    from mirix.schemas.user import User as PydanticUser
    from mirix.server.server import db_context
    from mirix.services import context_budget, skill_learning_trigger
    from mirix.services.auto_dream_manager import AutoDreamManager

    monkeypatch.setattr(settings, "images_dir", tmp_path / "images")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setattr(constants, "LLM_MAX_RETRIES", 0)
    context_budget._reset_tokenizer_state_for_tests()

    async def no_curation(self, **kwargs):
        from mirix.services.skill_experience_curator import CuratorOutcome

        return CuratorOutcome()

    async def no_checkpoint(self, *args, **kwargs):
        return None

    monkeypatch.setattr(AutoDreamManager, "_curate", no_curation)
    monkeypatch.setattr(AutoDreamManager, "write_checkpoint", no_checkpoint)
    monkeypatch.setattr(
        skill_learning_trigger, "schedule_learning_run", lambda *a, **k: None
    )

    org_id = f"it-org-{uuid.uuid4().hex[:8]}"
    client_id = f"it-client-{uuid.uuid4().hex[:8]}"
    user_id = f"it-user-{uuid.uuid4().hex[:8]}"
    agent_id = f"agent-{uuid.uuid4()}"
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
                write_scope="it",
                read_scopes=["it"],
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
        agent = AgentORM(
            id=agent_id,
            organization_id=org_id,
            agent_type="meta_memory_agent",
            name="meta",
        )
        agent._created_by_id = client_id
        db.add(agent)
        await db.commit()

    config = LLMConfig(
        model="fake-model",
        model_endpoint_type="openai",
        model_endpoint=fake_llm_server.base_url,
        context_window=200_000,
        max_output_tokens=16_000,
        api_key="fake-key",
    )
    yield SimpleNamespace(
        org=org_id,
        agent=agent_id,
        user=PydanticUser(
            id=user_id, name=user_id, organization_id=org_id, timezone="UTC"
        ),
        actor=PydanticClient(
            id=client_id,
            organization_id=org_id,
            name=client_id,
            write_scope="it",
            read_scopes=["it"],
        ),
        meta=SimpleNamespace(id=agent_id, llm_config=config),
        server=fake_llm_server,
    )
    context_budget._reset_tokenizer_state_for_tests()


async def _store_image(world, raw: bytes):
    from mirix.services.conversation_content import ImagePart
    from mirix.services.image_store import store_image_part

    return await store_image_part(
        ImagePart(
            data_b64=base64.b64encode(raw).decode(), declared_mime_type="image/png"
        ),
        organization_id=world.org,
        user_id=world.user.id,
    )


async def _record(world, sid, turns):
    from mirix.services.conversation_message_manager import ConversationMessageManager

    return await ConversationMessageManager().record_turns(
        session_id=sid,
        user_id=world.user.id,
        organization_id=world.org,
        turns=turns,
        actor=world.actor,
        session_tag="task",
        agent_id=world.agent,
    )


async def _close(world, sid):
    from mirix.services.conversation_session_manager import ConversationSessionManager

    await ConversationSessionManager().close_session(
        organization_id=world.org, user_id=world.user.id, session_id=sid
    )


async def _learn(world):
    from mirix.schemas.auto_dream import AutoDreamRequest
    from mirix.services.agent_trigger_state_manager import AgentTriggerStateManager
    from mirix.services.auto_dream_manager import AutoDreamManager
    from mirix.services.skill_learning_run_manager import SkillLearningRunManager

    claim = await AgentTriggerStateManager().check_and_claim_fire(
        agent_id=world.agent,
        user_id=world.user.id,
        organization_id=world.org,
        threshold=5,
        force=True,
    )
    await AutoDreamManager().run(
        request=AutoDreamRequest(mode="procedural"),
        user=world.user,
        actor=world.actor,
        meta_agent_state=world.meta,
        run_id=claim.run_id,
    )
    return await SkillLearningRunManager().get_run(claim.run_id)


async def _rows(world, sid):
    from mirix.orm.conversation_message import ConversationMessage as MessageORM
    from mirix.server.server import db_context

    async with db_context() as db:
        return (
            (
                await db.execute(
                    select(MessageORM)
                    .where(
                        MessageORM.user_id == world.user.id,
                        MessageORM.session_id == sid,
                    )
                    .order_by(MessageORM.created_at)
                )
            )
            .scalars()
            .all()
        )


async def _experiences(world, sid):
    from mirix.orm.skill_experience import SkillExperience as ExperienceORM
    from mirix.server.server import db_context

    async with db_context() as db:
        return (
            (
                await db.execute(
                    select(ExperienceORM).where(
                        ExperienceORM.user_id == world.user.id,
                        ExperienceORM.session_id == sid,
                    )
                )
            )
            .scalars()
            .all()
        )


async def _session(world, sid):
    from mirix.orm.conversation_session import ConversationSession as SessionORM
    from mirix.server.server import db_context

    async with db_context() as db:
        return (
            await db.execute(
                select(SessionORM).where(
                    SessionORM.user_id == world.user.id, SessionORM.session_id == sid
                )
            )
        ).scalar_one()


class TestDistillOnPostgres:
    async def test_images_reach_the_model_in_place_and_the_session_commits(self, world):
        chart, photo = _png("red"), _png("blue", (30, 20))
        chart_ref = await _store_image(world, chart)
        photo_ref = await _store_image(world, photo)
        await _record(
            world,
            "docvqa-1",
            [
                {
                    "role": "user",
                    "content_parts": [
                        {"type": "text", "text": "What is the total on this invoice?"},
                        chart_ref,
                        {"type": "text", "text": "Answer with a number."},
                    ],
                },
                {"role": "assistant", "content": "The total is 42."},
                {
                    "role": "user",
                    "content_parts": [
                        {"type": "text", "text": "Correct. Now this one:"},
                        photo_ref,
                    ],
                },
                {"role": "assistant", "content": "It is 7."},
            ],
        )
        await _close(world, "docvqa-1")
        await _record(
            world,
            "alf-1",
            [
                {"role": "user", "content": "Your task is to: put a mug in the sink."},
                {"role": "assistant", "content": "go to countertop 1"},
                {"role": "user", "content": "You see a mug 1."},
                {"role": "assistant", "content": "take mug 1 from countertop 1"},
            ],
        )
        await _close(world, "alf-1")
        world.server.set_default_responder(_answer)

        run = await _learn(world)
        assert run.status == "succeeded" and sorted(run.session_ids) == [
            "alf-1",
            "docvqa-1",
        ]

        [doc_request] = [
            r for r in world.server.chat_requests() if _session_of(r.json) == "docvqa-1"
        ]
        parts = _user_parts(doc_request.json)
        kinds = [p["type"] for p in parts]
        assert kinds == [
            "text",
            "text",
            "image_url",
            "text",
            "text",
            "image_url",
            "text",
        ]
        assert parts[0]["text"].endswith("user: What is the total on this invoice?\n")
        assert parts[1]["text"] == "<image 0>"
        assert parts[3]["text"].startswith(
            "\nAnswer with a number.\nassistant: The total is 42."
        )
        assert parts[3]["text"].endswith("user: Correct. Now this one:\n")
        assert parts[6]["text"].startswith("\nassistant: It is 7.")
        for part, raw in ((parts[2], chart), (parts[5], photo)):
            url = part["image_url"]["url"]
            assert url.startswith("data:image/png;base64,")
            assert base64.b64decode(url.split(",", 1)[1]) == raw
        # The store never held base64, only the placeholder text.
        stored = await _rows(world, "docvqa-1")
        assert (
            "base64" not in stored[0].content and "[image: file-" in stored[0].content
        )

        for sid in ("docvqa-1", "alf-1"):
            [experience] = await _experiences(world, sid)
            assert experience.title == f"{sid} whole" and experience.status == "pending"
            assert json.loads(experience.evidence)["signal_type"] == "user_confirmation"
            assert all(r.distilled_at is not None for r in await _rows(world, sid))

        by_session = {e["session_id"]: e for e in run.distill_metrics}
        assert by_session["docvqa-1"]["images"] == 2
        assert (
            by_session["docvqa-1"]["messages"] == 4
            and by_session["docvqa-1"]["rounds"] == 2
        )
        assert by_session["alf-1"]["images"] == 0
        assert all(
            e["status"] == "ok" and e["experiences"] == 1 for e in by_session.values()
        )
        assert run.llm_usage["calls"] == 2 and run.llm_usage["prompt_tokens"] == 600
        assert run.llm_usage["by_stage"]["distill"]["total_tokens"] == 640
        assert run.retry_events == []

    async def test_a_long_session_is_chunked_under_a_small_window(self, world):
        world.meta.llm_config = world.meta.llm_config.model_copy(
            update={"context_window": 12_000, "max_output_tokens": 2_000}
        )
        turns = []
        for i in range(40):
            turns.append({"role": "user", "content": f"obs {i:02d} " + "o" * 900})
            turns.append({"role": "assistant", "content": f"act {i:02d}"})
        await _record(world, "long-1", turns)
        await _close(world, "long-1")
        world.server.set_default_responder(_answer)

        run = await _learn(world)
        [entry] = run.distill_metrics
        n = entry["chunks"]
        assert n >= 2 and entry["status"] == "ok"
        assert entry["chunk_budget_tokens"] is not None
        labels = sorted(_chunk_label(r.json) for r in world.server.chat_requests())
        assert labels == sorted(f"{i}/{n}" for i in range(1, n + 1))
        for request in world.server.chat_requests():
            assert request.json["max_completion_tokens"] >= 1024
        titles = sorted(e.title for e in await _experiences(world, "long-1"))
        assert titles == sorted(f"long-1 {i}/{n}" for i in range(1, n + 1))
        assert all(r.distilled_at is not None for r in await _rows(world, "long-1"))

    async def test_three_counted_failures_retire_the_session(self, world):
        from mirix.services.conversation_session_manager import (
            ConversationSessionManager,
        )

        await _record(
            world,
            "bad-1",
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        )
        await _close(world, "bad-1")
        world.server.set_default_responder(lambda request_json: "not json at all")
        for attempt in (1, 2, 3):
            run = await _learn(world)
            [entry] = run.distill_metrics
            assert (
                entry["failure_kind"] == "parse"
                and entry["distill_failures"] == attempt
            )
            assert [e["kind"] for e in run.retry_events] == ["parse", "parse"]
        assert entry["status"] == "failed_permanently"
        session = await _session(world, "bad-1")
        assert session.distill_failures == 3 and session.distill_failed_at is not None
        assert all(r.distilled_at is None for r in await _rows(world, "bad-1"))
        pending = await ConversationSessionManager().list_pending_distill_sessions(
            organization_id=world.org, user_id=world.user.id, agent_id=world.agent
        )
        assert "bad-1" not in pending

    async def test_an_infrastructure_error_backs_the_session_off(self, world):
        from fakes.fake_llm_server import Fault

        from mirix.services.conversation_session_manager import (
            ConversationSessionManager,
        )

        await _record(
            world,
            "infra-1",
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
        )
        await _close(world, "infra-1")
        world.server.inject_fault(Fault.http_error(503))
        run = await _learn(world)
        [entry] = run.distill_metrics
        assert entry["status"] == "deferred" and entry["failure_kind"] == "infra"
        session = await _session(world, "infra-1")
        assert session.distill_failures == 0 and session.distill_infra_failures == 1
        assert session.distill_retry_after is not None
        pending = await ConversationSessionManager().list_pending_distill_sessions(
            organization_id=world.org, user_id=world.user.id, agent_id=world.agent
        )
        assert "infra-1" not in pending

    async def test_a_repeated_commit_inserts_nothing_twice(self, world):
        from mirix.services.agent_trigger_state_manager import AgentTriggerStateManager
        from mirix.services.conversation_message_manager import parse_watermark
        from mirix.services.skill_learning_run_manager import SkillLearningRunManager

        await _record(
            world,
            "twice-1",
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        )
        await _close(world, "twice-1")
        claim = await AgentTriggerStateManager().check_and_claim_fire(
            agent_id=world.agent,
            user_id=world.user.id,
            organization_id=world.org,
            threshold=5,
            force=True,
        )
        runs = SkillLearningRunManager()
        run = await runs.get_run(claim.run_id)
        payload = {
            "agent_id": world.agent,
            "user_id": world.user.id,
            "organization_id": world.org,
            "session_id": "twice-1",
            "experience_type": "worth_learning",
            "title": "once",
            "content": "c",
            "importance": 0.5,
            "credibility": 0.5,
            "evidence": "",
            "status": "pending",
        }
        kwargs = dict(
            organization_id=world.org,
            user_id=world.user.id,
            session_id="twice-1",
            watermark=parse_watermark(run.session_watermarks["twice-1"]),
            experiences=[payload],
        )
        first, marked = await runs.commit_distilled_session(claim.run_id, **kwargs)
        again, marked_again = await runs.commit_distilled_session(
            claim.run_id, **kwargs
        )
        assert (len(first), marked) == (1, 2)
        assert (again, marked_again) == ([], 0)
        assert [e.title for e in await _experiences(world, "twice-1")] == ["once"]
