"""Each skill's usage record — the sessions it was served in, how
often it was followed, its effects, those sessions' outcomes (per version
served too) and its provenance — shown to the curator in a batch's header
(for the skills served in the batch's sessions or named by its experiences),
in ``skill_read`` and in ``skill_list``. Information only.

Pure parts need nothing; the rest runs hermetic and offline on the curator
harness (a fresh database initialized like server startup, a real meta agent
with its procedural child, the local fake LLM server; nothing reaches a
model provider).
"""

import time
import uuid

import pytest
import pytest_asyncio
from curator_harness import (
    add_experience,
    experiences,
    make_world,
    procedural_agent,
    start_run,
)
from db_isolation import block_non_loopback_network, isolated_database
from fakes.fake_llm_server import FakeLLMServer

from mirix.functions.function_sets import memory_tools as mt
from mirix.schemas.usage import MirixUsageStatistics
from mirix.services.context_budget import TokenCounter
from mirix.services.skill_experience_curator import (
    SKILL_USAGE_HEADING,
    build_batch_payload,
    fit_skill_usage,
)
from mirix.services.skill_session_feedback import (
    USAGE_RECORD_HEADING,
    SkillSessionFeedbackManager,
    SkillUsageRecord,
    UsageCounts,
    render_usage_line,
    render_usage_record,
    skills_named_in,
)


class _QuarterEncoding:
    def encode(self, text, disallowed_special=()):
        return [0] * ((len(text) + 3) // 4)


COUNTER = TokenCounter(encoding=_QuarterEncoding())


class _Row:
    def __init__(self, followed, effect, outcome):
        self.followed = followed
        self.effect = effect
        self.session_outcome = outcome


def _record(rows, *, current=None, sources=0):
    """A record from ``(session_id, version, followed, effect, outcome)``."""
    record = SkillUsageRecord(
        "proc-x", current_version=current, source_sessions=sources
    )
    for session_id, version, followed, effect, outcome in rows:
        row = _Row(followed, effect, outcome)
        record.total.add(row)
        record.by_version.setdefault(version, UsageCounts()).add(row)
        if session_id not in record.sessions:
            record.sessions.append(session_id)
    return record


# --------------------------------------------------------------------------- #
# Pure                                                                         #
# --------------------------------------------------------------------------- #


class TestWhichSkillsATextNames:
    def test_whole_names_and_ids_case_ignored(self):
        skills = [
            ("proc-1", "check-folder"),
            ("proc-2", "check"),
            ("proc-3", "Folder Sync"),
            ("proc-4", "unrelated"),
            ("proc-5", "old"),
        ]
        texts = [
            "The CHECK-FOLDER skill was followed.",
            "folder sync helped; see proc-4.",
            None,
        ]
        # "check" is only part of a hyphenated name, "old" only part of a
        # word; proc-4 is named by its id.
        assert skills_named_in(texts, skills) == ["proc-1", "proc-3", "proc-4"]

    def test_a_one_word_name_must_be_specific_to_count(self):
        skills = [
            ("proc-1", "search"),
            ("proc-2", "a"),
            ("proc-3", "troubleshooting"),
            ("proc-4", "冰箱门把手检查"),
            ("proc-5", "web search"),
        ]
        texts = [
            "I ran a search, then a web search; troubleshooting took a while.",
            "先检查冰箱门把手检查流程。",
        ]
        # "search" and "a" are ordinary words: those skills are found by id.
        assert skills_named_in(texts, skills) == ["proc-3", "proc-4", "proc-5"]
        assert skills_named_in(["see proc-1"], skills) == ["proc-1"]

    def test_nothing_to_match(self):
        assert skills_named_in([], [("proc-1", "a")]) == []
        assert skills_named_in(["text"], []) == []


class TestTheToolsReadTheRecordsWhereTheDistillerWritesThem:
    @pytest.mark.asyncio
    async def test_a_client_without_an_organization_reads_the_default_one(
        self, monkeypatch
    ):
        """The distiller keys rows by owner_org(client): the client's
        organization, else the default one — never the user's."""
        from types import SimpleNamespace

        from mirix.constants import DEFAULT_ORG_ID
        from mirix.services import skill_session_feedback as feedback_module

        seen = {}

        async def records(self, *, organization_id, user_id, skill_ids):
            seen["organization_id"] = organization_id
            return {}

        monkeypatch.setattr(
            feedback_module.SkillSessionFeedbackManager, "usage_records", records
        )
        agent = SimpleNamespace(
            actor=SimpleNamespace(organization_id=None),
            user=SimpleNamespace(id="user-1", organization_id="org-of-the-user"),
            procedural_memory_manager=SimpleNamespace(session_maker=None),
        )
        monkeypatch.setattr(mt, "_skill_owner_id", lambda self: "user-1")
        await mt._skill_usage_records(agent, ["proc-1"])
        assert seen["organization_id"] == DEFAULT_ORG_ID
        agent.actor.organization_id = "org-of-the-client"
        await mt._skill_usage_records(agent, ["proc-1"])
        assert seen["organization_id"] == "org-of-the-client"


class TestRenderedRecords:
    def test_a_skill_never_served(self):
        record = SkillUsageRecord("proc-x", source_sessions=2)
        assert render_usage_line(record) == (
            "usage: not served in any distilled session yet; cited, with a "
            "verified quote, by experiences of 2 session(s)"
        )
        assert render_usage_record(record).splitlines() == [
            USAGE_RECORD_HEADING,
            "- not served in any distilled session yet",
            "- cited, with a verified quote, by experiences of 2 distinct session(s)",
        ]
        assert "use it as evidence" in USAGE_RECORD_HEADING

    def test_counts_versions_and_judgements(self):
        record = _record(
            [
                ("s4", 3, "yes", "helped", "succeeded"),
                ("s3", 2, "no", "hurt", "failed"),
                ("s3", 2, "partly", "unclear", "failed"),  # reopened, judged twice
                ("s2", None, "unclear", "unclear", "unclear"),
                ("s1", 3, "yes", "no_effect", "succeeded"),
            ],
            current=3,
            sources=1,
        )
        line = render_usage_line(record)
        assert line.startswith(
            "usage: served in 4 distilled session(s) (5 judgements: a reopened "
            "session is judged per part); followed yes 2/partly 1/no 1/unclear 1; "
            "effect helped 1/no effect 1/hurt 1/unclear 2; session outcomes "
            "succeeded 2/failed 2/unclear 1;"
        )
        assert line.endswith(
            "cited, with a verified quote, by experiences of 1 session(s)"
        )
        lines = render_usage_record(record).splitlines()
        assert "most recent: s4, s3, s2, … 1 more" in lines[1]
        assert lines[3].startswith("- served at version 3 (current): 2 judgement(s)")
        assert lines[4].startswith("- served at version 2: 2 judgement(s)")
        assert lines[5].startswith("- served at version not recorded: 1 judgement(s)")

    def test_no_per_version_lines_when_no_version_was_recorded(self):
        record = _record([("s1", None, "yes", "helped", "succeeded")])
        text = render_usage_record(record)
        assert "served at version" not in text
        assert "most recent: s1" in text
        assert "judgements" not in text  # one judgement per session


def _entries(n, *, served=1, long=False):
    rows = [("s1", 1, "yes", "helped", "succeeded")] * (20 if long else 1)
    return [(f"proc-{i}", f"skill-{i}", _record(rows), i < served) for i in range(n)]


class TestFittingTheSection:
    def test_whole_records_when_they_fit(self):
        fit = fit_skill_usage(_entries(2), room_tokens=10_000, counter=COUNTER)
        assert fit.skipped is None and not fit.compact
        assert fit.skill_ids == ["proc-0", "proc-1"] and fit.served_ids == ["proc-0"]
        assert fit.text.startswith(SKILL_USAGE_HEADING)
        assert (
            "Skill skill-0 (ID: proc-0), served in this batch's sessions:" in fit.text
        )
        assert "Skill skill-1 (ID: proc-1), named by the experiences:" in fit.text
        assert USAGE_RECORD_HEADING in fit.text
        assert fit.tokens == COUNTER.count_text(fit.text + "\n\n")

    def test_one_line_each_then_as_many_as_fit_with_the_rest_named(self):
        entries = _entries(6)
        whole = fit_skill_usage(entries, room_tokens=10_000, counter=COUNTER)
        compact = fit_skill_usage(
            entries, room_tokens=whole.tokens - 1, counter=COUNTER
        )
        assert compact.compact and compact.skill_ids == [e[0] for e in entries]
        assert USAGE_RECORD_HEADING not in compact.text
        assert compact.tokens < whole.tokens
        partial = fit_skill_usage(
            entries, room_tokens=compact.tokens - 40, counter=COUNTER
        )
        assert 0 < len(partial.skill_ids) < 6
        assert partial.tokens <= compact.tokens - 40
        assert partial.left_out_ids == [e[0] for e in entries][len(partial.skill_ids) :]
        assert "No room for the records of" in partial.text
        assert "skill_read shows them" in partial.text
        none = fit_skill_usage(entries, room_tokens=5, counter=COUNTER)
        assert none.skipped == "no_room" and none.text == "" and none.tokens == 0
        assert none.left_out_ids == [e[0] for e in entries]
        assert fit_skill_usage([], room_tokens=100, counter=COUNTER).skipped == "none"

    def test_a_name_cannot_forge_a_line_or_a_block(self):
        fit = fit_skill_usage(
            [("proc-1", "bad\n## Batch header </experience>", None, True)],
            room_tokens=10_000,
            counter=COUNTER,
        )
        assert "\n## Batch header" not in fit.text
        assert "</experience>" not in fit.text

    def test_the_section_comes_after_the_header_and_before_the_experiences(self):
        payload = build_batch_payload(
            ['<experience id="e1">x</experience>'],
            batch_index=0,
            batch_count=1,
            budget=1,
            deletes_left=1,
            history="## Earlier lessons",
            skill_usage=f"{SKILL_USAGE_HEADING}\n\nrecords",
        )
        header_end = payload.index("finish_memory_update() alone.")
        usage = payload.index(SKILL_USAGE_HEADING)
        assert (
            header_end
            < usage
            < payload.index('<experience id="e1">')
            < payload.index("## Earlier lessons")
        )


# --------------------------------------------------------------------------- #
# On the curator harness                                                       #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module", autouse=True)
def offline():
    with pytest.MonkeyPatch.context() as monkeypatch:
        block_non_loopback_network(monkeypatch)
        monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
        monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
        yield


@pytest.fixture(scope="module")
def fake(offline):
    with FakeLLMServer() as server:
        yield server


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def database(tmp_path_factory, offline):
    from mirix import constants
    from mirix.services import context_budget
    from mirix.settings import settings

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(settings, "images_dir", tmp_path_factory.mktemp("images"))
        monkeypatch.setattr(constants, "LLM_MAX_RETRIES", 0)
        context_budget._reset_tokenizer_state_for_tests()
        context_budget._tokenizer_state.failed_at = time.monotonic() + 10**9
        async with isolated_database(tmp_path_factory.mktemp("usage-db")) as engine:
            yield engine
        context_budget._reset_tokenizer_state_for_tests()


async def _skill(world, name, *, scope=None):
    agent = (
        procedural_agent(world)
        if scope is None
        else procedural_agent(world, scope=scope)
    )
    answer = await mt.skill_create(
        agent,
        name=name,
        description="Opens an item and checks it",
        instructions="Open the item, then check the result twice.",
        entry_type="workflow",
    )
    assert "created successfully" in answer, answer
    found = await agent.procedural_memory_manager.get_live_skill_by_name(
        user=world.user, name=name, scopes=None
    )
    return found.id


def _feedback(world, skill_id, session_id, version, followed, effect, outcome, **kw):
    return {
        "organization_id": kw.get("org", world.org),
        "user_id": kw.get("user_id", world.user.id),
        "agent_id": world.meta.id,
        "session_id": session_id,
        "skill_id": skill_id,
        "skill_name": "n",
        "skill_version_no": version,
        "served_count": 1,
        "session_outcome": outcome,
        "followed": followed,
        "followed_evidence": "",
        "followed_source": None,
        "stated_followed": None,
        "effect": effect,
        "effect_evidence": "",
        "effect_source": None,
        "stated_effect": None,
    }


async def _cite(experience_ids, skill_id):
    from sqlalchemy import update

    from mirix.orm.skill_experience import SkillExperience as SkillExperienceORM
    from mirix.server.server import db_context

    async with db_context() as db:
        await db.execute(
            update(SkillExperienceORM)
            .where(SkillExperienceORM.id.in_(experience_ids))
            .values(cited_skill_ids=[skill_id], status="consumed", disposition="used")
        )
        await db.commit()


def _finish(steps=1):
    return MirixUsageStatistics(step_count=steps, stop_reason="finish")


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.usefixtures("database")
class TestUsageRecords:
    async def test_aggregates_the_owners_rows_and_the_provenance(self, fake):
        from mirix.services.user_manager import UserManager

        world = await make_world(fake)
        skill_id = await _skill(world, f"agg-{uuid.uuid4().hex[:6]}")
        agent = procedural_agent(world)
        edited = await mt.skill_edit(
            agent,
            skill_id=skill_id,
            field="instructions",
            old_text="check the result twice",
            new_text="check the result three times",
        )
        assert "updated (field: instructions" in edited, edited
        feedback = SkillSessionFeedbackManager()
        await feedback.create_rows(
            [
                _feedback(world, skill_id, "s-1", 1, "yes", "helped", "succeeded"),
                _feedback(world, skill_id, "s-2", 2, "no", "hurt", "failed"),
                _feedback(world, skill_id, "s-2", 2, "partly", "unclear", "failed"),
                # Another user's row is not this user's record.
                _feedback(
                    world,
                    skill_id,
                    "s-9",
                    2,
                    "yes",
                    "helped",
                    "succeeded",
                    user_id=UserManager.ADMIN_USER_ID,
                ),
            ]
        )
        sources = [
            await add_experience(world, title="a", session_id="src-a"),
            await add_experience(world, title="b", session_id="src-b"),
            await add_experience(world, title="c", session_id="src-b"),
        ]
        await _cite([e.id for e in sources], skill_id)
        # Provenance reads only experiences whose citations name the skill:
        # one cited by another skill alone does not count; one cited by both
        # does (src-c).
        from sqlalchemy import update

        from mirix.orm.skill_experience import SkillExperience as SkillExperienceORM
        from mirix.server.server import db_context

        other = await add_experience(world, title="d", session_id="src-other")
        both = await add_experience(world, title="e", session_id="src-c")
        async with db_context() as db:
            for experience, cited in (
                (other, ["proc-other"]),
                (both, ["proc-other", skill_id]),
            ):
                await db.execute(
                    update(SkillExperienceORM)
                    .where(SkillExperienceORM.id == experience.id)
                    .values(cited_skill_ids=cited)
                )
            await db.commit()
        records = await feedback.usage_records(
            organization_id=world.org,
            user_id=world.user.id,
            skill_ids=[skill_id, "proc-missing"],
        )
        record = records[skill_id]
        assert record.session_count == 2 and record.total.served == 3
        assert sorted(record.sessions) == ["s-1", "s-2"]
        assert record.total.followed == {"yes": 1, "partly": 1, "no": 1, "unclear": 0}
        assert record.total.effect["hurt"] == 1 and record.total.effect["helped"] == 1
        assert record.total.outcomes["failed"] == 2
        assert set(record.by_version) == {1, 2} and record.by_version[2].served == 2
        assert record.current_version == 2
        assert record.source_sessions == 3  # src-a, src-b, src-c
        assert records["proc-missing"].total.served == 0
        assert await feedback.skills_served_in_sessions(
            organization_id=world.org, user_id=world.user.id, session_ids=["s-2"]
        ) == [skill_id]

    async def test_skill_list_and_skill_read_show_the_record(self, fake):
        world = await make_world(fake)
        name = f"shown-{uuid.uuid4().hex[:6]}"
        skill_id = await _skill(world, name)
        await SkillSessionFeedbackManager().create_rows(
            [_feedback(world, skill_id, "s-read", 1, "yes", "helped", "succeeded")]
        )
        agent = procedural_agent(world)
        listed = (await mt.skill_list(agent)).splitlines()
        at = next(
            i for i, line in enumerate(listed) if line.startswith(f"[ID: {skill_id}]")
        )
        assert listed[at + 1].startswith("    usage: served in 1 distilled session(s);")
        assert "effect helped 1/" in listed[at + 1]
        read = await mt.skill_read(agent, skill_id)
        assert USAGE_RECORD_HEADING in read
        assert "- served in 1 distilled session(s); most recent: s-read" in read
        assert "- served at version 1 (current): 1 judgement(s)" in read


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.usefixtures("database")
class TestTheCuratorBatchShowsTheRecords:
    async def test_served_and_named_skills_of_the_scope_before_the_experiences(
        self, fake
    ):
        from mirix.services.skill_experience_curator import curate_pending_experiences
        from mirix.services.skill_learning_run_manager import SkillLearningRunManager

        world = await make_world(fake)
        tag = uuid.uuid4().hex[:6]
        served = await _skill(world, f"served-{tag}")
        named = await _skill(world, f"named-{tag}")
        quiet = await _skill(world, f"quiet-{tag}")
        hidden = await _skill(world, f"hidden-{tag}", scope="another-scope")
        session = f"s-batch-{tag}"
        await SkillSessionFeedbackManager().create_rows(
            [
                _feedback(world, served, session, 1, "partly", "hurt", "failed"),
                _feedback(world, hidden, session, 1, "yes", "helped", "failed"),
            ]
        )
        exp = await add_experience(
            world,
            title=f"Named-{tag} skipped its check",
            content=f"The named-{tag} skill lacks a check; hidden-{tag} too.",
            session_id=session,
        )
        run_id = await start_run(world)
        seen = {}

        async def runner(agent, payload, *, actor, user):
            seen["payload"] = payload
            return _finish()

        await curate_pending_experiences(
            run_id=run_id,
            user=world.user,
            actor=world.actor,
            meta_agent_state=world.meta,
            run_batch=runner,
        )
        payload = seen["payload"]
        first_block = payload.index("<experience id=")
        section = payload[payload.index(SKILL_USAGE_HEADING) : first_block]
        assert (
            f"Skill served-{tag} (ID: {served}), served in this batch's sessions:"
            in section
        )
        assert f"Skill named-{tag} (ID: {named}), named by the experiences:" in section
        assert "effect helped 0/no effect 0/hurt 1/unclear 0" in section
        assert hidden not in payload[:first_block]  # another scope
        assert quiet not in section
        [entry] = (await SkillLearningRunManager().get_run(run_id)).curator_metrics
        usage = entry["skill_usage"]
        assert usage["skill_ids"] == [served, named]
        assert usage["served_ids"] == [served]
        assert usage["skipped"] is None and usage["tokens"] > 0
        assert (
            entry["payload_tokens"] + usage["tokens"] + entry["history_tokens"]
            <= entry["capacity_tokens"]
        )
        assert (await experiences(world, [exp.id]))[exp.id].status == "consumed"

    async def test_no_skill_concerned_no_section(self, fake):
        from mirix.services.skill_experience_curator import curate_pending_experiences
        from mirix.services.skill_learning_run_manager import SkillLearningRunManager

        world = await make_world(fake)
        await _skill(world, f"unrelated-{uuid.uuid4().hex[:6]}")
        await add_experience(world, title="a lesson about nothing in particular")
        run_id = await start_run(world)
        seen = {}

        async def runner(agent, payload, *, actor, user):
            seen["payload"] = payload
            return _finish()

        await curate_pending_experiences(
            run_id=run_id,
            user=world.user,
            actor=world.actor,
            meta_agent_state=world.meta,
            run_batch=runner,
        )
        assert SKILL_USAGE_HEADING not in seen["payload"]
        [entry] = (await SkillLearningRunManager().get_run(run_id)).curator_metrics
        assert entry["skill_usage"]["skipped"] == "none"

    async def test_a_failure_reading_the_records_keeps_the_batch(self, fake):
        from mirix.services.skill_experience_curator import curate_pending_experiences
        from mirix.services.skill_learning_run_manager import SkillLearningRunManager

        world = await make_world(fake)
        exp = await add_experience(world, title="kept when the records fail")
        run_id = await start_run(world)

        class _Broken:
            async def skills_served_in_sessions(self, **_):
                raise RuntimeError("usage read bug")

        async def runner(agent, payload, *, actor, user):  # pragma: no cover
            raise AssertionError("the batch must not run")

        with pytest.raises(RuntimeError, match="usage read bug"):
            await curate_pending_experiences(
                run_id=run_id,
                user=world.user,
                actor=world.actor,
                meta_agent_state=world.meta,
                run_batch=runner,
                feedback_manager=_Broken(),
            )
        [entry] = (await SkillLearningRunManager().get_run(run_id)).curator_metrics
        assert entry["error_stage"] == "skill_usage"
        assert "usage read bug" in entry["error"]
        assert (await experiences(world, [exp.id]))[exp.id].status == "pending"


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.usefixtures("database")
class TestTheDistillerReadsTheServedVersion:
    """Served-skill feedback on the harness database (SQLite here, Postgres under
    scripts/run_pg_integration_tests.sh): a session's retrieval log and agent
    context come back as the distiller's session context, each skill at the
    version it was served."""

    async def test_served_version_text_and_agent_context(self, fake):
        import datetime as dt

        from mirix.orm.retrieval_log import RetrievalLog as RetrievalLogORM
        from mirix.server.server import db_context
        from mirix.services.conversation_message_manager import (
            ConversationMessageManager,
        )
        from mirix.services.distill_session_context import SessionContextLoader

        world = await make_world(fake)
        skill_id = await _skill(world, f"ctx-{uuid.uuid4().hex[:6]}")
        edited = await mt.skill_edit(
            procedural_agent(world),
            skill_id=skill_id,
            field="instructions",
            old_text="check the result twice",
            new_text="check the result three times",
        )
        assert "updated (field: instructions" in edited, edited
        session = f"s-ctx-{uuid.uuid4().hex[:6]}"
        await ConversationMessageManager().record_turns(
            session_id=session,
            user_id=world.user.id,
            organization_id=world.org,
            turns=[{"role": "user", "content": "file the report"}],
            actor=world.actor,
            agent_context="Template: {task} then {observation}",
        )
        async with db_context() as db:
            db.add(
                RetrievalLogORM(
                    id=f"rlog-{uuid.uuid4()}",
                    organization_id=world.org,
                    user_id=world.user.id,
                    client_id=world.actor.id,
                    session_id=session,
                    per_type={
                        "procedural": {
                            "returned_ids": [skill_id],
                            "returned_versions": {skill_id: 1},
                        }
                    },
                    created_at=dt.datetime.now(dt.timezone.utc),
                )
            )
            await db.commit()
        context = await SessionContextLoader().load(
            organization_id=world.org,
            user_id=world.user.id,
            session_id=session,
            watermark=dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=1),
            scopes=[world.scope],
        )
        [served] = context.served
        assert served.skill_id == skill_id and served.version_no == 1
        assert "check the result twice" in served.instructions  # as served
        assert context.agent_context == "Template: {task} then {observation}"
        hidden = await SessionContextLoader().load(
            organization_id=world.org,
            user_id=world.user.id,
            session_id=session,
            watermark=None,
            scopes=["another-scope"],
        )
        assert hidden.served == () and hidden.served_hidden == 1

    async def test_the_feedback_commits_once_with_the_session(self, fake):
        import datetime as dt

        from sqlalchemy import select

        from mirix.orm.skill_session_feedback import (
            SkillSessionFeedback as FeedbackORM,
        )
        from mirix.server.server import db_context
        from mirix.services.conversation_message_manager import (
            ConversationMessageManager,
        )
        from mirix.services.skill_learning_run_manager import SkillLearningRunManager

        world = await make_world(fake)
        skill_id = await _skill(world, f"commit-{uuid.uuid4().hex[:6]}")
        session = f"s-commit-{uuid.uuid4().hex[:6]}"
        await ConversationMessageManager().record_turns(
            session_id=session,
            user_id=world.user.id,
            organization_id=world.org,
            turns=[{"role": "user", "content": "do it"}],
            actor=world.actor,
        )
        run_id = await start_run(world)
        row = _feedback(world, skill_id, session, 1, "yes", "helped", "succeeded")
        watermark = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=5)
        for _ in range(2):  # a repeated commit writes nothing twice
            await SkillLearningRunManager().commit_distilled_session(
                run_id,
                organization_id=world.org,
                user_id=world.user.id,
                session_id=session,
                watermark=watermark,
                experiences=[],
                created_by_id=world.actor.id,
                skill_feedback=[row],
            )
        async with db_context() as db:
            stored = (
                (
                    await db.execute(
                        select(FeedbackORM).where(FeedbackORM.session_id == session)
                    )
                )
                .scalars()
                .all()
            )
        assert [(r.skill_id, r.learning_run_id) for r in stored] == [(skill_id, run_id)]
