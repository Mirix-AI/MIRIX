"""Postgres integration tests of the skill retrieval pipeline.

Run through the fresh-database runner (MIRIX_PG_URI must be set before import):

    scripts/run_pg_integration_tests.sh tests/test_skill_retrieval_pg.py

What only Postgres can show (the SQLite path is in tests/test_skill_retrieval.py):

* term frequencies and document lengths come from ``retrieval_tsv``'s
  positions: A (name, triggers) counts 1.0, B (description) 0.4; query terms
  are the lexemes of ``to_tsvector('english', query)``;
* above ``MIRIX_SKILL_BM25_MAX_DOCS`` only the documents holding a query term
  are scored (GIN ``@@``) while N, df and avgdl are aggregated over EVERY
  candidate: the prefilter changes no score, no rank and no result;
* cosine comes from pgvector, the gate and the BM25 fallback work on it;
* scopes, owners and soft-deleted rows are filtered in SQL on both paths;
* all the statements of one ranking read one snapshot;
* the per-field keyword search is a single OR query (the old "AND first, OR
  when fewer than 10" pass is gone).

Skill rows are inserted directly with chosen vectors (cosine to the query
vector (1, 0, ...) = the number named); nothing reaches a model provider.
"""

import json
import math
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import text

from mirix.settings import settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.skipif(
        not settings.mirix_pg_uri_no_default,
        reason="needs Postgres: run via scripts/run_pg_integration_tests.sh",
    ),
]

SCOPE = "test"
DIM = 4096
Q = [1.0, 0.0]


def vec(cosine):
    return [cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine))] + [0.0] * (DIM - 2)


async def _execute(sql: str, **params):
    from mirix.server.server import engine

    async with engine.begin() as conn:
        await conn.execute(text(sql), params)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def schema():
    from mirix.orm.base import Base
    from mirix.server.server import engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


async def _new_user(org):
    from mirix.schemas.user import User as PydanticUser

    uid = f"user-{uuid.uuid4().hex[:8]}"
    await _execute(
        "INSERT INTO users (id, name, organization_id, status, timezone, is_admin) "
        "VALUES (:u, :u, :o, 'active', 'UTC', FALSE)",
        u=uid,
        o=org,
    )
    return PydanticUser(id=uid, name=uid, organization_id=org, timezone="UTC")


@pytest_asyncio.fixture(loop_scope="module")
async def t(schema):
    from mirix.services.procedural_memory_manager import ProceduralMemoryManager

    org = f"org-{uuid.uuid4().hex[:8]}"
    await _execute("INSERT INTO organizations (id, name) VALUES (:o, :o)", o=org)
    user = await _new_user(org)

    async def no_sleep(_s):
        return None

    return SimpleNamespace(
        org=org,
        user=user,
        mgr=ProceduralMemoryManager(sleep=no_sleep),
        # Skill ids are "<tag><name>": unique across tests sharing the database,
        # in the same relative (id tie-break) order as the names.
        tag=f"{uuid.uuid4().hex[:8]}-",
    )


async def add(
    t,
    skill_id,
    *,
    name,
    description="",
    triggers=(),
    instructions="Do it.",
    scope=SCOPE,
    cosine=None,
    deleted=False,
    user=None,
):
    user = user or t.user
    vector = vec(cosine) if cosine is not None else None
    skill_id = t.tag + skill_id
    await _execute(
        "INSERT INTO procedural_memory (id, organization_id, user_id, name, entry_type, "
        "description, instructions, triggers, examples, version, filter_tags, last_modify, "
        "is_deleted, retrieval_embedding) VALUES (:id, :o, :u, :n, 'workflow', :d, :i, "
        "CAST(:tr AS json), '[]', '0.1.0', CAST(:ft AS json), '{}', :del, "
        "CAST(CAST(:v AS text) AS vector))",
        id=skill_id,
        o=user.organization_id,
        u=user.id,
        n=name,
        d=description,
        i=instructions,
        tr=json.dumps(list(triggers)),
        ft=json.dumps({"scope": scope} if scope is not None else {}),
        **{"del": deleted},
        v=json.dumps(vector) if vector is not None else None,
    )
    return skill_id


def _local(skill_id):
    return skill_id.split("-", 1)[1]


def _ids(result):
    return [_local(r.skill.id) for r in result.skills]


def _trace(result, skill_id):
    return next(c for c in result.diagnostics.candidates if _local(c.id) == skill_id)


async def test_frequencies_and_lengths_come_from_the_tsvector_positions(t):
    await add(
        t,
        "pos-a",
        name="deploy",
        triggers=["on friday"],
        description="deploy service",
        cosine=0.9,
    )
    await add(t, "pos-b", name="cook", description="cook pasta dinner", cosine=0.1)
    result = await t.mgr.retrieve_skills(
        "Deploying services", t.user, [SCOPE], method="bm25"
    )
    d = result.diagnostics
    assert d.backend == "postgresql"
    # to_tsvector('english', ...) stems the query.
    assert d.bm25.query_terms == ["deploy", "servic"]
    # pos-a: deploy A 1.0 + friday A 1.0 + deploy B 0.4 + servic B 0.4 = 2.8
    # pos-b: cook A 1.0 + cook B 0.4 + pasta B 0.4 + dinner B 0.4 = 2.2
    assert d.bm25.n_docs == 2
    assert d.bm25.avgdl == pytest.approx((2.8 + 2.2) / 2)
    assert d.bm25.doc_freq == {"deploy": 1, "servic": 1}
    k1, b, avgdl = 1.2, 0.75, 2.5
    idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))

    def part(tf, dl):
        return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))

    assert _trace(result, "pos-a").bm25 == pytest.approx(
        part(1.4, 2.8) + part(0.4, 2.8)
    )
    assert _trace(result, "pos-b").bm25 == 0.0
    assert _ids(result) == ["pos-a"]


async def test_common_words_are_down_weighted_by_idf(t):
    await add(t, "idf-a", name="deploy deploy deploy", description="deploy deploy")
    await add(t, "idf-b", name="deploy kubernetes")
    for i in range(4):
        await add(t, f"idf-f{i}", name=f"deploy filler{i}")
    result = await t.mgr.retrieve_skills(
        "deploy kubernetes", t.user, [SCOPE], method="bm25"
    )
    assert _ids(result)[0] == "idf-b"
    assert result.diagnostics.bm25.doc_freq == {"deploy": 6, "kubernet": 1}
    assert _trace(result, "idf-a").bm25 < 0.5 * _trace(result, "idf-b").bm25


async def _prefilter_corpus(t):
    """40 skills: 10 hold "kubernetes", 25 "deploy", 8 "rollout"; lengths vary."""
    for i in range(40):
        words = [f"topic{i}"]
        if i % 4 == 0:
            words.append("kubernetes")
        if i % 8 < 5:
            words.append("deploy")
        if i % 5 == 1:
            words.append("rollout")
        description = " ".join(f"note{j}" for j in range(i % 6))
        await add(
            t,
            f"pf-{i:02d}",
            name=" ".join(words),
            description=description,
            cosine=round(0.3 + (i % 13) * 0.05, 2),
        )


@pytest.mark.parametrize("method", ["bm25", "hybrid"])
async def test_the_gin_prefilter_does_not_distort_idf(t, monkeypatch, method):
    from mirix import constants

    await _prefilter_corpus(t)
    query = "deploy kubernetes rollout"

    async def run(max_docs):
        monkeypatch.setattr(constants, "SKILL_BM25_MAX_DOCS", max_docs)
        return await t.mgr.retrieve_skills(
            query,
            t.user,
            [SCOPE],
            limit=15,
            method=method,
            query_embedding=Q if method == "hybrid" else None,
        )

    full = await run(20000)
    pre = await run(5)
    bf, bp = full.diagnostics.bm25, pre.diagnostics.bm25
    assert (bf.prefiltered, bp.prefiltered) == (False, True)
    assert bf.n_docs == bp.n_docs == 40  # N over EVERY candidate
    assert bf.doc_freq == bp.doc_freq == {"deploy": 25, "kubernet": 10, "rollout": 8}
    assert bp.avgdl == pytest.approx(bf.avgdl, rel=1e-12)
    matching = sum(1 for i in range(40) if i % 4 == 0 or i % 8 < 5 or i % 5 == 1)
    assert bf.scored_docs == 40 and bp.scored_docs == matching < 40
    # Every score, rank and result is identical.
    for a, b in zip(full.diagnostics.candidates, pre.diagnostics.candidates):
        assert a.id == b.id
        assert a.bm25 == pytest.approx(b.bm25, rel=1e-12, abs=1e-12)
        assert (a.bm25_rank, a.cosine_rank, a.rrf_rank) == (
            b.bm25_rank,
            b.cosine_rank,
            b.rrf_rank,
        )
        assert (
            a.cosine == pytest.approx(b.cosine)
            if a.cosine is not None
            else b.cosine is None
        )
    assert _ids(full) == _ids(pre) and len(_ids(full)) == 15
    if method == "hybrid":
        # The prefilter limits BM25 scoring only: every candidate keeps its cosine.
        assert all(c.cosine is not None for c in pre.diagnostics.candidates)
        assert len(pre.diagnostics.candidates) == 40
    # Statistics of the matching subset alone WOULD have moved the scores.
    from mirix.services.skill_retrieval import bm25_idf

    assert bm25_idf(matching, 25) != pytest.approx(bm25_idf(40, 25))


async def test_prefilter_quotes_every_lexeme_verbatim(t, monkeypatch):
    """Lexemes with quotes, backslashes, & or | reach the GIN filter intact."""
    from mirix import constants

    odd = "o'reilly C:\\temp http://a.com/x?y=1&z=2 pipe|bar"
    await add(t, "odd-a", name="odd names", description=odd)
    for i in range(6):
        await add(t, f"odd-f{i}", name=f"filler{i}")
    monkeypatch.setattr(constants, "SKILL_BM25_MAX_DOCS", 2)
    pre = await t.mgr.retrieve_skills(odd, t.user, [SCOPE], method="bm25")
    monkeypatch.setattr(constants, "SKILL_BM25_MAX_DOCS", 20000)
    full = await t.mgr.retrieve_skills(odd, t.user, [SCOPE], method="bm25")
    assert pre.diagnostics.bm25.prefiltered
    assert _ids(pre) == _ids(full) == ["odd-a"]
    assert _trace(pre, "odd-a").bm25 == pytest.approx(_trace(full, "odd-a").bm25)


async def test_the_gate_can_return_nothing_and_keeps_the_skills_near_the_best(t):
    for skill_id, cosine in (("g-a", 0.55), ("g-b", 0.4)):
        await add(t, skill_id, name=f"brew coffee {skill_id}", cosine=cosine)
    gated = await t.mgr.retrieve_skills(
        "brew coffee", t.user, [SCOPE], apply_gate=True, query_embedding=Q
    )
    assert gated.skills == [] and gated.diagnostics.gate.passed == 0
    ungated = await t.mgr.retrieve_skills(
        "brew coffee", t.user, [SCOPE], query_embedding=Q
    )
    assert set(_ids(ungated)) == {"g-a", "g-b"}
    for skill_id, cosine in (("g-c", 0.95), ("g-d", 0.9), ("g-e", 0.8)):
        await add(t, skill_id, name=f"brew coffee {skill_id}", cosine=cosine)
    relative = await t.mgr.retrieve_skills(
        "brew coffee", t.user, [SCOPE], apply_gate=True, query_embedding=Q
    )
    assert sorted(_ids(relative)) == ["g-c", "g-d"]
    assert relative.diagnostics.gate.max_cosine == pytest.approx(0.95, abs=1e-6)


async def test_a_failed_query_vector_falls_back_to_bm25_without_gate(t, monkeypatch):
    import mirix.services.procedural_memory_manager as pmm
    from mirix.schemas.embedding_config import EmbeddingConfig

    await add(t, "fb-hit", name="restart pods", cosine=0.1)
    await add(t, "fb-miss", name="bake bread", cosine=0.99)

    class Down:
        async def get_text_embedding(self, text_value):
            raise ConnectionError("embedding endpoint down")

    async def factory(config, *args, **kwargs):
        return Down()

    monkeypatch.setattr(pmm, "embedding_model", factory)
    result = await t.mgr.retrieve_skills(
        "restart pods",
        t.user,
        [SCOPE],
        apply_gate=True,
        embedding_config=EmbeddingConfig(
            embedding_endpoint_type="openai",
            embedding_endpoint="http://127.0.0.1:9/v1",
            embedding_model="x",
            embedding_dim=2,
        ),
    )
    d = result.diagnostics
    assert (d.embedding_status, d.method, d.lanes) == ("failed", "bm25", ["bm25"])
    assert d.gate.applied is False
    assert _ids(result) == ["fb-hit"]
    assert result.skills[0].scores.cosine is None


async def test_ties_are_broken_by_id(t):
    for skill_id in ("tie-c", "tie-a", "tie-d", "tie-b"):
        await add(t, skill_id, name=f"clean kitchen x{skill_id[-1]}", cosine=0.7)
    first = await t.mgr.retrieve_skills(
        "clean kitchen", t.user, [SCOPE], query_embedding=Q
    )
    again = await t.mgr.retrieve_skills(
        "clean kitchen", t.user, [SCOPE], query_embedding=Q
    )
    assert _ids(first) == ["tie-a", "tie-b", "tie-c", "tie-d"] == _ids(again)


@pytest.mark.parametrize("max_docs", [20000, 1])
async def test_scopes_owners_and_deleted_rows_are_filtered_on_both_paths(
    t, monkeypatch, max_docs
):
    from mirix import constants

    monkeypatch.setattr(constants, "SKILL_BM25_MAX_DOCS", max_docs)
    await add(t, "iso-mine", name="walk dog", cosine=0.9)
    await add(t, "iso-mine-2", name="walk dog daily", cosine=0.85)
    await add(t, "iso-scope", name="walk dog o", scope="other", cosine=0.9)
    await add(t, "iso-deleted", name="walk dog d", cosine=0.99, deleted=True)
    other = await _new_user(t.org)
    await add(t, "iso-other-user", name="walk dog", cosine=0.9, user=other)
    result = await t.mgr.retrieve_skills(
        "walk dog", t.user, [SCOPE], apply_gate=True, query_embedding=Q
    )
    assert sorted(_ids(result)) == ["iso-mine", "iso-mine-2"]
    d = result.diagnostics
    assert d.candidate_count == d.bm25.n_docs == 2
    assert d.bm25.prefiltered is (max_docs == 1)
    assert {_local(c.id) for c in d.candidates} == {"iso-mine", "iso-mine-2"}
    assert (
        await t.mgr.retrieve_skills("walk dog", t.user, [], query_embedding=Q)
    ).skills == []


@pytest.mark.parametrize("max_docs", [20000, 1])
async def test_missing_vectors_are_counted_on_both_paths(t, monkeypatch, max_docs):
    from mirix import constants

    monkeypatch.setattr(constants, "SKILL_BM25_MAX_DOCS", max_docs)
    await add(t, "mv-none", name="fix bike chain")
    await add(t, "mv-none-2", name="unrelated words")
    await add(t, "mv-has", name="fix bike", cosine=0.9)
    result = await t.mgr.retrieve_skills(
        "fix bike chain", t.user, [SCOPE], apply_gate=True, query_embedding=Q
    )
    assert result.diagnostics.missing_vector_count == 2
    assert _ids(result) == ["mv-has"]
    assert _trace(result, "mv-none").gate_passed is False


async def test_one_ranking_reads_one_snapshot(t, monkeypatch):
    """A skill committed between the ranking's statements is invisible to all
    of them: N, df and avgdl describe exactly the candidates ranked."""
    from mirix import constants
    from mirix.server.server import db_context

    for i in range(6):
        await add(t, f"snap-{i}", name=f"sweep floor {i}", cosine=0.7)
    monkeypatch.setattr(constants, "SKILL_BM25_MAX_DOCS", 2)  # several statements
    executed = []

    class Session:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        async def execute(self, *args, **kwargs):
            result = await self._inner.execute(*args, **kwargs)
            executed.append(1)
            if len(executed) == 2:  # after the query terms and the count
                await add(t, "snap-late", name="sweep floor late", cosine=0.99)
            return result

    @asynccontextmanager
    async def spying():
        async with db_context() as session:
            yield Session(session)

    from mirix.services import skill_retrieval

    result = await skill_retrieval.retrieve_skills(
        "sweep floor",
        t.user,
        [SCOPE],
        session_maker=spying,
        query_embedding=Q,
    )
    assert len(executed) >= 5
    d = result.diagnostics
    assert d.bm25.prefiltered
    assert d.candidate_count == d.bm25.n_docs == 6
    assert "snap-late" not in {_local(c.id) for c in d.candidates}
    # A new ranking sees it (and ranks it first by cosine).
    later = await t.mgr.retrieve_skills(
        "sweep floor", t.user, [SCOPE], query_embedding=Q
    )
    assert later.diagnostics.candidate_count == 7
    assert _trace(later, "snap-late").cosine_rank == 1


async def test_the_per_field_keyword_search_is_one_or_query(t):
    """The old pass returned only the AND matches once there were 10 of them;
    one OR query also finds the skill that holds only one of the words."""
    from mirix.schemas.agent import AgentState, AgentType
    from mirix.schemas.embedding_config import EmbeddingConfig
    from mirix.schemas.llm_config import LLMConfig

    for i in range(10):
        await add(t, f"or-both-{i}", name=f"n{i}", instructions="alpha beta steps")
    await add(t, "or-one", name="lonely", instructions="alpha only")
    agent_state = AgentState(
        id="agent-or",
        name="a",
        system="s",
        agent_type=AgentType.procedural_memory_agent,
        llm_config=LLMConfig(
            model="x", model_endpoint_type="openai", context_window=8000
        ),
        embedding_config=EmbeddingConfig(
            embedding_endpoint_type="openai",
            embedding_endpoint="http://127.0.0.1:9/v1",
            embedding_model="x",
            embedding_dim=2,
        ),
        tools=[],
    )
    items = await t.mgr.list_procedures(
        agent_state=agent_state,
        user=t.user,
        query="alpha beta",
        search_field="instructions",
        search_method="bm25",
        scopes=[SCOPE],
        limit=50,
    )
    ids = [_local(i.id) for i in items]
    assert len(ids) == 11 and ids[-1] == "or-one"  # both-word matches rank higher
    # And the hybrid list path is the skill ranking on Postgres too.
    ranked = await t.mgr.list_procedures(
        agent_state=agent_state,
        user=t.user,
        query="lonely",
        embedded_text=Q,
        search_method="hybrid",
        scopes=[SCOPE],
        limit=3,
    )
    assert _local(ranked[0].id) == "or-one"


async def _create(t, name, triggers, description="Store the fruit."):
    """A skill through the one write path, as the curator writes it (the
    triggers go through SQLAlchemy's JSON serializer: ensure_ascii escapes)."""
    result = await t.mgr.apply_skill_change(
        "create",
        actor=None,
        user_id=t.user.id,
        fields=dict(
            name=name,
            entry_type="workflow",
            description=description,
            instructions="Do it.",
            triggers=list(triggers),
            examples=[],
            filter_tags={"scope": SCOPE},
            organization_id=t.org,
        ),
    )
    return result.skill.id


async def _lexemes(skill_id):
    from mirix.server.server import engine

    async with engine.connect() as conn:
        value = (
            await conn.execute(
                text(
                    "SELECT array_to_string(tsvector_to_array(retrieval_tsv), '|') "
                    "FROM procedural_memory WHERE id = :i"
                ),
                {"i": skill_id},
            )
        ).scalar_one()
    return set(value.split("|"))


async def test_non_ascii_triggers_are_indexed_as_words(t):
    """`triggers` is a JSON column whose stored text escapes every
    non-ASCII character; indexing that text lost the words next to a curly
    quote or a dash ('u201ccool', 'u2014then') and every CJK or accented
    word. The index reads the decoded JSON, so they are words like any other
    and both backends agree."""
    curly = await _create(t, f"curly-{t.tag}", ["“cool” the apple—then store it"])
    ascii_twin = await _create(
        t, f"ascii-{t.tag}", ['"cool" the apple - then store it']
    )
    cjk = await _create(t, f"cjk-{t.tag}", ["导出表格", "café crème"])

    words = await _lexemes(curly)
    assert {"cool", "appl", "store"} <= words
    assert not any(w.startswith("u20") for w in words), words
    assert {"导出表格", "café", "crème"} <= await _lexemes(cjk)

    result = await t.mgr.retrieve_skills("cool store", t.user, [SCOPE], method="bm25")
    d = result.diagnostics
    assert d.bm25.doc_freq["cool"] == 2  # both twins hold the word
    scores = {c.id: c.bm25 for c in d.candidates}
    assert scores[curly] == pytest.approx(scores[ascii_twin]) and scores[curly] > 0
    ranked = await t.mgr.retrieve_skills("导出表格", t.user, [SCOPE], method="bm25")
    assert [r.skill.id for r in ranked.skills] == [cjk]
    accented = await t.mgr.retrieve_skills("café", t.user, [SCOPE], method="bm25")
    assert [r.skill.id for r in accented.skills] == [cjk]


async def test_control_characters_in_triggers_separate_words(t):
    """jsonb's text escapes a newline, a tab, a quote
    or a backslash (`["line1\\nline2"]`); read as it is, 'label' behind a
    newline was the lexeme 'nlabel' and 'budget' behind a tab 'tbudget'. The
    index turns the escapes into spaces, so the words are the ones the
    SQLite path's tokenizer finds, and a query for them matches."""
    from mirix.services.skill_retrieval import tokenize

    triggers = [
        "unpack boxes\nlabel shelves",
        "tab\tbudget\r\nreview",
        'say "hello" to C:\\drive',
    ]
    skill = await _create(t, f"ctl-{t.tag}", triggers, description="Move house.")
    words = await _lexemes(skill)
    assert not ({"nlabel", "tbudget", "nreview", "rnreview"} & words), words
    # Unstemmed words agree with the SQLite tokenizer's (Postgres stems:
    # 'boxes' -> 'box', 'shelves' -> 'shelv').
    for word in ("unpack", "label", "tab", "budget", "review", "hello", "drive"):
        assert word in words
        assert any(word in tokenize(trigger) for trigger in triggers)
    for query in ("label", "budget", "review"):
        result = await t.mgr.retrieve_skills(query, t.user, [SCOPE], method="bm25")
        assert [r.skill.id for r in result.skills] == [skill], query
        assert result.skills[0].scores.bm25 > 0


async def test_a_nul_in_a_trigger_is_a_clear_refusal_not_a_database_error(t):
    """The index decodes the triggers through JSONB,
    which cannot hold U+0000 (SQLSTATE 22P05, "unsupported Unicode escape
    sequence"). The write path refuses such a trigger first, by name; the
    database never sees it, and the user's other skills are untouched."""
    from mirix.services.procedural_memory_manager import SkillFieldInvalid

    kept = await _create(t, f"nul-ok-{t.tag}", ["a fine trigger"])
    with pytest.raises(SkillFieldInvalid, match=r"'triggers\[0\]'.*U\+0000"):
        await _create(t, f"nul-bad-{t.tag}", ["bad\x00trigger"])
    result = await t.mgr.retrieve_skills("fine", t.user, [SCOPE], method="bm25")
    assert [r.skill.id for r in result.skills] == [kept]


async def test_a_raw_conversation_query_scores_its_first_terms_only(t, monkeypatch):
    """The lexemes of a long query in the order they first appear,
    capped at MIRIX_SKILL_BM25_MAX_QUERY_TERMS; a document holding none of the
    kept terms is never scanned per lexeme and scores 0, the others exactly
    as before."""
    from mirix import constants

    monkeypatch.setattr(constants, "SKILL_BM25_MAX_QUERY_TERMS", 3)
    await add(t, "cap-a", name="prune roses", triggers=["garden"], cosine=0.5)
    await add(t, "cap-b", name="paint fence", cosine=0.5)
    result = await t.mgr.retrieve_skills(
        "prune the roses in the garden, then paint the fence",
        t.user,
        [SCOPE],
        method="bm25",
    )
    bm25 = result.diagnostics.bm25
    # to_tsvector order: prune, rose, garden | paint, fenc (dropped).
    assert bm25.query_terms == ["garden", "prune", "rose"]
    assert bm25.query_terms_dropped == 2
    assert bm25.doc_freq == {"garden": 1, "prune": 1, "rose": 1}
    assert _ids(result) == ["cap-a"]
    assert _trace(result, "cap-b").bm25 == 0.0
