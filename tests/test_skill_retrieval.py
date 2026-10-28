"""Hermetic tests of the skill retrieval pipeline.

``retrieve_skills`` ranks a user's live skills in the caller's scopes: true
BM25 (N, df and avgdl over ALL candidates), cosine on the retrieval vector,
RRF (k = 60), the similarity gate only when asked, then the top ``limit`` — possibly
none. These tests run on SQLite (the Python-tokenizer path, scored with
rank_bm25) and on the pure scoring and ranking functions; the Postgres path
(``retrieval_tsv`` positions, the GIN prefilter above
``MIRIX_SKILL_BM25_MAX_DOCS``) is in tests/test_skill_retrieval_pg.py.

Skill vectors are chosen so that their cosine to the query vector (1, 0, ...)
is exactly the number the test names. No model provider is ever reached: the
query vector is given, comes from an in-process fake, or from the offline fake
server.
"""

import json
import math
import random
import uuid
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import mirix.orm  # noqa: F401 -- register every ORM class before create_all
import mirix.services.procedural_memory_manager as pmm
from mirix import constants
from mirix.constants import MAX_EMBEDDING_DIM
from mirix.orm.base import Base
from mirix.orm.organization import Organization as OrganizationORM
from mirix.orm.procedural_memory import ProceduralMemoryItem as ProceduralORM
from mirix.orm.user import User as UserORM
from mirix.schemas.agent import AgentState, AgentType
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.user import User as PydanticUser
from mirix.services import skill_retrieval as sr
from mirix.services.procedural_memory_manager import ProceduralMemoryManager

# Async tests share the module's event loop (and SQLite engine).
asyncio_module = pytest.mark.asyncio(loop_scope="module")

SCOPE = "test"
Q = [1.0, 0.0]  # the query vector; a skill vector's cosine to it is its x


def vec(cosine: float) -> list:
    """A unit vector whose cosine to Q is ``cosine``, padded to 4096."""
    return [cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine))] + [0.0] * (
        MAX_EMBEDDING_DIM - 2
    )


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sm(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("skill_retrieval") / "test.db"
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


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def org(sm):
    org_id = f"sr-org-{uuid.uuid4().hex[:8]}"
    async with sm() as session:
        session.add(OrganizationORM(id=org_id, name=org_id))
        await session.commit()
    return org_id


async def _new_user(sm, org_id) -> PydanticUser:
    user_id = f"sr-user-{uuid.uuid4().hex[:8]}"
    async with sm() as session:
        session.add(
            UserORM(
                id=user_id,
                name=user_id,
                organization_id=org_id,
                status="active",
                timezone="UTC",
            )
        )
        await session.commit()
    return PydanticUser(
        id=user_id, name=user_id, organization_id=org_id, timezone="UTC"
    )


@pytest_asyncio.fixture(loop_scope="module")
async def user(sm, org):
    return await _new_user(sm, org)


async def add_skill(
    sm,
    user,
    skill_id,
    *,
    name,
    description="",
    triggers=(),
    instructions="Do it.",
    scope=SCOPE,
    cosine=None,
    vector=None,
    deleted=False,
    tags=None,
):
    filter_tags = dict(tags or {})
    if scope is not None:
        filter_tags["scope"] = scope
    if vector is None and cosine is not None:
        vector = vec(cosine)
    async with sm() as session:
        session.add(
            ProceduralORM(
                id=skill_id,
                organization_id=user.organization_id,
                user_id=user.id,
                name=name,
                entry_type="workflow",
                description=description,
                instructions=instructions,
                triggers=list(triggers),
                examples=[],
                filter_tags=filter_tags or None,
                retrieval_embedding=vector,
                is_deleted=deleted,
            )
        )
        await session.commit()
    return skill_id


async def _no_sleep(_seconds):
    return None


@pytest.fixture
def mgr(sm):
    return ProceduralMemoryManager(session_maker=sm, sleep=_no_sleep)


def _ids(result):
    return [r.skill.id for r in result.skills]


def _trace(result, skill_id):
    return next(c for c in result.diagnostics.candidates if c.id == skill_id)


# --------------------------------------------------------------------------- #
# BM25 (pure)                                                                  #
# --------------------------------------------------------------------------- #


class TestBM25:
    def test_idf_is_the_spec_formula_and_never_negative(self):
        assert sr.bm25_idf(10, 1) == pytest.approx(math.log(1 + 9.5 / 1.5))
        assert sr.bm25_idf(10, 10) == pytest.approx(math.log(1 + 0.5 / 10.5))
        assert sr.bm25_idf(10, 10) > 0  # a term in every document still counts a little
        idfs = [sr.bm25_idf(100, df) for df in range(0, 101)]
        assert idfs == sorted(idfs, reverse=True)  # rarer terms weigh more

    def test_scores_are_okapi_with_weighted_frequencies(self):
        # name "deploy" (A) + description "deploy service" (B):
        # tf(deploy) = 1.0 + 0.4, tf(servic) = 0.4, length 1.8.
        docs = [{"deploy": 1.4, "servic": 0.4}, {"deploy": 2.0}, {"cook": 1.0}]
        lens = [1.8, 2.0, 1.0]
        scorer = sr.SkillBM25.from_documents(docs, lens, ["deploy", "servic"])
        n, avgdl = 3, (1.8 + 2.0 + 1.0) / 3
        k1, b = 1.2, 0.75

        def term(tf, dl, df):
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))

        expected = [
            term(1.4, 1.8, 2) + term(0.4, 1.8, 1),
            term(2.0, 2.0, 2),
            0.0,
        ]
        assert scorer.score(["deploy", "servic"]) == pytest.approx(expected)
        assert (scorer.k1, scorer.b) == (
            constants.SKILL_BM25_K1,
            constants.SKILL_BM25_B,
        )

    def test_scoring_a_subset_with_the_full_statistics_changes_no_score(self):
        """The prefilter's contract, backend-free: scoring only the documents
        that hold a query term, with N/df/avgdl of ALL documents, gives each of
        them exactly the score it gets when every document is scored."""
        rng = random.Random(7)
        vocabulary = ["alpha", "beta", "gamma", "delta", "eps", "zeta"]
        docs, lens = [], []
        for _ in range(60):
            tf = {}
            for word in rng.sample(vocabulary, rng.randint(1, 4)):
                tf[word] = rng.choice([1.0, 0.4, 1.4, 2.0])
            docs.append(tf)
            lens.append(sum(tf.values()) + rng.choice([0.0, 0.4, 1.0]))
        query = ["alpha", "zeta"]
        full = sr.SkillBM25.from_documents(docs, lens, query).score(query)
        stats = sr.corpus_stats(docs, lens, query)
        hits = [i for i, tf in enumerate(docs) if any(t in tf for t in query)]
        assert 0 < len(hits) < len(docs)
        subset = sr.SkillBM25(
            [docs[i] for i in hits],
            [lens[i] for i in hits],
            n_docs=stats.n_docs,
            doc_freq=stats.doc_freq,
            avgdl=stats.avgdl,
        ).score(query)
        assert subset == pytest.approx([full[i] for i in hits], rel=0, abs=1e-12)
        # ...whereas statistics of the subset alone WOULD distort them.
        distorted = sr.SkillBM25.from_documents(
            [docs[i] for i in hits], [lens[i] for i in hits], query
        ).score(query)
        assert distorted != pytest.approx(subset)

    def test_degenerate_corpora_score_zero(self):
        assert sr.SkillBM25.from_documents([], [], ["a"]).score(["a"]) == []
        empty_docs = sr.SkillBM25.from_documents([{}, {}], [0.0, 0.0], ["a"])
        assert empty_docs.score(["a"]) == [0.0, 0.0]
        assert sr.SkillBM25.from_documents([{"a": 1.0}], [1.0], []).score([]) == [0.0]

    def test_tokenizer(self):
        assert sr.tokenize("Deploy the Deploy-Production service, NOW!") == [
            "deploy",
            "deploy",
            "production",
            "service",
        ]
        assert sr.tokenize(None) == [] and sr.tokenize("the a of") == []
        assert sr.query_terms_python("kubectl the Kubectl pods") == ["kubectl", "pods"]

    def test_the_query_terms_are_capped_in_query_order(self, monkeypatch):
        # The first MIRIX_SKILL_BM25_MAX_QUERY_TERMS distinct terms, in
        # the order they appear; the rest counted, never scored.
        assert sr.cap_query_terms(["b", "a", "b", "c"], 2) == (["a", "b"], 1)
        assert sr.cap_query_terms([], 2) == ([], 0)
        monkeypatch.setattr(sr.constants, "SKILL_BM25_MAX_QUERY_TERMS", 3)
        assert sr.cap_query_terms(["z", "y", "x", "w", "z"]) == (["x", "y", "z"], 1)

    def test_weighted_frequencies_follow_the_retrieval_tsv_weights(self):
        tf, length = sr.weighted_term_frequencies(
            sr.skill_weighted_texts("deploy", ["deploy on friday"], "deploy service")
        )
        # name + trigger at A (1.0 each), description at B (0.4 each).
        assert tf == pytest.approx({"deploy": 2.4, "friday": 1.0, "service": 0.4})
        assert length == pytest.approx(3.8)
        tf, _ = sr.weighted_term_frequencies(
            sr.skill_weighted_texts("x", "a trigger", None)
        )
        assert tf == pytest.approx({"x": 1.0, "trigger": 1.0})


# --------------------------------------------------------------------------- #
# Fusion and gate (pure)                                                       #
# --------------------------------------------------------------------------- #


def _cand(skill_id, cosine=None):
    return sr._Candidate(id=skill_id, has_vector=cosine is not None, cosine=cosine)


def _rank(cands, bm25, **kw):
    params = dict(
        lanes=[sr.LANE_BM25, sr.LANE_COSINE],
        apply_gate=False,
        limit=10,
        rrf_k=60,
        min_similarity=0.6,
        relative_ratio=0.9,
    )
    params.update(kw)
    return sr.rank_candidates(cands, bm25, **params)


class TestFusionAndGate:
    def test_rrf_sums_one_over_k_plus_rank_per_lane(self):
        cands = [_cand("a", 0.9), _cand("b", 0.8), _cand("c", 0.95)]
        ranking = _rank(cands, {"a": 3.0, "b": 5.0, "c": 0.0})
        traces = {t.id: t for t in ranking.traces}
        # bm25 lane: b(1), a(2); c scored 0 -> not ranked. cosine: c, a, b.
        assert (traces["a"].bm25_rank, traces["a"].cosine_rank) == (2, 2)
        assert (traces["b"].bm25_rank, traces["b"].cosine_rank) == (1, 3)
        assert (traces["c"].bm25_rank, traces["c"].cosine_rank) == (None, 1)
        assert traces["a"].rrf == pytest.approx(2 / 62)
        assert traces["b"].rrf == pytest.approx(1 / 61 + 1 / 63)
        assert traces["c"].rrf == pytest.approx(1 / 61)
        # 1/61 + 1/63 > 2/62: b edges out a.
        assert [t.id for t in ranking.returned] == ["b", "a", "c"]
        assert [t.rrf_rank for t in ranking.traces] == [1, 2, 3]

    def test_the_order_is_deterministic_whatever_the_input_order(self):
        # Four exact ties on every score: the id decides.
        cands = [_cand(i, 0.7) for i in ("d", "b", "a", "c")]
        bm25 = {i: 1.0 for i in "abcd"}
        orders = set()
        for seed in range(10):
            shuffled = list(cands)
            random.Random(seed).shuffle(shuffled)
            orders.add(tuple(t.id for t in _rank(shuffled, bm25).returned))
        assert orders == {("a", "b", "c", "d")}

    def test_an_rrf_tie_is_broken_by_cosine_then_bm25_then_id(self):
        # x: bm25 rank 1 only; y: cosine rank 1 only -> equal RRF 1/61.
        cands = [_cand("x"), _cand("y", 0.5)]
        ranking = _rank(cands, {"x": 2.0, "y": 0.0})
        assert ranking.traces[0].rrf == ranking.traces[1].rrf
        assert [t.id for t in ranking.returned] == ["y", "x"]  # a cosine beats none

    def test_gate_absolute_and_relative(self):
        cands = [_cand("a", 0.95), _cand("b", 0.9), _cand("c", 0.8), _cand("d", 0.55)]
        ranking = _rank(cands, {}, apply_gate=True)
        # threshold = max(0.6, 0.95 * 0.9 = 0.855): a and b only.
        assert [t.id for t in ranking.returned] == ["a", "b"]
        assert ranking.gate.applied and ranking.gate.passed == 2
        assert ranking.gate.max_cosine == pytest.approx(0.95)
        assert ranking.gate.threshold == pytest.approx(0.855)
        by_id = {t.id: t.gate_passed for t in ranking.traces}
        assert by_id == {"a": True, "b": True, "c": False, "d": False}

    def test_the_gate_can_return_nothing(self):
        cands = [_cand("a", 0.59), _cand("b", 0.3)]
        gated = _rank(cands, {"a": 1.0, "b": 2.0}, apply_gate=True)
        assert gated.returned == [] and gated.gate.passed == 0
        ungated = _rank(cands, {"a": 1.0, "b": 2.0})
        assert len(ungated.returned) == 2
        assert all(t.gate_passed is None for t in ungated.traces)

    def test_a_skill_without_a_vector_never_passes_the_gate(self):
        cands = [_cand("vectorless"), _cand("b", 0.9)]
        ranking = _rank(cands, {"vectorless": 9.0, "b": 0.1}, apply_gate=True)
        assert [t.id for t in ranking.returned] == ["b"]
        # With no vector at all anywhere, nothing passes.
        none = _rank(
            [_cand("v1"), _cand("v2")], {"v1": 1.0, "v2": 1.0}, apply_gate=True
        )
        assert none.returned == [] and none.gate.threshold is None

    def test_without_a_cosine_lane_no_gate_is_applied(self):
        cands = [_cand("a", 0.1), _cand("b", 0.2)]
        ranking = _rank(
            cands, {"a": 2.0, "b": 1.0}, lanes=[sr.LANE_BM25], apply_gate=True
        )
        assert ranking.gate.applied is False
        assert ranking.gate.skipped_reason == "no query vector"
        assert [t.id for t in ranking.returned] == ["a", "b"]
        assert all(t.cosine is None and t.cosine_rank is None for t in ranking.traces)

    def test_limit_cuts_after_the_gate(self):
        cands = [_cand(f"s{i}", 0.99 - i * 0.001) for i in range(20)]
        ranking = _rank(cands, {}, apply_gate=True, limit=3)
        assert [t.id for t in ranking.returned] == ["s0", "s1", "s2"]
        assert ranking.gate.passed == 20
        assert _rank(cands, {}, limit=0).returned == []

    def test_max_cosine_distance_keeps_distant_skills_out_of_the_cosine_lane(self):
        cands = [_cand("near", 0.9), _cand("far", 0.2)]
        ranking = _rank(cands, {}, max_cosine_distance=0.5)
        assert [t.id for t in ranking.returned] == ["near"]
        far = next(t for t in ranking.traces if t.id == "far")
        assert far.cosine == pytest.approx(0.2) and far.cosine_rank is None


# --------------------------------------------------------------------------- #
# retrieve_skills on SQLite                                                    #
# --------------------------------------------------------------------------- #


@asyncio_module
class TestRetrieveSkills:
    async def test_common_words_are_down_weighted_by_idf(self, sm, user, mgr):
        # "deploy" is in every skill, "kubernetes" in one. The deploy-heavy
        # skill repeats the common word; the rare word still wins.
        await add_skill(
            sm, user, "idf-a", name="deploy deploy deploy", description="deploy deploy"
        )
        await add_skill(sm, user, "idf-b", name="deploy kubernetes")
        for i in range(4):
            await add_skill(sm, user, f"idf-f{i}", name=f"deploy filler{i}")
        result = await mgr.retrieve_skills(
            "deploy kubernetes", user, [SCOPE], method="bm25"
        )
        assert _ids(result)[0] == "idf-b"
        bm25 = result.diagnostics.bm25
        assert bm25.doc_freq == {"deploy": 6, "kubernetes": 1}
        assert bm25.n_docs == 6
        assert sr.bm25_idf(6, 6) < sr.bm25_idf(6, 1)
        assert _trace(result, "idf-b").bm25 > _trace(result, "idf-a").bm25
        # The repeated common word alone scores well below the rare hit.
        assert _trace(result, "idf-a").bm25 < 0.5 * _trace(result, "idf-b").bm25

    async def test_hybrid_fuses_bm25_and_cosine_with_rrf(self, sm, user, mgr):
        await add_skill(sm, user, "h-a", name="rotate logs", cosine=0.62)
        await add_skill(sm, user, "h-b", name="archive files", cosine=0.95)
        await add_skill(sm, user, "h-c", name="rotate keys safely", cosine=0.80)
        result = await mgr.retrieve_skills("rotate", user, [SCOPE], query_embedding=Q)
        d = result.diagnostics
        assert d.method == "hybrid" and d.lanes == ["bm25", "cosine"]
        assert d.embedding_status == "ok" and d.backend == "sqlite"
        a, b, c = (_trace(result, i) for i in ("h-a", "h-b", "h-c"))
        # bm25: a (shorter doc) before c; b has no "rotate". cosine: b, c, a.
        assert (a.bm25_rank, c.bm25_rank, b.bm25_rank) == (1, 2, None)
        assert (b.cosine_rank, c.cosine_rank, a.cosine_rank) == (1, 2, 3)
        assert a.rrf == pytest.approx(1 / 61 + 1 / 63)
        assert c.rrf == pytest.approx(2 / 62)
        assert b.rrf == pytest.approx(1 / 61)
        # 1/61 + 1/63 edges out 2/62: agreement at the top beats two seconds.
        assert _ids(result) == ["h-a", "h-c", "h-b"]
        scores = {r.skill.id: r.scores.to_dict() for r in result.skills}
        assert scores["h-b"] == {
            "bm25": 0.0,
            "cosine": pytest.approx(0.95),
            "rrf": pytest.approx(1 / 61),
        }
        assert set(scores["h-a"]) == {"bm25", "cosine", "rrf"}

    async def test_the_gate_can_return_no_skill_while_search_returns_top_k(
        self, sm, user, mgr
    ):
        for i, cosine in enumerate((0.55, 0.4, 0.1)):
            await add_skill(sm, user, f"g0-{i}", name=f"brew coffee {i}", cosine=cosine)
        gated = await mgr.retrieve_skills(
            "brew coffee", user, [SCOPE], apply_gate=True, query_embedding=Q
        )
        assert gated.skills == []
        assert gated.diagnostics.gate.applied and gated.diagnostics.gate.passed == 0
        assert gated.diagnostics.returned_ids == []
        ungated = await mgr.retrieve_skills(
            "brew coffee", user, [SCOPE], limit=2, query_embedding=Q
        )
        assert len(ungated.skills) == 2
        assert ungated.diagnostics.gate.applied is False

    async def test_the_relative_gate_keeps_the_skills_near_the_best(
        self, sm, user, mgr
    ):
        for skill_id, cosine in (
            ("r-a", 0.95),
            ("r-b", 0.9),
            ("r-c", 0.8),
            ("r-d", 0.61),
        ):
            await add_skill(sm, user, skill_id, name=f"water {skill_id}", cosine=cosine)
        result = await mgr.retrieve_skills(
            "water", user, [SCOPE], apply_gate=True, query_embedding=Q
        )
        assert sorted(_ids(result)) == ["r-a", "r-b"]
        gate = result.diagnostics.gate
        assert (gate.min_similarity, gate.relative_ratio) == (0.62, 0.9)
        assert gate.threshold == pytest.approx(0.855)

    async def test_the_calibrated_floor_alone_decides_a_lone_weak_match(
        self, sm, user, mgr
    ):
        # A best match at 0.61 is below the configured 0.62 floor. A match
        # at 0.63 passes both the floor and the 0.9-relative threshold.
        await add_skill(sm, user, "cf-weak", name="knit a scarf", cosine=0.61)
        result = await mgr.retrieve_skills(
            "knit", user, [SCOPE], apply_gate=True, query_embedding=Q
        )
        assert result.skills == []
        await add_skill(sm, user, "cf-ok", name="knit a hat", cosine=0.63)
        result = await mgr.retrieve_skills(
            "knit", user, [SCOPE], apply_gate=True, query_embedding=Q
        )
        assert _ids(result) == ["cf-ok"]

    async def test_gate_settings_come_from_the_configuration(
        self, sm, user, mgr, monkeypatch
    ):
        monkeypatch.setattr(constants, "SKILL_RETRIEVAL_MIN_SIMILARITY", 0.5)
        monkeypatch.setattr(constants, "SKILL_RETRIEVAL_RELATIVE_RATIO", 0.5)
        for skill_id, cosine in (("gc-a", 0.95), ("gc-b", 0.55), ("gc-c", 0.45)):
            await add_skill(sm, user, skill_id, name=skill_id, cosine=cosine)
        result = await mgr.retrieve_skills(
            "anything", user, [SCOPE], apply_gate=True, query_embedding=Q
        )
        assert sorted(_ids(result)) == ["gc-a", "gc-b"]

    async def test_a_failed_query_vector_falls_back_to_bm25_without_gate(
        self, sm, user, mgr
    ):
        await add_skill(sm, user, "f-hit", name="restart pods", cosine=0.1)
        await add_skill(sm, user, "f-miss", name="bake bread", cosine=0.99)
        calls = []

        async def failing(text_value):
            calls.append(text_value)
            raise ConnectionError("embedding endpoint down")

        from mirix.services import skill_retrieval

        result = await skill_retrieval.retrieve_skills(
            "restart pods",
            user,
            [SCOPE],
            apply_gate=True,
            session_maker=sm,
            embed_query=failing,
        )
        assert calls == ["restart pods"]
        d = result.diagnostics
        assert d.embedding_status == "failed"
        assert "embedding endpoint down" in d.embedding_error
        assert d.method == "bm25" and d.requested_method == "hybrid"
        assert d.lanes == ["bm25"]
        # No gate (the hit's cosine would be 0.1), BM25 matches only.
        assert d.gate.applied is False and d.gate.skipped_reason == "no query vector"
        assert _ids(result) == ["f-hit"]
        assert result.skills[0].scores.cosine is None
        assert result.skills[0].scores.rrf == pytest.approx(1 / 61)

    async def test_the_manager_retries_the_embedding_then_falls_back(
        self, sm, user, mgr, monkeypatch
    ):
        await add_skill(sm, user, "fr-hit", name="patch servers", cosine=0.9)
        attempts = []

        class Down:
            async def get_text_embedding(self, text_value):
                attempts.append(text_value)
                raise TimeoutError("read timed out")

        async def factory(config, *args, **kwargs):
            return Down()

        monkeypatch.setattr(pmm, "embedding_model", factory)
        result = await mgr.retrieve_skills(
            "patch servers", user, [SCOPE], apply_gate=True, embedding_config=_config()
        )
        assert len(attempts) > 1  # Embedding failures are retried first
        assert result.diagnostics.embedding_status == "failed"
        assert _ids(result) == ["fr-hit"]

    async def test_without_an_embedding_config_the_ranking_is_bm25(self, sm, user, mgr):
        await add_skill(sm, user, "nc-a", name="sort mail", cosine=0.9)
        result = await mgr.retrieve_skills("sort mail", user, [SCOPE], apply_gate=True)
        assert result.diagnostics.embedding_status == "skipped"
        assert result.diagnostics.method == "bm25"
        assert _ids(result) == ["nc-a"]

    async def test_the_bm25_method_never_embeds(self, sm, user, mgr, monkeypatch):
        async def forbidden(*args, **kwargs):
            raise AssertionError("method=bm25 must not embed the query")

        monkeypatch.setattr(pmm, "embedding_model", forbidden)
        await add_skill(sm, user, "bo-a", name="file taxes", cosine=0.9)
        result = await mgr.retrieve_skills(
            "file taxes", user, [SCOPE], method="bm25", embedding_config=_config()
        )
        assert result.diagnostics.embedding_status == "skipped"
        assert _ids(result) == ["bo-a"]

    async def test_the_embedding_method_ranks_by_cosine_alone(self, sm, user, mgr):
        await add_skill(sm, user, "eo-lex", name="paint fence", cosine=0.2)
        await add_skill(sm, user, "eo-vec", name="unrelated", cosine=0.9)
        result = await mgr.retrieve_skills(
            "paint fence", user, [SCOPE], method="embedding", query_embedding=Q
        )
        assert result.diagnostics.lanes == ["cosine"]
        assert _ids(result) == ["eo-vec", "eo-lex"]
        # The BM25 score is still reported, but it did not rank anything.
        lex = _trace(result, "eo-lex")
        assert lex.bm25 > 0 and lex.bm25_rank is None

    async def test_ties_are_broken_by_id(self, sm, user, mgr):
        # Same document length, same vector: every score ties.
        for skill_id in ("tie-c", "tie-a", "tie-d", "tie-b"):
            await add_skill(
                sm, user, skill_id, name=f"clean kitchen x{skill_id[-1]}", cosine=0.7
            )
        first = await mgr.retrieve_skills(
            "clean kitchen", user, [SCOPE], query_embedding=Q
        )
        again = await mgr.retrieve_skills(
            "clean kitchen", user, [SCOPE], query_embedding=Q
        )
        assert _ids(first) == ["tie-a", "tie-b", "tie-c", "tie-d"] == _ids(again)

    async def test_scopes_users_and_organizations_are_isolated(
        self, sm, org, user, mgr
    ):
        await add_skill(sm, user, "sc-mine", name="walk dog", cosine=0.9)
        await add_skill(
            sm, user, "sc-other-scope", name="walk dog o", scope="other", cosine=0.9
        )
        await add_skill(
            sm, user, "sc-no-scope", name="walk dog n", scope=None, cosine=0.9
        )
        other_user = await _new_user(sm, org)
        await add_skill(sm, other_user, "sc-other-user", name="walk dog", cosine=0.9)
        other_org_id = f"sr-org-{uuid.uuid4().hex[:8]}"
        async with sm() as session:
            session.add(OrganizationORM(id=other_org_id, name=other_org_id))
            await session.commit()
        stranger = await _new_user(sm, other_org_id)
        await add_skill(sm, stranger, "sc-other-org", name="walk dog", cosine=0.9)

        result = await mgr.retrieve_skills("walk dog", user, [SCOPE], query_embedding=Q)
        assert _ids(result) == ["sc-mine"]
        assert result.diagnostics.candidate_count == 1  # N counts the scope only
        both = await mgr.retrieve_skills(
            "walk dog", user, [SCOPE, "other"], query_embedding=Q
        )
        assert sorted(_ids(both)) == ["sc-mine", "sc-other-scope"]
        assert (
            await mgr.retrieve_skills("walk dog", user, [], query_embedding=Q)
        ).skills == []
        unrestricted = await mgr.retrieve_skills(
            "walk dog", user, None, query_embedding=Q
        )
        assert sorted(_ids(unrestricted)) == [
            "sc-mine",
            "sc-no-scope",
            "sc-other-scope",
        ]
        # The same id space never leaks another tenant's skill, whoever asks.
        mine = await mgr.retrieve_skills(
            "walk dog", other_user, [SCOPE], query_embedding=Q
        )
        assert _ids(mine) == ["sc-other-user"]

    async def test_filter_tags_restrict_the_candidates(self, sm, user, mgr):
        await add_skill(
            sm, user, "ft-a", name="plan trip", tags={"project": "a"}, cosine=0.9
        )
        await add_skill(
            sm, user, "ft-b", name="plan trip b", tags={"project": "b"}, cosine=0.9
        )
        result = await mgr.retrieve_skills(
            "plan trip", user, [SCOPE], query_embedding=Q, filter_tags={"project": "b"}
        )
        assert _ids(result) == ["ft-b"]
        assert result.diagnostics.candidate_count == 1

    async def test_deleted_skills_are_not_candidates(self, sm, user, mgr):
        await add_skill(sm, user, "del-live", name="feed cat", cosine=0.7)
        await add_skill(
            sm, user, "del-gone", name="feed cat now", cosine=0.99, deleted=True
        )
        result = await mgr.retrieve_skills("feed cat", user, [SCOPE], query_embedding=Q)
        assert _ids(result) == ["del-live"]
        assert result.diagnostics.candidate_count == 1
        assert result.diagnostics.bm25.n_docs == 1
        assert all(c.id != "del-gone" for c in result.diagnostics.candidates)

    async def test_missing_vectors_are_counted_and_never_pass_the_gate(
        self, sm, user, mgr
    ):
        await add_skill(sm, user, "mv-none", name="fix bike chain")
        await add_skill(sm, user, "mv-has", name="fix bike", cosine=0.9)
        gated = await mgr.retrieve_skills(
            "fix bike chain", user, [SCOPE], apply_gate=True, query_embedding=Q
        )
        assert gated.diagnostics.missing_vector_count == 1
        assert _ids(gated) == ["mv-has"]
        none = _trace(gated, "mv-none")
        assert none.cosine is None and none.cosine_rank is None
        assert none.bm25_rank == 1 and none.gate_passed is False
        ungated = await mgr.retrieve_skills(
            "fix bike chain", user, [SCOPE], query_embedding=Q
        )
        assert set(_ids(ungated)) == {"mv-none", "mv-has"}

    async def test_the_default_limit_is_ten_and_zero_returns_nothing(
        self, sm, user, mgr
    ):
        for i in range(12):
            await add_skill(
                sm, user, f"lim-{i:02d}", name=f"water plants n{i:02d}", cosine=0.9
            )
        result = await mgr.retrieve_skills(
            "water plants", user, [SCOPE], query_embedding=Q
        )
        assert len(result.skills) == 10 == constants.SKILL_RETRIEVAL_DEFAULT_LIMIT
        assert result.diagnostics.limit == 10
        assert len(result.diagnostics.candidates) == 12  # every candidate is traced
        zero = await mgr.retrieve_skills(
            "water plants", user, [SCOPE], limit=0, query_embedding=Q
        )
        assert zero.skills == [] and zero.diagnostics.returned_ids == []
        with pytest.raises(ValueError):
            await mgr.retrieve_skills("water plants", user, [SCOPE], limit=-1)
        with pytest.raises(ValueError):
            await mgr.retrieve_skills("water plants", user, [SCOPE], method="fuzzy")

    async def test_an_empty_query_ranks_nothing_and_embeds_nothing(self, sm, user, mgr):
        await add_skill(sm, user, "eq-a", name="anything", cosine=0.9)

        async def forbidden(_text):
            raise AssertionError("an empty query must not be embedded")

        from mirix.services import skill_retrieval

        result = await skill_retrieval.retrieve_skills(
            "   ", user, [SCOPE], session_maker=sm, embed_query=forbidden
        )
        assert result.skills == []
        assert result.diagnostics.embedding_status == "skipped"

    async def test_the_query_is_embedded_before_any_session_opens(self, sm, user):
        await add_skill(sm, user, "order-a", name="tune guitar", cosine=0.9)
        events = []

        @asynccontextmanager
        async def spy():
            events.append("session")
            async with sm() as session:
                yield session

        async def embed(text_value):
            events.append("embed")
            return Q

        from mirix.services import skill_retrieval

        await skill_retrieval.retrieve_skills(
            "tune guitar", user, [SCOPE], session_maker=spy, embed_query=embed
        )
        # The corpus in one session, then (after scoring) the returned rows.
        assert events == ["embed", "session", "session"]

    async def test_scoring_holds_no_session(self, sm, user, monkeypatch):
        """BM25 and the ranking run in a worker thread with no pooled
        connection held; the returned skills are read afterwards."""
        await add_skill(sm, user, "hold-a", name="water plants", cosine=0.9)
        open_sessions = []
        seen = []

        @asynccontextmanager
        async def spy():
            open_sessions.append(1)
            try:
                async with sm() as session:
                    yield session
            finally:
                open_sessions.pop()

        from mirix.services import skill_retrieval

        real_rank = skill_retrieval.rank_candidates

        def rank(*args, **kwargs):
            seen.append(len(open_sessions))
            return real_rank(*args, **kwargs)

        monkeypatch.setattr(skill_retrieval, "rank_candidates", rank)
        result = await skill_retrieval.retrieve_skills(
            "water plants", user, [SCOPE], session_maker=spy, query_embedding=Q
        )
        assert seen == [0]
        assert [r.skill.name for r in result.skills] == ["water plants"]

    async def test_a_skill_deleted_while_ranking_is_not_returned(
        self, sm, user, monkeypatch
    ):
        from sqlalchemy import update

        from mirix.services import skill_retrieval

        await add_skill(sm, user, "gone-a", name="feed fish", cosine=0.9)
        await add_skill(sm, user, "gone-b", name="feed fish daily", cosine=0.8)
        sessions = []

        @asynccontextmanager
        async def deleting_before_the_reread():
            sessions.append(1)
            async with sm() as session:
                if len(sessions) == 2:  # ranked; the rows are read next
                    await session.execute(
                        update(ProceduralORM)
                        .where(ProceduralORM.id == "gone-a")
                        .values(is_deleted=True)
                    )
                    await session.commit()
                yield session

        result = await skill_retrieval.retrieve_skills(
            "feed fish",
            user,
            [SCOPE],
            session_maker=deleting_before_the_reread,
            query_embedding=Q,
        )
        assert [r.skill.id for r in result.skills] == ["gone-b"]
        assert result.diagnostics.returned_ids == ["gone-b"]

    async def test_a_raw_conversation_query_scores_its_first_terms_only(
        self, sm, user, mgr, monkeypatch
    ):
        """After a failed extraction the query is the raw conversation
        (thousands of terms). Only its first MIRIX_SKILL_BM25_MAX_QUERY_TERMS
        distinct terms are scored; the diagnostics count the others."""
        monkeypatch.setattr(sr.constants, "SKILL_BM25_MAX_QUERY_TERMS", 4)
        await add_skill(sm, user, "cap-a", name="prune roses", cosine=0.5)
        await add_skill(sm, user, "cap-b", name="paint fence", cosine=0.5)
        # "fence" is the 6th distinct term: dropped, so cap-b gets no BM25.
        query = "please prune roses today fence fence paint"
        result = await mgr.retrieve_skills(query, user, [SCOPE], method="bm25")
        bm25 = result.diagnostics.bm25
        assert bm25.query_terms == ["please", "prune", "roses", "today"]
        assert bm25.query_terms_dropped == 2 and bm25.max_query_terms == 4
        assert _ids(result) == ["cap-a"]
        payload = result.diagnostics.to_dict()["bm25"]
        assert payload["query_terms_dropped"] == 2

    async def test_the_logged_trace_is_bounded(self, sm, user, mgr):
        """A retrieval log keeps every returned skill and the next
        candidates in fused order, with counts — not every candidate."""
        for i in range(8):
            await add_skill(
                sm, user, f"bound-{i}", name=f"sort mail {i}", cosine=0.9 - i / 100
            )
        await add_skill(sm, user, "bound-far", name="unrelated", cosine=None)
        result = await mgr.retrieve_skills(
            "sort mail", user, [SCOPE], limit=2, query_embedding=Q
        )
        full = result.diagnostics.to_dict()
        assert "candidates_logged" not in full
        bounded = result.diagnostics.to_dict(max_candidates=3)
        ids = [c["id"] for c in bounded["candidates"]]
        returned = [c["id"] for c in full["candidates"] if c["returned"]]
        ranked = [c["id"] for c in full["candidates"] if c["rrf_rank"] is not None]
        assert returned and set(returned) <= set(ids)
        others = [i for i in ids if i not in returned]
        assert others == [i for i in ranked if i not in returned][:3]
        assert bounded["candidates_logged"] == len(ids) == len(returned) + 3
        assert bounded["candidates_omitted"] == len(full["candidates"]) - len(ids)
        assert "bound-far" not in ids  # unranked candidates are only counted
        assert bounded["returned_ids"] == full["returned_ids"]

    async def test_unusable_query_vectors_are_failures(self, sm, user, mgr):
        await add_skill(sm, user, "uv-a", name="mow lawn", cosine=0.9)
        for bad in (
            [0.0, 0.0],
            [float("nan"), 1.0],
            [1.0] * (MAX_EMBEDDING_DIM + 1),
            [],
        ):
            result = await mgr.retrieve_skills(
                "mow lawn", user, [SCOPE], apply_gate=True, query_embedding=bad
            )
            assert result.diagnostics.embedding_status == "failed", bad
            assert _ids(result) == ["uv-a"]

    async def test_each_returned_skill_carries_the_version_it_was_served_at(
        self, sm, user, mgr
    ):
        """The retrieval records, for each skill it returns, the
        newest version of its history when the row was read; the log keeps it
        (``returned_versions``)."""
        from mirix.orm.skill_version import SkillVersion as SkillVersionORM

        await add_skill(sm, user, "sv-a", name="weigh parcels", cosine=0.9)
        await add_skill(sm, user, "sv-b", name="weigh letters", cosine=0.88)
        async with sm() as session:
            for number in (1, 2, 3):
                session.add(
                    SkillVersionORM(
                        skill_id="sv-a",
                        organization_id=user.organization_id,
                        user_id=user.id,
                        version_no=number,
                        change_type="create" if number == 1 else "edit",
                        name_sha256="0" * 64,
                        entry_type_sha256="0" * 64,
                        description_sha256="0" * 64,
                        instructions_sha256="0" * 64,
                        triggers_sha256="0" * 64,
                        examples_sha256="0" * 64,
                        filter_tags_sha256="0" * 64,
                    )
                )
            await session.commit()
        result = await mgr.retrieve_skills(
            "weigh parcels letters", user, [SCOPE], apply_gate=False, query_embedding=Q
        )
        served = {r.skill.id: r.version_no for r in result.skills}
        assert served == {"sv-a": 3, "sv-b": None}
        payload = result.diagnostics.to_dict(max_candidates=0)
        assert payload["returned_versions"] == {"sv-a": 3, "sv-b": None}
        assert list(payload["returned_versions"]) == payload["returned_ids"]
        # A trace built without versions keeps its earlier shape (the
        # distiller then shows those skills' current text).
        from dataclasses import replace

        bare = replace(result.diagnostics, returned_versions={}).to_dict()
        assert "returned_versions" not in bare

    async def test_diagnostics_hold_the_retrieval_log_fields(self, sm, user, mgr):
        await add_skill(
            sm, user, "dg-a", name="order pizza", description="dinner", cosine=0.9
        )
        await add_skill(sm, user, "dg-b", name="cook pasta", cosine=0.3)
        await add_skill(sm, user, "dg-c", name="nothing matches", cosine=None)
        result = await mgr.retrieve_skills(
            "order pizza", user, [SCOPE], apply_gate=True, query_embedding=Q
        )
        payload = result.diagnostics.to_dict()
        assert json.loads(json.dumps(payload)) == payload  # JSON-safe
        assert set(payload) == {
            "requested_method",
            "method",
            "lanes",
            "backend",
            "embedding_status",
            "embedding_error",
            "candidate_count",
            "missing_vector_count",
            "bm25",
            "rrf_k",
            "gate",
            "limit",
            "max_cosine_distance",
            "candidates",
            "returned_ids",
            "returned_versions",
        }
        assert payload["rrf_k"] == 60
        assert payload["returned_ids"] == ["dg-a"]
        # A skill without history is served at no recorded version.
        assert payload["returned_versions"] == {"dg-a": None}
        assert payload["missing_vector_count"] == 1
        assert payload["bm25"]["query_terms"] == ["order", "pizza"]
        assert payload["bm25"]["weights"] == {"A": 1.0, "B": 0.4}
        assert (payload["bm25"]["k1"], payload["bm25"]["b"]) == (1.2, 0.75)
        assert payload["bm25"]["prefiltered"] is False
        assert payload["gate"]["applied"] is True
        # Ranked candidates first (fused order), then the unranked by id.
        ranked = [c for c in payload["candidates"] if c["rrf_rank"] is not None]
        assert [c["rrf_rank"] for c in ranked] == list(range(1, len(ranked) + 1))
        assert payload["candidates"][: len(ranked)] == ranked
        assert set(payload["candidates"][0]) == {
            "id",
            "bm25",
            "bm25_rank",
            "cosine",
            "cosine_rank",
            "rrf",
            "rrf_rank",
            "gate_passed",
            "returned",
        }
        returned = [c["id"] for c in payload["candidates"] if c["returned"]]
        assert returned == payload["returned_ids"]
        # The scores returned with the skill are the traced ones.
        trace = _trace(result, "dg-a")
        assert result.skills[0].scores.to_dict() == {
            "bm25": trace.bm25,
            "cosine": trace.cosine,
            "rrf": trace.rrf,
        }

    async def test_returned_skills_are_full_rows_with_localized_timestamps(
        self, sm, user, mgr
    ):
        await add_skill(
            sm,
            user,
            "full-a",
            name="book flight",
            description="travel",
            triggers=["a trip"],
            instructions="Compare fares.",
            cosine=0.9,
        )
        result = await mgr.retrieve_skills(
            "book flight", user, [SCOPE], query_embedding=Q, timezone_str="Asia/Tokyo"
        )
        skill = result.skills[0].skill
        assert (skill.name, skill.description, skill.instructions) == (
            "book flight",
            "travel",
            "Compare fares.",
        )
        assert skill.triggers == ["a trip"]
        assert skill.created_at.utcoffset().total_seconds() == 9 * 3600

    async def test_reads_never_touch_a_cache(self, sm, user, mgr, monkeypatch):
        import mirix.database.cache_provider as cache_provider
        import mirix.database.redis_client as redis_client

        def forbidden(*args, **kwargs):
            raise AssertionError("procedural retrieval must not touch a cache")

        monkeypatch.setattr(cache_provider, "get_cache_provider", forbidden)
        monkeypatch.setattr(redis_client, "get_redis_client", forbidden)
        await add_skill(sm, user, "cache-a", name="iron shirt", cosine=0.9)
        result = await mgr.retrieve_skills(
            "iron shirt", user, [SCOPE], query_embedding=Q
        )
        assert _ids(result) == ["cache-a"]


def _config():
    return EmbeddingConfig(
        embedding_endpoint_type="openai",
        embedding_endpoint="http://127.0.0.1:9/v1",
        embedding_model="x",
        embedding_dim=2,
    )


def _agent_state(config):
    return AgentState(
        id="agent-sr",
        name="sr-agent",
        system="s",
        agent_type=AgentType.procedural_memory_agent,
        llm_config=LLMConfig(
            model="x", model_endpoint_type="openai", context_window=8000
        ),
        embedding_config=config,
        tools=[],
    )


@asyncio_module
class TestManagerEntryPoints:
    async def test_the_query_goes_through_the_embedding_factory(
        self, sm, user, mgr, fake_llm_server, monkeypatch
    ):
        """End to end on the offline fake server: the agent's embedding config
        embeds the raw query once; the gate then applies to real cosines."""
        from mirix.settings import model_settings

        monkeypatch.setattr(model_settings, "openrouter_api_key", None)

        def embed(text_value, dim):
            # "alpha" texts point one way, everything else another.
            base = [1.0, 0.0] if "alpha" in text_value else [0.0, 1.0]
            return base + [0.0] * (dim - 2)

        fake_llm_server.embedding_fn = embed
        await add_skill(sm, user, "e2e-alpha", name="alpha task", cosine=0.99)
        await add_skill(sm, user, "e2e-beta", name="alpha beta", cosine=0.0)
        config = EmbeddingConfig(
            embedding_endpoint_type="openrouter",
            embedding_endpoint=fake_llm_server.base_url,
            embedding_model="google/gemini-embedding-001",
            embedding_dim=8,
            api_key="sk-or-fake",
        )
        result = await mgr.retrieve_skills(
            "alpha task please",
            user,
            [SCOPE],
            apply_gate=True,
            agent_state=_agent_state(config),
        )
        assert [r.json["input"] for r in fake_llm_server.embedding_requests()] == [
            "alpha task please"
        ]
        assert result.diagnostics.embedding_status == "ok"
        assert _ids(result) == ["e2e-alpha"]  # e2e-beta: cosine 0 < 0.6

    async def test_list_procedures_hybrid_is_the_ungated_ranking(
        self, sm, user, mgr, monkeypatch
    ):
        await add_skill(sm, user, "lp-a", name="wash car", cosine=0.2)
        await add_skill(sm, user, "lp-b", name="wash dishes", cosine=0.1)
        calls = []
        real = mgr.retrieve_skills

        async def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return await real(*args, **kwargs)

        monkeypatch.setattr(mgr, "retrieve_skills", spy)
        items = await mgr.list_procedures(
            agent_state=_agent_state(_config()),
            user=user,
            query="wash car",
            embedded_text=Q,
            search_field="description",
            search_method="hybrid",
            scopes=[SCOPE],
            limit=5,
        )
        assert [i.id for i in items] == ["lp-a", "lp-b"]  # both cosines < 0.6
        ((args, kwargs),) = calls
        assert args[:4] == ("wash car", user, [SCOPE], 5)
        assert kwargs["apply_gate"] is False
        assert kwargs["query_embedding"][:2] == Q

    async def test_list_procedures_hybrid_on_instructions_is_keyword_search(
        self, sm, user, mgr, monkeypatch
    ):
        async def forbidden(*args, **kwargs):
            raise AssertionError("instructions carry no vector")

        monkeypatch.setattr(mgr, "retrieve_skills", forbidden)
        monkeypatch.setattr(pmm, "embedding_model", forbidden)
        await add_skill(
            sm, user, "li-a", name="x", instructions="run kubectl rollout", cosine=0.9
        )
        items = await mgr.list_procedures(
            agent_state=_agent_state(_config()),
            user=user,
            query="kubectl",
            search_field="instructions",
            search_method="hybrid",
            scopes=[SCOPE],
        )
        assert items[0].id == "li-a"


def test_every_method_and_field_is_routed():
    route = pmm.uses_skill_ranking
    assert route("hybrid", "") and route("hybrid", "description")
    assert route("hybrid", "name") and route("hybrid", "null")
    assert route("bm25", "description") and route("embedding", "")
    assert not route("bm25", "instructions")
    assert not route("hybrid", "instructions")
    assert not route("string_match", "description")
    assert not route("fuzzy_match", "")
    assert not route("bm25", "entry_type")


# --------------------------------------------------------------------------- #
# Where the gate's calibration holds                                           #
# --------------------------------------------------------------------------- #


def _embedding_config(model):
    return EmbeddingConfig(
        embedding_endpoint_type="openai",
        embedding_endpoint="http://127.0.0.1:9/v1",
        embedding_model=model,
        embedding_dim=2,
    )


@pytest.fixture
def gate_warnings(monkeypatch):
    """The gate's warnings (the MIRIX logger does not propagate to caplog),
    with a fresh warn-once memory and no environment override."""
    warnings = []
    monkeypatch.setattr(sr, "_warned_uncalibrated_models", set())
    monkeypatch.setattr(constants, "SKILL_RETRIEVAL_GATE_FROM_ENV", False)
    monkeypatch.setattr(
        sr.logger, "warning", lambda msg, *args: warnings.append(msg % args)
    )
    return warnings


class TestGateCalibration:
    """The similarity gate's defaults were calibrated on gemini-embedding-001:
    they are applied to that model as calibrated, to any other as
    ``uncalibrated`` (with one warning per model), and an operator's
    environment setting wins for every model (``env``)."""

    @pytest.mark.parametrize(
        "model", ["gemini-embedding-001", "google/gemini-embedding-001"]
    )
    def test_the_calibrated_model_as_given_or_without_its_provider(
        self, model, gate_warnings
    ):
        gate = sr.skill_retrieval_gate(model)
        assert (gate.min_similarity, gate.relative_ratio, gate.source) == (
            0.62,
            0.90,
            "gemini-embedding-001",
        )
        assert gate_warnings == []

    def test_another_model_is_uncalibrated_and_warned_about_once(self, gate_warnings):
        first = sr.skill_retrieval_gate("text-embedding-3-small")
        second = sr.skill_retrieval_gate("text-embedding-3-small")
        assert first == second
        assert (first.min_similarity, first.relative_ratio, first.source) == (
            constants.SKILL_RETRIEVAL_MIN_SIMILARITY_DEFAULT,
            constants.SKILL_RETRIEVAL_RELATIVE_RATIO_DEFAULT,
            "uncalibrated",
        )
        [warning] = [w for w in gate_warnings if "text-embedding-3-small" in w]
        assert "gemini-embedding-001" in warning
        assert "MIRIX_SKILL_RETRIEVAL_MIN_SIMILARITY" in warning
        # Another uncalibrated model is warned about on its own.
        sr.skill_retrieval_gate("nomic-embed-text")
        assert len(gate_warnings) == 2

    @pytest.mark.parametrize(
        "model", ["gemini-embedding-001", "text-embedding-3-small", None]
    )
    def test_an_environment_setting_wins_for_every_model(
        self, model, gate_warnings, monkeypatch
    ):
        monkeypatch.setattr(constants, "SKILL_RETRIEVAL_GATE_FROM_ENV", True)
        monkeypatch.setattr(constants, "SKILL_RETRIEVAL_MIN_SIMILARITY", 0.5)
        monkeypatch.setattr(constants, "SKILL_RETRIEVAL_RELATIVE_RATIO", 0.8)
        gate = sr.skill_retrieval_gate(model)
        assert (gate.min_similarity, gate.relative_ratio, gate.source) == (
            0.5,
            0.8,
            "env",
        )
        assert gate_warnings == []

    @pytest.mark.parametrize(
        "raw_min, raw_ratio, expected",
        [
            (None, None, False),
            ("", "  ", False),
            ("0.55", None, True),
            (None, "0.8", True),
            ("0.62", None, True),  # the default value, set on purpose
            ("abc", None, False),  # invalid: the default is used (with a warning)
            ("1.5", "7", False),
        ],
    )
    def test_what_counts_as_an_environment_setting(self, raw_min, raw_ratio, expected):
        assert constants._skill_retrieval_gate_from_env(raw_min, raw_ratio) is expected

    def test_the_trace_names_the_calibration(self):
        trace = sr.GateTrace(
            applied=True,
            min_similarity=0.62,
            relative_ratio=0.9,
            calibration="uncalibrated",
        )
        assert trace.to_dict()["calibration"] == "uncalibrated"
        assert sr.GateTrace(False, 0.62, 0.9).to_dict()["calibration"] is None


@asyncio_module
class TestGateSettingsOfARetrieval:
    async def test_explicit_gate_settings_override_the_defaults(self, sm, user):
        await add_skill(sm, user, "gs-weak", name="fold towels", cosine=0.55)
        default = await sr.retrieve_skills(
            "fold towels",
            user,
            [SCOPE],
            apply_gate=True,
            session_maker=sm,
            query_embedding=Q,
        )
        assert default.skills == []
        assert default.diagnostics.gate.calibration is None
        lowered = await sr.retrieve_skills(
            "fold towels",
            user,
            [SCOPE],
            apply_gate=True,
            session_maker=sm,
            query_embedding=Q,
            min_similarity=0.5,
            relative_ratio=0.8,
            gate_calibration="env",
        )
        assert _ids(lowered) == ["gs-weak"]
        gate = lowered.diagnostics.gate
        assert (gate.min_similarity, gate.relative_ratio) == (0.5, 0.8)
        assert gate.calibration == "env"
        assert lowered.diagnostics.to_dict()["gate"]["calibration"] == "env"

    @pytest.mark.parametrize(
        "model, source",
        [
            ("google/gemini-embedding-001", "gemini-embedding-001"),
            ("text-embedding-3-small", "uncalibrated"),
        ],
    )
    async def test_the_manager_resolves_the_gate_of_its_embedding_model(
        self, sm, user, mgr, model, source, gate_warnings
    ):
        await add_skill(sm, user, f"gm-{source}", name="stack chairs", cosine=0.7)
        result = await mgr.retrieve_skills(
            "stack chairs",
            user,
            [SCOPE],
            apply_gate=True,
            embedding_config=_embedding_config(model),
            query_embedding=Q,
        )
        assert _ids(result) == [f"gm-{source}"]
        gate = result.diagnostics.gate
        assert gate.applied and gate.calibration == source
        assert (gate.min_similarity, gate.relative_ratio) == (0.62, 0.9)
        assert len(gate_warnings) == (source == "uncalibrated")

    async def test_an_ungated_search_resolves_nothing(
        self, sm, user, mgr, gate_warnings
    ):
        await add_skill(sm, user, "gu-a", name="sweep floor", cosine=0.3)
        result = await mgr.retrieve_skills(
            "sweep floor",
            user,
            [SCOPE],
            embedding_config=_embedding_config("text-embedding-3-small"),
            query_embedding=Q,
        )
        assert _ids(result) == ["gu-a"]
        assert result.diagnostics.gate.calibration is None
        assert gate_warnings == []  # /memory/search never gates
