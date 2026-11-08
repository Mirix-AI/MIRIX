"""
v7 graph manager - minimal semantic+episodic linkage graph.

v7 keeps the useful part of v6 (Neo4j as an index into PG flat memory), but
tightens the ontology:

- Details stay in PostgreSQL. Graph nodes store ids, types, canonical names,
  timestamps, and a short title/preview only for debugging.
- Anchors must be specific enough to be useful. Generic noun phrases are
  discarded instead of becoming graph nodes.
- Semantic and episodic memory refs live in one graph and share the same
  anchors. Semantic refs are linked back to episodic refs from the same
  source chunk when provenance is available.
- No entity-entity co-occurrence edges. Every edge must be a retrieval path:
  anchor -> memory ref, semantic ref -> supporting episode, or temporal next.
"""

from __future__ import annotations

import re
import asyncio
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from mirix.log import get_logger
from mirix.schemas.agent import AgentState
from mirix.services._graph_common import (
    embed_batch,
    gen_id,
    iso,
    llm_model_from_agent,
    normalize_name,
)
from mirix.services.lightrag_extractor import ExtractedEntity
from mirix.settings import settings

logger = get_logger(__name__)


SourceKind = Literal["episodic", "semantic"]

MAX_ANCHORS_PER_EPISODE = 8
MAX_ANCHORS_PER_SEMANTIC = 10
# v7.12 frames routinely name more participants than the v7.10 budget allowed: a
# 7-arg frame alone exhausted MAX_ANCHORS_PER_EPISODE, and every arg that failed to
# become an anchor silently dropped an arm off the frame. Measured on a 12-memory
# probe: 31% of frames landed with fewer than 2 live arms. The cap still exists (an
# unbounded anchor space is what it was protecting against) — it is just sized for
# arguments of an assertion rather than for a bag of keywords.
MAX_FRAME_ANCHORS_PER_EPISODE = 16
MAX_FRAME_ANCHORS_PER_SEMANTIC = 18
PREVIEW_CHARS = 160

_GENERIC_NAMES = {
    "advice", "approach", "benefits", "best practices", "challenge",
    "challenges", "concept", "considerations", "details", "example",
    "examples", "experience", "feedback", "flexibility", "goal", "goals",
    "guidance", "habit", "help", "idea", "ideas", "information",
    "insights", "issue", "issues", "method", "methods", "option",
    "options", "plan", "plans", "practice", "practices", "preference",
    "recommendation", "recommendations", "routine", "schedule", "skills",
    "social media", "steps", "strategy", "support", "task", "tasks",
    "thing", "things", "tips", "topic", "topics", "update", "updates",
    "way", "ways",
}

_GENERIC_SUFFIXES = (
    " advice", " approach", " benefits", " considerations", " details",
    " examples", " experience", " feedback", " guidance", " ideas",
    " information", " method", " methods", " options", " plan", " plans",
    " recommendations", " routine", " schedule", " strategy", " tips",
)

_SPECIFIC_HINT_RE = re.compile(r"(\d|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|['\u2019])")

# Anchor merge key (A): collapses case / punctuation / word-order and
# singular/plural so "Mobile App"/"Mobile Apps", "Flight"/"Flights",
# "Price"/"Prices" merge onto ONE V7Anchor instead of fragmenting into
# near-duplicate nodes. Deliberately conservative \u2014 numbers are preserved
# (so "10 Gallons" != "20 Gallons", and distinct dates stay distinct) and only
# plural endings are normalized (no verb stemming, so "Training" != "Trains").
_ANCHOR_STOPWORDS = {"the", "a", "an", "of", "and", "or", "to", "for", "in", "on", "with"}


def _singularize(token: str) -> str:
    if token.isdigit() or len(token) < 4:
        return token
    if token.endswith(("ss", "us", "is")):
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if re.search(r"(ses|xes|zes|ches|shes)$", token):
        return token[:-2]
    if token.endswith("s"):
        return token[:-1]
    return token


def anchor_canonical_key(name: str) -> str:
    """Canonical merge key for a V7 anchor name (see note above)."""
    cleaned = re.sub(r"[^a-z0-9 ]", " ", (name or "").lower())
    tokens = {_singularize(t) for t in cleaned.split() if t and t not in _ANCHOR_STOPWORDS}
    return " ".join(sorted(tokens))


# Canonical keys of dialogue-role names that must never become anchors — checked in
# _select_anchors so it covers EVERY extractor path (LightRAG has no _NOISE filter of
# its own). anchor_canonical_key folds case, articles and plurals, so "user" here
# blocks "User", "Users", "The User", "the users", ... in one entry.
# NB: entries must be in the key's own form — canonical keys are SORTED token sets,
# so the combined role is "assistant user" (never "user assistant"), and possessives
# leave a stray "s" token ("User's" -> "s user") because the bare s survives
# _singularize's length guard.
_ROLE_NOISE_KEYS = {"user", "assistant", "assistant user", "s user", "assistant s"}

_PRED_COPULA = re.compile(r"^(is|are|was|were|be|been|being)\s+", re.I)



def frame_identity(user_id: str, predicate: str, role_keys: list) -> str:
    """Deterministic id for an n-ary frame.

    Keyed on the SET of (role, anchor) pairs, sorted, so the same frame extracted
    from another memory — or with its args listed in another order — lands on the
    same node and merely adds a citation. Binary facts hash the same way with two
    pairs, so v7.10 and v7.12 ids differ (by design: the graphs are rebuilt, never
    migrated in place).
    """
    payload = "|".join([user_id, predicate] + sorted(role_keys))
    return "v7frm-" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def frame_identity_for_source(
    user_id: str,
    predicate: str,
    role_keys: list[str],
    *,
    literals: dict[str, Any] | None,
    source_kind: SourceKind,
    memory_id: str,
) -> str:
    """Return a merge-safe frame id for semantic claims and episodic events.

    Semantic rows may cite the same durable claim, so their identity remains the
    predicate plus role-typed anchors.  An episodic frame additionally carries
    an occurrence key.  An explicit event time joins repeat mentions of
    the same event; when no event time was extracted, the immutable memory id
    keeps separate occurrences from collapsing merely because they were ingested
    in the same chunk.
    """

    identity_parts = list(role_keys)
    if source_kind == "episodic":
        literal_map = literals or {}
        event_key = next(
            (
                str(literal_map[key]).strip()
                for key in ("event_time", "event_date", "occurred_at", "date", "time")
                if key in literal_map and str(literal_map[key]).strip()
            ),
            None,
        )
        identity_parts.append(f"event:{event_key or f'memory:{memory_id}'}")
    return frame_identity(user_id, predicate, identity_parts)


def canon_predicate(pred: str) -> str:
    """Canonical predicate so surface variants are ONE relation, not several:
    ``includes``->``include``, ``is located in``->``located in``, ``offers``->``offer``.
    Applied at write time so the graph never accumulates the variants in the first
    place (a corpus-wide cleanup previously had to fold 99 of them)."""
    if not pred:
        return "related to"
    p = _PRED_COPULA.sub("", pred.lower().strip())
    tokens = p.split()
    if tokens:
        head = tokens[0]
        # singularize the verb, but keep has/is/ss-words intact
        if len(head) > 4 and head.endswith("s") and not head.endswith(("ss", "us", "is", "as")):
            tokens[0] = head[:-1]
    return " ".join(tokens) or "related to"


def fact_identity(user_id: str, subj_key: str, predicate: str, obj_key: str) -> str:
    """Deterministic id for a fact, so the SAME (subject, predicate, object) asserted
    in N memories MERGEs to ONE V7Fact cited N times — instead of N duplicate nodes.
    A random id here was what generated the duplicate-triple redundancy."""
    raw = f"{user_id}|{subj_key}|{predicate}|{obj_key}"
    return "v7fact-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class V7AnchorCandidate:
    name: str
    name_lower: str
    anchor_type: str
    score: float


class V7GraphManager:
    """Stateless. Construct one per graph write."""

    async def process_memory(
        self,
        *,
        source_kind: SourceKind,
        source_id: str,
        text: str,
        agent_state: AgentState,
        organization_id: str,
        user_id: str,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        occurred_at: Optional[object] = None,
        source_meta: Optional[dict[str, Any]] = None,
        entities: Optional[list[ExtractedEntity]] = None,
        role: Optional[str] = None,
    ) -> dict[str, Any]:
        # startswith, not an exact tuple: the old tuple silently skipped ingest for any
        # version it didn't list (it already omitted v7.5, and every new v7.x had to
        # remember to add itself) while the retrieval side accepts startswith("v7") —
        # a version outside the tuple would retrieve against a graph nothing ingests to.
        if not settings.enable_graph_memory:
            return {"skipped": "disabled"}

        from mirix.database.neo4j_client import get_neo4j_driver

        driver = get_neo4j_driver()
        if driver is None:
            return {"skipped": "no_driver"}
        if not text or not text.strip():
            return {"anchors": 0}

        # Only name/entity_type are consumed downstream, so a caller that already
        # knows the entities can pass them and skip the extraction round-trip.
        # (The v7.3 proposition and v7.4 GLiNER extraction branches are archived
        # under archive/legacy_graph/ — both were superseded by direction D.)
        relations: list[tuple[str, str, str]] = []
        frames: list = []
        if entities is None:
            # v7.12: n-ary frames. One predicate, any number of typed args, with
            # dates/amounts kept as literals on the frame instead of polluting the
            # anchor space. The binary projection still feeds the anchor<->anchor
            # relation edges, so retrieval keeps everything v7.10 had.
            from mirix.services.frame_extractor import extract_frames
            fres = await extract_frames(text, model=llm_model_from_agent(agent_state))
            entities = fres.entities
            relations = fres.as_relations()
            frames = fres.frames
            from mirix.services.entity_resolver import canonicalize_entities
            rename = await canonicalize_entities(
                entities, driver=driver, user_id=user_id, agent_state=agent_state)
            if rename:
                relations = [(rename.get(a, a), r, rename.get(b, b)) for a, r, b in relations]
                for fr in frames:
                    fr.args = [(role, rename.get(nm, nm)) for role, nm in fr.args]
        candidates = self._select_anchors(
            entities,
            max_anchors=(
                MAX_FRAME_ANCHORS_PER_EPISODE if source_kind == "episodic"
                else MAX_FRAME_ANCHORS_PER_SEMANTIC
            ),
        )

        memory_ref_id = f"{source_kind}:{source_id}"
        source_key = self._source_key(source_meta)
        timestamp = self._to_iso(occurred_at) or (source_meta or {}).get("occurred_at")


        if candidates:
            await self._upsert_anchors_and_edges(
                driver,
                anchors=candidates,
                source_kind=source_kind,
                memory_ref_id=memory_ref_id,
                memory_id=source_id,
                agent_state=agent_state,
                organization_id=organization_id,
                user_id=user_id,
            )


        # v7.10 (hypergraph): reify each triple as a V7Fact HYPEREDGE node linking its
        # subject entity, object entity, the source memory, plus role (who) + time (when)
        # as node properties — a multi-dimensional (entity × person/role × time) index
        # over facts, instead of the binary anchor→anchor edge alone.
        await self._upsert_frames(
            driver, frames=frames, memory_id=source_id,
            source_kind=source_kind,
            role=role, timestamp=timestamp,
            mentioned_at=(source_meta or {}).get("mentioned_at"),
            user_id=user_id)


        return {
            "anchors": len(candidates),
            "memory_ref": memory_ref_id,
            "source_key": source_key,
        }

    # ------------------------------------------------------------------ gate

    def _select_anchors(self, entities: list[ExtractedEntity], *, max_anchors: int) -> list[V7AnchorCandidate]:
        by_name: dict[str, V7AnchorCandidate] = {}
        for entity in entities:
            name = self._clean_name(entity.name)
            nl = anchor_canonical_key(name)
            if not name or not nl:
                continue
            # Dialogue roles must never anchor, regardless of which extractor produced
            # them. The per-extractor _NOISE sets are surface-form blocklists and keep
            # leaking variants ("Users" reached degree 1016 through triple extraction;
            # LightRAG has no filter at all, so plain v7/v8 could mint the same hub).
            # Gating on the canonical key here — the funnel every path goes through —
            # is case/article/plural-insensitive by construction ("The Users" -> "user").
            if nl in _ROLE_NOISE_KEYS:
                continue
            score = self._specificity_score(name, entity.entity_type or "Other")
            if score <= 0:
                continue
            candidate = V7AnchorCandidate(
                name=name,
                name_lower=nl,
                anchor_type=entity.entity_type or "Other",
                score=score,
            )
            existing = by_name.get(nl)
            if existing is None or candidate.score > existing.score:
                by_name[nl] = candidate

        return sorted(by_name.values(), key=lambda c: (-c.score, c.name_lower))[:max_anchors]

    def _specificity_score(self, name: str, entity_type: str) -> float:
        nl = normalize_name(name)
        if not nl or nl in _GENERIC_NAMES:
            return 0.0
        if any(nl.endswith(suffix) for suffix in _GENERIC_SUFFIXES):
            return 0.0
        if len(nl) < 3:
            return 0.0

        score = 0.0
        type_norm = entity_type.strip().lower()
        if type_norm in {"person", "location", "organization", "event"}:
            score += 4.0
        elif type_norm in {"content", "object"}:
            score += 3.0
        elif type_norm in {"concept", "method"}:
            score += 1.0
        else:
            score += 0.5

        words = nl.split()
        if len(words) >= 2:
            score += 1.5
        if len(words) >= 3:
            score += 0.5
        if any(ch.isdigit() for ch in name):
            score += 2.0
        if _SPECIFIC_HINT_RE.search(name):
            score += 1.0
        if name[:1].isupper():
            score += 0.5
        if len(words) == 1 and name[:1].isupper() and len(nl) >= 4:
            score += 2.5
        if len(words) == 1 and type_norm in {"concept", "method", "other"} and score < 3.0:
            return 0.0
        return score

    @staticmethod
    def _clean_name(name: str) -> str:
        return " ".join((name or "").strip().strip("\"'`").split())

    # ------------------------------------------------------------- neo4j write

    async def _upsert_anchors_and_edges(
        self,
        driver,
        *,
        anchors: list[V7AnchorCandidate],
        source_kind: SourceKind,
        memory_ref_id: str,
        memory_id: str,
        agent_state: AgentState,
        organization_id: str,
        user_id: str,
    ) -> None:
        existing = await self._fetch_existing_anchors(driver, user_id, [a.name_lower for a in anchors])
        new_anchors = [a for a in anchors if a.name_lower not in existing]
        embeddings = await embed_batch([a.name for a in new_anchors], agent_state) if new_anchors else []
        emb_by_lower = {a.name_lower: emb for a, emb in zip(new_anchors, embeddings)}

        now = iso(datetime.now(timezone.utc))
        rows = [
            {
                "id": existing.get(a.name_lower, {}).get("id") or gen_id("v7anc"),
                "name": a.name,
                "name_lower": a.name_lower,
                "anchor_type": a.anchor_type,
                "score": a.score,
                "name_embedding": emb_by_lower.get(a.name_lower),
            }
            for a in anchors
        ]
        # v7.12 drops the ref-node layer. An anchor records WHICH PG rows it came
        # from as a property and retrieval goes straight to PG with those ids. The
        # ref node never held anything PG did not already have: measured on the
        # longmem graph, pointer edges into refs (1,491) plus ref-to-ref
        # bookkeeping (407) were 55% of all 3,493 edges and carried no knowledge.
        # Dedup is an O(n) list scan rather than an O(1) MERGE — acceptable at this
        # store size, and the alternative was an edge per (anchor, memory) pair.
        id_prop = "episodic_ids" if source_kind == "episodic" else "semantic_ids"
        tail = f"""
            WITH a
            SET a.{id_prop} = CASE
                WHEN $memory_id IN coalesce(a.{id_prop}, []) THEN a.{id_prop}
                ELSE coalesce(a.{id_prop}, []) + $memory_id END
        """

        async with driver.session(database=settings.neo4j_database) as session:
            await session.run(
                """
                UNWIND $rows AS row
                MERGE (a:V7Anchor {user_id: $user_id, name_lower: row.name_lower})
                ON CREATE SET
                    a.id = row.id,
                    a.name = row.name,
                    a.anchor_type = row.anchor_type,
                    a.organization_id = $organization_id,
                    a.created_at = $now,
                    a.mention_count = 0
                SET a.updated_at = $now,
                    a.admission_score = row.score,
                    a.mention_count = coalesce(a.mention_count, 0) + 1
                WITH a, row
                CALL (a, row) {
                    WITH a, row WHERE row.name_embedding IS NOT NULL
                    CALL db.create.setNodeVectorProperty(a, 'name_embedding', row.name_embedding)
                    RETURN count(*) AS _
                }
                """ + tail,
                rows=rows,
                user_id=user_id,
                organization_id=organization_id,
                memory_ref_id=memory_ref_id,
                memory_id=memory_id,
                now=now,
            )

    async def _fetch_existing_anchors(
        self, driver, user_id: str, name_lowers: list[str]
    ) -> dict[str, dict[str, Any]]:
        if not name_lowers:
            return {}
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                """
                UNWIND $names AS nl
                MATCH (a:V7Anchor {user_id: $user_id, name_lower: nl})
                RETURN a.id AS id, a.name_lower AS name_lower
                """,
                names=name_lowers,
                user_id=user_id,
            )
            return {rec["name_lower"]: dict(rec) async for rec in result}

    async def _upsert_frames(
        self, driver, *, frames: list, memory_id: str, source_kind: SourceKind,
        role: Optional[str], timestamp: Optional[str], mentioned_at: Optional[str],
        user_id: str,
    ) -> None:
        """v7.12: store each frame as ONE V7Fact hyperedge with N role-typed arms.

        The difference from _upsert_facts is the arity. v7.10 gave every fact exactly
        two arms (SUBJECT/OBJECT) because the extractor only produced triples; here a
        flight with an origin, a destination, a carrier and a companion is one node
        with four V7_FACT_ARG edges, each carrying its role. That keeps the
        co-participants of an event reachable from each other in one hop through the
        fact, which a triple store can only approximate by joining several facts.

        Literals (dates, prices, counts) go on the fact as parallel key/value arrays
        rather than becoming anchors. Neo4j has no map property and dynamic property
        keys need APOC, which is not installed; two aligned lists are queryable enough
        (``f.lit_vals[indexOf(f.lit_keys,'date')]``) and keep the entity space clean.
        """
        rows = []
        seen: set[str] = set()
        for fr in frames:
            arms = []
            arm_seen: set[tuple[str, str]] = set()
            for arg_role, name in getattr(fr, "args", []):
                key = anchor_canonical_key(self._clean_name(name))
                if not key or (arg_role, key) in arm_seen:
                    continue
                arm_seen.add((arg_role, key))
                arms.append({"role": arg_role, "k": key})
            # Distinct ANCHORS, not distinct arms: "Boston as origin and destination"
            # asserts nothing, same as v7.10's sk == ok tautology guard.
            if len({a["k"] for a in arms}) < 2:
                continue
            pred = canon_predicate(fr.predicate)[:80]
            identity_parts = [f"{a['role']}:{a['k']}" for a in arms]
            lits = getattr(fr, "literals", {}) or {}
            fid = frame_identity_for_source(
                user_id,
                pred,
                identity_parts,
                literals=lits,
                source_kind=source_kind,
                memory_id=memory_id,
            )
            if fid in seen:
                continue
            seen.add(fid)
            rows.append({
                "fid": fid, "pred": pred, "arms": arms, "arity": len(arms),
                "lit_keys": list(lits.keys()), "lit_vals": [lits[k] for k in lits],
            })
        if not rows:
            return
        rows.sort(key=lambda r: r["fid"])  # stable lock order, see _upsert_facts
        now = iso(datetime.now(timezone.utc))
        role_norm = (role or "shared").lower()
        # Resolve the arms FIRST and only then decide whether the frame is worth
        # writing. An arg only becomes an arm if it survived anchor selection, and a
        # frame reduced to one live anchor asserts nothing — the n-ary equivalent of
        # v7.10's subject==object tautology. Writing it anyway is what put 31% arity-1
        # facts in the first probe: the per-arm MATCH succeeds or fails independently,
        # so partial frames landed instead of being dropped the way a triple with a
        # missing endpoint was.
        cypher = """
                UNWIND $rows AS row
                UNWIND row.arms AS arm
                OPTIONAL MATCH (a:V7Anchor {user_id: $user_id, name_lower: arm.k})
                WITH row,
                     collect(CASE WHEN a IS NULL THEN NULL
                                  ELSE {role: arm.role, id: a.id} END) AS raw_arms,
                     collect(DISTINCT coalesce(a.id, '')) AS raw_ids
                WITH row, [x IN raw_arms WHERE x IS NOT NULL] AS arms,
                          [x IN raw_ids  WHERE x <> ''] AS ids
                WHERE size(ids) >= 2
                MERGE (f:V7Fact {id: row.fid})
                  ON CREATE SET f.user_id = $user_id, f.predicate = row.pred,
                                f.arity = size(ids), f.role = $role, f.timestamp = $ts,
                                f.mentioned_at = $mentioned_at,
                                f.created_at = $now, f.lit_keys = row.lit_keys,
                                f.lit_vals = row.lit_vals
                SET f.memory_ids = CASE
                        WHEN $mid IN coalesce(f.memory_ids, []) THEN f.memory_ids
                        ELSE coalesce(f.memory_ids, []) + $mid END,
                    f.memory_roles = CASE
                        WHEN $mid IN coalesce(f.memory_ids, []) THEN f.memory_roles
                        ELSE coalesce(f.memory_roles, []) + $role END
                WITH f, arms
                UNWIND arms AS arm
                MATCH (a:V7Anchor {id: arm.id})
                MERGE (f)-[:V7_FACT_ARG {role: arm.role}]->(a)
                """
        for attempt in range(5):
            try:
                async with driver.session(database=settings.neo4j_database) as session:
                    await session.run(
                        cypher, mid=memory_id, user_id=user_id, rows=rows,
                        role=role_norm, ts=timestamp, mentioned_at=mentioned_at, now=now)
                return
            except Exception as exc:  # noqa: BLE001
                transient = "Deadlock" in type(exc).__name__ or "Deadlock" in str(exc) \
                    or "TransientError" in type(exc).__name__ or "TransientError" in str(exc)
                if not transient or attempt == 4:
                    raise
                await asyncio.sleep(0.1 * (2 ** attempt))

    # ---------------------------------------------------------------- utils

    @staticmethod
    def _to_iso(value: object) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    @staticmethod
    def _source_key(source_meta: Optional[dict[str, Any]]) -> Optional[str]:
        if not source_meta:
            return None
        for key in ("chunk_id", "turn_id", "serial"):
            if source_meta.get(key) is not None:
                return f"{key}:{source_meta[key]}"
        if source_meta.get("occurred_at"):
            return f"occurred_at:{source_meta['occurred_at']}"
        return None

    # ------------------------------------------------------------- v8 finalize

    async def maintain_graph(
        self, user_id: str, *, valid_memory_ids: Optional[set[str]] = None
    ) -> dict[str, Any]:
        """Periodic graph maintenance — the redundancy that CANNOT be prevented at
        write time. Wired into the auto_dream cycle because that is where the memory
        store gets mutated.

        NB: auto_dream is **not** self-scheduling — nothing in the codebase invokes it
        on a timer; it is a REST endpoint (`/memory/auto_dream`) somebody has to call.
        So this maintenance runs exactly as often as auto_dream is triggered, which may
        be never. Call it directly if you need it on a real schedule.

        Ingest-time guards (``fact_identity`` / ``canon_predicate`` / the tautology
        skip in ``_upsert_facts``) keep NEW facts clean, but two things are only
        knowable corpus-globally, after the fact:

        * **orphaned memory refs** — auto_dream consolidates/deletes PG memories, so
          graph refs to them dangle. Pass ``valid_memory_ids`` to sweep them.
        * **dead-weight anchors** — whether an anchor ever earns a fact or a second
          memory depends on the whole corpus, unknowable when it was created.

        The tautology/duplicate passes are defensive: those are prevented at ingest
        now, but graphs built before that still carry them.

        Idempotent and cheap (a handful of scans) relative to the LLM step
        auto_dream already performs. Only ever removes graph nodes that can no
        longer contribute — never PG rows.
        """
        if not settings.enable_graph_memory:
            return {"skipped": "graph_disabled"}

        from mirix.database.neo4j_client import get_neo4j_driver

        driver = get_neo4j_driver()
        if driver is None:
            return {"skipped": "no_driver"}

        stats: dict[str, Any] = {}
        async with driver.session(database=settings.neo4j_database) as session:
            # 1. pointers to memories that no longer exist (auto_dream deletions).
            #    v7.12 keeps them in node properties, so the sweep filters the arrays
            #    rather than deleting ref nodes.
            res = await session.run(
                """
                MATCH (a:V7Anchor {user_id: $uid})
                WHERE any(x IN coalesce(a.episodic_ids, []) + coalesce(a.semantic_ids, [])
                          WHERE NOT x IN $valid)
                SET a.episodic_ids = [x IN coalesce(a.episodic_ids, []) WHERE x IN $valid],
                    a.semantic_ids = [x IN coalesce(a.semantic_ids, []) WHERE x IN $valid]
                RETURN count(*) AS n
                """,
                uid=user_id, valid=list(valid_memory_ids),
            )
            rec = await res.single()
            stats["anchors_repointed"] = int(rec["n"]) if rec else 0
            res = await session.run(
                """
                MATCH (f:V7Fact {user_id: $uid})
                WHERE any(x IN coalesce(f.memory_ids, []) WHERE NOT x IN $valid)
                WITH f, [i IN range(0, size(f.memory_ids) - 1)
                         WHERE f.memory_ids[i] IN $valid] AS keep
                SET f.memory_roles = [i IN keep | coalesce(f.memory_roles, [])[i]],
                    f.memory_ids = [i IN keep | f.memory_ids[i]]
                RETURN count(*) AS n
                """,
                uid=user_id, valid=list(valid_memory_ids),
            )
            rec = await res.single()
            stats["facts_repointed"] = int(rec["n"]) if rec else 0

            # 2. tautological facts (subject == object) — asserts nothing
            res = await session.run(
                """
                MATCH (x)<-[:V7_FACT_SUBJECT]-(f:V7Fact {user_id: $uid})-[:V7_FACT_OBJECT]->(y)
                WHERE toLower(x.name) = toLower(y.name)
                DETACH DELETE f
                RETURN count(*) AS n
                """,
                uid=user_id,
            )
            rec = await res.single()
            stats["tautologies_removed"] = int(rec["n"]) if rec else 0

            # 2b. degenerate n-ary frames: a v7.12 frame is only vacuous when ALL its
            #     arms land on ONE anchor. Unlike a triple, repeating an anchor in a
            #     wide frame is legitimate ("met X at Y about X"), so the binary
            #     subject==object test above would delete real facts here.
            res = await session.run(
                """
                MATCH (f:V7Fact {user_id: $uid})-[:V7_FACT_ARG]->(a:V7Anchor)
                WITH f, count(DISTINCT a) AS distinct_args
                WHERE distinct_args < 2
                DETACH DELETE f
                RETURN count(*) AS n
                """,
                uid=user_id,
            )
            rec = await res.single()
            stats["degenerate_frames_removed"] = int(rec["n"]) if rec else 0

            # 3. duplicate triples -> one fact, every source kept as a citation edge
            res = await session.run(
                """
                MATCH (su)<-[:V7_FACT_SUBJECT]-(f:V7Fact {user_id: $uid})-[:V7_FACT_OBJECT]->(o)
                WITH toLower(su.name) + '|' + coalesce(f.predicate, '') + '|' + toLower(o.name) AS k,
                     collect(f) AS fs
                WHERE size(fs) > 1
                WITH head(fs) AS keep, tail(fs) AS dupes
                UNWIND dupes AS dupe
                OPTIONAL MATCH (dupe)-[:V7_FACT_FROM]->(m:V7MemoryRef)
                FOREACH (_ IN CASE WHEN m IS NULL THEN [] ELSE [1] END |
                         MERGE (keep)-[:V7_FACT_FROM]->(m))
                WITH DISTINCT dupe
                DETACH DELETE dupe
                RETURN count(*) AS n
                """,
                uid=user_id,
            )
            rec = await res.single()
            stats["duplicate_facts_merged"] = int(rec["n"]) if rec else 0

            # 3c. duplicate n-ary frames. frame_identity dedups at write time, but an
            #     anchor merge can make two previously-distinct frames identical after
            #     the fact ("flew to NYC" + "flew to New York City", once NYC and New
            #     York City are one anchor). Signature is the role:anchor arm set sorted
            #     before collect (plain Cypher — APOC is not installed), so arg order
            #     cannot split a duplicate pair.
            res = await session.run(
                """
                MATCH (f:V7Fact {user_id: $uid})-[r:V7_FACT_ARG]->(a:V7Anchor)
                WITH f, r.role + ':' + toLower(a.name) AS arm
                ORDER BY arm
                WITH f, collect(arm) AS arms
                WITH coalesce(f.predicate, '') + '#'
                     + reduce(acc = '', x IN arms | acc + '|' + x) AS k, collect(f) AS fs
                WHERE size(fs) > 1
                WITH head(fs) AS keep, tail(fs) AS dupes
                UNWIND dupes AS dupe
                // Citations on a v7.12 frame live in f.memory_ids, NOT on a
                // V7_FACT_FROM edge — v7.12 never writes that edge at all. Migrating
                // over it here matched nothing and the DETACH DELETE below then took
                // the duplicate's citations with it: every source memory that only
                // ever asserted the losing node became unreachable, silently, the
                // first time anyone ran auto_dream against a v7.12+ store.
                WITH keep, dupe,
                     [i IN range(0, size(coalesce(dupe.memory_ids, [])) - 1)
                      WHERE NOT dupe.memory_ids[i] IN coalesce(keep.memory_ids, [])] AS add
                SET keep.memory_ids = coalesce(keep.memory_ids, [])
                                      + [i IN add | dupe.memory_ids[i]],
                    keep.memory_roles = coalesce(keep.memory_roles, [])
                                      + [i IN add | coalesce(dupe.memory_roles, [])[i]]
                WITH DISTINCT dupe
                DETACH DELETE dupe
                RETURN count(*) AS n
                """,
                uid=user_id,
            )
            rec = await res.single()
            stats["duplicate_frames_merged"] = int(rec["n"]) if rec else 0

            # 3b. zombie facts: a fact whose every V7_FACT_FROM citation died (its
            #    source memories were consolidated away) is unverifiable residue — in
            #    v7.10 a fact's existence is justified by its citations. Without this
            #    sweep the graph GROWS through consolidation instead of shrinking:
            #    merged memories re-extract new facts while the old ones linger
            #    (measured on LoCoMo conv-26: 1583 facts post-dream vs 1249 no-dream,
            #    +27% for a 2% smaller store). Runs before the dead-anchor pass so
            #    anchors orphaned by this deletion are pruned in the same cycle.
            res = await session.run(
                """
                MATCH (f:V7Fact {user_id: $uid})
                WHERE CASE WHEN $ref_free THEN size(coalesce(f.memory_ids, [])) = 0
                           ELSE NOT (f)-[:V7_FACT_FROM]->() END
                DETACH DELETE f
                RETURN count(*) AS n
                """, ref_free=True,
                uid=user_id,
            )
            rec = await res.single()
            stats["zombie_facts_removed"] = int(rec["n"]) if rec else 0

            # 4. dead-weight anchors: no fact touches them AND they link <=1 memory,
            #    so they can neither answer nor bridge
            res = await session.run(
                """
                MATCH (a:V7Anchor {user_id: $uid})
                WHERE NOT (a)<-[:V7_FACT_SUBJECT|V7_FACT_OBJECT|V7_FACT_ARG]-(:V7Fact)
                WITH a, CASE WHEN $ref_free
                        THEN size(coalesce(a.episodic_ids, [])) + size(coalesce(a.semantic_ids, []))
                        ELSE size([(a)-[:V7_APPEARS_IN|V7_DESCRIBED_BY]->(:V7MemoryRef) | 1])
                        END AS deg
                WHERE deg <= 1
                DETACH DELETE a
                RETURN count(*) AS n
                """,
                uid=user_id, ref_free=True,
            )
            rec = await res.single()
            stats["dead_anchors_pruned"] = int(rec["n"]) if rec else 0

            # 5. role-noise anchors: dialogue roles that slipped past older extractor
            #    filters ("Users" once reached degree 1016 with 504 role-noise facts —
            #    10.5% of all facts). The _select_anchors gate stops NEW ones, but on a
            #    graph built before that fix the existing anchor keeps accreting edges
            #    (relation/fact endpoints MATCH it by canonical name). Purge it and the
            #    facts that cite it as subject/object — those facts are role noise by
            #    construction ("Users interested in X" carries no entity identity).
            res = await session.run(
                """
                MATCH (a:V7Anchor {user_id: $uid})
                WHERE a.name_lower IN $noise
                OPTIONAL MATCH (f:V7Fact)-[:V7_FACT_SUBJECT|V7_FACT_OBJECT]->(a)
                // NB: V7_FACT_ARG is deliberately NOT here. A wide frame that happens
                // to name a dialogue role in one arm is still a real assertion about
                // its other args; the arm dies with the DETACH DELETE below and pass
                // 2b removes the frame only if it drops under 2 distinct anchors.
                WITH a, collect(DISTINCT f) AS facts
                FOREACH (f IN facts | DETACH DELETE f)
                DETACH DELETE a
                RETURN count(a) AS anchors, sum(size(facts)) AS facts
                """,
                uid=user_id, noise=sorted(_ROLE_NOISE_KEYS),
            )
            rec = await res.single()
            stats["role_noise_anchors_purged"] = int(rec["anchors"] or 0) if rec else 0
            stats["role_noise_facts_purged"] = int(rec["facts"] or 0) if rec else 0

        logger.info("graph maintenance for user=%s: %s", user_id, stats)
        return stats
