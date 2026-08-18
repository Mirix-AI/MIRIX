import asyncio
import json
import os
import re
from typing import Any, Dict, Optional, List

from openai import OpenAI
from dotenv import load_dotenv
import uuid
from pathlib import Path
import yaml
from mirix import MirixClient
from mirix_memory_system import _resolve_api_keys
from v720_policy import (
    aggregation_instruction,
    deduplicate_results,
    is_v720,
    question_operator as v720_question_operator,
)
from v721_policy import (
    aggregation_instruction as v721_aggregation_instruction,
    is_v721,
    normalize_count_answer as v721_normalize_count_answer,
    question_operator as v721_question_operator,
)
from mirix.services.retrieval_policy_v721 import predicate_hints as v721_predicate_hints
from v722_policy import (
    aggregation_instruction as v722_aggregation_instruction,
    is_v722,
    normalize_count_answer as v722_normalize_count_answer,
    question_operator as v722_question_operator,
)
from mirix.services.retrieval_policy_v722 import plan_query as v722_plan_query
from v723_policy import (
    aggregation_instruction as v723_aggregation_instruction,
    is_v723,
    normalize_count_answer as v723_normalize_count_answer,
    question_operator as v723_question_operator,
    verification_decision as v723_verification_decision,
)
from mirix.services.retrieval_policy_v723 import plan_query as v723_plan_query
from v724_policy import (
    aggregation_instruction as v724_aggregation_instruction,
    deduplicate_results as v724_deduplicate_results,
    is_v724,
    normalize_count_answer as v724_normalize_count_answer,
    question_operator as v724_question_operator,
    verification_decision as v724_verification_decision,
)
from mirix.services.retrieval_policy_v724 import plan_query as v724_plan_query

load_dotenv(".env")


_SCRATCH_RE = re.compile(r"<scratch>.*?</scratch>", re.S | re.I)


def _strip_scratch(answer: str) -> str:
    """Drop the working-out block before anything scores the answer.

    Under MIRIX_ANSWER_STYLE=scratch the model enumerates its matching instances inside
    <scratch>...</scratch> and then states the answer. If that block survived into the
    returned answer the judge would be reading the working rather than the conclusion,
    and the arm would measure formatting rather than reasoning. Also tolerates an
    unclosed tag, which is what a truncated generation leaves behind.
    """
    if not answer or "<scratch" not in answer.lower():
        return answer
    out = _SCRATCH_RE.sub("", answer)
    if "<scratch" not in out.lower():
        return out.strip()
    # Unclosed tag: the generation was cut off before it stated a conclusion, so there
    # is no answer to recover cleanly. Fall back to the last non-empty line, which is
    # where the answer sits when the model got that far, and leaves a wrong-looking
    # answer when it did not — which is the honest outcome for a truncated generation.
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


_FINAL_RE = re.compile(r"final answer\s*:\s*(.*)", re.I | re.S)


def _strip_reasoning(answer: str) -> str:
    """Keep only what follows FINAL ANSWER:, under MIRIX_ANSWER_STYLE=cot.

    HyperMem (ACL 2026, 92.73 on LoCoMo) generates with chain-of-thought against the same
    judge and the same gpt-4.1-mini answerer we use, while our prompt says "Be VERY
    CONCISE, only output the answer and nothing else". Testing that difference honestly
    requires grading the CONCLUSION, not the reasoning: this judge was measured tonight to
    reward length — of 26 regressions from a terser answerer, roughly a third were cases
    where a longer answer carrying the same core content was marked correct and the short
    one was not. Handing it the whole chain of thought would manufacture a gain out of
    verbosity.

    The full text stays in the returned `messages`, so the same generation can be re-judged
    unstripped and the two numbers subtracted. That difference IS the judge's length bias,
    measured rather than argued about.
    """
    if not answer:
        return answer
    m = _FINAL_RE.search(answer)
    if m and m.group(1).strip():
        return m.group(1).strip()
    # No marker: the model reasoned and stopped, or ignored the format. The last non-empty
    # line is where a conclusion sits when there is one.
    lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
    return lines[-1] if lines else answer




def _token_budget(model: str, visible: int) -> int:
    """Completion budget, widened for reasoning models.

    ``max_completion_tokens`` caps reasoning tokens as well as visible ones on the
    gpt-5 / o-series, so a limit sized for a short answer is spent on hidden reasoning
    and the call fails with "Could not finish the message because max_tokens or model
    output limit was reached". The visible answer these prompts want is still short;
    only the headroom changes, and only for models that need it.
    """
    # The visible budget is sized for "output the answer and nothing else". A style that
    # asks for reasoning BEFORE the answer spends the same budget on the reasoning: the
    # HyperMem-ported six-step prompt truncated at 128 tokens and the FINAL ANSWER line
    # never arrived, leaving a half sentence ("- Melanie read a book recommended") that
    # would have scored as a wrong answer for the whole run. The style is what changes the
    # length here, not the model, so it has to enter the budget.
    style = os.environ.get("MIRIX_ANSWER_STYLE", "")
    if style.startswith("cot") or style == "scratch":
        visible = max(visible, 2000)
    m = (model or "").lower()
    if not m.startswith(("gpt-5", "o1", "o3", "o4")):
        return visible
    # Scaling the VISIBLE budget was the wrong model of the cost. Measured on gpt-5-mini,
    # mean completion is 1412 tokens of which 1291 are reasoning, so visible*16 = 2048 sat
    # at 1.45x the MEAN and the tail went straight through it: a full LoCoMo run died at
    # question 601 on "Could not finish the message because max_tokens ... was reached".
    # Reasoning length is a property of the question, not of how long the answer should be.
    # An absolute floor is close to free — billing is on tokens produced, not on the cap.
    return max(visible * 16, 16384)


def _sampling_kwargs(model: str) -> dict:
    """temperature=0 / seed=42 where the model accepts them, nothing where it does not.

    The gpt-5 family rejects any temperature but the default: "Unsupported value:
    'temperature' does not support 0 with this model." Hardcoding the pair made those
    models unusable as answerers. Note the consequence for any comparison that includes
    them — they run at the provider's default temperature, so a gpt-5 arm carries more
    decoding variance than a gpt-4.1 arm, and a difference between the two is not purely
    model capability.

    MIRIX_ANSWER_SAMPLING=default forces ANY model onto that same unpinned decoding.
    That is what makes the confound measurable rather than merely disclosed: running
    gpt-4.1-mini with it gives a control arm differing from the gpt-5 arm in the model
    alone, so the gap between the two arms separates capability from decoding variance.
    """
    if os.environ.get("MIRIX_ANSWER_SAMPLING", "").lower() == "default":
        return {}
    m = (model or "").lower()
    if m.startswith(("gpt-5", "o1", "o3", "o4")):
        return {}
    return {"temperature": 0, "seed": 42}


class TaskAgent:
    def __init__(
        self,
        mirix_config_path: str,
        client_id: Optional[str] = None,
        org_id: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        user_id: Optional[str] = None,
        max_tool_rounds: int = 5,
    ):

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for TaskAgent.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.user_id = user_id
        self.max_tool_rounds = max_tool_rounds
        self._coldfacts = None  # lazy-loaded cold-fact index (MIRIX_COLDFACT gate)
        # Rows actually delivered per search, so a budget experiment can verify its
        # intervention landed rather than assume it (see MIRIX_SEARCH_CAP below).
        self._last_row_count = 0
        self._row_counts: List[int] = []
        self._v720_enabled = is_v720(os.environ.get("MIRIX_GRAPH_VERSION"))
        self._v721_enabled = is_v721(os.environ.get("MIRIX_GRAPH_VERSION"))
        self._v722_enabled = is_v722(os.environ.get("MIRIX_GRAPH_VERSION"))
        self._v723_enabled = is_v723(os.environ.get("MIRIX_GRAPH_VERSION"))
        self._v724_enabled = is_v724(os.environ.get("MIRIX_GRAPH_VERSION"))
        self._v722_force_raw_query = False
        self._v721_relation_query = False
        self._v721_question = ""
        self._v720_seen_evidence: set[str] = set()
        self._v720_operator = "single"
        self._v720_ledger: list[dict[str, Any]] = []
        self._v724_occurrence_ledger: list[tuple[str, frozenset[str]]] = []
        # timeout: per-request budget. Question-answer calls are short,
        # but this client is shared with the ingest path (see
        # MirixMemorySystem) where a single /memory/add_sync on a 4096-
        # token chunk takes 3-6 min server-side. Graph hooks can add
        # extraction + Neo4j writes, so keep 30 min headroom.
        self.mirix_client = MirixClient(client_id=client_id, org_id=org_id, base_url="http://127.0.0.1:8531", write_scope="read_write", timeout=1800)
        self.user_id = user_id if user_id is not None else str(uuid.uuid4())
        config_path = Path(mirix_config_path)
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        config = _resolve_api_keys(config)
        asyncio.run(self.mirix_client.initialize_meta_agent(
            config=config
        ))

    def _build_tools(self) -> list:
        if not self.mirix_client:
            return []
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": (
                        "Search Mirix memories for information related to a user query. "
                        "For best results, try multiple search strategies: "
                        "(1) Different phrasings of the query, "
                        "(2) Searching both 'episodic' and 'semantic' memory types separately, "
                        "(3) Using specific keywords from the question."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query string.",
                            },
                            "memory_type": {
                                "type": "string",
                                "enum": [
                                    "episodic",
                                    "resource",
                                    "procedural",
                                    "knowledge",
                                    "semantic",
                                    "all",
                                ],
                                "default": "all",
                            },
                            "search_field": {
                                "type": "string",
                                "default": "null",
                                "description": "Field to search. Use 'null' for defaults.",
                            },
                            "search_method": {
                                "type": "string",
                                "enum": ["bm25", "embedding"],
                                "default": None,
                                "description": "If not provided, the search method will be determined by the meta agent."
                            },
                            "limit": {
                                "type": "integer",
                                "default": 10,
                                "minimum": 1,
                            },
                            "filter_tags": {
                                "type": "object",
                                "description": "Optional tags to filter results.",
                            },
                            "similarity_threshold": {
                                "type": "number",
                                "description": "Optional threshold for embedding search (0.0-2.0).",
                            },
                            "start_date": {
                                "type": "string",
                                "description": "ISO 8601 start date for episodic filtering.",
                            },
                            "end_date": {
                                "type": "string",
                                "description": "ISO 8601 end date for episodic filtering.",
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
            ,
            {
                "type": "function",
                "function": {
                    "name": "check_raw_item",
                    "description": (
                        "Fetch the raw input payload for a memory item using raw_input_id."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "raw_input_id": {
                                "type": "string",
                                "description": "The raw_input_id returned by search_memory.",
                            }
                        },
                        "required": ["raw_input_id"],
                    },
                },
            },
        ]
        # (rejected answerer experiments — v7.11 consolidate tool, persona store,
        #  temporal prompt, graph-search merge — are archived; see
        #  docs/graph_memory_v7/development_history.md)
        return tools

    def _search_memory(
        self, user_id: Optional[str], params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not self.mirix_client:
            return {"success": False, "error": "Mirix client not configured."}
        resolved_user_id = user_id or self.user_id
        if not resolved_user_id:
            return {"success": False, "error": "user_id is required for memory search."}

        if not params or not isinstance(params, dict):
            return {"success": False, "error": "Missing search parameters.", "skipped": True}

        # Drop keys the LLM occasionally hallucinates with falsy/blank names —
        # `**params` would TypeError on '' or None keys and kill the whole eval.
        params = {k: v for k, v in params.items() if isinstance(k, str) and k}

        # Set reasonable defaults for better search quality
        if 'search_method' not in params or params['search_method'] is None:
            params['search_method'] = "embedding"

        # Evidence budget. The floor was raised to 15 "for better coverage", but
        # coverage is not the binding constraint: session-level recall@15 sits at a
        # saturated 0.940 while the supporting row's POSITION separates right answers
        # from wrong ones (rank 1 in 69.7% of correct vs 30.1% of wrong). And 15 per
        # kind returns ~30 rows, with the buried-support failures — abstentions,
        # miscounts, discursive non-answers — carrying median support rank 6 to 16.
        # MIRIX_SEARCH_LIMIT makes the budget sweepable so "fewer, better rows" can be
        # tested against "more rows" instead of assumed either way.
        _floor = int(os.environ.get("MIRIX_SEARCH_LIMIT", "15") or 15)
        if 'limit' not in params or params['limit'] is None or params['limit'] < _floor:
            params['limit'] = _floor

        # The answer model often rewrites "What instruments does Priya play?"
        # into "Priya instruments", silently deleting the predicate that makes
        # v7.21's fact lane selective. Preserve the raw question as the graph query;
        # the model-generated wording remains useful only when no relation plan was
        # recognized. This is query planning, not a flat-index bypass.
        if (
            (
                self._v721_enabled or self._v722_enabled
                or self._v723_enabled or self._v724_enabled
            )
            and self._v721_relation_query
            and self._v721_question
            and (
                not (
                    getattr(self, "_v722_enabled", False)
                    or getattr(self, "_v723_enabled", False)
                    or getattr(self, "_v724_enabled", False)
                )
                or getattr(self, "_v722_force_raw_query", False)
            )
        ):
            params["query"] = self._v721_question

        try:
            results = asyncio.run(self.mirix_client.search(user_id=resolved_user_id, **params))
        except TypeError as e:
            # MirixClient.search rejected an unknown kwarg (LLM produced
            # a key not in the schema). Skip this search rather than
            # crashing the entire eval — model will see empty results
            # and try another query.
            return {"success": False, "error": f"Invalid search args: {e}", "skipped": True}

        if results['success']:
            for result in results['results']:
                # Format timestamp with description if available
                if 'occurred_at' in result:
                    timestamp = result['occurred_at']
                    if 'occurred_at_description' in result and result['occurred_at_description']:
                        result['occurred_at'] = f"{timestamp} ({result['occurred_at_description']})"
                        del result['occurred_at_description']  # Remove redundant field

                if "id" in result:
                    if self._v720_enabled:
                        result["_evidence_id"] = result["id"]
                    del result["id"]
                if "actor" in result:
                    del result["actor"]
            out = results['results']
            # Hybrid retrieval (MIRIX_HYBRID_SEARCH): the default is embedding-only, which
            # misses a memory whose embedding is DILUTED even though the literal term is in
            # its text — the exact failure consolidation creates (merging 5 topics into one
            # broad memory blurs its vector, so "Glass Menagerie" no longer ranks, though
            # the words are right there). A BM25 full-text pass over the same query recovers
            # those by exact term. Union + dedup against the embedding hits.
            if os.environ.get("MIRIX_HYBRID_SEARCH"):
                bm = dict(params)
                bm["search_method"] = "bm25"
                try:
                    bres = asyncio.run(self.mirix_client.search(user_id=resolved_user_id, **bm))
                except Exception:  # noqa: BLE001
                    bres = None
                if bres and bres.get("success"):
                    seen = {(r.get("summary") or "")[:120] for r in out if isinstance(r, dict)}
                    for r in bres["results"]:
                        s = (r.get("summary") or "")[:120]
                        if not s or s in seen:
                            continue
                        seen.add(s)
                        if "occurred_at" in r and r.get("occurred_at_description"):
                            r["occurred_at"] = f"{r['occurred_at']} ({r['occurred_at_description']})"
                        for k in ("id", "actor", "occurred_at_description"):
                            if k == "id" and self._v720_enabled and r.get(k):
                                r["_evidence_id"] = r[k]
                            r.pop(k, None)
                        out.append(r)
            # Cold-fact merge (MIRIX_COLDFACT): surface verbatim specifics the summarizing
            # ingest dropped, AS REGULAR retrieved evidence competing with summaries for THIS
            # search query — not force-injected ground truth. Lets normal retrieval filtering
            # decide relevance (avoids the over-trust collateral of prompt injection).
            for f in self._retrieve_coldfacts(resolved_user_id, params.get("query", ""),
                                              k=3, thresh=0.82):
                out.append({"memory_type": "semantic", "summary": f,
                            "source": "recovered detail from original conversation"})
            if self._v720_enabled:
                if getattr(self, "_v724_enabled", False):
                    deduped = v724_deduplicate_results(
                        out,
                        self._v720_seen_evidence,
                        getattr(self, "_v724_occurrence_ledger", []),
                    )
                    (
                        out,
                        self._v720_seen_evidence,
                        self._v724_occurrence_ledger,
                    ) = deduped
                else:
                    out, self._v720_seen_evidence = deduplicate_results(
                        out, self._v720_seen_evidence
                    )
                for item in out:
                    item.pop("_evidence_id", None)
                self._v720_ledger.extend(out)

            # A REAL ceiling on what the answerer sees. MIRIX_SEARCH_LIMIT above is a
            # FLOOR (`params['limit'] < _floor`), so it only ever raises the request: a
            # model that asks for limit=10 keeps 10 even when the floor is set to 5, and
            # nothing there constrains the rows hybrid search, cold facts and graph
            # expansion append AFTER retrieval. That is why "cut the budget 31 -> 11" came
            # back with no change and must not be read as evidence that volume is
            # harmless — the cut only partly landed, and was never verified.
            #
            # This truncates the final list, so the number here is the number delivered.
            # _last_row_count records it so an experiment can PROVE its arm differed
            # instead of assuming it.
            _cap = int(os.environ.get("MIRIX_SEARCH_CAP", "0") or 0)
            if _cap > 0 and len(out) > _cap:
                out = out[:_cap]
            self._last_row_count = len(out)
            self._row_counts.append(len(out))
            return out

        return results


    def _check_raw_item(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.mirix_client:
            return {"success": False, "error": "Mirix client not configured."}
        raw_input_id = params.get("raw_input_id")
        if not raw_input_id:
            return {"success": False, "error": "raw_input_id is required."}
        fn = getattr(self.mirix_client, "check_raw_item", None)
        if fn is None:
            return {"success": False, "error": "Raw item lookup is unavailable; answer from the memories already retrieved."}
        try:
            import inspect
            res = fn(raw_input_id)
            return asyncio.run(res) if inspect.isawaitable(res) else res
        except Exception as e:
            return {"success": False, "error": f"Raw item lookup failed: {e}"}


    def _serialize_tool_calls(self, tool_calls: Any) -> list:
        serialized = []
        for call in tool_calls:
            if hasattr(call, "model_dump"):
                serialized.append(call.model_dump())
            else:
                serialized.append(call)
        return serialized

    def _v723_verify_answer(
        self,
        *,
        question: str,
        answer: str,
        messages: list[dict[str, Any]],
        user_id: Optional[str],
        usage_entries: list[dict[str, Any]],
        usage_total: dict[str, int],
    ) -> str:
        return self._bounded_verify_answer(
            question=question,
            answer=answer,
            messages=messages,
            user_id=user_id,
            usage_entries=usage_entries,
            usage_total=usage_total,
            decision_fn=v723_verification_decision,
            model_env="MIRIX_V723_VERIFIER_MODEL",
        )

    def _v724_verify_answer(
        self,
        *,
        question: str,
        answer: str,
        messages: list[dict[str, Any]],
        user_id: Optional[str],
        usage_entries: list[dict[str, Any]],
        usage_total: dict[str, int],
    ) -> str:
        return self._bounded_verify_answer(
            question=question,
            answer=answer,
            messages=messages,
            user_id=user_id,
            usage_entries=usage_entries,
            usage_total=usage_total,
            decision_fn=v724_verification_decision,
            model_env="MIRIX_V724_VERIFIER_MODEL",
        )

    def _bounded_verify_answer(
        self,
        *,
        question: str,
        answer: str,
        messages: list[dict[str, Any]],
        user_id: Optional[str],
        usage_entries: list[dict[str, Any]],
        usage_total: dict[str, int],
        decision_fn,
        model_env: str,
    ) -> str:
        """Run at most one evidence-led correction for an observable answer defect."""

        decision = decision_fn(
            question,
            answer,
            self._v720_ledger if self._v720_ledger else None,
        )
        if not decision.retry:
            return answer

        retry_evidence = self._search_memory(
            user_id,
            {
                "query": decision.query,
                "memory_type": "all",
                "search_method": "embedding",
                "limit": 12,
            },
        )
        evidence_payload = retry_evidence if isinstance(retry_evidence, list) else []
        verify_messages = [
            *messages,
            {
                "role": "system",
                "content": (
                    "V7.23 evidence verification found these mechanical issues in the "
                    f"draft answer: {', '.join(decision.reasons)}. Review the cited "
                    "graph evidence below and return one corrected, minimal answer. "
                    "Do not introduce a name, date, number, or item unsupported by the "
                    "evidence. If the evidence does not justify a change, repeat the draft.\n"
                    + json.dumps(evidence_payload, ensure_ascii=False)
                ),
            },
        ]
        model = os.environ.get(
            model_env,
            os.environ.get("MIRIX_V723_VERIFIER_MODEL", self.model),
        )
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=verify_messages,
                max_completion_tokens=_token_budget(model, 160),
                # keyed on the model actually being called, which an env override can
                # make different from self.model
                **_sampling_kwargs(model),
            )
        except Exception:  # noqa: BLE001
            return answer

        usage = getattr(response, "usage", None)
        if usage:
            entry = {
                "model": model,
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
            usage_entries.append(entry)
            usage_total["prompt_tokens"] += entry["prompt_tokens"]
            usage_total["completion_tokens"] += entry["completion_tokens"]
            usage_total["total_tokens"] += entry["total_tokens"]
        revised = (response.choices[0].message.content or "").strip()
        return revised or answer


    def _load_coldfacts(self, user_id: Optional[str]):
        """Load the cold-fact index (MIRIX_COLDFACT gate): verbatim numbers/names the
        summarizing ingest dropped, kept as a SEPARATE retrieval index (LongMemEval
        key-expansion / Dense-X style). Cached per instance as (facts, np.ndarray)."""
        if not os.environ.get("MIRIX_COLDFACT"):
            return None
        if self._coldfacts is None:
            uid = user_id or self.user_id or "unknown"
            path = os.path.expanduser(f"~/MIRIX_eval/coldfacts_{uid}.json")
            try:
                import numpy as np
                idx = json.load(open(path, encoding="utf-8"))
                mat = np.array([x["emb"] for x in idx], dtype="float32")
                mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
                self._coldfacts = ([x["fact"] for x in idx], mat)
            except (OSError, ValueError, KeyError):
                self._coldfacts = ([], None)
        return self._coldfacts

    def _retrieve_coldfacts(self, user_id, q_text, k=3, thresh=0.83):
        """Top-k cold facts for the question by ada-002 cosine (only strongly-relevant
        ones, to avoid perturbing questions with no matching literal)."""
        cf = self._load_coldfacts(user_id)
        if not cf or cf[1] is None or not q_text.strip():
            return []
        facts, mat = cf
        import numpy as np
        qe = self.client.embeddings.create(model="text-embedding-ada-002",
                                           input=q_text[:400]).data[0].embedding
        q = np.array(qe, dtype="float32"); q /= (np.linalg.norm(q) + 1e-8)
        sims = mat @ q
        order = np.argsort(-sims)[:k]
        return [facts[i] for i in order if sims[i] >= thresh]

    _DIGEST_PROMPT = (
        "Normalise the retrieved memories below into an evidence table for ONE question. "
        "You are not answering it — you are making the evidence usable.\n\n"
        "QUESTION: {q}\n\nRETRIEVED MEMORIES:\n{rows}\n\n"
        "Rules:\n"
        "1. One line per DISTINCT real-world event or fact.\n"
        "2. Resolve every relative time expression (\"yesterday\", \"last week\", \"a few "
        "months ago\") to an absolute date, using the timestamp attached to the memory "
        "that contains it. Write the absolute date first. If it cannot be resolved, "
        "write UNRESOLVED rather than guessing.\n"
        "3. MERGE rows that describe the SAME event even when the wording differs, and "
        "mark them (mentioned in N memories). Do NOT merge distinct events that merely "
        "look alike — different dates mean different events.\n"
        "4. If two memories disagree on a detail, keep both on one line marked "
        "CONFLICT, and never silently pick one.\n"
        "5. Copy specifics verbatim — names, titles, numbers, places. Add nothing that "
        "is not in the rows above.\n\n"
        "Output only the table."
    )

    # v2. The largest identified block of remaining failures is not a dirty table: in 18
    # of 41 the gold IS in the normalised table and the model answers with something else
    # ("What book did Caroline recommend?" -> gold "Becoming Nicole" is present, answer
    # was about gathering documents). A general normalisation is not question-directed —
    # it produces ~1450 characters and leaves the selection to the answering turn, which
    # is the step that was already failing.
    #
    # So v2 ends with the candidates themselves. VERBATIM is load-bearing: a free-form
    # candidate list is a second rewriting pass, and this codebase already has evidence
    # of what that costs — 71% of quoted work-titles in the store appear nowhere in the
    # source, because ingest completes vague references into invented canon.
    _DIGEST2_TAIL = (
        "\n\nThen, after the table, add:\n\n"
        "CANDIDATE ANSWERS\n"
        "- Every span in the rows above that could directly answer the question, copied "
        "VERBATIM. Never paraphrase, never complete a partial name or title, never add "
        "a candidate that is not written in the rows.\n"
        "- One per line as: <verbatim span> | <absolute date or UNRESOLVED> | supported "
        "by N memories\n"
        "- Order them by how directly they answer the question.\n"
        "- If nothing in the rows can answer it, write exactly: NONE"
    )

    def _evidence_digest(self, question, messages, usage_entries, usage_total):
        """One normalisation pass over the accumulated tool output.

        Deliberately a separate call rather than another instruction in the system
        prompt: the answering prompt already carries a normalisation instruction that is
        obeyed a few percent of the time, and the countfirst result showed the model does
        follow a format rule when that rule is the only thing it is being asked to do.
        """
        rows = [str(m.get("content") or "") for m in messages if m.get("role") == "tool"]
        blob = "\n".join(rows)[:24000]
        if not blob.strip():
            return ""
        prompt = self._DIGEST_PROMPT.format(q=question, rows=blob)
        if os.environ.get("MIRIX_EVIDENCE_DIGEST") == "2":
            prompt += self._DIGEST2_TAIL
        try:
            r = self.client.chat.completions.create(
                model=os.environ.get("MIRIX_DIGEST_MODEL", self.model),
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=_token_budget(self.model, 900),
                **_sampling_kwargs(self.model),
            )
        except Exception as e:  # noqa: BLE001
            print(f"[digest] failed, answering without it: {e}", flush=True)
            return ""
        u = getattr(r, "usage", None)
        if u:
            usage_entries.append({"model": "digest",
                                  "prompt_tokens": getattr(u, "prompt_tokens", 0),
                                  "completion_tokens": getattr(u, "completion_tokens", 0),
                                  "total_tokens": getattr(u, "total_tokens", 0)})
            usage_total["prompt_tokens"] += getattr(u, "prompt_tokens", 0)
            usage_total["completion_tokens"] += getattr(u, "completion_tokens", 0)
            usage_total["total_tokens"] += getattr(u, "total_tokens", 0)
        text = (r.choices[0].message.content or "").strip()
        if not text:
            return ""
        return ("NORMALISED EVIDENCE (dates resolved, duplicate events merged, "
                "conflicts marked). Answer from this:\n" + text)

    def answer(self, input_messages: List[Dict[str, Any]], user_id: Optional[str] = None) -> Dict[str, Any]:
        tools = self._build_tools()
        question = ""
        for input_message in reversed(input_messages):
            if input_message.get("role") == "user":
                question = str(input_message.get("content") or "")
                break
        self._v720_seen_evidence = set()
        self._v720_ledger = []
        self._v724_occurrence_ledger = []
        if self._v724_enabled:
            v724_plan = v724_plan_query(question)
            self._v721_question = question
            self._v720_operator = v724_question_operator(question)
            self._v721_relation_query = v724_plan.is_relation_query
            self._v722_force_raw_query = v724_plan.preserve_raw_query
        elif self._v723_enabled:
            v723_plan = v723_plan_query(question)
            self._v721_question = question
            self._v720_operator = v723_question_operator(question)
            self._v721_relation_query = v723_plan.is_relation_query
            self._v722_force_raw_query = v723_plan.preserve_raw_query
        elif self._v722_enabled:
            v722_plan = v722_plan_query(question)
            self._v721_question = question
            self._v720_operator = v722_question_operator(question)
            self._v721_relation_query = v722_plan.is_relation_query
            self._v722_force_raw_query = v722_plan.preserve_raw_query
        elif self._v721_enabled:
            self._v722_force_raw_query = False
            self._v721_question = question
            self._v720_operator = v721_question_operator(question)
            self._v721_relation_query = bool(v721_predicate_hints(question))
        elif self._v720_enabled:
            self._v722_force_raw_query = False
            self._v721_question = ""
            self._v720_operator = v720_question_operator(question)
            self._v721_relation_query = False
        else:
            self._v722_force_raw_query = False
            self._v721_question = ""
            self._v720_operator = "single"
            self._v721_relation_query = False
        system_prompt = (
            "You are the Chat Agent, a component of the personal assistant system. "
            "Your primary responsibility is managing user communication. "
            "You have access to a unified memory infrastructure shared with other specialized agents. "
            "\n\nMemory Components:\n"
            "1. Core Memory: Essential user information and your persona.\n"
            "2. Episodic Memory: Chronological records of interactions.\n"
            "3. Procedural Memory: Step-by-step processes and guidelines.\n"
            "4. Resource Memory: Documents and reference materials.\n"
            "5. Knowledge: Factual data like contacts and credentials.\n"
            "6. Semantic Memory: Conceptual knowledge and contextual information.\n"
            "\n\nOperational Requirements:\n"
            "Whenever a user sends a query, an initial high-level (preliminary) search is automatically conducted, and the results are provided to you. "
            "However, this initial search may not be comprehensive or fully accurate. "
            "You MUST evaluate the provided information and utilize the `search_memory` tool to conduct additional, more specific searches if you believe further context is necessary to provide a complete and accurate response. "
            "\n\nSearch Strategy (CRITICAL):\n"
            "1. VERIFY RESULTS: After each search, check if results contain key terms from the question. If not, the search likely returned wrong memories.\n"
            "2. MULTI-ANGLE SEARCH: Try different search phrasings if initial results seem off-topic.\n"
            "   - Example: 'book Priya read Tomas suggestion' + 'The Salt Road Priya' + 'book recommendation Tomas Priya'\n"
            "3. CROSS-MEMORY SEARCH: For most questions, search BOTH episodic AND semantic memory types separately and combine results.\n"
            "   - Episodic contains events/activities (when things happened)\n"
            "   - Semantic contains stable facts/attributes (interests, possessions, skills)\n"
            "4. LIST AGGREGATION: For questions asking 'What items...', 'What activities...', search multiple times with different keywords and aggregate ALL results.\n"
            "   - Example: 'What has X painted?' → Search 'X painted', 'X painting', 'X artwork', then combine all unique items found\n"
            "5. SMART STOPPING: After 2-3 searches, evaluate if you have enough information to answer. If yes, STOP SEARCHING and provide your answer.\n"
            "   - Don't keep searching indefinitely if you already found relevant information\n"
            "   - You have a maximum of 5 search rounds - use them wisely\n"
            "6. KEYWORD VARIANTS: If searching for a specific item (book, painting, activity), try searching for:\n"
            "   - The item name directly ('The Salt Road')\n"
            "   - The person + activity ('Priya read book')\n"
            "   - The relationship context ('Tomas suggested book Priya')\n"
            "\n"
            "Be persistent but efficient: if you find relevant information after 2-3 searches, provide your answer. "
            "Do NOT give up or state that you don't know the answer unless multiple searches with different parameters have failed to yield relevant information. "
            "You may call the tool multiple times if needed. "
            "Each memory item may include a `raw_input_id` that points to the raw user input. "
            "Use the `check_raw_item` tool when you need the original input for disambiguation or exact wording. "
            "\n\nMessage Processing Protocol:\n"
            "1. Analyze the user's query and use `search_memory` to gather necessary context.\n"
            "   - If a result includes `raw_input_id` and you need the original text, call `check_raw_item`.\n"
            "2. Provide a helpful and concise answer based on the retrieved information.\n"
            "3. Only inform the user that you don't know the answer if at least three consecutive searches with different parameters have failed to yield relevant information.\n"
            + (
                # MIRIX_ANSWER_STYLE=scratch resolves a conflict inside this prompt.
                # "Be VERY CONCISE, only output the answer and nothing else" is obeyed —
                # answers average 83-150 characters — while the enumeration instruction
                # 40 lines below ("write out EVERY distinct matching instance as an
                # explicit numbered list") is obeyed 7% of the time on counting
                # questions, 0% on list questions, and in ZERO of the 169 wrong answers.
                # The two cannot both hold: one forbids showing work, the other requires
                # it. Separating the work from the final answer lets both stand.
                "4. Do your working inside <scratch>...</scratch> first: list every "
                "distinct matching instance, one per line, before committing. Then, "
                "AFTER the closing tag, output ONLY the final answer and nothing else. "
                "Everything inside <scratch> is discarded.\n"
                if os.environ.get("MIRIX_ANSWER_STYLE") == "scratch"
                else
                # MIRIX_ANSWER_STYLE=cot. The published system that leads this benchmark
                # (HyperMem, 92.73) uses the SAME judge and the SAME gpt-4.1-mini answerer
                # and differs here: it generates with chain-of-thought where this prompt
                # forbids reasoning outright. Our largest category gap is multi-hop
                # (82.98 vs 93.62) — the category that averages 2.68 sessions and 3.13
                # evidence turns per question, i.e. the one that cannot be answered
                # without weighing several memories against each other.
                #
                # This also matches the largest residual failure mode found by hand:
                # of the bucket-C errors that survive evidence normalisation, the biggest
                # block is "both the right and the wrong candidate are really in the
                # store and it picked the wrong one". An answerer forbidden to deliberate
                # has no step in which that choice could be made.
                "4. Think it through first: state which retrieved memories bear on the "
                "question, and where two candidates compete, say why one wins. Then, on a "
                "new final line, write exactly `FINAL ANSWER: <answer>`. Only that line "
                "is graded, and it must be as concise as the question allows — everything "
                "before it is discarded.\n"
                if os.environ.get("MIRIX_ANSWER_STYLE") == "cot"
                else
                # MIRIX_ANSWER_STYLE=cot_hypermem. A port of HyperMem's
                # ANSWER_PROMPT_NEMORI_COT (EverMind-AI/HyperMem,
                # hypermem/prompts/answer_prompts.py), the answerer behind the 92.73 that
                # this system's 88.44 is being compared against. Ported rather than
                # copied: theirs is a single-shot prompt over a fixed context block, ours
                # is a tool-calling loop, so the RESPONSE FORMAT, the inference mandate
                # and the detail-preservation requirements carry over and the context
                # plumbing does not.
                #
                # Using their prompt is the point. With the judge prompt now known to be
                # byte-identical and the answerer model the same gpt-4.1-mini, the answer
                # prompt was the last uncontrolled difference between the two systems.
                # Holding it fixed is what turns "we score 4.3 points lower" into a
                # statement about the MEMORY rather than about prompt engineering.
                "4. Follow this structure, then stop:\n"
                "   ## STEP 1: RELEVANT MEMORIES — list each retrieved memory bearing on "
                "the question, with its timestamp.\n"
                "   ## STEP 2: KEY DETAILS — every person/place/company name, every "
                "number, amount and date, every frequency, every proper noun.\n"
                "   ## STEP 3: CROSS-MEMORY LINKING — entities appearing in more than one "
                "memory, and what follows from combining them. Move beyond fact "
                "extraction and perform logical inference: where the evidence strongly "
                "suggests a connection, state it. Do NOT dismiss a reasonable inference "
                "as speculation.\n"
                "   ## STEP 4: TIME RESOLUTION — convert every relative reference "
                "(\"last year\", \"two months ago\") to an absolute date using the "
                "timestamp of the memory that contains it.\n"
                "   ## STEP 5: CONTRADICTIONS — if memories disagree, say which wins and "
                "why (prefer the most recent).\n"
                "   ## STEP 6: CHECK — names included? numbers exact? frequencies "
                "specific (\"every Tuesday and Thursday\", not \"twice a week\")? dates "
                "precise?\n"
                "   ## FINAL ANSWER: <the concise answer, keeping every specific detail>\n"
                "   Only the FINAL ANSWER line is graded; everything above it is "
                "discarded.\n"
                if os.environ.get("MIRIX_ANSWER_STYLE") == "cot_hypermem"
                else "4. Be VERY CONCISE in your response, only output the answer and nothing else.\n"
            )
            + "5. There are some open-ended questions where you may not find explicit evidences, you still need to answer it based on your understanding. Never say you don't know or 'there is no specific information', ...\n"
            "6. If there is no information found, you still need to answer it. Guess an answer if you don't have enough information.\n"
            "\n\nCOUNTING / AGGREGATION QUESTIONS (how many, how much, total, number of):\n"
            "- These fail when you estimate a number in your head. DO NOT estimate.\n"
            "- First search exhaustively with multiple keyword variants so NO instance is missed.\n"
            "- Then write out EVERY distinct matching instance as an explicit numbered list (1., 2., 3., ...).\n"
            "- Your final answer's number = the count of items in that list (or their sum for amounts). Count the list, never guess.\n"
            + (
                # MIRIX_ANSWER_STYLE=countfirst. The instruction above already says the
                # answer is the count, and gpt-5-mini still returns the LIST as its answer:
                # in 4 of its 9 "how many" errors it enumerated exactly the gold number of
                # instances and never stated the total. It did the hard part — finding
                # every scattered instance — and failed the trivial one. So the missing
                # instruction is not "count them", it is that the list is WORKING and the
                # answer is its length.
                "- OUTPUT FORMAT: the answer to a counting question is THE NUMBER, alone, "
                "as the first thing you write (e.g. \"2\"). The numbered list is your "
                "working, not your answer — never return the list as the answer.\n"
                if os.environ.get("MIRIX_ANSWER_STYLE") == "countfirst" else ""
            )
            + "- Include instances that are phrased differently or belong to the same category even if not obviously a match (e.g. a yoga session counts as a 'fitness class').\n"
            +
            "\n\nAnswer Format Guidelines (CRITICAL):\n"
            "- For list questions (What books, What instruments, What activities, etc.), provide a simple comma-separated list or use 'and' between items.\n"
            "  Example: \"trombone and cello\" NOT \"She plays clarinet\"\n"
            "  Example: '\"Nothing is Impossible\", \"Charlotte\\'s Web\"' NOT \"She read several books\"\n"
            "- For simple fact questions (What is X's relationship status?, How old?, etc.), provide direct factual answers.\n"
            "  Example: \"Single\" NOT \"She experienced a breakup but is...\"\n"
            "  Example: \"28 years old\" NOT \"She is currently 28 years old and...\"\n"
            "- For specific detail questions (What kind of art?, What type of pot?, etc.), provide the specific detail.\n"
            "  Example: \"abstract art\" NOT \"art inspired by...\"\n"
            "  Example: \"a tankard with a fox on the lid\" NOT \"pottery items\"\n"
            "- ALWAYS extract the minimal, direct answer that matches what's being asked. Do NOT add ANY additional information!\n"
            "- If the question asks for multiple items, search until you find ALL items, not just the first one."
        )
        if self._v720_enabled:
            system_prompt += (
                "\n\nV7.20 EVIDENCE POLICY:\n"
                "- The current question operator is " + self._v720_operator + ".\n"
                "- Prefer concrete assertions with exact entities/predicates/event times over broad profile summaries.\n"
                "- Treat evidence_id as a memory-row identity, not automatically as a distinct real-world event.\n"
                "- For COUNT, deduplicate repeated descriptions by event_time + event + participants; semantic memories corroborate but do not create another occurrence.\n"
                "- A general frequency or habitual statement (for example, 'every other Thursday') describes a pattern; it is not an additional dated occurrence and must not be added to the count.\n"
                "- For LIST_UNION, union concrete items across searches and emit each canonical item once.\n"
                "- For count/list questions, perform targeted searches before answering; do not rely only on the broad initial context."
            )
        if (self._v721_enabled or self._v724_enabled) and self._v721_relation_query:
            system_prompt += (
                "\n\nV7.21 RELATION-AWARE POLICY:\n"
                "- Graph facts are high-precision evidence written as predicate(role=participant); use their roles, not just topic similarity.\n"
                "- Keep searches faithful to the raw question. Do not insert a guessed answer into a search query before evidence supports it.\n"
                "- For INTERSECTION, require the same canonical subject/object to be supported for every named participant.\n"
                "- For multi-hop questions, compose cited relations by their shared item; an unrelated direct fact must not replace the requested chain.\n"
                "- A 'Graph resolved relation (unique typed chain)' already performed that cited composition. For a question with a relation qualifier (for example, from someone's recommendation), answer its resolved item; never substitute a direct fact that lacks the qualifier.\n"
                "- If graph facts and broad profile summaries disagree, prefer the concrete fact with matching predicate, roles, and event time."
            )
        if (
            self._v722_enabled or self._v723_enabled or self._v724_enabled
        ) and self._v721_relation_query:
            system_prompt += (
                "\n\nV7.22+ GENERAL RELATION POLICY:\n"
                "- Treat graph facts as candidates, not automatically as truth; prefer facts whose relation, named entities, object/literal constraints, and event time all match the question.\n"
                "- Preserve ordinary graph-traversal evidence alongside relation-cited evidence. A broad relation match must not override a more concrete cited memory.\n"
                "- For first/latest/current/before/after questions, distinguish occurrences by event time before selecting an answer.\n"
                "- For COUNT and LIST_UNION, aggregate distinct supported occurrences or items; do not turn repeated citations into additional events.\n"
                "- If a relation fact lacks the requested object or temporal qualifier, use it only as context and keep searching through graph citations."
            )
        if self._v723_enabled or self._v724_enabled:
            system_prompt += (
                "\n\nV7.23 SOURCE-GROUNDED ANSWER POLICY:\n"
                "- Retrieval remains graph-owned. Use exact episodic citations for dates, counts, labels, titles, colors, and other source-specific details.\n"
                "- Keep separately timed occurrences distinct. Semantic summaries corroborate an occurrence but do not create an additional event.\n"
                "- For multi-hop answers, compose only relations sharing a cited participant or item; never bridge on topic similarity alone.\n"
                "- Before answering, verify every output name, date, number, and list item against cited evidence.\n"
                "- If evidence is insufficient, issue a narrower graph search rather than filling the gap with a plausible guess."
            )
        if self._v724_enabled:
            system_prompt += (
                "\n\nV7.24 ROLE, EVENT, AND CITATION POLICY:\n"
                "- Match every named participant to the argument role expressed by the question; agent and recipient/source direction are not interchangeable.\n"
                "- Prefer event_time over mentioned_at. For as-of questions, exclude facts that became true only after the cutoff.\n"
                "- Planned or intended actions are not completed occurrences unless the question explicitly asks about plans.\n"
                "- Relation-cited episodic rows are exact refinements of graph facts; use their concrete wording before broad semantic profiles.\n"
                "- Fill the grammatical answer slot directly: when asked what someone does, return the cited actions or objects, not a broad restatement of the surrounding activity.\n"
                "- For counts, collapse paraphrased citations of one event but never merge separately timed events."
            )
        messages = [
            {"role": "system", "content": system_prompt},
            *input_messages
        ]

        usage_entries = []
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        digested = False
        for round_num in range(self.max_tool_rounds + 1):
            # On the last round, force answer generation by not providing tools
            is_last_round = (round_num == self.max_tool_rounds)

            if is_last_round:
                # Add instruction to provide final answer
                messages.append({
                    "role": "system",
                    "content": "You have reached the maximum number of searches. Please provide your best answer based on the information you've gathered so far."
                })

            tool_choice = None
            if not is_last_round and tools:
                if self._v720_enabled and round_num == 0 and self._v720_operator in (
                    "count", "list_union", "intersection"
                ):
                    tool_choice = "required"
                else:
                    tool_choice = "auto"
            # One question must not be able to kill a six-hour run. The budget below is
            # generous, but a reasoning model can still exhaust it, and main_eval does not
            # catch it: the previous attempt lost 939 unanswered questions to a single 400
            # on question 601. Retry once with room to spare, then give up on THIS
            # question and let the eval continue — a blank answer scores zero, which is
            # honest, whereas a dead run scores nothing at all.
            _budget = _token_budget(self.model, 128)
            for _attempt in (0, 1):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=None if is_last_round else (tools or None),
                        tool_choice=tool_choice,
                        max_completion_tokens=_budget,
                        **_sampling_kwargs(self.model),
                    )
                    break
                except Exception as _e:  # noqa: BLE001
                    if "max_tokens" not in str(_e) and "output limit" not in str(_e):
                        raise
                    if _attempt == 0:
                        _budget *= 4
                        print(f"[token budget] retrying at {_budget}", flush=True)
                        continue
                    print(f"[token budget] giving up on this question: {_e}", flush=True)
                    return {"answer": "", "error": "token_budget_exhausted",
                            "messages": messages}

            usage = getattr(response, "usage", None)
            if usage:
                entry = {
                    "model": self.model,
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                }
                usage_entries.append(entry)
                usage_total["prompt_tokens"] += entry["prompt_tokens"]
                usage_total["completion_tokens"] += entry["completion_tokens"]
                usage_total["total_tokens"] += entry["total_tokens"]

            message = response.choices[0].message
            if not message.tool_calls:
                # MIRIX_EVIDENCE_DIGEST: normalise the retrieved rows ONCE, then let the
                # model answer again with the normalised view in front of it.
                #
                # Aimed at bucket C, where by construction a single returned row already
                # covers the gold and the answer is still wrong. Perfect filtering of
                # those same rows recovers only ~25%, ranking them perfectly only +4, and
                # cutting them is strictly worse — so the rows are neither missing nor
                # buried nor too many. What they are is UNNORMALISED: relative dates
                # unresolved against each row's own timestamp, and the same event
                # restated across several rows. Those are exactly the two largest
                # mechanisms in the source-adjudicated partition of this bucket
                # (10 time-anchor, 10 multi-episode aggregation of ~60 real errors).
                if (
                    os.environ.get("MIRIX_EVIDENCE_DIGEST")
                    and not digested
                    and any(m.get("role") == "tool" for m in messages)
                ):
                    digested = True
                    digest = self._evidence_digest(
                        question, messages, usage_entries, usage_total)
                    if digest:
                        messages.append({"role": "system", "content": digest})
                        continue
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                    }
                )
                final_answer = message.content or ""
                if (
                    (
                        self._v721_enabled or self._v722_enabled
                        or self._v723_enabled or self._v724_enabled
                    )
                    and self._v720_operator == "count"
                ):
                    final_answer = (
                        v724_normalize_count_answer(final_answer)
                        if self._v724_enabled
                        else (
                            v723_normalize_count_answer(final_answer)
                            if self._v723_enabled
                            else (
                                v722_normalize_count_answer(final_answer)
                                if self._v722_enabled
                                else v721_normalize_count_answer(final_answer)
                            )
                        )
                    )
                    messages[-1]["content"] = final_answer
                if self._v724_enabled:
                    final_answer = self._v724_verify_answer(
                        question=question,
                        answer=final_answer,
                        messages=messages,
                        user_id=user_id,
                        usage_entries=usage_entries,
                        usage_total=usage_total,
                    )
                    if self._v720_operator == "count":
                        final_answer = v724_normalize_count_answer(final_answer)
                    messages[-1]["content"] = final_answer
                elif self._v723_enabled:
                    final_answer = self._v723_verify_answer(
                        question=question,
                        answer=final_answer,
                        messages=messages,
                        user_id=user_id,
                        usage_entries=usage_entries,
                        usage_total=usage_total,
                    )
                    if self._v720_operator == "count":
                        final_answer = v723_normalize_count_answer(final_answer)
                    messages[-1]["content"] = final_answer
                final_answer = _strip_scratch(final_answer)
                # Preserved BEFORE stripping and returned separately: re-judging the same
                # generation unstripped is what measures the judge's length bias, and
                # messages[-1] is about to be overwritten with the stripped text.
                raw_answer = final_answer
                if os.environ.get("MIRIX_ANSWER_STYLE", "").startswith("cot"):
                    final_answer = _strip_reasoning(final_answer)
                messages[-1]["content"] = final_answer
                return {
                    "answer": final_answer,
                    "raw_answer": raw_answer,
                    "messages": messages,
                    "usage": usage_entries,
                    "usage_total": usage_total,
                }

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": self._serialize_tool_calls(message.tool_calls),
                }
            )

            for tool_call in message.tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    tool_result = {"success": False, "error": "Invalid tool arguments."}
                else:
                    if tool_call.function.name == "search_memory":
                        tool_result = self._search_memory(user_id, args)
                        if self._v720_enabled:
                            tool_result = {
                                "operator": self._v720_operator,
                                "instruction": (
                                    v724_aggregation_instruction(self._v720_operator)
                                    if self._v724_enabled
                                    else (
                                        v723_aggregation_instruction(self._v720_operator)
                                        if self._v723_enabled
                                        else (
                                            v722_aggregation_instruction(self._v720_operator)
                                            if self._v722_enabled
                                            else (
                                                v721_aggregation_instruction(self._v720_operator)
                                                if self._v721_enabled
                                                else aggregation_instruction(self._v720_operator)
                                            )
                                        )
                                    )
                                ),
                                "new_evidence": tool_result,
                                "ledger_unique_rows": len(self._v720_ledger),
                            }
                    elif tool_call.function.name == "check_raw_item":
                        tool_result = self._check_raw_item(args)
                    else:
                        tool_result = {"success": False, "error": "Unknown tool."}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result),
                    }
                )

        return {
            "answer": "I don't know",
            "messages": messages,
            "usage": usage_entries,
            "usage_total": usage_total,
        }
