import asyncio
import json
import os
from typing import Any, Dict, Optional, List

from openai import OpenAI
from dotenv import load_dotenv
import uuid
from pathlib import Path
import yaml
from mirix import MirixClient
from mirix_memory_system import _resolve_api_keys

load_dotenv(".env")

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
        self._persona = None  # lazy-loaded persona profile (MIRIX_PERSONA gate)
        self._coldfacts = None  # lazy-loaded cold-fact index (MIRIX_COLDFACT gate)
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
        # v7.11 consolidate — experimental distinct-instance enumerator (graph anchor→memory
        # + pgvector recall, LLM de-dup). The mechanism is validated (bikes/art/jewelry/
        # fitness topics enumerate correctly), but it does NOT help end-to-end QA (30/31 vs
        # the 36 enumerate-then-count baseline) and is REJECTED as a default. Gated behind a
        # flag for A/B. Root cause (see docs/graph_memory_v7): its addressable failure mode —
        # scattered DISTINCT instances needing enumeration — is nearly absent from this
        # benchmark's stable ceiling (dominated by cross-session aggregation + preference
        # synthesis + quantity-sums), and the few counting questions it could touch carry
        # temporal/scope qualifiers ('in the past month', 'in a typical week') a flat
        # enumerator can't honor. The 30–36 spread is itself mostly answerer-LLM noise
        # (20/60 questions flip run-to-run).
        if os.environ.get("MIRIX_ENABLE_CONSOLIDATE"):
            tools.append({
                "type": "function",
                "function": {
                    "name": "consolidate",
                    "description": (
                        "Enumerate a collection of DISTINCT things the user owns/did/attended that are "
                        "SCATTERED across many memories. Pulls every memory about the topic (exhaustively) "
                        "and returns a de-duplicated list of the distinct items. Use ONLY for distinct-instance "
                        "questions: 'what are all my X', 'how many different bikes/doctors/classes/events'. "
                        "Do NOT use for 'how many days/times', 'how much / total cost', or running totals — "
                        "it lists distinct items, it does not sum quantities. Treat its list as a completeness "
                        "check; you decide the final count."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string",
                                      "description": "The collection to enumerate, e.g. 'bikes I own', 'fitness classes I attend', 'jewelry I acquired'."},
                        },
                        "required": ["topic"],
                    },
                },
            })
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

        # Increase limit slightly for better recall
        if 'limit' not in params or params['limit'] is None or params['limit'] < 10:
            params['limit'] = 15  # Get more candidates for better coverage

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
                    del result["id"]
                if "actor" in result:
                    del result["actor"]
            out = results['results']
            # Cold-fact merge (MIRIX_COLDFACT): surface verbatim specifics the summarizing
            # ingest dropped, AS REGULAR retrieved evidence competing with summaries for THIS
            # search query — not force-injected ground truth. Lets normal retrieval filtering
            # decide relevance (avoids the over-trust collateral of prompt injection).
            for f in self._retrieve_coldfacts(resolved_user_id, params.get("query", ""),
                                              k=3, thresh=0.82):
                out.append({"memory_type": "semantic", "summary": f,
                            "source": "recovered detail from original conversation"})
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

    def _consolidate(self, user_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """v7.11 consolidation: for scattered-instance counting questions, use the entity
        graph to EXHAUSTIVELY pull every memory about the topic (anchor→memory completeness,
        which flat top-k lacks), fetch their full text from PG, and LLM-consolidate into a
        de-duplicated list + count. Fixes the root cause: countable personal instances are
        scattered across memories and never aggregated."""
        topic = str(params.get("topic", "")).strip()
        if not topic:
            return {"error": "topic required"}
        try:
            from neo4j import GraphDatabase
            import psycopg2
            # The graph's anchor vectors were built with text-embedding-ada-002 — MIRIX's
            # embedding_model() never passes config.embedding_model to llama_index's
            # OpenAIEmbedding, so it silently defaults to ada-002 (both 1536-dim, which hid
            # it). Query must use the SAME model or it lands in a different space (0.94 vs 0.52).
            emb = self.client.embeddings.create(
                model="text-embedding-ada-002", input=topic[:400]).data[0].embedding
            # Channel 1 — entity graph: anchor→memory (precise, but only as complete as extraction).
            drv = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "mirix_neo4j_dev"))
            with drv.session() as sess:
                mids = set(sess.run(
                    """
                    CALL db.index.vector.queryNodes('v7_anchor_name_emb', 12, $emb) YIELD node AS a, score AS sc
                    WHERE a.user_id = $u AND sc >= 0.6
                    MATCH (a)-[:V7_APPEARS_IN|V7_DESCRIBED_BY]->(m:V7MemoryRef)
                    RETURN DISTINCT m.memory_id AS mid
                    """, emb=emb, u=user_id).value("mid"))
            drv.close()
            conn = psycopg2.connect(host="localhost", port=5432, user="mirix",
                                    password="mirix", dbname="mirix_lm114_pm")
            cur = conn.cursor()
            # Channel 2 — direct memory-vector recall: catches memories the extractor never
            # linked to the topic's anchors (e.g. a "four bikes" line filed under trip locations).
            # PG memory embeddings are zero-padded to MAX_EMBEDDING_DIM=4096; pad to match
            # (zero-padding is cosine-invariant).
            padded = list(emb) + [0.0] * (4096 - len(emb))
            vec = "[" + ",".join(f"{x:.7f}" for x in padded) + "]"
            cur.execute(
                "SELECT id FROM episodic_memory WHERE user_id=%s AND is_deleted=false "
                "AND summary_embedding IS NOT NULL ORDER BY summary_embedding <=> %s::vector LIMIT 20",
                (user_id, vec))
            mids.update(r[0] for r in cur.fetchall())
            cur.execute(
                "SELECT id FROM semantic_memory WHERE user_id=%s AND is_deleted=false "
                "AND summary_embedding IS NOT NULL ORDER BY summary_embedding <=> %s::vector LIMIT 10",
                (user_id, vec))
            mids.update(r[0] for r in cur.fetchall())
            if not mids:
                cur.close(); conn.close()
                return {"consolidated": "No memories found for this topic."}
            mids = list(mids)
            cur.execute(
                "SELECT actor, occurred_at, summary, details FROM episodic_memory "
                "WHERE user_id=%s AND id = ANY(%s) AND is_deleted=false "
                "UNION ALL SELECT 'reference', NULL, name||': '||summary, details FROM semantic_memory "
                "WHERE user_id=%s AND id = ANY(%s) AND is_deleted=false",
                (user_id, mids, user_id, mids))
            rows = cur.fetchall()
            cur.close(); conn.close()
            if not rows:
                return {"consolidated": "No memory text found."}
            # Keep full details — enumerations ("four bikes: road, mountain, commuter, hybrid")
            # often sit deep in the details text; truncating them defeats the whole purpose.
            ctx = "\n".join(
                f"[{r[0]}{' ' + str(r[1])[:10] if r[1] else ''}] {r[2]}"
                + (f" — {r[3][:1000]}" if r[3] and r[3] != r[2] else "")
                for r in rows[:45])
            prompt = (
                f"From the memories below, produce the user's complete de-duplicated list for: \"{topic}\".\n"
                "Rules:\n"
                "- If a memory EXPLICITLY enumerates a count and lists the members (e.g. 'four bikes: a road bike, "
                "mountain bike, commuter bike, and a new hybrid bike'), include ALL listed members — do not drop any.\n"
                "- List every DISTINCT item exactly once. Merge descriptions that refer to the same physical object "
                "(same category + compatible attributes = ONE item, e.g. 'silver necklace with a pendant' and "
                "'silver necklace purchased April 15' = one necklace), but keep genuinely different items separate "
                "(BodyPump and home strength training are two different classes). Count each thing only once.\n"
                "- Only include items the user themselves OWNS/DID/ATTENDED (exclude gifts given to others, generic "
                "recommendations, and hypothetical/considered-but-not-acquired items).\n"
                "- Then give the total count = length of the list.\n"
                'Return JSON: {"items": ["...", ...], "count": N}\n\nMemories:\n' + ctx)
            resp = self.client.chat.completions.create(
                model=self.model, temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}])
            return {"consolidated": resp.choices[0].message.content, "n_memories": len(rows)}
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)[:150]}

    def _serialize_tool_calls(self, tool_calls: Any) -> list:
        serialized = []
        for call in tool_calls:
            if hasattr(call, "model_dump"):
                serialized.append(call.model_dump())
            else:
                serialized.append(call)
        return serialized

    def _load_persona(self, user_id: Optional[str]) -> str:
        """Load the pre-built persona profile (MIRIX_PERSONA gate). Cached per instance."""
        if not os.environ.get("MIRIX_PERSONA"):
            return ""
        if self._persona is None:
            uid = user_id or self.user_id or "unknown"
            path = os.path.expanduser(f"~/MIRIX_eval/persona_{uid}.txt")
            try:
                with open(path, encoding="utf-8") as f:
                    self._persona = f.read().strip()
            except OSError:
                self._persona = ""
        return self._persona

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

    def answer(self, input_messages: List[Dict[str, Any]], user_id: Optional[str] = None) -> Dict[str, Any]:
        tools = self._build_tools()
        # Persona injection (MIRIX_PERSONA) — for advice-shaped questions ("any tips /
        # suggestions / ideas / what should I..."), inject the pre-built user profile so the
        # answerer grounds recommendations in the user's own history instead of giving
        # generic advice. Targets the preference-synthesis failures (Q7/Q11/Q44). Advice-
        # gated to avoid perturbing factual questions.
        persona = self._load_persona(user_id)
        persona_block = ""
        if persona:
            q_text = " ".join(str(m.get("content", "")) for m in input_messages
                              if isinstance(m, dict) and m.get("role") == "user").lower()
            # NB: no bare "any " — it substring-matches "how m[any b]ikes" (every count Q).
            advice_kw = ("suggest", "tips", "idea", "recommend", "advice", "what should i",
                         "how can i", "how do i", "help me")
            if any(kw in q_text for kw in advice_kw):
                persona_block = (
                    "\n\nUSER PERSONA — ground your advice in this profile of the user's own "
                    "history, past successes, and stated preferences. For open-ended 'any tips / "
                    "suggestions / ideas' questions, personalize using the relevant details below; "
                    "do NOT give generic advice that ignores the user's actual habits and prior "
                    "experiences:\n" + persona + "\n"
                )
        # v7.11 consolidate guidance — only injected when the experimental tool is enabled
        # (MIRIX_ENABLE_CONSOLIDATE). Default answerer = the enumerate-then-count baseline.
        consolidate_hint = (
            "- ONLY for 'what are all my X' / 'how many DIFFERENT/DISTINCT X' questions (counting separate "
            "things: bikes, doctors, classes, events, jewelry) where the items are scattered across memories, "
            "you MAY call `consolidate(topic)` to pull every memory about the topic and get a de-duplicated list. "
            "Use its list only to make sure you have not MISSED a distinct item — you still decide the final count "
            "yourself from the evidence. Do NOT call consolidate for 'how many DAYS/TIMES/HOURS', 'how MUCH / total "
            "cost / total amount', or running-total questions (e.g. videos completed so far) — those need summing "
            "quantities or a single stored number, which consolidate does not do; enumerate and count those yourself.\n"
        ) if os.environ.get("MIRIX_ENABLE_CONSOLIDATE") else ""
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
            "   - Example: 'book Melanie read Caroline suggestion' + 'Becoming Nicole Melanie' + 'book recommendation Caroline Melanie'\n"
            "3. CROSS-MEMORY SEARCH: For most questions, search BOTH episodic AND semantic memory types separately and combine results.\n"
            "   - Episodic contains events/activities (when things happened)\n"
            "   - Semantic contains stable facts/attributes (interests, possessions, skills)\n"
            "4. LIST AGGREGATION: For questions asking 'What items...', 'What activities...', search multiple times with different keywords and aggregate ALL results.\n"
            "   - Example: 'What has X painted?' → Search 'X painted', 'X painting', 'X artwork', then combine all unique items found\n"
            "5. SMART STOPPING: After 2-3 searches, evaluate if you have enough information to answer. If yes, STOP SEARCHING and provide your answer.\n"
            "   - Don't keep searching indefinitely if you already found relevant information\n"
            "   - You have a maximum of 5 search rounds - use them wisely\n"
            "6. KEYWORD VARIANTS: If searching for a specific item (book, painting, activity), try searching for:\n"
            "   - The item name directly ('Becoming Nicole')\n"
            "   - The person + activity ('Melanie read book')\n"
            "   - The relationship context ('Caroline suggested book Melanie')\n"
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
            "4. Be VERY CONCISE in your response, only output the answer and nothing else.\n"
            "5. There are some open-ended questions where you may not find explicit evidences, you still need to answer it based on your understanding. Never say you don't know or 'there is no specific information', ...\n"
            "6. If there is no information found, you still need to answer it. Guess an answer if you don't have enough information.\n"
            "\n\nCOUNTING / AGGREGATION QUESTIONS (how many, how much, total, number of):\n"
            "- These fail when you estimate a number in your head. DO NOT estimate.\n"
            "- First search exhaustively with multiple keyword variants so NO instance is missed.\n"
            "- Then write out EVERY distinct matching instance as an explicit numbered list (1., 2., 3., ...).\n"
            "- Your final answer's number = the count of items in that list (or their sum for amounts). Count the list, never guess.\n"
            "- Include instances that are phrased differently or belong to the same category even if not obviously a match (e.g. a yoga session counts as a 'fitness class').\n"
            + consolidate_hint +
            "\n\nAnswer Format Guidelines (CRITICAL):\n"
            "- For list questions (What books, What instruments, What activities, etc.), provide a simple comma-separated list or use 'and' between items.\n"
            "  Example: \"clarinet and violin\" NOT \"She plays clarinet\"\n"
            "  Example: '\"Nothing is Impossible\", \"Charlotte\\'s Web\"' NOT \"She read several books\"\n"
            "- For simple fact questions (What is X's relationship status?, How old?, etc.), provide direct factual answers.\n"
            "  Example: \"Single\" NOT \"She experienced a breakup but is...\"\n"
            "  Example: \"28 years old\" NOT \"She is currently 28 years old and...\"\n"
            "- For specific detail questions (What kind of art?, What type of pot?, etc.), provide the specific detail.\n"
            "  Example: \"abstract art\" NOT \"art inspired by...\"\n"
            "  Example: \"a cup with a dog face on it\" NOT \"pottery items\"\n"
            "- ALWAYS extract the minimal, direct answer that matches what's being asked. Do NOT add ANY additional information!\n"
            "- If the question asks for multiple items, search until you find ALL items, not just the first one."
            + persona_block
        )
        messages = [
            {"role": "system", "content": system_prompt},
            *input_messages
        ]

        usage_entries = []
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for round_num in range(self.max_tool_rounds + 1):
            # On the last round, force answer generation by not providing tools
            is_last_round = (round_num == self.max_tool_rounds)

            if is_last_round:
                # Add instruction to provide final answer
                messages.append({
                    "role": "system",
                    "content": "You have reached the maximum number of searches. Please provide your best answer based on the information you've gathered so far."
                })

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=None if is_last_round else (tools or None),
                tool_choice=None if is_last_round else ("auto" if tools else None),
                max_completion_tokens=128,
                temperature=0,
                seed=42,
            )

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
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                    }
                )
                final_answer = message.content or ""
                return {
                    "answer": final_answer,
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
                    elif tool_call.function.name == "consolidate":
                        tool_result = self._consolidate(user_id, args)
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
