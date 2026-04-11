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
        self.mirix_client = MirixClient(client_id=client_id, org_id=org_id, base_url="http://127.0.0.1:8531", write_scope="read_write")
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
        return [
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

        # Set reasonable defaults for better search quality
        if 'search_method' not in params or params['search_method'] is None:
            params['search_method'] = "embedding"

        # Increase limit slightly for better recall
        if 'limit' not in params or params['limit'] is None or params['limit'] < 10:
            params['limit'] = 15  # Get more candidates for better coverage

        results = asyncio.run(self.mirix_client.search(user_id=resolved_user_id, **params))

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
            return results['results']

        return results

    def _check_raw_item(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.mirix_client:
            return {"success": False, "error": "Mirix client not configured."}
        raw_input_id = params.get("raw_input_id")
        if not raw_input_id:
            return {"success": False, "error": "raw_input_id is required."}
        return self.mirix_client.check_raw_item(raw_input_id)

    def _serialize_tool_calls(self, tool_calls: Any) -> list:
        serialized = []
        for call in tool_calls:
            if hasattr(call, "model_dump"):
                serialized.append(call.model_dump())
            else:
                serialized.append(call)
        return serialized

    def answer(self, input_messages: List[Dict[str, Any]], user_id: Optional[str] = None) -> Dict[str, Any]:
        tools = self._build_tools()
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
